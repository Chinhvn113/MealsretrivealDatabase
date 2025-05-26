from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from ram.models import ram_plus
from ram import inference_ram as inference, get_transform
from PIL import Image
import json
import os
from tqdm import tqdm
from .transform import load_image
import glob

class MilvusManager:
    def __init__(self,
                 embedding_dim=1024,
                 host="localhost",
                 port="19530",
                 model_path="jinaai/jina-clip-v2",
                 tag_model_weights='/root/Database/recognize-anything-plus-model/ram_plus_swin_large_14m.pth',
                 image_size=384,
                 max_num=12,
                 ocr_model_path="OpenGVLab/InternVL3-38B"):
        # Connect to Milvus
        connections.connect("default", host=host, port=port)

        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.max_num = max_num

        # Model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(ocr_model_path, trust_remote_code=True)
        self.ocr_model = AutoModel.from_pretrained(
            ocr_model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            trust_remote_code=True,
            use_flash_attn=True
        ).eval().to(self.device)

        self.tag_model = ram_plus(
            pretrained=tag_model_weights,
            image_size=self.image_size,
            vit='swin_l',
        ).eval().to(self.device)

        self._init_milvus_collections()

    def _init_milvus_collections(self):
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, description="Multimodal Image/Text Search")

        if not utility.has_collection("multimodal_index"):
            self.collection = Collection(name="multimodal_index", schema=schema)
            self.collection.create_index("embedding", {
                "metric_type": "IP",
                "index_type": "FLAT",
                "params": {}
            })
        else:
            self.collection = Collection(name="multimodal_index")

    def encode_image(self, image_path):
        emb = self.model.encode_image([image_path], truncate_dim=self.embedding_dim)[0]
        return emb / np.linalg.norm(emb)

    def encode_text(self, text):
        if isinstance(text, str):
            text = [text]
        emb = self.model.encode_text(text, truncate_dim=self.embedding_dim)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb[0] if len(text) == 1 else emb

    def insert(self, embeddings, metadata_list):
        """
        Insert embeddings and metadata into Milvus
        """
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        self.collection.insert([embeddings.tolist(), metadata_list])

    def search(self, query, query_type="text", top_k=5):
        """
        Perform a similarity search
        """
        query_vector = self.encode_text(query) if query_type == "text" else self.encode_image(query)
        self.collection.load()
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
            output_fields=["metadata"]
        )
        return results[0]

    def get_tag(self, image_path):
        transform = get_transform(image_size=self.image_size)
        img_tensor = transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        result = inference(img_tensor, self.tag_model)
        return result[0].split(' | ')

    def get_ocr(self, image_path):
        pixel_values = load_image(image_path, max_num=self.max_num).to(torch.bfloat16).to(self.device)
        question = '''...OCR question prompt...'''
        config = dict(max_new_tokens=1024, do_sample=True, temperature=0.01, top_p=0.9)
        response = self.ocr_model.chat(self.tokenizer, pixel_values, question, config)
        try:
            return json.loads(response)
        except:
            return {}

    def build(self, data_root, image_batch_size=8, mode=None):
        if mode == 'AIC':
            self.__build_keyframe_AIC(data_root, image_batch_size=image_batch_size)
        elif mode == 'ACM':
            self.__build_keyframe_ACM(data_root, image_batch_size=image_batch_size)
        elif mode == 'roomelsa':
            self.__build_roomelsa(data_root, image_batch_size=image_batch_size)
        else:
            raise ValueError("Invalid mode. Choose 'AIC', 'ACM', or 'roomelsa'.")

    def __build_keyframe_AIC(self, data_root, image_batch_size=8):
        video_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        for video_dir in tqdm(video_dirs, desc="Building Milvus AIC"):
            metadata_path = os.path.join(video_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                continue

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            frame_paths = glob.glob(os.path.join(video_dir, "*.jpg"))
            for frame_path in frame_paths:
                frame_name = os.path.basename(frame_path)
                if frame_name not in metadata:
                    continue

                embed_image = self.encode_image(frame_path)
                tags = self.get_tag(frame_path)
                ocr_info = self.get_ocr(frame_path)

                embed_main_news = self.encode_text(ocr_info.get("main_news_text", ""))
                embed_thumbnail = self.encode_text(ocr_info.get("thumbnail_text", ""))

                meta = {
                    "video_name": os.path.basename(video_dir),
                    "frame_name": frame_name,
                    "frame_path": frame_path,
                    "shot": metadata[frame_name]["shot"],
                    "timestamp": metadata[frame_name]["time_stamp"],
                    "tags": tags,
                    "ocr": ocr_info
                }

                self.insert(embed_image, [meta])
                self.insert(embed_main_news, [meta])
                self.insert(embed_thumbnail, [meta])

    def __build_roomelsa(self, data_root, image_batch_size=8):
        object_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

        for obj_dir in tqdm(object_dirs, desc="Building Milvus RoomElsa"):
            obj_name = os.path.basename(obj_dir)
            image_files = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            text_files = [f for f in os.listdir(obj_dir) if f.lower().endswith(".txt")]

            if image_files:
                embeddings = self.encode_image_batch(image_files, batch_size=image_batch_size)
                for emb, path in zip(embeddings, image_files):
                    meta = {"object_name": obj_name, "image_path": path}
                    self.insert(emb, [meta])

            if text_files:
                text_path = os.path.join(obj_dir, text_files[0])
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    text_embed = self.encode_text(text)
                    meta = {"object_name": obj_name, "text_path": text_path, "text": text}
                    self.insert(text_embed, [meta])

    def __build_keyframe_ACM(self, data_root, image_batch_size=8):
        video_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        for video_dir in tqdm(video_dirs, desc="Building Milvus ACM"):
            metadata_path = os.path.join(video_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                continue

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            frame_paths = glob.glob(os.path.join(video_dir, "*.jpg"))
            speak_descriptions = [
                os.path.join(video_dir, f)
                for f in os.listdir(video_dir)
                if f.lower().endswith(".txt") and f.startswith("speak_")
            ]

            for frame_path in frame_paths:
                frame_name = os.path.basename(frame_path)
                if frame_name not in metadata:
                    continue

                embed_image = self.encode_image(frame_path)
                tags = self.get_tag(frame_path)
                ocr_info = self.get_ocr(frame_path)

                embed_main_news = self.encode_text(ocr_info.get("main_news_text", ""))
                embed_thumbnail = self.encode_text(ocr_info.get("thumbnail_text", ""))

                meta = {
                    "video_name": os.path.basename(video_dir),
                    "frame_name": frame_name,
                    "frame_path": frame_path,
                    "shot": metadata[frame_name]["shot"],
                    "timestamp": metadata[frame_name]["time_stamp"],
                    "tags": tags,
                    "ocr": ocr_info,
                    "speak_descriptions": speak_descriptions
                }

                self.insert(embed_image, [meta])
                self.insert(embed_main_news, [meta])
                self.insert(embed_thumbnail, [meta])
