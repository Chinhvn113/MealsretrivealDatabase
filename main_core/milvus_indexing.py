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
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

            FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="main_news_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="thumbnail_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),

            FieldSchema(name="image_metadata", dtype=DataType.JSON),
            FieldSchema(name="main_news_metadata", dtype=DataType.JSON),
            FieldSchema(name="thumbnail_metadata", dtype=DataType.JSON),

            FieldSchema(name="ocr_info", dtype=DataType.JSON),
            FieldSchema(name="ocr_metadata", dtype=DataType.JSON)
        ]

        schema = CollectionSchema(fields, description="Multimodal with separated OCR embeddings")

        if not utility.has_collection("multimodal_index"):
            self.collection = Collection(name="multimodal_index", schema=schema)
            self.collection.create_index("image_embedding", {"metric_type": "IP", "index_type": "FLAT", "params": {}})
            self.collection.create_index("main_news_embedding", {"metric_type": "IP", "index_type": "FLAT", "params": {}})
            self.collection.create_index("thumbnail_embedding", {"metric_type": "IP", "index_type": "FLAT", "params": {}})
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

    def insert(
        self,
        image_embedding=None,
        main_news_embedding=None,
        thumbnail_embedding=None,
        image_metadata=None,
        main_news_metadata=None,
        thumbnail_metadata=None,
        ocr_info=None,
        ocr_metadata=None
    ):
        def wrap(x):
            if x is None:
                return [None]
            if isinstance(x, dict):
                return [x]
            if isinstance(x, np.ndarray) and x.ndim == 1:
                return [x.tolist()]
            return x

        self.collection.insert([
            wrap(image_embedding),
            wrap(main_news_embedding),
            wrap(thumbnail_embedding),
            wrap(image_metadata),
            wrap(main_news_metadata),
            wrap(thumbnail_metadata),
            wrap(ocr_info),
            wrap(ocr_metadata)
        ])

        
    def search(
        self,
        query,
        mode="text2img",
        top_k=5,
        filter_expr=None,
        text_field="main_news"  # or "thumbnail"
    ):
        """
        Multimodal search with metadata filtering support.

        Args:
            query (str or path): text or image depending on mode.
            mode (str): "text2text", "text2img", or "img2img"
            top_k (int): number of top matches to return
            filter_expr (str): optional metadata filter (Milvus expr syntax)
            text_field (str): for text2text mode only, choose "main_news" or "thumbnail"
        """
        self.collection.load()

        if mode == "text2text":
            query_vector = self.encode_text(query)
            if text_field == "main_news":
                search_field = "main_news_embedding"
                output_fields = ["main_news_metadata", "ocr_info", "ocr_metadata"]
            elif text_field == "thumbnail":
                search_field = "thumbnail_embedding"
                output_fields = ["thumbnail_metadata", "ocr_info", "ocr_metadata"]
            else:
                raise ValueError("Invalid text_field: choose 'main_news' or 'thumbnail'")

        elif mode == "text2img":
            query_vector = self.encode_text(query)
            search_field = "image_embedding"
            output_fields = ["image_metadata", "ocr_info", "ocr_metadata"]

        elif mode == "img2img":
            query_vector = self.encode_image(query)
            search_field = "image_embedding"
            output_fields = ["image_metadata", "ocr_info", "ocr_metadata"]

        else:
            raise ValueError("Invalid mode: choose 'text2text', 'text2img', or 'img2img'")

        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field=search_field,
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )
        return results[0] if results else []


    def get_tag(self, image_path):
        transform = get_transform(image_size=self.image_size)
        img_tensor = transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        result = inference(img_tensor, self.tag_model)
        return result[0].split(' | ')

    def get_ocr(self, image_path):
        pixel_values = load_image(image_path, max_num=self.max_num).to(torch.bfloat16).to(self.device)
        question = '''    
    You are a structured OCR and information extraction model. Given a news broadcast image, extract key elements of the screen as structured data. Ignore any moving or scrolling text (tickers or subtitles). The banner may appear in any color and is typically used for concise summaries or headlines.

    Extract the following fields (leave blank if not present):
    **Channel Name:** The news channel name, typically visible as a logo or text in the top corners (e.g., HTV9, VTV, CNN, BBC, ANTV).

    **Main News Text:**
    - Paragraph-like or block text that conveys the main news content.
    - Typically located in the center or main visual area of the screen (upper-middle or mid-screen).
    - Includes large bold title text or speaker names and roles.
    - Should not include branding, logos, or show names.
    - If there is no informative news content, return main_news_text: null.

    **Thumbnail Text:**
    - A static overlay banner, regardless of its color.
    - Positioned above the ticker, near the bottom, or on the side.
    - Typically contains concise summaries, headlines, or topic teasers.
    - Can appear in any color (red, blue, yellow, etc.).
    - If the text is purely stylistic, logo-based, or branding-related (e.g., “60 giây” intro splash), return: "thumbnail_text": null.

    **Time:** On-screen timestamp showing current broadcast time (e.g., 06:54:33).

    **Return answer in this json format:**
    {
        "channel_name": "",
        "main_news_text": "",
        "thumbnail_text": "",
        "time": ""
    }

    Now extract the information from this image: 
    <image>
    '''
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

                ocr_info = self.get_ocr(frame_path)
                tags = self.get_tag(frame_path)
                video_name = os.path.basename(video_dir)
                ocr_meta = {
                    "video_name": video_name,
                    "key_frame": frame_name,
                    "video_path": video_dir,
                    "frame_path": frame_path
                }
                base_meta = {
                    "video_path": video_dir,
                    "video_name": video_name,
                    "frame_name": frame_name,
                    "frame_path": frame_path,
                    "shot": metadata[frame_name]["shot"],
                    "timestamp": metadata[frame_name]["time_stamp"],
                    "tags": tags
                }

                # Embeddings
                embed_image = self.encode_image(frame_path)
                embed_main_news = self.encode_text(ocr_info.get("main_news_text", ""))
                embed_thumbnail = self.encode_text(ocr_info.get("thumbnail_text", ""))

                # Final insert
                self.insert(
                    image_embedding=embed_image,
                    image_metadata={**base_meta, "embedding_type": "image"},
                    main_news_embedding=embed_main_news,
                    main_news_metadata={**base_meta, "embedding_type": "main_news"},
                    thumbnail_embedding=embed_thumbnail,
                    thumbnail_metadata={**base_meta, "embedding_type": "thumbnail"},
                    ocr_info=ocr_info,
                    ocr_metadata=ocr_meta
                )
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

                ocr_info = self.get_ocr(frame_path)
                tags = self.get_tag(frame_path)
                video_name = os.path.basename(video_dir)
                ocr_meta = {
                    "video_name": video_name,
                    "key_frame": frame_name,
                    "video_path": video_dir,
                    "frame_path": frame_path
                }
                base_meta = {
                    "video_path": video_dir,
                    "video_name": video_name,
                    "frame_name": frame_name,
                    "frame_path": frame_path,
                    "shot": metadata[frame_name]["shot"],
                    "timestamp": metadata[frame_name]["time_stamp"],
                    "tags": tags
                }

                # Embeddings
                embed_image = self.encode_image(frame_path)
                embed_main_news = self.encode_text(ocr_info.get("main_news_text", ""))
                embed_thumbnail = self.encode_text(ocr_info.get("thumbnail_text", ""))

                # Final insert
                self.insert(
                    image_embedding=embed_image,
                    image_metadata={**base_meta, "embedding_type": "image"},
                    main_news_embedding=embed_main_news,
                    main_news_metadata={**base_meta, "embedding_type": "main_news"},
                    thumbnail_embedding=embed_thumbnail,
                    thumbnail_metadata={**base_meta, "embedding_type": "thumbnail"},
                    ocr_info=ocr_info,
                    ocr_metadata=ocr_meta
                )
