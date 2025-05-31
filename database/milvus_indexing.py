from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import numpy as np
import torch
from transformers import AutoModel
import os
import json
from tqdm import tqdm
from PIL import Image
from .transform import load_image

class MilvusManager:
    def __init__(self, 
                 embedding_dim=1024, 
                 host="localhost", 
                 port="19530", 
                 model_path="jinaai/jina-clip-v2"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim

        # Connect to Milvus
        connections.connect("default", host=host, port=port)

        # Load model
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(self.device).eval()

        # Initialize metadata stores
        self.image_metadata = []
        self.text_metadata = []

        # Init Milvus schema & collection
        self._init_milvus_collections()

    def _init_milvus_collections(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        schema = CollectionSchema(fields, description="Unified multimodal index")

        if not utility.has_collection("multimodal_index"):
            self.collection = Collection(name="multimodal_index", schema=schema)
            self.collection.create_index("image_embedding", {"metric_type": "IP", "index_type": "FLAT", "params": {}})
            self.collection.create_index("text_embedding", {"metric_type": "IP", "index_type": "FLAT", "params": {}})
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

    def insert(self, image_embedding=None, text_embedding=None, metadata=None):
        def wrap(x):
            if x is None:
                return [None]
            if isinstance(x, np.ndarray) and x.ndim == 1:
                return [x.tolist()]
            return [x]

        self.collection.insert([
            wrap(image_embedding if image_embedding is not None else np.zeros(self.embedding_dim)),
            wrap(text_embedding if text_embedding is not None else np.zeros(self.embedding_dim)),
            wrap(metadata if metadata else {})
        ])

    def search(self, query, mode="text", top_k=5):
        self.collection.load()

        if mode == "text":
            query_vector = self.encode_text(query)
            anns_field = "text_embedding"
        elif mode == "image":
            query_vector = self.encode_image(query)
            anns_field = "image_embedding"
        else:
            raise ValueError("Mode must be 'text' or 'image'")

        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field=anns_field,
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
            output_fields=["metadata"]
        )

        return [{"metadata": hit.entity.get("metadata"), "score": hit.distance} for hit in results[0]] if results else []

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

            frame_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.lower().endswith('.jpg')]
            for frame_path in frame_paths:
                frame_name = os.path.basename(frame_path)
                if frame_name not in metadata:
                    continue

                meta = metadata[frame_name]
                embed_image = self.encode_image(frame_path)
                embed_text = self.encode_text(meta.get("ocr", ""))

                frame_meta = {
                    "video_name": os.path.basename(video_dir),
                    "frame_name": frame_name,
                    "frame_path": frame_path,
                    "shot": meta.get("shot"),
                    "timestamp": meta.get("time-stamp"),
                    "tags": meta.get("tags", [])
                }

                self.insert(image_embedding=embed_image, text_embedding=embed_text, metadata=frame_meta)

    def __build_keyframe_ACM(self, data_root, image_batch_size=8):
        self.__build_keyframe_AIC(data_root, image_batch_size=image_batch_size)

    def __build_roomelsa(self, data_root, image_batch_size=8):
        object_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        for obj_dir in tqdm(object_dirs, desc="Building Milvus RoomElsa"):
            obj_name = os.path.basename(obj_dir)
            image_files = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            text_files = [f for f in os.listdir(obj_dir) if f.lower().endswith(".txt")]

            if image_files:
                for path in image_files:
                    emb = self.encode_image(path)
                    self.insert(image_embedding=emb, metadata={"object_name": obj_name, "image_path": path})

            if text_files:
                text_path = os.path.join(obj_dir, text_files[0])
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    text_embed = self.encode_text(text)
                    self.insert(text_embedding=text_embed, metadata={"object_name": obj_name, "text_path": text_path, "text": text})
