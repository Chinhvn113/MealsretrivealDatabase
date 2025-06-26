from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import numpy as np
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
from transform import load_image
import time
import datetime
import math
from indexing import FAISSManager
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from transformers import SiglipProcessor, SiglipModel
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer\

class MilvusManager:
    def __init__(self, 
                 host="localhost", 
                 port="19530", 
                 model_path="jinaai/jina-clip-v2",
                 #"jinaai/jina-clip-v2"
                 #"google/siglip2-large-patch16-512"
                 #"google/siglip2-so400m-patch16-384"
                 database_mode="nvidia_aic", # "AIC", "ACM", "roomelsa", "nvidia_aic"
                 ):     
        self.model_name = model_path
        if database_mode not in ["AIC", "ACM", "roomelsa", "nvidia_aic"]:
            raise ValueError("Invalid database_mode. Choose from 'AIC', 'ACM', 'roomelsa', 'nvidia_aic'.")
        self.database_mode = database_mode
        if self.database_mode == 'nvidia_aic':
            self.model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
            self.collection_name = "fewshot_retrieve"
            self.embedding_dim = 384  # Default dimension for MiniLM
            self.model = SentenceTransformer(self.model_name)
            self.processor = None  # No processor needed for SentenceTransformer
        else:
            if "siglip" in self.model_name: 
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = SiglipModel.from_pretrained(self.model_name, device_map="auto", attn_implementation="sdpa")
                self.processor = SiglipProcessor.from_pretrained(self.model_name)
                if self.model_name == "google/siglip2-large-patch16-512":
                    self.embedding_dim = 1024
                elif self.model_name == "google/siglip2-so400m-patch16-384":
                    self.embedding_dim = 1152
                elif self.model_name == "google/siglip2-base-patch16-512":
                    self.embedding_dim = 768
                elif self.model_name == "google/siglip2-so400m-patch14-384":
                    self.embedding_dim = 1152
                
                self.collection_name = "multimodal_index_siglip"
            if "jina" in self.model_name:
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device).eval().half()
                self.embedding_dim = 1024
                self.collection_name = "multimodal_index_jina"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.metric_type = "COSINE"  # Fixed to use COSINE metric
        self.index_type = "HNSW"
        self.params={"M": 16, "efConstruction": 200}
        # Connect to Milvus
        connections.connect(alias="default", host=host, port=port)
        # Initialize metadata stores
        self.image_metadata = []
        self.text_metadata = []
        self.temporal_state = {
            "query_history": [],
            "query_results": {},
            "videos_in_A": [],
            "keyframe_A_reranked": []
        }
        # Init Milvus schema & collection
        self._init_milvus_collections(self.collection_name)
        # self.ensure_collection_loaded()
        self.collection.load()

    def _init_milvus_collections(self, collection_name, dim=None, override_name=False):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(fields, description="Unified multimodal index")

        if not utility.has_collection(collection_name):
            self.collection = Collection(name=collection_name, schema=schema)
            
            # Create indexes with explicit names and specified metric type
            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": 1024} if self.index_type == "IVF_FLAT" else self.params
            }
            
            # Create indexes with explicit names to avoid ambiguity
            self.collection.create_index(
                field_name="image_embedding", 
                index_params=index_params,
                index_name="image_index"
            )
            self.collection.create_index(
                field_name="text_embedding", 
                index_params=index_params,
                index_name="text_index"
            )
        else:
            self.collection = Collection(name=collection_name)
            print("Connected to existing collection")

    def encode_image(self, image_path):
        """Encode a single image"""
        encode_start = time.time()
        if "jina" in self.model_name:
            with torch.no_grad():
                emb = self.model.encode_image([image_path], truncate_dim=self.embedding_dim)[0]#, truncate_dim=self.embedding_dim)[0]
                print("embeding shape:",emb.shape)
        else:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=[image], return_tensors="pt").to(model.device)
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs).cpu()   
                print("embeding shape:",emb.shape) 
        encode_end = time.time()
        print(f'Encode image speed:{(encode_end-encode_start):.3f}')
        return emb.numpy()

    def encode_image_batch(self, image_paths, batch_size=8):
        """Encode a batch of image paths (no normalization)"""
        start = time.time()
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            if "jina" in self.model_name:
                with torch.no_grad():
                    emb = self.model.encode_image(batch, truncate_dim=self.embedding_dim)
            else:
                images = [Image.open(p).convert("RGB") for p in batch]
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    emb = self.model.get_image_features(**inputs).cpu()
                    print("embeding shape:",emb.shape)
                all_embeddings.append(emb.numpy())
        end = time.time()
        print('Encoding time:', round((end - start), 3))
        return np.vstack(all_embeddings)
    
    def encode_text(self, text:str):
        encode_start = time.time()
        """Encode text"""
        if isinstance(text, str):
            text = [text]
        # encode_end = time.time()
        if "jina" in self.model_name:
            with torch.no_grad():
                emb = self.model.encode_text(text, truncate_dim=self.embedding_dim)#, truncate_dim=self.embedding_dim)
                print("embeding shape:",emb.shape)
        else:
            inputs = self.processor(text=text, return_tensors="pt", padding="max_length", max_length=64)
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs).cpu()
                emb = emb.numpy()
                print("embeding shape:",emb.shape)
        encode_end = time.time()
        print(f'Encode text speed:{(encode_end-encode_start):.3f}')
        return emb[0] if len(text) == 1 else emb

    def encode_text_batch(self, texts, batch_size=32):
        """Encode a batch of texts (no normalization)"""
        start = time.time()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = self.encode_text(batch)
            all_embeddings.append(emb)
        end = time.time()
        print('Encoding time:', round((end - start), 3))
        return np.vstack(all_embeddings)

    def insert(self, image_embedding=None, text_embedding=None, metadata=None):
        def wrap(x):
            if x is None:
                return [None]
            if isinstance(x, np.ndarray):
                if x.ndim == 1:
                    return [x.tolist()]
                elif x.ndim == 2 and x.shape[0] == 1:
                    return [x[0].tolist()]
                else:
                    raise ValueError(f"Unexpected embedding shape: {x.shape}")
            if isinstance(x, list):
                return [x]
            return [x]
        
        try:
            self.collection.insert([
                wrap(image_embedding if image_embedding is not None else np.zeros(self.embedding_dim)),
                wrap(text_embedding if text_embedding is not None else np.zeros(self.embedding_dim)),
                wrap(metadata if metadata else {})
            ])
            # self.collection.flush()  # Ensure data is persisted
        except Exception as e:
            print(f"Error inserting data: {e}")
            raise

    def insert_batch(self, image_embeddings=None, text_embeddings=None, metadatas=None):
        """
        Insert multiple records in a single batch operation for improved performance.
        
        Args:
            image_embeddings: List of image embeddings or None
            text_embeddings: List of text embeddings or None  
            metadatas: List of metadata dictionaries
        """
        if not any([image_embeddings, text_embeddings, metadatas]):
            return
            
        # Determine batch size
        batch_size = 0
        if image_embeddings is not None:
            batch_size = len(image_embeddings)
        elif text_embeddings is not None:
            batch_size = len(text_embeddings)
        elif metadatas is not None:
            batch_size = len(metadatas)
            
        if batch_size == 0:
            return

        # Prepare data lists
        image_data = []
        text_data = []
        metadata_data = []

        for i in range(batch_size):
            # Handle image embeddings
            if image_embeddings is not None and i < len(image_embeddings):
                emb = image_embeddings[i]
                if isinstance(emb, np.ndarray):
                    image_data.append(emb.tolist())
                else:
                    image_data.append(emb)
            else:
                image_data.append(np.zeros(self.embedding_dim).tolist())
            
            # Handle text embeddings
            if text_embeddings is not None and i < len(text_embeddings):
                emb = text_embeddings[i]
                if isinstance(emb, np.ndarray):
                    text_data.append(emb.tolist())
                else:
                    text_data.append(emb)
            else:
                text_data.append(np.zeros(self.embedding_dim).tolist())
            
            # Handle metadata
            if metadatas is not None and i < len(metadatas):
                metadata_data.append(metadatas[i])
            else:
                metadata_data.append({})

        try:
            self.collection.insert([image_data, text_data, metadata_data])
            print(f"Batch inserted {batch_size} records")
        except Exception as e:
            print(f"Error in batch insert: {e}")
            raise

    def reset_collection(self):
        """Drop and recreate collection - WARNING: This deletes all data"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print("Dropped existing collection")
        # self.collection.drop_index()
        self._init_milvus_collections(self.collection_name)
        print("Recreated collection with proper indexes")
    def ensure_collection_loaded(self):
        """Ensure collection is properly loaded with index checks"""
        try:
            # Get existing indexes
            index_infos = self.collection.indexes
            index_fields = {index.field_name for index in index_infos}

            # Only create index if it doesn't exist
            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": 1024} if self.index_type == "IVF_FLAT" else self.params
            }

            if "image_embedding" not in index_fields:
                self.collection.create_index(
                    field_name="image_embedding",
                    index_params=index_params,
                    index_name="image_index"
                )
                print("Created image index.")

            if "text_embedding" not in index_fields:
                self.collection.create_index(
                    field_name="text_embedding",
                    index_params=index_params,
                    index_name="text_index"
                )
                print("Created text index.")

            if not utility.has_collection(self.collection.name):
                self.collection.load()

        except Exception as e:
            print(f"Error ensuring collection loaded: {e}")
            raise

    def search_fewshot(self, query: str, top_k: int = 5):
        """
        Search the fewshot_retrieve collection using a natural language query.

        Args:
            query (str): User query string.
            top_k (int): Number of similar queries to retrieve.

        Returns:
            List[Dict]: Each result with keys: 'query', 'explanation', 'visual_checks', 'spatial_instructions', 'score'
        """
        if self.database_mode != "nvidia_aic":
            raise ValueError("search_fewshot can only be used in 'nvidia_aic' mode.")


        query_vector = self.model.encode(query).tolist()

        results = self.collection.search(
            data=[query_vector],
            anns_field="text_embedding",
            param={"metric_type": "COSINE", "params": {"ef": 32}},
            limit=top_k,
            output_fields=["metadata"]
        )

        formatted = []
        if results and results[0]:
            for hit in results[0]:
                meta = hit.entity.get("metadata", {})
                formatted.append({
                    "query": meta.get("query", ""),
                    "explanation": meta.get("explanation", ""),
                    "visual_checks": meta.get("visual_checks", []),
                    "spatial_instructions": meta.get("spatial_instructions", []),
                    "score": hit.distance
                })

        return formatted


        
    def search(self, query, mode="text", search_in='image', top_k=5, tags_filter=None, tags_filter_mode="any", start_temporal_chain=False):
        """
        Searches the Milvus collection for relevant keyframes with optional tag filtering.
        Optionally starts a temporal search chain with this query as Query A.

        Args:
            query (str): The search query (text or image path/data).
            mode (str): "text" or "image".
            top_k (int): Number of top results.
            tags_filter (list): List of tags to filter by.
            tags_filter_mode (str): "any" or "all" for tag filtering logic.
            start_temporal_chain (bool): If True, initializes temporal chain with this query.

        Returns:
            list: A list of result dictionaries, each with:
                - 'metadata'
                - 'score'
                - 'text_embedding' or 'image_embedding'
        """
        encode_start = time.time()        
        if mode == "text":
            query_vector = self.encode_text(query)
        elif mode == "image":
            query_vector = self.encode_image(query)
        else:
            raise ValueError("Mode must be 'text' or 'image'")
        if search_in == 'text':
            anns_field = 'text_embedding'
        elif search_in == 'image':
            anns_field = 'image_embedding'
        encode_end = time.time()
        print('Encode in search:', round((encode_end-encode_start),3))
        expr = None
        if tags_filter:
            if not isinstance(tags_filter, list):
                raise TypeError("tags_filter must be a list of strings.")
            if tags_filter_mode == "any":
                formatted_tags = ", ".join([f"'{tag}'" for tag in tags_filter])
                expr = f"metadata['tags'] array_contains_any [{formatted_tags}]"
            elif tags_filter_mode == "all":
                expr = " AND ".join([f"metadata['tags'] array_contains '{tag}'" for tag in tags_filter])
            else:
                raise ValueError("tags_filter_mode must be 'any' or 'all'.")
        output_fields = ["metadata", anns_field]
        search_start = time.time()
        search_param = {"ef": 32}
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field=anns_field,
            param={"metric_type": self.metric_type, "params": search_param},
            limit=top_k,
            output_fields=output_fields,
            expr=expr
        )
        search_end = time.time()
        print(f'Searching time: {(search_end-search_start):.2f}s')
        formatted_results = []
        embeddings = []
        if results and results[0]:
            for hit in results[0]:
                emb = hit.entity.get(anns_field)
                formatted_results.append({
                    "metadata": hit.entity.get("metadata"),
                    "score": hit.distance,
                    anns_field: emb
                })
                embeddings.append(emb)

        # Handle temporal chain start
        if start_temporal_chain:
            self._init_temporal_state(query, mode, formatted_results, embeddings)

        return formatted_results
    def _init_temporal_state(self, query, mode, results, embeddings):
        """Initialize temporal search state properly"""
        self.temporal_state = {
            "query_history": [{"query": query, "mode": mode}],
            "query_results": {"A": results},
            "videos_in_A": list(set([
                item["metadata"]["video_name"] 
                for item in results 
                if "video_name" in item["metadata"]
            ])),
            "keyframe_A_reranked": [
                {
                    "metadata": item["metadata"],
                    "original_score": item["score"],
                    "temporal_score": item["score"]
                }
                for item in results
            ]
        }

    def build(self, data_root, image_batch_size=8, insert_batch_size=100, mode=None):
        if mode == 'AIC':
            self.__build_keyframe_AIC(data_root, image_batch_size=image_batch_size, insert_batch_size=insert_batch_size)
        elif mode == 'ACM':
            self.__build_keyframe_ACM(data_root, image_batch_size=image_batch_size, insert_batch_size=insert_batch_size)
        elif mode == 'roomelsa':
            self.__build_roomelsa(data_root, image_batch_size=image_batch_size, insert_batch_size=insert_batch_size)
        elif mode == 'nvidia_aic':
            self.__build_nvidia_aic(data_root)
        else:
            raise ValueError("Invalid mode. Choose 'AIC', 'ACM', or 'roomelsa'.")
        
    def __build_nvidia_aic(self, json_path):
        """
        Builds a text-based database from Nvidia AIC-style few-shot instruction dataset.
        Stores only text embeddings (from the `query` field) with all fields as metadata.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected a list of JSON objects.")

        batch_texts = []
        batch_metadatas = []

        for item in data:
            query_text = item.get("query", "").strip()
            if not query_text:
                continue
            batch_texts.append(query_text)
            batch_metadatas.append(item)

        embeddings = self.model.encode(batch_texts)
        image_embeddings = [np.zeros(self.embedding_dim)] * len(batch_texts)

        self.insert_batch(
            image_embeddings=image_embeddings,
            text_embeddings=embeddings,
            metadatas=batch_metadatas
        )

        self.collection.flush()
        print(f"Inserted {len(batch_texts)} instruction queries into '{self.collection_name}' collection.")



 
    def __build_keyframe_AIC(self, data_root, image_batch_size=8, insert_batch_size=100):
        video_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        print("videos:", video_dirs)
        metadata_path = os.path.join(data_root, "metadata.json")
        if not os.path.exists(metadata_path):
            return
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Batch processing variables
        batch_image_embeddings = []
        batch_text_embeddings = []
        batch_metadatas = []
        batch_count = 0
        
        for video_dir in tqdm(video_dirs, desc="Building Milvus AIC"):
            video_name = os.path.basename(video_dir)
            metadata_4vid = metadata[video_name]
            frame_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.lower().endswith(('.jpg', '.webp'))]
            
            # Collect valid frames and their metadata
            valid_frames = []
            valid_texts = []
            valid_metadatas = []
            
            for frame_path in frame_paths:
                frame_name = os.path.basename(frame_path).split('.')[0]
                if frame_name not in metadata_4vid:
                    continue
                    
                meta = metadata_4vid[frame_name]
                frame_meta = {
                    "video_name": video_name,
                    "frame_name": frame_name,
                    "frame_path": frame_path,
                    "shot": meta.get("shot"),
                    "timestamp": meta.get("time-stamp"),
                    "tags": meta.get("tags", [])
                }
                
                valid_frames.append(frame_path)
                valid_texts.append(meta.get("ocr", ""))
                valid_metadatas.append(frame_meta)
            
            if not valid_frames:
                continue
                
            # Process in batches for this video
            for i in tqdm(range(0, len(valid_frames), image_batch_size), desc=f'Processing {video_name}'):
                batch_frames = valid_frames[i:i+image_batch_size]
                batch_texts_chunk = valid_texts[i:i+image_batch_size]
                batch_metas_chunk = valid_metadatas[i:i+image_batch_size]
                
                # Encode images and texts in batch
                image_embeddings = self.encode_image_batch(batch_frames, batch_size=image_batch_size)
                text_embeddings = self.encode_text_batch(batch_texts_chunk, batch_size=len(batch_texts_chunk))
                
                # Add to batch buffers
                for j in range(len(batch_frames)):
                    batch_image_embeddings.append(image_embeddings[j])
                    batch_text_embeddings.append(text_embeddings[j])
                    batch_metadatas.append(batch_metas_chunk[j])
                    batch_count += 1
                    
                    # Insert when batch is full
                    if batch_count >= insert_batch_size:
                        self.insert_batch(
                            image_embeddings=batch_image_embeddings,
                            text_embeddings=batch_text_embeddings,
                            metadatas=batch_metadatas
                        )
                        # Clear batch buffers
                        batch_image_embeddings = []
                        batch_text_embeddings = []
                        batch_metadatas = []
                        batch_count = 0
        
        # Insert remaining items
        if batch_count > 0:
            self.insert_batch(
                image_embeddings=batch_image_embeddings,
                text_embeddings=batch_text_embeddings,
                metadatas=batch_metadatas
            )
        
        # Final flush
        self.collection.flush()
        print("AIC build completed with batch insertion")

    def __build_keyframe_ACM(self, data_root, image_batch_size=8, insert_batch_size=100):
        "update later for Ctrl+F GOD mode :DD"
        self.__build_keyframe_AIC(data_root, image_batch_size=image_batch_size, insert_batch_size=insert_batch_size)

    def __build_roomelsa(self, data_root, image_batch_size=8, insert_batch_size=100):
        object_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        
        # Batch processing variables
        batch_image_embeddings = []
        batch_text_embeddings = []
        batch_metadatas = []
        batch_count = 0
        
        for obj_dir in tqdm(object_dirs, desc="Building Milvus RoomElsa"):
            obj_name = os.path.basename(obj_dir)
            image_files = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            text_files = [f for f in os.listdir(obj_dir) if f.lower().endswith(".txt")]

            # Process images in batches
            if image_files:
                for i in range(0, len(image_files), image_batch_size):
                    batch_files = image_files[i:i+image_batch_size]
                    image_embeddings = self.encode_image_batch(batch_files, batch_size=len(batch_files))
                    
                    for j, path in enumerate(batch_files):
                        batch_image_embeddings.append(image_embeddings[j])
                        batch_text_embeddings.append(np.zeros(self.embedding_dim))  # No text for images
                        batch_metadatas.append({"object_name": obj_name, "image_path": path})
                        batch_count += 1
                        
                        if batch_count >= insert_batch_size:
                            self.insert_batch(
                                image_embeddings=batch_image_embeddings,
                                text_embeddings=batch_text_embeddings,
                                metadatas=batch_metadatas
                            )
                            batch_image_embeddings = []
                            batch_text_embeddings = []
                            batch_metadatas = []
                            batch_count = 0

            # Process text files
            if text_files:
                text_path = os.path.join(obj_dir, text_files[0])
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    text_embed = self.encode_text(text)
                    batch_image_embeddings.append(np.zeros(self.embedding_dim))  # No image for text
                    batch_text_embeddings.append(text_embed)
                    batch_metadatas.append({"object_name": obj_name, "text_path": text_path, "text": text})
                    batch_count += 1
                    
                    if batch_count >= insert_batch_size:
                        self.insert_batch(
                            image_embeddings=batch_image_embeddings,
                            text_embeddings=batch_text_embeddings,
                            metadatas=batch_metadatas
                        )
                        batch_image_embeddings = []
                        batch_text_embeddings = []
                        batch_metadatas = []
                        batch_count = 0
        
        # Insert remaining items
        if batch_count > 0:
            self.insert_batch(
                image_embeddings=batch_image_embeddings,
                text_embeddings=batch_text_embeddings,
                metadatas=batch_metadatas
            )
        
        # Final flush
        self.collection.flush()
        print("RoomElsa build completed with batch insertion")
                    
    def search_queryB_in_video(self, query_b_text, video_name, top_k=5, object_filtering=False, required_tags=None):
        query_b_vector = self.encode_text(query_b_text).tolist()
        expr = f'metadata["video_name"] == "{video_name}"'

        if object_filtering and required_tags:
            tag_filters = [f'"{tag}" in metadata["tags"]' for tag in required_tags]
            expr += " and " + " and ".join(tag_filters)

        self.collection.load()
        results = self.collection.search(
            data=[query_b_vector],
            anns_field="text_embedding",
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
            output_fields=["metadata", "text_embedding"],
            expr=expr
        )
        return results[0]  # list of top-k results in the video

    def compute_temporal_score(self, keyframeA, keyframeB, *,
                                threshold_maximum=300,
                                threshold_alpha=60.0,
                                weight_A=0.5,
                                weight_lamda=0.3,
                                weight_gamma=1.0,
                                mode: int=2):# 1 for linear, 2 for natural exponential
        def to_seconds(ts):
            if isinstance(ts, int):
                return ts
            minute = float(ts.split(':')[0])
            second = float(ts.split(':')[1])
            return minute * 60 + second

        tsA = to_seconds(keyframeA["timestamp"])
        tsB = to_seconds(keyframeB["timestamp"])
        # tsA = keyframeA["timestamp"]
        # tsB = keyframeB["timestamp"]
        simA = keyframeA["sim_score"]
        simB = keyframeB["sim_score"]
        distance = abs(tsA - tsB)

        if distance >= threshold_maximum:
            return None

        weight_B = 1 - weight_A
        if mode == 2:
            penalty = weight_lamda * (1 - math.exp(-weight_gamma * distance / threshold_alpha))
        elif mode == 1:
            penalty = weight_lamda * min(distance / threshold_alpha, 1.0)
        else:
            raise ValueError(f"Unsupported mode: {mode} use 1 for linear or 2 for natural exponential")

        return weight_A * simA + weight_B * simB - penalty
    
    def temporal_search_sequence(self, query, mode="text", search_in='image', top_k=5):
        """
        Handles sequential temporal search reranking based on temporal consistency
        across multiple queries (A, B, C, ...). Query A must be searched externally
        using self.search() and passed in before calling this function.
        top_k query B,C format:
        {
            "metadata": h.entity["metadata"],
            "embedding": h.entity[anns_field],
            "sim_score": h.distance
        }
        reranked query A output format:
        
        {
            "metadata": a["metadata"],
            "temporal_score": best_temporal_score
        }
        
        """
        # Proper validation
        if not hasattr(self, 'temporal_state') or not self.temporal_state.get("query_history"):
            raise ValueError("Temporal search chain not initialized. Call search() with start_temporal_chain=True first.")
        
        state = self.temporal_state
        query_id = len(state["query_history"])
        
        if query_id == 0:
            raise ValueError("No Query A found. Initialize temporal chain first.")
        query_name = f"{chr(65 + query_id)}"  # "B", "C", ...

        # Encode new query
        embedding = self.encode_text(query) if mode == "text" else self.encode_image(query)

        # Restrict search to same videos as keyframe A
        expr = f'metadata["video_name"] in ["' + '", "'.join(state["videos_in_A"]) + '"]'

        # Perform filtered search
        anns_field = "text_embedding" if search_in == "text" else "image_embedding"
        hits = self.collection.search(
            data=[embedding.tolist()],
            anns_field=anns_field,
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
            output_fields=["metadata", anns_field],
            expr=expr
        )[0]

        # Format current query hits
        parsed_hits = []
        for h in hits:
            parsed_hits.append({
                "metadata": h.entity["metadata"],
                "embedding": h.entity[anns_field],
                "sim_score": h.distance
            })

        # Save query in state
        state["query_history"].append({"query": query, "mode": mode})
        state["query_results"][query_name] = parsed_hits

        # Group B hits by video
        b_hits_by_video = {}
        for b in parsed_hits:
            vid = b["metadata"]["video_name"]
            b_hits_by_video.setdefault(vid, []).append(b)

        # Re-rank keyframe A
        reranked_A = []
        for a in state["keyframe_A_reranked"]:
            vid = a["metadata"]["video_name"]
            if vid not in b_hits_by_video:
                continue
            best_temporal_score = -float("inf")
            for b in b_hits_by_video[vid]:
                score = self.compute_temporal_score(
                    keyframeA={"timestamp": a["metadata"]["timestamp"], "sim_score": a["original_score"]},
                    keyframeB={"timestamp": b["metadata"]["timestamp"], "sim_score": b["sim_score"]}
                )
                if score is not None and score > best_temporal_score:
                    best_temporal_score = score
            if best_temporal_score > -float("inf"):
                reranked_A.append({
                    "metadata": a["metadata"],
                    "temporal_score": best_temporal_score
                })

        reranked_A.sort(key=lambda x: x["temporal_score"], reverse=True)
        state["keyframe_A_reranked"] = reranked_A

        return {
            "query": f"query_{query_name}",
            "keyframe_A_reranked": reranked_A,
            f"keyframes_{query_name}": parsed_hits
        }
        
    def __del__(self):
        """Cleanup connections when object is destroyed."""
        try:
            if hasattr(self, 'collection'):
                self.collection.release()
            if hasattr(self, 'tag_collection'):
                self.tag_collection.release()
            connections.disconnect("default")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            
if __name__ == '__main__':
    manager = MilvusManager(port = 19530, host = 'milvus-standalone', database_mode='nvidia_aic')
    manager.build(data_root='/root/demo_data/MealsretrivevalDatabase/batch_4_refined.json', mode='nvidia_aic')
    # manager.ensure_collection_loaded()
    # query='Considering the pallets <region0> <region1> <region2> <region3> <region4> <region5> <region6> <region7> <region8>, what is the total number of pallets in the left buffer region among <region9> <region10> <region11>?'
    # results = manager.search_fewshot(query)
    # for result in results:
    #     print(f"Query: {result['query']}")
    #     print(f"Explanation: {result['explanation']}")
    #     print(f"Visual Checks: {result['visual_checks']}")
    #     print(f"Spatial Instructions: {result['spatial_instructions']}")
    #     print(f"Score: {result['score']:.4f}\n")
    # tag_manager = FAISSManager(tag_dir='tag_index')
    # manager.reset_collection()
    # manager.build(data_root='/root/demo_data/data', mode='AIC')
    # manager.ensure_collection_loaded()
    ##################SEARCHING TEST################################################################################
    # start = time.time()
    # query_A = 'A man walking on a road'
    # tags_A = tag_manager.search_tag( query=query_A, top_k=1000)
    # print('tags_A:', *tags_A)
    # end = time.perf_counter()
    # print(f"Elapsed time: {end - start:.4f} seconds")
    # results = manager.search(mode='text', query=query_A, search_in = 'image', start_temporal_chain = True)
    # print('Original top k query A:')
    # for result in results:
    #     print(f"frame: {result['metadata']['frame_name']}, score: {result['score']}")
    # end = time.time()
    # print(f"Elapsed time: {end - start:.2f} seconds")

    # query_B = 'A man eating a woman'
    # tags_B = tag_manager.search_tag(query=query_B, top_k=1000)
    # print('tags_B:', *tags_B)
    # temporal_answer = manager.temporal_search_sequence(query_B, mode ='text')
    # print('Top K query B:')
    # for result in temporal_answer["keyframes_B"]:
    #     print(f"frame: {result['metadata']['frame_name']}, score: {result['sim_score']}")
    # print('Top K query A reranked:')
    # for result in temporal_answer["keyframe_A_reranked"]:
    #     print(f"frame: {result['metadata']['frame_name']}, temporal score: {result['temporal_score']}")
    # query_C = 'Car floating in a flooded river.'
    # tags_C = tag_manager.search_tag(query=query_C, top_k=1000)
    # print('tags_C:', *tags_C)


