from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import numpy as np
import torch
from transformers import AutoModel
import os
import json
from tqdm import tqdm
from PIL import Image
from transform import load_image
import time
import datetime
import math
from indexing import FAISSManager

class MilvusManager:
    def __init__(self, 
                 embedding_dim=1024, 
                 host="localhost", 
                 port="19530", 
                 model_path="jinaai/jina-clip-v2"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.metric_type = "IP"  # Fixed to use COSINE metric
        self.index_type = "HNSW"
        # Connect to Milvus
        connections.connect(alias="default", host=host, port=port)

        # Load model
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(self.device).eval()

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
        self._init_milvus_collections()

    def _init_milvus_collections(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(fields, description="Unified multimodal index")

        if not utility.has_collection("multimodal_index"):
            self.collection = Collection(name="multimodal_index", schema=schema)
            
            # Create indexes with explicit names and specified metric type
            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": 1024} if self.index_type == "IVF_FLAT" else {}
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
            self.collection = Collection(name="multimodal_index")
            print("Connected to existing collection")

    def encode_image(self, image_path):
        """Encode a single image"""
        emb = self.model.encode_image([image_path], truncate_dim=self.embedding_dim)[0]#, truncate_dim=self.embedding_dim)[0]
        return emb / np.linalg.norm(emb)
    
    def encode_image_batch(self, image_paths, batch_size=8):
        """Encode a batch of images"""
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            emb = self.model.encode_image(batch, truncate_dim=self.embedding_dim)#, truncate_dim=self.embedding_dim)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)
    
    def encode_text(self, text):
        """Encode text"""
        if isinstance(text, str):
            text = [text]
        emb = self.model.encode_text(text, truncate_dim=self.embedding_dim)#, truncate_dim=self.embedding_dim)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb[0] if len(text) == 1 else emb
    

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
            self.collection.flush()  # Ensure data is persisted
        except Exception as e:
            print(f"Error inserting data: {e}")
            raise
    def reset_collection(self):
        """Drop and recreate collection - WARNING: This deletes all data"""
        if utility.has_collection("multimodal_index"):
            utility.drop_collection("multimodal_index")
            print("Dropped existing collection")
        
        self._init_milvus_collections()
        print("Recreated collection with proper indexes")
    def _ensure_collection_loaded(self):
        """Ensure collection is properly loaded with explicit index checking"""
        try:
            # Check if both indexes exist
            image_index_exists = False
            text_index_exists = False
            
            try:
                self.collection.describe_index("image_index")
                image_index_exists = True
            except Exception:
                pass
                
            try:
                self.collection.describe_index("text_index")
                text_index_exists = True
            except Exception:
                pass
            
            # Create missing indexes
            if not image_index_exists or not text_index_exists:
                index_params = {
                    "metric_type": self.metric_type,
                    "index_type": self.index_type,
                    "params": {"nlist": 1024} if self.index_type == "IVF_FLAT" else {}
                }
                
                if not image_index_exists:
                    self.collection.create_index(
                        field_name="image_embedding", 
                        index_params=index_params,
                        index_name="image_index"
                    )
                    print("Created image_index")
                
                if not text_index_exists:
                    self.collection.create_index(
                        field_name="text_embedding", 
                        index_params=index_params,
                        index_name="text_index"
                    )
                    print("Created text_index")
            
            # Load collection if not already loaded
            self.collection.load()
            print("Collection loaded successfully")
                
        except Exception as e:
            print(f"Error ensuring collection loaded: {e}")
            raise
        

        
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
        self._ensure_collection_loaded()
        
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

        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field=anns_field,
            param={"metric_type": self.metric_type, "params": {}},
            limit=top_k,
            output_fields=output_fields,
            expr=expr
        )

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
        print("videos:", video_dirs)
        metadata_path = os.path.join(data_root, "metadata.json")
        for video_dir in tqdm(video_dirs, desc="Building Milvus AIC"):
            if not os.path.exists(metadata_path):
                continue

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            video_name = os.path.basename(video_dir)
            metadata_4vid = metadata[video_name]
            frame_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.lower().endswith(('.jpg', '.webp'))]
            # print('frame:', frame_paths)
            for frame_path in tqdm(frame_paths, desc=f'buiding in {video_name}'):
                frame_name = os.path.basename(frame_path).split('.')[0]
                if frame_name not in metadata_4vid:
                    continue
                meta = metadata_4vid[frame_name]
                embed_image = self.encode_image(frame_path)
                embed_text = self.encode_text(meta.get("ocr", ""))

                frame_meta = {
                    "video_name": video_name,
                    "frame_name": frame_name,
                    "frame_path": frame_path,
                    "shot": meta.get("shot"),
                    "timestamp": meta.get("time-stamp"),
                    "tags": meta.get("tags", [])
                }

                self.insert(image_embedding=embed_image, text_embedding=embed_text, metadata=frame_meta)

    def __build_keyframe_ACM(self, data_root, image_batch_size=8):
        "update later for Ctrl+F GOD mode :DD"
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
    manager = MilvusManager(port = 19530, host = 'milvus-standalone')
    tag_manager = FAISSManager(index_dir='tag_index')
    # Build tag database de ho tro tag filtering + multiprocessing
    # tag_manager.build_tag(tag_txt_file='tag.txt')
    # tag_manager.save_tag(save_dir='tag_index')
    # manager.reset_collection()
    # Hàm reset này không nổ gì đừng sài, nó xóa sạch dữ liệu đó :))))
    # manager.build(mode = 'AIC', data_root= '/root/demo_data/data')
    # Database build xong rồi, không nổ gì thì đừng chạy hàm build nha, nó build lại từ đầu bomboclat đó
    query_A = 'A Man on a road'
    tags_A = tag_manager.search_tag(search_in='tag', query_type='text', top_k=5)
    results = manager.search(mode='text', query=query_A, search_in = 'image', start_temporal_chain = True)
    print('Original top k query A:')
    for result in results:
        print(f"frame: {result['metadata']['frame_name']}, score: {result['score']}")
    query_B = 'A man eating a women'
    tags_B = tag_manager.search_tag(search_in='tag', query_type='text', top_k=5)
    temporal_answer = manager.temporal_search_sequence(query_B, mode ='text')
    print('Top K query B:')
    for result in temporal_answer["keyframes_B"]:
        print(f"frame: {result['metadata']['frame_name']}, score: {result['sim_score']}")
    print('Top K query A reranked:')
    for result in temporal_answer["keyframe_A_reranked"]:
        print(f"frame: {result['metadata']['frame_name']}, temporal score: {result['temporal_score']}")

        # Test similarity calculation
    # text1 = "A man walking on the road"
    # text2 = "Person walking on street"
    # text3 = "Cat sitting on chair"

    # emb1 = manager.encode_image('/root/demo_data/data/L01_V001/frame_545.webp')
    # emb2 = manager.encode_image('/root/demo_data/data/L01_V001/frame_542.webp')
    # emb3 = manager.encode_image('/root/demo_data/data/L01_V001/frame_543.webp')

    # # # Calculate cosine similarity manually
    # def cosine_similarity(a, b):
    #     return np.dot(a, b) 

    # print(f"Similarity text1-text2: {cosine_similarity(emb1, emb2)}")
    # print(f"Similarity text1-text3: {cosine_similarity(emb1, emb3)}")
