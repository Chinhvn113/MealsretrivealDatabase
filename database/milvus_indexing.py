from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import numpy as np
import torch
from transformers import AutoModel
import os
import json
from tqdm import tqdm
from PIL import Image
from .transform import load_image
from time import datetime
import math
class keyframe():
    def __init__(self, video_id, timestamp, sim_score):
        self.video_id = video_id
        if type(timestamp) == str:
            dt = datetime.strptime(timestamp, '%H:%M:%S')
            timestamp = dt.hour * 3600 + dt.minute * 60 + dt.second
        self.timestamp = timestamp
        self.sim_score = sim_score
        
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

    def search(self, query, mode="text", top_k=5, tags_filter=None, tags_filter_mode="any"):
        """
        Searches the Milvus collection for relevant keyframes with optional tag filtering.

        Args:
            query (str): The search query (text or image path/data).
            mode (str): The search mode, either "text" or "image".
            top_k (int): The number of top results to return.
            tags_filter (list, optional): A list of tags (strings) to filter the results by.
                                          Only results containing these tags will be returned.
            tags_filter_mode (str, optional): Determines how tags_filter is applied:
                                              - "any": (default) Returns documents containing at least one of the tags. (OR logic)
                                              - "all": Returns documents containing all specified tags. (AND logic)

        Returns:
            list: A list of dictionaries, each containing:
                  - 'metadata': The metadata of the keyframe.
                  - 'score': The similarity score.
                  - 'text_embedding' or 'image_embedding': The embedding vector of the keyframe.
        """
        self.collection.load()

        if mode == "text":
            query_vector = self.encode_text(query)
            anns_field = "text_embedding"
        elif mode == "image":
            query_vector = self.encode_image(query)
            anns_field = "image_embedding"
        else:
            raise ValueError("Mode must be 'text' or 'image'")

        expr = None
        if tags_filter:
            if not isinstance(tags_filter, list):
                raise TypeError("tags_filter must be a list of strings.")
            if not tags_filter:
                pass # Empty tags_filter list, no filtering
            elif tags_filter_mode == "any":
                # OR logic: metadata['tags'] contains any of the tags in tags_filter
                formatted_tags = ", ".join([f"'{tag}'" for tag in tags_filter])
                expr = f"metadata['tags'] array_contains_any [{formatted_tags}]"
            elif tags_filter_mode == "all":
                # AND logic: metadata['tags'] contains ALL of the tags in tags_filter
                contains_clauses = [f"metadata['tags'] array_contains '{tag}'" for tag in tags_filter]
                expr = " AND ".join(contains_clauses)
            else:
                raise ValueError("tags_filter_mode must be 'any' or 'all'.")

        output_fields = ["metadata", anns_field]

        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field=anns_field,
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
            output_fields=output_fields,
            expr=expr
        )

        formatted_results = []
        if results and results[0]:
            for hit in results[0]:
                result_item = {
                    "metadata": hit.entity.get("metadata"),
                    "score": hit.distance,
                    anns_field: hit.entity.get(anns_field)
                }
                formatted_results.append(result_item)

        return formatted_results

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
                                mode: int=1):# 1 for linear, 2 for natural exponential
        def to_seconds(ts):
            if isinstance(ts, int):
                return ts
            dt = datetime.strptime(ts, "%H:%M:%S")
            return dt.hour * 3600 + dt.minute * 60 + dt.second

        tsA = to_seconds(keyframeA["timestamp"])
        tsB = to_seconds(keyframeB["timestamp"])
        simA = keyframeA["sim_score"]
        simB = keyframeB["sim_score"]
        distance = abs(tsA - tsB)

        if distance >= threshold_maximum:
            return None

        weight_B = 1 - weight_A
        if mode == "natural exponential":
            penalty = weight_lamda * (1 - math.exp(-weight_gamma * distance / threshold_alpha))
        elif mode == "linear":
            penalty = weight_lamda * (distance / threshold_alpha)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return weight_A * simA + weight_B * simB - penalty
    
    def temporal_search_sequence(self, query, mode="text", top_k=5,
                                    temporal_params=None):
        if temporal_params is None:
            temporal_params = {
                'threshold_maximum': 300,
                'threshold_alpha': 60.0,
                'weight_A': 0.5,
                'weight_lamda': 0.3,
                'weight_gamma': 1.0,
                'mode': 1
            }
        # Determine if this is the first query in the sequence (A)
        query_is_A = not hasattr(self, 'temporal_state') or len(self.temporal_state["query_history"]) == 0

        # Update temporal_params if provided
        if hasattr(self, 'temporal_params'):
            self.temporal_params.update(temporal_params)
        else:
            self.temporal_params = temporal_params.copy()

        if query_is_A:
            # New chain: reset
            self.temporal_state = {
                "query_history": [],
                "query_results": {},
                "keyframe_A_reranked": [],
                "videos_in_A": set()
            }

        state = self.temporal_state
        query_id = len(state["query_history"])
        query_name = f"{chr(65 + query_id)}"  # "A", "B", "C", ...

        # 1. Encode query
        embedding = self.encode_text(query) if mode == "text" else self.encode_image(query)

        # 2. Milvus expr filter if not A
        expr = ""
        if query_name != "A":
            expr = f'video_name in ["' + '", "'.join(state["videos_in_A"]) + '"]'

        # 3. Perform Milvus search ONCE
        hits = self.milvus_client.search(
            data=[embedding],
            anns_field="embedding",
            param=self.search_param,
            limit=top_k,
            expr=expr if expr else None,
            output_fields=["video_name", "timestamp"]
        )[0]

        # 4. Format search results
        parsed_hits = [{
            "embedding": h.entity["embedding"],
            "metadata": h.entity["metadata"],
            "sim_score": h.distance
        } for h in hits]

        state["query_history"].append({"query": query, "mode": mode})
        state["query_results"][query_name] = parsed_hits

        if query_name == "A":
            # Save keyframe A with original scores and video list
            for h in parsed_hits:
                h["original_score"] = h["sim_score"]
            state["keyframe_A_reranked"] = parsed_hits
            state["videos_in_A"] = {h["metadata"]["video_name"] for h in parsed_hits}
        else:
            # Re-rank A with respect to new query (B, C, ...)
            b_hits_by_video = {}
            for b in parsed_hits:
                vid = b["metadata"]["video_name"]
                b_hits_by_video.setdefault(vid, []).append(b)

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

        # === Final return
        return {
            "query": f"query_{query_name}",
            "keyframe_A_reranked": state["keyframe_A_reranked"],
            f"keyframes_{query_name}": parsed_hits if query_name != "A" else None
        }

        
