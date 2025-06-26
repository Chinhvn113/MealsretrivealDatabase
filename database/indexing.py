import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import json
import faiss
from tqdm import tqdm
from transform import load_image
from sentence_transformers import SentenceTransformer
from time import time
class FAISSManager:
    def __init__(
        self,
        embedding_dim=1024,
        device=None,
        index_dir=None,
        tag_dir=None,
        model_path="jinaai/jina-clip-v2",
        image_size=384,
        max_num=12,
        database_mode='nvidia_aic'  # Default mode, can be overridden
        ):
        """
        Initialize the FAISS Manager for both building and retrieving from FAISS indexes

        Args:
            embedding_dim: Dimension of the embedding vectors
            device: Device to use for model inference ('cuda' or 'cpu')
            index_dir: Directory containing existing indexes to load (optional)
            tag_dir: Directory containing existing tag index to load (optional)
            model_path: Path to the vision-language model
            image_size: Size to which images are resized
            max_num: Max number of search results (not used consistently, consider removing)
        """
        print("[DEBUG] Initializing FAISSManager")
        if database_mode == 'nvidia_aic':
            self.model= SentenceTransformer(model_path, device=device)
            self.embedding_dim = 384
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.embedding_dim = embedding_dim
            self.image_size = image_size
            self.max_num = max_num
            # self.database_mode = None  # To be set by the build method

            # Load CLIP model
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half()
            self.model.to(self.device)
            self.model.eval()

        # Initialize FAISS indexes
        try:
            self.res = faiss.StandardGpuResources()
            self.image_index = faiss.GpuIndexFlatIP(self.res, self.embedding_dim)
            self.text_index = faiss.GpuIndexFlatIP(self.res, self.embedding_dim)
            self.mean_pooling_image_index = faiss.GpuIndexFlatIP(self.res, self.embedding_dim)
            self.tag_index = faiss.GpuIndexFlatIP(self.res, self.embedding_dim)
            self.fewshot_index = faiss.GpuIndexFlatIP(self.res, self.embedding_dim) # New index for few-shot
        except Exception:
            print("[WARNING] No GPU found for FAISS. Falling back to CPU.")
            self.image_index = faiss.IndexFlatIP(self.embedding_dim)
            self.text_index = faiss.IndexFlatIP(self.embedding_dim)
            self.mean_pooling_image_index = faiss.IndexFlatIP(self.embedding_dim)
            self.tag_index = faiss.IndexFlatIP(self.embedding_dim)
            self.fewshot_index = faiss.IndexFlatIP(self.embedding_dim) # New index for few-shot

        # Metadata mapping
        self.image_metadata = []
        self.text_metadata = []
        self.mean_pooling_image_metadata = []
        self.tag_metadata = []
        self.fewshot_metadata = [] # New metadata list for few-shot

        # Load existing indexes if provided
        if index_dir and os.path.exists(index_dir):
            self.load(index_dir)
        if tag_dir and os.path.exists(tag_dir):
            self.load_tag(tag_dir)

    def encode_image(self, image_path):
        """Encode a single image"""
        emb = self.model.encode_image([image_path], truncate_dim=self.embedding_dim)[0]
        return emb / np.linalg.norm(emb)

    def encode_image_batch(self, image_paths, batch_size=8):
        """Encode a batch of images"""
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            emb = self.model.encode_image(batch, truncate_dim=self.embedding_dim)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)

    def encode_text(self, text):
        """Encode text"""
        if isinstance(text, str):
            text = [text]
        emb = self.model.encode_text(text, truncate_dim=self.embedding_dim)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb[0] if len(text) == 1 else emb

    def __build_nvidia_aic(self, json_path):
        """
        Builds a text-based database from an Nvidia AIC-style few-shot instruction dataset.
        This method is private and called by the main `build` method.

        Args:
            json_path (str): The path to the JSON file containing the dataset.
        """
        print(f"Building few-shot database from: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected the JSON file to contain a list of objects.")
        existed_queries = set()
        for item in tqdm(data, desc="Processing few-shot queries"):
            query_text = item.get("query", "").strip()
            if not query_text or query_text in existed_queries:
                continue
            existed_queries.add(query_text)

            try:
                # Encode the query text
                embedding = self.model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)

                # Add embedding to the FAISS index
                self.fewshot_index.add(np.expand_dims(embedding.astype(np.float32), axis=0))

                # Store the entire original item as metadata
                self.fewshot_metadata.append(item)

            except Exception as e:
                print(f"[ERROR] Error processing query '{query_text}': {e}")

        print(f"Inserted {self.fewshot_index.ntotal} instruction queries into the few-shot index.")


    def __build_roomelsa(self, data_root, image_batch_size=8):
        """
        Build FAISS indexes from dataset directory
        
        Args:
            data_root: Root directory containing object data
            image_batch_size: Batch size for image encoding
        """
        object_dirs = [os.path.join(data_root, object) for object in os.listdir(data_root) 
                     if os.path.isdir(os.path.join(data_root, object))]
        print('object dirs:', object_dirs[:5])
        
        for obj_dir in tqdm(object_dirs, desc="Processing objects"):
            obj_name = os.path.basename(obj_dir)
            print(f"[INFO] Processing {obj_name}")
            
            view_subdirs = [d for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d))]
            demo_img = [f for f in os.listdir(obj_dir) if f.lower().endswith('.jpg')]
            if not demo_img:
                continue
            demo_img_embeds = self.encode_image(os.path.join(obj_dir, demo_img[0]))
            
            if not view_subdirs:
                print(f"[WARNING] No 2D view directory in {obj_name}")
                continue
            
            view_dir = os.path.join(obj_dir, view_subdirs[0])
            img_files = [os.path.join(view_dir, f) for f in os.listdir(view_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if img_files:
                img_embeds = self.encode_image_batch(img_files, batch_size=image_batch_size)
                img_embeds = np.vstack([img_embeds, demo_img_embeds])
                
                for embed in img_embeds:
                    self.image_index.add(np.expand_dims(embed.astype(np.float32), axis=0))
                    self.image_object_dirs.append(obj_name)
                
                mean_img_embed = np.mean(img_embeds, axis=0)
                mean_img_embed = mean_img_embed / np.linalg.norm(mean_img_embed)
                self.mean_pooling_image_index.add(np.expand_dims(mean_img_embed.astype(np.float32), axis=0))
                self.mean_pooling_image_dirs.append(obj_name)
            
            text_files = [f for f in os.listdir(obj_dir) if f.lower().endswith(".txt")]
            if text_files:
                text_path = os.path.join(obj_dir, text_files[0])
                try:
                    with open(text_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
                            text_embed = self.encode_text(text)
                            self.text_index.add(np.expand_dims(text_embed.astype(np.float32), axis=0))
                            self.text_object_dirs.append(obj_name)
                except Exception as e:
                    print(f"[WARNING] Failed to read text for {obj_name}: {e}")
        
        print("[INFO] Database build finished.")
    
    def __build_keyframe_AIC(self, data_root, image_batch_size=8):
        """
        Build FAISS indexes from dataset directory for ACM format

        Args:
            data_root: Root directory containing object data
            image_batch_size: Batch size for image encoding
        """
        video_dirs = [os.path.join(data_root, obj) for obj in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, obj))]
        print('video dirs:', video_dirs[:5])

        for video in tqdm(video_dirs, desc="Processing video"):
            video_parent = os.path.dirname(video)
            metadata_path = os.path.join(video_parent, 'metadata.json')
            with open(metadata_path, "r") as m:
                metadata = json.load(m)
                print('metadata:', list(metadata.keys())[:5])

            video_name = os.path.basename(video)
            print(f"[INFO] Processing {video_name}")

            video_path = [f for f in os.listdir(video)
                        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'))]

            frames_path = [os.path.join(video, f) for f in os.listdir(video)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', 'webp'))]
            for frame in frames_path:
                frame_name_ = os.path.basename(frame)
                frame_name = frame_name_.split('.')[0]
                frame_id = frame_name.split('_')[-1]

                print('frame name:', frame_name)

                thumbnail_text = metadata[video_name][frame_name].get("ocr", None)
                tags = metadata[video_name][frame_name].get("tags", [])
                shot = metadata[video_name][frame_name].get("shot", None)
                timestamp = metadata[video_name][frame_name].get("time-stamp", None)

                frame_metadata = {
                    "frame_id": frame_id,
                    "video_name": video_name,
                    "video_path": video_path,
                    "frame_name": frame_name,
                    "frame_path": os.path.join(video, frame),
                    "video_shot": shot,
                    "video_time": timestamp,
                    "objects": tags
                }

                # Text index
                embed_text = self.encode_text(thumbnail_text) if thumbnail_text else self.encode_text("")
                self.text_index.add(np.expand_dims(embed_text.astype(np.float32), axis=0))
                self.text_metadata.append(frame_metadata)

                # Image index
                frame_embed = self.encode_image(os.path.join(video, frame))
                self.image_index.add(np.expand_dims(frame_embed.astype(np.float32), axis=0))
                self.image_metadata.append(frame_metadata)
                
    def __build_keyframe_ACM(self, data_root, image_batch_size=8):
        """
        Build FAISS indexes from dataset directory for ACM format

        Args:
            data_root: Root directory containing object data
            image_batch_size: Batch size for image encoding
        """
        video_dirs = [os.path.join(data_root, obj) for obj in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, obj))]
        print('video dirs:', video_dirs[:5])

        for video in tqdm(video_dirs, desc="Processing video"):
            video_parent = os.path.dirname(video)
            metadata_path = os.path.join(video_parent, 'metadata.json')
            with open(metadata_path, "r") as m:
                metadata = json.load(m)
                print('metadata:', list(metadata.keys())[:5])

            video_name = os.path.basename(video)
            print(f"[INFO] Processing {video_name}")

            video_path = [f for f in os.listdir(video)
                        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'))]

            frames_path = [os.path.join(video, f) for f in os.listdir(video)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', 'webp'))]
            for frame in frames_path:
                frame_name_ = os.path.basename(frame)
                frame_name = frame_name_.split('.')[0]
                frame_id = frame_name.split('_')[-1]

                print('frame name:', frame_name)

                thumbnail_text = metadata[video_name][frame_name].get("ocr", None)
                tags = metadata[video_name][frame_name].get("tags", [])
                shot = metadata[video_name][frame_name].get("shot", None)
                timestamp = metadata[video_name][frame_name].get("time-stamp", None)

                frame_metadata = {
                    "frame_id": frame_id,
                    "video_name": video_name,
                    "video_path": video_path,
                    "frame_name": frame_name,
                    "frame_path": os.path.join(video, frame),
                    "video_shot": shot,
                    "video_time": timestamp,
                    "objects": tags
                }

                # Text index
                embed_text = self.encode_text(thumbnail_text) if thumbnail_text else self.encode_text("")
                self.text_index.add(np.expand_dims(embed_text.astype(np.float32), axis=0))
                self.text_metadata.append(frame_metadata)

                # Image index
                frame_embed = self.encode_image(os.path.join(video, frame))
                self.image_index.add(np.expand_dims(frame_embed.astype(np.float32), axis=0))
                self.image_metadata.append(frame_metadata)
                
    def build_tag(self, tag_txt_file, batch_size=32):
        """
        Build tag embeddings from a text file and insert them into separate tag collection.
        This will NOT affect your existing image/text database at all.
        
        Args:
            tag_txt_file (str): Path to the text file containing tags (one tag per line)
            batch_size (int): Number of tags to process in each batch for efficiency
        """
        if not os.path.exists(tag_txt_file):
            raise FileNotFoundError(f"Tag file not found: {tag_txt_file}")
        
        print(f"Building tag database from: {tag_txt_file}")
        
        # Read all tags from file
        with open(tag_txt_file, 'r', encoding='utf-8') as f:
            tags = [line.strip() for line in f.readlines() if line.strip()]
        
        if not tags:
            print("No tags found in the file")
            return
        
        print(f"Found {len(tags)} tags to process")
        
        # Process tags in batches for efficiency
        for i in tqdm(range(0, len(tags), batch_size), desc="Processing tag batches"):
            batch_tags = tags[i:i+batch_size]
            
            # Process each tag in the batch
            for j, tag in enumerate(batch_tags):
                try:
                    # Encode tag
                    tag_embedding = self.encode_text(tag)
                    
                    # Create metadata
                    tag_metadata = {
                        "tag_text": tag,
                        "tag_id": i + j,
                    }
                    self.tag_index.add(np.expand_dims(tag_embedding.astype(np.float32), axis=0))
                    self.tag_metadata.append(tag_metadata)
                except:
                    raise ValueError('[ERROR] Failed building tag database')  
    
    def build(self, data_root, image_batch_size=8, mode=None):
        """
        Builds the FAISS database from a data source based on the specified mode.

        Args:
            data_root (str): The path to the data source. This is a directory for modes
                             'AIC', 'ACM', and 'roomelsa', but a path to a JSON file
                             for 'nvidia_aic' mode.
            image_batch_size (int): Batch size for encoding images (if applicable).
            mode (str): The build mode. One of 'AIC', 'ACM', 'roomelsa', 'nvidia_aic'.
        """
        self.database_mode = mode
        if mode == 'AIC':
            self.__build_keyframe_AIC(data_root, image_batch_size=image_batch_size)
        elif mode == 'ACM':
            self.__build_keyframe_ACM(data_root, image_batch_size=image_batch_size)
        elif mode == 'roomelsa':
            self.__build_roomelsa(data_root, image_batch_size=image_batch_size)
        elif mode == 'nvidia_aic':
            self.__build_nvidia_aic(data_root) # data_root is the json_path
        else:
            raise ValueError("Invalid mode. Choose 'AIC', 'ACM', 'roomelsa', or 'nvidia_aic'.")

    def search_fewshot(self, query: str, top_k: int = 5):
        """
        Search the fewshot_retrieve collection using a natural language query.

        Args:
            query (str): User query string.
            top_k (int): Number of similar queries to retrieve.

        Returns:
            List[Dict]: Each result with keys: 'query', 'explanation', 'visual_checks', 'spatial_instructions', 'score'
        """
        # if self.database_mode != "nvidia_aic":
        #     raise ValueError("search_fewshot can only be used when the database is built in 'nvidia_aic' mode.")

        if self.fewshot_index.ntotal == 0:
            print("[WARNING] The few-shot index is empty. No search can be performed.")
            return []

        query_vector = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        query_vector = np.expand_dims(query_vector.astype(np.float32), axis=0)

        distances, indices = self.fewshot_index.search(query_vector, top_k)

        formatted_results = []
        if len(indices) > 0:
            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1: # FAISS returns -1 for no result
                    continue
                
                meta = self.fewshot_metadata[idx]
                #check if retrived metadata query exists (only retrieved once for each query)                 
                formatted_results.append({
                    "query": meta.get("query", ""),
                    "explanation": meta.get("explanation", ""),
                    "visual_checks": meta.get("visual_checks", []),
                    "spatial_instructions": meta.get("spatial_instructions", []),
                    "score": float(dist)
                })

        return formatted_results

    def search_tag(self, query, top_k=5, return_answer_vector=False):
        query_vector = self.encode_text(query)
        query_vector = query_vector[np.newaxis, :]
        index = self.tag_index
        object_dirs = self.tag_metadata
        limit = top_k
        distances, indices= index.search(query_vector, limit)
        hits = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            if object_dirs[idx]['tag_text'] in query.lower():
                hits.append(object_dirs[idx]['tag_text'])
        return hits
        
    def search(self, query, query_type="text", top_k=5, search_in="image", return_answer_vector=False):
        """
        Search for similar objects
        
        Args:
            query: str (text) or str (path to image)
            query_type: 'text' or 'image'
            top_k: Number of results to return
            search_in: 'image', 'text', or 'mean pooling images'
            return_answer_vector: If True, returns (results, query_embedding, answer_embedding)
            
        Returns:
            List of dictionaries with object_dir and confidence, and optionally embeddings
        """
        if query_type == "text":
            query_vector = self.encode_text(query)
        elif query_type == "image":
            query_vector = self.encode_image(query)
        else:
            raise ValueError("query_type must be 'text' or 'image'")
        
        query_vector = query_vector[np.newaxis, :]
        
        if search_in == "image":
            index = self.image_index
            object_dirs = self.image_metadata
            # object_root = self.image_object_root
            limit = top_k * 7
        elif search_in == "text":
            index = self.text_index
            object_dirs = self.text_metadata
            # object_root = self.text_object_root
            limit = top_k * 7
        elif search_in == "mean pooling image":
            index = self.mean_pooling_image_index
            object_dirs = self.mean_pooling_image_metadata
            # object_root = self.mean_pooling_image_root
            limit = top_k
        elif search_in == "tag":
            index = self.tag_index
            object_dirs = self.tag_metadata
            # object_root = self.mean_pooling_image_root
            limit = top_k
        else:
            raise ValueError("search_in must be 'image', 'text', or 'mean pooling image'")
        
        distances, indices= index.search(query_vector, limit)
        
        hits = []
        answer_embeddings = []  # To store answer embeddings
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            hits.append({
                "metadata": object_dirs[idx],
                "confidence": dist,
                # "object_root": object_root[idx]
            })
            # Store the answer embeddings
            answer_embeddings.append(index.reconstruct(int(idx)))  # Reconstruct the embedding from the index
        
        # Deduplicate if not mean-pooling
        output = []
        if search_in == "mean pooling images":
            output = hits[:top_k]
        else:
            hit_set = set()
            for hit in hits:
                if hit["metadata"] not in hit_set:
                    hit_set.add(hit["metadata"])
                    output.append(hit)
                    if len(output) == top_k:
                        break
        
        if return_answer_vector:
            return output, answer_embeddings
        else:
            return output
    
    def batch_search(self, queries, query_type="text", top_k=5, search_in="image"):
        """
        Batch search for multiple queries at once
        
        Args:
            queries: List of queries (text strings or image paths)
            query_type: 'text' or 'image'
            top_k: Number of results to return per query
            search_in: 'image', 'text', or 'mean pooling images'
            
        Returns:
            List of lists of dictionaries with object_dir and confidence
        """
        if query_type == "text":
            query_vectors = np.array([self.encode_text(query) for query in queries])
        elif query_type == "image":
            query_vectors = np.array([self.encode_image(query) for query in queries])
        else:
            raise ValueError("query_type must be 'text' or 'image'")
        
        if search_in == "image":
            index = self.image_index
            object_dirs = self.image_object_dirs
        elif search_in == "text":
            index = self.text_index
            object_dirs = self.text_object_dirs
        elif search_in == "mean pooling images":
            index = self.mean_pooling_image_index
            object_dirs = self.mean_pooling_image_dirs
        else:
            raise ValueError("search_in must be 'image', 'text', or 'mean pooling images'")
        
        distances, indices = index.search(query_vectors, top_k)
        
        results = []
        for idx_set, dist_set in zip(indices, distances):
            batch_hits = []
            for idx, dist in zip(idx_set, dist_set):
                if idx == -1:
                    continue
                batch_hits.append({
                    "object_dir": object_dirs[idx],
                    "confidence": dist
                })
            results.append(sorted(batch_hits, key=lambda x: -x["confidence"]))
        
        return results
    
    def save(self, save_dir):
        """
        Save all indexes and metadata to a directory.

        Args:
            save_dir: Directory to save the files.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Helper function to convert to CPU index if it's a GpuIndex
        def to_cpu(index):
            if hasattr(faiss, 'GpuIndex') and isinstance(index, faiss.GpuIndex):
                return faiss.index_gpu_to_cpu(index)
            return index

        # Save main indexes
        faiss.write_index(to_cpu(self.image_index), os.path.join(save_dir, "image_index.faiss"))
        faiss.write_index(to_cpu(self.text_index), os.path.join(save_dir, "text_index.faiss"))
        np.save(os.path.join(save_dir, "image_metadata.npy"), np.array(self.image_metadata, dtype=object))
        np.save(os.path.join(save_dir, "text_metadata.npy"), np.array(self.text_metadata, dtype=object))

        # Save few-shot index if it has data
        if self.fewshot_index.ntotal > 0:
            faiss.write_index(to_cpu(self.fewshot_index), os.path.join(save_dir, "fewshot_index.faiss"))
            np.save(os.path.join(save_dir, "fewshot_metadata.npy"), np.array(self.fewshot_metadata, dtype=object))

        # Save a config file with the mode
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump({'database_mode': self.database_mode}, f)

        print(f"[INFO] All indexes and metadata saved to {save_dir}.")

    def save_tag(self, save_dir='tag_index'):
        """
        Save tag index and metadata.
        
        Args:
            save_dir: Directory to save the tag files.
        """
        os.makedirs(save_dir, exist_ok=True)
        def to_cpu(index):
            if hasattr(faiss, 'GpuIndex') and isinstance(index, faiss.GpuIndex):
                return faiss.index_gpu_to_cpu(index)
            return index
        faiss.write_index(to_cpu(self.tag_index), os.path.join(save_dir, "tag_index.faiss"))
        np.save(os.path.join(save_dir, "tag_metadata.npy"), np.array(self.tag_metadata, dtype=object))
        print(f"[INFO] Tag index and metadata saved to {save_dir}.")
    
    def load(self, save_dir):
        """
        Load all indexes and metadata from a directory.

        Args:
            save_dir: Directory containing the saved files.
        """
        if not os.path.isdir(save_dir):
            print(f"[ERROR] Load directory not found: {save_dir}")
            return
            
        # Helper to load to GPU if available
        def to_gpu(cpu_index):
            if self.device == 'cuda' and hasattr(self, 'res'):
                return faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
            return cpu_index

        # Load main indexes and metadata
        try:
            image_index_path = os.path.join(save_dir, "image_index.faiss")
            if os.path.exists(image_index_path):
                self.image_index = to_gpu(faiss.read_index(image_index_path))
                self.image_metadata = np.load(os.path.join(save_dir, "image_metadata.npy"), allow_pickle=True).tolist()
            
            text_index_path = os.path.join(save_dir, "text_index.faiss")
            if os.path.exists(text_index_path):
                self.text_index = to_gpu(faiss.read_index(text_index_path))
                self.text_metadata = np.load(os.path.join(save_dir, "text_metadata.npy"), allow_pickle=True).tolist()
            
            # Load few-shot index and metadata if they exist
            fewshot_index_path = os.path.join(save_dir, "fewshot_index.faiss")
            if os.path.exists(fewshot_index_path):
                self.fewshot_index = to_gpu(faiss.read_index(fewshot_index_path))
                self.fewshot_metadata = np.load(os.path.join(save_dir, "fewshot_metadata.npy"), allow_pickle=True).tolist()

            # Load the config
            config_path = os.path.join(save_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.database_mode = config.get('database_mode')

            print(f"[INFO] Indexes and metadata loaded successfully from {save_dir}. Mode: {self.database_mode}")

        except Exception as e:
            print(f"[ERROR] Failed to load indexes from {save_dir}: {e}")

    def load_tag(self, save_dir='tag_index'):
        """
        Load tag index and metadata.

        Args:
            save_dir: Directory containing the tag files.
        """
        if not os.path.isdir(save_dir):
            print(f"[ERROR] Tag load directory not found: {save_dir}")
            return

        def to_gpu(cpu_index):
            if self.device.type == 'cuda' and hasattr(self, 'res'):
                return faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
            return cpu_index
            
        try:
            tag_index_path = os.path.join(save_dir, "tag_index.faiss")
            if os.path.exists(tag_index_path):
                self.tag_index = to_gpu(faiss.read_index(tag_index_path))
                self.tag_metadata = np.load(os.path.join(save_dir, "tag_metadata.npy"), allow_pickle=True).tolist()
                print(f"[INFO] Tag index loaded from {save_dir}.")
        except Exception as e:
            print(f"[ERROR] Failed to load tag index: {e}")

if __name__ == "__main__":
    # Example usage
    manager = FAISSManager(device='cuda', model_path='sentence-transformers/paraphrase-MiniLM-L6-v2', database_mode='nvidia_aic')
    # manager.build(mode='nvidia_aic', data_root='/root/demo_data/MealsretrivevalDatabase/batch_4_refined.json')
    # manager.save('/root/demo_data/PhysicalAI_Dataset/indexes')
    manager.load('/root/demo_data/PhysicalAI_Dataset/indexes')
    start_time = time()
    results = manager.search_fewshot("From this viewpoint, does the pallet <region0> appear on the left-hand side of the pallet <region1>?", top_k=5)
    end_time = time()
    print(f"Search completed in {end_time - start_time:.2f} seconds")
    for res in results:
        print(f"Query: {res['query']}, Score: {res['score']}")
        print(f"Explanation: {res['explanation']}")
        print(f"Visual Checks: {res['visual_checks']}")
        print(f"Spatial Instructions: {res['spatial_instructions']}")
        