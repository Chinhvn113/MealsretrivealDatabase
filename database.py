import os
import time
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Tuple, Optional, Any
from datetime import datetime
import json
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from transformers import CLIPProcessor, CLIPModel


class MilvusMultimodalDB:
    """
    A Milvus database class for multimodal search capabilities,
    supporting text-to-text, image-to-text, text-to-image, and image-to-image search.
    """
    
    def __init__(
        self,
        embed_model: str = "jinaai/jina-embeddings-v2-base-en",
        tag_model: str = "microsoft/florence-2-base",
        ocr_model: str = "microsoft/trocr-base-printed",
        host: str = "localhost",
        port: str = "19530",
        db_name: str = "multimodal_db",
        text_collection_name: str = "text_collection",
        image_collection_name: str = "image_collection",
        vector_dim: int = 768,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
        index_params: Dict = None,
        cache_dir: str = None,
    ):
        """
        Initialize the Milvus multimodal database.
        
        Args:
            embed_model: Model name for embedding generation (JINA-CLIP-v2 by default)
            tag_model: Model name for image tagging
            ocr_model: Model name for OCR
            host: Milvus server host
            port: Milvus server port
            db_name: Milvus database name
            text_collection_name: Name for the text collection
            image_collection_name: Name for the image collection
            vector_dim: Dimension of embedding vectors
            index_type: Type of index for vectors
            metric_type: Distance metric type
            index_params: Additional index parameters
            cache_dir: Directory to cache embeddings to avoid recomputation
        """
        self.embed_model_name = embed_model
        self.tag_model_name = tag_model
        self.ocr_model_name = ocr_model
        self.host = host
        self.port = port
        self.db_name = db_name
        self.text_collection_name = text_collection_name
        self.image_collection_name = image_collection_name
        self.vector_dim = vector_dim
        self.index_type = index_type
        self.metric_type = metric_type
        self.cache_dir = cache_dir
        
        # Create cache directory if specified
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, "embeddings"), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, "tags"), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, "ocr"), exist_ok=True)
            
        if index_params is None:
            self.index_params = {"nlist": 1024}
        else:
            self.index_params = index_params
            
        # Flag to check if the database is built
        self.is_built = False
        
        # Initialize models as None, they will be loaded on demand
        self.embed_model = None
        self.embed_processor = None
        self.tag_model = None
        self.tag_processor = None
        self.ocr_model = None
        self.ocr_processor = None
        
        # Collections
        self.text_collection = None
        self.image_collection = None
        
        # Statistics
        self.stats = {
            "total_videos": 0,
            "total_frames": 0,
            "total_texts": 0,
            "processed_videos": []
        }
        
    def load(self):
        """Load the required models for embedding, tagging, and OCR."""
        # Load embedding model (JINA-CLIP-v2)
        try:
            from transformers import AutoModel, AutoProcessor
            print(f"Loading embedding model: {self.embed_model_name}")
            self.embed_model = AutoModel.from_pretrained(self.embed_model_name)
            self.embed_processor = AutoProcessor.from_pretrained(self.embed_model_name)
            print("Embedding model loaded successfully")
            
            # Load tag model
            print(f"Loading tag model: {self.tag_model_name}")
            # This is a placeholder - you would use the appropriate loading method
            # for your specific tag model
            import torch
            self.tag_model = CLIPModel.from_pretrained(self.tag_model_name)
            self.tag_processor = CLIPProcessor.from_pretrained(self.tag_model_name)
            print("Tag model loaded successfully")
            
            # Load OCR model
            print(f"Loading OCR model: {self.ocr_model_name}")
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.ocr_processor = TrOCRProcessor.from_pretrained(self.ocr_model_name)
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained(self.ocr_model_name)
            print("OCR model loaded successfully")
            
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
            
    def connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias="default", 
                host=self.host,
                port=self.port
            )
            print(f"Connected to Milvus server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Milvus: {str(e)}")
            return False
            
    def build(self, data_dir=None):
        """
        Build the Milvus database with appropriate collections and indexes.
        Optionally process a data directory containing video keyframes.
        
        Args:
            data_dir: Path to data directory containing video subdirectories with keyframes
                      Structure: data_dir/video_name/key_frame_XXX.webp
        
        Returns:
            bool: True if build successful, False otherwise
        """
        if not self.connect():
            print("Failed to connect to Milvus. Cannot build database.")
            return False
            
        try:
            # Create database if it doesn't exist
            if not utility.has_database(self.db_name):
                utility.create_database(self.db_name)
                print(f"Created database: {self.db_name}")
                
            # Text collection schema
            text_fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="channel", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
            ]
            
            text_schema = CollectionSchema(fields=text_fields, description="Text collection for multimodal search")
            
            # Image collection schema (updated with video_path and frame_number)
            image_fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="video_path", dtype=DataType.VARCHAR, max_length=512),  # Path to the video directory
                FieldSchema(name="frame_number", dtype=DataType.INT64),  # Frame number extracted from filename
                FieldSchema(name="tags", dtype=DataType.JSON),  # Store tags as JSON
                FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="channel", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
            ]
            
            image_schema = CollectionSchema(fields=image_fields, description="Image collection for multimodal search")
            
            # Create collections if they don't exist
            if utility.has_collection(self.text_collection_name):
                self.text_collection = Collection(self.text_collection_name)
                print(f"Using existing text collection: {self.text_collection_name}")
            else:
                self.text_collection = Collection(self.text_collection_name, schema=text_schema)
                print(f"Created text collection: {self.text_collection_name}")
                
                # Create index for text collection
                self.text_collection.create_index(
                    field_name="vector", 
                    index_params={
                        "index_type": self.index_type,
                        "metric_type": self.metric_type,
                        "params": self.index_params
                    }
                )
                print(f"Created index for text collection")
                
            if utility.has_collection(self.image_collection_name):
                self.image_collection = Collection(self.image_collection_name)
                print(f"Using existing image collection: {self.image_collection_name}")
            else:
                self.image_collection = Collection(self.image_collection_name, schema=image_schema)
                print(f"Created image collection: {self.image_collection_name}")
                
                # Create index for image collection
                self.image_collection.create_index(
                    field_name="vector", 
                    index_params={
                        "index_type": self.index_type,
                        "metric_type": self.metric_type,
                        "params": self.index_params
                    }
                )
                print(f"Created index for image collection")
                
            # Load collections
            self.text_collection.load()
            self.image_collection.load()
            
            self.is_built = True
            print("Database built successfully")
            
            # Process data directory if provided
            if data_dir and os.path.isdir(data_dir):
                self._process_data_directory(data_dir)
            
            return True
        except Exception as e:
            print(f"Error building database: {str(e)}")
            return False
            
    def _process_data_directory(self, data_dir):
        """
        Process a directory containing video subdirectories with keyframe images.
        
        Args:
            data_dir: Path to data directory
        """
        try:
            import re
            import glob
            
            # Ensure models are loaded
            if self.embed_model is None:
                print("Models not loaded. Loading now...")
                self.load()
                
            print(f"Processing data directory: {data_dir}")
            
            # Get list of video directories
            video_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            total_videos = len(video_dirs)
            
            for idx, video_name in enumerate(video_dirs):
                video_dir = os.path.join(data_dir, video_name)
                video_path = os.path.abspath(video_dir)
                
                print(f"Processing video {idx+1}/{total_videos}: {video_name}")
                
                # Find all keyframe images in this video directory
                keyframe_pattern = os.path.join(video_dir, "key_frame_*.webp")
                keyframe_files = glob.glob(keyframe_pattern)
                
                if not keyframe_files:
                    print(f"  No keyframes found for video: {video_name}")
                    continue
                    
                print(f"  Found {len(keyframe_files)} keyframes")
                
                # Process each keyframe
                batch_size = 100  # Process in batches to avoid memory issues
                for i in range(0, len(keyframe_files), batch_size):
                    batch_files = keyframe_files[i:i+batch_size]
                    
                    # Prepare batch data
                    image_paths = []
                    video_paths = []
                    frame_numbers = []
                    tags_list = []
                    dates_list = []
                    channels_list = []
                    vectors_list = []
                    
                    for keyframe_file in batch_files:
                        # Extract frame number from filename
                        frame_match = re.search(r'key_frame_(\d+)\.webp', os.path.basename(keyframe_file))
                        if frame_match:
                            frame_number = int(frame_match.group(1))
                        else:
                            frame_number = 0
                            
                        # Process image
                        try:
                            # Full path to image
                            image_path = os.path.abspath(keyframe_file)
                            
                            # Extract tags
                            tags = self.extract_tags(image_path)
                            
                            # Extract OCR data
                            ocr_data = self.extract_text_from_image(image_path)
                            date = ocr_data.get("date", "")
                            channel = ocr_data.get("channel", "")
                            
                            # Also add OCR text to text collection if meaningful text was found
                            ocr_text = ocr_data.get("text", "").strip()
                            if len(ocr_text) > 10:  # Only add if there's meaningful text
                                text_vector = self.encode_text(ocr_text)
                                if text_vector is not None:
                                    self.text_collection.insert([
                                        [ocr_text],     # text field
                                        [image_path],   # source field (using image path)
                                        [date],         # date field
                                        [channel],      # channel field
                                        [text_vector]   # vector field
                                    ])
                            
                            # Encode image
                            vector = self.encode_image(image_path)
                            if vector is None:
                                continue
                                
                            # Add to batch lists
                            image_paths.append(image_path)
                            video_paths.append(video_path)
                            frame_numbers.append(frame_number)
                            tags_list.append(tags)
                            dates_list.append(date)
                            channels_list.append(channel)
                            vectors_list.append(vector)
                            
                        except Exception as e:
                            print(f"  Error processing keyframe {keyframe_file}: {str(e)}")
                    
                    # Insert batch to collection if we have items
                    if image_paths:
                        try:
                            self.image_collection.insert([
                                image_paths,    # image_path field
                                video_paths,    # video_path field
                                frame_numbers,  # frame_number field
                                tags_list,      # tags field
                                dates_list,     # date field
                                channels_list,  # channel field
                                vectors_list    # vector field
                            ])
                            print(f"  Added batch of {len(image_paths)} keyframes")
                        except Exception as e:
                            print(f"  Error inserting batch: {str(e)}")
                
            print(f"Finished processing data directory: {data_dir}")
            return True
            
        except Exception as e:
            print(f"Error processing data directory: {str(e)}")
            return False
            
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using the embedding model.
        
        Args:
            text: Input text to encode
            
        Returns:
            numpy.ndarray: Text embedding vector
        """
        if self.embed_model is None:
            print("Embedding model not loaded. Loading now...")
            self.load()
            
        # Check cache first if cache directory is specified
        if self.cache_dir:
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_file = os.path.join(self.cache_dir, "embeddings", f"text_{text_hash}.npy")
            
            if os.path.exists(cache_file):
                try:
                    return np.load(cache_file)
                except Exception as e:
                    print(f"Error loading cached text embedding: {str(e)}")
            
        try:
            import torch
            
            # Process the text input
            inputs = self.embed_processor(text=text, return_tensors="pt")
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                
            # Extract text embeddings - adjust based on your specific model
            text_embeddings = outputs.text_embeds.cpu().numpy()
            
            # Normalize embeddings
            text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            
            # Cache the embedding if cache directory is specified
            if self.cache_dir:
                try:
                    np.save(cache_file, text_embeddings[0])
                except Exception as e:
                    print(f"Error caching text embedding: {str(e)}")
            
            return text_embeddings[0]  # Return the first embedding (single text input)
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            return None
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode image using the embedding model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy.ndarray: Image embedding vector
        """
        if self.embed_model is None:
            print("Embedding model not loaded. Loading now...")
            self.load()
            
        # Check cache first if cache directory is specified
        if self.cache_dir:
            import hashlib
            with open(image_path, "rb") as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
            cache_file = os.path.join(self.cache_dir, "embeddings", f"image_{image_hash}.npy")
            
            if os.path.exists(cache_file):
                try:
                    return np.load(cache_file)
                except Exception as e:
                    print(f"Error loading cached image embedding: {str(e)}")
            
        try:
            import torch
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process the image input
            inputs = self.embed_processor(images=image, return_tensors="pt")
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                
            # Extract image embeddings - adjust based on your specific model
            image_embeddings = outputs.image_embeds.cpu().numpy()
            
            # Normalize embeddings
            image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
            
            # Cache the embedding if cache directory is specified
            if self.cache_dir:
                try:
                    np.save(cache_file, image_embeddings[0])
                except Exception as e:
                    print(f"Error caching image embedding: {str(e)}")
            
            return image_embeddings[0]  # Return the first embedding (single image input)
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            return None
    
    def extract_tags(self, image_path: str) -> Dict[str, float]:
        """
        Extract tags from an image using the tag model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict[str, float]: Dictionary of tags and their confidence scores
        """
        if self.tag_model is None:
            print("Tag model not loaded. Loading now...")
            self.load()
            
        # Check cache first if cache directory is specified
        if self.cache_dir:
            import hashlib
            with open(image_path, "rb") as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
            cache_file = os.path.join(self.cache_dir, "tags", f"{image_hash}.json")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading cached tags: {str(e)}")
            
        try:
            import torch
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process the image
            inputs = self.tag_processor(images=image, return_tensors="pt")
            
            # Generate tag predictions
            with torch.no_grad():
                outputs = self.tag_model(**inputs)
                
            # This is a placeholder - you would need to adapt this based on your tag model's output format
            # For demonstration, we're creating a mock tags dictionary
            probs = torch.nn.functional.softmax(outputs.logits_per_image[0], dim=0)
            tags = {f"tag_{i}": float(prob) for i, prob in enumerate(probs[:10])}
            
            # Cache the tags if cache directory is specified
            if self.cache_dir:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(tags, f)
                except Exception as e:
                    print(f"Error caching tags: {str(e)}")
            
            return tags
        except Exception as e:
            print(f"Error extracting tags: {str(e)}")
            return {}
    
    def extract_text_from_image(self, image_path: str) -> Dict[str, str]:
        """
        Extract text from an image using the OCR model.
        Attempts to extract date and channel information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict[str, str]: Dictionary with 'text', 'date', and 'channel' keys
        """
        if self.ocr_model is None:
            print("OCR model not loaded. Loading now...")
            self.load()
            
        # Check cache first if cache directory is specified
        if self.cache_dir:
            import hashlib
            with open(image_path, "rb") as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
            cache_file = os.path.join(self.cache_dir, "ocr", f"{image_hash}.json")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading cached OCR data: {str(e)}")
            
        try:
            import torch
            import re
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process the image
            pixel_values = self.ocr_processor(image, return_tensors="pt").pixel_values
            
            # Generate OCR output
            generated_ids = self.ocr_model.generate(pixel_values)
            generated_text = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Try to extract date and channel
            date_pattern = r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2}'
            date_match = re.search(date_pattern, generated_text)
            date = date_match.group(0) if date_match else ""
            
            # This is a simple channel extraction - you might need a more sophisticated approach
            channel_patterns = ["CNN", "BBC", "FOX", "MSNBC", "Channel", "HTV 9", "VTV", "VTC", "VTV1", "VTV2", "VTV3", "VTV6", "HTV", "HTV7", "HTV8"]
            channel = ""
            for pattern in channel_patterns:
                if pattern in generated_text:
                    channel = pattern
                    break
            
            result = {
                "text": generated_text,
                "date": date,
                "channel": channel
            }
            
            # Cache the OCR data if cache directory is specified
            if self.cache_dir:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(result, f)
                except Exception as e:
                    print(f"Error caching OCR data: {str(e)}")
                    
            return result
        except Exception as e:
            print(f"Error extracting text from image: {str(e)}")
            return {"text": "", "date": "", "channel": ""}
            
    def add(self, item_type: str, data: Union[str, Dict[str, Any]]) -> bool:
        """
        Add an item to the database.
        
        Args:
            item_type: Type of item ('text' or 'image')
            data: Data to add
                For text: string content or dict with metadata
                For image: dict with 'path' key and optional metadata
                
        Returns:
            bool: True if added successfully, False otherwise
        """
        if not self.is_built:
            print("Database not built. Please call build() first.")
            return False
            
        try:
            if item_type == "text":
                # For text items
                if isinstance(data, str):
                    # Simple case - just text content
                    text_content = data
                    source = ""
                    date = datetime.now().strftime("%Y-%m-%d")
                    channel = ""
                elif isinstance(data, dict):
                    # Dict case with metadata
                    text_content = data.get("text", "")
                    source = data.get("source", "")
                    date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
                    channel = data.get("channel", "")
                else:
                    print("Invalid data format for text item")
                    return False
                    
                # Encode text
                vector = self.encode_text(text_content)
                if vector is None:
                    print("Failed to encode text")
                    return False
                    
                # Insert text data
                self.text_collection.insert([
                    [text_content],  # text field
                    [source],        # source field
                    [date],          # date field
                    [channel],       # channel field
                    [vector]         # vector field
                ])
                
                print(f"Added text item: {text_content[:50]}...")
                return True
                
            elif item_type == "image":
                if not isinstance(data, dict) or "path" not in data:
                    print("Image data must be a dictionary with 'path' key")
                    return False
                    
                image_path = data["path"]
                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    return False
                    
                # Extract video path and frame number if possible
                video_path = data.get("video_path", os.path.dirname(image_path))
                
                # Try to extract frame number from filename
                frame_number = 0
                import re
                frame_match = re.search(r'key_frame_(\d+)\.webp', os.path.basename(image_path))
                if frame_match:
                    frame_number = int(frame_match.group(1))
                else:
                    frame_number = data.get("frame_number", 0)
                    
                # Extract tags
                tags = data.get("tags", self.extract_tags(image_path))
                
                # Extract OCR information
                ocr_data = data.get("ocr_data", self.extract_text_from_image(image_path))
                date = ocr_data.get("date", datetime.now().strftime("%Y-%m-%d"))
                channel = ocr_data.get("channel", "")
                
                # Encode image
                vector = self.encode_image(image_path)
                if vector is None:
                    print("Failed to encode image")
                    return False
                    
                # Insert image data
                self.image_collection.insert([
                    [image_path],   # image_path field
                    [video_path],   # video_path field
                    [frame_number], # frame_number field
                    [tags],         # tags field
                    [date],         # date field
                    [channel],      # channel field
                    [vector]        # vector field
                ])
                
                print(f"Added image item: {image_path}")
                return True
            else:
                print(f"Unknown item type: {item_type}")
                return False
                
        except Exception as e:
            print(f"Error adding item: {str(e)}")
            return False
            
    def search(
        self, 
        query_type: str,
        query: Union[str, Dict[str, Any]],
        top_k: int = 10,
        filter_expr: str = None,
        video_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search the database based on query type.
        
        Args:
            query_type: Type of search ('text2text', 'text2image', 'image2text', 'image2image')
            query: Query text or dict with image path
            top_k: Number of results to return
            filter_expr: Milvus filter expression
            video_filter: Filter results to specific video path
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        if not self.is_built:
            print("Database not built. Please call build() first.")
            return []
            
        try:
            # Prepare query vector based on query type
            query_vector = None
            
            if query_type in ['text2text', 'text2image']:
                if isinstance(query, str):
                    query_vector = self.encode_text(query)
                else:
                    print("Text query must be a string")
                    return []
                    
            elif query_type in ['image2text', 'image2image']:
                if isinstance(query, dict) and 'path' in query:
                    image_path = query['path']
                    if not os.path.exists(image_path):
                        print(f"Image file not found: {image_path}")
                        return []
                    query_vector = self.encode_image(image_path)
                else:
                    print("Image query must be a dictionary with 'path' key")
                    return []
                    
            if query_vector is None:
                print("Failed to encode query")
                return []
                
            # Determine target collection based on query type
            if query_type in ['text2text', 'image2text']:
                target_collection = self.text_collection
                result_type = 'text'
            else:  # text2image, image2image
                target_collection = self.image_collection
                result_type = 'image'
                
            # Combine filters if needed
            if video_filter and result_type == 'image':
                if filter_expr:
                    filter_expr = f"({filter_expr}) && video_path == '{video_filter}'"
                else:
                    filter_expr = f"video_path == '{video_filter}'"
                
            # Perform search
            search_params = {"metric_type": self.metric_type, "params": {"nprobe": 10}}
            results = target_collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["*"]  # Return all fields
            )
            
            # Process results
            processed_results = []
            for hits in results:
                for hit in hits:
                    result_dict = {
                        "id": hit.id,
                        "distance": hit.distance,
                    }
                    
                    # Add fields based on result type
                    if result_type == 'text':
                        result_dict.update({
                            "text": hit.entity.get('text'),
                            "source": hit.entity.get('source'),
                            "date": hit.entity.get('date'),
                            "channel": hit.entity.get('channel')
                        })
                    else:  # image
                        result_dict.update({
                            "image_path": hit.entity.get('image_path'),
                            "video_path": hit.entity.get('video_path'),
                            "frame_number": hit.entity.get('frame_number'),
                            "tags": hit.entity.get('tags'),
                            "date": hit.entity.get('date'),
                            "channel": hit.entity.get('channel')
                        })
                        
                    processed_results.append(result_dict)
                    
            return processed_results
            
        except Exception as e:
            print(f"Error searching: {str(e)}")
            return []
            
    def close(self):
        """Close connections and resources."""
        try:
            if self.text_collection:
                self.text_collection.release()
            if self.image_collection:
                self.image_collection.release()
            connections.disconnect("default")
            print("Database connections closed")
        except Exception as e:
            print(f"Error closing database: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = MilvusMultimodalDB()
    
    # Load models
    db.load()
    
    # Build database with data directory
    data_dir = "/path/to/data"
    db.build(data_dir)  # This will process all videos and keyframes
    
    # Search examples
    
    # Text to image search
    text_results = db.search(
        query_type="text2image", 
        query="people talking in a news broadcast", 
        top_k=5
    )
    
    # Filter by specific video
    video_specific_results = db.search(
        query_type="text2image",
        query="breaking news", 
        top_k=5,
        video_filter="/path/to/data/video1"
    )
    
    # Image to image search (find similar frames)
    image_results = db.search(
        query_type="image2image", 
        query={"path": "/path/to/data/video1/key_frame_001.webp"}, 
        top_k=5
    )
    
    # Close database
    db.close()path": "/path/to/image.jpg"})
    
    # Perform searches
    text_results = db.search("text2text", "artificial intelligence", top_k=5)
    image_results = db.search("text2image", "artificial intelligence", top_k=5)
    
    # Close database
    db.close()