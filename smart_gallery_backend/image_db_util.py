import os
import logging
from tqdm import tqdm
import chromadb
import torch
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import Dict, Any, Optional
from clip_model import CLIPFeatureExtractor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for ImageDBManager instances
_db_managers_cache: Dict[str, 'ImageDBManager'] = {}

class CLIPEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.feature_extractor = CLIPFeatureExtractor()
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            if isinstance(doc, str) and os.path.exists(doc):
                try:
                    embedding = self.feature_extractor.extract_image_features(doc)
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.detach().cpu().numpy()
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.squeeze().tolist()
                    if isinstance(embedding, list) and all(isinstance(i, float) for i in embedding):
                        embeddings.append(embedding)
                    else:
                        raise ValueError(f"Invalid embedding format for {doc}")
                except Exception as e:
                    logger.error(f"Error creating embedding for {doc}: {str(e)}")
                    embeddings.append([0.0] * 512) 
        return embeddings

class ImageDBManager:
    @classmethod
    def get_chroma_client(cls):
        """Get the ChromaDB client from the main application."""
        try:
            from main import CHROMA_CLIENT
            return CHROMA_CLIENT
        except ImportError:
            logger.warning("Could not import CHROMA_CLIENT from main, creating a new client")
            return chromadb.HttpClient(host="34.123.164.56", port=8000) # fallback to reinitializing client
    
    @classmethod
    def get_db_manager(cls, user_id: str, chroma_client=None) -> 'ImageDBManager':
        """Get or create an ImageDBManager instance for the given user_id.
        
        Args:
            user_id: The unique identifier for the user
            chroma_client: Optional ChromaDB client to use
            
        Returns:
            An ImageDBManager instance for the user
        """
        if user_id not in _db_managers_cache:
            if chroma_client is None:
                chroma_client = cls.get_chroma_client()
            _db_managers_cache[user_id] = ImageDBManager(user_id, chroma_client)
            logger.info(f"Created new ImageDBManager for user {user_id}")
        return _db_managers_cache[user_id]

    def __init__(self, user_id: str, chroma_client=None):
        """
        Initialize the ImageDBManager.
        
        Args:
            user_id: The unique identifier for the user.
            chroma_client: An existing ChromaDB client to use. If None, a new one is created.
        """
        self.user_id = user_id
        self.embedding_function = CLIPEmbeddingFunction()
        
        # Use the provided client or create a new one
        self.client = chroma_client or self.get_chroma_client()
        
        # Get or create the collection for this user
        collection_name = f"image_embeddings_{user_id}"
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Found existing collection for user {user_id}")
        except Exception:
            logger.info(f"Creating new collection for user {user_id}")
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )

    def add_image(self, image_path: str) -> Dict[str, Any]:
        """Add a single image to the database."""
        try:
            if not os.path.exists(image_path):
                return {"status": "error", "message": f"Image path does not exist: {image_path}"}
                
            image_name = os.path.basename(image_path)
            
            # Check if image already exists
            try:
                existing = self.collection.get(ids=[image_name])
                if existing and existing.get("ids"):
                    return {"status": "error", "message": f"Image '{image_name}' already exists"}
            except Exception as e:
                logger.debug(f"Error checking if image exists: {str(e)}")
                # Continue with adding the image
            
            # Add image to database
            self.collection.add(
                documents=[image_path],
                ids=[image_name],
                metadatas=[{"filename": image_name, "path": image_path}]
            )
            return {"status": "success", "message": f"Added '{image_name}'"}
        except Exception as e:
            logger.error(f"Error adding image {image_path}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def add_images_from_folder(self, folder_path: str) -> Dict[str, Any]:
        """Add all images from a folder to the database."""
        if not os.path.isdir(folder_path):
            return {"status": "error", "message": "Invalid folder path"}

        image_files = [
            os.path.join(folder_path, image_name)
            for image_name in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, image_name)) and 
            image_name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            return {"status": "error", "message": "No valid images found"}

        errors = []
        added_images = 0

        for image_path in tqdm(image_files, desc="Adding images to database"):
            result = self.add_image(image_path)
            if result["status"] == "success":
                added_images += 1
            else:
                errors.append(result["message"])

        return {
            "status": "success" if added_images > 0 else "error",
            "added_images": added_images,
            "total_images": len(image_files),
            "errors": errors
        }


