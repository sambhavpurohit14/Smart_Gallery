import os
import logging
from tqdm import tqdm
from clip_model import CLIPFeatureExtractor
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class CLIPEmbeddingFunction(EmbeddingFunction):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CLIPEmbeddingFunction, cls).__new__(cls)
            cls._instance.feature_extractor = CLIPFeatureExtractor()
        return cls._instance
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            if isinstance(doc, str) and os.path.exists(doc):
                try:
                    embedding = self.feature_extractor.extract_image_features(doc)
                    if isinstance(embedding, torch.Tensor):
                        embeddings.append(embedding.detach().cpu().tolist())
                    else:
                        raise ValueError(f"Expected torch.Tensor but got {type(embedding)}")
                except Exception as e:
                    logger.error(f"Error creating embedding for {doc}: {str(e)}")
                    embeddings.append([0.0] * 512) 
        return embeddings

class ImageDBManager:
    _instances: Dict[str, 'ImageDBManager'] = {} #Maps user_id to ImageDBManager instance
    _embedding_function = CLIPEmbeddingFunction()  # Single shared instance
    
    # Return the singleton instance for the given user_id
    @classmethod
    def get_instance(cls, user_id: str) -> 'ImageDBManager':
        """Get or create an ImageDBManager instance for the given user_id."""
        if user_id not in cls._instances:
            from main import CHROMA_CLIENT
            cls._instances[user_id] = cls(user_id, CHROMA_CLIENT)
        return cls._instances[user_id]

    def __init__(self, user_id: str, chroma_client):
        """Initialize ImageDBManager with user_id and ChromaDB client."""
        self.user_id = user_id
        self.embedding_function = self._embedding_function
        self.client = chroma_client
        self.collection = self.client.get_or_create_collection(
            name=f"image_embeddings_{user_id}",
            embedding_function=self.embedding_function
        )

    def add_image(self, image_path):
        try:
            image_name = os.path.basename(image_path)
            existing = self.collection.get(ids=[image_name])
            if existing and existing.get("ids"):
                return {"status": "error", "message": f"Image '{image_name}' already exists"}

            self.collection.add(
                documents=[image_path],
                ids=[image_name],
                metadatas=[{"filename": image_name, "path": image_path}]
            )
            return {"status": "success", "message": f"Added '{image_name}'"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def add_images_from_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            return {"status": "error", "message": "Invalid folder path"}

        image_files = [
            os.path.join(folder_path, image_name)
            for image_name in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, image_name)) and image_name.lower().endswith((".png", ".jpg", ".jpeg"))
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
            "errors": errors
        }
