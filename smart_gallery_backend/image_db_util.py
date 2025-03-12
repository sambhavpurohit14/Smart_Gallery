import chromadb
from tqdm import tqdm
import os
from clip_model import CLIPFeatureExtractor
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
import numpy as np
from typing import List

class CLIPEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path):
        self.feature_extractor = CLIPFeatureExtractor(model_path)
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            if isinstance(doc, str) and os.path.exists(doc):
                try:
                    # Extract features
                    embedding = self.feature_extractor.extract_image_features(doc)
                    
                    # Convert to list format
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.detach().cpu().numpy()
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.squeeze().tolist()
                    
                    if not embedding or len(embedding) == 0:  
                        raise ValueError(f"Empty embedding generated for {doc}")

                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error creating embedding for {doc}: {str(e)}")
                    embeddings.append([0.0] * 512)  # Fallback to zero embedding
        return embeddings

class ImageDBManager:
    def __init__(self, model_path, db_path):
        self.db_path = db_path
        self.embedding_function = CLIPEmbeddingFunction(model_path)
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="image_embeddings")

    def add_image(self, image_path):
        """Adds a single image to the database."""
        try:
            image_name = os.path.basename(image_path)

            # Check if image already exists
            existing = self.collection.get(ids=[image_name])
            if existing and existing.get("ids"):
                return {"status": "error", "message": f"Image '{image_name}' already exists"}

            # Compute embedding
            embedding = self.embedding_function([image_path])[0]

            # Convert embedding to a list format
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy().tolist()

            if not isinstance(embedding, list) or len(embedding) == 0:
                return {"status": "error", "message": f"Invalid embedding for '{image_name}'"}

            # Add image to database
            self.collection.add(
                documents=[image_path],
                embeddings=[embedding],
                ids=[image_name],
                metadatas=[{"filename": image_name, "path": image_path}]
            )
            return {"status": "success", "message": f"Added '{image_name}'"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def add_images_from_folder(self, folder_path):
        """Adds all images from a folder using add_image method."""
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
