import chromadb
from tqdm import tqdm
import os
from clip_model import CLIPFeatureExtractor
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
import numpy as np

# Initialize the Chroma HTTTP once
CHROMA_CLIENT = chromadb.HttpClient(host="35.238.154.187", port =8000)

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
                    print(f"Error creating embedding for {doc}: {str(e)}")
                    embeddings.append([0.0] * 512) 
        return embeddings

class ImageDBManager:
    def __init__(self, user_id):
        self.embedding_function = CLIPEmbeddingFunction()
        self.collection = CHROMA_CLIENT.get_or_create_collection(
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
