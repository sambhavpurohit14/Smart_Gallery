import chromadb
from tqdm import tqdm
import os
import logging
from clip_model import CLIPFeatureExtractor
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
import numpy as np
from google.cloud import storage

# Setup logging
logging.basicConfig(level=logging.INFO)

# GCS Setup 
GCS_BUCKET_NAME = "your-gcs-bucket-name"
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

class CLIPEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.feature_extractor = CLIPFeatureExtractor()
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            if isinstance(doc, str) and os.path.exists(doc):
                try:
                    # Extract features using CLIP
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
                    logging.error(f"Error creating embedding for {doc}: {str(e)}")
                    embeddings.append([0.0] * 512)  # Fallback embedding
        return embeddings

class ImageDBManager:
    def __init__(self, user_id):
        self.user_id = user_id
        self.embedding_function = CLIPEmbeddingFunction()
        
        self.chroma_client = chromadb.PersistentClient(
            path="gs://your-chroma-gcs-bucket"
        )
        self.collection = self.chroma_client.create_collection(
            name=f"image_embeddings_of_{user_id}",
            embedding_function=self.embedding_function
        )

    def upload_to_gcs(self, local_path):
        """Uploads an image to Google Cloud Storage and returns the GCS URL."""
        try:
            blob_name = f"images/{self.user_id}/{os.path.basename(local_path)}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            gcs_url = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
            return gcs_url
        except Exception as e:
            logging.error(f"Failed to upload {local_path} to GCS: {e}")
            return None

    def add_image(self, image_path):
        """Adds a single image to the database with GCS storage."""
        try:
            image_name = os.path.basename(image_path)

            # Check if image already exists in DB
            existing = self.collection.get(ids=[image_name])
            if existing and existing.get("ids"):
                return {"status": "error", "message": f"Image '{image_name}' already exists"}

            # Upload to GCS
            gcs_url = self.upload_to_gcs(image_path)
            if not gcs_url:
                return {"status": "error", "message": f"Failed to upload {image_name} to GCS"}

            # Add image metadata and embedding
            self.collection.add(
                documents=[gcs_url],  # Store GCS URL instead of local path
                ids=[image_name],
                metadatas=[{"filename": image_name, "gcs_url": gcs_url}]
            )
            return {"status": "success", "message": f"Added '{image_name}' to DB"}
        except Exception as e:
            logging.error(f"Error adding image {image_path}: {e}")
            return {"status": "error", "message": str(e)}

    def add_images_from_folder(self, folder_path):
        """Adds all images from a folder."""
        if not os.path.isdir(folder_path):
            return {"status": "error", "message": "Invalid folder path"}

        image_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))
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
