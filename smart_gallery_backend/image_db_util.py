import torch
import chromadb
from tqdm import tqdm
import os
from clip_model import CLIPFeatureExtractor

class ImageDBManager:
    def __init__(self, model_path, db_path):
        self.db_path = db_path
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="image_embeddings",
            embedding_function=None
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.feature_extractor = CLIPFeatureExtractor(model_path)

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

        for image_path in tqdm(image_files, desc="Creating Image Embeddings"):
            try:
                image_name = os.path.basename(image_path)
                existing = self.collection.get(ids=[image_name])

                if existing and "ids" in existing and existing["ids"]:
                    print(f"Skipping {image_name}, already exists in database.")
                    continue

                print(f"Processing image: {image_name}")
                image_features = self.feature_extractor.extract_image_features(image_path)
                print(f"Raw embedding shape: {image_features.shape}")
                image_features = image_features.squeeze().tolist()
                print(f"Processed embedding for {image_name}: {image_features[:5]}")

                self.collection.add(
                    ids=[image_name],
                    embeddings=[image_features],  
                    metadatas=[{"filename": image_name, "path": image_path}]
                )
                print(f"Successfully added {image_name} to database")
                added_images += 1
            except Exception as e:
                errors.append({"image": image_path, "error": str(e)})
                print(f"Error processing {image_name}: {e}")

        return {
            "status": "success" if added_images > 0 else "error",
            "added_images": added_images,
            "errors": errors
        }

    def add_image(self, image_path):
        try:
            image_name = os.path.basename(image_path)
            existing = self.collection.get(ids=[image_name])
            if existing and "ids" in existing and existing["ids"]:
                return {"status": "error", "message": "Image already exists"}

            print(f"Processing single image: {image_name}")
            image_features = self.feature_extractor.extract_image_features(image_path)
            print(f"Raw embedding shape: {image_features.shape}")
            image_features = image_features.squeeze().tolist()
            print(f"Processed embedding for {image_name}: {image_features[:5]}")

            self.collection.add(
                ids=[image_name],
                embeddings=[image_features],  
                metadatas=[{"filename": image_name, "path": image_path}]
            )
            print(f"Successfully added {image_name} to database")
            return {"status": "success", "message": f"Added {image_name}"}
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            return {"status": "error", "message": str(e)}
