import torch
import chromadb
from tqdm import tqdm
import os
import traceback
from clip_model import CLIPModel, CLIPFeatureExtractor


class ImageDBManager:
    def __init__(self, model_path, db_path):
        self.db_path = db_path
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="image_embeddings")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model
        self.model = CLIPModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.feature_extractor = CLIPFeatureExtractor(self.device, self.model)

    def add_images_from_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            return {"status": "error", "message": "Invalid folder path"}

        image_files = [
            os.path.join(folder_path, image_name)
            for image_name in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, image_name)) and image_name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not image_files:
            return {"status": "error", "message": "No valid images found"}

        errors = []
        added_images = 0

        for image_path in tqdm(image_files, desc="Creating Image Embeddings"):
            try:
                image_name = os.path.basename(image_path)

                # Check for existing image
                existing = self.collection.get(ids=[image_name])
                if existing and existing['ids']:
                    print(f"Skipping {image_name}, already exists.")
                    continue

                image_features = self.feature_extractor.extract_image_features(image_path)

                self.collection.add(
                    ids=[image_name],
                    embeddings=[image_features.tolist()],
                    metadatas=[{"filename": image_name, "path": image_path}]
                )
                added_images += 1
            except Exception as e:
                errors.append({"image": image_path, "error": str(e)})
                print(f"Error processing {image_path}: {traceback.format_exc()}")

        return {
            "status": "success" if added_images > 0 else "error",
            "added_images": added_images,
            "errors": errors
        }

    def add_image(self, image_path):
        try:
            image_name = os.path.basename(image_path)

            # Check for duplicate
            existing = self.collection.get(ids=[image_name])
            if existing and existing['ids']:
                print(f"Skipping {image_name}, already exists.")
                return {"status": "error", "message": "Image already exists"}

            image_features = self.feature_extractor.extract_image_features(image_path)

            self.collection.add(
                ids=[image_name],
                embeddings=[image_features.tolist()],
                metadatas=[{"filename": image_name, "path": image_path}]
            )
            return {"status": "success", "message": f"Added {image_name}"}
        except Exception as e:
            print(f"Error processing {image_path}: {traceback.format_exc()}")
            return {"status": "error", "message": str(e)}





