import torch
from fastapi.responses import FileResponse
from fastapi import HTTPException
import os
from clip_model import CLIPModel, CLIPFeatureExtractor
import chromadb
import numpy as np

class ImageSearcher:
    def __init__(self, model_path, db_path):
        self.db_path = db_path
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_collection(name="image_embeddings")
        self.device = torch.device("cpu")
        self.model = CLIPModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.feature_extractor = CLIPFeatureExtractor(model=self.model)

    def search_image(self, query: str):
        try:
            # Extract query features
            query_features = self.feature_extractor.extract_text_features(query)
            
            # Retrieve all image embeddings from the database
            all_embeddings = self.collection.get()
            print("Database content:", all_embeddings)
            image_embeddings = all_embeddings.get('embeddings', [])
            image_ids = all_embeddings.get('ids', [])
            metadata_list = all_embeddings.get('metadatas', [])
            
            if not image_embeddings or not image_ids:
                return {"message": "No images in the database."}
            
            # Convert embeddings to tensor
            img_embeddings_tensor = torch.tensor([np.array(embedding) for embedding in image_embeddings], dtype=torch.float32)
            query_features_tensor = torch.tensor(query_features, dtype=torch.float32)
            
            # Compute similarity scores
            similarity_scores = torch.matmul(img_embeddings_tensor, query_features_tensor.T).squeeze()
            similarities = [(float(score), img_id, metadata) for score, img_id, metadata in zip(similarity_scores, image_ids, metadata_list)]
            similarities.sort(reverse=True)
            
            # Retrieve the top image result
            if similarities:
                top_score, top_image_id, top_image_metadata = similarities[0]
                image_path = top_image_metadata.get('path')
                
                if image_path and os.path.exists(image_path):
                    return FileResponse(image_path)
                
            return {"message": "No matching images found"}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")
