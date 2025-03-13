from fastapi import HTTPException
import chromadb
from clip_model import CLIPFeatureExtractor
import os
from image_db_util import ImageDBManager

class ImageSearcher:
    def __init__(self, user_id: str):
        self.chroma_client = ImageDBManager(user_id)
        self.collection = self.get_collection(f"image_embeddings_{user_id}")

    def search_image(self, query: str):
        """Searches for images based on a query."""
        feature_extractor = CLIPFeatureExtractor(model_path="smart_gallery_backend/clip_model_epoch_12.pt")
        query_embeddings = feature_extractor.extract_text_features(query)
        try:
            results = self.collection.query(query_embeddings=query_embeddings)
            print("Query results:", results)
            
            if not results or 'documents' not in results or not results['documents']:
                return {"message": "No results found"}

            # Retrieve the top image result
            top_result = results['documents'][0][1]
            image_path = top_result

            if image_path and os.path.exists(image_path):
                return {"image_path": image_path}
            else:
                return {"message": "Image path not found"}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")
