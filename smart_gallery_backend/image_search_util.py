from fastapi import HTTPException
import os
import logging
from clip_model import CLIPFeatureExtractor
from image_db_util import ImageDBManager

logger = logging.getLogger(__name__)

class ImageSearcher:
    def __init__(self, user_id: str):
        """Initialize ImageSearcher with user_id."""
        try:
            # Get cached ImageDBManager instance
            self.db_manager = ImageDBManager.get_instance(user_id)
            self.collection = self.db_manager.collection
            self.feature_extractor = CLIPFeatureExtractor()
        except Exception as e:
            logger.error(f"Failed to initialize ImageSearcher: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def search_image(self, query: str):
        """Searches for images based on a query."""
        query_embeddings = self.feature_extractor.extract_text_features(query)
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
