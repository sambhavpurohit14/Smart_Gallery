import os
import logging
import torch
from typing import Dict, Any, List
from clip_model import CLIPFeatureExtractor
from image_db_util import ImageDBManager

logger = logging.getLogger(__name__)

class ImageSearcher:
    def __init__(self, user_id: str):
        """Initialize the ImageSearcher with a user ID.
        
        Args:
            user_id: The unique identifier for the user.
        """
        try:
            # Use the class method to get the cached ImageDBManager
            self.user_id = user_id
            self.db_manager = ImageDBManager.get_db_manager(user_id)
            self.collection = self.db_manager.collection
            self.feature_extractor = CLIPFeatureExtractor()
            logger.info(f"Initialized ImageSearcher for user {user_id}")
        except Exception as e:
            logger.error(f"Error initializing ImageSearcher for user {user_id}: {str(e)}")
            raise ValueError(f"Could not initialize search for user {user_id}: {str(e)}")

    def search_images(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for images based on a text query."""
        try:
            # Get text embeddings from the query
            text_embedding = self.feature_extractor.extract_text_features(query)
            
            # Query the collection
            query_embedding_list = text_embedding.tolist() if isinstance(text_embedding, torch.Tensor) else text_embedding
            
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=n_results
            )
            
            if not results or not results.get("documents") or not results["documents"][0]:
                return {"status": "error", "message": "No results found", "images": []}
                
            # Format the results
            images = []
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] and i < len(results["metadatas"][0]) else {}
                images.append({
                    "path": doc,
                    "filename": metadata.get("filename", os.path.basename(doc)),
                    "metadata": metadata,
                    "distance": results["distances"][0][i] if "distances" in results and results["distances"] and i < len(results["distances"][0]) else None
                })
                
            return {
                "status": "success",
                "query": query,
                "images": images,
                "count": len(images)
            }
        except Exception as e:
            logger.error(f"Error searching for images: {str(e)}")
            return {"status": "error", "message": str(e), "images": []}
