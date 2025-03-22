import os
import logging
from tqdm import tqdm
import chromadb
import torch
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import Dict, Any, Optional
from clip_model import CLIPFeatureExtractor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for ImageDBManager instances
_db_managers_cache: Dict[str, 'ImageDBManager'] = {}

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
                    logger.error(f"Error creating embedding for {doc}: {str(e)}")
                    embeddings.append([0.0] * 512) 
        return embeddings

class ImageDBManager:
    _instances = {}
    
    @classmethod
    def get_db_manager(cls, user_id, client=None):
        if user_id not in cls._instances:
            cls._instances[user_id] = ImageDBManager(user_id, client)
        return cls._instances[user_id]
        
    def __init__(self, user_id: str, chroma_client=None, db_path=None):
        """
        Initialize the ImageDBManager.
        
        Args:
            user_id: The unique identifier for the user.
            chroma_client: An existing ChromaDB client to use.
            db_path: Path where the ChromaDB files should be stored.
        """
        self.user_id = user_id
        self.embedding_function = CLIPEmbeddingFunction()
        
        # Use the provided client or create a new one with the provided path
        if chroma_client is not None:
            self.client = chroma_client
        elif db_path is not None:
            self.client = self.get_chroma_client(db_path)
        else:
            raise ValueError("Either chroma_client or db_path must be provided")
        
        # Get or create the collection for this user
        collection_name = f"image_embeddings_{user_id}"
        try:
            self.collection = self.client.get_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
            logger.info(f"Found existing collection for user {user_id}")
        except Exception:
            logger.info(f"Creating new collection for user {user_id}")
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )

    @classmethod
    def get_chroma_client(cls, db_path=None):
        """Get the ChromaDB client from the main application.
        
        Args:
            db_path: Path to store the ChromaDB files
        """
        try:
            from main import CHROMA_CLIENT
            if (CHROMA_CLIENT is not None and db_path is None):
                return CHROMA_CLIENT
        except ImportError:
            logger.warning("Could not import CHROMA_CLIENT from main")
        
        # If we have a db_path, always create a PersistentClient at that location
        if db_path:
            logger.info(f"Creating new PersistentClient at: {db_path}")
            # Create the directory if it doesn't exist
            os.makedirs(db_path, exist_ok=True)
            client = chromadb.PersistentClient(path=db_path)
            
            # Update the global client reference if possible
            try:
                import main
                main.CHROMA_CLIENT = client
            except ImportError:
                pass
                
            return client
        
        # If we get here, we couldn't get a client and don't have a path
        raise ValueError("No ChromaDB client available and no db_path provided")
    
    @classmethod
    def get_db_manager(cls, user_id: str, chroma_client=None) -> 'ImageDBManager':
        """Get or create an ImageDBManager instance for the given user_id.
        
        Args:
            user_id: The unique identifier for the user
            chroma_client: Optional ChromaDB client to use
            
        Returns:
            An ImageDBManager instance for the user
        """
        if user_id not in _db_managers_cache:
            if chroma_client is None:
                chroma_client = cls.get_chroma_client()
            _db_managers_cache[user_id] = ImageDBManager(user_id, chroma_client)
            logger.info(f"Created new ImageDBManager for user {user_id}")
        return _db_managers_cache[user_id]

    def add_image(self, image_path: str) -> Dict[str, Any]:
        """Add a single image to the database."""
        try:
            if not os.path.exists(image_path):
                return {"status": "error", "message": f"Image path does not exist: {image_path}"}
                
            image_name = os.path.basename(image_path)
            
            # Check if image already exists
            try:
                existing = self.collection.get(ids=[image_name])
                if existing and existing.get("ids"):
                    return {"status": "error", "message": f"Image '{image_name}' already exists"}
            except Exception as e:
                logger.debug(f"Error checking if image exists: {str(e)}")
                # Continue with adding the image
            
            # Add image to database
            self.collection.add(
                documents=[image_path],
                ids=[image_name],
                metadatas=[{"filename": image_name, "path": image_path}]
            )
            return {"status": "success", "message": f"Added '{image_name}'"}
        except Exception as e:
            logger.error(f"Error adding image {image_path}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def add_images_from_folder(self, folder_path: str, use_as_db_path: bool = True) -> Dict[str, Any]:
        """Add all images from a folder to the database.
        
        Args:
            folder_path: Path to the folder containing images
            use_as_db_path: If True, store the ChromaDB files in a subdirectory of this folder
        """
        if not os.path.isdir(folder_path):
            return {"status": "error", "message": "Invalid folder path"}

        # If requested, create a database directory within the folder
        db_path = None
        if use_as_db_path:
            db_path = os.path.join(folder_path, ".chromadb")
            
            try:
                # Update the client to a PersistentClient at the new location
                self.client = self.get_chroma_client(db_path)
                
                # We need to recreate/reconnect to the collection with the new client
                collection_name = f"image_embeddings_{self.user_id}"
                try:
                    self.collection = self.client.get_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    logger.info(f"Using existing collection for user {self.user_id}")
                except Exception:
                    logger.info(f"Creating new collection for user {self.user_id}")
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                
                logger.info(f"ChromaDB initialized at {db_path}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB at {db_path}: {str(e)}")
                return {"status": "error", "message": f"Database initialization failed: {str(e)}"}

        # Find all image files
        image_files = [
            os.path.join(folder_path, image_name)
            for image_name in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, image_name)) and 
            image_name.lower().endswith((".png", ".jpg", ".jpeg"))
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
            "total_images": len(image_files),
            "errors": errors,
            "db_path": db_path
        }


