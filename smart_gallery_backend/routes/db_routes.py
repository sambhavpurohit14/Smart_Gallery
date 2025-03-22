from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
import logging
import os
import chromadb
from image_db_util import ImageDBManager
import db_client  

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class InitializeDBRequest(BaseModel):
    user_name: str
    db_path: str  # Path where Chroma will store database files

@router.post("/initialize", summary="Initialize a database for a user")
async def initialize_db(request: InitializeDBRequest):
    """Initialize a database for a user and return a unique user_id."""
    try:
        user_id = str(uuid.uuid4())
        logger.info(f"Initializing database for user {request.user_name} with ID {user_id}")
        
        # Create the directory if it doesn't exist
        os.makedirs(request.db_path, exist_ok=True)
        
        # Initialize Chroma PersistentClient with the specified path
        db_client.CHROMA_CLIENT = chromadb.PersistentClient(path=request.db_path)
        logger.info(f"ChromaDB PersistentClient initialized at path: {request.db_path}")
        
        # Use the class method to get or create DB manager
        db_manager = ImageDBManager.get_db_manager(user_id, db_client.CHROMA_CLIENT)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Database initialized for user {request.user_name}",
                "user_id": user_id,
                "db_path": request.db_path
            }
        )
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to initialize database: {str(e)}"}
        )