from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
import logging
from image_db_util import ImageDBManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class InitializeDBRequest(BaseModel):
    user_name: str

@router.post("/initialize", summary="Initialize a database for a user")
async def initialize_db(request: InitializeDBRequest):
    """Initialize a database for a user and return a unique user_id."""
    try:
        user_id = str(uuid.uuid4())
        logger.info(f"Initializing database for user {request.user_name} with ID {user_id}")
        
        # Use the class method to get or create DB manager
        db_manager = ImageDBManager.get_db_manager(user_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Database initialized for user {request.user_name}",
                "user_id": user_id
            }
        )
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to initialize database: {str(e)}"}
        )