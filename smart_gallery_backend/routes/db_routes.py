from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from image_db_util import ImageDBManager
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/db", tags=["database"])

class InitializeDBRequest(BaseModel):
    user_name: str

@router.post("/initialize")
async def initialize_db(request: InitializeDBRequest):
    try:
        user_id = str(uuid.uuid4())
        # Get cached ImageDBManager instance
        db_manager = ImageDBManager.get_instance(user_id)
        return {
            "status": "success",
            "message": f"Initialized DB for user {request.user_name}",
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise HTTPException(status_code=500, detail=str(e))