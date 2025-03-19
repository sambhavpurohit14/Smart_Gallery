from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from image_db_util import ImageDBManager
import uuid

router = APIRouter(prefix="/db", tags=["database"])

class InitializeDBRequest(BaseModel):
    user_name: str

@router.post("/initialize")
async def initialize_db(request: InitializeDBRequest):
    try:
        user_id = str(uuid.uuid4())
        db_manager = ImageDBManager(user_id)
        return {"message": f"Initialized DB for user {request.user_name} with ID {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))