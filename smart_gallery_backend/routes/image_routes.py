from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from image_db_util import ImageDBManager
from image_search_util import ImageSearcher

router = APIRouter(prefix="/images", tags=["images"])

class ImageRequest(BaseModel):
    image_path: str

class FolderRequest(BaseModel):
    folder_path: str

class ImageSearch(BaseModel):
    query: str

@router.post("/add")
async def add_image(request: ImageRequest, user_id: str = Query(...)):
    try:
        db_manager = ImageDBManager(user_id)
        return db_manager.add_image(request.image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-folder")
async def add_images_from_folder(request: FolderRequest, user_id: str = Query(...)):
    try:
        db_manager = ImageDBManager(user_id)
        return db_manager.add_images_from_folder(request.folder_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_image(request: ImageSearch, user_id: str = Query(...)):
    try:
        image_searcher = ImageSearcher(user_id)
        result = image_searcher.search_image(request.query)
        return JSONResponse(content={
            "image_path": result,
            "query": request.query
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))