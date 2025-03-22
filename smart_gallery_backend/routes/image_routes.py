from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from image_db_util import ImageDBManager
from image_search_util import ImageSearcher

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class ImageRequest(BaseModel):
    image_path: str

class FolderRequest(BaseModel):
    folder_path: str

class ImageSearch(BaseModel):
    query: str
    n_results: int = 5

@router.post("/add", summary="Add a single image to the database")
async def add_image(request: ImageRequest, user_id: str = Query(...)):
    """Add a single image to the database."""
    try:
        logger.info(f"Adding image {request.image_path} for user {user_id}")
        db_manager = ImageDBManager.get_db_manager(user_id)
        result = db_manager.add_image(request.image_path)
        return JSONResponse(
            status_code=200 if result["status"] == "success" else 400,
            content=result
        )
    except Exception as e:
        logger.error(f"Error adding image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to add image: {str(e)}"}
        )

@router.post("/add_folder", summary="Add all images from a folder to the database")
async def add_images_from_folder(request: FolderRequest, user_id: str = Query(...)):
    """Add all images from a folder to the database."""
    try:
        logger.info(f"Adding images from folder {request.folder_path} for user {user_id}")
        db_manager = ImageDBManager.get_db_manager(user_id)
        result = db_manager.add_images_from_folder(request.folder_path)
        return JSONResponse(
            status_code=200 if result["status"] == "success" else 400,
            content=result
        )
    except Exception as e:
        logger.error(f"Error adding images from folder: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": f"Failed to add images from folder: {str(e)}"}
        )

@router.post("/search", summary="Search for images based on a text query")
async def search_image(request: ImageSearch, user_id: str = Query(...)):
    """Search for images based on a text query."""
    try:
        logger.info(f"Searching for images with query '{request.query}' for user {user_id}")
        # Create ImageSearcher and reuse the cached DB manager
        searcher = ImageSearcher(user_id)
        results = searcher.search_images(request.query, n_results=request.n_results)
        
        return JSONResponse(
            status_code=200 if results.get("status") == "success" else 404,
            content=results
        )
    except Exception as e:
        logger.error(f"Error searching for images: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to search for images: {str(e)}", "images": []}
        )