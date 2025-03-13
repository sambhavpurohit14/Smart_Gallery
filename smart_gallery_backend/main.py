from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from image_db_util import ImageDBManager
import os
import uuid

app = FastAPI()

class InitializeDBRequest(BaseModel):
    user_name: str

class ImageRequest(BaseModel):
    image_path: str

class FolderRequest(BaseModel):
    folder_path: str

class ImageSearch(BaseModel):
    query: str

@app.post("/initialize_db")
async def initialize_db(request: InitializeDBRequest):
    try:
        user_id = str(uuid.uuid4())
        db_manager = ImageDBManager(user_id)
        return {"message": f"Initialized DB for user {request.user_name} with ID {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_image")
async def add_image(request: ImageRequest, user_id: str = Query(...)):
    try:
        db_manager = ImageDBManager(user_id)
        return db_manager.add_image(request.image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_images_from_folder")
async def add_images_from_folder(request: FolderRequest, user_id: str = Query(...)):
    try:
        db_manager = ImageDBManager(user_id)
        return db_manager.add_images_from_folder(request.folder_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_image")
async def search_image(request: ImageSearch, user_id: str = Query(...)):
    try:
        result = ImageSearch(user_id)
        return JSONResponse(content={
            "image_path": result,
            "query": request.query
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)













