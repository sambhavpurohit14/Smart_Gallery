from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from image_db_util import ImageDBManager
from image_search_util import ImageSearcher
import os
import uvicorn


app = FastAPI()


model_path = "smart_gallery_backend\clip_model_epoch_12.pt"
db_path = "test_image_embeddings"

# Initialize ImageDBManager
db_manager = ImageDBManager(model_path, db_path)

# Initialize ImageSearcher
image_searcher = ImageSearcher(model_path, db_path)

class SearchQuery(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Smart Gallery API Running"}

@app.post("/add_images_from_folder")
async def add_images(folder_path: str):
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail="Folder path does not exist")
    
    return db_manager.add_images_from_folder(folder_path)

@app.post("/add_image")
async def add_image(image_path: str):
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="Image file does not exist")
    
    return db_manager.add_image(image_path)

@app.post("/search_images")
async def search_images(search_query: SearchQuery):
    return image_searcher.search_images(search_query.query)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  
    uvicorn.run(app, host="127.0.0.1", port=port)













