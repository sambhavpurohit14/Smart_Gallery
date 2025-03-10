from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import torch
from clip_model import CLIPModel, CLIPFeatureExtractor
from image_db_util import ImageDBManager
import os
import uvicorn


app = FastAPI()


model_path = "smart_gallery_backend\clip_model.py"
db_path = "smart_gallery_backend"

# Initialize ImageDBManager
db_manager = ImageDBManager(model_path, db_path)

# Load the CLIP model and feature extractor once
device = torch.device("cpu")
model = CLIPModel()
model.load_state_dict(torch.load(model_path, map_location=device))
feature_extractor = CLIPFeatureExtractor(device=device, model=model)

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
async def search_images(query: str):
    try:
        # Extract query features
        query_features = feature_extractor.extract_text_features(query)
        
        # Retrieve all image embeddings from the database
        all_embeddings = db_manager.collection.get()
        image_embeddings = all_embeddings.get('embeddings', [])
        image_ids = all_embeddings.get('ids', [])
        
        if not image_embeddings or not image_ids:
            return {"message": "No images in the database."}
        
        # Convert to tensor safely
        img_embeddings_tensor = torch.tensor(image_embeddings, dtype=torch.float32)

        # Compute similarity scores
        similarity_scores = torch.matmul(img_embeddings_tensor, query_features.T).squeeze()
        similarities = [(float(score), img_id) for score, img_id in zip(similarity_scores, image_ids)]
        similarities.sort(reverse=True)

        # Return the top image result
        if similarities:
            top_score, top_image_id = similarities[0]
            image_metadata = db_manager.collection.get(ids=[top_image_id])
            if image_metadata and 'metadatas' in image_metadata and image_metadata['metadatas']:
                image_path = image_metadata['metadatas'][0].get('path')
                if image_path and os.path.exists(image_path):
                    return FileResponse(image_path)
        
        return {"message": "No matching images found"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing search: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)













