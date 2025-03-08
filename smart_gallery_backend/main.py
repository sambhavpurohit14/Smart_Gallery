from fastapi import FastAPI
from fastapi.responses import FileResponse
import torch
from clip_model import CLIPModel, CLIPFeatureExtractor
from image_db_util import ImageDBManager
import os


app = FastAPI()
db_path = ""
model_path = "clip_model_epoch_12.pt"
db_manager = ImageDBManager(model_path, db_path)

# Load the model and feature extractor once
model = CLIPModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
feature_extractor = CLIPFeatureExtractor(device=torch.device("cpu"), model=model)

@app.post("/add_images_from_folder")
async def add_images(folder_path: str):
    return db_manager.add_images_from_folder(folder_path)

@app.post("/add_image")
async def add_image(image_path: str):
    return db_manager.add_single_image(image_path)

@app.post("/search_images")
async def search_images(query: str):
    similarities = []
    query_features = feature_extractor.extract_text_features(query)
    all_embeddings = db_manager.collection.get()
    image_embeddings = all_embeddings['embeddings']
    image_ids = all_embeddings['ids']
    img_embeddings_tensor = torch.tensor(image_embeddings)
    similarity_scores = torch.matmul(img_embeddings_tensor, query_features.T).squeeze()
    similarities = [(float(score), img_id) for score, img_id in zip(similarity_scores, image_ids)]
    similarities.sort(reverse=True)
    
    if similarities:
        top_score, top_image_id = similarities[0]
        image_metadata = db_manager.collection.get(ids=[top_image_id])
        if 'metadatas' in image_metadata and image_metadata['metadatas']:
            image_path = image_metadata['metadatas'][0].get('image_path')
            if image_path and os.path.exists(image_path):
                return FileResponse(image_path)
    
    return {"message": "No matching images found"}












