import os
from image_db_util import ImageDBManager  

# Set paths
MODEL_PATH = "smart_gallery_backend\clip_model_epoch_12.pt"  
DB_PATH = "smart_gallery_backend/test_embeddings"  
IMAGE_FOLDER = "smart_gallery_backend\smart_gallery_test"  

# Initialize ImageDBManager
image_db_manager = ImageDBManager(model_path=MODEL_PATH, db_path=DB_PATH)

result = image_db_manager.add_images_from_folder(IMAGE_FOLDER)
    
print(result)  

