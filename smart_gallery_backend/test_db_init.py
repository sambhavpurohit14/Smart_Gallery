import os
from image_db_util import ImageDBManager  

# Set paths
MODEL_PATH = "smart_gallery_backend\clip_model_epoch_12.pt"  
DB_PATH = "smart_gallery_backend/test_embeddings"  
IMAGE_FOLDER = "smart_gallery_backend\smart_gallery_test"  

# Initialize ImageDBManager
image_db_manager = ImageDBManager(model_path=MODEL_PATH, db_path=DB_PATH)

def create_new_database():
    """Creates a new ChromaDB and adds images from a folder."""
    if not os.path.isdir(IMAGE_FOLDER):
        print("Invalid image folder path.")
        return

    print(f"Creating a new database at {DB_PATH}...")
    
    # Add images to the database
    result = image_db_manager.add_images_from_folder(IMAGE_FOLDER)
    
    # Print the result
    if result["status"] == "success":
        print(f"Database created successfully. {result['added_images']} images added.")
    else:
        print(f"Error: {result['message']}")
    
    if result["errors"]:
        print(f"{len(result['errors'])} images had errors.")

if __name__ == "__main__":
    create_new_database()

