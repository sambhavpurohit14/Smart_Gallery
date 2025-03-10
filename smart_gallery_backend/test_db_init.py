from image_db_util import ImageDBManager
import os

test_model_path = r"smart_gallery_backend/clip_model_epoch_12.pt"
test_db_path = r"smart_gallery_backend/test_image_embeddings"
folder_path = r"smart_gallery_backend/smart_gallery_test"

assert os.path.exists(test_model_path), f"Error: Model file not found at {test_model_path}"

image_db = ImageDBManager(test_model_path, test_db_path)
print("ImageDBManager initialized")

create_embeddings = image_db.add_images_from_folder(folder_path)
print(create_embeddings)

