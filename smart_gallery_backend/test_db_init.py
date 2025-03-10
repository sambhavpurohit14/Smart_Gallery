from image_db_util import ImageDBManager

test_model_path="smart_gallery_backend\clip_model_epoch_12.pt"
test_db_path = "test_image_embeddings"

imgage_db=ImageDBManager(test_model_path, test_db_path)
print("ImageDBManager initialized")

create_embeddings = imgage_db.add_images_from_folder("smart_gallery_backend\smart_gallery_test")
