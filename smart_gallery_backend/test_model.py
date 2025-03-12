from clip_model import CLIPFeatureExtractor
import os
model_path = os.path.join("smart_gallery_backend", "clip_model_epoch_12.pt")
feature_extractor = CLIPFeatureExtractor(model_path)
test_image_path = os.path.join("smart_gallery_backend", "smart_gallery_test", "000000018737.jpg")
image_features = feature_extractor.extract_image_features(test_image_path)
print(image_features[:5])