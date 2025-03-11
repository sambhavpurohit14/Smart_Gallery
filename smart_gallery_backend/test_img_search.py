from image_search_util import ImageSearcher
import os

model_path = os.path.join("smart_gallery_backend", "clip_model_epoch_12.pt")
db_path = os.path.join("smart_gallery_backend", "test_image_embeddings")


# Initialize ImageSearcher
image_searcher = ImageSearcher(model_path, db_path)
print("ImageSearcher initialized")

# Perform a search query
query = "a photo of a cat"
search_result = image_searcher.search_image(query)

# Check if the result is a FileResponse or a message
if isinstance(search_result, dict) and "message" in search_result:
    print(search_result["message"])
else:
    # Save the result to a file if it's a FileResponse
    result_path = "search_result.jpg"
    with open(result_path, "wb") as f:
        f.write(search_result.body)
    print(f"Search result saved to {result_path}")