from image_search_util import ImageSearcher
import os

db_path = os.path.join("smart_gallery_backend", "test_embeddings")


# Initialize ImageSearcher
image_searcher = ImageSearcher(db_path)
print("ImageSearcher initialized")

# Perform a search query
query = "a bird flying in the sky"
search_result = image_searcher.search_image(query)
print(search_result)
