import os
import chromadb

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="smart_gallery_backend/test_image_embeddings")
collection = client.get_or_create_collection("image_embeddings")

# Retrieve all embeddings from the collection
all_embeddings = collection.get()

# Extract embeddings, ids, and metadata
image_embeddings = all_embeddings.get('embeddings', [])
image_ids = all_embeddings.get('ids', [])
metadata_list = all_embeddings.get('metadatas', [])

# Print the embeddings
if not image_embeddings or not image_ids:
    print("No embeddings found in the database.")
else:
    print("Embeddings found in the database:")
    for image_id, embedding, metadata in zip(image_ids, image_embeddings, metadata_list):
        print(f"ID: {image_id}")
        print(f"Embedding: {embedding}")
        print(f"Metadata: {metadata}")
        print()
