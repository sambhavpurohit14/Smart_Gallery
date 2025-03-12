import chromadb

chroma_client = chromadb.PersistentClient(path="smart_gallery_backend/test_embeddings")

collection = chroma_client.get_collection("image_embeddings")

# Fetch all stored data
entries = collection.get()

print(f"Stored IDs: {entries.get('ids', [])}")
print(f"Stored Embeddings: {entries.get('embeddings', [])}")  
