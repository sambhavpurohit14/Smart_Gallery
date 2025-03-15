import chromadb

chroma_client = chromadb.PersistentClient(path="/mnt/chromadb")

# Create a collection
collection = chroma_client.get_or_create_collection(name="my_collection")

collection.add(
    ids=["test1"],
    documents=["Hello, ChromaDB from persistent disk!"]
)

print("ChromaDB is set up with persistence!")
