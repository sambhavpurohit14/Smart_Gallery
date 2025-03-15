import chromadb

chroma_client = chromadb.HttpClient(
    host="34.58.248.218",
    port=8000
)

result = chroma_client.heartbeat()
print("ChromaDB Heartbeat Response:", result)


