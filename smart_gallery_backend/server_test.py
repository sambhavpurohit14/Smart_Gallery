import chromadb

chroma_client = chromadb.HttpClient(
    host="34.58.248.218",
    port=8000
)
chroma_client.heartbeat()

