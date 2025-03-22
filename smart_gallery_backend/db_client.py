import chromadb

# Initialize global variable for ChromaDB client
CHROMA_CLIENT = None

def get_client():
    return CHROMA_CLIENT

def set_client(client):
    global CHROMA_CLIENT
    CHROMA_CLIENT = client