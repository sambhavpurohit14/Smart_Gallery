import os

CHROMA_HOST = os.getenv("CHROMA_HOST", "34.123.164.56")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))