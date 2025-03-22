from fastapi import FastAPI
import uvicorn
import os
import chromadb
import logging
from routes import db_router, image_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Chroma HTTP client once
CHROMA_CLIENT = chromadb.HttpClient(host="34.123.164.56", port=8000)

# Create FastAPI application
app = FastAPI(title="Smart Gallery API")

@app.get('/heartbeat', tags=["health"])
async def heartbeat():
    """Check if the service is running."""
    return {"status": "ok"}

app.include_router(
    db_router,
    prefix="/db",
    tags=["database"]
)

app.include_router(
    image_router,
    prefix="/images",
    tags=["images"]
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5080))
    logger.info(f"Starting Smart Gallery API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)













