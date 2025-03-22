from fastapi import FastAPI
import uvicorn
import os
import logging
from routes import db_router, image_router
import db_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(title="Smart Gallery API")

@app.get('/', tags=["health"])
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

# Set up initial client
db_client.set_client(None)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5080))
    logger.info(f"Starting Smart Gallery API on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port)















