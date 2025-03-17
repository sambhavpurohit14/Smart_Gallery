from fastapi import FastAPI
import uvicorn
import os
from routes import db_router, image_router

app = FastAPI(title="Smart Gallery API")

@app.get('/', tags=["health"])
async def heartbeat():
    return {"status": "ok"}

# Include routers
app.include_router(db_router)
app.include_router(image_router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)













