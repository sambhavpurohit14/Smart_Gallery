from .db_routes import router as db_router
from .image_routes import router as image_router

__all__ = ['db_router', 'image_router']