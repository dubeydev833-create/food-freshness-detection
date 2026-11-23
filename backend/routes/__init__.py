"""API route modules for FreshCheck AI."""

from backend.routes.health import router as health_router
from backend.routes.history import router as history_router
from backend.routes.predict import router as predict_router

__all__ = ["health_router", "predict_router", "history_router"]

