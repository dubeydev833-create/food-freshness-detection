"""FreshCheck AI — FastAPI Application Entry Point.

College Mini Project: Food Freshness Assessment Application.
Member 3: FastAPI Backend Integration.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database import init_db
from backend.routes.health import router as health_router

# Configure root logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("FreshCheckAI.Backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Initializing FreshCheck AI Backend...")
    # Initialize SQLite database
    init_db(settings.DB_PATH)
    logger.info(f"Database initialized at: {settings.DB_PATH}")

    # Check model presence on startup
    if settings.MODEL_PATH.exists():
        logger.info(f"Found trained model at: {settings.MODEL_PATH}")
    else:
        logger.warning(
            f"Trained model not found at '{settings.MODEL_PATH}'. "
            "Requests to /predict will require model placement or test mock."
        )

    yield

    logger.info("FreshCheck AI Backend shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Backend REST API for FreshCheck AI — an AI-powered food freshness assessment application.\n\n"
        "Connects Member 4's Streamlit frontend with Member 2's image processing/inference engine "
        "and Member 1's trained deep-learning model.\n\n"
        "Supported Categories: **Fruits**, **Vegetables**, **Meat**, **Dairy**\n"
        "Freshness Classes: **Fresh**, **Moderate**, **Spoiled**"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware for local frontend development (Member 4's Streamlit app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount health routes
app.include_router(health_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
