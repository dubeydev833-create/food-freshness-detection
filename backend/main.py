"""FreshCheck AI — FastAPI Application Entry Point.

College Mini Project: Food Freshness Assessment Application.
Member 3: FastAPI Backend Integration.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.database import init_db
from backend.routes.health import router as health_router
from backend.routes.history import router as history_router
from backend.routes.predict import router as predict_router
from image_processing.exceptions import (
    FreshCheckError,
    ImageFormatError,
    ImageReadError,
    ImageSizeError,
    ImageValidationError,
    MetadataValidationError,
    ModelInferenceError,
    ModelNotFoundError,
)

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

# Exception handlers for domain errors
@app.exception_handler(ImageFormatError)
async def image_format_exception_handler(request: Request, exc: ImageFormatError):
    return JSONResponse(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        content={"success": False, "error": "UnsupportedImageFormat", "detail": str(exc)},
    )


@app.exception_handler(ImageSizeError)
async def image_size_exception_handler(request: Request, exc: ImageSizeError):
    return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content={"success": False, "error": "ImageSizeExceeded", "detail": str(exc)},
    )


@app.exception_handler(ImageValidationError)
async def image_validation_exception_handler(request: Request, exc: ImageValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"success": False, "error": "ImageValidationError", "detail": str(exc)},
    )


@app.exception_handler(ImageReadError)
async def image_read_exception_handler(request: Request, exc: ImageReadError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"success": False, "error": "CorruptImageError", "detail": str(exc)},
    )


@app.exception_handler(MetadataValidationError)
async def metadata_validation_exception_handler(request: Request, exc: MetadataValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"success": False, "error": "MetadataValidationError", "detail": str(exc)},
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_exception_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"success": False, "error": "ModelUnavailable", "detail": str(exc)},
    )


@app.exception_handler(ModelInferenceError)
async def model_inference_exception_handler(request: Request, exc: ModelInferenceError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": "ModelInferenceError", "detail": str(exc)},
    )


# Mount routes
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(history_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
