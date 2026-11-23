"""Configuration settings for FreshCheck AI Backend."""

import os
from pathlib import Path
from typing import List, Set


class Settings:
    """Application configuration parameters."""

    # Project metadata
    PROJECT_NAME: str = "FreshCheck AI Backend"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    DEBUG: bool = os.getenv("FRESHCHECK_DEBUG", "false").lower() in ("true", "1", "yes")

    # File paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DEFAULT_MODEL_PATH: Path = BASE_DIR / "models" / "freshcheck_model.keras"
    MODEL_PATH: Path = Path(os.getenv("FRESHCHECK_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

    # Database
    DEFAULT_DB_PATH: Path = BASE_DIR / "freshcheck.db"
    DB_PATH: Path = Path(os.getenv("FRESHCHECK_DB_PATH", str(DEFAULT_DB_PATH)))

    # Image upload restrictions
    MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    MAX_IMAGE_DIMENSION: int = 8192  # max width/height in px
    MIN_IMAGE_DIMENSION: int = 10    # min width/height in px
    ALLOWED_EXTENSIONS: Set[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
    ALLOWED_MIME_TYPES: Set[str] = frozenset({
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    })

    # Allowed categories per FreshCheck AI specification
    VALID_CATEGORIES: Set[str] = frozenset({"Fruits", "Vegetables", "Meat", "Dairy"})

    # Freshness classes in exact order
    FRESHNESS_CLASSES: List[str] = ["Fresh", "Moderate", "Spoiled"]

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",  # Streamlit default
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",  # Local development wildcard
    ]


settings = Settings()
