"""Application Configuration for FreshCheck AI Frontend.

Centralizes all endpoints, styling constants, classification classes,
category definitions, and environmental parameters.
"""

import os
from typing import Dict, List, Tuple

# App Information
APP_NAME = "FreshCheck AI"
APP_SUBTITLE = "Smart Food Freshness Detection & Spoilage Prevention System"
APP_VERSION = "1.0.0"
APP_ICON = "🥗"

# Backend Configuration
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 15

# API Endpoints
API_HEALTH_ENDPOINT = "/health"
API_PREDICT_ENDPOINT = "/predict"
API_HISTORY_ENDPOINT = "/history"
API_CLEAR_HISTORY_ENDPOINT = "/history"  # Typically DELETE /history

# Food Categories (Strictly aligned with Member 2 & Member 3)
SUPPORTED_CATEGORIES: List[str] = [
    "Fruits",
    "Vegetables",
    "Meat",
    "Dairy",
]

CATEGORY_ICONS: Dict[str, str] = {
    "Fruits": "🍎",
    "Vegetables": "🥦",
    "Meat": "🥩",
    "Dairy": "🧀",
}

# Freshness Classes (Strict Order)
FRESHNESS_CLASSES: Tuple[str, str, str] = ("Fresh", "Moderate", "Spoiled")

# Freshness Palette & Visual Indicators
CLASS_COLORS: Dict[str, str] = {
    "Fresh": "#10B981",     # Vibrant Emerald Green
    "Moderate": "#F59E0B",  # Warning Amber
    "Spoiled": "#EF4444",   # Critical Crimson Red
}

CLASS_BG_COLORS: Dict[str, str] = {
    "Fresh": "#ECFDF5",
    "Moderate": "#FFFBEB",
    "Spoiled": "#FEF2F2",
}

CLASS_BORDER_COLORS: Dict[str, str] = {
    "Fresh": "#A7F3D0",
    "Moderate": "#FDE68A",
    "Spoiled": "#FECACA",
}

CLASS_BADGE_TEXT_COLORS: Dict[str, str] = {
    "Fresh": "#065F46",
    "Moderate": "#92400E",
    "Spoiled": "#991B1B",
}

CLASS_ICONS: Dict[str, str] = {
    "Fresh": "🟢",
    "Moderate": "🟡",
    "Spoiled": "🔴",
}

# Supported Storage Conditions
STORAGE_CONDITIONS: List[str] = [
    "Refrigerated",
    "Room Temperature",
    "Frozen",
    "Cool & Dry",
    "Pantry",
    "Controlled Atmosphere",
]

# Supported Image Formats
ALLOWED_IMAGE_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]
MAX_IMAGE_SIZE_MB = 20

# UI Theme Color Palette (Neutral + Green Accent)
PALETTE = {
    "primary": "#10B981",
    "primary_dark": "#059669",
    "primary_light": "#D1FAE5",
    "bg_main": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "border_light": "#E2E8F0",
    "text_dark": "#0F172A",
    "text_muted": "#64748B",
    "text_body": "#334155",
}

# Explicit Food Safety Legal / Academic Disclaimer
FOOD_SAFETY_DISCLAIMER = (
    "⚠️ **Important Food Safety Disclaimer**: FreshCheck AI provides optical classifications "
    "based strictly on computer vision image analysis. Image-based predictions cannot detect internal bacteria, "
    "pathogens (e.g., Salmonella, E. coli), chemical toxins, or microbial contamination. Confidence scores reflect "
    "visual machine learning model certainty, NOT food safety probability. Always inspect smell, texture, storage history, "
    "and expiration dates before consuming food."
)
