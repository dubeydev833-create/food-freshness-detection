"""Validation and security service for FreshCheck AI API.

Enforces strict image file format, magic bytes, size limits, dimensional checks,
executable file rejection, food category checks, and environmental metadata bounds.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

from backend.config import settings

logger = logging.getLogger("FreshCheckAI.Validation")

# Binary magic bytes signatures for allowed image formats
JPEG_MAGIC = b"\xFF\xD8\xFF"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
WEBP_MAGIC_RIFF = b"RIFF"
WEBP_MAGIC_TYPE = b"WEBP"

# Known dangerous signatures to explicitly block
DANGEROUS_SIGNATURES = {
    b"MZ": "Windows Executable (.exe / .dll)",
    b"\x7fELF": "Linux ELF Binary",
    b"PK\x03\x04": "Zip Archive / Jar file",
    b"%PDF": "PDF Document",
    b"<!DOC": "HTML Document",
    b"<html": "HTML Document",
    b"<?xml": "XML Document",
    b"<?php": "PHP Script",
    b"#!": "Shell Script",
    b"\xca\xfe\xba\xbe": "Java Class / Mach-O Binary",
    b"\xfe\xed\xfa\xce": "Mach-O Binary",
}

DISALLOWED_EXTENSIONS = frozenset({
    ".exe", ".bat", ".cmd", ".sh", ".py", ".js", ".html", ".htm",
    ".php", ".dll", ".bin", ".iso", ".msi", ".jar", ".zip", ".tar",
    ".gz", ".pdf", ".docx", ".xlsx", ".apk", ".scr", ".com",
})


def validate_image_file(
    contents: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Tuple[int, int]:
    """Validate uploaded image for security, formatting, integrity, and dimensions.

    Args:
        contents: Raw bytes of the uploaded file.
        filename: Original filename provided in the multipart header.
        content_type: MIME type reported by the client.

    Returns:
        Tuple[int, int]: Validated (width, height) of the image.

    Raises:
        HTTPException: With appropriate HTTP status code (400, 413, 415).
    """
    # 1. Missing / Empty input check
    if not contents or len(contents) == 0:
        logger.warning("Rejected upload: Empty file (0 bytes)")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes). Please upload a valid food image.",
        )

    # 2. Oversized image check
    if len(contents) > settings.MAX_IMAGE_SIZE_BYTES:
        max_mb = settings.MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
        actual_mb = len(contents) / (1024 * 1024)
        logger.warning(f"Rejected upload: Size {actual_mb:.2f}MB exceeds {max_mb}MB limit")
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=(
                f"Image file exceeds maximum allowable size of {max_mb:.0f} MB "
                f"(received {actual_mb:.2f} MB)."
            ),
        )

    # 3. Filename extension security check
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in DISALLOWED_EXTENSIONS:
            logger.warning(f"Security alert: Blocked dangerous file upload with extension '{ext}'")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Executable and arbitrary files with extension '{ext}' are strictly prohibited. "
                    f"Only image files (JPEG, PNG, WEBP) are accepted."
                ),
            )
        if ext and ext not in settings.ALLOWED_EXTENSIONS:
            logger.warning(f"Rejected unsupported extension: '{ext}'")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Unsupported file extension '{ext}'. "
                    f"Supported formats: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}."
                ),
            )

    # 4. Content-Type check (if provided)
    if content_type:
        clean_mime = content_type.split(";")[0].strip().lower()
        if clean_mime not in settings.ALLOWED_MIME_TYPES:
            logger.warning(f"Rejected unsupported Content-Type: '{clean_mime}'")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Unsupported media type '{clean_mime}'. "
                    f"Supported Content-Types: {', '.join(sorted(settings.ALLOWED_MIME_TYPES))}."
                ),
            )

    # 5. Magic bytes & anti-malware signature inspection
    for sig, desc in DANGEROUS_SIGNATURES.items():
        if contents.startswith(sig):
            logger.warning(f"Security alert: File starts with dangerous signature: {desc}")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Security rejection: Uploaded file identified as {desc}, not an image.",
            )

    is_jpeg = contents.startswith(JPEG_MAGIC)
    is_png = contents.startswith(PNG_MAGIC)
    is_webp = contents.startswith(WEBP_MAGIC_RIFF) and len(contents) > 12 and contents[8:12] == WEBP_MAGIC_TYPE

    if not (is_jpeg or is_png or is_webp):
        logger.warning("Rejected upload: Magic bytes do not match JPEG, PNG, or WEBP")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Invalid image format: File header does not match valid JPEG, PNG, or WEBP signatures. "
                "Arbitrary binary uploads are not permitted."
            ),
        )

    # 6. Deep image decode & corruption verification with PIL
    try:
        with Image.open(io.BytesIO(contents)) as pil_img:
            # Verify internal structure
            pil_img.verify()
            fmt = (pil_img.format or "").upper()
            if fmt not in ("JPEG", "PNG", "WEBP"):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Decoded image format '{fmt}' is not supported. Use JPEG, PNG, or WEBP.",
                )

        # Re-open after verify() to read dimensions safely
        with Image.open(io.BytesIO(contents)) as pil_img:
            width, height = pil_img.size
            if width < settings.MIN_IMAGE_DIMENSION or height < settings.MIN_IMAGE_DIMENSION:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Image dimensions ({width}x{height}) are too small. "
                        f"Minimum resolution is {settings.MIN_IMAGE_DIMENSION}x{settings.MIN_IMAGE_DIMENSION} pixels."
                    ),
                )
            if width > settings.MAX_IMAGE_DIMENSION or height > settings.MAX_IMAGE_DIMENSION:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of "
                        f"{settings.MAX_IMAGE_DIMENSION} pixels."
                    ),
                )
            return width, height

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning(f"Corrupted or invalid image upload: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image file: Failed to decode image stream.",
        ) from exc


def validate_category_input(category: str) -> str:
    """Validate food category against the four supported FreshCheck AI categories.

    Args:
        category: Input string from client.

    Returns:
        str: Canonical capitalized category name ('Fruits', 'Vegetables', 'Meat', 'Dairy').

    Raises:
        HTTPException: HTTP 400 Bad Request if missing or invalid.
    """
    if not category or not isinstance(category, str) or not category.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Missing food category. You must provide one of: "
                f"{', '.join(sorted(settings.VALID_CATEGORIES))}."
            ),
        )

    clean = category.strip().capitalize()
    for valid in settings.VALID_CATEGORIES:
        if clean.lower() == valid.lower():
            return valid

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            f"Invalid category '{category}'. "
            f"Supported categories are: {', '.join(sorted(settings.VALID_CATEGORIES))}."
        ),
    )


def validate_temperature_input(temperature: Optional[Union[int, float]]) -> Optional[float]:
    """Validate optional storage temperature.

    Args:
        temperature: Observed temperature in Celsius.

    Returns:
        Optional[float]: Rounded temperature, or None if omitted.

    Raises:
        HTTPException: HTTP 400 Bad Request if outside physical food storage bounds.
    """
    if temperature is None:
        return None

    try:
        temp_val = float(temperature)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"temperature must be a numeric value, got: '{temperature}'.",
        ) from exc

    if temp_val < -40.0 or temp_val > 60.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Temperature {temp_val}°C is outside realistic terrestrial food storage bounds "
                f"(-40°C to 60°C)."
            ),
        )
    return round(temp_val, 2)
