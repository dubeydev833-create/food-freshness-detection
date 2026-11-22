"""Metadata processing utilities for FreshCheck AI (Phase 5).

Handles validation, normalization, and packaging of optional environmental
and storage metadata without altering deep-learning image predictions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from image_processing.exceptions import MetadataValidationError

# Valid food categories per project specification
VALID_CATEGORIES = frozenset({"Fruits", "Vegetables", "Meat", "Dairy"})

# Valid storage conditions
COMMON_STORAGE_CONDITIONS = frozenset({
    "Room Temperature",
    "Refrigerated",
    "Frozen",
    "Cool & Dry",
    "Pantry",
    "Controlled Atmosphere",
})


def validate_category(category: str) -> str:
    """Validate and format the food category name.

    Args:
        category: Food category string (e.g. 'fruits', 'Fruits', 'meat').

    Returns:
        str: Canonical capitalized category name.

    Raises:
        MetadataValidationError: If category is not in the supported list.
    """
    if not isinstance(category, str) or not category.strip():
        raise MetadataValidationError(
            f"Category must be a non-empty string. Supported: {sorted(VALID_CATEGORIES)}"
        )

    clean_cat = category.strip().capitalize()
    if clean_cat not in VALID_CATEGORIES:
        # Check case-insensitive match
        for valid in VALID_CATEGORIES:
            if clean_cat.lower() == valid.lower():
                return valid
        raise MetadataValidationError(
            f"Unsupported food category '{category}'. "
            f"Must be one of: {sorted(VALID_CATEGORIES)}"
        )
    return clean_cat


def process_food_metadata(
    category: str,
    temperature_celsius: Optional[Union[int, float]] = None,
    storage_condition: Optional[str] = None,
    storage_duration_days: Optional[Union[int, float]] = None,
    storage_duration_hours: Optional[Union[int, float]] = None,
) -> Dict[str, Any]:
    """Process, validate, and structure optional food storage metadata.

    IMPORTANT SCIENTIFIC INTEGRITY RULE:
        This function strictly processes and validates supplied metadata.
        It DOES NOT invent sensor values and DOES NOT modify the deep-learning
        model's image prediction, preserving separate telemetry for the backend.

    Args:
        category: Food category ('Fruits', 'Vegetables', 'Meat', 'Dairy').
        temperature_celsius: Observed temperature in Celsius, when supplied.
        storage_condition: Storage setting (e.g. 'Refrigerated', 'Room Temperature').
        storage_duration_days: Storage duration in days, when supplied.
        storage_duration_hours: Storage duration in hours, when supplied.

    Returns:
        Dict[str, Any]: Validated metadata dictionary.

    Raises:
        MetadataValidationError: If category or physical ranges are invalid.
    """
    canonical_category = validate_category(category)

    # Validate temperature if supplied
    validated_temp: Optional[float] = None
    if temperature_celsius is not None:
        try:
            temp_val = float(temperature_celsius)
        except (ValueError, TypeError) as exc:
            raise MetadataValidationError(
                f"temperature_celsius must be numeric, got: {temperature_celsius}"
            ) from exc

        # Physical sanity check: -40 C (deep freeze) to 60 C (hot ambient)
        if temp_val < -40.0 or temp_val > 60.0:
            raise MetadataValidationError(
                f"Temperature {temp_val} C is outside realistic terrestrial food storage bounds (-40C to 60C)."
            )
        validated_temp = round(temp_val, 2)

    # Validate storage duration if supplied
    validated_days: Optional[float] = None
    if storage_duration_days is not None:
        try:
            days_val = float(storage_duration_days)
        except (ValueError, TypeError) as exc:
            raise MetadataValidationError(
                f"storage_duration_days must be numeric, got: {storage_duration_days}"
            ) from exc

        if days_val < 0.0:
            raise MetadataValidationError(
                f"storage_duration_days cannot be negative: {days_val}"
            )
        validated_days = round(days_val, 2)

    validated_hours: Optional[float] = None
    if storage_duration_hours is not None:
        try:
            hours_val = float(storage_duration_hours)
        except (ValueError, TypeError) as exc:
            raise MetadataValidationError(
                f"storage_duration_hours must be numeric, got: {storage_duration_hours}"
            ) from exc

        if hours_val < 0.0:
            raise MetadataValidationError(
                f"storage_duration_hours cannot be negative: {hours_val}"
            )
        validated_hours = round(hours_val, 2)

    # Normalize storage condition string if supplied
    cleaned_condition: Optional[str] = None
    if storage_condition is not None:
        cond_str = str(storage_condition).strip()
        if cond_str:
            cleaned_condition = cond_str.title()

    return {
        "category": canonical_category,
        "temperature_celsius": validated_temp,
        "storage_condition": cleaned_condition,
        "storage_duration_days": validated_days,
        "storage_duration_hours": validated_hours,
        "metadata_supplied": any(
            v is not None
            for v in (validated_temp, cleaned_condition, validated_days, validated_hours)
        ),
    }
