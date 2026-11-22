"""Unit tests for metadata validation and normalization (Phase 5)."""

import pytest

from image_processing.exceptions import MetadataValidationError
from image_processing.metadata import (
    VALID_CATEGORIES,
    process_food_metadata,
    validate_category,
)


class TestCategoryValidation:
    """Tests for validating food categories."""

    @pytest.mark.parametrize("cat", ["Fruits", "Vegetables", "Meat", "Dairy"])
    def test_valid_exact_categories(self, cat: str):
        validated = validate_category(cat)
        assert validated == cat

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("fruits", "Fruits"),
            ("VEGETABLES", "Vegetables"),
            ("meat", "Meat"),
            (" dairy ", "Dairy"),
        ],
    )
    def test_case_insensitive_categories(self, raw: str, expected: str):
        validated = validate_category(raw)
        assert validated == expected

    def test_invalid_category_raises(self):
        with pytest.raises(MetadataValidationError, match="Unsupported food category"):
            validate_category("Seafood")

    def test_empty_category_raises(self):
        with pytest.raises(MetadataValidationError, match="non-empty string"):
            validate_category("")


class TestMetadataProcessing:
    """Tests for environmental and storage metadata processing."""

    def test_full_metadata_processing(self):
        meta = process_food_metadata(
            category="fruits",
            temperature_celsius=4.5,
            storage_condition="Refrigerated",
            storage_duration_days=3.0,
            storage_duration_hours=72.0,
        )

        assert meta["category"] == "Fruits"
        assert meta["temperature_celsius"] == 4.5
        assert meta["storage_condition"] == "Refrigerated"
        assert meta["storage_duration_days"] == 3.0
        assert meta["storage_duration_hours"] == 72.0
        assert meta["metadata_supplied"] is True

    def test_empty_optional_metadata(self):
        meta = process_food_metadata(category="Dairy")
        assert meta["category"] == "Dairy"
        assert meta["temperature_celsius"] is None
        assert meta["storage_condition"] is None
        assert meta["storage_duration_days"] is None
        assert meta["metadata_supplied"] is False

    def test_negative_temperature_valid_for_frozen(self):
        meta = process_food_metadata(
            category="Meat",
            temperature_celsius=-18.0,
            storage_condition="Frozen",
        )
        assert meta["temperature_celsius"] == -18.0

    def test_reject_unrealistic_temperature(self):
        with pytest.raises(MetadataValidationError, match="outside realistic terrestrial"):
            process_food_metadata(category="Fruits", temperature_celsius=95.0)

        with pytest.raises(MetadataValidationError, match="outside realistic terrestrial"):
            process_food_metadata(category="Fruits", temperature_celsius=-60.0)

    def test_reject_negative_duration(self):
        with pytest.raises(MetadataValidationError, match="cannot be negative"):
            process_food_metadata(category="Vegetables", storage_duration_days=-1.0)

        with pytest.raises(MetadataValidationError, match="cannot be negative"):
            process_food_metadata(category="Vegetables", storage_duration_hours=-5.0)
