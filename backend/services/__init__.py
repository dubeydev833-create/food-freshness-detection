"""Service layer packages for FreshCheck AI backend."""

from backend.services.inference_service import InferenceService, get_inference_service

__all__ = ["InferenceService", "get_inference_service"]
