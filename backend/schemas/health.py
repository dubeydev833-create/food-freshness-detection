"""Health and root status schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    """Application status response for GET /."""

    app: str = Field(..., description="Application name")
    version: str = Field(..., description="API version")
    status: str = Field("online", description="Service status")
    docs_url: str = Field("/docs", description="Interactive API documentation URL")
    health_url: str = Field("/health", description="Health check endpoint URL")


class HealthResponse(BaseModel):
    """Health status response for GET /health."""

    status: str = Field(..., description="'healthy' or 'degraded'")
    backend_running: bool = Field(True, description="Whether the FastAPI backend is operational")
    model_available: bool = Field(..., description="Whether Member 1's trained model is available")
    model_path: str = Field(..., description="Target model file path")
    database_connected: bool = Field(..., description="Whether SQLite database is accessible")
    message: Optional[str] = Field(None, description="Diagnostic message or guidance")
