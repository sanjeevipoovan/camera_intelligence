"""System readiness and health check endpoint."""

from fastapi import APIRouter, status
from pydantic import BaseModel
from app.config import settings
from app.inference.engine import engine

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    version: str
    model_name: str
    device: str
    model_loaded: bool
    moderation_enabled: bool


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def check_health() -> HealthResponse:
    """Return backend operational status and active hardware targets."""
    return HealthResponse(
        status="healthy",
        app=settings.APP_NAME,
        environment=settings.APP_ENV.value,
        version=settings.VERSION,
        model_name=settings.MODEL_NAME,
        device=engine.device,
        model_loaded=engine.is_loaded,
        moderation_enabled=settings.IMAGE_MODERATION_ENABLED,
    )