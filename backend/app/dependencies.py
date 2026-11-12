"""Reusable dependencies for FastAPI routers."""

from fastapi import Header, HTTPException, status
from app.config import settings


async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Enforce API Key validation if API_KEY_AUTH_ENABLED is True."""
    if not settings.API_KEY_AUTH_ENABLED:
        return

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required X-API-Key header.",
        )

    configured_secret = settings.API_SECRET_KEY.get_secret_value() if settings.API_SECRET_KEY else ""
    if x_api_key != configured_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid X-API-Key header provided.",
        )