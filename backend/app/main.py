"""
FastAPI Application Entry Point.

Integrates dynamic local-network CORS middleware, custom exception handlers,
lifespan events, and the API v1 routing tree.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
from app.moderation.service import moderation_service
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.config import AppEnvironment, get_settings
from app.core.handlers import register_exception_handlers
from app.inference.engine import engine

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup readiness and shutdown cleanup."""
    app_settings = get_settings()
    logger.info(
        "Booting %s [Env: %s, Debug: %s, Version: %s]",
        app_settings.APP_NAME,
        app_settings.APP_ENV.value,
        app_settings.DEBUG,
        app_settings.VERSION,
    )

    if app_settings.MODEL_LOAD_ON_STARTUP:
        logger.info("MODEL_LOAD_ON_STARTUP is enabled. Preloading %s...", app_settings.MODEL_NAME)
        engine.load_model()
    for provider in moderation_service.providers:
        if hasattr(provider, "load_model"):
            provider.load_model()
    yield

    logger.info("Application shutdown completed.")


def create_application() -> FastAPI:
    """Build and configure the FastAPI ASGI application."""
    app_settings = get_settings()

    app = FastAPI(
        title=app_settings.API_TITLE,
        description=app_settings.API_DESCRIPTION,
        version=app_settings.VERSION,
        docs_url="/docs" if app_settings.APP_ENV != AppEnvironment.PRODUCTION else None,
        redoc_url="/redoc" if app_settings.APP_ENV != AppEnvironment.PRODUCTION else None,
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing (CORS)
    # In development: automatically allows localhost and any Wi-Fi IP (192.168.x.x, 10.x.x.x)
    if app_settings.APP_ENV == AppEnvironment.DEVELOPMENT:
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=r"^https?://.*",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    # Global Exception Handlers
    register_exception_handlers(app)

    # Mount Version 1 Router
    app.include_router(api_router, prefix=app_settings.API_V1_PREFIX)

    return app


app = create_application()