"""
Global Exception Handlers for FastAPI.

Translates domain exceptions into structured JSON responses with standard HTTP status codes.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    DecompressionBombError,
    ImageDimensionsTooSmallError,
    ImageProcessingError,
    ImageSizeExceededError,
    ImageTooBlurryError,
    ImageTooDarkError,
    InferenceQueueTimeoutError,
    InvalidImageFormatError,
    ModelTimeoutError,
    ModerationServiceUnavailableError,
    ModerationTimeoutError,
    UnsafeContentError,
)


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception mapping onto FastAPI application."""

    @app.exception_handler(ImageSizeExceededError)
    async def handle_size_exceeded(_: Request, exc: ImageSizeExceededError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={"error": "PAYLOAD_TOO_LARGE", "message": exc.client_message},
        )

    @app.exception_handler(InvalidImageFormatError)
    async def handle_invalid_format(_: Request, exc: InvalidImageFormatError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"error": "UNSUPPORTED_MEDIA_TYPE", "message": exc.client_message},
        )

    @app.exception_handler(DecompressionBombError)
    async def handle_decompression_bomb(_: Request, exc: DecompressionBombError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "IMAGE_TOO_LARGE", "message": exc.client_message},
        )

    @app.exception_handler(ImageTooDarkError)
    async def handle_too_dark(_: Request, exc: ImageTooDarkError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"error": "IMAGE_TOO_DARK", "message": exc.client_message},
        )

    @app.exception_handler(ImageTooBlurryError)
    async def handle_too_blurry(_: Request, exc: ImageTooBlurryError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"error": "IMAGE_TOO_BLURRY", "message": exc.client_message},
        )

    @app.exception_handler(ImageDimensionsTooSmallError)
    async def handle_too_small(_: Request, exc: ImageDimensionsTooSmallError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"error": "IMAGE_TOO_SMALL", "message": exc.client_message},
        )

    @app.exception_handler(UnsafeContentError)
    async def handle_unsafe_content(_: Request, exc: UnsafeContentError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "CONTENT_UNSAFE", "message": exc.client_message, "category": exc.category},
        )

    @app.exception_handler(InferenceQueueTimeoutError)
    async def handle_queue_timeout(_: Request, exc: InferenceQueueTimeoutError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "SERVER_BUSY", "message": exc.client_message},
        )

    @app.exception_handler(ModelTimeoutError)
    async def handle_model_timeout(_: Request, exc: ModelTimeoutError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"error": "TIMEOUT", "message": exc.client_message},
        )

    @app.exception_handler(ModerationTimeoutError)
    @app.exception_handler(ModerationServiceUnavailableError)
    async def handle_moderation_failure(_: Request, exc: ImageProcessingError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "SAFETY_CHECK_UNAVAILABLE", "message": exc.client_message},
        )
