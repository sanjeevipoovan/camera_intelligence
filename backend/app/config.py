"""
Application Configuration Module.

Centralized configuration using Pydantic Settings v2.
Supports vLLM (production deployment) and Ollama (local development).
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import (
    Field,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnvironment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class InferenceBackend(StrEnum):
    VLLM = "vllm"          # Production High-Throughput Engine
    OLLAMA = "ollama"      # Local Workstation Engine
    SIMULATED = "simulated"# CI/CD & Unit Tests


class ModelDevice(StrEnum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class ModelDType(StrEnum):
    AUTO = "auto"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class ImageOutputFormat(StrEnum):
    JPEG = "JPEG"
    PNG = "PNG"
    WEBP = "WEBP"


class ModerationFailPolicy(StrEnum):
    FAIL_CLOSED = "fail_closed"
    FAIL_OPEN = "fail_open"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(StrEnum):
    CONSOLE = "console"
    JSON = "json"


class Settings(BaseSettings):
    """Central settings schema enforcing strict boundaries and security guarantees."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # A. Application Configuration
    APP_NAME: str = Field(default="Visual-Intelligence-Backend")
    APP_ENV: AppEnvironment = Field(default=AppEnvironment.DEVELOPMENT)
    DEBUG: bool = Field(default=False)
    API_V1_PREFIX: str = Field(default="/api/v1")
    API_TITLE: str = Field(default="Visual Intelligence API")
    API_DESCRIPTION: str = Field(default="Mobile visual intelligence inference backend")
    VERSION: str = Field(default="1.0.0")

    # B. Model & Inference Backend Configuration
    INFERENCE_BACKEND: InferenceBackend = Field(
        default=InferenceBackend.OLLAMA,
        description="Active backend runtime: 'vllm' for cloud deployment, 'ollama' for local dev",
    )
    MODEL_NAME: str = Field(default="lfm3-omni:latest")
    
    # vLLM Server Settings (OpenAI-compatible VLM endpoint)
    VLLM_BASE_URL: str = Field(
        default="http://localhost:8000/v1",
        description="Endpoint of the deployed vLLM engine",
    )
    VLLM_API_KEY: SecretStr | None = Field(
        default=None,
        description="Bearer token for protected vLLM deployments",
    )

    # Ollama Settings (Local fallback)
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")

    MODEL_DEVICE: ModelDevice = Field(default=ModelDevice.AUTO)
    MODEL_DTYPE: ModelDType = Field(default=ModelDType.AUTO)
    MODEL_TIMEOUT: PositiveFloat = Field(default=60.0)
    MODEL_MAX_NEW_TOKENS: Annotated[int, Field(ge=1, le=4096)] = Field(default=512)
    MODEL_TEMPERATURE: Annotated[float, Field(ge=0.0, le=2.0)] = Field(default=0.7)
    MODEL_TOP_P: Annotated[float, Field(gt=0.0, le=1.0)] = Field(default=0.9)
    MODEL_TRUST_REMOTE_CODE: bool = Field(default=True)
    MODEL_LOAD_ON_STARTUP: bool = Field(default=True)
    # --------------------------------------------------------------------------
    # KV Cache & Advanced Inference Optimization
    # --------------------------------------------------------------------------
    ENABLE_PREFIX_CACHING: bool = Field(
        default=True,
        description="Enable automatic KV cache reuse across multiple turns on the same image",
    )
    KV_CACHE_DTYPE: str = Field(
        default="auto",
        description="KV cache precision: 'auto', 'fp8', 'fp16'. fp8 saves 50% VRAM on supported GPUs",
    )
    ENABLE_CHUNKED_PREFILL: bool = Field(
        default=True,
        description="Interleave image prefill with token decode to prevent streaming stutter",
    )
    MAX_MODEL_LEN: PositiveInt = Field(
        default=4096,
        description="Maximum context window budget (tokens) including visual tokens",
    )
    MAX_NUM_SEQS: PositiveInt = Field(
        default=16,
        description="Maximum concurrent sequences handled by PagedAttention",
    )

    # C. Concurrency Configuration
    MAX_CONCURRENT_INFERENCES: PositiveInt = Field(default=4)
    INFERENCE_QUEUE_ENABLED: bool = Field(default=True)
    INFERENCE_QUEUE_TIMEOUT: PositiveFloat = Field(default=20.0)

    # D. Image Upload Configuration
    MAX_IMAGE_SIZE_MB: PositiveFloat = Field(default=10.0)
    MAX_IMAGE_WIDTH: PositiveInt = Field(default=4096)
    MAX_IMAGE_HEIGHT: PositiveInt = Field(default=4096)
    MAX_IMAGE_PIXELS: PositiveInt = Field(default=16_777_216)
    ALLOWED_IMAGE_FORMATS: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"]
    )
    IMAGE_DECODE_TIMEOUT: PositiveFloat = Field(default=5.0)
    IMAGE_TEMP_DIR: Path = Field(default=Path("/tmp/visual_intel_images"))

    # E. Image Preprocessing Configuration
    IMAGE_AUTO_ORIENTATION: bool = Field(default=True)
    IMAGE_CONVERT_RGB: bool = Field(default=True)
    IMAGE_RESIZE_ENABLED: bool = Field(default=True)
    IMAGE_MAX_LONG_EDGE: PositiveInt = Field(default=1920)
    IMAGE_OUTPUT_FORMAT: ImageOutputFormat = Field(default=ImageOutputFormat.JPEG)
    IMAGE_JPEG_QUALITY: Annotated[int, Field(ge=1, le=100)] = Field(default=85)

    # F. Image Moderation Configuration
    IMAGE_MODERATION_ENABLED: bool = Field(default=True)
    NUDENET_ENABLED: bool = Field(default=True)
    NUDENET_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.60)
    SAFETY_CLASSIFIER_ENABLED: bool = Field(default=True)
    SAFETY_CLASSIFIER_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.70)
    MODERATION_TIMEOUT: PositiveFloat = Field(default=5.0)
    MODERATION_FAIL_POLICY: ModerationFailPolicy = Field(default=ModerationFailPolicy.FAIL_CLOSED)

    # G. Image Quality Configuration
    IMAGE_QUALITY_CHECK_ENABLED: bool = Field(default=True)
    BLUR_THRESHOLD: PositiveFloat = Field(default=100.0)
    BRIGHTNESS_MIN_THRESHOLD: Annotated[float, Field(ge=0.0, le=255.0)] = Field(default=30.0)
    IMAGE_MIN_WIDTH: PositiveInt = Field(default=224)
    IMAGE_MIN_HEIGHT: PositiveInt = Field(default=224)

    # H. Security Configuration
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    RATE_LIMIT: str = Field(default="60/minute")
    REQUEST_TIMEOUT: PositiveFloat = Field(default=60.0)
    MAX_REQUEST_SIZE_MB: PositiveFloat = Field(default=15.0)
    API_KEY_AUTH_ENABLED: bool = Field(default=False)
    API_SECRET_KEY: SecretStr | None = Field(default=None)

    # I. Privacy & Logging
    STORE_IMAGES: bool = Field(default=False)
    TEMP_IMAGE_AUTO_DELETE: bool = Field(default=True)
    TEMP_IMAGE_RETENTION_SECONDS: PositiveInt = Field(default=60)
    LOG_IMAGE_CONTENT: bool = Field(default=False)
    STORE_REQUEST_METADATA: bool = Field(default=True)
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO)
    LOG_FORMAT: LogFormat = Field(default=LogFormat.CONSOLE)
    ENABLE_REQUEST_ID: bool = Field(default=True)
    ENABLE_PERFORMANCE_METRICS: bool = Field(default=True)
    ENABLE_MODEL_METRICS: bool = Field(default=True)

    # J. Frontend Integration
    FRONTEND_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    FRONTEND_REQUEST_TIMEOUT: PositiveFloat = Field(default=45.0)
    MAX_QUESTION_LENGTH: Annotated[int, Field(ge=1, le=2000)] = Field(default=500)
    SESSION_ENABLED: bool = Field(default=False)

    @field_validator("CORS_ORIGINS", "FRONTEND_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                import json
                try:
                    return json.loads(value)
                except Exception as exc:
                    raise ValueError(f"Malformed JSON in origin list: {value}") from exc
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("ALLOWED_IMAGE_FORMATS", mode="before")
    @classmethod
    def parse_image_formats(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                import json
                try:
                    return json.loads(value)
                except Exception as exc:
                    raise ValueError(f"Malformed JSON in image formats: {value}") from exc
            return [fmt.strip().lower() for fmt in value.split(",") if fmt.strip()]
        return [fmt.lower() for fmt in value]

    @field_validator("ALLOWED_IMAGE_FORMATS")
    @classmethod
    def validate_image_formats(cls, formats: list[str]) -> list[str]:
        valid_mime_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
        for fmt in formats:
            if fmt not in valid_mime_types:
                raise ValueError(f"Unsupported format '{fmt}'.")
        return formats

    @model_validator(mode="after")
    def validate_production_invariants(self) -> "Settings":
        if self.APP_ENV == AppEnvironment.PRODUCTION and self.DEBUG:
            raise ValueError("DEBUG must be set to False when APP_ENV is 'production'.")

        if self.APP_ENV == AppEnvironment.PRODUCTION:
            for origin in self.CORS_ORIGINS:
                if origin == "*":
                    raise ValueError("Wildcard origin '*' is forbidden when APP_ENV is 'production'.")

        if self.API_KEY_AUTH_ENABLED:
            if not self.API_SECRET_KEY or len(self.API_SECRET_KEY.get_secret_value().strip()) < 16:
                raise ValueError("API_KEY_AUTH_ENABLED is True, but API_SECRET_KEY is missing or < 16 chars.")

        if (
            self.APP_ENV == AppEnvironment.PRODUCTION
            and self.IMAGE_MODERATION_ENABLED
            and self.MODERATION_FAIL_POLICY != ModerationFailPolicy.FAIL_CLOSED
        ):
            raise ValueError("MODERATION_FAIL_POLICY must be 'fail_closed' in production.")

        if self.LOG_IMAGE_CONTENT:
            raise ValueError("LOG_IMAGE_CONTENT cannot be set to True.")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()