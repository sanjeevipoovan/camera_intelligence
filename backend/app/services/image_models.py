"""Data schemas for image validation, preprocessing, and quality metrics."""

from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field


class ImageMetadata(BaseModel):
    """Extracted technical properties of an ingested image."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(description="Width in pixels")
    height: int = Field(description="Height in pixels")
    format: str = Field(description="Standardized format identifier (JPEG, PNG, WEBP)")
    mime_type: str = Field(description="MIME type identifier (e.g. image/jpeg)")
    size_bytes: int = Field(description="Raw payload size in bytes")
    aspect_ratio: float = Field(description="Width divided by height")

    @property
    def total_pixels(self) -> int:
        return self.width * self.height


class QualityAssessment(BaseModel):
    """Diagnostic outcome of OpenCV sharpness and lighting checks."""

    model_config = ConfigDict(frozen=True)

    is_acceptable: bool = Field(description="Whether the image passes all quality gates")
    blur_score: float = Field(description="Calculated Laplacian variance")
    blur_passed: bool = Field(description="Whether sharpness meets BLUR_THRESHOLD")
    brightness_score: float = Field(description="Calculated mean luminance (0-255)")
    brightness_passed: bool = Field(description="Whether lighting meets BRIGHTNESS_MIN_THRESHOLD")
    dimensions_passed: bool = Field(description="Whether image meets minimum width/height")
    rejection_reasons: list[str] = Field(
        default_factory=list,
        description="User-facing explanations for quality check failures",
    )


@dataclass(slots=True, frozen=True)
class ProcessedImage:
    """Normalized, preprocessed image container ready for moderation or model inference."""

    data: bytes
    format: str
    width: int
    height: int
    channels: int
    metadata: ImageMetadata
    quality: QualityAssessment | None = None