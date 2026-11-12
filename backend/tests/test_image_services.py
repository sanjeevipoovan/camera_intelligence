"""
Unit tests for the Phase 1 Image Processing Subsystem.

Verifies MIME inspection, decompression bomb mitigation, EXIF handling,
alpha-channel compositing, downscaling, Laplacian sharpness, and luminance gating.
"""

import io
import numpy as np
import pytest
from PIL import Image

from app.core.exceptions import (
    DecompressionBombError,
    ImageDimensionsTooSmallError,
    ImageSizeExceededError,
    ImageTooBlurryError,
    ImageTooDarkError,
    InvalidImageFormatError,
)
from app.services.image_preprocessor import image_preprocessor
from app.services.image_quality import image_quality_service
from app.services.image_validator import image_validator


# ==============================================================================
# In-Memory Test Fixtures
# ==============================================================================

def create_synthetic_image(
    width: int = 400,
    height: int = 400,
    color: tuple[int, int, int] = (120, 120, 120),
    fmt: str = "JPEG",
    mode: str = "RGB",
    add_noise: bool = True,
) -> bytes:
    """Generate in-memory image bytes with optional high-frequency edge noise."""
    if add_noise:
        # Create random high-contrast pattern to guarantee high Laplacian variance (sharpness)
        np_arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(np_arr, mode="RGB")
        if mode != "RGB":
            img = img.convert(mode)
    else:
        img = Image.new(mode, (width, height), color=color)

    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


# ==============================================================================
# Validator Tests
# ==============================================================================

class TestImageValidator:

    def test_valid_jpeg_intake(self) -> None:
        data = create_synthetic_image(fmt="JPEG")
        metadata = image_validator.validate(data)
        assert metadata.format == "JPEG"
        assert metadata.mime_type == "image/jpeg"
        assert metadata.width == 400
        assert metadata.height == 400

    def test_valid_png_intake(self) -> None:
        data = create_synthetic_image(fmt="PNG", mode="RGBA")
        metadata = image_validator.validate(data)
        assert metadata.format == "PNG"
        assert metadata.mime_type == "image/png"

    def test_valid_webp_intake(self) -> None:
        data = create_synthetic_image(fmt="WEBP")
        metadata = image_validator.validate(data)
        assert metadata.format == "WEBP"
        assert metadata.mime_type == "image/webp"

    def test_spoofed_file_rejection(self) -> None:
        fake_payload = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff"
        with pytest.raises(InvalidImageFormatError):
            image_validator.validate(fake_payload)

    def test_empty_payload_rejection(self) -> None:
        with pytest.raises(InvalidImageFormatError):
            image_validator.validate(b"")

    def test_oversized_payload_rejection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.config.settings.MAX_IMAGE_SIZE_MB", 0.001)  # ~1 KB ceiling
        data = create_synthetic_image(width=500, height=500, fmt="JPEG")
        with pytest.raises(ImageSizeExceededError):
            image_validator.validate(data)

    def test_decompression_bomb_dimension_rejection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.config.settings.MAX_IMAGE_WIDTH", 300)
        data = create_synthetic_image(width=400, height=200, fmt="JPEG")
        with pytest.raises(DecompressionBombError):
            image_validator.validate(data)


# ==============================================================================
# Preprocessor Tests
# ==============================================================================

class TestImagePreprocessor:

    def test_alpha_compositing_to_rgb(self) -> None:
        # Transparent PNG
        png_data = create_synthetic_image(width=300, height=300, fmt="PNG", mode="RGBA", add_noise=False)
        metadata = image_validator.validate(png_data)

        processed = image_preprocessor.process(png_data, metadata)
        assert processed.channels == 3
        assert processed.format == "JPEG"

        # Ensure processed output is valid JPEG
        with Image.open(io.BytesIO(processed.data)) as check_img:
            assert check_img.mode == "RGB"

    def test_downscale_preserves_aspect_ratio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.config.settings.IMAGE_MAX_LONG_EDGE", 1000)
        # Create 2000x1000 image (2:1 aspect ratio)
        large_data = create_synthetic_image(width=2000, height=1000, fmt="JPEG")
        metadata = image_validator.validate(large_data)

        processed = image_preprocessor.process(large_data, metadata)
        assert processed.width == 1000
        assert processed.height == 500  # Preserved 2:1 aspect ratio

    def test_exif_orientation_correction(self) -> None:
        # Create image with EXIF tag 6 (rotated 90 degrees CCW)
        img = Image.new("RGB", (200, 100), color=(200, 200, 200))
        exif = img.getexif()
        exif[0x0112] = 6  # Orientation tag
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", exif=exif)
        raw_bytes = buffer.getvalue()

        metadata = image_validator.validate(raw_bytes)
        processed = image_preprocessor.process(raw_bytes, metadata)

        # Tag 6 rotates 90 deg, swapping width and height: 200x100 -> 100x200
        assert processed.width == 100
        assert processed.height == 200


# ==============================================================================
# Quality Service Tests
# ==============================================================================

class TestImageQualityService:

    def test_sharp_image_passes(self) -> None:
        sharp_data = create_synthetic_image(width=300, height=300, fmt="JPEG", add_noise=True)
        metadata = image_validator.validate(sharp_data)
        processed = image_preprocessor.process(sharp_data, metadata)

        assessment = image_quality_service.assess_quality(processed)
        assert assessment.is_acceptable is True
        assert assessment.blur_passed is True
        assert assessment.brightness_passed is True

    def test_solid_color_fails_blur(self) -> None:
        # Solid flat color image contains zero high-frequency edges (variance = 0.0)
        flat_data = create_synthetic_image(width=300, height=300, color=(128, 128, 128), fmt="JPEG", add_noise=False)
        metadata = image_validator.validate(flat_data)
        processed = image_preprocessor.process(flat_data, metadata)

        assessment = image_quality_service.assess_quality(processed)
        assert assessment.blur_passed is False
        assert assessment.is_acceptable is False

        with pytest.raises(ImageTooBlurryError):
            image_quality_service.validate_or_raise(processed)

    def test_pitch_black_fails_brightness(self) -> None:
        black_data = create_synthetic_image(width=300, height=300, color=(0, 0, 0), fmt="JPEG", add_noise=False)
        metadata = image_validator.validate(black_data)
        processed = image_preprocessor.process(black_data, metadata)

        assessment = image_quality_service.assess_quality(processed)
        assert assessment.brightness_passed is False
        assert assessment.brightness_score == 0.0

        with pytest.raises(ImageTooDarkError):
            image_quality_service.validate_or_raise(processed)

    def test_undersized_image_fails_dimensions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.config.settings.IMAGE_MIN_WIDTH", 400)
        data = create_synthetic_image(width=250, height=250, fmt="JPEG", add_noise=True)
        metadata = image_validator.validate(data)
        processed = image_preprocessor.process(data, metadata)

        with pytest.raises(ImageDimensionsTooSmallError):
            image_quality_service.validate_or_raise(processed)