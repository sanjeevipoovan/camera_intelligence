"""
Image Validation Service.

Enforces binary signature validation, MIME verification, payload boundaries,
and decompression bomb defenses.
"""

import io
from PIL import Image

from app.config import settings
from app.core.exceptions import (
    CorruptImageError,
    DecompressionBombError,
    ImageSizeExceededError,
    InvalidImageFormatError,
)
from app.services.image_models import ImageMetadata

# Magic signatures for supported formats
# Prevents spoofed Content-Type attacks (e.g. renaming an executable to .jpg)
MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/webp": [b"RIFF"],  # WebP also contains 'WEBP' at offset 8-12
}

FORMAT_TO_MIME: dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class ImageValidator:
    """Validates raw image uploads prior to in-memory processing."""

    def __init__(self) -> None:
        # Enforce PIL safety ceiling against decompression bombs
        Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS

    def validate(self, raw_bytes: bytes, declared_content_type: str | None = None) -> ImageMetadata:
        """
        Validate raw image bytes and extract technical metadata.

        Args:
            raw_bytes: Raw binary stream of the uploaded file.
            declared_content_type: Content-Type header sent by client.

        Returns:
            ImageMetadata containing dimensions, MIME type, and size.

        Raises:
            ImageSizeExceededError: If byte size exceeds MAX_IMAGE_SIZE_MB.
            InvalidImageFormatError: If magic bytes or format are unsupported.
            DecompressionBombError: If pixel count exceeds safety threshold.
            CorruptImageError: If header structure is truncated or unreadable.
        """
        self._validate_byte_size(raw_bytes)
        detected_mime = self._verify_magic_signatures(raw_bytes)
        return self._inspect_image_header(raw_bytes, detected_mime)

    def _validate_byte_size(self, raw_bytes: bytes) -> None:
        """Verify upload size does not exceed the allowed threshold."""
        size_bytes = len(raw_bytes)
        if size_bytes == 0:
            raise InvalidImageFormatError(
                message="Uploaded file is empty.",
                client_message="The uploaded image contains no data. Please snap a new picture.",
            )

        max_allowed_bytes = int(settings.MAX_IMAGE_SIZE_MB * 1024 * 1024)
        if size_bytes > max_allowed_bytes:
            raise ImageSizeExceededError(
                message=f"Payload size {size_bytes} bytes exceeds ceiling {max_allowed_bytes} bytes.",
                client_message=(
                    f"Image size exceeds the {settings.MAX_IMAGE_SIZE_MB:.1f} MB limit. "
                    "Please select a smaller photo."
                ),
            )

    def _verify_magic_signatures(self, raw_bytes: bytes) -> str:
        """Inspect initial binary bytes to detect actual file type."""
        header_sample = raw_bytes[:16]

        # Check JPEG
        if header_sample.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"

        # Check PNG
        if header_sample.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"

        # Check WebP (RIFF....WEBP)
        if header_sample.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP":
            return "image/webp"

        raise InvalidImageFormatError(
            message="Binary signature does not match any allowed image format.",
            client_message="Unsupported or invalid file format. Please upload a JPEG, PNG, or WebP photo.",
        )

    def _inspect_image_header(self, raw_bytes: bytes, detected_mime: str) -> ImageMetadata:
        """Inspect headers using Pillow without loading full raster data into memory."""
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img_format = img.format
                width, height = img.size

                if not img_format or img_format not in FORMAT_TO_MIME:
                    raise InvalidImageFormatError(
                        message=f"Unrecognized Pillow format: {img_format}",
                        client_message="Unsupported image file format.",
                    )

                actual_mime = FORMAT_TO_MIME[img_format]
                if actual_mime not in settings.ALLOWED_IMAGE_FORMATS:
                    raise InvalidImageFormatError(
                        message=f"MIME type '{actual_mime}' not in allowed list.",
                        client_message=f"Image type '{actual_mime}' is not supported.",
                    )

                # Guard against decompression bomb exploits
                total_pixels = width * height
                if total_pixels > settings.MAX_IMAGE_PIXELS:
                    raise DecompressionBombError(
                        message=f"Image resolution {width}x{height} ({total_pixels} pixels) exceeds limit.",
                        client_message="Image resolution is excessively high. Please upload a standard-sized photo.",
                    )

                if width > settings.MAX_IMAGE_WIDTH or height > settings.MAX_IMAGE_HEIGHT:
                    raise DecompressionBombError(
                        message=f"Dimensions {width}x{height} exceed limits ({settings.MAX_IMAGE_WIDTH}x{settings.MAX_IMAGE_HEIGHT}).",
                        client_message="Image dimensions exceed maximum allowed bounds.",
                    )

                # Light verification to catch truncated file structures
                img.verify()

                return ImageMetadata(
                    width=width,
                    height=height,
                    format=img_format,
                    mime_type=actual_mime,
                    size_bytes=len(raw_bytes),
                    aspect_ratio=round(width / height, 4) if height > 0 else 0.0,
                )

        except (Image.DecompressionBombError, DecompressionBombError) as exc:
            raise DecompressionBombError(
                message=f"Decompression bomb safety triggered: {exc}",
                client_message="Image file rejected due to excessive decompression size.",
            ) from exc
        except (InvalidImageFormatError, ImageSizeExceededError):
            raise
        except Exception as exc:
            raise CorruptImageError(
                message=f"Failed to parse image headers: {exc}",
                client_message="Image file appears corrupted and could not be read.",
            ) from exc


image_validator = ImageValidator()