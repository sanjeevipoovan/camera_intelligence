"""
Image Preprocessing Service.

Handles EXIF auto-rotation, alpha compositing, RGB normalization,
aspect-ratio preserved downsampling, and output encoding.
"""

import io
from PIL import Image, ImageOps

from app.config import settings
from app.core.exceptions import ImageProcessingError
from app.services.image_models import ImageMetadata, ProcessedImage


class ImagePreprocessor:
    """Prepares verified image payloads for AI model ingestion."""

    def process(self, raw_bytes: bytes, metadata: ImageMetadata) -> ProcessedImage:
        """
        Execute full normalization pipeline on raw image bytes.

        Args:
            raw_bytes: Raw binary image payload.
            metadata: Verified technical metadata from ImageValidator.

        Returns:
            ProcessedImage containing normalized binary data, dimensions, and channel info.
        """
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                # 1. Correct mobile orientation via EXIF metadata
                if settings.IMAGE_AUTO_ORIENTATION:
                    img = ImageOps.exif_transpose(img) or img

                # 2. Normalize color space and alpha channels
                img = self._normalize_color_channels(img)

                # 3. Downscale if dimensions exceed MAX_LONG_EDGE
                if settings.IMAGE_RESIZE_ENABLED:
                    img = self._downscale_aspect_preserving(img)

                # 4. Serialize to standardized target format (e.g. JPEG)
                output_bytes = self._encode_image(img)

                return ProcessedImage(
                    data=output_bytes,
                    format=settings.IMAGE_OUTPUT_FORMAT.value,
                    width=img.width,
                    height=img.height,
                    channels=len(img.getbands()),
                    metadata=metadata,
                )

        except Exception as exc:
            raise ImageProcessingError(
                message=f"Unexpected error in preprocessing pipeline: {exc}",
                client_message="Failed to process image for model analysis.",
            ) from exc

    def _normalize_color_channels(self, img: Image.Image) -> Image.Image:
        """
        Convert image to clean 3-channel RGB.
        Composites transparent layers onto solid white backgrounds.
        """
        if not settings.IMAGE_CONVERT_RGB:
            return img

        # If image contains transparency (RGBA, LA) or palette transparency (P)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Create a solid white background image
            rgba_image = img.convert("RGBA")
            background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
            # Alpha composite the image on top of white
            composite = Image.alpha_composite(background, rgba_image)
            return composite.convert("RGB")

        # Convert grayscale, CMYK, or palette modes directly to RGB
        if img.mode != "RGB":
            return img.convert("RGB")

        return img

    def _downscale_aspect_preserving(self, img: Image.Image) -> Image.Image:
        """
        Downscale image so the longest edge does not exceed IMAGE_MAX_LONG_EDGE.
        Uses Lanczos resampling to preserve edge fidelity for OCR and small details.
        """
        width, height = img.size
        longest_edge = max(width, height)

        if longest_edge <= settings.IMAGE_MAX_LONG_EDGE:
            return img

        scaling_factor = settings.IMAGE_MAX_LONG_EDGE / float(longest_edge)
        new_width = max(1, int(round(width * scaling_factor)))
        new_height = max(1, int(round(height * scaling_factor)))

        return img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    def _encode_image(self, img: Image.Image) -> bytes:
        """Encode normalized Pillow image into output bytes."""
        buffer = io.BytesIO()
        target_format = settings.IMAGE_OUTPUT_FORMAT.value.upper()

        if target_format in ("JPEG", "JPG"):
            img.save(
                buffer,
                format="JPEG",
                quality=settings.IMAGE_JPEG_QUALITY,
                optimize=True,
            )
        elif target_format == "WEBP":
            img.save(
                buffer,
                format="WEBP",
                quality=settings.IMAGE_JPEG_QUALITY,
                method=6,  # Highest quality compression effort
            )
        elif target_format == "PNG":
            img.save(buffer, format="PNG", optimize=True)
        else:
            img.save(buffer, format=target_format)

        return buffer.getvalue()


image_preprocessor = ImagePreprocessor()