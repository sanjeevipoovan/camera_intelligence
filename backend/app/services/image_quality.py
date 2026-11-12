"""
Image Quality Assessment Service.

Evaluates image sharpness via Laplacian edge variance and verifies
lighting conditions via mean luminance before invoking model inference.
"""

import io
import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.core.exceptions import (
    ImageDimensionsTooSmallError,
    ImageTooBlurryError,
    ImageTooDarkError,
)
from app.services.image_models import ProcessedImage, QualityAssessment


class ImageQualityService:
    """Evaluates whether an image meets visual standards for AI reasoning."""

    def assess_quality(self, processed: ProcessedImage) -> QualityAssessment:
        """
        Evaluate image sharpness, brightness, and dimensions.

        Args:
            processed: The normalized ProcessedImage output.

        Returns:
            QualityAssessment containing raw scores, pass/fail booleans, and diagnostic notes.
        """
        rejection_reasons: list[str] = []

        # 1. Dimension validation
        dimensions_passed = (
            processed.width >= settings.IMAGE_MIN_WIDTH
            and processed.height >= settings.IMAGE_MIN_HEIGHT
        )
        if not dimensions_passed:
            rejection_reasons.append(
                f"Image dimensions ({processed.width}x{processed.height}) are below "
                f"minimum requirements ({settings.IMAGE_MIN_WIDTH}x{settings.IMAGE_MIN_HEIGHT})."
            )

        # 2. Decode raw normalized bytes into OpenCV grayscale array
        cv_gray = self._to_opencv_grayscale(processed.data)

        # 3. Blur calculation (Laplacian variance)
        blur_score = self._calculate_blur_score(cv_gray)
        blur_passed = blur_score >= settings.BLUR_THRESHOLD
        if not blur_passed:
            rejection_reasons.append(
                f"Image is too blurry (sharpness score {blur_score:.1f} is below "
                f"minimum threshold {settings.BLUR_THRESHOLD:.1f})."
            )

        # 4. Brightness calculation (Mean luminance)
        brightness_score = self._calculate_brightness_score(cv_gray)
        brightness_passed = brightness_score >= settings.BRIGHTNESS_MIN_THRESHOLD
        if not brightness_passed:
            rejection_reasons.append(
                f"Image is too dark (brightness score {brightness_score:.1f} is below "
                f"minimum threshold {settings.BRIGHTNESS_MIN_THRESHOLD:.1f})."
            )

        is_acceptable = dimensions_passed and blur_passed and brightness_passed

        return QualityAssessment(
            is_acceptable=is_acceptable,
            blur_score=round(blur_score, 2),
            blur_passed=blur_passed,
            brightness_score=round(brightness_score, 2),
            brightness_passed=brightness_passed,
            dimensions_passed=dimensions_passed,
            rejection_reasons=rejection_reasons,
        )

    def validate_or_raise(self, processed: ProcessedImage) -> QualityAssessment:
        """
        Evaluate quality and raise typed domain exceptions on failure.

        Raises:
            ImageDimensionsTooSmallError: If resolution is insufficient.
            ImageTooBlurryError: If Laplacian variance is below threshold.
            ImageTooDarkError: If mean luminance is below threshold.
        """
        if not settings.IMAGE_QUALITY_CHECK_ENABLED:
            return QualityAssessment(
                is_acceptable=True,
                blur_score=999.0,
                blur_passed=True,
                brightness_score=255.0,
                brightness_passed=True,
                dimensions_passed=True,
                rejection_reasons=[],
            )

        assessment = self.assess_quality(processed)
        if assessment.is_acceptable:
            return assessment

        # Raise the specific failure for proper client feedback
        if not assessment.dimensions_passed:
            raise ImageDimensionsTooSmallError(
                message=assessment.rejection_reasons[0],
                client_message="Image resolution is too low. Please move closer or check camera settings.",
            )
        if not assessment.blur_passed:
            raise ImageTooBlurryError(
                message=assessment.rejection_reasons[0],
                client_message="The photo is too blurry to inspect. Please hold your phone steady and retake.",
            )
        if not assessment.brightness_passed:
            raise ImageTooDarkError(
                message=assessment.rejection_reasons[0],
                client_message="The photo is too dark. Please enable flash or move to a better-lit area.",
            )

        return assessment

    def _to_opencv_grayscale(self, image_bytes: bytes) -> np.ndarray:
        """Convert binary image data into a single-channel 8-bit grayscale numpy array."""
        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(np_buffer, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            # Fallback using Pillow if OpenCV decode fails on platform-specific encodings
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                return np.array(pil_img.convert("L"))
        return decoded

    def _calculate_blur_score(self, gray_image: np.ndarray) -> float:
        """
        Calculate focus sharpness via the variance of the 2D Laplacian operator.
        Sharp images with distinct edges yield higher variance.
        """
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return float(laplacian.var())

    def _calculate_brightness_score(self, gray_image: np.ndarray) -> float:
        """
        Calculate average scene illumination as mean pixel luminance [0.0 - 255.0].
        """
        return float(np.mean(gray_image))


image_quality_service = ImageQualityService()