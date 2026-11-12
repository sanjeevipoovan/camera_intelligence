"""
Image Quality Assessment Service.

Evaluates image sharpness via Laplacian edge variance and verifies
lighting conditions via mean luminance before invoking model inference.
Supports both OpenCV and a pure NumPy fallback.
"""

import io
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

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
        Evaluate image dimensions, lighting, and sharpness.

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

        # 2. Decode raw normalized bytes into grayscale array
        gray_image = self._to_grayscale_array(processed.data)

        # 3. Brightness calculation (Evaluated before blur)
        brightness_score = self._calculate_brightness_score(gray_image)
        brightness_passed = brightness_score >= settings.BRIGHTNESS_MIN_THRESHOLD
        if not brightness_passed:
            rejection_reasons.append(
                f"Image is too dark (brightness score {brightness_score:.1f} is below "
                f"minimum threshold {settings.BRIGHTNESS_MIN_THRESHOLD:.1f})."
            )

        # 4. Blur calculation (Laplacian variance)
        blur_score = self._calculate_blur_score(gray_image)
        blur_passed = blur_score >= settings.BLUR_THRESHOLD
        if not blur_passed:
            rejection_reasons.append(
                f"Image is too blurry (sharpness score {blur_score:.1f} is below "
                f"minimum threshold {settings.BLUR_THRESHOLD:.1f})."
            )

        is_acceptable = dimensions_passed and brightness_passed and blur_passed

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
        Enforces hierarchical inspection order: Dimensions -> Brightness -> Blur.
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

        # 1. Gate: Dimension insufficiency
        if not assessment.dimensions_passed:
            reason = next((r for r in assessment.rejection_reasons if "dimensions" in r), assessment.rejection_reasons[0])
            raise ImageDimensionsTooSmallError(
                message=reason,
                client_message="Image resolution is too low. Please move closer or check camera settings.",
            )

        # 2. Gate: Scene darkness (Checked before blur because darkness causes zero gradients)
        if not assessment.brightness_passed:
            reason = next((r for r in assessment.rejection_reasons if "dark" in r), assessment.rejection_reasons[0])
            raise ImageTooDarkError(
                message=reason,
                client_message="The photo is too dark. Please enable flash or move to a better-lit area.",
            )

        # 3. Gate: Optical blur
        if not assessment.blur_passed:
            reason = next((r for r in assessment.rejection_reasons if "blurry" in r), assessment.rejection_reasons[0])
            raise ImageTooBlurryError(
                message=reason,
                client_message="The photo is too blurry to inspect. Please hold your phone steady and retake.",
            )

        return assessment

    def _to_grayscale_array(self, image_bytes: bytes) -> np.ndarray:
        """Convert binary image data into a single-channel 8-bit grayscale numpy array."""
        if HAS_OPENCV:
            np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(np_buffer, cv2.IMREAD_GRAYSCALE)
            if decoded is not None:
                return decoded

        # Pillow fallback
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            return np.array(pil_img.convert("L"), dtype=np.float64)

    def _calculate_blur_score(self, gray_image: np.ndarray) -> float:
        """
        Calculate focus sharpness via the variance of the 2D Laplacian operator.
        Uses OpenCV when available, with a vectorized NumPy fallback.
        """
        if HAS_OPENCV:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            return float(laplacian.var())

        # Vectorized 2D discrete Laplacian via NumPy
        if gray_image.shape[0] < 3 or gray_image.shape[1] < 3:
            return 0.0

        img = gray_image.astype(np.float64)
        laplacian = (
            img[:-2, 1:-1]
            + img[2:, 1:-1]
            + img[1:-1, :-2]
            + img[1:-1, 2:]
            - 4.0 * img[1:-1, 1:-1]
        )
        return float(np.var(laplacian))

    def _calculate_brightness_score(self, gray_image: np.ndarray) -> float:
        """Calculate average scene illumination as mean pixel luminance [0.0 - 255.0]."""
        return float(np.mean(gray_image))


image_quality_service = ImageQualityService()