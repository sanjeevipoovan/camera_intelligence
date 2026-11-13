"""
NudeNet Moderation Provider.

Scans images using NudeNet ONNX detector for explicit nudity and 18+ content.
"""

import asyncio
import logging
import os
import tempfile
from nudenet import NudeDetector

from app.config import settings
from app.moderation.base import BaseModerationProvider
from app.moderation.models import ModerationCategory, ProviderResult

logger = logging.getLogger("app.moderation")

EXPLICIT_CLASSES = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}


class NudeNetProvider(BaseModerationProvider):
    """Adult content detector adapter using NudeNet ONNX."""

    def __init__(self) -> None:
        self._detector: NudeDetector | None = None

    @property
    def name(self) -> str:
        return "NudeNet"

    def load_model(self) -> None:
        """Initialize the NudeNet ONNX detector once."""
        if self._detector is None:
            logger.info("Loading NudeNet ONNX detector...")
            self._detector = NudeDetector()
            logger.info("NudeNet ONNX detector loaded.")

    async def scan(self, image_bytes: bytes) -> ProviderResult:
        """Scan image bytes for nudity and adult content."""
        if not settings.NUDENET_ENABLED:
            return ProviderResult(
                provider_name=self.name,
                is_safe=True,
                confidence_score=0.0,
                category=ModerationCategory.SAFE,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_detection, image_bytes)

    def _run_detection(self, image_bytes: bytes) -> ProviderResult:
        """Run synchronous NudeNet detection and immediately delete temp file."""
        self.load_model()

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

            detections = self._detector.detect(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        explicit_hits = [
            d for d in detections if d.get("class") in EXPLICIT_CLASSES
        ]
        max_score = max(
            (float(d.get("score", 0.0)) for d in explicit_hits),
            default=0.0,
        )
        is_safe = max_score < settings.NUDENET_THRESHOLD

        if not is_safe:
            logger.warning(
                "Blocked unsafe image (score=%.4f, classes=%s)",
                max_score,
                [d.get("class") for d in explicit_hits],
            )

        return ProviderResult(
            provider_name=self.name,
            is_safe=is_safe,
            confidence_score=max_score,
            category=ModerationCategory.SAFE if is_safe else ModerationCategory.EXPLICIT_NUDITY,
            details={
                "unsafe_score": round(max_score, 4),
                "explicit_hits_count": float(len(explicit_hits)),
            },
        )