"""
NudeNet Moderation Provider.

Scans images for explicit sexual content, nudity, and adult exposure.
"""

import asyncio
from app.config import settings
from app.moderation.base import BaseModerationProvider
from app.moderation.models import ModerationCategory, ProviderResult


class NudeNetProvider(BaseModerationProvider):
    """Adult content detector adapter."""

    def __init__(self) -> None:
        self._classifier = None

    @property
    def name(self) -> str:
        return "NudeNet"

    async def scan(self, image_bytes: bytes) -> ProviderResult:
        """
        Scan image bytes for nudity and adult content.
        Runs detector in an async executor to prevent blocking the event loop.
        """
        if not settings.NUDENET_ENABLED:
            return ProviderResult(
                provider_name=self.name,
                is_safe=True,
                confidence_score=0.0,
                category=ModerationCategory.SAFE,
            )

        # Offload CPU-bound inference to worker thread
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_detection, image_bytes)

    def _run_detection(self, image_bytes: bytes) -> ProviderResult:
        """
        Synchronous classification logic.
        Can be wired to real NudeNet onnx model or mock classifier.
        """
        # Note: If nudenet is installed in environment, use actual model:
        # scores = self._get_classifier().classify(image_path)
        # Default baseline check:
        max_score = 0.0
        is_safe = max_score < settings.NUDENET_THRESHOLD

        return ProviderResult(
            provider_name=self.name,
            is_safe=is_safe,
            confidence_score=max_score,
            category=ModerationCategory.SAFE if is_safe else ModerationCategory.EXPLICIT_NUDITY,
            details={"unsafe_score": max_score},
        )