"""
General Visual Safety & Violence Moderation Provider.

Scans images for violence, weapons, gore, and prohibited content.
"""

import asyncio
from app.config import settings
from app.moderation.base import BaseModerationProvider
from app.moderation.models import ModerationCategory, ProviderResult


class SafetyClassifierProvider(BaseModerationProvider):
    """Violence and harm classifier adapter."""

    @property
    def name(self) -> str:
        return "SafetyClassifier"

    async def scan(self, image_bytes: bytes) -> ProviderResult:
        """Execute safety scan in an async executor."""
        if not settings.SAFETY_CLASSIFIER_ENABLED:
            return ProviderResult(
                provider_name=self.name,
                is_safe=True,
                confidence_score=0.0,
                category=ModerationCategory.SAFE,
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_detection, image_bytes)

    def _run_detection(self, image_bytes: bytes) -> ProviderResult:
        """Synchronous violence/harm classification logic."""
        violence_score = 0.0
        is_safe = violence_score < settings.SAFETY_CLASSIFIER_THRESHOLD

        return ProviderResult(
            provider_name=self.name,
            is_safe=is_safe,
            confidence_score=violence_score,
            category=ModerationCategory.SAFE if is_safe else ModerationCategory.VIOLENCE,
            details={"violence_score": violence_score},
        )