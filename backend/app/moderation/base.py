"""Abstract Base Class for pluggable moderation providers."""

from abc import ABC, abstractmethod
from app.moderation.models import ProviderResult


class BaseModerationProvider(ABC):
    """Interface for visual safety classifiers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider identifier."""

    @abstractmethod
    async def scan(self, image_bytes: bytes) -> ProviderResult:
        """
        Scan normalized image bytes and return safety scores.

        Args:
            image_bytes: Normalized image binary data.

        Returns:
            ProviderResult containing safety verdict and scores.
        """