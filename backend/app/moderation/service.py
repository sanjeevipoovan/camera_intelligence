"""
Moderation Orchestration Service.

Coordinates parallel provider execution, enforces timeout boundaries,
and applies fail-closed security postures.
"""

import asyncio
import logging
from app.config import ModerationFailPolicy, settings
from app.core.exceptions import (
    ModerationServiceUnavailableError,
    ModerationTimeoutError,
    UnsafeContentError,
)
from app.moderation.base import BaseModerationProvider
from app.moderation.models import (
    ModerationCategory,
    ModerationVerdict,
    ProviderResult,
)
from app.moderation.nudenet_provider import NudeNetProvider
from app.moderation.safety_provider import SafetyClassifierProvider

logger = logging.getLogger("app.moderation")


class ModerationService:
    """Master moderation orchestrator."""

    def __init__(self, providers: list[BaseModerationProvider] | None = None) -> None:
        self.providers = providers or [
            NudeNetProvider(),
            SafetyClassifierProvider(),
        ]

    async def moderate(self, image_bytes: bytes) -> ModerationVerdict:
        """
        Scan image using all active providers in parallel within MODERATION_TIMEOUT.
        """
        # Short-circuit if moderation is globally disabled
        if not settings.IMAGE_MODERATION_ENABLED:
            return ModerationVerdict(
                is_safe=True,
                highest_risk_category=ModerationCategory.SAFE,
                highest_risk_score=0.0,
                rejection_reason=None,
            )

        tasks = [provider.scan(image_bytes) for provider in self.providers]

        try:
            results: list[ProviderResult] = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=False),
                timeout=settings.MODERATION_TIMEOUT,
            )
        except TimeoutError as exc:
            logger.error("Moderation timed out after %.2f seconds", settings.MODERATION_TIMEOUT)
            return self._handle_provider_failure(
                reason=f"Moderation timed out after {settings.MODERATION_TIMEOUT}s",
                is_timeout=True,
            )
        except Exception as exc:
            logger.error("Moderation provider crashed: %s", exc)
            return self._handle_provider_failure(
                reason=f"Moderation service error: {exc}",
                is_timeout=False,
            )

        return self._aggregate_results(results)

    async def moderate_or_raise(self, image_bytes: bytes) -> ModerationVerdict:
        """
        Run moderation scan and raise domain exceptions if unsafe or failed.
        Never allows unsafe images through to primary vision model.
        """
        verdict = await self.moderate(image_bytes)
        if verdict.is_safe:
            return verdict

        # Handle system failures under fail-closed policy
        if verdict.rejection_reason and "timed out" in verdict.rejection_reason:
            raise ModerationTimeoutError(
                message=verdict.rejection_reason,
                client_message="Image safety verification timed out. Please try again.",
            )

        if verdict.rejection_reason and "service error" in verdict.rejection_reason:
            raise ModerationServiceUnavailableError(
                message=verdict.rejection_reason,
                client_message="Image safety verification is temporarily unavailable.",
            )

        # Standard unsafe content rejection
        raise UnsafeContentError(
            message=f"Image rejected: Flagged by {verdict.flagged_by} for {verdict.highest_risk_category}",
            client_message="Your photo was flagged as potentially sensitive or unsafe and cannot be processed.",
            category=verdict.highest_risk_category.value,
        )

    def _aggregate_results(self, results: list[ProviderResult]) -> ModerationVerdict:
        """Combine scores from multiple providers into a single final verdict."""
        flagged_by: list[str] = []
        max_score: float = 0.0
        highest_category = ModerationCategory.SAFE

        for res in results:
            if not res.is_safe:
                flagged_by.append(res.provider_name)
                if res.confidence_score > max_score:
                    max_score = res.confidence_score
                    highest_category = res.category

        is_safe = len(flagged_by) == 0

        return ModerationVerdict(
            is_safe=is_safe,
            flagged_by=flagged_by,
            highest_risk_category=highest_category,
            highest_risk_score=round(max_score, 4),
            provider_results=results,
            rejection_reason=None if is_safe else f"Flagged by {', '.join(flagged_by)}",
        )

    def _handle_provider_failure(self, reason: str, is_timeout: bool) -> ModerationVerdict:
        """Enforce configured fail-closed or fail-open policy."""
        if settings.MODERATION_FAIL_POLICY == ModerationFailPolicy.FAIL_CLOSED:
            return ModerationVerdict(
                is_safe=False,
                highest_risk_category=ModerationCategory.UNSAFE_GENERAL,
                highest_risk_score=1.0,
                rejection_reason=reason,
            )

        # Fail open (only allowed in development/testing if explicitly configured)
        logger.warning("Moderation failed but continuing due to fail_open policy: %s", reason)
        return ModerationVerdict(
            is_safe=True,
            highest_risk_category=ModerationCategory.SAFE,
            highest_risk_score=0.0,
            rejection_reason=f"Ignored failure due to policy: {reason}",
        )


moderation_service = ModerationService()