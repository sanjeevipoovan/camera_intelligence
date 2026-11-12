"""
Unit tests for Phase 2: Moderation Subsystem.

Tests provider aggregation, threshold triggers, fail-closed mechanics,
timeout handling, and bypass behaviors.
"""

import asyncio
import pytest

from app.config import ModerationFailPolicy
from app.core.exceptions import (
    ModerationServiceUnavailableError,
    ModerationTimeoutError,
    UnsafeContentError,
)
from app.moderation.base import BaseModerationProvider
from app.moderation.models import ModerationCategory, ProviderResult
from app.moderation.service import ModerationService


# Mock Providers for deterministic testing
class MockSafeProvider(BaseModerationProvider):
    @property
    def name(self) -> str:
        return "MockSafe"

    async def scan(self, image_bytes: bytes) -> ProviderResult:
        return ProviderResult(
            provider_name=self.name,
            is_safe=True,
            confidence_score=0.05,
            category=ModerationCategory.SAFE,
        )


class MockExplicitProvider(BaseModerationProvider):
    @property
    def name(self) -> str:
        return "MockExplicit"

    async def scan(self, image_bytes: bytes) -> ProviderResult:
        return ProviderResult(
            provider_name=self.name,
            is_safe=False,
            confidence_score=0.92,
            category=ModerationCategory.EXPLICIT_NUDITY,
        )


class MockCrashingProvider(BaseModerationProvider):
    @property
    def name(self) -> str:
        return "MockCrash"

    async def scan(self, image_bytes: bytes) -> ProviderResult:
        raise RuntimeError("GPU Driver Crashed")


class MockHangingProvider(BaseModerationProvider):
    @property
    def name(self) -> str:
        return "MockHang"

    async def scan(self, image_bytes: bytes) -> ProviderResult:
        await asyncio.sleep(5.0)  # Exceeds timeout
        return ProviderResult(
            provider_name=self.name,
            is_safe=True,
            confidence_score=0.0,
            category=ModerationCategory.SAFE,
        )


@pytest.mark.asyncio
async def test_safe_image_clears_moderation() -> None:
    service = ModerationService(providers=[MockSafeProvider()])
    verdict = await service.moderate(b"fake_image_bytes")
    assert verdict.is_safe is True
    assert verdict.highest_risk_score == 0.0
    assert len(verdict.flagged_by) == 0


@pytest.mark.asyncio
async def test_explicit_image_gets_flagged() -> None:
    service = ModerationService(providers=[MockSafeProvider(), MockExplicitProvider()])
    verdict = await service.moderate(b"fake_image_bytes")
    assert verdict.is_safe is False
    assert "MockExplicit" in verdict.flagged_by
    assert verdict.highest_risk_category == ModerationCategory.EXPLICIT_NUDITY
    assert verdict.highest_risk_score == 0.92

    with pytest.raises(UnsafeContentError):
        await service.moderate_or_raise(b"fake_image_bytes")


@pytest.mark.asyncio
async def test_fail_closed_on_provider_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.MODERATION_FAIL_POLICY", ModerationFailPolicy.FAIL_CLOSED)
    service = ModerationService(providers=[MockCrashingProvider()])

    verdict = await service.moderate(b"fake_image_bytes")
    assert verdict.is_safe is False
    assert "service error" in verdict.rejection_reason

    with pytest.raises(ModerationServiceUnavailableError):
        await service.moderate_or_raise(b"fake_image_bytes")


@pytest.mark.asyncio
async def test_fail_open_on_provider_crash_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.MODERATION_FAIL_POLICY", ModerationFailPolicy.FAIL_OPEN)
    service = ModerationService(providers=[MockCrashingProvider()])

    verdict = await service.moderate(b"fake_image_bytes")
    assert verdict.is_safe is True  # Permitted under fail_open


@pytest.mark.asyncio
async def test_timeout_triggers_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.MODERATION_TIMEOUT", 0.05)  # 50ms timeout
    monkeypatch.setattr("app.config.settings.MODERATION_FAIL_POLICY", ModerationFailPolicy.FAIL_CLOSED)
    service = ModerationService(providers=[MockHangingProvider()])

    verdict = await service.moderate(b"fake_image_bytes")
    assert verdict.is_safe is False
    assert "timed out" in verdict.rejection_reason

    with pytest.raises(ModerationTimeoutError):
        await service.moderate_or_raise(b"fake_image_bytes")


@pytest.mark.asyncio
async def test_global_moderation_disabled_bypasses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.IMAGE_MODERATION_ENABLED", False)
    # Even with explicit provider, it must clear because moderation is disabled
    service = ModerationService(providers=[MockExplicitProvider()])

    verdict = await service.moderate(b"fake_image_bytes")
    assert verdict.is_safe is True