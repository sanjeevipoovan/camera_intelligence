"""Data schemas for moderation results, scores, and policy verdicts."""

from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field


class ModerationCategory(StrEnum):
    SAFE = "safe"
    EXPLICIT_NUDITY = "explicit_nudity"
    VIOLENCE = "violence"
    HATE_SYMBOLS = "hate_symbols"
    UNSAFE_GENERAL = "unsafe_general"


class ProviderResult(BaseModel):
    """Result returned by a single moderation provider."""

    model_config = ConfigDict(frozen=True)

    provider_name: str
    is_safe: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    category: ModerationCategory
    details: dict[str, float] = Field(default_factory=dict)


class ModerationVerdict(BaseModel):
    """Aggregated final safety decision returned by the ModerationService."""

    model_config = ConfigDict(frozen=True)

    is_safe: bool = Field(description="Whether the image is cleared for model inference")
    flagged_by: list[str] = Field(default_factory=list, description="List of providers that flagged the image")
    highest_risk_category: ModerationCategory = ModerationCategory.SAFE
    highest_risk_score: float = 0.0
    provider_results: list[ProviderResult] = Field(default_factory=list)
    rejection_reason: str | None = None