"""Data schemas for inference requests, generated responses, and execution telemetry."""

from pydantic import BaseModel, ConfigDict, Field


class InferenceRequest(BaseModel):
    """Input payload container for the vision-language inference pipeline."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1, max_length=2000, description="User question or instruction")
    max_new_tokens: int | None = Field(default=None, description="Generation token limit override")
    temperature: float | None = Field(default=None, description="Sampling randomness override")
    top_p: float | None = Field(default=None, description="Nucleus sampling threshold override")


class InferenceMetrics(BaseModel):
    """Structured performance and resource metrics for a single inference call."""

    model_config = ConfigDict(frozen=True)

    queue_wait_time_ms: float = Field(description="Time spent waiting for GPU slot in milliseconds")
    inference_time_ms: float = Field(description="Time spent in forward pass in milliseconds")
    total_time_ms: float = Field(description="Total server processing duration in milliseconds")
    tokens_generated: int = Field(description="Number of output tokens produced")
    tokens_per_second: float = Field(description="Generation throughput rate")
    device_used: str = Field(description="Execution device (cuda, cpu)")


class InferenceResponse(BaseModel):
    """Final answer returned to the client application."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Generated textual answer from the vision model")
    metrics: InferenceMetrics = Field(description="Execution performance telemetry")