"""
Unit tests for Phase 3: Inference & Concurrency Queue Subsystem.

Tests generation output, concurrency throttling, queue timeout handling,
and latency telemetry calculation.
"""

import asyncio
import time
import pytest

from app.core.exceptions import InferenceQueueTimeoutError, ModelTimeoutError
from app.inference.engine import LiquidAIEngine
from app.inference.models import InferenceRequest
from app.inference.queue_manager import InferenceQueueManager
from app.inference.service import InferenceService
from app.services.image_models import ImageMetadata, ProcessedImage


# ==============================================================================
# In-Memory Test Fixtures
# ==============================================================================

@pytest.fixture
def dummy_processed_image() -> ProcessedImage:
    """Fixture providing a verified ProcessedImage instance."""
    return ProcessedImage(
        data=b"fake_jpeg_image_data",
        format="JPEG",
        width=400,
        height=400,
        channels=3,
        metadata=ImageMetadata(
            width=400,
            height=400,
            format="JPEG",
            mime_type="image/jpeg",
            size_bytes=20,
            aspect_ratio=1.0,
        ),
    )


class SlowMockEngine(LiquidAIEngine):
    """Engine mock that sleeps to simulate real forward-pass computation time."""

    def __init__(self, delay_seconds: float = 0.2) -> None:
        super().__init__()
        self.delay_seconds = delay_seconds
        self._is_loaded = True

    def generate(self, *args, **kwargs) -> tuple[str, int]:
        time.sleep(self.delay_seconds)
        return "Simulated answer from slow engine", 20


# ==============================================================================
# Inference Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_single_inference_success(dummy_processed_image: ProcessedImage) -> None:
    service = InferenceService()
    request = InferenceRequest(prompt="What color is the object?")

    response = await service.generate_answer(dummy_processed_image, request)

    assert response.text != ""
    assert "What color is the object?" in response.text
    assert response.metrics.tokens_generated > 0
    assert response.metrics.total_time_ms > 0.0


@pytest.mark.asyncio
async def test_concurrency_queue_throttling(dummy_processed_image: ProcessedImage) -> None:
    # Max concurrency = 1, so tasks run sequentially
    slow_engine = SlowMockEngine(delay_seconds=0.1)
    queue = InferenceQueueManager(max_concurrent=1, queue_timeout=2.0)
    service = InferenceService(engine=slow_engine, queue=queue)

    request = InferenceRequest(prompt="Inspect this item")

    # Launch two requests simultaneously
    start_time = time.perf_counter()
    res1, res2 = await asyncio.gather(
        service.generate_answer(dummy_processed_image, request),
        service.generate_answer(dummy_processed_image, request),
    )
    total_elapsed = time.perf_counter() - start_time

    # Each takes 0.1s; because concurrency is 1, total elapsed time must be >= 0.2s
    assert total_elapsed >= 0.18
    # Second request had to wait in the queue for the first one to finish
    assert (res1.metrics.queue_wait_time_ms > 50.0) or (res2.metrics.queue_wait_time_ms > 50.0)


@pytest.mark.asyncio
async def test_inference_queue_timeout(dummy_processed_image: ProcessedImage) -> None:
    # Capacity = 1, task duration = 0.3s, but queue timeout = 0.05s
    slow_engine = SlowMockEngine(delay_seconds=0.3)
    queue = InferenceQueueManager(max_concurrent=1, queue_timeout=0.05)
    service = InferenceService(engine=slow_engine, queue=queue)

    request = InferenceRequest(prompt="Inspect this item")

    # First task occupies the only slot
    task1 = asyncio.create_task(service.generate_answer(dummy_processed_image, request))
    await asyncio.sleep(0.01)  # Ensure task1 has acquired the slot

    # Second task tries to acquire the slot and must time out
    with pytest.raises(InferenceQueueTimeoutError):
        await service.generate_answer(dummy_processed_image, request)

    await task1  # Clean up first task


@pytest.mark.asyncio
async def test_inference_model_timeout(dummy_processed_image: ProcessedImage, monkeypatch: pytest.MonkeyPatch) -> None:
    # Model forward pass takes 0.3s, but MODEL_TIMEOUT is set to 0.05s
    monkeypatch.setattr("app.config.settings.MODEL_TIMEOUT", 0.05)
    slow_engine = SlowMockEngine(delay_seconds=0.3)
    service = InferenceService(engine=slow_engine)

    request = InferenceRequest(prompt="Inspect this item")

    with pytest.raises(ModelTimeoutError):
        await service.generate_answer(dummy_processed_image, request)