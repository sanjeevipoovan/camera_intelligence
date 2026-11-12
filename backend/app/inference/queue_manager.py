"""
Inference Concurrency and Queue Manager.

Guards GPU VRAM against concurrent execution overload using an asynchronous
semaphore with strict timeout boundaries.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from app.config import settings
from app.core.exceptions import InferenceQueueTimeoutError


class InferenceQueueManager:
    """Manages access to GPU compute slots to avoid memory thrashing."""

    def __init__(self, max_concurrent: int | None = None, queue_timeout: float | None = None) -> None:
        self.max_concurrent = max_concurrent or settings.MAX_CONCURRENT_INFERENCES
        self.queue_timeout = queue_timeout or settings.INFERENCE_QUEUE_TIMEOUT
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._active_inferences = 0
        self._queued_requests = 0

    @property
    def active_inferences(self) -> int:
        """Current number of requests executing on the model."""
        return self._active_inferences

    @property
    def queued_requests(self) -> int:
        """Current number of requests waiting for a free GPU slot."""
        return self._queued_requests

    @asynccontextmanager
    async def acquire_slot(self) -> AsyncGenerator[float, None]:
        """
        Asynchronously acquire an inference execution slot.

        Yields:
            wait_time_ms: Milliseconds spent waiting in the queue.

        Raises:
            InferenceQueueTimeoutError: If slot cannot be acquired within timeout.
        """
        if not settings.INFERENCE_QUEUE_ENABLED:
            yield 0.0
            return

        start_wait = time.perf_counter()
        self._queued_requests += 1

        try:
            # Wait for an available slot within the configured timeout
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.queue_timeout)
        except TimeoutError as exc:
            wait_time = (time.perf_counter() - start_wait) * 1000.0
            raise InferenceQueueTimeoutError(
                message=f"Inference queue timeout after {wait_time:.1f}ms (queue limit: {self.queue_timeout}s)."
            ) from exc
        finally:
            self._queued_requests -= 1

        wait_time_ms = (time.perf_counter() - start_wait) * 1000.0
        self._active_inferences += 1

        try:
            yield wait_time_ms
        finally:
            self._active_inferences -= 1
            self._semaphore.release()


queue_manager = InferenceQueueManager()