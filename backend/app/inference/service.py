"""
Inference Service Coordinator.

Combines concurrency queuing, thread offloading, timeout enforcement,
and telemetry calculations into a single clean API.
"""

import asyncio
import time
from app.config import settings
from app.core.exceptions import ModelTimeoutError
from app.inference.engine import LiquidAIEngine, engine as default_engine
from app.inference.models import InferenceMetrics, InferenceRequest, InferenceResponse
from app.inference.queue_manager import InferenceQueueManager, queue_manager as default_queue_manager
from app.services.image_models import ProcessedImage


class InferenceService:
    """Master orchestrator for model inference passes."""

    def __init__(
        self,
        engine: LiquidAIEngine | None = None,
        queue: InferenceQueueManager | None = None,
    ) -> None:
        self.engine = engine or default_engine
        self.queue = queue or default_queue_manager

    async def generate_answer(
        self,
        processed_image: ProcessedImage,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """
        Execute an end-to-end question-answering forward pass.

        1. Acquires a GPU queue slot (guarded by timeout).
        2. Executes inference in a background thread.
        3. Enforces MODEL_TIMEOUT during generation.
        4. Calculates timing and throughput metrics.
        """
        overall_start = time.perf_counter()

        # Step 1: Guard GPU with concurrency semaphore
        async with self.queue.acquire_slot() as queue_wait_time_ms:
            inference_start = time.perf_counter()

            # Step 2: Offload CPU/GPU forward pass to background thread
            loop = asyncio.get_running_loop()
            try:
                answer_text, token_count = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        self.engine.generate,
                        processed_image.data,
                        request.prompt,
                        request.max_new_tokens,
                        request.temperature,
                        request.top_p,
                    ),
                    timeout=settings.MODEL_TIMEOUT,
                )
            except TimeoutError as exc:
                raise ModelTimeoutError(
                    message=f"Model inference timed out after {settings.MODEL_TIMEOUT}s.",
                    client_message="Analysis took longer than permitted. Please try asking a more specific question.",
                ) from exc

            inference_duration_ms = (time.perf_counter() - inference_start) * 1000.0

        total_duration_ms = (time.perf_counter() - overall_start) * 1000.0

        # Calculate throughput (tokens per second)
        inf_seconds = inference_duration_ms / 1000.0
        tokens_per_sec = (token_count / inf_seconds) if inf_seconds > 0 else 0.0

        metrics = InferenceMetrics(
            queue_wait_time_ms=round(queue_wait_time_ms, 2),
            inference_time_ms=round(inference_duration_ms, 2),
            total_time_ms=round(total_duration_ms, 2),
            tokens_generated=token_count,
            tokens_per_second=round(tokens_per_sec, 2),
            device_used=self.engine.device,
        )

        return InferenceResponse(text=answer_text, metrics=metrics)


inference_service = InferenceService()