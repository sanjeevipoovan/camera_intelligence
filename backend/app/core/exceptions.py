"""Domain-specific exceptions for image validation, preprocessing, and quality checks."""


class ImageProcessingError(Exception):
    """Base class for all image processing pipeline exceptions."""

    def __init__(self, message: str, client_message: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.client_message = client_message or message


class ImageValidationError(ImageProcessingError):
    """Raised when an uploaded file violates format, size, or safety constraints."""


class InvalidImageFormatError(ImageValidationError):
    """Raised when file headers or MIME types are invalid or unsupported."""


class ImageSizeExceededError(ImageValidationError):
    """Raised when raw payload byte size exceeds configured ceilings."""


class DecompressionBombError(ImageValidationError):
    """Raised when pixel dimensions or memory expansion threshold is breached."""


class CorruptImageError(ImageValidationError):
    """Raised when binary image data is truncated or cannot be parsed."""


class ImageQualityError(ImageProcessingError):
    """Base class for visual quality failures preventing reliable AI inference."""


class ImageTooBlurryError(ImageQualityError):
    """Raised when Laplacian edge variance falls below acceptable sharpness."""


class ImageTooDarkError(ImageQualityError):
    """Raised when scene luminance falls below minimum illumination thresholds."""


class ImageDimensionsTooSmallError(ImageQualityError):
    """Raised when image resolution is insufficient for vision-language encoding."""
"""Domain-specific exceptions for image validation, preprocessing, quality, and moderation."""


class ImageProcessingError(Exception):
    """Base class for all image processing pipeline exceptions."""

    def __init__(self, message: str, client_message: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.client_message = client_message or message


class ImageValidationError(ImageProcessingError):
    """Raised when an uploaded file violates format, size, or safety constraints."""


class InvalidImageFormatError(ImageValidationError):
    """Raised when file headers or MIME types are invalid or unsupported."""


class ImageSizeExceededError(ImageValidationError):
    """Raised when raw payload byte size exceeds configured ceilings."""


class DecompressionBombError(ImageValidationError):
    """Raised when pixel dimensions or memory expansion threshold is breached."""


class CorruptImageError(ImageValidationError):
    """Raised when binary image data is truncated or cannot be parsed."""


class ImageQualityError(ImageProcessingError):
    """Base class for visual quality failures preventing reliable AI inference."""


class ImageTooBlurryError(ImageQualityError):
    """Raised when Laplacian edge variance falls below acceptable sharpness."""


class ImageTooDarkError(ImageQualityError):
    """Raised when scene luminance falls below minimum illumination thresholds."""


class ImageDimensionsTooSmallError(ImageQualityError):
    """Raised when image resolution is insufficient for vision-language encoding."""


# ==============================================================================
# Phase 2: Moderation Domain Exceptions
# ==============================================================================

class ModerationError(ImageProcessingError):
    """Base class for safety and content moderation rejections."""


class UnsafeContentError(ModerationError):
    """Raised when an image triggers an explicit, violent, or banned content threshold."""

    def __init__(self, message: str, client_message: str | None = None, category: str = "general") -> None:
        super().__init__(message, client_message or "The uploaded image was flagged by automated safety filters.")
        self.category = category


class ModerationTimeoutError(ModerationError):
    """Raised when moderation inspection exceeds MODERATION_TIMEOUT."""


class ModerationServiceUnavailableError(ModerationError):
    """Raised when moderation backend fails or crashes under a fail-closed policy."""
# (Append these to the bottom of app/core/exceptions.py)

# ==============================================================================
# Phase 3: Inference Domain Exceptions
# ==============================================================================

class ModelInferenceError(ImageProcessingError):
    """Base class for vision-language inference failures."""


class ModelNotLoadedError(ModelInferenceError):
    """Raised when inference is attempted before weights are loaded into memory."""


class ModelTimeoutError(ModelInferenceError):
    """Raised when an inference forward pass exceeds MODEL_TIMEOUT."""


class InferenceQueueTimeoutError(ModelInferenceError):
    """Raised when a request waits in the concurrency queue longer than INFERENCE_QUEUE_TIMEOUT."""

    def __init__(self, message: str, client_message: str | None = None) -> None:
        super().__init__(
            message,
            client_message or "The inference server is currently experiencing high load. Please try again in a moment.",
        )


class InferenceQueueFullError(ModelInferenceError):
    """Raised when the maximum queue capacity is exceeded."""