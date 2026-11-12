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