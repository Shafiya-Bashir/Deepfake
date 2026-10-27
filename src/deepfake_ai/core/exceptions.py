"""Custom exception hierarchy for DeepFake-AI."""

class DeepfakeAIError(Exception):
    """Base exception for all DeepFake-AI errors."""
    pass

class ConfigurationError(DeepfakeAIError):
    """Raised when configuration is invalid or missing."""
    pass

class DataProcessingError(DeepfakeAIError):
    """Raised when data ingestion, video decoding, or splitting fails."""
    pass

class FaceDetectionError(DataProcessingError):
    """Raised when face detection or landmark alignment fails on media."""
    pass

class ModelError(DeepfakeAIError):
    """Raised when model instantiation, forward pass, or weight loading fails."""
    pass

class HardwareError(DeepfakeAIError):
    """Raised when requested hardware device is unavailable or runs out of memory."""
    pass

class CheckpointError(DeepfakeAIError):
    """Raised when saving or loading model checkpoints fails."""
    pass
