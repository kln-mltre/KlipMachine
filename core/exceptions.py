"""
Custom exceptions for KlipMachine.

Exception hierarchy:
    KlipMachineError
    ├── DownloadError
    │   ├── VideoNotFoundError
    │   └── NetworkError
    ├── TranscriptionError
    │   └── AudioNotFoundError
    ├── AnalysisError
    │   ├── APIError
    │   ├── RateLimitError
    │   └── JSONParseError
    ├── ExportError
    │   ├── FFmpegError
    │   └── InvalidClipError
    └── ConfigError
"""


class KlipMachineError(Exception):
    """Base exception for all KlipMachine errors."""
    pass


# ============================================================================
# DOWNLOAD ERRORS
# ============================================================================

class DownloadError(KlipMachineError):
    """Error during video download."""
    pass


class VideoNotFoundError(DownloadError):
    """Video is unavailable, private, or not found."""
    pass


class NetworkError(DownloadError):
    """Network error during download."""
    pass


# ============================================================================
# TRANSCRIPTION ERRORS
# ============================================================================

class TranscriptionError(KlipMachineError):
    """Error during audio transcription."""
    pass


class AudioNotFoundError(TranscriptionError):
    """Audio file not found or invalid."""
    pass


# ============================================================================
# ANALYSIS ERRORS (AI)
# ============================================================================

class AnalysisError(KlipMachineError):
    """Error during AI transcript analysis."""
    pass


class APIError(AnalysisError):
    """API call error (Groq, OpenAI, Ollama)."""
    pass


class RateLimitError(APIError):
    """API rate limit exceeded."""
    pass


class JSONParseError(AnalysisError):
    """Error parsing LLM JSON response."""
    pass


# ============================================================================
# EXPORT ERRORS (FFmpeg)
# ============================================================================

class ExportError(KlipMachineError):
    """Error during video export."""
    pass


class FFmpegError(ExportError):
    """FFmpeg not available or encoding error."""
    pass


class InvalidClipError(ExportError):
    """Invalid clip (incorrect timestamps, negative duration, etc.)."""
    pass


# ============================================================================
# CONFIGURATION ERRORS
# ============================================================================

class ConfigError(KlipMachineError):
    """Configuration error (missing API keys, etc.)."""
    pass
