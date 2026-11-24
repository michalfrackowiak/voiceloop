"""Shared exception hierarchy for VoiceLoop components."""


class VoiceLoopError(Exception):
    """Base class for all expected VoiceLoop failures."""


class ConfigError(VoiceLoopError):
    """Configuration is missing, malformed, or inconsistent."""


class ValidationError(VoiceLoopError):
    """Input or contract data failed validation."""


class StorageError(VoiceLoopError):
    """A storage backend could not complete an operation."""


class TransientError(VoiceLoopError):
    """A temporary failure that callers may safely retry."""
