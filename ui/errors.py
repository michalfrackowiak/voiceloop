"""Fail-closed error type shared by the synthetic review UI harness."""


class UiValidationError(ValueError):
    """Raised when a synthetic UI input violates a deterministic safety rule."""
