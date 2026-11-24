"""Tests for the shared exception hierarchy."""

from voiceloop_common import (
    ConfigError,
    StorageError,
    TransientError,
    ValidationError,
    VoiceLoopError,
)
from voiceloop_common import errors as errors_module


def test_all_component_errors_share_the_base() -> None:
    for error_type in (
        ConfigError,
        ValidationError,
        StorageError,
        TransientError,
    ):
        assert issubclass(error_type, VoiceLoopError)
        assert issubclass(error_type, Exception)


def test_errors_module_exports_the_same_classes() -> None:
    assert errors_module.VoiceLoopError is VoiceLoopError
    assert errors_module.ConfigError is ConfigError
    assert errors_module.ValidationError is ValidationError
    assert errors_module.StorageError is StorageError
    assert errors_module.TransientError is TransientError


def test_error_keeps_its_message() -> None:
    assert str(ValidationError("bad payload")) == "bad payload"
