"""Strict JSON-schema constrained-decoding check for parser outputs (P2-07.3).

SYNTHETIC / FIXTURE TRAINING STAND-IN: this module validates candidate text
against the project's booking schema, it does not implement XGrammar/Outlines
token masking. A real constrained-generation backend must be injected as a
:class:`ParserRuntime`; invalid output is always rejected, never repaired.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from schemas.booking import BookingValidationError, validate_booking
from voiceloop_common import ValidationError

from .adapter import ParserRuntime


class ConstrainedDecoder(Protocol):
    """Injectable strict decoder for raw model output."""

    def decode(self, raw_output: str) -> dict[str, Any]:
        """Parse and validate raw output, rejecting anything schema-invalid."""
        ...


class BookingSchemaDecoder:
    """Parse JSON strictly and validate booking slots with cross-field checks.

    Uses ``schemas.booking.validate_booking`` (the project's validated helper),
    which applies the active booking schema, honours ``VOICELOOP_BOOKING_SCHEMA``
    and enforces cross-field rules such as check-out after check-in. No field is
    added, coerced or removed on failure.
    """

    def decode(self, raw_output: str) -> dict[str, Any]:
        """Decode and validate raw candidate text.

        Args:
            raw_output: JSON text produced by the parser/runtime.

        Returns:
            The decoded booking slot object.

        Raises:
            ValidationError: If the text is not a JSON object or violates the
                active booking schema or its cross-field validators.

        """
        if not isinstance(raw_output, str):
            raise ValidationError("constrained output must be JSON text")
        try:
            decoded = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"constrained output is not valid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValidationError("constrained output must be a JSON object")
        try:
            validate_booking(decoded)
        except BookingValidationError as exc:
            raise ValidationError(f"constrained output rejected by booking schema: {exc}") from exc
        return decoded


def decode_booking_output(raw_output: str) -> dict[str, Any]:
    """Decode one raw parser output through the strict booking-schema gate.

    Args:
        raw_output: JSON text to check.

    Returns:
        The validated booking slot object.

    """
    return BookingSchemaDecoder().decode(raw_output)


class ConstrainedParserRuntime:
    """Wrap a serving runtime so generated text must pass the schema gate."""

    def __init__(
        self, runtime: ParserRuntime, *, decoder: ConstrainedDecoder | None = None
    ) -> None:
        """Compose a runtime with a strict decoder.

        Args:
            runtime: Fake or injected real vLLM-like runtime.
            decoder: Strict decoder; :class:`BookingSchemaDecoder` by default.

        """
        self._runtime = runtime
        self._decoder: ConstrainedDecoder = decoder or BookingSchemaDecoder()

    def load_adapter(self, uri: str) -> str:
        """Load an adapter through the wrapped runtime and return its handle."""
        return self._runtime.load_adapter(uri)

    def unload_adapter(self, handle: str) -> None:
        """Unload a handle through the wrapped runtime (idempotent)."""
        self._runtime.unload_adapter(handle)

    def generate(self, handle: str, prompt: str) -> dict[str, Any]:
        """Generate raw output and reject it unless it satisfies the schema.

        Raises:
            ValidationError: If the runtime has no such handle or the generated
                text fails the constrained-decoding check.

        """
        return self._decoder.decode(self._runtime.generate(handle, prompt))
