"""Size-bounded, JSON-safe payload preparation for the ingest call path."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping
from typing import Any, cast


class PayloadError(ValueError):
    """Raised when an event is not JSON-safe or exceeds the size budget."""


class PayloadTimeout(TimeoutError):
    """Raised when preparation exhausts the caller's enqueue budget."""


class _Frame:
    """One open container while walking an event without recursion."""

    def __init__(
        self,
        source: Mapping[str, Any] | list[Any],
        target: dict[str, Any] | list[Any],
        is_mapping: bool,
        ident: int,
    ) -> None:
        """Create an iterator frame for a mapping or sequence container."""
        self.items: Iterator[Any] = (
            iter(cast(Mapping[str, Any], source).items())
            if is_mapping
            else iter(cast(list[Any], source))
        )
        self.target = target
        self.is_mapping = is_mapping
        self.ident = ident


def _text_size(value: str, max_bytes: int) -> int:
    """Return quoted UTF-8 size, rejecting strings that cannot fit."""
    if len(value) > max_bytes:
        raise PayloadError("string exceeds payload budget")
    return len(value.encode("utf-8")) + 2


def _charge(used: list[int], amount: int, max_bytes: int) -> None:
    """Add ``amount`` to the running serialised-size estimate."""
    used[0] += amount
    if used[0] > max_bytes:
        raise PayloadError("payload exceeds max_payload_bytes")


def _assign(
    target: dict[str, Any] | list[Any],
    key: str,
    value: Any,
) -> None:
    """Assign a prepared value into the current target container."""
    if isinstance(target, list):
        target.append(value)
    else:
        target[key] = value


def sanitize_payload(
    event: Mapping[str, Any],
    max_bytes: int,
    deadline: float,
) -> dict[str, Any]:
    """Return a private JSON-safe copy within ``max_bytes`` and ``deadline``.

    The walk is iterative and checks both size and time, so hostile payloads
    (cycles, huge strings, arbitrary objects) cannot block the live call path.
    """
    if time.monotonic() > deadline:
        raise PayloadTimeout("enqueue budget exhausted before preparation")
    root: dict[str, Any] = {}
    if max_bytes < 2:
        raise PayloadError("payload budget too small for an event")
    used = [2]
    frames = [_Frame(event, root, True, id(event))]
    active = {id(event)}
    nodes = 0

    while frames:
        nodes += 1
        if nodes % 32 == 0 and time.monotonic() > deadline:
            raise PayloadTimeout("enqueue budget exhausted during preparation")

        frame = frames[-1]
        try:
            raw = next(frame.items)
        except StopIteration:
            frames.pop()
            active.remove(frame.ident)
            continue

        if frame.is_mapping:
            key, value = raw
            if not isinstance(key, str):
                raise PayloadError("event keys must be strings")
            _charge(used, _text_size(key, max_bytes) + 1, max_bytes)
            target_key = key
        else:
            value = raw
            _charge(used, 1, max_bytes)
            target_key = ""

        if value is None:
            _charge(used, 4, max_bytes)
            _assign(frame.target, target_key, None)
        elif isinstance(value, bool):
            _charge(used, 4 if value else 5, max_bytes)
            _assign(frame.target, target_key, value)
        elif isinstance(value, int):
            if value.bit_length() > max_bytes * 4:
                raise PayloadError("integer exceeds payload budget")
            _charge(used, int(value.bit_length() * 0.30103) + 2, max_bytes)
            _assign(frame.target, target_key, value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise PayloadError("non-finite floats are not JSON-safe")
            _charge(used, 24, max_bytes)
            _assign(frame.target, target_key, value)
        elif isinstance(value, str):
            _charge(used, _text_size(value, max_bytes), max_bytes)
            _assign(frame.target, target_key, value)
        elif isinstance(value, Mapping):
            ident = id(value)
            if ident in active:
                raise PayloadError("cyclic payloads are not supported")
            child: dict[str, Any] = {}
            _charge(used, 2, max_bytes)
            _assign(frame.target, target_key, child)
            frames.append(_Frame(value, child, True, ident))
            active.add(ident)
        elif isinstance(value, (list, tuple)):
            ident = id(value)
            if ident in active:
                raise PayloadError("cyclic payloads are not supported")
            child_list: list[Any] = []
            _charge(used, 2, max_bytes)
            _assign(frame.target, target_key, child_list)
            frames.append(_Frame(list(value), child_list, False, id(value)))
            active.add(ident)
        else:
            raise PayloadError(f"unsupported payload type: {type(value).__name__}")

    return root
