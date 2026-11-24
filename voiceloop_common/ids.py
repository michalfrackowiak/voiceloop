"""Identifier helpers shared by services and the SDK."""

from __future__ import annotations

from uuid import uuid4


def new_id() -> str:
    """Return a freshly generated UUID version 4 as a lowercase string."""
    return str(uuid4())
