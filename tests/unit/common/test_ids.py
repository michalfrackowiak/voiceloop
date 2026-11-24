"""Tests for identifier helpers."""

from uuid import UUID

from voiceloop_common import new_id


def test_new_id_is_a_unique_uuid4_string() -> None:
    values = {new_id() for _ in range(50)}

    assert len(values) == 50
    for value in values:
        assert UUID(value).version == 4
