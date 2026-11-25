"""Regression tests for P3-03 scalar PII screening (harden recorded seam)."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any

import pytest

from services.datasets.augment.cli import build_augmentation
from services.datasets.augment.manifest import (
    build_manifest,
    validate_manifest,
)
from services.datasets.augment.variants import VariantSpec
from services.datasets.manifest import canonical_json
from voiceloop_common import ValidationError

CARD = "4111111111111111"
TIMESTAMP = "2026-11-25T00:00:00Z"
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class ParamsTts:
    """Injected fake backend that returns fixed transform params."""

    def __init__(self, params: dict[str, int | str]) -> None:
        """Store the params every synthesized record will carry."""
        self._params = params

    def synthesize(
        self,
        *,
        token: str,
        sample_rate_hz: int,
        duration_ms: int,
        seed: int,
    ) -> tuple[int, ...]:
        """Return a constant in-memory stand-in signal."""
        del token, seed
        return (1,) * (sample_rate_hz * duration_ms // 1000)

    def params(self, *, spec: VariantSpec) -> dict[str, int | str]:
        """Return the injected params; ``spec`` keeps the backend protocol shape."""
        del spec
        return dict(self._params)


def _build_with_params(
    settings: Any,
    source_manifest: dict[str, Any],
    params: dict[str, int | str],
    *,
    created_at: str = TIMESTAMP,
) -> dict[str, Any]:
    """Build one augmentation batch with injected backend params."""
    result = build_augmentation(
        settings,
        tts=ParamsTts(params),
        dataset_manifest=source_manifest,
        golden_call_ids=("call-0001",),
        regression_call_ids=(),
        created_at=created_at,
    )
    return result.manifest


def _rehashed(
    manifest: dict[str, Any],
    mutate: Any,
) -> dict[str, Any]:
    """Copy, mutate and rehash a manifest so shape checks pass before screening."""
    candidate = copy.deepcopy(manifest)
    mutate(candidate)
    payload = {key: value for key, value in candidate.items() if key != "content_hash"}
    candidate["content_hash"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return candidate


@pytest.mark.parametrize("key", ("token_digest", "digest"))
def test_digest_keyed_pii_fails_closed(
    settings: Any,
    source_manifest: dict[str, Any],
    key: str,
) -> None:
    """A digest key must not launder a card number past PII screening."""
    with pytest.raises(ValidationError) as exc:
        _build_with_params(settings, source_manifest, {key: CARD})
    assert "names, emails, phones or cards" in str(exc.value)
    assert CARD not in str(exc.value)


def test_source_manifest_content_hash_pii_fails_closed(
    settings: Any,
    source_manifest: dict[str, Any],
) -> None:
    """PII copied into ``source_manifest_content_hash`` fails closed."""
    poisoned = copy.deepcopy(source_manifest)
    poisoned["content_hash"] = CARD
    with pytest.raises(ValidationError) as exc:
        build_manifest(
            (),
            source_manifest=poisoned,
            golden_call_ids=(),
            regression_call_ids=(),
            created_at=TIMESTAMP,
        )
    assert "names, emails, phones or cards" in str(exc.value)
    assert CARD not in str(exc.value)


@pytest.mark.parametrize(
    "value",
    (
        CARD,
        float(f"{CARD}.0"),
        5551234567,
        15551234567.0,
    ),
)
def test_numeric_card_and_phone_scalars_fail_closed(
    run: Any,
    value: int | float,
) -> None:
    """Int/float leaves are converted to text before screening, including bare phones."""
    candidate = _rehashed(
        run.manifest,
        lambda manifest: manifest["variants"][0]["transforms"][0]["params"].update(
            {"numeric": value}
        ),
    )
    with pytest.raises(ValidationError) as exc:
        validate_manifest(candidate)
    assert "names, emails, phones or cards" in str(exc.value)
    assert str(value) not in str(exc.value)


def test_digest_keyed_number_fails_closed(
    settings: Any,
    source_manifest: dict[str, Any],
) -> None:
    """A non-hex scalar under a digest key is screened, then rejected."""
    with pytest.raises(ValidationError) as exc:
        _build_with_params(settings, source_manifest, {"token_digest": int(CARD)})
    assert "names, emails, phones or cards" in str(exc.value)
    assert CARD not in str(exc.value)


def test_valid_digests_and_timestamp_pass(
    settings: Any,
    source_manifest: dict[str, Any],
) -> None:
    """Strict SHA-256 hex digests and a strict UTC timestamp remain valid."""
    digest = hashlib.sha256(b"hardening").hexdigest()
    manifest = _build_with_params(
        settings,
        source_manifest,
        {"token_digest": digest, "digest": digest},
    )
    validate_manifest(manifest)
    assert manifest["created_at"] == TIMESTAMP
    assert manifest["source_manifest_content_hash"] == source_manifest["content_hash"]
    assert (
        manifest["content_hash"]
        == hashlib.sha256(
            canonical_json({k: v for k, v in manifest.items() if k != "content_hash"}).encode()
        ).hexdigest()
    )
    assert DIGEST_RE.fullmatch(manifest["content_hash"])
    assert DIGEST_RE.fullmatch(manifest["source_manifest_content_hash"])
    assert all(
        DIGEST_RE.fullmatch(record["token_digest"])
        and DIGEST_RE.fullmatch(record["audio"]["digest"])
        for record in manifest["variants"]
    )


def test_non_timestamp_created_at_is_screened(
    settings: Any,
    source_manifest: dict[str, Any],
) -> None:
    """``created_at`` is exempt only as a strict timestamp; otherwise it is text."""
    with pytest.raises(ValidationError) as exc:
        _build_with_params(settings, source_manifest, {}, created_at=CARD)
    assert "names, emails, phones or cards" in str(exc.value)
    assert CARD not in str(exc.value)

    ordinary = _build_with_params(settings, source_manifest, {}, created_at="fixture-clock")
    validate_manifest(ordinary)
    assert ordinary["created_at"] == "fixture-clock"


def test_bool_and_none_leaves_are_not_screened_as_numbers() -> None:
    """Boolean and null leaves stay out of numeric screening; keys are screened."""
    from services.datasets.augment.manifest import _iter_strings

    assert list(_iter_strings({"a": True, "b": None, "c": "ok"})) == ["a", "b", "c", "ok"]


def test_audio_bytes_guard_is_still_enforced(run: Any) -> None:
    """The pre-existing audio-byte guard remains active after hardening."""
    candidate = _rehashed(
        run.manifest,
        lambda manifest: manifest["variants"][0]["audio"].update({"samples": [1, 2, 3]}),
    )
    # Shape validation rejects the extra audio field before the byte guard.
    with pytest.raises(ValidationError):
        validate_manifest(candidate)

    from services.datasets.augment.manifest import _assert_no_audio_bytes

    with pytest.raises(ValidationError, match="audio bytes"):
        _assert_no_audio_bytes({"unknown": {"samples": [1, 2, 3]}})
    with pytest.raises(ValidationError, match="raw integer sample arrays"):
        _assert_no_audio_bytes({"unknown": [1] * 65})
