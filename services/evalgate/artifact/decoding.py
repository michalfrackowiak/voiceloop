"""Exact decoding settings and deterministic combination matrix.

Converted CTranslate2 models can emit empty or truncated text when hotwords, an
initial prompt or ``condition_on_previous_text`` are combined (``context.md``
section 10). Every combination here is synthetic and deterministic; the SHA-256
of the canonical settings is recorded in each generated STT report.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from voiceloop_common import ConfigError

from .artifact import canonical_bytes
from .settings import ArtifactSettings


@dataclass(frozen=True)
class DecodingSettings:
    """Exact decoding settings used for one probe run."""

    hotwords: tuple[str, ...]
    initial_prompt: str
    condition_on_previous_text: bool
    language: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate settings and compute their canonical hash."""
        if any(not word.strip() for word in self.hotwords):
            raise ConfigError("hotwords must be non-blank strings")
        if not self.language.strip():
            raise ConfigError("decoding language must be non-blank")
        object.__setattr__(
            self, "sha256", hashlib.sha256(canonical_bytes(self.as_mapping())).hexdigest()
        )

    def as_mapping(self) -> dict[str, object]:
        """Return canonical JSON data covered by :attr:`sha256`."""
        return {
            "hotwords": list(self.hotwords),
            "initial_prompt": self.initial_prompt,
            "condition_on_previous_text": self.condition_on_previous_text,
            "language": self.language,
            "settings_schema_version": "internal-1.0.0",
        }

    def to_report_fields(self) -> dict[str, object]:
        """Return schema-compatible ``decoding_settings`` fields for a report."""
        return {
            "hotwords": list(self.hotwords),
            "initial_prompt": self.initial_prompt,
            "condition_on_previous_text": self.condition_on_previous_text,
            "language": self.language,
            "decoding_settings_sha256": self.sha256,
        }


@dataclass(frozen=True)
class DecodeCombination:
    """One named, deterministic combination of decoding switches."""

    name: str
    settings: DecodingSettings


def decoding_hash_from_fields(fields: Mapping[str, object]) -> str:
    """Recompute a decoding hash from report fields, failing closed on bad input.

    Args:
        fields: Report ``decoding_settings`` mapping produced by
            :meth:`DecodingSettings.to_report_fields`.

    Returns:
        The SHA-256 digest over the canonical four settings plus version.

    Raises:
        ConfigError: If a required field is missing or wrongly typed.

    """
    try:
        hotwords = fields["hotwords"]
        initial_prompt = fields["initial_prompt"]
        condition = fields["condition_on_previous_text"]
        language = fields["language"]
    except KeyError as exc:
        raise ConfigError(f"decoding settings missing: {exc.args[0]}") from exc
    if not isinstance(hotwords, list) or not all(isinstance(word, str) for word in hotwords):
        raise ConfigError("hotwords must be a list of strings")
    if not isinstance(initial_prompt, str) or not isinstance(language, str):
        raise ConfigError("initial_prompt and language must be strings")
    if not isinstance(condition, bool):
        raise ConfigError("condition_on_previous_text must be a boolean")
    payload = {
        "hotwords": hotwords,
        "initial_prompt": initial_prompt,
        "condition_on_previous_text": condition,
        "language": language,
        "settings_schema_version": "internal-1.0.0",
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def decode_combinations(settings: ArtifactSettings) -> tuple[DecodeCombination, ...]:
    """Enumerate the eight hotword/prompt/previous-text combinations.

    Args:
        settings: Typed settings supplying configured hotwords, prompt and language.

    Returns:
        Eight immutable combinations in a fixed order, so reports are reproducible.

    """
    hotword_options: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("off", ()),
        ("on", tuple(settings.hotwords)),
    )
    prompt_options: tuple[tuple[str, str], ...] = (("off", ""), ("on", settings.initial_prompt))
    context_options: tuple[tuple[str, bool], ...] = (("off", False), ("on", True))
    combinations: list[DecodeCombination] = []
    for hotword_name, hotwords in hotword_options:
        for prompt_name, prompt in prompt_options:
            for context_name, condition in context_options:
                name = f"hotwords_{hotword_name}_prompt_{prompt_name}_context_{context_name}"
                combinations.append(
                    DecodeCombination(
                        name=name,
                        settings=DecodingSettings(
                            hotwords=hotwords,
                            initial_prompt=prompt,
                            condition_on_previous_text=condition,
                            language=settings.language,
                        ),
                    )
                )
    return tuple(combinations)


def production_combination(settings: ArtifactSettings) -> DecodeCombination:
    """Return the combination matching the exact configured production settings."""
    return DecodeCombination(
        name="production_settings",
        settings=DecodingSettings(
            hotwords=tuple(settings.hotwords),
            initial_prompt=settings.initial_prompt,
            condition_on_previous_text=settings.condition_on_previous_text,
            language=settings.language,
        ),
    )


def language_combinations(
    combinations: Sequence[DecodeCombination], languages: Sequence[str]
) -> tuple[DecodeCombination, ...]:
    """Clone decoding combinations for each configured non-English language.

    Args:
        combinations: Base decoding combinations.
        languages: Served language codes other than the base report language.

    Returns:
        Additional deterministic combinations with language-specific names.

    """
    return tuple(
        DecodeCombination(
            name=f"{combination.name}_lang_{language}",
            settings=replace(combination.settings, language=language),
        )
        for language in languages
        for combination in combinations
    )


def describe_combination(combination: DecodeCombination) -> dict[str, object]:
    """Return a hash-and-switch summary without prompt text or hotword values."""
    return {
        "name": combination.name,
        "decoding_settings_sha256": combination.settings.sha256,
        "hotwords_enabled": bool(combination.settings.hotwords),
        "initial_prompt_enabled": bool(combination.settings.initial_prompt),
        "condition_on_previous_text": combination.settings.condition_on_previous_text,
    }


__all__ = [
    "DecodeCombination",
    "DecodingSettings",
    "decode_combinations",
    "decoding_hash_from_fields",
    "describe_combination",
    "language_combinations",
    "production_combination",
]
