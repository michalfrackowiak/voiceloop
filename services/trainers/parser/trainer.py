"""Parser SFT trainer plugins (P2-07.1) and optional GLiNER2 slot trainer (P2-07.2).

SYNTHETIC / FIXTURE TRAINING STAND-IN: the shipped backend produces deterministic
pseudo-deltas, not trained neural weights. Both plugins implement the P2-06
``TrainerPlugin`` interface and return contract 1.0.0 results through
``build_trainer_result``. Real training backends are injected via
:class:`~.adapter.SlotTrainingBackend`; no heavy dependency is imported here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from schemas.booking import BookingValidationError, validate_booking
from services.trainers.core import (
    TrainerPlugin,
    TrainerResult,
    TrainerSettings,
    build_trainer_result,
)
from voiceloop_common import ConfigError, ValidationError

from .adapter import (
    AdapterDelta,
    AdapterKind,
    AdapterStore,
    FakeAdapterStore,
    FakeLoraAdapter,
    FakeLoraTrainingBackend,
    ParserExample,
    SlotTrainingBackend,
)
from .settings import ParserSettings


class ExampleLoader(Protocol):
    """Injectable bridge from a P2-05 dataset manifest to parser SFT examples."""

    def load_examples(self, manifest: Mapping[str, Any]) -> tuple[ParserExample, ...]:
        """Return validated gold and noisy examples for one manifest."""
        ...


class InMemoryExampleLoader:
    """Loader for injected fake examples; ignores manifest contents by design."""

    def __init__(self, examples: Sequence[ParserExample] = ()) -> None:
        """Hold an immutable snapshot of examples for later training runs."""
        self._examples = tuple(examples)

    def load_examples(self, manifest: Mapping[str, Any]) -> tuple[ParserExample, ...]:
        """Return the injected examples (the manifest is accepted but unused)."""
        return self._examples


class SftJsonlExampleLoader:
    """Loader for P2-05's TRL-style SFT JSONL export.

    Reads ``manifest['sft_path']`` unless an explicit path is supplied. Golden
    call IDs must be passed by the caller (from P1-06 via P2-05); they are never
    read directly from the sealed set here.
    """

    def __init__(
        self, path: str | Path | None = None, *, exclude_call_ids: Iterable[str] = ()
    ) -> None:
        """Configure a loader path override and excluded call IDs.

        Args:
            path: Optional JSONL path; defaults to ``manifest['sft_path']``.
            exclude_call_ids: Sealed golden / regression call IDs never trained on.

        """
        self._path = Path(path) if path is not None else None
        self._excluded = frozenset(exclude_call_ids)

    def load_examples(self, manifest: Mapping[str, Any]) -> tuple[ParserExample, ...]:
        """Parse and validate SFT chat rows into parser examples.

        Raises:
            ValidationError: If the path is missing/unreadable or a row is
                malformed, duplicated, or belongs to an excluded call.

        """
        raw_path = self._path if self._path is not None else manifest.get("sft_path")
        if not isinstance(raw_path, (str, Path)) or not str(raw_path):
            raise ValidationError("SFT example loader needs a path or manifest['sft_path']")
        path = Path(raw_path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValidationError(f"cannot read SFT JSONL at {path}: {exc}") from exc

        examples: list[ParserExample] = []
        seen: set[str] = set()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid SFT JSON on line {number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValidationError(f"SFT line {number} must be a JSON object")
            call_id = str(row.get("call_id", ""))
            if call_id in self._excluded:
                continue
            kind_raw = row.get("kind")
            if kind_raw == "parser_clean":
                kind = "gold"
            elif kind_raw == "parser_noisy":
                kind = "noisy"
            else:
                raise ValidationError(
                    f"unsupported SFT example kind on line {number}: {kind_raw!r}"
                )
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValidationError(f"SFT line {number} has no messages array")
            user = next(
                (
                    message.get("content")
                    for message in messages
                    if isinstance(message, dict) and message.get("role") == "user"
                ),
                "",
            )
            assistant = next(
                (
                    message.get("content")
                    for message in messages
                    if isinstance(message, dict) and message.get("role") == "assistant"
                ),
                "",
            )
            if not isinstance(user, str) or not isinstance(assistant, str):
                raise ValidationError(f"SFT line {number} has non-text messages")
            _prefix, marker, utterance = user.rpartition("Utterance: ")
            if not marker:
                raise ValidationError(f"SFT line {number} has no Utterance section")
            try:
                slots = json.loads(assistant)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    f"SFT line {number} has invalid assistant JSON: {exc}"
                ) from exc
            example = ParserExample.from_mapping(
                {
                    "example_id": row.get("example_id"),
                    "kind": kind,
                    "source_text": utterance.strip(),
                    "slots": slots,
                    "call_id": call_id,
                    "synthetic": row.get("synthetic", True),
                }
            )
            if example.example_id in seen:
                raise ValidationError(f"duplicate SFT example_id: {example.example_id!r}")
            seen.add(example.example_id)
            examples.append(example)
        return tuple(examples)


def validate_training_examples(
    examples: Sequence[ParserExample], *, excluded_call_ids: Iterable[str] = ()
) -> None:
    """Fail closed on data that would violate dataset or schema invariants.

    Args:
        examples: Gold and noisy parser examples to check.
        excluded_call_ids: Golden/regression calls that must never be trained on.

    Raises:
        ValidationError: If examples are empty, duplicated, missing either
            kind, contain an excluded call, or have an invalid slot target.

    """
    if not examples:
        raise ValidationError("parser training requires at least one example")
    identifiers = [example.example_id for example in examples]
    if len(set(identifiers)) != len(identifiers):
        raise ValidationError("parser training examples have duplicate example_id values")
    kinds = {example.kind for example in examples}
    if "gold" not in kinds or "noisy" not in kinds:
        raise ValidationError("parser training requires both gold and noisy examples")
    excluded = set(excluded_call_ids)
    for example in examples:
        if example.call_id in excluded:
            raise ValidationError(f"parser example {example.example_id!r} uses an excluded call")
        try:
            validate_booking(example.slots)
        except BookingValidationError as exc:
            raise ValidationError(
                f"invalid parser training target for {example.example_id!r}: {exc}"
            ) from exc


class _SlotTrainer(TrainerPlugin):
    """Shared contract plumbing for the LoRA and GLiNER2 stand-in plugins."""

    def __init__(
        self,
        *,
        plugin_name: str,
        adapter_kind: AdapterKind,
        settings: ParserSettings | None,
        example_loader: ExampleLoader | None,
        examples: Sequence[ParserExample] | None,
        backend: SlotTrainingBackend | None,
        store: AdapterStore | None,
        missing_backend_message: str,
    ) -> None:
        """Wire settings, example source, backend and artifact store.

        Args:
            plugin_name: Stable registered name for the P2-06 registry.
            adapter_kind: Kind recorded on the exported stand-in artifact.
            settings: Parser settings; defaults to ``ParserSettings()``.
            example_loader: Injected dataset seam; defaults to the examples arg.
            examples: In-memory fake examples when no loader is supplied.
            backend: Injected training backend; may be absent for GLiNER2.
            store: Artifact store; in-memory fake by default.
            missing_backend_message: Fail-closed error when no backend exists.

        """
        self._settings = settings or ParserSettings()
        self._loader = example_loader or InMemoryExampleLoader(examples or ())
        self._backend = backend
        self._store: AdapterStore = store or FakeAdapterStore()
        self._plugin_name = plugin_name
        self._adapter_kind = adapter_kind
        self._missing_backend_message = missing_backend_message

    @property
    def name(self) -> str:
        """Return the stable registered plugin name."""
        return self._plugin_name

    def train(self, manifest: Mapping[str, Any], settings: TrainerSettings) -> TrainerResult:
        """Run deterministic stand-in training and export a hash-addressed adapter.

        Args:
            manifest: Dataset manifest with a non-empty ``content_hash``.
            settings: Core job settings supplying model, seed and artifact prefix.

        Returns:
            A validated 1.0.0 contract result whose ``artifact_uri`` addresses
            the exported adapter, not a raw training checkpoint.

        Raises:
            ConfigError: If no training backend was injected.
            ValidationError: If examples or the manifest are invalid.

        """
        if self._backend is None:
            raise ConfigError(self._missing_backend_message)
        content_hash = manifest.get("content_hash")
        if not isinstance(content_hash, str) or not content_hash:
            raise ValidationError("dataset_manifest.content_hash must be a non-empty string")
        examples = self._loader.load_examples(manifest)
        validate_training_examples(examples, excluded_call_ids=self._settings.exclude_call_ids)
        params = self._settings.as_training_params()
        deltas: tuple[AdapterDelta, ...] = self._backend.train(
            examples, model=settings.model, seed=settings.seed, params=params
        )
        adapter = FakeLoraAdapter(
            dataset_hash=content_hash,
            deltas=deltas,
            kind=self._adapter_kind,
            base_model=settings.model,
            seed=settings.seed,
            params=params,
        )
        artifact_uri = self._store.save(adapter, prefix=settings.artifact_uri_prefix)
        gold = sum(1 for example in examples if example.kind == "gold")
        noisy = len(examples) - gold
        metrics = {
            "fake_train_loss": 0.0,
            "examples_total": float(len(examples)),
            "examples_gold": float(gold),
            "examples_noisy": float(noisy),
            "synthetic_examples": float(sum(1 for example in examples if example.synthetic)),
        }
        return build_trainer_result(manifest, settings, artifact_uri=artifact_uri, metrics=metrics)


class ParserLoraTrainer(_SlotTrainer):
    """LoRA SFT plugin for gold and noisy parser examples (P2-07.1)."""

    def __init__(
        self,
        *,
        settings: ParserSettings | None = None,
        example_loader: ExampleLoader | None = None,
        examples: Sequence[ParserExample] | None = None,
        backend: SlotTrainingBackend | None = None,
        store: AdapterStore | None = None,
    ) -> None:
        """Select the fake backend unless a real one is explicitly injected.

        Raises:
            ConfigError: If settings request a real backend without injection.

        """
        config = settings or ParserSettings()
        if config.backend != "fake" and backend is None:
            raise ConfigError(
                f"parser backend {config.backend!r} requires an injected SlotTrainingBackend"
            )
        super().__init__(
            plugin_name="parser-lora",
            adapter_kind="fake-lora",
            settings=config,
            example_loader=example_loader,
            examples=examples,
            backend=backend or FakeLoraTrainingBackend(),
            store=store,
            missing_backend_message="parser LoRA backend is not configured",
        )


class Gliner2SlotTrainer(_SlotTrainer):
    """Optional GLiNER2 slot-filler plugin behind a fake protocol (P2-07.2)."""

    def __init__(
        self,
        *,
        settings: ParserSettings | None = None,
        example_loader: ExampleLoader | None = None,
        examples: Sequence[ParserExample] | None = None,
        backend: SlotTrainingBackend | None = None,
        store: AdapterStore | None = None,
    ) -> None:
        """Create an optional trainer that fails closed without an injected backend."""
        super().__init__(
            plugin_name="parser-gliner2",
            adapter_kind="fake-gliner2",
            settings=settings,
            example_loader=example_loader,
            examples=examples,
            backend=backend,
            store=store,
            missing_backend_message=(
                "GLiNER2 is optional and has no default dependency; inject a "
                "SlotTrainingBackend implementation to use parser-gliner2"
            ),
        )
