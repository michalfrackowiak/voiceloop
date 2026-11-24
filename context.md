# context.md — Shared Agent Reference

**Read this file fully before starting any task.** It is the stable source of truth for every agent working on this project. If something here conflicts with a task in `roadmap.md`, stop and raise it rather than guessing. If you change a decision, update this file in the same change.

Project: **VoiceLoop** (working title). Human-in-the-loop continuous improvement for voice agents. See `README.md` for the human-facing overview.

---

## 1. Mission

Close the loop between live voice-agent failures and better models:

**capture → triage → human review → dataset → train → evaluate → gated release → monitor**

Reference use case: a hotel-booking voice agent. Guest speaks → STT produces a hypothesis → a text parser fills a booking form (dates, guests, room type, name, etc.) → booking record is created.

## 2. Core principles (non-negotiable invariants)

Any change that violates one of these is a bug.

1. **Never in the call path.** The service must not add latency to, or be able to break, a live call. All agent-facing integrations are async, fire-and-forget, and fail open.
2. **A human listens to audio.** LLM judges may assist triage but never replace human review for golden-set or training labels.
3. **Evaluate the deployed artifact.** Gates run on the converted/quantised artifact that will actually serve (e.g. CTranslate2, ONNX, merged adapter), not the training checkpoint.
4. **Promote on end-to-end booking-field accuracy from audio.** WER alone is never sufficient to promote.
5. **Golden set is sealed.** It is never trained on, never used for tuning, and changes only through an audited process with a second reviewer.
6. **Keep an unbiased random slice.** 3–5% of turns bypass the failure filter so real error rates are measurable.
7. **Split by call, never by utterance.** Prevents speaker leakage between train and test.
8. **Consent and PII first.** No audio or transcript enters a dataset without a consent flag and PII screening. Card numbers are always dropped.
9. **Lineage everywhere.** Every training example traces to its source call; every model traces to its dataset hash, git SHA, and image digest.
10. **Rollback is one step.** Promotion and rollback are both an MLflow alias flip.
11. **Cheapest fix first.** Route failures through the adaptation ladder (§7) before reaching for retraining.

## 3. Glossary

| Term | Meaning |
|---|---|
| Turn | One guest utterance plus the agent's processing of it |
| Hypothesis | STT output text (with word-level confidences) |
| Parser | Component mapping transcript text + dialogue context to structured slots (small LLM with LoRA, or GLiNER2) |
| Slots | Structured fields defined by the booking JSON Schema |
| Booking record | Final record in the hotel/booking system; used as free supervision |
| Golden set | Hand-labelled, sealed evaluation set from a random sample of real traffic |
| Regression set | Every past reviewed failure, kept to prevent re-breakage |
| General-capability set | Silence, line noise, other served languages; detects forgetting and hallucination |
| Champion / Challenger | MLflow aliases: `@champion` serves traffic; `@challenger` is under evaluation |
| Shadow mode | Candidate model scores live traffic silently; output is not used |
| Canary | Candidate serves a small share of live traffic |
| Adaptation ladder | Ordered fixes from cheapest/safest to most expensive (§7) |
| Replay data | Earlier verified examples mixed into training to limit forgetting |
| Artifact | The deployable form of a model after conversion |

## 4. Architecture

Components (each maps to a directory in the planned layout):

| Component | Responsibility | Depends on |
|---|---|---|
| `sdk/` | `log_turn` call; Pipecat observer and per-turn audio hook | Event schema |
| `services/ingest` | HTTP + OpenTelemetry ingest; writes metadata to Postgres, audio to object store | Event schema |
| `services/triage` | Failure scoring, clustering, sampling, daily queue cap | Ingest data |
| `services/review_adapter` | Pushes queue items to Label Studio/Argilla; pulls back corrected results | Triage, booking schema |
| `services/datasets` | Builds versioned, hashed datasets; splits; exporters | Review results, consent/PII |
| `services/trainers` | Plugin interface; Whisper, Parakeet, parser trainers | Datasets |
| `services/evalgate` | Metrics, bootstrap CIs, promotion rules | Eval sets, artifacts |
| `services/release` | Shadow, canary, alias sync agent | MLflow, serving layer |
| `ui/` | Custom review UI (later phase) | Review adapter API |
| `infra/` | Docker, compose, CI, dashboards | All |

Data flow: agent → SDK → ingest → triage → review → datasets → trainers → MLflow → evalgate → release → agent.

## 5. Draft turn event schema

Authoritative copy will live in `schemas/turn_event.schema.json`. Treat the below as the starting point.

```json
{
  "event_id": "uuid",
  "call_id": "string",
  "turn_index": 0,
  "ts": "RFC3339",
  "consent": { "recording": true, "training": true },
  "audio": {
    "uri": "s3://bucket/path.wav",
    "sample_rate_hz": 8000,
    "channels": 1,
    "duration_ms": 0
  },
  "stt": {
    "model_id": "string",
    "model_version": "string",
    "text": "string",
    "words": [{ "w": "string", "start_ms": 0, "end_ms": 0, "conf": 0.0 }],
    "decoder_stats": { "avg_logprob": 0.0, "compression_ratio": 0.0, "no_speech_prob": 0.0 }
  },
  "parser": {
    "model_id": "string",
    "model_version": "string",
    "input_text": "string",
    "output": {},
    "schema_valid": true,
    "logprob": 0.0
  },
  "dialogue_context": {
    "expected_slot": "string|null",
    "reference_date": "YYYY-MM-DD",
    "timezone": "IANA tz",
    "prior_turns_ref": "string|null"
  },
  "outcome": {
    "status": "confirmed|edited_by_staff|abandoned|transferred|unknown",
    "final_record": {},
    "updated_at": "RFC3339"
  }
}
```

`outcome` arrives later and is merged by `(call_id)`. All fields beyond `event_id`, `call_id`, `turn_index`, `audio`, and `stt` are optional at ingest so integrations can adopt incrementally.

## 6. Triage signals

Failure score combines (weights are configurable, start simple):

- low word confidence; decoder thresholds (log-prob, compression ratio, no-speech prob) for Whisper-family
- schema-invalid or low-probability parse
- repair behaviour: guest correction ("no, I said…"), agent re-asking, transfer to human
- disagreement with a larger offline teacher model on the same audio
- mismatch between parser output and final booking record (**highest weight**)

Rules: reserve a 3–5% random slice; cluster near-duplicates and show one representative; cap daily queue size.

## 7. Adaptation ladder

| Tier | Fix | Trigger (rule of thumb, tunable) |
|---|---|---|
| 1 | Vocabulary boost / LM fusion, alias tables, post-ASR normalisation | Any reviewed failure that is a vocabulary or normalisation issue |
| 2 | Parser LoRA retrain on gold + noisy transcripts | ~few hundred verified examples |
| 3 | STT LoRA/fine-tune with replay data | Low thousands of utterances, or a repeated failure class |
| 4 | Base-model swap | A better model appears; reuse the benchmark harness |

Reviewer routing labels: `stt_wrong`, `parser_wrong`, `both_fine`, `unusable_audio`. Triage/ladder logic maps labels and failure type to a tier.

## 8. Dataset rules

One reviewed item yields up to three examples: (a) audio → corrected transcript, (b) corrected transcript → slots, (c) noisy hypothesis → slots. Each keeps dialogue context. Splits are by call. Mix replay data. Drop non-consented rows and any containing card numbers. Emit a manifest with content hash, source call IDs, split assignment, and filter counts. Exporters: NeMo manifest JSONL, Hugging Face datasets, SFT JSONL.

## 9. Evaluation gate rules

- Sets: golden, regression, general-capability.
- Metrics: WER; entity-level error (names, dates, numbers); parser slot F1; schema-valid rate; **end-to-end booking-field accuracy from audio** (primary).
- Statistics: paired bootstrap confidence intervals (sets are small and noisy).
- Pass criteria: required margin over champion on the primary metric, and no regression on critical slices (names, dates, numbers, silence, other languages).
- Release path: eval on converted artifact → shadow → canary → alias flip → monitor. Auto-rollback is a later phase.

## 10. Known model risks (design for these)

- Whisper single-language fine-tuning can degrade other languages.
- Over-scaled LoRA can cause hallucinated text on silence. Use conservative alpha/r (reported fix: 2.0 → 0.2; single data point, verify on our data).
- Converted CTranslate2 models can misbehave with hotwords, an initial prompt, or `condition_on_previous_text` enabled (empty or truncated output). Test the exact decoding settings used in production.
- Model rankings shift quickly; do not hard-code a base model. Keep trainers and benchmarks model-agnostic.
- Small-data retraining (100 utterances ≈ 8–10 min of audio) mostly adds risk; prefer tiers 1–2 until the pool grows.

## 11. Tech decisions

| Area | Decision | Notes |
|---|---|---|
| Language | Python 3.11+ for services/SDK | TypeScript/React for custom UI later |
| API | FastAPI | OpenTelemetry-compatible ingest |
| Storage | Postgres + S3-compatible object store | Audio never stored in Postgres |
| Review (v1) | Label Studio Community or Argilla behind an adapter | Community edition lacks reviewer assignment; the adapter must hide the tool choice |
| Orchestration | cron + CLI first; Prefect or Dagster later | |
| Training | Transformers + PEFT (Whisper), NeMo (Parakeet), Unsloth/TRL/LLaMA-Factory (parser), GLiNER2 (slot-filler) | Behind one trainer interface |
| Registry | MLflow, aliases `@champion` / `@challenger` | |
| Parser serving | vLLM with runtime LoRA load/unload and JSON-schema constrained decoding (XGrammar/Outlines) | Fine-tune for correctness, constrain for validity |
| STT serving | faster-whisper (CTranslate2) or ONNX/NeMo | Confirm current runtime (open question) |
| Observability | OpenTelemetry, Prometheus, Grafana | |
| Training jobs | Pinned Docker images, one job at a time, off the serving GPU | Log dataset hash + git SHA to MLflow |

## 12. Governance requirements

- Per-call consent flags (recording, training).
- PII detection on transcripts; timestamp-based redaction in audio.
- Retention limits on raw audio.
- Full lineage from training example to source call to support deletion requests.
- Roles: reviewer, ML engineer, admin.
- Pseudonymise before training (adapters cannot be surgically unlearned).
- For EU guests: AI Act Article 50 transparency and a GDPR lawful basis for training on recordings. Not legal advice; flag for counsel review.

## 13. Conventions

**Repo and code**
- One component per directory under `services/`; shared types in `schemas/`.
- Type hints and docstrings on public functions; `ruff` + `mypy` clean; `pytest` for tests.
- Config via environment variables and typed settings objects; no secrets in the repo.
- Every service has a `README.md` with run, test, and contract notes.
- Conventional Commits; small PRs scoped to one task ID.

**Contracts**
- Cross-service interfaces are defined in `schemas/` (JSON Schema / OpenAPI) before implementation. Schema changes need a version bump and a note in `docs/CHANGELOG.md`.
- Trainer interface: `train(dataset_manifest, config) -> {artifact_uri, metrics, run_id}`.
- Eval interface: `evaluate(artifact_uri, eval_sets, config) -> report.json` including CIs and slice metrics.

**Testing**
- Unit tests per task; integration tests per phase using a synthetic call fixture set in `tests/fixtures/`.
- Never commit real guest audio or transcripts. Use synthetic or consented, anonymised fixtures only.

## 14. Multi-agent working agreement

- **Claim by task ID.** An agent works one task (or subtask) at a time and states the ID in commits and PRs.
- **Respect ownership.** Each task lists an owner role and the paths it may modify. Do not edit outside those paths without raising it.
- **Contracts first.** If a task depends on a schema or interface that does not exist, create the contract task output first or stop and ask.
- **Definition of done:** acceptance criteria met, tests pass, docs updated, no invariant in §2 violated, and a short handoff note (what changed, what is next, known gaps) appended to `docs/handoffs/<task-id>.md`.
- **Integration checkpoints** in `roadmap.md` are merge gates: downstream lanes start only after the checkpoint is green.
- **Escalate** (stop and ask a human) for: legal or compliance interpretation, changes to the sealed golden set, any change to promotion thresholds, and any conflict with §2.
- **Do not invent data.** Thresholds in this file marked "rule of thumb" are starting points; record the evidence when tuning them.

## 15. Agent roles

| Role | Scope |
|---|---|
| `lead` | Architecture decisions, contract review, integration checkpoints, this file |
| `backend` | Ingest, storage, triage service, review adapter, release services |
| `sdk` | SDK, Pipecat observer, outcome connectors |
| `data` | Dataset builder, splits, exporters, manifests, golden/regression set tooling |
| `ml-train` | Trainer plugins, vocab boost, conversion, hyperparameter safety |
| `ml-eval` | Metrics, bootstrap CIs, eval harness, benchmarks, gate logic |
| `frontend` | Custom review UI |
| `infra` | Docker, compose, CI, MLflow, orchestration, dashboards |
| `governance` | Consent, PII, redaction, retention, lineage, access control |
| `qa` | Integration tests, fixtures, failure injection, release rehearsal |
| `docs` | Service READMEs, runbooks, changelog |

## 16. Assumptions and open questions

Assumptions (revise when answered):
- Python-first stack; telephony audio may be 8 kHz and must be preserved as-is for training.
- Agent is Pipecat-based or can call `log_turn` directly.
- Single project/tenant at first; multi-project support comes later.

Open questions for the human owner:
1. Which STT model and runtime serve calls today (faster-whisper, Vosk, Parakeet, other)?
2. Is the parser an LLM prompt or a trained classifier/slot-filler?
3. Which languages are served? Which jurisdictions apply?
4. What is the booking system's API for fetching the final record, and how is it joined to `call_id`?
5. GPU availability: is there a non-serving GPU, or only a nightly window?
6. Who are the human reviewers, and how many hours per week are available?
