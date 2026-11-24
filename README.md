# VoiceLoop (working title)

A human-in-the-loop improvement system for voice agents. It captures real call turns, finds the ones the agent got wrong, routes them to a human reviewer, turns the corrections into training data, retrains the speech-to-text (STT) model and the text parser, and only releases a new model after it passes an evaluation gate.

> **Status:** design phase. Nothing here is implemented yet. See [`roadmap.md`](./roadmap.md) for the build plan and [`context.md`](./context.md) for the shared reference used by development agents.
>
> "VoiceLoop" is a placeholder name. The reference use case is a hotel-booking voice agent (guest speaks → STT → parser fills a booking form), but the design is not specific to hotels.

---

## Why this exists

Training is the easy part. What decides whether the loop actually improves your agent is:

1. **What you capture** from each call turn.
2. **Which failures reach a human** reviewer.
3. **The gate** a new model must pass before it touches a live call.

No existing tool covers this whole loop for a voice agent with a human transcription step. Voice-observability products record audio, transcripts and tool calls but don't feed retraining. Generic data-flywheel systems target LLMs and use LLM judges instead of a person listening to the audio. VoiceLoop is mostly glue around existing tools plus three purpose-built pieces:

- **Capture and triage**
- **A review UI**
- **An evaluation-and-release gate**

---

## How it works

```
 ┌────────────┐   turn events    ┌──────────────┐  scored   ┌─────────────┐
 │ Voice agent│ ───────────────► │ Ingest + store│ ───────► │   Triage    │
 │ (Pipecat,  │  (never in the   │ (FastAPI,     │          │ (failure    │
 │  other)    │   call path)     │  Postgres, S3)│          │  score,     │
 └─────▲──────┘                  └──────────────┘          │  dedupe,    │
       │                                                    │  daily cap) │
       │ alias poll                                         └──────┬──────┘
       │ (@champion)                                               │ queue
 ┌─────┴──────┐   ┌───────────────┐   ┌──────────────┐   ┌────────▼───────┐
 │ MLflow     │◄──│ Eval gate +   │◄──│ Training     │◄──│  Review UI     │
 │ registry   │   │ shadow/canary │   │ plane (STT,  │   │ (fix transcript│
 │            │   │               │   │ parser)      │   │  → fill slots) │
 └────────────┘   └───────────────┘   └──────▲───────┘   └────────┬───────┘
                                              │                    │
                                       ┌──────┴────────────────────▼──┐
                                       │ Dataset builder (versioned,  │
                                       │ hashed, split by call)       │
                                       └──────────────────────────────┘
```

### 1. Attach layer
A small SDK plus a plain HTTP and OpenTelemetry ingest API. For Pipecat-style apps it is an observer plus a per-turn audio hook; for anything else it is a single `log_turn` call. Each turn event carries:

- the audio exactly as the STT model received it (if telephony is 8 kHz, train on 8 kHz)
- the hypothesis with per-word confidences
- the parser's input and output, and the dialogue context it saw (expected slot, reference date, timezone)
- model version IDs
- later, the outcome: booking confirmed, edited by staff, abandoned, or transferred

The service never sits in the call path, so an outage on its side cannot affect a guest.

### 2. Triage
Every turn gets a failure score from cheap signals:

- low word confidence (for Whisper-family decoders: log-probability, compression ratio and no-speech probability thresholds)
- schema-invalid or low-probability parses
- repair behaviour ("no, I said…", the agent re-asking, transfer to a human)
- disagreement with a larger offline teacher model on the same audio
- mismatch between what the parser extracted and the final booking record (the most valuable signal, because that record is free supervision)

A random slice (3–5%) is kept outside the failure filter, because a test set built only from failures cannot tell you your real error rate. Near-duplicates are clustered so a reviewer sees one example of the same misheard surname, not thirty, and the daily queue is capped.

### 3. Review UI
One screen, two stages:

- **Stage 1: transcript.** Waveform player with speed control and looping; hypothesis with low-confidence words highlighted; edit in place with a diff against the original.
- **Stage 2: slots.** A form generated from your booking JSON Schema, pre-filled by running the current parser on the corrected transcript, shown beside the final booking record, with validators (check-out after check-in, room type exists).

One click routes each example: **STT wrong / parser wrong / both fine / unusable audio**. Some items are sent blind (no prefill) to measure how much prefill anchors reviewers, and anything entering the golden set gets a second opinion.

Version one puts Label Studio Community or Argilla behind an adapter. The custom UI comes after the team knows which clicks hurt.

### 4. Dataset builder
One reviewed item becomes up to three training examples:

1. audio → corrected transcript (STT)
2. corrected transcript → slots (parser)
3. original noisy hypothesis → slots (parser learns to cope with real STT errors)

Each example keeps its dialogue context. Splits are by call, never by utterance. A sealed golden set and a regression set of every past failure are held out, with replay data from earlier verified turns mixed in. Anything without recording consent or containing card numbers is dropped, and each dataset gets a manifest with a content hash. Exporters: NeMo manifest JSONL, Hugging Face datasets, SFT JSONL.

### 5. Training plane
Trainer plugins behind one interface (dataset in → adapter or checkpoint plus metrics out):

- Whisper via Transformers + PEFT
- Parakeet via NeMo
- Parser via Unsloth, TRL or LLaMA-Factory, or via GLiNER2

Jobs run in pinned Docker images, one at a time, on a GPU that is not serving calls (or at night), and are logged to MLflow with the dataset hash and git SHA. For Whisper, use conservative LoRA scaling and merge the adapter before converting to CTranslate2.

### 6. Evaluation gate and release
Three evaluation sets:

- **Golden:** never trained on.
- **Regression:** every past failure.
- **General-capability:** silence, line noise and any other languages served, to catch forgetting and hallucination.

Metrics: WER, entity-level error (names, dates, numbers), parser slot F1, schema-valid rate, and **end-to-end booking-field accuracy from audio** (the number you promote on). Use paired bootstrap confidence intervals, and require a margin over the champion with no regression on critical slices.

Release path: evaluate the **converted deployment artifact** → shadow mode (candidate scores live traffic silently) → small canary → flip the MLflow alias. Rollback is flipping it back.

### 7. Governance
- per-call consent flag
- PII detection on transcripts and timestamp-based redaction in audio
- retention limits on raw audio
- lineage from every training example to its call, so deletion requests are tractable
- roles: reviewer, ML engineer, admin

If you serve EU guests, check AI Act Article 50 transparency duties and your GDPR lawful basis for training on recordings. A trained adapter cannot be surgically unlearned, so pseudonymise before training. This is not legal advice; run it past counsel.

---

## The adaptation ladder

With roughly 10–100 failures per week, retraining the STT model weekly from one week's batch would mostly add risk (100 short utterances is about 8–10 minutes of audio). Each reviewed failure is therefore routed to the cheapest fix that works:

| Tier | Fix | Effort | Risk |
|---|---|---|---|
| 1 | Vocabulary and rules: phrase boosting / LM fusion, alias tables, post-ASR normalisation | Minutes | None to the model |
| 2 | Parser LoRA retrain on gold + noisy transcripts (first trigger around a few hundred verified examples) | Minutes–1 hr on one GPU | Low |
| 3 | STT LoRA/fine-tune with replay data (low thousands of utterances, or sooner for a repeated failure class); TTS-synthesised and noise-augmented variants stretch scarce data | Hours | Medium |
| 4 | Base-model swap when a better model appears, reusing the same benchmark harness | Days | Medium–high |

The thresholds are rules of thumb, not measurements. Tune them against your own eval curves. Trigger retraining on enough new verified data, a drift alarm, or a manual button, not on the calendar alone.

---

## Tech stack

| Layer | Reuse | Build |
|---|---|---|
| Ingest and storage | FastAPI, Postgres, S3-compatible object store | SDK, event schema, outcome connectors |
| Review | Label Studio Community or Argilla via adapter | Custom React UI (waveform + JSON-Schema form) later |
| Orchestration | Prefect or Dagster (cron + CLI is fine at first) | Trigger policies |
| Training | Transformers + PEFT, NeMo, Unsloth/TRL/LLaMA-Factory, GLiNER2 | Trainer plugin interface, dataset exporters |
| Tracking and registry | MLflow with `@champion` / `@challenger` aliases | Promotion logic |
| Serving handoff | Sync agent polling the alias; CTranslate2 or ONNX swap for STT, adapter load for the parser (vLLM runtime LoRA endpoints, JSON-schema constrained decoding) | Shadow and canary routing |
| Monitoring | OpenTelemetry, Prometheus, Grafana | Failure-rate and queue dashboards |

### Model notes
- **Whisper** has the biggest ecosystem (faster-whisper converts any Transformers-compatible Whisper model). Risks: single-language fine-tuning can degrade other languages; one study traced hallucinated text on silent audio to an over-scaled LoRA adapter (alpha/r 2.0 → 0.2 fixed it); converted models can behave differently, so always test the deployed artifact.
- **Parakeet TDT 0.6B v3** is a small alternative (25 European languages) with per-word confidence and word boosting.
- **Qwen3-ASR** (0.6B / 1.7B) ships a reproducible fine-tuning recipe under Apache 2.0.
- Leaderboard rankings move fast; re-benchmark on your own recordings and treat a base-model swap as routine.
- **Parser:** small LLM with LoRA (Unsloth, Axolotl or LLaMA-Factory) served by vLLM, or a GLiNER2 slot-filler running on CPU. Fine-tune for correctness, constrain decoding for validity.

---

## Planned repository layout

```
voiceloop/
├── README.md
├── context.md              # shared agent reference
├── roadmap.md              # tasks and subtasks for multi-agent work
├── sdk/                    # Python SDK + Pipecat observer
├── services/
│   ├── ingest/             # FastAPI ingest + OTel endpoint
│   ├── triage/             # scoring, clustering, queue
│   ├── review_adapter/     # Label Studio / Argilla adapter
│   ├── datasets/           # builder, splits, manifests, exporters
│   ├── trainers/           # plugin interface + whisper/parakeet/parser
│   ├── evalgate/           # metrics, bootstrap CIs, promotion logic
│   └── release/            # shadow, canary, alias sync agent
├── ui/                     # custom review UI (later phase)
├── infra/                  # docker, compose, CI, dashboards
├── schemas/                # event + booking JSON Schemas
└── docs/
```

## Getting started

Not available yet. The first runnable milestone (roadmap Phase 1) will provide:

1. `docker compose up` for Postgres, object store, and the ingest service
2. SDK install and a `log_turn` example
3. A baseline evaluation script producing WER and end-to-end booking accuracy on a hand-labelled golden set

## Phasing (one engineer, rough)

| Weeks | Focus |
|---|---|
| 1–2 | Instrument capture; hand-label a random golden set of 300–500 utterances; measure baseline WER and end-to-end booking accuracy. Nothing else is trustworthy until this exists. |
| 3–5 | Triage, review UI via Label Studio, parser retrain loop, eval gate, manual promotion |
| 6–9 | STT fine-tuning path, artifact conversion tests, shadow mode, vocabulary-boost tier |
| After | Scheduled and drift-triggered retraining, canary with auto-rollback, synthetic augmentation, custom review UI, multi-project support |

## Open questions

- Which STT model and runtime does the agent run today (faster-whisper, Vosk, Parakeet)?
- Is the parser an LLM prompt or a trained classifier?
- Which languages and jurisdictions are served (affects forgetting tests and compliance)?

## License

TBD.
