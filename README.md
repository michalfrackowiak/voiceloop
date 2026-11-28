# VoiceLoop

VoiceLoop is a Python reference implementation for a human-reviewed improvement loop for voice agents: capture turn events, identify likely failures, prepare review and training data, evaluate candidate changes, and model a gated release workflow.

> **Project status:** The repository contains working components and synthetic end-to-end rehearsals, but the reviewed checkpoints are limited to fake-backed and synthetic-fixture behavior. It is **not production-ready**: real model training and serving, production storage adapters, human-reviewed data, live traffic, and several Phase 4 acceptance items remain open. See [build progress](docs/PROGRESS.md) for the current scope and blockers.

The reference use case is a voice agent that turns speech into structured booking fields. The components are intended to be reusable for other voice-agent workflows; the booking schema is an example, not a universal domain model.

## What the repository contains

- **Python SDK and integrations:** asynchronous, fail-open turn logging, native-sample-rate audio metadata, a Pipecat observer example, and webhook/poller outcome connector interfaces.
- **Ingest API:** FastAPI endpoints for turn events, audio-upload flows, outcomes, and OpenTelemetry-compatible spans. The default repository and object-store backends are in-memory fakes.
- **Triage and review workflow:** deterministic failure scoring, sampling and queue building, review-tool adapter contracts, and review UX measurement utilities.
- **Dataset and governance utilities:** call-level splits, manifests and exporters, consent and PII filters, golden-set workflow utilities, lineage, and audio-redaction tooling.
- **Training and evaluation seams:** trainer plugins for parser, Whisper, and NeMo/Parakeet workflows; vocabulary rules; evaluation metrics and reports; artifact probes; and gate logic. Shipped model-training and serving implementations are fake or fixture-backed, not real fine-tuning or inference systems.
- **Release and operations prototypes:** shadow, canary, alias-sync, rollback, pipeline, drift, and dashboard modules with injectable interfaces and fake-backed workflows.
- **Offline review UI prototype:** a static browser artifact built from synthetic fixture data; it does not play audio or connect to a review service.

The project is designed around fail-open capture, human review for labels, call-level dataset splits, consent and PII screening, artifact-aware evaluation, and reversible promotion. See [the shared project reference](context.md) and [contracts](docs/contracts.md) for the detailed invariants and interfaces.

## Architecture

```text
Voice agent
    │  SDK / turn events (asynchronous, fail-open)
    ▼
Ingest API ──► Triage and review queue ──► Human-review adapter seam
    │                                         │
    │                                         ▼
    └────────────────────────────────── Dataset builder
                                              │
                         ┌────────────────────┴────────────────────┐
                         ▼                                         ▼
                  Trainer plugins                         Evaluation gate
                                                                    │
                                                                    ▼
                                                   Shadow / canary / alias workflows
```

The implementation is organized by capability:

| Path | Responsibility |
| --- | --- |
| `sdk/` | Python SDK, Pipecat observer, outcome connectors |
| `services/ingest/` | FastAPI event ingest, validation, audio upload interfaces, storage queue |
| `services/triage/` | Failure scoring, sampling, clustering, review queue |
| `services/review_adapter/` | Review-tool contracts and review metrics |
| `services/datasets/` | Dataset construction, filtering, golden data, lineage, export and redaction |
| `services/trainers/` | Trainer interfaces, parser/STT stand-ins, vocabulary rules |
| `services/evalgate/` | Metrics, benchmark reports, artifact probes and promotion gate |
| `services/release/` | Shadow, canary, alias synchronization and rollback workflows |
| `infra/` | Local Compose stack, pipelines, drift checks and dashboards |
| `schemas/` | Turn, booking, dataset, evaluation and service contracts |
| `ui/` | Offline static review UI prototype |
| `tests/` | Unit tests, synthetic fixtures and fake-backed integration rehearsals |

## Requirements

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Docker Compose only if you want to run the optional local infrastructure stack

The default test and fixture workflows do not require a GPU, model downloads, database, object store, or external review tool.

## Getting started

Install the project and development tools from the repository root:

```sh
uv sync --extra dev
```

Run the test suite and quality checks:

```sh
uv run pytest -q
uv run ruff check .
uv run mypy .
```

The integration suite is fake-backed and can also be run separately:

```sh
uv run pytest tests/integration -q
```

### Run selected local workflows

Start the ingest API with its default in-memory backends:

```sh
uv run uvicorn services.ingest.app:app --reload --port 8000
```

Run the evaluation harness against the committed synthetic fixtures:

```sh
uv run python -m services.evalgate.harness
```

Score fixture turns for triage:

```sh
uv run python -m services.triage.scoring.scoring
```

These commands exercise local reference workflows; fixture results are not evidence of real model quality or production readiness. See each component's README for its settings, contracts, and additional commands.

### Optional local infrastructure

The Compose file starts Postgres 16, MinIO, and MLflow for local development. It does **not** wire every application component to production database or object-store adapters.

```sh
# Optional: copy the development-only defaults before starting the stack.
# On PowerShell, use: Copy-Item .env.example .env
docker compose up -d
docker compose ps
docker compose down
```

Default endpoints are Postgres at `localhost:5432`, MinIO at `http://localhost:9000` (console `http://localhost:9001`), and MLflow at `http://localhost:5000`. Values in `.env.example` are development placeholders only; do not use them for shared or production deployments.

## Tests and continuous integration

Tests are organized under `tests/unit/`, `tests/integration/`, and `tests/fixtures/`. The fixtures contain 36 labelled turns across six synthetic calls. Their WAV files are tone, noise, or silence carriers—not intelligible speech or real guest recordings. Integration tests connect project components through in-memory fakes and verify contracts and failure-handling behavior.

GitHub Actions runs Ruff linting, mypy, and pytest on Python 3.11 for pushes to `main` and pull requests. The same checks can be run locally with the commands above. Some release and usability acceptance criteria still require human or real-system evidence; passing fixture tests does not satisfy those criteria.

## Documentation

- [Build progress and current blockers](docs/PROGRESS.md)
- [Project invariants and technical reference](context.md)
- [Roadmap and acceptance criteria](roadmap.md)
- [Contracts](docs/contracts.md)
- [Local infrastructure](infra/README.md)
- [Runbooks](docs/runbooks/README.md)
- Component-specific READMEs are located alongside their code.
