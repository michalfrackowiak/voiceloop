# roadmap.md — Tasks and Subtasks for Multi-Agent Development

Read `context.md` first. This roadmap breaks VoiceLoop into tasks that can be assigned to parallel agents.

## How to read this file

Each task has:

- **ID**: `P<phase>-<nn>`; subtasks are `P<phase>-<nn>.<k>`
- **Role**: the owner agent role (see `context.md` §15)
- **Deps**: task IDs that must be done first (`—` means none)
- **Paths**: files/directories the task may modify
- **Done when**: acceptance criteria

Status markers: `[ ]` todo, `[~]` in progress, `[x]` done.

**Parallelism rule:** tasks with no unmet deps can run at the same time. Tasks in the same *lane* touch disjoint paths, so they will not conflict. **Checkpoints** (`CP-n`) are merge gates: the lead verifies them before downstream lanes start.

Rough timing assumes one engineer; with parallel agents the calendar time compresses but the dependency order does not.

---

## Phase 0 — Foundations and contracts (days 1–3)

Goal: shared contracts and a runnable skeleton so every later lane can work independently.

- [x] **P0-01 Repo skeleton and tooling** — Role: `infra` — Deps: — — Paths: root, `infra/`, `.github/`
  - [x] P0-01.1 Create the directory layout from `README.md`
  - [x] P0-01.2 Set up `pyproject.toml`, `ruff`, `mypy`, `pytest`, pre-commit
  - [x] P0-01.3 CI pipeline running lint, type check, tests
  - [x] P0-01.4 `docker-compose.yml` with Postgres, MinIO (S3-compatible), MLflow
  - **Done when:** `docker compose up` starts all three; CI is green on an empty test.

- [x] **P0-02 Turn event schema v1** — Role: `lead` — Deps: — — Paths: `schemas/`
  - [x] P0-02.1 Convert the draft in `context.md` §5 to JSON Schema
  - [x] P0-02.2 Define required vs optional fields (incremental adoption)
  - [x] P0-02.3 Provide valid and invalid example payloads
  - **Done when:** schema validates the examples; versioning rule documented.

- [x] **P0-03 Booking JSON Schema (example)** — Role: `lead` — Deps: — — Paths: `schemas/booking.schema.json`
  - [x] P0-03.1 Define slots: check-in, check-out, guests, room type, guest name, contact
  - [x] P0-03.2 Document cross-field validators (check-out after check-in, room type exists)
  - **Done when:** schema and validator function with tests; designed to be swappable per project.

- [x] **P0-04 Interface contracts** — Role: `lead` — Deps: — — Paths: `schemas/`, `docs/`
  - [x] P0-04.1 Trainer interface spec (`context.md` §13)
  - [x] P0-04.2 Eval interface and `report.json` schema (metrics, CIs, slice metrics)
  - [x] P0-04.3 Dataset manifest schema (hash, source calls, splits, filter counts)
  - [x] P0-04.4 Review adapter API spec (push item, pull result, item routing labels)
  - **Done when:** each spec is JSON Schema or OpenAPI with examples.

- [ ] **P0-05 Synthetic fixtures** — Role: `qa` — Deps: P0-02 — Paths: `tests/fixtures/`
  - [ ] P0-05.1 Generate synthetic call audio and event payloads (TTS + noise, including silence and an 8 kHz variant)
  - [ ] P0-05.2 Include known failure cases: misheard surname, date confusion ("three" nights vs guests), schema-invalid parse
  - **Done when:** at least 30 synthetic turns with labelled expected outcomes; no real guest data.

**CP-0 (lead):** schemas and contracts reviewed; compose stack runs; fixtures available.

---

## Phase 1 — Capture and baseline (weeks 1–2)

Goal: measure reality before building anything else. Nothing after this is trustworthy without a golden set and baseline.

### Lane A: Ingest (backend)

- [ ] **P1-01 Ingest API** — Role: `backend` — Deps: P0-01, P0-02 — Paths: `services/ingest/`
  - [ ] P1-01.1 `POST /v1/turns` validating against the schema; idempotent on `event_id`
  - [ ] P1-01.2 Audio upload path to object store (presigned URL or multipart)
  - [ ] P1-01.3 OpenTelemetry-compatible ingest endpoint mapping spans/attributes to turn events
  - [ ] P1-01.4 `PATCH /v1/calls/{call_id}/outcome` for late-arriving outcomes
  - [ ] P1-01.5 Postgres migrations (calls, turns, outcomes, model versions)
  - **Done when:** fixture events ingest end to end; invalid events return structured errors; load test shows ingest does not block on storage failures (async queue or buffered write).

- [ ] **P1-02 Fail-open behaviour** — Role: `backend` — Deps: P1-01 — Paths: `services/ingest/`
  - [ ] P1-02.1 Bounded in-memory/disk buffer when storage is down
  - [ ] P1-02.2 Health and backpressure metrics
  - **Done when:** test kills Postgres/S3 mid-stream; ingest accepts and later flushes, or sheds load with a defined signal; never blocks callers beyond a set timeout.

### Lane B: SDK (sdk)

- [ ] **P1-03 Python SDK** — Role: `sdk` — Deps: P0-02 — Paths: `sdk/`
  - [ ] P1-03.1 `log_turn(...)` with async, non-blocking send and local retry queue
  - [ ] P1-03.2 Timeouts, circuit breaker, drop-on-failure policy (never raise into the call path)
  - [ ] P1-03.3 Audio capture helper that preserves original sample rate
  - **Done when:** SDK tests prove that an unreachable server adds no more than a few ms to the caller and raises no exceptions.

- [ ] **P1-04 Pipecat integration** — Role: `sdk` — Deps: P1-03 — Paths: `sdk/pipecat/`
  - [ ] P1-04.1 Observer capturing per-turn user audio and STT/parser results
  - [ ] P1-04.2 Conversation ID propagation to OpenTelemetry traces
  - [ ] P1-04.3 Example app
  - **Done when:** example Pipecat bot produces correct turn events in a local ingest.

- [ ] **P1-05 Outcome connector** — Role: `sdk` — Deps: P1-01 — Paths: `sdk/connectors/`
  - [ ] P1-05.1 Generic webhook/poller interface for the booking system
  - [ ] P1-05.2 Mapping from booking status to `outcome.status` and `final_record`
  - **Done when:** a mock booking system updates outcomes for fixture calls.

### Lane C: Golden set and baseline (data + ml-eval)

- [ ] **P1-06 Golden set sampling and labelling workflow** — Role: `data` — Deps: P1-01 — Paths: `services/datasets/golden/`
  - [ ] P1-06.1 Random sampler (300–500 utterances) stratified by call, not by failure
  - [ ] P1-06.2 Export to the labelling tool; import labels; second-reviewer flow
  - [ ] P1-06.3 Seal and hash the set; access audit log
  - **Done when:** golden set is versioned, hashed, immutable; second-review coverage recorded.

- [ ] **P1-07 Evaluation harness v0** — Role: `ml-eval` — Deps: P0-04 — Paths: `services/evalgate/`
  - [ ] P1-07.1 WER and entity-level error (names, dates, numbers)
  - [ ] P1-07.2 Parser slot F1 and schema-valid rate
  - [ ] P1-07.3 End-to-end booking-field accuracy from audio
  - [ ] P1-07.4 Paired bootstrap confidence intervals and slice metrics
  - [ ] P1-07.5 `report.json` writer conforming to contract
  - **Done when:** harness runs on fixtures and produces a valid report; unit tests for each metric against hand-computed cases.

- [ ] **P1-08 Baseline measurement** — Role: `ml-eval` — Deps: P1-06, P1-07 — Paths: `docs/baseline.md`
  - [ ] P1-08.1 Run harness on the deployed STT + parser artifacts
  - [ ] P1-08.2 Record baseline with CIs and slice breakdown
  - **Done when:** baseline report committed; becomes the champion reference.

### Lane D: Governance foundations (governance)

- [ ] **P1-09 Consent and PII basics** — Role: `governance` — Deps: P0-02, P1-01 — Paths: `services/ingest/consent/`, `services/datasets/filters/`
  - [ ] P1-09.1 Enforce consent flags on dataset eligibility
  - [ ] P1-09.2 Transcript PII detection (including card numbers) with tests
  - [ ] P1-09.3 Retention policy config for raw audio
  - **Done when:** non-consented and card-containing rows are provably excluded in tests.

**CP-1 (lead):** live or fixture traffic flows through SDK → ingest → storage; golden set sealed; baseline recorded.

---

## Phase 2 — Triage, review, parser loop, gate (weeks 3–5)

Goal: the first complete loop on the cheap path (tiers 1–2), with manual promotion.

### Lane A: Triage (backend)

- [ ] **P2-01 Failure scoring** — Role: `backend` — Deps: CP-1 — Paths: `services/triage/scoring/`
  - [ ] P2-01.1 Signal extractors: word confidence, decoder stats, schema validity, parse probability
  - [ ] P2-01.2 Repair-behaviour detector (guest correction, agent re-ask, transfer)
  - [ ] P2-01.3 Parser-vs-booking-record mismatch signal (highest weight)
  - [ ] P2-01.4 Optional offline teacher-model disagreement signal
  - [ ] P2-01.5 Configurable weights; score explanation stored per turn
  - **Done when:** on fixtures, known failures rank above clean turns; each score shows which signals fired.

- [ ] **P2-02 Sampling, clustering, queue** — Role: `backend` — Deps: P2-01 — Paths: `services/triage/queue/`
  - [ ] P2-02.1 3–5% random slice outside the failure filter, tagged as such
  - [ ] P2-02.2 Near-duplicate clustering (e.g. phonetic/text similarity on hypotheses)
  - [ ] P2-02.3 Daily queue cap and prioritisation
  - [ ] P2-02.4 Blind-item flag (no prefill) for a configurable share of items
  - **Done when:** queue output on fixtures has one representative per cluster, respects cap, includes random slice.

### Lane B: Review (backend; frontend later)

- [ ] **P2-03 Review adapter (Label Studio or Argilla)** — Role: `backend` — Deps: P0-04, P2-02 — Paths: `services/review_adapter/`
  - [ ] P2-03.1 Pick tool (see open questions); stand it up in compose
  - [ ] P2-03.2 Push items with audio, hypothesis, low-confidence highlights
  - [ ] P2-03.3 Two-stage task: transcript correction, then slot form generated from the booking JSON Schema
  - [ ] P2-03.4 Pre-fill stage 2 by running the current parser on the corrected transcript (skipped for blind items)
  - [ ] P2-03.5 Routing label capture: `stt_wrong` / `parser_wrong` / `both_fine` / `unusable_audio`
  - [ ] P2-03.6 Pull back results; store with reviewer ID and timestamps
  - **Done when:** a fixture item completes both stages and its result lands in Postgres; swapping the tool requires changes only inside the adapter.

- [ ] **P2-04 Review UX measurement** — Role: `ml-eval` — Deps: P2-03 — Paths: `docs/review_metrics.md`, `services/review_adapter/metrics/`
  - [ ] P2-04.1 Per-item time, edit distance from hypothesis, label distribution
  - [ ] P2-04.2 Prefill-anchoring comparison (blind vs prefilled agreement)
  - [ ] P2-04.3 Inter-reviewer agreement on golden-set candidates
  - **Done when:** metrics report generated from fixture reviews; used later to decide whether a custom UI is worth building.

### Lane C: Dataset builder (data)

- [ ] **P2-05 Dataset builder** — Role: `data` — Deps: P0-04, P1-09 — Paths: `services/datasets/`
  - [ ] P2-05.1 Expand reviewed item into up to three examples (STT, clean parser, noisy parser) with dialogue context
  - [ ] P2-05.2 Split by call; enforce golden and regression exclusion
  - [ ] P2-05.3 Regression set accumulation (every reviewed failure)
  - [ ] P2-05.4 Replay sampler from earlier verified turns
  - [ ] P2-05.5 Manifest with content hash, source call IDs, filter counts
  - [ ] P2-05.6 Exporters: NeMo manifest JSONL, Hugging Face datasets, SFT JSONL
  - **Done when:** no call appears in more than one split (property test); consent and PII filters verified; exports load in their target libraries.

### Lane D: Parser training (ml-train)

- [ ] **P2-06 Trainer plugin interface** — Role: `ml-train` — Deps: P0-04 — Paths: `services/trainers/core/`
  - [ ] P2-06.1 Base class and registry implementing the contract
  - [ ] P2-06.2 Pinned-image job runner, single-job lock, MLflow logging (dataset hash, git SHA, image digest)
  - **Done when:** a dummy trainer runs through the runner and logs a run with all lineage fields.

- [ ] **P2-07 Parser trainer** — Role: `ml-train` — Deps: P2-05, P2-06 — Paths: `services/trainers/parser/`
  - [ ] P2-07.1 LoRA SFT trainer (Unsloth, TRL or LLaMA-Factory) for gold and noisy examples
  - [ ] P2-07.2 Optional GLiNER2 trainer for slot-filler deployments
  - [ ] P2-07.3 Adapter export and vLLM load/unload test; JSON-schema constrained decoding check
  - **Done when:** a small fixture dataset trains end to end, adapter loads at runtime, outputs are schema-valid under constraint.

### Lane E: Tier 1 fixes (ml-train)

- [ ] **P2-08 Vocabulary and rules tier** — Role: `ml-train` — Deps: CP-1 — Paths: `services/trainers/vocab/`
  - [ ] P2-08.1 Alias tables for names and room types
  - [ ] P2-08.2 Post-ASR normalisation rules (numbers, dates)
  - [ ] P2-08.3 Phrase boosting / LM fusion config generator for the current STT runtime
  - **Done when:** each rule type fixes a fixture failure with no regression on the golden set (measured by the harness).

### Lane F: Evaluation gate (ml-eval)

- [ ] **P2-09 Gate and promotion logic** — Role: `ml-eval` — Deps: P1-07, P2-06 — Paths: `services/evalgate/gate/`
  - [ ] P2-09.1 Regression and general-capability evaluation (silence, noise, other languages)
  - [ ] P2-09.2 Pass rule: margin over champion on end-to-end accuracy, no critical-slice regression, using CI bounds
  - [ ] P2-09.3 Evaluate the converted artifact (hook for conversion step)
  - [ ] P2-09.4 MLflow alias promotion command (`@challenger` → `@champion`) with audit record; one-step rollback
  - **Done when:** a deliberately worse fixture model fails the gate; a better one passes; rollback restores the previous champion.

### Lane G: Orchestration and monitoring (infra)

- [ ] **P2-10 Pipelines and triggers** — Role: `infra` — Deps: P2-05, P2-06 — Paths: `infra/pipelines/`
  - [ ] P2-10.1 CLI and cron for: build dataset → train → evaluate
  - [ ] P2-10.2 Trigger policy: enough new verified data, drift alarm, or manual (not calendar alone)
  - **Done when:** manual trigger runs the full chain on fixtures.

- [ ] **P2-11 Dashboards** — Role: `infra` — Deps: P1-01, P2-02 — Paths: `infra/dashboards/`
  - [ ] P2-11.1 Failure rate, queue depth, review throughput
  - [ ] P2-11.2 Model version in service and last promotion
  - **Done when:** dashboards render from fixture or local data.

### Lane H: Governance (governance)

- [ ] **P2-12 Lineage and access control** — Role: `governance` — Deps: P2-05 — Paths: `services/datasets/lineage/`, `services/*/auth/`
  - [ ] P2-12.1 Lineage queries: training example → call; call → every model trained on it
  - [ ] P2-12.2 Deletion workflow that removes a call from future datasets and flags affected models
  - [ ] P2-12.3 Roles: reviewer, ML engineer, admin
  - **Done when:** deleting a fixture call produces a report of impacted datasets and models.

### Lane I: Integration (qa)

- [ ] **P2-13 Loop rehearsal** — Role: `qa` — Deps: P2-03, P2-07, P2-09 — Paths: `tests/integration/`
  - [ ] P2-13.1 Automated test: fixture traffic → triage → simulated review → dataset → parser train → gate → manual promotion
  - [ ] P2-13.2 Failure injection: bad data, regressing model, storage outage
  - **Done when:** the test passes in CI; injected failures are caught by the intended stage.

**CP-2 (lead):** first full loop on the parser path works with manual promotion; all invariants in `context.md` §2 have at least one test.

---

## Phase 3 — STT path, release safety (weeks 6–9)

Goal: safely add STT improvement (tier 3) and staged release.

### Lane A: STT training (ml-train)

- [ ] **P3-01 Whisper trainer** — Role: `ml-train` — Deps: CP-2 — Paths: `services/trainers/whisper/`
  - [ ] P3-01.1 Transformers + PEFT LoRA with conservative alpha/r (default well below 2.0; document the evidence)
  - [ ] P3-01.2 Replay data mixing; 8 kHz-aware preprocessing
  - [ ] P3-01.3 Merge adapter, then convert to CTranslate2
  - **Done when:** fixture fine-tune produces a converted model that transcribes fixtures; hyperparameters and scaling logged.

- [ ] **P3-02 Parakeet/NeMo trainer** — Role: `ml-train` — Deps: CP-2 — Paths: `services/trainers/nemo/`
  - [ ] P3-02.1 Fine-tune from checkpoint using the NeMo manifest export
  - [ ] P3-02.2 Per-word confidence output wired to the triage signal
  - **Done when:** fixture fine-tune runs; confidences flow into events.

- [ ] **P3-03 Augmentation** — Role: `ml-train` — Deps: P3-01 — Paths: `services/datasets/augment/`
  - [ ] P3-03.1 TTS-synthesised variants for repeated failure classes (e.g. new name lists)
  - [ ] P3-03.2 Noise and telephony-band augmentation
  - [ ] P3-03.3 Tag all synthetic data; keep it out of golden and regression sets
  - **Done when:** synthetic examples are labelled as such in manifests and excluded from evaluation sets.

### Lane B: Artifact safety (ml-eval)

- [ ] **P3-04 Conversion and artifact tests** — Role: `ml-eval` — Deps: P3-01 — Paths: `services/evalgate/artifact/`
  - [ ] P3-04.1 Run the gate on the exact deployed artifact and decoding settings
  - [ ] P3-04.2 Test combinations: hotwords, initial prompt, `condition_on_previous_text`; detect empty/truncated output
  - [ ] P3-04.3 Silence/no-speech hallucination test set and metric
  - [ ] P3-04.4 Multilingual forgetting test on other served languages
  - **Done when:** a fixture model with injected empty-output behaviour is caught; hallucination metric reported in every STT report.

- [ ] **P3-05 Benchmark harness for base-model swaps** — Role: `ml-eval` — Deps: P1-07 — Paths: `services/evalgate/benchmarks/`
  - [ ] P3-05.1 Pluggable candidate-model runner (Whisper variants, Parakeet, Qwen3-ASR, others)
  - [ ] P3-05.2 Comparison report on our own golden set, not public leaderboards
  - **Done when:** two candidate models compared in one command with CIs.

### Lane C: Release (backend)

- [ ] **P3-06 Alias sync agent** — Role: `backend` — Deps: P2-09 — Paths: `services/release/sync/`
  - [ ] P3-06.1 Poll MLflow `@champion`; download artifact; atomic swap on the serving side
  - [ ] P3-06.2 STT swap (CTranslate2/ONNX) and parser adapter load/unload via vLLM endpoints
  - [ ] P3-06.3 Health check after swap; auto-revert on failure
  - **Done when:** flipping the alias updates a local serving stack; a broken artifact auto-reverts.

- [ ] **P3-07 Shadow mode** — Role: `backend` — Deps: P3-06 — Paths: `services/release/shadow/`
  - [ ] P3-07.1 Mirror live turn audio to the candidate; log results without affecting the call
  - [ ] P3-07.2 Candidate-vs-champion comparison on live traffic with outcome-based scoring
  - **Done when:** shadow runs add no call latency (verified) and produce a comparison report.

- [ ] **P3-08 Promotion workflow update** — Role: `ml-eval` — Deps: P3-04, P3-07 — Paths: `services/evalgate/gate/`
  - [ ] P3-08.1 Require shadow evidence before promotion for STT changes
  - **Done when:** gate refuses STT promotion without shadow results.

### Lane D: Review quality (data)

- [ ] **P3-09 Golden-set maintenance** — Role: `data` — Deps: CP-2 — Paths: `services/datasets/golden/`
  - [ ] P3-09.1 Process to refresh or extend the golden set with audit and second review
  - [ ] P3-09.2 Drift check: compare incoming traffic distribution to golden set
  - **Done when:** extension is versioned, audited, and the previous set stays reproducible.

### Lane E: Governance (governance)

- [ ] **P3-10 Audio redaction and pseudonymisation** — Role: `governance` — Deps: P1-09 — Paths: `services/datasets/redaction/`
  - [ ] P3-10.1 Timestamp-based audio redaction for detected PII spans
  - [ ] P3-10.2 Pseudonymisation of names in transcripts used for training
  - [ ] P3-10.3 Retention job deleting expired raw audio
  - **Done when:** redacted fixtures contain no PII spans in audio or text; retention job tested.

- [ ] **P3-11 Compliance checklist** — Role: `governance` — Deps: — Paths: `docs/compliance.md`
  - [ ] P3-11.1 List AI Act Article 50 transparency and GDPR lawful-basis questions for counsel
  - [ ] P3-11.2 Map data flows and retention for review
  - **Done when:** checklist written and flagged for human legal review. Do not mark any compliance conclusion as final.

### Lane F: Integration (qa)

- [ ] **P3-12 STT loop rehearsal** — Role: `qa` — Deps: P3-01, P3-04, P3-07 — Paths: `tests/integration/`
  - [ ] P3-12.1 End-to-end test of STT fine-tune → conversion → gate → shadow → promotion → rollback
  - [ ] P3-12.2 Failure injection: hallucinating adapter, empty output after conversion, forgetting in another language
  - **Done when:** each injected failure is caught before promotion.

**CP-3 (lead):** STT path gated end to end with shadow evidence; rollback rehearsed.

---

## Phase 4 — Hardening and scale (after week 9)

Tasks are independent unless noted; each can run as its own lane.

- [ ] **P4-01 Canary with auto-rollback** — Role: `backend` — Deps: CP-3
  - [ ] P4-01.1 Traffic split config and canary metrics
  - [ ] P4-01.2 Automatic rollback on outcome-metric regression
  - **Done when:** simulated regression during canary triggers rollback within the configured window.

- [ ] **P4-02 Scheduled and drift-triggered retraining** — Role: `infra` — Deps: CP-3
  - [ ] P4-02.1 Drift detectors on triage score distribution and outcome rates
  - [ ] P4-02.2 Trigger policy combining data volume, drift, and manual override
  - **Done when:** drift in a replayed dataset triggers the pipeline; no calendar-only retraining.

- [ ] **P4-03 Custom review UI** — Role: `frontend` — Deps: P2-04, CP-3
  - [ ] P4-03.1 Decide go/no-go from review metrics (which clicks hurt)
  - [ ] P4-03.2 Waveform player with speed and loop; low-confidence highlighting; diff view
  - [ ] P4-03.3 JSON-Schema-generated slot form with validators and booking record side panel
  - [ ] P4-03.4 Routing buttons; blind-item mode; second-opinion flow
  - **Done when:** usability test shows time per item no worse than the adapter-based tool.

- [ ] **P4-04 Teacher-model triage signal at scale** — Role: `ml-train` — Deps: P2-01
  - [ ] P4-04.1 Batch offline transcription with a larger model; disagreement scoring
  - **Done when:** signal improves precision of the review queue against reviewed labels.

- [ ] **P4-05 Multi-project support** — Role: `lead` — Deps: CP-3
  - [ ] P4-05.1 Project/tenant scoping across schemas, storage, and registry
  - [ ] P4-05.2 Per-project booking schemas and eval sets
  - **Done when:** two projects run side by side with no data crossing.

- [ ] **P4-06 Operations docs** — Role: `docs` — Deps: CP-3
  - [ ] P4-06.1 Runbooks: onboarding a project, retraining, rollback, deletion requests
  - **Done when:** each runbook was followed by someone who did not write it.

---

## Dependency and parallelism summary

| Phase | Parallel lanes that can run at once | Merge gate |
|---|---|---|
| 0 | `infra` (P0-01), `lead` (P0-02 to P0-04), then `qa` (P0-05) | CP-0 |
| 1 | A ingest, B SDK, C golden/baseline, D governance | CP-1 |
| 2 | A triage, B review, C datasets, D parser training, E vocab, F gate, G pipelines, H governance, then I rehearsal | CP-2 |
| 3 | A STT training, B artifact safety, C release, D golden maintenance, E governance, then F rehearsal | CP-3 |
| 4 | All independent | n/a |

Critical path: `P0-02 → P1-01 → P1-06 → P1-08 → P2-01 → P2-02 → P2-03 → P2-05 → P2-07 → P2-09 → P3-01 → P3-04 → P3-07 → P3-08`.

## Decisions pending (blockers for specific tasks)

| Question | Blocks | Default if unanswered |
|---|---|---|
| Review tool: Label Studio Community vs Argilla | P2-03 | Label Studio Community, behind the adapter |
| Current STT model and runtime | P2-08, P3-01, P3-02, P3-06 | Assume faster-whisper (CTranslate2) |
| Parser type: LLM prompt vs trained classifier/slot-filler | P2-07 | Assume small LLM + LoRA served with vLLM |
| Languages served | P3-04.4 | English only until stated |
| Booking system API and `call_id` join | P1-05 | Mock connector |
| GPU availability for training | P2-10, P3-01 | Nightly window on a non-serving GPU |
| Reviewer capacity (hours/week) | P2-02.3 | Queue cap sized to 5 hours/week |

## Handoff template (`docs/handoffs/<task-id>.md`)

```
Task: <ID and title>
Agent role:
What changed (files/paths):
How to verify:
Known gaps / follow-ups:
Contracts touched (schema versions):
Next task(s) unblocked:
```
