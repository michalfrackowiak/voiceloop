# P2-13 loop rehearsal — scoped integration tests

**SYNTHETIC / FIXTURE STAND-IN. NOT CP-2 EVIDENCE. NOT A PRODUCTION CLAIM.**

This package rehearses the parser loop with committed modules wired together
through their public APIs. Everything is fake-backed: in-memory SDK transport,
P2-01 scoring, P2-02 queue, P2-03 scripted two-stage review, P2-05 dataset
builder/exporters, P2-07 fake LoRA trainer, P1-07 evaluator, P2-09 gate and
P1-02 fail-open buffer. No Docker, network, GPU, real review tool, real Postgres
or real MLflow is used.

## Run

```powershell
# scoped gates
.\.venv\Scripts\python.exe -m pytest tests/integration -q
.\.venv\Scripts\python.exe -m ruff check tests/integration
.\.venv\Scripts\python.exe -m ruff format --check tests/integration
.\.venv\Scripts\python.exe -m mypy tests/integration

# full-repo gates (other lanes may be red; report as notes, do not patch here)
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check . --output-format concise
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy .
```

The committed fixture and sealed golden artifacts are hash-guarded by an
autouse fixture. Pinned values:

- `tests/fixtures/turns.jsonl` SHA-256:
  `aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd`
- synthetic golden content hash:
  `662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e`

## Public API and fake seams

| Stage | Public API | Injected fake / seam |
|---|---|---|
| SDK ingest | `sdk.SdkClient`, `sdk.InMemoryTransport` | background worker validates and delivers to memory only |
| Triage scoring | `services.triage.scoring.ScoringEngine`, `InMemoryExplanationStore` | deterministic fixture signals, no teacher service |
| Review queue | `services.triage.queue.build_queue`, `ExplanationScoreSource`, `InMemoryTurnSource` | seeded random slice, blind items, fake scores for independence assertion |
| Review | `services.review_adapter.ReviewCoordinator`, `QueueReportSource`, `InMemoryReviewTool`, `ScriptedReview` | answers derived only from synthetic fixture labels; no human review |
| Dataset | `services.datasets.build_dataset`, `build_manifest`, `content_hash`, `export_all` | sealed golden call IDs, consent/PII filters, plain JSON/JSONL exports |
| Parser training | `services.trainers.core.TrainerRunner`, `TrainerRegistry`, `services.trainers.parser.ParserLoraTrainer`, `SftJsonlExampleLoader` | deterministic fake adapter and in-memory MLflow logger |
| Evaluation | `services.evalgate.harness.evaluate`, `EvalgateSettings` | identity/oracle fake artifacts on a derived fixture |
| Gate/promotion | `services.evalgate.gate.Gate`, `PrefixArtifactConverter`, `StaticReportEvaluator`, `InMemoryAliasStore`, `PromotionAuditLog` | converted-URI lineage check, alias flip and hash-chained audit |
| Lineage / tombstones | `services.datasets.lineage.LineageGraph`, `P205ExampleIdResolver`, `FakeManifestSource`, `FakeTrainerLineageSource`, `InMemoryTombstoneSink`, `guard_manifest` | exact example index injected from committed build; in-memory fake manifest/trainer/tombstone stores; pseudonym-only guard error |
| Fail-open | `services.ingest.failopen.BoundedBuffer`, `RecordingSink`, `FailOpenSettings` | bounded queue, down-sink injection and health counters |

No service source, schema, fixture, unit test or configuration file is modified
by this package.

## Tests

- `test_parser_loop.py::test_fixture_loop_rehearsal_with_manual_promotion`
  rehearses the full loop, asserts hash/ID lineage between stages, resolves
  every example to its exact call with `P205ExampleIdResolver` and
  `LineageGraph`, promotes manually through the fake gate, then exercises
  one-step rollback.
- `test_failure_injection.py::test_bad_data_is_excluded_before_training`
  injects a non-consented row and a card-bearing row; both are dropped before
  export/training and no card digits appear in exports.
- `test_failure_injection.py::test_card_number_as_mapping_key_is_still_leaked`
  is a strict `xfail` documenting a known cross-lane defect: the dataset PII
  screener checks mapping values, not mapping keys.
- `test_failure_injection.py::test_regressing_model_gate_refusal_preserves_aliases_and_audit`
  proves that a WER improvement with an e2e regression fails closed with no
  alias or audit mutation.
- `test_failure_injection.py::test_tombstoned_call_excluded_before_hash_and_guard_fails_closed`
  injects a fake tombstone, excludes its call before hashing, verifies a guarded
  manifest and proves that an unguarded manifest is rejected with a pseudonym.
- `test_failure_injection.py::test_storage_outage_is_bounded_and_visible`
  proves that a down sink is shed/retried within a bound and reported in health.

## CP-2 invariant matrix

Each of the 11 invariants in `context.md` §2 has at least one test assertion.

| # | Invariant | Test(s) and assertion |
|---|---|---|
| 1 | Never in the call path | `test_fixture_loop...`: `SdkClient.enqueue` is accepted without awaiting delivery; `test_storage_outage...`: enqueue and flush return under 0.5 s with a down sink. |
| 2 | A human listens to audio | `test_fixture_loop...`: P2-03 stage 1/2 timestamps and reviewer ID are recorded; README/handoff state the reviewer is a scripted fake, not a human. |
| 3 | Evaluate the deployed artifact | `test_fixture_loop...`: gate converts the trainer URI to a distinct `converted://` URI and evaluates that converted URI with the fake evaluator. |
| 4 | Promote on e2e accuracy from audio | `test_fixture_loop...`: gate passes only when the paired e2e delta lower bound exceeds the margin; `test_regressing_model...`: lower WER with negative e2e delta fails. |
| 5 | Golden set is sealed | `conftest.sealed_golden`: `verify_seal` passes and pins the content hash; full-loop test excludes all sealed call IDs from dataset and SFT loader. |
| 6 | Unbiased random slice | `test_fixture_loop...`: queue has a tagged random item and blind item; rebuilding with uniform scores yields the same random-slice IDs. |
| 7 | Split by call, never utterance | `test_fixture_loop...`: `assert_no_call_in_multiple_splits` passes; manifest source calls and SFT call IDs exclude sealed golden calls. `test_tombstoned_call...`: a denied call is excluded before hashing and accepted by `guard_manifest`. |
| 8 | Consent and PII first | `test_bad_data...`: consent denial and card-value hard-drop are counted; no card digits in NeMo/HF/SFT exports. Mapping-key gap is an explicit xfail lead note. |
| 9 | Lineage everywhere | `test_fixture_loop...`: a `LineageGraph` uses `P205ExampleIdResolver` over `bundle.build.examples`, `FakeManifestSource([bundle.manifest])` and `FakeTrainerLineageSource([training.result])`; every `example_to_call(...).call_id` equals the committed example's call, and `models_for_call(...)` returns the single fake run joined by dataset hash (`content_hash` = trainer `lineage.dataset_hash`). Also chains event/queue/review IDs to dataset hash, logger run, converted URI and gate report. |
| 10 | Rollback is one step | `test_fixture_loop...`: promotion and rollback are single alias flips with two hash-chained audit records; `test_regressing_model...`: refusal changes neither. |
| 11 | Cheapest fix first | `test_fixture_loop...`: review routing labels are preserved into `ReviewedItem.routing`/`failure_type` for the adaptation ladder; no retraining decision is inferred from fixture labels alone. |

## Explicit conflicts and scope tensions with `context.md`

1. **Sealed golden covers all fixture calls.** `services/datasets/golden/artifacts/golden.v1.json`
   seals all six synthetic calls. Exercising training therefore requires labelled
   non-golden `fixture-standin-*` call IDs cloned from the same synthetic
   reviewed content. These are fixture stand-ins, not real labels and not a
   human-reviewed dataset; this is a scope tension with §2.5/§2.7, not a CP-2 pass.
2. **Deployed-artifact invariant under fakes.** §2.3 requires the converted
   serving artifact. This rehearsal only proves URI-level conversion via
   `PrefixArtifactConverter` and the fake merged-adapter stand-in; real
   quantisation/serving remains a lead/CI follow-up.
3. **Human review and real PII tooling are absent.** §2.2 and §2.8 are exercised
   through scripted fakes and public filter APIs, not a review tool, human
   listener or production redaction service.
4. **Card-number mapping-key defect.** `docs/LEAD_BRIEFS.md` requires this case.
   The current builder screens mapping values only, so a consented card number
   used as a key leaks to HF/SFT exports. It is captured as a strict xfail and
   lead note, not patched here.
5. **Thresholds remain unsigned.** `GateSettings` defaults (`min_e2e_margin=0.01`,
   `min_ci_level=0.95`, `slice_regression_tolerance=0.0`) are engineering
   defaults; §14 human sign-off is still pending. No threshold is changed.
6. **Random-slice caveat.** 1/36 fixture turns is below the 3–5% band; the queue
   uses its documented deterministic small-pool override. The test proves score
   independence, not statistical validity.
7. **Example-index contract gap.** `dataset_manifest` 1.0.0 has `source_call_ids`
   but no per-example IDs. The rehearsal injects `P205ExampleIdResolver` over
   committed `example.example_id` values; no production example index or real
   lineage store is exercised.

The `other_languages` slice evidence needed by the gate uses a temporary
derived fixture with `expected_transcript` suffixed by ` (SYNTHETIC-\u00d8)`.
The committed fixture is never edited; this is not a multilingual capability
claim.
