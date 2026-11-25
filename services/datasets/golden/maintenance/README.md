# P3-09 golden-set maintenance (fixture-only rehearsal)

SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN-REVIEWED GOLDEN SET

This package is an audited **rehearsal** for refreshing or extending the sealed
P1-06 golden workflow. It never edits the sealed artifacts, never promotes a
model, never changes a threshold and never claims human review or live-traffic
evidence. Every candidate is a separate synthetic candidate set derived from
`tests/fixtures/turns.jsonl`; the previous `golden.v1` stand-in stays immutable
and reproducible by content hash and seal.

## Run

Use a maintenance-owned output directory. The default is
`services/datasets/golden/maintenance/artifacts`, which is outside the sealed
`services/datasets/golden/artifacts` directory.

```powershell
.\.venv\Scripts\python.exe -m services.datasets.golden.maintenance.cli propose `
  --output-dir <temp-output-dir> --actor data-steward `
  --purpose "P3-09 synthetic rehearsal"

.\.venv\Scripts\python.exe -m services.datasets.golden.maintenance.cli verify `
  --output-dir <temp-output-dir> --version candidate.v1

.\.venv\Scripts\python.exe -m services.datasets.golden.maintenance.cli drift `
  --output <temp-output-dir>/drift.json
```

`propose` requires `--actor` (a short pseudonymous ID from the trusted
allowlist, never a reviewer) and `--purpose` (a PII-screened reason). Add
`--predecessor candidate.v1` for a `candidate.v2` rehearsal; a version after
`candidate.v1` without a verified predecessor fails closed. `drift` prints a
summary and writes a full JSON report only when `--output` is supplied.
`verify` is read-only, appends no audit entry and validates manifest/bundle
reviewer IDs against the trusted settings, never against IDs from the artifact.

Typed settings read only `VOICELOOP_GOLDEN_MAINT_*` environment variables. For
example, `VOICELOOP_GOLDEN_MAINT_OUTPUT_DIR` overrides the output directory,
`VOICELOOP_GOLDEN_MAINT_CANDIDATE_CALLS` changes the whole-call cap and
`VOICELOOP_GOLDEN_MAINT_ALLOWED_ACTOR_IDS` sets a comma-separated actor
allowlist. The pinned
fixture hash `aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd`
and sealed hash `662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e`
are checked on every run; a mismatch aborts.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/golden_maintenance -q
.\.venv\Scripts\python.exe -m ruff check services/datasets/golden/maintenance tests/unit/golden_maintenance
.\.venv\Scripts\python.exe -m ruff format --check services/datasets/golden/maintenance tests/unit/golden_maintenance
.\.venv\Scripts\python.exe -m mypy services/datasets/golden/maintenance tests/unit/golden_maintenance
```

The tests freeze `voiceloop_common.utcnow()` and use temporary output paths.
They assert sealed artifact and fixture hashes before and after every write
path, two distinct reviewer actor IDs, audited reasons/timestamps/hashes,
per-candidate second-review coverage, unlisted/reviewer actor rejection,
failure on a missing or tampered audit prefix, trusted reviewer pins during
verification, predecessor reproducibility, overwrite refusal, the mandatory
human-review gate and descriptive-only drift output.

## Contract

All written artifacts are canonical JSON (sorted keys, no optional spaces),
carry `label = "SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN-REVIEWED GOLDEN SET"`
and contain no audio bytes, transcripts, slot values, names, emails, phone
numbers, card digits or credentials. IDs are pseudonymous `<prefix>-<sha256>`
values.

For each `candidate.vN` the output directory contains:

- `candidate.vN.candidates.json` — candidate records plus one review bundle.
  Candidate fields are limited to `candidate_id`, `call_ref`, `turn_index`,
  `audio_kind`, `sample_rate_hz`, `slot_present` and `source_digest`. Raw call
  IDs and `failure_type`/`case_id` never leave the fixture projection.
- `review_audit.jsonl` — append-only hash chain. Each entry records
  `actor_id`, `action`, `reason`, `task_id`, `timestamp`, `version`,
  `content_hash`, `label`, `previous_hash` and `entry_hash`. Actions are
  `candidate_build`, `first_review`, `second_review`, `version_write` and
  `version_seal`.
- `candidate.vN.json` — immutable manifest with candidate count, content hash,
  reviewer IDs, source fixture hash, sealed golden version/hash/seal hash,
  predecessor version/hash/seal hash and the sealed audit pointer.
- `candidate.vN.seal.json` — manifest hash, content hash, seal actor, second
  reviewer ID, predecessor content hash and `immutable: true`.

Approval requires two distinct scripted reviewer IDs, one `include` answer from
each for every candidate, matching SHA-256 answer hashes and matching audit
entries. The primary candidate IDs must form exactly the recorded candidate set,
with no duplicate or missing candidate. A single reviewer, a duplicated or
missing candidate pair, a disagreement, a missing audit entry, a broken chain,
a modified manifest/audit prefix, an unlisted proposal actor or a reviewer used
as the proposal actor fails closed before any version is sealed. Verification
uses reviewer IDs from trusted settings, rejects manifest/bundle IDs that do not
match those pins, re-reads the sealed golden artifact and compares its content
hash and seal with the recorded values; no P1-06 file is opened for writing.

The drift report compares only failure-blind categorical metadata
(`audio_kind`, `sample_rate_hz`, `slot_present`) for the sealed source keys and
a deterministic synthetic fixture replay. It always reports explicit turn and
call counts, per-feature counts/shares and deltas. Its interpretation is fixed:
`mode = descriptive_only`, `causal_claim = none`, `alarm = not_evaluated`,
`promotion = not_evaluated`, `threshold_change = none`, and
`human_review_gate = required_before_any_golden_or_threshold_change` at both the
top level and inside `interpretation`. Reports missing either gate field are
invalid. No threshold, alias, review rule or promotion decision is made.

These are internal task-local shapes, not changes to `schemas/`. The sealed
`golden.v1` artifact and `dataset_manifest.schema.json` 1.0.0 are consumed
read-only.

## Public API

Import from `services.datasets.golden.maintenance`:

- Settings/helpers: `GoldenMaintenanceSettings`, `ARTIFACT_LABEL`,
  `SEALED_CONTENT_HASH`, `FIXTURE_SHA256`, `canonical_json`, `sha256_text`,
  `sha256_file`, `assert_safe_text`, `assert_allowed_actor`,
  `assert_outside_sealed`, `next_version`.
- Candidates: `CandidateRecord`, `CandidateSet`, `FixtureProjection`,
  `build_candidate_set`, `load_projected_turns`, `select_projected_calls`,
  `candidate_content_hash`, `candidate_document`, `load_candidate_document`,
  `verify_fixture_hash`.
- Review/audit: `AuditEntry`, `ReviewAuditLog`, `ReviewSubmission`,
  `ReviewBundle`, `ScriptedReviewer`, `FixedDecisionReviewer`, `answer_digest`,
  `record_candidate_build`, `run_scripted_review`, `verify_review_bundle`,
  `assert_review_approved`.
- Versioning: `CandidateManifest`, `CandidateSeal`, `SealedProvenance`,
  `SealedCandidateArtifact`, `CandidateVerificationReport`,
  `create_candidate_version`, `verify_candidate_version`,
  `assert_candidate_version`, `sealed_provenance`, `sealed_source_keys`,
  `candidate_manifest_hash`, `candidate_seal_hash`, `load_candidate_bundle`.
- Drift: `DistributionSnapshot`, `HUMAN_REVIEW_GATE`, `snapshot_from_rows`,
  `golden_snapshot`, `incoming_snapshot`, `compare_snapshots`,
  `build_drift_report`, `validate_drift_report`, `write_drift_report`.
- CLI: `services.datasets.golden.maintenance.cli.main(argv)`.

## Fake seams and real-human seams

Fake seams (deterministic, offline, fixture-only):

- `tests/fixtures/turns.jsonl` is the only source of candidate/incoming rows;
  its hash is pinned and verified before use.
- `FixedDecisionReviewer` implements `ScriptedReviewer` and supplies two
  pseudonymous, non-human actor IDs with a fixed reason and answer hash.
- `voiceloop_common.utcnow()` is injectable via `frozen_clock`/`use_clock`, so
  audit timestamps and seals are deterministic.
- `ReviewAuditLog`, candidate output paths and drift report paths are
  injectable and are guarded away from the sealed artifact directory.
- `allowed_actor_ids` is a configurable pseudonymous allowlist; scripted reviewer
  IDs can never be proposal actors and verification pins reviewers from trusted
  settings rather than from candidate artifacts.

Real-human seams (deliberately **not** implemented here):

- No Label Studio/Argilla adapter, human listener, real reviewer identity,
  consent workflow, live traffic source or network call exists in this package.
  Real identity verification and approval of actor allowlists remain a lead seam;
  the IDs here are only configured pseudonyms, not authenticated identities.
- No API can edit `services/datasets/golden/artifacts/*`, amend a sealed label,
  promote a model, change a threshold or replace `golden.v1`.
- A production golden set would require a new lead-approved adapter behind a
  distinct service path with real reviewers, real consent, real storage and
  tamper-resistant sealing. This package's outputs must remain workflow
  stand-ins and must not be used as that evidence.

## context.md conflict note

No new conflict with `context.md` is introduced. Section 2 invariants 2, 5, 8
and 9 are preserved by keeping this package fixture-only, pseudonymous,
read-only toward the sealed set and descriptive-only for drift. Treating any
candidate version as a human-reviewed production golden set would conflict with
those invariants and must be escalated to the lead; this package explicitly
forbids that use.
