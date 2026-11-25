# Failure scoring (P2-01)

Offline, deterministic failure scoring for the triage lane. It consumes
**schema-valid turn-event dictionaries** (`turn_event` 1.0.0), never touches a
live call, and returns a score in `[0, 1]` plus an explanation naming every
signal that fired. All integrations are protocol seams with in-memory fakes;
there are no ML, network, GPU or database dependencies.

Fixture failure labels are used **only by tests and the CLI summary** to check
ranking. They are never passed to the scorer: no `expected_transcript`,
`expected_slots` or `failure_type` reaches a signal extractor.

## Layout

| File | Responsibility |
|---|---|
| `settings.py` | Typed `VOICELOOP_TRIAGE_*` thresholds and weights. |
| `signals.py` | Word confidence, decoder stats, schema validity, parse probability, teacher protocol/fake. |
| `repair.py` | Guest correction, agent re-ask, transfer; `prior_turns_ref` JSONL loader. |
| `mismatch.py` | Final-record mismatch (highest weight) and hypothesis-vs-parser slot mismatch. |
| `explanation.py` | `FiredSignal`, `ScoreExplanation`, `ExplanationStore`, in-memory fake. |
| `scoring.py` | `ScoringEngine`, public helpers, fixture loader, CLI. |
| `__init__.py` | Public API re-exports. |

No `services/triage/queue/`, ingest, evalgate, SDK, schema or fixture files are
owned by this task.

## Run and test

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m services.triage.scoring.scoring
.\.venv\Scripts\python.exe -m services.triage.scoring.scoring --fixtures tests/fixtures/turns.jsonl --output reports/triage/scores.json
.\.venv\Scripts\python.exe -m pytest tests/unit/triage_scoring -q
.\.venv\Scripts\python.exe -m ruff check services/triage/scoring tests/unit/triage_scoring
.\.venv\Scripts\python.exe -m ruff format --check services/triage/scoring tests/unit/triage_scoring
.\.venv\Scripts\python.exe -m mypy services/triage/scoring tests/unit/triage_scoring
```

The CLI reads the read-only `tests/fixtures/turns.jsonl` manifest to locate
events, scores all 36 turns and prints (or writes) one explanation per turn plus
mean scores by `failure_type`. It never edits fixtures. The `-m` invocation can
emit a harmless `RuntimeWarning` about the package already importing its module;
the result is unaffected.

## Public API

```python
from services.triage.scoring import (
    InMemoryExplanationStore,
    ScoringEngine,
    TriageScoringSettings,
    score_turn,
)

settings = TriageScoringSettings()  # VOICELOOP_TRIAGE_* overrides defaults
store = InMemoryExplanationStore()
engine = ScoringEngine(settings=settings, store=store)
explanation = engine.score(event)  # event: turn_event 1.0.0 dict

# One-shot convenience (fresh engine, same defaults):
explanation = score_turn(event, settings=settings, store=store)
record = explanation.to_dict()
stored = store.get(explanation.event_id)
```

`load_fixture_scores(path, ...)` returns `FixtureScore(label, explanation)` rows
for the synthetic manifest; it uses labels only to find event files and to
annotate the returned rows.

`ExplanationStore` is a three-method protocol (`put`, `get`, `all`). The default
`InMemoryExplanationStore` upserts on `event_id`, which makes re-scoring
idempotent. A real store seam should upsert on `event_id`, index by
`(call_id, turn_index)`, and follow the same retention policy as turn metadata.
No audio bytes are stored here; explanations contain only signal names, weights,
severities, contributions and short human-readable reasons.

## Signals and thresholds

| Signal name | Fires when | Default weight |
|---|---|---:|
| `final_record_mismatch` | `parser.output` differs from `outcome.final_record` (nested fields compared as dotted paths) | **1.00 (highest)** |
| `hypothesis_guest_count_mismatch` | STT says "N guests" but `parser.output.guests` differs | 0.50 |
| `hypothesis_nights_mismatch` | STT says "N nights" but check-out minus check-in differs | 0.50 |
| `low_word_confidence` | min word `conf < 0.60` or mean `< 0.75`; empty hypothesis with no word confidences also fires | 0.35 |
| `decoder_stats` | `no_speech_prob > 0.50`, `avg_logprob < -0.60` or `compression_ratio > 2.50` | 0.30 |
| `schema_invalid` | `parser.schema_valid is false` on a turn with parser output | 0.25 |
| `low_parse_probability` | `parser.logprob < -1.00` on a turn with parser output | 0.20 |
| `guest_correction`, `agent_reask`, `transfer` | explicit correction phrase, prior agent re-ask, or `outcome.status == "transferred"` | 0.15 each |
| `teacher_disagreement` | injected teacher disagrees **and** `VOICELOOP_TRIAGE_TEACHER_ENABLED=true` | 0.30 |

Every threshold and weight is an environment variable with prefix
`VOICELOOP_TRIAGE_`, e.g. `VOICELOOP_TRIAGE_MIN_WORD_CONF=0.5` or
`VOICELOOP_TRIAGE_WEIGHT_FINAL_RECORD_MISMATCH=1.5`.

Severity ramps linearly from 0 at the threshold to 1 one configured span beyond
it; binary signals use severity 1. The raw score is the sum of
`weight × severity`, mapped to `[0, 1)` by `1 - exp(-raw_score)` and rounded to
six decimals. Long or repeated signals therefore saturate rather than overflow.

### Why hypothesis-vs-parser mismatch exists

Fixture parser failures (`date_confusion`, `nights_misparsed`) have high word
confidence, healthy decoder stats and a schema-valid parse, and their outcomes
have not arrived, so the final-record signal has nothing to compare. The
hypothesis-vs-parser check compares the event's own STT numbers with its parser
slots. It uses no labels and never outranks the final-record signal.

### No-slot exemption

The synthetic greeting turns carry `parser.schema_valid: false`, `logprob: -1.85`
and `parser.output: {}` by design (there is nothing to parse). An empty parser
output is therefore exempt from the schema-invalid and parse-probability
signals. Exact-silence turns are still caught by decoder/confidence signals.

### Dialogue context and teacher seams

`load_prior_turns(event, root=...)` reads `dialogue_context.prior_turns_ref`
(JSON Lines) if a caller supplies a base path; `ScoringEngine` accepts a
`prior_turn_loader` callable, so production can supply its own resolver without
file IO in the core. The fixture CLI passes the repo root. Agent re-ask and
transfer have **no fixture evidence**: their detectors are covered by synthetic
unit-test events only.

`TeacherModel.judge(event) -> TeacherVerdict | None` is injectable. The default
`FakeTeacherModel` always agrees and the signal is disabled unless explicitly
enabled; P4-04 can plug in a batch offline teacher without changing this package.

## Contract and verification

Input: a dict conforming to `schemas/turn_event.schema.json` 1.0.0
(`event_id`, `call_id`, `turn_index`, `audio`, `stt` required; all other blocks
optional). Output: `ScoreExplanation` with `event_id`, `call_id`, `turn_index`,
`score`, and `signals[{name, weight, severity, contribution, reason}]`;
`to_dict()`/`from_dict()` round-trip through JSON. No schema file changed.

On the 36 read-only fixtures, turns labelled `stt_wrong`, `parser_wrong` or
`unusable_audio` score above `clean`/`both_fine` turns by mean score, and every
known failure has at least one named signal. No fixture final record disagrees
with its parser output, so the highest-weight signal is tested with an explicitly
derived mutation rather than faking a fixture label.

## Known gaps

- The WAV fixtures are tone/noise/silence carriers; scoring reads recorded
  metadata, not audio. Real ASR/teacher adapters belong to later tasks.
- Empty parser output that should have had slots cannot be distinguished from a
  legitimate no-slot parse from the event alone; a late final-record mismatch or
  repair behaviour is needed to flag it.
- No agent-side turns exist in the fixture context JSONLs, so agent re-ask
  detection is unit-tested with synthetic records only.
- Random-slice sampling, clustering, daily caps and blind items are P2-02;
  review labels and dataset expansion come later. Triage scores are not labels
  and must never substitute for the human review required by context.md §2.2.
