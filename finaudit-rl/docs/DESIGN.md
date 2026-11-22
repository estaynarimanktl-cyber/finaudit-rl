# FinAudit RL — Design Note

This document describes the environment, the data model, the matching and
reward contract, the baselines, and the verification approach of the
submission. It complements `README.md`.

## 1. Purpose

`finaudit` is a reproducible RL environment for financial-audit agents.  The
model plays an auditor: it reads synthetic company records, runs safe
calculations, and submits findings about hidden errors.  The environment is
the evaluator and grader; there is no training loop inside it.

The two major hard requirements are:

1. **Trust in the environment.**  The same seed and config must produce the
   same episode, trajectory, and reward.  The RNG is a SHA-256
   counter-mode RNG (`finaudit/rng.py`), so generation is independent of
   Python-version internals and machine state.
2. **Trust in the grader.**  Hidden truth is private.  `truth.json` and
   `provenance.json` are written only to `private/`; `PolicyEnv` exposes a
   reset/step-only interface and cannot reach them.  Grading is performed by
   `AuditEnv.grade()` (evaluator-only) or the `finaudit` evaluator commands.

## 2. Episode data model

One episode = one fictional company + one reporting period.

### Open files (visible to the policy)

| File | Contents |
|---|---|
| `manifest.json` | `episode_id`, `base_currency`, period, timezone, row/document counts, limits, `seed_commitment`, `config` (public part only). |
| `policy.json` | FX date rule, fallback, recognition rule, canonical affected-ID rule, rounding, account map. |
| `ledger.csv` | `row_id, timestamp, account, debit_minor, credit_minor, currency, entity_id, invoice_id, payment_id`. |
| `invoices.jsonl` | `doc_id, invoice_id, entity_id, issued_at, due_at, currency, total_minor, line_items`. |
| `payments.jsonl` | `doc_id, payment_id, entity_id, timestamp, currency, amount_minor, invoice_refs, kind`. |
| `fx.csv` | `date, source_currency, target_currency, rate`. |

The ledger is a balanced double-entry extract in the base currency.  Invoices
and payments may be foreign; their legs are translated by `fx.csv` and
`policy.json`.

### Private files (evaluator only)

| File | Contents |
|---|---|
| `truth.json` | `episode_id`, issues with `issue_id, type, subtype, affected_ids, expected_amount_minor, currency, evidence_sets, description, causal_change`. |
| `provenance.json` | Generation history, seed, config, company, baseline counters, modifications. |

`AuditEnv.reset()` loads the episode with private files by default;
`PolicyEnv` receives only the public observation.

## 3. Error taxonomy

Four independent error types are supported by the generator, validator and
matcher:

| Type | Canonical affected IDs | Semantics |
|---|---|---|
| `duplicate_account` | the second balanced journal entry: every duplicated ledger row (both copies). | Same economic entry posted twice; the second entry duplicates the first. |
| `payment_allocation` | payment doc, the affected invoice doc, suspense/source rows. | Payment missing, partial, or applied to the wrong invoice. |
| `fx_conversion` | source doc + cash/AP/AR rows with the wrong base translation. | Entry uses the wrong FX date/rate. |
| `period_cutoff` | source doc + every row of the cross-period entry. | Entry is posted in a different reporting period than its source document. |

"Canonical affected IDs" means records whose own values are wrong, not every
record needed as evidence.  Evidence can be a superset; the truth requires at
least one complete allowed set.

### Interacting scenarios

Two optional interacting scenarios are planned:

1. `payment_fx_boundary` — a partial payment in another currency posted at a
   period boundary.  This creates an allocation, an FX error and a
   period-cutoff candidate in one economic story.
2. `duplicate_fx` — a duplicated payment while an invoice accrual is mistranslated.

They are used sparingly (`interacting: true`) so the auditor must reason about
combined evidence, not just one cue.

## 4. Clean baseline and distractors

`finaudit/baseline.py` builds a coherent hidden company story:

* Balanced double-entry ledger in the base currency.
* Invoices and payments may be foreign; FX tables are daily random walks from
  60 days before the period to 60 days after.
* Legitimate distractors: split payments, partial payments, overpayments,
  base-currency refunds, margin invoices, recurring bank fees/payroll, opening
  balances.
* The final ledger is shuffled after ID assignment, so file order is
  independent of canonical IDs.

Errors are injected by `finaudit/inject.py` with a balance-preserving mutation
and a causal `provenance` record.  Every mutation is validated independently
by `finaudit/validate.py`.

## 5. Environment contract

`AuditEnv` implements the Gymnasium-style contract:

```python
obs, info = env.reset(seed=None, config=None)
obs, reward, terminated, truncated, info = env.step(action)
```

### Actions

```python
{"type": "list_records", "table": "ledger|invoices|payments|fx|policy",
 "cursor": null, "limit": 50}
{"type": "get_record", "table": "ledger|invoices|payments|fx|policy", "id": "..."}
{"type": "get_document", "doc_id": "..."}
{"type": "calculator", "expression": "125000 * 0.0375"}
{"type": "submit", "findings": {...}}
```

* Pages are capped at 50 rows; cursors are HMAC-protected by episode ID.
* Business `invoice_id`/`payment_id` aliases are normalized to canonical IDs.
* The calculator accepts only numeric literals with `+ - * /` and parentheses
  (safe AST evaluation; no `eval`, no attributes, no indexing, no functions).
* Unknown IDs/actions consume a step and return a limited error.
* The environment never exposes `truth.json`, `provenance.json`, the seed,
  the filesystem, SQL, shell or the internet.

### Observations

The initial observation includes the task text, manifest, policy, action and
submission schemas, and budget.  `PolicyEnv` exposes the same reset/step
contract through `PolicyEnv(env).reset()/step()`; it raises
`AttributeError` for truth, grading and seed access.

### Limits

* `max_steps` defaults to 100, including the submit step.
* Time limit is a safety valve; deterministic replay uses the step limit.

## 6. Reward contract

```text
P = TP/(TP+FP)          if TP+FP > 0 else 0
R = TP/(TP+FN)          if TP+FN > 0 else 0
F1 = 2TP/(2TP+FP+FN)    if 2TP+FP+FN > 0 else 1
reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)
```

Rules:

* Empty truth + empty findings → F1 = 1 (reward 1.0).
* Empty truth + non-empty findings → F1 = 0.
* Invalid submission structure → reward `-1`.
* One-to-one maximum-cardinality matching; Kuhn's algorithm with
  deterministic finding/issue order.
* TP requires:
  * same type,
  * same currency (base currency),
  * exact canonical affected-ID set,
  * claimed amount within tolerance `max(2, round(expected * 5 / 10000))`
    minor units,
  * at least one complete allowed evidence set; extra existing evidence is
    allowed, unknown evidence is not,
  * all affected IDs exist.
* Duplicates are false positives; unmatched issues are FNs.
* Step economy: `0.002` per inspection step.  With an oracle answer, the
  reward falls by 0.02/0.10 over 10/50 inspection steps.

## 7. Matching (public)

Matching is one-to-one and maximum-cardinality.  Findings are considered in
submission order; candidate issues are in ascending `issue_id` order.  Ties
resolve to the first free issue.  `finaudit/matcher.py` is the public
specification of normalization, structure validation, and metrics.

## 8. Baselines

| Policy | Type |
|---|---|
| `random` | Explore all tables, submit 0–3 arbitrary structurally valid findings. |
| `reconciliation` | Deterministic rule-based auditor over public records: duplicate clusters, period mismatches, allocation breaks, FX translation mismatches.  Uses no truth. |
| `mark_all` | Flags every repeated-looking row, every foreign doc, every payment. |
| `silence` | Submits no findings.  Reference boundary: 1.0 on clean, -1.0 on erroneous. |
| `oracle` | Evaluator-only upper bound built from hidden truth. |

The reconciliation policy is intentionally a *solution-style* baseline, not a
vendor product: it uses only public data and the published policy, so it is
validatable and reproducible.  Its scores on `episodes/dev` are typically
F1 = 1.0, FP = 0, while mark-all and silence are negative and random/silence
are clean boundaries.

## 9. Evaluation protocol

`finaudit evaluate --episodes episodes/dev --out trajectories/trajectories.jsonl`
runs `random`, `reconciliation`, `mark_all`, `silence` and `oracle` on every dev
episode in eval mode.  The evaluator uses `AuditEnv.grade()` (reward and
diagnostics are hidden from the policy in eval mode).  The JSONL contains
`episode_id, step, action, observation, reward, terminated, truncated` per step
and per policy; the summary contains mean/min/max reward, mean steps, TP/FP/FN.

## 10. Validation

`finaudit/validate.py` is an independent validator used by the test-suite and
`finaudit validate`.  It checks schema, balances, references, line-item
arithmetic, manifest/policy consistency, truth IDs/evidence, and per-type
conditions:

* `duplicate_account` — every affected row has an exact twin; amount is
  half the duplicated debit total.
* `payment_allocation` — suspense/source/wrong-invoice invariants.
* `fx_conversion` — the correct FX row is in evidence.
* `period_cutoff` — the document period differs from row periods; amount is
  the entry debit total.

## 11. Reproducibility

* `finaudit/splits.py` defines 10 train + 10 dev named specs.
* `finaudit generate --split all` writes episodes to `episodes/train/*` and
  `episodes/dev/*` with `open/` and `private/` trees; `episodes/index.json`
  records episode IDs, configs and SHA-256 fingerprints.
* `finaudit/transform.py::remap_episode` produces isomorphic episodes with
  shuffled file order and fresh IDs; used by paired invariance tests.
* `finaudit demo` prints one episode and baseline comparison.
* `finaudit check` is the single verification command.

## 12. Testing

The `tests/` suite is unittest-based and has no external dependencies:

* `test_env.py` — environment contract, pagination, invalid actions, step and
  time limits, eval hiding, `PolicyEnv` isolation.
* `test_matcher.py` — reward, matching, tolerance, aliases, invalid structure,
  step economy, duplicates, mark-all.
* `test_policies.py` — baseline behavior, per-type detection, determinism,
  attack invariants.
* `test_attacks.py` — the required attack behavior: correct answer beats
  silence on erroneous books; silence beats false accusations on clean books;
  duplicates give no benefit; mark-all is bad; step economy is slight; unknown
  IDs/evidence are not TP; invalid structures are `-1`.
* `test_generation.py`, `test_validation.py`, `test_transform.py`,
  `test_trajectory.py`, `test_cli.py` — reproducibility, validation, ID
  remap, trajectories, CLI.

## 13. Safety

* No network, SQL, shell, filesystem or training APIs are available to the
  policy.
* The environment never exposes truth or seed to the policy.
* Evaluator mode hides reward and diagnostics from the policy; grading remains
  with `AuditEnv.grade()`.
* Everything is offline and dependency-free.

## 14. Known limitations

* Reconciliation is a rule-based auditor: it has no internal memory and relies
  on the same public evidence an auditor could read.
* The generator produces at most 12 issues per episode; allowed error types and
  interacting scenarios are fixed by design.
* Matching is deterministic, not probabilistic; agent stochasticity is the
  agent's, not the environment's.
* The environment does not require the policy to use tools efficiently; a
  policy may read the entire dataset in ~33–70 steps.  The reward mildly
  penalizes inspection steps; the environment does not forbid reading all data.
