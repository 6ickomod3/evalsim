# M6 data-free implementation checkpoint

**Date:** 2026-07-29

**Status:** Portable implementation checkpoint; M6 is not complete or result-accepted

**Accepted pre-registration baseline:** commit
`a312254c9143987ed71c6a2a0bc44d637da7823a`

**Governing plan:** [M6 counterfactual closed-loop
reactivity](2026-07-29-m6-counterfactual-reactivity.md)

## Purpose

This checkpoint preserves enough tracked context to resume M6 from a fresh session or
another machine after pulling the repository. It records an implementation boundary,
not an M6 scientific result. Chat transcripts, agent identities, local paths, datasets,
run outputs, and private material are deliberately excluded.

Project-wide operating and safety rules are tracked in
[`AGENTS.md`](../../AGENTS.md). In particular, the accepted history-only/privileged
policy boundary and the prohibition on publishing datasets or generated experiment
artifacts will therefore travel with a clone or pull. `AGENTS.md` is the durable
project context; this file is the milestone-specific resume note.

## What this checkpoint contains

The data-free implementation foundation now includes:

- source-neutral counterfactual contracts, canonical intervention/plan identities,
  immutable paired snapshots, and typed paired metrics;
- explicit `HistoryOnlySimulatorPolicy` and `PrivilegedSimulatorPolicy`
  initialization capabilities, with constant velocity and EvalSim IDM on the
  history-only side and log replay on the privileged side;
- a typed ego-plan rollout path, synchronous world/ego transition ordering, execution
  traces, and preservation tests for the legacy no-plan path;
- the identity control and source-templated `b=2.0 m/s²` and `b=4.0 m/s²` brake plans,
  source-only qualification logic, paired measures, and deterministic fixed-cohort
  reweighting foundations;
- a deterministic synthetic analytic-oracle acceptance layer;
- a bounded 20-transition Waymax adapter and metric/evidence boundary behind EvalSim
  contracts; and
- M6-specific guarded result-schema, store, verification, terminal-capability, and
  sanitized-aggregate foundations.

The repaired synthetic acceptance layer and the bounded Waymax implementation have
completed independent adversarial review. That review acceptance applies to their
data-free implementation and trust boundaries only. It does not accept an official
runner, live Waymax determinism receipt, real-WOMD execution, or an M6 result claim.

The last confirmed full-suite result before the final result-store hardening was
**1,173 passed and 1 expected local-data skip**. It is historical checkpoint evidence,
not the final verification of the commit containing this note.

After the final fail-closed store changes, the focused result-store suite passed
**53/53**. Compilation and whitespace checks also passed.

**Final checkpoint full-suite rerun:** explicitly deferred after the last fail-closed
store changes. A fresh session must run the focused M6 tests and `uv run pytest` before
building the official verifier or opening WOMD.

## Exact incomplete boundary

The following remain deliberately deferred and must not be inferred from the
implemented foundations:

- the official repository/runtime/source verifier and public M6 command-line entry
  points;
- the accepted-M4 eligibility-only scan and its complete 128-member disposition
  ledger;
- the outcome-suppressed compute pilot;
- any M6 WOMD policy execution, eligibility count, per-scene value, aggregate outcome,
  or primary/secondary/Waymax result matrix;
- official live Waymax repeat/JIT determinism evidence;
- terminal success for eligibility-only, compute-pilot, or official modes;
- the sealed-result reconstruction and independent result/claim reviews;
- an M6 acceptance report, public result claim, claim-ledger update, presentation
  update, site deployment, or M6 completion declaration.

Non-data-free terminalization and caller-supplied live-determinism evidence fail closed
at this checkpoint. They must remain unavailable until the official verifier/runner
mints and independently validates their runtime-bound evidence. Do not weaken that
boundary merely to make a mocked or official path finalize.

No WOMD shard was opened while implementing or reviewing this checkpoint. No M6 WOMD
eligibility or outcome has been observed. Data and generated outputs remain ignored,
local-only artifacts and are not needed to resume the next implementation mini-step.

## Resume map

The main implementation and focused tests are:

| Area | Implementation | Focused tests |
|---|---|---|
| contracts and policy access | `evalsim/contracts/counterfactual.py`, `evalsim/contracts/simulator.py` | `tests/test_m6_counterfactual_contracts.py`, `tests/test_m6_policy_access.py` |
| typed rollout and trace | `evalsim/rollout/engine.py`, `evalsim/rollout/trace.py` | `tests/test_m6_rollout_engine.py`, `tests/test_m6_execution_trace.py` |
| plans, measures, statistics | `evalsim/perturb/m6.py`, `evalsim/metrics/m6.py`, `evalsim/stats/m6.py`, `evalsim/evaluation/m6.py` | `tests/test_m6_interventions.py`, `tests/test_m6_metrics.py`, `tests/test_m6_stats.py`, `tests/test_m6_evaluation.py` |
| synthetic oracle | `evalsim/evaluation/m6_synthetic.py` | `tests/test_m6_synthetic_acceptance.py` |
| bounded Waymax boundary | `evalsim/simulators/waymax_m6.py`, `evalsim/evaluation/m6_waymax_metrics.py` | `tests/test_m6_waymax.py`, `tests/test_m6_waymax_metrics.py` |
| local result foundation | `evalsim/results/m6.py` | `tests/test_m6_results.py` |

## Exact resume sequence

1. Pull the pushed checkpoint with a fast-forward-only update and confirm that the
   checkout is clean.
2. Read `AGENTS.md`, this checkpoint, and the accepted M6 pre-registration before
   changing code.
3. Recreate the core development environment from the lockfile and run the focused
   M6 tests, then the full repository suite. Do not open WOMD in this step.
4. Re-review the result-store hardening and repair any remaining P1/P2 finding. Preserve
   fail-closed non-data-free terminalization and live-determinism behavior.
5. Implement the official verifier and CLI/lifecycle tests as the next bounded
   mini-step. Obtain adversarial implementation acceptance and commit/push a clean
   implementation snapshot before any dataset access.
6. Only from that clean, pushed snapshot, install the optional pinned Waymo runtime and
   execute the pre-registered sequence: eligibility-only, outcome-suppressed compute
   pilot, then official run if every prior gate passes. Each attempt uses a fresh
   ignored run name.
7. Seal and independently review the local result before writing any sanitized result
   documentation. Documentation/site closure is a separate commit and release audit.

Use these commands to resume the data-free verification:

```bash
git pull --ff-only
git status --short
git log -1 --oneline

uv sync --frozen --extra dev
uv run pytest \
  tests/test_m6_counterfactual_contracts.py \
  tests/test_m6_policy_access.py \
  tests/test_m6_rollout_engine.py \
  tests/test_m6_execution_trace.py \
  tests/test_m6_interventions.py \
  tests/test_m6_metrics.py \
  tests/test_m6_stats.py \
  tests/test_m6_evaluation.py \
  tests/test_m6_synthetic_acceptance.py \
  tests/test_m6_waymax.py \
  tests/test_m6_waymax_metrics.py \
  tests/test_m6_results.py
uv run pytest
```

Do not invent an official command from this checkpoint: the official verifier and CLI
are intentionally absent. After those are implemented, use only the exact reviewed
commands added by that later commit. Optional Waymo dependencies should remain lazy
until then; when authorized by the accepted plan, install them reproducibly with:

```bash
uv sync --frozen --extra dev --extra waymo
```

Before every future commit or push, inspect both `git status --short` and
`git diff --cached --name-only`. Dataset, output, private, cache, and generated
experiment files must never enter the staged set.
