# M6 data-free implementation checkpoint

**Date:** 2026-07-30

**Status:** Implementation/security acceptance complete with no P0--P2 blockers; M6
remains in progress and has no accepted scientific result

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

The complete pre-data implementation now includes the capture-first official
bootstrap, repository/runtime/source verification, authority-bound evidence, the
eligibility and compute-pilot predecessors, the official runner, and the separate
review finalizer. Fresh independent adversarial security review found no P0--P2
blockers. That acceptance applies to the implementation and its trust boundaries only;
it does not accept a live Waymax determinism receipt, real-WOMD execution, a sealed
result, or an M6 scientific claim.

## Pre-data result-review lifecycle correction (2026-07-30)

This is an implementation clarification, not an amendment to or weakening of the
frozen pre-registration. Its fresh independent implementation/security review is now
complete with no P0--P2 blockers. No WOMD shard or generated result was opened while
implementing or reviewing this correction.

The earlier synthetic/Waymax implementation review is not an official M6 result
review. Mechanical precursor verification now records facts only and cannot issue an
acceptance decision. An official execution seals its complete post-outcome precursor,
issues a fresh challenge, writes `AWAITING_REVIEW`, and exits without review rows, an
execution summary, `COMMITTED`, or terminal success. Its CLI status uses
`m6-cli-status-2.0.0` and exposes only status, mode, profile, and the allowlisted local
result path; evidence and mechanical hashes remain local.

A separate tracked, noninteractive `finalize-review` invocation supplies exactly one
decision and P1/P2/P3 count for each fixed role: `architecture`,
`methods_statistics`, and `privacy_claim`. The finalizer re-authenticates source and
runtime, re-verifies accepted M4 and all ten shard identities, reopens both terminal
predecessors, checks fresh typed provenance against the sealed precursor, and binds the
decisions to the stored post-precursor challenge. It never calls an eligibility or
outcome executor. Every role must explicitly `accept` with P1=P2=0 to commit or reach
terminal success. Any explicit `reject`, P1, or P2 closes the run as
`TERMINAL_FAILURE` with reason `review_rejected`. A contradictory supplied
`accept` with nonzero P1/P2 is preserved exactly as reviewer evidence and is still
release-blocking; it is never normalized away or made retryable. Counts are bounded by
the persisted signed-int32 domain (`0..2**31-1`). P3 is recorded but nonblocking; any
P3 rationale remains in separate local review notes and is not ingested into the fixed
result artifact or public output. A review-rejected failure marker authenticates the
review request, awaiting marker, decision rows, execution summary, precursor digest,
and mechanical-verification digest; a dedicated verifier reopens that rejected record.

The bootstrap registers a one-shot store invalidator before dispatch. If inner status
delivery, outer capture validation, or the final saved-descriptor write fails, any
created store becomes terminally failed before the command returns. If
`TERMINAL_SUCCESS` was already written irreversibly, the emergency failure marker is
added deliberately so marker-exclusivity verification permanently rejects the
ambiguous store.

The data-free synthetic path creates no independent-review identities: it writes zero
review rows and is permanently `nonpromotable`.

The implementation review gate is complete. Before any M6 data access, complete the
exact staged-path and source release audit, push `main`, and obtain independent approval
of that exact pushed commit. Represent that independent approval with a **lightweight
(not annotated)** `m6-approved-v1` tag exactly at pushed `main`, push the tag, and
verify that remote `main` and the remote tag resolve to the same commit. The tag records
the independent approval; creating or moving it is not self-approval. Only after those
release checks pass, create the official immutable Python 3.11.5 environment without
the `dev` extra:

```bash
uv sync --frozen --extra waymo --python 3.11.5
.venv/bin/python --version
```

The version check must report Python 3.11.5. Each capture-first invocation below
rechecks the complete environment catalog and must fail closed before data access on
any mismatch. Run the lifecycle from the repository root in this exact order, beginning
with the eligibility-only predecessor:

```bash
EVALSIM_RUN_WAYMO_LOCAL=1 .venv/bin/python -I -S -B \
  evalsim/cli/m6_bootstrap.py \
  --project-root "$PWD" \
  --data-dir data/raw/womd/v1.3.1/tf_example/validation \
  --m4-run-dir outputs/m4/<accepted-m4-run> \
  --run-name <eligibility-run> \
  --mode eligibility_only
```

Only after that predecessor reaches verified terminal success, run the
outcome-suppressed compute pilot:

```bash
EVALSIM_RUN_WAYMO_LOCAL=1 .venv/bin/python -I -S -B \
  evalsim/cli/m6_bootstrap.py \
  --project-root "$PWD" \
  --data-dir data/raw/womd/v1.3.1/tf_example/validation \
  --m4-run-dir outputs/m4/<accepted-m4-run> \
  --run-name <compute-pilot-run> \
  --eligibility-run-name <eligibility-run> \
  --mode compute_pilot
```

Only after the compute pilot reaches verified terminal success and all frozen gates
pass, run the official outcome execution:

```bash
EVALSIM_RUN_WAYMO_LOCAL=1 .venv/bin/python -I -S -B \
  evalsim/cli/m6_bootstrap.py \
  --project-root "$PWD" \
  --data-dir data/raw/womd/v1.3.1/tf_example/validation \
  --m4-run-dir outputs/m4/<accepted-m4-run> \
  --run-name <official-run> \
  --eligibility-run-name <eligibility-run> \
  --pilot-run-name <compute-pilot-run> \
  --mode official
```

That invocation must stop at `AWAITING_REVIEW`; it cannot self-accept. Only after
independent reviews of the sealed precursor by the `architecture`,
`methods_statistics`, and `privacy_claim` roles, run the tracked finalizer with all
twelve explicit decision/count fields:

```bash
EVALSIM_RUN_WAYMO_LOCAL=1 .venv/bin/python -I -S -B \
  evalsim/cli/m6_bootstrap.py \
  --project-root "$PWD" \
  --data-dir data/raw/womd/v1.3.1/tf_example/validation \
  --m4-run-dir outputs/m4/<accepted-m4-run> \
  --run-name <official-run> \
  --eligibility-run-name <eligibility-run> \
  --pilot-run-name <compute-pilot-run> \
  --mode official \
  --action finalize-review \
  --architecture-decision <accept-or-reject> \
  --architecture-p1-count <count> \
  --architecture-p2-count <count> \
  --architecture-p3-count <count> \
  --methods-statistics-decision <accept-or-reject> \
  --methods-statistics-p1-count <count> \
  --methods-statistics-p2-count <count> \
  --methods-statistics-p3-count <count> \
  --privacy-claim-decision <accept-or-reject> \
  --privacy-claim-p1-count <count> \
  --privacy-claim-p2-count <count> \
  --privacy-claim-p3-count <count>
```

The last confirmed full-suite result before the final result-store hardening was
**1,173 passed and 1 expected local-data skip**. It is historical checkpoint evidence,
not the final verification of the commit containing this note.

After the final fail-closed store changes, the focused result-store suite passed
**53/53**. Compilation and whitespace checks also passed.

**Current verification (2026-07-30):** a single green `uv run pytest` is still not
recorded. The full suite is impractically slow (hours) because M6 scene validation
re-runs the deliberate snapshot tamper-detection re-hash on every metric recompute
(~48 `CounterfactualPair.revalidate()` calls per scene, measured), and roughly fifteen
tests each rebuild a complete official-evidence bundle. Every M6 test still passes
individually. On the exact worktree of this note a representative subset was rerun green:
the non-M6 and M5 suite (695 passed, 24 skipped), the fast M6 core suite (264 passed, 2
skipped: contracts, policy access, rollout, trace, interventions, metrics, statistics,
evaluation, synthetic acceptance, and both Waymax-metric boundaries), and an end-to-end
official-runner flow smoke (5 passed, sharing one module-scoped `run_m6_official_numpy`
build). Two follow-ups remain before source release or any WOMD access: (1) reduce the
redundant in-scope revalidation without weakening the per-entry tamper check, then record
one full-suite green run; and (2) add explicit `finalize-review` precursor-drift and
review-import environment-recheck coverage (the production gates run, but the sole
finalize test currently stubs them out).

## Exact incomplete boundary

The following remain deliberately deferred and must not be inferred from the
implemented foundations:

- release of the exact reviewed source snapshot and its final immutable environment
  catalog recheck;
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

The official terminalization and runtime-bound evidence paths are implemented,
reviewed, and fail closed, but remain unexercised against WOMD. Only the reviewed
official verifier/runner may mint and validate that runtime evidence. Do not weaken
this boundary or substitute caller-supplied evidence merely to make a mocked or
official path finalize.

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
| official source/runtime and lifecycle | `evalsim/cli/m6_bootstrap.py`, `evalsim/cli/m6_official.py`, `evalsim/evaluation/m6_official.py` | `tests/test_m6_official_cli.py`, `tests/test_m6_official_runner.py` |
| pilot and official Waymax execution | `evalsim/evaluation/m6_pilot.py`, `evalsim/evaluation/m6_waymax_official.py` | `tests/test_m6_pilot.py`, `tests/test_m6_waymax_official.py` |
| local result foundation | `evalsim/results/m6.py` | `tests/test_m6_results.py` |

## Exact resume sequence

1. Pull the pushed checkpoint with a fast-forward-only update and confirm that the
   checkout is clean.
2. Read `AGENTS.md`, this checkpoint, and the accepted M6 pre-registration before
   changing code.
3. Recreate the Python 3.11.5 development environment from the lockfile and run the
   focused M6 tests, then the full repository suite. Do not open WOMD in this step.
4. Confirm the completed independent security review still applies to the exact
   implementation and that it has no P0--P2 blockers. Any substantive implementation
   change requires proportionate re-verification and review before data access.
5. Complete the exact staged-path and source release audit, commit, and push `main`.
   After independent approval of that exact pushed commit, create a lightweight (not
   annotated) `m6-approved-v1` tag exactly at pushed `main`, push the tag, and verify
   that remote `main` and the remote tag resolve to the same commit. The tag represents
   the independent approval; it does not create or replace that approval. Then recreate
   the no-`dev` official environment and require its complete environment-catalog
   recheck to pass.
6. Only from that clean, pushed snapshot, execute the pre-registered sequence:
   eligibility-only, outcome-suppressed compute pilot, then official run if every prior
   gate passes. Each attempt uses a fresh ignored run name. The official run stops at
   `AWAITING_REVIEW`.
7. Independently review the sealed precursor across all three fixed roles, then invoke
   `finalize-review`. Only a verified terminal success may support sanitized result
   documentation; documentation/site closure is a separate commit and release audit.

Use these commands to resume the data-free verification:

```bash
git pull --ff-only
git status --short
git log -1 --oneline

uv sync --frozen --extra dev --python 3.11.5
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
  tests/test_m6_official_cli.py \
  tests/test_m6_official_runner.py \
  tests/test_m6_pilot.py \
  tests/test_m6_synthetic_cli.py \
  tests/test_m6_waymax_official.py \
  tests/test_m6_results.py
uv run pytest
```

The implementation/security review gate is complete. Do not run the official or
`finalize-review` commands until the exact reviewed source snapshot is clean, pushed,
and approved and the required predecessors are ready. Use only the tracked command
surfaces above. A local integration-test environment may include both extras:

```bash
uv sync --frozen --extra dev --extra waymo --python 3.11.5
```

Immediately before official execution, remove the `dev` extra by recreating the exact
immutable runtime. The bootstrap must recheck the complete catalog before data access:

```bash
uv sync --frozen --extra waymo --python 3.11.5
```

Before every future commit or push, inspect both `git status --short` and
`git diff --cached --name-only`. Dataset, output, private, cache, and generated
experiment files must never enter the staged set.
