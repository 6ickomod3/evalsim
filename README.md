# EvalSim — Closed-Loop Simulation-Evaluation Platform

A reproducible platform for evaluating autonomous-driving traffic simulators along
independent dimensions — kinematic realism, agent interaction, map adherence, behavioral
diversity, closed-loop reactivity, and physical feasibility — rather than collapsing
simulation quality into a single leaderboard number.

**Hosted presentation:** [Explore the EvalSim policy lab](https://evalsim-policy-lab.jdai2013.chatgpt.site/)

> **The contribution is the evaluation methodology, not the simulator.** We use simple
> baselines (log-replay, constant-velocity, IDM) with deliberately different failure modes
> and invest the engineering in the system *around* them.

**New here? Read in this order:** this README → the interactive technical presentation
[`index.html`](index.html) → the design doc
[`docs/plans/2026-07-27-evalsim-design.md`](docs/plans/2026-07-27-evalsim-design.md) →
the canonical
[Waymo-aligned roadmap](docs/plans/2026-07-28-waymo-aligned-roadmap.md).

> 📌 **This README is kept up to date as work progresses.** The
> [Progress](#progress) table and [Completed work](#completed-work) changelog below always
> reflect the current state of the repo.

## Design in one paragraph

The platform is built **contract-first** (ports-and-adapters). The validated `Scenario`
contract has both a synthetic producer and an M3 local WOMD producer through pinned
Waymax/JAX on Apple Silicon CPU. M4 makes Waymax a recurring reference rollout and
semantic-parity path. Downstream components continue to consume **only** project
contracts. Synthetic scenes remain analytic oracles while every core feature from M3
onward also gets a real-WOMD acceptance path.

## Progress

Legend: ✅ done · 🚧 in progress · ⬜ not started · 🖥️ Mac/CPU · ☁️ optional cloud/accelerator

| # | Milestone | Status | Where |
|---|-----------|:------:|:-----:|
| M0 | Repo skeleton + data contracts | ✅ | 🖥️ |
| M1 | Synthetic scenario source + visualization | ✅ | 🖥️ |
| M2 | Simulator adapters + rollout engine | ✅ | 🖥️ |
| M3 | Local WOMD → Waymax/JAX → EvalSim vertical slice | ✅ | 🖥️ |
| M4 | Deterministic WOMD cohort + Waymax parity | ✅ | 🖥️ |
| M5 | Real-WOMD metrics + statistical scorecards | ⬜ | 🖥️ |
| M6 | Counterfactual closed-loop reactivity | ⬜ | 🖥️ |
| M7 | Evaluator red-team + metric governance | ⬜ | 🖥️ |
| M8 | JAX/Flax learned realism discriminator | ⬜ | 🖥️/☁️ |
| M9 | Semantic-video and real camera/VLM bridge | ⬜ | 🖥️/☁️ |
| M10 | Scalable, resumable evaluation pipeline | ⬜ | 🖥️/☁️ |
| M11 | Decision package + staff-caliber communication | ⬜ | 🖥️ |

**Tests:** 493 passing + 1 local-data skip with the Waymo extra; 418 passing + 22
optional-runtime skips core-only; focused M4 CLI 110 passing; locked M4 matrix 421
passing + 1 local-data skip · **Last updated:** 2026-07-28 (M4 implemented, accepted,
released, and post-release verified)

## Completed work

### M0 — Package skeleton + data contracts ✅
The validated seams every later layer depends on:

- **`Scenario` / `Agent` / `MapPolyline`** (Seam A) — substrate-agnostic scene
  representation with shape/range validation.
- **`Rollout`** — a simulator's output plus provenance (sim name/version, seed,
  perturbation) for reproducibility and caching.
- **`SimulatorPolicy`** + **`PolicyMetadata`** (Seam B) — the interface every simulator
  implements (`initialize` / `step` / `metadata`).
- **`Metric` / `MetricSpec` / `MetricResult`** — the pluggable-metric contract; component
  values and distributions stay visible (no single composite score).
- **`RunManifest`** — complete run description with JSON round-trip.
- **Lossless Parquet serialization** — one self-contained `.parquet` per scenario/rollout
  (agents as list-columns; scalars/map/metadata in file-level metadata).
- **Skeleton + tooling** — 10 placeholder subpackages, `uv`-managed venv, `pyproject.toml`.
- **13 tests**: contract round-trip, invariant validation, abstract-interface enforcement,
  manifest JSON.

### M1 — Synthetic scenario source + visualization ✅

Local, deterministic scenarios with the same contract that the WOMD adapter also
produces:

- **Five parametric families** — following, four-way intersection, merge, left turn,
  and pedestrian crossing; a 50-scenario set contains 10 of each.
- **Reproducible generation** — independent per-scenario random streams and
  configuration-fingerprinted IDs prevent call-order dependence and cache collisions.
- **Real-shaped contract data** — monotonic timestamps, finite trajectories, tangent
  velocities/headings, collision-free reference trajectories, meaningful lifecycle
  masks, directed lane/road-edge geometry, a valid ego, and canonical tags for later
  slicing.
- **Immutable `ScenarioManifest`** — ordered unique scenario IDs in deterministic JSON;
  existing manifests are never overwritten.
- **`plot_scenario()`** — map-aware Matplotlib rendering with valid-mask gaps, distinct
  agent types, and an emphasized ego trajectory.
- **55 M1 tests** — 50-scenario contract and Parquet acceptance, determinism,
  collision/map/kinematic invariants, family semantics, manifest provenance and
  immutability, non-aliased agent buffers, and headless PNG renders for all five
  families.

### M2 — Simulator adapters + rollout engine ✅

Three deliberately limited baselines now run through one typed, reusable NumPy engine:

- **`LogReplayPolicy`** — exact recorded world-agent replay; deliberately nonreactive.
- **`ConstantVelocityPolicy`** — causal world-frame extrapolation with no interaction or
  map following.
- **`IDMPolicy`** — canonical bumper-gap longitudinal interaction for vehicles, with
  explicit constant-velocity fallback for pedestrians and cyclists.
- **Typed Seam B** — immutable `AgentFrame`, `PolicyObservation`, and `PolicyStep`
  objects distinguish engine-integrated controls from exact absolute-state overrides.
- **Closed-loop lifecycle** — actual per-step `dt`, logged history through
  `metadata.current_index`, synchronous world-agent updates, observed-state
  births/re-entries, preserved masks and identity, and an exogenous logged ego ready for
  M6 perturbations.
- **Audited point-mass dynamics** — acceleration, braking, speed, and yaw-rate clamps;
  physically timed stops/speed caps; no reverse motion; clamp counts and complete
  engine/policy config in rollout provenance.
- **66 M2 tests** — analytic CV/IDM/dynamics oracles, future-log poisoning, lifecycle and
  determinism checks, malformed-policy rejection, 50 scenarios × 3 policies, and exact
  Parquet round-trips.

Experimental scope is intentionally honest: with logged ego control, the current
synthetic set gives simulated world-agent curvature mainly in the merge family; the turn
family's 90° actor is the logged ego. M2 proves CV-vs-replay separation on merge and
IDM-vs-CV separation on following, but broader nonlinear world-agent coverage should be
added (with a synthetic source-version bump) before M5 scorecards claim five-family
differentiation.

### M3 — Local WOMD / Waymax / JAX vertical slice ✅

One real, ignored WOMD v1.3.1 validation record now traverses the full local source
boundary on Apple Silicon CPU:

- **Pinned optional runtime** — NumPy 1.26.4, JAX/jaxlib 0.4.38, TensorFlow 2.18.1,
  Flax 0.10.4, and immutable Waymax commit
  `a64dfec9be8576b60d9cecc94f406d9812d4a7d0`.
- **Exact local reader** — `WaymaxSource` resolves only shard `00000`, preserves the
  native scenario identity internally, and applies the pre-registered earliest record
  within the first 32 that retains supported map geometry and has both the SDC and a
  non-SDC vehicle valid from current to next.
- **Simulator-neutral conversion** — agent identity/type/order, SDC identity, the
  10-past/1-current/80-future boundary, validity, supported motion, dimensions, and
  strictly gated lane/road-edge geometry enter the existing `Scenario` contract.
- **Independent parity** — source tensors and raw Waymax leaves are compared against
  EvalSim without reusing the adapter's conversion helpers.
- **End-to-end acceptance** — deterministic conversion, exact Parquet round-trip,
  current-boundary visualization, log replay, real CV and IDM vehicle transitions, and
  a compiled JAX CPU operation pass locally.
- **Safe publication boundary** — raw data, converted scenarios, acceptance reports,
  native IDs, coordinates, and real-data images remain local and ignored.

M3 does **not** claim cohort scale, Waymax environment/policy parity, JAX batching,
metrics, statistical comparison, or learned evaluation. See the
[field mapping](docs/data/womd-waymax-m3-mapping.md), the
[reviewed M3 plan](docs/plans/2026-07-28-m3-local-waymo-vertical-slice.md), and the
[third-party notice](NOTICE.md).

### M4 — Deterministic cohort and Waymax parity ✅ accepted

The pre-registered M4 acceptance passed locally on Apple Silicon CPU. Its public
evidence is deliberately aggregate-only:

- **Auditable population accounting** — exactly shards `00000`–`00009` contained
  2,916 raw records. The frozen complete-case predicate found 1,527 eligible records
  and classified 1,389 as `source_no_supported_map`: no roadgraph group met the frozen
  strict supported-map predicate, which does not mean those records contained no map
  data. The deterministic selector chose 128 scenarios with no quota deficit,
  redistribution, or fallback.
- **Full-cohort contract checks** — every source-eligible record passed adapter and
  independent supported-field parity. On the frozen 128-scenario cohort, direct
  exact-log reference checks and conversion back to `Rollout` passed; EvalSim log
  replay, constant velocity, and IDM each completed all 80 requested transitions.
- **Bounded Waymax reference** — the privileged logged-trajectory waypoint-following
  Waymax IDM reference repeated deterministically on its pre-registered 16-scene ×
  20-transition subset; a separate one-scene kernel JIT gate passed. This is not a
  causal map-route policy or independent ground truth.
- **Measured JAX CPU path** — a fresh-worker, batch-two, 20-warm-run exact-log
  microbenchmark recorded 217.983625 ms compilation, 1.897854 ms median execution,
  2.617709 ms nearest-rank empirical p95, and 1,053.821843 scenarios/s at the median.
  Process peak RSS was 587,808,768 bytes; it is a process high-water mark, not JAX
  device memory or end-to-end cohort throughput.
- **Verification** — 110 focused M4 CLI tests, the locked matrix at 421 passed + 1
  local-data skip, the full Waymo-extra suite at 493 passed + 1 local-data skip, and
  the core-only suite at 418 passed + 22 optional-runtime skips.

This is a complete-case conditional sample from the first ten validation shards, not a
random or representative WOMD benchmark. EvalSim and the reference path share the
pinned Waymax WOMD decode, so the cross-check is not fully independent. M4 establishes
cohort construction, supported-field/exact-log semantics, bounded reference execution,
and a narrow JAX microbenchmark. It does **not** establish simulator realism,
generalization, production scale, or custom/Waymax metric agreement; metrics,
behavioral slices, and statistical comparisons begin in M5. See the
[reviewed M4 plan](docs/plans/2026-07-28-m4-womd-cohort-waymax-parity.md).

## Setup

Uses [`uv`](https://docs.astral.sh/uv/) for an isolated environment (independent of the
system Python):

```bash
export PATH="$HOME/.local/bin:$PATH"   # uv lives here
uv sync --extra dev                    # create .venv and install deps
uv run pytest                          # run the test suite
```

A verified core-only environment reports **418 passed, 22 optional-runtime skips**.
After installing the licensed Waymo extra documented below, the full suite reports
**493 passed, 1 local-data skip**. The focused M4 CLI suite reports **110 passed**, and
the locked M4/Waymax/rollout matrix reports **421 passed, 1 local-data skip**.

## Generate synthetic scenarios

M1 runs entirely on a Mac/CPU and does not read WOMD:

```python
from pathlib import Path

from evalsim.sources import SyntheticSource
from evalsim.viz import plot_scenario

source = SyntheticSource(seed=2026)
scenarios = source.generate(50)

# Write-once: raises FileExistsError rather than replacing an evaluation population.
source.write_manifest(scenarios, "manifests/synthetic_eval_50.json")

Path("outputs").mkdir(exist_ok=True)
figure, axes = plot_scenario(scenarios[0])
figure.savefig("outputs/synthetic_example.png", dpi=150)
```

## Run the M2 baselines

Policies simulate non-ego/world agents; ego follows its logged trajectory. An M3
scenario must set `metadata["current_index"]` to its last observed history frame.

```python
from evalsim.rollout import RolloutEngine
from evalsim.simulators import (
    ConstantVelocityPolicy,
    IDMPolicy,
    LogReplayPolicy,
)

engine = RolloutEngine()
for policy in (LogReplayPolicy(), ConstantVelocityPolicy(), IDMPolicy()):
    rollout = engine.run(scenarios[0], policy, seed=2026)
    print(rollout.sim_name, rollout.metadata["dynamics"]["clamp_counts"])
```

## WOMD data (not included in Git)

The Waymo Open Motion Dataset (WOMD) is **not stored in this repository**. Raw
TFRecords and converted Parquet files are too large for source control and are covered
by the repository's `.gitignore` rules. Do not commit them, push them to GitHub, or add
them with Git LFS.

> **Use restriction:** the M3+ Waymo path is maintained only for the repository
> owner's stated personal, non-commercial interview preparation and experimentation.
> WOMD and Waymax access/use remain governed by their upstream non-commercial
> agreements, including additional restrictions summarized in [`NOTICE.md`](NOTICE.md).
> If the purpose may become commercial, production, real-world-vehicle work, or
> prohibited foundation-model work, stop and obtain a fresh license review.

To run M3 and later Waymo-backed milestones, request access to and download
**Motion Dataset v1.3.1**
from the [official Waymo Open Dataset download page](https://waymo.com/open/download/).
Use the uncompressed TFExample validation shards and place them under:

```text
data/raw/womd/v1.3.1/tf_example/validation/
```

The current development subset uses the first 10 of the 150 validation shards:

```text
validation_tfexample.tfrecord-00000-of-00150
validation_tfexample.tfrecord-00001-of-00150
...
validation_tfexample.tfrecord-00009-of-00150
```

Some browser and cloud-console downloads flatten the object path into the local
filename. Those files may instead look like:

```text
uncompressed_tf_example_validation_validation_tfexample.tfrecord-00000-of-00150
```

Keep either filename form unchanged. Shard identity comes from the
`tfrecord-NNNNN-of-00150` suffix. Additional shards may remain in the directory; treat
`00000` through `00009` as the exact reference subset rather than globbing every file
present.

The resulting local layout should look like:

```text
data/
  raw/
    womd/
      v1.3.1/
        tf_example/
          validation/
            validation_tfexample.tfrecord-00000-of-00150
            validation_tfexample.tfrecord-00001-of-00150
            ...
            validation_tfexample.tfrecord-00009-of-00150
```

One shard is sufficient for the M3 smoke test; M4 uses all 10 fixed shards to construct
the project's deterministic reference cohort. Dataset access and use remain subject to the
[Waymo Open Dataset terms](https://waymo.com/open/terms/). Small manifests and checksums
may be committed for reproducibility, but the dataset files themselves must remain
local.

## Run the M3 local Waymo vertical slice

> Installing the optional `waymo` extra downloads pinned Waymax and related runtime
> dependencies. Read and accept the complete
> [Waymax non-commercial license](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/LICENSE)
> and the [Waymo Open Dataset terms](https://waymo.com/open/terms/) before continuing.
> The summaries in [`NOTICE.md`](NOTICE.md) are not a substitute for those agreements.

After shard `00000` is in the local directory documented above:

```bash
uv sync --extra dev --extra waymo
EVALSIM_RUN_WAYMO_LOCAL=1 uv run --extra waymo pytest -m waymo_local
EVALSIM_RUN_WAYMO_LOCAL=1 uv run --extra waymo evalsim-waymax-smoke --shard 00000
```

Run these commands from the EvalSim Git checkout root. A normally installed wheel can
also run the entry point from that checkout, or use `--project-root /path/to/evalsim`;
the command deliberately refuses an unverified or non-ignored output location.

The environment flag is a deliberate data-access opt-in. The command examines only the
fixed M3 shard and writes a sanitized summary, converted Parquet, and visualization
under ignored `outputs/m3/`. It prints no native scenario ID, trajectory, coordinate,
map sample, or absolute data path.

## Run the M4 local cohort acceptance

After all ten fixed shards are present, run M4 only from a clean Git worktree. The
output directory must be a new ignored descendant of `outputs/m4/`:

```bash
EVALSIM_RUN_WAYMO_LOCAL=1 uv run --extra waymo evalsim-waymax-m4 \
  --project-root "$PWD" \
  --data-dir "$PWD/data/raw/womd/v1.3.1/tf_example/validation" \
  --output-dir "$PWD/outputs/m4/manual-acceptance"
```

The command scans exactly shards `00000`–`00009`, binds evidence to the clean source
commit, and keeps manifests, provenance, native identities, and detailed run artifacts
local and ignored. Use a different new output-directory name for a later run.

## Layout

```
evalsim/
  contracts/    # Seam A/B: Scenario, Rollout, SimulatorPolicy, Metric, RunManifest
  sources/      # scenario producers (synthetic — M1; Waymax/WOMD — M3)
  simulators/   # EvalSim policies (M2) + Waymax reference adapters (M4)
  rollout/      # rollout engine (M2) + counterfactual ego control (M6)
  metrics/      # metric registry + implementations (M5)
  slices/       # real-WOMD scenario slicing (M5)
  stats/        # paired scenario statistics and aggregation (M5)
  perturb/      # counterfactual ego interventions (M6)
  stress/       # evaluator corruptions, calibration, and governance (M7)
  report/       # scorecards and decision artifacts (M5/M11)
  config/       # reproducible CLI, caching, and resume (M10)
  viz.py        # static scenario visualization (M1)
tests/          # contracts, sources, policies, rollout engine, dynamics, and visualization
docs/plans/     # historical M0–M2 design + canonical Waymo-aligned roadmap
docs/data/      # tracked schema mappings and omission crosswalks (never dataset payloads)
docs/interview/ # sanitized role matrix, claim ledger, and study plan
index.html      # interactive, evidence-led technical presentation
public/og.png   # social preview artwork (not an experiment artifact)
scripts/        # dependency-free presentation build and structural checks
```

## Résumé framing

The current implemented stack is **Python · NumPy · PyArrow · Matplotlib**, plus an
optional pinned **WOMD · Waymax · JAX · TensorFlow · Flax** runtime. M3 established the
one-scenario vertical slice; M4 now supports the narrower substantive claim that exact
log-playback mapping and rollout-contract semantics were cross-checked on a
deterministic 128-scenario complete-case conditional cohort from ten WOMD validation
shards, with bounded Waymax reference execution and a measured batch-two JAX CPU
microbenchmark. Metrics, slices, confidence intervals, counterfactuals, stress tests,
learned evaluators, scalable execution, and video/VLM work each remain unavailable as
claims until their evidence gate passes. See the
[claim-to-evidence ledger](docs/interview/claim-evidence-ledger.md).
