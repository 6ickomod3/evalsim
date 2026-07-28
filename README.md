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
contract currently has a synthetic producer that runs locally. M3 now adds a second
WOMD producer through Waymax/JAX on the local Mac/CPU, and M4 makes Waymax a recurring
reference rollout and semantic-parity path. Downstream components continue to consume
**only** project contracts. Synthetic scenes remain analytic oracles while every core
feature from M3 onward also gets a real-WOMD acceptance path.

## Progress

Legend: ✅ done · 🚧 in progress · ⬜ not started · 🖥️ Mac/CPU · ☁️ optional cloud/accelerator

| # | Milestone | Status | Where |
|---|-----------|:------:|:-----:|
| M0 | Repo skeleton + data contracts | ✅ | 🖥️ |
| M1 | Synthetic scenario source + visualization | ✅ | 🖥️ |
| M2 | Simulator adapters + rollout engine | ✅ | 🖥️ |
| M3 | Local WOMD → Waymax/JAX → EvalSim vertical slice | ⬜ | 🖥️ |
| M4 | Deterministic WOMD cohort + Waymax parity | ⬜ | 🖥️ |
| M5 | Real-WOMD metrics + statistical scorecards | ⬜ | 🖥️ |
| M6 | Counterfactual closed-loop reactivity | ⬜ | 🖥️ |
| M7 | Evaluator red-team + metric governance | ⬜ | 🖥️ |
| M8 | JAX/Flax learned realism discriminator | ⬜ | 🖥️/☁️ |
| M9 | Semantic-video and real camera/VLM bridge | ⬜ | 🖥️/☁️ |
| M10 | Scalable, resumable evaluation pipeline | ⬜ | 🖥️/☁️ |
| M11 | Decision package + staff-caliber communication | ⬜ | 🖥️ |

**Tests:** 134 passing · **Last updated:** 2026-07-28 (roadmap revised; M2 remains the implementation boundary)

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

Local, deterministic scenarios with the same contract that the later WOMD adapter will
produce:

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
  M5 perturbations.
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

## Setup

Uses [`uv`](https://docs.astral.sh/uv/) for an isolated environment (independent of the
system Python):

```bash
export PATH="$HOME/.local/bin:$PATH"   # uv lives here
uv sync --extra dev                    # create .venv and install deps
uv run pytest                          # run the test suite
```

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
docs/interview/ # sanitized role matrix, claim ledger, and study plan
index.html      # interactive, evidence-led technical presentation
public/og.png   # social preview artwork (not an experiment artifact)
scripts/        # dependency-free presentation build and structural checks
```

## Résumé framing

The current implemented stack is **Python · NumPy · PyArrow · Matplotlib**. Downloaded
data does not yet make **Waymo Open Motion Dataset · Waymax · JAX** an implemented
claim. M3 makes the integration minimally true; M4 makes the Waymax/JAX usage
substantive. Metrics, slices, confidence intervals, counterfactuals, stress tests,
learned evaluators, scale, and video/VLM work each remain unavailable as claims until
their evidence gate passes. See the
[claim-to-evidence ledger](docs/interview/claim-evidence-ledger.md).
