# EvalSim — Closed-Loop Simulation-Evaluation Platform

A reproducible platform for evaluating autonomous-driving traffic simulators along
independent dimensions — kinematic realism, agent interaction, map adherence, behavioral
diversity, closed-loop reactivity, and physical feasibility — rather than collapsing
simulation quality into a single leaderboard number.

> **The contribution is the evaluation methodology, not the simulator.** We use simple
> baselines (log-replay, constant-velocity, IDM) with deliberately different failure modes
> and invest the engineering in the system *around* them.

**New here? Read in this order:** this README → the live dashboard
[`index.html`](index.html) → the design doc
[`docs/plans/2026-07-27-evalsim-design.md`](docs/plans/2026-07-27-evalsim-design.md) →
the [implementation plan](docs/plans/2026-07-27-evalsim-implementation-plan.md).

> 📌 **This README is kept up to date as work progresses.** The
> [Progress](#progress) table and [Completed work](#completed-work) changelog below always
> reflect the current state of the repo.

## Design in one paragraph

The platform is built **contract-first** (ports-and-adapters). A frozen `Scenario`
contract has two producers — a synthetic generator that runs locally (Mac/CPU) and a
Waymax/WOMD loader that runs once on a GPU cloud session. Everything downstream
(simulators, rollout engine, metrics, slices, statistics, perturbations, stress-tests,
reporting) consumes **only** the contract, so ~95% of the project is built and validated
on a laptop, and the GPU-only work is quarantined to a single bounded stage.

## Progress

Legend: ✅ done · 🚧 in progress · ⬜ not started · 🖥️ runs on Mac/CPU · ☁️ runs on Cloud/GPU

| # | Milestone | Status | Where |
|---|-----------|:------:|:-----:|
| M0 | Repo skeleton + data contracts | ✅ | 🖥️ |
| M1 | Synthetic scenario source + visualization | ✅ | 🖥️ |
| M2 | Simulator adapters + rollout engine | ⬜ | 🖥️ |
| M3 | Metric registry + core metrics | ⬜ | 🖥️ |
| M4 | Scenario slicing + statistics *(MVP)* | ⬜ | 🖥️ |
| M5 | Counterfactual ego-perturbation + reactivity | ⬜ | 🖥️ |
| M6 | Metric stress-test suite | ⬜ | 🖥️ |
| S1 | Learned real-vs-sim discriminator *(Strong)* | ⬜ | 🖥️ |
| M7 | Config, caching, reproducibility | ⬜ | 🖥️ |
| M8 | Report, visualizations, tests | ⬜ | 🖥️ |
| W1 | WaymaxSource — WOMD ingestion | ⬜ | ☁️ |
| W2 | Rollout dynamics cross-check *(optional)* | ⬜ | ☁️ |

**Tests:** 67 passing · **Last updated:** 2026-07-27 (M1)

## Completed work

### M0 — Package skeleton + data contracts ✅
The frozen seams every later layer depends on:

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
- **54 M1 tests** — 50-scenario contract and Parquet acceptance, determinism,
  collision/map/kinematic invariants, family semantics, manifest provenance and
  immutability, and headless PNG renders for all five families.

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

## WOMD data (not included in Git)

The Waymo Open Motion Dataset (WOMD) is **not stored in this repository**. Raw
TFRecords and converted Parquet files are too large for source control and are covered
by the repository's `.gitignore` rules. Do not commit them, push them to GitHub, or add
them with Git LFS.

To run the WOMD/Waymax stage, request access to and download **Motion Dataset v1.3.1**
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

At least one shard is sufficient for a smoke test; use all 10 shards to reproduce the
project's reference WOMD subset. Dataset access and use remain subject to the
[Waymo Open Dataset terms](https://waymo.com/open/terms/). Small manifests and checksums
may be committed for reproducibility, but the dataset files themselves must remain
local.

## Layout

```
evalsim/
  contracts/    # Seam A/B: Scenario, Rollout, SimulatorPolicy, Metric, RunManifest
  sources/      # scenario producers (synthetic — M1; Waymax/WOMD — W1)
  simulators/   # log-replay, constant-velocity, IDM (M2), corruptions (M6)
  rollout/      # closed-loop rollout engine (M2) + ego perturbations (M5)
  metrics/      # metric registry + implementations (M3)
  slices/       # scenario-slice registry (M4)
  stats/        # scenario-cluster bootstrap + aggregation (M4)
  perturb/      # counterfactual ego-perturbation suite (M5)
  stress/       # metric stress-test corruptions + detection matrix (M6)
  report/       # scorecards, visualizations, report (M8)
  config/       # YAML run config + CLI (M7)
  viz.py        # static scenario visualization (M1)
tests/          # contracts, synthetic source, manifests, and visualization
docs/plans/     # design doc + implementation plan
index.html      # live progress dashboard
```

## Résumé framing

**EvalSim — Closed-Loop Simulation-Evaluation Platform** · Waymo Open Motion Dataset ·
Waymax · JAX · Python. Every claim in the résumé bullets maps to a milestone above; see
the design doc's "Résumé mapping" section for the full traceability.
