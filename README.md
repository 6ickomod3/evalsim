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
| M1 | Synthetic scenario source + visualization | ⬜ | 🖥️ |
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

**Tests:** 13 passing · **Last updated:** 2026-07-27 (M0)

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

## Setup

Uses [`uv`](https://docs.astral.sh/uv/) for an isolated environment (independent of the
system Python):

```bash
export PATH="$HOME/.local/bin:$PATH"   # uv lives here
uv sync --extra dev                    # create .venv and install deps
uv run pytest                          # run the test suite
```

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
tests/          # contract round-trip, validation, interfaces, manifest
docs/plans/     # design doc + implementation plan
index.html      # live progress dashboard
```

## Résumé framing

**EvalSim — Closed-Loop Simulation-Evaluation Platform** · Waymo Open Motion Dataset ·
Waymax · JAX · Python. Every claim in the résumé bullets maps to a milestone above; see
the design doc's "Résumé mapping" section for the full traceability.
