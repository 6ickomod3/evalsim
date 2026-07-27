# EvalSim — Closed-Loop Simulation-Evaluation Platform

A reproducible platform for evaluating autonomous-driving traffic simulators along
independent dimensions — kinematic realism, agent interaction, map adherence, behavioral
diversity, closed-loop reactivity, and physical feasibility — rather than collapsing
simulation quality into a single leaderboard number.

**The contribution is the evaluation methodology, not the simulator.** See
[`docs/plans/2026-07-27-evalsim-design.md`](docs/plans/2026-07-27-evalsim-design.md) for
the full design and [`index.html`](index.html) for the live progress dashboard.

## Design in one paragraph

The platform is built **contract-first**: a frozen `Scenario` contract has two producers —
a synthetic generator that runs locally (Mac/CPU) and a Waymax/WOMD loader that runs once
on a GPU cloud session. Everything downstream (simulators, rollout engine, metrics,
slices, statistics, perturbations, stress-tests, reporting) consumes only the contract, so
~95% of the project is built and validated on a laptop.

## Status

**M0 complete** — package skeleton and the core data contracts:

- `Scenario`, `Agent`, `MapPolyline`, `Rollout` (Seam A) with lossless Parquet round-trip
- `SimulatorPolicy` (Seam B) and `Metric` interfaces
- `RunManifest` (reproducibility)

See the [implementation plan](docs/plans/2026-07-27-evalsim-implementation-plan.md) for
the milestone roadmap (M0–M8, S1, W1–W2).

## Setup

Uses [`uv`](https://docs.astral.sh/uv/) for an isolated environment:

```bash
uv sync --extra dev     # create .venv and install deps
uv run pytest           # run the test suite
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
```
