# EvalSim — Implementation Plan

**Date:** 2026-07-27
**Companion to:** `2026-07-27-evalsim-design.md`
**Sequence:** M0 → M1 → M2 → M3 → M4 → M5 → M6 → S1 → M7 → M8 → W1 → W2

Legend — **Runs:** `Mac` (local, CPU) or `Cloud` (Google Colab/Vertex/GCE GPU).
Stop-lines: **MVP** after M4 · **Strong** after M6+S1 · **Header-true** after W1.

---

## M0 — Repo skeleton + contracts   ·  Runs: Mac  ·  ~0.5 day

**Goal:** Establish the seams everything else depends on.

Tasks
- [x] `pyproject.toml` / venv; deps: `numpy pandas pyarrow duckdb matplotlib scipy scikit-learn pytest pyyaml`.
- [x] Package layout: `evalsim/{contracts,sources,simulators,rollout,metrics,slices,stats,perturb,stress,report,config}`.
- [x] Define `Scenario`, `Agent`, `MapPolyline`, `Rollout` dataclasses (Seam A) + Parquet (de)serialization.
- [x] Define `SimulatorPolicy` (Seam B) and `Metric` base classes.
- [x] `RunManifest` dataclass + JSON serialization.

**Acceptance:** a `Scenario` round-trips to/from Parquet losslessly; `pytest` runs (even if ~empty).

---

## M1 — Synthetic scenario source + visualization   ·  Runs: Mac  ·  ~1 day

**Goal:** Get real-shaped scenarios into the contract, locally and instantly.

Tasks
- [x] `SyntheticSource` generating parametric scenarios: straight-road following, 4-way intersection,
      merge, turning, pedestrian-crossing. Deterministic by seed.
- [x] Emit valid masks, lane/edge polylines, ego index, tags for later slicing.
- [x] `viz.plot_scenario()` — matplotlib render of map + agent trajectories + ego highlight.
- [x] Scenario manifest writer (immutable list of generated scenario_ids).

**Acceptance:** generate 50 scenarios, render 3 to PNG, all pass contract tests.

---

## M2 — Simulator adapters + rollout engine   ·  Runs: Mac  ·  ~1.5 days

**Goal:** Three baselines with genuinely different failure modes + a closed-loop rollout engine.

Tasks
- [x] `LogReplayPolicy` (replays recorded agent states).
- [x] `ConstantVelocityPolicy` (extrapolate current velocity).
- [x] `IDMPolicy` (longitudinal reactive; desired speed, min gap, safe headway).
- [x] `RolloutEngine.run(scenario, policy, seed, perturbation=None)` → `Rollout`; NumPy, deterministic.
- [x] Bicycle/point-mass kinematic integration with feasibility clamping.

**Acceptance:** log-replay (no perturbation) reproduces the log within tolerance; frozen-input sanity holds;
fixed seed → identical rollout.

---

## M3 — Metric registry + core metrics   ·  Runs: Mac  ·  ~2 days

**Goal:** Pluggable metrics across the first three dimensions + per-scenario result store.

Tasks
- [ ] `MetricRegistry` (register by name/version, validate inputs, compute, aggregate).
- [ ] Kinematic: speed dist, longitudinal/lateral accel, jerk, yaw rate, log-divergence (report as distributions).
- [ ] Interaction: collision/overlap, min inter-agent distance, TTC, following distance, response latency.
- [ ] Map adherence: off-road rate, wrong-way, distance-to-lane-center, lane departure (with per-agent-type eligibility).
- [ ] Per-scenario result store → Parquet (partitioned by sim/scenario); DuckDB query helpers.

**Acceptance:** run all metrics for 3 sims over 50 scenarios; results land in Parquet; distributional metrics
return distributions not just means; metric math unit-tested on synthetic trajectories.

---

## M4 — Scenario slicing + statistics   ·  Runs: Mac  ·  ~1.5 days   ·  **[MVP complete]**

**Goal:** Sliced scorecards with honest uncertainty.

Tasks
- [ ] Slice registry (intersection, turn, following, dense, low-TTC, pedestrian-present, …), versioned + deterministic.
- [ ] Predefined vs exploratory slice distinction; per-slice sample-size reporting + small-slice flags.
- [ ] Paired per-scenario differences between sim pairs.
- [ ] Scenario-cluster bootstrap (resample scenarios, ~1000 resamples, 95% CI, effect sizes).
- [ ] FDR handling across exploratory slices.
- [ ] Scorecard renderer (per metric × per slice, with CIs).

**Acceptance:** a scorecard shows a sim pair with CIs; bootstrap resamples scenarios (not frames); tiny slices
flagged; a paired comparison reproduces on rerun.

---

## M5 — Counterfactual ego-perturbation + reactivity   ·  Runs: Mac  ·  ~1.5 days   ·  **[highest signal]**

**Goal:** The distinctive contribution — measure closed-loop reactivity under ego divergence.

Tasks
- [ ] Perturbation families with severities: braking (early/hard/stop), acceleration (delay/slow), timing/lateral (late entry/lane shift).
- [ ] Explicit per-scenario eligibility logic (e.g. braking test needs a relevant following agent).
- [ ] Apply perturbation to ego while holding initial context constant; re-run world agents.
- [ ] Reactivity metrics: Δfollowing-accel, Δmin-distance, ΔTTC, collision-avoidance rate, response latency,
      progress lost, response smoothness — distinguishing **nonreactivity** vs **overreaction**.
- [ ] Reactivity report: log-replay (nonreactive) vs IDM (reactive/over-conservative) comparison.

**Acceptance:** log-replay shows near-zero reactivity / induced collisions under ego braking; IDM adapts;
report quantifies both with paired CIs; ≥1 qualitative failure viz.

---

## M6 — Metric stress-test suite   ·  Runs: Mac  ·  ~1 day   ·  **[highest signal]**

**Goal:** Treat metrics as systems that can fail; prove which metrics catch which defects.

Tasks
- [ ] Corrupted simulators: `FrozenPolicy`, `ModeCollapsePolicy`, `OverDispersionPolicy`, `InfeasibleMotionPolicy`
      (+ reuse log-replay as nonreactive, IDM-tuned as over-conservative).
- [ ] Run full metric suite against each corruption.
- [ ] Generate the **metric-detection matrix** from real results (defect × metric → strong/weak/mixed).
- [ ] Document which "reasonable" metrics can be gamed (e.g. collision rate by a frozen sim).

**Acceptance:** matrix is data-derived; each corruption is detected by ≥1 metric and *missed* by ≥1 (proving
non-triviality); frozen sim shows low collision but fails speed/progress/reactivity.

---

## S1 — Learned real-vs-sim discriminator   ·  Runs: Mac  ·  ~1 day   ·  **[Strong complete]**

**Goal:** A learned realism metric + disagreement analysis vs hand-designed metrics.

Tasks
- [ ] Feature extraction (speed/accel/jerk/yaw/min-dist/TTC/off-road/interaction summaries + optional temporal embedding).
- [ ] **Leakage-safe splits: all rollouts from one source scenario stay in the same split.**
- [ ] Train logistic reg / gradient-boosted trees / small MLP (CPU-fine).
- [ ] Report held-out ROC-AUC + CI, calibration, feature ablations, per-sim / per-slice performance.
- [ ] **Disagreement analysis:** cases where the classifier and hand-designed metrics diverge (incl. artifact detection).

**Acceptance:** splits provably non-overlapping (test); AUC with CI reported; ≥1 documented disagreement case;
honest negative result acceptable and stated if found.

---

## M7 — Config, caching, reproducibility   ·  Runs: Mac  ·  ~1 day

**Goal:** One-command, config-driven, cached, reproducible runs.

Tasks
- [ ] YAML run config (dataset/manifest/sim/params/rollouts/seeds/perturbations/metrics/slices/stats/output).
- [ ] CLI: `evalsim run config.yaml` → manifest + results + scorecard.
- [ ] Deterministic cache keys; skip recompute when inputs unchanged; retry only failed scenarios.
- [ ] Measure & log cached-vs-full speedup (report the real number).

**Acceptance:** two identical runs — second hits cache and is measurably faster; changing a slice def reuses
per-scenario metrics; every scorecard number traces to a manifest.

---

## M8 — Report, visualizations, tests   ·  Runs: Mac  ·  ~1.5 days

**Goal:** Interview-ready artifacts.

Tasks
- [ ] Technical report (exec summary, architecture, setup, results, counterfactual, stress matrix, learned-eval, failure cases, scalability, limitations, future work).
- [ ] README with install + one-command benchmark.
- [ ] 10–20 curated failure visualizations (not hundreds).
- [ ] Bring test count to ~30 across unit / integration / property.
- [ ] Fill résumé numbers ([N] scenarios, [R] rollouts, [S] slices, [P] perturbations, [K] blind spots) from a saved run.

**Acceptance:** `README` install → one-command run works from clean checkout; report renders; ~30 tests green.

---

## W1 — WaymaxSource (WOMD ingestion)   ·  Runs: Cloud  ·  ~0.5–1 day

**Goal:** Make the résumé header literally true; add real-data validation.

Tasks
- [ ] (Mac) Write `WaymaxSource` targeting the **same `Scenario` contract**; unit-test against contract + golden fixture.
- [ ] (Cloud) Colab/Vertex: install `waymo-open-dataset` + Waymax; obtain WOMD access + a shard.
- [ ] (Cloud) **Smoke test:** convert 1 scenario, assert it passes contract tests.
- [ ] (Cloud) Convert scenario manifest (start 100–500) → Parquet; download to Mac.
- [ ] (Mac) Re-run the entire platform on real Parquet — unchanged.

**Acceptance:** real WOMD Parquet passes all contract tests; scorecards/perturbations/stress-tests run on real
data with no code changes downstream.

---

## W2 — Rollout dynamics cross-check (optional)   ·  Runs: Cloud  ·  ~0.5 day

**Goal:** Validate the NumPy rollout engine against Waymax reference dynamics.

Tasks
- [ ] (Cloud) Run one scenario through Waymax dynamics; compare trajectories vs NumPy engine.
- [ ] Document divergence + tolerance; note as validation evidence in the report.

**Acceptance:** documented agreement (or explained divergence) for ≥1 scenario.

---

## Suggested 2-week schedule

| Days | Work |
|---|---|
| 1 | M0 + M1 |
| 2–3 | M2 + start M3 |
| 4–5 | finish M3 + M4  → **MVP** |
| 6–7 | M5 |
| 8 | M6  → **Strong build core** |
| 9 | S1  → **Strong complete** |
| 10 | M7 |
| 11–12 | M8 (report/tests/viz) |
| 13 | W1 on Google Cloud → real-data re-run |
| 14 | buffer / W2 / résumé numbers |
