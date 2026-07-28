# EvalSim — Design Document

**Date:** 2026-07-27
**Status:** Historical baseline for M0–M2
**Owner:** jdai

> The contract-first architecture and completed M0–M2 decisions remain valid. The
> hardware assumptions, Waymax scope, testing strategy for real fixtures, and milestone
> sequence from M3 onward are superseded by the
> [Waymo-aligned roadmap](2026-07-28-waymo-aligned-roadmap.md). In particular, WOMD,
> Waymax, and JAX now begin in M3 on the local CPU rather than a late cloud-only stage.

---

## 1. What this project is

EvalSim is a **closed-loop simulation-evaluation platform** for autonomous-driving traffic simulators. The
central question it answers:

> How do we determine whether a traffic simulator produces behavior that is realistic, reactive, diverse,
> physically feasible, and *useful* for evaluating an autonomous-driving system — rather than collapsing all of
> that into a single leaderboard number?

**The contribution is the evaluation methodology, not the simulator.** We deliberately use simple baseline
simulators (log-replay, constant-velocity, IDM) with clearly different failure modes, and invest all the
engineering in the system *around* them: a pluggable metric registry, closed-loop rollout, counterfactual
ego-perturbation, metric stress-testing, scenario slicing, scenario-level statistical validation, a learned
real-vs-sim discriminator, reproducibility, and release-oriented reporting.

---

## 2. Constraints that shaped this design

| Constraint | Consequence |
|---|---|
| Interview deadline ~1–2 weeks | Ruthless YAGNI; protect the highest-signal milestones (M5, M6). |
| Development machine is **Apple Silicon Mac, CPU-only** | `waymo-open-dataset` (Linux/x86 wheels only) and Waymax (GPU-oriented) cannot run locally. |
| Google Cloud (Colab / Vertex / GCE GPU) available | One bounded cloud session can run the Waymax/WOMD stage. |
| New to JAX/Waymax | Minimize the Waymax surface to a single, well-contained stage. |
| Résumé header names **WOMD · Waymax · JAX · Python** | The real AV stack must be genuinely used — satisfied by the one cloud stage (W1). |

---

## 3. Key design decision: contract-first, substrate-agnostic

The platform is built with **dependency inversion at the data boundary** (ports-and-adapters). We define a
frozen **`Scenario` contract** and build two producers for it:

- **`SyntheticSource`** — runs on the Mac; generates scenarios directly into the contract. Drives all local
  development, testing, demos, and — importantly — the metric stress-tests, where synthetic *known ground truth*
  is superior to real data.
- **`WaymaxSource`** — written and unit-tested on the Mac against the same contract; **executed once on Google
  Cloud** to convert real WOMD scenarios into the contract, then cached to Parquet and pulled back to the Mac.

Everything downstream (metrics, slices, stats, perturbations, stress-tests, discriminator, reports) consumes
**only the contract** and is physically unable to tell which producer created the data. Result:

> **~95% of the project is built and validated on the Mac.** The Mac-blocked surface shrinks to one stage
> (real WOMD ingestion), which is code-complete on the Mac and *run* in one short cloud session.

### Minimal Waymax surface

Waymax is reduced to **(a) WOMD loader** and **(b) a one-time rollout dynamics cross-check**. The three
baseline simulators and the closed-loop rollout engine are pure NumPy on the Mac. We do **not** use Waymax to
drive rollouts or to compute metrics — building our own metric registry *is* the project.

---

## 4. Architecture

```text
                 ┌───────────────────────────┐
   PRODUCERS →   │   Scenario Contract (A)    │   ← frozen schema (Parquet-serializable)
                 └───────────────────────────┘
       ▲                     │
   ┌───┴────────┐            ▼
   │ Synthetic  │   ┌───────────────────────────┐
   │  Source    │   │  Scenario Slicer / Filters │
   │  (Mac)     │   └───────────────────────────┘
   └────────────┘            │
   ┌────────────┐            ▼
   │ Waymax     │   ┌───────────────────────────┐
   │  Source    │   │  Simulator Adapter Layer(B)│ log-replay · const-vel · IDM · corrupted
   │ (Cloud→PQ) │   └───────────────────────────┘
   └────────────┘            │
                             ▼
                 ┌───────────────────────────┐
                 │  Closed-Loop Rollout Engine│ standard + counterfactual ego perturbations
                 └───────────────────────────┘
                             │
                             ▼
                 ┌───────────────────────────┐
                 │   Metric Registry          │ kinematic · interaction · map · diversity · reactivity
                 └───────────────────────────┘
                             │
                             ▼
                 ┌───────────────────────────┐
                 │ Per-Scenario Result Store  │ Parquet + run manifest + cache keys (DuckDB queries)
                 └───────────────────────────┘
                             │
                             ▼
                 ┌───────────────────────────┐
                 │ Aggregation & Statistics   │ slice scorecards · paired diffs · scenario-cluster bootstrap
                 └───────────────────────────┘
                             │
                             ▼
                 ┌───────────────────────────┐
                 │ Reports & Visualizations   │ scorecards · stress matrix · failure case studies
                 └───────────────────────────┘
```

---

## 5. Contracts (the seams)

### Seam A — `Scenario`

```text
Scenario:
  scenario_id: str
  timestamps: float[T]
  agents: [
    { id: int, type: {vehicle,pedestrian,cyclist},
      valid: bool[T], x: float[T], y: float[T], heading: float[T],
      vx: float[T], vy: float[T], length: float, width: float }
  ]
  map: [ { type: {lane,road_edge,crosswalk}, xy: float[P,2] } ]
  ego_index: int
  metadata: { source: "synthetic"|"womd", split, tags[] }
```

`Rollout` mirrors this: the per-agent state produced by a simulator over the horizon, plus `sim_name`,
`sim_version`, `seed`, `perturbation`.

### Seam B — `SimulatorPolicy` (§7 of the draft)

```python
class SimulatorPolicy:
    def initialize(self, scenario, seed): ...
    def step(self, state, observation): ...
    def metadata(self): ...   # name, version, deterministic?, required features, known limits
```

### Metric contract (§8)

```python
class Metric:
    name: str; version: str
    def validate_inputs(self, scenario, rollout): ...
    def compute(self, scenario, rollout): ...     # frame/agent/scenario level values
    def aggregate(self, per_scenario_values): ... # with higher_is_better, unit-of-analysis
```

No single composite score is the headline; component metrics and slices stay visible.

---

## 6. Milestone map (Mac vs Cloud)

| # | Milestone | Runs where |
|---|---|---|
| M0 | Repo skeleton + contracts (Seam A/B, Metric) + config-lite | Mac |
| M1 | `SyntheticSource` + scenario visualizer | Mac |
| M2 | Simulator adapters (log-replay, const-vel, IDM) + NumPy rollout engine | Mac |
| M3 | Metric registry + kinematic/interaction/map metrics + Parquet result store | Mac |
| M4 | Scenario-slice registry + scorecards + scenario-cluster bootstrap + paired diffs | Mac |
| M5 | Counterfactual ego-perturbation suite + reactivity metrics | Mac |
| M6 | Metric stress-test suite (frozen/mode-collapse/over-dispersion/infeasible) + detection matrix | Mac |
| S1 | Learned real-vs-sim discriminator + metric-disagreement analysis | Mac |
| M7 | Config-driven runs, run manifest, deterministic caching | Mac |
| M8 | README + technical report + curated failure viz + ~30 tests | Mac |
| W1 | `WaymaxSource`: WOMD → Scenario contract → Parquet | **Cloud** |
| W2 | (optional) rollout dynamics cross-check vs Waymax | **Cloud** |

**MVP = M0–M4.** Highest-signal, most-unique = **M5, M6** (protect these). S1 is required to honor the
"…and learned realism metrics" résumé claim. W1 makes the header (WOMD/Waymax/JAX) literally true.

---

## 7. Testing strategy (~30 tests)

- **Contract tests (Mac):** schema conformance + invariants (valid-mask consistency, headings in range, no
  NaNs, monotonic timestamps). Both producers must pass; passing gives confidence in the unrun `WaymaxSource`.
- **Unit tests:** metric math on synthetic trajectories, TTC edge cases, collision geometry, map-adherence
  eligibility, slice definitions, perturbation severity, cache-key generation, manifest serialization,
  bootstrap aggregation.
- **Property/invariant tests:** log-replay with no perturbation ≈ recorded log; frozen agents ≈ zero speed;
  fixed seed reproduces stochastic rollouts; scenario splits never overlap.
- **Golden fixture:** one hand-checked WOMD scenario committed to diff `WaymaxSource` output against.
- **Cloud smoke test:** run `WaymaxSource` on **1 scenario** in Colab; assert it passes the same contract tests.

---

## 8. Reproducibility

Every result traces to a run manifest (dataset version, scenario manifest, sim name+version+config, rollouts,
seeds, metric versions, slice versions, perturbation defs, commit, timestamp, parent run). Per-scenario results
stored **before** aggregation so slices/stats can be recomputed without re-simulating. Deterministic cache key
= (scenario_id, sim_version, sim_config, seed, perturbation, metric_version). Immutable, partitionable Parquet
→ clean migration path to distributed execution (documented, not built).

---

## 9. Résumé mapping (truthfulness check)

| Résumé claim | Milestone |
|---|---|
| Reproducible platform | M0, M7 |
| log-replay / const-velocity / IDM | M2 |
| kinematic realism / interaction / map adherence | M3 |
| behavioral slices | M4 |
| scenario-level confidence intervals | M4 |
| counterfactual ego-perturbation | M5 |
| reactivity + **nonreactivity** detection | M5 |
| metric stress-test | M6 |
| mode collapse / excessive dispersion / kinematic infeasibility detection | M6 |
| hand-designed realism metrics | M3 |
| **learned** realism metrics | S1 |
| header: **WOMD · Waymax · JAX** | W1 (executed on Google Cloud) |

Every claim is covered. Header requires only the single bounded cloud stage.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| WOMD access latency | Platform is fully functional on synthetic; W1 can land late without blocking anything else. |
| Waymax/JAX learning curve | Surface minimized to WOMD loader; smoke-tested on 1 scenario before full run. |
| "Not real data" interview pushback | Synthetic is *superior* for stress-tests (ground truth); W1 adds real-data validation; framed honestly. |
| Time overrun | Stop-lines: MVP after M4; strong version after M6+S1; W1 last. |
| Metrics silently wrong | Contract + property tests; corrupted simulators act as metric unit tests (M6). |

---

## 11. Out of scope / future work

Learned stochastic policy (GRU/MLP simulator); distributed/multi-process execution; interactive dashboard;
advanced generative distribution metrics; automatic failure-discovery/scenario search; severity-weighted
release policy. Begin only after M0–M8 + S1 + W1 are complete and reproducible.
