# WOMD / Waymax / EvalSim M5 metric crosswalk

**Status:** Accepted pre-registered semantics, data-free overlap-boundary amendment,
outcome-blind cadence-domain amendment, data-free implementation, and official-runner
implementation; no M5 WOMD outcome or native M5 WOMD parity result computed
**Pinned Waymax commit:** `a64dfec9be8576b60d9cecc94f406d9812d4a7d0`

This document separates numerical equivalence, deliberate EvalSim definitions, and
unsupported semantics. A shared name is not evidence of equivalence.

Data-free adversarial testing on 2026-07-29 found rotated, strict zero-margin float32
boxes for which NumPy/libm and XLA make different discrete overlap decisions because
their trigonometric results differ at the bit level. This falsifies universal backend
bit-equivalence before any M5 WOMD outcome access. The counterexample is retained in
tests; exact observed flags remain mandatory on the frozen parity subset.

Before a fresh official execution, a preserved pre-metric failure falsified the
additional assumption that every accepted source interval must equal exactly
`100_000` microseconds. No metric, slice, scorecard, determinism, or native-parity
result was produced or inspected. The accepted outcome-blind correction removes that
cadence eligibility gate without changing the cohort, states, masks, thresholds, or
formula: the pinned Waymax kinematic diagnostic uses a fixed `0.1 s`
inverse-dynamics timebase and does not consume trajectory timestamps.

## Accepted data-free and official-runner implementation evidence

The data-free M5 path was accepted on 2026-07-29 after architecture, methods/statistics,
and privacy/publication review. The M5-focused suite reports 264 passing tests; the
locked Waymo-extra suite reports 757 passed with one expected local-data skip; and the
fresh clean core-only suite without JAX, jaxlib, TensorFlow, Flax, or Waymax reports
779 passed with 25 expected optional/local skips. This is software and
analytic-oracle evidence only:

- five fixed 91-frame synthetic scenarios run through three EvalSim policies and
  thirteen metrics, producing exactly 195 metric rows;
- eight source-only slices produce exactly 40 membership rows;
- the complete thirteen-metric × eight-slice × three-contrast domain produces exactly
  312 scorecard rows, while the native Waymax parity table has zero rows;
- 25 exact log-replay zero oracles cover all five synthetic cases and all five
  registered error metrics, checking every eligible component; and
- every data-free scorecard has paired N between zero and five, is labeled
  `insufficient_n`, suppresses effects and stability bands, and forbids directional
  language.

The command has no WOMD, accepted-M4, official, or Waymax execution argument. Its
terminal status, manifest, and scorecard use the `data_free_test` profile, and normal
result verification rejects that profile unless the caller explicitly opts into
data-free verification.

The accepted official-runner boundary adds a source-only 16-case selector, exact-log
reference execution, one-case-at-a-time native parity evaluation, pre-metric parity
order receipt, post-evaluation determinism receipt, and exact official result-store
domains. It passed 14 runner tests, a public mocked 128-case lifecycle, 18 injected
failure boundaries, 34 adversarial lifecycle cases, the full repository suite with 876
passing tests and one expected local-data skip, and final adversarial review with
**ACCEPT**. A fresh core-only verification after the cadence amendment passed 790
tests with 28 expected optional/local skips. The mocked lifecycle enforces 6,656
metric rows, 1,024 slice rows, 312 scorecard rows, and 144 parity-summary rows.

No real-WOMD metric effect, slice prevalence, missingness result, or native metric
parity result is established here: the official lifecycle used mocks and no M5 WOMD
outcome was computed. The later data execution remains conditional on the unchanged
128-scenario complete-case M4 cohort. It is not population inference; the custom and
reference paths share the pinned Waymax decoder; and log replay remains a privileged
logged-future construction oracle rather than independent ground truth.

Pinned sources:

- [`imitation.py`](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/metrics/imitation.py)
- [`overlap.py`](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/metrics/overlap.py)
- [`comfort.py`](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/metrics/comfort.py)
- [`bicycle_model.py`](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/dynamics/bicycle_model.py)
- [`roadgraph.py`](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/metrics/roadgraph.py)
- [`route.py`](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/metrics/route.py)
- [`metric_factory.py`](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/metrics/metric_factory.py)

## Equivalence matrix

| Upstream registry name | Pinned upstream output | EvalSim M5 treatment | Numerical status |
|---|---|---|---|
| `log_divergence` | Per-object current XY Euclidean distance; valid iff simulated and logged object are valid | `position_error_m`, then restrict to future non-ego target components for scorecards | Bounded-tolerance continuous parity target |
| `overlap` | Per-object binary flag for strict oriented-box overlap with any other valid object; output validity is target validity | `oriented_box_overlap_rate`; preserve flags before target/window aggregation | Exact mask and bounded observed discrete parity gate; universal zero-margin bit-equivalence falsified |
| `kinematic_infeasibility` | Per-object binary inverse-bicycle acceleration/curvature threshold for the transition ending at current frame, evaluated with fixed `dt = 0.1 s` | `waymax_kinematic_infeasibility_rate` version `1.0.1`; preserve action mask and flags before aggregation | Exact mask and discrete parity target; fixed-step diagnostic, not physical-time-normalized |
| `offroad` | Any box corner lies on the positive signed side of its nearest eligible 3-D road-edge sample | No custom native-equivalent metric in M5 | Unsupported by current contract |
| `sdc_wrongway` | Thresholded XY distance to any valid SDC path sample; no direction test | No metric under this name; possible M6 typed path metric must be called path-sample distance | Unsupported and upstream name is misleading |
| `sdc_progression` | SDC projection to the nearest valid on-route path samples and arc-length ratio | Deferred to typed M6 route context | Unsupported by current contract |
| `sdc_off_route` | Thresholded comparison of SDC distance to on-route and off-route path samples | Deferred to typed M6 route context | Unsupported by current contract |

## Parity anchors

All three anchors use one canonical NumPy float32 contract view. Candidate/logged
motion and candidate dimensions are cast once; the exact resulting bits feed both the
custom functions and pinned Waymax adapter. Boolean masks and integer identity are
unchanged. This is necessary because EvalSim contracts otherwise hold float64 values
while pinned Waymax `Trajectory` requires float32.

Canonical inputs remove input-quantization ambiguity; they do not force NumPy/libm and
XLA to use identical trigonometric rounding. Position and kinematic anchors reproduce
the pinned arithmetic branches and are checked natively. Overlap uses the
source-neutral NumPy scorecard definition and must match every valid native flag in
the bounded observed parity subset; any mismatch fails the run and is retained rather
than normalized.

The kinematic metric and its native parity anchor use version `1.0.1`. Every other M5
metric and the log-divergence and overlap parity anchors remain version `1.0.0`; the
result schema itself is unchanged because it already records and validates the
per-metric version.

### Logged-position divergence

For object `i` at frame `t`:

```text
value[i,t] = sqrt(
    (sim_x[i,t] - log_x[i,t])**2
  + (sim_y[i,t] - log_y[i,t])**2
)
valid[i,t] = sim_valid[i,t] AND log_valid[i,t]
```

The custom implementation keeps all object/frame components for parity. The M5
scorecard subsequently restricts targets to non-ego future frames and takes one
per-scenario mean.

Continuous parity uses exact masks and:

```text
abs(custom - reference)
<= max(1e-6, 8 * float32_ulp(abs(reference)))
```

with `rtol = 0`. Exact log replay being zero is a mapping oracle, so the parity subset
also applies the formula to constant-velocity and EvalSim-IDM snapshots.

For finite `x`,
`float32_ulp(x) = float(nextafter(float32(abs(x)), float32(+inf)) -
float32(abs(x)))`, with `nextafter` evaluated in float32.

### Oriented-box overlap

For target object `i`, upstream geometry tests every oriented
`[x, y, length, width, yaw]` box against all others. Self-overlap is removed. An
invalid other object cannot trigger the target. The target's returned validity is its
own current validity.

Strict positive separating-axis intersection is overlap. Exact edge touching is not.
The per-target flag means “overlaps at least one object”; summing flags is not a count
of unique collision pairs.

Parity requires exact identity/order and target validity everywhere, then every
discrete flag only where the target is valid. Upstream raw values for invalid targets
are semantically masked and may be nonzero; M5 neither compares them nor calls
post-mask zero-filling raw parity.

This requirement is exact for the observed parity matrix, not universal
bit-equivalence over every possible float32 box. A permanent synthetic rotated
zero-margin counterexample demonstrates the backend boundary. M5 therefore claims a
bounded native semantic cross-check only if all frozen observed flags agree.

### Kinematic infeasibility

The pinned implementation calls inverse bicycle dynamics on the transition ending at
the current frame. It uses fixed `dt = 0.1 s`, not a timestamp-derived value.

```text
speed     = hypot(vx_old, vy_old)
new_speed = hypot(vx_new, vy_new)
accel     = (new_speed - speed) / 0.1
```

`old_yaw` is the candidate rollout heading at `t-1`. New yaw uses candidate velocity
direction when `abs(new_speed) > 0.6 m/s`; at or below `0.6 m/s`, `new_yaw` is the
candidate rollout heading at `t`. The logged trajectory remains separate and is used
only for logged-reference metrics. The wrapped candidate yaw delta is divided by:

```text
speed * 0.1 + 0.5 * accel * 0.1**2
```

Steering curvature is then forced to zero if either old or new speed is strictly below
`0.6 m/s`. The output flag is:

```text
abs(accel) > 10.4 + 1e-3
OR
abs(steering_curvature) > 0.3 + 1e-3
```

The action validity mask is old-valid AND new-valid. Parity retains all object types;
the scorecard subsequently restricts target components to non-ego vehicles because a
bicycle-model threshold is weak quality semantics for pedestrians and cyclists.
Parity requires exact masks, threshold branches, and binary flags.

Version `1.0.1` names the required timebase
`fixed_0.1_s_inverse_dynamics_timebase`. Source cadence does not select, exclude,
reweight, round, or normalize a case or transition, and nonuniform positive source
intervals do not alter the fixed-step calculation for an identical candidate state
sequence. They can still change policy rollouts indirectly because rollout dynamics
use the actual interval duration. Exact valid-object timestamp consensus, strict
monotonicity, source-to-contract timestamp identity, rollout timestamp identity, and
source immutability remain required.

This Waymax-semantic diagnostic is deliberately not a physical-time-normalized
feasibility measure. EvalSim rollout dynamics and the acceleration, jerk, yaw-rate,
and continuity metrics continue to use the actual positive source intervals.

## Unsupported native semantics

### Offroad

Pinned Waymax:

- attaches object-center Z to each of four box corners;
- uses only valid raw road-edge types `15` and `16`;
- chooses the nearest raw road-edge point using
  `dx**2 + dy**2 + (2*dz)**2`;
- uses raw direction vectors and, conditionally, the preceding raw point when source
  IDs agree **and** the predecessor cross product is less than the nearest-point cross
  product to determine signed side; and
- flags offroad iff any signed corner distance is strictly positive.

Its returned validity is true for every object slot, independent of trajectory
validity. With no eligible edge, nearest-index fallback can still produce an upstream
value marked valid. Any analysis would therefore need an explicitly separate
edge-availability and target-validity mask.

EvalSim `Scenario.map` keeps only 2-D grouped polylines and a broad `ROAD_EDGE` type.
It intentionally does not define drivable side and drops Z, direction, IDs, subtypes,
and raw ordering. A nearest 2-D edge distance cannot answer the same question and must
not be called offroad.

### Upstream `sdc_wrongway`

Despite its registry name, the pinned metric does not compare heading or lane
direction. Let `d` be minimum XY point distance to any valid SDC-path sample,
irrespective of `on_route`:

```text
valid = isfinite(d)
value = 0        when not valid or d < 3.5 m
value = d        otherwise
```

At exactly `3.5 m`, the value is `3.5`. Current SDC validity is not part of validity.
Any later equivalent must be named thresholded SDC-path-sample distance. EvalSim's
M5 lane-heading disagreement is a separate neutral diagnostic, not Waymax parity.

### SDC progression

At every frame, the pinned metric:

- selects the valid `on_route` path with the nearest point sample to the current valid
  simulated SDC;
- requires current simulated SDC validity and a finite distance to such a path;
- projects logged frame zero, logged final frame, and the current SDC independently
  to samples on that selected path;
- reads sample arc lengths; and
- returns `1` for an exactly zero start/end denominator, otherwise
  `(current - start) / (end - start)` without clipping.

Logged frame-zero and final-frame validity are not checked. Path and sample ties use
first array order. Invalid output is zero with `valid=false`.

The current contract lacks path samples, validity, `on_route`, arc length, and source
order. Also, M5 keeps ego logged/exogenous, so this SDC-only metric would be identical
among current policies and unsuitable for ranking.

### SDC off-route

Pinned Waymax computes current SDC distances `d_on` and `d_off` to valid on-route and
off-route samples and flags when:

```text
d_on > 5 m
OR
d_on - d_off > 2 m
```

Equality does not flag. A flag returns `d_on`; otherwise value is zero. No finite
on-route sample or an invalid current SDC yields zero with invalid output. Finite
`d_on` is required; absence of any off-route sample is allowed. These path tensors are
absent from the current contract and are deferred.

## EvalSim-only diagnostics

The M5 acceleration, jerk, yaw-rate, center-distance, capped constant-velocity TTC,
lane-center distance, lane-heading disagreement, continuity residual, and lifecycle
metrics are independent source-neutral definitions. They must be validated by
analytic oracles and invariance tests, not presented as Waymax-equivalent.

Map proximity is not offroad. Lane-heading disagreement is not wrong-way. The TTC
proxy is not a collision forecast. Neutral-direction diagnostics cannot produce a
favorable simulator claim.

## Parity execution scope

- **Population:** first 16 accepted-cohort events under the source-only domain
  `evalsim-m5-metric-parity-v1`, SHA-256 ranking, with exact M4 record encoding and
  shard/ordinal tie-break; ordered fingerprint recorded before metrics.
- **Window:** first 20 post-current transitions.
- **Policies:** EvalSim log replay, constant velocity, and EvalSim IDM.
- **State identity:** custom and native functions receive the same snapshot fields.
- **Comparison order:** masks and identities, discrete branches/flags, then continuous
  values, then aggregation.

For every parity frame, the native adapter starts from the exact reloaded source
`SimulatorState`, maps retained contract agents back to verified source slots, writes
canonical candidate motion/validity and contract dimensions into retained
`sim_trajectory` slots, preserves the source log trajectory, metadata, Z, height,
timestamps, roadgraph, and padding, sets the current timestep, and calls the native
metric class directly.

The official runner now implements the separate full 128-scene/80-transition Waymax
exact-log gate and exact 144-row parity-summary domain. The public mocked lifecycle and
data-free acceptance produced no native WOMD parity result. Any later observed 16 × 20
parity result is a bounded semantic cross-check, not a policy benchmark or a rerun of
Waymax IDM.
