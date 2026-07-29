# M5 — Real-WOMD metric system and statistical scorecards

**Date:** 2026-07-28
**Status:** Accepted pre-registration, data-free overlap-boundary amendment, and
data-free implementation; the implementation is committed and pushed, bound real-WOMD
acceptance remains pending, and no M5 WOMD outcomes or native M5 parity results have
been inspected
**Depends on:** accepted M4 execution snapshot
`a7a20e5de89c9c988f36a4b2f10ff4acc49246f0`
**Population:** the unchanged accepted M4 complete-case conditional cohort of
128 scenarios from exactly validation shards `00000`–`00009`

This document freezes M5 before implementation or outcome access. It supersedes the
less-specific M5 build and evidence bullets in the canonical roadmap. If the
implementation cannot meet a rule below, the run fails or the plan is versioned and
reviewed again before a fresh run; the rule is not silently relaxed after looking at
results.

**Accepted data-free amendment, 2026-07-29:** adversarial synthetic testing falsified universal
bit-equivalence between NumPy/libm and XLA for strict oriented-box overlap at
float32 zero-margin boundaries. No M5 WOMD outcome, ignored M4 artifact content, or
data-derived count was inspected. The universal claim is withdrawn before execution;
the frozen observed 16 × 20 × three-policy native flag comparison remains exact and
fails on any mismatch. The synthetic counterexample remains a permanent test rather
than being hidden or tolerance-normalized. Independent metric/statistics and
publication-claim reviewers accepted the bounded claim and permanent counterexample.

**Accepted data-free implementation closure, 2026-07-29:** the fixed five-scene
synthetic runner, thirteen metrics, eight source-only slices, paired finite-cohort
statistics, immutable result store, aggregate scorecard, and privacy-safe CLI passed
264 focused tests. The locked Waymo-extra suite passed 757 tests with one expected
local-data skip; the clean locked core-only suite passed 676 tests with 23 expected
optional-runtime skips. Independent architecture, methods/statistics, and
privacy/claim reviews returned **ACCEPT** after the terminal/finalization and
failure-record findings were corrected. The exact data-free matrix is 195 metric rows,
40 slice rows, 312 scorecard rows, and zero native Waymax parity rows. All 25
log-replay error oracles are exact zero across their eligible components. All 312
synthetic scorecards have paired N from zero to five, are `insufficient_n`, suppress
effects and bands, and forbid directional language.

This closure is implementation evidence, not real-WOMD result evidence. No real-WOMD
metric result, slice count, policy difference, resampling interval, or native M5
metric-parity result was inspected. The unchanged 128-scenario cohort, shared pinned
Waymax decode, and privileged logged-future status of log replay remain mandatory
limitations. The data-free implementation is accepted and pushed but has not been used
to inspect a real-WOMD M5 outcome. The reviewed implementation was committed as
`9b2676ac4b1c7bfb9f35a1c92f0159158756544a` before that boundary was opened.

No real-WOMD M5 metric result, slice count, policy difference, resampling interval,
native scenario identity, coordinate, or ignored M4 artifact **content** was inspected
while selecting these definitions. The existing tracked M4 aggregate and pinned
upstream source were used to establish feasibility.

## 1. Question, hypothesis, and claim boundary

**Question:** Can a contract-first set of independent motion metrics expose different
failure modes among three deterministic simulators on the frozen real-WOMD cohort,
while preserving component distributions, pairing, missingness, scene-reweighting
stability, and contradictory results?

**Exploratory expectation:** At least two metric families may distinguish the policies
in different ways. “Distinguish” is deliberately not a release criterion or a
hypothesis test in this fixed conditional cohort. No policy winner, effect sign, slice
prevalence, or minimum effect size is pre-registered. A null result, a sign reversal,
or an intuitively uncomfortable result is valid evidence and must remain visible.

M5 can support only this bounded claim after acceptance:

> Built a sliced motion-simulation evaluation framework on a locally auditable,
> complete-case conditional WOMD cohort, with paired scenario-resampling stability
> analysis and pinned Waymax semantic cross-checks.

M5 will not establish:

- a representative WOMD benchmark or a population-level Waymo conclusion;
- causal simulator superiority;
- production, accelerator, or fleet-scale performance;
- one composite realism score or total ordering of simulators;
- native Waymax offroad, directional wrong-way, route, signal, or condition adherence
  through the current EvalSim contract;
- camera, sensor, or video realism;
- a comparison between EvalSim IDM and Waymax IDM as numerical twins; or
- independence from the pinned Waymax WOMD decoder.

Log replay uses the recorded future and is a privileged construction oracle. Its zero
logged-reference error does not make it the best causal simulator.

## 2. Frozen inputs and execution matrix

### 2.1 Cohort reuse

M5 reuses the accepted M4 cohort exactly. It must not rescan the ten shards for
eligibility, rerank records, replace a failed record, or select a metric-friendly
subset.

The M5 loader will accept an explicit ignored M4 run directory and require:

- `aggregate-summary.json` reports accepted M4 completion;
- `execution-provenance.json` is canonical and binds `git_commit` to
  `a7a20e5de89c9c988f36a4b2f10ff4acc49246f0`; its `git_tree`, exact executable-file
  hashes, selector-config fingerprint, and reference-config fingerprint match the
  accepted M4 snapshot;
- `cohort/manifest-pass-1.json` and `cohort/manifest-pass-2.json` are canonical and
  byte-identical;
- manifest/event selector version, adapter/schema versions, dataset-config
  fingerprint, and pinned Waymax commit match the accepted M4 contracts;
- aggregate acceptance state and exact runtime versions match the tracked accepted M4
  evidence;
- exactly 128 selected events are present in the canonical M4 order;
- aggregate and manifest population counts agree; and
- every exact locator reloads with unchanged identity, shard digest, source predicate,
  adapter parity, and provenance.

The local M5 result tables use opaque `cohort_index` values `0..127` in canonical M4
order. Native scenario IDs and locators remain in the ignored M4 manifests and
transient memory. The M5 manifest binds the M4 provenance SHA-256, canonical manifest
SHA-256, and canonical selected-order fingerprint.

### 2.2 Policy paths

All three EvalSim policies run with seed `0` over all 128 scenarios and all 80
post-current transitions:

1. `log_replay`;
2. `constant_velocity`; and
3. `idm`.

The full-horizon pinned Waymax exact-log path also runs over all 128 scenarios and is
converted to the same `Rollout` contract. It is a reference execution and mapping
oracle, not a fourth causal policy.

M5 defines a new source-only parity subset before metric execution. Rank all 128
accepted events using the existing M4 `rank_record` encoding with domain
`evalsim-m5-metric-parity-v1`: strict ASCII domain, NUL, five-byte ASCII shard suffix,
NUL, unsigned uint64 big-endian record ordinal, NUL, strict UTF-8 native scenario ID;
SHA-256 digest; ties by shard suffix then record ordinal. The first 16 form the ordered
subset, and an ordered-membership fingerprint is written locally before any policy
metric runs. Its first 20 post-current transitions form the parity window.

This subset is not the unpersisted M4 IDM subset. M5 does **not** rerun or rank
Waymax's privileged logged-trajectory IDM. M5 parity applies pinned native metric
functions to the same canonical candidate snapshots used by the custom implementation.

### 2.3 Target and time scope

- `current_index` is the final observed/current frame.
- Metric outcomes use frames `current_index + 1` through the final frame.
- A derivative at the first simulated frame may use `current_index` as its predecessor;
  the predecessor is context, not a scored component.
- Two-step derivatives may use one additional preceding frame only when the complete
  contiguous valid sequence exists.
- Primary target objects are all non-ego agents except that the bicycle-model
  kinematic-infeasibility scorecard is restricted to non-ego vehicles. The
  logged/exogenous ego is excluded as a scored target so it cannot dilute policy
  effects.
- Ego and other valid objects may still be interaction counterparts.
- Agent/frame eligibility is derived from the source `Scenario`, identically for every
  policy. A rollout validity or identity mismatch is an execution defect, not a reason
  for pairwise deletion.
- Derivatives never bridge an invalid frame, birth, death, or re-entry.
- Actual positive scenario timestamp differences are used except for the explicitly
  pinned Waymax kinematic-infeasibility definition, which uses its upstream fixed
  `0.1 s` step. The official run requires every scored contiguous source transition
  to have an exact reloaded `timestamp_micros` delta of `100_000`; drift is fatal.

### 2.4 Canonical numeric representation

The three Waymax parity anchors—position error, overlap, and kinematic
infeasibility—use one canonical float32 view in both production scorecards and parity:

- cast candidate and logged `x`, `y`, `heading`, `vx`, and `vy`, plus candidate
  `length` and `width`, once with NumPy `float32`;
- preserve boolean masks and integer identity exactly;
- use frame-major then contract agent order for components; and
- aggregate finite component values with `math.fsum` after conversion to Python
  `float`, divided by the exact integer denominator.

The custom functions and the pinned Waymax adapter receive copies of those exact
float32 bits. Other EvalSim-only diagnostics use the contract's float64 values.
This prevents input quantization from being mistaken for metric disagreement, but it
does not imply universal backend bit-equivalence: NumPy/libm and XLA trigonometric
rounding can flip a strict overlap decision at an adversarial zero-margin boundary.
M5 retains that counterexample, describes the scorecard implementation as the
source-neutral NumPy definition, and requires exact observed native flags throughout
the frozen parity subset.

## 3. Metric contract

`MetricSpec` becomes a validated immutable declaration with:

- name and semantic version;
- value unit;
- statistical unit of analysis;
- direction: `higher`, `lower`, or `neutral`;
- aggregation;
- agent scope and evaluation window;
- human-readable eligibility;
- registered invalid-reason codes;
- required fields and dependencies; and
- determinism and known failure modes.

`MetricResult` becomes a validated immutable per-scenario result with:

- metric and scenario identity;
- finite scalar value or `None`;
- retained finite component distribution;
- validity and exactly one registered invalid reason when invalid;
- eligible and total component counts; and
- a recursively frozen JSON-compatible detail mapping.

The following implications are mandatory:

- valid result ⇒ finite non-null value, no invalid reason, and at least one eligible
  component;
- invalid result ⇒ null value and a registered invalid reason, never a NaN sentinel;
- result identity and version must match the input and registered metric;
- every retained component must be finite;
- metric registration and evaluation order are deterministic; and
- duplicate names or name/version ambiguity fail closed.

`Metric.eligibility()` returns a typed eligibility/reason value; a Boolean-only
`validate_inputs()` is retired. `Metric.compute()` produces only a per-scenario result.
The cross-scenario `Metric.aggregate()` seam is removed: the central paired-statistics
module is the sole owner of cross-scenario aggregation and may apply only the reducers
declared by `MetricSpec`.

The existing single-simulator `RunManifest` remains backward-compatible for earlier
milestones. M5 introduces a separately validated multi-execution
`EvaluationManifest`. No metric is allowed to convert an execution error into ordinary
missingness.

## 4. Frozen metric definitions

All per-scenario means give each eligible component equal weight inside the scene.
Statistics first reduce to exactly one scalar per scenario, so scenes—not agents or
frames—are the summary and resampling units.

### 4.1 Primary family

The primary family is exactly four metrics by three ordered policy contrasts, for
12 cells.

#### `position_error_m` version `1.0.0`

- **Unit/direction:** metres; lower.
- **Components:** for each source-valid non-ego target and future frame,
  `hypot(x_sim - x_log, y_sim - y_log)`.
- **Scenario scalar:** arithmetic mean of components.
- **Expected eligibility:** unconditional for all 128 M4 scenarios.
- **Waymax crosswalk:** equivalent to upstream `log_divergence` after target/window
  restriction.

#### `speed_error_mps` version `1.0.0`

- **Unit/direction:** metres/second; lower.
- **Components:** absolute difference between simulated and logged
  `hypot(vx, vy)` for each source-valid non-ego target and future frame.
- **Scenario scalar:** arithmetic mean.
- **Expected eligibility:** unconditional for all 128 M4 scenarios.
- **Waymax crosswalk:** no pinned native equivalent.

#### `oriented_box_overlap_rate` version `1.0.0`

- **Unit/direction:** fraction in `[0, 1]`; lower.
- **Components:** one binary flag for each source-valid non-ego target and future
  frame. It is one iff the target's oriented `[x, y, length, width, heading]` box has
  strictly positive separating-axis overlap with at least one other source-valid
  object. Self-pairs are removed; exact edge touching is not overlap.
- **Scenario scalar:** flagged target-frame count divided by eligible target-frame
  count.
- **Interpretation:** overlapping-target-frame rate, not collision-pair count,
  geometric occupancy, or collision severity.
- **Expected eligibility:** unconditional for all 128 M4 scenarios.
- **Waymax crosswalk:** same intended strict separating-axis semantics after
  target/window restriction, with exact observed flag comparison on the frozen parity
  subset. Universal bit-equivalence at float32 zero-margin boundaries is explicitly
  falsified and not claimed.

#### `waymax_kinematic_infeasibility_rate` version `1.0.0`

- **Unit/direction:** fraction in `[0, 1]`; lower.
- **Components:** one binary flag for every contiguous-valid non-ego **vehicle**
  transition in the future window, using the pinned Waymax inverse bicycle formula.
  All object types are retained in component-level native parity, but pedestrian and
  cyclist bicycle-model flags are not promoted into the scorecard:
  - fixed `dt = 0.1 s`;
  - `speed = hypot(vx, vy)`;
  - `accel = (new_speed - speed) / dt`;
  - `old_yaw` is the candidate rollout heading at `t-1`;
  - new yaw is velocity-derived `atan2(new_vy, new_vx)` only when
    `abs(new_speed) > 0.6 m/s`; otherwise `new_yaw` is the candidate rollout
    heading at `t`;
  - wrapped `delta_yaw` is divided by
    `speed * dt + 0.5 * accel * dt**2`;
  - steering curvature is forced to zero when either old or new speed is strictly
    below `0.6 m/s`;
  - infeasible is `abs(accel) > 10.4 + 1e-3` or
    `abs(steering_curvature) > 0.3 + 1e-3`.
- **Scenario scalar:** infeasible transition count divided by eligible transition
  count.
- **Expected eligibility:** unconditional for all 128 M4 scenarios; cadence drift is
  fatal.
- **Waymax crosswalk:** intended to be component-equivalent to pinned
  `KinematicsInfeasibilityMetric`.

### 4.2 Secondary diagnostic family

These metrics remain visible but cannot independently unlock a primary policy-quality
claim.

#### Logged-reference derivative errors

- `acceleration_error_mps2` `1.0.0`: Euclidean norm of the difference between
  simulated and logged vector accelerations
  `(v[t] - v[t-1]) / dt`; mean; lower.
- `jerk_error_mps3` `1.0.0`: Euclidean norm of the difference between simulated and
  logged vector jerk. Acceleration samples are placed at interval midpoints, so the
  jerk denominator is `(dt_previous + dt_current) / 2`; mean; lower.
- `yaw_rate_error_radps` `1.0.0`: absolute difference between simulated and logged
  wrapped heading increments divided by `dt`; mean; lower.

Each requires the corresponding contiguous source-valid window. Zero eligible
components is explicit `no_contiguous_valid_window`.

#### Interaction diagnostics

- `minimum_center_distance_m` `1.0.0`: for each future frame, retain the minimum
  Euclidean center distance across unordered valid-object pairs with at least one
  non-ego object; the scenario scalar is the minimum of those frame minima; metres;
  neutral. It is not box clearance. Record total evaluated pair count separately.
- `constant_velocity_ttc_cap_5s` `1.0.0`: minimum right-censored constant-velocity
  disc-contact time across the same pairs; retain one frame minimum and take the
  scenario minimum; seconds; neutral. Record total evaluated pair count separately.
  - Object radius is half its box diagonal.
  - Existing disc overlap yields `0`.
  - Otherwise solve the standard relative-motion quadratic for the first nonnegative
    contact root.
  - No root, a separating pair, or a root above five seconds yields the cap `5`.
  - This is an explicitly conservative geometric proxy, not a lane-aware collision
    forecast.

#### Map diagnostics

- `lane_center_distance_m` `1.0.0`: mean minimum 2-D point-to-segment distance from
  each valid non-ego vehicle center to retained `MapType.LANE` segments; metres;
  neutral.
- `lane_heading_disagreement_rad` `1.0.0`: mean absolute wrapped heading difference
  to the tangent of the nearest retained lane segment; radians in `[0, pi]`; neutral.
  Exact distance ties choose the first map/segment in contract order.

These are conditional on at least one valid retained lane segment and one eligible
vehicle-frame. The explicit invalid reasons are `no_supported_lane` and
`no_eligible_vehicle_frame`. Neither metric is called offroad or wrong-way.

#### Temporal diagnostics

- `kinematic_continuity_residual_m` `1.0.0`: mean
  `norm((p[t]-p[t-1]) - 0.5*(v[t]+v[t-1])*dt)` over contiguous-valid non-ego future
  transitions; metres; lower.
- `lifecycle_reentry_per_agent` `1.0.0`: number of future invalid-to-valid re-entries
  after an object has already been valid at or after the current frame, divided by the
  number of non-ego agents; retain one re-entry count per agent; events/agent; neutral.

The lifecycle metric should be identical across current policies because the engine
owns the source validity mask. A difference is a contract failure, not a finding.

### 4.3 Unsupported names

M5 must not implement a custom metric named `offroad`, `wrong_way`,
`route_progress`, `off_route`, `signal_adherence`, or `condition_adherence`.
The current `Scenario` contract lacks the fields required for equivalent semantics:
agent Z, road-edge Z/direction/source IDs/subtypes/raw order, SDC path samples,
validity, `on_route`, arc length, traffic-light timelines, and route conditions.

Adding a generic 2-D approximation under an upstream name is an acceptance failure.
Typed path and map-evaluation sidecars may be designed in M6 when ego becomes
counterfactual, but M5 does not hide source tensors in `Scenario.metadata`.

## 5. Waymax metric parity

The pinned source definitions and bounded semantic crosswalk live in
`docs/data/womd-waymax-m5-metric-crosswalk.md`.

For the frozen 16-scene × 20-transition M5 parity subset, the runner compares custom
and upstream components **before aggregation** for log replay, constant velocity, and
EvalSim IDM.

For each candidate and frame, the native adapter:

- starts from the exact reloaded source `SimulatorState`;
- maps every retained EvalSim agent back to its verified source slot;
- writes the canonical float32 candidate `x`, `y`, `yaw`, `vel_x`, `vel_y`, validity,
  and contract scalar `length`/`width` into retained `sim_trajectory` slots;
- preserves source `log_trajectory`, object metadata, Z, height, timestamps,
  roadgraph, and all padding slots;
- sets the exact current timestep;
- calls the pinned native metric class directly; and
- fails on any identity drift or field that cannot be represented.

This is the pinned Waymax metric implementation applied to the EvalSim contract view;
it is not a claim that contract scalar dimensions reproduce every raw source dimension
sample.

Acceptance rules:

- identity/order and validity masks are exact;
- overlap validity masks are exact everywhere and discrete flags are exact only where
  the target mask is valid; invalid-target raw values are ignored rather than
  normalized and claimed as parity; this is a bounded observed parity gate, not a
  universal NumPy/XLA bit-equivalence claim;
- kinematic-infeasibility masks and binary flags are exact;
- jointly valid finite continuous log-divergence values satisfy
  `abs(custom - reference) <= max(1e-6, 8 * float32_ulp(abs(reference)))`
  with `rtol = 0`;
- a non-finite valid value is fatal; and
- tolerance cannot excuse a different object, mask, threshold branch, or state
  snapshot.

For a finite float32 `x`,
`float32_ulp(x) = float(nextafter(float32(abs(x)), float32(+inf)) -
float32(abs(x)))`, with the `nextafter` operation itself performed in float32.

The full-cohort Waymax exact-log execution remains required even though the numerical
parity subset is bounded.

## 6. Source-only slices

Slice version is `m5-womd-slices-1.0.0`. Membership is computed from `Scenario` before
any rollout or metric result exists. A slice implementation never receives a
`Rollout`, simulator name, metric value, or observed policy difference.

No slice intersections are evaluated in M5. All slices are secondary/exploratory.

1. `all`: every scenario.
2. `vru_present_current`: at least one current-valid non-ego pedestrian or cyclist.
3. `current_world_count_ge_8`: at least eight current-valid non-ego agents.
4. `retained_world_count_ge_16`: at least 16 non-ego agents valid at any contract
   frame. This is a retained-count proxy, not WOMD's modeled-object role.
5. `observed_ego_turn_ge_15deg`: absolute wrapped ego-heading change between the first
   and last valid frames in the inclusive window
   `[max(0, current_index - 9), current_index]` is at least `pi/12`. Fewer than two
   valid ego frames yields `eligible=false`, `member=false`, and reason
   `insufficient_observed_ego_heading`.
6. `low_current_cv_ttc_le_3s`: evaluate only ego-versus-each-current-valid-non-ego
   pair at the current source frame in float64. Let relative position and velocity be
   `r` and `v`, combined half-diagonal radius be `R`, and
   `a = dot(v,v)`, `b = 2*dot(r,v)`, `c = dot(r,r)-R**2`.
   - `c <= 0` gives TTC zero.
   - Otherwise `a == 0` or discriminant `b**2-4*a*c < 0` gives the five-second cap.
   - Otherwise choose the smallest nonnegative quadratic root; no nonnegative root
     gives the cap; a root above five is capped at five.
   - No numerical epsilon is applied and the slice is a member iff the minimum capped
     TTC is `<= 3.0`.
   - No eligible current counterpart yields `eligible=false`, `member=false`, and
     reason `no_current_counterpart`.
7. `future_lifecycle_change`: at least one non-ego source validity bit changes after
   `current_index`, including the adjacent `current_index → current_index + 1`
   comparison. This conditions on the logged future and is not a deploy-time
   observable context slice.
8. `supported_lane_available`: at least one retained lane contains a finite nonzero
   segment.

Traffic-light, route, condition, true offroad, and directional wrong-way slices are
deferred because the source-neutral contract does not expose their required fields.

## 7. Paired statistical design

### 7.1 Estimands and contrasts

The ordered raw contrasts are:

1. `constant_velocity - log_replay`;
2. `idm - log_replay`; and
3. `idm - constant_velocity`.

For scene `i`, `d_i = value_A_i - value_B_i`. The exact finite-cohort estimand is
`mean(d)` across every paired member of the accepted cohort or declared source-only
slice. Because this fixed conditional cohort is fully observed, its mean has no
sampling error. The reported resampling bands below describe sensitivity to empirical
scene reweighting; they are not confidence intervals for WOMD or another population.

Supplemental outputs are:

- median paired difference;
- favorable proportion `(wins + 0.5 * ties) / n` for directional metrics, with a tie
  defined as exact equality of the stored finite scenario scalars;
- paired standardized signal-to-heterogeneity ratio
  `mean(oriented_d) / sample_sd(oriented_d, ddof=1)` only when `n >= 30`, at least ten
  effects are nonzero, and the sample standard deviation is finite and positive; and
- baseline and candidate scene-level mean/median.

Raw signs are always retained. An oriented advantage, when defined, is a separate
field whose positive sign means favorable according to the registered direction.
Neutral metrics have no favorable proportion or oriented standardized effect.

### 7.2 Missingness and pairing

- No imputation.
- A no-event condition is represented by explicit zero only when zero is the
  pre-registered meaningful observed value.
- A zero denominator is missing with a registered source reason.
- Non-finite values, shape drift, policy failure, or output mismatch are defects.
- Every cell reports cohort N, valid A, valid B, paired N, excluded N, component
  denominators, asymmetric missingness, and counts by reason.
- Pairing uses the same source-derived scenario and component eligibility for A and B.
- The four unconditional primary metrics require paired `n = 128` and zero unexpected
  invalids for every contrast.
- Conditional metrics may have smaller N only for pre-registered source reasons.
  Policy-dependent eligibility or silent pairwise deletion fails acceptance.

### 7.3 Deterministic scenario-resampling stability bands

- Resamples: `B = 100_000` for the 12 primary `all`-slice cells and `B = 10_000` for
  every other cell.
- Interval: two-sided percentile.
- Pointwise stability level: `95%`.
- Quantiles: NumPy `quantile(..., method="linear")`.
- Base seed: `20260728`.
- RNG: NumPy `PCG64` through `SeedSequence`.
- Resampling unit: scenario row; the entire paired record and nested component
  summaries travel together.
- Canonical input order: frozen `cohort_index`.

Each cell derives an independent deterministic substream as follows:

1. Serialize a mapping containing statistics schema version, metric name/version,
   slice name/version, ordered policy pair, paired N, and B as UTF-8 JSON using sorted
   keys, compact separators `(",", ":")`, `ensure_ascii=True`, and
   `allow_nan=False`.
2. Compute SHA-256 and decode the 32 digest bytes as eight unsigned big-endian uint32
   words.
3. Construct `SeedSequence([20260728, *digest_words])`, then `PCG64`.
4. Generate exactly
   `rng.integers(0, n, size=(B, n), dtype=np.int64)`.
5. Use that one draw-index matrix for every estimator and percentile band in the
   cell.

Python `hash()` and native scenario IDs are forbidden as seed material.

The local evaluation manifest records NumPy version, RNG, base seed, B, stability
level, quantile method, canonical key bytes, digest words, and index dtype. Identical
inputs must produce byte-identical scorecard tables.

### 7.4 Multiplicity and small slices

The 12-cell primary family receives:

- required pointwise 95% scenario-resampling stability bands; and
- simultaneous Bonferroni-adjusted stability bands at marginal percentile level
  `1 - 0.05 / 12 = 0.9958333333333333`.

Exclusion of zero from the adjusted band means the finite-cohort sign is robust to
the pre-registered empirical scene-reweighting procedure. It is not a hypothesis test,
familywise population confidence statement, or WOMD-wide directional claim. Fewer
than ten nonzero paired effects forces the label `event_sparse` and forbids even this
bounded directional language.

Every secondary metric or non-`all` slice cell is exploratory:

- paired `n >= 30` and at least ten nonzero paired effects: retain exact effect and
  pointwise stability band;
- paired `n = 10..29` or fewer than ten nonzero effects: retain exact effect and
  pointwise stability band, label `small_or_sparse`, and forbid directional language;
- paired `n < 10`: retain counts/reasons only and suppress effect and band.

M5 computes no p-values, q-values, discovery labels, or exploratory significance
claims. This avoids pretending that the nonrandom conditional cohort supports a
population null test. Multiplicity is made visible by emitting the complete fixed
13-metric × 8-slice × 3-contrast matrix, using the adjusted primary stability bands,
and treating every remaining cell as descriptive.

## 8. Local result store and publication boundary

M5 writes only beneath a new ignored `outputs/m5/<run-name>/` directory. It never
overwrites or deletes an earlier run.

Typed immutable outputs:

- final `evaluation-manifest.json`: accepted M4 provenance/manifest hashes,
  selected-order version, code commit/tree, simulator specifications,
  metric/slice/stat fingerprints, runtime versions, file hashes, and `complete=true`;
- partitioned `metric-results/*.parquet`: one row per
  cohort-index × execution × seed × metric, with
  `execution_role = policy | reference`;
- `slice-membership.parquet`: one row per cohort-index × slice;
- `scorecards.parquet`: one row per metric × slice × ordered pair;
- `waymax-parity-summary.parquet`: one row per
  parity-scenario × EvalSim policy × parity metric;
- local detailed diagnostics for Waymax components and explicit failures; and
- a local human-readable scorecard.

The exact finalized matrix is:

- 13 metrics × 4 executions × 128 scenarios = **6,656 metric rows**;
- 8 slices × 128 scenarios = **1,024 slice rows**;
- 13 metrics × 8 slices × 3 EvalSim contrasts = **312 scorecard rows**; and
- 16 parity scenarios × 3 EvalSim policies × 3 parity metrics =
  **144 parity-summary rows**.

The four executions are the three policy paths plus the Waymax exact-log reference.
Only the three `execution_role=policy` paths enter contrasts.

The writer uses exclusive creation, fixed PyArrow schemas, and unique keys. It first
writes immutable parts beneath a pending subdirectory, validates and fsyncs them,
hashes every artifact except the not-yet-created final manifest, then exclusively
creates `evaluation-manifest.json` with `complete=true` and a `SUCCESS` marker.
The final manifest never hashes itself. A failed run keeps pending artifacts and gains
an exclusive failure record; it is never converted into success in place.

Scorecards cannot be generated until metric and slice row accounting passes, and final
success cannot be created until every expected row and part hash passes. DuckDB reads
Parquet directly; no mutable database is needed.

Detailed local diagnostics may contain derived values and may be inspected under the
two-tier `AGENTS.md` rule. Dataset payloads are never copied into outputs.

Git, deployment, and chat publication may include only deliberately promoted,
sanitized aggregates with no native IDs, locators, shard digests, coordinates,
per-scene rows, component distributions, private paths, or raw/generated dataset
payload. Generated run tables remain ignored and untracked.

### 8.1 Terminal boundary

The official M5 command treats stdout/stderr as a publication boundary because a
Codex-run terminal transcript can enter chat:

- success stdout is one canonical allowlisted status object containing only fixed
  schema/status labels, aggregate row counts, elapsed stage labels, and the
  project-relative ignored result path;
- failure stderr is one fixed allowlisted reason code plus a project-relative ignored
  failure-record path;
- no exception representation, traceback, native ID, locator, digest, coordinate,
  per-component value, per-scene result, or absolute path is written to either stream;
  and
- detailed progress, diagnostics, and tracebacks remain useful but go exclusively to
  ignored local log/failure files.

This is not a general zero-terminal-output rule. It is the pre-registered contract for
the one official data-backed CLI. Success and injected failure tests capture both
streams and scan them against forbidden local sentinel values and forbidden field
names.

### 8.2 Frozen promoted-evidence schema

If M5 passes, the tracked documentation and owner-only presentation must publish the
complete 12-cell primary `all`-slice matrix—never a favorable subset. Every cell
contains:

- metric name/version/unit/direction and ordered A/B policy names;
- exact finite-cohort paired effect and paired N;
- valid A, valid B, excluded, asymmetric-missing, and reason counts;
- pointwise and adjusted scenario-reweighting stability bands;
- nonzero-effect count plus `event_sparse`/suppression flags; and
- raw sign, with oriented sign shown only as a separate field.

The promoted evidence must also include:

- all three parity-anchor names, exact 16 × 20 × three-policy scope, compared-component
  counts, and mismatch counts;
- all null or contradictory primary cells;
- the complete cohort construction label and shared-decoder limitation;
- the privileged recorded-future limitation of log replay;
- the fixed-cohort/stability-not-population interpretation; and
- unsupported offroad/route/signal semantics and neutral-diagnostic caveats.

No exploratory numeric cell may be selectively promoted. Public documentation either
omits exploratory slice numbers entirely while reporting their pre-registered
availability/count limitations, or publishes one complete sanitized 312-cell
scorecard appendix under the same schema. Generated Parquet/JSON run tables are never
used as tracked publication artifacts.

## 9. Synthetic and data-free acceptance

Before any M5 WOMD execution, tests must prove:

- exact log replay yields zero position, speed, acceleration, jerk, and yaw-rate error;
- known constant speed/acceleration, nonuniform timestamp, and wrapped-yaw examples;
- derivatives never cross invalid gaps, births, deaths, or re-entries;
- exact box separation, edge touch, infinitesimal penetration, invalid target, and
  invalid counterpart behavior;
- a retained rotated zero-margin float32 counterexample demonstrating that
  NumPy/libm and XLA overlap flags are not universally bit-equivalent, plus exact
  pinned-native checks for ordinary and threshold fixtures;
- pinned Waymax kinematic boundaries using each exact float32 threshold and float32
  `nextafter` immediately below and above `0.6`, `10.401`, and `0.301`, preserving
  new-yaw `<= 0.6`, suppression `< 0.6`, and infeasibility strict `>`;
- known center distance and TTC roots, separating motion, existing overlap, and
  five-second censoring;
- lane point-to-segment projection, canonical tie-breaking, translation/rotation
  invariance, and no-lane ineligibility;
- lifecycle re-entry counting;
- agent-order, padding, rigid-transform, and serialization invariance where applicable;
- registry duplicate, identity, version, finite/null, invalid-reason, and component
  accounting failures;
- slice threshold boundaries and invariance to policy/result order;
- Parquet schema, unique keys, part hashes, round trip, interrupted finalization, and
  ignored-output containment;
- official-CLI success and injected failures capture stdout/stderr, preserve detailed
  diagnostics in ignored files, and reject native-ID/locator/digest/coordinate/
  per-component/absolute-path sentinels from both terminal streams;
- identical pairs give effect and stability band `[0, 0]`;
- a constant paired delta gives an exact constant stability band;
- frames/agents duplicated inside a scene do not increase statistical N;
- row permutation canonicalizes to byte-identical statistics;
- substream keys are stable and distinct;
- empty, singleton, all-zero, sparse, zero-SD, missing, and contradictory cases;
- small-slice thresholds at `9/10/29/30` and nonzero thresholds at `9/10`;
- exact resampling-key serialization, digest word order, index draws, and adjusted
  primary percentile levels; and
- no output schema or renderer contains a composite realism score.

The existing full suite must continue to pass in both Waymo-extra and core-only
environments. Optional Waymax imports remain lazy.

## 10. Bound real-WOMD acceptance

The first official M5 run occurs only after:

1. this pre-registration and adversarial reviews are accepted, committed, and pushed;
2. the data-free implementation and tests are accepted, committed, and pushed;
3. the worktree is clean;
4. all ten immutable shards are present;
5. the accepted ignored M4 run passes reuse validation; and
6. the command binds its executable-source fingerprint to the pushed commit.

The fingerprint enumerates, rejects symlinks in, lexically sorts, length-prefixes, and
SHA-256 hashes every tracked regular file under these fixed roots plus the named
top-level files:

- `evalsim/contracts/`, `evalsim/metrics/`, `evalsim/results/`, `evalsim/slices/`,
  `evalsim/stats/`, and `evalsim/report/`;
- `evalsim/rollout/`, `evalsim/simulators/`, and `evalsim/sources/`;
- `tests/`;
- this plan and the M5 crosswalk; and
- `pyproject.toml`, `uv.lock`, `NOTICE.md`, and `AGENTS.md`.

The manifest records both the ordered path list and digest. An untracked executable
file beneath a listed root, a dirty listed path, or a path outside the worktree fails
the bound run.

### 10.1 Release-leakage audit

Before any M5 evidence commit, push, package, or deployment, automation must inspect:

1. `git ls-files` and the exact staged file list;
2. an archive built from the exact candidate commit, not from the working directory;
3. rebuilt wheel and sdist member lists and extracted text;
4. the exact Sites deployment source bundle derived from that same tracked commit; and
5. the post-deploy site, access mode, and required notice text.

Every surface rejects:

- roots `data/`, `outputs/`, `private/`, `.venv/`, caches, and local environment files;
- TFRecord, Parquet, Arrow, NumPy array/archive, pickle, checkpoint, database, and
  generated-run formats;
- native IDs, exact locators, shard digests, absolute local paths, per-scene rows,
  per-component values, and locally derived forbidden sentinels loaded from the
  ignored manifests solely for the scan;
- unmodified/vendored Waymax source, documentation, wheel, or cache; and
- a missing or altered required WOMD attribution, Waymax prescribed notice/citation,
  canonical license link, or non-commercial limitation.

Ignored workspace contents are never deployment input. The source bundle must be
constructed from the exact clean tracked commit, its member hashes recorded, and its
commit verified against GitHub `main`. After deployment, the existing site must remain
owner-only with the same access controls; M5 does not change access mode or users.

The run passes only if:

- exactly the accepted 128 M4 scenarios reload; no rescan, reselection, replacement,
  or silent drop occurs;
- log replay, constant velocity, EvalSim IDM, and full-horizon Waymax exact log complete
  their declared scopes;
- every expected primary result is finite and paired for all 128 scenes;
- every other expected result is valid or has one pre-registered source-only reason;
- component, result-row, slice-row, pair-row, and file-hash accounting is exact;
- the 16 × 20 × three-policy Waymax metric-parity matrix meets the frozen mask,
  discrete, and float rules;
- repeated metric/statistical execution is byte-identical;
- pointwise and adjusted primary stability bands, the complete descriptive matrix,
  missingness, nulls, and contradictions are all retained;
- no composite score or post-result threshold/slice change appears;
- local output containment and Git visibility checks pass; and
- adversarial semantic, numerical/statistical, and privacy/claim reviews accept the
  completed evidence.

A failed run is preserved locally. Diagnostics may be inspected. Any code or semantic
correction requires a new metric/run version, a reviewed committed snapshot, and a new
output directory. Previous failures are not rewritten or erased.

## 11. Falsification conditions

M5 is falsified or blocked from release by any of:

- broken scenario pairing or frame/agent resampling;
- unexpected invalid or asymmetric eligibility in a primary metric;
- a source mask or identity mismatch treated as missingness;
- a failed synthetic analytic oracle;
- a claimed-equivalent Waymax/custom component mismatch beyond the frozen rule;
- a 2-D approximation labeled native offroad, wrong-way, or route adherence;
- data-dependent threshold, slice, metric, or cohort revision without a new
  pre-registration;
- non-reproducible scenario-resampling stability bands or result store;
- omitted null, contradictory, failed, or small-slice result;
- a headline composite realism score;
- native IDs, per-scene results, data, or generated run artifacts staged for Git or
  deployment; or
- a public claim that omits the conditional-cohort, shared-decode, deterministic-policy,
  or privileged-log-replay limitations.

## 12. Implementation sequence and rollback

1. Finalize this plan and the pinned Waymax crosswalk.
2. Obtain independent architecture, metric-semantics, statistics, and publication
   reviews; revise until no blocker remains.
3. Commit and push the accepted pre-registration.
4. Strengthen metric contracts and implement registry plus pure metric families.
5. Implement source-only slices and deterministic paired statistics.
6. Implement the immutable local result store and scorecard renderer.
7. Implement accepted-M4 cohort reuse and the optional Waymax metric adapter.
8. Implement the M5 CLI and data-free end-to-end synthetic path.
9. Run targeted, full Waymo-extra, core-only, package, and site checks; adversarially
   review the implementation.
10. Commit and push the data-free implementation.
11. Run one new bound ignored real-WOMD acceptance directory.
12. Inspect local diagnostics and obtain three independent result reviews.
13. Promote only accepted aggregate evidence into the README, roadmap, crosswalk,
    interview claim ledger, limitations, and owner-only presentation.
14. Audit tracked/staged/archive contents, commit, push, deploy, and verify the remote
    and owner-only release.

Rollback is additive: disable or revert M5 code while preserving prior commits and
ignored evidence directories. Raw data and accepted M4 outputs are never modified or
deleted.

## 13. Pre-implementation and data-free closure checklist

Four independent adversarial review tracks—architecture/feasibility, pinned Waymax
semantics, finite-cohort statistics, and privacy/publication claims—initially rejected
the draft with actionable findings. After revision, all four returned **ACCEPT** with
no remaining blocker or major issue. Material corrections included the canonical
float32 parity bridge, a new result-independent M5 parity subset, exact finite-cohort
estimands and scene-reweighting language, deterministic resampling bytes, typed
missingness, terminal sanitization, complete promoted-primary evidence, and exact
release-surface audits.

- [x] Existing implementation and contract seams inspected.
- [x] Existing full baseline suite: 493 passed, one expected local-data skip.
- [x] Required ten-shard directory and optional runtime are present.
- [x] No M5 WOMD outcome or ignored M4 artifact content inspected during metric
      selection.
- [x] Independent architecture, Waymax-semantics, and statistical design analyses
      completed.
- [x] Pinned Waymax crosswalk complete and adversarially accepted.
- [x] Four adversarial plan reviews accepted with no unresolved blocker or major issue.
- [x] Accepted pre-registration committed and pushed before M5 implementation
      (`5a52203cdbb5b055d2a152aeddd114ab09083eb4`).
- [x] Data-free metrics, slices, statistics, immutable store, report, source-neutral
      runner, and synthetic CLI implemented and accepted.
- [x] Data-free verification completed: 264 focused tests passed; the locked Waymo-extra
      suite passed 757 tests with one expected local-data skip; the clean locked
      core-only suite passed 676 tests with 23 expected optional-runtime skips.
- [x] Final independent architecture, methods/statistics, and privacy/claim reviews
      accepted the data-free implementation with no remaining blocker.
- [x] Accepted data-free implementation committed and pushed before WOMD execution
      (`9b2676ac4b1c7bfb9f35a1c92f0159158756544a`).
- [ ] Bound real-WOMD M5 execution, native metric parity, and result reviews completed.
