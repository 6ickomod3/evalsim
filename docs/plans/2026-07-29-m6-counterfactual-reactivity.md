# M6 — Counterfactual closed-loop reactivity

**Date:** 2026-07-29
**Status:** Accepted outcome-blind pre-registration. Independent architecture,
methods/statistics, and privacy/claim reviews returned no P1/P2 finding. Implementation
may begin; no M6 WOMD eligibility or policy outcome has been opened.
**Depends on:** accepted M4 cohort snapshot
`a7a20e5de89c9c988f36a4b2f10ff4acc49246f0` and accepted M5 closure
`a5afdb575d32f9e6342b0cb76407f6193c02d873`
**Source population:** the unchanged accepted M4 complete-case conditional cohort of
128 scenarios from exactly WOMD validation shards `00000`–`00009`
**Primary horizon:** 40 post-current transitions
**Primary intervention:** source-templated longitudinal brake pulse with positive
deceleration magnitude `b = 2.0 m/s²` for `1.0 s`, beginning on the first simulated
transition
**Primary matrix:** 1 intervention × 3 NumPy policy roles × 4 paired measures =
12 cells

This document freezes M6 before implementation or M6 outcome access. It narrows the
canonical roadmap into one rigorous two-dose braking vertical slice and explicitly
defers the larger intervention library. If implementation cannot meet a rule below,
the run fails or this plan is versioned and reviewed again before a fresh execution. A
rule is never silently relaxed after looking at a policy outcome.

## 0. Outcome-access ledger and preflight

Before this draft:

- no M6 WOMD eligibility count, trajectory, policy response, paired measure, or
  per-scene result was inspected;
- no M5 per-scene table or ignored result payload was used to choose M6 definitions;
- the tracked M4/M5 aggregate evidence, public upstream source, current contracts, and
  synthetic fixtures were inspected for feasibility;
- local `main` was clean at
  `a5afdb575d32f9e6342b0cb76407f6193c02d873` and matched `origin/main`;
- the accepted ignored M4 and M5 directories were present;
- exact shards `00000`–`00009` were present; an additional local shard `00010` is
  outside this plan and must not be read by M6;
- the pinned local NumPy/JAX/TensorFlow/Waymax environment was available; and
- the full Waymo-enabled baseline passed with **876 tests passed and 1 expected skip**
  in `213.71 s`.

The three read-only pre-plan audits inspected no local dataset payload or per-scene
result. They found that M6 is locally feasible but that the existing policy interface
does not enforce the roadmap's logged-future-leakage gate. That issue is a release
blocker, not a documentation caveat.

## 1. Question, estimand, hypotheses, and claim boundary

### 1.1 Question

Under a fixed simulator, how do simulated world-agent trajectories change when only a
prescribed exogenous ego trajectory changes after the observed/current frame, and can
independent measures distinguish no response from a costly response?

The primary estimand is a **within-simulator paired intervention effect** on one
source-frozen aligned geometric follower. Baseline and treatment share the same source
scene, observed history, current state, policy, seed, timestamps, lifecycle convention,
engine configuration, and horizon. They differ only in the realized ego trajectory
after `current_index`.

This is not a real-traffic causal effect. It does not estimate what a human or deployed
vehicle would do, establish safety, rank simulators, or imply WOMD-population behavior.

### 1.2 Falsifiable software hypotheses

1. An identity intervention reproduces the existing logged-ego rollout exactly.
2. An audited history-only built-in policy receives no direct channel containing
   future world state, future ego state, future validity, unrealized ego-plan state,
   intervention identity, or configured dose.
3. Two plans with identical realized prefixes but different later futures yield
   identical history-only policy actions until the realized prefixes diverge.
4. Log replay and constant velocity produce exactly unchanged **world-agent**
   trajectories under every ego intervention.
5. Because updates are synchronous, world state cannot respond during the transition
   that first changes ego. A legitimate world-state response first appears at the end
   of the following transition, after the changed ego has been observed.
6. A stronger treatment dose produces no greater planned ego speed or path progress
   than a nested weaker dose after onset. This is a treatment-construction property,
   not an assumed monotonic world response.

### 1.3 Falsifiable synthetic scientific hypotheses

Four independent analytic fixtures are mandatory:

- **aligned follower:** the primary ego brake causes EvalSim IDM to add follower
  braking after the synchronous response floor;
- **dose:** the stronger positive deceleration magnitude `b=4.0 m/s²` produces no
  smaller early IDM additional braking impulse than the primary magnitude
  `b=2.0 m/s²`;
- **no conflict:** an ego intervention outside the target's interaction corridor
  produces no EvalSim IDM response; and
- **overreactive sentinel:** a test-only exaggerated responder is responsive but also
  incurs more progress loss, jerk, or hard-braking exposure than nominal IDM.

The measures must distinguish nonresponse, response, and deliberately costly response
without collapsing them into a composite score.

### 1.4 Real-WOMD expectation

On the frozen source-eligible cohort:

- log replay and constant velocity must retain exact-zero world-response measures; and
- EvalSim IDM is expected to have at least 10 registered persistent-response events
  under the primary dose.

A null, sparse, sign-reversed, or uncomfortable real result is retained. Fewer than 10
responding EvalSim-IDM scenes does not invalidate the implementation, but it blocks the
claim that M6 demonstrated reactive behavior on the real-WOMD cohort.

### 1.5 Bounded claim after acceptance

If every gate passes, M6 may support only:

> Implemented and evaluated a typed paired ego-braking intervention on a fixed,
> source-eligible subset of the accepted local WOMD cohort. Audited history-only world
> policies received fixed observed history/static context at initialization and only
> realized current state thereafter; privileged log-replay, ego-plan, and Waymax
> references remained explicitly separated. Independent reactivity and response-cost
> measures detected nonresponse and controlled synthetic overreaction without a
> simulator winner or real-world causal/safety claim.

The final wording must state the actual eligibility count, intervention scope,
backend scope, nulls, and limitations.

M6 does **not** establish:

- real-world, human-driver, fleet, or WOMD-population causal effects;
- collision prevention, safety, route compliance, lane following, offroad behavior,
  traffic-signal compliance, or driving quality;
- a causal Waymax policy, independent ground truth, or numerical equivalence between
  EvalSim IDM and Waymax IDM;
- a simulator winner, replacement decision, composite reactivity score, or production
  readiness;
- representativeness beyond the accepted ten-shard conditional cohort; or
- general validity of the response measures, which remains an M7 question.

## 2. Frozen source and pairing boundary

### 2.1 Cohort reuse

M6 reuses the accepted M4 cohort exactly through the existing fail-closed accepted-M4
verifier and bounded reload adapter. It must not:

- rescan the ten shards to form a new base cohort;
- use shard `00010` or any other shard;
- rerank or replace an accepted M4 member;
- select from M5 metric outcomes, slices, effects, or per-scene tables;
- convert an adapter, parity, identity, or execution defect into ordinary
  ineligibility; or
- publish native scenario IDs, locators, source coordinates, or shard digests.

Every accepted M4 member receives one opaque `cohort_index` in canonical order. All 128
members appear exactly once in the M6 eligibility ledger.

### 2.2 Time and lifecycle

- `Scenario.metadata["current_index"]` is the final observed/current frame.
- Logged history through that frame remains exact and identical in every pair.
- Every NumPy M6 typed-plan rollout ends exactly at
  `stop_index = current_index + 40`, inclusive. It retains source history from frame
  zero through `current_index` and contains exactly the next 40 simulated transitions,
  even when the accepted source scenario has a longer future.
- The legacy `ego_plan=None` path remains full-length and unchanged. Any M6 comparison
  with that path uses its exact inclusive prefix through `stop_index`; typed-plan
  truncation is never applied to or mistaken for the legacy API.
- Actual positive source timestamp differences define durations, derivatives, and
  NumPy dynamics. Durations are never converted to frame counts.
- Ego identity, timestamps, and validity are unchanged by an intervention.
- The engine privately owns source-scheduled world validity, births, deaths, and
  re-entries. Birth/re-entry state remains an exogenous logged lifecycle oracle.
- The frozen primary follower must remain source-valid throughout the 40-transition
  window, so its response never depends on a birth/re-entry fallback.

The logged lifecycle convention is a known limitation. It is not evidence of a fully
generative world model.

### 2.3 Pair invariants

For one eligible scene, every baseline/treatment pair must share:

- scenario and frozen target identity;
- exact history/current state;
- seed `0`;
- policy and policy configuration;
- backend and dynamics configuration;
- timestamps and horizon;
- lifecycle convention and component masks; and
- every input other than the post-current ego plan.

Unexpected invalid state, asymmetric missingness, identity drift, policy failure, plan
failure, non-finite output, or component-mask drift fails the complete pair. It is not
imputed, replaced, or silently dropped.

## 3. Enforced policy-access architecture

### 3.1 Common policy surface

`SimulatorPolicy` remains the common downstream policy type, but it no longer supplies
an initialization capability by itself. Initialization is split into two explicit
audited built-in capabilities:

1. `HistoryOnlySimulatorPolicy` receives a `HistoryOnlyPolicyContext`.
2. `PrivilegedSimulatorPolicy` receives an explicitly privileged initialization
   object containing a defensive full-reference copy.

The engine rejects a plain `SimulatorPolicy`, a class implementing both capabilities,
or a mismatch between protocol type and M6 execution registration. Access role is
recorded in the separately versioned M6 sidecar; the legacy `PolicyMetadata.to_dict()`
schema and algorithm versions remain unchanged when algorithm behavior is unchanged.

Constant velocity and EvalSim IDM migrate to `HistoryOnlySimulatorPolicy`. Log replay
migrates to `PrivilegedSimulatorPolicy`. The pinned Waymax IDM adapter remains outside
the history-only policy protocol and is always privileged. The access-contract version
is separate from the simulator algorithm version.

This is an enforceable data-flow boundary for the audited built-ins, not a Python
sandbox, authentication mechanism, or security boundary against malicious same-process
code. After ego has changed, a legitimate reactive policy may infer aspects of the
intervention from the realized physical state. The guarantee is that no direct
configuration or unrealized-future channel is supplied.

### 3.2 History-only context

The immutable history-only initialization context contains:

- scenario identity and static typed map features;
- agent IDs, types, dimensions, and ego index;
- timestamps and agent frames only through `current_index`;
- the current index and future step count, but no future motion payload; and
- recursively frozen source-provenance scalars under the exact allowlist `source`,
  `source_version`, and `source_time_unit`; absent keys remain absent and no other
  scenario-metadata key is copied.

All arrays are copied and backed by immutable storage. It contains no future agent
state, future ego state, future validity, hidden full `Scenario`, intervention
configuration, future ego plan, or mutable reference.

### 3.3 Step observation

The history-only step observation contains:

- the current simulated frame;
- current/next timestamp and actual `dt`;
- IDs, types, dimensions, and ego index; and
- no source-derived `next_valid`, future plan, intervention name, or severity.

The engine masks controls using its private lifecycle state. History-only policies learn that
an agent was born or disappeared only through a later realized current frame.

History-only policies may return bounded acceleration/yaw controls only. Any nonempty
absolute override from a history-only policy is rejected because that policy cannot
know the engine-private next validity mask. Privileged policies may return an absolute
override under the existing contract; the engine still validates its validity against
the private lifecycle mask.

### 3.4 Leakage tests

Acceptance tests must:

- poison every post-current coordinate, velocity, heading, and ego state while keeping
  history fixed and prove identical history-only initialization and action prefixes
  while realized observations remain identical;
- independently poison future validity and prove it is absent from history-only policy
  inputs;
- prove privileged replay changes when its recorded future changes and retains its
  privileged label;
- run interleaved controller/policy instances to detect retained cross-run state;
- attempt to mutate every public array and nested provenance value; and
- prove the engine never supplies intervention identity, configured dose, or unrealized
  plan state as a direct policy input.

Engine rollouts may still change when the engine-owned future lifecycle changes. The
leakage gate concerns policy access and action decisions, not denial that lifecycle is
an engine oracle.

Existing repository test policies migrate explicitly to one of the two capabilities.
Unmigrated external/plain subclasses fail with a stable type error. The optional
`ego_plan=None` engine path must keep the pre-M6 canonical serialized rollout bytes
unchanged on frozen regression fixtures, including `Rollout.perturbation=None` and
legacy metadata. M6 access-role provenance lives only in the M6 sidecar. This preserves
M2/M5 no-plan reproducibility without pretending the new access contract is invisible
to the Git-bound M6 implementation.

## 4. Typed ego-intervention contract

### 4.1 Core types

M6 adds immutable, versioned source-neutral contracts:

- `EgoInterventionSpec`: family, version, dose, duration, parameters, access class,
  and canonical configuration fingerprint;
- `InterventionEligibility`: accepted/rejected decision, registered reason, analysis
  window, and frozen target index;
- `EgoTrajectoryPlan`: exact post-current ego states, validity/applied masks,
  realization type, feasibility audit, and canonical plan identity;
- `CounterfactualPair`: source scenario, baseline rollout, intervention rollout,
  frozen eligibility, and intervention identity; and
- `PairedMetric`: validates and reduces a counterfactual pair to one scene scalar.

The pair never wraps caller-owned mutable `Scenario` or `Rollout` objects directly.
Construction creates defensive immutable snapshots whose NumPy buffers are backed by
immutable bytes. Contract identity, dimensions, timestamps, masks, states, and typed
provenance are revalidated when the pair is constructed and before each metric pass.

`Rollout.perturbation` remains backward-compatible as `str | None`. The legacy
`ego_plan=None` path retains `None`. A typed M6 plan derives its own canonical string
identity, including sham `identity/v1`; callers cannot attach arbitrary unapplied text,
and the legacy free-text `perturbation=` argument remains rejected. The engine gains a
separate keyword-only typed-plan argument. Full typed provenance lives in a validated
M6 sidecar/result schema, not an unbounded rollout-metadata blob.

Canonical configuration JSON uses UTF-8, sorted keys, compact separators,
`ensure_ascii=True`, and `allow_nan=False`, preceded by an ASCII domain separator and
NUL byte. Canonical plan bytes use the domain
`ASCII("evalsim-ego-trajectory-plan-v1") || NUL` and then encode, in this fixed order:
the unsigned 64-bit big-endian byte length and bytes of the canonical configuration
fingerprint; the unsigned 64-bit big-endian byte length and UTF-8 bytes of the
realization type; one unsigned 64-bit big-endian frame count; validity as one-byte
`0/1`; applied mask as one-byte `0/1`; and
`timestamps/x/y/heading/vx/vy` as little-endian float64 C-order arrays. Every plan
array and mask is one-dimensional with that exact frame count and spans
`current_index..stop_index`, inclusive. The Waymax view uses separately identified
little-endian float32 arrays. Configuration and plan hashes are local mutation
detectors, not authentication or proof of source ownership.

### 4.2 Plan realization

The intervention compiler creates one ego plan before any world-policy run. Every
policy/backend condition consumes the same realized plan for that scene and dose.
Policies observe only its current realized ego state.

The engine owns timestamps, identity, validity, and feasibility enforcement:

- every M6 v1 plan is explicitly `logged_future_privileged`; no M6 v1 ego generator is
  described as a causal/current-only controller;
- source-templated transforms are independently differentiated and rejected if their
  implied acceleration, speed, yaw rate, continuity, or finite-value checks exceed the
  registered bounds;
- plan application cannot modify a world agent or any history frame; and
- the engine revalidates the plan immediately before execution.

For every transition, the engine order is exactly:

1. snapshot the current assembled frame;
2. call the world policy with that immutable current observation;
3. validate its action, privately apply the next lifecycle mask, and integrate or
   privileged-override world agents;
4. apply `ego_plan[next_index]` only to ego; and
5. atomically assemble, validate, and write the next frame.

Thus an ego plan first changes frame `t+1`; the world action that produced frame `t+1`
saw only frame `t`, and the earliest world-state divergence is at frame `t+2`.

The plan's local canonical hash detects accidental mutation and cross-backend drift.
Plan hashes remain local and are not promoted into tracked/public evidence.

### 4.3 Intervention library

The release-blocking M6 v1 library contains one identity control and one versioned
brake-pulse family with two nested doses. This is the smallest complete vertical slice
that can enforce leakage, pairing, timing, real-WOMD, and Waymax gates without a
combinatorial outcome surface.

#### `identity/v1`

- Exact logged ego trajectory.
- Mandatory sham-control and no-regression oracle.
- Uses access class `logged_future_privileged`.
- Its timestamps; every agent's ordered ID, type, length, width, validity, `x`, `y`,
  heading, `vx`, and `vy`; and every M6-sidecar world-policy acceleration, yaw-rate,
  absolute-override flag/state, and effective-control mask must equal the corresponding
  inclusive `0..stop_index` prefix of the legacy `ego_plan=None` run exactly.
- Its serialized rollout intentionally differs because
  `Rollout.perturbation` and the M6 sidecar identify the sham plan.

The separate legacy regression requires `ego_plan=None` itself to preserve the
pre-M6 canonical serialized bytes. Sham equality never falsely includes provenance.

#### `longitudinal_brake_pulse/v1`

- Begins on the first simulated transition.
- Uses access class `logged_future_privileged`; this describes the ego-plan generator,
  not the history-only world policies.
- Primary positive deceleration magnitude: `b = 2.0 m/s²`.
- Secondary positive deceleration magnitude: `b = 4.0 m/s²`.
- The speed deficit ramps for exactly `D = 1.0 s` and is then carried.
- Dose duration is resolved by interval overlap on actual timestamps.
- Speed is floored at zero; reverse motion is forbidden.
- Path progression follows a completely specified source ego arc-length template.
- The primary dose is the sole intervention in the 12-cell primary matrix.

Let `j = 0..40` index frames beginning at the current frame, let
`τ_j = timestamp[current+j] - timestamp[current]`, and let
`Δt_j = τ_{j+1} - τ_j`. Source ego position, wrapped heading, velocity, and scalar
speed are `p_j`, `θ_j`, `w_j = (vx_j, vy_j)`, and `v_j = ||w_j||`.

Define source arc knots:

`s_0 = 0` and `s_{j+1} = s_j + ||p_{j+1} - p_j||`.

Every source segment must have length strictly greater than `1e-6 m`; otherwise the
primary compiler returns `source_ego_path_degenerate`. This makes the arc knots
strictly increasing.

For positive magnitude `b`, the nonnegative speed deficit at a knot is:

`d_b(τ_j) = b * min(max(τ_j, 0), D)`.

The treatment speed is:

`v_b,j = max(0, v_j - d_b(τ_j))`.

The lost source-path distance on interval `j` is:

`q_b,j = min(s_{j+1} - s_j, 0.5 * (d_b(τ_j) + d_b(τ_{j+1})) * Δt_j)`.

Treatment arc progress is:

`r_b,0 = 0` and
`r_b,j+1 = r_b,j + (s_{j+1} - s_j) - q_b,j`.

After `D`, the deficit remains `b * D`; it does not continue increasing. Therefore the
post-pulse plan carries the speed/progress lag but adds no further braking increment.

At each `r_b,j`, use left-closed/right-open piecewise-linear interpolation on the
strict source arc knots, except that the last knot is closed. If `r_b,j` is bitwise
equal to an exact source arc knot, the interpolation primitive returns that knot's
original source position, velocity vector, and heading bits; it does not unwrap or
wrap the heading. Between knots, position and source velocity vector interpolate
linearly. Heading uses `numpy.unwrap` over the source heading sequence with
discontinuity `π`, interpolates the unwrapped values, and wraps the result to
`[-π, π]`.

For every `b>0`, the final plan velocity uses the primitive's velocity only as a
direction input and is always rescaled to magnitude `v_b,j`, including at an exact
knot. If its norm is at most `1e-12`, use the selected heading unit vector. Only
`b=0` returns final source velocity bits unchanged. The current state `j=0` is copied
exactly.

A zero-magnitude reconstruction is a required source-only oracle: it special-cases
`b=0` by copying every source ego field exactly, and the independent general
interpolator's exact-knot branch must return the same raw source bits for every field.
Comparison uses `numpy.array_equal`, not an angle wrap or tolerance. A mismatch
produces `zero_dose_reconstruction_mismatch` before any world-policy outcome.
Independent straight, curved, branch-crossing-heading, stopped-boundary, path-end, and
irregular-timestamp fixtures validate the formulas without calling the production
compiler.

#### Deferred intervention families

Positive-acceleration delay, stop/hold, path-normal offset, and arc-time warp remain
roadmap designs, not M6 v1 implementations or evidence. They receive no WOMD access,
synthetic result, code claim, or résumé claim in this milestone. Each requires its own
exact formulas, eligibility, metrics, compute budget, and outcome-blind amendment
before implementation. This plan deliberately does not pretend their former
high-level names were executable specifications.

### 4.4 Nested-dose gates

On analytic fixtures:

- `b=4.0` braking never yields greater post-onset ego speed or arc progress than
  `b=2.0`;
- every plan starts from the exact same current ego state.

These gates apply to plan construction. Real world-agent response is not required to
be monotonic. Real monotonicity violations are counted and retained.

### 4.5 Frozen feasibility audit

The primary eligibility compiler uses float64 and the following exact checks over
frames `j=0..40`:

- every timestamp and state value is finite;
- timestamps, ego identity, validity, and current state equal the source exactly;
- `0 <= ||velocity_j|| <= 60.0 m/s`;
- interval acceleration
  `a_j = (||velocity_j+1|| - ||velocity_j||) / Δt_j`
  lies in the closed interval `[-8.0, 4.0] m/s²`;
- wrapped yaw rate
  `ω_j = wrap(heading_j+1 - heading_j) / Δt_j`
  satisfies `|ω_j| <= 1.0 rad/s`;
- interval center displacement never exceeds
  `0.5 * (speed_j + speed_j+1) * Δt_j + 0.05 m`;
- its absolute distance residual from that trapezoidal-speed distance is at most
  `max(0.05 m, 0.10 * trapezoidal_distance)`; and
- when speed exceeds `0.6 m/s`, the absolute wrapped disagreement between heading and
  velocity direction is at most `π/6`.

All threshold comparisons are inclusive except the strictly positive source segment
length. Angles use the same `[-π, π]` wrap as the rollout engine. No finite-difference
endpoint is invented: controls exist only on the 40 actual intervals.

The primary population requires only identity and `b=2.0` feasibility. The `b=4.0`
plan is compiled outcome-blind on a frozen nested severity subset of primary-eligible
scenes; failure of that secondary plan cannot remove a scene from the primary
estimand.

## 5. Source-only eligibility

### 5.1 Primary brake eligibility

All predicates are computed from the source scenario and compiled ego plan before
policy/backend execution. The priority-ordered reasons are:

1. `insufficient_future_horizon`;
2. `ego_invalid_in_window`;
3. `ego_speed_below_5_mps`;
4. `source_ego_path_degenerate`;
5. `zero_dose_reconstruction_mismatch`;
6. `primary_ego_plan_infeasible`;
7. `no_stable_aligned_follower`; and
8. `current_ego_follower_overlap`.

A primary member must satisfy:

- at least 40 transitions after `current_index`;
- ego continuously source-valid over the complete window;
- current ego scalar speed at least `5.0 m/s`;
- the source path and zero-dose checks in §4 pass;
- the primary `b=2.0` plan passes the exact §4.5 feasibility audit;
- at least one non-ego `VEHICLE` is current-valid and remains source-valid over all 40
  transitions; and
- the deterministic two-stage target rule below succeeds.

For every current-valid non-ego vehicle candidate `i`, define its direction
`h_i = velocity_i / ||velocity_i||` when speed is greater than `1e-12`; otherwise use
`h_i = (cos(heading_i), sin(heading_i))`. Let
`r_i = position_ego - position_i`,
`center_i = dot(r_i, h_i)`,
`lateral_i = abs(h_i.x * r_i.y - h_i.y * r_i.x)`, and
`gap_i = center_i - 0.5 * (length_ego + length_i)`.
Heading disagreement is
`abs(wrap(heading_ego - heading_i))`.

Stage A retains candidates that:

- remain source-valid through frame `current+40`;
- have `center_i > 0`;
- have `lateral_i <= 2.75 m`; and
- have heading disagreement `<= π/6`.

If Stage A is empty, return `no_stable_aligned_follower`. If any Stage-A ego/candidate
pair has strict positive oriented-box overlap under the M5 `1.0.0` source-neutral
separating-axis definition, return `current_ego_follower_overlap`; exact edge touching
is not overlap.

Stage B retains a Stage-A candidate only when:

- `2.0 <= gap_i <= 40.0 m`; and
- ego is that candidate's nearest current forward leader under the exact EvalSim-IDM
  geometry.

For the nearest-leader check, consider **all** other current-valid contract agents,
regardless of type. Use candidate direction `h_i`; require positive longitudinal
center distance, lateral offset `<= 2.75 m`, and direction alignment at least
`cos(π/6)`. Another agent's direction uses normalized velocity above `1e-12` and its
heading fallback otherwise. Rank leaders by `(bumper_gap, integer_agent_id)`, exactly
matching EvalSim IDM. Ego must win that ordering.

If Stage B is empty, return `no_stable_aligned_follower`. Otherwise choose the target
by `(gap_i, integer_agent_id, contract_index)`. All inequalities above are exact and
inclusive as written; there is no floating tie tolerance. The selected target and
every component mask are frozen before execution and remain identical across policy,
baseline/treatment, dose, and backend.

The target is an **aligned geometric follower**, not a lane follower. No lane, route,
right-of-way, or traffic-rule inference follows from this predicate.

### 5.2 Eligibility-only gate

After plan acceptance and implementation freeze, an eligibility-only pass may reveal
only:

- total accepted M4 cases;
- eligible count; and
- registered aggregate rejection-reason counts.

It may not execute or expose a world-policy outcome. If the source-only rule requires
revision, the amendment must be outcome-blind, committed, and adversarially reviewed
before any rollout result.

Acceptance levels are:

- `N >= 30`: complete finite-cohort stability summaries and gated directional effect
  language are permitted;
- `N = 10..29`: descriptive/sparse results only; and
- `N < 10`: counts/reasons only, with the real-WOMD M6 reactivity claim blocked.

After any world-policy outcome is accessed, eligibility, target selection, thresholds,
cohort, doses, horizons, and primary measures are immutable.

### 5.3 Secondary intervention boundary

Positive-acceleration delay, stop/hold, path-normal offset, and arc-time warp are
deferred entirely. They do not receive M6 code/evidence, read WOMD, form real-WOMD
subsets, produce numeric results, or enter the M6 claim. Any future implementation or
execution requires its own exact outcome-blind preregistration and fresh review.

The only secondary WOMD plan is `b=4.0` braking. Before primary world outcomes, compile
it for every primary-eligible scene, freeze the successful opaque cohort indices as a
nested severity subset, and report only complete local secondary results. Its
feasibility cannot change primary membership.

## 6. Frozen execution matrix

### 6.1 Primary NumPy matrix

For every primary-eligible scene, run seed `0` for:

1. privileged `log_replay`;
2. history-only `constant_velocity`; and
3. history-only EvalSim `idm`.

Each policy receives:

- identity/logged ego baseline;
- primary `b=2.0 m/s²` brake plan; and
- secondary `b=4.0 m/s²` brake plan only on the frozen feasible severity subset.

The 12-cell primary family uses only baseline versus the primary dose over the first 40
transitions. The stronger dose is a complete secondary severity analysis and cannot
replace an unfavorable primary result.

### 6.2 Secondary severity matrix

The real-WOMD secondary matrix is baseline versus `b=4.0` for the same three NumPy
policy roles and four paired measures on the frozen severity subset: 12 complete local
cells. It is descriptive, cannot replace a primary cell, and its numeric values are
not promoted into tracked/public M6 evidence.

### 6.3 No policy-to-policy winner contrast

The primary estimand compares treatment with baseline **within the same policy**.
Differences between policy effect profiles may be shown descriptively, but M6 has no
policy-winner estimand and no total ordering.

## 7. Paired measures

The source-frozen target and component masks are used for every measure. Acceleration
and jerk use actual positive timestamp differences and never bridge invalidity.

### 7.1 Primary 12-cell family

The primary family is four measures × three NumPy policy roles for the primary brake
dose.

#### `additional_target_braking_impulse_mps` version `1.0.0`

- Compute target scalar-speed acceleration for baseline and treatment.
- Per transition, retain
  `max(0, acceleration_baseline - acceleration_treatment) * dt`.
- Scene scalar is the sum over the 40-transition window.
- Exact zero means no additional target braking.
- Higher means more braking response, not better driving quality.

#### `response_timeliness_s` version `1.0.0`

- Let target scalar-speed acceleration on transition `j` be
  `a_j = (speed_j+1 - speed_j) / (timestamp_j+1 - timestamp_j)` for
  `j=0..39`, where `j=0` starts at the current frame.
- Let `delta_a_j = a_treatment,j - a_baseline,j`.
- Search only starts `r >= 1`; transition `j=0` cannot respond because its policy
  action saw the unchanged current ego.
- A braking response is the smallest transition end `e+1` for which a contiguous run
  `j=r..e` has every `delta_a_j <= -0.5 m/s²` and
  `sum(j=r..e, dt_j) >= 0.2 s`.
- The response event time is `timestamp_e+1 - timestamp_current`. Attaching it to the
  transition end makes the earliest possible event occur at frame `current+2`.
- Let `W = timestamp_current+40 - timestamp_current`.
- With an event, `restricted_latency = min(event_time, W)`; without an event,
  `restricted_latency = W` and `responded=False`.
- An event ending exactly at `W` remains `responded=True` even though its scalar is
  zero.
- Store responder and censor counts separately.
- Primary scalar is `W - restricted_latency`, so exact nonresponse is zero and a faster
  valid response is larger.
- Conditional responder-only latency is secondary and suppressed below 10 responders.

#### `minimum_longitudinal_bumper_gap_change_m` version `1.0.0`

- Freeze `h_target,current` using the same speed/heading fallback as §5.
- At every scored frame `j`, define
  `gap_j = dot(position_ego,j - position_target,j, h_target,current)
  - 0.5 * (length_ego + length_target)`.
- Scene scalar is treatment minimum minus baseline minimum.
- This is a frozen geometric proxy, not lane headway or safety ground truth.

#### `target_progress_loss_m` version `1.0.0`

- Project target displacement from the current frame to the final scored frame onto
  the target's current-frame heading.
- Scene scalar is baseline progress minus treatment progress.
- Higher means more simulated response cost; it is not proof of overreaction.

All four directions are neutral with respect to overall simulator quality. They remain
separate and cannot be combined into one score.

### 7.2 Secondary measures

The following versioned secondary per-scene measures are implemented for analytic
diagnostics:

- `target_world_displacement_mean_m@1.0.0`: arithmetic mean of per-frame target
  Euclidean treatment-versus-baseline position distance;
- `target_speed_reduction_max_mps@1.0.0`: maximum
  `speed_baseline - speed_treatment`;
- `additional_absolute_jerk_integral_mps2@1.0.0`: sum over contiguous derivative
  intervals of
  `max(0, abs(jerk_treatment) - abs(jerk_baseline)) * dt`;
- `additional_hard_braking_exposure_s@1.0.0`: treatment exposure with target
  acceleration `<= -4.0 m/s²` minus baseline exposure under the same inclusive rule;

Only the last two cost measures participate in the pre-registered synthetic
overreactive-sentinel gate in §7.3, together with primary progress loss. All four may
be inspected locally on complete real-WOMD pairs, but they are exploratory,
non-gating, receive no inferential band or directional claim, and are never promoted
as numeric public evidence.

Strict oriented-box-overlap onset/resolution episodes for the frozen pair, minimum
center-distance change, capped constant-velocity disc-TTC change, treatment ego
speed/progress manipulation traces, and all-world response propagation beyond the
direct target are local exploratory debugging outputs only. They are not registered
metrics, acceptance gates, selection inputs, or tracked/public numeric evidence in M6.

Relational distance, TTC, and overlap can change even when the world policy is
nonreactive because ego moved. They must never be presented as world-state reactivity
without the world-only measures.

Use “oriented-box overlap onset/resolution,” not “collision avoided/introduced.”
TTC remains a capped constant-velocity proxy, not a collision forecast.

### 7.3 Nonreactivity and costly response

The exact world-only equality tensor includes, for every non-ego contract agent and
frames `current+1..current+40`, ordered identity, dimensions, validity, `x`, `y`,
heading, `vx`, and `vy`, plus rollout timestamps. Equality uses `numpy.array_equal`
field by field; there is no tolerance. A run is classified `nonreactive` only when
that entire tensor is equal and braking impulse, timeliness, and target progress loss
are exact structural zero. Relational gap, TTC, center-distance, and overlap measures
are excluded from this classification.

Costly response is measured independently by progress loss,
`additional_absolute_jerk_integral_mps2`, and
`additional_hard_braking_exposure_s`.

M6 intentionally defines no binary “overreaction” truth threshold for real WOMD. The
test-only overreactive sentinel must respond and, relative to nominal IDM on the same
analytic fixture, have both at least `1.0 m` more target progress loss and at least
`0.2 s` more additional hard-braking exposure. There is no post-hoc choice among cost
axes. M7 will evaluate real thresholds, false positives, and construct validity.

Minimum gap is a relational ego-plus-world effect. It can never contribute to a
sentence asserting that the world policy reacted.

## 8. Statistics and multiplicity

- Scenario is the statistical and resampling unit.
- Frames and agents first reduce to one scalar per scene.
- The primary family contains exactly 12 cells.
- A cell's ordered inputs are the primary-eligible scene scalars in ascending opaque
  `cohort_index`.
- The registered point estimator is the arithmetic mean, computed as
  `math.fsum(float(value) for value in ordered_values) / N`.
- The sample median is also reported descriptively using the ordinary midpoint of the
  two central values when `N` is even.
- Each primary cell uses `100,000` deterministic scenario resamples.
- Base seed is `20260729`.
- A canonical cell key contains statistics schema/version, metric name/version,
  policy name/access role, intervention config fingerprint, `N`, resample count, and
  base seed. Encode it with the canonical JSON rule in §4, hash with SHA-256, unpack the
  digest as eight big-endian unsigned 32-bit words, and seed
  `numpy.random.SeedSequence([20260729, *digest_words])` followed by
  `numpy.random.PCG64`.
- Draw one `int64` matrix of shape `[100000, N]` with replacement using
  `rng.integers(0, N, ...)`. Process one cell at a time; the matrix is at most
  `102.4 MB` for `N=128`, and sampled-value means are evaluated in chunks containing at
  most one million scalar elements.
- Each resample statistic is the NumPy float64 arithmetic mean of its selected scene
  scalars.
- Quantiles use `numpy.quantile(..., method="linear")`.
- Report the two-sided pointwise 95% band at quantiles `0.025` and `0.975`.
- The primary family also reports the Bonferroni-adjusted marginal level
  `L = 1 - 0.05 / 12 = 0.9958333333333333`, with two-sided quantiles
  `(1-L)/2` and `1-(1-L)/2`.
- These are finite-cohort reweighting sensitivity summaries, not confidence intervals,
  p-values, hypothesis tests, or population uncertainty.
- Metric-specific numeric nonzero thresholds are fixed at:
  - braking impulse `> 1e-9 m/s`;
  - timeliness `> 1e-9 s`;
  - absolute gap change `> 1e-6 m`; and
  - absolute progress loss `> 1e-6 m`.
- Exact structural equality of the negative-control world tensor remains a separate
  bitwise software gate; numeric thresholds never weaken it.
- Fewer than 10 thresholded nonzero scene effects is `event_sparse`.
- The real-reactivity expectation is at least 10 registered persistent-response events
  for EvalSim IDM, not merely floating-point nonzero impulse.
- `N < 30`, asymmetric source missingness, component drift, or a band containing zero
  forbids directional language.
- Directional effect language is allowed only when `N >= 30`, thresholded
  `nonzero_n >= 10`, source pairing is complete, and the **adjusted** band is strictly
  above or strictly below zero. This licenses only the registered effect sign, never a
  quality/winner statement.
- Conditional responder-only summaries show the total responder and censor counts and
  are suppressed below 10 responders.
- Every null, sign reversal, sparse cell, secondary dose, and backend disagreement
  remains visible.
- No slice intersections, policy winner, p-values, or composite score are permitted.

Primary status priority is exact:

1. any structural/pairing/component defect fails the run rather than producing a row;
2. `N < 10` blocks outcome execution under §5;
3. thresholded `nonzero_n < 10` gives `event_sparse`;
4. `10 <= N < 30` gives `small_n`;
5. otherwise an adjusted band containing zero gives `descriptive`;
6. otherwise the row gives `direction_supported`.

Only `direction_supported` permits the registered effect-sign sentence, and never a
quality sentence. Responder-only latency has its own `responder_sparse` suppression
when responder count is below 10.

The observed mean, median, both bands, `N`, thresholded nonzero count,
responder/censor counts, status, and directional flag are mandatory outputs for every
primary cell. The M5 statistical implementation may inform independent tests, but M6
receives a separately versioned paired-effect schema. Frozen M5 metrics, scorecards,
and accepted result artifacts are not reinterpreted or silently modified.

## 9. Bounded Waymax role

### 9.1 Execution roles

Pinned Waymax is used substantively as:

- the accepted-M4 WOMD data and identity adapter;
- a `PlanningAgentEnvironment`/`StateDynamics` executor that exact-sets the prescribed
  SDC ego plan;
- a no-world-sim-actor logged-world negative control; and
- a non-SDC `IDMRoutePolicy` reference.

The Waymax IDM path is described exactly as:

> a current-state-reactive, privileged logged-trajectory waypoint-following reference.

It is not causal, independent ground truth, map-route-aware, or a numerical twin of
EvalSim IDM. Every controlled world agent can use its complete logged future waypoint
trajectory.

### 9.2 Waymax subset

From primary-eligible scenes, require every adjacent source
`timestamp_micros[current+j+1] - timestamp_micros[current+j]` for `j=0..19` to equal
the integer `100000` exactly. Also require the frozen target to be requested by the
pinned non-SDC-vehicle control mask on all 20 transitions and absent from Waymax's
pinned frame-zero initialized-overlap exclusion. All three are computed from source
state before actor output. Rank each remaining member with:

`ASCII("evalsim-m6-waymax-reactivity-v1") || NUL || uint32_be(cohort_index)`.

Sort by SHA-256 digest bytes and then numeric `cohort_index`. Take the first 16 scenes,
or all only when 8–15 qualify. Fewer than 8 blocks the real-scene Waymax gate.

The subset and target are frozen before Waymax policy output. If fewer than 8 qualify,
the Waymax real-scene gate remains unsupported and the final claim is downgraded.

### 9.3 Matched and unmatched semantics

For every selected scene, run exactly two ego conditions—sham identity and the primary
`b=2.0` plan—through:

1. Waymax logged-world fallback;
2. Waymax privileged logged-trajectory IDM;
3. the same-scene, same-target, same-20-transition NumPy log-replay view; and
4. the same-scene, same-target, same-20-transition NumPy EvalSim-IDM view.

The matched backend gate applies only to the prescribed ego plan and logged-world
negative-control pair. EvalSim IDM versus Waymax IDM is a model/backend comparison, not
parity.

The canonical plan is float64. Its deterministic Waymax view casts
`x/y/heading/vx/vy` to little-endian float32 once in C order and receives a separate
local mutation hash. Compare Waymax output with that exact float32 view using:

- identity, timestamps, validity, actor masks, and lifecycle categories: exact;
- `x`, `y`, `vx`, and `vy`: `atol=1e-5`, `rtol=1e-6`; and
- circular yaw error `abs(wrap(actual-expected))`: `<=1e-5 rad`.

No NaN or non-finite value is tolerance-accepted. Logged-world values use the same
field rules. The result records per field/component denominator, maximum absolute
error, tolerance-failure count, and binary mismatch count; it never reports only a
pass Boolean.

The bounded gate also verifies:

- deterministic eager repeat and one-scene JIT agreement;
- no logged-world tensor response under ego intervention;
- the same synchronous response floor for the Waymax IDM bundle;
- exact target and 20-transition scope;
- requested/effective IDM control, logged lifecycle fallback, and initialized-overlap
  exclusion counts; and
- no mutation of source state, canonical float64 plan, or float32 view.

Because pinned Waymax actor logic uses fixed `0.1 s` assumptions while EvalSim dynamics
use actual source intervals, no universal backend parity is claimed. Exact cadence is a
source-only requirement for this narrow matched subset.

The four paired measures are recomputed over the exact 20-transition view for both
Waymax bundles. This forms exactly 2 bundles × 4 measures = 8 complete secondary
Waymax cells. With at least 10 scenes, each receives `10,000` deterministic resamples
and a pointwise 95% band under the §8 algorithm; below 10, point/band values are
suppressed as `insufficient_n` while counts remain. EvalSim IDM versus Waymax IDM is a
combined policy/backend comparison. Any disagreement is retained and explained; it is
never relabeled as numerical backend error or hidden behind a tolerance.

### 9.4 Compute boundary

- Decode an accepted scene once and evaluate every required condition while resident.
- Use compact `jax.lax.scan`, not stock full-state rollout, except for a tiny in-memory
  API oracle.
- Execute real Waymax scenes sequentially to bound dense pairwise-geometry memory.
- Pass condition/severity as fixed-shape array data where possible to avoid needless
  recompilation.
- Before official outcomes, an outcome-suppressed source-ranked pilot of at most 8
  eligible scenes may retain only wall time, stage time, and fresh-worker peak process
  RSS. World-policy values are neither displayed nor persisted.
- The pilot passes only if its total wall time is at most `30 minutes`, no individual
  scene exceeds `10 minutes`, and fresh-worker peak process RSS is at most `16 GiB`.
  Missing or non-finite measurements fail the pilot. These thresholds are fixed before
  outcome access and are not scaled after observing performance.
- The pilot cannot change the scientific matrix. A compute-driven amendment must occur
  before any outcome access and pass fresh review.

No paid cloud, accelerator, or new dataset is authorized or required.

## 10. Implementation and verification plan

### 10.1 Contract and engine increment

Add:

- `evalsim/contracts/counterfactual.py`;
- history-only and privileged policy initialization types;
- history-only and privileged policy protocol subclasses;
- typed intervention, plan, eligibility, pair, and paired-metric contracts; and
- versioned serialization with tamper rejection.

Modify the NumPy engine behind a keyword-only typed ego-plan path. Preserve the exact
legacy `ego_plan=None` path as the rollback and M2/M5 byte-regression boundary.

### 10.2 Policy migration

- Migrate constant velocity and EvalSim IDM to the history-only interface.
- Migrate log replay to the explicit privileged interface.
- Remove `next_valid` from policy observation.
- Reject absolute overrides from history-only policies.
- Keep lifecycle masks private to the engine.
- Record access role in the M6 sidecar without changing legacy no-plan rollout
  metadata.

### 10.3 Intervention and metric implementation

Add:

- `evalsim/perturb/m6.py`;
- `evalsim/metrics/m6.py`;
- `evalsim/stats/m6.py`; and
- `evalsim/evaluation/m6.py`.

Implement sham identity and the two-dose brake family, source-only eligibility, frozen
target selection, paired measures, resampling, and complete matrix accounting.

### 10.4 Waymax adapter

Add a separate M6 module, rather than changing accepted M4/M5 behavior:

- `evalsim/simulators/waymax_m6.py`.

It consumes the exact compiled ego plan, runs logged-world and privileged-IDM bundles,
emits compact arrays, converts through the project contracts, and records every
fallback/control category.

### 10.5 Local result lifecycle

Add M6-specific:

- immutable result schemas and verification;
- aggregate report builder;
- data-free/synthetic command; and
- opt-in official local-WOMD command.

Register explicit M6 console scripts in `pyproject.toml`, while keeping TensorFlow,
JAX, Flax, and Waymax imports lazy behind the optional command boundary. Do not
overload M5 stores, `RunManifest`, or schemas.

### 10.6 Test layers

Required layers are:

1. contract construction, canonical identity, roundtrip, tamper, and mutation tests;
2. policy access-boundary and future-poisoning tests;
3. engine history/time/lifecycle/synchronous-order tests;
4. intervention analytic, feasibility, identity, and nested-dose tests;
5. paired-measure formula, censoring, invariance, and missingness tests;
6. synthetic nonreaction, response, no-conflict, dose, and overreaction tests;
7. Waymax exact ego-plan injection, logged-world, repeat, JIT, and fixed-cadence tests;
8. public mocked official-run lifecycle and injected-failure tests;
9. full core-only and Waymo-extra repository suites;
10. opt-in eligibility-only and official real-WOMD paths; and
11. wheel and sdist build/install/import/NOTICE checks, plus Git, privacy,
    presentation, and deployment checks.

Independent oracles must not call the production helper they validate.

## 11. Official command and local evidence boundary

### 11.1 Repository, source, and runtime binding

Eligibility-only, compute-pilot, and official modes must all fail closed unless:

- the explicit personal/non-commercial local-use opt-in is present;
- the resolved project root is the canonical checkout, with no symlinked source path;
- `origin` is the credential-free canonical repository URL
  `https://github.com/6ickomod3/evalsim.git`;
- local `HEAD`, `refs/heads/main`, the live credential-safe `origin/main` lookup, and
  the approved implementation commit agree;
- the worktree/index are clean and contain no untracked executable source;
- the explicit accepted M4 run passes exact reuse verification;
- the data directory resolves exactly immutable shards `00000`–`00009`;
- pinned lock/runtime/config/schema fingerprints match; and
- every loaded `evalsim` module resolves inside the exact checkout.

The executable-source allowlist is exhaustive and ordered. It includes:

- every imported file under `evalsim/`;
- every M6 test and command helper;
- this accepted plan;
- `AGENTS.md`, `.gitignore`, `NOTICE.md`, `pyproject.toml`, and `uv.lock`; and
- the presentation build/check scripts only when release mode is invoked.

Reject symlinked source files, non-regular source nodes, untracked executable code, an
import outside the checkout, or an unexpected imported project module. Local
provenance records commit, tree, ordered relative source paths and their aggregate
digest, lock/runtime versions, JAX backend/device class, pinned Waymax commit, and
schema/config fingerprints.

Provenance is an allowlist. It must never serialize environment variables, full
`argv`, Git config, credential helpers, tokens, credential-bearing remote URLs,
absolute paths, native IDs, locators, source/shard hashes, or source coordinates.
Code/source/shard/result hashes remain local and never enter tracked/public evidence.

### 11.2 Safe result path and exclusive lifecycle

The run name must match exactly
`[a-z0-9][a-z0-9._-]{0,127}` and be one safe path component. Resolve
`outputs/m6/<run-name>` beneath the canonical worktree and reject:

- existing targets;
- symlinked ancestors or children;
- non-directory ancestors;
- non-regular files;
- any file with link count other than one; or
- a path visible to Git.

Create every directory with mode `0700` and every file, temporary, receipt,
transcript, diagnostic, and marker with mode `0600`, using exclusive no-overwrite
creation. Recheck mode, ownership, containment, node type, and link count on every
read/write.

The store has the one-way state machine:

`ABSENT → PENDING → COMMITTED → TERMINAL_SUCCESS`

or:

`ABSENT/PENDING/COMMITTED → TERMINAL_FAILURE`.

Rules:

- Create `PENDING` exclusively before any optional runtime import or data access.
- Write the eligibility receipt before any world-policy outcome.
- Write fixed-schema parts with unique primary keys; duplicate/missing/unexpected
  rows fail.
- Flush and `fsync` every file and containing directory before a transition.
- `result-manifest.json` contains expected relative files, exact sizes, and per-file
  hashes and does not list/hash itself. The exclusive `COMMITTED` receipt contains the
  hash of the exact canonical manifest bytes plus schema/row-domain identities.
- Reopen every file through guarded descriptors and independently recompute the
  complete store before success.
- Immediately before `TERMINAL_SUCCESS`, recheck Git/live main, source/runtime,
  accepted M4, shards, output layout, terminal transcript, eligibility receipt,
  determinism receipt, row domains, hashes, and permissions.
- Create `TERMINAL_SUCCESS` exclusively only after every check passes; it binds the
  exact manifest hash and `COMMITTED` receipt hash.
- On any `BaseException`, create `TERMINAL_FAILURE` before optional diagnostics when
  the store exists.
- A failed, interrupted, or ambiguous run can never resume, finalize, or promote in
  place. Use a fresh run name.
- Success and failure markers are mutually exclusive; any contradiction fails
  verification.

The M6 implementation may mechanically reuse the proven M5 guarded-store primitives,
but receives its own schema and injected-failure tests.

The ignored store contains only opaque cohort indices, aggregate/reason eligibility
accounting, compact paired scene scalars, responder/censor status, typed provenance,
determinism receipts, complete Waymax control/fallback accounting, aggregate stage
timing/RSS, and bounded diagnostics.

### 11.3 Terminal and diagnostic boundary

All three local modes capture Python `sys.stdout/sys.stderr` **and native file
descriptors 1/2** before argparse validation, optional imports, TensorFlow/JAX/Waymax
initialization, subprocesses, or worker creation. Child/native-library noise is
captured. Descriptor and stream restoration occurs under success, ordinary exception,
`BaseException`, interrupted write, nested capture, and child failure.

Exactly one canonical ASCII JSON status is emitted after capture closes:

- success goes to stdout and leaves stderr empty;
- rejection/failure goes to stderr and leaves stdout empty.

The allowlisted status contains only schema version, `success|rejected|failure`, stable
reason code, safe relative `outputs/m6/<run-name>` path (or safe relative
`FAILURE` marker), aggregate counts, and integer stage durations. It contains no
traceback, exception text/repr, absolute path, native ID, locator, hash, coordinate,
per-scene/component value, environment value, `argv`, or dependency log.

Persist a bounded native transcript and traceback only in ignored `0600` diagnostics.
The bound is `2 MiB` per transcript/diagnostic; truncation is explicit. Diagnostics
must not copy a TFRecord/example, array payload, full trajectory, environment dump,
credential material, or command line. Eligibility-only and compute-pilot modes obey
the same terminal contract.

Reuse the proven M5 terminal-capture semantics and add M6 injected tests for argparse,
import-time noise, native FD writes, subprocess/child writes, restoration,
`BaseException`, transcript overflow, failure-before-store, and contradictory markers.

### 11.4 Exact promoted aggregate schema

The tracked report, README, claim ledger, and site may promote only one sanitized
aggregate object with these required domains:

1. **Provenance labels:** M6 plan/schema/config versions, population label, source
   shard suffix range, policy names/access roles, intervention name/version/dose,
   horizons, and fixed limitations. No commit/tree/source/result digest or local path.
2. **Eligibility:** total `128`, primary eligible count, and every registered primary
   rejection reason with an integer count including zeros. Counts must sum to 128.
3. **Primary matrix:** exactly 12 rows. Each row contains metric name/version/unit,
   policy name/access role, pair `N`, thresholded nonzero count, responder/censor
   counts or JSON `null` when not applicable, arithmetic mean, median, pointwise band,
   adjusted band, status/suppression reason, source-pairing flag, and directional
   language flag. Every null, zero, adverse sign, and sparse row remains.
4. **Negative-control and timing gates:** complete exact world-tensor equality counts,
   sham/legacy equality, synchronous-floor violations, and plan-feasibility/
   dose-monotonicity violation counts.
5. **Waymax scope:** selected count and complete counts for
   `waymax_cadence_mismatch`, `waymax_target_control_incomplete`, and
   `waymax_target_overlap_excluded`; exact `16-or-floor` and 20-transition scope; all
   per-field component denominators/max errors/tolerance failures/binary mismatches;
   requested/effective/fallback/overlap-exclusion counts; and all 8 declared Waymax
   cell rows with suppression rules.
6. **Execution:** required row-domain counts, deterministic repeat status, aggregate
   stage durations, fresh-worker peak process RSS, review decisions, and gate status.
   Review decisions are a fixed ordered array with only
   `role ∈ {architecture, methods_statistics, privacy_claim}` and
   `decision ∈ {accept, reject}` plus integer `p1_count`, `p2_count`, and `p3_count`;
   reviewer names, timestamps, free-form text, and finding excerpts are forbidden.
7. **Claim and limitations:** the exact accepted bounded claim plus every limitation
   in §§1 and 9.

No numeric secondary `b=4.0` WOMD result and no numeric real-WOMD diagnostic from
§7.2 is promoted. Deferred intervention families produce no M6 result at all. Local
secondary presence cannot be used to select a favorable tracked cell. The public
matrix is generated mechanically from the sealed store and independently reconstructed
before release.

### 11.5 Release-surface audit

Before commit and again from the exact candidate commit:

- inspect tracked files and the exact staged set;
- build a fresh `git archive` from the candidate commit, not from the workspace;
- rebuild wheel and sdist from that archive;
- enumerate every archive/package member and scan both text and binary payloads;
- install each artifact in a fresh environment and verify lazy core import, M6 scripts,
  notices, and expected tests;
- build the exact site source bundle from that archive; and
- scan the final saved site bundle before deployment.

Reject any member or content from `data/`, `outputs/`, `private/`, `.venv/`, cache,
run-result formats, native/derived identity sentinels, coordinates, per-scene rows,
real-scene imagery, secrets, or vendored Waymax. Require `NOTICE.md` in both wheel and
sdist; direct prescribed notice text and pinned links in the site; and the README
notice before optional Waymo installation instructions.

Manually review every changed `docs/interview/` and presentation string for personal
information. Reuse the exact opaque project ID in `.openai/hosting.json`. Before and
after deployment, verify owner-only mode and the exact principal set are unchanged.
Do not invoke an access-control mutation.

## 12. Adversarial review gates

### 12.1 First exact-draft review

The first exact draft was **rejected before implementation**. It had no data/privacy
P1, but methods and architecture reviewers found that the signed brake wording could
reverse the treatment, the estimator/resampling rules were not executable, policy
secrecy was overstated, sham identity conflicted with provenance, and eligibility,
timing, Waymax tolerances, result sealing, terminal capture, source binding, and the
promoted schema were under-specified.

This revision:

- uses positive deceleration magnitude and a complete arc/time/state equation;
- freezes feasibility, geometry, timing, estimators, resampling, quantiles, numeric
  events, statuses, and directional gates;
- renames the policy role `history_only`, limits the boundary to audited data flow, and
  explicitly allows inference from realized state;
- separates legacy no-plan bytes from sham-identity trajectory equality;
- defers four under-specified intervention families;
- freezes the exact Waymax matrix, ranking, float views, tolerances, and accounting;
  and
- adds terminal, exclusive-store, provenance, promoted-schema, and release-surface
  state machines.

This revision remained blocked until the same three review disciplines returned no
P1/P2 finding; §12.2 records that closure.

Before implementation, independent reviewers must inspect this exact draft for:

- future leakage and capability separation;
- synchronous response timing;
- intervention feasibility and dose semantics;
- outcome-independent eligibility and target selection;
- metric construct validity, censoring, missingness, and gaming;
- statistical unit, multiplicity, sparse/null handling, and claim language;
- Waymax privilege and fixed-step/backend semantics;
- compute feasibility and result-lifecycle failure modes; and
- dataset, privacy, license, packaging, Git, and deployment boundaries.

Every P1/P2 finding must be corrected and the exact revised plan re-reviewed. P3
improvements may be accepted with explicit rationale. Implementation begins only after
the plan is marked accepted.

After execution, separate reviewers reconstruct eligibility accounting, primary cells,
resampling, null/sparse handling, Waymax receipts, and every tracked/public claim from
the sealed local result store.

### 12.2 Exact revised-plan acceptance

The second exact review rejected the draft before implementation for three architecture
ambiguities: typed-plan versus legacy rollout horizon, bit-preserving zero-dose heading
reconstruction, and incomplete plan-identity encoding. The revised plan froze a
40-transition typed prefix, raw exact-knot heading behavior, both plan masks, and
unsigned 64-bit canonical widths.

A final review then caught one remaining conflict between raw exact-knot velocity and
positive-dose treatment speed. The plan now separates the bit-preserving interpolation
primitive from final `b>0` velocity rescaling and reserves raw final velocity bits for
`b=0`.

After those outcome-blind corrections, independent architecture, methods/statistics,
and privacy/claim reviewers each returned **ACCEPT with no P1/P2 finding**. Their P3
improvements were also incorporated: positive dose notation, exact integer cadence,
non-gating exploratory diagnostics, an exact sham tensor list, a fixed history
provenance allowlist, pre-declared pilot limits, and a closed review-decision schema.
Implementation is authorized only under this accepted scope; every execution and
result review gate below remains open.

## 13. Release-blocking failure criteria

M6 fails or downgrades its claim if:

- the engine directly supplies a history-only policy with full future state, future
  validity, unrealized ego-plan state, intervention identity, or configured dose;
- legacy `ego_plan=None` canonical bytes drift, or sham identity trajectory/time/mask
  tensors differ from the logged-ego legacy baseline;
- a world state responds during the same transition that first changes ego;
- log replay or constant-velocity world trajectories change under intervention;
- eligibility, target choice, or component masks vary by policy, treatment, backend,
  or outcome;
- an intervention directly changes world state or violates frozen feasibility bounds;
- an execution error becomes missingness or one side of a pair is silently removed;
- conditional latency hides nonresponders;
- frames or agents are treated as independent statistical samples;
- a synthetic nonreaction, response, no-conflict, dose, or overreaction-separation
  oracle fails;
- a backend disagreement is hidden or a fixed-step check is generalized;
- EvalSim IDM and Waymax IDM are presented as equivalent;
- log replay or Waymax is presented as causal truth or a realism upper bound;
- a threshold, dose, eligibility rule, primary metric, horizon, or cohort changes after
  outcome access;
- a null, contradiction, sparse result, adverse result, or failed gate is omitted;
- a native identity, locator, coordinate, dataset, private input, per-scene result, or
  generated artifact reaches Git or deployment; or
- required Waymo/Waymax notices, pinned links, or owner-only presentation access are
  weakened;
- terminal output violates the single-status/empty-opposite-stream contract;
- a failed/interrupted store is resumed or promoted, a store seal/permission/path
  check fails, or source/runtime provenance is incomplete; or
- the tracked report/site omits or selects from a required promoted-schema row.

## 14. Rollback, documentation, and release

### 14.1 Rollback

- The existing no-ego-plan engine path remains exact and fully regression-tested.
- M6 code is isolated in versioned modules and result schemas.
- Frozen M5 metrics, statistics, evaluation schemas, and accepted local results are not
  migrated.
- If the policy-access migration breaks earlier rollouts, revert the M6 implementation
  commit rather than weakening the access gate.
- Failed ignored runs are preserved with a terminal failure marker and are never
  promoted in place.

### 14.2 Documentation closure

At accepted-plan commit, update README and the canonical roadmap only to say that M6
is in progress and link this pre-registration. After, and only after, sealed-result
acceptance:

- update this document with exact implementation and result closure;
- update the canonical roadmap;
- update README status, commands, evidence, and limitations;
- add a sanitized aggregate M6 acceptance report;
- update the claim-evidence ledger, competency matrix, and study plan;
- update the owner-only presentation with the complete primary matrix, nulls, sparse
  cells, Waymax boundary, and limitations; and
- correct the historical placeholder milestone labels in `evalsim/perturb` and
  `evalsim/stress`.

The accepted history-only/privileged policy boundary is a durable repository-wide
architectural decision. Add it to project-scoped `AGENTS.md` with the accepted plan;
do not copy it to global guidance.

### 14.3 Release sequence

1. Accept and commit the reviewed pre-registration.
2. Implement and pass data-free/synthetic and mocked lifecycle gates.
3. Obtain adversarial implementation acceptance.
4. Commit and push the exact clean implementation snapshot.
5. Run eligibility-only feasibility and the outcome-suppressed compute pilot.
6. If gates hold, execute the official M6 fixed-cohort run once.
7. Seal and independently review the local result.
8. Close tracked documentation with sanitized aggregate evidence.
9. Audit the working tree, stage, exact commit archive, package artifacts, notices,
   secrets, data, outputs, private material, and presentation bundle.
10. Run full Waymo/core tests and presentation checks.
11. Commit and push the M6 result closure.
12. Deploy the saved exact-source site version without changing owner-only access.
13. Verify remote commit, deployment health/access, and every displayed claim.

## 15. Acceptance checklist

- [x] Exact plan accepted by architecture, methods/statistics, and privacy/claim review.
- [ ] History-only policy capability enforced; privileged replay separated.
- [ ] `next_valid`, direct intervention config, and unrealized future plan absent from
      history-only policy inputs.
- [ ] Legacy no-plan bytes remain frozen; sham identity preserves trajectory/time/mask
      tensors with intentionally distinct provenance.
- [ ] Sham identity and both brake doses pass analytic and feasibility tests; four
      future intervention families remain explicitly unimplemented.
- [ ] Synthetic nonreaction, response, no-conflict, dose, and overreaction gates pass.
- [ ] All 128 accepted-M4 members have one source-only eligibility disposition.
- [ ] Primary eligibility floor and complete pair accounting pass.
- [ ] Full 12-cell primary matrix, 12 local severity cells, and 8 Waymax cells remain
      present under their declared publication/suppression rules.
- [ ] Deterministic paired scene-reweighting results reproduce independently.
- [ ] Waymax ego-plan/logged-world/privileged-IDM gates pass or claim is downgraded.
- [ ] Null, sparse, contradictory, and adverse results remain visible.
- [ ] No data, output, private material, native IDs, locators, coordinates, or secrets
      are staged, committed, pushed, or deployed.
- [ ] Full core-only and Waymo-extra suites pass.
- [ ] README, roadmap, acceptance report, ledger, study artifacts, and presentation
      match reviewed evidence.
- [ ] Required notices and owner-only presentation access remain unchanged.
- [ ] Terminal, exclusive result-store, exhaustive provenance, and exact promoted
      aggregate-schema gates pass.
- [ ] Exact commit is pushed, deployed, and verified.
