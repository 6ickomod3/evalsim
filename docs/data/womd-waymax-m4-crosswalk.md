# WOMD / Waymax M4 execution crosswalk

**Status:** M4 implemented, empirically accepted at
`a7a20e5de89c9c988f36a4b2f10ff4acc49246f0`, released, and post-release
verified; M4 metric parity is not run here
**Dataset profile:** WOMD v1.3.1 TFExample validation, 10 past + 1 current +
80 future frames
**Waymax revision:** `a64dfec9be8576b60d9cecc94f406d9812d4a7d0`
**Companion plan:** [`2026-07-28-m4-womd-cohort-waymax-parity.md`](../plans/2026-07-28-m4-womd-cohort-waymax-parity.md)

This document fixes the semantic boundary between the M4 WOMD selector, pinned
Waymax reference execution, and the source-neutral EvalSim `Rollout` contract.
It also records the pinned Waymax metric definitions that M5 will compare.

It contains only the accepted privacy-safe aggregate evidence below. It
contains no native scenario identities, record locators, shard digests,
coordinates, trajectories, local paths, generated result artifacts, or M4
numerical metric claims.

Primary implementation references:

- [Pinned Waymax configuration](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/config.py)
- [Pinned environment rollout](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/env/rollout.py)
- [Pinned state dynamics](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/dynamics/state_dynamics.py)
- [Pinned waypoint-following IDM](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/agents/waypoint_following_agent.py)
- [Pinned metric registry](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/metrics/metric_factory.py)
- [M3 source mapping](womd-waymax-m3-mapping.md)

## Claim boundary

M4 asks whether one frozen, complete-case cohort can preserve source identity,
ordering, time, validity, and supported motion semantics across exact log
execution and EvalSim conversion. It does not ask whether one simulator is
more realistic, select metric thresholds, produce scorecards, or establish
metric parity.

Both the M3 adapter and the M4 Waymax references consume the **same pinned
Waymax decode**. The direct-log oracle reads the decoded state's
`log_trajectory`; it is not a second independent WOMD decoder. Independent
source-array audits, compact-vs-stock execution checks, and contract checks
make the boundary non-tautological, but claims must still disclose the shared
decode.

The selected cohort is a **complete-case conditional sample from exactly the
first ten validation shards**. Eligibility requires one retained SDC with a
complete future, a usable non-SDC vehicle transition, and supported map
geometry. It is neither a random sample nor representative of WOMD, production
driving, geography, or any broader population.

## Selector and source boundary

### Selector-v2/v3/v4 representation corrections

The first bound attempt exposed an implementation mismatch between the
pre-registered source boundary and the arrays used by the selector. Selector
v2 freezes native identity and audit arrays from the eager pre-JAX decoded
mapping, validates the pinned pre-JAX schema, and only then performs lossless
semantic normalization. The four eligibility predicates, their priority, and
all ranking rules remain unchanged. This correction is a representation
contract repair, not WOMD selection, metric, or comparative evidence; all M4
result claims remain locked pending a fresh clean-commit run.

Independent review of the pinned decoder and factory contracts established that
fixed-size `state/is_sdc` padding permits the schema-level sentinel `-1` on
never-valid object slots. This describes the upstream representation contract,
not a per-record observation or result. Selector v3 admits only exact int64
`{-1, 0, 1}` for that field, requires `-1` exclusively on never-valid slots, and
maps only exact `1` to semantic true as the pinned Waymax factory does. Validity
masks remain strictly binary. Eligibility predicates and ranking are still
unchanged, and this correction unlocks no M4 result claim.

Independent review of the pinned factory contract also established that raw
per-frame length and width are reduced to a validity-masked float32 mean and
broadcast across time. An initial selector-v4 exact diff admitted every finite,
strictly positive float32 sample and compared an independently reduced NumPy mean
with fixed `atol=1e-6`. Adversarial semantic review **BLOCKED** that diff before
any further payload access: invented synthetic counterexamples exposed
NumPy/JAX reduction-order disagreement outside a fixed absolute tolerance and
showed that a positive subnormal input can flush to zero in the pinned JAX path.
Those counterexamples are numerical-contract evidence only, not WOMD evidence.

Corrected selector v4 ignores invalid-frame payload and requires each valid-frame
dimension sample to be a finite, positive **normal** float32 value, at least
`finfo(float32).tiny`. For `n` valid samples, let `k = n - 1`, define float32
machine epsilon `eps` and maximum finite value `max`, and let
`S = math.fsum(valid_samples)` be the high-accuracy reference sum. The analytic
float32 margin dominates its binary64 rounding. Define
`gamma_k = k * eps / (1 - k * eps)`. The source gate requires
`S * (1 + gamma_k) <= max`; this is an order-independent conservative overflow
guard for any positive-sample reduction order.

The actual pinned factory must still emit one finite, strictly positive scalar
broadcast across all 91 trajectory steps. A 91-step adversarial synthetic oracle
must exercise the actual pinned factory, including ordering, magnitude-boundary,
normal-boundary, and ignored-invalid-payload cases. Independent dimension parity
uses `mean = math.fsum(valid_samples) / n`, `rtol=0`, and
`abs_tol = mean * (gamma_k + eps * (1 + gamma_k))`. This analytic
machine-error bound is derived from the float32 reduction and division operations;
it is not an observed-data threshold, result-tuned tolerance, or claim of bitwise
NumPy/JAX reduction identity. Unrelated supported float32 fields retain
`rtol=0, atol=1e-6`.

Selector v4 adds no observed variation threshold. Eligibility predicates,
their priority, ranking domains and vectors, quotas, redistribution, fallback,
and execution scopes remain unchanged. The payload gate remains closed, and
this correction unlocks no M4 result claim.

### Selector-v4 terminal-output correction

The first clean selector-v4 bound attempt was stopped after the pinned TensorFlow
runtime emitted a deprecation diagnostic containing the absolute checkout source
path. The diagnostic was generated from static API/call-site metadata and exposed
no WOMD path, identity, locator, digest, coordinate, payload value, or result. The
attempt completed no manifest or acceptance result, is excluded in full from M4
evidence, and unlocks no claim.

Before another payload read, the deprecated compatibility iterator is replaced by
the pinned supported eager `TFRecordDataset` reader over one literal resolved file,
with uncompressed bytes, sequential reads, deterministic ordering, and no shuffle,
repeat, map, explicit prefetch, interleave, parallel reads, wildcard, or directory
expansion. Raw byte order, ordinals, counters, corruption fatality, clean EOF, file
identity, and grouped reload semantics remain unchanged.

Optional-runtime stdout and stderr—including native and child-process writes—are
captured at the file-descriptor boundary into fixed child `terminal-output.bin`,
created exclusively without symlink following and with mode `0o600` under the
validated ignored run directory. The exact path must also pass containment,
regular-file identity, and ignore gates; an existing file or symlink is refused.
Captured execution returns only an in-memory pending aggregate. After every child is
confirmed no longer alive—using terminate, kill, and join as needed on timeout, EOF,
or exceptional exit—normal execution flushes Python and native output, both terminal
descriptors undergo independent best-effort restoration, and the transcript is fsynced,
identity-checked, closed, and proven exactly empty. Partial setup starts no optional
work and restores both descriptors before skipping those process-global flushes, so
an unredirected descriptor cannot receive buffered bytes. Only then may the sanitized
local accepted aggregate be created; no optional runtime runs after the transcript
gate.

The transcript is never re-emitted, tracked, pushed, deployed, exposed in the
aggregate, interpreted for selection, or used as result evidence; only whether it is
empty participates in acceptance, and its contents are never parsed or allowlisted.
Official PASS/FAIL output occurs after finalization. If normal restoration fails, the
single stable failure line uses a preserved non-inheritable original status
descriptor; an already-raised stable failure retains precedence.
All acceptance-critical descriptors are closed before publication or status; only the
dedicated status descriptor remains until output, and its final close is best-effort
because the status is then irrevocable.
The pre-existing `TF_CPP_MIN_LOG_LEVEL=2` constraint is unchanged and is not evidence
for this correction; no Python warning filter or TensorFlow logger-level change is
added. This changes the executable snapshot, not selector-v4 population, ranking,
schemas, or metric scope.

### Fifth bound attempt interruption

The fifth fresh bound attempt started from clean, pushed commit
`aadfa6f0a37c85f8bc37b8a34335688c8adbb185` and was manually interrupted when
the user explicitly requested a pause. Its stable CLI status was
`FAIL (unexpected_failure)`. The ignored attempt directory was checked only for
filenames: execution provenance, the terminal transcript, and both manifest
filenames exist, while no accepted `aggregate-summary.json` exists. No artifact
content was opened or interpreted.

Existence of those files establishes no scan, manifest, cohort, reference, policy,
metric, benchmark, or acceptance result. The entire fifth attempt is excluded from
M4 evidence and unlocks no claim. It and the four earlier attempt directories remain
retained unchanged and may not be reused, overwritten, deleted, published, or
combined with a later run.

Selector-v4 population, fingerprint, ranking, schemas, execution scopes, and metric
scope remain unchanged. This documentation-only interruption record changes the Git
tree and executable-source fingerprint. The payload gate remains closed until this
snapshot passes its required gates, is independently reviewed, committed, and
pushed, and clean local `HEAD` equals `origin/main`. A sixth fresh attempt must then
repeat both complete scans from record zero through clean EOF.

### Sixth bound attempt aggregate-privacy deviation

The sixth fresh bound attempt ran from verified clean, pushed commit
`cb4e2c981194a22400f8e56a375c44390aaf9c12` and exited only with
`FAIL (aggregate_privacy_key)`. Privacy validation precedes exclusive accepted-report
creation, so no accepted aggregate was created. No ignored run artifact or content
was inspected after execution. The code establishes no accepted scan, manifest,
cohort, reference, policy, metric, benchmark, or comparative result; the attempt is
excluded in full and unlocks no claim.

Tracked-source inspection, without WOMD or ignored-output access, found two independent
schema-label collisions: `benchmark.nearest_rank_p95_seconds` contains the forbidden
fragment `rank`, and `checks.selected_locator_reload_complete` contains
`locator`. No claim is made about which collision the runtime encountered first.

Schema version 1 and both field names remain fixed. The broad case-insensitive
forbidden-fragment gate remains fail-closed, with only two exact path-and-type
exceptions: the benchmark label must be a finite, strictly positive built-in `float`,
and the reload-check label must be exactly `True`. The same key at another path, a
wrong value or type, a case variant or near-collision, a private child key, or any
other private-key fragment remains rejected. Absolute paths, digest-like values,
non-string keys, and non-finite JSON remain rejected.

Validation deep-copies only exact built-in JSON containers, keys, and scalar types
into an owned plain tree while applying every privacy gate. Tuples, subclasses,
mapping proxies, and other objects are rejected without calling overridable methods.
The owned tree is canonically encoded once, and the publisher writes those exact
validated bytes through its exclusive atomic seam; it never serializes the
caller-owned object after validation.

Pure production benchmark and aggregate builders plus a complete invented canonical
aggregate must prove the exact schema passes. Adversarial mutations must prove every
near-collision and private field still fails without publication. Hostile key,
mapping, list, and scalar subclasses must prove that masked spellings and
state-switching observations cannot bypass the snapshot or change serialized bytes.
Self-referential exact dictionaries/lists must be detected by active-container
identity, and any remaining excessive-depth `RecursionError` must become stable
`aggregate_json`; neither case may reach the exclusive writer.
Selector-v4 population, fingerprint, ranking, schemas, execution scopes, estimator,
and metric scope remain unchanged; only the plan, companion crosswalk, CLI, tests, Git
tree, and executable-source fingerprint change.

All six attempt directories remain local, ignored, retained unchanged, and excluded.
They may not be inspected for evidence, resumed, reused, overwritten, deleted,
published, or combined. The payload gate remains closed until the exact correction,
tests, package/site audits, and independent reviews pass; the final snapshot is
committed and pushed cleanly; and local and remote refs match. A seventh fresh attempt
must then repeat both complete scans from record zero through clean EOF.

### Aggregate-privacy implementation readiness

The correction now uses the two exact schema-v1 path/type exceptions, pure production
benchmark and aggregate builders, an exact-built-in deeply owned JSON snapshot,
active-container cycle detection, stable snapshot/encoder recursion failures, and
single-encoding exclusive byte publication. Hostile key/container/scalar subclasses,
wrong paths and values, private children, cycles, excessive depth, non-finite values,
absolute paths, 64-hex strings, and mutation at the byte-write seam all fail without
publication.

Final independent semantic, stable-error, privacy-bypass, and evidence reviews
accepted the implementation with no unresolved blocker. Before this record, the
focused CLI suite passed 110 tests; the locked matrix passed 421 tests with one
expected local-data skip; full Waymo-extra passed 493 tests with one expected
local-data skip; and core-only passed 418 tests with 22 expected optional-runtime
skips. Fresh package/notices, installed-help, forbidden-content, and presentation
checks also passed.

No WOMD or ignored output was accessed for this correction. These are readiness facts,
not M4 execution evidence. Because this record changes the executable snapshot, all
exact-snapshot gates must repeat before a clean commit/push and verified local/remote
ref equality can open the seventh-run payload gate.

### Seventh bound attempt accepted execution

The seventh fresh attempt ran from verified clean, pushed commit
`a7a20e5de89c9c988f36a4b2f10ff4acc49246f0`, repeated both complete scans of
exactly validation shards `00000`–`00009` through clean EOF, and exited
**PASS**. The accepted aggregate stayed under ignored local output storage,
the terminal transcript was empty, and the tracked worktree stayed clean.
Three independent adversarial method/semantics, privacy/release, and
numerical/evidence reviewers accepted the sanitized evidence with no
unresolved blocker.

The selected 128 scenarios are a complete-case conditional sample, not a
random or representative sample:

| Shard | Raw | Eligible | Rejected | Selected |
|---:|---:|---:|---:|---:|
| `00000` | 286 | 148 | 138 | 13 |
| `00001` | 309 | 165 | 144 | 13 |
| `00002` | 306 | 158 | 148 | 13 |
| `00003` | 307 | 151 | 156 | 13 |
| `00004` | 269 | 133 | 136 | 13 |
| `00005` | 278 | 150 | 128 | 13 |
| `00006` | 285 | 154 | 131 | 13 |
| `00007` | 297 | 171 | 126 | 13 |
| `00008` | 284 | 143 | 141 | 12 |
| `00009` | 295 | 154 | 141 | 12 |
| **Total** | **2,916** | **1,527** | **1,389** | **128** |

All 1,389 rejections were `source_no_supported_map`; the other three
pre-registered source-rejection counts were zero. This code means that no
roadgraph ID group passed the frozen strict supported-map predicate, not that
the record contained no map data. Every shard filled its quota, all quota
deficits and redistribution were zero, and the under-128 fallback was not
used.

Both manifests were byte-identical; selected-entry provenance reload,
full-cohort M3 conversion, and independent source-array parity passed. EvalSim
log replay, constant velocity, and IDM completed 80 transitions on all 128
scenarios. The compact Waymax exact-log path, direct logged-state oracle, and
conversion to the EvalSim `Rollout` contract passed the same cohort and
horizon; the bounded stock-Waymax check passed one transition on the first
selected scene. Both paths share the pinned Waymax WOMD decode, so this is not
an independent-decoder comparison.

All 128 scenarios qualified for the Waymax privileged logged-trajectory
waypoint-following IDM reference. Its pre-registered 16-scenario,
20-transition subset repeated byte-identically and the one-scene kernel
JIT-compiled. Accounting recorded 8,467 requested controlled transitions:
7,134 effective controls and 1,333 initialized-overlap fallbacks; 112 vehicles
used that fallback, the full cohort had 3,308 initialized-overlap vehicle
exclusions, and 6,673 lifecycle fallback transitions were recorded. The
minimum qualifying-vehicle effective-transition count was 20 and nonfallback
motion was observed. These facts do not establish realism or numerical parity
with EvalSim IDM.

The Apple-CPU batch-two exact-log `jit(vmap(...))` gate matched eager
sequential output and passed inverse-permutation invariance. Its fresh-worker
80-transition microbenchmark used 20 synchronized warm runs: compilation
0.217983625 s, median 0.001897854 s, nearest-rank empirical p95
0.002617709 s, and median throughput 1,053.8218429868682 scenarios/s. Peak RSS
was 587,808,768 bytes (about 560.6 MiB), which is process high-water RSS—not
JAX device memory. This is not end-to-end latency, full-cohort batching,
accelerator or scaling evidence, or production throughput.

M4 ran no custom-versus-Waymax numerical metric parity and makes no statistical
realism conclusion. The empirical gates are closed. Evidence-closure commit
`9e4012d` is on GitHub `main`, and its exact source was successfully deployed
through the existing owner-only presentation as version 6. Data and generated
run artifacts remained local and ignored.

### Population

- Resolve suffixes `00000` through `00009` individually and in order.
- Require exactly one local file per suffix.
- Never use a wildcard, directory-wide input, or Waymax `@N` expansion.
- Never open suffix `00010` or any other extra local file.
- Read every raw record in every in-scope shard from ordinal zero to clean EOF.
- Decode with one immutable configuration: repeat once, no shuffle, one source
  shard, deterministic execution, no batch dimensions, 128 object slots,
  30,000 roadgraph points, and 45 × 800 SDC-path samples.

The scanner advances separate `raw_seen`, `decode_attempted`, and
`event_emitted` counters. At clean EOF they must equal one another and the sum
of eligible and rejected events. Stream corruption, decode drift, duplicate
native identity, adapter failure, contract failure, and parity failure are
fatal; none can be relabeled as an ordinary eligibility rejection.

### Result-independent base eligibility

The source predicate may reject a decoded record only for the following
conditions, in this priority:

1. the retained source does not identify exactly one SDC;
2. that SDC is not valid for every frame from current through final future;
3. no non-SDC vehicle is valid at both current and the first future frame; or
4. no source roadgraph group passes the frozen supported-map rule.

The predicate inspects only pre-registered source fields: object identity,
type, SDC role, validity, valid-frame planar motion, dimensions, timestamps,
past/current/future markers, and roadgraph position, direction, type,
identity, and validity. It never inspects simulator output or metric values.

Every source-eligible record must subsequently pass adapter, `Scenario`, and
independent M3 parity checks. These are fatal gates rather than selection
filters, so policy success cannot change cohort membership.

### Deterministic selection and local-only state

Selection uses pre-registered domain-separated SHA-256 rankings, fixed
per-shard quotas, deterministic redistribution, and a declared evidence floor.
The complete scan and canonical local manifest are reproduced independently
before policy execution. Selected entries are then reloaded through their
exact local locator and provenance.

The full manifest remains under ignored output storage. Tracked code and
documentation may contain algorithms, schemas, versions, reason-code names,
aggregate counts, and sanitized benchmark definitions, but not native
identities, per-record locators, source coordinates, trajectories, maps,
digests, or generated result artifacts.

## Reference execution matrix

| Reference path | Scope | M4 purpose |
|---|---|---|
| M3 source → `Scenario` | selected cohort | Identity, ordering, time, validity, supported motion/map, and provenance boundary |
| EvalSim log replay | selected cohort, full future | Exact equality to the converted `Scenario` |
| Waymax exact log → `Rollout` | selected cohort, full future | Exact supported-field parity after declared invalid-fill normalization |
| EvalSim CV and IDM | selected cohort, full future | Determinism, lifecycle, contract, and finite-output gates only |
| Waymax privileged waypoint-following IDM | frozen nested subset, 20 transitions | Controlled/fallback accounting and `Rollout` conversion for a deliberately different policy |
| Sequential vs compiled/vmapped exact log | fixed batch of two | Common-output equality and permutation invariance |

EvalSim IDM and Waymax IDM are different models. M4 does not require or imply
numerical equivalence between them.

## Exact-log Waymax reference

### Environment

The exact-log path uses:

```text
BaseEnvironment
  dynamics_model = StateDynamics
  init_steps = 11
  max_num_objects = 128
  controlled_object = SDC
  allow_new_objects_after_warmup = true
  compute_reward = false
  metrics_to_run = ()
```

`reset` invalidates the simulation trajectory and copies the first 11 logged
frames, leaving the current frame at index 10. An expert SDC actor applies
`StateDynamics.inverse`, whose action is the next logged
`[x, y, yaw, vel_x, vel_y]` state. `StateDynamics.compute_update` sets those
fields directly; it is a state-setting reference, not a bicycle or physical
dynamics model.

`BaseEnvironment.step` controls the SDC. Noncontrolled objects take the next
logged state through the reference-trajectory fallback. Logged timestamps are
copied by the common Waymax dynamics update. Allowing post-warmup object
injection preserves logged births for noncontrolled objects.

### Compact output

The M4 scan carries the full immutable `SimulatorState`, but emits only these
post-transition leaves:

| Compact leaf | Meaning |
|---|---|
| `x`, `y` | global planar center |
| `yaw` | bounding-box heading |
| `vx`, `vy` | global planar velocity |
| `valid` | source-slot validity |
| `timestamp_micros` | Waymax-emitted source timestamp |
| `timestep` | post-transition Waymax index |

For a full future, numeric, validity, and timestamp leaves have
`[transition, object_slot]` shape; timestep has `[transition]` shape. The scan
does not return full state or observation histories, which would repeat the
91-frame trajectory, 30,000-point roadgraph, and 45 × 800 path tensors at
every emission. This reduces materialized output memory; it does not imply
that the scan carry, compiled executable, or working set omits those tensors.

### Stock and direct oracles

The compact scan has two deliberately separate checks:

1. **Stock execution oracle.** On an in-memory synthetic Waymax fixture, five
   compact transitions must equal the post-transition frames from
   `waymax.env.rollout`. Stock rollout returns the initialized state plus each
   transition, so the initial element is removed before comparison. The first
   selected local scene uses only one stock transition as a bounded public-API
   gate.
2. **Direct logged-state oracle.** Every compact real-scene emission is
   compared with the corresponding source `log_trajectory` frame, including
   the emitted `timestamp_micros`. Timestamps are not reconstructed from an
   EvalSim timeline for this check.

The stock oracle catches carry/emission-order and public-API integration
errors. The direct oracle catches a compact and stock path that could otherwise
agree on the same wrong value. Synthetic contradiction fixtures alter one
supported field at a time, including timestamp, and must be rejected.

Integer identities, ordering, masks, timestamps, and timesteps are exact.
Supported valid float32 fields other than dimensions use
`rtol=0, atol=1e-6`. Dimension scalar parity uses the pre-registered
high-accuracy-sum analytic bound above with `rtol=0`. Only values under a false
validity mask may be canonicalized before comparison.

## Waymax privileged logged-trajectory waypoint-following IDM reference

The exact M4 name is:

> **Waymax privileged logged-trajectory waypoint-following IDM reference**

The implementation identifier is
`waymax_privileged_logged_trajectory_waypoint_following_idm`.

This is the pinned upstream `IDMRoutePolicy`, despite the misleading “Route”
name. At the pinned revision it does not consume the roadgraph or `sdc_paths`.
It uses each controlled object's complete logged trajectory as a privileged
geometric waypoint reference. It is not causal ground truth, a map-route-aware
policy, or a numerical twin of EvalSim IDM.

### Locked upstream defaults

| Parameter | Pinned default |
|---|---:|
| desired velocity | 30 m/s |
| minimum spacing | 2 m |
| safe time headway | 2 s |
| maximum acceleration | 2 m/s² |
| maximum deceleration magnitude | 4 m/s² |
| exponent | 4 |
| maximum lookahead | 10 |
| lookahead from current position | enabled |
| additional lookahead points | 10 |
| additional lookahead distance | 10 m |
| invalidate at path end | disabled |
| integration time step | 0.1 s |

Runtime inspection must reproduce these values exactly; upstream default drift
is fatal.

### Leader and motion semantics

With the locked `lookahead_from_current_position=True`, leader search begins
from the controlled object's current simulated pose and appends ten synthetic
headway points over 10 m. It tests oriented-box overlap between those candidate
points and current agent boxes, removes self-pairs, and masks invalid
candidate/other-agent states. The earliest collision along this candidate
geometry supplies the leader speed and the Euclidean distance from the
controlled object's current position.

The policy then applies its IDM acceleration formula, updates speed by 0.1 s,
and clips speed at zero. Motion advances by trapezoidal travel distance along
the current simulated yaw, then projects the candidate point onto the closest
point/direction of that object's full logged trajectory. Thus the IDM speed is
interactive, while the geometric path and returned yaw are privileged by the
log.

With `invalidate_on_end=False`, projection may extrapolate along the final
logged direction. Reaching the last waypoint does not invalidate the object,
and the end-of-path velocity fallback retains its current velocity.

### Control, lifecycle, and fallback

The nested reference uses `PlanningAgentEnvironment` with:

- a logged expert SDC through `StateDynamics`;
- one `IDMRoutePolicy` sim-agent actor;
- rewards and metrics disabled; and
- the same 11-frame initialization and 128-slot state.

Before each transition, the EvalSim wrapper requests IDM control only for an
object that is:

- not the SDC;
- source type vehicle; and
- valid in the logged trajectory at both current and next frame.

The planning environment additionally excludes every object marked by its
initialized-overlap rule. Therefore:

```text
effective_control =
  requested_vehicle_control AND NOT initialized_overlap_excluded
```

A non-SDC vehicle that fails current/next validity uses logged fallback and is
counted as a lifecycle fallback. An otherwise requested vehicle that is
initialized-overlap excluded also uses logged fallback and is counted
separately. The actor is not restricted to the one continuously valid vehicle
used to qualify a scene; every vehicle satisfying the transition mask is
requested.

### Initialized-overlap semantics

The pinned planning environment computes initialized overlap from all 128
logged `[x, y, length, width, yaw]` boxes at **source frame zero**:

- pairwise oriented-box overlap uses a strict separating-axis test;
- self-pairs are removed;
- edge touching is not overlap because projected intersection must be strictly
  positive; and
- no validity mask is applied.

The lack of a validity mask is intentional parity with the pinned code, even
for invalid or padding slots. A separate NumPy separating-axis implementation
must reproduce the Waymax mask on adversarial synthetic fixtures and each
nested-subset state before execution.

## Waymax compact output → EvalSim `Rollout`

The converter exposes no Waymax datatype downstream.

| Boundary | Conversion rule |
|---|---|
| history | Copy the converted `Scenario` through current frame 10 without mutation. |
| simulated future | Append compact post-transition frames beginning at source frame 11. |
| agent alignment | Retain source-valid object slots in source order; verify the same unique ordered object identities as `Scenario.agents`; never align by compact-list position alone. |
| validity | Preserve compact validity exactly. |
| invalid numeric payload | Replace with deterministic finite zero while keeping validity false. |
| heading | Normalize circular yaw using the M3 rule. |
| timestamps | Compare emitted microseconds directly with the source; retain the already validated, normalized EvalSim timeline in `Rollout`. |
| dimensions and type | Preserve the corresponding `Scenario` agent values. |
| provenance | Record backend/policy name and version, Waymax revision, configuration fingerprint, horizon, seed, source lineage, and control/fallback aggregates as JSON-native values. |

Conversion rejects missing or duplicate identity, order drift, shape drift,
time drift, non-finite valid motion, mutation of logged history, unsupported
horizon, or undeclared control fallback.

## JAX compilation, batching, and memory evidence

The substantive batching gate is:

```python
jax.jit(jax.vmap(single_scene_exact_log_kernel))
```

It runs on a pre-selected batch of two states with identical static shapes:
128 object slots, 91 trajectory frames, 30,000 roadgraph points, and
45 × 800 SDC-path points.

The gate requires:

- a sequential eager baseline;
- compiled output equal to the sequential output;
- reversed input followed by inverse permutation restoring the same output;
- explicit `lower(...).compile()` timing;
- completed host-to-device transfer before execution timing;
- one untimed compiled execution; and
- synchronized warm executions using `block_until_ready()`.

The benchmark runs in a fresh worker. Process peak RSS is reported only as
process high-water memory, never as JAX device memory. Compilation and warm
execution are reported separately. The waypoint-following IDM kernel must JIT
for one state; its `vmap` is optional because dense pairwise 128-slot geometry
is an expected CPU-memory risk.

M4 makes no claim about accelerator speedup, scaling efficiency, full-cohort
batching, or production throughput.

## Discrepancy taxonomy

Every observed difference must have exactly one stable category:

| Category | Permitted proof |
|---|---|
| `unsupported_field` | The pre-registered crosswalk excludes the field and no claim or gate consumes it. |
| `invalid_fill_normalization` | Validity masks are exactly equal and only numeric payload under `valid=false` differs before zero canonicalization. |
| `lifecycle_fallback` | The precomputed current/next validity request mask is false and the resulting agent/frame equals direct logged fallback. |
| `initialized_overlap_exclusion` | The independent frame-zero overlap oracle matches Waymax and the excluded agent/frame equals logged fallback. |
| `float32_representation` | Only a declared valid non-dimension float field differs within `rtol=0, atol=1e-6`, or a dimension scalar differs within its pre-registered high-accuracy-sum analytic bound with `rtol=0`; identity, mask, time, order, and selection cannot use this category. |
| `policy_definition_difference` | The comparison is explicitly between differently defined EvalSim CV/IDM and Waymax IDM, outside an exact-parity gate. |
| `defect` | Every other difference. |

Classification never removes a selected scenario, waives a required path, or
turns a failure into eligibility. Any unresolved `defect` blocks M4
acceptance.

## Pinned Waymax metric definitions for M5

M4 records these executable definitions but runs **no custom-versus-Waymax
numeric metric parity**. Metric parity, aggregation, eligibility alignment,
and statistical comparison begin in M5. Upstream registry names are not
treated as mathematical definitions.

All Waymax metrics return a float32 value and a boolean validity mask at the
current simulator timestep.

### `log_divergence`

- Scope: per object.
- Value: Euclidean distance in XY between simulated and logged position at the
  same current timestep.
- Valid when both the simulated and logged object states are valid.
- It is a positional distance in meters, not MSE and not a trajectory-level
  distribution by itself.

### `overlap`

- Scope: per object.
- Value: `1.0` when its current oriented
  `[x, y, length, width, yaw]` box strictly overlaps at least one other valid
  object's box; otherwise `0.0`.
- Self-pairs are removed and edge touching is not overlap.
- The reported object's current validity is the output validity mask.

### `offroad`

- Scope: per object.
- Construct the four current oriented-box corners and add the object's center
  Z to each corner.
- For each corner, find the nearest valid road-edge boundary or median sample,
  using squared XYZ distance with Z stretched by a factor of 2.
- Determine side from the source road-edge direction and, when contiguous,
  its predecessor direction.
- Value: `1.0` if any corner has positive signed distance; otherwise `0.0`.
- The pinned implementation marks every returned object value valid; it does
  not propagate the trajectory-valid mask. M5 must state any additional
  EvalSim eligibility mask rather than silently attributing it to Waymax.

### `sdc_wrongway`

Despite its name, the pinned implementation does **not** compare SDC heading
with lane or path direction.

- Scope: SDC scalar.
- Compute Euclidean XY distance from the current SDC position to every valid
  `sdc_paths` sample.
- It does not filter on `sdc_paths.on_route`.
- Let `d` be the minimum finite point distance.
- Value: `0.0` when `d < 3.5 m`; otherwise `d`.
- Valid exactly when a finite path-point distance exists.

M5 must call this a thresholded **distance-to-SDC-path-samples** metric unless
the implementation changes. It cannot support a directional wrong-way claim.

### `sdc_progression`

- Scope: SDC scalar; requires `sdc_paths`.
- Among valid `on_route` paths, choose the path whose samples are closest in
  Euclidean XY distance to the current valid SDC position.
- Project logged frame-zero SDC position, logged final-frame SDC position, and
  current simulated SDC position independently to their nearest valid samples
  on that chosen path and read sample arc lengths.
- Value:

  ```text
  (current_arc - start_arc) / (end_arc - start_arc)
  ```

- If start and end arc lengths are equal, the value is `1.0`.
- The pinned code does not clip the ratio.
- It is valid when the current SDC is valid and a finite valid on-route path
  distance exists; otherwise it returns zero with `valid=false`.

### `sdc_off_route`

- Scope: SDC scalar; requires `sdc_paths`.
- Compute the minimum Euclidean XY point distance to valid on-route samples
  and separately to valid off-route samples.
- Flag off-route when the on-route distance is greater than 5 m, or when it
  exceeds the off-route distance by more than 2 m.
- If flagged, return the minimum on-route point distance; otherwise return
  `0.0`.
- The current SDC validity participates in path-distance eligibility.
- In the pinned executable code, absence of a finite on-route distance leads
  to a zero value with `valid=false`; despite its docstring, it does not return
  the closest off-route distance in that edge case.

### `kinematic_infeasibility`

- Scope: per object in the metric implementation; the planning environment
  selects the SDC result.
- Apply Waymax bicycle-model inverse dynamics to the transition from
  `timestep - 1` to the current timestep with `dt=0.1 s`.
- Value: `1.0` if either:
  - acceleration magnitude exceeds `10.4 + 1e-3 m/s²`; or
  - steering-curvature magnitude exceeds `0.3 + 1e-3 m⁻¹`;
  otherwise `0.0`.
- Validity comes from the inverse-dynamics action.
- `PlanningAgentEnvironment` suppresses the initialized state's value because
  that transition was copied from the log rather than selected by the actor.

## M5 parity obligations

Before M5 compares a custom metric numerically with Waymax, it must freeze:

- the exact per-frame and per-agent eligibility intersection;
- SDC selection versus all-agent output;
- units, threshold inclusivity, and validity propagation;
- treatment of invalid/padded objects;
- whether frame values become rates, counts, distributions, or
  per-scenario scalars;
- horizon and initialization-frame exclusions;
- float tolerance and exact fields; and
- the expected effect of EvalSim's 2-D contract where Waymax uses Z,
  source paths, or full roadgraph tensors.

Metric disagreement is not automatically a defect: the first task is to show
that both sides implement the same mathematical quantity on the same eligible
population. No M4 result may be presented as metric parity or a realism
conclusion.
