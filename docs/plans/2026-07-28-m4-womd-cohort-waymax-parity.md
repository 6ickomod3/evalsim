# M4 implementation plan — deterministic WOMD cohort and Waymax parity

**Date:** 2026-07-28
**Status:** ⚠️ First bound attempt stopped before event emission; selector-v2
correction and clean-commit rerun pending
**Milestone:** M4 — exact ten-shard cohort → Waymax reference execution → EvalSim
rollout contract

This plan is written before scanning WOMD eligibility, selecting the cohort, running
comparative policies, or inspecting M4 results. It is the falsifiable execution record
required by `AGENTS.md`.

## 1. Baseline and readiness

The starting source is Git commit
`c38b2220cf0995b2ceb9c0e29c175421d60055cf`. At pre-registration:

- the locked normal suite passes 170 tests with one gated local-data skip;
- the exact validation shard suffixes `00000` through `00009` each resolve to one local
  ignored file;
- an additional local shard `00010` exists and is explicitly out of scope;
- the pinned Waymax/JAX/TensorFlow/Flax environment from M3 imports on one Apple CPU
  device; and
- the worktree is clean, while `data/`, `outputs/`, TFRecords, Parquet, and generated
  reports remain ignored.

No M4 WOMD payload was read and no dataset-derived M4 value was inspected before this
selector and evidence plan was written.

## 2. Question, hypothesis, and falsification

**Question:** Can the M3 single-record boundary support a deterministic, auditable
ten-shard cohort while exact log playback, compact Waymax execution, EvalSim contracts,
and explicit JAX batching agree on their pre-registered common semantics?

**Hypothesis:** A frozen 128-scenario cohort can be selected without cherry-picking from
the exact ten local shards. On that cohort, EvalSim and a pinned Waymax reference will
preserve identity, agent ordering, SDC ownership, the 10/1/80 time boundary, validity,
and exact log-playback motion after canonical invalid-value normalization. A small
explicitly vmapped CPU batch will match sequential execution. EvalSim IDM and Waymax
IDM will be compared as different models, not forced into numerical agreement.

M4 is falsified, and the public claim remains locked, if any of the following occurs:

- a required shard is missing or ambiguous, shard `00010` or another out-of-scope file
  enters the population, or a configured glob/`@N` expansion is used;
- a record disappears without exactly one local scan event, an unclassified exception is
  treated as an ordinary rejection, or a shard read terminates without a clean EOF;
- native scenario identity is duplicated within the ten-shard population;
- the complete local manifest is not byte-identical on an independent repeat scan;
- fewer than 128 eligible records exist and the documented fallback evidence floor is
  not met;
- any selected locator reloads to a different identity, checksum, config, adapter, or
  schema version;
- exact log playback changes an eligible valid motion value, timestamp, identity,
  ordering, SDC, or mask outside the declared normalization;
- a supported rollout converter aligns agents by compact list position rather than
  source slot/object identity;
- sequential and `jit(vmap(...))` reference outputs disagree outside the locked
  tolerance, or inverse permutation does not restore the original order;
- any selected full-cohort path fails, whether or not the failure has a stable
  classification, or any discrepancy in the supported matrix remains unexplained;
- tracked or deployed content includes native IDs, coordinates, trajectories, map
  samples, local paths, TFRecords, Parquet, manifests, or generated experiment reports;
  or
- a claim describes the Waymax IDM reference as causal, map-route-aware, ground truth,
  or numerically equivalent to EvalSim IDM.

An honest negative result is acceptable. Changing the selector, support matrix,
tolerance, subset, or horizon after seeing comparative results requires a dated plan
amendment and a fresh adversarial review.

## 3. Exact population and deterministic selector

### 3.1 Shard boundary

Resolve exactly these suffixes, in this order:

`00000, 00001, 00002, 00003, 00004, 00005, 00006, 00007, 00008, 00009`

Each suffix must match exactly one filename ending in
`tfrecord-SSSSS-of-00150`. The coordinator calls the existing exact resolver ten times.
It must not enumerate the directory to expand scope, use a wildcard as the dataset
input, or use Waymax sharded-path notation. The resolved `00010` file is never opened.

Each shard is read from ordinal zero to clean EOF with the locked M3 dataset config:
repeat 1, no shuffle, deterministic execution, one source shard, unbatched scenarios,
128 object slots, 30,000 roadgraph points, SDC paths enabled, and the 10/1/80 temporal
profile. File SHA-256 is computed once per command with a before/after file-identity
check and reused only while the immutable path, device, inode, size, and modification
time remain unchanged.

### 3.2 Scan accounting

The raw serialized-record iterator increments `raw_seen` immediately after yielding a
TFRecord and before parsing or preprocessing. The scanner then increments
`decode_attempted` before decode and `event_emitted` only after producing one local
`ScanEvent` for that same ordinal:

- `eligible`, with the selector fields below; or
- `rejected`, with exactly one stable reason code chosen by a documented priority.

A corrupt shard stream, decode/config drift, adapter/contract/parity failure, duplicate
identity, non-clean EOF, or any unclassified exception is a fatal scan failure, not a
rejected scenario. At each clean EOF, independently maintained counters must satisfy
`raw_seen == decode_attempted == event_emitted == eligible + rejected`. The repeat scan
uses a fresh raw iterator and independently reproduces every per-shard raw count and
ordinal sequence.

The full local event contains:

- shard suffix and ordinal;
- native scenario ID;
- shard SHA-256 and path-independent dataset-config fingerprint;
- WOMD, Waymax commit, adapter, adapter-schema, selector, and manifest-schema versions;
- eligibility/rejection code;
- selection rank and membership; and
- only JSON-native scalar accounting fields.

It contains no coordinates, trajectories, map samples, tensors, rendered images, or
absolute paths. Canonical manifest content also excludes wall-clock timestamps,
duration measurements, output paths, hostnames, and execution-order fields.

### 3.3 Base eligibility

Only these four result-independent source predicates may reject a decoded record, in
this exact priority order:

1. `source_sdc_count_not_one`: among object slots with `any(valid[0:91])`, the source
   does not identify exactly one SDC slot;
2. `source_sdc_future_incomplete`: that unique retained SDC is not valid at every frame
   10 through 90 inclusive;
3. `source_no_world_vehicle_transition`: no non-SDC source object of type ID 1 is valid
   at both frames 10 and 11; or
4. `source_no_supported_map`: no roadgraph ID group satisfies the independent M3 map
   rule—one source type throughout; type in lane IDs `{0,1,2,3}` or road-edge IDs
   `{14,15,16}`; at least two finite, unique XY points; every segment length strictly
   greater than `1e-6` m and at most `0.75` m; and every finite, nonzero source direction
   agrees with its following segment by at most 10 degrees.

The source predicate reads exactly these fields: object `id`, `type`, `is_sdc`, and
91-frame `valid`; valid-frame `x`, `y`, `velocity_x`, `velocity_y`, `bbox_yaw`,
`timestamp_micros`, `length`, and `width`; `state/which_time`; and roadgraph `xyz`,
`dir`, `type`, `id`, and `valid`. Shape/dtype drift, duplicate retained object IDs,
invalid/nonconstant dimensions, non-finite valid motion, timestamp absence/disagreement,
non-increasing time, unexpected `which_time`, or any other source/contract invariant
failure is fatal.

Every source-eligible record must then pass adapter conversion, the `Scenario` contract,
and the independent M3 parity check. Any failure is fatal to M4; it can never exclude a
record or change the ranking population. The base rule is independent of comparative
policy output, and a record cannot become eligible because one simulator happens to run.

### 3.4 Cohort ranking

The target is 128 scenarios. All ranking messages use this unambiguous byte encoding:
ASCII domain separator, one zero byte, five ASCII shard-suffix bytes, one zero byte,
the ordinal as an unsigned 64-bit big-endian integer, one zero byte, then the native
scenario ID's strict UTF-8 bytes. No rendered integer, whitespace, locale, or platform
newline participates.

The within-shard SHA-256 domain separator is `evalsim-m4-cohort-v1`.

The domain separator is public and fixed above; it is not a secret salt. Within each
shard, eligible records are sorted by `(rank, ordinal)`. Initial quotas are 13 records
for suffixes `00000`–`00007` and 12 for `00008`–`00009`, the quotient/remainder
allocation of 128 across ten ordered shards.

If a shard cannot fill its quota, its deficit is filled from all remaining eligible
records using the same encoding with domain separator
`evalsim-m4-redistribution-v1`, sorted by
`(redistribution_rank, shard_suffix, ordinal)`. The selected manifest order is canonical
suffix order, then within-shard selection rank—not execution completion order.

If the ten shards contain fewer than 128 base-eligible records, select the complete
eligible population without replacement only when at least 32 records are eligible and
every one of the ten shards contributes at least one. Otherwise M4 stops for a plan
revision and no cohort claim is made. The exact selected count, including any fallback,
must be stated in every later claim.

### 3.5 Determinism gate

Before any policy comparison, perform a second complete scan in a fresh scanner
instance. Per-shard raw counts, ordinal sequences, canonical JSON bytes, and SHA-256 for
both manifests must match. Reload every selected locator by exact suffix and ordinal and
require native identity, shard digest, config fingerprint, and adapter/schema versions
to match the frozen manifest.

## 4. Local manifest and publication boundary

The full manifest is an ignored local artifact under
`outputs/m4/cohort/manifest.json`. It is exclusive-create and never silently overwritten.
A second run uses a separate path. It is never staged, pushed, uploaded, or deployed.

Tracked code may include the manifest schema, selector algorithm, public domain
separator, stable reason-code registry, and tests with invented identities. Tracked
documentation may include only aggregate counts, sanitized booleans, runtime versions,
declared tolerances, and non-reconstructive performance measurements. It must not
include:

- native scenario IDs or their rank hashes;
- per-record suffix/ordinal pairs;
- shard digests;
- coordinates, trajectories, map values, or real-data images;
- absolute local paths; or
- the local manifest or comparative output files.

If M4 is accepted, public evidence must report privacy-safe aggregate construction
accounting: total and per-shard raw/eligible/selected counts; total rejection counts for
each of the four registered reasons; quota deficits and redistribution counts; whether
the under-128 fallback was used; IDM-qualifying/subset counts; initialized-overlap
vehicle exclusions; effective controlled transitions; lifecycle fallbacks; and the
actual cohort/subset/horizon. The cohort must be labeled a complete-case conditional
sample from exactly the first ten validation shards—not random, representative of WOMD,
or representative of production driving.

## 5. Reference execution architecture

All optional Waymax/JAX/TensorFlow imports remain lazy. Core EvalSim modules continue to
depend only on `Scenario`, `Rollout`, `SimulatorPolicy`, and related contracts.

### 5.1 Full-cohort EvalSim paths

For every selected `Scenario`, run:

- exact `LogReplayPolicy`;
- `ConstantVelocityPolicy`; and
- `IDMPolicy`.

Each path runs all 80 transitions after current frame 10. These are
contract/lifecycle execution gates in M4. Comparative realism metrics and statistical
scorecards remain M5 work.

### 5.2 Full-cohort Waymax exact-log reference

Use the pinned public Waymax interfaces:

- `BaseEnvironment`;
- `StateDynamics`;
- `EnvironmentConfig(init_steps=11, max_num_objects=128,
  controlled_object=SDC, compute_reward=False, metrics=MetricsConfig(()))`; and
- an expert SDC actor inferred from the environment dynamics.

The SDC is set from its next logged state; noncontrolled objects use the environment's
logged fallback. `StateDynamics` is an exact state-setting reference, not a physics
model.

For the cohort path, implement a thin public-API `jax.lax.scan` around
`env.reset`/`env.step` that emits only current `x`, `y`, `yaw`, `vel_x`, `vel_y`,
`valid`, `timestamp_micros`, and timestep. The emitted timestamp is compared directly
and is never reconstructed from the input during parity. Do not materialize stock
`RolloutOutput.state` and `observation` for 81 steps because they repeat the full
roadgraph, route, and trajectory state.

The compact scan must match stock `waymax.env.rollout` over five transitions on a small
in-memory synthetic Waymax fixture, including every emitted frame and timestep, so
carry/emission-order defects are observable. On the first selected local scene, run only
one stock transition as a memory-bounded API gate. Independently compare all 80 compact
real-scene emissions to the direct source log, including exact emitted
`timestamp_micros`. A timestamp-only contradiction fixture must fail this oracle.

### 5.3 Nested Waymax IDM reference

At the pinned revision, `IDMRoutePolicy` does not consume the roadgraph or `sdc_paths`.
It projects controlled agents onto their complete logged trajectories. M4 names it:

> Waymax privileged logged-trajectory waypoint-following IDM reference.

It is not causal ground truth and is not expected to match EvalSim IDM numerically.
Defaults also differ, so the semantic crosswalk records desired speed, headway,
acceleration/deceleration, leader logic, path projection, lifecycle, and control-mask
differences.

The reference uses the pinned upstream defaults without tuning: desired velocity
30 m/s, minimum spacing 2 m, safe headway 2 s, maximum acceleration 2 m/s², maximum
deceleration 4 m/s², exponent 4, maximum lookahead 10, lookahead from current position
enabled, ten additional lookahead points, 10 m additional lookahead distance, and
`invalidate_on_end=False`. Any upstream/default drift is fatal.

The required IDM evidence uses a deterministic nested subset of 16 selected scenarios
and exactly 20 future transitions:

1. identify non-SDC vehicles valid at every frame 10–30;
2. compute the pinned initialized-overlap exclusion exactly from all 128 logged
   `[x, y, length, width, yaw]` boxes at source frame 0, remove the diagonal, and mark an
   object excluded when it overlaps any other box; apply no explicit validity mask at
   this stage, matching the pinned environment;
3. retain a scenario only if at least one non-SDC vehicle remains that is valid at every
   frame 10–30 and is not initialized-overlap excluded;
4. rank qualifying scenarios using the Section 3.4 byte encoding and the independent
   domain separator `evalsim-m4-idm-subset-v1`; and
5. take the lowest 16 ranks, breaking ties by suffix then ordinal.

If fewer than 16 qualify, use all only when at least eight qualify; otherwise stop for
plan revision. This subset rule is computed before any IDM output.

Use `PlanningAgentEnvironment` with logged/expert SDC, `StateDynamics`, rewards and
metrics disabled, and an `IDMRoutePolicy` actor restricted to non-SDC vehicles that are
valid at the current and next logged frame. Birth, disappearance, and invalid-transition
frames fall back to the log and are counted. The environment's initialized-overlap
exclusion remains visible. The actor controls every non-SDC vehicle satisfying that
dynamic mask; it does not restrict control to only the continuously valid qualifying
vehicle. Each subset scene must have at least 20 effective controlled
vehicle-transitions from the continuously valid vehicle, and the aggregate output must
contain at least one controlled valid motion value that differs from log fallback by
more than `1e-6`.

An independent NumPy separating-axis box-overlap implementation must reproduce the
pinned frame-0 overlap mask on synthetic fixtures and on every nested-subset state
before execution. Fixtures include overlap, edge-touching, rotated, invalid, and padding
boxes so the absence of an upstream validity mask is explicit.

A separate analytic synthetic oracle must exercise both a free-road and close-leader
vehicle. It independently computes the first IDM speed/action direction from the locked
parameters, proves positive free-road acceleration and close-leader deceleration,
confirms the actor's controlled branch is nonzero, and confirms the integrated state
uses that action rather than logged fallback. The real nested-subset IDM run is repeated
from identical inputs and its canonical compact output bytes must match exactly.

Full-cohort or full-80-transition Waymax IDM is out of M4 scope. Adding it requires a
pre-result plan amendment; no optional pilot can silently broaden the M4 claim.

### 5.4 Waymax-to-EvalSim rollout conversion

The converter:

- takes compact emitted frames plus the original unbatched Waymax state and EvalSim
  `Scenario`;
- copies logged history through current frame 10;
- appends emitted simulated frames 11 through the requested horizon;
- maps retained agents by unique Waymax object slot/ID and verifies the exact ordered ID
  sequence, never by compact list position alone;
- canonicalizes invalid numeric payloads to finite zero while preserving validity;
- normalizes circular yaw consistently with the M3 adapter;
- preserves timestamps, dimensions, source lineage, policy/backend configuration,
  horizon, seed, and Waymax commit as JSON-native provenance; and
- rejects missing/duplicate IDs, shape drift, time drift, non-finite valid values,
  history mutation, or undeclared control fallback.

The resulting object is a normal EvalSim `Rollout`; downstream M5 code never receives a
Waymax datatype.

## 6. Supported parity matrix

M4 requires and reports only this matrix:

| Path | Population | Required comparison |
|---|---:|---|
| M3 source → EvalSim scenario | full selected cohort | independent identity, time, mask, supported trajectory/map, and provenance parity |
| EvalSim log replay | full selected cohort, 80 transitions | exact equality to `Scenario` |
| Waymax exact-log reference → EvalSim rollout | full selected cohort, 80 transitions | exact valid-field/mask/time/ID/order equality after invalid-fill normalization |
| EvalSim CV and IDM | full selected cohort, 80 transitions | deterministic contract/lifecycle/finite-output gates; no Waymax numerical equality claim |
| Waymax waypoint-following IDM | nested 16 scenes, 20 transitions | deterministic finite output, declared control/fallback accounting, contract conversion |
| sequential vs `jit(vmap(...))` exact-log kernel | fixed batch 2 | common output equality and permutation invariance |

For valid float32 source/reference values, the only allowed numerical tolerance is
`rtol=0`, `atol=1e-6`; integer identities, masks, ordering, timestamps in canonical
microseconds, and selection are exact. A wider tolerance requires a pre-result plan
amendment.

Every discrepancy is assigned one stable category with these proof rules:

- `unsupported_field`: the field is excluded in the pre-registered crosswalk and is not
  used by a claim or parity gate;
- `invalid_fill_normalization`: masks are exactly equal and only numeric values under a
  false mask differ before both are canonicalized to zero;
- `lifecycle_fallback`: the precomputed current/next validity control mask is false and
  the resulting agent/frame equals the direct log fallback;
- `initialized_overlap_exclusion`: the independent frame-0 overlap oracle agrees with
  the pinned mask and the excluded agent/frame equals log fallback;
- `float32_representation`: only a declared valid float field differs within
  `rtol=0, atol=1e-6`; identity, mask, time, order, and selection never use this category;
- `policy_definition_difference`: the comparison is explicitly between EvalSim IDM/CV
  and the differently defined Waymax IDM, not within a parity gate; or
- `defect`: anything else.

Classification never waives, removes, or replaces a selected scenario. Any failure of a
required selected path blocks acceptance even if it has a stable name, and every
unresolved `defect` blocks the public claim.

M4 documents the definitions and eligibility of Waymax overlap, offroad, wrong-way,
route, log-divergence, and kinematic metrics, but runs no custom/Waymax numerical metric
parity. That comparison begins in M5. In particular, a metric's upstream name is not
treated as its mathematical definition.

## 7. JAX batching and benchmark

The required batching gate is a real explicit
`jax.jit(jax.vmap(single_scene_exact_log_kernel))` over a fixed batch of two selected
states with identical static shapes: 128 object slots, 91 frames, 30,000 roadgraph
points, and 45 × 800 SDC-path samples.

The two states are the selected cohort entries with the lowest SHA-256 ranks under the
Section 3.4 byte encoding and domain separator `evalsim-m4-vmap-v1`, with suffix then
ordinal as the tie-break. They are chosen before any execution result.

Acceptance requires:

1. sequential eager output as an independent baseline;
2. compiled vmapped output equal to sequential output under the parity matrix;
3. reversed input order followed by inverse permutation restoring identical output;
4. one measured compilation using `lower(...).compile()`;
5. device transfer completed before execution timing;
6. one untimed compiled execution;
7. twenty synchronized warm executions, each completed with `block_until_ready()`; and
8. all twenty durations plus median, nearest-rank empirical p95
   (`sorted[ceil(0.95*n)-1]`), and scenarios/second recorded locally.

Run the benchmark in a fresh worker process. Record Python/platform, backend/devices,
NumPy, JAX/jaxlib, TensorFlow, Flax, Waymax commit, config/adapter fingerprints, code
commit, horizon, batch size, and process peak RSS. Label RSS as process high-water
memory, not JAX device memory. Do not claim accelerator speedup, scaling efficiency, or
batch-128 execution.

The 20-transition Waymax IDM kernel must also pass `jax.jit` on one selected state.
IDM `vmap` is optional because its dense 128-slot pairwise geometry is an expected local
CPU risk and the exact-log path already provides the substantive batching gate.

## 8. Implementation surface

Expected changes are confined to:

- `evalsim/sources/waymax_cohort.py` for pure scan-event/manifest/selector types;
- `evalsim/sources/waymax_loader.py` for exact ten-shard resolution, classified raw
  streaming, and safe in-process digest reuse;
- `evalsim/simulators/waymax_reference.py` for lazy environments, compact scans,
  control masks, and rollout conversion;
- `evalsim/sources/waymax_m4_cli.py` and a locked console entry point;
- unit, contract, synthetic Waymax-fixture, safe-CLI, and opt-in ten-shard tests;
- `docs/data/womd-waymax-m4-crosswalk.md`;
- this plan, README, roadmap, presentation, claim ledger, and role matrix; and
- existing notice/package surfaces only if archive checks reveal a gap.

No core contract field changes are planned. If M4 discovers that the existing
`Scenario` or `Rollout` cannot express a required supported semantic, execution stops
for an additive contract plan and review rather than hiding tensors or arrays in
metadata.

No WOMD payload scan may run from uncommitted implementation code. After this plan is
accepted and committed, M4 code is implemented and tested only with synthetic/in-memory
fixtures, independently reviewed, committed as a clean pre-execution implementation
commit, and pushed. The local command rejects any tracked/staged/untracked non-ignored
change, records exact `HEAD`, and binds the manifest/results to:

- the Git commit and tree IDs;
- `uv.lock`;
- the plan;
- scanner/selector/loader/reference/CLI source files; and
- the canonical selector-config fingerprint covering suffixes, fields, four reason
  codes and priority, map rule, hash encodings/domains, quotas/fallback, parity fields,
  tolerances, vmap pair, and IDM subset/control/horizon.

Before final release, verify that the executable-source fingerprint is unchanged from
the pre-execution commit; otherwise rerun the complete local acceptance from a new clean
commit. Documentation-only evidence closure may follow without pretending it was the
executed code commit.

The command is gated by `EVALSIM_RUN_WAYMO_LOCAL=1`, accepts an explicit checkout/data
root, resolves only the hard-coded ten suffixes, writes only under ignored
`outputs/m4/`, fails if that path is not Git-ignored, and prints no native IDs, digests,
coordinates, trajectories, map values, or absolute paths.

## 9. Verification

### Core-only and optional-runtime tests

- exact resolver rejects missing, ambiguous, duplicate, and out-of-range shards;
- shard `00010` and arbitrary extra files never enter the selected paths;
- raw/preprocess/event counters are independently maintained, one event follows every
  raw ordinal, repeat raw counts match, and fatal stream/adapter/parity errors stay fatal;
- event/manifest schemas reject duplicate IDs, invalid locators, unknown codes,
  noncanonical order, non-finite values, count drift, and tampering;
- the exact four source-rejection predicates, byte encoding, domain-separated hash
  ranks, quotas, redistribution, fallback floor, IDM subset, and vmap-pair selection use
  invented IDs only;
- complete manifest round-trip is byte-stable and exclusive-create;
- lazy imports preserve the core-only install;
- rollout conversion tests cover padding, ID/slot mismatch, births, disappearances,
  invalid fill, current boundary, circular yaw, and provenance;
- exact-log synthetic fixture is contradicted one field at a time, including emitted
  `timestamp_micros`, to prove the oracle is independent;
- waypoint-following IDM control/fallback masks exclude SDC, nonvehicles, invalid
  transitions, and initialized overlaps;
- an independent NumPy overlap oracle matches the pinned frame-0, no-validity-mask
  exclusion on adversarial boxes;
- free-road and close-leader scalar IDM oracles prove nonzero controlled execution and
  expected acceleration direction;
- compact scan matches five stock Waymax transitions on a synthetic fixture and one on
  the first selected real scene;
- sequential/vmapped/JIT output and permutation invariance pass on synthetic fixed-shape
  fixtures;
- CLI help works from an installed wheel outside the checkout; unsafe roots/output paths
  fail before optional imports or data reads; and
- wheel/sdist contain `NOTICE.md` and no Waymax source, data, local manifest, or output.

### Opt-in local acceptance

The explicit local command must:

1. reject a dirty tracked tree and bind execution to the clean pre-execution commit;
2. verify the exact ten files and ignore `00010`;
3. scan every raw record twice to clean EOF with equal raw/decode/event counters;
4. produce byte-identical local manifests;
5. reload all selected locators;
6. pass M3 conversion and independent parity for the full cohort;
7. run all three EvalSim policies for 80 transitions on the full cohort;
8. run compact Waymax exact log playback for 80 transitions on the full cohort;
9. convert every reference output to `Rollout` and pass the supported matrix;
10. run and exactly repeat the pre-registered Waymax IDM subset/horizon with positive
    controlled-transition/non-fallback evidence;
11. pass sequential/JIT/vmap/permutation checks;
12. write a sanitized local summary, aggregate construction report, and benchmark; and
13. exit nonzero on any silent drop, unclassified failure, selected-path failure,
    unexplained discrepancy,
    privacy failure, or resource-gate failure.

After code changes, run the clean core-only suite, the locked Waymo-extra suite, the
explicit local marker, the standalone M4 command, site build/checks, archive inspection,
and staged-content audit. Test counts are evidence only after rerun.

## 10. Privacy, license, and claims

The existing WOD attribution, exact Waymax notice/citation, canonical and pinned license
links, and personal non-commercial restriction remain required in `NOTICE.md`, before
optional installation instructions, in package archives, and directly in the deployed
presentation. No new dependency or general project license is introduced in M4.

Allowed claim after every gate passes:

> Built a deterministic local WOMD cohort from exactly ten validation shards and
> cross-checked EvalSim's exact log-playback mapping and rollout-contract semantics
> against pinned Waymax/JAX on Apple CPU, with measured two-scene `jit`/`vmap`
> execution; compared a separately scoped privileged Waymax waypoint-following IDM
> reference without equating it to causal EvalSim IDM.

The claim must state the actual cohort count and actual IDM subset/horizon, disclose
that both paths share the pinned Waymax decode, and label the cohort a complete-case
conditional sample of the exact first ten validation shards. It must not claim random or
representative sampling, metric parity, statistical realism conclusions, causal Waymax
IDM, full-dataset scale, accelerator execution, production deployment, commercial use,
or learned evaluation.

## 11. Rollback and release

M4 is additive. Rollback is a normal Git revert of the milestone commit followed by the
locked core/optional tests. It does not delete raw data, local outputs, environments, or
caches; force push; change access; or mutate the M3 contract.

Release follows the project nine-step workflow:

1. adversarially accept this plan before payload scan;
2. commit and push the accepted pre-registration before implementation;
3. implement without WOMD payload reads, verify with synthetic fixtures, obtain a
   pre-execution code review, then commit and push a clean implementation;
4. run the complete local acceptance only from that exact clean commit;
5. obtain independent execution and publication reviews;
6. update all public evidence surfaces without local data and verify the executable
   fingerprint did not change;
7. audit the staged tree and package/site archives and create the evidence-closure
   commit;
8. push GitHub and the exact Sites source state, then deploy a saved version with the
   existing owner-only access; and
9. verify remote commit and deployment before closing the final checkbox.

## 12. Acceptance checklist

- [x] Two adversarial plan reviews are accepted with no unresolved blocker.
- [ ] Accepted pre-registration and clean pre-execution implementation commits are
  pushed before any WOMD payload scan.
- [ ] Exact ten-shard scan has zero silent drops and a clean EOF for every shard.
- [ ] Independent repeat scan produces a byte-identical local manifest.
- [ ] Frozen cohort meets the target or the pre-registered fallback floor.
- [ ] Every selected locator reloads with exact identity/provenance.
- [ ] M3 conversion and independent parity pass for the full cohort.
- [ ] EvalSim replay, CV, and IDM run deterministically for the full cohort.
- [ ] Waymax exact-log reference passes the full 80-transition parity matrix.
- [ ] Waymax waypoint-following IDM passes the declared nested subset/horizon.
- [ ] Explicit two-scene `jit`/`vmap` and permutation gates pass.
- [ ] Compile/warm timing and process peak RSS are recorded accurately.
- [ ] Core-only, Waymo-extra, and opt-in local suites pass.
- [ ] Crosswalk and public limitations match the supported evidence.
- [ ] No dataset, identity, local manifest/output, private material, or secret is tracked.
- [ ] Wheel, sdist, and site archives contain notices and no forbidden artifacts.
- [ ] Adversarial execution/publication reviews have no unresolved blocker.
- [ ] Milestone commit is pushed and the owner-only presentation deploy is verified.

## 13. Pre-registration record

Two independent reviews rejected the first draft before any M4 payload scan:

- the architecture review required the exact frame-0/no-validity initialized-overlap
  semantics, direct timestamp emissions, multi-step stock-rollout equivalence,
  non-vacuous analytic IDM integration oracles, positive control accounting, and a
  correction to the roadmap's causal-language restriction; and
- the methods/privacy review rejected adapter/parity failures as eligibility filters,
  tautological scan accounting, the shard-biased/vacuous IDM subset, underspecified
  hashes/fields/discrepancy rules, dirty-tree execution, and missing public
  cohort-construction accounting.

The plan now makes adapter/contract/parity failures fatal, counts raw records before
preprocessing, freezes exact source predicates and byte encodings, separately hash-ranks
a post-overlap IDM subset, binds execution to a clean reviewed commit, requires
privacy-safe aggregate exclusion reporting, and narrows every claim. Both reviewers
then returned **ACCEPTED — no unresolved blocker**.

During execution, append only factual deviations, local aggregate evidence, adversarial
verdicts, and release verification. Never rewrite the selector or acceptance gates as
though a post-result choice had been pre-registered.

### Pre-execution implementation record

Before any M4 WOMD payload access, the synthetic-only implementation passed independent
reviews of four boundaries:

- population selection, classified streaming, exact-shard safety, and grouped selected
  reload: **ACCEPTED**;
- compact Waymax exact-log/IDM execution, overlap oracle, conversion, and semantic
  crosswalk: **ACCEPTED**;
- the local acceptance command, after an initial blocking review and correction of
  actor-level IDM evidence, conversion auditing, output publication order, checkout
  binding, terminal privacy, and contradiction coverage: **ACCEPTED**; and
- optional-dependency packaging, installed-wheel help, notices, archive contents, and
  core-only imports: **ACCEPTED**.

The final pre-execution snapshot passed 297 non-local tests in the locked Waymo-extra
environment. A separate locked core-only environment, with JAX, jaxlib, TensorFlow,
Flax, and Waymax absent, passed 242 tests with 17 expected optional-runtime skips.
The dedicated M4 cohort, loader, reference, and CLI suites account for 127 passing
synthetic tests. Wheel and sdist inspection found the required `NOTICE.md` and no WOMD,
TFRecord, Parquet, generated output, private input, cache, or vendored Waymax package.

These are implementation-readiness facts, not WOMD execution evidence. The payload
boundary remains closed until this exact reviewed implementation is committed, pushed,
and the worktree is clean.

### Execution deviation and selector-v2 correction — 2026-07-28

The first bound local acceptance ran from clean, pushed commit
`1e294ee82427e8b622ceb28df351053975728cb6`. It stopped before event emission
with fatal code `audit_shape_or_dtype_drift`; `event_emitted` remained zero and
`clean_eof` was false. The command created only the ignored
`execution-provenance.json`; it emitted no scan event or manifest and performed
no cohort selection, policy/reference execution, benchmark, or comparative
evaluation. This failed attempt is not M4 acceptance evidence, and no public M4
claim was unlocked.

Source review identified an implementation error at the audit boundary. M4 was
taking source-audit arrays from the jitted Waymax postprocess result. With JAX
x64 disabled in the locked environment, decoded `int64` fields could reach the
audit as narrowed `int32` arrays. The selector was therefore validating a
post-JAX representation rather than the pre-registered pre-JAX source arrays.

A separate process deviation occurred during follow-up diagnosis. Because the
agents shared one working tree, one single-record structural diagnostic
unknowingly imported an agent's uncommitted source-predicate revision. This
violated the clean-commit payload gate, so the diagnostic is excluded from all
acceptance evidence. It exposed no native identity, locator, field value,
coordinate, trajectory, policy output, metric value, or comparative result.
Payload work stopped immediately and the clean-commit gate was reclosed.

Before rerun, the selector contract is corrected from version 1 to version 2.
Selector v2 captures and freezes the native identity and audit arrays from the
eager pre-JAX mapping before Waymax postprocessing, validates the exact pinned
pre-JAX shapes and dtypes, and applies explicit lossless normalization for
encoded identities/types, binary masks, and timestamps. Dimension invariants
apply only to valid frames, matching the original pre-registration; invalid-frame
payload does not affect eligibility. Waymax state construction continues through
the same pinned postprocess.

This is a representation-boundary defect correction, not an outcome-based
selector amendment. The failed bound run emitted no event, and neither check
produced cohort membership, policy output, metric output, or comparative
results that could be used to tune selection. The four rejection predicates
and their priority, supported-map rule, ranking byte encodings and domain
strings, quotas, redistribution, fallback floor, and execution scopes remain
unchanged. Because the audited representation and normalization are part of
the selector contract, selector v2 receives a new configuration fingerprint
and executable-source fingerprint. The dataset configuration and ranking
domains remain unchanged; commit `1e294ee` is superseded as the executable
snapshot for the next acceptance attempt.

The payload gate remains closed until selector-v2 code, tests, this correction,
and the corrected crosswalk pass the locked synthetic/core suites and
independent adversarial review; are committed and pushed from a clean tree;
and local `HEAD` is verified equal to `origin/main`. The rerun must use that
exact commit and fingerprints, start from record zero in a new ignored output
directory, and perform both complete scans of exactly shards `00000`–`00009`
through clean EOF. It must not resume or reuse the failed attempt, and the
failed ignored output is retained rather than deleted. Any later executable or
selector-fingerprint change requires another fresh full rerun.
