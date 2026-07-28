# M3 implementation plan — local Waymo vertical slice

**Date:** 2026-07-28
**Status:** ✅ Accepted, released, and post-release verified on 2026-07-28
**Milestone:** M3 — WOMD TFRecord → Waymax/JAX → EvalSim → Parquet → visualization

This plan is the M3 execution record required by the repository's milestone delivery
workflow. It must be adversarially reviewed before dependency or implementation changes.
Comparative results discovered after implementation must not be used to silently weaken
the gates below.

The repository owner has explicitly characterized the work as personal,
non-commercial interview preparation. The implementation records that stated purpose
but does not make a legal determination that a particular use qualifies under either
upstream agreement. The public technical purpose is education, research, and personal
experimentation; the project must not become a paid service, hiring assessment product,
production system, or real-world vehicle-development/validation tool. If the actual use
changes, Waymax/WOMD work stops pending a fresh license review.

## 1. Pre-registration

### Hypothesis

An immutable Waymax revision and a compatible, pinned JAX/TensorFlow stack can read one
real scenario from local WOMD v1.3.1 validation shard `00000` on Apple Silicon CPU and
convert the motion and coarse-map fields EvalSim intentionally supports into the
existing `Scenario` contract deterministically.

The converted scenario should pass the existing contract and Parquet round-trip,
visualize locally, and execute through the existing rollout engine without requiring
Waymax types downstream of the source adapter.

### Falsification criteria

M3 is not accepted if any of the following remains true:

- the pinned optional environment cannot resolve, import, or run a basic JAX `jit` on
  this Apple Silicon CPU;
- the reader cannot decode a native WOMD scenario ID from shard `00000`;
- the adapter substitutes a record ordinal or local filename for the native scenario ID;
- the adapter cannot account for the observation/future boundary, SDC identity, active
  object ordering, validity masks, supported trajectory fields, or every retained map
  point;
- repeated conversion of the same record is not deterministic;
- the supported Waymax fields disagree with the EvalSim conversion outside declared
  tolerances;
- the converted scenario fails contract, Parquet, visualization, log-replay, CV, or IDM
  smoke checks;
- any raw/converted dataset payload or generated real-data artifact is staged, committed,
  pushed, or deployed;
- required Waymo Open Dataset or Waymax attribution/use notices are absent from the
  release;
- the core package cannot still import and pass its normal tests without the optional
  Waymo dependencies; or
- a new public project or résumé claim exceeds the verified evidence.

### Fixed local inputs

- Platform: Apple Silicon macOS, Python 3.11, local CPU.
- Required M3 input: the exact local file whose shard suffix is
  `tfrecord-00000-of-00150`.
- Reference set reserved for M4: shard suffixes `00000`–`00009`, selected explicitly.
- Additional local shard `00010` is out of scope and must never enter a population
  through a directory glob.
- Raw data remains immutable and ignored under
  `data/raw/womd/v1.3.1/tf_example/validation/`.

### Baseline evidence

Before implementation:

- Git `main` is clean and synchronized with `origin/main` at `0ff8003`.
- The core test suite passes: **134 passed**.
- The presentation build and structural checks pass: **21 IDs and 7 links**.
- The ten required shard suffixes `00000`–`00009` are present and ignored by Git.
- The existing environment does not contain JAX, TensorFlow, Flax, or Waymax.

### Scope

M3 will:

- perform an isolated compatibility spike before changing the repository environment;
- add a pinned optional `waymo` dependency set and commit its lockfile;
- preserve the native WOMD scenario ID using the documented custom parsing pattern;
- implement a lazy optional-dependency `WaymaxSource`;
- implement a pure Waymax-state-to-`Scenario` conversion boundary;
- map the existing supported agent motion, SDC identity, time boundary, and coarse map;
- record small, JSON-safe provenance and conversion-accounting metadata;
- add synthetic Waymax-shaped unit/contract fixtures generated in memory;
- add a marked, opt-in local integration test and a one-scenario smoke command;
- produce ignored local Parquet and visualization outputs for acceptance;
- add the required non-commercial-use attribution and license notices; and
- update public documentation only after the evidence gates pass.

### Explicit non-goals

M3 will not:

- scan all ten shards, freeze a cohort, benchmark batching, or claim population parity
  (M4);
- add route-aware execution, Waymax reference rollouts, or JAX batching (M4);
- implement traffic-light, route, object-role, or metric semantics before a consuming
  milestone defines their typed contract (M4/M5);
- implement scorecards, uncertainty, counterfactuals, or stress tests (M5–M7);
- upload data or outputs, use paid cloud, use a Mac GPU, or publish a real-data fixture;
- redistribute unmodified Waymax source; or
- use Waymax source, documentation, or derivatives to train, fine-tune, distill, or
  otherwise develop or improve a foundation or generative AI model; or
- describe intentionally unsupported WOMD fields as preserved.

## 2. Supported-field contract

M3 keeps the current `Scenario` schema unchanged. This is an additive adapter, not a
schema migration. “Semantic preservation” in the acceptance gate applies only to the
fields listed as supported below. Every other field must be explicitly counted or
documented as intentionally unsupported.

| Source semantic | M3 EvalSim representation | Rule |
|---|---|---|
| WOMD scenario ID | `Scenario.scenario_id` | Decode and preserve the native ID unchanged. |
| Object slot/order | `Scenario.agents` | Drop only never-valid padding; preserve remaining Waymax slot order. |
| Object ID | `Agent.id` | Preserve native integer ID; reject duplicates among retained objects. |
| Object type | `Agent.type` | Vehicle/pedestrian/cyclist map directly; all other values map to `UNKNOWN`. |
| SDC identity | `Scenario.ego_index` | Require exactly one typed SDC among retained objects. |
| Timestamp | `Scenario.timestamps` | Build each canonical step from exact consensus across retained objects valid at that step, require at least one contributor, convert microseconds to seconds, and normalize to the first frame. Require every retained object's valid timestamp to match that consensus. |
| Past/current/future boundary | `metadata["current_index"]` | Derive from the locked WOMD profile and Waymax environment initialization semantics; validate 10 past + 1 current, never infer it from raw `SimulatorState.timestep == 0`. |
| Validity | `Agent.valid` | Preserve exact supported-frame masks. |
| Position/yaw/velocity | `Agent.x/y/heading/vx/vy` | Preserve valid values; wrap heading consistently. Fill invalid payload positions with deterministic finite zeroes while leaving validity false. |
| Object dimensions | `Agent.length/width` | Preserve Waymax's broadcast, validity-masked mean dimensions after verifying they are finite, positive, and constant in the state. Independently audit the preprocessed TF dict against the documented raw-dimension rule; do not claim access to original per-frame dimensions from `SimulatorState`. |
| Lane centers | `MapPolyline(LANE)` | Group valid points by feature ID in source array order, but retain a polyline only when continuity and source-direction alignment gates pass. |
| Road edges | `MapPolyline(ROAD_EDGE)` | Use the same continuity/alignment gate and preserve source order. EvalSim must remove its unsupported claim that every producer orients road edges with drivable surface on the left. |
| Crosswalks | Intentionally unsupported | Count and omit in M3. TFExample polygon sampling is not the same as the 0.5 m polyline sampling, so support waits for a typed polygon-aware map contract. |
| Provenance | `Scenario.metadata` | JSON-native scalars/strings only: source/version/split, shard suffix, ordinal, adapter/config/schema versions, coordinate/time units, fingerprint, and conversion counts/rules. No absolute paths or tensors. |

Intentionally unsupported in M3:

- Z/height and 3-D roadgraph directions;
- stable map feature IDs, detailed lane/road-line subtypes, speed limits, stop signs, and
  speed bumps;
- traffic-light timelines and lane associations;
- SDC route/path samples;
- modeled/object-of-interest/difficulty roles;
- sensor or media payloads;
- fixed-array padding and meaningless invalid-frame payload values.

A WOMD stop-sign point must not be mislabeled as an EvalSim `STOP_LINE`. Unsupported or
ambiguous map groups must be omitted with counts and reason codes; `UNKNOWN` is allowed
only when retaining their geometry has a specified downstream use.

Waymax exposes sampled points, not a `MapPolyline` guarantee. For lane centers and road
edges, each group must contain at least two distinct points; every consecutive segment
must be longer than `1e-6 m` and no longer than `0.75 m`; and each non-terminal segment
tangent must align with its source direction vector within 10 degrees. A zero terminal
direction is allowed. These thresholds follow the documented 0.5 m sampling while
allowing endpoint and floating-point tolerance. The adapter must omit and account for a
group that fails, and must never repair ordering by nearest-neighbor guessing. The
independent reference comparison operates on retained
`(feature_id, type, source_indices, xy)` tuples so map acceptance is not merely a visual
judgment.

Typed roles, stable map identity/subtypes, and routes are deferred to M4. Typed
traffic-light timelines are deferred to M5. M6 ego interventions belong at the rollout
engine/controller seam rather than in `Scenario`.

The pure translator receives an immutable temporal profile alongside the state. It
validates `past_steps + current_steps + future_steps == state horizon` and derives
`current_index = past_steps + current_steps - 1`. M3 rejects batched states rather than
silently squeezing an axis.

## 3. Dependency and compatibility plan

### Immutable upstream

Pin Waymax to commit:

```text
a64dfec9be8576b60d9cecc94f406d9812d4a7d0
```

Never resolve a floating `main` branch.

### Candidate compatibility matrix

Test this exact matrix first in an isolated temporary uv environment:

```text
Python       3.11
NumPy        1.26.4
JAX          0.4.38
jaxlib       0.4.38
TensorFlow   2.18.1
Flax         0.10.4
Waymax       a64dfec9be8576b60d9cecc94f406d9812d4a7d0
```

The spike must print Python/platform and all package versions, report a CPU JAX backend
and device, and run one compiled JAX operation before reading data. It must then import
Waymax, build the exact shard config, decode one record, and report only non-sensitive
structural facts.

If the matrix fails, change one pin at a time and record the failure and replacement.
Do not float to latest packages. Only the evidence-backed final matrix enters the
`waymo` optional dependency group and `uv.lock`.

Installing the optional group may downgrade NumPy in the shared repository environment.
That is acceptable only if uv resolves it cleanly and all existing tests remain green.
The environment must not be deleted or rebuilt destructively.

### Import isolation

JAX, TensorFlow, Flax, and Waymax imports must stay inside the optional source/CLI
boundary. A normal `import evalsim` and the core test suite must work when the `waymo`
extra is absent. Missing optional dependencies must produce a concise installation
instruction instead of an unrelated import traceback.

## 4. Reader and adapter design

### Dataset configuration

Start from Waymax's locked WOMD v1.3.1 validation profile and replace it with an exact
single-file configuration:

- path is the explicitly resolved `00000` shard, never `@150`;
- `repeat=1`;
- `batch_dims=()`;
- `shuffle_seed=None`;
- `shuffle_buffer_size=1000` (pinned and fingerprinted even though shuffling is
  disabled);
- `num_shards=1`;
- `deterministic=True`;
- `drop_remainder=False`;
- `distributed=False`;
- `tf_data_service_address=None`;
- `batch_by_scenario=True`;
- `max_num_objects=128`;
- `aggregate_timesteps=True`;
- `max_num_rg_points=30000`;
- `include_sdc_paths=True`;
- `num_paths=45`;
- `num_points_per_path=800`;
- `data_format=TFRECORD`;
- SDC paths parsed consistently even though M3 does not place them in `Scenario`.

Waymax's `@N` expansion expects all sequential files. M4 must therefore iterate the ten
exact shard paths or implement a verified explicit-file-list reader; it must not pretend
the local subset is a complete `@10` dataset.

### Native scenario identity

The default Waymax high-level loader omits `scenario/id`. Follow the official custom
WOSAC loader pattern:

1. construct the normal feature description;
2. add the fixed-length `scenario/id` string feature;
3. parse with TensorFlow;
4. pop and decode the ID separately;
5. run the normal Waymax preprocessing and state construction; and
6. pair the decoded native ID with the resulting `SimulatorState`.

The record ordinal and shard suffix remain separate provenance fields.

### Time boundary

The raw loader state starts with `state.timestep == 0`; that is not the WOMD “current”
frame. The locked profile concatenates 10 past, 1 current, and 80 future frames. Waymax
environment initialization uses 11 initialization steps and resets the simulation
timestep to 10.

The adapter must therefore derive `current_index` from validated profile counts and
environment initialization configuration, assert the expected relationship, and write
the derived value to metadata. Tests must fail if those settings drift. Code must not
contain an unexplained naked `10`.

### Conversion boundaries

Keep three independently testable layers:

1. path/profile validation and exact-shard selection;
2. WOMD TFExample → `(native_scenario_id, preprocessed audit view, Waymax
   SimulatorState)`; and
3. pure Waymax `SimulatorState` → EvalSim `Scenario`.

The small audit view is source-boundary-only and is used to independently verify raw
dimension reduction, timestamps, IDs/masks, and retained roadgraph tuples. It must not
escape into `Scenario.metadata`. Everything downstream receives only the EvalSim
contract. The translator must not read local files or mutate a Waymax state.

## 5. Verification matrix

### Normal test suite, no real data

Use in-memory Python builders for small deterministic Waymax datatypes. Do not commit a
serialized upstream test record.

Tests must cover:

1. active-slot filtering, stable ordering, unique IDs, and exact one-SDC validation;
2. direct and unknown type mappings;
3. exact validity masks, finite invalid fills, yaw normalization, scalar dimension
   selection, variation accounting, and non-aliased arrays;
4. SDC-derived timestamp units/normalization/monotonicity, valid-object timestamp
   consensus, and derived `current_index`;
5. roadgraph grouping, point order, supported type crosswalk, invalid samples, and
   unsupported-feature accounting;
6. JSON-safe provenance with no absolute path or tensor payload;
7. deterministic repeat conversion;
8. `Scenario` → Parquet → `Scenario` equality;
9. log-replay preservation plus CV and IDM smoke execution; and
10. custom `scenario/id` preservation through a temporary in-memory
    `tf.train.Example`;
11. explicit rejection of batched states and drifted temporal/config profiles;
12. distinct shard-content, adapter, and dataset-config fingerprints; and
13. the base package's behavior when optional dependencies are missing.

Register the `waymo_local` marker in project configuration. In addition to unit
isolation tests, create a clean core-only temporary environment from the locked base and
`dev` dependencies and run an actual `import evalsim` plus the normal test suite without
the `waymo` extra.

### Opt-in local integration

A marker alone is not an opt-in because ordinary pytest still collects and executes
marked tests. Local tests must additionally skip unless
`EVALSIM_RUN_WAYMO_LOCAL=1` is set. All Waymax/TensorFlow/JAX imports remain lazy so
collection without the extra is safe.

The fixed selection rule is the earliest record within the first 32 records of shard
`00000` that passes source-contract validation, retains at least one gated lane or
road-edge map group, and has the SDC plus at least one non-SDC vehicle valid at both the
current and next frame. This is an eligibility rule, not a performance filter. If no
such record exists, M3 stops for plan revision rather than searching farther after
seeing comparative policy output.

The gated local test and smoke command must:

- resolve only shard `00000`;
- select and record the deterministic first-eligible ordinal;
- preserve the native scenario ID internally but print only a boolean identity check,
  never the raw ID;
- compare a reference summary built directly from the preprocessed TF dict and raw
  Waymax leaves—without calling adapter conversion helpers—against EvalSim for SDC
  identity, retained object indices/order, exact `state/which_time` 10/1/80 partition,
  horizon, current boundary, valid masks, supported trajectories, dimensions, and
  retained map tuples;
- convert twice and compare deterministically;
- assert IDs, masks, timestamp microseconds, source indices, feature/type IDs, and
  directly copied float32 trajectory/map/dimension values exactly after the declared
  dtype conversion; compare only transformed normalized timestamps and circular yaw
  with `rtol=0, atol=1e-6`;
- assert at least one real current→next non-SDC vehicle transition, execute that
  transition through log replay, CV, and IDM, verify moving CV and nonzero/distinct IDM
  first transitions against an independent scalar integration oracle, and verify the
  IDM vehicle-control branch (not its fallback) controlled that agent;
- verify `outputs/m3/` with `git check-ignore` before writing;
- round-trip through an ignored `outputs/m3/` Parquet file;
- render an ignored `outputs/m3/` PNG with the observed/current boundary visible; and
- emit an ignored `outputs/m3/` JSON acceptance summary without raw IDs, coordinates,
  trajectories, map points, or absolute paths.

The integration path must skip with a clear reason when the optional environment or
local shard is absent. It must never run implicitly in ordinary CI.

### Required commands

The final release audit must include:

```bash
uv run pytest
EVALSIM_RUN_WAYMO_LOCAL=1 uv run --extra waymo pytest -m waymo_local
EVALSIM_RUN_WAYMO_LOCAL=1 uv run --extra waymo evalsim-waymax-smoke --shard 00000
node scripts/build-site.mjs
node scripts/check-site.mjs
```

Exact CLI syntax may change during implementation, but the same evidence must be
produced.

## 6. License and publication gate

Before M3 code is pushed:

- add a root `NOTICE.md` containing the exact Waymo Open Dataset software attribution
  required by the March 2025 terms and the exact Waymax Derivative IP notice plus full
  Waymax citation;
- link both the canonical Waymax license URL required by its prescribed notice and the
  immutable license at the pinned commit for provenance;
- state in `NOTICE.md` that this combined M3+ distribution is subject to the upstream
  conditions to the extent applicable, including downstream Waymax compliance,
  non-commercial use, no real-world vehicle development/testing, no Production Systems,
  and the foundation-model restriction;
- state that EvalSim's own copyright remains with its owner and that the notice does not
  silently grant MIT, Apache, open-source, or other rights to EvalSim;
- put a prominent short restriction and link before the README's Waymo installation
  command, because installing the optional extra downloads Waymax and requires the user
  to accept its agreement;
- embed the exact WOD attribution, exact prescribed Waymax notice, full Waymax citation,
  applicable restriction language, and canonical/pinned links directly in `index.html`,
  because deployment is a separate public copy and cannot rely on a repository notice
  that is not deployed with it;
- configure project packaging so `NOTICE.md` ships in both wheel and sdist, build both
  archives, and inspect their file lists during release audit;
- reference the pinned Waymax license directly rather than vendoring unmodified Waymax
  source, documentation, license text, wheel, or cache;
- state that WOMD data is not included and access remains governed by its terms; and
- do not publish native scenario IDs, real trajectories/coordinates/map snippets,
  real-WOMD images, dataset payloads, or generated real-data outputs.

No new general project license will be selected during M3; that is a separate owner
decision. The release documentation is a conservative compliance record, not legal
advice or a claim that every possible interview-related use qualifies. If the required
conditions cannot be expressed consistently in every conveyed surface, public
push/deploy is blocked even if the local code works.

## 7. Evidence, documentation, and claim control

Local ignored outputs may include:

- a compatibility report;
- one acceptance-summary JSON;
- one converted Parquet;
- one visualization PNG; and
- test/cache output.

All M3 outputs must live under `outputs/m3/`, and the writer must fail closed unless that
directory is confirmed ignored. Raw IDs and source-derived values may exist inside the
local Parquet needed to prove round-trip preservation, but no terminal/chat summary,
tracked documentation, presentation, commit, or deployed artifact may expose them.

Tracked documentation must include:

- this reviewed plan;
- a field-by-field WOMD/Waymax mapping and omission crosswalk;
- dependency and upstream revision provenance;
- setup and one-scenario smoke instructions;
- the local evidence summary without dataset payloads;
- known limitations and M4 handoff decisions;
- updated README, roadmap status, presentation, and claim ledger; and
- required license/attribution notices.

The M3 claim is allowed only after acceptance:

> Integrated one local WOMD v1.3.1 scenario through pinned Waymax/JAX into a validated,
> simulator-neutral EvalSim contract on Apple Silicon CPU.

It must not imply cohort scale, Waymax rollout parity, learned evaluation, production
deployment, or commercial use.

## 8. Rollback

M3 changes must remain confined to:

- optional dependency/lock configuration;
- the source adapter and smoke entry point;
- synthetic/optional/local tests;
- mapping, license, and progress documentation; and
- a small visualization enhancement if needed for the time boundary.

Rollback consists of reverting the milestone commit. No core contract migration, raw
data change, cache deletion, force push, or environment deletion is permitted. The
pre-M3 `uv.lock` hash and `uv pip freeze` snapshot must be saved locally before syncing
the extra. After a Git revert, restore the reverted lock with `uv sync --frozen --extra
dev`, compare the package snapshot, and run the baseline tests. The pre-M3 core test path
must continue to work throughout implementation.

## 9. Acceptance checklist

M3 is accepted only when all boxes can be supported by reviewed evidence:

- [x] Plan received adversarial review and all blocking findings were resolved.
- [x] Exact pinned optional dependency stack resolves and runs on Apple Silicon CPU.
- [x] Native ID-preserving reader loads one real record from exact shard `00000`.
- [x] Supported-field mapping and every intentional omission are documented.
- [x] Conversion and Parquet round-trip are deterministic.
- [x] Independent Waymax/EvalSim structural and numeric checks pass.
- [x] Visualization and all three EvalSim policy smokes pass.
- [x] Core tests pass without Waymo extras; full tests pass with them.
- [x] No dataset, private material, generated output, cache, or secret is tracked.
- [x] Required license notices and non-commercial restrictions are present.
- [x] Wheel and sdist both contain `NOTICE.md`; neither contains Waymax source or data.
- [x] README, roadmap, presentation, claim ledger, and limitations match the evidence.
- [x] Adversarial execution review has no unresolved acceptance blocker.
- [x] Milestone-scoped commit is pushed and the presentation deploy is verified.

## 10. Adversarial plan-review record

Two independent reviews challenged the pre-implementation plan:

- the architecture/semantics review rejected unproved map-polyline semantics, an
  impossible post-Waymax dimension rule, marker-only local-test gating, tautological
  parity, permissive global-coordinate tolerance, vacuous policy/map acceptance, and an
  incomplete environment rollback;
- the license/data-safety review required concrete notice surfaces, direct deployed-site
  text, package-archive inspection, output privacy, no silently selected project
  license, and explicit upstream use restrictions.

The plan was revised to address each blocking finding and both reviewers subsequently
returned **ACCEPTED**. This records plan acceptance only; it is not implementation
evidence or legal advice.

## 11. Local execution evidence

The following gates passed on the stated Apple Silicon CPU environment without
publishing source-derived values:

- isolated compatibility spike: Python 3.11.5, NumPy 1.26.4, JAX/jaxlib 0.4.38,
  TensorFlow 2.18.1, Flax 0.10.4, and pinned Waymax
  `a64dfec9be8576b60d9cecc94f406d9812d4a7d0`;
- JAX backend/device: CPU, with a compiled operation returning its expected result;
- clean core-only environment: 152 passed, 3 optional skips;
- locked repository environment: 170 passed, 1 gated local-data skip;
- explicitly opted-in local integration: 1 passed, 170 deselected;
- standalone M3 smoke: passed independent supported-field parity, deterministic
  conversion, Parquet round-trip, current-boundary rendering, exact log replay, a real
  constant-velocity world-agent transition, an IDM vehicle-branch transition, and JAX
  CPU `jit`; and
- local ignored artifacts remained confined to `outputs/m3/`.

The acceptance report does not print or publish the native scenario identity,
coordinates, trajectories, map samples, absolute data paths, or the real-data image.
Archive, staged-content, and adversarial execution audits passed.

### Release and post-release verification

The milestone implementation was committed as
`94541a2318d4f4a2d7ae5ec164f4a4c7fc35a654`, pushed to the GitHub `main` branch, and
pushed as the exact source state for Sites version 4. The validated presentation archive
was built from that source and deployed successfully to the existing owner-only
presentation at
[`evalsim-policy-lab.jdai2013.chatgpt.site`](https://evalsim-policy-lab.jdai2013.chatgpt.site/).
The Sites deployment reported success without changing the existing access policy.
This execution record is completed in a follow-up documentation-only commit so the
release gate is never claimed before it happens.

### Post-registration execution deviation

The first implementation of the selector required SDC validity from the current frame
through the entire future horizon, while the pre-registered rule required SDC validity
only at current and next. The implementation was stricter, and the selected record
satisfied both rules, so the observed selection did not change. The execution reviewer
nevertheless treated the mismatch as a release blocker.

Before release, the selector was corrected to the declared current→next rule and a
regression assertion was added. Full-future SDC validity remains a downstream rollout
acceptance gate: if the earliest selected record cannot run, M3 fails instead of
silently scanning ahead. The local acceptance was rerun after this correction. This
deviation is recorded rather than retroactively describing the original execution as
perfectly pre-registered.
