# WOMD/Waymax → EvalSim M3 mapping

**Status:** Implemented and locally accepted
**Dataset profile:** WOMD v1.3.1 TFExample validation, 10 past + 1 current + 80 future
**Waymax revision:** `a64dfec9be8576b60d9cecc94f406d9812d4a7d0`
**Adapter schema:** `1` / adapter version `0.1.0`

This document defines the exact semantic boundary for M3. It does not imply that all
WOMD fields are represented. It identifies what EvalSim preserves, how it transforms
those fields, and what it deliberately defers.

Primary references:

- [WOMD TFExample schema](https://waymo.com/intl/fil/open/data/motion/tfexample/)
- [Pinned Waymax configuration](https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/waymax/config.py)
- [Waymax dataloader API](https://waymo-research.github.io/waymax/docs/autoapi/waymax/dataloader/index.html)
- [Official native scenario-ID loader pattern](https://waymo-research.github.io/waymax/docs/notebooks/wosac_submission_via_waymax.html)

## Reader boundary

M3 resolves exactly one local file by the suffix
`tfrecord-00000-of-00150`. It does not use Waymax's `@150` expansion and does not glob
every file in the directory. The reader starts from `WOD_1_3_1_VALIDATION` and freezes
all material settings:

| Setting | M3 value |
|---|---|
| repeat | `1` |
| batch dimensions | none |
| shuffle | disabled (`shuffle_seed=None`; unused buffer size pinned to `1000`) |
| loader shards | `1` |
| deterministic | `true` |
| distributed | `false` |
| drop remainder | `false` |
| TF data service | none |
| batch by scenario | `true` |
| aggregate timesteps | `true` |
| maximum objects | `128` |
| maximum roadgraph points | `30000` |
| include SDC paths | `true` |
| paths / points per path | `45` / `800` |

Waymax's default loader omits `scenario/id`. EvalSim follows the official WOSAC
pattern: it adds the string feature before parsing, converts the ID to a byte array for
the JAX-compatible generator, and decodes it after iteration. The native ID is stored
only in `Scenario.scenario_id`; it is never duplicated in metadata or printed by the
acceptance command.

## Time and initialization

Waymax aggregates WOMD fields in `past`, `current`, `future` order. The locked temporal
profile is:

```text
past_steps=10
current_steps=1
future_steps=80
init_steps=11
horizon=91
current_index=past_steps + current_steps - 1 = 10
```

A freshly decoded `SimulatorState.timestep` is zero, so it is not used as the logged
current-frame boundary. The independent source audit also requires Waymax
`state/which_time` to be exactly 10 past markers (`-1`), one current marker (`0`), and
80 future markers (`1`), with the sole current marker at index 10. The adapter receives
the immutable profile, validates the horizon and
`init_steps == current_index + 1`, and rejects profile drift.

For each timestep, the source timestamp is the exact consensus of
`state/all/timestamp_micros` among retained objects valid at that step. Every step must
have at least one contributor, and the resulting timeline must be strictly increasing.
EvalSim converts microseconds to seconds and normalizes the first supported frame to
zero.

## Agent mapping

| WOMD / Waymax | EvalSim | Rule |
|---|---|---|
| fixed object slot | position in `Scenario.agents` | Drop only never-valid padding; preserve remaining slot order. |
| `state/id` / `ObjectMetadata.ids` | `Agent.id` | Preserve the integer value; retained IDs must be unique. |
| type `1` | `vehicle` | Direct mapping. |
| type `2` | `pedestrian` | Direct mapping. |
| type `3` | `cyclist` | Direct mapping. |
| unset/other | `unknown` | Explicit fallback, counted in metadata. |
| `state/is_sdc` | `Scenario.ego_index` | Require exactly one retained SDC. |
| `state/all/valid` | `Agent.valid` | Preserve exactly. |
| X/Y | `Agent.x`, `Agent.y` | Copy valid float32 values into float64 arrays. |
| velocity X/Y | `Agent.vx`, `Agent.vy` | Copy valid float32 values into float64 arrays. |
| bounding-box yaw | `Agent.heading` | Waymax preprocessing wraps yaw; the adapter normalizes to `[-π, π)` and verifies circular parity. |
| invalid numeric payloads | numeric series | Replace with deterministic finite zeroes while validity remains false. |
| length/width | scalar `Agent.length/width` | Waymax computes a validity-masked mean and broadcasts it over time. Require the broadcast state value to be finite, positive, and constant; independently reproduce the masked mean from the preprocessed TF tensors. |

The following object metadata is available at the source boundary but intentionally not
stored in M3's motion contract:

- `tracks_to_predict` / `is_modeled`;
- `objects_of_interest`;
- `is_controlled`;
- difficulty and other evaluation roles;
- Z position and height.

M4 adds typed role/control eligibility keyed by object ID before any Waymax parity or
route-aware control claim.

## Map mapping

Waymax exposes `RoadgraphPoints`, not ready-made polylines. EvalSim groups valid samples
by native feature ID in first-occurrence order and preserves the sample order within
each group.

| Source type IDs | M3 target |
|---|---|
| `0, 1, 2, 3` | `MapType.LANE` |
| `14, 15, 16` | `MapType.ROAD_EDGE` |
| all others | omitted with type/count accounting |

A lane or road-edge group is retained only when:

- it has at least two distinct finite XY points;
- every consecutive segment is longer than `1e-6 m` and no longer than `0.75 m`;
- every non-terminal source direction is non-zero; and
- its segment tangent is within 10 degrees of the corresponding source direction.

The final point has no outgoing segment, so its direction is intentionally ignored.
EvalSim never repairs order using nearest-neighbor reconstruction. Groups with mixed
types, unsupported types, bad spacing, non-finite values, or misaligned directions are
omitted with stable reason/count metadata.

M3 intentionally omits crosswalk polygons. TFExample polygon sampling differs from the
documented 0.5 m polyline sampling, and the existing contract cannot preserve stable
feature identity or polygon semantics. Stop signs are never relabeled as stop lines.
Stable feature IDs/subtypes and polygon/path semantics move to M4's typed map context.

## Other source context intentionally deferred

| Source context | M3 disposition | Planned consumer |
|---|---|---|
| traffic-light timeline and lane association | parsed by Waymax, not placed in `Scenario` | M5 signalized slices/metrics |
| SDC path samples / route compatibility | parsed consistently, not placed in `Scenario` | M4 route-aware parity |
| roadgraph Z and 3-D directions | omitted | only if a later evaluator requires 3-D |
| road-line subtype, speed limit, stop sign, speed bump | omitted with accounting | M4 typed map context |
| sensor/camera/media tensors | not present in WOMD motion TFExample | separate M9 sidecar/data gate |

No source tensor, role mask, route, traffic-light array, or absolute local path is hidden
inside JSON metadata.

## Provenance

Every converted local scenario carries JSON-native provenance:

- dataset source/version/split;
- adapter version, schema, immutable Waymax commit, and adapter-rule fingerprint;
- independent dataset-config fingerprint;
- exact five-digit shard suffix and local record ordinal;
- local shard SHA-256;
- combined source fingerprint;
- temporal, coordinate, time-origin, invalid-fill, and dimension rules; and
- agent/map conversion and omission counts.

The config fingerprint is computed from the actual runtime `DatasetConfig`, excluding
only its local path. Every upstream dataclass field is accounted for and compared with
the locked canonical values before reading; a new field or changed runtime value fails
closed instead of retaining a stale hand-written fingerprint.

Raw data, converted Parquet, acceptance JSON, and visualization PNG remain ignored under
`outputs/m3/`.

## Independent acceptance

The local acceptance path reconstructs expected values directly from preprocessed TF
tensors and raw Waymax leaves without calling adapter conversion helpers:

- IDs, ordering, types, SDC, masks, directly copied float32 values, map source order,
  feature IDs/types, and timestamps in microseconds are checked exactly;
- normalized timestamps, independent dimension means, and circular yaw use
  `rtol=0, atol=1e-6`;
- at least one gated lane or road-edge group must remain;
- at least one non-SDC vehicle must be valid across current → next;
- log replay is exact; a moving CV first transition and a nonzero, distinct IDM vehicle
  first transition both match an independent scalar point-mass oracle; and the IDM
  vehicle-control branch—not its nonvehicle fallback—executes.

The committed tests use generated in-memory Waymax/JAX datatypes. The real-data test is
both marked `waymo_local` and gated by `EVALSIM_RUN_WAYMO_LOCAL=1`.

The record selector is the earliest within the first 32 that passes source conversion,
retains a supported map group, and has both the SDC and a non-SDC vehicle valid from the
current frame to the next frame. Full-future SDC validity remains a downstream rollout
acceptance requirement; it is not used to scan ahead for a different record.
