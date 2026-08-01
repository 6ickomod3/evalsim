# M7 metrics-first construct-audit amendment

**Date:** 2026-07-31

**Status:** Implemented and verified bounded construct audit; original v1 gates remain unmet

**Amends:** [M7 evaluator red-team v1](2026-07-31-m7-evaluator-red-team.md)

**Governing strategy:** [metrics-first learning strategy](2026-07-31-metrics-first-strategy.md)

**Result:** [sanitized construct-audit result](../results/m7-construct-audit.md)

## Decision

Implement a small analytic construct audit, not a new held-out benchmark.

Two independent reviews found that a defensible sealed held-out study would require a
new source generator, identity manifest, calibration receipt, code lock, one-shot state
machine, cohort-retirement rules, and crash semantics. That machinery is feasible but
does not answer the interview project's main question: what each metric measures, why
the metrics disagree, and what can be concluded from the three M5 baselines.

The accepted M5 128-scenario comparison remains the project's empirical result. This M7
amendment adds only a transparent teaching artifact that checks metric field-of-view and
corrects one generator error. A calibrated source-disjoint or real-scene validation study
is optional future work and is not needed for the current presentation.

## Disclosure and known development evidence

This amendment is outcome-aware. The metrics, defects, and original development results
were selected before it was written. It cannot independently discover or validate a
metric.

The existing three-case development matrix uses four prevalence doses
`(0.25, 0.50, 0.75, 1.00)`, clean log-copy rollouts, and a `1e-6` cohort-mean response
tolerance:

| Existing v1 defect | Position | Overlap | Kinematic infeasibility |
|---|---|---|---|
| frozen agent | detects | misses | misses |
| position-only teleport | detects | misses | misses |
| velocity-only spike | misses | misses | detects |
| forced overlap | detects | detects | misses |

All clean means are zero. Flat curves are labelled monotone by the old harness, so
monotonicity alone is not sensitivity evidence.

### Generator error found during review

`FrozenAgentDefect` v1 zeros velocity beginning at `current_index`, the last observed
frame. It therefore changes history and removes the future deceleration that the
kinematic metric should see. The v1 implementation and matrix remain as explicitly
historical development evidence.

The construct audit adds `FrozenAgentDefectV2`. It preserves every rollout field through
`current_index`; for the selected world-agent prefix it holds x, y, and heading at the
current value and sets vx and vy to zero only from `current_index + 1` onward. Selection
remains `ceil(dose * number_of_world_agents)`, with dose zero selecting none. Every
selected agent's validity is preserved; every unselected agent and every non-series
rollout field is copied exactly. The transform is pure and deterministic, does not mutate
either input, and the four positive doses affect exactly `(1, 2, 3, 4)` world agents.

This correction changes the frozen-by-kinematic expectation from “misses” to “detects.”
Finding and correcting that error is itself a result: defect generators can teach an
evaluator test to the answer just as metrics can be gamed.

## Exact analytic cases

Production code will expose exactly three deterministic contract-only cases, indexed
`0, 1, 2`:

- float64 timestamps from `np.arange(6, dtype=np.float64) * 0.1`;
  scenario ID `m7-construct-{index}`; `ego_index=0`; `current_index=1`;
  metadata exactly `{"source": "m7_construct",
  "current_index": 1, "case_index": index}`; and `map=[]`;
- one fully valid ego vehicle, ID 0, type vehicle, stationary at
  `x=-100-index, y=0`, with zero heading/vx/vy, length 2 m, and width 2 m;
- four fully valid world vehicles with `k=0..3`, ID `10+k`, and type vehicle;
- world vehicle `k` has float64 `x[t]=(2+k)*timestamps[t]+index`,
  `y[t]=50*k`, `vx[t]=2+k`, `vy[t]=0`, and `heading[t]=0`, with
  length 2 m and width 2 m; and
- the clean rollout is an exact defensive copy with `sim_name="m7_construct_clean"`,
  `sim_version="1.0.0"`, seed 0, `perturbation=None`, and empty metadata. It preserves
  timestamps, validity, IDs, types, dimensions, and every series without shared buffers.

The cases contain no WOMD, Waymax, JAX, TensorFlow, private material, or generated
dataset. They are analytic fixtures, not traffic diversity.

## Frozen audit

Metrics:

1. `position_error_m` v1.0.0;
2. `oriented_box_overlap_rate` v1.0.0; and
3. `waymax_kinematic_infeasibility_rate` v1.0.1.

Defects:

1. future-only `frozen_agent` v2;
2. position-only +50 m `teleportation` v1;
3. velocity-only +100 m/s `kinematic_spike` v1; and
4. position/heading-only `overlap` v1.

Use corrected-audit doses exactly `(0.25, 0.50, 0.75, 1.00)`. At each dose, compute the
valid per-case metric value, then the three-case arithmetic mean. Every clean and
injected result must have `valid=True` and
`eligible_components == total_components == 16` (four world agents times four future
frames/transitions). A positive-control cell responds only when **every** dose mean
exceeds its clean mean by more than `1e-6`, with no adverse delta below `-1e-6`. An
expected non-response matches only when every dose obeys
`abs(dose_mean - clean_mean) <= 1e-6`. A positive response curve is monotone only when
every adjacent mean obeys `value[j+1] >= value[j] - 1e-6` and
`max(value)-min(value) > 1e-6`. Invalid or partial
values, zero affected counts at positive dose, duplicate/missing cells, or a version
mismatch fail the audit rather than being counted as misses.

Expected corrected matrix:

| Corrected defect | Position | Overlap | Kinematic infeasibility |
|---|---|---|---|
| future-only freeze v2 | responds | no response | responds |
| position-only teleport v1 | responds | no response | no response |
| velocity-only spike v1 | no response | no response | responds |
| forced overlap v1 | responds | responds | no response |

This is six expected responses and six expected non-responses. All twelve values and all
four-dose curves must be emitted; the implementation must not report only successful
cells.

The output is a pure immutable in-memory result, not a generated file: exactly twelve
cells in defect-family then metric-name order. Each cell contains defect family/version,
metric name/version, clean mean, four `(dose, mean)` pairs, `responds`, and
`monotone_sensitive`. The top-level result also reports exact response/non-response
counts and whether the frozen matrix matched.

## Invariance and non-tautology checks

On all three analytic cases and all three metrics, agent permutation with seed 4 and
rigid translation by `(dx=5.0, dy=-3.0)` of scenario, rollout, and map geometry must
preserve values within `1e-6`. Add the kinematic metric to the existing invariance test.
Correct `TranslationProbe` so its “whole scene” claim includes `MapPolyline.xy`.
Because the audit cases intentionally use `map=[]`, a separate fixture contains one
lane polyline `[[0.0, 0.0], [1.0, 0.0]]`; its translated xy must equal the original plus
`(5.0, -3.0)` while type/order and both inputs remain unchanged.

On every analytic case, the existing rollout-only translation by `(5.0, -3.0)` must
change position error above `1e-6`. A rollout-only velocity control leaves the scenario
unchanged and adds +100 m/s vx to the first world agent at
`min(current + max(1, (num_steps - 1 - current) // 2), num_steps - 1)`; it must change
kinematic infeasibility above `1e-6` on every case. The forced-overlap oracle must first
prove directly that the selected future boxes share x/y/heading coordinates, independent
of the overlap metric, and then change overlap above `1e-6` on every case. These checks
establish that the three paths are not vacuous; they are not real-world validity evidence.

## Acceptance and claim

Acceptance requires:

1. v1 remains available and its historical tests still pass;
2. v2 preserves history/current state exactly, does not mutate the source rollout, and
   its returned dose-zero rollout is fieldwise identical across the entire contract;
   its manifests retain family/version and exact nested affected ordinals/counts;
3. the three analytic cases match the formulas above;
4. the complete corrected 6-response/6-non-response matrix and non-flat monotone positive
   curves match the frozen expectation;
5. all selected invariance and non-tautology checks pass; and
6. targeted tests, the full test suite, site build/check, and dataset/private-artifact
   audit pass; and
7. README, roadmap, claim ledger, and site status use the corrected construct-audit
   boundary and do not describe it as held-out validation.

Permitted claim:

> On three hand-built analytic cases, the corrected controlled probes produced the
> reported six-response/six-non-response matrix. This verifies the implementation's
> intended field-of-view and demonstrates complementary blind spots: position error
> cannot see a velocity-only corruption, the fixed-step kinematic diagnostic cannot see
> a position-only corruption, and overlap only responds to interpenetration in these
> cases.

Always add:

> This is outcome-aware construct evidence, not calibrated, source-disjoint, held-out,
> WOMD-backed, population-level, or general metric validation.

Do not compute an aggregate realism score or metric winner. Do not infer anything about
`speed_error_mps` from this audit. The original M7 v1 gates remain unmet.

## Deferred extension

If future work needs a validated synthetic or real-scene stress claim, create a separate
pre-registration with exact source construction, calibration/null controls, sealed
partitions, write-once execution, cohort retirement, fixed-cohort uncertainty, and
licensed-data review. It does not inherit validity from this construct audit.
