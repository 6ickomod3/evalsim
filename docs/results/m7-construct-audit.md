# M7 outcome-aware construct-audit result

**Date:** 2026-07-31

**Scope:** Three hand-built, contract-only analytic cases; four controlled defect
families; four prevalence doses; three selected M5 metrics.

**Plan:** [M7 construct-audit amendment](../plans/2026-07-31-m7-metrics-first-amendment.md)

## Result

The corrected audit produced the frozen complete matrix: **six responding cells and six
exact non-responses**. Every responding curve was nondecreasing and globally non-flat.
All clean means were zero, and every clean and corrupted result was valid with all
16 expected components eligible.

| Controlled probe | `position_error_m` | `oriented_box_overlap_rate` | `waymax_kinematic_infeasibility_rate` |
|---|---|---|---|
| Future-only freeze v2 | responds | no response | responds |
| Position-only teleport v1 | responds | no response | no response |
| Velocity-only spike v1 | no response | no response | responds |
| Forced overlap v1 | responds | responds | no response |

The doses are `[0.25, 0.50, 0.75, 1.00]`. Curves below are three-case arithmetic
means, shown to six decimal places:

| Controlled probe | Metric | Four-dose means |
|---|---|---|
| Future-only freeze v2 | position error | `[0.125000, 0.312500, 0.562500, 0.875000]` |
| Future-only freeze v2 | overlap rate | `[0, 0, 0, 0]` |
| Future-only freeze v2 | kinematic infeasibility | `[0.062500, 0.125000, 0.187500, 0.250000]` |
| Position-only teleport v1 | position error | `[9.375000, 18.750000, 28.125000, 37.500000]` |
| Position-only teleport v1 | overlap rate | `[0, 0, 0, 0]` |
| Position-only teleport v1 | kinematic infeasibility | `[0, 0, 0, 0]` |
| Velocity-only spike v1 | position error | `[0, 0, 0, 0]` |
| Velocity-only spike v1 | overlap rate | `[0, 0, 0, 0]` |
| Velocity-only spike v1 | kinematic infeasibility | `[0.125000, 0.250000, 0.375000, 0.500000]` |
| Forced overlap v1 | position error | `[12.500337, 37.501012, 75.002025, 75.002025]` |
| Forced overlap v1 | overlap rate | `[0.500000, 0.750000, 1.000000, 1.000000]` |
| Forced overlap v1 | kinematic infeasibility | `[0, 0, 0, 0]` |

The plateau in the forced-overlap curves at the two largest doses is retained rather
than hidden; both curves still satisfy the frozen nondecreasing, globally non-flat
criterion.

## What changed during review

Frozen-agent v1 zeroed velocity at the last observed/current frame. That altered history
and removed the current-to-future deceleration, so its apparent kinematic non-response
was a **generator artifact**. Future-only freeze v2 preserves history through the
current frame, then creates an abrupt stop. The fixed-step kinematic metric responds to
that abrupt transition.

This does **not** show that the metric detects generic nonreactivity. It shows that this
specific controlled abrupt stop enters the metric's velocity-derived field of view.

## Bounded interpretation

> On three hand-built analytic cases, the corrected controlled probes produced the
> reported six-response/six-non-response matrix. This verifies the implementation's
> intended field-of-view and demonstrates complementary blind spots: position error
> cannot see a velocity-only corruption, the fixed-step kinematic diagnostic cannot see
> a position-only corruption, and overlap only responds to interpenetration in these
> cases.

The field-separated non-responses are construct and wiring evidence. They are partly
true by construction and do not validate the metrics as general realism, safety, or
decision-quality measures.

This audit is **outcome-aware**. It is not calibrated, source-disjoint, held-out,
WOMD-backed, population-level, or general metric validation. It does not evaluate
`speed_error_mps`, support an aggregate realism score, or identify a metric or simulator
winner. The original M7 v1 acceptance gates remain unmet; broader validation is optional
future work with its own pre-registration and evidence boundary.
