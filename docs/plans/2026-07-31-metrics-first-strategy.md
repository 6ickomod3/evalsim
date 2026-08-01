# EvalSim — Metrics-first learning strategy

**Date:** 2026-07-31

**Status:** Accepted project strategy; in-source presentation redesign completed
(`index.html` and structural checks); release/deployment remain separate

**Decision owner:** repository owner

**Related evidence:** [M5 official acceptance](../results/m5-official-acceptance.md),
[M7 evaluator red-team](2026-07-31-m7-evaluator-red-team.md),
[M7 outcome-aware construct-audit amendment](2026-07-31-m7-metrics-first-amendment.md),
the [construct-audit result](../results/m7-construct-audit.md), and the
[technical evidence roadmap](2026-07-28-waymo-aligned-roadmap.md)

## 1. Decision

EvalSim is a personal, **metrics-first learning project**. Its current center is the
accepted M5 experiment: three deliberately simple comparison baselines evaluated through
four primary diagnostics that ask different questions and produce conflicting orderings.

The project will lead with the reasoning behind the evaluators—what each metric measures,
rewards, and misses—rather than milestone count, technology breadth, or official-runner
complexity. The accepted M5 evidence remains unchanged. M6, M7, and later ideas are
extensions that must earn their own claims.

This document governs current project positioning, learning priority, and presentation
information architecture. The Waymo-aligned roadmap remains the technical evidence plan.
Historical pre-registrations, accepted result reports, schemas, and crosswalks retain
their original semantics and must not be rewritten to fit the new narrative.

## 2. Core question and thesis

**Question:** When log replay, constant velocity, and IDM are evaluated on the same
scenes, do distinct diagnostic metrics support one consistent simulator ordering?

**Observed answer on the accepted M5 cohort:** No. Logged-position and logged-speed
fidelity favor replay, the overlap proxy also has the lowest mean for replay, and the
fixed-step kinematic diagnostic favors CV and IDM over replay. Current adjusted evidence
does not support a directional IDM--CV conclusion on overlap or fixed-step kinematics.

**Thesis:** Simulator quality is not supported as one scalar by this evidence. Metric
choice encodes a preference about which behavior matters, so conclusions must remain
metric- and use-case-specific.

The bounded conclusion is:

> On the fixed 128-scene complete-case conditional cohort, no baseline is universally
> best. Replay is best at reproducing recorded trajectories; CV matches the log better
> than IDM among the two history-only baselines; and CV and IDM have lower fixed-step
> kinematic-infeasibility rates than replay. IDM does not have a supported overall
> advantage over CV. These findings support a multi-metric evaluation method, not a
> winner, causal superiority, safety conclusion, or WOMD-population claim.

## 3. Deliberately simple comparison probes

The three baselines are calibration probes with legible failure modes, not candidate
production systems:

| Baseline | Capability isolated | Known limitation |
|---|---|---|
| Log replay | Exact recorded-future construction oracle | Privileged and nonreactive |
| Constant velocity | Minimal history-only causal extrapolation | No interaction or map following |
| EvalSim IDM | History-only longitudinal bumper-gap response | No lateral control; geometric rather than route-aware leader selection |

Simple baselines are intentional: evaluator behavior is easier to understand when model
failure modes are known. New simulator complexity is not a current priority unless it
answers a distinct evaluation question.

## 4. Primary metric story

The public story uses the existing four pre-registered M5 primary metrics, grouped into
three lenses. Their formulas, versions, eligibility, and accepted results do not change.

| Lens | Metric | Question | Required interpretation boundary |
|---|---|---|---|
| Fidelity | `position_error_m` | How far is simulated position from the recorded future? | Favors privileged replay and does not measure reactivity |
| Fidelity | `speed_error_mps` | How different is scalar speed from the recorded future? | Omits signed direction, position, and interaction quality |
| Interaction proxy | `oriented_box_overlap_rate` | How often does a target box interpenetrate another valid object? | Not collision-pair count, severity, safety, or useful motion |
| Feasibility diagnostic | `waymax_kinematic_infeasibility_rate` | How often does a vehicle transition violate pinned acceleration/curvature thresholds? | Fixed `0.1 s` Waymax semantic, vehicle-only, and not physical-time-normalized safety |

The remaining nine M5 metrics and eight source-only slices remain part of the complete
evidence ledger and advanced appendix. They are not removed from the implementation or
accepted result.

Every metric explanation must answer four questions:

1. What construct is this metric intended to observe?
2. Why is that construct decision-relevant?
3. What behavior does the metric reward or penalize by construction?
4. Which plausible defects or use cases can it miss?

## 5. Evaluation reasoning to present

The main presentation must make these design decisions visible:

- all policies run on the same source scenarios;
- eligibility and slice membership are source-only, before policy outcomes;
- frame/agent components reduce to one scalar per scenario;
- the scenario is the paired comparison and resampling unit;
- missingness, sparse effects, nulls, and contradictions remain visible;
- metric definitions and primary gates were frozen before official outcome access;
- synthetic analytic oracles test computation and known defects, while the fixed WOMD
  cohort supplies bounded real-scene application evidence; and
- native parity checks computation and semantics, not ground truth or construct validity.

The evaluator evidence is discussed in three layers:

1. **Implementation correctness:** was the metric computed as specified?
2. **Construct validity:** does it detect the intended behavior for the intended reason?
3. **Decision relevance:** does that construct answer the downstream AV evaluation
   question?

M5 provides strong evidence for the first layer and a bounded fixed-cohort application.
The M7 outcome-aware construct audit supplies narrower field-of-view, wiring, and
generator-audit evidence for the second layer; it does not establish general construct
validity. Real counterfactual decision relevance remains unaccepted.

## 6. Presentation information architecture

The main presentation should follow this argument:

1. AV simulator evaluation is a decision problem, not a leaderboard.
2. Three simple baselines isolate known behaviors.
3. Four primary metrics ask different questions and have explicit blind spots.
4. Same-scene pairing prevents scenario-mix differences from driving comparisons.
5. The accepted result reverses ordering across metric families.
6. Controlled defects expose field-separated blind spots and show how a flawed
   generator can create the wrong evaluator conclusion.
7. The conclusion is use-case-specific; the remaining evidence gaps are explicit.

Milestone history, full 12-cell statistics, all metrics/slices, architecture internals,
Waymax/JAX details, result-store machinery, exact row domains, and licensing remain
available as advanced evidence or appendix material. Required Waymo/Waymax notice text
must remain directly visible in any deployed presentation.

## 7. Current status and learning priority

- **M3–M5:** accepted historical implementation and empirical evidence at their bound
  source snapshots.
- **M6:** pre-data implementation exists, but no official WOMD scientific result is
  accepted. Because this strategy adoption changes `AGENTS.md`, the prior exact-source
  security review does not automatically approve the new source snapshot; fresh review,
  approval, and the required tag are mandatory before any official M6 data access.
- **M7:** the data-free defect, detection-matrix, metric-card, and invariance foundation
  is implemented. The accepted outcome-aware construct-audit amendment adds three exact
  analytic cases and a corrected four-defect × three-metric matrix: six cells respond
  and six do not across four doses. It verifies intended field-of-view and wiring only.
  It is not calibrated, source-disjoint, held-out, WOMD-backed, or general metric
  validation, and it does not retroactively satisfy the original v1 gates.
- **M8 and later:** optional learning extensions, not committed current deliverables or
  current claims.

The current learning priority is:

1. make the four M5 primary metrics and their conflicting conclusions understandable;
2. use the corrected M7 construct audit to explain evaluator field-of-view, blind spots,
   and why generator validity matters;
3. keep broader calibrated, held-out, or WOMD stress validation optional unless a later
   decision question requires it; and
4. resume M6 only as a distinct counterfactual-learning question, not to complete a
   milestone checklist.

## 8. Compute and environment strategy

The project is CPU-first. Linux and GPU are not requirements for the core learning path,
the data-free metric demonstrations, the public presentation, or the accepted aggregate
result discussion.

- Core contracts, baselines, rollout, metrics, statistics, and synthetic defect tests use
  the repository's CPython 3.11.5/NumPy environment.
- Optional WOMD reproduction requires the licensed local shards and pinned
  TensorFlow/Waymax/JAX runtime, but the accepted path ran on Apple Silicon CPU.
- GPU or Linux environments become relevant only for a separately approved learned
  evaluator, large-scale performance study, or video/VLM extension.

Do not claim generic cross-platform support, production throughput, accelerator speedup,
or that every optional upstream environment works without its own verification.

## 9. Implemented presentation redesign

The current source now:

- rewrites the README and presentation around the argument in section 6;
- adds four metric cards and a compact four-row result summary;
- retains the complete accepted 12-cell matrix in a collapsed engineering appendix and
  linked report;
- includes the compact corrected six-response/six-non-response M7 construct matrix;
- moves milestone, architecture, JAX/Waymax, and official-runner details behind the
  engineering appendix; and
- synchronizes status wording and structural site checks.

This records the local source state only. Commit, push, deployment, and post-release
verification remain separate workflow steps.

An optional later usability layer may expose a small contract-first evaluation façade,
but it must preserve `MetricResult`, eligibility, missingness, units, distributions, and
pairing. It must never label exploratory output as official evidence.

## 10. Explicit non-goals

- No change to accepted M5 metric formulas, versions, directions, eligibility, slices,
  statistics, stored rows, or conclusions.
- No composite score, universal realism rank, policy winner, or simulator-superiority
  claim.
- No new WOMD access or result rerun merely to simplify the presentation.
- No claim that overlap is safety, kinematic infeasibility is complete physical realism,
  Waymax is ground truth, or finite-cohort bands are confidence intervals/tests.
- No deletion of Waymax/JAX, official-runner, result-store, or M6 evidence machinery.
- No claim of an accepted M6 outcome or fully validated M7 evaluator suite.
- No Linux, GPU, distributed, learned-model, video/VLM, or production-scale requirement
  for the current core project.

## 11. Acceptance gates

The strategy and later presentation are accepted only when:

1. README, presentation, roadmap, claim ledger, and study plan use consistent status and
   claim boundaries.
2. The main story uses exactly the three comparison baselines, four canonical primary
   metric IDs, and three lenses without changing scientific semantics.
3. The complete M5 12-cell primary result remains discoverable; the two unsupported
   directional comparisons are not hidden.
4. No composite score or overall winner is introduced.
5. M6 is labeled pre-data/no accepted WOMD result. M7 is labeled an outcome-aware,
   data-free construct audit with original v1 gates unmet, never calibrated or held-out
   validation.
6. CPU-first wording does not imply universal platform support or production scale.
7. Required Waymo/Waymax notices, data-publication boundaries, and non-commercial scope
   remain prominent.
8. Full Python tests, site build/structural checks, link checks, and public-claim review
   pass.
9. The working tree and staged-file audit contain no dataset, generated experiment
   artifact, private material, secret, or unrelated change.
10. Any later official M6 attempt obtains fresh exact-source approval after this
    strategy/`AGENTS.md` change.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Simplification hides contradictory evidence | Keep the four-row summary honest and the full 12-cell matrix directly discoverable |
| Metrics-first becomes metrics-only | Retain scenarios, rollout generation, and counterfactual evaluation as producer/context layers |
| A friendly metric name changes semantics | Keep canonical metric IDs and versions; friendly labels are explanatory only |
| Learning/WIP wording erases completed evidence | Say “core M5 comparison complete; validation/optional extensions remain” |
| Historical acceptance appears to cover new code | Bind every claim to its accepted snapshot and require fresh M6 approval |
| A simpler M7 story rewrites known outcomes | Preserve historical v1, disclose the v1 freeze-generator artifact, and label the corrected construct audit outcome-aware |
| Advanced evidence becomes undiscoverable | Preserve direct links to technical roadmap, crosswalks, acceptance report, and limitations |
| Reordered presentation buries license notices | Keep direct notice text and installation warnings, regardless of page order |

## 13. Verification and rollback

The strategy adoption itself created no empirical result. Its documentation and
presentation are verified through repository tests, site checks, link/status review,
and release audit; the later M7 construct audit has separate code, tests, and a bounded
result record.

Rollback is a normal Git revert of the strategy/documentation change. Historical M5
evidence and local-only data/results are not modified or deleted.
