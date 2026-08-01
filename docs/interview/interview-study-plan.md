# Interview study plan

This plan turns current evidence—and optional extensions only if pursued—into
interview-ready technical arguments. It is organized by evidence loops rather than a
milestone checklist or calendar weeks.

## Current metrics-first priority

The presentation should be a technical argument, not a list of finished milestones.
Start with three simple baselines, explain the four primary M5 diagnostics and their
assumptions, show how their rankings conflict on the fixed 128-scene conditional cohort,
and use M7's controlled defects to explain evaluator blind spots. State explicitly that this is a
personal learning project and work in progress, but place that status after the core
question rather than using it as the opening. M6 is a pre-data extension; M8--M11 are
optional topics rather than required completion targets.

## The repeatable loop

For each topic:

1. implement one narrow vertical slice;
2. write down the hypothesis and falsification condition;
3. derive or identify an independent oracle;
4. run the experiment and retain negative results;
5. explain the design aloud in ten minutes;
6. answer an adversarial follow-up without looking at the code;
7. update the claim ledger only after the evidence gate passes.

## Study tracks

### 1. WOMD, Waymax, and JAX semantics

Study the accepted M3–M4 evidence. Be able to explain:

- WOMD history/current/future boundaries, validity masks, object roles, roadgraph, and
  traffic-light semantics;
- why EvalSim keeps a substrate-neutral contract;
- what Waymax provides that the NumPy engine does not;
- coordinate, padding, mask, and eligibility mismatches;
- `jit`, `vmap`, static shapes, compilation cost, host/device transfer, PRNG discipline,
  and why local CPU is acceptable for development.

Use the accepted M4 result as a bounded case study: explain why 2,916 raw records became
1,527 eligible records and a deterministic 128-scenario complete-case cohort; why that
cohort is conditional rather than representative; why a shared Waymax decode limits
independence; why the privileged logged-trajectory waypoint-following Waymax IDM reference was restricted to 16 scenes × 20
transitions; and why a batch-two exact-log kernel benchmark is not end-to-end or
production-scale throughput. Do not turn successful execution into a claim about
realism; M5 supplies a separate metric and statistical evidence gate.

### 2. Metric and statistical design

Study the accepted M5 result. Be able to explain:

- realism versus downstream usefulness;
- construct validity, units, eligibility, missingness, and aggregation;
- why incompatible axes should not become one score;
- paired scenario comparisons and deterministic scenario-level resampling;
- rare events, small slices, pre-registered versus exploratory analysis, and multiple
  testing;
- how a release decision changes when metrics disagree.

Use the official ten-shard result as the case study: all 12 pre-registered primary
`all`-slice cells retained paired `n = 128`, all 312 scorecards were independently
reconstructed, and bounded native Waymax checks passed with zero tolerance-gate
failures or binary mismatches. Explain why 10 of 12 multiplicity-adjusted
scene-reweighting bands excluding zero still does **not** produce an overall winner
when metric families disagree; why the other two cells must remain visible; why
logged-position parity is tolerance-bounded rather than bit-exact; and why the fixed
`0.1 s` kinematic diagnostic is not physical-time-normalized. Keep the conclusion
conditional on the deterministic
complete-case cohort: the bands are not confidence intervals or hypothesis tests, and
the result is not WOMD-population inference, causal evidence, or a policy replacement
decision.

### 3. Implemented pre-data extension — closed-loop causal evaluation

Study the implemented M6 design without claiming a WOMD result. Be able to explain:

- why log replay is invalid as a reactive reference after ego perturbation;
- open-loop fidelity versus closed-loop usefulness;
- treatment, control, eligibility, paired seeds, and leakage;
- nonreactivity versus overreaction;
- what causal language the experiment supports and what it does not.

### 4. Evaluating the evaluator

Use the implemented M7 foundation and outcome-aware construct audit. Be able to explain:

- why the v1 frozen-agent/kinematic miss was a generator artifact: changing velocity at
  the current frame erased the future deceleration;
- why future-only freeze v2 makes the fixed-step kinematic diagnostic respond to one
  abrupt stop, without proving detection of generic nonreactivity;
- the corrected six-response/six-non-response matrix across three analytic cases and
  four doses;
- why position missing velocity-only corruption and kinematic infeasibility missing
  position-only corruption verify field-of-view and wiring, but are partly tautological
  and do not establish general construct validity;
- why the overlap response is specific to forced interpenetration in these cases;
- Goodharting, generator validity, non-vacuity controls, invariance, sensitivity,
  monotonicity, and evaluator versioning; and
- why this outcome-aware audit is not calibrated, source-disjoint, held-out,
  WOMD-backed, or population evidence. Calibration, false-positive analysis, sealed
  held-out defects, and real-scene stress tests are optional future validation work, not
  completed evidence.

### 5. Optional if pursued — learned discriminators

If M8 is separately chosen, be able to explain:

- scenario-grouped splits and simulator-fingerprint leakage;
- why logistic/feature baselines precede a Flax temporal model;
- calibration versus ranking;
- AUROC versus AUPRC, Brier/ECE, uncertainty, and slice/OOD behavior;
- generator-held-out evaluation and shortcut audits;
- what a discriminator learns when “real” and “simulated” preprocessing differ.

### 6. Optional if pursued — multimodal/video/VLM evaluation

If M9 is separately chosen, be able to explain:

- temporal stability, geometric consistency, cross-modal consistency, and condition
  following when ground truth is incomplete;
- why BEV semantic video is not camera realism;
- how to validate a VLM judge without treating it as truth;
- randomized paired prompts, prompt sensitivity, positional bias, abstention, and human
  audit;
- how motion evaluation and real sensor evaluation share governance but require
  different data and contracts.

### 7. Optional project extension — scale and production

If M10–M11 are separately chosen, study the following; use real professional examples,
not this solo project, for organizational questions:

- streaming, batching, caching, backpressure, resume/retry, deduplication, and lineage;
- measured compile time, throughput, memory, utilization, and cost;
- when a local CPU, GPU, TPU, or distributed path is justified;
- interfaces, ownership, SLOs, observability, model/evaluator registries, and rollout
  gates;
- an architecture decision rejected after evidence;
- how work would be divided, reviewed, and mentored on a team.

## Core mock-interview questions

1. Define “realistic” and “useful.” When do they diverge?
2. What evidence would falsify each metric?
3. How do you prevent Goodharting?
4. Could simulator B replace A for a particular AV-development task? Make a decision.
5. Why is scenario the sampling unit for uncertainty?
6. How do you handle rare-event slices and multiple comparisons?
7. How do you know a learned evaluator is not detecting source artifacts?
8. What happens under geography, behavior, weather, or generator shift?
9. Why retain both EvalSim and Waymax paths?
10. How would you validate a VLM-based evaluator?
11. What scale did you actually run, and what was the bottleneck?
12. What would the production interfaces, owners, gates, and observability be?

## Five stories to prepare

- **Integration judgment:** moving Waymo tooling forward after disproving a cloud-only
  assumption while preserving dependency inversion.
- **Semantic discrepancy:** finding and resolving a mask, coordinate, horizon, or metric
  mismatch between EvalSim and Waymax.
- **Metric/generator failure:** discovering that a frozen-agent generator erased the
  transition the metric should observe, correcting it, and narrowing the resulting
  claim rather than presenting the original miss as a metric weakness.
- **Learned-evaluator failure:** detecting leakage, a simulator fingerprint, calibration
  failure, or OOD collapse.
- **Scale/reliability tradeoff:** changing a batching, caching, or execution design based
  on measured cost, throughput, or fault behavior.

The project supplies the technical artifacts. Mentoring, cross-team influence, and
shipped-product impact should be answered with genuine professional experience.
