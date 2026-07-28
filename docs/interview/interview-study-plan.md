# Interview study plan

This plan turns each build milestone into an interview-ready technical argument. It is
organized by evidence loops rather than calendar weeks so it remains useful if the
interview timeline changes.

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

Build with M3–M4. Be able to explain:

- WOMD history/current/future boundaries, validity masks, object roles, roadgraph, and
  traffic-light semantics;
- why EvalSim keeps a substrate-neutral contract;
- what Waymax provides that the NumPy engine does not;
- coordinate, padding, mask, and eligibility mismatches;
- `jit`, `vmap`, static shapes, compilation cost, host/device transfer, PRNG discipline,
  and why local CPU is acceptable for development.

### 2. Metric and statistical design

Build with M5. Be able to explain:

- realism versus downstream usefulness;
- construct validity, units, eligibility, missingness, and aggregation;
- why incompatible axes should not become one score;
- paired scenario comparisons and scenario-cluster bootstrap;
- rare events, small slices, pre-registered versus exploratory analysis, and multiple
  testing;
- how a release decision changes when metrics disagree.

### 3. Closed-loop causal evaluation

Build with M6. Be able to explain:

- why log replay is invalid as a reactive reference after ego perturbation;
- open-loop fidelity versus closed-loop usefulness;
- treatment, control, eligibility, paired seeds, and leakage;
- nonreactivity versus overreaction;
- what causal language the experiment supports and what it does not.

### 4. Evaluating the evaluator

Build with M7. Be able to explain:

- Goodharting and metric gaming;
- invariance, sensitivity, specificity, monotonicity, and calibration;
- held-out defect families and false-positive analysis;
- a plausible metric that failed and how the design changed;
- ownership, versioning, launch thresholds, and evaluator regressions.

### 5. Learned discriminators

Build with M8. Be able to explain:

- scenario-grouped splits and simulator-fingerprint leakage;
- why logistic/feature baselines precede a Flax temporal model;
- calibration versus ranking;
- AUROC versus AUPRC, Brier/ECE, uncertainty, and slice/OOD behavior;
- generator-held-out evaluation and shortcut audits;
- what a discriminator learns when “real” and “simulated” preprocessing differ.

### 6. Multimodal/video/VLM evaluation

Build with M9. Be able to explain:

- temporal stability, geometric consistency, cross-modal consistency, and condition
  following when ground truth is incomplete;
- why BEV semantic video is not camera realism;
- how to validate a VLM judge without treating it as truth;
- randomized paired prompts, prompt sensitivity, positional bias, abstention, and human
  audit;
- how motion evaluation and real sensor evaluation share governance but require
  different data and contracts.

### 7. Scale, production, and technical leadership

Build with M10–M11 and use real professional examples for organizational questions. Be
able to explain:

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
- **Metric failure:** discovering that a reasonable metric missed or rewarded a known
  defect.
- **Learned-evaluator failure:** detecting leakage, a simulator fingerprint, calibration
  failure, or OOD collapse.
- **Scale/reliability tradeoff:** changing a batching, caching, or execution design based
  on measured cost, throughput, or fault behavior.

The project supplies the technical artifacts. Mentoring, cross-team influence, and
shipped-product impact should be answered with genuine professional experience.
