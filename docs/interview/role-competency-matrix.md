# Role competency matrix

**Target role:** Senior Machine Learning Engineer, Simulation Evaluation
**Source:** [Official Waymo posting](https://careers.withwaymo.com/jobs/senior-machine-learning-engineer-simulation-evaluation-mountain-view-california-united-states)
**Updated:** 2026-07-31

This is a sanitized planning artifact. It records evidence categories, not personal
contact information, private correspondence, or detailed employment history.

## Overall assessment

The strongest positioning is:

> An experienced production-ML and evaluation leader who uses a bounded AV simulation
> case study to show evaluation judgment: simple comparison probes, explicit metric
> constructs, contradictory results, controlled blind spots, and honest claim limits.

The repository now includes the M0–M2 software foundation, the M3 one-record Waymo
vertical slice, the accepted M4 cohort/reference path, and the accepted M5 metric and
finite-cohort statistical evaluation. M4 accounts for every record in the first ten
validation shards and freezes a deterministic 128-scenario complete-case conditional
cohort. M5 evaluates that unchanged cohort with 13 metrics, 8 source-only slices,
complete paired scorecards, and bounded native Waymax cross-checks. The mixed result
supports the methodology but no overall policy winner. M7 now adds an outcome-aware,
data-free construct audit: three hand-built cases, four controlled probes at four doses,
and a corrected six-response/six-non-response matrix across three selected metrics. It
verifies intended field-of-view and wiring but does not prove general metric construct
validity, simulator superiority, dataset representativeness, or generalization. Learned
evaluation, video/VLM work, broader evaluator validation, and scalable execution are
optional extensions rather than prerequisites for this positioning.

## Evidence matrix

Legend: **Strong** = defensible evidence already exists; **Partial** = adjacent evidence
exists but the role-specific proof is incomplete; **Gap** = no credible current evidence;
**Optional gap** = absent but not required by the current project strategy.

| Role signal | Current evidence | Status | Evidence to build |
|---|---|:---:|---|
| Python and ML systems engineering | Typed Python architecture, deterministic execution, serialization, manifests, fail-closed official execution, an accepted local-WOMD metric path, and snapshot-bound test evidence (current: 1,444 passes with 1 local-data skip; M5: 876 Waymo-extra passes with 1 local-data skip; clean core-only: 790 passes with 28 optional/local skips); prior professional evidence is strong | Strong | Preserve quality; add counterfactual, learned, or scalable paths only when they answer a separately scoped question |
| Evaluation frameworks for complex ML systems | Contract-first evaluator seams, 13 separately reported metrics, 8 source-only slices, complete paired finite-cohort scorecards, immutable result verification, explicit limitations, and a data-free M7 construct audit that also caught a generator artifact; prior production-evaluation experience is relevant | Partial | Explain the current field-of-view evidence precisely; add calibrated/source-disjoint or real-scene validation only if a later claim requires it |
| Autonomous-driving simulation | Five synthetic scene families and three limited policies share the contract with a frozen 128-scenario WOMD cohort; M5 evaluated all three over 80 transitions with complete paired scorecards, the privileged logged-trajectory waypoint-following Waymax IDM reference remains a bounded M4 path, and M6 counterfactual controls exist pre-data | Partial | Keep the mixed M5 result and M6 status bounded; an accepted counterfactual WOMD result is needed before claiming demonstrated ego-conditioned reactivity |
| WOMD workflow | Exact ten-shard TFRecord resolution, immutable raw data, full raw/eligible/rejected accounting, deterministic complete-case selection, native-ID preservation, field mapping, local-only evidence controls, and accepted M5 slices/statistics over the unchanged cohort | Strong | Broader data only when a later question requires it; do not claim the ten-shard cohort is representative or use its slice counts as WOMD prevalence |
| Waymax proficiency | Pinned dataloader/state construction, supported-field parity, exact-log reference and `Rollout` conversion, bounded deterministic execution of the privileged logged-trajectory waypoint-following Waymax IDM reference, batch-two JIT/vmap gates, and native logged-position/overlap/fixed-step-kinematic metric cross-checks ran locally | Partial | An accepted live M6 outcome or another separately reviewed comparison plus broader independent validation; preserve the shared-decode, tolerance, fixed-step, and privileged-reference caveats |
| JAX/Flax depth | Pinned JAX runs meaningful batch-two `jit`/`vmap` exact-log computation plus a bounded JIT execution of the privileged logged-trajectory waypoint-following Waymax IDM reference on CPU; compile, warm latency/throughput, and process peak RSS are measured. Flax remains compatibility-pinned but unused | Partial | Optional: scaling curves/device evidence or Flax training only for a separately approved scale/learned-evaluator question |
| Learned discriminator pipelines | No learned evaluator is implemented | Optional gap | M8 only if a learned-evaluator question becomes important; require leakage-safe baselines, calibration, and generator-held-out tests |
| Evaluator validity and governance | M7 has four controlled defect families, analytic severity oracles, a detection matrix, bounded governance cards/invariance probes, and an implemented outcome-aware construct audit. Its corrected three-case × four-dose matrix has six responses and six non-responses; the v1 freeze miss was a generator artifact, while the remaining field-separated blind spots are construct/wiring evidence | Partial | Keep the original v1 gates explicitly unmet. A calibrated, source-disjoint, held-out, or WOMD-backed validation claim requires a separate pre-registration and may remain optional |
| Temporal stability evaluation | M5 continuity, lifecycle, acceleration, jerk, and yaw-rate diagnostics ran on synthetic oracles and the fixed WOMD cohort; M7 adds velocity-only and position-only controlled defects | Partial | Complete only the controlled-defect validity needed for the metric argument; video temporal evaluation is optional M9 work |
| Geometric discrepancy | M5 overlap, center-distance, lane-distance, and lane-heading diagnostics have analytic/invariance tests and fixed-cohort results | Partial | Typed offroad/route semantics in M6–M7; do not rename the neutral lane diagnostics |
| Condition following / controllability | Typed M6 identity/braking interventions, paired metrics, synthetic oracles, and an outcome-blind runner are implemented pre-data; no official WOMD result exists | Partial | Fresh exact-source approval and accepted paired WOMD analysis before a demonstrated controllability/reactivity claim; video conditions are optional M9 work |
| Multimodal consistency and sensor simulation | Adjacent prior computer-vision/sensor-evaluation experience exists, but EvalSim is trajectory-only | Partial | Optional, if separately pursued: a clearly separated real sensor/video pilot in M9B |
| VLM-based semantic evaluation | No current evidence | Optional gap | Optional, if separately pursued: versioned rubric, randomized paired judgments, corruption labels, and judge-bias analysis in M9B |
| Generative/world/video model evaluation | Project evaluates simple traffic policies, not multimodal world models | Optional gap | Optional, if separately pursued: controlled generated/corrupted outputs in M9; do not claim world-model training |
| Scalable training/evaluation pipelines | Strong prior production evidence; repository has useful provenance primitives but no measured scalable executor | Partial | Optional, if separately pursued: streaming, batching, resume/retry, fault injection, and measured local/cloud performance in M10 |
| GPU/TPU execution and serving | Strong adjacent professional systems evidence; no project-specific accelerator run | Partial | Optional, if separately pursued: one real, modest accelerator benchmark plus honest cost/performance analysis |
| Research and experimental rigor | Strong prior research evidence plus M4/M5 pre-registration, classified population accounting, preserved failure/correction records, complete finite-cohort effects, deterministic scene-reweighting sensitivity analysis, independent result reconstruction, mixed-result retention, explicit claim limitations, and an M7 generator error found and corrected after adversarial review | Partial | Do not promote the outcome-aware audit into validation. Add held-out, OOD, learned, multimodal, or real-scene evidence only for a separately scoped claim |
| Architecture and technical direction | Strong prior leadership evidence plus clear project seams and design documentation | Strong | Optional, if separately pursued: broader ADR/RFC, launch-criteria, ownership, and risk artifacts in M11 |
| Mentoring and cross-functional influence | Strong professional evidence may support this; a solo repository cannot create it | Outside repo | Prepare real work-history stories; never infer mentoring from this project |
| C++ | Not demonstrated in EvalSim and only preferred in the posting | Lower-priority gap | Do not distort the roadmap; add it only for a profiled bottleneck or explicit interview need |

## Highest-priority gaps

1. **Evaluator validity remains bounded.** The M5 mixed/no-winner result and M7
   construct/wiring evidence are defensible. The priority is to explain exactly why the
   corrected probes respond; held-out or real-scene validation is optional unless a
   stronger claim is needed.
2. **The conclusion must remain decision-specific.** Explain why fidelity, overlap, and
   feasibility disagree, what each misses, and why no replacement decision follows.
3. **Counterfactual evidence is distinct.** M6 is implemented pre-data but requires fresh
   exact-source approval and an accepted WOMD result before supporting reactivity claims.
4. **Learned, multimodal, and scale work are optional.** If pursued, each needs its own
   leakage, calibration, held-out, cost, or platform evidence; none is required to finish
   the metrics-first argument.

## What not to force into EvalSim

- Training a large diffusion, flow, or world model.
- Fake distributed infrastructure.
- C++ without a measured bottleneck.
- Mentoring or organizational-impact claims from solo work.
- Camera/video claims from BEV motion renders.

The repository should demonstrate evaluation judgment. Professional history should carry
the shipped-product, mentoring, and organizational-scale stories that a solo project
cannot recreate.
