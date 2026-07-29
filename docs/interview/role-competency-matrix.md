# Role competency matrix

**Target role:** Senior Machine Learning Engineer, Simulation Evaluation
**Source:** [Official Waymo posting](https://careers.withwaymo.com/jobs/senior-machine-learning-engineer-simulation-evaluation-mountain-view-california-united-states)
**Updated:** 2026-07-29

This is a sanitized planning artifact. It records evidence categories, not personal
contact information, private correspondence, or detailed employment history.

## Overall assessment

The strongest positioning is:

> An experienced production-ML and evaluation leader who is adding concrete
> Waymo-domain evidence: real WOMD/Waymax/JAX execution, evaluator validation,
> learned discriminators, and a bounded multimodal/VLM bridge.

The repository now includes the M0–M2 software foundation, the M3 one-record Waymo
vertical slice, the accepted M4 cohort/reference path, and the accepted M5 metric and
finite-cohort statistical evaluation. M4 accounts for every record in the first ten
validation shards and freezes a deterministic 128-scenario complete-case conditional
cohort. M5 evaluates that unchanged cohort with 13 metrics, 8 source-only slices,
complete paired scorecards, and bounded native Waymax cross-checks. The mixed result
supports the methodology but no overall policy winner. It does not yet prove metric
construct validity, simulator superiority, dataset representativeness, generalization,
learned evaluation, or scalable production execution.

## Evidence matrix

Legend: **Strong** = defensible evidence already exists; **Partial** = adjacent evidence
exists but the role-specific proof is incomplete; **Gap** = no credible current evidence.

| Role signal | Current evidence | Status | Evidence to build |
|---|---|:---:|---|
| Python and ML systems engineering | Typed Python architecture, deterministic execution, serialization, manifests, fail-closed official execution, 876 Waymo-extra tests passing with 1 local-data skip, 790 core-only tests passing with 28 optional/local skips, and an accepted local-WOMD metric path; prior professional evidence is strong | Strong | Preserve quality while adding counterfactual, learned, and scalable execution paths |
| Evaluation frameworks for complex ML systems | Contract-first evaluator seams, 13 separately reported metrics, 8 source-only slices, complete paired finite-cohort scorecards, immutable result verification, and explicit limitations; prior production-evaluation experience is relevant | Partial | M7 evaluator validation and governance, then M11 decision memo; M5 application alone does not prove construct validity |
| Autonomous-driving simulation | Five synthetic scene families and three limited policies share the contract with a frozen 128-scenario WOMD cohort; M5 evaluated all three over 80 transitions with complete paired scorecards, while the privileged Waymax IDM reference remains a bounded M4 execution path | Partial | M6 counterfactual reactivity and M7 evaluator validation; the mixed M5 result does not select a simulator |
| WOMD workflow | Exact ten-shard TFRecord resolution, immutable raw data, full raw/eligible/rejected accounting, deterministic complete-case selection, native-ID preservation, field mapping, local-only evidence controls, and accepted M5 slices/statistics over the unchanged cohort | Strong | Broader data only when a later question requires it; do not claim the ten-shard cohort is representative or use its slice counts as WOMD prevalence |
| Waymax proficiency | Pinned dataloader/state construction, supported-field parity, exact-log reference and `Rollout` conversion, bounded deterministic IDM-reference execution, batch-two JIT/vmap gates, and native logged-position/overlap/fixed-step-kinematic metric cross-checks ran locally | Partial | M6 reactive/backend comparisons and broader independent validation; preserve the shared-decode, tolerance, fixed-step, and privileged-reference caveats |
| JAX/Flax depth | Pinned JAX runs meaningful batch-two `jit`/`vmap` exact-log computation plus a bounded IDM-reference JIT on CPU; compile, warm latency/throughput, and process peak RSS are measured. Flax remains compatibility-pinned but unused | Partial | Scaling curves/device evidence in M10 and Flax training/evaluation in M8 |
| Learned discriminator pipelines | No learned evaluator is implemented | Gap | M8 leakage-safe baselines, temporal model, calibration, OOD tests, and batch inference |
| Evaluator validity and governance | Good adversarial software tests, but no proof that metrics detect meaningful realism defects | Gap | M7 severity curves, invariances, false positives/misses, metric cards, and held-out defects |
| Temporal stability evaluation | M5 continuity, lifecycle, acceleration, jerk, and yaw-rate diagnostics ran on synthetic oracles and the fixed WOMD cohort | Partial | M7 controlled-defect validity; true video temporal evaluation remains M9 |
| Geometric discrepancy | M5 overlap, center-distance, lane-distance, and lane-heading diagnostics have analytic/invariance tests and fixed-cohort results | Partial | Typed offroad/route semantics in M6–M7; do not rename the neutral lane diagnostics |
| Condition following / controllability | Counterfactual design exists only as a plan | Gap | Typed interventions and paired causal analysis in M6; video conditions in M9 |
| Multimodal consistency and sensor simulation | Adjacent prior computer-vision/sensor-evaluation experience exists, but EvalSim is trajectory-only | Partial | A clearly separated real sensor/video pilot in M9B |
| VLM-based semantic evaluation | No current evidence | Gap | Versioned rubric, randomized paired judgments, corruption labels, and judge-bias analysis in M9B |
| Generative/world/video model evaluation | Project evaluates simple traffic policies, not multimodal world models | Gap | Evaluate controlled generated/corrupted outputs in M9; do not claim world-model training |
| Scalable training/evaluation pipelines | Strong prior production evidence; repository has useful provenance primitives but no measured scalable executor | Partial | Streaming, batching, resume/retry, fault injection, and measured local/cloud performance in M10 |
| GPU/TPU execution and serving | Strong adjacent professional systems evidence; no project-specific accelerator run | Partial | One real, modest accelerator benchmark plus honest cost/performance analysis |
| Research and experimental rigor | Strong prior research evidence plus M4/M5 pre-registration, classified population accounting, preserved failure/correction records, complete finite-cohort effects, deterministic scene-reweighting sensitivity analysis, independent result reconstruction, mixed-result retention, and explicit claim limitations | Partial | Controlled ablations, OOD analysis, and evaluator falsification in M7–M9 before broader validity/generalization claims |
| Architecture and technical direction | Strong prior leadership evidence plus clear project seams and design documentation | Strong | ADR/RFC trail, launch criteria, ownership model, and risk register in M11 |
| Mentoring and cross-functional influence | Strong professional evidence may support this; a solo repository cannot create it | Outside repo | Prepare real work-history stories; never infer mentoring from this project |
| C++ | Not demonstrated in EvalSim and only preferred in the posting | Lower-priority gap | Do not distort the roadmap; add it only for a profiled bottleneck or explicit interview need |

## Highest-priority gaps

1. **The named stack now has bounded M5 evidence, not broad validity evidence.** The
   ten-shard conditional cohort, complete paired scorecards, bounded native Waymax
   metric cross-checks, and mixed/no-winner result are defensible. M6 counterfactuals
   and M7 controlled-defect validation are still required before a simulator-quality or
   replacement conclusion.
2. **The role emphasizes learned evaluators.** A one-day classifier would look toy;
   M8 must test calibration, leakage, shortcuts, and generator-held-out behavior.
3. **The role emphasizes evaluator validity.** M7 should be central: known defects,
   dose-response, false positives, invariance, blind spots, and governance.
4. **The posting is multimodal/video/VLM-heavy.** Motion-only WOMD work is relevant but
   insufficient. M9 must remain a separate, honestly labeled evidence track.
5. **Scale needs measurements.** Prior professional scale is valuable, but this project
   should report only throughput, cost, reliability, or accelerator use it actually ran.

## What not to force into EvalSim

- Training a large diffusion, flow, or world model.
- Fake distributed infrastructure.
- C++ without a measured bottleneck.
- Mentoring or organizational-impact claims from solo work.
- Camera/video claims from BEV motion renders.

The repository should demonstrate evaluation judgment. Professional history should carry
the shipped-product, mentoring, and organizational-scale stories that a solo project
cannot recreate.
