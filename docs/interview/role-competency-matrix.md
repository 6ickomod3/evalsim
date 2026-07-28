# Role competency matrix

**Target role:** Senior Machine Learning Engineer, Simulation Evaluation
**Source:** [Official Waymo posting](https://careers.withwaymo.com/jobs/senior-machine-learning-engineer-simulation-evaluation-mountain-view-california-united-states)
**Updated:** 2026-07-28

This is a sanitized planning artifact. It records evidence categories, not personal
contact information, private correspondence, or detailed employment history.

## Overall assessment

The strongest positioning is:

> An experienced production-ML and evaluation leader who is adding concrete
> Waymo-domain evidence: real WOMD/Waymax/JAX execution, evaluator validation,
> learned discriminators, and a bounded multimodal/VLM bridge.

The current repository is a strong software foundation, but it is not yet an empirical
simulation-evaluation system. M0–M2 prove contracts, deterministic scenarios, baseline
policies, rollout mechanics, provenance, and testing. They do not yet prove metric
validity, realism, generalization, learned evaluation, Waymo-tool proficiency, or scale.

## Evidence matrix

Legend: **Strong** = defensible evidence already exists; **Partial** = adjacent evidence
exists but the role-specific proof is incomplete; **Gap** = no credible current evidence.

| Role signal | Current evidence | Status | Evidence to build |
|---|---|:---:|---|
| Python and ML systems engineering | Typed Python architecture, deterministic execution, serialization, manifests, and 134 tests; prior professional evidence is strong | Strong | Preserve quality while adding real-data and ML paths |
| Evaluation frameworks for complex ML systems | Contract-first evaluator seams and explicit limitations; prior production-evaluation experience is relevant | Partial | M5 metrics/statistics, M7 evaluator validation, M11 decision memo |
| Autonomous-driving simulation | Five synthetic scene families and three limited policies in a world-agent rollout engine | Partial | M3–M6 on real WOMD scenes and Waymax environments |
| WOMD workflow | Reference validation shards are local and documented but unused by code | Gap | M3 adapter and M4 deterministic cohort |
| Waymax proficiency | Planned only | Gap | Use loading, immutable state, environment/dynamics, policies, batching, and metric cross-checks in M3–M7 |
| JAX/Flax depth | Current implementation is NumPy; no JAX or Flax code | Gap | Real `jit`/`vmap` path in M3–M4 and Flax training/evaluation in M8 |
| Learned discriminator pipelines | No learned evaluator is implemented | Gap | M8 leakage-safe baselines, temporal model, calibration, OOD tests, and batch inference |
| Evaluator validity and governance | Good adversarial software tests, but no proof that metrics detect meaningful realism defects | Gap | M7 severity curves, invariances, false positives/misses, metric cards, and held-out defects |
| Temporal stability evaluation | Trajectory representation can support it, but no evaluator exists | Gap | Motion temporal checks in M5/M7; true video temporal evaluation in M9 |
| Geometric discrepancy | Map and agent geometry are represented; no geometry metrics are evaluated | Partial | Map/route/overlap metrics, coordinate invariance, and Waymax parity in M4–M7 |
| Condition following / controllability | Counterfactual design exists only as a plan | Gap | Typed interventions and paired causal analysis in M6; video conditions in M9 |
| Multimodal consistency and sensor simulation | Adjacent prior computer-vision/sensor-evaluation experience exists, but EvalSim is trajectory-only | Partial | A clearly separated real sensor/video pilot in M9B |
| VLM-based semantic evaluation | No current evidence | Gap | Versioned rubric, randomized paired judgments, corruption labels, and judge-bias analysis in M9B |
| Generative/world/video model evaluation | Project evaluates simple traffic policies, not multimodal world models | Gap | Evaluate controlled generated/corrupted outputs in M9; do not claim world-model training |
| Scalable training/evaluation pipelines | Strong prior production evidence; repository has useful provenance primitives but no measured scalable executor | Partial | Streaming, batching, resume/retry, fault injection, and measured local/cloud performance in M10 |
| GPU/TPU execution and serving | Strong adjacent professional systems evidence; no project-specific accelerator run | Partial | One real, modest accelerator benchmark plus honest cost/performance analysis |
| Research and experimental rigor | Strong prior research evidence; project has analytic tests but no empirical evaluation study | Partial | Pre-registration, uncertainty, negative results, ablations, OOD analysis, and falsification in M5–M9 |
| Architecture and technical direction | Strong prior leadership evidence plus clear project seams and design documentation | Strong | ADR/RFC trail, launch criteria, ownership model, and risk register in M11 |
| Mentoring and cross-functional influence | Strong professional evidence may support this; a solo repository cannot create it | Outside repo | Prepare real work-history stories; never infer mentoring from this project |
| C++ | Not demonstrated in EvalSim and only preferred in the posting | Lower-priority gap | Do not distort the roadmap; add it only for a profiled bottleneck or explicit interview need |

## Highest-priority gaps

1. **The named stack is not implemented.** Downloaded WOMD data does not make
   WOMD/Waymax/JAX a current-tense project claim.
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
