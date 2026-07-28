# Claim-to-evidence ledger

**Updated:** 2026-07-28
**Rule:** A plan, README sentence, downloaded file, or dependency import is not
implementation evidence.

Statuses:

- **Verified** — supported by committed code, tests, and an honest scope statement.
- **Partial** — a foundation exists, but the natural-language claim would imply more.
- **Not yet** — required implementation or empirical evidence does not exist.

## Current ledger

| Candidate claim | Status | Evidence that exists now | Required before using the full claim |
|---|:---:|---|---|
| Python | Verified | Python package, contracts, sources, policies, rollout engine, and tests | Keep the claim scoped to implemented work |
| M3 local Waymo vertical slice | Verified | One ignored WOMD v1.3.1 scenario from exact shard `00000` passed native-identity-preserving Waymax decode, independent field parity, EvalSim conversion, Parquet, visualization, three policy smokes, and JAX CPU `jit` acceptance | Keep the wording to one local scenario and the supported M3 fields |
| WOMD | Partial | Actual local TFRecord decode and validated one-scenario adapter acceptance; no data or native identity is published | M4 deterministic cohort and accounting before a substantive dataset-scale claim |
| Waymax | Partial | Pinned Waymax dataloader/state construction and independent raw-state parity execute locally | M4 environment/reference policies, rollout conversion, batching, and semantic discrepancy report |
| JAX | Partial | Pinned JAX/jaxlib run a real compiled CPU operation during the gated M3 acceptance | M4 `vmap`, meaningful batched computation, numerical parity, compile/warm benchmark, and memory evidence |
| Reproducible evaluation platform | Partial | Deterministic seeds, typed manifests, lossless Parquet, and immutable scenario manifests | M5 results, then M10 config/cache/resume and end-to-end command |
| Compares log replay, constant velocity, and IDM | Partial | All three adapters run through one engine and separate on selected synthetic semantics | M5 common metrics and statistical comparison on the frozen WOMD cohort |
| Closed-loop simulation | Partial | World agents advance from current simulated state; ego remains logged/exogenous and policies are limited | Qualify the current claim; M6 adds paired counterfactual ego control and genuinely reactive comparisons |
| Kinematic realism metrics | Not yet | Metric contract only | M5 implementation, analytic oracles, WOMD results, units/eligibility, and Waymax cross-check |
| Interaction metrics | Not yet | No implementation | M5 overlap/distance/TTC/headway implementation and validation |
| Map-adherence metrics | Not yet | Map geometry exists; no evaluator | M5 offroad/wrong-way/lane/route metrics and coordinate invariance |
| Behavioral slices | Not yet | Synthetic tags exist | Pre-registered WOMD slices, sample counts, missingness, and robustness in M5 |
| Scenario-level confidence intervals | Not yet | No statistical implementation | Paired scenario-cluster bootstrap and verified coverage/edge cases in M5 |
| Counterfactual ego perturbation | Not yet | Ego-control seam is planned but absent | Typed interventions, causal controls, eligibility, and paired WOMD results in M6 |
| Detects nonreactivity and overreaction | Not yet | Log replay and IDM have the intended conceptual contrast only | Validated independent measures and paired counterfactual evidence in M6 |
| Metric stress testing | Not yet | Adversarial software tests are not metric-validation results | M7 severity curves, held-out defects, false positives/misses, and detection matrix |
| Hand-designed realism evaluators | Not yet | Metric interface only | Validated M5 metrics plus M7 governance |
| Learned realism evaluator | Not yet | No learned model | M8 baseline + Flax model, leakage audit, calibration, OOD/generator holdout, and disagreement analysis |
| Scalable JAX pipeline | Not yet | Only the M3 single-operation CPU `jit`; no batching, scalable executor, or measured scaling path | M10 streaming/batching/resume/fault recovery and measured cost/performance |
| Video/VLM evaluation | Not yet | WOMD motion data has no camera pixels | M9B real camera sequences, controlled defects, validated VLM rubric, and blinded audit |

## Evidence package required to change a status

Every full claim needs:

1. a committed implementation path;
2. unit and end-to-end tests, including an independent oracle where possible;
3. a reproducible command and immutable manifest;
4. local measured results with population size, exclusions, uncertainty, and failure cases;
5. explicit limitations and a claim wording that matches actual scale;
6. an adversarial review that checks leakage, shortcuts, semantic errors, and
   overclaiming.

Under current repository rules, raw/processed data and generated experiment artifacts
stay local. The ledger may reference permitted small manifests, checksums, schemas, code,
tests, and reproduction commands. It must not contain a native scenario identity,
source-derived coordinates, private data, or dataset payloads.

## Claim-language guardrails

- Do not call log replay a realism upper bound after an ego perturbation.
- Do not equate test count with realism, generalization, or scale.
- Do not say production-ready, deployed, large-scale, distributed, GPU/TPU, or
  multimodal unless the corresponding path actually ran.
- Do not call a BEV raster a camera-video simulator.
- Do not treat Waymax, a learned discriminator, or a VLM as ground truth.
- A negative result may still unlock an honest methodology claim if the experiment is
  rigorous and the failure is explained.
