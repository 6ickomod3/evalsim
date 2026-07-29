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
| Python | Verified | Python package, contracts, sources, policies, rollout engine, and tests; 493 Waymo-extra tests pass with 1 local-data skip, while 418 core-only tests pass with 22 optional-runtime skips | Keep the claim scoped to implemented work; test count is software evidence, not realism evidence |
| M3 local Waymo vertical slice | Verified | One ignored WOMD v1.3.1 scenario from exact shard `00000` passed native-identity-preserving Waymax decode, independent field parity, EvalSim conversion, Parquet, visualization, three policy smokes, and JAX CPU `jit` acceptance | Keep the wording to one local scenario and the supported M3 fields |
| M4 deterministic local cohort and exact-log cross-check | Verified | Exactly ten validation shards produced 2,916 raw, 1,527 eligible, and 1,389 supported-map rejections; 128 were selected without deficit or fallback. Full-cohort exact-log/rollout gates passed, all three EvalSim policies ran 80 transitions, the Waymax IDM reference passed its bounded 16-scene × 20-transition gate, and a batch-two JAX CPU exact-log microbenchmark recorded compile/warm/peak-process-RSS evidence | State that this is a complete-case conditional sample, that both paths share Waymax decode, that the Waymax IDM reference is privileged, and that no M5 metric/statistical comparison exists |
| WOMD | Partial | Actual local TFRecord decode plus deterministic accounting over the first ten validation shards and a 128-scenario accepted complete-case cohort; no data, native identity, locator, coordinate, or detailed run artifact is published | M5 metric/slice/statistical results before claiming an evaluated dataset population; broader shards or splits before claiming WOMD representativeness |
| Waymax | Partial | Pinned dataloader/state construction, supported-field parity, exact-log reference/conversion, deterministic bounded IDM-reference execution, and batch-two JIT/vmap checks run locally | M5 numerical custom/Waymax metric cross-checks and M6 reactive comparisons; never describe Waymax as independent ground truth |
| JAX | Partial | Pinned JAX/jaxlib run M4 batch-two `jit`/`vmap` exact-log and bounded IDM-reference gates on CPU; the narrow exact-log microbenchmark measured 217.983625 ms compile, 1.897854 ms median, 2.617709 ms nearest-rank empirical p95, 1,053.821843 scenarios/s, and 587,808,768-byte process peak RSS | Broader scaling, device-memory/utilization evidence, accelerator comparison, and M10 streaming/resume before claiming a scalable JAX pipeline |
| Reproducible evaluation platform | Partial | Deterministic seeds, typed manifests, lossless Parquet, and immutable scenario manifests | M5 results, then M10 config/cache/resume and end-to-end command |
| Compares log replay, constant velocity, and IDM | Partial | All three adapters run through one engine, separate on selected synthetic semantics, and completed 80 transitions on the frozen 128-scenario WOMD cohort | M5 common metrics and paired statistical comparison; M4 execution alone does not rank realism |
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
| Scalable JAX pipeline | Not yet | M4 adds a real batch-two CPU `jit`/`vmap` exact-log microbenchmark, but no streaming, scalable executor, device-memory evidence, or measured scaling curve | M10 streaming/batching/resume/fault recovery and measured cost/performance |
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
