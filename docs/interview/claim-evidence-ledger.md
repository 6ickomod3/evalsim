# Claim-to-evidence ledger

**Updated:** 2026-07-29
**Rule:** A plan, README sentence, downloaded file, or dependency import is not
implementation evidence.

Statuses:

- **Verified** — supported by committed code, tests, and an honest scope statement.
- **Partial** — a foundation exists, but the natural-language claim would imply more.
- **Not yet** — required implementation or empirical evidence does not exist.

## Current ledger

| Candidate claim | Status | Evidence that exists now | Required before using the full claim |
|---|:---:|---|---|
| Python | Verified | Python package, contracts, sources, policies, rollout engine, and tests; the reviewed official-runner and outcome-blind cadence amendments bring the full repository suite to 876 passing tests with 1 expected local-data skip. Its focused evidence includes 14 runner tests, a public mocked 128-case lifecycle, 18 injected failure boundaries, 34 adversarial lifecycle cases, 7 private-authentication regression cases, and fixed-step/nonuniform-time version-isolation gates. A fresh environment without JAX, jaxlib, TensorFlow, Flax, or Waymax reports 790 passed with 28 expected optional/local skips | Keep the claim scoped to implemented work; test count is software evidence, not realism evidence |
| M3 local Waymo vertical slice | Verified | One ignored WOMD v1.3.1 scenario from exact shard `00000` passed native-identity-preserving Waymax decode, independent field parity, EvalSim conversion, Parquet, visualization, three policy smokes, and JAX CPU `jit` acceptance | Keep the wording to one local scenario and the supported M3 fields |
| M4 deterministic local cohort and exact-log cross-check | Verified | Exactly ten validation shards produced 2,916 raw, 1,527 eligible, and 1,389 supported-map rejections; 128 were selected without deficit or fallback. Full-cohort exact-log/rollout gates passed, all three EvalSim policies ran 80 transitions, the Waymax IDM reference passed its bounded 16-scene × 20-transition gate, and a batch-two JAX CPU exact-log microbenchmark recorded compile/warm/peak-process-RSS evidence | State that this is a complete-case conditional sample, that both paths share Waymax decode, that the Waymax IDM reference is privileged, and that no real-WOMD M5 metric/statistical comparison exists |
| M5 data-free evaluation path | Verified | Commit `9b2676ac4b1c7bfb9f35a1c92f0159158756544a` runs five fixed synthetic scenarios through 3 policies × 13 metrics (195 rows), 8 source-only slices (40 rows), and the complete 13 × 8 × 3 scorecard domain (312 rows), then writes and re-verifies an immutable `data_free_test` result. Twenty-five exact log-replay zero oracles cover all five cases and five registered error metrics. Every synthetic scorecard has paired N from 0 to 5, is labeled `insufficient_n`, suppresses effects and bands, and forbids directional language. Independent architecture, methods/statistics, and privacy/claim reviews found no remaining data-free blocker, and the reviewed implementation is pushed to `main` | Keep the claim exactly scoped to the data-free path. The separately gated real-WOMD run remains required for any WOMD effect, slice-prevalence, native metric-parity, policy-ranking, or milestone-completion result |
| M5 official-runner implementation | Verified | Commit `c155d0d199b827a56417f288a41ea2437ea65127` contains the accepted streaming runner, which consumes the explicit accepted 128-case M4 cohort once, separates three policy roles from the Waymax exact-log reference, writes pre-metric parity-order and post-evaluation determinism receipts, and fail-closes source/Git/shard/M4/output/terminal/finalization drift. Private-repository live-ref support was added and adversarially accepted at `828f01524e5bdcd6bbf1932fdea31c2d242e9bae` without changing the scientific contract. The outcome-blind cadence amendment versions the fixed-`0.1 s` Waymax kinematic diagnostic as `1.0.1` without changing physical-time rollout or derivative metrics. A public mocked 128-case lifecycle enforces exact domains of 6,656 metric, 1,024 slice, 312 scorecard, and 144 parity-summary rows. Fourteen runner tests, 18 injected failure boundaries, 34 adversarial lifecycle cases, seven private-authentication regression cases, the full suite with 876 passing tests and 1 expected local-data skip, and final adversarial **ACCEPT** support the implementation | Keep this claim scoped to runner software and mocked lifecycle evidence. No accepted M5 WOMD outcome or native WOMD parity result exists; the bound official execution and result reviews remain required for any empirical M5 claim |
| WOMD | Partial | Actual local TFRecord decode plus deterministic accounting over the first ten validation shards and a 128-scenario accepted complete-case cohort; no data, native identity, locator, coordinate, or detailed run artifact is published | M5 metric/slice results support only the fixed conditional cohort; broader probability sampling or justified target-population design is required before representativeness or WOMD-population inference |
| Waymax | Partial | Pinned dataloader/state construction, supported-field parity, exact-log reference/conversion, deterministic bounded IDM-reference execution, and batch-two JIT/vmap checks run locally | M5 numerical custom/Waymax metric cross-checks and M6 reactive comparisons; never describe Waymax as independent ground truth |
| JAX | Partial | Pinned JAX/jaxlib run M4 batch-two `jit`/`vmap` exact-log and bounded IDM-reference gates on CPU; the narrow exact-log microbenchmark measured 217.983625 ms compile, 1.897854 ms median, 2.617709 ms nearest-rank empirical p95, 1,053.821843 scenarios/s, and 587,808,768-byte process peak RSS | Broader scaling, device-memory/utilization evidence, accelerator comparison, and M10 streaming/resume before claiming a scalable JAX pipeline |
| Reproducible evaluation platform | Partial | Deterministic seeds, typed manifests, lossless Parquet, immutable scenario manifests, a committed data-free end-to-end command, and an accepted official runner with exhaustive source binding, typed receipts, exact row domains, failure preservation, terminal sealing, and pre-success re-verification | Complete the bound real-WOMD M5 acceptance, then add M10 config/cache/resume evidence; mocked lifecycle coverage is not a measured data run or resumable-scale result |
| Compares log replay, constant velocity, and IDM | Partial | All three adapters run through one engine, completed 80 transitions on the frozen 128-scenario WOMD cohort in M4, and now share 13 implemented metrics in the five-scene data-free M5 acceptance. The synthetic scorecards intentionally suppress every effect because paired N is below 10 | Run and review the fixed real-WOMD M5 paired comparison; neither M4 execution nor the data-free acceptance ranks realism |
| Closed-loop simulation | Partial | World agents advance from current simulated state; ego remains logged/exogenous and policies are limited | Qualify the current claim; M6 adds paired counterfactual ego control and genuinely reactive comparisons |
| Kinematic motion metrics | Partial | Registered source-neutral position, speed, acceleration, jerk, yaw-rate, continuity, overlap, and pinned-threshold kinematic metrics have accepted analytic, boundary, invariance, and data-free end-to-end tests in the pushed implementation | Obtain fixed-cohort WOMD results and the pre-registered native Waymax cross-check before making empirical quality claims |
| Interaction metrics | Partial | Oriented-overlap target-frame rate, minimum center distance, and capped constant-velocity disc-TTC proxy are implemented with analytic and edge-case tests | Fixed-cohort results and later M6 reactivity/headway evidence; overlap is not a unique collision count and TTC is not a collision forecast |
| Map-context diagnostics | Partial | Lane-center distance and lane-heading disagreement are implemented as neutral 2-D diagnostics with invariance and no-lane tests | Fixed-cohort results; true offroad, route, signal, and wrong-way semantics still require typed context and must not be inferred from these proxies |
| Behavioral slices | Partial | Eight pre-registered source-only slices and exact membership accounting are implemented and exercised by the data-free runner | Fixed real-WOMD membership counts, missingness, and robustness review; synthetic membership is not WOMD prevalence |
| Paired finite-cohort stability analysis | Partial | Deterministic scenario-level pairing, missingness accounting, resampling substreams, suppression thresholds, and stability-band edge cases are implemented and tested. The five-scene end-to-end path correctly suppresses all effects and bands | Apply the frozen analysis to the unchanged 128-scenario conditional cohort; bands remain empirical scenario-reweighting summaries, not WOMD-population confidence intervals or hypothesis tests |
| Counterfactual ego perturbation | Not yet | Ego-control seam is planned but absent | Typed interventions, causal controls, eligibility, and paired WOMD results in M6 |
| Detects nonreactivity and overreaction | Not yet | Log replay and IDM have the intended conceptual contrast only | Validated independent measures and paired counterfactual evidence in M6 |
| Metric stress testing | Not yet | Adversarial software tests are not metric-validation results | M7 severity curves, held-out defects, false positives/misses, and detection matrix |
| Hand-designed realism evaluators | Not yet | Thirteen hand-designed metric implementations and their data-free software oracles now exist, but no real-WOMD M5 outcome or M7 validation result exists | Fixed-cohort M5 evidence plus M7 governance; do not promote implementation tests into a realism-validity claim |
| Learned realism evaluator | Not yet | No learned model | M8 baseline + Flax model, leakage audit, calibration, OOD/generator holdout, and disagreement analysis |
| Scalable JAX pipeline | Not yet | M4 adds a real batch-two CPU `jit`/`vmap` exact-log microbenchmark, but no streaming, scalable executor, device-memory evidence, or measured scaling curve | M10 streaming/batching/resume/fault recovery and measured cost/performance |
| Video/VLM evaluation | Not yet | WOMD motion data has no camera pixels | M9B real camera sequences, controlled defects, validated VLM rubric, and blinded audit |

## Evidence package required to change a status

Every full claim needs:

1. a committed implementation path;
2. unit and end-to-end tests, including an independent oracle where possible;
3. a reproducible command and immutable manifest;
4. local measured results with cohort size, exclusions, estimand-appropriate uncertainty
   or stability analysis, and failure cases;
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
