# M5 official WOMD acceptance

- **Accepted:** 2026-07-29
- **Scope:** fixed 128-scenario complete-case conditional cohort from WOMD v1.3.1
  validation TFRecord shards `00000`–`00009`
- **Policies:** `log_replay`, `constant_velocity`, and `idm`, all deterministic with
  seed `0`
- **Result:** accepted with mixed evidence; there is no overall policy winner

This report is the deliberately promoted, aggregate-only record of the official M5
execution. The run passed its immutable-store verifier and independent semantic,
statistical, and privacy/claim reviews. It contains the complete pre-registered
12-cell primary `all`-slice matrix, including the two cells that did not pass the
bounded directional gate. It does not contain native scenario IDs, record locators,
coordinates, per-scene values, component distributions, shard digests, private
paths, or generated result files.

The definitions and claim gates were frozen before outcome access in the
[M5 pre-registration](../plans/2026-07-28-m5-real-womd-metrics-scorecards.md).
The native-semantic boundary is documented in the
[M5 Waymax crosswalk](../data/womd-waymax-m5-metric-crosswalk.md), and the applicable
attribution and use restrictions are in [NOTICE.md](../../NOTICE.md).

## Acceptance summary

- The official store is complete under the `official_m5` profile: **6,656 metric
  results**, **1,024 slice memberships**, **312 scorecards**, and **144 native-parity
  summaries**. Its manifest accounts for 131 immutable artifact records.
- All **2,048 primary metric rows** are valid. Every primary comparison has paired
  `n = 128`, zero exclusions, and zero policy-dependent or asymmetric missingness.
- The metric and statistical second passes reproduced the first-pass digests.
  Independent reconstruction reproduced all 312 scorecards exactly.
- Of the 12 primary cells, 11 pointwise bands and 10 multiplicity-adjusted bands
  exclude zero. Only those 10 cells pass the pre-registered bounded directional gate.
- The results are intentionally mixed. Position and speed logged-error metrics order
  the means as log replay, constant velocity, then IDM; overlap orders them as log
  replay, IDM, then constant velocity; and the fixed-step Waymax kinematic diagnostic
  orders them as IDM, constant velocity, then log replay. All four metrics register
  lower values as better.
- The IDM-versus-constant-velocity overlap band crosses zero. Their kinematic
  difference has only seven nonzero scene effects and its adjusted band crosses zero,
  so it is `event_sparse`. Neither contrast supports directional language.
- There is no composite score, policy superiority claim, total ordering, causal
  conclusion, or WOMD-population conclusion.

## Complete primary `all`-slice matrix

Raw effects are `A - B`. All four metrics are lower-is-better, so the separately
reported oriented advantage is `-(raw mean)`: positive means A is better and negative
means A is worse. Bands are deterministic empirical scene-reweighting stability
bands, not confidence intervals or hypothesis tests.

In each compact validity field,
`paired/valid A/valid B/excluded/asymmetric` is followed by both-missing and
A/B reason counts. Empty maps mean no missing reason was observed. `Directional`
means only that the adjusted band excludes zero and every other pre-registered gate
passed; it is not a population-level significance statement.

### Position error

`position_error_m` version `1.0.0`; unit `m`; direction `lower`.

| A - B | Validity and missingness | Raw mean | Median | Raw sign | Pointwise 0.95 band | Adjusted 0.9958333333333333 band | Oriented advantage | Gate |
|---|---|---:|---:|---|---|---|---:|---|
| `constant_velocity - log_replay` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `2.997305058703697` | `2.477351421225486` | positive | `[2.6279802249751203, 3.3901418187934973]` | `[2.4703977791024374, 3.5883347567281167]` | `-2.997305058703697` (A worse) | nonzero `128`; `descriptive`; suppressed no; directional yes |
| `idm - log_replay` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `7.7500647664453` | `7.45671017576238` | positive | `[7.254792072654283, 8.238701486809102]` | `[7.039911175083419, 8.46896905650854]` | `-7.7500647664453` (A worse) | nonzero `128`; `descriptive`; suppressed no; directional yes |
| `idm - constant_velocity` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `4.752759707741603` | `4.4283333456607075` | positive | `[4.202246972287565, 5.3075581383441]` | `[3.95256553101306, 5.56545970470591]` | `-4.752759707741603` (A worse) | nonzero `128`; `descriptive`; suppressed no; directional yes |

### Speed error

`speed_error_mps` version `1.0.0`; unit `m/s`; direction `lower`.

| A - B | Validity and missingness | Raw mean | Median | Raw sign | Pointwise 0.95 band | Adjusted 0.9958333333333333 band | Oriented advantage | Gate |
|---|---|---:|---:|---|---|---|---:|---|
| `constant_velocity - log_replay` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `1.0965694309637917` | `0.9199204495335616` | positive | `[0.9619761038524979, 1.2391368628083081]` | `[0.9032779039639633, 1.3053634771162963]` | `-1.0965694309637917` (A worse) | nonzero `128`; `descriptive`; suppressed no; directional yes |
| `idm - log_replay` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `2.9496225910279303` | `2.873040872369729` | positive | `[2.7967982543093335, 3.104855524781929]` | `[2.7243675139407824, 3.1765169409116614]` | `-2.9496225910279303` (A worse) | nonzero `128`; `descriptive`; suppressed no; directional yes |
| `idm - constant_velocity` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `1.8530531600641384` | `1.760009216719928` | positive | `[1.6431704146322648, 2.063347933439761]` | `[1.5420387726424956, 2.1559393836500935]` | `-1.8530531600641384` (A worse) | nonzero `128`; `descriptive`; suppressed no; directional yes |

### Oriented-box overlap rate

`oriented_box_overlap_rate` version `1.0.0`; unit `fraction`; direction `lower`.

| A - B | Validity and missingness | Raw mean | Median | Raw sign | Pointwise 0.95 band | Adjusted 0.9958333333333333 band | Oriented advantage | Gate |
|---|---|---:|---:|---|---|---|---:|---|
| `constant_velocity - log_replay` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `0.04966195639263592` | `0.031511287625418063` | positive | `[0.03709369889680477, 0.06256488985435854]` | `[0.031503882406754925, 0.06888376611885266]` | `-0.04966195639263592` (A worse) | nonzero `112`; `descriptive`; suppressed no; directional yes |
| `idm - log_replay` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `0.04324182820631069` | `0.03648263187219852` | positive | `[0.034523267487685035, 0.05171025074417448]` | `[0.03023094433374868, 0.05548078521175095]` | `-0.04324182820631069` (A worse) | nonzero `117`; `descriptive`; suppressed no; directional yes |
| `idm - constant_velocity` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `-0.006420128186325231` | `0.0` | negative | `[-0.015279928885177754, 0.0020568185134115127]` | `[-0.01956296032512017, 0.005847794738256519]` | `0.006420128186325231` (A better) | nonzero `117`; `descriptive`; suppressed no; directional **no** |

### Waymax kinematic infeasibility rate

`waymax_kinematic_infeasibility_rate` version `1.0.1`; unit `fraction`; direction
`lower`. This is the pinned fixed-`0.1 s` inverse-dynamics diagnostic, not a
physical-time-normalized feasibility metric.

| A - B | Validity and missingness | Raw mean | Median | Raw sign | Pointwise 0.95 band | Adjusted 0.9958333333333333 band | Oriented advantage | Gate |
|---|---|---:|---:|---|---|---|---:|---|
| `constant_velocity - log_replay` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `-0.04413296796842321` | `-0.03888274164858441` | negative | `[-0.04984095659342634, -0.038919178663599646]` | `[-0.052764339963662525, -0.036711842617360955]` | `0.04413296796842321` (A better) | nonzero `128`; `descriptive`; suppressed no; directional yes |
| `idm - log_replay` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `-0.044146410267774536` | `-0.03888274164858441` | negative | `[-0.04983342988405147, -0.03893367324809029]` | `[-0.05276528621539924, -0.036806857987709767]` | `0.044146410267774536` (A better) | nonzero `128`; `descriptive`; suppressed no; directional yes |
| `idm - constant_velocity` | `128/128/128/0/0`; both `0`; reasons `{}/{}` | `-1.3442299351326267e-05` | `0.0` | negative | `[-2.7997259380531403e-05, -1.0808611332096852e-06]` | `[-3.6473133948291926e-05, 3.2271037770856812e-06]` | `1.3442299351326267e-05` (A better) | nonzero `7`; `event_sparse`; suppressed no; directional **no** |

## Native Waymax parity and mapping oracles

The native comparison used the pre-registered, source-only subset of 16 scenarios,
the first 20 post-current transitions, all three EvalSim policies, and three parity
anchors. The 144 summary rows cover 48 scenario-policy rows per anchor.

| Parity anchor | Version | Compared components | Mismatches | Accepted interpretation |
|---|---:|---:|---:|---|
| Waymax `log_divergence` / EvalSim position error | `1.0.0` | `38,754` (`12,918` per policy) | `0` tolerance failures | Exact masks and all jointly valid continuous values within the frozen float32 tolerance. 24 of 48 rows had nonzero numerical error within tolerance, so this is **not** bit-exact parity. |
| Waymax `overlap` / EvalSim oriented-box overlap | `1.0.0` | `38,754` (`12,918` per policy) | `0` binary mismatches | Exact observed validity-mask and discrete-flag agreement on the bounded subset; not universal NumPy/XLA bit-equivalence at zero-margin geometry. |
| Waymax `kinematic_infeasibility` / EvalSim fixed-step diagnostic | `1.0.1` | `37,770` (`12,590` per policy) | `0` binary mismatches | Exact observed action-mask and binary-flag agreement for the fixed-`0.1 s` semantic; not a timestamp-normalized dynamics claim. |

The full-cohort pinned Waymax exact-log reference also matched EvalSim log replay over
the complete shared metric-result domain. All 640 logged-reference error-oracle
results—128 scenarios across five eligible logged-error metrics—were exactly zero
across their eligible components. These are mapping and construction oracles, not
evidence that log replay is an independent or causal simulator.

## Source-only slice and missingness accounting

Slice membership was computed from `Scenario` before a rollout, simulator name, metric
value, or policy difference was available. These are the complete eight slice counts;
no intersections were evaluated.

| Source-only slice | Members |
|---|---:|
| `all` | `128` |
| `vru_present_current` | `82` |
| `current_world_count_ge_8` | `122` |
| `retained_world_count_ge_16` | `121` |
| `observed_ego_turn_ge_15deg` | `0` |
| `low_current_cv_ttc_le_3s` | `64` |
| `future_lifecycle_change` | `128` |
| `supported_lane_available` | `105` |

The empty observed-ego-turn slice was retained rather than removed; all of its 39
scorecards are `insufficient_n`. Across the full fixed 312-cell matrix, status counts
are **217 `descriptive`**, **55 `small_or_sparse`**, **39 `insufficient_n`**, and
**1 `event_sparse`**.

The only metric-result missingness is the pre-registered
`no_supported_lane` reason for the two lane diagnostics. Each affected diagnostic is
missing for the same 23 scenarios in all four executions—92 result rows per
diagnostic—with symmetric policy pairing. No primary metric is affected. This report
does not selectively publish exploratory scorecard values; the complete generated
312-cell table remains local and ignored.

## Statistical reconstruction

- The finite-cohort estimand is the mean paired scene difference. The scenario, not
  an agent or frame, is the statistical and resampling unit.
- The 12 primary cells each use 100,000 resamples; the other 300 cells each use
  10,000. The base seed is `20260728`, the generator is NumPy `PCG64`, indices are
  `int64`, and percentiles use linear quantiles.
- All 312 cells use unique SHA-256-derived deterministic substreams. Independent
  reconstruction reproduced every point estimate and band exactly.
- The 95% pointwise bands and Bonferroni-adjusted
  `0.9958333333333333` bands measure sensitivity to empirical scene reweighting in
  this fixed cohort. They are not sampling-error estimates, confidence intervals,
  p-values, hypothesis tests, or familywise population statements.
- Fewer than 10 nonzero paired effects forces `event_sparse` and blocks directional
  language even when the pointwise band excludes zero.

## Interpretation and limitations

Position and speed error measure similarity to the recorded trajectory, so the
privileged replay oracle is expected to look favorable. The overlap target-frame rate
is a source-neutral geometric proxy, not a collision-pair count, severity measure, or
safety claim. The fixed-step kinematic diagnostic is intentionally tied to Waymax's
`0.1 s` inverse-dynamics semantics; actual positive source intervals still drive
EvalSim rollout integration and the actual-time derivative metrics.

The two non-replay policies have robustly lower rates than replay on the fixed-step
kinematic diagnostic, while replay has lower logged-error and overlap means. Constant
velocity has robustly lower position and speed error than IDM, while their apparent
overlap and kinematic differences do not pass the adjusted directional gate. These
contradictions are retained. They show why M5 does not collapse metrics into a
composite score or declare a winner.

The accepted claim is bounded by all of the following:

- The population is the unchanged 128-case complete-case conditional cohort selected
  from exactly ten validation shards. It is not representative of WOMD, Waymo
  operations, or a deployment distribution.
- All policies are deterministic seed-zero executions. M5 does not estimate
  stochastic-policy variance.
- Log replay uses the recorded future and is a privileged construction oracle, not an
  independent ground truth or deployable causal policy.
- Ego follows the logged/exogenous trajectory in M5 while world agents are simulated.
- The custom and native-reference paths share the same pinned Waymax WOMD decoder, so
  parity does not establish decoder independence.
- The official run revalidated and cryptographically bound the accepted local M4
  artifact set. The optional separate external M4 verification receipt was absent.
- Native parity is bounded to 16 scenarios × 20 post-current transitions × three
  policies. It does not prove numerical equivalence for every valid WOMD scene or
  floating-point backend.
- M5 does not support native-equivalent offroad, route progress, off-route,
  directional wrong-way, traffic-signal, or route-condition claims. The current
  contract does not preserve the fields needed for those semantics.
- Neutral-direction diagnostics are descriptive only and cannot support a
  favorable-policy or simulator-ranking claim.
- No result supports vehicle development, production performance, causal superiority,
  fleet scale, or commercial use.

## Reproducibility and provenance boundary

The accepted execution was bound to:

- EvalSim implementation commit
  `51c881af95d963ce05a638a8a5c1fee79a11757a`;
- accepted M4 cohort snapshot
  `a7a20e5de89c9c988f36a4b2f10ff4acc49246f0`;
- pinned Waymax commit
  `a64dfec9be8576b60d9cecc94f406d9812d4a7d0`; and
- WOMD v1.3.1 `tf_example` validation shards `00000`–`00009`.

After placing the ten local shards and retaining the accepted local M4 artifacts, the
official command shape is:

```bash
uv run --extra waymo env EVALSIM_RUN_WAYMO_LOCAL=1 \
  evalsim-m5-official \
  --project-root . \
  --data-dir <local-womd-validation-directory> \
  --m4-run-dir <accepted-local-m4-run-directory> \
  --run-name <fresh-lowercase-run-name>
```

The runner independently reloads and verifies the accepted cohort, exact source
identity, Git and dependency bindings, row domains, schemas, artifact hashes,
determinism receipts, reference-equality oracles, and parity gates before sealing
success. The cadence-domain correction that produced metric version `1.0.1` passed
876 tests with one expected local-data skip in the pinned Waymo environment and 790
tests with 28 expected optional/local skips in a fresh core-only environment.

WOMD shards, accepted M4 artifacts, detailed diagnostics, and generated M5
Parquet/JSON outputs remain ignored and local-only. They are never staged, committed,
pushed, uploaded, or deployed. This tracked report is a hand-curated aggregate
derivative, not a copy of the generated result store.
