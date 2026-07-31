# M7 — Evaluator red-team and metric governance

**Date:** 2026-07-31
**Status:** Data-free foundation implemented and verified; WOMD detection matrix, invariance
harness, and calibration/held-out split still pending (see Implementation status below)

## Implementation status (2026-07-31)

The **data-free** M7 foundation is implemented in `evalsim/stress/` and verified by 27
analytic-oracle tests (`tests/test_m7_defects.py`, `test_m7_detection.py`,
`test_m7_metric_cards.py`, `test_m7_invariance.py`), each independently reviewed by an
adversarial subagent (no P0/P1; two P2s fixed — cohort ordinals are now 0-based cohort
ranks, and the manifest test was tightened). Landed:

- a typed, seeded defect framework (`DefectSpec`, `DefectManifest`, `Defect`,
  `DefectRegistry`) with strict severity-0 identity, no in-place mutation, and sanitized
  manifests (0-based cohort ranks only — no native id/coordinate/payload);
- three severity-controlled families with monotone analytic oracles: `frozen_agent`
  (nonreactivity), `teleportation` (position jump + implied velocity spike),
  `overlap` (forced interpenetration);
- a detection matrix (`detection.py`) computing per (defect × metric × severity) clean
  baseline, severity curve, `detected`, and `monotone`;
- metric governance cards (`metric_cards.py`) that record each evaluator's detected
  families and blind spots;
- an invariance-probe harness (`invariance.py`): agent-order permutation and rigid
  global translation leave the M5 error/overlap metrics unchanged, and a
  semantics-breaking control (translating only the rollout) is correctly flagged
  non-invariant, so the harness is proven to detect real violations;
- **the required negative result**: the `waymax_kinematic_infeasibility_rate` evaluator is
  **blind to frozen (nonreactive) world agents** (a held agent has zero, constant velocity
  = feasible), while `position_error_m` catches them — a plausible metric shown to be
  misleading, tying directly to the M6 nonreactivity theme. Overlap-rate is likewise blind
  to freezing.

Still pending before an accepted M7 result: a calibration/held-out defect split; the
broader defect taxonomy (kinematic spikes, off-road/route, dispersion, identity/mask bugs);
and the WOMD detection matrix on the accepted M4 cohort (deferred to a separate accepted
run, like M5/M6 — needs pre-registration acceptance and a data-access decision). No WOMD
data was opened for this foundation.
**Governing roadmap:** [Waymo-aligned roadmap](2026-07-28-waymo-aligned-roadmap.md) (M7 section)
**Depends on:** M5 metric system (accepted). Does **not** depend on the M6 scientific
result, so M7 can proceed in parallel with the gated M6 run.

## 0. Purpose and honesty boundary

M5 shipped thirteen hand-designed motion metrics with analytic oracles and a fixed-cohort
WOMD application, but the claim ledger keeps "metric stress testing" and "hand-designed
realism evaluators" at **Partial / Not yet**: adversarial *software* tests are not
metric-*validation* results. M7 closes that gap by red-teaming the evaluators themselves:
inject known, severity-controlled defects and measure whether each metric detects them
**for the right reason**, what it misses, and what it falsely flags.

The M7 claim unlocked only after acceptance: *"developed a severity-calibrated metric
stress suite that exposed evaluator blind spots."* M7 does **not** claim real-traffic
realism, causality, simulator superiority, or that any metric is a validated safety
signal beyond the registered defect families.

## 1. Question, estimand, hypotheses, claim boundary

### 1.1 Question
Do the M5 evaluators detect known defects for the right reason, and what do they miss or
falsely flag across a controlled severity range?

### 1.2 Estimand
For each (metric, defect-family) pair, the metric's **detection curve** over injected
severity — plus its **false-positive rate** on defect-free (clean) scenes — computed on a
fixed, source-only cohort. Thresholds are selected on a **calibration** defect set and
reported on a disjoint **held-out** defect set. All quantities are conditional on the
fixed cohort and the registered defect generators; they are not population claims.

### 1.3 Falsifiable software hypotheses
- Every defect generator is a pure, seeded, source-only transform of a `Scenario`/rollout
  that produces a typed, reproducible corruption with a declared severity parameter.
- Clean (severity 0) scenes are a strict identity: the generator at severity 0 returns a
  byte-identical contract, and all metrics score them within their no-defect band.
- Invariance probes (agent-order permutation, validity padding, global rigid
  translation/rotation, serialization round-trip) leave every registered metric's value
  unchanged within a declared tolerance.

### 1.4 Falsifiable analytic scientific hypotheses (synthetic, before WOMD)
- For each defect family there exists at least one metric whose value is monotone
  non-decreasing in severity on synthetic oracles.
- At least one **plausible** metric is demonstrably **misleading** on at least one defect
  family (insensitive, or sensitive in the wrong direction) — a required negative result.
- Detection is characterized by a curve across the full severity range, never a single
  hand-picked example.

### 1.5 Real-WOMD expectation (fixed cohort)
Applying the calibrated thresholds to defect-injected variants of the accepted 128-scene
cohort yields a detection matrix (true/false positives and negatives) with finite-cohort
stability bands, honest about which metrics are blind to which families.

### 1.6 Bounded claim after acceptance
A severity-calibrated metric stress suite with a data-derived detection matrix (including
false positives, false negatives, and at least one exposed blind spot), conditional on
the fixed cohort and the registered defect generators. No realism, causal, safety, or
population claim.

## 2. Frozen defect taxonomy (severity-controlled)

Each family is a typed generator `defect(scenario_or_rollout, severity: float, seed) ->`
corrupted contract, with severity in a declared closed interval and a documented physical
meaning. Candidate families (final set fixed at plan review):

1. **Nonreactive / frozen agents** — freeze a fraction/severity of world agents at their
   current state.
2. **Teleportation** — inject position discontinuities of controlled magnitude.
3. **Kinematic spikes** — jerk / yaw-rate impulses of controlled amplitude.
4. **Infeasible motion** — accelerations beyond the audited dynamics envelope.
5. **Overlap / collision** — force controlled bounding-box interpenetration.
6. **Off-road / wrong-way** — displace agents off supported lanes / reverse heading
   (uses typed map context only where available; otherwise labeled proxy).
7. **Trajectory collapse** — collapse dispersion toward a single mode.
8. **Excessive dispersion** — inflate spread beyond plausible bounds.
9. **Route / condition violation** — depart from the logged route/condition envelope.
10. **Identity / mask / coordinate bugs** — agent-id shuffles, validity-mask corruption,
    coordinate-frame errors (these double as invariance-probe adversaries).

Each generator: pure, seeded, reversible-at-severity-0, and emits a typed
`DefectManifest` (family, severity, seed, affected-agent accounting) with no native
identity or source coordinates in any promoted output.

## 3. Invariance probes

A metric that changes under a semantics-preserving transform is buggy. Probes applied to
every registered metric on clean scenes: agent-order permutation, validity padding,
global rigid translation + rotation, and serialization round-trip. Each metric declares
an expected invariance and a numeric tolerance; violations fail the release-candidate
gate.

## 4. Metric cards (governance)

Every evaluator (M5's thirteen + any M7 additions) receives a versioned **metric card**:
intended use, statistical unit, eligibility, declared invariances, expected sensitivity,
known blind spots, calibration population, threshold rationale, version, and owner.
Cards are the governance artifact; a metric without a complete card is not
release-candidate.

## 5. Calibration and held-out evaluation

- Split defect instances into a **calibration** set (threshold selection) and a disjoint
  **held-out** set (reporting), partitioned by source scenario so no scene appears in
  both.
- Thresholds are chosen on calibration defects only; all detection/FP/FN numbers are
  reported on held-out defects and on clean scenes.
- No threshold is tuned on the held-out set; a test proves partition disjointness.

## 6. Detection matrix and statistics

- For each (metric × defect-family × severity) cell: detection rate; for clean scenes:
  false-positive rate. Aggregate into a metric × family **detection matrix**.
- Uncertainty uses the same deterministic finite-cohort scene-reweighting bands as M5
  (not confidence intervals / hypothesis tests). Retain null, sparse, adverse, and
  contradictory results.
- Report severity curves, not single points.

## 7. Evidence gates (acceptance)

1. Release-candidate metrics pass identity, invariance, and expected-direction tests.
2. Detection is measured across severity curves, not one example.
3. The data-derived detection matrix includes false positives, false negatives, and
   stability bands.
4. At least one plausible metric is shown to be misleading, with root-cause analysis.
5. Adversarial review finds no trivial provenance, padding, or preprocessing shortcut in
   the defect generators or the detection pipeline.
6. Analytic oracles exist for every generator and every claimed detection direction.

## 8. Explicit non-goals

- No learned evaluator (that is M8).
- No new real-traffic realism, causal, safety, or simulator-superiority claim.
- No change to the M5 metric definitions or the accepted M5 result; M7 adds a separate
  validation layer on top.
- No off-road/route/wrong-way *semantic* claim beyond typed map context; proxies stay
  labeled as proxies.
- No dataset, generated defect corpus, or detection artifact is committed (local-only per
  `AGENTS.md`).

## 9. Proposed implementation slices (post-review)

1. `evalsim/stress/` defect-generator framework: typed `DefectSpec`/`DefectManifest`,
   a registry, and the severity-0 identity + purity/seed tests.
2. Two or three defect families end-to-end (e.g., frozen-agent, teleportation, overlap)
   with analytic oracles and monotone-detection tests, on synthetic scenes.
3. Invariance-probe harness over all M5 metrics.
4. Metric-card schema + cards for the M5 thirteen.
5. Calibration/held-out split + detection-matrix computation on synthetic, then the fixed
   WOMD cohort (reusing the accepted M4 cohort, source-only).
6. The required "misleading metric" negative result + root-cause write-up.
7. Close documentation (README, roadmap, claim ledger) and release audit.

## 10. Open design questions for plan review

- Final defect-family set and each family's severity parameterization/units.
- Which metric to target for the mandated "misleading" negative result.
- Whether the WOMD detection matrix runs now (needs `waymo` extra + the regenerated M4
  cohort, both available locally) or stays synthetic-only until a separate data gate.
- Metric-card storage format (Markdown vs typed schema) and where cards live.
