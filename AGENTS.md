# EvalSim repository instructions

## Maintaining these instructions

- Keep this `AGENTS.md` project-scoped. Do not copy its rules into
  `~/.codex/AGENTS.md`, global Codex guidance, or unrelated repositories.
- On every task, consider whether the user established a new durable, repository-wide
  convention. If so, update this file as part of the task and mention the change in the
  handoff. If not, leave this file unchanged.
- Record only durable project conventions, commands, safety constraints, and architectural
  decisions. Do not record temporary task details, transient state, secrets, credentials,
  personal data, or speculative preferences.
- Add or change durable instructions only when they follow from an explicit user decision
  or an established repository requirement; do not infer broad policy from a one-off task.

## Dataset safety

- Treat all downloaded and generated datasets as local-only artifacts.
- Never stage, commit, push, upload, or otherwise publish WOMD files or any other raw
  or processed dataset files. This includes everything under `data/`, TFRecord shards,
  Parquet datasets, downloaded archives, caches, run outputs, and generated experiment
  artifacts.
- Keep raw WOMD files immutable under
  `data/raw/womd/v1.3.1/tf_example/validation/`.
- Only small reproducibility metadata such as manifests, checksums, schemas, and download
  instructions may be committed. They must not contain dataset payloads.
- Detailed data-derived diagnostics may be written and inspected locally under ignored
  `outputs/` paths for debugging and learning. Keep them out of Git, deployments, and
  chat/public summaries; sanitize only the facts intentionally promoted across that
  publication boundary.
- Do not impose a general zero-terminal-output requirement on local experiments.
  Capture or suppress terminal output only when a pre-registered acceptance gate
  requires it; otherwise preserve useful local diagnostics and review them in place.
- Before any commit or push, inspect both the working tree and staged file list and
  confirm that no dataset or generated experiment artifact is included.
- If a dataset file is staged accidentally, stop and remove it from the Git index without
  deleting the user's local copy, then report what happened.
- Do not bypass or weaken the dataset-related `.gitignore` rules.

## Waymo and Waymax use restrictions

- The Waymo-backed path is scoped to the repository owner's stated personal,
  non-commercial interview preparation and experimentation. This is an operating
  assumption, not legal advice or a conclusion that every use is permitted.
- Keep the required Waymo Open Dataset attribution, Waymax prescribed notice/citation,
  canonical and pinned license links, and applicable restriction summary in
  `NOTICE.md`. Keep a prominent notice before optional Waymo installation instructions
  and direct notice text in any separately deployed presentation.
- Do not use the Waymo Open Dataset or Waymax path for paid/commercial services,
  Production Systems, real-world vehicle development/operation/testing/validation, or
  prohibited foundation-model development.
- If the actual purpose or publication scope changes, stop the affected work and obtain
  a fresh license review and any required permission before proceeding.
- Do not vendor, redistribute, or publish unmodified Waymax source, documentation,
  wheels, caches, or WOMD data. Pin upstream dependencies to immutable revisions and
  preserve the applicable notices in wheel and sdist artifacts.

## Project workflow

- The tracked .python-version pins the repository environment to exact CPython
  3.11.5; keep the repository-local .venv and default uv sync aligned with it.
- Use `uv` and the repository-local `.venv`.
- Recreate the development/test environment with
  `uv sync --frozen --extra dev --python 3.11.5`.
- Run `uv run pytest` after code changes.
- Preserve the contract-first architecture: downstream components consume the
  `Scenario`, `Rollout`, `SimulatorPolicy`, and `Metric` contracts rather than depending
  directly on WOMD or Waymax representations.
- Run the official M6 command only from the repository root as
  `.venv/bin/python -I -S -B evalsim/cli/m6_bootstrap.py ...`. Keep the unsafe
  `evalsim-m6-official` project console entry disabled; the tracked bootstrap is the
  required capture-first, isolated, no-site, no-bytecode boundary.
- Before official M6 execution, create the immutable runtime with
  `uv sync --frozen --extra waymo --python 3.11.5` (no `dev` extra), then require the
  bootstrap's complete environment-catalog recheck to pass before data access.
- Use synthetic analytic oracles first. Any claim promoted as real-scene evidence
  requires a separately gated real-WOMD acceptance path; learning, explanation, and
  data-free evaluator validation do not require WOMD. When real-scene evidence is in
  scope, use Waymax as a data/execution/reference adapter while keeping it behind the
  project contracts.

## Metrics-first project strategy

- Keep the primary project surface metrics-first and learning-oriented. Explain what
  each evaluator measures, rewards, and misses before presenting milestone breadth,
  simulator implementation detail, or optional technology plans.
- Use log replay, constant velocity, and EvalSim IDM as deliberately simple comparison
  probes with known failure modes. Do not describe them as three equivalent causal
  simulators or candidate production systems.
- Lead with the four canonical M5 primary metrics, grouped as fidelity
  (`position_error_m`, `speed_error_mps`), interaction proxy
  (`oriented_box_overlap_rate`), and feasibility diagnostic
  (`waymax_kinematic_infeasibility_rate`). Preserve their canonical names, versions,
  units, eligibility, directions, missingness, component distributions, and known
  limitations.
- Preserve metric plurality. Never collapse incompatible metrics into a composite
  realism score, overall simulator winner, or total ordering. Keep contradictory,
  sparse, null, and adverse results visible.
- Keep the core learning and presentation path CPU-first and independent of optional
  WOMD, Waymax, JAX, TensorFlow, Linux, or GPU requirements. Treat those dependencies
  as licensed reproduction/reference adapters or separately evidenced future
  extensions.
- Treat accepted pre-registrations, result reports, schemas, receipts, and source-bound
  claims as historical evidence. Narrative or API changes must not rewrite accepted
  definitions or imply that an accepted run executed at a newer commit.
- `AGENTS.md` participates in the official M5/M6 source catalogs. After changing this
  file, require fresh exact-source M6 review, approval, and the prescribed tag before
  any official M6 data access; prior source-snapshot approval does not carry forward.

## Milestone delivery workflow

For every milestone, follow this sequence:

1. **Understand and pre-register:** inspect the current implementation and evidence,
   confirm required local data/environment readiness, run the existing baseline tests,
   and define the hypothesis, scope, falsification criteria, acceptance gates, risks,
   and explicit non-goals before implementation.
2. **Plan:** write an implementation, verification, evidence, documentation, and
   rollback plan that preserves the project contracts and provenance boundaries.
3. **Review the plan:** adversarially review architecture, semantics, leakage,
   dataset/privacy safety, metric gaming, feasibility, and claim risk; revise the plan
   before execution.
4. **Execute:** implement incrementally with targeted tests and complete provenance.
   Keep plans and unverified results from being represented as completed evidence.
5. **Verify and review the execution:** run unit, contract, analytic-oracle, and
   end-to-end tests; exercise any pre-registered real-WOMD path required by the claim;
   cross-check against an independent reference where possible; inspect negative and
   contradictory results; and obtain an adversarial review for semantic errors, leakage,
   and overclaiming. Data-free claims remain bounded to their analytic or held-out
   evidence.
6. **Close documentation:** update the README, presentation/webpage, canonical roadmap,
   claim ledger, limitations, and any relevant schemas or small reproducibility metadata.
7. **Audit the release:** inspect the working tree and staged files for datasets,
   private material, generated artifacts, secrets, and unrelated changes; run the full
   test and presentation checks; and confirm every new claim has passed its evidence
   gate.
8. **Release:** commit, push, and deploy the presentation when applicable, using a
   milestone-scoped commit and without publishing local-only artifacts.
9. **Verify after release:** confirm the remote commit, deployed site/health checks, and
   public claims match the reviewed local state; record any follow-up or rollback need.

## Interview-material safety

- Store raw job descriptions, résumés, recruiter notes, interview scheduling details,
  self-assessments, and other personal inputs under `private/interview/`.
- Treat everything under `private/interview/` as local-only and sensitive. Never stage,
  commit, push, upload, deploy, or quote personal details into public project artifacts.
- Put only sanitized, project-relevant derivatives—competency matrices, revised
  milestones, learning plans, and claim-to-evidence ledgers—under `docs/interview/`.
- Before committing interview documentation, verify that it contains no contact details,
  private correspondence, scheduling information, or other personally identifying data.

## Rollout semantics

- `Scenario.metadata["current_index"]` is the last observed/current frame. The rollout
  engine copies history through that frame and simulates subsequent transitions. The
  synthetic source defaults to index 0; M3 adapters must set the real history boundary.
- At the pinned Waymax revision, `IDMRoutePolicy` projects each controlled agent onto
  that agent's complete logged future trajectory. Describe it as a privileged
  logged-trajectory waypoint-following IDM reference, not a causal map-route policy,
  independent ground truth, or numerical twin of EvalSim IDM.
- Ego is exogenous: M2 copies the logged ego trajectory while policies simulate world
  agents. M6 may replace the ego trajectory through a typed perturbation/controller,
  while world policies continue to observe the current ego state.
- Audited built-in policies must declare exactly one initialization capability:
  `HistoryOnlySimulatorPolicy` receives only immutable observed history, static context,
  and subsequently realized current state; `PrivilegedSimulatorPolicy` may receive an
  explicit defensive full-reference copy. Reject plain, dual-capability, or mismatched
  policies. Never give a history-only policy future validity, unrealized plan state,
  intervention identity, configured dose, or an absolute-state override capability.
- The rollout engine owns timestamps, validity masks, lifecycle births/re-entries, and
  feasibility integration. Policies synchronously return typed controls or explicit
  absolute-state overrides and must not mutate observations or retain run-local state on
  reusable policy instances.
- In a typed ego-plan transition, the world policy acts on the assembled current frame
  before the engine applies the next ego state. An ego change at `t+1` therefore cannot
  produce a world-state response before `t+2`.
