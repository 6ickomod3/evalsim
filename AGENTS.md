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
- Before any commit or push, inspect both the working tree and staged file list and
  confirm that no dataset or generated experiment artifact is included.
- If a dataset file is staged accidentally, stop and remove it from the Git index without
  deleting the user's local copy, then report what happened.
- Do not bypass or weaken the dataset-related `.gitignore` rules.

## Project workflow

- Use `uv` and the repository-local `.venv`.
- Run `uv run pytest` after code changes.
- Preserve the contract-first architecture: downstream components consume the
  `Scenario`, `Rollout`, `SimulatorPolicy`, and `Metric` contracts rather than depending
  directly on WOMD or Waymax representations.
- From M3 onward, retain synthetic scenarios as analytic oracles but require a real-WOMD
  acceptance path for every core evaluation feature. Use Waymax repeatedly as a
  data/execution/reference adapter while keeping it behind the project contracts.

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
   end-to-end tests; exercise the required real-WOMD path; cross-check against an
   independent reference where possible; inspect negative and contradictory results;
   and obtain an adversarial review for semantic errors, leakage, and overclaiming.
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
- Ego is exogenous: M2 copies the logged ego trajectory while policies simulate world
  agents. M6 may replace the ego trajectory through a typed perturbation/controller,
  while world policies continue to observe the current ego state.
- The rollout engine owns timestamps, validity masks, lifecycle births/re-entries, and
  feasibility integration. Policies synchronously return typed controls or explicit
  absolute-state overrides and must not mutate observations or retain run-local state on
  reusable policy instances.
