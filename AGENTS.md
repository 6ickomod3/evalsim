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
