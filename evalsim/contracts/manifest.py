"""Run manifest (draft §19).

Every reported result must be traceable to a complete manifest so any résumé number can
be regenerated. Stored as JSON alongside a run's results.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunManifest:
    """Complete, reproducible description of one evaluation run."""

    run_id: str
    dataset_version: str
    scenario_manifest: str
    sim_name: str
    sim_version: str
    sim_config: dict = field(default_factory=dict)
    num_rollouts: int = 1
    seeds: list[int] = field(default_factory=lambda: [0])
    ego_control: str = "logged"
    perturbations: list[str] = field(default_factory=list)
    metric_versions: dict[str, str] = field(default_factory=dict)
    slice_versions: dict[str, str] = field(default_factory=dict)
    statistics: dict = field(default_factory=dict)
    code_commit: str = "unknown"
    environment: dict = field(default_factory=dict)
    output_dir: str = ""
    timestamp: str = ""
    parent_run: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RunManifest":
        return cls(**json.loads(text))

    def to_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def from_file(cls, path: str | Path) -> "RunManifest":
        return cls.from_json(Path(path).read_text())
