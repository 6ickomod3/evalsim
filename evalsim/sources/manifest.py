"""Immutable scenario manifests used to pin evaluation populations."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .identity import (
    SYNTHETIC_SOURCE_VERSION,
    parse_synthetic_scenario_id,
    synthetic_source_fingerprint,
)

SCENARIO_MANIFEST_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class ScenarioManifest:
    """An ordered, reproducible list of scenario IDs.

    The object is frozen and stores IDs in a tuple.  ``to_file`` uses exclusive-create
    mode, so a manifest already on disk can never be silently changed in place.  Schema
    version 1 covers the M1 synthetic source; later producers can add source-specific
    provenance under a new schema without weakening these checks.
    """

    scenario_ids: tuple[str, ...]
    seed: int
    num_steps: int
    dt: float
    split: str
    source_fingerprint: str
    source: str = "synthetic"
    source_version: str = SYNTHETIC_SOURCE_VERSION
    schema_version: str = SCENARIO_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.scenario_ids, (str, bytes)):
            raise ValueError("scenario_ids must be a sequence of IDs, not a string")
        scenario_ids = tuple(self.scenario_ids)
        if any(not isinstance(item, str) or not item for item in scenario_ids):
            raise ValueError("scenario_ids must contain only non-empty strings")
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario_ids must be unique")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("split must be a non-empty string")
        object.__setattr__(self, "split", self.split.strip())
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 2**32 - 1
        ):
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")
        if (
            isinstance(self.num_steps, bool)
            or not isinstance(self.num_steps, int)
            or self.num_steps < 10
        ):
            raise ValueError("num_steps must be an integer >= 10")
        if (
            isinstance(self.dt, bool)
            or not isinstance(self.dt, (int, float))
            or not math.isfinite(float(self.dt))
            or self.dt <= 0.0
        ):
            raise ValueError("dt must be a finite positive number")
        try:
            duration = (self.num_steps - 1) * float(self.dt)
        except OverflowError as exc:
            raise ValueError("scenario duration must be finite") from exc
        if not 0.01 <= float(self.dt) <= 1.0:
            raise ValueError("dt must be in the practical range [0.01, 1.0] seconds")
        if not math.isfinite(duration) or not 6.0 <= duration <= 120.0:
            raise ValueError("scenario duration must be finite and in [6, 120] seconds")
        for name in (
            "split",
            "source_fingerprint",
            "source",
            "source_version",
            "schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema_version != SCENARIO_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported scenario manifest schema version "
                f"{self.schema_version!r}"
            )
        if self.source != "synthetic":
            raise ValueError(f"unsupported scenario manifest source {self.source!r}")
        if self.source_version != SYNTHETIC_SOURCE_VERSION:
            raise ValueError(
                f"unsupported synthetic source version {self.source_version!r}"
            )
        expected_fingerprint = synthetic_source_fingerprint(
            seed=self.seed,
            num_steps=self.num_steps,
            dt=float(self.dt),
            split=self.split,
            source_version=self.source_version,
        )
        if self.source_fingerprint != expected_fingerprint:
            raise ValueError(
                "source_fingerprint does not match the synthetic source configuration"
            )
        for scenario_id in scenario_ids:
            _, fingerprint, _ = parse_synthetic_scenario_id(scenario_id)
            if fingerprint != expected_fingerprint:
                raise ValueError(
                    f"scenario ID {scenario_id!r} does not match "
                    "source_fingerprint"
                )
        object.__setattr__(self, "scenario_ids", scenario_ids)
        object.__setattr__(self, "dt", float(self.dt))

    @property
    def count(self) -> int:
        return len(self.scenario_ids)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "source_version": self.source_version,
            "source_fingerprint": self.source_fingerprint,
            "seed": self.seed,
            "num_steps": self.num_steps,
            "dt": self.dt,
            "split": self.split,
            "count": self.count,
            "scenario_ids": list(self.scenario_ids),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ScenarioManifest":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("scenario manifest JSON must contain an object")
        raw_scenario_ids = payload.pop("scenario_ids")
        if not isinstance(raw_scenario_ids, list):
            raise ValueError("scenario_ids must be a JSON array")
        scenario_ids = tuple(raw_scenario_ids)
        declared_count = payload.pop("count")
        manifest = cls(scenario_ids=scenario_ids, **payload)
        if (
            isinstance(declared_count, bool)
            or not isinstance(declared_count, int)
            or declared_count != manifest.count
        ):
            raise ValueError(
                f"manifest count is {declared_count}, but it contains "
                f"{manifest.count} scenario IDs"
            )
        return manifest

    def to_file(self, path: str | Path) -> None:
        """Write once; raise ``FileExistsError`` instead of replacing a manifest."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(self.to_json())
            handle.write("\n")

    @classmethod
    def from_file(cls, path: str | Path) -> "ScenarioManifest":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
