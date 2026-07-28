"""M1 immutable scenario-manifest tests."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from evalsim.sources import ScenarioManifest, SyntheticSource


def test_manifest_preserves_order_and_round_trips(tmp_path) -> None:
    source = SyntheticSource(seed=17)
    scenarios = source.generate(7)
    path = tmp_path / "manifests" / "synthetic_eval.json"

    manifest = source.write_manifest(scenarios, path)
    restored = ScenarioManifest.from_file(path)

    assert restored == manifest
    assert restored.count == 7
    assert restored.scenario_ids == tuple(
        scenario.scenario_id for scenario in scenarios
    )
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_manifest_normalizes_split_like_source() -> None:
    source = SyntheticSource(seed=17, split=" train ")
    scenario = source.generate_one(0)
    manifest = ScenarioManifest(
        scenario_ids=(scenario.scenario_id,),
        seed=source.seed,
        num_steps=source.num_steps,
        dt=source.dt,
        split=" train ",
        source_fingerprint=source.fingerprint,
    )

    assert source.split == "train"
    assert manifest.split == "train"


def test_manifest_refuses_to_overwrite_existing_file(tmp_path) -> None:
    source = SyntheticSource(seed=17)
    scenarios = source.generate(3)
    path = tmp_path / "manifest.json"
    source.write_manifest(scenarios, path)
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        source.write_manifest(source.generate(4), path)

    assert path.read_bytes() == original


def test_manifest_rejects_duplicate_ids() -> None:
    source = SyntheticSource()
    scenario = source.generate_one(0)

    with pytest.raises(ValueError, match="unique"):
        ScenarioManifest(
            scenario_ids=(scenario.scenario_id, scenario.scenario_id),
            seed=source.seed,
            num_steps=source.num_steps,
            dt=source.dt,
            split=source.split,
            source_fingerprint=source.fingerprint,
        )


def test_manifest_rejects_tampered_count() -> None:
    source = SyntheticSource()
    manifest = source.make_manifest(source.generate(2))
    payload = json.loads(manifest.to_json())
    payload["count"] = 999

    with pytest.raises(ValueError, match="contains 2"):
        ScenarioManifest.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 18),
        ("dt", 0.2),
        ("num_steps", 101),
        ("split", "train"),
        ("source_fingerprint", "000000000000"),
        ("source", "womd"),
        ("source_version", "999"),
        ("schema_version", "999"),
    ],
)
def test_manifest_rejects_tampered_provenance(field: str, value: object) -> None:
    source = SyntheticSource(seed=17)
    payload = json.loads(source.make_manifest(source.generate(2)).to_json())
    payload[field] = value

    with pytest.raises(ValueError):
        ScenarioManifest.from_json(json.dumps(payload))


def test_manifest_rejects_noncanonical_scenario_id() -> None:
    source = SyntheticSource(seed=17)
    payload = json.loads(source.make_manifest(source.generate(2)).to_json())
    payload["scenario_ids"][0] = "not-regenerable"

    with pytest.raises(ValueError, match="scenario ID"):
        ScenarioManifest.from_json(json.dumps(payload))


def test_source_rejects_scenarios_from_another_configuration() -> None:
    source = SyntheticSource(seed=1)
    other_scenario = SyntheticSource(seed=2).generate_one(0)

    with pytest.raises(ValueError, match="not generated"):
        source.make_manifest([other_scenario])


def test_source_rejects_mutated_generated_scenario() -> None:
    source = SyntheticSource(seed=1)
    scenario = source.generate_one(0)
    scenario.timestamps[:] = 123.0

    with pytest.raises(ValueError, match="modified"):
        source.make_manifest([scenario])


def test_source_rejects_forged_id_or_metadata() -> None:
    source = SyntheticSource(seed=1)
    scenario = source.generate_one(0)
    forged_id = replace(scenario, scenario_id="not-regenerable")
    with pytest.raises(ValueError, match="not generated"):
        source.make_manifest([forged_id])

    alien = SyntheticSource(seed=2).generate_one(0)
    alien.metadata["source_fingerprint"] = source.fingerprint
    with pytest.raises(ValueError, match="not generated"):
        source.make_manifest([alien])
