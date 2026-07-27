"""M0 acceptance: Scenario/Rollout round-trip to/from Parquet losslessly."""
from __future__ import annotations

import numpy as np
import pytest

from evalsim import (
    Rollout,
    Scenario,
    rollout_from_parquet,
    rollout_to_parquet,
    scenario_from_parquet,
    scenario_to_parquet,
)
from tests.conftest import assert_agents_equal


def test_scenario_roundtrip(scenario: Scenario, tmp_path):
    path = tmp_path / "scenario.parquet"
    scenario_to_parquet(scenario, path)
    loaded = scenario_from_parquet(path)

    assert loaded.scenario_id == scenario.scenario_id
    assert loaded.ego_index == scenario.ego_index
    assert loaded.metadata == scenario.metadata
    np.testing.assert_allclose(loaded.timestamps, scenario.timestamps)

    assert loaded.num_agents == scenario.num_agents
    for a, b in zip(loaded.agents, scenario.agents):
        assert_agents_equal(a, b)

    assert len(loaded.map) == len(scenario.map)
    for p, q in zip(loaded.map, scenario.map):
        assert p.type == q.type
        np.testing.assert_allclose(p.xy, q.xy)


def test_rollout_roundtrip(rollout: Rollout, tmp_path):
    path = tmp_path / "rollout.parquet"
    rollout_to_parquet(rollout, path)
    loaded = rollout_from_parquet(path)

    assert loaded.scenario_id == rollout.scenario_id
    assert loaded.sim_name == rollout.sim_name
    assert loaded.sim_version == rollout.sim_version
    assert loaded.seed == rollout.seed
    assert loaded.perturbation == rollout.perturbation
    assert loaded.metadata == rollout.metadata
    np.testing.assert_allclose(loaded.timestamps, rollout.timestamps)
    for a, b in zip(loaded.agents, rollout.agents):
        assert_agents_equal(a, b)


def test_wrong_kind_rejected(scenario: Scenario, tmp_path):
    path = tmp_path / "scenario.parquet"
    scenario_to_parquet(scenario, path)
    with pytest.raises(ValueError):
        rollout_from_parquet(path)  # it's a scenario file, not a rollout
