"""Contract invariants: shape/range validation in __post_init__."""
from __future__ import annotations

import numpy as np
import pytest

from evalsim import Agent, AgentType, MapPolyline, MapType, Scenario


def _agent(T: int, agent_id: int = 0) -> Agent:
    return Agent(
        id=agent_id, type=AgentType.VEHICLE,
        valid=np.ones(T, dtype=bool), x=np.zeros(T), y=np.zeros(T),
        heading=np.zeros(T), vx=np.zeros(T), vy=np.zeros(T),
    )


def test_agent_mismatched_series_length_raises():
    with pytest.raises(ValueError):
        Agent(
            id=0, type=AgentType.VEHICLE,
            valid=np.ones(5, dtype=bool), x=np.zeros(4), y=np.zeros(5),
            heading=np.zeros(5), vx=np.zeros(5), vy=np.zeros(5),
        )


def test_map_polyline_bad_shape_raises():
    with pytest.raises(ValueError):
        MapPolyline(type=MapType.LANE, xy=np.zeros((3,)))


def test_scenario_agent_step_mismatch_raises():
    with pytest.raises(ValueError):
        Scenario(scenario_id="s", timestamps=np.zeros(10), agents=[_agent(9)])


def test_scenario_ego_index_out_of_range_raises():
    with pytest.raises(ValueError):
        Scenario(scenario_id="s", timestamps=np.zeros(10), agents=[_agent(10)], ego_index=5)


@pytest.mark.parametrize("ego_index", [True, np.bool_(True), 0.0])
def test_scenario_ego_index_must_be_an_integer(ego_index):
    with pytest.raises(ValueError, match="integer"):
        Scenario(
            scenario_id="s",
            timestamps=np.zeros(10),
            agents=[_agent(10)],
            ego_index=ego_index,
        )


def test_scenario_defaults_source_metadata():
    scn = Scenario(scenario_id="s", timestamps=np.zeros(10), agents=[_agent(10)])
    assert scn.metadata["source"] == "unknown"
    assert scn.ego is scn.agents[0]
    assert scn.num_steps == 10 and scn.num_agents == 1


def test_agent_speed():
    a = Agent(
        id=0, type=AgentType.VEHICLE, valid=np.ones(3, dtype=bool),
        x=np.zeros(3), y=np.zeros(3), heading=np.zeros(3),
        vx=np.array([3.0, 0.0, 3.0]), vy=np.array([4.0, 0.0, 4.0]),
    )
    np.testing.assert_allclose(a.speed(), [5.0, 0.0, 5.0])
