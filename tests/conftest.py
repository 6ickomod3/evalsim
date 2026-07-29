"""Shared test fixtures and helpers."""
from __future__ import annotations

import numpy as np
import pytest

from evalsim import Agent, AgentType, MapPolyline, MapType, Rollout, Scenario


def make_agent(agent_id: int, T: int, seed: int = 0, atype: AgentType = AgentType.VEHICLE) -> Agent:
    rng = np.random.default_rng(seed + agent_id)
    t = np.arange(T, dtype=float)
    x = float(agent_id) * 5.0 + 2.0 * t
    y = np.sin(0.1 * t) + agent_id
    heading = np.full(T, 0.1 * agent_id)
    vx = np.full(T, 2.0)
    vy = 0.1 * np.cos(0.1 * t)
    valid = np.ones(T, dtype=bool)
    if T > 2:  # exercise validity masks
        valid[0] = False
    return Agent(
        id=agent_id, type=atype, valid=valid, x=x, y=y, heading=heading,
        vx=vx, vy=vy, length=4.5 + 0.1 * agent_id, width=2.0,
    )


@pytest.fixture
def scenario() -> Scenario:
    T = 10
    agents = [make_agent(i, T, seed=1) for i in range(3)]
    agents.append(make_agent(3, T, seed=1, atype=AgentType.PEDESTRIAN))
    polylines = [
        MapPolyline(type=MapType.LANE, xy=np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 1.0]])),
        MapPolyline(type=MapType.ROAD_EDGE, xy=np.array([[0.0, -2.0], [20.0, -2.0]])),
    ]
    return Scenario(
        scenario_id="scn_0001",
        timestamps=np.arange(T, dtype=float) * 0.1,
        agents=agents,
        map=polylines,
        ego_index=0,
        metadata={"source": "synthetic", "tags": ["intersection", "pedestrian_present"]},
    )


@pytest.fixture
def rollout(scenario: Scenario) -> Rollout:
    return Rollout(
        scenario_id=scenario.scenario_id,
        sim_name="log_replay",
        sim_version="0.1.0",
        seed=42,
        timestamps=scenario.timestamps.copy(),
        agents=[
            make_agent(
                i,
                scenario.num_steps,
                seed=2,
                atype=scenario.agents[i].type,
            )
            for i in range(scenario.num_agents)
        ],
        perturbation=None,
        metadata={"note": "unit-test rollout"},
    )


def assert_agents_equal(a: Agent, b: Agent) -> None:
    assert a.id == b.id
    assert a.type == b.type
    assert a.length == pytest.approx(b.length)
    assert a.width == pytest.approx(b.width)
    np.testing.assert_array_equal(a.valid, b.valid)
    for f in ("x", "y", "heading", "vx", "vy"):
        np.testing.assert_allclose(getattr(a, f), getattr(b, f))
