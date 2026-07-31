"""M7 detection matrix: which evaluator detects which defect, and the blind spots.

The central M7 negative result is encoded here as an oracle: the kinematic-infeasibility
evaluator is BLIND to frozen (nonreactive) world agents -- a held agent has zero,
constant velocity, which the inverse-dynamics check reads as perfectly feasible -- even
though nonreactivity is exactly the M6 failure mode. Position error catches it; the
infeasibility rate does not. That is a plausible metric shown to be misleading.
"""
from __future__ import annotations

import copy

import numpy as np

from evalsim import Agent, AgentType, Rollout, Scenario
from evalsim.metrics.m5 import (
    OrientedBoxOverlapRateMetric,
    PositionErrorMetric,
    WaymaxKinematicInfeasibilityRateMetric,
)
from evalsim.stress.defects import default_defect_registry
from evalsim.stress.detection import (
    DetectionCase,
    DetectionCell,
    cell,
    detection_matrix,
)


def _agent(agent_id, t, **s) -> Agent:
    n = len(t)
    d = {"x": 0.0, "y": 0.0, "heading": 0.0, "vx": 0.0, "vy": 0.0}
    d.update(s)
    arr = lambda v, dt=float: (  # noqa: E731
        np.full(n, v, dtype=dt) if np.ndim(v) == 0 else np.array(v, dtype=dt)
    )
    return Agent(
        id=agent_id, type=AgentType.VEHICLE, valid=arr(s.get("valid", True), bool),
        x=arr(d["x"]), y=arr(d["y"]), heading=arr(d["heading"]),
        vx=arr(d["vx"]), vy=arr(d["vy"]), length=2.0, width=2.0,
    )


def _case(idx: int, n_world: int = 4, steps: int = 6, current_index: int = 1):
    t = np.arange(steps, dtype=float) * 0.1
    source = [_agent(0, t, x=-100.0 - idx)]
    for k in range(n_world):
        source.append(_agent(10 + k, t, x=(2.0 + k) * t + idx, y=50.0 * k, vx=2.0 + k))
    scenario = Scenario(
        scenario_id=f"m7-det-{idx}", timestamps=np.array(t, copy=True), agents=source,
        ego_index=0, metadata={"source": "unit", "current_index": current_index},
    )
    clean = Rollout(
        scenario_id=scenario.scenario_id, sim_name="candidate", sim_version="1.0.0",
        seed=0, timestamps=np.array(t, copy=True), agents=copy.deepcopy(source),
    )
    return DetectionCase(scenario=scenario, clean_rollout=clean)


def _matrix():
    cases = [_case(i) for i in range(3)]
    metrics = [
        PositionErrorMetric(),
        WaymaxKinematicInfeasibilityRateMetric(),
        OrientedBoxOverlapRateMetric(),
    ]
    return detection_matrix(
        default_defect_registry(),
        metrics,
        cases,
        severities=(0.25, 0.5, 0.75, 1.0),
        seed=11,
    )


def test_matrix_has_a_cell_per_family_metric_pair() -> None:
    matrix = _matrix()
    assert len(matrix) == 3 * 3
    assert all(isinstance(c, DetectionCell) for c in matrix)


def test_position_error_detects_frozen_monotone() -> None:
    c = cell(_matrix(), "frozen_agent", "position_error_m")
    assert c.clean_value == 0.0
    assert c.detected is True
    assert c.monotone is True


def test_infeasibility_is_blind_to_frozen_agents() -> None:
    # The M7 negative result: nonreactive (frozen) agents are kinematically feasible.
    c = cell(_matrix(), "frozen_agent", "waymax_kinematic_infeasibility_rate")
    assert c.clean_value == 0.0
    assert c.detected is False


def test_infeasibility_detects_teleportation() -> None:
    c = cell(_matrix(), "teleportation", "waymax_kinematic_infeasibility_rate")
    assert c.detected is True


def test_overlap_rate_detects_overlap_defect() -> None:
    c = cell(_matrix(), "overlap", "oriented_box_overlap_rate")
    assert c.detected is True


def test_overlap_rate_is_blind_to_frozen_agents() -> None:
    # Freezing agents that stay separated creates no interpenetration.
    c = cell(_matrix(), "frozen_agent", "oriented_box_overlap_rate")
    assert c.detected is False
