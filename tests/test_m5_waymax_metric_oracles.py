"""Data-free numerical cross-checks against the pinned optional Waymax runtime."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from evalsim import Agent, AgentType, Rollout, Scenario
from evalsim.metrics import (
    kinematic_infeasibility_components,
    oriented_box_overlap_components,
    position_divergence_components,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("waymax")

from waymax import datatypes
from waymax.dynamics import bicycle_model
from waymax.metrics import comfort, imitation, overlap
from waymax.utils import geometry


def _agent(
    agent_id: int,
    *,
    valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    length: float,
    width: float,
) -> Agent:
    return Agent(
        id=agent_id,
        type=AgentType.VEHICLE,
        valid=valid,
        x=x,
        y=y,
        heading=heading,
        vx=vx,
        vy=vy,
        length=length,
        width=width,
    )


def _contract_pair() -> tuple[Scenario, Rollout]:
    frames = 6
    timestamps = (
        np.arange(frames, dtype=np.int64) * 100_000
    ).astype(np.float64) * 1e-6
    valid = (
        np.array([True, True, True, True, True, True]),
        np.array([True, True, False, True, True, True]),
        np.array([True, True, True, True, False, True]),
    )
    logged = [
        _agent(
            10,
            valid=valid[0],
            x=np.linspace(0.0, 2.5, frames),
            y=np.zeros(frames),
            heading=np.zeros(frames),
            vx=np.full(frames, 5.0),
            vy=np.zeros(frames),
            length=4.5,
            width=2.0,
        ),
        _agent(
            11,
            valid=valid[1],
            x=np.linspace(8.0, 5.5, frames),
            y=np.linspace(0.0, 0.2, frames),
            heading=np.linspace(0.0, 0.08, frames),
            vx=np.linspace(-2.0, -1.0, frames),
            vy=np.linspace(0.0, 0.2, frames),
            length=4.0,
            width=1.8,
        ),
        _agent(
            12,
            valid=valid[2],
            x=np.full(frames, 15.0),
            y=np.linspace(3.0, 2.0, frames),
            heading=np.full(frames, -0.2),
            vx=np.zeros(frames),
            vy=np.full(frames, -1.0),
            length=1.8,
            width=0.7,
        ),
    ]
    candidate = [
        _agent(
            agent.id,
            valid=np.array(agent.valid, copy=True),
            x=np.asarray(agent.x) + np.linspace(0.0, 0.15, frames),
            y=np.asarray(agent.y) + np.linspace(0.0, -0.08, frames),
            heading=np.asarray(agent.heading) + np.linspace(0.0, 0.03, frames),
            vx=np.asarray(agent.vx) + np.linspace(0.0, 0.4, frames),
            vy=np.asarray(agent.vy) + np.linspace(0.0, -0.2, frames),
            length=agent.length,
            width=agent.width,
        )
        for agent in logged
    ]
    scenario = Scenario(
        scenario_id="m5-waymax-oracle",
        timestamps=timestamps,
        agents=logged,
        ego_index=0,
        metadata={"source": "synthetic", "current_index": 1},
    )
    rollout = Rollout(
        scenario_id=scenario.scenario_id,
        sim_name="oracle_candidate",
        sim_version="1.0.0",
        seed=0,
        timestamps=timestamps,
        agents=candidate,
    )
    return scenario, rollout


def _native_trajectory(agents: list[Agent], timestamps: np.ndarray):
    objects = len(agents)
    frames = len(timestamps)

    def stack(field: str, dtype=np.float32):
        return np.asarray(
            [np.asarray(getattr(agent, field), dtype=dtype) for agent in agents]
        )

    timestamp_micros = np.broadcast_to(
        np.rint(timestamps * 1_000_000.0).astype(np.int32),
        (objects, frames),
    ).copy()
    length = np.broadcast_to(
        np.asarray([agent.length for agent in agents], dtype=np.float32)[:, None],
        (objects, frames),
    ).copy()
    width = np.broadcast_to(
        np.asarray([agent.width for agent in agents], dtype=np.float32)[:, None],
        (objects, frames),
    ).copy()
    return datatypes.Trajectory(
        x=jnp.asarray(stack("x")),
        y=jnp.asarray(stack("y")),
        z=jnp.zeros((objects, frames), dtype=jnp.float32),
        vel_x=jnp.asarray(stack("vx")),
        vel_y=jnp.asarray(stack("vy")),
        yaw=jnp.asarray(stack("heading")),
        valid=jnp.asarray(stack("valid", bool)),
        timestamp_micros=jnp.asarray(timestamp_micros),
        length=jnp.asarray(length),
        width=jnp.asarray(width),
        height=jnp.ones((objects, frames), dtype=jnp.float32),
    )


def _one_frame(trajectory, frame: int):
    return datatypes.Trajectory(
        **{
            field: getattr(trajectory, field)[:, frame : frame + 1]
            for field in (
                "x",
                "y",
                "z",
                "vel_x",
                "vel_y",
                "yaw",
                "valid",
                "timestamp_micros",
                "length",
                "width",
                "height",
            )
        }
    )


def test_position_components_match_pinned_waymax_float32_oracle() -> None:
    scenario, rollout = _contract_pair()
    custom, custom_valid = position_divergence_components(scenario, rollout)
    native_sim = _native_trajectory(rollout.agents, rollout.timestamps)
    native_log = _native_trajectory(scenario.agents, scenario.timestamps)
    native = np.asarray(
        imitation.LogDivergenceMetric.compute_log_divergence(
            native_sim.xy,
            native_log.xy,
        )
    ).T
    native_valid = (
        np.asarray(native_sim.valid) & np.asarray(native_log.valid)
    ).T
    np.testing.assert_array_equal(custom_valid, native_valid)
    references = native[native_valid]
    tolerances = np.maximum(
        1e-6,
        8.0
        * (
            np.nextafter(
                np.abs(references).astype(np.float32),
                np.float32(np.inf),
                dtype=np.float32,
            )
            - np.abs(references).astype(np.float32)
        ),
    )
    assert np.all(np.abs(custom[custom_valid] - references) <= tolerances)


def test_overlap_flags_and_masks_match_pinned_waymax_oracle() -> None:
    scenario, rollout = _contract_pair()
    custom, custom_valid = oriented_box_overlap_components(scenario, rollout)
    native_trajectory = _native_trajectory(rollout.agents, rollout.timestamps)
    native_values = np.zeros_like(custom, dtype=np.float32)
    native_valid = np.zeros_like(custom_valid)
    metric = overlap.OverlapMetric()
    for frame in range(rollout.num_steps):
        result = metric.compute_overlap(_one_frame(native_trajectory, frame))
        native_values[frame] = np.asarray(result.value)
        native_valid[frame] = np.asarray(result.valid)
    np.testing.assert_array_equal(custom_valid, native_valid)
    np.testing.assert_array_equal(
        custom[custom_valid],
        native_values[custom_valid].astype(bool),
    )


def test_rotated_near_contact_overlap_matches_native_boundary_bits() -> None:
    def box_from_bits(values: list[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.uint32).view(np.float32)

    first = box_from_bits(
        [0, 0, 1074609456, 1077566592, 3223756496]
    )
    second = box_from_bits(
        [3217000687, 3211045169, 1067689431, 1081125145, 3223756496]
    )
    timestamps = np.array([0.0, 0.1])
    agents = [
        _agent(
            1,
            valid=np.ones(2, dtype=bool),
            x=np.full(2, first[0]),
            y=np.full(2, first[1]),
            heading=np.full(2, first[4]),
            vx=np.zeros(2),
            vy=np.zeros(2),
            length=float(first[2]),
            width=float(first[3]),
        ),
        _agent(
            2,
            valid=np.ones(2, dtype=bool),
            x=np.full(2, second[0]),
            y=np.full(2, second[1]),
            heading=np.full(2, second[4]),
            vx=np.zeros(2),
            vy=np.zeros(2),
            length=float(second[2]),
            width=float(second[3]),
        ),
    ]
    scenario = Scenario(
        scenario_id="m5-waymax-overlap-boundary",
        timestamps=timestamps,
        agents=agents,
        ego_index=0,
        metadata={"source": "synthetic", "current_index": 0},
    )
    rollout = Rollout(
        scenario_id=scenario.scenario_id,
        sim_name="candidate",
        sim_version="1.0.0",
        seed=0,
        timestamps=timestamps,
        agents=copy.deepcopy(agents),
    )
    custom, valid = oriented_box_overlap_components(scenario, rollout)
    native = bool(np.asarray(geometry.has_overlap(first, second)))
    assert valid[1, 0] and valid[1, 1]
    assert custom[1, 0] == native
    assert custom[1, 1] == native


def test_rotated_zero_margin_backend_counterexample_remains_explicit() -> None:
    """NumPy/libm and XLA may flip a strict zero-margin overlap decision.

    This sentinel prevents a bounded observed parity check from being promoted into
    a universal bit-equivalence claim. The official parity runner must compare every
    observed target flag to the pinned native metric and fail on any mismatch.
    """

    def box_from_bits(values: list[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.uint32).view(np.float32)

    first = box_from_bits(
        [0, 0, 1083048713, 1076431310, 3223317161]
    )
    second = box_from_bits(
        [3223953347, 3221101940, 1074518026, 1062145481, 3223317161]
    )
    timestamps = np.array([0.0, 0.1])
    agents = [
        _agent(
            1,
            valid=np.ones(2, dtype=bool),
            x=np.full(2, first[0]),
            y=np.full(2, first[1]),
            heading=np.full(2, first[4]),
            vx=np.zeros(2),
            vy=np.zeros(2),
            length=float(first[2]),
            width=float(first[3]),
        ),
        _agent(
            2,
            valid=np.ones(2, dtype=bool),
            x=np.full(2, second[0]),
            y=np.full(2, second[1]),
            heading=np.full(2, second[4]),
            vx=np.zeros(2),
            vy=np.zeros(2),
            length=float(second[2]),
            width=float(second[3]),
        ),
    ]
    scenario = Scenario(
        scenario_id="m5-waymax-overlap-counterexample",
        timestamps=timestamps,
        agents=agents,
        ego_index=0,
        metadata={"source": "synthetic", "current_index": 0},
    )
    rollout = Rollout(
        scenario_id=scenario.scenario_id,
        sim_name="candidate",
        sim_version="1.0.0",
        seed=0,
        timestamps=timestamps,
        agents=copy.deepcopy(agents),
    )
    custom, valid = oriented_box_overlap_components(scenario, rollout)
    native = bool(np.asarray(geometry.has_overlap(first, second)))
    assert valid[1, 0] and valid[1, 1]
    assert bool(custom[1, 0]) != native


def test_kinematic_flags_and_masks_match_pinned_waymax_oracle() -> None:
    scenario, rollout = _contract_pair()
    custom, custom_valid, _, _ = kinematic_infeasibility_components(
        scenario,
        rollout,
    )
    native_trajectory = _native_trajectory(rollout.agents, rollout.timestamps)
    metric = comfort.KinematicsInfeasibilityMetric()
    native_values = np.zeros_like(custom, dtype=np.float32)
    native_valid = np.zeros_like(custom_valid)
    for transition in range(rollout.num_steps - 1):
        result = metric.compute_kinematics_infeasibility(
            native_trajectory,
            jnp.asarray(transition + 1, dtype=jnp.int32),
        )
        native_values[transition] = np.asarray(result.value)
        native_valid[transition] = np.asarray(result.valid)
    np.testing.assert_array_equal(custom_valid, native_valid)
    np.testing.assert_array_equal(
        custom,
        native_values.astype(bool),
    )


def test_kinematic_dt_squared_association_matches_native_boundary_bits() -> None:
    timestamps = np.array([0.0, 0.1])

    def from_bits(value: int) -> np.float32:
        return np.asarray(value, dtype=np.uint32).view(np.float32).item()

    old_vx = from_bits(1073420204)
    old_heading = from_bits(3221622894)
    new_vx = from_bits(3214169873)
    new_vy = from_bits(3222779344)
    ego = _agent(
        1,
        valid=np.ones(2, dtype=bool),
        x=np.zeros(2),
        y=np.zeros(2),
        heading=np.zeros(2),
        vx=np.zeros(2),
        vy=np.zeros(2),
        length=4.0,
        width=2.0,
    )
    world = _agent(
        2,
        valid=np.ones(2, dtype=bool),
        x=np.zeros(2),
        y=np.zeros(2),
        heading=np.array([old_heading, 0.0]),
        vx=np.array([old_vx, new_vx]),
        vy=np.array([0.0, new_vy]),
        length=4.0,
        width=2.0,
    )
    scenario = Scenario(
        scenario_id="m5-waymax-kinematic-boundary",
        timestamps=timestamps,
        agents=[ego, world],
        ego_index=0,
        metadata={"source": "synthetic", "current_index": 0},
    )
    rollout = Rollout(
        scenario_id=scenario.scenario_id,
        sim_name="candidate",
        sim_version="1.0.0",
        seed=0,
        timestamps=timestamps,
        agents=[copy.deepcopy(ego), copy.deepcopy(world)],
    )
    custom_flags, custom_valid, _, custom_curvature = (
        kinematic_infeasibility_components(scenario, rollout)
    )
    trajectory = _native_trajectory(rollout.agents, timestamps)
    native_actions = bicycle_model.compute_inverse(
        trajectory,
        jnp.asarray(0, dtype=jnp.int32),
        dt=0.1,
    )
    native_result = comfort.KinematicsInfeasibilityMetric().compute_kinematics_infeasibility(
        trajectory,
        jnp.asarray(1, dtype=jnp.int32),
    )
    native_curvature = np.asarray(native_actions.data)[1, 1]
    assert custom_valid[0, 1] == bool(np.asarray(native_result.valid)[1])
    assert custom_curvature[0, 1].view(np.uint32) == native_curvature.view(
        np.uint32
    )
    assert custom_flags[0, 1] == bool(np.asarray(native_result.value)[1])
