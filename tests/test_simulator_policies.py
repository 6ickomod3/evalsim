"""Analytic and metamorphic tests for the three M2 simulator policies."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from evalsim import Agent, AgentType, Scenario
from evalsim.rollout import RolloutEngine
from evalsim.simulators import (
    ConstantVelocityPolicy,
    IDMParameters,
    IDMPolicy,
    LogReplayPolicy,
)


def _agent(
    agent_id: int,
    timestamps: np.ndarray,
    *,
    x: np.ndarray | float,
    y: np.ndarray | float = 0.0,
    vx: np.ndarray | float = 0.0,
    vy: np.ndarray | float = 0.0,
    heading: np.ndarray | float = 0.0,
    valid: np.ndarray | None = None,
    agent_type: AgentType = AgentType.VEHICLE,
    length: float = 4.5,
    width: float = 2.0,
) -> Agent:
    count = len(timestamps)

    def series(value: np.ndarray | float) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            return np.full(count, float(array))
        return np.array(array, dtype=float)

    return Agent(
        id=agent_id,
        type=agent_type,
        valid=(
            np.ones(count, dtype=bool)
            if valid is None
            else np.array(valid, dtype=bool)
        ),
        x=series(x),
        y=series(y),
        heading=series(heading),
        vx=series(vx),
        vy=series(vy),
        length=length,
        width=width,
    )


def _scenario(
    timestamps: np.ndarray,
    agents: list[Agent],
    *,
    ego_index: int = 0,
    current_index: int = 0,
) -> Scenario:
    return Scenario(
        scenario_id="unit-policy-scene",
        timestamps=np.array(timestamps, dtype=float),
        agents=agents,
        ego_index=ego_index,
        metadata={"source": "unit", "current_index": current_index},
    )


def _world_fields(rollout, index: int = 1) -> tuple[np.ndarray, ...]:
    agent = rollout.agents[index]
    return tuple(
        np.array(getattr(agent, field), copy=True)
        for field in ("x", "y", "heading", "vx", "vy")
    )


def test_constant_velocity_matches_irregular_timestamp_oracle() -> None:
    timestamps = np.array([0.0, 0.05, 0.2, 0.45, 1.0])
    ego = _agent(
        100,
        timestamps,
        x=timestamps**2,
        vx=2.0 * timestamps,
    )
    initial_x = 2.0
    initial_y = -1.0
    vx = 3.0
    vy = -4.0
    world = _agent(
        200,
        timestamps,
        x=np.array([initial_x, 90.0, 80.0, 70.0, 60.0]),
        y=np.array([initial_y, 50.0, 40.0, 30.0, 20.0]),
        vx=np.array([vx, 9.0, 8.0, 7.0, 6.0]),
        vy=np.array([vy, 9.0, 8.0, 7.0, 6.0]),
        heading=np.arctan2(vy, vx),
    )
    scenario = _scenario(timestamps, [ego, world])

    rollout = RolloutEngine().run(
        scenario,
        ConstantVelocityPolicy(),
        seed=7,
    )
    simulated = rollout.agents[1]
    np.testing.assert_allclose(
        simulated.x,
        initial_x + vx * timestamps,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        simulated.y,
        initial_y + vy * timestamps,
        atol=1e-12,
    )
    np.testing.assert_allclose(simulated.vx, vx, atol=1e-12)
    np.testing.assert_allclose(simulated.vy, vy, atol=1e-12)
    np.testing.assert_allclose(
        simulated.heading,
        np.arctan2(vy, vx),
        atol=1e-12,
    )
    # Ego remains the nonlinear logged trajectory.
    np.testing.assert_array_equal(rollout.agents[0].x, ego.x)


def test_constant_velocity_stationary_agent_stays_frozen() -> None:
    timestamps = np.array([0.0, 0.2, 0.7, 1.5])
    ego = _agent(0, timestamps, x=timestamps)
    stationary = _agent(
        1,
        timestamps,
        x=8.0,
        y=-3.0,
        heading=1.1,
    )
    rollout = RolloutEngine().run(
        _scenario(timestamps, [ego, stationary]),
        ConstantVelocityPolicy(),
    )
    np.testing.assert_array_equal(rollout.agents[1].x, stationary.x)
    np.testing.assert_array_equal(rollout.agents[1].y, stationary.y)
    np.testing.assert_array_equal(
        rollout.agents[1].heading,
        stationary.heading,
    )
    np.testing.assert_array_equal(rollout.agents[1].speed(), np.zeros(4))


def test_lifecycle_birth_death_and_reentry_are_engine_owned() -> None:
    timestamps = np.arange(6.0)
    valid = np.array([False, False, True, True, False, True])
    ego = _agent(0, timestamps, x=timestamps)
    world = _agent(
        1,
        timestamps,
        valid=valid,
        x=np.array([-900.0, -800.0, 10.0, 999.0, -700.0, 100.0]),
        vx=np.array([999.0, 999.0, 2.0, 777.0, 999.0, -3.0]),
    )
    rollout = RolloutEngine().run(
        _scenario(timestamps, [ego, world]),
        ConstantVelocityPolicy(),
    )
    simulated = rollout.agents[1]

    np.testing.assert_array_equal(simulated.valid, valid)
    assert simulated.x[2] == 10.0  # valid-segment birth from the log
    assert simulated.x[3] == pytest.approx(12.0)  # then simulated
    assert simulated.x[4] == pytest.approx(12.0)  # held while absent
    assert simulated.x[5] == 100.0  # re-entry starts a new segment
    assert simulated.vx[5] == -3.0


@pytest.mark.parametrize(
    "policy",
    [ConstantVelocityPolicy(), IDMPolicy()],
    ids=["constant_velocity", "idm"],
)
def test_future_world_log_poisoning_does_not_leak_into_causal_policies(
    policy,
) -> None:
    timestamps = np.arange(5.0) * 0.1
    ego = _agent(0, timestamps, x=-100.0, y=10.0)
    clean_world = _agent(1, timestamps, x=2.0 + timestamps, vx=1.0)
    poisoned_world = copy.deepcopy(clean_world)
    poisoned_world.x[1:] = np.array([1e3, -2e3, 3e3, -4e3])
    poisoned_world.y[1:] = np.array([-5e3, 6e3, -7e3, 8e3])
    poisoned_world.vx[1:] = np.array([20.0, -30.0, 40.0, -50.0])
    poisoned_world.vy[1:] = np.array([60.0, -70.0, 80.0, -90.0])

    engine = RolloutEngine()
    clean = engine.run(_scenario(timestamps, [ego, clean_world]), policy)
    poisoned = engine.run(
        _scenario(timestamps, [copy.deepcopy(ego), poisoned_world]),
        policy,
    )
    for clean_field, poisoned_field in zip(
        _world_fields(clean),
        _world_fields(poisoned),
    ):
        np.testing.assert_array_equal(clean_field, poisoned_field)

    replay = engine.run(
        _scenario(
            timestamps,
            [copy.deepcopy(ego), copy.deepcopy(poisoned_world)],
        ),
        LogReplayPolicy(),
    )
    np.testing.assert_array_equal(replay.agents[1].x, poisoned_world.x)


def _following_fixture(
    *,
    center_gap: float,
    leader_length: float = 4.5,
    leader_y: float = 0.0,
    leader_heading: float = 0.0,
) -> Scenario:
    timestamps = np.array([0.0, 0.1])
    ego = _agent(
        10,
        timestamps,
        x=np.array([center_gap, center_gap + 0.5]),
        y=leader_y,
        vx=5.0 * np.cos(leader_heading),
        vy=5.0 * np.sin(leader_heading),
        heading=leader_heading,
        length=leader_length,
    )
    follower = _agent(
        20,
        timestamps,
        x=np.array([0.0, 999.0]),
        vx=np.array([10.0, 999.0]),
    )
    return _scenario(timestamps, [ego, follower])


def _realized_first_acceleration(scenario: Scenario) -> float:
    rollout = RolloutEngine().run(scenario, IDMPolicy())
    dt = scenario.timestamps[1] - scenario.timestamps[0]
    return float(
        (rollout.agents[1].speed()[1] - rollout.agents[1].speed()[0]) / dt
    )


def test_idm_first_step_matches_independent_formula() -> None:
    scenario = _following_fixture(center_gap=20.0)
    params = IDMParameters()
    follower_speed = 10.0
    leader_speed = 5.0
    bumper_gap = 20.0 - 0.5 * (4.5 + 4.5)
    desired_gap = params.minimum_gap_m + max(
        0.0,
        follower_speed * params.safe_headway_s
        + follower_speed
        * (follower_speed - leader_speed)
        / (
            2.0
            * np.sqrt(
                params.max_acceleration_mps2
                * params.comfortable_deceleration_mps2
            )
        ),
    )
    expected = params.max_acceleration_mps2 * (
        1.0
        - (follower_speed / params.desired_speed_mps)
        ** params.acceleration_exponent
        - (desired_gap / bumper_gap) ** 2
    )
    assert _realized_first_acceleration(scenario) == pytest.approx(
        expected,
        rel=1e-6,
        abs=1e-9,
    )


def test_idm_monotonic_gap_and_vehicle_length_effects() -> None:
    close = _realized_first_acceleration(
        _following_fixture(center_gap=20.0)
    )
    far = _realized_first_acceleration(
        _following_fixture(center_gap=35.0)
    )
    long_leader = _realized_first_acceleration(
        _following_fixture(center_gap=35.0, leader_length=10.0)
    )

    assert close < far
    assert long_leader < far


@pytest.mark.parametrize(
    ("leader_y", "leader_heading"),
    [(4.0, 0.0), (0.0, np.pi)],
    ids=["adjacent", "opposing"],
)
def test_idm_ignores_adjacent_and_opposing_candidates(
    leader_y: float,
    leader_heading: float,
) -> None:
    acceleration = _realized_first_acceleration(
        _following_fixture(
            center_gap=12.0,
            leader_y=leader_y,
            leader_heading=leader_heading,
        )
    )
    params = IDMParameters()
    expected_free_road = params.max_acceleration_mps2 * (
        1.0
        - (10.0 / params.desired_speed_mps)
        ** params.acceleration_exponent
    )
    assert acceleration == pytest.approx(expected_free_road, rel=1e-6)


def test_idm_nonvehicle_fallback_is_constant_velocity() -> None:
    timestamps = np.array([0.0, 0.2, 0.6, 1.0])
    ego = _agent(0, timestamps, x=-50.0)
    pedestrian = _agent(
        1,
        timestamps,
        x=2.0,
        y=-1.0,
        vy=1.5,
        heading=np.pi / 2.0,
        agent_type=AgentType.PEDESTRIAN,
        length=0.6,
        width=0.6,
    )
    rollout = RolloutEngine().run(
        _scenario(timestamps, [ego, pedestrian]),
        IDMPolicy(),
    )
    np.testing.assert_allclose(
        rollout.agents[1].y,
        -1.0 + 1.5 * timestamps,
        atol=1e-12,
    )
    assert rollout.metadata["policy"]["fallback_policy"] == "constant_velocity"


def test_idm_vehicle_treats_aligned_nonvehicle_as_physical_leader() -> None:
    timestamps = np.array([0.0, 0.1])
    ego = _agent(0, timestamps, x=-100.0, y=20.0)
    follower = _agent(
        1,
        timestamps,
        x=np.array([0.0, 999.0]),
        vx=np.array([10.0, 999.0]),
    )
    pedestrian = _agent(
        2,
        timestamps,
        x=12.0,
        heading=0.0,
        agent_type=AgentType.PEDESTRIAN,
        length=0.6,
        width=0.6,
    )
    rollout = RolloutEngine().run(
        _scenario(timestamps, [ego, follower, pedestrian]),
        IDMPolicy(),
    )
    realized_acceleration = (
        rollout.agents[1].speed()[1] - rollout.agents[1].speed()[0]
    ) / 0.1
    assert realized_acceleration < 0.0


def test_idm_extreme_finite_speed_reaches_engine_clamp_without_overflow() -> None:
    timestamps = np.array([0.0, 0.1])
    ego = _agent(0, timestamps, x=-100.0, y=20.0)
    world = _agent(
        1,
        timestamps,
        x=0.0,
        vx=1e100,
    )
    rollout = RolloutEngine().run(
        _scenario(timestamps, [ego, world]),
        IDMPolicy(),
    )
    assert np.all(np.isfinite(rollout.agents[1].x))
    assert np.all(np.isfinite(rollout.agents[1].vx))
    assert rollout.agents[1].speed()[1] <= 60.0
    assert rollout.metadata["dynamics"]["clamp_counts"]["speed"] >= 1
    assert rollout.metadata["dynamics"]["clamp_counts"]["deceleration"] >= 1


def test_idm_is_agent_order_invariant_when_keyed_by_id() -> None:
    timestamps = np.array([0.0, 0.1, 0.2, 0.3])
    ego = _agent(100, timestamps, x=20.0 + 5.0 * timestamps, vx=5.0)
    follower = _agent(200, timestamps, x=10.0 * timestamps, vx=10.0)
    far_leader = _agent(300, timestamps, x=50.0 + 4.0 * timestamps, vx=4.0)
    original = _scenario(timestamps, [ego, follower, far_leader], ego_index=0)
    permuted = _scenario(
        timestamps,
        [copy.deepcopy(far_leader), copy.deepcopy(ego), copy.deepcopy(follower)],
        ego_index=1,
    )

    engine = RolloutEngine()
    first = engine.run(original, IDMPolicy())
    second = engine.run(permuted, IDMPolicy())
    first_by_id = {agent.id: agent for agent in first.agents}
    second_by_id = {agent.id: agent for agent in second.agents}
    for agent_id in first_by_id:
        for field in ("valid", "x", "y", "heading", "vx", "vy"):
            np.testing.assert_array_equal(
                getattr(first_by_id[agent_id], field),
                getattr(second_by_id[agent_id], field),
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"desired_speed_mps": 0.0},
        {"minimum_gap_m": -1.0},
        {"safe_headway_s": -0.1},
        {"max_acceleration_mps2": np.nan},
        {"comfortable_deceleration_mps2": 0.0},
        {"acceleration_exponent": -1.0},
        {"max_heading_difference_rad": np.pi / 2.0},
    ],
)
def test_invalid_idm_parameters_reject(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        IDMParameters(**kwargs)


def test_policy_metadata_is_truthful_and_seed_independent() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    scenario = _following_fixture(center_gap=30.0)
    engine = RolloutEngine()
    policies = [LogReplayPolicy(), ConstantVelocityPolicy(), IDMPolicy()]
    names = {policy.metadata().name for policy in policies}
    assert names == {"log_replay", "constant_velocity", "idm"}

    for policy in policies:
        metadata = policy.metadata()
        assert metadata.deterministic is True
        assert metadata.version
        first = engine.run(scenario, policy, seed=0)
        second = engine.run(scenario, policy, seed=2**32 - 1)
        for left, right in zip(first.agents, second.agents):
            for field in ("valid", "x", "y", "heading", "vx", "vy"):
                np.testing.assert_array_equal(
                    getattr(left, field),
                    getattr(right, field),
                )
        assert first.seed == 0
        assert second.seed == 2**32 - 1
