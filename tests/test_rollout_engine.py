"""M2 closed-loop engine integration, provenance, and adversarial boundaries."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np
import pytest

from evalsim import (
    Agent,
    AgentFrame,
    AgentType,
    PolicyMetadata,
    PolicyObservation,
    PolicyStep,
    Scenario,
    SimulatorPolicy,
    rollout_from_parquet,
    rollout_to_parquet,
)
from evalsim.rollout import DynamicsLimits, RolloutEngine
from evalsim.simulators import (
    ConstantVelocityPolicy,
    IDMPolicy,
    LogReplayPolicy,
)
from evalsim.sources import SyntheticSource

_SERIES_FIELDS = ("valid", "x", "y", "heading", "vx", "vy")


def _simple_agent(
    agent_id: int,
    timestamps: np.ndarray,
    *,
    x: np.ndarray | float = 0.0,
    y: np.ndarray | float = 0.0,
    vx: np.ndarray | float = 0.0,
    vy: np.ndarray | float = 0.0,
    heading: np.ndarray | float = 0.0,
    valid: np.ndarray | None = None,
    agent_type: AgentType = AgentType.VEHICLE,
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
    )


def _simple_scenario(
    timestamps: np.ndarray,
    agents: list[Agent],
    *,
    ego_index: int = 0,
    metadata: dict | None = None,
) -> Scenario:
    return Scenario(
        scenario_id="engine-unit-scene",
        timestamps=np.array(timestamps, dtype=float),
        agents=agents,
        ego_index=ego_index,
        metadata={"source": "unit", **(metadata or {})},
    )


def _assert_agent_equal(left: Agent, right: Agent) -> None:
    assert left.id == right.id
    assert left.type == right.type
    assert left.length == right.length
    assert left.width == right.width
    for field in _SERIES_FIELDS:
        np.testing.assert_array_equal(
            getattr(left, field),
            getattr(right, field),
        )


def _assert_rollout_equal(left, right) -> None:
    assert left.scenario_id == right.scenario_id
    assert left.sim_name == right.sim_name
    assert left.sim_version == right.sim_version
    assert left.seed == right.seed
    assert left.perturbation == right.perturbation
    assert left.metadata == right.metadata
    np.testing.assert_array_equal(left.timestamps, right.timestamps)
    assert len(left.agents) == len(right.agents)
    for left_agent, right_agent in zip(left.agents, right.agents):
        _assert_agent_equal(left_agent, right_agent)


def _assert_scenario_equal(left: Scenario, right: Scenario) -> None:
    assert left.scenario_id == right.scenario_id
    assert left.ego_index == right.ego_index
    assert left.metadata == right.metadata
    np.testing.assert_array_equal(left.timestamps, right.timestamps)
    for left_agent, right_agent in zip(left.agents, right.agents):
        _assert_agent_equal(left_agent, right_agent)
    for left_feature, right_feature in zip(left.map, right.map):
        assert left_feature.type == right_feature.type
        np.testing.assert_array_equal(left_feature.xy, right_feature.xy)


def test_all_50_scenarios_x_three_policies_end_to_end_and_parquet(
    tmp_path,
) -> None:
    scenarios = SyntheticSource(seed=2026).generate(50)
    policies = [
        LogReplayPolicy(),
        ConstantVelocityPolicy(),
        IDMPolicy(),
    ]
    engine = RolloutEngine()

    rollout_count = 0
    for policy in policies:
        metadata = policy.metadata()
        for scenario_index, scenario in enumerate(scenarios):
            before = copy.deepcopy(scenario)
            rollout = engine.run(scenario, policy, seed=17)
            rollout_count += 1

            _assert_scenario_equal(scenario, before)
            assert rollout.scenario_id == scenario.scenario_id
            assert rollout.sim_name == metadata.name
            assert rollout.sim_version == metadata.version
            assert rollout.seed == 17
            assert rollout.perturbation is None
            np.testing.assert_array_equal(
                rollout.timestamps,
                scenario.timestamps,
            )
            assert rollout.timestamps is not scenario.timestamps
            assert rollout.num_agents == scenario.num_agents
            for output_agent, reference_agent in zip(
                rollout.agents,
                scenario.agents,
            ):
                assert output_agent.id == reference_agent.id
                assert output_agent.type == reference_agent.type
                assert output_agent.length == reference_agent.length
                assert output_agent.width == reference_agent.width
                np.testing.assert_array_equal(
                    output_agent.valid,
                    reference_agent.valid,
                )
                for field in ("x", "y", "heading", "vx", "vy"):
                    values = getattr(output_agent, field)
                    assert np.all(np.isfinite(values[output_agent.valid]))
                assert np.all(output_agent.heading >= -np.pi)
                assert np.all(output_agent.heading <= np.pi)

            assert rollout.metadata["ego_control"] == "logged"
            assert rollout.metadata["rollout_start_index"] == 0
            assert rollout.metadata["policy"] == metadata.to_dict()
            assert rollout.metadata["engine"]["name"] == engine.name
            assert (
                rollout.metadata["dynamics"]["limits"]
                == engine.dynamics_limits.to_dict()
            )
            json.dumps(rollout.metadata, allow_nan=False, sort_keys=True)

            path = (
                tmp_path
                / metadata.name
                / f"{scenario_index:03d}.parquet"
            )
            rollout_to_parquet(rollout, path)
            _assert_rollout_equal(rollout, rollout_from_parquet(path))

    assert rollout_count == 150


def test_log_replay_is_exact_for_every_field_and_deep_copied() -> None:
    engine = RolloutEngine()
    for scenario in SyntheticSource(seed=91).generate(50):
        rollout = engine.run(scenario, LogReplayPolicy(), seed=999)
        np.testing.assert_array_equal(rollout.timestamps, scenario.timestamps)
        for output_agent, reference_agent in zip(
            rollout.agents,
            scenario.agents,
        ):
            _assert_agent_equal(output_agent, reference_agent)
            for field in _SERIES_FIELDS:
                assert getattr(output_agent, field) is not getattr(
                    reference_agent,
                    field,
                )

        original_x = float(scenario.agents[0].x[0])
        rollout.agents[0].x[0] += 123.0
        assert scenario.agents[0].x[0] == original_x


@pytest.mark.parametrize(
    "policy",
    [LogReplayPolicy(), ConstantVelocityPolicy(), IDMPolicy()],
    ids=["replay", "constant_velocity", "idm"],
)
def test_policy_reuse_interleaving_has_no_cross_run_state(policy) -> None:
    first_scenario = SyntheticSource(seed=3).generate_one(0)
    second_scenario = SyntheticSource(seed=3).generate_one(2)
    engine = RolloutEngine()

    first = engine.run(first_scenario, policy, seed=42)
    engine.run(second_scenario, policy, seed=9)
    repeated = engine.run(first_scenario, policy, seed=42)
    _assert_rollout_equal(first, repeated)


def test_three_policies_have_distinct_failure_modes() -> None:
    source = SyntheticSource(seed=42)
    engine = RolloutEngine()

    merge = source.generate_one(2, "merge")
    replay_merge = engine.run(merge, LogReplayPolicy())
    cv_merge = engine.run(merge, ConstantVelocityPolicy())
    assert not np.allclose(
        replay_merge.agents[2].y[merge.agents[2].valid],
        cv_merge.agents[2].y[merge.agents[2].valid],
    )

    following = source.generate_one(0, "following")
    cv_following = engine.run(following, ConstantVelocityPolicy())
    idm_following = engine.run(following, IDMPolicy())
    follower_valid = following.agents[2].valid
    assert not np.allclose(
        cv_following.agents[2].x[follower_valid],
        idm_following.agents[2].x[follower_valid],
    )


def test_current_index_copies_history_then_starts_simulation() -> None:
    timestamps = np.array([0.0, 0.1, 0.4, 1.0, 1.5])
    ego = _simple_agent(0, timestamps, x=timestamps**2)
    world = _simple_agent(
        1,
        timestamps,
        x=np.array([0.0, 1.0, 5.0, 999.0, -999.0]),
        vx=np.array([1.0, 2.0, 3.0, 500.0, -500.0]),
    )
    scenario = _simple_scenario(
        timestamps,
        [ego, world],
        metadata={"current_index": 2},
    )
    rollout = RolloutEngine().run(scenario, ConstantVelocityPolicy())

    np.testing.assert_array_equal(rollout.agents[1].x[:3], world.x[:3])
    assert rollout.agents[1].x[3] == pytest.approx(5.0 + 3.0 * 0.6)
    assert rollout.agents[1].x[4] == pytest.approx(5.0 + 3.0 * 1.1)
    assert rollout.metadata["rollout_start_index"] == 2


@dataclass
class _SpyPolicy(SimulatorPolicy):
    observations: list[PolicyObservation]
    initialize_calls: int = 0

    def initialize(self, scenario: Scenario, seed: int):
        self.initialize_calls += 1
        return {"count": scenario.num_agents}

    def step(self, state, observation: PolicyObservation) -> PolicyStep:
        self.observations.append(observation)
        count = observation.frame.num_agents
        return PolicyStep(
            next_state=state,
            longitudinal_acceleration=np.zeros(count),
            yaw_rate=np.zeros(count),
        )

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata("spy", "0.1.0", True)


def test_policy_observes_prior_simulated_frame_not_logged_future() -> None:
    timestamps = np.array([0.0, 0.2, 0.5, 1.0])
    ego = _simple_agent(0, timestamps, x=-100.0)
    world = _simple_agent(
        1,
        timestamps,
        x=np.array([1.0, 1000.0, -2000.0, 3000.0]),
        vx=np.array([2.0, 90.0, -80.0, 70.0]),
    )
    policy = _SpyPolicy(observations=[])
    rollout = RolloutEngine().run(
        _simple_scenario(timestamps, [ego, world]),
        policy,
    )

    assert policy.initialize_calls == 1
    assert len(policy.observations) == len(timestamps) - 1
    for observation in policy.observations:
        np.testing.assert_array_equal(
            observation.frame.x,
            np.array(
                [
                    rollout.agents[0].x[observation.current_index],
                    rollout.agents[1].x[observation.current_index],
                ]
            ),
        )
    np.testing.assert_allclose(
        rollout.agents[1].x,
        1.0 + 2.0 * timestamps,
        atol=1e-12,
    )


def test_one_frame_scenario_initializes_once_without_stepping() -> None:
    timestamps = np.array([3.0])
    ego = _simple_agent(0, timestamps, x=5.0, heading=0.2)
    policy = _SpyPolicy(observations=[])
    rollout = RolloutEngine().run(
        _simple_scenario(timestamps, [ego]),
        policy,
    )
    assert policy.initialize_calls == 1
    assert policy.observations == []
    np.testing.assert_array_equal(rollout.agents[0].x, [5.0])


class _WrongOutputPolicy(SimulatorPolicy):
    def initialize(self, scenario: Scenario, seed: int):
        return None

    def step(self, state, observation):
        return {"not": "a PolicyStep"}

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata("wrong", "0.1.0", True)


class _NaNPolicy(SimulatorPolicy):
    def initialize(self, scenario: Scenario, seed: int):
        return None

    def step(self, state, observation):
        count = observation.frame.num_agents
        return PolicyStep(
            next_state=None,
            longitudinal_acceleration=np.full(count, np.nan),
            yaw_rate=np.zeros(count),
        )

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata("nan", "0.1.0", True)


@pytest.mark.parametrize(
    ("policy", "error_type", "message"),
    [
        (_WrongOutputPolicy(), TypeError, "expected PolicyStep"),
        (_NaNPolicy(), RuntimeError, "transition 0->1"),
    ],
)
def test_malformed_policy_outputs_fail_with_transition_context(
    policy,
    error_type,
    message,
) -> None:
    timestamps = np.array([0.0, 0.1])
    scenario = _simple_scenario(
        timestamps,
        [
            _simple_agent(0, timestamps),
            _simple_agent(1, timestamps, vx=1.0),
        ],
    )
    with pytest.raises(error_type, match=message):
        RolloutEngine().run(scenario, policy)


class _ExtremePolicy(SimulatorPolicy):
    def initialize(self, scenario: Scenario, seed: int):
        return None

    def step(self, state, observation) -> PolicyStep:
        count = observation.frame.num_agents
        return PolicyStep(
            next_state=None,
            longitudinal_acceleration=np.full(count, 1e6),
            yaw_rate=np.full(count, -1e6),
        )

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            "extreme",
            "0.1.0",
            True,
            supported_agent_types=tuple(AgentType),
        )


def test_clamps_are_recorded_without_counting_logged_ego() -> None:
    timestamps = np.array([0.0, 1.0])
    scenario = _simple_scenario(
        timestamps,
        [
            _simple_agent(0, timestamps, vx=100.0),
            _simple_agent(1, timestamps),
        ],
    )
    rollout = RolloutEngine(
        dynamics_limits=DynamicsLimits(
            max_acceleration_mps2=2.0,
            max_deceleration_mps2=3.0,
            max_speed_mps=1.0,
            max_yaw_rate_radps=0.2,
        )
    ).run(scenario, _ExtremePolicy())
    counts = rollout.metadata["dynamics"]["clamp_counts"]
    assert counts["acceleration"] == 1
    assert counts["speed"] == 1
    assert counts["yaw_rate"] == 1
    np.testing.assert_array_equal(rollout.agents[0].vx, scenario.ego.vx)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_invalid_seed_rejects(seed) -> None:
    scenario = SyntheticSource(seed=1).generate_one(0)
    with pytest.raises(ValueError, match="seed"):
        RolloutEngine().run(scenario, ConstantVelocityPolicy(), seed=seed)


def test_non_none_perturbation_rejects_until_m5() -> None:
    scenario = SyntheticSource(seed=1).generate_one(0)
    with pytest.raises(ValueError, match="M5"):
        RolloutEngine().run(
            scenario,
            ConstantVelocityPolicy(),
            perturbation="ego_brake",
        )


def test_invalid_scenario_boundaries_reject_before_rollout() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    duplicate_ids = _simple_scenario(
        timestamps,
        [
            _simple_agent(1, timestamps),
            _simple_agent(1, timestamps),
        ],
    )
    with pytest.raises(ValueError, match="unique"):
        RolloutEngine().run(duplicate_ids, ConstantVelocityPolicy())

    nonmonotonic = _simple_scenario(
        np.array([0.0, 0.2, 0.1]),
        [_simple_agent(0, timestamps)],
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        RolloutEngine().run(nonmonotonic, ConstantVelocityPolicy())

    invalid_ego = _simple_scenario(
        timestamps,
        [
            _simple_agent(
                0,
                timestamps,
                valid=np.array([True, False, True]),
            )
        ],
    )
    with pytest.raises(ValueError, match="ego"):
        RolloutEngine().run(invalid_ego, ConstantVelocityPolicy())

    never_valid_world = _simple_scenario(
        timestamps,
        [
            _simple_agent(0, timestamps),
            _simple_agent(
                1,
                timestamps,
                valid=np.zeros(3, dtype=bool),
            ),
        ],
    )
    with pytest.raises(ValueError, match="never valid"):
        RolloutEngine().run(never_valid_world, ConstantVelocityPolicy())

    mutated_ego_index = _simple_scenario(
        timestamps,
        [
            _simple_agent(0, timestamps),
            _simple_agent(1, timestamps),
        ],
    )
    mutated_ego_index.ego_index = True
    with pytest.raises(ValueError, match="ego_index"):
        RolloutEngine().run(
            mutated_ego_index,
            ConstantVelocityPolicy(),
        )


class _VehicleOnlyNoFallback(SimulatorPolicy):
    def initialize(self, scenario: Scenario, seed: int):
        return None

    def step(self, state, observation):
        count = observation.frame.num_agents
        return PolicyStep(None, np.zeros(count), np.zeros(count))

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            "vehicle_only",
            "0.1.0",
            True,
            supported_agent_types=(AgentType.VEHICLE,),
        )


def test_unsupported_agent_requires_explicit_fallback() -> None:
    timestamps = np.array([0.0, 0.1])
    scenario = _simple_scenario(
        timestamps,
        [
            _simple_agent(0, timestamps),
            _simple_agent(
                1,
                timestamps,
                agent_type=AgentType.PEDESTRIAN,
            ),
        ],
    )
    with pytest.raises(ValueError, match="fallback"):
        RolloutEngine().run(scenario, _VehicleOnlyNoFallback())


def test_policy_metadata_is_immutable_and_detached() -> None:
    original = {
        "desired_speed_mps": 13.0,
        "nested": {"thresholds": [1.0, 2.0]},
    }
    metadata = PolicyMetadata(
        "test",
        "1.0",
        True,
        params=original,
        supported_agent_types=[AgentType.VEHICLE],
        known_limitations=["unit fixture"],
    )
    original["desired_speed_mps"] = 99.0
    original["nested"]["thresholds"].append(3.0)
    assert metadata.params["desired_speed_mps"] == 13.0
    assert metadata.params["nested"]["thresholds"] == (1.0, 2.0)
    with pytest.raises(TypeError):
        metadata.params["desired_speed_mps"] = 10.0
    with pytest.raises(TypeError):
        metadata.params["nested"]["new"] = "value"
    with pytest.raises(AttributeError):
        metadata.params["nested"]["thresholds"].append(3.0)
    payload = metadata.to_dict()
    payload["params"]["desired_speed_mps"] = -1.0
    payload["params"]["nested"]["thresholds"].append(3.0)
    assert metadata.params["desired_speed_mps"] == 13.0
    assert metadata.params["nested"]["thresholds"] == (1.0, 2.0)
    json.dumps(metadata.to_dict(), allow_nan=False)


def test_policy_metadata_rejects_non_string_nested_keys() -> None:
    with pytest.raises(ValueError, match="keys"):
        PolicyMetadata(
            "bad_params",
            "1.0",
            True,
            params={"nested": {1: "silently coercing this is unsafe"}},
        )


@pytest.mark.parametrize(
    "ego_index",
    [True, np.bool_(True), 0.0, np.float64(0.5)],
)
def test_policy_observation_ego_index_requires_strict_integer(
    ego_index,
) -> None:
    frame = AgentFrame(
        valid=np.ones(2, dtype=bool),
        x=np.zeros(2),
        y=np.zeros(2),
        heading=np.zeros(2),
        vx=np.zeros(2),
        vy=np.zeros(2),
    )
    with pytest.raises(ValueError, match="ego_index"):
        PolicyObservation(
            current_index=0,
            next_index=1,
            timestamp=0.0,
            next_timestamp=0.1,
            dt=0.1,
            frame=frame,
            next_valid=np.ones(2, dtype=bool),
            agent_ids=(0, 1),
            agent_types=(AgentType.VEHICLE, AgentType.VEHICLE),
            lengths=np.full(2, 4.5),
            widths=np.full(2, 2.0),
            ego_index=ego_index,
        )


class _WriteFlagAttackPolicy(SimulatorPolicy):
    def initialize(self, scenario: Scenario, seed: int):
        return None

    def step(self, state, observation):
        # An owning ndarray with only WRITEABLE cleared can be reopened. The typed
        # observation must be backed by an immutable buffer instead.
        observation.frame.x.setflags(write=True)
        observation.frame.x[1] += 1000.0
        count = observation.frame.num_agents
        return PolicyStep(None, np.zeros(count), np.zeros(count))

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            "write_flag_attack",
            "0.1.0",
            True,
            supported_agent_types=tuple(AgentType),
        )


def test_observation_arrays_cannot_reenable_writes() -> None:
    default_step = PolicyStep(None, np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError):
        default_step.override_mask.setflags(write=True)
    with pytest.raises(ValueError):
        default_step.longitudinal_acceleration.setflags(write=True)

    timestamps = np.array([0.0, 1.0])
    scenario = _simple_scenario(
        timestamps,
        [
            _simple_agent(0, timestamps),
            _simple_agent(1, timestamps, x=np.array([1.0, 2.0]), vx=1.0),
        ],
    )
    before = copy.deepcopy(scenario)
    with pytest.raises(RuntimeError, match="transition 0->1"):
        RolloutEngine().run(scenario, _WriteFlagAttackPolicy())
    _assert_scenario_equal(scenario, before)


class _FrozenDataclassBypassPolicy(SimulatorPolicy):
    def __init__(self) -> None:
        self.policy_metadata = PolicyMetadata(
            "frozen_bypass",
            "0.1.0",
            True,
            supported_agent_types=tuple(AgentType),
        )

    def initialize(self, scenario: Scenario, seed: int):
        object.__setattr__(self.policy_metadata, "name", "mutated_mid_run")
        return None

    def step(self, state, observation):
        # ``frozen=True`` is not a security boundary: object.__setattr__ bypasses it.
        # None of these replacements may alter engine-owned transition semantics.
        object.__setattr__(observation, "ego_index", 1)
        object.__setattr__(observation, "dt", 1000.0)
        object.__setattr__(
            observation,
            "next_valid",
            np.array([True, False]),
        )
        object.__setattr__(
            observation.frame,
            "x",
            np.array([5000.0, 6000.0]),
        )
        object.__setattr__(
            observation.frame,
            "vx",
            np.array([7000.0, 8000.0]),
        )
        count = observation.frame.num_agents
        return PolicyStep(None, np.zeros(count), np.zeros(count))

    def metadata(self) -> PolicyMetadata:
        return self.policy_metadata


def test_engine_uses_private_transition_state_after_policy_callback() -> None:
    timestamps = np.array([0.0, 1.0])
    scenario = _simple_scenario(
        timestamps,
        [
            _simple_agent(
                0,
                timestamps,
                x=np.array([0.0, 10.0]),
                vx=np.array([0.0, 10.0]),
            ),
            _simple_agent(
                1,
                timestamps,
                x=np.array([5.0, 999.0]),
                vx=np.array([1.0, 999.0]),
            ),
        ],
    )
    rollout = RolloutEngine().run(scenario, _FrozenDataclassBypassPolicy())
    np.testing.assert_array_equal(rollout.agents[0].x, [0.0, 10.0])
    np.testing.assert_array_equal(rollout.agents[1].valid, [True, True])
    np.testing.assert_array_equal(rollout.agents[1].x, [5.0, 6.0])
    assert rollout.sim_name == "frozen_bypass"


class _MutatedPolicyStep(SimulatorPolicy):
    def initialize(self, scenario: Scenario, seed: int):
        return None

    def step(self, state, observation):
        count = observation.frame.num_agents
        step = PolicyStep(None, np.zeros(count), np.zeros(count))
        object.__setattr__(
            step,
            "longitudinal_acceleration",
            np.full(count, np.nan),
        )
        return step

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            "mutated_step",
            "0.1.0",
            True,
            supported_agent_types=tuple(AgentType),
        )


def test_policy_step_is_revalidated_after_callback() -> None:
    timestamps = np.array([0.0, 1.0])
    scenario = _simple_scenario(
        timestamps,
        [
            _simple_agent(0, timestamps),
            _simple_agent(1, timestamps, vx=1.0),
        ],
    )
    with pytest.raises(ValueError, match="invalid output.*transition 0->1"):
        RolloutEngine().run(scenario, _MutatedPolicyStep())
