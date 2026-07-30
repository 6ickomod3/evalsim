"""M6 policy-access capability and future-leakage contract tests."""
from __future__ import annotations

import copy
from dataclasses import fields
from types import MappingProxyType

import numpy as np
import pytest

from evalsim.contracts import (
    Agent,
    AgentFrame,
    AgentType,
    HistoryOnlyPolicyContext,
    HistoryOnlyPolicyObservation,
    HistoryOnlySimulatorPolicy,
    MapPolyline,
    MapType,
    PolicyMetadata,
    PolicyObservation,
    PolicyStep,
    PrivilegedPolicyContext,
    PrivilegedSimulatorPolicy,
    Scenario,
    SimulatorPolicy,
)
from evalsim.simulators import (
    CONSTANT_VELOCITY_VERSION,
    IDM_VERSION,
    LOG_REPLAY_VERSION,
    ConstantVelocityPolicy,
    IDMPolicy,
    LogReplayPolicy,
)


def _agent(
    agent_id: int,
    *,
    x: np.ndarray,
    valid: np.ndarray | None = None,
    vx: np.ndarray | None = None,
) -> Agent:
    count = len(x)
    return Agent(
        id=agent_id,
        type=AgentType.VEHICLE,
        valid=(
            np.ones(count, dtype=bool)
            if valid is None
            else np.array(valid, dtype=bool)
        ),
        x=np.array(x, dtype=float),
        y=np.zeros(count),
        heading=np.zeros(count),
        vx=(
            np.ones(count)
            if vx is None
            else np.array(vx, dtype=float)
        ),
        vy=np.zeros(count),
        length=4.5,
        width=2.0,
    )


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="m6-policy-access",
        timestamps=np.array([0.0, 0.1, 0.25, 0.4]),
        agents=[
            _agent(100, x=np.array([10.0, 11.0, 12.0, 13.0])),
            _agent(200, x=np.array([0.0, 1.0, 2.0, 3.0])),
        ],
        map=[
            MapPolyline(
                type=MapType.LANE,
                xy=np.array([[0.0, 0.0], [20.0, 0.0]]),
            )
        ],
        ego_index=0,
        metadata={
            "source": "unit",
            "source_version": {
                "release": ["1.3.1", {"patch": 0}],
            },
            "source_time_unit": "seconds",
            "current_index": 1,
            "future_ego_plan": {"dose": 999.0},
            "private_source_path": "/must/not/cross/the/boundary",
        },
    )


def _observation(
    context: HistoryOnlyPolicyContext,
) -> HistoryOnlyPolicyObservation:
    next_index = context.current_index + 1
    # The next timestamp is permitted timing information. No next-frame validity or
    # other source motion enters the observation.
    return HistoryOnlyPolicyObservation(
        current_index=context.current_index,
        next_index=next_index,
        timestamp=float(context.timestamps[-1]),
        next_timestamp=0.25,
        dt=0.15,
        frame=context.frames[-1],
        agent_ids=context.agent_ids,
        agent_types=context.agent_types,
        lengths=context.lengths,
        widths=context.widths,
        ego_index=context.ego_index,
    )


def _assert_frames_equal(
    left: tuple[AgentFrame, ...],
    right: tuple[AgentFrame, ...],
) -> None:
    assert len(left) == len(right)
    for left_frame, right_frame in zip(left, right, strict=True):
        for name in ("valid", "x", "y", "heading", "vx", "vy"):
            np.testing.assert_array_equal(
                getattr(left_frame, name),
                getattr(right_frame, name),
            )


def test_capabilities_are_disjoint_and_builtin_roles_are_explicit() -> None:
    assert isinstance(ConstantVelocityPolicy(), HistoryOnlySimulatorPolicy)
    assert isinstance(IDMPolicy(), HistoryOnlySimulatorPolicy)
    assert not isinstance(ConstantVelocityPolicy(), PrivilegedSimulatorPolicy)
    assert not isinstance(IDMPolicy(), PrivilegedSimulatorPolicy)

    assert isinstance(LogReplayPolicy(), PrivilegedSimulatorPolicy)
    assert not isinstance(LogReplayPolicy(), HistoryOnlySimulatorPolicy)

    class PlainPolicy(SimulatorPolicy):
        def step(self, state, observation):
            return PolicyStep(state, np.zeros(1), np.zeros(1))

        def metadata(self):
            return PolicyMetadata("plain", "0.0.1", True)

    class DualPolicy(HistoryOnlySimulatorPolicy, PrivilegedSimulatorPolicy):
        def initialize(self, context, seed):
            return None

        def step(self, state, observation):
            return PolicyStep(state, np.zeros(1), np.zeros(1))

        def metadata(self):
            return PolicyMetadata("dual", "0.0.1", True)

    plain = PlainPolicy()
    dual = DualPolicy()
    assert isinstance(plain, SimulatorPolicy)
    assert not isinstance(plain, HistoryOnlySimulatorPolicy)
    assert not isinstance(plain, PrivilegedSimulatorPolicy)
    assert isinstance(dual, HistoryOnlySimulatorPolicy)
    assert isinstance(dual, PrivilegedSimulatorPolicy)


def test_history_context_is_allowlisted_defensive_and_deeply_immutable() -> None:
    scenario = _scenario()
    context = HistoryOnlyPolicyContext.from_scenario(scenario)

    assert context.current_index == 1
    assert context.future_step_count == 2
    assert len(context.frames) == 2
    np.testing.assert_array_equal(context.timestamps, [0.0, 0.1])
    assert set(context.source_provenance) == {
        "source",
        "source_version",
        "source_time_unit",
    }
    assert isinstance(context.source_provenance, MappingProxyType)
    release = context.source_provenance["source_version"]["release"]
    assert isinstance(release, tuple)
    assert isinstance(release[1], MappingProxyType)
    assert not {
        "scenario",
        "reference",
        "next_valid",
        "future_ego_plan",
        "intervention",
        "dose",
        "ego_plan",
    } & {item.name for item in fields(context)}

    immutable_arrays = [
        context.lengths,
        context.widths,
        context.timestamps,
        context.map_features[0].xy,
    ]
    for frame in context.frames:
        immutable_arrays.extend(
            getattr(frame, name)
            for name in ("valid", "x", "y", "heading", "vx", "vy")
        )
    for array in immutable_arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        context.source_provenance["source"] = "changed"
    with pytest.raises(TypeError):
        release[1]["patch"] = 1

    # Mutating the caller-owned Scenario after construction cannot rewrite either
    # observed history, static map data, or frozen provenance.
    scenario.timestamps[0] = 99.0
    scenario.agents[0].x[0] = 99.0
    scenario.map[0].xy[0, 0] = 99.0
    scenario.metadata["source_version"]["release"][1]["patch"] = 99
    np.testing.assert_array_equal(context.timestamps, [0.0, 0.1])
    np.testing.assert_array_equal(context.frames[0].x, [10.0, 0.0])
    np.testing.assert_array_equal(
        context.map_features[0].xy,
        [[0.0, 0.0], [20.0, 0.0]],
    )
    assert release[1]["patch"] == 0


@pytest.mark.parametrize(
    "policy",
    [ConstantVelocityPolicy(), IDMPolicy()],
    ids=["constant_velocity", "idm"],
)
def test_future_motion_and_validity_poison_do_not_change_history_only_actions(
    policy: HistoryOnlySimulatorPolicy,
) -> None:
    clean = _scenario()
    poisoned = copy.deepcopy(clean)
    current = int(clean.metadata["current_index"])
    for agent_index, agent in enumerate(poisoned.agents):
        agent.valid[current + 1 :] = np.array([False, True])
        agent.x[current + 1 :] = 1000.0 + 100.0 * agent_index
        agent.y[current + 1 :] = -2000.0 - 100.0 * agent_index
        agent.heading[current + 1 :] = np.array([1.0, -1.0])
        agent.vx[current + 1 :] = np.array([50.0, -50.0])
        agent.vy[current + 1 :] = np.array([-40.0, 40.0])

    clean_context = HistoryOnlyPolicyContext.from_scenario(clean)
    poisoned_context = HistoryOnlyPolicyContext.from_scenario(poisoned)
    np.testing.assert_array_equal(
        clean_context.timestamps,
        poisoned_context.timestamps,
    )
    _assert_frames_equal(clean_context.frames, poisoned_context.frames)

    clean_state = policy.initialize(clean_context, seed=20260729)
    poisoned_state = policy.initialize(poisoned_context, seed=20260729)
    clean_step = policy.step(clean_state, _observation(clean_context))
    poisoned_step = policy.step(
        poisoned_state,
        _observation(poisoned_context),
    )
    np.testing.assert_array_equal(
        clean_step.longitudinal_acceleration,
        poisoned_step.longitudinal_acceleration,
    )
    np.testing.assert_array_equal(clean_step.yaw_rate, poisoned_step.yaw_rate)
    assert not hasattr(_observation(clean_context), "next_valid")
    with pytest.raises(TypeError, match="next_valid"):
        HistoryOnlyPolicyObservation(
            current_index=1,
            next_index=2,
            timestamp=0.1,
            next_timestamp=0.25,
            dt=0.15,
            frame=clean_context.frames[-1],
            agent_ids=clean_context.agent_ids,
            agent_types=clean_context.agent_types,
            lengths=clean_context.lengths,
            widths=clean_context.widths,
            ego_index=clean_context.ego_index,
            next_valid=np.ones(2, dtype=bool),  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    "policy",
    [ConstantVelocityPolicy(), IDMPolicy()],
    ids=["constant_velocity", "idm"],
)
def test_future_validity_alone_is_absent_from_history_only_access(
    policy: HistoryOnlySimulatorPolicy,
) -> None:
    clean = _scenario()
    poisoned = copy.deepcopy(clean)
    current = int(clean.metadata["current_index"])
    for agent in poisoned.agents:
        agent.valid[current + 1 :] = ~agent.valid[current + 1 :]

    clean_context = HistoryOnlyPolicyContext.from_scenario(clean)
    poisoned_context = HistoryOnlyPolicyContext.from_scenario(poisoned)
    _assert_frames_equal(clean_context.frames, poisoned_context.frames)
    clean_step = policy.step(
        policy.initialize(clean_context, seed=11),
        _observation(clean_context),
    )
    poisoned_step = policy.step(
        policy.initialize(poisoned_context, seed=11),
        _observation(poisoned_context),
    )
    np.testing.assert_array_equal(
        clean_step.longitudinal_acceleration,
        poisoned_step.longitudinal_acceleration,
    )
    np.testing.assert_array_equal(clean_step.yaw_rate, poisoned_step.yaw_rate)


def test_privileged_replay_gets_immutable_full_reference_and_tracks_poison() -> None:
    clean = _scenario()
    poisoned = copy.deepcopy(clean)
    poisoned.agents[1].x[2] = 1234.0
    poisoned.agents[1].valid[2] = False

    clean_context = PrivilegedPolicyContext.from_scenario(clean)
    poisoned_context = PrivilegedPolicyContext.from_scenario(poisoned)
    assert len(clean_context.frames) == clean.num_steps
    assert clean_context.future_step_count == 2
    with pytest.raises(ValueError):
        clean_context.frames[2].x.setflags(write=True)

    policy = LogReplayPolicy()
    observation = _observation(HistoryOnlyPolicyContext.from_scenario(clean))
    clean_step = policy.step(
        policy.initialize(clean_context, seed=7),
        observation,
    )
    poisoned_step = policy.step(
        policy.initialize(poisoned_context, seed=7),
        observation,
    )
    assert clean_step.override is not None
    assert poisoned_step.override is not None
    assert clean_step.override.x[1] == 2.0
    assert poisoned_step.override.x[1] == 1234.0
    assert clean_step.override.valid[1]
    assert not poisoned_step.override.valid[1]

    # Both privileged references are detached from later caller mutation.
    clean.agents[1].x[2] = -999.0
    assert clean_step.override.x[1] == 2.0


def test_run_local_states_can_be_interleaved_on_reused_policy_instances() -> None:
    clean = _scenario()
    changed_history = copy.deepcopy(clean)
    changed_history.agents[0].x[1] = 4.0
    changed_history.agents[1].x[1] = -6.0

    for policy in (ConstantVelocityPolicy(), IDMPolicy()):
        clean_context = HistoryOnlyPolicyContext.from_scenario(clean)
        changed_context = HistoryOnlyPolicyContext.from_scenario(changed_history)
        clean_state = policy.initialize(clean_context, seed=5)
        changed_state = policy.initialize(changed_context, seed=9)
        first_clean = policy.step(
            clean_state,
            _observation(clean_context),
        )
        policy.step(changed_state, _observation(changed_context))
        second_clean = policy.step(
            clean_state,
            _observation(clean_context),
        )
        np.testing.assert_array_equal(
            first_clean.longitudinal_acceleration,
            second_clean.longitudinal_acceleration,
        )
        np.testing.assert_array_equal(
            first_clean.yaw_rate,
            second_clean.yaw_rate,
        )

    replay = LogReplayPolicy()
    clean_privileged = PrivilegedPolicyContext.from_scenario(clean)
    changed_privileged = PrivilegedPolicyContext.from_scenario(changed_history)
    clean_state = replay.initialize(clean_privileged, seed=5)
    changed_state = replay.initialize(changed_privileged, seed=9)
    observation = _observation(HistoryOnlyPolicyContext.from_scenario(clean))
    clean_before = replay.step(clean_state, observation)
    replay.step(changed_state, observation)
    clean_after = replay.step(clean_state, observation)
    assert clean_before.override is not None
    assert clean_after.override is not None
    np.testing.assert_array_equal(clean_before.override.x, clean_after.override.x)


def test_legacy_observation_name_and_algorithm_metadata_are_preserved() -> None:
    assert PolicyObservation is HistoryOnlyPolicyObservation

    policies_and_versions = (
        (ConstantVelocityPolicy(), CONSTANT_VELOCITY_VERSION),
        (IDMPolicy(), IDM_VERSION),
        (LogReplayPolicy(), LOG_REPLAY_VERSION),
    )
    for policy, expected_version in policies_and_versions:
        metadata = policy.metadata().to_dict()
        assert metadata["version"] == expected_version == "0.1.0"
        assert "access_role" not in metadata


def test_builtin_initializers_reject_the_legacy_full_scenario_argument() -> None:
    scenario = _scenario()
    with pytest.raises(TypeError, match="HistoryOnlyPolicyContext"):
        ConstantVelocityPolicy().initialize(scenario, seed=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="HistoryOnlyPolicyContext"):
        IDMPolicy().initialize(scenario, seed=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PrivilegedPolicyContext"):
        LogReplayPolicy().initialize(scenario, seed=0)  # type: ignore[arg-type]


def test_history_context_rejects_non_allowlisted_provenance() -> None:
    context = HistoryOnlyPolicyContext.from_scenario(_scenario())
    values = {
        item.name: getattr(context, item.name)
        for item in fields(context)
    }
    values["source_provenance"] = {
        **dict(context.source_provenance),
        "current_index": 1,
    }
    with pytest.raises(ValueError, match="non-allowlisted"):
        HistoryOnlyPolicyContext(**values)
