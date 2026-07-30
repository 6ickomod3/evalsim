"""M6 policy-execution sidecar and sham/legacy equality gates."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from evalsim.contracts import Agent, AgentType, Scenario
from evalsim.contracts.counterfactual import RolloutSnapshot
from evalsim.perturb.m6 import (
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    compile_identity_plan,
    compile_longitudinal_brake_pulse_plan,
)
from evalsim.rollout import (
    PolicyExecutionTrace,
    RolloutEngine,
    TracedRollout,
    policy_trace_prefix_equal,
)
from evalsim.simulators import (
    ConstantVelocityPolicy,
    IDMPolicy,
    LogReplayPolicy,
)

_FIELDS = ("valid", "x", "y", "heading", "vx", "vy")


def _scenario(*, lifecycle_birth: bool = False) -> Scenario:
    frames = 45
    timestamps = np.arange(frames, dtype=np.float64) * 0.1
    ego = Agent(
        id=101,
        type=AgentType.VEHICLE,
        valid=np.ones(frames, dtype=bool),
        x=10.0 * timestamps,
        y=np.zeros(frames),
        heading=np.zeros(frames),
        vx=np.full(frames, 10.0),
        vy=np.zeros(frames),
        length=4.5,
        width=2.0,
    )
    world_valid = np.ones(frames, dtype=bool)
    if lifecycle_birth:
        world_valid[0] = False
    world = Agent(
        id=202,
        type=AgentType.VEHICLE,
        valid=world_valid,
        x=-10.0 + 8.0 * timestamps,
        y=np.zeros(frames),
        heading=np.zeros(frames),
        vx=np.full(frames, 8.0),
        vy=np.zeros(frames),
        length=4.5,
        width=2.0,
    )
    return Scenario(
        scenario_id="m6-trace-unit",
        timestamps=timestamps,
        agents=[ego, world],
        ego_index=0,
        metadata={
            "current_index": 0,
            "source": "unit",
            "source_version": "v1",
            "source_time_unit": "seconds",
        },
    )


def _assert_numeric_prefix(typed: TracedRollout, legacy: TracedRollout) -> None:
    stop = typed.rollout.num_steps
    np.testing.assert_array_equal(
        typed.rollout.timestamps,
        legacy.rollout.timestamps[:stop],
    )
    for actual, expected in zip(
        typed.rollout.agents,
        legacy.rollout.agents,
        strict=True,
    ):
        for name in _FIELDS:
            np.testing.assert_array_equal(
                getattr(actual, name),
                getattr(expected, name)[:stop],
            )


@pytest.mark.parametrize(
    ("policy", "access_role"),
    [
        (ConstantVelocityPolicy(), "history_only"),
        (IDMPolicy(), "history_only"),
        (LogReplayPolicy(), "privileged"),
    ],
    ids=["constant_velocity", "idm", "log_replay"],
)
def test_identity_trace_exactly_matches_legacy_world_policy_prefix(
    policy,
    access_role: str,
) -> None:
    scenario = _scenario()
    identity = compile_identity_plan(scenario)
    engine = RolloutEngine()
    legacy = engine.run_with_trace(scenario, policy, seed=0)
    typed = engine.run_with_trace(
        scenario,
        policy,
        seed=0,
        ego_plan=identity,
    )

    _assert_numeric_prefix(typed, legacy)
    assert policy_trace_prefix_equal(typed.trace, legacy.trace)
    assert typed.trace.policy_access_role == access_role
    assert typed.trace.control_modes[scenario.ego_index] == "typed_ego_plan"
    assert legacy.trace.control_modes[scenario.ego_index] == "logged_ego"
    assert typed.trace.perturbation_identity == identity.perturbation_identity
    assert legacy.trace.perturbation_identity is None
    assert not hasattr(typed.trace, "agent_ids")
    with pytest.raises(ValueError):
        typed.trace.longitudinal_acceleration.setflags(write=True)


def test_trace_records_engine_effective_and_birth_masks() -> None:
    scenario = _scenario(lifecycle_birth=True)
    identity = compile_identity_plan(scenario)
    traced = RolloutEngine().run_with_trace(
        scenario,
        ConstantVelocityPolicy(),
        ego_plan=identity,
    )

    assert traced.trace.lifecycle_birth_mask[0].tolist() == [False, True]
    assert traced.trace.effective_control_mask[0].tolist() == [False, False]
    assert traced.trace.effective_control_mask[1].tolist() == [False, True]
    assert not np.any(traced.trace.override_mask)


def test_privileged_trace_records_only_effective_world_overrides() -> None:
    scenario = _scenario()
    identity = compile_identity_plan(scenario)
    traced = RolloutEngine().run_with_trace(
        scenario,
        LogReplayPolicy(),
        ego_plan=identity,
    )

    assert not np.any(traced.trace.override_mask[:, scenario.ego_index])
    assert np.all(traced.trace.override_mask[:, 1])
    assert np.all(traced.trace.override_valid[:, 1])
    np.testing.assert_array_equal(
        traced.trace.override_x[:, 1],
        scenario.agents[1].x[1:41],
    )

    lifecycle = _scenario(lifecycle_birth=True)
    lifecycle_trace = RolloutEngine().run_with_trace(
        lifecycle,
        LogReplayPolicy(),
        ego_plan=compile_identity_plan(lifecycle),
    ).trace
    # The privileged policy requested its logged override, but the engine-owned
    # birth fallback superseded it on this transition.
    assert lifecycle_trace.override_mask[0, 1]
    assert lifecycle_trace.lifecycle_birth_mask[0, 1]
    assert not lifecycle_trace.effective_control_mask[0, 1]


def test_trace_revalidation_detects_postconstruction_bypass() -> None:
    traced = RolloutEngine().run_with_trace(
        _scenario(),
        ConstantVelocityPolicy(),
    )
    assert isinstance(traced.trace, PolicyExecutionTrace)
    object.__setattr__(
        traced.trace,
        "yaw_rate",
        np.ones_like(traced.trace.yaw_rate),
    )
    with pytest.raises(ValueError, match="trace was mutated"):
        traced.trace.revalidate()


def test_traced_rollout_rejects_plan_and_trace_misbinding() -> None:
    scenario = _scenario()
    engine = RolloutEngine()
    left = engine.run_with_trace(scenario, ConstantVelocityPolicy())
    right = engine.run_with_trace(scenario, IDMPolicy())
    with pytest.raises(ValueError, match="rollout_fingerprint|policy identity"):
        TracedRollout(left.rollout, right.trace)


@pytest.mark.parametrize(
    "field_name",
    ["effective_control_mask", "lifecycle_birth_mask"],
)
def test_trace_rebinds_validity_derived_masks(field_name: str) -> None:
    scenario = _scenario(lifecycle_birth=True)
    traced = RolloutEngine().run_with_trace(
        scenario,
        ConstantVelocityPolicy(),
        ego_plan=compile_identity_plan(scenario),
    )
    values = np.array(getattr(traced.trace, field_name), copy=True)
    values[0, 1] = ~values[0, 1]
    with pytest.raises(
        ValueError,
        match=f"{field_name}|birth fallback",
    ):
        tampered = replace(traced.trace, **{field_name: values})
        TracedRollout(traced.rollout, tampered)


def test_trace_rejects_override_payload_outside_mask() -> None:
    scenario = _scenario()
    traced = RolloutEngine().run_with_trace(
        scenario,
        ConstantVelocityPolicy(),
        ego_plan=compile_identity_plan(scenario),
    )
    override_x = np.array(traced.trace.override_x, copy=True)
    override_x[0, 1] = 123.0
    tampered = replace(traced.trace, override_x=override_x)
    with pytest.raises(ValueError, match="zero outside override_mask"):
        TracedRollout(traced.rollout, tampered)


def test_trace_rejects_effective_override_that_does_not_match_output() -> None:
    scenario = _scenario()
    traced = RolloutEngine().run_with_trace(
        scenario,
        LogReplayPolicy(),
        ego_plan=compile_identity_plan(scenario),
    )
    override_x = np.array(traced.trace.override_x, copy=True)
    override_x[0, 1] += 1.0
    tampered = replace(traced.trace, override_x=override_x)
    with pytest.raises(ValueError, match="effective override_x"):
        TracedRollout(traced.rollout, tampered)


def test_trace_rejects_controls_that_do_not_replay_exact_output() -> None:
    scenario = _scenario()
    traced = RolloutEngine().run_with_trace(
        scenario,
        ConstantVelocityPolicy(),
        ego_plan=compile_identity_plan(scenario),
    )
    acceleration = np.array(
        traced.trace.longitudinal_acceleration,
        copy=True,
    )
    acceleration[0, 1] = 1.0
    tampered = replace(
        traced.trace,
        longitudinal_acceleration=acceleration,
    )
    with pytest.raises(ValueError, match="kinematic controls"):
        TracedRollout(traced.rollout, tampered)


def test_trace_rejects_clamp_count_metadata_even_with_rebound_digest() -> None:
    scenario = _scenario()
    traced = RolloutEngine().run_with_trace(
        scenario,
        IDMPolicy(),
        ego_plan=compile_identity_plan(scenario),
    )
    rollout = copy.deepcopy(traced.rollout)
    rollout.metadata["dynamics"]["clamp_counts"]["speed"] += 1
    rebound = replace(
        traced.trace,
        rollout_fingerprint=(
            RolloutSnapshot.from_rollout(rollout)._integrity_fingerprint
        ),
    )
    with pytest.raises(ValueError, match="clamp counts"):
        TracedRollout(rollout, rebound)


@pytest.mark.parametrize(
    "policy",
    [LogReplayPolicy(), ConstantVelocityPolicy(), IDMPolicy()],
    ids=["log_replay", "constant_velocity", "idm"],
)
@pytest.mark.parametrize(
    "condition",
    ["legacy", "identity", "primary_b2"],
)
def test_run_with_trace_rollout_is_exact_direct_run_oracle(
    policy,
    condition: str,
) -> None:
    scenario = _scenario()
    plan = None
    if condition == "identity":
        plan = compile_identity_plan(scenario)
    elif condition == "primary_b2":
        plan = compile_longitudinal_brake_pulse_plan(
            scenario,
            PRIMARY_BRAKE_MAGNITUDE_MPS2,
        )
    engine = RolloutEngine()
    direct = engine.run(scenario, policy, seed=0, ego_plan=plan)
    traced = engine.run_with_trace(
        scenario,
        policy,
        seed=0,
        ego_plan=plan,
    )
    direct_snapshot = RolloutSnapshot.from_rollout(direct)
    traced_snapshot = RolloutSnapshot.from_rollout(traced.rollout)
    assert (
        direct_snapshot._integrity_fingerprint
        == traced_snapshot._integrity_fingerprint
    )
    traced.revalidate()
