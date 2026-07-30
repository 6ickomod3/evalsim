"""M6 typed ego-plan integration and synchronous-order gates."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pytest

from evalsim.contracts import (
    Agent,
    AgentFrame,
    AgentType,
    EgoInterventionSpec,
    EgoTrajectoryPlan,
    FeasibilityAudit,
    HistoryOnlyPolicyContext,
    HistoryOnlyPolicyObservation,
    HistoryOnlySimulatorPolicy,
    PolicyMetadata,
    PolicyStep,
    PrivilegedPolicyContext,
    PrivilegedSimulatorPolicy,
    Scenario,
    SimulatorPolicy,
)
from evalsim.perturb.m6 import (
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    compile_identity_plan,
    compile_longitudinal_brake_pulse_plan,
)
from evalsim.rollout import RolloutEngine
from evalsim.simulators import (
    ConstantVelocityPolicy,
    IDMPolicy,
    LogReplayPolicy,
)
from evalsim.sources import SyntheticSource

_SERIES_FIELDS = ("valid", "x", "y", "heading", "vx", "vy")
_LEGACY_CV_FIXTURE_SHA256 = (
    "e802b8ce49cc487772a7267bcfca0c1f8fbb972feb287ec865959a7938ced3b0"
)
_LEGACY_IDM_FIXTURE_SHA256 = (
    "737117a02d316f21f7a23eea32c0ee1ea88e215f2475bda0237d073c29567ed7"
)
_LEGACY_LOG_REPLAY_FIXTURE_SHA256 = (
    "8acb47e2806399e868db22d1ad2024f9147be4b8b2ae8971f872e0baa2768877"
)


def _linear_scenario(
    *,
    frame_count: int = 45,
    ego_valid: np.ndarray | None = None,
) -> Scenario:
    timestamps = np.arange(frame_count, dtype=np.float64) * 0.1
    ego = Agent(
        id=100,
        type=AgentType.VEHICLE,
        valid=(
            np.ones(frame_count, dtype=bool)
            if ego_valid is None
            else np.asarray(ego_valid, dtype=bool)
        ),
        x=10.0 * timestamps,
        y=np.zeros(frame_count),
        heading=np.zeros(frame_count),
        vx=np.full(frame_count, 10.0),
        vy=np.zeros(frame_count),
        length=4.5,
        width=2.0,
    )
    world = Agent(
        id=200,
        type=AgentType.VEHICLE,
        valid=np.ones(frame_count, dtype=bool),
        x=-8.0 + 5.0 * timestamps,
        y=np.zeros(frame_count),
        heading=np.zeros(frame_count),
        vx=np.full(frame_count, 5.0),
        vy=np.zeros(frame_count),
        length=4.5,
        width=2.0,
    )
    return Scenario(
        scenario_id="m6-engine-linear",
        timestamps=timestamps,
        agents=[ego, world],
        ego_index=0,
        metadata={
            "source": "unit",
            "source_version": "v1",
            "source_time_unit": "seconds",
            "current_index": 0,
        },
    )


def _assert_rollout_matches_prefix(typed, legacy, stop_index: int) -> None:
    np.testing.assert_array_equal(
        typed.timestamps,
        legacy.timestamps[: stop_index + 1],
    )
    assert typed.num_agents == legacy.num_agents
    for actual, expected in zip(typed.agents, legacy.agents, strict=True):
        assert actual.id == expected.id
        assert actual.type == expected.type
        assert actual.length == expected.length
        assert actual.width == expected.width
        for field_name in _SERIES_FIELDS:
            np.testing.assert_array_equal(
                getattr(actual, field_name),
                getattr(expected, field_name)[: stop_index + 1],
            )


def _legacy_rollout_fixture_digest(rollout) -> str:
    """Canonical test digest frozen from accepted pre-M6 commit ``a312254``."""

    digest = hashlib.sha256(b"evalsim-legacy-rollout-fixture-v1\0")
    parts = [
        json.dumps(
            {
                "scenario_id": rollout.scenario_id,
                "sim_name": rollout.sim_name,
                "sim_version": rollout.sim_version,
                "seed": rollout.seed,
                "perturbation": rollout.perturbation,
                "metadata": rollout.metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
        np.asarray(rollout.timestamps, dtype="<f8").tobytes(order="C"),
    ]
    for agent in rollout.agents:
        parts.extend(
            [
                json.dumps(
                    {
                        "id": agent.id,
                        "type": agent.type.value,
                        "length": agent.length,
                        "width": agent.width,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8"),
                np.asarray(agent.valid, dtype="u1").tobytes(order="C"),
                *(
                    np.asarray(
                        getattr(agent, field_name),
                        dtype="<f8",
                    ).tobytes(order="C")
                    for field_name in ("x", "y", "heading", "vx", "vy")
                ),
            ]
        )
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("policy", "expected_sha256"),
    [
        pytest.param(
            ConstantVelocityPolicy(),
            _LEGACY_CV_FIXTURE_SHA256,
            id="constant_velocity",
        ),
        pytest.param(
            IDMPolicy(),
            _LEGACY_IDM_FIXTURE_SHA256,
            id="idm",
        ),
        pytest.param(
            LogReplayPolicy(),
            _LEGACY_LOG_REPLAY_FIXTURE_SHA256,
            id="log_replay",
        ),
    ],
)
def test_legacy_no_plan_bytes_match_pre_m6_frozen_fixture(
    policy,
    expected_sha256: str,
) -> None:
    scenario = SyntheticSource(seed=2026).generate_one(0)
    rollout = RolloutEngine().run(
        scenario,
        policy,
        seed=17,
    )
    assert rollout.perturbation is None
    assert rollout.metadata["ego_control"] == "logged"
    assert (
        rollout.metadata["agent_control_modes"][str(scenario.ego.id)]
        == "logged_ego"
    )
    assert _legacy_rollout_fixture_digest(rollout) == expected_sha256


@pytest.mark.parametrize(
    "policy",
    [LogReplayPolicy(), ConstantVelocityPolicy(), IDMPolicy()],
    ids=["privileged_log_replay", "history_only_cv", "history_only_idm"],
)
def test_identity_plan_is_exact_legacy_prefix_for_all_builtins(policy) -> None:
    scenario = _linear_scenario()
    before = copy.deepcopy(scenario)
    plan = compile_identity_plan(scenario)
    engine = RolloutEngine()

    legacy = engine.run(scenario, policy, seed=0)
    typed = engine.run(scenario, policy, seed=0, ego_plan=plan)

    assert typed.num_steps == 41
    assert legacy.num_steps == scenario.num_steps
    assert typed.perturbation == plan.perturbation_identity
    assert typed.metadata["ego_control"] == "typed_ego_plan"
    assert legacy.perturbation is None
    assert legacy.metadata["ego_control"] == "logged"
    _assert_rollout_matches_prefix(typed, legacy, stop_index=40)
    np.testing.assert_array_equal(scenario.timestamps, before.timestamps)
    for actual, expected in zip(scenario.agents, before.agents, strict=True):
        for field_name in _SERIES_FIELDS:
            np.testing.assert_array_equal(
                getattr(actual, field_name),
                getattr(expected, field_name),
            )


@dataclass(frozen=True, slots=True)
class _EgoSensitivePolicy(HistoryOnlySimulatorPolicy):
    """React only after the realized ego speed reveals the treatment."""

    def initialize(
        self,
        context: HistoryOnlyPolicyContext,
        seed: int,
    ) -> int:
        return len(context.agent_ids)

    def step(
        self,
        state: int,
        observation: HistoryOnlyPolicyObservation,
    ) -> PolicyStep:
        assert state == observation.frame.num_agents
        acceleration = np.zeros(state)
        ego_speed = float(
            np.hypot(
                observation.frame.vx[observation.ego_index],
                observation.frame.vy[observation.ego_index],
            )
        )
        if ego_speed < 9.9:
            acceleration[1] = -2.0
        return PolicyStep(state, acceleration, np.zeros(state))

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="ego_sensitive_unit",
            version="1.0.0",
            deterministic=True,
            supported_agent_types=tuple(AgentType),
        )


def test_typed_plan_applies_ego_after_world_action_and_response_starts_at_t2() -> None:
    scenario = _linear_scenario(frame_count=41)
    identity = compile_identity_plan(scenario)
    brake = compile_longitudinal_brake_pulse_plan(
        scenario,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    engine = RolloutEngine()
    policy = _EgoSensitivePolicy()

    baseline = engine.run(scenario, policy, seed=0, ego_plan=identity)
    treatment = engine.run(scenario, policy, seed=0, ego_plan=brake)

    # The first world transition saw the exact shared current ego.
    for field_name in ("x", "y", "heading", "vx", "vy"):
        np.testing.assert_array_equal(
            getattr(baseline.agents[1], field_name)[:2],
            getattr(treatment.agents[1], field_name)[:2],
        )
    assert treatment.agents[0].x[1] < baseline.agents[0].x[1]
    assert treatment.agents[0].vx[1] < baseline.agents[0].vx[1]
    # The changed ego is first observable to the next action, whose state lands at t+2.
    assert treatment.agents[1].vx[2] < baseline.agents[1].vx[2]
    assert treatment.agents[1].x[2] < baseline.agents[1].x[2]


@dataclass(frozen=True, slots=True)
class _HistoryOverrideAttack(HistoryOnlySimulatorPolicy):
    def initialize(
        self,
        context: HistoryOnlyPolicyContext,
        seed: int,
    ) -> None:
        return None

    def step(
        self,
        state: None,
        observation: HistoryOnlyPolicyObservation,
    ) -> PolicyStep:
        count = observation.frame.num_agents
        mask = np.zeros(count, dtype=bool)
        mask[1] = True
        return PolicyStep(
            next_state=None,
            longitudinal_acceleration=np.zeros(count),
            yaw_rate=np.zeros(count),
            override=AgentFrame(
                valid=observation.frame.valid,
                x=observation.frame.x,
                y=observation.frame.y,
                heading=observation.frame.heading,
                vx=observation.frame.vx,
                vy=observation.frame.vy,
            ),
            override_mask=mask,
        )

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            "history_override_attack",
            "1.0.0",
            True,
            supported_agent_types=tuple(AgentType),
        )


def test_engine_rejects_absolute_override_from_history_only_policy() -> None:
    scenario = _linear_scenario(frame_count=41)
    with pytest.raises(
        ValueError,
        match="history-only policies cannot return absolute-state overrides",
    ):
        RolloutEngine().run(
            scenario,
            _HistoryOverrideAttack(),
            ego_plan=compile_identity_plan(scenario),
        )


class _PlainPolicy(SimulatorPolicy):
    def step(self, state, observation) -> PolicyStep:
        count = observation.frame.num_agents
        return PolicyStep(None, np.zeros(count), np.zeros(count))

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata("plain_policy", "1.0.0", True)


class _DualPolicy(HistoryOnlySimulatorPolicy, PrivilegedSimulatorPolicy):
    def initialize(
        self,
        context: HistoryOnlyPolicyContext | PrivilegedPolicyContext,
        seed: int,
    ) -> None:
        return None

    def step(self, state, observation) -> PolicyStep:
        count = observation.frame.num_agents
        return PolicyStep(None, np.zeros(count), np.zeros(count))

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata("dual_policy", "1.0.0", True)


@pytest.mark.parametrize("policy", [_PlainPolicy(), _DualPolicy()])
def test_engine_rejects_plain_or_dual_capability_policy(policy) -> None:
    with pytest.raises(TypeError, match="exactly one"):
        RolloutEngine().run(_linear_scenario(frame_count=41), policy)


def test_typed_horizon_ignores_source_ego_invalidity_after_stop_only() -> None:
    ego_valid = np.ones(45, dtype=bool)
    ego_valid[41:] = False
    scenario = _linear_scenario(ego_valid=ego_valid)
    plan = compile_identity_plan(scenario)

    typed = RolloutEngine().run(
        scenario,
        ConstantVelocityPolicy(),
        ego_plan=plan,
    )
    assert typed.num_steps == 41
    with pytest.raises(ValueError, match="execution horizon"):
        RolloutEngine().run(scenario, ConstantVelocityPolicy())


def test_plan_compiled_for_another_current_state_is_rejected() -> None:
    source = _linear_scenario(frame_count=41)
    alien = _linear_scenario(frame_count=41)
    alien.agents[0].x = alien.agents[0].x + 1.0
    alien_plan = compile_identity_plan(alien)

    with pytest.raises(ValueError, match="source_binding_mismatch"):
        RolloutEngine().run(
            source,
            ConstantVelocityPolicy(),
            ego_plan=alien_plan,
        )


def test_engine_recompiles_registered_plan_instead_of_trusting_forged_audit() -> None:
    scenario = _linear_scenario(frame_count=41)
    identity = compile_identity_plan(scenario)
    forged_x = np.array(identity.x, copy=True)
    forged_x[1:] = 1e9
    forged_vx = np.array(identity.vx, copy=True)
    forged_vx[1:] = 1e9
    forged = EgoTrajectoryPlan(
        spec=identity.spec,
        timestamps=identity.timestamps,
        valid=identity.valid,
        applied=identity.applied,
        x=forged_x,
        y=identity.y,
        heading=identity.heading,
        vx=forged_vx,
        vy=identity.vy,
        realization_type=identity.realization_type,
        feasibility=FeasibilityAudit.accepted({"fabricated": True}),
    )

    with pytest.raises(ValueError, match="source_binding_mismatch"):
        RolloutEngine().run(
            scenario,
            ConstantVelocityPolicy(),
            ego_plan=forged,
        )


def test_engine_rejects_canonical_but_unregistered_intervention_family() -> None:
    scenario = _linear_scenario(frame_count=41)
    identity = compile_identity_plan(scenario)
    deferred = EgoTrajectoryPlan(
        spec=EgoInterventionSpec(
            family="path_normal_offset",
            version="v1",
            dose=1.0,
            duration_s=1.0,
            parameters={},
        ),
        timestamps=identity.timestamps,
        valid=identity.valid,
        applied=identity.applied,
        x=identity.x,
        y=identity.y,
        heading=identity.heading,
        vx=identity.vx,
        vy=identity.vy,
        realization_type=identity.realization_type,
        feasibility=FeasibilityAudit.accepted({"fabricated": True}),
    )

    with pytest.raises(ValueError, match="family_unregistered"):
        RolloutEngine().run(
            scenario,
            ConstantVelocityPolicy(),
            ego_plan=deferred,
        )
