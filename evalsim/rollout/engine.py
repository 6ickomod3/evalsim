"""Stateless, deterministic closed-loop rollout orchestration."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evalsim.contracts import (
    Agent,
    AgentFrame,
    HistoryOnlyPolicyContext,
    HistoryOnlyPolicyObservation,
    HistoryOnlySimulatorPolicy,
    PolicyMetadata,
    PolicyStep,
    PrivilegedPolicyContext,
    PrivilegedSimulatorPolicy,
    Rollout,
    Scenario,
    SimulatorPolicy,
)
from evalsim.contracts.counterfactual import (
    M6_ANALYSIS_TRANSITIONS,
    EgoTrajectoryPlan,
    RolloutSnapshot,
)

from .dynamics import (
    DYNAMICS_NAME,
    DYNAMICS_VERSION,
    ClampCounts,
    DynamicsLimits,
    integrate_point_mass,
)
from .trace import PolicyExecutionTrace, TracedRollout

ROLLOUT_ENGINE_NAME = "numpy_rollout_engine"
ROLLOUT_ENGINE_VERSION = "0.1.0"
_FRAME_FIELDS = ("x", "y", "heading", "vx", "vy")


def _validate_seed(seed: int) -> int:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= int(seed) <= np.iinfo(np.uint32).max
    ):
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    return int(seed)


def _rollout_start_index(scenario: Scenario) -> int:
    raw = scenario.metadata.get("current_index", 0)
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, np.integer))
        or not 0 <= int(raw) < scenario.num_steps
    ):
        raise ValueError(
            "scenario.metadata['current_index'] must be an integer inside "
            "the scenario horizon"
        )
    return int(raw)


def _validate_scenario_for_rollout(
    scenario: Scenario,
    *,
    future_steps: int | None = None,
) -> int:
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if not isinstance(scenario.scenario_id, str) or not scenario.scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    if scenario.num_steps < 1:
        raise ValueError("scenario must contain at least one timestep")
    if scenario.num_agents < 1:
        raise ValueError("scenario must contain at least one agent")
    if (
        isinstance(scenario.ego_index, (bool, np.bool_))
        or not isinstance(scenario.ego_index, (int, np.integer))
        or not 0 <= int(scenario.ego_index) < scenario.num_agents
    ):
        raise ValueError("scenario ego_index must be an in-range integer")
    if (
        not np.all(np.isfinite(scenario.timestamps))
        or (
            scenario.num_steps > 1
            and not np.all(np.diff(scenario.timestamps) > 0.0)
        )
    ):
        raise ValueError("scenario timestamps must be finite and strictly increasing")

    ids = [agent.id for agent in scenario.agents]
    if any(
        isinstance(agent_id, bool)
        or not isinstance(agent_id, (int, np.integer))
        for agent_id in ids
    ):
        raise ValueError("scenario agent IDs must be integers")
    if len(set(int(agent_id) for agent_id in ids)) != len(ids):
        raise ValueError("scenario agent IDs must be unique")

    for agent in scenario.agents:
        if not np.any(agent.valid):
            raise ValueError(f"agent {agent.id} is never valid")
        if (
            not np.isfinite(agent.length)
            or not np.isfinite(agent.width)
            or agent.length <= 0.0
            or agent.width <= 0.0
        ):
            raise ValueError(
                f"agent {agent.id} dimensions must be finite and positive"
            )
        for field_name in _FRAME_FIELDS:
            values = getattr(agent, field_name)
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"agent {agent.id} field {field_name} must be finite"
                )
        if np.any(agent.heading < -np.pi) or np.any(agent.heading > np.pi):
            raise ValueError(
                f"agent {agent.id} headings must be in [-pi, pi]"
            )

    start_index = _rollout_start_index(scenario)
    if future_steps is None:
        stop_index = scenario.num_steps - 1
    else:
        if (
            isinstance(future_steps, (bool, np.bool_))
            or not isinstance(future_steps, (int, np.integer))
            or int(future_steps) < 1
        ):
            raise ValueError("future_steps must be a positive integer")
        stop_index = start_index + int(future_steps)
        if stop_index >= scenario.num_steps:
            raise ValueError(
                "scenario does not contain the required typed-plan horizon"
            )
    if not np.all(scenario.ego.valid[start_index : stop_index + 1]):
        raise ValueError(
            "ego must remain valid from rollout start through the execution horizon"
        )
    return start_index


def _validate_ego_plan(
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
    *,
    start_index: int,
) -> int:
    """Bind one immutable typed ego plan to an exact source prefix."""

    if not isinstance(plan, EgoTrajectoryPlan):
        raise TypeError("ego_plan must be an EgoTrajectoryPlan or None")
    plan.revalidate()
    # Import locally to keep the legacy engine import surface lightweight and avoid
    # a module-initialization cycle through ``evalsim.rollout``.
    from evalsim.perturb.m6 import validate_registered_ego_plan

    validate_registered_ego_plan(scenario, plan)
    stop_index = start_index + M6_ANALYSIS_TRANSITIONS
    plan_slice = slice(start_index, stop_index + 1)
    if not np.array_equal(plan.timestamps, scenario.timestamps[plan_slice]):
        raise ValueError("ego_plan timestamps must equal the exact source window")
    if not np.array_equal(plan.valid, scenario.ego.valid[plan_slice]):
        raise ValueError("ego_plan validity must equal the source ego lifecycle")
    for field_name in _FRAME_FIELDS:
        if not np.array_equal(
            np.asarray(getattr(plan, field_name))[0:1],
            np.asarray(getattr(scenario.ego, field_name))[start_index : start_index + 1],
        ):
            raise ValueError(
                "ego_plan current state must equal the exact source current state"
            )
    return stop_index


def _copy_frame_with_valid(frame: AgentFrame, valid: np.ndarray) -> AgentFrame:
    return AgentFrame(
        valid=valid,
        x=frame.x,
        y=frame.y,
        heading=frame.heading,
        vx=frame.vx,
        vy=frame.vy,
    )


def _snapshot_policy_step(step: PolicyStep) -> PolicyStep:
    """Revalidate a returned step in case frozen fields were bypassed."""

    override = step.override
    if override is not None:
        if not isinstance(override, AgentFrame):
            raise TypeError("PolicyStep.override must be AgentFrame or None")
        override = _copy_frame_with_valid(override, override.valid)
    return PolicyStep(
        next_state=step.next_state,
        longitudinal_acceleration=step.longitudinal_acceleration,
        yaw_rate=step.yaw_rate,
        override=override,
        override_mask=step.override_mask,
    )


def _apply_absolute_values(
    destination: dict[str, np.ndarray],
    source: AgentFrame,
    mask: np.ndarray,
) -> None:
    for field_name in _FRAME_FIELDS:
        destination[field_name][mask] = getattr(source, field_name)[mask]


def _next_frame(
    *,
    current: AgentFrame,
    reference_next: AgentFrame,
    step: PolicyStep,
    current_index: int,
    next_index: int,
    dt: float,
    next_valid: np.ndarray,
    ego_index: int,
    ego_plan: EgoTrajectoryPlan | None,
    ego_plan_next_index: int | None,
    limits: DynamicsLimits,
) -> tuple[AgentFrame, ClampCounts]:
    count = current.num_agents
    if step.num_agents != count:
        raise ValueError(
            f"policy returned {step.num_agents} agents at transition "
            f"{current_index}->{next_index}; expected {count}"
        )

    world_mask = np.ones(count, dtype=bool)
    world_mask[ego_index] = False
    override_mask = np.asarray(step.override_mask, dtype=bool) & world_mask
    if step.override is not None and np.any(
        step.override.valid[override_mask]
        != next_valid[override_mask]
    ):
        raise ValueError(
            "policy absolute override validity must match scenario lifecycle mask"
        )

    continuing = (
        current.valid
        & next_valid
        & world_mask
    )
    kinematic_mask = continuing & ~override_mask
    dynamics = integrate_point_mass(
        current,
        step.longitudinal_acceleration,
        step.yaw_rate,
        dt,
        limits=limits,
        update_mask=kinematic_mask,
    )
    values = {
        field_name: np.array(
            getattr(dynamics.frame, field_name),
            copy=True,
        )
        for field_name in _FRAME_FIELDS
    }

    if step.override is not None:
        _apply_absolute_values(values, step.override, override_mask)

    # Every contiguous valid segment begins from an observed state. This prevents
    # hidden invalid payloads from affecting controls or bridging lifecycle gaps.
    births = (~current.valid) & next_valid & world_mask
    _apply_absolute_values(values, reference_next, births)

    # Ego remains engine-owned. The legacy path copies the logged source; the typed
    # path applies only the already-validated ego plan after the world transition.
    if ego_plan is None:
        ego_mask = np.zeros(count, dtype=bool)
        ego_mask[ego_index] = True
        _apply_absolute_values(values, reference_next, ego_mask)
    else:
        if ego_plan_next_index is None or not (
            1 <= ego_plan_next_index < ego_plan.frame_count
        ):
            raise ValueError("typed ego plan index is outside the execution window")
        for field_name in _FRAME_FIELDS:
            values[field_name][ego_index] = getattr(
                ego_plan,
                field_name,
            )[ego_plan_next_index]

    return (
        AgentFrame(
            valid=next_valid,
            **values,
        ),
        dynamics.clamp_counts,
    )


def _write_frame(
    history: dict[str, np.ndarray],
    frame: AgentFrame,
    step_index: int,
) -> None:
    for field_name in _FRAME_FIELDS:
        history[field_name][:, step_index] = getattr(frame, field_name)


def _policy_control_modes(
    scenario: Scenario,
    metadata: PolicyMetadata,
    *,
    typed_ego_plan: bool,
) -> dict[str, str]:
    supported = set(metadata.supported_agent_types)
    modes: dict[str, str] = {}
    for index, agent in enumerate(scenario.agents):
        if index == scenario.ego_index:
            mode = "typed_ego_plan" if typed_ego_plan else "logged_ego"
        elif agent.type in supported:
            mode = metadata.name
        elif metadata.fallback_policy is not None:
            mode = metadata.fallback_policy
        else:
            raise ValueError(
                f"policy {metadata.name!r} does not support agent type "
                f"{agent.type.value!r} and declares no fallback"
            )
        modes[str(agent.id)] = mode
    return modes


class _HistoryTracePolicy(HistoryOnlySimulatorPolicy):
    """Ephemeral engine-owned recorder around one history-only policy run."""

    def __init__(
        self,
        delegate: HistoryOnlySimulatorPolicy,
        steps: list[PolicyStep],
    ) -> None:
        self._delegate = delegate
        self._steps = steps

    def initialize(
        self,
        context: HistoryOnlyPolicyContext,
        seed: int,
    ) -> Any:
        return self._delegate.initialize(context, seed)

    def step(
        self,
        state: Any,
        observation: HistoryOnlyPolicyObservation,
    ) -> PolicyStep:
        step = self._delegate.step(state, observation)
        if not isinstance(step, PolicyStep):
            return step
        snapshot = _snapshot_policy_step(step)
        self._steps.append(snapshot)
        return snapshot

    def metadata(self) -> PolicyMetadata:
        return self._delegate.metadata()


class _PrivilegedTracePolicy(PrivilegedSimulatorPolicy):
    """Ephemeral engine-owned recorder around one privileged policy run."""

    def __init__(
        self,
        delegate: PrivilegedSimulatorPolicy,
        steps: list[PolicyStep],
    ) -> None:
        self._delegate = delegate
        self._steps = steps

    def initialize(
        self,
        context: PrivilegedPolicyContext,
        seed: int,
    ) -> Any:
        return self._delegate.initialize(context, seed)

    def step(
        self,
        state: Any,
        observation: HistoryOnlyPolicyObservation,
    ) -> PolicyStep:
        step = self._delegate.step(state, observation)
        if not isinstance(step, PolicyStep):
            return step
        snapshot = _snapshot_policy_step(step)
        self._steps.append(snapshot)
        return snapshot

    def metadata(self) -> PolicyMetadata:
        return self._delegate.metadata()


def _build_policy_execution_trace(
    source: Scenario,
    rollout: Rollout,
    steps: list[PolicyStep],
    *,
    policy_access_role: str,
) -> PolicyExecutionTrace:
    start = int(rollout.metadata["rollout_start_index"])
    stop = rollout.num_steps - 1
    transition_count = stop - start
    if len(steps) != transition_count:
        raise RuntimeError("policy trace step count differs from rollout horizon")
    count = source.num_agents
    shape = (transition_count, count)
    acceleration = np.empty(shape, dtype=np.float64)
    yaw_rate = np.empty(shape, dtype=np.float64)
    override_mask = np.zeros(shape, dtype=bool)
    override_valid = np.zeros(shape, dtype=bool)
    override_values = {
        name: np.zeros(shape, dtype=np.float64)
        for name in _FRAME_FIELDS
    }
    effective = np.zeros(shape, dtype=bool)
    births = np.zeros(shape, dtype=bool)
    world_mask = np.ones(count, dtype=bool)
    world_mask[source.ego_index] = False

    for offset, step in enumerate(steps):
        current_index = start + offset
        next_index = current_index + 1
        current_valid = np.asarray(
            [agent.valid[current_index] for agent in source.agents],
            dtype=bool,
        )
        next_valid = np.asarray(
            [agent.valid[next_index] for agent in source.agents],
            dtype=bool,
        )
        continuing = current_valid & next_valid & world_mask
        birth_mask = ~current_valid & next_valid & world_mask
        submitted_override = (
            np.asarray(step.override_mask, dtype=bool)
            & world_mask
        )
        acceleration[offset] = step.longitudinal_acceleration
        yaw_rate[offset] = step.yaw_rate
        effective[offset] = continuing
        births[offset] = birth_mask
        override_mask[offset] = submitted_override
        if step.override is not None:
            override_valid[offset, submitted_override] = step.override.valid[
                submitted_override
            ]
            for field_name in _FRAME_FIELDS:
                override_values[field_name][offset, submitted_override] = getattr(
                    step.override,
                    field_name,
                )[submitted_override]

    modes = rollout.metadata.get("agent_control_modes")
    if not isinstance(modes, dict):
        raise RuntimeError("rollout is missing agent control modes")
    policy = rollout.metadata.get("policy")
    if not isinstance(policy, dict):
        raise RuntimeError("rollout is missing policy metadata")
    return PolicyExecutionTrace(
        policy_name=str(policy["name"]),
        policy_version=str(policy["version"]),
        policy_access_role=policy_access_role,
        start_index=start,
        stop_index=stop,
        ego_index=source.ego_index,
        timestamps=np.array(
            source.timestamps[start : stop + 1],
            copy=True,
        ),
        control_modes=tuple(
            str(modes[str(agent.id)]) for agent in source.agents
        ),
        longitudinal_acceleration=acceleration,
        yaw_rate=yaw_rate,
        override_mask=override_mask,
        override_valid=override_valid,
        override_x=override_values["x"],
        override_y=override_values["y"],
        override_heading=override_values["heading"],
        override_vx=override_values["vx"],
        override_vy=override_values["vy"],
        effective_control_mask=effective,
        lifecycle_birth_mask=births,
        perturbation_identity=rollout.perturbation,
        rollout_fingerprint=(
            RolloutSnapshot.from_rollout(rollout)._integrity_fingerprint
        ),
    )


@dataclass(frozen=True, slots=True)
class RolloutEngine:
    """Run a policy over a scenario without retaining cross-run state."""

    dynamics_limits: DynamicsLimits = field(default_factory=DynamicsLimits)
    name: str = ROLLOUT_ENGINE_NAME
    version: str = ROLLOUT_ENGINE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.dynamics_limits, DynamicsLimits):
            raise TypeError("dynamics_limits must be DynamicsLimits")
        for field_name in ("name", "version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"RolloutEngine.{field_name} must be non-empty")
            object.__setattr__(self, field_name, value.strip())

    def run(
        self,
        scenario: Scenario,
        policy: SimulatorPolicy,
        seed: int = 0,
        perturbation: str | None = None,
        *,
        ego_plan: EgoTrajectoryPlan | None = None,
    ) -> Rollout:
        """Run one deterministic legacy or typed-plan rollout.

        ``ego_plan=None`` preserves the full-length M2/M5 path. A typed M6 plan runs
        the exact current-plus-40-transition source prefix. Free-text perturbation
        labels remain forbidden because they could claim an unapplied intervention.
        """

        if perturbation is not None:
            raise ValueError(
                "free-text perturbation labels are forbidden; use ego_plan"
            )
        seed = _validate_seed(seed)
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        # All validation, registered-plan recompilation, policy initialization, and
        # execution use this one defensive snapshot. The mutable caller-owned source
        # is never re-read after this point.
        reference = copy.deepcopy(scenario)
        if ego_plan is not None and not isinstance(ego_plan, EgoTrajectoryPlan):
            raise TypeError("ego_plan must be an EgoTrajectoryPlan or None")
        start_index = _validate_scenario_for_rollout(
            reference,
            future_steps=(
                M6_ANALYSIS_TRANSITIONS if ego_plan is not None else None
            ),
        )
        if ego_plan is None:
            stop_index = reference.num_steps - 1
        else:
            stop_index = _validate_ego_plan(
                reference,
                ego_plan,
                start_index=start_index,
            )
        if not isinstance(policy, SimulatorPolicy):
            raise TypeError("policy must implement SimulatorPolicy")
        history_only = isinstance(policy, HistoryOnlySimulatorPolicy)
        privileged = isinstance(policy, PrivilegedSimulatorPolicy)
        if history_only == privileged:
            raise TypeError(
                "policy must implement exactly one of "
                "HistoryOnlySimulatorPolicy or PrivilegedSimulatorPolicy"
            )
        metadata = policy.metadata()
        if not isinstance(metadata, PolicyMetadata):
            raise TypeError("policy.metadata() must return PolicyMetadata")
        # Snapshot provenance before policy initialization/steps. Policy metadata is
        # recursively immutable, and this detached copy cannot be rewritten mid-run.
        policy_snapshot = metadata.to_dict()
        policy_name = str(policy_snapshot["name"])
        policy_version = str(policy_snapshot["version"])
        control_modes = _policy_control_modes(
            reference,
            metadata,
            typed_ego_plan=ego_plan is not None,
        )

        # The engine and policy receive separate immutable snapshots. A buggy policy
        # therefore cannot mutate the caller's Scenario or the engine's private
        # lifecycle/reference data.
        if history_only:
            policy_context = HistoryOnlyPolicyContext.from_scenario(reference)
        else:
            policy_context = PrivilegedPolicyContext.from_scenario(reference)
        try:
            policy_state = policy.initialize(policy_context, seed)
        except Exception as exc:
            raise RuntimeError(
                f"policy {policy_name!r} failed during initialization"
            ) from exc

        count = reference.num_agents
        horizon = stop_index + 1
        history = {
            field_name: np.zeros((count, horizon), dtype=float)
            for field_name in _FRAME_FIELDS
        }
        validity = np.stack(
            [
                np.array(agent.valid[:horizon], copy=True)
                for agent in reference.agents
            ]
        )

        # Logged context through current_index is observational history, not rollout.
        for step_index in range(start_index + 1):
            _write_frame(
                history,
                AgentFrame.from_scenario(reference, step_index),
                step_index,
            )

        current = AgentFrame.from_scenario(reference, start_index)
        clamp_counts = ClampCounts()
        ids = tuple(int(agent.id) for agent in reference.agents)
        types = tuple(agent.type for agent in reference.agents)
        lengths = np.array([agent.length for agent in reference.agents])
        widths = np.array([agent.width for agent in reference.agents])

        for current_index in range(start_index, horizon - 1):
            next_index = current_index + 1
            reference_next = AgentFrame.from_scenario(reference, next_index)
            transition_dt = float(
                reference.timestamps[next_index]
                - reference.timestamps[current_index]
            )
            engine_next_valid = validity[:, next_index]
            observation = HistoryOnlyPolicyObservation(
                current_index=current_index,
                next_index=next_index,
                timestamp=reference.timestamps[current_index],
                next_timestamp=reference.timestamps[next_index],
                dt=transition_dt,
                # Never expose the engine-owned frame object to plugin code.
                frame=_copy_frame_with_valid(current, current.valid),
                agent_ids=ids,
                agent_types=types,
                lengths=lengths,
                widths=widths,
                ego_index=reference.ego_index,
            )
            try:
                policy_step = policy.step(policy_state, observation)
            except Exception as exc:
                raise RuntimeError(
                    f"policy {policy_name!r} failed at transition "
                    f"{current_index}->{next_index}"
                ) from exc
            if not isinstance(policy_step, PolicyStep):
                raise TypeError(
                    f"policy {policy_name!r} returned "
                    f"{type(policy_step).__name__} at transition "
                    f"{current_index}->{next_index}; expected PolicyStep"
                )
            try:
                policy_step = _snapshot_policy_step(policy_step)
                if history_only and np.any(policy_step.override_mask):
                    raise ValueError(
                        "history-only policies cannot return absolute-state overrides"
                    )
                current, transition_clamps = _next_frame(
                    current=current,
                    reference_next=reference_next,
                    step=policy_step,
                    current_index=current_index,
                    next_index=next_index,
                    dt=transition_dt,
                    next_valid=engine_next_valid,
                    ego_index=reference.ego_index,
                    ego_plan=ego_plan,
                    ego_plan_next_index=(
                        None
                        if ego_plan is None
                        else next_index - start_index
                    ),
                    limits=self.dynamics_limits,
                )
            except Exception as exc:
                raise ValueError(
                    f"invalid output from policy {policy_name!r} at transition "
                    f"{current_index}->{next_index}: {exc}"
                ) from exc
            clamp_counts += transition_clamps
            policy_state = policy_step.next_state
            _write_frame(history, current, next_index)

        agents = [
            Agent(
                id=int(reference_agent.id),
                type=reference_agent.type,
                valid=validity[index],
                x=history["x"][index],
                y=history["y"][index],
                heading=history["heading"][index],
                vx=history["vx"][index],
                vy=history["vy"][index],
                length=float(reference_agent.length),
                width=float(reference_agent.width),
            )
            for index, reference_agent in enumerate(reference.agents)
        ]
        rollout_metadata: dict[str, Any] = {
            "engine": {
                "name": self.name,
                "version": self.version,
            },
            "dynamics": {
                "name": DYNAMICS_NAME,
                "version": DYNAMICS_VERSION,
                "integration": "midpoint_heading_trapezoidal_speed",
                "limits": self.dynamics_limits.to_dict(),
                "clamp_counts": clamp_counts.to_dict(),
            },
            "policy": policy_snapshot,
            "ego_control": (
                "logged" if ego_plan is None else "typed_ego_plan"
            ),
            "rollout_start_index": start_index,
            "controlled_agent_ids": [
                int(agent.id)
                for index, agent in enumerate(reference.agents)
                if index != reference.ego_index
            ],
            "agent_control_modes": control_modes,
            "scenario_source": reference.metadata.get("source", "unknown"),
            "scenario_source_fingerprint": reference.metadata.get(
                "source_fingerprint"
            ),
        }
        return Rollout(
            scenario_id=reference.scenario_id,
            sim_name=policy_name,
            sim_version=policy_version,
            seed=seed,
            timestamps=np.array(reference.timestamps[:horizon], copy=True),
            agents=agents,
            perturbation=(
                None if ego_plan is None else ego_plan.perturbation_identity
            ),
            metadata=rollout_metadata,
        )

    def run_with_trace(
        self,
        scenario: Scenario,
        policy: SimulatorPolicy,
        seed: int = 0,
        *,
        ego_plan: EgoTrajectoryPlan | None = None,
    ) -> TracedRollout:
        """Run once and return an immutable M6 sidecar without metadata drift."""

        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        history_only = isinstance(policy, HistoryOnlySimulatorPolicy)
        privileged = isinstance(policy, PrivilegedSimulatorPolicy)
        if history_only == privileged:
            raise TypeError(
                "policy must implement exactly one of "
                "HistoryOnlySimulatorPolicy or PrivilegedSimulatorPolicy"
            )
        # This snapshot is never exposed to the policy. It is the exact source used
        # to reconstruct engine-owned lifecycle/effective-control categories.
        source = copy.deepcopy(scenario)
        steps: list[PolicyStep] = []
        if history_only:
            assert isinstance(policy, HistoryOnlySimulatorPolicy)
            traced_policy: SimulatorPolicy = _HistoryTracePolicy(policy, steps)
            access_role = "history_only"
        else:
            assert isinstance(policy, PrivilegedSimulatorPolicy)
            traced_policy = _PrivilegedTracePolicy(policy, steps)
            access_role = "privileged"
        rollout = self.run(
            source,
            traced_policy,
            seed=seed,
            ego_plan=ego_plan,
        )
        trace = _build_policy_execution_trace(
            source,
            rollout,
            steps,
            policy_access_role=access_role,
        )
        return TracedRollout(rollout=rollout, trace=trace)


__all__ = [
    "ROLLOUT_ENGINE_NAME",
    "ROLLOUT_ENGINE_VERSION",
    "RolloutEngine",
]
