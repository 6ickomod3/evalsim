"""Stateless, deterministic closed-loop rollout orchestration."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evalsim.contracts import (
    Agent,
    AgentFrame,
    PolicyMetadata,
    PolicyObservation,
    PolicyStep,
    Rollout,
    Scenario,
    SimulatorPolicy,
)

from .dynamics import (
    DYNAMICS_NAME,
    DYNAMICS_VERSION,
    ClampCounts,
    DynamicsLimits,
    integrate_point_mass,
)

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


def _validate_scenario_for_rollout(scenario: Scenario) -> int:
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
    if not np.all(scenario.ego.valid[start_index:]):
        raise ValueError(
            "ego must remain valid from rollout start through the horizon"
        )
    return start_index


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

    # M2 treats ego as exogenous. Policies observe it but cannot command it.
    ego_mask = np.zeros(count, dtype=bool)
    ego_mask[ego_index] = True
    _apply_absolute_values(values, reference_next, ego_mask)

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
) -> dict[str, str]:
    supported = set(metadata.supported_agent_types)
    modes: dict[str, str] = {}
    for index, agent in enumerate(scenario.agents):
        if index == scenario.ego_index:
            mode = "logged_ego"
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
    ) -> Rollout:
        """Run one deterministic M2 rollout.

        Non-``None`` perturbations are rejected until M5 supplies a typed ego-control
        protocol. Silently attaching an unapplied perturbation label would corrupt
        provenance.
        """

        if perturbation is not None:
            raise ValueError(
                "counterfactual perturbations are not implemented until M5"
            )
        seed = _validate_seed(seed)
        start_index = _validate_scenario_for_rollout(scenario)
        if not isinstance(policy, SimulatorPolicy):
            raise TypeError("policy must implement SimulatorPolicy")
        metadata = policy.metadata()
        if not isinstance(metadata, PolicyMetadata):
            raise TypeError("policy.metadata() must return PolicyMetadata")
        # Snapshot provenance before policy initialization/steps. Policy metadata is
        # recursively immutable, and this detached copy cannot be rewritten mid-run.
        policy_snapshot = metadata.to_dict()
        policy_name = str(policy_snapshot["name"])
        policy_version = str(policy_snapshot["version"])
        control_modes = _policy_control_modes(scenario, metadata)

        # The engine and policy receive separate copies. A buggy policy therefore
        # cannot mutate the caller's Scenario or the engine's lifecycle/reference data.
        reference = copy.deepcopy(scenario)
        policy_scenario = copy.deepcopy(reference)
        try:
            policy_state = policy.initialize(policy_scenario, seed)
        except Exception as exc:
            raise RuntimeError(
                f"policy {policy_name!r} failed during initialization"
            ) from exc

        count = reference.num_agents
        horizon = reference.num_steps
        history = {
            field_name: np.zeros((count, horizon), dtype=float)
            for field_name in _FRAME_FIELDS
        }
        validity = np.stack(
            [np.array(agent.valid, copy=True) for agent in reference.agents]
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
            observation = PolicyObservation(
                current_index=current_index,
                next_index=next_index,
                timestamp=reference.timestamps[current_index],
                next_timestamp=reference.timestamps[next_index],
                dt=transition_dt,
                # Never expose the engine-owned frame object to plugin code.
                frame=_copy_frame_with_valid(current, current.valid),
                next_valid=engine_next_valid,
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
                current, transition_clamps = _next_frame(
                    current=current,
                    reference_next=reference_next,
                    step=policy_step,
                    current_index=current_index,
                    next_index=next_index,
                    dt=transition_dt,
                    next_valid=engine_next_valid,
                    ego_index=reference.ego_index,
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
            "ego_control": "logged",
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
            timestamps=np.array(reference.timestamps, copy=True),
            agents=agents,
            perturbation=None,
            metadata=rollout_metadata,
        )


__all__ = [
    "ROLLOUT_ENGINE_NAME",
    "ROLLOUT_ENGINE_VERSION",
    "RolloutEngine",
]
