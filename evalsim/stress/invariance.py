"""M7 invariance probes: verify metrics are unchanged by semantics-preserving transforms.

A release-candidate evaluator must be invariant to reordering world agents and to a rigid
translation of the whole scene. This harness applies such probes consistently to both the
scenario and the rollout, then measures the per-metric value delta. It also exposes a
deliberately semantics-breaking probe (translating only the rollout) so tests can prove the
harness actually detects violations rather than rubber-stamping them.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from evalsim.contracts.metric import Metric
from evalsim.contracts.rollout import Rollout
from evalsim.contracts.scenario import Agent, MapPolyline, Scenario
from evalsim.metrics.registry import MetricRegistry

_TOL = 1e-6


@runtime_checkable
class InvarianceProbe(Protocol):
    name: str

    def apply(
        self, scenario: Scenario, rollout: Rollout, *, seed: int
    ) -> tuple[Scenario, Rollout]: ...


@dataclass(frozen=True)
class InvarianceResult:
    metric_name: str
    probe_name: str
    baseline: float | None
    transformed: float | None
    delta: float
    invariant: bool


def _copy_agent(agent: Agent, *, dx: float = 0.0, dy: float = 0.0) -> Agent:
    return Agent(
        id=agent.id,
        type=agent.type,
        valid=np.array(agent.valid, copy=True),
        x=np.array(agent.x, copy=True) + dx,
        y=np.array(agent.y, copy=True) + dy,
        heading=np.array(agent.heading, copy=True),
        vx=np.array(agent.vx, copy=True),
        vy=np.array(agent.vy, copy=True),
        length=agent.length,
        width=agent.width,
    )


def _copy_map_feature(
    feature: MapPolyline,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> MapPolyline:
    offset = np.asarray([dx, dy], dtype=float)
    return MapPolyline(
        type=feature.type,
        xy=np.array(feature.xy, copy=True) + offset,
    )


def _new_scenario(
    scenario: Scenario,
    agents: list[Agent],
    *,
    map_features: Sequence[MapPolyline] | None = None,
) -> Scenario:
    source_map = scenario.map if map_features is None else map_features
    return Scenario(
        scenario_id=scenario.scenario_id,
        timestamps=np.array(scenario.timestamps, copy=True),
        agents=agents,
        map=[_copy_map_feature(feature) for feature in source_map],
        ego_index=scenario.ego_index,
        metadata=dict(scenario.metadata),
    )


def _new_rollout(rollout: Rollout, agents: list[Agent]) -> Rollout:
    return Rollout(
        scenario_id=rollout.scenario_id,
        sim_name=rollout.sim_name,
        sim_version=rollout.sim_version,
        seed=rollout.seed,
        timestamps=np.array(rollout.timestamps, copy=True),
        agents=agents,
        perturbation=rollout.perturbation,
        metadata=dict(rollout.metadata),
    )


class AgentPermutationProbe:
    """Reorder non-ego world agents identically in the scenario and rollout."""

    name = "agent_permutation"

    def apply(
        self, scenario: Scenario, rollout: Rollout, *, seed: int
    ) -> tuple[Scenario, Rollout]:
        ego = scenario.ego_index
        others = [p for p in range(len(scenario.agents)) if p != ego]
        perm = np.random.default_rng(seed).permutation(len(others))
        new_order = [others[i] for i in perm]

        source_for_slot = {ego: ego}
        for slot, source in zip(others, new_order, strict=True):
            source_for_slot[slot] = source

        def reorder(agents: Sequence[Agent]) -> list[Agent]:
            return [
                _copy_agent(agents[source_for_slot[slot]])
                for slot in range(len(agents))
            ]

        return (
            _new_scenario(scenario, reorder(scenario.agents)),
            _new_rollout(rollout, reorder(rollout.agents)),
        )


class TranslationProbe:
    """Rigidly translate the whole scene (scenario + rollout) by (dx, dy)."""

    def __init__(self, dx: float, dy: float) -> None:
        self.dx = float(dx)
        self.dy = float(dy)
        self.name = "translation"

    def apply(
        self, scenario: Scenario, rollout: Rollout, *, seed: int
    ) -> tuple[Scenario, Rollout]:
        del seed
        scn = [_copy_agent(a, dx=self.dx, dy=self.dy) for a in scenario.agents]
        rol = [_copy_agent(a, dx=self.dx, dy=self.dy) for a in rollout.agents]
        map_features = [
            _copy_map_feature(feature, dx=self.dx, dy=self.dy)
            for feature in scenario.map
        ]
        return (
            _new_scenario(scenario, scn, map_features=map_features),
            _new_rollout(rollout, rol),
        )


class RolloutOnlyTranslationProbe:
    """Semantics-BREAKING control: translate only the rollout (scenario unchanged).

    Used to prove the harness detects genuine non-invariance; not a valid probe.
    """

    def __init__(self, dx: float, dy: float) -> None:
        self.dx = float(dx)
        self.dy = float(dy)
        self.name = "rollout_only_translation"

    def apply(
        self, scenario: Scenario, rollout: Rollout, *, seed: int
    ) -> tuple[Scenario, Rollout]:
        del seed
        scn = [_copy_agent(a) for a in scenario.agents]
        rol = [_copy_agent(a, dx=self.dx, dy=self.dy) for a in rollout.agents]
        return _new_scenario(scenario, scn), _new_rollout(rollout, rol)


class RolloutOnlyVelocityImpulseProbe:
    """Semantics-BREAKING control: spike one world agent's rollout velocity.

    The first non-ego agent receives a one-frame ``vx`` impulse at the registered
    mid-future frame. The scenario and every other rollout field are defensively copied
    without modification so this control isolates the kinematic metric's velocity path.
    """

    def __init__(self, impulse_mps: float = 100.0) -> None:
        self.impulse_mps = float(impulse_mps)
        if not np.isfinite(self.impulse_mps):
            raise ValueError("impulse_mps must be finite")
        self.name = "rollout_only_velocity_impulse"

    def apply(
        self, scenario: Scenario, rollout: Rollout, *, seed: int
    ) -> tuple[Scenario, Rollout]:
        del seed
        current = scenario.metadata.get("current_index", 0)
        if (
            isinstance(current, (bool, np.bool_))
            or not isinstance(current, (int, np.integer))
            or not 0 <= int(current) < scenario.num_steps
        ):
            raise ValueError("current_index must index the scenario horizon")
        current = int(current)

        scn = [_copy_agent(agent) for agent in scenario.agents]
        rol = [_copy_agent(agent) for agent in rollout.agents]
        world_positions = [
            position
            for position in range(len(scenario.agents))
            if position != scenario.ego_index
        ]
        if world_positions and current + 1 < rollout.num_steps:
            impulse_frame = min(
                current + max(1, (rollout.num_steps - 1 - current) // 2),
                rollout.num_steps - 1,
            )
            rol[world_positions[0]].vx[impulse_frame] += self.impulse_mps
        return _new_scenario(scenario, scn), _new_rollout(rollout, rol)


def _value(metric: Metric, scenario: Scenario, rollout: Rollout) -> float | None:
    result = MetricRegistry([metric]).evaluate(scenario, rollout)[0]
    return float(result.value) if result.valid else None


def check_invariance(
    metric: Metric,
    scenario: Scenario,
    rollout: Rollout,
    probe: InvarianceProbe,
    *,
    seed: int,
    tol: float = _TOL,
) -> InvarianceResult:
    baseline = _value(metric, scenario, rollout)
    t_scenario, t_rollout = probe.apply(scenario, rollout, seed=seed)
    transformed = _value(metric, t_scenario, t_rollout)
    if baseline is None or transformed is None:
        delta = float("inf")
    else:
        delta = abs(baseline - transformed)
    return InvarianceResult(
        metric_name=metric.spec.name,
        probe_name=probe.name,
        baseline=baseline,
        transformed=transformed,
        delta=delta,
        invariant=delta <= tol,
    )


def invariance_matrix(
    metrics: Iterable[Metric],
    cases: Sequence[tuple[Scenario, Rollout]],
    probes: Iterable[InvarianceProbe],
    *,
    seed: int,
    tol: float = _TOL,
) -> tuple[InvarianceResult, ...]:
    """Check every (metric x probe) pair across all cases; one result per case."""
    metric_list = list(metrics)
    probe_list = list(probes)
    results: list[InvarianceResult] = []
    for scenario, rollout in cases:
        for probe in probe_list:
            for metric in metric_list:
                results.append(
                    check_invariance(metric, scenario, rollout, probe, seed=seed, tol=tol)
                )
    return tuple(results)


__all__ = [
    "AgentPermutationProbe",
    "InvarianceProbe",
    "InvarianceResult",
    "RolloutOnlyTranslationProbe",
    "RolloutOnlyVelocityImpulseProbe",
    "TranslationProbe",
    "check_invariance",
    "invariance_matrix",
]
