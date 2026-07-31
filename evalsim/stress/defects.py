"""M7 evaluator red-team: typed, severity-controlled defect generators.

A *defect* is a pure, deterministic transform of a simulated ``Rollout`` that injects
a known corruption of declared severity, so the M5 evaluators can be stress-tested for
detection sensitivity, blind spots, and false positives. Every generator satisfies a
strict severity-0 identity and emits a sanitized :class:`DefectManifest` carrying only
cohort-relative accounting -- never native agent ids, coordinates, or source payloads.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from evalsim.contracts.rollout import Rollout
from evalsim.contracts.scenario import Agent, Scenario

_FAMILY = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^v[0-9]+$")
_TARGETS = frozenset({"rollout", "scenario"})


class DefectRegistryError(ValueError):
    """A defect registration or generator-contract violation."""


@dataclass(frozen=True)
class DefectSpec:
    """Immutable declaration of one severity-controlled defect family."""

    family: str
    version: str
    severity_min: float
    severity_max: float
    severity_unit: str
    target: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or _FAMILY.fullmatch(self.family) is None:
            raise ValueError("defect family must be lowercase snake_case")
        if not isinstance(self.version, str) or _VERSION.fullmatch(self.version) is None:
            raise ValueError("defect version must have the form vN")
        lo = float(self.severity_min)
        hi = float(self.severity_max)
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            raise ValueError("severity_min must be finite and <= severity_max")
        if not isinstance(self.severity_unit, str) or not self.severity_unit.strip():
            raise ValueError("severity_unit must be a non-empty string")
        if self.target not in _TARGETS:
            raise ValueError(f"target must be one of {sorted(_TARGETS)}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")

    def clamp(self, severity: float) -> float:
        value = float(severity)
        if not math.isfinite(value):
            raise ValueError("severity must be finite")
        if value < self.severity_min or value > self.severity_max:
            raise ValueError(
                f"severity {value} outside [{self.severity_min}, {self.severity_max}]"
            )
        return value


@dataclass(frozen=True)
class DefectManifest:
    """Sanitized accounting for one applied defect (no native id/coordinates)."""

    family: str
    version: str
    severity: float
    seed: int
    total_world_agent_count: int
    affected_agent_ordinals: tuple[int, ...]

    @property
    def affected_agent_count(self) -> int:
        return len(self.affected_agent_ordinals)


@runtime_checkable
class Defect(Protocol):
    """A severity-controlled, deterministic rollout corruption."""

    spec: DefectSpec

    def apply(
        self,
        scenario: Scenario,
        rollout: Rollout,
        severity: float,
        *,
        seed: int,
    ) -> tuple[Rollout, DefectManifest]: ...


def _current_index(scenario: Scenario) -> int:
    value = scenario.metadata.get("current_index", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("scenario current_index must be a non-negative integer")
    if value >= scenario.num_steps:
        raise ValueError("scenario current_index must be < num_steps")
    return value


def _world_positions(scenario: Scenario, rollout: Rollout) -> tuple[int, ...]:
    """Absolute rollout array positions of non-ego agents, in order.

    The k-th entry is the array index of the world agent with 0-based cohort rank k;
    manifests record the rank, never the array position or the native agent id.
    """
    ego = scenario.ego_index
    return tuple(pos for pos in range(len(rollout.agents)) if pos != ego)


def _copy_agent(agent: Agent) -> Agent:
    return Agent(
        id=agent.id,
        type=agent.type,
        valid=np.array(agent.valid, copy=True),
        x=np.array(agent.x, copy=True),
        y=np.array(agent.y, copy=True),
        heading=np.array(agent.heading, copy=True),
        vx=np.array(agent.vx, copy=True),
        vy=np.array(agent.vy, copy=True),
        length=agent.length,
        width=agent.width,
    )


def _rebuild(rollout: Rollout, agents: list[Agent]) -> Rollout:
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


class FrozenAgentDefect:
    """Freeze a nested, severity-controlled fraction of world agents.

    severity in [0, 1] is the fraction of non-ego agents held at their current-index
    state (position held, velocity zeroed) from ``current_index`` onward. Selection is
    the first ``ceil(severity * n)`` world agents by cohort ordinal, so higher severity
    freezes a strict superset -- a monotone, oracle-friendly corruption.
    """

    spec = DefectSpec(
        family="frozen_agent",
        version="v1",
        severity_min=0.0,
        severity_max=1.0,
        severity_unit="fraction_of_world_agents",
        target="rollout",
        description="hold a nested fraction of world agents static from current_index",
    )

    def apply(
        self,
        scenario: Scenario,
        rollout: Rollout,
        severity: float,
        *,
        seed: int,
    ) -> tuple[Rollout, DefectManifest]:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an int")
        severity = self.spec.clamp(severity)
        current = _current_index(scenario)
        positions = _world_positions(scenario, rollout)
        n_world = len(positions)
        count = math.ceil(severity * n_world)
        agents = [_copy_agent(agent) for agent in rollout.agents]
        for position in positions[:count]:
            agent = agents[position]
            for name in ("x", "y", "heading"):
                series = getattr(agent, name)
                series[current:] = series[current]
            agent.vx[current:] = 0.0
            agent.vy[current:] = 0.0
        manifest = DefectManifest(
            family=self.spec.family,
            version=self.spec.version,
            severity=severity,
            seed=seed,
            total_world_agent_count=n_world,
            affected_agent_ordinals=tuple(range(count)),
        )
        return _rebuild(rollout, agents), manifest


def _mid_future_frame(current: int, num_steps: int) -> int:
    """A future frame strictly after ``current`` (clamped to the last frame)."""
    return min(current + max(1, (num_steps - 1 - current) // 2), num_steps - 1)


class TeleportationDefect:
    """Displace a nested fraction of world agents by a large *position-only* step.

    This is a pure position discontinuity (taxonomy #2): the affected agents receive a
    persistent +50 m x-displacement from a mid-future frame, with velocities left
    untouched. It is detected by ``position_error_m`` (deviation from the logged future)
    and is deliberately a BLIND SPOT for ``waymax_kinematic_infeasibility_rate``, which
    reads stored velocities/headings and never finite-diffs position -- an honest negative
    result. For a velocity-domain corruption the metric *does* catch, see
    :class:`KinematicSpikeDefect`. severity in [0, 1] is the fraction of world agents,
    selected as a nested prefix by cohort rank; the seed is recorded for provenance but
    the selection is deterministic.
    """

    _JUMP_METRES = 50.0

    spec = DefectSpec(
        family="teleportation",
        version="v1",
        severity_min=0.0,
        severity_max=1.0,
        severity_unit="fraction_of_world_agents",
        target="rollout",
        description="position-only discontinuity: displace a nested fraction of world agents",
    )

    def apply(
        self,
        scenario: Scenario,
        rollout: Rollout,
        severity: float,
        *,
        seed: int,
    ) -> tuple[Rollout, DefectManifest]:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an int")
        severity = self.spec.clamp(severity)
        current = _current_index(scenario)
        positions = _world_positions(scenario, rollout)
        n_world = len(positions)
        count = math.ceil(severity * n_world)
        num_steps = len(rollout.timestamps)
        jump = _mid_future_frame(current, num_steps)
        applied = count if jump > current else 0  # no future frame -> honest no-op
        agents = [_copy_agent(agent) for agent in rollout.agents]
        for position in positions[:applied]:
            agents[position].x[jump:] = agents[position].x[jump:] + self._JUMP_METRES
        manifest = DefectManifest(
            family=self.spec.family,
            version=self.spec.version,
            severity=severity,
            seed=seed,
            total_world_agent_count=n_world,
            affected_agent_ordinals=tuple(range(applied)),
        )
        return _rebuild(rollout, agents), manifest


class KinematicSpikeDefect:
    """Inject a velocity impulse into a nested fraction of world agents (taxonomy #3).

    A pure velocity-domain corruption: the affected agents get a one-step +100 m/s vx
    spike at a mid-future frame, with positions left untouched. It is detected by
    ``waymax_kinematic_infeasibility_rate`` (the implied acceleration is infeasible) and
    is a BLIND SPOT for ``position_error_m`` (positions still match the log). Together
    with :class:`TeleportationDefect` this exposes the complementary blind spots of the
    position- and velocity-based evaluators. severity in [0, 1] is the fraction of world
    agents, a nested prefix by cohort rank; the seed is recorded but selection is
    deterministic.
    """

    _VELOCITY_SPIKE_MPS = 100.0

    spec = DefectSpec(
        family="kinematic_spike",
        version="v1",
        severity_min=0.0,
        severity_max=1.0,
        severity_unit="fraction_of_world_agents",
        target="rollout",
        description="velocity-only impulse: spike vx for a nested fraction of world agents",
    )

    def apply(
        self,
        scenario: Scenario,
        rollout: Rollout,
        severity: float,
        *,
        seed: int,
    ) -> tuple[Rollout, DefectManifest]:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an int")
        severity = self.spec.clamp(severity)
        current = _current_index(scenario)
        positions = _world_positions(scenario, rollout)
        n_world = len(positions)
        count = math.ceil(severity * n_world)
        num_steps = len(rollout.timestamps)
        jump = _mid_future_frame(current, num_steps)
        applied = count if jump > current else 0  # no future frame -> honest no-op
        agents = [_copy_agent(agent) for agent in rollout.agents]
        for position in positions[:applied]:
            agents[position].vx[jump] = (
                agents[position].vx[jump] + self._VELOCITY_SPIKE_MPS
            )
        manifest = DefectManifest(
            family=self.spec.family,
            version=self.spec.version,
            severity=severity,
            seed=seed,
            total_world_agent_count=n_world,
            affected_agent_ordinals=tuple(range(applied)),
        )
        return _rebuild(rollout, agents), manifest


class OverlapDefect:
    """Relocate a nested fraction of world agents onto a reference agent's future path.

    severity in [0, 1] is the fraction of the *non-reference* world agents moved to
    coincide with the first world agent from ``current_index + 1`` onward, forcing
    oriented-box overlap -- detected by the overlap-rate evaluator. Requires >= 2 world
    agents; with fewer it is a no-op identity.
    """

    spec = DefectSpec(
        family="overlap",
        version="v1",
        severity_min=0.0,
        severity_max=1.0,
        severity_unit="fraction_of_nonreference_world_agents",
        target="rollout",
        description="force a nested fraction of world agents to interpenetrate a reference",
    )

    def apply(
        self,
        scenario: Scenario,
        rollout: Rollout,
        severity: float,
        *,
        seed: int,
    ) -> tuple[Rollout, DefectManifest]:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an int")
        severity = self.spec.clamp(severity)
        current = _current_index(scenario)
        positions = _world_positions(scenario, rollout)
        n_world = len(positions)
        num_steps = len(rollout.timestamps)
        agents = [_copy_agent(agent) for agent in rollout.agents]
        affected_ranks: tuple[int, ...] = ()
        if n_world >= 2 and current + 1 < num_steps:  # needs >=2 agents and a future frame
            reference = agents[positions[0]]  # world rank 0 is the reference
            candidates = positions[1:]
            count = math.ceil(severity * len(candidates))
            affected_ranks = tuple(range(1, 1 + count))
            for position in candidates[:count]:
                agent = agents[position]
                for name in ("x", "y", "heading"):
                    getattr(agent, name)[current + 1:] = getattr(reference, name)[
                        current + 1:
                    ]
        manifest = DefectManifest(
            family=self.spec.family,
            version=self.spec.version,
            severity=severity,
            seed=seed,
            total_world_agent_count=n_world,
            affected_agent_ordinals=affected_ranks,
        )
        return _rebuild(rollout, agents), manifest


class DefectRegistry:
    """One unambiguous active version per defect family, iterated by sorted family."""

    def __init__(self, defects: Iterable[Defect] = ()) -> None:
        self._defects: dict[str, Defect] = {}
        for defect in defects:
            self.register(defect)

    def register(self, defect: Defect) -> None:
        if not isinstance(defect, Defect):
            raise DefectRegistryError(
                "registered object is not a Defect (needs spec + apply)"
            )
        family = defect.spec.family
        if family in self._defects:
            raise DefectRegistryError(f"defect family already registered: {family}")
        self._defects[family] = defect

    def get(self, family: str) -> Defect:
        try:
            return self._defects[family]
        except KeyError:
            raise DefectRegistryError(f"unknown defect family: {family}") from None

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._defects))

    def __len__(self) -> int:
        return len(self._defects)

    def __iter__(self) -> Iterator[Defect]:
        for family in sorted(self._defects):
            yield self._defects[family]


def default_defect_registry() -> DefectRegistry:
    """The registered M7 defect families (sorted: frozen_agent, kinematic_spike,
    overlap, teleportation)."""
    return DefectRegistry(
        [
            FrozenAgentDefect(),
            KinematicSpikeDefect(),
            OverlapDefect(),
            TeleportationDefect(),
        ]
    )


__all__ = [
    "Defect",
    "DefectManifest",
    "DefectRegistry",
    "DefectRegistryError",
    "DefectSpec",
    "FrozenAgentDefect",
    "KinematicSpikeDefect",
    "OverlapDefect",
    "TeleportationDefect",
    "default_defect_registry",
]
