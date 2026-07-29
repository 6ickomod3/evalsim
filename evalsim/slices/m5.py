"""Pre-registered, source-only M5 scenario slices.

Slice membership is a property of the source :class:`~evalsim.contracts.Scenario`.
This module deliberately has no rollout, policy, metric, or result input, preventing
post-outcome slice selection.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from evalsim.contracts import Agent, AgentType, MapType, Scenario

M5_SLICE_VERSION = "m5-womd-slices-1.0.0"

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_TTC_CAP_SECONDS = 5.0
_TTC_MEMBER_THRESHOLD_SECONDS = 3.0
_TURN_THRESHOLD_RADIANS = math.pi / 12.0


@dataclass(frozen=True, slots=True)
class SliceSpec:
    """Immutable declaration of one source-only M5 slice."""

    name: str
    version: str
    description: str
    ineligible_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_PATTERN.fullmatch(self.name):
            raise ValueError("slice name must be a non-empty lower_snake_case string")
        if self.version != M5_SLICE_VERSION:
            raise ValueError(
                f"slice version must be exactly {M5_SLICE_VERSION!r}"
            )
        if (
            not isinstance(self.description, str)
            or not self.description
            or self.description != self.description.strip()
        ):
            raise ValueError("slice description must be a non-empty trimmed string")
        if isinstance(self.ineligible_reasons, (str, bytes)):
            raise ValueError("ineligible_reasons must be a sequence, not a string")
        reasons = tuple(self.ineligible_reasons)
        if any(
            not isinstance(reason, str)
            or not _REASON_PATTERN.fullmatch(reason)
            for reason in reasons
        ):
            raise ValueError(
                "ineligible reasons must be non-empty lower_snake_case strings"
            )
        if len(set(reasons)) != len(reasons):
            raise ValueError("ineligible reasons must be unique")
        object.__setattr__(self, "ineligible_reasons", reasons)


@dataclass(frozen=True, slots=True)
class SliceResult:
    """Immutable membership result for one scenario and one registered slice."""

    slice_name: str
    slice_version: str
    scenario_id: str
    eligible: bool
    member: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.slice_name, str)
            or not _NAME_PATTERN.fullmatch(self.slice_name)
        ):
            raise ValueError(
                "slice_name must be a non-empty lower_snake_case string"
            )
        if self.slice_version != M5_SLICE_VERSION:
            raise ValueError(
                f"slice_version must be exactly {M5_SLICE_VERSION!r}"
            )
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario_id must be a non-empty string")
        if type(self.eligible) is not bool or type(self.member) is not bool:
            raise ValueError("eligible and member must be booleans")
        if self.eligible:
            if self.reason is not None:
                raise ValueError("an eligible slice result cannot have a reason")
        else:
            if self.member:
                raise ValueError("an ineligible slice result cannot be a member")
            if (
                not isinstance(self.reason, str)
                or not _REASON_PATTERN.fullmatch(self.reason)
            ):
                raise ValueError(
                    "an ineligible result requires a lower_snake_case reason"
                )


M5_SLICE_SPECS = (
    SliceSpec(
        name="all",
        version=M5_SLICE_VERSION,
        description="Every accepted scenario.",
    ),
    SliceSpec(
        name="vru_present_current",
        version=M5_SLICE_VERSION,
        description=(
            "At least one current-valid non-ego pedestrian or cyclist is present."
        ),
    ),
    SliceSpec(
        name="current_world_count_ge_8",
        version=M5_SLICE_VERSION,
        description="At least eight non-ego agents are valid at the current frame.",
    ),
    SliceSpec(
        name="retained_world_count_ge_16",
        version=M5_SLICE_VERSION,
        description=(
            "At least sixteen non-ego agents are valid at any contract frame."
        ),
    ),
    SliceSpec(
        name="observed_ego_turn_ge_15deg",
        version=M5_SLICE_VERSION,
        description=(
            "Observed ego heading changes by at least fifteen degrees in the "
            "inclusive ten-frame history window."
        ),
        ineligible_reasons=("insufficient_observed_ego_heading",),
    ),
    SliceSpec(
        name="low_current_cv_ttc_le_3s",
        version=M5_SLICE_VERSION,
        description=(
            "Minimum current ego-to-world constant-velocity disc TTC is at most "
            "three seconds."
        ),
        ineligible_reasons=("no_current_counterpart",),
    ),
    SliceSpec(
        name="future_lifecycle_change",
        version=M5_SLICE_VERSION,
        description=(
            "A non-ego source validity bit changes after the current frame."
        ),
    ),
    SliceSpec(
        name="supported_lane_available",
        version=M5_SLICE_VERSION,
        description="At least one retained lane has a finite nonzero segment.",
    ),
)

M5_SLICE_NAMES = tuple(spec.name for spec in M5_SLICE_SPECS)


def _current_index(scenario: Scenario) -> int:
    if scenario.num_agents == 0:
        raise ValueError("M5 slices require a scenario with an ego agent")
    raw_index = scenario.metadata.get("current_index", 0)
    if (
        isinstance(raw_index, (bool, np.bool_))
        or not isinstance(raw_index, (int, np.integer))
    ):
        raise ValueError("Scenario.metadata['current_index'] must be an integer")
    current_index = int(raw_index)
    if not 0 <= current_index < scenario.num_steps:
        raise ValueError(
            "Scenario.metadata['current_index'] is outside the scenario horizon"
        )
    return current_index


def _result(
    spec: SliceSpec,
    scenario: Scenario,
    *,
    eligible: bool,
    member: bool,
    reason: str | None = None,
) -> SliceResult:
    result = SliceResult(
        slice_name=spec.name,
        slice_version=spec.version,
        scenario_id=scenario.scenario_id,
        eligible=eligible,
        member=member,
        reason=reason,
    )
    if result.reason is not None and result.reason not in spec.ineligible_reasons:
        raise ValueError(
            f"slice {spec.name!r} returned unregistered reason {result.reason!r}"
        )
    return result


def _valid_non_ego_agents(
    scenario: Scenario,
    current_index: int,
) -> tuple[Agent, ...]:
    return tuple(
        agent
        for index, agent in enumerate(scenario.agents)
        if index != scenario.ego_index and bool(agent.valid[current_index])
    )


def _all(
    spec: SliceSpec,
    scenario: Scenario,
    current_index: int,
) -> SliceResult:
    del current_index
    return _result(spec, scenario, eligible=True, member=True)


def _vru_present_current(
    spec: SliceSpec,
    scenario: Scenario,
    current_index: int,
) -> SliceResult:
    member = any(
        agent.type in (AgentType.PEDESTRIAN, AgentType.CYCLIST)
        for agent in _valid_non_ego_agents(scenario, current_index)
    )
    return _result(spec, scenario, eligible=True, member=member)


def _current_world_count_ge_8(
    spec: SliceSpec,
    scenario: Scenario,
    current_index: int,
) -> SliceResult:
    member = len(_valid_non_ego_agents(scenario, current_index)) >= 8
    return _result(spec, scenario, eligible=True, member=member)


def _retained_world_count_ge_16(
    spec: SliceSpec,
    scenario: Scenario,
    current_index: int,
) -> SliceResult:
    del current_index
    count = sum(
        bool(np.any(agent.valid))
        for index, agent in enumerate(scenario.agents)
        if index != scenario.ego_index
    )
    return _result(spec, scenario, eligible=True, member=count >= 16)


def _observed_ego_turn_ge_15deg(
    spec: SliceSpec,
    scenario: Scenario,
    current_index: int,
) -> SliceResult:
    ego = scenario.ego
    window_start = max(0, current_index - 9)
    window = np.arange(window_start, current_index + 1, dtype=np.int64)
    valid_indices = window[ego.valid[window]]
    if valid_indices.size < 2:
        return _result(
            spec,
            scenario,
            eligible=False,
            member=False,
            reason="insufficient_observed_ego_heading",
        )
    first_heading = float(ego.heading[int(valid_indices[0])])
    last_heading = float(ego.heading[int(valid_indices[-1])])
    if not math.isfinite(first_heading) or not math.isfinite(last_heading):
        raise ValueError("valid observed ego headings must be finite")
    heading_change = abs(
        math.remainder(last_heading - first_heading, 2.0 * math.pi)
    )
    return _result(
        spec,
        scenario,
        eligible=True,
        member=heading_change >= _TURN_THRESHOLD_RADIANS,
    )


def _agent_current_state(
    agent: Agent,
    current_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    position = np.asarray(
        [agent.x[current_index], agent.y[current_index]],
        dtype=np.float64,
    )
    velocity = np.asarray(
        [agent.vx[current_index], agent.vy[current_index]],
        dtype=np.float64,
    )
    dimensions = np.asarray([agent.length, agent.width], dtype=np.float64)
    if (
        not np.all(np.isfinite(position))
        or not np.all(np.isfinite(velocity))
        or not np.all(np.isfinite(dimensions))
        or np.any(dimensions <= 0.0)
    ):
        raise ValueError(
            "current-valid TTC agents require finite state and positive dimensions"
        )
    radius = 0.5 * float(np.hypot(dimensions[0], dimensions[1]))
    if not math.isfinite(radius):
        raise ValueError("current-valid TTC agent radius must be finite")
    return position, velocity, radius


def _capped_constant_velocity_ttc(
    ego: Agent,
    other: Agent,
    current_index: int,
) -> float:
    ego_position, ego_velocity, ego_radius = _agent_current_state(
        ego,
        current_index,
    )
    other_position, other_velocity, other_radius = _agent_current_state(
        other,
        current_index,
    )
    relative_position = other_position - ego_position
    relative_velocity = other_velocity - ego_velocity
    combined_radius = ego_radius + other_radius

    a = float(np.dot(relative_velocity, relative_velocity))
    b = 2.0 * float(np.dot(relative_position, relative_velocity))
    c = float(np.dot(relative_position, relative_position)) - combined_radius**2
    if not all(math.isfinite(value) for value in (a, b, c, combined_radius)):
        raise ValueError("constant-velocity TTC intermediates must be finite")
    if c <= 0.0:
        return 0.0
    if a == 0.0:
        return _TTC_CAP_SECONDS

    discriminant = b**2 - 4.0 * a * c
    if not math.isfinite(discriminant):
        raise ValueError("constant-velocity TTC discriminant must be finite")
    if discriminant < 0.0:
        return _TTC_CAP_SECONDS
    root_term = math.sqrt(discriminant)
    denominator = 2.0 * a
    roots = (
        (-b - root_term) / denominator,
        (-b + root_term) / denominator,
    )
    nonnegative_roots = tuple(root for root in roots if root >= 0.0)
    if not nonnegative_roots:
        return _TTC_CAP_SECONDS
    return min(min(nonnegative_roots), _TTC_CAP_SECONDS)


def _low_current_cv_ttc_le_3s(
    spec: SliceSpec,
    scenario: Scenario,
    current_index: int,
) -> SliceResult:
    ego = scenario.ego
    counterparts = _valid_non_ego_agents(scenario, current_index)
    if not bool(ego.valid[current_index]) or not counterparts:
        return _result(
            spec,
            scenario,
            eligible=False,
            member=False,
            reason="no_current_counterpart",
        )
    minimum_ttc = min(
        _capped_constant_velocity_ttc(ego, other, current_index)
        for other in counterparts
    )
    return _result(
        spec,
        scenario,
        eligible=True,
        member=minimum_ttc <= _TTC_MEMBER_THRESHOLD_SECONDS,
    )


def _future_lifecycle_change(
    spec: SliceSpec,
    scenario: Scenario,
    current_index: int,
) -> SliceResult:
    member = any(
        bool(np.any(agent.valid[current_index + 1 :] != agent.valid[current_index:-1]))
        for index, agent in enumerate(scenario.agents)
        if index != scenario.ego_index and current_index + 1 < scenario.num_steps
    )
    return _result(spec, scenario, eligible=True, member=member)


def _supported_lane_available(
    spec: SliceSpec,
    scenario: Scenario,
    current_index: int,
) -> SliceResult:
    del current_index
    member = False
    for feature in scenario.map:
        if feature.type != MapType.LANE or feature.xy.shape[0] < 2:
            continue
        starts = feature.xy[:-1]
        ends = feature.xy[1:]
        with np.errstate(invalid="ignore", over="ignore"):
            segments = ends - starts
        finite_endpoints = np.all(np.isfinite(starts), axis=1) & np.all(
            np.isfinite(ends),
            axis=1,
        )
        finite_segments = np.all(np.isfinite(segments), axis=1)
        nonzero = np.any(segments != 0.0, axis=1)
        if bool(np.any(finite_endpoints & finite_segments & nonzero)):
            member = True
            break
    return _result(spec, scenario, eligible=True, member=member)


_SliceEvaluator = Callable[[SliceSpec, Scenario, int], SliceResult]
_EVALUATORS: Mapping[str, _SliceEvaluator] = MappingProxyType(
    {
        "all": _all,
        "vru_present_current": _vru_present_current,
        "current_world_count_ge_8": _current_world_count_ge_8,
        "retained_world_count_ge_16": _retained_world_count_ge_16,
        "observed_ego_turn_ge_15deg": _observed_ego_turn_ge_15deg,
        "low_current_cv_ttc_le_3s": _low_current_cv_ttc_le_3s,
        "future_lifecycle_change": _future_lifecycle_change,
        "supported_lane_available": _supported_lane_available,
    }
)


@dataclass(frozen=True, slots=True)
class SliceRegistry:
    """Immutable registry for exactly the eight pre-registered M5 slices."""

    specs: tuple[SliceSpec, ...] = M5_SLICE_SPECS

    def __post_init__(self) -> None:
        if isinstance(self.specs, (str, bytes)):
            raise ValueError("registry specs must be a sequence, not a string")
        specs = tuple(self.specs)
        if specs != M5_SLICE_SPECS:
            raise ValueError(
                "the M5 registry must contain exactly the canonical eight specs"
            )
        if tuple(_EVALUATORS) != M5_SLICE_NAMES:
            raise RuntimeError("M5 evaluator order does not match the slice specs")
        object.__setattr__(self, "specs", specs)

    @property
    def names(self) -> tuple[str, ...]:
        return M5_SLICE_NAMES

    def get(self, name: str) -> SliceSpec:
        """Return one registered spec, failing closed for unknown names."""

        if not isinstance(name, str):
            raise TypeError("slice name must be a string")
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def evaluate(self, scenario: Scenario) -> tuple[SliceResult, ...]:
        """Evaluate all slices in canonical order using only source scenario fields."""

        if not isinstance(scenario, Scenario):
            raise TypeError("M5 slice evaluation accepts only a Scenario")
        current_index = _current_index(scenario)
        return tuple(
            _EVALUATORS[spec.name](spec, scenario, current_index)
            for spec in self.specs
        )


M5_SLICE_REGISTRY = SliceRegistry()


def evaluate_m5_slices(scenario: Scenario) -> tuple[SliceResult, ...]:
    """Evaluate the canonical M5 registry for one source scenario."""

    return M5_SLICE_REGISTRY.evaluate(scenario)
