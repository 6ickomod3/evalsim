"""Tests for the pre-registered source-only M5 slices."""

from __future__ import annotations

import inspect
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from evalsim import Agent, AgentType, MapPolyline, MapType, Scenario
from evalsim.slices import (
    M5_SLICE_NAMES,
    M5_SLICE_REGISTRY,
    M5_SLICE_SPECS,
    M5_SLICE_VERSION,
    SliceRegistry,
    SliceResult,
    SliceSpec,
    evaluate_m5_slices,
)


def _agent(
    agent_id: int,
    *,
    agent_type: AgentType = AgentType.VEHICLE,
    valid: tuple[bool, ...] = (True, True, True, True),
    x: tuple[float, ...] | None = None,
    y: tuple[float, ...] | None = None,
    heading: tuple[float, ...] | None = None,
    vx: tuple[float, ...] | None = None,
    vy: tuple[float, ...] | None = None,
    length: float = 2.0,
    width: float = 2.0,
) -> Agent:
    steps = len(valid)
    zeros = (0.0,) * steps
    return Agent(
        id=agent_id,
        type=agent_type,
        valid=np.asarray(valid, dtype=bool),
        x=np.asarray(zeros if x is None else x, dtype=float),
        y=np.asarray(zeros if y is None else y, dtype=float),
        heading=np.asarray(zeros if heading is None else heading, dtype=float),
        vx=np.asarray(zeros if vx is None else vx, dtype=float),
        vy=np.asarray(zeros if vy is None else vy, dtype=float),
        length=length,
        width=width,
    )


def _scenario(
    agents: list[Agent],
    *,
    current_index: int = 1,
    map_features: list[MapPolyline] | None = None,
    metadata_extra: dict | None = None,
) -> Scenario:
    metadata = {"source": "unit", "current_index": current_index}
    if metadata_extra:
        metadata.update(metadata_extra)
    return Scenario(
        scenario_id="slice-fixture",
        timestamps=np.arange(agents[0].num_steps, dtype=float) * 0.1,
        agents=agents,
        map=[] if map_features is None else map_features,
        ego_index=0,
        metadata=metadata,
    )


def _by_name(scenario: Scenario) -> dict[str, SliceResult]:
    return {result.slice_name: result for result in evaluate_m5_slices(scenario)}


def _replace_current_x(agent: Agent, current_index: int, value: float) -> Agent:
    x = agent.x.copy()
    x[current_index] = value
    return Agent(
        id=agent.id,
        type=agent.type,
        valid=agent.valid.copy(),
        x=x,
        y=agent.y.copy(),
        heading=agent.heading.copy(),
        vx=agent.vx.copy(),
        vy=agent.vy.copy(),
        length=agent.length,
        width=agent.width,
    )


def _rigid_transform(scenario: Scenario, angle: float, offset: np.ndarray) -> Scenario:
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    agents: list[Agent] = []
    for agent in scenario.agents:
        positions = np.column_stack((agent.x, agent.y)) @ rotation.T + offset
        velocities = np.column_stack((agent.vx, agent.vy)) @ rotation.T
        headings = np.asarray(
            [
                math.remainder(float(value) + angle, 2.0 * math.pi)
                for value in agent.heading
            ]
        )
        agents.append(
            Agent(
                id=agent.id,
                type=agent.type,
                valid=agent.valid.copy(),
                x=positions[:, 0],
                y=positions[:, 1],
                heading=headings,
                vx=velocities[:, 0],
                vy=velocities[:, 1],
                length=agent.length,
                width=agent.width,
            )
        )
    features = [
        MapPolyline(type=feature.type, xy=feature.xy @ rotation.T + offset)
        for feature in scenario.map
    ]
    return Scenario(
        scenario_id=scenario.scenario_id,
        timestamps=scenario.timestamps.copy(),
        agents=agents,
        map=features,
        ego_index=scenario.ego_index,
        metadata=dict(scenario.metadata),
    )


def test_registry_is_exactly_eight_ordered_source_only_slices() -> None:
    assert M5_SLICE_VERSION == "m5-womd-slices-1.0.0"
    assert len(M5_SLICE_SPECS) == 8
    assert M5_SLICE_NAMES == (
        "all",
        "vru_present_current",
        "current_world_count_ge_8",
        "retained_world_count_ge_16",
        "observed_ego_turn_ge_15deg",
        "low_current_cv_ttc_le_3s",
        "future_lifecycle_change",
        "supported_lane_available",
    )
    assert M5_SLICE_REGISTRY.names == M5_SLICE_NAMES
    assert tuple(inspect.signature(evaluate_m5_slices).parameters) == ("scenario",)
    with pytest.raises(ValueError, match="canonical eight"):
        SliceRegistry(specs=M5_SLICE_SPECS[:-1])


def test_specs_and_results_are_validated_and_immutable() -> None:
    with pytest.raises(ValueError, match="lower_snake_case"):
        SliceSpec("Not Valid", M5_SLICE_VERSION, "description")
    with pytest.raises(ValueError, match="exactly"):
        SliceSpec("valid_name", "other", "description")
    with pytest.raises(ValueError, match="eligible"):
        SliceResult(
            "all",
            M5_SLICE_VERSION,
            "scenario",
            eligible=True,
            member=True,
            reason="unexpected",
        )
    with pytest.raises(ValueError, match="cannot be a member"):
        SliceResult(
            "all",
            M5_SLICE_VERSION,
            "scenario",
            eligible=False,
            member=True,
            reason="reason",
        )
    result = SliceResult(
        "all",
        M5_SLICE_VERSION,
        "scenario",
        eligible=True,
        member=True,
    )
    with pytest.raises(FrozenInstanceError):
        result.member = False  # type: ignore[misc]


def test_evaluator_rejects_non_scenario_and_bad_current_index() -> None:
    with pytest.raises(TypeError, match="only a Scenario"):
        M5_SLICE_REGISTRY.evaluate(object())  # type: ignore[arg-type]
    scenario = _scenario([_agent(0), _agent(1)])
    scenario.metadata["current_index"] = True
    with pytest.raises(ValueError, match="must be an integer"):
        evaluate_m5_slices(scenario)


@pytest.mark.parametrize(
    ("world_type", "world_valid", "expected"),
    [
        (AgentType.PEDESTRIAN, (False, True, True, True), True),
        (AgentType.CYCLIST, (False, True, True, True), True),
        (AgentType.PEDESTRIAN, (True, False, True, True), False),
        (AgentType.VEHICLE, (False, True, True, True), False),
    ],
)
def test_vru_slice_uses_non_ego_current_validity(
    world_type: AgentType,
    world_valid: tuple[bool, ...],
    expected: bool,
) -> None:
    scenario = _scenario(
        [_agent(0), _agent(1, agent_type=world_type, valid=world_valid)]
    )
    assert _by_name(scenario)["vru_present_current"].member is expected


@pytest.mark.parametrize(("count", "expected"), [(7, False), (8, True)])
def test_current_world_count_threshold(count: int, expected: bool) -> None:
    agents = [_agent(0)] + [_agent(index + 1) for index in range(count)]
    assert _by_name(_scenario(agents))["current_world_count_ge_8"].member is expected


@pytest.mark.parametrize(("count", "expected"), [(15, False), (16, True)])
def test_retained_world_count_threshold_counts_any_valid_frame(
    count: int,
    expected: bool,
) -> None:
    retained = [
        _agent(index + 1, valid=(True, False, False, False))
        for index in range(count)
    ]
    never_retained = _agent(100, valid=(False, False, False, False))
    result = _by_name(_scenario([_agent(0), *retained, never_retained]))[
        "retained_world_count_ge_16"
    ]
    assert result.member is expected


def test_observed_ego_turn_inclusive_threshold_wrap_and_invalid_reason() -> None:
    at_threshold = _scenario(
        [
            _agent(
                0,
                heading=(0.0, math.pi / 12.0, math.pi / 12.0, math.pi / 12.0),
            ),
            _agent(1),
        ]
    )
    assert _by_name(at_threshold)["observed_ego_turn_ge_15deg"].member

    below = _scenario(
        [
            _agent(
                0,
                heading=(
                    0.0,
                    math.nextafter(math.pi / 12.0, 0.0),
                    0.0,
                    0.0,
                ),
            ),
            _agent(1),
        ]
    )
    assert not _by_name(below)["observed_ego_turn_ge_15deg"].member

    wrapped = _scenario(
        [
            _agent(
                0,
                heading=(
                    math.radians(179.0),
                    math.radians(-165.0),
                    0.0,
                    0.0,
                ),
            ),
            _agent(1),
        ]
    )
    assert _by_name(wrapped)["observed_ego_turn_ge_15deg"].member

    insufficient = _scenario(
        [
            _agent(0, valid=(False, True, False, False)),
            _agent(1),
        ]
    )
    result = _by_name(insufficient)["observed_ego_turn_ge_15deg"]
    assert (result.eligible, result.member, result.reason) == (
        False,
        False,
        "insufficient_observed_ego_heading",
    )


@pytest.mark.parametrize(
    ("contact_time", "expected"),
    [
        (6.0, False),
        (3.0, True),
        (3.0 + 1e-9, False),
    ],
)
def test_current_cv_ttc_three_second_boundary(
    contact_time: float,
    expected: bool,
) -> None:
    # Each 2x2 box has half-diagonal sqrt(2); use an explicit separation whose
    # contact time is controlled by replacing the world center below.
    radius_sum = 2.0 * math.sqrt(2.0)
    ego = _agent(0, vx=(0.0, 0.0, 0.0, 0.0))
    world = _agent(1, vx=(-1.0, -1.0, -1.0, -1.0))
    world = _replace_current_x(world, 1, radius_sum + contact_time)
    result = _by_name(_scenario([ego, world]))["low_current_cv_ttc_le_3s"]
    assert result.eligible
    assert result.member is expected


def test_current_cv_ttc_overlap_separating_and_no_counterpart() -> None:
    ego = _agent(0)
    overlapping = _replace_current_x(_agent(1), 1, 0.0)
    assert _by_name(_scenario([ego, overlapping]))[
        "low_current_cv_ttc_le_3s"
    ].member

    separating = _replace_current_x(
        _agent(1, vx=(1.0, 1.0, 1.0, 1.0)),
        1,
        10.0,
    )
    assert not _by_name(_scenario([ego, separating]))[
        "low_current_cv_ttc_le_3s"
    ].member

    no_counterpart = _scenario(
        [ego, _agent(1, valid=(True, False, True, True))]
    )
    result = _by_name(no_counterpart)["low_current_cv_ttc_le_3s"]
    assert (result.eligible, result.member, result.reason) == (
        False,
        False,
        "no_current_counterpart",
    )


def test_future_lifecycle_includes_current_to_next_and_excludes_ego() -> None:
    adjacent_change = _scenario(
        [_agent(0), _agent(1, valid=(True, True, False, False))]
    )
    assert _by_name(adjacent_change)["future_lifecycle_change"].member

    ego_only_change = _scenario(
        [
            _agent(0, valid=(True, True, False, False)),
            _agent(1, valid=(True, True, True, True)),
        ]
    )
    assert not _by_name(ego_only_change)["future_lifecycle_change"].member

    final_current = _scenario([_agent(0), _agent(1)], current_index=3)
    assert not _by_name(final_current)["future_lifecycle_change"].member


@pytest.mark.parametrize(
    ("xy", "expected"),
    [
        (np.asarray([[0.0, 0.0], [0.0, 0.0]]), False),
        (np.asarray([[0.0, 0.0], [1.0, 0.0]]), True),
        (np.asarray([[np.nan, 0.0], [1.0, 0.0]]), False),
        (
            np.asarray([[np.nan, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            True,
        ),
    ],
)
def test_supported_lane_requires_a_finite_nonzero_segment(
    xy: np.ndarray,
    expected: bool,
) -> None:
    scenario = _scenario(
        [_agent(0), _agent(1)],
        map_features=[MapPolyline(type=MapType.LANE, xy=xy)],
    )
    assert _by_name(scenario)["supported_lane_available"].member is expected


def test_all_slice_is_always_eligible_and_member() -> None:
    result = _by_name(_scenario([_agent(0), _agent(1)]))["all"]
    assert (result.eligible, result.member, result.reason) == (True, True, None)


def test_membership_is_rigid_transform_invariant() -> None:
    radius_sum = 2.0 * math.sqrt(2.0)
    scenario = _scenario(
        [
            _agent(
                0,
                heading=(
                    0.0,
                    math.radians(16.0),
                    math.radians(16.0),
                    math.radians(16.0),
                ),
            ),
            _replace_current_x(
                _agent(
                    1,
                    agent_type=AgentType.CYCLIST,
                    vx=(-1.0, -1.0, -1.0, -1.0),
                    valid=(True, True, False, False),
                ),
                1,
                radius_sum + 2.0,
            ),
        ],
        map_features=[
            MapPolyline(
                type=MapType.LANE,
                xy=np.asarray([[0.0, 0.0], [10.0, 0.0]]),
            )
        ],
    )
    transformed = _rigid_transform(
        scenario,
        angle=1.123,
        offset=np.asarray([123.0, -77.0]),
    )
    assert evaluate_m5_slices(transformed) == evaluate_m5_slices(scenario)


def test_policy_result_metadata_and_order_cannot_change_membership() -> None:
    agents = [_agent(0), _agent(1, agent_type=AgentType.PEDESTRIAN)]
    left = _scenario(
        agents,
        metadata_extra={
            "policy_order": ["log_replay", "idm"],
            "result_order": ["metric_a", "metric_b"],
        },
    )
    right = _scenario(
        agents,
        metadata_extra={
            "result_order": ["metric_b", "metric_a"],
            "policy_order": ["idm", "log_replay"],
        },
    )
    assert evaluate_m5_slices(left) == evaluate_m5_slices(right)
