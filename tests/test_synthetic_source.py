"""M1 synthetic-source acceptance and determinism tests."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from evalsim import (
    Agent,
    AgentType,
    MapType,
    Scenario,
    scenario_from_parquet,
    scenario_to_parquet,
)
from evalsim.sources import SCENARIO_KINDS, ScenarioKind, SyntheticSource


def _assert_scenarios_equal(left: Scenario, right: Scenario) -> None:
    assert left.scenario_id == right.scenario_id
    assert left.ego_index == right.ego_index
    assert left.metadata == right.metadata
    np.testing.assert_array_equal(left.timestamps, right.timestamps)
    assert len(left.agents) == len(right.agents)
    for left_agent, right_agent in zip(left.agents, right.agents):
        assert left_agent.id == right_agent.id
        assert left_agent.type == right_agent.type
        assert left_agent.length == right_agent.length
        assert left_agent.width == right_agent.width
        for field in ("valid", "x", "y", "heading", "vx", "vy"):
            np.testing.assert_array_equal(
                getattr(left_agent, field),
                getattr(right_agent, field),
            )
    assert len(left.map) == len(right.map)
    for left_feature, right_feature in zip(left.map, right.map):
        assert left_feature.type == right_feature.type
        np.testing.assert_array_equal(left_feature.xy, right_feature.xy)


def _oriented_boxes_overlap(
    left: Agent,
    right: Agent,
    step: int,
) -> bool:
    left_long = np.array(
        [np.cos(left.heading[step]), np.sin(left.heading[step])]
    )
    left_lat = np.array([-left_long[1], left_long[0]])
    right_long = np.array(
        [np.cos(right.heading[step]), np.sin(right.heading[step])]
    )
    right_lat = np.array([-right_long[1], right_long[0]])
    displacement = np.array(
        [right.x[step] - left.x[step], right.y[step] - left.y[step]]
    )

    def projection_radius(
        agent: Agent,
        longitudinal: np.ndarray,
        lateral: np.ndarray,
        axis: np.ndarray,
    ) -> float:
        return (
            agent.length / 2.0 * abs(float(np.dot(longitudinal, axis)))
            + agent.width / 2.0 * abs(float(np.dot(lateral, axis)))
        )

    for axis in (left_long, left_lat, right_long, right_lat):
        center_distance = abs(float(np.dot(displacement, axis)))
        combined_radius = projection_radius(
            left,
            left_long,
            left_lat,
            axis,
        ) + projection_radius(
            right,
            right_long,
            right_lat,
            axis,
        )
        if center_distance > combined_radius:
            return False
    return True


def _scenario_has_overlap(scenario: Scenario) -> bool:
    for left_index, left in enumerate(scenario.agents):
        for right in scenario.agents[left_index + 1 :]:
            mutually_valid = np.flatnonzero(left.valid & right.valid)
            if any(
                _oriented_boxes_overlap(left, right, int(step))
                for step in mutually_valid
            ):
                return True
    return False


def _point_to_polyline_distance(
    point: np.ndarray,
    polyline: np.ndarray,
) -> tuple[float, np.ndarray]:
    starts = polyline[:-1]
    segments = polyline[1:] - starts
    relative = point - starts
    squared_lengths = np.sum(segments * segments, axis=1)
    fractions = np.divide(
        np.sum(relative * segments, axis=1),
        squared_lengths,
        out=np.zeros_like(squared_lengths),
        where=squared_lengths > 0.0,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts + fractions[:, None] * segments
    distances = np.linalg.norm(point - projections, axis=1)
    segment_index = int(np.argmin(distances))
    tangent = segments[segment_index]
    tangent /= np.linalg.norm(tangent)
    return float(distances[segment_index]), tangent


def _nearest_feature(
    point: np.ndarray,
    scenario: Scenario,
    map_type: MapType,
) -> tuple[float, np.ndarray]:
    candidates = [
        _point_to_polyline_distance(point, feature.xy)
        for feature in scenario.map
        if feature.type == map_type and len(feature.xy) >= 2
    ]
    return min(candidates, key=lambda item: item[0])


def _signed_side_of_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    starts = polyline[:-1]
    segments = polyline[1:] - starts
    squared_lengths = np.sum(segments * segments, axis=1)
    fractions = np.divide(
        np.sum((point - starts) * segments, axis=1),
        squared_lengths,
        out=np.zeros_like(squared_lengths),
        where=squared_lengths > 0.0,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts + fractions[:, None] * segments
    distances = np.linalg.norm(point - projections, axis=1)
    segment_index = int(np.argmin(distances))
    tangent = segments[segment_index]
    tangent /= np.linalg.norm(tangent)
    offset = point - projections[segment_index]
    return float(tangent[0] * offset[1] - tangent[1] * offset[0])


def test_generate_50_is_balanced_and_contract_valid() -> None:
    source = SyntheticSource(seed=2026)
    scenarios = source.generate(50)

    assert len(scenarios) == 50
    assert len({scenario.scenario_id for scenario in scenarios}) == 50
    kind_counts = Counter(
        scenario.metadata["scenario_kind"] for scenario in scenarios
    )
    assert kind_counts == Counter({kind.value: 10 for kind in SCENARIO_KINDS})

    required_tag = {
        ScenarioKind.FOLLOWING: "following",
        ScenarioKind.INTERSECTION: "intersection",
        ScenarioKind.MERGE: "merge",
        ScenarioKind.TURN: "turn",
        ScenarioKind.PEDESTRIAN_CROSSING: "pedestrian_present",
    }
    partial_mask_count = 0
    for scenario in scenarios:
        assert scenario.num_steps == source.num_steps
        assert scenario.num_agents >= 2
        assert scenario.ego_index == 0
        assert np.all(np.isfinite(scenario.timestamps))
        assert np.all(np.diff(scenario.timestamps) > 0.0)
        np.testing.assert_allclose(np.diff(scenario.timestamps), source.dt)
        assert scenario.metadata["source"] == "synthetic"
        assert scenario.metadata["source_fingerprint"] == source.fingerprint
        assert scenario.metadata["split"] == source.split

        kind = ScenarioKind(scenario.metadata["scenario_kind"])
        assert required_tag[kind] in scenario.metadata["tags"]
        map_types = {feature.type for feature in scenario.map}
        assert MapType.LANE in map_types
        assert MapType.ROAD_EDGE in map_types
        assert all(np.all(np.isfinite(feature.xy)) for feature in scenario.map)

        agent_ids = [agent.id for agent in scenario.agents]
        assert len(agent_ids) == len(set(agent_ids))
        assert np.all(scenario.ego.valid)
        for agent in scenario.agents:
            assert agent.valid.dtype == np.bool_
            assert agent.valid.shape == (source.num_steps,)
            assert agent.length > 0.0
            assert agent.width > 0.0
            assert np.any(agent.valid)
            for field in ("x", "y", "heading", "vx", "vy"):
                values = getattr(agent, field)
                assert values.shape == (source.num_steps,)
                assert np.all(np.isfinite(values))
            assert np.all(agent.heading >= -np.pi)
            assert np.all(agent.heading <= np.pi)
            partial_mask_count += int(not np.all(agent.valid))

    assert partial_mask_count > 0


def test_kind_specific_semantics_are_present() -> None:
    source = SyntheticSource(seed=11)
    scenarios = {
        kind: source.generate_one(index, kind)
        for index, kind in enumerate(SCENARIO_KINDS)
    }

    following = scenarios[ScenarioKind.FOLLOWING]
    assert following.num_agents >= 3

    intersection = scenarios[ScenarioKind.INTERSECTION]
    lane_features = [
        feature for feature in intersection.map if feature.type == MapType.LANE
    ]
    assert any(np.ptp(feature.xy[:, 0]) > 50.0 for feature in lane_features)
    assert any(np.ptp(feature.xy[:, 1]) > 50.0 for feature in lane_features)

    merge = scenarios[ScenarioKind.MERGE]
    assert any(
        np.ptp(feature.xy[:, 1]) > 3.0
        for feature in merge.map
        if feature.type == MapType.LANE
    )

    turn = scenarios[ScenarioKind.TURN]
    assert np.ptp(np.unwrap(turn.ego.heading)) > np.pi / 3.0

    pedestrian = scenarios[ScenarioKind.PEDESTRIAN_CROSSING]
    assert any(agent.type.value == "pedestrian" for agent in pedestrian.agents)
    assert any(feature.type == MapType.CROSSWALK for feature in pedestrian.map)


@pytest.mark.parametrize("seed", [0, 1, 42, 2026, 2**32 - 1])
def test_reference_trajectories_are_collision_free(seed: int) -> None:
    scenarios = SyntheticSource(seed=seed).generate(50)
    overlapping = [
        scenario.scenario_id
        for scenario in scenarios
        if _scenario_has_overlap(scenario)
    ]
    assert overlapping == []


@pytest.mark.parametrize(
    "source_kwargs",
    [
        {"num_steps": 10, "dt": 0.7},
        {"num_steps": 121, "dt": 1.0},
    ],
)
def test_supported_horizon_bounds_remain_collision_free(
    source_kwargs: dict,
) -> None:
    for seed in (0, 1, 42):
        scenarios = SyntheticSource(seed=seed, **source_kwargs).generate(50)
        assert not any(_scenario_has_overlap(scenario) for scenario in scenarios)


def test_vehicle_trajectories_follow_directed_lanes_and_stay_inside_edges() -> None:
    scenarios = SyntheticSource(seed=2026).generate(50)

    for scenario in scenarios:
        for agent in scenario.agents:
            if agent.type != AgentType.VEHICLE:
                continue
            for step in np.flatnonzero(agent.valid):
                point = np.array([agent.x[step], agent.y[step]])
                lane_distance, lane_tangent = _nearest_feature(
                    point,
                    scenario,
                    MapType.LANE,
                )
                velocity = np.array([agent.vx[step], agent.vy[step]])
                assert lane_distance < 0.15, (
                    scenario.scenario_id,
                    agent.id,
                    step,
                    lane_distance,
                )
                assert float(np.dot(velocity, lane_tangent)) >= -1e-8, (
                    scenario.scenario_id,
                    agent.id,
                    step,
                )

                edge_distance, _ = _nearest_feature(
                    point,
                    scenario,
                    MapType.ROAD_EDGE,
                )
                assert edge_distance > agent.width / 2.0, (
                    scenario.scenario_id,
                    agent.id,
                    step,
                    edge_distance,
                )


def test_merge_edges_connect_and_keep_ramp_drivable_on_the_left() -> None:
    scenario = SyntheticSource(seed=2026).generate_one(2, ScenarioKind.MERGE)
    road_edges = [
        feature.xy
        for feature in scenario.map
        if feature.type == MapType.ROAD_EDGE
    ]
    parameters = scenario.metadata["parameters"]
    lane_width = parameters["lane_width_m"]
    gore = np.array([parameters["gore_x_m"], -lane_width])
    merge_end = np.array([parameters["merge_end_m"], -lane_width])

    def endpoint_count(target: np.ndarray) -> int:
        return sum(
            int(np.array_equal(polyline[0], target))
            + int(np.array_equal(polyline[-1], target))
            for polyline in road_edges
        )

    assert endpoint_count(gore) == 2
    assert endpoint_count(merge_end) == 2

    ramp_edges = [
        polyline for polyline in road_edges if np.ptp(polyline[:, 1]) > 1.0
    ]
    assert len(ramp_edges) == 2
    merging_agent = scenario.agents[2]
    for edge in ramp_edges:
        edge_x_min = float(np.min(edge[:, 0]))
        edge_x_max = float(np.max(edge[:, 0]))
        for step in np.flatnonzero(merging_agent.valid):
            if edge_x_min <= merging_agent.x[step] <= edge_x_max:
                point = np.array(
                    [merging_agent.x[step], merging_agent.y[step]]
                )
                assert (
                    _signed_side_of_polyline(point, edge)
                    > merging_agent.width / 2.0
                )


def test_turning_trajectories_have_plausible_acceleration() -> None:
    source = SyntheticSource(seed=2026)
    turns = [
        source.generate_one(index, ScenarioKind.TURN)
        for index in range(10)
    ]

    for scenario in turns:
        acceleration = np.hypot(
            np.gradient(scenario.ego.vx, source.dt, edge_order=2),
            np.gradient(scenario.ego.vy, source.dt, edge_order=2),
        )
        assert float(np.max(acceleration)) < 4.0


def test_generation_is_reproducible_and_call_order_independent() -> None:
    source = SyntheticSource(seed=42, num_steps=61, dt=0.2)
    first = source.generate(12)
    second = source.generate(12)

    for left, right in zip(first, second):
        _assert_scenarios_equal(left, right)

    for index in reversed(range(12)):
        _assert_scenarios_equal(first[index], source.generate_one(index))


def test_source_configuration_changes_scenario_identity() -> None:
    baseline = SyntheticSource(seed=3).generate_one(0)
    new_seed = SyntheticSource(seed=4).generate_one(0)
    new_timestep = SyntheticSource(seed=3, num_steps=41, dt=0.2).generate_one(0)

    assert baseline.scenario_id != new_seed.scenario_id
    assert baseline.scenario_id != new_timestep.scenario_id
    assert not np.array_equal(baseline.ego.x, new_seed.ego.x)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": 2**32},
        {"seed": True},
        {"num_steps": 2},
        {"num_steps": 9},
        {"num_steps": 10**400},
        {"num_steps": 3.5},
        {"dt": 0.0},
        {"dt": 0.001},
        {"dt": 1.1},
        {"dt": float("nan")},
        {"num_steps": 10, "dt": 0.1},
        {"num_steps": 122, "dt": 1.0},
        {"split": ""},
        {"split": "   "},
    ],
)
def test_invalid_source_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SyntheticSource(**kwargs)


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_invalid_generate_count_is_rejected(count: object) -> None:
    with pytest.raises(ValueError):
        SyntheticSource().generate(count)  # type: ignore[arg-type]


def test_invalid_index_and_kind_are_rejected() -> None:
    source = SyntheticSource()
    with pytest.raises(ValueError):
        source.generate_one(-1)
    with pytest.raises(ValueError):
        source.generate_one(0, "roundabout")


def test_all_50_scenarios_round_trip_through_contract_parquet(tmp_path) -> None:
    scenarios = SyntheticSource(seed=8).generate(50)

    for index, scenario in enumerate(scenarios):
        path = tmp_path / f"{index:05d}.parquet"
        scenario_to_parquet(scenario, path)
        _assert_scenarios_equal(scenario, scenario_from_parquet(path))
