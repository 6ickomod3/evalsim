"""Deterministic, parametric scenarios for local EvalSim development.

The synthetic source is intentionally independent of WOMD and Waymax.  It produces the
same frozen :class:`~evalsim.contracts.Scenario` contract as a real-data adapter, which
lets every downstream milestone run locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np

from evalsim.contracts import Agent, AgentType, MapPolyline, MapType, Scenario

from .identity import (
    SYNTHETIC_RNG_ALGORITHM,
    SYNTHETIC_SOURCE_VERSION,
    parse_synthetic_scenario_id,
    synthetic_scenario_id,
    synthetic_source_fingerprint,
)
from .manifest import ScenarioManifest


class ScenarioKind(str, Enum):
    """Supported synthetic scene families."""

    FOLLOWING = "following"
    INTERSECTION = "intersection"
    MERGE = "merge"
    TURN = "turn"
    PEDESTRIAN_CROSSING = "pedestrian_crossing"


SCENARIO_KINDS: tuple[ScenarioKind, ...] = tuple(ScenarioKind)


@dataclass(frozen=True, slots=True)
class SyntheticSource:
    """Generate deterministic, real-shaped scenarios from a compact configuration.

    ``generate`` assigns scene families round-robin, so the M1 acceptance set of 50
    scenarios contains exactly 10 examples from each family.  A scenario's random
    stream depends only on the source seed, its index, and its family; calls therefore
    do not share mutable random state.
    """

    seed: int = 0
    num_steps: int = 81
    dt: float = 0.1
    split: str = "validation"

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, (int, np.integer))
            or not 0 <= int(self.seed) <= np.iinfo(np.uint32).max
        ):
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")
        if (
            isinstance(self.num_steps, bool)
            or not isinstance(self.num_steps, (int, np.integer))
            or int(self.num_steps) < 10
        ):
            raise ValueError("num_steps must be an integer >= 10")
        if (
            isinstance(self.dt, bool)
            or not isinstance(self.dt, (int, float, np.integer, np.floating))
            or not np.isfinite(float(self.dt))
            or float(self.dt) <= 0.0
        ):
            raise ValueError("dt must be a finite positive number")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("split must be a non-empty string")

        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "num_steps", int(self.num_steps))
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "split", self.split.strip())

        try:
            duration = (self.num_steps - 1) * self.dt
        except OverflowError as exc:
            raise ValueError("scenario duration must be finite") from exc
        if not 0.01 <= self.dt <= 1.0:
            raise ValueError("dt must be in the practical range [0.01, 1.0] seconds")
        if not np.isfinite(duration) or not 6.0 <= duration <= 120.0:
            raise ValueError("scenario duration must be finite and in [6, 120] seconds")

    @property
    def fingerprint(self) -> str:
        """Stable source-configuration fingerprint embedded in every scenario ID."""

        return synthetic_source_fingerprint(
            seed=self.seed,
            num_steps=self.num_steps,
            dt=self.dt,
            split=self.split,
        )

    @property
    def timestamps(self) -> np.ndarray:
        """A fresh, regularly sampled timestamp vector."""

        return np.arange(self.num_steps, dtype=float) * self.dt

    def generate(self, count: int = 50) -> list[Scenario]:
        """Generate ``count`` scenarios, balanced round-robin across scene families."""

        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or int(count) < 0
        ):
            raise ValueError("count must be a non-negative integer")
        return [self.generate_one(index) for index in range(int(count))]

    def generate_one(
        self,
        index: int,
        kind: ScenarioKind | str | None = None,
    ) -> Scenario:
        """Generate one scenario independently of call order."""

        if (
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or int(index) < 0
        ):
            raise ValueError("index must be a non-negative integer")
        index = int(index)

        if kind is None:
            resolved_kind = SCENARIO_KINDS[index % len(SCENARIO_KINDS)]
        else:
            try:
                resolved_kind = ScenarioKind(kind)
            except ValueError as exc:
                choices = ", ".join(item.value for item in SCENARIO_KINDS)
                raise ValueError(f"unknown scenario kind {kind!r}; choose from {choices}") from exc

        kind_index = SCENARIO_KINDS.index(resolved_kind)
        seed_sequence = np.random.SeedSequence([self.seed, index, kind_index])
        rng = np.random.Generator(np.random.PCG64(seed_sequence))
        builder = _BUILDERS[resolved_kind]
        return builder(self, index, rng)

    def make_manifest(self, scenarios: Iterable[Scenario]) -> ScenarioManifest:
        """Build an immutable manifest for scenarios generated by this source."""

        scenario_list = tuple(scenarios)
        for scenario in scenario_list:
            try:
                kind_value, fingerprint, index = parse_synthetic_scenario_id(
                    scenario.scenario_id
                )
                expected = self.generate_one(index, kind_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"scenario {scenario.scenario_id!r} was not generated by this "
                    "SyntheticSource configuration"
                ) from exc
            if (
                fingerprint != self.fingerprint
                or not _scenarios_equal(scenario, expected)
            ):
                raise ValueError(
                    f"scenario {scenario.scenario_id!r} was not generated by this "
                    "SyntheticSource configuration or was modified after generation"
                )
        return ScenarioManifest(
            scenario_ids=tuple(scenario.scenario_id for scenario in scenario_list),
            seed=self.seed,
            num_steps=self.num_steps,
            dt=self.dt,
            split=self.split,
            source_fingerprint=self.fingerprint,
            source_version=SYNTHETIC_SOURCE_VERSION,
        )

    def write_manifest(
        self,
        scenarios: Iterable[Scenario],
        path: str | Path,
    ) -> ScenarioManifest:
        """Create a scenario manifest without ever overwriting an existing file."""

        manifest = self.make_manifest(scenarios)
        manifest.to_file(path)
        return manifest

    def _scenario(
        self,
        *,
        index: int,
        kind: ScenarioKind,
        agents: list[Agent],
        map_features: list[MapPolyline],
        tags: tuple[str, ...],
        parameters: dict[str, float],
    ) -> Scenario:
        scenario_id = synthetic_scenario_id(
            kind=kind.value,
            source_fingerprint=self.fingerprint,
            index=index,
        )
        scenario = Scenario(
            scenario_id=scenario_id,
            timestamps=self.timestamps,
            agents=agents,
            map=map_features,
            ego_index=0,
            metadata={
                "source": "synthetic",
                "source_version": SYNTHETIC_SOURCE_VERSION,
                "source_fingerprint": self.fingerprint,
                "split": self.split,
                "scenario_kind": kind.value,
                "scenario_index": index,
                "seed": self.seed,
                "rng_algorithm": SYNTHETIC_RNG_ALGORITHM,
                "dt": self.dt,
                "tags": list(tags),
                "parameters": parameters,
            },
        )
        _validate_generated_scenario(scenario)
        return scenario


def _scenarios_equal(left: Scenario, right: Scenario) -> bool:
    if (
        left.scenario_id != right.scenario_id
        or left.ego_index != right.ego_index
        or left.metadata != right.metadata
        or not np.array_equal(left.timestamps, right.timestamps)
        or len(left.agents) != len(right.agents)
        or len(left.map) != len(right.map)
    ):
        return False
    for left_agent, right_agent in zip(left.agents, right.agents):
        if (
            left_agent.id != right_agent.id
            or left_agent.type != right_agent.type
            or left_agent.length != right_agent.length
            or left_agent.width != right_agent.width
        ):
            return False
        if any(
            not np.array_equal(
                getattr(left_agent, field),
                getattr(right_agent, field),
            )
            for field in ("valid", "x", "y", "heading", "vx", "vy")
        ):
            return False
    return all(
        left_feature.type == right_feature.type
        and np.array_equal(left_feature.xy, right_feature.xy)
        for left_feature, right_feature in zip(left.map, right.map)
    )


def _validate_generated_scenario(scenario: Scenario) -> None:
    """Fail at the source boundary instead of propagating malformed reference data."""

    if (
        scenario.num_steps < 2
        or not np.all(np.isfinite(scenario.timestamps))
        or not np.all(np.diff(scenario.timestamps) > 0.0)
    ):
        raise RuntimeError("synthetic generator produced invalid timestamps")
    if not scenario.agents or not np.all(scenario.ego.valid):
        raise RuntimeError("synthetic generator produced an invalid ego trajectory")
    for agent in scenario.agents:
        if (
            not np.any(agent.valid)
            or not np.isfinite(agent.length)
            or not np.isfinite(agent.width)
            or agent.length <= 0.0
            or agent.width <= 0.0
        ):
            raise RuntimeError(
                f"synthetic generator produced invalid agent {agent.id}"
            )
        if any(
            not np.all(np.isfinite(getattr(agent, field)))
            for field in ("x", "y", "heading", "vx", "vy")
        ):
            raise RuntimeError(
                f"synthetic generator produced non-finite agent {agent.id}"
            )
    map_types = {feature.type for feature in scenario.map}
    if MapType.LANE not in map_types or MapType.ROAD_EDGE not in map_types:
        raise RuntimeError("synthetic generator omitted required map geometry")
    if any(not np.all(np.isfinite(feature.xy)) for feature in scenario.map):
        raise RuntimeError("synthetic generator produced non-finite map geometry")


def _lifecycle_mask(
    num_steps: int,
    *,
    trim_start: bool = True,
    trim_end: bool = True,
) -> np.ndarray:
    """Make a meaningful partial mask that scales with the configured horizon."""

    trim = max(1, int(round(0.05 * num_steps)))
    valid = np.ones(num_steps, dtype=bool)
    if trim_start:
        valid[:trim] = False
    if trim_end:
        valid[-trim:] = False
    if not np.any(valid):
        raise RuntimeError("synthetic lifecycle mask has no valid samples")
    return valid


def _agent_from_xy(
    *,
    agent_id: int,
    agent_type: AgentType,
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    valid: np.ndarray | None = None,
    length: float = 4.5,
    width: float = 2.0,
    stationary_heading: float = 0.0,
) -> Agent:
    """Construct an agent with velocities and headings tangent to its path."""

    # Each agent owns its buffers. Callers often reuse a constant lane array, and
    # NumPy views here would make mutating one agent silently mutate its peers.
    x = np.array(x, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    vx = np.gradient(x, dt, edge_order=2)
    vy = np.gradient(y, dt, edge_order=2)
    moving = np.hypot(vx, vy) > 1e-9

    if np.any(moving):
        moving_indices = np.flatnonzero(moving)
        moving_heading = np.unwrap(np.arctan2(vy[moving], vx[moving]))
        heading = np.interp(np.arange(x.size), moving_indices, moving_heading)
        heading = (heading + np.pi) % (2.0 * np.pi) - np.pi
    else:
        heading = np.full(x.shape, stationary_heading, dtype=float)

    if valid is None:
        valid = np.ones(x.shape, dtype=bool)
    return Agent(
        id=agent_id,
        type=agent_type,
        valid=np.array(valid, dtype=bool, copy=True),
        x=x,
        y=y,
        heading=heading,
        vx=vx,
        vy=vy,
        length=length,
        width=width,
    )


def _polyline(
    map_type: MapType,
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
) -> MapPolyline:
    return MapPolyline(
        type=map_type,
        xy=np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float))),
    )


def _line(
    map_type: MapType,
    start: tuple[float, float],
    end: tuple[float, float],
    points: int = 101,
) -> MapPolyline:
    return _polyline(
        map_type,
        np.linspace(start[0], end[0], points),
        np.linspace(start[1], end[1], points),
    )


def _straight_map(
    x_min: float,
    x_max: float,
    lane_width: float,
) -> list[MapPolyline]:
    half_lane = lane_width / 2.0
    return [
        _line(MapType.LANE, (x_min, -half_lane), (x_max, -half_lane)),
        _line(MapType.LANE, (x_max, half_lane), (x_min, half_lane)),
        _line(MapType.ROAD_EDGE, (x_min, -lane_width), (x_max, -lane_width)),
        _line(MapType.ROAD_EDGE, (x_max, lane_width), (x_min, lane_width)),
    ]


def _intersection_map(
    extent: float,
    lane_width: float,
) -> list[MapPolyline]:
    half_lane = lane_width / 2.0
    features = [
        _line(MapType.LANE, (-extent, -half_lane), (extent, -half_lane)),
        _line(MapType.LANE, (extent, half_lane), (-extent, half_lane)),
        _line(MapType.LANE, (half_lane, -extent), (half_lane, extent)),
        _line(MapType.LANE, (-half_lane, extent), (-half_lane, -extent)),
    ]

    features.extend(
        [
            # Edge direction keeps the drivable surface on the left.
            _line(
                MapType.ROAD_EDGE,
                (-extent, -lane_width),
                (-lane_width, -lane_width),
                points=41,
            ),
            _line(
                MapType.ROAD_EDGE,
                (lane_width, -lane_width),
                (extent, -lane_width),
                points=41,
            ),
            _line(
                MapType.ROAD_EDGE,
                (-lane_width, lane_width),
                (-extent, lane_width),
                points=41,
            ),
            _line(
                MapType.ROAD_EDGE,
                (extent, lane_width),
                (lane_width, lane_width),
                points=41,
            ),
            _line(
                MapType.ROAD_EDGE,
                (-lane_width, -lane_width),
                (-lane_width, -extent),
                points=41,
            ),
            _line(
                MapType.ROAD_EDGE,
                (-lane_width, extent),
                (-lane_width, lane_width),
                points=41,
            ),
            _line(
                MapType.ROAD_EDGE,
                (lane_width, -extent),
                (lane_width, -lane_width),
                points=41,
            ),
            _line(
                MapType.ROAD_EDGE,
                (lane_width, lane_width),
                (lane_width, extent),
                points=41,
            ),
        ]
    )

    stop_offset = lane_width + 1.0
    features.extend(
        [
            _line(
                MapType.STOP_LINE,
                (-stop_offset, -lane_width),
                (-stop_offset, 0.0),
                points=5,
            ),
            _line(
                MapType.STOP_LINE,
                (stop_offset, 0.0),
                (stop_offset, lane_width),
                points=5,
            ),
            _line(
                MapType.STOP_LINE,
                (0.0, -stop_offset),
                (lane_width, -stop_offset),
                points=5,
            ),
            _line(
                MapType.STOP_LINE,
                (-lane_width, stop_offset),
                (0.0, stop_offset),
                points=5,
            ),
        ]
    )
    return features


def _following_scenario(
    source: SyntheticSource,
    index: int,
    rng: np.random.Generator,
) -> Scenario:
    t = source.timestamps
    lane_width = float(rng.uniform(3.4, 3.8))
    ego_speed = float(rng.uniform(8.0, 11.0))
    leader_speed = float(ego_speed + rng.uniform(0.0, 1.0))
    follower_speed = float(ego_speed - rng.uniform(0.0, 0.8))
    lead_gap = float(rng.uniform(18.0, 26.0))
    follow_gap = float(rng.uniform(14.0, 22.0))
    x0 = float(rng.uniform(-38.0, -32.0))
    lane_y = -lane_width / 2.0

    ego_x = x0 + ego_speed * t
    leader_x = x0 + lead_gap + leader_speed * t
    follower_x = x0 - follow_gap + follower_speed * t
    y = np.full(t.shape, lane_y)
    follower_valid = _lifecycle_mask(source.num_steps)

    agents = [
        _agent_from_xy(
            agent_id=0,
            agent_type=AgentType.VEHICLE,
            x=ego_x,
            y=y,
            dt=source.dt,
        ),
        _agent_from_xy(
            agent_id=1,
            agent_type=AgentType.VEHICLE,
            x=leader_x,
            y=y,
            dt=source.dt,
        ),
        _agent_from_xy(
            agent_id=2,
            agent_type=AgentType.VEHICLE,
            x=follower_x,
            y=y,
            dt=source.dt,
            valid=follower_valid,
        ),
    ]
    x_min = float(min(follower_x.min(), ego_x.min()) - 15.0)
    x_max = float(max(leader_x.max(), ego_x.max()) + 15.0)
    return source._scenario(
        index=index,
        kind=ScenarioKind.FOLLOWING,
        agents=agents,
        map_features=_straight_map(x_min, x_max, lane_width),
        tags=("following", "straight_road"),
        parameters={
            "ego_speed_mps": ego_speed,
            "leader_speed_mps": leader_speed,
            "follower_speed_mps": follower_speed,
            "lead_gap_m": lead_gap,
            "follow_gap_m": follow_gap,
            "lane_width_m": lane_width,
        },
    )


def _intersection_scenario(
    source: SyntheticSource,
    index: int,
    rng: np.random.Generator,
) -> Scenario:
    t = source.timestamps
    lane_width = float(rng.uniform(3.4, 3.8))
    half_lane = lane_width / 2.0
    arrival_time = float(t[-1] * rng.uniform(0.47, 0.53))
    ego_speed = float(rng.uniform(7.5, 9.5))
    cross_speed = float(rng.uniform(6.5, 8.5))
    conflict_gap = float(
        rng.choice(np.array([-1.0, 1.0])) * rng.uniform(1.4, 2.0)
    )
    ego_conflict_time = arrival_time + half_lane / ego_speed
    cross_conflict_time = ego_conflict_time + conflict_gap
    cross_offset = (
        cross_conflict_time - arrival_time + half_lane / cross_speed
    )

    ego_x = ego_speed * (t - arrival_time)
    ego_y = np.full(t.shape, -half_lane)
    cross_x = np.full(t.shape, half_lane)
    cross_y = cross_speed * (t - arrival_time - cross_offset)

    agents = [
        _agent_from_xy(
            agent_id=0,
            agent_type=AgentType.VEHICLE,
            x=ego_x,
            y=ego_y,
            dt=source.dt,
        ),
        _agent_from_xy(
            agent_id=1,
            agent_type=AgentType.VEHICLE,
            x=cross_x,
            y=cross_y,
            dt=source.dt,
            valid=_lifecycle_mask(source.num_steps),
        ),
    ]
    extent = float(
        max(
            45.0,
            np.max(np.abs(ego_x)) + 8.0,
            np.max(np.abs(cross_y)) + 8.0,
        )
    )
    return source._scenario(
        index=index,
        kind=ScenarioKind.INTERSECTION,
        agents=agents,
        map_features=_intersection_map(extent, lane_width),
        tags=("intersection", "four_way_intersection"),
        parameters={
            "ego_speed_mps": ego_speed,
            "cross_speed_mps": cross_speed,
            "cross_arrival_offset_s": cross_offset,
            "cross_conflict_gap_s": conflict_gap,
            "lane_width_m": lane_width,
        },
    )


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _inverse_smoothstep(value: float) -> float:
    """Invert smoothstep on [0, 1] with deterministic bisection."""

    lower = 0.0
    upper = 1.0
    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        midpoint_value = midpoint * midpoint * (3.0 - 2.0 * midpoint)
        if midpoint_value < value:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _merge_y(
    x: np.ndarray,
    *,
    main_y: float,
    ramp_y: float,
    merge_start: float,
    merge_end: float,
) -> np.ndarray:
    progress = (x - merge_start) / (merge_end - merge_start)
    return ramp_y + (main_y - ramp_y) * _smoothstep(progress)


def _merge_scenario(
    source: SyntheticSource,
    index: int,
    rng: np.random.Generator,
) -> Scenario:
    t = source.timestamps
    lane_width = float(rng.uniform(3.4, 3.8))
    main_y = -lane_width / 2.0
    ramp_y = float(-lane_width - rng.uniform(5.5, 7.0))
    merge_start = float(rng.uniform(-28.0, -23.0))
    merge_end = float(rng.uniform(8.0, 13.0))
    merge_time = float(t[-1] * rng.uniform(0.48, 0.56))
    ego_speed = float(rng.uniform(8.0, 10.0))
    merging_speed = float(ego_speed - rng.uniform(0.0, 0.8))
    leader_speed = ego_speed
    merge_gap = float(rng.uniform(16.0, 22.0))
    leader_gap = float(rng.uniform(20.0, 28.0))

    ego_x = merge_end + merge_gap + ego_speed * (t - merge_time)
    ego_y = np.full(t.shape, main_y)
    leader_x = (
        merge_end + merge_gap + leader_gap + leader_speed * (t - merge_time)
    )
    leader_y = np.full(t.shape, main_y)
    merging_x = merge_end + merging_speed * (t - merge_time)
    merging_y = _merge_y(
        merging_x,
        main_y=main_y,
        ramp_y=ramp_y,
        merge_start=merge_start,
        merge_end=merge_end,
    )

    agents = [
        _agent_from_xy(
            agent_id=0,
            agent_type=AgentType.VEHICLE,
            x=ego_x,
            y=ego_y,
            dt=source.dt,
        ),
        _agent_from_xy(
            agent_id=1,
            agent_type=AgentType.VEHICLE,
            x=leader_x,
            y=leader_y,
            dt=source.dt,
        ),
        _agent_from_xy(
            agent_id=2,
            agent_type=AgentType.VEHICLE,
            x=merging_x,
            y=merging_y,
            dt=source.dt,
            valid=_lifecycle_mask(source.num_steps),
        ),
    ]

    x_min = float(min(ego_x.min(), merging_x.min()) - 12.0)
    x_max = float(max(leader_x.max(), merging_x.max()) + 12.0)
    ramp_x = np.concatenate(
        (
            np.linspace(x_min, merge_start, 61, endpoint=False),
            np.linspace(merge_start, merge_end, 121),
        )
    )
    ramp_x[-1] = merge_end
    ramp_center = _merge_y(
        ramp_x,
        main_y=main_y,
        ramp_y=ramp_y,
        merge_start=merge_start,
        merge_end=merge_end,
    )
    # The ramp's inner edge meets the main-road lower edge at the gore point.
    # Before that point both road branches are separately bounded; after it, the
    # opening between them is intentionally drivable.
    gore_center_y = -1.5 * lane_width
    gore_progress = (gore_center_y - ramp_y) / (main_y - ramp_y)
    gore_x = merge_start + _inverse_smoothstep(gore_progress) * (
        merge_end - merge_start
    )
    ramp_inner_x = np.concatenate(
        (
            np.linspace(x_min, merge_start, 61, endpoint=False),
            np.linspace(merge_start, gore_x, 81),
        )
    )
    ramp_inner_x[-1] = gore_x
    ramp_inner_y = (
        _merge_y(
            ramp_inner_x,
            main_y=main_y,
            ramp_y=ramp_y,
            merge_start=merge_start,
            merge_end=merge_end,
        )
        + lane_width / 2.0
    )
    ramp_inner_y[-1] = -lane_width
    ramp_outer_y = ramp_center - lane_width / 2.0
    ramp_outer_y[-1] = -lane_width
    map_features = [
        _line(MapType.LANE, (x_min, main_y), (x_max, main_y)),
        _line(
            MapType.LANE,
            (x_max, lane_width / 2.0),
            (x_min, lane_width / 2.0),
        ),
        _polyline(MapType.LANE, ramp_x, ramp_center),
        _line(
            MapType.ROAD_EDGE,
            (x_max, lane_width),
            (x_min, lane_width),
        ),
        _line(
            MapType.ROAD_EDGE,
            (x_min, -lane_width),
            (gore_x, -lane_width),
            points=61,
        ),
        _line(
            MapType.ROAD_EDGE,
            (merge_end, -lane_width),
            (x_max, -lane_width),
            points=61,
        ),
        _polyline(
            MapType.ROAD_EDGE,
            ramp_inner_x[::-1],
            ramp_inner_y[::-1],
        ),
        _polyline(
            MapType.ROAD_EDGE,
            ramp_x,
            ramp_outer_y,
        ),
    ]
    return source._scenario(
        index=index,
        kind=ScenarioKind.MERGE,
        agents=agents,
        map_features=map_features,
        tags=("merge", "interaction"),
        parameters={
            "ego_speed_mps": ego_speed,
            "merging_speed_mps": merging_speed,
            "merge_time_s": merge_time,
            "merge_gap_m": merge_gap,
            "merge_start_m": merge_start,
            "merge_end_m": merge_end,
            "gore_x_m": gore_x,
            "lane_width_m": lane_width,
        },
    )


def _left_turn_path(
    distance: np.ndarray,
    *,
    lane_width: float,
    radius: float,
    approach_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    inbound_y = -lane_width / 2.0
    outbound_x = lane_width / 2.0
    center_x = outbound_x - radius
    center_y = inbound_y + radius
    entry_x = center_x
    arc_length = radius * np.pi / 2.0

    x = np.empty(distance.shape, dtype=float)
    y = np.empty(distance.shape, dtype=float)
    approach = distance <= approach_length
    arc = (distance > approach_length) & (
        distance <= approach_length + arc_length
    )
    departure = ~(approach | arc)

    x[approach] = entry_x - approach_length + distance[approach]
    y[approach] = inbound_y

    theta = -np.pi / 2.0 + (distance[arc] - approach_length) / radius
    x[arc] = center_x + radius * np.cos(theta)
    y[arc] = center_y + radius * np.sin(theta)

    x[departure] = outbound_x
    y[departure] = center_y + (
        distance[departure] - approach_length - arc_length
    )
    return x, y


def _turn_scenario(
    source: SyntheticSource,
    index: int,
    rng: np.random.Generator,
) -> Scenario:
    t = source.timestamps
    lane_width = float(rng.uniform(3.4, 3.8))
    half_lane = lane_width / 2.0
    radius = float(rng.uniform(10.0, 13.0))
    ego_speed = float(rng.uniform(4.0, 5.5))
    turn_start_time = float(t[-1] * rng.uniform(0.28, 0.34))
    approach_length = ego_speed * turn_start_time
    opposing_speed = float(rng.uniform(7.0, 9.0))

    ego_x, ego_y = _left_turn_path(
        ego_speed * t,
        lane_width=lane_width,
        radius=radius,
        approach_length=approach_length,
    )
    center_x = half_lane - radius
    center_y = -half_lane + radius
    conflict_theta = float(
        np.arcsin(np.clip((half_lane - center_y) / radius, -1.0, 1.0))
    )
    conflict_x = center_x + radius * np.cos(conflict_theta)
    ego_conflict_time = (
        approach_length + radius * (conflict_theta + np.pi / 2.0)
    ) / ego_speed
    # The oncoming vehicle clears the conflict point before the ego begins its turn.
    opposing_gap = float(-rng.uniform(2.4, 3.0))
    opposing_conflict_time = ego_conflict_time + opposing_gap
    opposing_x = conflict_x - opposing_speed * (t - opposing_conflict_time)
    opposing_y = np.full(t.shape, half_lane)

    agents = [
        _agent_from_xy(
            agent_id=0,
            agent_type=AgentType.VEHICLE,
            x=ego_x,
            y=ego_y,
            dt=source.dt,
        ),
        _agent_from_xy(
            agent_id=1,
            agent_type=AgentType.VEHICLE,
            x=opposing_x,
            y=opposing_y,
            dt=source.dt,
            valid=_lifecycle_mask(
                source.num_steps,
                trim_start=False,
                trim_end=True,
            ),
        ),
    ]
    extent = float(
        max(
            45.0,
            np.max(np.abs(ego_x)) + 10.0,
            np.max(np.abs(ego_y)) + 10.0,
            np.max(np.abs(opposing_x)) + 10.0,
        )
    )
    turn_theta = np.linspace(-np.pi / 2.0, 0.0, 121)
    turn_connector = _polyline(
        MapType.LANE,
        center_x + radius * np.cos(turn_theta),
        center_y + radius * np.sin(turn_theta),
    )
    map_features = _intersection_map(extent, lane_width)
    map_features.append(turn_connector)
    return source._scenario(
        index=index,
        kind=ScenarioKind.TURN,
        agents=agents,
        map_features=map_features,
        tags=("intersection", "turn", "left_turn"),
        parameters={
            "ego_speed_mps": ego_speed,
            "turn_radius_m": radius,
            "approach_length_m": approach_length,
            "opposing_conflict_gap_s": opposing_gap,
            "lane_width_m": lane_width,
        },
    )


def _pedestrian_crossing_scenario(
    source: SyntheticSource,
    index: int,
    rng: np.random.Generator,
) -> Scenario:
    t = source.timestamps
    lane_width = float(rng.uniform(3.4, 3.8))
    half_lane = lane_width / 2.0
    arrival_time = float(t[-1] * rng.uniform(0.47, 0.53))
    pedestrian_offset = float(
        rng.choice(np.array([-1.0, 1.0])) * rng.uniform(1.6, 2.2)
    )
    ego_speed = float(rng.uniform(7.5, 9.0))
    pedestrian_speed = float(rng.uniform(1.2, 1.7))

    ego_x = ego_speed * (t - arrival_time)
    ego_y = np.full(t.shape, -half_lane)
    pedestrian_x = np.zeros(t.shape)
    pedestrian_y = -half_lane + pedestrian_speed * (
        t - arrival_time - pedestrian_offset
    )

    agents = [
        _agent_from_xy(
            agent_id=0,
            agent_type=AgentType.VEHICLE,
            x=ego_x,
            y=ego_y,
            dt=source.dt,
        ),
        _agent_from_xy(
            agent_id=1,
            agent_type=AgentType.PEDESTRIAN,
            x=pedestrian_x,
            y=pedestrian_y,
            dt=source.dt,
            valid=_lifecycle_mask(source.num_steps),
            length=0.6,
            width=0.6,
        ),
    ]
    x_min = float(ego_x.min() - 12.0)
    x_max = float(ego_x.max() + 12.0)
    map_features = _straight_map(x_min, x_max, lane_width)
    crossing_half_width = 2.0
    crosswalk_x = np.array(
        [
            -crossing_half_width,
            crossing_half_width,
            crossing_half_width,
            -crossing_half_width,
            -crossing_half_width,
        ]
    )
    crosswalk_y = np.array(
        [-lane_width, -lane_width, lane_width, lane_width, -lane_width]
    )
    map_features.append(_polyline(MapType.CROSSWALK, crosswalk_x, crosswalk_y))
    return source._scenario(
        index=index,
        kind=ScenarioKind.PEDESTRIAN_CROSSING,
        agents=agents,
        map_features=map_features,
        tags=("pedestrian_present", "pedestrian_crossing", "straight_road"),
        parameters={
            "ego_speed_mps": ego_speed,
            "pedestrian_speed_mps": pedestrian_speed,
            "pedestrian_arrival_offset_s": pedestrian_offset,
            "lane_width_m": lane_width,
        },
    )


_BUILDERS = {
    ScenarioKind.FOLLOWING: _following_scenario,
    ScenarioKind.INTERSECTION: _intersection_scenario,
    ScenarioKind.MERGE: _merge_scenario,
    ScenarioKind.TURN: _turn_scenario,
    ScenarioKind.PEDESTRIAN_CROSSING: _pedestrian_crossing_scenario,
}
