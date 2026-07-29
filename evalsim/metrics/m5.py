"""Pure, source-neutral metric families pre-registered for EvalSim M5.

The metric implementations in this module consume only the public ``Scenario`` and
``Rollout`` contracts.  WOMD and Waymax remain optional adapters outside this layer.
Every metric first reduces to exactly one scalar per scenario; cross-scenario
statistics are owned by :mod:`evalsim.stats`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from evalsim.contracts import (
    AgentType,
    MapType,
    Metric,
    MetricEligibility,
    MetricResult,
    MetricSpec,
    Rollout,
    Scenario,
)

M5_METRIC_VERSION = "1.0.0"
_FLOAT32_DT = np.float32(0.1)
_FLOAT32_DT_SQUARED = np.float32(0.1**2)
_FLOAT32_SPEED_THRESHOLD = np.float32(0.6)
_FLOAT32_ACCEL_THRESHOLD = np.float32(10.401)
_FLOAT32_CURVATURE_THRESHOLD = np.float32(0.301)
_FLOAT32_PI = np.float32(np.pi)
_FLOAT32_TWO_PI = np.float32(2.0 * np.pi)
_SERIES_FIELDS = ("x", "y", "heading", "vx", "vy")


class MetricInputError(ValueError):
    """Structural execution failure that must not be converted to missingness."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class _Outcome:
    distribution: tuple[float, ...]
    total_components: int
    value: float | None = None
    invalid_reason: str | None = None
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.total_components < len(self.distribution):
            raise RuntimeError("eligible components cannot exceed total components")
        if self.invalid_reason is None and not self.distribution:
            raise RuntimeError("a valid metric outcome needs at least one component")
        if self.invalid_reason is not None and self.distribution:
            raise RuntimeError("an invalid metric outcome cannot retain components")


def _fail(code: str, message: str) -> None:
    raise MetricInputError(code, message)


def _current_index(scenario: Scenario) -> int:
    value = scenario.metadata.get("current_index", 0)
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= int(value) < scenario.num_steps
    ):
        _fail(
            "current_index_invalid",
            "Scenario.metadata['current_index'] must index the scenario horizon",
        )
    return int(value)


def _validate_source(scenario: Scenario) -> int:
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if not scenario.scenario_id:
        _fail("scenario_identity_invalid", "scenario_id must be non-empty")
    if scenario.num_steps < 1 or scenario.num_agents < 1:
        _fail("empty_scenario", "a scenario needs at least one frame and one agent")
    if (
        not np.all(np.isfinite(scenario.timestamps))
        or (
            scenario.num_steps > 1
            and not np.all(np.diff(scenario.timestamps) > 0.0)
        )
    ):
        _fail("timestamps_invalid", "timestamps must be finite and strictly increasing")

    identifiers = [agent.id for agent in scenario.agents]
    if len(set(identifiers)) != len(identifiers):
        _fail("agent_identity_invalid", "scenario agent IDs must be unique")
    for index, logged in enumerate(scenario.agents):
        if (
            not math.isfinite(float(logged.length))
            or not math.isfinite(float(logged.width))
            or logged.length <= 0.0
            or logged.width <= 0.0
        ):
            _fail(
                "agent_dimension_invalid",
                f"agent dimensions are invalid at contract index {index}",
            )
        for field_name in _SERIES_FIELDS:
            source_values = np.asarray(getattr(logged, field_name))
            if not np.all(np.isfinite(source_values[logged.valid])):
                _fail(
                    "non_finite_state",
                    f"valid {field_name} values must be finite at contract index {index}",
                )
    return _current_index(scenario)


def _validate_pair(scenario: Scenario, rollout: Rollout) -> int:
    current = _validate_source(scenario)
    if not isinstance(rollout, Rollout):
        raise TypeError("rollout must be a Rollout")
    if rollout.scenario_id != scenario.scenario_id:
        _fail("scenario_identity_mismatch", "scenario identities must match")
    if rollout.num_steps != scenario.num_steps:
        _fail("horizon_mismatch", "scenario and rollout horizons must match")
    if rollout.num_agents != scenario.num_agents:
        _fail("agent_count_mismatch", "scenario and rollout agent counts must match")
    if not np.array_equal(rollout.timestamps, scenario.timestamps):
        _fail("timestamp_mismatch", "scenario and rollout timestamps must match exactly")

    for index, (logged, candidate) in enumerate(
        zip(scenario.agents, rollout.agents, strict=True)
    ):
        if logged.id != candidate.id or logged.type != candidate.type:
            _fail(
                "agent_identity_mismatch",
                f"agent identity/order differs at contract index {index}",
            )
        if (
            logged.length != candidate.length
            or logged.width != candidate.width
        ):
            _fail(
                "agent_dimension_mismatch",
                f"agent dimensions differ or are invalid at contract index {index}",
            )
        if not np.array_equal(logged.valid, candidate.valid):
            _fail(
                "validity_mismatch",
                f"source and rollout validity differ at contract index {index}",
            )
        for field_name in _SERIES_FIELDS:
            candidate_values = np.asarray(getattr(candidate, field_name))
            if not np.all(np.isfinite(candidate_values[logged.valid])):
                _fail(
                    "non_finite_state",
                    f"valid {field_name} values must be finite at contract index {index}",
                )
    return current


def _future_frames(scenario: Scenario) -> range:
    return range(_current_index(scenario) + 1, scenario.num_steps)


def _non_ego_indices(scenario: Scenario) -> tuple[int, ...]:
    return tuple(
        index for index in range(scenario.num_agents) if index != scenario.ego_index
    )


def _vehicle_indices(scenario: Scenario) -> tuple[int, ...]:
    return tuple(
        index
        for index, agent in enumerate(scenario.agents)
        if index != scenario.ego_index and agent.type == AgentType.VEHICLE
    )


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(float(value) for value in values) / len(values)


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _valid_outcome(
    values: list[float] | tuple[float, ...],
    *,
    total_components: int,
    reducer: str = "mean",
    details: dict[str, Any] | None = None,
) -> _Outcome:
    distribution = tuple(float(value) for value in values)
    if not distribution or not all(math.isfinite(value) for value in distribution):
        raise RuntimeError("valid outcomes require a non-empty finite distribution")
    if reducer == "mean":
        value = _mean(distribution)
    elif reducer == "min":
        value = min(distribution)
    else:  # pragma: no cover - internal programming error
        raise RuntimeError(f"unsupported reducer {reducer!r}")
    return _Outcome(
        distribution=distribution,
        total_components=total_components,
        value=value,
        details=details or {},
    )


def _invalid_outcome(reason: str, *, total_components: int) -> _Outcome:
    return _Outcome(
        distribution=(),
        total_components=total_components,
        invalid_reason=reason,
        details={},
    )


def _source_invalid_reason(metric_name: str, scenario: Scenario) -> str | None:
    """Classify eligibility without receiving candidate values or metric outcomes."""

    current = _current_index(scenario)
    future = range(current + 1, scenario.num_steps)
    targets = _non_ego_indices(scenario)

    if metric_name in {
        "position_error_m",
        "speed_error_mps",
        "oriented_box_overlap_rate",
    }:
        if any(
            scenario.agents[index].valid[frame]
            for frame in future
            for index in targets
        ):
            return None
        return "no_eligible_target_frame"

    if metric_name == "waymax_kinematic_infeasibility_rate":
        vehicles = _vehicle_indices(scenario)
        if any(
            _contiguous_pair(scenario, index, frame)
            for frame in future
            for index in vehicles
        ):
            return None
        return "no_eligible_vehicle_transition"

    if metric_name in {
        "acceleration_error_mps2",
        "yaw_rate_error_radps",
        "kinematic_continuity_residual_m",
    }:
        if any(
            _contiguous_pair(scenario, index, frame)
            for frame in future
            for index in targets
        ):
            return None
        return "no_contiguous_valid_window"

    if metric_name == "jerk_error_mps3":
        if any(
            frame >= 2
            and bool(
                np.all(
                    scenario.agents[index].valid[
                        frame - 2 : frame + 1
                    ]
                )
            )
            for frame in future
            for index in targets
        ):
            return None
        return "no_contiguous_valid_window"

    if metric_name in {
        "minimum_center_distance_m",
        "constant_velocity_ttc_cap_5s",
    }:
        if any(_future_valid_pairs(scenario, frame) for frame in future):
            return None
        return "no_valid_object_pair"

    if metric_name in {
        "lane_center_distance_m",
        "lane_heading_disagreement_rad",
    }:
        if not _lane_segments(scenario):
            return "no_supported_lane"
        vehicles = _vehicle_indices(scenario)
        if any(
            scenario.agents[index].valid[frame]
            for frame in future
            for index in vehicles
        ):
            return None
        return "no_eligible_vehicle_frame"

    if metric_name == "lifecycle_reentry_per_agent":
        return None if targets else "no_non_ego_agent"

    raise RuntimeError(f"no source eligibility rule for metric {metric_name!r}")


class _BaseMetric(Metric):
    """Shared result construction; subclasses provide a pure scenario outcome."""

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        raise NotImplementedError

    def eligibility(
        self,
        scenario: Scenario,
    ) -> MetricEligibility:
        _validate_source(scenario)
        if self.spec.name == "waymax_kinematic_infeasibility_rate":
            _assert_100ms_cadence(scenario, _current_index(scenario))
        reason = _source_invalid_reason(self.spec.name, scenario)
        if reason is None:
            return MetricEligibility.accepted()
        return MetricEligibility.rejected(reason)

    def compute(self, scenario: Scenario, rollout: Rollout) -> MetricResult:
        outcome = self._outcome(scenario, rollout)
        if outcome.invalid_reason is not None:
            return MetricResult(
                metric_name=self.spec.name,
                metric_version=self.spec.version,
                scenario_id=scenario.scenario_id,
                value=None,
                distribution=(),
                valid=False,
                invalid_reason=outcome.invalid_reason,
                eligible_components=0,
                total_components=outcome.total_components,
                details=outcome.details or {},
            )
        return MetricResult(
            metric_name=self.spec.name,
            metric_version=self.spec.version,
            scenario_id=scenario.scenario_id,
            value=outcome.value,
            distribution=outcome.distribution,
            valid=True,
            invalid_reason=None,
            eligible_components=len(outcome.distribution),
            total_components=outcome.total_components,
            details=outcome.details or {},
        )


def canonical_float32_view(
    scenario: Scenario,
    rollout: Rollout,
) -> dict[str, np.ndarray]:
    """Return the exact frame-major float32 view shared with Waymax parity.

    Motion has shape ``[frames, agents]`` and dimensions have shape ``[agents]``.
    Every returned array owns its storage.
    """

    _validate_pair(scenario, rollout)

    def stack(agents: list[Any], field: str, dtype: Any) -> np.ndarray:
        return np.array(
            [np.asarray(getattr(agent, field), dtype=dtype) for agent in agents],
            dtype=dtype,
        ).T.copy()

    view = {
        f"log_{field}": stack(scenario.agents, field, np.float32)
        for field in _SERIES_FIELDS
    }
    view.update(
        {
            f"sim_{field}": stack(rollout.agents, field, np.float32)
            for field in _SERIES_FIELDS
        }
    )
    view["log_valid"] = stack(scenario.agents, "valid", np.bool_)
    view["sim_valid"] = stack(rollout.agents, "valid", np.bool_)
    view["length"] = np.array(
        [agent.length for agent in rollout.agents],
        dtype=np.float32,
    )
    view["width"] = np.array(
        [agent.width for agent in rollout.agents],
        dtype=np.float32,
    )
    return view


def position_divergence_components(
    scenario: Scenario,
    rollout: Rollout,
) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 position divergence and its exact joint-validity mask."""

    view = canonical_float32_view(scenario, rollout)
    dx = np.subtract(view["sim_x"], view["log_x"], dtype=np.float32)
    dy = np.subtract(view["sim_y"], view["log_y"], dtype=np.float32)
    squared = np.add(
        np.multiply(dx, dx, dtype=np.float32),
        np.multiply(dy, dy, dtype=np.float32),
        dtype=np.float32,
    )
    values = np.sqrt(squared, dtype=np.float32)
    mask = view["sim_valid"] & view["log_valid"]
    return values, mask


def _float32_box_corners(box: np.ndarray) -> np.ndarray:
    """Mirror pinned Waymax ``geometry.corners_from_bbox`` operation order."""

    # Scalar libm followed by an explicit float32 cast tracks XLA CPU trigonometric
    # rounding more closely than NumPy's float32 ufunc on near-contact boxes. Exact
    # discrete agreement is still verified against the pinned native metric.
    cosine = np.float32(math.cos(float(box[4])))
    sine = np.float32(math.sin(float(box[4])))
    half = np.float32(0.5)
    length_cosine = np.multiply(
        np.multiply(box[2], half, dtype=np.float32),
        cosine,
        dtype=np.float32,
    )
    length_sine = np.multiply(
        np.multiply(box[2], half, dtype=np.float32),
        sine,
        dtype=np.float32,
    )
    width_cosine = np.multiply(
        np.multiply(box[3], half, dtype=np.float32),
        cosine,
        dtype=np.float32,
    )
    width_sine = np.multiply(
        np.multiply(box[3], half, dtype=np.float32),
        sine,
        dtype=np.float32,
    )
    corners = np.array(
        [
            [
                np.add(length_cosine, width_sine, dtype=np.float32),
                np.subtract(length_sine, width_cosine, dtype=np.float32),
            ],
            [
                np.subtract(length_cosine, width_sine, dtype=np.float32),
                np.add(length_sine, width_cosine, dtype=np.float32),
            ],
            [
                np.negative(
                    np.add(length_cosine, width_sine, dtype=np.float32),
                    dtype=np.float32,
                ),
                np.subtract(width_cosine, length_sine, dtype=np.float32),
            ],
            [
                np.subtract(width_sine, length_cosine, dtype=np.float32),
                np.negative(
                    np.add(length_sine, width_cosine, dtype=np.float32),
                    dtype=np.float32,
                ),
            ],
        ],
        dtype=np.float32,
    )
    return np.add(corners, box[np.newaxis, :2], dtype=np.float32)


def _positive_projection_overlap(
    first: np.ndarray,
    second: np.ndarray,
) -> bool:
    """Mirror pinned Waymax ``geometry.has_overlap`` for one axis pair."""

    cosine = np.float32(math.cos(float(first[4])))
    sine = np.float32(math.sin(float(first[4])))
    normals_t = np.array(
        [[cosine, np.negative(sine)], [sine, cosine]],
        dtype=np.float32,
    )
    projection_first = np.matmul(
        _float32_box_corners(first),
        normals_t,
    )
    projection_second = np.matmul(
        _float32_box_corners(second),
        normals_t,
    )
    distance = np.subtract(
        np.minimum(
            np.max(projection_first, axis=0),
            np.max(projection_second, axis=0),
        ),
        np.maximum(
            np.min(projection_first, axis=0),
            np.min(projection_second, axis=0),
        ),
        dtype=np.float32,
    )
    return bool(np.all(distance > np.float32(0.0)))


def _boxes_overlap_strict(
    x_a: np.float32,
    y_a: np.float32,
    heading_a: np.float32,
    length_a: np.float32,
    width_a: np.float32,
    x_b: np.float32,
    y_b: np.float32,
    heading_b: np.float32,
    length_b: np.float32,
    width_b: np.float32,
) -> bool:
    first = np.array(
        [x_a, y_a, length_a, width_a, heading_a],
        dtype=np.float32,
    )
    second = np.array(
        [x_b, y_b, length_b, width_b, heading_b],
        dtype=np.float32,
    )
    return _positive_projection_overlap(
        first,
        second,
    ) and _positive_projection_overlap(second, first)


def oriented_box_overlap_components(
    scenario: Scenario,
    rollout: Rollout,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-target overlap flags and target-validity masks."""

    view = canonical_float32_view(scenario, rollout)
    frames, agents = view["sim_x"].shape
    flags = np.zeros((frames, agents), dtype=bool)
    target_valid = np.array(view["sim_valid"], dtype=bool, copy=True)
    for frame in range(frames):
        valid_indices = np.flatnonzero(target_valid[frame])
        for left_offset, left in enumerate(valid_indices):
            for right in valid_indices[left_offset + 1 :]:
                if _boxes_overlap_strict(
                    view["sim_x"][frame, left],
                    view["sim_y"][frame, left],
                    view["sim_heading"][frame, left],
                    view["length"][left],
                    view["width"][left],
                    view["sim_x"][frame, right],
                    view["sim_y"][frame, right],
                    view["sim_heading"][frame, right],
                    view["length"][right],
                    view["width"][right],
                ):
                    flags[frame, left] = True
                    flags[frame, right] = True
    return flags, target_valid


def _wrap_float32(delta: np.ndarray) -> np.ndarray:
    """Mirror pinned Waymax ``geometry.wrap_yaws`` in float32."""

    return np.subtract(
        np.remainder(
            np.add(delta, _FLOAT32_PI, dtype=np.float32),
            _FLOAT32_TWO_PI,
        ),
        _FLOAT32_PI,
        dtype=np.float32,
    )


def kinematic_infeasibility_flags(
    acceleration: np.ndarray,
    steering_curvature: np.ndarray,
) -> np.ndarray:
    """Apply the pinned strict float32 feasibility thresholds."""

    acceleration = np.asarray(acceleration)
    steering_curvature = np.asarray(steering_curvature)
    if (
        acceleration.dtype != np.float32
        or steering_curvature.dtype != np.float32
        or acceleration.shape != steering_curvature.shape
        or not np.all(np.isfinite(acceleration))
        or not np.all(np.isfinite(steering_curvature))
    ):
        raise ValueError(
            "kinematic threshold inputs must be same-shape finite float32 arrays"
        )
    return (
        (np.abs(acceleration) > _FLOAT32_ACCEL_THRESHOLD)
        | (np.abs(steering_curvature) > _FLOAT32_CURVATURE_THRESHOLD)
    )


def kinematic_infeasibility_components(
    scenario: Scenario,
    rollout: Rollout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pinned inverse-bicycle flags, mask, acceleration, and curvature.

    Arrays have shape ``[transitions, agents]``; transition row zero ends at frame one.
    """

    view = canonical_float32_view(scenario, rollout)
    old_speed = np.sqrt(
        np.add(
            np.multiply(
                view["sim_vx"][:-1],
                view["sim_vx"][:-1],
                dtype=np.float32,
            ),
            np.multiply(
                view["sim_vy"][:-1],
                view["sim_vy"][:-1],
                dtype=np.float32,
            ),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    new_speed = np.sqrt(
        np.add(
            np.multiply(
                view["sim_vx"][1:],
                view["sim_vx"][1:],
                dtype=np.float32,
            ),
            np.multiply(
                view["sim_vy"][1:],
                view["sim_vy"][1:],
                dtype=np.float32,
            ),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    accel = np.divide(
        np.subtract(new_speed, old_speed, dtype=np.float32),
        _FLOAT32_DT,
        dtype=np.float32,
    )
    velocity_yaw = np.arctan2(
        view["sim_vy"][1:],
        view["sim_vx"][1:],
        dtype=np.float32,
    )
    candidate_new_yaw = _wrap_float32(view["sim_heading"][1:])
    old_yaw = _wrap_float32(view["sim_heading"][:-1])
    new_yaw = np.where(
        np.abs(new_speed) > _FLOAT32_SPEED_THRESHOLD,
        velocity_yaw,
        candidate_new_yaw,
    ).astype(np.float32, copy=False)
    delta_yaw = _wrap_float32(
        np.subtract(new_yaw, old_yaw, dtype=np.float32)
    )
    distance = np.add(
        np.multiply(old_speed, _FLOAT32_DT, dtype=np.float32),
        np.multiply(
            np.multiply(np.float32(0.5), accel, dtype=np.float32),
            _FLOAT32_DT_SQUARED,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.divide(delta_yaw, distance, dtype=np.float32)
    suppress = (old_speed < _FLOAT32_SPEED_THRESHOLD) | (
        new_speed < _FLOAT32_SPEED_THRESHOLD
    )
    curvature = np.where(suppress, np.float32(0.0), curvature).astype(
        np.float32,
        copy=False,
    )
    action_valid = view["sim_valid"][:-1] & view["sim_valid"][1:]
    if np.any(action_valid & ~np.isfinite(accel)):
        _fail("kinematic_non_finite", "valid acceleration must be finite")
    if np.any(action_valid & ~np.isfinite(curvature)):
        _fail("kinematic_non_finite", "valid steering curvature must be finite")
    accel = np.where(action_valid, accel, np.float32(0.0)).astype(
        np.float32,
        copy=False,
    )
    curvature = np.where(
        action_valid,
        curvature,
        np.float32(0.0),
    ).astype(np.float32, copy=False)
    flags = kinematic_infeasibility_flags(accel, curvature) & action_valid
    return flags, action_valid, accel, curvature


def _assert_100ms_cadence(scenario: Scenario, current: int) -> None:
    timestamps = np.asarray(scenario.timestamps, dtype=np.float64)
    deltas_micros = np.diff(timestamps[current:]) * 1_000_000.0
    rounded = np.rint(deltas_micros)
    if np.any(np.abs(deltas_micros - rounded) > 1e-6):
        _fail(
            "cadence_drift",
            "timestamp deltas must agree with an integral microsecond cadence",
        )
    if np.any(rounded.astype(np.int64) != 100_000):
        _fail("cadence_drift", "kinematic metric requires exact 100000 us cadence")


class PositionErrorMetric(_BaseMetric):
    spec = MetricSpec(
        name="position_error_m",
        version=M5_METRIC_VERSION,
        value_unit="m",
        direction="lower",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="post-current future frames",
        eligibility="at least one source-valid non-ego future target-frame",
        invalid_reason_codes=("no_eligible_target_frame",),
        required_fields=("valid", "x", "y"),
        known_failure_modes=("privileged recorded-future reference",),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        values, mask = position_divergence_components(scenario, rollout)
        targets = _non_ego_indices(scenario)
        total = len(targets) * max(0, scenario.num_steps - current - 1)
        components = [
            float(values[frame, agent])
            for frame in range(current + 1, scenario.num_steps)
            for agent in targets
            if mask[frame, agent]
        ]
        if not components:
            return _invalid_outcome(
                "no_eligible_target_frame",
                total_components=total,
            )
        return _valid_outcome(components, total_components=total)


class SpeedErrorMetric(_BaseMetric):
    spec = MetricSpec(
        name="speed_error_mps",
        version=M5_METRIC_VERSION,
        value_unit="m/s",
        direction="lower",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="post-current future frames",
        eligibility="at least one source-valid non-ego future target-frame",
        invalid_reason_codes=("no_eligible_target_frame",),
        required_fields=("valid", "vx", "vy"),
        known_failure_modes=("speed omits signed longitudinal direction",),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        targets = _non_ego_indices(scenario)
        total = len(targets) * max(0, scenario.num_steps - current - 1)
        components: list[float] = []
        for frame in range(current + 1, scenario.num_steps):
            for index in targets:
                source = scenario.agents[index]
                if source.valid[frame]:
                    candidate = rollout.agents[index]
                    components.append(
                        abs(
                            math.hypot(candidate.vx[frame], candidate.vy[frame])
                            - math.hypot(source.vx[frame], source.vy[frame])
                        )
                    )
        if not components:
            return _invalid_outcome(
                "no_eligible_target_frame",
                total_components=total,
            )
        return _valid_outcome(components, total_components=total)


class OrientedBoxOverlapRateMetric(_BaseMetric):
    spec = MetricSpec(
        name="oriented_box_overlap_rate",
        version=M5_METRIC_VERSION,
        value_unit="fraction",
        direction="lower",
        aggregation="rate",
        agent_scope="world",
        evaluation_window="post-current future frames",
        eligibility="at least one source-valid non-ego future target-frame",
        invalid_reason_codes=("no_eligible_target_frame",),
        required_fields=("valid", "x", "y", "heading", "length", "width"),
        known_failure_modes=(
            "target-frame rate is not a unique collision-pair count",
            "strict geometric overlap has no severity semantics",
            (
                "float32 trigonometric backends can flip a zero-margin strict "
                "overlap; bounded native parity is mandatory"
            ),
        ),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        flags, target_valid = oriented_box_overlap_components(scenario, rollout)
        targets = _non_ego_indices(scenario)
        total = len(targets) * max(0, scenario.num_steps - current - 1)
        components = [
            float(flags[frame, agent])
            for frame in range(current + 1, scenario.num_steps)
            for agent in targets
            if target_valid[frame, agent]
        ]
        if not components:
            return _invalid_outcome(
                "no_eligible_target_frame",
                total_components=total,
            )
        return _valid_outcome(components, total_components=total)


class WaymaxKinematicInfeasibilityRateMetric(_BaseMetric):
    spec = MetricSpec(
        name="waymax_kinematic_infeasibility_rate",
        version=M5_METRIC_VERSION,
        value_unit="fraction",
        direction="lower",
        aggregation="rate",
        agent_scope="world",
        evaluation_window="post-current future transitions",
        eligibility="at least one contiguous-valid non-ego vehicle future transition",
        invalid_reason_codes=("no_eligible_vehicle_transition",),
        required_fields=(
            "valid",
            "heading",
            "vx",
            "vy",
            "type",
            "fixed_100000_us_source_cadence",
        ),
        known_failure_modes=(
            "inverse bicycle thresholds are weak semantics for non-vehicles",
        ),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        _assert_100ms_cadence(scenario, current)
        flags, valid, _, _ = kinematic_infeasibility_components(scenario, rollout)
        vehicles = _vehicle_indices(scenario)
        total = len(vehicles) * max(0, scenario.num_steps - current - 1)
        components = [
            float(flags[frame - 1, agent])
            for frame in range(current + 1, scenario.num_steps)
            for agent in vehicles
            if valid[frame - 1, agent]
        ]
        if not components:
            return _invalid_outcome(
                "no_eligible_vehicle_transition",
                total_components=total,
            )
        return _valid_outcome(components, total_components=total)


def _contiguous_pair(
    scenario: Scenario,
    index: int,
    frame: int,
) -> bool:
    return bool(
        frame >= 1
        and scenario.agents[index].valid[frame - 1]
        and scenario.agents[index].valid[frame]
    )


class AccelerationErrorMetric(_BaseMetric):
    spec = MetricSpec(
        name="acceleration_error_mps2",
        version=M5_METRIC_VERSION,
        value_unit="m/s^2",
        direction="lower",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="post-current future transitions",
        eligibility="at least one contiguous-valid future velocity interval",
        invalid_reason_codes=("no_contiguous_valid_window",),
        required_fields=("valid", "vx", "vy", "timestamps"),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        targets = _non_ego_indices(scenario)
        total = len(targets) * max(0, scenario.num_steps - current - 1)
        components: list[float] = []
        for frame in range(current + 1, scenario.num_steps):
            dt = float(scenario.timestamps[frame] - scenario.timestamps[frame - 1])
            for index in targets:
                if not _contiguous_pair(scenario, index, frame):
                    continue
                source = scenario.agents[index]
                candidate = rollout.agents[index]
                source_ax = (source.vx[frame] - source.vx[frame - 1]) / dt
                source_ay = (source.vy[frame] - source.vy[frame - 1]) / dt
                candidate_ax = (
                    candidate.vx[frame] - candidate.vx[frame - 1]
                ) / dt
                candidate_ay = (
                    candidate.vy[frame] - candidate.vy[frame - 1]
                ) / dt
                components.append(
                    math.hypot(
                        candidate_ax - source_ax,
                        candidate_ay - source_ay,
                    )
                )
        if not components:
            return _invalid_outcome(
                "no_contiguous_valid_window",
                total_components=total,
            )
        return _valid_outcome(components, total_components=total)


class JerkErrorMetric(_BaseMetric):
    spec = MetricSpec(
        name="jerk_error_mps3",
        version=M5_METRIC_VERSION,
        value_unit="m/s^3",
        direction="lower",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="post-current future frames",
        eligibility="at least one contiguous pair of velocity intervals",
        invalid_reason_codes=("no_contiguous_valid_window",),
        required_fields=("valid", "vx", "vy", "timestamps"),
    )

    @staticmethod
    def _jerk(agent: Any, frame: int, dt_previous: float, dt_current: float) -> tuple[float, float]:
        previous_ax = (agent.vx[frame - 1] - agent.vx[frame - 2]) / dt_previous
        previous_ay = (agent.vy[frame - 1] - agent.vy[frame - 2]) / dt_previous
        current_ax = (agent.vx[frame] - agent.vx[frame - 1]) / dt_current
        current_ay = (agent.vy[frame] - agent.vy[frame - 1]) / dt_current
        midpoint_dt = (dt_previous + dt_current) / 2.0
        return (
            (current_ax - previous_ax) / midpoint_dt,
            (current_ay - previous_ay) / midpoint_dt,
        )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        targets = _non_ego_indices(scenario)
        first = max(current + 1, 2)
        total = len(targets) * max(0, scenario.num_steps - first)
        components: list[float] = []
        for frame in range(first, scenario.num_steps):
            dt_previous = float(
                scenario.timestamps[frame - 1] - scenario.timestamps[frame - 2]
            )
            dt_current = float(
                scenario.timestamps[frame] - scenario.timestamps[frame - 1]
            )
            for index in targets:
                source = scenario.agents[index]
                if not bool(np.all(source.valid[frame - 2 : frame + 1])):
                    continue
                source_jerk = self._jerk(
                    source,
                    frame,
                    dt_previous,
                    dt_current,
                )
                candidate_jerk = self._jerk(
                    rollout.agents[index],
                    frame,
                    dt_previous,
                    dt_current,
                )
                components.append(
                    math.hypot(
                        candidate_jerk[0] - source_jerk[0],
                        candidate_jerk[1] - source_jerk[1],
                    )
                )
        if not components:
            return _invalid_outcome(
                "no_contiguous_valid_window",
                total_components=total,
            )
        return _valid_outcome(components, total_components=total)


class YawRateErrorMetric(_BaseMetric):
    spec = MetricSpec(
        name="yaw_rate_error_radps",
        version=M5_METRIC_VERSION,
        value_unit="rad/s",
        direction="lower",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="post-current future transitions",
        eligibility="at least one contiguous-valid future heading interval",
        invalid_reason_codes=("no_contiguous_valid_window",),
        required_fields=("valid", "heading", "timestamps"),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        targets = _non_ego_indices(scenario)
        total = len(targets) * max(0, scenario.num_steps - current - 1)
        components: list[float] = []
        for frame in range(current + 1, scenario.num_steps):
            dt = float(scenario.timestamps[frame] - scenario.timestamps[frame - 1])
            for index in targets:
                if not _contiguous_pair(scenario, index, frame):
                    continue
                source = scenario.agents[index]
                candidate = rollout.agents[index]
                source_rate = _wrap_angle(
                    source.heading[frame] - source.heading[frame - 1]
                ) / dt
                candidate_rate = _wrap_angle(
                    candidate.heading[frame] - candidate.heading[frame - 1]
                ) / dt
                components.append(abs(candidate_rate - source_rate))
        if not components:
            return _invalid_outcome(
                "no_contiguous_valid_window",
                total_components=total,
            )
        return _valid_outcome(components, total_components=total)


def _future_valid_pairs(
    scenario: Scenario,
    frame: int,
) -> tuple[tuple[int, int], ...]:
    valid = [
        index
        for index, agent in enumerate(scenario.agents)
        if agent.valid[frame]
    ]
    return tuple(
        (left, right)
        for left_offset, left in enumerate(valid)
        for right in valid[left_offset + 1 :]
        if left != scenario.ego_index or right != scenario.ego_index
    )


class MinimumCenterDistanceMetric(_BaseMetric):
    spec = MetricSpec(
        name="minimum_center_distance_m",
        version=M5_METRIC_VERSION,
        value_unit="m",
        direction="neutral",
        aggregation="minimum",
        agent_scope="all",
        evaluation_window="post-current future frames",
        eligibility="at least one valid future object pair",
        invalid_reason_codes=("no_valid_object_pair",),
        required_fields=("valid", "x", "y"),
        known_failure_modes=("center distance is not box clearance",),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        total = max(0, scenario.num_steps - current - 1)
        frame_minima: list[float] = []
        pair_count = 0
        for frame in range(current + 1, scenario.num_steps):
            distances: list[float] = []
            for left, right in _future_valid_pairs(scenario, frame):
                pair_count += 1
                distances.append(
                    math.hypot(
                        rollout.agents[right].x[frame]
                        - rollout.agents[left].x[frame],
                        rollout.agents[right].y[frame]
                        - rollout.agents[left].y[frame],
                    )
                )
            if distances:
                frame_minima.append(min(distances))
        if not frame_minima:
            return _invalid_outcome(
                "no_valid_object_pair",
                total_components=total,
            )
        return _valid_outcome(
            frame_minima,
            total_components=total,
            reducer="min",
            details={"evaluated_pair_count": pair_count},
        )


def constant_velocity_disc_ttc(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    combined_radius: float,
    *,
    cap_seconds: float = 5.0,
) -> float:
    """First nonnegative constant-velocity disc-contact time, right-censored."""

    r = np.asarray(relative_position, dtype=np.float64)
    v = np.asarray(relative_velocity, dtype=np.float64)
    radius = float(combined_radius)
    if r.shape != (2,) or v.shape != (2,):
        raise ValueError("relative position and velocity must each have shape [2]")
    if (
        not np.all(np.isfinite(r))
        or not np.all(np.isfinite(v))
        or not math.isfinite(radius)
        or radius <= 0.0
        or not math.isfinite(cap_seconds)
        or cap_seconds <= 0.0
    ):
        raise ValueError("TTC inputs must be finite with positive radius and cap")
    a = float(np.dot(v, v))
    b = 2.0 * float(np.dot(r, v))
    c = float(np.dot(r, r)) - radius * radius
    if c <= 0.0:
        return 0.0
    if a == 0.0:
        return cap_seconds
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return cap_seconds
    root_term = math.sqrt(discriminant)
    roots = ((-b - root_term) / (2.0 * a), (-b + root_term) / (2.0 * a))
    nonnegative = [root for root in roots if root >= 0.0]
    if not nonnegative:
        return cap_seconds
    return min(min(nonnegative), cap_seconds)


class ConstantVelocityTTCCapMetric(_BaseMetric):
    spec = MetricSpec(
        name="constant_velocity_ttc_cap_5s",
        version=M5_METRIC_VERSION,
        value_unit="s",
        direction="neutral",
        aggregation="minimum",
        agent_scope="all",
        evaluation_window="post-current future frames",
        eligibility="at least one valid future object pair",
        invalid_reason_codes=("no_valid_object_pair",),
        required_fields=("valid", "x", "y", "vx", "vy", "length", "width"),
        known_failure_modes=(
            "disc proxy is not lane-aware",
            "five-second right censoring hides longer contact times",
        ),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        total = max(0, scenario.num_steps - current - 1)
        frame_minima: list[float] = []
        pair_count = 0
        for frame in range(current + 1, scenario.num_steps):
            values: list[float] = []
            for left, right in _future_valid_pairs(scenario, frame):
                pair_count += 1
                left_agent = rollout.agents[left]
                right_agent = rollout.agents[right]
                left_radius = 0.5 * math.hypot(
                    left_agent.length,
                    left_agent.width,
                )
                right_radius = 0.5 * math.hypot(
                    right_agent.length,
                    right_agent.width,
                )
                values.append(
                    constant_velocity_disc_ttc(
                        np.array(
                            [
                                right_agent.x[frame] - left_agent.x[frame],
                                right_agent.y[frame] - left_agent.y[frame],
                            ]
                        ),
                        np.array(
                            [
                                right_agent.vx[frame] - left_agent.vx[frame],
                                right_agent.vy[frame] - left_agent.vy[frame],
                            ]
                        ),
                        left_radius + right_radius,
                    )
                )
            if values:
                frame_minima.append(min(values))
        if not frame_minima:
            return _invalid_outcome(
                "no_valid_object_pair",
                total_components=total,
            )
        return _valid_outcome(
            frame_minima,
            total_components=total,
            reducer="min",
            details={"evaluated_pair_count": pair_count},
        )


def _lane_segments(
    scenario: Scenario,
) -> tuple[tuple[np.ndarray, np.ndarray, int, int], ...]:
    segments: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for map_index, feature in enumerate(scenario.map):
        if feature.type != MapType.LANE:
            continue
        xy = np.asarray(feature.xy, dtype=np.float64)
        for segment_index in range(max(0, xy.shape[0] - 1)):
            start = xy[segment_index]
            end = xy[segment_index + 1]
            delta = end - start
            if (
                np.all(np.isfinite(start))
                and np.all(np.isfinite(end))
                and float(np.dot(delta, delta)) > 0.0
            ):
                segments.append((start, end, map_index, segment_index))
    return tuple(segments)


def _nearest_lane_segment(
    point: np.ndarray,
    segments: tuple[tuple[np.ndarray, np.ndarray, int, int], ...],
) -> tuple[float, np.ndarray, int, int]:
    best: tuple[float, np.ndarray, int, int] | None = None
    for start, end, map_index, segment_index in segments:
        delta = end - start
        parameter = float(np.dot(point - start, delta) / np.dot(delta, delta))
        parameter = min(1.0, max(0.0, parameter))
        projection = start + parameter * delta
        distance = float(np.linalg.norm(point - projection))
        candidate = (distance, delta, map_index, segment_index)
        if best is None or distance < best[0]:
            best = candidate
    if best is None:  # pragma: no cover - guarded by callers
        raise RuntimeError("no lane segment")
    return best


class LaneCenterDistanceMetric(_BaseMetric):
    spec = MetricSpec(
        name="lane_center_distance_m",
        version=M5_METRIC_VERSION,
        value_unit="m",
        direction="neutral",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="post-current future frames",
        eligibility="supported lane segment and eligible vehicle-frame",
        invalid_reason_codes=("no_supported_lane", "no_eligible_vehicle_frame"),
        required_fields=("map.lane.xy", "valid", "x", "y", "type"),
        known_failure_modes=("lane proximity is not an offroad metric",),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        vehicles = _vehicle_indices(scenario)
        total = len(vehicles) * max(0, scenario.num_steps - current - 1)
        segments = _lane_segments(scenario)
        if not segments:
            return _invalid_outcome(
                "no_supported_lane",
                total_components=total,
            )
        components: list[float] = []
        for frame in range(current + 1, scenario.num_steps):
            for index in vehicles:
                if scenario.agents[index].valid[frame]:
                    point = np.array(
                        [
                            rollout.agents[index].x[frame],
                            rollout.agents[index].y[frame],
                        ],
                        dtype=np.float64,
                    )
                    components.append(_nearest_lane_segment(point, segments)[0])
        if not components:
            return _invalid_outcome(
                "no_eligible_vehicle_frame",
                total_components=total,
            )
        return _valid_outcome(components, total_components=total)


class LaneHeadingDisagreementMetric(_BaseMetric):
    spec = MetricSpec(
        name="lane_heading_disagreement_rad",
        version=M5_METRIC_VERSION,
        value_unit="rad",
        direction="neutral",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="post-current future frames",
        eligibility="supported lane segment and eligible vehicle-frame",
        invalid_reason_codes=("no_supported_lane", "no_eligible_vehicle_frame"),
        required_fields=("map.lane.xy", "valid", "x", "y", "heading", "type"),
        known_failure_modes=("lane tangent disagreement is not wrong-way",),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        vehicles = _vehicle_indices(scenario)
        total = len(vehicles) * max(0, scenario.num_steps - current - 1)
        segments = _lane_segments(scenario)
        if not segments:
            return _invalid_outcome(
                "no_supported_lane",
                total_components=total,
            )
        components: list[float] = []
        for frame in range(current + 1, scenario.num_steps):
            for index in vehicles:
                if not scenario.agents[index].valid[frame]:
                    continue
                candidate = rollout.agents[index]
                point = np.array(
                    [candidate.x[frame], candidate.y[frame]],
                    dtype=np.float64,
                )
                _, tangent, _, _ = _nearest_lane_segment(point, segments)
                lane_heading = math.atan2(float(tangent[1]), float(tangent[0]))
                components.append(
                    abs(_wrap_angle(candidate.heading[frame] - lane_heading))
                )
        if not components:
            return _invalid_outcome(
                "no_eligible_vehicle_frame",
                total_components=total,
            )
        return _valid_outcome(components, total_components=total)


class KinematicContinuityResidualMetric(_BaseMetric):
    spec = MetricSpec(
        name="kinematic_continuity_residual_m",
        version=M5_METRIC_VERSION,
        value_unit="m",
        direction="lower",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="post-current future transitions",
        eligibility="at least one contiguous-valid non-ego future transition",
        invalid_reason_codes=("no_contiguous_valid_window",),
        required_fields=("valid", "x", "y", "vx", "vy", "timestamps"),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        targets = _non_ego_indices(scenario)
        total = len(targets) * max(0, scenario.num_steps - current - 1)
        components: list[float] = []
        for frame in range(current + 1, scenario.num_steps):
            dt = float(scenario.timestamps[frame] - scenario.timestamps[frame - 1])
            for index in targets:
                if not _contiguous_pair(scenario, index, frame):
                    continue
                candidate = rollout.agents[index]
                residual_x = (
                    candidate.x[frame]
                    - candidate.x[frame - 1]
                    - 0.5 * (candidate.vx[frame] + candidate.vx[frame - 1]) * dt
                )
                residual_y = (
                    candidate.y[frame]
                    - candidate.y[frame - 1]
                    - 0.5 * (candidate.vy[frame] + candidate.vy[frame - 1]) * dt
                )
                components.append(math.hypot(residual_x, residual_y))
        if not components:
            return _invalid_outcome(
                "no_contiguous_valid_window",
                total_components=total,
            )
        return _valid_outcome(components, total_components=total)


class LifecycleReentryPerAgentMetric(_BaseMetric):
    spec = MetricSpec(
        name="lifecycle_reentry_per_agent",
        version=M5_METRIC_VERSION,
        value_unit="events/agent",
        direction="neutral",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="current frame through future horizon",
        eligibility="at least one retained non-ego agent",
        invalid_reason_codes=("no_non_ego_agent",),
        required_fields=("valid",),
        known_failure_modes=(
            "policy differences indicate a lifecycle contract failure",
        ),
    )

    def _outcome(self, scenario: Scenario, rollout: Rollout) -> _Outcome:
        current = _validate_pair(scenario, rollout)
        targets = _non_ego_indices(scenario)
        total = len(targets)
        if not targets:
            return _invalid_outcome("no_non_ego_agent", total_components=0)
        counts: list[float] = []
        for index in targets:
            valid = scenario.agents[index].valid
            seen_valid = bool(valid[current])
            reentries = 0
            for frame in range(current + 1, scenario.num_steps):
                if valid[frame] and not valid[frame - 1]:
                    if seen_valid:
                        reentries += 1
                    seen_valid = True
                elif valid[frame]:
                    seen_valid = True
            counts.append(float(reentries))
        return _valid_outcome(counts, total_components=total)


M5_METRIC_TYPES = (
    PositionErrorMetric,
    SpeedErrorMetric,
    OrientedBoxOverlapRateMetric,
    WaymaxKinematicInfeasibilityRateMetric,
    AccelerationErrorMetric,
    JerkErrorMetric,
    YawRateErrorMetric,
    MinimumCenterDistanceMetric,
    ConstantVelocityTTCCapMetric,
    LaneCenterDistanceMetric,
    LaneHeadingDisagreementMetric,
    KinematicContinuityResidualMetric,
    LifecycleReentryPerAgentMetric,
)
M5_METRIC_SPECS = MappingProxyType(
    {
        metric_type.spec.name: metric_type.spec
        for metric_type in sorted(
            M5_METRIC_TYPES,
            key=lambda item: (item.spec.name, item.spec.version),
        )
    }
)


def m5_metrics() -> tuple[Metric, ...]:
    """Return all thirteen M5 metrics in deterministic registry order."""

    return tuple(
        metric_type()
        for metric_type in sorted(
            M5_METRIC_TYPES,
            key=lambda item: (item.spec.name, item.spec.version),
        )
    )
