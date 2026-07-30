"""Source-neutral paired counterfactual measures pre-registered for EvalSim M6.

The measures in this module consume only the immutable
:class:`~evalsim.contracts.counterfactual.CounterfactualPair` contract.  They do not
load source data, compile interventions, select targets, or aggregate across scenes.

Acceleration samples belong to their source timestamp intervals.  For the secondary
jerk diagnostic, adjacent acceleration samples are differentiated across the distance
between their interval midpoints,
``(dt_previous + dt_current) / 2``.  The same duration is the integration weight.  This
is the physical-time analogue of the existing M5 jerk convention and never bridges an
invalid frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from evalsim.contracts.counterfactual import (
    CounterfactualPair,
    PairedMetricResult,
)
from evalsim.contracts.metric import MetricSpec
from evalsim.contracts.types import AgentType

M6_PAIRED_METRIC_VERSION = "1.0.0"
M6_RESPONSE_ACCELERATION_THRESHOLD_MPS2 = -0.5
M6_RESPONSE_PERSISTENCE_S = 0.2
M6_HARD_BRAKING_THRESHOLD_MPS2 = -4.0
_WORLD_STATE_FIELDS = ("valid", "x", "y", "heading", "vx", "vy")


@dataclass(frozen=True, slots=True)
class _PairView:
    pair: CounterfactualPair
    current: int
    stop: int
    target_index: int

    @property
    def future(self) -> slice:
        return slice(self.current + 1, self.stop + 1)

    @property
    def window(self) -> slice:
        return slice(self.current, self.stop + 1)

    @property
    def timestamps(self) -> np.ndarray:
        return self.pair.baseline.timestamps[self.window]

    @property
    def dt(self) -> np.ndarray:
        return np.diff(self.timestamps)

    @property
    def source_target(self) -> Any:
        return self.pair.scenario.agents[self.target_index]

    @property
    def baseline_target(self) -> Any:
        return self.pair.baseline.agents[self.target_index]

    @property
    def intervention_target(self) -> Any:
        return self.pair.intervention.agents[self.target_index]


def _validated_pair_view(pair: CounterfactualPair) -> _PairView:
    """Return the complete frozen-target view or fail instead of imputing."""

    if not isinstance(pair, CounterfactualPair):
        raise TypeError("pair must be a CounterfactualPair")
    pair.revalidate()
    eligibility = pair.eligibility
    target_index = eligibility.target_index
    if target_index is None:  # Defensive: accepted eligibility already forbids this.
        raise ValueError("paired metrics require a frozen target_index")
    current = eligibility.current_index
    stop = eligibility.stop_index
    target = pair.scenario.agents[target_index]
    if target_index == pair.scenario.ego_index or target.type != AgentType.VEHICLE:
        raise ValueError("frozen target must be a non-ego vehicle")

    required = slice(current, stop + 1)
    if not bool(np.all(target.valid[required])):
        raise ValueError(
            "frozen target must be source-valid throughout the analysis window"
        )
    if not bool(np.all(pair.scenario.ego.valid[required])):
        raise ValueError(
            "ego must be source-valid throughout the analysis window"
        )
    return _PairView(
        pair=pair,
        current=current,
        stop=stop,
        target_index=target_index,
    )


def _target_accelerations(view: _PairView) -> tuple[np.ndarray, np.ndarray]:
    """Return baseline and treatment scalar-speed accelerations."""

    baseline_speed = view.baseline_target.speed()[view.window]
    intervention_speed = view.intervention_target.speed()[view.window]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        baseline = np.diff(baseline_speed) / view.dt
        intervention = np.diff(intervention_speed) / view.dt
    if not np.all(np.isfinite(baseline)) or not np.all(
        np.isfinite(intervention)
    ):
        raise ValueError(
            "target scalar-speed accelerations must be finite on every transition"
        )
    return baseline, intervention


def _target_current_direction(view: _PairView) -> tuple[float, float]:
    """Freeze the §5 velocity direction with heading fallback."""

    target = view.source_target
    vx = float(target.vx[view.current])
    vy = float(target.vy[view.current])
    speed = math.hypot(vx, vy)
    if speed > 1e-12:
        return vx / speed, vy / speed
    heading = float(target.heading[view.current])
    return math.cos(heading), math.sin(heading)


def _target_current_heading_direction(view: _PairView) -> tuple[float, float]:
    """Return the heading direction used by target-progress loss."""

    heading = float(view.source_target.heading[view.current])
    return math.cos(heading), math.sin(heading)


def _result(
    spec: MetricSpec,
    view: _PairView,
    value: float,
    details: dict[str, Any],
) -> PairedMetricResult:
    return PairedMetricResult(
        metric_name=spec.name,
        metric_version=spec.version,
        scenario_id=view.pair.scenario.scenario_id,
        intervention_identity=view.pair.intervention_identity,
        value=float(value),
        details={
            "target_index": view.target_index,
            "transition_count": view.stop - view.current,
            **details,
        },
    )


class _PairedMetricBase:
    """Common complete-pair validation for the pure one-scene measures."""

    spec: ClassVar[MetricSpec]

    def _compute(self, view: _PairView) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError

    def compute(self, pair: CounterfactualPair) -> PairedMetricResult:
        view = _validated_pair_view(pair)
        value, details = self._compute(view)
        if not math.isfinite(float(value)):
            raise ValueError("paired metric produced a non-finite scalar")
        return _result(self.spec, view, value, details)


class AdditionalTargetBrakingImpulseMetric(_PairedMetricBase):
    """Additional target braking integrated over the 40 transitions."""

    spec = MetricSpec(
        name="additional_target_braking_impulse_mps",
        version=M6_PAIRED_METRIC_VERSION,
        value_unit="m/s",
        unit_of_analysis="scenario",
        direction="neutral",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="current through 40 post-current transitions",
        eligibility="complete immutable pair with a source-valid frozen vehicle target",
        required_fields=("valid", "vx", "vy", "timestamps"),
        known_failure_modes=(
            "more braking is reactivity, not evidence of better driving quality",
        ),
    )

    def _compute(self, view: _PairView) -> tuple[float, dict[str, Any]]:
        baseline, intervention = _target_accelerations(view)
        components = np.maximum(0.0, baseline - intervention) * view.dt
        value = math.fsum(float(component) for component in components)
        return value, {}


class ResponseTimelinessMetric(_PairedMetricBase):
    """Persistent target-braking response with explicit right censoring."""

    spec = MetricSpec(
        name="response_timeliness_s",
        version=M6_PAIRED_METRIC_VERSION,
        value_unit="s",
        unit_of_analysis="scenario",
        direction="neutral",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="current through 40 post-current transitions",
        eligibility="complete immutable pair with a source-valid frozen vehicle target",
        required_fields=("valid", "vx", "vy", "timestamps"),
        known_failure_modes=(
            "restricted timeliness is zero both for censoring and an event at W",
            "faster response is not evidence of better driving quality",
        ),
    )

    def _compute(self, view: _PairView) -> tuple[float, dict[str, Any]]:
        baseline, intervention = _target_accelerations(view)
        delta = intervention - baseline
        response_start: int | None = None
        response_end: int | None = None
        run_start: int | None = None

        # Transition 0 acted on the unchanged current ego and is excluded.  A run is
        # reset on any threshold violation; math.fsum makes duration boundaries stable
        # for irregular timestamp intervals.
        for transition in range(1, len(delta)):
            if float(delta[transition]) <= M6_RESPONSE_ACCELERATION_THRESHOLD_MPS2:
                if run_start is None:
                    run_start = transition
                duration = math.fsum(
                    float(value) for value in view.dt[run_start : transition + 1]
                )
                if duration >= M6_RESPONSE_PERSISTENCE_S:
                    response_start = run_start
                    response_end = transition
                    break
            else:
                run_start = None

        window_s = float(view.timestamps[-1] - view.timestamps[0])
        responded = response_end is not None
        event_time_s: float | None
        if responded:
            assert response_start is not None
            event_time_s = float(
                view.timestamps[response_end + 1] - view.timestamps[0]
            )
            restricted_latency_s = min(event_time_s, window_s)
        else:
            event_time_s = None
            restricted_latency_s = window_s
        value = window_s - restricted_latency_s
        return value, {
            "responded": responded,
            "censored": not responded,
            "event_time_s": event_time_s,
            "restricted_latency_s": restricted_latency_s,
            "window_s": window_s,
            "response_start_transition": response_start,
            "response_end_transition": response_end,
            "search_start_transition": 1,
            "acceleration_threshold_mps2": (
                M6_RESPONSE_ACCELERATION_THRESHOLD_MPS2
            ),
            "persistence_s": M6_RESPONSE_PERSISTENCE_S,
        }


class MinimumLongitudinalBumperGapChangeMetric(_PairedMetricBase):
    """Treatment-minus-baseline minimum frozen-heading bumper-gap proxy."""

    spec = MetricSpec(
        name="minimum_longitudinal_bumper_gap_change_m",
        version=M6_PAIRED_METRIC_VERSION,
        value_unit="m",
        unit_of_analysis="scenario",
        direction="neutral",
        aggregation="mean",
        agent_scope="all",
        evaluation_window="40 post-current scored frames",
        eligibility="complete immutable pair with a source-valid frozen vehicle target",
        required_fields=(
            "valid",
            "x",
            "y",
            "heading",
            "vx",
            "vy",
            "length",
        ),
        known_failure_modes=(
            "relational proxy changes when ego moves even if the world is nonreactive",
            "not lane headway or safety ground truth",
        ),
    )

    @staticmethod
    def _gaps(
        ego: Any,
        target: Any,
        frames: range,
        hx: float,
        hy: float,
    ) -> tuple[float, ...]:
        half_length = 0.5 * (float(ego.length) + float(target.length))
        return tuple(
            (
                (float(ego.x[frame]) - float(target.x[frame])) * hx
                + (float(ego.y[frame]) - float(target.y[frame])) * hy
                - half_length
            )
            for frame in frames
        )

    def _compute(self, view: _PairView) -> tuple[float, dict[str, Any]]:
        hx, hy = _target_current_direction(view)
        frames = range(view.current + 1, view.stop + 1)
        baseline_gaps = self._gaps(
            view.pair.baseline.agents[view.pair.scenario.ego_index],
            view.baseline_target,
            frames,
            hx,
            hy,
        )
        intervention_gaps = self._gaps(
            view.pair.intervention.agents[view.pair.scenario.ego_index],
            view.intervention_target,
            frames,
            hx,
            hy,
        )
        baseline_min = min(baseline_gaps)
        intervention_min = min(intervention_gaps)
        return intervention_min - baseline_min, {
            "baseline_minimum_m": baseline_min,
            "intervention_minimum_m": intervention_min,
        }


class TargetProgressLossMetric(_PairedMetricBase):
    """Baseline-minus-treatment target progress along current target heading."""

    spec = MetricSpec(
        name="target_progress_loss_m",
        version=M6_PAIRED_METRIC_VERSION,
        value_unit="m",
        unit_of_analysis="scenario",
        direction="neutral",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="current to final scored frame",
        eligibility="complete immutable pair with a source-valid frozen vehicle target",
        required_fields=("valid", "x", "y", "heading"),
        known_failure_modes=(
            "progress loss is response cost, not proof of overreaction",
        ),
    )

    def _compute(self, view: _PairView) -> tuple[float, dict[str, Any]]:
        hx, hy = _target_current_heading_direction(view)
        source = view.source_target
        origin_x = float(source.x[view.current])
        origin_y = float(source.y[view.current])

        def progress(target: Any) -> float:
            return (
                (float(target.x[view.stop]) - origin_x) * hx
                + (float(target.y[view.stop]) - origin_y) * hy
            )

        baseline_progress = progress(view.baseline_target)
        intervention_progress = progress(view.intervention_target)
        return baseline_progress - intervention_progress, {
            "baseline_progress_m": baseline_progress,
            "intervention_progress_m": intervention_progress,
        }


class TargetWorldDisplacementMeanMetric(_PairedMetricBase):
    """Mean future target position displacement between paired conditions."""

    spec = MetricSpec(
        name="target_world_displacement_mean_m",
        version=M6_PAIRED_METRIC_VERSION,
        value_unit="m",
        unit_of_analysis="scenario",
        direction="neutral",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="40 post-current scored frames",
        eligibility="complete immutable pair with a source-valid frozen vehicle target",
        required_fields=("valid", "x", "y"),
    )

    def _compute(self, view: _PairView) -> tuple[float, dict[str, Any]]:
        baseline = view.baseline_target
        intervention = view.intervention_target
        distances = np.hypot(
            intervention.x[view.future] - baseline.x[view.future],
            intervention.y[view.future] - baseline.y[view.future],
        )
        value = math.fsum(float(distance) for distance in distances) / len(
            distances
        )
        return value, {}


class TargetSpeedReductionMaxMetric(_PairedMetricBase):
    """Maximum future baseline-minus-treatment target scalar speed."""

    spec = MetricSpec(
        name="target_speed_reduction_max_mps",
        version=M6_PAIRED_METRIC_VERSION,
        value_unit="m/s",
        unit_of_analysis="scenario",
        direction="neutral",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="40 post-current scored frames",
        eligibility="complete immutable pair with a source-valid frozen vehicle target",
        required_fields=("valid", "vx", "vy"),
    )

    def _compute(self, view: _PairView) -> tuple[float, dict[str, Any]]:
        reduction = (
            view.baseline_target.speed()[view.future]
            - view.intervention_target.speed()[view.future]
        )
        return float(np.max(reduction)), {}


class AdditionalAbsoluteJerkIntegralMetric(_PairedMetricBase):
    """Additional absolute scalar-speed jerk integrated over adjacent intervals."""

    spec = MetricSpec(
        name="additional_absolute_jerk_integral_mps2",
        version=M6_PAIRED_METRIC_VERSION,
        value_unit="m/s^2",
        unit_of_analysis="scenario",
        direction="neutral",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="contiguous derivatives within the 40 transitions",
        eligibility="complete immutable pair with a source-valid frozen vehicle target",
        required_fields=("valid", "vx", "vy", "timestamps"),
    )

    def _compute(self, view: _PairView) -> tuple[float, dict[str, Any]]:
        baseline, intervention = _target_accelerations(view)
        midpoint_dt = (view.dt[:-1] + view.dt[1:]) / 2.0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            baseline_jerk = np.diff(baseline) / midpoint_dt
            intervention_jerk = np.diff(intervention) / midpoint_dt
        if not np.all(np.isfinite(baseline_jerk)) or not np.all(
            np.isfinite(intervention_jerk)
        ):
            raise ValueError(
                "target scalar-speed jerk must be finite on every derivative interval"
            )
        components = (
            np.maximum(
                0.0,
                np.abs(intervention_jerk) - np.abs(baseline_jerk),
            )
            * midpoint_dt
        )
        value = math.fsum(float(component) for component in components)
        return value, {"derivative_interval_count": len(components)}


class AdditionalHardBrakingExposureMetric(_PairedMetricBase):
    """Treatment-minus-baseline time at or below -4 m/s²."""

    spec = MetricSpec(
        name="additional_hard_braking_exposure_s",
        version=M6_PAIRED_METRIC_VERSION,
        value_unit="s",
        unit_of_analysis="scenario",
        direction="neutral",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="current through 40 post-current transitions",
        eligibility="complete immutable pair with a source-valid frozen vehicle target",
        required_fields=("valid", "vx", "vy", "timestamps"),
    )

    def _compute(self, view: _PairView) -> tuple[float, dict[str, Any]]:
        baseline, intervention = _target_accelerations(view)
        baseline_exposure = math.fsum(
            float(dt)
            for acceleration, dt in zip(baseline, view.dt, strict=True)
            if float(acceleration) <= M6_HARD_BRAKING_THRESHOLD_MPS2
        )
        intervention_exposure = math.fsum(
            float(dt)
            for acceleration, dt in zip(intervention, view.dt, strict=True)
            if float(acceleration) <= M6_HARD_BRAKING_THRESHOLD_MPS2
        )
        return intervention_exposure - baseline_exposure, {
            "baseline_exposure_s": baseline_exposure,
            "intervention_exposure_s": intervention_exposure,
            "inclusive_acceleration_threshold_mps2": (
                M6_HARD_BRAKING_THRESHOLD_MPS2
            ),
        }


M6_PRIMARY_PAIRED_METRIC_TYPES = (
    AdditionalTargetBrakingImpulseMetric,
    ResponseTimelinessMetric,
    MinimumLongitudinalBumperGapChangeMetric,
    TargetProgressLossMetric,
)
M6_SECONDARY_PAIRED_METRIC_TYPES = (
    TargetWorldDisplacementMeanMetric,
    TargetSpeedReductionMaxMetric,
    AdditionalAbsoluteJerkIntegralMetric,
    AdditionalHardBrakingExposureMetric,
)
M6_PRIMARY_PAIRED_METRIC_SPECS = tuple(
    metric_type.spec for metric_type in M6_PRIMARY_PAIRED_METRIC_TYPES
)
M6_SECONDARY_PAIRED_METRIC_SPECS = tuple(
    metric_type.spec for metric_type in M6_SECONDARY_PAIRED_METRIC_TYPES
)


def m6_primary_paired_metrics() -> tuple[_PairedMetricBase, ...]:
    """Return the four primary metrics in pre-registered order."""

    return tuple(metric_type() for metric_type in M6_PRIMARY_PAIRED_METRIC_TYPES)


def m6_secondary_paired_metrics() -> tuple[_PairedMetricBase, ...]:
    """Return the four registered analytic secondary metrics in plan order."""

    return tuple(metric_type() for metric_type in M6_SECONDARY_PAIRED_METRIC_TYPES)


def m6_paired_metrics() -> tuple[_PairedMetricBase, ...]:
    """Return primary followed by registered analytic secondary metrics."""

    return m6_primary_paired_metrics() + m6_secondary_paired_metrics()


def world_trajectory_tensor_equal(pair: CounterfactualPair) -> bool:
    """Compare the exact pre-registered non-ego world tensor field by field.

    The comparison contains no numeric tolerance.  Ordered identity, type,
    dimensions, validity, every state field, and timestamps must all be identical.
    """

    view = _validated_pair_view(pair)
    baseline = pair.baseline
    intervention = pair.intervention
    if not np.array_equal(baseline.timestamps, intervention.timestamps):
        return False
    for index, (left, right) in enumerate(
        zip(baseline.agents, intervention.agents, strict=True)
    ):
        if index == pair.scenario.ego_index:
            continue
        if (
            left.id != right.id
            or left.type != right.type
            or left.length != right.length
            or left.width != right.width
        ):
            return False
        for field in _WORLD_STATE_FIELDS:
            if not np.array_equal(
                getattr(left, field)[view.future],
                getattr(right, field)[view.future],
            ):
                return False
    return True


def is_exactly_nonreactive(pair: CounterfactualPair) -> bool:
    """Apply the M6 structural nonreactivity definition.

    Minimum gap is deliberately absent: it is a relational ego-plus-world effect and
    cannot classify world response.
    """

    if not world_trajectory_tensor_equal(pair):
        return False
    impulse = AdditionalTargetBrakingImpulseMetric().compute(pair).value
    timeliness = ResponseTimelinessMetric().compute(pair).value
    progress = TargetProgressLossMetric().compute(pair).value
    return impulse == 0.0 and timeliness == 0.0 and progress == 0.0


__all__ = [
    "AdditionalAbsoluteJerkIntegralMetric",
    "AdditionalHardBrakingExposureMetric",
    "AdditionalTargetBrakingImpulseMetric",
    "M6_HARD_BRAKING_THRESHOLD_MPS2",
    "M6_PAIRED_METRIC_VERSION",
    "M6_PRIMARY_PAIRED_METRIC_SPECS",
    "M6_PRIMARY_PAIRED_METRIC_TYPES",
    "M6_RESPONSE_ACCELERATION_THRESHOLD_MPS2",
    "M6_RESPONSE_PERSISTENCE_S",
    "M6_SECONDARY_PAIRED_METRIC_SPECS",
    "M6_SECONDARY_PAIRED_METRIC_TYPES",
    "MinimumLongitudinalBumperGapChangeMetric",
    "ResponseTimelinessMetric",
    "TargetProgressLossMetric",
    "TargetSpeedReductionMaxMetric",
    "TargetWorldDisplacementMeanMetric",
    "is_exactly_nonreactive",
    "m6_paired_metrics",
    "m6_primary_paired_metrics",
    "m6_secondary_paired_metrics",
    "world_trajectory_tensor_equal",
]
