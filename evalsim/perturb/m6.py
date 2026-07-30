"""Outcome-blind M6 ego interventions and source-only eligibility.

This module deliberately depends only on the EvalSim contracts and source-neutral
geometry.  It does not import WOMD or Waymax and it never executes a world policy.
The intervention formulas and rejection order are frozen by the accepted M6
pre-registration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np

from evalsim.contracts.counterfactual import (
    EgoInterventionSpec,
    EgoTrajectoryPlan,
    FeasibilityAudit,
    InterventionEligibility,
    M6_ANALYSIS_TRANSITIONS,
    M6_PLAN_FRAME_COUNT,
    M6_PRIMARY_ELIGIBILITY_REASONS,
)
from evalsim.contracts.scenario import Agent, Scenario
from evalsim.contracts.types import AgentType
from evalsim.rollout.dynamics import wrap_heading

M6_INTERVENTION_VERSION: Final = "v1"
M6_ACCESS_CLASS: Final = "logged_future_privileged"

IDENTITY_FAMILY: Final = "identity"
LONGITUDINAL_BRAKE_PULSE_FAMILY: Final = "longitudinal_brake_pulse"
PRIMARY_BRAKE_MAGNITUDE_MPS2: Final = 2.0
SECONDARY_BRAKE_MAGNITUDE_MPS2: Final = 4.0
BRAKE_PULSE_DURATION_S: Final = 1.0
REGISTERED_BRAKE_MAGNITUDES_MPS2: Final = (
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    SECONDARY_BRAKE_MAGNITUDE_MPS2,
)

PRIMARY_ELIGIBILITY_REASONS: Final = M6_PRIMARY_ELIGIBILITY_REASONS

_STATE_FIELDS: Final = ("x", "y", "heading", "vx", "vy")
_SOURCE_SEGMENT_EPSILON_M: Final = 1e-6
_VELOCITY_DIRECTION_EPSILON_MPS: Final = 1e-12
_MAX_SPEED_MPS: Final = 60.0
_MIN_ACCELERATION_MPS2: Final = -8.0
_MAX_ACCELERATION_MPS2: Final = 4.0
_MAX_ABS_YAW_RATE_RADPS: Final = 1.0
_DISPLACEMENT_ABSOLUTE_TOLERANCE_M: Final = 0.05
_DISPLACEMENT_RELATIVE_TOLERANCE: Final = 0.10
_HEADING_VELOCITY_SPEED_FLOOR_MPS: Final = 0.6
_MAX_HEADING_VELOCITY_DISAGREEMENT_RAD: Final = math.pi / 6.0
_FOLLOWER_LATERAL_TOLERANCE_M: Final = 2.75
_FOLLOWER_MAX_HEADING_DIFFERENCE_RAD: Final = math.pi / 6.0
_FOLLOWER_MIN_GAP_M: Final = 2.0
_FOLLOWER_MAX_GAP_M: Final = 40.0
_FROZEN_M5_OVERLAP_VERSION: Final = "1.0.0"


class InterventionCompilationError(ValueError):
    """A deterministic source-only plan compilation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        feasibility: FeasibilityAudit | None = None,
    ) -> None:
        self.code = code
        self.feasibility = feasibility
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class _SourceWindow:
    current_index: int
    stop_index: int
    timestamps: np.ndarray
    valid: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    vx: np.ndarray
    vy: np.ndarray


@dataclass(frozen=True, slots=True)
class _PlanArrays:
    timestamps: np.ndarray
    valid: np.ndarray
    applied: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    vx: np.ndarray
    vy: np.ndarray


def identity_spec() -> EgoInterventionSpec:
    """Return the sole registered M6 sham-control specification."""

    return EgoInterventionSpec(
        family=IDENTITY_FAMILY,
        version=M6_INTERVENTION_VERSION,
        dose=0.0,
        duration_s=0.0,
        parameters={"realization": "exact_logged_ego_copy"},
        access_class=M6_ACCESS_CLASS,
    )


def longitudinal_brake_pulse_spec(
    magnitude_mps2: float,
) -> EgoInterventionSpec:
    """Return a registered positive-magnitude M6 braking specification.

    M6 v1 registers only the primary 2 m/s² and nested secondary 4 m/s²
    treatments.  Zero magnitude is reserved for the reconstruction oracle and is
    not exposed as a treatment specification.
    """

    magnitude = _registered_brake_magnitude(magnitude_mps2)
    return EgoInterventionSpec(
        family=LONGITUDINAL_BRAKE_PULSE_FAMILY,
        version=M6_INTERVENTION_VERSION,
        dose=magnitude,
        duration_s=BRAKE_PULSE_DURATION_S,
        parameters={
            "deceleration_magnitude_mps2": magnitude,
            "deficit_after_pulse": "carry",
            "path_template": "source_ego_arc_length_piecewise_linear",
        },
        access_class=M6_ACCESS_CLASS,
    )


def compile_identity_plan(
    scenario: Scenario,
    *,
    stop_index: int | None = None,
) -> EgoTrajectoryPlan:
    """Compile the exact logged-ego sham plan over the frozen M6 horizon."""

    source = _source_window(scenario, stop_index=stop_index)
    _require_valid_ego_window(source)
    arrays = _identity_arrays(source)
    audit = _audit_arrays(source, arrays)
    if not audit.passed:
        raise InterventionCompilationError(
            "ego_plan_infeasible",
            "the identity plan failed the frozen M6 feasibility audit",
            feasibility=audit,
        )
    return _make_plan(identity_spec(), arrays, audit)


def compile_longitudinal_brake_pulse_plan(
    scenario: Scenario,
    magnitude_mps2: float,
    *,
    stop_index: int | None = None,
) -> EgoTrajectoryPlan:
    """Compile one registered source-templated M6 brake-pulse plan."""

    magnitude = _registered_brake_magnitude(magnitude_mps2)
    source = _source_window(scenario, stop_index=stop_index)
    _require_valid_ego_window(source)
    arc_knots = _source_arc_knots(source)
    if not _arc_is_strict(arc_knots):
        raise InterventionCompilationError(
            "source_ego_path_degenerate",
            "every source ego segment must be strictly longer than 1e-6 m",
        )
    if not _zero_dose_reconstruction_matches(source, arc_knots):
        raise InterventionCompilationError(
            "zero_dose_reconstruction_mismatch",
            "the general exact-knot interpolator did not reconstruct the source",
        )

    arrays = _brake_arrays(source, arc_knots, magnitude)
    audit = _audit_arrays(source, arrays)
    if not audit.passed:
        code = (
            "primary_ego_plan_infeasible"
            if magnitude == PRIMARY_BRAKE_MAGNITUDE_MPS2
            else "secondary_ego_plan_infeasible"
        )
        raise InterventionCompilationError(
            code,
            "the brake plan failed the frozen M6 feasibility audit",
            feasibility=audit,
        )
    return _make_plan(
        longitudinal_brake_pulse_spec(magnitude),
        arrays,
        audit,
    )


def zero_dose_reconstruction_matches(
    scenario: Scenario,
    *,
    stop_index: int | None = None,
) -> bool:
    """Run the required b=0 exact-knot source reconstruction oracle."""

    source = _source_window(scenario, stop_index=stop_index)
    arc_knots = _source_arc_knots(source)
    if not _arc_is_strict(arc_knots):
        raise InterventionCompilationError(
            "source_ego_path_degenerate",
            "every source ego segment must be strictly longer than 1e-6 m",
        )
    return _zero_dose_reconstruction_matches(source, arc_knots)


def validate_registered_ego_plan(
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
) -> None:
    """Recompile and exactly bind one executable M6 plan to its source.

    A caller-supplied feasibility object or canonical-looking intervention label is
    not evidence that the registered compiler produced the trajectory. Recompilation
    is the executable source-of-truth gate used immediately before engine execution.
    """

    if not isinstance(plan, EgoTrajectoryPlan):
        raise TypeError("plan must be an EgoTrajectoryPlan")
    plan.revalidate()
    if (
        plan.spec.family == IDENTITY_FAMILY
        and plan.spec.version == M6_INTERVENTION_VERSION
    ):
        expected = compile_identity_plan(scenario)
    elif (
        plan.spec.family == LONGITUDINAL_BRAKE_PULSE_FAMILY
        and plan.spec.version == M6_INTERVENTION_VERSION
    ):
        expected = compile_longitudinal_brake_pulse_plan(
            scenario,
            plan.spec.dose,
        )
    else:
        raise InterventionCompilationError(
            "ego_plan_family_unregistered",
            "M6 v1 executes only identity/v1 and longitudinal_brake_pulse/v1",
        )
    if plan.serialize() != expected.serialize():
        raise InterventionCompilationError(
            "ego_plan_source_binding_mismatch",
            "the supplied plan is not the exact registered source recompilation",
        )


def audit_ego_plan_feasibility(
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
    *,
    stop_index: int | None = None,
) -> FeasibilityAudit:
    """Independently repeat the frozen source-only feasibility audit."""

    if not isinstance(plan, EgoTrajectoryPlan):
        raise TypeError("plan must be an EgoTrajectoryPlan")
    plan.revalidate()
    source = _source_window(scenario, stop_index=stop_index)
    arrays = _PlanArrays(
        timestamps=np.asarray(plan.timestamps),
        valid=np.asarray(plan.valid),
        applied=np.asarray(plan.applied),
        x=np.asarray(plan.x),
        y=np.asarray(plan.y),
        heading=np.asarray(plan.heading),
        vx=np.asarray(plan.vx),
        vy=np.asarray(plan.vy),
    )
    return _audit_arrays(source, arrays)


def evaluate_primary_brake_eligibility(
    scenario: Scenario,
) -> InterventionEligibility:
    """Evaluate the frozen primary M6 source-only gate and target rule."""

    current = _current_index(scenario)
    stop = current + M6_ANALYSIS_TRANSITIONS
    analysis_window = (current, stop)
    if stop >= scenario.num_steps:
        return InterventionEligibility.rejected(
            "insufficient_future_horizon",
            analysis_window,
        )

    ego = scenario.ego
    window = slice(current, stop + 1)
    if not bool(np.all(np.asarray(ego.valid[window], dtype=bool))):
        return InterventionEligibility.rejected(
            "ego_invalid_in_window",
            analysis_window,
        )

    current_speed = float(np.hypot(ego.vx[current], ego.vy[current]))
    if not math.isfinite(current_speed) or current_speed < 5.0:
        return InterventionEligibility.rejected(
            "ego_speed_below_5_mps",
            analysis_window,
        )

    source = _source_window(scenario, stop_index=stop)
    arc_knots = _source_arc_knots(source)
    if not _arc_is_strict(arc_knots):
        return InterventionEligibility.rejected(
            "source_ego_path_degenerate",
            analysis_window,
        )
    if not _zero_dose_reconstruction_matches(source, arc_knots):
        return InterventionEligibility.rejected(
            "zero_dose_reconstruction_mismatch",
            analysis_window,
        )

    identity_arrays = _identity_arrays(source)
    if not _audit_arrays(source, identity_arrays).passed:
        return InterventionEligibility.rejected(
            "primary_ego_plan_infeasible",
            analysis_window,
        )
    primary_arrays = _brake_arrays(
        source,
        arc_knots,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    if not _audit_arrays(source, primary_arrays).passed:
        return InterventionEligibility.rejected(
            "primary_ego_plan_infeasible",
            analysis_window,
        )

    stage_a = _stage_a_followers(scenario, current=current, stop=stop)
    if not stage_a:
        return InterventionEligibility.rejected(
            "no_stable_aligned_follower",
            analysis_window,
        )
    if any(
        _source_boxes_overlap(
            scenario.ego,
            scenario.agents[candidate_index],
            frame=current,
        )
        for candidate_index, _ in stage_a
    ):
        return InterventionEligibility.rejected(
            "current_ego_follower_overlap",
            analysis_window,
        )

    stage_b = [
        (candidate_index, gap)
        for candidate_index, gap in stage_a
        if _FOLLOWER_MIN_GAP_M <= gap <= _FOLLOWER_MAX_GAP_M
        and _ego_is_nearest_forward_leader(
            scenario,
            follower_index=candidate_index,
            frame=current,
        )
    ]
    if not stage_b:
        return InterventionEligibility.rejected(
            "no_stable_aligned_follower",
            analysis_window,
        )

    target_index, _ = min(
        stage_b,
        key=lambda item: (
            item[1],
            int(scenario.agents[item[0]].id),
            item[0],
        ),
    )
    return InterventionEligibility.accepted(
        analysis_window,
        target_index,
    )


def _registered_brake_magnitude(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("brake magnitude must be a registered finite number")
    magnitude = float(value)
    if not math.isfinite(magnitude) or magnitude not in (
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
        SECONDARY_BRAKE_MAGNITUDE_MPS2,
    ):
        raise ValueError(
            "M6 v1 brake magnitude must be exactly 2.0 or 4.0 m/s^2"
        )
    return magnitude


def _current_index(scenario: Scenario) -> int:
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    raw = scenario.metadata.get("current_index", 0)
    if (
        isinstance(raw, (bool, np.bool_))
        or not isinstance(raw, (int, np.integer))
        or not 0 <= int(raw) < scenario.num_steps
    ):
        raise ValueError(
            "Scenario.metadata['current_index'] must index the scenario horizon"
        )
    return int(raw)


def _source_window(
    scenario: Scenario,
    *,
    stop_index: int | None,
) -> _SourceWindow:
    current = _current_index(scenario)
    expected_stop = current + M6_ANALYSIS_TRANSITIONS
    if stop_index is not None and (
        isinstance(stop_index, (bool, np.bool_))
        or not isinstance(stop_index, (int, np.integer))
        or int(stop_index) != expected_stop
    ):
        raise ValueError(
            "stop_index must equal current_index + 40 for the frozen M6 horizon"
        )
    stop = expected_stop
    if stop >= scenario.num_steps:
        raise InterventionCompilationError(
            "insufficient_future_horizon",
            "the scenario has fewer than 40 post-current transitions",
        )

    ego = scenario.ego
    window = slice(current, stop + 1)
    timestamps = np.array(scenario.timestamps[window], dtype=np.float64, copy=True)
    if (
        not np.all(np.isfinite(timestamps))
        or not np.all(np.diff(timestamps) > 0.0)
    ):
        raise InterventionCompilationError(
            "source_timestamps_invalid",
            "source timestamps must be finite and strictly increasing",
        )
    fields = {
        name: np.array(
            getattr(ego, name)[window],
            dtype=np.float64,
            copy=True,
        )
        for name in _STATE_FIELDS
    }
    return _SourceWindow(
        current_index=current,
        stop_index=stop,
        timestamps=timestamps,
        valid=np.array(ego.valid[window], dtype=bool, copy=True),
        **fields,
    )


def _identity_arrays(source: _SourceWindow) -> _PlanArrays:
    applied = np.ones(M6_PLAN_FRAME_COUNT, dtype=bool)
    applied[0] = False
    return _PlanArrays(
        timestamps=np.array(source.timestamps, copy=True),
        valid=np.array(source.valid, copy=True),
        applied=applied,
        x=np.array(source.x, copy=True),
        y=np.array(source.y, copy=True),
        heading=np.array(source.heading, copy=True),
        vx=np.array(source.vx, copy=True),
        vy=np.array(source.vy, copy=True),
    )


def _require_valid_ego_window(source: _SourceWindow) -> None:
    if not bool(np.all(source.valid)):
        raise InterventionCompilationError(
            "ego_invalid_in_window",
            "the ego must remain source-valid over all 41 M6 frames",
        )


def _source_arc_knots(source: _SourceWindow) -> np.ndarray:
    segment_lengths = np.hypot(
        np.diff(source.x),
        np.diff(source.y),
    )
    knots = np.empty(M6_PLAN_FRAME_COUNT, dtype=np.float64)
    knots[0] = 0.0
    knots[1:] = np.cumsum(segment_lengths, dtype=np.float64)
    return knots


def _arc_is_strict(arc_knots: np.ndarray) -> bool:
    segments = np.diff(arc_knots)
    return bool(
        np.all(np.isfinite(arc_knots))
        and np.all(segments > _SOURCE_SEGMENT_EPSILON_M)
    )


def _interpolate_source(
    source: _SourceWindow,
    arc_knots: np.ndarray,
    unwrapped_heading: np.ndarray,
    progress: float,
) -> tuple[float, float, float, float, float]:
    exact = np.flatnonzero(arc_knots == progress)
    if exact.size:
        index = int(exact[0])
        return (
            float(source.x[index]),
            float(source.y[index]),
            float(source.heading[index]),
            float(source.vx[index]),
            float(source.vy[index]),
        )

    if not 0.0 <= progress <= float(arc_knots[-1]):
        raise RuntimeError("treatment progress escaped the source arc")
    left = int(np.searchsorted(arc_knots, progress, side="right") - 1)
    right = left + 1
    denominator = float(arc_knots[right] - arc_knots[left])
    alpha = float((progress - arc_knots[left]) / denominator)
    x = float(source.x[left] + alpha * (source.x[right] - source.x[left]))
    y = float(source.y[left] + alpha * (source.y[right] - source.y[left]))
    vx = float(source.vx[left] + alpha * (source.vx[right] - source.vx[left]))
    vy = float(source.vy[left] + alpha * (source.vy[right] - source.vy[left]))
    heading_unwrapped = float(
        unwrapped_heading[left]
        + alpha * (unwrapped_heading[right] - unwrapped_heading[left])
    )
    heading = float(wrap_heading(heading_unwrapped))
    return x, y, heading, vx, vy


def _zero_dose_reconstruction_matches(
    source: _SourceWindow,
    arc_knots: np.ndarray,
) -> bool:
    zero = _brake_arrays(source, arc_knots, 0.0)
    if not (
        np.array_equal(zero.timestamps, source.timestamps)
        and np.array_equal(zero.valid, source.valid)
        and all(
            np.array_equal(
                np.asarray(getattr(zero, field_name)),
                np.asarray(getattr(source, field_name)),
            )
            for field_name in _STATE_FIELDS
        )
    ):
        return False

    unwrapped = np.unwrap(source.heading, discont=np.pi)
    reconstructed = np.empty(
        (len(_STATE_FIELDS), M6_PLAN_FRAME_COUNT),
        dtype=np.float64,
    )
    for frame, progress in enumerate(arc_knots):
        reconstructed[:, frame] = _interpolate_source(
            source,
            arc_knots,
            unwrapped,
            float(progress),
        )
    return all(
        np.array_equal(
            reconstructed[field_index],
            getattr(source, field_name),
        )
        for field_index, field_name in enumerate(_STATE_FIELDS)
    )


def _brake_arrays(
    source: _SourceWindow,
    arc_knots: np.ndarray,
    magnitude_mps2: float,
) -> _PlanArrays:
    """Compile b=0/2/4 arrays; b=0 is used only by internal source oracles."""

    magnitude = float(magnitude_mps2)
    if magnitude not in (0.0, *REGISTERED_BRAKE_MAGNITUDES_MPS2):
        raise ValueError("internal brake magnitude must be 0.0, 2.0, or 4.0")
    if magnitude == 0.0:
        return _identity_arrays(source)

    elapsed = source.timestamps - source.timestamps[0]
    dt = np.diff(elapsed)
    deficit = magnitude * np.minimum(
        np.maximum(elapsed, 0.0),
        BRAKE_PULSE_DURATION_S,
    )
    source_segments = np.diff(arc_knots)
    lost_distance = np.minimum(
        source_segments,
        0.5 * (deficit[:-1] + deficit[1:]) * dt,
    )
    progress = np.empty(M6_PLAN_FRAME_COUNT, dtype=np.float64)
    progress[0] = 0.0
    progress[1:] = np.cumsum(
        source_segments - lost_distance,
        dtype=np.float64,
    )

    unwrapped = np.unwrap(source.heading, discont=np.pi)
    x = np.empty(M6_PLAN_FRAME_COUNT, dtype=np.float64)
    y = np.empty(M6_PLAN_FRAME_COUNT, dtype=np.float64)
    heading = np.empty(M6_PLAN_FRAME_COUNT, dtype=np.float64)
    vx = np.empty(M6_PLAN_FRAME_COUNT, dtype=np.float64)
    vy = np.empty(M6_PLAN_FRAME_COUNT, dtype=np.float64)
    source_speed = np.hypot(source.vx, source.vy)
    treatment_speed = np.maximum(0.0, source_speed - deficit)

    for frame, frame_progress in enumerate(progress):
        (
            x[frame],
            y[frame],
            heading[frame],
            direction_vx,
            direction_vy,
        ) = _interpolate_source(
            source,
            arc_knots,
            unwrapped,
            float(frame_progress),
        )
        direction_norm = float(np.hypot(direction_vx, direction_vy))
        if direction_norm > _VELOCITY_DIRECTION_EPSILON_MPS:
            unit_x = direction_vx / direction_norm
            unit_y = direction_vy / direction_norm
        else:
            unit_x = math.cos(float(heading[frame]))
            unit_y = math.sin(float(heading[frame]))
        vx[frame] = treatment_speed[frame] * unit_x
        vy[frame] = treatment_speed[frame] * unit_y

    # The current state is a contractual bit-for-bit source copy even though the
    # rescaling formula is numerically equivalent at zero elapsed time.
    for field_name, destination in (
        ("x", x),
        ("y", y),
        ("heading", heading),
        ("vx", vx),
        ("vy", vy),
    ):
        destination[0] = getattr(source, field_name)[0]

    applied = np.ones(M6_PLAN_FRAME_COUNT, dtype=bool)
    applied[0] = False
    return _PlanArrays(
        timestamps=np.array(source.timestamps, copy=True),
        valid=np.array(source.valid, copy=True),
        applied=applied,
        x=x,
        y=y,
        heading=heading,
        vx=vx,
        vy=vy,
    )


def _audit_arrays(
    source: _SourceWindow,
    arrays: _PlanArrays,
) -> FeasibilityAudit:
    state_arrays = [
        np.asarray(getattr(arrays, name), dtype=np.float64)
        for name in _STATE_FIELDS
    ]
    timestamps = np.asarray(arrays.timestamps, dtype=np.float64)
    finite = bool(
        np.all(np.isfinite(timestamps))
        and all(np.all(np.isfinite(values)) for values in state_arrays)
    )
    source_identity = bool(
        np.array_equal(timestamps, source.timestamps)
        and np.array_equal(np.asarray(arrays.valid), source.valid)
        and all(
            np.array_equal(
                np.asarray(getattr(arrays, name))[0:1],
                np.asarray(getattr(source, name))[0:1],
            )
            for name in _STATE_FIELDS
        )
    )

    if finite:
        dt = np.diff(timestamps)
        speed = np.hypot(arrays.vx, arrays.vy)
        acceleration = np.diff(speed) / dt
        yaw_rate = np.asarray(
            wrap_heading(np.diff(arrays.heading)),
            dtype=np.float64,
        ) / dt
        displacement = np.hypot(np.diff(arrays.x), np.diff(arrays.y))
        trapezoidal_distance = 0.5 * (speed[:-1] + speed[1:]) * dt
        residual = np.abs(displacement - trapezoidal_distance)
        disagreement = np.abs(
            np.asarray(
                wrap_heading(
                    arrays.heading - np.arctan2(arrays.vy, arrays.vx)
                ),
                dtype=np.float64,
            )
        )
        moving = speed > _HEADING_VELOCITY_SPEED_FLOOR_MPS
        speed_bounds = bool(
            np.all(speed >= 0.0) and np.all(speed <= _MAX_SPEED_MPS)
        )
        acceleration_bounds = bool(
            np.all(acceleration >= _MIN_ACCELERATION_MPS2)
            and np.all(acceleration <= _MAX_ACCELERATION_MPS2)
        )
        yaw_rate_bounds = bool(
            np.all(np.abs(yaw_rate) <= _MAX_ABS_YAW_RATE_RADPS)
        )
        displacement_upper_bound = bool(
            np.all(
                displacement
                <= trapezoidal_distance
                + _DISPLACEMENT_ABSOLUTE_TOLERANCE_M
            )
        )
        distance_residual = bool(
            np.all(
                residual
                <= np.maximum(
                    _DISPLACEMENT_ABSOLUTE_TOLERANCE_M,
                    _DISPLACEMENT_RELATIVE_TOLERANCE
                    * trapezoidal_distance,
                )
            )
        )
        heading_velocity_alignment = bool(
            np.all(
                disagreement[moving]
                <= _MAX_HEADING_VELOCITY_DISAGREEMENT_RAD
            )
        )
    else:
        speed_bounds = False
        acceleration_bounds = False
        yaw_rate_bounds = False
        displacement_upper_bound = False
        distance_residual = False
        heading_velocity_alignment = False

    checks = {
        "finite": finite,
        "source_identity": source_identity,
        "speed_bounds": speed_bounds,
        "acceleration_bounds": acceleration_bounds,
        "yaw_rate_bounds": yaw_rate_bounds,
        "displacement_upper_bound": displacement_upper_bound,
        "distance_residual": distance_residual,
        "heading_velocity_alignment": heading_velocity_alignment,
    }
    failed = next((name for name, passed in checks.items() if not passed), None)
    details = {"frame_count": M6_PLAN_FRAME_COUNT}
    if failed is None:
        return FeasibilityAudit.accepted(checks, details=details)
    return FeasibilityAudit.rejected(
        f"{failed}_failed",
        checks,
        details=details,
    )


def _make_plan(
    spec: EgoInterventionSpec,
    arrays: _PlanArrays,
    feasibility: FeasibilityAudit,
) -> EgoTrajectoryPlan:
    return EgoTrajectoryPlan(
        spec=spec,
        timestamps=arrays.timestamps,
        valid=arrays.valid,
        applied=arrays.applied,
        x=arrays.x,
        y=arrays.y,
        heading=arrays.heading,
        vx=arrays.vx,
        vy=arrays.vy,
        realization_type=M6_ACCESS_CLASS,
        feasibility=feasibility,
    )


def _motion_direction(
    agent: Agent,
    frame: int,
) -> np.ndarray:
    vx = float(agent.vx[frame])
    vy = float(agent.vy[frame])
    speed = float(np.hypot(vx, vy))
    if speed > _VELOCITY_DIRECTION_EPSILON_MPS:
        return np.array([vx / speed, vy / speed], dtype=np.float64)
    heading = float(agent.heading[frame])
    return np.array(
        [math.cos(heading), math.sin(heading)],
        dtype=np.float64,
    )


def _stage_a_followers(
    scenario: Scenario,
    *,
    current: int,
    stop: int,
) -> list[tuple[int, float]]:
    ego = scenario.ego
    ego_position = np.array(
        [ego.x[current], ego.y[current]],
        dtype=np.float64,
    )
    candidates: list[tuple[int, float]] = []
    for index, agent in enumerate(scenario.agents):
        if (
            index == scenario.ego_index
            or agent.type != AgentType.VEHICLE
            or not bool(agent.valid[current])
            or not bool(np.all(agent.valid[current : stop + 1]))
        ):
            continue
        direction = _motion_direction(agent, current)
        relative = ego_position - np.array(
            [agent.x[current], agent.y[current]],
            dtype=np.float64,
        )
        center = float(np.dot(relative, direction))
        lateral = abs(
            float(
                direction[0] * relative[1]
                - direction[1] * relative[0]
            )
        )
        heading_disagreement = abs(
            float(
                wrap_heading(
                    float(ego.heading[current] - agent.heading[current])
                )
            )
        )
        if (
            center > 0.0
            and lateral <= _FOLLOWER_LATERAL_TOLERANCE_M
            and heading_disagreement
            <= _FOLLOWER_MAX_HEADING_DIFFERENCE_RAD
        ):
            gap = center - 0.5 * (float(ego.length) + float(agent.length))
            candidates.append((index, float(gap)))
    return candidates


def _source_boxes_overlap(
    first: Agent,
    second: Agent,
    *,
    frame: int,
) -> bool:
    # Deliberately reuse the accepted M5 1.0.0 float32 SAT primitive so the
    # eligibility boundary cannot drift from the registered overlap definition.
    from evalsim.metrics.m5 import M5_METRIC_VERSION, _boxes_overlap_strict

    if M5_METRIC_VERSION != _FROZEN_M5_OVERLAP_VERSION:
        raise RuntimeError(
            "M6 eligibility requires the frozen M5 overlap definition 1.0.0"
        )

    return _boxes_overlap_strict(
        np.float32(first.x[frame]),
        np.float32(first.y[frame]),
        np.float32(first.heading[frame]),
        np.float32(first.length),
        np.float32(first.width),
        np.float32(second.x[frame]),
        np.float32(second.y[frame]),
        np.float32(second.heading[frame]),
        np.float32(second.length),
        np.float32(second.width),
    )


def _ego_is_nearest_forward_leader(
    scenario: Scenario,
    *,
    follower_index: int,
    frame: int,
) -> bool:
    follower = scenario.agents[follower_index]
    follower_direction = _motion_direction(follower, frame)
    follower_position = np.array(
        [follower.x[frame], follower.y[frame]],
        dtype=np.float64,
    )
    alignment_threshold = math.cos(_FOLLOWER_MAX_HEADING_DIFFERENCE_RAD)
    leaders: list[tuple[float, int, int]] = []

    for leader_index, leader in enumerate(scenario.agents):
        if leader_index == follower_index or not bool(leader.valid[frame]):
            continue
        leader_direction = _motion_direction(leader, frame)
        if (
            float(np.dot(follower_direction, leader_direction))
            < alignment_threshold
        ):
            continue
        relative = np.array(
            [
                float(leader.x[frame] - follower_position[0]),
                float(leader.y[frame] - follower_position[1]),
            ],
            dtype=np.float64,
        )
        center = float(np.dot(relative, follower_direction))
        if center <= 0.0:
            continue
        lateral = abs(
            float(
                follower_direction[0] * relative[1]
                - follower_direction[1] * relative[0]
            )
        )
        if lateral > _FOLLOWER_LATERAL_TOLERANCE_M:
            continue
        bumper_gap = center - 0.5 * (
            float(follower.length) + float(leader.length)
        )
        leaders.append((float(bumper_gap), int(leader.id), leader_index))

    if not leaders:
        return False
    _, _, nearest_index = min(leaders, key=lambda item: (item[0], item[1]))
    return nearest_index == scenario.ego_index


__all__ = [
    "BRAKE_PULSE_DURATION_S",
    "IDENTITY_FAMILY",
    "LONGITUDINAL_BRAKE_PULSE_FAMILY",
    "M6_ACCESS_CLASS",
    "M6_ANALYSIS_TRANSITIONS",
    "M6_INTERVENTION_VERSION",
    "M6_PLAN_FRAME_COUNT",
    "PRIMARY_BRAKE_MAGNITUDE_MPS2",
    "PRIMARY_ELIGIBILITY_REASONS",
    "REGISTERED_BRAKE_MAGNITUDES_MPS2",
    "SECONDARY_BRAKE_MAGNITUDE_MPS2",
    "InterventionCompilationError",
    "audit_ego_plan_feasibility",
    "compile_identity_plan",
    "compile_longitudinal_brake_pulse_plan",
    "evaluate_primary_brake_eligibility",
    "identity_spec",
    "longitudinal_brake_pulse_spec",
    "validate_registered_ego_plan",
    "zero_dose_reconstruction_matches",
]
