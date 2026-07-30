"""M6 bounded-Waymax paired measures; every fixture is invented in memory."""
from __future__ import annotations

import dataclasses
import copy
import hashlib
import json
import math
import struct

import numpy as np
import pytest

import evalsim.evaluation.m6_waymax_metrics as _waymax_metrics
from evalsim.contracts import Agent, AgentType, InterventionEligibility, Scenario
from evalsim.contracts.counterfactual import M6_ANALYSIS_TRANSITIONS
from evalsim.evaluation.m6_waymax_metrics import (
    M6_WAYMAX_BASE_SEED,
    M6_WAYMAX_CELL_COUNT,
    M6_WAYMAX_DETERMINISM_CONDITIONS,
    M6_WAYMAX_DETERMINISM_ROW_COUNT,
    M6_WAYMAX_RESAMPLES,
    M6WaymaxIssuedScalarTable,
    M6WaymaxLiveDeterminismExecution,
    M6WaymaxLiveDeterminismRow,
    M6WaymaxLiveDeterminismTable,
    M6WaymaxMatrixResult,
    M6WaymaxMeasureError,
    M6WaymaxNoExecutionDeterminismRow,
    M6WaymaxNoExecutionDeterminismTable,
    M6WaymaxResamplingKey,
    M6WaymaxSceneScalar,
    M6WaymaxTwentyTransitionPairView,
    M6WaymaxVerifiedStoredSelection,
    analyze_m6_waymax_cells,
    analyze_m6_waymax_matrix,
    build_m6_waymax_data_free_determinism_table,
    build_m6_waymax_live_determinism_table,
    build_m6_waymax_scene_scalar_table,
    build_m6_waymax_unsupported_determinism_table,
    build_m6_waymax_twenty_transition_pair_view,
    compute_m6_waymax_paired_measures,
    m6_waymax_measure_contract,
    parse_m6_waymax_scene_scalar_table,
    reconstruct_m6_waymax_stored_cells,
    validate_m6_waymax_live_determinism_table,
    validate_m6_waymax_no_execution_determinism_table,
    verify_m6_waymax_stored_selection,
)
from evalsim.perturb.m6 import (
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    compile_identity_plan,
    compile_longitudinal_brake_pulse_plan,
    identity_spec,
    longitudinal_brake_pulse_spec,
)
from evalsim.simulators.waymax_m6 import (
    M6_WAYMAX_BUNDLES,
    M6_WAYMAX_LOGGED_WORLD,
    M6_WAYMAX_PRIVILEGED_IDM,
    CompactM6WaymaxRollout,
    M6WaymaxEligibility,
    M6WaymaxPrimaryDomain,
    M6WaymaxPrimaryDomainEntry,
    M6WaymaxSelection,
    build_m6_waymax_primary_domain_entry,
    build_waymax_ego_plan_view,
    compact_selected_m6_waymax_rollout,
    evaluate_m6_waymax_eligibility,
    m6_waymax_rank_sha256,
    select_m6_waymax_subset,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _b2_fingerprint() -> str:
    return longitudinal_brake_pulse_spec(
        PRIMARY_BRAKE_MAGNITUDE_MPS2
    ).configuration_fingerprint


@dataclasses.dataclass
class _FakeTrajectory:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vel_x: np.ndarray
    vel_y: np.ndarray
    yaw: np.ndarray
    valid: np.ndarray
    timestamp_micros: np.ndarray
    length: np.ndarray
    width: np.ndarray
    height: np.ndarray

    @property
    def num_objects(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_timesteps(self) -> int:
        return int(self.x.shape[1])


@dataclasses.dataclass
class _FakeMetadata:
    ids: np.ndarray
    object_types: np.ndarray
    is_sdc: np.ndarray
    is_valid: np.ndarray


@dataclasses.dataclass
class _FakeState:
    sim_trajectory: _FakeTrajectory
    log_trajectory: _FakeTrajectory
    object_metadata: _FakeMetadata
    timestep: np.ndarray

    @property
    def shape(self) -> tuple[()]:
        return ()


def _adapter_source() -> tuple[_FakeState, Scenario, InterventionEligibility]:
    objects = 128
    steps = 91
    shape = (objects, steps)
    frame = np.arange(steps, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    valid[:3] = True
    x = np.full(shape, -1000.0, dtype=np.float32)
    y = np.full(shape, -1000.0, dtype=np.float32)
    vx = np.full(shape, -1.0, dtype=np.float32)
    vy = np.full(shape, -1.0, dtype=np.float32)
    yaw = np.full(shape, -1.0, dtype=np.float32)
    x[0], x[1], x[2] = frame, frame - 10.0, frame - 30.0
    y[0], y[1], y[2] = 0.0, 0.0, 8.0
    vx[:3], vy[:3], yaw[:3] = 10.0, 0.0, 0.0
    length = np.full(shape, -1.0, dtype=np.float32)
    width = np.full(shape, -1.0, dtype=np.float32)
    height = np.full(shape, -1.0, dtype=np.float32)
    length[:3], width[:3], height[:3] = 4.0, 2.0, 1.5
    timestamp_micros = np.broadcast_to(
        np.arange(steps, dtype=np.int32) * 100_000,
        shape,
    ).copy()
    trajectory = _FakeTrajectory(
        x=x,
        y=y,
        z=np.where(valid, 0.0, -1000.0).astype(np.float32),
        vel_x=vx,
        vel_y=vy,
        yaw=yaw,
        valid=valid,
        timestamp_micros=timestamp_micros,
        length=length,
        width=width,
        height=height,
    )
    ids = np.full(objects, -1, dtype=np.int32)
    ids[:3] = (101, 102, 103)
    object_types = np.zeros(objects, dtype=np.int32)
    object_types[:3] = 1
    is_sdc = np.zeros(objects, dtype=bool)
    is_sdc[0] = True
    is_valid = np.zeros(objects, dtype=bool)
    is_valid[:3] = True
    state = _FakeState(
        sim_trajectory=trajectory,
        log_trajectory=trajectory,
        object_metadata=_FakeMetadata(
            ids=ids,
            object_types=object_types,
            is_sdc=is_sdc,
            is_valid=is_valid,
        ),
        timestep=np.asarray(0, dtype=np.int32),
    )
    timestamps = (
        np.arange(steps, dtype=np.int64) * 100_000
    ).astype(np.float64) * 1e-6
    scenario = Scenario(
        scenario_id="invented-adapter-scene",
        timestamps=timestamps,
        agents=[
            Agent(
                id=101,
                type=AgentType.VEHICLE,
                valid=np.ones(steps, dtype=bool),
                x=np.arange(steps, dtype=np.float64),
                y=np.zeros(steps),
                heading=np.zeros(steps),
                vx=np.full(steps, 10.0),
                vy=np.zeros(steps),
                length=4.0,
                width=2.0,
            ),
            Agent(
                id=102,
                type=AgentType.VEHICLE,
                valid=np.ones(steps, dtype=bool),
                x=np.arange(steps, dtype=np.float64) - 10.0,
                y=np.zeros(steps),
                heading=np.zeros(steps),
                vx=np.full(steps, 10.0),
                vy=np.zeros(steps),
                length=4.0,
                width=2.0,
            ),
            Agent(
                id=103,
                type=AgentType.VEHICLE,
                valid=np.ones(steps, dtype=bool),
                x=np.arange(steps, dtype=np.float64) - 30.0,
                y=np.full(steps, 8.0),
                heading=np.zeros(steps),
                vx=np.full(steps, 10.0),
                vy=np.zeros(steps),
                length=4.0,
                width=2.0,
            ),
        ],
        ego_index=0,
        metadata={
            "current_index": 10,
            "source": "synthetic",
            "source_version": "invented",
            "source_time_unit": "microseconds",
        },
    )
    return (
        state,
        scenario,
        InterventionEligibility.accepted((10, 50), target_index=1),
    )


def _adapter_compact(
    state: _FakeState,
    plan_view,
    *,
    bundle: str,
    responsive_target: bool = False,
) -> CompactM6WaymaxRollout:
    interval = slice(11, 31)
    trajectory = state.log_trajectory
    numeric = {
        "x": np.asarray(trajectory.x[:, interval]).T.copy(),
        "y": np.asarray(trajectory.y[:, interval]).T.copy(),
        "yaw": np.asarray(trajectory.yaw[:, interval]).T.copy(),
        "vx": np.asarray(trajectory.vel_x[:, interval]).T.copy(),
        "vy": np.asarray(trajectory.vel_y[:, interval]).T.copy(),
    }
    for name, view_name in (
        ("x", "x"),
        ("y", "y"),
        ("yaw", "heading"),
        ("vx", "vx"),
        ("vy", "vy"),
    ):
        numeric[name][:, 0] = np.asarray(getattr(plan_view, view_name)[1:])
    if responsive_target:
        numeric["vx"][1, 1] = 9.9
        numeric["vx"][2:, 1] = 9.8
        for index in range(1, 20):
            numeric["x"][index, 1] = (
                numeric["x"][index - 1, 1]
                + 0.1 * numeric["vx"][index - 1, 1]
            )
    for name in numeric:
        numeric[name] = numeric[name].astype(np.float32)
    non_sdc_vehicle = (
        ~state.object_metadata.is_sdc
        & (state.object_metadata.object_types == 1)
    )
    requested = np.zeros((20, 128), dtype=bool)
    if bundle == M6_WAYMAX_PRIVILEGED_IDM:
        requested[:, non_sdc_vehicle] = True
    effective = requested.copy()
    lifecycle = (
        np.broadcast_to(non_sdc_vehicle, (20, 128)).copy() & ~requested
    )
    return CompactM6WaymaxRollout(
        x=numeric["x"],
        y=numeric["y"],
        yaw=numeric["yaw"],
        vx=numeric["vx"],
        vy=numeric["vy"],
        valid=np.asarray(trajectory.valid[:, interval]).T.copy(),
        timestamp_micros=np.asarray(
            trajectory.timestamp_micros[:, interval]
        ).T.copy(),
        timestep=np.arange(11, 31, dtype=np.int32),
        requested_control=requested,
        effective_control=effective,
        lifecycle_fallback=lifecycle,
        initialized_overlap_excluded=np.zeros((20, 128), dtype=bool),
    )


def _adapter_domain_selection(
    state: _FakeState,
    scenario: Scenario,
    primary: InterventionEligibility,
) -> tuple[
    M6WaymaxPrimaryDomain,
    M6WaymaxSelection,
    M6WaymaxEligibility,
    int,
]:
    entries = tuple(
        build_m6_waymax_primary_domain_entry(
            state,
            scenario,
            primary,
            cohort_index=index,
        )
        for index in range(8)
    )
    qualifications = tuple(
        evaluate_m6_waymax_eligibility(
            state,
            scenario,
            primary,
            cohort_index=entry.cohort_index,
            primary_entry=entry,
        )
        for entry in entries
    )
    domain = M6WaymaxPrimaryDomain(entries)
    selection = select_m6_waymax_subset(
        qualifications,
        primary_domain=domain,
        primary_scenarios={
            entry.cohort_index: scenario for entry in entries
        },
    )
    qualification = next(
        row for row in selection.members if row.cohort_index == 4
    )
    return (
        domain,
        selection,
        qualification,
        selection.members.index(qualification),
    )


def _invented_view(
    *,
    selection_position: int,
    cohort_index: int,
    bundle: str,
    response_scale: float = 1.0,
    selection_binding_sha256: str | None = None,
    selection_member_count: int = 8,
    qualification_binding_sha256: str | None = None,
    primary_domain_sha256: str | None = None,
    scenario_id: str | None = None,
    target_agent_id: int | None = None,
) -> M6WaymaxTwentyTransitionPairView:
    frame = np.arange(21, dtype=np.float64)
    timestamps = np.arange(21, dtype=np.int64) * 100_000
    baseline_speed = np.full(21, 10.0)
    baseline_target_x = -10.0 + frame
    treatment_speed = baseline_speed.copy()
    treatment_target_x = baseline_target_x.copy()
    if bundle == M6_WAYMAX_PRIVILEGED_IDM:
        treatment_speed[2] = 10.0 - 0.1 * response_scale
        treatment_speed[3:] = 10.0 - 0.2 * response_scale
        for index in range(2, 21):
            treatment_target_x[index] = (
                treatment_target_x[index - 1]
                + 0.1 * treatment_speed[index - 1]
            )
    baseline_ego_x = frame
    treatment_ego_x = baseline_ego_x - 0.01 * frame**2
    zeros = np.zeros(21)
    selection_binding_sha256 = (
        _digest("one-selection")
        if selection_binding_sha256 is None
        else selection_binding_sha256
    )
    qualification_binding_sha256 = (
        _digest(f"qualification-{cohort_index}")
        if qualification_binding_sha256 is None
        else qualification_binding_sha256
    )
    primary_domain_sha256 = (
        _digest("one-primary-domain")
        if primary_domain_sha256 is None
        else primary_domain_sha256
    )
    scenario_id = (
        f"invented-waymax-{cohort_index}"
        if scenario_id is None
        else scenario_id
    )
    target_agent_id = (
        1000 + cohort_index
        if target_agent_id is None
        else target_agent_id
    )
    return M6WaymaxTwentyTransitionPairView(
        selection_position=selection_position,
        cohort_index=cohort_index,
        scenario_id=scenario_id,
        bundle=bundle,
        target_index=1,
        target_agent_id=target_agent_id,
        target_slot=1,
        ego_index=0,
        ego_agent_id=500,
        source_state_sha256=_digest(f"source-{cohort_index}"),
        qualification_binding_sha256=qualification_binding_sha256,
        primary_domain_sha256=primary_domain_sha256,
        selection_binding_sha256=selection_binding_sha256,
        selection_member_count=selection_member_count,
        baseline_plan_fingerprint=_digest(
            f"identity-plan-{cohort_index}"
        ),
        treatment_plan_fingerprint=_digest(
            f"brake-plan-{cohort_index}"
        ),
        baseline_configuration_fingerprint=(
            identity_spec().configuration_fingerprint
        ),
        intervention_configuration_fingerprint=(
            longitudinal_brake_pulse_spec(
                PRIMARY_BRAKE_MAGNITUDE_MPS2
            ).configuration_fingerprint
        ),
        baseline_perturbation_identity=(
            f"identity/v1@sha256:{_digest(f'identity-plan-{cohort_index}')}"
        ),
        treatment_perturbation_identity=(
            "longitudinal_brake_pulse/v1@sha256:"
            f"{_digest(f'brake-plan-{cohort_index}')}"
        ),
        target_length_m=4.0,
        ego_length_m=4.0,
        timestamps_micros=timestamps,
        target_valid=np.ones(21, dtype=bool),
        ego_valid=np.ones(21, dtype=bool),
        baseline_ego_x=baseline_ego_x,
        baseline_ego_y=zeros,
        baseline_ego_heading=zeros,
        baseline_ego_vx=baseline_speed,
        baseline_ego_vy=zeros,
        treatment_ego_x=treatment_ego_x,
        treatment_ego_y=zeros,
        treatment_ego_heading=zeros,
        treatment_ego_vx=np.gradient(treatment_ego_x, 0.1),
        treatment_ego_vy=zeros,
        baseline_target_x=baseline_target_x,
        baseline_target_y=zeros,
        baseline_target_heading=zeros,
        baseline_target_vx=baseline_speed,
        baseline_target_vy=zeros,
        treatment_target_x=treatment_target_x,
        treatment_target_y=zeros,
        treatment_target_heading=zeros,
        treatment_target_vx=treatment_speed,
        treatment_target_vy=zeros,
        world_pair_gate_sha256=_digest(
            f"world-gate-{bundle}-{cohort_index}"
        ),
        _issuance_capability=_waymax_metrics._PAIR_VIEW_ISSUER,
    )


def _invented_selection_fixture(
    n: int,
) -> tuple[M6WaymaxSelection, M6WaymaxPrimaryDomain]:
    cohort_indices = sorted(
        range(128),
        key=lambda index: (bytes.fromhex(m6_waymax_rank_sha256(index)), index),
    )[:max(n, 1)]
    template_state, template, primary = _adapter_source()
    entries = []
    rows = []
    scenarios = {}
    for cohort_index in cohort_indices:
        state = copy.deepcopy(template_state)
        scenario = copy.deepcopy(template)
        scenario.scenario_id = f"invented-waymax-{cohort_index}"
        entry = build_m6_waymax_primary_domain_entry(
            state,
            scenario,
            primary,
            cohort_index=cohort_index,
        )
        entries.append(entry)
        scenarios[cohort_index] = scenario
        rows.append(
            evaluate_m6_waymax_eligibility(
                state,
                scenario,
                primary,
                cohort_index=cohort_index,
                primary_entry=entry,
            )
        )
    domain = M6WaymaxPrimaryDomain(tuple(entries))
    selection = select_m6_waymax_subset(
        rows,
        primary_domain=domain,
        primary_scenarios=scenarios,
    )
    return selection, domain


def _invented_selection(n: int) -> M6WaymaxSelection:
    selection, _ = _invented_selection_fixture(n)
    return selection


def _invented_views(
    selection: M6WaymaxSelection,
) -> tuple[M6WaymaxTwentyTransitionPairView, ...]:
    rows = []
    selection_binding = _waymax_metrics._selection_binding_sha256(selection)
    for position, member in enumerate(selection.members):
        for bundle in M6_WAYMAX_BUNDLES:
            rows.append(
                _invented_view(
                    selection_position=position,
                    cohort_index=member.cohort_index,
                    bundle=bundle,
                    response_scale=1.0 + position / 20.0,
                    selection_binding_sha256=selection_binding,
                    selection_member_count=len(selection.members),
                    qualification_binding_sha256=(
                        member.qualification_binding_sha256
                    ),
                    primary_domain_sha256=selection.primary_domain_sha256,
                    scenario_id=member.scenario_id,
                    target_agent_id=member.target_agent_id,
                )
            )
    return tuple(rows)


def _determinism_compact(token: int) -> CompactM6WaymaxRollout:
    shape = (20, 128)
    x = np.zeros(shape, dtype=np.float32)
    x[0, 0] = np.float32(token)
    timestamps = np.broadcast_to(
        np.arange(1, 21, dtype=np.int64)[:, np.newaxis] * 100_000,
        shape,
    ).copy()
    return CompactM6WaymaxRollout(
        x=x,
        y=np.zeros(shape, dtype=np.float32),
        yaw=np.zeros(shape, dtype=np.float32),
        vx=np.zeros(shape, dtype=np.float32),
        vy=np.zeros(shape, dtype=np.float32),
        valid=np.ones(shape, dtype=bool),
        timestamp_micros=timestamps,
        timestep=np.arange(1, 21, dtype=np.int32),
        requested_control=np.zeros(shape, dtype=bool),
        effective_control=np.zeros(shape, dtype=bool),
        lifecycle_fallback=np.zeros(shape, dtype=bool),
        initialized_overlap_excluded=np.zeros(shape, dtype=bool),
    )


def _live_determinism_executions(
    selection: M6WaymaxSelection,
) -> tuple[M6WaymaxLiveDeterminismExecution, ...]:
    rows = []
    for position, member in enumerate(selection.members):
        for bundle_index, bundle in enumerate(M6_WAYMAX_BUNDLES):
            for condition_index, condition in enumerate(
                M6_WAYMAX_DETERMINISM_CONDITIONS
            ):
                token = 1 + position * 4 + bundle_index * 2 + condition_index
                rows.append(
                    M6WaymaxLiveDeterminismExecution(
                        selection_position=position,
                        bundle=bundle,
                        condition=condition,
                        qualification=member,
                        eager_pass_1=_determinism_compact(token),
                        eager_pass_2=_determinism_compact(token),
                        jit_eager=(
                            _determinism_compact(token)
                            if position == 0
                            else None
                        ),
                        jit_compiled=(
                            _determinism_compact(token)
                            if position == 0
                            else None
                        ),
                    )
                )
    return tuple(rows)


def _replace_issued_view(
    view: M6WaymaxTwentyTransitionPairView,
    **changes,
) -> M6WaymaxTwentyTransitionPairView:
    return dataclasses.replace(
        view,
        _issuance_capability=_waymax_metrics._PAIR_VIEW_ISSUER,
        **changes,
    )


def _independent_formula_oracle(
    view: M6WaymaxTwentyTransitionPairView,
) -> dict[str, float]:
    dt = np.diff(view.timestamps_micros).astype(float) / 1_000_000.0
    baseline_speed = np.sqrt(
        view.baseline_target_vx**2 + view.baseline_target_vy**2
    )
    treatment_speed = np.sqrt(
        view.treatment_target_vx**2 + view.treatment_target_vy**2
    )
    baseline_acceleration = np.diff(baseline_speed) / dt
    treatment_acceleration = np.diff(treatment_speed) / dt
    impulse = sum(
        max(0.0, float(left - right)) * float(duration)
        for left, right, duration in zip(
            baseline_acceleration,
            treatment_acceleration,
            dt,
            strict=True,
        )
    )
    event_time = None
    delta = treatment_acceleration - baseline_acceleration
    for start in range(1, 20):
        duration = 0.0
        for end in range(start, 20):
            if float(delta[end]) > -0.5:
                break
            duration += float(dt[end])
            if duration >= 0.2:
                event_time = (
                    int(view.timestamps_micros[end + 1])
                    - int(view.timestamps_micros[0])
                ) / 1_000_000.0
                break
        if event_time is not None:
            break
    window = (
        int(view.timestamps_micros[-1])
        - int(view.timestamps_micros[0])
    ) / 1_000_000.0
    timeliness = 0.0 if event_time is None else window - min(event_time, window)

    speed = math.hypot(
        float(view.baseline_target_vx[0]),
        float(view.baseline_target_vy[0]),
    )
    if speed > 1e-12:
        hx = float(view.baseline_target_vx[0]) / speed
        hy = float(view.baseline_target_vy[0]) / speed
    else:
        hx = math.cos(float(view.baseline_target_heading[0]))
        hy = math.sin(float(view.baseline_target_heading[0]))
    half_length = 0.5 * (view.ego_length_m + view.target_length_m)
    baseline_gaps = [
        (
            (view.baseline_ego_x[index] - view.baseline_target_x[index]) * hx
            + (view.baseline_ego_y[index] - view.baseline_target_y[index]) * hy
            - half_length
        )
        for index in range(1, 21)
    ]
    treatment_gaps = [
        (
            (view.treatment_ego_x[index] - view.treatment_target_x[index]) * hx
            + (view.treatment_ego_y[index] - view.treatment_target_y[index]) * hy
            - half_length
        )
        for index in range(1, 21)
    ]
    gap_change = min(treatment_gaps) - min(baseline_gaps)

    heading = float(view.baseline_target_heading[0])
    progress_hx, progress_hy = math.cos(heading), math.sin(heading)
    origin_x = float(view.baseline_target_x[0])
    origin_y = float(view.baseline_target_y[0])
    baseline_progress = (
        (float(view.baseline_target_x[-1]) - origin_x) * progress_hx
        + (float(view.baseline_target_y[-1]) - origin_y) * progress_hy
    )
    treatment_progress = (
        (float(view.treatment_target_x[-1]) - origin_x) * progress_hx
        + (float(view.treatment_target_y[-1]) - origin_y) * progress_hy
    )
    return {
        "additional_target_braking_impulse_mps": impulse,
        "response_timeliness_s": timeliness,
        "minimum_longitudinal_bumper_gap_change_m": gap_change,
        "target_progress_loss_m": baseline_progress - treatment_progress,
    }


def test_twenty_transition_contract_is_separate_and_immutable() -> None:
    view = _invented_view(
        selection_position=0,
        cohort_index=7,
        bundle=M6_WAYMAX_PRIVILEGED_IDM,
    )
    assert M6_ANALYSIS_TRANSITIONS == 40
    assert view.transition_count == 20
    assert view.timestamps_micros.shape == (21,)
    assert not view.baseline_target_x.flags.writeable
    assert m6_waymax_measure_contract()["store_scalar_row_count"] == 128
    with pytest.raises(ValueError):
        view.baseline_target_x[2] = 99.0

    replacement = np.asarray(view.baseline_target_x).copy()
    replacement[2] += 1.0
    object.__setattr__(view, "baseline_target_x", replacement)
    with pytest.raises(M6WaymaxMeasureError, match="pair_view_mutated"):
        compute_m6_waymax_paired_measures(view)


def test_four_formulas_match_independent_oracle_for_both_bundles() -> None:
    for bundle in M6_WAYMAX_BUNDLES:
        view = _invented_view(
            selection_position=0,
            cohort_index=9,
            bundle=bundle,
        )
        expected = _independent_formula_oracle(view)
        actual = {
            row.metric_name: row.value
            for row in compute_m6_waymax_paired_measures(view)
        }
        assert actual == pytest.approx(expected, abs=1e-12)
        if bundle == M6_WAYMAX_LOGGED_WORLD:
            assert actual["additional_target_braking_impulse_mps"] == 0.0
            assert actual["response_timeliness_s"] == 0.0
            assert actual["target_progress_loss_m"] == 0.0
        else:
            assert actual["additional_target_braking_impulse_mps"] > 0.0
            assert actual["response_timeliness_s"] == pytest.approx(1.7)
            timeliness = compute_m6_waymax_paired_measures(view)[1]
            assert timeliness.responded is True
            assert timeliness.responder_latency_s == pytest.approx(0.3)


def test_nonfinite_t_plus_2_and_logged_world_attacks_fail_closed() -> None:
    idm = _invented_view(
        selection_position=0,
        cohort_index=2,
        bundle=M6_WAYMAX_PRIVILEGED_IDM,
    )
    nonfinite = np.asarray(idm.treatment_target_vx).copy()
    nonfinite[20] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _replace_issued_view(
            idm,
            treatment_target_vx=nonfinite,
            view_binding_sha256=None,
        )

    early = np.asarray(idm.treatment_target_x).copy()
    early[1] += 0.01
    with pytest.raises(ValueError, match=r"t\+2"):
        _replace_issued_view(
            idm,
            treatment_target_x=early,
            view_binding_sha256=None,
        )

    logged = _invented_view(
        selection_position=0,
        cohort_index=2,
        bundle=M6_WAYMAX_LOGGED_WORLD,
    )
    response = np.asarray(logged.treatment_target_vx).copy()
    response[10] -= 0.1
    with pytest.raises(ValueError, match="nonreactive"):
        _replace_issued_view(
            logged,
            treatment_target_vx=response,
            view_binding_sha256=None,
        )


def test_safe_store_projection_roundtrip_excludes_private_execution_fields() -> None:
    selection = _invented_selection(8)
    table = build_m6_waymax_scene_scalar_table(
        _invented_views(selection),
        selection=selection,
    )
    assert isinstance(table, M6WaymaxIssuedScalarTable)
    assert len(table) == 128
    selected = table[0]
    payload = selected.to_store_dict()
    assert set(payload) == M6WaymaxSceneScalar.STORE_FIELDS
    assert not {
        "scenario_id",
        "target_agent_id",
        "source_state_sha256",
        "view_binding_sha256",
        "result_binding_sha256",
    } & set(payload)
    roundtrip = M6WaymaxSceneScalar.from_store_dict(
        json.loads(selected.canonical_json)
    )
    assert roundtrip.to_store_dict() == payload
    assert roundtrip.scalar_binding_sha256 == selected.scalar_binding_sha256
    assert selected.primary_domain_sha256 == selection.primary_domain_sha256
    assert selected.selection_binding_sha256 == (
        _waymax_metrics._selection_binding_sha256(selection)
    )
    assert selected.identity_configuration_fingerprint == (
        identity_spec().configuration_fingerprint
    )
    assert selected.primary_b2_configuration_fingerprint == _b2_fingerprint()
    assert all(
        row.cohort_index is None or 0 <= row.cohort_index <= 127
        for row in table
    )
    assert sum(row.status == "selected" for row in table) == 64
    assert sum(row.status == "not_selected" for row in table) == 64
    assert all(
        row.value is None and row.cohort_index is None
        for row in table
        if row.status == "not_selected"
    )


@pytest.mark.parametrize(
    ("pair_n", "expected_status"),
    [
        (8, "insufficient_n"),
        (9, "insufficient_n"),
        (10, "descriptive"),
    ],
)
def test_exact_eight_cells_and_small_n_suppression(
    pair_n: int,
    expected_status: str,
) -> None:
    selection = _invented_selection(pair_n)
    result = analyze_m6_waymax_matrix(
        _invented_views(selection),
        selection=selection,
    )
    assert result.pair_n == pair_n
    assert len(result.scene_scalars) == 128
    assert len(result.cells) == M6_WAYMAX_CELL_COUNT
    assert result.status == expected_status
    identities = {
        (cell.bundle, cell.metric_name) for cell in result.cells
    }
    assert len(identities) == 8
    assert all(not cell.directional_language_allowed for cell in result.cells)
    if pair_n < 10:
        assert all(cell.arithmetic_mean is None for cell in result.cells)
        assert all(cell.pointwise_band is None for cell in result.cells)
        assert all(cell.resampling_key is None for cell in result.cells)
    else:
        assert all(cell.arithmetic_mean is not None for cell in result.cells)
        assert all(cell.pointwise_band is not None for cell in result.cells)
        assert all(
            cell.resampling_key is not None
            and json.loads(cell.resampling_key.canonical_json)["resamples"]
            == M6_WAYMAX_RESAMPLES
            for cell in result.cells
        )


def test_zero_selected_safe_table_is_unsupported_with_exact_eight_cells() -> None:
    selection = _invented_selection(7)
    empty = build_m6_waymax_scene_scalar_table(
        (),
        selection=selection,
    )
    assert len(empty) == 128
    assert all(row.status == "not_selected" and row.value is None for row in empty)
    result = analyze_m6_waymax_cells(
        empty,
        selection=selection,
        intervention_configuration_fingerprint=_b2_fingerprint(),
    )
    assert result.pair_n == 0
    assert result.cohort_indices == ()
    assert len(result.cells) == 8
    assert all(cell.status == "unsupported" for cell in result.cells)
    assert all(cell.arithmetic_mean is None for cell in result.cells)


def test_safe_cell_analysis_is_deterministic_and_matches_rng_oracle() -> None:
    selection = _invented_selection(10)
    views = _invented_views(selection)
    table = build_m6_waymax_scene_scalar_table(
        views,
        selection=selection,
    )
    forward = analyze_m6_waymax_cells(
        table,
        selection=selection,
        intervention_configuration_fingerprint=_b2_fingerprint(),
    )
    with pytest.raises(TypeError, match="IssuedScalarTable"):
        analyze_m6_waymax_cells(
            tuple(reversed(table)),  # type: ignore[arg-type]
            selection=selection,
            intervention_configuration_fingerprint=_b2_fingerprint(),
        )
    parsed = parse_m6_waymax_scene_scalar_table(
        tuple(reversed(table))
    )
    stored = reconstruct_m6_waymax_stored_cells(
        parsed,
        selection=selection,
        verified_selection_binding_sha256=(
            _waymax_metrics._selection_binding_sha256(selection)
        ),
        intervention_configuration_fingerprint=_b2_fingerprint(),
    )
    assert stored.promotable is False
    assert stored.pair_n == forward.pair_n

    cell = next(
        row
        for row in forward.cells
        if row.bundle == M6_WAYMAX_PRIVILEGED_IDM
        and row.metric_name == "target_progress_loss_m"
    )
    assert cell.resampling_key is not None
    selected = sorted(
        (
            row
            for row in table
            if row.status == "selected"
            and row.bundle == cell.bundle
            and row.metric_name == cell.metric_name
        ),
        key=lambda row: int(row.cohort_index),
    )
    values = np.asarray([row.value for row in selected], dtype=np.float64)
    payload = {
        "base_seed": 20260729,
        "bundle": (
            "waymax_privileged_logged_trajectory_waypoint_following_idm"
        ),
        "intervention_configuration_fingerprint": _b2_fingerprint(),
        "metric_name": "target_progress_loss_m",
        "metric_version": "1.0.0",
        "paired_n": 10,
        "policy_access_role": (
            "privileged_logged_trajectory_waypoint_following"
        ),
        "policy_name": (
            "waymax_privileged_logged_trajectory_waypoint_following_idm"
        ),
        "resamples": 10_000,
        "statistics_schema_version": (
            "m6-waymax-paired-statistics-1.0.0"
        ),
    }
    assert json.loads(cell.resampling_key.canonical_json) == payload
    expected_digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).digest()
    words = struct.unpack(">8I", expected_digest)
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([M6_WAYMAX_BASE_SEED, *words])
        )
    )
    draws = rng.integers(
        0,
        len(values),
        size=(M6_WAYMAX_RESAMPLES, len(values)),
        dtype=np.int64,
    )
    sampled = np.mean(values[draws], axis=1, dtype=np.float64)
    expected_band = np.quantile(sampled, [0.025, 0.975], method="linear")
    assert cell.pointwise_band is not None
    assert cell.pointwise_band.lower == expected_band[0]
    assert cell.pointwise_band.upper == expected_band[1]


def test_pairing_relabel_and_safe_scalar_mutation_attacks_fail_closed() -> None:
    selection = _invented_selection(8)
    views = list(_invented_views(selection))
    with pytest.raises(M6WaymaxMeasureError, match="selection_completeness"):
        build_m6_waymax_scene_scalar_table(
            views[:-1],
            selection=selection,
        )
    with pytest.raises(M6WaymaxMeasureError, match="selection_completeness"):
        build_m6_waymax_scene_scalar_table(
            (*views, views[0]),
            selection=selection,
        )

    relabeled = _replace_issued_view(
        views[1],
        scenario_id="self-rehashed-other-scene",
        view_binding_sha256=None,
    )
    with pytest.raises(M6WaymaxMeasureError, match="cross_bundle_pairing"):
        build_m6_waymax_scene_scalar_table(
            (views[0], relabeled, *views[2:]),
            selection=selection,
        )

    issued = build_m6_waymax_scene_scalar_table(
        views,
        selection=selection,
    )
    source_row = issued[0]
    object.__setattr__(source_row, "value", float(source_row.value) + 1.0)
    with pytest.raises(M6WaymaxMeasureError, match="scene_scalar_mutated"):
        analyze_m6_waymax_cells(
            issued,
            selection=selection,
            intervention_configuration_fingerprint=_b2_fingerprint(),
        )


def test_safe_analyzer_rejects_qualification_and_selection_cherry_picking() -> None:
    selection = _invented_selection(8)
    table = list(
        build_m6_waymax_scene_scalar_table(
            _invented_views(selection),
            selection=selection,
        )
    )
    # A self-consistent relabel in only one of the eight position rows is still
    # rejected by cross-measure qualification pairing.
    table[1] = dataclasses.replace(
        table[1],
        qualification_binding_sha256=_digest("forged-qualification"),
        scalar_binding_sha256=None,
    )
    with pytest.raises(M6WaymaxMeasureError, match="qualification_pairing"):
        parse_m6_waymax_scene_scalar_table(table)

    # A globally self-rehashed position is structurally parseable post-seal, but
    # cannot cross the verified canonical selection ledger.
    table = list(
        build_m6_waymax_scene_scalar_table(
            _invented_views(selection),
            selection=selection,
        )
    )
    for index, row in enumerate(table):
        if row.selection_position == 0:
            table[index] = dataclasses.replace(
                row,
                qualification_binding_sha256=_digest(
                    "globally-forged-qualification"
                ),
                scalar_binding_sha256=None,
            )
    parsed = parse_m6_waymax_scene_scalar_table(table)
    with pytest.raises(
        M6WaymaxMeasureError,
        match="selection_cross_binding",
    ):
        reconstruct_m6_waymax_stored_cells(
            parsed,
            selection=selection,
            verified_selection_binding_sha256=(
                _waymax_metrics._selection_binding_sha256(selection)
            ),
            intervention_configuration_fingerprint=_b2_fingerprint(),
        )

    table = list(
        build_m6_waymax_scene_scalar_table(
            _invented_views(selection),
            selection=selection,
        )
    )
    # Removing one selected scalar while retaining the other seven at that position
    # is a mixed-status selection attack, not a sparse cell.
    victim = table[0]
    table[0] = M6WaymaxSceneScalar(
        selection_position=victim.selection_position,
        cohort_index=None,
        qualification_binding_sha256=None,
        primary_domain_sha256=victim.primary_domain_sha256,
        selection_binding_sha256=victim.selection_binding_sha256,
        selection_supported=victim.selection_supported,
        selection_member_count=victim.selection_member_count,
        identity_configuration_fingerprint=(
            victim.identity_configuration_fingerprint
        ),
        primary_b2_configuration_fingerprint=(
            victim.primary_b2_configuration_fingerprint
        ),
        bundle=victim.bundle,
        metric_name=victim.metric_name,
        metric_version=victim.metric_version,
        value_unit=victim.value_unit,
        value=None,
        responded=None,
        responder_latency_s=None,
        source_pairing_complete=False,
        status="not_selected",
    )
    with pytest.raises(M6WaymaxMeasureError, match="selection_pairing"):
        parse_m6_waymax_scene_scalar_table(table)


def test_pair_cell_and_matrix_results_are_factory_issued() -> None:
    selection = _invented_selection(10)
    views = _invented_views(selection)
    with pytest.raises(TypeError, match="builder-issued"):
        dataclasses.replace(views[0])

    result = analyze_m6_waymax_matrix(views, selection=selection)
    with pytest.raises(TypeError, match="analyzer-issued"):
        dataclasses.replace(result.cells[0])
    with pytest.raises(TypeError, match="analyzer-issued"):
        dataclasses.replace(result)


def test_issued_and_parsed_scalar_trust_boundaries_are_disjoint() -> None:
    selection = _invented_selection(8)
    issued = build_m6_waymax_scene_scalar_table(
        _invented_views(selection),
        selection=selection,
    )
    with pytest.raises(TypeError, match="builder-issued"):
        M6WaymaxIssuedScalarTable(
            rows=issued.rows,
            selection_binding_sha256=issued.selection_binding_sha256,
            primary_domain_sha256=issued.primary_domain_sha256,
        )

    forged_rows = list(issued.rows)
    source = forged_rows[0]
    forged_rows[0] = dataclasses.replace(
        source,
        value=float(source.value) + 1.0,
        scalar_binding_sha256=None,
    )
    parsed = parse_m6_waymax_scene_scalar_table(forged_rows)
    assert parsed.promotable is False
    receipt = reconstruct_m6_waymax_stored_cells(
        parsed,
        selection=selection,
        verified_selection_binding_sha256=(
            _waymax_metrics._selection_binding_sha256(selection)
        ),
        intervention_configuration_fingerprint=_b2_fingerprint(),
    )
    assert receipt.promotable is False
    with pytest.raises(TypeError, match="IssuedScalarTable"):
        analyze_m6_waymax_cells(
            parsed,  # type: ignore[arg-type]
            selection=selection,
            intervention_configuration_fingerprint=_b2_fingerprint(),
        )

    # Even after recomputing every public hash, a changed value cannot replace
    # the original builder-issued live evidence.
    object.__setattr__(issued, "rows", tuple(forged_rows))
    public_rehash = _waymax_metrics._scalar_table_binding_sha256(
        domain=_waymax_metrics._ISSUED_SCALAR_TABLE_DOMAIN,
        rows=forged_rows,
        selection_binding_sha256=issued.selection_binding_sha256,
    )
    object.__setattr__(issued, "table_binding_sha256", public_rehash)
    with pytest.raises(
        M6WaymaxMeasureError,
        match="issued_scalar_table_mutated",
    ):
        issued.revalidate(selection=selection)


def test_data_free_determinism_placeholder_is_exact_and_unforgeable() -> None:
    table = build_m6_waymax_data_free_determinism_table()
    repeated = build_m6_waymax_data_free_determinism_table()
    assert isinstance(table, M6WaymaxNoExecutionDeterminismTable)
    assert table.reason == "data_free"
    assert table.promotable is False
    assert table.selection_binding_sha256 is None
    assert table.primary_domain_sha256 is None
    assert table.table_binding_sha256 == repeated.table_binding_sha256
    assert len(table) == M6_WAYMAX_DETERMINISM_ROW_COUNT == 64
    assert tuple(
        (row.selection_position, row.bundle, row.condition)
        for row in table
    ) == tuple(
        (position, bundle, condition)
        for position in range(16)
        for bundle in M6_WAYMAX_BUNDLES
        for condition in M6_WAYMAX_DETERMINISM_CONDITIONS
    )
    store_rows = table.to_store_rows()
    assert all(
        row["status"] == "not_applicable"
        and all(
            row[name] is None
            for name in (
                "cohort_index",
                "qualification_binding_sha256",
                "eager_pass_1_sha256",
                "eager_pass_2_sha256",
                "jit_eager_sha256",
                "jit_compiled_sha256",
            )
        )
        for row in store_rows
    )
    assert (
        m6_waymax_measure_contract()["live_determinism_issuance_available"]
        is True
    )

    invented = _digest("caller-invented-equal-execution")
    with pytest.raises(TypeError, match="factory-issued"):
        M6WaymaxNoExecutionDeterminismRow(
            selection_position=0,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            condition="identity",
            cohort_index=0,  # type: ignore[arg-type]
            qualification_binding_sha256=invented,  # type: ignore[arg-type]
            status="passed",  # type: ignore[arg-type]
            eager_pass_1_sha256=invented,  # type: ignore[arg-type]
            eager_pass_2_sha256=invented,  # type: ignore[arg-type]
            jit_eager_sha256=invented,  # type: ignore[arg-type]
            jit_compiled_sha256=invented,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="factory-issued"):
        M6WaymaxNoExecutionDeterminismTable(
            reason="data_free",
            rows=table.rows,
            selection_binding_sha256=None,
            primary_domain_sha256=None,
        )
    with pytest.raises(TypeError, match="factory-issued"):
        dataclasses.replace(table.rows[0])
    with pytest.raises(TypeError, match="factory-issued"):
        dataclasses.replace(table)

    invented_rows = [dict(row) for row in store_rows]
    for row in invented_rows:
        row["status"] = "passed"
        row["eager_pass_1_sha256"] = invented
        row["eager_pass_2_sha256"] = invented
        if row["selection_position"] == 0:
            row["jit_eager_sha256"] = invented
            row["jit_compiled_sha256"] = invented
    with pytest.raises(TypeError, match="factory-issued"):
        validate_m6_waymax_no_execution_determinism_table(invented_rows)
    invented_rows[0]["eager_pass_2_sha256"] = None
    invented_rows[0]["jit_compiled_sha256"] = None
    with pytest.raises(TypeError, match="factory-issued"):
        validate_m6_waymax_no_execution_determinism_table(invented_rows)

    mutated = copy.deepcopy(table)
    object.__setattr__(mutated.rows[0], "status", "passed")
    with pytest.raises(
        M6WaymaxMeasureError,
        match="no_execution_determinism_row_mutated",
    ):
        validate_m6_waymax_no_execution_determinism_table(mutated)

    self_rehashed_row = copy.deepcopy(table.rows[0])
    object.__setattr__(self_rehashed_row, "status", "passed")
    object.__setattr__(
        self_rehashed_row,
        "row_binding_sha256",
        self_rehashed_row._binding_sha256(),
    )
    with pytest.raises(
        M6WaymaxMeasureError,
        match="no_execution_determinism_row_mutated",
    ):
        self_rehashed_row.revalidate()

    rebound = copy.deepcopy(table)
    object.__setattr__(rebound, "reason", "unsupported_selection")
    object.__setattr__(
        rebound,
        "selection_binding_sha256",
        _digest("invented-selection"),
    )
    object.__setattr__(
        rebound,
        "primary_domain_sha256",
        _digest("invented-domain"),
    )
    object.__setattr__(
        rebound,
        "table_binding_sha256",
        _waymax_metrics._no_execution_determinism_table_sha256(
            reason=rebound.reason,
            selection_binding_sha256=rebound.selection_binding_sha256,
            primary_domain_sha256=rebound.primary_domain_sha256,
            rows=rebound.rows,
        ),
    )
    with pytest.raises(
        M6WaymaxMeasureError,
        match="no_execution_determinism_table_mutated",
    ):
        validate_m6_waymax_no_execution_determinism_table(rebound)


def test_unsupported_determinism_placeholder_binds_canonical_selection(
    monkeypatch,
) -> None:
    selection, domain = _invented_selection_fixture(7)
    assert not selection.supported
    table = build_m6_waymax_unsupported_determinism_table(
        selection=selection,
        primary_domain=domain,
    )
    assert table.reason == "unsupported_selection"
    assert table.promotable is False
    assert table.selection_binding_sha256 == (
        _waymax_metrics._selection_binding_sha256(selection)
    )
    assert table.primary_domain_sha256 == domain.domain_sha256
    assert all(row.status == "not_applicable" for row in table)
    assert (
        table.to_store_rows()
        == build_m6_waymax_data_free_determinism_table().to_store_rows()
    )
    assert (
        table.table_binding_sha256
        != build_m6_waymax_data_free_determinism_table().table_binding_sha256
    )
    assert (
        validate_m6_waymax_no_execution_determinism_table(
            table,
            selection=selection,
            primary_domain=domain,
        )
        is table
    )

    wrong_selection, wrong_domain = _invented_selection_fixture(6)
    with pytest.raises(M6WaymaxMeasureError):
        validate_m6_waymax_no_execution_determinism_table(
            table,
            selection=wrong_selection,
            primary_domain=wrong_domain,
        )

    placeholder_calls = 0

    def forbidden_placeholder_rows():
        nonlocal placeholder_calls
        placeholder_calls += 1
        raise AssertionError("supported selection must issue no placeholder")

    monkeypatch.setattr(
        _waymax_metrics,
        "_build_no_execution_determinism_rows",
        forbidden_placeholder_rows,
    )
    supported, supported_domain = _invented_selection_fixture(8)
    with pytest.raises(
        M6WaymaxMeasureError,
        match="determinism_live_unavailable",
    ):
        build_m6_waymax_unsupported_determinism_table(
            selection=supported,
            primary_domain=supported_domain,
        )
    assert placeholder_calls == 0

    mutated_selection = copy.deepcopy(selection)
    object.__setattr__(mutated_selection, "eligible_count", 8)
    with pytest.raises(ValueError, match="selection_mutated"):
        build_m6_waymax_unsupported_determinism_table(
            selection=mutated_selection,
            primary_domain=domain,
        )
    assert placeholder_calls == 0


def test_live_determinism_table_is_exact_bound_and_factory_issued() -> None:
    selection, domain = _invented_selection_fixture(8)
    table = build_m6_waymax_live_determinism_table(
        _live_determinism_executions(selection),
        selection=selection,
        primary_domain=domain,
    )
    assert isinstance(table, M6WaymaxLiveDeterminismTable)
    assert table.promotable is True
    assert table.selected_member_count == 8
    assert len(table) == M6_WAYMAX_DETERMINISM_ROW_COUNT == 64
    assert table.selection_binding_sha256 == (
        _waymax_metrics._selection_binding_sha256(selection)
    )
    assert table.primary_domain_sha256 == domain.domain_sha256
    assert (
        validate_m6_waymax_live_determinism_table(
            table,
            selection=selection,
            primary_domain=domain,
        )
        is table
    )
    store_rows = table.to_store_rows()
    assert all(
        set(row) == M6WaymaxLiveDeterminismRow.STORE_FIELDS
        for row in store_rows
    )
    for row in table:
        if row.selection_position < len(selection.members):
            member = selection.members[row.selection_position]
            assert row.status == "passed"
            assert row.cohort_index == member.cohort_index
            assert row.qualification_binding_sha256 == (
                member.qualification_binding_sha256
            )
            assert row.eager_pass_1_sha256 == row.eager_pass_2_sha256
            if row.selection_position == 0:
                assert row.jit_eager_sha256 == row.eager_pass_1_sha256
                assert row.jit_compiled_sha256 == row.jit_eager_sha256
            else:
                assert row.jit_eager_sha256 is None
                assert row.jit_compiled_sha256 is None
        else:
            assert row.status == "not_applicable"
            assert all(
                getattr(row, name) is None
                for name in (
                    "cohort_index",
                    "qualification_binding_sha256",
                    "eager_pass_1_sha256",
                    "eager_pass_2_sha256",
                    "jit_eager_sha256",
                    "jit_compiled_sha256",
                )
            )
    with pytest.raises(TypeError, match="factory-issued"):
        dataclasses.replace(table.rows[0])
    with pytest.raises(TypeError, match="factory-issued"):
        dataclasses.replace(table)


def test_live_determinism_rejects_hash_invention_replay_and_output_drift() -> None:
    selection, domain = _invented_selection_fixture(8)
    executions = _live_determinism_executions(selection)
    with pytest.raises(TypeError, match="M6WaymaxLiveDeterminismExecution"):
        build_m6_waymax_live_determinism_table(
            tuple({"eager_pass_1_sha256": _digest("invented")} for _ in executions),
            selection=selection,
            primary_domain=domain,
        )

    first = executions[0]
    with pytest.raises(M6WaymaxMeasureError, match="eager_replay"):
        dataclasses.replace(first, eager_pass_2=first.eager_pass_1)

    drifted = copy.deepcopy(first.eager_pass_2)
    drifted.x[0, 0] += np.float32(1.0)
    mismatched = list(executions)
    mismatched[0] = dataclasses.replace(first, eager_pass_2=drifted)
    with pytest.raises(M6WaymaxMeasureError, match="eager_mismatch"):
        build_m6_waymax_live_determinism_table(
            tuple(mismatched), selection=selection, primary_domain=domain
        )

    jit_drifted = copy.deepcopy(first.jit_compiled)
    assert jit_drifted is not None
    jit_drifted.x[0, 0] += np.float32(2.0)
    mismatched[0] = dataclasses.replace(first, jit_compiled=jit_drifted)
    with pytest.raises(M6WaymaxMeasureError, match="jit_mismatch"):
        build_m6_waymax_live_determinism_table(
            tuple(mismatched), selection=selection, primary_domain=domain
        )

    replayed = list(executions)
    replayed[1] = dataclasses.replace(
        replayed[1], eager_pass_1=first.eager_pass_1
    )
    with pytest.raises(M6WaymaxMeasureError, match="execution_replay"):
        build_m6_waymax_live_determinism_table(
            tuple(replayed), selection=selection, primary_domain=domain
        )


def test_live_determinism_rejects_incomplete_wrong_selection_and_mutation() -> None:
    selection, domain = _invented_selection_fixture(8)
    executions = _live_determinism_executions(selection)
    with pytest.raises(ValueError, match="canonical selected"):
        build_m6_waymax_live_determinism_table(
            executions[:-1], selection=selection, primary_domain=domain
        )

    wrong_qualification = list(executions)
    wrong_qualification[4] = dataclasses.replace(
        wrong_qualification[4], qualification=selection.members[0]
    )
    with pytest.raises(M6WaymaxMeasureError, match="selection_mismatch"):
        build_m6_waymax_live_determinism_table(
            tuple(wrong_qualification),
            selection=selection,
            primary_domain=domain,
        )

    table = build_m6_waymax_live_determinism_table(
        executions, selection=selection, primary_domain=domain
    )
    wrong_selection, wrong_domain = _invented_selection_fixture(9)
    with pytest.raises(M6WaymaxMeasureError, match="selection_mismatch"):
        validate_m6_waymax_live_determinism_table(
            table,
            selection=wrong_selection,
            primary_domain=wrong_domain,
        )

    mutated = copy.deepcopy(table)
    object.__setattr__(
        mutated.rows[0],
        "eager_pass_1_sha256",
        _digest("mutated-live-row"),
    )
    with pytest.raises(M6WaymaxMeasureError, match="row_mutated"):
        validate_m6_waymax_live_determinism_table(
            mutated, selection=selection, primary_domain=domain
        )


def test_verified_stored_selection_is_nonpromotable_reconstruction_only() -> None:
    selection = _invented_selection(8)
    issued = build_m6_waymax_scene_scalar_table(
        _invented_views(selection),
        selection=selection,
    )
    parsed = parse_m6_waymax_scene_scalar_table(issued.rows)
    members = tuple(
        (
            position,
            member.cohort_index,
            member.qualification_binding_sha256,
        )
        for position, member in enumerate(selection.members)
    )
    selection_binding = _waymax_metrics._selection_binding_sha256(selection)
    manifest_sha256 = _digest("verified-manifest")
    with pytest.raises(TypeError, match="verifier-issued"):
        M6WaymaxVerifiedStoredSelection(
            manifest_sha256=manifest_sha256,
            selection_binding_sha256=selection_binding,
            primary_domain_sha256=selection.primary_domain_sha256,
            supported=True,
            members=members,
        )
    stored_selection = verify_m6_waymax_stored_selection(
        parsed,
        manifest_sha256=manifest_sha256,
        selection_binding_sha256=selection_binding,
        primary_domain_sha256=selection.primary_domain_sha256,
        supported=True,
        members=members,
    )
    assert stored_selection.promotable is False
    assert stored_selection.members == members
    stored = reconstruct_m6_waymax_stored_cells(
        parsed,
        stored_selection=stored_selection,
        intervention_configuration_fingerprint=_b2_fingerprint(),
    )
    assert stored.promotable is False
    assert stored.pair_n == 8

    with pytest.raises(TypeError, match="verifier-issued"):
        dataclasses.replace(stored_selection)
    with pytest.raises(TypeError, match="M6WaymaxSelection"):
        analyze_m6_waymax_cells(
            issued,
            selection=stored_selection,  # type: ignore[arg-type]
            intervention_configuration_fingerprint=_b2_fingerprint(),
        )
    with pytest.raises(TypeError, match="M6WaymaxSelection"):
        compact_selected_m6_waymax_rollout(
            None,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            bundle=M6_WAYMAX_LOGGED_WORLD,
            selection=stored_selection,  # type: ignore[arg-type]
            primary_domain=None,  # type: ignore[arg-type]
            selection_position=0,
        )

    with pytest.raises(
        M6WaymaxMeasureError,
        match="stored_selection_cross_binding",
    ):
        verify_m6_waymax_stored_selection(
            parsed,
            manifest_sha256=manifest_sha256,
            selection_binding_sha256=selection_binding,
            primary_domain_sha256=selection.primary_domain_sha256,
            supported=True,
            members=tuple(reversed(members)),
        )
    self_rehashed = copy.deepcopy(stored_selection)
    object.__setattr__(
        self_rehashed,
        "manifest_sha256",
        _digest("other-manifest"),
    )
    object.__setattr__(
        self_rehashed,
        "receipt_sha256",
        self_rehashed._binding_sha256(),
    )
    with pytest.raises(
        M6WaymaxMeasureError,
        match="stored_selection_receipt_mutated",
    ):
        reconstruct_m6_waymax_stored_cells(
            parsed,
            stored_selection=self_rehashed,
            intervention_configuration_fingerprint=_b2_fingerprint(),
        )


@pytest.mark.parametrize("attack", ["global_selection", "position_swap"])
def test_self_rehashed_global_or_position_rows_fail_selection_verification(
    attack: str,
) -> None:
    selection = _invented_selection(8)
    issued = build_m6_waymax_scene_scalar_table(
        _invented_views(selection),
        selection=selection,
    )
    rows = []
    for row in issued.rows:
        if attack == "global_selection":
            rows.append(
                dataclasses.replace(
                    row,
                    selection_binding_sha256=_digest(
                        "forged-global-selection"
                    ),
                    scalar_binding_sha256=None,
                )
            )
        elif row.selection_position in (0, 1):
            rows.append(
                dataclasses.replace(
                    row,
                    selection_position=1 - row.selection_position,
                    scalar_binding_sha256=None,
                )
            )
        else:
            rows.append(row)
    parsed = parse_m6_waymax_scene_scalar_table(rows)
    with pytest.raises(
        M6WaymaxMeasureError,
        match="selection_cross_binding",
    ):
        reconstruct_m6_waymax_stored_cells(
            parsed,
            selection=selection,
            verified_selection_binding_sha256=(
                _waymax_metrics._selection_binding_sha256(selection)
            ),
            intervention_configuration_fingerprint=_b2_fingerprint(),
        )


@pytest.mark.parametrize(
    "drift",
    ["timestamps", "dimensions", "ego_realization"],
)
def test_cross_bundle_execution_identity_attacks_fail_closed(
    drift: str,
) -> None:
    selection = _invented_selection(8)
    views = list(_invented_views(selection))
    right = views[1]
    changes: dict[str, object]
    if drift == "timestamps":
        changes = {
            "timestamps_micros": (
                np.asarray(right.timestamps_micros) + 100_000
            ),
        }
    elif drift == "dimensions":
        changes = {"target_length_m": right.target_length_m + 0.1}
    else:
        treatment_ego_x = np.asarray(right.treatment_ego_x).copy()
        treatment_ego_x[-1] += 0.1
        changes = {"treatment_ego_x": treatment_ego_x}
    views[1] = _replace_issued_view(
        right,
        **changes,
        view_binding_sha256=None,
    )
    with pytest.raises(
        M6WaymaxMeasureError,
        match="cross_bundle_(pairing|realization)",
    ):
        build_m6_waymax_scene_scalar_table(
            views,
            selection=selection,
        )


def test_fewer_than_eight_cannot_retain_any_outcome_scalar() -> None:
    supported = _invented_selection(8)
    table = list(
        build_m6_waymax_scene_scalar_table(
            _invented_views(supported),
            selection=supported,
        )
    )
    # Self-consistently remove one complete position. The global frozen selection
    # still has eight members, so this cannot be reinterpreted as an N=7 outcome.
    for index, row in enumerate(table):
        if row.selection_position == 7:
            table[index] = dataclasses.replace(
                row,
                cohort_index=None,
                qualification_binding_sha256=None,
                value=None,
                responded=None,
                responder_latency_s=None,
                source_pairing_complete=False,
                status="not_selected",
                scalar_binding_sha256=None,
            )
    with pytest.raises(M6WaymaxMeasureError, match="selection_completeness"):
        parse_m6_waymax_scene_scalar_table(table)


def test_matrix_reconstructs_statistics_instead_of_trusting_cells() -> None:
    selection = _invented_selection(10)
    result = analyze_m6_waymax_matrix(
        _invented_views(selection),
        selection=selection,
    )
    cells = list(result.cells)
    source = cells[0]
    cells[0] = dataclasses.replace(
        source,
        arithmetic_mean=123456.0,
        cell_binding_sha256=None,
        _issuance_capability=_waymax_metrics._CELL_RESULT_ISSUER,
    )
    with pytest.raises(ValueError, match="reconstructed from 128 scalars"):
        M6WaymaxMatrixResult(
            pair_n=result.pair_n,
            cohort_indices=result.cohort_indices,
            intervention_configuration_fingerprint=(
                result.intervention_configuration_fingerprint
            ),
            scene_scalars=result.scene_scalars,
            cells=tuple(cells),
            _issuance_capability=_waymax_metrics._MATRIX_RESULT_ISSUER,
        )
    with pytest.raises(ValueError, match="exact registered eight"):
        M6WaymaxMatrixResult(
            pair_n=result.pair_n,
            cohort_indices=result.cohort_indices,
            intervention_configuration_fingerprint=(
                result.intervention_configuration_fingerprint
            ),
            scene_scalars=result.scene_scalars,
            cells=tuple(reversed(result.cells)),
            _issuance_capability=_waymax_metrics._MATRIX_RESULT_ISSUER,
        )
    with pytest.raises(ValueError, match="primary b=2 fingerprint"):
        M6WaymaxMatrixResult(
            pair_n=result.pair_n,
            cohort_indices=result.cohort_indices,
            intervention_configuration_fingerprint=_digest(
                "forged-matrix-config"
            ),
            scene_scalars=result.scene_scalars,
            cells=result.cells,
            _issuance_capability=_waymax_metrics._MATRIX_RESULT_ISSUER,
        )


def test_resampling_key_rejects_arbitrary_json_and_type_drift() -> None:
    for payload in (
        {},
        {
            "base_seed": True,
            "bundle": M6_WAYMAX_PRIVILEGED_IDM,
            "intervention_configuration_fingerprint": _b2_fingerprint(),
            "metric_name": "target_progress_loss_m",
            "metric_version": "1.0.0",
            "paired_n": 10,
            "policy_access_role": (
                "privileged_logged_trajectory_waypoint_following"
            ),
            "policy_name": M6_WAYMAX_PRIVILEGED_IDM,
            "resamples": 10_000,
            "statistics_schema_version": (
                "m6-waymax-paired-statistics-1.0.0"
            ),
        },
    ):
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).digest()
        with pytest.raises(ValueError, match="schema|constants|types"):
            M6WaymaxResamplingKey(
                canonical_json=canonical,
                sha256=digest.hex(),
                digest_words=struct.unpack(">8I", digest),
                pair_n=10,
            )

    selection = _invented_selection(10)
    matrix = analyze_m6_waymax_matrix(
        _invented_views(selection),
        selection=selection,
    )
    key = next(
        cell.resampling_key
        for cell in matrix.cells
        if cell.resampling_key is not None
    )
    mutated_words = list(key.digest_words)
    mutated_words[0] ^= 1
    object.__setattr__(key, "digest_words", tuple(mutated_words))
    with pytest.raises(M6WaymaxMeasureError, match="digest changed"):
        key.revalidate()
    with pytest.raises(M6WaymaxMeasureError, match="resampling_key_mutated"):
        matrix.revalidate()


def test_public_builder_consumes_adapter_validated_compact_pairs() -> None:
    state, scenario, primary = _adapter_source()
    (
        domain,
        selection,
        qualification,
        selection_position,
    ) = _adapter_domain_selection(
        state,
        scenario,
        primary,
    )
    assert qualification.eligible
    baseline_plan = compile_identity_plan(scenario)
    treatment_plan = compile_longitudinal_brake_pulse_plan(
        scenario,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    baseline_plan_view = build_waymax_ego_plan_view(
        state,
        scenario,
        baseline_plan,
    )
    treatment_plan_view = build_waymax_ego_plan_view(
        state,
        scenario,
        treatment_plan,
    )

    for bundle in M6_WAYMAX_BUNDLES:
        baseline = _adapter_compact(
            state,
            baseline_plan_view,
            bundle=bundle,
        )
        treatment = _adapter_compact(
            state,
            treatment_plan_view,
            bundle=bundle,
            responsive_target=(
                bundle == M6_WAYMAX_PRIVILEGED_IDM
            ),
        )
        view = build_m6_waymax_twenty_transition_pair_view(
            baseline,
            treatment,
            selection_position=selection_position,
            state=state,
            scenario=scenario,
            baseline_plan=baseline_plan,
            treatment_plan=treatment_plan,
            baseline_view=baseline_plan_view,
            treatment_view=treatment_plan_view,
            bundle=bundle,
            qualification=qualification,
            primary_domain=domain,
            selection=selection,
        )
        assert view.selection_position == selection_position
        assert view.cohort_index == 4
        assert view.target_agent_id == 102
        if bundle == M6_WAYMAX_LOGGED_WORLD:
            wrong_position = (selection_position + 1) % len(selection.members)
            with pytest.raises(M6WaymaxMeasureError, match="selection_member"):
                build_m6_waymax_twenty_transition_pair_view(
                    baseline,
                    treatment,
                    selection_position=wrong_position,
                    state=state,
                    scenario=scenario,
                    baseline_plan=baseline_plan,
                    treatment_plan=treatment_plan,
                    baseline_view=baseline_plan_view,
                    treatment_view=treatment_plan_view,
                    bundle=bundle,
                    qualification=qualification,
                    primary_domain=domain,
                    selection=selection,
                )
            drifted_selection = _invented_selection(8)
            with pytest.raises(ValueError, match="selection_mutated"):
                build_m6_waymax_twenty_transition_pair_view(
                    baseline,
                    treatment,
                    selection_position=selection_position,
                    state=state,
                    scenario=scenario,
                    baseline_plan=baseline_plan,
                    treatment_plan=treatment_plan,
                    baseline_view=baseline_plan_view,
                    treatment_view=treatment_plan_view,
                    bundle=bundle,
                    qualification=qualification,
                    primary_domain=domain,
                    selection=drifted_selection,
                )
            with pytest.raises(TypeError, match="evaluator-issued"):
                dataclasses.replace(
                    selection.qualification_ledger.rows[0],
                    eligible=False,
                    reason="source_cadence_not_100ms",
                    qualification_binding_sha256=None,
                )
        measures = {
            row.metric_name: row.value
            for row in compute_m6_waymax_paired_measures(view)
        }
        if bundle == M6_WAYMAX_LOGGED_WORLD:
            assert measures["additional_target_braking_impulse_mps"] == 0.0
            assert measures["target_progress_loss_m"] == 0.0
        else:
            assert measures["additional_target_braking_impulse_mps"] > 0.0
            assert measures["target_progress_loss_m"] > 0.0


def test_public_builder_rejects_compact_nonfinite_and_pair_gate_attacks() -> None:
    state, scenario, primary = _adapter_source()
    (
        domain,
        selection,
        qualification,
        selection_position,
    ) = _adapter_domain_selection(
        state,
        scenario,
        primary,
    )
    baseline_plan = compile_identity_plan(scenario)
    treatment_plan = compile_longitudinal_brake_pulse_plan(
        scenario,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    baseline_plan_view = build_waymax_ego_plan_view(
        state,
        scenario,
        baseline_plan,
    )
    treatment_plan_view = build_waymax_ego_plan_view(
        state,
        scenario,
        treatment_plan,
    )

    baseline = _adapter_compact(
        state,
        baseline_plan_view,
        bundle=M6_WAYMAX_PRIVILEGED_IDM,
    )
    treatment = _adapter_compact(
        state,
        treatment_plan_view,
        bundle=M6_WAYMAX_PRIVILEGED_IDM,
        responsive_target=True,
    )
    nonfinite_x = np.asarray(treatment.x).copy()
    nonfinite_x[19, 127] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        build_m6_waymax_twenty_transition_pair_view(
            baseline,
            treatment._replace(x=nonfinite_x),
            selection_position=selection_position,
            state=state,
            scenario=scenario,
            baseline_plan=baseline_plan,
            treatment_plan=treatment_plan,
            baseline_view=baseline_plan_view,
            treatment_view=treatment_plan_view,
            bundle=M6_WAYMAX_PRIVILEGED_IDM,
            qualification=qualification,
            primary_domain=domain,
            selection=selection,
        )

    early_vx = np.asarray(treatment.vx).copy()
    early_vx[0, 1] -= 0.1
    with pytest.raises(ValueError, match=r"t_plus_2"):
        build_m6_waymax_twenty_transition_pair_view(
            baseline,
            treatment._replace(vx=early_vx),
            selection_position=selection_position,
            state=state,
            scenario=scenario,
            baseline_plan=baseline_plan,
            treatment_plan=treatment_plan,
            baseline_view=baseline_plan_view,
            treatment_view=treatment_plan_view,
            bundle=M6_WAYMAX_PRIVILEGED_IDM,
            qualification=qualification,
            primary_domain=domain,
            selection=selection,
        )

    logged_baseline = _adapter_compact(
        state,
        baseline_plan_view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
    )
    logged_treatment = _adapter_compact(
        state,
        treatment_plan_view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
    )
    response_x = np.asarray(logged_treatment.x).copy()
    response_x[5, 1] += 0.1
    with pytest.raises(ValueError, match="logged_world_no_response"):
        build_m6_waymax_twenty_transition_pair_view(
            logged_baseline,
            logged_treatment._replace(x=response_x),
            selection_position=selection_position,
            state=state,
            scenario=scenario,
            baseline_plan=baseline_plan,
            treatment_plan=treatment_plan,
            baseline_view=baseline_plan_view,
            treatment_view=treatment_plan_view,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            qualification=qualification,
            primary_domain=domain,
            selection=selection,
        )
