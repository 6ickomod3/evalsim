"""Bounded M6 Waymax adapter tests; all fixtures are invented in memory."""
from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import importlib.util
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import evalsim.simulators.waymax_m6 as _waymax_m6
from evalsim.contracts import Agent, AgentType, InterventionEligibility, Scenario
from evalsim.perturb.m6 import (
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    compile_identity_plan,
    compile_longitudinal_brake_pulse_plan,
    evaluate_primary_brake_eligibility,
)
from evalsim.simulators.waymax_m6 import (
    M6_WAYMAX_LOGGED_WORLD,
    M6_WAYMAX_PRIVILEGED_IDM,
    M6_WAYMAX_RANK_DOMAIN,
    CompactM6WaymaxRollout,
    M6WaymaxEligibility,
    M6WaymaxError,
    M6WaymaxPrimaryDomain,
    M6WaymaxPrimaryDomainEntry,
    M6WaymaxQualificationLedger,
    M6WaymaxSelection,
    build_m6_waymax_primary_domain_entry,
    build_m6_waymax_qualification_ledger,
    build_waymax_ego_plan_view,
    compact_m6_waymax_rollout,
    compact_selected_m6_waymax_rollout,
    evaluate_m6_waymax_eligibility,
    m6_waymax_rank_sha256,
    m6_waymax_runtime_config,
    m6_waymax_to_rollout,
    select_m6_waymax_subset,
    single_scene_m6_idm_kernel,
    single_scene_m6_logged_world_kernel,
    source_state_mutation_sha256,
    tiny_m6_waymax_api_oracle,
    validate_m6_waymax_compact,
    validate_m6_waymax_pair,
)


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


def _invented_source() -> tuple[_FakeState, Scenario, InterventionEligibility]:
    objects = 128
    steps = 91
    frame = np.arange(steps, dtype=np.float32)
    shape = (objects, steps)
    valid = np.zeros(shape, dtype=bool)
    valid[:3] = True
    x = np.full(shape, -1000.0, dtype=np.float32)
    y = np.full(shape, -1000.0, dtype=np.float32)
    vx = np.full(shape, -1.0, dtype=np.float32)
    vy = np.full(shape, -1.0, dtype=np.float32)
    yaw = np.full(shape, -1.0, dtype=np.float32)
    x[0] = frame
    x[1] = frame - np.float32(10.0)
    x[2] = frame - np.float32(30.0)
    y[:2] = 0.0
    y[2] = 8.0
    vx[:3] = 10.0
    vy[:3] = 0.0
    yaw[:3] = 0.0
    length = np.full(shape, -1.0, dtype=np.float32)
    width = np.full(shape, -1.0, dtype=np.float32)
    height = np.full(shape, -1.0, dtype=np.float32)
    length[:3] = 4.0
    width[:3] = 2.0
    height[:3] = 1.5
    timestamps = np.broadcast_to(
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
        timestamp_micros=timestamps,
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
    scenario = Scenario(
        scenario_id="invented-m6-waymax",
        timestamps=(
            np.arange(steps, dtype=np.int64) * 100_000
        ).astype(np.float64)
        * 1e-6,
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
    eligibility = InterventionEligibility.accepted((10, 50), target_index=1)
    return state, scenario, eligibility


def _selection_fixture(
    indices,
    *,
    ineligible_indices=(),
) -> tuple[
    list[M6WaymaxEligibility],
    M6WaymaxPrimaryDomain,
    dict[int, Scenario],
]:
    template_state, template, _ = _invented_source()
    ineligible = set(ineligible_indices)
    rows = []
    entries = []
    scenarios = {}
    for index in indices:
        state = copy.deepcopy(template_state)
        scenario = copy.deepcopy(template)
        scenario.scenario_id = f"invented-{index}"
        if index in ineligible:
            valid_slots = np.flatnonzero(
                np.asarray(state.object_metadata.is_valid, dtype=bool)
            )
            state.log_trajectory.timestamp_micros[
                valid_slots,
                11,
            ] += 1
            canonical = np.asarray(
                state.log_trajectory.timestamp_micros[0],
                dtype=np.int64,
            )
            scenario.timestamps = (
                canonical - canonical[0]
            ).astype(np.float64) * 1e-6
        primary = evaluate_primary_brake_eligibility(scenario)
        entry = build_m6_waymax_primary_domain_entry(
            state,
            scenario,
            primary,
            cohort_index=index,
        )
        rows.append(
            evaluate_m6_waymax_eligibility(
                state,
                scenario,
                primary,
                cohort_index=index,
                primary_entry=entry,
            )
        )
        entries.append(entry)
        scenarios[index] = scenario
    return rows, M6WaymaxPrimaryDomain(tuple(entries)), scenarios


def _qualified(
    state,
    scenario: Scenario,
    primary: InterventionEligibility,
    *,
    cohort_index: int = 0,
) -> tuple[
    M6WaymaxEligibility,
    M6WaymaxPrimaryDomain,
    M6WaymaxPrimaryDomainEntry,
]:
    entry = build_m6_waymax_primary_domain_entry(
        state,
        scenario,
        primary,
        cohort_index=cohort_index,
    )
    qualification = evaluate_m6_waymax_eligibility(
        state,
        scenario,
        primary,
        cohort_index=cohort_index,
        primary_entry=entry,
    )
    return (
        qualification,
        M6WaymaxPrimaryDomain((entry,)),
        entry,
    )


def _mock_compact(
    state: _FakeState,
    view,
    *,
    bundle: str,
) -> CompactM6WaymaxRollout:
    current = 10
    interval = slice(current + 1, current + 21)
    trajectory = state.log_trajectory
    numeric = {
        "x": np.asarray(trajectory.x[:, interval]).T.copy(),
        "y": np.asarray(trajectory.y[:, interval]).T.copy(),
        "yaw": np.asarray(trajectory.yaw[:, interval]).T.copy(),
        "vx": np.asarray(trajectory.vel_x[:, interval]).T.copy(),
        "vy": np.asarray(trajectory.vel_y[:, interval]).T.copy(),
    }
    for name, source_name in (
        ("x", "x"),
        ("y", "y"),
        ("yaw", "heading"),
        ("vx", "vx"),
        ("vy", "vy"),
    ):
        numeric[name][:, 0] = np.asarray(getattr(view, source_name)[1:])
        numeric[name] = numeric[name].astype(np.float32)
    non_sdc_vehicle = (
        ~state.object_metadata.is_sdc
        & (state.object_metadata.object_types == 1)
    )
    if bundle == M6_WAYMAX_PRIVILEGED_IDM:
        requested = np.zeros((20, 128), dtype=bool)
        requested[:, non_sdc_vehicle] = True
    else:
        requested = np.zeros((20, 128), dtype=bool)
    effective = requested.copy()
    lifecycle = np.broadcast_to(non_sdc_vehicle, (20, 128)).copy() & ~requested
    overlap = np.zeros((20, 128), dtype=bool)
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
        initialized_overlap_excluded=overlap,
    )


def test_import_is_safe_without_optional_runtimes() -> None:
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname.split(".", 1)[0] in {"jax", "tensorflow", "waymax"}:
            raise AssertionError(f"optional import attempted: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptional())
import evalsim.simulators
import evalsim.simulators.waymax_m6
assert not ({"jax", "tensorflow", "waymax"} & set(sys.modules))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_rank_bytes_and_bounded_selection_are_exact() -> None:
    for index in (0, 1, 2**32 - 1):
        expected = hashlib.sha256(
            M6_WAYMAX_RANK_DOMAIN.encode("ascii")
            + b"\x00"
            + struct.pack(">I", index)
        ).hexdigest()
        assert m6_waymax_rank_sha256(index) == expected
    with pytest.raises(ValueError, match="uint32"):
        m6_waymax_rank_sha256(2**32)

    unsupported_rows, unsupported_domain, unsupported_scenarios = (
        _selection_fixture(range(7))
    )
    unsupported = select_m6_waymax_subset(
        unsupported_rows,
        primary_domain=unsupported_domain,
        primary_scenarios=unsupported_scenarios,
    )
    assert not unsupported.supported
    assert unsupported.eligible_count == 7
    assert unsupported.members == ()
    for count in (8, 15, 16, 20):
        rows, domain, scenarios = _selection_fixture(reversed(range(count)))
        selection = select_m6_waymax_subset(
            rows,
            primary_domain=domain,
            primary_scenarios=scenarios,
        )
        assert selection.supported
        assert len(selection.members) == min(count, 16)
        assert [member.cohort_index for member in selection.members] == [
            member.cohort_index
            for member in sorted(
                selection.members,
                key=lambda value: (
                    bytes.fromhex(value.rank_sha256),
                    value.cohort_index,
                ),
            )
        ]

    complete, complete_domain, complete_scenarios = _selection_fixture(range(8))
    with pytest.raises(M6WaymaxError, match="primary_domain_incomplete"):
        select_m6_waymax_subset(
            complete[:-1],
            primary_domain=complete_domain,
            primary_scenarios=complete_scenarios,
        )
    extended, _, extended_scenarios = _selection_fixture(range(9))
    with pytest.raises(M6WaymaxError, match="primary_domain_incomplete"):
        select_m6_waymax_subset(
            extended,
            primary_domain=complete_domain,
            primary_scenarios=extended_scenarios,
        )
    contradicted_entries = list(complete_domain.entries)
    contradicted_entries[0] = dataclasses.replace(
        contradicted_entries[0],
        source_state_sha256="b" * 64,
        entry_sha256=None,
    )
    contradicted_domain = M6WaymaxPrimaryDomain(
        tuple(contradicted_entries)
    )
    with pytest.raises(M6WaymaxError, match="primary_domain_source_mismatch"):
        select_m6_waymax_subset(
            complete,
            primary_domain=contradicted_domain,
            primary_scenarios=complete_scenarios,
        )


def test_complete_qualification_ledger_and_selection_are_factory_authentic() -> None:
    rows, domain, scenarios = _selection_fixture(range(20))
    ledger = build_m6_waymax_qualification_ledger(
        rows,
        primary_domain=domain,
        primary_scenarios=scenarios,
    )
    assert isinstance(ledger, M6WaymaxQualificationLedger)
    assert len(ledger.rows) == domain.member_count == 20
    assert tuple(row.cohort_index for row in ledger.rows) == tuple(range(20))
    ledger.revalidate()

    selection = select_m6_waymax_subset(
        ledger,
        primary_domain=domain,
    )
    assert selection.qualification_ledger_sha256 == ledger.ledger_sha256
    assert selection.qualification_ledger is ledger
    assert len(selection.members) == 16
    selection.revalidate(primary_domain=domain)

    with pytest.raises(TypeError, match="builder-issued"):
        M6WaymaxQualificationLedger(
            primary_domain_sha256=domain.domain_sha256,
            primary_domain_member_count=domain.member_count,
            rows=tuple(rows),
        )
    with pytest.raises(TypeError, match="selector-issued"):
        M6WaymaxSelection(
            supported=selection.supported,
            primary_domain_sha256=selection.primary_domain_sha256,
            primary_domain_member_count=selection.primary_domain_member_count,
            eligible_count=selection.eligible_count,
            members=selection.members,
            qualification_ledger=ledger,
            qualification_ledger_sha256=ledger.ledger_sha256,
        )

    eligible = sorted(
        (row for row in ledger.rows if row.eligible),
        key=lambda row: (bytes.fromhex(row.rank_sha256), row.cohort_index),
    )
    alternate_top16 = tuple((*eligible[:15], eligible[16]))
    object.__setattr__(selection, "members", alternate_top16)
    object.__setattr__(selection, "selection_sha256", "f" * 64)
    with pytest.raises(M6WaymaxError, match="selection_mutated"):
        selection.revalidate(primary_domain=domain)


def test_evaluator_issuance_blocks_eight_self_rehashed_rejections(
    monkeypatch,
) -> None:
    rows, domain, scenarios = _selection_fixture(
        range(8),
        ineligible_indices=set(range(8)),
    )
    assert all(
        not row.eligible and row.reason == "source_cadence_not_100ms"
        for row in rows
    )
    source_row = rows[0]
    with pytest.raises(TypeError, match="evaluator-issued"):
        M6WaymaxEligibility(
            cohort_index=source_row.cohort_index,
            eligible=source_row.eligible,
            reason=source_row.reason,
            scenario_id=source_row.scenario_id,
            target_index=source_row.target_index,
            target_agent_id=source_row.target_agent_id,
            target_slot=source_row.target_slot,
            rank_sha256=source_row.rank_sha256,
            source_binding_sha256=source_row.source_binding_sha256,
            primary_entry_sha256=source_row.primary_entry_sha256,
            qualification_binding_sha256=(
                source_row.qualification_binding_sha256
            ),
        )
    forged = []
    for row in rows:
        with pytest.raises(TypeError, match="evaluator-issued"):
            dataclasses.replace(
                row,
                eligible=True,
                reason=None,
                qualification_binding_sha256=None,
            )
        attack = copy.deepcopy(row)
        object.__setattr__(attack, "eligible", True)
        object.__setattr__(attack, "reason", None)
        binding = _waymax_m6._qualification_binding_sha256(
            scenario_id=attack.scenario_id,
            source_binding_sha256=attack.source_binding_sha256,
            primary_entry_sha256=attack.primary_entry_sha256,
            cohort_index=attack.cohort_index,
            eligible=True,
            reason=None,
            target_index=attack.target_index,
            target_agent_id=attack.target_agent_id,
            target_slot=attack.target_slot,
        )
        object.__setattr__(
            attack,
            "qualification_binding_sha256",
            binding,
        )
        forged.append(attack)
    with pytest.raises(M6WaymaxError, match="qualification_mutated"):
        build_m6_waymax_qualification_ledger(
            forged,
            primary_domain=domain,
            primary_scenarios=scenarios,
        )
    with pytest.raises(M6WaymaxError, match="qualification_mutated"):
        select_m6_waymax_subset(
            forged,
            primary_domain=domain,
            primary_scenarios=scenarios,
        )

    selection = select_m6_waymax_subset(
        rows,
        primary_domain=domain,
        primary_scenarios=scenarios,
    )
    assert not selection.supported
    state, scenario, _ = _invented_source()
    plan = compile_identity_plan(scenario)
    calls = 0

    def forbidden_kernel(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("guarded kernel must not be called")

    monkeypatch.setattr(
        "evalsim.simulators.waymax_m6.compact_m6_waymax_rollout",
        forbidden_kernel,
    )
    with pytest.raises(M6WaymaxError, match="selection_unsupported"):
        compact_selected_m6_waymax_rollout(
            state,
            scenario,
            plan,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            selection=selection,
            primary_domain=domain,
            selection_position=0,
        )
    assert calls == 0


def test_live_execution_guard_runs_zero_kernels_for_invalid_selection(
    monkeypatch,
) -> None:
    state, template, primary = _invented_source()
    entries = []
    rows = []
    scenarios = {}
    for index in range(8):
        scenario = copy.deepcopy(template)
        scenario.scenario_id = f"live-guard-{index}"
        entry = build_m6_waymax_primary_domain_entry(
            state,
            scenario,
            primary,
            cohort_index=index,
        )
        entries.append(entry)
        scenarios[index] = scenario
        rows.append(
            evaluate_m6_waymax_eligibility(
                state,
                scenario,
                primary,
                cohort_index=index,
                primary_entry=entry,
            )
        )
    domain = M6WaymaxPrimaryDomain(tuple(entries))
    selection = select_m6_waymax_subset(
        rows,
        primary_domain=domain,
        primary_scenarios=scenarios,
    )
    selected_position = 0
    selected_member = selection.members[selected_position]
    selected_scenario = scenarios[selected_member.cohort_index]
    plan = compile_identity_plan(selected_scenario)
    calls = 0

    def fake_compact(state_arg, scenario_arg, plan_arg, *, bundle):
        nonlocal calls
        del state_arg, scenario_arg, plan_arg, bundle
        calls += 1
        return "compact", "view"

    monkeypatch.setattr(
        "evalsim.simulators.waymax_m6.compact_m6_waymax_rollout",
        fake_compact,
    )
    assert compact_selected_m6_waymax_rollout(
        state,
        selected_scenario,
        plan,
        bundle=M6_WAYMAX_LOGGED_WORLD,
        selection=selection,
        primary_domain=domain,
        selection_position=selected_position,
    ) == ("compact", "view")
    assert calls == 1

    wrong_position = 1
    with pytest.raises(M6WaymaxError, match="selection_member"):
        compact_selected_m6_waymax_rollout(
            state,
            selected_scenario,
            plan,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            selection=selection,
            primary_domain=domain,
            selection_position=wrong_position,
        )
    assert calls == 1

    (
        unsupported_rows,
        unsupported_domain,
        unsupported_scenarios,
    ) = _selection_fixture(
        range(8),
        ineligible_indices={0},
    )
    unsupported = select_m6_waymax_subset(
        unsupported_rows,
        primary_domain=unsupported_domain,
        primary_scenarios=unsupported_scenarios,
    )
    assert not unsupported.supported
    with pytest.raises(M6WaymaxError, match="selection_unsupported"):
        compact_selected_m6_waymax_rollout(
            state,
            selected_scenario,
            plan,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            selection=unsupported,
            primary_domain=unsupported_domain,
            selection_position=0,
        )
    assert calls == 1


def test_runtime_config_is_recursively_immutable() -> None:
    config = m6_waymax_runtime_config()
    assert isinstance(config["bundles"], tuple)
    with pytest.raises(TypeError):
        config["adapter_version"] = "forged"  # type: ignore[index]
    with pytest.raises(TypeError):
        config["idm_defaults"]["desired_vel"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        config["bundles"][0] = "forged"  # type: ignore[index]


def test_source_only_eligibility_uses_cadence_control_and_frame_zero_overlap() -> None:
    state, scenario, primary = _invented_source()
    before = source_state_mutation_sha256(state)
    entry = build_m6_waymax_primary_domain_entry(
        state,
        scenario,
        primary,
        cohort_index=3,
    )
    result = evaluate_m6_waymax_eligibility(
        state,
        scenario,
        primary,
        cohort_index=3,
        primary_entry=entry,
    )
    assert result.eligible
    assert result.target_slot == 1
    assert result.target_agent_id == 102
    assert source_state_mutation_sha256(state) == before
    with pytest.raises(TypeError, match="primary_eligibility"):
        evaluate_m6_waymax_eligibility(
            state,
            scenario,
            object(),  # type: ignore[arg-type]
            cohort_index=3,
            primary_entry=entry,
        )

    cadence_state, cadence_scenario, _ = _invented_source()
    cadence_state.log_trajectory.timestamp_micros[:3, 17] += 1
    cadence_scenario.timestamps[17] += 1e-6
    cadence_entry = build_m6_waymax_primary_domain_entry(
        cadence_state,
        cadence_scenario,
        primary,
        cohort_index=3,
    )
    rejected = evaluate_m6_waymax_eligibility(
        cadence_state,
        cadence_scenario,
        primary,
        cohort_index=3,
        primary_entry=cadence_entry,
    )
    assert rejected.reason == "source_cadence_not_100ms"

    lifecycle_state, lifecycle_scenario, _ = _invented_source()
    lifecycle_entry = build_m6_waymax_primary_domain_entry(
        lifecycle_state,
        lifecycle_scenario,
        primary,
        cohort_index=3,
    )
    lifecycle_state.object_metadata.object_types[1] = 2
    lifecycle_scenario.agents[1].type = AgentType.PEDESTRIAN
    with pytest.raises(M6WaymaxError, match="primary_entry_mismatch"):
        evaluate_m6_waymax_eligibility(
            lifecycle_state,
            lifecycle_scenario,
            primary,
            cohort_index=3,
            primary_entry=lifecycle_entry,
        )

    overlap_state, overlap_scenario, _ = _invented_source()
    overlap_state.log_trajectory.x[1, 0] = 1.0
    overlap_state.log_trajectory.y[1, 0] = 0.0
    overlap_scenario.agents[1].x[0] = 1.0
    overlap_entry = build_m6_waymax_primary_domain_entry(
        overlap_state,
        overlap_scenario,
        primary,
        cohort_index=3,
    )
    rejected = evaluate_m6_waymax_eligibility(
        overlap_state,
        overlap_scenario,
        primary,
        cohort_index=3,
        primary_entry=overlap_entry,
    )
    assert rejected.reason == "target_initialized_overlap_excluded"


def test_plan_view_is_bound_float32_immutable_and_does_not_mutate_inputs() -> None:
    state, scenario, _ = _invented_source()
    plan = compile_longitudinal_brake_pulse_plan(
        scenario,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    source_before = source_state_mutation_sha256(state)
    plan_before = plan.serialize()
    view = build_waymax_ego_plan_view(state, scenario, plan)
    assert view.x.dtype == np.dtype("<f4")
    assert view.x.shape == (21,)
    assert view.future_action_data.shape == (20, 5)
    assert not view.x.flags.writeable
    assert not view.future_action_data.flags.writeable
    assert np.array_equal(view.x, plan.x[:21].astype("<f4"))
    assert source_state_mutation_sha256(state) == source_before
    assert plan.serialize() == plan_before
    with pytest.raises(ValueError):
        view.x[1] = 0.0

    tampered = np.asarray(view.x).copy()
    tampered[1] += 1.0
    object.__setattr__(view, "x", tampered)
    with pytest.raises(M6WaymaxError, match="plan_view_mutated"):
        view.revalidate()


def test_compact_validation_records_each_denominator_and_contradiction() -> None:
    state, scenario, primary = _invented_source()
    qualification, primary_domain, _ = _qualified(
        state,
        scenario,
        primary,
        cohort_index=0,
    )
    plan = compile_identity_plan(scenario)
    view = build_waymax_ego_plan_view(state, scenario, plan)
    compact = _mock_compact(state, view, bundle=M6_WAYMAX_LOGGED_WORLD)
    validation = validate_m6_waymax_compact(
        compact,
        state=state,
        scenario=scenario,
        plan=plan,
        view=view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    assert validation.passed
    assert validation.components["ego_plan.x"].denominator == 20
    assert validation.components["logged_fallback.x"].denominator == 40
    assert validation.components["actor.requested_control"].denominator == 2560
    forged_x = np.asarray(view.x).copy()
    forged_x[1] += 0.25
    self_hashed_field_forgery = dataclasses.replace(
        view,
        x=forged_x,
        local_mutation_sha256=None,
    )
    with pytest.raises(M6WaymaxError, match="plan_view_binding"):
        validate_m6_waymax_compact(
            compact,
            state=state,
            scenario=scenario,
            plan=plan,
            view=self_hashed_field_forgery,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            qualification=qualification,
            primary_domain=primary_domain,
        )
    self_hashed_audit_forgery = dataclasses.replace(
        view,
        canonical_plan_audit_fingerprint="a" * 64,
        local_mutation_sha256=None,
    )
    with pytest.raises(M6WaymaxError, match="plan_view_binding"):
        validate_m6_waymax_compact(
            compact,
            state=state,
            scenario=scenario,
            plan=plan,
            view=self_hashed_audit_forgery,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            qualification=qualification,
            primary_domain=primary_domain,
        )
    treatment_plan = compile_longitudinal_brake_pulse_plan(
        scenario,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    self_hashed_other_plan = build_waymax_ego_plan_view(
        state,
        scenario,
        treatment_plan,
    )
    with pytest.raises(M6WaymaxError, match="plan_view_binding"):
        validate_m6_waymax_compact(
            compact,
            state=state,
            scenario=scenario,
            plan=plan,
            view=self_hashed_other_plan,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            qualification=qualification,
            primary_domain=primary_domain,
        )
    object.__setattr__(qualification, "target_slot", 2)
    with pytest.raises(M6WaymaxError, match="qualification_mutated"):
        validate_m6_waymax_compact(
            compact,
            state=state,
            scenario=scenario,
            plan=plan,
            view=view,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            qualification=qualification,
            primary_domain=primary_domain,
        )
    object.__setattr__(qualification, "target_slot", 1)
    with pytest.raises(TypeError, match="evaluator-issued"):
        dataclasses.replace(
            qualification,
            cohort_index=1,
            rank_sha256=m6_waymax_rank_sha256(1),
            qualification_binding_sha256=None,
        )
    with pytest.raises(TypeError, match="evaluator-issued"):
        dataclasses.replace(
            qualification,
            scenario_id="different-invented-scene",
            qualification_binding_sha256=None,
        )

    x = np.asarray(compact.x).copy()
    x[4, 1] += 0.1
    contradicted = compact._replace(x=x)
    failed = validate_m6_waymax_compact(
        contradicted,
        state=state,
        scenario=scenario,
        plan=plan,
        view=view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    component = failed.components["logged_fallback.x"]
    assert component.tolerance_failure_count == 1
    assert component.maximum_absolute_error == pytest.approx(0.1, abs=1e-6)
    with pytest.raises(M6WaymaxError, match="logged_fallback.x"):
        failed.require_passed()

    requested = np.asarray(compact.requested_control).copy()
    requested[0, 1] = True
    binary_failed = validate_m6_waymax_compact(
        compact._replace(requested_control=requested),
        state=state,
        scenario=scenario,
        plan=plan,
        view=view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    assert (
        binary_failed.components[
            "actor.requested_control"
        ].binary_mismatch_count
        == 1
    )


def test_complete_primary_entry_rejects_self_rehashed_target_and_relabel_attacks(
) -> None:
    state, scenario, primary = _invented_source()
    qualification, primary_domain, entry = _qualified(
        state,
        scenario,
        primary,
    )
    plan = compile_identity_plan(scenario)
    view = build_waymax_ego_plan_view(state, scenario, plan)
    compact = _mock_compact(state, view, bundle=M6_WAYMAX_LOGGED_WORLD)

    substituted_primary = InterventionEligibility.accepted(
        primary.analysis_window,
        target_index=2,
    )
    substituted_entry = M6WaymaxPrimaryDomainEntry(
        cohort_index=0,
        scenario_id=scenario.scenario_id,
        source_state_sha256=entry.source_state_sha256,
        upstream_eligibility=substituted_primary,
        target_contract_id=103,
    )
    with pytest.raises(TypeError, match="evaluator-issued"):
        dataclasses.replace(
            qualification,
            target_index=2,
            target_agent_id=103,
            target_slot=2,
            primary_entry_sha256=substituted_entry.entry_sha256,
            qualification_binding_sha256=None,
        )
    substituted_domain = M6WaymaxPrimaryDomain((substituted_entry,))
    with pytest.raises(M6WaymaxError, match="primary_entry_mismatch"):
        select_m6_waymax_subset(
            (qualification,),
            primary_domain=substituted_domain,
            primary_scenarios={0: scenario},
        )
    with pytest.raises(M6WaymaxError, match="primary_entry_mismatch"):
        validate_m6_waymax_compact(
            compact,
            state=state,
            scenario=scenario,
            plan=plan,
            view=view,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            qualification=qualification,
            primary_domain=substituted_domain,
        )

    relabeled_scenario = copy.deepcopy(scenario)
    relabeled_scenario.scenario_id = "self-rehashed-relabeled-scenario"
    relabeled_primary = evaluate_primary_brake_eligibility(relabeled_scenario)
    relabeled_entry = build_m6_waymax_primary_domain_entry(
        state,
        relabeled_scenario,
        relabeled_primary,
        cohort_index=0,
    )
    relabeled_qualification = evaluate_m6_waymax_eligibility(
        state,
        relabeled_scenario,
        relabeled_primary,
        cohort_index=0,
        primary_entry=relabeled_entry,
    )
    relabeled_plan = compile_identity_plan(relabeled_scenario)
    relabeled_view = build_waymax_ego_plan_view(
        state,
        relabeled_scenario,
        relabeled_plan,
    )
    relabeled_compact = _mock_compact(
        state,
        relabeled_view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
    )
    with pytest.raises(M6WaymaxError, match="primary_entry_mismatch"):
        select_m6_waymax_subset(
            (relabeled_qualification,),
            primary_domain=primary_domain,
            primary_scenarios={0: relabeled_scenario},
        )
    with pytest.raises(M6WaymaxError, match="primary_entry_mismatch"):
        validate_m6_waymax_compact(
            relabeled_compact,
            state=state,
            scenario=relabeled_scenario,
            plan=relabeled_plan,
            view=relabeled_view,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            qualification=relabeled_qualification,
            primary_domain=primary_domain,
        )


@pytest.mark.parametrize(
    ("field_name", "slot", "value", "bundle"),
    [
        ("x", 0, np.nan, M6_WAYMAX_LOGGED_WORLD),
        ("y", 1, np.inf, M6_WAYMAX_PRIVILEGED_IDM),
        ("yaw", 1, -np.inf, M6_WAYMAX_LOGGED_WORLD),
        ("vx", 127, np.nan, M6_WAYMAX_LOGGED_WORLD),
        ("vy", 127, np.inf, M6_WAYMAX_PRIVILEGED_IDM),
    ],
)
def test_full_float_domain_rejects_nonfinite_before_any_mask(
    field_name,
    slot,
    value,
    bundle,
) -> None:
    state, scenario, primary = _invented_source()
    qualification, primary_domain, _ = _qualified(
        state,
        scenario,
        primary,
        cohort_index=0,
    )
    plan = compile_identity_plan(scenario)
    view = build_waymax_ego_plan_view(state, scenario, plan)
    compact = _mock_compact(state, view, bundle=bundle)
    values = np.asarray(getattr(compact, field_name)).copy()
    values[0, slot] = value
    attacked = compact._replace(**{field_name: values})
    with pytest.raises(M6WaymaxError, match="compact_nonfinite"):
        validate_m6_waymax_compact(
            attacked,
            state=state,
            scenario=scenario,
            plan=plan,
            view=view,
            bundle=bundle,
            qualification=qualification,
            primary_domain=primary_domain,
        )


def test_pair_rejects_nonfinite_on_either_or_both_sides() -> None:
    state, scenario, primary = _invented_source()
    qualification, primary_domain, _ = _qualified(
        state,
        scenario,
        primary,
        cohort_index=0,
    )
    baseline_plan = compile_identity_plan(scenario)
    treatment_plan = compile_longitudinal_brake_pulse_plan(
        scenario,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    baseline_view = build_waymax_ego_plan_view(state, scenario, baseline_plan)
    treatment_view = build_waymax_ego_plan_view(state, scenario, treatment_plan)
    baseline = _mock_compact(
        state,
        baseline_view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
    )
    treatment = _mock_compact(
        state,
        treatment_view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
    )
    baseline_x = np.asarray(baseline.x).copy()
    baseline_x[0, 127] = np.nan
    treatment_y = np.asarray(treatment.y).copy()
    treatment_y[0, 1] = np.inf
    bad_baseline = baseline._replace(x=baseline_x)
    bad_treatment = treatment._replace(y=treatment_y)
    for left, right in (
        (bad_baseline, treatment),
        (baseline, bad_treatment),
        (bad_baseline, bad_treatment),
    ):
        with pytest.raises(M6WaymaxError, match="compact_nonfinite"):
            validate_m6_waymax_pair(
                left,
                right,
                state=state,
                scenario=scenario,
                baseline_plan=baseline_plan,
                treatment_plan=treatment_plan,
                baseline_view=baseline_view,
                treatment_view=treatment_view,
                bundle=M6_WAYMAX_LOGGED_WORLD,
                qualification=qualification,
                primary_domain=primary_domain,
            )


def test_pair_gates_logged_nonresponse_and_synchronous_t_plus_2_floor() -> None:
    state, scenario, primary = _invented_source()
    qualification, primary_domain, _ = _qualified(
        state,
        scenario,
        primary,
        cohort_index=0,
    )
    baseline_plan = compile_identity_plan(scenario)
    treatment_plan = compile_longitudinal_brake_pulse_plan(
        scenario,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    baseline_view = build_waymax_ego_plan_view(state, scenario, baseline_plan)
    treatment_view = build_waymax_ego_plan_view(state, scenario, treatment_plan)
    logged_baseline = _mock_compact(
        state,
        baseline_view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
    )
    logged_treatment = _mock_compact(
        state,
        treatment_view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
    )
    logged_gate = validate_m6_waymax_pair(
        logged_baseline,
        logged_treatment,
        state=state,
        scenario=scenario,
        baseline_plan=baseline_plan,
        treatment_plan=treatment_plan,
        baseline_view=baseline_view,
        treatment_view=treatment_view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    assert logged_gate.passed
    with pytest.raises(M6WaymaxError, match="pair_condition"):
        validate_m6_waymax_pair(
            logged_treatment,
            logged_baseline,
            state=state,
            scenario=scenario,
            baseline_plan=treatment_plan,
            treatment_plan=baseline_plan,
            baseline_view=treatment_view,
            treatment_view=baseline_view,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            qualification=qualification,
            primary_domain=primary_domain,
        )

    idm_baseline = _mock_compact(
        state,
        baseline_view,
        bundle=M6_WAYMAX_PRIVILEGED_IDM,
    )
    idm_treatment = _mock_compact(
        state,
        treatment_view,
        bundle=M6_WAYMAX_PRIVILEGED_IDM,
    )
    response = np.asarray(idm_treatment.x).copy()
    response[1:, 1] -= 0.25
    idm_treatment = idm_treatment._replace(x=response)
    floor_gate = validate_m6_waymax_pair(
        idm_baseline,
        idm_treatment,
        state=state,
        scenario=scenario,
        baseline_plan=baseline_plan,
        treatment_plan=treatment_plan,
        baseline_view=baseline_view,
        treatment_view=treatment_view,
        bundle=M6_WAYMAX_PRIVILEGED_IDM,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    assert floor_gate.passed
    too_early = np.asarray(idm_treatment.x).copy()
    too_early[0, 1] -= 0.25
    rejected = validate_m6_waymax_pair(
        idm_baseline,
        idm_treatment._replace(x=too_early),
        state=state,
        scenario=scenario,
        baseline_plan=baseline_plan,
        treatment_plan=treatment_plan,
        baseline_view=baseline_view,
        treatment_view=treatment_view,
        bundle=M6_WAYMAX_PRIVILEGED_IDM,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    assert (
        rejected.components[
            "synchronous_t_plus_2_floor.x"
        ].tolerance_failure_count
        == 1
    )


def test_source_neutral_rollout_reconstruction_preserves_identity_and_masks() -> None:
    state, scenario, primary = _invented_source()
    qualification, primary_domain, _ = _qualified(
        state,
        scenario,
        primary,
        cohort_index=0,
    )
    plan = compile_identity_plan(scenario)
    view = build_waymax_ego_plan_view(state, scenario, plan)
    compact = _mock_compact(state, view, bundle=M6_WAYMAX_LOGGED_WORLD)
    rollout, validation = m6_waymax_to_rollout(
        compact,
        state=state,
        scenario=scenario,
        plan=plan,
        view=view,
        bundle=M6_WAYMAX_LOGGED_WORLD,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    assert validation.passed
    assert rollout.num_steps == 31
    assert rollout.scenario_id == scenario.scenario_id
    assert rollout.perturbation == plan.perturbation_identity
    assert [agent.id for agent in rollout.agents] == [101, 102, 103]
    np.testing.assert_array_equal(rollout.agents[1].valid, True)
    assert rollout.metadata["control_accounting"]["requested_control"] == 0
    assert rollout.metadata["control_accounting"]["lifecycle_fallback"] == 40
    with pytest.raises(ValueError, match="seed must equal 0"):
        m6_waymax_to_rollout(
            compact,
            state=state,
            scenario=scenario,
            plan=plan,
            view=view,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            qualification=qualification,
            primary_domain=primary_domain,
            seed=1,
        )
    invalid_padding = np.asarray(compact.x).copy()
    invalid_padding[0, 127] = -np.inf
    with pytest.raises(M6WaymaxError, match="rollout_nonfinite"):
        m6_waymax_to_rollout(
            compact._replace(x=invalid_padding),
            state=state,
            scenario=scenario,
            plan=plan,
            view=view,
            bundle=M6_WAYMAX_LOGGED_WORLD,
            qualification=qualification,
            primary_domain=primary_domain,
        )


def _native_state_and_scenario():
    jnp = pytest.importorskip("jax.numpy")
    datatypes = pytest.importorskip("waymax.datatypes")

    fake, scenario, eligibility = _invented_source()
    trajectory = datatypes.Trajectory(
        x=jnp.asarray(fake.log_trajectory.x),
        y=jnp.asarray(fake.log_trajectory.y),
        z=jnp.asarray(fake.log_trajectory.z),
        vel_x=jnp.asarray(fake.log_trajectory.vel_x),
        vel_y=jnp.asarray(fake.log_trajectory.vel_y),
        yaw=jnp.asarray(fake.log_trajectory.yaw),
        valid=jnp.asarray(fake.log_trajectory.valid),
        timestamp_micros=jnp.asarray(fake.log_trajectory.timestamp_micros),
        length=jnp.asarray(fake.log_trajectory.length),
        width=jnp.asarray(fake.log_trajectory.width),
        height=jnp.asarray(fake.log_trajectory.height),
    )
    metadata = datatypes.ObjectMetadata(
        ids=jnp.asarray(fake.object_metadata.ids),
        object_types=jnp.asarray(fake.object_metadata.object_types),
        is_sdc=jnp.asarray(fake.object_metadata.is_sdc),
        is_modeled=jnp.asarray(fake.object_metadata.is_sdc),
        is_valid=jnp.asarray(fake.object_metadata.is_valid),
        objects_of_interest=jnp.zeros((128,), dtype=jnp.bool_),
        is_controlled=jnp.asarray(fake.object_metadata.is_sdc),
    )
    traffic_lights = datatypes.TrafficLights(
        x=jnp.zeros((1, 91), dtype=jnp.float32),
        y=jnp.zeros((1, 91), dtype=jnp.float32),
        z=jnp.zeros((1, 91), dtype=jnp.float32),
        state=jnp.zeros((1, 91), dtype=jnp.int32),
        lane_ids=jnp.zeros((1, 91), dtype=jnp.int32),
        valid=jnp.zeros((1, 91), dtype=jnp.bool_),
    )
    state = datatypes.SimulatorState(
        sim_trajectory=trajectory,
        log_trajectory=trajectory,
        log_traffic_light=traffic_lights,
        object_metadata=metadata,
        timestep=jnp.asarray(0, dtype=jnp.int32),
    )
    state.validate()
    return state, scenario, eligibility


def _block(tree):
    import jax

    return jax.tree.map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


@pytest.mark.parametrize(
    ("bundle", "kernel"),
    [
        (M6_WAYMAX_LOGGED_WORLD, single_scene_m6_logged_world_kernel),
        (M6_WAYMAX_PRIVILEGED_IDM, single_scene_m6_idm_kernel),
    ],
)
def test_optional_runtime_eager_repeat_jit_api_oracle_and_pair_gates(
    bundle,
    kernel,
) -> None:
    if importlib.util.find_spec("waymax") is None:
        pytest.skip("optional Waymax runtime is unavailable")
    import jax
    import jax.numpy as jnp

    state, scenario, eligibility = _native_state_and_scenario()
    qualified, primary_domain, _ = _qualified(
        state,
        scenario,
        eligibility,
        cohort_index=0,
    )
    assert qualified.eligible
    baseline_plan = compile_identity_plan(scenario)
    treatment_plan = compile_longitudinal_brake_pulse_plan(
        scenario,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    first, baseline_view = compact_m6_waymax_rollout(
        state,
        scenario,
        baseline_plan,
        bundle=bundle,
    )
    first = _block(first)
    second, _ = compact_m6_waymax_rollout(
        state,
        scenario,
        baseline_plan,
        bundle=bundle,
    )
    second = _block(second)
    for name in CompactM6WaymaxRollout._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(first, name)),
            np.asarray(getattr(second, name)),
        )

    compiled = jax.jit(kernel)
    jit_result = _block(
        compiled(state, jnp.asarray(baseline_view.future_action_data))
    )
    for name in CompactM6WaymaxRollout._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(first, name)),
            np.asarray(getattr(jit_result, name)),
        )
    oracle = _block(
        tiny_m6_waymax_api_oracle(
            state,
            baseline_view,
            bundle=bundle,
            num_steps=2,
        )
    )
    for name in CompactM6WaymaxRollout._fields:
        actual = np.asarray(getattr(first, name))[:2]
        expected = np.asarray(getattr(oracle, name))
        if name in {"x", "y", "yaw", "vx", "vy"}:
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-6,
                atol=1e-5,
            )
        else:
            np.testing.assert_array_equal(actual, expected)

    treatment, treatment_view = compact_m6_waymax_rollout(
        state,
        scenario,
        treatment_plan,
        bundle=bundle,
    )
    treatment = _block(treatment)
    individual = validate_m6_waymax_compact(
        first,
        state=state,
        scenario=scenario,
        plan=baseline_plan,
        view=baseline_view,
        bundle=bundle,
        qualification=qualified,
        primary_domain=primary_domain,
    )
    assert individual.passed
    paired = validate_m6_waymax_pair(
        first,
        treatment,
        state=state,
        scenario=scenario,
        baseline_plan=baseline_plan,
        treatment_plan=treatment_plan,
        baseline_view=baseline_view,
        treatment_view=treatment_view,
        bundle=bundle,
        qualification=qualified,
        primary_domain=primary_domain,
    )
    assert paired.passed
    if bundle == M6_WAYMAX_LOGGED_WORLD:
        assert all(
            result.passed
            for name, result in paired.components.items()
            if name.startswith("logged_world_no_response.")
        )
    else:
        assert np.all(np.asarray(first.requested_control)[:, qualified.target_slot])
        assert np.all(np.asarray(first.effective_control)[:, qualified.target_slot])
        assert all(
            result.passed
            for name, result in paired.components.items()
            if name.startswith("synchronous_t_plus_2_floor.")
        )
