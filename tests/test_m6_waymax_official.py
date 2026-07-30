"""Invented-only tests for the runtime-injected official M6 Waymax seam."""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import evalsim.evaluation.m6_waymax_official as module
from evalsim.evaluation.m6 import evaluate_m6_source_eligibility
from evalsim.evaluation.m6_official import (
    M6OfficialAdapterError,
    M6OfficialCaseCollector,
    run_m6_official_numpy,
)
from evalsim.evaluation.m6_waymax_official import (
    M6_WAYMAX_OFFICIAL_COMPARISON_FIELDS,
    M6_WAYMAX_OFFICIAL_FIELD_ROW_COUNT,
    M6_WAYMAX_NUMPY_COMPARISON_ROW_COUNT,
    M6WaymaxExecutionAuthority,
    M6WaymaxNumpyComparisonTable,
    M6WaymaxPilotObservation,
    M6WaymaxSourceAuthority,
    build_m6_waymax_test_execution_authority,
    build_m6_waymax_test_source_authority,
    M6WaymaxOfficialCollector,
    M6WaymaxOfficialError,
    M6WaymaxOfficialExecutors,
    M6WaymaxOfficialFieldComparisonRow,
    run_m6_waymax_official,
    run_m6_waymax_outcome_suppressed_pilot,
)
from evalsim.simulators.waymax_m6 import (
    M6_WAYMAX_BUNDLES,
    M6_WAYMAX_PRIVILEGED_IDM,
    CompactM6WaymaxRollout,
    build_waymax_ego_plan_view,
    m6_waymax_rank_sha256,
)
from evalsim.sources.m5_m4_reuse import (
    AcceptedM4Cohort,
    ReloadedM4Member,
)
from evalsim.sources.waymax_loader import WaymaxRecord
from tests.test_m6_waymax import _invented_source


_HOSTILE_EXECUTOR_GLOBAL_TARGET = object()
_HOSTILE_EXECUTOR_HELPER_STATE = {"enabled": True}


def _hostile_executor_helper() -> bool:
    return _HOSTILE_EXECUTOR_HELPER_STATE["enabled"]


def _record(
    index: int,
    state: Any,
    *,
    scenario_id: str | None = None,
) -> WaymaxRecord:
    return WaymaxRecord(
        scenario_id=(
            f"private-native-{index}" if scenario_id is None else scenario_id
        ),
        state=state,
        audit={"private": np.asarray([index], dtype=np.int64)},
        shard_suffix=f"{index % 10:05d}",
        record_ordinal=index,
        shard_sha256=f"{index + 1:064x}",
        dataset_config_fingerprint=f"{index + 129:064x}",
    )


def _member(
    index: int,
    *,
    primary_eligible: bool,
    verified: bool = False,
    waymax_eligible: bool = True,
) -> ReloadedM4Member:
    state, scenario, _primary = _invented_source()
    offset = (index + 1) * 1_000
    state.object_metadata.ids[:3] += offset
    for agent in scenario.agents:
        agent.id += offset
    scenario.scenario_id = f"invented-official-{index}"
    record_scenario_id = scenario.scenario_id if verified else None
    if not primary_eligible:
        scenario.ego.vx[:] = 0.0
        scenario.ego.vy[:] = 0.0
        return ReloadedM4Member(
            cohort_index=index,
            scenario=scenario,
            record=_record(
                index,
                object(),
                scenario_id=record_scenario_id,
            ),
        )
    if not waymax_eligible:
        timestamp_micros = np.arange(91, dtype=np.int64) * 101_000
        state.log_trajectory.timestamp_micros[:] = timestamp_micros[
            np.newaxis, :
        ]
        scenario.timestamps[:] = timestamp_micros.astype(np.float64) * 1e-6
    return ReloadedM4Member(
        cohort_index=index,
        scenario=scenario,
        record=_record(index, state, scenario_id=record_scenario_id),
    )



def _source(
    eligible_n: int,
    *,
    full_numpy: bool,
    waymax_eligible_n: int | None = None,
):
    if waymax_eligible_n is None:
        waymax_eligible_n = eligible_n

    members = tuple(
        _member(
            index,
            primary_eligible=index < eligible_n,
            waymax_eligible=index < waymax_eligible_n,
        )
        for index in reversed(range(128))
    )
    collector = M6WaymaxOfficialCollector(
        build_m6_waymax_test_source_authority()
    )
    case_collector = M6OfficialCaseCollector()
    for member in members:
        case_collector(member)
        collector(member)
        assert collector.resident_candidate_count <= 16
    if full_numpy:
        numpy_evidence = run_m6_official_numpy(case_collector.cases)
        ledger = numpy_evidence.typed_result.eligibility_ledger
    else:
        numpy_evidence = None
        ledger = evaluate_m6_source_eligibility(case_collector.cases)
    return collector, collector.finalize(ledger), numpy_evidence


def _compact(state: Any, view: Any, *, bundle: str) -> CompactM6WaymaxRollout:
    current = view.current_index
    interval = slice(current + 1, current + 21)
    trajectory = state.log_trajectory
    numeric = {
        "x": np.asarray(trajectory.x[:, interval]).T.copy(),
        "y": np.asarray(trajectory.y[:, interval]).T.copy(),
        "yaw": np.asarray(trajectory.yaw[:, interval]).T.copy(),
        "vx": np.asarray(trajectory.vel_x[:, interval]).T.copy(),
        "vy": np.asarray(trajectory.vel_y[:, interval]).T.copy(),
    }
    for compact_name, view_name in (
        ("x", "x"),
        ("y", "y"),
        ("yaw", "heading"),
        ("vx", "vx"),
        ("vy", "vy"),
    ):
        numeric[compact_name][:, view.ego_slot] = np.asarray(
            getattr(view, view_name)[1:]
        )
        numeric[compact_name] = numeric[compact_name].astype(np.float32)
    non_sdc_vehicle = (
        ~np.asarray(state.object_metadata.is_sdc, dtype=bool)
        & (np.asarray(state.object_metadata.object_types) == 1)
    )
    requested = np.zeros((20, 128), dtype=bool)
    if bundle == M6_WAYMAX_PRIVILEGED_IDM:
        requested[:, non_sdc_vehicle] = True
    effective = requested.copy()
    lifecycle = (
        np.broadcast_to(non_sdc_vehicle, requested.shape).copy() & ~requested
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
        timestep=np.arange(current + 1, current + 21, dtype=np.int32),
        requested_control=requested,
        effective_control=effective,
        lifecycle_fallback=lifecycle,
        initialized_overlap_excluded=np.zeros((20, 128), dtype=bool),
    )


class _MockRuntime:
    def __init__(
        self,
        *,
        replay: bool = False,
        plan_alias: bool = False,
        cross_replay: bool = False,
    ) -> None:
        self.calls: list[tuple[str, int, str, str]] = []
        self.replay = replay
        self.plan_alias = plan_alias
        self.cross_replay = cross_replay
        self._cached: tuple[CompactM6WaymaxRollout, Any] | None = None

    def execute(
        self,
        label: str,
        state: Any,
        scenario: Any,
        plan: Any,
        *,
        bundle: str,
        selection: Any,
        primary_domain: Any,
        selection_position: int,
    ):
        del selection, primary_domain
        condition = (
            "identity" if plan.spec.family == "identity" else "primary_brake"
        )
        self.calls.append((label, selection_position, bundle, condition))
        if self.replay and self._cached is not None:
            return self._cached
        view = build_waymax_ego_plan_view(state, scenario, plan)
        compact = _compact(state, view, bundle=bundle)
        if self.plan_alias:
            compact = compact._replace(x=plan.x)
        if self.cross_replay and self._cached is not None:
            compact = compact._replace(x=self._cached[1].x)
        result = (compact, view)
        if (self.replay or self.cross_replay) and self._cached is None:
            self._cached = result
        return result

    def executors(self) -> M6WaymaxOfficialExecutors:
        def eager(*args, **kwargs):
            return self.execute("eager", *args, **kwargs)

        def jit_eager(*args, **kwargs):
            return self.execute("jit_eager", *args, **kwargs)

        def jit_compiled(*args, **kwargs):
            return self.execute("jit_compiled", *args, **kwargs)

        return M6WaymaxOfficialExecutors(
            eager=eager,
            jit_eager=jit_eager,
            jit_compiled=jit_compiled,
        )


def _invented_na_field_table() -> module.M6WaymaxOfficialFieldComparisonTable:
    rows = tuple(
        module.M6WaymaxOfficialFieldComparisonRow(
            selection_position=position,
            bundle=bundle,
            condition=condition,
            field_name=field_name,
            cohort_index=None,
            qualification_binding_sha256=None,
            comparison_kind=(
                "exact"
                if field_name in module.M6_WAYMAX_OFFICIAL_EXACT_FIELDS
                else "tolerance"
            ),
            denominator=None,
            max_abs_error=None,
            max_normalized_error=None,
            tolerance_failures=None,
            binary_mismatches=None,
            status="not_applicable",
            _issuance_capability=module._FIELD_ISSUER,
        )
        for position, bundle, condition, field_name in module._field_keys()
    )
    return module.M6WaymaxOfficialFieldComparisonTable(
        selection_supported=False,
        selected_member_count=0,
        selection_sha256="1" * 64,
        primary_domain_sha256="2" * 64,
        rows=rows,
        promotable=False,
        _issuance_capability=module._FIELD_ISSUER,
    )


def _invented_na_numpy_table() -> module.M6WaymaxNumpyComparisonTable:
    metric_by_name = {item[0]: item for item in module._NUMPY_METRICS}
    rows = []
    for position, policy_name, metric_name in module._numpy_comparison_keys():
        _name, metric_version, value_unit = metric_by_name[metric_name]
        rows.append(
            module.M6WaymaxNumpyComparisonRow(
                selection_position=position,
                cohort_index=None,
                qualification_binding_sha256=None,
                primary_domain_sha256="3" * 64,
                selection_binding_sha256="4" * 64,
                numpy_eligibility_ledger_sha256="5" * 64,
                stored_eligibility_rows_sha256="6" * 64,
                policy_name=policy_name,
                policy_access_role=module._NUMPY_POLICY_ACCESS[policy_name],
                metric_name=metric_name,
                metric_version=metric_version,
                value_unit=value_unit,
                value=None,
                responded=None,
                responder_latency_s=None,
                view_binding_sha256=None,
                source_pairing_complete=False,
                status="not_selected",
                _issuance_capability=module._NUMPY_ISSUER,
            )
        )
    return module.M6WaymaxNumpyComparisonTable(
        selected_member_count=0,
        primary_domain_sha256="3" * 64,
        selection_binding_sha256="4" * 64,
        numpy_eligibility_ledger_sha256="5" * 64,
        rows=tuple(rows),
        stored_eligibility_rows_sha256="6" * 64,
        promotable=False,
        _issuance_capability=module._NUMPY_ISSUER,
    )


def test_external_issuance_rejects_resealed_rows_and_tables() -> None:
    field_table = _invented_na_field_table()
    field_row = field_table.rows[0]
    original_bundle = field_row.bundle
    original_row_binding = field_row.row_binding_sha256
    original_row_original = field_row._issued_original_binding_sha256
    object.__setattr__(
        field_row,
        "bundle",
        next(bundle for bundle in M6_WAYMAX_BUNDLES if bundle != original_bundle),
    )
    resealed_row = field_row._binding()
    object.__setattr__(field_row, "row_binding_sha256", resealed_row)
    object.__setattr__(
        field_row,
        "_issued_original_binding_sha256",
        resealed_row,
    )
    try:
        with pytest.raises(M6WaymaxOfficialError, match="field_row_mutated"):
            field_row.revalidate()
    finally:
        object.__setattr__(field_row, "bundle", original_bundle)
        object.__setattr__(
            field_row,
            "row_binding_sha256",
            original_row_binding,
        )
        object.__setattr__(
            field_row,
            "_issued_original_binding_sha256",
            original_row_original,
        )
    field_row.revalidate()

    original_selection = field_table.selection_sha256
    original_table_binding = field_table.table_binding_sha256
    original_table_original = field_table._issued_original_binding_sha256
    object.__setattr__(field_table, "selection_sha256", "a" * 64)
    resealed_table = field_table._binding(field_table.rows)
    object.__setattr__(
        field_table,
        "table_binding_sha256",
        resealed_table,
    )
    object.__setattr__(
        field_table,
        "_issued_original_binding_sha256",
        resealed_table,
    )
    try:
        with pytest.raises(M6WaymaxOfficialError, match="field_table_mutated"):
            field_table.revalidate()
    finally:
        object.__setattr__(field_table, "selection_sha256", original_selection)
        object.__setattr__(
            field_table,
            "table_binding_sha256",
            original_table_binding,
        )
        object.__setattr__(
            field_table,
            "_issued_original_binding_sha256",
            original_table_original,
        )
    field_table.revalidate()

    numpy_table = _invented_na_numpy_table()
    numpy_row = numpy_table.rows[0]
    original_domain = numpy_row.primary_domain_sha256
    original_numpy_row_binding = numpy_row.row_binding_sha256
    original_numpy_row_original = numpy_row._issued_original_sha256
    object.__setattr__(numpy_row, "primary_domain_sha256", "b" * 64)
    resealed_numpy_row = numpy_row._binding()
    object.__setattr__(numpy_row, "row_binding_sha256", resealed_numpy_row)
    object.__setattr__(
        numpy_row,
        "_issued_original_sha256",
        resealed_numpy_row,
    )
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="numpy_comparison_row_mutated",
        ):
            numpy_row.revalidate()
    finally:
        object.__setattr__(numpy_row, "primary_domain_sha256", original_domain)
        object.__setattr__(
            numpy_row,
            "row_binding_sha256",
            original_numpy_row_binding,
        )
        object.__setattr__(
            numpy_row,
            "_issued_original_sha256",
            original_numpy_row_original,
        )
    numpy_row.revalidate()

    original_rows = numpy_table.rows
    original_numpy_table_binding = numpy_table.table_binding_sha256
    original_numpy_table_original = numpy_table._issued_original_sha256
    transplanted_rows = []
    for row in original_rows:
        values = row.to_store_dict()
        values["primary_domain_sha256"] = "c" * 64
        transplanted_rows.append(
            module.M6WaymaxNumpyComparisonRow(
                **values,
                _issuance_capability=module._NUMPY_ISSUER,
            )
        )
    object.__setattr__(numpy_table, "primary_domain_sha256", "c" * 64)
    object.__setattr__(numpy_table, "rows", tuple(transplanted_rows))
    resealed_numpy_table = numpy_table._binding()
    object.__setattr__(
        numpy_table,
        "table_binding_sha256",
        resealed_numpy_table,
    )
    object.__setattr__(
        numpy_table,
        "_issued_original_sha256",
        resealed_numpy_table,
    )
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="numpy_comparison_table_mutated",
        ):
            numpy_table.revalidate()
    finally:
        object.__setattr__(numpy_table, "primary_domain_sha256", "3" * 64)
        object.__setattr__(numpy_table, "rows", original_rows)
        object.__setattr__(
            numpy_table,
            "table_binding_sha256",
            original_numpy_table_binding,
        )
        object.__setattr__(
            numpy_table,
            "_issued_original_sha256",
            original_numpy_table_original,
        )
    numpy_table.revalidate()


def test_module_import_is_optional_runtime_safe() -> None:
    project = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'tensorflow', 'waymax'}:
            raise AssertionError(fullname)
        return None
sys.meta_path.insert(0, Block())
import evalsim.evaluation.m6_waymax_official as module
assert module.M6_WAYMAX_OFFICIAL_FIELD_ROW_COUNT == 640
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_direct_url_provenance_requires_every_exact_pin() -> None:
    canonical = {
        "url": module._WAYMAX_CANONICAL_GIT_URL,
        "vcs_info": {
            "vcs": "git",
            "requested_revision": module.WAYMAX_COMMIT,
            "commit_id": module.WAYMAX_COMMIT,
        },
    }
    assert module._validate_waymax_direct_url(canonical) == (
        module._WAYMAX_CANONICAL_GIT_URL,
        "git",
        module.WAYMAX_COMMIT,
        module.WAYMAX_COMMIT,
    )
    invalid = (
        {**canonical, "url": "https://example.invalid/waymax.git"},
        {**canonical, "vcs_info": {**canonical["vcs_info"], "vcs": "hg"}},
        {
            **canonical,
            "vcs_info": {
                **canonical["vcs_info"],
                "requested_revision": "main",
            },
        },
        {
            **canonical,
            "vcs_info": {**canonical["vcs_info"], "commit_id": "0" * 40},
        },
    )
    for candidate in invalid:
        with pytest.raises(M6WaymaxOfficialError, match="runtime_waymax_pin"):
            module._validate_waymax_direct_url(candidate)


def test_source_authority_external_issuance_rejects_forged_reseal() -> None:
    authority = build_m6_waymax_test_source_authority()
    authority.revalidate()
    with pytest.raises(TypeError, match="factory-issued"):
        dataclasses.replace(authority)

    transplant = object.__new__(M6WaymaxSourceAuthority)
    for name in (
        "kind",
        "promotable",
        "_member_bindings",
        "authority_sha256",
        "_issued_original_sha256",
    ):
        object.__setattr__(transplant, name, getattr(authority, name))
    with pytest.raises(
        M6WaymaxOfficialError,
        match="source_authority_mutated",
    ):
        transplant.revalidate()

    original_kind = authority.kind
    original_promotable = authority.promotable
    original_bindings = authority._member_bindings
    original_sha256 = authority.authority_sha256
    original_issued = authority._issued_original_sha256
    forged_bindings = tuple(
        (
            index,
            f"{index % 10:05d}",
            index,
            f"forged-{index}",
            f"{index + 1:064x}",
            f"{index + 129:064x}",
        )
        for index in range(128)
    )
    object.__setattr__(authority, "kind", "verified_accepted_m4")
    object.__setattr__(authority, "promotable", True)
    object.__setattr__(authority, "_member_bindings", forged_bindings)
    forged_sha256 = module._source_authority_sha256(
        authority.kind,
        authority.promotable,
        authority._member_bindings,
    )
    object.__setattr__(authority, "authority_sha256", forged_sha256)
    object.__setattr__(
        authority,
        "_issued_original_sha256",
        forged_sha256,
    )
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="source_authority_mutated",
        ):
            authority.revalidate()
        with pytest.raises(
            M6WaymaxOfficialError,
            match="source_authority_mutated",
        ):
            M6WaymaxOfficialCollector(authority)
    finally:
        object.__setattr__(authority, "kind", original_kind)
        object.__setattr__(authority, "promotable", original_promotable)
        object.__setattr__(authority, "_member_bindings", original_bindings)
        object.__setattr__(authority, "authority_sha256", original_sha256)
        object.__setattr__(
            authority,
            "_issued_original_sha256",
            original_issued,
        )
    authority.revalidate()


def test_authority_factory_closures_expose_no_generic_issuer() -> None:
    factories = (
        module.build_m6_waymax_test_source_authority,
        module.build_m6_waymax_verified_source_authority,
        module.build_m6_waymax_test_execution_authority,
        module.build_pinned_m6_waymax_execution_authority,
    )
    for factory in factories:
        values = tuple(
            cell.cell_contents for cell in (factory.__closure__ or ())
        )
        assert values
        assert all(not callable(value) for value in values)
        registries = tuple(
            value for value in values if isinstance(value, dict)
        )
        assert len(registries) == 1
        assert registries[0] == {}
        locks = tuple(
            value for value in values if type(value).__name__ == "RLock"
        )
        assert len(locks) == 1


def test_production_authority_labels_require_verifier_owned_receipts() -> None:
    absent_names = (
        "_SOURCE_AUTHORITY_ISSUER",
        "_EXECUTION_AUTHORITY_ISSUER",
        "_consume_source_authority_receipt",
        "_consume_execution_authority_receipt",
        "_initialize_source_authority_api",
        "_initialize_execution_authority_api",
        "_SOURCE_AUTHORITY_ISSUANCE_REGISTRY",
        "_EXECUTION_AUTHORITY_ISSUANCE_REGISTRY",
    )
    assert all(not hasattr(module, name) for name in absent_names)

    bindings = tuple(
        (
            index,
            f"{index % 10:05d}",
            index,
            f"forged-{index}",
            f"{index + 1:064x}",
            f"{index + 129:064x}",
        )
        for index in range(128)
    )
    with patch.object(
        module,
        "_consume_source_authority_receipt",
        lambda *_args: None,
        create=True,
    ):
        with pytest.raises(TypeError, match="factory-issued"):
            M6WaymaxSourceAuthority(
                kind="verified_accepted_m4",
                promotable=True,
                _member_bindings=bindings,
                _issuance_receipt=object(),
            )

    def inert(*_args, **_kwargs):
        return None

    executors = M6WaymaxOfficialExecutors(inert, inert, inert)
    with patch.object(
        module,
        "_consume_execution_authority_receipt",
        lambda *_args: None,
        create=True,
    ):
        with pytest.raises(TypeError, match="factory-issued"):
            M6WaymaxExecutionAuthority(
                kind="pinned_cpu_waymax_jax",
                promotable=True,
                _executors=executors,
                _runtime_facts=(("forged", "true"),),
                _issuance_receipt=object(),
            )

    references = tuple(
        SimpleNamespace(
            cohort_index=row[0],
            expectation=SimpleNamespace(
                locator=SimpleNamespace(
                    shard_suffix=row[1],
                    record_ordinal=row[2],
                ),
                expected_scenario_id=row[3],
                expected_shard_sha256=row[4],
                expected_dataset_config_fingerprint=row[5],
            ),
        )
        for row in bindings
    )
    cohort = object.__new__(AcceptedM4Cohort)
    object.__setattr__(cohort, "members", references)
    with patch.object(
        module,
        "reverify_accepted_m4_run",
        autospec=True,
    ) as substituted_verifier:
        with pytest.raises(AttributeError):
            module.build_m6_waymax_verified_source_authority(cohort)
    substituted_verifier.assert_not_called()

    forged_material = (executors, (("invented_test_runtime", "true"),))
    with (
        patch.dict(sys.modules, {"jax": None}),
        patch.object(
            module,
            "_verified_pinned_m6_waymax_execution_material",
            autospec=True,
            return_value=forged_material,
        ) as substituted_resolver,
    ):
        with pytest.raises(M6WaymaxOfficialError, match="runtime_unavailable"):
            module.build_pinned_m6_waymax_execution_authority()
    substituted_resolver.assert_not_called()

    source_type = M6WaymaxSourceAuthority
    execution_type = M6WaymaxExecutionAuthority
    with (
        patch.object(module, "M6WaymaxSourceAuthority", object),
        patch.object(module, "M6WaymaxExecutionAuthority", object),
    ):
        test_source = build_m6_waymax_test_source_authority()
        test_execution = build_m6_waymax_test_execution_authority(executors)
    assert type(test_source) is source_type
    assert type(test_execution) is execution_type
    with pytest.raises(TypeError, match="factory-issued"):
        dataclasses.replace(test_source)
    with pytest.raises(TypeError, match="factory-issued"):
        dataclasses.replace(test_execution)


def test_authority_factory_failure_cleans_one_use_receipts() -> None:
    def registry(factory):
        values = tuple(
            cell.cell_contents for cell in (factory.__closure__ or ())
        )
        registries = tuple(
            value for value in values if isinstance(value, dict)
        )
        assert len(registries) == 1
        return registries[0]

    source_registry = registry(build_m6_waymax_test_source_authority)
    with patch.object(
        M6WaymaxSourceAuthority,
        "__post_init__",
        side_effect=RuntimeError("injected source construction failure"),
    ):
        with pytest.raises(RuntimeError, match="injected source"):
            build_m6_waymax_test_source_authority()
    assert source_registry == {}

    def inert(*_args, **_kwargs):
        return None

    executors = M6WaymaxOfficialExecutors(inert, inert, inert)
    execution_registry = registry(build_m6_waymax_test_execution_authority)
    with patch.object(
        M6WaymaxExecutionAuthority,
        "__post_init__",
        side_effect=RuntimeError("injected execution construction failure"),
    ):
        with pytest.raises(RuntimeError, match="injected execution"):
            build_m6_waymax_test_execution_authority(executors)
    assert execution_registry == {}

    build_m6_waymax_test_source_authority().revalidate()
    build_m6_waymax_test_execution_authority(executors).revalidate()


def test_downstream_authority_checks_ignore_module_class_rebinding() -> None:
    _collector, source, _numpy = _source(10, full_numpy=False)

    class FakeSourceAuthority:
        kind = "verified_accepted_m4"
        promotable = True
        authority_sha256 = "0" * 64

        def revalidate(self):
            return None

        def verify_member(self, _member):
            return None

    class FakeExecutionAuthority:
        kind = "pinned_cpu_waymax_jax"
        promotable = True
        authority_sha256 = "f" * 64

        def revalidate(self):
            return None

        def _claim(self):
            return None

    fake_source = FakeSourceAuthority()
    fake_execution = FakeExecutionAuthority()
    with (
        patch.object(module, "M6WaymaxSourceAuthority", FakeSourceAuthority),
        patch.object(
            module,
            "M6WaymaxExecutionAuthority",
            FakeExecutionAuthority,
        ),
    ):
        source.revalidate()
        with pytest.raises(
            TypeError,
            match="exact M6WaymaxSourceAuthority",
        ):
            module.M6WaymaxOfficialCollector(fake_source)
        with pytest.raises(
            TypeError,
            match="exact M6WaymaxExecutionAuthority",
        ):
            module.run_m6_waymax_official(
                source,
                fake_execution,
                object(),
            )
        with pytest.raises(
            TypeError,
            match="exact M6WaymaxOfficialSource",
        ):
            module.run_m6_waymax_outcome_suppressed_pilot(
                object(),
                fake_execution,
            )


def test_collector_requires_complete_unique_domain() -> None:
    collector = M6WaymaxOfficialCollector(
        build_m6_waymax_test_source_authority()
    )
    member = _member(0, primary_eligible=False)
    collector(member)
    with pytest.raises(M6WaymaxOfficialError, match="duplicate"):
        collector(member)
    with pytest.raises(M6WaymaxOfficialError, match="incomplete"):
        collector.finalize(object())


def test_collector_retains_only_canonical_top_sixteen() -> None:
    collector, source, _numpy = _source(20, full_numpy=False)
    expected = tuple(
        sorted(
            range(20),
            key=lambda index: (
                bytes.fromhex(m6_waymax_rank_sha256(index)),
                index,
            ),
        )[:16]
    )
    assert source.resident_candidate_count == 16
    assert tuple(member.cohort_index for member in source.selection.members) == expected
    assert not hasattr(source, "states")
    assert not hasattr(source, "candidates")
    assert "invented-official" not in repr(source)
    assert collector.resident_candidate_count == 0
    source.revalidate()
    with pytest.raises(TypeError, match="collector-issued"):
        module.M6WaymaxOfficialSource(
            source_authority=source._source_authority,
            numpy_eligibility_ledger=source._numpy_eligibility_ledger,
            primary_domain=source._primary_domain,
            qualification_ledger=source._qualification_ledger,
            selection=source._selection,
            residents=source._residents,
            member_bindings=source._member_bindings,
            source_scenarios=source._source_scenarios,
        )
    original_authority = source._source_authority
    object.__setattr__(
        source,
        "_source_authority",
        build_m6_waymax_test_source_authority(),
    )
    try:
        with pytest.raises(M6WaymaxOfficialError, match="official_source_mutated"):
            source.revalidate()
    finally:
        object.__setattr__(source, "_source_authority", original_authority)
    source.revalidate()
    original_seal = source.source_binding_sha256
    object.__setattr__(source, "_source_binding_sha256", "0" * 64)
    try:
        with pytest.raises(M6WaymaxOfficialError, match="official_source_mutated"):
            source.revalidate()
    finally:
        object.__setattr__(source, "_source_binding_sha256", original_seal)
    source.revalidate()
    with pytest.raises(M6WaymaxOfficialError, match="finalized"):
        collector.finalize(source.numpy_eligibility_ledger)


@pytest.fixture(scope="module")
def supported_execution():
    _collector, source, numpy_evidence = _source(10, full_numpy=True)
    runtime = _MockRuntime()
    authority = build_m6_waymax_test_execution_authority(
        runtime.executors()
    )
    evidence = run_m6_waymax_official(
        source,
        authority,
        numpy_evidence,
    )
    return source, numpy_evidence, runtime, authority, evidence


def test_supported_execution_issues_complete_typed_evidence(
    supported_execution,
) -> None:
    source, numpy_evidence, runtime, _authority, evidence = supported_execution
    assert evidence.supported
    assert len(evidence.scene_scalars) == 128
    assert len(evidence.field_comparisons) == M6_WAYMAX_OFFICIAL_FIELD_ROW_COUNT
    assert len(evidence.determinism) == 64
    assert not evidence.promotable
    assert not evidence.field_comparisons.promotable
    assert not evidence.determinism.promotable
    assert isinstance(evidence.numpy_comparisons, M6WaymaxNumpyComparisonTable)
    assert len(evidence.numpy_comparisons) == M6_WAYMAX_NUMPY_COMPARISON_ROW_COUNT
    assert not evidence.numpy_comparisons.promotable
    assert source.resident_candidate_count == 10

    field_rows = evidence.field_comparisons.to_store_rows()
    assert tuple(
        row["field_name"] for row in field_rows[:10]
    ) == M6_WAYMAX_OFFICIAL_COMPARISON_FIELDS
    assert sum(row["status"] == "passed" for row in field_rows) == 400
    assert sum(row["status"] == "not_applicable" for row in field_rows) == 240
    timestamps = next(
        row
        for row in field_rows
        if row["selection_position"] == 0
        and row["bundle"] == M6_WAYMAX_BUNDLES[0]
        and row["condition"] == "identity"
        and row["field_name"] == "timestamps"
    )
    assert timestamps["denominator"] == 20 * 128 + 20
    assert all(
        set(row)
        == {
            "selection_position",
            "bundle",
            "condition",
            "field_name",
            "cohort_index",
            "qualification_binding_sha256",
            "comparison_kind",
            "denominator",
            "max_abs_error",
            "max_normalized_error",
            "tolerance_failures",
            "binary_mismatches",
            "status",
        }
        for row in field_rows
    )

    expected_calls = 10 * 2 * 2 * 2 + 2 * 2 * 2
    assert len(runtime.calls) == expected_calls
    assert runtime.calls[:4] == [
        ("eager", 0, M6_WAYMAX_BUNDLES[0], "identity"),
        ("eager", 0, M6_WAYMAX_BUNDLES[0], "identity"),
        ("jit_eager", 0, M6_WAYMAX_BUNDLES[0], "identity"),
        ("jit_compiled", 0, M6_WAYMAX_BUNDLES[0], "identity"),
    ]
    assert not any(
        label.startswith("jit") and position != 0
        for label, position, _bundle, _condition in runtime.calls
    )
    evidence.revalidate()


def test_field_evidence_is_factory_issued_and_tamper_evident(
    supported_execution,
) -> None:
    _source_value, _numpy, _runtime, _authority, evidence = supported_execution
    with pytest.raises(TypeError, match="factory-issued"):
        M6WaymaxOfficialFieldComparisonRow(
            selection_position=0,
            bundle=M6_WAYMAX_BUNDLES[0],
            condition="identity",
            field_name="x",
            cohort_index=0,
            qualification_binding_sha256="0" * 64,
            comparison_kind="tolerance",
            denominator=20,
            max_abs_error=0.0,
            max_normalized_error=0.0,
            tolerance_failures=0,
            binary_mismatches=0,
            status="passed",
        )
    victim = evidence.field_comparisons.rows[0]
    original = victim.status
    object.__setattr__(victim, "status", "failed")
    try:
        with pytest.raises(M6WaymaxOfficialError, match="field_row_mutated"):
            evidence.field_comparisons.revalidate()
    finally:
        object.__setattr__(victim, "status", original)
    evidence.field_comparisons.revalidate(
        selection=evidence.selection,
        primary_domain=evidence.primary_domain,
    )
    table = evidence.field_comparisons
    object.__setattr__(table, "promotable", True)
    try:
        with pytest.raises(M6WaymaxOfficialError, match="field_table_mutated"):
            table.revalidate()
    finally:
        object.__setattr__(table, "promotable", False)
    table.revalidate(
        selection=evidence.selection, primary_domain=evidence.primary_domain
    )


def test_unsupported_selection_executes_nothing_and_issues_strict_na() -> None:
    _collector, source, numpy_evidence = _source(
        10,
        full_numpy=True,
        waymax_eligible_n=7,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsupported selection called runtime")

    authority = build_m6_waymax_test_execution_authority(
        M6WaymaxOfficialExecutors(
            eager=forbidden,
            jit_eager=forbidden,
            jit_compiled=forbidden,
        )
    )
    evidence = run_m6_waymax_official(
        source,
        authority,
        numpy_evidence,
    )
    assert not evidence.supported
    assert not evidence.promotable
    assert source.resident_candidate_count == 0
    assert not evidence.field_comparisons.promotable
    assert not evidence.numpy_comparisons.promotable
    assert not evidence.determinism.promotable
    assert all(
        row["status"] == "not_applicable"
        and row["cohort_index"] is None
        and row["denominator"] is None
        for row in evidence.field_comparisons.to_store_rows()
    )
    assert all(
        row["status"] == "not_selected"
        and row["cohort_index"] is None
        and row["value"] is None
        for row in evidence.numpy_comparisons.to_store_rows()
    )
    assert all(
        scalar.status == "not_selected" for scalar in evidence.scene_scalars
    )


def test_runtime_replayed_storage_fails_closed(supported_execution) -> None:
    source, numpy_evidence, _prior_runtime, _prior_authority, _evidence = (
        supported_execution
    )
    runtime = _MockRuntime(replay=True)
    with pytest.raises(M6WaymaxOfficialError, match="(execution|view)_replay"):
        run_m6_waymax_official(
            source,
            build_m6_waymax_test_execution_authority(runtime.executors()),
            numpy_evidence,
        )
    cross_runtime = _MockRuntime(cross_replay=True)
    with pytest.raises(
        M6WaymaxOfficialError,
        match="official_cross_category_replay",
    ):
        run_m6_waymax_official(
            source,
            build_m6_waymax_test_execution_authority(
                cross_runtime.executors()
            ),
            numpy_evidence,
        )
    plan_runtime = _MockRuntime(plan_alias=True)
    with pytest.raises(
        M6WaymaxOfficialError,
        match="official_cross_category_alias",
    ):
        run_m6_waymax_official(
            source,
            build_m6_waymax_test_execution_authority(
                plan_runtime.executors()
            ),
            numpy_evidence,
        )


def test_execution_authority_seals_callable_behavior() -> None:
    def make_executor():
        marker = object()
        closure_state = {"enabled": True}
        closure_state["self"] = closure_state

        def inner():
            return marker

        def executor(*_args, enabled: bool = True, **_kwargs):
            if _HOSTILE_EXECUTOR_GLOBAL_TARGET is None:
                return None
            if not _hostile_executor_helper():
                return None
            if not closure_state["enabled"]:
                return None
            return inner() if enabled else None

        return executor, inner

    eager, nested = make_executor()
    jit_eager, _jit_eager_nested = make_executor()
    jit_compiled, _jit_compiled_nested = make_executor()
    executors = M6WaymaxOfficialExecutors(
        eager=eager,
        jit_eager=jit_eager,
        jit_compiled=jit_compiled,
    )
    authority = build_m6_waymax_test_execution_authority(executors)
    authority.revalidate()
    original_authority_sha256 = authority.authority_sha256
    original_issued_sha256 = authority._issued_original_sha256

    replacement, replacement_nested = make_executor()
    original_code = eager.__code__
    eager.__code__ = replacement.__code__.replace()
    forged_sha256 = module._execution_authority_sha256(
        authority.kind,
        authority.promotable,
        authority._runtime_facts,
        executors,
    )
    object.__setattr__(authority, "authority_sha256", forged_sha256)
    object.__setattr__(
        authority,
        "_issued_original_sha256",
        forged_sha256,
    )
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="execution_authority_mutated",
        ):
            authority.revalidate()
    finally:
        eager.__code__ = original_code
        object.__setattr__(
            authority,
            "authority_sha256",
            original_authority_sha256,
        )
        object.__setattr__(
            authority,
            "_issued_original_sha256",
            original_issued_sha256,
        )
    authority.revalidate()

    assert eager.__kwdefaults__ is not None
    original_enabled = eager.__kwdefaults__["enabled"]
    eager.__kwdefaults__["enabled"] = False
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="execution_authority_mutated",
        ):
            authority.revalidate()
    finally:
        eager.__kwdefaults__["enabled"] = original_enabled
    authority.revalidate()

    assert eager.__closure__ is not None
    inner_cell = next(
        cell for cell in eager.__closure__ if cell.cell_contents is nested
    )
    inner_cell.cell_contents = replacement_nested
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="execution_authority_mutated",
        ):
            authority.revalidate()
    finally:
        inner_cell.cell_contents = nested
    authority.revalidate()

    original_nested_code = nested.__code__
    nested.__code__ = replacement_nested.__code__.replace()
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="execution_authority_mutated",
        ):
            authority.revalidate()
    finally:
        nested.__code__ = original_nested_code
    authority.revalidate()

    original_global = globals()["_HOSTILE_EXECUTOR_GLOBAL_TARGET"]
    globals()["_HOSTILE_EXECUTOR_GLOBAL_TARGET"] = object()
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="execution_authority_mutated",
        ):
            authority.revalidate()
    finally:
        globals()["_HOSTILE_EXECUTOR_GLOBAL_TARGET"] = original_global
    authority.revalidate()

    original_helper_enabled = _HOSTILE_EXECUTOR_HELPER_STATE["enabled"]
    _HOSTILE_EXECUTOR_HELPER_STATE["enabled"] = False
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="execution_authority_mutated",
        ):
            authority.revalidate()
    finally:
        _HOSTILE_EXECUTOR_HELPER_STATE["enabled"] = original_helper_enabled
    authority.revalidate()

    assert eager.__closure__ is not None
    mutable_cell = next(
        cell
        for cell in eager.__closure__
        if type(cell.cell_contents) is dict
        and "enabled" in cell.cell_contents
    )
    closure_state = mutable_cell.cell_contents
    original_closure_enabled = closure_state["enabled"]
    closure_state["enabled"] = False
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="execution_authority_mutated",
        ):
            authority.revalidate()
    finally:
        closure_state["enabled"] = original_closure_enabled
    authority.revalidate()


def test_execution_authority_is_one_use(supported_execution) -> None:
    source, numpy_evidence, _runtime, authority, _evidence = supported_execution
    with pytest.raises(M6WaymaxOfficialError, match="execution_authority_consumed"):
        run_m6_waymax_official(source, authority, numpy_evidence)
    object.__setattr__(authority, "_consumed", False)
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="execution_authority_mutated",
        ):
            authority.revalidate()
    finally:
        object.__setattr__(authority, "_consumed", True)
    authority.revalidate()

    runtime = _MockRuntime()
    executors = runtime.executors()
    fresh = build_m6_waymax_test_execution_authority(executors)
    original_eager = executors.eager
    object.__setattr__(executors, "eager", lambda *_args, **_kwargs: None)
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="execution_authority_mutated",
        ):
            fresh.revalidate()
    finally:
        object.__setattr__(executors, "eager", original_eager)
    fresh.revalidate()
    original_executors = fresh._executors
    object.__setattr__(fresh, "_executors", runtime.executors())
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="execution_authority_mutated",
        ):
            fresh.revalidate()
    finally:
        object.__setattr__(fresh, "_executors", original_executors)
    fresh.revalidate()


def test_result_cannot_be_reconstructed_with_dataclass_replace(
    supported_execution,
) -> None:
    _source_value, numpy_evidence, _runtime, _authority, evidence = (
        supported_execution
    )
    with pytest.raises(TypeError, match="runner-issued"):
        dataclasses.replace(evidence)
    with pytest.raises(TypeError, match="factory-issued"):
        dataclasses.replace(numpy_evidence)


def test_result_bundle_binding_is_shared_and_tamper_evident(
    supported_execution,
) -> None:
    _source, numpy_evidence, _runtime, _authority, evidence = supported_execution
    original = evidence.evidence_binding_sha256
    assert isinstance(original, str) and len(original) == 64
    object.__setattr__(evidence, "evidence_binding_sha256", "0" * 64)
    try:
        with pytest.raises(M6WaymaxOfficialError, match="evidence_binding"):
            evidence.revalidate()
    finally:
        object.__setattr__(evidence, "evidence_binding_sha256", original)
    evidence.revalidate()
    original_field_table = evidence.field_comparisons
    transplanted_field_table = module.M6WaymaxOfficialFieldComparisonTable(
        selection_supported=original_field_table.selection_supported,
        selected_member_count=original_field_table.selected_member_count,
        selection_sha256=original_field_table.selection_sha256,
        primary_domain_sha256=original_field_table.primary_domain_sha256,
        rows=original_field_table.rows,
        promotable=original_field_table.promotable,
        _issuance_capability=module._FIELD_ISSUER,
    )
    assert (
        transplanted_field_table.table_binding_sha256
        == original_field_table.table_binding_sha256
    )
    original_private = evidence._issued_original_binding_sha256
    object.__setattr__(
        evidence,
        "field_comparisons",
        transplanted_field_table,
    )
    resealed_evidence = evidence._binding_sha256()
    object.__setattr__(
        evidence,
        "evidence_binding_sha256",
        resealed_evidence,
    )
    object.__setattr__(
        evidence,
        "_issued_original_binding_sha256",
        resealed_evidence,
    )
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="official_evidence_binding",
        ):
            evidence.revalidate()
    finally:
        object.__setattr__(
            evidence,
            "field_comparisons",
            original_field_table,
        )
        object.__setattr__(evidence, "evidence_binding_sha256", original)
        object.__setattr__(
            evidence,
            "_issued_original_binding_sha256",
            original_private,
        )
    evidence.revalidate()
    original_phases = numpy_evidence.phase_durations_ms
    changed_phases = dict(original_phases)
    first_phase = next(iter(changed_phases))
    changed_phases[first_phase] += 1
    object.__setattr__(
        numpy_evidence,
        "phase_durations_ms",
        changed_phases,
    )
    try:
        with pytest.raises(
            M6OfficialAdapterError,
            match="m6_official_numpy_rows_mutated",
        ):
            run_m6_waymax_official(
                _source,
                build_m6_waymax_test_execution_authority(
                    _MockRuntime().executors()
                ),
                numpy_evidence,
            )
        with pytest.raises(
            M6OfficialAdapterError,
            match="m6_official_numpy_rows_mutated",
        ):
            evidence.revalidate()
    finally:
        object.__setattr__(
            numpy_evidence,
            "phase_durations_ms",
            original_phases,
        )
    numpy_evidence.revalidate()

    def reject_numpy_mutation(name: str, mutated: Any) -> None:
        original_value = getattr(numpy_evidence, name)
        object.__setattr__(numpy_evidence, name, mutated)
        try:
            with pytest.raises(
                M6OfficialAdapterError,
                match="m6_official_numpy_rows_mutated",
            ):
                numpy_evidence.revalidate()
            with pytest.raises(
                (M6OfficialAdapterError, M6WaymaxOfficialError),
                match=(
                    "m6_official_numpy_rows_mutated"
                    "|official_evidence_binding"
                ),
            ):
                evidence.revalidate()
        finally:
            object.__setattr__(numpy_evidence, name, original_value)
        numpy_evidence.revalidate()

    original_rows = numpy_evidence.eligibility_rows
    changed_first_row = dict(original_rows[0])
    changed_first_row["cohort_index"] = 999
    reject_numpy_mutation(
        "eligibility_rows",
        (changed_first_row, *original_rows[1:]),
    )
    reject_numpy_mutation("adapter_version", "transplanted")
    reject_numpy_mutation("typed_result", object())
    reject_numpy_mutation("issuance_sha256", "0" * 64)
    reject_numpy_mutation("_issued_original_sha256", "f" * 64)
    evidence.revalidate()


def test_pilot_rejects_nonproduction_authorities_before_execution() -> None:
    _collector, source, _numpy = _source(10, full_numpy=False)
    runtime = _MockRuntime()
    authority = build_m6_waymax_test_execution_authority(
        runtime.executors()
    )
    with pytest.raises(M6WaymaxOfficialError, match="pilot_authority"):
        run_m6_waymax_outcome_suppressed_pilot(
            source,
            authority,
            clock_ns=lambda: 0,
        )
    assert runtime.calls == []
    source.revalidate()
    authority.revalidate()


def test_pilot_schema_and_elapsed_domain_are_exact() -> None:
    assert tuple(M6WaymaxPilotObservation.__annotations__) == (
        "status",
        "scene_count",
        "validation_ms",
        "scene_durations_ms",
        "execution_ms",
        "total_wall_ms",
        "max_scene_ms",
        "peak_process_rss_bytes",
        "source_binding_sha256",
        "selection_binding_sha256",
        "selected_cohort_indices_sha256",
        "execution_authority_sha256",
        "runner_binding_sha256",
        "schema_version",
        "_issuance_capability",
    )
    assert tuple(M6WaymaxPilotObservation.__annotations__).count(
        "max_scene_ms"
    ) == 1
    assert module.M6_WAYMAX_PILOT_SCHEMA_VERSION == "m6-waymax-pilot-1.2.0"

    source_binding = "1" * 64
    selection_binding = "2" * 64
    authority_binding = "3" * 64
    selected_indices_binding = "5" * 64
    runner_binding = module._pilot_runner_binding_sha256(
        source_binding_sha256=source_binding,
        selection_binding_sha256=selection_binding,
        selected_cohort_indices_sha256=selected_indices_binding,
        execution_authority_sha256=authority_binding,
    )
    observation = M6WaymaxPilotObservation(
        status="completed",
        scene_count=8,
        validation_ms=1,
        scene_durations_ms=(1,) * 8,
        execution_ms=8,
        total_wall_ms=9,
        max_scene_ms=1,
        peak_process_rss_bytes=4096,
        source_binding_sha256=source_binding,
        selection_binding_sha256=selection_binding,
        selected_cohort_indices_sha256=selected_indices_binding,
        execution_authority_sha256=authority_binding,
        runner_binding_sha256=runner_binding,
        _issuance_capability=module._PILOT_ISSUER,
    )
    observation.revalidate()
    original_observation_binding = observation.observation_binding_sha256
    assert original_observation_binding == (
        module._pilot_observation_content_sha256(observation)
    )
    assert len(original_observation_binding) == 64

    object.__setattr__(observation, "validation_ms", 2)
    object.__setattr__(observation, "total_wall_ms", 10)
    resealed_observation = module._pilot_observation_content_sha256(
        observation
    )
    assert resealed_observation != original_observation_binding
    with pytest.raises(AttributeError):
        object.__setattr__(
            observation,
            "observation_binding_sha256",
            resealed_observation,
        )
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="pilot_observation_mutated",
        ):
            observation.revalidate()
    finally:
        object.__setattr__(observation, "validation_ms", 1)
        object.__setattr__(observation, "total_wall_ms", 9)
    observation.revalidate()

    original_durations = observation.scene_durations_ms
    transplanted_durations = tuple(list(original_durations))
    assert transplanted_durations is not original_durations
    object.__setattr__(
        observation,
        "scene_durations_ms",
        transplanted_durations,
    )
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="pilot_observation_mutated",
        ):
            observation.revalidate()
    finally:
        object.__setattr__(
            observation,
            "scene_durations_ms",
            original_durations,
        )
    observation.revalidate()

    resealed_source = "4" * 64
    resealed_runner = module._pilot_runner_binding_sha256(
        source_binding_sha256=resealed_source,
        selection_binding_sha256=selection_binding,
        selected_cohort_indices_sha256=selected_indices_binding,
        execution_authority_sha256=authority_binding,
    )
    object.__setattr__(
        observation,
        "source_binding_sha256",
        resealed_source,
    )
    object.__setattr__(
        observation,
        "runner_binding_sha256",
        resealed_runner,
    )
    try:
        with pytest.raises(
            M6WaymaxOfficialError,
            match="pilot_observation_mutated",
        ):
            observation.revalidate()
    finally:
        object.__setattr__(
            observation,
            "source_binding_sha256",
            source_binding,
        )
        object.__setattr__(
            observation,
            "runner_binding_sha256",
            runner_binding,
        )
    observation.revalidate()
    with pytest.raises(M6WaymaxOfficialError, match="pilot_clock"):
        module._positive_elapsed_ms(10, 10)
