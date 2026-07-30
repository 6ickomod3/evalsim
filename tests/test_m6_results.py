"""Data-free and adversarial tests for the mode-bound M6 evidence store."""
from __future__ import annotations

import dataclasses
import copy
import hashlib
import json
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import subprocess
from unittest import mock

import pytest
import numpy as np
import pyarrow as pa

import evalsim.results.m6 as m6
import evalsim.cli.m6_official as m6_cli
import evalsim.evaluation.m6_pilot as numpy_pilot
import evalsim.evaluation.m6_waymax_official as waymax_official
from evalsim.evaluation.m6_waymax_official import (
    build_m6_waymax_unsupported_field_comparison_table,
)
from evalsim.evaluation.m6_waymax_metrics import (
    M6WaymaxParsedScalarTable,
    build_m6_waymax_live_determinism_table,
    build_m6_waymax_unsupported_determinism_table,
    parse_m6_waymax_scene_scalar_table,
)
from evalsim.results.m6 import (
    COMMITTED_MARKER,
    COMPUTE_PILOT_MODE,
    DATA_FREE_MODE,
    ELIGIBILITY_ONLY_MODE,
    M6ResultStore,
    M6ResultStoreIntegrityError,
    M6ResultStoreStateError,
    M6SanitizedAggregate,
    OFFICIAL_MODE,
    PRIMARY_MATRIX,
    PRIMARY_REPEAT_MATRIX,
    PRIMARY_REPEAT_SCENE_SCALARS,
    PRIMARY_SCENE_SCALARS,
    TERMINAL_FAILURE_MARKER,
    TERMINAL_SUCCESS_MARKER,
    reconstruct_sanitized_m6_aggregate,
    verify_m6_result_store,
)
from evalsim.perturb.m6 import evaluate_primary_brake_eligibility
from evalsim.stats.m6 import M6_STATISTICS_SCHEMA_VERSION
from evalsim.simulators.waymax_m6 import (
    M6WaymaxPrimaryDomain,
    build_m6_waymax_primary_domain_entry,
    evaluate_m6_waymax_eligibility,
    select_m6_waymax_subset,
)
from tests.test_m6_waymax import _invented_source
from tests.test_m6_waymax_official import (
    _MockRuntime as _OfficialMockRuntime,
    _source as _official_source,
)
from tests.test_m6_waymax_metrics import (
    _live_determinism_executions,
)


_ALL_PREFLIGHT_CHECKS = {
    name: True for name in m6.M6_PREFLIGHT_CHECK_DOMAIN
}
_TEST_SOURCE_PATHS = tuple(
    sorted(
        {
            ".gitignore",
            "AGENTS.md",
            "NOTICE.md",
            "docs/plans/2026-07-29-m6-counterfactual-reactivity.md",
            "evalsim/results/m6.py",
            "pyproject.toml",
            "tests/test_m6_results.py",
            "uv.lock",
        }
    )
)


def _verifier_observation(
    store: M6ResultStore,
) -> m6.M6ObservedPreflightResult:
    verified = m6.verify_committed_m6_result_store(
        store.project_root,
        store.run_name,
        expected_mode=store.profile.mode,
    )
    manifest_sha256 = hashlib.sha256(
        (store.run_path / m6.MANIFEST_PATH).read_bytes()
    ).hexdigest()
    committed_sha256 = hashlib.sha256(
        (store.run_path / COMMITTED_MARKER).read_bytes()
    ).hexdigest()
    return m6.M6ObservedPreflightResult(
        mode=store.profile.mode,
        result_path=store.project_relative_path.as_posix(),
        manifest_sha256=manifest_sha256,
        committed_sha256=committed_sha256,
        evidence_catalog_sha256=m6._terminal_evidence_catalog_sha256(
            verified.receipt,
            verified.artifacts,
        ),
        provenance_context_sha256=verified.read_dataset(
            m6.TYPED_PROVENANCE
        ).to_pylist()[0]["verification_context_sha256"],
        checks=_ALL_PREFLIGHT_CHECKS,
        _factory_sentinel=m6._OBSERVED_PREFLIGHT_SENTINEL,
    )


def _terminalize_verified(store: M6ResultStore) -> None:
    capability = m6._mint_m6_terminal_capability(
        store,
        _verifier_observation(store),
        _verified_provenance(store.profile.mode),
    )
    store.mark_terminal_success(capability=capability)


def _committed_eligibility_store(
    project: Path,
    name: str,
    *,
    eligible_n: int = 0,
) -> M6ResultStore:
    store = M6ResultStore.create(
        project,
        name,
        mode=ELIGIBILITY_ONLY_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, eligible_n, mode=ELIGIBILITY_ONLY_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    store.write_waymax_qualification(_waymax_selection(receipt, qualified_n=0))
    store.write_typed_provenance(_verified_provenance(store.profile.mode))
    store.commit()
    return store


def _project(tmp_path: Path, *, ignored: bool = True) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".gitignore").write_text(
        "outputs/\n" if ignored else "cache/\n",
        encoding="utf-8",
    )
    subprocess.run(
        ("git", "init", "-b", "main"),
        cwd=project,
        check=True,
        capture_output=True,
    )
    return project


def _eligibility_rows(
    population: int,
    eligible_n: int,
    *,
    mode: str,
    secondary_n: int | None = None,
) -> list[dict[str, object]]:
    if secondary_n is None:
        secondary_n = eligible_n
    return [
        {
            "cohort_index": index,
            "primary_eligible": index < eligible_n,
            "rejection_reason": (
                None if index < eligible_n else "ego_speed_below_5_mps"
            ),
            "secondary_b4_feasible": (
                None
                if mode == ELIGIBILITY_ONLY_MODE or index >= eligible_n
                else index < secondary_n
            ),
        }
        for index in range(population)
    ]


def _write_eligibility(
    store: M6ResultStore,
    rows: list[dict[str, object]],
) -> None:
    store.write_eligibility_ledger(rows)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _qualification_rows(
    receipt: m6.M6EligibilityReceipt,
    *,
    qualified_n: int = 0,
) -> list[dict[str, object]]:
    if receipt.mode == DATA_FREE_MODE:
        return [
            dict(row)
            for row in m6.m6_data_free_waymax_qualification_rows(
                receipt.eligible_cohort_indices
            )
        ]
    rows: list[dict[str, object]] = []
    for position, cohort_index in enumerate(
        receipt.eligible_cohort_indices
    ):
        qualified = position < qualified_n
        rows.append(
            {
                "cohort_index": cohort_index,
                "assessment_status": (
                    "qualified" if qualified else "rejected"
                ),
                "rejection_reason": (
                    None if qualified else "source_cadence_not_100ms"
                ),
                "rank_sha256": m6.m6_waymax_rank_sha256(cohort_index),
                "source_binding_sha256": _sha(f"source-{cohort_index}"),
                "primary_entry_sha256": _sha(f"entry-{cohort_index}"),
                "qualification_binding_sha256": _sha(
                    f"qualification-{cohort_index}"
                ),
                "selected": False,
                "selection_position": None,
            }
        )
    qualified = sorted(
        (row for row in rows if row["assessment_status"] == "qualified"),
        key=lambda row: (
            bytes.fromhex(str(row["rank_sha256"])),
            int(row["cohort_index"]),
        ),
    )
    selected = qualified[:16] if len(qualified) >= 8 else []
    for selection_position, row in enumerate(selected):
        row["selected"] = True
        row["selection_position"] = selection_position
    return rows


def _waymax_selection_and_domain(
    receipt: m6.M6EligibilityReceipt,
    *,
    qualified_n: int,
):
    indices = receipt.eligible_cohort_indices
    template_state, template, _primary = _invented_source()
    rows = []
    entries = []
    scenarios = {}
    ineligible = set(indices[qualified_n:])
    for cohort_index in indices:
        state = copy.deepcopy(template_state)
        scenario = copy.deepcopy(template)
        scenario.scenario_id = f"store-waymax-{cohort_index}"
        # Preserve adapter/scenario equality while making each invented source
        # commitment distinct.
        offset = np.float32(cohort_index + 1) * np.float32(0.01)
        state.log_trajectory.y[2] += offset
        scenario.agents[2].y = np.asarray(
            state.log_trajectory.y[2],
            dtype=np.float64,
        ).copy()
        if cohort_index in ineligible:
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


def _waymax_selection(
    receipt: m6.M6EligibilityReceipt,
    *,
    qualified_n: int,
):
    selection, _domain = _waymax_selection_and_domain(
        receipt,
        qualified_n=qualified_n,
    )
    return selection


def _seal_waymax_selection(
    store: M6ResultStore,
    receipt: m6.M6EligibilityReceipt,
    *,
    qualified_n: int,
):
    selection = _waymax_selection(receipt, qualified_n=qualified_n)
    store.write_waymax_qualification(selection)
    qualification = [
        dict(row)
        for row in m6._waymax_qualification_rows_from_selection(
            selection,
            receipt,
        )
    ]
    return selection, qualification


@lru_cache(maxsize=8)
def _official_waymax_evidence(qualified_n: int):
    _collector, source, numpy_evidence = _official_source(
        10,
        full_numpy=True,
        waymax_eligible_n=qualified_n,
    )
    assert numpy_evidence is not None
    runtime = _OfficialMockRuntime()
    authority = waymax_official.build_m6_waymax_test_execution_authority(
        runtime.executors()
    )
    evidence = waymax_official.run_m6_waymax_official(
        source,
        authority,
        numpy_evidence,
    )
    evidence.revalidate()
    return evidence


class _TestWaymaxStoreEvidenceProxy:
    """Expose issued tables without changing the test-authority bundle."""

    def __init__(self, evidence):
        self._evidence = evidence

    @property
    def scene_scalars(self):
        scalars = self._evidence.scene_scalars
        return getattr(scalars, "_issued", scalars)

    @property
    def determinism(self):
        determinism = self._evidence.determinism
        return getattr(determinism, "_issued", determinism)

    def __getattr__(self, name: str):
        return getattr(self._evidence, name)


@contextmanager
def _admit_test_waymax_evidence(
    store: M6ResultStore,
    expected_evidence,
):
    """Exercise production store writers with one exact test-only bundle."""

    # Real production admission is covered by the opt-in official integration;
    # separate unit tests retain rejection coverage for nonproduction evidence.
    def bind(bound_store: M6ResultStore, evidence):
        if (
            bound_store is not store
            or evidence is not expected_evidence
            or type(evidence)
            is not waymax_official.M6WaymaxOfficialEvidence
        ):
            raise M6ResultStoreIntegrityError(
                "test store writer received different Waymax evidence"
            )
        evidence.revalidate()
        selection = bound_store._waymax_selection
        selection_receipt = bound_store._require_waymax_selection_receipt()
        binding = evidence.evidence_binding_sha256
        numpy_binding = (
            evidence.numpy_comparisons.numpy_eligibility_ledger_sha256
        )
        if (
            evidence.production_authoritative is not False
            or evidence.promotable is not False
            or evidence._source_authority.kind != "test_only"
            or evidence._source_authority.promotable is not False
            or evidence._execution_authority.kind != "test_only"
            or evidence._execution_authority.promotable is not False
            or evidence.selection is not selection
            or evidence.selection.primary_domain_sha256
            != selection_receipt.primary_domain_sha256
            or evidence.selection.selection_sha256
            != selection_receipt.selector_selection_sha256
            or evidence.supported
            is not selection_receipt.selection_supported
            or type(binding) is not str
            or m6._SHA256.fullmatch(binding) is None
            or type(numpy_binding) is not str
            or m6._SHA256.fullmatch(numpy_binding) is None
        ):
            raise M6ResultStoreIntegrityError(
                "test Waymax evidence differs from its sealed selection"
            )
        if bound_store._waymax_official_evidence is None:
            bound_store._waymax_official_evidence = evidence
            bound_store._waymax_official_evidence_binding_sha256 = binding
            bound_store._waymax_numpy_eligibility_ledger_sha256 = (
                numpy_binding
            )
        elif (
            bound_store._waymax_official_evidence is not evidence
            or bound_store._waymax_official_evidence_binding_sha256 != binding
            or bound_store._waymax_numpy_eligibility_ledger_sha256
            != numpy_binding
        ):
            raise M6ResultStoreIntegrityError(
                "test Waymax writers did not share one evidence bundle"
            )
        return _TestWaymaxStoreEvidenceProxy(evidence)

    with mock.patch.object(M6ResultStore, "_bind_waymax_official_evidence", bind):
        yield


def _scene_rows(
    indices: tuple[int, ...],
    fingerprint: str,
    *,
    value: float = 0.0,
) -> list[dict[str, object]]:
    units = {name: unit for name, _version, unit in m6.M6_PRIMARY_METRICS}
    rows: list[dict[str, object]] = []
    for cohort_index in indices:
        for policy, access, metric, version in m6.M6_PRIMARY_CELL_DOMAIN:
            timeliness = metric == "response_timeliness_s"
            rows.append(
                {
                    "cohort_index": cohort_index,
                    "policy_name": policy,
                    "policy_access_role": access,
                    "metric_name": metric,
                    "metric_version": version,
                    "unit": units[metric],
                    "value": 0.0 if timeliness else value,
                    "responded": False if timeliness else None,
                    "responder_latency_s": None,
                    "source_pairing_complete": True,
                    "intervention_config_fingerprint": fingerprint,
                }
            )
    return rows


def _negative_observation_rows(
    receipt: m6.M6EligibilityReceipt,
) -> list[dict[str, object]]:
    return [
        {
            "gate_name": gate,
            "cohort_index": cohort,
            "policy_name": policy,
            "assessed_n": 1,
            "violation_n": 0,
            "observation_sha256": _sha(
                f"{gate}:{cohort}:{policy or 'none'}"
            ),
        }
        for gate, cohort, policy in m6._negative_timing_observation_domain(
            receipt
        )
    ]


def _selected_by_position(
    qualification: list[dict[str, object]],
) -> dict[int, dict[str, object]]:
    return {
        int(row["selection_position"]): row
        for row in qualification
        if row["selected"] is True
    }


def _waymax_scalar_rows(
    qualification: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected = _selected_by_position(qualification)
    units = {name: unit for name, _version, unit in m6.M6_PRIMARY_METRICS}
    versions = {
        name: version for name, version, _unit in m6.M6_PRIMARY_METRICS
    }
    rows: list[dict[str, object]] = []
    for position in range(16):
        selected_row = selected.get(position)
        for bundle in m6.M6_WAYMAX_BUNDLES:
            for metric, _version, _unit in m6.M6_PRIMARY_METRICS:
                selected_status = selected_row is not None
                rows.append(
                    {
                        "selection_position": position,
                        "cohort_index": (
                            None
                            if selected_row is None
                            else selected_row["cohort_index"]
                        ),
                        "qualification_binding_sha256": (
                            None
                            if selected_row is None
                            else selected_row[
                                "qualification_binding_sha256"
                            ]
                        ),
                        "bundle": bundle,
                        "metric_name": metric,
                        "metric_version": versions[metric],
                        "value_unit": units[metric],
                        "value": 0.0 if selected_status else None,
                        "responded": (
                            False
                            if selected_status
                            and metric == "response_timeliness_s"
                            else None
                        ),
                        "responder_latency_s": None,
                        "source_pairing_complete": selected_status,
                        "status": (
                            "selected"
                            if selected_status
                            else "not_selected"
                        ),
                    }
                )
    return rows


def _waymax_field_rows(
    qualification: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected = _selected_by_position(qualification)
    rows: list[dict[str, object]] = []
    for position in range(16):
        selected_row = selected.get(position)
        for bundle in m6.M6_WAYMAX_BUNDLES:
            for condition in m6.M6_WAYMAX_CONDITIONS:
                for field_name in m6.M6_WAYMAX_COMPARISON_FIELDS:
                    exact = field_name in m6.M6_WAYMAX_EXACT_FIELDS
                    rows.append(
                        {
                            "selection_position": position,
                            "bundle": bundle,
                            "condition": condition,
                            "field_name": field_name,
                            "cohort_index": (
                                None
                                if selected_row is None
                                else selected_row["cohort_index"]
                            ),
                            "qualification_binding_sha256": (
                                None
                                if selected_row is None
                                else selected_row[
                                    "qualification_binding_sha256"
                                ]
                            ),
                            "comparison_kind": (
                                "exact" if exact else "tolerance"
                            ),
                            "denominator": (
                                20 if selected_row is not None else None
                            ),
                            "max_abs_error": (
                                0.0 if selected_row is not None else None
                            ),
                            "max_normalized_error": (
                                0.0
                                if selected_row is not None and not exact
                                else None
                            ),
                            "tolerance_failures": (
                                0 if selected_row is not None else None
                            ),
                            "binary_mismatches": (
                                0 if selected_row is not None else None
                            ),
                            "status": (
                                "passed"
                                if selected_row is not None
                                else "not_applicable"
                            ),
                        }
                    )
    return rows


def _waymax_determinism_rows(
    qualification: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected = _selected_by_position(qualification)
    rows: list[dict[str, object]] = []
    for position in range(16):
        selected_row = selected.get(position)
        for bundle in m6.M6_WAYMAX_BUNDLES:
            for condition in m6.M6_WAYMAX_CONDITIONS:
                digest = _sha(f"repeat:{position}:{bundle}:{condition}")
                rows.append(
                    {
                        "selection_position": position,
                        "bundle": bundle,
                        "condition": condition,
                        "cohort_index": (
                            None
                            if selected_row is None
                            else selected_row["cohort_index"]
                        ),
                        "qualification_binding_sha256": (
                            None
                            if selected_row is None
                            else selected_row[
                                "qualification_binding_sha256"
                            ]
                        ),
                        "status": (
                            "passed"
                            if selected_row is not None
                            else "not_applicable"
                        ),
                        "eager_pass_1_sha256": (
                            digest if selected_row is not None else None
                        ),
                        "eager_pass_2_sha256": (
                            digest if selected_row is not None else None
                        ),
                        "jit_eager_sha256": (
                            digest
                            if selected_row is not None and position == 0
                            else None
                        ),
                        "jit_compiled_sha256": (
                            digest
                            if selected_row is not None and position == 0
                            else None
                        ),
                    }
                )
    return rows


def _provenance(mode: str) -> dict[str, object]:
    data_free = mode == DATA_FREE_MODE
    return {
        "plan_version": m6.M6_PLAN_VERSION,
        "config_version": m6.M6_CONFIG_VERSION,
        "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
        "population_label": (
            "synthetic_data_free_n10"
            if data_free
            else "accepted_m4_complete_case_ten_shard_cohort"
        ),
        "source_shard_start": None if data_free else "00000",
        "source_shard_end": None if data_free else "00009",
        "approved_git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "executable_source_sha256": "c" * 64,
        "uv_lock_sha256": "d" * 64,
        "runtime_config_sha256": "e" * 64,
        "accepted_m4_manifest_sha256": None if data_free else "f" * 64,
        "accepted_m4_provenance_sha256": None if data_free else "0" * 64,
        "python_version": "3.11.0",
        "numpy_version": "2.2.6",
        "pyarrow_version": "19.0.1",
        "jax_version": None if data_free else "0.4.38",
        "jaxlib_version": None if data_free else "0.4.38",
        "tensorflow_version": None if data_free else "2.18.1",
        "waymax_commit": None if data_free else m6.WAYMAX_COMMIT,
        "jax_backend": None if data_free else "cpu",
        "jax_device_class": None if data_free else "cpu",
        "primary_intervention_fingerprint": (
            m6.M6_PRIMARY_INTERVENTION_FINGERPRINT
        ),
        "secondary_intervention_fingerprint": (
            m6.M6_SECONDARY_INTERVENTION_FINGERPRINT
        ),
    }


def _verified_provenance(
    mode: str,
    *,
    row: dict[str, object] | None = None,
) -> m6.M6VerifiedProvenance:
    return m6._issue_m6_verified_provenance(
        mode=mode,
        row=_provenance(mode) if row is None else row,
        source_paths=_TEST_SOURCE_PATHS,
    )


def _pilot_summary_row(
    *,
    total_wall_ms: int = 1_000,
    max_scene_ms: int = 200,
    decode_ms: int = 100,
    numpy_ms: int | None = None,
    waymax_ms: int | None = None,
    verification_ms: int = 100,
    fresh_worker_peak_rss_bytes: int = 1024,
    passed: bool | None = None,
) -> dict[str, object]:
    observed_numpy_ms = (
        max_scene_ms + 7 if numpy_ms is None else numpy_ms
    )
    observed_waymax_ms = 1 if waymax_ms is None else waymax_ms
    expected_passed = (
        total_wall_ms <= 30 * 60 * 1000
        and max_scene_ms <= 10 * 60 * 1000
        and fresh_worker_peak_rss_bytes <= 16 * 1024**3
    )
    return {
        "pilot_scene_n": 8,
        "total_wall_ms": total_wall_ms,
        "max_scene_ms": max_scene_ms,
        "decode_ms": decode_ms,
        "numpy_ms": observed_numpy_ms,
        "waymax_ms": observed_waymax_ms,
        "verification_ms": verification_ms,
        "fresh_worker_peak_rss_bytes": fresh_worker_peak_rss_bytes,
        "passed": expected_passed if passed is None else passed,
    }


def _pilot_observations(
    selection,
    summary: dict[str, object],
    *,
    selected_cohort_indices_override: tuple[int, ...] | None = None,
):
    selection_binding = waymax_official.m6_waymax_selection_binding_sha256(
        selection
    )
    selected_cohort_indices = tuple(
        member.cohort_index
        for member in (
            selection.members[:8]
            if selection.supported
            else sorted(
                selection.qualification_ledger.rows,
                key=lambda row: (
                    bytes.fromhex(row.rank_sha256),
                    row.cohort_index,
                ),
            )[:8]
        )
    )
    if selected_cohort_indices_override is not None:
        selected_cohort_indices = selected_cohort_indices_override
    selected_indices_sha256 = (
        numpy_pilot.m6_numpy_pilot_selected_cohort_indices_sha256(
            selected_cohort_indices
        )
    )
    max_scene_ms = int(summary["max_scene_ms"])
    numpy_durations = (max_scene_ms,) + (1,) * 7
    numpy_observation = numpy_pilot.M6NumpyPilotObservation(
        scene_count=8,
        scene_durations_ms=numpy_durations,
        total_execution_ms=sum(numpy_durations),
        max_scene_ms=max_scene_ms,
        selected_cohort_indices_sha256=selected_indices_sha256,
        source_selection_binding_sha256=selection_binding,
        execution_binding_sha256=_sha("pilot-numpy-execution"),
        _issuance_capability=numpy_pilot._ISSUER,
    )
    source_binding = _sha("pilot-waymax-source")
    authority_binding = _sha("pilot-waymax-authority")
    runner_binding = waymax_official._pilot_runner_binding_sha256(
        source_binding_sha256=source_binding,
        selection_binding_sha256=selection_binding,
        selected_cohort_indices_sha256=selected_indices_sha256,
        execution_authority_sha256=authority_binding,
    )
    if selection.supported:
        waymax_durations = (max_scene_ms,) + (1,) * 7
        waymax_observation = waymax_official.M6WaymaxPilotObservation(
            status="completed",
            scene_count=8,
            validation_ms=1,
            scene_durations_ms=waymax_durations,
            execution_ms=sum(waymax_durations),
            total_wall_ms=1 + sum(waymax_durations),
            max_scene_ms=max_scene_ms,
            peak_process_rss_bytes=int(
                summary["fresh_worker_peak_rss_bytes"]
            ),
            source_binding_sha256=source_binding,
            selection_binding_sha256=selection_binding,
            selected_cohort_indices_sha256=selected_indices_sha256,
            execution_authority_sha256=authority_binding,
            runner_binding_sha256=runner_binding,
            _issuance_capability=waymax_official._PILOT_ISSUER,
        )
    else:
        waymax_observation = waymax_official.M6WaymaxPilotObservation(
            status="unsupported",
            scene_count=0,
            validation_ms=1,
            scene_durations_ms=(),
            execution_ms=0,
            total_wall_ms=0,
            max_scene_ms=0,
            peak_process_rss_bytes=0,
            source_binding_sha256=source_binding,
            selection_binding_sha256=selection_binding,
            selected_cohort_indices_sha256=selected_indices_sha256,
            execution_authority_sha256=authority_binding,
            runner_binding_sha256=runner_binding,
            _issuance_capability=waymax_official._PILOT_ISSUER,
        )
    return numpy_observation, waymax_observation


def _pilot_execution_evidence(
    store: M6ResultStore,
    selection,
    provenance: m6.M6VerifiedProvenance,
    *,
    summary: dict[str, object] | None = None,
    observation_summary: dict[str, object] | None = None,
    observation_selected_cohort_indices: tuple[int, ...] | None = None,
):
    row = _pilot_summary_row() if summary is None else dict(summary)
    numpy_observation, waymax_observation = _pilot_observations(
        selection,
        row if observation_summary is None else observation_summary,
        selected_cohort_indices_override=(
            observation_selected_cohort_indices
        ),
    )
    return m6_cli._issue_m6_compute_pilot_evidence(
        eligibility_rows=_eligibility_rows(
            128,
            10,
            mode=COMPUTE_PILOT_MODE,
        ),
        selection=selection,
        pilot_summary=row,
        pilot_selection_positions=tuple(range(8)),
        numpy_observation=numpy_observation,
        waymax_observation=waymax_observation,
        verified_provenance=provenance,
        run_name=store.run_name,
        result_path=store.project_relative_path.as_posix(),
        fresh_worker_peak_rss_bytes=int(
            row["fresh_worker_peak_rss_bytes"]
        ),
    )


def _pilot_normalization_row(**overrides: object) -> dict[str, object]:
    row = {
        **_pilot_summary_row(),
        "selection_binding_sha256": "1" * 64,
        "selected_cohort_indices_sha256": "5" * 64,
        "numpy_observation_content_sha256": "2" * 64,
        "waymax_observation_content_sha256": "3" * 64,
        "pilot_report_binding_sha256": "4" * 64,
    }
    row.update(overrides)
    return row


def _stage_rows(mode: str) -> list[dict[str, object]]:
    return [
        {
            "stage_name": name,
            "duration_ms": index + 1 if mode == OFFICIAL_MODE else index,
        }
        for index, name in enumerate(m6.M6_STAGE_DOMAIN)
    ]


def _official_claim_receipt() -> m6.M6EligibilityReceipt:
    return m6.M6EligibilityReceipt(
        mode=OFFICIAL_MODE,
        population_size=128,
        eligible_cohort_indices=tuple(range(10)),
        secondary_b4_cohort_indices=tuple(range(10)),
        rejection_reason_counts={
            reason: 118 if index == 0 else 0
            for index, reason in enumerate(m6.M6_PRIMARY_REJECTION_REASONS)
        },
        primary_intervention_fingerprint=(
            m6.M6_PRIMARY_INTERVENTION_FINGERPRINT
        ),
        secondary_intervention_fingerprint=(
            m6.M6_SECONDARY_INTERVENTION_FINGERPRINT
        ),
    )


def _claim_primary_matrix(
    receipt: m6.M6EligibilityReceipt,
    *,
    idm_responder_n: int,
) -> tuple[dict[str, object], ...]:
    rows = [
        dict(row)
        for row in m6._derive_primary_matrix_rows(
            _scene_rows(
                receipt.eligible_cohort_indices,
                m6.M6_PRIMARY_INTERVENTION_FINGERPRINT,
            ),
            receipt,
        )
    ]
    target = next(
        row
        for row in rows
        if row["policy_name"] == "idm"
        and row["metric_name"] == "response_timeliness_s"
    )
    target["responder_n"] = idm_responder_n
    target["censor_n"] = receipt.eligible_count - idm_responder_n
    if idm_responder_n >= 10:
        target.update(
            {
                "conditional_latency_status": "descriptive",
                "conditional_latency_suppression_reason": None,
                "conditional_latency_mean_s": 0.2,
                "conditional_latency_median_s": 0.2,
            }
        )
    return tuple(rows)


def _accepted_claim_accounting() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for raw in m6.m6_waymax_unsupported_rows(10):
        row = dict(raw)
        record_type = row["record_type"]
        name = row["name"]
        row["status"] = "accepted"
        if record_type == "scope":
            row["count"] = {
                "qualified_count": 10,
                "selected_count": 10,
                "transition_count": 20,
            }[name]
        elif record_type == "selection_rejection":
            row["count"] = 0
            row["opportunity_n"] = 10
        elif record_type == "field_comparison":
            row["denominator"] = 1
            row["binary_mismatches"] = 0
            row["max_abs_error"] = 0.0
        elif record_type == "control_partition":
            row["opportunity_n"] = 400
            row["count"] = (
                400
                if name
                in {"target_requested_control", "target_effective_control"}
                else 0
            )
        else:
            row.update(
                {
                    "status": "descriptive",
                    "pair_n": 10,
                    "thresholded_nonzero_n": 0,
                    "suppression_reason": None,
                    "arithmetic_mean": 0.0,
                    "median": 0.0,
                    "pointwise_level": 0.95,
                    "pointwise_lower": 0.0,
                    "pointwise_upper": 0.0,
                }
            )
            if row["metric_name"] == "response_timeliness_s":
                row["responder_n"] = 0
                row["censor_n"] = 10
        rows.append(row)
    return tuple(rows)


def _claim_determinism(*, executed: bool) -> m6.M6DeterminismReceipt:
    return m6.M6DeterminismReceipt(
        mode=OFFICIAL_MODE,
        primary_scene_pass_1_sha256="a" * 64,
        primary_scene_pass_2_sha256="a" * 64,
        primary_matrix_pass_1_sha256="b" * 64,
        primary_matrix_pass_2_sha256="b" * 64,
        waymax_repeat_status="passed" if executed else "not_applicable",
        waymax_repeat_rows=40 if executed else 0,
    )


def _review_rows(
    *,
    p3_count: int = 0,
) -> list[dict[str, object]]:
    return [
        {
            "role": role,
            "approved_git_commit": "a" * 40,
            "decision": "accept",
            "p1_count": 0,
            "p2_count": 0,
            "p3_count": p3_count,
        }
        for role in m6.M6_REVIEW_ROLE_DOMAIN
    ]


def _write_complete(
    store: M6ResultStore,
    *,
    qualified_n: int = 0,
    secondary_value: float = 0.0,
    seal_reviews: bool = True,
    official_evidence=None,
) -> M6ResultStore:
    eligible_n = 10
    _write_eligibility(
        store,
        _eligibility_rows(
            store.profile.population_size,
            eligible_n,
            mode=store.profile.mode,
            secondary_n=eligible_n,
        ),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    if store.profile.data_free:
        qualification = _qualification_rows(receipt)
        store.write_waymax_qualification()
        selection = None
    else:
        waymax_evidence = (
            _official_waymax_evidence(qualified_n)
            if official_evidence is None
            else official_evidence
        )
        waymax_evidence.revalidate()
        selection = waymax_evidence.selection
        store.write_waymax_qualification(selection)
        qualification = [
            dict(row)
            for row in m6._waymax_qualification_rows_from_selection(
                selection,
                receipt,
            )
        ]
    if store.profile.data_free:
        primary = _scene_rows(
            receipt.eligible_cohort_indices,
            m6.M6_PRIMARY_INTERVENTION_FINGERPRINT,
        )
        secondary = _scene_rows(
            receipt.secondary_b4_cohort_indices,
            m6.M6_SECONDARY_INTERVENTION_FINGERPRINT,
            value=secondary_value,
        )
        negative = _negative_observation_rows(receipt)
    else:
        numpy_source = waymax_evidence._numpy_evidence
        primary = [
            dict(row) for row in numpy_source.primary_scene_scalar_rows
        ]
        secondary = [
            dict(row) for row in numpy_source.secondary_scene_scalar_rows
        ]
        negative = [
            dict(row)
            for row in numpy_source.negative_timing_observation_rows
        ]
    store.write_primary_scene_scalars(primary)
    store.write_primary_matrix()
    store.write_primary_repeat_scene_scalars([dict(row) for row in primary])
    store.write_primary_repeat_matrix()
    store.write_secondary_scene_scalars(secondary)
    store.write_secondary_matrix()
    store.write_negative_timing_observations(negative)
    if store.profile.data_free:
        store.write_waymax_scene_scalars()
        store.write_waymax_field_comparisons()
        store.write_waymax_numpy_comparisons()
        store.write_waymax_determinism()
    else:
        with _admit_test_waymax_evidence(store, waymax_evidence):
            store.write_waymax_scene_scalars(waymax_evidence)
            store.write_waymax_field_comparisons(waymax_evidence)
            store.write_waymax_numpy_comparisons(waymax_evidence)
            store.write_waymax_determinism(waymax_evidence)
    store.write_waymax_accounting()
    store.write_typed_provenance(_verified_provenance(store.profile.mode))
    store.write_stage_timings(_stage_rows(store.profile.mode))
    store.write_determinism_receipt()
    store.write_claim_limitations()
    if not seal_reviews:
        return store
    if store.profile.data_free:
        store.write_data_free_review_absence()
    else:
        verification = store.write_mechanical_verification_receipt()
        store.seal_awaiting_review(fresh_worker_peak_rss_bytes=1024)
        store = M6ResultStore.adopt_awaiting_review(
            store.project_root,
            store.run_name,
        )
        decisions = tuple(
            m6.issue_m6_review_decision(
                verification,
                role=role,
                decision="accept",
                p1_count=0,
                p2_count=0,
                p3_count=0,
            )
            for role in m6.M6_REVIEW_ROLE_DOMAIN
        )
        store.write_review_decisions(verification, decisions)
    store.write_execution_summary(
        fresh_worker_peak_rss_bytes=(
            1024 if store.profile.mode == OFFICIAL_MODE else 0
        )
    )
    return store

def _successful_data_free(project: Path, name: str = "data-free") -> M6ResultStore:
    store = M6ResultStore.create(project, name, mode=DATA_FREE_MODE)
    _write_complete(store)
    store.finalize()
    return store


def _valid_promoted_payload(
    project: Path,
) -> dict[str, object]:
    store = _successful_data_free(project, "aggregate-fixture")
    verified = verify_m6_result_store(
        project,
        "aggregate-fixture",
        allow_data_free=True,
    )
    receipt = verified.receipt
    eligible = receipt.eligible_count
    secondary = receipt.secondary_b4_count
    matrix = [
        m6._promoted_primary_row(row)
        for row in m6._normalize_primary_matrix(
            verified.read_dataset(m6.PRIMARY_MATRIX).to_pylist(),
            receipt,
        )
    ]
    gates = [
        {
            "assessed_n": row["assessed_n"],
            "gate_name": row["gate_name"],
            "passed_n": row["passed_n"],
            "status": row["status"],
            "violation_n": row["violation_n"],
        }
        for row in m6._normalize_negative_timing_gates(
            verified.read_dataset(
                m6.NEGATIVE_TIMING_GATES
            ).to_pylist(),
            receipt,
        )
    ]
    waymax = m6._promoted_waymax_scope(
        m6._normalize_waymax_accounting(
            verified.read_dataset(m6.WAYMAX_ACCOUNTING).to_pylist(),
            receipt,
        )
    )
    waymax["rejection_reason_counts"] = {
        reason: (
            eligible
            if index == 0
            else 0
        )
        for index, reason in enumerate(m6.M6_WAYMAX_REJECTION_REASONS)
    }
    required_counts = {
        "eligibility_rows": 128,
        "primary_scene_rows": eligible * 12,
        "primary_matrix_rows": 12,
        "primary_repeat_scene_rows": eligible * 12,
        "primary_repeat_matrix_rows": 12,
        "secondary_scene_rows": secondary * 12,
        "secondary_matrix_rows": 12,
        "negative_timing_observation_rows": eligible * 9 + secondary,
        "negative_timing_gate_rows": len(
            m6.M6_NEGATIVE_TIMING_GATE_DOMAIN
        ),
        "waymax_accounting_rows": len(m6.M6_WAYMAX_ROW_DOMAIN),
        "waymax_qualification_rows": eligible,
        "waymax_scene_scalar_rows": 128,
        "waymax_field_comparison_rows": 640,
        "waymax_determinism_rows": 64,
        "stage_timing_rows": len(m6.M6_STAGE_DOMAIN),
        "review_decision_rows": len(m6.M6_REVIEW_ROLE_DOMAIN),
    }
    rejection_reasons = {
        reason: (
            128 - eligible
            if reason == "ego_speed_below_5_mps"
            else 0
        )
        for reason in m6.M6_PRIMARY_REJECTION_REASONS
    }
    return {
        "claim_and_limitations": {
            "bounded_claim": m6.M6_BLOCKED_BOUNDED_CLAIM,
            "claim_status": "blocked",
            "limitations": list(m6.M6_FIXED_LIMITATIONS),
        },
        "eligibility": {
            "primary_eligible_count": eligible,
            "rejection_reason_counts": rejection_reasons,
            "total": 128,
        },
        "execution": {
            "aggregate_stage_durations_ms": {
                name: index + 1
                for index, name in enumerate(m6.M6_STAGE_DOMAIN)
            },
            "deterministic_repeat_status": "passed",
            "fresh_worker_peak_rss_bytes": 1024,
            "gate_status": {
                "release": "accepted",
                "real_reactivity_claim": "blocked",
                "waymax": "unsupported",
            },
            "required_row_domain_counts": required_counts,
            "review_decisions": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "approved_git_commit"
                }
                for row in _review_rows()
            ],
        },
        "negative_control_and_timing_gates": gates,
        "primary_matrix": matrix,
        "provenance_labels": {
            "aggregate_schema_version": (
                m6.M6_PROMOTED_AGGREGATE_SCHEMA_VERSION
            ),
            "config_version": m6.M6_CONFIG_VERSION,
            "fixed_limitations": list(m6.M6_FIXED_LIMITATIONS),
            "horizons": {
                "numpy_transitions": 40,
                "waymax_transitions": 20,
            },
            "interventions": [
                {
                    "deceleration_mps2": 0.0,
                    "name": "identity",
                    "version": "v1",
                },
                {
                    "deceleration_mps2": 2.0,
                    "duration_s": 1.0,
                    "name": "longitudinal_brake_pulse",
                    "role": "primary",
                    "version": "v1",
                },
                {
                    "deceleration_mps2": 4.0,
                    "duration_s": 1.0,
                    "name": "longitudinal_brake_pulse",
                    "role": "local_secondary_not_numeric_public",
                    "version": "v1",
                },
            ],
            "plan_version": m6.M6_PLAN_VERSION,
            "policies": [
                {"access_role": access, "name": policy}
                for policy, access in m6.M6_PRIMARY_POLICY_ROLES
            ],
            "population_label": (
                "accepted_m4_complete_case_ten_shard_cohort"
            ),
            "result_schema_version": m6.M6_RESULT_STORE_SCHEMA_VERSION,
            "source_shard_suffix_range": ["00000", "00009"],
            "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
        },
        "waymax_scope": waymax,
    }

@pytest.mark.parametrize("eligible_n", [0, 9, 128])
def test_eligibility_only_seals_source_qualification_and_terminalizes(
    tmp_path: Path,
    eligible_n: int,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        f"eligibility-{eligible_n}",
        mode=ELIGIBILITY_ONLY_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(
            128,
            eligible_n,
            mode=ELIGIBILITY_ONLY_MODE,
        ),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    store.write_waymax_qualification(
        _waymax_selection(receipt, qualified_n=0)
    )
    store.write_typed_provenance(_verified_provenance(store.profile.mode))
    store.commit()
    verified = m6.verify_committed_m6_result_store(
        project,
        f"eligibility-{eligible_n}",
        expected_mode=ELIGIBILITY_ONLY_MODE,
    )
    assert set(verified.tables) == {
        m6.ELIGIBILITY_LEDGER,
        m6.WAYMAX_QUALIFICATION,
        m6.TYPED_PROVENANCE,
    }
    _terminalize_verified(store)
    terminal = verify_m6_result_store(
        project,
        f"eligibility-{eligible_n}",
        expected_mode=ELIGIBILITY_ONLY_MODE,
    )
    assert terminal.run_path == store.run_path
    assert (store.run_path / TERMINAL_SUCCESS_MARKER).is_file()
    assert not (store.run_path / TERMINAL_FAILURE_MARKER).exists()


def test_eligibility_only_requires_complete_typed_provenance(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "eligibility-missing-provenance",
        mode=ELIGIBILITY_ONLY_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=ELIGIBILITY_ONLY_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    store.write_waymax_qualification(
        _waymax_selection(receipt, qualified_n=0)
    )

    with pytest.raises(M6ResultStoreIntegrityError, match="missing required"):
        store.commit()
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()


def test_non_data_free_provenance_requires_complete_runtime(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "eligibility-runtime-provenance",
        mode=ELIGIBILITY_ONLY_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=ELIGIBILITY_ONLY_MODE),
    )
    row = _provenance(ELIGIBILITY_ONLY_MODE)
    row["jax_version"] = None

    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="requires the complete",
    ):
        store.write_typed_provenance(
            _verified_provenance(ELIGIBILITY_ONLY_MODE, row=row)
        )
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()


def test_typed_provenance_rejects_caller_mapping(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "eligibility-caller-provenance",
        mode=ELIGIBILITY_ONLY_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=ELIGIBILITY_ONLY_MODE),
    )

    with pytest.raises(TypeError, match="verifier-issued"):
        store.write_typed_provenance(  # type: ignore[arg-type]
            _provenance(ELIGIBILITY_ONLY_MODE)
        )
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()


def test_compute_pilot_is_outcome_suppressed_and_capability_bound(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "pilot", mode=COMPUTE_PILOT_MODE)
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=COMPUTE_PILOT_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection = _waymax_selection(receipt, qualified_n=0)
    store.write_waymax_qualification(selection)
    provenance = _verified_provenance(store.profile.mode)
    evidence = _pilot_execution_evidence(
        store,
        selection,
        provenance,
    )
    store.write_compute_pilot_summary(evidence)
    store.write_typed_provenance(provenance)
    store.commit()
    verified = m6.verify_committed_m6_result_store(
        project,
        "pilot",
        expected_mode=COMPUTE_PILOT_MODE,
    )
    assert set(verified.tables) == {
        m6.ELIGIBILITY_LEDGER,
        m6.WAYMAX_QUALIFICATION,
        m6.COMPUTE_PILOT_SUMMARY,
        m6.TYPED_PROVENANCE,
    }
    pilot_row = verified.read_dataset(
        m6.COMPUTE_PILOT_SUMMARY
    ).to_pylist()[0]
    assert pilot_row["selection_binding_sha256"] == (
        verified.waymax_selection_receipt.selection_binding_sha256
    )
    assert pilot_row["selected_cohort_indices_sha256"] == (
        evidence.pilot_selected_cohort_indices_sha256
    )
    assert pilot_row["numpy_observation_content_sha256"] == (
        evidence.pilot_numpy_observation_content_sha256
    )
    assert pilot_row["waymax_observation_content_sha256"] == (
        evidence.pilot_waymax_observation_content_sha256
    )
    assert pilot_row["pilot_report_binding_sha256"] == (
        evidence.pilot_report_binding_sha256
    )
    assert not any("scene_durations" in name for name in pilot_row)
    assert {
        name for name in pilot_row if "cohort" in name
    } == {"selected_cohort_indices_sha256"}
    _terminalize_verified(store)
    terminal = verify_m6_result_store(
        project,
        "pilot",
        expected_mode=COMPUTE_PILOT_MODE,
    )
    assert terminal.run_path == store.run_path
    assert (store.run_path / TERMINAL_SUCCESS_MARKER).is_file()


def test_compute_pilot_store_rejects_raw_mapping(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "pilot-raw-mapping",
        mode=COMPUTE_PILOT_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=COMPUTE_PILOT_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    store.write_waymax_qualification(
        _waymax_selection(receipt, qualified_n=0)
    )
    with pytest.raises(TypeError, match="runner-issued mode evidence"):
        store.write_compute_pilot_summary(_pilot_summary_row())
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()


def test_authentic_pilot_observations_reject_fabricated_summary(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "pilot-fabricated-summary",
        mode=COMPUTE_PILOT_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=COMPUTE_PILOT_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection = _waymax_selection(receipt, qualified_n=0)
    provenance = _verified_provenance(COMPUTE_PILOT_MODE)
    authentic_summary = _pilot_summary_row()
    fabricated_summary = {
        **authentic_summary,
        "max_scene_ms": int(authentic_summary["max_scene_ms"]) + 1,
    }
    with pytest.raises(ValueError, match="mechanically consistent"):
        _pilot_execution_evidence(
            store,
            selection,
            provenance,
            summary=fabricated_summary,
            observation_summary=authentic_summary,
        )


def test_observation_derived_stage_times_reject_forged_one_ms_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "pilot-forged-one-ms",
        mode=COMPUTE_PILOT_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=COMPUTE_PILOT_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection = _waymax_selection(receipt, qualified_n=0)
    store.write_waymax_qualification(selection)
    provenance = _verified_provenance(COMPUTE_PILOT_MODE)
    authentic_summary = _pilot_summary_row()
    forged_summary = {
        **authentic_summary,
        "total_wall_ms": 200,
        "max_scene_ms": 200,
        "decode_ms": 1,
        "numpy_ms": 1,
        "waymax_ms": 1,
        "verification_ms": 1,
    }
    with pytest.raises(ValueError, match="mechanically consistent"):
        _pilot_execution_evidence(
            store,
            selection,
            provenance,
            summary=forged_summary,
            observation_summary=authentic_summary,
        )

    evidence = _pilot_execution_evidence(
        store,
        selection,
        provenance,
        summary=authentic_summary,
    )
    original_summary = evidence.pilot_summary
    object.__setattr__(evidence, "pilot_summary", forged_summary)
    monkeypatch.setattr(
        m6_cli.M6ModeExecutionEvidence,
        "revalidate_pilot",
        lambda self, **kwargs: None,
    )
    try:
        with pytest.raises(
            M6ResultStoreIntegrityError,
            match="observations disagree",
        ):
            store.write_compute_pilot_summary(evidence)
    finally:
        object.__setattr__(evidence, "pilot_summary", original_summary)
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()
    assert not (store.run_path / TERMINAL_SUCCESS_MARKER).exists()


def test_different_valid_first_eight_is_rejected_by_seal_and_reopen(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "pilot-wrong-first-eight",
        mode=COMPUTE_PILOT_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=COMPUTE_PILOT_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection = _waymax_selection(receipt, qualified_n=0)
    store.write_waymax_qualification(selection)
    provenance = _verified_provenance(COMPUTE_PILOT_MODE)
    ranked_rows = tuple(
        sorted(
            selection.qualification_ledger.rows,
            key=lambda row: (
                bytes.fromhex(row.rank_sha256),
                row.cohort_index,
            ),
        )
    )
    canonical_indices = tuple(row.cohort_index for row in ranked_rows[:8])
    different_indices = tuple(row.cohort_index for row in ranked_rows[1:9])
    assert len(different_indices) == 8
    assert set(different_indices) != set(canonical_indices)
    with pytest.raises(ValueError, match="mechanically consistent"):
        _pilot_execution_evidence(
            store,
            selection,
            provenance,
            observation_selected_cohort_indices=different_indices,
        )

    evidence = _pilot_execution_evidence(
        store,
        selection,
        provenance,
    )
    stored_row = {
        **dict(evidence.pilot_summary or {}),
        "selection_binding_sha256": evidence.pilot_selection_binding_sha256,
        "selected_cohort_indices_sha256": (
            numpy_pilot.m6_numpy_pilot_selected_cohort_indices_sha256(
                different_indices
            )
        ),
        "numpy_observation_content_sha256": (
            evidence.pilot_numpy_observation_content_sha256
        ),
        "waymax_observation_content_sha256": (
            evidence.pilot_waymax_observation_content_sha256
        ),
        "pilot_report_binding_sha256": evidence.pilot_report_binding_sha256,
    }
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="selected first-eight binding",
    ):
        m6._normalize_compute_pilot(
            (stored_row,),
            receipt,
            selected_cohort_indices_sha256=(
                evidence.pilot_selected_cohort_indices_sha256
            ),
            waymax_scene_n=0,
        )


def test_compute_pilot_reseal_and_cross_run_transplant_are_rejected(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    origin = M6ResultStore.create(
        project,
        "pilot-origin",
        mode=COMPUTE_PILOT_MODE,
    )
    rows = _eligibility_rows(128, 10, mode=COMPUTE_PILOT_MODE)
    _write_eligibility(origin, rows)
    receipt = origin.eligibility_receipt
    assert receipt is not None
    selection = _waymax_selection(receipt, qualified_n=0)
    origin.write_waymax_qualification(selection)
    provenance = _verified_provenance(COMPUTE_PILOT_MODE)
    evidence = _pilot_execution_evidence(
        origin,
        selection,
        provenance,
    )
    original_report = evidence.pilot_report_binding_sha256
    assert original_report is not None
    object.__setattr__(
        evidence,
        "pilot_report_binding_sha256",
        "f" * 64,
    )
    try:
        with pytest.raises(ValueError, match="integrity binding"):
            evidence.revalidate_pilot()
    finally:
        object.__setattr__(
            evidence,
            "pilot_report_binding_sha256",
            original_report,
        )
    evidence.revalidate_pilot()
    with pytest.raises(TypeError, match="runner-issued"):
        dataclasses.replace(evidence)

    destination = M6ResultStore.create(
        project,
        "pilot-destination",
        mode=COMPUTE_PILOT_MODE,
    )
    _write_eligibility(destination, rows)
    transplanted_selection = copy.deepcopy(selection)
    assert transplanted_selection is not selection
    assert waymax_official.m6_waymax_selection_binding_sha256(
        transplanted_selection
    ) == waymax_official.m6_waymax_selection_binding_sha256(selection)
    destination.write_waymax_qualification(transplanted_selection)
    with pytest.raises(ValueError, match="integrity binding"):
        destination.write_compute_pilot_summary(evidence)
    assert (destination.run_path / TERMINAL_FAILURE_MARKER).is_file()


def test_compute_pilot_provenance_identity_transplant_is_rejected(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "pilot-provenance-transplant",
        mode=COMPUTE_PILOT_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=COMPUTE_PILOT_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection = _waymax_selection(receipt, qualified_n=0)
    store.write_waymax_qualification(selection)
    issued_provenance = _verified_provenance(COMPUTE_PILOT_MODE)
    evidence = _pilot_execution_evidence(
        store,
        selection,
        issued_provenance,
    )
    store.write_compute_pilot_summary(evidence)
    transplanted_provenance = _verified_provenance(COMPUTE_PILOT_MODE)
    assert transplanted_provenance is not issued_provenance
    with pytest.raises(ValueError, match="integrity binding"):
        store.write_typed_provenance(transplanted_provenance)
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()


@pytest.mark.parametrize(
    "field",
    (
        "total_wall_ms",
        "max_scene_ms",
        "decode_ms",
        "numpy_ms",
        "waymax_ms",
        "verification_ms",
    ),
)
def test_compute_pilot_requires_every_measured_duration_positive(
    tmp_path: Path,
    field: str,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        f"pilot-zero-{field.replace('_', '-')}",
        mode=COMPUTE_PILOT_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=COMPUTE_PILOT_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    row = _pilot_normalization_row()
    row[field] = 0
    with pytest.raises(M6ResultStoreIntegrityError):
        m6._normalize_compute_pilot((row,), receipt, waymax_scene_n=0)


def test_compute_pilot_uses_derived_subphase_ceiling_slack(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "pilot-rounding-boundary",
        mode=COMPUTE_PILOT_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=COMPUTE_PILOT_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    assert m6._m6_compute_pilot_rounding_overage_ms(
        numpy_scene_n=8,
        waymax_scene_n=0,
    ) == 10
    assert m6._m6_compute_pilot_rounding_overage_ms(
        numpy_scene_n=8,
        waymax_scene_n=8,
    ) == 18
    row = _pilot_normalization_row(
        total_wall_ms=990,
        max_scene_ms=250,
        decode_ms=250,
        numpy_ms=250,
        waymax_ms=250,
        verification_ms=250,
    )
    assert m6._normalize_compute_pilot(
        (row,), receipt, waymax_scene_n=0
    )[0][
        "total_wall_ms"
    ] == 990
    too_short = {**row, "total_wall_ms": 989}
    with pytest.raises(M6ResultStoreIntegrityError, match="exceed total"):
        m6._normalize_compute_pilot(
            (too_short,), receipt, waymax_scene_n=0
        )


def test_failed_compute_pilot_can_commit_but_never_terminalize(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "pilot-failed",
        mode=COMPUTE_PILOT_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=COMPUTE_PILOT_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection = _waymax_selection(receipt, qualified_n=0)
    store.write_waymax_qualification(selection)
    provenance = _verified_provenance(store.profile.mode)
    failed_summary = _pilot_summary_row(
        total_wall_ms=30 * 60 * 1_000 + 1,
    )
    evidence = _pilot_execution_evidence(
        store,
        selection,
        provenance,
        summary=failed_summary,
    )
    store.write_compute_pilot_summary(evidence)
    store.write_typed_provenance(provenance)
    store.commit()
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="failed compute pilot",
    ):
        m6._mint_m6_terminal_capability(
            store,
            _verifier_observation(store),
            _verified_provenance(store.profile.mode),
        )
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()
    assert not (store.run_path / TERMINAL_SUCCESS_MARKER).exists()


def test_forged_terminal_marker_cannot_promote_failed_compute_pilot(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "pilot-forged-success",
        mode=COMPUTE_PILOT_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=COMPUTE_PILOT_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection = _waymax_selection(receipt, qualified_n=0)
    store.write_waymax_qualification(selection)
    provenance = _verified_provenance(store.profile.mode)
    failed_summary = _pilot_summary_row(
        total_wall_ms=30 * 60 * 1_000 + 1,
    )
    evidence = _pilot_execution_evidence(
        store,
        selection,
        provenance,
        summary=failed_summary,
    )
    store.write_compute_pilot_summary(evidence)
    store.write_typed_provenance(provenance)
    store.commit()
    observed = _verifier_observation(store)
    manifest_sha256 = observed.manifest_sha256
    committed_sha256 = observed.committed_sha256
    m6._write_bytes_exclusive(
        store.run_path / TERMINAL_SUCCESS_MARKER,
        m6._canonical_json_bytes(
            {
                "committed_sha256": committed_sha256,
                "evidence_catalog_sha256": observed.evidence_catalog_sha256,
                "manifest_sha256": manifest_sha256,
                "mode": COMPUTE_PILOT_MODE,
                "observed_preflight_sha256": observed.canonical_sha256,
                "provenance_context_sha256": (
                    observed.provenance_context_sha256
                ),
                "schema_version": m6.M6_RESULT_STORE_SCHEMA_VERSION,
                "state": "TERMINAL_SUCCESS",
                "writer_capability_preimage": "00" * 32,
            }
        ),
        store.run_path,
    )
    with pytest.raises(M6ResultStoreIntegrityError, match="writer capability"):
        verify_m6_result_store(
            project,
            "pilot-forged-success",
            expected_mode=COMPUTE_PILOT_MODE,
        )


def test_registered_fingerprints_and_v1_are_not_caller_choices(
    tmp_path: Path,
) -> None:
    assert m6.M6_PRIMARY_INTERVENTION.version == "v1"
    assert m6.M6_SECONDARY_INTERVENTION.version == "v1"
    assert (
        m6.M6_PRIMARY_INTERVENTION_FINGERPRINT
        == m6.longitudinal_brake_pulse_spec(2.0).configuration_fingerprint
    )
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "forged-config", mode=DATA_FREE_MODE)
    with pytest.raises(ValueError, match="exact registered"):
        store.write_eligibility_ledger(
            _eligibility_rows(10, 10, mode=DATA_FREE_MODE),
            primary_intervention_fingerprint="0" * 64,
            secondary_intervention_fingerprint="1" * 64,
        )
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()


def test_data_free_roundtrip_has_all_fixed_observation_domains(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = _successful_data_free(project)
    verified = verify_m6_result_store(
        project,
        "data-free",
        allow_data_free=True,
        expected_mode=DATA_FREE_MODE,
    )
    expected = verified.receipt.expected_rows
    assert expected[m6.NEGATIVE_TIMING_OBSERVATIONS] == 100
    assert expected[m6.WAYMAX_SCENE_SCALARS] == 128
    assert expected[m6.WAYMAX_FIELD_COMPARISONS] == 640
    assert expected[m6.WAYMAX_NUMPY_COMPARISONS] == 128
    assert expected[m6.WAYMAX_DETERMINISM] == 64
    assert expected[m6.WAYMAX_ACCOUNTING] == 58
    assert (
        verified.read_dataset(PRIMARY_REPEAT_SCENE_SCALARS).num_rows
        == 120
    )
    assert verified.read_dataset(PRIMARY_REPEAT_MATRIX).num_rows == 12
    assert (store.run_path / TERMINAL_SUCCESS_MARKER).is_file()


def test_data_free_waymax_qualification_rejects_caller_values(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "bad-data-free-q", mode=DATA_FREE_MODE)
    _write_eligibility(
        store,
        _eligibility_rows(10, 10, mode=DATA_FREE_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    rows = _qualification_rows(receipt)
    rows[0]["rank_sha256"] = m6.m6_waymax_rank_sha256(0)
    with pytest.raises(TypeError, match="no caller selection"):
        store.write_waymax_qualification(rows)  # type: ignore[arg-type]


def test_data_free_waymax_comparison_writers_are_caller_free(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    for name, dataset in (
        ("field", m6.WAYMAX_FIELD_COMPARISONS),
        ("numpy", m6.WAYMAX_NUMPY_COMPARISONS),
    ):
        store = M6ResultStore.create(
            project,
            f"data-free-{name}-caller",
            mode=DATA_FREE_MODE,
        )
        eligibility = _eligibility_rows(10, 10, mode=DATA_FREE_MODE)
        _write_eligibility(store, eligibility)
        store.write_waymax_qualification()
        caller_rows = (
            m6.m6_data_free_waymax_field_comparison_rows()
            if name == "field"
            else m6.m6_data_free_waymax_numpy_comparison_rows(
                eligibility
            )
        )
        writer = (
            store.write_waymax_field_comparisons
            if name == "field"
            else store.write_waymax_numpy_comparisons
        )
        with pytest.raises(TypeError, match="no caller evidence"):
            writer(caller_rows)
        assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()

        accepted = M6ResultStore.create(
            project,
            f"data-free-{name}-internal",
            mode=DATA_FREE_MODE,
        )
        _write_eligibility(accepted, eligibility)
        accepted.write_waymax_qualification()
        writer = (
            accepted.write_waymax_field_comparisons
            if name == "field"
            else accepted.write_waymax_numpy_comparisons
        )
        writer()
        stored = accepted._read_dataset_rows(dataset)
        expected = (
            m6.m6_data_free_waymax_field_comparison_rows()
            if name == "field"
            else m6.m6_data_free_waymax_numpy_comparison_rows(
                eligibility
            )
        )
        assert stored == expected


def test_data_free_numpy_rows_bind_actual_eligibility_projection(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)

    def prepared(name: str, eligible_n: int) -> M6ResultStore:
        store = M6ResultStore.create(
            project,
            name,
            mode=DATA_FREE_MODE,
        )
        _write_eligibility(
            store,
            _eligibility_rows(
                10,
                eligible_n,
                mode=DATA_FREE_MODE,
            ),
        )
        store.write_waymax_qualification()
        store.write_waymax_numpy_comparisons()
        return store

    first = prepared("numpy-binding-a", 10)
    second = prepared("numpy-binding-b", 10)
    first_rows = first._read_dataset_rows(m6.WAYMAX_NUMPY_COMPARISONS)
    second_rows = second._read_dataset_rows(m6.WAYMAX_NUMPY_COMPARISONS)
    first_digest = {
        row["stored_eligibility_rows_sha256"] for row in first_rows
    }
    second_digest = {
        row["stored_eligibility_rows_sha256"] for row in second_rows
    }
    assert len(first_digest) == len(second_digest) == 1
    assert first_digest == second_digest

    other_scope_digest = m6.m6_stored_eligibility_rows_sha256(
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE)
    )
    assert first_digest != {other_scope_digest}
    transplanted = [dict(row) for row in first_rows]
    for row in transplanted:
        row["stored_eligibility_rows_sha256"] = other_scope_digest

    second_receipt = second.eligibility_receipt
    assert second_receipt is not None
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="eligibility projection",
    ):
        m6._normalize_waymax_numpy_comparisons_from_qualification(
            transplanted,
            second_receipt,
            second._read_dataset_rows(m6.ELIGIBILITY_LEDGER),
            second._read_dataset_rows(m6.WAYMAX_QUALIFICATION),
            second._require_waymax_selection_receipt(),
            expected_numpy_eligibility_sha256=(
                m6.M6_DATA_FREE_WAYMAX_NUMPY_ELIGIBILITY_LEDGER_SHA256
            ),
        )

@pytest.mark.parametrize(
    "mutation",
    ["rank", "selection_relabel", "binding_reuse"],
)

def test_typed_waymax_selection_relabels_fail_before_seal(
    tmp_path: Path,
    mutation: str,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        f"bad-qualification-{mutation}",
        mode=OFFICIAL_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection = _waymax_selection(receipt, qualified_n=10)
    if mutation == "rank":
        object.__setattr__(selection.members[0], "rank_sha256", "0" * 64)
    elif mutation == "selection_relabel":
        object.__setattr__(
            selection,
            "members",
            (selection.members[1], selection.members[0], *selection.members[2:]),
        )
    else:
        object.__setattr__(
            selection.members[1],
            "qualification_binding_sha256",
            selection.members[0].qualification_binding_sha256,
        )
    with pytest.raises((ValueError, M6ResultStoreIntegrityError)):
        store.write_waymax_qualification(selection)


@pytest.mark.parametrize(
    "attack",
    ["mapping", "parsed", "wrong_domain", "rebound_selection"],
)
def test_waymax_scalar_writer_rejects_nonissued_or_rebound_evidence(
    tmp_path: Path,
    attack: str,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        f"scalar-boundary-{attack}",
        mode=OFFICIAL_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    official_evidence = _official_waymax_evidence(10)
    selection = official_evidence.selection
    store.write_waymax_qualification(selection)
    issued = official_evidence.scene_scalars._issued
    original_rows = None
    evidence: object
    if attack == "mapping":
        evidence = [row.to_store_dict() for row in issued.rows]
    elif attack == "parsed":
        evidence = parse_m6_waymax_scene_scalar_table(issued.rows)
        assert isinstance(evidence, M6WaymaxParsedScalarTable)
        assert evidence.promotable is False
    else:
        original_rows = issued.rows
        replacement_rows = list(original_rows)
        victim = replacement_rows[0]
        replacement_rows[0] = dataclasses.replace(
            victim,
            **(
                {"primary_domain_sha256": "0" * 64}
                if attack == "wrong_domain"
                else {"selection_binding_sha256": "1" * 64}
            ),
            scalar_binding_sha256=None,
        )
        object.__setattr__(issued, "rows", tuple(replacement_rows))
        evidence = official_evidence
    try:
        if attack in {"mapping", "parsed"}:
            with pytest.raises(TypeError, match="shared runner-issued"):
                store.write_waymax_scene_scalars(evidence)  # type: ignore[arg-type]
        else:
            with pytest.raises(ValueError):
                store.write_waymax_scene_scalars(evidence)  # type: ignore[arg-type]
    finally:
        if original_rows is not None:
            object.__setattr__(issued, "rows", original_rows)
            official_evidence.revalidate()


def test_waymax_tolerance_contradiction_and_short_denominator_rejected(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    for mutation in ("normalized", "denominator"):
        store = M6ResultStore.create(
            project,
            f"field-{mutation}",
            mode=OFFICIAL_MODE,
        )
        _write_eligibility(
            store,
            _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
        )
        receipt = store.eligibility_receipt
        assert receipt is not None
        _selection, qualification = _seal_waymax_selection(
            store,
            receipt,
            qualified_n=10,
        )
        rows = _waymax_field_rows(qualification)
        target = next(
            row
            for row in rows
            if row["status"] == "passed"
            and row["comparison_kind"] == "tolerance"
        )
        if mutation == "normalized":
            target["max_normalized_error"] = 1.01
        else:
            target["denominator"] = 19
        with pytest.raises(TypeError, match="shared runner-issued"):
            store.write_waymax_field_comparisons(rows)



def test_official_numpy_writer_rejects_raw_rows(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "numpy-caller-rows",
        mode=OFFICIAL_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection, _domain = _waymax_selection_and_domain(
        receipt,
        qualified_n=0,
    )
    store.write_waymax_qualification(selection)

    with pytest.raises(TypeError, match="shared runner-issued"):
        store.write_waymax_numpy_comparisons(())
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()

def test_official_field_writer_rejects_nonpromotable_typed_table(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "field-test-authority",
        mode=OFFICIAL_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection, domain = _waymax_selection_and_domain(
        receipt,
        qualified_n=8,
    )
    store.write_waymax_qualification(selection)
    comparisons = {}
    for position in range(len(selection.members)):
        for bundle in m6.M6_WAYMAX_BUNDLES:
            for condition in m6.M6_WAYMAX_CONDITIONS:
                for field_name in m6.M6_WAYMAX_COMPARISON_FIELDS:
                    comparisons[
                        (position, bundle, condition, field_name)
                    ] = waymax_official._Comparison(
                        denominator=20,
                        maximum_absolute_error=0.0,
                        maximum_normalized_error=(
                            None
                            if field_name in m6.M6_WAYMAX_EXACT_FIELDS
                            else 0.0
                        ),
                        tolerance_failures=0,
                        binary_mismatches=0,
                    )
    evidence = waymax_official._issue_field_table(
        comparisons,
        selection=selection,
        promotable=False,
        primary_domain=domain,
    )
    assert evidence.promotable is False

    with pytest.raises(
        TypeError,
        match="shared runner-issued",
    ):
        store.write_waymax_field_comparisons(evidence)
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()


def test_official_field_writer_accepts_only_bound_unsupported_table(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "field-unsupported-issued",
        mode=OFFICIAL_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    evidence = _official_waymax_evidence(0)
    assert evidence.production_authoritative is False
    store.write_waymax_qualification(evidence.selection)

    with _admit_test_waymax_evidence(store, evidence):
        store.write_waymax_field_comparisons(evidence)

    rows = store._read_dataset_rows(m6.WAYMAX_FIELD_COMPARISONS)
    assert len(rows) == 640
    assert all(row["status"] == "not_applicable" for row in rows)


def test_official_store_rejects_test_only_unsupported_bundle(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "field-unsupported-test-authority",
        mode=OFFICIAL_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
    )
    _collector, source, numpy_evidence = _official_source(
        10,
        full_numpy=True,
        waymax_eligible_n=0,
    )
    assert numpy_evidence is not None
    runtime = _OfficialMockRuntime()
    evidence = waymax_official.run_m6_waymax_official(
        source,
        waymax_official.build_m6_waymax_test_execution_authority(
            runtime.executors()
        ),
        numpy_evidence,
    )
    assert not evidence.supported
    assert evidence.production_authoritative is False
    store.write_waymax_qualification(evidence.selection)

    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="production-authoritative",
    ):
        store.write_waymax_field_comparisons(evidence)


@pytest.mark.parametrize("supported", (False, True))
def test_official_shared_bundle_terminal_roundtrip_and_numpy_privacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supported: bool,
) -> None:
    project = _project(tmp_path)
    qualified_n = 10 if supported else 0
    evidence = _official_waymax_evidence(qualified_n)
    store = M6ResultStore.create(
        project,
        f"official-roundtrip-{str(supported).lower()}",
        mode=OFFICIAL_MODE,
    )
    store = _write_complete(
        store,
        qualified_n=qualified_n,
        official_evidence=evidence,
    )
    store.commit()
    committed = m6.verify_committed_m6_result_store(
        project,
        store.run_name,
        expected_mode=OFFICIAL_MODE,
    )
    assert committed.waymax_selection_receipt.selection_supported is supported
    _terminalize_verified(store)
    terminal = verify_m6_result_store(
        project,
        store.run_name,
        expected_mode=OFFICIAL_MODE,
    )
    execution = terminal.read_dataset(m6.EXECUTION_SUMMARY).to_pylist()[0]
    assert execution["waymax_gate_status"] == (
        "accepted" if supported else "unsupported"
    )

    baseline = reconstruct_sanitized_m6_aggregate(
        project,
        store.run_name,
    )
    claim = baseline.to_dict()["claim_and_limitations"]
    expected_claim_status = execution["real_reactivity_claim_status"]
    assert claim == m6._promoted_claim_and_limitations(
        expected_claim_status
    )
    stored_claim = json.loads(
        (terminal.run_path / m6.CLAIM_LIMITATIONS_PATH).read_text(
            encoding="ascii"
        )
    )
    assert stored_claim == m6._claim_limitations_payload(
        OFFICIAL_MODE,
        expected_claim_status,
    )
    if expected_claim_status == "blocked":
        assert m6.M6_ACCEPTED_BOUNDED_CLAIM not in (
            terminal.run_path / m6.CLAIM_LIMITATIONS_PATH
        ).read_text(encoding="ascii")

    if supported:
        class NoLocalNumpyRead:
            def __init__(self, verified):
                self._verified = verified
                self.profile = verified.profile
                self.receipt = verified.receipt
                self.run_path = verified.run_path

            def read_dataset(self, name: str):
                if name == m6.WAYMAX_NUMPY_COMPARISONS:
                    raise AssertionError(
                        "local-only NumPy evidence reached publication"
                    )
                return self._verified.read_dataset(name)

        proxy = NoLocalNumpyRead(terminal)
        monkeypatch.setattr(
            m6,
            "verify_m6_result_store",
            lambda *_args, **_kwargs: proxy,
        )
        without_numpy = reconstruct_sanitized_m6_aggregate(
            project,
            store.run_name,
        )
        assert without_numpy.canonical_bytes == baseline.canonical_bytes


@pytest.mark.parametrize(
    "mode",
    (ELIGIBILITY_ONLY_MODE, COMPUTE_PILOT_MODE),
)
def test_non_result_modes_cannot_write_waymax_determinism(
    tmp_path: Path,
    mode: str,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, f"det-unavailable-{mode}", mode=mode)
    with pytest.raises(M6ResultStoreStateError, match="complete-result"):
        store.write_waymax_determinism()
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()


def test_official_waymax_determinism_accepts_factory_issued_live_table(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "det-live", mode=OFFICIAL_MODE)
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection, domain = _waymax_selection_and_domain(receipt, qualified_n=10)
    store.write_waymax_qualification(selection)
    issued = build_m6_waymax_live_determinism_table(
        _live_determinism_executions(selection),
        selection=selection,
        primary_domain=domain,
    )

    with pytest.raises(TypeError, match="shared runner-issued"):
        store.write_waymax_determinism(issued)


def test_official_waymax_determinism_accepts_bound_unsupported_placeholder(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "det-unsupported",
        mode=OFFICIAL_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    evidence = _official_waymax_evidence(0)
    store.write_waymax_qualification(evidence.selection)

    with _admit_test_waymax_evidence(store, evidence):
        store.write_waymax_determinism(evidence)

    assert all(
        row["status"] == "not_applicable"
        for row in store._read_dataset_rows(m6.WAYMAX_DETERMINISM)
    )


@pytest.mark.parametrize("attack", ("none", "raw_rows", "wrong_kind"))
def test_official_waymax_determinism_rejects_nonissued_or_wrong_kind(
    tmp_path: Path,
    attack: str,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        f"det-caller-{attack}",
        mode=OFFICIAL_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection, domain = _waymax_selection_and_domain(receipt, qualified_n=10)
    store.write_waymax_qualification(selection)
    live = build_m6_waymax_live_determinism_table(
        _live_determinism_executions(selection),
        selection=selection,
        primary_domain=domain,
    )
    if attack == "none":
        evidence: object | None = None
    elif attack == "raw_rows":
        evidence = live.to_store_rows()
    else:
        unsupported_selection, unsupported_domain = (
            _waymax_selection_and_domain(receipt, qualified_n=0)
        )
        evidence = build_m6_waymax_unsupported_determinism_table(
            selection=unsupported_selection,
            primary_domain=unsupported_domain,
        )

    with pytest.raises(TypeError, match="shared runner-issued"):
        store.write_waymax_determinism(evidence)


def test_official_waymax_determinism_rejects_other_selection_binding(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "det-other-selection",
        mode=OFFICIAL_MODE,
    )
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    selection, _domain = _waymax_selection_and_domain(receipt, qualified_n=10)
    other_selection, other_domain = _waymax_selection_and_domain(
        receipt,
        qualified_n=8,
    )
    store.write_waymax_qualification(selection)
    other = build_m6_waymax_live_determinism_table(
        _live_determinism_executions(other_selection),
        selection=other_selection,
        primary_domain=other_domain,
    )

    with pytest.raises(TypeError, match="shared runner-issued"):
        store.write_waymax_determinism(other)


def test_data_free_determinism_accepts_only_caller_free_issued_placeholder(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    rejected = M6ResultStore.create(
        project,
        "data-free-det-caller",
        mode=DATA_FREE_MODE,
    )
    _write_eligibility(
        rejected,
        _eligibility_rows(10, 10, mode=DATA_FREE_MODE),
    )
    rejected.write_waymax_qualification()
    with pytest.raises(TypeError, match="accepts no caller evidence"):
        rejected.write_waymax_determinism(
            m6.m6_data_free_waymax_determinism_rows()
        )

    accepted = M6ResultStore.create(
        project,
        "data-free-det-issued",
        mode=DATA_FREE_MODE,
    )
    _write_eligibility(
        accepted,
        _eligibility_rows(10, 10, mode=DATA_FREE_MODE),
    )
    accepted.write_waymax_qualification()
    accepted.write_waymax_determinism()
    assert accepted._read_dataset_rows(m6.WAYMAX_DETERMINISM) == (
        m6.m6_data_free_waymax_determinism_rows()
    )


def test_negative_timing_gates_are_derived_from_exact_observations(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "negative-domain", mode=DATA_FREE_MODE)
    _write_eligibility(
        store,
        _eligibility_rows(10, 10, mode=DATA_FREE_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    rows = _negative_observation_rows(receipt)
    assert len(rows) == 100
    rows.pop()
    with pytest.raises(M6ResultStoreIntegrityError, match="incomplete"):
        store.write_negative_timing_observations(rows)


def test_primary_repeat_is_sealed_rows_not_caller_digest(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "repeat-copy", mode=DATA_FREE_MODE)
    _write_eligibility(
        store,
        _eligibility_rows(10, 10, mode=DATA_FREE_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    primary = _scene_rows(
        receipt.eligible_cohort_indices,
        m6.M6_PRIMARY_INTERVENTION_FINGERPRINT,
    )
    store.write_primary_scene_scalars(primary)
    store.write_primary_matrix()
    repeated = [dict(row) for row in primary]
    repeated[0]["value"] = 1.0
    store.write_primary_repeat_scene_scalars(repeated)
    store.write_primary_repeat_matrix()
    store.write_waymax_qualification()
    store.write_waymax_determinism()
    with pytest.raises(ValueError, match="disagrees"):
        store.write_determinism_receipt()


def test_execution_status_and_counts_are_not_caller_aggregates(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "no-caller-summary", mode=DATA_FREE_MODE)
    with pytest.raises(TypeError):
        store.write_execution_summary(  # type: ignore[call-arg]
            {
                "release_gate_status": "accepted",
                "deterministic_repeat_status": "passed",
            }
        )


def test_review_rows_allow_explicit_rejects_and_nonblocking_p3() -> None:
    receipt = _official_claim_receipt()
    digest = _sha("sealed-review-precursor")
    mechanical = _sha("mechanical-verification")
    accepted_rows = [
        {
            **row,
            "p3_count": 2,
            "evidence_catalog_sha256": digest,
            "mechanical_verification_sha256": mechanical,
        }
        for row in _review_rows()
    ]
    accepted = m6._normalize_review_decisions(
        accepted_rows,
        receipt,
        expected_evidence_catalog_sha256=digest,
        expected_approved_git_commit="a" * 40,
        expected_mechanical_verification_sha256=mechanical,
    )
    assert all(
        row["decision"] == "accept"
        and row["p1_count"] == row["p2_count"] == 0
        and row["p3_count"] == 2
        for row in accepted
    )

    rejected_rows = [dict(row) for row in accepted_rows]
    rejected_rows[0]["decision"] = "reject"
    rejected_rows[0]["p1_count"] = 1
    rejected = m6._normalize_review_decisions(rejected_rows, receipt)
    assert rejected[0]["decision"] == "reject"
    assert rejected[0]["p1_count"] == 1

    inconsistent = [dict(row) for row in accepted_rows]
    inconsistent[0]["p2_count"] = 1
    contradictory = m6._normalize_review_decisions(inconsistent, receipt)
    assert contradictory[0]["decision"] == "accept"
    assert contradictory[0]["p2_count"] == 1
    inconsistent[0]["decision"] = "reject"
    inconsistent[0]["p2_count"] = 0
    explicit_reject = m6._normalize_review_decisions(inconsistent, receipt)
    assert explicit_reject[0]["decision"] == "reject"
    assert explicit_reject[0]["p1_count"] == explicit_reject[0]["p2_count"] == 0
    with pytest.raises(M6ResultStoreIntegrityError, match="sealed precursor"):
        m6._normalize_review_decisions(
            accepted_rows,
            receipt,
            expected_evidence_catalog_sha256=_sha("different"),
        )
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="mechanical verification",
    ):
        m6._normalize_review_decisions(
            accepted_rows,
            receipt,
            expected_mechanical_verification_sha256=_sha("different"),
        )


def test_explicit_review_issuance_is_separate_from_mechanical_facts() -> None:
    verification = m6.M6MechanicalVerificationReceipt(
        mode=OFFICIAL_MODE,
        result_path="outputs/m6/explicit-review",
        approved_git_commit="a" * 40,
        evidence_catalog_sha256=_sha("precursor"),
        review_challenge=_sha("fresh-post-precursor-challenge"),
        _factory_sentinel=m6._MECHANICAL_VERIFICATION_SENTINEL,
    )
    accepted = m6.issue_m6_review_decision(
        verification,
        role="architecture",
        decision="accept",
        p1_count=0,
        p2_count=0,
        p3_count=3,
    )
    rejected = m6.issue_m6_review_decision(
        verification,
        role="methods_statistics",
        decision="reject",
        p1_count=0,
        p2_count=1,
        p3_count=0,
    )

    assert accepted.decision == "accept" and accepted.p3_count == 3
    assert rejected.decision == "reject" and rejected.p2_count == 1
    assert accepted.mechanical_verification_sha256 == (
        verification.verification_sha256
    )
    contradictory = m6.issue_m6_review_decision(
        verification,
        role="privacy_claim",
        decision="accept",
        p1_count=1,
        p2_count=0,
        p3_count=0,
    )
    assert contradictory.decision == "accept"
    assert contradictory.p1_count == 1
    maximum = m6.issue_m6_review_decision(
        verification,
        role="privacy_claim",
        decision="accept",
        p1_count=0,
        p2_count=0,
        p3_count=m6.M6_REVIEW_COUNT_MAX,
    )
    table = pa.Table.from_pylist(
        [maximum.to_store_row()],
        schema=m6.REVIEW_DECISIONS_SCHEMA,
    )
    assert table["p3_count"][0].as_py() == m6.M6_REVIEW_COUNT_MAX
    with pytest.raises(ValueError, match="int32"):
        m6.issue_m6_review_decision(
            verification,
            role="privacy_claim",
            decision="accept",
            p1_count=0,
            p2_count=0,
            p3_count=m6.M6_REVIEW_COUNT_MAX + 1,
        )


def test_official_review_decisions_require_adopted_awaiting_store(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "pending-review-bypass", mode=OFFICIAL_MODE)

    with pytest.raises(
        M6ResultStoreStateError,
        match="adopted AWAITING_REVIEW",
    ):
        store.write_review_decisions(object(), ())  # type: ignore[arg-type]

    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()
    assert not (store.run_path / m6._DATASET_PATHS[m6.REVIEW_DECISIONS]).exists()


def test_terminal_status_failure_invalidates_existing_success_marker(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "ambiguous-status", mode=OFFICIAL_MODE)
    success_path = store.run_path / TERMINAL_SUCCESS_MARKER
    m6._write_bytes_exclusive(
        success_path,
        b"terminal-success-fixture\n",
        store.run_path,
    )

    failure_path = store._invalidate_terminal_status_failure(
        "terminal_capture_failed"
    )
    assert success_path.is_file()
    assert failure_path.is_file()
    assert store._invalidate_terminal_status_failure(
        "terminal_capture_failed"
    ) == failure_path
    with pytest.raises(M6ResultStoreIntegrityError, match="mutually exclusive"):
        m6.verify_m6_result_store(
            project,
            "ambiguous-status",
            expected_mode=OFFICIAL_MODE,
        )


def test_official_store_pauses_and_resumes_through_awaiting_review(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "awaiting-review", mode=OFFICIAL_MODE)
    _write_complete(store, seal_reviews=False)
    requested = store.write_mechanical_verification_receipt()
    sealed = store.seal_awaiting_review(
        fresh_worker_peak_rss_bytes=4096,
    )
    assert sealed.to_dict() == requested.to_dict()
    assert store._phase == "awaiting_review"
    assert (store.run_path / m6.AWAITING_REVIEW_MARKER).is_file()
    assert not (store.run_path / m6._DATASET_PATHS[m6.REVIEW_DECISIONS]).exists()
    assert not (store.run_path / m6._DATASET_PATHS[m6.EXECUTION_SUMMARY]).exists()
    assert not (store.run_path / COMMITTED_MARKER).exists()

    resumed = M6ResultStore.adopt_awaiting_review(
        project,
        "awaiting-review",
    )
    decisions = tuple(
        m6.issue_m6_review_decision(
            sealed,
            role=role,
            decision="accept",
            p1_count=0,
            p2_count=0,
            p3_count=1,
        )
        for role in m6.M6_REVIEW_ROLE_DOMAIN
    )
    resumed.write_review_decisions(sealed, decisions)
    resumed.write_execution_summary(
        fresh_worker_peak_rss_bytes=(
            resumed.awaiting_review_fresh_worker_peak_rss_bytes
        )
    )
    summary = resumed._read_dataset_rows(m6.EXECUTION_SUMMARY)
    assert summary[0]["release_gate_status"] == "accepted"
    resumed.commit()
    verified = m6.verify_committed_m6_result_store(
        project,
        "awaiting-review",
        expected_mode=OFFICIAL_MODE,
    )
    assert verified.read_dataset(m6.REVIEW_DECISIONS).num_rows == 3

    awaiting_path = resumed.run_path / m6.AWAITING_REVIEW_MARKER
    original = awaiting_path.read_bytes()
    tampered = json.loads(original)
    tampered["fresh_worker_peak_rss_bytes"] += 1
    awaiting_path.write_bytes(m6._canonical_json_bytes(tampered))
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="AWAITING_REVIEW",
    ):
        m6.verify_committed_m6_result_store(
            project,
            "awaiting-review",
            expected_mode=OFFICIAL_MODE,
        )
    awaiting_path.write_bytes(original)
    m6.verify_committed_m6_result_store(
        project,
        "awaiting-review",
        expected_mode=OFFICIAL_MODE,
    )
    awaiting_path.unlink()
    with pytest.raises(M6ResultStoreIntegrityError, match="AWAITING_REVIEW"):
        m6.verify_committed_m6_result_store(
            project,
            "awaiting-review",
            expected_mode=OFFICIAL_MODE,
        )


def test_blocking_review_counts_close_awaiting_review_without_commit(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "review-reject", mode=OFFICIAL_MODE)
    _write_complete(store, seal_reviews=False)
    verification = store.write_mechanical_verification_receipt()
    store.seal_awaiting_review(fresh_worker_peak_rss_bytes=4096)
    resumed = M6ResultStore.adopt_awaiting_review(project, "review-reject")
    decisions = tuple(
        m6.issue_m6_review_decision(
            verification,
            role=role,
            decision="accept",
            p1_count=1 if role == "architecture" else 0,
            p2_count=0,
            p3_count=0,
        )
        for role in m6.M6_REVIEW_ROLE_DOMAIN
    )
    resumed.write_review_decisions(verification, decisions)
    resumed.write_execution_summary(
        fresh_worker_peak_rss_bytes=4096,
    )
    assert resumed._read_dataset_rows(m6.EXECUTION_SUMMARY)[0][
        "release_gate_status"
    ] == "rejected"
    resumed.fail("review_rejected")
    assert (resumed.run_path / TERMINAL_FAILURE_MARKER).is_file()
    assert not (resumed.run_path / COMMITTED_MARKER).exists()
    rejected = m6.verify_rejected_m6_review_store(project, "review-reject")
    assert rejected.execution_summary["release_gate_status"] == "rejected"
    assert rejected.review_decisions[0]["decision"] == "accept"
    assert rejected.review_decisions[0]["p1_count"] == 1
    decision_path = resumed.run_path / m6._DATASET_PATHS[m6.REVIEW_DECISIONS]
    original_decisions = decision_path.read_bytes()
    decision_path.write_bytes(original_decisions + b"tamper")
    with pytest.raises(M6ResultStoreIntegrityError):
        m6.verify_rejected_m6_review_store(project, "review-reject")
    decision_path.write_bytes(original_decisions)
    (resumed.run_path / m6.AWAITING_REVIEW_MARKER).unlink()
    with pytest.raises(M6ResultStoreIntegrityError):
        m6.verify_rejected_m6_review_store(project, "review-reject")


def test_data_free_records_review_absence_and_nonpromotable_release(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    successful = _successful_data_free(project, "review-absent")
    review_rows = successful._read_dataset_rows(m6.REVIEW_DECISIONS)
    execution = successful._read_dataset_rows(m6.EXECUTION_SUMMARY)

    assert review_rows == ()
    assert execution[0]["review_decision_rows"] == 0
    assert execution[0]["release_gate_status"] == "nonpromotable"
    assert not (successful.run_path / m6.REVIEW_REQUEST_PATH).exists()



def test_official_requires_positive_timings_and_rss(tmp_path: Path) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "zero-timing", mode=OFFICIAL_MODE)
    _write_eligibility(
        store,
        _eligibility_rows(128, 10, mode=OFFICIAL_MODE),
    )
    receipt = store.eligibility_receipt
    assert receipt is not None
    rows = _stage_rows(OFFICIAL_MODE)
    rows[0]["duration_ms"] = 0
    with pytest.raises(M6ResultStoreIntegrityError, match="positive"):
        store.write_stage_timings(rows)


def test_noop_callback_cannot_authorize_terminal_success(tmp_path: Path) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "no-callback", mode=ELIGIBILITY_ONLY_MODE)
    with pytest.raises(TypeError):
        store.mark_terminal_success(  # type: ignore[call-arg]
            pre_success_check=lambda: None
        )
    with pytest.raises(M6ResultStoreStateError, match="commit"):
        store.finalize()


def test_preflight_and_terminal_capability_constructors_are_private(
    tmp_path: Path,
) -> None:
    with pytest.raises(M6ResultStoreStateError, match="verifier-minted"):
        m6.M6ObservedPreflightResult(
            mode=OFFICIAL_MODE,
            result_path="outputs/m6/x",
            manifest_sha256="0" * 64,
            committed_sha256="1" * 64,
            evidence_catalog_sha256="2" * 64,
            provenance_context_sha256="3" * 64,
            checks=_ALL_PREFLIGHT_CHECKS,
            _factory_sentinel=object(),
        )
    with pytest.raises(M6ResultStoreStateError, match="verifier hook"):
        m6.M6TerminalCapability(
            mode=OFFICIAL_MODE,
            result_path="outputs/m6/x",
            manifest_sha256="0" * 64,
            committed_sha256="1" * 64,
            evidence_catalog_sha256="2" * 64,
            provenance_context_sha256="3" * 64,
            observed_preflight_sha256="4" * 64,
            nonce=b"x" * 32,
            _factory_sentinel=object(),
        )


def test_asserted_preflight_and_terminal_mint_paths_are_disabled(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(
        project,
        "disabled-preflight",
        mode=OFFICIAL_MODE,
    )
    with pytest.raises(M6ResultStoreStateError, match="asserted preflight"):
        m6._make_m6_observed_preflight_result(
            store,
            checks=_ALL_PREFLIGHT_CHECKS,
            evidence_catalog_sha256=_sha("caller"),
        )
    with pytest.raises(M6ResultStoreStateError, match="committed"):
        m6._mint_m6_terminal_capability(
            store,
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
        )


def test_terminal_capability_is_bound_to_one_store(tmp_path: Path) -> None:
    project = _project(tmp_path)
    left = _committed_eligibility_store(project, "cap-left")
    right = _committed_eligibility_store(project, "cap-right")
    capability = m6._mint_m6_terminal_capability(
        left,
        _verifier_observation(left),
        _verified_provenance(left.profile.mode),
    )

    with pytest.raises(M6ResultStoreStateError, match="one-use"):
        right.mark_terminal_success(capability=capability)
    assert (right.run_path / TERMINAL_FAILURE_MARKER).is_file()
    assert not (right.run_path / TERMINAL_SUCCESS_MARKER).exists()

    left.mark_terminal_success(capability=capability)
    assert (left.run_path / TERMINAL_SUCCESS_MARKER).is_file()
    assert not (left.run_path / TERMINAL_FAILURE_MARKER).exists()


@pytest.mark.parametrize("mutation", ("copy", "nonce"))
def test_terminal_capability_rejects_copy_or_nonce_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    project = _project(tmp_path)
    store = _committed_eligibility_store(project, f"cap-{mutation}")
    capability = m6._mint_m6_terminal_capability(
        store,
        _verifier_observation(store),
        _verified_provenance(store.profile.mode),
    )
    if mutation == "copy":
        presented = dataclasses.replace(capability)
    else:
        object.__setattr__(capability, "nonce", b"z" * 32)
        presented = capability

    with pytest.raises(M6ResultStoreStateError, match="one-use"):
        store.mark_terminal_success(capability=presented)
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()
    assert not (store.run_path / TERMINAL_SUCCESS_MARKER).exists()


def test_terminal_capability_cannot_be_replayed_after_success(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = _committed_eligibility_store(project, "cap-replay")
    capability = m6._mint_m6_terminal_capability(
        store,
        _verifier_observation(store),
        _verified_provenance(store.profile.mode),
    )
    store.mark_terminal_success(capability=capability)

    with pytest.raises(M6ResultStoreStateError, match="committed capability"):
        store.mark_terminal_success(capability=capability)
    assert (store.run_path / TERMINAL_SUCCESS_MARKER).is_file()
    assert not (store.run_path / TERMINAL_FAILURE_MARKER).exists()


def test_post_mint_artifact_tamper_poisoned_before_terminal_success(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = _committed_eligibility_store(project, "cap-stale")
    capability = m6._mint_m6_terminal_capability(
        store,
        _verifier_observation(store),
        _verified_provenance(store.profile.mode),
    )
    path = store.run_path / "eligibility-ledger.parquet"
    with path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0xFF]))
        handle.flush()
        os.fsync(handle.fileno())

    with pytest.raises(M6ResultStoreIntegrityError, match="size/SHA-256"):
        store.mark_terminal_success(capability=capability)
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()
    assert not (store.run_path / TERMINAL_SUCCESS_MARKER).exists()


@pytest.mark.parametrize("timing", ("before_original", "after_original"))
def test_terminal_writer_boundary_mutation_never_returns_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timing: str,
) -> None:
    project = _project(tmp_path)
    store = _committed_eligibility_store(
        project,
        f"terminal-boundary-{timing}",
    )
    capability = m6._mint_m6_terminal_capability(
        store,
        _verifier_observation(store),
        _verified_provenance(store.profile.mode),
    )
    artifact = store.run_path / "eligibility-ledger.parquet"
    original_writer = m6._write_terminal_success_final

    def mutate_artifact() -> None:
        with artifact.open("r+b") as handle:
            first = handle.read(1)
            handle.seek(0)
            handle.write(bytes([first[0] ^ 0xFF]))
            handle.flush()
            os.fsync(handle.fileno())

    def boundary_writer(
        path: Path,
        payload: bytes,
        run_path: Path,
        *,
        revalidate,
    ) -> None:
        if timing == "before_original":
            mutate_artifact()
        original_writer(
            path,
            payload,
            run_path,
            revalidate=revalidate,
        )
        if timing == "after_original":
            mutate_artifact()

    monkeypatch.setattr(
        m6,
        "_write_terminal_success_final",
        boundary_writer,
    )
    with pytest.raises(M6ResultStoreIntegrityError, match="size/SHA-256"):
        store.mark_terminal_success(capability=capability)

    if timing == "before_original":
        assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()
        assert not (store.run_path / TERMINAL_SUCCESS_MARKER).exists()
    else:
        assert store._phase == "ambiguous"
        assert (store.run_path / TERMINAL_SUCCESS_MARKER).is_file()
        assert not (store.run_path / TERMINAL_FAILURE_MARKER).exists()


def test_terminal_mint_rejects_fresh_provenance_fact_drift(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = _committed_eligibility_store(project, "provenance-drift")
    row = _provenance(ELIGIBILITY_ONLY_MODE)
    row["executable_source_sha256"] = "9" * 64
    drifted = _verified_provenance(ELIGIBILITY_ONLY_MODE, row=row)

    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="differs from final verified facts",
    ):
        m6._mint_m6_terminal_capability(
            store,
            _verifier_observation(store),
            drifted,
        )
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()
    assert not (store.run_path / TERMINAL_SUCCESS_MARKER).exists()


@pytest.mark.parametrize("mutation", ("catalog", "preflight"))
def test_terminal_verifier_rejects_forged_catalog_or_preflight_binding(
    tmp_path: Path,
    mutation: str,
) -> None:
    project = _project(tmp_path)
    store = _committed_eligibility_store(project, f"terminal-{mutation}")
    _terminalize_verified(store)
    marker_path = store.run_path / TERMINAL_SUCCESS_MARKER
    marker = json.loads(marker_path.read_text(encoding="ascii"))
    if mutation == "catalog":
        marker["evidence_catalog_sha256"] = "0" * 64
        marker["observed_preflight_sha256"] = (
            m6._expected_m6_observed_preflight(
                mode=ELIGIBILITY_ONLY_MODE,
                result_path=store.project_relative_path.as_posix(),
                manifest_sha256=marker["manifest_sha256"],
                committed_sha256=marker["committed_sha256"],
                evidence_catalog_sha256=marker["evidence_catalog_sha256"],
                provenance_context_sha256=marker[
                    "provenance_context_sha256"
                ],
            ).canonical_sha256
        )
    else:
        marker["observed_preflight_sha256"] = "0" * 64
    marker_path.write_bytes(m6._canonical_json_bytes(marker))

    expected = (
        "catalog is not mechanically derived"
        if mutation == "catalog"
        else "observed-preflight binding drifted"
    )
    with pytest.raises(M6ResultStoreIntegrityError, match=expected):
        verify_m6_result_store(
            project,
            f"terminal-{mutation}",
            expected_mode=ELIGIBILITY_ONLY_MODE,
        )


@pytest.mark.parametrize(
    (
        "waymax_executed",
        "idm_responder_n",
        "expected_waymax_status",
        "expected_claim_status",
    ),
    (
        (True, 10, "accepted", "supported"),
        (False, 10, "unsupported", "blocked"),
        (True, 9, "accepted", "blocked"),
    ),
)
def test_claim_artifact_status_is_mechanically_derived(
    waymax_executed: bool,
    idm_responder_n: int,
    expected_waymax_status: str,
    expected_claim_status: str,
) -> None:
    receipt = _official_claim_receipt()
    qualification = _qualification_rows(
        receipt,
        qualified_n=10 if waymax_executed else 0,
    )
    accounting = (
        _accepted_claim_accounting()
        if waymax_executed
        else m6.m6_waymax_unsupported_rows(10)
    )
    waymax_status, claim_status = (
        m6._derive_waymax_and_real_reactivity_statuses(
            receipt=receipt,
            primary_matrix=_claim_primary_matrix(
                receipt,
                idm_responder_n=idm_responder_n,
            ),
            qualification=qualification,
            accounting=accounting,
            determinism=_claim_determinism(executed=waymax_executed),
        )
    )
    payload = m6._claim_limitations_payload(OFFICIAL_MODE, claim_status)

    assert waymax_status == expected_waymax_status
    assert claim_status == expected_claim_status
    assert payload["claim_status"] == expected_claim_status
    assert payload["bounded_claim"] == (
        m6.M6_ACCEPTED_BOUNDED_CLAIM
        if expected_claim_status == "supported"
        else m6.M6_BLOCKED_BOUNDED_CLAIM
    )
    if expected_claim_status == "blocked":
        assert m6.M6_ACCEPTED_BOUNDED_CLAIM not in m6._canonical_json_text(
            payload
        )


def test_data_free_claim_artifact_is_explicitly_nonpromotable(
    tmp_path: Path,
) -> None:
    store = _successful_data_free(_project(tmp_path), "claim-nonpromotable")
    claim = json.loads(
        (store.run_path / m6.CLAIM_LIMITATIONS_PATH).read_text(
            encoding="ascii"
        )
    )

    assert claim == m6._claim_limitations_payload(DATA_FREE_MODE, "blocked")
    assert claim["claim_status"] == "nonpromotable"
    assert m6.M6_ACCEPTED_BOUNDED_CLAIM not in (
        store.run_path / m6.CLAIM_LIMITATIONS_PATH
    ).read_text(encoding="ascii")


def test_blocked_claim_artifact_tamper_cannot_verify(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = _successful_data_free(project, "claim-tamper")
    path = store.run_path / m6.CLAIM_LIMITATIONS_PATH
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["bounded_claim"] = m6.M6_ACCEPTED_BOUNDED_CLAIM
    payload["claim_status"] = "supported"
    path.write_bytes(m6._canonical_json_bytes(payload))

    with pytest.raises(M6ResultStoreIntegrityError):
        verify_m6_result_store(
            project,
            store.run_name,
            allow_data_free=True,
        )


def test_promoted_claim_templates_are_status_conditional() -> None:
    supported = m6._promoted_claim_and_limitations("supported")
    blocked = m6._promoted_claim_and_limitations("blocked")

    assert supported == {
        "bounded_claim": m6.M6_ACCEPTED_BOUNDED_CLAIM,
        "claim_status": "supported",
        "limitations": list(m6.M6_FIXED_LIMITATIONS),
    }
    assert blocked == {
        "bounded_claim": m6.M6_BLOCKED_BOUNDED_CLAIM,
        "claim_status": "blocked",
        "limitations": list(m6.M6_FIXED_LIMITATIONS),
    }
    assert blocked["bounded_claim"] != supported["bounded_claim"]
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="claim status is invalid",
    ):
        m6._promoted_claim_and_limitations("accepted")


def test_unsupported_waymax_blocks_claim_above_responder_floor(
    tmp_path: Path,
) -> None:
    payload = _valid_promoted_payload(_project(tmp_path))
    timing = next(
        row
        for row in payload["primary_matrix"]
        if row["policy_name"] == "idm"
        and row["metric_name"] == "response_timeliness_s"
    )
    timing["responder_n"] = 10
    timing["censor_n"] = 0

    m6._validate_sanitized_aggregate_payload(payload)
    claim = payload["claim_and_limitations"]
    assert claim == m6._promoted_claim_and_limitations("blocked")

    wrong_text = copy.deepcopy(payload)
    wrong_text["claim_and_limitations"]["bounded_claim"] = (
        m6.M6_ACCEPTED_BOUNDED_CLAIM
    )
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="claim text/status is not mechanically derived",
    ):
        m6._validate_sanitized_aggregate_payload(wrong_text)

    wrong_gate = copy.deepcopy(payload)
    wrong_gate["execution"]["gate_status"][
        "real_reactivity_claim"
    ] = "supported"
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="execution gate statuses drifted",
    ):
        m6._validate_sanitized_aggregate_payload(wrong_gate)

    forged_support = copy.deepcopy(wrong_gate)
    forged_support["claim_and_limitations"] = (
        m6._promoted_claim_and_limitations("supported")
    )
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="claim text/status is not mechanically derived",
    ):
        m6._validate_sanitized_aggregate_payload(forged_support)


def test_sanitized_aggregate_validates_every_recursive_public_field(
    tmp_path: Path,
) -> None:
    payload = _valid_promoted_payload(_project(tmp_path))
    m6._validate_sanitized_aggregate_payload(payload)
    canonical = m6._canonical_json_text(payload)
    aggregate = M6SanitizedAggregate(
        canonical,
        _factory_sentinel=m6._AGGREGATE_SENTINEL,
    )
    assert aggregate.to_dict() == payload
    with pytest.raises(M6ResultStoreStateError, match="verified terminal"):
        M6SanitizedAggregate(
            canonical,
            _factory_sentinel=object(),
        )

    def value_at(root: object, path: tuple[object, ...]) -> object:
        value = root
        for part in path:
            value = value[part]  # type: ignore[index]
        return value

    def mapping_paths(
        value: object,
        path: tuple[object, ...] = (),
    ):
        if type(value) is dict:
            yield path
            for key, child in value.items():
                yield from mapping_paths(child, (*path, key))
        elif type(value) is list:
            for index, child in enumerate(value):
                yield from mapping_paths(child, (*path, index))

    def scalar_paths(
        value: object,
        path: tuple[object, ...] = (),
    ):
        if type(value) is dict:
            for key, child in value.items():
                yield from scalar_paths(child, (*path, key))
        elif type(value) is list:
            for index, child in enumerate(value):
                yield from scalar_paths(child, (*path, index))
        else:
            yield path

    removed_fields = 0
    for mapping_path in tuple(mapping_paths(payload)):
        mapping = value_at(payload, mapping_path)
        assert type(mapping) is dict
        for key in tuple(mapping):
            mutated = copy.deepcopy(payload)
            target = value_at(mutated, mapping_path)
            assert type(target) is dict
            del target[key]
            with pytest.raises(M6ResultStoreIntegrityError):
                m6._validate_sanitized_aggregate_payload(mutated)
            removed_fields += 1

    mutated_leaves = 0
    for leaf_path in tuple(scalar_paths(payload)):
        mutated = copy.deepcopy(payload)
        parent = value_at(mutated, leaf_path[:-1])
        parent[leaf_path[-1]] = {"invalid_public_scalar": True}  # type: ignore[index]
        with pytest.raises(M6ResultStoreIntegrityError):
            m6._validate_sanitized_aggregate_payload(mutated)
        mutated_leaves += 1
    assert removed_fields > 200
    assert mutated_leaves > 500


def test_data_free_is_nonpromotable(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _successful_data_free(project)
    with pytest.raises(M6ResultStoreIntegrityError, match="explicit"):
        reconstruct_sanitized_m6_aggregate(project, "data-free")
    with pytest.raises(M6ResultStoreIntegrityError, match="non-promotable"):
        reconstruct_sanitized_m6_aggregate(
            project,
            "data-free",
            allow_data_free=True,
        )


def test_terminal_marker_retries_transient_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "fsync-retry", mode=DATA_FREE_MODE)
    _write_complete(store)
    store.commit()
    original = m6._fsync_directory
    calls = 0

    def transient(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient")
        original(path)

    monkeypatch.setattr(m6, "_fsync_directory", transient)
    store.mark_terminal_success()
    assert calls >= 2
    assert (store.run_path / TERMINAL_SUCCESS_MARKER).is_file()


def test_persistent_directory_fsync_failure_never_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "fsync-fail", mode=DATA_FREE_MODE)
    _write_complete(store)
    store.commit()

    def persistent(_path: Path) -> None:
        raise OSError("persistent")

    monkeypatch.setattr(m6, "_fsync_directory", persistent)
    with pytest.raises(OSError, match="persistent"):
        store.mark_terminal_success()
    with pytest.raises(M6ResultStoreStateError):
        store.mark_terminal_success()


@pytest.mark.parametrize("mutation", ["hash", "mode", "hardlink", "symlink"])
def test_terminal_verifier_rejects_tamper_and_unsafe_nodes(
    tmp_path: Path,
    mutation: str,
) -> None:
    project = _project(tmp_path)
    store = _successful_data_free(project, f"unsafe-{mutation}")
    path = store.run_path / "primary-matrix.parquet"
    if mutation == "hash":
        with path.open("r+b") as handle:
            first = handle.read(1)
            handle.seek(0)
            handle.write(bytes([first[0] ^ 0xFF]))
            handle.flush()
            os.fsync(handle.fileno())
    elif mutation == "mode":
        path.chmod(0o640)
    elif mutation == "hardlink":
        os.link(path, tmp_path / "second-link.parquet")
    else:
        payload = path.read_bytes()
        replacement = tmp_path / "replacement.parquet"
        replacement.write_bytes(payload)
        path.unlink()
        path.symlink_to(replacement)
    with pytest.raises(M6ResultStoreIntegrityError):
        verify_m6_result_store(
            project,
            f"unsafe-{mutation}",
            allow_data_free=True,
        )


def test_waymax_post_seal_tamper_fails_before_scalar_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    store = _successful_data_free(project, "waymax-post-seal-tamper")
    path = store.run_path / "waymax-scene-scalars.parquet"
    with path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0xFF]))
        handle.flush()
        os.fsync(handle.fileno())

    def forbidden_parse(_rows):
        raise AssertionError("parser ran before artifact hash verification")

    monkeypatch.setattr(
        m6,
        "parse_m6_waymax_scene_scalar_table",
        forbidden_parse,
    )
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="size/SHA-256",
    ):
        verify_m6_result_store(
            project,
            "waymax-post-seal-tamper",
            allow_data_free=True,
        )


def test_verifier_parses_authenticated_snapshot_then_rejects_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    store = _successful_data_free(project, "snapshot-swap")
    target = store.run_path / "primary-matrix.parquet"
    expected = target.read_bytes()
    original_snapshot = m6._read_guarded_snapshot
    swapped = False

    def swap_after_snapshot(path: Path, run_path: Path):
        nonlocal swapped
        snapshot = original_snapshot(path, run_path)
        if path == target and not swapped:
            swapped = True
            with target.open("r+b") as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first[0] ^ 0xFF]))
                handle.flush()
                os.fsync(handle.fileno())
        return snapshot

    original_parse = m6._parse_guarded_parquet_payload
    parsed_authenticated_snapshot = False

    def assert_authenticated(payload: bytes, dataset: str):
        nonlocal parsed_authenticated_snapshot
        if dataset == PRIMARY_MATRIX:
            assert payload == expected
            parsed_authenticated_snapshot = True
        return original_parse(payload, dataset)

    monkeypatch.setattr(m6, "_read_guarded_snapshot", swap_after_snapshot)
    monkeypatch.setattr(
        m6,
        "_parse_guarded_parquet_payload",
        assert_authenticated,
    )
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="changed during complete verification",
    ):
        verify_m6_result_store(
            project,
            "snapshot-swap",
            allow_data_free=True,
        )
    assert swapped
    assert parsed_authenticated_snapshot


@pytest.mark.parametrize("rewrite_committed", (False, True))
def test_rewritten_artifact_and_manifest_fail_before_any_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rewrite_committed: bool,
) -> None:
    project = _project(tmp_path)
    run_name = f"manifest-parser-trap-{int(rewrite_committed)}"
    store = _successful_data_free(project, run_name)
    artifact_path = store.run_path / "waymax-scene-scalars.parquet"
    payload = bytearray(artifact_path.read_bytes())
    payload[0] ^= 0xFF
    artifact_path.write_bytes(payload)

    manifest_path = store.run_path / m6.MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    record = next(
        item
        for item in manifest["artifacts"]
        if item["path"] == artifact_path.name
    )
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    record["size_bytes"] = len(payload)
    manifest_bytes = m6._canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    if rewrite_committed:
        committed_path = store.run_path / COMMITTED_MARKER
        committed = json.loads(committed_path.read_text(encoding="ascii"))
        committed["manifest_sha256"] = hashlib.sha256(
            manifest_bytes
        ).hexdigest()
        committed["manifest_size_bytes"] = len(manifest_bytes)
        committed_path.write_bytes(m6._canonical_json_bytes(committed))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("semantic parser ran before raw marker binding")

    monkeypatch.setattr(m6, "_decode_canonical_mapping", forbidden)
    monkeypatch.setattr(m6, "_parse_guarded_parquet_payload", forbidden)
    monkeypatch.setattr(
        m6,
        "parse_m6_waymax_scene_scalar_table",
        forbidden,
    )
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match=(
            "raw TERMINAL_SUCCESS"
            if rewrite_committed
            else "raw COMMITTED"
        ),
    ):
        verify_m6_result_store(
            project,
            run_name,
            allow_data_free=True,
        )


def test_git_visible_path_and_direct_writer_construction_rejected(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path, ignored=False)
    with pytest.raises(M6ResultStoreIntegrityError, match="visible to Git"):
        M6ResultStore.create(project, "visible", mode=DATA_FREE_MODE)
    with pytest.raises(M6ResultStoreStateError, match="only be created"):
        M6ResultStore(
            project_root=project,
            run_name="forged",
            run_path=project / "outputs" / "m6" / "forged",
            profile=m6.DATA_FREE_M6_TEST_PROFILE,
            capability_nonce=b"x" * 32,
            _create_sentinel=object(),
        )


def test_failed_or_premature_store_is_nonresumable(tmp_path: Path) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "premature", mode=DATA_FREE_MODE)
    with pytest.raises(
        (M6ResultStoreStateError, M6ResultStoreIntegrityError),
    ):
        store.finalize()
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()
    assert not (store.run_path / COMMITTED_MARKER).exists()
    with pytest.raises(M6ResultStoreStateError):
        store.write_eligibility_ledger(
            _eligibility_rows(10, 10, mode=DATA_FREE_MODE)
        )


def test_privacy_scan_rejects_nested_private_identity() -> None:
    with pytest.raises(M6ResultStoreIntegrityError, match="private key"):
        m6._assert_promoted_privacy(
            {"safe": [{"nested": {"cohort_index": 7}}]}
        )


def test_exact_json_bool_and_integer_aliases_fail_closed() -> None:
    receipt = m6.M6EligibilityReceipt(
        mode=DATA_FREE_MODE,
        population_size=10,
        eligible_cohort_indices=tuple(range(10)),
        secondary_b4_cohort_indices=tuple(range(10)),
        rejection_reason_counts={
            reason: 0 for reason in m6.M6_PRIMARY_REJECTION_REASONS
        },
        primary_intervention_fingerprint=(
            m6.M6_PRIMARY_INTERVENTION_FINGERPRINT
        ),
        secondary_intervention_fingerprint=(
            m6.M6_SECONDARY_INTERVENTION_FINGERPRINT
        ),
    ).to_dict()
    receipt["population_size"] = 10.0
    with pytest.raises(M6ResultStoreIntegrityError, match="JSON integer"):
        m6.M6EligibilityReceipt.from_dict(receipt)
