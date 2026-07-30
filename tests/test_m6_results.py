"""Data-free and adversarial tests for the mode-bound M6 evidence store."""
from __future__ import annotations

import dataclasses
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import numpy as np

import evalsim.results.m6 as m6
from evalsim.evaluation.m6_waymax_metrics import (
    M6WaymaxParsedScalarTable,
    build_m6_waymax_scene_scalar_table,
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
from tests.test_m6_waymax_metrics import _invented_views


_ALL_PREFLIGHT_CHECKS = {
    name: True for name in m6.M6_PREFLIGHT_CHECK_DOMAIN
}


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


def _waymax_selection(
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
    return select_m6_waymax_subset(
        rows,
        primary_domain=domain,
        primary_scenarios=scenarios,
    )


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


def _stage_rows(mode: str) -> list[dict[str, object]]:
    return [
        {
            "stage_name": name,
            "duration_ms": index + 1 if mode == OFFICIAL_MODE else index,
        }
        for index, name in enumerate(m6.M6_STAGE_DOMAIN)
    ]


def _review_rows(
    *,
    p3_count: int = 0,
) -> list[dict[str, object]]:
    return [
        {
            "role": role,
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
    review_rows: list[dict[str, object]] | None = None,
) -> None:
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
        issued = None
        selection = None
    else:
        selection = _waymax_selection(
            receipt,
            qualified_n=qualified_n,
        )
        store.write_waymax_qualification(selection)
        qualification = [
            dict(row)
            for row in m6._waymax_qualification_rows_from_selection(
                selection,
                receipt,
            )
        ]
        issued = build_m6_waymax_scene_scalar_table(
            _invented_views(selection) if selection.supported else (),
            selection=selection,
        )
    primary = _scene_rows(
        receipt.eligible_cohort_indices,
        m6.M6_PRIMARY_INTERVENTION_FINGERPRINT,
    )
    store.write_primary_scene_scalars(primary)
    store.write_primary_matrix()
    store.write_primary_repeat_scene_scalars([dict(row) for row in primary])
    store.write_primary_repeat_matrix()
    store.write_secondary_scene_scalars(
        _scene_rows(
            receipt.secondary_b4_cohort_indices,
            m6.M6_SECONDARY_INTERVENTION_FINGERPRINT,
            value=secondary_value,
        )
    )
    store.write_secondary_matrix()
    store.write_negative_timing_observations(
        _negative_observation_rows(receipt)
    )
    if store.profile.data_free:
        store.write_waymax_scene_scalars()
    else:
        assert issued is not None and selection is not None
        store.write_waymax_scene_scalars(
            issued,
            selection=selection,
        )
    store.write_waymax_field_comparisons(_waymax_field_rows(qualification))
    store.write_waymax_determinism()
    store.write_waymax_accounting()
    store.write_typed_provenance(_provenance(store.profile.mode))
    store.write_stage_timings(_stage_rows(store.profile.mode))
    store.write_determinism_receipt()
    store.write_claim_limitations()
    store.write_review_decisions(
        _review_rows(p3_count=1)
        if review_rows is None
        else review_rows
    )
    store.write_execution_summary(
        fresh_worker_peak_rss_bytes=(
            1024 if store.profile.mode == OFFICIAL_MODE else 0
        )
    )

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
            "bounded_claim": m6.M6_ACCEPTED_BOUNDED_CLAIM,
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
            "review_decisions": _review_rows(p3_count=1),
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
def test_eligibility_only_seals_source_qualification_and_needs_capability(
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
    store.commit()
    verified = m6.verify_committed_m6_result_store(
        project,
        f"eligibility-{eligible_n}",
        expected_mode=ELIGIBILITY_ONLY_MODE,
    )
    assert set(verified.tables) == {
        m6.ELIGIBILITY_LEDGER,
        m6.WAYMAX_QUALIFICATION,
    }
    with pytest.raises(M6ResultStoreStateError, match="disabled"):
        store.mark_terminal_success()
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
    store.write_waymax_qualification(
        _waymax_selection(receipt, qualified_n=0)
    )
    store.write_compute_pilot_summary(
        {
            "pilot_scene_n": 8,
            "total_wall_ms": 1_000,
            "max_scene_ms": 200,
            "decode_ms": 100,
            "numpy_ms": 200,
            "waymax_ms": 300,
            "verification_ms": 100,
            "fresh_worker_peak_rss_bytes": 1024,
            "passed": True,
        }
    )
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
    }
    with pytest.raises(M6ResultStoreStateError, match="disabled"):
        store.mark_terminal_success()


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
    store.write_waymax_qualification(
        _waymax_selection(receipt, qualified_n=0)
    )
    row = {
        "pilot_scene_n": 8,
        "total_wall_ms": 1_000,
        "max_scene_ms": 200,
        "decode_ms": 100,
        "numpy_ms": 200,
        "waymax_ms": 300,
        "verification_ms": 100,
        "fresh_worker_peak_rss_bytes": 1024,
        "passed": True,
    }
    row[field] = 0
    with pytest.raises(M6ResultStoreIntegrityError):
        store.write_compute_pilot_summary(row)


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
    store.write_waymax_qualification(
        _waymax_selection(receipt, qualified_n=0)
    )
    store.write_compute_pilot_summary(
        {
            "pilot_scene_n": 8,
            "total_wall_ms": 30 * 60 * 1_000 + 1,
            "max_scene_ms": 200,
            "decode_ms": 100,
            "numpy_ms": 200,
            "waymax_ms": 300,
            "verification_ms": 100,
            "fresh_worker_peak_rss_bytes": 1024,
            "passed": False,
        }
    )
    store.commit()
    with pytest.raises(M6ResultStoreStateError, match="disabled"):
        store.mark_terminal_success()


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
    selection = _waymax_selection(receipt, qualified_n=10)
    store.write_waymax_qualification(selection)
    issued = build_m6_waymax_scene_scalar_table(
        _invented_views(selection),
        selection=selection,
    )
    evidence: object
    if attack == "mapping":
        evidence = [row.to_store_dict() for row in issued.rows]
    elif attack == "parsed":
        evidence = parse_m6_waymax_scene_scalar_table(issued.rows)
        assert isinstance(evidence, M6WaymaxParsedScalarTable)
        assert evidence.promotable is False
    else:
        replacement_rows = list(issued.rows)
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
        evidence = issued
    with pytest.raises((TypeError, ValueError)):
        store.write_waymax_scene_scalars(  # type: ignore[arg-type]
            evidence,
            selection=selection,
        )


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
        with pytest.raises(M6ResultStoreIntegrityError):
            store.write_waymax_field_comparisons(rows)


@pytest.mark.parametrize(
    "mode",
    (ELIGIBILITY_ONLY_MODE, COMPUTE_PILOT_MODE, OFFICIAL_MODE),
)
def test_non_data_free_waymax_determinism_is_fully_disabled(
    tmp_path: Path,
    mode: str,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, f"det-disabled-{mode}", mode=mode)
    with pytest.raises(M6ResultStoreStateError, match="disabled"):
        store.write_waymax_determinism()
    assert (store.run_path / TERMINAL_FAILURE_MARKER).is_file()


@pytest.mark.parametrize(
    "caller_value",
    (
        (),
        ({"status": "passed"},),
        m6.m6_data_free_waymax_determinism_rows(),
    ),
)
def test_official_waymax_determinism_rejects_every_caller_value(
    tmp_path: Path,
    caller_value: object,
) -> None:
    project = _project(tmp_path)
    store = M6ResultStore.create(project, "det-caller", mode=OFFICIAL_MODE)
    with pytest.raises(M6ResultStoreStateError, match="disabled"):
        store.write_waymax_determinism(caller_value)


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
    with pytest.raises(TypeError, match="accepts no caller rows"):
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


def test_review_decision_is_derived_from_p1_p2_while_p3_is_retained() -> None:
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
    )
    digest = _sha("sealed-review-precursor")
    accepted_rows = [
        {**row, "evidence_catalog_sha256": digest}
        for row in _review_rows(p3_count=2)
    ]
    accepted = m6._normalize_review_decisions(
        accepted_rows,
        receipt,
        expected_evidence_catalog_sha256=digest,
    )
    assert all(row["decision"] == "accept" for row in accepted)
    forged = [dict(row) for row in accepted_rows]
    forged[0]["p2_count"] = 1
    with pytest.raises(M6ResultStoreIntegrityError, match="differs"):
        m6._normalize_review_decisions(forged, receipt)
    with pytest.raises(M6ResultStoreIntegrityError, match="sealed precursor"):
        m6._normalize_review_decisions(
            accepted_rows,
            receipt,
            expected_evidence_catalog_sha256=_sha("different"),
        )


def test_review_writer_derives_digest_and_rejects_caller_digest(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    forged = _review_rows()
    for row in forged:
        row["evidence_catalog_sha256"] = _sha("caller-chosen")
    store = M6ResultStore.create(
        project,
        "review-caller-digest",
        mode=DATA_FREE_MODE,
    )
    with pytest.raises(
        M6ResultStoreIntegrityError,
        match="fields do not match",
    ):
        _write_complete(store, review_rows=forged)

    successful = _successful_data_free(project, "review-derived")
    review_rows = successful._read_dataset_rows(m6.REVIEW_DECISIONS)
    receipt = successful.eligibility_receipt
    assert receipt is not None
    expected = m6._review_precursor_sha256(
        receipt,
        successful.artifacts,
    )
    assert {
        row["evidence_catalog_sha256"] for row in review_rows
    } == {expected}


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
            observed_preflight_sha256="3" * 64,
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
    with pytest.raises(M6ResultStoreStateError, match="minting is disabled"):
        m6._mint_m6_terminal_capability(
            store,
            object(),  # type: ignore[arg-type]
        )


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
    monkeypatch.setattr(m6, "_read_guarded_parquet", forbidden)
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
