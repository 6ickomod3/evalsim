"""Pure adapters between accepted-M4 reloads, M6 NumPy results, and store rows.

This module is deliberately narrower than an official command.  It performs no data
access, optional-runtime import, Git inspection, result-store mutation, or terminal
transition.  A command may use :class:`M6OfficialCaseCollector` as the visitor supplied
to the accepted-M4 reload boundary, then pass the resulting detached cases to
:func:`run_m6_official_numpy`.

Every public row adapter revalidates the typed evidence it consumes and projects only
the fixed, privacy-safe M6 store fields.  Scenario IDs, contract agent IDs, native
Waymax slots, source records, and source states never enter a returned row.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
import hashlib
import json
import math
import threading
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from evalsim.contracts import evaluate_paired_metric
from evalsim.contracts.counterfactual import ScenarioSnapshot
from evalsim.evaluation.m6 import (
    M6EligibilityLedger,
    M6EvaluationCase,
    M6EvaluationResult,
    M6PairedSceneResult,
    M6SecondaryPlanEntry,
    _run_m6_numpy_evaluation_with_phase_observer,
    assert_m6_sham_matches_legacy_prefix,
)
from evalsim.metrics.m6 import (
    m6_primary_paired_metrics,
    world_trajectory_tensor_equal,
)
from evalsim.perturb.m6 import (
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    SECONDARY_BRAKE_MAGNITUDE_MPS2,
    InterventionCompilationError,
    audit_ego_plan_feasibility,
    compile_longitudinal_brake_pulse_plan,
    longitudinal_brake_pulse_spec,
)
from evalsim.rollout import policy_trace_prefix_equal
from evalsim.sources.m5_m4_reuse import ReloadedM4Member
from evalsim.stats.m6 import (
    M6_PRIMARY_METRICS,
    M6_PRIMARY_POLICY_ROLES,
    M6_RESPONSE_TIMELINESS_METRIC,
)
_NUMPY_ROWS_ISSUER = object()


M6_OFFICIAL_ADAPTER_VERSION = "m6-official-adapter-1.1.0"
M6_OFFICIAL_POPULATION_SIZE = 128
_NUMPY_EXECUTION_PHASES = (
    "numpy_rollouts",
    "paired_metrics",
    "statistics",
)
_NUMPY_PHASES = (*_NUMPY_EXECUTION_PHASES, "verification")

_ELIGIBILITY_ONLY_MODE = "eligibility_only"
_COMPUTE_PILOT_MODE = "compute_pilot"
_OFFICIAL_MODE = "official"
_DATA_FREE_MODE = "data_free"
_MODES = frozenset(
    {
        _ELIGIBILITY_ONLY_MODE,
        _COMPUTE_PILOT_MODE,
        _OFFICIAL_MODE,
        _DATA_FREE_MODE,
    }
)
_MODE_POPULATION = {
    _ELIGIBILITY_ONLY_MODE: M6_OFFICIAL_POPULATION_SIZE,
    _COMPUTE_PILOT_MODE: M6_OFFICIAL_POPULATION_SIZE,
    _OFFICIAL_MODE: M6_OFFICIAL_POPULATION_SIZE,
    _DATA_FREE_MODE: 10,
}
_ELIGIBILITY_FIELDS = (
    "cohort_index",
    "primary_eligible",
    "rejection_reason",
    "secondary_b4_feasible",
)
_SCENE_FIELDS = (
    "cohort_index",
    "policy_name",
    "policy_access_role",
    "metric_name",
    "metric_version",
    "unit",
    "value",
    "responded",
    "responder_latency_s",
    "source_pairing_complete",
    "intervention_config_fingerprint",
)
_NEGATIVE_FIELDS = (
    "gate_name",
    "cohort_index",
    "policy_name",
    "assessed_n",
    "violation_n",
    "observation_sha256",
)
_NEGATIVE_GATE_POLICIES: tuple[tuple[str, tuple[str | None, ...]], ...] = (
    ("log_replay_world_tensor_equality", ("log_replay",)),
    ("constant_velocity_world_tensor_equality", ("constant_velocity",)),
    (
        "sham_legacy_equality",
        tuple(role.policy_name for role in M6_PRIMARY_POLICY_ROLES),
    ),
    (
        "synchronous_response_floor",
        tuple(role.policy_name for role in M6_PRIMARY_POLICY_ROLES),
    ),
    ("primary_plan_feasibility", (None,)),
    ("nested_dose_monotonicity", (None,)),
)
_METRICS_BY_NAME = {metric.metric_name: metric for metric in M6_PRIMARY_METRICS}
_ACCESS_BY_POLICY = {
    role.policy_name: role.access_role for role in M6_PRIMARY_POLICY_ROLES
}
_PRIMARY_CONFIGURATION_FINGERPRINT = longitudinal_brake_pulse_spec(
    PRIMARY_BRAKE_MAGNITUDE_MPS2
).configuration_fingerprint
_SECONDARY_CONFIGURATION_FINGERPRINT = longitudinal_brake_pulse_spec(
    SECONDARY_BRAKE_MAGNITUDE_MPS2
).configuration_fingerprint
assert _PRIMARY_CONFIGURATION_FINGERPRINT is not None
assert _SECONDARY_CONFIGURATION_FINGERPRINT is not None


class M6OfficialAdapterError(RuntimeError):
    """Typed evidence cannot be projected into the fixed official row domain."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise M6OfficialAdapterError(
            "m6_official_adapter_noncanonical_evidence"
        ) from exc


def _observation_sha256(
    *,
    gate_name: str,
    cohort_index: int,
    policy_name: str | None,
    violation: bool,
    evidence: Mapping[str, Any],
) -> str:
    payload = {
        "adapter_version": M6_OFFICIAL_ADAPTER_VERSION,
        "assessed_n": 1,
        "cohort_index": cohort_index,
        "evidence": dict(evidence),
        "gate_name": gate_name,
        "policy_name": policy_name,
        "violation_n": int(violation),
    }
    return hashlib.sha256(
        b"evalsim-m6-official-negative-observation-v1\x00"
        + _canonical_json_bytes(payload)
    ).hexdigest()


def _frozen_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    fields: tuple[str, ...],
) -> tuple[Mapping[str, Any], ...]:
    expected = set(fields)
    frozen: list[Mapping[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if set(row) != expected:
            raise M6OfficialAdapterError(
                "m6_official_adapter_row_fields_drifted"
            )
        frozen.append(MappingProxyType(row))
    return tuple(frozen)


def m6_case_from_reloaded_member(member: ReloadedM4Member) -> M6EvaluationCase:
    """Detach one verified reload from its native record/state ownership.

    The returned case contains a fresh contract ``Scenario`` only.  It retains no
    reference to ``member``, ``member.record``, its native Waymax state, or raw audit
    arrays.
    """

    if not isinstance(member, ReloadedM4Member):
        raise TypeError("member must be a ReloadedM4Member")
    snapshot = ScenarioSnapshot.from_scenario(member.scenario)
    snapshot.revalidate()
    return M6EvaluationCase(
        cohort_index=member.cohort_index,
        scenario=snapshot.to_scenario(),
    )


class M6OfficialCaseCollector:
    """Accepted-M4 visitor that retains only detached contract snapshots.

    Accepted-M4 reload visitation is shard-major, not necessarily cohort-index-major.
    The collector therefore accepts each opaque index once and exposes cases only after
    the complete ``0..127`` domain has arrived, sorted into evaluator order.
    """

    __slots__ = ("_snapshots",)

    def __init__(self) -> None:
        self._snapshots: dict[int, ScenarioSnapshot] = {}

    def __call__(self, member: ReloadedM4Member) -> None:
        case = m6_case_from_reloaded_member(member)
        if case.cohort_index in self._snapshots:
            raise M6OfficialAdapterError(
                "m6_official_duplicate_cohort_member"
            )
        snapshot = ScenarioSnapshot.from_scenario(case.scenario)
        snapshot.revalidate()
        self._snapshots[case.cohort_index] = snapshot

    @property
    def count(self) -> int:
        return len(self._snapshots)

    @property
    def cases(self) -> tuple[M6EvaluationCase, ...]:
        expected = set(range(M6_OFFICIAL_POPULATION_SIZE))
        if set(self._snapshots) != expected:
            raise M6OfficialAdapterError(
                "m6_official_cohort_incomplete"
            )
        cases: list[M6EvaluationCase] = []
        for cohort_index in range(M6_OFFICIAL_POPULATION_SIZE):
            snapshot = self._snapshots[cohort_index]
            snapshot.revalidate()
            cases.append(
                M6EvaluationCase(
                    cohort_index=cohort_index,
                    scenario=snapshot.to_scenario(),
                )
            )
        return tuple(cases)


def _validated_ledger(ledger: M6EligibilityLedger) -> M6EligibilityLedger:
    if not isinstance(ledger, M6EligibilityLedger):
        raise TypeError("ledger must be an M6EligibilityLedger")
    for entry in ledger.entries:
        entry.revalidate()
    # Construction repeats ordering, uniqueness, disposition, and snapshot checks.
    return M6EligibilityLedger(tuple(ledger.entries))


def _secondary_feasibility(
    ledger: M6EligibilityLedger,
) -> tuple[M6SecondaryPlanEntry, ...]:
    entries: list[M6SecondaryPlanEntry] = []
    fingerprints: set[str] = set()
    for entry in ledger.entries:
        if not entry.eligible:
            continue
        source = entry.source_snapshot.to_scenario()
        try:
            plan = compile_longitudinal_brake_pulse_plan(
                source,
                SECONDARY_BRAKE_MAGNITUDE_MPS2,
            )
        except InterventionCompilationError as exc:
            if exc.code != "secondary_ego_plan_infeasible":
                raise M6OfficialAdapterError(
                    "m6_official_secondary_plan_unexpected_failure"
                ) from exc
            entries.append(
                M6SecondaryPlanEntry(
                    cohort_index=entry.cohort_index,
                    feasible=False,
                    reason=exc.code,
                )
            )
        except Exception as exc:
            raise M6OfficialAdapterError(
                "m6_official_secondary_plan_evaluation_failed"
            ) from exc
        else:
            plan.revalidate()
            if plan.configuration_fingerprint != (
                _SECONDARY_CONFIGURATION_FINGERPRINT
            ):
                raise M6OfficialAdapterError(
                    "m6_official_secondary_configuration_drifted"
                )
            fingerprints.add(plan.configuration_fingerprint)
            entries.append(
                M6SecondaryPlanEntry(
                    cohort_index=entry.cohort_index,
                    feasible=True,
                    reason=None,
                )
            )
    if len(fingerprints) > 1:
        raise M6OfficialAdapterError(
            "m6_official_secondary_configuration_drifted"
        )
    return tuple(entries)


def m6_eligibility_rows(
    ledger: M6EligibilityLedger,
    *,
    mode: str,
    secondary_plan_ledger: Sequence[M6SecondaryPlanEntry] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Project the complete source-only disposition ledger to exact store rows.

    For compute-pilot, official, and data-free modes, b=4 feasibility is independently
    recomputed from retained source snapshots before any policy outcome is needed.  A
    supplied evaluator ledger is accepted only as a cross-check of that reconstruction.
    """

    if mode not in _MODES:
        raise ValueError("mode is not a registered M6 mode")
    checked = _validated_ledger(ledger)
    population = _MODE_POPULATION[mode]
    if tuple(entry.cohort_index for entry in checked.entries) != tuple(
        range(population)
    ):
        raise M6OfficialAdapterError(
            "m6_official_eligibility_population_incomplete"
        )

    secondary_by_index: dict[int, bool] = {}
    if mode == _ELIGIBILITY_ONLY_MODE:
        if secondary_plan_ledger not in (None, ()):
            raise M6OfficialAdapterError(
                "m6_official_eligibility_only_has_secondary_evidence"
            )
    else:
        independently_derived = _secondary_feasibility(checked)
        if secondary_plan_ledger is not None:
            supplied = tuple(secondary_plan_ledger)
            if supplied != independently_derived:
                raise M6OfficialAdapterError(
                    "m6_official_secondary_plan_ledger_drifted"
                )
        secondary_by_index = {
            entry.cohort_index: entry.feasible for entry in independently_derived
        }

    rows = (
        {
            "cohort_index": entry.cohort_index,
            "primary_eligible": entry.eligible,
            "rejection_reason": entry.reason,
            "secondary_b4_feasible": (
                None
                if mode == _ELIGIBILITY_ONLY_MODE or not entry.eligible
                else secondary_by_index[entry.cohort_index]
            ),
        }
        for entry in checked.entries
    )
    return _frozen_rows(rows, fields=_ELIGIBILITY_FIELDS)


def _timeliness_fields(metric_result: Any) -> tuple[bool | None, float | None]:
    if metric_result.metric_name != M6_RESPONSE_TIMELINESS_METRIC:
        return None, None
    responded = metric_result.details.get("responded")
    censored = metric_result.details.get("censored")
    event_time = metric_result.details.get("event_time_s")
    if type(responded) is not bool or type(censored) is not bool or (
        responded is censored
    ):
        raise M6OfficialAdapterError(
            "m6_official_timeliness_accounting_invalid"
        )
    if responded:
        if isinstance(event_time, (bool, np.bool_)) or not isinstance(
            event_time,
            (int, float, np.integer, np.floating),
        ):
            raise M6OfficialAdapterError(
                "m6_official_timeliness_latency_invalid"
            )
        latency = float(event_time)
        if not math.isfinite(latency) or latency < 0.0:
            raise M6OfficialAdapterError(
                "m6_official_timeliness_latency_invalid"
            )
        return True, latency
    if event_time is not None or metric_result.value != 0.0:
        raise M6OfficialAdapterError(
            "m6_official_censored_timeliness_invalid"
        )
    return False, None


def _recomputed_primary_metrics(
    scene: M6PairedSceneResult,
) -> tuple[Any, ...]:
    scene.revalidate()
    scene.pair.revalidate()
    recomputed = tuple(
        evaluate_paired_metric(metric, scene.pair)
        for metric in m6_primary_paired_metrics()
    )
    if tuple(item.to_dict() for item in recomputed) != tuple(
        item.to_dict() for item in scene.primary_metric_results
    ):
        raise M6OfficialAdapterError(
            "m6_official_scene_metric_reconstruction_drifted"
        )
    return recomputed


def _scene_scalar_rows(
    scenes: Sequence[M6PairedSceneResult],
    *,
    cohort_indices: tuple[int, ...],
    fingerprint: str,
) -> tuple[Mapping[str, Any], ...]:
    expected_keys = tuple(
        (
            cohort_index,
            role.policy_name,
            role.access_role,
            metric.metric_name,
            metric.metric_version,
        )
        for cohort_index in cohort_indices
        for role in M6_PRIMARY_POLICY_ROLES
        for metric in M6_PRIMARY_METRICS
    )
    rows: list[dict[str, Any]] = []
    actual_keys: list[tuple[Any, ...]] = []
    for scene in scenes:
        if not isinstance(scene, M6PairedSceneResult):
            raise TypeError("scenes must contain M6PairedSceneResult values")
        if scene.policy_access_role != _ACCESS_BY_POLICY.get(scene.policy_name):
            raise M6OfficialAdapterError(
                "m6_official_policy_access_role_drifted"
            )
        if scene.pair.intervention_plan.configuration_fingerprint != fingerprint:
            raise M6OfficialAdapterError(
                "m6_official_scene_intervention_drifted"
            )
        for result in _recomputed_primary_metrics(scene):
            metric = _METRICS_BY_NAME.get(result.metric_name)
            if metric is None or result.metric_version != metric.metric_version:
                raise M6OfficialAdapterError(
                    "m6_official_scene_metric_identity_drifted"
                )
            responded, latency = _timeliness_fields(result)
            key = (
                scene.cohort_index,
                scene.policy_name,
                scene.policy_access_role,
                result.metric_name,
                result.metric_version,
            )
            actual_keys.append(key)
            rows.append(
                {
                    "cohort_index": scene.cohort_index,
                    "policy_name": scene.policy_name,
                    "policy_access_role": scene.policy_access_role,
                    "metric_name": result.metric_name,
                    "metric_version": result.metric_version,
                    "unit": metric.value_unit,
                    "value": float(result.value),
                    "responded": responded,
                    "responder_latency_s": latency,
                    "source_pairing_complete": True,
                    "intervention_config_fingerprint": fingerprint,
                }
            )
    if tuple(actual_keys) != expected_keys:
        raise M6OfficialAdapterError(
            "m6_official_scene_row_domain_drifted"
        )
    return _frozen_rows(rows, fields=_SCENE_FIELDS)


def m6_primary_scene_scalar_rows(
    result: M6EvaluationResult,
) -> tuple[Mapping[str, Any], ...]:
    """Return the exact primary eligible-N x 12 safe scalar table."""

    if not isinstance(result, M6EvaluationResult):
        raise TypeError("result must be an M6EvaluationResult")
    result.revalidate()
    return _scene_scalar_rows(
        result.primary_scene_results,
        cohort_indices=result.eligibility_ledger.eligible_indices,
        fingerprint=_PRIMARY_CONFIGURATION_FINGERPRINT,
    )


def m6_secondary_scene_scalar_rows(
    result: M6EvaluationResult,
) -> tuple[Mapping[str, Any], ...]:
    """Return primary-measure rows for the separately frozen feasible b=4 subset."""

    if not isinstance(result, M6EvaluationResult):
        raise TypeError("result must be an M6EvaluationResult")
    result.revalidate()
    if not result.secondary_plan_ledger:
        raise M6OfficialAdapterError(
            "m6_official_secondary_evidence_not_enabled"
        )
    feasible = tuple(
        entry.cohort_index
        for entry in result.secondary_plan_ledger
        if entry.feasible
    )
    return _scene_scalar_rows(
        result.secondary_scene_results,
        cohort_indices=feasible,
        fingerprint=_SECONDARY_CONFIGURATION_FINGERPRINT,
    )


def _primary_scene_by_key(
    result: M6EvaluationResult,
) -> Mapping[tuple[int, str], M6PairedSceneResult]:
    expected = {
        (cohort_index, role.policy_name)
        for cohort_index in result.eligibility_ledger.eligible_indices
        for role in M6_PRIMARY_POLICY_ROLES
    }
    by_key = {
        (scene.cohort_index, scene.policy_name): scene
        for scene in result.primary_scene_results
    }
    if set(by_key) != expected or len(by_key) != len(result.primary_scene_results):
        raise M6OfficialAdapterError(
            "m6_official_primary_scene_domain_drifted"
        )
    return MappingProxyType(by_key)


def _sham_legacy_holds(scene: M6PairedSceneResult) -> bool:
    try:
        assert_m6_sham_matches_legacy_prefix(
            scene.pair.scenario.to_scenario(),
            scene.legacy_rollout,
            scene.pair.baseline,
            scene.pair.baseline_plan,
            legacy_trace=scene.legacy_trace,
            sham_trace=scene.baseline_trace,
        )
        return policy_trace_prefix_equal(
            scene.baseline_trace,
            scene.legacy_trace,
        )
    except Exception:
        return False


def _synchronous_floor_holds(scene: M6PairedSceneResult) -> bool:
    pair = scene.pair
    pair.revalidate()
    current = pair.eligibility.current_index
    if not np.array_equal(
        pair.baseline.timestamps[: current + 2],
        pair.intervention.timestamps[: current + 2],
    ):
        return False
    for index, (baseline, treatment) in enumerate(
        zip(pair.baseline.agents, pair.intervention.agents, strict=True)
    ):
        if index == pair.scenario.ego_index:
            continue
        for name in ("valid", "x", "y", "heading", "vx", "vy"):
            if not np.array_equal(
                np.asarray(getattr(baseline, name))[: current + 2],
                np.asarray(getattr(treatment, name))[: current + 2],
            ):
                return False
    return True


def _primary_plan_feasible(scene: M6PairedSceneResult) -> bool:
    plan = scene.pair.intervention_plan
    plan.revalidate()
    audit = audit_ego_plan_feasibility(
        scene.pair.scenario.to_scenario(),
        plan,
    )
    return audit.passed and audit.to_dict() == plan.feasibility.to_dict()


def _nested_plans_monotone(
    primary: M6PairedSceneResult,
    secondary: M6PairedSceneResult,
) -> bool:
    weak = primary.pair.intervention_plan
    strong = secondary.pair.intervention_plan
    weak.revalidate()
    strong.revalidate()
    if (
        weak.configuration_fingerprint != _PRIMARY_CONFIGURATION_FINGERPRINT
        or strong.configuration_fingerprint
        != _SECONDARY_CONFIGURATION_FINGERPRINT
        or not np.array_equal(weak.timestamps, strong.timestamps)
    ):
        return False
    for name in ("x", "y", "heading", "vx", "vy"):
        if not np.array_equal(
            np.asarray(getattr(weak, name))[:1],
            np.asarray(getattr(strong, name))[:1],
        ):
            return False
    weak_speed = np.hypot(weak.vx, weak.vy)
    strong_speed = np.hypot(strong.vx, strong.vy)
    dt = np.diff(weak.timestamps)
    weak_progress = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(0.5 * (weak_speed[:-1] + weak_speed[1:]) * dt),
        )
    )
    strong_progress = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(0.5 * (strong_speed[:-1] + strong_speed[1:]) * dt),
        )
    )
    return bool(
        np.all(strong_speed[1:] <= weak_speed[1:])
        and np.all(strong_progress[1:] <= weak_progress[1:])
    )


def _negative_row(
    *,
    gate_name: str,
    cohort_index: int,
    policy_name: str | None,
    violation: bool,
    evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "gate_name": gate_name,
            "cohort_index": cohort_index,
            "policy_name": policy_name,
            "assessed_n": 1,
            "violation_n": int(violation),
            "observation_sha256": _observation_sha256(
                gate_name=gate_name,
                cohort_index=cohort_index,
                policy_name=policy_name,
                violation=violation,
                evidence=evidence,
            ),
        }
    )


def m6_negative_timing_observation_rows(
    result: M6EvaluationResult,
) -> tuple[Mapping[str, Any], ...]:
    """Derive the exact per-case/per-policy six-gate observation domain."""

    if not isinstance(result, M6EvaluationResult):
        raise TypeError("result must be an M6EvaluationResult")
    result.revalidate()
    primary = _primary_scene_by_key(result)
    secondary = {
        (scene.cohort_index, scene.policy_name): scene
        for scene in result.secondary_scene_results
    }
    feasible = tuple(
        entry.cohort_index
        for entry in result.secondary_plan_ledger
        if entry.feasible
    )
    expected_secondary = {
        (cohort_index, role.policy_name)
        for cohort_index in feasible
        for role in M6_PRIMARY_POLICY_ROLES
    }
    if set(secondary) != expected_secondary or len(secondary) != len(
        result.secondary_scene_results
    ):
        raise M6OfficialAdapterError(
            "m6_official_secondary_scene_domain_drifted"
        )

    rows: list[Mapping[str, Any]] = []
    for gate_name, policies in _NEGATIVE_GATE_POLICIES:
        indices = (
            feasible
            if gate_name == "nested_dose_monotonicity"
            else result.eligibility_ledger.eligible_indices
        )
        for cohort_index in indices:
            for policy_name in policies:
                reference = primary[(cohort_index, "log_replay")]
                if gate_name == "log_replay_world_tensor_equality":
                    scene = primary[(cohort_index, "log_replay")]
                    passed = world_trajectory_tensor_equal(scene.pair)
                    evidence = {
                        "intervention_plan_sha256": (
                            scene.pair.intervention_plan.plan_fingerprint
                        ),
                        "world_tensor_equal": passed,
                    }
                elif gate_name == "constant_velocity_world_tensor_equality":
                    scene = primary[(cohort_index, "constant_velocity")]
                    passed = world_trajectory_tensor_equal(scene.pair)
                    evidence = {
                        "intervention_plan_sha256": (
                            scene.pair.intervention_plan.plan_fingerprint
                        ),
                        "world_tensor_equal": passed,
                    }
                elif gate_name == "sham_legacy_equality":
                    assert policy_name is not None
                    scene = primary[(cohort_index, policy_name)]
                    passed = _sham_legacy_holds(scene)
                    evidence = {
                        "identity_plan_sha256": (
                            scene.pair.baseline_plan.plan_fingerprint
                        ),
                        "sham_legacy_equal": passed,
                    }
                elif gate_name == "synchronous_response_floor":
                    assert policy_name is not None
                    scene = primary[(cohort_index, policy_name)]
                    passed = _synchronous_floor_holds(scene)
                    evidence = {
                        "intervention_plan_sha256": (
                            scene.pair.intervention_plan.plan_fingerprint
                        ),
                        "synchronous_floor_holds": passed,
                    }
                elif gate_name == "primary_plan_feasibility":
                    passed = _primary_plan_feasible(reference)
                    evidence = {
                        "plan_audit_sha256": (
                            reference.pair.intervention_plan.audit_fingerprint
                        ),
                        "plan_feasible": passed,
                    }
                else:
                    severe = secondary[(cohort_index, "log_replay")]
                    passed = _nested_plans_monotone(reference, severe)
                    evidence = {
                        "primary_plan_sha256": (
                            reference.pair.intervention_plan.plan_fingerprint
                        ),
                        "secondary_plan_sha256": (
                            severe.pair.intervention_plan.plan_fingerprint
                        ),
                        "nested_dose_monotone": passed,
                    }
                rows.append(
                    _negative_row(
                        gate_name=gate_name,
                        cohort_index=cohort_index,
                        policy_name=policy_name,
                        violation=not passed,
                        evidence=evidence,
                    )
                )
    expected_n = result.eligibility_ledger.eligible_n * 9 + len(feasible)
    if len(rows) != expected_n:
        raise M6OfficialAdapterError(
            "m6_official_negative_observation_domain_drifted"
        )
    return _frozen_rows(rows, fields=_NEGATIVE_FIELDS)


_NUMPY_ROWS_ISSUANCE_DOMAIN = b"evalsim-m6-official-numpy-rows-issuance-v1"


@dataclass(frozen=True, slots=True)
class _M6OfficialNumpyRowsIssuance:
    evidence: Any
    eligibility_rows: object
    primary_scene_scalar_rows: object
    primary_repeat_scene_scalar_rows: object
    secondary_scene_scalar_rows: object
    negative_timing_observation_rows: object
    phase_durations_ms: object
    typed_result: M6EvaluationResult
    adapter_version: str
    issuance_sha256: str


_NUMPY_ROWS_ISSUANCE_LOCK = threading.Lock()
_NUMPY_ROWS_ISSUANCE_REGISTRY: dict[int, _M6OfficialNumpyRowsIssuance] = {}


def _m6_official_numpy_rows_sha256(value: "M6OfficialNumpyRows") -> str:
    digest = hashlib.sha256()
    digest.update(_NUMPY_ROWS_ISSUANCE_DOMAIN)
    digest.update(b"\x00")
    adapter = value.adapter_version.encode("ascii")
    digest.update(len(adapter).to_bytes(4, "big"))
    digest.update(adapter)
    phase_payload = _canonical_json_bytes(dict(value.phase_durations_ms))
    digest.update(len(phase_payload).to_bytes(8, "big"))
    digest.update(phase_payload)
    digest.update(id(value.typed_result).to_bytes(16, "big"))
    for entry in value.typed_result.eligibility_ledger.entries:
        entry.revalidate()
        digest.update(entry.cohort_index.to_bytes(4, "big"))
        digest.update(bytes.fromhex(entry.source_snapshot._integrity_fingerprint))
        eligibility = _canonical_json_bytes(entry.eligibility.to_dict())
        digest.update(len(eligibility).to_bytes(8, "big"))
        digest.update(eligibility)
    collections = (
        ("eligibility", value.eligibility_rows),
        ("primary", value.primary_scene_scalar_rows),
        ("primary_repeat", value.primary_repeat_scene_scalar_rows),
        ("secondary", value.secondary_scene_scalar_rows),
        ("negative", value.negative_timing_observation_rows),
    )
    for label, rows in collections:
        encoded_label = label.encode("ascii")
        digest.update(len(encoded_label).to_bytes(4, "big"))
        digest.update(encoded_label)
        digest.update(len(rows).to_bytes(8, "big"))
        for row in rows:
            payload = _canonical_json_bytes(dict(row))
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _validate_m6_official_numpy_rows_semantics(
    value: "M6OfficialNumpyRows",
) -> None:
    if value.adapter_version != M6_OFFICIAL_ADAPTER_VERSION:
        raise M6OfficialAdapterError("m6_official_numpy_rows_mutated")
    if not isinstance(value.typed_result, M6EvaluationResult):
        raise M6OfficialAdapterError("m6_official_numpy_rows_mutated")
    value.typed_result.revalidate()
    durations = dict(value.phase_durations_ms)
    if set(durations) != set(_NUMPY_PHASES) or any(
        type(duration) is not int or duration <= 0
        for duration in durations.values()
    ):
        raise M6OfficialAdapterError("m6_official_numpy_rows_mutated")
    specifications = (
        ("eligibility_rows", _ELIGIBILITY_FIELDS),
        ("primary_scene_scalar_rows", _SCENE_FIELDS),
        ("primary_repeat_scene_scalar_rows", _SCENE_FIELDS),
        ("secondary_scene_scalar_rows", _SCENE_FIELDS),
        ("negative_timing_observation_rows", _NEGATIVE_FIELDS),
    )
    for name, fields in specifications:
        rows = getattr(value, name)
        if not isinstance(rows, tuple):
            raise M6OfficialAdapterError("m6_official_numpy_rows_mutated")
        normalized = _frozen_rows(rows, fields=fields)
        if tuple(dict(row) for row in rows) != tuple(
            dict(row) for row in normalized
        ):
            raise M6OfficialAdapterError("m6_official_numpy_rows_mutated")
    expected = (
        m6_eligibility_rows(
            value.typed_result.eligibility_ledger,
            mode=_OFFICIAL_MODE,
            secondary_plan_ledger=value.typed_result.secondary_plan_ledger,
        ),
        m6_primary_scene_scalar_rows(value.typed_result),
        m6_primary_scene_scalar_rows(value.typed_result),
        m6_secondary_scene_scalar_rows(value.typed_result),
        m6_negative_timing_observation_rows(value.typed_result),
    )
    actual = tuple(
        tuple(dict(row) for row in getattr(value, name))
        for name, _fields in specifications
    )
    normalized_expected = tuple(
        tuple(dict(row) for row in rows) for rows in expected
    )
    if actual != normalized_expected:
        raise M6OfficialAdapterError(
            "m6_official_numpy_rows_do_not_match_typed_result"
        )


@dataclass(frozen=True, slots=True)
class M6OfficialNumpyRows:
    """Factory-issued immutable store inputs from two complete NumPy passes."""

    eligibility_rows: tuple[Mapping[str, Any], ...]
    primary_scene_scalar_rows: tuple[Mapping[str, Any], ...]
    primary_repeat_scene_scalar_rows: tuple[Mapping[str, Any], ...]
    secondary_scene_scalar_rows: tuple[Mapping[str, Any], ...]
    negative_timing_observation_rows: tuple[Mapping[str, Any], ...]
    phase_durations_ms: Mapping[str, int]
    typed_result: M6EvaluationResult = field(repr=False, compare=False)
    adapter_version: str = field(default=M6_OFFICIAL_ADAPTER_VERSION)
    issuance_sha256: str | None = field(default=None, repr=False)
    _issued_original_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _NUMPY_ROWS_ISSUER:
            raise TypeError("M6OfficialNumpyRows is factory-issued only")
        object.__setattr__(
            self,
            "phase_durations_ms",
            MappingProxyType(dict(self.phase_durations_ms)),
        )
        specifications = (
            ("eligibility_rows", _ELIGIBILITY_FIELDS),
            ("primary_scene_scalar_rows", _SCENE_FIELDS),
            ("primary_repeat_scene_scalar_rows", _SCENE_FIELDS),
            ("secondary_scene_scalar_rows", _SCENE_FIELDS),
            ("negative_timing_observation_rows", _NEGATIVE_FIELDS),
        )
        for name, fields in specifications:
            object.__setattr__(
                self,
                name,
                _frozen_rows(getattr(self, name), fields=fields),
            )
        _validate_m6_official_numpy_rows_semantics(self)
        expected = _m6_official_numpy_rows_sha256(self)
        if self.issuance_sha256 is not None and self.issuance_sha256 != expected:
            raise ValueError("issuance_sha256 does not bind NumPy rows")
        object.__setattr__(self, "issuance_sha256", expected)
        object.__setattr__(self, "_issued_original_sha256", expected)
        record = _M6OfficialNumpyRowsIssuance(
            evidence=self,
            eligibility_rows=self.eligibility_rows,
            primary_scene_scalar_rows=self.primary_scene_scalar_rows,
            primary_repeat_scene_scalar_rows=(
                self.primary_repeat_scene_scalar_rows
            ),
            secondary_scene_scalar_rows=self.secondary_scene_scalar_rows,
            negative_timing_observation_rows=(
                self.negative_timing_observation_rows
            ),
            phase_durations_ms=self.phase_durations_ms,
            typed_result=self.typed_result,
            adapter_version=self.adapter_version,
            issuance_sha256=expected,
        )
        with _NUMPY_ROWS_ISSUANCE_LOCK:
            if id(self) in _NUMPY_ROWS_ISSUANCE_REGISTRY:
                raise RuntimeError("NumPy rows issuance identity was reused")
            _NUMPY_ROWS_ISSUANCE_REGISTRY[id(self)] = record

    def revalidate(self) -> None:
        with _NUMPY_ROWS_ISSUANCE_LOCK:
            record = _NUMPY_ROWS_ISSUANCE_REGISTRY.get(id(self))
        if (
            record is None
            or record.evidence is not self
            or self.eligibility_rows is not record.eligibility_rows
            or self.primary_scene_scalar_rows
            is not record.primary_scene_scalar_rows
            or self.primary_repeat_scene_scalar_rows
            is not record.primary_repeat_scene_scalar_rows
            or self.secondary_scene_scalar_rows
            is not record.secondary_scene_scalar_rows
            or self.negative_timing_observation_rows
            is not record.negative_timing_observation_rows
            or self.phase_durations_ms is not record.phase_durations_ms
            or self.typed_result is not record.typed_result
            or self.adapter_version != record.adapter_version
        ):
            raise M6OfficialAdapterError("m6_official_numpy_rows_mutated")
        _validate_m6_official_numpy_rows_semantics(self)
        expected = _m6_official_numpy_rows_sha256(self)
        if (
            expected != record.issuance_sha256
            or self.issuance_sha256 != expected
            or self._issued_original_sha256 != expected
        ):
            raise M6OfficialAdapterError("m6_official_numpy_rows_mutated")



def _snapshot_official_cases(
    cases: Iterable[M6EvaluationCase],
) -> tuple[tuple[int, ScenarioSnapshot], ...]:
    normalized = tuple(cases)
    if len(normalized) != M6_OFFICIAL_POPULATION_SIZE:
        raise M6OfficialAdapterError("m6_official_requires_128_cases")
    snapshots: list[tuple[int, ScenarioSnapshot]] = []
    for expected_index, case in enumerate(normalized):
        if not isinstance(case, M6EvaluationCase):
            raise TypeError("cases must contain M6EvaluationCase values")
        if case.cohort_index != expected_index:
            raise M6OfficialAdapterError(
                "m6_official_cases_not_in_canonical_order"
            )
        snapshot = ScenarioSnapshot.from_scenario(case.scenario)
        snapshot.revalidate()
        snapshots.append((expected_index, snapshot))
    return tuple(snapshots)


def _materialize_cases(
    snapshots: Sequence[tuple[int, ScenarioSnapshot]],
) -> tuple[M6EvaluationCase, ...]:
    cases: list[M6EvaluationCase] = []
    for cohort_index, snapshot in snapshots:
        snapshot.revalidate()
        cases.append(
            M6EvaluationCase(
                cohort_index=cohort_index,
                scenario=snapshot.to_scenario(),
            )
        )
    return tuple(cases)


def run_m6_official_numpy(
    cases: Iterable[M6EvaluationCase],
) -> M6OfficialNumpyRows:
    """Run two independent complete official NumPy passes and issue safe rows.

    This helper intentionally stops before any store write, Waymax execution, review,
    provenance, or terminal lifecycle action.  Pass two is materialized from immutable
    source snapshots rather than reusing pass-one case objects.
    """

    wall_started = time.monotonic_ns()
    snapshots = _snapshot_official_cases(cases)
    phase_ns = {name: 0 for name in _NUMPY_EXECUTION_PHASES}

    def observe(name: str, duration_ns: int) -> None:
        if name not in phase_ns or type(duration_ns) is not int or duration_ns <= 0:
            raise M6OfficialAdapterError("m6_official_phase_timing_invalid")
        phase_ns[name] += duration_ns

    first = _run_m6_numpy_evaluation_with_phase_observer(
        _materialize_cases(snapshots),
        observe,
        time.monotonic_ns,
    )
    second = _run_m6_numpy_evaluation_with_phase_observer(
        _materialize_cases(snapshots),
        observe,
        time.monotonic_ns,
    )
    first.revalidate()
    second.revalidate()

    first_eligibility = m6_eligibility_rows(
        first.eligibility_ledger,
        mode=_OFFICIAL_MODE,
        secondary_plan_ledger=first.secondary_plan_ledger,
    )
    second_eligibility = m6_eligibility_rows(
        second.eligibility_ledger,
        mode=_OFFICIAL_MODE,
        secondary_plan_ledger=second.secondary_plan_ledger,
    )
    first_primary = m6_primary_scene_scalar_rows(first)
    second_primary = m6_primary_scene_scalar_rows(second)
    first_secondary = m6_secondary_scene_scalar_rows(first)
    second_secondary = m6_secondary_scene_scalar_rows(second)
    first_negative = m6_negative_timing_observation_rows(first)
    second_negative = m6_negative_timing_observation_rows(second)
    if (
        first_eligibility != second_eligibility
        or first_primary != second_primary
        or first_secondary != second_secondary
        or first_negative != second_negative
    ):
        raise M6OfficialAdapterError(
            "m6_official_numpy_repeat_drifted"
        )
    wall_stopped = time.monotonic_ns()
    observed_ns = sum(phase_ns.values())
    verification_ns = wall_stopped - wall_started - observed_ns
    if verification_ns <= 0:
        raise M6OfficialAdapterError(
            "m6_official_phase_timing_did_not_cover_full_wall_interval"
        )
    complete_phase_ns = {
        **phase_ns,
        "verification": verification_ns,
    }
    phase_durations_ms = {
        name: (complete_phase_ns[name] + 999_999) // 1_000_000
        for name in _NUMPY_PHASES
    }
    return M6OfficialNumpyRows(
        eligibility_rows=first_eligibility,
        primary_scene_scalar_rows=first_primary,
        primary_repeat_scene_scalar_rows=second_primary,
        secondary_scene_scalar_rows=first_secondary,
        negative_timing_observation_rows=first_negative,
        phase_durations_ms=phase_durations_ms,
        typed_result=first,
        _issuance_capability=_NUMPY_ROWS_ISSUER,
    )


__all__ = [
    "M6_OFFICIAL_ADAPTER_VERSION",
    "M6_OFFICIAL_POPULATION_SIZE",
    "M6OfficialAdapterError",
    "M6OfficialCaseCollector",
    "M6OfficialNumpyRows",
    "m6_case_from_reloaded_member",
    "m6_eligibility_rows",
    "m6_negative_timing_observation_rows",
    "m6_primary_scene_scalar_rows",
    "m6_secondary_scene_scalar_rows",
    "run_m6_official_numpy",
]
