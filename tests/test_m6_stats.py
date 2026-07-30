"""Exact deterministic finite-cohort statistics tests for EvalSim M6."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import replace

import numpy as np
import pytest

import evalsim.stats.m6 as m6_stats
from evalsim.metrics.m6 import M6_PRIMARY_PAIRED_METRIC_SPECS
from evalsim.stats.m6 import (
    M6_ADJUSTED_REWEIGHTING_LEVEL,
    M6_BASE_SEED,
    M6_MAX_PRIMARY_DRAW_MATRIX_BYTES,
    M6_MAX_PRIMARY_PAIR_N,
    M6_MAX_SAMPLED_SCALARS_PER_CHUNK,
    M6_POINTWISE_REWEIGHTING_LEVEL,
    M6_PRIMARY_FAMILY_SIZE,
    M6_PRIMARY_METRICS,
    M6_PRIMARY_POLICY_ROLES,
    M6_PRIMARY_RESAMPLES,
    M6_REWEIGHTING_INTERPRETATION,
    M6_STATISTICS_SCHEMA_VERSION,
    M6ConditionalLatencySummary,
    M6PrimaryCellInput,
    M6PrimaryCellSpec,
    M6ResamplingKey,
    M6SceneEffect,
    analyze_m6_primary_cell,
    analyze_m6_primary_matrix,
    draw_m6_resample_indices,
    m6_primary_cell_specs,
    m6_resampled_means,
    make_m6_resampling_key,
)


_FINGERPRINT = hashlib.sha256(b"m6-primary-brake-b2").hexdigest()
_OTHER_FINGERPRINT = hashlib.sha256(b"wrong-primary-config").hexdigest()


def _spec(
    metric_name: str = "minimum_longitudinal_bumper_gap_change_m",
    *,
    policy_name: str = "constant_velocity",
    fingerprint: str = _FINGERPRINT,
) -> M6PrimaryCellSpec:
    role = {role.policy_name: role for role in M6_PRIMARY_POLICY_ROLES}[
        policy_name
    ]
    return M6PrimaryCellSpec(
        metric_name=metric_name,
        metric_version="1.0.0",
        policy_name=policy_name,
        policy_access_role=role.access_role,
        intervention_config_fingerprint=fingerprint,
    )


def _effects(values: list[float] | np.ndarray) -> tuple[M6SceneEffect, ...]:
    return tuple(
        M6SceneEffect(cohort_index=index, value=float(value))
        for index, value in enumerate(values)
    )


def _timeliness_effects(
    values: list[float] | np.ndarray,
    responded: list[bool] | np.ndarray,
) -> tuple[M6SceneEffect, ...]:
    return tuple(
        M6SceneEffect(
            cohort_index=index,
            value=float(value),
            responded=bool(did_respond),
            responder_latency_s=(0.5 + index / 10.0 if did_respond else None),
        )
        for index, (value, did_respond) in enumerate(
            zip(values, responded, strict=True)
        )
    )


def _matrix_inputs(
    *,
    n: int = 10,
    fingerprint: str = _FINGERPRINT,
) -> list[M6PrimaryCellInput]:
    cells: list[M6PrimaryCellInput] = []
    for spec in m6_primary_cell_specs(fingerprint):
        if spec.metric_name == "response_timeliness_s":
            rows = _timeliness_effects(
                np.ones(n),
                np.ones(n, dtype=bool),
            )
        else:
            rows = _effects(np.ones(n))
        cells.append(M6PrimaryCellInput(spec=spec, scene_effects=rows))
    return cells


def test_registered_schema_matches_metric_contracts_and_is_exactly_12() -> None:
    assert tuple(
        (metric.metric_name, metric.metric_version, metric.value_unit)
        for metric in M6_PRIMARY_METRICS
    ) == tuple(
        (spec.name, spec.version, spec.value_unit)
        for spec in M6_PRIMARY_PAIRED_METRIC_SPECS
    )
    assert tuple(
        (role.policy_name, role.access_role)
        for role in M6_PRIMARY_POLICY_ROLES
    ) == (
        ("log_replay", "privileged"),
        ("constant_velocity", "history_only"),
        ("idm", "history_only"),
    )
    specs = m6_primary_cell_specs(_FINGERPRINT)
    assert len(specs) == M6_PRIMARY_FAMILY_SIZE == 12
    assert tuple(
        (spec.policy_name, spec.metric_name) for spec in specs
    ) == tuple(
        (role.policy_name, metric.metric_name)
        for role in M6_PRIMARY_POLICY_ROLES
        for metric in M6_PRIMARY_METRICS
    )


def test_cell_spec_rejects_unregistered_or_tampered_identity() -> None:
    with pytest.raises(ValueError, match="unregistered M6 primary metric"):
        _spec("posthoc_winner_score")
    with pytest.raises(ValueError, match="unregistered M6 primary policy"):
        M6PrimaryCellSpec(
            metric_name="target_progress_loss_m",
            policy_name="oracle",
            policy_access_role="history_only",
            intervention_config_fingerprint=_FINGERPRINT,
        )
    with pytest.raises(ValueError, match="policy_access_role"):
        M6PrimaryCellSpec(
            metric_name="target_progress_loss_m",
            policy_name="log_replay",
            policy_access_role="history_only",
            intervention_config_fingerprint=_FINGERPRINT,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _spec(fingerprint="not-a-fingerprint")


def test_canonical_key_and_draws_match_independent_literal_oracle() -> None:
    spec = _spec()
    key = make_m6_resampling_key(spec, pair_n=10)
    expected_payload = {
        "base_seed": M6_BASE_SEED,
        "intervention_config_fingerprint": _FINGERPRINT,
        "metric_name": "minimum_longitudinal_bumper_gap_change_m",
        "metric_version": "1.0.0",
        "paired_n": 10,
        "policy_access_role": "history_only",
        "policy_name": "constant_velocity",
        "resamples": M6_PRIMARY_RESAMPLES,
        "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
    }
    expected_json = json.dumps(
        expected_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    digest = hashlib.sha256(expected_json.encode("utf-8")).digest()
    words = tuple(struct.unpack(">8I", digest))
    expected_rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([M6_BASE_SEED, *words])
        )
    )
    expected_draws = expected_rng.integers(
        0,
        10,
        size=(M6_PRIMARY_RESAMPLES, 10),
        dtype=np.int64,
    )

    assert key.canonical_json == expected_json
    assert key.sha256 == digest.hex()
    assert key.digest_words == words
    actual_draws = draw_m6_resample_indices(key)
    assert actual_draws.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(actual_draws, expected_draws)


def test_exact_vector_summary_quantiles_and_point_estimators_match_oracle() -> None:
    values = np.asarray(
        [-4.0, -1.5, -0.5, 0.0, 1.0, 1.25, 2.0, 3.0, 7.0, 12.0],
        dtype=np.float64,
    )
    result = analyze_m6_primary_cell(_spec(), _effects(values))

    payload = json.loads(result.resampling_key.canonical_json)
    digest = hashlib.sha256(
        result.resampling_key.canonical_json.encode("utf-8")
    ).digest()
    literal_words = struct.unpack(">8I", digest)
    literal_rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [payload["base_seed"], *literal_words]
            )
        )
    )
    literal_draws = literal_rng.integers(
        0,
        len(values),
        size=(100_000, len(values)),
        dtype=np.int64,
    )
    literal_means = np.mean(
        values[literal_draws],
        axis=1,
        dtype=np.float64,
    )
    adjusted_tail = (1.0 - M6_ADJUSTED_REWEIGHTING_LEVEL) / 2.0

    assert result.arithmetic_mean == (
        math.fsum(float(value) for value in values) / len(values)
    )
    assert result.median == math.fsum((1.0, 1.25)) / 2.0
    np.testing.assert_array_equal(
        [
            result.pointwise_band.lower,
            result.pointwise_band.upper,
        ],
        np.quantile(literal_means, [0.025, 0.975], method="linear"),
    )
    np.testing.assert_array_equal(
        [
            result.adjusted_band.lower,
            result.adjusted_band.upper,
        ],
        np.quantile(
            literal_means,
            [adjusted_tail, 1.0 - adjusted_tail],
            method="linear",
        ),
    )
    assert result.pointwise_band.level == M6_POINTWISE_REWEIGHTING_LEVEL
    assert result.adjusted_band.level == M6_ADJUSTED_REWEIGHTING_LEVEL
    assert (
        result.pointwise_band.interpretation
        == M6_REWEIGHTING_INTERPRETATION
    )


def test_row_order_is_canonicalized_before_arithmetic_and_resampling() -> None:
    rows = _effects(np.linspace(-2.0, 3.0, 30))
    forward = analyze_m6_primary_cell(_spec(), rows)
    reverse = analyze_m6_primary_cell(_spec(), reversed(rows))
    assert forward == reverse
    assert forward.to_local_dict() == reverse.to_local_dict()


def test_metric_specific_nonzero_rules_use_exact_strict_thresholds() -> None:
    impulse, timeliness, gap, progress = M6_PRIMARY_METRICS
    assert not impulse.is_thresholded_nonzero(-1.0)
    assert not impulse.is_thresholded_nonzero(1e-9)
    assert impulse.is_thresholded_nonzero(np.nextafter(1e-9, math.inf))
    assert not timeliness.is_thresholded_nonzero(1e-9)
    assert timeliness.is_thresholded_nonzero(np.nextafter(1e-9, math.inf))
    for metric in (gap, progress):
        assert not metric.is_thresholded_nonzero(1e-6)
        assert not metric.is_thresholded_nonzero(-1e-6)
        assert metric.is_thresholded_nonzero(
            np.nextafter(1e-6, math.inf)
        )
        assert metric.is_thresholded_nonzero(
            np.nextafter(-1e-6, -math.inf)
        )


@pytest.mark.parametrize(
    "n,values,expected_status,expected_sign",
    [
        (30, [1.0] * 9 + [0.0] * 21, "event_sparse", None),
        (10, [1.0] * 10, "small_n", None),
        (30, [-1.0, 1.0] * 15, "descriptive", None),
        (30, [1.0] * 30, "direction_supported", "positive"),
        (30, [-1.0] * 30, "direction_supported", "negative"),
    ],
)
def test_frozen_status_priority_preserves_sparse_null_and_adverse_signs(
    n: int,
    values: list[float],
    expected_status: str,
    expected_sign: str | None,
) -> None:
    assert len(values) == n
    result = analyze_m6_primary_cell(_spec(), _effects(values))
    assert result.status == expected_status
    assert result.directional_effect_sign == expected_sign
    assert result.directional_language_allowed is (
        expected_status == "direction_supported"
    )
    if expected_sign == "negative":
        assert result.arithmetic_mean < 0.0
        assert result.adjusted_band.upper < 0.0
    if expected_status == "descriptive":
        assert result.adjusted_band.lower <= 0.0 <= result.adjusted_band.upper


def test_responder_censor_fields_and_conditional_latency_suppression() -> None:
    values = np.asarray([1.0] * 9 + [0.0] * 21)
    responded = np.asarray([True] * 9 + [False] * 21)
    sparse = analyze_m6_primary_cell(
        _spec("response_timeliness_s"),
        _timeliness_effects(values, responded),
    )
    assert sparse.responder_n == 9
    assert sparse.censor_n == 21
    assert sparse.conditional_responder_latency is not None
    assert sparse.conditional_responder_latency.status == "responder_sparse"
    assert sparse.conditional_responder_latency.arithmetic_mean_s is None
    assert sparse.conditional_responder_latency.median_s is None

    # An event exactly at W remains a responder even though its primary scalar is zero.
    values = np.asarray([1.0] * 9 + [0.0] * 21)
    responded = np.asarray([True] * 10 + [False] * 20)
    complete = analyze_m6_primary_cell(
        _spec("response_timeliness_s"),
        _timeliness_effects(values, responded),
    )
    assert complete.responder_n == 10
    assert complete.censor_n == 20
    assert complete.thresholded_nonzero_n == 9
    assert complete.conditional_responder_latency is not None
    assert complete.conditional_responder_latency.status == "descriptive"
    assert complete.conditional_responder_latency.arithmetic_mean_s is not None
    assert complete.conditional_responder_latency.median_s is not None

    non_latency = analyze_m6_primary_cell(
        _spec("target_progress_loss_m"),
        _effects(np.ones(30)),
    )
    assert non_latency.responder_n is None
    assert non_latency.censor_n is None
    assert non_latency.conditional_responder_latency is None
    promoted = non_latency.to_promoted_dict()
    assert promoted["responder_n"] is None
    assert promoted["censor_n"] is None


def test_incomplete_pairing_low_n_and_bad_response_accounting_fail_closed() -> None:
    with pytest.raises(ValueError, match="N < 10"):
        analyze_m6_primary_cell(_spec(), _effects(np.ones(9)))
    with pytest.raises(ValueError, match="structural/pairing defect"):
        analyze_m6_primary_cell(
            _spec(),
            _effects(np.ones(10)),
            source_pairing_complete=False,
        )
    with pytest.raises(ValueError, match="explicit responded"):
        M6PrimaryCellInput(
            spec=_spec("response_timeliness_s"),
            scene_effects=_effects(np.ones(10)),
        )
    with pytest.raises(ValueError, match="censored timeliness effect"):
        M6PrimaryCellInput(
            spec=_spec("response_timeliness_s"),
            scene_effects=tuple(
                M6SceneEffect(index, 1.0, responded=False)
                for index in range(10)
            ),
        )
    with pytest.raises(ValueError, match="only valid"):
        M6PrimaryCellInput(
            spec=_spec(),
            scene_effects=tuple(
                M6SceneEffect(index, 1.0, responded=True, responder_latency_s=1.0)
                for index in range(10)
            ),
        )


def test_complete_matrix_is_order_independent_but_output_is_canonical() -> None:
    cells = _matrix_inputs()
    result = analyze_m6_primary_matrix(reversed(cells))
    assert len(result.rows) == 12
    assert result.pair_n == 10
    assert tuple(row.spec for row in result.rows) == m6_primary_cell_specs(
        _FINGERPRINT
    )
    assert all(row.source_pairing_flag for row in result.rows)
    assert result.evalsim_idm_response_event_expectation_met is True
    promoted = result.to_promoted_rows()
    assert len(promoted) == 12
    assert all(
        "intervention_config_fingerprint" not in row
        and "sha256" not in row
        and "canonical_cell_key" not in row
        for row in promoted
    )
    required = {
        "adjusted_band",
        "arithmetic_mean",
        "censor_n",
        "directional_language_allowed",
        "median",
        "metric_name",
        "metric_version",
        "pair_n",
        "pointwise_band",
        "policy_access_role",
        "policy_name",
        "responder_n",
        "source_pairing_complete",
        "status",
        "suppression_reason",
        "thresholded_nonzero_n",
        "unit",
    }
    assert all(set(row) == required for row in promoted)


def test_matrix_rejects_missing_duplicate_fingerprint_and_cohort_drift_pre_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = _matrix_inputs()

    def forbidden_draw(key: M6ResamplingKey) -> np.ndarray:
        raise AssertionError("matrix validation must finish before drawing")

    monkeypatch.setattr(m6_stats, "draw_m6_resample_indices", forbidden_draw)
    with pytest.raises(ValueError, match="exactly 12"):
        analyze_m6_primary_matrix(cells[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        analyze_m6_primary_matrix([*cells[:-1], cells[0]])

    fingerprint_drift = list(cells)
    last = fingerprint_drift[-1]
    fingerprint_drift[-1] = M6PrimaryCellInput(
        spec=_spec(
            last.spec.metric_name,
            policy_name=last.spec.policy_name,
            fingerprint=_OTHER_FINGERPRINT,
        ),
        scene_effects=last.scene_effects,
    )
    with pytest.raises(ValueError, match="one intervention fingerprint"):
        analyze_m6_primary_matrix(fingerprint_drift)

    cohort_drift = list(cells)
    last = cohort_drift[-1]
    changed_rows = list(last.scene_effects)
    changed_rows[-1] = M6SceneEffect(cohort_index=99, value=1.0)
    cohort_drift[-1] = M6PrimaryCellInput(
        spec=last.spec,
        scene_effects=tuple(changed_rows),
    )
    with pytest.raises(ValueError, match="same complete cohort"):
        analyze_m6_primary_matrix(cohort_drift)


def test_max_draw_matrix_bound_and_chunk_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert M6_MAX_PRIMARY_DRAW_MATRIX_BYTES == 102_400_000
    max_key = make_m6_resampling_key(
        _spec(),
        pair_n=M6_MAX_PRIMARY_PAIR_N,
    )
    max_draws = draw_m6_resample_indices(max_key)
    assert max_draws.shape == (
        M6_PRIMARY_RESAMPLES,
        M6_MAX_PRIMARY_PAIR_N,
    )
    assert max_draws.dtype == np.dtype(np.int64)
    assert max_draws.nbytes == M6_MAX_PRIMARY_DRAW_MATRIX_BYTES
    del max_draws

    effects = np.arange(4, dtype=np.float64)
    draws = np.tile(
        np.arange(4, dtype=np.int64),
        (250_001, 1),
    )
    expected = np.full(250_001, 1.5)
    original_mean = np.mean
    sampled_sizes: list[int] = []

    def recording_mean(values: np.ndarray, *args, **kwargs):
        sampled_sizes.append(values.size)
        return original_mean(values, *args, **kwargs)

    monkeypatch.setattr(m6_stats.np, "mean", recording_mean)
    actual = m6_resampled_means(effects, draws)
    np.testing.assert_array_equal(actual, expected)
    assert sampled_sizes == [M6_MAX_SAMPLED_SCALARS_PER_CHUNK, 4]


def test_resampling_key_detects_constructor_and_postconstruction_tampering() -> None:
    key = make_m6_resampling_key(_spec(), pair_n=10)
    with pytest.raises(ValueError, match="sha256"):
        M6ResamplingKey(
            canonical_json=key.canonical_json,
            sha256="0" * 64,
            digest_words=key.digest_words,
            pair_n=key.pair_n,
        )

    payload = json.loads(key.canonical_json)
    payload["unregistered"] = "posthoc"
    tampered_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(tampered_json.encode()).digest()
    with pytest.raises(ValueError, match="fields"):
        M6ResamplingKey(
            canonical_json=tampered_json,
            sha256=digest.hex(),
            digest_words=struct.unpack(">8I", digest),
            pair_n=10,
        )

    numeric_type_drift = json.loads(key.canonical_json)
    numeric_type_drift["paired_n"] = 10.0
    numeric_type_json = json.dumps(
        numeric_type_drift,
        sort_keys=True,
        separators=(",", ":"),
    )
    numeric_type_digest = hashlib.sha256(numeric_type_json.encode()).digest()
    with pytest.raises(ValueError, match="exact JSON integer"):
        M6ResamplingKey(
            canonical_json=numeric_type_json,
            sha256=numeric_type_digest.hex(),
            digest_words=struct.unpack(">8I", numeric_type_digest),
            pair_n=10,
        )

    object.__setattr__(key, "digest_words", (0,) * 8)
    with pytest.raises(ValueError, match="digest_words"):
        draw_m6_resample_indices(key)


@pytest.mark.parametrize(("mean", "median"), [(-0.1, 0.0), (0.0, -0.1)])
def test_conditional_latency_summary_rejects_negative_values(
    mean: float,
    median: float,
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        M6ConditionalLatencySummary(
            responder_n=10,
            censor_n=0,
            status="descriptive",
            arithmetic_mean_s=mean,
            median_s=median,
        )


def test_result_rejects_a_valid_key_from_a_different_cell() -> None:
    result = analyze_m6_primary_cell(_spec(), _effects(np.ones(10)))
    other_key = make_m6_resampling_key(
        _spec("target_progress_loss_m"),
        pair_n=10,
    )
    with pytest.raises(ValueError, match="result cell identity"):
        replace(result, resampling_key=other_key)


def test_every_declared_cell_key_dimension_changes_the_stream() -> None:
    base = make_m6_resampling_key(_spec(), pair_n=10)
    variants = (
        make_m6_resampling_key(
            _spec("target_progress_loss_m"),
            pair_n=10,
        ),
        make_m6_resampling_key(
            _spec(policy_name="idm"),
            pair_n=10,
        ),
        make_m6_resampling_key(
            _spec(fingerprint=_OTHER_FINGERPRINT),
            pair_n=10,
        ),
        make_m6_resampling_key(_spec(), pair_n=11),
    )
    assert len({base.sha256, *(key.sha256 for key in variants)}) == 5
