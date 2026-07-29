"""M5 paired finite-cohort statistics and deterministic resampling tests."""
from __future__ import annotations

import hashlib
import json
import math
import struct

import numpy as np
import pytest

from evalsim.metrics import M5_METRIC_SPECS
from evalsim.slices import M5_SLICE_NAMES as CANONICAL_SLICE_NAMES
from evalsim.stats import (
    CONSTANT_VELOCITY_MINUS_LOG_REPLAY,
    IDM_MINUS_CONSTANT_VELOCITY,
    IDM_MINUS_LOG_REPLAY,
    M5_BASE_SEED,
    M5_OTHER_RESAMPLES,
    M5_POLICY_CONTRASTS,
    M5_PRIMARY_ADJUSTED_STABILITY_LEVEL,
    M5_PRIMARY_FAMILY_SIZE,
    M5_PRIMARY_METRIC_NAMES,
    M5_PRIMARY_RESAMPLES,
    M5_SLICE_NAMES as STATS_SLICE_NAMES,
    M5_STATISTICS_SCHEMA_VERSION,
    PairedCellSpec,
    PolicyContrast,
    ScenarioScalar,
    analyze_paired_cell,
    draw_resample_indices,
    make_resampling_key,
)


def _spec(
    metric_name: str = "acceleration_error_mps2",
    *,
    slice_name: str = "all",
    contrast=CONSTANT_VELOCITY_MINUS_LOG_REPLAY,
    metric_version: str | None = None,
) -> PairedCellSpec:
    if metric_version is None:
        metric_spec = M5_METRIC_SPECS.get(metric_name)
        metric_version = metric_spec.version if metric_spec is not None else "1.0.0"
    return PairedCellSpec(
        metric_name=metric_name,
        slice_name=slice_name,
        contrast=contrast,
        metric_version=metric_version,
    )


def _paired_rows(
    effects: list[float] | np.ndarray,
) -> tuple[list[ScenarioScalar], list[ScenarioScalar]]:
    rows_a: list[ScenarioScalar] = []
    rows_b: list[ScenarioScalar] = []
    for index, effect in enumerate(effects):
        baseline = 100.0 + index
        rows_a.append(
            ScenarioScalar(
                index,
                baseline + float(effect),
                eligible_components=2,
                total_components=3,
            )
        )
        rows_b.append(
            ScenarioScalar(
                index,
                baseline,
                eligible_components=2,
                total_components=3,
            )
        )
    return rows_a, rows_b


def test_frozen_contrasts_and_primary_family_have_exact_scope() -> None:
    assert M5_POLICY_CONTRASTS == (
        PolicyContrast("constant_velocity", "log_replay"),
        PolicyContrast("idm", "log_replay"),
        PolicyContrast("idm", "constant_velocity"),
    )
    assert IDM_MINUS_LOG_REPLAY.label == "idm - log_replay"
    assert (
        IDM_MINUS_CONSTANT_VELOCITY.label
        == "idm - constant_velocity"
    )
    assert len(M5_PRIMARY_METRIC_NAMES) * len(M5_POLICY_CONTRASTS) == 12
    assert M5_PRIMARY_FAMILY_SIZE == 12


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"cohort_index": True, "value": 1.0}, TypeError),
        ({"cohort_index": -1, "value": 1.0}, ValueError),
        ({"cohort_index": 0, "value": float("nan")}, ValueError),
        ({"cohort_index": 0, "value": float("inf")}, ValueError),
        ({"cohort_index": 0, "value": True}, TypeError),
        (
            {
                "cohort_index": 0,
                "value": 1.0,
                "invalid_reason": "not_allowed",
            },
            ValueError,
        ),
        (
            {
                "cohort_index": 0,
                "value": 1.0,
                "eligible_components": 0,
            },
            ValueError,
        ),
        ({"cohort_index": 0, "value": None}, ValueError),
        (
            {
                "cohort_index": 0,
                "value": None,
                "invalid_reason": "not a reason",
                "eligible_components": 0,
            },
            ValueError,
        ),
        (
            {
                "cohort_index": 0,
                "value": None,
                "invalid_reason": "missing",
                "eligible_components": 1,
            },
            ValueError,
        ),
        (
            {
                "cohort_index": 0,
                "value": 1.0,
                "eligible_components": 2,
                "total_components": 1,
            },
            ValueError,
        ),
    ],
)
def test_scenario_scalar_fails_closed(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ScenarioScalar(**kwargs)


def test_scenario_scalar_missing_constructor_is_explicit() -> None:
    row = ScenarioScalar.missing(
        np.int64(4),
        "no_supported_lane",
        total_components=3,
    )
    assert row.cohort_index == 4
    assert row.value is None
    assert row.invalid_reason == "no_supported_lane"
    assert row.eligible_components == 0
    assert row.total_components == 3
    assert row.valid is False


def test_cell_spec_rejects_unregistered_semantics() -> None:
    with pytest.raises(ValueError, match="unregistered M5 metric"):
        _spec("made_up")
    with pytest.raises(ValueError, match="unregistered M5 slice"):
        _spec(slice_name="post_result_winner")
    with pytest.raises(ValueError, match="three frozen"):
        _spec(contrast=PolicyContrast("log_replay", "idm"))


def test_resampling_key_and_draws_match_exact_frozen_algorithm() -> None:
    spec = _spec()
    key = make_resampling_key(spec, paired_n=3, resamples=4)
    expected_payload = {
        "metric_name": "acceleration_error_mps2",
        "metric_version": "1.0.0",
        "ordered_policy_pair": ["constant_velocity", "log_replay"],
        "paired_n": 3,
        "resamples": 4,
        "slice_name": "all",
        "slice_version": "m5-womd-slices-1.0.0",
        "statistics_schema_version": M5_STATISTICS_SCHEMA_VERSION,
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
            np.random.SeedSequence([M5_BASE_SEED, *words])
        )
    )
    expected_draws = expected_rng.integers(
        0,
        3,
        size=(4, 3),
        dtype=np.int64,
    )

    assert key.canonical_json == expected_json
    assert key.sha256 == digest.hex()
    assert key.digest_words == words
    draws = draw_resample_indices(key)
    assert draws.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(draws, expected_draws)


def test_cell_key_changes_for_every_declared_stream_dimension() -> None:
    base = make_resampling_key(_spec(), paired_n=30, resamples=20)
    variants = (
        make_resampling_key(
            _spec(metric_name="jerk_error_mps3"),
            paired_n=30,
            resamples=20,
        ),
        make_resampling_key(
            _spec(slice_name="vru_present_current"),
            paired_n=30,
            resamples=20,
        ),
        make_resampling_key(
            _spec(contrast=IDM_MINUS_LOG_REPLAY),
            paired_n=30,
            resamples=20,
        ),
        make_resampling_key(_spec(), paired_n=31, resamples=20),
        make_resampling_key(_spec(), paired_n=30, resamples=21),
    )
    assert len({base.sha256, *(item.sha256 for item in variants)}) == 6


def test_raw_oriented_effects_favorable_ties_and_exact_summaries() -> None:
    effects = np.array([-1.0] * 10 + [0.0] * 10 + [2.0] * 10)
    rows_a, rows_b = _paired_rows(effects)
    result = analyze_paired_cell(_spec(), rows_a, rows_b)

    assert result.paired_n == 30
    assert result.raw_mean_difference == pytest.approx(1.0 / 3.0)
    assert result.raw_median_difference == 0.0
    assert result.oriented_mean_advantage == pytest.approx(-1.0 / 3.0)
    assert result.favorable_proportion == 0.5
    assert result.nonzero_effect_n == 20
    assert result.standardized_signal_to_heterogeneity is not None
    assert result.policy_a_mean == pytest.approx(
        math.fsum(row.value for row in rows_a) / 30
    )
    assert result.policy_b_mean == pytest.approx(
        math.fsum(row.value for row in rows_b) / 30
    )
    assert result.status == "descriptive"
    assert result.pointwise_stability_band.lower < 0.0
    assert result.pointwise_stability_band.upper > 0.0
    assert result.directional_language_allowed is False


def test_exact_tie_rule_does_not_treat_nearby_float_as_tie() -> None:
    effects = np.zeros(30)
    effects[0] = np.nextafter(100.0, math.inf) - 100.0
    effects[1:10] = -1.0
    rows_a, rows_b = _paired_rows(effects)
    result = analyze_paired_cell(_spec(), rows_a, rows_b)
    # Lower is favorable: nine negative effects win; 20 exact zeros tie.
    assert result.favorable_proportion == (9 + 0.5 * 20) / 30
    assert result.nonzero_effect_n == 10


def test_neutral_metric_has_no_oriented_or_favorable_outputs() -> None:
    rows_a, rows_b = _paired_rows(np.linspace(-2.0, 2.0, 30))
    result = analyze_paired_cell(
        _spec(metric_name="minimum_center_distance_m"),
        rows_a,
        rows_b,
    )
    assert result.raw_mean_difference is not None
    assert result.oriented_mean_advantage is None
    assert result.favorable_proportion is None
    assert result.standardized_signal_to_heterogeneity is None
    assert result.directional_language_allowed is False


def test_primary_cell_uses_100k_draws_and_both_frozen_levels() -> None:
    rows_a, rows_b = _paired_rows(np.full(30, 2.0))
    result = analyze_paired_cell(
        _spec(metric_name="position_error_m"),
        rows_a,
        rows_b,
    )
    assert result.resampling_key.resamples == M5_PRIMARY_RESAMPLES
    assert result.pointwise_stability_band is not None
    assert result.pointwise_stability_band.level == 0.95
    assert result.pointwise_stability_band.lower == 2.0
    assert result.pointwise_stability_band.upper == 2.0
    assert result.adjusted_primary_stability_band is not None
    assert (
        result.adjusted_primary_stability_band.level
        == M5_PRIMARY_ADJUSTED_STABILITY_LEVEL
    )
    assert result.adjusted_primary_stability_band.lower == 2.0
    assert result.adjusted_primary_stability_band.upper == 2.0
    assert result.standardized_signal_to_heterogeneity is None


def test_primary_percentiles_use_the_one_frozen_draw_matrix() -> None:
    effects = np.arange(30, dtype=float)
    rows_a, rows_b = _paired_rows(effects)
    result = analyze_paired_cell(
        _spec(metric_name="speed_error_mps"),
        rows_a,
        rows_b,
    )
    draws = draw_resample_indices(result.resampling_key)
    means = np.mean(effects[draws], axis=1, dtype=np.float64)
    expected_pointwise = np.quantile(
        means,
        [0.025, 0.975],
        method="linear",
    )
    adjusted_tail = (1.0 - M5_PRIMARY_ADJUSTED_STABILITY_LEVEL) / 2.0
    expected_adjusted = np.quantile(
        means,
        [adjusted_tail, 1.0 - adjusted_tail],
        method="linear",
    )

    assert result.pointwise_stability_band.lower == expected_pointwise[0]
    assert result.pointwise_stability_band.upper == expected_pointwise[1]
    assert (
        result.adjusted_primary_stability_band.lower
        == expected_adjusted[0]
    )
    assert (
        result.adjusted_primary_stability_band.upper
        == expected_adjusted[1]
    )


@pytest.mark.parametrize(
    "n, expected_status, effect_suppressed",
    [
        (9, "insufficient_n", True),
        (10, "small_or_sparse", False),
        (29, "small_or_sparse", False),
        (30, "descriptive", False),
    ],
)
def test_exact_small_n_boundaries(
    n: int,
    expected_status: str,
    effect_suppressed: bool,
) -> None:
    rows_a, rows_b = _paired_rows(np.arange(1, n + 1, dtype=float))
    result = analyze_paired_cell(_spec(), rows_a, rows_b)
    assert result.status == expected_status
    assert (result.raw_mean_difference is None) is effect_suppressed
    assert (result.pointwise_stability_band is None) is effect_suppressed
    if n < 30:
        assert result.directional_language_allowed is False
        assert result.standardized_signal_to_heterogeneity is None
    else:
        assert result.directional_language_allowed is True
        assert result.standardized_signal_to_heterogeneity is not None


@pytest.mark.parametrize(
    "nonzero_n, expected_status, directional",
    [
        (9, "small_or_sparse", False),
        (10, "descriptive", True),
    ],
)
def test_exact_nonzero_effect_boundaries(
    nonzero_n: int,
    expected_status: str,
    directional: bool,
) -> None:
    effects = np.zeros(30)
    effects[:nonzero_n] = np.arange(1, nonzero_n + 1)
    rows_a, rows_b = _paired_rows(effects)
    result = analyze_paired_cell(_spec(), rows_a, rows_b)
    assert result.nonzero_effect_n == nonzero_n
    assert result.status == expected_status
    assert result.directional_language_allowed is directional
    assert (
        result.standardized_signal_to_heterogeneity is not None
    ) is directional


def test_sparse_primary_retains_bands_but_forbids_directional_language() -> None:
    effects = np.zeros(30)
    effects[:9] = 1.0
    rows_a, rows_b = _paired_rows(effects)
    result = analyze_paired_cell(
        _spec(metric_name="oriented_box_overlap_rate"),
        rows_a,
        rows_b,
    )
    assert result.status == "event_sparse"
    assert result.nonzero_effect_n == 9
    assert result.pointwise_stability_band is not None
    assert result.adjusted_primary_stability_band is not None
    assert result.directional_language_allowed is False
    assert result.standardized_signal_to_heterogeneity is None


def test_all_zero_primary_is_retained_as_sparse_not_discarded() -> None:
    rows_a, rows_b = _paired_rows(np.zeros(30))
    result = analyze_paired_cell(
        _spec(metric_name="waymax_kinematic_infeasibility_rate"),
        rows_a,
        rows_b,
    )
    assert result.raw_mean_difference == 0.0
    assert result.raw_median_difference == 0.0
    assert result.favorable_proportion == 0.5
    assert result.nonzero_effect_n == 0
    assert result.standardized_signal_to_heterogeneity is None
    assert result.status == "event_sparse"
    assert result.pointwise_stability_band.lower == 0.0
    assert result.pointwise_stability_band.upper == 0.0
    assert result.adjusted_primary_stability_band.lower == 0.0
    assert result.adjusted_primary_stability_band.upper == 0.0


def test_missingness_and_component_asymmetry_are_retained_and_block_direction() -> None:
    rows_a, rows_b = _paired_rows(np.arange(10, dtype=float))
    rows_a.extend(
        [
            ScenarioScalar.missing(10, "no_supported_lane", total_components=4),
            ScenarioScalar.missing(11, "no_supported_lane", total_components=4),
        ]
    )
    rows_b.extend(
        [
            ScenarioScalar.missing(
                10,
                "no_eligible_vehicle_frame",
                total_components=4,
            ),
            ScenarioScalar(
                11,
                4.0,
                eligible_components=1,
                total_components=4,
            ),
        ]
    )
    # A source-component mismatch on an otherwise paired valid row is also visible.
    rows_b[0] = ScenarioScalar(
        0,
        rows_b[0].value,
        eligible_components=1,
        total_components=3,
    )

    result = analyze_paired_cell(
        _spec(metric_name="lane_center_distance_m"),
        rows_a,
        rows_b,
    )
    assert result.cohort_n == 12
    assert result.valid_a_n == 10
    assert result.valid_b_n == 11
    assert result.paired_n == 10
    assert result.excluded_n == 2
    assert result.both_missing_n == 1
    assert result.asymmetric_missing_n == 1
    assert result.asymmetric_reason_n == 1
    assert result.asymmetric_component_n == 2
    assert result.missing_reasons_a == (("no_supported_lane", 2),)
    assert result.missing_reasons_b == (("no_eligible_vehicle_frame", 1),)
    assert result.eligible_components_a == 20
    assert result.eligible_components_b == 20
    assert result.total_components_a == 38
    assert result.total_components_b == 38
    assert result.source_pairing_complete is False
    assert result.status == "pairing_incomplete"
    assert result.directional_language_allowed is False


def test_execution_failure_cannot_be_converted_to_source_missingness() -> None:
    rows_a, rows_b = _paired_rows(np.ones(10))
    rows_a[0] = ScenarioScalar.missing(
        0,
        "policy_failure",
        total_components=3,
    )
    with pytest.raises(ValueError, match="execution failures"):
        analyze_paired_cell(_spec(), rows_a, rows_b)


def test_aggregate_semantics_come_from_canonical_metric_and_slice_catalogs() -> None:
    spec = _spec()
    rows_a, rows_b = _paired_rows(np.ones(30))
    result = analyze_paired_cell(spec, rows_a, rows_b)
    serialized = result.to_dict()
    canonical = M5_METRIC_SPECS[spec.metric_name]
    assert serialized["value_unit"] == canonical.value_unit
    assert serialized["direction"] == canonical.direction
    assert spec.invalid_reason_codes == canonical.invalid_reason_codes
    assert set(CANONICAL_SLICE_NAMES) == set(STATS_SLICE_NAMES)


@pytest.mark.parametrize("n", [0, 1, 9])
def test_empty_singleton_and_nine_rows_suppress_effects_and_bands(n: int) -> None:
    rows_a, rows_b = _paired_rows(np.ones(n))
    result = analyze_paired_cell(_spec(), rows_a, rows_b)
    assert result.paired_n == n
    assert result.nonzero_effect_n is None
    assert result.raw_mean_difference is None
    assert result.raw_median_difference is None
    assert result.pointwise_stability_band is None
    assert result.adjusted_primary_stability_band is None
    assert result.status == "insufficient_n"


def test_duplicate_or_structurally_missing_rows_fail_closed() -> None:
    row = ScenarioScalar(0, 1.0)
    with pytest.raises(ValueError, match="duplicate cohort_index"):
        analyze_paired_cell(_spec(), [row, row], [row])
    with pytest.raises(ValueError, match="same cohort_index set"):
        analyze_paired_cell(
            _spec(),
            [ScenarioScalar(0, 1.0)],
            [ScenarioScalar(1, 1.0)],
        )
    with pytest.raises(TypeError, match="ScenarioScalar"):
        analyze_paired_cell(_spec(), [object()], [object()])


def test_row_order_canonicalization_is_byte_stable() -> None:
    effects = np.array(
        [-3.0, 1.0, 0.0, 4.0, -2.0] * 6,
        dtype=float,
    )
    rows_a, rows_b = _paired_rows(effects)
    forward = analyze_paired_cell(_spec(), rows_a, rows_b)
    reverse = analyze_paired_cell(
        _spec(),
        list(reversed(rows_a)),
        list(reversed(rows_b)),
    )
    assert forward == reverse
    assert forward.to_json() == reverse.to_json()
    assert "confidence" not in forward.to_json().lower()
    assert "p_value" not in forward.to_json().lower()
    assert "q_value" not in forward.to_json().lower()


def test_contradictory_sign_is_preserved_without_ranking_rewrite() -> None:
    # For a lower-is-favorable metric, A is consistently worse than B.
    effects = np.linspace(0.1, 3.0, 30)
    rows_a, rows_b = _paired_rows(effects)
    result = analyze_paired_cell(_spec(), rows_a, rows_b)
    assert result.raw_mean_difference > 0.0
    assert result.oriented_mean_advantage < 0.0
    assert result.favorable_proportion == 0.0
    assert result.raw_mean_difference == pytest.approx(float(np.mean(effects)))


def test_primary_and_nonprimary_resample_counts_are_frozen() -> None:
    assert _spec(metric_name="position_error_m").resamples == M5_PRIMARY_RESAMPLES
    assert (
        _spec(
            metric_name="position_error_m",
            slice_name="vru_present_current",
        ).resamples
        == M5_OTHER_RESAMPLES
    )
    assert _spec().resamples == M5_OTHER_RESAMPLES
