"""Data-free tests for the deterministic aggregate M5 scorecard renderer."""
from __future__ import annotations

import hashlib

import pytest

from evalsim.report.m5 import (
    DATA_FREE_TEST_PROFILE,
    M5_SCORECARD_INPUT_FIELDS,
    M5_SCORECARD_RENDERED_ROW_FIELDS,
    M5_SCORECARD_RENDERER_VERSION,
    OFFICIAL_M5_PROFILE,
    render_m5_scorecard,
)
from evalsim.results import scorecard_row_from_result
from evalsim.stats.m5 import (
    PairedCellSpec,
    PolicyContrast,
    ScenarioScalar,
    analyze_paired_cell,
)


def _suppressed_row(
    policy_a: str = "constant_velocity",
    policy_b: str = "log_replay",
) -> dict[str, object]:
    spec = PairedCellSpec(
        metric_name="position_error_m",
        contrast=PolicyContrast(policy_a, policy_b),
        slice_name="all",
    )
    result = analyze_paired_cell(
        spec,
        [ScenarioScalar(index, float(index)) for index in range(2)],
        [ScenarioScalar(index, float(index)) for index in range(2)],
    )
    return scorecard_row_from_result(result)


def _neutral_effect_row() -> dict[str, object]:
    spec = PairedCellSpec(
        metric_name="minimum_center_distance_m",
        contrast=PolicyContrast("idm", "constant_velocity"),
        slice_name="all",
    )
    result = analyze_paired_cell(
        spec,
        [ScenarioScalar(index, 2.0) for index in range(10)],
        [ScenarioScalar(index, 1.0) for index in range(10)],
    )
    return scorecard_row_from_result(result)


def test_data_free_golden_bytes_are_deterministic_and_lexically_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _suppressed_row("idm", "log_replay")
    second = _suppressed_row("constant_velocity", "log_replay")
    rows = [first, second, _neutral_effect_row()]

    expected = render_m5_scorecard(
        rows,
        row_accounting_profile=DATA_FREE_TEST_PROFILE,
    )
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    reordered = render_m5_scorecard(
        reversed(rows),
        row_accounting_profile=DATA_FREE_TEST_PROFILE,
    )

    assert reordered == expected
    assert expected.endswith(b"\n")
    assert b"\r" not in expected
    position_rows = expected[expected.index(b"## Metric: position_error_m") :]
    assert position_rows.index(
        b"| constant_velocity | log_replay |"
    ) < position_rows.index(b"| idm | log_replay |")
    assert b"[1.0,1.0]@0.95" in expected
    assert hashlib.sha256(expected).hexdigest() == (
        "0d4afed5fa7ad996bdf2963c34e591222fbfe19fc45c0e7b5e4f6d1cd71d6dc9"
    )


def test_renderer_has_a_closed_input_schema_and_aggregate_output_boundary() -> None:
    row = _suppressed_row()
    assert set(row) == M5_SCORECARD_INPUT_FIELDS
    assert "scenario_id" not in M5_SCORECARD_RENDERED_ROW_FIELDS
    assert "resampling_key_json" not in M5_SCORECARD_RENDERED_ROW_FIELDS
    baseline = render_m5_scorecard(
        [row],
        row_accounting_profile=DATA_FREE_TEST_PROFILE,
    )
    assert str(row["resampling_sha256"]).encode("ascii") not in baseline
    assert b"resampling_key_json" not in baseline
    assert b"resampling_digest_words" not in baseline

    injected = dict(row)
    injected["scenario_id"] = "must-not-cross-boundary"
    with pytest.raises(ValueError, match="fixed M5 schema"):
        render_m5_scorecard(
            [injected],
            row_accounting_profile=DATA_FREE_TEST_PROFILE,
        )


def test_renderer_states_claim_limits_without_composite_ranking() -> None:
    rendered = render_m5_scorecard(
        [_suppressed_row()],
        row_accounting_profile=DATA_FREE_TEST_PROFILE,
    ).decode("utf-8")

    assert M5_SCORECARD_RENDERER_VERSION in rendered
    assert "privileged logged-future reference" in rendered
    assert "not population confidence intervals or hypothesis tests" in rendered
    assert "no winner, ranking, or cross-metric aggregate claim" in rendered
    assert "| Winner |" not in rendered
    assert "| Rank |" not in rendered


def test_empty_data_free_render_is_fixed_but_official_requires_full_domain() -> None:
    empty = render_m5_scorecard(
        [],
        row_accounting_profile=DATA_FREE_TEST_PROFILE,
    )
    assert b"No scorecard rows.\n" in empty

    with pytest.raises(ValueError, match="312-cell"):
        render_m5_scorecard(
            [],
            row_accounting_profile=OFFICIAL_M5_PROFILE,
        )


@pytest.mark.parametrize(
    "rows",
    (
        "not rows",
        b"not rows",
        {"not": "rows"},
        [42],
    ),
)
def test_renderer_rejects_non_row_iterables(rows: object) -> None:
    with pytest.raises(TypeError):
        render_m5_scorecard(
            rows,  # type: ignore[arg-type]
            row_accounting_profile=DATA_FREE_TEST_PROFILE,
        )
