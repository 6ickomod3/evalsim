"""M5 metric-contract and deterministic-registry acceptance tests."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from evalsim import (
    Metric,
    MetricEligibility,
    MetricResult,
    MetricSpec,
    Rollout,
    Scenario,
)
from evalsim.metrics import MetricRegistry, MetricRegistryError


def _spec(
    name: str = "position_error_m",
    version: str = "1.0.0",
    *,
    invalid_reason_codes: tuple[str, ...] = ("no_components",),
) -> MetricSpec:
    return MetricSpec(
        name=name,
        version=version,
        value_unit="metres",
        unit_of_analysis="scenario",
        direction="lower",
        aggregation="mean",
        agent_scope="world",
        evaluation_window="simulated_future",
        eligibility="at least one valid world-agent frame",
        invalid_reason_codes=invalid_reason_codes,
        required_fields=("scenario.agents", "rollout.agents"),
        depends_on=(),
        deterministic=True,
        known_failure_modes=("validity drift",),
    )


class _FixedMetric(Metric):
    def __init__(
        self,
        spec: MetricSpec,
        *,
        eligibility: MetricEligibility | None = None,
        result: MetricResult | object | None = None,
    ) -> None:
        self.spec = spec
        self._eligibility = eligibility or MetricEligibility.accepted()
        self._result = result

    def eligibility(
        self,
        scenario: Scenario,
    ) -> MetricEligibility:
        return self._eligibility

    def compute(self, scenario: Scenario, rollout: Rollout) -> MetricResult:
        if self._result is not None:
            return self._result  # type: ignore[return-value]
        return MetricResult(
            metric_name=self.spec.name,
            metric_version=self.spec.version,
            scenario_id=scenario.scenario_id,
            value=1.0,
            distribution=(0.5, 1.5),
            eligible_components=2,
            total_components=3,
            details={"source": "unit"},
        )


def _valid_result(
    *,
    name: str = "position_error_m",
    version: str = "1.0.0",
    scenario_id: str = "scn_0001",
) -> MetricResult:
    return MetricResult(
        metric_name=name,
        metric_version=version,
        scenario_id=scenario_id,
        value=1.0,
        distribution=(0.5, 1.5),
        eligible_components=2,
        total_components=3,
    )


def _invalid_result(
    *,
    reason: str = "no_components",
    scenario_id: str = "scn_0001",
) -> MetricResult:
    return MetricResult(
        metric_name="position_error_m",
        metric_version="1.0.0",
        scenario_id=scenario_id,
        value=None,
        valid=False,
        invalid_reason=reason,
        eligible_components=0,
        total_components=3,
    )


def test_metric_spec_is_frozen_validated_and_normalizes_sequences() -> None:
    required = ["scenario.agents", "rollout.agents"]
    spec = MetricSpec(
        name="position_error_m",
        version="1.2.3-rc.1+cpu",
        value_unit=" metres ",
        invalid_reason_codes=["no_components"],
        required_fields=required,
        depends_on=["speed_error_mps"],
        known_failure_modes=[" mask drift "],
    )
    required.append("mutated")

    assert spec.metric_id == "position_error_m@1.2.3-rc.1+cpu"
    assert spec.value_unit == "metres"
    assert spec.invalid_reason_codes == ("no_components",)
    assert spec.required_fields == (
        "scenario.agents",
        "rollout.agents",
    )
    assert spec.depends_on == ("speed_error_mps",)
    assert spec.known_failure_modes == ("mask drift",)
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Position Error"),
        ("version", "1.0"),
        ("value_unit", ""),
        ("unit_of_analysis", "object"),
        ("direction", "ascending"),
        ("aggregation", "composite"),
        ("agent_scope", "vehicle"),
        ("evaluation_window", ""),
        ("eligibility", ""),
        ("deterministic", 1),
    ],
)
def test_metric_spec_rejects_invalid_scalar_fields(
    field: str,
    value: object,
) -> None:
    kwargs = {"name": "position_error_m", "version": "1.0.0", field: value}
    with pytest.raises(ValueError):
        MetricSpec(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("invalid_reason_codes", "not-a-sequence"),
        ("invalid_reason_codes", ("bad-code",)),
        ("invalid_reason_codes", ("same", "same")),
        ("required_fields", ("field", "field")),
        ("depends_on", ("position_error_m",)),
        ("known_failure_modes", ("",)),
    ],
)
def test_metric_spec_rejects_invalid_sequence_fields(
    field: str,
    value: object,
) -> None:
    kwargs = {"name": "position_error_m", "version": "1.0.0", field: value}
    with pytest.raises(ValueError):
        MetricSpec(**kwargs)


def test_metric_eligibility_implications_and_factories() -> None:
    assert MetricEligibility.accepted() == MetricEligibility(True)
    assert MetricEligibility.rejected("no_components") == MetricEligibility(
        False,
        "no_components",
    )
    with pytest.raises(ValueError):
        MetricEligibility(True, "no_components")
    with pytest.raises(ValueError):
        MetricEligibility(False)
    with pytest.raises(ValueError):
        MetricEligibility(False, "bad-code")
    with pytest.raises(ValueError):
        MetricEligibility(1)  # type: ignore[arg-type]


def test_metric_result_freezes_detached_json_details() -> None:
    source = {
        "nested": {"threshold": 3.0},
        "labels": ["a", {"enabled": True}],
    }
    result = MetricResult(
        "position_error_m",
        "1.0.0",
        "scenario",
        1.0,
        distribution=[0.0, 2.0],
        eligible_components=2,
        total_components=4,
        details=source,
    )
    source["nested"]["threshold"] = 99.0
    source["labels"].append("mutated")

    assert result.value == 1.0
    assert result.distribution == (0.0, 2.0)
    assert result.details["nested"]["threshold"] == 3.0
    assert result.details["labels"] == ("a", {"enabled": True})
    with pytest.raises(TypeError):
        result.details["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        result.details["nested"]["threshold"] = 1  # type: ignore[index]


def test_metric_result_rejects_non_json_or_cyclic_details() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    for details in (
        {"bad": math.inf},
        {"bad": object()},
        {1: "non-string-key"},
        cyclic,
    ):
        with pytest.raises(ValueError):
            MetricResult(
                "position_error_m",
                "1.0.0",
                "scenario",
                1.0,
                distribution=(1.0,),
                details=details,
            )


@pytest.mark.parametrize(
    "changes",
    [
        {"value": None},
        {"value": math.nan},
        {"value": math.inf},
        {"value": True},
        {"invalid_reason": "no_components"},
        {"eligible_components": 0, "distribution": ()},
        {"eligible_components": 2, "total_components": 1},
        {"eligible_components": 2, "distribution": (1.0,)},
        {"distribution": (math.nan,)},
    ],
)
def test_valid_result_implications_are_enforced(changes: dict) -> None:
    kwargs = {
        "metric_name": "position_error_m",
        "metric_version": "1.0.0",
        "scenario_id": "scenario",
        "value": 1.0,
        "distribution": (1.0,),
        "valid": True,
        "invalid_reason": None,
        "eligible_components": 1,
        "total_components": 1,
    }
    kwargs.update(changes)
    with pytest.raises(ValueError):
        MetricResult(**kwargs)


@pytest.mark.parametrize(
    "changes",
    [
        {"value": 0.0},
        {"invalid_reason": None},
        {"invalid_reason": "bad-code"},
        {"eligible_components": 1, "distribution": (0.0,)},
    ],
)
def test_invalid_result_implications_are_enforced(changes: dict) -> None:
    kwargs = {
        "metric_name": "position_error_m",
        "metric_version": "1.0.0",
        "scenario_id": "scenario",
        "value": None,
        "distribution": (),
        "valid": False,
        "invalid_reason": "no_components",
        "eligible_components": 0,
        "total_components": 1,
    }
    kwargs.update(changes)
    with pytest.raises(ValueError):
        MetricResult(**kwargs)


def test_metric_base_has_only_per_scenario_abstract_operations() -> None:
    assert Metric.__abstractmethods__ == frozenset({"eligibility", "compute"})
    assert not hasattr(Metric, "aggregate")
    assert not hasattr(Metric, "validate_inputs")


def test_registry_rejects_duplicates_and_name_version_ambiguity() -> None:
    registry = MetricRegistry()
    registry.register(_FixedMetric(_spec()))
    with pytest.raises(MetricRegistryError, match="duplicate"):
        registry.register(_FixedMetric(_spec()))
    with pytest.raises(MetricRegistryError, match="ambiguous"):
        registry.register(_FixedMetric(_spec(version="2.0.0")))
    with pytest.raises(TypeError):
        registry.register(object())  # type: ignore[arg-type]


def test_registry_order_is_independent_of_registration_order(
    scenario: Scenario,
    rollout: Rollout,
) -> None:
    alpha = _FixedMetric(_spec("alpha_metric"))
    zeta = _FixedMetric(_spec("zeta_metric"))

    first = MetricRegistry((zeta, alpha))
    second = MetricRegistry((alpha, zeta))

    assert first.names == second.names == ("alpha_metric", "zeta_metric")
    assert tuple(spec.name for spec in first.specs) == first.names
    assert [result.metric_name for result in first.evaluate(scenario, rollout)] == [
        "alpha_metric",
        "zeta_metric",
    ]
    assert first.evaluate(scenario, rollout) == second.evaluate(
        scenario,
        rollout,
    )


def test_registry_subset_lookup_is_sorted_and_exact(
    scenario: Scenario,
    rollout: Rollout,
) -> None:
    registry = MetricRegistry(
        (
            _FixedMetric(_spec("zeta_metric")),
            _FixedMetric(_spec("alpha_metric")),
        )
    )
    assert registry.get("alpha_metric", "1.0.0").spec.name == "alpha_metric"
    assert [
        result.metric_name
        for result in registry.evaluate(
            scenario,
            rollout,
            metric_names=("zeta_metric", "alpha_metric"),
        )
    ] == ["alpha_metric", "zeta_metric"]
    with pytest.raises(KeyError):
        registry.get("missing")
    with pytest.raises(KeyError):
        registry.get("alpha_metric", "2.0.0")
    with pytest.raises(ValueError):
        registry.evaluate(scenario, rollout, metric_names=("alpha_metric",) * 2)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (_valid_result(name="other_metric"), "identity/version"),
        (_valid_result(version="2.0.0"), "identity/version"),
        (_valid_result(scenario_id="other_scene"), "scenario_id"),
    ],
)
def test_registry_rejects_result_identity_drift(
    scenario: Scenario,
    rollout: Rollout,
    result: MetricResult,
    message: str,
) -> None:
    registry = MetricRegistry((_FixedMetric(_spec(), result=result),))
    with pytest.raises(MetricRegistryError, match=message):
        registry.evaluate(scenario, rollout)


def test_registry_enforces_registered_reasons_and_eligibility_match(
    scenario: Scenario,
    rollout: Rollout,
) -> None:
    unregistered = _FixedMetric(
        _spec(invalid_reason_codes=()),
        eligibility=MetricEligibility.rejected("no_components"),
        result=_invalid_result(),
    )
    with pytest.raises(MetricRegistryError, match="unregistered eligibility"):
        MetricRegistry((unregistered,)).evaluate(scenario, rollout)

    mismatch = _FixedMetric(
        _spec(),
        eligibility=MetricEligibility.rejected("no_components"),
        result=_invalid_result(reason="different_reason"),
    )
    with pytest.raises(MetricRegistryError, match="unregistered invalid"):
        MetricRegistry((mismatch,)).evaluate(scenario, rollout)

    contradictory = _FixedMetric(
        _spec(),
        eligibility=MetricEligibility.rejected("no_components"),
        result=_valid_result(),
    )
    with pytest.raises(MetricRegistryError, match="contradicts eligibility"):
        MetricRegistry((contradictory,)).evaluate(scenario, rollout)


def test_registry_accepts_matching_source_only_invalid_result(
    scenario: Scenario,
    rollout: Rollout,
) -> None:
    metric = _FixedMetric(
        _spec(),
        eligibility=MetricEligibility.rejected("no_components"),
        result=_invalid_result(),
    )
    assert MetricRegistry((metric,)).evaluate(scenario, rollout) == (
        _invalid_result(),
    )


def test_registry_rejects_wrong_method_outputs_and_scenario_pair(
    scenario: Scenario,
    rollout: Rollout,
) -> None:
    wrong_eligibility = _FixedMetric(_spec())
    wrong_eligibility._eligibility = True  # type: ignore[assignment]
    with pytest.raises(MetricRegistryError, match="MetricEligibility"):
        MetricRegistry((wrong_eligibility,)).evaluate(scenario, rollout)

    wrong_result = _FixedMetric(_spec(), result={"value": 1.0})
    with pytest.raises(MetricRegistryError, match="MetricResult"):
        MetricRegistry((wrong_result,)).evaluate(scenario, rollout)

    mismatched_rollout = Rollout(
        scenario_id="another-scene",
        sim_name=rollout.sim_name,
        sim_version=rollout.sim_version,
        seed=rollout.seed,
        timestamps=rollout.timestamps,
        agents=rollout.agents,
        metadata=rollout.metadata,
    )
    with pytest.raises(MetricRegistryError, match="does not match"):
        MetricRegistry().evaluate(scenario, mismatched_rollout)
