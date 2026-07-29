"""Deterministic registration and evaluation of per-scenario metrics."""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from evalsim.contracts.metric import (
    Metric,
    MetricEligibility,
    MetricResult,
    MetricSpec,
)
from evalsim.contracts.rollout import Rollout
from evalsim.contracts.scenario import Scenario


class MetricRegistryError(ValueError):
    """A metric registration or evaluated-result contract violation."""


class MetricRegistry:
    """One unambiguous active version per metric name, evaluated by sorted ID."""

    def __init__(self, metrics: Iterable[Metric] = ()) -> None:
        self._metrics: dict[str, Metric] = {}
        self._specs: dict[str, MetricSpec] = {}
        for metric in metrics:
            self.register(metric)

    def __len__(self) -> int:
        return len(self._metrics)

    def __iter__(self) -> Iterator[Metric]:
        for name in sorted(self._metrics):
            yield self._metrics[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._metrics))

    @property
    def specs(self) -> tuple[MetricSpec, ...]:
        return tuple(self._specs[name] for name in self.names)

    def register(self, metric: Metric) -> None:
        if not isinstance(metric, Metric):
            raise TypeError("metric must implement Metric")
        spec = getattr(metric, "spec", None)
        if not isinstance(spec, MetricSpec):
            raise MetricRegistryError(
                "metric.spec must be a validated MetricSpec"
            )
        if spec.name in self._metrics:
            existing = self._specs[spec.name]
            if existing.version == spec.version:
                detail = f"duplicate metric {spec.metric_id}"
            else:
                detail = (
                    f"metric name {spec.name!r} is ambiguous between "
                    f"versions {existing.version!r} and {spec.version!r}"
                )
            raise MetricRegistryError(detail)
        self._metrics[spec.name] = metric
        self._specs[spec.name] = spec

    def get(self, name: str, version: str | None = None) -> Metric:
        try:
            metric = self._metrics[name]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown metric {name!r}") from exc
        if version is not None and self._specs[name].version != version:
            raise KeyError(
                f"metric {name!r} does not have active version {version!r}"
            )
        return metric

    def evaluate(
        self,
        scenario: Scenario,
        rollout: Rollout,
        *,
        metric_names: Sequence[str] | None = None,
    ) -> tuple[MetricResult, ...]:
        """Evaluate selected metrics in canonical name order."""

        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        if not isinstance(rollout, Rollout):
            raise TypeError("rollout must be a Rollout")
        self._validate_execution_pair(scenario, rollout)

        names = self._selected_names(metric_names)
        results: list[MetricResult] = []
        for name in names:
            metric = self._metrics[name]
            registered_spec = self._specs[name]
            if metric.spec != registered_spec:
                raise MetricRegistryError(
                    f"registered metric {name!r} changed its spec"
                )

            eligibility = metric.eligibility(scenario)
            if not isinstance(eligibility, MetricEligibility):
                raise MetricRegistryError(
                    f"metric {name!r} eligibility() must return "
                    "MetricEligibility"
                )
            self._validate_eligibility(registered_spec, eligibility)

            result = metric.compute(scenario, rollout)
            if not isinstance(result, MetricResult):
                raise MetricRegistryError(
                    f"metric {name!r} compute() must return MetricResult"
                )
            self._validate_result(
                spec=registered_spec,
                scenario=scenario,
                eligibility=eligibility,
                result=result,
            )
            results.append(result)
        return tuple(results)

    @staticmethod
    def _validate_execution_pair(
        scenario: Scenario,
        rollout: Rollout,
    ) -> None:
        """Reject execution drift before source-only eligibility is evaluated."""

        if rollout.scenario_id != scenario.scenario_id:
            raise MetricRegistryError(
                "rollout scenario_id does not match the Scenario"
            )
        if rollout.num_steps != scenario.num_steps:
            raise MetricRegistryError(
                "rollout horizon does not match the Scenario"
            )
        if rollout.num_agents != scenario.num_agents:
            raise MetricRegistryError(
                "rollout agent count does not match the Scenario"
            )
        if not np.array_equal(rollout.timestamps, scenario.timestamps):
            raise MetricRegistryError(
                "rollout timestamps do not match the Scenario"
            )
        for index, (source, candidate) in enumerate(
            zip(scenario.agents, rollout.agents, strict=True)
        ):
            if source.id != candidate.id or source.type != candidate.type:
                raise MetricRegistryError(
                    "rollout agent identity/order does not match the Scenario "
                    f"at contract index {index}"
                )
            if (
                source.length != candidate.length
                or source.width != candidate.width
            ):
                raise MetricRegistryError(
                    "rollout dimensions do not match the Scenario at contract "
                    f"index {index}"
                )
            if not np.array_equal(source.valid, candidate.valid):
                raise MetricRegistryError(
                    "rollout validity does not match the Scenario at contract "
                    f"index {index}"
                )

    def _selected_names(
        self,
        metric_names: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if metric_names is None:
            return self.names
        if isinstance(metric_names, (str, bytes)):
            raise ValueError("metric_names must be a sequence, not a string")
        names = tuple(metric_names)
        if any(not isinstance(name, str) for name in names):
            raise ValueError("metric_names must contain strings")
        if len(set(names)) != len(names):
            raise ValueError("metric_names must not contain duplicates")
        unknown = sorted(set(names).difference(self._metrics))
        if unknown:
            raise KeyError(f"unknown metrics: {', '.join(unknown)}")
        return tuple(sorted(names))

    @staticmethod
    def _validate_eligibility(
        spec: MetricSpec,
        eligibility: MetricEligibility,
    ) -> None:
        if (
            not eligibility.eligible
            and eligibility.reason not in spec.invalid_reason_codes
        ):
            raise MetricRegistryError(
                f"metric {spec.name!r} returned unregistered eligibility "
                f"reason {eligibility.reason!r}"
            )

    @staticmethod
    def _validate_result(
        *,
        spec: MetricSpec,
        scenario: Scenario,
        eligibility: MetricEligibility,
        result: MetricResult,
    ) -> None:
        if (
            result.metric_name != spec.name
            or result.metric_version != spec.version
        ):
            raise MetricRegistryError(
                f"metric {spec.name!r} returned mismatched identity/version"
            )
        if result.scenario_id != scenario.scenario_id:
            raise MetricRegistryError(
                f"metric {spec.name!r} returned a different scenario_id"
            )
        if result.valid != eligibility.eligible:
            raise MetricRegistryError(
                f"metric {spec.name!r} result validity contradicts eligibility"
            )
        if not result.valid:
            if result.invalid_reason not in spec.invalid_reason_codes:
                raise MetricRegistryError(
                    f"metric {spec.name!r} returned unregistered invalid "
                    f"reason {result.invalid_reason!r}"
                )
            if result.invalid_reason != eligibility.reason:
                raise MetricRegistryError(
                    f"metric {spec.name!r} result reason contradicts "
                    "eligibility"
                )


__all__ = ["MetricRegistry", "MetricRegistryError"]
