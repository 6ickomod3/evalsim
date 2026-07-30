"""Outcome-suppressed NumPy compute-pilot evidence for M6.

The runner validates the complete detached 128-member source cohort and exact source
ledger, executes only the caller-ordered 1..8 eligible selection through the same
built-in per-case execution block as the official NumPy evaluator, and returns timing
metadata only. Rollouts, traces, pairs, metric results, response flags, signs, and
per-scene identities remain function-local and are discarded.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import InitVar, dataclass, field
import hashlib
import json
import threading
import time
from typing import Any

from evalsim.evaluation.m6 import (
    M6EligibilityLedger,
    M6EvaluationCase,
    _run_m6_numpy_pilot_execution,
)


M6_NUMPY_PILOT_SCHEMA_VERSION = "m6-numpy-pilot-1.1.0"

_ISSUER = object()
_HEX = frozenset("0123456789abcdef")


def m6_numpy_pilot_selected_cohort_indices_sha256(
    selected_cohort_indices: Sequence[int],
) -> str:
    """Commit to the exact ordered pilot selection without exposing its IDs."""

    if isinstance(selected_cohort_indices, (str, bytes)) or not isinstance(
        selected_cohort_indices,
        Sequence,
    ):
        raise TypeError("selected cohort indices must be an ordered sequence")
    selected = tuple(selected_cohort_indices)
    if (
        not 1 <= len(selected) <= 8
        or len(set(selected)) != len(selected)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 128
            for value in selected
        )
    ):
        raise ValueError("selected cohort indices are not a bounded unique domain")
    payload = json.dumps(
        {"ordered_cohort_indices": list(selected)},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(
        b"evalsim-m6-numpy-pilot-selected-cohort-indices-v1\0"
    )
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _M6NumpyPilotObservationIssuance:
    observation: Any
    scene_count: int
    scene_durations_ms: object
    total_execution_ms: int
    max_scene_ms: int
    selected_cohort_indices_sha256: str
    source_selection_binding_sha256: str
    execution_binding_sha256: str
    schema_version: str
    issued_binding_sha256: str


_ISSUANCE_LOCK = threading.Lock()
_ISSUANCE_REGISTRY: dict[int, _M6NumpyPilotObservationIssuance] = {}


@dataclass(frozen=True, slots=True)
class M6NumpyPilotObservation:
    """Factory-issued aggregate-only evidence for one bounded NumPy pilot."""

    scene_count: int
    scene_durations_ms: tuple[int, ...]
    total_execution_ms: int
    max_scene_ms: int
    selected_cohort_indices_sha256: str = field(repr=False)
    source_selection_binding_sha256: str = field(repr=False)
    execution_binding_sha256: str = field(repr=False)
    schema_version: str = M6_NUMPY_PILOT_SCHEMA_VERSION
    _issued_binding_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _ISSUER:
            raise TypeError("NumPy pilot observations are runner-issued only")
        durations = self._validated_public_fields()
        object.__setattr__(self, "scene_durations_ms", durations)
        object.__setattr__(
            self,
            "_issued_binding_sha256",
            self._public_fields_sha256(),
        )
        record = _M6NumpyPilotObservationIssuance(
            observation=self,
            scene_count=self.scene_count,
            scene_durations_ms=self.scene_durations_ms,
            total_execution_ms=self.total_execution_ms,
            max_scene_ms=self.max_scene_ms,
            selected_cohort_indices_sha256=(
                self.selected_cohort_indices_sha256
            ),
            source_selection_binding_sha256=(
                self.source_selection_binding_sha256
            ),
            execution_binding_sha256=self.execution_binding_sha256,
            schema_version=self.schema_version,
            issued_binding_sha256=self._issued_binding_sha256,
        )
        with _ISSUANCE_LOCK:
            if id(self) in _ISSUANCE_REGISTRY:
                raise RuntimeError("NumPy pilot issuance identity was reused")
            _ISSUANCE_REGISTRY[id(self)] = record

    def _validated_public_fields(self) -> tuple[int, ...]:
        durations = tuple(self.scene_durations_ms)
        if (
            isinstance(self.scene_count, bool)
            or not isinstance(self.scene_count, int)
            or not 1 <= self.scene_count <= 8
            or type(self.scene_durations_ms) is not tuple
            or len(durations) != self.scene_count
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in durations
            )
            or isinstance(self.total_execution_ms, bool)
            or not isinstance(self.total_execution_ms, int)
            or self.total_execution_ms != sum(durations)
            or isinstance(self.max_scene_ms, bool)
            or not isinstance(self.max_scene_ms, int)
            or self.max_scene_ms != max(durations)
            or not isinstance(self.selected_cohort_indices_sha256, str)
            or len(self.selected_cohort_indices_sha256) != 64
            or any(
                character not in _HEX
                for character in self.selected_cohort_indices_sha256
            )
            or not isinstance(self.source_selection_binding_sha256, str)
            or len(self.source_selection_binding_sha256) != 64
            or any(
                character not in _HEX
                for character in self.source_selection_binding_sha256
            )
            or not isinstance(self.execution_binding_sha256, str)
            or len(self.execution_binding_sha256) != 64
            or any(
                character not in _HEX
                for character in self.execution_binding_sha256
            )
            or self.execution_binding_sha256
            == self.source_selection_binding_sha256
            or self.schema_version != M6_NUMPY_PILOT_SCHEMA_VERSION
        ):
            raise ValueError("NumPy pilot observation is invalid")
        return durations

    def _public_fields_sha256(self) -> str:
        payload = json.dumps(
            {
                "execution_binding_sha256": self.execution_binding_sha256,
                "max_scene_ms": self.max_scene_ms,
                "scene_count": self.scene_count,
                "scene_durations_ms": list(self.scene_durations_ms),
                "schema_version": self.schema_version,
                "selected_cohort_indices_sha256": (
                    self.selected_cohort_indices_sha256
                ),
                "source_selection_binding_sha256": (
                    self.source_selection_binding_sha256
                ),
                "total_execution_ms": self.total_execution_ms,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest = hashlib.sha256(
            b"evalsim-m6-numpy-pilot-observation-v1\0"
        )
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        return digest.hexdigest()

    def revalidate(self) -> None:
        self._validated_public_fields()
        with _ISSUANCE_LOCK:
            record = _ISSUANCE_REGISTRY.get(id(self))
        if (
            record is None
            or record.observation is not self
            or self.scene_count != record.scene_count
            or self.scene_durations_ms is not record.scene_durations_ms
            or self.total_execution_ms != record.total_execution_ms
            or self.max_scene_ms != record.max_scene_ms
            or self.selected_cohort_indices_sha256
            != record.selected_cohort_indices_sha256
            or self.source_selection_binding_sha256
            != record.source_selection_binding_sha256
            or self.execution_binding_sha256
            != record.execution_binding_sha256
            or self.schema_version != record.schema_version
            or self._issued_binding_sha256
            != record.issued_binding_sha256
            or not isinstance(self._issued_binding_sha256, str)
            or len(self._issued_binding_sha256) != 64
            or any(
                character not in _HEX
                for character in self._issued_binding_sha256
            )
            or self._issued_binding_sha256
            != self._public_fields_sha256()
        ):
            raise ValueError(
                "NumPy pilot observation integrity binding is invalid"
            )

    @property
    def observation_binding_sha256(self) -> str:
        """Return the registry-authenticated digest of every public fact."""

        self.revalidate()
        return self._issued_binding_sha256

    def to_summary_fields(self) -> dict[str, int | str]:
        """Return the only NumPy fields admitted to the pilot store row."""

        self.revalidate()
        return {
            "max_scene_ms": self.max_scene_ms,
            "numpy_ms": self.total_execution_ms,
            "pilot_scene_n": self.scene_count,
            "selected_cohort_indices_sha256": (
                self.selected_cohort_indices_sha256
            ),
        }


def run_m6_numpy_pilot(
    cases: Iterable[M6EvaluationCase],
    ledger: M6EligibilityLedger,
    selected_cohort_indices: Sequence[int],
    *,
    selection_binding_sha256: str,
    clock_ns: Callable[[], int] | None = None,
) -> M6NumpyPilotObservation:
    """Run one exact bounded pilot and issue no outcome-bearing evidence."""

    clock = time.monotonic_ns if clock_ns is None else clock_ns
    selected = tuple(selected_cohort_indices)
    durations, binding = _run_m6_numpy_pilot_execution(
        cases,
        ledger,
        selected,
        selection_binding_sha256,
        clock_ns=clock,
    )
    selected_sha256 = m6_numpy_pilot_selected_cohort_indices_sha256(selected)
    return M6NumpyPilotObservation(
        scene_count=len(durations),
        scene_durations_ms=durations,
        total_execution_ms=sum(durations),
        max_scene_ms=max(durations),
        selected_cohort_indices_sha256=selected_sha256,
        source_selection_binding_sha256=selection_binding_sha256,
        execution_binding_sha256=binding,
        _issuance_capability=_ISSUER,
    )


__all__ = [
    "M6_NUMPY_PILOT_SCHEMA_VERSION",
    "M6NumpyPilotObservation",
    "m6_numpy_pilot_selected_cohort_indices_sha256",
    "run_m6_numpy_pilot",
]
