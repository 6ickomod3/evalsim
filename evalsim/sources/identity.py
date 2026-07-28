"""Stable identity helpers shared by the synthetic source and its manifests."""
from __future__ import annotations

import hashlib
import json

SYNTHETIC_SOURCE_VERSION = "0.1.1"
SYNTHETIC_RNG_ALGORITHM = "PCG64"
SYNTHETIC_SCENARIO_KIND_VALUES = (
    "following",
    "intersection",
    "merge",
    "turn",
    "pedestrian_crossing",
)


def synthetic_source_fingerprint(
    *,
    seed: int,
    num_steps: int,
    dt: float,
    split: str,
    source_version: str = SYNTHETIC_SOURCE_VERSION,
) -> str:
    """Hash every source setting that can affect generated scenario contents."""

    config = {
        "dt": float(dt),
        "num_steps": int(num_steps),
        "rng_algorithm": SYNTHETIC_RNG_ALGORITHM,
        "seed": int(seed),
        "source_version": source_version,
        "split": split,
    }
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def synthetic_scenario_id(
    *,
    kind: str,
    source_fingerprint: str,
    index: int,
) -> str:
    if kind not in SYNTHETIC_SCENARIO_KIND_VALUES:
        raise ValueError(f"unsupported synthetic scenario kind {kind!r}")
    if index < 0:
        raise ValueError("scenario index must be non-negative")
    return f"synthetic-{kind}-{source_fingerprint}-{index:05d}"


def parse_synthetic_scenario_id(scenario_id: str) -> tuple[str, str, int]:
    """Return ``(kind, fingerprint, index)`` for a canonical synthetic ID."""

    if not isinstance(scenario_id, str):
        raise ValueError("synthetic scenario ID must be a string")
    try:
        source_and_kind, fingerprint, index_text = scenario_id.rsplit("-", 2)
    except ValueError as exc:
        raise ValueError(f"invalid synthetic scenario ID {scenario_id!r}") from exc
    prefix = "synthetic-"
    if not source_and_kind.startswith(prefix):
        raise ValueError(f"invalid synthetic scenario ID {scenario_id!r}")
    kind = source_and_kind[len(prefix) :]
    if (
        kind not in SYNTHETIC_SCENARIO_KIND_VALUES
        or len(fingerprint) != 12
        or any(char not in "0123456789abcdef" for char in fingerprint)
        or not index_text.isascii()
        or not index_text.isdecimal()
    ):
        raise ValueError(f"invalid synthetic scenario ID {scenario_id!r}")
    index = int(index_text)
    if (
        synthetic_scenario_id(
            kind=kind,
            source_fingerprint=fingerprint,
            index=index,
        )
        != scenario_id
    ):
        raise ValueError(f"non-canonical synthetic scenario ID {scenario_id!r}")
    return kind, fingerprint, index
