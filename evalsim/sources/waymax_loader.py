"""Local-only WOMD loading through the pinned optional Waymax stack.

This module owns filesystem and TensorFlow concerns.  The conversion in
``evalsim.sources.waymax`` remains a pure Waymax-state-to-EvalSim boundary.
Optional dependencies are imported only inside runtime functions so the core package
continues to import without the ``waymo`` extra.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import re
import stat as stat_module
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np

from evalsim.contracts import AgentType, MapType, Scenario

DEFAULT_WOMD_VALIDATION_DIR = Path(
    "data/raw/womd/v1.3.1/tf_example/validation"
)
WOMD_TOTAL_VALIDATION_SHARDS = 150
WOMD_M3_SHARD_INDEX = 0
WOMD_M3_SEARCH_LIMIT = 32
WOMD_M4_SHARD_INDICES = tuple(range(10))
LOCAL_WAYMO_ENV_FLAG = "EVALSIM_RUN_WAYMO_LOCAL"

_SCENARIO_ID_PATTERN = re.compile(r"[0-9a-fA-F]+")
_LOWER_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_AUDIT_KEYS = (
    "state/id",
    "state/type",
    "state/is_sdc",
    "state/which_time",
    "state/all/valid",
    "state/all/x",
    "state/all/y",
    "state/all/velocity_x",
    "state/all/velocity_y",
    "state/all/bbox_yaw",
    "state/all/timestamp_micros",
    "state/all/length",
    "state/all/width",
    "roadgraph_samples/xyz",
    "roadgraph_samples/dir",
    "roadgraph_samples/type",
    "roadgraph_samples/id",
    "roadgraph_samples/valid",
)
_EXPECTED_DATASET_CONFIG_PAYLOAD = {
    "aggregate_timesteps": True,
    "batch_by_scenario": True,
    "batch_dims": [],
    "data_format": "TFRECORD",
    "deterministic": True,
    "distributed": False,
    "drop_remainder": False,
    "include_sdc_paths": True,
    "max_num_objects": 128,
    "max_num_rg_points": 30000,
    "num_paths": 45,
    "num_points_per_path": 800,
    "num_shards": 1,
    "repeat": 1,
    "shuffle_buffer_size": 1000,
    "shuffle_seed": None,
    "tf_data_service_address": None,
    "womd_split": "validation",
    "womd_version": "1.3.1",
}
_DATASET_CONFIG_FIELD_NAMES = frozenset(
    set(_EXPECTED_DATASET_CONFIG_PAYLOAD)
    .difference({"womd_split", "womd_version"})
    .union({"path"})
)
_EventT = TypeVar("_EventT")


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class _ShardDigestCacheEntry:
    identity: _FileIdentity
    sha256: str


_SHARD_DIGEST_CACHE: dict[Path, _ShardDigestCacheEntry] = {}
_SHARD_DIGEST_CACHE_LOCK = threading.Lock()


class WaymaxDependencyError(ImportError):
    """Raised when the optional Waymo runtime is not installed."""


class WaymaxDataError(ValueError):
    """A local-data or source-boundary error with a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class WaymaxRecord:
    """One source-boundary record; raw audit arrays never enter EvalSim metadata."""

    scenario_id: str
    state: Any
    audit: Mapping[str, np.ndarray]
    shard_suffix: str
    record_ordinal: int
    shard_sha256: str
    dataset_config_fingerprint: str


@dataclass(frozen=True, slots=True)
class M4ShardLocator:
    """An exact, privacy-sensitive local locator within M4's ten-shard scope."""

    shard_suffix: str
    record_ordinal: int

    def __post_init__(self) -> None:
        suffix = self.shard_suffix
        if (
            not isinstance(suffix, str)
            or len(suffix) != 5
            or not suffix.isascii()
            or not suffix.isdecimal()
            or int(suffix) not in WOMD_M4_SHARD_INDICES
            or suffix != f"{int(suffix):05d}"
        ):
            raise ValueError(
                "shard_suffix must be a canonical M4 suffix from 00000 through 00009"
            )
        ordinal = self.record_ordinal
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, (int, np.integer))
            or int(ordinal) < 0
        ):
            raise ValueError("record_ordinal must be a non-negative integer")
        object.__setattr__(self, "record_ordinal", int(ordinal))

    @property
    def shard_index(self) -> int:
        return int(self.shard_suffix)


@dataclass(frozen=True, slots=True)
class M4ReloadExpectation:
    """One selected locator and the provenance it must reproduce on reload."""

    locator: M4ShardLocator
    expected_scenario_id: str
    expected_shard_sha256: str
    expected_dataset_config_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.locator, M4ShardLocator):
            raise TypeError("locator must be an M4ShardLocator")
        if (
            not isinstance(self.expected_scenario_id, str)
            or _SCENARIO_ID_PATTERN.fullmatch(self.expected_scenario_id) is None
        ):
            raise ValueError(
                "expected_scenario_id must be a non-empty hexadecimal string"
            )
        for name in (
            "expected_shard_sha256",
            "expected_dataset_config_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or _LOWER_SHA256_PATTERN.fullmatch(value) is None
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")


@dataclass(frozen=True, slots=True)
class M4StreamRecord:
    """One decoded record and its stable exact locator for pure classification."""

    locator: M4ShardLocator
    record: WaymaxRecord

    def __post_init__(self) -> None:
        if (
            self.record.shard_suffix != self.locator.shard_suffix
            or self.record.record_ordinal != self.locator.record_ordinal
        ):
            raise WaymaxDataError(
                "stream_locator_mismatch",
                "the decoded record provenance differs from its stream locator",
            )


class M4StreamCounters:
    """Independently advanced counters for one exact M4 shard stream.

    The loader advances ``raw_seen`` immediately after the raw TFRecord iterator
    yields bytes and ``decode_attempted`` immediately before decoding those bytes.
    It advances ``event_emitted`` only after the caller's pure event factory returns.
    Public attributes are read-only so callers cannot repair accounting drift.
    """

    __slots__ = (
        "_clean_eof",
        "_decode_attempted",
        "_event_emitted",
        "_raw_seen",
        "_shard_suffix",
    )

    def __init__(self, shard_suffix: str) -> None:
        M4ShardLocator(shard_suffix=shard_suffix, record_ordinal=0)
        self._shard_suffix = shard_suffix
        self._raw_seen = 0
        self._decode_attempted = 0
        self._event_emitted = 0
        self._clean_eof = False

    @property
    def shard_suffix(self) -> str:
        return self._shard_suffix

    @property
    def raw_seen(self) -> int:
        return self._raw_seen

    @property
    def decode_attempted(self) -> int:
        return self._decode_attempted

    @property
    def event_emitted(self) -> int:
        return self._event_emitted

    @property
    def clean_eof(self) -> bool:
        return self._clean_eof

    def _note_raw_seen(self, locator: M4ShardLocator) -> None:
        self._require_locator(locator, expected_ordinal=self._raw_seen)
        if self._clean_eof:
            raise WaymaxDataError(
                "stream_after_eof",
                "raw accounting cannot advance after clean EOF",
            )
        self._raw_seen += 1

    def _note_decode_attempted(self, locator: M4ShardLocator) -> None:
        self._require_locator(locator, expected_ordinal=self._decode_attempted)
        if self._raw_seen != self._decode_attempted + 1:
            raise WaymaxDataError(
                "stream_accounting_drift",
                "decode accounting did not immediately follow one raw record",
            )
        self._decode_attempted += 1

    def _note_event_emitted(self, locator: M4ShardLocator) -> None:
        self._require_locator(locator, expected_ordinal=self._event_emitted)
        if self._decode_attempted != self._event_emitted + 1:
            raise WaymaxDataError(
                "stream_accounting_drift",
                "event accounting did not immediately follow one decode",
            )
        self._event_emitted += 1

    def _note_clean_eof(self) -> None:
        if not (
            self._raw_seen
            == self._decode_attempted
            == self._event_emitted
        ):
            raise WaymaxDataError(
                "stream_accounting_drift",
                "clean EOF requires equal raw, decode, and event counts",
            )
        self._clean_eof = True

    def _require_locator(
        self,
        locator: M4ShardLocator,
        *,
        expected_ordinal: int,
    ) -> None:
        if (
            locator.shard_suffix != self._shard_suffix
            or locator.record_ordinal != expected_ordinal
        ):
            raise WaymaxDataError(
                "stream_ordinal_drift",
                "stream records must use contiguous ordinals within one exact shard",
            )


@dataclass(frozen=True, slots=True)
class WaymaxRejection:
    """A classified pre-selection rejection without source-derived payload values."""

    record_ordinal: int
    code: str


@dataclass(frozen=True, slots=True)
class WaymaxSelection:
    """The deterministic M3 selection plus its local-only reference record."""

    scenario: Scenario
    record: WaymaxRecord
    rejections: tuple[WaymaxRejection, ...]


def _validate_shard_index(shard_index: int) -> int:
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, (int, np.integer))
        or not 0 <= int(shard_index) < WOMD_TOTAL_VALIDATION_SHARDS
    ):
        raise ValueError(
            "shard_index must be an integer in "
            f"[0, {WOMD_TOTAL_VALIDATION_SHARDS})"
        )
    return int(shard_index)


def shard_suffix(shard_index: int) -> str:
    """Return the exact suffix used to identify one validation shard."""

    index = _validate_shard_index(shard_index)
    return f"tfrecord-{index:05d}-of-{WOMD_TOTAL_VALIDATION_SHARDS:05d}"


def resolve_validation_shard(
    data_dir: str | Path = DEFAULT_WOMD_VALIDATION_DIR,
    *,
    shard_index: int = WOMD_M3_SHARD_INDEX,
) -> Path:
    """Resolve exactly one shard while ignoring every other file in the directory."""

    directory = Path(data_dir)
    if not directory.is_dir():
        raise WaymaxDataError(
            "data_directory_missing",
            "the local WOMD validation directory does not exist",
        )
    suffix = shard_suffix(shard_index)
    matches = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.name.endswith(suffix)
        ),
        key=lambda path: path.name,
    )
    if not matches:
        raise WaymaxDataError(
            "shard_missing",
            f"no local validation file has suffix {suffix}",
        )
    if len(matches) != 1:
        raise WaymaxDataError(
            "shard_ambiguous",
            f"expected one local file with suffix {suffix}, found {len(matches)}",
        )
    return matches[0]


def resolve_m4_validation_shards(
    data_dir: str | Path = DEFAULT_WOMD_VALIDATION_DIR,
) -> tuple[Path, ...]:
    """Resolve only M4 suffixes ``00000`` through ``00009`` in canonical order.

    This deliberately delegates to the existing exact resolver ten times. It never
    expands the input with a glob, Waymax ``@N`` notation, or files such as ``00010``.
    """

    return tuple(
        resolve_validation_shard(data_dir, shard_index=shard_index)
        for shard_index in WOMD_M4_SHARD_INDICES
    )


def file_sha256(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a shard read-only for local provenance."""

    if (
        isinstance(chunk_bytes, bool)
        or not isinstance(chunk_bytes, int)
        or chunk_bytes <= 0
    ):
        raise ValueError("chunk_bytes must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> _FileIdentity:
    try:
        stat = path.stat()
    except OSError as exc:
        raise WaymaxDataError(
            "shard_stat_failed",
            "the exact shard could not be inspected read-only",
        ) from exc
    if not stat_module.S_ISREG(stat.st_mode):
        raise WaymaxDataError("shard_missing", "the exact shard path is not a file")
    return _FileIdentity(
        device=int(stat.st_dev),
        inode=int(stat.st_ino),
        size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
    )


def _guarded_shard_digest(
    path: str | Path,
) -> tuple[Path, _FileIdentity, str]:
    """Return a process-local cached digest guarded by file identity twice."""

    lexical = Path(path)
    if lexical.is_symlink():
        raise WaymaxDataError(
            "shard_symlink_forbidden",
            "M4 exact shard files may not be symbolic links",
        )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise WaymaxDataError(
            "shard_missing",
            "the exact shard path could not be resolved",
        ) from exc
    with _SHARD_DIGEST_CACHE_LOCK:
        before = _file_identity(resolved)
        cached = _SHARD_DIGEST_CACHE.get(resolved)
        if cached is not None and cached.identity == before:
            after = _file_identity(resolved)
            if after != before:
                raise WaymaxDataError(
                    "shard_changed",
                    "the exact shard identity changed while reusing its digest",
                )
            return resolved, after, cached.sha256

        digest = file_sha256(resolved)
        after = _file_identity(resolved)
        if after != before:
            raise WaymaxDataError(
                "shard_changed",
                "the exact shard identity changed while computing its digest",
            )
        entry = _ShardDigestCacheEntry(identity=after, sha256=digest)
        _SHARD_DIGEST_CACHE[resolved] = entry
    return resolved, after, digest


def m4_shard_sha256(path: str | Path) -> str:
    """Return the safely cached SHA-256 for one immutable local shard."""

    candidate = Path(path)
    _m4_shard_suffix_from_path(candidate)
    return _guarded_shard_digest(candidate)[2]


def clear_shard_digest_cache() -> None:
    """Clear only the process-local digest memoization (primarily for tests)."""

    with _SHARD_DIGEST_CACHE_LOCK:
        _SHARD_DIGEST_CACHE.clear()


def _assert_file_identity(path: Path, expected: _FileIdentity) -> None:
    if _file_identity(path) != expected:
        raise WaymaxDataError(
            "shard_changed",
            "the exact shard identity changed during record streaming",
        )


def _dataset_config_payload(dataset_config: Any) -> dict[str, Any]:
    """Canonicalize and fail closed on the actual Waymax DatasetConfig."""

    if not dataclasses.is_dataclass(dataset_config):
        raise WaymaxDataError(
            "dataset_config_drift",
            "the runtime dataset configuration must be a dataclass",
        )
    actual_fields = {field.name for field in dataclasses.fields(dataset_config)}
    if actual_fields != _DATASET_CONFIG_FIELD_NAMES:
        raise WaymaxDataError(
            "dataset_config_drift",
            "the runtime DatasetConfig field set differs from the locked M3 schema",
        )
    path = getattr(dataset_config, "path")
    if not isinstance(path, str) or not path:
        raise WaymaxDataError(
            "dataset_config_drift",
            "the runtime DatasetConfig path must be a non-empty string",
        )

    payload: dict[str, Any] = {
        "womd_split": "validation",
        "womd_version": "1.3.1",
    }
    for name in sorted(_DATASET_CONFIG_FIELD_NAMES.difference({"path"})):
        value = getattr(dataset_config, name)
        if name == "data_format":
            value = getattr(value, "value", value)
        elif name == "batch_dims":
            value = list(value)
        payload[name] = value
    if payload != _EXPECTED_DATASET_CONFIG_PAYLOAD:
        raise WaymaxDataError(
            "dataset_config_drift",
            "the actual runtime DatasetConfig differs from the locked M3 values",
        )
    return payload


def dataset_config_fingerprint(dataset_config: Any | None = None) -> str:
    """Fingerprint the actual path-independent, fully frozen loader configuration."""

    payload = (
        dict(_EXPECTED_DATASET_CONFIG_PAYLOAD)
        if dataset_config is None
        else _dataset_config_payload(dataset_config)
    )

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_waymax_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import jax
        import tensorflow as tf
        from waymax import config
        from waymax import dataloader
    except ImportError as exc:
        raise WaymaxDependencyError(
            "Waymax support is optional; install it with "
            "`uv sync --extra dev --extra waymo`."
        ) from exc

    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        # TensorFlow may already be initialized by a caller. The CLI also sets the
        # documented bounded local execution path before importing this module.
        pass
    return jax, tf, config, dataloader


def runtime_summary() -> dict[str, Any]:
    """Return non-sensitive compatibility facts for an ignored local report."""

    jax, tf, _, _ = _require_waymax_runtime()
    import flax
    import jaxlib

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "tensorflow": tf.__version__,
        "flax": flax.__version__,
        "waymo_waymax": importlib.metadata.version("waymo-waymax"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [device.platform for device in jax.devices()],
    }


def _make_dataset_config(shard_path: Path, config: Any) -> Any:
    dataset_config = dataclasses.replace(
        config.WOD_1_3_1_VALIDATION,
        path=str(shard_path),
        repeat=1,
        batch_dims=(),
        shuffle_seed=None,
        shuffle_buffer_size=1000,
        num_shards=1,
        deterministic=True,
        include_sdc_paths=True,
        aggregate_timesteps=True,
        max_num_rg_points=30000,
        max_num_objects=128,
        num_paths=45,
        num_points_per_path=800,
        drop_remainder=False,
        tf_data_service_address=None,
        distributed=False,
        batch_by_scenario=True,
        data_format=config.DataFormat.TFRECORD,
    )
    _dataset_config_payload(dataset_config)
    return dataset_config


def _scenario_id_feature_spec(tf: Any) -> Any:
    return tf.io.FixedLenFeature([1], tf.string)


def _preserve_parsed_scenario_id(
    parsed: dict[str, Any],
    tf: Any,
) -> dict[str, Any]:
    if "scenario/id" not in parsed:
        raise WaymaxDataError(
            "scenario_id_missing",
            "the parsed TFExample omitted the native scenario/id field",
        )
    raw_id = parsed.pop("scenario/id")
    parsed["scenario/id"] = tf.io.decode_raw(raw_id, tf.uint8)
    return parsed


def _make_preprocess(dataset_config: Any, tf: Any, dataloader: Any) -> Any:
    features = dataloader.womd_utils.get_features_description(
        include_sdc_paths=dataset_config.include_sdc_paths,
        max_num_rg_points=dataset_config.max_num_rg_points,
        num_paths=dataset_config.num_paths,
        num_points_per_path=dataset_config.num_points_per_path,
    )
    features["scenario/id"] = _scenario_id_feature_spec(tf)

    def preprocess(serialized: bytes) -> dict[str, Any]:
        parsed = tf.io.parse_example(serialized, features)
        parsed = _preserve_parsed_scenario_id(parsed, tf)
        return dataloader.preprocess_womd_example(
            parsed,
            aggregate_timesteps=dataset_config.aggregate_timesteps,
            max_num_objects=dataset_config.max_num_objects,
        )

    return preprocess


def _make_postprocess(dataset_config: Any, dataloader: Any) -> Any:
    def postprocess(example: dict[str, Any]) -> tuple[Any, dict[str, Any], Any]:
        state = dataloader.simulator_state_from_womd_dict(
            example,
            include_sdc_paths=dataset_config.include_sdc_paths,
        )
        audit = {key: example[key] for key in _AUDIT_KEYS}
        return example["scenario/id"], audit, state

    return postprocess


def _decode_scenario_id(value: Any) -> str:
    encoded = np.asarray(value)
    if (
        encoded.dtype != np.uint8
        or encoded.ndim != 2
        or encoded.shape[0] != 1
        or encoded.shape[1] == 0
    ):
        raise WaymaxDataError(
            "scenario_id_encoding",
            "the native WOMD scenario ID must be exact uint8 shape [1, N] "
            "with N greater than zero",
        )
    try:
        decoded = encoded[0].tobytes().decode(
            "utf-8",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise WaymaxDataError(
            "scenario_id_decode",
            "the native WOMD scenario ID is not valid UTF-8 bytes",
        ) from exc
    if not decoded or _SCENARIO_ID_PATTERN.fullmatch(decoded) is None:
        raise WaymaxDataError(
            "scenario_id_format",
            "the native WOMD scenario ID is not a non-empty hexadecimal string",
        )
    return decoded


def _freeze_audit(audit: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for key in _AUDIT_KEYS:
        if key not in audit:
            raise WaymaxDataError(
                "audit_field_missing",
                f"the preprocessed source record omitted required field {key}",
            )
        array = np.array(audit[key], copy=True)
        array.setflags(write=False)
        frozen[key] = array
    return MappingProxyType(frozen)


def _m4_shard_suffix_from_path(path: Path) -> str:
    if path.is_symlink():
        raise WaymaxDataError(
            "shard_symlink_forbidden",
            "M4 exact shard files may not be symbolic links",
        )
    if not path.is_file():
        raise WaymaxDataError("shard_missing", "the exact shard path is not a file")
    matched_suffix = re.search(r"tfrecord-(\d{5})-of-(\d{5})$", path.name)
    if matched_suffix is None:
        raise WaymaxDataError(
            "shard_name_invalid",
            "the local filename does not end in a WOMD TFRecord shard suffix",
        )
    if int(matched_suffix.group(2)) != WOMD_TOTAL_VALIDATION_SHARDS:
        raise WaymaxDataError(
            "shard_total_invalid",
            "the shard suffix does not identify the 150-file validation split",
        )
    suffix = matched_suffix.group(1)
    if int(suffix) not in WOMD_M4_SHARD_INDICES:
        raise WaymaxDataError(
            "m4_shard_out_of_scope",
            "M4 record streaming is restricted to suffixes 00000 through 00009",
        )
    return suffix


def _verified_m4_scan_event(stream_record: M4StreamRecord) -> Any:
    """Construct the only permitted event after frozen source/adapter gates."""

    from .waymax import (
        DEFAULT_WAYMAX_TEMPORAL_PROFILE,
        scenario_from_waymax_state,
    )
    from .waymax_cohort import (
        ScanEvent,
        source_rejection_code,
    )

    record = stream_record.record
    rejection = source_rejection_code(record.audit)
    event_kwargs = {
        "shard_suffix": stream_record.locator.shard_suffix,
        "record_ordinal": stream_record.locator.record_ordinal,
        "native_scenario_id": record.scenario_id,
        "shard_sha256": record.shard_sha256,
        "dataset_config_fingerprint": record.dataset_config_fingerprint,
    }
    if rejection is not None:
        return ScanEvent.rejected_event(
            **event_kwargs,
            rejection_code=rejection,
        )

    scenario = scenario_from_waymax_state(
        record.state,
        scenario_id=record.scenario_id,
        temporal_profile=DEFAULT_WAYMAX_TEMPORAL_PROFILE,
        provenance={
            "dataset_config_fingerprint": record.dataset_config_fingerprint,
            "record_ordinal": record.record_ordinal,
            "shard_sha256": record.shard_sha256,
            "shard_suffix": record.shard_suffix,
        },
    )
    validate_record_parity(record, scenario)
    return ScanEvent.eligible_event(**event_kwargs)


@dataclass(frozen=True, slots=True)
class _M4RuntimeDecoder:
    tf: Any
    preprocess: Callable[[bytes], Mapping[str, Any]]
    postprocess: Callable[
        [Mapping[str, Any]],
        tuple[Any, Mapping[str, Any], Any],
    ]
    dataset_config_fingerprint: str

    def decode(
        self,
        serialized: bytes,
        *,
        locator: M4ShardLocator,
        shard_sha256: str,
    ) -> WaymaxRecord:
        parsed = self.preprocess(serialized)
        numpy_parsed = self.tf.nest.map_structure(
            lambda value: (
                value.numpy()
                if hasattr(value, "numpy")
                else np.asarray(value)
            ),
            parsed,
        )
        scenario_id = _decode_scenario_id(numpy_parsed["scenario/id"])
        audit = _freeze_audit(numpy_parsed)
        _, _, state = self.postprocess(numpy_parsed)
        return WaymaxRecord(
            scenario_id=scenario_id,
            state=state,
            audit=audit,
            shard_suffix=locator.shard_suffix,
            record_ordinal=locator.record_ordinal,
            shard_sha256=shard_sha256,
            dataset_config_fingerprint=self.dataset_config_fingerprint,
        )


def _make_m4_runtime_decoder(shard_path: Path) -> _M4RuntimeDecoder:
    jax, tf, config, dataloader = _require_waymax_runtime()
    dataset_config = _make_dataset_config(shard_path, config)
    return _M4RuntimeDecoder(
        tf=tf,
        preprocess=_make_preprocess(dataset_config, tf, dataloader),
        postprocess=jax.jit(
            _make_postprocess(dataset_config, dataloader)
        ),
        dataset_config_fingerprint=dataset_config_fingerprint(dataset_config),
    )


def _iter_raw_serialized_records(
    shard_path: Path,
    tf: Any,
) -> Iterator[tuple[int, bytes]]:
    """Yield raw TFRecords in physical order without preprocessing or prefetch."""

    raw_iterator = tf.compat.v1.io.tf_record_iterator(str(shard_path))
    for ordinal, serialized in enumerate(raw_iterator):
        if not isinstance(serialized, (bytes, bytearray, memoryview)):
            raise WaymaxDataError(
                "raw_record_type",
                "the TensorFlow record reader did not return serialized bytes",
            )
        yield ordinal, bytes(serialized)


def iter_m4_waymax_records(
    shard_path: str | Path,
    *,
    counters: M4StreamCounters,
    event_factory: Callable[[M4StreamRecord], _EventT] | None = None,
) -> Iterator[Any]:
    """Classify every raw record in one exact M4 shard through clean EOF.

    The loader itself applies the frozen source predicate and, for every eligible
    record, the adapter/contract/independent-parity gates before constructing the
    canonical ``ScanEvent``.  An optional ``event_factory`` is only a contradiction
    hook: its result must equal that verified event exactly. Conversion, contract,
    parity, source-invariant, and all other exceptions propagate unchanged.
    """

    path = Path(shard_path)
    suffix = _m4_shard_suffix_from_path(path)
    if counters.shard_suffix != suffix:
        raise WaymaxDataError(
            "stream_counter_shard_mismatch",
            "the counters belong to a different exact shard",
        )
    if (
        counters.raw_seen
        or counters.decode_attempted
        or counters.event_emitted
        or counters.clean_eof
    ):
        raise WaymaxDataError(
            "stream_counters_reused",
            "each shard scan requires fresh counters",
        )
    if event_factory is not None and not callable(event_factory):
        raise TypeError("event_factory must be callable or None")

    resolved, identity, shard_digest = _guarded_shard_digest(path)
    resolved_suffix = _m4_shard_suffix_from_path(resolved)
    if resolved_suffix != suffix:
        raise WaymaxDataError(
            "m4_shard_scope_mismatch",
            "the resolved shard target differs from its exact M4 suffix",
        )
    decoder = _make_m4_runtime_decoder(resolved)
    for ordinal, serialized in _iter_raw_serialized_records(
        resolved,
        decoder.tf,
    ):
        locator = M4ShardLocator(
            shard_suffix=suffix,
            record_ordinal=ordinal,
        )
        counters._note_raw_seen(locator)
        counters._note_decode_attempted(locator)
        record = decoder.decode(
            serialized,
            locator=locator,
            shard_sha256=shard_digest,
        )
        stream_record = M4StreamRecord(locator=locator, record=record)
        verified_event = _verified_m4_scan_event(stream_record)
        if event_factory is None:
            event = verified_event
        else:
            event = event_factory(stream_record)
            if event != verified_event:
                raise WaymaxDataError(
                    "stream_event_contradiction",
                    "the event factory contradicted the verified scan event",
                )
        counters._note_event_emitted(locator)
        yield event

    _assert_file_identity(resolved, identity)
    counters._note_clean_eof()


def reload_m4_waymax_records(
    data_dir: str | Path,
    expectations: Sequence[M4ReloadExpectation],
) -> tuple[M4StreamRecord, ...]:
    """Reload selected locators with one decoder and one stream per shard.

    The result preserves the caller's order. Locators must be unique, while the
    physical work is grouped by exact shard suffix so the CLI cannot accidentally
    rebuild Waymax's decoder for every member of the selected cohort.
    """

    requested = tuple(expectations)
    for expectation in requested:
        if not isinstance(expectation, M4ReloadExpectation):
            raise TypeError(
                "expectations must contain only M4ReloadExpectation values"
            )
    locators = tuple(expectation.locator for expectation in requested)
    if len(set(locators)) != len(locators):
        raise WaymaxDataError(
            "locator_duplicate",
            "selected reload locators must be unique",
        )
    if not requested:
        return ()

    grouped: dict[str, list[tuple[int, M4ReloadExpectation]]] = {}
    for result_index, expectation in enumerate(requested):
        grouped.setdefault(
            expectation.locator.shard_suffix,
            [],
        ).append((result_index, expectation))

    results: list[M4StreamRecord | None] = [None] * len(requested)
    for suffix in sorted(grouped):
        shard_path = resolve_validation_shard(
            data_dir,
            shard_index=int(suffix),
        )
        actual_suffix = _m4_shard_suffix_from_path(shard_path)
        if actual_suffix != suffix:
            raise WaymaxDataError(
                "locator_shard_mismatch",
                "the exact resolver returned a different shard suffix",
            )

        resolved, identity, shard_digest = _guarded_shard_digest(shard_path)
        resolved_suffix = _m4_shard_suffix_from_path(resolved)
        if resolved_suffix != suffix:
            raise WaymaxDataError(
                "m4_shard_scope_mismatch",
                "the resolved shard target differs from its exact M4 suffix",
            )
        decoder = _make_m4_runtime_decoder(resolved)
        shard_requests = grouped[suffix]
        for _, expectation in shard_requests:
            if (
                shard_digest != expectation.expected_shard_sha256
                or decoder.dataset_config_fingerprint
                != expectation.expected_dataset_config_fingerprint
            ):
                raise WaymaxDataError(
                    "locator_provenance_mismatch",
                    "the reloaded shard or dataset configuration differs from "
                    "the manifest",
                )

        by_ordinal = {
            expectation.locator.record_ordinal: (result_index, expectation)
            for result_index, expectation in shard_requests
        }
        remaining = set(by_ordinal)
        for ordinal, serialized in _iter_raw_serialized_records(
            resolved,
            decoder.tf,
        ):
            if ordinal not in remaining:
                continue
            result_index, expectation = by_ordinal[ordinal]
            record = decoder.decode(
                serialized,
                locator=expectation.locator,
                shard_sha256=shard_digest,
            )
            stream_record = M4StreamRecord(
                locator=expectation.locator,
                record=record,
            )
            if record.scenario_id != expectation.expected_scenario_id:
                raise WaymaxDataError(
                    "locator_identity_mismatch",
                    "the reloaded native identity differs from the manifest",
                )
            if (
                record.shard_sha256 != expectation.expected_shard_sha256
                or record.dataset_config_fingerprint
                != expectation.expected_dataset_config_fingerprint
            ):
                raise WaymaxDataError(
                    "locator_provenance_mismatch",
                    "the reloaded record provenance differs from the manifest",
                )
            results[result_index] = stream_record
            remaining.remove(ordinal)
            if not remaining:
                break

        _assert_file_identity(resolved, identity)
        if remaining:
            raise WaymaxDataError(
                "locator_ordinal_missing",
                "the exact shard ended before a requested record ordinal",
            )

    if any(item is None for item in results):
        raise AssertionError("grouped M4 reload did not populate every result")
    return tuple(item for item in results if item is not None)


def reload_m4_waymax_record(
    data_dir: str | Path,
    locator: M4ShardLocator,
    *,
    expected_scenario_id: str,
    expected_shard_sha256: str,
    expected_dataset_config_fingerprint: str,
) -> M4StreamRecord:
    """Reload one exact locator and fail closed on identity or provenance drift."""

    return reload_m4_waymax_records(
        data_dir,
        (
            M4ReloadExpectation(
                locator=locator,
                expected_scenario_id=expected_scenario_id,
                expected_shard_sha256=expected_shard_sha256,
                expected_dataset_config_fingerprint=(
                    expected_dataset_config_fingerprint
                ),
            ),
        ),
    )[0]


def iter_waymax_records(
    shard_path: str | Path,
    *,
    max_records: int | None = None,
) -> Iterator[WaymaxRecord]:
    """Yield bounded records from one exact shard using Waymax's parser/factory."""

    if (
        max_records is not None
        and (
            isinstance(max_records, bool)
            or not isinstance(max_records, int)
            or max_records <= 0
        )
    ):
        raise ValueError("max_records must be a positive integer or None")
    path = Path(shard_path)
    if not path.is_file():
        raise WaymaxDataError("shard_missing", "the exact shard path is not a file")
    matched_suffix = re.search(r"tfrecord-(\d{5})-of-(\d{5})$", path.name)
    if matched_suffix is None:
        raise WaymaxDataError(
            "shard_name_invalid",
            "the local filename does not end in a WOMD TFRecord shard suffix",
        )
    if int(matched_suffix.group(2)) != WOMD_TOTAL_VALIDATION_SHARDS:
        raise WaymaxDataError(
            "shard_total_invalid",
            "the shard suffix does not identify the 150-file validation split",
        )

    _, tf, config, dataloader = _require_waymax_runtime()
    dataset_config = _make_dataset_config(path, config)
    preprocess = _make_preprocess(dataset_config, tf, dataloader)
    postprocess = _make_postprocess(dataset_config, dataloader)
    generator = dataloader.get_data_generator(
        dataset_config,
        preprocess,
        postprocess,
    )
    content_fingerprint = file_sha256(path)
    config_fingerprint = dataset_config_fingerprint(dataset_config)
    suffix = matched_suffix.group(1)

    bounded_generator = (
        generator
        if max_records is None
        else itertools.islice(generator, max_records)
    )
    for ordinal, (raw_id, audit, state) in enumerate(bounded_generator):
        yield WaymaxRecord(
            scenario_id=_decode_scenario_id(raw_id),
            state=state,
            audit=_freeze_audit(audit),
            shard_suffix=suffix,
            record_ordinal=ordinal,
            shard_sha256=content_fingerprint,
            dataset_config_fingerprint=config_fingerprint,
        )


def validate_record_parity(
    record: WaymaxRecord,
    scenario: Scenario,
) -> Mapping[str, bool]:
    """Cross-check a conversion directly against the preprocessed source tensors.

    This intentionally duplicates the small expected-value calculation instead of
    calling adapter helpers. It returns booleans only; no source-derived values are
    suitable for tracked or deployed evidence.
    """

    from .waymax import (
        MAX_MAP_DIRECTION_ERROR_DEGREES,
        MAX_MAP_SEGMENT_METERS,
        MIN_MAP_SEGMENT_METERS,
    )

    if scenario.scenario_id != record.scenario_id:
        raise WaymaxDataError(
            "parity_scenario_id",
            "the EvalSim scenario did not preserve the internal native identity",
        )

    audit = record.audit
    valid = np.asarray(audit["state/all/valid"], dtype=bool)
    if valid.ndim != 2:
        raise WaymaxDataError(
            "parity_shape",
            "the reference validity tensor must be [objects, timesteps]",
        )
    retained = np.flatnonzero(np.any(valid, axis=1))
    if len(scenario.agents) != retained.size:
        raise WaymaxDataError(
            "parity_agent_count",
            "the retained agent count differs from the independent reference",
        )

    which_time = np.asarray(audit["state/which_time"])
    expected_time_partition = np.concatenate(
        (
            -np.ones(10, dtype=which_time.dtype),
            np.zeros(1, dtype=which_time.dtype),
            np.ones(80, dtype=which_time.dtype),
        )
    )
    if (
        which_time.ndim != 1
        or not np.array_equal(which_time, expected_time_partition)
        or valid.shape[1] != which_time.size
        or scenario.num_steps != which_time.size
        or scenario.metadata.get("current_index") != 10
        or int(np.flatnonzero(which_time == 0)[0]) != 10
    ):
        raise WaymaxDataError(
            "parity_time_boundary",
            "the real source past/current/future partition differs from the "
            "locked 10/1/80 boundary",
        )

    raw_ids = np.asarray(audit["state/id"])
    raw_types = np.asarray(audit["state/type"])
    raw_sdc = np.asarray(audit["state/is_sdc"], dtype=bool)
    type_map = {
        1: AgentType.VEHICLE,
        2: AgentType.PEDESTRIAN,
        3: AgentType.CYCLIST,
    }
    retained_sdc = raw_sdc[retained]
    if np.count_nonzero(retained_sdc) != 1:
        raise WaymaxDataError(
            "parity_sdc",
            "the independent reference does not contain exactly one retained SDC",
        )
    if scenario.ego_index != int(np.flatnonzero(retained_sdc)[0]):
        raise WaymaxDataError(
            "parity_sdc",
            "the EvalSim ego index differs from the independent reference",
        )

    direct_fields = {
        "x": "state/all/x",
        "y": "state/all/y",
        "vx": "state/all/velocity_x",
        "vy": "state/all/velocity_y",
    }
    raw_yaw = np.asarray(audit["state/all/bbox_yaw"])
    raw_length = np.asarray(audit["state/all/length"])
    raw_width = np.asarray(audit["state/all/width"])
    raw_timestamps = np.asarray(audit["state/all/timestamp_micros"])

    for target_index, source_index in enumerate(retained):
        agent = scenario.agents[target_index]
        source_valid = valid[source_index]
        if int(agent.id) != int(raw_ids[source_index]):
            raise WaymaxDataError(
                "parity_agent_id",
                "an EvalSim agent ID differs from the independent reference",
            )
        expected_type = type_map.get(
            int(raw_types[source_index]),
            AgentType.UNKNOWN,
        )
        if agent.type != expected_type:
            raise WaymaxDataError(
                "parity_agent_type",
                "an EvalSim agent type differs from the independent reference",
            )
        if not np.array_equal(agent.valid, source_valid):
            raise WaymaxDataError(
                "parity_validity",
                "an EvalSim validity mask differs from the independent reference",
            )
        for target_name, source_name in direct_fields.items():
            expected = np.asarray(
                audit[source_name][source_index],
                dtype=np.float64,
            )
            actual = np.asarray(getattr(agent, target_name), dtype=np.float64)
            if not np.array_equal(actual[source_valid], expected[source_valid]):
                raise WaymaxDataError(
                    "parity_trajectory",
                    f"valid {target_name} values differ from the independent reference",
                )
            if not np.array_equal(
                actual[~source_valid],
                np.zeros(np.count_nonzero(~source_valid), dtype=np.float64),
            ):
                raise WaymaxDataError(
                    "parity_invalid_fill",
                    f"invalid {target_name} payloads are not deterministic zeroes",
                )

        expected_yaw = np.asarray(
            raw_yaw[source_index],
            dtype=np.float64,
        )
        expected_yaw = (
            (expected_yaw + np.pi) % (2.0 * np.pi) - np.pi
        )
        actual_yaw = np.asarray(agent.heading, dtype=np.float64)
        yaw_delta = (
            (actual_yaw[source_valid] - expected_yaw[source_valid] + np.pi)
            % (2.0 * np.pi)
            - np.pi
        )
        if not np.allclose(yaw_delta, 0.0, rtol=0.0, atol=1e-6):
            raise WaymaxDataError(
                "parity_heading",
                "valid circular heading values differ from the reference",
            )
        if not np.array_equal(
            actual_yaw[~source_valid],
            np.zeros(np.count_nonzero(~source_valid), dtype=np.float64),
        ):
            raise WaymaxDataError(
                "parity_invalid_fill",
                "invalid heading payloads are not deterministic zeroes",
            )

        expected_length = _reference_masked_mean(
            raw_length[source_index],
            source_valid,
        )
        expected_width = _reference_masked_mean(
            raw_width[source_index],
            source_valid,
        )
        if not math.isclose(
            agent.length,
            expected_length,
            rel_tol=0.0,
            abs_tol=1e-6,
        ) or not math.isclose(
            agent.width,
            expected_width,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise WaymaxDataError(
                "parity_dimensions",
                "scalar dimensions differ from the independent masked mean",
            )

    canonical_micros = np.empty(valid.shape[1], dtype=np.int64)
    for step in range(valid.shape[1]):
        contributors = raw_timestamps[retained, step][valid[retained, step]]
        if contributors.size == 0 or not np.all(contributors == contributors[0]):
            raise WaymaxDataError(
                "parity_timestamps",
                "the independent source timestamp consensus is invalid",
            )
        canonical_micros[step] = int(contributors[0])
    expected_timestamps = (
        canonical_micros - canonical_micros[0]
    ).astype(np.float64) * 1e-6
    if not np.allclose(
        scenario.timestamps,
        expected_timestamps,
        rtol=0.0,
        atol=1e-6,
    ):
        raise WaymaxDataError(
            "parity_timestamps",
            "normalized timestamps differ from the independent reference",
        )

    expected_map = _independent_reference_map(
        audit,
        min_segment=MIN_MAP_SEGMENT_METERS,
        max_segment=MAX_MAP_SEGMENT_METERS,
        max_direction_error_degrees=MAX_MAP_DIRECTION_ERROR_DEGREES,
    )
    if len(scenario.map) != len(expected_map):
        raise WaymaxDataError(
            "parity_map_count",
            "retained map group count differs from the independent reference",
        )
    for actual, (expected_type, expected_xy) in zip(
        scenario.map,
        expected_map,
    ):
        if actual.type != expected_type or not np.array_equal(
            np.asarray(actual.xy, dtype=np.float64),
            expected_xy,
        ):
            raise WaymaxDataError(
                "parity_map",
                "retained map type/order/coordinates differ from the reference",
            )

    if not (
        scenario.metadata.get("dataset_config_fingerprint")
        == record.dataset_config_fingerprint
        and scenario.metadata.get("shard_sha256") == record.shard_sha256
        and scenario.metadata.get("shard_suffix") == record.shard_suffix
        and scenario.metadata.get("record_ordinal") == record.record_ordinal
    ):
        raise WaymaxDataError(
            "parity_provenance",
            "EvalSim provenance differs from the local source record",
        )
    json.dumps(scenario.metadata, allow_nan=False, sort_keys=True)
    return MappingProxyType(
        {
            "native_identity": True,
            "agents": True,
            "time_boundary": True,
            "timestamps": True,
            "map": True,
            "provenance": True,
        }
    )


def _reference_masked_mean(values: np.ndarray, valid: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float32)[valid]
    if selected.size == 0:
        raise WaymaxDataError(
            "parity_dimensions",
            "a retained reference object has no valid dimension samples",
        )
    return float(np.sum(selected, dtype=np.float32) / np.float32(selected.size))


def _independent_reference_map(
    audit: Mapping[str, np.ndarray],
    *,
    min_segment: float,
    max_segment: float,
    max_direction_error_degrees: float,
) -> list[tuple[MapType, np.ndarray]]:
    xyz = np.asarray(audit["roadgraph_samples/xyz"])
    directions = np.asarray(audit["roadgraph_samples/dir"])
    types = np.asarray(audit["roadgraph_samples/type"])[..., 0]
    ids = np.asarray(audit["roadgraph_samples/id"])[..., 0]
    valid = np.asarray(audit["roadgraph_samples/valid"])[..., 0].astype(bool)
    groups: OrderedDict[int, list[int]] = OrderedDict()
    for source_index in np.flatnonzero(valid):
        groups.setdefault(int(ids[source_index]), []).append(int(source_index))

    expected: list[tuple[MapType, np.ndarray]] = []
    threshold = math.cos(math.radians(max_direction_error_degrees))
    for source_indices in groups.values():
        indices = np.asarray(source_indices, dtype=int)
        group_types = np.unique(types[indices])
        if group_types.size != 1:
            continue
        source_type = int(group_types[0])
        if source_type in {0, 1, 2, 3}:
            target_type = MapType.LANE
        elif source_type in {14, 15, 16}:
            target_type = MapType.ROAD_EDGE
        else:
            continue
        xy = np.asarray(xyz[indices, :2], dtype=np.float64)
        direction_xy = np.asarray(directions[indices, :2], dtype=np.float64)
        if (
            not np.all(np.isfinite(xy))
            or not np.all(np.isfinite(direction_xy))
            or xy.shape[0] < 2
            or np.unique(xy, axis=0).shape[0] < 2
        ):
            continue
        segments = np.diff(xy, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(lengths <= min_segment) or np.any(lengths > max_segment):
            continue
        source_directions = direction_xy[:-1]
        norms = np.linalg.norm(source_directions, axis=1)
        if np.any(norms <= 0.0):
            continue
        cosines = np.sum(
            (segments / lengths[:, np.newaxis])
            * (source_directions / norms[:, np.newaxis]),
            axis=1,
        )
        if np.any(cosines < threshold):
            continue
        expected.append((target_type, np.array(xy, copy=True)))
    return expected


def _eligibility_rejection(scenario: Scenario) -> str | None:
    current = scenario.metadata.get("current_index")
    if not isinstance(current, int) or isinstance(current, bool):
        return "invalid_current_index"
    if not 0 <= current < scenario.num_steps - 1:
        return "no_future_transition"
    if not np.all(scenario.ego.valid[current : current + 2]):
        return "ego_transition_invalid"
    if not any(
        feature.type in (MapType.LANE, MapType.ROAD_EDGE)
        for feature in scenario.map
    ):
        return "no_supported_map"
    if not any(
        index != scenario.ego_index
        and agent.type == AgentType.VEHICLE
        and agent.valid[current]
        and agent.valid[current + 1]
        for index, agent in enumerate(scenario.agents)
    ):
        return "no_world_vehicle_transition"
    return None


@dataclass(frozen=True, slots=True)
class WaymaxSource:
    """A deterministic, exact-shard local WOMD source for the M3 vertical slice."""

    data_dir: Path = DEFAULT_WOMD_VALIDATION_DIR
    shard_index: int = WOMD_M3_SHARD_INDEX
    search_limit: int = WOMD_M3_SEARCH_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(
            self,
            "shard_index",
            _validate_shard_index(self.shard_index),
        )
        if (
            isinstance(self.search_limit, bool)
            or not isinstance(self.search_limit, int)
            or self.search_limit <= 0
        ):
            raise ValueError("search_limit must be a positive integer")

    @property
    def shard_path(self) -> Path:
        return resolve_validation_shard(
            self.data_dir,
            shard_index=self.shard_index,
        )

    def iter_records(self, *, max_records: int | None = None) -> Iterator[WaymaxRecord]:
        return iter_waymax_records(self.shard_path, max_records=max_records)

    def scenario_from_record(self, record: WaymaxRecord) -> Scenario:
        from .waymax import DEFAULT_WAYMAX_TEMPORAL_PROFILE
        from .waymax import scenario_from_waymax_state

        return scenario_from_waymax_state(
            record.state,
            scenario_id=record.scenario_id,
            temporal_profile=DEFAULT_WAYMAX_TEMPORAL_PROFILE,
            provenance={
                "dataset_config_fingerprint": record.dataset_config_fingerprint,
                "record_ordinal": record.record_ordinal,
                "shard_sha256": record.shard_sha256,
                "shard_suffix": record.shard_suffix,
            },
        )

    def load_first_eligible(self) -> WaymaxSelection:
        """Apply the pre-registered earliest-eligible rule within ``search_limit``."""

        from .waymax import WaymaxConversionError

        rejections: list[WaymaxRejection] = []
        for record in self.iter_records(max_records=self.search_limit):
            try:
                scenario = self.scenario_from_record(record)
            except WaymaxConversionError as exc:
                rejections.append(
                    WaymaxRejection(
                        record_ordinal=record.record_ordinal,
                        code=exc.code,
                    )
                )
                continue
            rejection = _eligibility_rejection(scenario)
            if rejection is not None:
                rejections.append(
                    WaymaxRejection(
                        record_ordinal=record.record_ordinal,
                        code=rejection,
                    )
                )
                continue
            return WaymaxSelection(
                scenario=scenario,
                record=record,
                rejections=tuple(rejections),
            )

        counts = Counter(item.code for item in rejections)
        summary = ", ".join(
            f"{code}={count}" for code, count in sorted(counts.items())
        )
        raise WaymaxDataError(
            "no_eligible_record",
            "no record satisfied the pre-registered M3 rule within the bounded "
            f"search ({summary or 'no records decoded'})",
        )


__all__ = [
    "DEFAULT_WOMD_VALIDATION_DIR",
    "LOCAL_WAYMO_ENV_FLAG",
    "M4ReloadExpectation",
    "M4ShardLocator",
    "M4StreamCounters",
    "M4StreamRecord",
    "WOMD_M3_SEARCH_LIMIT",
    "WOMD_M3_SHARD_INDEX",
    "WOMD_M4_SHARD_INDICES",
    "WOMD_TOTAL_VALIDATION_SHARDS",
    "WaymaxDataError",
    "WaymaxDependencyError",
    "WaymaxRecord",
    "WaymaxRejection",
    "WaymaxSelection",
    "WaymaxSource",
    "clear_shard_digest_cache",
    "dataset_config_fingerprint",
    "file_sha256",
    "iter_m4_waymax_records",
    "iter_waymax_records",
    "m4_shard_sha256",
    "reload_m4_waymax_record",
    "reload_m4_waymax_records",
    "resolve_m4_validation_shards",
    "resolve_validation_shard",
    "runtime_summary",
    "shard_suffix",
    "validate_record_parity",
]
