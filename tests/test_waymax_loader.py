"""M3 loader tests that do not read or commit WOMD payloads."""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from evalsim.sources.waymax_loader import (
    WaymaxDataError,
    WaymaxSource,
    _decode_scenario_id,
    _preserve_parsed_scenario_id,
    _scenario_id_feature_spec,
    dataset_config_fingerprint,
    file_sha256,
    iter_waymax_records,
    resolve_validation_shard,
    shard_suffix,
)


def test_exact_shard_resolution_ignores_other_suffixes(tmp_path) -> None:
    expected = tmp_path / (
        "uncompressed_tf_example_validation_validation_tfexample."
        "tfrecord-00000-of-00150"
    )
    expected.write_bytes(b"m3-synthetic-placeholder")
    (tmp_path / "validation_tfexample.tfrecord-00010-of-00150").write_bytes(
        b"out-of-scope"
    )

    assert resolve_validation_shard(tmp_path, shard_index=0) == expected


def test_exact_shard_resolution_rejects_missing_and_ambiguous(tmp_path) -> None:
    with pytest.raises(WaymaxDataError, match="shard_missing"):
        resolve_validation_shard(tmp_path, shard_index=0)

    suffix = shard_suffix(0)
    (tmp_path / f"first-{suffix}").write_bytes(b"one")
    (tmp_path / f"second-{suffix}").write_bytes(b"two")
    with pytest.raises(WaymaxDataError, match="shard_ambiguous"):
        resolve_validation_shard(tmp_path, shard_index=0)


@pytest.mark.parametrize("value", [True, -1, 150, 1.5, "0"])
def test_shard_index_validation(value) -> None:
    with pytest.raises(ValueError, match="shard_index"):
        shard_suffix(value)


def test_file_and_config_fingerprints_are_stable(tmp_path) -> None:
    payload = b"not-a-real-tfrecord"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()
    fingerprint = dataset_config_fingerprint()
    assert len(fingerprint) == 64
    assert fingerprint == dataset_config_fingerprint()


@dataclasses.dataclass(frozen=True)
class _FakeDatasetConfig:
    path: str = "ignored/local/exact-shard"
    data_format: str = "TFRECORD"
    repeat: int = 1
    batch_dims: tuple[int, ...] = ()
    shuffle_seed: int | None = None
    shuffle_buffer_size: int = 1000
    num_shards: int = 1
    deterministic: bool = True
    include_sdc_paths: bool = True
    aggregate_timesteps: bool = True
    max_num_rg_points: int = 30000
    max_num_objects: int = 128
    num_paths: int = 45
    num_points_per_path: int = 800
    drop_remainder: bool = False
    tf_data_service_address: str | None = None
    distributed: bool = False
    batch_by_scenario: bool = True


def test_config_fingerprint_is_bound_to_actual_runtime_values() -> None:
    config = _FakeDatasetConfig()
    assert dataset_config_fingerprint(config) == dataset_config_fingerprint()

    drifted = dataclasses.replace(config, shuffle_buffer_size=999)
    with pytest.raises(WaymaxDataError, match="dataset_config_drift"):
        dataset_config_fingerprint(drifted)


def test_config_fingerprint_rejects_unaccounted_runtime_fields() -> None:
    @dataclasses.dataclass(frozen=True)
    class ExtendedConfig(_FakeDatasetConfig):
        new_upstream_default: bool = True

    with pytest.raises(WaymaxDataError, match="dataset_config_drift"):
        dataset_config_fingerprint(ExtendedConfig())


@pytest.mark.parametrize("search_limit", [True, 0, -1, 1.5])
def test_waymax_source_rejects_invalid_search_limit(search_limit) -> None:
    with pytest.raises(ValueError, match="search_limit"):
        WaymaxSource(search_limit=search_limit)


def test_iter_records_validates_bounds_before_optional_import(tmp_path) -> None:
    missing = tmp_path / "validation_tfexample.tfrecord-00000-of-00150"
    with pytest.raises(ValueError, match="max_records"):
        next(iter_waymax_records(missing, max_records=0))
    with pytest.raises(WaymaxDataError, match="shard_missing"):
        next(iter_waymax_records(missing, max_records=1))


def test_synthetic_tfexample_preserves_native_scenario_id() -> None:
    tf = pytest.importorskip(
        "tensorflow",
        reason="custom TFExample parser requires the optional waymo dependencies",
    )
    synthetic_id = b"deadbeef01234567"
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "scenario/id": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[synthetic_id])
                )
            }
        )
    )
    parsed = tf.io.parse_example(
        example.SerializeToString(),
        {"scenario/id": _scenario_id_feature_spec(tf)},
    )
    preserved = _preserve_parsed_scenario_id(parsed, tf)

    assert _decode_scenario_id(preserved["scenario/id"]) == synthetic_id.decode()
