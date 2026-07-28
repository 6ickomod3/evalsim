"""Synthetic-only tests for the exact M4 WOMD record stream.

These tests create temporary invented TFRecords. They never inspect ``data/raw`` or
persist source-derived identities, coordinates, trajectories, or artifacts.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

import evalsim.sources.waymax_loader as loader
from evalsim.sources.waymax_cohort import ScanEvent, ShardScanCounts
from evalsim.sources.waymax_loader import (
    M4ReloadExpectation,
    M4ShardLocator,
    M4StreamCounters,
    M4StreamRecord,
    WaymaxDataError,
    WaymaxRecord,
    clear_shard_digest_cache,
    iter_m4_waymax_records,
    m4_shard_sha256,
    reload_m4_waymax_record,
    reload_m4_waymax_records,
    resolve_m4_validation_shards,
)


def _shard_path(directory, index: int):
    return directory / (
        "uncompressed_tf_example_validation_validation_tfexample."
        f"tfrecord-{index:05d}-of-00150"
    )


def _tensorflow():
    return pytest.importorskip(
        "tensorflow",
        reason="synthetic TFRecord streaming requires the optional Waymo stack",
    )


def _write_tfrecord(path, payloads: tuple[bytes, ...]):
    tf = _tensorflow()
    with tf.io.TFRecordWriter(str(path)) as writer:
        for payload in payloads:
            writer.write(payload)
    return tf


class _FakeRuntimeDecoder:
    def __init__(
        self,
        tf,
        *,
        config_fingerprint: str = "c" * 64,
        fail_ordinal: int | None = None,
    ) -> None:
        self.tf = tf
        self.dataset_config_fingerprint = config_fingerprint
        self.fail_ordinal = fail_ordinal
        self.calls: list[int] = []

    def decode(self, serialized, *, locator, shard_sha256):
        self.calls.append(locator.record_ordinal)
        if locator.record_ordinal == self.fail_ordinal:
            raise RuntimeError("synthetic decode failure")
        payload = bytes(serialized)
        audit = MappingProxyType(
            {
                "synthetic_payload": np.frombuffer(
                    payload,
                    dtype=np.uint8,
                ).copy()
            }
        )
        return WaymaxRecord(
            scenario_id=payload.decode("ascii"),
            state=object(),
            audit=audit,
            shard_suffix=locator.shard_suffix,
            record_ordinal=locator.record_ordinal,
            shard_sha256=shard_sha256,
            dataset_config_fingerprint=self.dataset_config_fingerprint,
        )


_REAL_VERIFIED_M4_SCAN_EVENT = loader._verified_m4_scan_event


def _invented_verified_event(item):
    record = item.record
    return ScanEvent.eligible_event(
        shard_suffix=item.locator.shard_suffix,
        record_ordinal=item.locator.record_ordinal,
        native_scenario_id=record.scenario_id,
        shard_sha256=record.shard_sha256,
        dataset_config_fingerprint=record.dataset_config_fingerprint,
    )


def _invented_stream_record(
    *,
    scenario_id: str = "aa",
    suffix: str = "00000",
    ordinal: int = 0,
) -> M4StreamRecord:
    locator = M4ShardLocator(suffix, ordinal)
    record = WaymaxRecord(
        scenario_id=scenario_id,
        state=object(),
        audit=MappingProxyType({"invented": np.asarray([1])}),
        shard_suffix=suffix,
        record_ordinal=ordinal,
        shard_sha256="d" * 64,
        dataset_config_fingerprint="c" * 64,
    )
    return M4StreamRecord(locator=locator, record=record)


def test_m4_runtime_decoder_preserves_waymax_numpy_postprocess_boundary() -> None:
    tf = _tensorflow()
    observed: list[type] = []
    raw_audit = {
        key: np.full(1, index, dtype=np.int64)
        for index, key in enumerate(loader._AUDIT_KEYS)
    }
    narrowed_audit = {
        key: np.full(1, -1, dtype=np.int32)
        for key in loader._AUDIT_KEYS
    }
    state = object()

    def postprocess(example):
        observed.append(type(example["invented"]))
        assert isinstance(example["invented"], np.ndarray)
        assert example["state/all/timestamp_micros"].dtype == np.int64
        return (
            np.frombuffer(b"bb", dtype=np.uint8).reshape(1, -1),
            narrowed_audit,
            state,
        )

    decoder = loader._M4RuntimeDecoder(
        tf=tf,
        preprocess=lambda _: dict(
            raw_audit,
            invented=tf.convert_to_tensor([1.0, 2.0]),
            **{
                "scenario/id": tf.convert_to_tensor(
                    np.frombuffer(b"aa", dtype=np.uint8).reshape(1, -1)
                )
            },
        ),
        postprocess=postprocess,
        dataset_config_fingerprint="c" * 64,
    )
    record = decoder.decode(
        b"invented",
        locator=M4ShardLocator("00000", 0),
        shard_sha256="d" * 64,
    )

    assert observed == [np.ndarray]
    assert record.scenario_id == "aa"
    assert record.state is state
    for index, key in enumerate(loader._AUDIT_KEYS):
        assert record.audit[key].dtype == np.int64
        assert record.audit[key].tolist() == [index]
        assert record.audit[key].flags.writeable is False


def test_m4_runtime_decoder_freezes_audit_before_postprocess_mutation() -> None:
    tf = _tensorflow()
    raw_audit = {
        key: np.full(1, index, dtype=np.int64)
        for index, key in enumerate(loader._AUDIT_KEYS)
    }

    def postprocess(example):
        for key in loader._AUDIT_KEYS:
            example[key][...] = -1
        return (
            np.frombuffer(b"bb", dtype=np.uint8).reshape(1, -1),
            {},
            object(),
        )

    decoder = loader._M4RuntimeDecoder(
        tf=tf,
        preprocess=lambda _: dict(
            raw_audit,
            **{
                "scenario/id": tf.convert_to_tensor(
                    np.frombuffer(b"aa", dtype=np.uint8).reshape(1, -1)
                )
            },
        ),
        postprocess=postprocess,
        dataset_config_fingerprint="c" * 64,
    )

    record = decoder.decode(
        b"invented",
        locator=M4ShardLocator("00000", 0),
        shard_sha256="d" * 64,
    )

    assert record.scenario_id == "aa"
    for index, key in enumerate(loader._AUDIT_KEYS):
        assert record.audit[key].tolist() == [index]


@pytest.fixture(autouse=True)
def _isolated_digest_cache(monkeypatch):
    clear_shard_digest_cache()
    monkeypatch.setattr(
        loader,
        "_verified_m4_scan_event",
        _invented_verified_event,
    )
    yield
    clear_shard_digest_cache()


def test_m4_exact_resolver_calls_only_indices_zero_through_nine(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[int] = []

    def exact_resolver(data_dir, *, shard_index):
        assert data_dir == tmp_path
        calls.append(shard_index)
        return _shard_path(tmp_path, shard_index)

    monkeypatch.setattr(loader, "resolve_validation_shard", exact_resolver)

    resolved = resolve_m4_validation_shards(tmp_path)

    assert calls == list(range(10))
    assert [
        path.name.rsplit("tfrecord-", maxsplit=1)[1].split("-of-", maxsplit=1)[0]
        for path in resolved
    ] == [
        f"{index:05d}" for index in range(10)
    ]


def test_m4_exact_resolver_ignores_present_00010(tmp_path) -> None:
    for index in range(11):
        _shard_path(tmp_path, index).write_bytes(f"shard-{index}".encode())

    resolved = resolve_m4_validation_shards(tmp_path)

    assert len(resolved) == 10
    assert all("00010-of-00150" not in path.name for path in resolved)


def test_m4_stream_rejects_00010_before_digest_or_runtime_open(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 10)
    path.write_bytes(b"must-not-be-opened")

    def forbidden(*args, **kwargs):
        raise AssertionError("out-of-scope shard reached an opening seam")

    monkeypatch.setattr(loader, "_guarded_shard_digest", forbidden)
    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", forbidden)

    with pytest.raises(WaymaxDataError, match="m4_shard_out_of_scope"):
        list(
            iter_m4_waymax_records(
                path,
                counters=M4StreamCounters("00000"),
            )
        )
    with pytest.raises(WaymaxDataError, match="m4_shard_out_of_scope"):
        m4_shard_sha256(path)


def test_m4_stream_rejects_in_scope_alias_to_out_of_scope_symlink(
    tmp_path,
) -> None:
    target = _shard_path(tmp_path, 10)
    target.write_bytes(b"must-not-be-opened")
    alias = _shard_path(tmp_path, 0)
    alias.symlink_to(target.name)

    with pytest.raises(WaymaxDataError, match="shard_symlink_forbidden"):
        list(
            iter_m4_waymax_records(
                alias,
                counters=M4StreamCounters("00000"),
            )
        )
    with pytest.raises(WaymaxDataError, match="shard_symlink_forbidden"):
        m4_shard_sha256(alias)
    with pytest.raises(WaymaxDataError, match="shard_symlink_forbidden"):
        reload_m4_waymax_record(
            tmp_path,
            M4ShardLocator("00000", 0),
            expected_scenario_id="aa",
            expected_shard_sha256="d" * 64,
            expected_dataset_config_fingerprint="c" * 64,
        )
    with pytest.raises(WaymaxDataError, match="shard_symlink_forbidden"):
        reload_m4_waymax_records(
            tmp_path,
            (
                M4ReloadExpectation(
                    locator=M4ShardLocator("00000", 0),
                    expected_scenario_id="aa",
                    expected_shard_sha256="d" * 64,
                    expected_dataset_config_fingerprint="c" * 64,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("suffix", "ordinal"),
    [
        ("00010", 0),
        ("0000", 0),
        ("abcde", 0),
        ("00000", True),
        ("00000", -1),
        ("00000", 1.5),
    ],
)
def test_m4_locator_rejects_out_of_scope_or_noncanonical_values(
    suffix,
    ordinal,
) -> None:
    with pytest.raises(ValueError):
        M4ShardLocator(shard_suffix=suffix, record_ordinal=ordinal)


def test_m4_stream_counts_raw_decode_and_events_at_clean_eof(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 0)
    tf = _write_tfrecord(path, (b"aa", b"bb"))
    decoder = _FakeRuntimeDecoder(tf)
    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", lambda _: decoder)
    counters = M4StreamCounters("00000")

    events = list(
        iter_m4_waymax_records(
            path,
            counters=counters,
        )
    )

    assert [event.locator for event in events] == [
        ("00000", 0),
        ("00000", 1),
    ]
    assert decoder.calls == [0, 1]
    assert counters.raw_seen == 2
    assert counters.decode_attempted == 2
    assert counters.event_emitted == 2
    assert counters.clean_eof is True
    with pytest.raises(AttributeError):
        counters.raw_seen = 99


def test_m4_stream_integrates_with_pure_scan_events_and_counts(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 1)
    tf = _write_tfrecord(path, (b"aa", b"bb"))
    decoder = _FakeRuntimeDecoder(tf)
    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", lambda _: decoder)
    counters = M4StreamCounters("00001")

    def eligible_event(item):
        record = item.record
        return ScanEvent.eligible_event(
            shard_suffix=item.locator.shard_suffix,
            record_ordinal=item.locator.record_ordinal,
            native_scenario_id=record.scenario_id,
            shard_sha256=record.shard_sha256,
            dataset_config_fingerprint=record.dataset_config_fingerprint,
        )

    events = tuple(
        iter_m4_waymax_records(
            path,
            counters=counters,
            event_factory=eligible_event,
        )
    )
    counts = ShardScanCounts(
        shard_suffix=counters.shard_suffix,
        raw_seen=counters.raw_seen,
        decode_attempted=counters.decode_attempted,
        event_emitted=counters.event_emitted,
        eligible=sum(event.outcome == "eligible" for event in events),
        rejected=sum(event.outcome == "rejected" for event in events),
        clean_eof=counters.clean_eof,
    )

    assert [event.record_ordinal for event in events] == [0, 1]
    assert counts.raw_seen == counts.decode_attempted == counts.event_emitted == 2


def test_m4_internal_verifier_owns_source_adapter_and_parity_gates(
    monkeypatch,
) -> None:
    import evalsim.sources.waymax as adapter
    import evalsim.sources.waymax_cohort as cohort

    stream_record = _invented_stream_record()
    scenario = object()
    calls: list[str] = []
    monkeypatch.setattr(
        cohort,
        "source_rejection_code",
        lambda audit: calls.append("source") or None,
    )
    monkeypatch.setattr(
        adapter,
        "scenario_from_waymax_state",
        lambda *args, **kwargs: calls.append("adapter") or scenario,
    )
    monkeypatch.setattr(
        loader,
        "validate_record_parity",
        lambda record, candidate: calls.append("parity")
        if candidate is scenario
        else None,
    )

    event = _REAL_VERIFIED_M4_SCAN_EVENT(stream_record)

    assert event == _invented_verified_event(stream_record)
    assert calls == ["source", "adapter", "parity"]

    monkeypatch.setattr(
        cohort,
        "source_rejection_code",
        lambda audit: "source_sdc_count_not_one",
    )
    monkeypatch.setattr(
        adapter,
        "scenario_from_waymax_state",
        lambda *args, **kwargs: pytest.fail("rejected event reached adapter"),
    )
    rejected = _REAL_VERIFIED_M4_SCAN_EVENT(stream_record)
    assert rejected.outcome == "rejected"
    assert rejected.rejection_code == "source_sdc_count_not_one"


def test_m4_internal_adapter_or_parity_failure_propagates(
    monkeypatch,
) -> None:
    import evalsim.sources.waymax as adapter
    import evalsim.sources.waymax_cohort as cohort

    stream_record = _invented_stream_record()
    monkeypatch.setattr(cohort, "source_rejection_code", lambda audit: None)
    monkeypatch.setattr(
        adapter,
        "scenario_from_waymax_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("adapter failed")
        ),
    )
    with pytest.raises(RuntimeError, match="adapter failed"):
        _REAL_VERIFIED_M4_SCAN_EVENT(stream_record)

    monkeypatch.setattr(
        adapter,
        "scenario_from_waymax_state",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        loader,
        "validate_record_parity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("parity failed")
        ),
    )
    with pytest.raises(RuntimeError, match="parity failed"):
        _REAL_VERIFIED_M4_SCAN_EVENT(stream_record)


def test_m4_stream_rejects_event_factory_contradictions(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 0)
    tf = _write_tfrecord(path, (b"aa",))
    decoder = _FakeRuntimeDecoder(tf)
    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", lambda _: decoder)
    verified = _invented_verified_event(_invented_stream_record())
    contradictions = (
        object(),
        ScanEvent.rejected_event(
            shard_suffix="00000",
            record_ordinal=0,
            native_scenario_id="aa",
            shard_sha256="d" * 64,
            dataset_config_fingerprint="c" * 64,
            rejection_code="source_sdc_count_not_one",
        ),
        ScanEvent.eligible_event(
            shard_suffix="00000",
            record_ordinal=0,
            native_scenario_id="bb",
            shard_sha256="d" * 64,
            dataset_config_fingerprint="c" * 64,
        ),
        ScanEvent.eligible_event(
            shard_suffix="00001",
            record_ordinal=0,
            native_scenario_id="aa",
            shard_sha256="d" * 64,
            dataset_config_fingerprint="c" * 64,
        ),
        ScanEvent.eligible_event(
            shard_suffix="00000",
            record_ordinal=0,
            native_scenario_id="aa",
            shard_sha256="e" * 64,
            dataset_config_fingerprint="c" * 64,
        ),
        ScanEvent.eligible_event(
            shard_suffix="00000",
            record_ordinal=0,
            native_scenario_id="aa",
            shard_sha256="d" * 64,
            dataset_config_fingerprint="e" * 64,
        ),
    )
    assert verified == _invented_verified_event(_invented_stream_record())
    for contradiction in contradictions:
        counters = M4StreamCounters("00000")
        with pytest.raises(WaymaxDataError, match="stream_event_contradiction"):
            list(
                iter_m4_waymax_records(
                    path,
                    counters=counters,
                    event_factory=lambda item, value=contradiction: value,
                )
            )
        assert (
            counters.raw_seen,
            counters.decode_attempted,
            counters.event_emitted,
            counters.clean_eof,
        ) == (1, 1, 0, False)


def test_m4_empty_stream_reaches_clean_eof(tmp_path, monkeypatch) -> None:
    path = _shard_path(tmp_path, 0)
    tf = _write_tfrecord(path, ())
    decoder = _FakeRuntimeDecoder(tf)
    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", lambda _: decoder)
    counters = M4StreamCounters("00000")

    assert list(
        iter_m4_waymax_records(
            path,
            counters=counters,
        )
    ) == []
    assert (
        counters.raw_seen,
        counters.decode_attempted,
        counters.event_emitted,
        counters.clean_eof,
    ) == (0, 0, 0, True)


def test_m4_repeat_stream_reproduces_ordinals_and_counts(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 4)
    tf = _write_tfrecord(path, (b"aa", b"bb", b"cc"))

    def run_once():
        decoder = _FakeRuntimeDecoder(tf)
        monkeypatch.setattr(
            loader,
            "_make_m4_runtime_decoder",
            lambda _: decoder,
        )
        counters = M4StreamCounters("00004")
        events = list(
            iter_m4_waymax_records(
                path,
                counters=counters,
            )
        )
        ordinals = [event.record_ordinal for event in events]
        return ordinals, (
            counters.raw_seen,
            counters.decode_attempted,
            counters.event_emitted,
            counters.clean_eof,
        )

    assert run_once() == run_once() == ([0, 1, 2], (3, 3, 3, True))


def test_m4_decode_failure_is_fatal_and_not_an_event(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 0)
    tf = _write_tfrecord(path, (b"aa", b"bb"))
    decoder = _FakeRuntimeDecoder(tf, fail_ordinal=1)
    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", lambda _: decoder)
    counters = M4StreamCounters("00000")

    with pytest.raises(RuntimeError, match="synthetic decode failure"):
        list(
            iter_m4_waymax_records(
                path,
                counters=counters,
            )
        )

    assert (
        counters.raw_seen,
        counters.decode_attempted,
        counters.event_emitted,
        counters.clean_eof,
    ) == (2, 2, 1, False)


def test_m4_adapter_or_parity_failure_is_fatal_and_not_an_event(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 0)
    tf = _write_tfrecord(path, (b"aa",))
    decoder = _FakeRuntimeDecoder(tf)
    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", lambda _: decoder)
    counters = M4StreamCounters("00000")

    def fail_after_decode(_):
        raise RuntimeError("synthetic adapter parity failure")

    with pytest.raises(RuntimeError, match="adapter parity failure"):
        list(
            iter_m4_waymax_records(
                path,
                counters=counters,
                event_factory=fail_after_decode,
            )
        )

    assert (
        counters.raw_seen,
        counters.decode_attempted,
        counters.event_emitted,
        counters.clean_eof,
    ) == (1, 1, 0, False)


def test_m4_config_failure_is_fatal_before_raw_enumeration(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 0)
    _write_tfrecord(path, (b"aa",))

    def fail_config(_):
        raise RuntimeError("synthetic config drift")

    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", fail_config)
    counters = M4StreamCounters("00000")
    with pytest.raises(RuntimeError, match="config drift"):
        list(
            iter_m4_waymax_records(
                path,
                counters=counters,
            )
        )

    assert (
        counters.raw_seen,
        counters.decode_attempted,
        counters.event_emitted,
        counters.clean_eof,
    ) == (0, 0, 0, False)


def test_m4_corrupt_raw_stream_does_not_report_clean_eof(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 0)
    tf = _write_tfrecord(path, (b"aa",))
    with path.open("ab") as handle:
        handle.write(b"truncated-record")
    decoder = _FakeRuntimeDecoder(tf)
    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", lambda _: decoder)
    counters = M4StreamCounters("00000")

    with pytest.raises(tf.errors.DataLossError):
        list(
            iter_m4_waymax_records(
                path,
                counters=counters,
            )
        )

    assert (
        counters.raw_seen,
        counters.decode_attempted,
        counters.event_emitted,
        counters.clean_eof,
    ) == (1, 1, 1, False)


def test_m4_digest_cache_reuses_only_unchanged_file_identity(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 0)
    path.write_bytes(b"first")
    original = loader.file_sha256
    calls: list[object] = []

    def counted(candidate, *, chunk_bytes=8 * 1024 * 1024):
        calls.append(candidate)
        return original(candidate, chunk_bytes=chunk_bytes)

    monkeypatch.setattr(loader, "file_sha256", counted)
    first = m4_shard_sha256(path)
    assert m4_shard_sha256(path) == first
    assert len(calls) == 1

    path.write_bytes(b"second-and-different-size")
    second = m4_shard_sha256(path)
    assert second != first
    assert len(calls) == 2


@pytest.mark.parametrize(
    "changed_identity",
    [
        loader._FileIdentity(device=2, inode=2, size=3, mtime_ns=4),
        loader._FileIdentity(device=1, inode=9, size=3, mtime_ns=4),
        loader._FileIdentity(device=1, inode=2, size=8, mtime_ns=4),
        loader._FileIdentity(device=1, inode=2, size=3, mtime_ns=7),
    ],
)
def test_m4_digest_cache_key_checks_every_identity_field(
    tmp_path,
    monkeypatch,
    changed_identity,
) -> None:
    path = _shard_path(tmp_path, 0)
    path.write_bytes(b"payload")
    original = loader.file_sha256
    calls = 0
    baseline = loader._FileIdentity(device=1, inode=2, size=3, mtime_ns=4)
    identities = iter((baseline, baseline, changed_identity, changed_identity))

    def fake_identity(_):
        return next(identities)

    def counted(candidate, *, chunk_bytes=8 * 1024 * 1024):
        nonlocal calls
        calls += 1
        return original(candidate, chunk_bytes=chunk_bytes)

    monkeypatch.setattr(loader, "_file_identity", fake_identity)
    monkeypatch.setattr(loader, "file_sha256", counted)

    assert m4_shard_sha256(path) == m4_shard_sha256(path)
    assert calls == 2


def test_m4_digest_rejects_identity_change_during_hash(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 0)
    path.write_bytes(b"before")
    original = loader.file_sha256

    def mutate_during_hash(candidate, *, chunk_bytes=8 * 1024 * 1024):
        digest = original(candidate, chunk_bytes=chunk_bytes)
        candidate.write_bytes(b"after-with-a-different-size")
        return digest

    monkeypatch.setattr(loader, "file_sha256", mutate_during_hash)
    with pytest.raises(WaymaxDataError, match="shard_changed"):
        m4_shard_sha256(path)


def test_m4_locator_reload_verifies_identity_and_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 3)
    tf = _write_tfrecord(path, (b"aa", b"bb", b"cc"))
    decoder = _FakeRuntimeDecoder(tf)
    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", lambda _: decoder)
    digest = m4_shard_sha256(path)
    locator = M4ShardLocator("00003", 1)

    reloaded = reload_m4_waymax_record(
        tmp_path,
        locator,
        expected_scenario_id="bb",
        expected_shard_sha256=digest,
        expected_dataset_config_fingerprint=decoder.dataset_config_fingerprint,
    )

    assert reloaded.locator == locator
    assert reloaded.record.scenario_id == "bb"
    assert decoder.calls == [1]

    with pytest.raises(WaymaxDataError, match="locator_identity_mismatch"):
        reload_m4_waymax_record(
            tmp_path,
            locator,
            expected_scenario_id="cc",
            expected_shard_sha256=digest,
            expected_dataset_config_fingerprint=decoder.dataset_config_fingerprint,
        )
    with pytest.raises(WaymaxDataError, match="locator_provenance_mismatch"):
        reload_m4_waymax_record(
            tmp_path,
            locator,
            expected_scenario_id="bb",
            expected_shard_sha256="0" * 64,
            expected_dataset_config_fingerprint=decoder.dataset_config_fingerprint,
        )


def test_m4_grouped_reload_builds_one_decoder_and_streams_once_per_shard(
    tmp_path,
    monkeypatch,
) -> None:
    paths = {
        suffix: _shard_path(tmp_path, int(suffix))
        for suffix in ("00000", "00001")
    }
    tf = _write_tfrecord(paths["00000"], (b"aa", b"ab", b"ac"))
    _write_tfrecord(paths["00001"], (b"ba", b"bb", b"bc"))
    decoders = {
        suffix: _FakeRuntimeDecoder(tf)
        for suffix in paths
    }
    builds: list[str] = []

    def make_decoder(path):
        suffix = loader._m4_shard_suffix_from_path(path)
        builds.append(suffix)
        return decoders[suffix]

    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", make_decoder)
    digests = {
        suffix: m4_shard_sha256(path)
        for suffix, path in paths.items()
    }
    expectations = (
        M4ReloadExpectation(
            locator=M4ShardLocator("00001", 2),
            expected_scenario_id="bc",
            expected_shard_sha256=digests["00001"],
            expected_dataset_config_fingerprint="c" * 64,
        ),
        M4ReloadExpectation(
            locator=M4ShardLocator("00000", 1),
            expected_scenario_id="ab",
            expected_shard_sha256=digests["00000"],
            expected_dataset_config_fingerprint="c" * 64,
        ),
        M4ReloadExpectation(
            locator=M4ShardLocator("00001", 0),
            expected_scenario_id="ba",
            expected_shard_sha256=digests["00001"],
            expected_dataset_config_fingerprint="c" * 64,
        ),
    )

    records = reload_m4_waymax_records(tmp_path, expectations)

    assert builds == ["00000", "00001"]
    assert [item.locator for item in records] == [
        expectation.locator for expectation in expectations
    ]
    assert [item.record.scenario_id for item in records] == ["bc", "ab", "ba"]
    assert decoders["00000"].calls == [1]
    assert decoders["00001"].calls == [0, 2]


def test_m4_grouped_reload_rejects_duplicate_locator_before_data_access(
    tmp_path,
) -> None:
    expectation = M4ReloadExpectation(
        locator=M4ShardLocator("00000", 0),
        expected_scenario_id="aa",
        expected_shard_sha256="d" * 64,
        expected_dataset_config_fingerprint="c" * 64,
    )

    with pytest.raises(WaymaxDataError, match="locator_duplicate"):
        reload_m4_waymax_records(tmp_path, (expectation, expectation))


def test_m4_grouped_reload_empty_request_does_not_access_data(tmp_path) -> None:
    assert reload_m4_waymax_records(tmp_path, ()) == ()


def test_m4_reload_expectation_rejects_noncanonical_provenance() -> None:
    locator = M4ShardLocator("00000", 0)
    with pytest.raises(ValueError, match="expected_scenario_id"):
        M4ReloadExpectation(locator, "not-hex", "d" * 64, "c" * 64)
    with pytest.raises(ValueError, match="expected_shard_sha256"):
        M4ReloadExpectation(locator, "aa", "D" * 64, "c" * 64)
    with pytest.raises(ValueError, match="expected_dataset_config_fingerprint"):
        M4ReloadExpectation(locator, "aa", "d" * 64, "short")


def test_m4_locator_reload_fails_when_ordinal_is_beyond_clean_eof(
    tmp_path,
    monkeypatch,
) -> None:
    path = _shard_path(tmp_path, 2)
    tf = _write_tfrecord(path, (b"aa",))
    decoder = _FakeRuntimeDecoder(tf)
    monkeypatch.setattr(loader, "_make_m4_runtime_decoder", lambda _: decoder)
    digest = m4_shard_sha256(path)

    with pytest.raises(WaymaxDataError, match="locator_ordinal_missing"):
        reload_m4_waymax_record(
            tmp_path,
            M4ShardLocator("00002", 4),
            expected_scenario_id="aa",
            expected_shard_sha256=digest,
            expected_dataset_config_fingerprint=decoder.dataset_config_fingerprint,
        )
    assert decoder.calls == []
