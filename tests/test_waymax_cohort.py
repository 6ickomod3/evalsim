"""Pure M4 cohort tests; all identities and arrays are invented."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import numpy as np
import pytest

from evalsim.sources.waymax_cohort import (
    CohortInvariantError,
    CohortSelectionError,
    M4_COHORT_DOMAIN,
    M4_COHORT_TARGET,
    M4_IDM_SUBSET_DOMAIN,
    M4_INITIAL_QUOTAS,
    M4_REDISTRIBUTION_DOMAIN,
    M4_SELECTOR_CONFIG_FINGERPRINT,
    M4_SHARD_SUFFIXES,
    M4_TFRECORD_SUFFIXES,
    M4_VMAP_DOMAIN,
    SOURCE_REJECTION_CODES,
    ScanEvent,
    ShardScanCounts,
    WaymaxCohortManifest,
    rank_record,
    ranking_message,
    select_cohort,
    select_idm_subset,
    select_vmap_pair,
    selector_config_fingerprint,
    selector_config_payload,
    source_rejection_code,
)

_SHARD_DIGEST = "a" * 64
_CONFIG_FINGERPRINT = "b" * 64


def _canonical_text(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _eligible_event(suffix: str, ordinal: int, *, identity: str | None = None):
    return ScanEvent.eligible_event(
        shard_suffix=suffix,
        record_ordinal=ordinal,
        native_scenario_id=identity or f"invented-{suffix}-{ordinal}",
        shard_sha256=_SHARD_DIGEST,
        dataset_config_fingerprint=_CONFIG_FINGERPRINT,
    )


def _rejected_event(
    suffix: str,
    ordinal: int,
    *,
    identity: str | None = None,
    code: str = SOURCE_REJECTION_CODES[-1],
):
    return ScanEvent.rejected_event(
        shard_suffix=suffix,
        record_ordinal=ordinal,
        native_scenario_id=identity or f"rejected-{suffix}-{ordinal}",
        shard_sha256=_SHARD_DIGEST,
        dataset_config_fingerprint=_CONFIG_FINGERPRINT,
        rejection_code=code,
    )


def _events_with_counts(
    eligible_counts: dict[str, int],
    *,
    rejected_per_shard: int = 0,
) -> tuple[list[ScanEvent], list[ShardScanCounts]]:
    events: list[ScanEvent] = []
    counts: list[ShardScanCounts] = []
    for suffix in M4_SHARD_SUFFIXES:
        eligible = eligible_counts.get(suffix, 0)
        for ordinal in range(eligible):
            events.append(_eligible_event(suffix, ordinal))
        for offset in range(rejected_per_shard):
            events.append(
                _rejected_event(
                    suffix,
                    eligible + offset,
                    code=SOURCE_REJECTION_CODES[
                        (int(suffix) + offset) % len(SOURCE_REJECTION_CODES)
                    ],
                )
            )
        total = eligible + rejected_per_shard
        counts.append(
            ShardScanCounts(
                shard_suffix=suffix,
                raw_seen=total,
                decode_attempted=total,
                event_emitted=total,
                eligible=eligible,
                rejected=rejected_per_shard,
                clean_eof=True,
            )
        )
    return events, counts


def _audit() -> dict[str, np.ndarray]:
    num_objects = 128
    horizon = 91
    valid = np.zeros((num_objects, horizon), dtype=bool)
    valid[0] = True
    valid[1] = True
    timestamps = np.broadcast_to(
        np.arange(horizon, dtype=np.int64) * 100_000,
        (num_objects, horizon),
    ).copy()
    zeros = np.zeros((num_objects, horizon), dtype=np.float32)
    object_ids = np.full(num_objects, -1, dtype=np.int64)
    object_ids[:2] = (101, 202)
    object_types = np.zeros(num_objects, dtype=np.int32)
    object_types[:2] = 1
    is_sdc = np.zeros(num_objects, dtype=bool)
    is_sdc[0] = True
    roadgraph_xyz = np.zeros((30000, 3), dtype=np.float32)
    roadgraph_xyz[:3] = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    roadgraph_direction = np.zeros((30000, 3), dtype=np.float32)
    roadgraph_direction[:2, 0] = 1.0
    roadgraph_types = np.zeros((30000, 1), dtype=np.int32)
    roadgraph_ids = np.full((30000, 1), -1, dtype=np.int64)
    roadgraph_ids[:3] = 77
    roadgraph_valid = np.zeros((30000, 1), dtype=bool)
    roadgraph_valid[:3] = True
    return {
        "state/id": object_ids,
        "state/type": object_types,
        "state/is_sdc": is_sdc,
        "state/which_time": np.concatenate(
            (
                -np.ones(10, dtype=np.int8),
                np.zeros(1, dtype=np.int8),
                np.ones(80, dtype=np.int8),
            )
        ),
        "state/all/valid": valid,
        "state/all/x": zeros.copy(),
        "state/all/y": zeros.copy(),
        "state/all/velocity_x": zeros.copy(),
        "state/all/velocity_y": zeros.copy(),
        "state/all/bbox_yaw": zeros.copy(),
        "state/all/timestamp_micros": timestamps,
        "state/all/length": np.full(
            (num_objects, horizon),
            4.0,
            dtype=np.float32,
        ),
        "state/all/width": np.full(
            (num_objects, horizon),
            2.0,
            dtype=np.float32,
        ),
        "roadgraph_samples/xyz": roadgraph_xyz,
        "roadgraph_samples/dir": roadgraph_direction,
        "roadgraph_samples/type": roadgraph_types,
        "roadgraph_samples/id": roadgraph_ids,
        "roadgraph_samples/valid": roadgraph_valid,
    }


def _copy_audit(audit: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.array(value, copy=True) for name, value in audit.items()}


def _drop_last_object_slot(audit: dict[str, np.ndarray]) -> None:
    for name in ("state/id", "state/type", "state/is_sdc"):
        audit[name] = audit[name][:-1]
    for name in (
        "state/all/valid",
        "state/all/x",
        "state/all/y",
        "state/all/velocity_x",
        "state/all/velocity_y",
        "state/all/bbox_yaw",
        "state/all/timestamp_micros",
        "state/all/length",
        "state/all/width",
    ):
        audit[name] = audit[name][:-1, :]


def _drop_last_roadgraph_point(audit: dict[str, np.ndarray]) -> None:
    for name in (
        "roadgraph_samples/xyz",
        "roadgraph_samples/dir",
        "roadgraph_samples/type",
        "roadgraph_samples/id",
        "roadgraph_samples/valid",
    ):
        audit[name] = audit[name][:-1, :]


def test_exact_ten_suffixes_and_quotas_are_frozen() -> None:
    assert M4_SHARD_SUFFIXES == tuple(f"{index:05d}" for index in range(10))
    assert M4_TFRECORD_SUFFIXES == tuple(
        f"tfrecord-{index:05d}-of-00150" for index in range(10)
    )
    assert tuple(M4_INITIAL_QUOTAS.values()) == (13,) * 8 + (12,) * 2
    assert sum(M4_INITIAL_QUOTAS.values()) == M4_COHORT_TARGET
    with pytest.raises(TypeError):
        M4_INITIAL_QUOTAS["00000"] = 99


def test_source_predicate_accepts_invented_supported_record_without_mutation() -> None:
    audit = _audit()
    before = _copy_audit(audit)

    assert source_rejection_code(audit) is None
    for name in audit:
        np.testing.assert_array_equal(audit[name], before[name])


def test_source_rejection_priority_is_exact() -> None:
    audit = _audit()
    audit["state/is_sdc"][1] = True
    audit["state/all/valid"][0, 90] = False
    audit["state/type"][1] = 2
    audit["roadgraph_samples/type"][:] = 99
    assert source_rejection_code(audit) == "source_sdc_count_not_one"

    audit = _audit()
    audit["state/all/valid"][0, 90] = False
    audit["state/type"][1] = 2
    audit["roadgraph_samples/type"][:] = 99
    assert source_rejection_code(audit) == "source_sdc_future_incomplete"

    audit = _audit()
    audit["state/type"][1] = 2
    audit["roadgraph_samples/type"][:] = 99
    assert source_rejection_code(audit) == "source_no_world_vehicle_transition"

    audit = _audit()
    audit["roadgraph_samples/type"][:] = 99
    assert source_rejection_code(audit) == "source_no_supported_map"
    assert SOURCE_REJECTION_CODES == (
        "source_sdc_count_not_one",
        "source_sdc_future_incomplete",
        "source_no_world_vehicle_transition",
        "source_no_supported_map",
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda audit: audit["roadgraph_samples/type"].__setitem__(
                (1, 0), 14
            ),
            "source_no_supported_map",
        ),
        (
            lambda audit: audit["roadgraph_samples/xyz"].__setitem__(
                (1, slice(None)), audit["roadgraph_samples/xyz"][0]
            ),
            "source_no_supported_map",
        ),
        (
            lambda audit: audit["roadgraph_samples/xyz"].__setitem__(
                (1, 0), 0.75 + 1e-5
            ),
            "source_no_supported_map",
        ),
        (
            lambda audit: audit["roadgraph_samples/xyz"].__setitem__(
                (1, 0), 1e-6
            ),
            "source_no_supported_map",
        ),
        (
            lambda audit: audit["roadgraph_samples/dir"].__setitem__(
                (0, slice(None)), 0.0
            ),
            "source_no_supported_map",
        ),
        (
            lambda audit: audit["roadgraph_samples/dir"].__setitem__(
                (0, 1), math.tan(math.radians(11.0))
            ),
            "source_no_supported_map",
        ),
        (
            lambda audit: audit["roadgraph_samples/dir"].__setitem__(
                (2, slice(None)), 0.0
            ),
            None,
        ),
    ],
)
def test_source_map_rule_matches_m3_boundaries(mutate, expected) -> None:
    audit = _audit()
    mutate(audit)
    assert source_rejection_code(audit) == expected


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda audit: audit.pop("state/all/x"),
            "audit_field_missing",
        ),
        (
            lambda audit: audit.__setitem__(
                "state/all/valid",
                audit["state/all/valid"][:, :-1],
            ),
            "audit_shape_or_dtype_drift",
        ),
        (
            lambda audit: audit["state/which_time"].__setitem__(10, 1),
            "which_time_drift",
        ),
        (
            lambda audit: audit["state/id"].__setitem__(1, 101),
            "duplicate_native_object_id",
        ),
        (
            lambda audit: audit["state/all/x"].__setitem__((0, 10), np.nan),
            "nonfinite_valid_motion",
        ),
        (
            lambda audit: audit["state/all/length"].__setitem__((0, 5), 4.1),
            "dimension_not_constant",
        ),
        (
            lambda audit: audit["state/all/timestamp_micros"].__setitem__(
                (1, 10), 999
            ),
            "timestamp_disagreement",
        ),
        (
            lambda audit: audit["state/all/timestamp_micros"].__setitem__(
                (slice(None), 11),
                audit["state/all/timestamp_micros"][:, 10],
            ),
            "timestamp_not_increasing",
        ),
    ],
)
def test_source_contract_drift_is_fatal_not_a_rejection(mutate, code) -> None:
    audit = _audit()
    mutate(audit)
    with pytest.raises(CohortInvariantError) as error:
        source_rejection_code(audit)
    assert error.value.code == code


@pytest.mark.parametrize(
    "mutate",
    [_drop_last_object_slot, _drop_last_roadgraph_point],
)
def test_source_config_shapes_require_exact_128_and_30000(mutate) -> None:
    audit = _audit()
    mutate(audit)
    with pytest.raises(
        CohortInvariantError,
        match="audit_shape_or_dtype_drift",
    ):
        source_rejection_code(audit)


def test_ranking_message_uses_zero_delimiters_and_uint64_big_endian() -> None:
    message = ranking_message(
        M4_COHORT_DOMAIN,
        "00003",
        256,
        "invented-α",
    )
    assert message == (
        M4_COHORT_DOMAIN.encode("ascii")
        + b"\x00"
        + b"00003"
        + b"\x00"
        + b"\x00\x00\x00\x00\x00\x00\x01\x00"
        + b"\x00"
        + "invented-α".encode("utf-8")
    )
    assert rank_record(
        M4_COHORT_DOMAIN,
        "00003",
        256,
        "invented-α",
    ) == hashlib.sha256(message).hexdigest()


@pytest.mark.parametrize(
    ("domain", "suffix", "ordinal", "identity"),
    [
        ("é", "00000", 0, "id"),
        ("domain\x00suffix", "00000", 0, "id"),
        ("domain", "00010", 0, "id"),
        ("domain", "00000", -1, "id"),
        ("domain", "00000", 2**64, "id"),
        ("domain", "00000", 0, ""),
    ],
)
def test_ranking_rejects_ambiguous_or_out_of_scope_inputs(
    domain,
    suffix,
    ordinal,
    identity,
) -> None:
    with pytest.raises(CohortInvariantError):
        ranking_message(domain, suffix, ordinal, identity)


def test_quota_selection_is_exact_and_canonically_ordered() -> None:
    counts = dict(M4_INITIAL_QUOTAS)
    events, _ = _events_with_counts(counts)
    selection = select_cohort(list(reversed(events)))

    assert selection.actual_count == 128
    assert selection.redistributed_count == 0
    assert selection.fallback_used is False
    assert all(deficit == 0 for _, deficit in selection.quota_deficits)
    assert tuple(event.shard_suffix for event in selection.events) == tuple(
        suffix
        for suffix in M4_SHARD_SUFFIXES
        for _ in range(M4_INITIAL_QUOTAS[suffix])
    )
    assert all(event.selected for event in selection.events)
    for suffix in M4_SHARD_SUFFIXES:
        shard_selected = [
            event
            for event in selection.selected_events
            if event.shard_suffix == suffix
        ]
        assert shard_selected == sorted(
            shard_selected,
            key=lambda event: (event.selection_rank, event.record_ordinal),
        )


def test_quota_deficit_uses_global_redistribution_rank() -> None:
    counts = {suffix: 20 for suffix in M4_SHARD_SUFFIXES}
    counts["00000"] = 5
    events, _ = _events_with_counts(counts)
    selection = select_cohort(events)

    assert selection.actual_count == 128
    assert dict(selection.quota_deficits)["00000"] == 8
    assert selection.redistributed_count == 8
    assert sum(
        event.selected and event.shard_suffix == "00000"
        for event in selection.events
    ) == 5

    initial = {
        event.locator
        for suffix in M4_SHARD_SUFFIXES
        for event in sorted(
            (
                item
                for item in events
                if item.shard_suffix == suffix and item.outcome == "eligible"
            ),
            key=lambda item: (item.selection_rank, item.record_ordinal),
        )[: M4_INITIAL_QUOTAS[suffix]]
    }
    expected_redistributed = sorted(
        (event for event in events if event.locator not in initial),
        key=lambda event: (
            rank_record(
                M4_REDISTRIBUTION_DOMAIN,
                event.shard_suffix,
                event.record_ordinal,
                event.native_scenario_id,
            ),
            event.shard_suffix,
            event.record_ordinal,
        ),
    )[:8]
    actual_redistributed = {
        event.locator
        for event in selection.selected_events
        if event.locator not in initial
    }
    assert actual_redistributed == {
        event.locator for event in expected_redistributed
    }


def test_under_target_fallback_selects_all_with_all_ten_represented() -> None:
    events, _ = _events_with_counts(
        {suffix: 4 for suffix in M4_SHARD_SUFFIXES}
    )
    selection = select_cohort(events)

    assert selection.actual_count == 40
    assert selection.fallback_used is True
    assert selection.redistributed_count == 0
    assert all(event.selected for event in selection.events)
    assert {event.shard_suffix for event in selection.selected_events} == set(
        M4_SHARD_SUFFIXES
    )


@pytest.mark.parametrize(
    "counts",
    [
        {
            suffix: 4 if index == 0 else 3
            for index, suffix in enumerate(M4_SHARD_SUFFIXES)
        },
        {
            suffix: 0 if index == 0 else 4
            for index, suffix in enumerate(M4_SHARD_SUFFIXES)
        },
    ],
)
def test_under_target_fallback_enforces_floor_and_all_ten(counts) -> None:
    events, _ = _events_with_counts(counts)
    with pytest.raises(CohortSelectionError, match="cohort_fallback_floor"):
        select_cohort(events)


def test_duplicate_locator_or_identity_is_fatal() -> None:
    events, _ = _events_with_counts(dict(M4_INITIAL_QUOTAS))
    with pytest.raises(CohortInvariantError, match="duplicate_locator"):
        select_cohort([*events, events[0]])

    duplicate_identity = _eligible_event(
        "00009",
        99,
        identity=events[0].native_scenario_id,
    )
    with pytest.raises(
        CohortInvariantError,
        match="duplicate_native_scenario_id",
    ):
        select_cohort([*events, duplicate_identity])


def test_nested_idm_subset_uses_independent_domain_and_floor() -> None:
    events, _ = _events_with_counts(dict(M4_INITIAL_QUOTAS))
    selected = select_cohort(events).selected_events
    qualification = {
        event.locator: index < 20 for index, event in enumerate(selected)
    }
    subset = select_idm_subset(selected, qualification)
    expected = sorted(
        selected[:20],
        key=lambda event: (
            rank_record(
                M4_IDM_SUBSET_DOMAIN,
                event.shard_suffix,
                event.record_ordinal,
                event.native_scenario_id,
            ),
            event.shard_suffix,
            event.record_ordinal,
        ),
    )[:16]
    assert subset == tuple(expected)

    ten_qualify = {
        event.locator: index < 10 for index, event in enumerate(selected)
    }
    assert len(select_idm_subset(selected, ten_qualify)) == 10
    seven_qualify = {
        event.locator: index < 7 for index, event in enumerate(selected)
    }
    with pytest.raises(CohortSelectionError, match="idm_subset_floor"):
        select_idm_subset(selected, seven_qualify)


def test_nested_idm_subset_requires_complete_boolean_classification() -> None:
    events, _ = _events_with_counts(dict(M4_INITIAL_QUOTAS))
    selected = select_cohort(events).selected_events
    incomplete = {event.locator: True for event in selected[:-1]}
    with pytest.raises(CohortInvariantError, match="classify every"):
        select_idm_subset(selected, incomplete)
    invalid = {event.locator: True for event in selected}
    invalid[selected[0].locator] = 1
    with pytest.raises(CohortInvariantError, match="booleans"):
        select_idm_subset(selected, invalid)


def test_vmap_pair_uses_its_own_domain() -> None:
    events, _ = _events_with_counts(dict(M4_INITIAL_QUOTAS))
    selected = select_cohort(events).selected_events
    pair = select_vmap_pair(selected)
    expected = sorted(
        selected,
        key=lambda event: (
            rank_record(
                M4_VMAP_DOMAIN,
                event.shard_suffix,
                event.record_ordinal,
                event.native_scenario_id,
            ),
            event.shard_suffix,
            event.record_ordinal,
        ),
    )[:2]
    assert pair == tuple(expected)


def test_scan_event_and_counters_fail_closed() -> None:
    event = _eligible_event("00000", 0)
    with pytest.raises(CohortInvariantError, match="selection_rank"):
        dataclasses.replace(event, selection_rank="0" * 64)
    with pytest.raises(CohortInvariantError, match="registered source reason"):
        _rejected_event("00000", 0, code="unregistered")
    with pytest.raises(CohortInvariantError, match="scan_counter_mismatch"):
        ShardScanCounts(
            shard_suffix="00000",
            raw_seen=2,
            decode_attempted=1,
            event_emitted=1,
            eligible=1,
            rejected=0,
            clean_eof=True,
        )
    with pytest.raises(CohortInvariantError, match="scan_not_clean_eof"):
        ShardScanCounts(
            shard_suffix="00000",
            raw_seen=1,
            decode_attempted=1,
            event_emitted=1,
            eligible=1,
            rejected=0,
            clean_eof=False,
        )


def test_manifest_scalars_are_normalized_to_json_native_ints() -> None:
    event = ScanEvent.eligible_event(
        shard_suffix="00000",
        record_ordinal=np.int64(7),
        native_scenario_id="invented-numpy-integer",
        shard_sha256=_SHARD_DIGEST,
        dataset_config_fingerprint=_CONFIG_FINGERPRINT,
    )
    counts = ShardScanCounts(
        shard_suffix="00000",
        raw_seen=np.int64(1),
        decode_attempted=np.int64(1),
        event_emitted=np.int64(1),
        eligible=np.int64(1),
        rejected=np.int64(0),
        clean_eof=True,
    )
    assert type(event.record_ordinal) is int
    assert type(counts.raw_seen) is int
    json.dumps(event.to_dict(), allow_nan=False)
    json.dumps(counts.to_dict(), allow_nan=False)


def test_manifest_is_byte_stable_immutable_and_exclusive_create(tmp_path) -> None:
    eligible_counts = dict(M4_INITIAL_QUOTAS)
    events, counts = _events_with_counts(
        eligible_counts,
        rejected_per_shard=1,
    )
    manifest = WaymaxCohortManifest.build(
        events=events,
        shard_counts=counts,
    )

    restored = WaymaxCohortManifest.from_json(manifest.to_json())
    assert restored == manifest
    assert restored.canonical_bytes() == manifest.canonical_bytes()
    assert restored.sha256 == manifest.sha256
    assert manifest.to_json() == manifest.to_json()

    destination = tmp_path / "cohort" / "manifest.json"
    manifest.to_file(destination)
    assert destination.read_bytes() == manifest.canonical_bytes()
    assert WaymaxCohortManifest.from_file(destination) == manifest
    with pytest.raises(FileExistsError):
        manifest.to_file(destination)
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.schema_version = "changed"


def test_manifest_rejects_semantically_equal_noncanonical_json(tmp_path) -> None:
    events, counts = _events_with_counts(dict(M4_INITIAL_QUOTAS))
    manifest = WaymaxCohortManifest.build(events=events, shard_counts=counts)
    pretty = json.dumps(
        manifest.to_dict(),
        indent=2,
        sort_keys=False,
        ensure_ascii=False,
    )
    assert json.loads(pretty) == json.loads(manifest.to_json())
    with pytest.raises(CohortInvariantError, match="json_noncanonical"):
        WaymaxCohortManifest.from_json(pretty)

    missing_newline = tmp_path / "missing-newline.json"
    missing_newline.write_bytes(manifest.to_json().encode("utf-8"))
    with pytest.raises(CohortInvariantError, match="json_noncanonical"):
        WaymaxCohortManifest.from_file(missing_newline)

    extra_newline = tmp_path / "extra-newline.json"
    extra_newline.write_bytes(manifest.canonical_bytes() + b"\n")
    with pytest.raises(CohortInvariantError, match="json_noncanonical"):
        WaymaxCohortManifest.from_file(extra_newline)


def test_manifest_rejects_counter_event_and_selection_tampering() -> None:
    events, counts = _events_with_counts(
        dict(M4_INITIAL_QUOTAS),
        rejected_per_shard=1,
    )
    manifest = WaymaxCohortManifest.build(events=events, shard_counts=counts)

    payload = manifest.to_dict()
    payload["events"][0]["selected"] = not payload["events"][0]["selected"]
    with pytest.raises(CohortInvariantError, match="selection_tampered"):
        WaymaxCohortManifest.from_json(_canonical_text(payload))

    payload = manifest.to_dict()
    payload["shard_counts"][0]["raw_seen"] += 1
    with pytest.raises(CohortInvariantError):
        WaymaxCohortManifest.from_json(_canonical_text(payload))

    payload = manifest.to_dict()
    payload["selection"]["actual_count"] -= 1
    with pytest.raises(CohortInvariantError, match="selection_tampered"):
        WaymaxCohortManifest.from_json(_canonical_text(payload))


def test_manifest_rejects_noncanonical_schema_and_duplicate_json_keys() -> None:
    events, counts = _events_with_counts(dict(M4_INITIAL_QUOTAS))
    manifest = WaymaxCohortManifest.build(events=events, shard_counts=counts)
    payload = manifest.to_dict()
    payload["unexpected"] = True
    with pytest.raises(CohortInvariantError, match="field set"):
        WaymaxCohortManifest.from_json(_canonical_text(payload))

    duplicate = (
        '{"events":[],"events":[],"schema_version":"1",'
        '"selection":{},"selector_config_fingerprint":"'
        + M4_SELECTOR_CONFIG_FINGERPRINT
        + '","selector_version":"1","shard_counts":[]}'
    )
    with pytest.raises(CohortInvariantError, match="duplicate JSON object key"):
        WaymaxCohortManifest.from_json(duplicate)


def test_manifest_requires_exact_ordinal_sequence_and_unique_identity() -> None:
    events, counts = _events_with_counts(dict(M4_INITIAL_QUOTAS))
    broken_events = list(events)
    broken_events[0] = _eligible_event("00000", 99)
    broken_selection = select_cohort(broken_events)
    with pytest.raises(CohortInvariantError, match="ordinal"):
        WaymaxCohortManifest(
            selection=broken_selection,
            shard_counts=tuple(counts),
        )

    duplicate = [
        *events,
        _rejected_event(
            "00009",
            12,
            identity=events[0].native_scenario_id,
        ),
    ]
    with pytest.raises(CohortInvariantError, match="duplicate_native"):
        select_cohort(duplicate)


def test_selector_payload_fingerprint_freezes_every_rule_and_is_defensive() -> None:
    payload = selector_config_payload()
    assert selector_config_fingerprint() == M4_SELECTOR_CONFIG_FINGERPRINT
    assert payload["source_predicate"]["rejection_priority"] == list(
        SOURCE_REJECTION_CODES
    )
    assert payload["ranking"]["domains"] == {
        "cohort": M4_COHORT_DOMAIN,
        "redistribution": M4_REDISTRIBUTION_DOMAIN,
        "idm_subset": M4_IDM_SUBSET_DOMAIN,
        "vmap_pair": M4_VMAP_DOMAIN,
    }
    assert payload["cohort"]["initial_quotas"] == dict(M4_INITIAL_QUOTAS)
    assert payload["cohort"]["fallback"] == {
        "condition": "total_eligible_below_target",
        "minimum_total": 32,
        "require_every_shard_represented": True,
        "selection": "complete_eligible_population",
    }
    assert payload["idm_subset"]["target"] == 16
    assert payload["idm_subset"]["fallback_floor"] == 8
    assert payload["idm_subset"]["actor_control_mask"] == {
        "not_sdc": True,
        "object_type_id": 1,
        "logged_valid_frames": ["current", "next"],
        "initialized_overlap_excluded": True,
    }
    assert payload["idm_subset"]["lifecycle_fallback"] == (
        "log_for_birth_disappearance_or_invalid_transition"
    )
    assert payload["idm_subset"]["upstream_idm_defaults"] == {
        "desired_velocity_mps": 30.0,
        "minimum_spacing_m": 2.0,
        "safe_headway_s": 2.0,
        "maximum_acceleration_mps2": 2.0,
        "maximum_deceleration_mps2": 4.0,
        "exponent": 4,
        "maximum_lookahead": 10,
        "lookahead_from_current_position": True,
        "additional_lookahead_points": 10,
        "additional_lookahead_distance_m": 10.0,
        "invalidate_on_end": False,
    }
    assert payload["vmap"]["pair_size"] == 2
    assert payload["parity"]["float_rtol"] == 0.0
    assert payload["parity"]["float_atol"] == 1e-6
    assert payload["manifest"]["clean_eof_required"] is True

    payload["cohort"]["target"] = 1
    payload["ranking"]["domains"]["cohort"] = "tampered"
    payload["idm_subset"]["upstream_idm_defaults"][
        "desired_velocity_mps"
    ] = 1.0
    assert selector_config_fingerprint() == M4_SELECTOR_CONFIG_FINGERPRINT
