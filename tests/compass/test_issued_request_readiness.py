"""Dynamic issued requests use the existing native writer/receiver queue."""
from dataclasses import FrozenInstanceError
import json

import pytest

from atom.compass.runtime import request_readiness as readiness_module
from atom.compass.runtime.native_ingress import NativeIngressReleaseQueue
from atom.compass.runtime.request_readiness import RequestReadiness, UnsupportedReadiness
from atom.sampling_params import SamplingParams
from .test_native_ingress_service import sources, write


@pytest.fixture
def readiness(tmp_path):
    # Test-only constant services make overlap visible: writer2s, receiver3s.
    options = sources(tmp_path, writer=(2_000_000., 0.), receiver=(3_000_000., 0.))
    fit = json.loads((tmp_path / "fit.json").read_text())
    fit["endpoint_summaries"] = [{"serialized_bytes_median": 10},
                                 {"serialized_bytes_median": 100_000}]
    options["fit"] = write(tmp_path / "fit.json", fit)
    validation = json.loads((tmp_path / "validation.json").read_text())
    validation["fit_freeze"] = options["fit"]
    options["validation"] = write(tmp_path / "validation.json", validation)
    profile = write(tmp_path / "profile.json", {
        "schema": "compass.request_readiness_profile/1",
        "resolver": "atom.compass.runtime.native_ingress.NativeWriterReceiver",
        "options": options, "source_law": "test-only fixed source services",
        "support": {"kind": "test fixture"},
        "origin_contract": {"declared_arrival_event": "actual issue",
            "writer_eligibility_event": "actual issue",
            "transition": "test-only immediate writer eligibility"},
    })
    return RequestReadiness(profile["path"])


def sequence(seq_factory, at, *, tokens=4):
    seq = seq_factory([10] * tokens, sampling_params=SamplingParams(max_tokens=1))
    seq.arrive_time = at
    seq.compass_workload_size = None
    seq.compass_workload_index = None
    return seq


def evidence(readiness):
    return readiness.input_manifest()["request_readiness"]


def test_empty_stream_has_no_descriptors_or_service_until_actual_issue(readiness, seq_factory, monkeypatch):
    original = readiness_module.ingress_descriptor
    described = []
    def descriptor(seq):
        described.append(seq.id)
        return original(seq)
    monkeypatch.setattr(readiness_module, "ingress_descriptor", descriptor)
    future = sequence(seq_factory, 30.)
    future.stop_strings = ["unissued metadata must not be examined"]
    readiness.begin_issued_requests()
    assert isinstance(readiness._issued_queue, NativeIngressReleaseQueue)
    assert readiness.records == {} and described == []
    assert evidence(readiness)["issued_releases"] == []
    assert evidence(readiness)["issued_queue"] == {
        "released_requests": 0, "writer_available_at": None, "receiver_available_at": None}
    first = sequence(seq_factory, 10.)
    readiness.admit_issued_request(first, 10.)
    assert described == [first.id]
    assert list(readiness.records) == [first.id]
    assert not future.block_table and not future.state_slots


def test_actual_persistent_queue_overlaps_writer_and_receiver_and_keeps_ties_fifo(readiness, seq_factory):
    readiness.begin_issued_requests()
    seqs = [sequence(seq_factory, at) for at in (10., 11., 11., 30.)]
    records = [readiness.admit_issued_request(seq, seq.arrive_time) for seq in seqs]
    assert [r.source_service_started_at for r in records] == pytest.approx([10., 12., 14., 30.])
    assert [r.ready_at for r in records] == pytest.approx([15., 18., 21., 35.])
    assert [r.receipt_order for r in records] == [0, 1, 2, 3]
    assert records[1].source_service_started_at < records[0].ready_at
    assert all(seq.compass_workload_size is None for seq in seqs)
    assert [readiness.record(seq) for seq in seqs] == records
    assert evidence(readiness)["issued_queue"] == {
        "released_requests": 4, "writer_available_at": 32., "receiver_available_at": 35.}


def test_admitted_record_and_descriptor_remain_immutable_during_sequence_progress(readiness, seq_factory):
    readiness.begin_issued_requests()
    seq = sequence(seq_factory, 10.)
    record = readiness.admit_issued_request(seq, 10.)
    original = evidence(readiness)["issued_releases"][0]
    with pytest.raises(TypeError):
        readiness.records[seq.id] = None
    with pytest.raises(FrozenInstanceError):
        record.ready_at = 0
    seq.num_cached_tokens = 2
    assert readiness.record(seq) is record
    assert evidence(readiness)["issued_releases"][0] == original
    seq.arrive_time += 1
    with pytest.raises(UnsupportedReadiness, match="arrival changed"):
        readiness.record(seq)


def test_issued_descriptor_matches_native_single_request_add(readiness, seq_factory):
    from atom.model_engine.engine_core_mgr import CoreManager
    readiness.begin_issued_requests()
    seq = sequence(seq_factory, 10.)
    sender = CoreManager.__new__(CoreManager)
    sender.label, sender.pp_size, sender.local_engine_count = "CPU sender", 1, 1
    payloads = []
    sender._send_request = lambda rank, payload: payloads.append(payload)
    sender.add_request([seq])
    readiness.admit_issued_request(seq, 10.)
    descriptor = evidence(readiness)["issued_releases"][0]["ingress"]
    assert descriptor["reconstructed_add_bytes"] == len(payloads[0])
    assert descriptor["frame_request_count"] == 1
    assert descriptor["token_typecode"] == "i"


@pytest.mark.parametrize("issued_at", [float("nan"), float("inf"), -float("inf"), None, True, "10", 9.])
def test_bad_or_backwards_issue_time_cannot_charge_service(readiness, seq_factory, issued_at):
    readiness.begin_issued_requests()
    first = sequence(seq_factory, 10.)
    readiness.admit_issued_request(first, 10.)
    before = evidence(readiness)
    candidate = sequence(seq_factory, issued_at)
    with pytest.raises(UnsupportedReadiness):
        readiness.admit_issued_request(candidate, issued_at)
    assert evidence(readiness) == before


def test_arrival_must_be_stamped_by_the_issuer_and_duplicate_id_is_not_recharged(readiness, seq_factory):
    readiness.begin_issued_requests()
    seq = sequence(seq_factory, 10.)
    with pytest.raises(UnsupportedReadiness, match="match"):
        readiness.admit_issued_request(seq, 11.)
    record = readiness.admit_issued_request(seq, 10.)
    before = evidence(readiness)
    with pytest.raises(UnsupportedReadiness, match="duplicated"):
        readiness.admit_issued_request(seq, 10.)
    assert evidence(readiness) == before and readiness.record(seq) is record


@pytest.mark.parametrize("kind", ["expanded", "cache_state", "domain"])
def test_existing_layout_and_source_domain_checks_refuse_without_charging(readiness, seq_factory, kind):
    readiness.begin_issued_requests()
    candidate = sequence(seq_factory, 10., tokens=40_000 if kind == "domain" else 4)
    if kind == "expanded":
        candidate.stop_strings = ["unsupported"]
    elif kind == "cache_state":
        candidate.num_cached_tokens = 1
    before = evidence(readiness)
    with pytest.raises(UnsupportedReadiness):
        readiness.admit_issued_request(candidate, 10.)
    assert evidence(readiness) == before
    good = sequence(seq_factory, 10.)
    assert readiness.admit_issued_request(good, 10.).ready_at == pytest.approx(15.)


@pytest.mark.parametrize("mode", ["register_serial_workload", "register_causal_workload", "resolve_closed_workload"])
@pytest.mark.parametrize("issued_first", [False, True])
def test_registration_modes_cannot_be_mixed(readiness, seq_factory, mode, issued_first):
    seqs = [sequence(seq_factory, 10.) for _ in range(2)]
    for index, seq in enumerate(seqs):
        seq.compass_workload_size = 2
        seq.compass_workload_index = index
    legacy = getattr(readiness, mode)
    if issued_first:
        readiness.begin_issued_requests()
        with pytest.raises(UnsupportedReadiness):
            legacy(seqs)
    else:
        legacy(seqs)
        with pytest.raises(UnsupportedReadiness):
            readiness.begin_issued_requests()
        with pytest.raises(UnsupportedReadiness, match="not initialized"):
            readiness.admit_issued_request(seqs[0], 10.)


def test_stream_initialization_is_explicit_and_cannot_be_repeated(readiness, seq_factory):
    seq = sequence(seq_factory, 10.)
    with pytest.raises(UnsupportedReadiness, match="not initialized"):
        readiness.admit_issued_request(seq, 10.)
    readiness.begin_issued_requests()
    with pytest.raises(UnsupportedReadiness, match="already registered"):
        readiness.begin_issued_requests()
    with pytest.raises(UnsupportedReadiness, match="not registered"):
        readiness.release_causal_request(seq, 10.)
    with pytest.raises(UnsupportedReadiness, match="not registered"):
        readiness.release_serial_request(seq, 10.)


def test_stream_requires_an_existing_persistent_source_service(readiness, monkeypatch):
    monkeypatch.setattr(readiness._provider, "new_release_queue", None)
    with pytest.raises(UnsupportedReadiness, match="persistent"):
        readiness.begin_issued_requests()
    assert readiness.records is None
