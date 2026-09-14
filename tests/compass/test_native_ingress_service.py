"""Frozen service inputs drive separate writer and receiver availability."""

import hashlib
import json

import pytest

from atom.compass.runtime.native_ingress import NativeWriterReceiver
from atom.compass.runtime.request_readiness import (
    IngressDescriptor, RegisteredRequest, UnsupportedReadiness,
)


def write(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sources(tmp_path, *, writer=(2., 0.), receiver=(3., 0.), passed=True):
    refs = {key: write(tmp_path / f"{key}.json", {"fixture": key})
            for key in ["plan", "contract", "source_result"]}
    fit = write(tmp_path / "fit.json", {
        "schema": "compass.frozen_endpoint_service_fit/1", **refs,
        "endpoint_summaries": [{"serialized_bytes_median": 10},
                               {"serialized_bytes_median": 1000}],
        "models": {name: {"intercept_us": values[0], "slope_us_per_byte": values[1]}
                   for name, values in [("writer", writer), ("receiver", receiver)]},
    })
    validation = write(tmp_path / "validation.json", {
        "schema": "compass.withheld_service_support_verdict/1", "fit_freeze": fit,
        "heldout_support_passed": passed, "profile_admissible_under_fixed_contract": passed,
    })
    return {"fit": fit, "validation": validation}


def request(index, arrival=0., *, size=100, protocol=4):
    return RegisteredRequest(index, arrival, 4, index,
        IngressDescriptor(size, protocol, 1, "i", 4, 16, True))


def test_writer_and_receiver_overlap_instead_of_serializing_elapsed_handoff(tmp_path):
    service = NativeWriterReceiver(**sources(tmp_path))
    events = service.resolve_closed_workload([request(0), request(1), request(2, 20e-6)])
    assert events[0].ready_at == pytest.approx(5e-6)
    assert events[1].ready_at == pytest.approx(8e-6)  # not two complete 5us handoffs
    assert events[2].ready_at == pytest.approx(25e-6)
    assert [event.receipt_order for event in events.values()] == [0, 1, 2]


def test_byte_costs_use_frozen_models_and_have_no_client_or_request_cap(tmp_path):
    service = NativeWriterReceiver(**sources(tmp_path, writer=(1, .01), receiver=(2, .02)))
    events = service.resolve_closed_workload([request(i) for i in range(32)])
    assert len(events) == 32
    assert events[0].ready_at == pytest.approx(6e-6)
    assert events[31].ready_at == pytest.approx(130e-6)


@pytest.mark.parametrize("size,protocol", [(9, 4), (1001, 4), (100, 5)])
def test_unsupported_payload_or_format_refuses(tmp_path, size, protocol):
    service = NativeWriterReceiver(**sources(tmp_path))
    with pytest.raises(UnsupportedReadiness):
        service.resolve_closed_workload([request(0, size=size, protocol=protocol)])


def test_failed_heldout_does_not_admit_the_fit(tmp_path):
    with pytest.raises(UnsupportedReadiness, match="heldout"):
        NativeWriterReceiver(**sources(tmp_path, passed=False))


def test_actual_source_reads_are_pinned_and_not_reopened_for_service(tmp_path):
    refs = sources(tmp_path)
    service = NativeWriterReceiver(**refs)
    (tmp_path / "fit.json").write_text("changed")
    assert service.resolve_closed_workload([request(0)])[0].ready_at == pytest.approx(5e-6)
    assert {item.role for item in service.loaded_inputs} == {
        "runtime.request_readiness.endpoint_fit", "runtime.request_readiness.validation",
        "runtime.request_readiness.plan", "runtime.request_readiness.contract",
        "runtime.request_readiness.source",
    }
    write(tmp_path / "fit.json", {"changed": True})
    with pytest.raises(UnsupportedReadiness, match="source identity changed"):
        NativeWriterReceiver(**refs)
