# SPDX-License-Identifier: MIT
"""`compass fit device` through `cli.main`, on probe records `compass measure
device` published with its probes stood in for: two nodes' records fold into
one `platform_readings` entry, rates kept as rates; records of two stacks and
a second fit to one key refuse; `fold` takes each field's median over the
records stating it; `--assume-from` states a measured entry's ratios for a
device never run, takes the entry at the PTL state named, and refuses several
entries to take from."""

import json

import pytest
from test_artifact_invalidation import STANZA
from test_measure_device import (  # noqa: F401  the node fixture
    CARRIERS,
    PLATFORM,
    STACK,
    measure,
    node,
)

from atom.compass import cli
from atom.compass.artifacts import ArtifactStore, Key, Kind
from atom.compass.drive import device_probe, measure_device
from atom.compass.fit import device
from atom.compass.spec.layers import datasheet

BF16 = "device.compute.attainable_flops.bf16"
FP4 = "device.compute.attainable_flops.fp4"
LINK = "interconnect.intra_node.attainable_bandwidth_bytes_per_s"
FLOOR = "device.kernel_floor_s"
CURVE = "device.memory.bandwidth_curve"
RCCL = "interconnect.intra_node.pp.rccl.latency_s"
EAGER = "eager_threshold_bytes"


@pytest.fixture
def nodes(node, monkeypatch, tmp_path):  # noqa: F811
    """Each node's record by host: gpu-40's over three cards as fast as each
    other, with the carriers; crs005's over three where card 2 is half as fast
    and no MXFP4 GEMM runs. Each is published under `stack`, gpu-40's by default."""
    monkeypatch.setattr(device, "roots_for", lambda m: STANZA.source_roots)

    def probe(host, gpus, *more, stack=STACK):
        monkeypatch.setattr(measure_device.socket, "gethostname", lambda: host)
        monkeypatch.setattr(measure_device, "_stack", lambda: stack)
        node["no_fp4"] = {0, 1, 2} if host == "crs005" else set()
        measure(tmp_path, gpus, *more)
        platform = PLATFORM | {"software_stack": _pins(stack)}
        key = Key.of(Kind.PROBE_RECORD, host=host, **platform)
        entry = ArtifactStore(tmp_path / "store").read(key)
        record = json.loads(entry.read_member("record", device.RANK, "json"))
        return key.token(), record["readings"]

    return probe


def _pins(stack: dict) -> str:
    return " / ".join(f"{c} {stack[c]}" for c in ("rocm", "aiter", "rccl"))


def fit(tmp_path, *argv) -> dict:
    out = tmp_path / f"fit{len(list(tmp_path.glob('fit*')))}"
    store = ["--store", str(tmp_path / "store"), "--out", str(out)]
    cli.main(["fit", "device", *argv, *store])
    return json.loads((out / "platform_readings.json").read_text())


def test_two_nodes_records_fold_into_one_entry_each_field_banded_over_both(
    nodes, tmp_path
):
    gpu40, at40 = nodes("mi355-gpu-40", "0,1,3", "--pp")
    crs005, at005 = nodes("crs005", "0,1,2")
    entry = fit(tmp_path, gpu40, crs005)
    key = Key.of(Kind.PLATFORM_READINGS, **PLATFORM)
    assert entry["key"] == key.values and entry["token"] == key.token()
    stored = ArtifactStore(tmp_path / "store").read(key)
    member = stored.read_member("readings", device.RANK, "json")
    assert json.loads(member) == {
        k: v for k, v in entry.items() if k not in ("key", "token")
    }
    origins = entry["provenance"]["origins"]
    for path in (BF16, LINK, FLOOR):
        # The median over the nodes, the band from the lowest p10 to the highest p90.
        assert entry[path] == pytest.approx((at40[path][0] + at005[path][0]) / 2)
        assert origins[path] == {
            "origin": "probed",
            "record": f"{gpu40}, {crs005}",
            "p10": min(at40[path][1], at005[path][1]),
            "p90": max(at40[path][2], at005[path][2]),
        }
    # crs005's slow card moves the band, not the value, past gpu-40's.
    assert origins[BF16]["p10"] == at005[BF16][1] < at40[BF16][1]
    # A field only one record states is that record's, and names it alone.
    assert (entry[FP4], origins[FP4]["record"]) == (at40[FP4][0], gpu40)
    assert (entry[RCCL], origins[RCCL]["p10"]) == (CARRIERS["rccl"]["latency_s"],) * 2
    assert entry[CURVE] == [
        [size, pytest.approx((a[1] + b[1]) / 2)]
        for size, a, b in zip((4096, 8192), at40[CURVE], at005[CURVE])
    ]
    # Rates stay rates: no peak and no derate, so no datasheet is named.
    assert not [p for p in entry if "derate" in p or p.endswith("_flops")]
    assert entry["provenance"]["origin"] == "probed"


def test_records_of_two_stacks_refuse_naming_each_and_publish_nothing(nodes, tmp_path):
    gpu40, _ = nodes("mi355-gpu-40", "0,1")
    crs005, _ = nodes("crs005", "0,1", stack=STACK | {"aiter": "41d4d8d7e"})
    with pytest.raises(SystemExit) as refused:
        fit(tmp_path, gpu40, crs005)
    assert (
        f"the records are of 2 software_stack values: {gpu40} 'rocm 7.2.4 / aiter "
        f"09d256420 / rccl 2.27.7'; {crs005} 'rocm 7.2.4 / aiter 41d4d8d7e / rccl "
        "2.27.7'"
    ) in str(refused.value)
    assert not (tmp_path / "store" / "platform_readings").exists()


def test_a_second_fit_to_a_published_key_refuses_and_the_entry_stands(nodes, tmp_path):
    gpu40, _ = nodes("mi355-gpu-40", "0,1")
    first = fit(tmp_path, gpu40)
    crs005, _ = nodes("crs005", "0,1")
    with pytest.raises(SystemExit) as refused:
        fit(tmp_path, gpu40, crs005)
    assert "was handed off at" in str(refused.value)
    key = Key.of(Kind.PLATFORM_READINGS, **PLATFORM)
    member = (
        ArtifactStore(tmp_path / "store")
        .read(key)
        .read_member("readings", device.RANK, "json")
    )
    assert json.loads(member)[BF16] == first[BF16]


def test_assume_from_states_the_sources_ratios_on_its_datasheet_every_field_assumed(
    nodes, tmp_path
):
    gpu40, _ = nodes("mi355-gpu-40", "0,1", "--pp")
    source = fit(tmp_path, gpu40)
    b0 = fit(
        tmp_path,
        "--device",
        "mi455x-b0",
        "--assume-from",
        "MI355X",
        "--ptl",
        "disabled",
    )
    sheet = datasheet("mi355x").values
    assert b0["key"] == PLATFORM | {"device": "MI455X B0 gfx1250"}
    assert (
        b0["device.compute.derate.bf16"]
        == source[BF16] / sheet["device.compute.bf16_flops"]
    )
    assert (
        b0["interconnect.intra_node.derate"]
        == source[LINK] / sheet["interconnect.intra_node.link_bandwidth_bytes_per_s"]
    )
    assert b0["device.memory.bandwidth_derate_curve"] == [
        [s, r / sheet["device.memory.bandwidth_bytes_per_s"]] for s, r in source[CURVE]
    ]
    assert (b0[FLOOR], b0[RCCL]) == (source[FLOOR], source[RCCL])
    assert not [p for p in b0 if "attainable" in p or p == CURVE]
    assert b0["provenance"]["origin"] == "assumed"
    assert {o["origin"] for o in b0["provenance"]["origins"].values()} == {"assumed"}
    assert set(b0["provenance"]["origins"]) == {
        p for p in b0 if p not in ("key", "token", *device.HEADER)
    }
    # An assumed entry is no source: its ratios are not rates to divide.
    a0 = ("--device", "mi455x-a0", "--assume-from", "MI455X B0", "--ptl", "disabled")
    with pytest.raises(SystemExit) as refused:
        fit(tmp_path, *a0)
    assert "is assumed itself" in str(refused.value)


def test_a_rate_whose_peak_the_source_datasheet_omits_is_left_out():
    readings = {
        "device.compute.attainable_flops.bf16": 1.0e15,
        "device.compute.attainable_flops.fp4": 3.0e15,
        FLOOR: 2e-6,
    }
    sheet = {"device.compute.bf16_flops": 2.0e15}
    assert device.assumed(readings, sheet) == {
        "device.compute.derate.bf16": 0.5,
        FLOOR: 2e-6,
    }


def test_assume_from_with_several_source_entries_refuses_listing_them(nodes, tmp_path):
    other = STACK | {"aiter": "41d4d8d7e"}
    keys = []
    for host, stack in (("mi355-gpu-40", STACK), ("crs005", other)):
        token, _ = nodes(host, "0,1", stack=stack)
        keys.append(fit(tmp_path, token)["token"])
    with pytest.raises(SystemExit) as refused:
        fit(
            tmp_path,
            "--device",
            "mi455x-b0",
            "--assume-from",
            "MI355X",
            "--ptl",
            "disabled",
        )
    held = str(refused.value)
    assert "holds 2 platform_readings entries for MI355X gfx950 at PTL disabled" in held
    assert all(f"as {k}" in held for k in keys)
    assert not list((tmp_path / "store" / "platform_readings").glob("MI455X*"))


def test_fold_takes_each_fields_median_over_the_records_stating_it():
    # Three records whose values and curves differ; two hold the carriers, so a
    # byte count's median falls between their values and stays one of them.
    def record(bf16, curve, *carrier):
        pp = {device.PP: {"rccl": dict(zip(("latency_s", EAGER), carrier))}}
        return {"readings": {BF16: bf16, CURVE: curve} | (pp if carrier else {})}

    values, origins = device.fold(
        {
            "a": record(
                [1e15, 0.9e15, 1.1e15], [[4096, 1e12], [8192, 1e12]], 1e-5, 65536
            ),
            "b": record(
                [2e15, 1.5e15, 2e15], [[4096, 3e12], [8192, 3e12]], 3e-5, 131073
            ),
            "c": record([9e15, 8e15, 9.5e15], [[4096, 2e12], [8192, 5e12]]),
        }
    )
    assert values[BF16] == 2e15
    assert origins[BF16] == {
        "origin": "probed",
        "record": "a, b, c",
        "p10": 0.9e15,
        "p90": 9.5e15,
    }
    assert values[CURVE] == [[4096, 2e12], [8192, 3e12]]
    eager = f"{device.PP}.rccl.{EAGER}"
    assert (values[eager], origins[eager]["record"]) == (65536, "a, b")
    assert values[RCCL] == pytest.approx(2e-5)


def test_assume_from_takes_the_source_at_the_ptl_state_named(
    nodes, monkeypatch, tmp_path
):
    gpu40, _ = nodes("mi355-gpu-40", "0,1")
    source = fit(tmp_path, gpu40)
    # crs005's entry for the same device and stack, its cards at PTL enabled.
    monkeypatch.setattr(measure_device, "_ptl", lambda gpus: "enabled")
    stubbed = device_probe.cards
    monkeypatch.setattr(
        device_probe,
        "cards",
        lambda bdfs: [c | {"ptl": "enabled"} for c in stubbed(bdfs)],
    )
    monkeypatch.setattr(measure_device.socket, "gethostname", lambda: "crs005")
    measure(tmp_path, "0,1")
    enabled = Key.of(Kind.PROBE_RECORD, host="crs005", **PLATFORM | {"ptl": "enabled"})
    assert fit(tmp_path, enabled.token())["key"]["ptl"] == "enabled"
    b0 = ("--device", "mi455x-b0", "--assume-from", "MI355X", "--ptl", "disabled")
    entry = fit(tmp_path, *b0)
    assert entry["key"]["ptl"] == "disabled"
    peak = datasheet("mi355x").values["device.compute.bf16_flops"]
    assert entry["device.compute.derate.bf16"] == source[BF16] / peak
