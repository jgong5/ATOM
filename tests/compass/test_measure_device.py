# SPDX-License-Identifier: MIT
"""`compass measure device` through `cli.main`, its probes and the cards' sysfs
stood in for: one record per node, read back from the store; a second on the
same node and platform, a busy card and cards on two PTL states refuse before
any probe runs."""

import json
import os
import signal
import subprocess
from types import SimpleNamespace

import pytest
from test_artifact_invalidation import STANZA

from atom.compass import cli
from atom.compass.artifacts import ArtifactStore, Key, Kind, Topology, member_name
from atom.compass.drive import device_probe, measure_device, pp_probe, serve
from atom.compass.spec import SpecRefusal
from atom.compass.spec.rules import Rule

STACK = {"rocm": "7.2.4", "aiter": "09d256420", "rccl": "2.27.7", "torch": "2.10.0"}
PLATFORM = {
    "device": "MI355X gfx950",
    "software_stack": "rocm 7.2.4 / aiter 09d256420 / rccl 2.27.7",
    "ptl": "disabled",
    "torch": "2.10.0",
}
BUS = {g: f"0000:{16 * (g + 1):02x}:00.0" for g in range(8)}
CARRIERS = {"rccl": {"latency_s": 1e-5}, "gloo": {"latency_s": 3e-5}}
PID = 2**22 + 7  # a probe's, above any pid_max, so no live process is in its group


def samples(card: int, formats) -> dict:
    """A probe's samples on visible cards 0 and 1 of process `card`: card 2 is
    half as fast as the others, card 0 copies twice as fast as card 1, and
    every GEMM shape but one runs slower."""
    slow = 2 if card == 2 else 1
    copy = {0: 0.5}.get(card, slow)
    gemm = {
        f: {"8192x8192x8192": [slow * 1e-3, slow * 1.2e-3], "8192x8192x16384": [9.0]}
        for f in formats
    }
    return {
        "gemm": gemm,
        "aiter": {},
        "triton": {},
        "failed": {},
        "read": {"4096": [slow * 8e-6], "8192": [slow * 8e-6]},
        "write": {"4096": [slow * 8e-6], "8192": [slow * 8e-6]},
        "copy": {"4096": [copy * 4e-6], "8192": [copy * 2e-6]},
        "link": [slow * 6e10, slow * 6.2e10],
        "floor": [slow * 1.4e-6, slow * 1.5e-6],
        "profiler_floor": [4e-6, 4.2e-6],
        "one_hop": [slow * 8e-7, slow * 9e-7],
    }


@pytest.fixture
def node(monkeypatch, tmp_path):
    """A node of eight MI355X cards, each idle unless named in `busy`; the
    probes the command starts run here, each recording the HIP devices it
    saw and the formats it was asked for; no MXFP4 GEMM runs on `no_fp4`. KFD
    lists each probe's process, on every card, for two polls after it exits."""
    seen = {"pairs": [], "busy": {}, "no_fp4": set(), "exited": [], "killed": []}
    monkeypatch.setattr(serve, "POLL_S", 0.001)
    monkeypatch.setattr(
        serve, "_kfd", lambda: set(seen["exited"] and seen["exited"].pop())
    )
    monkeypatch.setattr(
        measure_device,
        "_device",
        lambda gpus: [{"index": g, "name": "MI355X", "arch": "gfx950"} for g in gpus],
    )
    monkeypatch.setattr(measure_device, "_ptl", lambda gpus: "disabled")
    monkeypatch.setattr(measure_device, "_stack", lambda: STACK)
    monkeypatch.setattr(measure_device, "roots_for", lambda m: STANZA.source_roots)
    monkeypatch.setattr(measure_device.socket, "gethostname", lambda: "mi355-gpu-40")
    showbus = {f"card{g}": {"PCI Bus": bus.upper()} for g, bus in BUS.items()}
    monkeypatch.setattr(measure_device, "smi", lambda *a: json.dumps(showbus))
    monkeypatch.setattr(measure_device, "hip_visible", lambda g: ",".join(map(str, g)))
    monkeypatch.setattr(
        device_probe,
        "cards",
        lambda bdfs: [
            {
                "bdf": b,
                "vram_used": seen["busy"].get(b, 2**20),
                "processes": [p for listed in seen["exited"] for p in listed],
                "ptl": "disabled",
            }
            for b in bdfs
        ],
    )

    def probe(torch, formats):
        card = int(seen["pairs"][-1].split(",")[0])
        if "fp4" in formats and card in seen["no_fp4"]:
            raise SpecRefusal(Rule.MEASURED, "no MXFP4 GEMM runs on these cards", "")
        return samples(card, formats)

    monkeypatch.setattr(device_probe, "probe", probe)
    monkeypatch.setattr(
        measure_device,
        "host_link",
        lambda torch: {"host_latency": [5e-6, 6e-6], "host_link": [5e10, 5.2e10]},
    )
    monkeypatch.setattr(pp_probe, "probe", lambda: [{"carrier": "rccl"}])
    monkeypatch.setattr(pp_probe, "fit", lambda s: CARRIERS)

    class Probe:
        pid = PID

        def __init__(self, argv, env, stdout, stderr, start_new_session):
            seen["pairs"].append(env["HIP_VISIBLE_DEVICES"])
            seen.setdefault("argv", []).append(argv)
            measure_device.child(*argv[-2:])
            seen["exited"] = [[str(self.pid)]] * 2

        def poll(self):
            """0, or, with `interrupt` set, None after a SIGINT to this process."""
            if seen.pop("interrupt", False):
                os.kill(os.getpid(), signal.SIGINT)
                return None
            return 0

        wait = poll

    def gone(group, sig):
        seen["killed"].append((group, sig))
        raise ProcessLookupError

    monkeypatch.setattr(serve.os, "killpg", gone)
    started = SimpleNamespace(Popen=Probe, STDOUT=subprocess.STDOUT)
    monkeypatch.setattr(measure_device, "subprocess", started)
    return seen


def measure(tmp_path, gpus="0,1,2", *more):
    out = tmp_path / f"out{len(list(tmp_path.glob('out*')))}"
    argv = ["--gpu", gpus, "--store", str(tmp_path / "store"), "--out", str(out)]
    cli.main(["measure", "device", *argv, *more])


def published(tmp_path) -> dict:
    key = Key.of(Kind.PROBE_RECORD, host="mi355-gpu-40", **PLATFORM)
    entry = ArtifactStore(tmp_path / "store").read(key)
    rank = Topology().ranks()[0]
    assert set(entry.members) == {member_name("record", rank, "json")}
    return json.loads(entry.read_member("record", rank, "json"))


def test_one_record_per_node_holds_every_cards_samples_and_their_bands(node, tmp_path):
    node["no_fp4"] = {0, 1, 2}
    measure(tmp_path, "0,1,2", "--pp")
    record = published(tmp_path)
    # Each card's one-card probes and its link to the next, the last to the
    # first; the carriers on the first two cards, under torchrun.
    assert node["pairs"] == ["0,1", "1,2", "2,0", "0,1"]
    assert node["argv"][-1][1:3] == ["-m", "torch.distributed.run"]
    assert record["host"] == "mi355-gpu-40" and record["platform"] == PLATFORM
    assert set(record["cards"]) == {"0", "1", "2"}
    # Every card idle before each probe, the carriers' included, and after the last.
    assert [[c["bdf"] for c in check] for check in record["checks"]] == [
        [BUS[0], BUS[1], BUS[2]]
    ] * 5
    read = record["readings"]
    # [median, p10, p90] of the best shape's rates over every card's samples:
    # 2 * 8192**3 flops in 1, 1.2, 1, 1.2, 2 and 2.4 ms.
    flops = 2 * 8192**3
    assert read["device.compute.attainable_flops.bf16"] == pytest.approx(
        [flops / 1.2e-3, (flops / 2.4e-3 + flops / 2e-3) / 2, flops / 1e-3]
    )
    assert record["absent"] == {"fp4": "no MXFP4 GEMM runs on these cards"}
    assert "device.compute.attainable_flops.fp4" not in read
    # The curve is card 1's, the median card's at each size, not card 0's.
    assert read["device.memory.bandwidth_curve"] == [
        [4096, pytest.approx(4096 / 4e-6)],
        [8192, pytest.approx(8192 / 2e-6)],
    ]
    # Samples 0.8, 0.9, 0.8, 0.9, 1.6, 1.8 us.
    assert read["interconnect.intra_node.link_latency_s"] == pytest.approx(
        [9e-7, 8e-7, 1.7e-6]
    )
    assert read["interconnect.host_link.bandwidth_bytes_per_s"][0] == 5.1e10
    assert read["interconnect.intra_node.pp"] == CARRIERS
    assert (tmp_path / "out0" / "record.json").read_text() == json.dumps(record)


def test_a_second_record_on_the_same_node_and_platform_refuses_before_any_probe(
    node, tmp_path
):
    measure(tmp_path)
    first = published(tmp_path)
    assert "device.compute.attainable_flops.fp4" in first["readings"]
    probed = len(node["pairs"])
    with pytest.raises(SystemExit) as refused:
        measure(tmp_path, "3,4")
    assert "holds this node's record" in str(refused.value)
    assert len(node["pairs"]) == probed
    assert published(tmp_path) == first


def test_a_busy_card_refuses_naming_it_and_nothing_is_published(node, tmp_path):
    node["busy"][BUS[1]] = 3 * 2**30
    with pytest.raises(SystemExit) as refused:
        measure(tmp_path)
    assert f"{BUS[1]} (ptl disabled, {3 * 2**30} B used" in str(refused.value)
    assert BUS[0] not in str(refused.value)
    assert node["pairs"] == []
    assert not (tmp_path / "store" / "probe_record").exists()


def test_cards_on_two_ptl_states_refuse_naming_each(node, monkeypatch, tmp_path):
    from atom.compass.drive import identity

    for gpu, state in ((0, "disabled"), (1, "enabled")):
        (tmp_path / "pci" / BUS[gpu] / "ptl").mkdir(parents=True)
        (tmp_path / "pci" / BUS[gpu] / "ptl" / "ptl_enable").write_text(state)
    monkeypatch.setattr(identity, "PCI", str(tmp_path / "pci"))
    monkeypatch.setattr(identity, "smi", measure_device.smi)
    monkeypatch.setattr(measure_device, "_ptl", identity._ptl)
    with pytest.raises(SystemExit) as refused:
        measure(tmp_path, "0,1")
    assert "disagree on PTL: GPU 0 disabled, GPU 1 enabled" in str(refused.value)
    assert node["pairs"] == []


def test_a_sigint_during_a_probe_stops_its_session_and_nothing_is_published(
    node, tmp_path
):
    node["interrupt"] = True
    with pytest.raises((SystemExit, KeyboardInterrupt)) as refused:
        measure(tmp_path)
    assert node["killed"] == [(PID, signal.SIGTERM)]
    assert "SIGINT stopped the command; nothing is published" in str(refused.value)
    assert node["pairs"] == ["0,1"]
    assert not (tmp_path / "store" / "probe_record").exists()


def test_a_probes_memory_still_clearing_once_it_exits_is_waited_for(
    node, monkeypatch, tmp_path
):
    # KFD lists no probe process while card 1 still reads 1.37 GB used for
    # three reads after each probe, as crs005's did under a second after one.
    stubbed, reads = device_probe.cards, {}

    def cards(bdfs):
        read, probes = stubbed(bdfs), len(node["pairs"])
        reads[probes] = reads.get(probes, 0) + 1
        if probes and reads[probes] <= 3:
            read[1] = read[1] | {"vram_used": 1371521024}
        return read

    monkeypatch.setattr(device_probe, "cards", cards)
    measure(tmp_path, "0,1,2")
    record = published(tmp_path)
    assert node["pairs"] == ["0,1", "1,2", "2,0"]
    assert [c["vram_used"] for check in record["checks"] for c in check] == [2**20] * 12


def test_a_card_not_idle_by_settle_s_after_a_probe_refuses_naming_it(
    node, monkeypatch, tmp_path
):
    monkeypatch.setattr(measure_device, "SETTLE_S", 0.05)
    stubbed = device_probe.cards

    def cards(bdfs):
        read = stubbed(bdfs)
        if node["pairs"]:
            read[1] = read[1] | {"vram_used": 1371521024}
        return read

    monkeypatch.setattr(device_probe, "cards", cards)
    with pytest.raises(SystemExit) as refused:
        measure(tmp_path, "0,1,2")
    assert f"{BUS[1]} (ptl disabled, 1371521024 B used" in str(refused.value)
    assert node["pairs"] == ["0,1"]
    assert not (tmp_path / "store" / "probe_record").exists()
