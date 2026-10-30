"""The attainable-peak probe: the cards it refuses, and the fragment its
samples make, merged into a machine spec by that machine's script."""

import importlib.util
import json
from pathlib import Path

import pytest

from atom.compass.spec import Fragment, MachineSpec, SpecRefusal
from atom.compass.spec.rules import Rule

TREE = Path(__file__).resolve().parents[2]
SCRIPTS = TREE / "scripts" / "compass"


def _load(name):
    loaded = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    return module


probe = _load("probe_device")

#: Samples as a probe writes them: GEMM seconds per shape, read, write and copy
#: seconds per size moved, copy the fastest and its rate dipping on the
#: plateau, and the banded fields' samples.
RECORD = {
    "gemm": {
        "bf16": {"4096x4096x4096": [1e-3] * 3, "8192x4096x8192": [2e-3] * 3},
        "fp8": {"8192x8192x8192": [2e-3]},
    },
    "read": {"4096": [8e-6], "8192": [8e-6], "16384": [16.4e-6]},
    "write": {"4096": [8e-6], "8192": [8e-6], "16384": [16.4e-6]},
    "copy": {"4096": [4e-6], "8192": [4e-6], "16384": [8.2e-6]},
    "link": [3.2e10, 3.0e10, 3.4e10],
    "floor": [2e-6, 1e-6, 3e-6, 4e-6, 5e-6],
    "one_hop": [1e-6] * 5,
}


def test_readings_take_the_best_shape_and_keep_the_curve_monotone():
    found = probe.readings(RECORD)
    assert found["bf16"] == 2 * 8192 * 4096 * 8192 / 2e-3  # the second shape is best
    assert [r for _, r in found["bandwidth_curve"]] == [1.024e9, 2.048e9, 2.048e9]
    assert found["link"] == 3.2e10
    assert found["device.kernel_floor_s"][0] == 3e-6


def test_fp4_takes_the_faster_library_at_each_shape():
    gemm = RECORD["gemm"] | {
        "fp4": {"8192x8192x8192": [2e-3], "16384x8192x8192": [3e-3]}
    }
    aiter = {"fp4": {"8192x8192x8192": [1e-3], "16384x8192x8192": [8e-3]}}
    found = probe.readings(RECORD | {"gemm": gemm, "aiter": aiter})
    assert found["fp4"] == 2 * 8192**3 / 1e-3  # aiter's, at the first shape
    assert found["bf16"] == probe.readings(RECORD)["bf16"]


def test_the_curve_is_at_least_every_kind_at_every_size():
    record = RECORD | {
        "read": {"4096": [2e-6], "8192": [8e-6], "16384": [8e-6]},
        "write": {"4096": [8e-6], "8192": [2e-6], "16384": [5e-6]},
    }
    curve = probe.readings(record)["bandwidth_curve"]
    # read is the fastest at 4 KiB, write at 8 KiB; 16 KiB keeps 8 KiB's rate
    assert curve == [
        [4096, pytest.approx(2.048e9)],
        [8192, pytest.approx(4.096e9)],
        [16384, pytest.approx(4.096e9)],
    ]
    for kind in probe.rates(record).values():
        assert all(rate >= kind[size] for size, rate in curve)


def test_the_node18_script_publishes_the_probed_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPASS_MACHINE", "mi308x-hjbog18")
    script = _load("mi308x_spec")
    found = probe.readings(RECORD)
    monkeypatch.setattr(script, "PEAKS", ("2026-10-19", "abc", found, "f00d"))
    document = SCRIPTS / "machines" / "mi308x-hjbog18.json"
    out = tmp_path / "spec.json"
    assert script.main(["--sweep-from", str(document), "--out", str(out)]) == 0
    spec = MachineSpec.from_mapping(json.loads(out.read_text()))
    assert spec.value("device.compute.derate.fp8") == 2 * 8192**3 / 2e-3 / 6.88e14
    assert spec.value("device.memory.derate") == 2.048e9 / 5.3e12
    assert spec.value("interconnect.intra_node.derate") == 3.2e10 / 6.4e10
    floor = spec.origin("device.kernel_floor_s")
    assert floor["origin"] == "probed" and floor["record"] == "sha256:f00d"
    assert (floor["p10"], floor["p90"]) == pytest.approx((1.4e-6, 4.6e-6))


def test_a_rate_above_its_peak_is_refused():
    peaks = {
        "device.compute.bf16_flops": 1.0,
        "device.compute.fp8_flops": 1e30,
        "device.memory.bandwidth_bytes_per_s": 1e30,
        "interconnect.intra_node.link_bandwidth_bytes_per_s": 1e30,
    }
    found = probe.readings(RECORD)
    document = probe.fragment("m", found, peaks, "a", "2026-10-19", {"record": "r"})
    with pytest.raises(SpecRefusal) as refused:
        Fragment.from_mapping(document, "probe")
    assert refused.value.rule is Rule.DERATE
    assert "device.compute.derate.bf16" in str(refused.value)


def _card(drm, name, bdf, used, ptl=None, charged=()):
    """A card at `bdf`, its KFD node and the processes `charged` its memory:
    `(pid, bytes)` each. Every card reads 100% busy, as a gfx1250 card with no
    process does."""
    device = drm / "devices" / bdf
    device.mkdir(parents=True)
    (device / "mem_info_vram_used").write_text(f"{used}\n")
    (device / "gpu_busy_percent").write_text("100\n")
    if ptl:
        (device / "ptl").mkdir()
        (device / "ptl" / "ptl_enable").write_text(f"{ptl}\n")
    (drm / name).mkdir()
    (drm / name / "device").symlink_to(device)
    domain, bus, slot = bdf.split(":")
    dev, fn = slot.split(".")
    node = drm / "kfd" / "topology" / "nodes" / name
    node.mkdir(parents=True)
    location = int(bus, 16) << 8 | int(dev, 16) << 3 | int(fn)
    (node / "properties").write_text(
        f"cpu_cores_count 0\nlocation_id {location}\ndomain {int(domain, 16)}\n"
    )
    gpu_id = 1000 + int(name.removeprefix("card"))
    (node / "gpu_id").write_text(f"{gpu_id}\n")
    for pid, nbytes in charged:
        proc = drm / "kfd" / "proc" / pid
        proc.mkdir(parents=True, exist_ok=True)
        (proc / f"vram_{gpu_id}").write_text(f"{nbytes}\n")


def test_cards_disagreeing_with_the_spec_ptl_holding_vram_or_another_process_are_refused(
    tmp_path,
):
    # Process 77 is this one; process 5 holds no memory, as a node agent does.
    _card(
        tmp_path, "card1", "0000:0a:00.0", 3 * 10**8, "enabled", [("77", 9), ("5", 0)]
    )
    _card(tmp_path, "card9", "0000:80:00.0", 3 * 10**8)
    _card(tmp_path, "card17", "0000:a4:00.0", 9 * 10**9, "enabled")
    _card(tmp_path, "card25", "0000:c8:00.0", 3 * 10**8, "enabled", [("4242", 2**20)])
    read = probe.cards(
        ["0000:0a:00.0", "0000:80:00.0", "0000:a4:00.0", "0000:c8:00.0"],
        tmp_path,
        tmp_path / "kfd",
        me="77",
    )
    assert [c["ptl"] for c in read] == ["enabled", "disabled", "enabled", "enabled"]
    probe.refuse_unfit(read[:1], "enabled")
    with pytest.raises(SpecRefusal) as refused:
        probe.refuse_unfit(read, "enabled")
    assert "0000:0a" not in str(refused.value)
    assert "0000:80:00.0 (ptl disabled" in str(refused.value)
    assert "0000:a4:00.0 (ptl enabled, 9000000000 B used" in str(refused.value)
    assert "0000:c8:00.0 (ptl enabled, 300000000 B used, processes ['4242'])" in str(
        refused.value
    )


#: aiter's MXFP4 GEMM on an MI455 card, as it refused on 2026-10-27.
A0 = "gemm_a4w4 asm is only supported on gfx1250 B0+ (current device is gfx1250 A0)."


def test_fp4_is_timed_through_the_library_that_runs_and_names_the_one_that_failed():
    made = []

    def refusing():
        made.append("aiter")

        def call():
            raise RuntimeError(A0)

        return call

    hipblaslt, aiter, failed = {}, {}, {}
    fp4 = dict(zip(probe.FP4, ((hipblaslt, lambda: list), (aiter, refusing))))
    for shape in ("8192x8192x8192", "16384x8192x8192"):
        ((samples, call),) = probe._running(fp4, failed)
        samples[shape] = [2e-3]
    assert samples is hipblaslt and call is list
    assert failed == {"aiter.gemm_a4w4": f"RuntimeError: {A0}"}
    assert made == ["aiter"]  # tried at the first shape only
    gemm = RECORD["gemm"] | {"fp4": hipblaslt}
    found = probe.readings(RECORD | {"gemm": gemm, "aiter": {"fp4": aiter}})
    assert found["fp4"] == 2 * 16384 * 8192**2 / 2e-3


def test_the_triton_tile_is_the_fastest_within_one_percent_of_torch(monkeypatch):
    import torch

    expected = torch.ones(4, 4)
    off, broken, slow, fast = probe.TILES[False][:4]
    outputs = {off: expected * 1.02, slow: expected, fast: expected * 1.005}
    seconds = {off: 1e-4, slow: 3e-3, fast: 2e-3}

    launched = {}

    def triton(torch, operands, tile):
        if tile == broken:
            raise RuntimeError("no tile of that size on this arch")

        def call():
            pass

        launched[call] = tile
        return call, outputs.get(tile, expected * 9)

    monkeypatch.setattr(probe, "_triton", triton)
    monkeypatch.setattr(
        probe, "_timed", lambda torch, call, reps: [seconds.get(launched[call], 1)]
    )
    failed = {}
    assert (
        probe._tile(torch, ("x", "w"), lambda: expected, "triton bf16", failed) == fast
    )
    assert failed == {}
    # MXFP4 with no torch output to check against is not timed, and is named.
    assert (
        probe._tile(torch, ("x", "sx", "w", "sw"), None, "triton fp4", failed) is None
    )
    assert failed == {"triton fp4": "no torch output to check it against"}


def test_a_format_takes_triton_s_rate_where_it_is_faster():
    triton = {"bf16": {"8192x8192x8192": [1e-4]}, "fp8": {}}
    found = probe.readings(RECORD | {"triton": triton})
    assert found["bf16"] == 2 * 8192**3 / 1e-4
    assert found["fp8"] == probe.readings(RECORD)["fp8"]


def test_a_card_on_which_no_fp4_gemm_runs_is_refused_naming_each_error():
    def raising(error):
        def call():
            raise error

        return lambda: call

    fp4 = dict(
        zip(
            probe.FP4,
            (({}, raising(RuntimeError("no kernel"))), ({}, raising(RuntimeError(A0)))),
        )
    )
    with pytest.raises(SpecRefusal) as refused:
        probe._running(fp4, {})
    assert refused.value.rule is Rule.MEASURED
    assert "torch._scaled_mm RuntimeError: no kernel" in str(refused.value)
    assert f"aiter.gemm_a4w4 RuntimeError: {A0}" in str(refused.value)


def test_the_mi355x_script_publishes_the_probed_fields_fp4_included(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("COMPASS_MACHINE", raising=False)
    script = _load("mi355x_spec")
    gemm = RECORD["gemm"] | {"fp4": {"8192x8192x8192": [1e-3, 2e-3, 3e-3]}}
    found = probe.readings(RECORD | {"gemm": gemm})
    monkeypatch.setattr(script, "PEAKS", ("2026-10-19", "abc", found, "f00d"))
    document = SCRIPTS / "machines" / "mi355x-gpu45.json"
    out = tmp_path / "spec.json"
    assert script.main(["--sweep-from", str(document), "--out", str(out)]) == 0
    spec = MachineSpec.from_mapping(json.loads(out.read_text()))
    # Each derate is over the datasheet's peak, FP4's over `fp4_flops`.
    assert spec.value("device.compute.derate.fp4") == 2 * 8192**3 / 2e-3 / 1.00663e16
    assert spec.value("device.memory.derate") == 2.048e9 / 8.0e12
    for path in ("device.compute.derate.fp4", "device.memory.bandwidth_curve"):
        assert spec.origin(path)["record"] == "sha256:f00d"
    assert spec.origin("device.compute.fp4_flops")["origin"] == "datasheet"


def test_the_record_names_the_stack_aiter_s_commit_included(monkeypatch, tmp_path):
    import shutil
    import subprocess
    from types import SimpleNamespace

    import torch

    from atom.compass.drive import identity

    git = shutil.which("git")
    if git is None:
        pytest.skip("needs git")
    aiter = tmp_path / "aiter"
    aiter.mkdir()
    (aiter / "a.py").write_text("a = 1\n")
    for argv in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "s"]):
        subprocess.run(
            [git, "-C", str(aiter), "-c", "user.name=t", "-c", "user.email=t@t", *argv],
            check=True,
        )
    head = subprocess.run(
        [git, "-C", str(aiter), "rev-parse", "--short=9", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(identity, "module_root", lambda module: str(aiter))
    (tmp_path / "rocm" / ".info").mkdir(parents=True)
    (tmp_path / "rocm" / ".info" / "version").write_text("7.2.4\n")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "rocm"))
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: (2, 27, 7))
    card = SimpleNamespace(
        pci_domain_id=0, pci_bus_id=10, pci_device_id=0, gcnArchName="gfx942"
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: card)
    monkeypatch.setattr(probe, "cards", lambda bdfs: [])  # no card to refuse
    monkeypatch.setattr(probe, "probe", lambda torch, formats: RECORD)
    out = tmp_path / "record.json"
    spec = SCRIPTS / "machines" / "mi308x-hjbog18.json"
    assert probe.main([str(spec), str(out)]) == 0
    assert json.loads(out.read_text())["stack"] == {
        "rocm": "7.2.4",
        "aiter": head,
        "rccl": "2.27.7",
        "torch": torch.__version__,
    }


def test_the_cards_are_read_with_no_wait(monkeypatch, tmp_path):
    """VRAM used, KFD's charges and PTL are counters and a setting the driver
    keeps, none of which moves while a card wakes, so nothing waits for them."""
    from types import SimpleNamespace

    import torch

    from atom.compass.drive import identity

    order = []
    monkeypatch.setattr(probe, "time", SimpleNamespace(sleep=order.append))
    card = SimpleNamespace(
        pci_domain_id=0, pci_bus_id=10, pci_device_id=0, gcnArchName="gfx942"
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: card)
    monkeypatch.setattr(identity, "_stack", dict)
    monkeypatch.setattr(probe, "cards", lambda bdfs: order.append("cards") or [])
    monkeypatch.setattr(probe, "probe", lambda torch, formats: RECORD)
    spec = SCRIPTS / "machines" / "mi308x-hjbog18.json"
    assert probe.main([str(spec), str(tmp_path / "record.json")]) == 0
    assert order == ["cards"]
