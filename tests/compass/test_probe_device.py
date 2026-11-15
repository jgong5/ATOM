"""The attainable-peak probes: what their samples read, and the cards they
refuse."""

import pytest

from atom.compass.drive import device_probe
from atom.compass.spec import SpecRefusal
from atom.compass.spec.rules import Rule

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
    found = device_probe.readings(RECORD)
    assert found["bf16"] == 2 * 8192 * 4096 * 8192 / 2e-3  # the second shape is best
    assert [r for _, r in found["bandwidth_curve"]] == [1.024e9, 2.048e9, 2.048e9]
    assert found["link"] == 3.2e10
    assert found["device.kernel_floor_s"][0] == 3e-6


def test_fp4_takes_the_faster_library_at_each_shape():
    gemm = RECORD["gemm"] | {
        "fp4": {"8192x8192x8192": [2e-3], "16384x8192x8192": [3e-3]}
    }
    aiter = {"fp4": {"8192x8192x8192": [1e-3], "16384x8192x8192": [8e-3]}}
    found = device_probe.readings(RECORD | {"gemm": gemm, "aiter": aiter})
    assert found["fp4"] == 2 * 8192**3 / 1e-3  # aiter's, at the first shape
    assert found["bf16"] == device_probe.readings(RECORD)["bf16"]


def test_the_curve_is_at_least_every_kind_at_every_size():
    record = RECORD | {
        "read": {"4096": [2e-6], "8192": [8e-6], "16384": [8e-6]},
        "write": {"4096": [8e-6], "8192": [2e-6], "16384": [5e-6]},
    }
    curve = device_probe.readings(record)["bandwidth_curve"]
    # read is the fastest at 4 KiB, write at 8 KiB; 16 KiB keeps 8 KiB's rate
    assert curve == [
        [4096, pytest.approx(2.048e9)],
        [8192, pytest.approx(4.096e9)],
        [16384, pytest.approx(4.096e9)],
    ]
    for kind in device_probe.rates(record).values():
        assert all(rate >= kind[size] for size, rate in curve)


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
    read = device_probe.cards(
        ["0000:0a:00.0", "0000:80:00.0", "0000:a4:00.0", "0000:c8:00.0"],
        tmp_path,
        tmp_path / "kfd",
        me="77",
    )
    assert [c["ptl"] for c in read] == ["enabled", "disabled", "enabled", "enabled"]
    device_probe.refuse_unfit(read[:1], "enabled")
    with pytest.raises(SpecRefusal) as refused:
        device_probe.refuse_unfit(read, "enabled")
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
    fp4 = dict(zip(device_probe.FP4, ((hipblaslt, lambda: list), (aiter, refusing))))
    for shape in ("8192x8192x8192", "16384x8192x8192"):
        ((samples, call),) = device_probe._running(fp4, failed)
        samples[shape] = [2e-3]
    assert samples is hipblaslt and call is list
    assert failed == {"aiter.gemm_a4w4": f"RuntimeError: {A0}"}
    assert made == ["aiter"]  # tried at the first shape only
    gemm = RECORD["gemm"] | {"fp4": hipblaslt}
    found = device_probe.readings(RECORD | {"gemm": gemm, "aiter": {"fp4": aiter}})
    assert found["fp4"] == 2 * 16384 * 8192**2 / 2e-3


def test_the_triton_tile_is_the_fastest_within_one_percent_of_torch(monkeypatch):
    import torch

    expected = torch.ones(4, 4)
    off, broken, slow, fast = device_probe.TILES[False][:4]
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

    monkeypatch.setattr(device_probe, "_triton", triton)
    monkeypatch.setattr(
        device_probe,
        "_timed",
        lambda torch, call, reps: [seconds.get(launched[call], 1)],
    )
    failed = {}
    assert (
        device_probe._tile(torch, ("x", "w"), lambda: expected, "triton bf16", failed)
        == fast
    )
    assert failed == {}
    # MXFP4 with no torch output to check against is not timed, and is named.
    assert (
        device_probe._tile(torch, ("x", "sx", "w", "sw"), None, "triton fp4", failed)
        is None
    )
    assert failed == {"triton fp4": "no torch output to check it against"}


def test_a_format_takes_triton_s_rate_where_it_is_faster():
    triton = {"bf16": {"8192x8192x8192": [1e-4]}, "fp8": {}}
    found = device_probe.readings(RECORD | {"triton": triton})
    assert found["bf16"] == 2 * 8192**3 / 1e-4
    assert found["fp8"] == device_probe.readings(RECORD)["fp8"]


def test_a_card_on_which_no_fp4_gemm_runs_is_refused_naming_each_error():
    def raising(error):
        def call():
            raise error

        return lambda: call

    fp4 = dict(
        zip(
            device_probe.FP4,
            (({}, raising(RuntimeError("no kernel"))), ({}, raising(RuntimeError(A0)))),
        )
    )
    with pytest.raises(SpecRefusal) as refused:
        device_probe._running(fp4, {})
    assert refused.value.rule is Rule.MEASURED
    assert "torch._scaled_mm RuntimeError: no kernel" in str(refused.value)
    assert f"aiter.gemm_a4w4 RuntimeError: {A0}" in str(refused.value)
