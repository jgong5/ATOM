# SPDX-License-Identifier: MIT
"""A shipped law's platform's readings entry, committed under
`scripts/compass/laws/platform_readings/` as the readable copy `compass fit
device` writes, named by its key's token: every command that reads a readings
entry reads it where its store holds none at the device, PTL state and stack
it reads, and the spec it composes names the store or file it read; several
copies at one device and PTL state take `--stack`."""

import json
import shutil

import platforms
import pytest
from test_memory_readings import DOCUMENT
from test_platform_compose import law, model, parsed
from test_validate_bound import LINES, _main
from test_validate_project import MACHINES, ran

from atom.compass.artifacts import ArtifactStore, Key, Kind
from atom.compass.backends.json_file import read_json
from atom.compass.drive import runfile
from atom.compass.fit import device
from atom.compass.fit.device import COPIES, compose
from atom.compass.spec.layers import stack_of
from atom.compass.validate import bound

#: MI355X gfx950's copy, folded from every probe record at its key.
COPY = COPIES / "8f44fdb6fc43.json"
#: The law the first copy serves (#2893's DeepSeek-V4-Pro DPA8 1P1D).
FIRST = COPIES.parent / "deepseek-v4-pro-dpa8-mi355x-crs005-prefill.json.xz"


def test_a_run_from_a_store_with_no_entry_composes_the_copy_and_names_it(tmp_path):
    held = json.loads(COPY.read_text())
    at = {k: held["key"][k] for k in ("software_stack", "torch")}
    flags = ["--device", "mi355x", "--ptl", "disabled"]

    run = runfile.build(
        parsed(tmp_path, "--step-linear", law(tmp_path, "a.json", at), *flags),
        model(tmp_path),
    )

    machine = run["machine"]
    assert machine["device"]["software_pinned_to"]["aiter"] == "41d4d8d7e"
    assert machine["provenance"]["notes"].endswith(f"as {held['token']} in {COPY}")
    floor = machine["provenance"]["origins"]["device.kernel_floor_s"]
    assert floor == held["provenance"]["origins"]["device.kernel_floor_s"]
    assert floor["origin"] == "probed"


def test_the_stores_own_entry_comes_before_the_copy(tmp_path):
    store = tmp_path / "store"
    flags = platforms.flags(store, DOCUMENT, tmp_path)
    assert flags[3] == json.loads(COPY.read_text())["key"]["ptl"]

    spec = compose(flags[1], ArtifactStore(store), flags[3])

    assert spec.value("device.name") == "MI355X"
    assert spec.value("provenance.notes").endswith(f" in {ArtifactStore(store).root}")


def test_bound_at_another_devices_ratios_names_the_copy_it_read(monkeypatch, tmp_path):
    rec = ran(tmp_path / "rec", LINES[:3])
    priced, machines = bound.backends, []
    monkeypatch.setattr(
        bound, "backends", lambda *a: machines.append(a[3]) or priced(*a)
    )
    mi308x = json.loads((MACHINES / "mi308x-hjbog39.json").read_text())

    more = ["--ratios-from", "MI355X"]
    _main(monkeypatch, tmp_path, [rec], more=more, document=mi308x)

    (machine,) = machines
    token = json.loads(COPY.read_text())["token"]
    assert machine.value("provenance.notes").endswith(f"as {token} in {COPY}")
    assert machine.value("device.name") == "MI308X"


def test_a_platform_with_neither_an_entry_nor_a_copy_refuses(tmp_path):
    with pytest.raises(ValueError) as refused:
        compose("mi355x", ArtifactStore(tmp_path / "store"), "enabled")

    said = str(refused.value)
    assert "holds 0 platform_readings entries for MI355X gfx950 at PTL enabled" in said
    assert f", and {COPIES} 0 committed copies;" in said
    assert said.endswith("compass fit device publish one")


def test_several_copies_at_one_device_and_ptl_take_stack(tmp_path, monkeypatch):
    copies = tmp_path / "copies"
    copies.mkdir()
    shutil.copy(COPY, copies)
    other = json.loads(COPY.read_text())
    other["key"]["software_stack"] = "rocm 7.2.4 / aiter 9205e1059 / rccl 2.27.7"
    key = Key.of(Kind.PLATFORM_READINGS, **other["key"])
    other["token"] = key.token()
    (copies / f"{key.token()}.json").write_text(json.dumps(other))
    monkeypatch.setattr(device, "COPIES", copies)
    store = ArtifactStore(tmp_path / "store")

    with pytest.raises(ValueError) as refused:
        compose("mi355x", store, "disabled")
    said = str(refused.value)
    assert (
        f"holds 0 platform_readings entries for MI355X gfx950 at PTL disabled, and {copies} 2 committed copies; "
        in said
    )
    assert f"; {stack_of(other['key'])} as {key.token()}" in said
    assert said.endswith("--stack names one")

    picked = compose("mi355x", store, "disabled", stack_of(other["key"]))
    assert picked.value("device.software_pinned_to.aiter") == "9205e1059"
    named = copies / f"{key.token()}.json"
    assert picked.value("provenance.notes").endswith(f" in {named}")


def test_each_copy_is_named_by_its_keys_token():
    names = sorted(p.name for p in COPIES.iterdir())
    assert names
    for name in names:
        held = json.loads((COPIES / name).read_text())
        token = Key.of(Kind.PLATFORM_READINGS, **held["key"]).token()
        assert name == f"{token}.json" == f"{held['token']}.json"


def test_each_shipped_law_carrying_memory_readings_has_a_copy_at_its_platform():
    copied = [json.loads(p.read_text())["key"] for p in COPIES.glob("*.json")]
    laws = [read_json(p) for p in sorted(COPIES.parent.glob("*.json.xz"))]
    carrying = [d["platform"] for d in laws if Kind.MEMORY_READINGS.value in d]

    stated = [read_json(FIRST)["platform"], *carrying]

    assert [p for p in stated if p not in copied] == []
