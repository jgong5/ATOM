# SPDX-License-Identifier: MIT
"""Each artifact kind keys, by value, the device, software stack, PTL state and
torch its row of the invalidation matrix marks, and no other.

The checked-in PTL-off law loads under hjbog-srdc-39's machine file and refuses
under node 18's, naming the PTL state, while its host laws, whose row leaves
PTL out, answer there. Records whose cards read another PTL state than the spec
states refuse at the fit, naming it, and so do step grid bundles and step
measurements at load. Every checked-in law states the platform of the spec it
was fitted on and this tree's `atom` digest. `scripts/compass/migrate_keys.py`
moves a law or a store entry keyed by a machine-spec digest to these keys.
"""

import copy
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_artifact_invalidation import (
    BASE,
    BEFORE_HOST_CPU,
    GATES,
    KEY_OF,
    LAYOUT,
    STATED,
    publish,
)
from test_backend_step_linear import _installed
from test_backend_step_linear_fit import _steps, _timed_record
from test_fit_samples import edit
from test_fit_steps import clients, planted_roots, record  # noqa: F401
from test_measure_memory import a_start
from test_memory_readings import DOCUMENT, META_8B, STACK, publish_readings

from atom.compass import cli
from atom.compass.artifacts import (
    KEY_FIELDS,
    MATRIX,
    PLATFORM,
    ArtifactRefusal,
    ArtifactStore,
    Axis,
    Key,
    Kind,
    Row,
    keyed_platform,
    machine_platform,
    rows_for,
)
from atom.compass.backends import StepLinearBackend
from atom.compass.drive import identity, measure_memory
from atom.compass.drive.gpu import GpuRefusal
from atom.compass.run import host_key_of
from atom.compass.spec import MachineSpec

ROOT = Path(__file__).resolve().parents[2] / "scripts" / "compass"
LAW = json.loads(
    (ROOT / "laws" / "qwen3.8-27b-tp1-mi308x-ptl-disabled-661ef7661.json").read_text()
)
N18 = json.loads((ROOT / "machines" / "mi308x-hjbog18.json").read_text())
N39 = json.loads((ROOT / "machines" / "mi308x-hjbog39.json").read_text())
#: The law at the test deployment's engine config, which `_installed` states.
HERE = {k: v for k, v in LAW.items() if k != "engine_config"}


def platform(document: dict) -> dict:
    return machine_platform(MachineSpec.from_mapping(document))


@pytest.mark.parametrize("kind", list(Kind), ids=str)
def test_every_kind_keys_each_platform_axis_its_row_marks_and_no_other(kind):
    keyed = {
        axis.field
        for axis in PLATFORM
        if any(MATRIX[row][axis].depends for row in rows_for(kind))
    }
    assert {a.field for a in PLATFORM} & set(KEY_FIELDS[kind]) == keyed
    assert "machine_spec" not in KEY_FIELDS[kind]


def test_a_host_law_keys_no_ptl_and_an_op_graph_and_a_projection_key_torch_alone():
    platform_of = {kind: keyed_platform(kind, STATED) for kind in Kind}
    assert set(platform_of[Kind.HOST_LAWS]) == {"device", "software_stack", "torch"}
    assert (
        set(platform_of[Kind.OP_GRAPH])
        == set(platform_of[Kind.PROJECTION])
        == {"torch"}
    )
    assert platform_of[Kind.SHAPE_POPULATION] == {}
    for kind in (Kind.PRICE_LIST, Kind.MEMORY_READINGS, Kind.STEP_LINEAR_LAW):
        assert platform_of[kind] == STATED


def test_the_hjbog39_law_loads_under_its_own_spec_and_refuses_under_node_18s(
    monkeypatch, tmp_path
):
    assert LAW["platform"] == keyed_platform(Kind.STEP_LINEAR_LAW, platform(N39))
    assert keyed_platform(Kind.HOST_LAWS, platform(N39)).items() <= (
        LAW["host"]["key"].items()
    )
    assert LAW["engine_config"] == LAW["host"]["key"]["engine"]
    installed = _installed(monkeypatch, tmp_path, step_linear=HERE, machine=N39)
    assert isinstance(installed, StepLinearBackend)

    with pytest.raises(ValueError) as refused:
        _installed(monkeypatch, tmp_path, step_linear=HERE, machine=N18)
    assert str(refused.value) == (
        "the step linear law was fitted on another platform: ptl 'disabled' in "
        "the law and 'enabled' here; a law measured on one platform is no "
        "evidence about another"
    )


#: The spec each checked-in law was fitted under; a law missing here fails below.
LAW_SPECS = {
    "qwen3.8-27b-tp1-mi308x-ptl-disabled-661ef7661.json": "mi308x-hjbog39.json",
    "deepseek-v4-pro-dpa8-mi355x-gpu45-prefill.json": "mi355x-gpu45.json",
    "deepseek-v4-pro-dpa8-mi355x-gpu45-decode.json": "mi355x-gpu45.json",
    "deepseek-v4-pro-dpa8-mi355x-crs005-prefill.json": "mi355x-crs005.json",
    "deepseek-v4-pro-dpa8-mi355x-crs005-decode.json": "mi355x-crs005.json",
    "deepseek-v4-pro-dpa8-ep8-mi355x-ptl-disabled-661ef7661-prefill.json": "mi355x-gpu45.json",
    "deepseek-v4-pro-dpa8-ep8-mi355x-ptl-disabled-661ef7661-decode.json": "mi355x-gpu45.json",
    "deepseek-v4-pro-dpa8-ep8-mi355x-crs005-prefill.json": "mi355x-crs005.json",
    "deepseek-v4-pro-dpa8-ep8-mi355x-crs005-decode.json": "mi355x-crs005.json",
    "deepseek-v4.1-flash-tp4-dspark5-mi455-a0.json": "mi455-a0.json",
    "deepseek-v4.1-flash-tp4-ep4-dspark5-mi455-a0.json": "mi455-a0.json",
    "deepseek-v4.1-flash-tp4-ep4-dspark5-mi355x-gpu40.json": "mi355x-gpu40.json",
}


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "laws").glob("*.json")))
def test_every_checked_in_law_states_the_platform_of_its_spec(name):
    law = json.loads((ROOT / "laws" / name).read_text())
    spec = platform(json.loads((ROOT / "machines" / LAW_SPECS[name]).read_text()))
    assert law["platform"] == keyed_platform(Kind.STEP_LINEAR_LAW, spec)
    host = {k: v for k, v in law["host"]["key"].items() if k in spec}
    assert host == keyed_platform(Kind.HOST_LAWS, spec)


@pytest.mark.parametrize("node", ["gpu45", "gpu40"])
def test_another_nodes_laws_refuse_a_crs005_run_naming_its_aiter(
    monkeypatch, tmp_path, node
):
    crs005 = json.loads((ROOT / "machines" / "mi355x-crs005.json").read_text())
    name = next(n for n, s in LAW_SPECS.items() if s == f"mi355x-{node}.json")
    law = json.loads((ROOT / "laws" / name).read_text())
    here = {k: v for k, v in law.items() if k != "engine_config"}
    with pytest.raises(ValueError) as refused:
        _installed(monkeypatch, tmp_path, step_linear=here, machine=crs005)
    theirs, ours = law["platform"]["software_stack"], platform(crs005)["software_stack"]
    assert str(refused.value) == (
        "the step linear law was fitted on another platform: software_stack "
        f"{theirs!r} in the law and {ours!r} here; a law measured on one platform "
        "is no evidence about another"
    )
    assert "aiter 41d4d8d7e" in ours and "aiter 41d4d8d7e" not in theirs


def test_every_checked_in_law_stamps_this_trees_atom_digest():
    tree = host_key_of("/nonexistent-model", {})["atom"]
    laws = sorted((ROOT / "laws").glob("*.json"))
    stamps = {p.name: json.loads(p.read_text())["host"]["key"]["atom"] for p in laws}
    assert stamps == dict.fromkeys(stamps, tree)


def test_the_ptl_off_law_refuses_on_cards_of_its_stack_with_ptl_on(
    monkeypatch, tmp_path
):
    on = edited("device.ptl", "enabled")
    with pytest.raises(ValueError, match="ptl 'disabled' in the law and 'enabled'"):
        _installed(monkeypatch, tmp_path, step_linear=HERE, machine=on)


def with_ptl(tmp_path, ptl: str) -> Path:
    """A spec file of `DOCUMENT` stating `ptl`."""
    document = copy.deepcopy(DOCUMENT)
    document["device"]["ptl"] = ptl
    path = tmp_path / f"spec-{ptl}.json"
    path.write_text(json.dumps(document))
    return path


def fit(tmp_path, a, spec: Path, *flags):
    store, out = tmp_path / "store", tmp_path / "out"
    tail = ["--machine", spec, "--store", store, "--out", out, *flags]
    argv = ["fit", "steps", "--backend", "step-linear", a, *clients(a), *tail]
    cli.main(list(map(str, argv)))
    return json.loads((out / "step_linear_law.json").read_text())


@pytest.mark.parametrize(
    "ran, stated",
    [("enabled", "disabled"), ("disabled", "enabled"), (None, "disabled")],
    ids=["on-under-off", "off-under-on", "unrecorded"],
)
def test_records_fit_only_under_a_spec_of_their_cards_ptl(tmp_path, ran, stated):
    a = record(tmp_path / "a", 19)
    edit(a / "run.json", lambda r: r.update(ptl=ran) if ran else r.pop("ptl"))
    told = f"ptl {ran or 'unrecorded'!r} in the records and {stated!r} in"
    with pytest.raises(SystemExit, match=re.escape(told)):
        fit(tmp_path, a, with_ptl(tmp_path, stated))
    assert not (tmp_path / "store").exists()

    if ran:
        law = fit(tmp_path, a, with_ptl(tmp_path, ran))
        assert law["key"]["ptl"] == law["params"]["platform"]["ptl"] == ran


def test_ptl_states_the_state_of_records_that_carry_none_and_the_report_says_so(
    tmp_path,
):
    a = record(tmp_path / "a", 19)
    edit(a / "run.json", lambda r: r.pop("ptl"))
    law = fit(tmp_path, a, with_ptl(tmp_path, "enabled"), "--ptl", "enabled")
    assert law["report"]["stated_ptl"] == "enabled"
    assert law["key"]["ptl"] == "enabled"


@pytest.mark.parametrize("carried", ["enabled", "disabled"])
def test_ptl_refuses_for_records_that_carry_a_ptl_state(tmp_path, carried):
    a = record(tmp_path / "a", 19)
    edit(a / "run.json", lambda r: r.update(ptl=carried))
    told = (
        "--ptl enabled states the PTL state of records that carry none, "
        f"and these carry {carried!r}"
    )
    with pytest.raises(SystemExit, match=re.escape(told)):
        fit(tmp_path, a, with_ptl(tmp_path, "enabled"), "--ptl", "enabled")
    assert not (tmp_path / "store").exists()


def test_a_bundle_measured_with_ptl_on_refuses_where_it_is_off(tmp_path):
    store, key = ArtifactStore(tmp_path), KEY_OF[Row.STEP_GRID_BUNDLE]
    publish(store, key, conditions=BASE.with_reading(Axis.PTL, "enabled"))
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(key, conditions=BASE, gates=GATES)
    assert "`step grid bundle` x `PTL`: recorded 'enabled', this run reads " + (
        "'disabled'"
    ) in str(refused.value)
    on = BASE.with_reading(Axis.PTL, "enabled")
    assert store.load(key, conditions=on, gates=GATES).key == key


@pytest.mark.parametrize(
    "row",
    [Row.PRICE_LIST, Row.MEMORY_READINGS, Row.STEP_GRID_BUNDLE],
    ids=lambda row: row.name,
)
def test_an_entry_published_before_the_ptl_axis_refuses_naming_it(tmp_path, row):
    store = ArtifactStore(tmp_path)
    entry = publish(store, KEY_OF[row])
    path = entry.directory / "entry.json"
    document = json.loads(path.read_bytes())
    for field, recorded in document["fingerprints"].items():
        recorded["cells"] = {
            name: BASE.reading(Axis[name.upper()]).as_json()
            for name in BEFORE_HOST_CPU[field]
        }
    path.chmod(0o644)
    path.write_text(json.dumps(document))

    with pytest.raises(ArtifactRefusal, match="now rows it over .*device, PTL"):
        store.load(entry.key, conditions=BASE, gates=GATES)


def test_cards_read_ptl_under_their_pci_device_and_none_reads_disabled(
    monkeypatch, tmp_path
):
    buses = {"card0": "0000:0A:00.0", "card1": "0000:80:00.0", "card2": "0001:0B:00.0"}
    listing = {card: {"PCI Bus": bus} for card, bus in buses.items()}
    monkeypatch.setattr(identity, "smi", lambda *args: json.dumps(listing))
    monkeypatch.setattr(identity, "PCI", str(tmp_path))
    for card in ("card0", "card2"):
        (tmp_path / buses[card].lower() / "ptl").mkdir(parents=True)
        (tmp_path / buses[card].lower() / "ptl" / "ptl_enable").write_text("enabled\n")

    assert identity._ptl((0, 2)) == "enabled"
    assert identity._ptl((1,)) == "disabled"
    with pytest.raises(GpuRefusal) as refused:
        identity._ptl((2, 1, 0))
    assert str(refused.value) == (
        "the cards of this run disagree on PTL: GPU 2 enabled, GPU 1 disabled, "
        "GPU 0 enabled"
    )


def test_a_memory_start_on_cards_of_another_ptl_refuses_before_it_starts(
    tmp_path, monkeypatch
):
    _, argv = a_start(tmp_path, monkeypatch, STACK)
    monkeypatch.setattr(measure_memory, "_ptl", lambda gpus: "enabled")
    with pytest.raises(ValueError) as refused:
        measure_memory.main(argv)
    assert str(refused.value) == (
        f"the cards read ptl 'enabled', and {tmp_path / 'machine.json'} states "
        "'disabled'; name the machine spec of the cards' PTL state"
    )
    assert not (tmp_path / "out").exists()


def test_a_law_that_states_no_platform_refuses(monkeypatch, tmp_path):
    unkeyed = {k: v for k, v in HERE.items() if k != "platform"}
    with pytest.raises(ValueError, match="device None in the law and 'MI308X gfx942'"):
        _installed(monkeypatch, tmp_path, step_linear=unkeyed, machine=N39)


@pytest.mark.parametrize(
    "given",
    [[], ["--platform", '{"step_linear_law": {}}'], ["--platform", "MI308X"]],
    ids=["none", "no host laws", "not JSON"],
)
def test_the_fit_refuses_without_each_kinds_platform_and_writes_no_law(
    tmp_path, capsys, given
):
    from atom.compass.backends.step_linear_fit import main

    record = _timed_record(tmp_path / "dp0.jsonl", _steps(19))[0]
    with pytest.raises(SystemExit):
        main([str(record), "--out", str(tmp_path / "law.json"), *given])
    assert "--platform" in capsys.readouterr().err
    assert not (tmp_path / "law.json").exists()


def _host_run(tmp_path, monkeypatch, machine: dict, fitted_on: dict):
    """`run.runner` on `machine`, its memory readings there, with a step stub
    and host laws whose key is the deployment's on the platform of
    `fitted_on`; the runner and its laws."""
    from test_run_step_grid import deployment, run_file

    from atom.compass import run as compass_run

    r = deployment(tmp_path, enable_dp_attention=False, enable_expert_parallel=False)
    publish_readings(tmp_path / "store", r.config, machine)
    key = compass_run.host_key(r.config)
    key |= keyed_platform(Kind.HOST_LAWS, platform(fitted_on))
    laws = {"decode": {"E": [1e-4, 0.0, 0.0]}}
    run = run_file(tmp_path, monkeypatch, machine=machine)
    del run["step_grid"]
    run |= {"step_stub": {}, "host": {"laws": laws, "key": key}}
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    return r, laws


def test_host_laws_fitted_with_ptl_off_answer_a_run_with_it_on(tmp_path, monkeypatch):
    on = copy.deepcopy(DOCUMENT)
    on["device"]["ptl"] = "enabled"
    r, laws = _host_run(tmp_path, monkeypatch, on, DOCUMENT)
    assert r.compass_host.laws == laws


def test_host_laws_fitted_under_another_torch_refuse_naming_it(tmp_path, monkeypatch):
    other = copy.deepcopy(DOCUMENT)
    other["device"]["software_pinned_to"]["torch"] = "2.11.0"
    with pytest.raises(ValueError) as refused:
        _host_run(tmp_path, monkeypatch, DOCUMENT, other)
    assert "host laws were fitted under another ['torch']" in str(refused.value)
    assert "'torch': '2.11.0'" in str(refused.value)


def test_a_price_list_from_another_platform_misses_naming_the_field(tmp_path):
    def price_list(document):
        return Key.of(
            Kind.PRICE_LIST,
            model="Qwen/Qwen3.8-27B",
            source_root="t",
            **LAYOUT,
            **keyed_platform(Kind.PRICE_LIST, platform(document)),
        )

    store = ArtifactStore(tmp_path)
    publish(store, price_list(N39))
    assert store.read(price_list(N39)).key == price_list(N39)
    with pytest.raises(ArtifactRefusal) as refused:
        store.read(price_list(N18))
    assert "ptl='enabled'" in str(refused.value)
    assert "differs in ptl ('disabled' there)" in str(refused.value)


def test_a_host_law_keyed_with_ptl_off_is_the_one_a_ptl_on_run_asks_for(tmp_path):
    def law(kind, document):
        given = {name: "x" for name in KEY_FIELDS[kind]}
        return Key.of(kind, **{**given, **keyed_platform(kind, platform(document))})

    store = ArtifactStore(tmp_path)
    for kind in (Kind.STEP_LINEAR_LAW, Kind.HOST_LAWS):
        publish(store, law(kind, N39))
    assert law(Kind.HOST_LAWS, N18) == law(Kind.HOST_LAWS, N39)
    assert store.read(law(Kind.HOST_LAWS, N18)).key == law(Kind.HOST_LAWS, N39)
    with pytest.raises(ArtifactRefusal) as refused:
        store.read(law(Kind.STEP_LINEAR_LAW, N18))
    assert "differs in ptl ('disabled' there)" in str(refused.value)


def test_memory_readings_are_keyed_by_the_platform():
    with pytest.raises(
        ArtifactRefusal, match="states no device, software_stack, ptl, torch"
    ):
        Key.of(
            Kind.MEMORY_READINGS,
            model="Qwen/Qwen3.8-27B",
            engine_config="{}",
            **LAYOUT,
        )


def edited(path: str, value) -> dict:
    document = copy.deepcopy(N39)
    *blocks, leaf = path.split(".")
    node = document
    for block in blocks:
        node = node[block]
    node[leaf] = value
    return document


@pytest.mark.parametrize(
    "path, value",
    [
        ("interconnect.intra_node.link_latency_s", 3e-06),
        ("device.memory.derate", 0.8),
        ("device.memory.capacity_bytes", 1),
        ("interconnect.router_relay_s", 1e-3),
        ("provenance.notes", "re-probed"),
    ],
)
def test_a_spec_edit_outside_device_and_stack_moves_no_platform(path, value):
    assert platform(edited(path, value)) == platform(N39)


@pytest.mark.parametrize(
    "path, value",
    [
        ("device.software_pinned_to.aiter", "f4e7c7509"),
        ("device.software_pinned_to.rocm", "7.2.5"),
        ("device.software_pinned_to.torch", "2.11.0"),
        ("device.arch", "gfx950"),
        ("device.name", "MI355X"),
        ("device.ptl", "enabled"),
    ],
)
def test_a_device_or_stack_edit_moves_the_platform(path, value):
    assert platform(edited(path, value)) != platform(N39)


def test_a_spec_that_states_no_ptl_is_keyed_as_ptl_disabled():
    unstated = copy.deepcopy(N39)
    assert unstated["device"].pop("ptl") == "disabled"
    assert platform(unstated) == platform(N39)


# --- migrate_keys.py -------------------------------------------------------

MIGRATE = ROOT / "migrate_keys.py"
LAW_NAME = "qwen3.8-27b-tp1-mi308x-ptl-disabled-661ef7661.json"
PLATFORM_FIELDS = {a.field for a in PLATFORM}


def digest(document: dict) -> str:
    """The machine-spec digest a key stated for `document` before keys stated
    their platform."""
    old = {k: v for k, v in platform(document).items() if k != "torch"}
    canonical = json.dumps(
        {"kind": "machine_spec", "key": old}, sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def migrate(*args) -> list[str]:
    done = subprocess.run(
        [sys.executable, str(MIGRATE), *map(str, args)],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.splitlines()


def keyed_by_digest(law: dict, document: dict) -> dict:
    """`law` as it stood keyed by `document`'s machine-spec digest."""
    stale = {"machine_spec": digest(document)}
    host = {k: v for k, v in law["host"]["key"].items() if k not in PLATFORM_FIELDS}
    old = {k: v for k, v in law.items() if k != "platform"} | stale
    return old | {"host": {**law["host"], "key": host | stale}}


def test_a_law_keyed_by_a_digest_is_rewritten_to_its_platform_and_an_unknown_listed(
    tmp_path,
):
    old = keyed_by_digest(LAW, N39)
    (tmp_path / LAW_NAME).write_text(json.dumps(old) + "\n")
    stray = old | {"machine_spec": "sha256:" + "0" * 64}
    (tmp_path / "stray.json").write_text(json.dumps(stray) + "\n")

    lines = migrate("--laws", tmp_path)

    assert lines == [
        f"MOVED {LAW_NAME}",
        f"LISTED stray.json: its digests {sorted([stray['machine_spec'], digest(N39)])} "
        + "name no spec",
        "1 moved, 1 listed",
    ]
    assert json.loads((tmp_path / LAW_NAME).read_text()) == LAW
    assert json.loads((tmp_path / "stray.json").read_text()) == stray


def test_a_law_whose_digest_names_specs_differing_in_torch_is_listed(tmp_path):
    spec = importlib.util.spec_from_file_location("migrate_keys", MIGRATE)
    migrate_keys = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migrate_keys)
    other = copy.deepcopy(N39)
    other["device"]["software_pinned_to"]["torch"] = "2.11.0"
    machines = tmp_path / "machines"
    machines.mkdir()
    for name, document in (("a.json", N39), ("b.json", other)):
        (machines / name).write_text(json.dumps(document))
    (tmp_path / LAW_NAME).write_text(json.dumps(keyed_by_digest(LAW, N39)))

    known = migrate_keys.platforms(machines)

    assert known == {
        digest(N39): {k: v for k, v in platform(N39).items() if k != "torch"}
    }
    assert migrate_keys.law(tmp_path / LAW_NAME, known) == (
        f"LISTED {LAW_NAME}: the specs its digests name differ in 'torch'"
    )


def test_a_store_entry_keyed_by_a_digest_is_republished_at_its_platform_key(tmp_path):
    old_store, new_store = ArtifactStore(tmp_path / "old"), tmp_path / "new"
    key = Key.of(
        Kind.MEMORY_READINGS,
        model="Qwen/Qwen3.8-27B",
        engine_config="{}",
        **LAYOUT,
        **keyed_platform(Kind.MEMORY_READINGS, platform(N39)),
    )
    entry = publish(old_store, key)
    stated = {k: v for k, v in key.values.items() if k not in PLATFORM_FIELDS}
    for name, value in (("known", digest(N39)), ("unknown", "sha256:" + "0" * 64)):
        document = json.loads((entry.directory / "entry.json").read_bytes())
        document["key"] = {**stated, "machine_spec": value}
        place = entry.directory.with_name(name)
        place.mkdir()
        for member in document["members"]:
            shutil.copy(entry.directory / member, place / member)
        (place / "entry.json").write_text(json.dumps(document))
    shutil.rmtree(entry.directory)

    lines = migrate("--store", tmp_path / "old", new_store)

    assert lines[0] == f"MOVED memory_readings known -> {key.token()}"
    assert re.fullmatch(
        "LISTED memory_readings unknown: digest sha256:0{64} names no spec in .*",
        lines[1],
    )
    assert lines[2] == "1 moved, 1 listed"
    moved = ArtifactStore(new_store).load(key, conditions=BASE, gates=GATES)
    assert moved.members == entry.members
    assert '"machine_spec": "' + digest(N39) in moved.notes


def test_an_op_graph_entry_takes_its_torch_from_the_reading_it_recorded(tmp_path):
    key = KEY_OF[Row.OP_GRAPH]
    entry = publish(ArtifactStore(tmp_path / "old"), key)
    document = json.loads((entry.directory / "entry.json").read_bytes())
    document["key"].pop("torch")
    path = entry.directory / "entry.json"
    path.chmod(0o644)
    path.write_text(json.dumps(document))

    assert migrate("--store", tmp_path / "old", tmp_path / "new")[0] == (
        f"MOVED op_graph {entry.directory.name} -> {key.token()}"
    )
    assert ArtifactStore(tmp_path / "new").read(key).key == key
