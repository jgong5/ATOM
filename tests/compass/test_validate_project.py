# SPDX-License-Identifier: MIT
"""`compass validate project`: real runs' steps priced on a device none of them
ran on, published as a projection that ranks the cells and names each assumed
spec field a price read; and `validate bound --ratios-from`, a device that ran
priced at another's probed ratios, the method's error on its own records.
"""

import json
from types import SimpleNamespace

import pytest
from test_artifact_store import AITER_ROOT, ATOM_ROOT
from test_backend_step_roofline import STACK
from test_probe_device import SCRIPTS
from test_validate_bound import LINES, _main, graph, record

from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.backends.step_roofline import StepRooflineBackend
from atom.compass.spec import MachineSpec, SpecRefusal
from atom.compass.validate import bound, project

MACHINES = SCRIPTS / "machines"
MI308X, MI355X, MI455 = (
    MACHINES / f"{name}.json" for name in ("mi308x-hjbog18", "mi355x-gpu45", "mi455")
)


def spec(path):
    return MachineSpec.from_mapping(json.loads(path.read_text()))


def ran(path, lines):
    """A record of `lines` whose engine is the one `_main` resolves."""
    engine = {"tensor_parallel_size": 1, "kv_cache_dtype": "bf16"}
    (record(path, lines) / "steps" / "dp0.engine.json").write_text(json.dumps(engine))
    return path


def test_a_projected_device_keeps_its_peaks_at_the_others_probed_ratios():
    mi355x, mi308x = spec(MI355X), spec(MI308X)
    p = project.projected(mi355x, mi308x)

    for path in ("device.arch", "device.compute.bf16_flops", project.BANDWIDTH):
        assert p.value(path) == mi355x.value(path)
    for path in (
        "device.compute.derate.bf16",
        "device.memory.derate",
        "device.kernel_floor_s",
        "interconnect.intra_node.link_latency_s",
    ):
        assert (p.value(path), p.origin(path)["origin"]) == (
            mi308x.value(path),
            "assumed",
        )
    scale = mi355x.value(project.BANDWIDTH) / mi308x.value(project.BANDWIDTH)
    for (size, rate), (at, probed) in zip(
        p.value(project.CURVE), mi308x.value(project.CURVE)
    ):
        assert (size, rate) == (at, pytest.approx(probed * scale))
    # MI308X probed no FP4 rate, so a price that reads one is refused.
    with pytest.raises(SpecRefusal, match="fp4_flops"):
        p.value("device.compute.fp4_flops")
    # MI455's spec is the same method applied to MI355X.
    mi455 = project.projected(spec(MI455), mi355x).values
    assert mi455 | {"name": None} == spec(MI455).values | {"name": None}


def test_bound_ratios_from_prices_records_at_the_projected_spec(monkeypatch, tmp_path):
    rec = ran(tmp_path / "rec", LINES[:3])
    reports = []
    monkeypatch.setattr(bound, "text", lambda r: reports.append(r) or "")
    for more in ([], ["--ratios-from", str(MI308X)]):
        _main(monkeypatch, tmp_path, [rec], more=["--machine", str(MI355X), *more])

    own, at_ratios = (r["step"]["analytical_s"]["prefill"]["total"] for r in reports)
    machine = project.projected(spec(MI355X), spec(MI308X))
    direct = bound.gate([rec], {"step": StepRooflineBackend(machine, STACK)})
    assert at_ratios == direct["step"]["analytical_s"]["prefill"]["total"] != own


def _priced_here(monkeypatch):
    """Each cell priced in this process, by the stack and graphs `_main` uses."""
    config = SimpleNamespace(
        model="m",
        tensor_parallel_size=1,
        kv_cache_dtype="bf16",
        enable_tbo=False,
        enable_tbo_decode=False,
        speculative_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    tree = SimpleNamespace(modules=list)
    monkeypatch.setattr(bound, "_build", lambda *a: (config, tree))
    monkeypatch.setattr("atom.compass.run.modules.walk", lambda tree: STACK)
    monkeypatch.setattr("atom.compass.run._op_graphs", lambda *a: graph)
    monkeypatch.setattr("atom.compass.run._file_digest", lambda *a: "sha256:m")
    monkeypatch.setattr("atom.compass.run.step_grid.keyed", lambda *a, **k: {})
    monkeypatch.setattr(project, "_each", lambda jobs: [project.cell(*j) for j in jobs])
    monkeypatch.setattr(project, "roots_for", lambda modules: (ATOM_ROOT, AITER_ROOT))


def test_main_publishes_the_cells_ranked_naming_each_assumed_field_read(
    monkeypatch, tmp_path, capsys
):
    _priced_here(monkeypatch)
    decode = ran(tmp_path / "decode", LINES[:1])
    prefill = ran(tmp_path / "prefill", LINES[1:2])
    store = tmp_path / "store"
    argv = ["--model", "m", "--machine", str(MI455), "--store", str(store)]
    project.main([*argv, "--", str(decode), "--enable-tbo", "--", str(prefill)])

    token = capsys.readouterr().out.split()[-1]
    entry = ArtifactStore(store).read(
        ArtifactStore(store).key_from_token(Kind.PROJECTION, token)
    )
    doc = json.loads(entry.read_member("projection", Topology().ranks()[0], "json"))
    for level in ("step", "op"):
        assert doc["ranking"][level]["top_1"] == [str(prefill)]
        assert doc["ranking"][level]["order"] == [str(prefill), str(decode)]
    assert doc["cells"][str(decode)]["atom_args"] == ["--enable-tbo"]
    assert doc["cells"][str(decode)]["step"]["steps"] == 1
    # Assumed and read: named with its value. Datasheet, or never read: not.
    mi455 = spec(MI455)
    read = ("device.arch", "device.kernel_floor_s", "device.compute.derate.bf16")
    assert {p: mi455.value(p) for p in read}.items() <= doc["assumed"].items()
    assert project.CURVE in doc["assumed"]
    # The assumed gfx1250 device the graphs were traced at: 256 CUs, B0.
    declared = {"capture.compute_units": 256, "capture.revision": 1}
    assert declared.items() <= doc["assumed"].items()
    assert mi455.origin("host.ipc.zmq_roundtrip_s")["origin"] == "assumed"
    for unread in ("host.ipc.zmq_roundtrip_s", "device.compute.bf16_flops"):
        assert unread not in doc["assumed"]


def test_main_refuses_cells_of_two_models_and_a_run_with_no_cell(monkeypatch):
    two = [{"model": "sha256:a"}, {"model": "sha256:b"}]
    monkeypatch.setattr(project, "_each", lambda jobs: two)
    argv = ["--model", "m", "--machine", str(MI455), "--store", "s"]

    with pytest.raises(ValueError, match="different models"):
        project.main([*argv, "--", "a", "--", "b"])
    with pytest.raises(SystemExit):
        project.main(argv)
