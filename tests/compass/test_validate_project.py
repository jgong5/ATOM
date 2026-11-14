# SPDX-License-Identifier: MIT
"""`compass project`: real runs' steps priced on a device none of them
ran on, published as a projection that ranks the cells and names each assumed
spec field a price read; and `validate.bound --ratios-from`, a device that ran
priced at another's probed ratios, the method's error on its own records.
"""

import json
from types import SimpleNamespace

import platforms
import pytest
from test_artifact_store import AITER_ROOT, ATOM_ROOT
from test_backend_step_roofline import STACK
from test_probe_device import SCRIPTS
from test_validate_bound import LINES, _main, drafted, graph, record

from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.backends.step_roofline import StepRooflineBackend
from atom.compass.fit import device
from atom.compass.spec import MachineSpec
from atom.compass.validate import bound, project

MACHINES = SCRIPTS / "machines"
#: MI455's B0 spec is MI355X's probed ratios on MI455X's datasheet, for a
#: device no engine has run on.
MI355X, MI455 = (MACHINES / f"{name}.json" for name in ("mi355x-gpu45", "mi455-b0"))


def spec(path):
    return MachineSpec.from_mapping(json.loads(path.read_text()))


def on_mi455(store, directory) -> list[str]:
    """The platform flags of MI455's B0 spec, its readings in `store`."""
    return platforms.flags(store, json.loads(MI455.read_text()), directory)


def ran(path, lines):
    """A record of `lines` whose engine is the one `_main` resolves."""
    engine = {"tensor_parallel_size": 1, "kv_cache_dtype": "bf16"}
    (record(path, lines) / "steps" / "dp0.engine.json").write_text(json.dumps(engine))
    return path


def test_bound_ratios_from_prices_records_at_the_other_devices_ratios(
    monkeypatch, tmp_path
):
    rec = ran(tmp_path / "rec", LINES[:3])
    reports = []
    monkeypatch.setattr(bound, "text", lambda r: reports.append(r) or "")
    mi308x = MACHINES / "mi308x-hjbog39.json"
    platforms.flags(tmp_path / "store", json.loads(mi308x.read_text()), tmp_path)
    mi355x = json.loads(MI355X.read_text())
    for more in ([], ["--ratios-from", "MI308X"]):
        _main(monkeypatch, tmp_path, [rec], more=more, document=mi355x)

    own, at_ratios = (r["step"]["analytical_s"]["prefill"]["total"] for r in reports)
    # MI355X's peaks at MI308X's derates, MI308X's curve at MI355X's bandwidth,
    # and MI308X's values no datasheet states.
    target, probed = spec(MI355X), spec(mi308x)

    def taken(path):
        return "derate" in path.split(".") or path.startswith(device.CARRIED)

    values = {p: v for p, v in target.values.items() if not taken(p)}
    values |= {p: v for p, v in probed.values.items() if taken(p)}
    scale = target.value(project.BANDWIDTH) / probed.value(project.BANDWIDTH)
    values[project.CURVE] = [[s, r * scale] for s, r in probed.value(project.CURVE)]
    machine = MachineSpec(values)
    direct = bound.gate([rec], {"step": StepRooflineBackend(machine, STACK)})
    assert at_ratios == pytest.approx(
        direct["step"]["analytical_s"]["prefill"]["total"]
    )
    assert at_ratios != own


def _priced_here(monkeypatch, drafter=None):
    """Each cell priced in this process, by the stack and graphs `_main` uses,
    under `drafter`, a speculative config."""
    config = SimpleNamespace(
        model="m",
        tensor_parallel_size=1,
        kv_cache_dtype="bf16",
        enable_tbo=False,
        enable_tbo_decode=False,
        enable_expert_parallel=False,
        speculative_config=drafter,
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    tree = SimpleNamespace(modules=list)
    monkeypatch.setattr(bound, "_build", lambda *a: (config, tree))
    monkeypatch.setattr("atom.compass.run.modules.walk", lambda tree, d=None: STACK)
    monkeypatch.setattr("atom.compass.run.modules.drafter", lambda config: None)
    monkeypatch.setattr("atom.compass.run._op_graphs", lambda *a, **kw: graph)
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
    argv = ["--model", "m", *on_mi455(store, tmp_path), "--store", str(store)]
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
    assert mi455.origin("interconnect.router_relay_s")["origin"] == "assumed"
    for unread in ("interconnect.router_relay_s", "device.compute.bf16_flops"):
        assert unread not in doc["assumed"]
    # Each assumed ratio and the kernel floor read, every cell priced again at
    # 0.8 and 1.2 of it.
    assert set(doc["band"]) == {
        p
        for p in doc["assumed"]
        if p in (project.CURVE, project.FLOOR) or "derate" in p
    }
    for level in ("step", "op"):
        base = doc["cells"][str(prefill)][level]["tokens_per_s"]
        slower = doc["band"]["device.compute.derate.bf16"]["0.8"]["tokens_per_s"]
        assert slower[level][str(prefill)] < base


def test_a_cell_under_a_drafter_states_the_drafters_seconds_apart(
    monkeypatch, tmp_path
):
    mtp1 = SimpleNamespace(num_speculative_tokens=1, use_dspark=lambda: False)
    _priced_here(monkeypatch, mtp1)
    monkeypatch.setattr("atom.compass.run._op_graphs", lambda *a, **kw: drafted)
    line = {"rows": [[2, 900, True]] * 2, "rung": 2, "d_target_ns": 10**9}
    decode = ran(tmp_path / "mtp", [line])

    cell = project.cell("m", 1, json.loads(MI455.read_text()), "s", str(decode), [])

    op = cell["levels"]["op"]
    # The drafter costs more than the target, and is not in its seconds.
    assert op["draft_s"]["decode"] > op["analytical_s"]["decode"]["total"] > 0


def test_main_reads_a_records_atom_arguments_from_its_run_json(monkeypatch, tmp_path):
    """Less the server's and `--model`; a record whose run.json names none is
    refused alone and priced under its own `--` group."""
    _priced_here(monkeypatch)
    build, built = bound._build, []
    monkeypatch.setattr(bound, "_build", lambda *a: built.append(a[2]) or build(*a))
    served, bare = (ran(tmp_path / n, LINES[i : i + 1]) for i, n in enumerate("ab"))
    line = ["compass", "record", "--out", str(served), "--", "--model", "/there"]
    line += ["--enable-tbo", "--tool-call-parser", "qwen3"]
    (served / "run.json").write_text(json.dumps({"argv": line}))
    (bare / "run.json").write_text(json.dumps({"argv": ["compass", "record"]}))
    argv = ["--model", "m", *on_mi455(tmp_path / "s", tmp_path)]
    argv += ["--store", str(tmp_path / "s")]

    with pytest.raises(ValueError) as refused:
        project.main([*argv, str(served), str(bare)])
    assert str(refused.value).endswith(f"name the record as -- {bare} [ATOM args]")
    project.main([*argv, str(served), "--", str(bare), "--kv_cache_dtype", "bf16"])

    assert built[:2] == [["--enable-tbo"], ["--kv_cache_dtype", "bf16"]]


@pytest.mark.parametrize(
    "draft", [["--draft-model", "/there/draft"], ["--draft_model=/there/draft"]]
)
def test_a_record_naming_a_draft_model_is_refused_naming_the_dash_form(tmp_path, draft):
    """A draft model's path is the host's that ran the record, as `--model` is."""
    line = ["compass", "record", "--", "--model", "/there/target"]
    line += ["--method", "eagle3", *draft, "--num-speculative-tokens", "3"]
    (tmp_path / "run.json").write_text(json.dumps({"argv": line}))

    with pytest.raises(ValueError) as refused:
        project.recorded(str(tmp_path))
    assert str(refused.value) == (
        f"{tmp_path / 'run.json'} names --draft-model, a path on the host that "
        f"ran it; name the record as -- {tmp_path} [ATOM args]"
    )


def test_main_refuses_cells_of_two_models_and_a_run_with_no_cell(monkeypatch, tmp_path):
    two = [{"model": "sha256:a"}, {"model": "sha256:b"}]
    monkeypatch.setattr(project, "_each", lambda jobs: two)
    store = tmp_path / "s"
    argv = ["--model", "m", *on_mi455(store, tmp_path), "--store", str(store)]

    with pytest.raises(ValueError, match="different models"):
        project.main([*argv, "--", "a", "--", "b"])
    with pytest.raises(SystemExit):
        project.main(argv)


def test_a_ratio_times_its_factor_is_capped_at_the_datasheet_and_says_so():
    mi455 = spec(MI455)
    link = "interconnect.intra_node.derate"
    for factor, capped in ((1.2, False), (1.3, True)):
        p, cap = project.scaled(mi455, link, factor)
        assert (p.value(link), cap) == (min(mi455.value(link) * factor, 1.0), capped)
        assert p.values | {link: None} == mi455.values | {link: None}

    top = mi455.value(project.BANDWIDTH)
    p, cap = project.scaled(mi455, project.CURVE, 1.3)
    assert cap and p.value(project.CURVE) == [
        [size, min(rate * 1.3, top)] for size, rate in mi455.value(project.CURVE)
    ]
    assert not project.scaled(mi455, project.CURVE, 1.2)[1]


def test_the_band_ranks_the_cells_again_at_each_ratio_one_at_a_time(monkeypatch):
    bf16, link = "device.compute.derate.bf16", "interconnect.intra_node.derate"
    mi455 = spec(MI455)
    machine = MachineSpec(mi455.values | {link: 0.9})

    def rates(jobs):
        """Each cell's tokens a second, read off the spec it was priced on."""
        out = []
        for job in jobs:
            s = MachineSpec.from_mapping(job[2])
            by_cell = {"a": s.value(bf16), "b": 0.6, "c": s.value(link) / 2, "d": 0.42}
            levels = {
                "step": {"tokens_per_s": by_cell[job[4]]},
                "op": {"tokens_per_s": 1},
            }
            out.append({"levels": levels})
        return out

    monkeypatch.setattr(project, "_each", rates)
    jobs = [("m", 1, machine.echo(), "s", r, []) for r in "abcd"]
    base = project._ranked({j[4]: (None, c) for j, c in zip(jobs, rates(jobs))})
    assert base["step"]["order"] == ["a", "b", "c", "d"]
    floor = "device.kernel_floor_s"
    assumed = {p: machine.value(p) for p in (bf16, link, floor)}

    band = project.band(machine, assumed, jobs, base)

    # The kernel floor too, a carried value with no datasheet peak.
    assert set(band) == {bf16, link, floor}
    slow = band[bf16]["0.8"]
    assert slow["ranking"]["step"]["top_1"] == ["b"]
    assert slow["order_holds"] == {"step": False, "op": True}
    assert slow["tokens_per_s"]["step"]["a"] == pytest.approx(0.8 * assumed[bf16])
    assert band[bf16]["1.2"]["order_holds"]["step"]
    assert [band[link][f]["capped"] for f in ("0.8", "1.2")] == [False, True]
    assert band[link]["1.2"]["tokens_per_s"]["step"]["c"] == 0.5
    # At 0.8 the third and fourth cells change places; the top-1 holds.
    low = band[link]["0.8"]
    assert low["ranking"]["step"]["top_1"] == ["a"] and not low["order_holds"]["step"]


def test_no_process_prices_two_cells_whatever_the_pool_width(monkeypatch):
    """A build declares one engine per process, so a worker takes one cell
    and exits; the pool is as wide as the CPUs this process may use."""
    seen = []

    class Pool:
        def __init__(self, width, maxtasksperchild):
            seen.append((width, maxtasksperchild))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def starmap(self, fn, jobs, chunksize=None):
            seen.append(chunksize)
            return [fn] * len(jobs)

    spawn = SimpleNamespace(Pool=Pool)
    monkeypatch.setattr(project.multiprocessing, "get_context", lambda how: spawn)
    monkeypatch.setattr(project.os, "sched_getaffinity", lambda pid: range(8))

    assert project._each([()] * 40) == [project.cell] * 40
    assert seen == [(8, 1), 1]


def test_the_floor_and_each_link_latency_read_are_banded_uncapped(monkeypatch):
    floor, hop = project.FLOOR, "interconnect.intra_node.link_latency_s"
    mi455 = spec(MI455)
    priced = []

    def rates(jobs):
        """One cell, its tokens a second falling with the latency."""
        out = []
        for job in jobs:
            s = MachineSpec.from_mapping(job[2])
            priced.append((s.value(floor), s.value(hop)))
            rate = {"tokens_per_s": 1 / s.value(hop)}
            out.append({"levels": {"step": rate, "op": rate}})
        return out

    monkeypatch.setattr(project, "_each", rates)
    jobs = [("m", 1, mi455.echo(), "s", "a", [])]
    base = project._ranked({"a": (None, rates(jobs)[0])})
    priced.clear()
    assumed = {p: mi455.value(p) for p in (floor, hop, "device.arch")}

    band = project.band(mi455, assumed, jobs, base)

    assert {p: list(at) for p, at in band.items()} == {
        floor: ["0.8", "1.2"],
        hop: ["0.8", "1.2", "2", "4"],
    }
    assert not any(b["capped"] for at in band.values() for b in at.values())
    t0, alpha = assumed[floor], assumed[hop]
    assert priced == [(k * t0, alpha) for k in (0.8, 1.2)] + [
        (t0, k * alpha) for k in (0.8, 1.2, 2, 4)
    ]
    assert band[hop]["4"]["tokens_per_s"]["step"]["a"] == pytest.approx(1 / (4 * alpha))
