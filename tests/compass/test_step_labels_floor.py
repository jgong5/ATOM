# SPDX-License-Identifier: MIT
"""A trace run's drafter labels less what the profiler's floor added to each
drafter kernel shorter than it, from planted kineto traces of two TP ranks and
`measure ops` points; the grids those points record, `measure ops --drafter`,
the floor the device probe reads, and `compass fit steps --ops`, which takes
both."""

import contextlib
import gzip
import json
import pathlib
from types import SimpleNamespace

import pytest
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops import _graph, _leaf, _t, _timed
from test_bench_ops_drafter import DRAFTING
from test_bench_ops_plan import _graph_at
from test_fit_ops import LAYOUT, _point, measured  # noqa: F401  a fixture
from test_fit_samples import DEVICE, SOURCE
from test_fit_steps import HOST, LINEAR, clients, planted_roots, record  # noqa: F401
from test_memory_readings import DOCUMENT, STACK
from test_parity_recurrence import MS, line
from test_probe_device import RECORD

from atom.compass import cli
from atom.compass.artifacts import ArtifactRefusal, ArtifactStore, Rule
from atom.compass.backends.step_labels import Floor, label_run
from atom.compass.bench import ops
from atom.compass.capture.forward import decode
from atom.compass.drive import device_probe
from atom.compass.parity import MARK, TARGET_DONE, busy

#: The floor, in seconds: twice it is 8 us.
FLOOR = 4e-6
#: Where the recorder's ranges open on the host, in us.
MARKS = {
    MARK + "d_draft_kv_ns": (80, 90),
    TARGET_DONE: (100, 101),
    MARK + "d_propose_ns": (110, 200),
}
#: One output prefill of 100 tokens, in us: the target's T1, then the draft-KV
#: pass's KV1, a 4 us launch at grid [4, 1, 1], and KV2, 9 us; the propose
#: pass's P1 and P2, 4 us each and replayed from a graph, and P3, 3 us, which
#: the host paced. Each is (name, start, end, launch, grid).
KERNELS = [
    ("t1", 10, 85, 1, [64, 1, 1]),
    ("kv1", 86, 90, 81, [4, 1, 1]),
    ("kv2", 91, 100, 83, [8, 1, 1]),
    ("p1", 120, 124, 111, None),
    ("p2", 124, 128, 112, None),
    ("p3", 152, 155, 150, None),
]
SHAPE = "prefill T=100 B=1"
#: Each kernel's own length in a point launching it alone, in ns: KV1's at its
#: grid on another shape, the rest at the graph's.
POINTS = [
    {"shape": "prefill T=8 B=2", "op": "kv", "kernels": ["kv1"], "grids": [[4, 1, 1]]},
    {"shape": SHAPE, "op": "p", "kernels": ["p1"], "grids": [None]},
    {"shape": SHAPE, "op": "q", "kernels": ["p2"], "grids": [None]},
    {"shape": SHAPE, "op": "r", "kernels": ["p3"], "grids": [None]},
]
OWN = {"kv1": 1500.0, "p1": 3000.0, "p2": 5000.0, "p3": 2000.0}


def _trace(path, forwards):
    """A kineto trace of `forwards`, each (tokens, kernels, marks), 1000 us
    after the last: one ATOM forward running its kernels, each (name, start,
    end, launch, grid) in us from its start, and opening its marks, each
    launch call 1 us long."""
    events = []
    for i, (tokens, kernels, marks) in enumerate(forwards):
        t0, name = 1000 * i, f"ATOM::forward tokens={tokens}"
        events.append({"cat": "user_annotation", "name": name, "ts": t0, "dur": 1000})
        run = {"cat": "user_annotation", "name": "ATOM::run_model", "ts": t0}
        events.append(run | {"dur": 95})
        for mark, (a, b) in marks.items():
            span = {"ts": t0 + a, "dur": b - a}
            events.append({"cat": "user_annotation", "name": mark} | span)
        for name, start, end, launch, grid in kernels:
            args = {"correlation": len(events)}
            call = {"cat": "cuda_runtime", "name": "hipLaunchKernel", "ts": t0 + launch}
            events.append(call | {"dur": 1, "args": args})
            row = {"cat": "kernel", "name": name, "ts": t0 + start, "dur": end - start}
            grid = {} if grid is None else {"grid": grid}
            events.append(row | {"args": args | {"stream": 3} | grid})
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as f:
        json.dump({"traceEvents": events}, f)


def _ranks(directory, forwards):
    """Each TP rank's trace of `forwards` in `directory`, and the busy lines
    `compass record --trace` reduces from them."""
    for rank in ("rank_0", "rank_1"):
        _trace(directory / "trace" / rank / "m_ts_1.pt.trace.json.gz", forwards)
    busy.reduce(directory)


def _run(tmp_path, **stated):
    """A trace run's directory: one output prefill line, stating `stated` too,
    which runs `KERNELS`."""
    steps = tmp_path / "steps"
    steps.mkdir()
    s = line(7, [[100, 100, False]], True, 0, (0, 0), (1, 2), None, 3)
    s |= {"rung": None, "t_read_ns": [2 * MS, 3 * MS], "d_target_ns": MS}
    s |= {"d_draft_kv_ns": 1, "d_propose_ns": 1} | stated
    (steps / "dp0.jsonl").write_text(json.dumps(s) + "\n")
    (tmp_path / "run.json").write_text("{}")
    _ranks(tmp_path, [(100, KERNELS, MARKS)])
    return steps / "dp0.jsonl"


def _labels(record, points):
    floor = Floor(FLOOR, [p | {"warm_ns": OWN[p["kernels"][0]]} for p in points])
    [([s], _)] = label_run([record], floored=lambda r: busy.floored(r, floor))
    return s


def test_each_drafter_kernel_under_the_floor_loses_its_term(tmp_path):
    record = _run(tmp_path)
    [([before], _)] = label_run([record])

    after = _labels(record, POINTS)

    # KV1 runs to the floor from 1.5 us; KV2 is past twice it. P1 runs to it
    # from 3 us, P2's own 5 us is past it, and the host paced P3, recorded at
    # 3 us of its own 2 us.
    us = 1e-6
    assert before.draft_kv_s - after.draft_kv_s == pytest.approx(2.5 * us)
    assert before.propose_s - after.propose_s == pytest.approx(2.0 * us)
    # the draft label leaves the draft-KV pass out; the device label holds it
    assert before.draft_s - after.draft_s == pytest.approx(2.0 * us)
    assert before.device_s - after.device_s == pytest.approx(2.5 * us)
    assert (after.cut, after.floor) == (None, {})


def test_a_kernel_no_point_launches_alone_keeps_its_recorded_length_named(tmp_path):
    record = _run(tmp_path)
    [([before], _)] = label_run([record])

    after = _labels(record, POINTS[:1] + POINTS[2:])

    why = [f"no measured point launches p1 alone at {SHAPE}"]
    assert after.floor == {"b_propose_ns": why, "b_draft_ns": why}
    # P1 keeps its 4 us: the propose pass loses P3's term alone
    assert before.propose_s - after.propose_s == pytest.approx(1.0e-6)
    assert before.draft_s - after.draft_s == pytest.approx(1.0e-6)
    assert before.draft_kv_s - after.draft_kv_s == pytest.approx(2.5e-6)


def test_a_kernel_launched_beside_others_is_named_not_split(tmp_path):
    record = _run(tmp_path)
    [([before], _)] = label_run([record])
    beside = POINTS[0] | {"kernels": ["kv1", "kv0"], "grids": [[4, 1, 1]] * 2}

    after = _labels(record, [beside, *POINTS[1:]])

    why = ["kv1 at grid [4, 1, 1] launches beside other kernels in kv"]
    assert after.floor == {"b_draft_kv_ns": why, "b_draft_ns": why}
    assert (after.draft_kv_s, after.device_s) == (before.draft_kv_s, before.device_s)
    assert before.propose_s - after.propose_s == pytest.approx(2.0e-6)


def test_a_graph_replayed_kernel_takes_its_length_at_its_drafter_s_shape(tmp_path):
    record = _run(tmp_path, draft_bs=4)
    [([before], _)] = label_run([record])
    # the propose pass runs the drafter's block at 4 rows: its points stand there
    drafter = "prefill T=100 B=1 W=4"
    points = [POINTS[0], *(p | {"shape": drafter} for p in POINTS[1:])]

    after = _labels(record, points)

    assert before.propose_s - after.propose_s == pytest.approx(2.0e-6)
    assert after.floor == {}


def _drafted(directory, seed):
    """`test_fit_steps.record` of `_steps(seed)` under a block drafter, whose
    lines time its target and, on an output, its propose pass, and state no
    busy time; and its lines."""
    record(directory, seed)
    path = directory / "steps" / "dp0.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for s in lines:
        for k in [k for k in s if k.startswith("b_")]:
            del s[k]
        s["d_target_ns"] = s["d_busy_ns"] // 2
        if s["produces_output"]:
            s["d_propose_ns"] = s["d_busy_ns"] // 8
    path.write_text("".join(json.dumps(s) + "\n" for s in lines))
    return directory, lines


def _traced_record(directory, seed):
    """`_drafted`'s record as a trace run: each forward's second target kernel
    queued behind its first, and on an output the propose pass's one kernel,
    P1, which a graph replays."""
    directory, lines = _drafted(directory, seed)
    forwards = []
    for s in lines:
        kernels = [("t1", 10, 40, 1, None), ("t2", 41, 60, 2, None)]
        kernels += [("p1", 120, 124, 111, None)] * s["produces_output"]
        forwards.append((sum(r[0] for r in s["rows"]), kernels, MARKS))
    _ranks(directory, forwards)
    return directory, lines


def _ops(tmp_path, monkeypatch, **engine):
    """A `measure ops --out` directory of one point, P1's at a shape no step
    runs, measured on the planted records' platform, source root and engine,
    `engine` changed in it."""
    run = {"device": [DEVICE], "stack": STACK, "ptl": "disabled"}
    run |= {"host_key": HOST, "source": SOURCE}
    out = tmp_path / "ops"
    out.mkdir()
    (out / "run.json").write_text(json.dumps(run))
    (out / "engine.json").write_text(json.dumps(HOST["engine"] | engine))
    point = {"shape": "decode B=99", "context": 1, "op": "p", "leaf": "p1"}
    point |= {"kernels": ["p1"], "grids": [None], "warm_ns": 3000.0}
    with monkeypatch.context() as m:
        m.setattr("atom.compass.run.memory.key_fields", lambda config: LAYOUT)
        m.setattr("atom.compass.run.step_grid.keyed", lambda config: HOST["engine"])
        store = ArtifactStore(str(tmp_path / "store"))
        key = ops.keys(None, run)
        ops.publish(store, key, None, run, "compass measure ops", [point], out)
    return out


def _fit_traced(tmp_path, ops_dir):
    """`compass fit steps --ops ops_dir` of `_traced_record`'s trace run and
    an ordinary record under the same drafter; the trace run's lines."""
    a, lines = _traced_record(tmp_path / "a", 19)
    b = _drafted(tmp_path / "b", 20)[0]
    floored = DOCUMENT | {"device": DOCUMENT["device"] | {"profiler_floor_s": FLOOR}}
    machine = tmp_path / "spec.json"
    machine.write_text(json.dumps(floored))
    tail = ["--machine", machine, "--store", tmp_path / "store"]
    tail += ["--out", tmp_path / "o", "--ops", ops_dir]
    cli.main(list(map(str, ["fit", "steps", *LINEAR, a, b, *clients(a, b), *tail])))
    return lines


def test_fit_steps_ops_counts_the_floor_s_refusals_on_the_steps_it_labels(
    tmp_path, monkeypatch
):
    lines = _fit_traced(tmp_path, _ops(tmp_path, monkeypatch))

    report = json.loads((tmp_path / "o" / "step_linear_law.json").read_text())
    floor = report["report"]["floor"]
    # no point gives P1's length, so each output's propose and draft labels
    # keep it; the fit cuts every decode of a trace run, so finals alone count
    finals = sum(s["produces_output"] and not s["rows"][0][2] for s in lines)
    assert sum(floor.values()) == 2 * finals
    named = "no measured point launches p1 alone at prefill T="
    assert all(named in k for k in floor)
    assert {k.split(":")[0] for k in floor} == {"b_draft_ns", "b_propose_ns"}


def test_fit_steps_ops_refuses_points_at_another_kv_block_size(tmp_path, monkeypatch):
    # a Triton kernel keeps its name and grid across a block size it compiles in
    ops_dir = _ops(tmp_path, monkeypatch, kv_cache_block_size=64)

    with pytest.raises(SystemExit, match="records: kv_cache_block_size 64, not 16"):
        _fit_traced(tmp_path, ops_dir)


def test_a_point_records_each_kernel_s_grid_where_its_profile_states_one(
    monkeypatch,
):
    # kineto states a launch's grid in the chrome trace alone; a memory set
    # has none, and a copy beside a kernel is left out
    trace = [
        ("kernel", "gemm", 0, {"grid": [8, 1, 1]}),
        ("gpu_memcpy", "Memcpy DtoD (Device -> Device)", 2, {}),
        ("gpu_memset", "Memset (Device)", 4, {}),
    ]
    events = [
        {"cat": c, "name": n, "ts": t, "dur": 1.0, "args": a} for c, n, t, a in trace
    ]
    text = json.dumps({"traceEvents": events})

    @contextlib.contextmanager
    def profile(activities):
        yield SimpleNamespace(
            export_chrome_trace=lambda path: pathlib.Path(path).write_text(text)
        )

    monkeypatch.setattr("torch.profiler.profile", profile)

    point = _timed(monkeypatch, 1.0)

    assert point["kernels"] == ["gemm", "Memset (Device)"]
    assert point["grids"] == [[8, 1, 1], None]


def test_the_probe_reads_the_profiler_s_floor_and_a_record_before_it_has_none():
    samples = [4.0e-6, 4.1e-6, 3.9e-6, 4.2e-6, 4.0e-6]

    found = device_probe.readings(RECORD | {"profiler_floor": samples})

    assert found["device.profiler_floor_s"] == pytest.approx([4.0e-6, 3.94e-6, 4.16e-6])
    assert "device.profiler_floor_s" not in device_probe.readings(RECORD)


def _fit_with_ops(tmp_path, spec, ops_dir):
    """`compass fit steps` of one planted record under `spec`, taking its
    drafter kernels' own lengths from `ops_dir`."""
    a = record(tmp_path / "a", 19)
    machine = tmp_path / "spec.json"
    machine.write_text(json.dumps(spec))
    tail = ["--machine", machine, "--store", tmp_path / "store"]
    tail += ["--out", tmp_path / "o"]
    argv = ["fit", "steps", *LINEAR, a, *clients(a), "--ops", ops_dir, *tail]
    cli.main(list(map(str, argv)))


def test_fit_steps_ops_needs_the_profiler_s_floor_its_spec_states(
    tmp_path, measured  # noqa: F811
):
    ops_dir, _ = measured("ops", [_point(2, 10.0)])

    with pytest.raises(SystemExit, match="profiler_floor_s` is declared .* no value"):
        _fit_with_ops(tmp_path, DOCUMENT, ops_dir)


def test_fit_steps_ops_refuses_points_another_configuration_ran(
    tmp_path, measured  # noqa: F811
):
    ops_dir, _ = measured("ops", [_point(2, 10.0)])
    floored = DOCUMENT | {"device": DOCUMENT["device"] | {"profiler_floor_s": FLOOR}}

    # the planted points ran another model on an MI308X
    with pytest.raises(SystemExit, match="than the records: model 'sha256:0+', not"):
        _fit_with_ops(tmp_path, floored, ops_dir)


def _shared(store, config, shape, arch=None, output=True, draft=False):
    """The target's graph, or a drafter's part that runs the target's GEMM
    beside a leaf of its own; a decode has no middle chunk's."""
    if not draft:
        return _graph_at(store, config, shape, arch)
    if not output:
        raise ArtifactRefusal(Rule.RESOLUTION, f"no op_graph holds {shape}", "")
    x = ("x", _t([shape.seqs, 8], "activation"))
    return _graph(_leaf("aiter.gemm", x), _leaf("aiter.propose", x), layers=1)


def test_measure_ops_drafter_times_the_drafter_s_leaves_each_as_a_whole_shape_has_it(
    monkeypatch,
):
    monkeypatch.setattr(ops, "graph", _shared)
    whole = ops.leaves_at(None, DRAFTING, decode(4))

    drafted = ops.leaves_at(None, DRAFTING, decode(4), drafter=True)

    # the target's attention leaves; its GEMM, which the drafter runs too, keeps
    # every occurrence and stays the target's, bound as one
    assert {op["op"] for op, _ in drafted.values()} == {"aiter.gemm", "aiter.propose"}
    assert drafted == {sig: whole[sig] for sig in drafted}
    gemm = next(op for op, _ in drafted.values() if op["op"] == "aiter.gemm")
    assert (gemm["occurrences"], "drafted" in gemm) == (4, False)
