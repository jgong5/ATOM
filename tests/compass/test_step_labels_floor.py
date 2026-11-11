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
from test_fit_ops import _point, measured  # noqa: F401  the measured fixture
from test_fit_steps import LINEAR, clients, planted_roots, record  # noqa: F401
from test_memory_readings import DOCUMENT
from test_parity_recurrence import MS, line
from test_probe_device import RECORD

from atom.compass import cli
from atom.compass.artifacts import ArtifactRefusal, Rule
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


def _trace(path):
    """A kineto trace of one forward running `KERNELS` and opening `MARKS`,
    each launch call 1 us long."""
    events = [
        {"cat": "user_annotation", "name": "ATOM::forward tokens=100", "ts": 0},
        {"cat": "user_annotation", "name": "ATOM::run_model", "ts": 0, "dur": 95},
    ]
    events[0]["dur"] = 1000
    for name, (a, b) in MARKS.items():
        events.append({"cat": "user_annotation", "name": name, "ts": a, "dur": b - a})
    for corr, (name, start, end, launch, grid) in enumerate(KERNELS):
        args = {"correlation": corr}
        call = {"cat": "cuda_runtime", "name": "hipLaunchKernel", "ts": launch}
        events.append(call | {"dur": 1, "args": args})
        row = {"cat": "kernel", "name": name, "ts": start, "dur": end - start}
        grid = {} if grid is None else {"grid": grid}
        events.append(row | {"args": args | {"stream": 3} | grid})
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as f:
        json.dump({"traceEvents": events}, f)


def _run(tmp_path):
    """A trace run's directory: one output prefill line, each rank's trace and
    the busy lines `compass record --trace` reduces from them."""
    steps = tmp_path / "steps"
    steps.mkdir()
    s = line(7, [[100, 100, False]], True, 0, (0, 0), (1, 2), None, 3)
    s |= {"rung": None, "t_read_ns": [2 * MS, 3 * MS], "d_target_ns": MS}
    s |= {"d_draft_kv_ns": 1, "d_propose_ns": 1}
    (steps / "dp0.jsonl").write_text(json.dumps(s) + "\n")
    (tmp_path / "run.json").write_text("{}")
    for rank in ("rank_0", "rank_1"):
        _trace(tmp_path / "trace" / rank / "m_ts_1.pt.trace.json.gz")
    busy.reduce(tmp_path)
    return steps / "dp0.jsonl"


def _labels(record, points):
    floor = Floor(FLOOR, [p | {"warm_ns": OWN[p["kernels"][0]]} for p in points])
    [([s], _)] = label_run([record], floor=floor)
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
