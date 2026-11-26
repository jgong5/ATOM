# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A decode step of one token a row under an MTP drafter.

A P/D producer under an MTP drafter runs decode steps whose rows each decode
one token, verifying no draft. `OpGraphBackend` reads such a step's graph,
its target's and its drafter's, at its shape verifying none (``V=0``);
`compass trace ops --population` traces it there under the producer's
transfer config and refuses it without one; `measure ops --plan` measures
its drafter where a run asks it, and a trace run's floor reads its drafter's
kernels at that shape.
"""

import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_artifact_store import (
    clean_checkout,  # noqa: F401  autouse
    stanza,
)
from test_backend_step_linear import view
from test_bench_ops_drafter import _mtp
from test_bench_ops_drafter_dp_decode import DP, DRAFTER, _plan, _staging
from test_bench_ops_plan import CONFIG as PLAN
from test_bench_ops_plan import bench  # noqa: F401  a fixture
from test_bench_ops_real_rows import FUSED, _routed_bench, _routed_graph, _steps
from test_capture_ir import MAIN, TREE, _ops, _record
from test_op_graph import g
from test_op_graph_arch import ENGINE, run_readings
from test_op_graph_drafter import ATTENTION, CONFIG, VERIFIER
from test_parity_recurrence import MS, line
from test_step_labels_floor import FLOOR, KERNELS, MARKS, OWN, POINTS, _ranks
from test_work import SPEC

from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    GateState,
    Kind,
    Topology,
    member_name,
)
from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.op_graph import OpGraphBackend, step_shape, verified
from atom.compass.backends.provenance import Method
from atom.compass.backends.step_labels import Floor
from atom.compass.backends.work import gemm, price
from atom.compass.bench import ops
from atom.compass.capture import graph_key, population_shapes
from atom.compass.capture.forward import _batch, decode, structure
from atom.compass.fit import ops as fit_ops
from atom.compass.parity import busy
from atom.compass.run import _op_graphs

#: Two rows decoding one token each, replayed at rung 2.
ONE_A_ROW = view((1, 90, True), (1, 9, True), rung=2)


@pytest.mark.parametrize("drafts", [1, 3])
def test_a_decode_of_one_token_a_row_under_a_drafter_reads_graphs_verifying_none(
    drafts,
):
    asked = []

    def graph(*at, draft=False, verifies=None):
        asked.append((*at, draft, verifies))
        return g(16, 64, 64) if draft else g(4, 8, 8)

    cost = OpGraphBackend(SPEC, graph, kv="bf16", drafts=drafts).price(ONE_A_ROW)

    assert asked == [("decode", 2, 2, True, False, 0), ("decode", 2, 2, True, True, 0)]
    assert (cost.target_seconds, cost.draft_seconds) == (
        price(gemm(4, 8, 8), SPEC),
        price(gemm(16, 64, 64), SPEC),
    )


def test_a_verify_step_reads_its_graphs_at_its_drafter_s_drafts():
    asked = []

    def graph(*at, **kwargs):
        asked.append(kwargs)
        return g(4, 8, 8)

    batch = view((2, 90, True), (2, 9, True), rung=2)
    OpGraphBackend(SPEC, graph, kv="bf16", drafts=1).price(batch)

    assert asked == [{}, {"draft": True}]


def test_a_decode_mixing_one_token_rows_and_verify_rows_is_refused_by_name():
    batch = view((1, 90, True), (2, 9, True), rung=2)
    with pytest.raises(CostRefused, match="decodes a row of other than 2 tokens"):
        OpGraphBackend(SPEC, lambda *a, **k: g(4, 8, 8), kv="bf16", drafts=1).price(
            batch
        )


def test_a_graph_missing_at_a_shape_verifying_none_is_named_so():
    with pytest.raises(CostRefused, match="verifying 0 drafts a row"):
        OpGraphBackend(SPEC, lambda *a, **k: None, kv="bf16", drafts=1).price(ONE_A_ROW)


def test_verified_and_step_shape_place_a_one_token_decode_under_a_drafter():
    rows = [(1, True), (1, True)]
    assert verified(rows, 3) == 0
    assert verified([(4, True)], 3) == 3
    assert verified(rows, 0) == 0
    assert step_shape(rows, 4, 3) == ("decode", 4, 4)


def test_a_population_under_a_drafter_traces_a_one_token_decode_verifying_none(
    tmp_path,
):
    record = _record(
        tmp_path,
        [
            {"rows": [[1, 90, True]] * 3, "rung": 4, "draft_bs": 4},
            {"rows": [[2, 90, True]] * 2, "rung": 2, "draft_bs": 2},
        ],
    )

    shapes = population_shapes([record], run_drafts=1)

    assert list(map(str, shapes)) == [
        "decode B=4 V=0",
        "decode T=3 B=4 V=0",
        "decode B=2",
    ]
    assert shapes[0] == decode(4)._replace(verifies=0)


def test_a_population_with_no_drafter_traces_a_one_token_decode_as_before(tmp_path):
    record = _record(tmp_path, [{"rows": [[1, 90, True]] * 3, "rung": 4}])
    assert list(map(str, population_shapes([record]))) == ["decode B=4"]


def test_a_trace_at_a_shape_verifying_none_schedules_one_token_a_row_as_a_drafter_s():
    verify, _ = _batch(decode(4), 16, 100, True, 3)
    one, _ = _batch(decode(4)._replace(verifies=0), 16, 100, True, 3)

    assert list(verify.num_scheduled_tokens) == [4] * 4 and verify.num_spec_step == 3
    assert list(one.num_scheduled_tokens) == [1] * 4 and one.num_spec_step == 0
    # still a drafter's step, which ATOM drafts after, not a dummy run
    assert not one.is_dummy_run


#: A P/D prefill side's transfer config, as a 1P1D cell's prefill record states it.
PRODUCER = '{"kv_role":"kv_producer","kv_connector":"mooncake"}'


def _trace(tmp_path, *atom_args):
    """`compass trace ops` under an MTP drafter of one token, on
    Qwen3.5-35B-A3B-FP8's config and ATOM's arguments `atom_args`, of the
    width 2 and the `--population` of a record whose one step decodes 3 rows
    of one token each at rung 4: the process, and its graphs by shape and
    whether a drafter's."""
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    (tmp_path / "record").mkdir()
    step = {"rows": [[1, 90, True]] * 3, "rung": 4, "draft_bs": 4}
    record = _record(tmp_path / "record", [step | {"produces_output": True}])
    argv = ["trace", "ops", "--model", str(tmp_path / "model"), "--width", "2"]
    argv += ["--population", record, "--store", str(tmp_path / "store")]
    argv += ["--", "--method", "mtp"]
    argv += ["--num-speculative-tokens", "1", *atom_args]
    done = subprocess.run(
        [sys.executable, "-c", MAIN, *argv],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    store, rank = ArtifactStore(str(tmp_path / "store")), Topology().ranks()[0]
    graphs = {}
    for token in re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE):
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph = json.loads(entry.read_member("graph", rank, "json"))
        graphs[graph["shape"], graph["structure"].endswith(" draft")] = graph
    return done, graphs


def test_a_trace_off_a_producer_refuses_a_decode_verifying_none_by_name(tmp_path):
    done, graphs = _trace(tmp_path)

    assert done.returncode != 0 and not graphs
    assert "decode B=4 V=0 verifies fewer drafts than its drafter's" in done.stderr


def test_a_producer_traces_a_one_token_decode_verifying_none_and_its_drafter(
    tmp_path,
):
    done, graphs = _trace(tmp_path, "--kv-transfer-config", PRODUCER)

    assert done.returncode == 0, done.stderr[-4000:]
    target = _ops(graphs["decode B=4 V=0", False]["region"])
    drafter = _ops(graphs["decode T=3 B=4 V=0", True]["region"])
    # the target runs one token a row at its rung and verifies no draft
    assert {op["args"]["q"]["shape"][0] for op in target if op["op"] == ATTENTION} == {
        4
    }
    assert VERIFIER not in {op["op"] for op in target}
    # its drafter proposes after it
    assert [op["op"] for op in drafter].count(ATTENTION) == 1
    # a producer's width verifies none too
    width = _ops(graphs["decode B=2 V=0", False]["region"])
    assert {op["args"]["q"]["shape"][0] for op in width if op["op"] == ATTENTION} == {2}


def test_a_dp_drafter_s_later_pass_on_a_one_token_decode_is_measured_where_asked(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # 3 rows decoding one token each under a drafter of two, padded to 4, on DP
    # rank 0 of two: the target's MoE gathers 8 rows, the drafter's first pass
    # 6, each its default verifying none.
    step = {"rows": [[1, 1000, True]] * 3, "rung": 4, "draft_bs": 4}
    step |= {"moe_rows": 8, "draft_moe_rows": 6}

    records, report = _plan(bench, tmp_path, monkeypatch, step)

    params = fit_ops.fit([r for r in records if "refused" not in r])
    fit_ops.held_out(params)

    def graph(phase, t, b, output, *_, draft=False, width=None, verifies=None):
        shape = decode(b, None, t)._replace(verifies=verifies)
        return _staging(bench[0], DP, shape, None, output, draft)["region"]

    listed = fit_ops.PriceList(params)
    backend = OpGraphBackend(SPEC, graph, kv="bf16", dp=2, drafts=2, price_list=listed)
    batch = BatchView(
        tuple(RequestShape(*row) for row in step["rows"]),
        capture_rung=4,
        produces_output=True,
        **{f: step[f] for f in ("moe_rows", "draft_moe_rows", "draft_bs")},
    )
    cost = backend.price(batch)
    assert {t.provenance.method for t in cost.draft_terms} == {Method.MEASURED}
    assert cost.draft_seconds == pytest.approx(len(DRAFTER) * 1e-6)
    assert report["by part"] == {
        "target": {"steps": 1, "covered": 1},
        "draft": {"steps": 1, "covered": 1},
    }


def test_the_floor_reads_a_one_token_decode_s_drafter_kernels_at_its_shape(tmp_path):
    # A trace run under MTP1 whose one step decodes 2 rows of one token each:
    # each drafter kernel's own length stands at the step's shape verifying none.
    (steps := tmp_path / "steps").mkdir()
    s = line(7, [[1, 100, True]] * 2, True, 0, (0, 0), (1, 2), None, 3)
    s |= {"rung": 2, "t_read_ns": [2 * MS, 3 * MS], "d_target_ns": MS}
    s |= {"d_draft_kv_ns": 1, "d_propose_ns": 1}
    (steps / "dp0.jsonl").write_text(json.dumps(s) + "\n")
    engine = {"speculative": ["mtp", 1]}
    (tmp_path / "run.json").write_text(json.dumps({"host_key": {"engine": engine}}))
    _ranks(tmp_path, [(2, KERNELS, MARKS)])
    points = [POINTS[0], *(p | {"shape": "decode B=2 V=0"} for p in POINTS[1:])]
    floor = Floor(FLOOR, [p | {"warm_ns": OWN[p["kernels"][0]]} for p in points])

    [row] = busy.floored(steps / "dp0.jsonl", floor)

    assert row["floor"] == {}


def test_measure_ops_captures_a_producer_s_width_verifying_none(monkeypatch):
    argvs = []
    monkeypatch.setattr(ops, "_child", lambda fn, argv: argvs.append(argv))
    config = SimpleNamespace(model="m", speculative_config=SimpleNamespace())
    args = ["--method", "mtp", "--num-speculative-tokens", "1", "--kv-transfer-config"]
    one = decode(4)._replace(verifies=0)

    ops._capture(config, [one], "store", "gfx942", [*args, PRODUCER])

    [argv] = argvs
    assert argv[argv.index("--width") + 1] == "4"
    # off a producer no width traces it
    with pytest.raises(ArtifactRefusal, match="no op_graph holds decode B=4 V=0"):
        ops._capture(config, [one], "store", "gfx942", [*args, "{}"])


def test_a_run_reads_a_dp_decode_verifying_none_from_its_own_graph(tmp_path):
    # DPA DP8 under MTP1: 2 rows of one token each gather 16 rows, the default
    # there, beside the verify graph of the same width, which gathers 32.
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    config = SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=8),
        enable_expert_parallel=True,
        enforce_eager=True,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        **ENGINE | {"enable_dp_attention": True},
    )
    store, at = ArtifactStore(str(tmp_path / "store")), structure(
        config, "decode", True, "gfx950"
    )
    for shape in ("decode B=2", "decode B=2 V=0"):
        graph = json.dumps({"region": {"op": shape}}).encode()
        store.publish(
            graph_key(at, shape),
            provenance=stanza("compass trace ops"),
            topology=Topology(),
            conditions=run_readings(config.model),
            gates=GateState.of(),
            members={member_name("graph", Topology().ranks()[0], "json"): graph},
        )

    lookup = _op_graphs(store.root, config, "gfx950")

    assert lookup("decode", 2, 2, True, 16, verifies=0) == {"op": "decode B=2 V=0"}
    assert lookup("decode", 2, 2, True, 32) == {"op": "decode B=2"}


def test_a_producer_s_plan_measures_its_routed_leaf_at_its_steps_real_rows(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # DP 4 under MTP1, each step's rows one token each: the rung's MoE gathers
    # 16 rows, one a row on each rank, and its steps' MoE ran 5 and 8 real
    # ones of them, as `test_bench_ops_real_rows` plans them with no drafter.
    routed = _routed_graph(4)

    def graph(store, config, shape, arch=None, output=True, draft=False):
        if draft:
            return _mtp(store, config, shape, arch, output, draft)
        return routed(store, config, shape, arch)

    monkeypatch.setattr(ops, "graph", graph)
    monkeypatch.setattr(ops, "_child", _routed_bench)
    spec = SimpleNamespace(num_speculative_tokens=1, use_dspark=lambda: False)
    parallel = SimpleNamespace(data_parallel_size=4)
    config = SimpleNamespace(**vars(PLAN), parallel_config=parallel)
    config.speculative_config = spec
    store, root, key, run = bench
    steps = [([1000, 3000], 5, 0), ([2000] * 3, 9, 2)]
    population = _steps(tmp_path / "plan", 4, *steps)
    args = SimpleNamespace(plan=[population], store=root, llc_mib=256)
    (out := tmp_path / "out").mkdir()

    ops.plan(store, key, config, run, args, "compass measure ops --plan", out)

    records = [json.loads(x) for x in (out / "plan.jsonl").read_text().splitlines()]
    stated = {(r["op"], r["shape"]) for r in records if " R=" in r["shape"]}
    assert stated == {(FUSED, f"decode B=4 V=0 R={n}") for n in (1, 5, 8)}


def test_a_point_s_counts_leave_out_the_drafts_its_rows_verify():
    assert fit_ops.ran_counts("decode B=4 G=16 V=0 R=3", {}) == [{"B": 4, "G": 16}]
