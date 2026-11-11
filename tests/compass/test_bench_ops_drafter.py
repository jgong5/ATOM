# SPDX-License-Identifier: MIT
"""`compass measure ops` on a speculative drafter's graphs, with no device.

With the stand-in bench and graphs of `test_bench_ops_plan.py`, each part of a
drafter's step holds a leaf of its own: `--plan` measures them beside the
target's leaves, a padded decode's and a prefill's drafter at its own shape
too, and `--coverage` reads each step's parts as a run prices them. Beside the
stand-in runner of `test_bench_ops.py`, a leaf of the drafter's
alone counts the rows of its own pass as ATOM's `batch_info` states them and
draws its expert ids over its drafter's experts. DeepSeek-V4.1-Flash's
DSpark5 drafter, traced at #2086's trace run b's arguments where a capture
holds them, reaches each leaf's launch off the card (`v41_ops_rehearsal.py`).
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops import TREE, _graph, _leaf, _Runner, _t
from test_bench_ops_plan import (
    CONFIG,
    _bench,
    _graph_at,
    bench,  # noqa: F401  a fixture
)
from test_bench_ops_ranks import _sig
from test_capture_ir import MAIN
from test_capture_v41_flash import CONFIG as V41
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture

from atom.compass.artifacts import ArtifactRefusal, Rule
from atom.compass.backends.op_graph import KV
from atom.compass.bench import ops
from atom.compass.capture.forward import decode, prefill
from atom.compass.fit import ops as fit_ops
from atom.utils import forward_context as fctx

#: A block drafter of one draft: a decode row verifies two tokens.
DRAFTING = SimpleNamespace(**vars(CONFIG))
DRAFTING.speculative_config = SimpleNamespace(
    num_speculative_tokens=1, use_dspark=lambda: True
)
#: Each part's own leaf: its propose, and its draft-KV pass, which is a middle
#: chunk's drafter's part too.
OWN = {(True, True): "aiter.propose", (True, KV): "aiter.write_kv"}
OWN[False, True] = OWN[True, KV]


def _parted(store, config, shape, arch=None, output=True, draft=False):
    """The target's graph, or a part of the drafter's, as `trace ops` keeps
    them: a decode has no middle chunk's, and one padded past its rows its
    drafter's alone. A decode's drafter runs its block at the step's width, B,
    and stages its anchors at its rows, T, and past them cuts its rows from
    the width's, as DeepSeek-V4.1's DSpark drafter does."""
    padded = shape.phase == "decode" and shape.tokens < shape.seqs
    if shape.phase == "decode" and not output or padded and draft is not True:
        raise ArtifactRefusal(Rule.RESOLUTION, f"no op_graph holds {shape}", "")
    if not draft:
        return _graph_at(store, config, shape, arch)
    rows = shape.tokens if shape.phase == "prefill" else shape.seqs
    own = [_leaf(OWN[output, draft], ("x", _t([rows, 8], "activation")))]
    if shape.phase == "decode" and draft is True:
        own.append(_anchor(shape.tokens))
    if padded:
        own.append(_leaf("aiter.cut", ("x", _t([shape.seqs, 3], "activation"))))
    return _graph(*own, layers=1)


def _records(path, *steps):
    """A record directory of `steps`, each `(rows, rung, produces_output)`."""
    (path / "steps").mkdir(parents=True)
    (path / "run.json").write_text("{}")
    lines = [
        {"step": i, "is_dummy_run": False, "rows": rows, "rung": rung}
        | {"produces_output": output}
        for i, (rows, rung, output) in enumerate(steps)
    ]
    (path / "steps" / "dp0.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    return str(path)


def _moving_anchor_bench(fn, config, todo, llc):
    """`_bench`, the drafter's anchors timed as its attention is, by their
    longest row's context, their kernel switching at 3500 tokens: past the
    ragged step's rows, short of the padded step's one row."""
    got = _bench(fn, config, todo, llc)
    for r in got:
        if r["op"] == "aiter.anchor":
            n = ops._split(r["context"])[0]
            r.update(kernels=[f"anchor{int(n >= 3500)}"], cold_ns=n * 1e3)
    return got


def test_a_plan_measures_a_drafter_s_parts_and_coverage_reads_each_step_s(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(ops, "graph", _parted)
    monkeypatch.setattr(ops, "_child", _moving_anchor_bench)
    steps = _records(
        tmp_path / "steps",
        ([[2, 1000, True], [2, 3000, True]], 2, True),
        ([[2, 4000, True]], 2, True),  # one row padded to two
        ([[2, 12000, True]], 1, True),
        ([[40, 40, False]], None, True),  # a final chunk
        ([[150, 150, False]], None, False),  # a middle chunk
    )
    store, root, key, run = bench
    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(plan=[steps], store=root, llc_mib=256)

    ops.plan(store, key, DRAFTING, run, args, "compass measure ops --plan", out)

    records = [json.loads(x) for x in (out / "plan.jsonl").read_text().splitlines()]
    shapes = {r["shape"] for r in records if r["op"] == "aiter.propose"}
    assert shapes == {r["shape"] for r in records if r["op"] == "aiter.gemm"}
    # The decode padded past its rows runs its drafter at `decode T=1 B=2`: its
    # block is the width's, measured at rung 2, its anchors and cut points there.
    rungs = {"decode B=1", "decode B=2", "decode B=4"}
    anchors = {r["shape"] for r in records if r["op"] == "aiter.anchor"}
    assert anchors == rungs | {"decode T=1 B=2"} and rungs <= shapes
    assert {r["shape"] for r in records if r["op"] == "aiter.cut"} == {"decode T=1 B=2"}
    # The anchors move with context: a shape of more than one row takes ragged
    # points of them, one of one row none, its row being its rows' mean.
    ragged = {
        r["shape"]
        for r in records
        if r["op"] == "aiter.anchor" and isinstance(r["context"], list)
    }
    assert ragged == {"decode B=2", "decode B=4"}
    assert {r["op"] for r in records} == {
        "aiter.attn",
        "aiter.gemm",
        "aiter.anchor",
        "aiter.cut",
        *OWN.values(),
    }
    report = ops.coverage(store, root, DRAFTING, [steps], records, "gfx942")
    # Each step reads its target's and its drafter's graphs; one that samples
    # also its draft-KV pass's; a middle chunk its drafter's alone. The padded
    # decode's drafter is read at its own shape, each leaf at the rows it ran:
    # its cut, measured there alone, at its one row, and its anchors at that
    # row's context, past their kernel's switch, which the rung's two rows, a
    # pad among them, would reach back across.
    assert report["by part"] == {
        "target": {"steps": 5, "covered": 5},
        "draft": {"steps": 5, "covered": 5},
        "draft_kv": {"steps": 4, "covered": 4},
    }
    assert report["left out, by leaf kind"] == {}


def _anchor(rows):
    """The leaf staging a drafter's anchors, of `rows` rows."""
    return _leaf("aiter.anchor", ("ids", _t([rows, 6], "activation")))


def test_fit_ops_reads_a_drafter_s_own_point_at_the_rows_it_ran():
    # One row padded to two ran the anchors of its one row: a point of B=1,
    # beside the drafter's at B=1, not of B=2.
    leaf = {}

    def point(shape, rows, ns):
        [(sig, (op, _))] = ops.leaves(_graph(_anchor(rows), layers=1)["region"]).items()
        leaf[rows] = op
        point = {"shape": shape, "context": 1000, "op": "aiter.anchor", "leaf": sig}
        return point | {"cold_ns": ns, "kernels": ["anchor"]}

    points = [
        point("decode B=1", 1, 1000.0),
        point("decode T=1 B=2", 1, 1200.0),
        point("decode B=2", 2, 2000.0),
    ]
    listed = fit_ops.PriceList(fit_ops.fit(points))

    got = [
        listed.price((leaf[n], ("decode", n, n, None, 1000, 1000.0), None)).seconds
        for n in (1, 2)
    ]

    assert got == pytest.approx([1100e-9, 2000e-9])


#: An MTP drafter of two drafts, which runs its later passes at its step's
#: batch: past a prefill's rows where the ladder or a DP peer pads it.
MTP = SimpleNamespace(**vars(CONFIG))
MTP.speculative_config = SimpleNamespace(
    num_speculative_tokens=2, use_dspark=lambda: False
)


def _mtp(store, config, shape, arch=None, output=True, draft=False):
    """The target's graph, or its MTP drafter's, as `trace ops` keeps them:
    no draft-KV pass, and a decode's that samples alone. The target's LM head,
    the drafter's later pass and, on a step that samples, its sampler run a
    row a sequence at the batch the step pads to, its W where it states one;
    the drafter's first pass at the step's rows."""
    if draft == KV or shape.phase == "decode" and not output:
        raise ArtifactRefusal(Rule.RESOLUTION, f"no op_graph holds {shape}", "")
    width = ("x", _t([shape.width or shape.seqs, 8], "activation"))
    if not draft:
        target = _graph_at(store, config, shape, arch)
        head = _graph(_leaf("aiter.head", width), layers=1)["region"]
        return target | {"region": {"seq": [target["region"], head]}}
    rows = shape.tokens if shape.phase == "prefill" else shape.seqs
    first = _leaf("aiter.propose", ("x", _t([rows, 8], "activation")))
    sample = [_leaf("aiter.sample", width)] if output else []
    return _graph(first, _leaf("aiter.step", width), *sample, layers=1)


def _ranks(path, *ranks):
    """A record directory of a DP rank's lines each of `ranks`, a step each."""
    (path / "steps").mkdir(parents=True)
    (path / "run.json").write_text("{}")
    for rank, lines in enumerate(ranks):
        steps = [
            {"step": i, "is_dummy_run": False, "rung": None} | line
            for i, line in enumerate(lines)
        ]
        text = "".join(json.dumps(line) + "\n" for line in steps)
        (path / "steps" / f"dp{rank}.jsonl").write_text(text)
    return str(path)


#: Beside an idle DP peer, each padded to 4 for its drafter's later pass: a
#: prompt of one row that samples, its LM head at its row, and a middle chunk.
PROMPT = {"rows": [[40, 40, False]], "produces_output": True, "draft_bs": 4}
CHUNK = PROMPT | {"rows": [[60, 60, False]], "produces_output": False}


def _plan_mtp(bench, tmp_path, monkeypatch, *steps):  # noqa: F811
    """`--plan` under `MTP`, its graphs `_mtp`'s, of a decode and `steps` on DP
    rank 0 beside an idle rank 1: each `(shape, op)` of a point at a shape that
    states W, and `--coverage`'s report of the steps."""
    monkeypatch.setattr(ops, "graph", _mtp)
    decode = {"rows": [[3, 1000, True], [3, 3000, True]], "rung": 2}
    idle = [{"is_dummy_run": True}]
    record = _ranks(tmp_path / "steps", [decode, *steps], idle)
    store, root, key, run = bench
    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(plan=[record], store=root, llc_mib=256)

    ops.plan(store, key, MTP, run, args, "compass measure ops --plan", out)

    records = [json.loads(x) for x in (out / "plan.jsonl").read_text().splitlines()]
    wide = {(r["shape"], r["op"]) for r in records if " W=" in r["shape"]}
    return wide, ops.coverage(store, root, MTP, [record], records, "gfx942")


def test_a_plan_measures_a_drafter_s_own_prefill_shape_that_coverage_reads(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # Past the step's graphs, the drafter's at its own shape holds its later
    # pass alone, and of the prompt, which samples, its sampler.
    wide, report = _plan_mtp(bench, tmp_path, monkeypatch, PROMPT, CHUNK)

    assert wide == {
        ("prefill T=40 B=1 W=4", "aiter.step"),
        ("prefill T=40 B=1 W=4", "aiter.sample"),
        ("prefill T=60 B=1 W=4", "aiter.step"),
    }
    assert report["by part"] == {
        "target": {"steps": 3, "covered": 3},
        "draft": {"steps": 3, "covered": 3},
    }
    assert report["left out, by leaf kind"] == {}


def test_a_drafter_s_own_prefill_shape_that_is_a_step_s_is_measured_whole(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # The middle chunk's drafter runs at the shape of a prompt of its tokens
    # whose LM head pads too: the prompt's head is measured there.
    padded = CHUNK | {"produces_output": True, "target_bs": 4}

    wide, report = _plan_mtp(bench, tmp_path, monkeypatch, CHUNK, padded)

    assert wide == {
        ("prefill T=60 B=1 W=4", "aiter.step"),
        ("prefill T=60 B=1 W=4", "aiter.sample"),
        ("prefill T=60 B=1 W=4", "aiter.head"),
    }
    assert report["by part"] == {
        "target": {"steps": 3, "covered": 3},
        "draft": {"steps": 3, "covered": 3},
    }


def test_coverage_asks_a_padded_prompt_s_leaves_at_its_width_as_a_run_does(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # Prompts of one row padded to 4 and to 8, and none at its row: the LM
    # head, the drafter's later pass and its sampler hold points at W alone.
    padded = [
        PROMPT | {"rows": [[t, t, False]], "target_bs": w, "draft_bs": w}
        for t, w in ((40, 4), (60, 8))
    ]

    _, report = _plan_mtp(bench, tmp_path, monkeypatch, *padded)

    assert report["by part"] == {
        "target": {"steps": 3, "covered": 3},
        "draft": {"steps": 3, "covered": 3},
    }
    assert report["left out, by leaf kind"] == {}


def test_coverage_leaves_out_a_prompt_whose_t_is_its_w_where_a_run_declines_it(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    # Beside prompts of one row padded to 4 and to 8, one of 4 tokens padded to
    # 4, whose leaves lead with its T and its W alike: its LM head, drafter's
    # later pass and sampler, whose families read B, ran at 1 row or 4; its
    # attention, whose point there the fit left out, is below its measured T
    # at either; its GEMM and first pass read T alone and stay covered
    padded = [
        PROMPT | {"rows": [[t, t, False]], "target_bs": w, "draft_bs": w}
        for t, w in ((40, 4), (60, 8), (4, 4))
    ]

    _, report = _plan_mtp(bench, tmp_path, monkeypatch, *padded)

    records = (tmp_path / "out" / "plan.jsonl").read_text().splitlines()
    fitted = fit_ops.fit([r for r in map(json.loads, records) if "refused" not in r])
    left = {(p["shape"], p["op"]) for p in fitted["left_out"]}
    assert left == {
        ("prefill T=4 B=1 W=4", f"aiter.{op}")
        for op in ("attn", "head", "step", "sample")
    }
    why = "it leads with its T, a count of its width too, so it ran at "
    why += "T=# B=# or T=# B=#"
    assert report["by part"] == {
        "target": {"steps": 4, "covered": 3},
        "draft": {"steps": 4, "covered": 3},
    }
    assert report["left out, by leaf kind"] == {
        "aiter.attn": {"T = # is outside the measured # to #": 1},
        "aiter.head": {why: 1},
        "draft aiter.sample": {why: 1},
        "draft aiter.step": {why: 1},
    }


class _Padding(_Runner):
    """`_Runner` keeping, for each batch it prepares, the ladder its step pads
    to, ATOM's under `--enforce-eager` unless the bench pads it."""

    capture_sizes_np = np.array([0], dtype=np.int32)

    def prepare_model(self, batch):
        self.ladders.add(tuple(self.capture_sizes_np.tolist()))
        return super().prepare_model(batch)


def test_a_drafter_s_own_shape_holds_its_rows_on_a_step_padded_to_its_width(
    monkeypatch,
):
    # A decode of one row padded to two, beside one of two rows: the point
    # holds the row, verifying its draft, and the runner pads its step to
    # two, as the capture's runner did, so the drafter's passes run there.
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    monkeypatch.setattr(ops.Stub, "_time", lambda *_: {"cold_ns": 1.0})
    runner, got = _Padding(), {}
    batches = []
    rule = SimpleNamespace(
        block_size=16,
        drafts=1,
        batch=lambda view, block, slot: batches.append(view["rows"]) or (block, slot),
    )
    stub = ops.Stub(runner, 10_000)
    found = ops.leaves(_graph(_anchor(1), layers=1)["region"])
    for shape in (decode(2, None, 1), decode(2)):
        runner.ladders, batches[:] = set(), []
        stub.point(rule, shape, 20, found)
        got[str(shape)] = batches[0], runner.ladders

    assert got == {
        "decode T=1 B=2": ([[2, 21, True]], {(2,)}),
        "decode B=2": ([[2, 21, True], [2, 21, True]], {(0,)}),
    }


def test_a_prefill_that_states_its_width_is_prepared_on_a_step_padded_to_it(
    monkeypatch,
):
    # A prompt of one row whose drafter's later pass ran at 4: the runner pads
    # the point's step to 4, as the capture's runner did; one that states no
    # width runs at its rows.
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})
    monkeypatch.setattr(ops.Stub, "_time", lambda *_: {"cold_ns": 1.0})
    runner, got = _Padding(), {}
    rule = SimpleNamespace(block_size=16, batch=lambda view, block, slot: (block, slot))
    stub = ops.Stub(runner, 10_000)
    later = _leaf("aiter.step", ("x", _t([4, 8], "activation")))
    found = ops.leaves(_graph(later, layers=1)["region"])
    for shape in (prefill(40, 1, None, 4), prefill(40, 1)):
        runner.ladders = set()
        stub.point(rule, shape, None, found)
        got[str(shape)] = runner.ladders

    assert got == {"prefill T=40 B=1 W=4": {(4,)}, "prefill T=40 B=1": {(0,)}}


class _Drafting(_Runner):
    """`_Runner` whose model routes 6 of 384 experts, whose drafter's routes 3
    of 128, and whose step states its rows as ATOM's `batch_info` does: 4
    sequences of 6 tokens, 20 drafted."""

    def __init__(self):
        super().__init__()
        self.model.add_module("moe", _Moe(384, 6))
        self.drafter = SimpleNamespace(model=_Moe(128, 3))
        info = torch.tensor([4, 24, 20], dtype=torch.int32)
        self.forward_vars["batch_info"] = SimpleNamespace(np=info.numpy(), gpu=info)

    def prepare_model(self, batch):
        prepared = super().prepare_model(batch)
        context = fctx.get_forward_context().context
        context.running_tokens = 24
        context.scheduled_tokens_gpu = self.forward_vars["batch_info"].gpu[1:2]
        return prepared


class _Moe(torch.nn.Module):
    def __init__(self, experts, k):
        super().__init__()
        self.global_num_experts, self.top_k = experts, k


def test_a_drafter_s_pass_of_its_own_rows_counts_them_as_batch_info_states():
    stub = ops.Stub(_Drafting(), 10_000)
    count = ("num_local_tokens", _t([1], "metadata", dtype="torch.int32"))
    moe = {
        rows: _leaf(
            "aiter.fused_moe_.default", ("x", _t([rows, 8], "activation")), count
        )
        for rows in (24, 20)
    }
    stub.leaves = ops.leaves(_graph(*moe.values(), layers=1)["region"])
    sigs = {op["args"]["x"]["shape"][0]: sig for sig, (op, _) in stub.leaves.items()}
    stub.leaves[sigs[20]][0]["drafted"] = True

    stub._prepare((0, 0))

    bound = {sig: view for (sig, _), (_, view) in stub.placed[0][1].items()}
    # the drafted rows' count, not the target's tokens
    assert {rows: bound[sig].item() for rows, sig in sigs.items()} == {24: 24, 20: 20}


def test_a_placement_keeps_the_clone_of_a_view_whose_nans_stand_unchanged():
    # A drafter's draft-KV pass reads ATOM's uninitialised `outputs`: by value
    # its NaNs never equal their clone's, and each placement cloned it again.
    runner = _Runner()
    runner.forward_vars["outputs"] = torch.full((6,), float("nan"))
    stub = ops.Stub(runner, 10_000)
    leaf = _leaf("aten.cat.default", ("tensors", (_t([6], "metadata"),)))
    stub.leaves = ops.leaves(_graph(leaf, layers=1)["region"])

    for p in range(2):
        stub._prepare((4 * p, 2 * p))

    first, second = ([v for _, v in views.values()] for _, views in stub.placed)
    assert first[0] is second[0]


def test_a_collective_whose_bytes_fall_as_its_step_s_rows_rise_halves_no_step():
    # The DSpark drafter's all-reduce of its one anchor at a prefill of 8192
    # tokens, beside its block's of 5 rows at a decode of 1: no step between.
    fam = ops.exchange(json.loads(_sig(1)))[0]
    points = [
        (10240, ("one",), {"shape": str(prefill(8192, 1)), "context": None}),
        (51200, ("two",), {"shape": str(decode(1)), "context": 1024}),
    ]

    assert ops._densify({fam: points}, set()) == {}


def test_a_drafter_s_moe_draws_its_ids_over_its_own_experts():
    # The model's MoE leaf draws over its 384 experts, the drafter's over its
    # 128; a leaf of the model's whose integer view is as wide as the
    # drafter's top-k, as V4.1's target's at a decode of 3 rows, draws none.
    stub = ops.Stub(_Drafting(), 10_000)
    stub.placed = [(fctx.ForwardContext(), {})]
    moe = "aiter.fused_moe_.default"
    for op, rows, k, drafted in (
        (moe, 24, 6, False),
        (moe, 20, 3, True),
        ("aten.view.default", 24, 3, False),
    ):
        ids = {"shape": [rows, k], "stride": [k, 1], "dtype": "torch.int32"}
        args = {"ids": ids | {"role": "activation"}}
        stub._copy({"op": op, "args": args, "drafted": drafted}, "sig", [{}], 0)

    assert [len(loads) for loads in stub.loads] == [384, 128]


#: Trace run b's engine arguments past the model and TP, in the setup a capture
#: holds: a BF16 KV cache, an FP8 index plane, and V4.1's mono decode off.
CELL = (
    "--kv_cache_dtype bf16 --index-cache-dtype fp8 --block-size 16"
    " --max-num-seqs 128 --max-num-batched-tokens 16384"
    " --attn-prefill-chunk-size 16384 --enable_prefix_caching"
    " --state-checkpoint-interval-tokens 8192 --long-prefill-token-threshold 8192"
    " --method dspark --num-speculative-tokens 5 --spec-decode-acceptance-length 3.51"
)
SHAPES = ["--width", "4", "--prefill", "124", "1"]
#: Kernels only the drafter launches: its draft-KV pass's tail, its block's
#: attention and its Markov sampler.
OWN_KERNELS = {
    "atom.model_ops.deepseek_v41.dspark:fused_draft_kv_tail",
    "atom.model_ops.sparse_attn_v4:_sparse_attn_triton",
    "atom.model_ops.dspark_markov_sample:_launch",
}


def _run(*argv):
    env = {**os.environ, "PYTHONPATH": str(TREE), "ATOM_MONO_ENABLE": "0"}
    done = subprocess.run(
        [sys.executable, *map(str, argv)],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        cwd=TREE,
        env=env,
    )
    assert done.returncode == 0, done.stderr[-4000:]


@pytest.fixture(scope="module")
def rehearsed(model_dir, tmp_path_factory):  # noqa: F811
    """Each leaf of the drafter's graphs alone at a decode of 4 rows of 1024
    tokens and a prefill of 124 past as many, traced at TP2 on gfx950, as
    the rehearsal records it."""
    root = tmp_path_factory.mktemp("drafter")
    (model := root / "model").mkdir()
    shutil.copy(V41, model / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(model_dir / name, model / name)
    store, tp = root / "store", ["--tp", "2"]
    trace = ["trace", "ops", "--model", model, *tp, "--arch", "gfx950"]
    _run("-c", MAIN, *trace, "--store", store, *SHAPES, "--", *CELL.split())
    harness = pathlib.Path(__file__).with_name("v41_ops_rehearsal.py")
    point = [*SHAPES, "--context", "1024", "--llc-mib", "256", "--store", store]
    atom = ["--model", model, "-tp", "2", "--enforce-eager", "--level", "0"]
    flags = ["--drafter", "--arch", "gfx950"]
    _run(harness, root / "drafter.json", *flags, *point, "--", *atom, *CELL.split())
    return json.loads((root / "drafter.json").read_text())


def test_each_leaf_of_a_drafter_s_graphs_reaches_its_launch(rehearsed):
    """Each leaf of the drafter's propose and draft-KV pass, and of a middle
    chunk's drafter, at both shapes, its kernels its own among them, each
    index, slot and count it binds inside what it indexes."""
    assert {r["shape"] for r in rehearsed} == {"decode B=4", "prefill T=124 B=1"}
    assert OWN_KERNELS <= {r["op"] for r in rehearsed}
    assert [r for r in rehearsed if r["verdict"] != "reaches launch"] == []
