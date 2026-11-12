# SPDX-License-Identifier: MIT
"""`compass measure ops` on DeepSeek-V4.1-Flash's two MI455 cells, off the
card: every leaf of each cell's decode and prefill graphs, traced on card 0's
reading at the cells' engine arguments, taken through `ops.measure` and
`Stub.point` on ATOM's own runner up to its launch (`v41_ops_rehearsal.py`),
each index, slot, pointer and count it takes checked against what it indexes.

V4.1 binds its paged cache on its metadata builder, its leaves take views of
it, of the step its builder prepared, of its decoder's tail and of its
scorer's workspace, and a replaying prefill runs its decoder as passes of its
own: each leaf takes the buffers ATOM's own capture passes it
(`v41_capture_truth.py`).
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import torch
from test_bench_ops import _Runner
from test_capture_mi455_cells import A0, TREE, _trace
from v41_ops_rehearsal import ATEN_INDEX
from test_capture_v41_flash import ARGS, EP4
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture

from atom.compass.bench import ops
from atom.utils import forward_context as fctx

HARNESS = pathlib.Path(__file__).with_name("v41_ops_rehearsal.py")
#: The cells' drafter.
DRAFTER = " --method dspark --num-speculative-tokens 5"
DRAFTER += " --spec-decode-acceptance-length 3"
#: The cells' engine arguments past the model and TP, as their card run starts
#: them.
CELL = f"{ARGS}{DRAFTER}" + (
    " --gpu-memory-utilization 0.9"
    " --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384"
    " --block-size 16 --max-num-seqs 64 --max-model-len 262144"
    " --cudagraph-capture-sizes [1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64]"
)
SHAPES = "--width 16 --prefill 2048 1"
READ = "from atom.compass.capture import driverless\n"
READ += f"driverless.card = {{0: {A0!r}}}.get\n"
TRUTH = pathlib.Path(__file__).with_name("v41_capture_truth.py")
#: A prefill whose forward runs the rows of its decoder's tail: its tail
#: holds the whole prompt, a window of it.
WINDOW = "--prefill 128 1"
#: The leaves whose logits plane the capture's prefill, which caches none,
#: sized to its own rows' KV blocks: the indexer's scorer, its top-k and the
#: candidate pick's block maxima. Past 8192 cached tokens their row bounds
#: reach past those columns, and `measure ops` widens the plane to the point's.
PLANE_BOUND = {
    "aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_logits_gfx1250:"
    "flydsl_fp8_paged_mqa_logits_gfx1250",
    "aiter.ops.flydsl.topk.topk_per_row:flydsl_radix_topk_one_block",
    "atom.model_ops.deepseek_v41.indexer:_block_maxima",
}
#: V4.1's RoPE, over a query's heads or a ratio's index keys, one head.
ROPE = "aiter.rope_cached_positions_fwd_impl.default"
#: V4.1's read of its state cursors back to the host, `.cpu()`, a copy into
#: pageable memory, which no CUDA graph captures: `measure ops` refuses it.
HOST_COPY = "aten.to.dtype_layout"
#: What V4.1's step holds on the host and passes a leaf, by operator and slot:
#: the source of each copy of its staged cursors and slots, and a fill's value
#: that is a Python number.
HOSTED = {("aten.copy_.default", "src"), ("aten.fill_.Tensor", "value")}


def _run(script, *argv, cwd=TREE):
    done = subprocess.run(
        [sys.executable, str(script), *map(str, argv)],
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode == 0, done.stderr[-4000:]


@pytest.fixture(scope="module", params=["", EP4], ids=["cell_b", "cell_a"])
def rehearsed(request, model_dir, tmp_path_factory):  # noqa: F811
    """A cell's graphs traced into a store; what its own capture passes each
    leaf (`v41_capture_truth.py`); and the rehearsal's record of each leaf at
    a decode of 8192 tokens a row and a prefill past as many cached, and of
    each prefill leaf, and `WINDOW`'s, on no cached tokens."""
    root, ep = tmp_path_factory.mktemp("v41_rehearsal"), request.param
    traced = f"{SHAPES} {WINDOW}{CELL}{ep}"
    _trace(root, traced, ("--arch", "gfx1250", "--gpu", "0"), READ)
    model, cell = root / "model", (CELL + ep).split()[1:]
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(model_dir / name, model / name)
    _run(TRUTH, root / "truth.json", model, 4, *SHAPES.split(), "--", *cell)
    atom = ["--model", model, "-tp", "4", "--enforce-eager", "--level", "0", *cell]
    store = ["--llc-mib", "512", "--store", root / "store"]
    points = {"cached": [*SHAPES.split(), "--context", "8192"]}
    points["fresh"] = [*SHAPES.split()[2:], *WINDOW.split()]
    got = {}
    for name, shapes in points.items():
        _run(HARNESS, root / f"{name}.json", *shapes, *store, "--", *atom)
        got[name] = json.loads((root / f"{name}.json").read_text())
    return got, json.loads((root / "truth.json").read_text())


def test_every_leaf_reaches_its_launch_but_a_host_copy(rehearsed):
    """Each leaf's call, Triton's typing of each argument it launches with
    included, reaches its launch, each row bound in its plane, the indexer's
    at a prefill past cached tokens too; but the cursors' copy to the host,
    which `measure ops` refuses by name, at every point."""
    got, _ = rehearsed
    cached = {r["shape"] for r in got["cached"]}
    assert cached == {"decode B=16", "prefill T=2048 B=1"}
    fresh = [r for r in got["fresh"] if r["shape"] != "prefill T=128 B=1"]
    refused = [r for r in got["cached"] + fresh if r["verdict"] != "reaches launch"]
    assert refused and {r["op"] for r in refused} == {HOST_COPY}
    assert all("no CUDA graph captures" in r["why"] for r in refused)
    past = {r["op"] for r in got["cached"] if r["shape"] == "prefill T=2048 B=1"}
    assert PLANE_BOUND <= past
    triton = [
        r for r in got["cached"] if any(k.startswith("triton.") for k in r["kernels"])
    ]
    assert triton


def test_an_argument_atom_holds_on_the_host_is_built_there_bound_to_no_buffer(
    rehearsed,
):
    """The capture states on the host what ATOM holds there, and nothing else;
    each such argument is built on the host, pinned: no buffer of ATOM's binds
    it, and its leaf reaches its launch."""
    from atom.compass.bench import ops

    got, _ = rehearsed
    hosted = [
        (r, path)
        for r in got["cached"]
        for path, spec in ops._specs(json.loads(r["leaf"])["args"])
        if spec.get("device") == "cpu"
    ]
    assert {(r["op"], path) for r, path in hosted} == HOSTED
    for r, path in hosted:
        why = r["shape"], r["op"], path, r["bound"], r["why"]
        assert path not in {*r["bound"], *r["identity"]}, why
        assert r["verdict"] == "reaches launch", why


def test_a_prefill_whose_forward_runs_its_tail_s_rows_refuses_by_name(rehearsed):
    """Its leaves' rows cannot say which pass each runs, so none is bound."""
    got, _ = rehearsed
    window = [r for r in got["fresh"] if r["shape"] == "prefill T=128 B=1"]
    assert window and all("a leaf's rows name no pass" in r["why"] for r in window)


def test_each_leaf_takes_the_buffers_atom_s_capture_passes_it(rehearsed):
    """Each argument a kernel leaf takes of ATOM's metadata, and each integer
    table of its KV, and each index an aten index operator takes, is the
    buffer ATOM's own capture passes every occurrence of that leaf, or each of
    the two it passes its first and later ones (`ops.QUERIED`), by name and
    offset. aten's other leaves index nothing and launch the same work on any
    values."""
    got, truth = rehearsed
    wrong, checked = [], 0
    for r in [*got["cached"], *got["fresh"]]:
        for path, name in r["identity"].items():
            slot = path.split(".")[0]
            if r["op"].startswith("aten.") and (r["op"], slot) not in ATEN_INDEX:
                continue
            if r["shape"] == "prefill T=128 B=1":
                continue
            want = {w for w in truth.get(r["leaf"], {}).get(path, []) if w != "?"}
            checked += bool(want)
            names = {name, r["later"].get(path, name)}
            if want and want != names:
                wrong.append((r["shape"], r["op"], path, names, want))
    assert checked and wrong == []


def test_a_leaf_s_later_occurrences_take_what_atom_passes_them():
    """Copies of a leaf that runs twice alternate its occurrences' buffers:
    the tail's batch ids, then its layer 39's queries'."""
    stub = ops.Stub(_Runner(), 10_000)
    tail = torch.arange(4, dtype=torch.int32)
    stub.placed = [(fctx.ForwardContext(), {("sig", "batch_ids"): ("tail", tail)})]
    stub.later = {(0, "sig", "batch_ids"): ("queries", torch.full_like(tail, -1))}
    spec = {"shape": [4], "stride": [1], "dtype": "torch.int32", "role": "metadata"}
    op = {"op": "scan", "args": {"batch_ids": spec}, "occurrences": 2}

    got = [stub._copy(op, "sig", [{}], c)[0]["batch_ids"] for c in range(4)]

    assert [t.tolist() for t in got] == [[0, 1, 2, 3], [-1] * 4] * 2


def test_each_rope_takes_its_query_s_or_its_ratio_s_key_positions(rehearsed):
    """Each RoPE leaf takes the positions ATOM's capture passes it, at each
    point: the query's and each ratio's keys' apart, a buffer a leaf."""
    got, truth = rehearsed
    for shape in ("decode B=16", "prefill T=2048 B=1"):
        ropes = [r for r in got["cached"] if (r["op"], r["shape"]) == (ROPE, shape)]
        taken = {r["leaf"]: r["identity"].get("positions") for r in ropes}
        want = {s: truth.get(s, {}).get("positions") for s in taken}
        assert {s: [n] for s, n in taken.items()} == want
        assert len(set(taken.values())) == len(taken) >= 3


def test_the_cursor_write_takes_its_staged_cursors_at_its_step_s_slots(rehearsed):
    """V4.1's cursor write, `cursor[step.slots[:n].long()] = staged[:n]`:
    its values are the staged cursors ATOM's capture passes it, and its index,
    a cast no buffer holds, its step's slots widened to int64."""
    got, truth = rehearsed
    cursor = "aten.index_put_.default", "attn_metadata_builder.cache.cursor"
    writes = [r for r in got["cached"] if (r["op"], r["bound"].get("self")) == cursor]
    assert writes
    for r in writes:
        taken = r["identity"]["values"], r["bound"]["indices.0"]
        assert taken == (*truth[r["leaf"]]["values"], "step.slots.long")


@pytest.mark.parametrize("shape", [[4], [2, 4]], ids=["one_row", "two_rows"])
def test_the_cursor_write_s_values_are_its_staging_s_first_rows(shape):
    """One row is the first, its unit dim dropped as `cursor[i] = staged[:1]`
    drops it."""
    staged, slots = torch.zeros(64, 4, dtype=torch.int64), torch.zeros(64).long()
    cache = "attn_metadata_builder.cache"
    held = {f"{cache}._cursor_staging.gpu": staged, f"{cache}._reset_slots": slots}
    spec = {"shape": shape, "stride": [4, 1][-len(shape) :], "dtype": "torch.int64"}
    op = {"op": "aten.index_put_.default", "args": {"self": {"role": "kv"}}}
    _, view = ops._kv(spec, "values", ops._named(op, "values", spec, held) or held)
    assert (view.data_ptr(), list(view.shape)) == (staged.data_ptr(), shape)


@pytest.fixture(scope="module")
def undrafted(model_dir, tmp_path_factory):  # noqa: F811
    """Cell B's decode B=16 with no drafter, traced; what its capture passes
    each leaf; and the rehearsal's record of its RoPE leaves alone, no compress
    leaf beside them, at 8192 tokens a row."""
    root, cell = tmp_path_factory.mktemp("v41_undrafted"), CELL.replace(DRAFTER, "")
    _trace(root, f"--width 16{cell}", ("--arch", "gfx1250", "--gpu", "0"), READ)
    model, cell = root / "model", cell.split()[1:]
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(model_dir / name, model / name)
    _run(TRUTH, root / "truth.json", model, 4, "--width", "16", "--", *cell)
    atom = ["--model", model, "-tp", "4", "--enforce-eager", "--level", "0", *cell]
    point = ["--width", "16", "--context", "8192", "--only", ROPE, "--llc-mib", "512"]
    _run(HARNESS, root / "got.json", *point, "--store", root / "store", "--", *atom)
    return [json.loads((root / f).read_text()) for f in ("got.json", "truth.json")]


def test_each_rope_alone_takes_its_positions_at_a_decode_with_no_drafter(undrafted):
    """With no drafter both ratios' compress plans hold a row a request, so only
    the ratio the capture states tells their keys' RoPEs apart: each RoPE leaf,
    with no compress leaf beside it, takes the positions ATOM's capture passes
    it, the query's and each ratio's keys' apart, a buffer a leaf."""
    got, truth = undrafted
    taken = {r["leaf"]: r["identity"].get("positions") for r in got}
    want = {s: truth.get(s, {}).get("positions") for s in taken}
    assert {s: [n] for s, n in taken.items()} == want
    assert {(r["op"], r["verdict"]) for r in got} == {(ROPE, "reaches launch")}
    assert len(set(taken.values())) == len(taken) == 3
