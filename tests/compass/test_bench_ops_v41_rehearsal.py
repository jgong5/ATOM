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
import re
import shutil
import subprocess
import sys

import pytest
from test_capture_mi455_cells import A0, TREE, _trace
from v41_ops_rehearsal import ATEN_INDEX
from test_capture_v41_flash import ARGS, EP4
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture

HARNESS = pathlib.Path(__file__).with_name("v41_ops_rehearsal.py")
#: The cells' engine arguments past the model and TP, as their card run starts
#: them.
CELL = ARGS + (
    " --method dspark --num-speculative-tokens 5"
    " --spec-decode-acceptance-length 3 --gpu-memory-utilization 0.9"
    " --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384"
    " --block-size 16 --max-num-seqs 64 --max-model-len 262144"
    " --cudagraph-capture-sizes [1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64]"
)
SHAPES = "--width 16 --prefill 2048 1"
READ = "from atom.compass.capture import driverless\n"
READ += f"driverless.card = {{0: {A0!r}}}.get\n"
TRUTH = pathlib.Path(__file__).with_name("v41_capture_truth.py")
#: The slots where ATOM passes the occurrences of one leaf two buffers, of
#: which the bench binds one: a query's RoPE positions and a ratio-1 key's
#: (#2865), and the decoder tail's batch ids and its layer 39's queries'
#: (#2867).
SPLIT = {
    ("aiter.rope_cached_positions_fwd_impl.default", "positions"),
    ("atom.model_ops.attentions.deepseek_v41.indices:_launch_indices", "batch_ids"),
    ("atom.model_ops.attentions.deepseek_v41.indices:_launch_indptr_scan", "batch_ids"),
}
#: A prefill whose forward runs the rows of its decoder's tail: its tail
#: holds the whole prompt, a window of it.
WINDOW = "--prefill 128 1"
#: The leaves the rehearsal refuses at a prefill past 8192 cached tokens, each
#: reading rows of a logits plane past the columns the capture's prefill,
#: which caches none, sized it to: the indexer's scorer, its top-k and the
#: candidate pick's block maxima.
PLANE_BOUND = {
    "aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_logits_gfx1250:"
    "flydsl_fp8_paged_mqa_logits_gfx1250",
    "aiter.ops.flydsl.topk.topk_per_row:flydsl_radix_topk_one_block",
    "atom.model_ops.deepseek_v41.indexer:_block_maxima",
}
#: V4.1's read of its state cursors back to the host, `.cpu()`, a copy into
#: pageable memory, which no CUDA graph captures: `measure ops` refuses it.
HOST_COPY = "aten.to.dtype_layout"


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


def test_every_leaf_reaches_its_launch_but_a_plane_too_short_or_a_host_copy(
    rehearsed,
):
    """Each leaf's call, Triton's typing of each argument it launches with
    included, reaches its launch, but at a prefill past cached tokens the
    indexer's row bounds past the plane the capture's prefill, which caches
    none, sized, which the value check refuses by name; and the cursors' copy
    to the host, which `measure ops` refuses by name, at every point."""
    got, _ = rehearsed
    cached = {r["shape"] for r in got["cached"]}
    assert cached == {"decode B=16", "prefill T=2048 B=1"}
    fresh = [r for r in got["fresh"] if r["shape"] != "prefill T=128 B=1"]
    refused = [r for r in got["cached"] + fresh if r["verdict"] != "reaches launch"]
    host = [r for r in refused if r["op"] == HOST_COPY]
    assert host and all("pageable host memory" in r["why"] for r in host)
    short = [r for r in refused if r["op"] != HOST_COPY]
    assert short and {(r["shape"], r["op"]) for r in short} <= {
        ("prefill T=2048 B=1", op) for op in PLANE_BOUND
    }
    assert all(re.search(r"past 0 to \d+$", r["why"]) for r in short)
    assert not {id(r) for r in short} & {id(r) for r in fresh}
    triton = [
        r for r in got["cached"] if any(k.startswith("triton.") for k in r["kernels"])
    ]
    assert triton


def test_a_prefill_whose_forward_runs_its_tail_s_rows_refuses_by_name(rehearsed):
    """Its leaves' rows cannot say which pass each runs, so none is bound."""
    got, _ = rehearsed
    window = [r for r in got["fresh"] if r["shape"] == "prefill T=128 B=1"]
    assert window and all("a leaf's rows name no pass" in r["why"] for r in window)


def test_each_leaf_takes_the_buffers_atom_s_capture_passes_it(rehearsed):
    """Each argument a kernel leaf takes of ATOM's metadata, and each integer
    table of its KV, and each index an aten index operator takes, is the one
    buffer ATOM's own capture passes every occurrence of that leaf, by name and
    offset, but `SPLIT`'s. aten's other leaves index nothing and launch the
    same work on any values."""
    got, truth = rehearsed
    wrong, checked = [], 0
    for r in [*got["cached"], *got["fresh"]]:
        for path, name in r["identity"].items():
            slot = path.split(".")[0]
            if r["op"].startswith("aten.") and (r["op"], slot) not in ATEN_INDEX:
                continue
            if (r["op"], slot) in SPLIT or r["shape"] == "prefill T=128 B=1":
                continue
            want = {w for w in truth.get(r["leaf"], {}).get(path, []) if w != "?"}
            checked += bool(want)
            if want and want != {name}:
                wrong.append((r["shape"], r["op"], path, name, want))
    assert checked and wrong == []
