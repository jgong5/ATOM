# SPDX-License-Identifier: MIT
"""`compass measure ops` on DeepSeek-V4.1-Flash's two MI455 cells, off the
card: every leaf of each cell's decode and prefill graphs, traced on card 0's
reading at the cells' engine arguments, taken through `ops.measure` and
`Stub.point` on ATOM's own runner up to its launch (`v41_ops_rehearsal.py`).

V4.1 binds its paged cache on its metadata builder, and its leaves take views
of it, of the step its builder prepared and of its scorer's workspace, each
table at the leaf's compress ratio: every leaf reaches its launch, on the
buffers ATOM passes it.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
from test_capture_mi455_cells import A0, TREE, _trace
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
INDICES = "atom.model_ops.attentions.deepseek_v41.indices:"
SCORER = (
    "aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_logits_gfx1250:"
    "flydsl_fp8_paged_mqa_logits_gfx1250"
)
TOPK = "aiter.ops.flydsl.topk.topk_per_row:flydsl_radix_topk_one_block"
CACHE = "attn_metadata_builder.cache."
#: The compressor's leaves, which ATOM passes its kv and score state fields.
STATE = (
    "atom.model_ops.v4_kernels.state_writes:update_compressor_states",
    "atom.model_ops.v4_kernels.fused_compress:_launch_fused_compress",
)
#: The MoE's slots of its valid rows, which ATOM passes the step's token count.
MOE = {
    "aiter.fused_moe_.default": "num_local_tokens",
    "aiter.topk_gating_fwd.default": "num_valid_rows",
}
#: V4.1-Flash's compress ratios by the rows its 128-token page holds at each.
PAGE_ROWS = {64: 2, 128: 1}


@pytest.fixture(scope="module", params=["", EP4], ids=["cell_b", "cell_a"])
def rehearsed(request, model_dir, tmp_path_factory):  # noqa: F811
    """A cell's graphs traced into a store, and the rehearsal's record of
    each of their leaves at `--context 8192`."""
    root, ep = tmp_path_factory.mktemp("v41_rehearsal"), request.param
    _trace(root, SHAPES + CELL + ep, ("--arch", "gfx1250", "--gpu", "0"), READ)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(model_dir / name, root / "model" / name)
    atom = ["--model", str(root / "model"), "-tp", "4", "--enforce-eager"]
    atom += ["--level", "0", *(CELL + ep).split()[1:]]
    bench = [*SHAPES.split(), "--context", "8192", "--llc-mib", "512"]
    out = root / "records.json"
    done = subprocess.run(
        [sys.executable, str(HARNESS), str(out), *bench]
        + ["--store", str(root / "store"), "--", *atom],
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(out.read_text())


def test_every_leaf_of_both_shapes_reaches_its_launch(rehearsed):
    assert {r["shape"] for r in rehearsed} == {"decode B=16", "prefill T=2048 B=1"}
    assert [r for r in rehearsed if r["verdict"] != "reaches launch"] == []


def _ratio(leaf, slot):
    """The compress ratio ATOM runs `leaf` at for `slot`, read off its contract."""
    args = leaf["args"]
    if slot[-1].isdigit():  # the indptr scan's i-th table is its i-th ratio's
        return args["ratios"][int(slot.split(".")[-1])]
    if "rows_per_page" in args:  # the index build: no selection, no ratio
        return PAGE_ROWS[args["rows_per_page"]] if args["selected"] else 0
    return args.get("ratio", leaf.get("stated", {}).get("compress_ratio"))


def _indptr(leaf, slot):
    """The prefix or extend indptr of its ratio ATOM passes `leaf` as `slot`."""
    extend = slot.startswith("e") or slot.endswith("extend")
    return f"indptr_buffers.{_ratio(leaf, slot)}.{int(extend)}"


def test_a_v41_leaf_takes_the_buffer_atom_passes_it(rehearsed):
    """ATOM's V4.1 forward passes each leaf its ratio's indptr pair, which
    the scan fills (`fill_step_indptrs`), its scorer's logits plane, tile
    table and visible rows (`score_topk`), its compressor's kv and score
    state, and the step's token count to its MoE: none of which a buffer's
    shape and dtype alone names."""
    leaves = [(r, json.loads(r["leaf"])) for r in rehearsed]
    widths = {
        leaf["args"]["out_logits"]["shape"][1]: _ratio(leaf, "out_logits")
        for r, leaf in leaves
        if r["op"] == SCORER
    }
    checked = set()
    for r, leaf in leaves:
        op, bound = r["op"], r["bound"]
        want = {}
        if op in ("aiter.pa_sparse_prefill_opus.default", f"{INDICES}_launch_indices"):
            want = {s: _indptr(leaf, s) for s in bound if "ptr" in s}
        elif op == f"{INDICES}_launch_indptr_scan":
            want = {s: _indptr(leaf, s) for s in bound if "ptrs." in s}
        elif op == SCORER:
            ratio = _ratio(leaf, "out_logits")
            want = {"out_logits": "workspace._logits"}
            want |= {"context_lens": f"step.visible.{ratio}"}
            want |= {"kv_indices": f"workspace._tiles.{ratio}"}
        elif op == TOPK and leaf["args"]["logits"]["role"] == "metadata":
            ratio = widths[leaf["args"]["stride0"]]
            want = {s: f"step.visible.{ratio}" for s in ("row_starts", "row_ends")}
        elif op in STATE:  # a layer of the compressor's stacked state field
            want = {f"{f}_state": f"state.compress_{f}" for f in ("kv", "score")}
            bound = {s: n.rsplit(".", 1)[0] for s, n in bound.items()}
        elif op in MOE:
            want = {MOE[op]: "scheduled_tokens_gpu"}
        elif op == "aiter.masked_embedding.default":
            want = {"x": "input_ids"}
        # a scorer or top-k of an earlier layer's candidates reads its rows' own
        want = {s: w for s, w in want.items() if s in bound}
        got = {s: bound[s].removeprefix(CACHE) for s in want}
        assert got == want, (op, r["shape"])
        checked |= {op} if want else set()
    assert checked == {
        "aiter.pa_sparse_prefill_opus.default",
        f"{INDICES}_launch_indices",
        f"{INDICES}_launch_indptr_scan",
        SCORER,
        TOPK,
        *MOE,
        *STATE,
        "aiter.masked_embedding.default",
    }
