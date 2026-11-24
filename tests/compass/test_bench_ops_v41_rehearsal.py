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

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

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
#: V4.1's decode top-k on gfx1250, on a scorer's logits or the candidate pick's
#: maxima: aiter's routed operator, which V4.1's scorer calls since upstream
#: b7b5ec3f2, each row bounded by `seqLens`; a prefill's too.
TOPK = "aiter_routed.top_k_per_row_decode.default"
PLANE_BOUND = {
    "aiter.ops.flydsl.kernels.mqa_logits.fp8_paged_mqa_logits_gfx1250:"
    "flydsl_fp8_paged_mqa_logits_gfx1250",
    TOPK,
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
#: aten's leaves of V4.1 whose table, not an index, is bound by name: the
#: sampler's search of each request's draft counts, and the gather of a
#: replay's queries' pointers.
SEARCH = "aten.searchsorted.Tensor", "sorted_sequence"
GATHER = "aten.index.Tensor", "self"


def _run(script, *argv, cwd=TREE):
    """`script` run over `argv` in a process of its own: what it wrote to its
    first argument, as JSON."""
    done = subprocess.run(
        [sys.executable, str(script), *map(str, argv)],
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(pathlib.Path(argv[0]).read_text())


def _unstated(root) -> pathlib.Path:
    """A copy of `root`'s store as traced before V4.1's top-k and key RoPE
    stated their ratio: each graph's members stating none."""
    store = root / "unstated"
    shutil.copytree(root / "store", store, copy_function=shutil.copyfile)
    for entry in store.glob("op_graph/*/entry.json"):
        doc = json.loads(entry.read_text())
        for name in doc["members"]:
            data = (entry.parent / name).read_bytes()
            data = re.sub(rb', "stated": \{"compress_ratio": \d+\}', b"", data)
            (entry.parent / name).write_bytes(data)
            doc["members"][name] = f"sha256:{hashlib.sha256(data).hexdigest()}"
        entry.write_text(json.dumps(doc))
    return store


@pytest.fixture(scope="module")
def rehearsals(model_dir, tmp_path_factory):  # noqa: F811
    """Each cell's graphs traced into a store, and those of cell B's decode
    B=16 with no drafter (`None`), the three at once; then, all at once, by
    the cell and a name: what its own capture passes each leaf (`truth`,
    `v41_capture_truth.py`) and the rehearsal's record at each point a case
    reads, each in a process of its own."""
    undrafted = CELL.replace(DRAFTER, "")
    cells = {
        ep: (f"{SHAPES} {WINDOW}{CELL}{ep}", CELL + ep, SHAPES) for ep in ("", EP4)
    }
    cells[None] = f"--width 16{undrafted}", undrafted, "--width 16"
    roots = {cell: tmp_path_factory.mktemp("v41_rehearsal") for cell in cells}
    where = "--arch", "gfx1250", "--gpu", "0"
    with ThreadPoolExecutor(len(cells)) as pool:
        list(pool.map(lambda c: _trace(roots[c], cells[c][0], where, READ), cells))
    jobs = {}
    for cell, (_, args, shapes) in cells.items():
        root, args = roots[cell], args.split()[1:]
        model, store = root / "model", root / "store"
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy(model_dir / name, model / name)
        jobs[cell, "truth"] = (TRUTH, root / "truth.json", model, 4, *shapes.split())
        jobs[cell, "truth"] += ("--", *args)
        points = {"cached": (store, [*SHAPES.split(), "--context", "8192"])}
        points["fresh"] = store, [*SHAPES.split()[2:], *WINDOW.split()]
        if cell is None:  # each a decode of 8192 tokens a row
            unstated, decode = _unstated(root), ["--width", "16", "--context", "8192"]
            points = {
                "rope": (store, [*decode, "--only", ROPE]),
                "topk": (store, [*decode, "--only", TOPK]),
                "unstated_rope": (unstated, [*decode, "--only", ROPE]),
                "unstated": (unstated, decode),
                "unstated_topk": (unstated, [*decode, "--only", TOPK]),
                "unstated_kinds": (unstated, [*decode, "--kinds"]),
            }
        atom = ["--model", model, "-tp", "4", "--enforce-eager", "--level", "0", *args]
        for name, (at, point) in points.items():
            jobs[cell, name] = (HARNESS, root / f"{name}.json", *point, "--llc-mib")
            jobs[cell, name] += ("512", "--store", at, "--", *atom)
    with ThreadPoolExecutor(len(jobs)) as pool:
        return dict(zip(jobs, pool.map(lambda job: _run(*job), jobs.values())))


@pytest.fixture(scope="module", params=["", EP4], ids=["cell_b", "cell_a"])
def rehearsed(request, rehearsals):
    """The rehearsal's record of each leaf of a cell at a decode of 8192
    tokens a row and a prefill past as many cached, and of each prefill leaf,
    and `WINDOW`'s, on no cached tokens; and what its own capture passes each
    leaf."""
    got = {name: rehearsals[request.param, name] for name in ("cached", "fresh")}
    return got, rehearsals[request.param, "truth"]


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


def test_a_searched_or_gathered_table_is_one_atom_s_capture_passes(rehearsed):
    """The sampler's search takes its drafts' counts at a decode, and the
    queries' pointer gather, passed the prefix table and then the extend table
    at the last layer's ratio, one of them at a prefill, cached or not."""
    got, truth = rehearsed
    taken = [
        (r["shape"], (r["op"], slot), name, truth[r["leaf"]][slot])
        for r in [*got["cached"], *got["fresh"]]
        for slot, name in r["identity"].items()
        if (r["op"], slot) in (SEARCH, GATHER) and r["shape"] != "prefill T=128 B=1"
    ]
    assert all(name in want for *_, name, want in taken), taken
    shapes = {(s, leaf) for s, leaf, *_ in taken}
    assert {("decode B=16", SEARCH), ("prefill T=2048 B=1", GATHER)} <= shapes


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


def test_the_queries_pointer_gather_takes_its_last_layer_s_prefix_table():
    """At the ratio the last layer runs, not the first ratio's by layout."""
    cache = "attn_metadata_builder.cache.indptr_buffers"
    held = {f"{cache}.{n // 2}.{n % 2}": torch.zeros(129).int() for n in range(6)}
    spec = {"shape": [129], "stride": [1], "dtype": "torch.int32", "role": "kv"}
    op = {"op": GATHER[0], "args": {"self": spec}}
    known = {("rows_per_page", 64): 1, ("rows_per_page", 2): 2, ("queries", None): 2}
    pool = ops._named(op, "self", spec, held, known) or held
    assert ops._kv(spec, "self", pool)[0] == f"{cache}.2.0"


@pytest.fixture(scope="module")
def undrafted(rehearsals):
    """The rehearsal's record of cell B's RoPE leaves alone at a decode B=16
    with no drafter, no compress leaf beside them, at 8192 tokens a row; and
    its capture's truth."""
    return rehearsals[None, "rope"], rehearsals[None, "truth"]


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


def test_a_key_rope_stating_no_ratio_refuses_where_both_ratios_plans_hold_its_rows(
    rehearsals,
):
    """In a graph traced before V4.1's key RoPE stated its ratio, at a decode
    with no drafter, both ratios' keys' RoPEs are one leaf, and both ratios'
    compress plans hold its rows: alone or beside them, it refuses by name, to
    be traced again, and the query's RoPE takes what ATOM's capture passes it."""
    truth = rehearsals[None, "truth"]
    for name in ("unstated_rope", "unstated"):  # alone, and beside them
        ropes = [r for r in rehearsals[None, name] if r["op"] == ROPE]
        heads = [json.loads(r["leaf"])["args"]["input"]["shape"][-2] for r in ropes]
        (_, key), (_, query) = sorted(zip(heads, ropes), key=lambda h: h[0])
        assert key["why"].endswith("holds its 16 rows: trace its shape again")
        assert [query["identity"].get("positions")] == truth[query["leaf"]]["positions"]


def test_each_top_k_alone_takes_its_ratio_s_visible_rows(rehearsals):
    """The row bounds ATOM stages for a top-k are its ratio's visible rows,
    which the capture states: each such top-k, with no scorer beside it, takes
    those ATOM's capture passes it, each ratio's apart."""
    got, truth = rehearsals[None, "topk"], rehearsals[None, "truth"]
    staged = {
        r["leaf"]: r["identity"].get("seqLens")
        for r in got
        if json.loads(r["leaf"])["args"]["seqLens"]["role"] == "metadata"
    }
    assert staged == {leaf: truth[leaf]["seqLens"][0] for leaf in staged}
    assert len(set(staged.values())) == len(staged) == 2
    assert {r["verdict"] for r in got} == {"reaches launch"}


def test_a_top_k_or_key_rope_stating_no_ratio_refuses_where_its_point_cannot_tell(
    rehearsals,
):
    """In a graph traced before V4.1's top-k and key RoPE stated their ratio, a
    top-k whose row bounds ATOM stages, with no scorer beside it, and the key
    RoPE beside one ratio's compressor alone, both ratios' plans holding its
    rows, refuse by name, to be traced again; a top-k beside its scorer binds."""
    why = {}
    for only, name in (("--only", "unstated_topk"), ("--kinds", "unstated_kinds")):
        for r in rehearsals[None, name]:
            args = json.loads(r["leaf"])["args"]
            if r["op"] == ROPE or args.get("seqLens", {}).get("role") == "metadata":
                told = r["why"].rpartition("cannot tell which ratio's ")[-1]
                why[only, args.get("stride0")] = told
    again = ": trace its shape again"
    assert why == {
        ("--only", 131072): f"scorer writes its 131072-wide logits{again}",
        ("--only", 262144): f"scorer writes its 262144-wide logits{again}",
        ("--kinds", 131072): "",
        ("--kinds", None): f"compress plan holds its 16 rows{again}",
    }


def test_layer_39_s_queries_rotate_at_the_positions_its_replay_gathers(rehearsed):
    """At a replayed prefill, layer 39's query RoPE and its output's inverse,
    over its queries' rows alone, take the positions its decoder plan gathers
    for them."""
    got, _ = rehearsed
    rope = "atom.model_ops.deepseek_v41.rope_window:_launch_rope_window"
    inverse = "aiter._inverse_rope_group_quant_kernel.default"
    taken = {
        (r["op"], r["bound"].get("positions"))
        for r in got["cached"] + got["fresh"]
        if r["op"] in (rope, inverse) and r["shape"] == "prefill T=2048 B=1"
        if json.loads(r["leaf"])["args"]["positions"]["shape"] == [1]
    }
    assert taken == {(op, "plan.queries.positions") for op in (rope, inverse)}
