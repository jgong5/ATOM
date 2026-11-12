# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's paged cache judging replayed batches, in a process of its own.

    python v41_state_probe.py ROOT

`test_bench_replay_v41_state.py` runs it. The cache's module reads the device's
FP8 at import, so the process first declares a gfx950 device as a capture does
(`capture.driverless.install`), which reaches no device and is process-global;
the cache itself is ATOM's, on the host. The batches, and the rule of an engine
with no drafter, are the replay rule's, pickled into `ROOT/batches.pkl`; what
each scenario saw goes to `ROOT/probed.json`.

Most scenarios span a batch as `DeepseekV41MetadataBuilder._prepare` spans
one: each row a request from its forward's first token,
`context_lens - num_scheduled_tokens`, in the slot it commits to. `builder` and
`ops` prepare it through ATOM's own builder (`prepare_decode`), assembled by
hand as `tests/attentions/deepseek_v41` assembles it; `ops` drives the op
bench's `Stub.point` beside `test_bench_ops`'s stand-in runner, each placement
at the state slots `point` gives it; `runner` calls the step bench's
`BenchModelRunner.forward` with the forward below it recorded.
"""

import dataclasses
import json
import pathlib
import pickle
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

#: The cache's PAGE, the replay rule's block size; DSpark5's drafts.
BLOCK, DRAFTS = 32, 5


def cache_of(drafts=DRAFTS):
    """ATOM's V4.1 cache on the host: 64 PAGEs, 8 state slots."""
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )

    geometry = V41PoolGeometry(
        2, ((1, 2),), BLOCK, 8, 512, 32, speculative_tokens=drafts
    )
    return PagedAttentionCache(geometry, 64, 8, "cpu")


def assembled(cache, rows, tokens):
    """V4.1's builder over `cache`, assembled by hand, its buffers for `rows`
    requests of `tokens` tokens in all."""
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from tests.attentions.deepseek_v41.helpers import metadata_buffers

    built = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    built.geometry, built.device, built.block_size = cache.geometry, "cpu", BLOCK
    built.cache, built.score_workspace = cache, None
    built.model_runner = SimpleNamespace(
        forward_vars=metadata_buffers(rows, tokens, 8, "cpu", cache.geometry),
        tokenID_processor=SimpleNamespace(num_rejected=None),
    )
    return built


def prepared(built, batch):
    """`batch`'s decode or prefill metadata and positions as ATOM's runner
    prepares them, its state then judged as `prepare_model_inputs` judges it."""
    cu = built.model_runner.forward_vars["cu_seqlens_q"]
    lengths = [int(n) for n in batch.num_scheduled_tokens]
    cu.np[0], n = 0, len(lengths)
    cu.np[1 : n + 1] = np.cumsum(lengths)
    cu.copy_to_gpu(n + 1)
    if batch.total_seqs_num_prefill:
        metadata, positions = built.prepare_prefill(batch, n, sum(lengths))
    else:
        metadata, positions = built.prepare_decode(batch, n, sum(lengths), max(lengths))
    built.cache.prepare_state(metadata.step)
    return metadata, positions


def ops_placements(
    rule,
    seat,
    point=None,
    context=20,
    drafts=0,
    v41=False,
    whole=False,
    page=False,
    pool=None,
    slots=8,
    ratio=2,
    reach=0,
    query=None,
    tail=0,
):
    """The op bench's placements of a V4.1 `point` under `rule`, a two-row
    decode by default, at `context`, over `test_bench_ops`'s state-only leaf
    with 8 state slots on the runner, on a cache of `drafts` speculative
    tokens: the state slots of each placement `Stub.point` prepared, and the
    refusal that stopped them, if any. `seat` stands in for
    `replay.seat_states`. With `v41`, the leaf instead takes the cache's two
    compressor states as the bench binds them on its builder (`ops._runner`),
    at a cache that gives it 7 copies; its record and the bytes its states
    read a call are kept too. With `whole`, it takes the window and score
    fields whole, `[layers, slots, ...]`, as `aten.select.int` takes them; with
    `page` too, its page field whole, `[layers, pages, ...]`, of a capture's 4.
    With `pool`, the cache's pool, `[rows, dim]` over its PAGEs and state, as a
    capture's 4 PAGEs and `slots` state slots held it, whole, with its ring's
    window, as `_launch_rope_window` and `swa_write` take it; at `ring`, that
    and layer 1's window ring cut from it (`ring_view`), as
    `pa_sparse_prefill_opus` takes both, stating its window of 8, compress
    `ratio` and, past ratio 0, top-k 1, or at `prefix` the pool alone stating
    them. Its activation is a row a token, `query` rows if given, and the
    bytes its ring rows read a call, `reach` rows, are kept too, and each
    pool view's name and offset from ATOM's on the bench, or refusal and None.
    With `tail`, the step publishes a bounded replay's tail `tail` rows wide,
    each request's window starting at its first row (`_publish_decoder`)."""
    import torch

    here = pathlib.Path(__file__).parent
    sys.path[:0] = [str(here), str(here.parent)]  # its flat imports and conftest
    from test_bench_ops import KEY, _graph, _leaf, _state, _t
    from test_bench_ops import _Runner as OpsRunner

    from atom.compass import ir
    from atom.compass.bench import ops, replay
    from atom.compass.capture.forward import decode
    from atom.compass.ir.nodes import Reduced
    from atom.utils import forward_context as fctx

    point = point or decode(2)
    tokens = point.tokens * (1 + drafts if point.phase == "decode" else 1)
    built = assembled(cache_of(drafts), point.seqs, tokens)

    class Runner(OpsRunner):
        """The stand-in runner, preparing its input through V4.1's builder."""

        attn_metadata_builder = built

        def prepare_model(self, batch):
            self.batches.append(list(batch.state_slots_committed))
            metadata, positions = prepared(built, batch)
            step = metadata.step
            if tail:
                floors = step.request_positions.astype(np.int32)
                step.decoder = SimpleNamespace(
                    tail=dataclasses.replace(step, request_floors=floors),
                    tail_rows=torch.zeros(tail), last=torch.zeros(point.seqs),
                    query_ids=step.batch_ids, filled_gpu=None, finishing_gpu=None,
                )  # fmt: skip
            fctx._forward_context = fctx.ForwardContext(
                attn_metadata=metadata, context=fctx.Context(positions=positions)
            )
            return torch.zeros(len(positions), dtype=torch.int32), None

    runner = Runner()
    runner.pool_plan = SimpleNamespace(entries={"state": 8})
    runner.config.pool_entries_per_req = {"state": 1}
    runner.kv_held = _state((16,))
    fctx._forward_kv_cache_context.kv_cache_data = {}
    ops.Stub._time = lambda *_: {"cold_ns": 1.0}
    state = _t([8, 16], "kv", stride=[64, 1])
    args = ("kv", _t([2, 4], "activation")), ("kv_state", state), ("score", state)
    leaf = _leaf("v4.states", *args, ("layer", ir.ContextRef(KEY)), key=KEY)
    llc, read, ring = 1000, 0, 0
    if v41:
        cache = built.cache
        runner.kv_held = ops._bound(
            "attn_metadata_builder.cache", cache, set(), runner.device
        )
        fields = ("window" if whole else "compress_kv", "compress_score")
        states = [(f, cache.state.view(f)) for f in fields]
        states = [(f, t if whole else t[-1]) for f, t in states]
        if page:
            states = [("main_1", cache.pages.view("main_1")[:, :4])]
        if pool:
            from atom.model_ops.attentions.deepseek_v41.cache import (
                PagedAttentionCache,
            )

            capture = PagedAttentionCache(cache.geometry, 4, slots, "cpu")
            states = [("pool", capture.pool)]
            states += [("ring", capture.ring_view(1))] if pool == "ring" else []
        dtyped = [(f, _t(t.shape, "kv", t.stride(), str(t.dtype))) for f, t in states]
        query = query or tokens
        leaf = _leaf("v41.states", ("kv", _t([query, 4], "activation")), *dtyped)
        if pool in ("ring", "prefix"):  # two rows' blocks hold 32 rows at ratio 2
            runner.config.kv_cache_block_size = BLOCK
            top = ("index_topk", 1 if ratio else None)
            stated = ("window_size", 8), ("compress_ratio", ratio), top
            leaf = dataclasses.replace(leaf, stated=stated)
        # two rows' slots, every layer of each
        read = sum(2 * t.select(int(whole), 0).nbytes for _, t in states)
        if pool == "whole":  # and the ring's window, as the ring's writers take it
            window = cache.geometry.window(1, 4)
            cls = f"{type(window).__module__}:{type(window).__name__}"
            given = ("window", Reduced(cls, tuple(vars(window).items())))
            leaf = dataclasses.replace(leaf, args=(*leaf.args, given))
        if pool:  # its `reach` ring rows; beside a ring at a ratio, top-k 1 a row
            pages = point.seqs if pool != "whole" and ratio else 0
            ring = reach * capture.pool[0].nbytes
            read = ring + pages * capture.pool[0].nbytes
        llc = 3 * (16 * query + read)  # 7 copies of the activation and the states
    found = ops.leaves(_graph(leaf, layers=1)["region"])
    real, replay.seat_states = replay.seat_states, seat
    refused, got, stub = None, [], ops.Stub(runner, llc)
    try:
        got = stub.point(rule, point, context, found)
        refused = next((r["refused"] for r in got if "refused" in r), None)
    except (ValueError, RuntimeError) as error:
        refused = str(error)
    finally:
        replay.seat_states = real
    kept = {"slots": runner.batches, "refused": refused, "got": got}
    if pool:
        atom = {"pool": built.cache.pool, "ring": built.cache.ring_view(1)}
        kept["bound"] = {
            p: [n, v if v is None else v.data_ptr() - atom[p].data_ptr()]
            for (_, p), (n, v) in stub.kv.items()
        }
    return kept | {"read": read, "ring": ring}


def kv_bound(slots):
    """Each KV view a capture's cache of 4 PAGEs and `slots` state slots
    passes, as `ops._kv` binds it on `cache_of`'s as `ops._runner` holds it:
    the name it binds and how far its first byte lies from that view there."""
    import torch

    from atom.compass.bench import ops
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache

    bench = cache_of()
    capture = PagedAttentionCache(bench.geometry, 4, slots, "cpu")
    held = ops._bound("cache", bench, set(), torch.device("cpu"))
    views = {
        "pages": lambda c: c.pages.buf.view(torch.bfloat16),
        "page_bytes": lambda c: c.page_bytes.view(-1),
        "state_bytes": lambda c: c.state_bytes.view(-1),
        "state_fp32": lambda c: c.state.buf.view(torch.float32),
        "cursor": lambda c: c.cursor[:, 1:],
        "pool": lambda c: c.pool,
    }
    out = {}
    for case, view in views.items():
        t = view(capture)
        spec = {"shape": [*t.shape], "stride": [*t.stride()], "dtype": str(t.dtype)}
        name, got = ops._kv(spec, "self", held)
        out[case] = [name, got.data_ptr() - view(bench).data_ptr()]
    return out


def main(root: pathlib.Path) -> None:
    from atom.compass.capture import driverless

    driverless.install(tempfile.mkdtemp(), "gfx950")

    import torch

    from atom.compass.bench import replay
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan

    batches = pickle.loads((root / "batches.pkl").read_bytes())
    plain, drafted = batches.pop("rule"), batches.pop("drafted")

    def fresh():
        cache = cache_of()
        return cache, SimpleNamespace(cache=cache)

    def step(cache, batch, **kwargs):
        spans, tables, offset = [], [], 0
        rows = zip(batch.req_ids, batch.num_scheduled_tokens, batch.context_lens)
        for i, (rid, length, end) in enumerate(rows):
            slot = batch.state_slots_committed[i]
            spans.append(RequestSpan(rid, int(end - length), offset, int(length), slot))
            tables.append([int(b) for b in batch.block_tables[i]])
            offset += int(length)
        return cache.begin_step(spans, block_tables=tables, **kwargs)

    def judged(call, *args, **kwargs):
        """None, or the refusal `call(*args, **kwargs)` raised."""
        try:
            call(*args, **kwargs)
        except ValueError as error:
            return str(error)
        return None

    def positions(cache, batch):
        return cache.cursor[batch.state_slots_committed, 0].tolist()

    seen = {}
    decode = batches["decode"]

    cache, builder = fresh()
    seen["unseated"] = judged(cache.prepare_state, step(cache, decode))

    cache, builder = fresh()
    history = cache.cursor[:, 1:].clone()
    replay.seat_states(builder, decode)
    seen["seated"] = {
        "refused": judged(cache.prepare_state, step(cache, decode, tentative=True)),
        "positions": positions(cache, decode),
        "history_kept": torch.equal(cache.cursor[:, 1:], history),
    }

    cache, builder = fresh()
    cache.state_bytes.fill_(57)
    replay.seat_states(builder, decode)
    seated = cache.state_bytes.clone()
    refused = judged(cache.prepare_state, step(cache, decode))
    seen["untouched"] = {
        "refused": refused,
        "state_kept": torch.equal(cache.state_bytes, seated),
    }

    cache, builder = fresh()
    replay.seat_states(builder, decode)
    first = step(cache, decode)
    refused = judged(lambda: cache.advance_cursor(first, cache.prepare_state(first)))
    moved = positions(cache, decode)
    unseated = judged(cache.prepare_state, step(cache, decode))
    again = []
    for _ in range(3):
        replay.seat_states(builder, decode)
        later = step(cache, decode)
        again.append(judged(cache.prepare_state, later, histories=False))
    again.append(judged(cache.prepare_state, step(cache, decode)))
    seen["repeats"] = {
        "refused": refused,
        "moved": moved,
        "unseated": unseated,
        "seated": again,
    }

    for name in ("prefix", "chunks", "prompt"):
        cache, builder = fresh()
        cache.state_bytes.fill_(57)
        batch = batches[name]
        replay.seat_states(builder, batch)
        seen[name] = {
            "refused": judged(cache.prepare_state, step(cache, batch)),
            "positions": positions(cache, batch),
            # every byte of the rows' slots rewritten: a new request's reset
            "reset": bool((cache.state_bytes[batch.state_slots_committed] != 57).all()),
        }

    cache, _ = fresh()
    seen["builder"] = {"unseated": judged(prepared, assembled(cache, 4, 24), decode)}
    cache, builder = fresh()
    replay.seat_states(builder, decode)
    seen["builder"]["seated"] = judged(prepared, assembled(cache, 4, 24), decode)

    seen["ops"] = ops_placements(plain, replay.seat_states)
    seen["ops_unseated"] = ops_placements(plain, lambda *_: None)
    from atom.compass.capture.forward import decode as width, prefill

    # DSpark5's verify rows, tentative; a chunk past a cached prefix, scaled
    # down from cell A's 2048 on 8192 to the cache's 64 PAGEs
    seen["ops_verify"] = ops_placements(
        drafted, replay.seat_states, width(2), 20, DRAFTS
    )
    seen["ops_prefill"] = ops_placements(plain, replay.seat_states, prefill(32, 1), 128)
    seen["ops_v41_state"] = ops_placements(plain, replay.seat_states, v41=True)
    seen["ops_v41_field"] = ops_placements(
        plain, replay.seat_states, v41=True, whole=True
    )
    seen["ops_v41_page"] = ops_placements(
        plain, replay.seat_states, v41=True, whole=True, page=True
    )
    # the ring rows a call reaches: a row a token written, a window of 8 a row
    # read; a ring a request at most, of 8 rows, or 13 under DSpark5
    seated = {"seat": replay.seat_states, "v41": True}
    for pool, slots in (("whole", 8), *(("ring", n) for n in (8, 4, 2, 16))):
        reach = 2 if pool == "whole" else 16
        seen[f"ops_v41_{pool}_pool_{slots}"] = ops_placements(
            plain, replay.seat_states, v41=True, pool=pool, slots=slots, reach=reach
        )
    seen["ops_v41_ring0_pool_8"] = ops_placements(
        plain, pool="ring", ratio=0, reach=16, **seated
    )
    seen["ops_v41_write_verify"] = ops_placements(
        drafted, point=width(2), drafts=DRAFTS, pool="whole", reach=12, **seated
    )
    seen["ops_v41_write_prefill"] = ops_placements(
        plain, point=prefill(32, 1), context=128, pool="whole", reach=8, **seated
    )
    # a cached prefill's prefix reads each row's 7 history ring rows through the
    # pool; a bounded-replay tail's, 16 query rows of the forward's 32, none
    cut = {"point": prefill(32, 2), "context": 96, "pool": "prefix", **seated}
    for case, ratio, query in (("prefix0", 0, 32), ("prefix2", 2, 32), ("tail", 2, 16)):
        seen[f"ops_v41_{case}"] = ops_placements(
            plain, ratio=ratio, query=query, reach=14 * (query == 32), **cut
        )
    # nor a tail laid out 48 rows wide, past the forward's 32
    seen["ops_v41_wide_tail"] = ops_placements(plain, query=48, tail=48, **cut)
    seen["kv_bound"] = {slots: kv_bound(slots) for slots in (8, 2)}

    from atom.compass.bench.runner import BenchModelRunner
    from atom.compass.parity.runner import RecordingModelRunner

    cache, builder = fresh()
    reached = []
    RecordingModelRunner.forward = lambda _, batch: reached.append(
        positions(cache, batch)
    )
    runner = BenchModelRunner.__new__(BenchModelRunner)
    runner.attn_metadata_builder = builder
    BenchModelRunner.forward(runner, decode)
    seen["runner"] = reached

    cache, builder = fresh()
    before = cache.backing.clone()
    for other in (SimpleNamespace(cache=SimpleNamespace(cursor=None)), object()):
        replay.seat_states(other, decode)
    decode.is_dummy_run = True
    replay.seat_states(builder, decode)
    seen["left_alone"] = torch.equal(cache.backing, before)

    (root / "probed.json").write_text(json.dumps(seen))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
