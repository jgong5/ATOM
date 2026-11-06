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
    rule, seat, point=None, context=20, drafts=0, v41=False, whole=False, page=False
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
    `page` too, its page field whole, `[layers, pages, ...]`, of a capture's 4."""
    import torch

    here = pathlib.Path(__file__).parent
    sys.path[:0] = [str(here), str(here.parent)]  # its flat imports and conftest
    from test_bench_ops import KEY, _graph, _leaf, _state, _t
    from test_bench_ops import _Runner as OpsRunner

    from atom.compass import ir
    from atom.compass.bench import ops, replay
    from atom.compass.capture.forward import decode
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
    llc, read = 1000, 0
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
        dtyped = [(f, _t(t.shape, "kv", t.stride(), str(t.dtype))) for f, t in states]
        leaf = _leaf("v41.states", *args[:1], *dtyped)
        # two rows' slots, every layer of each
        read = sum(2 * t.select(int(whole), 0).nbytes for _, t in states)
        llc = 3 * (32 + read)  # 7 copies of 32 bytes of activation and the states
    found = ops.leaves(_graph(leaf, layers=1)["region"])
    real, replay.seat_states = replay.seat_states, seat
    refused, got = None, []
    try:
        got = ops.Stub(runner, llc).point(rule, point, context, found)
        refused = next((r["refused"] for r in got if "refused" in r), None)
    except (ValueError, RuntimeError) as error:
        refused = str(error)
    finally:
        replay.seat_states = real
    return {"slots": runner.batches, "refused": refused, "got": got, "read": read}


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
