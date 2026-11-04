# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's paged cache judging replayed batches, in a process of its own.

    python v41_state_probe.py ROOT

`test_bench_replay_v41_state.py` runs it. The cache's module reads the device's
FP8 at import, so the process first declares a gfx950 device as a capture does
(`capture.driverless.install`), which reaches no device and is process-global;
the cache itself is ATOM's, on the host. The batches are the replay rule's,
pickled by name into `ROOT/batches.pkl`; what each scenario saw goes to
`ROOT/probed.json`. Each step spans a batch as
`DeepseekV41MetadataBuilder._prepare` spans one: each row a request from its
forward's first token, `context_lens - num_scheduled_tokens`, in the slot it
commits to.
"""

import json
import pathlib
import pickle
import sys
import tempfile
from types import SimpleNamespace

#: The cache's PAGE, the replay rule's block size; DSpark5's drafts.
BLOCK, DRAFTS = 32, 5


def main(root: pathlib.Path) -> None:
    from atom.compass.capture import driverless

    driverless.install(tempfile.mkdtemp(), "gfx950")

    import torch

    from atom.compass.bench import replay
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import (
        V41PoolGeometry,
    )

    batches = pickle.loads((root / "batches.pkl").read_bytes())
    geo = V41PoolGeometry(2, ((1, 2),), BLOCK, 8, 512, 32, speculative_tokens=DRAFTS)

    def fresh():
        cache = PagedAttentionCache(geo, 64, 8, "cpu")
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
        batch = batches[name]
        replay.seat_states(builder, batch)
        seen[name] = {
            "refused": judged(cache.prepare_state, step(cache, batch)),
            "positions": positions(cache, batch),
        }

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
