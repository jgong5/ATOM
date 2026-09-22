# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The projection a cost backend is priced from: one row per scheduled request.

`BatchView` and `RequestShape` live in `atom.compass.backends` because the scan
that forbids engine imports reads that package and nothing else. This module is
their other half and does the opposite: it reads a scheduled batch, the
sequences that batch was built from, and the runner's graph-capture ladder, and
turns them into those two types. Keeping the builder here is what lets it read
the engine at all.

Two things the type cannot check, both of which belong to whoever builds the
projection rather than to whoever prices it.

**The row count.** A one-row `BatchView` is a legal batch, and a one-row batch
is `tokens x history` exactly -- the collapsed form that summing per request
exists to keep out. `RequestShape(512, 2048)` sums to 1048576 where the two
256-token rows it collapses sum to 524288, and nothing in the type ties its row
count to the number of requests a scheduler scheduled. So the rows here are one
per entry of `ScheduledBatch.num_scheduled_tokens`, and a `seqs` mapping whose
keys are not the batch's own `req_ids`, in the batch's own order, is refused.
That refusal is the load-bearing line in this file rather than housekeeping:
`zip` is how both structures are read everywhere else in the tree, including by
`Scheduler.compute_detailed_aggregates`, and `zip` over a mapping one request
short drops the last row in silence -- which is a shorter batch that prices
without complaining, and at two requests is exactly the collapse.

**The rung.** The type refuses a `capture_rung` with no decode rows and one
narrower than the decode row count. It cannot refuse one that is far too wide,
because rows cannot contradict a width -- only a ladder of captured widths can
-- so a width supplied by a caller is charged at face value, and one decode row
at `capture_rung=10**9` prices 99.9999999 s of graph padding. Nothing here
takes a width from a caller. `ForwardMode.decide` is the rule ATOM dispatches a
real step by, and its answer is taken whole: `running_bs` where it says a graph
replays, `None` where it says one does not. A prefill step, an eager runner and
a batch wider than the widest captured size therefore all come back `None` from
that one call rather than from three conditions restated here.

**Why each row's history is read twice.** `ScheduledBatch.context_lens` is the
`N_KV` ATOM computed when it built the batch -- a prefill chunk's cached tokens
plus the chunk, a decode's whole sequence -- and is what this builds rows from.
The sequence is read for its type and for a second derivation of that same
`N_KV`, from `Sequence.num_tokens` and `Sequence.num_cached_tokens`, which is
the one `Scheduler.compute_detailed_aggregates` takes. They must agree. A
disagreement means the sequences handed in are no longer the ones the batch was
built from -- they have advanced, or they are a later step's -- and the two
readings then differ by a whole chunk of history. `advance_on_schedule` makes
that a configuration rather than a hypothesis: it is on whenever
`pipeline_parallel_size > 1` (`scheduler.py:994`), and it advances each
sequence's offsets after the batch has snapshotted them (`:1730`). The
projection refuses there instead of picking one of the two readings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.runner.overrides import RunnerRefusal
from atom.model_engine.sequence import SequenceType
from atom.utils.forward_context import ForwardMode


def request_rows(batch: Any, seqs: dict[int, Any]) -> tuple[RequestShape, ...]:
    """One row per scheduled request, in the batch's own order.

    `batch.req_ids` is `list(seqs.keys())` taken when the batch was built, so
    requiring the two to be equal states both halves of the projection's
    guarantee -- the row count, and that row *i* describes the request row *i*
    of every parallel array on the batch describes.
    """
    if list(seqs.keys()) != list(batch.req_ids):
        raise RunnerRefusal(
            f"the batch scheduled {list(batch.req_ids)} and the sequences handed "
            f"in are {list(seqs.keys())}; one row per scheduled request is what "
            "keeps a batch of two from being priced as the single collapsed row "
            "it would otherwise sum to, and every reader of these two pairs them "
            "positionally"
        )
    rows = []
    for index, (seq, scheduled) in enumerate(
        zip(seqs.values(), batch.num_scheduled_tokens)
    ):
        query = int(scheduled)
        decode = seq.type == SequenceType.DECODE
        context = int(batch.context_lens[index])
        settled = int(seq.num_tokens) if decode else int(seq.num_cached_tokens) + query
        if settled != context:
            raise RunnerRefusal(
                f"request {batch.req_ids[index]} reads {context} context tokens "
                f"off the batch and {settled} off its sequence; the sequences "
                "handed in are not the ones this batch was built from, and the "
                "two readings differ by history this step either did or did not "
                "compute"
            )
        rows.append(RequestShape(query, context, decode))
    if len(rows) != batch.total_seqs_num:
        raise RunnerRefusal(
            f"{len(rows)} rows for the {batch.total_seqs_num} requests the "
            "scheduler counted into this batch"
        )
    return tuple(rows)


def capture_rung(batch: Any, runner: Any) -> int | None:
    """The width of the graph this step replays, or `None` if it replays none.

    Asked of `ForwardMode.decide`, which is the rule the real runner dispatches
    by, rather than derived from the ladder here: deriving it is how a rung gets
    assigned to a prefill step, which replays nothing and would then be charged
    graph padding for a graph that did not run.

    The arguments below that do not reach the rung are passed at their inert
    values: `decide` settles a query length and a token count as well, and those
    are what `captured_tokens`, `is_block_drafter`, `tbo_on` and `local_tbo`
    feed. The width and the replay decision come from `capture_sizes`,
    `enforce_eager` and the batch alone.
    """
    dp_size = _data_parallel_size(runner)
    if dp_size > 1:
        raise RunnerRefusal(
            f"a rung under data parallelism is the group's, not this rank's: "
            f"`decide` settles it from a collective over all {dp_size} ranks, "
            "and this projection has no group to run one on"
        )
    mode = ForwardMode.decide(
        batch=batch,
        dp_size=1,
        dp_group=None,
        enforce_eager=bool(runner.enforce_eager),
        capture_sizes=np.asarray(runner.capture_sizes_np, dtype=np.int32),
        captured_tokens=None,
        is_block_drafter=False,
        tbo_on=False,
        local_tbo=(False, False, 0, 0),
        max_seqlen_q=batch.num_spec_step + 1,
    )
    return int(mode.running_bs) if mode.use_cudagraph else None


def _data_parallel_size(runner: Any) -> int:
    parallel = getattr(getattr(runner, "config", None), "parallel_config", None)
    return int(getattr(parallel, "data_parallel_size", 1) or 1)


def project(batch: Any, seqs: dict[int, Any], runner: Any) -> BatchView:
    """The whole projection: the batch's rows, and the rung it replayed."""
    return BatchView(
        request_rows(batch, seqs), capture_rung=capture_rung(batch, runner)
    )
