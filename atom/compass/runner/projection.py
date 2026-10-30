# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The projection a cost backend is priced from: one row per scheduled request.

`BatchView` and `RequestShape` live in `atom.compass.backends` because the scan
that forbids engine imports reads that package and nothing else. This module is
their other half and does the opposite: it reads a scheduled batch and the
runner's graph-capture ladder, and turns them into those two types. Keeping the
builder here is what lets it read the engine at all.

**The row count.** `BatchView` cannot tie its row count to the number of
requests a scheduler scheduled, so `batch_view` builds one row per entry of
`ScheduledBatch.num_scheduled_tokens`, each with the `N_KV` ATOM computed when
it built the batch, `ScheduledBatch.context_lens`.

**The rung.** The type refuses a `capture_rung` with no decode rows and one
narrower than the decode row count. It cannot refuse one that is far too wide,
because rows cannot contradict a width -- only a ladder of captured widths can
-- so a width supplied by a caller is charged at face value, and one decode row
at `capture_rung=10**9` prices 99.9999999 s of graph padding. Nothing here
takes a width from a caller. `ForwardMode.decide` is the rule ATOM dispatches a
real step by, and its answer is taken whole: `running_bs` where it says a graph
replays, `None` where it says one does not. A prefill step, an eager runner and
a batch wider than the widest captured size therefore all come back `None` from
that one call rather than from three conditions restated here. `decide` finds
the width by binary search and so has a precondition -- an ascending ladder --
which is checked here rather than assumed, because an unsorted ladder answers a
width that is not the smallest captured one holding the batch, and too wide is
the direction the type cannot refuse.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.runner.overrides import RunnerRefusal
from atom.utils.forward_context import ForwardMode


def _rung(mode: ForwardMode) -> int | None:
    return int(mode.running_bs) if mode.use_cudagraph else None


def dp_group() -> Any:
    """The DP group's CPU group: the one `ModelRunner.prepare_model` hands `decide`."""
    from aiter.dist.parallel_state import get_dp_group

    return get_dp_group().cpu_group


def forward_mode(batch: Any, runner: Any) -> ForwardMode:
    """The step's shape, settled by `ForwardMode.decide` as the real runner settles it.

    Under data parallelism this runs the step's collective, `sync_dp_metadata`,
    on the DP group, so every rank calls it once per step, as the real forward
    does; the group's rung and token counts come back from it.

    `tbo_on` and `local_tbo` are what `ModelRunner.prepare_model` passes: the
    config's `enable_tbo` and the runner's own `_local_tbo_eligibility`, which
    ride the collective, so the group agrees the two-batch-overlap split a real
    run agrees. The other arguments that do not reach the rung are passed at
    their inert values: `decide` settles a query length as well, and that is
    what `is_block_drafter` feeds. The width and the replay decision come from
    `capture_sizes`, `enforce_eager`, the attention builder's
    `step_needs_eager`, `tbo_decode_graphs`, the batch and the collective
    alone; the group's running tokens land in the runner's own
    `running_tokens_per_dp_rank`, as the real forward's do.
    """
    dp_size = _data_parallel_size(runner)
    tbo_on = runner.config.enable_tbo
    ladder = np.asarray(runner.capture_sizes_np, dtype=np.int32)
    if bool(np.any(ladder[1:] < ladder[:-1])):
        raise RunnerRefusal(
            f"the capture ladder {ladder.tolist()} is not ascending, which is "
            "the precondition of the binary search `decide` resolves the rung "
            "with; out of order it answers a width that is not the narrowest "
            "captured one holding this batch, and a rung too wide is the one "
            "direction the rows cannot contradict"
        )
    return ForwardMode.decide(
        batch=batch,
        dp_size=dp_size,
        dp_group=dp_group() if dp_size > 1 else None,
        enforce_eager=bool(runner.enforce_eager),
        step_needs_eager=runner.attn_metadata_builder.step_needs_eager(batch),
        capture_sizes=ladder,
        is_block_drafter=False,
        tbo_on=tbo_on,
        local_tbo=(
            runner._local_tbo_eligibility(batch) if tbo_on else (False, False, 0, 0)
        ),
        max_seqlen_q=batch.num_spec_step + 1,
        running_tokens_per_dp_rank=runner.running_tokens_per_dp_rank,
        tbo_decode_graphs=runner.tbo_decode_graphs,
    )


def _data_parallel_size(runner: Any) -> int:
    """How many ranks `decide` would settle a rung across, or a refusal.

    Absent, zero, negative or `False` is refused rather than read as one,
    the set ATOM's `ParallelConfig` refuses too. A runner shaped differently
    from the engine's own would otherwise be priced as a single rank on a
    default nothing checked; zero or a negative width would pass the `> 1`
    test in `forward_mode` as the single-rank case.
    """
    parallel = getattr(getattr(runner, "config", None), "parallel_config", None)
    size = getattr(parallel, "data_parallel_size", None)
    if size is None or size < 1:
        raise RunnerRefusal(
            "this runner states no width in "
            f"`config.parallel_config.data_parallel_size` (it reads {size!r}), "
            "and a rung is one rank's answer or a group's depending on it; "
            "reading that as 1 is the guess this would have to make to carry on"
        )
    return int(size)


def batch_view(batch: Any, mode: ForwardMode, runner: Any) -> BatchView:
    """The projection a worker builds: it holds the batch and not the sequences.

    A row's kind is the convention `ModelRunner.prepare_inputs` reads: the
    first `total_seqs_num_prefill` rows are prefill. The rung and the MoE
    gather's rows come from `mode`, the step's one `forward_mode`; a
    data-parallel runner handed a mode with no group counts is refused.
    Whether the step produces output is `ScheduledBatch.produces_output`, and
    the TBO split is the collective's `ub_tokens_across_dp`, this rank's slot
    in it the config's `data_parallel_rank`. A data-parallel rank's
    `is_dummy_run` batch is its `dp_dummy`. A single rank reads none of the
    collective's counts, though `decide` returns them for a group of one.

    The MoE gather's rows follow `running_tokens_are_unified`, as
    `FusedMoE.forward_impl_graph` does. Every rank decoding, the padded
    `all_gather` carries `running_tokens` rows from each rank, the group's
    agreed height, under `enforce_eager` too. Otherwise (a rank prefilling) the
    variable-length gather carries each rank's own count. An MTP drafter's
    first pass runs each rank's target token stream unpadded, so its gather is
    always the variable-length one (`EagleProposer.propose`); a block drafter
    pads its passes to the group and is left out. Its later passes run at
    `running_bs`, on any step, when it declares a draft graph, else at the
    rows (`draft_bs`), where the gather carries each rank's rows (`draft_rows`):
    every rank decoding, its tokens over a row's query, every row's alike; else
    at least the widest rank's rows and one each peer, an idle one's dummy. A
    drafter of one draft (`mtp_k`) has no later passes.

    A target that replays no graph pads to it too (`target_bs`): an eager
    decode every rank decodes in, its inputs; a sampling prefill, its LM head.
    """
    draft_moe_rows = moe_tokens = dp_rank = draft_bs = target_bs = draft_rows = None
    drafter = getattr(runner, "drafter", None)
    if getattr(drafter, "step", None) is not None:  # `EagleProposer`'s pass
        draft_bs = int(mode.running_bs)
    prefill = batch.total_seqs_num_prefill
    if not mode.use_cudagraph and (
        batch.produces_output() if prefill else mode.running_tokens_are_unified
    ):
        wider = mode.running_bs > len(batch.num_scheduled_tokens)
        target_bs = mode.running_bs if wider else None
    if _data_parallel_size(runner) == 1:
        moe_rows = None
    elif mode.sync is not None and len(
        mode.sync.scheduled_tokens_per_dp_rank
    ) == _data_parallel_size(runner):
        dp_rank = runner.config.parallel_config.data_parallel_rank
        gathered = moe_tokens = int(mode.sync.scheduled_tokens_per_dp_rank.sum())
        if mode.running_tokens_are_unified:
            moe_rows = _data_parallel_size(runner) * mode.running_tokens
        else:
            moe_rows = gathered
        if drafter is not None and not drafter.is_block_drafter:
            draft_moe_rows = gathered
            if draft_bs is None and drafter.mtp_k > 1:
                draft_rows = (
                    gathered // int(batch.num_scheduled_tokens[0])
                    if mode.running_tokens_are_unified
                    else mode.sync.max_bs_across_dp + _data_parallel_size(runner) - 1
                )
    else:
        raise RunnerRefusal(
            "this rank is one of a data-parallel group and its forward mode "
            "states no scheduled_tokens_per_dp_rank for that group, so the MoE "
            "segment has no gathered row count to be priced from"
        )
    rows = zip(batch.num_scheduled_tokens, batch.context_lens)
    return BatchView(
        tuple(
            RequestShape(int(q), int(c), i >= prefill) for i, (q, c) in enumerate(rows)
        ),
        capture_rung=_rung(mode),
        moe_rows=moe_rows,
        draft_moe_rows=draft_moe_rows,
        draft_bs=draft_bs,
        target_bs=target_bs,
        produces_output=batch.produces_output(),
        tbo_tokens=None if dp_rank is None else mode.sync.ub_tokens_across_dp,
        image_tokens=image_tokens(batch, runner),
        moe_tokens=moe_tokens,
        dp_rank=dp_rank,
        dp_dummy=dp_rank is not None and bool(batch.is_dummy_run),
        draft_rows=draft_rows,
    )


def image_tokens(batch: Any, runner: Any) -> int:
    """The image rows a step's forward handles, as `ModelRunner.run_model`
    handles them. With embedding spans, a payload still carrying pixel values
    encodes every span on this chunk, even one that fills none; a later chunk
    fills the spans it overlaps. Without spans, the image's embeddings sit at
    the model's image token."""
    from atom.model_engine.multimodal_runtime import embedding_indices

    media = getattr(batch, "multimodal_data", None) or {}
    count, offset = 0, 0
    for req, length, end in zip(
        batch.req_ids, batch.num_scheduled_tokens, batch.context_lens
    ):
        length, data = int(length), media.get(req)
        if data is not None and "embedding_spans" in data:
            spans = data["embedding_spans"]
            if "pixel_values" in data:
                count += sum(rows for _, rows in spans)
            else:
                count += len(embedding_indices(spans, int(end) - length, length)[0])
        elif data is not None:
            image = runner.config.multimodal_config.image_token_id
            mine = batch.scheduled_tokens[offset : offset + length]
            count += int(np.count_nonzero(mine == image))
        offset += length
    return count
