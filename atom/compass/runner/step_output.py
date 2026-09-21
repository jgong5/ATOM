# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""What a step reports, for a runner that predicts steps instead of running them.

ATOM's runner does not hand the scheduler the tokens of the step it was just
given. Three rules govern what it hands over instead, and each fails quietly
rather than loudly when a replacement gets it wrong.

**Tokens are one step late.** `tokenIDProcessor.is_deferred_out` is true
whenever the pipeline is a single stage, which is the ordinary case. The real
runner starts an asynchronous copy of this step's sampled ids to the host and
reports the copy the *previous* step started, so the reply carries the previous
batch's request ids and the previous batch's tokens with the deferred flag set.
`Scheduler.postprocess` is written against exactly that: it appends one
placeholder token per sequence at the end of every deferring call and
overwrites that placeholder on the next one. Reporting the current batch's
tokens with the flag clear offers them to a loop that has not yet appended a
placeholder for them, and the request is then not offered again until a later
batch happens to include it.

**The unit of that lag is one output-producing step, not one step.** The real
runner returns before sampling when the whole batch is middle chunks of chunked
prefills, so the queued copy is neither read nor replaced and the tokens of the
last output-producing step stay queued across however many middle chunks
follow. A replacement that carries its tokens forward by one *step* surfaces
them early, by the number of middle chunks in between.

**A step that produces nothing still reports its request ids.** The early
return names the batch's requests with an empty token list, and on the two
paths that exist today nothing reads it. A single-stage scheduler skips every
request in such a batch before it reads a token, because they are all still
mid-prompt; a pipeline head drops the entry without waiting for a reply at all.
So this one is mirrored from ATOM's own early return rather than derived from
what a caller needs -- there is no caller to derive it from, and a reply that
differs from the original differs in silence until one appears.

This module holds those rules and imports nothing from the engine, so it can be
exercised where there is no driver. It builds the keyword arguments of a batch
output rather than the object itself, for the same reason.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def reported_token_id(eos_token_id: Any, stop_token_ids: Any) -> int:
    """The id every predicted step reports for every request.

    A predicted step has no logits, so what a request generates is not modelled
    -- only how many steps it takes to generate it. The scheduler still reads
    these ids and ends a request that emits the end-of-text id or one of the
    configured stop ids, so reporting either would cut every request short at
    its first token and a predicted run would decide for itself the length it
    was asked to predict. The lowest id that is neither is used, and requests
    end on their token budget instead.
    """
    stops = {int(t) for t in (stop_token_ids or ())}
    if eos_token_id is not None:
        stops.add(int(eos_token_id))
    token_id = 0
    while token_id in stops:
        token_id += 1
    return token_id


def reports_previous_step(pipeline_parallel_size: Any) -> bool:
    """Do this runner's replies carry the previous step's tokens?

    The condition ATOM's token processor resolves from, and the same number the
    scheduler reads in the other direction: with more than one stage it
    advances chunked-prefill progress at schedule time instead of after the
    forward, and a stage reports the step it just ran. So the two arrangements
    never both apply, and one number selects between them.
    """
    return int(pipeline_parallel_size or 1) == 1


class DeferredTokenStream:
    """Carries one batch's reported tokens to the next output-producing step.

    One instance per runner, holding the single piece of state ATOM's own token
    processor holds: the last batch that produced output. Nothing here is
    per-request, because the lag is a property of the step sequence rather than
    of any request in it.
    """

    def __init__(self, token_id: int, deferred: bool = True) -> None:
        self.token_id = token_id
        self.deferred = deferred
        self.prev_batch: Any = None

    @staticmethod
    def is_pure_middle_chunk(batch: Any) -> bool:
        """Does this batch sample nothing at all?

        The question ATOM's runner asks before it returns early: a batch with
        no decode sequence and no prefill sequence on its final chunk.
        """
        return not batch.produces_output()

    def step(self, batch: Any) -> dict[str, Any]:
        """The batch output a predicted step reports, as constructor arguments."""
        if self.is_pure_middle_chunk(batch):
            # Same requests, no tokens, and the deferred flag left clear. The
            # flag is passed rather than left to the constructor default
            # because its being clear is the load-bearing part of this reply.
            return {
                "req_ids": list(batch.req_ids),
                "token_ids": [],
                "num_rejected": None,
                "num_bonus": None,
                "draft_token_ids": None,
                "is_deferred_out": False,
            }
        if not self.deferred:
            return self._reply(batch, deferred=False)
        prev, self.prev_batch = self.prev_batch, batch
        # The first output-producing step of a run has nothing queued behind
        # it, and reports no request at all rather than a row of tokens that no
        # sequence holds a placeholder for.
        return self._reply(prev, deferred=True)

    def _reply(self, source: Any, deferred: bool) -> dict[str, Any]:
        """One token per request of *source*, which may be no batch at all."""
        req_ids = [] if source is None else list(source.req_ids)
        # Sized by the batch whose tokens these are, since that is the batch
        # the scheduler indexes them with. Zero throughout: nothing was
        # drafted, so nothing was rejected and nothing is a bonus. That is a
        # true description of a run with no speculation and a silent lie about
        # one with it, which is why `overrides.forward` refuses a speculative
        # config rather than reporting these zeros for it.
        width = 0 if source is None else int(source.total_seqs_num)
        return {
            "req_ids": req_ids,
            "token_ids": [(self.token_id,) for _ in req_ids],
            "num_rejected": np.zeros(width, dtype=np.int32),
            "num_bonus": np.zeros(width, dtype=np.int32),
            "draft_token_ids": None,
            "is_deferred_out": deferred,
        }
