"""The part of Compass that does not need a device.

Splitting this out is not tidiness. ``atom.model_engine.model_runner`` imports
``aiter``, which initialises a Triton driver at import time and raises when no
GPU is visible, so anything that inherits from ``ModelRunner`` cannot even be
*imported* on a machine without one. Predicting a step needs none of that: it
needs the batch, a cost oracle, and somewhere to write the row.

So the predict path lives here and is mixed into both runners:

* :class:`~atom.compass.runtime.runner.CompassModelRunner` -- a real
  ``ModelRunner`` that can also measure and trace.
* :class:`~atom.compass.replay.runner.ReplayModelRunner` -- no base runner at
  all, answering the engine's startup RPCs from a captured target.

One definition, two hosts. A change to how a step is priced or recorded lands on
both, which is the only reason a GPU-free replay is evidence about the GPU run.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from atom.compass.config import CompassConfig
from atom.compass.core.artifacts import rank_path
from atom.compass.core.cost.base import CostOracle, StepShape
from atom.compass.core.graph import OpGraph
from atom.model_engine.scheduler import (
    ScheduledBatch,
    ScheduledBatchOutput,
    is_pure_middle_chunk,
)
from atom.utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)

__all__ = ["CompassPredictMixin"]


class CompassPredictMixin:
    """Pricing, recording and step description, without a device.

    Expects the host class to provide ``config`` and ``rank``, and -- for the
    ``trace``/``measure`` branches of :meth:`forward`, which only a real runner
    reaches -- ``_forward_traced`` and ``_forward_measured``.
    """

    def _init_compass_state(self) -> None:
        """Everything Compass adds, once the runner underneath it exists.

        Split out of ``__init__`` so a runner that is *not* built on a device
        can reuse it -- ``ReplayModelRunner`` has no ``ModelRunner.__init__`` to
        call, but needs exactly this state to price and record a step. Anything
        added here is added to both, which is the point.
        """
        self._oracle: CostOracle = self._build_oracle(self._compass_config)
        self._graph = OpGraph()
        self._traced_steps = 0
        self._prefill_index = 0
        self._decode_index = 0
        self._step_index = 0
        self._measured_steps = 0
        self._measured_by_kind: dict = {}
        self._measure_fh = None
        # Timed steps whose CUDA events have not been read back yet.
        import collections as _collections

        self._pending = _collections.deque()
        # When the previous forward returned, on the wall clock. The wall time
        # between one forward returning and the next starting is everything the
        # engine does that is not a forward -- scheduling, block accounting,
        # sampling, routing output, crossing the process boundary. A simulated
        # run advances its clock by predicted forward durations alone, so that
        # time does not exist, and a quarter of TTFT was found to live in it.
        # Measured rather than inferred by subtraction, which cannot tell a
        # scheduler gap from a mis-measured forward.
        self._last_forward_ended: Optional[float] = None
        # The previous predicted step's request ids, held back one step because
        # the engine defers real output and `postprocess` depends on it. None
        # until the first forward has run; see the predict branch of `forward`.
        self._deferred_output: Optional[list] = None
        logger.info(
            "ATOMCompass active: mode=%s oracle=%s",
            self._compass_config.mode,
            self._oracle.describe(),
        )
        if self._compass_config.mode == "trace":
            self._warn_if_compiled()

    @property
    def _compass_config(self) -> CompassConfig:
        """The active settings, resolved on first use rather than in ``__init__``.

        ``ModelRunner.__init__`` warms the model up before it returns, and
        warmup drives a forward — so this class's own ``__init__`` body has not
        run yet the first time anything here asks which mode it is in. Assigning
        the config after ``super().__init__()`` therefore left every
        mode-dependent decision during startup reading an attribute that did not
        exist yet.

        ``self.config`` is set early in the base ``__init__`` (well before
        warmup), so resolving lazily is safe where assigning eagerly was not.
        """
        cached = self.__dict__.get("_compass_config_cache")
        if cached is None:
            cached = self._resolve_compass_config()
            self.__dict__["_compass_config_cache"] = cached
        return cached

    def _resolve_compass_config(self) -> CompassConfig:
        config = getattr(self.config, "compass_config", None)
        if config is None:
            # Injected by qualname without an explicit config: run with defaults
            # rather than fail, so the runner is usable from the CLI alone.
            config = CompassConfig(enabled=True)
        return config

    def _build_oracle(self, config: CompassConfig) -> CostOracle:
        """Construct the configured oracle, telling it which rank it serves.

        ``oracle_options`` is deliberately opaque — it is whatever the oracle's
        constructor takes — so the rank is offered rather than imposed: passed
        only to an oracle that names ``rank_coords`` in its signature. An oracle
        that reads a per-rank artifact needs it; one that computes a cost from
        shape alone does not, and should not have to accept an argument it would
        ignore.
        """
        import inspect

        oracle_cls = resolve_obj_by_qualname(config.oracle_qualname)
        options = dict(config.oracle_options or {})
        try:
            takes_rank = "rank_coords" in inspect.signature(oracle_cls).parameters
        except (TypeError, ValueError):  # builtins, C types, odd callables
            takes_rank = False
        # Guarded by the same condition the write side uses, so a single-rank
        # run behaves exactly as it did before this existed: no suffix was
        # written, so none should be looked for.
        if (
            takes_rank
            and "rank_coords" not in options
            and any(size > 1 for size in self._topology().values())
        ):
            options["rank_coords"] = self._rank_coords()
        return oracle_cls(**options)

    def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        """Predict the step, or trace it, depending on the configured mode."""
        if self._runs_real_forward and getattr(batch, "is_dummy_run", False):
            # Warmup drives synthetic batches through this same entry point.
            # They are real forwards, so they must actually run — but they are
            # not steps a deployment performs, and counting them would spend the
            # trace budget on a dummy shape and put dummy timings in the table
            # that a cost model is fitted to.
            #
            # `super()` here is whatever this mixin is mixed in *front of* --
            # `ModelRunner` for `CompassModelRunner`. A runner with no real
            # forward underneath (`ReplayModelRunner`) never reaches this line:
            # it only exists in `trace` and `measure`, and a replay refuses to
            # start in either.
            return super().forward(batch)
        if self._compass_config.mode == "trace":
            return self._forward_traced(batch)
        if self._compass_config.mode == "measure":
            return self._forward_measured(batch)

        # Stamped by the engine core, which owns the clock that arrivals and
        # first tokens are stamped on; see `_stamp_step_start`. Falling back to
        # this process's clock would put the step in a different time domain,
        # which is the defect the validity check exists to catch, so the field
        # is left absent instead.
        started_at = getattr(batch, "compass_started_at", None)
        shape = self._describe(batch)
        cost = self._oracle.estimate(shape)
        # Record what was predicted, in the same format a measure run records
        # what was timed. Without it a simulated run leaves no trace of *which
        # steps it ran*, and the shape distribution is an output of the
        # simulation rather than an input to it -- the scheduler batches
        # according to the clock the oracle drives. Comparing a real run with a
        # simulated one on aggregate latency alone cannot tell a wrong step cost
        # from a different set of steps, which is where the serving diagnosis
        # ran out of evidence twice.
        self._record_measurement(shape, cost.seconds, None,
                                 req_ids=list(batch.req_ids),
                                 started_at=started_at,
                                 decision=getattr(batch, "compass_decision", None))
        self._step_count = getattr(self, "_step_count", 0) + 1
        logger.debug(
            "COMPASS step %d: reqs=%d tokens=%d prefill_tokens=%d cost=%.6fs",
            self._step_count, shape.batch_size, shape.total_tokens,
            shape.num_prefill_tokens, cost.seconds,
        )

        # Deferred by one step, because the engine is built expecting it.
        #
        # ModelRunner defers output whenever pipeline_parallel_size == 1, which
        # is the default, and `Scheduler.postprocess` is written around that: it
        # walks `self.running` and skips any sequence absent from `fwd_output`.
        # Its own comment spells out why that matters -- "the prefill step's
        # postprocess sees idx=None and skips this seq. By the time the prefill
        # output surfaces, the next step's schedule has already flipped seq.type
        # to DECODE". A request finishing its last prompt chunk is not yet in
        # `running` when that step is postprocessed, so the *only* reason its
        # first token is ever picked up is that the output arrives a step late.
        #
        # Returning the current batch's tokens offers them one step too early,
        # when the sequence is still invisible to that loop, and never offers
        # them again until a decode batch happens to include it. Measured on the
        # 27B: every request whose prefill completed inside a 36-step prefill
        # streak was stamped at step 37, the first decode step. First tokens 64
        # seconds late and TTFT 95% over, while the schedule and the step costs
        # agreed with the real run to a fraction of a percent. The clock was
        # right; the bookkeeping was not.
        # ...but only on the steps the real runner defers, which is not all of
        # them. `ModelRunner.forward` returns early for a pure middle chunk --
        # `batch.produces_output()` is false, nothing was sampled -- with
        # `is_deferred_out` unset and no tokens, so a middle chunk never takes a
        # turn in the buffer. Deferring on every step instead makes the lag one
        # step where the engine's is one *meaningful* step, and on a chunked
        # prefill those differ by the whole prompt: the 27B's first request
        # completed at step 0 and its token surfaced at step 4, three middle
        # chunks later, about eight seconds. Deferring uniformly gave 1.7 s.
        filler = self._compass_config.filler_token_id
        if is_pure_middle_chunk(batch):
            return ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
                compass_step_seconds=cost.seconds,
            )

        previous = self._deferred_output
        self._deferred_output = list(batch.req_ids)

        req_ids = previous if previous is not None else []
        token_ids = [(filler,) for _ in req_ids]
        width = len(req_ids)

        return ScheduledBatchOutput(
            req_ids=req_ids,
            token_ids=token_ids,
            # Subscripted per request by postprocess once deferral is declared,
            # so these are arrays of zeros rather than None: nothing is
            # speculated here, but "nothing" still has to be indexable.
            num_rejected=np.zeros(width, dtype=np.int32),
            num_bonus=np.zeros(width, dtype=np.int32),
            draft_token_ids=None,
            is_deferred_out=True,
            # The *current* step's cost, not the deferred batch's: this is what
            # the virtual clock advances by, and it belongs to the step that
            # just ran.
            compass_step_seconds=cost.seconds,
        )

    def _count_and_record(self, shape: StepShape, seconds: float,
                          gap: Optional[float] = None,
                          req_ids: Optional[list] = None,
                          started_at: Optional[float] = None,
                          decision: Optional[dict] = None) -> None:
        kind = "prefill" if shape.is_prefill else "decode"
        seen = self._measured_by_kind.get(kind, 0) + 1
        self._measured_by_kind[kind] = seen
        self._measured_steps += 1
        # Counted per kind, not overall. Prefill happens a handful of times in a
        # whole run, so a warmup counted in total steps discards every prefill
        # sample there is.
        if seen > self._compass_config.measure_warmup_steps:
            self._record_measurement(shape, seconds, gap, req_ids=req_ids,
                                     started_at=started_at, decision=decision)

    def _record_measurement(self, shape: StepShape, seconds: float,
                            gap: Optional[float] = None,
                            req_ids: Optional[list] = None,
                            started_at: Optional[float] = None,
                            decision: Optional[dict] = None) -> None:
        """Append one timed step to the table.

        Appended and flushed per step rather than collected and written at exit.
        There is no shutdown hook on the runner to flush from, and a table that
        only exists if the process ends cleanly is a table that goes missing
        exactly when a run was interesting. A partial file is honest here in a
        way a partial op graph is not: every row in it is a step that really
        happened and was really timed.
        """
        path = self._compass_config.measure_out
        if not path:
            return
        import json

        if self._measure_fh is None:
            if any(size > 1 for size in self._topology().values()):
                path = self._rank_path(path, self._rank_coords())
            try:
                self._measure_fh = open(path, "w", encoding="utf-8")
            except OSError as exc:
                logger.warning("ATOMCompass WARNING: could not open %s for timings: %s",
                               path, exc)
                self._compass_config.measure_out = None
                return
            logger.info("ATOMCompass: writing step timings to %s", path)

        self._measure_fh.write(json.dumps({
            "seconds": seconds,
            "num_scheduled_tokens": list(shape.num_scheduled_tokens),
            "context_lens": list(shape.context_lens),
            "num_prefill_tokens": shape.num_prefill_tokens,
            "topology": dict(shape.topology),
            "rank_coords": dict(shape.rank_coords),
            "capture_bucket": shape.capture_bucket,
            # Wall seconds between the previous forward returning and this one
            # starting: the engine's own work, which a simulated run does not
            # advance its clock for. None on the first step of a process, where
            # there is no previous forward to measure from.
            "gap_seconds": gap,
            # Which requests this step served. Without it a table says how many
            # steps ran and of what shape, and cannot say whose they were -- so
            # a simulated run matching the real one on step counts, device
            # seconds and occupancy, while reporting TTFT at twice the truth,
            # has no evidence left to examine. That is exactly where the
            # cc-traces comparison stopped.
            "req_ids": [str(r) for r in req_ids] if req_ids else None,
            # Why the scheduler chose this step: what was waiting, how much of
            # it still had prefill to do, what was held for a declared arrival,
            # and how much of the token budget went. The step sequence alone
            # cannot distinguish a scheduler that decided differently from one
            # that saw different state, and that is where every real-versus-
            # simulated comparison here has run out of evidence.
            "decision": decision,
            # When this step began, on the clock that stamps request arrivals --
            # wall time on a real run, virtual time on a simulated one. Recorded
            # rather than reconstructed, so queueing can be measured instead of
            # inferred.
            "started_at": started_at,
        }) + "\n")
        self._measure_fh.flush()

    @staticmethod
    def _rank_path(path: str, coords: dict[str, int]) -> str:
        """Give each rank its own file.

        Every rank traces, and under any parallelism their graphs differ — that
        difference is the thing worth recording. Writing them all to one path
        makes them race for it and leaves a single file that names no rank, so
        the one artifact that survives cannot be attributed and the rest are
        lost without a trace.

        The convention itself lives in :mod:`atom.compass.core.artifacts`, so
        whatever reads these files back resolves them the same way this wrote
        them.
        """
        return rank_path(path, coords)

    def _compilation_level(self) -> Optional[int]:
        compilation = getattr(self.config, "compilation_config", None)
        return getattr(compilation, "level", None)

    def _describe(self, batch: ScheduledBatch) -> StepShape:
        """Translate an ATOM batch into the oracle's engine-agnostic input.

        Per-request token counts and history lengths are carried through
        unreduced; the oracle decides what to do with them.
        """
        num_scheduled = tuple(int(n) for n in batch.num_scheduled_tokens)
        # These arrive as numpy arrays, so test for None rather than truthiness:
        # `arr or default` raises on anything with more than one element.
        raw_context_lens = getattr(batch, "context_lens", None)
        if raw_context_lens is None:
            context_lens = tuple(0 for _ in num_scheduled)
        else:
            context_lens = tuple(int(n) for n in raw_context_lens)
        return StepShape(
            num_scheduled_tokens=num_scheduled,
            context_lens=context_lens,
            num_prefill_tokens=int(getattr(batch, "total_tokens_num_prefill", 0)),
            topology=self._topology(),
            rank_coords=self._rank_coords(),
            capture_bucket=self._capture_bucket(
                len(num_scheduled),
                prefilling=int(getattr(batch, "total_tokens_num_prefill", 0)) > 0),
            # A step that is neither replayed nor dispatched operator by
            # operator pays neither of those overheads, and the oracle cannot
            # tell which from the graph -- a compiled step and an eager one
            # dispatch the same operators when traced, because tracing forces
            # eager. Only the engine knows, so it says.
            compiled=(self._compilation_level() or 0) > 0,
            # Whether the LM head runs at all. Asked of the scheduler's own
            # predicate rather than reconstructed from the lengths, because
            # "is this a request's final chunk" is not in them -- and it is the
            # same predicate the runner asks before skipping `compute_logits`
            # entirely on a pure middle chunk.
            produces_output=not is_pure_middle_chunk(batch),
        )

    def _capture_bucket(self, batch_size: int,
                        prefilling: bool = False) -> Optional[int]:
        """Which rung of the CUDA-graph ladder this batch replays at.

        Mirrors `ForwardMode.decide`, which is what actually picks the graph::

            running_bs = next((x for x in capture_sizes if x >= unified_bs), ...)

        -- the smallest capture size no smaller than the batch. Written here as
        a `min` rather than copied verbatim, because that expression is only
        correct on an ascending list and `capture_sizes` is sorted both ways
        during a run: descending for the capture loop, ascending again once
        capture finishes. `min` does not care, and this has to stay right if the
        ordering changes again.

        (`ModelRunner`'s input-buffer padding at the `fill_to` bound reverses the
        list before scanning, so on the ascending list it holds at runtime it
        takes the *largest* rung rather than the smallest -- against its own
        comment about a 65-request batch replaying the 128 graph. That is an
        engine bug and it over-zeroes a buffer rather than mis-selecting a graph,
        so it is not copied here. Mirroring it is what first made every step in a
        sweep report bucket 512.)

        None when no graph is replayed: `enforce_eager`, a ladder not yet
        resolved, a batch larger than the top rung -- `ForwardMode.decide` falls
        back to eager there -- or a **prefill** step, which runs eager whatever
        the ladder holds, because the ladder is captured at one token per
        sequence and a prefill step has hundreds. A bucket that did not happen
        must not be fitted as though it did, and reporting one for prefill told
        a cost model the host was not in the loop when it was: the same graph
        costs its kernels plus 2 microseconds a launch replayed and plus 67
        microseconds an operator eager, so prefill priced at 0.42 of its step.
        """
        if prefilling or getattr(self, "enforce_eager", False):
            return None
        sizes = getattr(self, "capture_sizes", None)
        if not sizes or sizes == [0]:
            return None
        return min((g for g in sizes if g >= batch_size), default=None)

    def _topology(self) -> dict[str, int]:
        """Communication group sizes, by name.

        Names are opaque to Compass. A group is a size and a membership; what a
        given strategy means is carried by the operators recorded against it.
        """
        parallel = getattr(self.config, "parallel_config", None)
        groups = {
            "tp": int(getattr(self.config, "tensor_parallel_size", 1) or 1),
            "pcp": int(getattr(self.config, "prefill_context_parallel_size", 1) or 1),
            "dcp": int(getattr(self.config, "decode_context_parallel_size", 1) or 1),
        }
        if parallel is not None:
            groups["dp"] = int(getattr(parallel, "data_parallel_size", 1) or 1)
        return {name: size for name, size in groups.items() if size > 1} or {"tp": 1}

    def _rank_coords(self) -> dict[str, int]:
        """This rank's index within each group it belongs to.

        Must name every group `_topology()` names. A coordinate left out does not
        make the artifacts merge -- it makes the ranks that differ only in that
        coordinate resolve to one path and race for it, and the winner is
        whichever wrote last. Under TP=2 x DP=2 all four ranks wrote one
        `graph.tp0.json`, so three quarters of a run's evidence was silently
        discarded and the file looked complete.
        """
        coords = {"tp": int(getattr(self, "rank", 0) or 0)}
        parallel = getattr(self.config, "parallel_config", None)
        if parallel is not None:
            if int(getattr(parallel, "data_parallel_size", 1) or 1) > 1:
                coords["dp"] = int(getattr(parallel, "data_parallel_rank", 0) or 0)
            if int(getattr(parallel, "pipeline_parallel_size", 1) or 1) > 1:
                coords["pp"] = int(getattr(parallel, "pipeline_parallel_rank", 0) or 0)
        if os.environ.get("ATOM_LOG_RANK_COORDS"):
            import torch.distributed as dist

            world = dist.get_rank() if dist.is_initialized() else -1
            print(f"### RANK COORDS {coords} global={world}", flush=True)
        return coords

    @property
    def _runs_real_forward(self) -> bool:
        return self._compass_config.mode in ("trace", "measure")

    def _resolve_capture_ladder(self) -> None:
        """Work out which graphs a real run would have captured, without capturing.

        `capture_sizes` starts as `[0]` and is filled in by the capture this
        method's caller just skipped, so a predicting runner would otherwise know
        nothing about the ladder -- and the ladder is what decides a decode
        step's cost, since the replay runs a padded bucket rather than the batch.
        Skipping capture must not also discard the shape of the machine being
        simulated.

        Resolving it is a bound, not a side effect of capturing: the declared
        ladder from config, narrowed to what this deployment could schedule.
        Mirrors `ModelRunner.capture_cudagraph` and reuses its bound function, so
        the two cannot disagree about which rungs exist.
        """
        from atom.model_engine.model_runner import max_schedulable_decode_bs

        try:
            sizes = sorted(self.config.capture_sizes, reverse=True)
            full_q_len = self.drafter.mtp_k + 1 if hasattr(self, "drafter") else 1
            max_bs = max_schedulable_decode_bs(
                self.config.max_num_seqs,
                self.config.max_num_batched_tokens,
                full_q_len,
            )
            self.capture_sizes = [s for s in sizes if s <= max_bs]
        except Exception as exc:  # noqa: BLE001 - never block a run over this
            # Costs accuracy on decode, not correctness: without a ladder the
            # oracle sees no bucket and falls back to whatever it does for an
            # eager step. Say so rather than leaving a silent [0].
            logger.warning(
                "ATOMCompass WARNING: could not resolve the CUDA graph capture "
                "ladder (%s); decode steps will carry no bucket and a "
                "bucket-aware oracle will have nothing to key on.", exc,
            )
            return
        logger.info(
            "ATOMCompass: simulating a deployment whose capture ladder is %s",
            sorted(self.capture_sizes),
        )
