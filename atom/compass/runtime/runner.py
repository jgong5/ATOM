"""A model runner that predicts the forward pass instead of performing it.

ATOM already resolves its runner by name (``Config.runner_qualname``), so this
class is injected without changing the engine::

    --runner-qualname atom.compass.runtime.runner.CompassModelRunner

Everything except the forward pass is inherited and therefore real: weight
loading, KV-cache sizing, and — most importantly — the scheduler, block manager
and admission logic that decide what each batch contains. Only
:meth:`CompassModelRunner.forward` is replaced, with a cost oracle's prediction
and synthesised tokens.

Two consequences worth stating plainly:

* Generated text is meaningless. Tokens exist so sequences advance and finish;
  they are not what the model would have produced.
* The predicted duration is returned to the caller rather than slept away. The
  process that owns scheduling owns the clock, because in ATOM the runner and
  the scheduler are different processes even at world size one.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os
import logging
from typing import Optional

from atom.compass.config import CompassConfig
from atom.compass.core.artifacts import rank_path
from atom.compass.core.cost.base import CostOracle, StepShape
from atom.compass.core.graph import GraphKey, OpGraph
from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.scheduler import ScheduledBatch, ScheduledBatchOutput
from atom.utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)

__all__ = ["CompassModelRunner"]


class CompassModelRunner(ModelRunner):
    """ModelRunner whose forward pass is modelled rather than executed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        logger.info(
            "ATOMCompass active: mode=%s oracle=%s",
            self._compass_config.mode,
            self._oracle.describe(),
        )
        if self._compass_config.mode == "trace":
            self._warn_if_compiled()

    # -- setup ----------------------------------------------------------------

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

    # -- the seam -------------------------------------------------------------

    def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        """Predict the step, or trace it, depending on the configured mode."""
        if self._runs_real_forward and getattr(batch, "is_dummy_run", False):
            # Warmup drives synthetic batches through this same entry point.
            # They are real forwards, so they must actually run — but they are
            # not steps a deployment performs, and counting them would spend the
            # trace budget on a dummy shape and put dummy timings in the table
            # that a cost model is fitted to.
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

        req_ids = list(batch.req_ids)
        filler = self._compass_config.filler_token_id
        token_ids = [(filler,) for _ in req_ids]

        return ScheduledBatchOutput(
            req_ids=req_ids,
            token_ids=token_ids,
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
            compass_step_seconds=cost.seconds,
        )

    def _forward_measured(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        """Run the real forward and record how long the device spent on it.

        Timed with CUDA events rather than a host-side synchronise, and the
        distinction is not a refinement — measuring with a sync made the run
        **33% slower than the same run without it** (4.33 ms per output token
        against 3.26 ms), so the table described a machine that only exists
        while being measured.

        The reason is that a serving loop overlaps host and device: while the
        device runs one step the host is already preparing the next. A sync on
        every step destroys that overlap, so what gets recorded is each step's
        *isolated latency* — and a cost model fitted to isolated latencies then
        predicts a pipelined run, over-estimating it by however much the overlap
        was worth. That was most of the residual TPOT error.

        Events are recorded on the stream and read back later, so the host never
        waits. Completed pairs are drained on subsequent steps; a few of the
        very last ones are simply not written, which costs a calibration run
        nothing.
        """
        import torch

        shape = self._describe(batch)
        if not torch.cuda.is_available():
            # No device to time: fall back to wall clock, which is exact here
            # because there is nothing asynchronous to miss.
            import time

            began = time.perf_counter()
            output = super().forward(batch)
            self._count_and_record(shape, time.perf_counter() - began, None,
                                   req_ids=list(batch.req_ids),
                                   started_at=began,
                                   decision=getattr(batch, "compass_decision", None))
            return output

        import time

        entered = time.perf_counter()
        # Stamped by the engine core on the clock that arrivals are stamped on,
        # so a step and a request sit on one timeline. Reconstructing it
        # instead, by accumulating step durations and host gaps, does not work:
        # it put 207 of 300 requests' first step *after* their first token,
        # because a gap recorded before a forward was being added after it and
        # because device time and host time are not additive when they overlap.
        started_at = getattr(batch, "compass_started_at", None)
        gap = (entered - self._last_forward_ended
               if self._last_forward_ended is not None else None)

        began = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        began.record()
        output = super().forward(batch)
        ended.record()
        self._last_forward_ended = time.perf_counter()
        # The ids are read now rather than when the pair is drained: the batch
        # is the scheduler's and does not survive the step.
        self._pending.append((shape, began, ended, gap, list(batch.req_ids),
                              started_at,
                              getattr(batch, "compass_decision", None)))
        self._drain_pending()
        return output

    def _drain_pending(self) -> None:
        """Write out every timed step whose events have completed.

        Draining by ``query()`` rather than ``synchronize()`` keeps the host off
        the critical path: a step is written once the device has finished it
        anyway, never by waiting for it.
        """
        while self._pending:
            shape, began, ended, gap, req_ids, started_at, decision = \
                self._pending[0]
            if not ended.query():
                return
            self._pending.popleft()
            self._count_and_record(shape, began.elapsed_time(ended) / 1000.0,
                                   gap, req_ids=req_ids, started_at=started_at,
                                   decision=decision)

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

    def _forward_traced(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        """Run the real forward and record the operations it performed.

        The forward is ATOM's own, so the recorded graph is what a served batch
        actually produces — attention metadata, KV state and forward context all
        established by the runner rather than reconstructed. That is the whole
        point: a graph assembled by calling the model directly is a different
        forward, and would validate nothing.

        One step is recorded, and deliberately not the first. Triton autotunes
        on a kernel's first launch, benchmarking every candidate configuration:
        recording that yields tens of thousands of launches that steady-state
        serving never performs. Which step to take is ``trace_step``.
        """
        from atom.compass.runtime.derive import record_collectives
        from atom.compass.runtime.meta import MetaOpTracer
        from atom.compass.runtime.triton_trace import TritonLaunchTracer

        self._step_index += 1
        config = self._compass_config
        # Prefills are counted separately from forwards, because which forward
        # a prefill happens to be depends on the workload -- how many decode
        # steps ran before it -- while "the second prefill" does not.
        prefilling = int(getattr(batch, "total_tokens_num_prefill", 0)) > 0
        if prefilling:
            self._prefill_index += 1
        else:
            self._decode_index += 1
        # Counted per kind rather than by forward index. Which forward a decode
        # step happens to be depends on how many prefills ran before it, so a
        # workload that warms its shapes first -- which tracing a prefill needs,
        # since Triton autotunes on a shape's first launch -- would shift every
        # index and quietly record the wrong step.
        kind = None
        if prefilling:
            if config.trace_prefill and self._prefill_index == config.trace_prefill:
                kind = "prefill"
        elif self._decode_index == config.trace_step:
            kind = "decode"
        if kind is None:
            return super().forward(batch)
        # A graph per traced step, not one accumulated across them: a decode
        # graph and a prefill graph describe different work and merging them
        # would describe neither.
        self._graph = OpGraph()

        timing = None
        if self._compass_config.op_timings_out:
            from atom.compass.runtime.op_timing import OpTimingTracer

            timing = OpTimingTracer()
        ops = MetaOpTracer(graph=self._graph, topology=self._topology())
        triton = TritonLaunchTracer(graph=self._graph)
        # Under simulated TP the collective is replaced by a passthrough, so it
        # never dispatches and never gets recorded — a TP graph captured on one
        # device would show no communication at all. A no-op on a real
        # multi-device run, where the collective dispatches and is recorded once.
        collectives = record_collectives(self._graph)
        # Ground truth for the activation term, for this exact step. The
        # engine's `peak_torch` belongs to the warmup prefill, whose shape is
        # nobody's choice and is rarely the traced one -- so checking a
        # graph-derived activation peak against it compares two shapes. Read
        # here instead, around the step whose graph is about to be written, and
        # the two are the same work by construction. Resetting the peak is safe
        # only because `get_num_blocks` has long since run and recorded its own.
        resident = self._reset_activation_peak()
        try:
            with collectives, triton, ops:
                if timing is None:
                    output = super().forward(batch)
                else:
                    # Innermost, so it sees the same dispatches the graph does
                    # and their indices line up.
                    with timing:
                        output = super().forward(batch)
        except BaseException:
            # Deliberately do not write a graph here. A forward that died
            # part-way leaves a well-formed but truncated recording, and a
            # truncated graph is worse than none: it costs out at a fraction of
            # the model while looking like a complete artifact.
            self._traced_steps += 1
            logger.error(
                "ATOMCompass: forward failed after %d operators; no graph "
                "written. A partial trace is not a usable artifact.",
                len(self._graph),
            )
            raise
        self._traced_steps += 1
        self._activation_peak = self._read_activation_peak(resident)
        self._resident_before = resident
        self._allocated_curve = [ops.allocated.get(i)
                                 for i in range(len(self._graph.ops))]
        self._stamp_deaths(ops)
        self._write_graph(batch, kind)
        if timing is not None:
            self._write_op_timings(timing)
        return output

    def _stamp_deaths(self, ops) -> None:
        """Write each operator's observed death onto the operator itself.

        One entry per output, because a fused add-and-norm's two outputs have
        very different lives -- the normed activation dies into the next gemm,
        the new residual carries to the end of the block -- and one death for
        the pair holds an extra tensor per layer.

        A death is observed long after the operator that caused it is recorded,
        so it cannot be filled in as the trace runs. Stamped as late as
        possible -- immediately before the graph is written -- because by then
        the forward has returned and the locals holding its intermediates are
        gone, which is when most of the finalizers fire. An output still alive
        at that point keeps `dies_at` at -1 and is treated as living to the end
        of the step, which is what it did.
        """
        import dataclasses

        by_operator: dict = {}
        for (producer, position), death in ops.deaths.items():
            if 0 <= producer < len(self._graph.ops):
                by_operator.setdefault(producer, {})[position] = int(death)
        for producer, positions in by_operator.items():
            op = self._graph.ops[producer]
            width = max(len(op.output_shapes), max(positions) + 1)
            self._graph.ops[producer] = dataclasses.replace(
                op, dies_at=tuple(positions.get(p, -1) for p in range(width)))

    def _reset_activation_peak(self) -> Optional[int]:
        """Start this step's high-water mark, and say what was already held."""
        try:
            import torch

            resident = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            return resident
        except Exception as exc:  # noqa: BLE001 - never fail a trace over a number
            logger.debug("ATOMCompass: no activation peak for this step: %s", exc)
            return None

    def _read_activation_peak(self, resident: Optional[int]) -> Optional[int]:
        """How far above the resident baseline this step's allocation went."""
        if resident is None:
            return None
        try:
            import torch

            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            return max(int(peak) - resident, 0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ATOMCompass: no activation peak for this step: %s", exc)
            return None

    def _write_op_timings(self, timing) -> None:
        """Write what each operator cost, beside the graph it belongs to.

        Reports whether the operators account for the region containing them,
        because that is the question the artifact exists to answer and a reader
        should not have to compute it to find out the answer is no.
        """
        import json

        path = self._compass_config.op_timings_out
        if any(size > 1 for size in self._topology().values()):
            path = self._rank_path(path, self._rank_coords())
        summary = timing.summary()
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({
                    "version": 1,
                    "provenance": {
                        "source": "trace",
                        "eager": True,
                        "note": "eager device time; production replays a graph",
                    },
                    "summary": summary,
                    "operators": [t.as_dict() for t in timing.timings],
                }, fh, indent=1)
        except OSError as exc:
            logger.warning("ATOMCompass WARNING: could not write op timings to "
                           "%s: %s", path, exc)
            return
        logger.info(
            "ATOMCompass: %d operators timed, summing to %.3fms against a "
            "%.3fms region (%.1f%% covered); written to %s",
            summary["operators"], summary["sum_of_operators"] * 1000,
            summary["region"] * 1000, 100 * summary["covered"], path,
        )

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

    def _write_graph(self, batch: ScheduledBatch, kind: str = "decode") -> None:
        path = self._compass_config.graph_out
        if not path:
            return
        if kind == "prefill":
            from atom.compass.core.artifacts import kind_path

            path = kind_path(path, kind)
        shape = self._describe(batch)
        topology = self._topology()
        coords = self._rank_coords()
        self._graph.key = GraphKey.of(
            model_id=str(getattr(self.config, "model", "unknown")),
            topology=topology,
            rank_coords=coords,
            batch_signature=shape.num_scheduled_tokens,
        )
        if any(size > 1 for size in topology.values()):
            path = self._rank_path(path, coords)
        self._graph.provenance = {
            "source": "capture",
            "device": "cuda",
            "compilation_level": self._compilation_level(),
            "trace_step": self._step_index,
            # The shape this graph describes, so a reader can tell which step it
            # is a graph *of* without inferring it from the operators. An oracle
            # holding several needs to choose between them.
            "shape": {
                "num_scheduled_tokens": list(shape.num_scheduled_tokens),
                "context_lens": list(shape.context_lens),
                "num_prefill_tokens": shape.num_prefill_tokens,
                "capture_bucket": shape.capture_bucket,
            },
            # What the allocator actually went above its resident baseline for
            # this step. The graph is the input to the modelled activation
            # term, so its own measurement belongs beside it -- a derivation
            # and its ground truth in one artifact, at one shape.
            "activation_peak_bytes": getattr(self, "_activation_peak", None),
            # ...and the whole curve it is the maximum of, one reading per
            # operator. A peak checked against a peak says the model is wrong;
            # a curve checked against a curve says where.
            "allocated_after_bytes": getattr(self, "_allocated_curve", None),
            "allocated_before_bytes": getattr(self, "_resident_before", None),
        }
        self._warn_if_incomplete()
        try:
            self._graph.save(path)
        except OSError as exc:
            logger.warning("ATOMCompass WARNING: could not write graph to %s: %s", path, exc)
            return
        logger.info(
            "ATOMCompass: traced %d operators (%d distinct) -> %s",
            len(self._graph), len(self._graph.op_names()), path,
        )

    def _compilation_level(self) -> Optional[int]:
        compilation = getattr(self.config, "compilation_config", None)
        return getattr(compilation, "level", None)

    def _warn_if_compiled(self) -> None:
        """Note that a compiled graph is not comparable to an uncompiled one.

        Compilation is on by default, so it is the configuration that gets
        deployed and therefore the one worth modelling. Inductor's kernels are
        traced (``CachingAutotuner`` is intercepted alongside ``JITFunction``),
        so a compiled capture is complete — but it is a different graph, and
        legitimately so:

        * fused compute appears as one ``inductor::`` kernel where the
          uncompiled graph has the operators it replaced
        * views and allocations — ``split_with_sizes``, ``empty`` — do not
          appear at all, because inductor resolves them into offsets and a
          buffer plan rather than executing them

        On Qwen3-0.6B that is 330 operators compiled against 386 uncompiled,
        with identical compute in both: 283 operators either way. Neither is
        wrong; they describe different configurations, and a derivation can only
        be compared against a capture taken at its own level.

        ``--enforce-eager`` does not turn compilation off. It disables CUDA
        graphs; compilation is ``--level``.
        """
        level = self._compilation_level()
        if level:
            logger.info(
                "ATOMCompass: tracing at compilation level %d. Inductor kernels "
                "are traced, but fused compute appears as one operator and "
                "views and allocations do not appear at all. Compare only "
                "against a graph derived at the same level.",
                level,
            )

    def _warn_if_incomplete(self) -> None:
        """Sanity-check the recording against the model's depth.

        Attention runs once per layer, so the number of attention operators
        should match the layer count. A graph holding a handful of layers for a
        deep model is truncated, and nothing downstream would notice: it is
        structurally valid and merely wrong.
        """
        hf = getattr(self.config, "hf_config", None)
        layers = getattr(hf, "num_hidden_layers", None)
        if layers is None:
            text_config = getattr(hf, "text_config", None)
            layers = getattr(text_config, "num_hidden_layers", None)
        if not layers:
            return
        counts = self._graph.counts()
        seen = sum(n for name, n in counts.items() if "attention" in name.lower())
        if seen and seen < layers:
            logger.warning(
                "ATOMCompass WARNING: graph holds %d attention operators for a %d-layer "
                "model. It looks truncated; do not calibrate against it.",
                seen, layers,
            )

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

    # -- work whose meaning depends on the mode --------------------------------
    #
    # Only `predict` replaces the forward pass. `trace` and `measure` both run
    # the real thing, so anything skipped for them is skipped from a real run,
    # and the artifact then describes a machine configured unlike the
    # deployment it is meant to stand for.

    @property
    def _runs_real_forward(self) -> bool:
        return self._compass_config.mode in ("trace", "measure")

    def capture_cudagraph(self):
        """Capture CUDA graphs unless there is a reason not to — per mode.

        Skipping this unconditionally is what made the first end-to-end
        validation wrong by 800%. A measure run executed eagerly while the
        deployment it modelled replayed a captured graph, so the oracle was
        fitted to a machine running 8.9x slower than the one it predicts
        (Qwen3-0.6B decode: 28.78 ms eager against 3.24 ms replayed). The oracle
        reproduced its training data to about 1%; the training data was taken
        from the wrong configuration.

        So:

        * ``measure`` captures for real. Timings have to come from the path that
          runs in production, and in production that path is the replay.
        * ``predict`` skips: no kernels run, so there is nothing to capture.
        * ``trace`` skips as well, but for the opposite reason to ``predict``.
          A replay is a single opaque submission, so a traced step would record
          nothing at all. The operator sequence has to come from eager
          execution; what it costs has to come from a measure run.

        ``engine_core`` calls this across the worker boundary with
        ``wait_out=True`` and unpacks three values, so a skip still has to
        return the triple. Returning ``None`` does not skip the capture — it
        kills the worker mid-reply and hangs the parent on a broadcast that
        never arrives, naming neither CUDA graphs nor Compass.
        """
        if self._compass_config.mode == "measure":
            with self._measured_graph_pool():
                result = super().capture_cudagraph()
            # Priced here rather than after warmup, because warmup runs while
            # the runner is still being built -- before the KV cache exists. An
            # operator that walks paged KV cannot be called without one, and
            # attention is exactly such an operator. By this point the kernels
            # are registered and autotuned, the cache is allocated, and the
            # graphs are captured.
            self._run_microbenchmark()
            return result
        if self._compass_config.mode == "trace":
            logger.info(
                "ATOMCompass: skipping CUDA graph capture so the forward stays "
                "traceable — a replay is one opaque submission and would record "
                "nothing. The graph is the eager operator sequence; take its "
                "cost from a measure run."
            )
        else:
            logger.debug("ATOMCompass: skipping CUDA graph capture")
            self._resolve_capture_ladder()
        return 0.0, [], 0

    def _build_and_load_model(self, model_class):
        """Load the weights, and read how many bytes of them are resident.

        `peak_torch` is a single number covering the weights and the peak
        activations both, so a budget derived from it can be right in total
        while both its terms are wrong. The allocator read here -- after the
        weights are resident and before any forward has run -- is the only
        moment at which the weights are separable, and it costs one call.
        """
        import torch

        out = super()._build_and_load_model(model_class)
        try:
            self._weights_bytes = int(torch.cuda.memory_allocated())
        except Exception as exc:  # noqa: BLE001 - never fail a run over a record
            logger.warning("ATOMCompass WARNING: could not read the weight "
                           "bytes (%s); the term stays folded into peak_torch",
                           exc)
            self._weights_bytes = None
        self._parameter_bytes = self._resident_parameter_bytes()
        self._buffer_bytes = self._resident_parameter_bytes(buffers_only=True)
        self._buffer_breakdown = self._resident_buffer_breakdown()
        return out

    def _resident_buffer_breakdown(self) -> Optional[list]:
        """What the model's buffers are, by name, largest first.

        Aggregated by the last two components of the name, so the list does not
        grow with the layer count and one entry stands for the whole model's
        worth of a thing.
        """
        import torch

        model = getattr(self, "model", None)
        if model is None:
            return None
        try:
            seen, totals = set(), {}
            for name, tensor in model.named_buffers():
                if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                    continue
                storage = tensor.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                if key in seen:
                    continue
                seen.add(key)
                kind = ".".join(str(name).split(".")[-2:])
                count, total, shape = totals.get(kind, (0, 0, None))
                totals[kind] = (count + 1, total + storage.nbytes(),
                                shape or list(tensor.shape))
            ranked = sorted(totals.items(), key=lambda kv: -kv[1][1])[:8]
            return [{"name": kind, "count": count, "bytes": total,
                     "shape": shape, "dtype": None}
                    for kind, (count, total, shape) in ranked]
        except Exception as exc:  # noqa: BLE001 - never fail a run over a record
            logger.debug("ATOMCompass: no buffer breakdown: %s", exc)
            return None

    def _resident_parameter_bytes(self, buffers_only: bool = False) -> Optional[int]:
        """What the model's own parameters and buffers weigh.

        `buffers_only` takes the buffers alone. They are the part the
        checkpoint does not contain -- rotary tables and the like, computed at
        init -- so the weight term derived from a checkpoint can only ever be
        the rest, and separating them says whether a shortfall is a sharding
        mistake or simply a tensor that was never in the file.

        `memory_allocated()` after loading is the weights *and* anything the
        loader still holds, and the two are not the same number -- at TP=2 they
        differ by more than the weights themselves. Asking the model rather
        than the allocator gives the term its exact ground truth, and the
        difference between the two is then a residue that can be named instead
        of being charged to the weights.

        Storage is counted once per tensor, because a tied head and its
        embedding are two parameters over one allocation.
        """
        import torch

        model = getattr(self, "model", None)
        if model is None:
            return None
        try:
            seen, total = set(), 0
            resident = (list(model.buffers()) if buffers_only
                        else list(model.parameters()) + list(model.buffers()))
            for tensor in resident:
                if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                    continue
                storage = tensor.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                if key in seen:
                    continue
                seen.add(key)
                total += storage.nbytes()
            return int(total)
        except Exception as exc:  # noqa: BLE001 - never fail a run over a record
            logger.debug("ATOMCompass: no parameter bytes: %s", exc)
            return None

    @contextlib.contextmanager
    def _measured_graph_pool(self):
        """What capture actually costs, recorded beside what it was estimated at.

        The engine reserves for the pool from an estimate (`0.2 x` the peak
        activations under manual capture) and that estimate is 8-19x under what
        capture then takes -- 0.020 GB against 0.390 GB measured at TP=1. It is
        0.2% of a 192 GB card, so nothing here has depended on it, which is
        precisely why it needs an artifact: a term nobody can see is a term
        nobody notices being wrong until a configuration is tight.

        The pool is the *reserved* delta, not the allocated one. A captured
        graph pins its intermediates for replay, so the segments the allocator
        had to create are what the configuration must budget for.

        Recorded after `get_num_blocks` has already written its record, so the
        record is rewritten rather than extended -- capture cannot happen before
        the KV cache exists, and the budget cannot be computed after it does.
        """
        import torch

        try:
            before = (torch.cuda.memory_reserved(), torch.cuda.memory_allocated())
        except Exception:  # noqa: BLE001 - never fail a run over a measurement
            yield
            return
        try:
            yield
        finally:
            try:
                self._graph_pool = {
                    "reserved": max(0, int(torch.cuda.memory_reserved()
                                           - before[0])),
                    "allocated": max(0, int(torch.cuda.memory_allocated()
                                            - before[1])),
                    "capture_sizes": [int(b) for b in
                                      sorted(getattr(self, "capture_sizes", []))
                                      if b],
                    "graphs": len(getattr(self, "graphs", []) or ()),
                }
                self._rewrite_memory_with_pool()
            except Exception as exc:  # noqa: BLE001
                logger.debug("ATOMCompass: no graph pool measurement: %s", exc)

    def _rewrite_memory_with_pool(self) -> None:
        """Add the measured pool to the record `get_num_blocks` already wrote."""
        path = self._compass_config.memory_out
        if not path or not getattr(self, "_graph_pool", None):
            return
        coords = self._rank_coords()
        if any(size > 1 for size in self._topology().values()):
            path = self._rank_path(path, coords)
        try:
            with open(path, encoding="utf-8") as fh:
                record = json.load(fh)
            record["graph_pool"] = self._graph_pool
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=1)
        except (OSError, ValueError) as exc:
            logger.warning("ATOMCompass WARNING: could not record the measured "
                           "graph pool in %s: %s", path, exc)
            return
        logger.info(
            "ATOMCompass: CUDA graph capture reserved %.3f GB over %d buckets "
            "(the engine budgeted %.3f GB) -> %s",
            self._graph_pool["reserved"] / 2**30,
            len(self._graph_pool["capture_sizes"]),
            (record.get("readings") or {}).get("cudagraph_overhead", 0) / 2**30,
            path)

    def get_num_blocks(self) -> dict:
        """Size the KV cache, and record what the budget was made of.

        The engine measures four things off the device, derives a byte budget,
        and hands the arithmetic to `plan_pools` -- which is pure CPU and already
        correct for a hybrid. Take the device away and only the four readings are
        missing, so those are what an artifact has to hold.

        They are read here rather than parsed back out of the log line, and read
        *before* delegating rather than after: `super()` allocates nothing, so
        these are the same numbers it will see, while afterwards the KV cache
        exists and `mem_get_info` says something else entirely.

        Only the raw readings are recorded, never the derived budget. The
        arithmetic from readings to block count is the engine's, it is already
        CPU-only, and copying it here would be one more thing to drift.
        """
        import torch

        readings = {}
        try:
            free, total = torch.cuda.mem_get_info()
            stats = torch.cuda.memory_stats()
            readings = {
                # The four the device supplies, and nothing computed from them.
                "total": int(total),
                # Set by whatever else is on the box, not by this configuration.
                # A budget in which `free` was the binding term describes the
                # neighbours; record it so a reader can refuse such a budget.
                "free": int(free),
                "peak_torch": int(max(stats["allocated_bytes.all.peak"],
                                      stats["allocated_bytes.all.current"])),
                # `peak_torch` is weights, persistent buffers and peak
                # activations summed. These two split it: what the allocator
                # held once the weights were in, and what it still holds now
                # the profiling forward has finished. Recorded raw, so the
                # split is the reader's arithmetic and not a stored derivation.
                "weights_torch": getattr(self, "_weights_bytes", None),
                # The model's own parameters and buffers. `weights_torch` is
                # this plus whatever the loader has not let go of; keeping them
                # apart is what stops the residue being charged to the weights.
                "parameter_bytes": getattr(self, "_parameter_bytes", None),
                # ...of which this much is buffers, which the checkpoint does
                # not contain. A weight term derived from a checkpoint can only
                # ever account for the rest.
                "buffer_bytes": getattr(self, "_buffer_bytes", None),
                # ...and what they are. A formula for them was guessed from one
                # model (rotary tables: positions x head_dim x 2 matched the
                # 0.6B's 10.0 MiB exactly) and was 4x wrong on the second, so
                # what they are is recorded rather than inferred.
                "buffer_breakdown": getattr(self, "_buffer_breakdown", None),
                "current_torch": int(stats["allocated_bytes.all.current"]),
                "non_torch": int(max((total - free)
                                     - torch.cuda.memory_reserved(), 0)),
                "cudagraph_overhead": int(self._estimate_cudagraph_overhead()),
            }
        except Exception as exc:  # noqa: BLE001 - never fail a run over a record
            logger.warning("ATOMCompass WARNING: could not read the memory "
                           "terms (%s); none recorded", exc)

        with self._recorded_readings(readings):
            result = super().get_num_blocks()
        self._write_memory(readings, result)
        return result

    @contextlib.contextmanager
    def _recorded_readings(self, live: dict):
        """Run the engine's budget arithmetic on recorded readings.

        The readings are substituted, not the arithmetic. `super()` computes the
        budget from four device calls; each is made to return what an earlier
        run recorded, and everything downstream -- the utilization budget, the
        safety margin, the `min(budget, free)` clamp, `plan_pools` over the
        sub-pool specs -- is the engine's own and unchanged. Copying that
        arithmetic here to feed it numbers directly would be one more thing to
        drift out of step with the engine.

        A no-op unless `--compass-memory-in` names a record matching this
        configuration exactly.
        """
        import torch

        modelled = self._modelled_readings()
        if modelled is not None:
            was = self._substitute(modelled)
            try:
                yield
            finally:
                self._restore(was)
            return

        source = self._recorded_memory()
        config = self._memory_config()
        readings = source.readings_for(config) if source else None
        expected = self._expected_non_torch()
        if readings is None:
            if source is not None:
                logger.warning("ATOMCompass WARNING: sizing from this device: "
                               "%s", source.refusal(config, expected))
            yield
            return
        spread = source.rank_disagreement(config)
        if spread:
            logger.warning(
                "ATOMCompass WARNING: the ranks of this record disagree about "
                "`non_torch` by %.0f MiB. They do the same work, so a spread is "
                "the neighbours arriving on some cards and not others -- this "
                "budget is partly the box's.", spread / 2**20)
        refusal = source.refusal(config, expected)
        if refusal:
            logger.warning("ATOMCompass WARNING: sizing from this device: %s",
                           refusal)
            yield
            return

        was = self._substitute({
            "total": readings.total, "free": readings.free,
            "peak_torch": readings.peak_torch, "non_torch": readings.non_torch,
            "cudagraph_overhead": readings.cudagraph_overhead})
        logger.info("ATOMCompass: sizing from a recorded budget, not this "
                    "device (peak_torch %.2f GB, non_torch %.2f GB)",
                    readings.peak_torch / 2**30, readings.non_torch / 2**30)
        try:
            yield
        finally:
            self._restore(was)

    def _substitute(self, readings: dict) -> tuple:
        """Make the four device calls return these readings, and say what was.

        The readings are substituted, never the arithmetic -- see
        `_recorded_readings`. Shared by the recorded and the modelled paths
        because the substitution is the same either way; only where the numbers
        came from differs.
        """
        import torch

        # `non_torch` is `(total - free) - reserved`, so the reserved figure
        # that reproduces it is what to report.
        reserved = max(0, (readings["total"] - readings["free"])
                       - readings["non_torch"])
        stats = {"allocated_bytes.all.peak": readings["peak_torch"],
                 "allocated_bytes.all.current": readings["peak_torch"]}
        was = (torch.cuda.mem_get_info, torch.cuda.memory_stats,
               torch.cuda.memory_reserved, self._estimate_cudagraph_overhead)
        torch.cuda.mem_get_info = lambda *a, **k: (readings["free"],
                                                   readings["total"])
        torch.cuda.memory_stats = lambda *a, **k: stats
        torch.cuda.memory_reserved = lambda *a, **k: reserved
        self._estimate_cudagraph_overhead = (
            lambda *a, **k: readings["cudagraph_overhead"])
        return was

    def _restore(self, was: tuple) -> None:
        import torch

        (torch.cuda.mem_get_info, torch.cuda.memory_stats,
         torch.cuda.memory_reserved, self._estimate_cudagraph_overhead) = was

    def _warmup_tokens(self) -> int:
        """The token count of the prefill that sets `peak_torch`.

        `warmup_model` resets the allocator's high-water mark and runs one
        dummy prefill, so the peak belongs to that shape and no other. Mirrored
        here because a predicting run skips warmup entirely -- there is nothing
        to measure, which is the point.
        """
        config = self.config
        budget = int(getattr(config, "max_num_batched_tokens", 0) or 0)
        length = int(getattr(config, "max_model_len", 0) or 0)
        if not (budget and length):
            return 0
        seqs = max(1, min(budget // length,
                          int(getattr(config, "max_num_seqs", 1) or 1)))
        return seqs * max(1, min(length, budget // seqs))

    def _modelled_readings(self) -> Optional[dict]:
        """The five readings derived from a profile, or None if none was given.

        This is what makes a configuration nobody has run sizable: no term here
        came off a device. `total` is the target card's capacity, which is the
        one thing that has to be supplied, and `free` is modelled as a clean
        box rather than as whatever the neighbours left.
        """
        path = (self._compass_config.memory_model or "").strip()
        if not path:
            return None
        if getattr(self, "_modelled", "unset") != "unset":
            return self._modelled
        self._modelled = None
        try:
            from atom.compass.core.memory_model import (
                activation_bytes_at, modelled_readings)

            with open(path, encoding="utf-8") as fh:
                profile = json.load(fh)
            calibration = None
            if profile.get("calibration"):
                with open(profile["calibration"], encoding="utf-8") as fh:
                    calibration = json.load(fh)
            activation = 0
            if profile.get("graph"):
                with open(profile["graph"], encoding="utf-8") as fh:
                    activation = activation_bytes_at(json.load(fh),
                                                     self._warmup_tokens())
            total = int(profile.get("total") or 0)
            if not total:
                import torch

                total = int(torch.cuda.mem_get_info()[1])
                logger.info("ATOMCompass: the profile names no card capacity, "
                            "so this device's %.1f GB is used for `total`",
                            total / 2**30)
            self._modelled = modelled_readings(
                total_bytes=total,
                world_size=int(profile.get("world_size") or 1),
                parameters=int(profile["parameters"]),
                buffers=int(profile.get("buffers") or 0),
                activation_bytes=activation, calibration=calibration,
                enforce_eager=bool(getattr(self.config, "enforce_eager", False)))
            logger.info(
                "ATOMCompass: sizing from a modelled budget, not from any "
                "device (peak_torch %.2f GB, non_torch %.2f GB, activations "
                "%.2f GB over %d warmup tokens)",
                self._modelled["peak_torch"] / 2**30,
                self._modelled["non_torch"] / 2**30, activation / 2**30,
                self._warmup_tokens())
        except Exception as exc:  # noqa: BLE001 - never fail a run over a model
            logger.warning("ATOMCompass WARNING: could not model the memory "
                           "budget from %s (%s); sizing from this device",
                           path, exc)
        return self._modelled

    def _expected_non_torch(self) -> Optional[int]:
        """What the collective terms say this width should hold outside torch.

        Only a yardstick for the guard, never a term in a recorded budget --
        the point of reading a record is to use what was measured.
        """
        try:
            from atom.compass.core.memory_model import non_torch_bytes

            world = 1
            for size in self._topology().values():
                world *= max(1, int(size))
            return non_torch_bytes(world)
        except Exception as exc:  # noqa: BLE001 - a guard must not fail a run
            logger.debug("ATOMCompass: no expected non_torch: %s", exc)
            return None

    def _recorded_memory(self):
        """The memory records this run was given, loaded once."""
        paths = (self._compass_config.memory_in or "").strip()
        if not paths:
            return None
        if getattr(self, "_memory_source", None) is None:
            from atom.compass.core.memory import RecordedMemory

            found = [q for p in paths.split(",") for q in
                     sorted(glob.glob(p.strip())) if q]
            try:
                self._memory_source = RecordedMemory(found)
            except Exception as exc:  # noqa: BLE001 - never fail a run over it
                logger.warning("ATOMCompass WARNING: could not read %s: %s",
                               paths, exc)
                self._memory_source = None
        return self._memory_source

    def _memory_config(self) -> dict:
        """This configuration, keyed the way a record is."""
        config = self.config
        return {
            "model": config.model,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_num_seqs": config.max_num_seqs,
            "max_model_len": config.max_model_len,
            "kv_cache_dtype": config.kv_cache_dtype,
            "block_size": config.kv_cache_block_size,
            "topology": self._topology(),
            "rank_coords": self._rank_coords(),
        }

    def _write_memory(self, readings: dict, result: dict) -> None:
        path = self._compass_config.memory_out
        if not path or not readings:
            return
        coords = self._rank_coords()
        if any(size > 1 for size in self._topology().values()):
            path = self._rank_path(path, coords)
        config = self.config
        record = {
            "version": 1,
            "readings": readings,
            # What the engine made of them, so a modelled budget can be checked
            # against the decision it actually drove rather than against bytes.
            "blocks": {
                "num_kvcache_blocks": result.get("num_kvcache_blocks"),
                "pool_entries": result.get("pool_entries"),
                "pool_entries_per_req": result.get("pool_entries_per_req"),
            },
            "config": {
                "model": str(getattr(config, "model", "unknown")),
                "gpu_memory_utilization": float(
                    getattr(config, "gpu_memory_utilization", 0.0) or 0.0),
                "max_num_seqs": int(getattr(config, "max_num_seqs", 0) or 0),
                "max_model_len": getattr(config, "max_model_len", None),
                # Not part of the key -- two runs differing only in this get
                # the same readings. Recorded because it, with max_model_len,
                # is what sets the shape of the warmup prefill that `peak_torch`
                # belongs to, and a reader cannot check that term without it.
                "max_num_batched_tokens": getattr(
                    config, "max_num_batched_tokens", None),
                "kv_cache_dtype": str(getattr(config, "kv_cache_dtype", "")),
                "block_size": getattr(config, "kv_cache_block_size", None),
                "topology": self._topology(),
                "rank_coords": dict(coords),
            },
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=1)
        except OSError as exc:
            logger.warning("ATOMCompass WARNING: could not write memory terms "
                           "to %s: %s", path, exc)
            return
        logger.info("ATOMCompass: recorded the memory budget terms -> %s", path)

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

    def _run_microbenchmark(self) -> None:
        """Price the kernels this deployment just warmed up.

        Placed after CUDA graph capture, the first moment every condition
        holds: the operators are registered (``aiter`` does it lazily, on first
        call, in this process), they are autotuned for the shapes in use, and
        the KV cache has been allocated. Warmup satisfies the first two but runs
        during the runner construction, before any KV cache exists -- and an
        operator that walks paged KV cannot be called without one. In the parent
        process none of it holds, because the model runs here.
        """
        config = self._compass_config
        if not (config.bench_graph and config.bench_out):
            return
        from atom.compass.runtime.microbench import price_graph

        out = config.bench_out
        if any(size > 1 for size in self._topology().values()):
            out = self._rank_path(out, self._rank_coords())
        logger.info("ATOMCompass: pricing kernels from %s ...", config.bench_graph)
        try:
            result = price_graph(config.bench_graph,
                                 iters=config.bench_iters,
                                 cache=config.bench_cache)
        except OSError as exc:
            logger.warning("ATOMCompass WARNING: could not read %s: %s",
                           config.bench_graph, exc)
            return
        try:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=1)
        except OSError as exc:
            logger.warning("ATOMCompass WARNING: could not write %s: %s",
                           out, exc)
            return
        cov = result["coverage"]
        logger.info(
            "ATOMCompass: priced %d of %d signatures, covering %d of %d "
            "operators (%.1f%%) -> %s",
            cov["signatures_priced"], cov["signatures"],
            cov["operators_priced"], cov["operators"],
            100 * cov["fraction_of_operators"], out,
        )
        # A price taken outside a graph carries per-launch overhead the graph
        # would have amortised, so it is not the same kind of number as the
        # rest and should not be read as one silently.
        fell_back = sum(1 for e in result["prices"].values()
                        if e.get("cache") == "over"
                        and config.bench_cache == "graph")
        if fell_back:
            logger.info(
                "ATOMCompass: %d of those could not be graph-captured and were "
                "timed back-to-back instead", fell_back)

    def warmup_model(self) -> None:
        """Warm up for real whenever the forward is real.

        Warmup is where Triton autotunes and the allocator settles. Skipping it
        does not avoid that cost, it relocates it into the first measured step —
        which is how a 7.5 s prefill came to sit in a timing table beside a
        0.03 s one.
        """
        if self._runs_real_forward:
            return super().warmup_model()
        logger.debug("ATOMCompass: skipping model warmup")
