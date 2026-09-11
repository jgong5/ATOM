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
import numpy as np

from atom.compass.config import CompassConfig
from atom.compass.core.artifacts import rank_path
from atom.compass.core.cost.base import CostOracle, StepShape
from atom.compass.core.graph import GraphKey, OpGraph
from atom.compass.runtime.predict import CompassPredictMixin
from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.scheduler import ScheduledBatch, ScheduledBatchOutput
from atom.utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)

__all__ = ["CompassModelRunner"]


class CompassModelRunner(CompassPredictMixin, ModelRunner):
    """ModelRunner whose forward pass is modelled rather than executed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_compass_state()


    # -- setup ----------------------------------------------------------------




    # -- the seam -------------------------------------------------------------


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
            output = ModelRunner.forward(self, batch)
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
        # Opened for this forward only, so the event pairs recorded by the
        # overrides below belong to a step the table will have a row for, and
        # the same methods called during capture or a dummy run record nothing.
        self._subspans = {}
        began.record()
        try:
            output = ModelRunner.forward(self, batch)
        finally:
            # Closed even if the forward raises. A window left open would still
            # be open during the next CUDA graph capture, and the overrides
            # would then record their events *into* the graph.
            spans, self._subspans = self._subspans, None
        ended.record()
        self._last_forward_ended = time.perf_counter()
        # The ids are read now rather than when the pair is drained: the batch
        # is the scheduler's and does not survive the step.
        self._pending.append((shape, began, ended, gap, list(batch.req_ids),
                              started_at,
                              getattr(batch, "compass_decision", None), spans))
        self._drain_pending()
        return output

    # -- sub-spans ------------------------------------------------------------
    #
    # The outer pair above spans the whole of ``ModelRunner.forward``: input
    # preparation and its H2D staging, then ``run_model`` (the body and
    # ``compute_logits``), then ``postprocess`` (the sampler, any logprobs, the
    # TP broadcast and the sampled-id enqueue) -- plus whatever device idle the
    # host leaves inside it. A composed prediction that covers only the body and
    # the head is a strict subset of that, so comparing the two answers a
    # different question than it appears to.
    #
    # These inner pairs make the subset measurable. They are recorded on the
    # same stream and drained the same way, never synchronised: a sync here
    # would reintroduce exactly the 33% distortion the docstring above exists to
    # avoid, and would do it to the number the model is fitted against.

    def _timed_span(self, name: str, fn, *args, **kwargs):
        """Run one region of the forward inside its own event pair."""
        import torch

        spans = getattr(self, "_subspans", None)
        if spans is None:
            return fn(*args, **kwargs)
        began = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        began.record()
        out = fn(*args, **kwargs)
        ended.record()
        spans[name] = (began, ended)
        return out

    def run_model(self, *args, **kwargs):
        return self._timed_span("run_model", ModelRunner.run_model,
                                self, *args, **kwargs)

    def postprocess(self, *args, **kwargs):
        return self._timed_span("postprocess", ModelRunner.postprocess,
                                self, *args, **kwargs)

    def _drain_pending(self) -> None:
        """Write out every timed step whose events have completed.

        Draining by ``query()`` rather than ``synchronize()`` keeps the host off
        the critical path: a step is written once the device has finished it
        anyway, never by waiting for it.
        """
        while self._pending:
            shape, began, ended, gap, req_ids, started_at, decision, spans = \
                self._pending[0]
            if not ended.query():
                return
            self._pending.popleft()
            # The outer `ended` is the last event of the step, so an inner pair
            # that was recorded at all has completed by now.
            sub = {name: b.elapsed_time(e) / 1000.0
                   for name, (b, e) in (spans or {}).items()}
            self._count_and_record(shape, began.elapsed_time(ended) / 1000.0,
                                   gap, req_ids=req_ids, started_at=started_at,
                                   decision=decision, spans=sub)



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
            return ModelRunner.forward(self, batch)
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
                    output = ModelRunner.forward(self, batch)
                else:
                    # Innermost, so it sees the same dispatches the graph does
                    # and their indices line up.
                    with timing:
                        output = ModelRunner.forward(self, batch)
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





    # -- work whose meaning depends on the mode --------------------------------
    #
    # Only `predict` replaces the forward pass. `trace` and `measure` both run
    # the real thing, so anything skipped for them is skipped from a real run,
    # and the artifact then describes a machine configured unlike the
    # deployment it is meant to stand for.


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
            # The ladder and what it cost to build, for a replay that will
            # report the same rungs without capturing anything. `capture_bucket`
            # keys the cost table, so a replay reporting a different rung than
            # the deployment would look up a different row.
            cap_cost, sizes, pool_bytes = result
            self._write_replay_target(graph={
                "capture_seconds": float(cap_cost or 0.0),
                "capture_sizes": list(sizes or []),
                "pool_bytes": int(pool_bytes or 0),
            })
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
        # Recorded even here, where nothing was captured: the ladder is what
        # `capture_bucket` keys on, so a replay needs it whatever produced it.
        # `capture_seconds` is 0 and means it -- this run did not pay for a
        # capture, and an amortisation claim built on this target must not
        # pretend otherwise.
        self._write_replay_target(graph={
            "capture_seconds": 0.0,
            "capture_sizes": list(getattr(self, "capture_sizes", None) or []),
            "pool_bytes": 0,
        })
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
        self._write_replay_target(blocks=result)
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

    def _hardware_identity(self) -> dict:
        """Which device this ran on, in the terms a replay needs to name it.

        `arch` is `gcnArchName` verbatim, suffixes and all, because that is what
        the device said; AITER trims it the same way whatever the source. A
        failure here is recorded as such rather than guessed at -- a replay
        would rather refuse than adopt this host's architecture by default.
        """
        import torch

        out: dict = {}
        try:
            props = torch.cuda.get_device_properties(0)
            out["arch"] = str(getattr(props, "gcnArchName", "") or "")
            out["device_name"] = str(getattr(props, "name", "") or "")
            out["device_count"] = int(torch.cuda.device_count())
            out["torch_version"] = str(torch.__version__)
        except Exception as exc:  # noqa: BLE001 - never fail a run over a record
            logger.warning("ATOMCompass WARNING: could not read the device "
                           "identity (%s); a GPU-free replay of this target "
                           "will have no architecture to run as", exc)
            out["error"] = str(exc)
        return out

    def _write_replay_target(self, blocks=None, graph=None) -> None:
        """Record what a device answered at startup, for a run that has none.

        Merged into the file across two calls rather than written once, because
        the two answers arrive at different times: the block layout comes out of
        ``get_num_blocks`` and the graph ladder only exists after capture. A
        replay needs both, and a run that crashes between them should leave the
        half it got rather than nothing.

        Rank 0 only. The other ranks answer the same startup questions, but the
        engine asks rank 0 and a replay runs one rank; writing all of them would
        produce files that differ only by which rank raced last.
        """
        path = self._compass_config.replay_target_out
        if not path or int(getattr(self, "rank", 0) or 0) != 0:
            return
        import json

        record = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    record = json.load(fh)
            except (OSError, ValueError):
                record = {}
        record["version"] = 1
        config = self.config
        record["config"] = {
            "model": str(getattr(config, "model", "unknown")),
            "tensor_parallel_size": int(
                getattr(config, "tensor_parallel_size", 1) or 1),
            "max_model_len": int(getattr(config, "max_model_len", 0) or 0),
            "max_num_seqs": int(getattr(config, "max_num_seqs", 0) or 0),
            "gpu_memory_utilization": float(
                getattr(config, "gpu_memory_utilization", 0.0) or 0.0),
            "enable_prefix_caching": bool(
                getattr(config, "enable_prefix_caching", False)),
            "enforce_eager": bool(getattr(config, "enforce_eager", False)),
        }
        # The machine, not just the deployment. A GPU-free replay has to be
        # told which architecture it is about: AITER resolves capability flags
        # (`is_fp8_avail` and friends) from the arch name at import time, and
        # with no device to ask, the captured one is the correct answer rather
        # than a convenient one. See `atom.compass.replay.bootstrap`.
        record["hardware"] = self._hardware_identity()
        if blocks is not None:
            # Verbatim. The engine rebuilds `StateRuntime` from this wire form
            # and the block manager plans from `pool_entries`; reinterpreting
            # either here would put a second sizing implementation in the path
            # the replay is supposed to be running unmodified.
            record["blocks"] = dict(blocks)
        if graph is not None:
            record["graph"] = dict(graph)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=1)
        except OSError as exc:
            logger.warning("ATOMCompass WARNING: could not write the replay "
                           "target to %s: %s", path, exc)
            return
        logger.info("ATOMCompass: replay target -> %s", path)

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
