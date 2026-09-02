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

import json
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

        shape = self._describe(batch)
        cost = self._oracle.estimate(shape)
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
            self._count_and_record(shape, time.perf_counter() - began, None)
            return output

        import time

        entered = time.perf_counter()
        gap = (entered - self._last_forward_ended
               if self._last_forward_ended is not None else None)

        began = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        began.record()
        output = super().forward(batch)
        ended.record()
        self._last_forward_ended = time.perf_counter()
        self._pending.append((shape, began, ended, gap))
        self._drain_pending()
        return output

    def _drain_pending(self) -> None:
        """Write out every timed step whose events have completed.

        Draining by ``query()`` rather than ``synchronize()`` keeps the host off
        the critical path: a step is written once the device has finished it
        anyway, never by waiting for it.
        """
        while self._pending:
            shape, began, ended, gap = self._pending[0]
            if not ended.query():
                return
            self._pending.popleft()
            self._count_and_record(shape, began.elapsed_time(ended) / 1000.0, gap)

    def _count_and_record(self, shape: StepShape, seconds: float,
                          gap: Optional[float] = None) -> None:
        kind = "prefill" if shape.is_prefill else "decode"
        seen = self._measured_by_kind.get(kind, 0) + 1
        self._measured_by_kind[kind] = seen
        self._measured_steps += 1
        # Counted per kind, not overall. Prefill happens a handful of times in a
        # whole run, so a warmup counted in total steps discards every prefill
        # sample there is.
        if seen > self._compass_config.measure_warmup_steps:
            self._record_measurement(shape, seconds, gap)

    def _record_measurement(self, shape: StepShape, seconds: float,
                            gap: Optional[float] = None) -> None:
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
        if self._step_index != self._compass_config.trace_step:
            return super().forward(batch)

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
        self._write_graph(batch)
        if timing is not None:
            self._write_op_timings(timing)
        return output

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

    def _write_graph(self, batch: ScheduledBatch) -> None:
        path = self._compass_config.graph_out
        if not path:
            return
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
            "trace_step": self._compass_config.trace_step,
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
            capture_bucket=self._capture_bucket(len(num_scheduled)),
        )

    def _capture_bucket(self, batch_size: int) -> Optional[int]:
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
        resolved, or a batch larger than the top rung -- `ForwardMode.decide`
        falls back to eager there. A bucket that did not happen must not be
        fitted as though it did.
        """
        if getattr(self, "enforce_eager", False):
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
        return {"tp": int(getattr(self, "rank", 0) or 0)}

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
