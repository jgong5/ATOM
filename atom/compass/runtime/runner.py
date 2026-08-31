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

import logging
from typing import Optional

from atom.compass.config import CompassConfig
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

    @staticmethod
    def _build_oracle(config: CompassConfig) -> CostOracle:
        oracle_cls = resolve_obj_by_qualname(config.oracle_qualname)
        return oracle_cls(**(config.oracle_options or {}))

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
        """Run the real forward and record how long it took.

        This is the training data for a calibrated oracle, so what is recorded
        has to be the same description the oracle will later be asked to predict
        from — hence ``StepShape``, unreduced, rather than a summary. Reducing
        here and expecting the oracle to cope later is how a cost model ends up
        fitted to a feature it will never see again.

        Timing brackets a device synchronise on both sides. ATOM's forward
        returns once the work is enqueued, so timing it directly measures how
        long it took to submit, which is not what anybody wants to predict.

        The first steps are timed and discarded: Triton autotunes on a kernel's
        first launch and the allocator is still growing, so an oracle fitted to
        them describes a machine nobody runs.
        """
        import time

        import torch

        sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
        shape = self._describe(batch)

        sync()
        start = time.perf_counter()
        output = super().forward(batch)
        sync()
        seconds = time.perf_counter() - start

        # Counted per kind, not overall. Prefill happens at the very start of a
        # run, so a warmup counted in total steps discards every prefill sample
        # there is -- the oracle then has nothing to fit, predicts zero, and
        # TTFT comes out as 0 ms against a real 7.6 s. Both kinds autotune on
        # their own first launch, so per-kind counting still excludes what the
        # warmup exists to exclude.
        kind = "prefill" if shape.is_prefill else "decode"
        self._measured_steps += 1
        seen = self._measured_by_kind.get(kind, 0) + 1
        self._measured_by_kind[kind] = seen
        if seen > self._compass_config.measure_warmup_steps:
            self._record_measurement(shape, seconds)
        return output

    def _record_measurement(self, shape: StepShape, seconds: float) -> None:
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
                logger.warning("ATOMCompass: could not open %s for timings: %s",
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

        ops = MetaOpTracer(graph=self._graph, topology=self._topology())
        triton = TritonLaunchTracer(graph=self._graph)
        # Under simulated TP the collective is replaced by a passthrough, so it
        # never dispatches and never gets recorded — a TP graph captured on one
        # device would show no communication at all. A no-op on a real
        # multi-device run, where the collective dispatches and is recorded once.
        collectives = record_collectives(self._graph)
        try:
            with collectives, triton, ops:
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
        return output

    @staticmethod
    def _rank_path(path: str, coords: dict[str, int]) -> str:
        """Give each rank its own file.

        Every rank traces, and under any parallelism their graphs differ — that
        difference is the thing worth recording. Writing them all to one path
        makes them race for it and leaves a single file that names no rank, so
        the one artifact that survives cannot be attributed and the rest are
        lost without a trace.
        """
        import os

        suffix = "-".join(f"{name}{index}" for name, index in sorted(coords.items()))
        stem, ext = os.path.splitext(path)
        return f"{stem}.{suffix}{ext}" if suffix else path

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
            logger.warning("ATOMCompass: could not write graph to %s: %s", path, exc)
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
                "ATOMCompass: graph holds %d attention operators for a %d-layer "
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
        )

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
            return super().capture_cudagraph()
        if self._compass_config.mode == "trace":
            logger.info(
                "ATOMCompass: skipping CUDA graph capture so the forward stays "
                "traceable — a replay is one opaque submission and would record "
                "nothing. The graph is the eager operator sequence; take its "
                "cost from a measure run."
            )
        else:
            logger.debug("ATOMCompass: skipping CUDA graph capture")
        return 0.0, [], 0

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
