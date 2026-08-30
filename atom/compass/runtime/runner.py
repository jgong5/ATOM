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
        self._compass_config = self._resolve_compass_config()
        self._oracle: CostOracle = self._build_oracle(self._compass_config)
        self._graph = OpGraph()
        self._traced_steps = 0
        self._step_index = 0
        logger.info(
            "ATOMCompass active: mode=%s oracle=%s",
            self._compass_config.mode,
            self._oracle.describe(),
        )

    # -- setup ----------------------------------------------------------------

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
        if self._compass_config.mode == "trace":
            return self._forward_traced(batch)

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
        from atom.compass.runtime.meta import MetaOpTracer
        from atom.compass.runtime.triton_trace import TritonLaunchTracer

        self._step_index += 1
        if self._step_index != self._compass_config.trace_step:
            return super().forward(batch)

        ops = MetaOpTracer(graph=self._graph)
        triton = TritonLaunchTracer(graph=self._graph)
        try:
            with triton, ops:
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

    def _write_graph(self, batch: ScheduledBatch) -> None:
        path = self._compass_config.graph_out
        if not path:
            return
        shape = self._describe(batch)
        self._graph.key = GraphKey.of(
            model_id=str(getattr(self.config, "model", "unknown")),
            topology=self._topology(),
            rank_coords=self._rank_coords(),
            batch_signature=shape.num_scheduled_tokens,
        )
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

    # -- work that has no meaning without real compute -------------------------

    def capture_cudagraph(self) -> None:
        """No graphs to capture when no kernels run."""
        logger.debug("ATOMCompass: skipping CUDA graph capture")

    def warmup_model(self) -> None:
        """Nothing to warm up."""
        logger.debug("ATOMCompass: skipping model warmup")
