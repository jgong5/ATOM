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
        logger.info(
            "ATOMCompass active: oracle=%s filler_token_id=%d",
            self._oracle.describe(),
            self._compass_config.filler_token_id,
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
        """Predict the step, synthesise its output, and report the duration."""
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
