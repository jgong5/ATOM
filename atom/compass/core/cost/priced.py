"""A step costs what its operators cost, plus what it costs to run them.

The first oracle that predicts from the **op graph** rather than from a step's
shape. Everything else here -- tracing, derivation, the microbenchmark, the
recorded forward context -- was groundwork for this and until now paid for
nothing: both other oracles fit token counts and batch sizes and never look at
an operator.

Two terms, and the second is not a fudge:

    step = sum over operators of (price x occurrences)  +  launches x boundary

The first term is what the kernels compute. It comes to about three quarters of
a step and no amount of coverage closes the rest -- 98.8% of operators priced
still summed to 0.740 of the step. The missing quarter is *not* a multiplicative
error, and that matters, because a factor is what one reaches for first. Against
the same kernels measured inside a real step, the ratio runs from 0.43 to 0.99,
which no single number corrects. As a fixed cost per launch it is 0.80 to 3.56
microseconds, median 2.05 -- and the constant needed to close the whole step,
fitted with no instrument but the engine's own clock, is 2.02.

Those two agree and they were arrived at separately. A ratio looks wrong because
the cost is not proportional to the kernel: it is invisible on a 13 microsecond
gemm and doubles a 2.7 microsecond rmsnorm, which is exactly the spread. What it
physically is remains open -- a dependent kernel boundary needs a barrier and a
cache flush, and the captured copies a benchmark times have neither -- and open
problem 21 has the experiments. The model does not depend on knowing.

What this oracle cannot do yet is price a shape it has no graph for. It holds
the graphs it was given, keyed by whether the step is prefill and which capture
rung it replayed, and says so when asked about anything else rather than
extrapolating from one shape to another. Deriving a graph per shape is what
``runtime/derive.py`` is for and is the next piece.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from atom.compass.core.cost.base import StepCost, StepShape

logger = logging.getLogger(__name__)

__all__ = ["PricedGraphCostOracle"]

#: Seconds added per kernel launch. Fitted as (step - priced) / launches on a
#: Qwen3-0.6B decode step at batch 4: (3.115ms - 2.305ms) / 401. Independently,
#: the median gap between a priced kernel and the same kernel in a profile of a
#: real step is 2.05us. Deployment-specific, and an option for that reason.
DEFAULT_BOUNDARY_SECONDS = 2.02e-6


class PricedGraphCostOracle:
    """Costs a step by summing the priced operators of its op graph."""

    def __init__(self, prices: str, graph: str,
                 boundary_seconds: float = DEFAULT_BOUNDARY_SECONDS,
                 floor_seconds: float = 1e-6, fallback: str = "",
                 rank_coords: Optional[dict] = None) -> None:
        """
        Args:
            prices: A price list from ``--compass-bench-out``.
            graph: An op graph from ``--compass-graph-out``. Its operators are
                looked up in the price list by the same signature the benchmark
                priced them under.
            boundary_seconds: Added per kernel launch. See the module docstring;
                zero reproduces the naive sum, which is 26% low.
            floor_seconds: Smallest duration ever returned, so a virtual clock
                cannot be run backwards by an empty or unpriced graph.
            fallback: A calibration table, used for steps the graph does not
                describe. One step is traced, so a decode graph can say nothing
                about a prefill step; without this the oracle would answer with
                a decode cost, which is not an approximation but a different
                question. Optional, and its absence is a warning rather than an
                error so that a decode-only run needs nothing extra.
            rank_coords: This rank's coordinates. Each rank prices its own
                graph under parallelism.
        """
        from atom.compass.core.artifacts import resolve_rank_path
        from atom.compass.runtime.microbench import signature_of

        self.prices_path, _ = resolve_rank_path(prices, rank_coords)
        self.graph_path, _ = resolve_rank_path(graph, rank_coords)
        self.boundary_seconds = float(boundary_seconds)
        self.floor_seconds = float(floor_seconds)

        with open(self.prices_path, encoding="utf-8") as fh:
            price_list = json.load(fh)["prices"]
        with open(self.graph_path, encoding="utf-8") as fh:
            graph_blob = json.load(fh)

        self.seconds = 0.0
        self.launches = 0
        self.unpriced = 0
        breakdown: dict[str, float] = {}
        for op in graph_blob["ops"]:
            entry = price_list.get(signature_of(op))
            if entry is None:
                self.unpriced += 1
                continue
            self.seconds += entry["seconds"]
            # An operator is not one kernel. Attention launches three, and the
            # boundary cost is paid at each -- so the count comes from what the
            # benchmark saw the operator launch, not from the operator count.
            self.launches += max(1, len(entry.get("kernels") or {}))
            breakdown[op["name"]] = breakdown.get(op["name"], 0.0) + entry["seconds"]

        self.overhead = self.launches * self.boundary_seconds
        self.breakdown = dict(breakdown)
        # The step this graph describes: one shape, and the oracle knows only it.
        self.is_prefill = any(
            (dict(kv for kv in (tuple(x) for x in op.get("context") or ()))
             .get("is_prefill")) for op in graph_blob["ops"])

        self.fallback = None
        if fallback:
            from atom.compass.core.cost.calibrated import CalibratedCostOracle

            self.fallback = CalibratedCostOracle(fallback,
                                                 floor_seconds=floor_seconds,
                                                 rank_coords=rank_coords)
        self._warned = False

        if self.unpriced:
            logger.warning(
                "ATOMCompass WARNING: %d of %d operators in %s have no price "
                "and contribute nothing; the step will be predicted low.",
                self.unpriced, len(graph_blob["ops"]), self.graph_path)

    def estimate(self, shape: StepShape) -> StepCost:
        if shape.is_prefill != self.is_prefill:
            if self.fallback is not None:
                return self.fallback.estimate(shape)
            if not self._warned:
                self._warned = True
                logger.warning(
                    "ATOMCompass WARNING: the graph describes a %s step and "
                    "this one is %s; answering with the graph's cost anyway "
                    "because no fallback table was given. Pass "
                    "--compass-oracle-option fallback=<measure table>.",
                    "prefill" if self.is_prefill else "decode",
                    "prefill" if shape.is_prefill else "decode")
        total = max(self.seconds + self.overhead, self.floor_seconds)
        return StepCost(seconds=total,
                        breakdown=dict(self.breakdown,
                                       **{"<kernel boundaries>": self.overhead}))

    def describe(self) -> str:
        return (f"PricedGraphCostOracle({len(self.breakdown)} operator kinds, "
                f"{self.launches} launches, "
                f"{self.seconds*1e3:.3f}ms priced + "
                f"{self.overhead*1e3:.3f}ms boundaries"
                + (", calibrated fallback" if self.fallback else "") + ")")
