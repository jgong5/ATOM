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

import glob
import json
import logging
from dataclasses import dataclass, replace
from typing import Optional

from atom.compass.core.cost.base import StepCost, StepShape

logger = logging.getLogger(__name__)

__all__ = ["PricedGraphCostOracle"]


@dataclass(frozen=True)
class _Costed:
    """One graph, priced, and the shape it describes."""

    path: str
    seconds: float
    launches: int
    breakdown: dict
    is_prefill: bool
    batch: int
    context: float

    def at(self, seconds: float) -> "_Costed":
        return replace(self, seconds=seconds)

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

        # One graph describes one shape. Given several -- `graph` is a glob --
        # the oracle costs each and interpolates between them, which is what
        # lets a prediction move with context length instead of answering every
        # decode step with the number the traced step happened to cost.
        paths = sorted(glob.glob(self.graph_path)) or [self.graph_path]
        self.points: list[_Costed] = []
        self.unpriced = 0
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                self.points.append(self._cost(json.load(fh), price_list, path))
        self.points.sort(key=lambda p: p.context)

        self.fallback = None
        if fallback:
            from atom.compass.core.cost.calibrated import CalibratedCostOracle

            self.fallback = CalibratedCostOracle(fallback,
                                                 floor_seconds=floor_seconds,
                                                 rank_coords=rank_coords)
        self._warned = False

        if self.unpriced:
            logger.warning(
                "ATOMCompass WARNING: %d operators across %d graph(s) have no "
                "price and contribute nothing; steps will be predicted low.",
                self.unpriced, len(self.points))

    def _cost(self, graph_blob: dict, price_list: dict, path: str) -> "_Costed":
        """What one graph costs, and the shape it is a graph of."""
        from atom.compass.runtime.microbench import signature_of

        seconds = 0.0
        launches = 0
        breakdown: dict[str, float] = {}
        for op in graph_blob["ops"]:
            entry = price_list.get(signature_of(op))
            if entry is None:
                self.unpriced += 1
                continue
            seconds += entry["seconds"]
            # An operator is not one kernel. Attention launches three, and the
            # boundary cost is paid at each -- so the count comes from what the
            # benchmark saw the operator launch, not from the operator count.
            launches += max(1, len(entry.get("kernels") or {}))
            breakdown[op["name"]] = breakdown.get(op["name"], 0.0) + entry["seconds"]

        recorded = (graph_blob.get("provenance") or {}).get("shape") or {}
        contexts = recorded.get("context_lens") or []
        return _Costed(
            path=path,
            seconds=seconds + launches * self.boundary_seconds,
            launches=launches,
            breakdown=breakdown,
            is_prefill=bool(recorded.get("num_prefill_tokens", 0)),
            batch=len(recorded.get("num_scheduled_tokens") or []),
            context=(sum(contexts) / len(contexts)) if contexts else 0.0,
        )

    def _interpolate(self, context: float) -> "_Costed":
        """The cost at ``context``, from the decode graphs either side of it.

        Linear, and extrapolating past the ends rather than clamping: decode
        cost rises with context because attention walks more KV, and a run that
        generates past the last traced step is the ordinary case, not an edge
        one. Clamping there would reintroduce exactly the flat prediction this
        exists to remove.
        """
        usable = [p for p in self.points if not p.is_prefill] or self.points
        if len(usable) == 1:
            return usable[0]
        lower, upper = usable[0], usable[-1]
        for left, right in zip(usable, usable[1:]):
            if left.context <= context <= right.context:
                lower, upper = left, right
                break
        span = upper.context - lower.context
        if span <= 0:
            return lower
        at = (context - lower.context) / span
        return lower.at(lower.seconds + at * (upper.seconds - lower.seconds))

    def estimate(self, shape: StepShape) -> StepCost:
        if shape.is_prefill != any(p.is_prefill for p in self.points):
            if self.fallback is not None:
                return self.fallback.estimate(shape)
            if not self._warned:
                self._warned = True
                logger.warning(
                    "ATOMCompass WARNING: no graph describes a %s step; "
                    "answering with a %s one. Pass "
                    "--compass-oracle-option fallback=<measure table>.",
                    "prefill" if shape.is_prefill else "decode",
                    "prefill" if not shape.is_prefill else "decode")
        context = (sum(shape.context_lens) / len(shape.context_lens)
                   if shape.context_lens else 0.0)
        point = self._interpolate(context)
        total = max(point.seconds, self.floor_seconds)
        return StepCost(seconds=total, breakdown=dict(point.breakdown))

    def describe(self) -> str:
        span = (f"{self.points[0].context:.0f}-{self.points[-1].context:.0f} "
                f"tokens of context" if self.points else "no graphs")
        costs = ", ".join(f"{p.seconds*1e3:.3f}ms" for p in self.points[:4])
        return (f"PricedGraphCostOracle({len(self.points)} graph(s) over {span}: "
                f"{costs}"
                + (", calibrated fallback" if self.fallback else "") + ")")
