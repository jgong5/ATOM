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
from dataclasses import dataclass
from typing import Optional

from atom.compass.core.cost.base import StepCost, StepShape

logger = logging.getLogger(__name__)

__all__ = ["PricedGraphCostOracle"]


@dataclass(frozen=True)
class _Costed:
    """One graph, priced, and the shape it describes."""

    path: str
    seconds: float
    ops: int
    launches: int
    breakdown: dict
    is_prefill: bool
    batch: int
    context: float



#: Seconds added per *operator* on a step that was not replayed. A prefill step
#: runs eager -- its token count is not on the capture ladder -- so the host
#: dispatches every operator individually and that dispatch, not the kernels, is
#: most of the step. Prefill's kernels come to 15.8ms of a 37.3ms step, and the
#: remainder over 319 operators is 67us each. The same fit as the boundary
#: constant and the same caveat: one step, one deployment.
DEFAULT_EAGER_SECONDS_PER_OP = 86.35e-6

#: Seconds added per kernel launch. Fitted as (step - priced) / launches on a
#: Qwen3-0.6B decode step at batch 4, over three runs: (3.201ms - 2.341ms) / 382.
#: Independently, the median gap between a priced kernel and the same kernel in a
#: profile of a real step is 2.05us. Deployment-specific, and an option for that
#: reason.
DEFAULT_BOUNDARY_SECONDS = 2.25e-6


class PricedGraphCostOracle:
    """Costs a step by summing the priced operators of its op graph."""

    def __init__(self, prices: str, graph: str, prefill_graph: str = "",
                 boundary_seconds: float = DEFAULT_BOUNDARY_SECONDS,
                 eager_seconds_per_op: float = DEFAULT_EAGER_SECONDS_PER_OP,
                 floor_seconds: float = 1e-6, fallback: str = "",
                 rank_coords: Optional[dict] = None) -> None:
        """
        Args:
            prices: A price list from ``--compass-bench-out``.
            graph: A decode op graph from ``--compass-graph-out``. Its
                operators are looked up in the price list by the same signature
                the benchmark priced them under.
            prefill_graph: A prefill op graph, from the same run with
                ``--compass-trace-prefill``. Without one, prefill steps fall
                through to ``fallback`` and none of the op-graph work reaches
                TTFT.
            boundary_seconds: Added per kernel launch on a replayed step. See
                the module docstring; zero reproduces the naive sum, which is
                26% low.
            eager_seconds_per_op: Added per operator on a step that was not
                replayed, where the host dispatches each one. Two constants
                rather than one because the two kinds of step differ by more
                than a factor: the same graph costs its kernels plus 2 µs a
                launch when replayed and its kernels plus 67 µs an operator when
                not, and using the replayed constant for prefill priced it at
                0.42 of the step.
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
        self.eager_seconds_per_op = float(eager_seconds_per_op)
        self.floor_seconds = float(floor_seconds)

        with open(self.prices_path, encoding="utf-8") as fh:
            price_list = json.load(fh)["prices"]

        self.unpriced = 0
        with open(self.graph_path, encoding="utf-8") as fh:
            self.decode = self._cost(json.load(fh), price_list, self.graph_path)

        # A prefill step is different operators at different shapes, so a decode
        # graph cannot answer for one. Interpolating *within* a kind was tried
        # and dropped: four decode graphs across a run's context range moved the
        # held-out error from 3.7% to 1.5-2.3%, and covering the range rather
        # than extrapolating over it made no further difference -- so context is
        # not what dominates, and the machinery was not worth its complexity.
        # Two graphs of two kinds is a different proposition: it is the
        # difference between predicting TTFT and not predicting it at all.
        self.prefill = None
        if prefill_graph:
            self.prefill_path, _ = resolve_rank_path(prefill_graph, rank_coords)
            with open(self.prefill_path, encoding="utf-8") as fh:
                self.prefill = self._cost(json.load(fh), price_list,
                                          self.prefill_path)

        self.fallback = None
        if fallback:
            from atom.compass.core.cost.calibrated import CalibratedCostOracle

            self.fallback = CalibratedCostOracle(fallback,
                                                 floor_seconds=floor_seconds,
                                                 rank_coords=rank_coords)
        self._warned = False

        if self.unpriced:
            logger.warning(
                "ATOMCompass WARNING: %d operators have no price and contribute "
                "nothing; steps will be predicted low.", self.unpriced)

    def _cost(self, graph_blob: dict, price_list: dict, path: str) -> "_Costed":
        """What one graph costs, and the shape it is a graph of."""
        from atom.compass.runtime.microbench import signature_of

        seconds = 0.0
        launches = 0
        priced_ops = 0
        breakdown: dict[str, float] = {}
        for op in graph_blob["ops"]:
            entry = price_list.get(signature_of(op))
            if entry is None:
                self.unpriced += 1
                continue
            priced_ops += 1
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
            seconds=seconds,
            ops=priced_ops,
            launches=launches,
            breakdown=breakdown,
            is_prefill=bool(recorded.get("num_prefill_tokens", 0)),
            batch=len(recorded.get("num_scheduled_tokens") or []),
            context=(sum(contexts) / len(contexts)) if contexts else 0.0,
        )

    def estimate(self, shape: StepShape) -> StepCost:
        point = self.prefill if shape.is_prefill else self.decode
        if point is None:
            if self.fallback is not None:
                return self.fallback.estimate(shape)
            if not self._warned:
                self._warned = True
                logger.warning(
                    "ATOMCompass WARNING: no prefill graph, so prefill steps "
                    "are answered with the decode graph's cost. Trace one with "
                    "--compass-trace-prefill and pass it as "
                    "--compass-oracle-option prefill_graph=<path>.")
            point = self.decode
        # What a step pays on top of its kernels depends on how it ran, and the
        # difference is thirtyfold. A replayed step is one submission and the
        # host is not in the loop; an eager one dispatches every operator. The
        # engine says which through `capture_bucket`, which is None exactly when
        # nothing was replayed -- so this stays a property of the step rather
        # than of the model or of what "prefill" happens to mean.
        if shape.capture_bucket is None:
            overhead = point.ops * self.eager_seconds_per_op
        else:
            overhead = point.launches * self.boundary_seconds
        total = max(point.seconds + overhead, self.floor_seconds)
        return StepCost(seconds=total,
                        breakdown=dict(point.breakdown, **{"<overhead>": overhead}))

    def describe(self) -> str:
        prefill = (f"{self.prefill.seconds*1e3:.3f}ms prefill kernels"
                   if self.prefill else "no prefill graph")
        return (f"PricedGraphCostOracle("
                f"{self.decode.seconds*1e3:.3f}ms decode kernels, {prefill}; "
                f"+{self.boundary_seconds*1e6:.2f}us/launch replayed, "
                f"+{self.eager_seconds_per_op*1e6:.1f}us/op eager"
                + (", calibrated fallback" if self.fallback else "") + ")")
