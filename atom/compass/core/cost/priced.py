"""**empirical/measured** -- a model step costs what its operators cost, plus
what it costs to run them.

It reaches a model step by summing its operators, where `calibrated` fits the
step directly -- a difference of method, not of subject. The first oracle that
predicts from the **op graph** rather than from a step's shape. Everything else here -- tracing, derivation, the microbenchmark, the
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
import os
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
    kernel_seconds: tuple
    is_prefill: bool
    batch: int
    context: float



#: Seconds added per *operator* on a step that was not replayed. A prefill step
#: runs eager -- its token count is not on the capture ladder -- so the host
#: dispatches every operator individually and that dispatch, not the kernels, is
#: most of the step. Prefill's kernels come to 15.8ms of a 37.3ms step, and the
#: remainder over 319 operators is 67us each. The same fit as the boundary
#: constant and the same caveat: one step, one deployment.
DEFAULT_EAGER_SECONDS_PER_OP = None

#: Seconds the host takes to dispatch one operator, of which a step pays only
#: what its kernels do not hide. Fitted as the D solving
#: ``sum(max(0, D - kernel)) == step - priced`` : 132.70 us on Qwen3-0.6B and
#: 101.22 us on Qwen3.8-27B, against per-operator overheads of 86.35 and 34.62 --
#: so restating the quantity as a dispatch the kernels race narrows the spread
#: between two very different models from 2.5x to 1.3x. At 130 us both prefill
#: steps come out within about 3%.
#:
#: The 27B barely constrains it: 403 of its 708 operators hide the dispatch
#: entirely, so its prediction moves only 3% across the whole range 100-132 us.
#: The 0.6B, where only 29 of 316 hide it, is what pins the value. Two models is
#: a thin basis and this should be refitted as more are measured.
DEFAULT_DISPATCH_SECONDS = 130e-6

#: Seconds added per kernel launch. Fitted as (step - priced) / launches on a
#: Qwen3-0.6B decode step at batch 4, over three runs: (3.201ms - 2.341ms) / 382.
#: Deployment-specific, and an option for that reason.
#:
#: **It is not a boundary, whatever the residual it was fitted from.** A
#: replayed step's device timeline has no room for one. Measured on four
#: configurations -- 0.6B at TP=1 and the 27B at TP=2, 4 and 8 -- the gap
#: between consecutive kernels has a median of 1ns and a p90 of 2ns, and the
#: idle in a whole step is 12us, essentially all of it a single gap where the
#: graph is segmented. If every launch paid 2.25us, each step would show ~950
#: gaps in the 1-5us band; across 15200 gaps per configuration that band holds
#: **zero**, while the same traces record 16 gaps above 5us and an eager
#: prefill profiled the same way came out 63.6% idle. The instrument is not
#: blind at this scale; there is nothing there.
#:
#: Measured as idle, per launch, with a priced sum and a step from the same
#: machine (`scripts/compass/residual.py`): 10.5ns, 12.6ns and 12.8ns on three
#: configurations across two boxes, against the 2250ns modelled here. The
#: residual it is fitted from is pricing error almost in full.
#:
#: That error varies more between machines than between tensor-parallel widths:
#: +10.0% of the priced sum for the 27B at TP=4 on one box and +14.3% at TP=8
#: there, against -2.0% to +1.1% for the same model and width on another -- which
#: is that machine's whole residual sitting inside the +-1% a repeat pricing run
#: moves the total by, so it is not distinguishable from zero.
#:
#: Per kernel, with every operator's kernels named (see `_wants_breakdown`) and
#: each entry's own price distributed across them, the errors reproduce across
#: two pricing runs to about a point:
#:
#:     __amd_rocclr_copyBuffer          +260%
#:     fused_qk_rmsnorm_group_quant     -37%
#:     fused_recurrent_gated_delta_rule -26%
#:     paged_attention_decode           -25%
#:     cross_device_reduce_1stage       -21%
#:     the gemms                        -10% to +12%
#:
#: with one exception: `silu_and_mul` moved +14% to -23% between the same two
#: runs, and +47% then -32% in its per-call price, so it is not measured, it is
#: unstable. That aside, the error is a fixable one rather than noise, and
#: fixing it is what would let this constant go to zero. Until then the value
#: stands: removing it would make predictions worse without making them righter.
DEFAULT_BOUNDARY_SECONDS = 2.25e-6

#: Seconds added per kernel launch on a compiled step that was not replayed.
#: A compiled step submits its kernels from generated code, so it pays neither
#: eager dispatch nor a replay's boundary, and charging it the eager term
#: overstated a chunked prefill's overhead four times over -- 21.5ms against a
#: real 5.7ms. Measured by profiling the step and subtracting the time kernels
#: were running from the step's own device window: 5.722ms over the 589 launches
#: this model counts for that graph. Taken warm; the same step cold spends 31ms
#: more in compilation stalls, which is not overhead and must not be fitted as
#: any. One step of one model, so it is an argument.
#: Operators that make the host wait for the device and run no kernel worth the
#: name. They cannot be graph-captured -- a synchronise inside a capture is an
#: error -- so the benchmark times them back to back, and what it measures is
#: the synchronisation itself.
#:
#: They must not be summed into a step's *kernel* time. Priced at 17.15 and
#: 16.02 microseconds and appearing 48 and 24 times in one 27B decode graph,
#: they added 1.208 ms to a 10.5 ms step and put the priced total 8.4% *above*
#: the step it was inside -- which looked, for a while, like the isolated
#: kernels had somehow become more expensive than the in-situ ones.
#:
#: Their cost is real, and it belongs to the overhead term: a synchronise
#: stalls the pipeline, which is exactly the host-side waiting `max(kernels,
#: launches x host)` already describes. Counting it twice was the error.
HOST_SYNC = frozenset({
    "aten::item", "aten::is_nonzero", "aten::_local_scalar_dense",
    "aten::equal", "aten::allclose",
})



def _declared_topology(price_blob: dict, prices_path: str):
    """The parallel width a price list was measured at, or None if it cannot say.

    Lists written before the width was recorded still name the graphs they were
    priced from, and a graph has always carried its topology. Reading it back
    from there is a statement the artifact already makes, not an assumption
    about it -- so an existing list keeps paying for its own collectives while
    still refusing to pay for another width's. When the graphs are gone, or
    disagree, the list cannot certify anything and says so.
    """
    provenance = price_blob.get("provenance") or {}
    if provenance.get("topology") is not None:
        return provenance["topology"]
    widths = set()
    for name in provenance.get("graphs") or ():
        for candidate in (name, os.path.join(
                os.path.dirname(os.path.dirname(prices_path) or "."), name)):
            if os.path.exists(candidate):
                try:
                    with open(candidate, encoding="utf-8") as fh:
                        key = json.load(fh).get("key") or {}
                except (OSError, ValueError):
                    return None
                widths.add(tuple(sorted(
                    tuple(x) for x in (key.get("topology") or []))))
                break
        else:
            return None
    return dict(next(iter(widths))) if len(widths) == 1 else None

_WARNED_TOPOLOGY: set = set()


def _collectives_transferable(graph_topology, price_topology) -> bool:
    """Whether a collective in this graph may be paid for from this price list.

    Only when both sides declare the same group widths. A price list that does
    not declare its own width -- everything written before this check existed --
    cannot certify anything, so its collectives are refused rather than assumed
    to fit, and the refusal is warned about once per pair.

    A graph with no group wider than one rank contains no collective to price,
    so nothing is refused and nothing is said.
    """
    graph_widths = {g: w for g, w in (graph_topology or {}).items() if w > 1}
    if not graph_widths:
        return True
    price_widths = ({g: w for g, w in (price_topology or {}).items() if w > 1}
                    if price_topology is not None else None)
    if price_widths == graph_widths:
        return True
    token = (tuple(sorted(graph_widths.items())),
             None if price_widths is None else tuple(sorted(
                 price_widths.items())))
    if token not in _WARNED_TOPOLOGY:
        _WARNED_TOPOLOGY.add(token)
        logger.warning(
            "ATOMCompass WARNING: the price list was measured at %s and this "
            "graph is %s, so its collectives are left unpriced. A collective's "
            "signature carries its message and not its group width, so a price "
            "from another width would have matched exactly and been spent "
            "silently. Price the collectives at this width to close the gap.",
            "no declared width" if price_widths is None
            else price_widths or "no group wider than one rank",
            graph_widths)
    return False


DEFAULT_COMPILED_SECONDS_PER_LAUNCH = 9.71e-6

#: How long the host takes per kernel launch on a compiled, not-replayed step.
#: A compiled step lasts `max(kernel time, launches x this)` -- the host runs
#: ahead and the step waits on whichever is slower -- so this is a *floor*, not
#: an addend, and it is the only form that fits a step whose idle runs from 64%
#: to 0% as its kernels grow. 95.8 us on Qwen3-0.6B at TP=1, fitted to the two
#: host-bound shapes of four and holding all four to within 2%.
#:
#: **This default is a starting point, not a constant.** It survives a change
#: of machine (95.7 us on a second node whose GPUs are 20% faster) and a change
#: of width (89.3 us at TP=2 and at TP=4 alike, the 7% being the collectives'
#: cheaper launches). It does *not* survive a change of model: the 27B measures
#: 125.0 us, 40% higher. Calibrate per model with `step_accounting --calibrate`
#: on any host-bound shape.
#:
#: Set to 0 to fall back to the older `kernels + launches x
#: compiled_seconds_per_launch`, which no shape has ever supported over a range
#: and which is kept only so an old calibration still loads.
DEFAULT_HOST_SECONDS_PER_LAUNCH = 95.8e-6


class PricedGraphCostOracle:
    """Costs a step by summing the priced operators of its op graph."""

    def __init__(self, prices: str, graph: str, prefill_graph: str = "",
                 boundary_seconds: float = DEFAULT_BOUNDARY_SECONDS,
                 eager_seconds_per_op: Optional[float] = DEFAULT_EAGER_SECONDS_PER_OP,
                 dispatch_seconds: float = DEFAULT_DISPATCH_SECONDS,
                 compiled_seconds_per_launch: float =
                 DEFAULT_COMPILED_SECONDS_PER_LAUNCH,
                 host_seconds_per_launch: float =
                 DEFAULT_HOST_SECONDS_PER_LAUNCH,
                 calibration: str = "",
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
            eager_seconds_per_op: A flat cost per operator on a step that was
                not replayed. ``None`` -- the default -- computes it from
                ``dispatch_seconds`` instead, which transfers between models
                where a flat figure does not.
            dispatch_seconds: What the host takes to dispatch one operator. An
                eager step pays ``max(0, dispatch - kernel)`` of it per operator,
                because the host runs ahead while the device works and only the
                part the device cannot hide is spent. This is why a flat
                per-operator figure is 86 µs on a model with small kernels and
                35 µs on one with large ones, while the dispatch behind both is
                nearly the same number.
            host_seconds_per_launch: How long the host takes per launch. A
                compiled step lasts `max(kernel time, launches x this)`, so
                this is a floor rather than an addend -- see
                `DEFAULT_HOST_SECONDS_PER_LAUNCH`. Zero restores the older
                additive term.
            compiled_seconds_per_launch: Added per kernel launch on a compiled
                step that was not replayed. Used only when the runner says the
                step was compiled; a shape that does not say falls through to
                the eager terms, which is what every graph traced before this
                existed does.
            calibration: A file from ``step_accounting.py --calibrate``, whose
                measured overhead replaces ``compiled_seconds_per_launch``. The
                constant does not transfer between models -- 9.71us per launch
                on a 0.6B, 1.68us on a 27B -- so the honest default is not a
                better constant but a measurement of the deployment being
                modelled, which is one profiled step away.
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
        from atom.compass.core.cost.identity import cost_key

        self.prices_path, _ = resolve_rank_path(prices, rank_coords)
        self.graph_path, _ = resolve_rank_path(graph, rank_coords)
        self.boundary_seconds = float(boundary_seconds)
        self.eager_seconds_per_op = (None if eager_seconds_per_op is None
                                     else float(eager_seconds_per_op))
        self.dispatch_seconds = float(dispatch_seconds)
        self.compiled_seconds_per_launch = float(compiled_seconds_per_launch)
        self.host_seconds_per_launch = float(host_seconds_per_launch or 0.0)
        if calibration:
            with open(calibration, encoding="utf-8") as fh:
                blob = json.load(fh)
            # A calibration taken on a device-bound step carries only an upper
            # bound on the host constant, and `step_accounting` says so by
            # naming the field differently rather than by writing a number that
            # looks measured. Taking the bound would put a floor under every
            # step the size of the one it was taken on.
            if blob.get("host_seconds_per_launch"):
                self.host_seconds_per_launch = float(
                    blob["host_seconds_per_launch"])
            measured = blob["compiled_seconds_per_launch"]
            self.compiled_seconds_per_launch = float(measured)
        self.floor_seconds = float(floor_seconds)

        with open(self.prices_path, encoding="utf-8") as fh:
            price_blob = json.load(fh)
        # Reindexed onto the cost key, the same function a lookup applies to
        # the operator it is pricing. Old files were written under the raw
        # signature and would otherwise stop matching the moment the lookup
        # side normalises; this is the reindex, done in memory, so no retained
        # artifact is rewritten or recollected.
        price_list = {cost_key(k): v
                      for k, v in (price_blob["prices"] or {}).items()}
        # The width these prices were measured at. None means the list cannot
        # certify one, which is read as "refuse" rather than "any width" -- see
        # the collective guard in `_cost`.
        self.price_topology = _declared_topology(price_blob, self.prices_path)

        self.unpriced = 0
        #: Collectives refused a price because the list was measured at another
        #: parallel width. Counted separately from `unpriced` so a transfer
        #: result can say the communication term is missing, not merely thin.
        self.untransferable_collectives = 0
        # A glob may name one decode graph or one per capture rung. Decode shapes
        # are not arbitrary: they are the rungs of the CUDA-graph ladder, known
        # from config before anything runs, so they can be *measured* rather than
        # interpolated -- which matters because interpolating a price across
        # shapes was probed and does not work (the library re-tunes its kernel at
        # nearly every shape, and a retune can cost 2.4x with nothing to warn
        # you).
        import glob as _glob

        paths = sorted(_glob.glob(self.graph_path)) or [self.graph_path]
        self.by_rung: dict[int, _Costed] = {}
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
            costed = self._cost(blob, price_list, path)
            self.by_rung[costed.batch or 0] = costed
        self.decode = (self.by_rung[max(self.by_rung)] if self.by_rung
                       else self._cost({"ops": []}, price_list, self.graph_path))

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
        self._warned_rung = False

        if self.unpriced:
            logger.warning(
                "ATOMCompass WARNING: %d operators have no price and contribute "
                "nothing; steps will be predicted low.", self.unpriced)

    def _cost(self, graph_blob: dict, price_list: dict, path: str) -> "_Costed":
        """What one graph costs, and the shape it is a graph of."""
        from atom.compass.core.cost.identity import cost_key
        from atom.compass.runtime.microbench import (
            _is_collective_op, signature_of)

        # A collective price is a price for a message over a group of a given
        # width, and the signature does not carry the width: `all_reduce_` over
        # 2 ranks and over 4 sign identically -- same message, same dtype, and
        # `unique_name` is `tp:0` either way. So a 4-way price matches a 2-way
        # call exactly, at full coverage, with no warning, and tensor-parallel
        # communication silently costs what it costs at the other width. The
        # width lives in the graph and in the price list, not in the operator,
        # so the check belongs here.
        graph_topology = dict(((graph_blob.get("key") or {}).get("topology")
                               or []))
        transferable = _collectives_transferable(
            graph_topology, self.price_topology)

        seconds = 0.0
        launches = 0
        priced_ops = 0
        # Per operator, not just the total: what an eager step pays for dispatch
        # depends on how each operator's own kernels compare to it, and a mean
        # cannot say that -- a step of one huge kernel and one tiny one hides
        # dispatch on the first and pays it on the second.
        kernel_seconds: list = []
        breakdown: dict[str, float] = {}
        for op in graph_blob["ops"]:
            if op.get("name", "") in HOST_SYNC:
                continue
            if not transferable and _is_collective_op(op):
                self.unpriced += 1
                self.untransferable_collectives += 1
                continue
            entry = price_list.get(cost_key(signature_of(op)))
            if entry is None:
                self.unpriced += 1
                continue
            priced_ops += 1
            kernel_seconds.append(entry["seconds"])
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
            kernel_seconds=tuple(kernel_seconds),
            launches=launches,
            breakdown=breakdown,
            is_prefill=bool(recorded.get("num_prefill_tokens", 0)),
            batch=len(recorded.get("num_scheduled_tokens") or []),
            context=(sum(contexts) / len(contexts)) if contexts else 0.0,
        )

    def _for_rung(self, shape: StepShape) -> "_Costed":
        """The decode graph for the rung this step replays.

        Exact match only. The rungs are enumerable and were measured, so there is
        nothing to interpolate and nothing that would justify it: a neighbouring
        shape can run a different tuned kernel and cost 2.4x more, and the price
        list cannot tell in advance which neighbours are safe. Where a rung was
        not measured this falls back to the largest that was, and says so once --
        an answer that is wrong by a known mechanism rather than by a silent one.
        """
        if len(self.by_rung) <= 1:
            return self.decode
        rung = shape.capture_bucket or shape.batch_size
        point = self.by_rung.get(rung)
        if point is not None:
            return point
        if not self._warned_rung:
            self._warned_rung = True
            logger.warning(
                "ATOMCompass WARNING: no decode graph for rung %s (have %s); "
                "using the largest measured one. Trace that rung to fix it -- "
                "prices are not interpolated across shapes on purpose.",
                rung, sorted(self.by_rung))
        return self.decode

    def estimate(self, shape: StepShape) -> StepCost:
        point = self.prefill if shape.is_prefill else self._for_rung(shape)
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
        if shape.capture_bucket is None and shape.compiled:
            # Neither dispatched one operator at a time nor submitted as one
            # graph, so neither of the terms below describes it. What such a
            # step pays is not an *addition* to its kernels at all: the host
            # runs ahead of the device, so the step lasts as long as the slower
            # of the two. Measured on the 0.6B over four prefill shapes with
            # the launch count fixed at 391, the device window sat at ~37 ms
            # whatever the kernels did until the kernels exceeded it:
            #
            #   tokens   kernels    window
            #      794    13.4 ms   36.7 ms   <- host-bound, 64% idle
            #     2294    27.0 ms   38.2 ms   <- host-bound, 29% idle
            #     6594   110.0 ms  110.1 ms   <- device-bound, 0% idle
            #    15694   418.2 ms  418.2 ms   <- device-bound, 0% idle
            #
            # `kernels + launches x constant` cannot express that: fitted to
            # the first row it over-predicts the last by 5.6%, fitted to the
            # last it under-predicts the first by 64%. A max does, with one
            # constant, to within 2% across all four.
            if self.host_seconds_per_launch:
                host = point.launches * self.host_seconds_per_launch
                total = max(max(point.seconds, host), self.floor_seconds)
                return StepCost(
                    seconds=total,
                    breakdown=dict(point.breakdown,
                                   **{"<host-bound>": max(0.0, host - point.seconds)}))
            overhead = point.launches * self.compiled_seconds_per_launch
        elif shape.capture_bucket is None:
            if self.eager_seconds_per_op is not None:
                overhead = point.ops * self.eager_seconds_per_op
            else:
                overhead = sum(max(0.0, self.dispatch_seconds - k)
                               for k in point.kernel_seconds)
        else:
            overhead = point.launches * self.boundary_seconds
        total = max(point.seconds + overhead, self.floor_seconds)
        return StepCost(seconds=total,
                        breakdown=dict(point.breakdown, **{"<overhead>": overhead}))

    def describe(self) -> str:
        prefill = (f"{self.prefill.seconds*1e3:.3f}ms prefill kernels"
                   if self.prefill else "no prefill graph")
        rungs = (f", {len(self.by_rung)} decode rungs {sorted(self.by_rung)}"
                 if len(self.by_rung) > 1 else "")
        return (f"PricedGraphCostOracle("
                f"{self.decode.seconds*1e3:.3f}ms decode kernels, {prefill}; "
                f"+{self.boundary_seconds*1e6:.2f}us/launch replayed, "
                + (f"+{self.eager_seconds_per_op*1e6:.1f}us/op eager"
                   if self.eager_seconds_per_op is not None
                   else f"{self.dispatch_seconds*1e6:.0f}us dispatch, hidden by "
                        f"kernels")
                + rungs + (", calibrated fallback" if self.fallback else "") + ")")
