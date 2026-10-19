# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The op-level backend: a step priced from the graph captured at its shape.

A step runs at one shape (`step_shape`): a decode step at the CUDA-graph rung
it replayed, or at its width when it replayed none or the backend prices real
rows (``padded=False``, the run file's ``padding: none``), and any other step
as a prefill of its query tokens over its rows. `graph(phase, tokens, seqs,
output)` gives the region of the `op_graph` captured at that shape, as
`capture.graph.to_json` writes it, or None when there is none. `output` is
whether the step samples, as ATOM's `ScheduledBatch.produces_output` says: a
step with a decode row does, and a prefill step when the view says so, which
it does when any row is a final chunk.

Each leaf is one pricing unit, `max(t0, F / peak, M / BW(M), t_comm)` over its
work (`op_work.work`): its own tensors at the captured shape, the state it
reads at the step's rows, attention's context in `kv`, the run's KV cache
format. A leaf that launches no kernel costs nothing. Regions compose by the
IR's joins: a `Seq` adds its items in order, a `Repeat` re-emits its body's
prices once per instance, a `max` `Par` emits its longest branch's prices, and a
`resource_bound` `Par` one roofline over its branches' summed work. The price
is one rank's device busy time: the runner takes a DP group's max and
composes host time.

Two-batch overlap. A step the data-parallel group split (`tbo_tokens`) runs
this rank's tokens as two micro-batches (`step_roofline.micro_batches`), each
priced from the graph at its own prefill shape, the second with the step's
output and the first with none. Each graph's prices in run order fall into
segments that alternate streams: an MoE exchange of activations rides the
comm stream, everything else the compute stream. Both micro-batches' segments
go on the two streams in ATOM's order (`step_roofline.overlap`); the finish
time is the one term ``op.tbo``.

Terms. One per op kind, ``op.<kind>``, and ``par.resource_bound``, each
the fold of its prices in run order. A term is `declared`, its note naming
each input, when a spec field the winning bound of one of its prices read is
neither from a datasheet nor probed, and `analytical` otherwise.

Refused: a step with image tokens, since no graph holds the vision encoder; a
two-batch-overlap step on MoRI's split send and receive (`split_send_recv`),
whose transfer in flight lies on no stream, whose split names no pair for
this rank, whose micro-batches' graphs exchange a different number of times
or run an exchange beside a second stream (the dual-stream MoE, which ATOM
turns off under overlap), or whose ranks schedule different rows in one
micro-batch, since capture runs one rank and its peers at this rank's rows; a speculative decode step; a
prefill step whose view does not say whether it produces output; a shape with
no graph; and, each reason once, a data-parallel step's (`moe_rows`) MoE
exchange leaf, for that same reason, and a leaf whose kind has no extractor,
whose price reads a spec field the spec lacks, or whose collective leaves the
scale-up domain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost, fold_seconds
from atom.compass.backends.op_work import work
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import IMAGES, BatchView
from atom.compass.backends.step_roofline import _reads, micro_batches, overlap
from atom.compass.backends.work import Work, bounds

SOURCE = "op-graph"
#: The leaves a captured MoE layer exchanges its DP or EP group's rows by:
#: ATOM's gathers and scatters, and MORI's dispatch and combine.
MOE_EXCHANGES = ("atom.model_ops.moe:", "mori.ops.dispatch_combine:")
#: The term a two-batch-overlap step's one price is.
TBO = "op.tbo"
NONE = "; no successor: the analytical law is the last op-level source"


def step_shape(
    rows: Sequence[tuple[int, bool]], rung: int | None
) -> tuple[str, int, int]:
    """The shape a step of `rows`, each its query tokens and whether it
    decodes, runs at, ``(phase, tokens, seqs)``: a decode step at `rung` when it
    replayed one, else at its width, and any other step as a prefill of its
    query tokens over its rows. A speculative decode step is refused."""
    if not all(decode for _, decode in rows):
        return "prefill", sum(q for q, _ in rows), len(rows)
    if wide := [q for q, _ in rows if q != 1]:
        raise ValueError(
            f"decodes {wide[0]} tokens in a row: a speculative decode step is "
            "not captured"
        )
    width = rung or len(rows)
    return "decode", width, width


class OpGraphBackend(CostBackend):
    """A step as its shape's graph, one roofline per leaf, composed by its joins."""

    def __init__(
        self,
        machine,
        graph: Callable[[str, int, int], dict | None],
        *,
        tp: int = 1,
        kv: str,
        padded: bool = True,
        split_send_recv: bool = False,
    ) -> None:
        self.machine = machine
        self.graph = graph
        self.tp = tp
        self.kv = kv
        self.padded = padded
        self.split_send_recv = split_send_recv
        self.name = machine.values.get("name", "")
        self._stated_as: dict[str, str] = {}

    @property
    def granularity(self) -> Granularity:
        return Granularity.OP

    def price(self, batch_view: BatchView) -> StepCost:
        refused: list[Refusal] = []
        if batch_view.tbo_tokens is None or batch_view.image_tokens:
            at, region = self._region(batch_view)
            priced = self._priced(region, batch_view, refused)
        else:
            at, priced = self._overlapped(batch_view, refused)
        if refused:
            raise CostRefused(batch_view, list(dict.fromkeys(refused)))
        seconds: dict[str, list[float]] = {}
        stated: dict[str, list[str]] = {}
        for name, s, notes, *_ in priced:
            seconds.setdefault(name, []).append(s)
            stated.setdefault(name, []).extend(notes)
        terms = []
        for name, values in seconds.items():
            note = "; ".join(dict.fromkeys(stated[name]))
            method = Method.DECLARED if note else Method.ANALYTICAL
            detail = f"{len(values)} prices of the graph at {at}"
            if name == TBO:
                detail = f"two micro-batches overlapped, from the graphs at {at}"
            provenance = Provenance(method, detail, SOURCE, note=note and note + NONE)
            terms.append(CostTerm(name, fold_seconds(values), provenance))
        return StepCost(terms)

    def describe(self) -> str:
        rows = "the rows a graph pads to" if self.padded else "real rows"
        return f"op graph at TP {self.tp}, one roofline per leaf, at {rows}"

    def _region(self, view: BatchView) -> tuple[str, dict]:
        """The graph the step reads, by its shape and output, and its region."""
        if view.image_tokens:
            reason = IMAGES.format(view.image_tokens)
        else:
            try:
                rung = view.capture_rung if self.padded else None
                rows = [(r.query_tokens, r.decode) for r in view.requests]
                shape = step_shape(rows, rung)
            except ValueError as exc:
                reason = f"a step that {exc}"
            else:
                output = any(d for _, d in rows) or view.produces_output
                none = "" if output else " that produces no output"
                if output is None:
                    reason = (
                        "a prefill step whose view does not say whether it "
                        "produces output, so whether it runs the LM head and the "
                        "sampler"
                    )
                elif (region := self.graph(*shape, output)) is not None:
                    return "{} T={} B={}".format(*shape) + none, region
                else:
                    reason = (
                        "no op_graph holds a {} step of {} tokens over {} sequences"
                        "{}; compass trace ops captures one".format(*shape, none)
                    )
        raise CostRefused(view, [Refusal(SOURCE, reason)])

    def _overlapped(self, view: BatchView, refused: list) -> tuple[str, list]:
        """A split step's shapes and its one price: both micro-batches' graphs
        on one compute and one comm stream in ATOM's order."""
        try:
            if self.split_send_recv:
                raise ValueError(
                    "a two-batch-overlap step on MoRI's split send and receive "
                    "(--all2all-backend low-latency): its transfer in flight "
                    "lies on no stream"
                )
            halves = micro_batches(view)
            if uneven := [t for t in view.tbo_tokens if len(set(t)) > 1]:
                raise ValueError(
                    "a two-batch-overlap step whose ranks schedule "
                    f"{uneven[0]} rows of one micro-batch: capture runs one rank, "
                    "so its graph holds every peer at this rank's rows"
                )
        except ValueError as exc:
            raise CostRefused(view, [Refusal(SOURCE, str(exc))]) from None
        at, runs, notes = [], [], []
        for k, half in enumerate(halves):
            # Every peer schedules this rank's rows: the graph's own exchange.
            output = k == 1 and view.produces_output
            half = replace(half, moe_rows=None, produces_output=output)
            shape, region = self._region(half)
            at.append(shape)
            run = [0.0]
            for _, seconds, stated, _, comm in self._priced(region, half, refused):
                if comm is None:
                    reason = (
                        f"a two-batch-overlap step whose micro-batch graph at {shape} "
                        "runs an MoE exchange beside a second stream, as the "
                        "dual-stream MoE does, which ATOM turns off under overlap"
                    )
                    raise CostRefused(view, [Refusal(SOURCE, reason)])
                notes += stated
                if comm != (len(run) % 2 == 0):
                    run.append(0.0)
                run[-1] += seconds
            runs.append(run)
        if len(runs[0]) != len(runs[1]):
            reason = (
                "a two-batch-overlap step whose micro-batches' graphs change "
                f"streams {len(runs[0]) - 1} and {len(runs[1]) - 1} times"
            )
            raise CostRefused(view, [Refusal(SOURCE, reason)])
        return " and ".join(at), [(TBO, overlap([zip(*runs)]), notes, Work(), False)]

    def _priced(self, region: dict, view: BatchView, refused: list) -> list:
        """`region`'s prices in run order, each ``(term, seconds, stated, work,
        comm)``, `work` what it adds to an enclosing resource_bound, `comm`
        that it rides the comm stream under two-batch overlap, None when it
        joins an MoE exchange to work on another stream; what refuses goes to
        `refused`."""
        if "op" in region:
            leaf = self._leaf(region, view, refused)
            comm = region["op"].startswith(MOE_EXCHANGES) and any(
                isinstance(a, dict) and a.get("role") == "activation"
                for a in region["args"].values()
            )
            return [(f"op.{region['op']}", *leaf, comm)] if leaf else []
        if "repeat" in region:
            return self._priced(region["repeat"], view, refused) * region["count"]
        if "seq" in region:
            return [p for i in region["seq"] for p in self._priced(i, view, refused)]
        branches = [self._priced(b, view, refused) for b in region["par"]]
        summed = sum((p[3] for b in branches for p in b), Work())
        comm = None if any(p[4] is not False for b in branches for p in b) else False
        if region["join"] == "max":
            each = [fold_seconds(p[1] for p in b) for b in branches]
            longest = branches[each.index(max(each))]
            # The longest branch's prices stay under their own terms; the
            # first carries every branch's work for an enclosing resource_bound.
            return [
                (*p[:3], summed if i == 0 else Work(), comm)
                for i, p in enumerate(longest)
            ]
        name = f"par.{region['join']}"
        unit = self._unit(name, summed, refused)
        return [(name, *unit, summed, comm)] if unit else []

    def _leaf(self, leaf: dict, view: BatchView, refused: list):
        """One leaf's ``(seconds, stated, work)``; None when it launches no
        kernel or refuses."""
        if view.moe_rows is not None and leaf["op"].startswith(MOE_EXCHANGES):
            reason = (
                f"a data-parallel step whose MoE gathers {view.moe_rows} rows: "
                "capture runs one rank, so its graph holds the gather, the "
                "reduce-scatter and the experts at every peer scheduling this "
                "rank's rows"
            )
            refused.append(Refusal(SOURCE, reason))
            return None
        try:
            w = work(leaf, view, self.tp, self.kv)
        except CostRefused as exc:
            refused.extend(exc.declined)
            return None
        except ValueError as exc:
            refused.append(Refusal(SOURCE, f"{leaf['op']}: {exc}"))
            return None
        unit = w and self._unit(leaf["op"], w, refused)
        return unit and (*unit, w)

    def _unit(self, what: str, w: Work, refused: list):
        """One pricing unit's seconds and the stated spec fields its largest
        bound read; None, with its refusal, when it cannot be priced."""
        reads = _reads(w)
        values = self.machine.values
        missing = [p for ps in reads.values() for p in ps if p not in values]
        if missing:
            reason = (
                f"{what} reads {', '.join(dict.fromkeys(missing))}, which the "
                "machine spec does not state"
            )
            refused.append(Refusal(SOURCE, reason))
            return None
        try:
            b = bounds(w, self.machine)
        except ValueError as exc:
            refused.append(Refusal(SOURCE, f"{what}: {exc}"))
            return None
        won = max(b, key=b.get)
        return b[won], [s for p in reads[won] if (s := self._stated(p))]

    def _stated(self, path: str) -> str:
        """`path` and its origin when it is neither a datasheet's nor probed."""
        if path not in self._stated_as:
            origin = self.machine.origin(path)["origin"]
            spec = f"{path} {origin} in the machine spec {self.name}"
            self._stated_as[path] = "" if origin in ("datasheet", "probed") else spec
        return self._stated_as[path]
