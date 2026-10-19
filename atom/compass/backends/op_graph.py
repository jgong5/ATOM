# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The op-level backend: a step priced from the graph captured at its shape.

A step runs at one shape (`step_shape`): a decode step at the CUDA-graph rung
it replayed, or at its width when it replayed none or the backend prices real
rows (``padded=False``, the run file's ``padding: none``), and any other step
as a prefill of its query tokens over its rows. `graph(phase, tokens, seqs)`
gives the region of the `op_graph` captured at that shape, as
`capture.graph.to_json` writes it, or None when there is none.

Each leaf is one pricing unit, `max(t0, F / peak, M / BW(M), t_comm)` over its
work (`op_work.work`): its own tensors at the captured shape, the state it
reads at the step's rows, attention's context in `kv`, the run's KV cache
format. A leaf that launches no kernel costs nothing. Regions compose by the
IR's joins: a `Seq` adds its items in order, a `Repeat` re-emits its body's
prices once per instance, a `max` `Par` emits its longest branch's prices, and a
`resource_bound` `Par` one roofline over its branches' summed work. The price
is one rank's device busy time: the runner takes a DP group's max and
composes host time.

With a `price_list`, a source named `LISTED` (`fit.ops.PriceList` in a run),
each leaf outside a `resource_bound` `Par` resolves through a `Resolver`: the
price list, asked ``(leaf, bound, work)`` with `bound` the step's shape and
its rows' `band`, then its roofline. A `resource_bound` `Par` prices its
branches' summed work, so its leaves take no price of their own.

Terms. One per op kind, ``op.<kind>``, and ``par.resource_bound``, each
the fold of its prices in run order. A term is `declared`, its note naming
each input, when a spec field the winning bound of one of its prices read is
neither from a datasheet nor probed, and `analytical` otherwise; it carries
each reason the price list declined one of its leaves. A price list's prices
are ``op.<kind>.price_list``, `measured` when each sat on a measured point
and `estimated` otherwise.

Refused: a step with image tokens, since no graph holds the vision encoder; a
two-batch-overlap step, since no graph holds a micro-batch or the MoE
exchanges its comm stream carries; a data-parallel step (`moe_rows`), since
capture runs one rank; a speculative decode step; a shape with no
graph; and, each reason once, a leaf whose kind has no extractor, whose price
reads a spec field the spec lacks, or whose collective leaves the scale-up
domain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost, fold_seconds
from atom.compass.backends.op_work import work
from atom.compass.backends.provenance import Method, Provenance, Refusal, combine
from atom.compass.backends.resolver import CostRefused, CostSource, Resolver
from atom.compass.backends.shape import IMAGES, BatchView
from atom.compass.backends.step_roofline import _reads
from atom.compass.backends.work import Work, bounds

SOURCE = "op-graph"
LISTED = "price-list"
NONE = "; no successor: the analytical law is the last op-level source"


class _Roofline(CostSource):
    """A leaf's roofline, as `OpGraphBackend._unit` prices it."""

    name = SOURCE

    def __init__(self, backend: OpGraphBackend) -> None:
        self.backend = backend

    def price(self, request: tuple) -> CostTerm:
        (leaf, _, w), refused = request, []
        if isinstance(w, tuple):
            raise CostRefused(request, w)
        kind = leaf["op"]
        unit = self.backend._unit(kind, w, refused)
        if unit is None:
            raise CostRefused(request, refused)
        note = "; ".join(unit[1])
        method = Method.DECLARED if note else Method.ANALYTICAL
        return CostTerm(f"op.{kind}", unit[0], Provenance(method, note=note))


def band(rows, decode: bool) -> tuple[int, int]:
    """The mean and max context of `rows`, each ``[query, context, ...]``: a
    decode row's context, a prefill row's cached tokens."""
    c = [row[1] - (0 if decode else row[0]) for row in rows]
    return sum(c) // len(c), max(c)


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
        price_list: CostSource | None = None,
    ) -> None:
        self.machine = machine
        self.graph = graph
        self.tp = tp
        self.kv = kv
        self.padded = padded
        self.name = machine.values.get("name", "")
        self._stated_as: dict[str, str] = {}
        self.resolver = price_list and Resolver([price_list, _Roofline(self)])

    @property
    def granularity(self) -> Granularity:
        return Granularity.OP

    def price(self, batch_view: BatchView) -> StepCost:
        shape, region = self._region(batch_view)
        refused: list[Refusal] = []
        bound = self.resolver and self._bound(shape, batch_view)
        priced = self._priced(region, batch_view, refused, bound)
        if refused:
            raise CostRefused(batch_view, list(dict.fromkeys(refused)))
        seconds: dict[str, list[float]] = {}
        stated: dict[str, list[str]] = {}
        how: dict[str, tuple[set, list]] = {}
        for name, s, notes, _, (method, declined) in priced:
            seconds.setdefault(name, []).append(s)
            stated.setdefault(name, []).extend(notes)
            methods, why = how.setdefault(name, (set(), []))
            methods.add(method)
            why.extend(declined)
        at = "{} T={} B={}".format(*shape)
        terms = []
        for name, values in seconds.items():
            note = "; ".join(dict.fromkeys(stated[name]))
            methods, why = how[name]
            method = Method.DECLARED if note else Method.ANALYTICAL
            detail = f"{len(values)} prices of the {at} graph"
            source, why = (SOURCE, tuple(dict.fromkeys(why)))
            if None not in methods:
                method, source = combine(methods), LISTED
            provenance = Provenance(
                method, detail, source, why, note=note and note + NONE
            )
            terms.append(CostTerm(name, fold_seconds(values), provenance))
        return StepCost(terms)

    def describe(self) -> str:
        rows = "the rows a graph pads to" if self.padded else "real rows"
        first = "a price list, then " if self.resolver else ""
        return f"op graph at TP {self.tp}, {first}one roofline per leaf, at {rows}"

    @staticmethod
    def _bound(shape: tuple[str, int, int], view: BatchView) -> tuple:
        """What a price list reads of a step: its phase, its T and B, and the
        mean and max context of its rows (`band`)."""
        rows = [(r.query_tokens, r.context_tokens) for r in view.requests]
        return (*shape, *band(rows, shape[0] == "decode"))

    def _region(self, view: BatchView) -> tuple[tuple[str, int, int], dict]:
        """The step's shape and the region captured at it."""
        if view.image_tokens:
            reason = IMAGES.format(view.image_tokens)
        elif view.tbo_tokens is not None:
            reason = (
                "a two-batch-overlap step: no captured graph holds a micro-batch, "
                "nor the MoE exchanges its comm stream carries"
            )
        elif view.moe_rows is not None:
            reason = (
                f"a data-parallel step whose MoE gathers {view.moe_rows} rows: "
                "capture runs one rank, so no graph holds the gather, the "
                "reduce-scatter or the experts at the group's rows"
            )
        else:
            try:
                rung = view.capture_rung if self.padded else None
                rows = [(r.query_tokens, r.decode) for r in view.requests]
                shape = step_shape(rows, rung)
            except ValueError as exc:
                reason = f"a step that {exc}"
            else:
                region = self.graph(*shape)
                if region is not None:
                    return shape, region
                reason = (
                    "no op_graph holds a {} step of {} tokens over {} sequences; "
                    "compass trace ops captures one".format(*shape)
                )
        raise CostRefused(view, [Refusal(SOURCE, reason)])

    def _priced(self, region: dict, view: BatchView, refused: list, bound) -> list:
        """`region`'s prices in run order, each ``(term, seconds, stated, work,
        (method, declined))``, `work` what it adds to an enclosing
        resource_bound, `method` a price list's and `declined` its reasons for
        a roofline; what refuses goes to `refused`. With a `bound`, a leaf
        resolves through the price list first."""
        if "op" in region:
            leaf = self._leaf(region, view, refused, bound)
            return [leaf] if leaf else []
        if "repeat" in region:
            body = self._priced(region["repeat"], view, refused, bound)
            return body * region["count"]
        if "seq" in region:
            return [
                p for i in region["seq"] for p in self._priced(i, view, refused, bound)
            ]
        pooled = region["join"] == "resource_bound"
        branches = [
            self._priced(b, view, refused, None if pooled else bound)
            for b in region["par"]
        ]
        summed = sum((p[3] for b in branches for p in b), Work())
        if region["join"] == "max":
            each = [fold_seconds(p[1] for p in b) for b in branches]
            longest = branches[each.index(max(each))]
            # The longest branch's prices stay under their own terms; the
            # first carries every branch's work for an enclosing resource_bound.
            return [
                (*p[:3], summed if i == 0 else Work(), p[4])
                for i, p in enumerate(longest)
            ]
        name = f"par.{region['join']}"
        unit = self._unit(name, summed, refused)
        return [(name, *unit, summed, (None, ()))] if unit else []

    def _leaf(self, leaf: dict, view: BatchView, refused: list, bound):
        """One leaf's price as `_priced` lists it; None when it launches no
        kernel or refuses."""
        try:
            w = work(leaf, view, self.tp, self.kv)
        except CostRefused as exc:
            w = exc.declined
        except ValueError as exc:
            w = (Refusal(SOURCE, f"{leaf['op']}: {exc}"),)
        if w is None:
            return None
        if bound:
            try:
                got = self.resolver.resolve((leaf, bound, w))
            except CostRefused as exc:
                refused.extend(exc.declined)
                return None
            term, listed = got.term, got.source == LISTED
            notes = term.provenance.note.split("; ") if term.provenance.note else []
            name = f"{term.name}.price_list" if listed else term.name
            how = (term.provenance.method if listed else None, got.declined)
            return name, term.seconds, notes, Work(), how
        if isinstance(w, tuple):
            refused.extend(w)
            return None
        unit = self._unit(leaf["op"], w, refused)
        return unit and (f"op.{leaf['op']}", *unit, w, (None, ()))

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
