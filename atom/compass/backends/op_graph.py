# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The op-level backend: a step priced from the graph captured at its shape.

A step runs at one shape (`step_shape`): a decode step at the CUDA-graph rung
it replayed, or at its width when it replayed none or the backend prices real
rows (``padded=False``, the run file's ``padding: none``), and any other step
as a prefill of its query tokens over its rows; a data-parallel step also at
the rows its MoE gathers over the group (`moe_rows`). `graph(phase, tokens,
seqs, output[, gathered])` gives the region of the `op_graph` captured at that
shape, as `capture.graph.to_json` writes it, or when there is none None or
the text its refusal adds (`forward.missed`), `gathered` stated for a
data-parallel step. `output` is whether the step
samples, as ATOM's `ScheduledBatch.produces_output` says: a step with a decode
row does, and a prefill step when the view says so, which it does when any row
is a final chunk.

Under a drafter of `drafts` tokens a decode step verifies `drafts` and one in
each row, and a step also reads its drafter's graph, `graph(..., draft=True)`,
the work ATOM queues after `forward_done_event` on a step that samples, else
from its first draft pass on: its prices are its `draft_terms`, named
``draft op.<kind>``. Sampling is the target's. A decode's drafter is read at its
rows run at the step's width: its first pass runs the rows' tokens unpadded.
A target that replays no graph yet pads past its rows (`target_bs`) is read
at that width, `W=`: an eager decode's model, a sampling prefill's LM head.
A block drafter (`block`) runs its draft-KV pass between its target's forward
and the sampling on a step that samples: that step also reads the pass's
graph, `graph(..., draft=KV)`, at the target's shape, and prices it among the
target's terms, named ``draft_kv op.<kind>``, since the step's output waits
for it. On one rank, on a step that samples, a drafter's experts run each
pass's own rows (`_pass_views`): an MTP drafter's first pass the step's
tokens, each later pass a token a row, and a block drafter's block `drafts`
tokens a row.

Each leaf is one pricing unit, `max(t0, F / peak, M / BW(M), t_comm)` over its
work (`op_work.work`): its own tensors at the captured shape, the state it
reads at the step's rows, attention's context in `kv`, the run's KV cache
format, the routed experts over the run's `ep` ranks. A leaf that launches
no kernel costs nothing. A step prices once the leaves of a kind that agree on
every part of a leaf the kind's work has read, a leaf whose work reads nothing
of the step keeps its price and its work, or that it launches no kernel. A
graph keeps the prices of each of its leaves outside a `Par` and of each `Par`
(`_Plan`) until what they read of the step moves: its view, read by a work a
price takes, or an item of its bound a price list read (`_Read`). A later
step at the graph prices only those again, and folds again only the terms
whose prices they are. A step at the view of one of the last `VIEWED` steps,
reading the same graphs, prices as it did, as a DP rank's dummy forwards do.
Regions compose by the IR's joins: a `Seq` adds its items in order, a `Repeat`
re-emits its body's prices once per instance, a `max` `Par` emits its longest
branch's prices, and a `resource_bound` `Par` one roofline over its branches'
summed work, or with a price list its longest branch's prices where they are
longer. The price
is one rank's device busy time: the runner takes a DP group's max and
composes host time.

Two-batch overlap. A step the data-parallel group split (`tbo_tokens`) runs
this rank's tokens as two micro-batches (`step_roofline.micro_batches`), each
priced from the graph at its own prefill shape and the group's rows of it
(`moe_rows`), the most a rank runs of it its `group_max_tokens`, the second
with the step's output and the first with none,
each traced as a micro-batch (`forward.structure`).
Each graph's prices in run order fall into segments that alternate streams: an
MoE exchange of activations rides the comm stream, everything else the compute
stream. Both micro-batches' segments go on the two streams in ATOM's order
(`step_roofline.overlap`); the finish time is the one term ``op.tbo``.

With a `price_list`, a source named `LISTED` (`fit.ops.PriceList` in a run),
each leaf resolves through a `Resolver`: the
price list, asked ``(leaf, bound, work)`` with `bound` the step's or
micro-batch's (`_bound`) at the rows the leaf ran (`_ran_at`), or the list of
each it may have run at where it leads with a padded prefill's T that is also
a count of its width (`_ran_at_each`), then its roofline. `work` gives the
leaf's work when called, which the roofline alone does: a leaf known to launch
a kernel whose work reads the step has it worked out only where the list
declines it. A leaf inside a `resource_bound` `Par` also carries its work,
which the `Par`'s roofline sums.

Terms. One per op kind, ``op.<kind>``, and ``par.resource_bound``, each
the fold of its prices in run order. A term is `declared`, its note naming
each input, when a spec field the winning bound of one of its prices read is
neither from a datasheet nor probed, and `analytical` otherwise; it carries
each reason the price list declined one of its leaves. A price list's prices
are ``op.<kind>.price_list``, and ``op.tbo.price_list`` when it priced every
leaf of both micro-batches, `measured` when each sat on a measured point and
`estimated` otherwise, validated when each of its estimated leaves is.
A roofline term's `by_bound` is each price's seconds under its largest bound
(`work.binding`); a price list's term and ``op.tbo`` leave it empty.

Refused: a step with image tokens, since no graph holds the vision encoder; a
two-batch-overlap step on MoRI's split send and receive (`split_send_recv`),
whose transfer in flight lies on no stream, whose split names no pair for
this rank, whose micro-batches' graphs exchange a different number of times
or run an exchange beside a second stream (the dual-stream MoE, which ATOM
turns off under overlap), or that runs under a drafter, whose draft pass no
graph places on the overlapped streams; a decode step whose rows do not each
verify the drafter's tokens and one; a step that samples on one rank whose
MTP drafter's graph does not run its routed leaves as its passes alike, each
its own; a prefill step whose view
does not say whether it produces output; a shape with no graph, a
data-parallel step's naming the rows it gathers; and, each reason once, a leaf
whose kind has no extractor, whose price reads a spec field the spec lacks, or
whose collective leaves the scale-up domain.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost, fold_parts, fold_seconds
from atom.compass.backends.op_work import ROUTED, _tensors, work
from atom.compass.backends.provenance import Method, Provenance, Refusal, combine
from atom.compass.backends.resolver import CostRefused, CostSource, Resolver
from atom.compass.backends.shape import IMAGES, BatchView
from atom.compass.backends.step_roofline import _reads, micro_batches, overlap
from atom.compass.backends.work import Work, binding

SOURCE = "op-graph"
LISTED = "price-list"
#: A decode step's rows' mean context is read in these parts of their max.
PERMILLE = 1000
#: The leaves a captured MoE layer exchanges its DP or EP group's rows by:
#: ATOM's gathers and scatters, and MORI's dispatch and combine.
MOE_EXCHANGES = ("atom.model_ops.moe:", "mori.ops.dispatch_combine:")
#: The term a two-batch-overlap step's one price is.
TBO = "op.tbo"
NONE = "; no successor: the analytical law is the last op-level source"
#: The prefix of a drafter's terms.
DRAFT = "draft "
#: The prefix of a block drafter's draft-KV pass's terms on a step that
#: samples, and what its graph's structure adds to its drafter's.
DRAFT_KV, KV = "draft_kv ", "kv"
#: The leaves a backend keeps; past it, it forgets them all.
KEPT = 1 << 16
#: The graphs a backend keeps the prices of (`_Plan`), and the steps it keeps
#: the prices of by their views and graphs: as many as a run's graph lookup
#: keeps graphs (`run._op_graphs`), which parses a graph it dropped anew, so
#: a graph or step kept longer would only hold its graphs.
VIEWED = 64
#: What a price read of its step when it may have read anything of it: it
#: refused, or its work read a drafter's pass's view (`_pass_views`).
VIEW = None
#: The prefix of each part of a step's graphs' terms (`_Plan`): its target's,
#: its block drafter's draft-KV pass's and its drafter's.
PREFIXES = ("", DRAFT_KV, DRAFT)
#: A reference argument's keys: a context key or a module path, whose value a
#: leaf's work does not read.
REFS = ({"context"}, {"module"})
#: What a price list reads of a leaf, as `_Seen` paths.
LISTED_READS = {("args",), ("out",), ("stated",)}


class _Seen(Mapping):
    """A leaf, or its `args` or `stated`, that adds each path its reader
    looks up to `reads`: a key, present or not, or the path and None when it
    walks every key or takes `args` or `stated`, whose presence it reads."""

    def __init__(self, data: dict, reads: set, at: tuple = ()) -> None:
        self.data, self.reads, self.at = data, reads, at

    def __getitem__(self, key):
        path = self.at + (key,)
        if path in (("args",), ("stated",)) and isinstance(self.data.get(key), dict):
            self.reads.add(path + (None,))
            return _Seen(self.data[key], self.reads, path)
        self.reads.add(path)
        return self.data[key]

    def __iter__(self):
        self.reads.add(self.at + (None,))
        return iter(self.data)

    def __len__(self) -> int:
        self.reads.add(self.at + (None,))
        return len(self.data)


def _at(leaf: dict, path: tuple) -> list:
    """What a `_Seen` read of `path` in `leaf` returned: ``[1, value]``, the
    keys under it for a walk, a reference argument by its keys; or ``[0,
    depth]`` where the path ends."""
    value = leaf
    for depth, key in enumerate(path):
        if key is None:
            return [1, sorted(value)]
        if not isinstance(value, dict) or key not in value:
            return [0, depth]
        value = value[key]
    if path == ("args",):
        value = {s: _ref(v) for s, v in value.items()}
    elif path[:1] == ("args",):
        value = _ref(value)
    return [1, value]


class _Read(tuple):
    """A bound a price is asked at, one of a padded step's bounds or the
    step's own, that adds to `reads` the place in its step's bound laid flat
    (`_flat`) of each item its reader takes by index or slice, from `start`
    on, as `fit.ops.PriceList.price` reads a bound."""

    def __new__(cls, bound: tuple, start: int, reads: set):
        read = super().__new__(cls, bound)
        read.start, read.reads = start, reads
        return read

    def __getitem__(self, i):
        at = range(self.start, self.start + len(self))[i]
        self.reads.update(at if isinstance(at, range) else (at,))
        return super().__getitem__(i)


def _flat(bound) -> tuple:
    """A step's bound (`_bound`) laid flat, as `_Read` places what a price
    reads of it: a padded step's bounds at its rows, at its width and at
    both, then the counts that choose among them; none without a price list."""
    if not bound:
        return ()
    return (*bound[0], *bound[1], *bound[2], bound[3]) if len(bound) == 4 else bound


def _reading(bound: tuple, each: list, reads: set) -> list:
    """Each bound of `each` (`_ran_at_each`) as a `_Read` of `bound` into
    `reads`, with what choosing them read of a padded step's: its phase, its
    T and the counts that choose."""
    if len(bound) != 4:
        return [_Read(b, 0, reads) for b in each]
    starts = (0, len(bound[0]), len(bound[0]) + len(bound[1]))
    reads.update((0, 1, starts[2] + len(bound[2])))
    return [
        _Read(b, next(s for s, x in zip(starts, bound) if x is b), reads) for b in each
    ]


def _units(region: dict):
    """`region`'s units in run order: each leaf outside a `Par` and each
    `Par`, a repeat's body once an instance."""
    if "repeat" in region:
        for _ in range(region["count"]):
            yield from _units(region["repeat"])
    elif "seq" in region:
        for item in region["seq"]:
            yield from _units(item)
    else:
        yield region


def _term(name: str, prices: list, at: str) -> CostTerm:
    """The term `name` of `prices`, each as `_priced` lists it, from the graph
    at `at`: their seconds folded in run order; `declared` with a note naming
    each stated input, else `analytical`, or for a price list's term the
    methods of its prices combined, validated when each estimated one is."""
    note = "; ".join(dict.fromkeys(n for p in prices for n in p[2]))
    method = Method.DECLARED if note else Method.ANALYTICAL
    source, validated = SOURCE, False
    if name.endswith(".price_list"):
        origins = [o for p in prices for o in p[5][0]]
        method, source = combine(o.method for o in origins), LISTED
        validated = method is Method.ESTIMATED and all(
            o.validated for o in origins if o.method is Method.ESTIMATED
        )
    detail = f"{len(prices)} prices of the graph at {at}"
    if name.startswith(TBO):
        detail = f"two micro-batches overlapped, from the graphs at {at}"
    provenance = Provenance(
        method,
        detail,
        source,
        tuple(dict.fromkeys(r for p in prices for r in p[5][1])),
        note=note and note + NONE,
        validated=validated,
    )
    seconds = fold_seconds(p[1] for p in prices)
    return CostTerm(name, seconds, provenance, fold_parts(p[6] for p in prices))


def _cost(terms: dict) -> StepCost:
    """A step's cost of its `terms` by name, its drafter's apart."""
    split: dict[bool, list[CostTerm]] = {False: [], True: []}
    for name, term in terms.items():
        split[name.startswith(DRAFT)].append(term)
    return StepCost(split[False], draft_terms=split[True])


class _Plan:
    """A step's graphs, its target's, its block drafter's draft-KV pass's and
    its drafter's (`PREFIXES`), as a backend prices them step after step:
    their units (`_units`) in run order, each with its part, its prices as
    `_priced` lists them, and what they read of the step, by part: the places
    in its bound laid flat (`_flat`), the names of the view's attributes, or
    `VIEW`; the units by what they read, and every attribute they read; the
    last step's view and bounds laid flat; and the terms in run order, each
    with the units that hold its prices. `held` keeps the ids of the step's
    graphs theirs; `at` is its target's shape and its drafter's. A backend
    plans a graph at its second step."""

    def __init__(self, held: tuple, at: tuple) -> None:
        self.held, self.at = held, at
        # Lists of plain values and shared objects: a plan lives across steps,
        # and the collector walks every object it holds.
        self.units: list[dict] = []
        self.parts: list[int] = []
        for part, region in enumerate(held):
            if region:
                self.units += _units(region)
                self.parts += [part] * (len(self.units) - len(self.parts))
        self.priced: list = [()] * len(self.units)
        self.reads: list = [None] * len(self.units)
        # Each part's bound laid flat at the last step, None until one priced
        self.flat: list | None = None
        self.view: BatchView | None = None
        self.groups: dict[tuple, list[int]] = {}
        self.attrs: list[str] = []
        self.names: dict[str, list[int]] = {}
        self.terms: dict[str, CostTerm] = {}


def _ref(value):
    return sorted(value) if isinstance(value, dict) and value.keys() in REFS else value


def _ridden(leaf: dict, got: tuple) -> tuple:
    """`leaf`'s price `got` as `_priced` lists it: with whether it rides the
    comm stream under two-batch overlap, an MoE exchange of activations."""
    comm = leaf["op"].startswith(MOE_EXCHANGES) and any(
        isinstance(a, dict) and a.get("role") == "activation"
        for a in leaf["args"].values()
    )
    return (*got[:4], comm, *got[4:])


def _run(region: dict):
    """`region`'s leaves in run order, a repeat's body once an instance."""
    if "op" in region:
        yield region
    elif "repeat" in region:
        for _ in range(region["count"]):
            yield from _run(region["repeat"])
    else:
        for item in region.get("seq", region.get("par", ())):
            yield from _run(item)


class _Watched:
    """A step's view that notes each attribute a leaf's work reads of it."""

    def __init__(self, view: BatchView) -> None:
        self.view, self.read = view, set()

    def __getattr__(self, name: str):
        self.read.add(name)
        return getattr(self.view, name)


class _Roofline(CostSource):
    """A leaf's roofline, as `OpGraphBackend._unit` prices it, over the work
    its request's call gives."""

    name = SOURCE

    def __init__(self, backend: OpGraphBackend) -> None:
        self.backend = backend

    def price(self, request: tuple) -> CostTerm:
        (leaf, _, later), refused = request, []
        w = later()
        if isinstance(w, tuple):
            raise CostRefused(request, w)
        kind = leaf["op"]
        unit = self.backend._unit(kind, w, refused)
        if unit is None:
            raise CostRefused(request, refused)
        note = "; ".join(unit[1])
        method = Method.DECLARED if note else Method.ANALYTICAL
        return CostTerm(
            f"op.{kind}", unit[0], Provenance(method, note=note), by_bound=unit[2]
        )


def band(rows, decode: bool) -> tuple[int, int]:
    """The mean and max context of `rows`, each ``[query, context, ...]``: a
    decode row's context, a prefill row's cached tokens."""
    c = [row[1] - (0 if decode else row[0]) for row in rows]
    return sum(c) // len(c), max(c)


def step_shape(
    rows: Sequence[tuple[int, bool]], rung: int | None, drafts: int = 0
) -> tuple[str, int, int]:
    """The shape a step of `rows`, each its query tokens and whether it
    decodes, runs at, ``(phase, tokens, seqs)``: a decode step at `rung` when it
    replayed one, else at its width, and any other step as a prefill of its
    query tokens over its rows. A decode step whose rows do not each verify a
    drafter's `drafts` tokens and one is refused."""
    if not all(decode for _, decode in rows):
        return "prefill", sum(q for q, _ in rows), len(rows)
    if any(q != drafts + 1 for q, _ in rows):
        raise ValueError(
            f"decodes a row of other than {drafts + 1} tokens, which a drafter "
            f"of {drafts} verifies"
            if drafts
            else "decodes more than one token in a row: a speculative decode step "
            "is not captured"
        )
    width = rung or len(rows)
    return "decode", width, width


def gathered_off_default(
    phase: str, tokens: int, gathered: int | None, dp: int, drafts: int = 0
) -> int | None:
    """`gathered`, a DP step's MoE rows, or None at their default: `dp` ranks
    at `tokens`, a decode's rows each verifying `drafts` and one."""
    rows = tokens * (drafts + 1 if phase == "decode" else 1)
    return None if gathered == dp * rows else gathered


def real_rows(own: int, gathered: int | None, tokens: int | None, dummies: int):
    """The real rows a decode step's MoE runs, its others pad rows that route
    to no expert: where its DP group's MoE gathers `gathered` rows, the group's
    `tokens` with its `dummies` dummy forwards' rows as one, every dummy
    running the same row, or None where it states no tokens; else its `own`."""
    if gathered is None:
        return own
    return tokens and tokens - max(dummies - 1, 0)


def _ran_at(leaf: dict, bound: tuple) -> tuple:
    """`bound` at the rows `leaf` ran: a padded target's or drafter's
    (`_bound`) at its width where an activation, metadata or output of the
    leaf leads with the width's counts and none with the rows', at both
    (`_bound`) where it leads with both's, else at its rows."""
    if len(bound) != 4:
        return bound
    rows, wide, both, (counts, wides) = bound
    args = _tensors(list(leaf["args"].values()))
    shapes = [a["shape"] for a in args if a["role"] in ("activation", "metadata")]
    leads = {n for s in shapes + leaf["out"] for n in s[:1]}
    return (both if leads & counts else wide) if leads & wides else rows


def _ran_at_each(leaf: dict, bound: tuple) -> list:
    """`_ran_at`'s bound, then, where a padded prefill's T is also a count of
    its width, each other bound `leaf` may have run at: that T its rows'
    count alone, or its width's alone."""
    each = [_ran_at(leaf, bound)]
    if len(bound) == 4 and bound[0][0] == "prefill":
        *at, (counts, wides) = bound
        t = {at[0][1]} & wides
        for sets in ((counts - t, wides), (counts, wides - t)):
            if (got := _ran_at(leaf, (*at, sets))) not in each:
                each.append(got)
    return each


class OpGraphBackend(CostBackend):
    """A step as its shape's graph, one roofline per leaf, composed by its joins."""

    def __init__(
        self,
        machine,
        graph: Callable[..., dict | None],
        *,
        tp: int = 1,
        ep: int = 1,
        dp: int = 1,
        kv: str,
        padded: bool = True,
        split_send_recv: bool = False,
        price_list: CostSource | None = None,
        drafts: int = 0,
        block: bool = False,
    ) -> None:
        self.machine = machine
        self.graph = graph
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.kv = kv
        self.padded = padded
        self.split_send_recv = split_send_recv
        self.drafts = drafts
        self.block = block
        self.name = machine.values.get("name", "")
        self._stated_as: dict[str, str] = {}
        # By leaf object: the leaf, the text of what its price reads of it, and
        # its price at a bound, with what it read of the bound, when it read
        # nothing else of the step; by view, bound and that text: one step's
        # prices. By op kind: every path its work has read of a leaf. By the
        # graphs a step reads: their `_Plan`, or after their first step the
        # graphs alone.
        self._kept: dict[int, list] = {}
        self._step: dict[tuple, tuple] = {}
        self._reads: dict[str, set] = {}
        self._plans: dict[tuple, _Plan | tuple] = {}
        # By view and graphs: a step's price, while `_plans` holds the graphs.
        self._viewed: dict[tuple, StepCost] = {}
        # What the unit being priced has read of its step (`_Plan`), and each
        # value of what a price or a unit read, one object for all that hold it.
        self._read: set = set()
        self._interned: dict = {}
        # By leaf object: the view of the rows its pass runs, for a routed
        # leaf of a drafter's pass that does not run the step's tokens, this
        # step (`_pass_views`).
        self._passes: dict[int, BatchView] = {}
        self.resolver = price_list and Resolver([price_list, _Roofline(self)])

    @property
    def granularity(self) -> Granularity:
        return Granularity.OP

    def price(self, batch_view: BatchView) -> StepCost:
        refused: list[Refusal] = []
        self._step, self._passes, self._read = {}, {}, set()
        if batch_view.tbo_tokens is not None and not batch_view.image_tokens:
            at, priced = self._overlapped(batch_view, refused)
            if refused:
                raise CostRefused(batch_view, list(dict.fromkeys(refused)))
            return StepCost([_term(priced[0][0], priced, at)])
        at, region = self._region(batch_view)
        drafted_at, drafted = (
            self._region(batch_view, draft=True) if self.drafts else (None, None)
        )
        samples = batch_view.decode or batch_view.produces_output
        kv = self.block and samples and self._region(batch_view, draft=KV)[1]
        graph = (at, id(region), drafted_at, id(drafted), id(kv))
        # A graph's first step keeps only the graphs, whose ids `graph` names,
        # and prices them whole: a graph a run reads once, as most prefill
        # shapes are, keeps no plan.
        held = self._plans.get(graph)
        if held is None:
            if len(self._plans) >= VIEWED:
                self._plans.clear()
                self._viewed.clear()
            self._plans[graph] = (region, kv, drafted)
        elif not isinstance(held, _Plan):
            self._plans[graph] = _Plan(held, (at, drafted_at))
        # A view priced lately at these graphs prices as it did, a DP rank's
        # dummy forwards among them.
        if (cost := self._viewed.get((batch_view, graph))) is not None:
            return cost
        bound = self.resolver and self._bound(batch_view)
        draft = self.resolver and self.drafts and self._bound(batch_view, draft=True)
        if drafted and samples and batch_view.moe_rows is None:
            self._passes = self._pass_views(drafted, batch_view)
        plan, bounds = self._plans[graph], (bound, bound, draft)
        if isinstance(plan, _Plan):
            cost = self._planned(plan, batch_view, bounds)
        else:
            cost = self._whole(plan, batch_view, bounds, (at, drafted_at))
        if len(self._viewed) >= VIEWED:
            self._viewed.clear()
        self._viewed[batch_view, graph] = cost
        return cost

    def _planned(self, plan: _Plan, view: BatchView, bounds: tuple) -> StepCost:
        """`plan`'s step at `view`, each part at its bound of `bounds`: every
        unit priced at its first step, after it only each whose prices read
        an attribute of the view or an item of its part's bound that moved
        since the last step, or `VIEW`; the terms of their prices folded
        again, all of them where their names moved."""
        flat, refused = [_flat(b) for b in bounds], []
        again: list[int] | range = range(len(plan.units))
        if plan.flat is not None and list(map(len, flat)) == list(map(len, plan.flat)):
            attrs = {a for a in plan.attrs if getattr(view, a) != getattr(plan.view, a)}
            moved = [
                attrs | {i for i, (now, then) in enumerate(zip(f, last)) if now != then}
                for f, last in zip(flat, plan.flat)
            ]
            again = sorted(
                u
                for (part, reads), units in plan.groups.items()
                if VIEW in reads or reads & moved[part]
                for u in units
            )
        renamed, regrouped = isinstance(again, range), isinstance(again, range)
        dirty: set[str] = set()
        plan.flat = None  # until every unit priced again prices
        for u in again:
            part, unit = plan.parts[u], plan.units[u]
            self._read = set()
            priced = self._priced(unit, view, refused, bounds[part])
            if part:
                priced = [(PREFIXES[part] + p[0], *p[1:]) for p in priced]
            renamed |= [p[0] for p in priced] != [p[0] for p in plan.priced[u]]
            reads = self._shared((part, self._shared(frozenset(self._read))))
            regrouped |= reads != plan.reads[u]
            plan.priced[u], plan.reads[u] = priced or (), reads
            dirty.update(p[0] for p in priced)
        if refused:
            raise CostRefused(view, list(dict.fromkeys(refused)))
        plan.flat, plan.view = flat, view
        if regrouped:
            plan.groups = {}
            for u, reads in enumerate(plan.reads):
                plan.groups.setdefault(reads, []).append(u)
            plan.attrs = sorted(
                {a for _, r in plan.groups for a in r if isinstance(a, str)}
            )
        if renamed:
            plan.names = {}
            for u, priced in enumerate(plan.priced):
                for p in priced:
                    units = plan.names.setdefault(p[0], [])
                    if units[-1:] != [u]:
                        units.append(u)
            dirty, plan.terms = set(plan.names), dict.fromkeys(plan.names)
        for name in sorted(dirty):
            prices = [
                p for u in plan.names[name] for p in plan.priced[u] if p[0] == name
            ]
            plan.terms[name] = _term(name, prices, plan.at[name.startswith(DRAFT)])
        return _cost(plan.terms)

    def _whole(self, held: tuple, view: BatchView, bounds: tuple, at: tuple):
        """A step at the graphs `held` priced whole, each part at its bound of
        `bounds`, the target's from the graph at `at[0]`, its drafter's at
        `at[1]`."""
        priced, refused = [], []
        for part, region in enumerate(held):
            if region:
                got = self._priced(region, view, refused, bounds[part])
                priced += (
                    [(PREFIXES[part] + p[0], *p[1:]) for p in got] if part else got
                )
        if refused:
            raise CostRefused(view, list(dict.fromkeys(refused)))
        names: dict[str, list] = {}
        for p in priced:
            names.setdefault(p[0], []).append(p)
        return _cost(
            {n: _term(n, ps, at[n.startswith(DRAFT)]) for n, ps in names.items()}
        )

    def _shared(self, reads):
        """`reads`, as the one object of its value the backend holds."""
        return self._interned.setdefault(reads, reads)

    def describe(self) -> str:
        rows = "the rows a graph pads to" if self.padded else "real rows"
        first = "a price list, then " if self.resolver else ""
        return f"op graph at TP {self.tp}, {first}one roofline per leaf, at {rows}"

    def _pass_views(self, region: dict, view: BatchView) -> dict:
        """The routed leaves of a drafter's `region` whose pass does not run
        the step's tokens, by object, each with the view of the rows its pass
        runs, on one rank on a step that samples: a block drafter's every one,
        its block `drafts` tokens a row; an MTP drafter's after its first
        pass, a token a row. Each of its `drafts` passes runs its model once,
        so its routed leaves in run order fall into `drafts` runs of the same
        kinds, the first its first pass's; one whose do not, or whose runs
        share a leaf, is refused. A routed leaf reads only a row's queries, so
        a row shorter than its pass's queries keeps that many tokens of
        context."""
        leaves = [leaf for leaf in _run(region) if leaf["op"] in ROUTED]
        first = 0 if self.block else len(leaves) // self.drafts
        later = {id(leaf): leaf for leaf in leaves[first:]}
        kinds = [leaf["op"] for leaf in leaves]
        if not self.block and (
            kinds != kinds[:first] * self.drafts
            or any(id(leaf) in later for leaf in leaves[:first])
        ):
            reason = (
                f"an MTP drafter's graph whose {len(leaves)} routed leaves do "
                f"not run as its {self.drafts} passes alike, each its own"
            )
            raise CostRefused(view, [Refusal(SOURCE, reason)])
        tokens = self.drafts if self.block else 1
        rows = tuple(
            replace(
                r, query_tokens=tokens, context_tokens=max(r.context_tokens, tokens)
            )
            for r in view.requests
        )
        return dict.fromkeys(later, replace(view, requests=rows))

    def _bound(self, view: BatchView, draft: bool = False) -> tuple:
        """What a price list reads of a step: its phase, its T and B, the rows
        its MoE gathers, None at their default (`gathered_off_default`),
        and two of its rows' contexts (`band`): a decode's max and its mean
        over its B, a pad row holding none; a prefill's mean and max cached
        tokens. A padded target
        (`target_bs`) adds it at that width, G at its default there, and the
        counts a leaf leads with at its rows and at that width: a decode's
        tokens, its MoE's gathered at the width; a prefill's, its width.
        With `draft`, its drafter's: its first pass at its rows, unpadded, G
        `draft_moe_rows` off its rows' default; its later passes at `draft_bs`,
        G every rank's `draft_bs`; the first pass's gathered a rows' count, the
        later passes' a width's. An eager DP drafter with later passes runs
        them at its rows, G `draft_rows`, a capture's MoE gathering this rank's
        rows at every peer. A leaf with both counts is asked at its rows'
        T and B and the smaller G: on a drafter its rows carry the first
        pass's G, above a later pass's. A block drafter's pass on a decode
        that replayed a rung runs its block at the rung: the width's counts
        are its block's, the rung's rows and their `drafts` tokens each, on
        this rank and gathered over the DP group, and none is its rows', so
        a leaf that leads with its rows alone or beside the target's tokens,
        as its anchors do, is asked at its rows. Last, a decode's target
        states the real rows its MoE runs (`real_rows`), and any other None;
        a target at a step stating G then the most tokens a rank of its DP
        group scheduled (`group_max_tokens`), where its view states them,
        which MORI's exchange there is priced by."""
        rung = view.capture_rung if self.padded and not draft else None
        rows = [(r.query_tokens, r.decode) for r in view.requests]
        phase, t, b = step_shape(rows, rung, self.drafts)
        width = self.padded and (view.draft_bs if draft else view.target_bs)
        block = draft and self.block and phase == "decode"
        if block:
            width = self.padded and view.capture_rung
        if draft and self.drafts > 1 and view.draft_rows:
            width = b
        ran = phase == "decode" and width or t
        gathered = view.draft_moe_rows if draft else view.moe_rows
        at = t if draft else ran
        g = gathered_off_default(phase, at, gathered, self.dp, self.drafts)
        rows = [(r.query_tokens, r.context_tokens) for r in view.requests]
        mean, top = band(rows, phase == "decode")
        bound = (phase, t, b, g, mean, top)
        if phase == "decode":  # its max, then its mean over B: a pad row holds none
            bound = (phase, t, b, g, top, sum(c for _, c in rows) / b)
        own = sum(r.query_tokens for r in view.requests)
        real = real_rows(own, view.moe_rows, view.moe_tokens, view.group_dummies)
        bound += (real if phase == "decode" and not draft else None,)
        if g is not None and not draft and view.group_max_tokens is not None:
            bound += (view.group_max_tokens,)
        if not width:
            return bound
        if ran == t:
            counts, wides = {t, b}, {width}
        else:
            k = range(1, self.drafts + 2)
            counts = {b * j for j in k}
            wides = {width * j for j in k} | {self.dp * width * k[-1]}
        wide, both = (phase, ran, width, *bound[3:]), bound
        if block:
            tokens = width * self.drafts
            counts, wides = set(), {width, tokens, self.dp * tokens}
        elif draft:
            padded = self.dp * width
            later = view.draft_rows or padded
            counts, wides = counts | {gathered}, {width, padded, later}
            wide = (phase, ran, width, gathered and later, *bound[4:])
            low = gathered and min(gathered, later)
            g = gathered_off_default(phase, t, low, self.dp, self.drafts)
            both = (phase, t, b, g, *bound[4:])
        if phase == "decode":
            wide = (*wide[:5], wide[5] * b / width, *wide[6:])
        return bound, wide, both, (frozenset(counts), frozenset(wides))

    def _region(
        self, view: BatchView, draft: bool | str = False, micro_batch: bool = False
    ) -> tuple[str, dict]:
        """The graph the step reads, by its shape and output, and its region;
        with `draft`, its drafter's, a decode's at its rows, at `draft_moe_rows`,
        its later passes at `draft_bs`; else a target that pads past its rows
        without a replay at `target_bs`, as a block drafter's draft-KV pass in
        its step is read (`draft` is `KV`). With `micro_batch`, a
        two-batch-overlap micro-batch's."""
        if view.image_tokens:
            reason = IMAGES
        else:
            try:
                rung = view.capture_rung if self.padded else None
                rows = [(r.query_tokens, r.decode) for r in view.requests]
                shape = step_shape(rows, rung, self.drafts)
            except ValueError as exc:
                reason = f"a step that {exc}"
            else:
                output = any(d for _, d in rows) or view.produces_output
                own = draft is True  # the drafter's passes, not its draft-KV pass
                width = self.padded and (view.draft_bs if own else view.target_bs)
                if own and shape[0] == "decode":
                    shape = ("decode", len(rows), width or shape[2])
                    width = None
                none = "" if output else " that produces no output"
                gathered = view.draft_moe_rows if own else view.moe_rows
                g = () if gathered is None else (gathered,)
                drafted = {"draft": draft} if draft else {}
                micro = {"micro_batch": True} if micro_batch else {}
                # A DP middle chunk runs its drafter's propose too.
                if (output or self.dp > 1) and width and width != shape[2]:
                    drafted["width"] = width
                at = f" W={width}" * ("width" in drafted)
                at += "".join(f" G={n}" for n in g)
                if output is None:
                    reason = (
                        "a prefill step whose view does not say whether it "
                        "produces output, so whether it runs the LM head and the "
                        "sampler"
                    )
                elif isinstance(
                    region := self.graph(*shape, output, *g, **drafted, **micro), dict
                ):
                    return "{} T={} B={}".format(*shape) + at + none, region
                else:
                    reason = (
                        "no op_graph holds a {} step of {} tokens over {} sequences"
                        "{}{}{}{}; compass trace ops captures one".format(
                            *shape,
                            "".join(f" gathering {n} rows" for n in g),
                            none,
                            " as a two-batch-overlap micro-batch" * micro_batch
                            + " for its drafter" * bool(draft)
                            + "'s draft-KV pass" * (draft == KV)
                            + f", {'its later passes at' if own else 'padded to'} {width}"
                            * ("width" in drafted),
                            region or "",
                        )
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
            if self.drafts:
                raise ValueError(
                    "a two-batch-overlap step under a drafter: no graph places its "
                    "draft pass on the overlapped streams"
                )
            halves = micro_batches(view)
        except ValueError as exc:
            raise CostRefused(view, [Refusal(SOURCE, str(exc))]) from None
        at, runs, notes, origins, why = [], [], [], [], []
        listed = bool(self.resolver)
        for k, half in enumerate(halves):
            output = k == 1 and view.produces_output
            # the most tokens a rank of the group runs in this micro-batch
            peak = max(view.tbo_tokens[k])
            half = replace(half, produces_output=output, group_max_tokens=peak)
            shape, region = self._region(half, micro_batch=True)
            at.append(shape)
            bound = self.resolver and self._bound(half)
            run = [0.0]
            for name, seconds, stated, _, comm, how, _ in self._priced(
                region, half, refused, bound
            ):
                if comm is None:
                    reason = (
                        f"a two-batch-overlap step whose micro-batch graph at {shape} "
                        "runs an MoE exchange beside a second stream, as the "
                        "dual-stream MoE does, which ATOM turns off under overlap"
                    )
                    raise CostRefused(view, [Refusal(SOURCE, reason)])
                notes += stated
                origins += how[0]
                why += how[1]
                listed = listed and name.endswith(".price_list")
                if comm != (len(run) % 2 == 0):
                    run.append(0.0)
                run[-1] += seconds
            runs.append(run)
        # A refused leaf is missing from its run: `price` names it instead.
        if len(runs[0]) != len(runs[1]) and not refused:
            reason = (
                "a two-batch-overlap step whose micro-batches' graphs change "
                f"streams {len(runs[0]) - 1} and {len(runs[1]) - 1} times"
            )
            raise CostRefused(view, [Refusal(SOURCE, reason)])
        name = TBO + ".price_list" if listed else TBO
        price = (name, overlap([zip(*runs)]), notes, Work(), False, (origins, why), {})
        return " and ".join(at), [price]

    def _priced(
        self, region: dict, view: BatchView, refused: list, bound, pooled=False
    ) -> list:
        """`region`'s prices in run order, each ``(term, seconds, stated, work,
        comm, (origins, declined))``, `work` what it adds to an enclosing
        resource_bound, `comm` that it rides the comm stream under two-batch
        overlap, None when it joins an MoE exchange to work on another stream,
        `origins` the price list's provenances and `declined` its reasons for a
        roofline; what refuses goes to `refused`. With a `bound`, a leaf
        resolves through the price list first, and with `pooled`, inside a
        resource_bound, carries its work beside the list's price."""
        if "op" in region:
            got = self._leaf(region, view, refused, bound, pooled)
            return [_ridden(region, got)] if got else []
        if "repeat" in region:
            body = self._priced(region["repeat"], view, refused, bound, pooled)
            return body * region["count"]
        if "seq" in region:
            priced = []
            for item in region["seq"]:
                if "op" not in item:
                    priced += self._priced(item, view, refused, bound, pooled)
                elif got := self._leaf(item, view, refused, bound, pooled):
                    priced.append(_ridden(item, got))
            return priced
        pooled = bool(bound) and (pooled or region["join"] == "resource_bound")
        branches = [
            self._priced(b, view, refused, bound, pooled) for b in region["par"]
        ]
        summed = sum((p[3] for b in branches for p in b), Work())
        comm = None if any(p[4] is not False for b in branches for p in b) else False
        each = [fold_seconds(p[1] for p in b) for b in branches]
        if region["join"] == "resource_bound":
            name = "par.resource_bound"
            unit = self._unit(name, summed, refused)
            if unit is None:
                return []
            if not bound or unit[0] >= max(each):
                return [(name, *unit[:2], summed, comm, ((), ()), unit[2])]
        # The longest branch's prices stay under their own terms; the first
        # carries every branch's work for an enclosing resource_bound.
        longest = branches[each.index(max(each))]
        return [
            (*p[:3], summed if i == 0 else Work(), comm, *p[5:])
            for i, p in enumerate(longest)
        ]

    def _leaf(self, leaf: dict, view: BatchView, refused: list, bound, pooled=False):
        """`_price_leaf`, once a step for each value of every path its op
        kind's work has read of a leaf, and with a price list of the leaf's
        operator, arguments, outputs and stated fields; and once for good at a
        bound for a leaf object whose price read nothing else of the step and
        nothing refused. What its price read of the step joins `_read`: the
        places in `bound` laid flat (`_flat`) of the items it read and the
        names of the attributes of `view` its work read, or `VIEW` where it
        refused or read its pass's view. A leaf `_pass_views` holds reads its
        pass's view. A `pooled` leaf is kept apart, since it carries its work."""
        kept = self._kept.get(id(leaf))
        if kept is None:
            if len(self._kept) >= KEPT:
                self._kept.clear()
            kept = self._kept[id(leaf)] = [leaf, None, None, None]
        elif kept[3] and kept[3][0] is None:  # it launches no kernel at any step
            return None
        at = (bound, pooled)
        if kept[2] and kept[2][0] == at:
            self._read |= kept[2][2]
            return kept[2][1]
        own = view
        if self._passes:
            view = self._passes.get(id(leaf), view)
        # A step's refusals are named once, so a repeat need not add its own.
        key = (id(view), at, self._text(kept, bound))
        step = self._step.get(key)
        if step is None:
            n, watched, reads = len(refused), _Watched(view), set()
            got = self._price_leaf(kept, watched, refused, bound, reads, pooled)
            free = not (watched.read or refused[n:])
            if refused[n:] or watched.read and view is not own:
                reads = {VIEW}
            reads = self._shared(frozenset(reads | watched.read))
            step = self._step[key] = (view, got, reads, free)
            # Kept too by every path the work has read by now, its own among
            # them, which a path read anew changes.
            self._step[id(view), at, self._text(kept, bound)] = step
        _, got, reads, free = step
        if free:
            kept[2] = (at, got, reads)
        self._read |= reads
        return got

    def _text(self, kept: list, bound) -> str:
        """The text of what a leaf's price reads of it: every path its kind's
        work has read of a leaf, and with a price list all the list reads."""
        leaf, paths = kept[0], self._reads.setdefault(kept[0]["op"], set())
        at = (len(paths), bool(bound))
        if kept[1] is None or kept[1][0] != at:
            read = sorted(paths | LISTED_READS if bound else paths, key=repr)
            what = [leaf["op"], [[p, _at(leaf, p)] for p in read]]
            kept[1] = (at, json.dumps(what, sort_keys=True, default=repr))
        return kept[1][1]

    def _work(self, kept: list, view):
        """The work of the leaf `kept` holds at `view`, a `_Watched`, or its
        refusals; kept for good once it read nothing of the step, and else
        noted as a leaf that launches a kernel, as one whose work reads the
        step does at every step."""
        if kept[3]:
            return kept[3][0]
        leaf = kept[0]
        try:
            seen = _Seen(leaf, self._reads.setdefault(leaf["op"], set()))
            w = work(seen, view, self.tp, self.kv, self.ep)
        except CostRefused as exc:
            w = tuple(exc.declined)
        except ValueError as exc:
            w = (Refusal(SOURCE, f"{leaf['op']}: {exc}"),)
        if not view.read:
            kept[3] = (w,)
        elif w is not None:
            kept[3] = False
        return w

    def _price_leaf(
        self, kept: list, view, refused: list, bound, reads: set, pooled=False
    ):
        """The leaf `kept` holds as ``(term, seconds, stated, work, (origins,
        declined))``, as `_priced` lists it; None when it launches no kernel or
        refuses. With a `bound`, a leaf known to launch a kernel leaves its
        work to the roofline, which reads it only where the price list
        declines, and `reads` gains what the price list read of the bound
        (`_Read`); a `pooled` one works it out once and carries it, refusing
        where it cannot."""
        leaf = kept[0]
        if pooled or not bound or kept[3] is None:
            w = self._work(kept, view)
            if w is None:
                return None
            if isinstance(w, tuple) and (pooled or not bound):
                refused.extend(w)
                return None
            if bound and not pooled:  # whether it launches a kernel, which `kept`
                view.read.clear()  # holds now, is all the list's answer reads of it
        if bound:
            each = _reading(bound, _ran_at_each(leaf, bound), reads)
            later = (lambda: w) if pooled else functools.partial(self._work, kept, view)
            try:
                got = self.resolver.resolve(
                    (leaf, each if each[1:] else each[0], later)
                )
            except CostRefused as exc:
                refused.extend(exc.declined)
                return None
            term, listed = got.term, got.source == LISTED
            notes = term.provenance.note.split("; ") if term.provenance.note else []
            name = f"{term.name}.price_list" if listed else term.name
            how = ((term.provenance,) if listed else (), got.declined)
            carried = w if pooled else Work()
            return name, term.seconds, notes, carried, how, term.by_bound
        unit = self._unit(leaf["op"], w, refused)
        return unit and (f"op.{leaf['op']}", *unit[:2], w, ((), ()), unit[2])

    def _unit(self, what: str, w: Work, refused: list):
        """One pricing unit's seconds, the stated spec fields its largest
        bound read and its seconds by that bound (`work.binding`); None, with
        its refusal, when it cannot be priced."""
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
            by = binding(w, self.machine)
        except ValueError as exc:
            refused.append(Refusal(SOURCE, f"{what}: {exc}"))
            return None
        won = "comm" if "comm.latency" in by else next(iter(by))
        notes = [s for p in reads[won] if (s := self._stated(p))]
        return fold_seconds(by.values()), notes, by

    def _stated(self, path: str) -> str:
        """`path` and its origin when it is neither a datasheet's nor probed."""
        if path not in self._stated_as:
            origin = self.machine.origin(path)["origin"]
            spec = f"{path} {origin} in the machine spec {self.name}"
            self._stated_as[path] = "" if origin in ("datasheet", "probed") else spec
        return self._stated_as[path]
