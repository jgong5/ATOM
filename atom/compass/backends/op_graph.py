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
from its first draft pass on: its prices are terms named ``draft op.<kind>``,
each tagged draft. Sampling is the target's. A decode's drafter is read at its
rows run at the step's width: its first pass runs the rows' tokens unpadded.

Each leaf is one pricing unit, `max(t0, F / peak, M / BW(M), t_comm)` over its
work (`op_work.work`): its own tensors at the captured shape, the state it
reads at the step's rows, attention's context in `kv`, the run's KV cache
format, the routed experts over the run's `ep` ranks. A leaf that launches
no kernel costs nothing. A step prices once the leaves of a kind that agree on
every part of a leaf the kind's work has read, a leaf whose work reads nothing
of the step keeps its price, and with no price list a graph keeps its price
for each set of prices of its leaves that read the step. Regions compose by the
IR's joins: a `Seq` adds its items in order, a `Repeat` re-emits its body's
prices once per instance, a `max` `Par` emits its longest branch's prices, and a
`resource_bound` `Par` one roofline over its branches' summed work. The price
is one rank's device busy time: the runner takes a DP group's max and
composes host time.

Two-batch overlap. A step the data-parallel group split (`tbo_tokens`) runs
this rank's tokens as two micro-batches (`step_roofline.micro_batches`), each
priced from the graph at its own prefill shape and the group's rows of it
(`moe_rows`), the second with the step's output and the first with none.
Each graph's prices in run order fall into segments that alternate streams: an
MoE exchange of activations rides the comm stream, everything else the compute
stream. Both micro-batches' segments go on the two streams in ATOM's order
(`step_roofline.overlap`); the finish time is the one term ``op.tbo``.

With a `price_list`, a source named `LISTED` (`fit.ops.PriceList` in a run),
each leaf outside a `resource_bound` `Par` resolves through a `Resolver`: the
price list, asked ``(leaf, bound, work)`` with `bound` the step's or
micro-batch's (`_bound`), then its roofline. A `resource_bound` `Par` prices
its branches' summed work, so its leaves take no price of their own.

Terms. One per op kind, ``op.<kind>``, and ``par.resource_bound``, each
the fold of its prices in run order. A term is `declared`, its note naming
each input, when a spec field the winning bound of one of its prices read is
neither from a datasheet nor probed, and `analytical` otherwise; it carries
each reason the price list declined one of its leaves. A price list's prices
are ``op.<kind>.price_list``, and ``op.tbo.price_list`` when it priced every
leaf of both micro-batches, `measured` when each sat on a measured point and
`estimated` otherwise.

Refused: a step with image tokens, since no graph holds the vision encoder; a
two-batch-overlap step on MoRI's split send and receive (`split_send_recv`),
whose transfer in flight lies on no stream, whose split names no pair for
this rank, whose micro-batches' graphs exchange a different number of times
or run an exchange beside a second stream (the dual-stream MoE, which ATOM
turns off under overlap), or that runs under a drafter, whose draft pass no
graph places on the overlapped streams; a decode step whose rows do not each
verify the drafter's tokens and one; a prefill step whose view
does not say whether it produces output; a shape with no graph, a
data-parallel step's naming the rows it gathers; and, each reason once, a leaf
whose kind has no extractor, whose price reads a spec field the spec lacks, or
whose collective leaves the scale-up domain.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost, fold_seconds
from atom.compass.backends.op_work import work
from atom.compass.backends.provenance import Method, Provenance, Refusal, combine
from atom.compass.backends.resolver import CostRefused, CostSource, Resolver
from atom.compass.backends.shape import IMAGES, BatchView
from atom.compass.backends.step_roofline import _reads, micro_batches, overlap
from atom.compass.backends.work import Work, bounds

SOURCE = "op-graph"
LISTED = "price-list"
#: The leaves a captured MoE layer exchanges its DP or EP group's rows by:
#: ATOM's gathers and scatters, and MORI's dispatch and combine.
MOE_EXCHANGES = ("atom.model_ops.moe:", "mori.ops.dispatch_combine:")
#: The term a two-batch-overlap step's one price is.
TBO = "op.tbo"
NONE = "; no successor: the analytical law is the last op-level source"
#: The prefix of a drafter's terms.
DRAFT = "draft "
#: The leaves a backend keeps; past it, it forgets them all and every price
#: it kept across steps.
KEPT = 1 << 16
#: The prices a graph keeps by its step-reading leaves' prices; past it, it
#: forgets them.
COMPOSED = 1 << 12
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


def _same(priced: list) -> tuple:
    """A key that two lists of a graph's ``(leaf, in a Par, price)`` share
    only when each leaf is at a price of the same value, its work too in a
    `Par`, the only join that reads it. The share index stays: a step can
    share leaves another step prices apart, so equal values can sit on other
    leaves."""
    first: dict[tuple, tuple] = {}
    for _, par, p in priced:
        value = p if par or p is None else p[:3] + p[4:]
        first.setdefault((id(p), par), (len(first), value))
    return (
        tuple(first[id(p), par][0] for _, par, p in priced),
        repr([value for _, value in first.values()]),
    )


def _ref(value):
    return sorted(value) if isinstance(value, dict) and value.keys() in REFS else value


class _Watched:
    """A step's view that notes whether a leaf's work read anything of it."""

    def __init__(self, view: BatchView) -> None:
        self.view, self.read = view, False

    def __getattr__(self, name: str):
        self.read = True
        return getattr(self.view, name)


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
        self.name = machine.values.get("name", "")
        self._stated_as: dict[str, str] = {}
        # By leaf object: the leaf, the text of what its price reads of it, and
        # its price at a bound when its work read nothing of the step; by view,
        # bound and that text: one step's prices. By op kind: every path its
        # work has read of a leaf. By graph: the leaves whose prices read the
        # step, and the composed price by their prices.
        self._kept: dict[int, list] = {}
        self._step: dict[tuple, tuple] = {}
        self._reads: dict[str, set] = {}
        self._graphs: dict[tuple, tuple] = {}
        self._live: list | None = None
        self._pars = 0
        self.resolver = price_list and Resolver([price_list, _Roofline(self)])

    @property
    def granularity(self) -> Granularity:
        return Granularity.OP

    def price(self, batch_view: BatchView) -> StepCost:
        refused: list[Refusal] = []
        self._step, self._live, self._pars, graph = {}, None, 0, None
        if batch_view.tbo_tokens is None or batch_view.image_tokens:
            at, region = self._region(batch_view)
            bound = self.resolver and self._bound(batch_view)
            drafted_at, drafted = (
                self._region(batch_view, draft=True) if self.drafts else (None, None)
            )
            # A price list's bound reads each step's context, so its steps
            # compose afresh.
            graph = None if bound else (at, id(region), drafted_at, id(drafted))
            if graph in self._graphs:
                # The other leaves' prices read nothing of a step.
                missed: list[Refusal] = []
                got = [
                    (*p[:2], self._leaf(p[0], batch_view, missed, None))
                    for p in self._graphs[graph][1]
                ]
                if missed:
                    self._step = {}
                elif (cost := self._graphs[graph][2].get(_same(got))) is not None:
                    return cost
            self._live = [] if graph else None
            priced = self._priced(region, batch_view, refused, bound)
            if self.drafts:
                priced += [
                    (DRAFT + p[0], *p[1:])
                    for p in self._priced(drafted, batch_view, refused, bound)
                ]
            live, self._live = self._live, None
        else:
            at, priced = self._overlapped(batch_view, refused)
        if refused:
            raise CostRefused(batch_view, list(dict.fromkeys(refused)))
        seconds: dict[str, list[float]] = {}
        stated: dict[str, list[str]] = {}
        how: dict[str, tuple[list, list]] = {}
        for name, s, notes, _, _, (methods, declined) in priced:
            seconds.setdefault(name, []).append(s)
            stated.setdefault(name, []).extend(notes)
            got = how.setdefault(name, ([], []))
            got[0].extend(methods)
            got[1].extend(declined)
        terms = []
        for name, values in seconds.items():
            note = "; ".join(dict.fromkeys(stated[name]))
            method = Method.DECLARED if note else Method.ANALYTICAL
            source, (methods, why) = SOURCE, how[name]
            if name.endswith(".price_list"):
                method, source = combine(methods), LISTED
            drafted = name.startswith(DRAFT)
            detail = (
                f"{len(values)} prices of the graph at {drafted_at if drafted else at}"
            )
            if name.startswith(TBO):
                detail = f"two micro-batches overlapped, from the graphs at {at}"
            provenance = Provenance(
                method,
                detail,
                source,
                tuple(dict.fromkeys(why)),
                note=note and note + NONE,
            )
            terms.append(CostTerm(name, fold_seconds(values), provenance, drafted))
        cost = StepCost(terms)
        if graph is not None:
            composed = self._graphs.get(graph, (0, 0, {}))[2]
            if len(composed) >= COMPOSED:
                composed.clear()
            composed[_same(live)] = cost
            self._graphs[graph] = ((region, drafted), live, composed)
        return cost

    def describe(self) -> str:
        rows = "the rows a graph pads to" if self.padded else "real rows"
        first = "a price list, then " if self.resolver else ""
        return f"op graph at TP {self.tp}, {first}one roofline per leaf, at {rows}"

    def _bound(self, view: BatchView) -> tuple:
        """What a price list reads of a step: its phase, its T and B, the rows
        its MoE gathers, None at their default (`gathered_off_default`),
        and its rows' context and max context (`band`), the
        context a decode's max, a prefill's mean cached tokens."""
        rung = view.capture_rung if self.padded else None
        rows = [(r.query_tokens, r.decode) for r in view.requests]
        phase, t, b = step_shape(rows, rung, self.drafts)
        g = gathered_off_default(phase, t, view.moe_rows, self.dp, self.drafts)
        rows = [(r.query_tokens, r.context_tokens) for r in view.requests]
        mean, top = band(rows, phase == "decode")
        return (phase, t, b, g, top if phase == "decode" else mean, top)

    def _region(self, view: BatchView, draft: bool = False) -> tuple[str, dict]:
        """The graph the step reads, by its shape and output, and its region;
        with `draft`, its drafter's, a decode's at its rows, at `draft_moe_rows`,
        its later passes at `draft_bs`."""
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
                width = draft and self.padded and view.draft_bs
                if draft and shape[0] == "decode":
                    shape = ("decode", len(rows), width or shape[2])
                    width = None
                none = "" if output else " that produces no output"
                gathered = view.draft_moe_rows if draft else view.moe_rows
                g = () if gathered is None else (gathered,)
                drafted = {"draft": True} if draft else {}
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
                    region := self.graph(*shape, output, *g, **drafted), dict
                ):
                    return "{} T={} B={}".format(*shape) + at + none, region
                else:
                    reason = (
                        "no op_graph holds a {} step of {} tokens over {} sequences"
                        "{}{}{}{}; compass trace ops captures one".format(
                            *shape,
                            "".join(f" gathering {n} rows" for n in g),
                            none,
                            " for its drafter" * draft
                            + f", its later passes at {width}" * ("width" in drafted),
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
        at, runs, notes, methods, why = [], [], [], [], []
        listed = bool(self.resolver)
        for k, half in enumerate(halves):
            output = k == 1 and view.produces_output
            half = replace(half, produces_output=output)
            shape, region = self._region(half)
            at.append(shape)
            bound = self.resolver and self._bound(half)
            run = [0.0]
            for name, seconds, stated, _, comm, how in self._priced(
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
                methods += how[0]
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
        price = (name, overlap([zip(*runs)]), notes, Work(), False, (methods, why))
        return " and ".join(at), [price]

    def _priced(self, region: dict, view: BatchView, refused: list, bound) -> list:
        """`region`'s prices in run order, each ``(term, seconds, stated, work,
        comm, (methods, declined))``, `work` what it adds to an enclosing
        resource_bound, `comm` that it rides the comm stream under two-batch
        overlap, None when it joins an MoE exchange to work on another stream,
        `methods` the price list's and `declined` its reasons for a roofline;
        what refuses goes to `refused`. With a `bound`, a leaf resolves through
        the price list first."""
        if "op" in region:
            leaf = self._leaf(region, view, refused, bound)
            comm = region["op"].startswith(MOE_EXCHANGES) and any(
                isinstance(a, dict) and a.get("role") == "activation"
                for a in region["args"].values()
            )
            return [(*leaf[:4], comm, leaf[4])] if leaf else []
        if "repeat" in region:
            body = self._priced(region["repeat"], view, refused, bound)
            return body * region["count"]
        if "seq" in region:
            return [
                p for i in region["seq"] for p in self._priced(i, view, refused, bound)
            ]
        pooled = region["join"] == "resource_bound"
        self._pars += 1
        branches = [
            self._priced(b, view, refused, None if pooled else bound)
            for b in region["par"]
        ]
        self._pars -= 1
        summed = sum((p[3] for b in branches for p in b), Work())
        comm = None if any(p[4] is not False for b in branches for p in b) else False
        if region["join"] == "max":
            each = [fold_seconds(p[1] for p in b) for b in branches]
            longest = branches[each.index(max(each))]
            # The longest branch's prices stay under their own terms; the
            # first carries every branch's work for an enclosing resource_bound.
            return [
                (*p[:3], summed if i == 0 else Work(), comm, p[5])
                for i, p in enumerate(longest)
            ]
        name = f"par.{region['join']}"
        unit = self._unit(name, summed, refused)
        return [(name, *unit, summed, comm, ((), ()))] if unit else []

    def _leaf(self, leaf: dict, view: BatchView, refused: list, bound):
        """`_price_leaf`, once a step for each value of every path its op
        kind's work has read of a leaf, and with a price list of the leaf's
        operator, arguments, outputs and stated fields; and once for good for a
        leaf object whose work read nothing of the step and nothing refused."""
        kept = self._kept.get(id(leaf))
        if kept is None:
            if len(self._kept) >= KEPT:
                self._kept.clear()
                self._graphs.clear()
            kept = self._kept[id(leaf)] = [leaf, None, None]
        if kept[2] and kept[2][0] == bound:
            return kept[2][1]
        # A step's refusals are named once, so a repeat need not add its own.
        key = (id(view), bound, self._text(kept, bound))
        if key not in self._step:
            n, watched = len(refused), _Watched(view)
            got = self._price_leaf(leaf, watched, refused, bound)
            self._step[key] = (view, got, not (watched.read or refused[n:]))
            # Kept too by every path the work has read by now, its own among
            # them, which a path read anew changes.
            self._step[id(view), bound, self._text(kept, bound)] = self._step[key]
        _, got, free = self._step[key]
        if free:
            kept[2] = (bound, got)
        elif self._live is not None:
            self._live.append((leaf, self._pars > 0, got))
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

    def _price_leaf(self, leaf: dict, view, refused: list, bound):
        """One leaf's ``(term, seconds, stated, work, (methods, declined))`` as
        `_priced` lists it; None when it launches no kernel or refuses."""
        try:
            seen = _Seen(leaf, self._reads.setdefault(leaf["op"], set()))
            w = work(seen, view, self.tp, self.kv, self.ep)
        except CostRefused as exc:
            w = tuple(exc.declined)
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
            how = ((term.provenance.method,) if listed else (), got.declined)
            return name, term.seconds, notes, Work(), how
        if isinstance(w, tuple):
            refused.extend(w)
            return None
        unit = self._unit(leaf["op"], w, refused)
        return unit and (f"op.{leaf['op']}", *unit, w, ((), ()))

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
