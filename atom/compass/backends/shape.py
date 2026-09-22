# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A stand-in cost model: declared coefficients multiplied into a step's shapes.

**What this produces is not an estimate of anything.** The coefficients were
chosen, not measured and not regressed, and multiplying one by a real shape
does not make the product a prediction of a real duration. Nothing from this
backend may be quoted as accuracy. That sentence is not left in this file for
a reader who happens to open it: the provenance on every term carries it and
`describe()` repeats it, so whoever reads the record an artifact writes is
told without reading any source.

What it is for is the loop. A fixed pair of constants prices every step the
same, so nothing the scheduler does can depend on the price and no run can
show whether it did. That was measured rather than argued: an offline workload
presents one shape per kind, so an answer per kind looked exact -- and the same
answer cost +47.9% time-to-first-token the moment shapes varied under real
serving, three prefill steps of 16, 256 and 16128 tokens costing 38.6, 151.0
and 177.6 ms and all three answered as about 47 ms. A form that reads the
shapes makes a longer chunk cost longer, a longer step is when the next
request arrives, and the next batch is a different batch. That loop is the
thing this exists to make testable; accuracy is somebody else's milestone.

The form, declared rather than derived:

    prefill:  a  + b .tokens   + c .Sum N_Q^2 + d .Sum N_Q.(N_KV - N_Q)
    decode:   a' + b'.requests + c'.Sum N_KV  + e'.(rung.max N_KV - Sum N_KV)

plus one term per collective the deployment's widths make possible. A step
carrying both kinds of work is priced as both and pays both intercepts, because
a step that did both launched both.

A collective runs once per layer this worker holds, so its count is the depth
of the worker's span and not the number of layers in it that hold a cache of
every past token. The two are the same number only on a stack whose layers are
all paged, and they diverge with the architecture rather than by a factor
somebody could fold into the coefficient: a layer keeping a bounded recurrent
state costs a KV block nothing and still takes part in the all-reduce, so a
stack that is one paged layer in four charges four times what the paged count
would. That depth is handed in -- the KV geometry cannot supply it, since what
it counts is the paged subset -- and a deployment with a collective to price
and no depth stated is refused rather than charged on the count that happens to
be at hand.

Two things the form does not know, stated because a reader of a number will
otherwise assume it does. It has one quadratic coefficient for the whole batch
and no notion of a layer kind, so a stack whose layers do not all grow with
history -- a recurrent or windowed layer among full-attention ones -- is priced
as if they all did; whoever sets the coefficient owns that, and a form that
needs to tell them apart needs a term per kind rather than a different number.
And the collective list it charges from is necessary rather than sufficient: an
absence is an absence, a presence is a candidate the widths admit and two
conditions outside them can still rule out. Every collective term says so in
its own provenance, so a charge that may not correspond to a collective cannot
be read back as one that did.

**Every sum runs over requests, one at a time.** `BatchView` holds one row per
request and every shape sum is a function of those rows: no constructor takes
a sum in place of the rows, and no supplied sum can disagree with the rows it
was computed from. That is the point rather than fastidiousness: collapsing a
batch to `tokens x history` and multiplying makes the quadratic term and the
cross term fixed multiples of each other over every batch a scheduler can
build. A fit cannot then separate them, and a feature computed wrongly is
indistinguishable from one that is merely unconstrained. Summing per request
was the repair.

State the property precisely, because a stronger claim was made here and it
was measured false. The type does not make the collapsed form unreachable.
A one-row `BatchView` *is* `tokens x history` -- `RequestShape(512, 2048)`
alone sums to 1048576 where the two 256-token rows it collapses sum to
524288 -- and nothing in the type ties the row count to the number of
requests a scheduler scheduled. Shadowing a reader is refused outright (see
`_refuse_shadowing`); a subclass that adds a field and redefines nothing is
accepted, and its sums still come from its rows. That closes the two routes
that fabricate a reading from a field a subclass added. The row count it does
not close and cannot: what closes that is the projection -- one row per
scheduled request, built by whoever holds the scheduled batch -- and that
builder does not live in this package yet.

`capture_rung` is a batch-level scalar, correctly so, because a rung is a
property of the replayed graph rather than of any row. Two disagreements with
the rows are refused: a rung with no decode rows names a graph that replayed
nothing, and a rung narrower than the decode row count makes the padding
negative. Both are contradictions the rows settle by themselves. A rung far
wider than the rows is not -- rows cannot contradict a width, only a ladder of
captured widths can, and this backend has no ladder -- so such a rung is
priced and labelled rather than refused: `capture_rung=10**9` on one decode
row prices `decode.graph_padding` at 99.9999999 s, and that term's provenance
says the width was supplied and nothing here bounds it. A test prices that
batch, so a successor who bounds the rung fails a test rather than leaving
this paragraph stale.

The decode padding feature is the *rung's* rectangle, `rung.max - Sum`, and
not the batch's `len.max - Sum`. A replayed graph runs its rung's worth of
rows whatever the batch brought, so the batch's rectangle understates the
padding by exactly the ratio between the two -- which reads as a small
correction at a full rung and a large one at an empty one, and survives
evidence-widening because it is wrong in proportion to something nobody is
plotting. A rung on a step with no decode work is refused outright: that is
the same error in its other direction, a graph charged for a prefill that
replayed nothing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields

from atom.compass.backends.base import CostBackend, Tier
from atom.compass.backends.cost import CostTerm, StepCost
from atom.compass.backends.geometry import KvGeometry, Parallelism
from atom.compass.backends.provenance import Provenance, Species

# Carried into every term's provenance and into `describe()`, so the record an
# artifact writes says it and not only the source that produced the record.
DECLARED = "declared, not measured -- a plumbing figure, not an accuracy claim"

# Added to a collective's own provenance. The widths rule a collective out
# conclusively and rule it in only conditionally, so a charge for one states
# which of the two it is rather than leaving a reader to assume the stronger.
CANDIDATE = "a collective these widths admit, not one observed to run"

# Added to the decode padding term whenever a rung was supplied. The rows can
# show a rung is too narrow and can never show it is too wide, so that count
# rests on the caller in a way the coefficient's own note does not cover.
UNCHECKED_RUNG = "padding from a supplied rung width; nothing here bounds it above"

# Added to a collective's own provenance, because the count it multiplies is
# the one a reader is most likely to assume wrong. Two layer counts describe
# one worker and a record that states seconds states neither, so the term says
# which of them it was charged on where the charge is read.
PER_STACK_LAYER = "one charge per layer of the worker's stack, not per paged layer"


def _refuse_shadowing(owner: type, cls: type) -> None:
    """Refuse a subclass of `owner` that redefines anything `owner` reports.

    The shape features are functions of the rows, which is the property the
    whole form rests on -- but a subclass is accepted by the `isinstance`
    check the backend makes, and a subclass that redefines a reader can hand
    back a number that came from a field it added rather than from the rows.
    `cached_tokens` fabricates the cross term directly; `prefill` synthesises
    a single row holding a whole batch's tokens against a whole batch's
    history, which is the collapsed form the summing exists to keep out.
    Refusing that is the same call `StepCost` in this package already makes,
    for the same reason.

    The protected set is read off `owner` rather than listed, so a reader
    added later is covered the day it is added. `__post_init__` is added to
    it by name because it is the only non-public member whose replacement
    also changes what the object reports: it is where a row's integers and a
    rung's width are checked, and a subclass that drops it admits shapes the
    sums are not defined over.
    """
    protected = {
        name
        for name in (*vars(owner), *owner.__annotations__)
        if not name.startswith("_")
    } | {"__post_init__"}
    shadowed = sorted(protected & set(cls.__dict__))
    if shadowed:
        raise TypeError(
            f"{cls.__name__} redefines {', '.join(shadowed)} of "
            f"{owner.__name__}; the shape sums are functions of the rows, and "
            "a reader that answers from anything else is what summing per "
            "request exists to prevent"
        )


@dataclass(frozen=True)
class RequestShape:
    """One request's share of one step, as the scheduler sized it.

    `query_tokens` is what this step computes for the request: a prefill chunk,
    or the one-or-more tokens a decode step verifies. `context_tokens` is the
    KV that attention reads, which for a prefill chunk is what was cached plus
    the chunk itself. `decode` is given rather than inferred, because a
    one-token prefill and a one-token decode are the same pair of integers and
    different work.
    """

    query_tokens: int
    context_tokens: int
    decode: bool

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _refuse_shadowing(RequestShape, cls)

    def __post_init__(self) -> None:
        if self.query_tokens < 1:
            raise ValueError(
                f"a scheduled request computes at least one token, got "
                f"{self.query_tokens}"
            )
        if self.context_tokens < self.query_tokens:
            raise ValueError(
                f"{self.context_tokens} context tokens is fewer than the "
                f"{self.query_tokens} being computed; attention reads at least "
                "what this step writes"
            )

    @property
    def cached_tokens(self) -> int:
        """The history this step did not compute: N_KV minus N_Q."""
        return self.context_tokens - self.query_tokens


def _per_request(
    rows: Sequence[RequestShape], quantity: Callable[[RequestShape], int]
) -> int:
    """Add one quantity over rows, evaluating it on a row before adding.

    Every shape feature below comes through here. The alternative is to
    collapse the batch to a token count and a history length first and
    multiply those, which produces features that are fixed multiples of one
    another and cannot be told apart afterwards.
    """
    total = 0
    for row in rows:
        total += quantity(row)
    return total


def sum_tokens(rows: Sequence[RequestShape]) -> int:
    """Query tokens this step computes."""
    return _per_request(rows, lambda r: r.query_tokens)


def sum_query_square(rows: Sequence[RequestShape]) -> int:
    """Sum N_Q^2 -- the same quantity the scheduler publishes as `sqsq`."""
    return _per_request(rows, lambda r: r.query_tokens * r.query_tokens)


def sum_query_context(rows: Sequence[RequestShape]) -> int:
    """Sum N_Q.N_KV -- the scheduler's `sqsk`."""
    return _per_request(rows, lambda r: r.query_tokens * r.context_tokens)


def sum_query_cached(rows: Sequence[RequestShape]) -> int:
    """Sum N_Q.(N_KV - N_Q), which is `sqsk` minus `sqsq` and not a new reading."""
    return _per_request(rows, lambda r: r.query_tokens * r.cached_tokens)


def sum_context(rows: Sequence[RequestShape]) -> int:
    """Sum N_KV -- the scheduler's `sk`."""
    return _per_request(rows, lambda r: r.context_tokens)


@dataclass(frozen=True)
class BatchView:
    """The projection a cost backend is handed: one row per scheduled request.

    This is the type the backend seam deliberately leaves to whoever prices a
    step, and it lives in this package because the no-engine-imports property
    belongs to a package rather than to a type: the scan that enforces it
    reads the files here and nothing else, so a projection written beside the
    engine would be asserted by nobody. Nothing in this module names an engine
    class. Whoever holds a scheduled batch reads the integers off it and
    builds these rows.

    Every shape sum is a function of the rows: no constructor argument takes a
    sum in place of them, so a caller cannot supply a collapsed pair of
    scalars instead, and cannot supply a sum that disagrees with the rows it
    was supposedly computed from. `capture_rung` is the one batch-level
    scalar, and it is one because a rung is a property of the replayed graph
    and not of any row.

    What this does not do is make the collapsed form unreachable: one row is
    a legal batch, and a one-row batch is `tokens x history` by construction.
    The row count is the projection's guarantee, not the type's.
    """

    requests: tuple[RequestShape, ...]
    capture_rung: int | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _refuse_shadowing(BatchView, cls)

    def __post_init__(self) -> None:
        rows = tuple(self.requests)
        object.__setattr__(self, "requests", rows)
        if not rows:
            raise ValueError(
                "a step with no requests has no shape to price; the engine runs "
                "no forward for an empty batch, so one arriving here is a "
                "projection built from the wrong thing"
            )
        for row in rows:
            if not isinstance(row, RequestShape):
                raise TypeError(f"not a request shape: {row!r}")
        if self.capture_rung is None:
            return
        decode = self.decode
        if not decode:
            raise ValueError(
                f"capture rung {self.capture_rung} on a step with no decode "
                "requests: a rung names a replayed graph, and a step that "
                "replayed nothing has none"
            )
        if self.capture_rung < len(decode):
            raise ValueError(
                f"capture rung {self.capture_rung} is smaller than the "
                f"{len(decode)} decode requests it pads, which would make the "
                "padding negative"
            )

    @property
    def prefill(self) -> tuple[RequestShape, ...]:
        return tuple(row for row in self.requests if not row.decode)

    @property
    def decode(self) -> tuple[RequestShape, ...]:
        return tuple(row for row in self.requests if row.decode)

    @property
    def graph_padding(self) -> int:
        """`rung.max N_KV - Sum N_KV` over the decode rows; 0 with no rung."""
        rows = self.decode
        if self.capture_rung is None or not rows:
            return 0
        widest = max(row.context_tokens for row in rows)
        return self.capture_rung * widest - sum_context(rows)


@dataclass(frozen=True)
class Coefficients:
    """The declared multipliers of the form. None of them was measured.

    The defaults are round numbers in seconds-per-unit, picked so a test's
    arithmetic can be checked by hand and so a long chunk costs visibly more
    than a short one. They are not a fit and they carry no physics; a backend
    that means its numbers is a later milestone with measurements behind it.
    """

    prefill_step: float = 2.0e-5
    prefill_token: float = 5.0e-7
    prefill_query_square: float = 2.0e-11
    prefill_query_cached: float = 4.0e-12
    decode_step: float = 2.0e-5
    decode_request: float = 2.0e-6
    decode_context: float = 4.0e-9
    decode_padding: float = 1.0e-9
    collective_token_layer: float = 1.0e-9

    def __post_init__(self) -> None:
        for spec in fields(self):
            value = getattr(self, spec.name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{spec.name}={value} is not a usable coefficient")

    @property
    def shape_blind(self) -> bool:
        """True when only the two intercepts are non-zero: constant pricing."""
        shaped = [f.name for f in fields(self) if f.name not in _INTERCEPTS]
        return not any(getattr(self, name) for name in shaped)

    @classmethod
    def constant(cls, prefill_seconds: float, decode_seconds: float) -> Coefficients:
        """Bring-up pricing: the same form with every shape coefficient zero.

        Constant pricing is a coefficient set rather than a mode on the
        backend. `estimate` gains no branch, the breakdown still names every
        term, and the zeros are in the record instead of implied by a flag
        somewhere else. It is kept for first bring-up and is not the default,
        because a price that ignores the shape cannot show whether anything
        downstream reacted to the shape.
        """
        blind = {f.name: 0.0 for f in fields(cls)}
        blind["prefill_step"] = prefill_seconds
        blind["decode_step"] = decode_seconds
        return cls(**blind)


_INTERCEPTS = ("prefill_step", "decode_step")


class ShapeStubBackend(CostBackend):
    """Prices a step by multiplying declared coefficients into its shapes.

    A stub, not an analytic model. `Tier.COARSE` says this answers at the
    granularity of a whole step, which is true; what the tier cannot say, and
    what the provenance on every term says instead, is that the numbers behind
    the answer were declared rather than obtained.

    `parallelism` decides which collectives are charged, by asking it rather
    than by reasoning from the expert width -- an all-to-all exists on the
    data-parallel width alone, and a backend that decided from the experts
    would charge a single-rank deployment for a collective it never builds.
    What it returns is a candidate set: nothing it omits can run, and what it
    names can still be ruled out by conditions the widths do not express, so
    each such term carries that qualification in its own provenance rather
    than being charged as a certainty.

    Pricing one needs a layer count, and there are two of them. `stack_layers`
    is the one the charge is made on: every layer this worker runs, whatever
    each layer keeps. `geometry` is the KV a block is sized from, whose own
    `layers` counts only the layers holding a cache of every past token -- the
    same number on a uniform stack and a quarter of it on a stack that is one
    paged layer in four. Both are held, neither is derived from the other, and
    a deployment with a candidate collective and no stack depth is refused: a
    count taken from the geometry there would be wrong in kind on every hybrid
    and right by coincidence on everything else.
    """

    def __init__(
        self,
        coefficients: Coefficients | None = None,
        parallelism: Parallelism | None = None,
        geometry: KvGeometry | None = None,
        stack_layers: int | None = None,
    ) -> None:
        self.coefficients = Coefficients() if coefficients is None else coefficients
        self.parallelism = Parallelism() if parallelism is None else parallelism
        self.geometry = geometry
        self.stack_layers = None if stack_layers is None else int(stack_layers)
        named = self.parallelism.collectives()
        if named and self.stack_layers is None:
            raise ValueError(
                f"{', '.join(named)} to price and no stack depth to price it from; "
                "a collective runs once per layer this worker runs, and a KV "
                "geometry counts only the layers that hold a cache of every past "
                "token, so state how many layers deep the span is"
            )
        if self.stack_layers is not None and self.stack_layers < 1:
            raise ValueError(
                "a worker runs at least one layer, got "
                f"stack_layers={self.stack_layers}"
            )
        if (
            self.stack_layers is not None
            and geometry is not None
            and self.stack_layers < geometry.layers
        ):
            raise ValueError(
                f"a {self.stack_layers}-layer span holding {geometry.layers} paged "
                "layers is not a span: the layers that hold a cache are a subset "
                "of the layers there are, so the two counts have been crossed"
            )

    @property
    def tier(self) -> Tier:
        return Tier.COARSE

    def _term(
        self, name: str, count: int, coefficient: float, qualifier: str = ""
    ) -> CostTerm:
        """One term: a count read off the batch times a number nobody measured.

        `qualifier` is for a term whose count is less certain than the reading
        it came from, and it goes in the record beside the count rather than
        in a note somewhere else.
        """
        detail = f"{coefficient:g} s x {count}; {DECLARED}"
        return CostTerm(
            name,
            count * coefficient,
            Provenance(
                Species.FITTED,
                f"{detail}; {qualifier}" if qualifier else detail,
                "shape-stub",
            ),
        )

    def estimate(self, batch_view: BatchView) -> StepCost:
        """Price one step as the parts its total is folded from.

        No seconds are added here. Each term is one product, and the total is
        whatever `StepCost` folds from them, so there is no second summation
        to disagree with the first.

        The two refusals below and in `BatchView.__post_init__` are `TypeError`
        and `ValueError` rather than the `CostRefused` the seam names, because
        a batch that is not a batch is a caller defect and not a step this
        backend declines to price; the asymmetry is worth stating because a
        resolver that wraps this backend catches `CostRefused` and falls
        through to the next rung, and will not catch these.
        """
        if not isinstance(batch_view, BatchView):
            raise TypeError(
                f"this backend prices a BatchView, not {type(batch_view).__name__}; "
                "the projection is built by whoever holds the scheduled batch"
            )
        c = self.coefficients
        counted: list[tuple[str, int, float] | tuple[str, int, float, str]] = []
        prefill = batch_view.prefill
        if prefill:
            counted += [
                ("prefill.step", 1, c.prefill_step),
                ("prefill.tokens", sum_tokens(prefill), c.prefill_token),
                (
                    "prefill.query_square",
                    sum_query_square(prefill),
                    c.prefill_query_square,
                ),
                (
                    "prefill.query_cached",
                    sum_query_cached(prefill),
                    c.prefill_query_cached,
                ),
            ]
        decode = batch_view.decode
        if decode:
            counted += [
                ("decode.step", 1, c.decode_step),
                ("decode.requests", len(decode), c.decode_request),
                ("decode.context", sum_context(decode), c.decode_context),
                (
                    "decode.graph_padding",
                    batch_view.graph_padding,
                    c.decode_padding,
                    "" if batch_view.capture_rung is None else UNCHECKED_RUNG,
                ),
            ]
        terms = [self._term(*row) for row in counted]
        collectives = self.parallelism.collectives()
        if collectives:
            # The constructor refuses a candidate collective with no depth
            # stated, so there is a count here and it is the stack's.
            moved = sum_tokens(batch_view.requests) * self.stack_layers
            for collective in collectives:
                terms.append(
                    self._term(
                        f"collective.{collective}",
                        moved,
                        c.collective_token_layer,
                        f"{CANDIDATE}; {PER_STACK_LAYER}",
                    )
                )
        return StepCost(terms)

    def describe(self) -> str:
        """One line for the run record, including what the number is not."""
        pricing = "constant bring-up" if self.coefficients.shape_blind else "shape-read"
        widths = (
            f"tp{self.parallelism.tp_size}"
            f" pp{self.parallelism.pp_size}"
            f" dp{self.parallelism.dp_size}"
        )
        named = ", ".join(self.parallelism.collectives())
        if named:
            # The depth belongs in the line because the seconds do not carry
            # it: a collective charged on the paged layers of a hybrid and one
            # charged on its stack read alike once they are in a total.
            paged = "" if self.geometry is None else f", {self.geometry.layers} paged"
            named += f" on {self.stack_layers} layers{paged}"
        else:
            named = "no collectives"
        return (
            f"step-level stand-in, {pricing} coefficients, {widths}, {named}"
            f" -- {DECLARED}"
        )
