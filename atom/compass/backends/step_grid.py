# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step grid: a step looked up in a grid of measured whole forwards.

A batch maps to the homogeneous shape that keeps its token count and its
attention work, Sum (N_Q.c + N_Q(N_Q+1)/2) over rows with c the cached
context, in exact arithmetic: B rows of N_Q = tokens / B each, and the c that
gives the same work. That shape is read in one grid:

    prefill:                (B, T, c), T = B.N_Q the step's tokens, from the
                            grid of chunks that produce output or the grid of
                            chunks that do not
    decode, replay bundle:  (B, c), from the grid of the rung the step replays
    decode, eager bundle:   (B, c), the rung ignored

A decode row computes the grid's decode width, one token, or under a drafter
of draft width K (`draft_width`) the K + 1 of a verify step, so that c is the
rows' mean cached context, and the lookup there, G(mean), keeps the attention
work but not how it is spread over the rows. On a rung whose grid holds
`alpha`, a decode batch's
lookup is G(mean) + alpha (G(longest) - G(mean)), G read at B rows of the
mean and of the longest row's cached context, so between the two, with
alpha = clip(intercept + ratio . (longest / mean - 1) + spread . sd / mean
+ mean . mean / 2^16, 0, 1) over the rows' cached contexts
(`alpha_features`). On any other rung, or for a batch whose rows read one
context, the lookup is G(mean).

A cell is the box between adjacent nodes on every axis; a coordinate on a node
reads that node alone, so an exact hit needs one point. The answer is the
multilinear interpolation over the cell's corners when every corner was
measured. A decode cell whose every unmeasured corner is past the KV pool the
grid was planned for (`pool`), more blocks than it holds at B rows of c + N_Q
tokens and the `lookahead` positions a drafter writes past them, is
extrapolated instead: each B side of the cell is read along c from
its two highest c nodes inside the pool, extended past them. Any other step
whose cell misses a corner is refused naming the missing corners, as is one
outside the nodes: no clamp or fallback. A missing point does not fail
loading; it refuses the steps that need it. A step with a B side whose
extension past the pool is not positive is refused too, naming the falling edge.

Also refused: a step mixing prefill and decode rows, a decode row computing
a width not the grid's, a count that is not an int, a prefill step that does not
say whether it produces output, and under a replay bundle a decode step that
replays no graph, being wider than the widest rung or eager, and a step with
image tokens. A DP dummy forward (`BatchView.dp_dummy`) that replays no graph
is, under a replay bundle, `eager_dp_dummy`: 0 s, not refused.

The answer is the lookup times the residual's factor for the batch's ordered
rows (`step_grid_residual.py`), which the residual may refuse, and which
answers past its hull too. It is one `estimated` term for the whole forward,
validated iff its coordinates in its grid lie inside the hull of those of the
batches its kind's residual was held out on (`Residual.validated`), and never
multiplied by the TP width, since the measurement already holds the
collectives. A step extrapolated past the KV pool or the residual's hull is
unvalidated, its source `EXTRAPOLATED` naming how far past each. Its detail
calls the lookup `measured` for a homogeneous batch on a node, `extrapolated`
past the pool and `interpolated` otherwise, and gives each member's output.

Under a drafter every point also holds the drafter's value, `draft`, the part
of its window after the target's. `StepGrid(grid, drafter=True)` reads that
value at the same coordinates by the same rules, with the drafter's own alpha,
`draft_alpha`; the backend prices it, times its own residual's factor, as the
step's draft term, `draft`, beside the target's `forward`.

The grid is a mapping, as a bundle holds it:

    {"graph_mode": "replay" | "eager", "scope": <timing scope>,
     "unit": "s" | "ms" | "us", "draft_width": K,             # K optional
     "prefill": {"nodes": {"B": [...], "T": [...], "c": [...]},
                 "output": [POINT, ...], "no_output": [POINT, ...]},
     "decode": {"nodes": {"B": NODES, "c": NODES}, "points": [POINT, ...],
                "pool": {"block_size": n, "blocks": n, "lookahead": n},
                "alpha": {"<rung>": {"intercept": a, "ratio": b,
                                     "spread": d, "mean": m}, ...},
                "draft_alpha": {...}}}          # pool, alphas optional

A grid without `draft_width` has no drafter; one without `pool` extrapolates
no decode corner, and a pool without `lookahead` adds no position past a row;
one without `alpha` reads every decode batch at G(mean). An alpha without
`mean`, fitted before that term, reads it as 0.

    replay: NODES = {"<rung>": [...], ...}, a POINT at [rung, B, c]
    eager:  NODES = [...],                  a POINT at [B, c]
    POINT = {"at": [...], "time": <in unit>, "statistic": <name>, "repeats": n,
             "shared": n, "draft": <in unit>}  # shared optional, draft a drafter's

A point measured with its rows sharing leading KV blocks, past the pool
unshared, states how many as `shared`.
"""

from __future__ import annotations

import bisect
import itertools
import math
import statistics
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING

from atom.compass.backends import provenance
from atom.compass.backends.base import CostBackend, Granularity
from atom.compass.backends.cost import CostTerm, StepCost, eager_dp_dummy
from atom.compass.backends.provenance import Method, Provenance, Refusal
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import (
    IMAGES,
    BatchView,
    RequestShape,
    sum_query_cached,
    sum_query_square,
    sum_tokens,
)

if TYPE_CHECKING:
    from atom.compass.backends.step_grid_residual import Residual

SOURCE = "step-grid"
#: A decode rung's alpha terms, as the grid holds them.
ALPHA = ("intercept", "ratio", "spread", "mean")
#: The tokens alpha's `mean` term reads the mean cached context in.
MEAN_UNIT = 2**16
EXTRAPOLATED = f"{SOURCE}{provenance.EXTRAPOLATED}"
UNITS = {"s": Fraction(1), "ms": Fraction(1, 10**3), "us": Fraction(1, 10**6)}
#: A point's value and the decode grid's alpha, for the target and the drafter.
VALUES = {False: ("time", "alpha"), True: ("draft", "draft_alpha")}


def equivalent_shape(
    rows: Sequence[RequestShape],
) -> tuple[Fraction, Fraction, Fraction]:
    """(B, N_Q, c) of the homogeneous batch with these rows' tokens and work."""
    rows_n, tokens = len(rows), sum_tokens(rows)
    # Sum N_Q(N_Q+1)/2 is (Sum N_Q^2 + Sum N_Q) / 2, and N_Q^2 + N_Q is even.
    work = sum_query_cached(rows) + (sum_query_square(rows) + tokens) // 2
    query = Fraction(tokens, rows_n)
    return (
        Fraction(rows_n),
        query,
        (Fraction(work, rows_n) - query * (query + 1) / 2) / query,
    )


def coordinates(rows: Sequence[RequestShape], prefill: bool) -> tuple[Fraction, ...]:
    """The equivalent shape as a grid reads it: (B, T, c) in prefill, (B, c) in
    decode."""
    rows_n, query, cached = equivalent_shape(rows)
    return (rows_n, rows_n * query, cached) if prefill else (rows_n, cached)


def length_statistics(rows: Sequence[RequestShape]) -> tuple[float, float]:
    """The longest cached context over the mean, less 1, and the cached
    contexts' standard deviation over their mean: 0 and 0 when every row reads
    one context."""
    cached = [row.context_tokens - row.query_tokens for row in rows]
    mean = statistics.fmean(cached)
    if not mean:
        return 0.0, 0.0
    return max(cached) / mean - 1.0, statistics.pstdev(cached) / mean


def alpha_features(rows: Sequence[RequestShape]) -> tuple[float, ...] | None:
    """What `ALPHA`'s terms multiply, in its order: 1, `length_statistics`, and
    the mean cached context over `MEAN_UNIT`; None when every row reads one
    context."""
    ratio, spread = length_statistics(rows)
    if not ratio:
        return None
    mean = statistics.fmean(row.context_tokens - row.query_tokens for row in rows)
    return 1.0, ratio, spread, mean / MEAN_UNIT


def alpha_of(terms: Sequence[float], rows: Sequence[RequestShape]) -> float:
    """The alpha a rung's `terms` give a decode batch of `rows`, 0 when its rows
    read one context."""
    x = alpha_features(rows)
    if x is None:
        return 0.0
    return min(1.0, max(0.0, sum(t * v for t, v in zip(terms, x))))


def _refused(view: BatchView, reason: str) -> CostRefused:
    return CostRefused(view, [Refusal(SOURCE, reason)])


def _named(axes: Sequence[str], at: Sequence[Fraction]) -> str:
    return ", ".join(f"{axis}={value}" for axis, value in zip(axes, at))


class _Cells:
    """One grid: ascending nodes per axis, and the points measured on them,
    each holding its `value`."""

    def __init__(
        self, name: str, axes: tuple[str, ...], nodes, points, value="time"
    ) -> None:
        self.name, self.axes = name, axes
        self.nodes = [[Fraction(n) for n in axis_nodes] for axis_nodes in nodes]
        for axis, axis_nodes in zip(axes, self.nodes):
            if not axis_nodes or any(a >= b for a, b in itertools.pairwise(axis_nodes)):
                raise ValueError(
                    f"the {name} grid's {axis} nodes "
                    f"[{', '.join(map(str, axis_nodes))}] are not strictly ascending"
                )
        self.points: dict[tuple[Fraction, ...], Fraction] = {}
        for point in points:
            at = tuple(Fraction(v) for v in point["at"])
            time = float(point[value])
            repeats = point["repeats"]
            if len(at) != len(axes) or not all(v in n for v, n in zip(at, self.nodes)):
                raise ValueError(
                    f"a {name} point at {point['at']} is not on the grid's "
                    f"{'/'.join(axes)} nodes"
                )
            if at in self.points:
                raise ValueError(f"the {name} grid measures {point['at']} twice")
            if not (math.isfinite(time) and time > 0.0):
                raise ValueError(
                    f"the {name} point at {point['at']} takes {time}, which is "
                    "not a positive duration"
                )
            if not point["statistic"] or type(repeats) is not int or repeats < 1:
                raise ValueError(
                    f"the {name} point at {point['at']} records no statistic "
                    "and repeat count"
                )
            self.points[at] = Fraction(time)

    def _sides(self, view: BatchView, at: tuple[Fraction, ...]) -> list[tuple]:
        """Per axis, the cell's nodes holding `at`, each with its weight, one
        when `at` is on a node; or a refusal outside the nodes."""
        sides = []
        for axis, nodes, x in zip(self.axes, self.nodes, at):
            if x in nodes:
                sides.append(((x, Fraction(1)),))
                continue
            i = bisect.bisect(nodes, x)
            if i in (0, len(nodes)):
                raise _refused(
                    view,
                    f"{axis} is {'below' if i == 0 else 'above'} the {self.name} "
                    f"grid's nodes {nodes[0]}..{nodes[-1]}",
                )
            low, high = nodes[i - 1], nodes[i]
            weight = (x - low) / (high - low)
            sides.append(((low, 1 - weight), (high, weight)))
        return sides

    def corners(
        self, view: BatchView, at: tuple[Fraction, ...]
    ) -> list[tuple[tuple[Fraction, ...], Fraction]]:
        """The corners of the cell holding `at`, each with its weight, one when
        `at` is on a node; or a refusal outside the nodes."""
        return [
            (tuple(n for n, _ in c), math.prod(w for _, w in c))
            for c in itertools.product(*self._sides(view, at))
        ]

    def _cell(self, sides) -> str:
        return ", ".join(
            (
                f"{axis}={side[0][0]}"
                if len(side) == 1
                else f"{axis} {side[0][0]}..{side[1][0]}"
            )
            for axis, side in zip(self.axes, sides)
        )

    def lookup(
        self, view: BatchView, at: tuple[Fraction, ...]
    ) -> tuple[Fraction, bool, str]:
        """The value at `at`, whether it is a node, and the cell; or a refusal."""
        corners = self.corners(view, at)
        missing = [corner for corner, _ in corners if corner not in self.points]
        if missing:
            raise _refused(
                view,
                f"the {self.name} cell has no measurement at "
                f"{'; '.join(_named(self.axes, m) for m in missing)}",
            )
        value = sum(self.points[corner] * weight for corner, weight in corners)
        return value, len(corners) == 1, self._cell(self._sides(view, at))

    def past_pool(
        self,
        view: BatchView,
        at: tuple[Fraction, ...],
        size: int,
        blocks: int,
        extra: int = 1,
    ) -> tuple[Fraction, str, str] | None:
        """A decode step's value at (B, c) `at` when every unmeasured corner of
        its cell is past a pool of `blocks` KV blocks of `size` tokens, a row
        holding `extra` positions past its cached context, with the cell and
        how far past the pool's measured edge; else None. Each B side whose
        corners were measured is interpolated, and any other is read on its two
        highest c nodes at which B rows fit the pool, extended to c; a refusal
        naming the falling edge when a side so extended is not positive."""
        sides = self._sides(view, at)
        value, past, falls = Fraction(0), [], []
        for b, weight in sides[0]:
            limit, column = size * (blocks // b) - extra, sides[1]
            if any((b, c) not in self.points for c, _ in column):
                if any(c <= limit and (b, c) not in self.points for c, _ in column):
                    return None
                edge = [c for c in self.nodes[1] if c <= limit][-2:]
                if len(edge) < 2 or any((b, c) not in self.points for c in edge):
                    return None
                low, high = edge
                w = (at[1] - low) / (high - low)
                column = [(low, 1 - w), (high, w)]
                if sum(self.points[b, c] * x for c, x in column) <= 0:
                    falls.append(f"B={b}'s in-pool edge falls from c={low} to {high}")
                past.append(
                    f"c {float(at[1] - high):g} past {high}, the pool's edge at B={b}"
                )
            value += weight * sum(self.points[b, c] * w for c, w in column)
        if falls:
            raise _refused(
                view,
                f"the {self.name} cell is past the KV pool, and {'; '.join(falls)}, "
                "reaching no positive time at the step's c",
            )
        return value, self._cell(sides), "; ".join(past)


class StepGrid:
    """A bundle's grids: prefill by whether a chunk produces output, and decode;
    with `drafter`, read for each point's drafter value."""

    def __init__(self, bundle: Mapping, drafter: bool = False) -> None:
        self.graph_mode = bundle["graph_mode"]
        if self.graph_mode not in ("replay", "eager"):
            raise ValueError(
                f"graph mode {self.graph_mode!r} is neither replay nor eager"
            )
        if bundle["unit"] not in UNITS:
            raise ValueError(f"unit {bundle['unit']!r} is not one of {list(UNITS)}")
        self.unit = UNITS[bundle["unit"]]
        self.scope = bundle["scope"]
        self.draft_width = bundle.get("draft_width", 0)
        if type(self.draft_width) is not int or self.draft_width < 0:
            raise ValueError(f"the draft width {self.draft_width!r} is not a count")
        if drafter and not self.draft_width:
            raise ValueError("the grid states no draft width, so it holds no drafter")
        value, alpha_key = VALUES[drafter]
        part = "drafter " if drafter else ""
        pool = bundle["decode"].get("pool")
        self.pool = pool and (pool["block_size"], pool["blocks"])
        if pool and not all(type(n) is int and n > 0 for n in self.pool):
            raise ValueError(f"the decode KV pool {pool} is not two positive counts")
        self.lookahead = (pool or {}).get("lookahead", 0)
        if type(self.lookahead) is not int or self.lookahead < 0:
            raise ValueError(f"the decode KV pool's lookahead {pool} is not a count")
        prefill = bundle["prefill"]
        axes = ("B", "T", "c")
        nodes = [prefill["nodes"][axis] for axis in axes]
        self.prefill = {
            produces: _Cells(f"{part}prefill {key}", axes, nodes, prefill[key], value)
            for produces, key in ((True, "output"), (False, "no_output"))
        }
        decode = bundle["decode"]
        b_nodes, c_nodes = decode["nodes"]["B"], decode["nodes"]["c"]
        replay = self.graph_mode == "replay"
        for axis, nodes in (("B", b_nodes), ("c", c_nodes)):
            if isinstance(nodes, Mapping) != replay:
                raise ValueError(
                    f"under graph mode {self.graph_mode} the decode {axis} nodes "
                    f"are {'one list per rung' if replay else 'one list'}, not a "
                    f"{type(nodes).__name__}"
                )
        self.alpha = {}
        if not replay:
            if decode.get(alpha_key):
                raise ValueError("alpha is per rung, and an eager grid reads no rung")
            self.decode = _Cells(
                f"{part}decode", ("B", "c"), [b_nodes, c_nodes], decode["points"], value
            )
            return
        by_rung = {int(rung): [] for rung in b_nodes}
        for point in decode["points"]:
            rung, *at = point["at"]
            if rung not in by_rung:
                raise ValueError(
                    f"a decode point at {point['at']} is on rung {rung}, which "
                    "has no B nodes"
                )
            by_rung[rung].append({**point, "at": at})
        self.decode = {
            int(rung): _Cells(
                f"{part}decode rung {rung}",
                ("B", "c"),
                [nodes, c_nodes[rung]],
                by_rung[int(rung)],
                value,
            )
            for rung, nodes in b_nodes.items()
        }
        for rung, alpha in decode.get(alpha_key, {}).items():
            terms = tuple(float({"mean": 0, **alpha}[k]) for k in ALPHA)
            if int(rung) not in self.decode or not all(map(math.isfinite, terms)):
                raise ValueError(
                    f"alpha {alpha} is for rung {rung}, which has no decode grid, "
                    "or is not finite"
                )
            self.alpha[int(rung)] = terms

    def read(
        self, cells: _Cells, view: BatchView, at
    ) -> tuple[Fraction, bool, str, str]:
        """The value at `at` in `cells`, whether it is a node, the cell, and how
        far past the KV pool it is extrapolated, '' inside; or a refusal."""
        try:
            return (*cells.lookup(view, at), "")
        except CostRefused:
            pool, extra = self.pool, self.draft_width + 1 + self.lookahead
            extended = (
                pool and not view.prefill and cells.past_pool(view, at, *pool, extra)
            )
            if not extended:
                raise
            value, cell, beyond = extended
            return value, False, cell, beyond

    def longest(self, view: BatchView, at) -> tuple[float, tuple] | None:
        """A decode batch's alpha and where its lookup also reads, B rows of
        its longest row's cached context; None where it reads `at` alone."""
        terms = None if view.prefill else self.alpha.get(view.capture_rung)
        alpha = alpha_of(terms, view.requests) if terms else 0.0
        if not alpha:
            return None
        longest = max(r.context_tokens - r.query_tokens for r in view.requests)
        return alpha, (at[0], Fraction(longest))

    def value(self, view: BatchView) -> tuple[_Cells, tuple, Fraction, bool, str, str]:
        """The step's grid, its coordinates there, its lookup, whether that is
        one node's, the cells read, and how far past the KV pool; or a refusal."""
        cells, at = self.locate(view)
        value, on_node, cell, beyond = self.read(cells, view, at)
        read = self.longest(view, at)
        if read is None:
            return cells, at, value, on_node, cell, beyond
        alpha, top_at = read
        top, _, top_cell, top_beyond = self.read(cells, view, top_at)
        value += Fraction(alpha) * (top - value)
        cell = f"{cell}; longest row's cell {top_cell}, alpha {alpha!r}"
        beyond = "; ".join(p for p in (beyond, top_beyond) if p)
        return cells, at, value, False, cell, beyond

    def locate(self, view: BatchView) -> tuple[_Cells, tuple[Fraction, ...]]:
        """The grid a step is read in and its coordinates there; or a refusal."""
        rows = view.requests
        prefill, decode = view.prefill, view.decode
        if prefill and decode:
            raise _refused(
                view,
                "a step mixing prefill and decode rows; "
                "the grid prices a step of one kind",
            )
        counts = [n for row in rows for n in (row.query_tokens, row.context_tokens)]
        odd = [
            n
            for n in counts + [view.capture_rung]
            if n is not None and type(n) is not int
        ]
        if odd:
            raise _refused(view, f"a count is a {type(odd[0]).__name__}, not an int")
        at = coordinates(rows, bool(prefill))
        if prefill:
            if view.produces_output is None:
                raise _refused(
                    view,
                    "a prefill step that does not say whether it produces "
                    "output, which picks its grid",
                )
            cells = self.prefill[view.produces_output]
        else:
            width = self.draft_width + 1
            if any(row.query_tokens != width for row in decode):
                computes = (
                    "more than one token"
                    if width == 1
                    else f"other than the {width} tokens of a verify step of "
                    f"{self.draft_width} drafts"
                )
                raise _refused(
                    view,
                    f"a decode row computes {computes}; the decode grid has no "
                    "N_Q axis",
                )
            if self.graph_mode == "eager":
                cells = self.decode
            elif view.capture_rung is None:
                raise _refused(
                    view,
                    "a decode step replays no graph, "
                    "and a replay bundle has no eager decode",
                )
            else:
                cells = self.decode.get(view.capture_rung)
                if cells is None:
                    raise _refused(view, f"no decode rung {view.capture_rung}")
        return cells, at


class StepGridBackend(CostBackend):
    """Prices a step as the grid's value at its equivalent shape, times the
    residual; `bundle` names the artifact both came from, for `describe`.
    Under a drafter, `drafter` is the grid read for the drafter's value and
    the drafter's residual, which price the step's draft term alike."""

    def __init__(
        self,
        grid: StepGrid,
        residual: Residual,
        bundle: str,
        drafter: tuple[StepGrid, Residual] | None = None,
    ) -> None:
        self.grid, self.residual, self.bundle = grid, residual, bundle
        self.drafter = drafter

    @property
    def granularity(self) -> Granularity:
        return Granularity.STEP

    def price(self, batch_view: BatchView) -> StepCost:
        if not isinstance(batch_view, BatchView):
            raise TypeError(
                f"this backend prices a BatchView, not {type(batch_view).__name__}"
            )
        view = batch_view
        if view.image_tokens:
            raise _refused(view, IMAGES)
        eager = not view.prefill and view.capture_rung is None
        if eager and view.dp_dummy and self.grid.graph_mode != "eager":
            return eager_dp_dummy(SOURCE)
        target = _term("forward", self.grid, self.residual, view)
        drafts = [_term("draft", *self.drafter, view)] if self.drafter else []
        return StepCost([target], draft_terms=drafts)

    def describe(self) -> str:
        drafter = self.drafter and f"; drafter {self.drafter[1].describe()}"
        return (
            f"step grid, bundle {self.bundle}; {self.grid.graph_mode} "
            f"decode grid, timing scope {self.grid.scope}; {self.residual.describe()}"
            f"{drafter or ''}"
        )


def _term(name: str, grid: StepGrid, residual: Residual, view: BatchView) -> CostTerm:
    """The term `name`: `grid`'s lookup for the step times `residual`'s factor."""
    cells, at, value, on_node, cell, beyond = grid.value(view)
    rows = view.requests
    homogeneous = len({(r.query_tokens, r.context_tokens) for r in rows}) == 1
    looked_up = "measured" if on_node and homogeneous else "interpolated"
    lookup = float(value * grid.unit)
    factor, summed, outside = residual.factor(view)
    detail = (
        f"lookup {lookup!r} s, {'extrapolated' if beyond else looked_up}; "
        f"{cells.name} cell {cell}; residual {factor!r} = {summed}"
    )
    past = "; ".join(p for p in (beyond, outside) if p)
    source = f"{EXTRAPOLATED}: {past}" if past else SOURCE
    validated = not past and residual.validated(view, at)
    estimated = Provenance(Method.ESTIMATED, detail, source, validated=validated)
    return CostTerm(name, lookup * factor, estimated)
