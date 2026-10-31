# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The step grid's points planned from batch shapes, and steps checked for
coverage against a loaded step grid backend.

Both take steps as step-record lines (`atom/compass/parity/`), each in a
`(name, line)` pair whose name a report gives the step, and read each step
where the step grid backend reads it (`StepGrid.locate` and the cell's
corners), so the plan and the backend never disagree on a cell. Only shapes
are read, never timestamps. Reading records stays with the caller.

`plan` reads its target from the engine its inputs' ranks state, each in its
`dp<r>.engine.json`: `TARGET`, the scheduler's bounds, the KV block size and
count, and the decode ladder. The block count is the smallest any rank
states, since a real start sets it from its free memory; inputs whose ranks
state two engines otherwise are refused. A decode row computes the engine's
decode width N_Q: one token, or under the drafter its engine states
(`speculative`, [method, K]) the K + 1 of a verify step, whose draft round
also writes KV at K - 1 positions past the row, the lookahead ATOM's
scheduler allocates with it (`block_manager.draft_kv_lookahead`); a step
whose decode row computes another count is refused. The grid definition is a
mapping:

    {"prefill": {"B": [...], "T": [...], "c": [...]}, "decode": {"c": [...]},
     "margin": n}

`T` is a step's token count and `c` its cached context; `margin` is the cells
added outward on every axis, and the rungs either side on the ladder. Every
number is a non-negative int. The definition may also state a `TARGET` field,
which is refused unless the engine states the same value.

`plan` first places the nodes the target's bounds give, dropping any
definition node past one. Prefill's top nodes are `max_num_seqs` on B,
`max_num_batched_tokens` on T and `max_model_len - max_num_batched_tokens` on
c, so `max_model_len` below the token budget is refused. Each decode rung has
B nodes at the previous rung + 1, at the rung, and at each B between them that
a step on the rung reaches past the rung's KV limit; its c nodes add each B
node's KV limit, `block size * (num_kvcache_blocks // B) - N_Q - lookahead`,
and `max_model_len - N_Q`; the decode grid records that pool, with the
lookahead, which a lookup extrapolates past, and the draft width K.

It then plans every corner of each step's cell, widened by `margin` nodes on
every axis and to `margin` rungs either side, each such rung at all its B
nodes, a decode step's c span reaching on to the node at or above its longest
row's cached context, which the lookup also reads; and, on any rung between
two planned rungs, its B nodes at its c nodes spanning those planned, since a
bundle whose measured rungs skip one of the ladder's does not load. A point
that is no batch the target runs is not planned: B above `max_num_seqs`, T
above `max_num_batched_tokens`, T not a multiple of B or below it, c + N_Q
above `max_model_len`, or more KV blocks than the block count. B rows hold
B * ceil((c + N_Q + lookahead) / block size) blocks; past the count, a
decode point's rows share the fewest leading blocks that bring them into it,
as a server's prefix cache lets rows share, and its line states them as
`shared`; they are full blocks of the cached context, so at most
floor(c / block size).
"""

import bisect
import itertools
from collections import defaultdict

from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_grid import StepGrid

#: The one timing scope a bundle may declare, the whole `ModelRunner.forward`.
SCOPE = "forward"
#: The scheduler's bounds a bundle records, which its grid was planned for.
BOUNDS = ("max_num_seqs", "max_num_batched_tokens", "max_model_len")
#: The engine fields `plan` plans for.
TARGET = (*BOUNDS, "kv_cache_block_size", "num_kvcache_blocks", "ladder")
#: The grid definition's own fields.
DEFINITION = ("prefill", "decode", "margin")


def keyed_of(engine: dict) -> dict:
    """`run.step_grid.keyed`'s fields of an engine file's mapping `engine`, which
    adds the scheduler's bounds and the KV block count to them."""
    return {k: v for k, v in engine.items() if k not in (*BOUNDS, "num_kvcache_blocks")}


def _views(steps) -> list[tuple[str, BatchView]]:
    """Each named step-record line as the view a simulated runner prices."""
    views = [
        (
            name,
            BatchView(
                tuple(RequestShape(*row) for row in line["rows"]),
                capture_rung=line["rung"],
                produces_output=line["produces_output"],
            ),
        )
        for name, line in steps
    ]
    if not views:
        raise ValueError("there is no step to read")
    return views


def _target(definition: dict, engines: dict) -> dict:
    """The `TARGET` fields of the one engine every input rank states, at the
    smallest KV block count any states, whose KV limits hold for every input,
    and its drafter's draft width and lookahead, 0 without one."""
    if not engines:
        raise ValueError("no input states the engine its steps ran on")
    for name, engine in engines.items():
        missing = [key for key in TARGET if key not in engine]
        if missing:
            raise ValueError(f"{name} states no {', '.join(missing)}")
    (first, engine), *rest = engines.items()
    for name, other in rest:
        keys = sorted((engine.keys() | other.keys()) - {"num_kvcache_blocks"})
        differ = [k for k in keys if engine.get(k) != other.get(k)]
        if differ:
            raise ValueError(
                f"{first} and {name} state two engines, which differ on "
                f"{', '.join(differ)}"
            )
    blocks = min(e["num_kvcache_blocks"] for e in engines.values())
    engine = {**engine, "num_kvcache_blocks": blocks}
    unknown = sorted(set(definition) - {*DEFINITION, *TARGET})
    if unknown:
        raise ValueError(
            f"the grid definition states {', '.join(unknown)}, which plan does "
            "not read; its target is the engine's"
        )
    missing = [key for key in DEFINITION if key not in definition]
    if missing:
        raise ValueError(f"the grid definition states no {', '.join(missing)}")
    differ = [k for k in TARGET if k in definition and definition[k] != engine[k]]
    if differ:
        raise ValueError(
            "the grid definition states "
            + "; ".join(
                f"{k} {definition[k]!r} and the engine {engine[k]!r}" for k in differ
            )
        )
    drafts = (engine.get("speculative") or (None, 0))[1]
    if type(drafts) is not int or drafts < 0:
        raise ValueError(
            f"{first} states the drafter {engine['speculative']!r}, whose draft "
            "width is not a count"
        )
    drafter = {"draft_width": drafts, "lookahead": _lookahead(drafts)}
    return {key: engine[key] for key in TARGET} | drafter


def _reason(refused: CostRefused) -> str:
    return "; ".join(r.reason for r in refused.declined)


def _span(nodes, low, high, margin: int) -> list:
    """`nodes` from the last at or below `low` to the first at or above `high`,
    widened by `margin` nodes either way."""
    first = bisect.bisect_right(nodes, low) - 1
    last = bisect.bisect_left(nodes, high)
    return nodes[max(0, first - margin) : last + margin + 1]


def _lookahead(drafts: int) -> int:
    """The KV positions past a row's context that one forward's draft round
    writes under a drafter of draft width `drafts`, which ATOM's scheduler
    allocates with the row (`block_manager.draft_kv_lookahead`)."""
    return max(0, drafts - 1)


def _extra(target: dict) -> int:
    """The KV positions a decode row holds past its cached context: its
    tokens, and those its draft round writes past them."""
    return target.get("draft_width", 0) + 1 + target.get("lookahead", 0)


def _shared(target: dict, b: int, c: int) -> int:
    """The leading KV blocks B decode rows on c cached share to fit the
    target's pool: none where they fit apart, else the fewest that bring them
    into it, at most the full blocks their cached context fills."""
    size = target["kv_cache_block_size"]
    over = b * -(-(c + _extra(target)) // size) - target["num_kvcache_blocks"]
    return min(c // size, -(-over // (b - 1))) if over > 0 and b > 1 else 0


def _infeasible(target: dict, b: int, t: int, c: int, prefill: bool) -> str:
    """Why the target cannot run B rows of T tokens in all on c cached each,
    decode rows sharing `_shared` blocks, or ''."""
    n_q, shared = t // b, 0 if prefill else _shared(target, b, c)
    held = c + n_q + target.get("lookahead", 0)
    blocks = b * -(-held // target["kv_cache_block_size"]) - (b - 1) * shared
    checks = [
        (b > target["max_num_seqs"], f"B {b} is above max_num_seqs"),
        (
            prefill and t > target["max_num_batched_tokens"],
            f"T {t} is above max_num_batched_tokens",
        ),
        (t < b, f"T {t} is below B {b}, under one token a row"),
        (t % b, f"N_Q = T/B = {t}/{b} is not an integer"),
        (
            c + n_q > target["max_model_len"],
            f"c + N_Q {c + n_q} is above max_model_len",
        ),
        (
            blocks > target["num_kvcache_blocks"],
            f"{blocks} KV blocks are above num_kvcache_blocks"
            + (f", the rows sharing {shared}" if shared else ""),
        ),
    ]
    return next((why for over, why in checks if over), "")


def _c_max(target: dict, b: int) -> int:
    """The largest cached context at which B decode rows fit the KV pool."""
    blocks = target["num_kvcache_blocks"] // b
    return target["kv_cache_block_size"] * blocks - _extra(target)


def _capped(nodes, top: int) -> list:
    """`nodes` below `top`, then `top`."""
    return sorted({n for n in nodes if n < top} | {top})


def _layout(definition: dict, b_nodes: dict) -> dict:
    """The grid `definition` makes with these decode B nodes per rung, its
    nodes past a bound dropped and the bounds added: the token budget T's top,
    ``max_num_seqs`` B's and ``max_model_len`` less the budget c's for
    prefill; per decode rung, each B node's KV limit and ``max_model_len``
    less the decode width as c nodes, and the KV pool they come from."""
    budget = definition["max_num_batched_tokens"]
    tops = {
        "B": definition["max_num_seqs"],
        "T": budget,
        "c": definition["max_model_len"] - budget,
    }
    drafts, ahead = definition["draft_width"], definition["lookahead"]
    c, top = definition["decode"]["c"], definition["max_model_len"] - drafts - 1
    return {
        **({"draft_width": drafts} if drafts else {}),
        "graph_mode": "replay",
        "scope": SCOPE,
        "bounds": {name: definition[name] for name in BOUNDS},
        "prefill": {
            "nodes": {a: _capped(definition["prefill"][a], t) for a, t in tops.items()},
            "output": [],
            "no_output": [],
        },
        "decode": {
            "nodes": {
                "B": {str(r): sorted(bs) for r, bs in b_nodes.items()},
                "c": {
                    str(r): _capped(c + [_c_max(definition, b) for b in bs], top)
                    for r, bs in b_nodes.items()
                },
            },
            "points": [],
            "pool": {
                "block_size": definition["kv_cache_block_size"],
                "blocks": definition["num_kvcache_blocks"],
                **({"lookahead": ahead} if ahead else {}),
            },
        },
    }


def _line(step: int, phase: str, key: str, at: list, shared: int, width) -> dict:
    """The planned point `at`, `[B, T, c]` or `[rung, B, c]`, as a step-record
    line: B rows of N_Q = T/B tokens, or of the decode `width`, on c cached,
    sharing their first `shared` KV blocks."""
    if phase == "prefill":
        (b, t, c), rung, output = at, None, key == "output"
        row = [t // b, c + t // b, False]
    else:
        (rung, b, c), output = at, True
        row = [width, c + width, True]
    return {
        "step": step,
        "dp_rank": 0,
        "is_dummy_run": False,
        "produces_output": output,
        "batch": [],
        "named": [],
        "rows": [list(row) for _ in range(b)],
        "rung": rung,
        "origin": "plan",
    } | ({"shared": shared} if shared else {})


def plan(definition: dict, steps, engines: dict) -> dict:
    """The points to measure for `steps` in the grid `definition` makes for
    the engine `engines` states, a mapping from each input rank's name to the
    engine it states.

    It returns `graph_mode`, `scope` and `bounds`, under a drafter its
    `draft_width`, and `prefill` and `decode` in a bundle grid's layout
    (`step_grid.py`) with no point measured;
    `lines`, each planned point as a step-record line with origin `plan`,
    which that grid locates on the point's node, a decode point past the KV
    pool stating the blocks its rows share; `counts` per grid;
    `infeasible`, the points not planned and why; and `unplaced`, the steps
    outside the grid with the backend's reason.
    """
    steps = _views(steps)
    definition = {**definition, **_target(definition, engines)}
    width = definition["draft_width"] + 1
    for name, view in steps:
        if other := sorted({r.query_tokens for r in view.decode} - {width}):
            many, n = "more than one token", other[0]
            if width > 1:
                many = f"{n} token{'s' * (n > 1)}"
            verifies = f"verifies {width - 1} drafts" if width > 1 else "has no drafter"
            raise ValueError(
                f"{name} decodes {many} in a row, and the engine its inputs "
                f"state {verifies}"
            )
    if sorted(definition["prefill"]) != ["B", "T", "c"]:
        raise ValueError(
            f"the grid definition's prefill nodes are on "
            f"{', '.join(definition['prefill'])}, not B, T, c"
        )
    ladder = definition["ladder"]
    nodes = [*definition["prefill"].values(), definition["decode"]["c"]]
    counts = [definition[k] for k in TARGET[:-1]] + [definition["margin"]]
    counts += ladder + [n for axis in nodes for n in axis]
    odd = [n for n in counts if type(n) is not int or n < 0]
    if odd:
        raise ValueError(f"the grid definition's or engine's {odd[0]!r} is not a count")
    if definition["max_model_len"] < definition["max_num_batched_tokens"]:
        raise ValueError(
            "the engine's max_model_len is below its max_num_batched_tokens, so "
            "prefill's top c node, max_model_len less the token budget, would "
            "be negative"
        )
    margin = definition["margin"]
    b_nodes = {r: {p + 1, r} for p, r in zip([0, *ladder], ladder)}
    # A B at which a step reaches past its rung's KV limit is a B node of the
    # rung, so that its own limit is a c node there.
    grid = StepGrid({**_layout(definition, b_nodes), "unit": "s"})
    for _, view in steps:
        if view.prefill or view.capture_rung not in b_nodes:
            continue
        try:
            _, (b, c) = grid.locate(view)
        except CostRefused:
            continue
        if c > _c_max(definition, view.capture_rung):
            b_nodes[view.capture_rung].add(int(b))
    out = _layout(definition, b_nodes)
    grid = StepGrid({**out, "unit": "s"})
    planned, unplaced = defaultdict(set), []
    for where, view in steps:
        try:
            cells, at = grid.locate(view)
            corners = cells.corners(view, at)
        except CostRefused as refused:
            unplaced.append({"step": where, "reason": _reason(refused)})
            continue
        low, high = corners[0][0], corners[-1][0]
        if view.prefill:
            spans = [
                _span(n, low[i], high[i], margin) for i, n in enumerate(cells.nodes)
            ]
            key = "output" if view.produces_output else "no_output"
            planned["prefill", key].update(itertools.product(*spans))
            continue
        # On a rung either side, every B node, at the c nodes spanning the cell's
        # and on to the longest row's.
        longest = max(r.context_tokens - r.query_tokens for r in view.decode)
        k = ladder.index(view.capture_rung)
        for rung in ladder[max(0, k - margin) : k + margin + 1]:
            bs, cs = grid.decode[rung].nodes
            if rung == view.capture_rung:
                bs = _span(bs, low[0], high[0], margin)
            planned["decode", "points"].update(
                (rung, *p)
                for p in itertools.product(bs, _span(cs, low[1], longest, margin))
            )
    # A bundle whose measured rungs skip one of the ladder's does not load.
    decode = planned["decode", "points"]
    rungs = sorted({p[0] for p in decode})
    c_planned = [p[2] for p in decode]
    for rung in ladder:
        if rungs and rungs[0] < rung < rungs[-1] and rung not in rungs:
            bs, cs = grid.decode[rung].nodes
            spanned = _span(cs, min(c_planned), max(c_planned), 0)
            decode.update(itertools.product([rung], bs, spanned))
    out |= {"lines": [], "counts": {}, "infeasible": [], "unplaced": unplaced}
    for (phase, key), points in sorted(planned.items()):
        for point in sorted(points):
            at = [int(x) for x in point]
            if phase == "prefill":
                name, why = f"prefill {key}", _infeasible(definition, *at, prefill=True)
            else:
                name = f"decode rung {at[0]}"
                why = _infeasible(definition, at[1], at[1] * width, at[2], False)
            if why:
                out["infeasible"].append({"grid": name, "at": at, "reason": why})
                continue
            shared = 0 if phase == "prefill" else _shared(definition, *at[1:])
            line = _line(len(out["lines"]), phase, key, at, shared, width)
            out["lines"].append(line)
            out["counts"][name] = out["counts"].get(name, 0) + 1
    return out


def coverage(backend, steps) -> dict:
    """How the grid and residual of `backend`, a `StepGridBackend`, answer
    each step: inside the cells and the residual's hull (`covered`), past the
    KV pool or the hull (`extrapolated`), or not at all, the step roofline
    below the grid answering at run time (`fallback`). The roofline refuses
    only a step with image tokens, which no step-record line holds, so no
    step goes unanswered.

    It returns `steps`, `covered`, `share` (covered over steps),
    `extrapolated` and `fallback`, each a count; `declined`, each step the
    grid does not answer, with its reasons; `missing`, the unmeasured corners
    of their cells by grid, a decode step's longest row's cell among them where
    its rung holds alpha, keyed `prefill output`, `prefill no_output` or
    `decode rung <R>`; and, apart, `prefill_above_top_c`, how many prefill
    steps are cached past prefill's top c node, which fall back: a prefix hit
    near `max_model_len`, or a long prompt's last chunk when `max_model_len`
    is not a multiple of the token budget.
    """
    steps = _views(steps)
    extrapolated, above, declined, missing = 0, 0, [], defaultdict(set)
    for where, view in steps:
        gaps = None
        try:
            cells, at = backend.grid.locate(view)
            above += bool(view.prefill) and at[-1] > cells.nodes[-1][-1]
            longest = backend.grid.longest(view, at)
            gaps = cells.name, set()
            for x in [at] + ([longest[1]] if longest else []):
                corners = cells.corners(view, x)
                gaps[1].update(c for c, _ in corners if c not in cells.points)
        except CostRefused:
            pass
        try:
            cost = backend.price(view)
        except CostRefused as r:
            declined.append({"step": where, "reasons": [_reason(r)]})
            if gaps:
                missing[gaps[0]] |= gaps[1]
            continue
        extrapolated += cost.terms[0].provenance.extrapolated
    covered = len(steps) - extrapolated - len(declined)
    return {
        "steps": len(steps),
        "covered": covered,
        "share": covered / len(steps),
        "extrapolated": extrapolated,
        "fallback": len(declined),
        "declined": declined,
        "missing": {
            name: sorted([int(x) for x in corner] for corner in corners)
            for name, corners in sorted(missing.items())
            if corners
        },
        "prefill_above_top_c": above,
    }
