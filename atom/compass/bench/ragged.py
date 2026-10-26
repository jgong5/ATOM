# SPDX-License-Identifier: MIT
"""Ragged decode batches composed from recorded decode rows, for the bench.

`compose(seed, contexts, rng, blocks, block_size, max_len, ladder)` takes a
decode view, `seed`, and gives the ragged views that keep its rung and its
summed context: at each `RATIOS` value r, B rows whose longest context is
about r times their mean, for B the seed's row count, then the fewest and the
most rows its rung replays on the decode `ladder`, one more than the rung
below it, at least 2, and the rung itself, so the views' mean context spans
the rung's row range; a rung `ladder` lacks, or one replaying one row count,
composes at the seed's alone. At the seed's B their equivalent homogeneous shape, and the
step grid's lookup at it, are the seed's. The longest row is the context
of `contexts` nearest r times the mean; each other row is drawn uniformly from
`contexts` within `SPREADS` times the others' mean either side of it, and the
last one is the context that brings the sum back to the seed's, within one
token a row. Every row is a context `contexts` holds, a decode row's context
some record ran. At r = 1 every row is the context nearest the mean, once;
above it there is one view per spread. A ratio no such batch reaches, past
`max_len` or with a row below one token, a batch past `blocks` KV blocks of
`block_size` tokens, and a draw whose last row misses the sum after `TRIES`
draws give no view. The rows are shuffled, so the longest lands anywhere.
"""

import bisect
import itertools
import math

#: The longest row's context over the mean, the views of one seed sweep.
RATIOS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
#: How far the other rows spread around their mean, as a share of it.
SPREADS = (0.1, 1.0)
#: The draws a view gets before its last row must bring the sum back.
TRIES = 20
#: How far the longest row's ratio may miss its target, as a share of it.
NEAR = 0.05


def _nearest(contexts: list, x: float, top: int) -> int | None:
    """The context of `contexts` nearest `x` and at most `top`, or None."""
    i = bisect.bisect_left(contexts, x)
    near = [c for c in contexts[max(0, i - 1) : i + 1] if c <= top]
    return min(near, key=lambda c: abs(c - x)) if near else None


def _rows(b: int, total: int, ratio: float, spread: float, contexts, rng, top):
    """B contexts summing to `total` within B tokens, the longest about `ratio`
    times the mean; or None."""
    mean = total / b
    if ratio == 1.0:
        c = _nearest(contexts, mean, top)
        return None if c is None else [c] * b
    longest = _nearest(contexts, ratio * mean, top)
    if longest is None or abs(longest / mean - ratio) > NEAR * ratio:
        return None
    rest = (total - longest) / (b - 1)
    if not 1 <= rest <= longest:
        return None
    half = min(spread * rest, longest - rest, rest - 1)
    lo = bisect.bisect_left(contexts, rest - half)
    hi = bisect.bisect_right(contexts, rest + half)
    if lo == hi and b > 2:
        return None
    for _ in range(TRIES):
        others = [contexts[rng.randrange(lo, hi)] for _ in range(b - 2)]
        need = total - longest - sum(others)
        last = _nearest(contexts, need, longest)
        if last is not None and abs(last - need) <= b:
            return [longest, *others, last]
    return None


def compose(
    seed: dict, contexts: list, rng, blocks: int, block_size: int, max_len, ladder=()
):
    """The ragged views of the decode view `seed` (`rows`, `rung`,
    `produces_output`), over the sorted decode-row contexts `contexts`."""
    rows, rung = seed["rows"], seed["rung"]
    b, total = len(rows), sum(context for _, context, _ in rows)
    out = []
    if b < 2:
        return out
    counts = [b]
    if rung in ladder:
        below = [0, *ladder][ladder.index(rung)]
        counts += [n for n in (max(below + 1, 2), rung) if n not in counts]
    for b, ratio in itertools.product(counts, RATIOS):
        for spread in SPREADS[: 1 if ratio == 1.0 else None]:
            got = _rows(b, total, ratio, spread, contexts, rng, max_len)
            if got is None or sum(math.ceil(c / block_size) for c in got) > blocks:
                continue
            rng.shuffle(got)
            view = {**seed, "rows": [[1, c, True] for c in got]}
            if view not in out:
                out.append(view)
    return out
