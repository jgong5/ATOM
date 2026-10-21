# SPDX-License-Identifier: MIT
"""`ragged.compose`: ragged decode views that keep a seed's rung, row count and
summed context, the longest row at each ratio to the mean, every row a context
the records ran."""

import math
import random

import pytest

from atom.compass.backends.shape import RequestShape
from atom.compass.backends.step_grid import length_statistics
from atom.compass.bench import ragged

CONTEXTS = list(range(1, 20_000, 3))
SEED = {"rows": [[1, c, True] for c in (900, 1000, 1100, 1000)] * 2, "rung": 8}
SEED["produces_output"] = True


def compose(seed=SEED, contexts=CONTEXTS, blocks=10**6, max_len=20_000):
    return ragged.compose(seed, contexts, random.Random(7), blocks, 16, max_len)


def test_each_view_keeps_the_seeds_rung_rows_and_sum_and_reaches_its_ratio():
    views = compose()

    total = sum(c for _, c, _ in SEED["rows"])
    mean = total / len(SEED["rows"])
    longest = sorted(max(c for _, c, _ in v["rows"]) / mean for v in views)
    assert len(views) == 1 + 2 * (len(ragged.RATIOS) - 1)
    for view in views:
        assert (view["rung"], view["produces_output"]) == (8, True)
        assert len(view["rows"]) == len(SEED["rows"])
        assert abs(sum(c for _, c, _ in view["rows"]) - total) <= len(view["rows"])
        assert all(q == 1 and decode and c in CONTEXTS for q, c, decode in view["rows"])
    for ratio in ragged.RATIOS:
        hits = [r for r in longest if abs(r - ratio) <= ragged.NEAR * ratio]
        assert len(hits) == (1 if ratio == 1.0 else 2)
    # At each ratio the view of the wider spread, composed second, spreads its
    # rows further.
    spreads = {}
    for view in views:
        rows = [RequestShape(*r) for r in view["rows"]]
        ratio, spread = length_statistics(rows)
        spreads.setdefault(round(ratio, 1), []).append(spread)
    assert all(len(s) == 2 and s[1] > s[0] for r, s in spreads.items() if r)


def ratios(views, mean=1000):
    return {round(max(c for _, c, _ in v["rows"]) / mean, 2) for v in views}


def test_the_longest_row_ends_where_the_rows_and_max_len_end():
    """Two rows put the longest at most twice the mean, the other near 0; eight
    rows reach 2.5 times it under a `max_len` of 2600, and not 3."""
    two = {**SEED, "rows": [[1, 1000, True]] * 2}
    assert ratios(compose(two)) == {1.0, 1.25, 1.5, 2.0}
    assert ratios(compose(max_len=2600)) == {1.0, 1.25, 1.5, 2.0, 2.5}
    assert compose({**SEED, "rows": [[1, 1000, True]]}) == []


def test_a_view_past_the_kv_pool_is_not_composed():
    """Every view holds at least its summed context over 16 in blocks."""
    floor = sum(c for _, c, _ in SEED["rows"]) // 16
    assert compose(blocks=floor - len(SEED["rows"])) == []
    assert all(
        sum(math.ceil(c / 16) for _, c, _ in v["rows"]) <= floor + 8
        for v in compose(blocks=floor + 8)
    )


@pytest.mark.parametrize("contexts", [[1000, 1100], [1000]])
def test_a_ratio_the_contexts_cannot_reach_gives_no_view(contexts):
    views = compose(contexts=contexts)
    assert [max(c for _, c, _ in v["rows"]) for v in views] == [1000]
