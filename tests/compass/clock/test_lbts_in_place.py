# SPDX-License-Identifier: MIT
"""`ClockAuthority._lbts`, summed in place without building the row.

What the groups below defend:

* **The least term of the row.** On random tables, with each LP's N drawn from
  a few values so that ties and infinities occur, `_lbts` equals the least term
  `_row` gives, and is infinite for an LP no peer reaches.
* **Every grant decision of the generated runs** reads that least term: plain
  LPs, LPs below ceilings, and a DP group whose members are granted ahead.
* **A subclass that redefines `_row`** is bounded by its own row, not by the
  in-place sum.
* **No term source `_row` reads is left out of `_lbts`**: every attribute the
  row's code loads, its nested code's included, `_lbts` loads too. A source
  added to the row alone fails here, naming `_lbts`, at no grant-scan cost.
"""

import dis
import math
import random
import types

import pytest

from atom.compass.clock import ClockAuthority, LpId

from .test_grant_rule import (
    PICKS,
    SEEDS,
    _drive,
    _drive_below_ceilings,
    _drive_group,
    _table,
)

INF = math.inf


def _row_least(ca, i):
    return min((term for _, term in ca._row(i)), default=INF)


def _random_table(rng):
    lps = [f"lp{k}" for k in range(rng.randint(1, 7))]
    rows = [
        (src, dst, rng.choice((0.0, 2.0**-10, 0.5, 1.0, 3.0)))
        for src in lps
        for dst in lps
        if src != dst and rng.random() < 0.4
    ]
    return _table(lps, rows)


def test_lbts_is_the_least_term_of_the_row_on_random_tables():
    rng = random.Random(4231)
    seen = set()
    for _ in range(300):
        table = _random_table(rng)
        ca = ClockAuthority(table)
        ids = table.registry.ids()
        for _ in range(5):
            for i in ids:
                ca._nv[i] = rng.choice((0.0, 0.5, 1.25, INF, rng.uniform(0.0, 9.0)))
            for i in ids:
                least = _row_least(ca, i)
                assert ca._lbts(i) == least, (str(i), dict(ca._row(i)))
                seen.add("infinite" if least == INF else "finite")
    assert seen == {"finite", "infinite"}


@pytest.fixture
def checked(monkeypatch):
    """Every `_lbts` the authority asks, checked against the row's least term."""
    in_place, asked = ClockAuthority._lbts, []

    def lbts(self, i):
        least = in_place(self, i)
        assert least == _row_least(self, i), (str(i), dict(self._row(i)))
        asked.append(least)
        return least

    monkeypatch.setattr(ClockAuthority, "_lbts", lbts)
    return asked


def test_every_grant_decision_of_the_generated_runs_reads_the_least_term(checked):
    for seed in SEEDS:
        _drive(seed, PICKS["shuffled"], shuffle=seed)
        _drive_below_ceilings(seed, PICKS["name order"], ceilings=True)
        _drive_group(seed, PICKS["reverse"], True, ahead=True)
    assert any(t < INF for t in checked) and INF in checked


class _Constant(ClockAuthority):
    """A rule of its own: every peer bounds an LP at 42."""

    def _row(self, i):
        return tuple((j, 42.0) for j in self._channels.registry.ids() if j != i)


def test_a_subclass_that_redefines_the_row_is_bounded_by_its_own_row():
    table = _table(("a", "b"), [("a", "b", 0.5)])
    assert ClockAuthority(table)._lbts(LpId("b")) == 0.5
    assert ClockAuthority(table)._lbts(LpId("a")) == INF
    assert _Constant(table)._lbts(LpId("b")) == 42.0
    assert _Constant(table)._lbts(LpId("a")) == 42.0


def _attributes(function):
    """Every attribute `function`'s code loads, its nested code's included."""
    names, codes = set(), [function.__code__]
    while codes:
        code = codes.pop()
        names |= {
            ins.argval
            for ins in dis.get_instructions(code)
            if ins.opname in ("LOAD_ATTR", "LOAD_METHOD")
        }
        codes += [c for c in code.co_consts if isinstance(c, types.CodeType)]
    return names


def test_lbts_loads_every_attribute_the_row_loads():
    missing = _attributes(ClockAuthority._row) - _attributes(ClockAuthority._lbts)
    assert not missing, (
        f"ClockAuthority._lbts loads none of {sorted(missing)}, which _row loads "
        "for its terms: its in-place loop must take those terms too"
    )
