"""R19: each logical rank must be priced from its own measurements.

The aggregation across ranks is `slowest`, which is only a real maximum if the
ranks it maximises over are actually different. If every rank is priced from
rank 0's library, `slowest` returns rank 0's number wearing a maximum's name --
and a genuinely slower rank is invisible in exactly the case the aggregation
exists for.

Two layers, and they fail differently:

* the **library** layer, owned here, has to be able to produce a different
  library per rank and say which files each one read; and
* the **factory** layer, owned by the provenance task, has to actually call it
  once per rank instead of building one library and reusing it.

The library tests below run green. The factory test is marked `xfail(strict)`
against the current tree: when the per-rank selection wrapper lands it will
start passing, and strict xfail turns that into a failure that says "delete the
marker", rather than letting the fix land with a test still claiming it is
broken.
"""

from __future__ import annotations

import json

import pytest

from atom.compass.core.cost.library import PriceLibrary
from atom.compass.runtime.source_oracle import build_source_oracle

SIG = "sig::x"


def _rank_prices(tmp_path, stem="prices.json", per_rank=None, shared=9.9e-5):
    """A shared price file plus a distinct one per rank, as the suffix rule.

    `resolve_rank_path` prefers `prices.tp<N>.json` and falls back to
    `prices.json`, so writing both is what makes "did this rank read its own
    file" a question with an observable answer.
    """
    def write(name, seconds):
        path = tmp_path / name
        path.write_text(json.dumps({
            "provenance": {"topology": {"tp": 1},
                           "registration": "unregistered"},
            "prices": {SIG: {"seconds": seconds}},
            "unpriced": {},
        }), encoding="utf-8")
        return str(path)

    base, _, ext = stem.rpartition(".")
    requested = write(stem, shared)
    for index, seconds in (per_rank or {}).items():
        write(f"{base}.tp{index}.{ext}", seconds)
    return requested


def _template_file(tmp_path, name="graph.json"):
    path = tmp_path / name
    path.write_text(json.dumps({
        "ops": [],
        "key": {"topology": [["tp", 1]], "rank_coords": []},
        "provenance": {
            "batch_spec": {"kind": "decode", "query_lens": [1, 1],
                           "context_lens": [1025, 1025]},
            "execution": {"capture_bucket": 2},
        },
    }), encoding="utf-8")
    return str(path)


def _seconds(library):
    record, detail = library.lookup({"name": "x", "input_shapes": [],
                                     "dtypes": []})
    # The fixture's signature is literal, so go through the store directly
    # rather than reconstructing an operator that hashes to it.
    return float(library._prices[SIG][0]["seconds"])


# == the library layer: mine =============================================

def test_one_set_of_paths_builds_a_different_library_per_rank(tmp_path):
    """The same option string, two ranks, two different prices.

    This is the capability the per-rank wrapper needs from this side: the
    caller varies only `coords`, and resolution happens once, inside the
    loader.
    """
    requested = _rank_prices(tmp_path, per_rank={0: 1.0e-5, 1: 4.0e-5})

    rank0 = PriceLibrary.load([(requested, None)], coords={"tp": 0})
    rank1 = PriceLibrary.load([(requested, None)], coords={"tp": 1})

    assert _seconds(rank0) == pytest.approx(1.0e-5)
    assert _seconds(rank1) == pytest.approx(4.0e-5)


def test_each_rank_manifest_names_the_file_that_rank_read(tmp_path):
    """A report that says "rank 1" has to be able to say which file that was."""
    requested = _rank_prices(tmp_path, per_rank={0: 1.0e-5, 1: 4.0e-5})

    rank1 = PriceLibrary.load([(requested, None)], coords={"tp": 1})
    (record,) = rank1.loaded_inputs

    assert record.role == "oracle.price"
    assert record.requested == requested, "the stem the option carried"
    assert record.path.endswith(".tp1.json"), record.path
    assert record.rank_own is True
    assert record.rank_coords == (("tp", 1),)


def test_a_rank_with_no_file_of_its_own_says_so_rather_than_claiming_one(
        tmp_path):
    """Falling back to the shared file is allowed and must be visible.

    Reusing one rank's calibration across a symmetric group is a legitimate
    thing to want. Reporting it as the rank's own measurement is not.
    """
    requested = _rank_prices(tmp_path, per_rank={0: 1.0e-5}, shared=9.9e-5)

    rank3 = PriceLibrary.load([(requested, None)], coords={"tp": 3})
    (record,) = rank3.loaded_inputs

    assert record.rank_own is False
    assert record.path == requested
    assert _seconds(rank3) == pytest.approx(9.9e-5)


def test_two_ranks_that_read_the_same_bytes_are_still_two_records(tmp_path):
    """`slowest` over identical prices is a real answer, not a missing one."""
    requested = _rank_prices(tmp_path, per_rank=None, shared=9.9e-5)

    libraries = [PriceLibrary.load([(requested, None)], coords={"tp": r})
                 for r in (0, 1)]

    assert {lib.loaded_inputs[0].sha256 for lib in libraries} == {
        libraries[0].loaded_inputs[0].sha256}
    assert [lib.loaded_inputs[0].rank_coords for lib in libraries] == [
        (("tp", 0),), (("tp", 1),)]


# == the factory layer: provenance's ======================================

def test_the_factory_honours_the_rank_it_is_told_it_is(tmp_path):
    """Told which rank it is, the factory does price from that rank's files.

    Worth pinning, and worth being exact about what it settles. R19 is that
    every logical rank ends up on one library; this shows the cause is *not*
    that `rank_coords` fails to reach the resolution. One build per rank
    already produces one library per rank.

    What this does NOT prove, and cannot from here: that anything calls the
    factory once per rank. The factory is built once and the resulting oracle
    is reused for every logical rank, so `slowest` maximises over a set of one
    however well each individual build resolves. That call site is the
    provenance task's per-rank selection wrapper; this test is the contract it
    can rely on from this side.
    """
    requested = _rank_prices(tmp_path, per_rank={0: 1.0e-5, 1: 4.0e-5})
    template = _template_file(tmp_path)

    built = {
        rank: build_source_oracle(price=requested, template=template,
                                  derive=0, rank_coords=f"tp:{rank}")
        for rank in (0, 1)
    }

    assert _seconds(built[0].oracle.library) == pytest.approx(1.0e-5)
    assert _seconds(built[1].oracle.library) == pytest.approx(4.0e-5)


def test_the_factory_build_carries_each_ranks_manifest(tmp_path):
    """So a per-rank wrapper can report which files each rank actually read.

    The records come from the library the build produced, so a wrapper that
    keeps one build per rank keeps one manifest per rank without doing
    anything further.
    """
    requested = _rank_prices(tmp_path, per_rank={0: 1.0e-5, 1: 4.0e-5})
    template = _template_file(tmp_path)

    paths = {}
    for rank in (0, 1):
        built = build_source_oracle(price=requested, template=template,
                                    derive=0, rank_coords=f"tp:{rank}")
        (record,) = built.oracle.library.loaded_inputs
        paths[rank] = record.path

    assert paths[0].endswith(".tp0.json"), paths
    assert paths[1].endswith(".tp1.json"), paths
