"""What a price library can say about the files it was built from.

The identity is a by-product of the read the library already does, so these are
about the read: that it happens once, that the digest is of the bytes that were
parsed, and that the record survives whatever happens to the file afterwards.

Roles are asserted by name because the provenance side keys on them:
`oracle.price` and `oracle.price_graph` are inputs to *constructing the oracle*,
and are deliberately distinct from the `runtime.*` roles that decide a
deployment's actual capacity.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from atom.compass.core.cost.families import ParametricPriceLibrary
from atom.compass.core.cost.library import PriceLibrary
from atom.compass.runtime.microbench import signature_of


def _op(rows: int = 32) -> dict:
    return {
        "name": "aiter::gemm_a16w16",
        "input_shapes": [[rows, 17408], [5120, 17408]],
        "dtypes": ["bfloat16", "bfloat16"],
        "scalars": [["#2", "None"]],
    }


def _pair(tmp_path, tag="a", rows=32, seconds=1e-4):
    op = _op(rows)
    graph = {"ops": [op],
             "provenance": {"execution": {"body_rows_traced": rows}}}
    prices = {"prices": {signature_of(op): {
        "seconds": seconds, "kernels": {"k": seconds},
        "occurrences": 1, "name": op["name"]}}}
    gpath = tmp_path / f"g_{tag}.json"
    ppath = tmp_path / f"p_{tag}.json"
    gpath.write_text(json.dumps(graph))
    ppath.write_text(json.dumps(prices))
    return str(ppath), str(gpath)


def _digest(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_both_artifacts_are_retained_under_their_roles(tmp_path):
    ppath, gpath = _pair(tmp_path)
    library = PriceLibrary()
    library.add(ppath, gpath)

    assert [i.role for i in library.loaded_inputs] == [
        "oracle.price", "oracle.price_graph"]
    assert [i.requested for i in library.loaded_inputs] == [ppath, gpath]


def test_the_digest_is_of_the_bytes_that_were_parsed(tmp_path):
    ppath, gpath = _pair(tmp_path)
    library = PriceLibrary()
    library.add(ppath, gpath)

    assert library.loaded_inputs[0].sha256 == _digest(ppath)
    assert library.loaded_inputs[1].sha256 == _digest(gpath)


def test_replacing_the_file_afterwards_changes_neither_price_nor_record(
        tmp_path):
    """The record describes a read that happened, not a path that exists.

    If the digest were taken by reopening at query time, swapping the file
    would silently re-describe a library already built from the old bytes.
    """
    ppath, gpath = _pair(tmp_path, seconds=1e-4)
    library = PriceLibrary()
    library.add(ppath, gpath)
    before = library.loaded_inputs[0].sha256

    with open(ppath, "w", encoding="utf-8") as fh:
        json.dump({"prices": {}}, fh)

    assert library.loaded_inputs[0].sha256 == before
    assert before != _digest(ppath)
    record, _ = library.lookup(_op(32))
    assert record is not None and record["seconds"] == pytest.approx(1e-4)


def test_a_library_built_by_hand_says_so_with_an_empty_tuple(tmp_path):
    """Empty is a statement that there is no file, not a missing record."""
    library = PriceLibrary()
    assert library.loaded_inputs == ()


def test_the_record_is_immutable_and_not_mutated_in_place(tmp_path):
    """A holder that already read it cannot be changed under by a later add."""
    first = _pair(tmp_path, tag="a")
    second = _pair(tmp_path, tag="b", rows=64, seconds=2e-4)
    library = PriceLibrary()
    library.add(*first)
    held = library.loaded_inputs
    library.add(*second)

    assert isinstance(library.loaded_inputs, tuple)
    assert len(held) == 2 and len(library.loaded_inputs) == 4
    assert library.loaded_inputs[:2] == held


def test_a_price_file_with_no_graph_retains_only_the_price(tmp_path):
    ppath, _ = _pair(tmp_path)
    library = PriceLibrary()
    library.add(ppath, None)
    assert [i.role for i in library.loaded_inputs] == ["oracle.price"]


def test_load_carries_coords_down_to_every_add(tmp_path, monkeypatch):
    """Handed down unresolved, so resolution stays in exactly one place.

    A factory that resolves on the way in and passes the result would report
    `rank_own` false for a file that was this rank's own, and `requested` would
    hold a suffixed name no option ever carried.
    """
    from atom.compass.core import loaded_input

    seen = []
    real = loaded_input.resolve_rank_path
    monkeypatch.setattr(
        loaded_input, "resolve_rank_path",
        lambda requested, coords: (seen.append((requested, coords))
                                   or real(requested, coords)))

    ppath, gpath = _pair(tmp_path)
    PriceLibrary.load([(ppath, gpath)], coords={"tp": 2})

    assert seen == [(ppath, {"tp": 2}), (gpath, {"tp": 2})]


class TestParametricInherits:
    """The family library is the one the oracle actually builds."""

    def test_it_retains_the_same_records(self, tmp_path):
        ppath, gpath = _pair(tmp_path)
        library = ParametricPriceLibrary()
        library.add(ppath, gpath)

        assert [i.role for i in library.loaded_inputs] == [
            "oracle.price", "oracle.price_graph"]
        assert library.loaded_inputs[1].sha256 == _digest(gpath)

    def test_it_accepts_coords_too(self, tmp_path, monkeypatch):
        from atom.compass.core import loaded_input

        seen = []
        real = loaded_input.resolve_rank_path
        monkeypatch.setattr(
            loaded_input, "resolve_rank_path",
            lambda requested, coords: (seen.append(coords)
                                       or real(requested, coords)))

        ppath, gpath = _pair(tmp_path)
        ParametricPriceLibrary().add(ppath, gpath, coords={"tp": 4})

        assert seen == [{"tp": 4}, {"tp": 4}]

    def test_the_graph_is_opened_once_for_both_readers(self, tmp_path,
                                                       monkeypatch):
        """The base class reads layouts from it and this subclass reads rows.

        Two opens are not only wasted work: they are two chances to see
        different bytes, and the digest retained for provenance would then
        describe whichever read happened to be hashed.
        """
        import builtins

        opened: list[str] = []
        real_open = builtins.open

        def counting_open(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        ppath, gpath = _pair(tmp_path)
        monkeypatch.setattr(builtins, "open", counting_open)
        ParametricPriceLibrary().add(ppath, gpath)
        monkeypatch.undo()

        assert opened.count(gpath) == 1, opened
        assert opened.count(ppath) == 1, opened

    def test_it_still_prices_and_still_interpolates(self, tmp_path):
        """The read is a by-product; it must not have changed the behaviour."""
        library = ParametricPriceLibrary(max_gap_ratio=2.0)
        library.add(*_pair(tmp_path, tag="a", rows=32, seconds=1e-4))
        library.add(*_pair(tmp_path, tag="b", rows=64, seconds=2e-4))

        exact, _ = library.lookup(_op(32))
        assert exact is not None and exact["seconds"] == pytest.approx(1e-4)
        between, detail = library.lookup(_op(48))
        assert between is not None, detail
        assert between["interpolated"] is True
        assert len(library.loaded_inputs) == 4
