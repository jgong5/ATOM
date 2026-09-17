"""A run must attribute the region coefficients it priced with.

Deliberately importable against a tree that has no snapshot helper: everything
here is `json` and the cell fixture. That is what makes it a regression rather
than a statement that a new name is new -- run against a tree without the
binding, each of these fails on its assertion, because the cell is accepted
with its preparation and postprocess coefficients attributed to a name.

The producer side -- what a snapshot contains, and that moving a coefficient
moves its digest -- is `test_region_calibration.py`, which does import the new
helper because it is about it.
"""

import json

from . import test_cc_traces_validate as base

cell = base.cell
run = base.run
verdict = base.verdict


def _failures(cell_dir):
    return verdict(cell_dir)["failures"]


def _without_region_declaration(cell_dir):
    """Drop whatever declares a region preset, leaving every other input."""
    path = cell_dir / "registry.json"
    blob = json.loads(path.read_text())
    blob["artifacts"] = [entry for entry in blob["artifacts"]
                         if entry.get("kind") != "region_model"]
    path.write_text(json.dumps(blob))


def _rank(cell_dir, **changes):
    path = cell_dir / "modelled.r1.json"
    blob = json.loads(path.read_text())
    rank = blob["run"]["server"]["compass"]["loaded_inputs"]["ranks"][0]
    for key, value in changes.items():
        if value is base.KEEP:
            rank.pop(key, None)
        else:
            rank[key] = value
    base._write(path, blob)


class TestAnUnattributedRegionPresetIsRefused:

    def test_a_cell_whose_preset_is_undeclared_is_refused(self, cell):
        """Every file input and every scalar is still valid. What is missing
        is any declaration of the coefficients that price preparation and
        postprocess -- which is most of what is left of the step once the body
        and the head are accounted for."""
        _without_region_declaration(cell)

        assert run(cell) == 1
        assert any("registry does not declare" in f for f in _failures(cell))

    def test_a_cell_that_records_no_preset_is_refused(self, cell):
        """A run that cannot say which coefficients it used has not said what
        it predicted, whatever the option string names."""
        _rank(cell, regions=base.KEEP)

        assert run(cell) == 1
        assert any("does not record which region preset" in f
                   for f in _failures(cell))

    def test_a_preset_recorded_without_a_digest_is_refused(self, cell):
        """A name again, wearing the shape of a snapshot."""
        _rank(cell, regions={"schema": "compass.regions.selected/1",
                             "requested": "source-27b-tp1-conc-v2",
                             "parameters": {"postprocess_decode": 1.0}})

        assert run(cell) == 1
        assert any("no digest over its own values" in f
                   for f in _failures(cell))
