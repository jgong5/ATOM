"""The region preset a run priced with, and what attributes it.

A region model supplies preparation and postprocess -- everything in the step
that is not the body and not the head -- from measured coefficients, and those
go straight into every predicted duration. Nothing checked them. They are not
a file, so the calibration walk over `oracle_option_sha256` never saw them;
they are not a scalar option, so the overhead check did not; and
`regions=source-27b-tp1-conc-v2` is a *name*. A name is not a measurement: the
preset behind it can be edited, and two runs quoting the same name can have
been priced from different numbers with nothing in either record to show it.

These drive the real built-in presets through the real factory and the real
validator. The snapshot is a value, not a file read, and stays distinct from
`LoadedInput` throughout.
"""

import copy
import json

import pytest

from atom.compass.runtime.source_oracle import (
    REGION_SNAPSHOT_SCHEMA,
    region_snapshot,
    region_values,
    source_cost_oracle,
)

from . import test_cc_traces_validate as base
from .test_source_oracle import _price_file, _template_file

validate = base.validate
cell = base.cell
run = base.run
verdict = base.verdict

PRESET = base.CELL_REGIONS


def _snap(name):
    """The snapshot of the object `region_model` resolves, as the factory
    takes it: off the selected preset, not off the name a second time."""
    from atom.compass.core.cost.regions import region_model

    return region_snapshot(name, region_model(name))


def _failures(cell_dir):
    return verdict(cell_dir)["failures"]


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


def _registry(cell_dir, **overrides):
    path = cell_dir / "registry.json"
    blob = json.loads(path.read_text())
    for entry in blob["artifacts"]:
        if entry.get("kind") == "region_model":
            entry.update(overrides)
    path.write_text(json.dumps(blob))


class TestTheSnapshotIsTheValuesAndNotTheName:

    def test_it_carries_the_name_that_was_asked_for(self):
        snapshot = _snap(PRESET)

        assert snapshot["schema"] == REGION_SNAPSHOT_SCHEMA
        assert snapshot["requested"] == PRESET
        assert PRESET in snapshot["aliases"]

    def test_it_carries_the_presets_own_version_and_provenance(self):
        """Read off the dataclass, so a preset that states them states them
        here without this having to know which fields it has."""
        snapshot = _snap(PRESET)

        assert snapshot["version"]
        assert snapshot["provenance"]

    def test_it_carries_the_coefficients(self):
        snapshot = _snap(PRESET)

        assert snapshot["parameters"]["postprocess_decode"]["seconds"] > 0
        assert snapshot["parameters"]["prepare_decode_cells"]
        assert snapshot["parameters"]["topologies"]

    def test_none_selects_nothing_and_snapshots_as_such(self):
        """A real choice -- body plus head with no runner term -- and one that
        contributes no coefficients, so there is nothing to attribute."""
        snapshot = _snap("none")

        assert snapshot["parameters"] is None
        assert snapshot["sha256"]

    def test_an_unknown_name_is_refused_rather_than_snapshotted(self):
        with pytest.raises(ValueError, match="unknown region model"):
            _snap("source-27b-tp9")

    def test_every_built_in_preset_snapshots(self):
        """Including any added later: the snapshot reads the dataclass rather
        than a list kept here, so a new preset needs no change to consume."""
        from atom.compass.core.cost.regions import REGION_MODELS

        for name in REGION_MODELS:
            assert _snap(name)["sha256"]


class TestADigestOverEveryCoefficient:

    def test_two_reads_of_one_preset_agree(self):
        assert _snap(PRESET)["sha256"] == _snap(PRESET)["sha256"]

    def test_two_different_presets_do_not(self):
        assert (_snap("source-27b-tp1")["sha256"]
                != _snap("source-27b-tp1-conc")["sha256"])

    def test_a_changed_nested_coefficient_changes_the_digest(self, monkeypatch):
        """The case the whole thing exists for. Editing one measured second
        inside the preset must not leave the digest where it was, or a run
        priced from the new number is attributed to the old declaration."""
        import dataclasses

        from atom.compass.core.cost import regions

        before = _snap(PRESET)["sha256"]
        model = regions.REGION_MODELS[PRESET]
        moved = dataclasses.replace(
            model,
            postprocess_decode=dataclasses.replace(
                model.postprocess_decode,
                seconds=model.postprocess_decode.seconds * 1.05))
        monkeypatch.setitem(regions.REGION_MODELS, PRESET, moved)

        assert _snap(PRESET)["sha256"] != before

    def test_a_changed_cell_inside_a_table_changes_the_digest(self, monkeypatch):
        """Not only the top-level fields. The concurrency presets carry a
        table of per-rung measurements, and a coefficient buried in one of
        them prices real steps."""
        import dataclasses

        from atom.compass.core.cost import regions

        before = _snap(PRESET)["sha256"]
        model = regions.REGION_MODELS[PRESET]
        cells = list(model.prepare_decode_cells)
        key, measured = cells[0]
        cells[0] = (key, dataclasses.replace(measured,
                                             seconds=measured.seconds * 1.01))
        monkeypatch.setitem(
            regions.REGION_MODELS, PRESET,
            dataclasses.replace(model, prepare_decode_cells=tuple(cells)))

        assert _snap(PRESET)["sha256"] != before

    def test_a_narrowed_domain_changes_the_digest(self, monkeypatch):
        """The domain is part of what was selected: the same numbers over a
        different domain is a different claim about where they hold."""
        import dataclasses

        from atom.compass.core.cost import regions

        before = _snap(PRESET)["sha256"]
        model = regions.REGION_MODELS[PRESET]
        monkeypatch.setitem(
            regions.REGION_MODELS, PRESET,
            dataclasses.replace(model, topologies=(1,)))

        assert _snap(PRESET)["sha256"] != before

    def test_a_tuple_keyed_table_is_kept_rather_than_coerced(self):
        """The serialiser the snapshot digests through, on a shape no current
        preset uses but one is free to.

        `json.dumps` cannot write a tuple key: it raises, or a careless
        serialiser flattens it to a string. Either loses the coefficient, and
        a coefficient outside the digest is a number nobody is held to.
        """
        from atom.compass.core.cost.regions import REGION_MODELS

        model = REGION_MODELS[PRESET]
        table = {(2, True): model.postprocess_prefill,
                 (1, False): model.postprocess_decode}

        out = region_values({"cells": table})

        pairs = out["cells"]["__pairs__"]
        assert [pair[0] for pair in pairs] == [[1, False], [2, True]], (
            "sorted, so one table does not digest two ways")
        assert pairs[0][1]["seconds"] == model.postprocess_decode.seconds
        assert pairs[1][1]["seconds"] == model.postprocess_prefill.seconds

    def test_a_tuple_keyed_coefficient_reaches_the_digest(self):
        """Kept is not enough; it has to be *held*. Moving a value behind a
        tuple key must move the digest."""
        import hashlib
        import json as _json

        from atom.compass.core.cost.regions import REGION_MODELS

        model = REGION_MODELS[PRESET]

        def digest(seconds):
            import dataclasses

            table = {(1, False): dataclasses.replace(
                model.postprocess_decode, seconds=seconds)}
            body = _json.dumps(region_values({"cells": table}),
                               sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(body.encode()).hexdigest()

        assert digest(1.0e-4) != digest(1.05e-4)


class TestTheValidatorReadsTheSnapshot:
    """The cases that need the snapshot helper to set up. The ones that do
    not live in `test_region_declaration_required`, which is importable
    against a tree without it and is therefore the regression."""


    def test_the_passing_cell_passes(self, cell):
        assert run(cell) == 0


    def test_a_changed_coefficient_stops_matching_the_declaration(self, cell):
        """The registry still declares the preset it was told about; the run
        priced from different numbers, so the digest no longer meets it."""
        moved = copy.deepcopy(base.region_snapshot_of(PRESET))
        moved["parameters"]["postprocess_decode"]["seconds"] *= 1.2
        moved["sha256"] = "7" * 64
        _rank(cell, regions=moved)

        assert run(cell) == 1
        assert any("either the preset is unregistered, or its numbers have "
                   "moved" in f for f in _failures(cell))


    def test_a_preset_declared_as_something_else_is_refused(self, cell):
        _registry(cell, kind="source_calibration")

        assert run(cell) == 1
        assert any("a region preset is a region_model" in f
                   for f in _failures(cell))

    def test_a_preset_measured_at_the_predicted_width_is_refused(self, cell):
        _registry(cell, measured_at_tp=2)

        assert run(cell) == 1
        assert any("width being predicted" in f for f in _failures(cell))

    def test_a_preset_fitted_on_the_target_engine_is_refused(self, cell):
        _registry(cell, from_target_engine=True)

        assert run(cell) == 1
        assert any("fitted on the engine being predicted" in f
                   for f in _failures(cell))

    def test_a_declaration_claiming_files_is_refused(self, cell):
        """A value snapshot is not a file read. An entry enumerating contents
        for one describes bytes that never existed."""
        _registry(cell, contents={"regions.py": "c" * 64})

        assert run(cell) == 1
        assert any("did not report reading any file" in f
                   for f in _failures(cell))

    def test_a_preset_naming_no_sources_is_refused(self, cell):
        _registry(cell, sources=[])

        assert run(cell) == 1
        assert any("declares no sources" in f for f in _failures(cell))


class TestTheServedPathCarriesIt:

    def test_the_factory_attaches_the_snapshot_to_the_oracle(self, tmp_path):
        """`source_cost_oracle` returns the oracle alone, so the snapshot has
        to ride on it or a served run never reports one."""
        oracle = source_cost_oracle(
            price=_price_file(tmp_path), template=_template_file(tmp_path),
            derive=0, require_complete=0, regions=PRESET)

        assert oracle.compass_region_snapshot["requested"] == PRESET
        assert oracle.compass_region_snapshot["sha256"]

    def test_the_manifest_keeps_it_out_of_the_file_inputs(self, tmp_path):
        """Distinct from `LoadedInput` and filed separately: nothing here was
        read off disk, and a reader must not go looking for bytes."""
        import types

        from atom.compass.config import CompassConfig
        from atom.compass.runtime.predict import CompassPredictMixin

        oracle = source_cost_oracle(
            price=_price_file(tmp_path), template=_template_file(tmp_path),
            derive=0, require_complete=0, regions=PRESET)
        stub = CompassPredictMixin.__new__(CompassPredictMixin)
        stub.__dict__["_compass_config_cache"] = CompassConfig(enabled=True)
        stub.config = types.SimpleNamespace()
        stub._build_oracle = lambda config: oracle
        stub._topology = lambda: {"tp": 1}
        stub._rank_coords = dict
        stub._warn_if_compiled = lambda: None
        stub._init_compass_state()

        manifest = stub.compass_input_manifest()

        assert manifest["regions"]["requested"] == PRESET
        assert all(not row["role"].startswith("regions")
                   for row in manifest["inputs"])
        assert all("region" not in row["role"] for row in manifest["inputs"])
