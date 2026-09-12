"""What sized the deployment, and whether the reference side was a machine.

Two separate claims that were both unrecorded.

A prediction is a prediction of a deployment that could exist. The KV budget
decides how many requests fit, which decides the schedule, which is most of
what the numbers are -- and a run could reach a verdict without saying what
that budget was made from.

And the ground-truth side could be sized analytically. `get_num_blocks` reaches
the analytical path without consulting `compass.mode`; `CompassConfig` forces
the wall clock for `mode="measure"` and says nothing about memory. So a
"measured" run could be sized from a modelled profile, the comparison would be
between two models rather than between a model and a machine, and every check
there was looked at the clock.

These drive the real validator over the real passing cell.
"""

import json

from . import test_cc_traces_validate as base

cell = base.cell
run = base.run
verdict = base.verdict
validate = base.validate
_write = base._write

CAPTURED = {"kind": "captured", "hardware_reference": "MI308X",
            "served": {"num_kvcache_blocks": 4096},
            "lineage": ["/x/target.json"]}
MEASURED = dict(CAPTURED, kind="device-measured")


def _rank(cell_dir, side, **changes):
    """Change one rank's capacity record on one side of the comparison."""
    path = cell_dir / f"{side}.r1.json"
    blob = json.loads(path.read_text())
    rank = blob["run"]["server"]["compass"]["loaded_inputs"]["ranks"][0]
    for key, value in changes.items():
        if value is base.KEEP:
            rank.pop(key, None)
        else:
            rank[key] = value
    _write(path, blob)


def _failures(cell_dir):
    return verdict(cell_dir)["failures"]


def _input(role):
    return {"role": role, "requested": "/x/f.json", "path": "/x/f.json",
            "rank_own": False, "sha256": "f" * 64, "size": 1,
            "rank_coords": {}}


class TestThePredictorMustSayWhatSizedIt:

    def test_the_passing_cell_still_passes(self, cell):
        assert run(cell) == 0

    def test_a_run_that_read_no_capacity_input_is_refused(self, cell):
        """It read price lists and templates, which build the cost oracle and
        size nothing."""
        _rank(cell, "modelled", inputs=[_input("oracle.price"),
                                        _input("oracle.template")])

        assert run(cell) == 1
        assert any("read no capacity input" in f for f in _failures(cell))

    def test_the_oracles_own_replay_target_does_not_count(self, cell):
        """`oracle.replay_target` is read by the source factory to answer an
        architecture query. It is a different file read for a different
        purpose, and it sizes nothing."""
        _rank(cell, "modelled", inputs=[_input("oracle.replay_target")])

        assert run(cell) == 1
        assert any("read no capacity input" in f for f in _failures(cell))

    def test_any_runtime_role_satisfies_it(self, cell):
        _rank(cell, "modelled", inputs=[_input("runtime.memory_model")])

        assert run(cell) == 0

    def test_a_nested_runtime_role_satisfies_it(self, cell):
        """Roles are a namespace, not an enum: a profile that names further
        files gives them nested roles, and those are still capacity inputs."""
        _rank(cell, "modelled",
              inputs=[_input("runtime.memory_model.collective")])

        assert run(cell) == 0

    def test_a_run_that_names_no_budget_source_is_refused(self, cell):
        _rank(cell, "modelled", budget_source=None)

        assert run(cell) == 1
        assert any("does not say which reading its KV budget" in f
                   for f in _failures(cell))

    def test_an_unrecognised_budget_source_is_refused(self, cell):
        _rank(cell, "modelled", budget_source={"kind": "vibes"})

        assert run(cell) == 1
        assert any("not one the protocol recognises" in f
                   for f in _failures(cell))

    def test_a_modelled_run_sized_without_a_device_is_fine(self, cell):
        """The capability, not a defect. A modelled run is supposed to be
        sized without a device; this is a guard on an acceptance role, not a
        ban on analytical capacity."""
        _rank(cell, "modelled", budget_source={"kind": "source-derived"})

        assert run(cell) == 0


class TestTheReferenceSideMustHaveBeenAMachine:

    def test_a_reference_sized_analytically_is_refused(self, cell):
        """The defect: mode=measure forces the wall clock and says nothing
        about memory, so this run looked measured in every field there was."""
        _rank(cell, "real", budget_source={"kind": "source-derived"})

        assert run(cell) == 1
        assert any("the ground-truth side is itself a prediction" in f
                   for f in _failures(cell))

    def test_a_reference_sized_from_a_capture_is_refused(self, cell):
        """A capture of a device is not the device this run ran on."""
        _rank(cell, "real", budget_source=CAPTURED)

        assert run(cell) == 1
        assert any("the ground-truth side is itself a prediction" in f
                   for f in _failures(cell))

    def test_a_reference_that_says_nothing_is_refused(self, cell):
        _rank(cell, "real", budget_source=None)

        assert run(cell) == 1
        assert any("silence is not that" in f for f in _failures(cell))

    def test_a_device_measured_reference_passes(self, cell):
        _rank(cell, "real", budget_source=MEASURED)

        assert run(cell) == 0


class TestTheSelectorsOwnRecordIsCarriedWhole:
    """The kind is what this branches on; the rest is why anyone believes it.

    A bare word would have been enough for the guard and useless for a reader,
    so the record is carried as the selector publishes it and read for the one
    field the verdict needs.
    """

    def test_a_plain_string_is_read_as_the_kind_it_names(self, cell):
        _rank(cell, "real", budget_source="device-measured")

        assert run(cell) == 0

    def test_the_lineage_survives_into_the_artifact(self, cell):
        run(cell)
        blob = json.loads((cell / "real.r1.json").read_text())
        rank = blob["run"]["server"]["compass"]["loaded_inputs"]["ranks"][0]

        assert rank["budget_source"]["lineage"]
        assert rank["budget_source"]["hardware_reference"]
