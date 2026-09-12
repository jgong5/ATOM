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

CAPTURED = {"kind": "captured", "served": True,
            "hardware_reference": "MI308X", "lineage": ["/x/target.json"],
            "deployment": {"num_kvcache_blocks": 4096}}
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


def _input(role, sha=None):
    """One loaded-input row, defaulting to the declared capacity artifact."""
    return {"role": role, "requested": "/x/f.json", "path": "/x/f.json",
            "rank_own": False, "sha256": sha or base.TARGET_SHA, "size": 1,
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

    def test_the_kind_must_name_a_file_the_run_actually_opened(self, cell):
        """Any `runtime.*` row is not enough. A kind is a claim about where a
        number came from, and `captured` means a replay target was read -- a
        run that opened a memory profile instead has not described itself,
        whatever its record says."""
        _rank(cell, "modelled", inputs=[_input("runtime.memory_model")])

        assert run(cell) == 1
        assert any("the kind names a file and this run did not open it" in f
                   for f in _failures(cell))

    def test_a_source_derived_budget_must_have_read_a_profile(self, cell):
        _rank(cell, "modelled",
              budget_source=dict(CAPTURED, kind="source-derived"),
              inputs=[_input("runtime.memory_model", sha=base.TARGET_SHA)])

        assert run(cell) == 0

    def test_a_nested_runtime_role_is_a_capacity_input(self, cell):
        """Roles are a namespace, not an enum: a profile that names further
        files gives them nested roles, and those are still capacity inputs and
        still go through the registry."""
        _rank(cell, "modelled",
              inputs=[_input("runtime.replay_target", sha=base.TARGET_SHA),
                      _input("runtime.memory_model.collective",
                             sha=base.TARGET_SHA)])

        assert run(cell) == 0

    def test_a_run_that_names_no_budget_source_is_refused(self, cell):
        _rank(cell, "modelled", budget_source=None)

        assert run(cell) == 1
        assert any("does not say which reading its KV budget" in f
                   for f in _failures(cell))

    def test_an_unrecognised_budget_source_is_refused(self, cell):
        _rank(cell, "modelled", budget_source=dict(CAPTURED, kind="vibes"))

        assert run(cell) == 1
        assert any("not one the protocol recognises" in f
                   for f in _failures(cell))

    def test_a_modelled_run_sized_without_a_device_is_fine(self, cell):
        """The capability, not a defect. A modelled run is supposed to be
        sized without a device; this is a guard on an acceptance role, not a
        ban on analytical capacity."""
        _rank(cell, "modelled",
              budget_source=dict(CAPTURED, kind="source-derived"),
              inputs=[_input("runtime.memory_model")])

        assert run(cell) == 0


class TestACapacityInputLeaksLikeAPricedOne:
    """Every one of these leaves the oracle's own inputs immaculate.

    `check_calibration` walks `oracle_option_sha256`, which holds the files the
    command line named -- the price lists and the templates. A memory profile
    and a replay target are not oracle options; they reach the record through
    the loaded-input manifest, so nothing ever asked the registry about them.
    A run could be sized from a profile measured on the target engine, at the
    width being predicted, with every priced operator above suspicion. The KV
    budget decides how many requests fit, which decides the schedule, which is
    most of what the numbers are.
    """

    def _registry(self, cell_dir, **overrides):
        path = cell_dir / "registry.json"
        blob = json.loads(path.read_text())
        for entry in blob["artifacts"]:
            if entry.get("sha256") == base.TARGET_SHA:
                entry.update(overrides)
        path.write_text(json.dumps(blob))

    def test_an_unregistered_capacity_input_is_refused(self, cell):
        _rank(cell, "modelled",
              inputs=[_input("runtime.replay_target", sha="9" * 64)])

        assert run(cell) == 1
        assert any("not declared in the calibration registry" in f
                   for f in _failures(cell))
        assert run(cell) == 1

    def test_a_capacity_input_from_the_target_engine_is_refused(self, cell):
        """The central case. Every oracle input is valid and the deployment
        was sized by the engine whose capacity is being predicted."""
        self._registry(cell, from_target_engine=True)

        assert run(cell) == 1
        assert any("sized by the engine whose capacity is being predicted" in f
                   for f in _failures(cell))

    def test_a_capacity_input_measured_at_the_predicted_width_is_refused(
            self, cell):
        self._registry(cell, measured_at_tp=2)

        assert run(cell) == 1
        assert any("width being predicted" in f for f in _failures(cell))

    def test_a_capacity_input_fitted_to_the_acceptance_workload_is_refused(
            self, cell):
        self._registry(
            cell, workload_sha256=validate._digest(validate.WORKLOADS["long"]))

        assert run(cell) == 1
        assert any("produced from the acceptance workload" in f
                   for f in _failures(cell))

    def test_a_capacity_input_naming_no_sources_is_refused(self, cell):
        self._registry(cell, sources=[])

        assert run(cell) == 1
        assert any("declares no sources" in f for f in _failures(cell))

    def test_a_capacity_input_pinning_no_code_is_refused(self, cell):
        self._registry(cell, code={})

        assert run(cell) == 1
        assert any("no code digests" in f for f in _failures(cell))

    def test_a_capacity_input_of_an_unrecognised_kind_is_refused(self, cell):
        self._registry(cell, kind="vibes")

        assert run(cell) == 1
        assert any("not one the protocol recognises" in f
                   for f in _failures(cell))

    def test_a_capacity_input_with_no_digest_is_refused(self, cell):
        row = _input("runtime.replay_target")
        row["sha256"] = ""
        _rank(cell, "modelled", inputs=[row])

        assert run(cell) == 1
        assert any("names a file rather than a file's contents" in f
                   for f in _failures(cell))


class TestTheReferenceSideMustHaveBeenAMachine:

    def test_a_reference_sized_analytically_is_refused(self, cell):
        """The defect: mode=measure forces the wall clock and says nothing
        about memory, so this run looked measured in every field there was."""
        _rank(cell, "real",
              budget_source=dict(CAPTURED, kind="source-derived"))

        assert run(cell) == 1
        assert any("the ground-truth side is itself a prediction" in f
                   for f in _failures(cell))

    def test_a_measured_budget_that_was_not_served_is_refused(self, cell):
        """`kind` and `served` are different claims. Where the reading came
        from is not the same as what the engine ran on, and a budget computed
        from the device and then not used describes some other deployment."""
        _rank(cell, "real", budget_source=dict(MEASURED, served=False))

        assert run(cell) == 1
        assert any("does not say it served" in f for f in _failures(cell))

    def test_a_measured_budget_that_is_silent_about_serving_is_refused(
            self, cell):
        record = dict(MEASURED)
        record.pop("served")
        _rank(cell, "real", budget_source=record)

        assert run(cell) == 1
        assert any("does not say it served" in f for f in _failures(cell))

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
    """The kind is what the verdict branches on; the rest is why anyone
    believes it, and it is carried as the selector publishes it.
    """

    def test_a_bare_word_is_not_a_record(self, cell):
        """A kind with nothing behind it names a category and evidences
        nothing. It is the shape this replaced, not a shorter spelling."""
        _rank(cell, "real", budget_source="device-measured")

        assert run(cell) == 1
        assert any("does not say which reading its KV budget" in f
                   for f in _failures(cell))

    def test_the_lineage_survives_into_the_artifact(self, cell):
        run(cell)
        blob = json.loads((cell / "real.r1.json").read_text())
        rank = blob["run"]["server"]["compass"]["loaded_inputs"]["ranks"][0]

        assert rank["budget_source"]["lineage"]
        assert rank["budget_source"]["hardware_reference"]
