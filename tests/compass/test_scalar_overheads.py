"""Timing scalars a run carries as numbers, and what they are attributed to.

Calibration provenance is checked by digest. A number has no digest, so
`seconds_per_launch` and `admission_seconds` -- both measured, both added
straight into a predicted duration -- passed every check there was by not
being files. A scalar fitted to the target engine at the width being predicted
is the residual between the prediction and the run it is predicting, and it
reached a verdict with nothing in the record even naming it.

These drive the real validator over the real passing cell, changing only what
the defect is about.
"""

import json

from . import test_cc_traces_validate as base

# Aliased off the module rather than imported by name: `cell` is a fixture, and
# importing a fixture into a module that also names it as a test argument reads
# to a linter as a redefinition in every test that uses it.
cell = base.cell
run = base.run
verdict = base.verdict
validate = base.validate
_write = base._write
CODE_SHA = base.CODE_SHA
SWEEP_SHA = base.SWEEP_SHA

#: A measured per-launch overhead, as a fitted one actually looks.
LAUNCH_SECONDS = 0.0012
ADMISSION_SECONDS = 0.014


def _constant(value, option="seconds_per_launch", **overrides):
    """An `overhead_constant` declaration, in the shape the registry takes."""
    entry = {
        "sha256": "e" * 64,
        "kind": "overhead_constant",
        "option": option,
        "value": value,
        "measured_at_tp": 1,
        "produced_by": "microbench.py",
        "workload_sha256": None,
        "sources": [{"path": "/m/launch_sweep.json", "sha256": SWEEP_SHA}],
        "code": {"scripts/compass/microbench.py": CODE_SHA},
    }
    entry.update(overrides)
    return entry


def _with_scalar(cell_dir, name, value, declare=None):
    """Set one scalar on the modelled side, and say what the registry holds."""
    path = cell_dir / "modelled.r1.json"
    blob = json.loads(path.read_text())
    compass = blob["run"]["server"]["compass"]
    if name == "admission_seconds":
        compass["admission_seconds"] = value
    else:
        compass["oracle_options"][name] = value
    _write(path, blob)

    registry = json.loads((cell_dir / "registry.json").read_text())
    if declare is not None:
        registry["artifacts"].append(declare)
    (cell_dir / "registry.json").write_text(json.dumps(registry))


def _failures(cell_dir):
    return verdict(cell_dir)["failures"]


class TestZeroIsExempt:
    """The documented default of both, and it means "no such term".

    There is no measurement behind a zero, so there is nothing to attribute,
    and demanding a declaration for the absence of a term would refuse every
    run that never used one.
    """

    def test_the_passing_cell_still_passes(self, cell):
        assert run(cell) == 0

    def test_a_zero_launch_overhead_needs_no_declaration(self, cell):
        _with_scalar(cell, "seconds_per_launch", 0.0)
        assert run(cell) == 0

    def test_a_zero_admission_time_needs_no_declaration(self, cell):
        _with_scalar(cell, "admission_seconds", 0.0)
        assert run(cell) == 0


class TestANonzeroScalarMustBeDeclared:

    def test_an_undeclared_launch_overhead_is_refused(self, cell):
        """The defect: this number went straight into every predicted step and
        the registry did not have to mention it."""
        _with_scalar(cell, "seconds_per_launch", LAUNCH_SECONDS)

        assert run(cell) == 1
        assert any("naming that option at that value" in f
                   for f in _failures(cell))

    def test_an_undeclared_admission_time_is_refused(self, cell):
        _with_scalar(cell, "admission_seconds", ADMISSION_SECONDS)

        assert run(cell) == 1
        assert any("naming that option at that value" in f
                   for f in _failures(cell))

    def test_a_declaration_of_that_exact_value_is_accepted(self, cell):
        _with_scalar(cell, "seconds_per_launch", LAUNCH_SECONDS,
                     declare=_constant(LAUNCH_SECONDS))

        assert run(cell) == 0

    def test_a_declaration_of_a_nearby_value_is_not_that_value(self, cell):
        """Exact, not approximate. A constant that has to be matched to within
        something is a constant nobody can check, and the something is then
        where the fitting hides."""
        _with_scalar(cell, "seconds_per_launch", LAUNCH_SECONDS,
                     declare=_constant(0.0013))

        assert run(cell) == 1
        assert any("naming that option at that value" in f
                   for f in _failures(cell))

    def test_a_declaration_naming_another_option_does_not_answer_for_this_one(
            self, cell):
        """Same number, different quantity. A per-launch overhead and an
        admission time are measurements of different things in different
        units, and one is not evidence for the other."""
        _with_scalar(cell, "seconds_per_launch", LAUNCH_SECONDS,
                     declare=_constant(LAUNCH_SECONDS,
                                       option="admission_seconds"))

        assert run(cell) == 1
        assert any("naming that option at that value" in f
                   for f in _failures(cell))

    def test_an_anonymous_constant_of_the_same_value_does_not_bind(self, cell):
        """The value alone attributes the overhead to whatever was nearest."""
        _with_scalar(cell, "seconds_per_launch", LAUNCH_SECONDS,
                     declare=_constant(LAUNCH_SECONDS, option=None))

        assert run(cell) == 1
        assert any("naming that option at that value" in f
                   for f in _failures(cell))


class TestATargetFittedScalarIsRefused:
    """The reason the check exists.

    A scalar measured on the target engine at the width being predicted is the
    residual between the prediction and the run it is predicting. Added to the
    prediction it makes the two agree by construction, and every tolerance in
    the protocol is then met by arithmetic rather than by modelling.
    """

    def test_one_declared_as_coming_from_the_target_engine_is_refused(self, cell):
        _with_scalar(cell, "seconds_per_launch", LAUNCH_SECONDS,
                     declare=_constant(LAUNCH_SECONDS, from_target_engine=True))

        assert run(cell) == 1
        assert any("residual" in f for f in _failures(cell))

    def test_one_measured_at_the_predicted_width_is_refused(self, cell):
        """`CALIBRATION_KINDS` already says an overhead constant may only be a
        source at TP=1. A number carried no digest, so nothing applied it."""
        _with_scalar(cell, "seconds_per_launch", LAUNCH_SECONDS,
                     declare=_constant(LAUNCH_SECONDS, measured_at_tp=2))

        assert run(cell) == 1
        assert any("width being predicted" in f for f in _failures(cell))

    def test_one_fitted_to_the_acceptance_workload_is_refused(self, cell):
        workload_sha = validate._digest(validate.WORKLOADS["long"])
        _with_scalar(cell, "seconds_per_launch", LAUNCH_SECONDS,
                     declare=_constant(LAUNCH_SECONDS,
                                       workload_sha256=workload_sha))

        assert run(cell) == 1
        assert any("fitted to the run it is predicting" in f
                   for f in _failures(cell))


class TestTheDeclarationIsHeldToTheSameStandard:
    """An overhead constant is a measured input, so it is read like one."""

    def test_one_naming_no_sources_is_refused(self, cell):
        _with_scalar(cell, "seconds_per_launch", LAUNCH_SECONDS,
                     declare=_constant(LAUNCH_SECONDS, sources=[]))

        assert run(cell) == 1
        assert any("declares no sources" in f for f in _failures(cell))

    def test_one_pinning_no_code_is_refused(self, cell):
        _with_scalar(cell, "seconds_per_launch", LAUNCH_SECONDS,
                     declare=_constant(LAUNCH_SECONDS, code={}))

        assert run(cell) == 1
        assert any("no code digests" in f for f in _failures(cell))

    def test_a_scalar_that_is_not_a_number_is_refused(self, cell):
        _with_scalar(cell, "seconds_per_launch", "soon")

        assert run(cell) == 1
        assert any("is not a number" in f for f in _failures(cell))
