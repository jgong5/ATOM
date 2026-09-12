"""The capacity side of the record, which is not read when the oracle is.

`_init_compass_state` builds the oracle, and the oracle reads its tables there.
The memory profile and the replay target are not read there: `get_num_blocks`
opens them, and it runs afterwards. So a manifest frozen at init records "this
run read no capacity input", which is a finding about a race rather than about
the run -- and it is the wrong finding, because the run does read one.

So the oracle's inputs are frozen at init and the capacity inputs are collected
at readback. Nothing is reopened either way: every record was taken by the
reader that parsed the bytes.
"""

import types

from atom.compass.config import CompassConfig
from atom.compass.core.loaded_input import LoadedInput
from atom.compass.runtime.predict import CompassPredictMixin

ORACLE_INPUT = LoadedInput(role="oracle.price", requested="prices.json",
                           path="prices.tp0.json", rank_own=True,
                           sha256="a" * 64, size=11)
PROFILE_INPUT = LoadedInput(role="runtime.memory_model",
                            requested="profile.json", path="profile.json",
                            rank_own=False, sha256="b" * 64, size=22)
TARGET_INPUT = LoadedInput(role="runtime.replay_target",
                           requested="target.json", path="target.json",
                           rank_own=False, sha256="c" * 64, size=33)


class _Oracle:
    compass_loaded_inputs = (ORACLE_INPUT,)

    def describe(self):
        return "stub"


def _runner(oracle=None):
    """The mixin with its state initialised, as both real runners inherit it."""
    stub = CompassPredictMixin.__new__(CompassPredictMixin)
    stub.__dict__["_compass_config_cache"] = CompassConfig(enabled=True)
    stub.config = types.SimpleNamespace()
    stub._build_oracle = lambda config: oracle or _Oracle()
    stub._topology = lambda: {"tp": 1}
    stub._rank_coords = dict
    stub._warn_if_compiled = lambda: None
    stub._init_compass_state()
    return stub


def _roles(manifest):
    return {row["role"] for row in manifest["inputs"]}


class TestTheOracleSideIsFrozenAtInit:

    def test_what_the_oracle_read_is_in_the_manifest(self):
        runner = _runner()

        assert _roles(runner.compass_input_manifest()) == {"oracle.price"}

    def test_it_does_not_move_when_the_oracle_is_replaced_later(self):
        """Frozen means frozen: the record describes the build that read."""
        runner = _runner()
        runner._oracle = types.SimpleNamespace(compass_loaded_inputs=())

        assert _roles(runner.compass_input_manifest()) == {"oracle.price"}


class TestTheCapacitySideIsCollectedAtReadback:

    def test_a_profile_read_after_init_is_in_the_manifest(self):
        """The defect a frozen manifest would have had. `get_num_blocks` runs
        after `_init_compass_state`, so this is the normal order, not a late
        edge case."""
        runner = _runner()
        assert _roles(runner.compass_input_manifest()) == {"oracle.price"}

        runner.compass_runtime_inputs = (PROFILE_INPUT,)

        assert _roles(runner.compass_input_manifest()) == {
            "oracle.price", "runtime.memory_model"}

    def test_several_capacity_reads_are_all_kept(self):
        runner = _runner()
        runner.compass_runtime_inputs = (PROFILE_INPUT, TARGET_INPUT)

        assert _roles(runner.compass_input_manifest()) == {
            "oracle.price", "runtime.memory_model", "runtime.replay_target"}

    def test_the_two_namespaces_stay_apart(self):
        """An `oracle.replay_target` and a `runtime.replay_target` are read by
        different code for different purposes, and neither answers for the
        other."""
        runner = _runner()
        runner.compass_runtime_inputs = (TARGET_INPUT,)

        rows = {row["role"]: row for row in
                runner.compass_input_manifest()["inputs"]}
        assert rows["runtime.replay_target"]["sha256"] == "c" * 64
        assert "oracle.replay_target" not in rows

    def test_nothing_is_reopened_to_answer(self):
        """The records name paths that do not exist. If the manifest were
        re-derived from them rather than retained, this would raise."""
        runner = _runner()
        runner.compass_runtime_inputs = (PROFILE_INPUT,)

        rows = {row["role"]: row for row in
                runner.compass_input_manifest()["inputs"]}
        assert rows["runtime.memory_model"]["sha256"] == "b" * 64


class TestTheBudgetSource:

    def test_it_is_unknown_until_the_selector_runs(self):
        """None is a real state and is reported as one. A run whose capacity
        source is unknown is not a run whose capacity was measured."""
        runner = _runner()

        assert runner.compass_input_manifest()["budget_source"] is None

    def test_it_is_reported_as_the_selector_states_it(self):
        runner = _runner()
        runner.compass_budget_source = "analytical"

        assert runner.compass_input_manifest()["budget_source"] == "analytical"

    def test_it_is_not_inferred_from_the_cost_mode(self):
        """`mode="measure"` forces the wall clock and says nothing about
        memory, so a measured run can be sized from an analytical profile.
        Reading the mode here would report that run as measured."""
        runner = _runner()
        runner.__dict__["_compass_config_cache"] = CompassConfig(
            enabled=True, mode="measure", measure_out="/tmp/steps.jsonl")
        runner.compass_budget_source = "analytical"

        assert runner.compass_input_manifest()["budget_source"] == "analytical"


class TestTheSeamsAreThereForTheReaderToFill:

    def test_a_runner_that_never_sets_them_still_answers(self):
        runner = _runner()

        out = runner.compass_input_manifest()
        assert out["inputs"] and out["budget_source"] is None

    def test_the_names_the_capacity_reader_writes_to(self):
        """Pinned, because two owners write and read these."""
        runner = _runner()

        assert isinstance(runner.compass_runtime_inputs, tuple)
        assert hasattr(runner, "compass_budget_source")
