"""What the acceptance plan must say, so a leased hour is not spent finding out.

The plan runs nothing, so there is no result to check. What there is to check is
that the sequence it prints is the protocol's: the real side paced and prepared
on the leased node, the modelled side declared and unprepared in a container
with no device, the device-free probe taken where the prediction happened, and
the validator last.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan_mod = _load("cc_traces_plan")
validate = _load("cc_traces_validate")


@pytest.fixture
def plan():
    return json.loads(_run(["--root", "/r"]))


def _run(argv, capsys=None):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert plan_mod.main(argv) == 0
    return buffer.getvalue()


def _steps(plan, cell_index=0):
    return plan["cells"][cell_index]["steps"]


def _by_id(cell):
    return {s["id"]: s for s in cell["steps"]}


class TestTheMatrixItCovers:
    def test_every_width_and_class_appears_once(self, plan):
        seen = [(c["tp"], c["class"]) for c in plan["cells"]]
        assert sorted(seen) == sorted(
            (tp, klass) for tp in plan_mod.TPS for klass in plan_mod.CLASSES
        )
        assert len(seen) == len(set(seen))

    def test_each_cell_has_its_own_directory(self, plan):
        dirs = [c["cell"] for c in plan["cells"]]
        assert len(set(dirs)) == len(dirs)
        assert all(d.startswith("/r/") for d in dirs)

    def test_the_matrix_step_names_every_cell(self, plan):
        for cell in plan["cells"]:
            assert cell["cell"] in plan["matrix"]


class TestTheTwoSidesAreNotRunTheSameWay:
    def test_the_real_side_is_paced_and_prepared(self, plan):
        for cell in plan["cells"]:
            for step in cell["steps"]:
                if step["id"].startswith("replay-real"):
                    assert "--pace" in step["command"]
                    assert "--prepare" in step["command"]
                    assert step["where"] == "gpu"

    def test_the_modelled_side_is_neither(self, plan):
        """A paced modelled side answers a different arrival process, and a
        prepared one has been warmed -- both are refusals in the validator."""
        for cell in plan["cells"]:
            for step in cell["steps"]:
                if step["id"].startswith("replay-modelled"):
                    assert "--pace" not in step["command"]
                    assert "--prepare" not in step["command"]
                    assert step["where"] == "device_free"

    def test_the_modelled_side_predicts_and_the_real_side_measures(self, plan):
        steps = _by_id(plan["cells"][0])
        real = steps["serve-real"]["command"]
        modelled = steps["serve-modelled"]["command"]
        assert real[real.index("--compass-mode") + 1] == "measure"
        assert modelled[modelled.index("--compass-mode") + 1] == "predict"
        assert "--compass-measure-out" in real
        assert steps["serve-real"]["where"] == "gpu"
        assert steps["serve-modelled"]["where"] == "device_free"

    def test_both_sides_carry_the_registered_engine_configuration(self, plan):
        for cell in plan["cells"]:
            steps = _by_id(cell)
            for name in ("serve-real", "serve-modelled"):
                command = steps[name]["command"]
                for arg in plan_mod.ENGINE_ARGS:
                    assert arg in command
                assert "--no-enable_prefix_caching" in command
                assert command[command.index("-tp") + 1] == str(cell["tp"])

    def test_each_side_replays_the_registered_workload_for_its_class(self, plan):
        for cell in plan["cells"]:
            expected = f"atom/compass/cc_traces_{cell['class']}.jsonl"
            replays = [s for s in cell["steps"] if s["id"].startswith("replay-")]
            assert replays
            for step in replays:
                assert expected in step["command"]
                assert (ROOT / expected).exists()

    def test_the_repeat_count_is_the_protocol_s_on_both_sides(self, plan):
        for cell in plan["cells"]:
            for side in ("real", "modelled"):
                got = [s for s in cell["steps"] if s["id"].startswith(f"replay-{side}")]
                assert len(got) == plan["repeats"] == plan_mod.REPEATS

    def test_an_oracle_option_reaches_only_the_modelled_side(self):
        got = json.loads(
            _run(["--root", "/r", "--oracle", "O", "--oracle-option", "library=/w/l"])
        )
        steps = _by_id(got["cells"][0])
        assert "--compass-oracle-option" in steps["serve-modelled"]["command"]
        assert "library=/w/l" in steps["serve-modelled"]["command"]
        assert "--compass-oracle-option" not in steps["serve-real"]["command"]


class TestTheOrderIsTheProtocolS:
    def test_the_device_free_probe_happens_before_the_verdict(self, plan):
        for cell in plan["cells"]:
            ids = [s["id"] for s in cell["steps"]]
            assert ids.index("gpu-free") < ids.index("validate")

    def test_the_probe_runs_where_the_prediction_ran(self, plan):
        """Taken in the container that served the modelled side; taken on the
        leased node it would say nothing about that container."""
        for cell in plan["cells"]:
            steps = _by_id(cell)
            assert steps["gpu-free"]["where"] == "device_free"
            assert steps["gpu-free"]["where"] == steps["serve-modelled"]["where"]

    def test_the_isolation_audit_follows_the_measured_window(self, plan):
        ids = [s["id"] for s in plan["cells"][0]["steps"]]
        assert ids.index("sample-devices") < ids.index("replay-real-1")
        assert ids.index("replay-real-1") < ids.index("isolation")

    def test_the_verdict_is_the_last_thing_in_a_cell(self, plan):
        for cell in plan["cells"]:
            assert cell["steps"][-1]["id"] == "validate"
            assert cell["steps"][-1]["where"] == "cpu"

    def test_nothing_that_reads_artifacts_asks_for_a_device(self, plan):
        for cell in plan["cells"]:
            for step in cell["steps"]:
                if step["id"] in ("validate", "isolation", "verify-workload", "stamp"):
                    assert step["where"] == "cpu"


class TestItSaysWhatItDoesNotProvide:
    def test_a_step_with_no_command_says_who_provides_it(self, plan):
        for cell in plan["cells"]:
            for step in cell["steps"]:
                if not step.get("command"):
                    assert step["provided_by"]

    def test_the_gaps_are_carried_in_the_plan_itself(self, plan):
        assert plan["gaps"]
        assert any("gpu.jsonl" in gap for gap in plan["gaps"])
        assert any("costs.json" in gap for gap in plan["gaps"])

    def test_the_evidence_list_matches_what_the_validator_requires(self, plan):
        """A plan that forgets an artifact is a cell refused after the lease."""
        assert validate.GPU_FREE_EVIDENCE in plan["evidence"]
        assert "costs.json" in plan["evidence"]
        assert "isolation.json" in plan["evidence"]
        assert "cc_traces_protocol.json" in plan["evidence"]
        assert "registry.json" in plan["evidence"]

    def test_it_does_not_claim_to_have_run_anything(self, plan):
        assert "running them is not what this does" in plan["means"]


class TestWhatItPrints:
    def test_the_shell_rendering_is_the_same_plan(self, tmp_path):
        text = _run(["--root", "/r", "--shell"])
        assert "cc_traces_validate.py cell /r/tp4_long" in text
        assert "[device_free]" in text and "[gpu]" in text
        assert "what this plan does not provide" in text

    def test_the_json_can_be_written_out(self, tmp_path):
        out = tmp_path / "plan.json"
        assert plan_mod.main(["--root", "/r", "--out", str(out)]) == 0
        assert json.loads(out.read_text())["cells"]
