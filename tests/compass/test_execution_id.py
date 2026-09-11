"""The one identity rule, checked the way another reader would use it.

`execution_id.py` exists so that the harness that mints an id and whatever
later reads the artifact -- a classifier, a notebook, another repository --
agree by importing the same code rather than by reimplementing the same rule.
These tests are written from the reader's side: given a file, can its execution
be established, and can a lie about it be caught.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eid = _load("execution_id")


def _inputs(**over):
    base = {
        "host": "hjbog-srdc-18.amd.com",
        "cell": "/runs/tp2_long",
        "side": "real",
        "repeat": 1,
        "server_pid": 4242,
        "launched_at_ns": 1_700_000_000_000_000_000,
    }
    base.update(over)
    return base


def _record(**over):
    inputs = _inputs(**over)
    return {
        "schema": eid.EXECUTION_SCHEMA,
        "execution_id": eid.derive_execution_id(**inputs),
        "id_rule": eid.ID_RULE,
        "id_inputs": inputs,
        "cell": {"path": inputs["cell"], "tp": 2, "class": "long"},
        "side": inputs["side"],
        "repeat": inputs["repeat"],
    }


class TestItDependsOnlyOnTheLaunch:
    def test_the_same_launch_gives_the_same_id(self):
        assert eid.derive_execution_id(**_inputs()) == eid.derive_execution_id(
            **_inputs()
        )

    def test_every_input_changes_it(self):
        base = eid.derive_execution_id(**_inputs())
        changed = {
            "host": "other.amd.com",
            "cell": "/runs/tp4_long",
            "side": "modelled",
            "repeat": 2,
            "server_pid": 4243,
            "launched_at_ns": 1_700_000_000_000_000_001,
        }
        for field, value in changed.items():
            assert eid.derive_execution_id(**_inputs(**{field: value})) != base, field

    def test_the_fields_are_joined_in_a_fixed_order(self):
        """Swapping two values must not give the same id -- a separator-free
        concatenation would."""
        assert eid.derive_execution_id(**_inputs(side="real", cell="x")) != (
            eid.derive_execution_id(**_inputs(side="x", cell="real"))
        )

    def test_it_looks_like_what_the_rule_says(self):
        made = eid.derive_execution_id(**_inputs())
        assert made.startswith("cx-")
        assert len(made) == 19
        int(made[3:], 16)

    def test_the_rule_names_the_fields_it_uses(self):
        for field in eid.ID_FIELDS:
            assert field in eid.ID_RULE


class TestAReaderCanCheckIt:
    def test_a_well_formed_record_verifies(self):
        assert eid.verify_execution_id(_record())

    def test_a_record_whose_inputs_were_edited_does_not(self):
        record = _record()
        record["id_inputs"]["server_pid"] = 1
        assert not eid.verify_execution_id(record)

    def test_a_record_whose_id_was_edited_does_not(self):
        record = _record()
        record["execution_id"] = "cx-0000000000000000"
        assert not eid.verify_execution_id(record)

    def test_nothing_and_nonsense_are_false_rather_than_an_exception(self):
        for value in (None, {}, [], "cx-abc", {"execution_id": "cx-abc"}):
            assert eid.verify_execution_id(value) is False

    def test_a_missing_input_is_false_rather_than_a_different_id(self):
        record = _record()
        del record["id_inputs"]["host"]
        assert not eid.verify_execution_id(record)


class TestItSurvivesBeingCarriedAround:
    def test_a_stamp_verifies_on_its_own(self):
        """What travels inside an artifact has to be checkable without the
        manifest it came from."""
        stamp = eid.stamp_of(_record())
        assert eid.verify_execution_id(stamp)
        assert set(stamp) == set(eid.STAMP_FIELDS)

    def test_a_stamp_does_not_alias_the_record_it_came_from(self):
        record = _record()
        stamp = eid.stamp_of(record)
        stamp["id_inputs"]["server_pid"] = 9
        assert eid.verify_execution_id(record)

    def test_a_round_trip_through_json_still_verifies(self):
        stamp = json.loads(json.dumps(eid.stamp_of(_record())))
        assert eid.verify_execution_id(stamp)

    def test_a_copied_file_still_names_its_execution(self, tmp_path):
        record = _record()
        first = tmp_path / "real.r1.json"
        first.write_text(json.dumps({"run": {}, "execution": eid.stamp_of(record)}))
        second = tmp_path / "copied.json"
        second.write_text(first.read_text())
        carried = eid.read_stamp(json.loads(second.read_text()))
        assert eid.verify_execution_id(carried)
        assert carried["execution_id"] == record["execution_id"]

    def test_a_file_with_no_stamp_reads_as_none(self, tmp_path):
        assert eid.read_stamp({"run": {}}) is None
        assert eid.read_stamp({"execution": "cx-abc"}) is None
        assert eid.read_stamp("not a blob") is None


class TestTheDigestIsAboutBytesNotIdentity:
    def test_two_executions_can_write_identical_bytes(self, tmp_path):
        """The case the id exists for: same payload, different runs."""
        one, two = tmp_path / "a.json", tmp_path / "b.json"
        payload = json.dumps({"run": {"requests": 8}})
        one.write_text(payload)
        two.write_text(payload)
        assert eid.file_digest(one) == eid.file_digest(two)
        assert eid.derive_execution_id(**_inputs(repeat=1)) != eid.derive_execution_id(
            **_inputs(repeat=2)
        )

    def test_a_missing_file_has_no_digest(self, tmp_path):
        assert eid.file_digest(tmp_path / "nothing.json") is None

    def test_a_digest_carries_the_size_as_well(self, tmp_path):
        path = tmp_path / "a.json"
        path.write_text("hello")
        assert eid.file_digest(path)["bytes"] == 5


class TestThereIsOnlyOneImplementation:
    def test_the_harness_does_not_define_the_rule_a_second_time(self):
        """It may rebind the names -- it may not restate the rule."""
        source = (ROOT / "scripts" / "compass" / "cc_traces_run.py").read_text()
        assert "def derive_execution_id" not in source
        assert "def verify_execution_id" not in source
        assert "hashlib" not in source

    def test_the_harness_mints_the_ids_this_module_verifies(self):
        run_mod = _load("cc_traces_run")
        assert run_mod.EXECUTION_SCHEMA == eid.EXECUTION_SCHEMA
        assert run_mod.ID_RULE == eid.ID_RULE
        made = run_mod.derive_execution_id(**_inputs())
        assert made == eid.derive_execution_id(**_inputs())
        assert eid.verify_execution_id({"execution_id": made, "id_inputs": _inputs()})

    def test_it_imports_nothing_but_the_standard_library(self):
        """A reader outside this repository has to be able to import it."""
        source = (ROOT / "scripts" / "compass" / "execution_id.py").read_text()
        imports = [
            line.strip()
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and "__future__" not in line
        ]
        assert imports == ["import hashlib", "from pathlib import Path"]
