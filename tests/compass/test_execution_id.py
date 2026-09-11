"""The one identity rule, checked the way each of its two readers uses it.

Two ends read this schema. `cc_traces_run.py` mints an `execution_id` when it
launches a server; the memory classifier reads one to tell an independent
repeat from a residual. They must agree exactly -- a divergence would not fail
loudly, it would quietly classify every execution as unidentified, which is the
answer that looks like caution and is actually a silent loss of the only
reproducibility evidence the calibration has.

They agreed, briefly, by coincidence: there were two implementations of the
rule, one in `scripts/compass/execution_id.py` and one in
`atom/compass/core/execution_id.py`, minting identical ids because the second
was copied carefully from the first. These tests are written against the state
that replaced that -- one definition in core, re-exported by the script -- and
the pinned vector below is what says the collapse preserved the ids rather than
merely preserving the prose describing them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from atom.compass.core.execution_id import (
    EXECUTION_SCHEMA,
    ID_FIELDS,
    ID_INPUTS,
    ID_RULE,
    STAMP_FIELDS,
    derive_execution_id,
    read_stamp,
    stamp_of,
    verify_execution_id,
)

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: Taken from `cc_traces_run.py` at f4e06b0c by executing that file's own
#: `derive_execution_id` on these arguments, before either module existed.
#: Pinned as a literal so the ids already written under `compass.execution/1`
#: keep verifying no matter how the implementation is arranged.
CC_INPUTS = ("hjbog-srdc-18", "RESULTS/tp2_long", "real", 1, 31337,
             1757500000000000000)
CC_VECTOR = "cx-f118d05298843dc2"


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
        "schema": EXECUTION_SCHEMA,
        "execution_id": derive_execution_id(**inputs),
        "id_rule": ID_RULE,
        "id_inputs": inputs,
        "cell": {"path": inputs["cell"], "tp": 2, "class": "long"},
        "side": inputs["side"],
        "repeat": inputs["repeat"],
    }


class TestTheIdsAlreadyWrittenStillVerify:
    """The schema is `compass.execution/1` and records carry ids under it."""

    def test_the_vector_from_ccs_implementation_still_derives(self):
        assert derive_execution_id(*CC_INPUTS) == CC_VECTOR

    def test_the_rule_is_what_the_rule_says(self):
        """Spelled out once so the prose and the code cannot drift apart."""
        digest = hashlib.sha256(
            "\0".join(str(part) for part in CC_INPUTS).encode("utf-8")
        ).hexdigest()
        assert derive_execution_id(*CC_INPUTS) == "cx-" + digest[:16]
        assert "NUL" in ID_RULE and ", ".join(ID_INPUTS) in ID_RULE


class TestItDependsOnlyOnTheLaunch:
    def test_the_same_launch_gives_the_same_id(self):
        assert derive_execution_id(**_inputs()) == derive_execution_id(**_inputs())

    def test_every_input_changes_it(self):
        base = derive_execution_id(**_inputs())
        changed = {
            "host": "other.amd.com",
            "cell": "/runs/tp4_long",
            "side": "modelled",
            "repeat": 2,
            "server_pid": 4243,
            "launched_at_ns": 1_700_000_000_000_000_001,
        }
        for field, value in changed.items():
            assert derive_execution_id(**_inputs(**{field: value})) != base, field

    def test_the_fields_are_joined_in_a_fixed_order(self):
        """Swapping two values must not give the same id -- a separator-free
        concatenation would."""
        assert derive_execution_id(**_inputs(side="real", cell="x")) != (
            derive_execution_id(**_inputs(side="x", cell="real"))
        )

    def test_it_looks_like_what_the_rule_says(self):
        made = derive_execution_id(**_inputs())
        assert made.startswith("cx-")
        assert len(made) == 19
        int(made[3:], 16)

    def test_the_rule_names_the_fields_it_uses(self):
        for field in ID_FIELDS:
            assert field in ID_RULE


class TestAReaderCanCheckIt:
    def test_a_well_formed_record_verifies(self):
        assert verify_execution_id(_record())

    def test_a_record_whose_inputs_were_edited_does_not(self):
        record = _record()
        record["id_inputs"]["server_pid"] = 1
        assert not verify_execution_id(record)

    def test_a_record_whose_id_was_edited_does_not(self):
        record = _record()
        record["execution_id"] = "cx-0000000000000000"
        assert not verify_execution_id(record)

    def test_nothing_and_nonsense_are_false_rather_than_an_exception(self):
        for value in (None, {}, [], "cx-abc", {"execution_id": "cx-abc"}):
            assert verify_execution_id(value) is False

    def test_a_missing_input_is_false_rather_than_a_different_id(self):
        record = _record()
        del record["id_inputs"]["host"]
        assert not verify_execution_id(record)

    def test_differently_named_inputs_are_false_rather_than_guessed_at(self):
        """A pre-schema record names its fields its own way. False here means
        "cannot say", not "a different run"."""
        assert not verify_execution_id(
            {"execution_id": CC_VECTOR,
             "id_inputs": {"host": "h", "pid": 1, "started_at": "t"}})


class TestItSurvivesBeingCarriedAround:
    def test_a_stamp_verifies_on_its_own(self):
        """What travels inside an artifact has to be checkable without the
        manifest it came from."""
        stamp = stamp_of(_record())
        assert verify_execution_id(stamp)
        assert set(stamp) == set(STAMP_FIELDS)

    def test_a_stamp_does_not_alias_the_record_it_came_from(self):
        record = _record()
        stamp = stamp_of(record)
        stamp["id_inputs"]["server_pid"] = 9
        assert verify_execution_id(record)

    def test_a_round_trip_through_json_still_verifies(self):
        stamp = json.loads(json.dumps(stamp_of(_record())))
        assert verify_execution_id(stamp)

    def test_a_copied_file_still_names_its_execution(self, tmp_path):
        record = _record()
        first = tmp_path / "real.r1.json"
        first.write_text(json.dumps({"run": {}, "execution": stamp_of(record)}))
        second = tmp_path / "copied.json"
        second.write_text(first.read_text())
        carried = read_stamp(json.loads(second.read_text()))
        assert verify_execution_id(carried)
        assert carried["execution_id"] == record["execution_id"]

    def test_a_file_with_no_stamp_reads_as_none(self):
        assert read_stamp({"run": {}}) is None
        assert read_stamp({"execution": "cx-abc"}) is None
        assert read_stamp("not a blob") is None


class TestTheDigestIsAboutBytesNotIdentity:
    def test_two_executions_can_write_identical_bytes(self, tmp_path):
        """The case the id exists for: same payload, different runs."""
        eid = _load("execution_id")
        one, two = tmp_path / "a.json", tmp_path / "b.json"
        payload = json.dumps({"run": {"requests": 8}})
        one.write_text(payload)
        two.write_text(payload)
        assert eid.file_digest(one) == eid.file_digest(two)
        assert derive_execution_id(**_inputs(repeat=1)) != derive_execution_id(
            **_inputs(repeat=2)
        )

    def test_a_missing_file_has_no_digest(self, tmp_path):
        assert _load("execution_id").file_digest(tmp_path / "nothing.json") is None

    def test_a_digest_carries_the_size_as_well(self, tmp_path):
        path = tmp_path / "a.json"
        path.write_text("hello")
        assert _load("execution_id").file_digest(path)["bytes"] == 5


class TestThereIsOnlyOneImplementation:
    """The point of the collapse: one rule, bound by name everywhere else."""

    def test_the_core_module_imports_nothing_but_the_standard_library(self):
        """A reader outside this repository has to be able to import it, and a
        core module every reader holds must not reach the filesystem."""
        source = (ROOT / "atom" / "compass" / "core"
                  / "execution_id.py").read_text()
        imports = [
            line.strip()
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and "__future__" not in line
        ]
        assert imports == ["import hashlib"]

    def test_the_script_rebinds_the_rule_rather_than_restating_it(self):
        source = (ROOT / "scripts" / "compass" / "execution_id.py").read_text()
        assert "def derive_execution_id" not in source
        assert "def verify_execution_id" not in source
        assert "atom.compass.core.execution_id" in source

    def test_the_script_exports_the_cores_own_objects(self):
        """Not equal values -- the same objects, so a change cannot reach one
        reader and miss the other."""
        eid = _load("execution_id")
        assert eid.derive_execution_id is derive_execution_id
        assert eid.verify_execution_id is verify_execution_id
        assert eid.stamp_of is stamp_of
        assert eid.read_stamp is read_stamp
        assert eid.EXECUTION_SCHEMA == EXECUTION_SCHEMA
        assert eid.ID_RULE == ID_RULE

    def test_the_harness_does_not_define_the_rule_a_second_time(self):
        """It may rebind the names -- it may not restate the rule."""
        source = (ROOT / "scripts" / "compass" / "cc_traces_run.py").read_text()
        assert "def derive_execution_id" not in source
        assert "def verify_execution_id" not in source
        assert "hashlib" not in source

    def test_the_harness_mints_the_ids_this_module_verifies(self):
        run_mod = _load("cc_traces_run")
        assert run_mod.EXECUTION_SCHEMA == EXECUTION_SCHEMA
        assert run_mod.ID_RULE == ID_RULE
        assert run_mod.derive_execution_id(*CC_INPUTS) == CC_VECTOR
        made = run_mod.derive_execution_id(**_inputs())
        assert made == derive_execution_id(**_inputs())
        assert verify_execution_id({"execution_id": made, "id_inputs": _inputs()})
