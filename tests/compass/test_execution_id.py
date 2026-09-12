"""The one identity rule, checked the way each of its readers uses it.

Two ends read this schema. `cc_traces_run.py` mints an `execution_id` when it
launches a server; the memory classifier reads one to tell an independent
repeat from a residual. They must agree exactly -- a divergence would not fail
loudly, it would quietly classify every execution as unidentified, which is the
answer that looks like caution and is actually a silent loss of the only
reproducibility evidence the calibration has.

They agreed, briefly, by coincidence: there were two implementations, one in
`scripts/compass/execution_id.py` and one in `atom/compass/core/execution_id.py`,
minting identical ids because the second was copied carefully from the first.
This is the union of the three suites written against that situation -- the
harness's, the classifier's, and the merge -- kept whole because each of them
knew something the others did not. Two of their checks did not survive the
union, and both are decisions rather than repairs:

**A non-record is False, not an exception.** The harness's suite pinned
`verify_execution_id("cx-abc")` raising `AttributeError`, on the reading that
only a caller bug passes a bare string. But a pre-schema artifact that stores
`"execution": "cx-abc"` is data we will actually meet, and for it the honest
answer is the one this rule already defines everywhere else: False, meaning
"this record cannot say which run it was". Crashing a reader that was only
asking would turn "cannot say" into an outage, so the Mapping guard stays and
the raising test is not carried over.

**Delegation is checked by file, not by object identity.** The classifier's
suite asserted `helper.derive_execution_id is derive_execution_id`, which held
while both ends imported the package. The helper now loads the canonical file
*by path*, deliberately, so that verifying an id needs no engine import -- and
a path load necessarily produces a second module object, which no arrangement
of correct code can make identical to the package's. One file executed twice is
still one implementation; two files would not be. So what is asserted is that
both ends came from the same file, which is the property object identity was
standing in for.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
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
CANONICAL = ROOT / "atom" / "compass" / "core" / "execution_id.py"


def _load_path(name: str, path: Path):
    """By path, the way the helper itself loads the canonical rule."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load(name: str):
    return _load_path(f"compass_{name}", ROOT / "scripts" / "compass" / f"{name}.py")


eid = _load("execution_id")

#: Two frozen vectors, from the two implementations that existed before the
#: collapse -- the first taken from `cc_traces_run.py` at f4e06b0c by executing
#: that file's own derivation, the second from the script helper's suite.
#: Records already carry ids under both sets of inputs, so either one changing
#: is a schema break however the code is arranged.
CC_INPUTS = ("hjbog-srdc-18", "RESULTS/tp2_long", "real", 1, 31337,
             1757500000000000000)
CC_VECTOR = "cx-f118d05298843dc2"
HARNESS_INPUTS = ("hjbog-srdc-18.amd.com", "/runs/tp2_long", "real", 1, 4242,
                  1_700_000_000_000_000_000)
HARNESS_VECTOR = "cx-2fa6ef1092bfdd0a"


def _inputs(**over):
    base = dict(zip(ID_FIELDS, HARNESS_INPUTS))
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
        "source": "not travelling with the artifact",
    }


class TestTheIdsAlreadyWrittenStillVerify:
    """The schema is `compass.execution/1` and records carry ids under it."""

    def test_both_frozen_vectors_still_derive(self):
        assert derive_execution_id(*CC_INPUTS) == CC_VECTOR
        assert derive_execution_id(*HARNESS_INPUTS) == HARNESS_VECTOR

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

    def test_an_id_is_checked_against_its_own_inputs(self):
        """Against the inputs the record itself carries -- a record that moved
        an input and kept the id is the case this exists to catch."""
        record = _record()
        assert verify_execution_id(record)
        moved = dict(record,
                     id_inputs=dict(record["id_inputs"], server_pid=31338))
        assert not verify_execution_id(moved)

    def test_a_record_whose_inputs_were_edited_does_not(self):
        record = _record()
        record["id_inputs"]["server_pid"] = 1
        assert not verify_execution_id(record)

    def test_a_record_whose_id_was_edited_does_not(self):
        record = _record()
        record["execution_id"] = "cx-0000000000000000"
        assert not verify_execution_id(record)

    def test_a_missing_input_is_false_rather_than_a_different_id(self):
        record = _record()
        del record["id_inputs"]["host"]
        assert not verify_execution_id(record)

    def test_a_record_that_cannot_say_is_not_an_answer(self):
        """False here means "unidentified", never "a different run"."""
        for value in (None, {}, {"execution_id": CC_VECTOR}):
            assert verify_execution_id(value) is False

    def test_differently_named_inputs_are_false_rather_than_guessed_at(self):
        """A pre-schema record names its fields its own way."""
        assert not verify_execution_id(
            {"execution_id": CC_VECTOR,
             "id_inputs": {"host": "h", "pid": 1, "started_at": "t"}})

    def test_a_value_that_is_not_a_record_at_all_is_false(self):
        """Including the truthy ones.

        A bare id string is what a pre-schema artifact stores under
        `execution`, so this is data rather than a caller bug, and the answer
        it deserves is "cannot say" rather than a traceback out of a reader
        that was only asking. `read_stamp` guards the shape as well, but a
        reader that reaches `verify_execution_id` directly gets the same
        answer.
        """
        for value in ([], 0, "", "cx-abc", 7, ("a",)):
            assert verify_execution_id(value) is False


class TestItSurvivesBeingCarriedAround:
    def test_a_stamp_carries_what_a_reader_needs_and_not_the_rest(self):
        """Enough to re-derive the id, nothing that goes stale when copied."""
        stamp = stamp_of(_record())
        assert verify_execution_id(stamp)
        assert tuple(stamp) == STAMP_FIELDS
        assert "source" not in stamp and "server_pid" not in stamp

    def test_a_stamp_does_not_alias_the_record_it_came_from(self):
        record = _record()
        stamp = stamp_of(record)
        stamp["id_inputs"]["server_pid"] = 9
        assert verify_execution_id(record), "the stamp was a live view"

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

    def test_an_artifact_that_names_no_execution_says_so(self):
        assert read_stamp({"execution": _record()})["execution_id"] == (
            derive_execution_id(*HARNESS_INPUTS))
        assert read_stamp({}) is None
        assert read_stamp({"run": {}}) is None
        assert read_stamp({"execution": CC_VECTOR}) is None
        assert read_stamp(None) is None
        assert read_stamp("not a blob") is None


class TestTheDigestIsAboutBytesNotIdentity:
    def test_two_executions_can_write_identical_bytes(self, tmp_path):
        """The case the id exists for: same payload, different runs."""
        one, two = tmp_path / "a.json", tmp_path / "b.json"
        payload = json.dumps({"run": {"requests": 8}})
        one.write_text(payload)
        two.write_text(payload)
        assert eid.file_digest(one) == eid.file_digest(two)
        assert derive_execution_id(**_inputs(repeat=1)) != derive_execution_id(
            **_inputs(repeat=2))

    def test_a_missing_file_has_no_digest(self, tmp_path):
        assert eid.file_digest(tmp_path / "nothing.json") is None

    def test_a_digest_carries_the_size_as_well(self, tmp_path):
        path = tmp_path / "a.json"
        path.write_text("hello")
        assert eid.file_digest(path)["bytes"] == 5


class TestThereIsOnlyOneImplementation:
    """The point of the collapse: one rule, bound by name everywhere else."""

    def test_the_core_module_imports_nothing_but_the_standard_library(self):
        """A reader outside this repository has to be able to import it, and a
        core module every reader holds must not reach the filesystem.

        Read off the parse tree rather than off line prefixes: the docstring
        wraps a sentence onto a line beginning "from", and a check that counts
        that as an import is a check that fails on prose.
        """
        tree = ast.parse(CANONICAL.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported == {"__future__", "hashlib", "collections"}

    def test_the_script_helper_delegates_rather_than_defining(self):
        """A second copy would pass this suite on the day it landed and
        diverge silently afterwards, so what is checked is the absence of a
        derivation, not merely the same answer."""
        source = (ROOT / "scripts" / "compass" / "execution_id.py").read_text()
        assert "def derive_execution_id" not in source
        assert "def verify_execution_id" not in source
        # Not "the file contains no sha256": the helper hashes artifacts and is
        # meant to. What must not come back is a second copy of the rule.
        assert "sha256(" not in source.split("def file_digest")[0]

    def test_the_helper_and_the_package_ran_the_same_file(self):
        """The property object identity was standing in for.

        The helper loads the rule by path so that verifying an id never
        imports the engine, and a path load cannot produce the package's own
        module object. But it came from the same bytes, which is what "one
        implementation" means -- and two files would not survive this.
        """
        core = sys.modules["atom_compass_core_execution_id"]
        assert Path(core.__file__) == CANONICAL == Path(eid.CANONICAL)
        assert eid.derive_execution_id is core.derive_execution_id
        assert eid.verify_execution_id is core.verify_execution_id
        assert eid.stamp_of is core.stamp_of
        assert eid.read_stamp is core.read_stamp
        assert eid.derive_execution_id(*CC_INPUTS) == derive_execution_id(*CC_INPUTS)
        assert eid.EXECUTION_SCHEMA == EXECUTION_SCHEMA
        assert eid.ID_RULE == ID_RULE

    def test_it_loads_the_rule_by_path_and_not_as_a_package(self):
        assert "atom_compass_core_execution_id" in sys.modules

    def test_verifying_an_id_does_not_import_the_engine(self, tmp_path):
        """A machine with no device must still be able to check an id, so the
        package that would import the engine is never imported to do it.

        In a subprocess, because this session has already imported plenty.
        """
        script = tmp_path / "verify_only.py"
        script.write_text("""import importlib.util, json, sys

spec = importlib.util.spec_from_file_location("helper", sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
assert helper.verify_execution_id(json.loads(sys.argv[2]))
print([name for name in sys.modules if name.split(".")[0] == "atom"])
""")
        done = subprocess.run(
            [sys.executable, str(script),
             str(ROOT / "scripts" / "compass" / "execution_id.py"),
             json.dumps(_record())],
            capture_output=True, text=True, timeout=120, check=False)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "[]", done.stdout

    def test_the_two_names_for_the_field_order_are_one_object(self):
        """The memory side called it `ID_INPUTS`, the script side `ID_FIELDS`.

        An alias rather than a rename, so neither caller had to change on the
        commit that made this module canonical. Identity, not equality: two
        tuples that happen to match today are the drift this module exists to
        prevent.
        """
        assert ID_INPUTS is ID_FIELDS
        assert ID_INPUTS == ("host", "cell", "side", "repeat", "server_pid",
                             "launched_at_ns")
        assert tuple(eid.ID_FIELDS) == ID_FIELDS
        assert eid.ID_INPUTS is eid.ID_FIELDS

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

    def test_the_file_can_be_loaded_alone_for_the_rule_and_nothing_else(self):
        """The recipe the core module's own docstring offers. It is tested
        because an untested recipe is a recipe that stops working."""
        loaded = _load_path("_execution_id_alone", CANONICAL)
        assert loaded.derive_execution_id(*CC_INPUTS) == CC_VECTOR
        assert loaded.ID_RULE == ID_RULE
