"""The producer identity, pinned against the harness that mints it.

Two ends read this schema: `cc_traces_run.py` writes an `execution_id` when it
launches a server, and the memory classifier reads one to tell an independent
repeat from a residual. They must agree exactly -- a divergence would not fail
loudly, it would quietly classify every execution as unidentified, which is the
answer that looks like caution and is actually a silent loss of the only
reproducibility evidence the calibration has.
"""

import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest

from atom.compass.core.execution_id import (EXECUTION_SCHEMA, ID_FIELDS,
                                            ID_INPUTS, ID_RULE, STAMP_FIELDS,
                                            derive_execution_id, read_stamp,
                                            stamp_of, verify_execution_id)

#: Taken from `cc_traces_run.py` at f4e06b0c by executing that file's own
#: `derive_execution_id` on these arguments. Pinned as a literal because the
#: script cannot be imported for its two functions -- it loads sibling modules
#: at import time -- which is the reason this module exists at all.
CC_INPUTS = ("hjbog-srdc-18", "RESULTS/tp2_long", "real", 1, 31337,
             1757500000000000000)
CC_VECTOR = "cx-f118d05298843dc2"

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts/compass/cc_traces_run.py"
CANONICAL = ROOT / "atom/compass/core/execution_id.py"
SCRIPT_HELPER = ROOT / "scripts/compass/execution_id.py"


def _record():
    return {"schema": EXECUTION_SCHEMA,
            "execution_id": derive_execution_id(*CC_INPUTS),
            "id_rule": ID_RULE,
            "id_inputs": dict(zip(ID_FIELDS, CC_INPUTS)),
            "cell": "RESULTS/tp2_long", "side": "real", "repeat": 1,
            "server_pid": 31337, "source": "not travelling with the artifact"}


def test_the_vector_from_ccs_implementation_still_derives():
    assert derive_execution_id(*CC_INPUTS) == CC_VECTOR


def test_the_rule_is_what_the_rule_says():
    """Spelled out once here so the prose and the code cannot drift apart."""
    digest = hashlib.sha256(
        "\0".join(str(part) for part in CC_INPUTS).encode("utf-8")).hexdigest()
    assert derive_execution_id(*CC_INPUTS) == "cx-" + digest[:16]
    assert "NUL" in ID_RULE and ", ".join(ID_INPUTS) in ID_RULE


def test_an_id_is_checked_against_its_own_inputs():
    record = {"schema": EXECUTION_SCHEMA,
              "execution_id": derive_execution_id(*CC_INPUTS),
              "id_inputs": dict(zip(ID_INPUTS, CC_INPUTS))}
    assert verify_execution_id(record)

    moved = dict(record, id_inputs=dict(record["id_inputs"], server_pid=31338))
    assert not verify_execution_id(moved)


def test_a_record_that_cannot_say_is_not_an_answer():
    """False here means "unidentified", never "a different run"."""
    assert not verify_execution_id(None)
    assert not verify_execution_id({})
    assert not verify_execution_id({"execution_id": CC_VECTOR})
    assert not verify_execution_id(
        {"execution_id": CC_VECTOR,
         "id_inputs": {"host": "h", "pid": 1, "started_at": "t"}})


def test_the_two_ends_agree_where_the_harness_is_on_the_box():
    """The real cross-check, when `cc_traces_run.py` is in this tree.

    Its functions are lifted out by name and executed alone, because importing
    the module runs sibling loads that need the campaign's layout. Skipped
    where the harness has not been merged here yet -- the vector above still
    stands, it is simply no longer re-derived from the source of truth.
    """
    if not HARNESS.exists():
        pytest.skip("cc_traces_run.py is not in this tree")
    namespace = {"hashlib": hashlib}
    for node in ast.parse(HARNESS.read_text(encoding="utf-8")).body:
        named = (isinstance(node, ast.FunctionDef)
                 and node.name in ("derive_execution_id", "verify_execution_id"))
        constant = (isinstance(node, ast.Assign)
                    and getattr(node.targets[0], "id", "") == "EXECUTION_SCHEMA")
        if named or constant:
            exec(compile(ast.Module([node], []), str(HARNESS), "exec"),
                 namespace)
    assert namespace["EXECUTION_SCHEMA"] == EXECUTION_SCHEMA
    assert namespace["derive_execution_id"](*CC_INPUTS) == CC_VECTOR


def test_the_two_names_for_the_field_order_are_one_object():
    """The memory side called it `ID_INPUTS`, the script side `ID_FIELDS`.

    Kept as an alias rather than a rename so neither caller had to change on
    the commit that made this module canonical. Asserting identity, not
    equality: two tuples that happen to match today are exactly the drift this
    module exists to prevent.
    """
    assert ID_INPUTS is ID_FIELDS


def test_a_stamp_carries_what_a_reader_needs_and_not_the_rest():
    """Enough to re-derive the id, nothing that goes stale when copied."""
    stamp = stamp_of(_record())
    assert tuple(stamp) == STAMP_FIELDS
    assert verify_execution_id(stamp)
    assert "source" not in stamp and "server_pid" not in stamp

    stamp["id_inputs"]["server_pid"] = 31338
    assert verify_execution_id(_record()), "the stamp was not a live view"


def test_an_artifact_that_names_no_execution_says_so():
    assert read_stamp({"execution": _record()})["execution_id"] == CC_VECTOR
    assert read_stamp({}) is None
    assert read_stamp({"execution": "cx-f118d05298843dc2"}) is None
    assert read_stamp(None) is None


def test_the_module_verifies_an_id_without_importing_the_engine():
    """Loaded from the file alone: no `atom` package, no torch, no AITER.

    This is the loader a script may use when it wants the rule and nothing
    else. It is tested because it is offered in the module docstring, and an
    untested recipe is a recipe that stops working.
    """
    spec = importlib.util.spec_from_file_location("_execution_id", CANONICAL)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    assert loaded.derive_execution_id(*CC_INPUTS) == CC_VECTOR
    assert loaded.ID_RULE == ID_RULE


def test_the_script_helper_delegates_rather_than_copies():
    """When CC's helper is merged here, it must not carry its own derivation.

    A second copy would pass this suite on the day it landed and diverge
    silently afterwards, so what is checked is delegation itself -- the same
    function object -- not merely the same answer.
    """
    if not SCRIPT_HELPER.exists():
        pytest.skip("scripts/compass/execution_id.py is not in this tree")
    spec = importlib.util.spec_from_file_location("_cc_execution_id",
                                                  SCRIPT_HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert helper.derive_execution_id is derive_execution_id
    assert helper.verify_execution_id is verify_execution_id
    assert helper.EXECUTION_SCHEMA == EXECUTION_SCHEMA
    assert tuple(helper.ID_FIELDS) == ID_FIELDS
    assert helper.ID_RULE == ID_RULE
    # Deliberately not "the file contains no sha256": the helper hashes
    # artifacts and is meant to. What must not come back is a second copy of
    # the derivation, and function identity above is what says so.
