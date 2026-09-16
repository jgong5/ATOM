"""Diagnostic values retain evidence boundaries and never supply unknown work."""
from copy import deepcopy

import pytest

from atom.compass.core.cost.diagnostic_references import (
    DiagnosticReferencePrices, _bounded_prediction, _read_sources,
)
from atom.compass.core.cost.library import PriceLibrary
from atom.compass.core.cost.low_query import mha_prefix_interpolation
from atom.compass.core.cost.reached_primitives import work_identity
from atom.compass.replay.aiperf_profile import _check_source_options
from .test_cached_q16_prices import gdn, mha


def interpolation():
    sources = [{"reference_cell_id": "low", "weight": .5},
               {"reference_cell_id": "high", "weight": .5}]
    case = {"family": "mha", "frozen_prediction_sources": deepcopy(sources)}
    prediction = {"sources": sources, "seconds": .003, "source_qualified": False}
    points = {"low": {"seconds": .002, "source_qualified": True},
              "high": {"seconds": .004, "source_qualified": False}}
    graphs = {"low": mha(32752 + 14, query=14), "high": mha(65520 + 14, query=14)}
    return case, prediction, mha(49136 + 14, query=14), dict.fromkeys(graphs), points, graphs


def test_frozen_prediction_uses_same_math_and_retains_failed_precision():
    assert _bounded_prediction(*interpolation()) == (["low", "high"], .003)
    assert mha_prefix_interpolation(.002, .004, 49136) == (.003, [.5, .5])


@pytest.mark.parametrize("damage", ["query", "layout", "outside", "weight", "value", "precision", "bounds"])
def test_frozen_interpolation_refuses_changed_path_or_prediction(damage):
    case, prediction, op, refs, points, graphs = interpolation()
    if damage == "query":
        graphs["high"] = mha(65520 + 13, query=13)
    elif damage == "layout":
        graphs["high"]["abi"] = "different_attention_path"
    elif damage == "outside":
        op = mha(65536 + 14, query=14)
    elif damage == "weight":
        prediction["sources"][0]["weight"] = .6
    elif damage == "value":
        prediction["seconds"] = .0029
    elif damage == "precision":
        prediction["source_qualified"] = True
    else:
        graphs["low"] = mha(32768 + 14, query=14)
    with pytest.raises(ValueError):
        _bounded_prediction(case, prediction, op, refs, points, graphs)


def test_exact_overlay_preserves_aliases_and_missing_work_refusal():
    library = object.__new__(DiagnosticReferencePrices)
    library.base = PriceLibrary()
    library.handoff_sha256 = "a" * 64
    op = gdn(query=14, read=2, write=3)
    key, layer = work_identity(op)
    record = {"seconds": .002, "source": "original-median", "source_layer": layer,
              "source_qualified": False, "accepted": False, "reference_precision_passed": False}
    library._selected = {key: record}
    library._gdn_work = {key[0]}
    assert library.lookup(gdn(query=14, read=4, write=5))[0] == record
    missing, reason = library.lookup(gdn(query=14, read=4, write=4))
    assert missing is None and "state-alias" in reason
    assert library.lookup(gdn(query=15)) == library.base.lookup(gdn(query=15))
    assert library.lookup(op, {"tp": 2})[0] is None


@pytest.mark.parametrize("purpose,flag", [("acceptance", True), ("diagnostic", False), (None, True)])
def test_new_source_needs_explicit_diagnostic_purpose(purpose, flag):
    with pytest.raises(ValueError):
        _check_source_options({"purpose": purpose}, {"diagnostic_reference_handoff": "pinned.json",
                                                   "diagnostic_only": flag})


def test_diagnostic_source_never_relaxes_complete_coverage_or_becomes_qualified():
    _check_source_options({"purpose": "diagnostic"},
        {"diagnostic_reference_handoff": "pinned.json", "diagnostic_only": True, "require_complete": True})
    with pytest.raises(ValueError, match="complete cost coverage"):
        _check_source_options({"purpose": "diagnostic"}, {"require_complete": False})
    with pytest.raises(ValueError, match="unqualified diagnostic"):
        _read_sources(None, {"purpose": "diagnostic", "accepted": True}, "a" * 64)
