"""The profile emitter's source-derivation glue.

The emitter is where a served profile comes from, so the parts of it that
decide *what kind of number* the profile carries belong in the tree and under
test rather than in a scratch script beside it. Two of them:

* the TP=1 capture stream replayed at a width, which is the derived side the
  readback's `graph pool` term is gated against, and
* the base calibration, which has to come from the registry the rest of the
  tree reads rather than from constants retyped here.

The byte counts are not the subject -- `test_memory_capture` and
`test_memory_model` own those. What is checked here is that the glue stays
source-only and stays honest about where it got its inputs.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from atom.compass.core.memory_capture import LOGITS_IN_GRAPH_LINE

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "tests/compass/memory_records/qwen3_5_27b.config.json").read_text()
)
TEXT = CONFIG.get("text_config", CONFIG)
HIDDEN = int(TEXT["hidden_size"])
MLP_ACT = int(TEXT["intermediate_size"])
VOCAB = int(TEXT["vocab_size"])


def _emitter():
    """`emit_memory_profile.py` as a module. It is a script and lives in
    `scripts/`, so there is no package to import it from."""
    spec = importlib.util.spec_from_file_location(
        "compass_emit_memory_profile",
        ROOT / "scripts/compass/emit_memory_profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _alloc(addr, size, line):
    return {"action": "alloc", "addr": addr, "size": size,
            "frames": [{"name": "capture_cudagraph",
                        "filename": "/x/atom/model_engine/model_runner.py",
                        "line": line}]}


def _trace(bs=8):
    return [_alloc(10, bs * HIDDEN * 2, 4293),
            _alloc(11, bs * MLP_ACT * 2, 4293),
            _alloc(12, bs * HIDDEN * 2, 4296),
            _alloc(13, bs * VOCAB * 2, LOGITS_IN_GRAPH_LINE)]


class TestTheCapturePredictionIsDerived:

    def test_it_replays_the_recorded_stream_rather_than_stating_a_constant(
            self):
        """The one property that makes it a prediction.

        `capture_reserved_parts` reports whether its pool side came from
        replaying the request stream or from being handed a number, and the
        emitter refuses the second. A published constant wearing a derived
        provenance string is the failure this whole seam exists to catch.
        """
        emitter = _emitter()
        predicted = emitter.capture_prediction(_trace(), TEXT, 1, "/h.pickle",
                                               "ab" * 32)
        assert predicted["total"] == (predicted["pool_reserved"]
                                      + predicted["outside_pools"])
        assert predicted["source_history_sha256"] == "ab" * 32

    def test_the_width_it_was_replayed_at_is_in_its_provenance(self):
        """A capture number carried at the wrong width is the carry-forward
        failure with no symptom, so the string says which one it is."""
        emitter = _emitter()
        predicted = emitter.capture_prediction(_trace(), TEXT, 4, "/h.pickle",
                                               "ab" * 32)
        assert "replayed at TP=4" in predicted["provenance"]
        assert "no TP=4 measurement" in predicted["provenance"]

    def test_the_head_leaves_the_pool_above_one_rank(self):
        """Not arithmetic on the TP=1 answer: above one rank the logits
        allocation is not inside the captured region at all, so it is dropped
        from the request program rather than divided."""
        emitter = _emitter()
        one = emitter.capture_prediction(_trace(), TEXT, 1, "/h", "0" * 64)
        four = emitter.capture_prediction(_trace(), TEXT, 4, "/h", "0" * 64)
        assert one["dropped_logits_in_graph"] == 0
        assert four["dropped_logits_in_graph"] == 1
        assert four["pool_reserved"] < one["pool_reserved"]


class TestTheBaseCalibrationComesFromTheRegistry:

    def test_the_constants_are_the_registered_ones(self):
        """Retyping them here would make a second copy that can drift from the
        one every test in the tree checks."""
        emitter = _emitter()
        base = emitter.base_calibration("Qwen/Qwen3.8-27B")
        assert base["persistent"] == 252339712
        assert base["load_residue"][1] == 14924832

    def test_every_term_carries_the_basis_it_was_fitted_on(self):
        """`derived_readings` refuses a calibration whose provenance does not
        cover a term, and rightly: an unlabelled constant cannot be told from
        one read off the target."""
        emitter = _emitter()
        base = emitter.base_calibration("Qwen/Qwen3.8-27B")
        for term in ("persistent", "non_torch", "load_residue"):
            assert base["provenance"][term].startswith("S27")

    def test_an_unmeasured_model_is_refused_rather_than_defaulted(self):
        emitter = _emitter()
        with pytest.raises(SystemExit) as raised:
            emitter.base_calibration("Qwen/Qwen3-0.6B")
        assert "no default to fall back to" in str(raised.value)
