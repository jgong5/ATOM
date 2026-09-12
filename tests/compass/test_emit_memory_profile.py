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
import pickle
import struct
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


def _shard(path, tensors):
    header, offset = {}, 0
    for name, shape in tensors.items():
        size = 2
        for dim in shape:
            size *= dim
        header[name] = {"dtype": "BF16", "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * offset)
    return offset


@pytest.fixture
def emitter_inputs(tmp_path):
    """A checkpoint and a capture history the emitter will accept."""
    checkpoint = tmp_path / "snapshot-deadbeef"
    checkpoint.mkdir()
    total = _shard(str(checkpoint / "model.safetensors"),
                   {"model.layers.0.weight": (64, 32)})
    (checkpoint / "config.json").write_text(json.dumps(CONFIG))
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total},
         "weight_map": {"model.layers.0.weight": "model.safetensors"}}))

    history = tmp_path / "capture_history.pickle"
    with open(history, "wb") as fh:
        pickle.dump({"device_traces": [_trace()]}, fh)

    config = tmp_path / "config.json"
    config.write_text(json.dumps(CONFIG))
    return checkpoint, history, config


class TestTheEmitterPathAcrossWidths:
    """The loop the emitter actually runs: TP1, then TP2, then TP4, in one
    process. Every width after the first runs in a process the previous width
    has already configured, which is where the sequencing bugs live."""

    @staticmethod
    def _stub_build(emitter, seen, checkpoint):
        """Record what each width was asked, and answer as a build would."""
        def rank_inventory(model, width, replay_target=None, identity=None):
            seen.append((model, width))
            per_rank = (64 // width * 32 * 2, 4096)
            answer = {"ranks": {r: per_rank for r in range(width)},
                      "uniform": True, "parameters": per_rank[0],
                      "buffers": per_rank[1], "spread": 0,
                      "identity": {"requested": model,
                                   "resolved_path": str(checkpoint),
                                   "architectures": ["Stub"],
                                   "geometry": {}}}
            if identity is not None:
                identity.update(answer["identity"])
            return answer
        emitter.rank_inventory = rank_inventory

    def test_it_emits_every_width_in_one_process(self, tmp_path,
                                                 emitter_inputs):
        checkpoint, history, config = emitter_inputs
        emitter, seen = _emitter(), []
        self._stub_build(emitter, seen, checkpoint)
        out = tmp_path / "set"

        assert emitter.main([
            "--out", str(out), "--checkpoint", str(checkpoint),
            "--capture-history", str(history), "--model-config", str(config),
            "--widths", "1", "2", "4", "--rank-inventory"]) == 0

        assert [width for _, width in seen] == [1, 2, 4]
        manifest = json.loads((out / "MANIFEST.json").read_text())
        assert sorted(manifest["widths"]) == ["1", "2", "4"]
        for width in (1, 2, 4):
            profile = json.loads((out / ("profile.tp%d.json" % width)).read_text())
            assert profile["world_size"] == width
            assert profile["parameters"] == 64 // width * 32 * 2
            assert profile["buffers"] == 4096
            provenance = profile["provenance"]
            assert provenance["parameters_route"] == "built"
            assert provenance["parameters_rank_uniform"] is True
            assert sorted(provenance["parameters_by_rank"]) == [
                str(r) for r in range(width)]

    def test_the_build_is_the_resolved_snapshot_not_the_hub_name(
            self, tmp_path, emitter_inputs):
        """A hub name carries no revision: the cache decides what it means,
        and a newer download decides differently. Counting from the name while
        recording the checkpoint's digests would let the weight term and the
        provenance describe different weights, with nothing saying so."""
        checkpoint, history, config = emitter_inputs
        emitter, seen = _emitter(), []
        self._stub_build(emitter, seen, checkpoint)
        out = tmp_path / "set"

        assert emitter.main([
            "--out", str(out), "--checkpoint", str(checkpoint),
            "--capture-history", str(history), "--model-config", str(config),
            "--widths", "1", "--rank-inventory"]) == 0

        assert [model for model, _ in seen] == [str(checkpoint)]
        profile = json.loads((out / "profile.tp1.json").read_text())
        assert profile["provenance"]["parameters_build_input"] == str(checkpoint)
        assert profile["provenance"]["model"] == "Qwen/Qwen3.8-27B"

    def test_a_build_that_resolved_elsewhere_is_refused(self, tmp_path,
                                                        emitter_inputs):
        checkpoint, history, config = emitter_inputs
        emitter, seen = _emitter(), []
        self._stub_build(emitter, seen, tmp_path / "somewhere-else")
        out = tmp_path / "set"

        assert emitter.main([
            "--out", str(out), "--checkpoint", str(checkpoint),
            "--capture-history", str(history), "--model-config", str(config),
            "--widths", "1", "--rank-inventory"]) == 2

    def test_a_config_that_is_not_the_checkpoints_is_refused(self, tmp_path,
                                                             emitter_inputs):
        """Three terms, three inputs. Each is correct about its own input, so
        if they are not the same model nothing in the set says so."""
        checkpoint, history, _ = emitter_inputs
        other = tmp_path / "other.json"
        other.write_text(json.dumps(dict(CONFIG, vocab_size=7)))
        emitter, seen = _emitter(), []
        self._stub_build(emitter, seen, checkpoint)

        assert emitter.main([
            "--out", str(tmp_path / "set"), "--checkpoint", str(checkpoint),
            "--capture-history", str(history), "--model-config", str(other),
            "--widths", "1", "--rank-inventory"]) == 2

    def test_the_notes_are_carried_into_every_profile(self, tmp_path,
                                                      emitter_inputs):
        checkpoint, history, config = emitter_inputs
        emitter, seen = _emitter(), []
        self._stub_build(emitter, seen, checkpoint)
        out = tmp_path / "set"

        assert emitter.main([
            "--out", str(out), "--checkpoint", str(checkpoint),
            "--capture-history", str(history), "--model-config", str(config),
            "--widths", "1", "2", "--rank-inventory",
            "--note", "a residual nobody has explained"]) == 0

        for width in (1, 2):
            profile = json.loads(
                (out / ("profile.tp%d.json" % width)).read_text())
            assert profile["provenance"]["notes"] == [
                "a residual nobody has explained"]
