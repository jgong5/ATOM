"""Reading a shape list a caller wrote.

The CLI's own body is a composition of things tested elsewhere; keying a saved
graph and reading price entries moved to `test_source_oracle.py` with the code.
What is only here is the boundary between JSON a caller wrote and a
`StepShape`, where a wrong answer is quiet: a mistyped field priced as if the
field were absent.
"""

import importlib.util
import pathlib

import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "scripts" / "compass" / "predict_step.py")
_SPEC = importlib.util.spec_from_file_location("predict_step", _PATH)
predict_step = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(predict_step)


# -- the shape list ---------------------------------------------------------

def test_a_shape_is_read_with_its_declared_fields():
    shape = predict_step._shape_from(
        {"num_scheduled_tokens": [1, 1], "context_lens": [10, 20],
         "topology": {"tp": 2}, "rank_coords": {"tp": 1},
         "capture_bucket": 4, "produces_output": False})
    assert shape.num_scheduled_tokens == (1, 1)
    assert shape.context_lens == (10, 20)
    assert shape.capture_bucket == 4
    assert shape.produces_output is False


def test_a_shape_missing_lengths_is_refused():
    with pytest.raises(ValueError) as exc:
        predict_step._shape_from({"num_scheduled_tokens": [1]})
    assert "context_lens" in str(exc.value)


def test_a_misspelled_field_is_refused_rather_than_ignored():
    """`capture_buckets` would otherwise be dropped and read as no bucket."""
    with pytest.raises(ValueError) as exc:
        predict_step._shape_from({"num_scheduled_tokens": [1],
                                  "context_lens": [10],
                                  "capture_buckets": 4})
    assert "capture_buckets" in str(exc.value)


# -- what the report says it was priced from ---------------------------------

class TestTheReportRecordsItsOwnInputs:
    """A step time without its price list is a number nobody can re-derive.

    The same shape through two libraries is two answers under one heading, and
    `agent_scratch/g4/` holds two different files called
    `h27dec32.tp1.r0.json` -- so the record has to be the bytes, not the path.
    """

    @staticmethod
    def _args(tmp_path, **over):
        import argparse

        prices = tmp_path / "p.json"
        prices.write_text('{"entries": []}')
        graph = tmp_path / "g.json"
        graph.write_text('{"ops": []}')
        shapes = tmp_path / "s.json"
        shapes.write_text("[]")
        base = {
            "price": [f"{prices}:{graph}:unregistered"],
            "template": [str(graph)], "head_template": [],
            "replay_target": None, "shapes": str(shapes),
            "model": "/models/Q", "tp": 1, "device": "meta",
            "block_size": 16, "max_model_len": 262144, "position_rows": 1,
            "block_policy": "rounds", "cudagraph_mode": "full",
            "regions": "source-27b-tp1", "head": True,
            "seconds_per_launch": 0.0, "require_complete": True,
            "carry_allocation": False, "no_derive": False}
        base.update(over)
        return argparse.Namespace(**base), prices, graph

    def test_a_price_triple_is_recorded_part_by_part(self, tmp_path):
        import hashlib

        args, prices, graph = self._args(tmp_path)
        entry = predict_step._inputs(args)["prices"][0]
        assert entry["registration"] == "unregistered"
        assert entry["prices_digest"]["sha256"] == hashlib.sha256(
            prices.read_bytes()).hexdigest()
        # The graph is recorded too: it is what lets a layout mismatch be
        # refused instead of answered from a dense price.
        assert entry["graph_digest"]["sha256"] == hashlib.sha256(
            graph.read_bytes()).hexdigest()

    def test_a_price_list_with_no_graph_says_so_rather_than_omitting_it(
            self, tmp_path):
        args, _, _ = self._args(tmp_path, price=[str(tmp_path / "p.json")])
        entry = predict_step._inputs(args)["prices"][0]
        assert entry["graph"] is None
        assert entry["graph_digest"] is None
        assert entry["registration"] is None

    def test_a_missing_file_is_a_null_digest_not_a_crash(self, tmp_path):
        """Recording provenance must not be the thing that fails a run."""
        args, _, _ = self._args(tmp_path, replay_target=str(tmp_path / "no.json"))
        assert predict_step._inputs(args)["replay_target"]["digest"] is None

    def test_the_settings_that_change_the_answer_are_recorded(self, tmp_path):
        args, _, _ = self._args(tmp_path, require_complete=False, head=False)
        settings = predict_step._inputs(args)["settings"]
        # An incomplete sum is a diagnostic, not acceptance, and a report that
        # did not say which it was could be read as either.
        assert settings["require_complete"] is False
        assert settings["head"] is False
        assert settings["regions"] == "source-27b-tp1"
        assert settings["derive"] is True

    def test_the_digest_helper_is_the_shared_one(self):
        """Not a second implementation: same function, loaded by path."""
        import importlib.util
        import pathlib
        import sys

        path = (pathlib.Path(predict_step.__file__).resolve().parent
                / "execution_id.py")
        spec = importlib.util.spec_from_file_location("_eid_probe", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_eid_probe"] = module
        spec.loader.exec_module(module)
        assert predict_step._file_digest(path) == module.file_digest(path)
        # And it is that module's, not a local copy that happens to agree.
        assert "scripts_compass_execution_id" in sys.modules
