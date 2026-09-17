"""A prompt of N tokens must be N tokens.

The old construction was one made-up word per token and those are several
tokens each -- more as the counter grows, so the error was not even a constant
factor. Campaigns read by the shapes their steps recorded survived that; a
benchmark meant to reproduce a measured length distribution would not, because
a length-dependent distortion reshapes a distribution rather than shifting it.
"""
import pytest

from atom.compass.workload import WORDS, prompt_of_tokens


class TestExactLength:
    def test_a_prompt_has_one_word_per_token(self):
        # Exactness against a real tokenizer needs the model; the structural
        # guarantee is one whitespace-separated word per requested token, and
        # every word being a single token is the vocabulary's job.
        for n in (1, 2, 64, 1024, 40000):
            assert len(prompt_of_tokens(n, 3).split()) == n

    def test_zero_and_below_are_empty(self):
        assert prompt_of_tokens(0, 1) == ""
        assert prompt_of_tokens(-5, 1) == ""

    def test_it_starts_with_a_space(self):
        """So the first word tokenises the way the rest do."""
        assert prompt_of_tokens(4, 0).startswith(" ")

    def test_a_short_prompt_is_not_padded_past_its_length(self):
        assert len(prompt_of_tokens(2, 999).split()) == 2


class TestRequestsDoNotSharePrefixBlocks:
    """Blocks are hashed from position zero, so differing in the first token is
    enough to make every later block differ -- and two requests sharing a
    prefix would let the second skip the prefill being measured."""

    def test_the_opening_differs_between_requests(self):
        openings = {tuple(prompt_of_tokens(64, i).split()[:4])
                    for i in range(2000)}
        assert len(openings) == 2000

    def test_the_opening_encodes_the_index_in_base_52(self):
        assert len(WORDS) == 52
        assert prompt_of_tokens(4, 0).split()[:2] == [WORDS[0], WORDS[0]]
        assert prompt_of_tokens(4, 1).split()[0] == WORDS[1]
        assert prompt_of_tokens(4, 52).split()[:2] == [WORDS[0], WORDS[1]]


class TestARunDescribesItself:
    """An artifact that cannot say what produced it is not reproducible.

    A 40x arrival compression turned a 20-second arrival process into a
    half-second burst, and left no trace: the workload was scaled at send time
    and saved unscaled, so the saved file described a paced run that had not
    happened. The scaling now happens when the workload is built, so what is
    written is what ran.
    """

    def _replay(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "scripts/compass/replay.py"
        spec = importlib.util.spec_from_file_location("replay_manifest", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _workload(self, module, rows, scale):
        import argparse
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            trace = Path(d) / "t.jsonl"
            trace.write_text("\n".join(json.dumps(r) for r in rows))
            args = argparse.Namespace(
                trace=str(trace), num_requests=0, time_scale=scale,
                input_tokens=8, output_tokens=1, rate=0.0, seed=0)
            return module._workload(args)

    def test_the_saved_arrivals_are_the_arrivals_used(self):
        module = self._replay()
        rows = [{"arrival_s": 0.0, "input_tokens": 8, "output_tokens": 1},
                {"arrival_s": 20.0, "input_tokens": 8, "output_tokens": 1}]
        got = self._workload(module, rows, scale=40.0)
        assert got[-1]["arrival_s"] == pytest.approx(0.5)

    def test_scale_one_leaves_the_trace_alone(self):
        module = self._replay()
        rows = [{"arrival_s": 0.0, "input_tokens": 8, "output_tokens": 1},
                {"arrival_s": 20.0, "input_tokens": 8, "output_tokens": 1}]
        got = self._workload(module, rows, scale=1.0)
        assert got[-1]["arrival_s"] == pytest.approx(20.0)

    def test_a_digest_identifies_the_workload(self):
        """Two runs claiming the same workload should be checkable."""
        import tempfile
        from pathlib import Path

        module = self._replay()
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "a"; a.write_text("same")
            b = Path(d) / "b"; b.write_text("different")
            assert module._digest(str(a)) == module._digest(str(a))
            assert module._digest(str(a)) != module._digest(str(b))
            assert module._digest(None) is None
