"""A rank must read back the artifact that rank wrote.

The write side has always given each rank its own file; the read side did not
know that, and a single rank cannot expose the gap because no suffix is applied
there. A TP=2 calibration wrote ``steps.tp0.jsonl`` and ``steps.tp1.jsonl``, the
predicting run asked for ``steps.jsonl``, and both workers died on a bare
``FileNotFoundError`` that reached the terminal as an engine manager complaining
about a shutdown signal during initialization.
"""

import inspect
import json

import pytest

from atom.compass.core.artifacts import rank_path, resolve_rank_path
from atom.compass.core.cost.calibrated import CalibratedCostOracle


def _table(path, rows=24):
    """A table wide enough in both features to fit without extrapolating."""
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(rows):
            tokens = 64 + 32 * i
            fh.write(json.dumps({
                "seconds": 0.001 + 1e-6 * tokens,
                "num_scheduled_tokens": [tokens],
                "context_lens": [0],
                "num_prefill_tokens": tokens,
            }) + "\n")
    return path


class TestRankPath:
    def test_the_suffix_is_stable_and_sorted(self):
        assert rank_path("g.json", {"tp": 1}) == "g.tp1.json"
        assert rank_path("g.json", {"tp": 1, "dp": 3}) == "g.dp3-tp1.json"

    def test_no_coordinates_leaves_the_path_alone(self):
        assert rank_path("g.json", {}) == "g.json"

    def test_the_runner_and_the_reader_agree(self):
        """The two sides must not drift apart again."""
        from atom.compass.runtime.runner import CompassModelRunner

        coords = {"tp": 3}
        written = CompassModelRunner._rank_path("steps.jsonl", coords)
        read, _ = resolve_rank_path("steps.jsonl", coords)
        assert written == read == "steps.tp3.jsonl"


class TestResolution:
    def test_a_rank_prefers_its_own_table(self, tmp_path):
        shared = tmp_path / "steps.jsonl"
        _table(shared)
        _table(tmp_path / "steps.tp1.jsonl")
        path, own = resolve_rank_path(str(shared), {"tp": 1})
        assert own and path.endswith("steps.tp1.jsonl")

    def test_it_falls_back_to_the_shared_table(self, tmp_path):
        shared = tmp_path / "steps.jsonl"
        _table(shared)
        path, own = resolve_rank_path(str(shared), {"tp": 1})
        assert path == str(shared)
        assert not own, "a shared table is not this rank's own measurement"

    def test_a_single_rank_run_is_unchanged(self, tmp_path):
        shared = tmp_path / "steps.jsonl"
        _table(shared)
        assert resolve_rank_path(str(shared), None) == (str(shared), False)


class TestTheOracleReadsTheRightFile:
    def test_it_fits_to_its_own_rank(self, tmp_path):
        shared = tmp_path / "steps.jsonl"
        _table(shared)
        mine = _table(tmp_path / "steps.tp1.jsonl")
        oracle = CalibratedCostOracle(table=str(shared), rank_coords={"tp": 1})
        assert oracle.table == str(mine)

    def test_the_fallback_announces_itself(self, tmp_path, caplog):
        """Fitting to another rank's numbers is an assumption, so it is said."""
        shared = tmp_path / "steps.jsonl"
        _table(shared)
        with caplog.at_level("WARNING"):
            CalibratedCostOracle(table=str(shared), rank_coords={"tp": 1})
        assert any("ATOMCompass WARNING:" in r.getMessage()
                   for r in caplog.records)

    def test_a_missing_table_names_itself(self, tmp_path):
        """This raises inside a worker; the message is all that gets out."""
        missing = tmp_path / "steps.jsonl"
        with pytest.raises(FileNotFoundError) as excinfo:
            CalibratedCostOracle(table=str(missing), rank_coords={"tp": 1})
        message = str(excinfo.value)
        assert "ATOMCompass" in message
        assert "steps.tp1.jsonl" in message, "must name the path it wanted"
        assert "--compass-mode=measure" in message, "must say how to make one"


class TestTheRunnerOffersTheRank:
    def test_only_to_an_oracle_that_asked_for_it(self):
        """``oracle_options`` is opaque, so the rank is offered, not imposed."""
        from atom.compass.core.cost.constant import ConstantCostOracle

        assert "rank_coords" in inspect.signature(CalibratedCostOracle).parameters
        assert "rank_coords" not in inspect.signature(ConstantCostOracle).parameters
