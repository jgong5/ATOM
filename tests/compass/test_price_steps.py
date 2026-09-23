"""What has to be right before an offline pricing number is worth reading.

The script's value is that it takes the schedule out of the comparison: the
steps are given, so a cost model cannot change which steps ran, and the error
that comes out is the model's alone. That only holds if the oracle under test
is actually the one configured, and the way this can quietly fail is the table.
Every oracle here takes a step table, and they do not take it through the same
keyword -- the calibrated one fits it, the priced one falls back to it, the
constant one does not take one at all and raises if handed it. Silently passing
`table=` to all three would report the calibrated oracle's error under whatever
name was asked for, which is the one outcome this script exists to avoid.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "compass" / "price_steps.py")


def _load():
    spec = importlib.util.spec_from_file_location("price_steps", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ps = _load()


class TestTheTableReachesTheKeywordTheOracleReadsItThrough:

    def test_the_default_oracle_fits_it(self):
        qualname, options = ps._configure("calibrated", "sweep.jsonl", [])
        assert qualname.endswith("CalibratedCostOracle")
        assert options == {"table": "sweep.jsonl"}

    def test_the_priced_oracle_gets_it_as_a_fallback(self):
        # Not `table=`: PricedGraphCostOracle has no such parameter, and it
        # fits nothing. The file is where steps its graphs do not cover go.
        _, options = ps._configure("priced", "sweep.jsonl",
                                   ["prices=b.json", "graph=g.json"])
        assert options["fallback"] == "sweep.jsonl"
        assert "table" not in options

    def test_an_oracle_that_takes_no_table_is_handed_none(self):
        # ConstantCostOracle costs every step at two fixed rates and raises on
        # an unexpected keyword.
        _, options = ps._configure("constant", "sweep.jsonl", [])
        assert options == {}

    def test_an_unknown_qualname_is_handed_none(self):
        # It could take a table under either name or neither, so it has to say.
        _, options = ps._configure("pkg.mod.MyOracle", "sweep.jsonl", [])
        assert options == {}

    def test_a_named_option_wins_over_the_default(self):
        # The default is there so the common case needs no flag, not to stop
        # the fallback being pointed at a different table from the fit.
        _, options = ps._configure("priced", "sweep.jsonl",
                                   ["fallback=other.jsonl"])
        assert options["fallback"] == "other.jsonl"


class TestValuesAreTypedAsTheEngineTypesThem:
    """An oracle configured here and the same oracle configured on an engine
    command line have to be handed the same values, or the offline number does
    not describe the run."""

    @pytest.mark.parametrize("item, key, value", [
        ("bench_iters=200", "bench_iters", 200),
        ("boundary_seconds=3.2e-05", "boundary_seconds", 3.2e-05),
        ("prices=/p/bench.json", "prices", "/p/bench.json"),
    ])
    def test_one_value(self, item, key, value):
        assert ps._options([item]) == {key: value}

    def test_a_path_holding_an_equals_sign_keeps_it(self):
        assert ps._options(["graph=/p/g.json?x=1"])["graph"] == "/p/g.json?x=1"


def _steps(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return str(path)


class TestAPredictionIsNotLabelledAFitResidual:

    def test_an_oracle_that_fits_nothing_reports_no_residual(self, tmp_path,
                                                             capsys):
        # One file, named once, priced by an oracle that never read it. The
        # old check compared the two paths, so this said "fit residual" for a
        # model that had not been fitted to anything.
        table = _steps(tmp_path / "s.jsonl", [
            {"num_scheduled_tokens": [16], "context_lens": [16],
             "num_prefill_tokens": 16, "seconds": 0.040},
            {"num_scheduled_tokens": [1], "context_lens": [17],
             "num_prefill_tokens": 0, "seconds": 0.002},
        ])
        assert ps.main([table, "--oracle", "constant",
                        "--oracle-option", "prefill_seconds=0.02",
                        "--oracle-option", "decode_seconds=0.001"]) == 0
        out = capsys.readouterr().out
        assert "NOTE:" not in out
        assert "ConstantCostOracle(prefill=0.020000s" in out
        # Named, so a report cannot be read as the default oracle's.
        assert "ConstantCostOracle" in out.split("priced ")[1]
        # Both phases priced at half of measured, and each reported on its own.
        assert "prefill: 1 steps  measured 0.04s  priced 0.02s" in out
        assert "decode: 1 steps  measured 0.00s  priced 0.00s" in out
        # Totals and median, for each of the two phases.
        assert out.count("-50.00%") == 4

    def test_the_calibrated_oracle_still_reports_one(self):
        # Unchanged behaviour, pinned at the mechanism the note keys off: the
        # oracle was handed this file as the thing it fits.
        _, options = ps._configure("calibrated", "s.jsonl", [])
        assert options.get("table") == "s.jsonl"
