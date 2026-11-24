# SPDX-License-Identifier: MIT
"""`scripts/compass/cal_coverage.py`: the share of a record's steps a law's own
hull, or the hull of a population's steps, holds."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compass" / "cal_coverage.py"
ROOT = SCRIPT.parents[2]
VECTOR = {"step": 0.001, "tokens": 1e-6, "query_square": 0.0, "query_cached": 0.0}


def _law(path: Path) -> Path:
    decode = {"step": 0.01, "context": 1e-8, "graph_padding": 0.0}
    law = {
        "provenance": "test",
        "prefill": {
            "coefficients": VECTOR,
            "hull": {
                "equal": {"query_cached": 0},
                "facets": [
                    [-1, 0, 64],
                    [1, 0, -4096],
                    [0, -1, 4096],
                    [0, 1, -(4096**2)],
                ],
            },
        },
        "decode": {
            "1": {
                "coefficients": decode,
                "hull": {
                    "equal": {"graph_padding": 0},
                    "facets": [[-1, 100], [1, -1000]],
                },
            }
        },
        "draft": {"provenance": "dropped", "decode": {}},
    }
    path.write_text(json.dumps({"params": law}))
    return path


def _record(path: Path, contexts, rung=1) -> Path:
    lines = [
        {
            "step": i,
            "dp_rank": 0,
            "is_dummy_run": False,
            "produces_output": True,
            "rows": [[1, c, True]],
            "rung": rung,
            "d_busy_ns": 10_000_000,
        }
        for i, c in enumerate(contexts)
    ]
    lines.append(
        {
            "step": len(lines),
            "dp_rank": 0,
            "is_dummy_run": False,
            "produces_output": True,
            "rows": [[2048, 2048, False]],
            "rung": None,
            "d_busy_ns": 30_000_000,
        }
    )
    (path / "steps").mkdir(parents=True)
    (path / "steps" / "dp0.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines)
    )
    return path


def _rows(*args) -> dict:
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT)},
        check=True,
    ).stdout
    return {
        line.split(" | ")[0].strip(): line.split(" | ")[1:]
        for line in out.splitlines()
        if line.count(" | ") == 4 and not line.strip().startswith("vector")
    }


def test_the_law_s_own_hull_holds_the_steps_inside_it(tmp_path):
    law = _law(tmp_path / "law.json")
    rows = _rows(law, _record(tmp_path / "real", [500, 900, 2000, 5000]))
    assert rows["decode rung 1"] == ["4", "50.0%", "0.0", "50.0%"]
    assert rows["prefill"][:2] == ["1", "100.0%"]


def test_a_population_s_hull_replaces_the_law_s(tmp_path):
    law = _law(tmp_path / "law.json")
    real = _record(tmp_path / "real", [500, 900, 2000, 5000])
    dry = _record(tmp_path / "dry", [300, 1200, 2400, 4000, 5100])
    rows = _rows(law, real, "--population", dry)
    assert rows["decode rung 1"][:2] == ["4", "100.0%"]
    # A step range of the population leaves its deep steps out.
    rows = _rows(law, real, "--population", f"{dry}:0-2")
    assert rows["decode rung 1"][:2] == ["4", "75.0%"]
