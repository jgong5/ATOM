# SPDX-License-Identifier: MIT
"""`scripts/compass/compare_dashboard.py`: one page over many cells, by model,
deployment, parallel configuration and concurrency, from a manifest or a scan
of the directory layout, and its refusals."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "compass" / "compare_dashboard.py"
)
_spec = importlib.util.spec_from_file_location("compare_dashboard", SCRIPT)
dash = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dash)

WINDOW, STEP = 120, 30
WALL0 = 1_795_000_000
SIM0 = 1000.0
#: Each deployment's panels: (id, role), as CI's report names them.
PANELS = {
    "pd": [
        ("mesh_ttft", "overall"),
        ("decode_itl", "decode"),
        ("prefill_ttft", "prefill"),
    ],
    "aggregated": [("api_ttft", "standalone"), ("decode_itl", "standalone")],
}


def _report(path: Path, deployment: str, t0: float, value: float) -> None:
    ts = [t0 + k * STEP for k in range(WINDOW // STEP + 1)]
    panels = [
        {
            "id": pid,
            "title": pid,
            "unit": "ms",
            "role": role,
            "series": {"p50": [[t, value] for t in ts]},
        }
        for pid, role in PANELS[deployment]
    ]
    meta = {"title": "t", "start": ts[0], "end": ts[-1], "step": STEP, "window": 60}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "panels": panels}))


def _records(run: Path, first_ns: int) -> None:
    f = run / dash.RECORDS
    f.parent.mkdir(parents=True, exist_ok=True)
    meta = {"benchmark_phase": "profiling", "request_start_ns": first_ns}
    f.write_text(json.dumps({"metadata": meta}) + "\n")


def _cell(cdir: Path, deployment: str, sims=("sim_a",), sim_deployment=None):
    """A complete cell in the scan layout: the real run at 100, each sim at 110."""
    _records(cdir / dash.SCAN_REAL, WALL0 * 10**9)
    _report(cdir / dash.SCAN_REPORT, deployment, WALL0, 100.0)
    for label in sims:
        run = cdir / dash.SCAN_SIMS / label
        _records(run, dash.EPOCH_NS + int(SIM0 * 1e9))
        _report(run / "prometheus.json", sim_deployment or deployment, SIM0, 110.0)


def _page(out: Path) -> dict:
    text = out.read_text()
    m = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', text, re.DOTALL
    )
    return json.loads(m[1])


def _main(*argv):
    return dash.main([str(a) for a in argv])


def test_a_scan_shows_every_complete_cell_by_model_deployment_parallel_and_c(
    tmp_path, capsys
):
    root = tmp_path / "root"
    _cell(root / "M1/pd/1P1D/c64", "pd")
    _cell(root / "M1/pd/1P1D/c8", "pd")
    _cell(root / "M1/aggregated/TP8/c8", "aggregated", sims=("sim_a", "sim_b"))
    (root / "M1/aggregated/TP8/cell.json").write_text('{"parallel": "TP8 EP8"}')
    (root / "M1/aggregated/TP8/c8/cell.json").write_text('{"note": "n"}')
    # Not cells: a real run still going, a sim with no report, a stray directory.
    (root / "M1/pd/1P1D/c128/real").mkdir(parents=True)
    _records(root / "M1/pd/1P1D/c8/sims/sim_x", dash.EPOCH_NS)
    (root / "M1/notes").mkdir()
    out = tmp_path / "dash.html"

    assert _main("--scan", root, "--out", out, "--window", WINDOW) == 0

    data = _page(out)
    keys = [(c["model"], c["deployment"], c["parallel"], c["c"]) for c in data["cells"]]
    assert keys == [
        ("M1", "aggregated", "TP8 EP8", 8),
        ("M1", "pd", "1P1D", 8),
        ("M1", "pd", "1P1D", 64),
    ]
    agg = data["cells"][0]
    assert agg["note"] == "n"
    assert [r["label"] for r in agg["runs"]] == ["real", "sim_a", "sim_b"]
    assert agg["runs"][1]["pairs"]["api_ttft"]["p50"][:2] == [100.0, 110.0]
    assert set(agg["panels"]) == {"api_ttft", "decode_itl"}
    skipped = "\n".join(data["meta"]["skipped"])
    assert "c128: no prom/report-data.json" in skipped
    assert "sim_x: no prometheus.json" in skipped
    assert "notes: not a deployment directory" in skipped
    assert "skipped:" in capsys.readouterr().err


def test_overall_ttft_is_one_overview_metric_across_deployments(tmp_path):
    root = tmp_path / "root"
    _cell(root / "M1/pd/1P1D/c8", "pd")
    _cell(root / "M1/aggregated/TP8/c8", "aggregated")
    out = tmp_path / "dash.html"

    _main("--scan", root, "--out", out, "--window", WINDOW)

    cols = {c["pid"]: c for c in _page(out)["matrix_cols"]}
    assert cols["api_ttft"]["alias"] == cols["mesh_ttft"]["alias"] == "overall_ttft"
    assert cols["prefill_ttft"]["label"] == "Prefill TTFT"
    assert cols["decode_itl"]["label"] == "ITL"


def test_a_cell_whose_real_and_simulated_deployments_differ_is_refused(tmp_path):
    root = tmp_path / "root"
    _cell(root / "M1/pd/1P1D/c8", "pd", sim_deployment="aggregated")

    with pytest.raises(SystemExit, match="different deployments"):
        _main("--scan", root, "--out", tmp_path / "d.html", "--window", WINDOW)


def test_a_deployment_the_reports_contradict_is_refused(tmp_path):
    root = tmp_path / "root"
    _cell(root / "M1/aggregated/TP8/c8", "pd")

    with pytest.raises(SystemExit, match="the manifest says aggregated"):
        _main("--scan", root, "--out", tmp_path / "d.html", "--window", WINDOW)


def test_two_cells_of_one_model_deployment_parallel_and_c_are_refused(tmp_path):
    _cell(tmp_path / "a", "pd")
    cell = {
        "c": 8,
        "dir": "a",
        "real": {"run": "real", "report": "prom/report-data.json"},
        "sims": [{"label": "sim_a", "run": "sims/sim_a"}],
    }
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"model": "M1", "cells": [cell, cell]}))

    with pytest.raises(SystemExit, match="share model, deployment, parallel and c"):
        _main(manifest, "--out", tmp_path / "d.html", "--window", WINDOW)


def test_a_manifest_with_no_model_or_parallel_is_one_configuration(tmp_path):
    _cell(tmp_path / "a", "pd")
    cell = {
        "c": 8,
        "dir": "a",
        "real": {"run": "real", "report": "prom/report-data.json"},
        "sims": [{"label": "sim_a", "run": "sims/sim_a"}],
    }
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"cells": [cell]}))
    out = tmp_path / "d.html"

    assert _main(manifest, "--out", out, "--window", WINDOW) == 0

    (only,) = _page(out)["cells"]
    assert (only["model"], only["deployment"], only["parallel"]) == ("", "pd", "")


def test_a_manifest_and_a_scan_merge(tmp_path):
    _cell(tmp_path / "a", "pd")
    cell = {
        "c": 8,
        "dir": "a",
        "real": {"run": "real", "report": "prom/report-data.json"},
        "sims": [{"label": "sim_a", "run": "sims/sim_a"}],
        "model": "M0",
        "parallel": {"tp": 8, "ep": 8},
    }
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"cells": [cell]}))
    _cell(tmp_path / "root/M1/aggregated/TP8/c8", "aggregated")
    out = tmp_path / "d.html"

    _main(manifest, "--scan", tmp_path / "root", "--out", out, "--window", WINDOW)

    cells = _page(out)["cells"]
    assert [(c["model"], c["parallel"]) for c in cells] == [
        ("M0", "TP8 EP8"),
        ("M1", "TP8"),
    ]


def test_a_cell_s_window_s_keeps_its_runs_to_its_own_window(tmp_path):
    """Every report covers WINDOW s. On a page kept to a longer window, a cell
    without its own ``window_s`` reports that its runs end early; the cell whose
    ``window_s`` is WINDOW is kept to it and has no such warning."""
    root = tmp_path / "root"
    _cell(root / "M1/pd/1P1D/c8", "pd")
    _cell(root / "M1/pd/4P4D/c8", "pd")
    (root / "M1/pd/4P4D/cell.json").write_text(json.dumps({"window_s": WINDOW}))
    out = tmp_path / "dash.html"

    _main("--scan", root, "--out", out, "--window", WINDOW + 60)

    data = _page(out)
    cells = {c["parallel"]: c for c in data["cells"]}
    assert data["meta"]["window"] == WINDOW + 60
    assert cells["4P4D"]["window"] == WINDOW
    assert cells["4P4D"]["warnings"] == []
    assert len(cells["4P4D"]["runs"][0]["series"]["decode_itl"]["p50"]) == (
        WINDOW // STEP + 1
    )
    assert cells["1P1D"]["window"] == WINDOW + 60
    assert "real: its report ends before t0 + 180 s" in cells["1P1D"]["warnings"]


def test_a_window_s_of_no_whole_minutes_is_refused(tmp_path):
    root = tmp_path / "root"
    _cell(root / "M1/pd/1P1D/c8", "pd")
    (root / "M1/pd/1P1D/c8/cell.json").write_text('{"window_s": 90}')

    with pytest.raises(SystemExit, match="window_s 90 is not whole minutes"):
        _main("--scan", root, "--out", tmp_path / "d.html", "--window", WINDOW)


def test_the_ttft_p90_follows_the_p50_and_the_p99_is_not_shown(tmp_path):
    """one_real.py lists ``p90 TTFT`` among InferenceX's rows, after the TTFT
    P50 and P99: the page shows it after the P50, named TTFT p90, and leaves
    the P99 out of the per-request table and the request matrix."""
    root = tmp_path / "root"
    _cell(root / "M1/pd/1P1D/c8", "pd")
    rows = {"TTFT p50 ms": 1000.0, "TTFT p99 ms": 9000.0, "TPOT p50 ms": 9.0}
    rows |= {"p90 interactivity": 90.0, "p90 TTFT": 3.0}
    line = json.dumps({"real": rows, "sim_a": dict(rows, **{"p90 TTFT": 3.3})})
    (root / "M1/pd/1P1D/c8" / dash.SCAN_ONE_REAL).write_text(line + "\n")
    out = tmp_path / "dash.html"

    _main("--scan", root, "--out", out, "--window", WINDOW)

    data = _page(out)
    (cell,) = data["cells"]
    metrics = cell["one_real"]["metrics"]
    assert metrics[:3] == ["TTFT p50 ms", "p90 TTFT", "TPOT p50 ms"]
    assert "TTFT p99 ms" not in metrics
    assert data["request_names"]["p90 TTFT"] == "TTFT p90 ms"
    labels = [c["label"] for c in data["request_cols"]]
    assert labels[:3] == ["TTFT p50", "TTFT p90", "TPOT p50"]


def test_no_manifest_and_no_scan_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit):
        _main("--out", tmp_path / "d.html")
