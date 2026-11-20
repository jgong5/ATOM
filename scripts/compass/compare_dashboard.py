#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Real against simulated Prometheus series of one cell, one HTML page per cell,
switchable by concurrency.

    python3 scripts/compass/compare_dashboard.py MANIFEST --out DASH.html
        [--one-real-tree TREE] [--window 1800]

Each concurrency's real run is CI's collector report (``report-data.json`` by
``.github/scripts/atomesh/observability/collect_metrics.py``), each simulated
run Compass's re-render of the same report from its simulated series
(``prometheus.json`` by ``atom.compass.report.prometheus``). The page embeds
their data (Chart.js comes from cdnjs) and holds:

- a matrix, concurrency x metric, of each simulated run's error against the
  real run, (sim - real) / real, colored by |error| <= 5%, 5-15%, > 15%. It
  reads either the panels' series (each panel's mean of its one-minute values
  over the minutes both runs have one, histogram panels at the selected
  statistic) or the per-request metrics of ``one_real.py``. The two differ:
  a mean of per-minute P99s is not the run's P99;
- per concurrency: ``one_real.py``'s per-request table, the report's panels
  (solid real, dashed simulated; P50, P90, P95, P99 or mean) and a 60 s table.

The two clocks: the real run's series are on the wall clock, the simulated
run's on its simulated clock. Each run is shifted to start at its first graded
request (the earliest send of a profiling record with no error in
``<run>/artifacts/profile_export.jsonl``; a simulated record's ns stamp is its
simulated time plus ``atom.compass.carriers.EPOCH_NS``) and kept from 0 to the
window, then interpolated onto one grid of the reports' step.

MANIFEST is JSON (YAML when PyYAML is installed). Relative paths resolve
against a cell's ``dir``, which resolves against ``base`` (default: the
manifest's directory). A cell with ``dir`` alone takes the standard layout:
real ``real1`` with ``real1-prom/report-data.json``, each simulated run
``<sim>/prometheus.json``, and ``one_real.txt`` when present::

    {"title": "Cell A: real vs simulated", "base": "/data/dsv41-mi355",
     "window_s": 1800,
     "cells": [
       {"c": 256, "dir": "a_pw_c256", "sims": ["step_linear", "rep1", "rep2"],
        "one_real": ["one_real_reps.txt"], "note": "in-sample law"},
       {"c": 128, "dir": "a", "note": "older tree, development data",
        "real": {"label": "real2", "run": "real2",
                 "report": "real2-prom/report-data.json"},
        "sims": [{"label": "fm_in", "run": "step_linear_fm_in",
                  "report": "step_linear_fm_in/prometheus.json",
                  "key": "step_linear_fm_in"}],
        "one_real": ["one_real_step_linear_fm_in.txt"]}]}

A run's ``run`` is the directory ``client.sh`` exported into; ``t0`` (unix
seconds for the real run, simulated seconds for a simulated one) stands in for
it. A simulated run's ``key`` is its column in ``one_real.py``'s output, by
default its run directory's name. Several ``one_real`` files merge, a later
file's column replacing an earlier one's. With ``--one-real-tree``, a cell
whose ``one_real`` files are all missing has ``one_real.py`` run from that
tree over its runs, its output kept beside the page.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import html
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

#: atom.compass.carriers.EPOCH_NS: added to each simulated ns stamp a client records.
EPOCH_NS = 10**9
RECORDS = "artifacts/profile_export.jsonl"
STATS = ("p50", "p90", "p95", "p99", "mean")
#: Panel kinds whose points are sampled state: a minute's value is their mean.
GAUGES = {"queues", "blocks"}
#: The sub-series a panel of each kind puts in the matrix, those present.
MATRIX_SUBS = {
    "queues": ("running", "waiting"),
    "blocks": ("used",),
    "cache": ("reuse", "hit"),
}
#: The sub-series a panel of each kind shows first.
DEFAULT_SUB = {"queues": "running", "blocks": "used", "cache": "reuse"}
#: Short matrix headers by panel id suffix; others use the panel's title.
SHORT = {
    "ttft": "TTFT",
    "itl": "ITL",
    "queue_time": "Queue time",
    "kv_blocks": "KV",
    "decode_batch_size": "Decode batch",
    "cache_hit": "Cache reuse",
    "request_gpu_forward": "Prefill GPU / req",
    "gpu_forward": "GPU / step",
    "request_tokens": "Uncached prompt tok",
    "batch_tokens": "Prefill batch tok",
    "prefill_context_tokens": "Prefill ctx tok",
    "decode_context_tokens": "Decode ctx tok",
}
#: one_real.py's metrics the request matrix shows, in order, and their headers.
REQUEST_COLS = [
    ("TTFT p50 ms", "TTFT p50"),
    ("TTFT p99 ms", "TTFT p99"),
    ("p90 TTFT", "p90 TTFT"),
    ("TPOT p50 ms", "TPOT p50"),
    ("TPOT p90 ms", "TPOT p90"),
    ("E2E p50 ms", "E2E p50"),
    ("E2E paired", "E2E paired"),
    ("throughput per GPU", "tput / GPU"),
    ("p90 interactivity", "p90 interact."),
    ("GPU hit rate", "GPU hit"),
    ("requests", "requests"),
]
THRESHOLDS = (0.05, 0.15)
CHART_JS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"


def load_manifest(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # an optional dependency, for a YAML manifest alone

        return yaml.safe_load(text)
    return json.loads(text)


def _path(root: Path, value) -> Path | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def normalize(manifest: dict, manifest_path: Path) -> list[dict]:
    """The manifest's cells, each run's paths resolved, sorted by concurrency."""
    base = _path(manifest_path.parent.resolve(), manifest.get("base", "."))
    cells = []
    for cell in manifest["cells"]:
        root = _path(base, cell.get("dir", "."))
        real = cell.get("real") or {}
        real = {
            "label": real.get("label", "real1"),
            "run": _path(root, real.get("run", "real1")),
            "report": _path(root, real.get("report", "real1-prom/report-data.json")),
            "t0": real.get("t0"),
            "note": real.get("note", ""),
        }
        sims = []
        for sim in cell.get("sims", []):
            sim = {"label": sim} if isinstance(sim, str) else dict(sim)
            run = _path(root, sim.get("run", sim["label"]))
            report = _path(root, sim.get("report")) or run / "prometheus.json"
            sims.append(
                {
                    "label": sim["label"],
                    "run": run,
                    "report": report,
                    "key": sim.get("key", run.name if run else report.parent.name),
                    "t0": sim.get("t0"),
                    "note": sim.get("note", ""),
                }
            )
        one_real = cell.get("one_real")
        if one_real is None:
            one_real = ["one_real.txt"] if (root / "one_real.txt").is_file() else []
        if isinstance(one_real, str):
            one_real = [one_real]
        cells.append(
            {
                "c": cell["c"],
                "note": cell.get("note", ""),
                "one_real_note": cell.get("one_real_note", ""),
                "real": real,
                "sims": sims,
                "one_real": [_path(root, f) for f in one_real],
            }
        )
    return sorted(cells, key=lambda c: c["c"])


def first_graded(run: Path, simulated: bool) -> tuple[float, int]:
    """The send of `run`'s first graded request (unix seconds, or simulated
    seconds for a simulated run), and its graded requests: one_real.py's
    ``profiled`` records, of the profiling phase (or none) and no error."""
    starts = []
    with (run / RECORDS).open() as f:
        for line in f:
            r = json.loads(line)
            meta = r.get("metadata", {})
            if r.get("error") or meta.get("benchmark_phase") not in (None, "profiling"):
                continue
            if (ns := meta.get("request_start_ns")) is not None:
                starts.append(int(ns))
    if not starts:
        raise SystemExit(f"{run / RECORDS}: no graded request with a send stamp")
    first = min(starts)
    if simulated:
        if first < EPOCH_NS:
            raise SystemExit(
                f"{run}: a send stamp before EPOCH_NS; not a simulated run?"
            )
        first -= EPOCH_NS
    return first / 1e9, len(starts)


def read_report(path: Path) -> dict:
    data = json.loads(path.read_text())
    panels = {}
    for p in data["panels"]:
        if p.get("kind") == "requests":
            continue
        panels[p["id"]] = p
    return {"meta": data["meta"], "panels": panels}


def resample(points: list, t0: float, step: float, n: int) -> list:
    """`points` ([t, v or None], t increasing) at t0 + k * step for k < n,
    linear between the two points either side, None where either is None."""
    ts = [p[0] for p in points]
    out = []
    for k in range(n):
        g = t0 + k * step
        i = bisect.bisect_left(ts, g)
        if i < len(ts) and abs(ts[i] - g) < 1e-6:
            out.append(points[i][1])
            continue
        if i == 0 or i == len(ts):
            out.append(None)
            continue
        (ta, va), (tb, vb) = points[i - 1], points[i]
        if va is None or vb is None or tb - ta > 2 * step:
            out.append(None)
            continue
        out.append(va + (vb - va) * (g - ta) / (tb - ta))
    return out


def sig(v, digits=4):
    if v is None or not math.isfinite(v):
        return None
    if v == 0:
        return 0
    return round(v, digits - 1 - math.floor(math.log10(abs(v))))


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def minutes(values: list, per_minute: int, gauge: bool) -> list:
    """Each minute's value: a histogram or ratio point at the minute's end
    summarizes the window before it; sampled state is averaged over the minute."""
    out = []
    for k in range(1, (len(values) - 1) // per_minute + 1):
        end = k * per_minute
        if gauge:
            out.append(mean(values[end - per_minute + 1 : end + 1]))
        else:
            out.append(values[end])
    return out


def paired_means(real: dict, sim: dict) -> dict:
    """For each panel series both runs hold, the real and the simulated mean
    over the minutes both have a value in, and those minutes' count: a minute
    one run has no value in (no decode output at all in a minute, say) would
    otherwise weigh on one side alone."""
    out = {}
    for pid in real.keys() & sim.keys():
        for sub in real[pid].keys() & sim[pid].keys():
            both = [
                (a, b)
                for a, b in zip(real[pid][sub], sim[pid][sub])
                if a is not None and b is not None
            ]
            if both:
                out.setdefault(pid, {})[sub] = [
                    sig(mean([a for a, _ in both]), 6),
                    sig(mean([b for _, b in both]), 6),
                    len(both),
                ]
    return out


_TAIL = re.compile(
    r"(TTFT p99 ms|p90 TTFT|TPOT p90 ms) ([+-][\d.]+)%\.\.([+-][\d.]+)% \(([^)]*)\)"
)
_TAIL_SIM = re.compile(r"(\S+) ([+-][\d.]+)% (in|OUT)")
_PAIRED = re.compile(
    r"^(\S+): E2E paired over the (\d+) requests both ran: median ratio ([+-][\d.]+)%,"
    r" on totals ([+-][\d.]+)%; real E2E p50's own 95% interval ([+-][\d.]+)%\.\.([+-][\d.]+)%"
)


def parse_one_real(path: Path) -> dict:
    """one_real.py's output: its rows (the closing JSON line), the real tails'
    intervals, each simulated run's paired E2E, and its other lines."""
    lines = path.read_text().splitlines()
    rows = None
    for line in reversed(lines):
        if line.startswith('{"real"'):
            rows = json.loads(line)
            break
    if rows is None:
        raise SystemExit(f"{path}: no one_real.py JSON line")
    tails, tail_sims, paired, notes = {}, {}, {}, []
    names = set(rows["real"])
    for line in lines:
        if not line.strip() or line.startswith(("{", "metric ")):
            continue
        if re.split(r"\s{2,}", line.strip())[0] in names:
            continue
        notes.append(line)
        if line.startswith("real tails"):
            for m in _TAIL.finditer(line):
                tails[m[1]] = [float(m[2]) / 100, float(m[3]) / 100]
                for s in _TAIL_SIM.finditer(m[4]):
                    tail_sims.setdefault(s[1], {})[m[1]] = s[3]
        elif m := _PAIRED.match(line):
            paired[m[1]] = {
                "n": int(m[2]),
                "median": float(m[3]) / 100,
                "totals": float(m[4]) / 100,
                "interval": [float(m[5]) / 100, float(m[6]) / 100],
            }
    return {
        "file": str(path),
        "rows": rows,
        "tails": tails,
        "tail_sims": tail_sims,
        "paired": paired,
        "notes": notes,
    }


def merge_one_real(parsed: list[dict]) -> dict | None:
    if not parsed:
        return None
    out = {
        "files": [],
        "real": None,
        "sims": {},
        "tails": {},
        "tail_sims": {},
        "paired": {},
        "notes": [],
        "warnings": [],
    }
    for p in parsed:
        out["files"].append(p["file"])
        real = p["rows"]["real"]
        if out["real"] is None:
            out["real"] = real
        elif real.get("requests") != out["real"].get("requests"):
            out["warnings"].append(
                f"{p['file']}: its real run has {real.get('requests')} requests,"
                f" an earlier file's {out['real'].get('requests')}"
            )
        out["sims"].update({k: v for k, v in p["rows"].items() if k != "real"})
        out["tails"].update(p["tails"])
        out["tail_sims"].update(p["tail_sims"])
        out["paired"].update(p["paired"])
        out["notes"] += [n for n in p["notes"] if n not in out["notes"]]
    out["metrics"] = list(out["real"])
    return out


def run_one_real(tree: Path, cell: dict, out: Path) -> Path:
    runs = [cell["real"]["run"], *(s["run"] for s in cell["sims"])]
    env = dict(os.environ, PYTHONPATH=str(tree))
    with out.open("w") as f:
        done = subprocess.run(
            [
                sys.executable,
                str(tree / "scripts/compass/one_real.py"),
                *map(str, runs),
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/",
            check=False,
        )
    if done.returncode:
        raise SystemExit(f"one_real.py exited {done.returncode}; see {out}")
    return out


def short(panel: dict) -> str:
    for suffix, name in SHORT.items():
        if panel["id"].endswith(suffix):
            return name
    return panel["title"]


def build(cells: list[dict], window: float, args) -> dict:
    panels, order = {}, []
    out_cells = []
    for cell in cells:
        runs = [dict(cell["real"], sim=False)] + [
            dict(s, sim=True) for s in cell["sims"]
        ]
        reports = [read_report(r["report"]) for r in runs]
        step = max(float(r["meta"]["step"]) for r in reports)
        if 60 % step:
            raise SystemExit(
                f"c={cell['c']}: report step {step} s does not divide 60 s"
            )
        n = int(window // step) + 1
        per_minute = int(60 // step)
        for run, rep in zip(runs, reports):
            for pid, p in rep["panels"].items():
                if pid not in panels:
                    order.append(pid)
                    panels[pid] = {
                        "id": pid,
                        "title": p["title"],
                        "label": p.get("label", ""),
                        "detail": p.get("detail", ""),
                        "unit": p.get("unit", "ms"),
                        "kind": p.get("kind"),
                        "subs": [],
                    }
                info = panels[pid]
                if not run["sim"]:
                    # A simulated report relabels its GPU panels predicted.
                    info.update(label=p.get("label", ""), detail=p.get("detail", ""))
                for sub in p.get("series", {}):
                    if p.get("kind") and sub not in info["subs"]:
                        info["subs"].append(sub)
        out_runs = []
        for run, rep in zip(runs, reports):
            if run["t0"] is not None:
                t0, graded = float(run["t0"]), None
            else:
                t0, graded = first_graded(run["run"], run["sim"])
            series, mins, means = {}, {}, {}
            for pid, p in rep["panels"].items():
                gauge = p.get("kind") in GAUGES
                for sub, points in p.get("series", {}).items():
                    values = resample(points, t0, step, n)
                    if all(v is None for v in values):
                        continue
                    m = minutes(values, per_minute, gauge)
                    series.setdefault(pid, {})[sub] = [sig(v) for v in values]
                    mins.setdefault(pid, {})[sub] = [sig(v) for v in m]
                    means.setdefault(pid, {})[sub] = sig(mean(m), 6)
            meta = rep["meta"]
            if run["sim"]:
                t0_text = f"simulated clock, t0 = {t0:.1f} s"
            else:
                when = dt.datetime.fromtimestamp(t0, dt.timezone.utc)
                t0_text = f"wall clock, t0 = {when:%Y-%m-%d %H:%M:%S} UTC"
            out_runs.append(
                {
                    "label": run["label"],
                    "sim": run["sim"],
                    "key": run.get("key"),
                    "note": run["note"],
                    "t0": t0,
                    "t0_text": t0_text,
                    "graded": graded,
                    "report": str(run["report"]),
                    "run": str(run["run"]) if run["run"] else None,
                    "window": meta.get("window"),
                    "report_title": meta.get("title"),
                    "report_notes": meta.get("notes", []),
                    "covered": meta["end"] >= t0 + window,
                    "series": series,
                    "minutes": mins,
                    "means": means,
                }
            )
        for run in out_runs[1:]:
            run["pairs"] = paired_means(out_runs[0]["minutes"], run["minutes"])
        files = [f for f in cell["one_real"] if f.is_file()]
        missing = [str(f) for f in cell["one_real"] if not f.is_file()]
        if not files and args.one_real_tree and all(r["run"] for r in runs):
            target = args.out.parent / f"one_real_c{cell['c']}.txt"
            files, missing = [run_one_real(args.one_real_tree, cell, target)], []
        one_real = merge_one_real([parse_one_real(f) for f in files])
        warnings = []
        if missing:
            warnings.append(f"one_real file(s) not found: {', '.join(missing)}")
        if one_real:
            warnings += one_real["warnings"]
            real_n = one_real["real"].get("requests")
            if out_runs[0]["graded"] is not None and real_n != out_runs[0]["graded"]:
                warnings.append(
                    f"one_real.py's real run has {real_n} graded requests, the panels'"
                    f" real run ({out_runs[0]['label']}) {out_runs[0]['graded']}:"
                    " they are different runs"
                )
            for r in out_runs[1:]:
                row = one_real["sims"].get(r["key"])
                if row is None:
                    warnings.append(f"one_real output has no column {r['key']!r}")
                elif r["graded"] is not None and row.get("requests") != r["graded"]:
                    warnings.append(
                        f"one_real.py's {r['key']} has {row.get('requests')} graded"
                        f" requests, the panels' run {r['graded']}"
                    )
        for r in out_runs:
            if not r["covered"]:
                warnings.append(
                    f"{r['label']}: its report ends before t0 + {window:g} s"
                )
        out_cells.append(
            {
                "c": cell["c"],
                "note": cell["note"],
                "one_real_note": cell["one_real_note"],
                "step": step,
                "window": window,
                "runs": out_runs,
                "one_real": one_real,
                "warnings": warnings,
            }
        )
    panel_list = []
    for pid in order:
        p = panels[pid]
        if p["kind"]:
            p["default_sub"] = DEFAULT_SUB.get(
                p["kind"], p["subs"][0] if p["subs"] else None
            )
        panel_list.append(p)
    matrix_cols = []
    for p in panel_list:
        if not p["kind"]:
            matrix_cols.append(
                {"pid": p["id"], "sub": None, "label": short(p), "unit": p["unit"]}
            )
            continue
        for sub in MATRIX_SUBS.get(p["kind"], ()):
            if sub in p["subs"]:
                label = {"queues": sub.title(), "blocks": f"KV {sub}"}.get(
                    p["kind"], short(p)
                )
                matrix_cols.append(
                    {"pid": p["id"], "sub": sub, "label": label, "unit": p["unit"]}
                )
    return {
        "meta": {
            "title": args.title,
            "generated": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC",
            "manifest": str(args.manifest),
            "window": window,
            "thresholds": THRESHOLDS,
        },
        "panels": panel_list,
        "matrix_cols": matrix_cols,
        "request_cols": [{"key": k, "label": v} for k, v in REQUEST_COLS],
        "cells": out_cells,
    }


def summary(data: dict) -> str:
    """A text matrix: each simulated run's error on TTFT / TPOT, per request and
    on the panels' P50 window means."""
    lines = []
    for cell in data["cells"]:
        for run in cell["runs"][1:]:
            parts = []
            if cell["one_real"] and run["key"] in cell["one_real"]["sims"]:
                r, s = cell["one_real"]["real"], cell["one_real"]["sims"][run["key"]]
                for k in ("TTFT p50 ms", "TPOT p50 ms", "E2E p50 ms"):
                    if r.get(k) and s.get(k) is not None:
                        parts.append(f"{k.replace(' ms', '')} {s[k] / r[k] - 1:+.1%}")
            for pid, name in (("api_ttft", "panel TTFT"), ("decode_itl", "panel ITL")):
                a, b, _ = run["pairs"].get(pid, {}).get("p50", (None, None, 0))
                if a and b is not None:
                    parts.append(f"{name} p50 {b / a - 1:+.1%}")
            lines.append(f"c={cell['c']:<4} {run['label']:<20} " + ", ".join(parts))
        lines += [f"c={cell['c']:<4} warning: {w}" for w in cell["warnings"]]
    return "\n".join(lines)


def render(data: dict, out: Path) -> None:
    payload = (
        json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    page = (
        PAGE.replace("__TITLE__", html.escape(data["meta"]["title"]))
        .replace("__CHART_JS__", CHART_JS)
        .replace("__DATA__", payload)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--window",
        type=float,
        help="seconds kept from t0 (default: the manifest's window_s, else 1800)",
    )
    parser.add_argument("--title", help="page title (default: the manifest's)")
    parser.add_argument(
        "--one-real-tree",
        type=Path,
        help="an ATOM tree whose scripts/compass/one_real.py runs for a cell with no one_real file",
    )
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    window = args.window or float(manifest.get("window_s", 1800))
    if window % 60:
        parser.error("the window must be whole minutes")
    args.title = args.title or manifest.get("title", "Real vs simulated Prometheus")
    data = build(normalize(manifest, args.manifest), window, args)
    render(data, args.out)
    print(summary(data))
    print(f"{args.out.resolve()} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --hover: rgba(11,11,11,0.04);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --good: #0ca30c; --warn: #fab219; --crit: #d03b3b; --na: #898781;
  --tint: 20%;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.05);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
    --tint: 30%;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.05);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
  --tint: 30%;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body { background: var(--page); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1440px; margin: 0 auto; padding: 0 16px; }
header.wrap { padding-top: 20px; }
h1 { font-size: 20px; margin: 0 0 4px; font-weight: 650; }
h2 { font-size: 16px; margin: 0; font-weight: 650; }
h3 { font-size: 14px; margin: 0; font-weight: 650; }
.sub, .caption, .muted { color: var(--ink2); }
.small { font-size: 12px; }
.controls { position: sticky; top: 0; z-index: 5; background: var(--page); display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center; padding-top: 10px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.ctl { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.ctl > span { color: var(--ink2); font-size: 12px; }
.seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button { font: inherit; font-size: 13px; color: var(--ink2); background: none; border: 0; padding: 5px 11px; cursor: pointer; border-right: 1px solid var(--border); }
.seg button:last-child { border-right: 0; }
.seg button:hover { background: var(--hover); }
.seg button[aria-pressed="true"] { color: var(--ink); background: var(--hover); font-weight: 650; box-shadow: inset 0 -2px 0 var(--ink); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin: 16px 0; }
.card-head { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 8px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; background: var(--surface); }
th:first-child, td:first-child { text-align: left; }
td.lbl { text-align: left; }
tr.sel td:not(.err) { background: var(--hover); font-weight: 650; }
#matrix th { white-space: normal; vertical-align: bottom; line-height: 1.2; min-width: 62px; max-width: 96px; }
#matrix th:nth-child(-n+2) { max-width: none; }
tr.clickable { cursor: pointer; }
tr.clickable:hover td:not(.err) { background: var(--hover); }
tr.clickable:hover td.err { box-shadow: inset 0 1px 0 var(--ink2), inset 0 -1px 0 var(--ink2); }
tr.total td { font-weight: 650; border-top: 2px solid var(--axis); }
td.err { min-width: 58px; }
td.good { background: color-mix(in srgb, var(--good) var(--tint), transparent); }
td.warn { background: color-mix(in srgb, var(--warn) var(--tint), transparent); }
td.crit { background: color-mix(in srgb, var(--crit) var(--tint), transparent); }
td.na { color: var(--muted); }
.tag { font-size: 10.5px; color: var(--ink2); margin-left: 3px; }
.tag.out { color: var(--crit); font-weight: 650; }
.key { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px; color: var(--ink2); margin: 6px 0 10px; }
.key i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -2px; margin-right: 5px; border: 1px solid var(--border); }
.key i.good { background: color-mix(in srgb, var(--good) var(--tint), transparent); }
.key i.warn { background: color-mix(in srgb, var(--warn) var(--tint), transparent); }
.key i.crit { background: color-mix(in srgb, var(--crit) var(--tint), transparent); }
.key i.na { background: transparent; }
.runs { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 8px 0 2px; }
.run { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }
.run small { color: var(--ink2); }
.warnings { margin: 8px 0 0; padding-left: 18px; color: var(--ink); }
.warnings li::marker { content: "\26A0  "; color: var(--crit); }
.panels { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 430px), 1fr)); gap: 14px; }
.panel { margin: 0; padding: 12px 14px 10px; display: flex; flex-direction: column; }
.panel .lab { font-size: 10.5px; letter-spacing: .04em; color: var(--muted); text-transform: uppercase; }
.panel .det { font-size: 11.5px; color: var(--ink2); margin: 2px 0 6px; min-height: 2.6em; }
.chips { display: flex; flex-wrap: wrap; gap: 4px; margin: 0 0 6px; }
.chips button { font: inherit; font-size: 11.5px; color: var(--ink2); background: none; border: 1px solid var(--border); border-radius: 999px; padding: 1px 9px; cursor: pointer; }
.chips button[aria-pressed="true"] { color: var(--ink); border-color: var(--ink2); font-weight: 650; }
.chart { position: relative; height: 230px; }
.chart .empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 12px; text-align: center; padding: 0 20px; }
.foot { display: flex; flex-wrap: wrap; gap: 2px 12px; font-size: 11.5px; color: var(--ink2); margin-top: 6px; font-variant-numeric: tabular-nums; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin: 0 3px 0 4px; vertical-align: 0; }
.dot.good { background: var(--good); } .dot.warn { background: var(--warn); } .dot.crit { background: var(--crit); } .dot.na { background: transparent; border: 1px solid var(--muted); }
select { font: inherit; font-size: 13px; color: var(--ink); background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 4px 6px; max-width: 100%; }
details { margin-top: 8px; font-size: 12px; color: var(--ink2); }
details pre { white-space: pre-wrap; word-break: break-word; font-size: 11.5px; background: var(--page); padding: 8px; border-radius: 6px; }
dl.defs { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; font-size: 12.5px; margin: 6px 0 0; }
dl.defs dt { font-weight: 650; }
dl.defs dd { margin: 0; color: var(--ink2); }
@media (max-width: 640px) { dl.defs { grid-template-columns: 1fr; } dl.defs dd { margin-bottom: 6px; } .controls { position: static; } }
.nochart { color: var(--crit); font-size: 12.5px; }
</style>
<script src="__CHART_JS__"></script>
</head>
<body>
<header class="wrap">
  <h1 id="title"></h1>
  <div class="sub small" id="subtitle"></div>
</header>
<div class="controls">
  <div class="wrap" style="display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;padding:0 16px;margin:0 auto;width:100%">
    <div class="ctl"><span>Concurrency</span><div class="seg" id="cSel"></div></div>
    <div class="ctl"><span>Statistic</span><div class="seg" id="statSel"></div></div>
    <div class="ctl"><span>Y axis</span><div class="seg" id="ySel"></div></div>
  </div>
</div>
<main class="wrap">
  <section class="card" id="matrixCard">
    <div class="card-head">
      <h2>Error by concurrency and metric, (sim &minus; real) / real</h2>
      <div class="seg" id="modeSel"></div>
    </div>
    <div class="caption small" id="mCaption"></div>
    <div class="key">
      <span><i class="good"></i>|error| &le; 5%</span><span><i class="warn"></i>5&ndash;15%</span><span><i class="crit"></i>&gt; 15%</span><span><i class="na"></i>&mdash; no value</span>
      <span>Click a row to open its concurrency.</span>
    </div>
    <div class="tablewrap"><table id="matrix"></table></div>
  </section>
  <section class="card" id="cellHead"></section>
  <section class="card">
    <div class="card-head"><h3 id="reqTitle">Per-request errors (one_real.py)</h3></div>
    <div id="perReq"></div>
  </section>
  <div id="noChart" class="nochart" hidden>Chart.js did not load from cdnjs, so the panels have no plots; the tables hold the same numbers.</div>
  <div class="panels" id="panels"></div>
  <section class="card">
    <div class="card-head"><h3>60 s table</h3><select id="tblSel" aria-label="Panel for the 60 s table"></select></div>
    <div class="caption small" id="tblCaption"></div>
    <div class="tablewrap"><table id="minTable"></table></div>
  </section>
  <section class="card">
    <h3>Definitions</h3>
    <dl class="defs">
      <dt>Alignment</dt><dd>The real run's series are on the wall clock, each simulated run's on its simulated clock. Each run is shifted so that 0 is the send of its first graded request (the earliest profiling-phase record with no error in its client export), and kept to <span class="win"></span> s. Points are interpolated onto one grid of the reports' step.</dd>
      <dt>Panel points</dt><dd>As in CI's report: a histogram point (latency, batch, token panels) is a histogram estimate of the statistic over the rate window before it (60 s here); the cache panel's point is the reuse ratio over that window; queue and KV curves are sampled state.</dd>
      <dt>Y axis</dt><dd>Fit cuts a panel whose peak is over 3&times; its 98th-percentile point at 1.25&times; that point and says so under the chart; Full shows every point; Log is logarithmic (zeros not drawn).</dd>
      <dt>60 s table</dt><dd>One row per minute: a histogram or cache panel's point at the minute's end (so the statistic over that minute), a queue or KV panel's mean over the minute's samples. The last row is the mean of the rows.</dd>
      <dt>Panel matrix</dt><dd>The mean of a panel's one-minute values, the simulated run's against the real run's, over the minutes both runs have a value in; <b>*</b> marks fewer than all minutes (a minute with no decode output at all has no ITL point, say). Histogram panels use the selected statistic. A mean of per-minute P99s says in which minutes the long waits sit; it is not the run's P99. The "window mean" a panel's footer shows beside each run is that run's own mean over its minutes.</dd>
      <dt>Request matrix</dt><dd><code>one_real.py</code>'s per-request metrics over the graded requests of each run (aiperf's TTFT, inter-token latency as TPOT, request latency, and InferenceX's agentic metrics). These are the figures the issues report. "in" / "OUT" marks a tail error inside or outside the real run's own 95% interval from 60 s send-block resamples; E2E paired is the median ratio over the requests both runs finished.</dd>
      <dt>Simulated GPU panels</dt><dd>A simulated run's GPU panels are the predicted target device time on TP rank 0; a real run whose collector took no samples of them shows none.</dd>
    </dl>
  </section>
  <p class="small muted" id="footer" style="margin:18px 0 30px"></p>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const STATS = ["p50", "p90", "p95", "p99", "mean"];
const STAT_LABEL = { p50: "P50", p90: "P90", p95: "P95", p99: "P99", mean: "Mean" };
const DASH = [[], [6, 4], [2, 3], [10, 3, 2, 3], [4, 2], [8, 2, 2, 2], [1, 3], [12, 4]];
const $ = (id) => document.getElementById(id);
const YMODES = { fit: "Fit", full: "Full", log: "Log" };
const st = { ci: 0, stat: "p50", mode: "panel", y: "fit", subs: {}, tbl: null };
let charts = [];

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const i = D.cells.findIndex((x) => String(x.c) === h.get("c"));
  st.ci = i >= 0 ? i : D.cells.length - 1;
  if (STATS.includes(h.get("stat"))) st.stat = h.get("stat");
  if (["panel", "request"].includes(h.get("m"))) st.mode = h.get("m");
  if (h.get("y") in YMODES) st.y = h.get("y");
}
function writeHash() {
  history.replaceState(null, "", "#c=" + D.cells[st.ci].c + "&stat=" + st.stat + "&m=" + st.mode + "&y=" + st.y);
}
// Fit: a panel whose peak is over 3x its 98th percentile point is cut at 1.25x that point.
function fitMax(datasets) {
  const v = [];
  for (const d of datasets) for (const pt of d.data) if (pt.y != null) v.push(pt.y);
  if (!v.length) return null;
  v.sort((a, b) => a - b);
  const q = v[Math.floor(0.98 * (v.length - 1))], top = v[v.length - 1];
  return q > 0 && top > 3 * q ? { cut: 1.25 * q, top: top } : null;
}
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const runColor = (i) => css("--s" + (1 + (i % 8)));
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function band(e) {
  if (e == null || !isFinite(e)) return "na";
  const a = Math.abs(e);
  return a <= D.meta.thresholds[0] + 1e-12 ? "good" : a <= D.meta.thresholds[1] + 1e-12 ? "warn" : "crit";
}
const errOf = (r, s) => (r == null || s == null || Math.abs(r) < 1e-12 ? null : (s - r) / r);
const fmtErr = (e) => (e == null ? "—" : (e >= 0 ? "+" : "−") + Math.abs(100 * e).toFixed(1) + "%");
function fmtVal(v, unit) {
  if (v == null) return "—";
  const a = Math.abs(v);
  if (unit === "ms") return a >= 1000 ? (v / 1000).toFixed(a >= 10000 ? 1 : 2) + " s" : v.toFixed(a < 10 ? 2 : 1) + " ms";
  if (unit === "%") return v.toFixed(1) + "%";
  if (unit === "tokens") return a >= 1000 ? (v / 1000).toFixed(a >= 100000 ? 0 : 1) + "k" : v.toFixed(0);
  if (unit === "requests") return v.toFixed(a < 10 ? 2 : 1);
  return String(+v.toPrecision(4));
}
function fmtAxis(v, unit) {
  const a = Math.abs(v);
  if (unit === "ms") return a >= 1000 ? +(v / 1000).toPrecision(3) + " s" : +v.toPrecision(3) + " ms";
  if (unit === "tokens") return a >= 1000 ? +(v / 1000).toPrecision(3) + "k" : String(v);
  if (unit === "%") return v + "%";
  return String(+v.toPrecision(4));
}
// one_real.py's metrics: value formatting by name.
function fmtReq(name, v) {
  if (v == null) return "—";
  if (name === "requests" || name === "concurrency") return String(Math.round(v));
  if (/ ms$/.test(name)) return fmtVal(v, "ms");
  if (name === "p90 TTFT") return fmtVal(v * 1000, "ms");
  if (name === "median TPOT") return fmtVal(v * 1000, "ms");
  if (/hit rate/.test(name)) return (100 * v).toFixed(1) + "%";
  if (name === "throughput per GPU") return v >= 1000 ? (v / 1000).toFixed(1) + "k tok/s" : v.toFixed(0) + " tok/s";
  return String(+v.toPrecision(4));
}
const subOf = (p) => (p.kind ? st.subs[p.id] || p.default_sub : st.stat);
// [real mean, sim mean, minutes] over the minutes both runs have a value in.
const pairOf = (run, pid, sub) => ((run.pairs || {})[pid] || {})[sub] || null;
const nMin = (cell) => Math.round(cell.window / 60);
function pairErr(run, pid, sub, unit, cell) {
  const pr = pairOf(run, pid, sub);
  if (!pr) return { e: null, html: fmtErr(null), title: "no minute both runs have a value in" };
  const e = errOf(pr[0], pr[1]), part = pr[2] < nMin(cell) ? "*" : "";
  return { e: e, part: part, html: fmtErr(e) + part, n: pr[2],
    title: "over the " + pr[2] + " of " + nMin(cell) + " minutes both have: real " + fmtVal(pr[0], unit) + ", " + run.label + " " + fmtVal(pr[1], unit) };
}
const subName = (p, sub) => (p.kind ? sub : STAT_LABEL[sub]);
function seg(box, items, current, onPick) {
  box.innerHTML = "";
  for (const [value, label] of items) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.setAttribute("aria-pressed", String(value === current));
    b.addEventListener("click", () => onPick(value));
    box.appendChild(b);
  }
}
function lineSample(i, sim) {
  const dash = DASH[i % DASH.length].join(" ");
  return '<svg width="30" height="10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="' + runColor(i) +
    '" stroke-width="2" stroke-linecap="round"' + (dash ? ' stroke-dasharray="' + dash + '"' : "") + "/></svg>";
}

function drawControls() {
  seg($("cSel"), D.cells.map((c, i) => [i, "c = " + c.c]), st.ci, (i) => { st.ci = i; drawAll(); });
  seg($("statSel"), STATS.map((s) => [s, STAT_LABEL[s]]), st.stat, (s) => { st.stat = s; drawAll(); });
  seg($("ySel"), Object.entries(YMODES), st.y, (y) => { st.y = y; writeHash(); drawControls(); drawPanels(); });
  seg($("modeSel"), [["panel", "Panel series (window mean)"], ["request", "Per request (one_real.py)"]], st.mode, (m) => { st.mode = m; drawAll(); });
}

function drawMatrix() {
  const t = $("matrix");
  const panelMode = st.mode === "panel";
  const cols = panelMode ? D.matrix_cols : D.request_cols;
  $("mCaption").textContent = panelMode
    ? "Each cell: the mean of a panel's one-minute values over 0–" + D.meta.window + " s, over the minutes both runs have a value in (* when fewer than all), histogram panels at " +
      STAT_LABEL[st.stat] + " (switch above). Running, waiting and KV are sampled state and ignore the statistic. Hover a cell for both values."
    : "Each cell: one_real.py's per-request metric over each run's graded requests (the issues' figures). The statistic switch does not apply.";
  let h = "<thead><tr><th>c</th><th>simulated run</th>";
  for (const col of cols) h += "<th>" + esc(col.label) + (panelMode && !col.sub ? ' <span class="tag">' + STAT_LABEL[st.stat] + "</span>" : "") + "</th>";
  h += "</tr></thead><tbody>";
  D.cells.forEach((cell, ci) => {
    const sims = cell.runs.slice(1);
    sims.forEach((run, si) => {
      h += '<tr class="clickable' + (ci === st.ci ? " sel" : "") + '" data-ci="' + ci + '">';
      if (si === 0) h += '<td rowspan="' + sims.length + '"><b>' + cell.c + "</b></td>";
      h += '<td class="lbl">' + lineSample(si + 1, true) + " " + esc(run.label) + "</td>";
      for (const col of cols) {
        let e = null, title = "", tag = "";
        if (panelMode) {
          const pr = pairOf(run, col.pid, col.sub || st.stat);
          e = pr ? errOf(pr[0], pr[1]) : null;
          title = pr ? "real " + fmtVal(pr[0], col.unit) + " · " + run.label + " " + fmtVal(pr[1], col.unit) +
            " (means over the " + pr[2] + " of " + nMin(cell) + " minutes both have)" : "no minute both runs have a value in";
          if (pr && pr[2] < nMin(cell)) tag = '<span class="tag">*</span>';
        } else if (cell.one_real) {
          const o = cell.one_real;
          if (col.key === "E2E paired") {
            const p = o.paired[run.key];
            e = p ? p.median : null;
            title = p ? "median ratio over " + p.n + " requests both ran; on totals " + fmtErr(p.totals) : "no paired E2E in one_real output";
          } else {
            const a = o.real[col.key], s = (o.sims[run.key] || {})[col.key];
            e = errOf(a, s);
            title = "real " + fmtReq(col.key, a) + " · " + run.label + " " + fmtReq(col.key, s);
            const inout = (o.tail_sims[run.key] || {})[col.key];
            if (inout) tag = ' <span class="tag' + (inout === "OUT" ? " out" : "") + '">' + inout + "</span>";
          }
        }
        h += '<td class="err ' + band(e) + '" title="' + esc(title) + '">' + fmtErr(e) + tag + "</td>";
      }
      h += "</tr>";
    });
  });
  t.innerHTML = h + "</tbody>";
  t.querySelectorAll("tr.clickable").forEach((tr) =>
    tr.addEventListener("click", () => { st.ci = +tr.dataset.ci; drawAll(); $("cellHead").scrollIntoView({ behavior: "smooth" }); }));
}

function drawCellHead() {
  const cell = D.cells[st.ci];
  let h = '<div class="card-head"><h2>c = ' + cell.c + "</h2></div>";
  if (cell.note) h += '<div class="small">' + esc(cell.note) + "</div>";
  h += '<div class="runs">';
  cell.runs.forEach((r, i) => {
    h += '<span class="run">' + lineSample(i, r.sim) + "<b>" + esc(r.label) + "</b><small>" + (r.sim ? "simulated" : "real") + " · " +
      esc(r.t0_text) + (r.graded != null ? " · " + r.graded + " graded requests" : "") + "</small></span>";
  });
  h += "</div>";
  if (cell.warnings.length) h += '<ul class="warnings small">' + cell.warnings.map((w) => "<li>" + esc(w) + "</li>").join("") + "</ul>";
  h += "<details><summary>Sources and report notes</summary>";
  for (const r of cell.runs) {
    h += "<p><b>" + esc(r.label) + "</b>: " + esc(r.report) + (r.run ? "<br>run: " + esc(r.run) : "") +
      (r.note ? "<br>" + esc(r.note) : "") + "<br>" + esc(r.report_title) + ", rate window " + esc(r.window) + " s</p>";
    if (r.report_notes && r.report_notes.length) h += "<pre>" + esc(r.report_notes.join("\n")) + "</pre>";
  }
  h += "</details>";
  $("cellHead").innerHTML = h;
}

function drawPerRequest() {
  const cell = D.cells[st.ci], o = cell.one_real, box = $("perReq");
  if (!o) { box.innerHTML = '<p class="muted small">No one_real.py output for this concurrency.</p>'; return; }
  const sims = cell.runs.slice(1);
  let h = (cell.one_real_note ? '<p class="small">' + esc(cell.one_real_note) + "</p>" : "") +
    '<div class="tablewrap"><table><thead><tr><th>metric</th><th>real</th>';
  for (const r of sims) h += "<th>" + esc(r.label) + "</th><th>error</th>";
  h += "<th>real's own 95% interval</th></tr></thead><tbody>";
  for (const m of o.metrics) {
    h += '<tr><td class="lbl">' + esc(m) + "</td><td>" + fmtReq(m, o.real[m]) + "</td>";
    for (const r of sims) {
      const row = o.sims[r.key];
      const s = row ? row[m] : null, e = errOf(o.real[m], s);
      const inout = (o.tail_sims[r.key] || {})[m];
      h += "<td>" + fmtReq(m, s) + '</td><td class="err ' + band(e) + '">' + fmtErr(e) +
        (inout ? ' <span class="tag' + (inout === "OUT" ? " out" : "") + '">' + inout + "</span>" : "") + "</td>";
    }
    const iv = o.tails[m];
    h += "<td>" + (iv ? fmtErr(iv[0]) + " .. " + fmtErr(iv[1]) : "") + "</td></tr>";
  }
  if (Object.keys(o.paired).length) {
    for (const [what, k] of [["E2E paired, median ratio", "median"], ["E2E paired, on totals", "totals"]]) {
      h += '<tr><td class="lbl">' + what + "</td><td></td>";
      let iv = "";
      for (const r of sims) {
        const p = o.paired[r.key];
        const e = p ? p[k] : null;
        if (p && k === "median") iv = fmtErr(p.interval[0]) + " .. " + fmtErr(p.interval[1]) + " (E2E p50)";
        h += '<td class="small muted">' + (p ? p.n + " req" : "") + '</td><td class="err ' + band(e) + '">' + fmtErr(e) + "</td>";
      }
      h += "<td>" + iv + "</td></tr>";
    }
  }
  h += "</tbody></table></div>";
  h += "<details><summary>one_real.py notes and files</summary><pre>" + esc(o.notes.concat([""], o.files).join("\n")) + "</pre></details>";
  box.innerHTML = h;
}

function drawPanels() {
  charts.forEach((c) => c.destroy());
  charts = [];
  const cell = D.cells[st.ci], box = $("panels");
  box.innerHTML = "";
  const hasChart = typeof Chart !== "undefined";
  $("noChart").hidden = hasChart;
  const muted = css("--muted"), grid = css("--grid"), axis = css("--axis");
  for (const p of D.panels) {
    const sub = subOf(p);
    const card = document.createElement("div");
    card.className = "card panel";
    let h = '<div class="lab">' + esc(p.label) + "</div><h3>" + esc(p.title) + ' <span class="tag">' + esc(subName(p, sub)) +
      "</span></h3><div class=\"det\">" + esc(p.detail) + "</div>";
    if (p.kind && p.subs.length > 1) {
      h += '<div class="chips">' + p.subs.map((s) => '<button type="button" data-sub="' + esc(s) + '" aria-pressed="' + (s === sub) + '">' + esc(s) + "</button>").join("") + "</div>";
    }
    h += '<div class="chart"><canvas role="img" aria-label="' + esc(p.title) + ', real against simulated"></canvas></div><div class="foot"></div>';
    card.innerHTML = h;
    box.appendChild(card);
    card.querySelectorAll(".chips button").forEach((b) => b.addEventListener("click", () => { st.subs[p.id] = b.dataset.sub; drawPanels(); drawTable(); }));
    card.querySelector(".foot").innerHTML = "<span>window mean:</span>" + cell.runs.map((r, i) => {
      const v = (r.means[p.id] || {})[sub];
      if (v == null) return "<span>" + esc(r.label) + " —</span>";
      if (i === 0) return "<span>" + esc(r.label) + " <b>" + fmtVal(v, p.unit) + "</b></span>";
      const pe = pairErr(r, p.id, sub, p.unit, cell);
      return '<span title="' + esc(pe.title) + '">' + esc(r.label) + " <b>" + fmtVal(v, p.unit) + '</b><i class="dot ' + band(pe.e) + '"></i>' + pe.html +
        (pe.part ? ' <span class="tag">over ' + pe.n + " min</span>" : "") + "</span>";
    }).join("");
    const datasets = [];
    cell.runs.forEach((r, i) => {
      const v = (r.series[p.id] || {})[sub];
      if (!v) return;
      datasets.push({
        label: r.label, runIndex: i,
        // Log has no 0; the real run drawn first, so that a simulated run on it shows.
        data: v.map((y, k) => ({ x: k * cell.step, y: st.y === "log" && y != null && y <= 0 ? null : y })),
        order: i ? 0 : 1,
        borderColor: runColor(i), backgroundColor: runColor(i), borderDash: DASH[i % DASH.length],
        borderWidth: i ? 1.6 : 2, pointRadius: 0, pointHoverRadius: 3, pointHitRadius: 6, spanGaps: false, tension: 0,
      });
    });
    const holder = card.querySelector(".chart");
    if (!datasets.length || !hasChart) {
      if (!datasets.length) holder.insertAdjacentHTML("beforeend", '<div class="empty">No series in any run.</div>');
      continue;
    }
    if (!datasets.some((d) => d.runIndex === 0)) holder.insertAdjacentHTML("afterend", '<div class="small muted">The real report has no samples for this panel.</div>');
    const fit = st.y === "fit" ? fitMax(datasets) : null;
    if (fit) holder.insertAdjacentHTML("afterend", '<div class="small muted">Y axis cut at ' + fmtVal(fit.cut, p.unit) + "; the peak, " + fmtVal(fit.top, p.unit) + ", is off the chart (Y axis: Full or Log shows it).</div>");
    const yScale = { ticks: { color: muted, font: { size: 11 }, maxTicksLimit: 6, callback: (v) => fmtAxis(v, p.unit) }, grid: { color: grid }, border: { color: axis } };
    if (st.y === "log") {
      yScale.type = "logarithmic";
      yScale.ticks.callback = (v) => { const m = v / Math.pow(10, Math.floor(Math.log10(v) + 1e-9)); return [1, 2, 5].some((k) => Math.abs(m - k) < 1e-6) ? fmtAxis(v, p.unit) : ""; };
      delete yScale.ticks.maxTicksLimit;
    } else {
      yScale.beginAtZero = true;
      if (fit) yScale.max = fit.cut;
    }
    const realSeries = (cell.runs[0].series[p.id] || {})[sub];
    charts.push(new Chart(holder.querySelector("canvas"), {
      type: "line",
      data: { datasets: datasets },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false, normalized: true,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            itemSort: (a, b) => a.dataset.runIndex - b.dataset.runIndex,
            callbacks: {
              title: (items) => (items.length ? "t = " + items[0].parsed.x + " s" : ""),
              label: (it) => {
                const y = it.parsed.y;
                let s = " " + it.dataset.label + ": " + fmtVal(y, p.unit);
                if (it.dataset.runIndex > 0 && realSeries) {
                  const e = errOf(realSeries[it.dataIndex], y);
                  if (e != null) s += "  (" + fmtErr(e) + ")";
                }
                return s;
              },
            },
          },
        },
        scales: {
          x: { type: "linear", min: 0, max: cell.window, ticks: { stepSize: 300, color: muted, font: { size: 11 } },
            grid: { color: grid }, border: { color: axis }, title: { display: true, text: "s since first graded request", color: muted, font: { size: 11 } } },
          y: yScale,
        },
      },
    }));
  }
}

function tableOptions() {
  const out = [];
  for (const p of D.panels) {
    if (p.kind) for (const s of p.subs) out.push([p.id + "|" + s, p.title + " · " + s]);
    else out.push([p.id + "|*", p.title + " · " + STAT_LABEL[st.stat]]);
  }
  return out;
}
function drawTable() {
  const sel = $("tblSel"), opts = tableOptions();
  if (!st.tbl || !opts.some((o) => o[0] === st.tbl)) st.tbl = opts[0][0];
  sel.innerHTML = opts.map(([v, l]) => '<option value="' + esc(v) + '"' + (v === st.tbl ? " selected" : "") + ">" + esc(l) + "</option>").join("");
  const [pid, s] = st.tbl.split("|");
  const p = D.panels.find((x) => x.id === pid), sub = s === "*" ? st.stat : s;
  const cell = D.cells[st.ci], sims = cell.runs.slice(1);
  const rows = (r) => (r.minutes[pid] || {})[sub];
  const real = rows(cell.runs[0]);
  $("tblCaption").textContent = p.kind && ["queues", "blocks"].includes(p.kind)
    ? "Each row: the mean of the minute's samples (sampled state)."
    : "Each row: the panel's point at the minute's end, i.e. the statistic over that minute.";
  let h = "<thead><tr><th>minute</th><th>" + esc(cell.runs[0].label) + "</th>";
  for (const r of sims) h += "<th>" + esc(r.label) + "</th><th>error</th>";
  h += "</tr></thead><tbody>";
  const n = Math.round(cell.window / 60);
  const at = (r, k) => { const m = rows(r); return m ? m[k] : null; };
  for (let k = 0; k < n; k++) {
    h += '<tr><td class="lbl">' + 60 * k + "–" + 60 * (k + 1) + " s</td><td>" + fmtVal(at(cell.runs[0], k), p.unit) + "</td>";
    for (const r of sims) {
      const v = at(r, k), e = errOf(at(cell.runs[0], k), v);
      h += "<td>" + fmtVal(v, p.unit) + '</td><td class="err ' + band(e) + '">' + fmtErr(e) + "</td>";
    }
    h += "</tr>";
  }
  h += '<tr class="total"><td class="lbl">window mean</td><td>' + fmtVal((cell.runs[0].means[pid] || {})[sub], p.unit) + "</td>";
  for (const r of sims) {
    const pe = pairErr(r, pid, sub, p.unit, cell);
    h += "<td>" + fmtVal((r.means[pid] || {})[sub], p.unit) + '</td><td class="err ' + band(pe.e) + '" title="' + esc(pe.title) + '">' + pe.html + "</td>";
  }
  h += "</tr>";
  $("minTable").innerHTML = h + "</tbody>";
  if (!real) $("tblCaption").textContent += " The real report has no samples for this panel.";
}

function drawAll() {
  writeHash();
  drawControls();
  drawMatrix();
  drawCellHead();
  drawPerRequest();
  drawPanels();
  drawTable();
}

$("title").textContent = D.meta.title;
document.title = D.meta.title;
$("subtitle").textContent = "Generated " + D.meta.generated + " · window 0–" + D.meta.window +
  " s from each run's first graded request · solid = real, dashed = simulated · concurrencies: " + D.cells.map((c) => c.c).join(", ");
document.querySelectorAll(".win").forEach((e) => (e.textContent = D.meta.window));
$("footer").textContent = "Manifest: " + D.meta.manifest + " · scripts/compass/compare_dashboard.py";
$("tblSel").addEventListener("change", (e) => { st.tbl = e.target.value; drawTable(); });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawPanels(); drawCellHead(); drawMatrix(); });
readHash();
drawAll();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
