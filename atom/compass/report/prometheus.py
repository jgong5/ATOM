# SPDX-License-Identifier: MIT
"""A simulated run's Prometheus report, ATOM's CI report over its series:
`python -m atom.compass.report.prometheus OUT`

OUT is a run's directory as `split` reads it. Each engine member and frontend
LP of a simulated run wrote its series there as it handed in (`metrics.FILE`).
The report is CI's, from this tree's `CI` scripts imported unchanged: its page
goes to OUT/`HTML` and its data, which ``export_report.py --input-json``
renders again, to OUT/`DATA`.

- Each file is decompressed, since promtool reads no gzip, and backfilled by
  promtool into one TSDB, which a Prometheus with no scrape serves to
  ``export_report.collect_report``. ``collect_metrics.ensure_prometheus``
  finds both tools: ATOMESH_PROMTOOL_BIN and ATOMESH_PROMETHEUS_BIN, else on
  PATH, else extracted into `TOOLS` from the release archive it downloads
  once.
- The panels are those of the run's deployment kind, ``pd`` when OUT holds a
  deployment's two sides, else ``standalone``, from the series' first instant
  to the first grid point at or past their last, as CI's
  ``wait_for_final_scrape`` ends its query, at CI's step and window:
  ``max(60, 4 x interval)``, the interval the shortest between two instants of
  one file at ATOM's metrics timer, each file's last step left out, since its
  final sample lands under one interval after the timer's last.
- It is titled simulated, and its notes say the times are the simulated
  clock's and the GPU panels' predicted, name each engine member or frontend
  that wrote no file and each family the panels query that no sampler writes,
  and say why a deployment's Overall TTFT is empty. The two GPU panels' label
  and description say predicted.

When it cannot be made, OUT/`TEXT` says why: a real run, no series, no tool,
no ``export_report.py`` in this tree, or a backfill or query that failed.
Either way it leaves none of the three an earlier report wrote, and exits 0
unless OUT/`TEXT` itself cannot be written.
"""

import gzip
import itertools
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from atom.compass.backends.json_file import read_json
from atom.compass.run import LMCACHE, _hand_ins, metrics
from atom.compass.validate.runs import sides

HTML, DATA, TEXT = "prometheus.html", "prometheus.json", "prometheus.txt"
#: CI's observability scripts, in this tree.
CI = Path(__file__).resolve().parents[3] / ".github/scripts/atomesh/observability"
#: Where the tools are extracted, each version in its own directory.
TOOLS = Path("~/.cache/compass")
TITLES = {
    "pd": "Simulated agentic PD inference report",
    "standalone": "Simulated agentic aggregated inference report",
}
#: The description of each GPU forward panel, by its family.
PREDICTED = {
    "atom:gpu_forward_seconds": "Predicted target device time of each priced"
    " forward step · observed on TP rank 0 alone · not measured",
    "atom:prefill_request_gpu_forward_seconds": "Sum of predicted batch times"
    " across initial prefill chunks · once per completed request, on TP rank 0"
    " alone · not measured",
}
NOTES = [
    (
        "A simulated run: every time is its simulated clock's, seconds from the"
        " run's start, shown from 1970-01-01 00:00 UTC. Each engine and frontend"
        " sampled its own series on its own clock at ATOM's metrics timer."
    ),
    (
        "The GPU panels show each priced step's predicted target device time,"
        " observed on TP rank 0 alone, not a measured one."
    ),
]
OVERALL_TTFT = (
    "Overall TTFT has no series: the Atomesh router times"
    " mesh_router_ttft_seconds on the wall clock. report.html carries the"
    " client's TTFT."
)


class Unmade(Exception):
    """Why the report cannot be made."""


def write(out) -> Path:
    """Write OUT's report, or why there is none; the file written."""
    out = Path(out)
    try:
        for name in (HTML, DATA, TEXT):
            (out / name).unlink(missing_ok=True)
        return _report(out)
    # Whatever stops the report, the run's stop goes on, and TEXT says why.
    except Exception as e:  # noqa: BLE001
        why = str(e) if isinstance(e, Unmade) else f"{type(e).__name__}: {e}"
        (out / TEXT).write_text(f"No Prometheus report: {why}\n")
        return out / TEXT


def _report(out: Path) -> Path:
    run = read_json(out / "run.json").get("run_file")
    if run is None:
        raise Unmade(
            "a real run, on which no series is sampled; CI's collector,"
            " collect_metrics.py, reports a real run as it runs"
        )
    files = sorted(out.glob(metrics.FILE.format("*")))
    if not files:
        raise Unmade(f"no engine or frontend wrote its series in {out}")
    collect_metrics, export_report = _ci()
    tools = TOOLS.expanduser() / f"prometheus-{collect_metrics.PROMETHEUS_VERSION}"
    tools.mkdir(parents=True, exist_ok=True)
    promtool = collect_metrics.ensure_prometheus(tools, "promtool")
    prometheus = collect_metrics.ensure_prometheus(tools)
    deployment = "pd" if len(sides(out)) > 1 else "standalone"
    notes = [*NOTES, OVERALL_TTFT] if deployment == "pd" else [*NOTES]
    names = [n for n in _hand_ins(run) if n != LMCACHE.name]
    if missing := [n for n in names if not (out / metrics.FILE.format(n)).is_file()]:
        notes.append(f"No series from {', '.join(missing)}: none written.")
    with tempfile.TemporaryDirectory(prefix="compass-prometheus-") as work:
        work = Path(work)
        first, last, interval = _backfill(promtool, files, work / "data")
        step = collect_metrics.REPORT_STEP
        # query_range evaluates first + n x step: the end is the first of those
        # at or past the last instant, and past first, as collect_report asks.
        end = max(first + math.ceil((last - first) / step) * step, first + 1)
        window = max(60, math.ceil(4 * interval))
        notes.append(
            f"Histogram points summarize the preceding {window} seconds; queue"
            " and KV curves show sampled state. Percentiles are histogram"
            " estimates."
        )
        (work / "prometheus.yml").write_text(json.dumps({"scrape_configs": []}))
        errors = []
        with (work / "prometheus.log").open("w") as log:
            server = subprocess.Popen(
                [
                    prometheus,
                    f"--config.file={work / 'prometheus.yml'}",
                    f"--storage.tsdb.path={work / 'data'}",
                    "--web.listen-address=127.0.0.1:0",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                url = collect_metrics.wait_for_prometheus(
                    server, work / "prometheus.log", 0
                )
                data = export_report.collect_report(
                    url,
                    first,
                    end,
                    deployment=deployment,
                    step=step,
                    window=window,
                    title=TITLES[deployment],
                    diagnostics=errors,
                )
            finally:
                collect_metrics.stop_process(server)
    written = set().union(*metrics.FAMILIES.values())
    queried = {
        re.sub(r"_(bucket|count|sum|total)$", "", name)
        for panel in data["panels"]
        for query in panel["queries"].values()
        for name in re.findall(r"atom:\w+", query)
    }
    if lacking := sorted(queried - written):
        notes.append(f"No sampler writes {', '.join(lacking)}: no series to query.")
    for panel in data["panels"]:
        if panel["metric"] in PREDICTED:
            panel["label"] = f"{panel['role'].upper()} · PREDICTED GPU"
            panel["detail"] = PREDICTED[panel["metric"]]
    status = {"errors": errors, "benchmark_exit_code": None}
    collect_metrics.finalize_report(data, status, notes)
    export_report.write_report(data, out / HTML)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    (out / DATA).write_text(text)
    return out / HTML


def _ci():
    """CI's ``collect_metrics`` and ``export_report`` modules, from `CI`."""
    if not (CI / "export_report.py").is_file():
        raise Unmade(f"this tree holds no {CI / 'export_report.py'}")
    if str(CI) not in sys.path:
        sys.path.insert(0, str(CI))
    import collect_metrics
    import export_report

    return collect_metrics, export_report


def _backfill(promtool: str, files: list, tsdb: Path) -> tuple:
    """Backfill each of `files` into `tsdb`, decompressed; the first and last
    instant of their samples, and the shortest step between two instants of
    one file but its last step, its final sample's, 0 when none holds three."""
    first, last, steps = math.inf, -math.inf, []
    for path in files:
        text = tsdb.with_name(path.stem)
        stamps = set()
        with gzip.open(path) as f, text.open("wb") as to:
            for line in f:
                to.write(line)
                if line[:1] != b"#":
                    stamps.add(float(line[line.rindex(b" ") + 1 :]))
        done = subprocess.run(
            [promtool, "tsdb", "create-blocks-from", "openmetrics", text, tsdb],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode:
            raise Unmade(f"promtool could not backfill {path.name}: {done.stderr}")
        text.unlink()
        instants = sorted(stamps)
        if instants:
            first, last = min(first, instants[0]), max(last, instants[-1])
            steps += [b - a for a, b in itertools.pairwise(instants[:-1])]
    if first > last:
        raise Unmade("no engine or frontend sampled any series")
    return first, last, min(steps, default=0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m atom.compass.report.prometheus OUT")
    write(sys.argv[1])
