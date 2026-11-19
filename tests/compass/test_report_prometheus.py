# SPDX-License-Identifier: MIT
"""A simulated run's Prometheus report, `python -m atom.compass.report.prometheus
OUT`: the series each LP's sampler writes (`metrics.Sampler`), backfilled by
promtool and served by Prometheus to CI's report, both tools stubbed, and
Prometheus's query API too, as tests/test_observability_report.py stubs it;
and the report `client.sh --stop` and `deploy.sh` write as a run ends.
"""

import gzip
import io
import json
import re
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from prometheus_client.core import GaugeMetricFamily, HistogramMetricFamily
from test_client_sh import COMMAND, GOOD, _client, _setup
from test_deploy_sh import _until, argv, planted  # noqa: F401  the planted fixture
from test_report_split import _deployment

from atom.compass.report import html, prometheus
from atom.compass.run import metrics
from atom.utils import clock

#: A deployment's run file, two DP members a side.
PD = {"router_s": 2.0**-12, "kv_write_req_s": 2.0**-10}
PD |= {"prefill_data_parallel_size": 2, "decode_data_parallel_size": 2}


def _families(t: float) -> list:
    """What an engine's and a frontend's processes hold at `t`."""
    queue = HistogramMetricFamily("atom:request_queue_time_seconds", "Queue.")
    queue.add_metric([], [("0.1", t), ("+Inf", 2 * t)], sum_value=t)
    requests = GaugeMetricFamily(
        "atom:scheduler_requests", "Requests.", labels=["state"]
    )
    requests.add_metric(["running"], t)
    return [queue, requests]


def _series(monkeypatch, out: Path, lp: str, role: str, times) -> None:
    """LP `lp` of the side of `role`, sampled at each of `times`, writes its
    file in `out`, as it does when it hands in for the last time."""
    rt = SimpleNamespace(in_run=True, now=0.0, calls=0, refusals=[], me=lp)
    kind = "frontend" if lp.startswith("frontend") else "engine"
    instance = {"prefill": "frontend-P", "decode": "frontend-D"}.get(role, "frontend")
    sampler = metrics.Sampler(rt, kind, role, instance)
    with monkeypatch.context() as m:
        m.setattr(clock, "installed", lambda: rt)
        for t in times:
            rt.now = t
            sampler.sample(lambda: _families(rt.now))
    sampler.write(out, lp)


def _out(tmp_path, run_file: dict | None) -> Path:
    """A stopped run's directory, simulated from `run_file`, or real."""
    out = tmp_path / "out"
    out.mkdir()
    meta = {"argv": ["--", "--model", "m"]}
    if run_file is not None:
        meta["run_file"] = run_file | {"bound_s": 100.0, "out_dir": str(out)}
    (out / "run.json").write_text(json.dumps(meta))
    if run_file is not None and "router_s" in run_file:
        (out / "prefill").mkdir(), (out / "decode").mkdir()
    return out


@pytest.fixture
def served(monkeypatch, tmp_path):
    """promtool and Prometheus stubbed, the files promtool backfilled kept in
    ``seen``, and Prometheus's HTTP API stubbed, each range query it answers
    kept in ``queries``, as its URL's parameters."""
    bindir, seen = tmp_path / "bin", tmp_path / "seen"
    bindir.mkdir(), seen.mkdir()
    tools = {
        # tsdb create-blocks-from openmetrics INPUT OUTPUT
        "promtool": f'#!/bin/sh\ncp "$4" {seen}/\n',
        "prometheus": "#!/bin/sh\necho 'msg=Listening address=127.0.0.1:9090'\n"
        "exec sleep 60\n",
    }
    for name, body in tools.items():
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
        monkeypatch.setenv(f"ATOMESH_{name.upper()}_BIN", str(bindir / name))
    monkeypatch.setattr(prometheus, "TOOLS", tmp_path / "cache")
    monkeypatch.syspath_prepend(str(prometheus.CI))
    import collect_metrics

    queries = []

    def urlopen(request, timeout):
        url = urlsplit(getattr(request, "full_url", request))
        result = {"activeTargets": []}
        if url.path == "/api/v1/query_range":
            asked = {k: v[0] for k, v in parse_qs(url.query).items()}
            queries.append(asked)
            point = [float(asked["start"]), "1"]
            result = {"result": [{"metric": {"instance": "i"}, "values": [point]}]}
        body = {"status": "success", "data": result}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(collect_metrics, "urlopen", urlopen)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return SimpleNamespace(queries=queries, seen=seen)


def _windows(queries) -> set:
    return {w for q in queries for w in re.findall(r"\[(\d+s)\]", q["query"])}


def _roles(queries) -> set:
    return {r for q in queries for r in re.findall(r'role="(\w+)"', q["query"])}


def test_an_instance_s_report_spans_its_series_with_the_standalone_panels(
    tmp_path, monkeypatch, served
):
    out = _out(tmp_path, {})
    _series(monkeypatch, out, "engine", "standalone", [0.5 + k for k in range(10)])
    _series(monkeypatch, out, "frontend", "standalone", [1.0 + k for k in range(12)])
    (out / prometheus.TEXT).write_text("an earlier report's")

    assert prometheus.write(out) == out / prometheus.HTML

    files = sorted(out.glob(metrics.FILE.format("*")))
    assert {p.name: p.read_bytes() for p in served.seen.iterdir()} == {
        p.name.removesuffix(".gz"): gzip.decompress(p.read_bytes()) for p in files
    }
    assert {(q["start"], q["end"]) for q in served.queries} == {("0.5", "15.5")}
    stepped = [q for q in served.queries if "max_over_time" not in q["query"]]
    assert {q["step"] for q in stepped} == {"5"}
    assert _windows(served.queries) == {"60s"}
    assert _roles(served.queries) == {"standalone"}
    data = json.loads((out / prometheus.DATA).read_text())
    assert data["meta"]["title"] == "Simulated agentic aggregated inference report"
    assert data["meta"]["notes"][:2] == prometheus.NOTES
    assert prometheus.OVERALL_TTFT not in data["meta"]["notes"]
    gpu = [p for p in data["panels"] if p["metric"] in prometheus.PREDICTED]
    assert {p["label"] for p in gpu} == {"STANDALONE · PREDICTED GPU"}
    assert all(p["detail"] == prometheus.PREDICTED[p["metric"]] for p in gpu)
    assert not (out / prometheus.TEXT).exists()
    assert f'<a href="{prometheus.HTML}">' in html.page(out)


def test_a_deployment_s_report_shows_both_sides_and_names_what_it_lacks(
    tmp_path, monkeypatch, served
):
    """Samples 20 s apart widen the window to 80 s. One engine member wrote no
    file, and a family the panels query is in no sampler's class table."""
    engine = dict(metrics.FAMILIES["engine"])
    del engine["atom:decode_batch_size"]
    monkeypatch.setitem(metrics.FAMILIES, "engine", engine)
    out = _out(tmp_path, PD)
    times = [20.0 * k for k in range(1, 8)]
    for lp in ("frontend-P", "engine-P.dp0", "engine-P.dp1"):
        _series(monkeypatch, out, lp, "prefill", times)
    for lp in ("frontend-D", "engine-D.dp0"):
        _series(monkeypatch, out, lp, "decode", times)

    prometheus.write(out)

    assert {(q["start"], q["end"]) for q in served.queries} == {("20.0", "140.0")}
    assert _windows(served.queries) == {"80s"}
    assert _roles(served.queries) == {"prefill", "decode"}
    assert any("mesh_router_ttft_seconds" in q["query"] for q in served.queries)
    meta = json.loads((out / prometheus.DATA).read_text())["meta"]
    assert meta["title"] == "Simulated agentic PD inference report"
    assert prometheus.OVERALL_TTFT in meta["notes"]
    assert "No series from engine-D.dp1: none written." in meta["notes"]
    lacking = "No sampler writes atom:decode_batch_size: no series to query."
    assert lacking in meta["notes"]


@pytest.mark.parametrize(
    "times, end, window",
    [
        # a final sample 0.188 s after the timer's last, at a 1 s and a 20 s timer
        ([1.0 + k for k in range(11)] + [11.188], "16.0", "60s"),
        ([20.0 * k for k in range(1, 8)] + [140.188], "145.0", "80s"),
        # the last instant on the grid, and one instant alone
        ([1.0 + k for k in range(11)], "11.0", "60s"),
        ([3.0], "4.0", "60s"),
    ],
)
def test_the_query_ends_at_the_grid_point_at_or_past_the_last_instant(
    tmp_path, monkeypatch, served, times, end, window
):
    """Each file's last step, its final sample's, is shorter than its
    interval, and leaves the window at max(60, 4 x interval)."""
    out = _out(tmp_path, {})
    for lp in ("engine", "frontend"):
        _series(monkeypatch, out, lp, "standalone", times)

    assert prometheus.write(out) == out / prometheus.HTML

    assert {(q["start"], q["end"]) for q in served.queries} == {(str(times[0]), end)}
    assert _windows(served.queries) == {window}


@pytest.mark.parametrize(
    "case, why",
    [
        ("a real run", "a real run, on which no series is sampled"),
        ("no series", "no engine or frontend wrote its series"),
        ("no promtool", "FileNotFoundError: promtool executable not found"),
        ("no CI scripts", "this tree holds no"),
    ],
)
def test_without_a_report_its_text_says_why_and_report_html_quotes_it(
    tmp_path, monkeypatch, served, case, why
):
    out = _out(tmp_path, None if case == "a real run" else {})
    if case != "no series":
        _series(monkeypatch, out, "frontend", "standalone", [1.0, 2.0])
    if case == "no promtool":
        monkeypatch.setenv("ATOMESH_PROMTOOL_BIN", str(tmp_path / "absent"))
    if case == "no CI scripts":
        monkeypatch.setattr(prometheus, "CI", tmp_path / "absent")
    (out / prometheus.HTML).write_text("an earlier report")

    assert prometheus.write(out) == out / prometheus.TEXT

    text = (out / prometheus.TEXT).read_text()
    assert text.startswith(f"No Prometheus report: {why}")
    assert not (out / prometheus.HTML).exists() and not served.queries
    assert html.escape(text.strip()) in html.page(out)


@pytest.mark.parametrize("unwritable", [False, True])
def test_client_sh_stop_writes_it_before_report_html_and_its_failure_fails_no_stop(
    tmp_path, unwritable
):
    """The stand-in server is a real run's: the module says so, and when it
    cannot even write that, the stop still exits 0 and names the failure."""
    out, env = _setup(tmp_path)
    if unwritable:
        (out / prometheus.TEXT).mkdir()
    (tmp_path / "compass_command.py").write_text(COMMAND)
    command = subprocess.Popen([sys.executable, tmp_path / "compass_command.py", out])
    while not (out / "server.json").exists():
        assert command.poll() is None
        time.sleep(0.1)

    done = _client(out, *GOOD, "--stop", env=env)

    assert done.returncode == 0, done.stdout + done.stderr
    failed = "client: prometheus: atom.compass.report.prometheus exited 1"
    assert (failed in done.stderr) == unwritable
    page = (out / html.FILE).read_text()
    assert ("No Prometheus report: a real run" in page) != unwritable


def test_deploy_sh_writes_it_before_report_html(tmp_path, planted):  # noqa: F811
    env, flags = planted
    d = tmp_path / "d"
    script = subprocess.Popen(
        argv(tmp_path, flags), env=env, cwd=tmp_path, stdout=subprocess.PIPE, text=True
    )
    _until(d / "server.json", script)
    real = tmp_path / "real"
    _deployment(real, d=0.0, simulated=True)
    for side in ("prefill", "decode"):
        (real / side / "steps").rename(d / side / "steps")
    (real / "artifacts").rename(d / "artifacts")

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)

    assert script.returncode == 0, out
    why = (d / prometheus.TEXT).read_text()
    assert why.startswith("No Prometheus report: no engine or frontend wrote")
    assert html.escape(why.strip()) in (d / html.FILE).read_text()


def test_both_tools_come_from_one_archive_and_each_is_reused_once_extracted(
    tmp_path, monkeypatch
):
    """Neither named nor on PATH, each tool is extracted from the release
    archive already in the directory, which no call downloads; once
    extracted, each is reused with the archive gone, and no part is left."""
    monkeypatch.syspath_prepend(str(prometheus.CI))
    import collect_metrics

    for name in ("ATOMESH_PROMETHEUS_BIN", "ATOMESH_PROMTOOL_BIN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(collect_metrics.platform, "system", lambda: "Linux")
    monkeypatch.setattr(collect_metrics.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(collect_metrics, "urlopen", None)  # a download fails
    stem = f"prometheus-{collect_metrics.PROMETHEUS_VERSION}.linux-amd64"
    archive = tmp_path / f"{stem}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for tool in ("prometheus", "promtool"):
            body = f"#!/bin/sh\necho {tool}\n".encode()
            member = tarfile.TarInfo(f"{stem}/{tool}")
            member.size = len(body)
            tar.addfile(member, io.BytesIO(body))

    made = {
        t: collect_metrics.ensure_prometheus(tmp_path, t)
        for t in ("promtool", "prometheus")
    }
    archive.unlink()

    for tool, path in made.items():
        assert path == str(tmp_path / tool)
        ran = subprocess.run([path], capture_output=True, text=True, check=True)
        assert ran.stdout == f"{tool}\n"
        assert collect_metrics.ensure_prometheus(tmp_path, tool) == path
    assert sorted(p.name for p in tmp_path.iterdir()) == ["prometheus", "promtool"]
