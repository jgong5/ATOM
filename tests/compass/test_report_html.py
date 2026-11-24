# SPDX-License-Identifier: MIT
"""A run's report, `python -m atom.compass.report.html OUT`, rendered from run
directories laid out as `compass run`, `compass record`, `deploy.sh` and
`client.sh` write them, as the split's and the curves' tests build them, with
a run summary the run's own writer wrote; and the report `client.sh --stop`
and `deploy.sh` write as a run ends.
"""

import json
import signal
import subprocess
import sys
import time
from html.parser import HTMLParser

import pytest
from test_client_sh import COMMAND, GOOD, _client, _setup
from test_deploy_sh import _until, argv, planted  # noqa: F401  the planted fixture
from test_report_split import NODE_S, _deployment, _pairs, _two_nodes
from test_run_refusal_summary import STAGE1, AnalyticalBelow, summary
from test_validate_curves import _counted, _run

from atom.compass.report import html, split
from atom.compass.run import SUMMARY_FILE
from atom.compass.validate import curves

SOURCE = {
    "name": "atom",
    "root": "/t/atom",
    "revision": "c0ffee",
    "revision_kind": "commit",
}
SOURCE |= {"method": "git", "dirty": 0}
STACK = {"rocm": "7.2.4", "aiter": "41d4d8d7e"}


class _Page(HTMLParser):
    """Each section's tables, rows of cell texts, and paragraphs, by its title."""

    def __init__(self, run):
        super().__init__()
        self.parts, self._title, self._text = {}, None, None
        self.feed((run / html.FILE).read_text())

    def handle_starttag(self, tag, attrs):
        if tag in ("h2", "th", "td", "p"):
            self._text = ""
        elif tag == "table":
            self.parts[self._title]["tables"].append([])
        elif tag == "tr":
            self.parts[self._title]["tables"][-1].append([])

    def handle_data(self, data):
        if self._text is not None:
            self._text += data

    def handle_endtag(self, tag):
        if tag == "h2":
            self._title = self._text
            self.parts[self._title] = {"tables": [], "text": []}
        elif tag in ("th", "td"):
            self.parts[self._title]["tables"][-1][-1].append(self._text)
        elif tag == "p":
            self.parts[self._title]["text"].append(self._text)
        if tag in ("h2", "th", "td", "p"):
            self._text = None

    def rows(self, title, k=0) -> dict:
        """Table `k` of section `title`, by its first column."""
        return {r[0]: r[1:] for r in self.parts[title]["tables"][k]}


def _shown(m: dict) -> dict:
    """`curves.metrics`' values `m` as the report shows them, by label."""
    out = {
        label: [f"{m[n] * scale:.4g}" if m[n] is not None else "none"]
        for n, (label, scale) in curves.AXES.items()
    }
    return out | {
        n: [f"{m[n] * 100:.4g}%" if m[n] is not None else "none"] for n in html.RATES
    }


def _files(run) -> dict:
    return {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_a_simulated_instance_s_report_reads_its_metrics_speed_and_trust_from_its_files(
    tmp_path, monkeypatch
):
    run = _counted(_run(tmp_path / "run", 4, server=False))
    (tmp_path / "authority").mkdir()
    priced = summary(
        monkeypatch, tmp_path / "authority", AnalyticalBelow(STAGE1, 40), 8, 8, 8, 40
    )
    (run / SUMMARY_FILE).write_text(json.dumps(priced))
    run_file = {"step_stub": {"decode_request": 0.0}, "store": "/s"}
    run_file |= {"backend": {"digest": "sha256:ab", "provenance": None}}
    meta = {"argv": ["-c", "run", "--out", str(run), "--", "--model", "m"]}
    (run / "run.json").write_text(json.dumps(meta | {"run_file": run_file}))
    split.write(run)

    page = _Page(html.write(run).parent)

    metrics = page.rows("Headline metrics")
    # The writer measured no wall time.
    assert metrics.pop("speed ratio") == [priced["cost"]["speed_refused"]]
    assert metrics == _shown(curves.metrics(run, simulated=True))
    assert metrics["GPU hit rate"] == [f"{41900 / 60000 * 100:.4g}%"]
    trust = page.rows("Trust")
    assert trust["steps priced"] == ["4"]
    assert trust["steps extrapolated"] == ["0, 0% of the steps priced"]
    # The 40-token prefill, answered after two sources above declined it.
    assert trust["steps a fallback priced"] == ["1, 25% of the steps priced"]
    assert trust["coverage report"] == ["no"]
    # One prefill of 40 tokens priced analytically, three of 8 declared.
    methods = page.rows("Trust", 1)
    assert methods["method"] == ["steps", "share of predicted seconds"]
    for method in ("analytical", "declared"):
        share = priced["schedule"]["seconds_by_method"][method]
        steps, fraction = share["steps"], share["fraction_of_predicted_seconds"]
        assert methods[method] == [str(steps), f"{fraction * 100:.4g}%"]
    assert page.rows("Run")["run"] == ["simulated, step_stub"]
    assert page.rows("Run")["ATOM arguments"] == ["--model m"]
    assert page.rows("Law sources")["backend digest"] == ["sha256:ab"]
    steps = run / "steps"
    assert page.parts["Per-request split"]["text"] == [
        f"None: {steps} holds no DP rank's step record"
    ]
    for title in ("Grades", "A real run's per-request split", "Op-family split"):
        assert page.parts[title]["text"][0].startswith("None: "), title
    timed = {"speed_ratio": 2.5, "simulated_seconds": 10.0, "wall_seconds": 4.0}
    priced["cost"] |= timed | {"speed_refused": None}
    (run / SUMMARY_FILE).write_text(json.dumps(priced))
    ratio = "2.5: 10 simulated seconds in 4 wall seconds"
    assert _Page(html.write(run).parent).rows("Headline metrics")["speed ratio"] == [
        ratio
    ]


def test_a_real_deployment_s_report_shows_both_sides_and_its_split_and_changes_no_other_file(
    tmp_path,
):
    run = tmp_path / "run"
    _deployment(run)
    # Request B is the warmup's, which neither the metrics nor the table count.
    rows = [
        json.loads(line) for line in (run / curves.RECORDS).read_text().splitlines()
    ]
    rows[1]["metadata"]["benchmark_phase"] = "warmup"
    (run / curves.RECORDS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    split.write(run)
    for side, args in (("prefill", ["--enable-tbo"]), ("decode", [])):
        own = {"argv": ["-c", "record", "--", "--model", "m", *args]}
        own |= {"source": SOURCE, "device": [{"name": "MI355X", "arch": "gfx950"}] * 8}
        own |= {"ptl": "disabled", "stack": STACK, "host_key": {"host_cpu": "EPYC"}}
        (run / side / "run.json").write_text(json.dumps(own))
        engine = {"tensor_parallel_size": 1, "data_parallel_size": 8}
        (run / side / curves.ENGINE_RECORD).write_text(json.dumps(engine))
    given = {"arguments": {curves.CONCURRENCY: 2}, "traces": ["t0", "t1"]}
    (run / curves.CLIENT_RUN).write_text(json.dumps(given))
    before = _files(run)

    page = _Page(html.write(run).parent)

    assert _files(run) == before | {run / html.FILE: (run / html.FILE).read_bytes()}
    shown = page.rows("Run")
    assert shown["run"] == ["real"] and shown["deployment"] == [
        "1P1D, KV link inter_node"
    ]
    assert shown["prefill ATOM arguments"] == ["--model m --enable-tbo"]
    assert shown["decode ATOM tree"] == ["commit c0ffee, clean, at /t/atom"]
    machine = (
        "8 x MI355X (gfx950); PTL disabled; stack rocm 7.2.4, aiter 41d4d8d7e; EPYC"
    )
    assert shown["decode machine"] == [machine]
    assert page.rows("Workload")[curves.CONCURRENCY] == ["2"]
    assert page.rows("Workload")["traces"] == ["2"]
    assert page.rows("Headline metrics") == _shown(curves.metrics(run))
    made = json.loads((run / split.FILE).read_text())
    a, b = made["requests"]
    assert (a["phase"], b["phase"]) == ("profiling", "warmup")
    low, high = made["decode_offset_s"]
    joined = f"the decode side's joined to it at {high:.4f} s, its bounds {low:.4f} s"
    joined += (
        f" to {high:.4f} s; 3 not. Below, the 1 of them the headline metrics count."
    )
    assert page.parts["Per-request split"]["text"] == [
        f"2 requests split, on the monotonic clock, {joined}"
    ]
    segments = page.rows("Per-request split")
    for key, name in html.SEGMENTS.items():
        assert segments[name] == ["1", *[f"{a[key]:.4g}"] * 3], name
    assert segments["queue"] == ["1", "0.2", "0.2", "0.2"]
    assert page.rows("Per-request split", 1) == {
        "LMCache": ["tokens"],
        "loaded": ["64"],
        "stored": ["136"],
    }
    assert sorted(page.rows("Per-request split", 2)) == [
        "aiperf records it error", "it got no response id",
        "no step names it on every side", "not split because",
    ]  # fmt: skip
    assert page.parts["Trust"]["text"] == ["None: a real run is measured, not priced."]
    assert page.parts["Law sources"]["text"] == ["None: a real run prices nothing."]
    assert "A real run's per-request split" not in page.parts


def test_a_two_node_deployment_s_report_states_the_offset_joining_its_second_node(
    tmp_path,
):
    run = tmp_path / "run"
    _two_nodes(run)
    split.write(run)

    page = _Page(html.write(run).parent)

    joined = "10.0.0.2's instances joined to it at -1.0560 s, its bounds -1.2560 s"
    joined += " to -1.0560 s; 0 not. Below, the 4 of them the headline metrics count."
    assert page.parts["Per-request split"]["text"] == [
        f"4 requests split, on the monotonic clock, {joined}"
    ]


def test_a_two_node_deployment_s_report_states_the_measured_clock_of_its_second_node(
    tmp_path,
):
    run = tmp_path / "run"
    _two_nodes(run)
    _pairs(run, [(0.0, NODE_S), (4.0, NODE_S)])
    split.write(run)

    page = _Page(html.write(run).parent)

    joined = "10.0.0.2's instances joined to it by 2 measured offsets, -1.0560 s"
    joined += " to -1.0560 s, each within 0.020 ms, which put 0 requests out of"
    joined += " order, its bounds -1.2560 s to -1.0560 s; 0 not. Below, the 4 of"
    joined += " them the headline metrics count."
    assert page.parts["Per-request split"]["text"] == [
        f"4 requests split, on the monotonic clock, {joined}"
    ]


def test_client_sh_stop_writes_an_instances_report_after_its_split(tmp_path):
    """The stand-in command records no steps and its harness exports no
    records: the report says so in each part."""
    out, env = _setup(tmp_path)
    (tmp_path / "compass_command.py").write_text(COMMAND)
    command = subprocess.Popen([sys.executable, tmp_path / "compass_command.py", out])
    while not (out / "server.json").exists():
        assert command.poll() is None
        time.sleep(0.1)

    done = _client(out, *GOOD, "--stop", env=env)

    assert done.returncode == 0, done.stdout + done.stderr
    page = _Page(out)
    assert page.parts["Per-request split"]["text"] == [
        f"None: {out / 'steps'} holds no DP rank's step record"
    ]
    records = out / curves.RECORDS
    assert page.parts["Headline metrics"]["text"] == [
        f"None: [Errno 2] No such file or directory: '{records}'"
    ]
    given = json.loads((out / curves.CLIENT_RUN).read_text())["arguments"]
    assert page.rows("Workload")[curves.CONCURRENCY] == [str(given[curves.CONCURRENCY])]


def test_deploy_sh_writes_the_report_beside_the_split(tmp_path, planted):  # noqa: F811
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
    page = _Page(d)
    assert page.rows("Run")["run"] == ["simulated, step_stub"]
    counted = "Below, the 2 of them the headline metrics count."
    assert page.parts["Per-request split"]["text"] == [
        f"2 requests split, on the virtual clock; 3 not. {counted}"
    ]
    assert page.rows("Per-request split")["KV transfer"] == ["2", "0.4", "0.4", "0.4"]
    # The planted authority's summary is empty.
    assert page.parts["Trust"]["text"] == [
        f"None: its {SUMMARY_FILE} holds no schedule."
    ]
    assert page.parts["A real run's per-request split"]["text"] == [
        "None: a simulated run's directory holds no real run."
    ]


@pytest.mark.parametrize("missing", [html.RUN, SUMMARY_FILE])
def test_a_simulated_run_s_report_names_the_file_it_lacks(tmp_path, missing):
    run = _run(tmp_path / "run", 4, server=False)
    (run / html.RUN).write_text(json.dumps({"run_file": {}}))
    (run / split.FILE).write_text(json.dumps({"refused": "no steps"}))
    if missing == html.RUN:
        (run / html.RUN).unlink()
        with pytest.raises(FileNotFoundError, match=html.RUN):
            html.write(run)
        return

    page = _Page(html.write(run).parent)

    assert page.parts["Trust"]["text"] == [
        f"None: this run's directory holds no {SUMMARY_FILE}."
    ]
    assert page.rows("Headline metrics")["speed ratio"] == [
        f"none: no {SUMMARY_FILE} states it"
    ]
    assert page.parts["Per-request split"]["text"] == ["None: no steps"]
