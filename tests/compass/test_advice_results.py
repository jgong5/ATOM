# SPDX-License-Identifier: MIT
"""`compass advice SPACE OUT`'s results: each point's metrics from its client
export, the frontier, the recommendation, the runner-up, the result table and
the recommendation's text.

Each run's client export is `test_validate_curves`' `RECORDS`, every time
scaled by the run's ``scale`` (`_run`, `_deployment`), with
``usage_prompt_cache_read_tokens`` added to each record unless a test says
otherwise. At scale 1 its profiling records serve `TOKENS` tokens over 11.65 s
at p90 interactivity `I` and p90 TTFT `TTFT`; their p90 end-to-end latency is
`E2E`, worked by hand below.
"""

import json
import os
import re
import shlex
import subprocess
import sys

import platforms
import pytest
from test_advice_check import resolved, starts
from test_advice_simulate import simulating, stubbed  # noqa: F401  the fixtures
from test_calibrate import _path, _tree
from test_memory_readings import publish_readings
from test_run_step_grid import DOCUMENT
from test_validate_curves import INFERENCEX, _deployment, _run

from atom.compass import cli
from atom.compass.advice import check, results, simulate
from atom.compass.advice.space import expand
from atom.compass.backends.cost import GROUPS
from atom.compass.drive import lacking
from atom.compass.validate import curves

TOKENS, CACHED = 127300 + 693, 1000
I, TTFT = INFERENCEX["p90 interactivity"], INFERENCEX["p90 TTFT"]
# The eight profiling records' request latencies, 0.35 to 7.5 s: rank 6.3.
E2E = 5.75 + 0.3 * (7.5 - 5.75)
#: A run summary priced 75% estimated and 25% unvalidated, 3 of 40 steps
#: extrapolated, and with no refusal.
SHARES = dict.fromkeys(GROUPS, 0.0) | {"estimated": 0.75, "unvalidated": 0.25}
SUMMARY = {
    "coverage_report": False,
    "schedule": {
        "seconds_by_method": {
            g: {"steps": 1, "fraction_of_predicted_seconds": s}
            for g, s in SHARES.items()
        },
        "extrapolated_steps": 3,
        "refusals": {"steps": 40, "reasons": []},
    },
}
REFUSED = "cost:step-linear: no decode vector for rung 384"
COVERAGE = SUMMARY | {
    "coverage_report": True,
    "schedule": SUMMARY["schedule"]
    | {"refusals": {"steps": 40, "reasons": [[REFUSED, 3]]}},
}
LAW = "/laws/law.json"


def made(
    d, concurrency, scale=1.0, tp=1, sides=None, cached=True, itl=True, summary=SUMMARY
):
    """A simulated run of the export at `scale` at `d`, on `tp` GPUs or a
    deployment's (tensor parallel, data parallel) `sides`, its run summary
    `summary` where given; every record states its cached tokens if `cached`,
    and none its inter-token latency unless `itl`."""
    if sides:
        _deployment(d, concurrency, scale, server=False, **sides)
    else:
        _run(d, concurrency, scale, server=False, tp=tp)
    records = [
        json.loads(line) for line in (d / curves.RECORDS).read_text().splitlines()
    ]
    for r in records:
        if cached:
            r["metrics"][results.CACHED] = {"value": CACHED, "unit": "tokens"}
        if not itl:
            r["metrics"].pop("inter_token_latency", None)
    (d / curves.RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    if summary:
        (d / "summary.json").write_text(json.dumps(summary))
    return d


def search(tmp_path, configs: dict, concurrency: list, points: dict, slo: dict):
    """`results.results` over `configs`, each name to its block, at
    `concurrency`: each point of `points` a run it is made into, `made`'s
    arguments, else a state it is left out in, ``failed`` naming its tries."""
    space = {
        "model": "/model",
        "machine": {"device": "mi355x", "ptl": "disabled"},
        "store": "/store",
        "workload": {"traces": "/traces.jsonl", "select": "agentx"},
        "slo": slo,
        "common": {"basis": "a recipe"},
        "dimensions": {"config": configs},
    }
    space["workload"]["concurrency"] = concurrency
    (tmp_path / "space.json").write_text(json.dumps(space))
    expansion, out = expand(tmp_path / "space.json"), tmp_path / "out"
    out.mkdir()
    judged, found, outcome = {"candidates": {}}, {}, {"points": {}}
    for p in expansion["points"]:
        name, c = p["name"], p["candidate"]
        reuse = {
            "verdict": "reuse",
            "requirement": None,
            "reason": "",
            "run": [LAW] * 2,
        }
        law = dict.fromkeys(p["instances"], reuse)
        judged["candidates"].setdefault(c, {"points": {}})["points"][name] = {
            "law": law
        }
        state = found.setdefault(c, {"state": "searchable", "points": {}})
        given = points[name]
        if isinstance(given, dict) and "state" in given:
            held = {"why": [], "made_by": []} | given
            if held["state"] == "failed":
                outcome["points"][name] = held
                held = {"state": "searchable"}
            elif held.pop("candidate", False):
                found[c] = held | {"points": {}}
                continue
            state["points"][name] = held
            continue
        state["points"][name] = {"state": "searchable"}
        run = made(out / "runs" / name, p["concurrency"], **given)
        outcome["points"][name] = {"state": "finished", "run": str(run), "tries": 1}
    return results.results(expansion, judged, found, outcome, out)


def rows(held: dict) -> dict:
    return {r["point"]: r for r in held["rows"]}


def test_inferencex_metrics_are_curves_and_the_search_s_own_come_from_the_same_export(
    tmp_path,
):
    deployment = {"deployment": "1p1d"}
    sides = {"prefill": (1, 8), "decode": (4, 1)}

    held = rows(
        search(
            tmp_path,
            {"a": {}, "p": deployment, "n": {}},
            [4],
            {
                "a-c4": {"tp": 4},
                "p-c4": {"sides": sides},
                "n-c4": {"tp": 4, "cached": False},
            },
            {},
        )
    )

    for point, gpus in (("a-c4", 4), ("p-c4", 12)):
        row, run = held[point], tmp_path / "out" / "runs" / point
        theirs = curves.metrics(run, simulated=True)
        assert {k: row[k] for k in results.INFERENCEX} == {
            k: theirs[k] for k in results.INFERENCEX
        }
        assert row["throughput per GPU"] == pytest.approx(TOKENS / 11.65 / gpus)
        # Eight profiling records, each with 1000 cached tokens.
        computed = (TOKENS - 8 * CACHED) / 11.65 / gpus
        assert row["computed throughput per GPU"] == pytest.approx(computed, rel=1e-12)
        assert row["p90 E2E latency"] == pytest.approx(E2E, rel=1e-12)
        assert row["method_shares"] == SHARES
        assert row["extrapolated_share"] == 3 / 40
        assert (row["grading"], row["refusal"]) == ("ungraded", None)
        assert row["laws"] == {side: [LAW, LAW] for side in row["laws"]}
    assert list(held["p-c4"]["laws"]) == ["prefill", "decode"]
    # A record that states no cached tokens: the computed throughput has no value.
    assert held["n-c4"]["computed throughput per GPU"] is None
    assert held["n-c4"]["throughput per GPU"] == held["a-c4"]["throughput per GPU"]


#: The points: (throughput per GPU, p90 interactivity) at scale s on tp GPUs
#: is (TOKENS / 11.65 / (s tp), I / s), so a-c1 and e-c1 are (T/2, I), b-c2
#: (T/4, 2I) and c-c2 (T/1.5, I/1.5), which no other dominates; b-c1, at
#: (T/3, I), only a-c1 and e-c1 do, by throughput alone. d-c1, whose (2T, 4I)
#: would dominate them all, is a coverage report, d-c2 failed, and e-c2's
#: records state no inter-token latency, so it has no interactivity.
POINTS = {
    "a-c1": {"scale": 1.0, "tp": 2},
    "a-c2": {"scale": 2.0, "tp": 1},
    "b-c1": {"scale": 1.0, "tp": 3},
    "b-c2": {"scale": 0.5, "tp": 8},
    "c-c1": {"scale": 4.0, "tp": 1},
    "c-c2": {"scale": 1.5, "tp": 1},
    "d-c1": {"scale": 0.25, "sides": {"prefill": (1, 1), "decode": (1, 1)}}
    | {"summary": COVERAGE},
    "d-c2": {"state": "failed", "tries": 2, "why": ["client.sh exited 1"]},
    "e-c1": {"scale": 1.0, "tp": 2},
    "e-c2": {"scale": 1.0, "tp": 8, "itl": False},
}
CONFIGS = {"a": {}, "b": {}, "c": {}, "d": {"deployment": "1p1d"}, "e": {}}


def test_the_frontier_holds_exactly_the_points_no_other_dominates(tmp_path):
    held = search(tmp_path, CONFIGS, [1, 2], POINTS, {})

    assert rows(held)["e-c2"]["p90 interactivity"] is None
    placed = [
        r
        for r in held["rows"]
        if r["grading"] == "ungraded" and r["p90 interactivity"] is not None
    ]
    brute = {
        r["point"]
        for r in placed
        if not any(
            o["throughput per GPU"] >= r["throughput per GPU"]
            and o["p90 interactivity"] >= r["p90 interactivity"]
            and (o["throughput per GPU"], o["p90 interactivity"])
            != (r["throughput per GPU"], r["p90 interactivity"])
            for o in placed
        )
    }
    assert brute == {"a-c1", "b-c2", "c-c2", "e-c1"}
    assert {r["point"] for r in held["rows"] if r["frontier"]} == brute
    d = rows(held)["d-c1"]
    assert d["throughput per GPU"] == pytest.approx(2 * TOKENS / 11.65)
    assert d["grading"] == "coverage report"
    assert d["refusal"] == f"coverage report: {REFUSED}, 3 times"
    # A 1P1D point's: a record of the deployment, then each side's fits.
    assert d["made_by"] == [
        lacking.SIDE_LAW.format(fit=f, side=s)
        for s in ("prefill", "decode")
        for f in lacking.FITS.values()
    ]


@pytest.mark.parametrize(
    "slo, best, runner",
    [
        # TTFT at most 1.7 s and interactivity at least 23, which e-c2, with
        # none, misses; at c-c2's concurrency, b-c2, though a-c1 has more
        # throughput.
        ({"max_ttft_p90_s": 1.7, "min_interactivity_p90": 23}, "c-c2", "b-c2"),
        # a-c1 and e-c1 tie on throughput: the first in SPACE's order leads.
        ({"max_ttft_p90_s": 1.2}, "a-c1", "e-c1"),
        # No bound: d-c1 would lead, but has a refusal; a-c2 now meets it, and
        # e-c2 with no interactivity.
        ({}, "c-c2", "a-c2"),
        ({"min_interactivity_p90": 2.1 * I}, None, None),
    ],
)
def test_the_recommendation_and_runner_up_meet_the_slo_and_have_no_refusal(
    tmp_path, slo, best, runner
):
    held = search(tmp_path, CONFIGS, [1, 2], POINTS, slo)

    assert (held["recommendation"], held["runner_up"]) == (best, runner)
    met = {r["point"] for r in held["rows"] if r["slo_met"]}
    for r in held["rows"]:
        if r["grading"]:
            meets = r["p90 TTFT"] <= slo.get("max_ttft_p90_s", 1e9)
            if (lowest := slo.get("min_interactivity_p90")) is not None:
                got = r["p90 interactivity"]
                meets = meets and got is not None and got >= lowest
            assert (r["point"] in met) == meets, r["point"]
    assert rows(held)["d-c2"]["slo_met"] is None
    text = held["text"].splitlines()
    if best:
        assert text[0].startswith(f"Recommended: {best[0]} at concurrency {best[-1]}: ")
        assert text[1].startswith(
            f"Runner-up: {runner[0]} at concurrency {runner[-1]}: "
        )
        assert text[2] == (
            "Confidence: ungraded; priced estimated 75.0%, unvalidated 25.0%; 7.5% "
            "of steps extrapolated."
        )
        assert text[3] == f"Laws: server {LAW}."
    else:
        assert text[0] == "No recommendation: no point meets the SLO with no refusal."
    assert text[-2:] == [
        f"Left out: d at concurrency 1: coverage report: {REFUSED}, 3 times",
        "Left out: d at concurrency 2: failed after 2 tries: client.sh exited 1",
    ]


def _numbers(text: str) -> set[str]:
    """Each number standing alone in `text`, not part of a name or a path."""
    return set(
        re.findall(r"(?:(?<=\s)|^)(\d+(?:\.\d+)?%?)(?=[\s,;:)]|\.\s|\.$|$)", text)
    )


@pytest.mark.parametrize("slo", [{"max_ttft_p90_s": 1.7}, {"max_ttft_p90_s": 1.2}])
def test_every_number_in_the_text_is_in_the_table(tmp_path, slo):
    held = search(tmp_path, CONFIGS, [1, 2], POINTS, slo)

    printed = results.lines(held)
    table = printed[2 : 2 + len(held["rows"])]
    said = _numbers(held["text"])
    assert {"75.0%", "25.0%", "7.5%", "384", "3"} <= said
    assert said <= _numbers("\n".join(table))
    assert printed[0] == f"SLO: p90 TTFT at most {slo['max_ttft_p90_s']:g} s"
    assert printed[1].split(" | ")[:3] == [
        "configuration",
        "concurrency",
        "throughput per GPU (tok/s/GPU)",
    ]
    assert printed[-len(held["text"].splitlines()) :] == held["text"].splitlines()
    assert json.loads((tmp_path / "out" / results.RESULTS).read_text()) == held


def test_each_point_not_simulated_is_left_out_with_its_reason_and_command(tmp_path):
    calibrate, measure = "calibrate.sh OUT/cal/e", "compass measure memory -- f"
    points = {
        "e-c1": {"state": "missing a law", "candidate": True}
        | {"why": ["no decode vector for rung eager"], "made_by": [calibrate]},
        "f-c1": {"state": "missing readings", "why": ["no entry"]}
        | {"made_by": [measure]},
        "f-c2": {"state": "failed", "tries": 0, "why": [simulate.UNSET]},
        "g-c1": {"state": "refused", "candidate": True, "why": ["RunnerRefusal: no"]},
    }

    held = search(
        tmp_path, {"e": {}, "f": {}, "g": {}}, [1], points, {"max_ttft_p90_s": 5}
    )

    assert [(r["point"], r["refusal"], r["made_by"]) for r in held["rows"]] == [
        ("e-c1", "missing a law: no decode vector for rung eager", [calibrate]),
        ("f-c1", "missing readings: no entry", [measure]),
        ("g-c1", "refused: RunnerRefusal: no", []),
    ]
    assert held["text"].splitlines() == [
        "No recommendation: no point meets the SLO with no refusal.",
        "Left out: e at concurrency 1: missing a law: no decode vector for rung eager",
        "Left out: f at concurrency 1: missing readings: no entry",
        "Left out: g at concurrency 1: refused: RunnerRefusal: no",
    ]
    assert all(r[k] is None for r in held["rows"] for k in results.METRICS)
    printed = results.lines(held)
    assert f"e at concurrency 1, made by: {calibrate}" in printed
    assert f"f at concurrency 1, made by: {measure}" in printed


SERVER = """#!{python}
import json, os, signal, sys, time
from pathlib import Path

argv = sys.argv[1:]
at, stub = Path(argv[argv.index("--out") + 1]), Path(os.environ["STUB_DIR"])
with open(os.environ["STUB_CALLS"], "a") as calls:
    calls.write(json.dumps({{"argv": argv}}) + "\\n")
at.mkdir(parents=True)
def stop(*_):
    summaries = json.loads((stub / "summaries.json").read_text())
    summary = summaries.get(at.name, summaries["*"])
    (at / "summary.json").write_text(json.dumps(summary))
    (at / "run_file.json").write_text((stub / "run_file.json").read_text())
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
(at / "server.json").write_text(json.dumps({{"pid": os.getpid()}}))
while True:
    time.sleep(0.05)
"""
CLIENT = """#!{python}
import json, os, shutil, signal, sys, time
from pathlib import Path

at = Path(sys.argv[1])
with open(os.environ["STUB_CALLS"], "a") as calls:
    calls.write(json.dumps({{"client": sys.argv[1:]}}) + "\\n")
shutil.copytree(Path(os.environ["STUB_DIR"]) / "made" / at.name, at, dirs_exist_ok=True)
pid = json.loads((at / "server.json").read_text())["pid"]
os.kill(pid, signal.SIGTERM)
stat = Path(f"/proc/{{pid}}/stat")
while stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z":
    time.sleep(0.05)
"""


def test_a_search_ends_in_its_results_and_a_changed_slo_simulates_nothing(
    tmp_path, simulating, monkeypatch, capsys  # noqa: F811
):
    """`compass advice` on a, b and m, the check finding m missing a law: each
    point the client stub runs copies its export from ``made/``, and the server
    stub writes its run summary, b-c4's a coverage report, and a run file whose
    machine is the suite's spec."""
    for name, text in (("server", SERVER), ("client", CLIENT)):
        (tmp_path / name).write_text(text.format(python=sys.executable))
        (tmp_path / name).chmod(0o755)
    monkeypatch.setattr(check, "COMPASS", tmp_path / "server")
    monkeypatch.setattr(simulate, "CLIENT", tmp_path / "client")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))
    summaries = {"*": SUMMARY, "b-c4": COVERAGE}
    (tmp_path / "summaries.json").write_text(json.dumps(summaries))
    (tmp_path / "run_file.json").write_text(json.dumps({"machine": DOCUMENT}))
    # (throughput per GPU, p90 interactivity, p90 TTFT) over (T, I, TTFT):
    # a-c1 (1, 1, 1), a-c4 (1/1.5, 1/1.5, 1.5), b-c1 (1/1.6, 1.25, 0.8), and
    # b-c4 (2, 2, 0.5), a coverage report.
    for point, scale, tp in (("a-c1", 1, 1), ("a-c4", 1.5, 1), ("b-c1", 0.8, 2)):
        made(tmp_path / "made" / point, int(point[-1]), scale, tp, summary=None)
    made(tmp_path / "made" / "b-c4", 4, 0.5, summary=None)
    flags = platforms.flags(tmp_path / "store", DOCUMENT, tmp_path)
    space = {
        "model": str(tmp_path),
        "machine": {"device": flags[1], "ptl": flags[3]},
        "store": str(tmp_path / "store"),
        "workload": {"traces": str(tmp_path / "traces.jsonl"), "select": "agentx"},
        "common": {"basis": "a recipe"},
        "dimensions": {"config": {"a": {}, "b": {"args": {"--max-num-seqs": 8}}}},
    }
    space["workload"]["concurrency"] = [1, 4]
    space["dimensions"]["config"]["m"] = {"args": {"--max-num-seqs": 12}}
    out = tmp_path / "out"

    def advise(slo: dict) -> dict:
        (tmp_path / "space.json").write_text(json.dumps(space | {"slo": slo}))
        cli.main(["advice", str(tmp_path / "space.json"), str(out)])
        return json.loads((out / results.RESULTS).read_text())

    first = advise({"max_ttft_p90_s": 2})

    held = rows(first)
    assert (first["recommendation"], first["runner_up"]) == ("a-c1", "b-c1")
    assert held["a-c1"]["throughput per GPU"] == pytest.approx(TOKENS / 11.65)
    assert held["b-c4"]["refusal"] == f"coverage report: {REFUSED}, 3 times"
    (calibration,) = held["b-c4"]["made_by"]
    assert f"TRACES={tmp_path / 'traces.jsonl'} " in calibration
    assert f"calibrate.sh {out / 'cal' / 'b'} " in calibration
    (missing,) = held["m-c1"]["made_by"]
    assert held["m-c1"]["refusal"].startswith("missing a law: ")
    assert f"calibrate.sh {out / 'cal' / 'm'} " in missing
    capsys.readouterr()
    ran = len(starts(tmp_path))

    second = advise({"max_ttft_p90_s": 1, "min_interactivity_p90": 30})

    printed = capsys.readouterr().out
    assert (
        "only in its SLO: its results again, nothing checked or simulated\n" in printed
    )
    assert len(starts(tmp_path)) == ran
    assert (second["recommendation"], second["runner_up"]) == ("b-c1", None)
    met = [r["slo_met"] for r in second["rows"]]
    assert met == [False, False, True, True, None, None]
    assert "No runner-up: no other configuration's point at concurrency 1" in printed

    again = advise({"max_ttft_p90_s": 1, "min_interactivity_p90": 30})

    # The same SLO: the whole search again, each finished point skipped.
    assert len(starts(tmp_path)) > ran
    assert not [c for c in starts(tmp_path)[ran:] if "client" in c]
    assert again == second

    def stopped(*_):
        raise SystemExit("compass advice: stopped by signal 15")

    with monkeypatch.context() as stopping, pytest.raises(SystemExit):
        stopping.setattr(simulate, "simulate", stopped)
        advise({"max_ttft_p90_s": 1, "min_interactivity_p90": 30})
    ran = len(starts(tmp_path))

    advise({"max_ttft_p90_s": 2})

    # A search stopped before its outcomes: the next is whole, SLO changed or not.
    assert len(starts(tmp_path)) > ran
    assert "only in its SLO" not in capsys.readouterr().out
    ran = len(starts(tmp_path))
    space["dimensions"]["config"]["m"]["args"]["--max-num-seqs"] = 16

    changed = advise({"max_ttft_p90_s": 1, "min_interactivity_p90": 30})

    # The SLO and a block's arguments changed in one edit: the whole search.
    assert len(starts(tmp_path)) > ran
    assert "only in its SLO" not in capsys.readouterr().out
    assert "--max-num-seqs 16" in rows(changed)["m-c1"]["made_by"][0]


def _calibrated(tmp_path, command: str, **env) -> tuple[int, list[str]]:
    """The printed calibrate.sh `command` run in `test_calibrate`'s stub tree,
    under its variables and `env`: its exit status and the commands it ran."""
    if not (tmp_path / "ATOM").exists():
        _tree(tmp_path)
    words = shlex.split(command.split("\n")[0])
    at = words.index(str(lacking.CALIBRATE))
    log = tmp_path / "log"
    log.unlink(missing_ok=True)
    path = _path(tmp_path)
    given = dict(w.split("=", 1) for w in words[:at]) | env
    r = subprocess.run(
        ["bash", str(tmp_path / "ATOM" / "scripts" / "compass" / "calibrate.sh")]
        + words[at + 1 :],
        cwd=tmp_path,
        env=os.environ | {"PATH": path, "LOG": str(log)} | given,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return r.returncode, log.read_text().splitlines() if log.exists() else []


def test_a_calibrated_candidate_s_coverage_command_resumes_it_or_says_why_none(
    tmp_path, simulating, monkeypatch  # noqa: F811
):
    """b's one point a coverage report under the suite's law file: its
    calibrate.sh command, run in the stub tree into the shared store and failed
    at its last record, is printed again with that store, and takes only that
    record and the fits; once that calibration has ended, why no command can."""
    for name, text in (("server", SERVER), ("client", CLIENT)):
        (tmp_path / name).write_text(text.format(python=sys.executable))
        (tmp_path / name).chmod(0o755)
    monkeypatch.setattr(check, "COMPASS", tmp_path / "server")
    monkeypatch.setattr(simulate, "CLIENT", tmp_path / "client")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))
    monkeypatch.setenv("AIPERF_PYTHON", "/real/python")
    (tmp_path / "summaries.json").write_text(json.dumps({"*": COVERAGE}))
    (tmp_path / "run_file.json").write_text(json.dumps({"machine": DOCUMENT}))
    made(tmp_path / "made" / "b-c1", 1, summary=None)
    flags = platforms.flags(tmp_path / "store", DOCUMENT, tmp_path)
    space = {
        "model": str(tmp_path),
        "machine": {"device": flags[1], "ptl": flags[3]},
        "store": str(tmp_path / "store"),
        "workload": {"traces": str(tmp_path / "traces.jsonl"), "select": "agentx"},
        "slo": {},
        "common": {"basis": "a recipe"},
        "dimensions": {"config": {"b": {"args": {"--max-num-seqs": 8}}}},
    }
    space["workload"]["concurrency"] = [1]
    (tmp_path / "space.json").write_text(json.dumps(space))
    out, store = tmp_path / "out", str((tmp_path / "store").resolve())

    def coverage() -> list[str]:
        cli.main(["advice", str(tmp_path / "space.json"), str(out)])
        return rows(json.loads((out / results.RESULTS).read_text()))["b-c1"]["made_by"]

    (first,) = coverage()
    words, script = shlex.split(first.split("\n")[0]), f" {lacking.CALIBRATE} "
    names = [
        (
            f"trace{k}-c{r.split()[1]}"
            if r.startswith("trace ")
            else f"cal{k}-c{r.split()[0]}"
        )
        for k, r in enumerate(words[words.index(script.strip()) + 2 :], 1)
    ]
    # The last first-turn record fails; the stub's trace runs end done when stopped.
    left = names[max(k for k, n in enumerate(names) if n.startswith("cal")) :]
    assert _calibrated(tmp_path, first, STORE=store, CLIENT_FAIL=left[0])[0] == 1

    (command,) = coverage()

    assert command == first.replace(script, f" STORE={shlex.quote(store)}{script}")
    status, log = _calibrated(tmp_path, command)
    assert status == 0
    served = [line.split(" --out ")[1] for line in log if line.startswith("record ")]
    assert served == [str(out / "cal" / "b" / n) for n in left]
    ran = [line.split()[0] for line in log]
    assert ran == ["record", "client"] * len(left) + ["fit", "fit"]
    (why,) = coverage()
    assert why.startswith(
        f"no calibrate.sh command: the calibration in {out / 'cal' / 'b'} has ended"
    )


#: Two candidates at one law key, as the DeepSeek-V4-Pro template's S2 and S5,
#: whose LMCache no law key holds: the --max-model-len the stubbed resolution
#: reads keys S5's readings, and no law.
S2 = {"args": {"--max-num-seqs": 8}}
S5 = {"args": S2["args"] | {"--max-model-len": 512}}


@pytest.mark.parametrize(
    "started, ended",
    [(None, False), ("own", False), ("own", True), ("shared", False), ("shared", True)],
    ids=[
        "not started",
        "started into its own store",
        "ended into its own store",
        "started into the shared store",
        "ended into the shared store",
    ],
)
def test_two_coverage_reports_at_one_law_key_print_one_calibration(
    tmp_path, simulating, monkeypatch, capsys, started, ended  # noqa: F811
):
    """S2's and S5's one point each a coverage report under the suite's law
    file, both looked up in the shared store, with S2's calibration in OUT as
    `started` and `ended` say: S2's row gives its own answer, and S5's names
    S2's calibration, with the command measuring S5's readings into the store
    that calibration publishes into unless that is the shared one; there, once
    that calibration has ended, S2's answer."""
    for name, text in (("server", SERVER), ("client", CLIENT)):
        (tmp_path / name).write_text(text.format(python=sys.executable))
        (tmp_path / name).chmod(0o755)
    monkeypatch.setattr(check, "COMPASS", tmp_path / "server")
    monkeypatch.setattr(simulate, "CLIENT", tmp_path / "client")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))
    (tmp_path / "summaries.json").write_text(json.dumps({"*": COVERAGE}))
    (tmp_path / "run_file.json").write_text(json.dumps({"machine": DOCUMENT}))
    for point in ("S2-c1", "S5-c1"):
        made(tmp_path / "made" / point, 1, summary=None)
    store, out = tmp_path / "store", tmp_path / "out"
    flags = platforms.flags(store, DOCUMENT, tmp_path)
    space = {
        "model": str(tmp_path),
        "machine": {"device": flags[1], "ptl": flags[3]},
        "store": str(store),
        "workload": {"traces": str(tmp_path / "traces.jsonl"), "select": "agentx"},
        "slo": {},
        "common": {"basis": "a recipe"},
        "dimensions": {"config": {"S2": S2, "S5": S5}},
    }
    space["workload"]["concurrency"] = [1]
    (tmp_path / "space.json").write_text(json.dumps(space))
    args5 = ["--max-num-seqs", "8", "--max-model-len", "512"]
    publish_readings(store, resolved(simulating.config, args5))
    cal = out / "cal" / "S2"
    if started:
        cal.mkdir(parents=True)
        named = f"store {store}\n" if started == "shared" else ""
        (cal / "inputs").write_text(f"tree c0ffee\nrecord 1\n{named}")
    if ended:
        (cal / "status").write_text("2026-11-24T00:00:00Z end calibration 0.50\n")

    cli.main(["advice", str(tmp_path / "space.json"), str(out)])

    held = rows(json.loads((out / results.RESULTS).read_text()))
    (answer,) = held["S2-c1"]["made_by"]
    if ended:
        why = f"no calibrate.sh command: the calibration in {cal} has ended"
        assert answer.startswith(why)
    else:
        assert f"calibrate.sh {cal} " in answer
        given = f" STORE={shlex.quote(str(store))} " in answer
        assert given == (started == "shared")
    into = str(store if started == "shared" else cal / "store")
    borrowed = [check.BORROWED.format("S2", into)]
    if started == "shared":
        # S5 was looked up in the store S2's calibration publishes into.
        assert held["S5-c1"]["made_by"] == ([answer] if ended else borrowed)
    else:
        measure = f"compass measure memory --gpu G --out D --store {into} -- "
        measure += " ".join(["--model", str(tmp_path), *args5])
        measured = check.PENDING.format(into, "S2") if not started else measure
        assert held["S5-c1"]["made_by"] == [*borrowed, measured]
    printed = capsys.readouterr().out
    assert printed.count(f"{lacking.CALIBRATE} ") == (0 if ended else 1)
    for line in held["S5-c1"]["made_by"]:
        assert f"S5 at concurrency 1, made by: {line}\n" in printed
