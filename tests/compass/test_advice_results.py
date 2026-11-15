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
import re
import sys

import platforms
import pytest
from test_advice_check import starts
from test_advice_simulate import simulating, stubbed  # noqa: F401  the fixtures
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


def made(d, concurrency, scale=1.0, tp=1, sides=None, cached=True, summary=SUMMARY):
    """A simulated run of the export at `scale` at `d`, on `tp` GPUs or a
    deployment's (tensor parallel, data parallel) `sides`, its run summary
    `summary` where given; every record states its cached tokens if `cached`."""
    if sides:
        _deployment(d, concurrency, scale, server=False, **sides)
    else:
        _run(d, concurrency, scale, server=False, tp=tp)
    records = [
        json.loads(line) for line in (d / curves.RECORDS).read_text().splitlines()
    ]
    for r in records if cached else ():
        r["metrics"][results.CACHED] = {"value": CACHED, "unit": "tokens"}
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


#: Eight points: (throughput per GPU, p90 interactivity) at scale s on tp GPUs
#: is (TOKENS / 11.65 / (s tp), I / s), so a-c1 is (T/2, I), b-c2 (T/4, 2I)
#: and c-c2 (T/1.5, I/1.5), which no other dominates; d-c1, whose (2T, 4I)
#: would dominate them all, is a coverage report, and d-c2 failed.
EIGHT = {
    "a-c1": {"scale": 1.0, "tp": 2},
    "a-c2": {"scale": 2.0, "tp": 1},
    "b-c1": {"scale": 1.0, "tp": 8},
    "b-c2": {"scale": 0.5, "tp": 8},
    "c-c1": {"scale": 4.0, "tp": 1},
    "c-c2": {"scale": 1.5, "tp": 1},
    "d-c1": {"scale": 0.25, "sides": {"prefill": (1, 1), "decode": (1, 1)}}
    | {"summary": COVERAGE},
    "d-c2": {"state": "failed", "tries": 2, "why": ["client.sh exited 1"]},
}
CONFIGS = {"a": {}, "b": {}, "c": {}, "d": {"deployment": "1p1d"}}


def test_the_frontier_holds_exactly_the_points_no_other_dominates(tmp_path):
    held = search(tmp_path, CONFIGS, [1, 2], EIGHT, {})

    placed = [r for r in held["rows"] if r["grading"] == "ungraded"]
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
    assert brute == {"a-c1", "b-c2", "c-c2"}
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
        # The four whose TTFT is at most 1.7 s and interactivity at least 23;
        # at c-c2's concurrency, b-c2, though a-c1 has more throughput.
        ({"max_ttft_p90_s": 1.7, "min_interactivity_p90": 23}, "c-c2", "b-c2"),
        ({"max_ttft_p90_s": 1.2}, "a-c1", "b-c1"),
        # No bound: d-c1 would lead, but has a refusal; a-c2 now meets it.
        ({}, "c-c2", "a-c2"),
        ({"min_interactivity_p90": 2.1 * I}, None, None),
    ],
)
def test_the_recommendation_and_runner_up_meet_the_slo_and_have_no_refusal(
    tmp_path, slo, best, runner
):
    held = search(tmp_path, CONFIGS, [1, 2], EIGHT, slo)

    assert (held["recommendation"], held["runner_up"]) == (best, runner)
    met = {r["point"] for r in held["rows"] if r["slo_met"]}
    for r in held["rows"]:
        if r["grading"]:
            meets = r["p90 TTFT"] <= slo.get("max_ttft_p90_s", 1e9)
            meets &= r["p90 interactivity"] >= slo.get("min_interactivity_p90", 0)
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
    held = search(tmp_path, CONFIGS, [1, 2], EIGHT, slo)

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
