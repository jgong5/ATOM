# SPDX-License-Identifier: MIT
"""`compass advice`'s dashboard: one static page over the search's files, each
point placed by the metrics its results hold, marked by its grading state and
pricing-method shares, grey with its reason when it has a refusal, and opening
its run's report.

The searches are `test_advice_results`': `search` builds the results over its
`POINTS` directly, and the stub server and client run them through `compass
advice` itself.
"""

import json
import math
import re
import sys
from html import escape

import platforms
import pytest
from test_advice_check import starts
from test_advice_results import (
    CLIENT,
    CONFIGS,
    COVERAGE,
    LAW,
    POINTS,
    REFUSED,
    SERVER,
    SHARES,
    SUMMARY,
    made,
    search,
)
from test_advice_simulate import simulating, stubbed  # noqa: F401  the fixtures
from test_run_step_grid import DOCUMENT

from atom.compass import cli
from atom.compass.advice import check, dashboard, results, simulate
from atom.compass.advice.space import expand
from atom.compass.drive.serve import RUN_JSON
from atom.compass.validate.runs import sides

#: A drawn point: its link, place, hover text and mark.
POINT = re.compile(
    r'(?:<a href="(?P<href>[^"]*)">)?<g transform="translate\((?P<x>[\d.]+),'
    r'(?P<y>[\d.]+)\)"><title>(?P<title>[^<]*)</title>'
    r'(?P<mark>(?:(?!<title>).)*?)<text class="c"',
    re.DOTALL,
)
SLICE = re.compile(
    r'<path class="m-(\w+)" d="M0 0L(\S+) (\S+)A10 10 0 [01] 1 (\S+) (\S+)Z"/>'
)
SOURCE = "InferenceX, frozen 2026-10-29: another ATOM version"
#: The ATOM tree each run's sides state in their ``run.json``, a revision and
#: its uncommitted files, by point, and the tree its hover text names: d-c1's
#: two sides at two revisions, and e-c1's run stating none.
SOURCES = {"*": [("a" * 40, 0)], "d-c1": [("b" * 40, 0), ("c" * 40, 3)], "e-c1": []}
TREES = {
    "*": "commit aaaaaaaaa",
    "d-c1": "prefill commit bbbbbbbbb; decode commit ccccccccc, 3 files uncommitted",
    "e-c1": "not stated",
}


def drawn(page: str) -> dict:
    """Each view's points, by (y, x) metric: each point's name to its match."""
    views = {}
    for i, y in enumerate(dashboard.AXES["y"]):
        for j, x in enumerate(dashboard.AXES["x"]):
            view = page.split(f'<div class="y{i}x{j}">')[1].split("</svg>")[0]
            found = {}
            for m in POINT.finditer(view):
                name, _, at = m["title"].split("\n")[0].partition(" at concurrency ")
                found[f"{name}-c{at}"] = m
            views[y, x] = found
    return views


def shares(mark: str) -> dict:
    """Each slice's turn of a point's fill, by method, the slices tiling it
    clockwise from the top: each starts where the last ended."""
    held = {m: 1.0 for m in re.findall(r'<circle r="10" class="m-(\w+)"/>', mark)}
    last = (0.0, -10.0)
    for method, *ends in SLICE.findall(mark):
        x0, y0, x1, y1 = map(float, ends)
        assert (x0, y0) == pytest.approx(last, abs=0.01), method
        turn = (math.atan2(y1, x1) - math.atan2(y0, x0)) % (2 * math.pi)
        held[method], last = held.get(method, 0) + turn / (2 * math.pi), (x1, y1)
    return held


def labelled(page: str) -> str | None:
    """The source the legend labels the reference layer with, if any."""
    found = re.search(r"<p>Reference, not this search: <svg.*?</svg> ([^<]*)</p>", page)
    return found and found[1]


def dashed(tmp_path, slo: dict, layer=None, reports=()) -> tuple[dict, str]:
    """`search` over `POINTS` at `slo`, each run's sides given their
    `SOURCES` and each run in `reports` a report, then the dashboard over its
    results, expansion and reuse check: the results and the page."""
    held = search(tmp_path, CONFIGS, [1, 2], POINTS, slo)
    out = tmp_path / "out"
    head = {
        "commit": "f" * 40,
        "ref": "origin/feature/atomcompass_new",
        "atom": "e" * 64,
    }
    verdicts = {"head": head, "refused": None, "candidates": {}}
    (out / check.REUSE).write_text(json.dumps(verdicts))
    for run in (out / "runs").iterdir():
        for side, (revision, dirty) in zip(
            sides(run), SOURCES.get(run.name, SOURCES["*"])
        ):
            source = {"revision_kind": "commit", "revision": revision, "dirty": dirty}
            (side / RUN_JSON).write_text(json.dumps({"source": source}))
    for name in reports:
        (out / "runs" / name / dashboard.REPORT).write_text("<p>report</p>")
    expansion = expand(tmp_path / "space.json")
    (out / "candidates.json").write_text(json.dumps(expansion))
    page = dashboard.write(out, out / "candidates.json", layer).read_text()
    return held, page


def test_each_point_is_placed_by_its_metrics_on_each_view_and_opens_its_report(
    tmp_path,
):
    reports = ("a-c1", "a-c2", "b-c1", "b-c2", "c-c1", "c-c2", "d-c1", "e-c2")

    held, page = dashed(tmp_path, {"max_ttft_p90_s": 1.7}, reports=reports)

    rows, floor = {r["point"]: r for r in held["rows"]}, dashboard.H - dashboard.BOTTOM
    for (y, x), points in drawn(page).items():
        # Each point with both values, and no other; d-c2 failed and has none,
        # e-c2 no interactivity but an E2E-normalized one.
        placed = {n for n, r in rows.items() if None not in (r[x], r[y])}
        assert set(points) == placed
        assert ("e-c2" in placed) == (x == "p90 E2E-normalized interactivity")
        # Each place in proportion to its values, from the axes' origin, as
        # a-c1's is to its own, each place rounded to 0.05 pixels.
        a = points["a-c1"]
        for name, m in points.items():
            for metric, at, theirs in (
                (x, float(m["x"]) - dashboard.LEFT, float(a["x"]) - dashboard.LEFT),
                (y, floor - float(m["y"]), floor - float(a["y"])),
            ):
                ratio = rows[name][metric] / rows["a-c1"][metric]
                assert at == pytest.approx(theirs * ratio, abs=0.06 * (1 + ratio))
            href = f"runs/{name}/{dashboard.REPORT}" if name in reports else None
            assert m["href"] == href
            row, given = rows[name], POINTS[name]
            gpus = given.get("tp") or sum(t * d for t, d in given["sides"].values())
            total = row["throughput per GPU"] * gpus
            laws = "; ".join(f"{s} {LAW}" for s in given.get("sides", ["server"]))
            said = m["title"].split("\n")
            assert (
                said[0] == f"{row['configuration']} at concurrency {row['concurrency']}"
            )
            assert {
                f"GPUs: {gpus}",
                f"total throughput (tok/s): {total:.1f}",
                f"ATOM tree: {TREES.get(name, TREES['*'])}",
                f"laws: {laws}",
            } <= set(said)
    # The computed tokens are fewer than those served: every point lower.
    served = drawn(page)["throughput per GPU", "p90 interactivity"]
    computed = drawn(page)["computed throughput per GPU", "p90 interactivity"]
    assert all(float(computed[n]["y"]) > float(m["y"]) for n, m in served.items())
    # The table links the same reports; a run without one says so.
    for name in rows:
        cell = f'<a href="runs/{name}/{dashboard.REPORT}">report</a>'
        assert (cell in page) == (name in reports), name
    assert "<script" not in page and "src=" not in page


def test_each_point_carries_its_grading_state_and_method_shares_and_a_refusal_is_grey(
    tmp_path,
):
    held, page = dashed(tmp_path, {"max_ttft_p90_s": 1.7})

    points = drawn(page)["throughput per GPU", "p90 E2E-normalized interactivity"]
    for name, m in points.items():
        if name == "d-c1":
            continue
        # Ungraded: a circle filled by its method shares, 75% and 25% of a turn.
        assert 'clip-path="url(#clip-ungraded)"' in m["mark"]
        assert '<g class="outline"><circle r="7"/></g>' in m["mark"]
        assert shares(m["mark"]) == pytest.approx(
            {k: v for k, v in SHARES.items() if v}
        )
        assert 'class="refused"' not in m["mark"]
    # d-c1, a coverage report: grey, its shape the coverage report's, no slice,
    # and its reason in its hover text.
    d = points["d-c1"]["mark"]
    assert d.startswith('<circle r="12" fill="transparent"/><g class="refused">')
    assert dashboard.SHAPES["coverage report"] in d and "m-" not in d
    reason = f"refusal reason: coverage report: {REFUSED}, 3 times"
    assert reason in points["d-c1"]["title"]
    # Its row, and the failed d-c2's, grey with the reason.
    for name in ("d-c1", "d-c2"):
        (row,) = re.findall(
            rf'<tr class="refused"><td>d</td><td>{name[-1]}</td>.*?</tr>', page
        )
        why = next(r["refusal"] for r in held["rows"] if r["point"] == name)
        assert f"<td>{why}</td>" in row
    # The recommendation and the runner-up ringed and named, as the results hold.
    assert (held["recommendation"], held["runner_up"]) == ("c-c2", "b-c2")
    for name, ring in (("c-c2", "ring"), ("b-c2", "ring runner")):
        assert f'<circle class="{ring}" r="13"/>' in points[name]["mark"]
    assert sum(m["mark"].count('class="ring') for m in points.values()) == 2
    for line in held["text"].splitlines():
        assert f"<p>{escape(line)}</p>" in page


def test_each_switch_shows_its_definitions_view_and_the_frontier_joins_its_points(
    tmp_path,
):
    held, page = dashed(tmp_path, {})

    # One rule shows each view, at the switches of its own two definitions.
    style = page.split("<style>")[1].split("</style>")[0]
    shown = re.search(r"([^{}]*)\{ display: block; \}", style)[1]
    ys, xs = dashboard.AXES["y"], dashboard.AXES["x"]
    assert {s.strip() for s in shown.split(",")} == {
        f"#y{i}:checked ~ #x{j}:checked ~ .views .y{i}x{j}"
        for i in range(len(ys))
        for j in range(len(xs))
    }
    # Each switch labelled with its definition, InferenceX's checked.
    for axis, names in (("y", ys), ("x", xs)):
        for i, label in enumerate(names.values()):
            checked = " checked" if i == 0 else ""
            assert f'id="{axis}{i}"{checked}><label for="{axis}{i}">{label}<' in page
    # By p90 interactivity: c-c2's I/1.5, a-c1's and e-c1's I, b-c2's 2I.
    frontier = ["c-c2", "a-c1", "e-c1", "b-c2"]
    assert {r["point"] for r in held["rows"] if r["frontier"]} == set(frontier)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            view = page.split(f'<div class="y{i}x{j}">')[1].split("</svg>")[0]
            named = r'<text class="name" [^>]*text-anchor="middle">([^<]*)</text>'
            assert re.findall(named, view) == [
                results.METRICS[x][0],
                results.METRICS[y][0],
            ]
            # The frontier through its points' places on this view, in order.
            points = drawn(page)[y, x]
            (line,) = re.findall(r'<polyline class="frontier" points="([^"]*)"/>', view)
            assert line == " ".join(
                f"{points[n]['x']},{points[n]['y']}" for n in frontier
            )


def test_each_row_shows_the_result_tables_columns_as_compass_advice_prints_them(
    tmp_path,
):
    held, page = dashed(tmp_path, {"max_ttft_p90_s": 1.7})

    table = page.split('<div class="wide">')[1].split("</table>")[0]
    shown = [re.findall(r"<td>(.*?)</td>", r) for r in table.split("<tr")[2:]]
    printed = results.lines(held)[2 : 2 + len(held["rows"])]
    for cells, line in zip(shown, printed, strict=True):
        cell = dict(zip(dashboard.COLUMNS, cells))
        for column, value in zip(results.COLUMNS, line.split(" | "), strict=True):
            assert cell[column] == escape(value), (line, column)


def test_the_reference_layer_is_drawn_labelled_with_its_source(tmp_path):
    path = tmp_path / "reference.json"
    curve = [
        {"concurrency": 4, "throughput per GPU": 900.0, "p90 interactivity": 40},
        {"concurrency": 8, "throughput per GPU": 1500.0, "p90 interactivity": 25},
    ]
    path.write_text(json.dumps({"source": SOURCE, "curves": {"TP8 MTP3": curve}}))

    _, page = dashed(tmp_path, {}, dashboard.reference(path))

    assert labelled(page) == SOURCE
    inferencex = page.split('<div class="y0x0">')[1].split("</svg>")[0]
    titles = re.findall(r"<title>([^<]*)</title><polygon class=\"refpt\"", inferencex)
    assert [t.split("\n")[0] for t in titles] == [f"{SOURCE}: TP8 MTP3"] * 2
    # It states no E2E-normalized interactivity, so that view draws none of it.
    other = page.split('<div class="y0x1">')[1].split("</svg>")[0]
    assert 'class="refpt"' not in other


@pytest.mark.parametrize(
    "layer, why",
    [
        ({"curves": {}}, "a reference layer is an object of source and curves"),
        ({"source": " ", "curves": {}}, "its source is not a non-empty string"),
        ({"source": SOURCE, "curves": []}, "its curves are not names, each to a list"),
        ({"source": SOURCE, "curves": {"a": [1]}}, "its curves are not names, each to"),
        (
            {"source": SOURCE, "curves": {"a": [{"p99 TTFT": 1}]}},
            "curve 'a' names unknown fields ['p99 TTFT']",
        ),
        (
            {"source": SOURCE, "curves": {"a": [{"p90 TTFT": "1"}]}},
            "curve 'a' holds no number at ['p90 TTFT']",
        ),
    ],
)
def test_a_reference_layer_not_of_its_shape_is_refused_by_name(tmp_path, layer, why):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(layer))

    with pytest.raises(ValueError, match=re.escape(f"{path}: {why}")):
        dashboard.reference(path)


def test_compass_advice_ends_in_its_dashboard_and_a_changed_slo_draws_it_again(
    tmp_path, simulating, monkeypatch, capsys  # noqa: F811
):
    """`compass advice` on a, b and m as `test_advice_results` runs it, each
    point's client stub leaving its report beside its export."""
    for name, text in (("server", SERVER), ("client", CLIENT)):
        (tmp_path / name).write_text(text.format(python=sys.executable))
        (tmp_path / name).chmod(0o755)
    monkeypatch.setattr(check, "COMPASS", tmp_path / "server")
    monkeypatch.setattr(simulate, "CLIENT", tmp_path / "client")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))
    summaries = {"*": SUMMARY, "b-c4": COVERAGE}
    (tmp_path / "summaries.json").write_text(json.dumps(summaries))
    (tmp_path / "run_file.json").write_text(json.dumps({"machine": DOCUMENT}))
    for point, scale, tp in (("a-c1", 1, 1), ("a-c4", 1.5, 1), ("b-c1", 0.8, 2)):
        made(tmp_path / "made" / point, int(point[-1]), scale, tp, summary=None)
    made(tmp_path / "made" / "b-c4", 4, 0.5, summary=None)
    for point in ("a-c1", "a-c4", "b-c1", "b-c4"):
        (tmp_path / "made" / point / dashboard.REPORT).write_text("<p>report</p>")
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
    out, reference = tmp_path / "out", tmp_path / "reference.json"
    reference.write_text(json.dumps({"source": SOURCE, "curves": {}}))

    def advise(slo: dict, *flags: str) -> str:
        (tmp_path / "space.json").write_text(json.dumps(space | {"slo": slo}))
        cli.main(["advice", str(tmp_path / "space.json"), str(out), *flags])
        return (out / dashboard.FILE).read_text()

    page = advise({"max_ttft_p90_s": 2}, "--reference", str(reference))

    assert f"the dashboard in {out / dashboard.FILE}\n" in capsys.readouterr().out
    points = drawn(page)["throughput per GPU", "p90 interactivity"]
    assert set(points) == {"a-c1", "a-c4", "b-c1", "b-c4"}
    assert all(m["href"] == f"runs/{n}/{dashboard.REPORT}" for n, m in points.items())
    assert '<circle class="ring" r="13"/>' in points["a-c1"]["mark"]
    assert '<circle class="ring runner" r="13"/>' in points["b-c1"]["mark"]
    assert 'class="refused"' in points["b-c4"]["mark"]
    assert "2 of 6 points have no value on these axes" in page
    assert labelled(page) == SOURCE
    head = json.loads((out / "reuse.json").read_text())["head"]
    assert f"{head['commit']} ({head['ref']}), ATOM digest {head['atom']}" in page
    ran = len(starts(tmp_path))

    page = advise({"max_ttft_p90_s": 1, "min_interactivity_p90": 30})

    # The SLO alone changed: nothing started, the page drawn again without the
    # layer it was not given, b-c1 now recommended with no runner-up.
    assert len(starts(tmp_path)) == ran
    points = drawn(page)["throughput per GPU", "p90 interactivity"]
    assert '<circle class="ring" r="13"/>' in points["b-c1"]["mark"]
    assert "ring" not in points["a-c1"]["mark"]
    assert labelled(page) is None and "Reference" not in page
    reference.write_text(json.dumps({"source": SOURCE}))

    with pytest.raises(SystemExit, match=f"compass advice: {reference}: "):
        advise({"max_ttft_p90_s": 2}, "--reference", str(reference))

    # A refused layer writes nothing: the last search stands.
    assert (out / dashboard.FILE).read_text() == page
    assert (
        json.loads((out / "candidates.json").read_text())["slo"]["max_ttft_p90_s"] == 1
    )

    def stopped(*_):
        raise SystemExit("compass advice: stopped by signal 15")

    with monkeypatch.context() as stopping, pytest.raises(SystemExit):
        stopping.setattr(simulate, "simulate", stopped)
        advise({"max_ttft_p90_s": 1, "min_interactivity_p90": 30})

    # A search stopped before its outcomes leaves no dashboard of the last one.
    assert not (out / dashboard.FILE).exists()
