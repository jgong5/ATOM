# SPDX-License-Identifier: MIT
"""What grading keeps, `python -m atom.compass.validate CELL...`: the rows it
prints, in the cell's `grades.json`, and the report of each of the cell's runs
rendered again to show them while none of its runs ran again, its simulated
run's showing the real runs' per-request splits beside its own. The cells are
`test_validate_main`'s, laid out as `record` and `run` write them, each run's
report written as a run writes its own when it ends.
"""

import json
import shutil

import pytest
from test_report_html import _Page
from test_validate_main import _as_recorded, _rows

from atom.compass.report import html, split
from atom.compass.run import SUMMARY_FILE
from atom.compass.validate import __main__ as validate
from atom.compass.validate import curves
from atom.compass.validate.__main__ import main
from atom.compass.validate.runs import EXPORT

RUNS = ["real0", "real1", "real2", "step_linear"]
SHOWN = f"Cell c4, graded on its runs {', '.join(RUNS)}."


def _cell(capsys, tmp_path):
    """Cell c4 as `record` and `run` write it, ungraded, its simulated run's
    summary as the run's writer states it."""
    cell = tmp_path / "c4"
    _rows(capsys, tmp_path)
    (cell / html.GRADES).unlink(missing_ok=True)
    _as_recorded(cell)
    path = cell / "step_linear" / "summary.json"
    summary = json.loads(path.read_text())
    summary["schedule"] |= {"extrapolated_steps": 0, "fallback_steps": 0}
    summary["schedule"]["refusals"]["reasons"] = []
    path.write_text(json.dumps(summary))
    return cell


def _graded(capsys, cell) -> list[str]:
    main([str(cell)])
    return capsys.readouterr().out.splitlines()


def test_grading_keeps_the_rows_it_prints_and_renders_each_report_again(
    capsys, tmp_path
):
    cell = _cell(capsys, tmp_path)
    wrote = ["real0", "real1", "step_linear"]  # real2 ended before reports existed
    for run in wrote:
        html.write(cell / run)
    before = _Page(cell / "real0").parts

    printed = _graded(capsys, cell)

    assert printed[0] == "cell | result | metric | value | verdict"
    assert before.pop("Grades")["text"] == [
        f"None: {cell} holds no {html.GRADES}; it is not a graded cell."
    ]
    after = {run: _Page(cell / run).parts for run in wrote}
    shown = {run: page.pop("Grades") for run, page in after.items()}
    assert [s["text"] for s in shown.values()] == [[SHOWN]] * len(wrote)
    grades = json.loads((cell / html.GRADES).read_text())
    assert grades["cell"] == "c4"
    assert grades["runs"] == {run: html.digest(cell / run) for run in RUNS}
    assert grades["columns"] == ["result", "metric", "value", "verdict"]
    assert [" | ".join(("c4", *r)) for r in grades["rows"]] == printed[1:]
    assert grades["rows"][0] == ["cell", "verdict", "3 real runs", "graded"]
    for run in wrote:
        head, *rows = shown[run]["tables"][0]
        assert (head, rows) == (grades["columns"], grades["rows"]), run
    # On a report, grading changes its grades alone.
    assert after["real0"] == before
    assert not (cell / "real2" / html.FILE).exists()


def test_the_report_of_a_run_its_cell_gained_after_grading_says_it_was_not_graded(
    capsys, tmp_path
):
    cell = _cell(capsys, tmp_path)
    _graded(capsys, cell)
    shutil.copytree(cell / "real0", cell / "real3")

    page = _Page(html.write(cell / "real3").parent)

    graded = ", ".join(RUNS)
    assert page.parts["Grades"]["text"] == [
        f"None: {cell / html.GRADES} grades {graded}, not this run."
    ]


#: Each run rerun, the one of its `html.RESULTS` the test rewrites, and how: a
#: real run's export with new latencies; a simulated run's summary as a
#: coverage report, alone, though its rerun writes a new export beside it.
RERUNS = {
    "real1": (
        EXPORT,
        {"time_to_first_token": {"unit": "ms", "avg": 1300.0, "p50": 1200.0}},
    ),
    "step_linear": (SUMMARY_FILE, {"coverage_report": True}),
}


def _rerun(cell, run):
    """Rewrite `run`'s file in `RERUNS` as it says."""
    name, change = RERUNS[run]
    path = cell / run / name
    path.write_text(json.dumps(json.loads(path.read_text()) | change))


@pytest.mark.parametrize("rerun", RERUNS)
def test_a_run_rerun_in_its_graded_cell_shows_no_grades_until_graded_again(
    capsys, tmp_path, rerun
):
    cell = _cell(capsys, tmp_path)
    first = _graded(capsys, cell)
    _rerun(cell, rerun)

    graded = ", ".join(RUNS)
    for run in (rerun, "real0"):  # the rerun's own report, and another's
        page = _Page(html.write(cell / run).parent)
        assert page.parts["Grades"]["text"] == [
            f"None: {cell / html.GRADES} grades {graded}; {rerun} changed since."
        ], run

    again = _graded(capsys, cell)
    assert again != first
    shown = _Page(cell / rerun).parts["Grades"]
    assert shown["text"] == [SHOWN]
    assert [" | ".join(("c4", *r)) for r in shown["tables"][0][1:]] == again[1:]


def test_a_run_that_reruns_while_its_cell_is_graded_shows_no_grades(
    capsys, tmp_path, monkeypatch
):
    cell = _cell(capsys, tmp_path)
    html.write(cell / "step_linear")
    rows = validate.rows

    def rerun_once_read(c):
        read = rows(c)
        _rerun(cell, "step_linear")
        return read

    monkeypatch.setattr(validate, "rows", rerun_once_read)
    _graded(capsys, cell)

    assert _Page(cell / "step_linear").parts["Grades"]["text"] == [
        f"None: {cell / html.GRADES} grades {', '.join(RUNS)}; step_linear changed since."
    ]


@pytest.mark.parametrize("rerun", RERUNS)
def test_a_rerun_s_report_renders_the_reports_its_cell_graded_again(
    capsys, tmp_path, monkeypatch, rerun
):
    cell = _cell(capsys, tmp_path)
    wrote = ["real0", "real1", "step_linear"]  # real2 ended before reports existed
    for run in wrote:
        html.write(cell / run)
    _graded(capsys, cell)
    before = {run: _Page(cell / run).parts for run in wrote}
    _rerun(cell, rerun)
    monkeypatch.chdir(cell / rerun)

    html.write(".")  # `python -m atom.compass.report.html .` inside the rerun

    graded = ", ".join(RUNS)
    stale = f"None: {cell / html.GRADES} grades {graded}; {rerun} changed since."
    for run in wrote:
        after = _Page(cell / run).parts
        assert after.pop("Grades") == {"tables": [], "text": [stale]}, run
        before[run].pop("Grades")
        if run != rerun:  # its own report shows its new results too
            assert after == before[run], run
    assert not (cell / "real2" / html.FILE).exists()


def test_a_report_leaves_the_others_while_its_cell_is_ungraded_or_unchanged(
    capsys, tmp_path
):
    cell = _cell(capsys, tmp_path)
    kept = cell / "real0" / html.FILE

    def written(run):
        """real0's report, set to "kept", once `run`'s is written."""
        kept.write_text("kept")
        html.write(cell / run)
        return kept.read_text()

    _rerun(cell, "step_linear")
    assert written("step_linear") == "kept"  # a rerun in an ungraded cell
    _graded(capsys, cell)
    assert written("real1") == "kept"  # a run of the graded cell, unchanged
    shutil.copytree(cell / "real1", cell / "real3")
    assert written("real3") == "kept"  # a run the cell gained after grading


def test_a_cell_and_a_run_given_as_the_working_directory_are_named(
    capsys, tmp_path, monkeypatch
):
    """`cd CELL && python -m atom.compass.validate .`, then a report written
    from inside a run, `python -m atom.compass.report.html .`."""
    cell = _cell(capsys, tmp_path)
    monkeypatch.chdir(cell)
    _graded(capsys, ".")
    assert json.loads((cell / html.GRADES).read_text())["cell"] == "c4"

    monkeypatch.chdir(cell / "real0")
    html.write(".")

    assert _Page(cell / "real0").parts["Grades"]["text"] == [SHOWN]


#: Each split request's segments as shares of its end-to-end latency.
SHARES = {"queue_s": 0.1, "prefill_s": 0.5, "decode_s": 0.4, "e2el_s": 1.0}
#: The end-to-end latencies of the requests each run split.
SPLITS = {"step_linear": [1.0, 2.0, 3.0], "real0": [2.0, 4.0], "real1": [3.0, 5.0, 9.0]}
BESIDE = "A real run's per-request split"


def _requests(e2el: list[float], phase="profiling") -> list[dict]:
    """A split's requests, as `split` writes them, one per latency in `e2el`."""
    tokens = {"lmcache_load_tokens": 0, "lmcache_store_tokens": 0}
    return [
        {"phase": phase} | tokens | {k: s * e for k, s in SHARES.items()} for e in e2el
    ]


def test_a_graded_simulated_run_s_report_shows_each_real_run_s_split_beside_its_own(
    capsys, tmp_path
):
    cell = _cell(capsys, tmp_path)
    for run, e2el in SPLITS.items():
        # A warmup request, which the headline metrics and no percentile count.
        made = {"requests": _requests(e2el) + _requests([60.0], "warmup")}
        made |= {"clock": "virtual", "unsplit": {}}
        (cell / run / split.FILE).write_text(json.dumps(made))
    (cell / "real2" / split.FILE).write_text(json.dumps({"refused": "no steps"}))
    before = {run: _Page(html.write(cell / run).parent).parts for run in SPLITS}

    _graded(capsys, cell)

    ungraded = before["step_linear"].pop(BESIDE)["text"]
    assert ungraded == ["None: a simulated run's directory holds no real run."]
    page = _Page(cell / "step_linear").parts
    shown = (
        "Cell c4: step_linear, this run, beside real0, real1, real2, the real runs"
        " graded with it, each over the requests its headline metrics count."
    )
    assert page[BESIDE]["text"] == [shown]
    head, *rows = page[BESIDE]["tables"][0]
    assert head == ["segment", "run", "requests", "p10 (s)", "p50 (s)", "p90 (s)"]
    assert rows == [
        [html.SEGMENTS[key], run, str(len(e2el))]
        + [f"{curves.percentile([s * e for e in e2el], p):.4g}" for p in (10, 50, 90)]
        for key, s in SHARES.items()
        for run, e2el in SPLITS.items()
    ]
    assert page[BESIDE]["tables"][1] == [
        ["run", "no split because"],
        ["real2", "no steps"],
    ]
    # A real run's report changes its grades alone.
    after = _Page(cell / "real0").parts
    assert after.pop("Grades") != before["real0"].pop("Grades")
    assert after == before["real0"]

    (cell / "real1" / split.FILE).unlink()
    page = _Page(html.write(cell / "step_linear").parent).parts

    assert {r[1] for r in page[BESIDE]["tables"][0][1:]} == {"step_linear", "real0"}
    missing = f"[Errno 2] No such file or directory: '{cell / 'real1' / split.FILE}'"
    assert page[BESIDE]["tables"][1][1:] == [["real1", missing], ["real2", "no steps"]]
