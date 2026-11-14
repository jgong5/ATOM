# SPDX-License-Identifier: MIT
"""What grading keeps, `python -m atom.compass.validate CELL...`: the rows it
prints, in the cell's `grades.json`, and the report of each of the cell's runs
rendered again to show them. The cells are `test_validate_main`'s, laid out as
`record` and `run` write them, each run's report written as a run writes its
own when it ends.
"""

import json
import shutil

from test_report_html import _Page
from test_validate_main import _as_recorded, _rows

from atom.compass.report import html
from atom.compass.validate.__main__ import main

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
    assert (grades["cell"], grades["runs"]) == ("c4", RUNS)
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
