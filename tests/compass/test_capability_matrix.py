# SPDX-License-Identifier: MIT
"""The capability matrix (`08` D120), with no driver.

Every scenario case's mark names a form `capabilities.FORMS` holds, at level 0
or 1; every supported form has a level-0 case and every refused form's refusal
test collects, unless its entry names the open issue that adds it, which it
names only until that case or test lands.
`scripts/compass/capability_table.py` reads the marks off pytest's collection
of `tests/compass`, run here as a reader runs it, and names each defect; the six
slices each hold a level-1 case of their forms. A level-1 case's run file names
the step roofline alone, on a machine that prices the slice model.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from capabilities import (
    FORMS,
    ROOFLINE,
    ROOFLINE_MACHINE,
    Form,
    capability,
    cost_overrides,
    refused,
)
from test_backend_step_linear import _installed
from test_vertical_slice import CONSTANT, _run_file

from atom.compass.backends.step_roofline import StepRooflineBackend

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts" / "compass" / "capability_table.py"
_spec = importlib.util.spec_from_file_location("capability_table", SCRIPT)
table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(table)

#: Each slice file and the forms it holds a level-1 case of.
SLICES = {
    "test_vertical_slice.py": {"tp1"},
    "test_tp_slice.py": {"tp_above_1", "pcp", "dcp"},
    "test_dp_slice.py": {"dp_attention"},
    "test_pd_slice.py": {"pd_1p1d"},
    "test_pd_dp.py": {"pd_dp"},
    "test_multinode_dp.py": {"multinode_dp"},
}


@pytest.fixture(scope="module")
def printed():
    """What the script prints as JSON over this tree's tests/compass."""
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.stdout, done.stderr[-4000:]
    return done.returncode, json.loads(done.stdout.splitlines()[-1])


def test_every_mark_names_a_form_and_every_form_has_its_case_or_refusal_test(
    printed,
):
    code, out = printed
    print("\n" + "\n".join(table.rows(out["cases"])))
    assert out["findings"] == []
    assert code == 0


def test_each_slice_holds_a_level_0_and_a_level_1_case_of_its_forms(printed):
    _, out = printed
    held = {
        (c["case"].split("::")[0].rpartition("/")[2], c["name"], c["level"])
        for c in out["cases"]
    }
    for name, forms in SLICES.items():
        for form in forms:
            assert {(name, form, 0), (name, form, 1)} <= held, (name, form)
    assert all(out["levels"][f] == 1 for forms in SLICES.values() for f in forms)


# --- each finding, on a vocabulary of four forms -------------------------------

VOCABULARY = {
    "a": Form("supported"),
    "b": Form("supported, its first case to come", issue="#1"),
    "r": refused("refused", "t.py::test_r"),
    "q": refused("refused, its refusal test to come", issue="#2"),
}
A0 = {"case": "t.py::test_a", "name": "a", "level": 0}
R = ["t.py::test_r[x]"]


@pytest.mark.parametrize(
    "cases, ids, errors, found",
    [
        ([A0], R, [], []),
        ([A0, {**A0, "name": "z"}], R, [], ["t.py::test_a: marks 'z', a form"]),
        ([A0, {**A0, "level": 2}], R, [], ["t.py::test_a: marks 'a' at level 2"]),
        ([A0, {**A0, "name": "r"}], R, [], ["t.py::test_a: marks 'r', a refused"]),
        ([{**A0, "level": 1}], R, [], ["a: supported, and no case of it collects"]),
        ([A0], ["t.py::test_rr"], [], ["r: refused, and its refusal test 't.py"]),
        ([A0], R, ["u.py"], ["u.py: does not collect"]),
    ],
    ids=["none", "unknown", "level", "on-refused", "no-level-0", "no-test", "error"],
)
def test_the_table_names_each_defect_and_spares_a_form_an_open_issue_holds(
    monkeypatch, cases, ids, errors, found
):
    monkeypatch.setattr(table.capabilities, "FORMS", VOCABULARY)
    got = table.findings(cases, ids, errors)
    assert len(got) == len(found), got
    assert all(g.startswith(f) for g, f in zip(got, found)), got


def test_an_open_issue_spares_a_form_only_until_its_case_or_refusal_test_lands(
    monkeypatch,
):
    vocabulary = {**VOCABULARY, "q": refused("refused", "t.py::test_q", issue="#2")}
    monkeypatch.setattr(table.capabilities, "FORMS", vocabulary)
    b1 = {"case": "t.py::test_b", "name": "b", "level": 1}
    found = [f.split(";")[0] for f in table.findings([A0, b1], R + ["t.py::test_q"])]
    assert found == [
        "b: names #1 to add its first case, which it has",
        "q: names #2 to add its refusal test, which it has",
    ]


def test_a_log_counts_only_a_case_it_reports_passed(monkeypatch):
    monkeypatch.setattr(table.capabilities, "FORMS", VOCABULARY)
    log = (
        "t.py::test_a PASSED                         [ 20%]\n"
        "t.py::test_a1[x] FAILED                     [ 40%]\n"
        "t.py::test_a1[y] SKIPPED (no driver)        [ 60%]\n"
        "t.py::test_b PASSED                         [ 80%]\n"
        "t.py::test_b ERROR                          [ 80%]\n"
        "t.py::test_b0 PASSED                        [100%]\n"
        "FAILED t.py::test_b0 - AssertionError\n"
    )
    cases = [A0] + [
        {"case": f"t.py::{t}", "name": n, "level": lv}
        for t, n, lv in [
            ("test_a1[x]", "a", 1),
            ("test_a1[y]", "a", 1),
            ("test_b", "b", 1),
            ("test_b0", "b", 0),
        ]
    ]
    passed = table.read_log(log)
    assert passed == {
        "t.py::test_a": True,
        "t.py::test_a1[x]": False,
        "t.py::test_a1[y]": False,
        "t.py::test_b": False,
        "t.py::test_b0": False,
    }
    assert table.levels(cases) == {"a": 1, "b": 1}
    counted = table.counted(cases, passed)
    assert table.levels(counted) == {"a": 0}
    rows = dict(r.split(maxsplit=1) for r in table.rows(counted)[1:])
    assert rows["a"].split()[:2] == ["0", "1/0"]
    assert rows["b"].startswith("none") and rows["b"].endswith("waits on #1")
    assert rows["r"].split() == ["refused", "0/0", "t.py::test_r"]
    assert rows["q"].endswith("its refusal test waits on #2")


def test_every_form_is_supported_or_refused_with_its_test_or_issue():
    for name, form in FORMS.items():
        assert name.isidentifier() and form.what, name
        assert form.refused or not form.test, name
        assert form.test.count("::") == 1 or form.issue or not form.refused, name
        assert not form.issue or form.issue.startswith("#"), name
    assert capability("tp1", 1).args == ("tp1", 1)


# --- the run file of each level -------------------------------------------------


def test_a_level_0_run_file_names_the_stub_and_a_level_1_the_roofline_alone(
    tmp_path,
):
    (tmp_path / "0").mkdir()
    (tmp_path / "1").mkdir()
    stub = json.loads(
        _run_file(tmp_path / "0", **cost_overrides(CONSTANT, {})).read_text()
    )
    roof = json.loads(
        _run_file(
            tmp_path / "1", **cost_overrides(ROOFLINE, ROOFLINE_MACHINE)
        ).read_text()
    )
    assert stub["step_stub"] == CONSTANT and stub["machine"] == {}
    assert ROOFLINE not in stub
    assert "step_stub" not in roof and roof[ROOFLINE] is True
    assert roof["machine"] == ROOFLINE_MACHINE


def test_run_runner_installs_the_step_roofline_alone_on_the_level_1_machine(
    monkeypatch, tmp_path
):
    installed = _installed(
        monkeypatch, tmp_path, **cost_overrides(ROOFLINE, ROOFLINE_MACHINE)
    )
    assert type(installed) is StepRooflineBackend
    assert installed.name == ROOFLINE_MACHINE["name"]


def test_the_level_1_machine_prices_the_slice_model_at_tp1_with_no_refusal(tmp_path):
    """Qwen3-0.6B, the slice model, built on the meta device as
    `test_run_step_roofline_alone` builds it, priced under the roofline alone
    on `ROOFLINE_MACHINE`: one 512-token prefill above a four-row decode."""
    from test_run_step_roofline_alone import CONFIG, RUN

    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(CONFIG.read_text())
    (tmp_path / "machine.json").write_text(json.dumps(ROOFLINE_MACHINE))
    done = subprocess.run(
        [sys.executable, "-c", RUN, str(tmp_path), str(tmp_path / "machine.json")]
        + [str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    priced = json.loads(done.stdout.splitlines()[-1])
    print(f"\n{priced}")
    assert priced["installed"] == StepRooflineBackend.__name__
    assert priced["unclassified"] == []
    assert priced["prefill_s"] > priced["decode_s"] > 0
