# SPDX-License-Identifier: MIT
"""A step linear fit's figures: each run held out of a fit on the others, its
own steps priced by that fit and graded per kind as `validate.cost` grades a
run, and the ATOM digest the records ran at.

The records are `test_backend_step_linear_fit.py`'s planted ones, single-rank,
DP and MTP, each run's labels scaled by its own bias and a per-step jitter so
that its fold misses by a figure a pooled one would dilute. Each fold's law is
fitted here as the fit fits it and graded by `cost_accuracy` on the run's
record directory: the figures the law states are those numbers.
"""

import json
import re

import numpy as np
import pytest
import test_step_linear_dp as dp
from test_backend_step_linear_fit import (
    KEY,
    SPEC,
    _engine_file,
    _fit,
    _record,
    _steps,
    _timed_record,
)
from test_step_linear_draft import _mtp_record

from atom.compass.backends import StepLinearLaw
from atom.compass.backends.step_labels import digest
from atom.compass.backends.step_linear_fit import FIGURES, main
from atom.compass.run import host_key_of
from atom.compass.validate.cost import cost_accuracy

#: Each run's labels over the law's price: the third runs 4% slow.
BIAS = {0: 1.0, 1: 1.0, 2: 1.04}
#: The odd steps of `_steps`, its decodes.
DECODES = set(range(1, 120, 2))


def _jittered(seed, n=120):
    """`_steps(seed, n)`, each step's seconds times its run's bias and a jitter."""
    rng = np.random.default_rng(seed)
    return [
        (rows, rung, s * BIAS[seed] * (1 + rng.uniform(-0.02, 0.02)))
        for rows, rung, s in _steps(seed, n)
    ]


def _graded(directory, law, steps=None) -> dict:
    """`cost_accuracy`'s `FIGURES` per kind of the run in `directory` under `law`."""
    graded = cost_accuracy(directory, law, steps)["graded"]
    return {k: {f: g[f] for f in FIGURES} for k, g in graded.items()}


def _same(got: dict, expected: dict) -> None:
    """`got` and `expected`, `FIGURES` per kind, hold the same kinds and figures."""
    assert set(got) == set(expected)
    for kind, figures in expected.items():
        assert got[kind] == pytest.approx(figures, rel=1e-9), kind


def _fits(tmp_path, write) -> tuple[dict, list]:
    """The law the command writes on one record per seed of `BIAS`, each
    `write(path, seed)` in its own directory, and the records."""
    records = []
    for seed in BIAS:
        (tmp_path / f"r{seed}").mkdir()
        records.append(write(tmp_path / f"r{seed}" / "dp0.jsonl", seed))
    out = tmp_path / "law.json"
    main([*map(str, records), "--out", str(out), *SPEC])
    return json.loads(out.read_text()), records


def _plain(path, seed):
    return _record(path, _jittered(seed))


def _drafted(path, seed):
    return _mtp_record(path, _jittered(seed))


@pytest.mark.parametrize(
    "write, kinds",
    [(_plain, {"prefill", "decode"}), (_drafted, {"prefill", "decode", "draft"})],
    ids=["one rank", "a drafter's draft"],
)
def test_each_held_out_run_states_its_figures_as_the_cost_check_grades_it(
    tmp_path, write, kinds
):
    law, records = _fits(tmp_path, write)

    assert list(law["figures"]) == law["runs"] == [digest(r) for r in records]
    for record in records:
        expected = _graded(record.parent, _fit([r for r in records if r != record]))
        assert set(expected) == kinds
        _same(law["figures"][digest(record)], expected)
    # The slow run, priced by the two others' fit, misses by about its bias.
    slow = law["figures"][digest(records[2])]
    for kind in kinds:
        assert slow[kind]["on_totals"] == pytest.approx(1 / BIAS[2] - 1, abs=0.01)
        assert slow[kind]["median"] == pytest.approx(1 / BIAS[2] - 1, abs=0.01)
    assert StepLinearLaw(law)


def test_a_dp_run_is_graded_at_each_step_s_costliest_rank(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "DP", 2)
    directories, records = [], []
    for seed, bias in BIAS.items():
        rng = np.random.default_rng(seed)
        groups = [
            (views, s * bias * (1 + rng.uniform(-0.02, 0.02)))
            for views, s in dp._groups(seed, n=120)
        ]
        (tmp_path / str(seed)).mkdir()
        records += dp._records(tmp_path / str(seed), groups)
        directories.append(tmp_path / str(seed))
    for path in records:
        _engine_file(path, {"data_parallel_size": 2, "enable_dp_attention": True})
    main([*map(str, records), "--out", str(tmp_path / "law.json"), *SPEC])
    law = json.loads((tmp_path / "law.json").read_text())

    for i, directory in enumerate(directories):
        others = [r for r in records if r.parent != directory]
        _same(law["figures"][law["runs"][i]], _graded(directory, _fit(others, dp=2)))
    slow = law["figures"][law["runs"][2]]
    assert slow["decode"]["on_totals"] == pytest.approx(1 / BIAS[2] - 1, abs=0.01)


def test_a_law_s_sources_are_each_folded_with_their_own_records_form_and_options(
    tmp_path, capsys
):
    """A law whose prefill comes from three runs' base fit and its decode rungs
    from two of them in the history form: each source's figures grade only the
    steps its vectors price, on folds of its own runs."""
    law, records = _fits(tmp_path, _plain)
    rungs = [f"decode {k}" for k in law["decode"]]
    law["sources"] = {
        "three": {"runs": [0, 1, 2], "vectors": ["prefill", "host"]},
        "two": {"runs": [0, 1], "vectors": rungs},
    }
    del law["figures"]
    file = tmp_path / "law.json"
    file.write_text(json.dumps(law))

    def figures(*records, source=None, form="base"):
        flags = ["--source", source] if source else []
        main(
            ["figures", *map(str, records), "--law", str(file), "--form", form, *flags]
        )
        return json.loads(file.read_text())

    figures(*records, source="three")
    written = figures(*records[:2], source="two", form="history")
    assert "figures" not in written
    three, two = (written["sources"][s]["figures"] for s in ("three", "two"))
    assert list(three) == list(map(digest, records))
    assert list(two) == list(map(digest, records[:2]))
    for i, record in enumerate(records):
        others = [r for r in records if r != record]
        prefill = _graded(record.parent, _fit(others), set(range(120)) - DECODES)
        assert set(prefill) == {"prefill"}
        _same(three[digest(record)], prefill)
        if i < 2:
            history = _fit([records[1 - i]], form="history")
            decode = _graded(record.parent, history, DECODES)
            assert set(decode) == {"decode"}
            _same(two[digest(record)], decode)
    assert "run | kind | steps | on totals" in capsys.readouterr().out

    with pytest.raises(ValueError, match=re.escape("--source names None")):
        figures(*records)
    with pytest.raises(ValueError, match="in one and not the other") as refused:
        figures(*records, source="two")
    assert str(refused.value).endswith(f"[{digest(records[2])!r}]")


def test_the_figures_command_restates_the_figures_the_fit_wrote(tmp_path):
    law, records = _fits(tmp_path, _plain)
    file = tmp_path / "law.json"
    file.write_text(json.dumps({**law, "figures": {}}))

    main(["figures", *map(str, records), "--law", str(file)])
    assert json.loads(file.read_text()) == law
    del law["runs"]
    file.write_text(json.dumps(law))
    with pytest.raises(ValueError, match="states no runs to hold the records to"):
        main(["figures", *map(str, records), "--law", str(file)])


def test_the_law_states_the_atom_its_records_ran_at_whatever_its_host_key_is_re_keyed_to(
    tmp_path,
):
    ran = dict(KEY, atom="a" * 64)
    law, _ = _fits(
        tmp_path, lambda path, seed: _timed_record(path, _steps(seed), key=ran)[0]
    )
    assert law["records_atom"] == law["host"]["key"]["atom"] == "a" * 64

    # A re-key moves the host key to this tree's digest and no other field.
    tree = host_key_of("/nonexistent-model", {})["atom"]
    law["host"]["key"]["atom"] = tree
    assert law["records_atom"] == "a" * 64 != tree
    assert StepLinearLaw(law)
