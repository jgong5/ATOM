# SPDX-License-Identifier: MIT
"""A simulated run's extrapolation on each DP group step's critical rank, apart
from every rank's: `report.extrapolation` over step records planted as a
simulated run's runners write them, ``dp<r>.jsonl`` beside
``dp<r>.cost.jsonl``.

`DP2` is a DP 2 group whose every-rank share and critical-rank share part: its
dummy forwards and the cheaper rank of a decode are extrapolated, which the
group's max ignores, so 5 of its 8 rank-steps hold an extrapolated term while
the critical rank's target is extrapolated on 1 of 4 group steps, 0.05 of its
0.61 device seconds.
"""

import json

import pytest

from atom.compass.report import extrapolation
from atom.compass.report.extrapolation import of_run

#: Each group step of a DP 2 record: each rank's kind, ``t_s``, ``device_s``
#: and ``draft_s``, and whether its target and its draft were extrapolated.
DP2 = (
    (
        ("prefill", 0.0, 0.5, 0.0, False, False),
        ("dummy", 0.001, 0.01, 0.0, True, False),
    ),
    (
        ("rung8", 1.0, 0.02, 0.0, True, False),
        ("rung8", 1.001, 0.03, 0.0, False, False),
    ),
    (
        ("rung8", 2.0, 0.05, 0.0, True, False),
        ("dummy", 2.001, 0.01, 0.0, True, False),
    ),
    (
        ("rung8", 3.0, 0.02, 0.01, False, True),
        ("rung8", 3.001, 0.02, 0.0, False, False),
    ),
)
SOURCE = "step-linear"


def _term(name, seconds, extrapolated):
    source = SOURCE + extrapolation.EXTRAPOLATED if extrapolated else SOURCE
    return {
        "name": name,
        "seconds": seconds,
        "method": "estimated",
        "validated": False,
        "source": source,
    }


def cost_line(step, t, device_s, draft_s, target, draft, stated=True) -> dict:
    """A simulated runner's cost line: one target term and, under a draft, one
    draft term; `stated`, with `target_terms`."""
    terms = [_term("decode.tokens", device_s, target)]
    if draft_s or draft:
        terms.append(_term("draft.decode.tokens", draft_s, draft))
    line = {
        "step": step,
        "t_s": t,
        "device_s": device_s,
        "draft_s": draft_s,
        "charge_s": device_s + draft_s,
        "group_s": device_s + draft_s,
        "terms": terms,
    }
    if stated:
        line["target_terms"] = 1
    return line | {"refused": None}


def step_line(step, rank, kind) -> dict:
    """A simulated runner's step line of `kind`: ``prefill``, ``rung<n>``
    (a decode) or ``dummy``."""
    rows, rung = [[64, 64, False]], None
    if kind.startswith("rung"):
        rows, rung = [[1, 100, True]], int(kind[4:])
    line = {"step": step, "dp_rank": rank, "is_dummy_run": kind == "dummy"}
    return line | {"rows": [] if kind == "dummy" else rows, "rung": rung}


def _write(path, lines) -> None:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def plant(record, group_steps=DP2, **costs):
    """Write `group_steps` as a step record in the directory `record`, each
    cost line with `costs` over its fields; `record`."""
    record.mkdir(parents=True, exist_ok=True)
    for r in range(len(group_steps[0])):
        steps = [(k, g[r]) for k, g in enumerate(group_steps)]
        _write(record / f"dp{r}.jsonl", [step_line(k, r, s[0]) for k, s in steps])
        lines = [cost_line(k, *s[1:]) | costs for k, s in steps]
        _write(record / f"dp{r}.cost.jsonl", [{"version": 1}, *lines])
    return record


def share(steps, seconds, of_steps, of_seconds) -> dict:
    return {
        "steps": steps,
        "device_seconds": pytest.approx(seconds),
        "fraction_of_steps": pytest.approx(steps / of_steps) if of_steps else None,
        "fraction_of_device_seconds": (
            pytest.approx(seconds / of_seconds) if of_seconds else None
        ),
    }


def tally(steps, seconds, any_, target, draft) -> dict:
    """A tally of `steps` over `seconds`, each share a ``(steps, seconds)``."""
    return {
        "steps": steps,
        "device_seconds": pytest.approx(seconds),
        "extrapolated": share(*any_, steps, seconds),
        "target_extrapolated": share(*target, steps, seconds),
        "draft_extrapolated": share(*draft, steps, seconds),
    }


def test_the_critical_rank_s_share_is_apart_from_every_rank_s(tmp_path):
    got = of_run(plant(tmp_path / "steps").parent)

    # Every rank's: what `extrapolated_steps` counts, 5 of 8.
    assert got["every_rank"] == tally(8, 0.67, (5, 0.12), (4, 0.09), (1, 0.03))
    assert got["by_rank"] == {
        "dp0": tally(4, 0.6, (3, 0.1), (2, 0.07), (1, 0.03)),
        "dp1": tally(4, 0.07, (2, 0.02), (2, 0.02), (0, 0.0)),
    }
    # The group's max: dp0 on the prefill, dp1 on the first decode, dp0 after;
    # its target is extrapolated on the third alone, its draft on the fourth.
    critical = got["critical_rank"]
    by_kind = critical.pop("by_kind")
    assert critical == tally(4, 0.61, (2, 0.08), (1, 0.05), (1, 0.03))
    assert critical["target_extrapolated"]["fraction_of_device_seconds"] == (
        pytest.approx(0.05 / 0.61)
    )
    assert by_kind == {
        "prefill": tally(1, 0.5, (0, 0.0), (0, 0.0), (0, 0.0)),
        "decode/rung8": tally(3, 0.11, (2, 0.08), (1, 0.05), (1, 0.03)),
    }


def test_a_window_counts_the_group_steps_entering_in_it_and_each_rank_s_own(
    tmp_path,
):
    """The second group step enters at dp0's 1.0, before the window, though
    dp1's step of it enters in it."""
    got = of_run(plant(tmp_path / "steps").parent, (1.0005, 3.0))

    assert got["critical_rank"]["steps"] == 1
    assert got["critical_rank"]["target_extrapolated"]["steps"] == 1
    assert [got["by_rank"][r]["steps"] for r in ("dp0", "dp1")] == [1, 2]
    assert got["every_rank"]["extrapolated"]["steps"] == 2


def test_a_step_line_with_no_time_has_no_place_in_a_window(tmp_path):
    out = plant(tmp_path / "steps", t_s=None).parent

    assert of_run(out)["critical_rank"]["steps"] == 4
    with pytest.raises(ValueError, match=r"step 0 states no time \(t_s\)"):
        of_run(out, (0.0, 1.0))


def test_equal_ranks_leave_the_lowest_one_critical(tmp_path):
    group = (
        (
            ("rung8", 0.0, 0.02, 0.0, False, False),
            ("rung8", 0.0, 0.02, 0.0, True, False),
        ),
    )

    got = of_run(plant(tmp_path / "steps", group).parent)

    assert got["critical_rank"]["extrapolated"]["steps"] == 0
    assert got["every_rank"]["extrapolated"]["steps"] == 1


def test_a_line_stating_no_target_terms_names_its_draft_terms(tmp_path):
    """Before ``target_terms``, a draft term is told by its name, the
    step-linear and step-stub sources' ``draft.`` and the op-graph and
    roofline sources' ``draft ``; a ``draft_kv `` term is the target's."""
    record = plant(tmp_path / "steps", DP2[3:])
    for r in (0, 1):
        path = record / f"dp{r}.cost.jsonl"
        version, line = (json.loads(x) for x in path.read_text().splitlines())
        del line["target_terms"]
        line["terms"] = [
            _term("draft attn", 0.001, r == 0),
            _term("draft_kv attn", 0.001, r == 1),
            _term("draft.decode.tokens", 0.001, False),
        ]
        _write(path, [version, line])

    by_rank = of_run(record.parent)["by_rank"]

    assert by_rank["dp0"]["draft_extrapolated"]["steps"] == 1
    assert by_rank["dp0"]["target_extrapolated"]["steps"] == 0
    assert by_rank["dp1"]["draft_extrapolated"]["steps"] == 0
    assert by_rank["dp1"]["target_extrapolated"]["steps"] == 1


def test_a_deployment_s_records_are_summed_and_named_by_instance(tmp_path):
    plant(tmp_path / "prefill" / "steps", tuple((g[0],) for g in DP2[:2]))
    plant(tmp_path / "decode" / "steps")

    got = of_run(tmp_path)

    assert sorted(got["by_rank"]) == ["decode/dp0", "decode/dp1", "prefill/dp0"]
    assert got["by_rank"]["prefill/dp0"]["steps"] == 2
    assert got["critical_rank"]["steps"] == 2 + 4
    assert got["critical_rank"]["target_extrapolated"]["steps"] == 1 + 1
    assert got["every_rank"]["steps"] == 2 + 8


def test_a_record_being_written_is_read_up_to_its_last_whole_line(tmp_path):
    record = plant(tmp_path / "steps")
    for r in (0, 1):
        with open(record / f"dp{r}.cost.jsonl", "a") as table:
            table.write(json.dumps(cost_line(4, 4.0, 9.0, 0.0, True, False))[:40])

    assert of_run(record.parent)["every_rank"]["steps"] == 8


def test_a_rank_s_step_its_peers_have_not_priced_yet_is_no_group_step(tmp_path):
    record = plant(tmp_path / "steps")
    with open(record / "dp0.cost.jsonl", "a") as table:
        table.write(json.dumps(cost_line(4, 4.0, 9.0, 0.0, True, False)) + "\n")

    got = of_run(record.parent)

    assert got["by_rank"]["dp0"]["steps"] == 5
    assert got["critical_rank"]["steps"] == 4


@pytest.mark.parametrize(
    "files, refusal",
    [
        ({}, r"holds no step record with cost lines"),
        (
            {"pp1.dp0.cost.jsonl": [{"version": 1}]},
            r"is a pipeline's record, whose stages each price their own layers",
        ),
        (
            {"dp0.cost.jsonl": [{"version": 1}, {"step": 0, "d_busy_s": 0.1}]},
            r"holds a real record's labels, not a simulated run's cost lines",
        ),
        ({"dp0.cost.jsonl": [{"step": 0}]}, r"starts with no version line"),
    ],
)
def test_a_run_with_no_simulated_record_to_read_is_refused_by_name(
    tmp_path, files, refusal
):
    if files:
        plant(tmp_path / "steps", DP2[:1])
    for name, lines in files.items():
        _write(tmp_path / "steps" / name, lines)

    with pytest.raises(ValueError, match=refusal):
        of_run(tmp_path)


def test_a_line_that_is_not_json_is_refused_naming_it(tmp_path):
    record = plant(tmp_path / "steps")
    with open(record / "dp1.cost.jsonl", "a") as table:
        table.write("{not json\n")

    with pytest.raises(ValueError, match=r"dp1\.cost\.jsonl line 6 is not JSON"):
        of_run(tmp_path)


def test_the_command_prints_the_tallies_and_names_a_refusal(tmp_path, capsys):
    out = plant(tmp_path / "run" / "steps").parent

    extrapolation.main([str(out), "1.0005", "3.0"])
    assert json.loads(capsys.readouterr().out) == of_run(out, (1.0005, 3.0))
    extrapolation.main([str(out)])
    assert json.loads(capsys.readouterr().out) == of_run(out)
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match=r"refused: .* holds no step record"):
        extrapolation.main([str(tmp_path / "empty")])
    with pytest.raises(SystemExit, match="usage:"):
        extrapolation.main([str(out), "1.0"])
