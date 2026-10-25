# SPDX-License-Identifier: MIT
"""A step linear law fitted on an MTP record prices each forward's draft apart,
as terms tagged `draft`, and a speculative run charges them."""

import dataclasses
import json

import pytest
from test_backend_step_linear_fit import (
    HOST,
    _fit,
    _record,
    _steps,
    _timed_record,
    _view,
)
from test_dp_step_max import prefill, runner
from test_speculative_step import _speculative
from test_validate_cost import LAW, _two_ranks
from test_validate_cost import _record as _rank_record

from atom.compass.backends import (
    BatchView,
    CostRefused,
    CostTerm,
    Method,
    Provenance,
    RequestShape,
    StepCost,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_labels import digest, label
from atom.compass.backends.step_linear_fit import (
    fit,
    fit_host,
    label_records,
    mark_held_out,
)
from atom.compass.parity import recurrence
from atom.compass.validate.cost import cost_accuracy

#: Each forward's draft, as a share of its target.
SHARE = 0.1


def _output(rows, rung=None):
    """`_view` on a step that produces output, which a draft law's prefill reads."""
    return dataclasses.replace(_view(rows, rung), produces_output=True)


def _mtp_record(path, steps):
    """`steps` as an MTP record: each forward's target its price, then a draft of
    `SHARE` of it."""
    _record(path, [(rows, rung, s * (1 + SHARE)) for rows, rung, s in steps])
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line, (_, _, s) in zip(lines, steps):
        line["d_target_ns"] = round(s * 1e9)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


def test_an_mtp_record_fits_a_draft_law_that_prices_each_forward_apart(tmp_path):
    steps = _steps(3)
    law = _fit([_mtp_record(tmp_path / "dp0.jsonl", steps)])
    assert "draft" not in _fit([_record(tmp_path / "plain.jsonl", steps)])

    backend = StepLinearBackend(StepLinearLaw(law))
    for rows, rung, seconds in steps[:-1]:
        cost = backend.price(_output(rows, rung))
        assert cost.target_seconds == pytest.approx(seconds, rel=1e-5)
        assert cost.draft_seconds == pytest.approx(SHARE * seconds, rel=1e-5)
        drafts = [t.name for t in cost.draft_terms]
        assert drafts and all(n.startswith("draft.") for n in drafts)

    # A middle chunk's draft is its draft-KV pass, priced by the draft vector.
    row = (RequestShape(512, 512, False),)
    cost = backend.price(BatchView(row, produces_output=False))
    final = backend.price(BatchView(row, produces_output=True))
    assert cost.draft_seconds == final.draft_seconds > 0
    assert cost.target_seconds > 0


def test_an_mtp_record_s_middle_chunk_without_a_target_end_is_cut_unsplit(tmp_path):
    """Its device label holds its draft-KV pass, which the draft law prices
    again: a record from before the stamp, or a pass nothing stamps."""
    record = _mtp_record(tmp_path / "dp0.jsonl", _steps(3))
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    lines[0]["produces_output"] = False
    del lines[0]["d_target_ns"]
    record.write_text("".join(json.dumps(x) + "\n" for x in lines))

    samples = label(record)[0]
    assert [(s.step, s.cut, s.device_s) for s in samples if s.cut] == [
        (0, "unsplit", None)
    ]
    assert _fit([record])["cut"]["unsplit"] == 1


def test_a_speculative_run_charges_the_law_s_draft_and_refuses_a_law_without_one(
    tmp_path,
):
    steps = _steps(3)
    law = _fit([_mtp_record(tmp_path / "dp0.jsonl", steps)])
    priced = StepLinearBackend(StepLinearLaw(law)).price(_output([(8, 8, False)]))
    assert priced.draft_seconds > 0
    tagged = _speculative(runner(StepLinearBackend(StepLinearLaw(law))))
    assert tagged.forward(prefill(8)).predicted_s == pytest.approx(priced.seconds)
    assert tagged.refused_commands() == ()

    del law["draft"]
    plain = _speculative(runner(StepLinearBackend(StepLinearLaw(law))))
    assert plain.forward(prefill(8)).predicted_s == 0.0
    assert plain.refused_commands() == ("cost:StepLinearBackend: prices no draft",)


def test_cost_accuracy_grades_the_draft_apart_and_unvalidated(tmp_path):
    """The draft law records no held-out errors: its terms leave a step's
    validation and price to the target's, and its own grade is unvalidated."""
    records = [_mtp_record(tmp_path / f"dp0-{s}.jsonl", _steps(s)) for s in (3, 4)]
    samples = label_records(records)[0]
    law, report = fit(samples)
    mark_held_out(law, report, samples)
    (tmp_path / "steps").mkdir()
    _mtp_record(tmp_path / "steps" / "dp0.jsonl", _steps(5))
    report = cost_accuracy(tmp_path / "steps", law)
    for kind in ("prefill", "decode"):
        assert report["validated"][kind]["on_totals"] == pytest.approx(0.0, abs=1e-5)
    assert "draft" not in report["validated"]
    draft = report["unvalidated"]["draft"]
    assert draft["on_totals"] == pytest.approx(0.0, abs=1e-5)
    targets = sum(
        s["real_s"]
        for group in ("validated", "unvalidated")
        for kind, s in report[group].items()
        if kind != "draft"
    )
    assert draft["real_s"] == pytest.approx(SHARE * targets, rel=1e-5)

    del law["draft"]  # a law that prices no draft grades it as 0 s, unvalidated
    report = cost_accuracy(tmp_path / "steps", law)
    assert "draft" not in report["validated"]
    assert report["unvalidated"]["draft"]["on_totals"] == -1.0


def _drafted(tmp_path):
    """A record of drafts and the draft law fitted on two others."""
    records = [_mtp_record(tmp_path / f"dp0-{s}.jsonl", _steps(s)) for s in (3, 4)]
    samples = label_records(records)[0]
    law, report = fit(samples)
    mark_held_out(law, report, samples)
    (tmp_path / "steps").mkdir()
    _mtp_record(tmp_path / "steps" / "dp0.jsonl", _steps(5))
    return tmp_path / "steps", law


def test_cost_accuracy_grades_an_unvalidated_draft(tmp_path):
    report = cost_accuracy(*_drafted(tmp_path))
    assert report["graded"]["draft"]["steps"] > 0
    assert report["graded"]["draft"] == report["unvalidated"]["draft"]


class _DeclaredDraft:
    """A law's prices with each draft term declared."""

    def __init__(self, law):
        self.law = StepLinearBackend(StepLinearLaw(law))

    def price(self, view):
        cost = self.law.price(view)
        said = Provenance(Method.DECLARED, note="stated")
        drafts = [dataclasses.replace(t, provenance=said) for t in cost.draft_terms]
        return StepCost(cost.terms, cost.treatments, drafts)

    def describe(self):
        return "planted, with a declared draft"


def test_a_draft_with_a_declared_term_is_reported_apart_only(tmp_path):
    record, law = _drafted(tmp_path)
    report = cost_accuracy(record, _DeclaredDraft(law))
    assert list(report["declared"]) == ["draft"]
    assert report["declared"]["draft"]["steps"] > 0
    assert all("draft" not in report[g] for g in ("graded", "validated", "unvalidated"))
    assert set(report["graded"]) == {"prefill", "decode"}


class _DraftDearestWhenCheapest:
    """`LAW`'s prices plus a draft that is dearest on the cheapest target."""

    def __init__(self):
        self.law = StepLinearBackend(StepLinearLaw(LAW))

    def price(self, view):
        cost = self.law.price(view)
        made = cost.terms[0].provenance
        draft = CostTerm("draft", 1.0 / cost.target_seconds, made)
        return StepCost(cost.terms, cost.treatments, [*cost.draft_terms, draft])

    def describe(self):
        return "planted, with a draft"


def test_a_group_step_prices_as_its_rank_of_dearest_target_not_of_dearest_step(
    tmp_path,
):
    ranks = _two_ranks()
    for r, steps in enumerate(ranks):
        _rank_record(tmp_path / "steps", steps, rank=r, group=ranks)
    report = cost_accuracy(tmp_path / "steps", _DraftDearestWhenCheapest())
    for kind in ("prefill", "decode"):
        s = report["validated"][kind]
        assert s["steps"] == 10
        assert s["on_totals"] == pytest.approx(0.0, abs=1e-9)


#: A final chunk's `propose` pass, and the tail its draft-KV pass writes for
#: the requests that finish: seconds per step and per sequence, each.
PROPOSE, TAIL = (5e-4, 4e-5), (6e-5, 2e-5)


def test_a_final_chunk_s_propose_pass_and_tail_are_fitted_apart_and_a_middle_chunk_prices_neither(
    tmp_path,
):
    """A prefill step that produces output writes a draft-KV tail and runs
    `propose` after its draft-KV pass; a middle chunk runs the pass alone."""
    steps, plan = _steps(3), []
    for i, (rows, _, s) in enumerate(steps):
        final = not rows[0][2] and i % 4 != 0
        propose = PROPOSE[0] + PROPOSE[1] * len(rows) if final else 0
        tail = TAIL[0] + TAIL[1] * len(rows) if final else 0
        plan.append((i % 4 != 0, s, propose, tail))
    path = tmp_path / "dp0.jsonl"
    _record(
        path,
        [
            (r, g, s * (1 + SHARE) + p + t)
            for (r, g, _), (_, s, p, t) in zip(steps, plan)
        ],
    )
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line, (output, s, propose, _) in zip(lines, plan):
        line.update(produces_output=output, d_target_ns=round(s * 1e9))
        if propose:
            line["d_propose_ns"] = round(propose * 1e9)
    # A final chunk whose draft is all `propose` has no draft-KV label to fit.
    lines[2]["d_target_ns"] = lines[2]["d_busy_ns"] - lines[2]["d_propose_ns"]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    law = _fit([path])
    propose = law["draft"]["propose"]["coefficients"]
    assert (propose["step"], propose["seqs"]) == pytest.approx(PROPOSE, rel=1e-4)
    tail = law["draft"]["prefill"]["coefficients"]
    assert (tail["out"], tail["seqs"]) == pytest.approx(TAIL, rel=1e-4)

    backend = StepLinearBackend(StepLinearLaw(law))
    rows = (RequestShape(512, 512, False), RequestShape(256, 4096, False))
    middle, final = (
        backend.price(BatchView(rows, produces_output=output))
        for output in (False, True)
    )
    assert final.target_seconds == pytest.approx(middle.target_seconds, rel=1e-6)
    draft_kv = SHARE * middle.target_seconds
    assert middle.draft_seconds == pytest.approx(draft_kv, rel=1e-4)
    assert not [t for t in middle.draft_terms if "propose" in t.name]
    assert [
        t.seconds
        for t in middle.draft_terms
        if t.name in ("draft.prefill.seqs", "draft.prefill.out")
    ] == [0, 0]
    assert final.draft_seconds == pytest.approx(
        draft_kv + TAIL[0] + 2 * TAIL[1] + PROPOSE[0] + 2 * PROPOSE[1], rel=1e-4
    )
    with pytest.raises(CostRefused, match="whether it produces output"):
        backend.price(BatchView(rows))
    del law["draft"]["propose"]  # the tail alone depends on it too
    with pytest.raises(CostRefused, match="whether it produces output"):
        StepLinearBackend(StepLinearLaw(law)).price(BatchView(rows))


def _proposing(path, outlier=None):
    """`_mtp_record` of `_steps(3)` whose prefill steps each run `propose`,
    `outlier`'s at three times its price, and whose first is a middle chunk."""
    _mtp_record(path, _steps(3))
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    prefill = [line for line in lines if not line["rows"][0][2]]
    for i, line in enumerate(prefill):
        seconds = PROPOSE[0] + PROPOSE[1] * len(line["rows"])
        line["d_propose_ns"] = round(seconds * (3 if i == outlier else 1) * 1e9)
        line["d_busy_ns"] += line["d_propose_ns"]
    prefill[0]["produces_output"] = False
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path, prefill


def test_a_middle_chunk_s_propose_pass_is_no_propose_label_and_the_fit_states_its_drops(
    tmp_path,
):
    """Under DP lockstep a middle chunk runs `propose` too; it is the draft-KV
    pass's label, not a propose label."""
    path, prefill = _proposing(tmp_path / "dp0.jsonl", outlier=5)
    middle = next(s for s in label(path)[0] if s.step == prefill[0]["step"])
    assert middle.propose_s is None
    line = prefill[0]
    assert middle.draft_s == pytest.approx(
        (line["d_busy_ns"] - line["d_target_ns"]) * 1e-9
    )
    fitted = _fit([path])["draft"]["propose"]["fit"]
    assert (fitted["samples"], fitted["dropped"]) == (len(prefill) - 2, 1)


def test_the_steps_either_law_names_warm_are_the_run_s_warm_steps(tmp_path):
    """A cold target at step 0 and a cold draft at step 9, each 40 times its price."""
    path = _mtp_record(tmp_path / "dp0.jsonl", _steps(3))
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    lines[0]["d_busy_ns"] += 39 * lines[0]["d_target_ns"]
    lines[0]["d_target_ns"] *= 40
    lines[9]["d_busy_ns"] += 39 * (lines[9]["d_busy_ns"] - lines[9]["d_target_ns"])
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    law = _fit([path])
    assert recurrence.warm_steps(law, digest(path)) == {0, 9}
    assert recurrence.warm_steps(law, "another run") == set()


def test_host_laws_fit_an_mtp_decode_whose_tokens_are_its_rows_times_a_width(tmp_path):
    """Each decode row computes the token and its drafts, 6 tokens a row."""
    steps = [
        ([(6 if d else q, n, d) for q, n, d in rows], rung, s)
        for rows, rung, s in _steps(14, n=400)
    ]
    record, _ = _timed_record(tmp_path / "dp0.jsonl", steps)
    host, report = fit_host(label_records([record])[0])
    assert report["unfitted"] == {}
    for segment, c in HOST["decode"].items():
        got = host["laws"]["decode"][segment]
        assert got == pytest.approx(c, rel=1e-4, abs=1e-12), segment


def test_an_mtp_record_s_target_prefill_vector_fits_seqs_and_out(tmp_path):
    """A prefill that produces output runs 2e-3 s more on its target and 3e-4 s
    more a sequence; every fourth step's is a middle chunk."""
    steps = [
        (rows, rung, s + (3e-4 * len(rows) + 2e-3) * (i % 4 > 0 and not rows[0][2]))
        for i, (rows, rung, s) in enumerate(_steps(3))
    ]
    path = _mtp_record(tmp_path / "dp0.jsonl", steps)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for i, line in enumerate(lines):
        line["produces_output"] = bool(i % 4)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    target = _fit([path])["prefill"]["coefficients"]
    assert (target["seqs"], target["out"]) == pytest.approx((3e-4, 2e-3), rel=1e-4)
