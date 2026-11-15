# SPDX-License-Identifier: MIT
"""Cost accuracy: a law, a step grid or an op graph, priced on a real run's own step sequence.

    python -m atom.compass.validate.cost RECORD LAW

RECORD is a real run's step record directory (`ATOM_COMPASS_PARITY_RECORD`,
`OUT_DIR/steps` of a `REAL=1 cctraces_sim.sh` cell); LAW is a law file as
`step_linear_fit` writes it. Pricing the real run's own steps separates cost error
from schedule error: both sides run the same batches. The command prices every
step with a law; `python -m atom.compass.validate` prices only those of the client
run it grades, its warmup left out, with the backend the simulated run priced with.

Each step is labelled as the fit labels it, by its device time
(`step_labels.label_run` under ATOM's split rule, `bound._split`, which also says
which steps carry no label and refuses a record without device windows), and priced by `StepLinearBackend`, `StepGridBackend` or `OpGraphBackend`. The
error of a step is `priced / real - 1`, its target price over its target label on
an MTP record. A step that also carries a draft label is graded again under kind
`draft`, its draft price over that label, validated only if it has draft terms and
each is. Per kind the report gives the error
on totals, on totals holding out the step that contributes most to their error,
and the median per step; per decode rung or prefill token band, the p10, p50
and p90 of the per-step error and the fraction of steps within 10%.

The graded figures hold every step whose priced rank has no declared target
term, at every backend. The same figures are given beside them over its
validated steps and over its unvalidated ones: those whose terms are not all
validated (outside the hull of the law's held-out steps, every step of a law
that records none; for the grid, outside its residual's held-out hull). A step
with a declared term is reported apart, in no graded figure; a draft is
grouped by its draft terms alike. A step that ran more
than `step_linear_fit.WARM_FACTOR` times its price is warm, a one-off first touch the
law does not model: it is named with its seconds and enters no figure. So are,
under a law or an op graph, the steps that lead a vector's priced steps, each
slower than its price by more than `step_linear_fit.MAD_K` scaled MADs of the
errors the law's vector was fitted on, as the fit names them; of that vector's
errors here if the law records no `fit.mad` or the backend is an op graph. A step
grid, bare or below a fallback, has no such rule: its bundle's points are measured
after the bench warm-up and its fit names no warm step, so only a step above that
factor is warm. A step the backend declines is counted
under `refused` with its real seconds and its reasons, the most frequent named.
A step whose priced rank a fallback answered after a source declined it
(`StepCost.refusals`), as the step roofline below a step grid answers, is
graded by its terms and also counted under `fallback`, with what declined it.
A step whose priced rank the backend extrapolated, past its support domain, is
counted under `extrapolated`. An answer of any
method but an estimated one outside its hull is validated, as the run summary
groups seconds: an op graph's analytical and measured leaves are, its
interpolated ones only inside a held-out bracket on each line they interpolate
along, and are unvalidated otherwise.

Per kind the report also gives the priced seconds of those steps by method
(`backends.cost.GROUPS`), and per cost term, an op graph's op kind its
``op.<kind>`` and ``op.<kind>.price_list`` terms together, the term's seconds
over the measured ones: on totals, and the p10, p50 and p90 per step. Summed
over the terms, they are the composed step over the measured one.

On a record of several DP ranks it gives, per kind and per rank, the graded
steps that rank was priced as, their error on totals and the median per step,
with the PCI buses of the rank's cards that the run.json beside the record
names (`identity._device`): a law prices the card mix of its records, and this
split states what a rank's card leaves. ATOM runs DP rank r on the r-th equal slice of the GPUs its server
saw; a run.json that names no bus, as one written before it named any, leaves
each rank's cards unnamed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atom.compass.backends import step_labels
from atom.compass.backends.accuracy import run_accuracy
from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.drive import aiter_jit
from atom.compass.validate.bound import _split

REASONS = 3  # a declined band names this many of its most frequent reasons


def _cards(run: dict | None, ranks: list[int]) -> dict:
    """The PCI buses of each DP rank's cards in `ranks`, rank r the r-th of
    as many equal slices of the GPUs `run`, the record's run.json, names; None
    for each when it names no bus or its GPUs do not split so."""
    gpus = (run or {}).get("device", [])
    span, rest = divmod(len(gpus), len(ranks))
    if not span or rest or any("bus" not in g for g in gpus):
        return dict.fromkeys(ranks)
    return {
        r: [g["bus"] for g in gpus[r * span : (r + 1) * span]] or None for r in ranks
    }


def cost_accuracy(record, backend, steps=None) -> dict:
    """The report for `backend`, a `CostBackend` or a law mapping
    `StepLinearLaw` reads, on the record directory `record`: on its steps whose
    numbers are in `steps`, every step when None. Its ``backend`` names the
    law's provenance, or the backend's `describe()` line.

    The ranks' records (`dp0.jsonl` up) are joined as `step_linear_fit` joins
    them: a step is its ranks' lines of one step index, measured as the
    smallest rank label and priced, banded and validated as the rank
    `step_linear_fit.priced_rank` picks; a step the law refuses is banded as
    its rank of most tokens. Its draft is measured and priced the same way. A
    rank that ran the step as a dummy forward is priced 0 s: the step is
    measured, priced and banded by the ranks that ran a forward, and a rank
    that ran only dummy forwards adds no step. A step some
    rank lacks otherwise or cut, or whose
    ranks fall to different vectors, is counted under `cut.ranks`. Ranks whose
    records differ in length by more than the one step a stopped run leaves on
    some ranks only are refused, and so is a record whose every rank ran only
    dummy forwards.
    So is a record whose run.json, beside `record`, names `aiter_jit_builds`.
    On several ranks, the report's ``ranks`` holds per kind and rank number
    the graded steps that rank was priced as (`accuracy._rank`), with its cards' buses.
    """
    run = Path(record).resolve().parent / "run.json"
    identity = read_json(run) if run.is_file() else None
    if identity is not None and (built := aiter_jit.refusal(identity)):
        raise ValueError(f"{record} is not costed: {built}")
    ranks = sorted(Path(record).glob("dp*[0-9].jsonl"))
    records = [read_jsonl(r) for r in ranks]
    lines = {r.name: len(x) for r, x in zip(ranks, records)}
    if not ranks or max(lines.values()) - min(lines.values()) > 1:
        raise ValueError(
            f"{record} holds rank records of {lines} lines; one run's ranks "
            "differ by at most the step a stopped run leaves on some"
        )
    run = [got for got, _ in step_labels.label_run(ranks, split_rule=_split)]
    if not any(run):
        raise ValueError(
            f"{record}'s ranks ran only DP's dummy forwards: no step to price"
        )
    dummies = [frozenset(s["step"] for s in x if s["is_dummy_run"]) for x in records]
    number = [int(r.name[2:-6]) for r in ranks]
    cards = _cards(identity, number)
    return {"record": str(record), **run_accuracy(run, backend, steps, dummies, cards)}


def declined_text(r: dict) -> str:
    """A declined band's steps, real seconds and most frequent reasons."""
    reasons = r["reasons"].most_common()
    named = json.dumps(dict(reasons[:REASONS]))
    more = f" and {len(reasons) - REASONS} more" if len(reasons) > REASONS else ""
    return f"{r['steps']} steps, real {r['real_s']:.2f} s; reasons {named}{more}"


def _text(report: dict) -> str:
    out = [f"record {report['record']}", f"law {report['backend']}"]
    out.append(f"not labelled: {json.dumps(report['cut'])}")
    for group, title in (
        ("graded", "graded: every step with no declared term"),
        ("validated", "of which validated: inside the held-out hull"),
        ("unvalidated", "of which unvalidated"),
        ("declared", "declared, apart"),
    ):
        out.append(f"\n{title}")
        for kind, s in report[group].items():
            held = "-" if s["held_out"] is None else f"{s['held_out']:+.2%}"
            out += [
                (
                    f"{kind}  {s['steps']} steps  real {s['real_s']:.2f} s"
                    f"  priced {s['priced_s']:.2f} s"
                ),
                f"  on totals              {s['on_totals']:+.2%}",
                f"  holding out step {s['held_out_step']:<6}{held}",
                f"  median per step        {s['median']:+.2%}",
                "  band | steps | p10 | p50 | p90 | within 10%",
            ]
            for band, b in s["bands"].items():
                out.append(
                    f"  {band} | {b['steps']} | {b['p10']:+.1%} | {b['p50']:+.1%}"
                    f" | {b['p90']:+.1%} | {b['within_10pct']:.1%}"
                )
    for kind, ranks in report["ranks"].items():
        out.append(f"\n{kind} by priced rank")
        out.append("  rank | cards | steps | on totals | median")
        for rank, r in ranks.items():
            cards = ", ".join(r["buses"]) if r["buses"] else "unrecorded"
            out.append(
                f"  dp{rank} | {cards} | {r['steps']} | {r['on_totals']:+.2%}"
                f" | {r['median']:+.2%}"
            )
    for w in report["warm"]:
        out.append(
            f"\nwarm step {w['step']} ({w['band']}): real {w['real_s']:.2f} s,"
            f" priced {w['priced_s']:.2f} s"
        )
    for kind in ("fallback", "refused"):
        for band, r in report[kind].items():
            out.append(f"\n{kind} {band}: {declined_text(r)}")
    out.append(f"\nextrapolated: {report['extrapolated']} steps")
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.validate.cost",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("record", help="a real run's step record directory")
    parser.add_argument("law", help="a law file, as step_linear_fit writes it")
    args = parser.parse_args(argv)
    print(_text(cost_accuracy(args.record, json.loads(Path(args.law).read_text()))))


if __name__ == "__main__":
    main()
