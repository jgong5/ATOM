# SPDX-License-Identifier: MIT
"""A speculative cell's acceptance agreement: the mean acceptance length and
each accepted count's share, the simulated run's against each real run's,
graded against the real runs' spread.

A real run's histogram is the one its engine writes as SIGTERM stops a record
(`run._stop_recording`), a simulated run's the one its run summary states
(`run.engine_done`, `run.frontend_done`), each of ATOM's scheduler serving
`test_speculative_step`'s requests at stated rates, its draws seeded per run.
The cells are laid out as `test_validate_runs` lays them out.
"""

import json
import re
import signal
from types import SimpleNamespace

import pytest
import test_speculative_step
from test_run_refusal_summary import STAGE1, summary
from test_run_speculative_summary import _stated
from test_validate_runs import GRADED, _deployed, _four

from atom.compass import parity
from atom.compass import run as compass_run
from atom.compass.backends import StepStubBackend
from atom.compass.validate.runs import grade, main
from atom.model_engine.engine_stats import EngineStats
from atom.model_engine.scheduler import Scheduler

#: Each position's acceptance rate the real runs serve at, the first `k` at width `k`.
RATES = [0.8, 0.5, 0.3]
MEAN_LENGTH = "mean acceptance length"
#: The engine LP each side of a deployment hands in as.
LP = {"prefill": "engine-P", "decode": "engine-D"}


def _served(monkeypatch, rates, seed):
    """ATOM's scheduler once it has served `test_speculative_step`'s requests
    at `rates`, the stream's draws seeded `seed`."""
    made = []
    monkeypatch.setattr(
        test_speculative_step,
        "Scheduler",
        lambda config: made.append(Scheduler(config)) or made[-1],
    )
    test_speculative_step._serve(rates, seed=seed)
    spec = _stated(monkeypatch, len(rates), synthetic_acceptance_rates=rates)
    made[0].config.speculative_config = spec
    return made[0]


def _idle(monkeypatch, k):
    """A scheduler whose engine drafts `k` tokens a step and verified none."""
    spec = _stated(monkeypatch, k, synthetic_acceptance_rates=[0.5] * k)
    return SimpleNamespace(
        config=SimpleNamespace(speculative_config=spec),
        engine_stats=EngineStats(use_spec=True, mtp_k=k),
    )


def _stopped(monkeypatch, side, scheduler, rank=0):
    """Give `side`'s record the histogram DP rank `rank`'s engine, of
    `scheduler`, writes as SIGTERM stops it."""
    monkeypatch.setenv(parity.ENV, str(side / "steps"))
    scheduler.config.parallel_config = SimpleNamespace(data_parallel_rank=rank)
    engine = SimpleNamespace(scheduler=scheduler)
    compass_run._stop_recording(engine, lambda *_: None, signal.SIGTERM, None)


def _summarized(monkeypatch, cell, schedulers) -> dict:
    """Give `cell` a run summary whose `speculative` entries are, under each
    hand-in name in `schedulers`, the one a run summary states for an engine
    of that scheduler; return them."""
    said = {}
    for name, scheduler in schedulers.items():
        out = cell.parent / f"{cell.name}-{name}"
        out.mkdir()
        stub = StepStubBackend(STAGE1)
        whole = summary(monkeypatch, out, stub, 8, scheduler=scheduler)
        said[name] = whole["speculative"][compass_run.ENGINE.name]
    text = json.dumps(whole | {"speculative": said})
    (cell / compass_run.SUMMARY_FILE).write_text(text)
    return said


def _one_server(monkeypatch, tmp_path, rates=RATES, seed=0):
    """Three real runs of a single server at `RATES`, seeded 0, 1 and 2, and a
    simulated run at `rates` seeded `seed`; its run summary's entry."""
    *real, simulated = _four(tmp_path)
    for i, cell in enumerate(real):
        _stopped(monkeypatch, cell, _served(monkeypatch, RATES, i))
    scheduler = _served(monkeypatch, rates, seed)
    lp = compass_run.ENGINE.name
    return real, simulated, _summarized(monkeypatch, simulated, {lp: scheduler})[lp]


def _held(cell) -> dict:
    """Real `cell`'s one record's accepted-draft histogram."""
    held = json.loads((cell / "steps" / "dp0.acceptance.json").read_text())
    return {int(a): n for a, n in held["distribution"].items()}


def _deployment(monkeypatch, cell, seeds, k, **run):
    """`_deployed`'s deployment `cell`, with `run`'s entries simulated: its
    decode side's DP ranks served at `RATES[:k]`, seeded as `seeds`, and its
    prefill side's drafted `k` tokens a step and verified none. A real one's
    engines write their histograms into its records, a simulated one's run
    summary states them."""
    _deployed(cell, **run)
    held = {}
    for rank, seed in enumerate(seeds):
        held["prefill", rank] = _idle(monkeypatch, k)
        held["decode", rank] = _served(monkeypatch, RATES[:k], seed)
    if run:
        named = {f"{LP[side]}.dp{rank}": s for (side, rank), s in held.items()}
        _summarized(monkeypatch, cell, named)
    for (side, rank), scheduler in held.items() if not run else ():
        _stopped(monkeypatch, cell / side, scheduler, rank)
    return cell


def test_a_speculative_cell_grades_its_mean_acceptance_length_and_each_share(
    monkeypatch, tmp_path
):
    real, simulated, said = _one_server(monkeypatch, tmp_path)
    report = grade(real, simulated)

    accepted = report["acceptance"]
    print("acceptance:", json.dumps(accepted))
    names = [MEAN_LENGTH, *(f"share accepting {a}" for a in range(4))]
    assert list(accepted) == names
    mean = accepted[MEAN_LENGTH]
    assert mean["simulated"] == pytest.approx(said["mean_acceptance_length"])
    for cell, value in zip(real, mean["real"]):
        held = _held(cell)
        rows = sum(held.values())
        assert value == pytest.approx(1 + sum(a * n for a, n in held.items()) / rows)
        for a, n in held.items():
            assert accepted[f"share accepting {a}"]["real"][real.index(cell)] == (
                pytest.approx(n / rows)
            )
    for g in accepted.values():
        # Seeded as the first real run, the simulated run drew its histogram.
        assert g["simulated"] == pytest.approx(g["real"][0])
        assert g["verdict"] == "inside the real spread"
    zero = accepted["share accepting 0"]["real"]
    assert accepted["share accepting 0"]["spread"] == max(zero) - min(zero) > 0
    longest, shortest = max(mean["real"]), min(mean["real"])
    assert mean["spread"] == pytest.approx(longest / shortest - 1)


def test_a_simulated_histogram_outside_the_real_spread_grades_outside(
    monkeypatch, tmp_path, capsys
):
    real, simulated, _ = _one_server(monkeypatch, tmp_path, [0.3, 0.1, 0.05], 3)
    report = grade(real, simulated)

    assert {g["verdict"] for g in report["acceptance"].values()} == {"outside"}
    main([str(simulated), *map(str, real)])
    rows = [r for r in capsys.readouterr().out.splitlines() if "acceptance" in r]
    print("\n".join(rows))
    assert [r.split(" | ")[1] for r in rows] == [
        MEAN_LENGTH,
        *(f"share accepting {a}" for a in range(4)),
    ]
    assert {r.rsplit(" | ", 1)[1] for r in rows} == {"outside"}


@pytest.mark.parametrize("k", [3, 1], ids=["mtp3", "mtp1"])
def test_a_deployment_grades_acceptance_on_the_side_that_verified_drafts(
    monkeypatch, tmp_path, capsys, k
):
    real = [
        _deployment(monkeypatch, tmp_path / f"r{i}", (2 * i, 2 * i + 1), k)
        for i in range(3)
    ]
    simulated = _deployment(monkeypatch, tmp_path / "sim", (0, 1), k, **GRADED)
    report = grade(real, simulated)

    assert "acceptance prefill" not in report
    names = [MEAN_LENGTH, *(f"share accepting {a}" for a in range(k + 1))]
    accepted = report["acceptance decode"]
    assert list(accepted) == names
    assert {g["verdict"] for g in accepted.values()} == {"inside the real spread"}
    main([str(simulated), *map(str, real)])
    rows = [r for r in capsys.readouterr().out.splitlines() if "acceptance" in r]
    print("\n".join(rows))
    assert [r.split(" | ")[:2] for r in rows] == [
        ["acceptance decode", name] for name in names
    ]


def test_runs_that_differ_in_whether_they_verified_drafts_are_refused_naming_each(
    monkeypatch, tmp_path
):
    real, simulated, _ = _one_server(monkeypatch, tmp_path)
    (simulated / compass_run.SUMMARY_FILE).unlink()

    rows = {str(c): sum(_held(c).values()) for c in real} | {str(simulated): None}
    refusal = (
        "the runs' servers differ in whether they verified drafts; "
        f"verify rows each: {rows}"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        grade(real, simulated)


def test_runs_that_drafted_different_widths_are_refused_naming_each(
    monkeypatch, tmp_path
):
    real, simulated, _ = _one_server(monkeypatch, tmp_path, RATES[:2])

    widths = {str(c): [3] for c in real} | {str(simulated): [2]}
    refusal = f"the runs' servers drafted different widths: {widths}"
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        grade(real, simulated)


def test_a_real_side_lacking_a_dp_rank_s_histogram_is_refused_naming_it(
    monkeypatch, tmp_path
):
    real = [_deployment(monkeypatch, tmp_path / f"r{i}", (i, i), 1) for i in range(2)]
    simulated = _deployment(monkeypatch, tmp_path / "sim", (0, 0), 1, **GRADED)
    (real[1] / "decode" / "steps" / "dp1.acceptance.json").unlink()

    refusal = (
        f"{real[1] / 'decode'} holds no dp1.acceptance.json, and its record "
        "holds those DP ranks' steps"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        grade(real, simulated)
