# SPDX-License-Identifier: MIT
"""``client.sh --agentx``: InferenceX's agentx client, as aiperf builds its
agentic replay under compass-harness, its per-trace idle cap and its warmup's
drain on the Compass clock.

``client.sh`` runs against the stand-ins ``test_client_sh.py`` puts on ``PATH``;
the aiperf arguments it passes go through aiperf's own CLI parser, scenario
lock and timing config to the phase runners the orchestrator builds, which wait
on a real ``TrafficLP`` whose runtime grants each clock call its target at
once, with the event loop's timers made to raise. Skips by name as
``test_harness_pacing.py`` does.
"""

import asyncio
import json
import os
from importlib import metadata
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _missing() -> str | None:
    try:
        metadata.version("compass-harness")
        import compass_harness  # noqa: F401  checks aiperf against its pinned source
    except metadata.PackageNotFoundError as e:
        return f"{e.name} is not installed"
    except RuntimeError as e:
        return str(e)
    return None


if _why := _missing():
    pytest.skip(
        f"needs agentx-harness 56a0cf70 and compass-harness: {_why}",
        allow_module_level=True,
    )

from aiperf.plugin import plugins  # noqa: F401  plugin discovery, as in aiperf

# isort: split
from aiperf.cli import app
from aiperf.cli_runner import _make_benchmark_run
from aiperf.common.enums import CreditPhase
from aiperf.common.scenario import apply_scenario
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader import build_benchmark_plan
from aiperf.credit.structs import TurnToSend
from aiperf.orchestrator.orchestrator import resolve_run_seed
from aiperf.plugin.enums import TimingMode
from aiperf.timing import phase_orchestrator
from aiperf.timing.config import TimingConfig
from aiperf.timing.trajectory_source import TrajectorySource
from compass_harness.scheduler import ClockPacedLoopScheduler
from compass_harness.strategy import CompassAgenticReplay
from compass_harness.timesource import ClockTimer
from compass_harness.traffic_lp import TrafficLP
from test_client_sh import SIM, _client, _setup
from test_harness_timesource import GrantAtOnce, _wall_timer

CAP = 300.0
GRACE = 1800.0


def _agentx_client(tmp_path, *args):
    out, env = _setup(tmp_path, SIM)
    return out, _client(out, "--agentx", *args, env=env)


@pytest.fixture
def agentx(tmp_path):
    """The aiperf run and timing config that ``client.sh --agentx
    --concurrency 4`` hands aiperf, resolved as aiperf resolves them."""
    out, done = _agentx_client(tmp_path, "--concurrency", "4")
    assert done.returncode == 0, done.stdout + done.stderr
    argv, _, _ = json.loads((tmp_path / "seen.json").read_text())
    assert argv[:3] == ["-m", "aiperf", "profile"]
    _, bound, _ = app.parse_args(argv[2:], exit_on_error=False)
    plan = build_benchmark_plan(resolve_config(bound.arguments["cli_config"], None))
    run = _make_benchmark_run(
        plan.configs[0], random_seed=resolve_run_seed(plan, plan.variations[0])
    )
    apply_scenario(run)
    return SimpleNamespace(
        argv=argv, out=out, run=run, timing=TimingConfig.from_run(run)
    )


async def _traffic(tmp_path, monkeypatch) -> TrafficLP:
    traffic = TrafficLP(GrantAtOnce(), str(tmp_path / "lp"))
    monkeypatch.setattr(ClockPacedLoopScheduler, "clock", traffic)
    loop = asyncio.get_running_loop()
    loop.call_later = loop.call_at = _wall_timer
    return traffic


def _runner(agentx, phase: CreditPhase):
    """The phase runner the orchestrator builds for `phase` of the agentx run."""
    (config,) = [c for c in agentx.timing.phase_configs if c.phase == phase]
    source = MagicMock(spec=TrajectorySource)
    source.trajectories = []
    return phase_orchestrator.PhaseRunner(
        config=config,
        conversation_source=source,
        phase_publisher=AsyncMock(),
        credit_router=MagicMock(),
        concurrency_manager=MagicMock(),
        cancellation_policy=MagicMock(),
        callback_handler=MagicMock(),
        run=agentx.run,
    )


def test_agentx_replays_every_trace_with_the_scenario_defaults(agentx):
    line = " ".join(agentx.argv)
    for flag in (
        "--random-seed 42",
        "--benchmark-duration 3600",
        "--scenario inferencex-agentx-mvp",
        "--trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75",
        "--warmup-requests-per-lane 10 --agentic-warmup-grace-period 1800",
        "--trace-idle-gap-cap-seconds 300 --system-idle-gap-cap-seconds 10",
        "--apply-chat-template",
        "--use-server-token-count",
    ):
        assert f" {flag} " in f" {line} "
    assert "20260707" not in line
    files = ["00000.json", "00001.json", "00002.json", "00003.json"]
    assert sorted(os.listdir(agentx.out / "client/traces")) == files
    workload = json.loads((agentx.out / "artifacts/workload.json").read_text())
    assert (
        workload["aiperf_args"] == agentx.argv[agentx.argv.index("--endpoint-type") :]
    )


@pytest.mark.parametrize(
    "args, refusal",
    [
        (["--traces", "1-2", "--concurrency", "3"], "--agentx takes every trace"),
        (["--sweep", "--stop"], "--agentx takes every trace"),
        ([], "--concurrency is required"),
    ],
)
def test_agentx_refuses_a_trace_range_a_sweep_and_no_concurrency(
    tmp_path, args, refusal
):
    out, done = _agentx_client(tmp_path, *args)
    assert done.returncode == 2, done.stdout + done.stderr
    assert done.stderr.startswith(f"client: {refusal}")
    assert not (out / "client").exists()


def test_aiperf_resolves_client_sh_agentx_arguments_to_the_scenario(agentx):
    timing, run = agentx.timing, agentx.run
    warmup, profiling = timing.phase_configs
    assert (warmup.phase, warmup.timing_mode) == (
        CreditPhase.WARMUP,
        TimingMode.AGENTIC_REPLAY,
    )
    assert warmup.warmup_requests_per_lane == 10
    assert warmup.agentic_cache_warmup_duration_sec is None
    assert warmup.grace_period_sec == GRACE
    assert (profiling.phase, profiling.timing_mode) == (
        CreditPhase.PROFILING,
        TimingMode.AGENTIC_REPLAY,
    )
    assert profiling.expected_duration_sec == 3600
    assert timing.random_seed == 42
    assert (timing.trajectory_start_min_ratio, timing.trajectory_start_max_ratio) == (
        0.25,
        0.75,
    )
    assert run.cfg.get_default_dataset().trace_idle_gap_cap_seconds == CAP
    assert run.cfg.get_profiling_phases()[0].system_idle_gap_cap_seconds == 10.0
    assert run.cfg.tokenizer.apply_chat_template is True


def test_compass_builds_both_agentx_phases_with_the_idle_cap_on_the_clock(
    agentx, tmp_path, monkeypatch
):
    """Neither phase is refused: the per-trace idle cap and the warmup bounded
    by a request count per lane both run under compass-harness."""

    async def main():
        traffic = await _traffic(tmp_path, monkeypatch)
        built = {}
        for phase in (CreditPhase.WARMUP, CreditPhase.PROFILING):
            runner = _runner(agentx, phase)
            built[phase] = runner, runner._build_strategy()
        traffic.done.cancel()
        return built

    built = asyncio.run(main())
    for _, strategy in built.values():
        assert type(strategy) is CompassAgenticReplay
    assert built[CreditPhase.WARMUP][1]._cache_warmup_requests_per_lane == 10
    caps = {
        p: r._replay_barrier._root_idle_gap_cap_seconds for p, (r, _) in built.items()
    }
    assert caps == {CreditPhase.WARMUP: None, CreditPhase.PROFILING: CAP}


def _credit(root: str) -> SimpleNamespace:
    return SimpleNamespace(
        effective_root_correlation_id=root, conversation_id=root, turn_index=0
    )


@pytest.mark.parametrize(
    "gap, starts_at", [(1000.0, CAP), (CAP + 1.0, CAP), (100.0, 100.0)]
)
def test_a_trace_idle_past_the_cap_resumes_at_the_cap_on_the_clock(
    agentx, tmp_path, monkeypatch, gap, starts_at
):
    """One trace's last request returns and its next turn is due `gap` later.

    The tree's idle watchdog fires at the cap and pulls the turn in to it; a
    turn due before the cap starts at its own time, and its issue withdraws
    the watchdog's clock wait.
    """

    async def main():
        traffic = await _traffic(tmp_path, monkeypatch)
        runner = _runner(agentx, CreditPhase.PROFILING)
        barrier, scheduler = runner._replay_barrier, runner._scheduler
        barrier.activate()
        started = asyncio.get_running_loop().create_future()

        async def next_turn():
            barrier.observe_issued(_credit("r"))
            started.set_result(traffic.now())

        barrier.observe_issued(_credit("r"))
        barrier.complete(_credit("r"))
        scheduler.schedule_later(gap, next_turn(), group_id="r")
        at = await started
        for _ in range(100):  # let the owner settle with nothing left to ask for
            await asyncio.sleep(0)
        traffic.done.cancel()
        return at, barrier, traffic

    at, barrier, traffic = asyncio.run(main())
    assert at == starts_at
    assert barrier._root_idle_seconds_skipped == gap - starts_at
    assert barrier._root_idle_jumps == (gap > CAP)
    assert max(traffic.rt.calls) == starts_at
    assert traffic._waiters == []


def test_the_cap_is_per_trace_a_busy_trace_keeps_its_own_gap(
    agentx, tmp_path, monkeypatch
):
    """Trace `a` goes idle with its next turn due at 1000 s while trace `b`
    still has a request in flight and its own turn due at 600 s: the cap pulls
    in `a`'s turn alone."""

    async def main():
        traffic = await _traffic(tmp_path, monkeypatch)
        runner = _runner(agentx, CreditPhase.PROFILING)
        barrier, scheduler = runner._replay_barrier, runner._scheduler
        barrier.activate()
        loop = asyncio.get_running_loop()
        started = {root: loop.create_future() for root in ("a", "b")}

        async def turn(root):
            started[root].set_result(traffic.now())

        for root in started:
            barrier.observe_issued(_credit(root))
        barrier.complete(_credit("a"))
        scheduler.schedule_later(1000.0, turn("a"), group_id="a")
        scheduler.schedule_later(600.0, turn("b"), group_id="b")
        at = dict(zip(started, await asyncio.gather(*started.values())))
        traffic.done.cancel()
        return at

    assert asyncio.run(main()) == {"a": CAP, "b": 600.0}


def test_the_warmup_drain_ends_at_its_last_return_on_the_clock(
    agentx, tmp_path, monkeypatch
):
    """The warmup's per-lane budget is sent and one request is still out; it
    returns 5 s later. The drain ends then, not at its 1800 s grace."""

    async def main():
        traffic = await _traffic(tmp_path, monkeypatch)
        runner = _runner(agentx, CreditPhase.WARMUP)
        progress = runner._progress
        progress.increment_sent(TurnToSend("c", "x", 3, 9))
        ClockTimer(5.0, lambda: progress.increment_returned(False, False))
        await runner._wait_for_accelerated_warmup_handoff()
        at = traffic.now()
        for _ in range(100):  # let the owner settle with nothing left to ask for
            await asyncio.sleep(0)
        traffic.done.cancel()
        return at, progress, traffic

    at, progress, traffic = asyncio.run(main())
    assert 5.0 <= at <= 5.1 + 1e-9
    assert max(traffic.rt.calls) < GRACE
    assert progress.all_credits_returned_event.is_set()
    assert traffic._waiters == []
