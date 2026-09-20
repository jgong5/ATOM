"""P0.3 / T10 spike -- can AgenticReplayStrategy be subclassed rather than vendored?

Runs against agentx-harness 0.12.0 with ZERO edits to that repo.
Invocation -- both PYTHONPATH entries are required. src/ resolves aiperf; the
repo root resolves their tests.unit helpers, which claim 5 constructs through:

    R=<agentx-harness checkout>
    cd "$R" && PYTHONPATH="$R/src:$R" python <path to this file>

Each claim is asserted; exits non-zero on any failure.
"""

import asyncio
import dis
import sys
from unittest.mock import AsyncMock, MagicMock

import aiperf
from aiperf.common.enums import CreditPhase
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.timing.phase import runner as runner_mod
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy

results = []


def claim(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


print(f"aiperf at {aiperf.__file__}")
assert "agentx-harness" in aiperf.__file__, f"wrong aiperf tree: {aiperf.__file__}"


class ClockPacedScheduler:
    """Delegate that would route delays through the Compass clock client.

    Everything the base class does not pace is forwarded untouched.
    """

    def __init__(self, inner):
        self._inner = inner
        self.redirected = 0

    def schedule_later(self, delay_sec, coro, **kw):
        self.redirected += 1
        return self._inner.schedule_later(
            0.0, coro, **kw
        )  # a clock grant would go here

    def __getattr__(self, name):
        return getattr(self._inner, name)


class CompassAgenticReplay(AgenticReplayStrategy):
    """The whole adapter override: wrap the injected scheduler, nothing else."""

    def __init__(self, *, scheduler, **kw):
        super().__init__(scheduler=ClockPacedScheduler(scheduler), **kw)


# --- Claim 1: the strategy is built by the plugin factory, not hard-constructed.
src = dis.Bytecode(runner_mod.PhaseRunner._build_strategy).dis()
claim(
    "1. strategy is factory-built via plugins.get_class",
    "get_class" in src and "AgenticReplayStrategy" not in src,
)

# --- Claim 2: an out-of-tree subclass displaces the built-in in the registry.
plugins.register(
    PluginType.TIMING_STRATEGY, "agentic_replay", CompassAgenticReplay, priority=100
)
won = plugins.get_class(PluginType.TIMING_STRATEGY, "agentic_replay")
claim(
    "2. out-of-tree subclass wins the registry at priority 100",
    won is CompassAgenticReplay,
    won.__name__,
)

# --- Claim 3: the wrapper covers every LoopScheduler member the strategy touches.
strategy_needs = {
    "schedule_later",
    "set_drain_observer",
    "running_count",
    "cap_pending_delay",
}
missing = {n for n in strategy_needs if not hasattr(LoopScheduler, n)}
claim(
    "3. the four members the strategy touches all exist on LoopScheduler",
    not missing,
    ",".join(sorted(strategy_needs)),
)

# --- Claim 4: LoopScheduler is a module global in phase.runner, hence substitutable
# for the consumers a strategy subclass cannot reach (see claim 6).
init_src = dis.Bytecode(runner_mod.PhaseRunner.__init__).dis()
claim(
    "4. LoopScheduler is a module global in phase.runner (substitutable)",
    "LoopScheduler" in init_src and "LOAD_GLOBAL" in init_src,
)

# --- Claim 5: the subclass actually constructs through the real base __init__,
# and self.scheduler is our wrapper.
from tests.unit.timing.strategies.test_agentic_replay import (
    _build_real_trajectory_source,
)


async def _claim5():
    source = _build_real_trajectory_source(2, 3, [], dataset=None)
    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    cfg.concurrency = 1
    cfg.agentic_cache_warmup_duration_sec = None
    cfg.warmup_requests_per_lane = None
    issuer = AsyncMock()
    issuer.replay_gate = MagicMock()
    issuer.replay_gate.completed_prefixes.return_value = ()
    issuer.replay_gate.pending_turns.return_value = ()
    issuer.replay_gate.pending_turns_by_root.return_value = {}
    real_sched = LoopScheduler()

    strategy = won(
        config=cfg,
        conversation_source=source,
        scheduler=real_sched,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )
    return strategy, real_sched


strategy, real_sched = asyncio.run(_claim5())
claim(
    "5. subclass constructs through the real base __init__; self.scheduler is the wrapper",
    isinstance(strategy, AgenticReplayStrategy)
    and isinstance(strategy.scheduler, ClockPacedScheduler)
    and strategy.scheduler._inner is real_sched,
    type(strategy.scheduler).__name__,
)

# --- Claim 6: THE FINDING. The strategy is not the only production pacing consumer.
# PhaseRunner hands the SAME LoopScheduler to BranchOrchestrator (always built for
# AGENTIC_REPLAY) and ReplayBarrierCoordinator, whose schedule_later calls a strategy
# subclass never sees. This is a *negative* claim: it PASSES when the other sites exist.
import subprocess

other = (
    subprocess.run(
        [
            "grep",
            "-rn",
            r"_scheduler\.schedule_later(\|_scheduler\.cap_pending_delay",
            "src/aiperf/timing/branch_orchestrator.py",
            "src/aiperf/timing/replay_dependencies.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    .stdout.strip()
    .splitlines()
)
claim(
    "6. pacing sites OUTSIDE the strategy exist on the agentic path",
    len(other) >= 3,
    f"{len(other)} sites: "
    + "; ".join(l.split(":")[0] + ":" + l.split(":")[1] for l in other),
)

print()
n_ok = sum(1 for _, ok, _ in results if ok)
print(f"{n_ok}/{len(results)} claims passed")
sys.exit(0 if n_ok == len(results) else 1)
