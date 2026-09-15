"""Optional AIPerf dependency regression; run against its private patched source."""
import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("aiperf", reason="requires the pinned optional AIPerf dependency")

from aiperf.common.config import LoadGeneratorConfig
from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import (
    ConversationBranchInfo, ConversationMetadata, DatasetMetadata, TurnMetadata,
)
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import CreditPhaseConfig, TimingConfig
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.phase_orchestrator import PhaseOrchestrator
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy


def build():
    """A finished root with a future background child has nothing to prime."""
    metadata = DatasetMetadata(sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        has_timing_data=True, conversations=[
            ConversationMetadata(conversation_id="root", replay_scope_id="root",
                turns=[TurnMetadata(timestamp_ms=0, api_time_ms=1000, branch_ids=["background"])],
                branches=[ConversationBranchInfo(branch_id="background", mode=ConversationBranchMode.SPAWN,
                    is_background=True, child_conversation_ids=["child"], start_timestamp_ms=2000)]),
            ConversationMetadata(conversation_id="child", replay_scope_id="root", is_root=False,
                agent_depth=1, parent_conversation_id="root",
                turns=[TurnMetadata(timestamp_ms=2000, api_time_ms=1000)]),
        ])
    warmup = CreditPhaseConfig(phase=CreditPhase.WARMUP, timing_mode=TimingMode.AGENTIC_REPLAY,
                              concurrency=1, total_expected_requests=1)
    profiling = CreditPhaseConfig(phase=CreditPhase.PROFILING, timing_mode=TimingMode.AGENTIC_REPLAY,
                                 concurrency=1, expected_duration_sec=1)
    timing = TimingConfig(phase_configs=[warmup, profiling], concurrency=1, random_seed=42,
                          trajectory_start_min_ratio=.5, trajectory_start_max_ratio=.5)
    publisher = SimpleNamespace(**{name: AsyncMock() for name in (
        "publish_phase_start", "publish_phase_sending_complete", "publish_phase_complete",
        "publish_progress", "publish_profile_cancel")})
    router = SimpleNamespace(send_credit=AsyncMock(), cancel_all_credits=AsyncMock(),
        set_return_callback=lambda callback: None, set_first_token_callback=lambda callback: None)
    orch = PhaseOrchestrator(config=timing, phase_publisher=publisher, credit_router=router,
                             dataset_metadata=metadata)
    runner = PhaseRunner(config=warmup, conversation_source=orch.conversation_source,
        phase_publisher=publisher, credit_router=router, concurrency_manager=orch._concurrency_manager,
        cancellation_policy=orch._cancellation_policy, callback_handler=orch._callback_handler,
        session_tree_registry=orch._session_tree_registry)
    assert orch.conversation_source.warmup_credit_count == 0
    return orch, runner, router, publisher


async def settle():
    for _ in range(200):
        await asyncio.sleep(0)


def test_empty_warmup_completes_with_true_zero_counts_and_preserves_future_child():
    async def exercise():
        orch, runner, router, _ = build()
        snapshot = [asdict(t) for t in orch.conversation_source.trajectories]
        states = snapshot[0]["snapshot"]["states"]
        assert len(states) == 1 and states[0]["conversation_id"] == "child"
        assert states[0]["next_turn_index"] == 0 and states[0]["next_dispatch_offset_ms"] > 0
        task = asyncio.create_task(runner.run(is_final_phase=False))
        try:
            await settle()
            assert task.done(), "zero-warmup PhaseRunner did not complete"
            stats = await task
            assert stats.total_expected_requests == 0
            assert stats.requests_sent == stats.requests_completed == stats.requests_cancelled == 0
            assert stats.request_errors == 0 and not stats.timeout_triggered
            assert stats.is_requests_complete
            assert runner._progress.all_credits_sent_event.is_set()
            assert runner._progress.all_credits_returned_event.is_set()
            router.send_credit.assert_not_awaited()
            assert [asdict(t) for t in orch.conversation_source.trajectories] == snapshot
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(exercise())


def test_empty_warmup_enters_profiling_without_issuing_the_future_child_early():
    async def exercise():
        orch, _, router, publisher = build()
        task = asyncio.create_task(orch._execute_phases())
        try:
            await settle()
            phases = [call.args[0].phase for call in publisher.publish_phase_start.await_args_list]
            assert phases == [CreditPhase.WARMUP, CreditPhase.PROFILING]
            router.send_credit.assert_not_awaited()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(exercise())


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_empty_strategy_failure_or_cancellation_cannot_enter_profiling(monkeypatch, failure):
    async def fail(self):
        raise failure("empty strategy failed")
    monkeypatch.setattr(AgenticReplayStrategy, "execute_phase", fail)
    async def exercise():
        orch, _, router, publisher = build()
        with pytest.raises(failure, match="empty strategy failed"):
            await asyncio.wait_for(orch._execute_phases(), timeout=.25)
        phases = [call.args[0].phase for call in publisher.publish_phase_start.await_args_list]
        assert CreditPhase.PROFILING not in phases
        router.send_credit.assert_not_awaited()
    asyncio.run(exercise())


def test_unexpected_wait_failure_propagates_through_orchestrator(monkeypatch):
    async def fail(self, **kwargs):
        raise RuntimeError("unexpected wait failure")
    monkeypatch.setattr(PhaseRunner, "_wait_for_event_with_timeout", fail)
    async def exercise():
        orch, _, router, publisher = build()
        with pytest.raises(RuntimeError, match="unexpected wait failure"):
            await asyncio.wait_for(orch._execute_phases(), timeout=.25)
        phases = [call.args[0].phase for call in publisher.publish_phase_start.await_args_list]
        assert CreditPhase.PROFILING not in phases
        router.send_credit.assert_not_awaited()
    asyncio.run(exercise())


def test_zero_target_is_scoped_and_public_request_count_stays_positive():
    assert CreditPhaseConfig(phase=CreditPhase.WARMUP, timing_mode=TimingMode.AGENTIC_REPLAY,
                             total_expected_requests=0).total_expected_requests == 0
    for changes in ({"phase": CreditPhase.PROFILING}, {"timing_mode": TimingMode.REQUEST_RATE},
                    {"agentic_cache_warmup_duration_sec": 1.0}):
        options = dict(phase=CreditPhase.WARMUP, timing_mode=TimingMode.AGENTIC_REPLAY,
                       total_expected_requests=0)
        options.update(changes)
        with pytest.raises(ValueError):
            CreditPhaseConfig(**options)
    with pytest.raises(ValueError):
        LoadGeneratorConfig(request_count=0)
