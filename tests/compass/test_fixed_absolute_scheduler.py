"""Corrected finite releases through the native scheduler's admission path."""

from contextlib import contextmanager

import pytest
from conftest import MockConfig

from atom.compass.config import CompassConfig
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_runtime import StateRuntime, StateTransfer
from atom.sampling_params import SamplingParams
from atom.utils.clock import VirtualClock, get_clock, set_clock

from .test_fixed_absolute import bundle, ingress_profile, plan_file


@contextmanager
def configured(tmp_path, data=None, *, service_seconds=(.1, .2)):
    path, sha, plan = plan_file(tmp_path, data)
    profile, _ = ingress_profile(tmp_path, *service_seconds)
    previous = get_clock()
    clock = VirtualClock(epoch=1000.)
    set_clock(clock)
    try:
        scheduler = Scheduler(MockConfig(
            kv_cache_block_size=16, num_kvcache_blocks=8192, max_num_batched_tokens=16384,
            max_num_seqs=32, max_model_len=262144, enable_prefix_caching=True,
            pool_entries={"state": 32}, state_checkpoint_interval_tokens=8192,
            state_checkpoint_demand=True,
            compass_config=CompassConfig(enabled=True, epoch=1000.,
                request_readiness_profile=str(profile), fixed_absolute_plan=str(path),
                fixed_absolute_plan_sha256=sha)),
            state_runtime=StateRuntime(transfer=StateTransfer.fork(1)))
        sequences = [Sequence(row["prompt_token_ids"], 16, has_per_req_cache=True,
            sampling_params=SamplingParams(max_tokens=row["output_tokens"], ignore_eos=True))
            for row in plan.rows]
        for row, seq in zip(plan.rows, sequences):
            seq.arrive_time = clock.epoch + row["arrival_s"]
            seq.compass_workload_index = row["index"]
            seq.compass_workload_size = len(sequences)
        yield scheduler, clock, sequences, plan
    finally:
        set_clock(previous)


def finish_step(scheduler, clock, batch, selected):
    clock.advance(1.)
    scheduler.postprocess(list(selected.values()), ScheduledBatchOutput(
        req_ids=batch.req_ids, token_ids=[(99,) for _ in batch.req_ids],
        num_rejected=None, num_bonus=None, draft_token_ids=None), batch=batch)


def test_idle_clock_crosses_calendar_then_readiness_horizons_without_charging_future(tmp_path):
    with configured(tmp_path) as (scheduler, clock, sequences, plan):
        scheduler.extend(list(reversed(sequences)))
        calendar = scheduler._release_calendar
        first_seen, steps = [], 0
        while not calendar.state.done:
            result = scheduler.schedule()
            assert result is not None
            batch, selected = result
            for seq_id in batch.req_ids:
                if seq_id not in first_seen:
                    first_seen.append(seq_id)
            if batch.req_ids == [sequences[0].id]:
                assert calendar.released.keys() == {sequences[0].id}
                assert scheduler._request_readiness._causal_queue.count == 1
                assert not sequences[1].block_table and not sequences[1].state_slots
                assert sequences[1].checkpoint_demand_pos == 0
            finish_step(scheduler, clock, batch, selected)
            steps += 1
            assert steps < 20
        assert first_seen == [sequences[i].id for i in (0, 2, 3, 1)]
        releases = {row["index"]: row for row in calendar.evidence()["releases"]}
        assert [releases[i]["released_at"] for i in (0, 2, 3, 1)] == [1000., 1010., 1030., 1100.]
        assert [releases[i]["ready_at"] for i in (0, 2, 3, 1)] == pytest.approx([1000.3, 1010.3, 1030.3, 1100.3])
        assert calendar.evidence()["complete"] is True
        assert calendar.evidence()["root_completed_at"] == {"root0": sequences[1].finish_time}
        assert len(scheduler._request_readiness.input_manifest()["request_readiness"]["causal_releases"]) == 4


def test_simultaneous_roots_use_readiness_order_despite_reverse_registration(tmp_path):
    data = bundle(root_times=(0.,), child_times=(), clients=2)
    with configured(tmp_path, data, service_seconds=(0., 0.)) as (scheduler, clock, sequences, _):
        scheduler.extend(list(reversed(sequences)))
        batch, _ = scheduler.schedule()
        assert batch.req_ids == [seq.id for seq in sequences]
        assert [scheduler._request_readiness.record(seq).receipt_order for seq in sequences] == [0, 1]


def test_one_root_client_can_admit_three_outstanding_requests(tmp_path):
    data = bundle(root_times=(0., 1.), child_times=((1.,), (1.,)), clients=1)
    with configured(tmp_path, data, service_seconds=(0., 0.)) as (scheduler, clock, sequences, plan):
        scheduler.extend(list(reversed(sequences)))
        calendar = scheduler._release_calendar
        while sequences[0].id not in calendar.completed:
            batch, selected = scheduler.schedule()
            assert batch.req_ids == [sequences[0].id]
            finish_step(scheduler, clock, batch, selected)
        assert calendar.state.root_completion_times() == {}
        batch, selected = scheduler.schedule()
        assert batch.req_ids == [seq.id for seq in sequences[1:]]
        assert len(selected) == 3 > plan.evidence()["clients"]
        assert [calendar.released[seq.id]["receipt_order"] for seq in sequences[1:]] == [1, 2, 3]


def test_incomplete_registration_cannot_release_or_advance(tmp_path):
    with configured(tmp_path) as (scheduler, clock, sequences, _):
        scheduler.add(sequences[0])
        assert scheduler.schedule() is None
        assert scheduler._release_calendar.sequences is None
        assert scheduler._request_readiness.records is None and clock.elapsed == 0
        scheduler._arrival_barrier_since = float("-inf")
        with pytest.raises(ValueError, match="incomplete arrival barrier"):
            scheduler.schedule()
        assert clock.elapsed == 0


def test_profile_options_are_explicit_and_cannot_replace_the_opening(tmp_path):
    with pytest.raises(ValueError, match="digest"):
        CompassConfig(fixed_absolute_plan="plan.json")
    with pytest.raises(ValueError, match="source-backed"):
        CompassConfig(fixed_absolute_plan="plan.json", fixed_absolute_plan_sha256="a" * 64)
    with pytest.raises(ValueError, match="mutually exclusive"):
        CompassConfig(request_readiness_profile="service.json", opening_plan="opening.json",
            opening_plan_sha256="a" * 64, fixed_absolute_plan="plan.json", fixed_absolute_plan_sha256="b" * 64)
