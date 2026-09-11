# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Two gates, and one deployment that fails each of them.

The 27B at TP=1 was reported feasible, and it is -- at `--max-num-seqs 32`.
Raising the concurrency is the axis nothing in the campaign varied, and it is
where the hybrid's shape shows: each in-flight request reserves 74.8 MiB of
recurrent state before a single KV block is paged, so concurrency buys capacity
out of the same budget the context length is drawn from.

That produces two different failures, and only the first one looks like a
failure:

* at `--max-num-seqs 1551` the state floor exceeds the whole KV budget and the
  engine refuses to start, with `InsufficientPoolBudget`;
* at `--max-num-seqs 1400` the engine starts, sizes 11 190 blocks, reports
  itself healthy -- and can never admit the longest request in cc-traces, which
  needs 15 940.

Neither is a prediction at a configuration anything was calibrated on: the
memory records are all at `max_num_seqs 32`, and no constant here was fitted to
anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atom.compass.core.feasibility import (
    Request,
    admission_refusal,
    assess,
    blocks_for,
    longest_request,
    trace_requests,
)
from atom.compass.core.kv_geometry import gdn_state_bytes
from atom.compass.core.memory import MemoryReadings
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

from tests.conftest import MockConfig

RECORDS = Path(__file__).parent / "memory_records"

#: The longest request in `cc_pilot.jsonl` (62 requests, sha256 bf4049f8...),
#: the cc-traces slice the campaign runs against. Its two lengths are a
#: property of the workload, so they are stated here rather than shipped as
#: another copy of the trace -- and `test_the_trace_still_says_this` checks
#: them against the trace whenever it is on the box.
CC_LONGEST = Request(input_tokens=249344, output_tokens=5690)

#: The deployment everything was recorded at, other than the concurrency.
DEPLOYED = dict(
    utilization=0.9,
    max_model_len=262144,
    max_num_batched_tokens=16384,
    tensor_parallel=1,
    block_size=16,
)


def _config() -> dict:
    with open(RECORDS / "qwen3_5_27b.config.json", encoding="utf-8") as fh:
        return json.load(fh)


def _tp1_readings() -> MemoryReadings:
    with open(RECORDS / "27b.tp1.memory.json", encoding="utf-8") as fh:
        got = json.load(fh)["readings"]
    return MemoryReadings(
        total=got["total"],
        free=got["free"],
        peak_torch=got["peak_torch"],
        non_torch=got["non_torch"],
        cudagraph_overhead=got["cudagraph_overhead"],
    )


# ── the two gates ─────────────────────────────────────────────────────────


def test_the_recorded_deployment_is_feasible_and_for_the_right_reason():
    """`max_num_seqs 32` serves the longest cc-traces request with room over.

    The baseline this is all measured against: 112 740 blocks against 15 940
    needed, so the longest request fits seven times over and concurrency is
    nowhere near binding.
    """
    verdict = assess(
        _config(), _tp1_readings(), max_num_seqs=32, request=CC_LONGEST, **DEPLOYED
    )
    assert verdict
    assert verdict.gate is None
    assert verdict.blocks == 112740
    assert verdict.blocks_needed == 15940
    assert verdict.state_entries == 32


def test_the_state_floor_alone_can_refuse_the_deployment():
    """At 1551 concurrent requests the engine cannot start.

    The refusal is `plan_pools`', not this module's: `assess` catches ATOM's
    own `InsufficientPoolBudget` and reports its byte counts. 1551 x 74.8 MiB
    is 113.31 GB against a 113.30 GB budget -- one request over the line, which
    is the boundary being pinned.
    """
    verdict = assess(
        _config(), _tp1_readings(), max_num_seqs=1551, request=CC_LONGEST, **DEPLOYED
    )
    assert not verdict
    assert verdict.gate == "pool"
    assert "113.31GB of 113.30GB" in verdict.reason
    assert verdict.state_entries == 1551

    # And one request below it, the engine starts.
    assert assess(_config(), _tp1_readings(), max_num_seqs=1550, **DEPLOYED).feasible


def test_a_healthy_deployment_that_cannot_serve_the_workload():
    """1400 concurrent requests: starts, sizes a pool, admits nothing long.

    This is the case a start-up check calls feasible. `/health` returns OK,
    `rocm-smi` shows the VRAM resident, the pool is 11 190 blocks -- and the
    longest request in the trace needs 15 940, so it is refused on arrival for
    as long as the deployment lives.
    """
    verdict = assess(
        _config(), _tp1_readings(), max_num_seqs=1400, request=CC_LONGEST, **DEPLOYED
    )
    assert not verdict
    assert verdict.gate == "admission"
    assert verdict.blocks == 11190
    assert verdict.blocks_needed == 15940
    assert "15940 KV blocks" in verdict.reason


def test_without_the_workload_the_same_deployment_looks_fine():
    """Omitting the longest request is what makes 1400 read as feasible.

    The weaker question, asked explicitly so the difference between the two is
    a line in a test rather than an assumption in a reader's head.
    """
    verdict = assess(_config(), _tp1_readings(), max_num_seqs=1400, **DEPLOYED)
    assert verdict.feasible
    assert verdict.blocks_needed == 0


# ── the mirror against ATOM's own scheduler ───────────────────────────────


class _SchedulerScalars:
    """The four attributes `_unschedulable_reason` reads off its scheduler.

    `Scheduler.__init__` is not called: it opens a KV-event publisher and a
    connector, which on a box with no engine behind them does not return --
    `tests/test_scheduler.py` hangs the same way, so this is the environment
    and not the rule. The rule itself is ATOM's, called unbound below, over a
    real `BlockManager` built the way the engine builds it.
    """

    def __init__(self, blocks: int):
        self.max_model_len = DEPLOYED["max_model_len"]
        self.max_num_batched_tokens = DEPLOYED["max_num_batched_tokens"]
        self.enable_chunked_prefill = True
        self.block_manager = BlockManager(
            MockConfig(
                num_kvcache_blocks=blocks,
                kv_cache_block_size=16,
                max_model_len=DEPLOYED["max_model_len"],
                max_num_batched_tokens=DEPLOYED["max_num_batched_tokens"],
                enable_chunked_prefill=True,
                max_num_seqs=32,
            )
        )


def _engine_reason(blocks: int, tokens: int):
    """ATOM's own verdict on a request of `tokens` against a pool of `blocks`."""
    return Scheduler._unschedulable_reason(_SchedulerScalars(blocks), _sequence(tokens))


def _sequence(tokens: int) -> Sequence:
    return Sequence([1] * tokens, 16, sampling_params=SamplingParams())


@pytest.mark.parametrize(
    "blocks,tokens,refused",
    [
        (11190, 255034, True),  # the 1400-seq pool, the longest cc-traces request
        (112740, 255034, False),  # the 32-seq pool, the same request
        (11190, 24384, False),  # the same pool, the longest cc_small prompt
    ],
)
def test_atoms_scheduler_agrees_with_the_admission_mirror(blocks, tokens, refused):
    """The mirrored rule and `Scheduler._unschedulable_reason` decide alike.

    The mirror exists because reaching the original needs a BlockManager, a
    Config and a live Sequence; this test builds all three, so the copy is
    checked against the thing it copies rather than trusted.

    The scheduler sees a *prompt*, so the sequence here is the request at its
    full length -- which is the length that has to fit, and the length the
    mirror checks. Handing the scheduler the prompt alone is exactly the
    mistake this module was written to stop.
    """
    engine_reason = _engine_reason(blocks, tokens)
    mirror = admission_refusal(
        Request(tokens, 0),
        blocks=blocks,
        block_size=16,
        max_model_len=DEPLOYED["max_model_len"],
        max_num_batched_tokens=DEPLOYED["max_num_batched_tokens"],
    )

    assert (engine_reason is not None) == refused
    assert (mirror is not None) == refused
    if refused:
        assert "KV blocks" in engine_reason and "KV blocks" in mirror


def test_the_over_long_prompt_rule_agrees_too():
    """A request past `max_model_len` is refused by both, and named the same.

    The other end of the same gate: a pool big enough is not enough if the
    sequence cannot be addressed, and the scheduler checks that one first
    because it is the usual actionable cause.
    """
    too_long = DEPLOYED["max_model_len"] + 1
    assert "max_model_len" in _engine_reason(112740, too_long)
    assert "max_model_len" in admission_refusal(
        Request(too_long, 0),
        blocks=112740,
        block_size=16,
        max_model_len=DEPLOYED["max_model_len"],
        max_num_batched_tokens=DEPLOYED["max_num_batched_tokens"],
    )


# ── the workload's own numbers ────────────────────────────────────────────


def test_longest_is_by_total_length_not_by_prompt():
    """A shorter prompt that answers for longer is the harder request.

    Not hypothetical at the scale that matters here: picking by prompt length
    is a rule that happens to agree on this trace and has no reason to on the
    next one.
    """
    requests = [Request(200000, 60000), Request(240000, 20)]
    assert longest_request(requests) == requests[0]
    assert max(requests, key=lambda r: r.input_tokens) == requests[1]


def test_blocks_for_rounds_up_and_refuses_to_guess_under_dcp():
    assert blocks_for(255034, 16) == 15940
    assert blocks_for(1, 16) == 1
    assert blocks_for(16, 16) == 1
    assert blocks_for(17, 16) == 2
    with pytest.raises(NotImplementedError):
        blocks_for(1024, 16, dcp_world_size=2)


def test_the_trace_still_says_this():
    """`CC_LONGEST` against the trace itself, when the trace is on the box.

    Skipped rather than shipped: the trace is the workload and belongs with
    the campaign's artifacts, not in the test tree. The two numbers are pinned
    above so the rest of this module runs anywhere.
    """
    trace = (
        Path(__file__).resolve().parents[2]
        / "agent_scratch/mem/evidence/traces/cc_pilot.jsonl"
    )
    if not trace.exists():
        pytest.skip("cc_pilot.jsonl is not on this box")
    worst = longest_request(trace_requests(str(trace)))
    assert worst == CC_LONGEST


def test_state_floor_is_what_makes_concurrency_expensive_here():
    """74.8 MiB a request, which is why 1400 of them cost 97.7 GB.

    Stated because it is the whole mechanism: on a dense model `max_num_seqs`
    costs nothing at sizing time, so a reader carrying that intuition has no
    reason to expect any of the above.
    """
    per_request = gdn_state_bytes(_config(), tensor_parallel=1)
    assert per_request == 78446592
    assert 1400 * per_request / 2**30 == pytest.approx(102.3, abs=0.1)


# ── the frozen utilization-axis acceptance configuration ──────────────────


def test_the_frozen_predictions_are_still_what_the_model_says():
    """`frozen_util_predictions.json`, recomputed.

    The acceptance configuration moves `gpu_memory_utilization` and nothing
    else: `max_num_seqs` stays at the cc-traces 32 and the capture ladder stays
    at (1, 2, 4, 8, 16, 32), so `peak_torch`, `non_torch`, the activation peak
    and the capture pool all keep the shape they were measured at. The byte
    budget is the only thing that moves, which is the input the pool gate
    reads.

    That makes the configuration *sensitive* in a way the deployed one is not.
    At 0.90 the KV budget is 113 GB and a 100 MB error in the non-KV terms
    moves the block count by 0.08%; at 0.33 the budget is 4.15 GB and the same
    100 MB moves it by 2.4%. The four predictions below are therefore a much
    sharper test of the non-KV terms than the deployed configuration is, and
    they are frozen before any run at those settings so the comparison cannot
    be revised into agreement afterwards.

    The refusal at 0.32 is ATOM's, and the threshold it reports (>= 0.33) is
    the engine's own arithmetic in `get_num_blocks`, not this module's.
    """
    with open(RECORDS / "frozen_util_predictions.json", encoding="utf-8") as fh:
        frozen = json.load(fh)

    assert frozen["max_num_seqs"] == 32
    assert frozen["capture_ladder"] == [1, 2, 4, 8, 16, 32]
    assert frozen["enable_prefix_caching"] is False
    assert frozen["geometry_from_config"]["state_bytes_per_request"] == 78446592

    for want in frozen["predictions"]:
        verdict = assess(
            _config(),
            _tp1_readings(),
            max_num_seqs=32,
            request=CC_LONGEST,
            **{**DEPLOYED, "utilization": want["gpu_memory_utilization"]},
        )
        assert verdict.blocks == want["predicted_num_kvcache_blocks"]
        assert verdict.gate == want["predicted_gate"]
        assert verdict.reason == want["predicted_reason"]
        assert (verdict.gate != "pool") is want["expect_engine_starts"]
