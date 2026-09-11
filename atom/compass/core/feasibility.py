"""Whether a configuration can serve a workload, and why not when it cannot.

Feasibility has been treated as "the engine started", which is the weaker half
of the question. A deployment that starts, sizes a non-zero pool and then can
never admit the workload's longest request is not feasible; it is a deployment
that answers every short request and drops the long one, and a ranking built
over it is ranking a service nobody asked for.

So there are two gates here, and a configuration has to pass both:

* **Pool** -- the STATE floor has to leave something to page with. This is
  `plan_pools`, called rather than reimplemented, so a rejection carries ATOM's
  own `InsufficientPoolBudget` and the same byte counts the engine's start-up
  error would print.
* **Admission** -- the workload's longest request has to be schedulable. The
  three static rules are `Scheduler._unschedulable_reason`'s, mirrored here
  because reaching the original needs a `BlockManager`, a `Config` and a live
  `Sequence`. `tests/compass/test_feasibility.py` wires the derived block count
  into a real `Scheduler` and checks the two agree, which is what keeps the
  mirror honest.

**Prefill and decode are different lengths.** A prompt is `input` tokens when
the batched-token budget sees it and `input + output` tokens by the time it
finishes, and the rules divide on that: `max_num_batched_tokens` bounds the
prefill, while `max_model_len` and the block count have to hold the sequence at
its longest. Checking the prompt length against all three -- which is what the
scheduler does, because at submit time that is all it has -- passes
configurations that die part-way through the decode of the longest request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from atom.compass.core.kv_geometry import (
    InsufficientPoolBudget, blocks_from_readings)

__all__ = ["Request", "Verdict", "longest_request", "trace_requests",
           "blocks_for", "admission_refusal", "assess"]


@dataclass(frozen=True)
class Request:
    """One request's two lengths. Both matter, and for different rules."""

    input_tokens: int
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Verdict:
    """Feasible or not, which gate decided, and the numbers behind it."""

    feasible: bool
    #: ``"pool"``, ``"admission"`` or ``None`` when feasible.
    gate: Optional[str] = None
    reason: Optional[str] = None
    blocks: int = 0
    state_entries: int = 0
    blocks_needed: int = 0

    def __bool__(self) -> bool:
        return self.feasible


def trace_requests(path: str) -> list:
    """Every request in a cc-traces JSONL, as lengths.

    The trace is the workload, so the longest request in it is a property of
    the workload and not of any deployment -- which is what makes it a
    legitimate feasibility input. Nothing measured on the target is read.
    """
    requests = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            requests.append(Request(int(row.get("input_tokens") or 0),
                                    int(row.get("output_tokens") or 0)))
    return requests


def longest_request(requests) -> Optional[Request]:
    """The request that has to fit, which is the longest *in total*.

    Not the longest prompt. A 240k-token prompt with a 20-token answer is
    easier to serve than a 200k prompt answering for 60k, and picking by prompt
    length silently checks the wrong one.
    """
    return max(requests, key=lambda r: r.total_tokens, default=None)


def blocks_for(tokens: int, block_size: int, dcp_world_size: int = 1) -> int:
    """Pool blocks one sequence of `tokens` occupies on one rank.

    `BlockManager.num_pool_blocks`. Under DCP a rank holds only its shard, and
    this is the only count that may be compared against the pool's size --
    which is why the DCP case is named here rather than left to a reader who
    might use `ceil(tokens / block_size)` and admit prompts that do not fit.
    """
    if dcp_world_size > 1:
        raise NotImplementedError(
            "dcp>1 shards a sequence across ranks by an interleave this does "
            "not model; use BlockManager.num_pool_blocks")
    return (int(tokens) + int(block_size) - 1) // int(block_size)


def admission_refusal(request: Request, *, blocks: int, block_size: int,
                      max_model_len: int, max_num_batched_tokens: int,
                      enable_chunked_prefill: bool = True) -> Optional[str]:
    """Why this request can never be scheduled here, or None.

    The three static rules of `Scheduler._unschedulable_reason`, in its order,
    because the first one that fires is the one the engine would report.
    """
    total = request.total_tokens
    if max_model_len and total > max_model_len:
        return ("tokens=%d > max_model_len=%d at its longest (input %d + "
                "output %d)" % (total, max_model_len, request.input_tokens,
                                request.output_tokens))
    if (not enable_chunked_prefill and max_num_batched_tokens
            and request.input_tokens > max_num_batched_tokens):
        return ("input tokens=%d > max_num_batched_tokens=%d with chunked "
                "prefill off" % (request.input_tokens, max_num_batched_tokens))
    needed = blocks_for(total, block_size)
    if needed > blocks:
        return ("needs %d KV blocks for %d tokens > total pool blocks=%d"
                % (needed, total, blocks))
    return None


def assess(config: Mapping[str, Any], readings, *, utilization: float,
           max_num_seqs: int, max_model_len: int,
           max_num_batched_tokens: int = 0, tensor_parallel: int = 1,
           block_size: int = 16, kv_dtype_bytes: int = 2,
           state_dtype_bytes: int = 2, num_spec: int = 0,
           enable_chunked_prefill: bool = True,
           request: Optional[Request] = None) -> Verdict:
    """Both gates, in the order the engine would hit them.

    `request` is the workload's longest; omit it and only the pool gate is
    applied, which is the weaker question this module exists to stop being
    mistaken for the whole one -- so a verdict reached without one says so by
    leaving `blocks_needed` at zero.
    """
    try:
        plan = blocks_from_readings(
            config, readings, utilization=utilization,
            max_num_seqs=max_num_seqs, tensor_parallel=tensor_parallel,
            block_size=block_size, kv_dtype_bytes=kv_dtype_bytes,
            state_dtype_bytes=state_dtype_bytes, num_spec=num_spec)
    except InsufficientPoolBudget as exc:
        return Verdict(
            False, "pool",
            "state pool needs %.2fGB of %.2fGB available for %d entries"
            % (exc.reserved_bytes / 2**30, exc.available_bytes / 2**30,
               exc.entries),
            state_entries=exc.entries)

    blocks = plan.paged_entries
    state_entries = sum(count for name, count in plan.entries.items()
                        if name != plan.paged_class)
    if blocks <= 0:
        return Verdict(False, "pool", "the pool sized to zero blocks",
                       blocks=0, state_entries=state_entries)
    if request is None:
        return Verdict(True, None, None, blocks, state_entries)

    refusal = admission_refusal(
        request, blocks=blocks, block_size=block_size,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill)
    needed = blocks_for(request.total_tokens, block_size)
    if refusal:
        return Verdict(False, "admission", refusal, blocks, state_entries,
                       needed)
    return Verdict(True, None, None, blocks, state_entries, needed)
