"""Configuration for ATOMCompass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["CompassConfig"]


@dataclass
class CompassConfig:
    """Settings for a simulated run.

    Attributes:
        enabled: Master switch. When false the runner behaves as stock ATOM.
        mode: ``"predict"`` replaces the forward pass with a cost estimate.
            ``"trace"`` performs the real forward and records what it did,
            which is how a reference op graph is obtained. ``"measure"`` also
            performs the real forward but records how long it took, which is
            what a calibrated oracle is fitted to. Neither trace nor measure is
            free: they exist to produce artifacts, not to serve.
        graph_out: Where a traced graph is written. The runner is a separate
            process from the engine, so the graph leaves via the filesystem
            rather than a return value.
        measure_out: Where measured step timings are written, one JSON row per
            step. Separate from ``graph_out`` because the two artifacts answer
            different questions and are produced by different runs.
        measure_warmup_steps: Steps of each kind to time but discard. Defaults
            to zero, because discarding by step count is a bad way to exclude
            warmup: prefill happens a handful of times in a whole run, so any
            non-zero value here can consume every prefill sample there is,
            leaving an oracle with nothing to fit and a TTFT prediction of zero.
            Exclude Triton autotuning with throwaway *requests* instead — see
            ``--warmup-prompts`` in ``scripts/compass/run.py`` — which drops the
            expensive first launches without discarding a whole category.
        memory_out: Where to record what the memory budget was made of. Every
            run already computes the terms and throws them away once the block
            count is derived; a run that does not record them is a validation
            sample that cannot be recovered, because the numbers only exist
            while the device is in that state.
        trace_prefill: Which prefill step to record as well, counting prefills
            from one, or 0 to record none. A decode graph says nothing about a
            prefill step -- different operators at different shapes -- so an
            oracle holding only one has to fall through to a calibrated fallback
            for every TTFT it is asked about.
        trace_step: Which *decode* step to record, counting from one. Not the first:
            Triton autotunes on a kernel's first launch, benchmarking every
            candidate configuration, so an initial step records tens of
            thousands of launches that steady-state serving never performs.
        oracle_qualname: Fully-qualified class name of the cost oracle, resolved
            the same way ATOM resolves ``runner_qualname``. The default costs
            every step identically, which is useful only for proving the
            plumbing.
        oracle_options: Keyword arguments passed to the oracle's constructor.
        virtual_clock: Advance a virtual clock by each predicted duration
            instead of sleeping. This is what makes a simulated run faster than
            the run it stands in for.
        op_timings_out: Where a traced step writes how long each operator took.
            Only meaningful in ``trace`` mode, which is the one that runs
            eagerly -- a replayed CUDA graph is a single submission with nothing
            to observe inside it. The artifact is written beside the graph and
            joined to it by operator index, so the graph stays a purely
            structural record that a derived run can still be compared against.
        bench_graph: A captured graph whose operators to price, and
            ``bench_out`` where to write the price list. Runs once after warmup
            and only there: ``aiter`` registers its kernels lazily on first
            call, and the model runs in a worker, so a kernel is reachable
            nowhere else. Pricing after warmup also means the kernels priced are
            the ones this deployment autotuned, not a fresh build.
        bench_cache: ``hot`` reuses one set of inputs, ``cold`` rotates over
            enough of them to overflow the cache. Cache state is really per
            argument -- a gemm's activation is hot because the previous operator
            wrote it, while its weight is cold and every gemm in a step uses a
            different one -- so these bracket the answer rather than give it.
        bench_iters: Calls per signature, timed as one block. Per-call timing is
            what this exists to avoid -- it cost eleven times the step it was
            measuring.
        admission_seconds: How long a request takes to reach the point of being
            schedulable, in seconds. A simulated run advances its clock by
            predicted *forward* durations, so the time a request spends getting
            from ``preprocess`` to the engine core and from there to a worker --
            two process hops through polling loops, with an idle engine on the
            far side and so nothing to overlap against -- does not exist. It is
            not small: measured at 8-18 ms on this deployment, which is the
            whole of a −20% TTFT error that four rounds of work on the cost
            model could not touch.

            Defaults to zero, which reproduces the behaviour before this
            existed. It has to be **measured per deployment** -- it is a
            property of the machine and the process layout, not of the model --
            and a measure-mode run reports the value to use.
        filler_token_id: Token emitted for every generated position. Simulated
            output is not meaningful text; it exists so sequences advance and
            terminate. Must not collide with the model's EOS id, or requests
            would stop on their first token.
    """

    enabled: bool = False
    epoch: Optional[float] = None
    mode: str = "predict"
    graph_out: Optional[str] = None
    measure_out: Optional[str] = None
    measure_warmup_steps: int = 0
    trace_step: int = 2
    trace_prefill: int = 0
    memory_out: Optional[str] = None
    oracle_qualname: str = "atom.compass.core.cost.constant.ConstantCostOracle"
    oracle_options: Optional[dict] = None
    virtual_clock: bool = True
    admission_seconds: float = 0.0
    op_timings_out: Optional[str] = None
    bench_graph: Optional[str] = None
    bench_out: Optional[str] = None
    bench_iters: int = 2000
    bench_cache: str = "hot"
    filler_token_id: int = 100

    def __post_init__(self) -> None:
        if self.enabled and self.epoch is None:
            # Pinned once, then carried to every process that stamps request
            # timestamps. Without a shared origin, a time recorded in one
            # process is not comparable to one recorded in another — which
            # shows up immediately as a negative TTFT.
            import time as _time

            self.epoch = _time.time()
        if self.oracle_options is None:
            self.oracle_options = {}
        if self.filler_token_id < 0:
            raise ValueError(f"filler_token_id must be >= 0, got {self.filler_token_id}")
        if self.mode not in ("predict", "trace", "measure"):
            raise ValueError(
                f"mode must be 'predict', 'trace' or 'measure', got {self.mode!r}"
            )
        if self.mode == "trace" and not self.graph_out:
            raise ValueError("mode='trace' needs graph_out to write the graph to")
        if self.mode == "measure" and not self.measure_out:
            raise ValueError(
                "mode='measure' needs measure_out to write the timings to"
            )
        if self.mode != "predict" and self.virtual_clock:
            # trace and measure perform the real forward, so the wall clock is
            # the truthful one. Leaving the virtual clock installed makes a real
            # run report the timings of a simulation that never ran: every
            # request comes back with a TTFT of zero, which reads as a broken
            # measurement rather than a misconfigured clock.
            self.virtual_clock = False
        if self.measure_warmup_steps < 0:
            raise ValueError(
                f"measure_warmup_steps must be >= 0, got {self.measure_warmup_steps}"
            )
        if self.trace_step < 1:
            raise ValueError(f"trace_step counts from one, got {self.trace_step}")
        if self.trace_prefill < 0:
            raise ValueError(
                f"trace_prefill counts from one, got {self.trace_prefill}")
