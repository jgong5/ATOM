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
            which is how a reference op graph is obtained. Tracing is
            deliberately not free: it exists to produce artifacts, not to serve.
        graph_out: Where a traced graph is written. The runner is a separate
            process from the engine, so the graph leaves via the filesystem
            rather than a return value.
        trace_step: Which forward to record, counting from one. Not the first:
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
        filler_token_id: Token emitted for every generated position. Simulated
            output is not meaningful text; it exists so sequences advance and
            terminate. Must not collide with the model's EOS id, or requests
            would stop on their first token.
    """

    enabled: bool = False
    epoch: Optional[float] = None
    mode: str = "predict"
    graph_out: Optional[str] = None
    trace_step: int = 2
    oracle_qualname: str = "atom.compass.core.cost.constant.ConstantCostOracle"
    oracle_options: Optional[dict] = None
    virtual_clock: bool = True
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
        if self.mode not in ("predict", "trace"):
            raise ValueError(
                f"mode must be 'predict' or 'trace', got {self.mode!r}"
            )
        if self.mode == "trace" and not self.graph_out:
            raise ValueError("mode='trace' needs graph_out to write the graph to")
        if self.trace_step < 1:
            raise ValueError(f"trace_step counts from one, got {self.trace_step}")
