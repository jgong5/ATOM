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
    oracle_qualname: str = "atom.compass.core.cost.constant.ConstantCostOracle"
    oracle_options: Optional[dict] = None
    virtual_clock: bool = True
    filler_token_id: int = 100

    def __post_init__(self) -> None:
        if self.oracle_options is None:
            self.oracle_options = {}
        if self.filler_token_id < 0:
            raise ValueError(f"filler_token_id must be >= 0, got {self.filler_token_id}")
