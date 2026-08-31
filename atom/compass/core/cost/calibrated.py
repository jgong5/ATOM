"""A cost oracle fitted to steps that were actually timed.

This is the F1 "calibrated" point: the shape of the model is chosen by hand and
its coefficients come from measurement. It is deliberately the simplest thing
that could reproduce a serving run, because the purpose of the first one is to
produce an error number — until something predicts time, every claim about
Compass is structural, and structural claims cannot be ranked by how much they
matter.

Prefill and decode are fitted separately. They are not two regimes of one
function: prefill is compute-bound in the number of new tokens, decode is
bandwidth-bound in the KV history it must read. Fitting them together produces a
model that is wrong about both.

Features, per step:

* prefill — new tokens, and new tokens squared (attention is quadratic in the
  chunk, and chunked prefill makes the chunk a real variable)
* decode — batch size (one row of GEMM work each) and total context across the
  batch (the KV bytes that must be read)

Total context is summed rather than averaged deliberately. A decode batch mixing
short and long histories does not cost what its mean history suggests, and the
sum is the quantity the hardware actually moves.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from atom.compass.core.cost.base import StepCost, StepShape

logger = logging.getLogger(__name__)

__all__ = ["CalibratedCostOracle"]


def _prefill_features(shape: StepShape) -> list[float]:
    tokens = float(shape.num_prefill_tokens)
    return [1.0, tokens, tokens * tokens]


def _decode_features(shape: StepShape) -> list[float]:
    batch = float(shape.batch_size)
    context = float(sum(shape.context_lens)) if shape.context_lens else 0.0
    return [1.0, batch, context]


def _least_squares(rows: list[list[float]], targets: list[float]) -> Optional[list[float]]:
    """Non-negative-intercept least squares, or None if the fit is not supported.

    Refuses rather than extrapolates when there are fewer samples than
    coefficients. An underdetermined fit returns numbers that look like a model
    and predict nothing, which is the failure mode this whole project keeps
    running into.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dep of torch
        return None
    if len(rows) < len(rows[0]):
        return None
    a = np.asarray(rows, dtype=float)
    b = np.asarray(targets, dtype=float)
    coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
    return [float(c) for c in coeffs]


class CalibratedCostOracle:
    """Predicts step duration from coefficients fitted to measured steps."""

    def __init__(self, table: str, floor_seconds: float = 1e-6) -> None:
        """
        Args:
            table: Path to a JSONL file written by ``--compass-mode=measure``.
            floor_seconds: Smallest duration ever returned. A fitted model can
                produce a negative prediction outside the range it saw, and a
                negative step duration would run the virtual clock backwards.
        """
        self.table = table
        self.floor_seconds = floor_seconds
        self._prefill: Optional[list[float]] = None
        self._decode: Optional[list[float]] = None
        self._n_prefill = 0
        self._n_decode = 0
        self._fallback_prefill = 0.0
        self._fallback_decode = 0.0
        self._fit()

    def _fit(self) -> None:
        prefill_rows, prefill_targets = [], []
        decode_rows, decode_targets = [], []
        with open(self.table, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                shape = StepShape(
                    num_scheduled_tokens=tuple(row["num_scheduled_tokens"]),
                    context_lens=tuple(row["context_lens"]),
                    num_prefill_tokens=row["num_prefill_tokens"],
                )
                if shape.is_prefill:
                    prefill_rows.append(_prefill_features(shape))
                    prefill_targets.append(row["seconds"])
                else:
                    decode_rows.append(_decode_features(shape))
                    decode_targets.append(row["seconds"])

        self._n_prefill, self._n_decode = len(prefill_targets), len(decode_targets)
        # The mean is the fallback when a fit is refused: a poor predictor, but
        # one whose error is bounded by the spread of what was measured, rather
        # than an extrapolation that can be arbitrarily wrong.
        if prefill_targets:
            self._fallback_prefill = sum(prefill_targets) / len(prefill_targets)
            self._prefill = _least_squares(prefill_rows, prefill_targets)
        if decode_targets:
            self._fallback_decode = sum(decode_targets) / len(decode_targets)
            self._decode = _least_squares(decode_rows, decode_targets)

        if self._prefill is None and self._n_prefill:
            logger.warning(
                "ATOMCompass: %d prefill steps is too few to fit; using their "
                "mean. Predictions will not vary with prompt length.",
                self._n_prefill,
            )
        if self._decode is None and self._n_decode:
            logger.warning(
                "ATOMCompass: %d decode steps is too few to fit; using their "
                "mean. Predictions will not vary with batch size or context.",
                self._n_decode,
            )
        if not self._n_prefill and not self._n_decode:
            raise ValueError(f"no usable measurements in {self.table}")

    def estimate(self, shape: StepShape) -> StepCost:
        if shape.is_prefill:
            coeffs, features = self._prefill, _prefill_features(shape)
            fallback = self._fallback_prefill
        else:
            coeffs, features = self._decode, _decode_features(shape)
            fallback = self._fallback_decode
        if coeffs is None and not (self._n_prefill if shape.is_prefill else self._n_decode):
            # Never invent a number for a kind of step that was never measured.
            # Returning zero here is what made TTFT come back as 0 ms against a
            # real 7.6 s -- a confident, precise, entirely fictional answer. An
            # oracle asked something outside its evidence should say so.
            kind = "prefill" if shape.is_prefill else "decode"
            raise ValueError(
                f"{type(self).__name__}: asked to cost a {kind} step, but the "
                f"table {self.table} contains no {kind} measurements. Measure a "
                "workload that exercises it rather than extrapolating into it."
            )
        if coeffs is None:
            return StepCost(seconds=max(fallback, self.floor_seconds))
        predicted = sum(c * f for c, f in zip(coeffs, features))
        return StepCost(seconds=max(predicted, self.floor_seconds))

    def describe(self) -> str:
        kind = lambda c, n: "fitted" if c is not None else f"mean of {n}"  # noqa: E731
        return (
            f"CalibratedCostOracle(prefill={kind(self._prefill, self._n_prefill)}"
            f" on {self._n_prefill} steps, "
            f"decode={kind(self._decode, self._n_decode)}"
            f" on {self._n_decode} steps)"
        )
