"""A cost oracle that answers from nearby measurements rather than a fitted form.

This is the F1 "empirical" point. Where the calibrated oracle assumes a shape
for the cost function and fits its coefficients, this one assumes nothing and
looks up what similar steps actually took.

The reason to have both is not completeness. A global fit is pulled by every
point it was given, including points nowhere near the question being asked, and
ATOM's decode cost turns out to be dominated by a roughly fixed floor — about
3.5 ms of CUDA-graph replay and scheduling — with only weak dependence on batch
size and context. Measured at matched context, batch 1, 2 and 4 cost 3.59, 3.85
and 3.78 ms: nearly flat, and not monotonic. A linear term fitted across a range
where context spans two orders of magnitude will not reproduce that, and the
error shows up wherever the evaluation sits away from the calibration's centre
of mass.

Nearest-neighbour prediction has the opposite failure mode, which is the useful
one here: it is wrong only where it has no nearby evidence, and it can say so.

Features are the same as the calibrated oracle's, standardised so that a
thousand tokens of context and a batch of eight are comparable distances.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from atom.compass.core.artifacts import resolve_rank_path
from atom.compass.core.cost.base import StepCost, StepShape

logger = logging.getLogger(__name__)

__all__ = ["EmpiricalCostOracle"]


def _features(shape: StepShape) -> list[float]:
    """What a step is described by, for the purpose of finding similar ones."""
    if shape.is_prefill:
        return [float(shape.num_prefill_tokens), float(shape.batch_size)]
    context = float(sum(shape.context_lens)) if shape.context_lens else 0.0
    return [context, float(shape.batch_size)]


class _Neighbourhood:
    """Standardised samples of one kind, and the lookup over them."""

    def __init__(self, rows: list[list[float]], targets: list[float], k: int) -> None:
        import numpy as np

        self.k = k
        self.points = np.asarray(rows, dtype=float)
        self.targets = np.asarray(targets, dtype=float)
        # Standardise per feature. Without it, context (thousands) drowns out
        # batch size (single digits) and every neighbour is chosen by context
        # alone.
        self.centre = self.points.mean(axis=0)
        spread = self.points.std(axis=0)
        self.spread = np.where(spread > 0.0, spread, 1.0)
        self.scaled = (self.points - self.centre) / self.spread
        self.low = self.points.min(axis=0)
        self.high = self.points.max(axis=0)

    def predict(self, features: list[float]) -> float:
        import numpy as np

        query = (np.asarray(features, dtype=float) - self.centre) / self.spread
        distances = np.linalg.norm(self.scaled - query, axis=1)
        k = min(self.k, len(distances))
        nearest = np.argpartition(distances, k - 1)[:k]
        d = distances[nearest]
        # An exact hit answers exactly; otherwise weight by inverse distance so
        # a close neighbour counts for more than one at the edge of the set.
        if float(d.min()) <= 1e-12:
            return float(self.targets[nearest[int(np.argmin(d))]])
        weights = 1.0 / d
        return float((self.targets[nearest] * weights).sum() / weights.sum())

    def outside(self, features: list[float]) -> list[int]:
        return [
            i for i, value in enumerate(features)
            if value < self.low[i] or value > self.high[i]
        ]


class EmpiricalCostOracle:
    """Predicts a step's duration from the measured steps most like it."""

    def __init__(self, table: str, neighbours: int = 5,
                 floor_seconds: float = 1e-6,
                 rank_coords: Optional[dict] = None) -> None:
        self.table, own = resolve_rank_path(table, rank_coords)
        if rank_coords and not own:
            logger.warning(
                "ATOMCompass WARNING: rank %s has no calibration table of its "
                "own; fitting to the shared %s. Ranks of a symmetric group time "
                "within a fraction of a percent of each other, so this is "
                "usually fine — but it is an assumption, not a measurement.",
                rank_coords, self.table,
            )

        if not os.path.exists(self.table):
            # Named here rather than left to `open`, because this raises inside
            # a worker process: what reaches the terminal is the engine
            # manager's summary of a worker that vanished, which names neither
            # Compass nor the file. The message has to carry its own context.
            tried = f" (also tried {resolve_rank_path(table, rank_coords)[0]})" \
                if rank_coords else ""
            raise FileNotFoundError(
                f"ATOMCompass: no calibration table at {table}{tried}. "
                f"Record one with --compass-mode=measure --compass-measure-out "
                f"before predicting from it."
            )
        self.neighbours = neighbours
        self.floor_seconds = floor_seconds
        self._kinds: dict[str, _Neighbourhood] = {}
        self._counts: dict[str, int] = {"prefill": 0, "decode": 0}
        self._warned: set = set()
        self._load()

    def _load(self) -> None:
        gathered: dict[str, tuple[list, list]] = {
            "prefill": ([], []), "decode": ([], []),
        }
        with open(self.table, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "seconds" not in row:
                    continue  # a header or other non-measurement record
                shape = StepShape(
                    num_scheduled_tokens=tuple(row["num_scheduled_tokens"]),
                    context_lens=tuple(row["context_lens"]),
                    num_prefill_tokens=row["num_prefill_tokens"],
                )
                kind = "prefill" if shape.is_prefill else "decode"
                rows, targets = gathered[kind]
                rows.append(_features(shape))
                targets.append(row["seconds"])

        for kind, (rows, targets) in gathered.items():
            self._counts[kind] = len(targets)
            if targets:
                self._kinds[kind] = _Neighbourhood(rows, targets, self.neighbours)

        if not self._kinds:
            raise ValueError(f"no usable measurements in {self.table}")

    def estimate(self, shape: StepShape) -> StepCost:
        kind = "prefill" if shape.is_prefill else "decode"
        neighbourhood = self._kinds.get(kind)
        if neighbourhood is None:
            # Never invent a number for a kind of step never measured. Returning
            # zero here is what once made TTFT come back as 0 ms against a real
            # 7.6 s: precise, confident and fictional.
            raise ValueError(
                f"{type(self).__name__}: asked to cost a {kind} step, but the "
                f"table {self.table} contains no {kind} measurements."
            )
        features = _features(shape)
        self._warn_if_outside(kind, neighbourhood, features)
        return StepCost(
            seconds=max(neighbourhood.predict(features), self.floor_seconds)
        )

    def _warn_if_outside(self, kind, neighbourhood, features) -> None:
        """Nearest-neighbour degrades to "the closest edge" outside its data.

        That is a gentler failure than a linear extrapolation, which diverges,
        but it is still an answer given without evidence and it should say so.
        Warned once per kind and feature; a serving run asks thousands of times.
        """
        for index in neighbourhood.outside(features):
            key = (kind, index)
            if key in self._warned:
                continue
            self._warned.add(key)
            logger.warning(
                "ATOMCompass WARNING: costing a %s step whose feature %d is %.0f, "
                "outside the measured range [%.0f, %.0f]. The answer is the "
                "nearest measured step, not an estimate for this one.",
                kind, index, features[index],
                neighbourhood.low[index], neighbourhood.high[index],
            )

    def describe(self) -> str:
        return (
            f"EmpiricalCostOracle(k={self.neighbours}, "
            f"prefill={self._counts['prefill']} steps, "
            f"decode={self._counts['decode']} steps)"
        )
