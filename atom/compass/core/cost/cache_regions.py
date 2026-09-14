"""Explicit cached-prefill region additions without changing a base preset."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FinalPrefillRegion:
    query_tokens: int
    cached_history: tuple[int, int]
    prepare_intercept: float
    prepare_slope: float
    postprocess: float
    validation: str

    def contains(self, query, history):
        return query == self.query_tokens and self.cached_history[0] <= history <= self.cached_history[1]

    def breakdown(self, history):
        return {"<prepare>": self.prepare_intercept + self.prepare_slope * history,
                "<postprocess>": self.postprocess}


@dataclass(frozen=True)
class DiagnosticOutputlessRegion:
    histories: tuple[int, int]
    queries: tuple[int, ...]
    # One row per history, one column per query, all from source references.
    prepare: tuple[tuple[float, ...], ...]
    validation: str

    def contains(self, query, history):
        return (self.histories[0] <= history <= self.histories[-1]
                and self.queries[0] <= query <= self.queries[-1])

    def breakdown(self, query, history):
        index = next(i for i, (lo, hi) in enumerate(zip(self.queries, self.queries[1:]))
                     if lo <= query <= hi)
        lo, hi = self.queries[index:index + 2]
        fraction = (query - lo) / (hi - lo)
        values = [row[index] + fraction * (row[index + 1] - row[index]) for row in self.prepare]
        history_fraction = (history - self.histories[0]) / (self.histories[1] - self.histories[0])
        return {"<prepare>": values[0] + history_fraction * (values[1] - values[0]),
                "<postprocess>": 0.0}


@dataclass(frozen=True)
class CachedPrefillRegions:
    """Select explicit cached domains, delegating every other shape to the base.

    The final-query addition and the failed outputless experiment are distinct
    selections. Neither supplies an uncertainty interval: source scatter and
    heldout qualification travel in ``validation``, and a tolerance is not a
    confidence bound. Runtime policy is verified separately by the serving
    harness against the deployment scope retained in ``provenance``.
    """

    base: object
    final: FinalPrefillRegion | None
    diagnostic_outputless: DiagnosticOutputlessRegion | None
    version: str
    provenance: str

    @property
    def topologies(self):
        return self.base.topologies

    def _selection(self, shape):
        # New source support is TP1, compiled eager prefill, one cached row.
        # The failed-source rectangle is selected only by an explicit
        # diagnostic option; its frozen endpoints must not become a different
        # function just because the base also has a point at q=16384.
        if (len(shape.num_scheduled_tokens) != 1 or len(shape.context_lens) != 1
                or shape.num_prefill_tokens != shape.total_tokens
                or not shape.num_prefill_tokens or shape.compiled is not True
                or shape.capture_bucket is not None
                or any(int(width) != 1 for width in (shape.topology or {}).values())
                or any(int(rank) != 0 for rank in (shape.rank_coords or {}).values())):
            return None
        query = int(shape.num_scheduled_tokens[0])
        history = int(shape.context_lens[0]) - query
        if history <= 0:
            return None
        if shape.produces_output and self.final and self.final.contains(query, history):
            return self.final.breakdown(history)
        if (not shape.produces_output and self.diagnostic_outputless
                and self.diagnostic_outputless.contains(query, history)):
            return self.diagnostic_outputless.breakdown(query, history)
        return None

    def refusal(self, shape):
        return None if self._selection(shape) is not None else self.base.refusal(shape)

    def breakdown(self, shape):
        addition = self._selection(shape)
        return addition if addition is not None else self.base.breakdown(shape)

    def seconds(self, shape):
        return sum(self.breakdown(shape).values())

    def band(self, shape):
        if self._selection(shape) is not None:
            raise ValueError("cached-prefill source addition has no calibrated uncertainty band")
        return self.base.band(shape)

    def describe(self):
        return (f"{self.version}: passed final prefill={self.final is not None}; "
                f"FAILED-source diagnostic outputless={self.diagnostic_outputless is not None}; "
                f"base={self.base.describe()}; {self.provenance}")
