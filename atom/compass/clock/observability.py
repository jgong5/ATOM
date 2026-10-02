# SPDX-License-Identifier: MIT
"""What a run of the clock says about itself: a timeline, an LP table dump and
a run summary.

**The timeline.** One record per reply the Clock Authority issues, in issue
order: which LP, from which clock to which, the kind of request it answers
(``TAR`` or ``NER``; a reply to ``+inf`` finishes the run), and whether the
recovery branch issued it. It is off unless a run hands the authority a
`TimelineLog`; with none, a reply costs one ``is not None`` test.

**The dump.** `lp_table_dump` renders `ClockAuthority.lp_table()` as text: each
LP's state, clock, target and ``N``, the messages registered for it and not yet
released, and its ``N[j] + D(j->i)`` row with the binding term marked. It states
the table and draws no conclusion from it.

**The summary**, written once, by value, in two halves. `schedule_record` is a
function of the simulated schedule alone, so two runs of one configuration give
it byte for byte whatever order the LPs' requests arrived in. `cost_record` is
what the run cost to produce, and names what its numbers move with. Every value
is a number, a string, ``None`` or a list of those, and none is infinite or NaN,
so the record survives strict JSON.

Nothing here reads a clock or opens a file or a socket: the caller supplies the
sink and the wall seconds.
"""

import math
from dataclasses import dataclass, field

from .identity import LpId

#: The least simulated seconds per wall second a run must reach.
SPEED_TARGET_RATIO = 5.0


def _seconds(value: float | None) -> str:
    # `repr`, not a fixed precision: nine digits stop resolving a microsecond
    # once a clock passes a thousand seconds.
    if value is None:
        return "none"
    return "inf" if value == math.inf else f"{value!r}s"


def _finite_seconds(value: float, what: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(
            f"{what} must be a finite number of seconds and not negative, got {value!r}"
        )
    return seconds


# --- the timeline ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimelineRecord:
    """One reply: five columns, the last ``recovery`` or ``-``."""

    lp: LpId
    time_from: float
    time_to: float
    kind: str
    recovered: bool

    def __str__(self) -> str:
        return (
            f"{self.lp} {self.time_from!r} {self.time_to!r} {self.kind} "
            f"{'recovery' if self.recovered else '-'}"
        )


class TimelineLog:
    """Every reply, in issue order; append-only.

    Kept in memory, handed line by line to `sink` if one is given, or both.
    `retain=False` keeps no record, so a run streaming to a sink does not
    also hold every record; `records()` then refuses.
    """

    def __init__(self, sink=None, retain: bool = True) -> None:
        if sink is not None and not callable(sink):
            raise TypeError(f"sink must be callable, got {type(sink).__name__}")
        if not retain and sink is None:
            raise ValueError(
                "a log that neither keeps its records nor hands them to a sink "
                "would write nothing; pass a sink, or leave retain true"
            )
        self._records = [] if retain else None
        self._sink = sink

    def record(
        self, lp: LpId, time_from: float, time_to: float, kind: str, recovered: bool
    ):
        entry = TimelineRecord(lp, time_from, time_to, kind, recovered)
        if self._records is not None:
            self._records.append(entry)
        if self._sink is not None:
            self._sink(str(entry))

    def records(self) -> tuple[TimelineRecord, ...]:
        if self._records is None:
            raise ValueError(
                "this log was asked not to retain its records; they were handed "
                "to its sink"
            )
        return tuple(self._records)

    def lines(self) -> tuple[str, ...]:
        return tuple(str(entry) for entry in self.records())


# --- the dump ----------------------------------------------------------------


def lp_table_dump(authority) -> str:
    """`authority.lp_table()` as text, one block per LP, in name order."""
    lines = []
    for row in authority.lp_table():
        lines.append(
            f"{row.lp} {row.state} now {_seconds(row.now)} "
            f"target {_seconds(row.target)} N {_seconds(row.n)}"
        )
        for name, seq, arrival in row.undelivered:
            lines.append(f"    undelivered {name} seq {seq} at {_seconds(arrival)}")
        for j, term in row.row:
            binds = " <- binds" if j == row.binding else ""
            lines.append(f"    from {j} N+D {_seconds(term)}{binds}")
    return "\n".join(lines)


# --- the summary -------------------------------------------------------------


@dataclass(frozen=True)
class RefusalTally:
    """Every refusal in a run, each reason named ``source:detail``.

    The sources in use are ``cost`` (the cost model declined to price a step),
    ``command`` (a refused worker control command) and ``executor`` (a job the
    simulated executor refused); any other name is tallied the same way. The
    two fractions count ``cost`` reasons only, since only those are steps.
    """

    steps: int = 0
    refused_predicted_seconds: float = 0.0
    predicted_seconds: float = 0.0
    reasons: tuple[tuple[str, int], ...] = ()

    @classmethod
    def of(
        cls,
        reasons,
        steps: int,
        refused_predicted_seconds: float = 0.0,
        predicted_seconds: float = 0.0,
    ) -> "RefusalTally":
        """Count each reason; sorted, since the order refusals happen in is not
        reproducible between runs."""
        counted: dict[str, int] = {}
        for reason in reasons:
            source, _, detail = reason.partition(":")
            if not source or not detail:
                raise ValueError(f"a refusal reason is `source:detail`, got {reason!r}")
            counted[reason] = counted.get(reason, 0) + 1
        return cls(
            steps,
            _finite_seconds(refused_predicted_seconds, "refused_predicted_seconds"),
            _finite_seconds(predicted_seconds, "predicted_seconds"),
            tuple(sorted(counted.items())),
        )

    def record(self) -> dict:
        by_source: dict[str, int] = {}
        for reason, count in self.reasons:
            source = reason.partition(":")[0]
            by_source[source] = by_source.get(source, 0) + count
        cost = by_source.get("cost", 0)
        return {
            "count": sum(by_source.values()),
            "by_source": [[source, n] for source, n in sorted(by_source.items())],
            "steps": self.steps,
            "fraction_of_steps": cost / self.steps if self.steps else 0.0,
            "fraction_of_predicted_seconds": (
                self.refused_predicted_seconds / self.predicted_seconds
                if self.predicted_seconds
                else 0.0
            ),
            "reasons": [[reason, count] for reason, count in self.reasons],
        }


@dataclass(frozen=True)
class DetectorState:
    """What the causality detectors saw. A straggler above zero invalidates the run."""

    stragglers: int = 0
    clock_lint: str = "not-run"


@dataclass(frozen=True)
class RunSummary:
    """The run in two halves: what it simulated, and what that cost."""

    final_clocks: tuple[tuple[str, float], ...]
    grants: tuple[tuple[str, int], ...]
    wall_seconds: float
    lazy_traces: int = 0
    lazy_trace_wall_seconds: float = 0.0
    diagnostics: int = 0
    detectors: DetectorState = field(default_factory=DetectorState)
    refusals: RefusalTally = field(default_factory=RefusalTally)

    @classmethod
    def of(
        cls,
        authority,
        wall_seconds: float,
        lazy_traces: int = 0,
        lazy_trace_wall_seconds: float = 0.0,
        diagnostics: int = 0,
        detectors: DetectorState | None = None,
        refusals: RefusalTally | None = None,
    ) -> "RunSummary":
        """Read the authority once and take everything else by value.

        A finished run reports each LP's clock before the finish granted it
        ``+inf``; a run that has not finished, such as one a detector stopped,
        reports each LP's clock as it stands.
        """
        clocks = authority.final_clocks
        if clocks is None:
            clocks = tuple((row.lp, row.now) for row in authority.lp_table())
        return cls(
            tuple((str(lp), now) for lp, now in clocks),
            tuple(authority.grants.items()),
            _finite_seconds(wall_seconds, "wall_seconds"),
            lazy_traces,
            _finite_seconds(lazy_trace_wall_seconds, "lazy_trace_wall_seconds"),
            diagnostics,
            detectors if detectors is not None else DetectorState(),
            refusals if refusals is not None else RefusalTally(),
        )

    @property
    def simulated_seconds(self) -> float:
        return max((now for _, now in self.final_clocks), default=0.0)

    @property
    def speed_refused(self) -> str | None:
        """Why there is no speed result, or ``None`` when there is one.

        No measured wall time gives no ratio rather than an unlimited one,
        which a gate would read as a pass.
        """
        if self.wall_seconds == 0.0:
            return "no wall time was measured, so this run has no speed result"
        return None

    @property
    def speed_ratio(self) -> float | None:
        if self.speed_refused is not None:
            return None
        return self.simulated_seconds / self.wall_seconds

    def schedule_record(self) -> dict:
        """The half that is a function of the simulated schedule alone."""
        return {
            "final_clocks": [[name, clock] for name, clock in self.final_clocks],
            "grants": [[name, count] for name, count in self.grants],
            "grants_total": sum(count for _, count in self.grants),
            "stragglers": self.detectors.stragglers,
            "clock_lint": self.detectors.clock_lint,
            "refusals": self.refusals.record(),
        }

    def cost_record(self) -> dict:
        """The half that says what the run cost, and what each number moves with."""
        ratio = self.speed_ratio
        return {
            "varies_with": ["host", "wall-clock interleaving"],
            "wall_seconds": self.wall_seconds,
            "simulated_seconds": self.simulated_seconds,
            "speed_ratio": ratio,
            "speed_refused": self.speed_refused,
            "speed_target_ratio": SPEED_TARGET_RATIO,
            "meets_speed_target": (
                None if ratio is None else ratio >= SPEED_TARGET_RATIO
            ),
            "lazy_traces": self.lazy_traces,
            "lazy_trace_wall_seconds": self.lazy_trace_wall_seconds,
            "diagnostics": self.diagnostics,
        }

    def as_record(self) -> dict:
        return {"schedule": self.schedule_record(), "cost": self.cost_record()}
