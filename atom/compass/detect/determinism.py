# SPDX-License-Identifier: MIT
"""The record two runs of one configuration are compared by, and the comparison.

A simulated run that disagrees with itself cannot be compared with anything.
The comparison has to be made against something, though, and the obvious choice
is wrong: two runs of one configuration take different real durations, hand out
grants in a different real order, and schedule their threads differently, and
none of that makes them different runs. Real time is a property of the machine
the run happened on, not of the run.

So the record here holds **who moved, from when to when, and on account of
what** -- one row per advance of one participant's clock, in the order the run
produced them -- and holds nothing else. What is in it:

* the participant's name;
* the simulated time its clock moved from and to, to the nanosecond;
* the event, and one field of detail naming the peer involved.

What is deliberately not in it: how long the run took, when it started, which
thread did what, how many processes it used, and the seed the interpreter
happened to pick. A run that differs from another only in those has not
differed. A run that differs in one row of this record has produced a different
schedule, and every number taken off it afterwards describes a different run.

The header row names the configuration, so two records of *different*
configurations are told apart from two records of one configuration that
disagreed. The two are not the same finding and the second is the only one that
is a defect.
"""

from dataclasses import dataclass

#: What the header line says, so a record of one configuration is never
#: silently compared against a record of another.
CONFIGURATION_PREFIX = "configuration "

#: What stands in the diff for the record that ran out of rows first. A run
#: that stopped early differs from one that did not, and the row it stopped at
#: is where that difference begins.
ENDS_HERE = "<the record ends here>"


@dataclass(frozen=True)
class StepRow:
    """One advance of one participant's clock, and what caused it."""

    lp: str
    advance_from: float
    advance_to: float
    event: str
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.lp} {self.advance_from:.9f} {self.advance_to:.9f} "
            f"{self.event} {self.detail}"
        )


class StepTable:
    """The rows one run produced, and the text two runs are diffed by."""

    def __init__(self, configuration: str) -> None:
        self.configuration = configuration
        self.rows: list[StepRow] = []

    def record(self, lp, advance_from, advance_to, event, detail) -> None:
        """Append one advance. Called once per grant and once per message sent."""
        self.rows.append(
            StepRow(str(lp), float(advance_from), float(advance_to), event, str(detail))
        )

    def text(self) -> str:
        """The whole record as bytes two processes can be diffed on."""
        return "\n".join(
            [
                CONFIGURATION_PREFIX + self.configuration,
                *(str(row) for row in self.rows),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)


def compare_step_tables(
    left: str, right: str, left_name: str, right_name: str
) -> tuple[int, str]:
    """Diff two records and return an exit code beside the report.

    Non-zero fails CI. The report names the first row the two disagree on
    rather than the whole diff, because the first one is the one that has a
    cause and everything after it is that cause still unwinding.
    """
    left_rows = left.split("\n")
    right_rows = right.split("\n")
    if left_rows[0] != right_rows[0]:
        return 1, (
            "determinism: the two runs are not the same configuration, so there "
            "is nothing to conclude from their records differing:\n"
            f"  {left_name}  {left_rows[0]}\n"
            f"  {right_name}  {right_rows[0]}"
        )
    configuration = left_rows[0][len(CONFIGURATION_PREFIX) :]
    counted = f"{len(left_rows) - 1} and {len(right_rows) - 1}"
    for index, (one, other) in enumerate(zip(left_rows[1:], right_rows[1:])):
        if one != other:
            return 1, _differ(
                configuration, index + 1, counted, (left_name, one), (right_name, other)
            )
    if len(left_rows) < len(right_rows):
        short, long = (left_name, left_rows), (right_name, right_rows)
    elif len(right_rows) < len(left_rows):
        short, long = (right_name, right_rows), (left_name, left_rows)
    else:
        return 0, (
            f"determinism: {left_name} and {right_name} are identical over "
            f"{len(left_rows) - 1} row(s) of {configuration}"
        )
    return 1, _differ(
        configuration,
        len(short[1]),
        counted,
        (short[0], ENDS_HERE),
        (long[0], long[1][len(short[1])]),
    )


def _differ(configuration, row, counted, one, other):
    """What a divergence prints: where it started, and what it means."""
    return "\n".join(
        [
            (
                f"determinism: {one[0]} and {other[0]} ran {configuration} and "
                f"produced different records, first at row {row} of {counted}:"
            ),
            f"  {one[0]}   {one[1]}",
            f"  {other[0]}   {other[1]}",
            (
                "Both runs were given the same configuration, and a row is one "
                "participant's clock moving from one simulated time to another. "
                "A row that differs is a different participant, a different time, "
                "or a different event, so the two are not one run measured twice "
                "and no number taken off either can be compared with the other."
            ),
        ]
    )
