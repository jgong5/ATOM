# SPDX-License-Identifier: MIT
"""The step table two runs of one configuration are compared by, and the comparison.

A row is one event of one LP in virtual time: a reply that moved its clock,
with `channel` and `seq` empty, or a message it was handed. Wall time, threads,
processes and the hash seed are not in it, so two runs that differ only in
those give one table.

The text is the rows sorted by `(time, LP, channel, seq)`, the time being
`time_to`. Rows reach a recorder in wall order, the order replies are issued
and messages handed over, and the sort takes that order out. It is stable, so
rows of one LP under one key keep the order the LP produced them in, and the
channel is in the key, so two channels between one pair of LPs stay apart.
"""

from dataclasses import dataclass

#: The header line, so a table of one configuration is never compared as if
#: it were a table of another.
CONFIGURATION_PREFIX = "configuration "

#: What stands in a report for the table that ran out of rows first.
ENDS_HERE = "<the record ends here>"


@dataclass(frozen=True)
class StepRow:
    """One event of one LP. `channel` and `seq` are empty on a step row."""

    lp: str
    time_from: float
    time_to: float
    event: str
    channel: str = ""
    seq: int | str = ""
    detail: str = ""

    def key(self) -> tuple:
        return (self.time_to, self.lp, self.channel, self.seq)

    def __str__(self) -> str:
        return (
            f"{self.lp} {self.time_from!r} {self.time_to!r} {self.event} "
            f"{self.channel or '-'} {self.seq if self.channel else '-'} "
            f"{self.detail or '-'}"
        )


class StepTable:
    """The rows one run produced, in the order they reached it."""

    def __init__(self, configuration: str) -> None:
        self.configuration = configuration
        self.rows: list[StepRow] = []

    def record(
        self, lp, time_from, time_to, event, channel="", seq="", detail=""
    ) -> None:
        self.rows.append(
            StepRow(
                str(lp),
                float(time_from),
                float(time_to),
                event,
                channel,
                seq,
                str(detail),
            )
        )

    def text(self) -> str:
        """The header, then the rows sorted by `(time, LP, channel, seq)`."""
        rows = sorted(self.rows, key=StepRow.key)
        return "\n".join([CONFIGURATION_PREFIX + self.configuration, *map(str, rows)])


def compare_step_tables(
    left: str, right: str, left_name: str, right_name: str
) -> tuple[int, str]:
    """Diff two tables' text; return an exit code beside the report.

    Non-zero fails CI. A divergence is reported at its first row: that row has
    a cause, and the rows after it are the cause unwinding. Two tables of
    different configurations are not compared, and a table with no rows is
    refused: two runs that scheduled nothing agree, and the agreement means
    nothing.
    """
    left_rows, right_rows = left.split("\n"), right.split("\n")
    if left_rows[0] != right_rows[0]:
        return 1, (
            "determinism: the two runs are not the same configuration, so there "
            "is nothing to conclude from their records differing:\n"
            f"  {left_name}  {left_rows[0]}\n"
            f"  {right_name}  {right_rows[0]}"
        )
    configuration = left_rows[0][len(CONFIGURATION_PREFIX) :]
    counted = f"{len(left_rows) - 1} and {len(right_rows) - 1}"
    if len(left_rows) == 1 or len(right_rows) == 1:
        return 1, (
            "determinism: a record with no rows is a run that scheduled "
            "nothing, so there is nothing for the two to agree about:\n"
            f"  {left_name}  {len(left_rows) - 1} row(s) of {configuration}\n"
            f"  {right_name}  {len(right_rows) - 1} row(s) of {configuration}"
        )
    padded = max(len(left_rows), len(right_rows))
    left_rows += [ENDS_HERE] * (padded - len(left_rows))
    right_rows += [ENDS_HERE] * (padded - len(right_rows))
    for index in range(1, padded):
        if left_rows[index] != right_rows[index]:
            return 1, "\n".join(
                [
                    (
                        f"determinism: {left_name} and {right_name} ran "
                        f"{configuration} and produced different records, first "
                        f"at row {index} of {counted}:"
                    ),
                    f"  {left_name}   {left_rows[index]}",
                    f"  {right_name}   {right_rows[index]}",
                ]
            )
    return 0, (
        f"determinism: {left_name} and {right_name} are identical over "
        f"{padded - 1} row(s) of {configuration}"
    )
