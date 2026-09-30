# SPDX-License-Identifier: MIT
"""Two processes, two hash seeds, one configuration, and one step table diffed.

Every comparison that matters is between two child processes started with
different `PYTHONHASHSEED` values, and each prints the hash of one LP name so
the seeds are shown to have differed. Inside one process the check cannot fail:
both runs hash from one seed, so a `set` hands its members over in one order
twice. That is shown here too, on the seeded defect the children tell apart.

The seeded defect breaks a tie the design fixes: messages that arrive at one
instant are handed to their LP out of a `set` rather than in `(arrival,
channel, seq)` order. The engine then serves a same-arrival pair in the other
order, which changes what it sends. Wall-clock interleaving, which the design
lets vary, is varied in process by the driver's request order, and changes the
order rows reach the table in but not its text.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from atom.compass.detect.determinism import StepTable, compare_step_tables

from .repeat import CONFIGURATIONS, RecordedRun

TREE = Path(__file__).resolve().parents[3]

#: Both non-zero, since `0` turns hash randomisation off. The seeded defect is
#: shown to come apart at this pair on every configuration, so a clean table
#: that agrees at it has been compared by a check that can fail.
SEEDS = ("1", "2")


def child(configuration, seed, *flags):
    """One recorded run in its own process: its table, hash probe and `atom`."""
    done = subprocess.run(
        [sys.executable, "-m", "clock.repeat", configuration, *flags],
        cwd=TREE / "tests" / "compass",
        env=dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(TREE)),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, f"seed {seed} child failed:\n{done.stderr}"
    probe, module = done.stderr.splitlines()
    return done.stdout, probe, module


def names():
    return tuple(f"seed {seed}" for seed in SEEDS)


@pytest.fixture(scope="module", params=sorted(CONFIGURATIONS))
def clean(request):
    return request.param, [child(request.param, seed) for seed in SEEDS]


@pytest.fixture(scope="module", params=sorted(CONFIGURATIONS))
def defective(request):
    return request.param, [
        child(request.param, seed, "--hand-over-by-set") for seed in SEEDS
    ]


class TestTheSameConfigurationTwice:
    def test_the_two_processes_hashed_differently_on_the_tree_under_test(self, clean):
        _, runs = clean
        probes = [probe for _, probe, _ in runs]
        assert all(probe.startswith("hash-probe ") for probe in probes)
        assert probes[0] != probes[1]
        for _, _, module in runs:
            assert module == f"atom {TREE / 'atom' / '__init__.py'}"

    def test_the_tables_are_byte_identical_and_the_comparison_says_so(self, clean):
        configuration, runs = clean
        left, right = (table for table, _, _ in runs)
        assert left == right
        code, report = compare_step_tables(left, right, *names())
        assert code == 0
        rows = left.count("\n")
        assert report == (
            f"determinism: seed 1 and seed 2 are identical over "
            f"{rows} row(s) of {configuration}"
        )
        lps = len(RecordedRun(configuration).lps)
        assert left.count(" END ") == lps
        assert " TAR " in left and " NER " in left and " receive " in left

    def test_the_order_rows_reach_the_table_in_does_not_reach_its_text(self):
        """The driver's request order stands in for wall-clock interleaving.

        M4 only: on M1-M3 all six orders give one row order.
        """
        first = RecordedRun("M4")
        second = RecordedRun("M4", sorted(map(str, first.lps), reverse=True))
        reports = [first.run(), second.run()]
        assert [r.stopped_by for r in reports] == ["END from traffic"] * 2
        assert reports[0].submitted != reports[1].submitted
        assert first.step_table.rows != second.step_table.rows
        assert first.step_table.text() == second.step_table.text()


class TestTheSeededDefect:
    def test_two_seeds_give_two_tables_and_the_report_names_the_first_row(
        self, defective
    ):
        configuration, runs = defective
        left, right = (table for table, _, _ in runs)
        code, report = compare_step_tables(left, right, *names())
        assert code == 1
        first, one, other = report.split("\n")
        assert first.startswith(
            f"determinism: seed 1 and seed 2 ran {configuration} and produced "
            "different records, first at row "
        )
        assert one.startswith("  seed 1   ") and other.startswith("  seed 2   ")
        assert one[len("  seed 1") :] != other[len("  seed 2") :]

    def test_one_process_cannot_tell_the_defect_from_a_clean_run(self, defective):
        configuration, _ = defective
        tables = []
        for _ in range(2):
            run = RecordedRun(configuration, hand_over_by_set=True)
            run.run()
            tables.append(run.step_table.text())
        assert compare_step_tables(*tables, "first", "second")[0] == 0


def test_a_sleep_on_every_reply_costs_wall_time_and_changes_no_row():
    quick = RecordedRun("M1-M3")
    quick.run()
    slow = RecordedRun("M1-M3", sleep_seconds=0.001)
    started = time.perf_counter()
    slow.run()
    seconds = time.perf_counter() - started
    replies = sum(row.event != "receive" for row in slow.step_table.rows)
    assert seconds >= replies * 0.001
    assert slow.step_table.text() == quick.step_table.text()


class TestWhatTheComparisonReportsOnItsOwn:
    def table(self, name, *rows):
        table = StepTable(name)
        for row in rows:
            table.record(*row)
        return table

    def test_two_records_of_different_configurations_are_not_a_divergence(self):
        code, report = compare_step_tables(
            self.table("one").text(), self.table("another").text(), "left", "right"
        )
        assert code == 1
        assert report.split("\n")[0] == (
            "determinism: the two runs are not the same configuration, so there "
            "is nothing to conclude from their records differing:"
        )

    def test_a_record_with_no_rows_is_refused(self):
        empty = self.table("one").text()
        code, report = compare_step_tables(empty, empty, "left", "right")
        assert code == 1
        assert report.split("\n")[0] == (
            "determinism: a record with no rows is a run that scheduled "
            "nothing, so there is nothing for the two to agree about:"
        )

    def test_a_record_that_stops_early_diverges_at_the_row_it_stops(self):
        row = ("a", 0.0, 1.0, "TAR")
        short = self.table("one", row).text()
        long = self.table("one", row, ("b", 1.0, 2.0, "NER")).text()
        code, report = compare_step_tables(short, long, "short", "long")
        assert code == 1
        assert report.split("\n") == [
            (
                "determinism: short and long ran one and produced different "
                "records, first at row 2 of 1 and 2:"
            ),
            "  short   <the record ends here>",
            "  long   b 1.0 2.0 NER - - -",
        ]

    def test_rows_are_ordered_by_time_lp_channel_and_seq_not_as_recorded(self):
        """Two channels between one pair of LPs, at one time, stay apart."""
        rows = [
            ("engine", 0.5, 1.0, "TAR"),
            ("engine", 1.0, 1.0, "receive", "frontend->engine:request#dp0", 1, "b"),
            ("engine", 1.0, 1.0, "receive", "frontend->engine:control#dp0", 0, "c"),
            ("engine", 1.0, 1.0, "receive", "frontend->engine:request#dp0", 0, "a"),
            ("engine", 0.0, 0.5, "NER"),
            ("frontend", 0.0, 0.5, "NER"),
        ]
        text = self.table("one", *rows).text()
        assert text == self.table("one", *rows[::-1]).text()
        assert text.split("\n")[1:] == [
            "engine 0.0 0.5 NER - - -",
            "frontend 0.0 0.5 NER - - -",
            "engine 0.5 1.0 TAR - - -",
            "engine 1.0 1.0 receive frontend->engine:control#dp0 0 c",
            "engine 1.0 1.0 receive frontend->engine:request#dp0 0 a",
            "engine 1.0 1.0 receive frontend->engine:request#dp0 1 b",
        ]
