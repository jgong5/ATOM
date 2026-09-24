# SPDX-License-Identifier: MIT
"""Two processes, two hash seeds, one configuration, and one record diffed.

The claim under test is not that a run is repeatable. It is that the check
itself is capable of failing. A determinism test that runs both halves inside
one process cannot fail: both halves hash from the same seed, so a set comes
out in the same order twice and the records agree -- and they agree just as
readily when the code between them picks its schedule out of that set. That is
demonstrated here rather than asserted, by running the defective configuration
twice in this process and watching the two records match, and then running it
twice in two processes and watching them come apart.

So every comparison that matters is between two child processes, and each child
prints the hash of one participant's name so the two seeds can be shown to have
been different. A record that is identical across two seeds that were the same
is not evidence of anything.

**Different seeds is necessary and it is not sufficient.** What the harness can
see is bounded by how many orders the defect has to choose between, and that is
a property of the configuration rather than of the check: a set of many
participants comes out differently under almost any two seeds, a set of two
comes out one of two ways and two seeds can easily draw the same one. So the
seed pair is per configuration, and the same pair runs the defect and the clean
comparison. A pair that has stopped telling the defect apart fails in
`TestTheSeededDefect` rather than quietly turning the clean record beside it
into a comparison with nothing behind it.

The other half of the arrangement is what the record leaves out. Real time is
not in it, and the sleep test is what that costs: a run that spends real
seconds inside itself produces the same bytes as one that does not, and takes
measurably longer to do it. Wall time is a property of the machine, and there
is no path from it into the schedule.
"""

import os
import pathlib
import subprocess
import sys
import time

import pytest

from atom.compass.detect.determinism import (
    CONFIGURATION_PREFIX,
    StepRow,
    StepTable,
    compare_step_tables,
)

from . import repeat

#: The two seeds each configuration is run at, both here and defective. Both
#: are non-zero so both processes really do randomise; `0` turns randomisation
#: off, which makes a comparison against it weaker rather than stronger. The
#: pair differs by configuration because what a pair is worth differs by
#: configuration: it has to be a pair the seeded defect is shown to come apart
#: on, and the smaller the participant set the fewer such pairs there are.
SEED_PAIRS = {
    "tp8-pp4-four-requests": ("1", "12345"),
    "tp4-one-server-eight-requests": ("1", "2"),
}

#: The tree these runs import from, and the working directory the child is
#: started in, so the child resolves the same checkout the test came from.
TREE = pathlib.Path(__file__).resolve().parents[3]

#: How long a real sleep per grant costs the sleeping run. Small enough that
#: the test stays a test; large enough that it cannot be mistaken for noise.
SLEEP_SECONDS = 0.0002

#: The fields one row of the record has, in order. Named here so that adding a
#: sixth is a line somebody changes on purpose rather than ten assertions
#: somebody repairs.
ROW_FIELDS = ("lp", "advance_from", "advance_to", "event", "detail")


def child(configuration, seed, *flags):
    """One recorded run, in its own process, at the seed given."""
    environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(TREE))
    started = time.perf_counter()
    done = subprocess.run(
        [sys.executable, "-m", "tests.compass.clock.repeat", configuration, *flags],
        cwd=TREE,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout, done.stderr.strip(), time.perf_counter() - started


def rows(record, event):
    """The rows of one kind, so one sort of work can be counted on its own."""
    return [line for line in record.split("\n")[1:] if f" {event} " in line]


def named(configuration):
    """What the two runs of a configuration are called in a report."""
    return tuple(f"seed {seed}" for seed in SEED_PAIRS[configuration])


@pytest.fixture(scope="module", params=sorted(repeat.CONFIGURATIONS))
def both(request):
    """The same configuration run in two processes at its two seeds."""
    return request.param, [
        child(request.param, seed) for seed in SEED_PAIRS[request.param]
    ]


@pytest.fixture(scope="module", params=sorted(repeat.CONFIGURATIONS))
def defective(request):
    """The same configuration, picking its schedule out of a set, at the same two.

    Parametrised over every configuration the clean comparison claims, because
    a clean record is evidence only where the defective one is shown to differ.
    """
    return request.param, [
        child(request.param, seed, "--tie-break-by-set")
        for seed in SEED_PAIRS[request.param]
    ]


class TestTheSameConfigurationTwice:
    """The exit criterion: two processes, two seeds, one record."""

    def test_the_two_processes_hashed_differently(self, both):
        """Without this the comparison below is a comparison with itself."""
        _configuration, runs = both
        probes = [stderr for _record, stderr, _seconds in runs]
        assert all(probe.startswith("hash-probe ") for probe in probes)
        assert probes[0] != probes[1]

    def test_the_records_are_byte_identical(self, both):
        configuration, runs = both
        left, right = (record for record, _stderr, _seconds in runs)
        assert left == right
        assert left.startswith(CONFIGURATION_PREFIX + configuration)

    def test_the_comparison_says_so_and_counts_the_rows(self, both):
        configuration, runs = both
        left, right = (record for record, _stderr, _seconds in runs)
        one, other = named(configuration)
        code, report = compare_step_tables(left, right, one, other)
        counted = len(left.split("\n")) - 1
        assert code == 0
        assert report == (
            f"determinism: {one} and {other} are identical over "
            f"{counted} row(s) of {configuration}"
        )

    def test_the_record_holds_what_moved_and_nothing_about_the_machine(self, both):
        """Every row is a participant and two simulated times, and no real ones."""
        _configuration, runs = both
        record = runs[0][0]
        assert rows(record, "grant") and rows(record, "send")
        for line in record.split("\n")[1:]:
            name, advance_from, advance_to, event, _detail = line.split(
                " ", len(ROW_FIELDS) - 1
            )
            assert name and event in ("grant", "send")
            assert float(advance_to) >= float(advance_from)

    def test_a_row_is_these_five_fields_and_no_others(self):
        """The positive half of the exclusion, so the remedy is guarded too.

        Every assertion above breaks if a field is added, which is how a wall
        clock is kept out of the record. What none of them states is what the
        row *is*, so an author with ten reds in front of them can repair the
        assertions one at a time and never meet the decision. Here it is one
        line, and the line is the decision.
        """
        assert tuple(StepRow.__dataclass_fields__) == ROW_FIELDS

    def test_the_orderings_the_defect_can_take_are_recorded(self, both):
        """The harness's sensitivity, kept beside the seeds that rely on it.

        Two processes is not a fixed amount of evidence. The defect reorders
        the participants, so what two seeds can show is bounded by how many
        orders this many participants take, and that number belongs next to
        the configuration rather than in a reviewer's head.
        """
        configuration, runs = both
        record = runs[0][0]
        moved = {line.split(" ", 1)[0] for line in record.split("\n")[1:]}
        assert len(moved) == repeat.CONFIGURATIONS[configuration][-1]


class TestTheSeededDefect:
    """A schedule picked out of a set, and the two records it produces."""

    def test_two_seeds_produce_two_records(self, defective):
        _configuration, runs = defective
        left, right = (record for record, _stderr, _seconds in runs)
        assert left != right

    def test_the_comparison_names_the_first_row_they_disagree_on(self, defective):
        configuration, runs = defective
        left, right = (record for record, _stderr, _seconds in runs)
        one, other = named(configuration)
        code, report = compare_step_tables(left, right, one, other)
        assert code == 1
        lines = report.split("\n")
        assert lines[0].startswith(
            f"determinism: {one} and {other} ran "
            f"{configuration} and produced different records, first at row "
        )
        assert lines[1].startswith(f"  {one}   ")
        assert lines[2].startswith(f"  {other}   ")
        assert lines[1] != lines[2]

    def test_it_is_the_same_work_in_a_different_order(self, defective):
        """The defect moves the schedule, not the workload.

        Both runs send the same messages, so neither did more or less of the
        trace than the other; what differs is the order participants were
        served in and therefore how many grants it took. That is what makes
        this the shape of defect worth testing against: nothing about either
        run looks wrong on its own.
        """
        _configuration, runs = defective
        left, right = (record for record, _stderr, _seconds in runs)
        assert len(rows(left, "send")) == len(rows(right, "send"))
        assert len(rows(left, "grant")) != len(rows(right, "grant"))

    def test_one_process_cannot_tell_the_defect_from_a_clean_run(self, defective):
        """Why the children exist. Run in here, the defect passes the check.

        Two runs in this process hash from the seed this process was started
        with, so the set comes out in one order both times and the records
        agree. A determinism test written that way reports a pass against code
        carrying the very thing it is looking for -- and it does so on the same
        configuration the children have just shown coming apart.
        """
        configuration, _runs = defective
        once = repeat.record(configuration, tie_break_by_set=True)
        twice = repeat.record(configuration, tie_break_by_set=True)
        assert once.text() == twice.text()
        assert compare_step_tables(once.text(), twice.text(), "first", "second")[0] == 0


class TestWhatRealTimeCanReach:
    """Whether real seconds spent inside a run can get into its record."""

    CONFIGURATION = "tp4-one-server-eight-requests"

    def test_a_sleep_inside_the_run_costs_wall_time_and_changes_no_row(self):
        """Measured, because it is the difference between two kinds of defect.

        A real clock *read* puts machine seconds into a modelled number and the
        mixture is reported as one. A real *sleep* cannot: it spends seconds
        the run never asks about, and the record it produces is the same bytes.
        The cost is speed, and speed is not what this record is for.
        """
        seeds = SEED_PAIRS[self.CONFIGURATION]
        quick, _probe, quick_seconds = child(self.CONFIGURATION, seeds[0])
        slow, _probe, slow_seconds = child(
            self.CONFIGURATION, seeds[1], "--sleep-seconds", str(SLEEP_SECONDS)
        )
        assert slow == quick
        grants = len(rows(quick, "grant"))
        assert slow_seconds - quick_seconds > grants * SLEEP_SECONDS / 2


class TestWhatTheComparisonReportsOnItsOwn:
    """The report text, pinned away from the runs that happen to produce it."""

    def table(self, name, *rows_in):
        table = StepTable(name)
        for row in rows_in:
            table.record(*row)
        return table.text()

    def test_two_records_of_different_configurations_are_not_a_divergence(self):
        code, report = compare_step_tables(
            self.table("one"), self.table("another"), "left", "right"
        )
        assert code == 1
        assert report.split("\n")[0] == (
            "determinism: the two runs are not the same configuration, so there "
            "is nothing to conclude from their records differing:"
        )

    def test_a_record_with_no_rows_is_refused_rather_than_agreed_with(self):
        """Two runs that produced nothing agree, and agreeing means nothing.

        The suite above is safe because it asserts the rows exist. The next
        caller is not the suite; it is whatever points this at a real step
        table, and the refusal has to live with the primitive that caller uses.
        """
        code, report = compare_step_tables(
            self.table("one"), self.table("one"), "left", "right"
        )
        assert code == 1
        assert report.split("\n")[0] == (
            "determinism: a record with no rows is a run that scheduled "
            "nothing, so there is nothing for the two to agree about:"
        )

    def test_a_record_that_stops_early_is_a_divergence_at_the_row_it_stops(self):
        short = self.table("one", ("a", 0.0, 1.0, "grant", "b"))
        long = self.table(
            "one", ("a", 0.0, 1.0, "grant", "b"), ("b", 1.0, 2.0, "grant", "a")
        )
        code, report = compare_step_tables(short, long, "short", "long")
        assert code == 1
        lines = report.split("\n")
        assert lines[0].endswith("first at row 2 of 1 and 2:")
        assert lines[1] == "  short   <the record ends here>"
        assert lines[2] == "  long   b 1.000000000 2.000000000 grant a"

    def test_a_row_is_printed_to_the_nanosecond(self):
        text = self.table("one", ("a", 0.0, 1.0e-9, "grant", "b"))
        assert text.split("\n")[1] == "a 0.000000000 0.000000001 grant b"

    def test_a_detail_with_a_space_in_it_still_leaves_five_fields(self):
        """The row splits from the left, so only `detail` may hold a space."""
        text = self.table("one", ("a", 0.0, 1.0, "grant", "b and c"))
        assert text.split("\n")[1].split(" ", len(ROW_FIELDS) - 1)[-1] == "b and c"
