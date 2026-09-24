# SPDX-License-Identifier: MIT
"""Three injected violations, three checks firing, and what each one prints.

The claim under test is not that the checks exist. It is that without them the
run **does not fail** -- it finishes, and hands back a latency row that nobody
looking at it would query. So every scenario here is run twice, and both halves
are asserted: the row the uncaught run reports, and the report the caught run
raises. A check that has never been seen firing is not a check, and a scenario
that would have crashed anyway never needed one.

The exception is the missing annotation, which is arranged to be loud. A
participant that never declares itself idle cannot be given time, so its clock
cannot move and nothing can pass it: it blocks, for real, until something gives
up. That is asserted as a wall-clock bound, and separately from the several
other ways a test can sit still -- the run has to be shown hung *for the
designed reason*, not merely hung.

The three checks have three different severities on purpose. The receive-side
one raises. The watchdog collects a warning and lets the run carry on. The lint
returns an exit code for CI and touches nothing at runtime.
"""

import importlib.util
import math
import os

import pytest

import atom.compass
from atom.compass.detect import (
    AnnotationWatchdog,
    Arrival,
    CausalityViolation,
    ClockSourceLint,
    StragglerCheck,
    WatchdogWarning,
)
from atom.compass.detect.clock_source import DEFAULT_ALLOW_LIST

from . import scenarios
from .scenarios import DECODE, PREFILL, Injected

#: The row the wrong declaration produces when nothing is checking. It is 0.8 ms
#: out on a 355 ms time to first token -- two parts in a thousand, which is well
#: inside what anyone would accept from a model of anything.
PLAUSIBLE_WITH_A_WRONG_FLOOR = (
    "request-0  handed over at 0.301000s  decode step 0.054000s  "
    "time to first token 0.355000s"
)

#: The row the same run produces once the floor is declared at what the path
#: costs. The difference between this and the one above is the whole of the
#: error, and nothing in either row says which is which.
CORRECT_ROW = (
    "request-0  handed over at 0.300200s  decode step 0.054000s  "
    "time to first token 0.354200s"
)

#: The row an undeclared send produces. How far out it lands is set by how far
#: past the message this receiver happened to be granted -- one decode step --
#: and not by the defect, which is the same defect at every distance. At a
#: shorter distance the row is unremarkable; at this one it is outside the error
#: this project says it will reject a model for, so a reader holding a real
#: measurement beside it would query it -- as a validation failure, with nothing
#: pointing at the cause, which is the expensive way to find this out.
PLAUSIBLE_WITH_AN_UNDECLARED_SEND = (
    "request-0  handed over at 0.354000s  decode step 0.054000s  "
    "time to first token 0.408000s"
)

WRONG_LOOKAHEAD_REPORT = (
    "causality violation: prefill-00.stage-00 -> decode-00.stage-00 took effect "
    "0.0008s into the receiver's past\n"
    "  sent at              0.3s\n"
    "  takes effect at      0.3002s\n"
    "  receiver's clock at  0.301s\n"
    "  declared floor       0.001s\n"
    "the declared floor on prefill-00.stage-00 -> decode-00.stage-00 is 0.001s but "
    "the message took 0.0002s, so the declaration is 0.0008s longer than the path "
    "it stands for. Lower it to 0.0002s or less.\n"
    "Everything decode-00.stage-00 decided after 0.3002s was decided without this "
    "message, so the run stops here rather than reporting numbers taken off it."
)

INDUCED_STRAGGLER_REPORT = (
    "causality violation: prefill-00.stage-00 -> decode-00.stage-00 took effect "
    "0.0538s into the receiver's past\n"
    "  sent at              0.3s\n"
    "  takes effect at      0.3002s\n"
    "  receiver's clock at  0.354s\n"
    "  declared floor       0.0002s\n"
    "the declared floor on prefill-00.stage-00 -> decode-00.stage-00 is 0.0002s and "
    "the message took 0.0002s, so the declaration is not what let this happen. "
    "decode-00.stage-00 was granted past an arrival nothing had declared -- look at "
    "the send path, not at the constant.\n"
    "Everything decode-00.stage-00 decided after 0.3002s was decided without this "
    "message, so the run stops here rather than reporting numbers taken off it."
)

#: A cost backend that charges the time its own arithmetic took. It imports,
#: runs, and returns a number of the right size.
SIMULATED_PATH_WITH_A_REAL_READ = '''# SPDX-License-Identifier: MIT
"""A cost backend that charges a decode step, and its own arithmetic with it."""

import time

DECODE_STEP_SECONDS = 0.054


def step_seconds(rows):
    """What the model says a decode step of this many rows costs."""
    started = time.perf_counter()
    total = DECODE_STEP_SECONDS * rows
    return total + (time.perf_counter() - started)
'''

#: The tree the lint is run over: the package that was actually imported,
#: not a path relative to wherever pytest happened to be started.
SIMULATED_PATH = os.path.dirname(atom.compass.__file__)

_SOURCE_LINES = SIMULATED_PATH_WITH_A_REAL_READ.splitlines()
FIRST_READ_LINE = _SOURCE_LINES.index("    started = time.perf_counter()") + 1
SECOND_READ_LINE = (
    _SOURCE_LINES.index("    return total + (time.perf_counter() - started)") + 1
)


def caught(scenario, **kwargs):
    """Run a scenario with the receive-side check on, and return what it raised."""
    with pytest.raises(CausalityViolation) as raised:
        scenario(StragglerCheck(), **kwargs)
    return raised.value


def split_report(violation):
    """The report, and the participant table appended to it."""
    report, _, table = violation.report.partition("\n\n")
    return report, table


class TestWrongLookaheadOnALink:
    """A floor declared five times longer than the path it stands for."""

    def test_without_the_check_the_run_finishes_and_reports_a_row(self):
        """Nothing raises. The row is wrong and says nothing about being wrong."""
        assert (
            scenarios.wrong_lookahead(StragglerCheck(enabled=False))
            == PLAUSIBLE_WITH_A_WRONG_FLOOR
        )

    def test_the_declared_floor_is_the_only_thing_that_moved_the_number(self):
        """Declare the link at what the path costs and the same run reports 0.8 ms less."""
        assert (
            scenarios.wrong_lookahead(
                StragglerCheck(), role_floor=scenarios.TRANSFER_SECONDS
            )
            == CORRECT_ROW
        )

    def test_the_check_fires_and_names_the_constant_to_change(self):
        report, table = split_report(caught(scenarios.wrong_lookahead))
        assert report == WRONG_LOOKAHEAD_REPORT
        assert table.startswith("participant ")
        assert "decode-00.stage-00   running             0.301s" in table

    def test_the_run_counts_the_violation_and_refuses_to_claim_a_clean_order(self):
        check = StragglerCheck()
        with pytest.raises(CausalityViolation):
            scenarios.wrong_lookahead(check)
        assert (
            check.summary() == "straggler check: 1 message(s) checked, 1 violation(s)"
        )
        assert StragglerCheck(enabled=False).summary() == (
            "straggler check: disabled, so this run makes no claim about its time order"
        )


class TestAnInducedStraggler:
    """Floors that match the path, and a send the coordinator is never told about."""

    def test_without_the_check_the_run_finishes_and_reports_a_row(self):
        assert (
            scenarios.induced_straggler(StragglerCheck(enabled=False))
            == PLAUSIBLE_WITH_AN_UNDECLARED_SEND
        )

    def test_declaring_the_send_is_what_fixes_it(self):
        """The same run, the same floors, the send declared: 54 ms of the row goes."""
        assert (
            scenarios.induced_straggler(StragglerCheck(), tell_the_clock=True)
            == CORRECT_ROW
        )

    def test_the_check_fires_and_says_the_floor_is_not_the_defect(self):
        """Same check, same pair, opposite repair -- and the report distinguishes them."""
        report, _table = split_report(caught(scenarios.induced_straggler))
        assert report == INDUCED_STRAGGLER_REPORT
        assert "look at the send path, not at the constant" in report

    def test_the_two_scenarios_are_told_apart_by_the_report_they_print(self):
        wrong_floor, _ = split_report(caught(scenarios.wrong_lookahead))
        undeclared, _ = split_report(caught(scenarios.induced_straggler))
        assert "longer than the path it stands for" in wrong_floor
        assert "longer than the path it stands for" not in undeclared


@pytest.fixture(scope="module")
def annotated():
    """The declaring run: it reads its message and gets on with it."""
    watchdog = AnnotationWatchdog(stall_seconds=scenarios.STALL_SECONDS)
    return watchdog, scenarios.missing_annotation(watchdog, annotate=True)


@pytest.fixture(scope="module")
def forgetful():
    """The run where the decode side never declares itself idle."""
    watchdog = AnnotationWatchdog(stall_seconds=scenarios.STALL_SECONDS)
    return watchdog, scenarios.missing_annotation(watchdog, annotate=False)


class TestAMissingAnnotation:
    """The mistake the protocol was arranged to make loud."""

    def test_the_forgetful_run_hangs_until_the_bound_and_reports_nothing(
        self, forgetful
    ):
        """No row at all, which is the opposite of the other two scenarios."""
        _watchdog, stall = forgetful
        assert stall.read is None
        assert stall.table == ""
        assert stall.waited_seconds >= scenarios.HANG_BOUND_SECONDS

    def test_the_declaring_run_is_three_orders_of_magnitude_quicker(self, annotated):
        """The control that makes the bound above a reading rather than a timeout."""
        _watchdog, stall = annotated
        assert stall.read is not None
        assert stall.table == CORRECT_ROW
        assert stall.waited_seconds < scenarios.HANG_BOUND_SECONDS / 100

    def test_it_is_hung_for_the_designed_reason_and_not_a_broken_harness(
        self, forgetful, annotated
    ):
        """The control is the reading that excludes a harness which never drove it.

        A participant nobody has touched reads exactly as this one does: still
        running, at the clock it started from. Deliberately not declaring and
        failing to be driven at all are the same state, and no reading of that
        state separates them. What separates them is the same function with the
        declaration put back: it is driven, it reads its message, and it returns
        far inside the bound. The harness can drive this side, so this is not a
        run where it did not.

        The other readings say what kind of stop this is, once it is known to be
        one. The forgetful participant never asked for time, which is the defect
        itself. Its peer is parked one floor ahead and no further and names it
        as what bounds them, so the peer stopped because of this participant and
        not because it ran out of work. And the peer's own next event is still
        ahead of it, so there was work left to do.
        """
        _watchdog, stall = forgetful
        _control_watchdog, control = annotated
        assert control.read is not None
        assert control.waited_seconds < stall.waited_seconds / 100
        assert (stall.decode_status, stall.decode_now) == ("running", 0.0)
        assert stall.prefill_status == "blocked-on-message"
        assert stall.prefill_now == pytest.approx(scenarios.TRANSFER_SECONDS)
        assert stall.bound_from == str(DECODE)
        assert "prefill-00.stage-00  blocked-on-message  0.0002s        0.3s" in (
            stall.lp_table
        )

    def test_the_watchdog_warns_and_the_run_carries_on(self, forgetful):
        watchdog, _stall = forgetful
        assert len(watchdog.warnings) == 1
        warning = watchdog.warnings[0]
        assert warning.lp_id == "decode-00.stage-00"
        assert warning.clock_at == 0.0
        assert warning.running_for_seconds >= scenarios.STALL_SECONDS
        assert "in missing_annotation" in warning.frame
        assert watchdog.summary() == (
            "annotation watchdog: 1 warning(s) on decode-00.stage-00"
        )

    def test_the_declaring_run_draws_no_warning(self, annotated):
        watchdog, _stall = annotated
        assert watchdog.warnings == ()
        assert watchdog.summary() == "annotation watchdog: 0 warning(s)"

    def test_forgetting_the_declaration_can_cost_time_and_can_never_buy_it(self):
        """Why the loud half is structural rather than lucky.

        A grant is the only thing that moves a clock, and asking for one *is*
        the declaration. So a participant that skips it is stuck where it
        stands: omitting a declaration can cost a participant time and can never
        buy it any, and that is the property being asserted here.

        It is a property of omitting a declaration and not of mistakes in
        general. Moving past a message needs a declaration that *is* made, over
        a message the coordinator was not told about -- a second mistake on top
        of a correct annotation, which is the induced straggler above. One
        mistake on its own is silent often enough: the wrong floor in the first
        scenario is a single wrong constant with a correctly declared send, and
        it reports a row.
        """
        run = Injected(scenarios.TRANSFER_SECONDS, start_time=0.0)
        before = run.clock.now(DECODE)
        with pytest.raises(ValueError, match="holds no grant"):
            run.clock.take_up_grant(DECODE)
        assert run.clock.now(DECODE) == before
        assert run.clock.grants_issued(DECODE) == 0


class TestTheClockSourceLint:
    """The static half: a real clock read on the simulated path fails CI."""

    def test_the_simulated_path_is_clean_today(self):
        """This assertion is the gate. It is what breaks on the day a read lands."""
        lint = ClockSourceLint()
        code, report = lint.check(SIMULATED_PATH)
        assert code == 0
        assert report == (
            f"clock-source lint: clean over {len(lint.modules(SIMULATED_PATH))} "
            "module(s), 1 file(s) allow-listed"
        )

    def test_the_watchdog_is_allow_listed_with_the_reason_it_reads_a_real_clock(self):
        """The allow-list is reviewable because every entry carries its reason."""
        reason = DEFAULT_ALLOW_LIST["atom/compass/detect/watchdog.py"]
        assert "measured in simulated ones it would stop" in reason
        assert ClockSourceLint().allowed("/anywhere/atom/compass/detect/watchdog.py")

    def test_the_lint_fires_on_an_injected_read_and_names_every_one(self, tmp_path):
        module = tmp_path / "step.py"
        module.write_text(SIMULATED_PATH_WITH_A_REAL_READ, encoding="utf-8")
        lint = ClockSourceLint()
        code, report = lint.check(str(module))
        assert code == 1
        assert report == "\n".join(
            [
                "clock-source lint: 2 real-clock read(s) on the simulated path:",
                f"  {module}:{FIRST_READ_LINE}  time.perf_counter  in step_seconds",
                f"  {module}:{SECOND_READ_LINE}  time.perf_counter  in step_seconds",
                (
                    "A run charges simulated seconds for the work it models. A "
                    "real read puts machine seconds into the same record, and the "
                    "mixture is reported as one modelled number. Take the time "
                    "from the run's clock, or add the file to the allow-list with "
                    "the reason it stays real."
                ),
            ]
        )

    def test_without_the_lint_the_module_imports_and_returns_a_believable_number(
        self, tmp_path
    ):
        """The uncaught half. Machine seconds in a modelled record, and it reads fine.

        The corrupted step is strictly longer than the modelled one, because
        real time went into it; and it is indistinguishable from the modelled
        one at any precision a latency table prints, because not very much real
        time went into it. Both at once is what makes this class of defect
        expensive.
        """
        module = tmp_path / "step.py"
        module.write_text(SIMULATED_PATH_WITH_A_REAL_READ, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("injected_step", module)
        injected = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(injected)
        charged = injected.step_seconds(1)
        assert charged > injected.DECODE_STEP_SECONDS
        assert charged == pytest.approx(injected.DECODE_STEP_SECONDS, abs=1.0e-3)

    def test_a_read_the_import_renamed_is_still_found(self):
        """`from time import monotonic` is the form a text search walks past."""
        source = "from time import monotonic as clock\n\n\ndef age(since):\n    return clock() - since\n"
        reads = ClockSourceLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == ["engine.py:5  time.monotonic  in age"]

    def test_a_read_an_assignment_renamed_is_still_found(self):
        """The form a text search *would* have found, so the parse has to as well."""
        source = (
            "import time\n\nnow = time.monotonic\n\n\ndef age(since):\n"
            "    return now() - since\n"
        )
        reads = ClockSourceLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == ["engine.py:7  time.monotonic  in age"]

    def test_the_same_clock_in_nanoseconds_is_the_same_read(self):
        """One divide separates these from the seconds they are listed beside."""
        source = (
            "import time\n\n\ndef age(since):\n"
            "    return time.perf_counter_ns() / 1e9 - since\n"
        )
        reads = ClockSourceLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == [
            "engine.py:5  time.perf_counter_ns  in age"
        ]

    def test_the_allow_list_matches_whole_directories_and_not_letters(self):
        """A path that merely ends in the same text is a file nobody reviewed."""
        lint = ClockSourceLint()
        assert lint.allowed("atom/compass/detect/watchdog.py")
        assert lint.allowed("/anywhere/atom/compass/detect/watchdog.py")
        assert lint.allowed("/x/NOTatom/compass/detect/watchdog.py") is None

    def test_the_clean_line_counts_what_this_scan_skipped(self, tmp_path):
        """A tree holding none of the listed files is a tree with no exemptions."""
        (tmp_path / "step.py").write_text("x = 1\n", encoding="utf-8")
        code, report = ClockSourceLint().check(str(tmp_path))
        assert code == 0
        assert report == (
            "clock-source lint: clean over 1 module(s), 0 file(s) allow-listed"
        )

    def test_turning_the_lint_off_makes_the_tree_claim_nothing(self):
        off = ClockSourceLint(enabled=False)
        assert off.scan_source(SIMULATED_PATH_WITH_A_REAL_READ, "step.py") == ()
        assert off.report((), 1) == (
            "clock-source lint: disabled, so this tree makes no claim about its "
            "clock reads"
        )


class TestWhatTheChecksReportOnTheirOwn:
    """The report text, pinned away from the runs that happen to produce it."""

    def test_a_watchdog_warning_says_the_participant_the_clock_and_the_frame(self):
        warning = WatchdogWarning("decode-00.stage-00", 0.0603, 0.0, "q.py:9 in get")
        assert str(warning) == (
            "annotation watchdog: decode-00.stage-00 has been executing for 0.0603s "
            "of real time with its simulated clock still at 0s. Real seconds are "
            "passing inside something nothing declared, and until it is declared "
            "every peer is bounded at 0s plus one floor. Innermost frame: "
            "q.py:9 in get. The run continues."
        )

    def test_the_watchdog_is_quiet_while_the_simulated_clock_is_moving(self):
        """A long step is not a missing declaration; a stopped clock is."""
        ticks = iter([0.0, 10.0, 10.0, 20.0])
        watchdog = AnnotationWatchdog(
            stall_seconds=scenarios.STALL_SECONDS, wall_clock=lambda: next(ticks)
        )
        watchdog.running(PREFILL, 0.0)
        watchdog.running(PREFILL, 0.30)
        assert watchdog.sample() == ()
        assert watchdog.sample()[0].clock_at == 0.30

    def test_a_warning_names_where_the_participant_is_and_not_the_watchdog(self):
        """Sampled from the thread it is reporting on, the top of the stack is ours.

        A sampler on its own thread never sees this; one called straight from
        the participant's thread sees nothing else, and the most actionable
        field in the warning would name the detector instead of the wait.
        """
        ticks = iter([0.0, 10.0])
        watchdog = AnnotationWatchdog(
            stall_seconds=scenarios.STALL_SECONDS, wall_clock=lambda: next(ticks)
        )
        watchdog.running(PREFILL, 0.0)
        frame = watchdog.sample()[0].frame.replace(os.sep, "/")
        assert "compass/detect/watchdog.py" not in frame
        assert "in test_a_warning_names_where_the_participant_is" in frame

    def test_the_floor_is_judged_against_the_round_off_the_clock_produces(self):
        """A tight floor on a long run, where the difference is entirely round-off.

        Both stamps come off a clock hundreds of seconds in, and recovering the
        delay from them loses the last bits of it. What is left is far below the
        resolution of those stamps and far above any fraction of a microsecond
        floor, so a tolerance sized against the floor alone reads it as a
        constant to lower -- which is the misdiagnosis the tolerance exists to
        prevent. The other end of the scale is the induced straggler above,
        whose floor is sized at its path on a sub-second clock.
        """
        floor = 1.0e-6
        for clock in (100.0, 300.0):
            sent = clock - 1.0
            arrival = Arrival("a", "b", sent, sent + floor, floor)
            assert arrival.observed_delay_seconds < floor
            assert "look at the send path, not at the constant" in (
                StragglerCheck.report(arrival, clock)
            )
        blunt = Arrival("a", "b", 299.0, 299.0 + floor, 2.0 * floor)
        assert "longer than the path it stands for" in (
            StragglerCheck.report(blunt, 300.0)
        )

    def test_stamps_in_the_wrong_order_are_refused_where_they_are_made(self):
        """A negative delay has no floor to lower, so it never reaches the report.

        Left to the check, it produces a repair nobody can carry out -- lower
        the floor to a negative number -- while the defect that produced it goes
        unnamed.
        """
        with pytest.raises(ValueError) as refused:
            Arrival("a", "b", 1.0, 0.95, 1.0e-3)
        assert "so its stamps are wrong" in str(refused.value)
        assert "no floor can be declared at a negative delay" in str(refused.value)
        assert Arrival("a", "b", 1.0, 1.0, 0.0).observed_delay_seconds == 0.0

    def test_the_context_is_built_for_the_message_that_fails_and_no_other(self):
        """The check is one comparison; its context is a table over every participant.

        A diagnostic built on every message costs orders of magnitude more than
        the check it decorates, which is how an always-on check stops being on.
        """
        built = []

        def context():
            built.append(None)
            return "participant          status"

        check = StragglerCheck()
        check.arriving(Arrival("a", "b", 1.0, 1.001, 1.0e-3), 1.0, context)
        assert built == []
        with pytest.raises(CausalityViolation) as raised:
            check.arriving(Arrival("a", "b", 1.0, 1.001, 1.0e-3), 1.002, context)
        assert len(built) == 1
        assert raised.value.report.endswith("\n\nparticipant          status")

    def test_a_message_that_arrives_on_time_is_counted_and_passed_through(self):
        check = StragglerCheck()
        arrival = Arrival("a", "b", 1.0, 1.001, 1.0e-3)
        assert check.arriving(arrival, 1.0) is arrival
        assert (
            check.summary() == "straggler check: 1 message(s) checked, 0 violation(s)"
        )

    def test_a_message_that_arrives_at_the_clock_exactly_is_not_a_violation(self):
        """The boundary is the whole check, so it is stated rather than implied."""
        check = StragglerCheck()
        assert check.arriving(Arrival("a", "b", 1.0, 1.001, 1.0e-3), 1.001)
        assert check.violations == 0

    def test_the_scenarios_declare_a_floor_the_matrix_accepts_for_every_pair(self):
        """A pair with no floor would raise the bound rather than lower it."""
        run = Injected(scenarios.TRANSFER_SECONDS)
        assert run.matrix.undeclared() == ()
        assert run.clock.grant_bound(DECODE) < math.inf
