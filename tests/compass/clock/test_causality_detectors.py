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

#: The row an undeclared send produces: 54 ms out, and still unremarkable.
PLAUSIBLE_WITH_AN_UNDECLARED_SEND = (
    "request-0  handed over at 0.354000s  decode step 0.054000s  "
    "time to first token 0.408000s"
)

WRONG_LOOKAHEAD_REPORT = (
    "causality violation: prefill-00.stage-0 -> decode-00.stage-0 took effect "
    "0.0008s into the receiver's past\n"
    "  sent at              0.3s\n"
    "  takes effect at      0.3002s\n"
    "  receiver's clock at  0.301s\n"
    "  declared floor       0.001s\n"
    "the declared floor on prefill-00.stage-0 -> decode-00.stage-0 is 0.001s but "
    "the message took 0.0002s, so the declaration is 0.0008s longer than the path "
    "it stands for. Lower it to 0.0002s or less.\n"
    "Everything decode-00.stage-0 decided after 0.3002s was decided without this "
    "message, so the run stops here rather than reporting numbers taken off it."
)

INDUCED_STRAGGLER_REPORT = (
    "causality violation: prefill-00.stage-0 -> decode-00.stage-0 took effect "
    "0.0538s into the receiver's past\n"
    "  sent at              0.3s\n"
    "  takes effect at      0.3002s\n"
    "  receiver's clock at  0.354s\n"
    "  declared floor       0.0002s\n"
    "the declared floor on prefill-00.stage-0 -> decode-00.stage-0 is 0.0002s and "
    "the message took 0.0002s, so the declaration is not what let this happen. "
    "decode-00.stage-0 was granted past an arrival nothing had declared -- look at "
    "the send path, not at the constant.\n"
    "Everything decode-00.stage-0 decided after 0.3002s was decided without this "
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
        assert "decode-00.stage-0    running             0.301s" in table

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
        self, forgetful
    ):
        """Four readings, each ruling out a different way of sitting still.

        The forgetful participant is still executing at the clock it started
        from, so it never asked for time -- a harness that simply failed to
        drive it would have left it parked. Its peer is parked one floor ahead
        and no further, and names it as what bounds them, so the peer stopped
        because of this participant rather than because it ran out of work. And
        the peer's own next event is still ahead of it, so there was work left
        to do.
        """
        _watchdog, stall = forgetful
        assert (stall.decode_status, stall.decode_now) == ("running", 0.0)
        assert stall.prefill_status == "blocked-on-message"
        assert stall.prefill_now == pytest.approx(scenarios.TRANSFER_SECONDS)
        assert stall.bound_from == str(DECODE)
        assert "prefill-00.stage-0   blocked-on-message  0.0002s        0.3s" in (
            stall.lp_table
        )

    def test_the_watchdog_warns_and_the_run_carries_on(self, forgetful):
        watchdog, _stall = forgetful
        assert len(watchdog.warnings) == 1
        warning = watchdog.warnings[0]
        assert warning.lp_id == "decode-00.stage-0"
        assert warning.clock_at == 0.0
        assert warning.running_for_seconds >= scenarios.STALL_SECONDS
        assert "in missing_annotation" in warning.frame
        assert watchdog.summary() == (
            "annotation watchdog: 1 warning(s) on decode-00.stage-0"
        )

    def test_the_declaring_run_draws_no_warning(self, annotated):
        watchdog, _stall = annotated
        assert watchdog.warnings == ()
        assert watchdog.summary() == "annotation watchdog: 0 warning(s)"

    def test_forgetting_the_declaration_can_cost_time_and_can_never_buy_it(self):
        """Why the loud half is structural rather than lucky.

        A grant is the only thing that moves a clock, and asking for one *is*
        the declaration. So a participant that skips it is stuck where it
        stands. The silent failure -- moving past a message -- needs the
        opposite: a declaration that is made while a message nothing declared is
        already on the transport. That is a second mistake on top of a correct
        annotation, and it is the induced straggler above, not this.
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
        warning = WatchdogWarning("decode-00.stage-0", 0.0603, 0.0, "q.py:9 in get")
        assert str(warning) == (
            "annotation watchdog: decode-00.stage-0 has been executing for 0.0603s "
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
