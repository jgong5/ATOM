"""The engine-side half of simulated timing.

ATOM runs its workers in separate processes even at world size one, so a
simulated runner cannot advance time itself — it reports a predicted duration
and the scheduling process moves its own clock. These tests cover that handoff
and, importantly, that a normal run is unaffected by any of it.
"""

import pytest

from atom.model_engine.engine_core import (
    _advance_clock_for,
    _defers_output,
)
from atom.model_engine.scheduler import ScheduledBatchOutput
from atom.utils.clock import (
    VirtualClock,
    WallClock,
    get_clock,
    now,
    reset_clock,
    set_clock,
)


def _output(seconds=None):
    return ScheduledBatchOutput(
        req_ids=[1],
        token_ids=[(5,)],
        num_rejected=None,
        num_bonus=None,
        draft_token_ids=None,
        compass_step_seconds=seconds,
    )


@pytest.fixture(autouse=True)
def _restore_default_clock():
    yield
    reset_clock()


def test_a_real_step_carries_no_duration():
    assert _output().compass_step_seconds is None


def test_advancing_is_a_no_op_on_the_wall_clock():
    reset_clock()
    _advance_clock_for(_output(0.25))
    assert isinstance(get_clock(), WallClock)


def test_a_real_step_does_not_move_a_virtual_clock():
    set_clock(VirtualClock())
    before = now()
    _advance_clock_for(_output(None))
    assert now() == before


def test_a_simulated_step_moves_the_clock_by_its_prediction():
    set_clock(VirtualClock())
    before = now()
    _advance_clock_for(_output(0.25))
    assert now() - before == pytest.approx(0.25)


def test_durations_accumulate_without_spending_wall_time():
    clock = VirtualClock()
    set_clock(clock)
    for _ in range(4):
        _advance_clock_for(_output(0.25))
    # A second of modelled serving, in no time at all. This is the whole
    # reason a simulated run is faster than the run it stands in for.
    assert clock.elapsed == pytest.approx(1.0)


class TestADeferredOutputIsChargedAfterItIsDrained:
    """Every step's output carries the *previous* step's tokens.

    Witnessed on the 27B short cell: 130/130 real and 129/129 modelled
    `step_output` events were `deferred: true`. Charging the current step's
    cost before `postprocess` therefore publishes the previous step's tokens
    at this step's completion -- one whole step late. On the real engine the
    CPU launches step N+1 while the device is still finishing step N (measured
    gap 0.259s on that cell), so the wall clock at drain time already reads the
    producing step's completion and the ordering never shows. A virtual clock
    moves in whole steps and has no such head start.

    Measured consequence on 27B TP=1 short, before the fix: simulated TPOT
    median 0.0343s against 0.0681s real, while total latency stayed exact.
    Time moved between metrics rather than being lost, which is why a summed
    step-cost check could not see it.
    """

    def test_a_deferred_output_is_recognised(self):
        out = _output(0.25)
        out.is_deferred_out = True
        assert _defers_output(out) is True

    def test_an_undeferred_output_is_not(self):
        assert _defers_output(_output(0.25)) is False

    def test_an_output_without_the_field_is_not_deferred(self):
        """Third-party or older outputs must not silently defer."""
        assert _defers_output(object()) is False

    def test_the_clock_never_runs_backwards_over_a_deferred_loop(self):
        """Deferral changes when a cost is charged, not its sign.

        The loop below is the engine's order: stamp, forward, drain, charge.
        Every instant a token could be stamped at is read, and the sequence has
        to be non-decreasing -- a simulated run whose clock went backwards
        would produce negative durations rather than wrong ones.
        """
        set_clock(VirtualClock())
        seen = [now()]
        for cost in (7.005, 4.982, 4.469, 0.034, 0.034, 0.034):
            out = _output(cost)
            out.is_deferred_out = True
            seen.append(now())  # where postprocess stamps this drain
            _advance_clock_for(out)
            seen.append(now())  # where the next step starts
        assert seen == sorted(seen)
        # `now()` is the run epoch plus elapsed, so elapsed is the difference.
        assert now() - seen[0] == pytest.approx(
            sum((7.005, 4.982, 4.469, 0.034, 0.034, 0.034))
        )

    def test_a_deferred_drain_reads_the_producing_steps_completion(self):
        """The property the fix exists for, stated as an instant.

        Step N's tokens arrive in step N+1's output. They must be stamped at
        the completion of step N -- which is the clock's value while step N+1
        is in flight -- and not at step N+1's.
        """
        set_clock(VirtualClock())
        costs, drains = (7.005, 4.982, 4.469), []
        for cost in costs:
            out = _output(cost)
            out.is_deferred_out = True
            drains.append(now())
            _advance_clock_for(out)
        # The drain during step 1 reads step 0's completion, and so on,
        # measured from the run epoch `now()` is offset by.
        assert drains[1] - drains[0] == pytest.approx(costs[0])
        assert drains[2] - drains[0] == pytest.approx(costs[0] + costs[1])

    def test_charging_it_still_moves_the_clock_by_its_own_prediction(self):
        """Deferral moves *when* the cost is charged, never how much."""
        set_clock(VirtualClock())
        out = _output(0.25)
        out.is_deferred_out = True
        before = now()
        _advance_clock_for(out)
        assert now() - before == pytest.approx(0.25)


class TestTheEngineLoopChargesItInThatOrder:
    """Asserted on the source, because the loop needs a worker to run.

    Both sites are checked: the main step and the decode-scheduler override.
    """

    @staticmethod
    def _sites():
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "atom/model_engine/engine_core.py"
        ).read_text()
        tree = ast.parse(source)
        found = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            deferred, plain, drains = [], [], []
            for node in ast.walk(fn):
                if isinstance(node, ast.If):
                    body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
                    if "_advance_clock_for" not in body:
                        continue
                    test = ast.dump(node.test)
                    if "_defers_output" not in test:
                        continue
                    (plain if "Not()" in test else deferred).append(node.lineno)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "postprocess"
                ):
                    drains.append(node.lineno)
            if deferred or plain:
                found.append((fn.name, deferred, plain, drains))
        return found

    def test_both_call_sites_are_guarded(self):
        assert len(self._sites()) == 2, self._sites()

    def test_the_deferred_charge_comes_after_the_drain(self):
        for name, deferred, _plain, drains in self._sites():
            assert deferred and drains, name
            assert min(deferred) > max(drains), name

    def test_the_undeferred_charge_still_comes_before_it(self):
        """A step whose own tokens are in its own output is unchanged."""
        for name, _deferred, plain, drains in self._sites():
            assert plain and drains, name
            assert max(plain) < min(drains), name
