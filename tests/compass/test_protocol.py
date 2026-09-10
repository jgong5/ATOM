"""The protocol registration, and the one engine property preparation relies on.

`atom/compass/PROTOCOL.md` is only worth writing if a result can be tied to the
version of it that was in force when the result was produced. That is what
these cover: registering, detecting an edit, and stamping a cell with the
digest it actually ran under.

The last class covers something else, and it is the reason preparation is
shaped the way it is. The scheduler's arrival barrier latches open the first
time a declared workload fully arrives, and never re-arms. A preparation batch
that declared itself would spend the latch, and the measured workload would
then run unheld -- which is the bug the barrier exists for. So preparation is
sent undeclared, and that is asserted here against the scheduler's own method
rather than against a comment in the client.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load("protocol")


@pytest.fixture
def registry(tmp_path, monkeypatch):
    doc = tmp_path / "PROTOCOL.md"
    doc.write_text("prepare, then measure\n")
    monkeypatch.setattr(protocol, "PROTOCOL", doc)
    monkeypatch.setattr(protocol, "LOCK", tmp_path / "protocol.lock.json")
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    return doc


class TestRegistration:
    def test_registering_records_the_digest_and_the_instant(self, registry):
        assert protocol.register("2026-09-10T15:00:00Z") == 0
        lock = json.loads(protocol.LOCK.read_text())
        assert lock["sha256"] == protocol.digest(registry)
        assert lock["registered_at"] == "2026-09-10T15:00:00Z"
        assert lock["superseded"] is None

    def test_registering_the_same_text_twice_keeps_the_first_instant(
            self, registry):
        protocol.register("2026-09-10T15:00:00Z")
        protocol.register("2026-09-11T09:00:00Z")
        lock = json.loads(protocol.LOCK.read_text())
        assert lock["registered_at"] == "2026-09-10T15:00:00Z"

    def test_a_new_registration_keeps_the_one_it_replaces(self, registry):
        protocol.register("2026-09-10T15:00:00Z")
        first = json.loads(protocol.LOCK.read_text())["sha256"]
        registry.write_text("prepare, then measure, and say so\n")
        protocol.register("2026-09-11T09:00:00Z")
        lock = json.loads(protocol.LOCK.read_text())
        assert lock["sha256"] != first
        assert lock["superseded"]["sha256"] == first

    def test_the_instant_is_given_not_read_from_the_clock(self, registry):
        """A re-registration cannot be backdated by a machine whose clock
        disagrees, because the caller states the instant it means."""
        with pytest.raises(SystemExit):
            protocol.main(["register"])


class TestVerification:
    def test_an_unregistered_protocol_is_not_a_pass(self, registry, capsys):
        assert protocol.verify() == 2
        assert "ATOMCompass WARNING:" in capsys.readouterr().out

    def test_an_edit_after_registration_is_caught(self, registry, capsys):
        protocol.register("2026-09-10T15:00:00Z")
        registry.write_text("measure, then prepare\n")
        assert protocol.verify() == 1
        assert "has changed since it was" in capsys.readouterr().out

    def test_an_unchanged_protocol_verifies(self, registry):
        protocol.register("2026-09-10T15:00:00Z")
        assert protocol.verify() == 0


class TestStamping:
    def test_a_cell_carries_the_digest_it_ran_under(self, registry, tmp_path):
        protocol.register("2026-09-10T15:00:00Z")
        cell = tmp_path / "cell"
        assert protocol.stamp(str(cell)) == 0
        stamped = json.loads((cell / "protocol.json").read_text())
        assert stamped["matches_registration"] is True
        assert stamped["registered_at"] == "2026-09-10T15:00:00Z"

    def test_a_cell_run_under_an_edited_protocol_says_so(self, registry,
                                                         tmp_path):
        """The point of stamping. An edit cannot be made to apply backwards:
        the cell records what it ran under, not what is registered now."""
        protocol.register("2026-09-10T15:00:00Z")
        registry.write_text("measure, then prepare\n")
        cell = tmp_path / "cell"
        assert protocol.stamp(str(cell)) == 1
        stamped = json.loads((cell / "protocol.json").read_text())
        assert stamped["matches_registration"] is False
        assert stamped["sha256"] != stamped["registered_sha256"]


class TestPreparationDoesNotSpendTheArrivalBarrier:
    """Asserted against `Scheduler._arrival_barrier_unmet` itself.

    The method is called unbound on a stand-in carrying only the state it
    reads, so the test exercises the engine's logic without building an engine.
    """

    @staticmethod
    def _barrier(waiting):
        from atom.model_engine.scheduler import Scheduler
        from atom.utils.clock import VirtualClock, set_clock

        set_clock(VirtualClock())

        class Stand:
            _arrival_barrier_open = False
            _arrival_barrier_since = None
            ARRIVAL_BARRIER_TIMEOUT_S = Scheduler.ARRIVAL_BARRIER_TIMEOUT_S

        stand = Stand()
        stand.waiting = waiting
        return stand, Scheduler._arrival_barrier_unmet(stand)

    @staticmethod
    def _seq(declared=None):
        class Seq:
            pass

        seq = Seq()
        if declared is not None:
            seq.compass_workload_size = declared
        return seq

    def test_an_undeclared_batch_holds_nothing_and_leaves_the_latch_armed(self):
        stand, unmet = self._barrier([self._seq(), self._seq()])
        assert unmet is False
        assert stand._arrival_barrier_open is False

    def test_a_declared_batch_is_held_until_it_has_all_arrived(self):
        stand, unmet = self._barrier([self._seq(declared=4)])
        assert unmet is True
        assert stand._arrival_barrier_open is False

    def test_a_declared_batch_that_has_arrived_spends_the_latch(self):
        """Which is why preparation must not declare itself: this is
        one-way."""
        stand, unmet = self._barrier([self._seq(declared=2), self._seq()])
        assert unmet is False
        assert stand._arrival_barrier_open is True
