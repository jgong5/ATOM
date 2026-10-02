# SPDX-License-Identifier: MIT
"""Real-clock reads on ATOM's serving path: each one takes the LP clock or is listed by site."""

import os

import pytest

import atom
import atom.model_engine.scheduler
from atom.compass.detect.clock_source import (
    SERVING_ALLOW_LIST,
    SERVING_ROOTS,
    ClockSourceLint,
)
from atom.entrypoints.openai.metrics import AtomMetricsExporter
from atom.entrypoints.openai.protocol import ModelCard
from atom.utils import clock

REPO = os.path.dirname(os.path.dirname(atom.__file__))


def _serving_reads():
    lint = ClockSourceLint()
    for root in SERVING_ROOTS:
        for path in lint.modules(os.path.join(REPO, root)):
            with open(path, encoding="utf-8") as handle:
                yield from lint.scan_source(handle.read(), path)


class _Runtime:
    def __init__(self, t):
        self.t = t

    def read_clock(self):
        return self.t


def _no_real_read():
    raise AssertionError("a simulated run read the machine clock")


@pytest.fixture
def lp_clock():
    clock.install(_Runtime(7.25))
    yield 7.25
    clock.install(None)


class TestTheServingPathGate:
    @pytest.mark.parametrize("root", SERVING_ROOTS)
    def test_no_unlisted_real_read_is_left(self, root):
        """The set of reads still waiting for substitution is empty."""
        code, report = ClockSourceLint().check(os.path.join(REPO, root))
        assert code == 0, report

    def test_a_seeded_read_in_the_scheduler_fails(self, tmp_path):
        with open(atom.model_engine.scheduler.__file__, encoding="utf-8") as handle:
            source = handle.read()
        seeded = source + "\n\ndef _seeded():\n    return time.time()\n"
        target = tmp_path / "atom" / "model_engine" / "scheduler.py"
        target.parent.mkdir(parents=True)
        target.write_text(seeded, encoding="utf-8")
        code, report = ClockSourceLint().check(str(tmp_path))
        line = len(seeded.splitlines())
        assert code == 1
        assert report.splitlines()[:2] == [
            "clock-source lint: 1 real-clock read(s) on the simulated path:",
            f"  {target}:{line}  time.time  in _seeded",
        ]

    def test_every_entry_is_a_kept_class_with_a_reason(self):
        for site, (kind, why) in SERVING_ALLOW_LIST.items():
            assert kind in ("K8", "K9"), site
            assert why.strip(), site

    def test_every_entry_keeps_a_read_at_the_tip(self):
        """An entry whose read moved or went away excuses nothing and still reads as a decision."""
        reads = list(_serving_reads())
        for site in SERVING_ALLOW_LIST:
            one = ClockSourceLint(sites={site: ("", "")})
            assert any(one.site(read) is not None for read in reads), site


class TestASubstitutedReadTakesTheLpClock:
    def test_the_metrics_refresh_stamp(self, lp_clock):
        exporter = AtomMetricsExporter()
        exporter.update({})
        assert exporter.read()[2] == lp_clock

    def test_the_model_card_stamp(self, lp_clock):
        assert ModelCard(id="m").created == int(lp_clock)

    def test_the_real_clock_is_never_called(self, lp_clock):
        assert clock.now(_no_real_read) == lp_clock

    def test_a_real_run_reads_the_clock_it_passed(self):
        assert clock.now(lambda: 3.5) == 3.5
