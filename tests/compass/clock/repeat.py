# SPDX-License-Identifier: MIT
"""One run of one configuration, recorded and printed, for a caller to diff.

This is the child half of the determinism check. It has to be a child: the
order a set comes out in is decided by a seed the interpreter picks when it
starts, so two runs inside one process share that seed and agree with each
other whatever the code between them does. A check built that way passes
against the exact defect it exists to catch. Two processes with two seeds is
the whole mechanism, and everything here is arranged so a caller can start one.

The record goes to stdout and nothing else does, so the caller can compare the
streams byte for byte. What the run cost in real seconds is not printed at all;
the caller times the process if it wants that, and no part of it reaches the
record.

**The seed is printed too, on stderr, as the hash of one participant's name.**
That number is what makes the comparison mean something: two records that agree
prove nothing unless the two processes really were hashing differently, and
this is the evidence that they were.

Two of the flags exist to be turned on, not to be used:

* `--tie-break-by-set` picks the order participants are served in out of a set
  instead of sorting it. The run still serves the furthest-behind participant
  last, so it costs what it always cost and finishes with the same number of
  steps -- only the order of everything else moves, which is what makes it the
  right shape of defect to test against rather than an obvious break.
* `--sleep-seconds` spends real seconds inside the run, which answers by
  measurement whether real time can reach the record at all.
"""

import argparse
import sys
import time

from atom.compass.detect.determinism import StepTable

from .deployments import DEPLOYMENTS
from .harness import SyntheticRun
from .participants import DESIGN_WORKLOAD, scaled

#: The participant whose name is hashed to show the caller which seed this
#: process got. Any string would do; a name the run actually holds is clearer.
SEED_PROBE = "decode-00.stage-0"

#: The configurations this check runs. Neither is the arrangement the detectors
#: beside it were written against: the first is pipelined and split across the
#: prefill/decode boundary, which is six participants and the tightest floor in
#: the tree; the second is a single engine, which is two.
CONFIGURATIONS = {
    "tp8-pp4-four-requests": ("tp8-pp4", 4, 6),
    "tp4-one-server-eight-requests": ("tp4-one-server", 8, 10),
}


class _RecordingClock:
    """The clock, with every grant it hands over written down on the way past."""

    def __init__(self, clock, table, sleep_seconds):
        self._clock = clock
        self._table = table
        self._sleep_seconds = sleep_seconds

    def __getattr__(self, name):
        return getattr(self._clock, name)

    def take_up_grant(self, lp_id):
        grant = self._clock.take_up_grant(lp_id)
        self._table.record(
            grant.lp_id,
            grant.advance_from,
            grant.advance_to,
            "grant",
            grant.bound_from if grant.bound_from is not None else "unbounded",
        )
        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
        return grant


class RecordedRun(SyntheticRun):
    """A synthetic run that writes down what moved, and optionally misbehaves."""

    def __init__(
        self, name, deployment, workload, tie_break_by_set=False, sleep_seconds=0.0
    ):
        super().__init__(deployment, workload)
        self.table = StepTable(name)
        self.tie_break_by_set = tie_break_by_set
        self.clock = _RecordingClock(self.clock, self.table, sleep_seconds)

    def send(self, source, target, when, payload):
        """Record the send beside the grants, then let the run make it."""
        self.table.record(source, self.clock.now(source), when, "send", target)
        super().send(source, target, when, payload)

    def _service_order(self):
        """Who is offered a turn, and in what order.

        The sort is stable, so with `tie_break_by_set` on, participants whose
        clocks stand at the same time keep the order the set gave them -- which
        at the start of a run is every one of them.
        """
        if not self.tie_break_by_set:
            return super()._service_order()
        return sorted(set(self.ids), key=lambda lp_id: -self.clock.now(lp_id))


def record(name, tie_break_by_set=False, sleep_seconds=0.0) -> StepTable:
    """Run one named configuration and hand back what it did."""
    deployment_name, requests, decode_steps = CONFIGURATIONS[name]
    deployment = next(one for one in DEPLOYMENTS if one.name == deployment_name)
    workload = scaled(DESIGN_WORKLOAD, requests, decode_steps)
    run = RecordedRun(name, deployment, workload, tie_break_by_set, sleep_seconds)
    run.run()
    return run.table


def main(argv=None) -> int:
    """Record one run, print it on stdout, and print this process's seed on stderr."""
    parser = argparse.ArgumentParser(description="Record one simulated run.")
    parser.add_argument("configuration", choices=sorted(CONFIGURATIONS))
    parser.add_argument("--tie-break-by-set", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    arguments = parser.parse_args(argv)
    table = record(
        arguments.configuration,
        arguments.tie_break_by_set,
        arguments.sleep_seconds,
    )
    sys.stdout.write(table.text())
    sys.stderr.write(f"hash-probe {hash(SEED_PROBE)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
