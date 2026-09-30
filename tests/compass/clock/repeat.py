# SPDX-License-Identifier: MIT
"""One configuration run and recorded in this process, printed for a caller to diff.

The child half of the determinism check: `python -m clock.repeat CONFIGURATION`
from `tests/compass`. It has to be a child, because two runs inside one process
hash strings from one seed, and agree with each other whatever order a `set`
hands them. The step table goes to stdout and nothing else does. stderr carries
the hash of `SEED_PROBE`, which shows the caller the two seeds differed, and the
`atom` the child imported.

Step rows come from the Clock Authority's timeline hook, one per reply; message
rows from the driver's hand-over, one per message, with its payload as detail.

`--hand-over-by-set` is the seeded defect: messages that arrive at one instant
are handed to their LP in the order a `set` iterates them, not in `(arrival,
channel, seq)` order. `--sleep-seconds` spends real time on every reply.
"""

import argparse
import dataclasses
import sys
import time

import atom
from atom.compass.detect.determinism import StepTable

from .harness import SyntheticRun
from .participants import DESIGN_WORKLOAD

SEED_PROBE = "engine-P"

#: Equal prompt lengths, so a pair tokenized side by side reaches the engine at
#: one arrival and the hand-over order decides which request it serves first.
WORKLOAD = dataclasses.replace(
    DESIGN_WORKLOAD, requests=16, decode_steps=3, prompt_tokens=(2048, 2048)
)

CONFIGURATIONS = {"M1-M3": "single-deployment", "M4": "prefill-decode-1p1d"}


class RecordedRun(SyntheticRun):
    """A synthetic run that writes its step table, and optionally misbehaves."""

    def __init__(
        self, configuration, order=None, hand_over_by_set=False, sleep_seconds=0.0
    ):
        super().__init__(CONFIGURATIONS[configuration], WORKLOAD, order)
        self.step_table = StepTable(configuration)
        self.clock.timeline = self
        self.hand_over_by_set = hand_over_by_set
        self.sleep_seconds = sleep_seconds

    def record(self, lp, time_from, time_to, kind, recovered):
        """The authority's timeline hook: one step row per reply."""
        detail = "recovered" if recovered else ""
        self.step_table.record(lp, time_from, time_to, kind, detail=detail)
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)

    def _hand_over(self, lp, g, released):
        messages = super()._hand_over(lp, g, released)
        if self.hand_over_by_set:
            ties = list({m[:3] for m in messages})
            messages.sort(key=lambda m: (m[0], ties.index(m[:3])))
        for a, channel, seq, payload in messages:
            self.step_table.record(lp, a, a, "receive", channel, seq, payload)
        return messages


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Record one synthetic run.")
    parser.add_argument("configuration", choices=sorted(CONFIGURATIONS))
    parser.add_argument("--hand-over-by-set", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    arguments = parser.parse_args(argv)
    run = RecordedRun(
        arguments.configuration,
        hand_over_by_set=arguments.hand_over_by_set,
        sleep_seconds=arguments.sleep_seconds,
    )
    report = run.run()
    sys.stdout.write(run.step_table.text())
    sys.stderr.write(f"hash-probe {hash(SEED_PROBE)}\natom {atom.__file__}\n")
    return 0 if report.stopped_by == "END from traffic" else 1


if __name__ == "__main__":
    sys.exit(main())
