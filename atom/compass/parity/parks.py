# SPDX-License-Identifier: MIT
"""When a recorded P/D run's scheduler parks each request for remote KV.

A decode instance's scheduler parks a request it admits for a remote KV load,
`WAITING_FOR_REMOTE_KVS` (`Scheduler._park_for_remote_load`, or a running
partial prefill `should_park_partial_prefill_for_load` takes back), counting it
once in `Scheduler._count_inflight_load` until its KV lands or it is dropped.
Nothing stamped that moment: a write request's own stamps (`parity.writes`)
start at ``t_load_ns``, when a worker took the step's metadata that asked for
it, and a park's wait before that was only inferred (#3720 cell B: c256's
~13.6 ms between decode parking a request and prefill receiving its write
request).

On a real recording `install`, called in the engine's process
(`run.engine`), wraps `Scheduler._count_inflight_load` and writes a line per
park into ``dp<rank>.parks.jsonl`` of the record directory: ``request_id``,
the sequence id a consumer's write line names under ``request_id``;
``dp_rank``; and ``t_park_ns``, when the scheduler counted the request, on the
engine's monotonic clock (the clock of the step and write stamps). A request
parked again after its count was dropped gets another line. Each line is
written through, a few microseconds a park, so an engine that is killed keeps
its lines. ATOM's code is unchanged, and a run that does not record wraps
nothing. `read` gives each rank's lines. No reader of a record's other files
reads this one: ``dp<rank>.parks.jsonl`` matches none of their globs
(``dp*[0-9].jsonl``, ``dp*.handoff.jsonl``, ``dp*.engine.json``).
"""

import functools
import json
from pathlib import Path

#: The file a rank's parks go to, in the record directory.
FILE = "dp{rank}.parks.jsonl"
#: A park line's fields.
FIELDS = ("request_id", "dp_rank", "t_park_ns")


def install(path, rank: int, now) -> None:
    """Write each first count of a request as parked for remote KV into
    `path`, rank `rank`'s, stamped off the clock `now`. A second install in
    one process (a test's) wraps ATOM's method again, into its own `path`."""
    from atom.model_engine.scheduler import Scheduler

    count = Scheduler._count_inflight_load
    count = getattr(count, "parks_wrapped", count)
    out = open(path, "w", buffering=1)  # noqa: SIM115 - kept open for the engine's life

    @functools.wraps(count)
    def counted(self, seq) -> None:
        t = now()
        first = not getattr(seq, "_counted_as_inflight_load", False)
        count(self, seq)
        if first and getattr(seq, "_counted_as_inflight_load", False):
            line = {"request_id": seq.id, "dp_rank": rank, "t_park_ns": t}
            out.write(json.dumps(line) + "\n")

    counted.parks_wrapped = count
    Scheduler._count_inflight_load = counted


def read(directory) -> dict[int, list[dict]]:
    """Each rank's park lines in the record directory `directory`, in order."""
    out = {}
    for path in sorted(Path(directory).glob(FILE.format(rank="*"))):
        rank = int(path.name.split(".")[0][2:])
        out[rank] = [json.loads(line) for line in path.read_text().splitlines() if line]
    return out
