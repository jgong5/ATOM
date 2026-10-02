# SPDX-License-Identifier: MIT
"""The frontend LP's tokenizer station: thread-pool jobs on the simulated clock.

Tokenization runs for real, for its result, and is charged a modelled duration
for its time. A `Station` is a multi-server FIFO over that charged time: job k,
submitted at ``t_k``, starts at ``s_k = max(t_k, earliest free server)`` and
completes at ``c_k = s_k + its charged service``. Jobs are placed in submission
order, whatever order their real threads finish in.

`SimExecutor` is the event loop's default executor. From `SimExecutor.start_run`
on, `classify` sorts every job by name: ``do_preprocess`` is a service job on
the station; ``Queue.get`` is a wait job passed straight through, since a
channel event completes it and zero simulated time is right; anything else is
refused, so a job ATOM later adds to the pool is reported, not timed as free.
A service job's result reaches its coroutine at ``c_k``, through ``call_at``.

A thread serving a station job reads the clock as the job's start plus the
service charged so far (`atom.utils.clock.current_job_time`).
"""

import heapq
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

from atom.utils.clock import DIAG_S, job

logger = logging.getLogger("atom")


class Refused(Exception):
    """An operation the simulation declines; ``args[0]`` is ``(source, name)``."""


class Station:
    """A FIFO with `width` servers, over the service time its jobs are charged."""

    def __init__(self, width: int, diag_s: float = DIAG_S) -> None:
        if width < 1:
            raise ValueError(f"a station needs at least one server, got {width}")
        self.diag_s = diag_s
        self.cv = threading.Condition()
        # One row per job: [submitted, charged, done, start, completion].
        self.jobs: list[list] = []
        self.free = [float("-inf")] * width  # min-heap of server free times
        self.final = 0  # jobs[:final] are placed

    def enqueue(self, t: float) -> int:
        with self.cv:
            self.jobs.append([t, 0.0, False, None, None])
            return len(self.jobs) - 1

    def charge(self, k: int, d: float) -> None:
        with self.cv:
            self.jobs[k][1] += d

    def finish(self, k: int) -> range:
        """Job k is done; returns the jobs this placed, in submission order."""
        with self.cv:
            self.jobs[k][2] = True
            first = self.final
            while self.final < len(self.jobs) and self.jobs[self.final][2]:
                row = self.jobs[self.final]
                row[3] = max(row[0], heapq.heappop(self.free))
                row[4] = row[3] + row[1]
                heapq.heappush(self.free, row[4])
                self.final += 1
            self.cv.notify_all()
            return range(first, self.final)

    def start_of(self, k: int) -> float:
        """Job k's start; blocks until every earlier job is placed."""
        with self.cv:
            if not self.cv.wait_for(lambda: self.final >= k, self.diag_s):
                logger.warning(
                    "station job %d waits for job %d, which is not placed yet after "
                    "%s wall seconds; still waiting",
                    k,
                    self.final,
                    self.diag_s,
                )
                self.cv.wait_for(lambda: self.final >= k)
            s = self.jobs[k][3]
            return max(self.jobs[k][0], self.free[0]) if s is None else s

    def time_in_job(self, k: int) -> float:
        with self.cv:
            return self.start_of(k) + self.jobs[k][1]

    def completion(self, k: int) -> float | None:
        """Job k's completion, or ``None`` until it is placed."""
        with self.cv:
            return self.jobs[k][4]

    def unresolved(self) -> bool:
        with self.cv:
            return self.final < len(self.jobs)


def classify(fn) -> str:
    """``service``, ``wait`` or ``refuse``: the job registry, by qualified name."""
    name = getattr(fn, "__qualname__", "")
    if name.rpartition(".")[2] == "do_preprocess":
        return "service"
    if name == "Queue.get":
        return "wait"
    return "refuse"


class SimExecutor(ThreadPoolExecutor):
    """The loop's default executor; its jobs run in simulated time from `start_run`."""

    def __init__(self, loop, max_workers: int | None = None):
        super().__init__(max_workers)
        self.loop = loop
        self.station: Station | None = None
        self.passthrough = 0  # wait jobs in flight
        self.refusals: list[str] = []  # run-summary reasons, ``executor:<name>``
        self.settle: dict[int, partial | None] = {}  # finished, not yet placed

    def start_run(self) -> None:
        """Freeze the width: the pool's, less the wait jobs resident now."""
        self.station = Station(self._max_workers - self.passthrough)

    def submit(self, fn, /, *args, **kwargs) -> Future:
        kind = "wait" if self.station is None else classify(fn)
        if kind == "refuse":
            name = getattr(fn, "__qualname__", repr(fn))
            self.refusals.append(f"executor:{name}")
            raise Refused(("executor", name))
        if kind == "wait":
            self.passthrough += 1
            cf = super().submit(fn, *args, **kwargs)
            cf.add_done_callback(lambda _: self.loop.call_soon_threadsafe(self._waited))
            return cf
        k = self.station.enqueue(self.loop.time())
        out = Future()
        super().submit(self._serve, k, partial(fn, *args, **kwargs), out)
        return out

    def _waited(self) -> None:
        self.passthrough -= 1

    def _serve(self, k: int, fn, out: Future) -> None:
        settle = None  # a job cancelled before it started is skipped, as the pool does
        if out.set_running_or_notify_cancel():
            job.cur = (self.station, k)
            try:
                settle = partial(out.set_result, fn())
            except BaseException as e:  # noqa: BLE001 - the coroutine gets it
                settle = partial(out.set_exception, e)
            finally:
                job.cur = None
        self.loop.call_soon_threadsafe(self._finish, k, settle)

    def _finish(self, k: int, settle) -> None:
        # On the loop thread, so a placed job's timer is set before the loop's
        # next select can see the station resolved and advance the clock.
        self.settle[k] = settle
        for j in self.station.finish(k):
            done = self.settle.pop(j)
            if done is not None:
                self.loop.call_at(self.station.completion(j), done)


def _charge(d: float) -> None:
    cur = getattr(job, "cur", None)
    if cur is not None:
        cur[0].charge(cur[1], d)


def wrap_encode(encode, entry):
    """`encode`, charging the current job ``fixed + tokens / derated rate``."""
    rate = entry.encode_tokens_per_s * entry.derate

    def charged(*args, **kwargs):
        ids = encode(*args, **kwargs)
        _charge(entry.encode_fixed_s + len(ids) / rate)
        return ids

    return charged
