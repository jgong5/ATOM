# SPDX-License-Identifier: MIT
"""The frontend LP's tokenizer station: thread-pool jobs on the simulated clock.

Tokenization runs for real, for its result, and is charged a modelled duration
for its time. A `Station` is a multi-server FIFO over that charged time: job k,
submitted at ``t_k``, starts at ``s_k = max(t_k, earliest free server)`` and
completes at ``c_k = s_k + its charged service``. Jobs are placed in submission
order, whatever order their real threads finish in. `SimExecutor` also runs its
service jobs for real one at a time, in submission order, so a side effect two
jobs at one instant share, such as the request id each `Sequence` draws, follows
the station's order and not the threads'.

`SimExecutor` is the event loop's default executor. From `SimExecutor.start_run`
on, `classify` sorts every job by name: `SERVICE_JOBS` are service jobs on
the station; ``Queue.get`` is a wait job passed straight through, since a
channel event completes it and zero simulated time is right; anything else is
refused, so a job ATOM later adds to the pool is reported, not timed as free.
A service job's result reaches its coroutine at ``c_k``, through ``call_at``.

A thread serving a station job reads the clock as the job's start plus the
service charged so far (`atom.utils.clock.current_job_time`).

A frontend output thread has a width-1 detokenization station of its own:
handling one released frame is one job, from `LPRuntime.handed_over` to its
next `LPRuntime.back_at_wait_point`, charged by the `wrap_decode` calls it
makes. Callbacks it posts with ``call_soon_threadsafe`` run at its completion.
A `wrap_encode` or `wrap_decode` call on the event loop thread, such as the
final decode of a non-streaming completion, advances the LP clock instead, once
every open service job is placed: a job's result due inside the call runs when
it ends, whatever wall time the job's thread takes.

`CompassEventLoop` is the frontend's event loop in a simulated run: its
``time()`` is the installed `LPRuntime`'s clock, so every timer is on LP time,
timers due at one instant fire in the order they were set, and its selector is
the frontend's idle point. Idle with nothing to read, it
asks for time with `next_event`: its earliest essential timer as ``t``, its
earliest daemon timer (`DAEMON_TIMERS`) as ``t_daemon``. While a released
request is unread or a station job is open it only waits on its sockets, and
warns once, naming what it waits for, if no socket event ends that wait within
``diag_s`` wall seconds. Its select returns nothing only once ``time()`` has
reached the caller's timeout, or once it ends `SimExecutor.placed`'s wait. The
``+inf`` grant ends the run: from then on ``time()`` runs on from the run's last
LP time at wall rate. A uvicorn server on the loop is asked to exit, as a signal
would ask it, and shuts down with nothing pending; with none, the loop cancels
its timers and stops. Over a clock connection with a socket, it waits for the
grant and its own sockets at once, so HTTP that reaches a frontend already in its run without a stamp, such
as atomesh's worker registration at startup, is served while the grant is out.
A timer such a request sets earlier than the time asked for fires at the grant;
a daemon one, such as uvicorn's keep-alive timeout, keeps the run alive no more
than any other daemon timer. A cross-LP send such a handler makes is refused
(`LPRuntime.stamp_send`).
asyncio's retry of a listening socket whose accept ran out of file
descriptors waits on the wall, not on the LP clock (`CompassEventLoop._serve_again_on_wall`).
A SIGINT or SIGTERM ends the run as the ``+inf`` grant does, at the LP clock,
and `CompassEventLoop.stopped` names it; uvicorn then cancels the requests still
open rather than wait for them, since only a grant would finish them. A loop
built in the main thread takes both with ``add_signal_handler``, so one wakes it
from a wait for the grant over a socket; uvicorn, which catches them while it
serves and raises a caught one again once it has shut down, raises it into that
handler, so ``uvicorn.run`` returns. Closing the loop puts back the handlers it
found.
`HttpChannel` is the inline receive of HTTP requests.
Building the loop installs `atom.compass.run.fe_fastpaths`: anyio's memory
streams and uvicorn's h11 body send with the same awaits and bytes, at less
CPU per streamed message.
"""

import asyncio
import collections
import contextlib
import heapq
import itertools
import json
import logging
import math
import queue
import select
import selectors
import signal
import sys
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

from atom.compass.carriers import (
    cancel_seq,
    relay_carried,
    stamp_events,
    tracestate_jitter,
    with_relay_stamp,
)
from atom.utils import clock
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
                if (rt := clock.installed()) is not None:
                    rt.diagnostics += 1
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


#: Service jobs by function name: a request's tokenization, a chat request's
#: media preparation (`atom.compass.run` prices its images), a `/metrics`
#: scrape's render, which charges nothing, and a chat request's template render
#: (``apply_chat_template``, or ``_prepare_prompt``'s render and tokenize for
#: ``/v1/messages`` and ``/v1/responses``), which the API server hands to a
#: thread and which charges what its encode calls charge.
SERVICE_JOBS = frozenset(
    {
        "do_preprocess",
        "_prepare_multimodal_inputs",
        "AtomMetricsExporter.render",
        "apply_chat_template",
        "_prepare_prompt",
    }
)
#: Service jobs by module and qualified name, for names too common to match
#: bare: OpenTelemetry's tracer setup and flush, which the API server's lifespan
#: hands to a thread as it starts and as it leaves, and which charge nothing.
#: The frontend's run begins at its loop's first select, before the lifespan.
MODULE_SERVICE_JOBS = frozenset(
    {"atom.observability.tracing.initialize", "atom.observability.tracing.shutdown"}
)


def _job(fn):
    """`fn`, or the function `asyncio.to_thread` runs when `fn` is its
    ``partial(Context.run, func, ...)``."""
    if (
        isinstance(fn, partial)
        and getattr(fn.func, "__qualname__", "") == "Context.run"
    ):
        fn = fn.args[0]
    return fn


def job_name(fn) -> str:
    """`fn`'s qualified name, or that of the function `asyncio.to_thread` runs
    when `fn` is its ``partial(Context.run, func, ...)``."""
    fn = _job(fn)
    return getattr(fn, "__qualname__", repr(fn))


def classify(fn) -> str:
    """``service``, ``wait`` or ``refuse``: the job registry, by qualified name."""
    name = job_name(fn)
    if name in SERVICE_JOBS or name.rpartition(".")[2] in SERVICE_JOBS:
        return "service"
    if f"{getattr(_job(fn), '__module__', None)}.{name}" in MODULE_SERVICE_JOBS:
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
        self.done = queue.SimpleQueue()  # (k, settle) from the serving thread
        self.serial = ThreadPoolExecutor(1)  # service jobs, in submission order
        self.placing: asyncio.Future | None = None  # `placed`'s wait

    async def placed(self) -> None:
        """On the loop: until the loop is idle at the LP clock now, every service
        job submitted by then placed and nothing else due by now left to run, so
        whatever else runs at this instant has run, however its awaits fell
        against the wall (`CompassSelector.select` ends the wait)."""
        self.placing = self.loop.create_future()
        await self.placing

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.serial.shutdown(wait, cancel_futures=cancel_futures)
        super().shutdown(wait, cancel_futures=cancel_futures)

    def start_run(self) -> None:
        """Freeze the width: the pool's, less the wait jobs resident now."""
        self.station = Station(self._max_workers - self.passthrough)

    def submit(self, fn, /, *args, **kwargs) -> Future:
        kind = "wait" if self.station is None else classify(fn)
        if kind == "refuse":
            name = job_name(fn)
            self.refusals.append(f"executor:{name}")
            raise Refused(("executor", name))
        if kind == "wait":
            self.passthrough += 1
            cf = super().submit(fn, *args, **kwargs)
            cf.add_done_callback(lambda _: self.loop.call_soon_threadsafe(self._waited))
            return cf
        k = self.station.enqueue(self.loop.time())
        out = Future()
        self.serial.submit(self._serve, k, partial(fn, *args, **kwargs), out)
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
        self.done.put((k, settle))
        self.loop.call_soon_threadsafe(self.place)

    def place(self, wait: bool = False) -> None:
        """On the loop thread: place the jobs whose threads are done, so a placed
        job's timer is set before the clock moves; with `wait`, until none is
        open, warning for a job whose thread takes ``diag_s`` wall seconds."""
        while (wait and self.station.unresolved()) or not self.done.empty():
            try:
                k, settle = self.done.get(timeout=self.station.diag_s if wait else None)
            except queue.Empty:
                logger.warning(
                    "an event-loop charge waits for station job %d, open after "
                    "%s wall seconds; still waiting",
                    self.station.final,
                    self.station.diag_s,
                )
                clock.installed().diagnostics += 1
                k, settle = self.done.get()
            self.settle[k] = settle
            for j in self.station.finish(k):
                done = self.settle.pop(j)
                if done is not None:
                    self.loop.call_at(self.station.completion(j), done)

    def unplaced(self) -> bool:
        """On the loop thread: whether a service job is submitted and not yet
        placed. It reads the station without its lock, since only the loop
        thread enqueues and places. A job counts until `place` takes it, its
        thread done or not."""
        station = self.station
        return station is not None and station.final < len(station.jobs)

    def idle(self) -> bool:
        """The loop is idle at the LP clock now: end `placed`'s wait, if there
        is one and every service job is placed; returns whether it ended one."""
        if self.placing is None or self.station.unresolved():
            return False
        placing, self.placing = self.placing, None
        if not placing.done():
            placing.set_result(None)
        return True


def charge(d: float) -> None:
    """Charge `d` to this thread's station job; with none open, in the run, the
    caller is the clock owner and advances the LP clock by `d`, on an event
    loop only once every open service job is placed, so one that completes
    inside the charge is not late. `advance_to` refuses any other thread before
    it places anything, so the loop's jobs are placed only on its own thread."""
    cur = getattr(job, "cur", None)
    if cur is not None:
        cur[0].charge(cur[1], d)
        return
    rt = clock.installed()
    if rt is not None and rt.in_run:
        loop = rt.loop if threading.current_thread() is rt.owner else None
        if loop is not None and loop.executor.unplaced():
            loop.executor.place(wait=True)
        rt.advance_to(rt.now + d)


#: Whether this thread is inside a `wrap_encode` call, which charges the backend
#: calls its ``encode`` makes.
_in_encode = threading.local()
#: The ids the ``encode_batch`` calls this thread charged returned, over which a
#: segment tokenizer call's own work is charged.
_encoded = threading.local()


def _price(rate, tokens: int) -> float:
    """Seconds `rate`, a kind of call's term in a ``host`` entry's
    ``tokenizer``, charges a call of `tokens` tokens: ``fixed_s + tokens /
    tokens_per_s``, ``fixed_s`` alone with no ``tokens_per_s``; 0 with no
    `rate`."""
    if rate is None:
        return 0.0
    return rate["fixed_s"] + tokens / rate.get("tokens_per_s", math.inf)


def wrap_encode(encode, rate):
    """`encode`, charging the current job `_price` of `rate`, a ``host`` entry's
    ``tokenizer.encode``."""

    def charged(*args, **kwargs):
        _in_encode.on = True
        try:
            ids = encode(*args, **kwargs)
        finally:
            _in_encode.on = False
        charge(_price(rate, len(ids)))
        return ids

    return charged


def wrap_encode_batch(encode_batch, one, several):
    """A fast tokenizer backend's ``encode_batch``, charging the current job
    `_price` over the ids of all the encodings it returns, of `one`, the
    ``tokenizer.encode`` term, for one encoding, and of `several`, the
    ``tokenizer.encode_batch`` term, for more; a call made inside a
    `wrap_encode` call is that call's to charge."""

    def charged(*args, **kwargs):
        encodings = encode_batch(*args, **kwargs)
        if not getattr(_in_encode, "on", False):
            tokens = sum(len(e.ids) for e in encodings)
            _encoded.ids = getattr(_encoded, "ids", 0) + tokens
            charge(_price(one if len(encodings) == 1 else several, tokens))
        return encodings

    return charged


def wrap_segment(encode_cached, rate):
    """The segment tokenizer call, charging its own work: `_price` of `rate`, a
    ``tokenizer.segment``, over the ids its ``encode_batch`` calls returned."""

    def charged(*args, **kwargs):
        _encoded.ids = 0
        got = encode_cached(*args, **kwargs)
        charge(_price(rate, _encoded.ids))
        return got

    return charged


def wrap_decode(decode, rate):
    """`decode`, charging the current job `_price` of `rate`, a ``host`` entry's
    ``tokenizer.decode``. A decode of a token count outside the term's
    ``tokens`` is charged nothing and is a run-summary reason
    ``tokenizer:decode ...`` naming them."""
    lo, hi = (rate or {}).get("tokens", (0, math.inf))

    def charged(ids, *args, **kwargs):
        if lo <= len(ids) <= hi:
            charge(_price(rate, len(ids)))
        elif (rt := clock.installed()) is not None:
            rt.refusals.append(
                f"tokenizer:decode of {len(ids)} tokens, outside the {lo} to "
                f"{hi} tokens its decode was fitted on"
            )
        return decode(ids, *args, **kwargs)

    return charged


#: Timers that do not keep the run alive, by the qualified name of their
#: callback or of a coroutine on the stack that scheduled them: the API
#: server's periodic jobs (metrics refresh, collector watch), uvicorn's server
#: tick and its keep-alive timeout.
DAEMON_TIMERS = frozenset(
    {
        "_periodic",
        "Server.main_loop",
        "H11Protocol.timeout_keep_alive_handler",
        "HttpToolsProtocol.timeout_keep_alive_handler",
    }
)


def _is_daemon(callback) -> bool:
    if getattr(callback, "__qualname__", None) in DAEMON_TIMERS:
        return True
    f = sys._getframe(2)  # the caller of `call_at`
    while f is not None and f.f_code.co_qualname not in DAEMON_TIMERS:
        f = f.f_back
    return f is not None


class _Timer(asyncio.TimerHandle):
    """A timer that fires after every timer set before it for the same instant.

    asyncio's heap orders timers by deadline alone, so on a clock where many
    share one, two would fire in whatever order the rest of the heap left them,
    a cancelled timer it still holds included.
    """

    __slots__ = ("order",)
    _orders = itertools.count()

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.order = next(_Timer._orders)

    def __lt__(self, other) -> bool:
        return (self._when, self.order) < (other._when, other.order)


#: The signals that stop a simulated server, as they stop a real one.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)
#: What asyncio calls to serve a listening socket again once an accept that ran
#: out of file descriptors took it off the loop (`CompassEventLoop._serve_again_on_wall`).
_START_SERVING = asyncio.selector_events.BaseSelectorEventLoop._start_serving


class CompassEventLoop(asyncio.SelectorEventLoop):
    """The frontend's event loop on its LP clock; uvicorn's ``loop`` in a simulated run."""

    def __init__(self) -> None:
        self.rt = clock.installed()
        if self.rt is None:
            raise RuntimeError(
                "CompassEventLoop runs on an LP clock; install the frontend's "
                "LPRuntime with atom.utils.clock.install first"
            )
        self.daemon = weakref.WeakSet()  # the daemon timers scheduled
        # listening socket -> its pending wall-clock `_start_serving` retry
        self.serving_again: dict = {}
        # Requests read, not handed over: (channel, seq) -> (arrival, wake-up,
        # when its body is in, once it is read).
        self.held: dict[
            tuple[str, int], tuple[float, asyncio.Future, asyncio.Future]
        ] = {}
        # Requests woken and waiting for their turn, a heap of (order, turn);
        # `_give_turns` gives one turn at a time, and the request holds it
        # until it is counted handled, so the app gets them in that order.
        self.turns: list[tuple[tuple, asyncio.Future]] = []
        self.giving: asyncio.Task | None = None
        self.turn_over: asyncio.Future | None = None
        # Per output thread: its width-1 station, its open job, and the callbacks
        # that job posted.
        self.detok = threading.local()
        self.ended = None  # at the +inf grant: (last LP time, wall time() then)
        super().__init__(CompassSelector(self))
        self.executor = SimExecutor(self)
        self.set_default_executor(self.executor)
        from atom.compass.run import fe_fastpaths

        fe_fastpaths.install()
        self.rt.loop = self
        self.stopped = None  # the signal that stopped the run, by name
        self.handlers_before = {}  # each stop signal's handler before this loop's
        if threading.current_thread() is threading.main_thread():
            for sig in STOP_SIGNALS:
                self.handlers_before[sig] = signal.getsignal(sig)
                self.add_signal_handler(sig, self.stop_run, sig)

    def time(self) -> float:
        if self.ended is None:
            return self.rt.read_clock()
        return self.ended[0] + super().time() - self.ended[1]

    def finish(self, end: float) -> None:
        """The run ended at LP time `end`: at the ``+inf`` grant, or at a stop."""
        self.ended = (end, super().time())
        self._write_to_self()  # so the select that got the grant reports an fd
        for task in asyncio.all_tasks(self):
            coro = task.get_coro()
            if getattr(coro, "__qualname__", None) == "Server.serve":  # uvicorn's
                server = coro.cr_frame.f_locals["self"]
                server.should_exit = True
                if self.stopped is not None:  # what is open, only a grant finishes
                    server.config.timeout_graceful_shutdown = 0
                return
        for h in self._scheduled:
            h.cancel()
        self.stop()

    def stop_run(self, sig: int) -> None:
        """Signal `sig` ends the run at the LP clock; outside it, nothing."""
        if self.rt.in_run:
            self.stopped = signal.Signals(sig).name
            self.rt.end_run()
            self.finish(self.rt.now)

    def job_begin(self, arrival: float) -> None:
        """A thread took a released frame: handling it is one job on its station."""
        d = self.detok
        if not hasattr(d, "station"):
            d.station, d.due = Station(1), collections.deque()
        d.k, d.posted = d.station.enqueue(arrival), []
        job.cur = (d.station, d.k)

    def job_end(self) -> None:
        """The thread is back at its wait point: its job's callbacks run at completion."""
        d = self.detok
        posted = getattr(d, "posted", None)
        if posted is None:
            return
        job.cur = d.posted = None
        d.station.finish(d.k)
        d.due.append(posted)
        c = d.station.completion(d.k)
        super().call_soon_threadsafe(self.call_at, c, self._post, d.due)

    def _post(self, due) -> None:
        # Each timer takes the oldest job's callbacks, so a thread's jobs post
        # theirs in completion order.
        for callback, args in due.popleft():
            self.call_soon(callback, *args)

    def call_soon_threadsafe(self, callback, *args, context=None):
        posted = getattr(self.detok, "posted", None)
        if posted is None:
            return super().call_soon_threadsafe(callback, *args, context=context)
        posted.append((callback, args))

    def call_at(self, when, callback, *args, context=None):
        # asyncio's own, with a `_Timer`: `call_later` and `asyncio.sleep` come here.
        if when is None:
            raise TypeError("when cannot be None")
        self._check_closed()
        if getattr(callback, "__func__", None) is _START_SERVING:
            return self._serve_again_on_wall(
                when - self.time(), callback, args, context
            )
        handle = _Timer(when, callback, args, self, context)
        heapq.heappush(self._scheduled, handle)
        handle._scheduled = True
        if _is_daemon(callback):
            self.daemon.add(handle)
        return handle

    def _serve_again_on_wall(self, delay, callback, args, context) -> asyncio.Handle:
        """asyncio's `_start_serving` after `delay` wall seconds, not on the LP clock.

        asyncio takes a listening socket whose accept ran out of file
        descriptors off the loop and serves it again a second later, once for
        each accept that failed (a whole backlog, 2048 under uvicorn). That
        pause is the host's, not the run's: on the LP clock it would wait for
        time the loop does not ask for while the request the accept is to take
        in is released and unread, and the listener would never come back.
        One retry per socket stands for them all.
        """
        sock = args[1]
        if (pending := self.serving_again.get(sock)) is not None:
            return pending
        handle = self.serving_again[sock] = asyncio.Handle(
            callback, args, self, context
        )

        def serve() -> None:
            self.serving_again.pop(sock, None)
            if not handle.cancelled():
                handle._run()

        def due() -> None:
            with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
                self.call_soon_threadsafe(serve)

        timer = threading.Timer(max(delay, 0.0), due)
        timer.daemon = True
        timer.start()
        return handle

    def close(self) -> None:
        super().close()  # gives each stop signal its default handler
        for sig, handler in self.handlers_before.items():
            if handler is not None:  # None: not installed from Python
                signal.signal(sig, handler)
        if self.stopped is None:
            self.rt.close()

    def hand_over(self) -> bool:
        """Once every released request has been read, each one's body to its
        end, wake the released ones in the order they are ready
        (`HttpChannel`), then ``(arrival, seq)``; returns whether any was woken."""
        rt = self.rt
        with rt.lock:
            if any(rt.unread[ch] for ch, _ in self.held):
                return False
            out = [
                (ready, a, seq, ch)
                for (ch, seq), (a, _, ready) in self.held.items()
                if rt.is_released(ch, seq)
            ]
        if not all(ready.done() for ready, *_ in out):
            return False
        for *_, seq, ch in sorted((r.result(), a, s, c) for r, a, s, c in out):
            self.held.pop((ch, seq))[1].set_result(None)
        if out:
            self._write_to_self()  # so the select that woke them reports an fd
        return bool(out)

    @contextlib.asynccontextmanager
    async def turn(self, order: tuple):
        """Hold a woken request's turn: wait until `_give_turns` gives it, and
        end it on leaving. `order` places it among the requests waiting."""
        mine = self.create_future()
        heapq.heappush(self.turns, (order, mine))
        if self.giving is None:
            self.giving = self.create_task(self._give_turns())
        try:
            await mine
            yield
        finally:
            if mine.done() and not mine.cancelled() and not self.turn_over.done():
                self.turn_over.set_result(None)

    async def _give_turns(self) -> None:
        """Give the waiting requests their turns one at a time, each to the
        first in order once the loop is idle at its instant (`SimExecutor.placed`):
        every request released by then woken and waiting, so one released
        while others wait takes its place among them; every station job
        placed, since the clock may not pass a job's completion before it is;
        and the request before it gone as far as it can at this instant, or
        their tokenizer jobs reach the station in the wall order their awaits
        fall in."""
        try:
            while self.turns:
                await self.executor.placed()
                _, turn = heapq.heappop(self.turns)
                if turn.done():  # its request was cancelled waiting
                    continue
                self.turn_over = self.create_future()
                turn.set_result(None)
                await self.turn_over
        finally:
            self.giving = None

    def handing_over(self) -> bool:
        """Whether a released request waits to be woken: the loop is not idle
        at this instant until `hand_over` has woken every one, so the set it
        orders is all that is released when the clock moves on."""
        rt = self.rt
        with rt.lock:
            return any(rt.is_released(ch, seq) for ch, seq in self.held)

    def unread(self) -> tuple[int, str] | None:
        """The released request with the lowest seq not yet read, as
        ``(seq, channel)``, or ``None``."""
        rt = self.rt
        with rt.lock:
            return min(
                ((seq, ch) for ch, got in rt.unread.items() for seq in got),
                default=None,
            )


class CompassSelector(selectors.DefaultSelector):
    """The loop's one blocking point: a socket wait, or a next-event request."""

    def __init__(self, loop: CompassEventLoop) -> None:
        super().__init__()
        self.loop = loop
        # The wait the loop cannot ask time through: (what, wall time to warn
        # at, or None once warned). It outlives the selects an fd event ends.
        self.stall = None
        self.end = 0.0  # the LP clock when the request out was sent

    def select(self, timeout=None):
        loop, rt = self.loop, self.loop.rt
        if not rt.in_run:  # before the run, or after the +inf grant
            return super().select(timeout)
        if loop.executor.station is None:
            loop.executor.start_run()
        station = loop.executor.station
        # `timeout` is in time() units: it expires on the LP clock.
        deadline = math.inf if timeout is None else loop.time() + max(timeout, 0)
        while True:
            loop.hand_over()
            ready = super().select(0)
            if ready or loop.time() >= deadline:
                return ready
            # A positive timeout: nothing is ready or due, the loop is idle now.
            if not loop.handing_over() and loop.executor.idle():
                return ready
            if rt.inline_pending() or station.unresolved():
                # The LP clock stands still here: only an fd event ends the wait.
                unread = loop.unread()
                what = (
                    f"released {unread[1]} seq {unread[0]} is unread"
                    if unread
                    else (
                        f"station job {station.final} is open"
                        if station.unresolved()
                        else "a released request is read and not handled"
                    )
                )
                wall = time.monotonic()
                if self.stall is None or self.stall[0] != what:
                    self.stall = (what, wall + rt.diag_s)
                warn_at = self.stall[1]
                ready = super().select(None if warn_at is None else warn_at - wall)
                if ready:
                    return ready
                if warn_at is not None:
                    logger.warning(
                        "%s: event loop has had no fd event for %s wall seconds "
                        "while %s; still waiting",
                        rt.me,
                        rt.diag_s,
                        what,
                    )
                    rt.diagnostics += 1
                    self.stall = (what, None)
                continue
            self.stall = None
            if not rt.asked:
                t = t_daemon = math.inf
                for h in loop._scheduled:
                    if h.cancelled():
                        continue
                    if h in loop.daemon:
                        t_daemon = min(t_daemon, h.when())
                    else:
                        t = min(t, h.when())
                # With messages sent ahead, the next event is the LP's own
                # step below its ceiling, which waits on nothing.
                if not hasattr(rt.conn, "fileno") or rt.ahead:
                    end = rt.now
                    if self._granted(end, rt.next_event(t, t_daemon)):
                        return super().select(0)
                    continue
                self.end = rt.now
                rt.ask_next_event(t, t_daemon)
            waiting = select.poll()
            for fd in (self, rt.conn):
                waiting.register(fd, select.POLLIN)
            polled = {fd for fd, _ in waiting.poll()}
            if rt.conn.fileno() in polled and self._granted(self.end, rt.take_grant()):
                return super().select(0)

    def _granted(self, end: float, g: float) -> bool:
        """Whether grant `g` is ``+inf``, which ends the run at LP time `end`."""
        if g != math.inf:
            return False
        self.loop.finish(end)
        return True


#: The API server's endpoints that hand a request to the engine.
GENERATION_PATHS = frozenset(
    ("/v1/completions", "/v1/chat/completions", "/v1/messages", "/v1/responses")
)


class HttpChannel:
    """ASGI middleware: the frontend's inline receive of HTTP requests.

    `stamp(scope)` is a request's ``(arrival, seq)`` read from its carrier, or
    ``None`` for one sent outside the run, which passes straight through, except
    to a `GENERATION_PATHS` endpoint during the run: that one is answered 400,
    naming it, and is a run-summary reason ``http:unstamped <method> <path>``. A
    stamped request waits until it is released and every released request has
    been read, its body to the end, and counts as handled once its turn has
    come. The body is read off the socket as it comes, while the request waits
    its turn: the turns of a burst pass on the wall one tokenizer job at a
    time, and a client whose send stalls on an unread socket past its
    ``TCP_USER_TIMEOUT`` (aiperf's default is 30 s) drops the connection. The
    app gets the received body messages again from its ``receive``.

    Released requests take their turns in the order their bodies are in, the
    real frontend reading every socket at once: each one's arrival plus its own
    intake, ``fixed_s + body bytes x per_byte_s`` (none with no
    ``http_intake``), then ``(arrival, seq)``, the order the traffic LP makes
    the same in every run (`compass_harness.traffic_lp`). Each turn goes to the
    first of those waiting once the loop is idle at its instant
    (`CompassEventLoop.turn`), and the loop is not idle until every request
    released by then is waiting (`CompassEventLoop.handing_over`): a request
    released while others wait, such as a burst's part the traffic LP sends
    after a grant at the same instant, takes its place among them, and which
    requests are ordered together never depends on the wall. So a burst sent
    at one instant reaches the app smallest body first, as a real rank admits
    it. A cancel takes its turn after every request waiting with it, among
    them any it names.

    A request whose ``tracestate`` entry names an intake jitter, ``j:<bytes>``
    (`atom.compass.carriers`), is ordered as if its body were that many bytes
    longer: ``fixed_s + (body bytes + jitter) x per_byte_s``. The traffic LP
    draws one per conversation and turn from its send-order seed
    (`compass_harness.traffic_lp`), so a seed samples the order a real burst's
    wall spread of sends gives, and the same seed gives the same order. The
    jitter orders requests only: their intake charge is their own bodies'.
    With none, as with no seed, they take the exact intake order.

    A stamped request whose body is a cancel
    (`atom.compass.carriers.cancel_seq`) goes to no app: it is answered 204,
    and the request it names, if the app is still serving it, reads
    ``http.disconnect`` from its ``receive`` then. That is the only hang-up a
    stamped request reads; its socket closing is never handed on.

    An LP whose requests come in on a ``relay`` channel reads each one's stamp,
    a cancel's too, from its body's ``kv_transfer_params`` instead. An LP that
    sends on one stamps a send on it for each JSON response whose
    ``kv_transfer_params`` is an object, and writes the stamp there; with a
    ``relay`` term in its host entry (`atom.compass.run.relay`), it also writes
    there the relay's service time (`carriers.RELAY_SERVICE`): that term's
    time, less ``router_s``, where that is positive, for the bytes of
    decode's request, which the router builds from the client's request and
    the prompt's token ids on the response: the two bodies' bytes. The send
    arrives at the channel's lookahead, and the receiving LP charges the
    service on its loop at the request's turn, before its intake: the relays
    and intakes of requests in flight together take the loop one at a time,
    so a relay waits behind those ahead of it as on r256 (#3720), where
    prefill's response to decode's app call took p50 12 ms alone and 100 ms
    with 8 others in flight, while a lone one takes the term's time as
    before. A relayed request without the field is charged no relay.

    During the run, each event of a ``text/event-stream`` response body leaves
    with a comment line carrying its stamp on the LP's stream channel
    (`atom.compass.carriers.stamp_events`). Each body message is one frame the
    app's stream yielded, so it must end in a blank line. With an
    ``http_stream`` term in its host entry (`atom.compass.run.http_stream`),
    the loop is charged ``per_event_s`` for each event but a ``[DONE]`` before
    it stamps them: a real decode server's loop spends ~50 us of CPU an
    event, a fifth of its wall time at c=256 (r256, #3720), and the relays
    and intakes it takes in wait behind that work.

    Taking in a stamped request other than a cancel costs the event loop its
    relay's service time, if it came on a relay that states one, then
    ``fixed_s + body bytes x per_byte_s`` (`atom.compass.run.http_intake`),
    charged on the loop at its turn, once the loop is idle at that instant,
    every tokenizer station job placed and all else due by then run
    (`SimExecutor.placed`), so a burst reaches the app one request at a time:
    the term is the gap between the starts of a real burst's requests, which
    its loop takes in one at a time.
    A request waits for that idle loop with no intake charged too: its body no
    longer paces the turns, and without the wait the burst's tokenizer jobs
    would reach the station in the wall order their awaits fall in.
    A request that arrives once the last intake has ended is charged ``idle_s``
    in place of ``fixed_s``. A ``host`` entry with no ``http_intake`` charges nothing.
    A body outside the bytes the term was fitted over
    (`atom.compass.run.http_intake_hull`) is charged nothing and is a run-summary
    reason ``http:intake ...`` naming that hull.
    """

    def __init__(self, app, stamp) -> None:
        from atom.compass.run import http_intake, http_intake_hull, http_stream, relay

        self.app, self.stamp = app, stamp
        self.intake, self.hull = http_intake(), http_intake_hull()
        self.relay, self.per_event = relay(), http_stream()
        self.intake_end = -math.inf  # the LP time the last intake ended
        # http seq -> its hang-up, for each stamped request the app is serving
        self.hangups: dict[int, asyncio.Event] = {}

    async def __call__(self, scope, receive, send):
        rt = clock.installed()
        relay_in = relay_out = hangup = None
        taken = [0]  # the request's body bytes, once taken in
        if rt is not None and scope["type"] == "http":
            relay_in = _relay(rt.table.channels_into(rt.me))
            relay_out = _relay(rt.table.channels_from(rt.me))
            send = _stamping_events(send, rt, self.per_event)
        if relay_out is not None:
            send = _stamping(send, rt, relay_out, self.relay, taken)
        service = 0.0  # the seconds its relay serves it for, where stated
        if relay_in is not None:
            body, receive = await _read_body(receive)
            got, service = relay_carried(body)
        else:
            got = self.stamp(scope) if scope["type"] == "http" else None
        if (
            got is None
            and rt is not None
            and rt.in_run
            and scope["type"] == "http"
            and scope["path"] in GENERATION_PATHS
        ):
            return await _refuse_unstamped(rt, scope["method"], scope["path"], send)
        if got is not None:
            loop = asyncio.get_running_loop()
            if not isinstance(loop, CompassEventLoop):
                raise TypeError(
                    f"a stamped request is received on a CompassEventLoop, not on "
                    f"{type(loop).__qualname__}"
                )
            rt = loop.rt
            ch = relay_in or next(
                c.name
                for c in rt.table.channels_into(rt.me)
                if c.name.endswith(":http")
            )
            arrival, seq = got
            rt.check_arrival(ch, arrival, seq)
            jitter = 0.0 if relay_in is not None else _intake_jitter(scope)
            wake, ready = loop.create_future(), loop.create_future()
            # Its body leaves the socket as it comes, not at its turn.
            reading = asyncio.ensure_future(_body_messages(receive))
            reading.add_done_callback(
                partial(self._ready, ready, arrival, jitter, service)
            )
            loop.held[ch, seq] = (arrival, wake, ready)
            try:
                await wake
                async with loop.turn((ready.result(), arrival, seq, ch)):
                    # A request counts as handled only once nothing about it
                    # still waits on the socket.
                    body = await reading
                    cancelled = cancel_seq(body[0].get("body", b""))
                    nbytes = taken[0] = sum(len(m.get("body", b"")) for m in body)
                    if service and cancelled is None:
                        # The relay serves it on this loop, behind those ahead.
                        charge(service)
                    if self.intake is not None and cancelled is None:
                        fixed, per_byte, idle = self.intake
                        if arrival >= self.intake_end:
                            fixed = idle
                        lo, hi = self.hull or (0, math.inf)
                        if lo <= nbytes <= hi:
                            charge(fixed + per_byte * nbytes)
                        else:
                            rt.refusals.append(
                                f"http:intake of a {nbytes}-byte body, outside the "
                                f"{lo:.0f} to {hi:.0f} bytes its http_intake was "
                                "fitted on"
                            )
                        self.intake_end = loop.time()
                    with rt.lock:
                        rt.count_done_locked(ch, seq)
            finally:
                reading.cancel()  # a request cancelled before its body was read
            if cancelled is not None:
                if (hangup := self.hangups.get(cancelled)) is not None:
                    hangup.set()
                await send({"type": "http.response.start", "status": 204})
                await send({"type": "http.response.body", "body": b""})
                return
            hangup = self.hangups[seq] = asyncio.Event()
            inner = receive

            async def receive():
                if body:
                    return body.pop(0)
                if hangup is None:
                    return await inner()
                await hangup.wait()
                return {"type": "http.disconnect"}

        try:
            await self.app(scope, receive, send)
        finally:
            if hangup is not None:
                del self.hangups[seq]

    def _ready(
        self,
        ready: asyncio.Future,
        arrival: float,
        jitter: float,
        service: float,
        reading,
    ) -> None:
        """Resolve `ready` with when the request's body, read at once with the
        others', is in: its arrival plus its relay's `service` and its own
        intake, ``fixed_s + (body bytes + jitter) x per_byte_s``, which depends
        on nothing the wall decides; a cancel's after every request it can
        name. A body never read is in at once."""
        if ready.done():
            return
        if reading.cancelled() or reading.exception() is not None:
            ready.set_result(arrival)
        elif cancel_seq(reading.result()[0].get("body", b"")) is not None:
            ready.set_result(math.inf)
        else:
            fixed, per_byte, _ = self.intake or (0.0, 0.0, 0.0)
            nbytes = sum(len(m.get("body", b"")) for m in reading.result())
            ready.set_result(arrival + service + fixed + per_byte * (nbytes + jitter))


def _intake_jitter(scope) -> float:
    """The intake jitter, in bytes, a request's ``tracestate`` header lines
    carry (`atom.compass.carriers.tracestate_jitter`); 0.0 with none."""
    return tracestate_jitter(
        ",".join(
            v.decode("latin-1")
            for k, v in scope.get("headers", ())
            if k.lower() == b"tracestate"
        )
    )


async def _refuse_unstamped(rt, method: str, path: str, send) -> None:
    """Answer an unstamped generation request 400 and record why for the run summary."""
    why = f"unstamped {method} {path}"
    rt.refusals.append(f"http:{why}")
    message = (
        f"{rt.me}: {method} {path} carries no Compass stamp during a simulated "
        "run, so no time can be given to it; send it through the run's traffic "
        "LP (aiperf with compass_harness), which stamps every request"
    )
    body = json.dumps({"error": {"type": "invalid_request_error", "message": message}})
    await send(
        {
            "type": "http.response.start",
            "status": 400,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body.encode()})


def _relay(channels) -> str | None:
    return next((c.name for c in channels if c.name.endswith(":relay")), None)


async def _body_messages(receive) -> list[dict]:
    """The request's body messages, read to the last."""
    body = [await receive()]
    while body[-1].get("more_body"):
        body.append(await receive())
    return body


async def _read_body(receive):
    """The request's whole body, and a `receive` that hands it on again."""
    chunks, more = [], True
    while more:
        message = await receive()
        chunks.append(message.get("body", b""))
        more = message.get("more_body", False)
    body, replayed = b"".join(chunks), False

    async def replay():
        nonlocal replayed
        if replayed:
            return await receive()
        replayed = True
        return {"type": "http.request", "body": body, "more_body": False}

    return body, replay


def _stamping_events(send, rt, per_event=None):
    """`send`, stamping each event of an SSE response body sent during the run,
    with `per_event`, `atom.compass.run.http_stream`'s seconds, first charged
    on the loop for each event of the frame but a ``[DONE]``.

    The body message that ends in the ``[DONE]`` event ends the response too,
    and what the app sends after it is dropped. The traffic LP holds its clock
    until the client has the return of a request whose ``[DONE]`` it was
    released, and the client returns at the body's end. Left to the app, that
    end comes loop turns after ``[DONE]`` (Starlette hands each body message
    between tasks), and in those turns the loop's on-loop decodes advance the
    LP clock; one that reaches past the traffic LP's horizon waits for a grant
    the traffic LP only asks for once the return is in, and the run hangs.
    """
    events = ended = False

    async def stamping(message):
        nonlocal events, ended
        if ended:
            return
        if message["type"] == "http.response.start":
            content_type = dict(message.get("headers", [])).get(b"content-type", b"")
            events = content_type.startswith(b"text/event-stream")
        elif events and rt.in_run:
            text = message.get("body", b"").decode()
            if per_event:
                sent = text.split("\n\n")[:-1]
                n = sum(1 for e in sent if not e.endswith("data: [DONE]"))
                if n:
                    charge(per_event * n)
            text = stamp_events(text, rt)
            message = {**message, "body": text.encode()}
            if text.endswith("\ndata: [DONE]\n\n"):
                ended, message["more_body"] = True, False
        await send(message)

    return stamping


def _stamping(send, rt, ch: str, relay=None, taken=(0,)):
    """`send`, holding a JSON response back until its body is whole, to stamp it
    for `ch`, arriving at the channel's lookahead; with `relay`,
    `atom.compass.run.relay`'s ``(fixed_s, per_byte_s, router_s)``, carrying
    ``fixed_s + bytes x per_byte_s - router_s`` where that is positive as the
    relay's service time (`carriers.RELAY_SERVICE`), which the receiving
    `HttpChannel` charges on its loop. ``bytes`` are those of the request the
    router builds for decode from the client's request, whose body's bytes
    `taken` holds, and this response's prompt token ids: the two bodies'
    bytes."""
    start, chunks = None, []

    async def stamping(message):
        nonlocal start
        if message["type"] == "http.response.start" and (
            b"content-type",
            b"application/json",
        ) in message.get("headers", []):
            start = message
            return
        if start is None:
            return await send(message)
        chunks.append(message.get("body", b""))
        if message.get("more_body", False):
            return
        raw, service = b"".join(chunks), None
        if relay is not None:
            fixed_s, per_byte_s, router_s = relay
            extra = fixed_s + (taken[0] + len(raw)) * per_byte_s - router_s
            service = extra if extra > 0 else None
        body = with_relay_stamp(raw, lambda: rt.stamp_send(ch), service)
        headers = [(k, v) for k, v in start["headers"] if k != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        await send({**start, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    return stamping
