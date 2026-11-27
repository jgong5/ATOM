# SPDX-License-Identifier: MIT
"""The traffic LP: aiperf's ``timing_manager`` as one logical process on the Compass clock.

Its clock owner is the event loop thread, which blocks inside each clock call;
worker reports and credit returns queue in ZMQ meanwhile. It is the clock
``ClockPacedLoopScheduler`` paces on, stamps every request the router sends,
and publishes the stamp to the workers.

Before each clock call it holds until aiperf has caught up with the clock:
every stream event the Clock Authority released has been reported by a worker,
every credit whose final event is released has had its return passed to
aiperf (a return that comes before its final event's release waits for it, so
the next turn is scheduled from the final arrival either way), no return is
still inside aiperf's callback, every task on the loop is waiting on
something, and the done callbacks of a task that ended have run: the
scheduler's drain observer, which rechecks the system idle cap once a timer's
task ends, is one, and the clock moving on before it would let an idle gap run
past the cap. It asks for time only while a request is open, a clock wait (a
pacing timer, a phase deadline, the idle watchdog) is pending or the run is
finishing; between phases aiperf works on the wall clock and the simulation
waits for it. While a request is open it asks for a phase deadline as a
daemon deadline, and for no other wait due at or past it: the deadline fires
once the open requests' work reaches it, and when the last one ends first and
aiperf withdraws the deadline, nothing it asked for keeps an LP running past
that end. The Clock Authority's essential horizon only grows, so a wait asked
for and then withdrawn, or a pacing timer aiperf pulls earlier, would.

The requests aiperf sends at one instant are stamped together at the end of
the hold, when it has sent them all, before the clock call. aiperf sends them
in the order its tasks happen to run, which follows when, on the wall clock,
each worker's return reached it; the frontend takes them in seq order, and its
router places them in that order. So their seqs follow each one's rank
(`router.send_rank`: its phase, conversation and turn), and a run repeats
under the same inputs. With ``SEND_ORDER_SEED_ENV`` set, they follow a digest
of the seed and the rank instead: each seed is one draw of that order, the
same in every run, so the router's placements can be sampled on purpose.

The frontend hands one instant's requests on in the order their bodies are
in, smallest first (`atom.utils.compass_loop.HttpChannel`), so seqs only break
its ties. A seed therefore also gives each request an intake jitter (`jitter`),
which its stamp carries to the frontend: its body is ordered as if it were that
many bytes longer. It stands for the real client's wall spread of one
instant's sends, which the real frontend's concurrent reads mix with the
bodies' sizes. Each is a draw per seed and rank, its phase, conversation and
turn, log-normal with median ``INTAKE_JITTER_ENV`` bytes (by default
`INTAKE_JITTER_BYTES`) and the spread `INTAKE_JITTER_SIGMA` of the real burst's
send offsets. With no seed, or a median of 0, no request carries one and the
frontend takes the exact intake order. A frontend under
``COMPASS_HTTP_INTAKE_ORDER=send`` takes one instant's requests in seq order,
its bodies and their jitters ordering nothing (`atom.utils.compass_loop`).

A run file that states an encoding ingress (``ingress``, `atom.compass.ingress`)
puts it in front of the HTTP channel's receiver, and the traffic LP prices its
hop: each worker reports its request's body size before the request is sent,
the hold stamps no request before its size is in, and `Ingress.admit` gives
each, in seq order, its arrival at the receiver past the ingress's queue,
encode and the router's parse. Its stamp then ends ``"start", <time>``, its
arrival at the ingress, where its record starts, so a client run's TTFT and
latency hold the hop. A cancel arrives no earlier than its request: in the real
ingress a request whose client has left is still encoded and sent on. With no
``ingress`` nothing of this runs and no worker reports a size.

With ``CONN_POOL_ENV`` set to 1, the traffic LP also keeps aiperf's pooled
HTTP connections on the clock (`ConnPool`): one pool per aiperf worker, as each
worker's aiohttp connector is one, which a request takes its connection from
when its stamp is made, the oldest idle one first, kept alive for aiperf's
``AIPERF_HTTP_KEEPALIVE_TIMEOUT`` (300 s by default), and which a request
gives its connection back to at its final event's arrival; a cancelled
request's connection is closed. A request with no idle connection left opens a
new one, and its stamp ends ``"new", True``, which its worker writes into its
``tracestate`` (`atom.compass.carriers`): the frontend takes new connections
in one at a time, in their seqs' order, and only kept-alive ones' bodies at
once (`atom.utils.compass_loop.HttpChannel`). #752 DeepSeek-V4.1 Cell A v2:
replayed on the records' own client timelines, this pool gives aiperf's own
reuse of every request in 99.6-100% of the 16 target records' requests, at
c1-c512 in both configs; DEPA2 c512's 300 s burst comes 91-93% on new
connections, DPA4 c512's 7-13%.

When aiperf cancels the requests in flight at its grace deadline, the traffic
LP closes them at that time (`cancel`): each one's cancel is a stamped request
of its own, which its worker sends and the frontend answers by hanging the
cancelled request up at the cancel's arrival. Behind a PD router the router LP
answers it and hands it on to the frontend that holds the request. A PD router
that is no LP answers a cancel 400, as a chat request it cannot read, so behind
one, where the HTTP channel leads to frontend-P, each cancel goes to
frontend-P's own URL, from ``PREFILL_URL_ENV``. A cancel its worker has not
reported written after ``DIAG_S`` wall seconds is named in the log.

Every stamp, a send's or a cancel's, goes to every worker, and none may be
lost: a worker that never gets its request's stamp never sends it, and one that
never gets its cancel never writes it, so `cancel` and the clock wait forever.
So the stamp socket has no high-water mark. A PUB socket drops a subscriber's
messages once its pipe holds the mark's worth, by the count the I/O thread last
reported read, which it does only every half mark; at the default 1000, two
c=512 runs on a loaded node lost the last 114 of 769 cancels to every worker
(the burst's 655th took each pipe to 9,000 messages written, the last report
8,000 read).

A request that errors mid-stream never reports the events after its last read.
When released events stay unreported for ``DIAG_S`` wall seconds while an
errored request is still open, the hold fails, naming those requests.
"""

import asyncio
import collections
import hashlib
import logging
import math
import os
import pickle
from statistics import NormalDist

import zmq
import zmq.asyncio

from atom.compass import clock_transport
from atom.compass import ingress as compass_ingress
from atom.compass import run as compass_run
from atom.compass.clock import LpId
from atom.utils.clock import DIAG_S, LPRuntime
from compass_harness import ADDRESS_ENV
from compass_harness.transport import addresses, credit_key

# The single-engine run's two channels; a run file names its own (`from_env`).
HTTP = "traffic->frontend:http"
STREAM = "frontend->traffic:stream"
TRAFFIC = LpId("traffic")
#: The HTTP channel behind a PD router that is no LP, and the variable naming
#: the URL of frontend-P, where a cancel on that channel goes.
PREFILL_HTTP = "traffic->frontend-P:http"
PREFILL_URL_ENV = "COMPASS_HARNESS_PREFILL_URL"
#: The seed of the order the requests sent at one instant are stamped in; unset
#: or empty, they follow their ranks.
SEND_ORDER_SEED_ENV = "COMPASS_HARNESS_SEND_ORDER_SEED"
#: ``aiperf``: the requests sent at one instant take their seqs in the order
#: aiperf sent them, its credit issue order, as the real client's sends go out;
#: unset or empty, by rank or seed (`TrafficLP._place`).
SEND_ORDER_ENV = "COMPASS_HARNESS_SEND_ORDER"
#: The median, in body bytes, of a seeded run's intake jitter (`TrafficLP.jitter`);
#: unset or empty, `INTAKE_JITTER_BYTES`; 0, none.
INTAKE_JITTER_ENV = "COMPASS_HARNESS_INTAKE_JITTER_BYTES"
#: Cell A c=256's 300 s burst (#3742), whose bodies run 3.8 bytes a token: the
#: median at which a simulated burst's order matches the real one's. Within the
#: burst, Spearman of route order against input size reads +0.943 real (each
#: rank's prefill entry +0.935); a Monte Carlo over the burst's sizes gives
#: +0.942 +- 0.019 at this median, and ten seeded runs +0.941 +- 0.020 (each
#: rank's entry +0.940 +- 0.021).
INTAKE_JITTER_BYTES = 48_000.0
#: The log-normal sigma of the intake jitter: the real burst's send offsets
#: from its first send, p10 / p50 / p90 2.8 / 12.2 / 67 ms, ln(67 / 12.2) / 1.2816.
INTAKE_JITTER_SIGMA = 1.33
#: ``1``: the traffic LP keeps aiperf's pooled connections (`ConnPool`) and
#: marks each request sent on a new one; unset or empty, no pool.
CONN_POOL_ENV = "COMPASS_HARNESS_CONN_POOL"
#: Loop turns the hold waits for aiperf to block before it says so in the log.
SPIN_DIAG = 100_000
_NORMAL = NormalDist()

logger = logging.getLogger(__name__)


def conn_pool_on() -> bool:
    """Whether `CONN_POOL_ENV` asks for the pool: ``1``; unset or empty, no;
    any other value is refused."""
    value = os.environ.get(CONN_POOL_ENV) or ""
    if value not in ("", "1"):
        raise ValueError(f"{CONN_POOL_ENV}={value!r}; it takes 1, or unset")
    return value == "1"


def send_order_aiperf() -> bool:
    """Whether `SEND_ORDER_ENV` asks for aiperf's own send order: ``aiperf``;
    unset or empty, no; any other value is refused."""
    value = os.environ.get(SEND_ORDER_ENV) or ""
    if value not in ("", "aiperf"):
        raise ValueError(f"{SEND_ORDER_ENV}={value!r}; it takes aiperf, or unset")
    return value == "aiperf"


class ConnPool:
    """aiperf's pooled HTTP connections on the clock, one pool per worker.

    Each aiperf worker sends on its own aiohttp connector (aiperf's default
    ``POOLED`` reuse). A request takes the oldest idle connection, closing on
    the way those idle past `keepalive_s` (aiohttp 3.13 ``BaseConnector._get``,
    its ``popleft`` and ``<=``), or opens a new one; a request that ends gives its
    connection back at its end. With `force_close` every request opens one.
    """

    def __init__(self, keepalive_s: float, force_close: bool = False) -> None:
        if not (keepalive_s > 0 and math.isfinite(keepalive_s)):
            raise ValueError(
                f"keepalive {keepalive_s!r} s: the pool models a finite keepalive > 0"
            )
        self.keepalive_s, self.force_close = float(keepalive_s), force_close
        self.idle: dict[str, collections.deque[float]] = {}

    def release(self, worker: str | None, t: float) -> None:
        """A request on `worker` ended at `t`: its connection goes back idle."""
        if worker is not None and not self.force_close:
            self.idle.setdefault(worker, collections.deque()).append(t)

    def take(self, worker: str | None, t: float) -> bool:
        """A request on `worker` takes a connection at `t`: whether it is a
        kept-alive one (else a new one)."""
        idle = self.idle.get(worker)
        while idle:
            if t - idle.popleft() <= self.keepalive_s:
                return True
        return False


class UnansweredRequests(RuntimeError):
    """Requests that never had their final response."""


class SendAfterCancel(RuntimeError):
    """A request sent after `TrafficLP.cancel` closed others."""


class TrafficLP:
    def __init__(
        self,
        rt: LPRuntime,
        prefix: str,
        http: str = HTTP,
        stream: str = STREAM,
        cancel_url: str | None = None,
        seed: str | None = None,
        jitter_median: float = INTAKE_JITTER_BYTES,
        ingress: compass_ingress.Ingress | None = None,
        aiperf_order: bool = False,
    ) -> None:
        # cancel_url: where each cancel goes, if not to its request's URL.
        # seed: the send order's (`SEND_ORDER_SEED_ENV`), or None.
        # jitter_median: of a seeded request's intake jitter, in bytes.
        # ingress: the encoding ingress in front of the HTTP channel, or None.
        self.rt, self.http, self.stream = rt, http, stream
        self.cancel_url, self.seed = cancel_url, seed
        # aiperf_order: one instant's sends take seqs in aiperf's send order.
        self.aiperf_order = aiperf_order
        self.ingress = ingress
        # Behind an ingress: each sent request's body bytes, None for none, as
        # its worker reported it; and each open one's arrival past the ingress.
        self._bodies: dict[tuple, int | None] = {}
        self._arrivals: dict[tuple, float] = {}
        if not (math.isfinite(jitter_median) and jitter_median >= 0):
            raise ValueError(
                f"intake jitter median {jitter_median!r} is no count of bytes >= 0"
            )
        self.jitter_median = jitter_median
        stamps, reports = addresses(prefix)
        ctx = zmq.asyncio.Context.instance()
        self._pub = ctx.socket(zmq.XPUB)
        self._pub.setsockopt(zmq.XPUB_VERBOSE, 1)
        # No high-water mark: past one, a stamp a worker lags behind is dropped
        # (see the module docstring). Set before the bind, which makes the pipes.
        self._pub.setsockopt(zmq.SNDHWM, 0)
        self._pub.bind(stamps)
        self._pull = ctx.socket(zmq.PULL)
        self._pull.bind(reports)
        self._subscribers = 0
        # sent at this instant, not yet stamped: (rank, key), in aiperf's order
        self._unstamped: list[tuple[tuple, tuple]] = []
        # stamped, return not yet passed to aiperf; in seq order, as cancels go
        self.open: dict[tuple, None] = {}
        self.cancelled: set[tuple] = set()  # closed by `cancel`
        self.closed: dict[tuple, None] = {}  # cancelled, return not yet passed
        self._written: set[tuple] = set()  # cancels a worker has written
        self._caught_up: list[asyncio.Future] = []
        self._reports: dict[int, tuple] = {}  # stream seq -> (key, final)
        # key -> its return, for aiperf; None once passed
        self._held: dict[tuple, object] = {}
        self._returns: asyncio.Queue = asyncio.Queue()  # passed in order, as upstream
        self._waiters: list[tuple[float, asyncio.Future, bool]] = []
        self._passing = False  # a return is inside aiperf's callback
        # The tasks not done at the hold's last look. The loop does not turn while
        # the owner is in a clock call, so the next hold starts from them; None
        # once the loop may have turned with no look.
        self._alive: set | None = None
        self._finishing = False
        self._changed = asyncio.Event()
        self._tasks = [
            asyncio.ensure_future(self._read_reports()),
            asyncio.ensure_future(self._pass_returns()),
        ]
        # With `CONN_POOL_ENV`: aiperf's connections (`ConnPool`), set by the
        # router; each open request's worker, and each ended one's final arrival.
        self.conn_pool: ConnPool | None = None
        self._workers: dict[tuple, str] = {}
        self._finals: dict[tuple, float] = {}
        self.done = asyncio.ensure_future(self._own())

    @classmethod
    def from_env(cls) -> "TrafficLP":
        """Join the run the deployment's run file describes: its Clock Authority
        at ``clock_endpoint`` and its channel table."""
        run = compass_run.spec()
        if run is None:
            raise RuntimeError(
                f"{compass_run.ENV} is not set: the traffic LP has no run file naming "
                "the Clock Authority and the channel table."
            )
        conn = clock_transport.connect(TRAFFIC, run["clock_endpoint"])
        table = compass_run.channel_table(run)
        # One channel each way: into the frontend (frontend-P, or the router
        # LP, behind a PD router) and back from the one that streams
        # (frontend-D, or the router LP).
        (http,), (stream,) = table.channels_from(TRAFFIC), table.channels_into(TRAFFIC)
        entry = compass_ingress.stated(run)
        return cls(
            LPRuntime(TRAFFIC, table, conn),
            os.environ[ADDRESS_ENV],
            http.name,
            stream.name,
            os.environ.get(PREFILL_URL_ENV) if http.name == PREFILL_HTTP else None,
            os.environ.get(SEND_ORDER_SEED_ENV) or None,
            float(os.environ.get(INTAKE_JITTER_ENV) or INTAKE_JITTER_BYTES),
            (
                None
                if entry is None
                else compass_ingress.Ingress(entry, table.lookahead(http.name))
            ),
            send_order_aiperf(),
        )

    # ---- the clock ClockPacedLoopScheduler paces on ----

    def now(self) -> float:
        return self.rt.now

    def wait(self, t: float, phase_deadline: bool = False) -> asyncio.Future:
        """A clock wait for ``t``, resolved at the grant that reaches it or by
        ``withdraw``.

        A `phase_deadline` is asked for as a daemon deadline while a request
        is open, and a wait due at or past it only once it has fired.
        """
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append((t, fut, phase_deadline))
        self._changed.set()
        return fut

    def withdraw(self, fut: asyncio.Future) -> None:
        """Drop a wait, so it asks the clock for nothing, and resolve it now."""
        self._waiters = [w for w in self._waiters if w[1] is not fut]
        if not fut.done():
            fut.set_result(None)
        self._changed.set()

    # ---- the router's side ----

    def send(self, key: tuple, rank: tuple | None = None) -> None:
        """Take one request for the HTTP channel, ranked `rank` among those
        sent at this instant (by default by `key`); the hold stamps it and
        publishes the stamp (`_stamp_sends`).

        A send after `cancel` closed a request is refused and fails the run: the
        closed request's worker stopped reading, so a stream seq released for it
        later is never reported and the hold would wait for it forever.
        """
        if self.cancelled:
            refused = SendAfterCancel(
                f"request {key} sent after a cancel closed requests "
                f"{sorted(self.cancelled)}: their stream seqs released later on "
                f"{self.stream} are never reported, so the hold would wait forever"
            )
            failed = asyncio.get_running_loop().create_future()
            failed.set_exception(refused)
            self._tasks.append(failed)  # the hold raises it, failing `done`
            self._changed.set()
            raise refused
        self._unstamped.append((key if rank is None else rank, key))
        self._changed.set()

    def route(self, key: tuple, worker: str) -> None:
        """aiperf's router gave request `key` to `worker`, whose connections
        it takes (`conn_pool`); without a pool, nothing."""
        if self.conn_pool is not None:
            self._workers[key] = worker

    def _stamp_sends(self) -> None:
        """Stamp the requests sent at this instant and publish their stamps.

        Called once aiperf has caught up with the clock, before the clock call
        or a cancel, so aiperf has sent every one: they take their seqs in
        their ranks' order, or with a seed in the order of a digest of it and
        each rank, not in the order aiperf sent them in (but see below). A stamp is
        ``(key, arrival, seq)``, then ``"jitter", <bytes>`` with a seed
        (`jitter`), then, behind an ingress, ``"start", <time>``, its arrival
        at the ingress, which takes them in that order (`Ingress.admit`), each
        with the body size its worker reported, then with a connection pool
        ``"new", True`` for one that takes no kept-alive connection, which
        they take in seq order, as aiperf's tasks take them in send order.
        With `aiperf_order` (`SEND_ORDER_ENV`) they take their seqs in the
        order aiperf sent them instead, its credit issue order: #752 DeepSeek-V4.1
        Cell A v2's real client sends a 300 s burst in its credit issue order
        (Spearman +0.73..+0.77 with it in DEPA2 c512's three records, which
        agree with each other at +0.84..+0.89), and that order carries the
        trace: output length -0.21 in all three.
        """
        if self.aiperf_order:
            ordered = self._unstamped
        else:
            ordered = sorted(self._unstamped, key=lambda sent: self._place(sent[0]))
        for rank, key in ordered:
            if self.ingress is None:
                arrival, seq = self.rt.stamp_send(self.http)
            else:
                start, at = self.ingress.admit(self.rt.now, self._bodies.pop(key))
                arrival, seq = self.rt.stamp_send(self.http, at)
                self._arrivals[key] = arrival
            self.open[key] = None
            jitter = self.jitter(rank)
            more = () if jitter is None else ("jitter", jitter)
            if self.conn_pool is not None and not self.conn_pool.take(
                self._workers.get(key), self.rt.now
            ):
                more += ("new", True)
            if self.ingress is not None:
                more += ("start", start)
            self._pub.send_pyobj((key, arrival, seq, *more))
        self._unstamped.clear()

    def _unsized(self) -> list[tuple]:
        """Behind an ingress, the requests sent at this instant whose body
        size no worker has reported yet."""
        if self.ingress is None:
            return []
        return [key for _, key in self._unstamped if key not in self._bodies]

    def _place(self, rank: tuple):
        """Where a send ranked `rank` goes among those at its instant: by its
        rank, or with a seed by a digest of the seed and its rank, which no
        hash seed or platform moves."""
        if self.seed is None:
            return rank
        return hashlib.sha256(repr((self.seed, rank)).encode()).digest(), rank

    def jitter(self, rank: tuple) -> float | None:
        """The intake jitter, in bytes, of a send ranked `rank`: None with no
        seed or a median of 0; else log-normal, its median `jitter_median` and its
        sigma `INTAKE_JITTER_SIGMA`, drawn from a digest of the seed and the
        rank, which no hash seed or platform moves, apart from `_place`'s."""
        if self.seed is None or self.jitter_median == 0:
            return None
        digest = hashlib.sha256(repr((self.seed, "intake", rank)).encode()).digest()
        u = (int.from_bytes(digest[:8], "big") + 0.5) / 2**64
        return self.jitter_median * math.exp(INTAKE_JITTER_SIGMA * _NORMAL.inv_cdf(u))

    async def cancel(self) -> None:
        """aiperf cancels what is in flight: close every open request.

        Once aiperf has caught up with the clock, stamps one cancel per open
        request, in their seqs' order, publishes it for the worker that sent
        the request, with `cancel_url` when set, and returns once every one is
        written, naming those that are not every ``DIAG_S`` wall seconds until
        then. The hold passes each closed request's return before the next
        clock call; its stream events still to come are dropped. Behind an
        ingress a cancel arrives no earlier than its request does.
        """
        fut = asyncio.get_running_loop().create_future()
        self._caught_up.append(fut)
        self._changed.set()
        await fut
        while self._unsized():  # one sent since the hold, as below
            self._changed.clear()
            await self._changed.wait()
        self._stamp_sends()  # none left by the hold; one sent since is closed too
        keys = list(self.open)
        self.open.clear()
        self.cancelled.update(keys)
        for key in keys:  # a cancelled request's connection is closed
            self._workers.pop(key, None)
        self.closed.update(dict.fromkeys(keys))
        to = (self.cancel_url,) if self.cancel_url else ()
        for key in keys:
            if self.ingress is None:
                arrival, seq = self.rt.stamp_send(self.http)
            else:
                at = self.rt.now + self.ingress.lookahead_s
                at = max(at, self._arrivals.pop(key))
                arrival, seq = self.rt.stamp_send(self.http, at)
            self._pub.send_pyobj((key, arrival, seq, "cancel", *to))
        while not self._written.issuperset(keys):
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), DIAG_S)
            except TimeoutError:
                logger.warning(
                    "traffic at %s: cancels of %s not reported written by their "
                    "workers after %s wall seconds; still waiting",
                    self.rt.now,
                    [key for key in keys if key not in self._written],
                    DIAG_S,
                )

    def hold(self, callback):
        """aiperf's return callback, behind the hold."""

        async def held(worker_id, message) -> None:
            credit = message.credit
            key = credit_key(credit.phase, credit.phase_index, credit.id)
            if message.error is not None:
                # Passed now, as upstream. A released final event counts as its
                # return; with none, the key stays open and the finish names it.
                self._returns.put_nowait(lambda: callback(worker_id, message))
                self._held[key] = None
                if self.ingress is not None and key not in self._arrivals:
                    # It failed before its worker sent it, so no size comes.
                    self._bodies.setdefault(key, None)
            else:
                self._held[key] = lambda: callback(worker_id, message)
            self._changed.set()

        return held

    async def subscribed(self, workers: int) -> None:
        """Wait until `workers` stamp subscriptions have reached the publisher."""
        while self._subscribers < workers:
            if (await self._pub.recv())[:1] == b"\x01":
                self._subscribers += 1

    def finish(self) -> None:
        """aiperf has sent everything: ask for time until the ``+inf`` grant."""
        self._finishing = True
        self._changed.set()

    # ---- the owner ----

    async def _read_reports(self) -> None:
        while True:
            self._take(await self._pull.recv_pyobj())

    def _take_queued(self) -> None:
        """Take the reports already queued without a turn of the event loop:
        most reach the socket while the owner is blocked in a clock call."""
        while True:
            try:
                frame = self._pull.recv(zmq.NOBLOCK).result()
            except zmq.Again:
                return
            self._take(pickle.loads(frame))

    def _take(self, report) -> None:
        self._changed.set()
        if len(report) == 1:  # a worker wrote this request's cancel
            self._written.add(report[0])
            return
        if len(report) == 2:  # behind an ingress, a request's body bytes
            key, nbytes = report
            self._bodies[key] = nbytes
            return
        key, seq, arrival, final = report
        # A cancelled request's seq may be dropped already, before its report.
        if key not in self.cancelled:
            self.rt.check_arrival(self.stream, arrival, seq)
        self._reports[seq] = (key, final)
        if final and self.conn_pool is not None:
            self._finals[key] = arrival

    async def _pass_returns(self) -> None:
        while True:
            ret = await self._returns.get()
            self._passing = True
            try:
                await ret()
            except Exception:  # upstream loses that one return, not the rest
                logger.exception("traffic: a return callback raised")
            finally:
                self._passing = False
                self._changed.set()

    async def _own(self) -> None:
        while True:
            await self._settle()
            if not (self.open or self._waiters or self._finishing):
                self._changed.clear()
                self._alive = None
                await self._changed.wait()
                continue
            t, deadline = math.inf, math.inf
            for at, _, phase_deadline in self._waiters:
                if phase_deadline:
                    deadline = min(deadline, at)
                else:
                    t = min(t, at)
            if not self.open:
                t, deadline = min(t, deadline), math.inf
            elif t >= deadline:  # asked for once the deadline has fired
                t = math.inf
            g = self.rt.next_event(t, deadline)
            if g == math.inf:
                return self._close()

    async def _settle(self) -> None:
        """Hold until every released event is reported and aiperf has caught up."""
        waiting = []
        for wait in self._waiters:  # resolve the waits this grant made due
            if wait[0] <= self.rt.now:
                wait[1].set_result(None)
            else:
                waiting.append(wait)
        self._waiters = waiting
        spins = 0
        while True:
            # A reader that raised fails the run instead of leaving the hold waiting.
            for task in self._tasks:
                if task.done():
                    task.result()
            self._take_queued()
            unreported, unreturned = [], []
            # A handled seq leaves the runtime's released set, so a pass scans
            # only the open events, not every event of the run. It stops at the
            # first event it cannot handle yet, so returns pass in stream order
            # whatever order they reach the hold in.
            for seq in sorted(self.rt.released[self.stream]):
                key, final = self._reports.get(seq, (None, False))
                if key is None and self.open:
                    unreported.append(seq)
                    break
                # With no request open, an unreported event is a closed one's.
                if final and key not in self.cancelled:
                    if key not in self._held:
                        unreturned.append(key)
                        break
                    self.open.pop(key, None)
                    self._arrivals.pop(key, None)
                    if self.conn_pool is not None:  # finals pass in stream order
                        self.conn_pool.release(
                            self._workers.pop(key, None),
                            self._finals.pop(key, self.rt.now),
                        )
                    if (ret := self._held.pop(key)) is not None:
                        self._returns.put_nowait(ret)
                self.rt.released[self.stream].discard(seq)
            unreturned += [key for key in self.closed if key not in self._held]
            if self.closed and not unreturned:
                for key in self.closed:
                    if (ret := self._held.pop(key)) is not None:
                        self._returns.put_nowait(ret)
                self.closed.clear()
            # Behind an ingress no request is stamped before its body size is in.
            unsized = self._unsized()
            if unreported or unreturned or self._passing or unsized:
                self._changed.clear()
                self._alive = None
                try:
                    await asyncio.wait_for(self._changed.wait(), DIAG_S)
                except TimeoutError:
                    # An errored return leaves its key open with no return to pass.
                    errored = [k for k, ret in self._held.items() if ret is None]
                    if unreported and errored:
                        raise UnansweredRequests(
                            f"requests {errored} failed mid-stream: released stream "
                            f"seqs {unreported} were not reported after {DIAG_S} wall "
                            "seconds"
                        ) from None
                    if unsized:
                        logger.warning(
                            "traffic at %s: no worker reported the body size of %s "
                            "after %s wall seconds; still waiting",
                            self.rt.now,
                            unsized,
                            DIAG_S,
                        )
                    if unreported or unreturned or self._passing:
                        logger.warning(
                            "traffic at %s: released stream seqs %s not reported, "
                            "returns of %s not received after %s wall seconds; "
                            "still waiting",
                            self.rt.now,
                            unreported,
                            unreturned,
                            DIAG_S,
                        )
                continue
            ready, self._alive, ended = _look(self._alive)
            if not ready and not ended:
                self._stamp_sends()
                if not self._caught_up:
                    return
                for fut in self._caught_up:
                    fut.set_result(None)
                self._caught_up.clear()
                continue
            spins += 1
            if spins == SPIN_DIAG:
                logger.warning("traffic: tasks %s never block; still waiting", ready)
            await asyncio.sleep(0)

    def _close(self) -> None:
        self.rt.close()
        for task in self._tasks:
            task.cancel()
        self._pub.close(linger=0)
        self._pull.close(linger=0)
        if self.open:
            raise UnansweredRequests(
                f"the run finished with requests {list(self.open)} sent but never "
                "given their final response"
            )


def _look(alive: set | None) -> tuple[list[str], set, bool]:
    """One look at this loop's tasks, but the current one.

    Returns the names of those not waiting on a pending future, those not done,
    and whether one of ``alive``, the tasks not done at the last look, has ended
    since. A task that ends queues its done callbacks behind the current task,
    so they run only on the loop's next turn; with ``alive`` None, after turns
    of the loop no look saw, whether one has ended is unknown, and it reads as
    one.
    """
    me = asyncio.current_task()
    tasks = {task for task in asyncio.all_tasks() if task is not me and not task.done()}
    # ponytail: Task._fut_waiter is private; a loop-level idle hook would not be.
    ready = [
        task.get_name()
        for task in tasks
        if task._fut_waiter is None or task._fut_waiter.done()
    ]
    ended = alive is None or any(task.done() for task in alive)
    return ready, tasks, ended
