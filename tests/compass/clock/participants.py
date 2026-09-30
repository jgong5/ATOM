# SPDX-License-Identifier: MIT
"""What each synthetic LP does between grants.

An LP holds its own clock and nothing else. The driver hands it a grant `G` and
the messages the grant released. It handles them one at a time in
`(arrival, channel, seq)` order with its clock at each arrival, then runs at `G`
and returns its next request: `TAR(t)` while a step is being charged, `NER(t)`
while idle until its next local event (infinite when it has none), or `END`.

Only the run at `G` sends, and every send is stamped `G` plus the channel's
lookahead. A handler never sends: in ATOM a receiving thread hands the message
on and the clock owner acts on it.
"""

import collections
import dataclasses
import heapq
import math

from atom.compass.clock import END, NER, TAR, LpId

# Declared, not measured: the tokenizer station of each frontend, and the
# simulated KV transfer between the two engines of a prefill-decode deployment.
TOKENIZER_WIDTH = 2
ENCODE_FIXED_S = 1.0e-4
ENCODE_TOKENS_PER_S = 2.0e6
KV_LATENCY_S = 1.0e-4
KV_BYTES_PER_TOKEN = 70_000
KV_BANDWIDTH_BYTES_PER_S = 50.0e9
KV_NOTIFY_S = 1.0e-4
#: The idle KV drain pace, the value of `KV_IDLE_DRAIN_INTERVAL_S` in
#: `atom/model_engine/engine_core.py`.
KV_IDLE_DRAIN_S = 1.0e-3


@dataclasses.dataclass(frozen=True)
class Workload:
    """The trace a run replays.

    Requests arrive in groups of `len(prompt_tokens)`, one prompt length each.
    The lengths differ, so a group tokenized side by side on a station of width
    two or more still reaches the engine at distinct times.
    """

    requests: int
    prefill_steps: int
    decode_steps: int
    prefill_step_seconds: float
    decode_step_seconds: float
    arrival_interval_seconds: float
    prompt_tokens: tuple[int, ...]
    scrape_interval_seconds: float


#: A measured prior run: 106 prefill and 4,346 decode steps, the last response at
#: about 267 s of modelled time. The prompt lengths and the scrape interval are
#: declared, not taken from that run.
DESIGN_WORKLOAD = Workload(
    requests=106,
    prefill_steps=1,
    decode_steps=41,
    prefill_step_seconds=0.30,
    decode_step_seconds=0.054,
    arrival_interval_seconds=5.0861,
    prompt_tokens=(1000, 3000),
    scrape_interval_seconds=15.0,
)


def scaled(workload: Workload, requests: int, decode_steps: int = 0) -> Workload:
    """The same trace shape, shorter."""
    return dataclasses.replace(
        workload, requests=requests, decode_steps=decode_steps or workload.decode_steps
    )


def transfer_seconds(tokens: int) -> float:
    """The KV write of one request, which both engines compute on their own."""
    return KV_LATENCY_S + tokens * KV_BYTES_PER_TOKEN / KV_BANDWIDTH_BYTES_PER_S


class Station:
    """A FIFO resource station with `width` servers."""

    def __init__(self, width: int) -> None:
        self.free = [0.0] * width  # a heap of the times each server frees

    def admit(self, arrival: float, service: float) -> float:
        """Queue one job and return when it completes."""
        done = max(arrival, heapq.heappop(self.free)) + service
        heapq.heappush(self.free, done)
        return done


class _Lp:
    def __init__(self, name, table):
        self.name = LpId(name)
        self.table = table
        self.clock = 0.0
        self.outbox = []  # (channel, seq, arrival, payload) since the last request
        self.timers = 0  # local timer firings: scrapes and idle drain ticks
        self._seq = collections.Counter()

    def send(self, channel, payload):
        seq = self._seq[channel]
        self._seq[channel] += 1
        arrival = self.clock + self.table.lookahead(channel)
        self.outbox.append((channel, seq, arrival, payload))

    def on_grant(self, g, messages):
        """Step through the released messages at their arrivals, then run at `g`."""
        for arrival, channel, _seq, payload in messages:
            self.clock = arrival
            self.handle(channel, arrival, payload)
        self.clock = g
        return self.run(g)


class Traffic(_Lp):
    """The traffic source, and the metrics observer that scrapes on its clock.

    With `scrape_on` naming its request channel, a scrape is a `GET /metrics`
    answered on the stream. With `None`, the router answers the scrape itself,
    so no LP sees it. After the last response it waits for one more scrape to
    complete and then ends the run.
    """

    def __init__(self, table, workload, http, scrape_on):
        super().__init__("traffic", table)
        self.http = http
        self.scrape_on = scrape_on
        self.expected = workload.requests
        self.interval = workload.scrape_interval_seconds
        self.next_scrape = self.interval
        self.arrivals = collections.deque()
        group = len(workload.prompt_tokens)
        for request in range(workload.requests):
            slot, index = divmod(request, group)
            self.arrivals.append(
                (
                    slot * workload.arrival_interval_seconds,
                    request,
                    workload.prompt_tokens[index],
                )
            )
        self.responses = self.scrapes = self.answered = 0
        self.last_scrape = None

    def handle(self, channel, arrival, payload):
        if payload[0] == "scrape":
            self.answered = payload[1]
        elif payload[2]:  # ("chunk", request, finished)
            self.responses += 1

    def run(self, now):
        while self.arrivals and self.arrivals[0][0] <= now:
            _, request, tokens = self.arrivals.popleft()
            self.send(self.http, ("request", request, tokens))
        if now >= self.next_scrape:
            self.timers += 1
            self.scrapes += 1
            self.next_scrape += self.interval
            if self.last_scrape is None and self.responses == self.expected:
                self.last_scrape = self.scrapes
            if self.scrape_on is None:
                self.answered = self.scrapes
            else:
                self.send(self.scrape_on, ("scrape", self.scrapes))
        if self.finished:
            return END, math.inf
        if self.arrivals:
            return NER, min(self.next_scrape, self.arrivals[0][0])
        return NER, self.next_scrape

    @property
    def finished(self):
        return self.last_scrape is not None and self.answered >= self.last_scrape


class Frontend(_Lp):
    """The API server. Its event loop owns the clock, so it only ever waits in NER.

    Requests go through the tokenizer station and on to the engine when their
    job completes. Engine output goes back on the stream, or, on a prefill
    frontend, the finished prefill is relayed to the decode frontend as its
    request.
    """

    def __init__(self, name, table, to_engine, stream=None, relay=None):
        super().__init__(name, table)
        self.to_engine = to_engine
        self.stream = stream
        self.relay = relay
        self.station = Station(TOKENIZER_WIDTH)
        self.tokenized = []  # a heap of (done, request, tokens)
        self.answers = []  # (channel, payload) to send at the next run

    def handle(self, channel, arrival, payload):
        kind = payload[0]
        if kind == "request":
            _, request, tokens = payload
            service = ENCODE_FIXED_S + tokens / ENCODE_TOKENS_PER_S
            done = self.station.admit(arrival, service)
            heapq.heappush(self.tokenized, (done, request, tokens))
        elif kind == "scrape":
            self.answers.append((self.stream, payload))
        else:
            _, request, tokens, finished = payload
            if self.relay is None:
                self.answers.append((self.stream, ("chunk", request, finished)))
            elif finished:
                self.answers.append((self.relay, ("request", request, tokens)))

    def run(self, now):
        while self.tokenized and self.tokenized[0][0] <= now:
            _, request, tokens = heapq.heappop(self.tokenized)
            self.send(self.to_engine, ("request", request, tokens))
        for channel, payload in self.answers:
            self.send(channel, payload)
        self.answers.clear()
        return NER, self.tokenized[0][0] if self.tokenized else math.inf

    @property
    def finished(self):
        return not (self.tokenized or self.answers)


class Engine(_Lp):
    """One engine LP: every step is a TAR, and an idle engine waits in NER.

    A step serves one request, round robin, and puts its output. On a prefill
    engine a finished request keeps its blocks until the decode side's write
    request has arrived and the write is done at `max(a, r) + T`. On a decode
    engine, admitting a request sends the write request and parks it until
    `a + T + notify`. Both are local events, seen at a step boundary or at an
    idle drain tick, never messages.
    """

    def __init__(self, name, table, output, plan, write_req=None, producer=False):
        super().__init__(name, table)
        self.output = output
        self.plan = plan
        self.write_req = write_req
        self.producer = producer
        self.arrived = []  # (request, tokens) not yet admitted
        self.work = collections.deque()  # (request, tokens, steps left)
        self.step = None
        self.steps = 0
        self.deferred = {}  # request -> (r, tokens): finished, blocks held
        self.write_arrivals = {}  # request -> a
        self.parked = {}  # request -> (ready, tokens)
        self.drain_due = None

    def handle(self, channel, arrival, payload):
        if payload[0] == "write":
            self.write_arrivals[payload[1]] = arrival
        else:
            self.arrived.append(payload[1:])

    def run(self, now):
        boundary = self.step is not None
        if boundary:
            self._finish_step(now)
        tick = self.drain_due is not None and now >= self.drain_due
        if boundary or tick:
            self.timers += tick
            self.drain_due = None
            self._process_completions(now)
        self._admit(now)
        if self.work:
            self.step = self.work.popleft()
            return TAR, now + self.step[2][0]
        if self.deferred or self.parked:
            if self.drain_due is None:
                self.drain_due = now + KV_IDLE_DRAIN_S
            return NER, self.drain_due
        return NER, math.inf

    def _finish_step(self, now):
        request, tokens, left = self.step
        self.step = None
        self.steps += 1
        left.popleft()
        self.send(self.output, ("output", request, tokens, not left))
        if left:
            self.work.append((request, tokens, left))
        elif self.producer:
            self.deferred[request] = (now, tokens)

    def _process_completions(self, now):
        for request, (r, tokens) in list(self.deferred.items()):
            a = self.write_arrivals.get(request)
            if a is None:
                continue
            if r > a:
                raise ValueError(
                    f"{self.name}: request {request} finished at {r}, after its "
                    f"write request arrived at {a}"
                )
            if now >= max(a, r) + transfer_seconds(tokens):
                del self.deferred[request], self.write_arrivals[request]
        for request, (ready, tokens) in list(self.parked.items()):
            if now >= ready:
                del self.parked[request]
                self.work.append((request, tokens, collections.deque(self.plan)))

    def _admit(self, now):
        for request, tokens in self.arrived:
            if self.write_req is None:
                self.work.append((request, tokens, collections.deque(self.plan)))
                continue
            self.send(self.write_req, ("write", request))
            a = now + self.table.lookahead(self.write_req)
            self.parked[request] = (a + transfer_seconds(tokens) + KV_NOTIFY_S, tokens)
        self.arrived.clear()

    @property
    def finished(self):
        return not (
            self.arrived
            or self.work
            or self.step
            or self.deferred
            or self.write_arrivals
            or self.parked
        )
