"""Drive a served engine from a workload whose arrivals are declared, not raced.

`benchmark_serving` sends requests when the wall clock says to. Against a
simulated engine that advances a *virtual* clock by predicted step costs, the
two clocks race: the scheduler batches whatever has arrived by socket when a
step is decided, so the simulated run performs a different set of steps from the
run it stands for. Measured, on one workload: 189 decode steps against 127, in
buckets the real run never visits, and 7-20 prefill steps against 3 -- and two
runs at identical settings disagreed with each other.

So this client declares each request's arrival as an **offset into the run**
rather than delivering it at that moment. Requests are posted as fast as the
socket allows; `compass_arrival` says when the engine should treat each as having
arrived, and `compass_workload_size` tells the arrival barrier how many to expect
so it never advances virtual time past an arrival still in flight. Delivery order
and socket latency then stop mattering, which is the point.

Against a real server there is no start-of-run to offset from, so declared
arrivals are ignored and "now" is used -- the same script measures both sides.

    # synthetic open-loop arrivals
    python scripts/compass/replay.py --port 8006 --num-requests 64 --rate 40 \
        --input-tokens 128 --output-tokens 32 --out replay.json

    # a recorded trace: one JSON object per line, with
    #   {"arrival_s": 0.0, "input_tokens": 512, "output_tokens": 64}
    python scripts/compass/replay.py --port 8006 --trace trace.jsonl --out replay.json

The trace form is the one that matters: a real arrival process is bursty in ways
no rate parameter reproduces, and burstiness is exactly what decides whether
requests batch together.
"""

import argparse
import json
import random
import sys
import threading
import time as _time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


#: Requests to generate when there is no trace to replay. Not a default for the
#: trace path: a trace is a workload someone chose, and quietly keeping its
#: first N is a different workload wearing its name and sha256. This was that
#: default for the trace path once, and a 809-request replay ran 64 of them --
#: 55s of a 5726s timeline -- with the artifact naming the whole file.
_SYNTHETIC_REQUESTS = 64


def _workload(args) -> list[dict]:
    """The requests to send, each with the instant it should count as arriving.

    Sets ``args.trace_rows`` on the trace path, so the manifest can report what
    was on disk beside what ran and truncation cannot be silent.
    """
    if args.trace:
        rows = []
        with open(args.trace, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: float(r.get("arrival_s", 0.0)))
        args.trace_rows = len(rows)
        if args.num_requests:
            rows = rows[: args.num_requests]
        base = float(rows[0].get("arrival_s", 0.0)) if rows else 0.0
        # Scaled here rather than at send time, so the workload written to the
        # output file is the workload that ran. It was scaled at send time and
        # saved unscaled, which meant a 40x compression that turned a 20-second
        # arrival process into a half-second burst left no trace in the
        # artifact -- the run looked paced and was not.
        scale = max(1e-9, float(args.time_scale))
        out = []
        for r in rows:
            row = {"arrival_s": (float(r.get("arrival_s", 0.0)) - base) / scale,
                   "input_tokens": int(r.get("input_tokens", args.input_tokens)),
                   "output_tokens": int(r.get("output_tokens", args.output_tokens))}
            # Carried, not dropped. `hash_ids` decides what the prompt builder
            # shares, so without it a cc-traces replay is the same lengths with
            # none of the reuse; `session` namespaces those ids, which are
            # session-local; and `api_time_s` is the recorded duration, which is
            # the only ground truth that ever reaches the artifact. They were
            # projected away here, so a trace could carry all three and the
            # output file would show no sign they had existed.
            for key in ("hash_ids", "session", "api_time_s", "source_ttft_s",
                        "session_id"):
                if key in r:
                    row[key] = r[key]
            out.append(row)
        return out

    # Poisson arrivals at --rate, or all at zero when the rate is infinite.
    rng = random.Random(args.seed)
    out, t = [], 0.0
    for _ in range(args.num_requests or _SYNTHETIC_REQUESTS):
        out.append({"arrival_s": t,
                    "input_tokens": args.input_tokens,
                    "output_tokens": args.output_tokens})
        if args.rate > 0:
            t += rng.expovariate(args.rate)
    return out


#: Most requests this client will hold open at once. One thread each, so this
#: is a thread count as much as a connection count. Not a tuning knob: quietly
#: posting fewer would reintroduce the deadlock this bounds -- a declared
#: workload is held by the server until all of it has arrived, so a pool
#: smaller than the workload is threads waiting on responses the server will
#: not produce until the threads post more.
#:
#: Raised from 1024 because that refused the workload this harness exists for:
#: 32 sessions at 256k is 3,094 requests, and truncating it to 1,024 would drop
#: 2,070 of them while the artifact still named the whole trace. The cost is
#: one OS thread and one socket per request; at 3,094 that is real but well
#: inside a default `ulimit -n` of 65536, and the stack size is pinned below so
#: the address space cost stays bounded.
MAX_IN_FLIGHT = 8192

#: A profiling window shorter than this is refused. A 250k-token session on
#: this corpus takes minutes of engine time and tens of seconds of think time
#: per turn, so a 60-second window measures lane startup and nothing else --
#: and reports a throughput figure anyway. The golden harness uses the same
#: floor, with a 1800-second default above it.
MIN_BENCHMARK_DURATION_S = 900.0

#: Seconds of system-wide dead air allowed before every pending think timer is
#: shifted forward together. Not a cap on any one gap: see `_IdleGuard`.
SYSTEM_IDLE_GAP_CAP_S = 10.0

#: Bytes of stack per request thread. The default 8 MiB times 3,094 threads is
#: 24 GiB of address space reserved to hold a socket and a dict.
_THREAD_STACK_BYTES = 512 * 1024


def _digest(path):
    """SHA-256 of a file, or None when there is no file to name."""
    if not path:
        return None
    import hashlib

    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _revision():
    """The code this ran as, if the tree is a checkout."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None


def _prompt(row: dict, index: int, marker: str | None = None) -> str:
    """Text of exactly the requested token count, sharing what the row shares.

    Two opposite constructions, picked by whether the trace says anything about
    sharing. See `atom.compass.workload`, which run.py's sweep shares: a sweep
    that cannot target a token count cannot bracket a workload measured in
    tokens.

    Without `hash_ids` the prompt is built to *defeat* the prefix cache, so
    every request pays for its own prefill and a synthetic workload measures
    what it says it measures.

    With `hash_ids` -- which is every cc-traces row -- each id names a 64-token
    block and consecutive turns of a session share their leading ids. On the
    256k corpus 96.2% of all input tokens are a re-send of a block the session
    already sent, so ignoring the ids would not be a small simplification: it
    would be a workload with 26x the prefill work of the one recorded.
    """
    from atom.compass.workload import prompt_of_hash_ids, prompt_of_tokens

    tokens = int(row["input_tokens"])
    ids = row.get("hash_ids")
    if ids:
        return prompt_of_hash_ids(ids, tokens, session=int(row.get("session", 0)),
                                  marker=marker)
    return prompt_of_tokens(tokens, index)


def _sessions(workload: list[dict]) -> list[list[int]]:
    """Indices of each session's rows, sessions in first-arrival order.

    A cc-traces session is the unit a user holds: its turns re-send the whole
    conversation, so they share prefix blocks with each other and with nothing
    else. Splitting one across clients would replay the same token counts with
    none of that sharing.
    """
    order: dict[int, list[int]] = {}
    for i, row in enumerate(workload):
        order.setdefault(int(row.get("session", 0)), []).append(i)
    return [idxs for _, idxs in
            sorted(order.items(), key=lambda kv: workload[kv[1][0]]["arrival_s"])]


def _stream_of(row: dict) -> int:
    """Which chain of its session's tree a row belongs to.

    Rows extracted before `cc_traces.py` carried the id have no `stream`; they
    read as one chain, which is what they were treated as before.
    """
    s = row.get("stream")
    return int(s) if s is not None else 0


def _session_plan(rows: list[dict]) -> tuple[list[list[int]], list[float]]:
    """Which of a session's rows each row waited for, and how long it then waited.

    A session is not a straight line of turns. 43.5% of the corpus's requests
    overlap another request of their own session, because a turn can fan out
    into sub-agents -- and peak in-session concurrency runs from 1 to 23.

    Two different rules produce those edges, and conflating them was a real
    bug. *Within* a chain -- the root conversation, or one sub-agent's
    conversation -- turn k+1 is a reply to turn k and waits for it to come
    back, unconditionally, whatever the two recorded clocks say. *Between*
    chains there is no such relationship, so the recording decides: a row waits
    for each other chain's latest request known to have finished before it
    started, and overlapping intervals create no edge.

    The earlier version here had only the second rule, applied session-wide. On
    a recording where two consecutive root turns overlap by a hair -- different
    clocks, or a retry -- it replayed a conversation as a fan-out, putting two
    turns of the same chain in the server at once. That is load the session
    never had, and it is invisible in the aggregate because the token counts
    are unchanged.

    The cross-chain set is one entry per chain, then pruned of entries another
    kept entry already waits on directly. Pruning changes the payload size and
    not the release instant, which is `max(ends)` either way.

    The gap is kept, not dropped. Between a turn coming back and the next one
    going out sits a user reading the answer and typing, and on this corpus
    that think time is most of the session: 32 sessions whose requests total
    a few hours of engine work span 98,944 seconds end to end. Dropping it
    replaces an agentic workload with a benchmark that hammers -- which is a
    real workload, but not this one, and it cannot answer how many concurrent
    sessions a GPU carries.

    So each row gets `think[k]`: the recorded seconds between the last of its
    predecessors finishing and it starting. A row that waits for nothing
    measures from when its session opened, which is zero for the first row and
    the branch's own offset for a sub-agent that was already running.
    """
    starts = [float(r.get("arrival_s", 0.0)) for r in rows]
    ends = [s + float(r.get("api_time_s") or 0.0) for s, r in zip(starts, rows)]
    raw = [r.get("stream") for r in rows]
    if all(s is None for s in raw):
        # No chain ids at all -- a trace extracted before `cc_traces.py` emitted
        # them. Every row becomes its own chain, which leaves the spine empty
        # and reduces this to the recorded-overlap rule those traces were
        # replayed under. Calling them one chain instead would serialise a
        # genuine sub-agent fan-out into a conversation.
        streams = list(range(len(rows)))
    else:
        streams = [int(s) if s is not None else 0 for s in raw]
    opened = min(starts) if starts else 0.0
    order = sorted(range(len(rows)), key=lambda k: (starts[k], k))

    # The sequential spine: within a chain, the row before this one.
    chains: dict[int, list[int]] = {}
    for k in order:
        chains.setdefault(streams[k], []).append(k)
    spine: dict[int, int] = {}
    for chain in chains.values():
        for earlier, later in zip(chain, chain[1:]):
            spine[later] = earlier

    # Rows are visited in start order, so a row that has finished by one visit
    # has finished by every later one: each moves from `pending` to `best`
    # exactly once, which keeps this linear rather than quadratic. A session
    # runs to 866 rows and a lane re-instantiates one every few minutes, so the
    # quadratic form was minutes of CPU per run.
    pending: dict[int, list[int]] = {}
    best: dict[int, int] = {}
    deps: list[list[int]] = [[] for _ in rows]
    think: list[float] = [0.0] * len(rows)
    for k in order:
        now = starts[k]
        for stream, waiting in pending.items():
            if not waiting:
                continue
            still = []
            for j in waiting:
                # Golden's test, kept exactly: started strictly before, and
                # finished at or before. A zero-width interval starting at this
                # row's own start is not a predecessor.
                if starts[j] < now and ends[j] <= now:
                    held = best.get(stream)
                    if held is None or (starts[j], j) > (starts[held], held):
                        best[stream] = j
                else:
                    still.append(j)
            pending[stream] = still
        cross = sorted(j for s, j in best.items() if s != streams[k])
        keep = [j for j in cross
                if not any(o != j and j in deps[o] for o in cross)]
        own = spine.get(k)
        if own is not None:
            keep = sorted({own, *(j for j in keep
                                  if j != own and j not in deps[own])})
        deps[k] = keep
        base = max((ends[j] for j in keep), default=opened)
        # Floored: a recorded end can sit a hair past a recorded start when the
        # two came from different clocks, and a negative think time would mean
        # a request arriving before the answer it is a reply to.
        think[k] = max(0.0, now - base)
        pending.setdefault(streams[k], []).append(k)
    return deps, think


def _session_span(workload: list[dict], idxs: list[int]) -> float:
    """Recorded seconds from a session's first request starting to its last
    finishing -- think time included, because that is what a slot is occupied
    for."""
    starts = [float(workload[i].get("arrival_s", 0.0)) for i in idxs]
    ends = [s + float(workload[i].get("api_time_s") or 0.0)
            for s, i in zip(starts, idxs)]
    return (max(ends) - min(starts)) if idxs else 0.0


def _tstar_split(rows: list[dict], ratio: float):
    """`(warmup, profiled)` for a session joined `ratio` of the way through it.

    A lane's *first* session does not start at the user's first turn. Real
    sessions are already in progress when a measurement window opens, and one
    that always begins at turn 0 measures a server whose every session is cold
    -- maximum prefill, minimum reuse, and a throughput number well under what
    the same workload gives in steady state.

    So t* is sampled across the recorded session and everything before it is
    skipped, except for the single turn immediately preceding it. That turn is
    sent unmeasured: it puts the conversation's prefix in the cache, which is
    the state a session joined mid-flight is actually in. Without it the first
    profiled turn pays a full 250k-token prefill that the recording says was a
    re-send.

    Recycled sessions get no t* -- a lane that finishes one session and takes
    another is watching that one from its first turn, and that is what
    `ratio <= 0` means.
    """
    if not rows:
        return None, []
    if ratio <= 0.0:
        return None, list(range(len(rows)))
    starts = [float(r.get("arrival_s", 0.0)) for r in rows]
    tstar = min(starts) + float(ratio) * (max(starts) - min(starts))
    profiled = [k for k in range(len(rows)) if starts[k] >= tstar]
    earlier = [k for k in range(len(rows)) if starts[k] < tstar]
    if not profiled:
        return None, list(range(len(rows)))
    warmup = max(earlier, key=lambda k: (starts[k], k)) if earlier else None
    return warmup, profiled


def _rid(rng) -> str:
    """A name for one session instance, 48 bits as 12 hex digits."""
    return "%012x" % rng.getrandbits(48)


class _Sampler:
    """Which recorded session a lane takes next.

    A duration-bounded run draws from the whole pool repeatedly rather than
    dealing a fixed slice of it, so the pool's size stops setting the run's
    length: a 16-session corpus and a 393-session corpus both run for the
    duration asked for, and differ in how much they repeat. Seeded, so a rung
    can be re-run.
    """

    def __init__(self, count: int, seed: int, strategy: str = "shuffle"):
        self.count = max(1, int(count))
        self.strategy = strategy
        self._rng = random.Random(seed)
        self._order: list[int] = []
        self._at = 0
        self.drawn: list[int] = []

    def draw(self) -> int:
        if self.strategy == "random":
            g = self._rng.randrange(self.count)
        else:
            if self._at >= len(self._order):
                self._order = list(range(self.count))
                if self.strategy == "shuffle":
                    self._rng.shuffle(self._order)
                self._at = 0
            g = self._order[self._at]
            self._at += 1
        self.drawn.append(g)
        return g


class _IdleGuard:
    """Shift every pending think timer when the whole system goes quiet.

    The corpus's think times are most of its wall clock -- 32 sessions with a
    few hours of engine work in them span 98,944 seconds -- so a faithful
    replay spends nearly all of a fixed window asleep, and a 1800-second run
    would measure a handful of requests.

    Capping each gap individually is the wrong fix: it changes the workload,
    because a gap that overlaps another lane's work is load-shaping and a gap
    with nothing behind it is dead air. So the cap is on the *system*. While
    anything is in flight, every timer runs at its recorded length. Only when
    nothing is on the wire anywhere and every live lane is waiting does this
    subtract the same amount from all of them at once, bringing the soonest
    within `cap` seconds. Shifting all of them by one delta preserves every
    relative gap, so the arrival process between lanes is the recorded one,
    slid forward.

    Two consecutive quiet observations are required before a shift, because a
    thread between finishing a wait and opening its socket is momentarily
    neither sleeping nor in flight, and shifting on that instant would cut a
    gap the recording had work in.

    `shifted_s` and `shifts` go into the artifact: a run that shifted most of
    its clock away is a legitimate measurement of a saturated server and not a
    replay of this corpus's arrival process, and only these two numbers say
    which one happened.
    """

    def __init__(self, cap: float, period: float = 0.25):
        self.cap = float(cap or 0.0)
        self.period = period
        self.cv = threading.Condition()
        self.in_flight = 0
        self.shifted_s = 0.0
        self.shifts = 0
        self._deadlines: dict[int, float] = {}
        self._next = 0
        self._quiet = 0
        self._stop = False
        self._thread = None
        #: When the profiling window closes, and how many executions it cut.
        #: A lane has to be interruptible mid-session: this corpus's recorded
        #: sessions span 2.4 hours at the median and 98 hours at the longest,
        #: so a loop that only consults the clock between whole instances runs
        #: for one recorded session rather than for --benchmark-duration.
        self.expires_at = None
        self.cut = 0

    def expire_at(self, when: float) -> None:
        with self.cv:
            self.expires_at = when
            self.cv.notify_all()

    def expired(self) -> bool:
        return (self.expires_at is not None
                and _time.monotonic() >= self.expires_at)

    def start(self):
        if self.cap > 0:
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        with self.cv:
            self._stop = True
            self.cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def sending(self):
        return _Sending(self)

    def sleep(self, seconds: float) -> bool:
        """Wait out a recorded think time. False if the window closed first.

        A think time is never shortened to fit the window -- that would be the
        inter-turn cap the scenario forbids. It is abandoned: the turn behind
        it is simply not sent, and the run ends with that lane mid-session,
        which is what a fixed-duration benchmark of a days-long recording is.
        """
        if seconds <= 0:
            return not self.expired()
        with self.cv:
            token = self._next
            self._next += 1
            self._deadlines[token] = _time.monotonic() + seconds
            try:
                while not self._stop:
                    now = _time.monotonic()
                    if self.expires_at is not None and now >= self.expires_at:
                        return False
                    left = self._deadlines[token] - now
                    if left <= 0:
                        break
                    if self.expires_at is not None:
                        left = min(left, self.expires_at - now)
                    self.cv.wait(left)
            finally:
                self._deadlines.pop(token, None)
                self.cv.notify_all()
        return not self.expired()

    def _watch(self) -> None:
        with self.cv:
            while not self._stop:
                self.cv.wait(self.period)
                if self._stop:
                    break
                if self.in_flight or not self._deadlines:
                    self._quiet = 0
                    continue
                self._quiet += 1
                if self._quiet < 2:
                    continue
                slack = (min(self._deadlines.values()) - _time.monotonic()
                         - self.cap)
                if slack <= 0:
                    continue
                for token in list(self._deadlines):
                    self._deadlines[token] -= slack
                self.shifted_s += slack
                self.shifts += 1
                self._quiet = 0
                self.cv.notify_all()


class _Sending:
    """Marks the window a request is actually on the wire, for `_IdleGuard`."""

    def __init__(self, guard):
        self.guard = guard

    def __enter__(self):
        with self.guard.cv:
            self.guard.in_flight += 1
        return self

    def __exit__(self, *exc):
        with self.guard.cv:
            self.guard.in_flight -= 1
            self.guard.cv.notify_all()
        return False


def _instance(workload, idxs, *, eid0, lane, instance, session, marker, ratio,
              previous=()):
    """One session played by one lane, as a list of executions.

    An execution is a *dispatch*, not a trace row: a duration-bounded run sends
    the same row many times, under different markers, on different lanes. The
    list this returns is what both sides index into, and what `dag_sha256`
    names -- so `results[].index` stays a stable key across the pair while
    `results[].row` says which trace row it was.
    """
    rows = [workload[i] for i in idxs]
    warm_k, profiled = _tstar_split(rows, ratio)
    keep = sorted({*([warm_k] if warm_k is not None else []), *profiled})
    deps, think = _session_plan([rows[k] for k in keep])
    warm_pos = keep.index(warm_k) if warm_k is not None else None
    execs = []
    for pos, k in enumerate(keep):
        after = [eid0 + j for j in deps[pos]]
        wait = think[pos]
        if not after:
            # The instance's opening request waits on the lane, not on the
            # trace. The gap between two recorded sessions belongs to two
            # different users and says nothing about when a lane takes new work.
            after = [int(j) for j in previous]
            wait = 0.0
        elif warm_pos is not None and pos != warm_pos and after == [eid0 + warm_pos]:
            # The first profiled turn. Its recorded think time is dropped so
            # every lane's t* lands at the same instant, which is what makes
            # the warmup boundary a boundary. One turn per lane per run.
            wait = 0.0
        execs.append({
            "eid": eid0 + pos,
            "row": idxs[k],
            "lane": int(lane),
            "instance": int(instance),
            "session": int(session),
            "stream": _stream_of(rows[k]),
            "marker": marker,
            "phase": "warmup" if pos == warm_pos else "profile",
            "deps": after,
            "think_s": round(wait, 6),
            "think_recorded_s": round(think[pos], 6),
        })
    return execs


def _terminals(execs) -> list[int]:
    """Executions of an instance nothing else in it waits for -- the join a
    lane's next instance hangs off."""
    waited = {d for e in execs for d in e["deps"]}
    return [e["eid"] for e in execs if e["eid"] not in waited]


def _plan_digest(plan) -> str:
    """A name for the executed schedule, so two artifacts can be shown to have
    run the same one. Two sides that drew different sessions still produce a
    well-formed paired report, and it would be comparing different runs."""
    import hashlib

    canonical = json.dumps(
        [{k: e[k] for k in ("eid", "row", "lane", "instance", "session",
                            "stream", "phase", "marker", "deps", "think_s")}
         for e in plan],
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _run_executions(execs, send, out, guard) -> set:
    """Issue one instance's executions on a real clock, sleeping think time.

    A thread is started when a row's predecessors are done, not before, so the
    live thread count is the session's own concurrency -- at most 23 on this
    corpus -- rather than its request count. The earlier version started one
    thread per row up front, which is 866 threads for the longest session and
    ~13,000 across 16 lanes.

    An edge pointing outside this batch is already satisfied: it is either the
    warmup turn, which the boundary waited for, or the previous instance on
    this lane, which this lane ran to completion before calling here.

    Returns the executions the profiling window closed on before they were
    sent. The DAG is still walked to the end so the lane drains, but nothing
    past the deadline goes on the wire.
    """
    by_eid = {e["eid"]: e for e in execs}
    waiting = {e["eid"]: {d for d in e["deps"] if d in by_eid} for e in execs}
    successors: dict[int, list[int]] = {}
    for e in execs:
        for d in e["deps"]:
            if d in by_eid:
                successors.setdefault(d, []).append(e["eid"])

    lock = threading.Lock()
    live = [0]
    cut = set()
    drained = threading.Event()

    def go(eid: int) -> None:
        e = by_eid[eid]
        try:
            if guard.sleep(e["think_s"]):
                out[eid] = send(e)
            else:
                with lock:
                    cut.add(eid)
        finally:
            ready = []
            with lock:
                for s in successors.get(eid, ()):
                    waiting[s].discard(eid)
                    if not waiting[s]:
                        ready.append(s)
                live[0] += len(ready) - 1
                empty = live[0] == 0
            for s in ready:
                threading.Thread(target=go, args=(s,), daemon=True).start()
            if empty:
                drained.set()

    roots = [e["eid"] for e in execs if not waiting[e["eid"]]]
    if not roots:
        return cut
    live[0] = len(roots)
    for eid in roots:
        threading.Thread(target=go, args=(eid,), daemon=True).start()
    drained.wait()
    return cut


def _warmup(plan, send, out) -> list[dict]:
    """Dispatch every unmeasured turn at once and wait. Returns root failures.

    All lanes together, because the point of the phase is that profiling starts
    from one instant with every session's prefix already resident. A sub-agent
    warmup that fails costs one branch's reuse; a root one means that lane
    profiles a cold 250k-token conversation, which is the state this phase
    exists to avoid, so the run is refused rather than reported.
    """
    warm = [e for e in plan if e["phase"] == "warmup"]
    if not warm:
        return []
    threads = [threading.Thread(
        target=lambda e=e: out.__setitem__(e["eid"], send(e)), daemon=True)
        for e in warm]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [e for e in warm if e["stream"] == 0
            and not (out.get(e["eid"]) or {}).get("ok")]


def _recycle(workload, groups, *, clients, duration, instances, sampler, rng,
             guard, send, out, startup: bool):
    """Run `clients` lanes for `duration` seconds, recycling as they finish.

    A lane is one live agent session tree. It runs a whole session -- root
    turns and sub-agents together -- to the end, then immediately draws
    another. Sub-agents run inside the parent's lane and do not take a lane of
    their own, so instantaneous in-flight requests can exceed `clients`; that
    is the workload, not an error.

    Bounded by the clock and not by a pool. Dealing a fixed slice of sessions
    to slots made the pool's size decide the run's length and left lanes idle
    once their slice ran out -- c16 on a 16-session trace realised 9.1 lanes of
    16, which was then reported as an instrumented fact about the workload
    rather than as an artifact of the deal.

    Returns `(plan, began, ended)`: the schedule as executed, and the profiling
    window. The schedule is what the modelled side replays -- it cannot recycle
    on its own, because the arrival barrier holds every request until the whole
    declared workload has arrived and so needs a fixed graph up front.
    """
    plan: list[dict] = []
    lock = threading.Lock()

    def instantiate(lane, instance, ratio, previous):
        with lock:
            g = sampler.draw()
            execs = _instance(workload, groups[g], eid0=len(plan), lane=lane,
                              instance=instance, session=g, marker=_rid(rng),
                              ratio=ratio, previous=previous)
            plan.extend(execs)
        return execs

    initial = [instantiate(c, 0,
                           rng.random() if startup else 0.0, [])
               for c in range(clients)]
    failures = _warmup([e for ex in initial for e in ex], send, out)
    if failures:
        raise SystemExit(
            f"{len(failures)} of {clients} lanes failed their root warmup "
            f"turn, so they would profile a cold conversation the recording "
            f"says was a re-send. First: "
            f"{(out.get(failures[0]['eid']) or {}).get('error')}")

    began = _time.monotonic()
    if duration:
        guard.expire_at(began + duration)
    cut = set()

    def lane(c: int) -> None:
        execs = [e for e in initial[c] if e["phase"] == "profile"]
        instance = 0
        while True:
            dropped = _run_executions(execs, send, out, guard)
            with lock:
                cut.update(dropped)
            instance += 1
            if instances and instance >= instances:
                return
            if duration and _time.monotonic() - began >= duration:
                return
            if not duration and not instances:
                return
            execs = instantiate(c, instance, 0.0, _terminals(execs))

    threads = [threading.Thread(target=lane, args=(c,), daemon=True)
               for c in range(clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # The plan is the schedule as *executed*. An execution the window closed on
    # was never sent, so leaving it in would hand the modelled side a request
    # the real side does not have, and `_workloads_agree` would then refuse the
    # pair on a difference that is nothing but where the clock ran out. Only
    # successors can be cut -- a dependent waits for its predecessor -- so no
    # surviving execution is left pointing at a dropped one.
    guard.cut = len(cut)
    return [e for e in plan if e["eid"] not in cut], began, _time.monotonic()


def _replay_schedule(plan, send, out, guard) -> None:
    """Walk a recorded schedule on a real clock, lane by lane.

    The paired half of `_recycle`. Both sides must run the same executions in
    the same order, and a second recycling run would draw a different sequence
    -- the sampler is seeded, but the *number* of instances a lane gets through
    is decided by how fast the server answered.
    """
    lanes: dict[int, dict[int, list[dict]]] = {}
    for e in plan:
        lanes.setdefault(e["lane"], {}).setdefault(e["instance"], []).append(e)
    _warmup(plan, send, out)

    def lane(c: int) -> None:
        for instance in sorted(lanes[c]):
            batch = [e for e in lanes[c][instance] if e["phase"] != "warmup"]
            if batch:
                _run_executions(batch, send, out, guard)

    threads = [threading.Thread(target=lane, args=(c,), daemon=True)
               for c in sorted(lanes)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _send(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # The body says which field the server objected to; the status alone
        # does not, and "64 requests failed with 400" is not a diagnosis.
        try:
            detail = exc.read().decode()[:400]
        except Exception:  # noqa: BLE001
            detail = ""
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _cache_stats(base: str, timeout: float) -> dict:
    """Cumulative prefix-cache counters, or {} on a server without them.

    Read either side of the run and differenced, this is the check that the
    replay actually reproduced the trace's reuse. It matters because failing to
    is silent: prompts that share no blocks still complete, still report 0
    failed, and just do 26x the prefill the recorded workload did. The trace's
    own `reusable_tokens` (see `cc_traces.py --out ...meta.json`) is what the
    delta should be compared against.
    """
    try:
        with urllib.request.urlopen(base + "/debug/cache_stats",
                                    timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001 - absent on a server built without it
        return {}


def _served_model(base: str, timeout: float) -> str | None:
    """Ask the server what it is serving.

    The completions endpoint validates the model name, so a wrong one fails
    every request identically and looks like a transport problem.
    """
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=timeout) as resp:
            listing = json.loads(resp.read())
        return listing["data"][0]["id"]
    except Exception:  # noqa: BLE001 - fall back to whatever was passed
        return None

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", default=None,
                   help="defaults to whatever /v1/models reports")
    p.add_argument("--trace", help="JSONL of {arrival_s, input_tokens, output_tokens}")
    p.add_argument("--num-requests", type=int, default=None,
                   help="how many requests to send. With --trace the default "
                        "is the whole trace; this only ever truncates it, and "
                        "the artifact records that it did. Without a trace it "
                        "is how many synthetic requests to generate "
                        f"(default {_SYNTHETIC_REQUESTS})")
    p.add_argument("--rate", type=float, default=0.0,
                   help="Poisson arrivals per second; 0 means all arrive at once")
    p.add_argument("--input-tokens", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--out", required=True)
    p.add_argument("--pace", action="store_true",
                   help="drive the timeline from this client's own clock "
                        "instead of declaring it to the engine. Use against a "
                        "real engine, in either mode: a real clock discards a "
                        "declared arrival, so without this the real side "
                        "answers a burst while the simulated side answers the "
                        "trace. Open loop it sleeps until each arrival; with "
                        "--clients it sleeps each session's think time")
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="divide every arrival offset by this, to replay a "
                        "long trace in less time. 1.0 keeps the trace's own "
                        "timing; it changes how much requests batch, so it is "
                        "a property of the workload and not a free knob")
    p.add_argument("--ignore-eos", action="store_true",
                   help="generate exactly --output-tokens rather than at most. "
                        "A trace records how many tokens a request produced, "
                        "and max_tokens is only a ceiling: a row asking for 376 "
                        "may stop at 3 on an EOS the recorded run never hit, "
                        "and the replay quietly stops being the workload while "
                        "still reporting 0 failed")
    p.add_argument("--check-lengths", action="store_true",
                   help="compare the server's reported prompt_tokens against "
                        "what was asked for, and warn if they differ")
    p.add_argument("--clients", type=int, default=0,
                   help="run as this many concurrent lanes instead of "
                        "replaying the trace's arrivals. A lane is one live "
                        "agent session tree: it holds a session until every "
                        "request in it -- sub-agents included -- has finished, "
                        "then immediately draws another. Sub-agents run inside "
                        "the parent's lane, so in-flight requests can exceed "
                        "this number. The recorded in-session overlap and the "
                        "think time between turns are both kept; the gap "
                        "between sessions is dropped, because it belongs to a "
                        "different user. Add --pace against a real engine")
    p.add_argument("--benchmark-duration", type=float, default=0.0,
                   help="seconds of profiling to run, lanes recycling "
                        "throughout. This, and not the size of the trace, is "
                        "what sets the length of the run: a small pool is "
                        "replayed more times rather than leaving lanes idle. "
                        f"Refused below {MIN_BENCHMARK_DURATION_S:.0f}s, which "
                        "is too short for a 250k-token session to complete "
                        "even once. 0 means bound by --sessions-per-client "
                        "instead, which is the fixed-pool mode")
    p.add_argument("--sessions-per-client", type=int, default=0,
                   help="cap each lane at this many sessions. 0 with "
                        "--benchmark-duration means the clock decides; 0 with "
                        "neither means one session per lane")
    p.add_argument("--sampler", choices=("shuffle", "sequential", "random"),
                   default="shuffle",
                   help="how a recycling lane draws its next session. Seeded "
                        "from --seed, so a rung re-runs identically")
    p.add_argument("--startup-sampling", choices=("uniform", "none"),
                   default="uniform",
                   help="uniform: each lane's first session joins at a random "
                        "point in its recording, with the single preceding "
                        "turn sent unmeasured to warm its prefix -- which is "
                        "the state a real session in progress is in. none: "
                        "every lane starts at turn 0, so every session in the "
                        "window is cold and the run understates steady state")
    p.add_argument("--idle-gap-cap", type=float,
                   default=SYSTEM_IDLE_GAP_CAP_S,
                   help="seconds of system-wide dead air to allow before every "
                        "pending think timer is shifted forward by the same "
                        "amount. Only fires when nothing is in flight anywhere, "
                        "so a gap overlapping another lane's work is never "
                        "touched. 0 disables it, and a faithful replay of this "
                        "corpus then spends nearly all of its window asleep")
    p.add_argument("--schedule",
                   help="a prior run's artifact, whose executed schedule this "
                        "run repeats. A recycling loop cannot be declared to "
                        "the engine -- the arrival barrier needs the whole "
                        "graph before it starts -- so the modelled side of a "
                        "pair replays the real side's schedule from here")
    args = p.parse_args()

    workload = _workload(args)
    if not workload:
        print("empty workload", file=sys.stderr)
        return 2
    base = f"http://{args.host}:{args.port}"
    model = args.model or _served_model(base, args.timeout)
    if model is None:
        print("could not determine the served model; pass --model",
              file=sys.stderr)
        return 2

    # A closed loop has no arrival process to declare up front: a request goes
    # out because the one before it came back, and how long that took is what
    # the run is measuring. The two sides answer that differently.
    #
    # Real (--pace): the client owns the loop. It holds each request until its
    # predecessors' responses are in hand, sleeps the recorded think time, and
    # sends. A real clock needs nothing from the server.
    #
    # Modelled (no --pace): the client cannot own the loop, because the wall
    # clock it would sleep on and the virtual clock the engine advances race --
    # a millisecond of round trip can be seconds of simulated time, and the
    # request lands after work the loop meant it to precede. So the whole graph
    # is declared instead and the engine resolves each arrival as the requests
    # it waits on finish. That needs every request posted up front, which is
    # what `compass_workload_size` and the arrival barrier are for.
    closed = args.clients > 0 or bool(args.schedule)
    declared_loop = closed and not args.pace
    duration = float(args.benchmark_duration or 0.0)
    if duration and duration < MIN_BENCHMARK_DURATION_S:
        print(f"--benchmark-duration {duration:g} is under the "
              f"{MIN_BENCHMARK_DURATION_S:.0f}s floor: a 250k-token session on "
              f"this corpus does not complete once inside it, so the window "
              f"would measure startup and nothing else", file=sys.stderr)
        return 2
    if duration and declared_loop and not args.schedule:
        print("--benchmark-duration needs --pace. Lanes recycling on a clock "
              "cannot be declared to the engine: the arrival barrier holds "
              "every request until the whole workload has arrived, so it needs "
              "a fixed graph up front. Run the real side paced, then give this "
              "side its artifact with --schedule.", file=sys.stderr)
        return 2

    groups = _sessions(workload) if closed else []
    if closed and not groups:
        print("closed loop has no sessions to run", file=sys.stderr)
        return 2

    guard = _IdleGuard(args.idle_gap_cap if args.pace else 0.0).start()
    plan: list[dict] = []
    out: dict = {}

    def one(execution):
        """One dispatch: sleep already done, arrival already decided."""
        i = execution["row"]
        row = workload[i]
        eid = execution["eid"]
        body = {
            "model": model,
            "prompt": _prompt(row, i, execution.get("marker")),
            "max_tokens": row["output_tokens"],
            "temperature": 0.0,
        }
        if args.ignore_eos:
            body["ignore_eos"] = True
        if declared_loop:
            # The same graph the paced side walks with threads, stated so the
            # engine can walk it on its own clock. Ids are execution ids, not
            # row indices: a duration-bounded run sends the same row many
            # times and the graph has to name each dispatch separately.
            body["compass_workload_size"] = len(plan)
            body["compass_relative_arrival"] = {
                "id": str(eid),
                "after": [str(j) for j in execution["deps"]],
                "think_s": execution["think_s"],
            }
        try:
            with guard.sending():
                response = _send(base + "/v1/completions", body, args.timeout)
            return {"index": eid, "row": i, "ok": True, "response": response}
        except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
            return {"index": eid, "row": i, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}

    def open_one(i_row):
        i, row = i_row
        at = row["arrival_s"]  # already scaled when the workload was built
        if args.pace:
            # Hold the request until its moment really comes round. Against a
            # real engine this is what makes the arrival process real: the
            # queue is genuinely empty between arrivals, which declaring an
            # arrival cannot achieve -- a declared workload is posted up front
            # and sits in `waiting`, so the scheduler always sees a full queue
            # however the arrivals are stamped.
            delay = at - (_time.monotonic() - began)
            if delay > 0:
                _time.sleep(delay)
        body = {
            "model": model,
            "prompt": _prompt(row, i),
            "max_tokens": row["output_tokens"],
            "temperature": 0.0,
            "compass_workload_size": len(workload),
        }
        if args.ignore_eos:
            body["ignore_eos"] = True
        if not args.pace:
            # Declared rather than delivered: against a simulated engine the
            # wall clock and the virtual clock race, so the arrival is stated
            # and the engine honours it. Ignored by a server on a real clock,
            # which is why --pace exists for that side.
            body["compass_arrival"] = at
        try:
            return {"index": i, "row": i, "ok": True,
                    "response": _send(base + "/v1/completions", body, args.timeout)}
        except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
            return {"index": i, "row": i, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}

    def _refuse_oversized(workers: int) -> None:
        # The server holds every declared request until all of them have
        # arrived, so all of them must be in flight at once. A pool of 64
        # against a 300-request workload is a deadlock -- 64 threads each
        # blocked on a response the server will not produce until 300 have been
        # posted. It resolves only when the arrival barrier times out, and then
        # the run is not the workload that was asked for: requests enter as
        # earlier ones complete, which is not the declared arrival process.
        # This happened, went unnoticed because the client still reported
        # "0 failed", and a day's conclusions were drawn from the result.
        if workers > MAX_IN_FLIGHT:
            raise SystemExit(
                f"{workers} requests needs {workers} concurrent connections, "
                f"over the {MAX_IN_FLIGHT} this client will open. A declared "
                f"workload cannot be posted in batches -- the server waits for "
                f"all of it before it starts -- so this needs a bulk "
                f"submission endpoint rather than a larger pool. Use "
                f"--num-requests to bound the workload meanwhile.")

    cache_before = _cache_stats(base, args.timeout)
    threading.stack_size(_THREAD_STACK_BYTES)
    began = _time.monotonic()
    profile_began = profile_ended = began
    rng = random.Random(args.seed)
    sampler = _Sampler(len(groups), args.seed, args.sampler) if closed else None

    if args.schedule:
        # Both halves of a pair run the same executions in the same order. A
        # second recycling run would not: the sampler is seeded, but how many
        # instances a lane gets through is decided by how fast the server
        # answered, so the two sides would draw different sessions and
        # `dag_sha256` would refuse every rung.
        with open(args.schedule, encoding="utf-8") as fh:
            prior = json.load(fh)
        plan = prior.get("plan") or prior
        if not isinstance(plan, list) or not plan:
            print(f"{args.schedule} carries no executed schedule",
                  file=sys.stderr)
            return 2
        want = (prior.get("run") or {}).get("trace_sha256")
        got = _digest(args.trace)
        if want and got and want != got:
            print(f"--schedule was recorded against trace {want[:12]} and this "
                  f"run has {got[:12]}: the row indices in it name different "
                  f"requests", file=sys.stderr)
            return 2
        profile_began = _time.monotonic()
        if declared_loop:
            _refuse_oversized(len(plan))
            with ThreadPoolExecutor(max_workers=max(1, len(plan))) as pool:
                for r in pool.map(one, plan):
                    out[r["index"]] = r
        else:
            _replay_schedule(plan, one, out, guard)
        profile_ended = _time.monotonic()
    elif closed:
        plan, profile_began, profile_ended = _recycle(
            workload, groups, clients=args.clients, duration=duration,
            instances=int(args.sessions_per_client or 0), sampler=sampler,
            rng=rng, guard=guard, send=one, out=out,
            startup=(args.startup_sampling == "uniform"))
    else:
        _refuse_oversized(len(workload))
        with ThreadPoolExecutor(max_workers=max(1, len(workload))) as pool:
            for r in pool.map(open_one, enumerate(workload)):
                out[r["index"]] = r
        profile_ended = _time.monotonic()
    guard.stop()

    results = [out[k] for k in sorted(out) if out[k] is not None]
    if not results:
        print("nothing executed", file=sys.stderr)
        return 2
    failed = [r for r in results if not r["ok"]]
    phase_of = {e["eid"]: e["phase"] for e in plan}
    profiled = [r for r in results
                if phase_of.get(r["index"], "profile") == "profile"]

    # What the server says it received, against what was asked for. The builder
    # is exact by construction under a tokenizer giving one token per word in
    # `WORDS`; this checks the assumption held, and costs nothing because the
    # count is already in every response.
    length_check = "not requested"
    if args.check_lengths:
        off = []
        for r in results:
            if not r["ok"]:
                continue
            got = (r["response"].get("usage") or {}).get("prompt_tokens")
            want = workload[r["row"]]["input_tokens"]
            if got is not None and got != want:
                off.append((want, got))
        if off:
            worst = max(off, key=lambda pair: abs(pair[1] - pair[0]))
            length_check = f"{len(off)} of {len(results)} wrong"
            print(f"  WARNING: {len(off)} of {len(results)} prompts were not the "
                  f"requested length; worst asked {worst[0]} got {worst[1]}",
                  file=sys.stderr)
        else:
            length_check = "passed"
            print(f"  prompt lengths verified against the server for "
                  f"{len(results) - len(failed)} requests")
    cache_after = _cache_stats(base, args.timeout)
    engine = {}
    try:
        engine = _send(base + "/compass/requests", {}, args.timeout)
    except Exception:  # noqa: BLE001 - a real server has no such endpoint
        try:
            with urllib.request.urlopen(base + "/compass/requests",
                                        timeout=args.timeout) as resp:
                engine = json.loads(resp.read())
        except Exception:  # noqa: BLE001
            engine = {}

    lanes = sorted({e["lane"] for e in plan})
    instances = {c: len({e["instance"] for e in plan if e["lane"] == c})
                 for c in lanes}
    # Everything needed to say what this run was, next to what it produced. A
    # result whose arrival process, calibration or code revision cannot be
    # recovered from its own artifact is not reproducible, and one of these
    # runs was read as a measurement for a day after its arrival protocol had
    # silently failed.
    manifest = {
        "revision": _revision(),
        "paced": bool(args.pace) and not closed,
        # A closed-loop artifact must be able to say so on its own. Its
        # `arrival_span_s` is the trace's, not the run's, and reading one as an
        # open-loop replay would attribute the trace's arrival rate to a run
        # that ignored it.
        "closed_loop": closed,
        # Which executor ran the graph. The paced side sleeps the think time on
        # a real clock; the declared side hands the graph to the engine. An
        # artifact that cannot say which is not interpretable.
        "closed_loop_mode": ("paced" if closed and not declared_loop
                             else "declared" if closed else None),
        "clients": len(lanes),
        "sessions_run": len({(e["lane"], e["instance"]) for e in plan}),
        "sessions_per_client": [instances[c] for c in lanes] or None,
        # Which recorded sessions the lanes actually drew, in order. A
        # duration-bounded run does not deal a fixed slice, so this is the only
        # record of what was replayed and how often.
        "sessions_drawn": (sampler.drawn if sampler and not args.schedule
                           else None),
        "sampler": args.sampler if closed else None,
        "startup_sampling": args.startup_sampling if closed else None,
        "schedule": args.schedule,
        # A name for the whole executed schedule. Two sides that drew different
        # sessions still produce a well-formed paired report -- of two
        # different experiments. Comparing this field is how that is caught.
        "dag_sha256": _plan_digest(plan) if plan else None,
        "benchmark_duration_s": duration or None,
        "profile_window_s": round(profile_ended - profile_began, 3),
        "warmup_requests": sum(1 for e in plan if e["phase"] == "warmup"),
        "profile_requests": len(profiled),
        # How much of the corpus's dead air was skipped, and in how many jumps.
        # A run that shifted most of its clock away is a real measurement of a
        # saturated server and not a replay of this arrival process; these two
        # numbers are what say which one happened.
        "idle_shifts": guard.shifts,
        "idle_shifted_s": round(guard.shifted_s, 3),
        "idle_gap_cap_s": guard.cap or None,
        # Turns whose think time was still running when the window closed. A
        # fixed-duration benchmark of a days-long recording ends with every
        # lane mid-session, so this is expected to be nonzero and is reported
        # rather than silently pruned: zero here on a long trace would mean the
        # lanes were not cut but ran a whole recorded session each.
        "executions_cut_at_deadline": guard.cut,
        "think_time_s": (round(sum(e["think_s"] for e in plan), 3)
                         if plan else None),
        "requests_executed": len(results),
        "time_scale": float(args.time_scale),
        "requests": len(workload),
        "arrival_span_s": (round(workload[-1]["arrival_s"], 6)
                           if workload else 0.0),
        "trace": args.trace,
        "trace_sha256": _digest(args.trace),
        # What was on disk, beside what ran. The sha256 above names the whole
        # file however few of its rows were sent, so without these two a
        # truncated replay is indistinguishable from a complete one.
        "trace_rows": getattr(args, "trace_rows", None),
        "trace_truncated": (args.trace is not None
                            and getattr(args, "trace_rows", None) is not None
                            and len(workload) < args.trace_rows),
        "model": model,
        "failed": len(failed),
        "prompt_lengths": length_check,
        "ignore_eos": bool(args.ignore_eos),
        # How many rows carried sharing, so a run with no reuse can be told
        # apart from a trace that never asked for any.
        "rows_with_hash_ids": sum(1 for r in workload if r.get("hash_ids")),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"run": manifest, "workload": workload, "plan": plan,
                   "results": results, "engine": engine,
                   "cache_stats": {"before": cache_before, "after": cache_after}},
                  fh, indent=1)
    if closed:
        print(f"{len(lanes)} lanes ran {manifest['sessions_run']} sessions in "
              f"{manifest['profile_window_s']:.0f}s, {len(results)} requests "
              f"({manifest['warmup_requests']} warmup), {len(failed)} failed "
              f"-> {args.out}")
        if guard.shifts:
            print(f"  idle guard shifted {guard.shifted_s:.0f}s of dead air in "
                  f"{guard.shifts} jumps")
        if guard.cut:
            print(f"  {guard.cut} turns were still thinking when the window "
                  f"closed and were not sent")
    else:
        print(f"sent {len(workload)} requests, {len(failed)} failed -> {args.out}")
    if failed:
        print("  first failure:", failed[0]["error"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
