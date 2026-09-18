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


def _prompt(row: dict, index: int) -> str:
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
        return prompt_of_hash_ids(ids, tokens, session=int(row.get("session", 0)))
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


def _session_plan(rows: list[dict]) -> tuple[list[list[int]], list[float]]:
    """Which of a session's rows each row waited for, and how long it then waited.

    A session is not a straight line of turns. 43.5% of the corpus's requests
    overlap another request of their own session, because a turn can fan out
    into sub-agents -- and peak in-session concurrency runs from 1 to 23.

    But that concurrency is *recorded*, not structural. Only 8.4% of sub-agent
    window pairs actually overlap, and 218 of 393 sessions never have two
    sub-agent branches open at once, so treating every branch as parallel
    because the corpus nests it under a `subagent` wrapper would invent load
    the recording does not contain. The recorded windows say which requests
    were really in the server together, so that is what is read here: a row
    waits for exactly those rows that had already finished when it started, and
    runs alongside the rest.

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
    opened = min(starts) if starts else 0.0
    deps: list[list[int]] = []
    think: list[float] = []
    for k in range(len(rows)):
        before = [j for j in range(len(rows)) if j != k and ends[j] <= starts[k]]
        base = max((ends[j] for j in before), default=opened)
        deps.append(before)
        # Floored: a recorded end can sit a hair past a recorded start when the
        # two came from different clocks, and a negative think time would mean
        # a request arriving before the answer it is a reply to.
        think.append(max(0.0, starts[k] - base))
    return deps, think


def _session_span(workload: list[dict], idxs: list[int]) -> float:
    """Recorded seconds from a session's first request starting to its last
    finishing -- think time included, because that is what a slot is occupied
    for."""
    starts = [float(workload[i].get("arrival_s", 0.0)) for i in idxs]
    ends = [s + float(workload[i].get("api_time_s") or 0.0)
            for s, i in zip(starts, idxs)]
    return (max(ends) - min(starts)) if idxs else 0.0


def _balanced_deal(groups, workload, clients):
    """Sessions to slots: longest first, each to the slot holding least so far.

    A slot is a client. It runs one session to the end, then takes another, so
    a sweep over client counts has to deal the *same* pool of sessions to 1, 4,
    8 and 16 slots -- otherwise the rungs are four different workloads and the
    curve joins points that measure different things.

    Dealt up front rather than pulled from a queue as slots free up, for two
    reasons. The modelled side must declare its arrival graph before it runs,
    so a runtime queue is not expressible there at all. And a queue would hand
    different sessions to different slots on the two sides, because they finish
    at different moments -- the comparison would stop being paired per request
    and become merely distributional, with no way to tell a scheduling
    divergence from a cost-model error.

    Longest-processing-time-first is the standard greedy for this and leaves
    far less tail idle than dealing round-robin, which hands one slot the short
    sessions and lets it sit out the rest of the run. It is computed from the
    trace alone, so neither side is handed a decision the other made.
    """
    spans = [_session_span(workload, idxs) for idxs in groups]
    order = sorted(range(len(groups)), key=lambda g: (-spans[g], g))
    assignment: list[list[int]] = [[] for _ in range(clients)]
    load = [0.0] * clients
    for g in order:
        slot = min(range(clients), key=lambda c: (load[c], c))
        assignment[slot].append(g)
        load[slot] += spans[g]
    return assignment, spans


def _dag(workload: list[dict], clients: int, per_client: int):
    """The whole run as one happens-before graph over the workload's rows.

    Built once and used by both executors, so the real side and the modelled
    side cannot drift into running different experiments: the paced executor
    walks it with threads and sleeps, the declared executor hands it to the
    engine, and there is one definition of what the run is.

    Two kinds of edge. Inside a session, the recorded overlap (see
    `_session_plan`). Between sessions, the slot: a session's opening requests
    wait for every request of the session that slot ran before, which is what
    "finish one, then take the next" means. That edge carries no think time --
    the gap between two sessions in the trace is a gap between two different
    users, and says nothing about how soon a slot takes new work.
    """
    groups = _sessions(workload)
    if per_client:
        groups = groups[: clients * per_client]
    assignment, spans = _balanced_deal(groups, workload, clients)

    deps: list[list[int]] = [[] for _ in workload]
    think: list[float] = [0.0] * len(workload)
    selected: list[int] = []
    for slot in assignment:
        previous: list[int] = []
        for g in slot:
            idxs = groups[g]
            sdeps, sthink = _session_plan([workload[i] for i in idxs])
            for k, i in enumerate(idxs):
                think[i] = sthink[k]
                deps[i] = ([idxs[j] for j in sdeps[k]] if sdeps[k]
                           else list(previous))
            selected.extend(idxs)
            previous = list(idxs)
    return groups, assignment, spans, deps, think, sorted(selected)


def _dag_digest(assignment, deps, think, selected) -> str:
    """A name for the graph, so two artifacts can be shown to have run the same
    one. Two sides that dealt sessions differently would still produce a
    well-formed paired report, and it would be comparing different runs."""
    import hashlib

    canonical = json.dumps(
        {"assignment": assignment, "selected": selected,
         "deps": [deps[i] for i in selected],
         "think": [round(think[i], 6) for i in selected]},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _run_session(idxs: list[int], deps, think, send, out: list) -> None:
    """Issue one session's requests on a real clock, sleeping its think time.

    One thread per row, all started before any of them blocks, so a row whose
    predecessors are already done starts at its own offset rather than behind
    the sum of its siblings' waits. Sleeping in the loop that spawns them
    instead would serialise two concurrent sub-agents into one after the other.

    Cannot deadlock: an edge only ever points at a row that finished earlier in
    the recording, and every row's thread exists before any wait begins.
    """
    local = {i: k for k, i in enumerate(idxs)}
    done = [threading.Event() for _ in idxs]

    def go(k: int, i: int) -> None:
        try:
            for j in deps[i]:
                if j in local:
                    done[local[j]].wait()
            if think[i] > 0:
                _time.sleep(think[i])
            out[i] = send(i)
        finally:
            done[k].set()

    threads = [threading.Thread(target=go, args=(k, i), daemon=True)
               for k, i in enumerate(idxs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _closed_loop(workload, groups, assignment, deps, think, send):
    """Run the graph as `len(assignment)` slots on a real clock.

    Each slot walks its sessions in order; the slot's join between them is the
    between-session edge `_dag` declared, so the two executors agree without
    either re-deriving it.
    """
    out: list = [None] * len(workload)

    def slot(c: int) -> None:
        for g in assignment[c]:
            _run_session(groups[g], deps, think, send, out)

    threads = [threading.Thread(target=slot, args=(c,), daemon=True)
               for c in range(len(assignment))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [r for r in out if r is not None]


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
                   help="run closed-loop as this many users instead of "
                        "replaying the trace's arrivals. Each user holds one "
                        "session until every request in it -- sub-agents "
                        "included -- has finished, then takes the next. The "
                        "recorded in-session overlap is kept, and so is the "
                        "think time between a turn coming back and the next "
                        "going out; what is dropped is the gap between "
                        "sessions, which belongs to a different user. This is "
                        "the mode that produces a saturation curve: an "
                        "open-loop replay delivers the trace's own rate "
                        "whatever N is. Add --pace against a real engine")
    p.add_argument("--sessions-per-client", type=int, default=0,
                   help="with --clients, how many sessions each user works "
                        "through. 0 means every session in the trace. Fixing "
                        "it is what makes a sweep over client counts run "
                        "comparable amounts of work per user")
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

    cache_before = _cache_stats(base, args.timeout)
    began = _time.monotonic()

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
    closed = args.clients > 0
    declared_loop = closed and not args.pace

    groups = assignment = None
    deps: list[list[int]] = []
    think: list[float] = []
    selected: list[int] = list(range(len(workload)))
    if closed:
        groups, assignment, _spans, deps, think, selected = _dag(
            workload, args.clients, args.sessions_per_client)
        if not selected:
            print("closed loop executed no requests: the trace has fewer "
                  "sessions than it has clients, so some clients were dealt "
                  "nothing", file=sys.stderr)
            return 2

    def one(i_row):
        i, row = i_row
        at = row["arrival_s"]  # already scaled when the workload was built
        if args.pace and not closed:
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
        }
        if not closed:
            body["compass_workload_size"] = len(workload)
        if args.ignore_eos:
            body["ignore_eos"] = True
        if not args.pace and not closed:
            # Declared rather than delivered: against a simulated engine the
            # wall clock and the virtual clock race, so the arrival is stated
            # and the engine honours it. Ignored by a server on a real clock,
            # which is why --pace exists for that side.
            body["compass_arrival"] = at
        if declared_loop:
            # The same graph the paced side walks with threads, stated so the
            # engine can walk it on its own clock. The ids are row indices into
            # the workload, which is the only naming both sides already share.
            body["compass_workload_size"] = len(selected)
            body["compass_relative_arrival"] = {
                "id": str(i),
                "after": [str(j) for j in deps[i]],
                "think_s": round(think[i], 6),
            }
        try:
            return {"index": i, "ok": True, "response": _send(base + "/v1/completions",
                                                              body, args.timeout)}
        except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
            return {"index": i, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # Posted concurrently and as fast as the socket allows: *when* each lands is
    # deliberately not the arrival the engine uses.
    # One thread per request, in both modes, for two different reasons.
    #
    # Paced: each thread spends its wait sleeping, so a pool of 64 would
    # serialise the 65th arrival behind an earlier request's *generation*
    # rather than behind its arrival.
    #
    # Declared: the server holds every declared request until all of them have
    # arrived, so all of them must be in flight at once. A pool of 64 against a
    # 300-request workload is a deadlock -- 64 threads each blocked on a
    # response the server will not produce until 300 have been posted. It
    # resolves only when the arrival barrier times out, and then the run is not
    # the workload that was asked for: requests enter as earlier ones complete,
    # which is not the declared arrival process. This happened, went unnoticed
    # because the client still reported "0 failed", and a day's conclusions
    # were drawn from the result.
    def _refuse_oversized(workers: int) -> None:
        if workers > MAX_IN_FLIGHT:
            raise SystemExit(
                f"{workers} requests needs {workers} concurrent connections, "
                f"over the {MAX_IN_FLIGHT} this client will open. A declared "
                f"workload cannot be posted in batches -- the server waits for "
                f"all of it before it starts -- so this needs a bulk "
                f"submission endpoint rather than a larger pool. Use "
                f"--num-requests to bound the workload meanwhile.")

    threading.stack_size(_THREAD_STACK_BYTES)
    if closed and not declared_loop:
        # Paced: one thread per row of the session being run, not per row of
        # the trace. A slot holds one session at a time, and a session's own
        # recorded peak concurrency is at most 23.
        results = _closed_loop(workload, groups, assignment, deps, think,
                               lambda i: one((i, workload[i])))
    elif closed:
        # Declared: the whole graph has to be in flight at once, for the same
        # reason an open declared workload does -- the server holds all of it
        # until all of it has arrived.
        _refuse_oversized(len(selected))
        with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
            results = list(pool.map(lambda i: one((i, workload[i])), selected))
    else:
        _refuse_oversized(len(workload))
        with ThreadPoolExecutor(max_workers=max(1, len(workload))) as pool:
            results = list(pool.map(one, enumerate(workload)))
    if not results:
        print("closed loop executed no requests: the trace has fewer "
              "sessions than it has clients, so some clients were dealt "
              "nothing", file=sys.stderr)
        return 2

    failed = [r for r in results if not r["ok"]]

    # What the server says it received, against what was asked for. The builder
    # is exact by construction under a tokenizer giving one token per word in
    # `_WORDS`; this checks the assumption held, and costs nothing because the
    # count is already in every response.
    length_check = "not requested"
    if args.check_lengths:
        off = []
        for r in results:
            if not r["ok"]:
                continue
            got = (r["response"].get("usage") or {}).get("prompt_tokens")
            want = workload[r["index"]]["input_tokens"]
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
        "clients": int(args.clients) if closed else 0,
        "sessions_run": (sum(len(a) for a in assignment) if assignment else 0),
        "sessions_per_client": ([len(a) for a in assignment] if assignment
                                else None),
        # The deal itself, and a name for the whole graph. Two sides that dealt
        # sessions to slots differently still produce a well-formed paired
        # report -- of two different experiments. Comparing these two fields is
        # how that is caught.
        "session_assignment": assignment,
        "dag_sha256": (_dag_digest(assignment, deps, think, selected)
                       if closed else None),
        "think_time_s": (round(sum(think[i] for i in selected), 3)
                         if closed else None),
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
        json.dump({"run": manifest, "workload": workload, "results": results,
                   "engine": engine,
                   "cache_stats": {"before": cache_before, "after": cache_after}},
                  fh, indent=1)
    if closed:
        print(f"{args.clients} clients ran {manifest['sessions_run']} sessions, "
              f"{len(results)} requests, {len(failed)} failed -> {args.out}")
    else:
        print(f"sent {len(workload)} requests, {len(failed)} failed -> {args.out}")
    if failed:
        print("  first failure:", failed[0]["error"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
