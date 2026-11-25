"""reactions.py OUT: how far a decode engine of the 8P16D C1228 sim could lead if each LP declared its least
reaction delay, and the wall per simulated second that lead allows (#4178). OUT is a run of #4029's `run.sh sim`:
its step table, its engines' step records (prefill<k>/steps, decode<k>/steps), request_split.json, aiperf.log, and
the Clock Authority's stamped frames in OUT.frames.

1. Delays. Per LP and channel, the least simulated delay from a message's arrival to the arrival of the message it
   provokes, a request at a time. A decode request names its prefill (transfers.jsonl), whose pull, the end of the
   step that outputs it, the prefill engine's step records give; the relay between them is its frontend-P's next in
   the order its prefills finish. A prefill's request arrived no later than its first schedule, so its forward is
   timed from there: a lower bound. Each least delay less the outbound channel's lookahead is the reaction itself.
2. Lead. engine-D<k> takes messages from frontend-D<k> alone. With every other LP at simulated time p and every LP
   declaring its least delays, LBTS(engine-D<k>) - p is the least of: each busy prefill engine's next pull plus E2D,
   the least pull-to-decode-request distance (were each step to say at its schedule whether its pull outputs a
   finished prefill, a pull that does not adds the least step that does); each relay from its prefill's pull to the
   decode request it provokes, held at frontend-P, in flight, or in frontend-D<k>'s handler; a new request, which
   needs a prefill forward; and a cancel, traffic's own timed event at the duration's end. An engine-D<k> round from
   p to t needs no other LP past p when the lead covers t - p. Each exception caps the lead at the distance it opens.
3. Wall. The CA's frames replayed in order: a request follows its LP's last reply by its measured own work and
   transport, a reply follows the request whose serve made it due by that serve. A covered engine-D round's reply
   follows its own request instead of another LP's, covered from the member's last round that waited, so it never
   leads the others by more than the lead. The wall is the latest replayed event in aiperf's profiling window, the
   window floors.py (#4153) times. The CA serves every request at once and each LP has a core of its own.
"""

import bisect
import collections
import datetime
import glob
import json
import math
import re
import sys

out = sys.argv[1]
run = json.load(open(f"{out}/run.json"))["run_file"]
FLOOR = 2.0**-14  # the lookahead of a hop the run file leaves unset: ipc_out_s, admission_s and stream_s here
R_IN, R_S, IPC_P, IPC_D = run["router_in_s"], run["router_s"], run["ipc_s"], run["decode_hops"]["ipc_s"]
OUT, HTTP_OUT, RELAY_OUT = FLOOR, FLOOR + R_S - R_IN, R_S - R_IN  # engine output; the router's hops out


def jsonl(path):
    return [json.loads(line) for line in open(path)]


# the step table: each message's arrival by channel, in seq order; engine-D's grants
arr, rounds = collections.defaultdict(list), collections.defaultdict(list)
with open(f"{out}/step_table.txt") as f:
    next(f)
    for line in f:
        p = line.split()
        if p[3] == "release":
            arr[p[4]].append((int(p[5]), float(p[2])))
        elif p[3] == "TAR" and p[0].startswith("engine-D"):
            rounds[int(p[0][len("engine-D"):])].append((float(p[1]), float(p[2])))
arr = collections.defaultdict(list, {c: [a for _, a in sorted(v)] for c, v in arr.items()})


def merged(rx):
    return sorted(a for c in list(arr) if re.fullmatch(rx, c) for a in arr[c])


# the simulated window: aiperf's profiling phase, from its first request (the client's clock starts at 1 s)
log = open(f"{out}/artifacts/logs/aiperf.log", errors="replace").read()
duration = float(re.search(r"target: ([\d.]+)s duration", log).group(1))
split = json.load(open(f"{out}/request_split.json"))["requests"]
w0 = min(int(r["start_ns"]) for r in split if r["phase"] == "profiling") / 1e9 - 1.0
w1 = w0 + duration

# 1. delays, arrival to provoked arrival, per request
d = collections.defaultdict(list)
NP, ND = len(glob.glob(f"{out}/prefill*/steps")), len(glob.glob(f"{out}/decode*/steps"))
steps, pull_of, relay_at, unnamed = [], {}, {}, 0  # steps: per prefill member, sorted (start, pull)
for k in range(NP):
    fin = []  # (pull, request) of each prefill frontend-P<k> relays
    for r in range(2):
        path = f"{out}/prefill{k}/steps/dp{r}"
        cost = {c["step"]: c for c in jsonl(f"{path}.cost.jsonl") if "step" in c}
        pull = {s: c["t_s"] + c["charge_s"] + c["E_out"] for s, c in cost.items()}  # its output leaves here
        finishes = set()
        first = {}
        for st in jsonl(f"{path}.jsonl"):
            for b in st["batch"]:
                first.setdefault(b[0], cost[st["step"]]["t_s"])
            for seq, _, _ in st["named"]:
                finishes.add(st["step"])
                pull_of[(k, seq)] = pull[st["step"]]
                fin.append((pull[st["step"]], (k, seq)))
                d["engine-P: request -> output"].append(pull[st["step"]] + OUT - first[seq])
        steps.append(sorted((cost[s]["t_s"], pull[s], s in finishes) for s in cost))
        rq, lk = arr[f"frontend-P{k}->engine-P{k}:request#dp{r}"], arr[f"engine-P{k}->mooncake_store:lookup#dp{r}"]
        d["engine-P: request -> lookup (the latest request before each)"] += [a - rq[i] for a in lk if (i := bisect.bisect_right(rq, a) - 1) >= 0]
        d["mooncake_store: lookup -> lookup"] += [b - a for a, b in zip(lk, arr[f"mooncake_store->engine-P{k}:lookup#dp{r}"])]
    unnamed += len(merged(rf"frontend-P{k}->engine-P{k}:request#dp\d")) - len(fin)
    http, req = arr[f"router->frontend-P{k}:http"], merged(rf"frontend-P{k}->engine-P{k}:request#dp\d")
    d["frontend-P: http -> request"] += [b - a for a, b in zip(http, req)]
    for (x, key), a in zip(sorted(fin), arr[f"frontend-P{k}->router:relay"]):
        d["frontend-P: output -> relay"].append(a - x - OUT)
        relay_at[key] = a

chain = []  # per decode request: (its prefill's pull, its relay's arrival at the router, decode k, its arrival)
for k in range(ND):
    sent = collections.Counter()
    for route, tr in zip(jsonl(f"{out}/decode{k}/steps/routes.jsonl"), jsonl(f"{out}/decode{k}/steps/transfers.jsonl")):
        r = route["rank"][0]
        a = arr[f"frontend-D{k}->engine-D{k}:request#dp{r}"][sent[r]]
        sent[r] += 1
        key = (int(tr["remote_engine_id"].rsplit("p", 1)[1]), tr["transfer_id"])
        chain.append((pull_of[key], relay_at[key], k, a))
        d["router: relay -> relay"].append(RELAY_OUT)  # its restamp: compass_router.rs
        d["frontend-D: relay -> request"].append(a - relay_at[key] - RELAY_OUT)
        d["E2D: prefill pull -> decode request"].append(a - pull_of[key])
    ctl, rl = merged(rf"frontend-D{k}->engine-D{k}:control#dp\d"), arr[f"router->frontend-D{k}:relay"]
    d["frontend-D: relay -> control"] += [c - rl[bisect.bisect_right(rl, c) - 1] for c in ctl]
controls = merged(r"frontend-D\d+->engine-D\d+:control#dp\d")
cancel = controls[0] - IPC_D - R_S if controls else math.inf  # traffic's hang-up
cancels = len(merged(r"router->frontend-D\d+:relay")) - len(chain)
d["router: stream -> stream"] = [b - a for a, b in zip(merged(r"frontend-D\d+->router:stream"), merged(r"router->traffic:stream"))]
by_session = collections.defaultdict(list)
for r in split:
    by_session[r["session_id"]].append((int(r["start_ns"]), int(r["end_ns"])))
d["traffic: stream -> http"] = [(b[0] - a[1]) / 1e9 for v in map(sorted, by_session.values()) for a, b in zip(v, v[1:])]
OUTBOUND = {"engine-P: request -> output": OUT, "frontend-P: http -> request": IPC_P, "frontend-P: output -> relay": R_IN,
            "frontend-D: relay -> request": IPC_D, "frontend-D: relay -> control": IPC_D, "traffic: stream -> http": R_IN,
            "router: stream -> stream": FLOOR + R_S - R_IN, "router: relay -> relay": RELAY_OUT}
E2D, FWD, OR, FD, HR = (min(d[n]) for n in ("E2D: prefill pull -> decode request", "engine-P: request -> output",
                                            "frontend-P: output -> relay", "frontend-D: relay -> request",
                                            "frontend-P: http -> request"))
FIN = min(e - b for m in steps for b, e, fin in m if fin)  # the least step that outputs a finished prefill
NEW = IPC_P + FWD - OUT + E2D  # a new request, sent by a frontend-P at p, to engine-D
TRAFFIC = R_IN + HTTP_OUT + HR + FWD - OUT + E2D  # traffic's own event at p, or a stream it answers, to engine-D
AT_ONCE = IPC_P + E2D  # a request a frontend-P sends at p, were engine-P to output at its arrival
VARIANTS = [  # (declarations, cap on the lead, per-step outcome; or None and the LPs whose every grant wait is freed)
    ("1. each LP's least delay per inbound and outbound channel", math.inf, False),
    ("2. as 1, and each prefill step declares whether its pull outputs a finished prefill", math.inf, True),
    ("as 1 (a rejection leaves at any schedule), and a rejected prefill may output at its arrival", AT_ONCE, False),
    ("as 2, but engine-P declares one delay, which its offered Store lookup makes 0", AT_ONCE, True),
    ("as 2, but a stream may provoke a cancel", R_S + IPC_D, True),
    ("as 2, but frontend-D's relayed handlers never report", IPC_D, True),
    ("any lead: every engine-D round freed", None, ("engine-D",)),
    ("every engine-D and engine-P round freed", None, ("engine-D", "engine-P")),
    ("every LP's grant waits freed", None, ("",))]

# 2. the lead of engine-D<k> at p
phases = []  # (start, end, decode k or None for every k, term at p, in the handler) of each relay
for x, ra, k, a in chain:
    s, rd = ra - R_IN, ra + RELAY_OUT
    phases += [(x, s, None, lambda p, x=x: max(p, x + OUT + OR - R_IN) + R_S + FD, False),  # at frontend-P
               (s, ra, None, lambda p, rd=rd: rd + FD, False),  # to the router, which picks frontend-D<k>
               (ra, rd, k, lambda p, rd=rd: rd + FD, False),  # to frontend-D<k>
               (rd, a - IPC_D, k, lambda p, rd=rd: max(p, rd + FD - IPC_D) + IPC_D, True),  # in its handler
               (a - IPC_D, a, k, lambda p, a=a: a, False)]  # the decode request, to engine-D<k>
phases.sort(key=lambda ph: ph[0])
pstart, longest = [ph[0] for ph in phases], max(ph[1] - ph[0] for ph in phases)
starts = [[b for b, _, _ in m] for m in steps]


def relays(p, k):
    """The phases of the relays that may still reach engine-D<k>, at p."""
    i = bisect.bisect_right(pstart, p)
    while i > 0 and pstart[i - 1] >= p - longest:
        i -= 1
        if p < phases[i][1] and phases[i][2] in (None, k):
            yield phases[i]



def lead(p, k, cap=math.inf, outcome=False):
    """LBTS(engine-D<k>) - p, every other LP at p. With `outcome`, a prefill step that outputs no finished prefill
    says so at its schedule, and the next that may starts at its pull."""
    terms = [p + NEW, cancel + R_S + IPC_D, p + cap]
    for m, st in zip(steps, starts):  # a busy prefill member's next pull, or a pull whose output is in flight
        i = bisect.bisect_right(st, p) - 1
        terms += [e + (0.0 if fin or not outcome else FIN) + E2D for b, e, fin in m[max(i - 1, 0):i + 1]
                  if i >= 0 and b <= p < e + OUT]
    terms += [ph[3](p) for ph in relays(p, k)]
    return min(terms) - p


# 3. the CA's frames, replayed
rx = r"^(\S+ \S+) - ClockPhaseRunner - \w+ - Phase profiling \(profiling\) (started|sending complete)\b"
marks = {}
for line in log.splitlines():
    if m := re.match(rx, line):
        utc = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
        marks.setdefault(m.group(2), utc.replace(tzinfo=datetime.timezone.utc).timestamp())
t0, t1 = marks["started"], marks["sending complete"]


def norm(addr):
    m = re.match(r"\(LpId\(name='([^']+)'\), '([^']+)'\)", addr)
    return f"{m.group(1)}/{m.group(2)}" if m else addr


ev = []  # (wall, is a reply, clock owner, request kind, target)
ca = next(p for p in glob.glob(f"{out}.frames/*.trace") if open(p).readline().split(" ")[2] == "CA")
for line in open(ca):
    t, e, _, rest = line.rstrip("\n").split(" ", 3)
    if float(t) > t1:
        continue
    if e == "A":
        ev.append((float(t), True, norm(rest), "", 0.0))
    else:
        addr, kind, target = rest.rsplit(" ", 2)
        ev.append((float(t), False, norm(addr), kind, float(target)))
ev.sort(key=lambda v: (v[0], v[1]))


def replay(cap, outcome):
    """The window's replayed wall per simulated s, and the rounds per simulated s freed by their own request instead
    of another LP's: when `cap` is None, every TAR of the LPs `outcome` names; else each engine-D round the lead
    covers from the member's last round that waited, where the other LPs stood, so a member never leads further
    than the lead."""
    last_a, pend, synced, trig, end, freed = {}, {}, {}, (t0, t0, ""), t0, 0
    for t, is_a, name, kind, target in ev:
        g = name.split("/")[0]
        if not is_a:
            oa, na = last_a.get(name, (t, t))
            nq = na + t - oa
            own = False
            if cap is None:
                own = t >= t0 and kind == "TAR" and g.startswith(outcome)
            elif g.startswith("engine-D") and kind != "REPORT":
                at = synced.get(name)
                if kind != "TAR":  # idle or parked: where the others stand is not known
                    synced[name] = None
                elif t >= t0 and at is not None and target - at <= lead(at, int(g[8:]), cap, outcome):
                    own = True
                else:
                    synced[name] = target
            if kind != "REPORT":
                pend[name] = (nq, own)
            trig = (t, nq, g)
        else:
            nq, own = pend.pop(name, (trig[1], False))
            own = own and trig[2] != g
            freed += own
            last_a[name] = (t, (nq if own else max(trig[1], nq)) + t - trig[0])
            nq = last_a[name][1]
        end = max(end, nq)
    return (end - t0) / duration, freed / duration


# the report
print(f"## Reaction delays and the decode lead of {out}\n\nSimulated window {w0:.1f} s to {w1:.1f} s; wall window "
      f"{t1 - t0:.0f} s, **{(t1 - t0) / duration:.2f} wall s per simulated s**.\n")
print("| LP: in -> out | messages | below 0 | least, arrival to arrival, ms | median, ms | the reaction: least less "
      "the out lookahead, ms |\n|---|---|---|---|---|---|")
for name, v in d.items():
    v = sorted(v)
    print(f"| {name} | {len(v):,} | {sum(x < 0 for x in v)} | {1e3 * v[0]:.3f} | {1e3 * v[len(v) // 2]:.3f} | "
          + (f"{round(1e3 * (v[0] - OUTBOUND[name]), 3) + 0.0:.3f} |" if name in OUTBOUND else "- |"))
print(f"\nRequests sent to a prefill engine and never output by a prefill step (rejected, or unfinished at the end): "
      f"{unnamed}. Cancels: {cancels} relays, sent by traffic at simulated {cancel:.3f} s. Prefill steps in the "
      f"simulated window: {sum(w0 <= b < w1 for m in steps for b, _, _ in m):,}, of which "
      f"{sum(w0 <= b < w1 and fin for m in steps for b, _, fin in m):,} output a finished prefill; the least of all "
      f"such steps: {1e3 * FIN:.3f} ms. Relays from frontend-P in the simulated window: "
      f"{sum(w0 <= ra < w1 for _, ra, _, _ in chain) / duration:.1f} per simulated s, each a REPORT when its decode "
      f"request leaves frontend-D's handler.\n")
print("| a chain into engine-D, from | least distance, ms |\n|---|---|")
for label, v in (("frontend-D's relayed handler, until it sends", IPC_D), ("a relay at frontend-D", FD),
                 ("a relay at the router", RELAY_OUT + FD), ("a prefill output at frontend-P", OR + RELAY_OUT + FD),
                 ("a prefill engine's pull (E2D)", E2D), ("a new request at a prefill engine", FWD - OUT + E2D),
                 ("traffic's next request", TRAFFIC), ("traffic's cancel, at its own timed event", R_S + IPC_D),
                 ("a request from frontend-P, were engine-P to output at its arrival", AT_ONCE)):
    print(f"| {label} | {1e3 * v:.3f} |")
print("\n| declarations | engine-D rounds covered | forward rounds (over 5 ms) covered | lead at a forward round's "
      "start, p10 / median, ms | rounds freed from another LP in the replay, per simulated s | replayed wall s per "
      "simulated s |\n|---|---|---|---|---|---|")
for label, cap, outcome in VARIANTS:
    leads, n = [], [0, 0, 0, 0]
    for k, rs_k in rounds.items():
        for b, e in rs_k:
            if w0 <= b < w1:
                lv = math.inf if cap is None else lead(b, k, cap, outcome)
                n[0], n[1] = n[0] + 1, n[1] + (lv >= e - b)
                if e - b > 5e-3:
                    n[2], n[3] = n[2] + 1, n[3] + (lv >= e - b)
                    leads.append(lv)
    leads.sort()
    q = [leads[int(f * len(leads))] for f in (0.1, 0.5)] if leads else []
    lq = " / ".join("-" if math.isinf(v) else f"{1e3 * v:.2f}" for v in q)
    wall, freed = replay(cap, outcome)
    print(f"| {label} | {100 * n[1] / n[0]:.1f}% | {100 * n[3] / max(n[2], 1):.1f}% | {lq} | {freed:,.0f} | {wall:.2f} |")
traffic, last = 0.0, None  # traffic's own work and transport: from each reply to its next request
for t, is_a, name, kind, _ in ev:
    if name == "traffic":
        if not is_a and last is not None and t > t0:
            traffic += t - max(last, t0)
        last = t if is_a or kind == "REPORT" else None
print(f"\nTraffic's own work and transport alone, which no grant rule frees: {traffic / duration:.2f} wall s per "
      "simulated s.")
forward = [(b, k) for k, v in rounds.items() for b, e in v if w0 <= b < w1 and e - b > 5e-3]
print(f"\nForward rounds that start while frontend-D<k> holds a relay in its handler: "
      f"{100 * sum(any(ph[4] for ph in relays(b, k)) for b, k in forward) / len(forward):.1f}%.")
