"""floors.py OUT [SLICE_S]: which floor holds the 8P16D sim's wall per simulated second in aiperf's profiling
window (#4153). OUT is a run of #4029's `run.sh sim`: its aiperf.log, its stamped clock frames OUT.frames, the
Clock Authority's wall profile OUT.caprof.jsonl and, when #4029's sample.py ran, OUT.spy/schedstat.json.

Every figure is in wall seconds per simulated second of the window, the wall's own unit. Three floors:

1. LP. An LP's own work is serial, so the wall is at least the busiest LP's. Own work runs from a reply
   received to the last request sent before the next reply; a DP group's is the union of its ranks'. The Rust
   router stamps nothing: its own work runs from the CA's reply to its next request and holds its transport.
   With schedstat, the same floor in CPU: the busiest process, all threads (one GIL), over the sampled seconds.
2. CA. The CA serves one request at a time, under one lock: its serve seconds, from its wall profile. With
   schedstat, its process CPU, all threads: each connection's thread reads its frames under the same GIL.
3. Chain. The critical path, walked back from each SLICE_S (default 60) slice's last request as #4029's
   rank.py walks it: a hop is its LP's own work, then the CA round trip that freed it (transport, queueing
   behind other requests, and the serve of the request that made the reply due). The hop's LP heads it; the
   LP whose request was served freed it. The chain's own work alone is its floor were every round trip free.
"""

import bisect
import collections
import datetime
import glob
import json
import math
import os
import re
import sys

out = sys.argv[1]
slice_s = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0


def group(name):  # the LP a clock owner belongs to: a DP group's ranks are one LP
    return re.sub(r"/dp\d+$", "", name)


def norm(addr):
    m = re.match(r"\(LpId\(name='([^']+)'\), '([^']+)'\)", addr)
    return f"{m.group(1)}/{m.group(2)}" if m else addr


def latest(xs, t):  # the latest of the sorted xs at or before t, or None
    i = bisect.bisect_right(xs, t) - 1
    return xs[i] if i >= 0 else None


def union(spans):  # seconds the spans cover
    total, end = 0.0, -math.inf
    for b, e in sorted(spans):
        if e > end:
            total, end = total + e - max(b, end), e
    return total


# the window: aiperf's profiling phase, from its start to its sending complete
rx = r"^(\S+ \S+) - ClockPhaseRunner - \w+ - Phase profiling \(profiling\) (started|sending complete)\b(.*)"
marks, sim = {}, None
for line in open(f"{out}/artifacts/logs/aiperf.log", errors="replace"):
    if m := re.match(rx, line):
        utc = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
        marks.setdefault(m.group(2), utc.replace(tzinfo=datetime.timezone.utc).timestamp())
        sim = float(d.group(1)) if (d := re.search(r"target: ([\d.]+)s duration", m.group(3))) else sim
t0, t1 = marks["started"], marks["sending complete"]
wall = (t1 - t0) / sim

# frames: the CA's requests (Q) and replies (A); each clock owner's requests sent (S) and replies received (R)
ca, pids, kinds = [], {}, collections.Counter()
S, R = collections.defaultdict(list), collections.defaultdict(list)
for path in glob.glob(f"{out}.frames/*.trace"):
    for line in open(path):
        pids.setdefault(os.path.basename(path)[: -len(".trace")], line.split(" ", 3)[2])
        t = float(line[: line.find(" ")])
        if not t0 - 5 <= t <= t1:
            continue
        _, ev, tag, *rest = line.rstrip("\n").split(" ", 3)
        if tag == "CA" and ev == "Q":
            addr, kind, _ = rest[0].rsplit(" ", 2)
            ca.append((t, 0, norm(addr)))
            kinds[kind] += t >= t0
        elif tag == "CA":
            ca.append((t, 1, norm(rest[0])))
        else:
            (S if ev == "S" else R)[tag].append(t)
ca.sort()
for v in (*S.values(), *R.values()):
    v.sort()
A, Aat, Q = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
serves, last_q = [], None  # [start, end] of each serve: its request read, its last reply written
for t, is_a, name in ca:
    if not is_a:
        last_q = (t, name)
        serves.append([t, t])
        Q[name].append(t)
    elif last_q:
        A[name].append((t, *last_q))
        Aat[name].append(t)
        serves[-1][1] = t
starts, cum = [b for b, _ in serves], [0.0]
for b, e in serves:
    cum.append(cum[-1] + e - b)


def busy(t):  # the CA's serving seconds before t
    i = bisect.bisect_right(starts, t) - 1
    return 0.0 if i < 0 else cum[i] + min(t, serves[i][1]) - serves[i][0]


# 1. own work in the window, per clock owner, then per LP
own = collections.defaultdict(list)
for tag, rs in R.items():
    ss = S[tag]
    for k, r in enumerate(rs):
        i = bisect.bisect_left(ss, rs[k + 1] if k + 1 < len(rs) else math.inf) - 1
        if i >= 0 and ss[i] > r and ss[i] > t0 and r < t1:
            own[tag].append((max(r, t0), min(ss[i], t1)))
for a in Aat["router"]:
    i = bisect.bisect_right(Q["router"], a)
    if i < len(Q["router"]) and Q["router"][i] > t0 and a < t1:
        own["router"].append((max(a, t0), min(Q["router"][i], t1)))
by_lp = collections.defaultdict(list)
for tag, spans in own.items():
    by_lp[group(tag)] += spans
own_s = {g: union(v) for g, v in by_lp.items()}

# 2. the CA's serve seconds in the window, its wall profile interpolated to the window's ends
prof = [json.loads(line) for line in open(f"{out}.caprof.jsonl")]
unix = [p["unix"] for p in prof]


def served_by(t):
    i = min(max(bisect.bisect_right(unix, t), 1), len(prof) - 1)
    a, b = prof[i - 1], prof[i]
    return a["serve_s"] + (b["serve_s"] - a["serve_s"]) * (t - a["unix"]) / (b["unix"] - a["unix"])


ca_serve = served_by(t1) - served_by(t0)

# 3. the critical path, slice by slice, walked back from each slice's last request
hops, parts, freed, walked = collections.Counter(), collections.Counter(), collections.Counter(), 0.0
qs = [(t, name) for t, is_a, name in ca if not is_a]
n = int((t1 - t0) // slice_s)
for k in range(n):
    start = t0 + k * slice_s
    cur_t, cur = qs[bisect.bisect_right(qs, (start + slice_s, "")) - 1]
    top = cur_t
    while cur_t > start:
        g = group(cur)
        if cur in S:  # sent at s after receiving at r the reply to its request sent at s0, written at a
            s = latest(S[cur], cur_t)
            r = latest(R[cur], s) if s is not None else None
            s0 = latest(S[cur], r) if r is not None else None
            # the reply's stamp follows its write, so r may precede a: take the first reply after s0
            i = bisect.bisect_left(Aat[cur], s0) if s0 is not None else len(Aat[cur])
            if i == len(Aat[cur]):
                break
            a = Aat[cur][i]
            if a > s and (a := latest(Aat[cur], r)) is None:
                break
            queued = busy(cur_t) - busy(s)
            parts[(g, "own")] += s - max(r, a)
            parts[(g, "queueing")] += queued
            parts[(g, "transport")] += cur_t - s - queued + max(r - a, 0.0)
        else:  # the router: own work, queueing and transport in one
            if (a := latest(Aat[cur], cur_t)) is None:
                break
            parts[(g, "own")] += cur_t - a
        _, trig_t, trig = A[cur][bisect.bisect_right(Aat[cur], a) - 1]
        parts[(g, "serve")] += a - trig_t
        hops[g] += 1
        freed[(group(trig), g)] += cur_t - trig_t
        cur_t, cur = trig_t, trig
    walked += top - max(cur_t, start)
path_sim = walked / wall  # the simulated seconds the walk covers
chain = {p: sum(v for (_, q), v in parts.items() if q == p) / path_sim
         for p in ("own", "transport", "queueing", "serve")}

# schedstat, when sampled: CPU of this run's processes only (another lane's have no frames here)
cpu = {}
if os.path.exists(f"{out}.spy/schedstat.json"):
    seconds = len(json.load(open(f"{out}.spy/cpus.json"))["busy_per_s"])
    proc = collections.defaultdict(lambda: [0.0, 0.0])
    for key, (_, run_s, _) in json.load(open(f"{out}.spy/schedstat.json")).items():
        if (pid := key.split("/")[0].split(".")[1]) in pids:
            proc[pid][0] += run_s
            proc[pid][1] = max(proc[pid][1], run_s)
    cpu = {pids[pid]: (run_s / seconds * wall, top_s / seconds * wall) for pid, (run_s, top_s) in proc.items()}
ca_cpu = cpu.pop("CA", None)

# the floors, ranked
busiest = max(own_s, key=own_s.get)
rows = [("chain: the critical path walked", sum(chain.values()), ""),
        ("chain: its LPs' own work alone", chain["own"], ""),
        (f"LP: own work, the busiest LP ({busiest})", own_s[busiest] / sim, ""),
        ("CA: serve", ca_serve / sim, "")]
if cpu:
    name = max(cpu, key=lambda p: cpu[p][0])
    rows.append((f"LP: CPU, the busiest process, all threads ({name})", cpu[name][0],
                 f"its busiest thread {cpu[name][1]:.2f}"))
if ca_cpu:
    rows.append(("CA: CPU, all threads", ca_cpu[0], f"its busiest thread {ca_cpu[1]:.2f}"))
print(f"## Floors of {out}\n\nWindow: {t1 - t0:.0f} s of wall over {sim:.0f} simulated s, **{wall:.2f} wall s per "
      f"simulated s**. Critical path walked over {walked:.0f} s, {n} slices of {slice_s:.0f} s.\n")
print("| rank | floor | wall s per simulated s | share of the wall | note |\n|---|---|---|---|---|")
for i, (label, v, note) in enumerate(sorted(rows, key=lambda r: -r[1]), 1):
    print(f"| {i} | {label} | {v:.2f} | {100 * v / wall:.0f}% | {note} |")
print(f"\nThe chain per simulated s: own work {chain['own']:.2f}, transport {chain['transport']:.2f}, CA queueing "
      f"{chain['queueing']:.2f}, CA serve {chain['serve']:.2f}; {sum(hops.values()) / path_sim:,.0f} hops. CA requests "
      f"{sum(kinds.values()) / sim:,.0f} per simulated s, {kinds['REPORT'] / sim:,.0f} of them REPORTs. Parallelism, "
      f"the LPs' own work over the wall: {sum(own_s.values()) / (t1 - t0):.2f}; each DP rank apart, "
      f"{sum(union(v) for v in own.values()) / (t1 - t0):.2f}.\n")


def edge(a, b):  # which kind of dependency freed b's hop: a's request served
    if a == b:
        return "an LP by itself"
    if {re.sub(r"\d+$", "", a), re.sub(r"\d+$", "", b)} in ({"frontend-D", "engine-D"}, {"frontend-P", "engine-P"}) \
            and a[-1] == b[-1]:
        return "a frontend and its own engine"
    return "any other pair"


edges, kinds_of = collections.Counter(), collections.Counter()
for (a, b), v in freed.items():
    edges[edge(a, b)] += v
    kinds_of[f"{re.sub(r'[0-9]+$', '*', a)} -> {re.sub(r'[0-9]+$', '*', b)}" + (" (itself)" if a == b else "")] += v
print("The chain by what freed each hop, as shares of the walked wall: "
      + "; ".join(f"{k} {100 * v / walked:.1f}%" for k, v in edges.most_common()) + ". Its leading edges: "
      + "; ".join(f"{k} {100 * v / walked:.1f}%" for k, v in kinds_of.most_common(6)) + ".\n")
print("| LP | own work per simulated s | CPU per simulated s, all threads / busiest thread | hops it heads, per "
      "simulated s | share of the chain | own work per hop on the chain, ms |\n|---|---|---|---|---|---|")
for g in sorted(own_s, key=own_s.get, reverse=True):
    mine = [c for name, c in cpu.items() if group(name) == g]
    c = f"{max(x[0] for x in mine):.2f} / {max(x[1] for x in mine):.2f}" if mine else "-"
    share = sum(v for (h, _), v in parts.items() if h == g) / walked
    print(f"| {g} | {own_s[g] / sim:.2f} | {c} | {hops[g] / path_sim:,.0f} | {100 * share:.1f}% | "
          f"{1e3 * parts[(g, 'own')] / max(hops[g], 1):.2f} |")
