"""split.py OUT [SLICE_S]: split each simulated decode engine member's hop in OUT's profiling window, from its clock
grant to its next clock request, into its hand-offs (#4151), from the stamps hopstamps.py wrote into OUT.frames.

A hop runs from a grant (R) to the member's next request that waits for one (S, a REPORT does not). Its time between
consecutive stamps goes to one part each, so the parts sum to the hop: the member's own stamps, except that from the
ring's write (Qw) to the reply's put (O1) the forward's round trip runs on its worker's stamps. The interval of a
DP collective, the engine's shared-memory exchange (X0-X1) or a worker's gloo call (c0-c1), splits at the latest
entry of the same call across the group's ranks (matched by call count): before it, waiting for the slowest rank.

Hops are ranked twice: every engine-D hop, and the engine-D hops on the critical path, which rank.py's walk takes
from OUT.slices (run `rank.py OUT SLICE_S` first, SLICE_S default 60) and which match a stamped hop by its request.
"""

import array
import bisect
import collections
import datetime
import glob
import json
import re
import sys

out, slice_s = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
AFTER = {  # the part an interval goes to, by the stamp that opens it; any other is the engine core's own work
    "R": "grant: release walk (LPRuntime._grant)",
    "H0": "scheduling (Scheduler.schedule)",
    "Z0": "postprocess (EngineCore._postprocess)",
    "K0": "KV transfer poll round trip (call_func_with_aggregation)",
    "C0": "dispatch: RPC encode (call_func, encode_rpc)",
    "Q0": "dispatch: pickle and ring write (MessageQueue.enqueue)",
    "Qw": "dispatch: doorbell ring and worker wake-up (RingingQueue.enqueue, Doorbell.spin)",
    "W0": "dispatch: dequeue and unpickle (AsyncIOProc.get_func)",
    "W1": "worker: dispatch to forward (AsyncIOProc.busy_loop)",
    "F0": "forward stand-in: pricing (NonAllocatingRunner.forward)",
    "c1": "forward stand-in: pricing (NonAllocatingRunner.forward)",
    "F1": "reply: worker output thread wake-up (send_output_to_socket)",
    "wget": "reply: pickle, zmq, engine output thread (process_output_sockets)",
    "O1": "reply: engine owner wake-up (outputs_queue.get)",
}
COLLECTIVE = {"X0": ("DP state sync", "shared-memory exchange (_Segment.exchange)"),
              "c0": ("forward stand-in", "gloo collective (torch.distributed)")}
GROUPS = [  # the brief's hand-offs, each part in the first that takes it; the rest is the engine core's own work
    ("DP group's wait for its slowest rank", lambda p: "waiting for the slowest rank" in p),
    ("engine core to worker", lambda p: p.startswith(("dispatch", "worker:"))),
    ("worker's forward stand-in", lambda p: p.startswith("forward stand-in")),
    ("reply thread", lambda p: p.startswith("reply")),
]


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    a = array.array("d")
    a.frombytes(data[: len(data) // 16 * 16])  # a torn last pair is dropped
    head = json.load(open(path + ".json"))
    ev = sorted((a[i], head["events"][int(a[i + 1]) // 100], int(a[i + 1]) % 100) for i in range(0, len(a), 2))
    return head, ev


# the profiling window, as rank.py reads it
marks = {}
for line in open(f"{out}/artifacts/logs/aiperf.log", errors="replace"):
    if m := re.match(r"^(\S+ \S+) - ClockPhaseRunner - \w+ - Phase profiling \(profiling\) (started|sending complete)", line):
        utc = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc)
        marks.setdefault(m.group(2), utc.timestamp())
t0, t1 = marks["started"], marks["sending complete"]

members, workers = {}, {}
for p in glob.glob(f"{out}.frames/*.hop"):
    head, ev = load(p)
    (members if head["role"] == "engine" else workers)[head["tag"]] = (head, ev)
worker_of = {tag: workers.get(tag) for tag in members}


def last_entries(series, opener, closer):
    """Per LP group, the k-th call's latest entry across its ranks; and how many calls' exits spread over 2 ms."""
    by_group, last, bad = collections.defaultdict(dict), {}, collections.Counter()
    for tag, (_, ev) in series.items():
        by_group[tag.split("/")[0]][tag] = ([t for t, e, _ in ev if e == opener], [t for t, e, _ in ev if e == closer])
    for g, ranks in by_group.items():
        n = min(len(o) for o, _ in ranks.values())
        last[g] = [max(o[k] for o, _ in ranks.values()) for k in range(n)]
        bad[g] = sum(max(c[k] for _, c in ranks.values()) - min(c[k] for _, c in ranks.values()) > 2e-3
                     for k in range(min(n, *(len(c) for _, c in ranks.values()))))
        bad[g, "calls"] = (n, max(len(o) for o, _ in ranks.values()))
    return last, {tag: o for g in by_group.values() for tag, (o, _) in g.items()}, bad


LAST = {"X0": last_entries(members, "X0", "X1"), "c0": last_entries(workers, "c0", "c1")}


def split(tag, ev, i, j):
    """The parts of member `tag`'s hop from its grant ev[i] to its request ev[j]."""
    chain = ev[i : j + 1]
    fwd = [k for k, (_, e, d) in enumerate(chain) if e == "C0" and d in (11, 12)]
    if fwd and worker_of[tag]:
        k = fwd[-1]
        qw = next((n for n in range(k, len(chain)) if chain[n][1] == "Qw"), None)
        o1 = next((n for n in range(k, len(chain)) if chain[n][1] == "O1"), None)
        if qw is not None and o1 is not None:
            wev = worker_of[tag][1]
            lo, hi = bisect.bisect_right(wev, (chain[qw][0], "~")), bisect.bisect_left(wev, (chain[o1][0], ""))
            inner = [x for x in wev[lo:hi] if x[1] != "wsent"]
            chain = chain[: qw + 1] + inner + chain[o1:]
    parts, lock = collections.Counter(), None
    for (ta, a, _), (tb, b, _) in zip(chain, chain[1:]):
        if lock:
            parts[lock] += tb - ta
            lock = None if b == "K1" else lock
            continue
        if a in COLLECTIVE:
            who, what = COLLECTIVE[a]
            last, entries, _ = LAST[a]
            ks = entries.get(tag, [])  # a worker's stamps carry its engine member's tag
            k = bisect.bisect_left(ks, ta)
            lg = last.get(tag.split("/")[0], [])
            wait = min(max(lg[k] - ta, 0.0), tb - ta) if k < len(lg) and k < len(ks) and ks[k] == ta else 0.0
            parts[f"{who}: waiting for the slowest rank"] += wait
            parts[f"{who}: {what}"] += tb - ta - wait
            continue
        label = AFTER.get(a) or f"engine core: {a} -> {b}"
        lock = label if a == "K0" else None
        parts[label] += tb - ta
    return parts


def hops_of(ev):
    """(grant index, request index, what the grant answered) for each hop whose request falls in the window."""
    out_, r, y, adv, kind = [], None, False, False, None
    for n, (t, e, d) in enumerate(ev):
        y, adv = (e == "Y0") or (y and e != "Y1"), (e == "A0") or (adv and e != "A1")
        if e == "S" and d != 4:
            if r is not None and t0 <= t < t1:
                out_.append((r, n, kind))
            r, kind = None, "settle" if y else "advance" if adv else {1: "TAR", 2: "NER", 3: "PARK"}.get(d, "?")
        elif e == "R":
            r = n
    return out_


everything, by_kind, stamped = collections.Counter(), collections.defaultdict(list), {}
for tag, (_, ev) in sorted(members.items()):
    stamped[tag] = []
    for i, j, kind in hops_of(ev):
        p = split(tag, ev, i, j)
        everything.update(p)
        by_kind[kind].append(ev[j][0] - ev[i][0])
        stamped[tag].append((ev[j][0], ev[i][0], p))


# the critical path, as rank.py walks it; only its engine-D hops are kept
def norm(addr):
    m = re.match(r"\(LpId\(name='([^']+)'\), '([^']+)'\)", addr)
    return f"{m.group(1)}/{m.group(2)}" if m else re.sub(r"LpId\(name='([^']+)'\)", r"\1", addr)


def latest(xs, t):
    i = bisect.bisect_right(xs, t) - 1
    return xs[i] if i >= 0 else None


crit, crit_slices, match = collections.Counter(), [], collections.Counter()
for k in range(int((t1 - t0) // slice_s)):
    start, here = t0 + k * slice_s, collections.Counter()
    ca, S, R = [], collections.defaultdict(list), collections.defaultdict(list)
    for line in open(f"{out}.slices/{k}.txt"):
        p = line.split(None, 3)
        if p[2] == "CA":
            ca.append((float(p[0]), p[1] != "Q", norm(p[3].strip().rsplit(" ", 2)[0] if p[1] == "Q" else p[3].strip())))
        else:
            (S if p[1] == "S" else R)[p[2]].append(float(p[0]))
    ca.sort()
    for v in (*S.values(), *R.values()):
        v.sort()
    A, Aat, last_q = collections.defaultdict(list), collections.defaultdict(list), None
    for t, is_a, name in ca:
        if not is_a:
            last_q = (t, name)
        elif last_q:
            A[name].append((t, *last_q))
            Aat[name].append(t)
    cur_t, cur = next((t, name) for t, is_a, name in reversed(ca) if not is_a)
    while cur_t > start:
        if cur in S:  # sent at s after receiving at r the reply to its request sent at s0, written at a
            s = latest(S[cur], cur_t)
            r = latest(R[cur], s) if s is not None else None
            s0 = latest(S[cur], r) if r is not None else None
            i = bisect.bisect_left(Aat[cur], s0) if s0 is not None else len(Aat[cur])
            if i == len(Aat[cur]):
                break
            a = Aat[cur][i]
            if a > s and (a := latest(Aat[cur], r)) is None:
                break
            if cur.startswith("engine-D"):
                hs = stamped.get(cur, [])
                n = bisect.bisect_right(hs, (s + 1e-9,)) - 1
                if n >= 0 and s - hs[n][0] < 1e-3 and abs(hs[n][1] - r) < 1e-3:
                    here.update(hs[n][2])
                    here["#hops"] += 1
                    match["frames s"] += s - max(r, a)
                    match["stamps s"] += hs[n][0] - hs[n][1]
                else:
                    match["unmatched hops"] += 1
                    match["unmatched s"] += s - max(r, a)
        elif (a := latest(Aat[cur], cur_t)) is None:
            break
        _, trig_t, trig = A[cur][bisect.bisect_right(Aat[cur], a) - 1]
        cur_t, cur = trig_t, trig
    crit_slices.append(here)
    crit.update(here)


def shares(parts):
    """Each part's share of the hops' wall, each hand-off's, and the wall."""
    parts = {p: v for p, v in parts.items() if not p.startswith("#")}
    tot = sum(parts.values()) or 1e-9
    groups = collections.Counter()
    for p, v in parts.items():
        groups[next((g for g, takes in GROUPS if takes(p)), "engine core's own work")] += v
    return {p: v / tot for p, v in parts.items()}, {g: v / tot for g, v in groups.items()}, tot


def spread(xs, scale=100.0):
    xs = sorted(xs)
    return " / ".join(f"{scale * x:.1f}" for x in (xs[0], xs[len(xs) // 2], xs[-1])) if xs else "-"


def report(parts, n, title, per_slice=()):
    part_sh, group_sh, tot = shares(parts)
    sl = [(shares(s), s["#hops"]) for s in per_slice if s.get("#hops")]
    col, sep = (" | slices min / median / max %", "---|") if sl else ("", "")
    print(f"\n## {title}: {n} hops, {tot:.1f} s wall, {1e3 * tot / max(n, 1):.3f} ms a hop\n")
    if sl:
        print(f"Per {slice_s:.0f} s slice, ms a hop min / median / max: {spread([x[2] / h for x, h in sl], 1e3)}.\n")
    print(f"| hand-off | share{col} |\n|---|---|{sep}")
    for g, v in sorted(group_sh.items(), key=lambda kv: -kv[1]):
        print(f"| {g} | {100 * v:.1f}%" + (f" | {spread([x[1].get(g, 0.0) for x, _ in sl])}" if sl else "") + " |")
    print(f"\n| rank | part | share | ms a hop{col} |\n|---|---|---|---|{sep}")
    for rank, (p, v) in enumerate(sorted(part_sh.items(), key=lambda kv: -kv[1]), 1):
        if v >= 0.005:
            print(f"| {rank} | {p} | {100 * v:.1f}% | {1e3 * v * tot / max(n, 1):.3f}"
                  + (f" | {spread([x[0].get(p, 0.0) for x, _ in sl])}" if sl else "") + " |")


print(f"# Decode engine hop split of {out}\n\nProfiling window {t1 - t0:.0f} s wall; {len(members)} engine-D members "
      f"stamped, {sum(w is not None for w in worker_of.values())} of them with their worker.\n")
print("| hop opened by the grant of | hops | mean ms | median ms |\n|---|---|---|---|")
for kind, v in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
    v.sort()
    print(f"| {kind} | {len(v)} | {1e3 * sum(v) / len(v):.3f} | {1e3 * v[len(v) // 2]:.3f} |")
report(crit, crit["#hops"], "Engine-D hops on the critical path", crit_slices)
print(f"\nThose hops' wall: {match['stamps s']:.2f} s by the stamps, {match['frames s']:.2f} s by the frames' own work "
      f"({100 * (match['stamps s'] / max(match['frames s'], 1e-9) - 1):+.2f}%); {match['unmatched hops']:.0f} hops "
      f"({match['unmatched s']:.2f} s) on the path matched no stamped hop.")
report(everything, sum(len(v) for v in by_kind.values()), "Every engine-D hop")
print("\n## Collective matching\n\n| call | LP group | calls matched / most by one rank | matched calls whose exits "
      "spread over 2 ms |\n|---|---|---|---|")
for c, (_, _, bad) in LAST.items():
    for g in sorted(k for k in bad if isinstance(k, str)):
        print(f"| {c} | {g} | {bad[g, 'calls'][0]} / {bad[g, 'calls'][1]} | {bad[g]} |")
