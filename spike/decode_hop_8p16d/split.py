"""split.py OUT [SLICE_S]: split each simulated decode engine member's hop in OUT's profiling window, from its clock
grant to its next clock request, into its hand-offs (#4151), from the stamps hopstamps.py wrote into OUT.frames.

A hop runs from a grant (R) to the member's next request that waits for one (S; a REPORT does not). The time between
two consecutive stamps goes to one part, so the parts sum to the hop. The stamps are the member's own, except that
from the ring's write (Qw) to the reply's put (O1) the forward's round trip runs on its worker's. A part is named by
the point stamp that opens it (POINT), else by the innermost open call (REGION), else it is the busy loop. A
shared-memory exchange (X0-X1), the DP group's collectives in the engine and in the worker, splits at the latest
entry of the same call across the group's ranks, matched by count: before it, waiting for the slowest rank.

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
POINT = {  # the part from this stamp to the next
    "C0": "dispatch: RPC encode (AsyncIOProcManager.call_func)",
    "Q0": "dispatch: pickle and ring write (MessageQueue.enqueue)",
    "Qw": "dispatch: doorbell ring and worker wake-up (RingingQueue.enqueue, Doorbell.spin)",
    "W0": "dispatch: dequeue and unpickle (AsyncIOProc.get_func)",
    "W1": "worker: dispatch to forward (AsyncIOProc.busy_loop)",
    "F1": "reply: worker output thread wake-up (send_output_to_socket)",
    "wget": "reply: pickle, zmq, engine output thread (process_output_sockets)",
    "O1": "reply: engine owner wake-up (outputs_queue.get)",
}
REGION = {  # opener: (closer, the part inside it); an exchange inside it is named by it too
    "R": ("R1", "engine core: grant's release walk (LPRuntime._grant)"),
    "D0": ("D1", "engine core: DP state sync (DPEngineCoreProc._sync_dp_state)"),
    "Y0": ("Y1", "engine core: settle of owed rounds (LPRuntime.settle)"),
    "P0": ("P1", "engine core: engine step (EngineCore._process_engine_step)"),
    "H0": ("H1", "engine core: scheduling (Scheduler.schedule)"),
    "C0": ("C1", "engine core: call_func"),
    "K0": ("K1", "engine core: KV transfer poll round trip (call_func_with_aggregation)"),
    "Z0": ("Z1", "engine core: postprocess (EngineCore._postprocess)"),
    "A0": ("A1", "engine core: clock charge (LPRuntime.advance_group)"),
    "F0": ("F1", "forward stand-in: rest of NonAllocatingRunner.forward"),
    "r0": ("r1", "forward stand-in: step record (parity.record_step)"),
    "t0": ("t1", "forward stand-in: token ids (DeferredTokenStream.step)"),
    "g0": ("g1", "forward stand-in: pricing and group target (_group_step_seconds)"),
    "m0": ("m1", "forward stand-in: forward metrics (_time_target)"),
    "X0": ("X1", None),
}
CLOSES = {c: o for o, (c, _) in REGION.items()}
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
    names, role = head["events"], head["role"]
    return head, sorted((a[i], names[int(a[i + 1]) // 100], int(a[i + 1]) % 100, role) for i in range(0, len(a), 2))


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
    (members if head["role"] == "engine" else workers)[head["tag"]] = ev  # a worker carries its member's tag


def exchanges(series):
    """Per rank, its exchanges' entries; per LP group, the k-th exchange's latest entry across its ranks, how many
    exchanges each rank made (the fewest, the most), and how many matched ones' exits spread over 2 ms."""
    by_group, entries, last, check = collections.defaultdict(dict), {}, {}, {}
    for tag, ev in series.items():
        entries[tag] = [t for t, e, _, _ in ev if e == "X0"]
        by_group[tag.split("/")[0]][tag] = [t for t, e, _, _ in ev if e == "X1"]
    for g, exits in by_group.items():
        ins = [entries[tag] for tag in exits]
        n = min(map(len, ins))
        last[g] = [max(x[k] for x in ins) for k in range(n)]
        m = min(n, *map(len, exits.values()))
        spread = sum(max(x[k] for x in exits.values()) - min(x[k] for x in exits.values()) > 2e-3 for k in range(m))
        check[g] = (n, max(map(len, ins)), spread)
    return entries, last, check


XCH = {"engine": exchanges(members), "worker": exchanges(workers)}


def track(stack, e):
    """`stack`, the calls open, after stamp `e`."""
    if e in CLOSES and CLOSES[e] in stack:
        del stack[len(stack) - 1 - stack[::-1].index(CLOSES[e]) :]
    elif e in REGION:
        stack.append(e)


def split(tag, ev, i, j, stack):
    """The parts of member `tag`'s hop from its grant ev[i] to its request ev[j], `stack` open at the grant."""
    chain = ev[i : j + 1]
    fwd = [k for k, x in enumerate(chain) if x[1] == "C0" and x[2] in (11, 12)]
    if fwd and tag in workers:
        qw = next((n for n in range(fwd[-1], len(chain)) if chain[n][1] == "Qw"), None)
        o1 = next((n for n in range(fwd[-1], len(chain)) if chain[n][1] == "O1"), None)
        if qw is not None and o1 is not None:
            wev = workers[tag]
            lo, hi = bisect.bisect_right(wev, (chain[qw][0], "~")), bisect.bisect_left(wev, (chain[o1][0], ""))
            chain = chain[: qw + 1] + [x for x in wev[lo:hi] if x[1] != "wsent"] + chain[o1:]
    parts, stack = collections.Counter(), list(stack)
    for (ta, a, _, src), (tb, b, _, _) in zip(chain, chain[1:]):
        track(stack, a)
        if "K0" in stack:  # its round trip's own stamps are not split
            parts[REGION["K0"][1]] += tb - ta
            continue
        inside = next((REGION[o][1] for o in reversed(stack) if REGION[o][1]), "engine core: busy loop")
        if a == "X0":
            entries, last, _ = XCH[src]
            ks, lg = entries.get(tag, []), last.get(tag.split("/")[0], [])
            k = bisect.bisect_left(ks, ta)
            wait = min(max(lg[k] - ta, 0.0), tb - ta) if k < len(lg) and ks[k] == ta else 0.0
            who = inside.split(" (")[0]
            parts[f"{who}: waiting for the slowest rank"] += wait
            parts[f"{who}: shared-memory exchange (_Segment.exchange)"] += tb - ta - wait
            continue
        parts[POINT.get(a) or inside] += tb - ta
    return parts


def hops_of(ev):
    """(grant index, request index, what the grant answered, the calls open at the grant) for each hop whose request
    falls in the window."""
    out_, r, kind, stack = [], None, None, []
    for n, (t, e, d, _) in enumerate(ev):
        if e == "S" and d != 4:
            if r is not None and t0 <= t < t1:
                out_.append((r, n, kind, at_r))
            r = None
            kind = "settle" if "Y0" in stack else "advance" if "A0" in stack else {2: "NER", 3: "PARK"}.get(d, "TAR")
        elif e == "R":
            r, at_r = n, list(stack)
        track(stack, e)
    return out_


everything, by_kind, stamped = collections.Counter(), collections.defaultdict(list), {}
for tag, ev in sorted(members.items()):
    stamped[tag] = []
    for i, j, kind, stack in hops_of(ev):
        p = split(tag, ev, i, j, stack)
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
      f"stamped, {sum(tag in workers for tag in members)} of them with their worker.\n")
print("| hop opened by the grant of | hops | mean ms | median ms |\n|---|---|---|---|")
for kind, v in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
    v.sort()
    print(f"| {kind} | {len(v)} | {1e3 * sum(v) / len(v):.3f} | {1e3 * v[len(v) // 2]:.3f} |")
report(crit, crit["#hops"], "Engine-D hops on the critical path", crit_slices)
print(f"\nThose hops' wall: {match['stamps s']:.2f} s by the stamps, {match['frames s']:.2f} s by the frames' own work "
      f"({100 * (match['stamps s'] / max(match['frames s'], 1e-9) - 1):+.2f}%); {match['unmatched hops']:.0f} hops "
      f"({match['unmatched s']:.2f} s) on the path matched no stamped hop.")
report(everything, sum(len(v) for v in by_kind.values()), "Every engine-D hop")
print("\n## Exchange matching\n\n| ranks | LP group | exchanges matched / most by one rank | matched ones whose exits "
      "spread over 2 ms |\n|---|---|---|---|")
for role, (_, _, check) in XCH.items():
    for g, (n, most, spread) in sorted(check.items()):
        print(f"| {role} | {g} | {n} / {most} | {spread} |")
