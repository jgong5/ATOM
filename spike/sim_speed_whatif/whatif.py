"""whatif.py OUT [BASE]: the wall the 8P16D sim's profiling window would take with one LP group's own work
halved, the critical path recomputed (#4196). OUT is a run of #4029's `run.sh sim`: its aiperf.log and its
stamped clock frames OUT.frames.

1. Replay. The Clock Authority's frames, in order, as #4185's reactions.py replays them: a request follows its
   LP's last reply by its measured own work and transport; a reply follows, by its measured serve, the later of
   its LP's own request and the request whose serve made it due. Every grant keeps the request that freed it, so
   the replayed window is its longest path, and unchanged it gives the measured wall.
2. What-if. From aiperf's profiling start, each request of the group's LPs comes earlier by half its own work,
   as floors.py (#4153) times it: from the reply received to the request sent. The Rust router stamps nothing:
   its own work holds its transport, as there. Transport, CA queueing and serves stay as measured. A DP rank's
   wait on a peer is unshortened: its reply still follows its group's last request, as the CA joins the round,
   and moves only as far as that request does.
3. Check, with BASE, a run of the same recipe without the cut OUT measures. Client metrics: every field of
   BASE's profile_export_aiperf.json but paths, URLs and the command line, against OUT's. Then BASE is replayed
   with its own work cut to OUT's, group by group, two ways: each request's own work at its quantile of OUT's
   (the cut lands on the hops it shortened), and every request's scaled by the group's total, OUT's over BASE's
   (a cut spread over every hop, as a halving is). Each saving it predicts is set against the measured one,
   BASE's wall less OUT's.
"""

import bisect
import collections
import datetime
import glob
import json
import re
import sys


def group(name):  # an LP group: every frontend-D, every rank of every engine-D, ...
    return re.sub(r"\d+(/dp\d+)?$", "*", name)


def norm(addr):
    m = re.match(r"\(LpId\(name='([^']+)'\), '([^']+)'\)", addr)
    return f"{m.group(1)}/{m.group(2)}" if m else addr


def latest(xs, t):  # the latest of the sorted xs at or before t, or None
    i = bisect.bisect_right(xs, t) - 1
    return xs[i] if i >= 0 else None


def load(out):
    """The window and the CA's events up to its end, each with the own work in its offset from the last reply."""
    rx = r"^(\S+ \S+) - ClockPhaseRunner - \w+ - Phase profiling \(profiling\) (started|sending complete)\b(.*)"
    marks, sim = {}, None
    for line in open(f"{out}/artifacts/logs/aiperf.log", errors="replace"):
        if m := re.match(rx, line):
            utc = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
            marks.setdefault(m.group(2), utc.replace(tzinfo=datetime.timezone.utc).timestamp())
            sim = float(d.group(1)) if (d := re.search(r"target: ([\d.]+)s duration", m.group(3))) else sim
    t0, t1 = marks["started"], marks["sending complete"]
    ev, S, R = [], collections.defaultdict(list), collections.defaultdict(list)
    for path in glob.glob(f"{out}.frames/*.trace"):
        for line in open(path):
            t, e, tag, rest = (line.rstrip("\n").split(" ", 3) + [""])[:4]
            if float(t) > t1:
                continue
            if tag == "CA":
                ev.append((float(t), e == "A", norm(rest), "") if e == "A" else
                          (float(t), False, norm(rest.rsplit(" ", 2)[0]), rest.rsplit(" ", 2)[1]))
            else:
                (S if e == "S" else R)[tag].append(float(t))
    ev.sort(key=lambda v: (v[0], v[1]))
    for v in (*S.values(), *R.values()):
        v.sort()
    own, last, span, spans, dist = [], {}, collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for t, is_a, name, _ in ev:
        w = 0.0
        if is_a:
            if t >= t0:  # an LP's own work between two replies: its offset's own work at its last send
                spans[group(name)] += span.pop(name, 0.0)
            last[name] = t
        elif (oa := last.get(name)) is not None:
            if name not in S:  # the router
                w = t - oa
            elif (s := latest(S[name], t)) is not None and (r := latest(R[name], s)) is not None:
                w = min(max(s - max(r, oa), 0.0), t - oa)
            span[name] = max(span[name], w) if t >= t0 else 0.0
            if t >= t0:
                dist[group(name)].append(w)
        own.append(w)
    return {"t0": t0, "t1": t1, "sim": sim, "ev": ev, "own": own, "spans": spans,
            "dist": {g: sorted(v) for g, v in dist.items()}}


def replay(c, cut=lambda g, w: 0.0):
    """The replayed window in wall s per simulated s, `cut(group, own work)` taken off each request's offset."""
    t0, last_a, pend, trig, end = c["t0"], {}, {}, (c["t0"], c["t0"]), c["t0"]
    for (t, is_a, name, kind), w in zip(c["ev"], c["own"]):
        if not is_a:
            oa, na = last_a.get(name, (t, t))
            nq = na + t - oa - (cut(group(name), w) if t >= t0 else 0.0)
            if kind != "REPORT":
                pend[name] = nq
            trig = (t, nq)
        else:
            nq = max(trig[1], pend.pop(name, trig[1])) + t - trig[0]
            last_a[name] = (t, nq)
        end = max(end, nq)
    return (end - t0) / c["sim"]


def fields(out):
    def flat(d, p=""):
        if isinstance(d, (dict, list)):
            for k, v in d.items() if isinstance(d, dict) else enumerate(d):
                yield from flat(v, f"{p}.{k}")
        elif not re.search(r"\.(dir|path|urls\.\d+|cli_command)$", p):
            yield p, d
    return dict(flat(json.load(open(f"{out}/artifacts/profile_export_aiperf.json"))))


out = sys.argv[1]
c = load(out)
wall = (c["t1"] - c["t0"]) / c["sim"]
groups = sorted(c["spans"], key=c["spans"].get, reverse=True)
whatif = {g: replay(c, lambda h, w, g=g: w / 2 if h == g else 0.0) for g in groups}
print(f"## What each LP group's own work, halved, saves in {out}\n\nWindow: {c['t1'] - c['t0']:.0f} s of wall over "
      f"{c['sim']:.0f} simulated s, **{wall:.2f} wall s per simulated s**; replayed unchanged, {replay(c):.2f}.\n")
print("| rank | LP group halved | its own work per simulated s, its processes summed | replayed wall s per "
      "simulated s | saving | share of the wall |\n|---|---|---|---|---|---|")
for i, g in enumerate(sorted(groups, key=whatif.get), 1):
    print(f"| {i} | {g} | {c['spans'][g] / c['sim']:.2f} | {whatif[g]:.2f} | {wall - whatif[g]:.2f} | "
          f"{100 * (wall - whatif[g]) / wall:.1f}% |")
every = replay(c, lambda h, w: w / 2)
print(f"| - | every LP group | {sum(c['spans'].values()) / c['sim']:.2f} | {every:.2f} | {wall - every:.2f} | "
      f"{100 * (wall - every) / wall:.1f}% |")
if len(sys.argv) > 2:
    base = sys.argv[2]
    a, b = fields(out), fields(base)
    differ = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    print(f"\n### Against {base}\n\nClient metrics: {len(a)} fields here, {len(b)} there, {len(differ)} differ"
          + ("" if not differ else ": " + "; ".join(f"{k} {a.get(k)} / {b.get(k)}" for k in differ[:8])) + ".\n")
    cb = load(base)
    walls = {"here": wall, "there": (cb["t1"] - cb["t0"]) / cb["sim"]}
    scale = {g: c["spans"].get(g, 0.0) / v for g, v in cb["spans"].items() if v}

    def at_quantile(g, w):  # OUT's own work at the quantile w holds in BASE's group
        b, h = cb["dist"][g], c["dist"].get(g) or [w]
        q = (bisect.bisect_left(b, w) + bisect.bisect_right(b, w)) / 2 / len(b)
        return w - h[min(int(q * len(h)), len(h) - 1)]

    print("| LP group | own work per simulated s, there / here | own work per request, p50 / p90 / p99, ms, there | "
          "here |\n|---|---|---|---|")
    for g in sorted(scale, key=lambda g: cb["spans"][g], reverse=True):
        q = [" / ".join(f"{1e3 * d[int(f * (len(d) - 1))]:.2f}" for f in (0.5, 0.9, 0.99))
             for d in (cb["dist"][g], c["dist"].get(g, [0.0]))]
        print(f"| {g} | {cb['spans'][g] / cb['sim']:.3f} / {c['spans'].get(g, 0) / c['sim']:.3f} | {q[0]} | {q[1]} |")
    saved = walls["there"] - walls["here"]
    print(f"\nMeasured: {walls['there']:.2f} there, {walls['here']:.2f} here, saving **{saved:.3f}** wall s per "
          f"simulated s. Replayed there unchanged: {replay(cb):.2f}.\n\n| BASE replayed with its own work cut to "
          f"OUT's | replayed wall s per simulated s | saving | against the measured saving |\n|---|---|---|---|")
    for label, cut in (("each request at its quantile", at_quantile),
                       ("every request scaled by its group's total", lambda g, w: (1 - scale.get(g, 1.0)) * w)):
        told = walls["there"] - (p := replay(cb, cut))
        print(f"| {label} | {p:.2f} | {told:.3f} | {100 * (told - saved) / saved:+.1f}% |")
