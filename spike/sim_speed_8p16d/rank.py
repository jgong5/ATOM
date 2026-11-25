"""rank.py OUT [SLICE_S]: rank the 8P16D sim's wall by phase, LP and component (#4029), from run.sh's run OUT,
its stamped frames OUT.frames and sample.py's stack samples OUT.spy.

1. Phases. The client's aiperf.log, as the run's artifacts keep it, gives each phase's wall bounds and its
   simulated length: the warmup's elapsed, the profiling duration, and the drain to the phase's end.
2. Critical path. The CA serves one request at a time, so every reply it writes was made due by the request it
   is serving. In each SLICE_S (default 300) slice of the profiling window, from the slice's last request, the
   walk goes back: LP x sent its request after it received its latest reply, which the request the CA was then
   serving made due. A hop on x splits into x's own work (reply received -> request sent), CA queueing (the
   part of the request's way up during which the CA was serving others), transport to and from the CA and
   the CA's serve of the request that freed x. The Rust router stamps nothing: its own work, queueing and
   transport are one row.
3. Components. A sample whose leaf waits (WAITING) is not work. An LP group's own work splits in proportion to
   its processes' other samples, each going to the component of the first frame from the leaf that COMPONENTS
   names, else PLUMBING, else other.
"""

import bisect
import collections
import datetime
import glob
import json
import os
import re
import sys

out = sys.argv[1]
slice_s = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
WAITING = (r"SocketIO\.readinto|\(socket\.py|selectors\.py|CompassSelector\.select|Poller\.poll|recv_multipart"
           r"|Condition\.wait|Event\.wait|RelayQueue\.get|queue\.py|_until_end|take_grant|time\.sleep")
COMPONENTS = [
    ("tokenizer encode", r"tokenizers|encode_batch|token_ids\.py|segment_split|[Tt]okeniz"),
    ("response encoding (SSE, JSON dumps)", r"^(dumps|JSONEncoder\.\w+|iterencode) |model_dump|serializ|stream_gen"),
    ("request parsing (JSON loads, pydantic)", r"^(loads|JSONDecoder\.\w+) |validate_python|TypeAdapter|pydantic"),
    ("traffic LP scans (traffic_lp.py)", r"traffic_lp\.py"),
    ("Store LP settle (StationLP._settle)", r"StationLP\._settle"),
    ("Store master and stations", r"mooncake_store_server\.py|lmcache_server\.py"),
    ("scheduling", r"scheduler\.py|block_manager|prefill_delayer|prefix_cach"),
    ("pricing (step and host laws)", r"compass/backends/|step_linear|host_law|step_grid"),
    ("request and sequence processing", r"sequence\.py|llm_engine\.py|io_processor|preprocess"),
    ("DP collectives and engine IPC", r"shm_collectives|shm_broadcast|doorbell\.py|async_proc\.py|zmq"),
    ("metrics", r"prometheus_client|metrics\.py"),
    ("clock runtime and frames", r"clock_transport/|utils/clock\.py|compass/clock/"),
    ("compass sim runtime (executor, HTTP channel, carriers)", r"compass_loop\.py|carriers\.py|compass/runner/"),
]
PLUMBING = ("asyncio and HTTP plumbing", r"asyncio/|uvicorn|starlette|fastapi|anyio|h11|httptools|aiohttp")


def group(name):
    return re.sub(r"\d+(/dp\d+)?$", "*", name)


def norm(addr):
    if m := re.match(r"\(LpId\(name='([^']+)'\), '([^']+)'\)", addr):
        return f"{m.group(1)}/{m.group(2)}"
    m = re.match(r"LpId\(name='([^']+)'\)", addr)
    return m.group(1) if m else addr


def latest(xs, t):  # the latest of the sorted xs at or before t, or None
    i = bisect.bisect_right(xs, t) - 1
    return xs[i] if i >= 0 else None


# 1. phases
line_rx = (r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) - ClockPhaseRunner - \w+ - Phase (\w+) \(\w+\) "
           r"(started|sending complete|complete)\b(.*)")
marks, elapsed, duration = {}, {}, None
for line in open(f"{out}/artifacts/logs/aiperf.log", errors="replace"):
    if m := re.match(line_rx, line):
        utc = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
        marks.setdefault((m.group(2), m.group(3)), utc.replace(tzinfo=datetime.timezone.utc).timestamp())
        if e := re.search(r"elapsed=([\d.]+)s", m.group(4)):
            elapsed[m.group(2)] = float(e.group(1))
        if d := re.search(r"target: ([\d.]+)s duration", m.group(4)):
            duration = float(d.group(1))
phases = [("warmup", "warmup", "started", "warmup", "complete", elapsed["warmup"]),
          ("profiling window", "profiling", "started", "profiling", "sending complete", duration),
          ("drain", "profiling", "sending complete", "profiling", "complete", elapsed["profiling"] - duration)]
print(f"## Phases of {out}\n\n| phase | wall s | simulated s | wall per simulated s |\n|---|---|---|---|")
for label, pa, ea, pb, eb, sim in phases:
    wall = marks[(pb, eb)] - marks[(pa, ea)]
    print(f"| {label} | {wall:.0f} | {sim:.1f} | {wall / sim:.2f} |")
t0, t1 = marks[("profiling", "started")], marks[("profiling", "sending complete")]

# 2. critical path, slice by slice
n = int((t1 - t0) // slice_s)
os.makedirs(f"{out}.slices", exist_ok=True)
files = [open(f"{out}.slices/{k}.txt", "w") for k in range(n)]
for path in glob.glob(f"{out}.frames/*.trace"):
    for line in open(path):
        t = float(line[: line.find(" ")])
        k = int((t - t0) // slice_s)
        if 0 <= k < n:
            files[k].write(line)
        if 0 <= k + 1 < n and t - t0 > (k + 1) * slice_s - 5:  # the next slice's first hops reach back
            files[k + 1].write(line)
for f in files:
    f.close()
acc, hops, per_slice = collections.Counter(), collections.Counter(), []
for k in range(n):
    start, here = t0 + k * slice_s, collections.Counter()
    ca, S, R = [], collections.defaultdict(list), collections.defaultdict(list)
    for line in open(f"{out}.slices/{k}.txt"):
        p = line.split(None, 3)
        t, ev, tag = float(p[0]), p[1], p[2]
        if tag == "CA" and ev == "Q":
            ca.append((t, 0, norm(p[3].strip().rsplit(" ", 2)[0])))
        elif tag == "CA":
            ca.append((t, 1, norm(p[3].strip())))
        else:
            (S if ev == "S" else R)[tag].append(t)
    ca.sort()
    for v in (*S.values(), *R.values()):
        v.sort()
    A, Aat, last_q = collections.defaultdict(list), collections.defaultdict(list), None
    serves = []  # [start, end] of each serve: its request read, its last reply written
    for t, is_a, name in ca:
        if not is_a:
            last_q = (t, name)
            serves.append([t, t])
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

    cur_t, cur = next((t, name) for t, is_a, name in reversed(ca) if not is_a)
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
            here[(g, "CA queueing")] += queued
            here[(g, "transport to CA")] += cur_t - s - queued
            here[(g, "transport from CA")] += max(r - a, 0.0)
            here[(g, "own work")] += s - max(r, a)
        else:
            if (a := latest(Aat[cur], cur_t)) is None:
                break
            here[(g, "own work, queueing and transport (no stamps)")] += cur_t - a
        _, trig_t, trig = A[cur][bisect.bisect_right(Aat[cur], a) - 1]
        here[(group(trig), "CA serve")] += a - trig_t
        hops[g] += 1
        cur_t, cur = trig_t, trig
    acc.update(here)
    per_slice.append(here)

# 3. components of own work
samples, waiting = collections.defaultdict(collections.Counter), collections.Counter()
others = collections.defaultdict(collections.Counter)
for path in glob.glob(f"{out}.spy/*.raw"):
    g = group(os.path.basename(path).split(".")[0])
    for line in open(path):
        stack, _, count = line.rstrip("\n").rpartition(" ")
        frames = stack.split(";")[1:][::-1]
        if frames and re.search(WAITING, frames[0]):
            waiting[g] += int(count)
            continue
        label = next((lab for f in frames for lab, rx in COMPONENTS if re.search(rx, f)), None)
        if label is None:
            label = PLUMBING[0] if any(re.search(PLUMBING[1], f) for f in frames) else "other"
        samples[g][label] += int(count)
        if label == "other":
            others[g][frames[0] if frames else "(no Python frame)"] += int(count)
rows = collections.Counter()
for (g, part), v in acc.items():
    tot = sum(samples.get(g, {}).values())
    for label, c in samples[g].items() if part == "own work" and tot else [(None, 1)]:
        rows[(g, f"own work: {label}" if label else part)] += v * c / (tot if label else 1)
walked = sum(acc.values())
print(f"\n## Profiling window: {t1 - t0:.0f} s wall, {n} slices of {slice_s:.0f} s; critical path walked {walked:.0f} s "
      f"({100 * walked / (t1 - t0):.1f}% of the window)\n\n| rank | LP group | part | wall s | share |\n|---|---|---|---|---|")
for i, ((g, part), v) in enumerate(rows.most_common(), 1):
    print(f"| {i} | {g} | {part} | {v:.0f} | {100 * v / walked:.1f}% |")
print("\n## By LP group, and each slice's share (min / median / max)\n")
print("| LP group | share | own work | CA queueing | transport to / from CA | CA serve | hops | slices |")
print("|---|---|---|---|---|---|---|---|")
for g, v in collections.Counter({g: sum(v for (h, _), v in acc.items() if h == g) for g in hops}).most_common():
    sh = sorted(sum(v for (h, _), v in s.items() if h == g) / max(sum(s.values()), 1e-9) for s in per_slice)
    part = {p: 100 * sum(v for (h, q), v in acc.items() if h == g and q.startswith(p)) / walked
            for p in ("own work", "CA queueing", "transport to", "transport from", "CA serve")}
    print(f"| {g} | {100 * v / walked:.1f}% | {part['own work']:.1f}% | {part['CA queueing']:.1f}% | "
          f"{part['transport to']:.1f}% / {part['transport from']:.1f}% | "
          f"{part['CA serve']:.1f}% | {hops[g]} | {100 * sh[0]:.1f} / {100 * sh[len(sh) // 2]:.1f} / {100 * sh[-1]:.1f} |")
print("\n## Stack samples by LP group (all threads; waiting samples left out of the split)\n")
for g, c in sorted(samples.items()):
    tot = sum(c.values())
    print(f"- {g}: {tot} working, {waiting[g]} waiting. " + "; ".join(f"{lab} {100 * x / tot:.1f}%" for lab, x in c.most_common()))
    if others[g]:
        print("  - other, by leaf: " + "; ".join(f"{f} {100 * x / tot:.1f}%" for f, x in others[g].most_common(4)))
if os.path.exists(f"{out}.spy/schedstat.json"):
    sched, cpu = json.load(open(f"{out}.spy/schedstat.json")), json.load(open(f"{out}.spy/cpus.json"))
    b = sorted(cpu["busy_per_s"])
    print(f"\n## Scheduling over the sampled {len(b)} s\n\nBusy CPUs of {cpu['cpus']}: median {b[len(b) // 2]:.1f}, "
          f"p90 {b[int(0.9 * len(b))]:.1f}, max {b[-1]:.1f}.\n\n| LP group | CPU s per s | run-queue wait s per s | "
          f"its most-waiting thread, wait s per s |\n|---|---|---|---|")
    per = collections.defaultdict(lambda: [0.0, 0.0, ("", 0.0)])
    for key, (comm, run_s, wait_s) in sched.items():
        e = per[group(key.split(".")[0])]
        e[0], e[1] = e[0] + run_s / len(b), e[1] + wait_s / len(b)
        e[2] = max(e[2], (comm, wait_s / len(b)), key=lambda x: x[1])
    for g, (run_s, wait_s, (comm, worst)) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"| {g} | {run_s:.2f} | {wait_s:.3f} | {comm} {worst:.3f} |")
