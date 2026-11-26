# SPDX-License-Identifier: MIT
"""Fit a pipelined prefill stage's serving of its KV writes (the link's
``serve_*`` fields, `atom/compass/kv/transfer.py` `ServeModel`) on a recorded
Mooncake P/D run, and replay the record's transfers on it.

    serve_fit.py RECORD_DIR --network NETWORK [--points c24:600,...] [--out NETWORK_OUT]

RECORD_DIR is a `deploy.sh record` run of a PP prefill: its
``real/prefill/steps`` hold each stage's step lines (``dp0.jsonl``,
``pp<k>.dp0.jsonl``) and KV writes (``dp0.tp0.writes.jsonl``,
``pp<k>.dp0.tp0.writes.jsonl``: per write its receipt and its ``_rdma_write``
calls), and each client point its ``artifacts/profile_export_raw.jsonl.gz``.

Each stage's forwards make its `DeviceQueue`: a forward's launch
``t_launch_ns`` and its device run, from its launch event for ``d_busy_ns``,
no earlier than the forward before it ends. From the writes of the whole
record whose stage's device had caught up at their receipt (within
`IDLE_S`), it states:

- ``serve_chunk_bytes``: the calls' most common size (Mooncake's slice);
- ``serve_call_bandwidth_bytes_per_s`` and ``serve_call_s``: a full call's
  median rate, and the median gap between a write's calls;
- ``serve_gather_s``, ``serve_gather_one_s`` and ``serve_stagger_s``: the
  receipt-to-first-call delay of write ``j`` of a stage's writes of one
  transfer, in the order their first calls start, least squares on
  ``gather_s + (gather_one_s + j x stagger_s) / calls`` over the medians of
  each (calls, j);

and ``serve_device_wait``, the share of its device's backlog a call waits
(with --sim POINT=DIR, scored on that simulated run of the point instead:
its transfers replayed on its own stages' queues from its cost records, which
is what a full simulation prices, against the record's),
with the constant the gather's least squares
under-reads (a transfer waits for the slowest of its stages' writes), the
grid values whose replay of the POINTS' transfers comes closest to them:
per point, and per size band (`BANDS`), the p50 and p90 of a transfer's
receipt to its last call's end, scored as the mean absolute log ratio. The
replay runs `StageServe` on each stage's `DeviceQueue`, as a simulated stage
does, each stage's writes from its first receipt; a transfer ends at its last
stage's. It prints the replay beside the link's lone-write law (each stage's
bytes at the link's rate, the stages in parallel). With --out, NETWORK with
the fitted fields added, probed, each band its estimates' p10 and p90 over
the stages (``serve_device_wait``: over the points' own best values).
"""

import argparse
import bisect
import collections
import gzip
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

#: A write's stage device has caught up at its receipt within this.
IDLE_S = 1e-3
#: The transfer size bands the replay is scored on, bytes.
BANDS = ((0.0, 0.5e9, "<0.5GB"), (0.5e9, 2e9, "0.5-2GB"), (2e9, math.inf, ">2GB"))
#: The candidate shares of the backlog a call waits.
WAIT_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
#: The candidate seconds the transfer-level fit adds to the gather's constant:
#: a transfer waits for the slowest of its stages' writes, which the medians
#: of single writes under-read.
GATHER_GRID = tuple(i * 1e-3 for i in range(9))


def pct(values, p):
    v = sorted(values)
    if not v:
        return math.nan
    k = (len(v) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def stage_files(steps: Path):
    """Each stage's (steps file, writes file), stage order."""
    out = [(steps / "dp0.jsonl", steps / "dp0.tp0.writes.jsonl")]
    k = 1
    while (steps / f"pp{k}.dp0.jsonl").exists():
        out.append((steps / f"pp{k}.dp0.jsonl", steps / f"pp{k}.dp0.tp0.writes.jsonl"))
        k += 1
    return out


def queues(files):
    """Each stage's `DeviceQueue` over the whole record; the last stage's
    empty, as a serving last stage keeps it (`connector.served_forward`): on
    MiniMax-M3.1's CAL its writes moved at the link's pace however far its
    host had run ahead (49 GB in 0.67 s at c120 while the middle stages' took
    9.3 s)."""
    from atom.compass.kv.transfer import DeviceQueue

    out = []
    for i, (steps, _) in enumerate(files):
        if i == len(files) - 1:
            out.append(DeviceQueue())
            continue
        rows = []
        for line in steps.open():
            if '"t_launch_ns"' not in line:
                continue
            d = json.loads(line)
            if d.get("is_dummy_run"):
                continue
            l0, l1 = d["t_launch_ns"]
            ev = d.get("t_launch_event_ns") or l0
            rows.append((l0 / 1e9, l1 / 1e9, ev / 1e9, d.get("d_busy_ns", 0) / 1e9))
        rows.sort()
        q, end = DeviceQueue(), -math.inf
        for l0, l1, ev, busy in rows:
            d0 = max(ev, end)
            end = d0 + busy
            q.add(l0, l1, d0, end)
        out.append(q)
    return out


def transfers(files):
    """By transfer id, each stage's writes: (receipt s, [(start s, end s,
    bytes)]); and `sha256:` over the writes files read."""
    found = collections.defaultdict(lambda: collections.defaultdict(list))
    digest = hashlib.sha256()
    for k, (_, writes) in enumerate(files):
        data = writes.read_bytes()
        digest.update(data)
        for line in data.decode().splitlines():
            w = json.loads(line)
            if not w.get("rdma") or not w.get("success", True):
                continue
            calls = sorted((a / 1e9, b / 1e9, n) for a, b, n in w["rdma"])
            found[w["transfer_id"]][k].append((w["t_receipt_ns"] / 1e9, calls))
    return found, f"sha256:{digest.hexdigest()}"


def windows(record: Path, points):
    """Each point's client window on the record's clock: its profiling
    requests' first start to their last start."""
    out = {}
    for p in points:
        raw = record / p / "artifacts" / "profile_export_raw.jsonl.gz"
        starts = []
        for line in gzip.open(raw, "rt"):
            d = json.loads(line)
            if d["metadata"].get("benchmark_phase") == "profiling":
                starts.append(d["start_perf_ns"] / 1e9)
        out[p] = (min(starts), max(starts))
    return out


def idle_writes(found, qs):
    """The writes whose stage's device had caught up at their receipt, as
    (calls, j, delay s, gaps, calls' (bytes, seconds)), j the write's place
    among its stage's writes of one transfer by first call."""
    out = []
    for stages in found.values():
        for k, ws in stages.items():
            ws = sorted(ws, key=lambda w: w[1][0][0])
            for j, (receipt, calls) in enumerate(ws):
                if qs[k].reached(receipt) - receipt > IDLE_S:
                    continue
                gaps = [calls[i + 1][0] - calls[i][1] for i in range(len(calls) - 1)]
                out.append(
                    (len(calls), j, calls[0][0] - receipt, gaps, [(n, b - a) for a, b, n in calls])
                )
    return out


def fit_gather(idle):
    """Least squares of delay on (1, 1/n, j/n) over the medians of each (n, j)."""
    cells = collections.defaultdict(list)
    for n, j, delay, *_ in idle:
        cells[min(n, 64), j].append(delay)
    rows = [(1.0, 1.0 / n, j / n, statistics.median(v)) for (n, j), v in cells.items() if len(v) >= 5]
    # Normal equations, 3 x 3.
    a = [[sum(r[i] * r[k] for r in rows) for k in range(3)] for i in range(3)]
    b = [sum(r[i] * r[3] for r in rows) for i in range(3)]
    for i in range(3):
        p = a[i][i]
        for k in range(i + 1, 3):
            f = a[k][i] / p
            a[k] = [x - f * y for x, y in zip(a[k], a[i])]
            b[k] -= f * b[i]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (b[i] - sum(a[i][k] * x[k] for k in range(i + 1, 3))) / a[i][i]
    return x


def replay(found, qs, window, serve, lone):
    """Each transfer of `window` (first receipt inside it): (bytes, real s,
    served s, lone-law s)."""
    from atom.compass.kv.transfer import StageServe

    out = []
    lo, hi = window
    for stages in found.values():
        if len(stages) != len(qs):
            continue
        receipts = [w[0] for ws in stages.values() for w in ws]
        t0 = min(receipts)
        if not lo <= t0 <= hi:
            continue
        nbytes = sum(n for ws in stages.values() for _, calls in ws for *_, n in calls)
        end = max(c[1] for ws in stages.values() for _, calls in ws for c in calls)
        served, law = -math.inf, -math.inf
        for k, ws in stages.items():
            a = min(w[0] for w in ws)
            b = sum(n for _, calls in ws for *_, n in calls)
            served = max(served, StageServe(serve, a, b).advance(math.inf, qs[k]))
            law = max(law, a + lone[0] + b / lone[1])
        out.append((nbytes, end - t0, served - t0, law - t0))
    return out


SERVED = re.compile(
    r"compass: KV transfer (\S+) served on stage (\d+), (\d+) blocks, from (\S+) to (\S+) simulated s"
)
LAW = re.compile(r"compass: KV transfer (\S+), (\d+) blocks, from (\S+) to (\S+) simulated s")


def sim_queues(dep: Path):
    """A simulated prefill's stages' queues, from its cost records: a stage
    before the last launches over the W_launch before its exit (t_s +
    charge_s) less W_post, and its device runs device_s + draft_s (the
    queue chains them, `DeviceQueue.add`); the last stage's queue is empty."""
    from atom.compass.kv.transfer import DeviceQueue

    stages = len(list(dep.glob("now-engine-P-pp*.kv_block")))
    out = []
    for k in range(stages):
        q = DeviceQueue()
        if k < stages - 1:
            name = "dp0" if k == 0 else f"pp{k}.dp0"
            rows = []
            for line in (dep / f"prefill/steps/{name}.cost.jsonl").open():
                d = json.loads(line)
                if "step" not in d:
                    continue
                l1 = d["t_s"] + d["charge_s"] - d.get("W_post", 0.0)
                l0 = l1 - d.get("W_launch", 0.0)
                rows.append((l0, l1, d.get("device_s", 0.0) + d.get("draft_s", 0.0)))
            for l0, l1, busy in sorted(rows):
                q.add(l0, l1, l0, l0 + busy)
        out.append(q)
    return out


def sim_transfers(sim: Path, point):
    """A simulated run's transfers in its point's profiling window: (arrival
    s, [(stage, bytes)]), from its transfer lines (a stage's served or the
    link law's), each stage's bytes its planned block at the decode landing."""
    dep = sim / "deploy"
    stages = len(list(dep.glob("now-engine-P-pp*.kv_block")))
    pb = [json.loads((dep / f"now-engine-P-pp{k}.kv_block").read_text())["block_bytes"] for k in range(stages)]
    db = json.loads((dep / "now-engine-D.kv_block").read_text())
    landing = db["tp"] * db["block_bytes"] / sum(pb)
    seen = {}
    for line in (dep / "prefill/server.log").open(errors="replace"):
        m = SERVED.search(line)
        if m:
            seen[m[1]] = (int(m[3]), float(m[4]))
            continue
        m = LAW.search(line)
        if m and m[1] not in seen:
            seen[m[1]] = (int(m[2]), float(m[3]))
    starts = []
    for line in gzip.open(sim / point / "artifacts" / "profile_export_raw.jsonl.gz", "rt"):
        d = json.loads(line)
        if d["metadata"].get("benchmark_phase") == "profiling":
            starts.append(d["metadata"]["request_start_ns"] / 1e9)
    lo, hi = min(starts), max(starts)
    return [
        (a, [(k, blocks * round(pb[k] * landing)) for k in range(stages)])
        for blocks, a in seen.values()
        if lo <= a <= hi
    ]


def sim_replay(sim, qs, serve):
    """Each simulated transfer: (bytes, served s) on the sim's own queues."""
    from atom.compass.kv.transfer import StageServe

    return [
        (
            sum(b for _, b in per),
            max(StageServe(serve, a, b).advance(math.inf, qs[k]) for k, b in per) - a,
        )
        for a, per in sim
    ]


def score_sim(real_rows, sim_rows):
    """`score`'s form over a simulated run's replay against the record's."""
    errs, table = [], []
    for lo, hi, name in BANDS:
        r = [x[1] for x in real_rows if lo <= x[0] < hi]
        m = [x[1] for x in sim_rows if lo <= x[0] < hi]
        if len(r) < 5 or len(m) < 5:
            continue
        line = [name, len(m)]
        for p in (50, 90):
            a, b = pct(r, p), pct(m, p)
            errs.append(abs(math.log(b / a)))
            line += [a, b, math.nan]
        table.append(line)
    return (statistics.fmean(errs) if errs else math.inf), table


def score(rows):
    """Mean |log(model / real)| over the bands' p50 and p90, and the table."""
    errs, table = [], []
    for lo, hi, name in BANDS:
        sub = [r for r in rows if lo <= r[0] < hi]
        if len(sub) < 5:
            continue
        line = [name, len(sub)]
        for p in (50, 90):
            real, model, law = (pct([r[i] for r in sub], p) for i in (1, 2, 3))
            errs.append(abs(math.log(model / real)))
            line += [real, model, law]
        table.append(line)
    return (statistics.fmean(errs) if errs else math.inf), table


def main(argv=None) -> int:
    from atom.compass.kv.transfer import ServeModel
    from atom.compass.spec.layers import network

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record")
    parser.add_argument("--network", required=True, help="the link the run prices")
    parser.add_argument("--points", default="c24:600,c64:600,c120:600")
    parser.add_argument("--writes", type=int, default=4, help="decode's TP: writes a stage")
    parser.add_argument("--out", help="NETWORK with the fitted serve fields added")
    parser.add_argument(
        "--sim",
        action="append",
        default=[],
        metavar="POINT=DIR",
        help="score device_wait and the gather on a simulated run of POINT (its "
        "probe dir) replayed on its own stages' queues, not on the record's",
    )
    args = parser.parse_args(argv)
    record = Path(args.record)
    net = network(args.network)
    link = "interconnect.inter_node"
    lone = (
        net.values[f"{link}.link_latency_s"],
        net.values[f"{link}.link_bandwidth_bytes_per_s"] * net.values[f"{link}.derate"],
    )
    files = stage_files(record / "real/prefill/steps")
    qs = queues(files)
    found, digest = transfers(files)
    idle = idle_writes(found, qs)
    sizes = collections.Counter(n for *_, calls in idle for n, _ in calls)
    chunk = sizes.most_common(1)[0][0]
    full = [(n, s) for *_, calls in idle for n, s in calls if n == chunk and s > 0]
    rates = [n / s for n, s in full]
    gaps = [g for *_, gs, _ in idle for g in gs if g >= 0]
    gather, gather_one, stagger = fit_gather(idle)
    base = dict(
        chunk_bytes=chunk,
        call_bandwidth_bytes_per_s=statistics.median(rates),
        call_s=statistics.median(gaps),
        gather_s=max(gather, 1e-6),
        gather_one_s=max(gather_one, 1e-6),
        stagger_s=max(stagger, 1e-6),
    )
    print(
        f"{len(found)} transfers over {len(files)} stages, writes files {digest}; "
        f"{len(idle)} writes with their stage's device caught up at receipt"
    )
    print(
        f"chunk {chunk} B; call {base['call_bandwidth_bytes_per_s'] / 1e9:.3g} GB/s "
        f"(p10/p90 {pct(rates, 10) / 1e9:.3g}/{pct(rates, 90) / 1e9:.3g}) + "
        f"{base['call_s'] * 1e3:.3g} ms a gap; first call after "
        f"{base['gather_s'] * 1e3:.3g} + ({base['gather_one_s'] * 1e3:.3g} + j x "
        f"{base['stagger_s'] * 1e3:.3g}) / calls ms"
    )
    points = [p for p in args.points.split(",") if p]
    wins = windows(record, points)
    sims = dict(x.split("=", 1) for x in args.sim)
    if missing := sorted(set(sims) - set(points)):
        raise SystemExit(f"--sim names {missing}, not among --points {points}")
    simdata = {
        p: (sim_queues(Path(d) / "deploy"), sim_transfers(Path(d), p)) for p, d in sims.items()
    }

    def point_score(p, serve):
        """Point `p`'s score: on its simulated run's queues where --sim names one."""
        if p in simdata:
            qs_p, sim = simdata[p]
            real = [(r[0], r[1]) for r in replay(found, qs, wins[p], serve, lone)]
            return score_sim(real, sim_replay(sim, qs_p, serve))
        return score(replay(found, qs, wins[p], serve, lone))

    if sims:
        print(f"device_wait and the gather scored on simulated runs: {sims}")
    best, per_point = None, {}
    for extra in GATHER_GRID:
        for wait in WAIT_GRID:
            serve = ServeModel(
                **{**base, "gather_s": base["gather_s"] + extra},
                device_wait=wait,
                writes=args.writes,
            )
            total = []
            for p in points:
                s, _ = point_score(p, serve)
                total.append(s)
                if p not in per_point or s < per_point[p][0]:
                    per_point[p] = (s, wait)
            mean = statistics.fmean(total)
            if best is None or mean < best[0]:
                best = (mean, wait, extra)
        print(
            f"gather +{extra * 1e3:.0f} ms: best so far device_wait {best[1]}, "
            f"gather +{best[2] * 1e3:.0f} ms, score {best[0]:.3f}"
        )
    base["gather_s"] += best[2]
    serve = ServeModel(**base, device_wait=best[1], writes=args.writes)
    print(
        f"device_wait {best[1]}, gather_s {base['gather_s'] * 1e3:.3g} ms, score "
        f"{best[0]:.3f} (per point {dict((p, w) for p, (_, w) in per_point.items())})"
    )
    print(f"{'point':10s} {'band':8s} {'n':>5s}  p50 real/served/law ms     p90 real/served/law ms")
    for p in points:
        _, table = point_score(p, serve)
        for name, n, r5, m5, l5, r9, m9, l9 in table:
            print(
                f"{p:10s} {name:8s} {n:5d}  {r5 * 1e3:8.1f} {m5 * 1e3:8.1f} ({(m5 / r5 - 1) * 100:+4.0f}%) "
                f"{l5 * 1e3:7.1f}   {r9 * 1e3:8.1f} {m9 * 1e3:8.1f} ({(m9 / r9 - 1) * 100:+4.0f}%) {l9 * 1e3:7.1f}"
            )
    if args.out:
        path = Path(args.network)
        doc = json.loads(path.read_text())
        fields = {f"serve_{k}": v for k, v in base.items()} | {"serve_device_wait": best[1]}
        doc["interconnect"]["inter_node"].update(fields)
        prov = doc["provenance"]
        waits = [w for _, w in per_point.values()]
        bands = {
            "serve_chunk_bytes": (chunk, chunk),
            "serve_call_bandwidth_bytes_per_s": (pct(rates, 10), pct(rates, 90)),
            "serve_call_s": (pct(gaps, 10), pct(gaps, 90)),
            "serve_device_wait": (min(waits), max(waits)),
        }
        for f, v in fields.items():
            lo, hi = bands.get(f, (v, v))
            prov.setdefault("origins", {})[f"{link}.{f}"] = {
                "origin": "probed",
                "record": digest,
                "p10": lo,
                "p90": hi,
            }
        prov["notes"] = (
            f"{prov.get('notes', '')} serve_*: `serve_fit.py` over the {len(found)} "
            f"transfers of {record} ({len(idle)} writes on a caught-up device for the "
            f"calls and the gather; serve_device_wait the grid value whose replay of "
            f"{', '.join(points)} comes closest, the band the points' own"
            + (
                f"; scored on the simulated runs {sims} replayed on their own "
                "stages' queues"
                if sims
                else ""
            )
            + ")."
        ).strip()
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        network(args.out)  # the schema reads it back, or refuses
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
