# SPDX-License-Identifier: MIT
"""A two-node RDMA fabric under Mooncake: a TransferEngine microbenchmark, and
the link values and write discipline fitted from it and a recorded P/D run.

    rdma_link.py target --ip IP --out TARGETS
    rdma_link.py bench --ip IP --targets TARGETS --out BENCH
    rdma_link.py fit BENCH [RECORD_DIR]

`target` runs on the decode node: an engine per rail N on its HCA (`--hca`,
`ionic_N`), holding `BUFFER` bytes of GPU N, until it is killed. `bench` runs
on the prefill node: a process per GPU, as each prefill rank is one, writing
from its GPU through an engine started on the rail it names, as
`RailEnginePool` starts one under matched rails, in `ENTRY`-byte entries cut
into the connector's batches. Both ends need the settings of a 1P1D
deployment over RDMA (`scripts/compass/README.md`). The bench writes a JSON
line per write: `startup`, a rail engine's start, its first write and the
next; `sweep`, one writer at each of `SIZES`, every GPU on its own rail and
on the next GPU's; `rail`, 1 to 16 writers into one rail from GPUs `i % 8`,
started together each round; `spread`, GPU 0 writing on 1 to 8 rails at once.

`fit` reads only the standard library and `atom.compass.kv.transfer`. It
prints each rail case's link, `latency + bytes / bandwidth` fitted on relative
error, with the p10 and p90 of its fits per round; the start-ups; each
concurrency's finishing spread (the last write's time over the first's) and
aggregate rate; and, given the record directory of a Mooncake P/D run
(`atom/compass/parity/writes.py`), each sharing unit's error when every
`_rdma_write` call is replayed at its recorded start through the `own` link,
one at a time on a `Link` (FIFO) or with the calls in flight splitting the
bandwidth evenly (shared).
"""

import argparse
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

GPUS = 8
ENTRY = 64 << 10
BUFFER = 4 << 30
CHUNK = 2 << 30  # the connector's largest registration
SIZES = [ENTRY << 2 * i for i in range(8)]  # 64 KiB to 1 GiB
REPS = 8
ROUND_BYTES = 256 << 20
FLOOR = 2e9  # rounds are spaced as if writes moved no faster than this


def port_rate(hca: str) -> float:
    """The bytes per second `hca`'s port negotiated."""
    rate = Path(f"/sys/class/infiniband/{hca}/ports/1/rate").read_text()
    return float(rate.split()[0]) * 1e9 / 8


def engine(ip: str, hca: str, buf):
    from mooncake.engine import TransferEngine

    e = TransferEngine()
    ok = e.initialize(ip, "P2PHANDSHAKE", "rdma", hca) == 0
    for off in range(0, buf.numel(), CHUNK):
        size = min(CHUNK, buf.numel() - off)
        ok = ok and e.register_memory(buf.data_ptr() + off, size) == 0
    if not ok:
        raise RuntimeError(f"a Mooncake engine on {hca} did not start or register")
    return e


def batches(src: int, dst: int, n: int, per_batch: int) -> list:
    """`n` bytes from `src` to `dst` in `ENTRY`-byte entries, `per_batch` to a call."""
    offs = range(0, n, ENTRY)
    sizes = [min(ENTRY, n - o) for o in offs]
    return [
        (
            [src + o for o in offs[i : i + per_batch]],
            [dst + o for o in offs[i : i + per_batch]],
            sizes[i : i + per_batch],
        )
        for i in range(0, len(sizes), per_batch)
    ]


def write(e, target: str, plan: list) -> list:
    """`[start, end, ok]` of one write of `plan`'s batches, as `_rdma_write` posts them."""
    start = time.monotonic_ns()
    ok = all(e.batch_transfer_sync_write(target, *b) == 0 for b in plan)
    return [start, time.monotonic_ns(), ok]


def wait_until(t_ns: int) -> None:
    time.sleep(max(0.0, (t_ns - time.monotonic_ns()) / 1e9 - 0.002))
    while time.monotonic_ns() < t_ns:
        pass


def target(args) -> None:
    import torch

    held, out = [], {}
    for r in range(GPUS):
        buf = torch.empty(BUFFER, dtype=torch.uint8, device=f"cuda:{r}")
        e = engine(args.ip, args.hca.format(r), buf)
        held.append((buf, e))
        out[r] = {"target": f"{args.ip}:{e.get_rpc_port()}", "addr": buf.data_ptr()}
        out[r]["rate"] = port_rate(args.hca.format(r))
    Path(args.out).write_text(json.dumps(out))
    threading.Event().wait()


def worker(gpu, ip, hca, targets, inbox, outbox) -> None:
    """GPU `gpu`'s writes: each command's rails, a thread each, at its starts."""
    import torch

    from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import (
        MooncakeConnector,
    )

    per_batch = MooncakeConnector._MAX_RDMA_ENTRIES_PER_BATCH
    buf = torch.empty(BUFFER, dtype=torch.uint8, device=f"cuda:{gpu}")
    engines, lines = {}, []
    while (cmd := inbox.get()) is not None:
        lines.clear()
        for r in dict.fromkeys(cmd["rails"]):
            if r not in engines:
                t0 = time.monotonic_ns()
                engines[r] = engine(ip, hca.format(r), buf)
                small = batches(buf.data_ptr(), targets[r]["addr"], ENTRY, per_batch)
                cold = write(engines[r], targets[r]["target"], small)
                warm = write(engines[r], targets[r]["target"], small)
                init = [t0, cold[0]]
                lines.append(
                    {
                        "exp": "startup",
                        "during": cmd["exp"],
                        "gpu": gpu,
                        "rail": r,
                        "init": init,
                        "first": cold,
                        "next": warm,
                    }
                )
        outbox.put(None)
        starts = inbox.get()

        def run(r, starts):
            plan = batches(buf.data_ptr(), targets[r]["addr"], cmd["bytes"], per_batch)
            for i, at in enumerate(starts):
                wait_until(at)
                t = write(engines[r], targets[r]["target"], plan)
                lines.append(
                    {
                        "exp": cmd["exp"],
                        "gpu": gpu,
                        "rail": r,
                        "round": i,
                        "writers": cmd["writers"],
                        "bytes": cmd["bytes"],
                        "t": t,
                    }
                )

        threads = [threading.Thread(target=run, args=(r, starts)) for r in cmd["rails"]]
        [t.start() for t in threads]
        [t.join() for t in threads]
        outbox.put(list(lines))


def experiments(rail: int):
    """`(experiment, [(gpu, rail)], bytes, rounds)`, in the order they run."""
    for g in range(GPUS):
        for r in (g, (g + 1) % GPUS):
            for n in SIZES:
                yield "sweep", [(g, r)], n, REPS
    for k in (1, 2, 4, 8, 16):
        yield "rail", [(i % GPUS, rail) for i in range(k)], ROUND_BYTES, REPS
    for m in (1, 2, 4, 8):
        yield "spread", [(0, r) for r in range(m)], ROUND_BYTES, REPS


def bench(args) -> None:
    import multiprocessing as mp
    import os

    targets = {int(k): v for k, v in json.loads(Path(args.targets).read_text()).items()}
    ctx = mp.get_context("spawn")
    outbox, inboxes = ctx.Queue(), [ctx.Queue() for _ in range(GPUS)]
    for g in range(GPUS):
        work = (g, args.ip, args.hca, targets, inboxes[g], outbox)
        ctx.Process(target=worker, args=work, daemon=True).start()
    env = {k: os.environ.get(k) for k in ("MC_IB_TC", "MOONCAKE_DISABLE_HIP_DMABUF")}
    rates = {r: port_rate(args.hca.format(r)) for r in range(GPUS)}
    head = {"exp": "env", "env": env, "rates": rates, "targets": targets}
    with open(args.out, "w") as out:
        out.write(json.dumps(head) + "\n")
        for exp, writers, n, rounds in experiments(args.rail):
            by_gpu = {}
            for g, r in writers:
                by_gpu.setdefault(g, []).append(r)
            for g, rails in by_gpu.items():
                inboxes[g].put(
                    {"exp": exp, "rails": rails, "bytes": n, "writers": len(writers)}
                )
            [outbox.get(timeout=600) for _ in by_gpu]  # every engine started
            gap = 0.01 + len(writers) * n / FLOOR
            t0 = time.monotonic_ns() + 200_000_000
            starts = [t0 + round(i * gap * 1e9) for i in range(rounds)]
            [inboxes[g].put(starts) for g in by_gpu]
            for _ in by_gpu:
                out.writelines(json.dumps(x) + "\n" for x in outbox.get(timeout=600))
            out.flush()
    [q.put(None) for q in inboxes]


def link(rows) -> tuple:
    """`(latency, bandwidth)` of `t = latency + n / bandwidth` over `(n, t)`
    rows, least squares on relative error: `latency / t + (n / t) / bandwidth`
    against 1."""
    a, b = [1 / t for _, t in rows], [n / t for n, t in rows]
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    saa, sab, sbb, sa, sb = dot(a, a), dot(a, b), dot(b, b), sum(a), sum(b)
    det = saa * sbb - sab * sab
    return (sa * sbb - sb * sab) / det, det / (saa * sb - sab * sa)


def band(values) -> tuple:
    q = statistics.quantiles(values, n=10, method="inclusive")
    return q[0], q[-1]


def fifo(jobs, latency, bandwidth) -> list:
    from atom.compass.kv.transfer import Link, TransferModel

    one = Link(TransferModel(latency, bandwidth, bytes_per_block=1))
    return [one.start(a, n) for a, n in jobs]


def shared(jobs, latency, bandwidth) -> list:
    """Each job's done time when the jobs moving bytes split the bandwidth
    evenly; a job completes `latency` after its last byte."""
    done, left, now, i = [0.0] * len(jobs), {}, -math.inf, 0
    while i < len(jobs) or left:
        arrive = jobs[i][0] if i < len(jobs) else math.inf
        rate = bandwidth / max(1, len(left))
        first = min(left, key=left.get, default=None)
        finish = now + left[first] / rate if left else math.inf
        t = min(arrive, finish)
        for j in left:
            left[j] -= (t - now) * rate
        now = t
        if finish <= arrive:
            done[first] = now + latency
            del left[first]
        else:
            left[i], i = jobs[i][1], i + 1
    return done


UNITS = {
    "pair": lambda c: c[:2],
    "prefill rank": lambda c: c[0],
    "rail": lambda c: c[1],
}


def calls(record) -> list:
    """`(rank, hca, start s, bytes, seconds)` of each `_rdma_write` call a
    producer rank of `record` stamped, in start order."""
    found = []
    for path in sorted(Path(record).rglob("*.writes.jsonl")):
        for line in path.read_text().splitlines():
            w = json.loads(line)
            for s, e, n in w.get("rdma", []):
                found.append((str(path), w.get("hca"), s / 1e9, n, (e - s) / 1e9))
    return sorted(found, key=lambda c: c[2])


def replay(found, unit, discipline, latency, bandwidth) -> list:
    """Each call's predicted over recorded seconds, less 1."""
    queues = {}
    for c in found:
        queues.setdefault(UNITS[unit](c), []).append(c)
    errors = []
    for q in queues.values():
        done = discipline([(c[2], c[3]) for c in q], latency, bandwidth)
        errors += [(d - c[2]) / c[4] - 1 for d, c in zip(done, q) if c[4] > 0]
    return errors


def fit(args) -> None:
    lines = [json.loads(x) for x in Path(args.bench).read_text().splitlines()]
    head, lines = lines[0], lines[1:]
    rates = list(head["rates"].values()) + [t["rate"] for t in head["targets"].values()]
    print(
        f"port rate, B/s: {min(rates):.4g} to {max(rates):.4g} over {len(rates)} ports"
    )
    sweep = [x for x in lines if x["exp"] == "sweep" and x["t"][2]]
    cases = {}
    for case in ("own", "next"):
        rows = [x for x in sweep if (x["gpu"] == x["rail"]) == (case == "own")]
        pts = lambda xs: [(x["bytes"], (x["t"][1] - x["t"][0]) / 1e9) for x in xs]
        fits = [link(pts(x for x in rows if x["round"] == i)) for i in range(REPS)]
        cases[case] = link(pts(rows))
        lat, bw = band([f[0] for f in fits]), band([f[1] for f in fits])
        print(
            f"{case} rail: latency {cases[case][0]:.4g} s (p10 {lat[0]:.4g}, p90 "
            f"{lat[1]:.4g}), bandwidth {cases[case][1]:.4g} B/s (p10 {bw[0]:.4g}, "
            f"p90 {bw[1]:.4g}), derate {cases[case][1] / max(rates):.4g}"
        )
    ms = lambda t: (t[1] - t[0]) / 1e6
    starts = {}
    for x in (x for x in lines if x["exp"] == "startup"):
        starts.setdefault(x["during"], []).append(x)
    for during, xs in starts.items():
        p50 = [
            statistics.median(ms(x[k]) for x in xs) for k in ("init", "first", "next")
        ]
        print(
            f"startup during {during}, {len(xs)} engines, p50: start {p50[0]:.1f}"
            f" ms, first write {p50[1]:.2f} ms, next {p50[2]:.2f} ms"
        )
    for exp in ("rail", "spread"):
        rounds = {}
        for x in (x for x in lines if x["exp"] == exp):
            rounds.setdefault((x["writers"], x["round"]), []).append(x)
        for k in sorted({w for w, _ in rounds}):
            spread, rate = [], []
            for (w, _), xs in rounds.items():
                if w == k:
                    secs = [ms(x["t"]) / 1e3 for x in xs]
                    span = max(x["t"][1] for x in xs) - min(x["t"][0] for x in xs)
                    spread.append(max(secs) / min(secs))
                    rate.append(sum(x["bytes"] for x in xs) / span * 1e9)
            print(
                f"{exp} {k} writers: finishing spread p50 {statistics.median(spread):.3g}"
                f", aggregate {statistics.median(rate):.4g} B/s"
            )
    if args.record:
        found = calls(args.record)
        print(f"record: {len(found)} calls, {sum(c[3] for c in found)} bytes")
        for unit in UNITS:
            for name, discipline in (("FIFO", fifo), ("shared", shared)):
                errors = replay(found, unit, discipline, *cases["own"])
                p50, p90 = (
                    statistics.quantiles([abs(e) for e in errors], n=10)[i]
                    for i in (4, 8)
                )
                print(
                    f"per {unit}, {name}: |error| p50 {p50:.3g} p90 {p90:.3g}, "
                    f"signed p50 {statistics.median(errors):+.3g}"
                )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="verb", required=True)
    for verb in ("target", "bench"):
        p = sub.add_parser(verb)
        p.add_argument("--ip", required=True, help="this node's address")
        p.add_argument("--hca", default="ionic_{}", help="rail N's HCA name")
        p.add_argument("--out", required=True)
    sub.choices["bench"].add_argument("--targets", required=True)
    sub.choices["bench"].add_argument("--rail", type=int, default=0)
    p = sub.add_parser("fit")
    p.add_argument("bench")
    p.add_argument("record", nargs="?")
    args = parser.parse_args(argv)
    {"target": target, "bench": bench, "fit": fit}[args.verb](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
