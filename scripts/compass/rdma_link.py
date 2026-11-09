# SPDX-License-Identifier: MIT
"""A two-node RDMA fabric under Mooncake: a TransferEngine microbenchmark, and
the link values and write discipline fitted from it and a recorded P/D run.

    rdma_link.py target --ip IP --out TARGETS
    rdma_link.py bench --ip IP --targets TARGETS --out BENCH
    rdma_link.py fit BENCH [RECORD_DIR]

`target`, on the decode node, holds `BUFFER` bytes of each GPU N registered
with an engine on N's HCA, as the connector picks it, until it is killed.
`bench`, on the prefill node, runs a process per GPU, as each prefill rank is
one, writing from its GPU through an engine it starts on the rail it names, as
`RailEnginePool` does under matched rails, in `ENTRY`-byte entries cut into
the connector's batches. It writes a JSON line per write: `startup`, a rail
engine's start, its first write and the next; `sweep`, one writer at each of
`SIZES`, every GPU on its own rail and on the next GPU's; `rail`, 1 to 16
writers into one rail from GPUs `i % 8`, started together each round;
`spread`, GPU 0 writing on 1 to 8 rails at once.

`fit` prints each rail case's link, `latency + bytes / bandwidth` fitted on
relative error per round, as the p50, p10 and p90 of its fits; the start-ups;
each concurrency's finishing spread (the slowest write's time over the
fastest's) and aggregate rate. Given the record directory of a Mooncake P/D
run (`atom/compass/parity/writes.py`), it publishes the link the run's KV
writes attained: the port rate and the latency of every rail case, with the
derate of the writes none of whose calls was in flight beside another of its
prefill rank or rail, and the sha256 of the run's writes files it read. Then
it replays every `_rdma_write` call at its recorded
start on that link per sharing unit, one at a time on a `Link` (FIFO) or
splitting the bandwidth evenly among the calls in flight (shared), and prints
each one's error on every call and on the contended ones.
"""

import argparse
import hashlib
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
SIZES = [ENTRY << 2 * i for i in range(8)]  # 64 KiB to 1 GiB
REPS = 8
ROUND_BYTES = 256 << 20
FLOOR = 2e9  # rounds are spaced as if writes moved no faster than this
#: A startup line's `t`, `[start, first write, next write]`, read as times.
STARTUP = {
    "engine start": lambda t: (t[1][0] - t[0]) / 1e6,
    "first write": lambda t: (t[1][1] - t[1][0]) / 1e6,
    "next write": lambda t: (t[2][1] - t[2][0]) / 1e6,
}


def hca(gpu: int) -> str:
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    return mc._auto_select_ib_device(gpu)


def port_rate(gpu: int) -> float:
    """The bytes per second the port of `gpu`'s HCA negotiated."""
    rate = Path(f"/sys/class/infiniband/{hca(gpu)}/ports/1/rate").read_text()
    return float(rate.split()[0]) * 1e9 / 8


def engine(ip: str, rail: int, buf):
    """An engine on `rail`'s HCA with `buf` registered as the connector
    registers a region of `ENTRY`-byte units."""
    from mooncake.engine import TransferEngine

    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    chunk = mc.MooncakeConnector._MAX_RDMA_CHUNK_BYTES
    e = TransferEngine()
    ok = e.initialize(ip, "P2PHANDSHAKE", "rdma", hca(rail)) == 0
    for off in range(0, buf.numel(), chunk):
        size = min(chunk, buf.numel() - off)
        ok = ok and e.register_memory(buf.data_ptr() + off, size) == 0
    if not ok:
        raise RuntimeError(f"a Mooncake engine on {hca(rail)} did not start")
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
        held.append((buf, engine(args.ip, r, buf)))
        addr = f"{args.ip}:{held[-1][1].get_rpc_port()}"
        out[r] = {"target": addr, "addr": buf.data_ptr(), "rate": port_rate(r)}
    Path(args.out).write_text(json.dumps(out))
    threading.Event().wait()


def worker(gpu, ip, targets, inbox, outbox) -> None:
    """GPU `gpu`'s writes: each command's rails, a thread each, at its starts."""
    import torch

    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    per_batch = mc.MooncakeConnector._MAX_RDMA_ENTRIES_PER_BATCH
    buf = torch.empty(BUFFER, dtype=torch.uint8, device=f"cuda:{gpu}")
    plan = lambda r, n: batches(buf.data_ptr(), targets[r]["addr"], n, per_batch)
    engines, lines = {}, []
    while (cmd := inbox.get()) is not None:
        rails, line = cmd
        lines.clear()
        for r in dict.fromkeys(rails):
            if r not in engines:
                t0 = time.monotonic_ns()
                e = engines[r] = engine(ip, r, buf)
                t = [t0] + [
                    write(e, targets[r]["target"], plan(r, ENTRY)) for _ in range(2)
                ]
                lines.append(
                    {"exp": f"startup {line['exp']}", "gpu": gpu, "rail": r, "t": t}
                )
        outbox.put(None)
        starts = inbox.get()

        def run(r, starts, line):
            batched = plan(r, line["bytes"])
            for i, at in enumerate(starts):
                wait_until(at)
                t = write(engines[r], targets[r]["target"], batched)
                lines.append({**line, "gpu": gpu, "rail": r, "round": i, "t": t})

        threads = [threading.Thread(target=run, args=(r, starts, line)) for r in rails]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        outbox.put(list(lines))


def experiments(rail: int):
    """`(experiment, [(gpu, rail)], bytes)`, in the order they run, `REPS` rounds each."""
    for g in range(GPUS):
        for r in (g, (g + 1) % GPUS):
            for n in SIZES:
                yield "sweep", [(g, r)], n
    for k in (1, 2, 4, 8, 16):
        yield "rail", [(i % GPUS, rail) for i in range(k)], ROUND_BYTES
    for m in (1, 2, 4, 8):
        yield "spread", [(0, r) for r in range(m)], ROUND_BYTES


def bench(args) -> None:
    import multiprocessing as mp
    import os

    targets = {int(k): v for k, v in json.loads(Path(args.targets).read_text()).items()}
    ctx = mp.get_context("spawn")
    outbox, inboxes = ctx.Queue(), [ctx.Queue() for _ in range(GPUS)]
    for g in range(GPUS):
        work = (g, args.ip, targets, inboxes[g], outbox)
        ctx.Process(target=worker, args=work, daemon=True).start()
    env = {k: os.environ.get(k) for k in ("MC_IB_TC", "MOONCAKE_DISABLE_HIP_DMABUF")}
    rates = {r: port_rate(r) for r in range(GPUS)}
    with open(args.out, "w") as out:
        out.write(json.dumps({"env": env, "rates": rates, "targets": targets}) + "\n")
        for exp, writers, n in experiments(args.rail):
            by_gpu = {}
            for g, r in writers:
                by_gpu.setdefault(g, []).append(r)
            line = {"exp": exp, "writers": len(writers), "bytes": n}
            for g, rails in by_gpu.items():
                inboxes[g].put((rails, line))
            for _ in by_gpu:  # every engine started
                outbox.get(timeout=600)
            gap = 0.01 + len(writers) * n / FLOOR
            t0 = time.monotonic_ns() + 200_000_000
            for g in by_gpu:
                inboxes[g].put([t0 + round(i * gap * 1e9) for i in range(REPS)])
            for _ in by_gpu:
                out.writelines(json.dumps(x) + "\n" for x in outbox.get(timeout=600))
            out.flush()
    for q in inboxes:
        q.put(None)


def link(rows) -> tuple:
    """`(latency, bandwidth)` of `t = latency + n / bandwidth` over `(n, t)`
    rows, least squares on relative error: `latency / t + (n / t) / bandwidth`
    against 1."""
    a, b = [1 / t for _, t in rows], [n / t for n, t in rows]
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    saa, sab, sbb, sa, sb = dot(a, a), dot(a, b), dot(b, b), sum(a), sum(b)
    det = saa * sbb - sab * sab
    return (sa * sbb - sb * sab) / det, det / (saa * sb - sab * sa)


def spread3(values) -> list:
    """`[p50, p10, p90]` of `values`."""
    q = statistics.quantiles(values, n=10, method="inclusive")
    return [statistics.median(values), q[0], q[-1]]


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


def calls(record) -> tuple:
    """`(rank, hca, start s, bytes, seconds, write)` of each `_rdma_write` call
    a producer rank of `record` stamped, in start order, where `write` numbers
    the write that made it; and `sha256:` over the writes files, in the order
    read."""
    found, writes, digest = [], 0, hashlib.sha256()
    for path in sorted(Path(record).rglob("*.writes.jsonl")):
        data = path.read_bytes()
        digest.update(data)
        for line in data.decode().splitlines():
            w, writes = json.loads(line), writes + 1
            for s, e, n in w.get("rdma", []):
                found.append(
                    (str(path), w.get("hca"), s / 1e9, n, (e - s) / 1e9, writes)
                )
    return sorted(found, key=lambda c: c[2]), f"sha256:{digest.hexdigest()}"


def contended(found) -> set:
    """The calls in flight beside another of their prefill rank or their rail."""
    hit = set()
    for unit in ("prefill rank", "rail"):
        queues = {}
        for c in found:
            queues.setdefault(UNITS[unit](c), []).append(c)
        for q in queues.values():
            for a in q:
                if any(
                    b is not a and b[2] < a[2] + a[4] and a[2] < b[2] + b[4] for b in q
                ):
                    hit.add(a)
    return hit


def replay(found, unit, discipline, latency, bandwidth) -> dict:
    """Each call's predicted over recorded seconds, less 1."""
    queues = {}
    for c in found:
        queues.setdefault(UNITS[unit](c), []).append(c)
    errors = {}
    for q in queues.values():
        done = discipline([(c[2], c[3]) for c in q], latency, bandwidth)
        errors |= {c: (d - c[2]) / c[4] - 1 for d, c in zip(done, q) if c[4] > 0}
    return errors


def fit(args) -> None:
    lines = [json.loads(x) for x in Path(args.bench).read_text().splitlines()]
    head, lines = lines[0], lines[1:]
    rates = list(head["rates"].values()) + [t["rate"] for t in head["targets"].values()]
    print(
        f"port rate, B/s: {min(rates):.4g} to {max(rates):.4g} over {len(rates)} ports"
    )
    sweep = [x for x in lines if x["exp"] == "sweep" and x["t"][2]]
    pts = lambda xs: [(x["bytes"], (x["t"][1] - x["t"][0]) / 1e9) for x in xs]
    cases, port = {}, max(rates)
    for case, rows in (
        ("own", [x for x in sweep if x["gpu"] == x["rail"]]),
        ("next", [x for x in sweep if x["gpu"] != x["rail"]]),
        ("all", sweep),
    ):
        fits = [link(pts(x for x in rows if x["round"] == i)) for i in range(REPS)]
        lat, bw = (spread3([f[i] for f in fits]) for i in (0, 1))
        cases[case] = {
            "link_bandwidth_bytes_per_s": [port, min(rates), max(rates)],
            "link_latency_s": lat,
            "derate": [b / port for b in bw],
        }
        print(
            f"{case} rail, [p50, p10, p90] of the fits per round: latency {lat}, "
            f"bandwidth {bw}, derate {cases[case]['derate']}"
        )
    starts = {}
    for x in (x for x in lines if x["exp"].startswith("startup")):
        starts.setdefault(x["exp"], []).append(x["t"])
    for exp, ts in starts.items():
        p50 = [statistics.median(f(t) for t in ts) for f in STARTUP.values()]
        print(
            f"{exp}, {len(ts)} engines, p50 ms: "
            + ", ".join(f"{k} {v:.2f}" for k, v in zip(STARTUP, p50))
        )
    for exp in ("rail", "spread"):
        rounds = {}
        for x in (x for x in lines if x["exp"] == exp):
            rounds.setdefault((x["writers"], x["round"]), []).append(x)
        for k in sorted({w for w, _ in rounds}):
            spread, rate = [], []
            for (w, _), xs in rounds.items():
                if w == k:
                    secs = [x["t"][1] - x["t"][0] for x in xs]
                    span = max(x["t"][1] for x in xs) - min(x["t"][0] for x in xs)
                    spread.append(max(secs) / min(secs))
                    rate.append(sum(x["bytes"] for x in xs) / span * 1e9)
            print(
                f"{exp} {k} writers: finishing spread p50 {statistics.median(spread):.3g}"
                f", aggregate {statistics.median(rate):.4g} B/s"
            )
    if args.record:
        (found, digest), latency = calls(args.record), cases["all"]["link_latency_s"][0]
        busy, writes = contended(found), {}
        for c in found:
            writes.setdefault(c[5], []).append(c)
        alone = [w for w in writes.values() if busy.isdisjoint(w)]
        rate = spread3(
            [sum(c[3] for c in w) / sum(c[4] - latency for c in w) for w in alone]
        )
        published = {**cases["all"], "derate": [r / port for r in rate]}
        print(
            f"record: {len(writes)} writes, {len(found)} calls, {len(busy)} contended"
            f", writes files {digest}"
        )
        print(
            f"published, the derate of {len(alone)} uncontended writes: {json.dumps(published)}"
        )
        for unit in UNITS:
            for name, discipline in (("FIFO", fifo), ("shared", shared)):
                e = replay(found, unit, discipline, latency, rate[0])
                for among, errors in (
                    ("every", list(e.values())),
                    ("contended", [e[c] for c in busy]),
                ):
                    if len(errors) > 1:
                        q = spread3([abs(x) for x in errors])
                        print(
                            f"per {unit}, {name}, {among} call: |error| p50/p10/p90 "
                            f"{q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g}, signed p50 "
                            f"{statistics.median(errors):+.3g}"
                        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="verb", required=True)
    for verb in ("target", "bench"):
        p = sub.add_parser(verb)
        p.add_argument("--ip", required=True, help="this node's address")
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
