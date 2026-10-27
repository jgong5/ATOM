# SPDX-License-Identifier: MIT
"""Probe one GPU host's attainable peaks and print them as a spec fragment.

    HIP_VISIBLE_DEVICES=a,b python3 scripts/compass/probe_device.py SPEC OUT

SPEC is the machine file the peaks belong to. Card 0 runs the one-card probes,
cards 0 and 1 the link. Every sample is device time between two events queued
behind a spin kernel, so no host time is in it.

- Compute: per format, the median of `REPS` GEMMs ``x @ w.T``, x m by k and
  w n by k, at each of `SHAPES`, BF16 through `torch.matmul`, FP8 through
  `torch._scaled_mm`, and FP4 where SPEC states ``fp4_flops``: MXFP4, two
  values a byte and a power-of-two scale per 32 along k, through each of
  `torch._scaled_mm` (hipBLASLt) and `aiter.gemm_a4w4` with w pre-shuffled,
  as aiter's own tests call it, that runs on the card; one that raises is
  named in the record with its error, and with neither the run is refused.
  Every format is also timed through a Triton GEMM (`_gemm`) at the tile of
  `TILES` that ran fastest at the first shape with an output within 1% of
  torch's; a format no tile passes is named in the record. On a card whose
  stack ships untuned libraries it is the faster: 23x hipBLASLt's BF16 rate
  on MI455 under rocm/fw-bringup-mlperf:gfx1250-atom-20261026-dsv41.
  The best rate over shapes and libraries, over
  SPEC's ``*_flops``, is the format's derate. Each GEMM follows a spin, so the
  rate is a burst's: with PTL enabled a sustained GEMM slows within a second
  or two.
- Bandwidth: per kind in `KINDS` (read-only, write-only, copy) and per size
  in `SIZES` of bytes moved, a graph of up to `GRAPH` kernels through
  consecutive slices of one zeroed `2 * POOL`-byte buffer (a copy reads one
  half and writes the other), replayed `REPS` times. Kernels that together fit
  in the last-level cache run warm, as small operators do, and larger ones
  from HBM. The kernels are Triton: `copy_` reaches about 2 TB/s on MI308X, a
  kernel about 3.6. At each size the fastest kind's median rate bounds any mix
  of reads and writes; the running maximum of it over sizes is
  ``memory.bandwidth_curve``, so noise on the plateau keeps it monotone, and
  its largest point over SPEC's bandwidth is ``memory.derate``.
- Kernel floor: `GRAPH` empty kernels captured in one graph, replayed `REPS`
  times; ``kernel_floor_s`` is the median per kernel, with its p10 and p90.
- Link: one kernel per card passes a counter `ROUNDS` times through the other
  card's memory; half a round trip is ``link_latency_s``, with its p10 and p90
  over `REPS` runs. A `LINK_BYTES` copy from card 0 to card 1 over SPEC's link
  bandwidth is ``intra_node.derate``.

The run is refused unless every card is idle (under `IDLE_BYTES` of VRAM used,
and no other process KFD charges memory of it) and reads SPEC's ``device.ptl``
from ``ptl/ptl_enable`` under its sysfs device (`NO_PTL` where it has none).
``gpu_busy_percent`` is no test of idle: a gfx1250 card with no process reads
100. OUT is the record: the cards, the stack (ROCm, aiter's commit, RCCL and
torch), the capacity torch reads on card 0, every sample and this file's text; the fragment
printed names its sha256.
"""

import datetime
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from atom.compass.artifacts.keys import NO_PTL
from atom.compass.drive.gpu import kfd_pid
from atom.compass.spec import Fragment, MachineSpec, SpecRefusal
from atom.compass.spec.rules import Rule

#: GEMM shapes (m, n, k), every format timed at each: hipBLASLt's rate moves by
#: up to 2.7x between neighbouring large shapes, and each format peaks at another.
SHAPES = tuple(itertools.product((8192, 10240, 12288, 16384), repeat=3))
SIZES = tuple(2**k for k in range(12, 33))  # 4 KiB to 4 GiB moved
POOL, COPY_BLOCK = 2**31, 4096  # bytes in each copy half, floats a program
#: Each bandwidth kind: whether its kernel reads, and whether it writes.
KINDS = {"read": (True, False), "write": (False, True), "copy": (True, True)}
GRAPH, ROUNDS, LINK_BYTES = 1000, 1000, 2**30
REPS = 20
SPIN = 10**7  # device cycles a sample waits behind, well past its host calls
IDLE_BYTES = 2**30
DRM, KFD = "/sys/class/drm", "/sys/class/kfd/kfd"
#: The MXFP4 GEMMs timed, each as a failure names it.
FP4 = ("torch._scaled_mm", "aiter.gemm_a4w4")
#: The Triton GEMM's tiles (rows, columns, k), each dividing every `SHAPES`
#: size; MXFP4's k holds twice as many values a load.
TILES = {
    fp4: tuple(itertools.product((128, 256), (128, 256), (k, 2 * k)))
    for fp4, k in ((False, 64), (True, 128))
}
GROUP = tl.constexpr(8)  # rows of tiles a program group walks, w's tiles in cache
#: Each format's derate, by the spec-peak number it is taken over.
FORMATS = {"bf16": "bf16_flops", "fp8": "fp8_flops", "fp4": "fp4_flops"}
#: The fields read with a band, and each one's sample list in the record.
BANDED = {
    "device.kernel_floor_s": "floor",
    "interconnect.intra_node.link_latency_s": "one_hop",
}


@triton.jit
def _pingpong(mine, theirs, rounds, serve: tl.constexpr):
    """Write `theirs`, on the other card, then wait to see `mine` written
    back, `rounds` times; the side that does not serve answers each round."""
    for i in range(1, rounds + 1):
        if serve:
            tl.atomic_xchg(theirs, i, sem="release", scope="sys")
        seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
        while seen < i:
            seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
        if not serve:
            tl.atomic_xchg(theirs, i, sem="release", scope="sys")


@triton.jit
def _stream(src, dst, n, BLOCK: tl.constexpr, READ: tl.constexpr, WRITE: tl.constexpr):
    """Copy `n` floats, `BLOCK` a program, or only read or only write them. A
    read alone stores the negative floats of a zeroed `src`: none."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.zeros([BLOCK], tl.float32)
    if READ:
        x = tl.load(src + offs, mask=offs < n)
    tl.store(dst + offs, x, mask=(offs < n) & ((x < 0) | WRITE))


@triton.jit
def _gemm(
    x,
    w,
    sx,
    sw,
    out,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    FP4: tl.constexpr,
):
    """`out` = x @ w.T, x M by K and w N by K, row-major, a BM by BN tile a
    program. With FP4, two e2m1 values a byte and an e8m0 scale per 32 along K
    in `sx` and `sw`. Every size is a multiple of its tile."""
    pid, cols = tl.program_id(0), N // BN
    first = pid // (GROUP * cols) * GROUP
    size = min(M // BM - first, GROUP)
    rm = (first + pid % (GROUP * cols) % size) * BM + tl.arange(0, BM)
    rn = pid % (GROUP * cols) // size * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    if FP4:
        rk, rs = tl.arange(0, BK // 2), tl.arange(0, BK // 32)
        for k in range(0, K, BK):
            a = tl.load(x + rm[:, None] * (K // 2) + (k // 2 + rk)[None, :])
            b = tl.load(w + rn[:, None] * (K // 2) + (k // 2 + rk)[None, :])
            sa = tl.load(sx + rm[:, None] * (K // 32) + (k // 32 + rs)[None, :])
            sb = tl.load(sw + rn[:, None] * (K // 32) + (k // 32 + rs)[None, :])
            acc = tl.dot_scaled(a, sa, "e2m1", b.T, sb, "e2m1", acc)
    else:
        rk = tl.arange(0, BK)
        for k in range(0, K, BK):
            a = tl.load(x + rm[:, None] * K + (k + rk)[None, :])
            b = tl.load(w + rn[None, :] * K + (k + rk)[:, None])
            acc += tl.dot(a, b)
    tl.store(out + rm[:, None] * N + rn[None, :], acc.to(out.dtype.element_ty))


def cards(bdfs: list[str], drm: str = DRM, kfd: str = KFD, me=None) -> list[dict]:
    """Each card's sysfs reading by PCI address: VRAM used, the processes
    other than `me` (this one's KFD entry by default) that KFD charges its
    memory to, and PTL."""
    me = kfd_pid() if me is None else me
    found = {}
    for card in Path(drm).glob("card[0-9]*"):
        device = card / "device"
        if "-" not in card.name and device.exists():
            found[device.resolve().name] = device
    ids = _gpu_ids(kfd)
    procs = [p for p in Path(kfd, "proc").glob("*") if p.name != me]
    out = []
    for bdf in bdfs:
        device = found[bdf]
        ptl = device / "ptl" / "ptl_enable"
        out.append(
            {
                "bdf": bdf,
                "vram_used": int((device / "mem_info_vram_used").read_text()),
                "processes": [
                    p.name for p in procs if _charged(p / f"vram_{ids[bdf]}")
                ],
                "ptl": ptl.read_text().strip() if ptl.exists() else NO_PTL,
            }
        )
    return out


def _gpu_ids(kfd: str) -> dict[str, str]:
    """Each KFD topology node's `gpu_id` by its PCI address."""
    ids = {}
    for node in Path(kfd, "topology", "nodes").glob("*"):
        lines = (node / "properties").read_text().splitlines()
        props = dict(line.split() for line in lines)
        loc = int(props["location_id"])
        bdf = f"{int(props['domain']):04x}:{loc >> 8:02x}:{loc >> 3 & 31:02x}.{loc & 7}"
        ids[bdf] = (node / "gpu_id").read_text().strip()
    return ids


def _charged(path: Path) -> int:
    """The bytes a KFD `vram_<gpu_id>` file reads; 0 once its process has exited."""
    try:
        return int(path.read_text())
    except FileNotFoundError:
        return 0


def refuse_unfit(read: list[dict], ptl: str) -> None:
    """Refuse cards that hold other work or disagree with the spec's PTL."""
    wrong = [
        f"{c['bdf']} (ptl {c['ptl']}, {c['vram_used']} B used, "
        f"processes {c['processes']})"
        for c in read
        if c["ptl"] != ptl or c["vram_used"] > IDLE_BYTES or c["processes"]
    ]
    if wrong:
        raise SpecRefusal(
            Rule.MEASURED,
            f"cards {', '.join(wrong)} are not idle cards with PTL {ptl}",
            "pick idle cards whose ptl_enable is the spec's `device.ptl`; a peak "
            "read beside a neighbour or under another PTL state is not this "
            "machine's",
        )


def _timed(torch, fn, reps: int = REPS) -> list[float]:
    """Device seconds of `reps` calls of `fn`, each behind a spin."""
    out = []
    for _ in range(reps):
        torch.cuda._sleep(SPIN)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) / 1e3)
    return out


def _fp4(torch, rows: int, k: int):
    """An MXFP4 operand of `rows` by `k` random values and its unit scales."""
    cuda = {"dtype": torch.uint8, "device": "cuda"}
    packed = torch.randint(0, 256, (rows, k // 2), **cuda)
    scales = torch.full((rows, k // 32), 127, **cuda)  # 2**0 in E8M0
    return packed.view(torch.float4_e2m1fn_x2), scales.view(torch.float8_e8m0fnu)


def _triton(torch, operands: tuple, tile: tuple) -> tuple:
    """Triton's GEMM of `operands`, x and w or for MXFP4 x, its scales, w and
    its scales, at `tile`: the call and the bf16 output it writes."""
    fp4 = len(operands) == 4
    x, sx, w, sw = operands if fp4 else (operands[0], None, operands[1], None)
    (m, k), n = x.shape, w.shape[0]
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    args = [t.view(torch.uint8) for t in (x, w, sx, sw)] if fp4 else (x, w, x, w)
    bm, bn, bk = tile
    grid = (m // bm * (n // bn),)
    k *= 2 if fp4 else 1
    launch = _gemm[grid]
    return partial(launch, *args, out, m, n, k, bm, bn, bk, fp4, num_warps=8), out


def _tile(torch, operands: tuple, reference, name: str, failed: dict):
    """The tile of `TILES` whose Triton GEMM of `operands` ran fastest with an
    output within 1% of what `reference` returns, by norm; None, named in
    `failed` under `name`, when none did or there is no `reference`."""
    best, error = None, "no torch output to check it against"
    expected = reference().float() if reference else None
    for tile in TILES[len(operands) == 4] if reference else ():
        try:
            call, out = _triton(torch, operands, tile)
            call()
            miss = ((out.float() - expected).norm() / expected.norm()).item()
            seconds = statistics.median(_timed(torch, call, reps=5))
        except Exception as e:  # noqa: BLE001 - a tile this arch cannot build
            error = f"{type(e).__name__}: {e}".splitlines()[0]
            continue
        if miss > 0.01:
            error = f"tile {tile} is off torch's output by {miss:.3g} of its norm"
        elif best is None or seconds < best[0]:
            best = (seconds, tile)
    if best is None:
        failed[name] = error
    return best and best[1]


def _a4w4(x, w):
    """aiter's MXFP4 GEMM of bf16 `x` and `w`, quantised per 1x32 with shuffled
    scales and `w` pre-shuffled, as aiter's own tests call it."""
    import aiter
    from aiter.ops.shuffle import shuffle_weight

    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    (x4, sx), (w4, sw) = quant(x, shuffle=True), quant(w, shuffle=True)
    w4 = shuffle_weight(w4, layout=(16, 16))
    return partial(aiter.gemm_a4w4, x4, w4, sx, sw, bpreshuffle=True)


def _running(fp4: dict, failed: dict) -> list:
    """Each `FP4` library's samples and call, of those in `fp4` whose call
    runs; one that raises is named in `failed` with its error and not made
    again. Refuses when none runs."""
    found = []
    for name, (samples, make) in fp4.items():
        if name in failed:
            continue
        try:
            call = make()
            call()
        except Exception as error:  # noqa: BLE001 - named, and the other timed
            failed[name] = f"{type(error).__name__}: {error}".splitlines()[0]
            continue
        found.append((samples, call))
    if not found:
        raise SpecRefusal(
            Rule.MEASURED,
            "no MXFP4 GEMM runs on these cards: "
            + "; ".join(f"{name} {error}" for name, error in failed.items()),
            "probe where one of them runs, or drop `fp4_flops` from the spec",
        )
    return found


def probe(torch, formats) -> dict:
    """Every sample, on visible cards 0 and 1, GEMMs in `formats`; FP4 also
    through aiter, under ``aiter``; each format through Triton, under
    ``triton``, at the tile under ``tiles``; each FP4 library or Triton format
    that failed, under ``failed``; and card 0's capacity as torch reads it."""
    torch.cuda.set_device(0)
    capacity = torch.cuda.mem_get_info(0)[1]  # the total an engine start reads
    fp8 = torch.float8_e4m3fn
    if torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942"):
        fp8 = torch.float8_e4m3fnuz
    one = torch.ones((), device="cuda")
    gemm, by_triton = {f: {} for f in formats}, {f: {} for f in formats}
    aiter = {"fp4": {}} if "fp4" in formats else {}
    failed: dict[str, str] = {}
    tiles: dict[str, tuple | None] = {}
    for m, n, k in SHAPES:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        scaled = partial(torch._scaled_mm, out_dtype=torch.bfloat16)
        ops = {"bf16": (x, w), "fp8": (x.to(fp8), w.to(fp8))}
        calls = {
            "bf16": partial(torch.matmul, x, w.T),
            "fp8": partial(scaled, ops["fp8"][0], ops["fp8"][1].T, one, one),
        }
        timings = [(gemm[f], calls[f]) for f in formats if f != "fp4"]
        if aiter:
            ops["fp4"] = x4, sx, w4, sw = (*_fp4(torch, m, k), *_fp4(torch, n, k))
            makes = partial(partial, scaled, x4, w4.T, sx, sw), partial(_a4w4, x, w)
            libraries = zip((gemm["fp4"], aiter["fp4"]), makes)
            ran = _running(dict(zip(FP4, libraries)), failed)
            timings += ran
            calls["fp4"] = next((c for got, c in ran if got is gemm["fp4"]), None)
        for f in formats:
            if f not in tiles:  # the first shape
                tiles[f] = _tile(torch, ops[f], calls.get(f), f"triton {f}", failed)
            if tiles[f]:
                timings.append((by_triton[f], _triton(torch, ops[f], tiles[f])[0]))
        for samples, call in timings:
            samples[f"{m}x{n}x{k}"] = _timed(torch, call)
            time.sleep(1)  # let a PTL card recover before the next burst
    pool = torch.zeros(2 * POOL // 4, device="cuda")
    streams = {kind: {} for kind in KINDS}
    for (kind, (read, write)), size in itertools.product(KINDS.items(), SIZES):
        src, dst = pool.chunk(2) if read and write else (pool, pool)
        n, k = size // 4 // (read + write), min(GRAPH, 2 * POOL // size)
        grid, flags = (triton.cdiv(n, COPY_BLOCK),), (COPY_BLOCK, read, write)
        _stream[grid](src, dst, n, *flags, num_warps=8)  # compiles uncaptured
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(k):
                _stream[grid](src[i * n :], dst[i * n :], n, *flags, num_warps=8)
        streams[kind][size] = [t / k for t in _timed(torch, graph.replay)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH):
            torch.cuda._sleep(0)
    floor = [t / GRAPH for t in _timed(torch, graph.replay)]

    near = torch.empty(LINK_BYTES, dtype=torch.uint8, device="cuda:0")
    far = torch.empty_like(near, device="cuda:1")
    far.copy_(near)  # also maps each card's memory into the other
    near.copy_(far)
    link = [LINK_BYTES / t for t in _timed(torch, partial(far.copy_, near))]
    mine = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    theirs = torch.zeros(1, dtype=torch.int32, device="cuda:1")
    serve = partial(_pingpong[(1,)], mine, theirs, ROUNDS, serve=True, num_warps=1)
    trips = []
    for _ in range(REPS + 1):  # the first compiles both kernels
        mine.zero_()
        theirs.zero_()
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        with torch.cuda.device(1):
            _pingpong[(1,)](theirs, mine, ROUNDS, serve=False, num_warps=1)
        trips += _timed(torch, serve, reps=1)
        torch.cuda.synchronize(1)
    return {
        "gemm": gemm,
        "aiter": aiter,
        "triton": by_triton,
        "tiles": tiles,
        "failed": failed,
        "capacity_bytes": capacity,
        **streams,
        "floor": floor,
        "link": link,
        "one_hop": [t / ROUNDS / 2 for t in trips[1:]],
    }


def rates(record: dict) -> dict:
    """Each bandwidth kind's median rate per size, bytes moved a second."""
    med = statistics.median
    return {k: {int(n): int(n) / med(t) for n, t in record[k].items()} for k in KINDS}


def readings(record: dict) -> dict:
    """The attainable values a record's samples give: each format's best
    median rate over shapes and libraries, the running maximum over sizes of
    the fastest kind's median rate, the link's median rate, and each banded
    field's median, p10 and p90."""
    med = statistics.median

    def band(samples):
        cut = statistics.quantiles(samples, n=10, method="inclusive")
        return [med(samples), cut[0], cut[8]]

    libraries = (record["gemm"], record.get("aiter", {}), record.get("triton", {}))
    found = {
        f: max(
            2 * math.prod(map(int, s.split("x"))) / med(t)
            for gemm in libraries
            for s, t in gemm.get(f, {}).items()
        )
        for f in record["gemm"]
    }
    kinds = rates(record)
    sizes = sorted(kinds["copy"])
    best = itertools.accumulate((max(r[n] for r in kinds.values()) for n in sizes), max)
    found["bandwidth_curve"] = [[n, r] for n, r in zip(sizes, best)]
    found["link"] = med(record["link"])
    return found | {path: band(record[key]) for path, key in BANDED.items()}


def fragment(
    name: str, found: dict, peaks: dict, author: str, date: str, subject: dict
) -> dict:
    """The fragment `readings` make on machine `name`, whose spec-peak numbers
    `peaks` holds by path, every field probed on `subject`."""
    curve = found["bandwidth_curve"]
    values = {
        **{
            f"device.compute.derate.{f}": found[f] / peaks[f"device.compute.{peak}"]
            for f, peak in FORMATS.items()
            if f in found
        },
        "device.memory.bandwidth_curve": curve,
        "device.memory.derate": curve[-1][1]
        / peaks["device.memory.bandwidth_bytes_per_s"],
        "interconnect.intra_node.derate": found["link"]
        / peaks["interconnect.intra_node.link_bandwidth_bytes_per_s"],
        **{path: found[path][0] for path in BANDED},
    }
    origins = {path: {"origin": "probed", **subject} for path in values}
    for path in BANDED:
        origins[path] |= {"p10": found[path][1], "p90": found[path][2]}
    document = {
        "schema_version": 1,
        "name": name,
        "provenance": {
            "authored_by": author,
            "date": date,
            "origin": "mixed",
            "origins": origins,
            "notes": "probe_device.py: GEMM, read, write, copy, graph and link samples",
        },
    }
    for path, value in values.items():
        *blocks, leaf = path.split(".")
        node = document
        for block in blocks:
            node = node.setdefault(block, {})
        node[leaf] = value
    return document


def main(argv: list[str] | None = None) -> int:
    import torch

    from atom.compass.artifacts.provenance import git_tree_root
    from atom.compass.drive.identity import _stack

    spec_path, out = (Path(a) for a in (argv or sys.argv[1:]))
    kfd_pid()  # this process's KFD entry, taken before torch opens a card
    spec = MachineSpec.from_mapping(json.loads(spec_path.read_text()))
    ptl = spec.values.get("device.ptl", NO_PTL)
    props = [torch.cuda.get_device_properties(i) for i in (0, 1)]
    bdfs = [
        f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        for p in props
    ]
    record = {"cards": cards(bdfs), "arch": props[0].gcnArchName, "stack": _stack()}
    refuse_unfit(record["cards"], ptl)
    stated = [
        f for f, peak in FORMATS.items() if f"device.compute.{peak}" in spec.values
    ]
    record |= probe(torch, stated)
    for name, error in record.get("failed", {}).items():
        print(f"FP4 through {name} failed: {error}", file=sys.stderr)
    record["source"] = Path(__file__).read_text()
    out.write_text(json.dumps(record) + "\n")
    root = git_tree_root("atom", str(ROOT))
    subject = {"source_root": f"{root.revision_kind}:{root.revision}"}
    subject["record"] = f"sha256:{hashlib.sha256(out.read_bytes()).hexdigest()}"
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    document = fragment(
        spec.value("name"),
        readings(record),
        spec.values,
        "probe_device.py",
        today,
        subject,
    )
    Fragment.from_mapping(document, "probe")  # refuses a derate above 1
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
