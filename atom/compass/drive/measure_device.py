# SPDX-License-Identifier: MIT
"""`compass measure device`: one node's probe record, published as a
`probe_record` entry keyed by the cards' device, stack, PTL state and torch
and the node's host name.

    compass measure device --gpu G,G... [--pp] --store S --out D

Each card `--gpu` names runs `device_probe.probe` in a process of its own that
sees it and the next card named, the last card the first: the one-card probes
on the card, the link probe from it to the next. The same process runs the
host-link probe on the card: a `HOST_SMALL`-byte and a `LINK_BYTES` copy into
pinned host memory, each `REPS` times behind a spin. With `--pp`, `pp_probe`
fits a PP send's carriers on the first two cards. Before each process and
after the last, `device_probe.refuse_unfit` refuses cards that hold memory or
another process, or read another PTL state; each reading is kept in the
record.

Every format of `device_probe.FORMATS` is probed: a format no library runs on
these cards is recorded absent with each library's error. Cards of two
devices or two PTL states refuse, and so does a record the store already holds
for this node and platform, before any probe runs.

The record (``record.json`` in `--out`, the entry's one member) holds each
card's samples and, under ``readings``, the node's raw values: each scalar as
[median, p10, p90] over every card's samples, a compute format's at each
card's best shape and library; the bandwidth curve as the median over cards
at each size; and under `--pp` the carriers as `pp_probe.fit` gives them.
"""

import argparse
import datetime
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

from atom.compass.artifacts import (
    ArtifactStore,
    Conditions,
    GateState,
    Key,
    Kind,
    Provenance,
    Topology,
    member_name,
    platform_of,
    roots_for,
)
from atom.compass.drive import device_probe, pp_probe, serve
from atom.compass.drive.gpu import add_gpu_flag, hip_visible, smi
from atom.compass.drive.identity import _device, _ptl, _stack
from atom.compass.spec import SpecRefusal

PROG = "compass measure device"
MODULE = "atom.compass.drive.measure_device"
#: Bytes of the host-link probe's small copy, whose seconds are its latency.
HOST_SMALL = 4
#: Each scalar reading, to the samples of a card's probe it is read from.
SAMPLED = {
    "device.kernel_floor_s": "floor",
    "device.profiler_floor_s": "profiler_floor",
    "interconnect.intra_node.link_latency_s": "one_hop",
    "interconnect.intra_node.attainable_bandwidth_bytes_per_s": "link",
    "interconnect.host_link.latency_s": "host_latency",
    "interconnect.host_link.bandwidth_bytes_per_s": "host_link",
}


def main(argv: list[str]) -> None:
    args = _parser().parse_args(argv)
    gpus = args.gpu
    if len(gpus) < 2:
        raise ValueError(f"--gpu names {len(gpus)} card; the link probe needs two")
    devices = sorted({f"{d['name']} {d['arch']}" for d in _device(gpus)})
    if len(devices) > 1:
        raise ValueError(f"the cards are {' and '.join(devices)}; probe one device")
    ptl = _ptl(gpus)
    platform = platform_of(devices[0], _stack(), ptl)
    key = Key.of(Kind.PROBE_RECORD, host=socket.gethostname(), **platform)
    store = ArtifactStore(args.store)
    if store.directory_for(key).exists():
        raise ValueError(
            f"{args.store} holds this node's record for {key} as {key.token()}; "
            "a published record is never replaced"
        )
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} holds an earlier run's files; one run per directory")
    out.mkdir(parents=True, exist_ok=True)
    buses = json.loads(smi("--showbus", "--json"))
    bdfs = [buses[f"card{gpu}"]["PCI Bus"].lower() for gpu in gpus]
    checks, cards = [], {}
    for i, gpu in enumerate(gpus):
        checks.append(_idle(bdfs, ptl))
        # ponytail: one card at a time, 10-15 min a card; disjoint pairs could
        # run at once if a card's rates are shown unmoved by a busy neighbour.
        cards[str(gpu)] = _child(out, "pair", (gpu, gpus[(i + 1) % len(gpus)]))
    pp = _child(out, "pp", gpus[:2]) if args.pp else None
    checks.append(_idle(bdfs, ptl))
    record = {
        "host": key.values["host"],
        "platform": platform,
        "gpus": list(gpus),
        "checks": checks,
        "cards": cards,
        "readings": readings(cards, pp),
        "absent": {f: e for c in cards.values() for f, e in c["absent"].items()},
        "pp_samples": pp["samples"] if pp else None,
    }
    body = json.dumps(record).encode()
    (out / "record.json").write_bytes(body)
    store.publish(
        key,
        provenance=Provenance(
            " ".join([PROG, *argv]),
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            roots_for({"atom": "atom", "aiter": "aiter"}),
        ),
        topology=Topology(),
        # A probe runs no model, engine or ATOM engine code: no row reads them.
        conditions=Conditions.of(
            **platform,
            atom_src="no ATOM engine code",
            model="no model",
            host_cpu="no host CPU term",
            engine_config="no engine",
        ),
        gates=GateState.of(),
        members={member_name("record", Topology().ranks()[0], "json"): body},
    )
    print(f"published {key} as {key.token()}")


def _idle(bdfs: list[str], ptl: str) -> list[dict]:
    """The cards' sysfs readings, refused unless each is idle at `ptl`."""
    read = device_probe.cards(bdfs)
    device_probe.refuse_unfit(read, ptl)
    return read


def _child(out: Path, task: str, gpus: tuple[int, ...]) -> dict:
    """What `child(task, ...)` wrote, run in a process that sees `gpus`; under
    torchrun, two ranks, for ``pp``. It returns once KFD lists no process it
    did not list before, or after `serve.STOP_GRACE_S`: KFD lists an exited
    process until it has freed its GPU memory."""
    kfd = serve._kfd()
    path = out / f"{task}-{'-'.join(map(str, gpus))}.json"
    argv = [sys.executable, "-m", MODULE, task, str(path)]
    if task == "pp":
        run = ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2"]
        argv[1:1] = run
    env = os.environ | {"HIP_VISIBLE_DEVICES": hip_visible(gpus)}
    log = path.with_suffix(".log")
    with log.open("w") as f:
        done = subprocess.run(
            argv, env=env, stdout=f, stderr=subprocess.STDOUT, check=False
        )
    for _ in range(int(serve.STOP_GRACE_S / serve.POLL_S)):
        if not serve._kfd() - kfd:
            break
        time.sleep(serve.POLL_S)
    if done.returncode:
        raise ValueError(
            f"the {task} probe on GPUs {gpus} exited {done.returncode}; see {log}"
        )
    return json.loads(path.read_text())


def child(task: str, path: str) -> None:
    """Write to `path` this process's probe: for ``pair``, `device_probe.probe`
    of every format on visible cards 0 and 1, every format no library runs
    under ``absent``, and the host link's samples; for ``pp``, the carriers
    and their samples, from rank 0."""
    if task == "pp":
        if samples := pp_probe.probe():
            found = {"carriers": pp_probe.fit(samples), "samples": samples}
            Path(path).write_text(json.dumps(found))
        return
    import torch

    formats, absent = tuple(device_probe.FORMATS), {}
    try:
        found = device_probe.probe(torch, formats)
    except SpecRefusal as refused:  # no MXFP4 GEMM runs; the rest are timed
        absent["fp4"] = refused.what
        found = device_probe.probe(torch, tuple(f for f in formats if f != "fp4"))
    found |= host_link(torch) | {"absent": absent}
    Path(path).write_text(json.dumps(found))


def host_link(torch) -> dict:
    """Seconds of a `HOST_SMALL`-byte copy from card 0 into pinned host memory
    and bytes a second of a `LINK_BYTES` one, `REPS` samples each."""
    src = torch.empty(device_probe.LINK_BYTES, dtype=torch.uint8, device="cuda:0")
    dst = torch.empty(src.numel(), dtype=torch.uint8, pin_memory=True)
    small = partial(dst[:HOST_SMALL].copy_, src[:HOST_SMALL], non_blocking=True)
    large = partial(dst.copy_, src, non_blocking=True)
    return {
        "host_latency": device_probe._timed(torch, small),
        "host_link": [src.numel() / t for t in device_probe._timed(torch, large)],
    }


def readings(cards: dict, pp: dict | None) -> dict:
    """The node's raw values from each card's samples in `cards`: every scalar
    as [median, p10, p90] over every card's samples, the bandwidth curve as the
    median over cards at each size, and `pp`'s carriers when there are any."""
    found = {
        f"device.compute.attainable_flops.{f}": _band(
            [r for card in cards.values() for r in _best(card, f)]
        )
        for f in device_probe.FORMATS
        if not any(f in card["absent"] for card in cards.values())
    }
    for path, samples in SAMPLED.items():
        found[path] = _band([s for card in cards.values() for s in card[samples]])
    curves = [device_probe.readings(card)["bandwidth_curve"] for card in cards.values()]
    found["device.memory.bandwidth_curve"] = [
        [points[0][0], statistics.median(r for _, r in points)]
        for points in zip(*curves)
    ]
    if pp:
        found["interconnect.intra_node.pp"] = pp["carriers"]
    return found


def _best(card: dict, f: str) -> list[float]:
    """Format `f`'s rates, flops a second, at the card's best shape and
    library by median."""
    libraries = (card["gemm"], card.get("aiter", {}), card.get("triton", {}))
    return max(
        (
            [2 * math.prod(map(int, shape.split("x"))) / t for t in seconds]
            for library in libraries
            for shape, seconds in library.get(f, {}).items()
        ),
        key=statistics.median,
    )


def _band(samples: list[float]) -> list[float]:
    cut = statistics.quantiles(samples, n=10, method="inclusive")
    return [statistics.median(samples), cut[0], cut[8]]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    add_gpu_flag(p)
    p.add_argument("--pp", action="store_true", help="also fit a PP send's carriers")
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="an empty directory for the logs")
    return p


if __name__ == "__main__":
    child(*sys.argv[1:])
