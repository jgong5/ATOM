# SPDX-License-Identifier: MIT
"""Each request's latency, split by where its time went:
`python -m atom.compass.report.split OUT`

OUT is a run's directory, an instance's as `compass record` or `compass run`
writes it or a deployment's as `deploy.sh` does, with its client's export
(`client.sh`), once the run has ended and written its `run.json`. The split goes
to OUT/`FILE`, one JSON document, and no other file changes:

- ``requests``: each request aiperf finished, as `atom-benchmark/v1` records a
  request (`atom.benchmarks.results.records.from_aiperf`), with
  ``response_id``, the id the server's step record names it by, and
  - ``queue_s``, from its send to the entry of the first forward holding it;
  - ``prefill_s``, from there to the return of the forward holding its final
    prefill chunk;
  - ``kv_transfer_s``, a deployment's only: from there to the entry of the
    decode side's first forward holding it;
  - ``decode_s``, from there, an instance's from its prefill's end, to its
    end at the client.
  They sum to aiperf's ``e2el_s``. ``forward_ttft_s`` runs from its send
  to the return of the forward that hands its first token over: off a
  pipeline, the runner a client's tokens come from hands an output-producing
  forward's tokens over at the return of its rank's next one
  (`tokenIDProcessor.is_deferred_out`), so the next after the decode side's
  first forward holding it, or after an instance's final prefill chunk's.
  aiperf's ``ttft_s`` adds the token's way out. ``lmcache_load_tokens`` and ``lmcache_store_tokens`` are the
  tokens the scheduler handed its worker to load from LMCache and to store
  there.
- ``unsplit``: each other request aiperf recorded, by its ``request_id``, and why.
- ``clock``: ``virtual`` for a simulated run; ``monotonic`` for a real one, the
  host clock its client shares with the server it reaches at 127.0.0.1. A real
  deployment's decode side on another node (`run.json`'s ``kv_link``
  ``inter_node``) is joined to it: ``decode_offset_s`` bounds what its times
  take to land on this clock, at least each request's prefill return less its
  decode entry and at most its first token less its hand-over's return; they
  take the upper bound.

A real forward's entry and return are its step line's ``t_enter_ns`` and
``t_exit_ns``; a simulated one's are its cost line's ``t_s`` and that plus
``group_s``. A simulated client's stamps are `EPOCH_NS` past the virtual clock.
A deployment's sides join through the decode side's `TRANSFERS`. In place of
the split, ``refused`` says why there is none: a file it reads is missing or
holds no step times, a side is a pipeline's, whose first tokens leave a stage
whose forwards it does not read, or a request's events are out of order, each of
its send, its prefill's first entry and final return, its decode side's first
entry, its hand-over's return, its first token and its end at least the one
before.
"""

import collections
import gzip
import itertools
import json
import sys
from pathlib import Path

from atom.benchmarks.results.records import from_aiperf
from atom.compass.backends.json_file import read_json
from atom.compass.carriers import EPOCH_NS
from atom.compass.parity import TRANSFERS, transfers
from atom.compass.parity.client_run import RAW, response_id

FILE = "request_split.json"
RECORDS = "artifacts/profile_export.jsonl"
STEPS = "steps"
#: The clocks' resolution, a nanosecond stamp, in seconds.
RESOLUTION_S = 1e-9


def _sides(run: Path) -> list[Path]:
    """Where `run`'s servers recorded: a deployment's `prefill/` and `decode/`,
    as `deploy.sh` lays them out, else `run` itself."""
    return [run / "prefill", run / "decode"] if (run / "prefill").is_dir() else [run]


def _client(run: Path, simulated: bool) -> tuple[dict, dict]:
    """aiperf's finished requests by response id, each its record and its send,
    first token and end on the run's clock; and the others by request id, with why."""
    path = run / "artifacts" / RAW
    if not path.exists():  # client.sh compresses it once its export is written
        path = Path(f"{path}.gz")
    sent = {}
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as lines:
        for line in lines:
            raw = json.loads(line.decode())
            name = response_id(raw.get("responses"))
            sent[raw["metadata"]["x_request_id"]] = (raw["start_perf_ns"], name)
    epoch = EPOCH_NS if simulated else 0
    done, unsplit = {}, {}
    with open(run / RECORDS, "rb") as rows:
        for i, line in enumerate(rows):
            record = from_aiperf(json.loads(line.decode()), i)
            ns, name = sent.get(record["request_id"], (None, None))
            if record["status"] != "success" or "e2el_s" not in record:
                unsplit[record["request_id"]] = f"aiperf records it {record['status']}"
            elif name is None:
                unsplit[record["request_id"]] = "it got no response id"
            else:
                t = (ns - epoch) * 1e-9
                ends = (t, t + record["ttft_s"], t + record["e2el_s"])
                done[name] = (record | {"response_id": name}, ends)
    return done, unsplit


def _costs(path: Path, simulated: bool):
    """Each step's cost line beside the simulated record at `path`, unread; for a
    real record, None for each."""
    if not simulated:
        yield from itertools.repeat(None)
        return
    with open(path.with_suffix(".cost.jsonl"), "rb") as lines:
        next(lines, None)  # its version line
        yield from lines


def _window(path: Path, step: dict, cost) -> tuple[float, float]:
    """A forward's entry and return, in seconds on the run's clock."""
    if cost is None:
        if "t_enter_ns" not in step:
            raise ValueError(f"{path} step {step['step']} holds no t_enter_ns")
        return step["t_enter_ns"] * 1e-9, step["t_exit_ns"] * 1e-9
    cost = json.loads(cost.decode())
    if cost["step"] != step["step"]:
        raise ValueError(
            f"{path}'s cost line of step {cost['step']} stands beside step {step['step']}"
        )
    if cost.get("t_s") is None:
        raise ValueError(f"{path.with_suffix('.cost.jsonl')} holds no step times (t_s)")
    return cost["t_s"], cost["t_s"] + cost["group_s"]


def _record(side: Path, simulated: bool, every: bool) -> tuple[dict, dict, dict]:
    """`side`'s step record, by (DP rank, request id): the entry of the first
    forward holding each request; the response id, the entry and return of the
    forward that names it and, once one follows, the return of its rank's next
    output-producing forward, which hands that one's tokens over; and its
    LMCache tokens loaded and stored. Unless `every`, only the forwards naming
    a request and those handing one over are read."""
    steps = side / STEPS
    if any(steps.glob("pp*.dp*.jsonl")):
        raise ValueError(
            f"{steps} is a pipeline's record, whose first tokens leave a stage "
            "the split does not read"
        )
    paths = sorted(steps.glob("dp*[0-9].jsonl"))
    if not paths:
        raise ValueError(f"{steps} holds no DP rank's step record")
    first, named, lmcache = {}, {}, collections.defaultdict(lambda: [0, 0])
    for path in paths:
        rank, waiting = int(path.stem[2:]), []
        with open(path, "rb") as lines:
            for line, cost in zip(lines, _costs(path, simulated)):
                # A record line is `json.dumps`'s: one naming nobody holds this.
                if not (every or waiting) and b'"named": []' in line:
                    continue
                step = json.loads(line.decode())
                if not (step["batch"] or step.get("kv") or waiting):
                    continue
                enter, out = _window(path, step, cost)
                if step["produces_output"]:
                    for key in waiting:
                        named[key] += (out,)
                    waiting = []
                for request, *_ in step["batch"]:
                    first.setdefault((rank, request), enter)
                for request, _key, *name in step["named"]:
                    if name:
                        named[rank, request] = (name[0], enter, out)
                        waiting.append((rank, request))
                for request, kind, held, upto in step.get("kv", ()):
                    lmcache[rank, request][kind == "save"] += upto - held
    return first, named, lmcache


def split(run) -> dict:
    """`run`'s split, as the module says."""
    run = Path(run)
    meta = read_json(run / "run.json")
    simulated = "run_file" in meta
    sides = _sides(run)
    first, named, lmcache = _record(sides[0], simulated, every=True)
    done, unsplit = _client(run, simulated)
    if len(sides) == 1:
        own = {name: key for key, (name, *_) in named.items()}
    else:
        if not (sides[1] / STEPS / TRANSFERS).exists():
            raise ValueError(f"{sides[1] / STEPS} holds no {TRANSFERS}")
        own = {name: key for key, name in transfers(sides[1] / STEPS).items()}
        _, decoded, _ = _record(sides[1], simulated, every=False)
        decode = {name: forwards for name, *forwards in decoded.values()}
    events = {}
    for name, (record, (send, token, end)) in done.items():
        key = own.get(name)
        if key not in named or (len(sides) == 2 and name not in decode):
            unsplit[record["request_id"]] = "no step names it on every side"
            continue
        e = {"send": send, "prefill entry": first[key]}
        e["prefill return"], forwards = named[key][2], named[key][1:]
        if len(sides) == 2:
            forwards = decode[name]
            e["decode entry"] = forwards[0]
        if len(forwards) < 3:
            unsplit[record["request_id"]] = (
                "no later forward hands its first token over"
            )
            continue
        events[name] = e | {"hand-over": forwards[2], "first token": token, "end": end}
    if not events:
        raise ValueError("no request aiperf finished is named by its step records")
    clock = {"clock": "virtual" if simulated else "monotonic"}
    if not simulated and meta.get("kv_link") == "inter_node":
        low = max(e["prefill return"] - e["decode entry"] for e in events.values())
        high = min(e["first token"] - e["hand-over"] for e in events.values())
        for e in events.values():
            e["decode entry"] += high
            e["hand-over"] += high
        clock["decode_offset_s"] = [low, high]
    requests = []
    for name, e in events.items():
        for (a, x), (b, y) in itertools.pairwise(e.items()):
            if y < x - RESOLUTION_S:
                raise ValueError(
                    f"request {name}'s {b} is {x - y:.9f} s before its {a}, "
                    "so its events are not on one clock"
                )
        record, key = done[name][0], own[name]
        start = e.get("decode entry", e["prefill return"])
        record |= {
            "queue_s": e["prefill entry"] - e["send"],
            "prefill_s": e["prefill return"] - e["prefill entry"],
        }
        if "decode entry" in e:
            record["kv_transfer_s"] = start - e["prefill return"]
        record |= {
            "decode_s": e["end"] - start,
            "forward_ttft_s": e["hand-over"] - e["send"],
        }
        loaded, stored = lmcache.get(key, (0, 0))
        requests.append(
            record | {"lmcache_load_tokens": loaded, "lmcache_store_tokens": stored}
        )
    return clock | {"requests": requests, "unsplit": unsplit}


def write(run) -> dict:
    """Write `run`'s split, or why it has none, to its `FILE`; that document."""
    try:
        out = split(run)
    except (ValueError, FileNotFoundError) as refused:
        out = {"refused": str(refused)}
    (Path(run) / FILE).write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m atom.compass.report.split OUT")
    out, path = write(sys.argv[1]), Path(sys.argv[1]) / FILE
    if "refused" in out:
        print(f"split: {path}: refused: {out['refused']}")
    else:
        n, m = len(out["requests"]), len(out["unsplit"])
        print(f"split: wrote {path}: {n} requests split, {m} not")
