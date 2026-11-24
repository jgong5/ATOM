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
  deployment across nodes (`run.json`'s ``across_nodes``, or, where an older
  one states none, its ``kv_link`` ``inter_node``) has a monotonic clock a
  node, each counting from its own boot; one whose KV crossed the network from
  this node (`deploy.sh --kv-over-network`) shares this one. Across nodes, each
  instance ran on the node its ``server.json`` URL names (`compass record
  --host`), this one for a loopback or unspecified address, and each other
  node's instances, prefill and decode alike, are joined to this clock by that
  node's offset. Where OUT holds that node's clock paired with this one's
  (`clock_pair.FILE`, which `deploy.sh` record writes), each of its events
  takes the offset measured at it, piecewise linear between the pairs
  (`clock_pair.Offsets`); else the node takes one offset, which the events of
  each request that cross between this node and that one bound.
  ``node_offsets_s`` gives each other node's, by its address, as ``offset_s``,
  the one taken, and ``bounds_s``: at least each request's send less its
  prefill entry there and its prefill return here less its decode entry there;
  at most its first token less its hand-over's return there and its decode
  entry here less its prefill return there; null where no request sets one. A
  node holding a decode instance takes its upper bound, as though its fastest
  first token reached the client the moment its forward handed it over; a node
  of prefill instances alone its lower, as though its fastest request entered a
  forward the moment it was sent. A node no request reached is null. A
  request's hop between two other nodes bounds neither; its order is checked as
  every request's is. A measured node's entry gives, with ``bounds_s``, then a
  check, its ``source``, ``measured``, its pairs, ``samples``, the least and
  the most of their offsets in the sign of ``offset_s``, ``measured_s``, the
  largest error they state, ``error_s``, and ``order_violations``, the
  requests that one of its events and one on another node, so joined, put out
  of order by no more than the two offsets' errors: these are split as
  joined, and a request out of order by more is refused, as on one clock.
  Beside a measured node, each other one gives its ``source``, ``bounds``. Where
  an instance's directory holds no ``server.json``, every prefill instance is
  taken to run here and each decode instance on a node of its own, as before
  nodes were read: ``decode_offset_s`` bounds what the decode side's times take
  to land on this clock, as ``bounds_s`` does a node's, and they take the upper
  bound.

A real forward's entry and return are its step line's ``t_enter_ns`` and
``t_exit_ns``; a simulated one's are its cost line's ``t_s`` and that plus
``group_s``. A simulated client's stamps are `EPOCH_NS` past the virtual clock.
A deployment's sides join through the decode side's `TRANSFERS`. Above one
instance of a side (`deploy.sh`'s ``prefill<k>/`` and ``decode<k>/``), each
transfer names the prefill instance it came from (`parity.INSTANCE_FIELDS`): a
simulated one by its engine id, a real one by the Mooncake handshake port
`run.json`'s ``instances`` gives it; where the instances name no node, each
real decode instance takes its own ``decode_offset_s``, listed by instance as
``decode_offsets_s``, the first field then spanning them all. In place of
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
from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.carriers import EPOCH_NS
from atom.compass.drive import clock_pair
from atom.compass.parity import TRANSFERS
from atom.compass.parity.client_run import RAW, response_id

FILE = "request_split.json"
RECORDS = "artifacts/profile_export.jsonl"
STEPS = "steps"
#: Where an instance names the URL it served at (`drive.serve.READY_FILE`).
READY = "server.json"
#: The clocks' resolution, a nanosecond stamp, in seconds.
RESOLUTION_S = 1e-9


def _side(run: Path, side: str) -> list[Path]:
    """A deployment's instances of `side` as `deploy.sh` lays them out: its one
    `side/`, or `side0/`, `side1/`, ... above one; none on one server."""
    if (run / side).is_dir():
        return [run / side]
    held = {p.name[len(side) :]: p for p in run.glob(f"{side}[0-9]*") if p.is_dir()}
    return [held[k] for k in sorted(held, key=int) if k.isdigit()]


def _instances(run: Path, meta: dict, simulated: bool) -> tuple[list, list]:
    """Where `run`'s servers recorded: a deployment's prefill instances, each
    its directory and the `parity.INSTANCE_FIELDS` its transfers name it by
    (none where it is the only one), and its decode instances' directories;
    else `run` itself, one server, and no decode side."""
    prefills, decodes = _side(run, "prefill"), _side(run, "decode")
    if not prefills:
        return [(run, {})], []
    if len(prefills) == 1:
        return [(prefills[0], {})], decodes
    held = {i["name"]: i for i in meta.get("instances", ())}
    named = []
    for k, path in enumerate(prefills):
        if simulated:
            from atom.compass.kv.connector import engine_id

            named.append((path, {"remote_engine_id": engine_id(f"engine-P{k}")}))
        elif "handshake_port" in held.get(path.name, {}):
            named.append(
                (path, {"remote_handshake_port": held[path.name]["handshake_port"]})
            )
        else:
            raise ValueError(f"run.json names no handshake port of {path.name}")
    return named, decodes


def _sender(transfer: dict, prefills: list) -> int:
    """The index of the prefill instance `transfer` came from."""
    if len(prefills) == 1:
        return 0
    hits = [
        k
        for k, (_, fields) in enumerate(prefills)
        if all(transfer.get(f) == v for f, v in fields.items())
    ]
    if len(hits) != 1:
        raise ValueError(
            f"transfer {transfer} names {len(hits)} of the {len(prefills)} "
            "prefill instances"
        )
    return hits[0]


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
            elif "ttft_s" not in record:  # aiperf times no first token without content
                unsplit[record["request_id"]] = "aiperf records no first token for it"
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


def _node(path: Path) -> str | None:
    """The node the instance at `path` ran on: the host its `READY` URL names,
    None for the client's (`clock_pair.node`)."""
    return clock_pair.node(read_json(path / READY)["url"])


def _join(
    events: dict,
    own: dict,
    decoded_by: dict,
    prefills: list,
    decodes: list,
    measured: dict,
) -> tuple[dict, dict]:
    """Move each request's `events` on a node other than its client's onto the
    client's clock, by the offset `measured` holds for that node at each event
    or else by that node's one offset, as the module says; the fields that say
    how, and each event a measured offset moved, by request, its node and that
    offset's error. Each request's prefill instance is the first of its key in
    `own`, its decode instance its `decoded_by`."""
    paths = [path for path, _ in prefills] + decodes
    located = all((path / READY).exists() for path in paths)
    if located:
        nodes = [_node(path) for path in paths]
    else:  # every prefill instance here, each decode instance a node of its own
        nodes = [None] * len(prefills) + [("decode", j) for j in range(len(decodes))]
    prefill_on, decode_on = nodes[: len(prefills)], nodes[len(prefills) :]
    # Each request's events by the node they ran on, None for the client's.
    where = {}
    lows, highs = collections.defaultdict(list), collections.defaultdict(list)
    for name, e in events.items():
        p = prefill_on[own[name][0]]
        d = decode_on[decoded_by[name]] if decodes else p
        where[name] = {"prefill entry": p, "prefill return": p}
        where[name] |= {"decode entry": d, "hand-over": d}
        for (a, x), (b, y) in itertools.pairwise(e.items()):
            here, there = where[name].get(a), where[name].get(b)
            if here is None and there is not None:
                lows[there].append(x - y)
            elif there is None and here is not None:
                highs[here].append(y - x)
    bounds, taken = {}, {}
    for node in dict.fromkeys(n for n in nodes if n is not None):
        if node not in lows and node not in highs:
            bounds[node] = None
            continue
        low, high = max(lows[node], default=None), min(highs[node], default=None)
        bounds[node] = [low, high]
        upper = high is not None and (node in decode_on or low is None)
        taken[node] = high if upper else low
    measured = {node: measured[node] for node in bounds if located and node in measured}
    moved = collections.defaultdict(dict)
    for name, e in events.items():
        for event, node in where[name].items():
            if node is None or event not in e:
                continue
            if node in measured:
                offset, error = measured[node].at(e[event])
                e[event] -= offset
                moved[name][event] = (node, error)
            else:
                e[event] += taken[node]
    if not located:
        offsets = [bounds[node] for node in decode_on]
        held = [o for o in offsets if o]
        out = {"decode_offset_s": [min(o[0] for o in held), max(o[1] for o in held)]}
        return out | ({"decode_offsets_s": offsets} if len(decodes) > 1 else {}), moved
    joined = {node: None for node in bounds}
    joined |= {
        node: {"offset_s": taken[node], "bounds_s": bounds[node]} for node in taken
    }
    if measured:
        joined = {
            node: (
                _measured(measured[node], bounds[node])
                if node in measured
                else held and {"source": "bounds"} | held
            )
            for node, held in joined.items()
        }
    return {"node_offsets_s": joined}, moved


def _measured(offsets: clock_pair.Offsets, bounds: list | None) -> dict:
    """A node's entry in ``node_offsets_s``, its clock measured as `offsets`
    holds it and bounded by `bounds`; no request yet out of order."""
    # The client's clock less that node's, as an offset the bounds hold is.
    held = [-offset * 1e-9 for _, offset, _ in offsets.rows]
    return {
        "source": "measured",
        "samples": len(held),
        "measured_s": [min(held), max(held)],
        "error_s": max(error for *_, error in offsets.rows) * 1e-9,
        "bounds_s": bounds,
        "order_violations": 0,
    }


def _slack(moved: dict, a: str, b: str) -> float:
    """How far a request's event `b` may come before its `a` with no fault of
    its own: the errors of the measured offsets that moved them, `moved`
    holding each such event's node and error, where they ran on two nodes."""
    (here, early), (there, late) = (moved.get(k, (None, 0.0)) for k in (a, b))
    return 0.0 if here == there else early + late


def split(run) -> dict:
    """`run`'s split, as the module says."""
    run = Path(run)
    meta = read_json(run / "run.json")
    simulated = "run_file" in meta
    prefills, decodes = _instances(run, meta, simulated)
    # Each prefill instance's forwards, by (instance, DP rank, request id).
    first, named, lmcache = {}, {}, {}
    for k, (path, _) in enumerate(prefills):
        held = _record(path, simulated, every=True)
        for into, got in zip((first, named, lmcache), held):
            into |= {(k, *key): value for key, value in got.items()}
    done, unsplit = _client(run, simulated)
    # Each decode-side request's instance and forwards, by its response id.
    own, decode = {}, {}
    if not decodes:
        own = {name: key for key, (name, *_) in named.items()}
    for j, path in enumerate(decodes):
        if not (path / STEPS / TRANSFERS).exists():
            raise ValueError(f"{path / STEPS} holds no {TRANSFERS}")
        for t in read_jsonl(path / STEPS / TRANSFERS):
            own[t["response_id"]] = (
                _sender(t, prefills),
                t["remote_dp_rank"],
                t["transfer_id"],
            )
        _, decoded, _ = _record(path, simulated, every=False)
        decode |= {name: (j, forwards) for name, *forwards in decoded.values()}
    events, decoded_by = {}, {}
    for name, (record, (send, token, end)) in done.items():
        key = own.get(name)
        if key not in named or (decodes and name not in decode):
            unsplit[record["request_id"]] = "no step names it on every side"
            continue
        e = {"send": send, "prefill entry": first[key]}
        e["prefill return"], forwards = named[key][2], named[key][1:]
        if decodes:
            decoded_by[name], forwards = decode[name]
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
    moved = {}
    if not simulated and meta.get("across_nodes", meta.get("kv_link") == "inter_node"):
        pairs = run / clock_pair.FILE
        measured = clock_pair.read(pairs) if pairs.exists() else {}
        joined, moved = _join(events, own, decoded_by, prefills, decodes, measured)
        clock |= joined
    # Each measured node's requests out of order within its offsets' errors.
    violated = collections.defaultdict(set)
    requests = []
    for name, e in events.items():
        for (a, x), (b, y) in itertools.pairwise(e.items()):
            if y >= x - RESOLUTION_S:
                continue
            if y < x - RESOLUTION_S - _slack(moved.get(name, {}), a, b):
                raise ValueError(
                    f"request {name}'s {b} is {x - y:.9f} s before its {a}, "
                    "so its events are not on one clock"
                )
            for node, _ in (moved[name][k] for k in (a, b) if k in moved[name]):
                violated[node].add(name)
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
    for node, names in violated.items():
        clock["node_offsets_s"][node]["order_violations"] = len(names)
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
