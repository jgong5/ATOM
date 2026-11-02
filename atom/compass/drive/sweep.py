# SPDX-License-Identifier: MIT
"""`client.sh --sweep`: a cell's JIT sweep sent to a running server, and its
coverage kept in the JIT cache library of the server's stack.

    python -m atom.compass.drive.sweep send OUT TRACES A B
    python -m atom.compass.drive.sweep cover OUT

`send` plans the cell's points (`sweep_plan.plan`) from the engine files in
``OUT/steps`` and TRACES' traces A to B, each line one trace, counted from 0,
with the sampling arguments client.sh's aiperf adds to every request. A record
with no engine file for some DP rank is first sent a one-token request on each
rank. It sends every point the library has not seen covered, writes them to
``OUT/sweep.json`` and prints the counts; a cell with none missing sends
nothing more.

A point's requests are completions of fresh token ids, each asking for its
two output tokens and naming its DP rank, so placement needs no session
affinity. First, and read to their end, the prompts whose KV a row reuses:
the shared prefix, or a first turn, which the row's prompt starts with. Then,
in this order, each group admitted before the next is sent: a blocker filling
the engine's batched-token budget; the held requests, all at once; a second
blocker filling the budget beside them; and each row. A budget-filling blocker
has a step to itself, so the held requests, admitted while it runs, are
prefilled in the step after it, and the rows queue and run as one batch in the
next. A row on its own prompt's chunks is one prompt, its earlier chunks the
steps before. The DP ranks are sent in parallel, one point at a time each.

`cover`, once the server has stopped, reads OUT's step record and adds the
points it covers (`sweep_plan.covered`) to the library's coverage of the cell,
``<library>/sweep/<cell>.json``, the cell named by a digest of its planned
points. The cell is swept when every point is covered. It writes the points
still missing to ``OUT/sweep.json`` and prints the counts.

The library is the one ``OUT/server.json`` names, a real server's given
`--store`. A simulated server's is its run file's store's for this
container's stack, and its requests are sent on the run's clock, each group
`EPS` after the one before, as its traffic LP.
"""

import hashlib
import itertools
import json
import math
import os
import random
import shlex
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from atom.compass.carriers import tracestate_with
from atom.compass.drive import jit_cache
from atom.compass.drive.identity import _stack
from atom.compass.drive.sweep_plan import OUTPUT_TOKENS, Point, covered, engines, plan
from atom.compass.parity import read
from atom.compass.parity.traffic import DONE, Traffic, post

#: A library's directory of cell coverage.
COVERAGE = "sweep"
#: Written to OUT: the library, the cell, its planned points and, after
#: `cover`, those still missing.
STATE = "sweep.json"
#: Simulated seconds between two groups' sends on a simulated server.
EPS = 1e-4
#: The token ids fresh prompts draw from, clear of every tokenizer's special
#: tokens at either end.
TOKENS = range(1000, 20000)


class _Wall:
    """A real server: each group's requests sent at once and admitted before
    the next group is sent, and every one read to its end."""

    def __init__(self, port: int) -> None:
        self.port = port

    def send(self, groups: list[list[dict]]) -> None:
        with ThreadPoolExecutor(max(1, 2 * sum(map(len, groups)))) as pool:
            reads = []
            for group in groups:
                heads = [pool.submit(post, self.port, b, {}) for b in group]
                reads += [pool.submit(list, h.result()) for h in heads]
        for r in reads:
            if DONE not in r.result():
                raise RuntimeError(f"a sweep request ended before {DONE}")

    def close(self) -> None:
        pass


class _Clocked(Traffic):
    """A simulated server's traffic LP: each group sent `EPS` after the one
    before on the run's clock, and every request read to its end."""

    def __init__(self, run: dict, port: int) -> None:
        super().__init__(run, port, [], [])
        self.rt.start_run()
        self.sent = 0

    def send(self, groups: list[list[dict]]) -> None:
        rt = self.rt
        for group in groups:
            at = rt.now + EPS
            while rt.now < at:
                rt.next_event(at)
                self._take()
            for body in group:
                stamp = tracestate_with(None, *rt.stamp_send(self.http))
                thread = threading.Thread(target=self._read, args=(body, stamp))
                thread.daemon = True
                thread.start()
            self.sent += len(group)
        while self.done < self.sent:
            if rt.next_event(math.inf) == math.inf:
                raise RuntimeError(
                    f"{self.sent - self.done} sweep requests never ended"
                )
            self._take()

    def close(self) -> None:
        self.rt.close()


def send(out, workload: dict, sender=None) -> list[Point]:
    """Send the points of the cell `workload` and OUT's engine make that the
    library has not seen covered, to the server ``OUT/server.json`` names, or
    through `sender`; return them. `workload` is as `sweep_plan.plan` takes it."""
    out = Path(out)
    server = json.loads((out / "server.json").read_text())
    run = (
        json.loads(Path(server["run_file"]).read_text())
        if "run_file" in server
        else None
    )
    library = _library(server, run)
    if sender is None:
        url = urlsplit(server["url"])
        if url.hostname != "127.0.0.1":
            raise ValueError(
                f"the sweep sends to 127.0.0.1, and the server is at {url.hostname}"
            )
        sender = _Clocked(run, url.port) if run else _Wall(url.port)
    fresh = _fresh(random.Random(0))
    body = {**workload.get("sampling", {}), "max_tokens": OUTPUT_TOKENS, "stream": True}
    try:
        _prime(
            out / "steps",
            sender,
            lambda r: {**body, "prompt_token_ids": fresh(1), "data_parallel_rank": r},
        )
        points = plan(out / "steps", workload)
        cell = _cell(points)
        seen = _seen(library, cell)
        missing = [p for p in points if _key(p) not in seen]
        state = {
            "library": str(library),
            "cell": cell,
            "points": [_entry(p) for p in points],
        }
        (out / STATE).write_text(json.dumps(state))
        print(
            f"sweep: cell {cell}: {len(points)} points planned, {len(points) - len(missing)} covered before, {len(missing)} to send"
        )
        stated = engines(out / "steps")

        def lane(rank):
            for p in missing:
                if p.rank == rank:
                    yield from _phases(p, stated[rank], fresh)

        lanes = map(lane, sorted({p.rank for p in missing}))
        for phases in itertools.zip_longest(*lanes, fillvalue=[]):
            groups = itertools.zip_longest(*phases, fillvalue=[])
            sender.send([[b for group in same for b in group] for same in groups])
    finally:
        sender.close()
    return missing


def cover(out) -> list[Point]:
    """Add the points OUT's step record covers to the library's coverage of
    its cell, and return those still missing."""
    out = Path(out)
    state = json.loads((out / STATE).read_text())
    points = [
        Point(r, tuple(map(tuple, rows)), tuple(how), held, body)
        for r, rows, how, held, body in state["points"]
    ]
    hit, _ = covered(points, read(out / "steps"))
    path = Path(state["library"], COVERAGE, f"{state['cell']}.json")
    # ponytail: two covers of one cell at once can lose one's points; the next round sends them again.
    seen = _seen(state["library"], state["cell"]) | {_key(p) for p in hit}
    missing = [p for p in points if _key(p) not in seen]
    _write(path, {"points": len(points), "covered": sorted(seen), "swept": not missing})
    _write(out / STATE, state | {"missing": [_entry(p) for p in missing]})
    print(
        f"sweep: cell {state['cell']}: {len(hit)} of {len(points)} points covered by this record, "
        f"{len(seen)} in all, {len(missing)} missing; "
        + ("swept" if not missing else "not swept")
    )
    return missing


def _library(server: dict, run: dict | None) -> Path:
    if "library" in server:
        return Path(server["library"])
    if run is not None:
        return jit_cache.library(run["store"], _stack(), [])
    raise ValueError(
        "server.json names no JIT cache library, which keeps a sweep's coverage: "
        "start the server with --store"
    )


def _prime(steps: Path, sender, primer) -> None:
    """Until every DP rank has written its engine file, send each rank one
    request: rank 0 while no file states the group's size."""
    for _ in range(2):
        files = sorted(steps.glob("dp*[0-9].engine.json"))
        size = json.loads(files[0].read_text())["data_parallel_size"] if files else 1
        if len(files) >= size:
            return
        sender.send([[primer(r) for r in range(size)]])


def _phases(point: Point, engine: dict, fresh) -> list[list[list[dict]]]:
    """The point's requests, in groups: those whose KV a row reuses, read to
    their end first, then the blockers, the held requests and the rows."""
    body = {**point.body, "stream": True, "data_parallel_rank": point.rank}

    def req(ids):
        return {**body, "prompt_token_ids": ids}

    def blockers(tokens):
        longest = engine["max_model_len"] - OUTPUT_TOKENS
        return [req(fresh(min(longest, tokens - k))) for k in range(0, tokens, longest)]

    first, rows = [], []
    for (q, c), how in zip(point.rows, point.cached):
        if how in ("prefix", "turn"):
            before = fresh(c - q)
            first.append(req(before))
            rows.append([req(before + fresh(q))])
        else:
            rows.append([req(fresh(c))])
    budget = engine["max_num_batched_tokens"]
    held = [req(fresh(1)) for _ in range(point.held)]
    main = [blockers(budget), held, blockers(budget - point.held) if held else [], *rows]
    main = [group for group in main if group]
    return [[first], main] if first else [main]


def _fresh(rng: random.Random):
    return lambda n: rng.choices(TOKENS, k=n)


def _key(point: Point) -> str:
    """What a record line matches a point by: its rank and rows."""
    return json.dumps([point.rank, point.rows])


def _entry(point: Point) -> list:
    return [point.rank, point.rows, point.cached, point.held, point.body]


def _cell(points: list[Point]) -> str:
    planned = json.dumps([_entry(p) for p in points], sort_keys=True)
    return hashlib.sha256(planned.encode()).hexdigest()[:16]


def _seen(library, cell: str) -> set[str]:
    path = Path(library, COVERAGE, f"{cell}.json")
    return set(json.loads(path.read_text())["covered"]) if path.exists() else set()


def _write(path: Path, data: dict) -> None:
    """`data` as `path`'s JSON, renamed into place whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.replace(staged, path)


def _sampling(extra: str) -> dict:
    """The fields client.sh's aiperf adds to every request body: ``ignore_eos``
    and each ``--extra-inputs K:V`` in `extra`, AIPERF_EXTRA_ARGS."""
    words = ["--extra-inputs", "ignore_eos:true", *shlex.split(extra)]
    fields = {}
    for flag, value in itertools.pairwise(words):
        if flag == "--extra-inputs":
            key, _, raw = value.partition(":")
            try:
                fields[key] = json.loads(raw)
            except ValueError:
                fields[key] = raw
    return fields


def main(argv=None) -> None:
    action, out, *rest = sys.argv[1:] if argv is None else argv
    if action == "send" and len(rest) == 3:
        traces, first, last = rest
        with open(traces) as f:
            lines = itertools.islice(f, int(first), int(last) + 1)
            workload = {"traces": [json.loads(line) for line in lines]}
        workload["sampling"] = _sampling(os.environ.get("AIPERF_EXTRA_ARGS", ""))
        send(out, workload)
    elif action == "cover" and not rest:
        cover(out)
    else:
        sys.exit("usage: sweep send OUT TRACES A B | cover OUT")


if __name__ == "__main__":
    main()
