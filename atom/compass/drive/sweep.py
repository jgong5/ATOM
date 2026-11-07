# SPDX-License-Identifier: MIT
"""`client.sh --sweep`: a cell's JIT sweep sent to a running server, and its
coverage kept in the JIT cache library of the server's stack.

    python -m atom.compass.drive.sweep send OUT TRACES A B
    python -m atom.compass.drive.sweep cover OUT

`send` plans the cell's points (`sweep_plan.plan`) from the engine files in
``OUT/steps`` and TRACES' traces A to B, each line one trace, counted from 0,
with the sampling arguments client.sh's aiperf adds to every request, and
the server's two-batch overlap variables (`TBO`) to plan its splits. A record
with no engine file for some DP rank is first sent a one-token request on each
rank. It sends every step holding a point the library has not seen covered,
writes the points to ``OUT/sweep.json`` and prints the counts; a cell with none
missing sends nothing more. Under pipeline parallelism or RapidServe, where
ATOM builds no PrefillDelayer, it plans A alone. First it refuses a server whose
environment, read from ``/proc``, lacks a variable `pacing` sets for the ATOM
arguments on its command line, as `compass record --sweep` starts it.

A point's requests are completions of fresh token ids, each asking for its
two output tokens and naming its DP rank, so placement needs no session
affinity. First, and read to their end, the prompts whose KV a row reuses:
the shared prefix the row's prompt starts with. Then every rank's parked
decodes at once, each decoding before the rows follow, so the PrefillDelayer
holds the rows until all are admitted and runs them as one batch; then the
rows, every rank's first, each group admitted before the next is sent. A row
on its own prompt's chunks is one prompt, its earlier chunks the steps before,
behind a first row of the budget's rest where its context is no whole number
of budgets. A request that fails is printed, and its point stays missing.

`cover`, once the server has stopped, reads OUT's step record and adds the
points it covers (`sweep_plan.covered`) to the library's coverage of the cell,
``<library>/sweep/<cell>.json``, the cell named by a digest of its planned
points. The cell is swept when every point is covered and B was planned. It
writes the points still missing to ``OUT/sweep.json`` and prints the counts.

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
from operator import attrgetter
from pathlib import Path
from urllib.parse import urlsplit

from atom.compass.carriers import tracestate_with
from atom.compass.drive import jit_cache, runfile, sweep_plan
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
#: The PrefillDelayer's settings a sweep server runs with: it coalesces a
#: step's rows behind the parked decodes, released only when pending prefill
#: tokens stop growing for 16 decode steps.
PACING = {
    "ATOM_ENABLE_PREFILL_DELAYER": "1",
    "ATOM_PREFILL_DELAYER_COST_GATE": "0",
    "ATOM_PREFILL_DELAYER_TARGET_FILL": "1.0",
    "ATOM_PREFILL_DELAYER_STALL_TICKS": "16",
}
#: The variables ATOM's two-batch overlap splits a prefill step by.
TBO = ("ATOM_TBO_PREFILL_MIN_TOKENS", "ATOM_TBO_PREFILL_TOKEN_SPLIT")


def pacing(atom_args) -> dict[str, str]:
    """The environment a sweep server of ATOM's arguments `atom_args` runs with:
    `PACING`, ATOM's DP prefill balancing floor at the batched-token budget, so
    balancing cuts no row a step admits, and at DP and PP width 1 a decode
    interval of 1, without which ATOM builds no delayer there."""
    engine = runfile._engine_args(atom_args)
    floor = str(engine.max_num_batched_tokens)
    env = PACING | {"ATOM_DP_PREFILL_BALANCE_MIN_TOKENS": floor}
    if runfile._dp_width(atom_args) == 1 and engine.pipeline_parallel_size == 1:
        env["ATOM_PREFILL_DECODE_INTERVAL"] = "1"
    return env


class _Wall:
    """A real server: a step's requests sent group by group, each admitted
    before the next is sent, a parked decode once its first token came, and
    every one read to its end; a failed request is printed, not raised."""

    def __init__(self, port: int) -> None:
        self.port = port

    def send(self, first, parked, groups) -> None:
        size = len(first) + len(parked) + sum(map(len, groups))
        with ThreadPoolExecutor(max(1, size)) as pool:

            def group(bodies, token=False):
                ready = [threading.Event() for _ in bodies]
                ends = [pool.submit(self._read, *r, token) for r in zip(bodies, ready)]
                for r in ready:
                    r.wait()
                return ends

            for end in group(first):
                end.result()
            group(parked, token=True)
            for bodies in groups:
                group(bodies)

    def _read(self, body: dict, ready: threading.Event, token: bool) -> None:
        """Read the response to its end, setting `ready` once the server
        admitted the request, with `token` once its first token came."""
        try:
            lines = post(self.port, body, {})
            if token:
                next(line for line in lines if line.startswith("data:"))
            ready.set()
            if DONE not in lines:
                raise RuntimeError(f"it ended before {DONE}")
        except Exception as e:  # noqa: BLE001 - its point stays missing
            print(
                f"sweep: a request failed, its point stays missing: {e}",
                file=sys.stderr,
            )
        ready.set()

    def close(self) -> None:
        pass


class _Clocked(Traffic):
    """A simulated server's traffic LP: each group sent `EPS` after the one
    before on the run's clock, a parked decode's before its first token came,
    and every request read to its end."""

    def __init__(self, run: dict, port: int) -> None:
        super().__init__(run, port, [], [])
        self.rt.start_run()
        self.sent = 0

    def send(self, first, parked, groups) -> None:
        self._group(first)
        self._drain()
        tokens = self._group(parked)
        while not all(t.is_set() for t in tokens):
            self._next()
        for bodies in groups:
            self._group(bodies)
        self._drain()

    def _group(self, bodies) -> list[threading.Event]:
        rt, tokens = self.rt, []
        at = rt.now + EPS
        while bodies and rt.now < at:
            rt.next_event(at)
            self._take()
        for body in bodies:
            stamp = tracestate_with(None, *rt.stamp_send(self.http))
            tokens.append(threading.Event())
            thread = threading.Thread(
                target=self._read, args=(body, stamp, tokens[-1]), daemon=True
            )
            thread.start()
        self.sent += len(bodies)
        return tokens

    def _next(self) -> None:
        if self.rt.next_event(math.inf) == math.inf:
            raise RuntimeError(f"{self.sent - self.done} sweep requests never ended")
        self._take()

    def _drain(self) -> None:
        while self.done < self.sent:
            self._next()

    def close(self) -> None:
        """Take the run's events to its +inf grant, which ends the client run,
        then leave."""
        while self.rt.next_event(math.inf) != math.inf:
            self._take()
        self.rt.close()


def send(out, workload: dict, sender=None) -> list[Point]:
    """Send the steps holding points of the cell `workload` and OUT's engine
    make that the library has not seen covered, to the server
    ``OUT/server.json`` names, or through `sender`; return those points.
    `workload` is as `sweep_plan.plan` takes it."""
    out = Path(out)
    server = json.loads((out / "server.json").read_text())
    argv, environ = _deployment(server["pid"])
    wanted = pacing(argv)
    if wrong := [f"{k}={v}" for k, v in wanted.items() if environ.get(k) != v]:
        raise ValueError(
            f"the server runs without {', '.join(wrong)}, which `compass record "
            "--sweep` sets and a sweep's steps need"
        )
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
        engine = runfile._engine_args(argv)
        a_only = engine.pipeline_parallel_size > 1 or engine.enable_rapidserve
        # The plan reads its splits off this process's TBO variables.
        for k in TBO:
            if k in environ:
                os.environ[k] = environ[k]
            else:
                os.environ.pop(k, None)
        points = plan(out / "steps", workload, a_only)
        cell = _cell(points)
        seen = _seen(library, cell)
        missing = [p for p in points if _key(p) not in seen]
        state = {
            "library": str(library),
            "cell": cell,
            "points": [_entry(p) for p in points],
            "a_only": a_only,
        }
        (out / STATE).write_text(json.dumps(state))
        print(
            f"sweep: cell {cell}: {len(points)} points planned, {len(points) - len(missing)} covered before, {len(missing)} to send"
        )
        stated = engines(out / "steps")[0]
        todo = {p.step for p in missing}
        for k, step in itertools.groupby(points, attrgetter("step")):
            if k in todo:
                sender.send(*_requests(list(step), stated, fresh))
    finally:
        sender.close()
    return missing


def cover(out) -> list[Point]:
    """Add the points OUT's step record covers to the library's coverage of
    its cell, and return those still missing."""
    out = Path(out)
    state = json.loads((out / STATE).read_text())
    points = [
        Point(r, tuple(map(tuple, rows)), tuple(how), held, split, step, body)
        for r, rows, how, held, split, step, body in state["points"]
    ]
    hit, _ = covered(points, read(out / "steps"))
    path = Path(state["library"], COVERAGE, f"{state['cell']}.json")
    # ponytail: two covers of one cell at once can lose one's points; the next round sends them again.
    seen = _seen(state["library"], state["cell"]) | {_key(p) for p in hit}
    missing = [p for p in points if _key(p) not in seen]
    swept = not missing and not state["a_only"]
    _write(path, {"points": len(points), "covered": sorted(seen), "swept": swept})
    _write(out / STATE, state | {"missing": [_entry(p) for p in missing]})
    print(
        f"sweep: cell {state['cell']}: {len(hit)} of {len(points)} points covered by this record, "
        f"{len(seen)} in all, {len(missing)} missing; "
        + ("swept" if swept else "not swept")
    )
    return missing


def _deployment(pid: int) -> tuple[list[str], dict]:
    """The command line and environment of process `pid`, the server."""
    proc = Path("/proc", str(pid))
    argv = (proc / "cmdline").read_bytes().decode().split("\0")[:-1]
    pairs = (proc / "environ").read_bytes().decode().split("\0")
    return argv, dict(p.split("=", 1) for p in pairs if "=" in p)


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
        sender.send([], [], [[primer(r) for r in range(size)]])


def _requests(step: list[Point], engine: dict, fresh) -> tuple[list, list, list]:
    """A step's requests: the prefixes its rows reuse, its parked decodes, and
    its rows in groups, every rank's i-th in the i-th."""
    first, parked, lanes = [], [], []
    for point in step:
        body = {**point.body, "stream": True, "data_parallel_rank": point.rank}
        parked += [
            {
                **body,
                "prompt_token_ids": fresh(1),
                "max_tokens": sweep_plan.PARKED_TOKENS,
            }
            for _ in range(point.held)
        ]
        lane = []
        for (q, c), how in zip(point.rows, point.cached):
            ids = fresh(c)
            if how == "prefix":
                first.append({**body, "prompt_token_ids": ids[: c - q]})
            elif how == "chunk" and (pad := -(c - q) % sweep_plan.budget(engine)):
                lane.append({**body, "prompt_token_ids": fresh(pad)})
            lane.append({**body, "prompt_token_ids": ids})
        lanes.append(lane)
    rows = [[r for r in group if r] for group in itertools.zip_longest(*lanes)]
    return first, parked, rows


def _fresh(rng: random.Random):
    return lambda n: rng.choices(TOKENS, k=n)


def _key(point: Point) -> str:
    return json.dumps(sweep_plan.key(point))


def _entry(point: Point) -> list:
    p = point
    return [p.rank, p.rows, p.cached, p.held, p.split, p.step, p.body]


def _cell(points: list[Point]) -> str:
    planned = json.dumps([_entry(p) for p in points], sort_keys=True)
    return hashlib.sha256(planned.encode()).hexdigest()[:16]


def _seen(library, cell: str) -> set[str]:
    """The cell's points the library has seen covered."""
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
