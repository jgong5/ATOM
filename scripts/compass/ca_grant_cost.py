# SPDX-License-Identifier: MIT
"""Time a Clock Authority grant over the ``tcp`` carrier, with N synthetic LPs.

    python3 scripts/compass/ca_grant_cost.py serve N tcp://HOST:PORT [--lookahead L]
    python3 scripts/compass/ca_grant_cost.py lps N tcp://HOST:PORT [--rounds R]

``serve`` builds an authority over N LPs, every one joined to every other by a
channel of lookahead L, serves it at the endpoint (port 0 binds a free one),
prints the endpoint it bound, and once the run finishes prints its own CPU
seconds per request served, all its threads counted, as JSON. ``lps`` runs the
N LPs, one process each: each binds, asks ``TAR`` one second past its clock R
times, waiting for each grant, then ``NER(+inf)``, and the last of those
finishes the run. It prints the round trip of a ``TAR``, from its send to its
grant, as p10, median and p90 over every LP's rounds, as JSON.

With the default lookahead of one second, an LP's ``TAR`` is granted only once
every other LP has asked for the same time: each round is a barrier, every
waiting LP's LBTS is scanned on every request, and a round trip includes the
wait for the last LP to ask. A lookahead of two lets an LP run up to one round
ahead of the slowest.
"""

import argparse
import json
import math
import multiprocessing
import socket
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atom.compass.clock import NER, TAR, ChannelTable, ClockAuthority, LpId, LpRegistry
from atom.compass.clock_transport import connect, serve


def _name(k: int) -> LpId:
    return LpId(f"lp{k:03d}")


def authority(n: int, lookahead: float) -> ClockAuthority:
    """An authority over `n` LPs, each with a channel to every other."""
    registry = LpRegistry()
    lps = [registry.register(_name(k)) for k in range(n)]
    table = ChannelTable(registry)
    for a in lps:
        for b in lps:
            if a is not b:
                table.declare(f"{a}->{b}:x", a, b, lookahead, "inline")
    return ClockAuthority(table)


def ca(n: int, endpoint: str, lookahead: float = 1.0) -> dict:
    """Serve `n` LPs at `endpoint` until the run finishes; the CPU it took."""
    clock = authority(n, lookahead)
    server = serve(clock, endpoint)
    print(server.endpoint, flush=True)
    cpu = time.process_time()
    while clock.final_clocks is None:
        time.sleep(0.05)
    cpu = time.process_time() - cpu
    server.close()
    requests = sum(clock.grants.values()) + n  # every TAR, then each NER(+inf)
    return {
        "host": socket.gethostname(),
        "lps": n,
        "lookahead_s": lookahead,
        "requests": requests,
        "cpu_s": cpu,
        "cpu_per_request_s": cpu / requests,
    }


def _lp(k: int, endpoint: str, rounds: int) -> list[float]:
    conn = connect(_name(k), endpoint)
    now, rtts = conn.joined[0], []
    for _ in range(rounds):
        start = time.perf_counter()
        conn.send((TAR, now + 1.0, [], math.inf))
        now, _ = conn.recv()
        rtts.append(time.perf_counter() - start)
    conn.send((NER, math.inf, [], math.inf))
    conn.recv()
    conn.close()
    return rtts


def lps(n: int, endpoint: str, rounds: int = 1000) -> dict:
    """Run `n` LPs against `endpoint`, `rounds` grants each; the round trips."""
    with multiprocessing.get_context("spawn").Pool(n) as pool:
        rtts = pool.starmap(_lp, [(k, endpoint, rounds) for k in range(n)])
    deciles = statistics.quantiles([r for lp in rtts for r in lp], n=10)
    return {
        "host": socket.gethostname(),
        "lps": n,
        "rounds": rounds,
        "rtt_s": {"p10": deciles[0], "p50": deciles[4], "p90": deciles[8]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("serve", "lps"))
    parser.add_argument("n", type=int, help="how many LPs")
    parser.add_argument("endpoint", help="tcp://HOST:PORT")
    parser.add_argument("--lookahead", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=1000)
    args = parser.parse_args()
    if args.mode == "serve":
        result = ca(args.n, args.endpoint, args.lookahead)
    else:
        result = lps(args.n, args.endpoint, args.rounds)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
