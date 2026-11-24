# SPDX-License-Identifier: MIT
"""Another node's CLOCK_MONOTONIC against this node's, paired over the command
that reaches it: `python -m atom.compass.drive.clock_pair FILE [--every S]
[--timeout S] -- URL N WORD... [URL N WORD...]`

Each URL is an instance's, as its `server.json` names it, and its N words the
command prefix that started it on its node (`deploy.sh --on`), such as `ssh
<host>`; its node is the host the URL names (`node`), and one on this node, a
loopback or unspecified address, is none to pair. Each other node, once
however many instances it holds, is paired as an NTP client pairs with its
server, though over a prefix's own channel, so no port opens: the first
instance's prefix that reaches it, or, where the environment sets one, the
prefix `override` names for it, its words split on whitespace, such as `ssh
-o BatchMode=yes <host>` in place of a relay that carries no stdin. The
prefix runs this file there (`serve`, by its path, which that node sees as
this one does), and each of `ROUNDS` exchanges, after `WARM` that are not
kept, reads this node's clock, asks that one for its own, and reads this
one's again.
That node's clock less the middle of the two is its offset, off by at most
half the round trip, as the exchange's two ways may differ. The offset is the
median over the tenth of the exchanges, at least 5, that came back fastest,
and its error the half round trip of the slowest of those: the method of
c2verify's clock_pair.py, its CLOCK_MONOTONIC alone (#3720). Each node's
pair is one line appended to FILE, `{"node", "mono_ns", "offset_ns",
"error_ns"}`: the address, this node's clock at the middle of the fastest
exchange, that node's clock less this one's then, and the error, all in
nanoseconds. A node whose pair fails, its prefix closing, answering in no
`--timeout` (default 60) seconds or with no clock, writes no line; why goes
to stderr, and the command exits 1 once each node is tried. With `--every`,
the command pairs each node at once and every S seconds after, until a
SIGTERM, at which it exits 0.

`Offsets` reads a node's lines back for a run's split
(atom/compass/report/split.py): the offset at an instant of that node's clock,
piecewise linear between its pairs and the nearest pair's outside them.

This file imports the standard library alone, so any python3 on that node
runs `serve`.
"""

from __future__ import annotations

import argparse
import bisect
import ipaddress
import json
import os
import re
import select
import shlex
import signal
import statistics
import subprocess
import sys
import time
from urllib.parse import urlsplit

#: Where a deployment keeps its pairs (`deploy.sh` record).
FILE = "clock_pairs.jsonl"
#: The exchanges a pair keeps, and those before them it does not.
ROUNDS, WARM = 400, 20
#: The fastest share of the exchanges the offset is taken from, and their least number.
FASTEST, AT_LEAST = 10, 5
CLOCK = time.CLOCK_MONOTONIC


def override(node: str) -> str:
    """The variable naming the prefix `node` is paired through in place of its
    instances': ``CLOCK_PAIR_PREFIX_`` and the address, each character other
    than a letter, a digit or ``_`` an ``_``, so 10.19.0.127's is
    ``CLOCK_PAIR_PREFIX_10_19_0_127`` and fd00::7's ``CLOCK_PAIR_PREFIX_fd00__7``."""
    return "CLOCK_PAIR_PREFIX_" + re.sub(r"[^A-Za-z0-9_]", "_", node)


def node(url: str) -> str | None:
    """The node an instance serving at `url` ran on: the host the URL names,
    None for this one, a loopback or unspecified address."""
    host = urlsplit(url).hostname
    try:
        address = ipaddress.ip_address(host)
    except ValueError:  # a name
        return None if host == "localhost" else host
    return None if address.is_loopback or address.is_unspecified else host


def serve() -> None:
    """That node's side: its clock, in nanoseconds, a line for each line read."""
    read = sys.stdin.buffer.readline
    while read():
        os.write(1, b"%d\n" % time.clock_gettime_ns(CLOCK))


def pair(prefix: list, rounds: int = ROUNDS, timeout_s: float = 60.0) -> dict:
    """The node `prefix` reaches paired with this one, as the module says:
    ``mono_ns``, ``offset_ns`` and ``error_ns``. Raises OSError, ValueError or
    TimeoutError when it cannot be."""
    here = os.path.abspath(__file__)
    if shlex.quote(here) != here:
        raise ValueError(f"a shell there would parse {here} again")
    deadline = time.monotonic() + timeout_s
    child = subprocess.Popen(
        [*prefix, "python3", "-I", "-u", here, "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    into, out, held, kept = child.stdin.fileno(), child.stdout.fileno(), b"", []
    closed = f"`{shlex.join(prefix)}` closed with no answer"
    try:
        for k in range(WARM + rounds):
            sent = time.clock_gettime_ns(CLOCK)
            try:
                os.write(into, b"\n")
            except BrokenPipeError:
                raise ValueError(closed) from None
            while b"\n" not in held:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([out], [], [], left)[0]:
                    raise TimeoutError(f"no answer in {timeout_s:g} s")
                got = os.read(out, 4096)
                if not got:
                    raise ValueError(closed)
                held += got
            back = time.clock_gettime_ns(CLOCK)
            line, _, held = held.partition(b"\n")
            if not line.strip().isdigit():
                raise ValueError(f"it answered {line[:80]!r}, no clock")
            if k >= WARM:  # the round trip, twice the offset and twice the middle
                kept.append((back - sent, 2 * int(line) - sent - back, sent + back))
    finally:
        try:
            child.stdin.close()
        except BrokenPipeError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        child.stdout.close()
    kept.sort()
    fastest = kept[: max(AT_LEAST, len(kept) // FASTEST)]
    return {
        "mono_ns": fastest[0][2] // 2,
        "offset_ns": round(statistics.median(k[1] for k in fastest) / 2),
        "error_ns": (fastest[-1][0] + 1) // 2,
    }


class Offsets:
    """A node's pairs, as FILE holds them: its offset, that node's clock less
    this one's, at an instant of that node's clock, in seconds."""

    def __init__(self, rows: list):
        """`rows`, each ``(mono_ns, offset_ns, error_ns)``. Each pair is an
        instant of that node's clock, its `mono_ns` plus its `offset_ns`; of
        two at or out of that order, as only an exchange's error can put them,
        the one of smaller error is kept."""
        kept = []
        for row in sorted(rows):
            if kept and row[0] + row[1] <= kept[-1][0] + kept[-1][1]:
                if row[2] < kept[-1][2]:
                    kept[-1] = row
                continue
            kept.append(row)
        self.rows = kept
        self._there = [(m + o) * 1e-9 for m, o, _ in kept]

    def at(self, there_s: float) -> tuple[float, float]:
        """The offset at `there_s` on that node's clock and its error, in
        seconds: linear between the pairs either side, whose larger error it
        takes, and the nearest pair's outside them. Its offset is linear in
        this node's clock between two pairs, as c2verify's c2split.py takes
        it, and so in that node's too, between the same two instants."""
        k = bisect.bisect_right(self._there, there_s)
        if k in (0, len(self.rows)):
            _, offset, error = self.rows[min(k, len(self.rows) - 1)]
            return offset * 1e-9, error * 1e-9
        (_, a, ea), (_, b, eb) = self.rows[k - 1], self.rows[k]
        x0, x1 = self._there[k - 1], self._there[k]
        offset = a * 1e-9 + (there_s - x0) / (x1 - x0) * (b - a) * 1e-9
        return offset, max(ea, eb) * 1e-9


def read(path) -> dict:
    """The pairs in the file at `path`, an `Offsets` a node."""
    rows = {}
    with open(path, "rb") as lines:
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                held = json.loads(line)
                row = (held["mono_ns"], held["offset_ns"], held["error_ns"])
                name = held["node"]
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path} line {n} is no clock pair: {error}")
            rows.setdefault(name, []).append(row)
    return {name: Offsets(held) for name, held in rows.items()}


def _nodes(words: list) -> dict:
    """Each other node `words`, URL N WORD..., name, by the prefix it is
    paired through: its `override`'s, set and not blank, else the first that
    reaches it."""
    reached = {}
    while words:
        url, n = words[0], int(words[1])
        prefix, words = words[2 : 2 + n], words[2 + n :]
        name = node(url)
        if name is not None and prefix:
            reached.setdefault(
                name, os.environ.get(override(name), "").split() or prefix
            )
    return reached


def _pair_each(path, reached: dict, timeout_s: float) -> bool:
    """Append a pair of each node in `reached` to `path`; whether every one
    was paired."""
    paired = True
    for name, prefix in reached.items():
        try:
            row = {"node": name} | pair(prefix, timeout_s=timeout_s)
        except (OSError, ValueError, TimeoutError) as why:
            print(f"clock_pair: {name}: {why}", file=sys.stderr, flush=True)
            paired = False
            continue
        with open(path, "a") as out:
            out.write(json.dumps(row) + "\n")
    return paired


def main(argv: list) -> int:
    at = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.drive.clock_pair",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("file")
    parser.add_argument("--every", type=float)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv[:at])
    reached = _nodes(argv[at + 1 :])
    if args.every is None:
        return 0 if _pair_each(args.file, reached, args.timeout) else 1
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while reached:
        _pair_each(args.file, reached, args.timeout)
        time.sleep(args.every)
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        serve()
    else:
        sys.exit(main(sys.argv[1:]))
