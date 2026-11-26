# SPDX-License-Identifier: MIT
"""An encoding ingress in front of a simulated deployment: its hop, priced per
request on the traffic LP's clock.

Behind an encoding ingress (`scripts/compass/encoding_ingress.py`, the C1228
recipe's AgentX front end) each chat request is rendered and tokenized before
the router sees it, and the router then parses a body that carries the
prompt's token ids. A run file's ``ingress`` entry states that hop, and the
traffic LP (`compass_harness.traffic_lp`) prices it as `Ingress` does: a
request reaches the ingress at its send plus the HTTP channel's lookahead,
the time a client run's records start it at, waits there for the first of
``workers`` encode threads to come free, first come first served, holds that
thread ``fixed_s + per_byte_s * bytes``, and reaches the router
``router_per_byte_s * bytes`` after it leaves the thread: the router's parse of
the body the ingress forwards, before it places the request. ``bytes`` is the
body the client sent, the one size a client knows before it sends: the ingress
encodes tokens, and the body it forwards adds their ids, so both terms are per
byte of a corpus whose bytes a token are those of the requests they were
fitted on (AgentX's 3.8). A request that has no body, one aiperf failed before
it sent it, takes no thread and reaches the router at its arrival at the
ingress.

The entry's terms: ``fixed_s`` and ``per_byte_s``, each a non-negative number
of seconds; ``workers``, the encode threads of all the ingress's processes
(the recipe's 4 x 16), taken as one pool, since the kernel spreads the client's
connections over the processes; ``router_per_byte_s``, optional, 0 when absent; and
``bytes``, optional, the client bodies it was fitted over, ``[lo, hi]``.
`compass fit ingress` fits one from an ingress's stamps (`atom.compass.fit.ingress`).
The ingress's own refusals (``--queued`` beyond the busy threads answers 503)
are not priced: a simulated ingress takes every request.

A `host_laws` entry states its deployment's ingress as its ``ingress`` term,
which `atom.compass.drive.runfile` writes into the run file, a prefill-decode
run's from its prefill side's entry; a run file with no ``ingress`` sends as
it did before. Only aiperf's traffic LP prices it: `atom.compass.parity.traffic`
and the JIT sweep (`atom.compass.drive.sweep`) send as they did. The module
imports nothing of ATOM's, so an aiperf worker reads it without loading the
engine.
"""

import heapq
import json
import math
import os

#: The run-file key and the ``host`` entry's term.
KEY = "ingress"
#: The variable naming a simulated run's run file (`atom.compass.run.ENV`).
RUN_ENV = "ATOM_COMPASS_RUN"
#: A diagnostic ingress for a simulated run whose run file states none: an
#: ``ingress`` entry as JSON (`stated`). It stands for the client's own
#: preparation of each request before its head reaches the frontend, as the
#: #752 DeepSeek-V4.1 Cell A v2 records show it: a burst whose heads arrive
#: within ~0.3 s is read smallest body first (engine receipt against input
#: size +0.42..+0.89, every target c8-c512 and M3.1's), and one the client
#: spreads over seconds (DEPA2 c=512's 300 s burst: 287 MB over 15 s) in the
#: order its heads arrive (-0.24..+0.27). Its terms are read off target
#: records, so a run under it is a diagnostic, never a graded candidate.
DIAG_ENV = "COMPASS_DIAG_CLIENT_INGRESS"
#: The terms an ``ingress`` entry requires, and those it may state.
REQUIRED = ("fixed_s", "per_byte_s", "workers")
OPTIONAL = ("router_per_byte_s", "bytes")


def check(entry, where: str = "ingress") -> dict:
    """`entry`, refused unless it is an ``ingress`` entry: `REQUIRED` and any of
    `OPTIONAL`, each seconds a finite non-negative number, ``workers`` a
    positive int, ``bytes`` two non-negative numbers, the lower first."""
    seconds = ("fixed_s", "per_byte_s", "router_per_byte_s")
    hull = entry.get("bytes", [0, 0]) if isinstance(entry, dict) else None
    if not (
        isinstance(entry, dict)
        and set(REQUIRED) <= entry.keys() <= {*REQUIRED, *OPTIONAL}
        and all(_number(entry[k]) for k in seconds if k in entry)
        and type(entry["workers"]) is int
        and entry["workers"] >= 1
        and isinstance(hull, list)
        and len(hull) == 2
        and all(map(_number, hull))
        and hull[0] <= hull[1]
    ):
        raise ValueError(
            f"{where} is {entry!r}; it takes fixed_s and per_byte_s, each a "
            "non-negative number of seconds, workers, a positive count of encode "
            "threads, and, optional, router_per_byte_s, non-negative seconds, and "
            "bytes, the client bodies it was fitted over, [lo, hi]"
        )
    return entry


def _number(v) -> bool:
    return type(v) in (int, float) and math.isfinite(v) and v >= 0


def stated(run: dict | None) -> dict | None:
    """The ``ingress`` entry the run file `run` states, checked; None for a
    real run or a run file with none, unless `DIAG_ENV` names one as JSON,
    which a run file with none then takes (its traffic LP and workers alike)."""
    if run is None:
        return None
    if run.get(KEY) is None:
        diag = os.environ.get(DIAG_ENV)
        return check(json.loads(diag), DIAG_ENV) if diag else None
    return check(run[KEY], f"the run file's {KEY}")


def stated_here() -> dict | None:
    """The ``ingress`` entry of the run file `RUN_ENV` names, as `stated`."""
    path = os.environ.get(RUN_ENV)
    if not path:
        return None
    with open(path) as f:
        return stated(json.load(f))


class Ingress:
    """The ingress's encode threads on the traffic LP's clock.

    `admit` takes the requests in the order the traffic LP stamps them, which
    is the order of their sends and then of their seqs, so a run repeats.
    """

    def __init__(self, entry: dict, lookahead_s: float) -> None:
        # lookahead_s: the HTTP channel's, the client's send to the ingress.
        check(entry)
        self.fixed_s, self.per_byte_s = entry["fixed_s"], entry["per_byte_s"]
        self.router_per_byte_s = entry.get("router_per_byte_s", 0)
        self.lookahead_s = lookahead_s
        self._free = [-math.inf] * entry["workers"]  # each thread's next free time

    def admit(self, now: float, body_bytes: int | None) -> tuple[float, float]:
        """A request sent at `now` with a body of `body_bytes`, or None for
        none: when it reaches the ingress, its start as the client records it,
        and when it reaches the router."""
        at = now + self.lookahead_s
        if body_bytes is None:
            return at, at
        start = max(at, heapq.heappop(self._free))
        done = start + self.fixed_s + self.per_byte_s * body_bytes
        heapq.heappush(self._free, done)
        return at, done + self.router_per_byte_s * body_bytes
