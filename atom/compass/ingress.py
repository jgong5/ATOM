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

import collections
import heapq
import json
import math
import os

#: The run-file key and the ``host`` entry's term.
KEY = "ingress"
#: The variable naming a simulated run's run file (`atom.compass.run.ENV`).
RUN_ENV = "ATOM_COMPASS_RUN"
#: The client's own preparation of each request before its head reaches the
#: channel (`ClientPrep`): ``rate=<bytes a second>,workers=<W>``, optionally
#: ``,fixed=<seconds>``; unset or empty, none.
CLIENT_PREP_ENV = "COMPASS_HARNESS_CLIENT_PREP"
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
    real run or a run file with none."""
    if run is None or run.get(KEY) is None:
        return None
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


def client_prep_spec() -> dict | None:
    """`CLIENT_PREP_ENV`'s terms as ``{"rate": bytes/s, "workers": W, "fixed":
    s}``, None when unset or empty: ``rate`` a positive number, ``workers`` a
    positive int, ``fixed`` a number >= 0 (0 when absent); anything else is
    refused."""
    raw = os.environ.get(CLIENT_PREP_ENV) or ""
    if not raw:
        return None
    terms, bad = {"fixed": 0.0}, False
    for part in raw.split(","):
        name, _, value = part.partition("=")
        try:
            terms[name] = int(value) if name == "workers" else float(value)
        except ValueError:
            bad = True
    rate, workers, fixed = terms.get("rate"), terms.get("workers"), terms["fixed"]
    if (
        bad
        or set(terms) != {"rate", "workers", "fixed"}
        or not (_number(rate) and rate > 0)
        or not (type(workers) is int and workers >= 1)
        or not _number(fixed)
    ):
        raise ValueError(
            f"{CLIENT_PREP_ENV}={raw!r}; it takes rate=<bytes a second, > 0>,"
            "workers=<a positive count>[,fixed=<seconds >= 0>]"
        )
    return terms


class ClientPrep:
    """The client's preparation of each request before its head reaches the
    channel, on the traffic LP's clock: processor sharing.

    The client prepares at most ``workers`` requests at once and shares
    ``rate`` bytes a second equally among those it prepares; a request's work
    is its body's bytes plus ``fixed`` seconds' worth (``fixed x rate``). The
    requests sent at one instant are taken in the order the traffic LP stamps
    them, its first ``workers`` at once and each of the rest as one finishes,
    so a burst of no more than ``workers`` finishes smallest first and a larger
    one mostly in that order, spread over its bytes / ``rate``. Requests sent
    later start once every earlier one is prepared: what was stamped stays
    stamped. A request reaches the channel its lookahead after it is
    prepared, and its record starts at its send, where the client's does.

    #752 DeepSeek-V4.1 Cell A v2 and M3.1 records: a burst whose heads reach
    the frontend within ~0.3 s is received about smallest first (engine
    receipt against input size +0.42..+0.89), and the c=512 300 s burst,
    287 MB whose heads arrive over 15 s, in head order (-0.24..+0.27); a lone
    request's head follows its send by tens of milliseconds.
    """

    def __init__(self, spec: dict, lookahead_s: float) -> None:
        self.rate, self.workers = spec["rate"], spec["workers"]
        self.fixed_work = spec.get("fixed", 0.0) * self.rate
        self.lookahead_s = lookahead_s
        self._free = -math.inf  # when every request admitted so far is prepared

    def admit_batch(
        self, now: float, sizes: list[int | None]
    ) -> list[tuple[float, float]]:
        """The requests sent at `now` with bodies of `sizes` bytes, None for a
        request with none, in stamp order: each one's record start, `now`, and
        its arrival at the channel. One with no body takes no preparation."""
        at = now + self.lookahead_s
        out: list[tuple[float, float]] = [(now, at)] * len(sizes)
        queue = collections.deque(i for i, b in enumerate(sizes) if b is not None)
        if not queue:
            return out
        t = max(now, self._free)
        left: dict[int, float] = {}
        while queue or left:
            while queue and len(left) < self.workers:
                i = queue.popleft()
                left[i] = sizes[i] + self.fixed_work
            least = min(left.values())
            t += least * len(left) / self.rate
            for i in list(left):
                left[i] -= least
                if left[i] <= 1e-9 * (sizes[i] + self.fixed_work + 1):
                    del left[i]
                    out[i] = (now, t + self.lookahead_s)
        self._free = t
        return out
