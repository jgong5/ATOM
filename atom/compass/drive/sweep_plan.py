# SPDX-License-Identifier: MIT
"""The points a JIT sweep of one cell sets out to cause, and which of them a
step record shows happened.

A point is one DP rank's prefill step: the batch it runs, as a step-record
line states it (`atom/compass/parity/`), and the requests that cause it. `plan`
derives them from the cell's engine, each DP rank's ``dp<r>.engine.json``,
and its workload, the traces a client sends and the sampling arguments it
sends them with. The ranks share one walk, since they write one set of cache
roots: each point is planned once, on the rank its place in the walk names.

- A: one step for every prefill token count from 1 to a rank's budget, the
  tokens `ModelRunner.warmup_max_tokens` gives one forward, each one row, in
  rounds of one count a rank, from rank 0 up. A round that two-batch overlap
  would split leaves its last rank without a row, since a rank with no
  prefill vetoes the split and a split step runs its halves, not the count. A
  count past the longest prompt runs as several rows behind one parked decode.
- B: lockstep steps, one point a rank, its rows queued behind 1 to 16 parked
  decodes. Two rows whose first is 1 to 16 tokens long, so the second row's
  offset in the batch takes every residue mod 16, by every count of parked
  decodes; under two-batch overlap again, each step's first point past
  `ATOM_TBO_PREFILL_MIN_TOKENS` so the step splits, and again with the rows
  swapped and every first row past the minimum, so the split falls inside it.
  Then one row of 1, 2, 4 and so on tokens up to the budget, each with the 15
  sizes past it, on a context already computed: its own prompt's earlier
  chunks, from the scheduler's chunk alignment doubling below the budget,
  where a first row of the budget's rest ends the first chunk, then the
  budget doubling, then the last aligned context a whole budget fits behind
  in the longest prompt; with prefix caching on, a prefix another request
  computed, from one KV block up. A step's chunked rows take as many earlier
  chunks each.

Every request asks for two output tokens, so its decode is reached too, with
the workload's other sampling arguments copied unchanged; a parked decode asks
for `PARKED_TOKENS`. No point is outside the engine's bounds (rows and parked
decodes within `max_num_seqs`, contexts and their output tokens within
`max_model_len`, KV blocks within the smallest pool, with a drafter's tokens)
or the workload's (no prompt longer than its longest). Whether two-batch
overlap splits a step is ATOM's prefill rule over the step's ranks
(`atom.compass.validate.bound._split`), under this process's
``ATOM_TBO_PREFILL_MIN_TOKENS`` and ``ATOM_TBO_PREFILL_TOKEN_SPLIT``, which
must be the server's. A step-record line covers an A point when it is any
rank's unsplit step of the point's token count, and a B point when its
prefill rows are the point's, in batch order, and its split is the point's.
"""

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

from atom.compass.validate.bound import _split
from atom.model_engine.dynamic_chunking import chunk_alignment
from atom.utils import envs

#: Residues mod 16 the second row's offset takes, and the parked decodes 1 to 16.
SLOTS = 16
#: The output tokens every request asks for: the first ends its prefill, the
#: second its one decode.
OUTPUT_TOKENS = 2
#: A parked decode's output tokens: more decode steps than parking the rest,
#: the rows' arrival and the PrefillDelayer's stall take, so it outlasts the
#: step's fire, and few enough that waiting for its end costs little.
PARKED_TOKENS = 128
#: The engine fields `plan` reads; `speculative` and
#: `prefill_context_parallel_size` too, where a rank states them.
ENGINE = (
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "kv_cache_block_size",
    "num_kvcache_blocks",
    "data_parallel_size",
    "enable_dp_attention",
    "enable_tbo",
    "enable_prefix_caching",
)


@dataclass(frozen=True)
class Point:
    """One DP rank's prefill step and the requests that cause it.

    `rows` is the step's prefill rows in batch order, each (query tokens,
    context tokens), as its record line states them. `cached` says for each
    row how the context before its query was computed: '' none, 'chunk' its
    own prompt's earlier chunks, 'prefix' another request sharing that many
    leading tokens of its prompt. `held` decodes, each a one-token prompt, are
    parked on the rank before the step. `split` is None for an A point, whose
    step two-batch overlap does not split, else whether it splits the B
    point's step. `step` is the A round
    or B step it is sent in. `body` is the sampling arguments every one of
    these requests sends.
    """

    rank: int
    rows: tuple[tuple[int, int], ...]
    cached: tuple[str, ...]
    held: int
    split: bool | None
    step: int
    body: dict = field(hash=False)


def key(point: Point) -> tuple:
    """What a record line covers `point` by: an A point's token count, a B
    point's split and rows."""
    if point.split is None:
        return None, sum(q for q, _ in point.rows)
    return point.split, point.rows


def budget(engine: dict) -> int:
    """A rank's prefill tokens in one forward, as `ModelRunner.warmup_max_tokens`
    sizes them from `engine`."""
    tokens = engine["max_num_batched_tokens"]
    if not engine["enable_dp_attention"]:
        tokens //= engine["data_parallel_size"]
    return max(1, tokens // engine.get("prefill_context_parallel_size", 1))


def engines(directory) -> dict[int, dict]:
    """Each DP rank's ``dp<r>.engine.json`` in `directory`, by rank; refused
    unless every rank of the group states one engine, KV block count aside."""
    files = Path(directory).glob("dp*[0-9].engine.json")
    found = {int(p.name[2 : -len(".engine.json")]): p for p in files}
    if not found:
        raise ValueError(f"{directory} holds no dp<r>.engine.json")
    stated = {r: json.loads(p.read_text()) for r, p in sorted(found.items())}
    for rank, engine in stated.items():
        if missing := [k for k in ENGINE if k not in engine]:
            raise ValueError(f"dp{rank}.engine.json states no {', '.join(missing)}")
    (first, engine), *rest = stated.items()
    for rank, other in rest:
        keys = (engine.keys() | other.keys()) - {"num_kvcache_blocks"}
        if differ := [k for k in sorted(keys) if engine.get(k) != other.get(k)]:
            raise ValueError(
                f"dp{first} and dp{rank} state two engines, which differ on "
                + ", ".join(differ)
            )
    if absent := sorted(set(range(engine["data_parallel_size"])) - stated.keys()):
        raise ValueError(
            f"{directory} states no engine for DP rank "
            + ", ".join(map(str, absent))
            + f" of {engine['data_parallel_size']}"
        )
    return stated


def longest(requests) -> int:
    """The longest prompt among a trace's `requests`, a subagent's included."""
    return max(
        (r["in"] if "in" in r else longest(r.get("requests", ())) for r in requests),
        default=0,
    )


def _levels(top: int) -> list[int]:
    """1, 2, 4 and so on up to `top`, and `top`."""
    return sorted({*(1 << j for j in range(top.bit_length())), top})


def _doubling(start: int, below: int) -> list[int]:
    """`start`, twice it and so on, each under `below`."""
    return [start << j for j in range(max(0, (below - 1) // start).bit_length())]


def _queries(b: int) -> list[int]:
    """Each of `_levels(b)` and the 15 sizes past it, or the 16 up to `b`
    where those pass it: every residue mod 16 at every size."""
    starts = {max(1, min(q, b + 1 - SLOTS)) for q in _levels(b)}
    return sorted({s + r for s in starts for r in range(SLOTS) if s + r <= b})


def _contexts(start: int, b: int, cap: int) -> list[int]:
    """`start` doubling below the budget `b`, `b` doubling below `cap`, and
    the top: the last multiple of `start` that leaves a budget before `cap`."""
    top = (cap - b) // start * start
    return sorted(
        {*_doubling(start, b), *_doubling(b, cap), *([top] if top > 0 else [])}
    )


def _fits(engine: dict, rows, held: int) -> bool:
    """Whether the rows and parked decodes fit the engine's sequences and KV
    pool, each with its output tokens and a drafter's tokens past them."""
    drafts = (engine.get("speculative") or (None, 0))[1]
    extra = drafts + max(0, drafts - 1)
    size = engine["kv_cache_block_size"]
    blocks = sum(-(-(c + OUTPUT_TOKENS + extra) // size) for _, c in rows)
    blocks += held * -(-(1 + PARKED_TOKENS + extra) // size)
    return (
        len(rows) + held <= engine["max_num_seqs"]
        and blocks <= engine["num_kvcache_blocks"]
    )


def _rows(t: int, cap: int) -> tuple:
    """Count `t` on no context in as few rows as hold it, none past `cap`."""
    n = -(-t // cap)
    lengths = (cap,) * (n - 1) + (t - cap * (n - 1),)
    return tuple((q, q) for q in lengths)


def _splits(engine: dict, ranks: int, step) -> bool:
    """Whether two-batch overlap splits a step whose ranks, from 0 up, run
    `step`'s rows, one rows tuple a rank, and the ranks past them none. At DP
    width 1 no record line states a split, so none is planned."""
    if not engine["enable_tbo"] or ranks == 1:
        return False
    lines = [{"rows": [(q, c, False) for q, c in rows]} for rows in step]
    return _split(lines + [{"rows": []}] * (ranks - len(lines))) is not None


def plan(engine, workload: dict, a_only: bool = False) -> list[Point]:
    """The points a sweep of a cell sets out to cause, A's rounds then B's steps.

    `engine` is the directory holding each DP rank's ``dp<r>.engine.json``
    (`engines`). `workload` maps ``traces`` to the traces the client sends,
    each a mapping holding its ``requests``, and ``sampling`` to the fields
    the client adds to every request body, ``ignore_eos`` among them; a
    point's `body` is those with ``max_tokens`` set to `OUTPUT_TOKENS`. With
    `a_only`, for a deployment no PrefillDelayer coalesces, B has no point.
    """
    top = max((longest(t["requests"]) for t in workload["traces"]), default=0)
    if top < 1:
        raise ValueError("the workload states no prompt")
    body = {**workload.get("sampling", {}), "max_tokens": OUTPUT_TOKENS}
    stated = engines(engine)
    pool = min(e["num_kvcache_blocks"] for e in stated.values())
    e = {**stated[0], "num_kvcache_blocks": pool}
    ranks, b = len(stated), budget(e)
    cap = min(e["max_model_len"] - OUTPUT_TOKENS, top)
    counts = [(r, int(len(r) > 1)) for t in range(1, b + 1) for r in [_rows(t, cap)]]
    counts = [(r, h) for r, h in counts if _fits(e, r, h)]
    points, step, i = [], 0, 0
    while i < len(counts):
        n = ranks - _splits(e, ranks, [r for r, _ in counts[i : i + ranks]])
        points += [
            Point(rank, r, ("",) * len(r), h, None, step, body)
            for rank, (r, h) in enumerate(counts[i : i + n])
        ]
        i, step = i + n, step + 1
    if a_only:
        return points
    tbo_min = envs.ATOM_TBO_PREFILL_MIN_TOKENS
    two = [
        (o, SLOTS * h + o, h) for h in range(1, SLOTS + 1) for o in range(1, SLOTS + 1)
    ]
    two = [
        (o, s, h)
        for o, s, h in two
        if o + s <= b and s <= cap and _fits(e, ((o, o), (s, s)), h)
    ]
    tbo = e["enable_tbo"] and tbo_min + 18 * SLOTS <= min(b, cap)
    families = []
    for split in (False, True) if tbo else (False,):
        # Split, each step's first rank passes the minimum.
        big = [
            s + tbo_min * (split and i % ranks == 0) for i, (_, s, _) in enumerate(two)
        ]
        families.append(
            [(((o, o), (s, s)), ("", ""), h, split) for (o, _, h), s in zip(two, big)]
        )
    if tbo:
        # Every rank's first row past the minimum, so the split falls inside it
        # and the ranks cost alike, which admits each second row.
        families.append(
            [(((tbo_min + s,) * 2, (o, o)), ("", ""), h, True) for o, s, h in two]
        )
    starts = {"chunk": chunk_alignment(e["kv_cache_block_size"])}
    if e["enable_prefix_caching"]:
        starts["prefix"] = e["kv_cache_block_size"]
    families += [
        [
            (((q, c + q),), (kind,), 1, False)
            for c in _contexts(start, b, cap)
            for q in _queries(b)
            if c + q <= cap
        ]
        for kind, start in starts.items()
    ]
    seen = set()
    for family in families:
        kept = [d for d in family if (d[3], d[0]) not in seen and _fits(e, d[0], d[2])]
        seen |= {(d[3], d[0]) for d in kept}
        # A step's chunked rows need as many earlier chunks each, so their
        # last chunks run in one step.
        for _, same in itertools.groupby(kept, lambda d: _chunks(d, b)):
            same = list(same)
            for i in range(0, len(same), ranks):
                group = same[i : i + ranks]
                split = _splits(e, ranks, [d[0] for d in group])
                points += [
                    Point(rank, rows, how, held, split, step, body)
                    for rank, (rows, how, held, _) in enumerate(group)
                ]
                step += 1
    return points


def _chunks(design, b: int) -> int:
    """The earlier chunks a designed point's row on its own prompt takes."""
    ((q, c), *_), how, *_ = design
    return -(-(c - q) // b) if how == ("chunk",) else 0


def covered(points, steps: dict[int, list[dict]]) -> tuple[list, list]:
    """`points` split into those `steps` holds and those it lacks, each in
    order. `steps` is a record as `atom.compass.parity.read` gives it; a line
    that is no DP dummy step holds the B point of its prefill rows and split, a
    ``tbo_tokens`` field, and unsplit the A point of its prefill tokens."""
    ran = set()
    for lines in steps.values():
        for line in lines:
            rows = tuple((q, c) for q, c, decode in line["rows"] if not decode)
            if rows and not line["is_dummy_run"]:
                split = "tbo_tokens" in line
                ran.add((split, rows))
                if not split:
                    ran.add((None, sum(q for q, _ in rows)))
    hit, missing = [], []
    for point in points:
        (hit if key(point) in ran else missing).append(point)
    return hit, missing
