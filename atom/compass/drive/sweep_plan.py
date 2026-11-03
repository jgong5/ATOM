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
  tokens `ModelRunner.warmup_max_tokens` gives one forward, each one row,
  count t on rank (t - 1) mod the DP width, one count a rank a round. A count
  past the longest prompt runs as several rows behind one parked decode.
- B: lockstep steps, one point a rank, its rows queued behind 1 to 16 parked
  decodes. Two rows whose first is 1 to 16 tokens long, so the second row's
  offset in the batch takes every residue mod 16, by every count of parked
  decodes; under two-batch overlap again, each step's first point past
  `ATOM_TBO_PREFILL_MIN_TOKENS` so the step splits. Then one row of 1, 2, 4
  and so on tokens up to the budget on a context already computed, doubling up
  to the longest prompt: its own prompt's earlier chunks, from the scheduler's
  chunk alignment below the budget, where a first row of the budget's rest
  ends the first chunk, and from the budget up; with prefix caching on, a
  prefix another request computed, from one KV block up.

Every request asks for two output tokens, so its decode is reached too, with
the workload's other sampling arguments copied unchanged; a parked decode asks
for `PARKED_TOKENS`. No point is outside the engine's bounds (rows and parked
decodes within `max_num_seqs`, contexts and their output tokens within
`max_model_len`, KV blocks within the smallest pool, with a drafter's tokens)
or the workload's (no prompt longer than its longest). A step-record line
covers an A point when it is any rank's unsplit step of the point's token
count, and a B point when its prefill rows are the point's, in batch order,
and its split is the point's.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from atom.model_engine.dynamic_chunking import chunk_alignment

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
    parked on the rank before the step. `split` is None for an A point, else
    whether two-batch overlap splits the B point's step. `step` is the A round
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


def plan(engine, workload: dict, tbo_min: int, a_only: bool = False) -> list[Point]:
    """The points a sweep of a cell sets out to cause, A's rounds then B's steps.

    `engine` is the directory holding each DP rank's ``dp<r>.engine.json``
    (`engines`). `workload` maps ``traces`` to the traces the client sends,
    each a mapping holding its ``requests``, and ``sampling`` to the fields
    the client adds to every request body, ``ignore_eos`` among them; a
    point's `body` is those with ``max_tokens`` set to `OUTPUT_TOKENS`.
    `tbo_min` is the server's ``ATOM_TBO_PREFILL_MIN_TOKENS``; with `a_only`,
    for a deployment no PrefillDelayer coalesces, B has no point.
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
    points = []
    for t in range(1, b + 1):
        rows = _rows(t, cap)
        held, rank, step = int(len(rows) > 1), (t - 1) % ranks, (t - 1) // ranks
        if _fits(e, rows, held):
            cached = ("",) * len(rows)
            points.append(Point(rank, rows, cached, held, None, step, body))
    if a_only:
        return points
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
    contexts = {
        "chunk": _doubling(chunk_alignment(e["kv_cache_block_size"]), b)
        + _doubling(b, cap)
    }
    if e["enable_prefix_caching"]:
        contexts["prefix"] = _doubling(e["kv_cache_block_size"], cap)
    families += [
        [
            (((q, c + q),), (kind,), 1, False)
            for c in levels
            for q in _levels(b)
            if c + q <= cap
        ]
        for kind, levels in contexts.items()
    ]
    step, seen = -(-b // ranks), set()
    for family in families:
        kept = [d for d in family if (d[3], d[0]) not in seen and _fits(e, d[0], d[2])]
        seen |= {(d[3], d[0]) for d in kept}
        points += [
            Point(i % ranks, *d, step + i // ranks, body) for i, d in enumerate(kept)
        ]
        step += -(-len(kept) // ranks)
    return points


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
