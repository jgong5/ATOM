# SPDX-License-Identifier: MIT
"""The points a JIT sweep of one cell sets out to cause, and which of them a
step record shows happened.

A point is one DP rank's prefill step: the batch it runs, as a step-record
line states it (`atom/compass/parity/`), and the requests that cause it. `plan`
derives them from the cell's engine, each DP rank's ``dp<r>.engine.json``,
and its workload, the traces a client sends and the sampling arguments it
sends them with. Per rank it plans one step for every prefill token count
from 1 to the rank's budget, the tokens `ModelRunner.warmup_max_tokens`
gives one forward, and designs each step's composition:

- every other count runs two rows whose first is 1 to 16 tokens long, so the
  second row's offset in the batch takes every residue mod 16; the rest run
  one row, or under two-batch overlap two rows of half the count each, so
  the split, which falls on a request boundary, has one to fall on;
- a step runs with 0 to 15 requests held on the rank, each a one-token
  prompt whose prefill ran before the step's and whose decode runs after it,
  so the step's requests take state slots past theirs;
- every `SPACING`-th count runs one row on a context already computed: the
  final chunk of a prompt chunked in earlier steps of the engine's whole
  budget, and with prefix caching on a prompt behind a prefix another
  request computed, or a second turn behind its first, whose two output
  tokens open the row. The contexts before the row step through 1, 2, 4 and
  so on chunks or KV blocks, up to the longest the workload sends.

Every request asks for two output tokens, so its decode is reached too, with
the workload's other sampling arguments copied unchanged. No point is outside
the engine's bounds (rows within `max_num_seqs`, contexts and their output
tokens within `max_model_len`, KV blocks within the pool, with a drafter's tokens) or the
workload's (no prompt longer than its longest); a count no composition fits
has no point. A step-record line covers a point when it is a step of the
point's rank whose prefill rows are the point's, in batch order.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

#: Residues mod 16 the second row's offset takes, and the held requests 0 to 15.
SLOTS = 16
#: Every SPACING-th token count runs a row on a context already computed.
# ponytail: one count in SPACING; denser if a context length's key stays unbuilt.
SPACING = 128
#: The output tokens every request asks for: the first ends its prefill, the
#: second its one decode.
OUTPUT_TOKENS = 2
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
    leading tokens of its prompt, 'turn' its first turn, a prompt of that many
    tokens whose two output tokens open the row's query. `held` requests, each
    a one-token prompt, are prefilled on the rank before the step and decode
    after it. `body` is the sampling arguments every one of these requests
    sends.
    """

    rank: int
    rows: tuple[tuple[int, int], ...]
    cached: tuple[str, ...]
    held: int
    body: dict = field(hash=False)


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


def _fits(engine: dict, rows, held: int) -> bool:
    """Whether the rows and held requests fit the engine's sequences and KV
    pool, each with its output tokens and a drafter's tokens past them."""
    drafts = (engine.get("speculative") or (None, 0))[1]
    extra = OUTPUT_TOKENS + drafts + max(0, drafts - 1)
    size = engine["kv_cache_block_size"]
    blocks = sum(-(-(c + extra) // size) for _, c in rows)
    blocks += held * -(-(1 + extra) // size)
    return (
        held >= 0
        and len(rows) + held <= engine["max_num_seqs"]
        and blocks <= engine["num_kvcache_blocks"]
    )


def _cached(engine: dict, i: int, t: int, cap: int, kinds: tuple) -> tuple | None:
    """Count `t`'s one row on a context already computed, where count index
    `i` is one `SPACING` gives one and a level of its kind fits under `cap`."""
    if not kinds or i % SPACING != SPACING - 1:
        return None
    slot = i // SPACING
    kind = kinds[slot % len(kinds)]
    if kind == "turn" and t <= OUTPUT_TOKENS:
        return None
    unit = (
        engine["max_num_batched_tokens"]
        if kind == "chunk"
        else engine["kv_cache_block_size"]
    )
    top = (cap - t) // unit
    if top < 1:
        return None
    levels = _levels(top)
    return ((t, levels[slot // len(kinds) % len(levels)] * unit + t),), (kind,)


def _two(engine: dict, i: int, t: int, cap: int) -> tuple | None:
    """Count `t`'s two rows on no context: at odd `i` the first 1 to 16
    tokens, and under two-batch overlap at even `i` two halves; None where
    neither holds or a row would pass `cap`."""
    first = 1 + i // 2 % SLOTS if i % 2 else t // 2 if engine["enable_tbo"] else 0
    if 1 <= first < t and max(first, t - first) <= cap:
        return ((first, first), (t - first, t - first)), ("", "")
    return None


def _rows(t: int, cap: int) -> tuple:
    """Count `t` on no context in as few rows as hold it, none past `cap`."""
    n = -(-t // cap)
    lengths = (cap,) * (n - 1) + (t - cap * (n - 1),)
    return tuple((q, q) for q in lengths), ("",) * n


def plan(engine, workload: dict) -> list[Point]:
    """The points a sweep of a cell sets out to cause, by rank and token count.

    `engine` is the directory holding each DP rank's ``dp<r>.engine.json``
    (`engines`). `workload` maps ``traces`` to the traces the client sends,
    each a mapping holding its ``requests``, and ``sampling`` to the fields
    the client adds to every request body, ``ignore_eos`` among them; a
    point's `body` is those with ``max_tokens`` set to `OUTPUT_TOKENS`.
    """
    top = max((longest(t["requests"]) for t in workload["traces"]), default=0)
    if top < 1:
        raise ValueError("the workload states no prompt")
    body = {**workload.get("sampling", {}), "max_tokens": OUTPUT_TOKENS}
    points = []
    for rank, e in engines(engine).items():
        cap = min(e["max_model_len"] - OUTPUT_TOKENS, top)
        kinds = ("chunk",) if cap > e["max_num_batched_tokens"] else ()
        if e["enable_prefix_caching"]:
            kinds += ("prefix", "turn")
        for i in range(budget(e)):
            t = i + 1
            designed = (_cached(e, i, t, cap, kinds), _two(e, i, t, cap))
            for composed in (*designed, _rows(t, cap)):
                if composed is None:
                    continue
                rows, cached = composed
                held = min(i % SLOTS, e["max_num_seqs"] - len(rows))
                if _fits(e, rows, held):
                    points.append(Point(rank, rows, cached, held, body))
                    break
    return points


def covered(points, steps: dict[int, list[dict]]) -> tuple[list, list]:
    """`points` split into those `steps` holds and those it lacks, each in
    order. `steps` is a record as `atom.compass.parity.read` gives it; a line
    holds a point when it is no DP dummy step of the point's rank and its
    prefill rows, in batch order, are the point's rows."""
    ran = {
        (rank, tuple((q, c) for q, c, decode in line["rows"] if not decode))
        for rank, lines in steps.items()
        for line in lines
        if not line["is_dummy_run"]
    }
    hit, missing = [], []
    for point in points:
        (hit if (point.rank, point.rows) in ran else missing).append(point)
    return hit, missing
