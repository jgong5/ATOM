# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The records a step linear law is fitted from, planned from a configuration
alone, as `scripts/compass/calibrate.sh` takes them: each its RECORD argument.

A law has a decode vector per capture rung (`step_linear.py`), and a rung N
comes from a record at N sessions that starts every trajectory at its first
turn: those sessions decode together at the start, where a record of sampled
starts batches far fewer of them
([#1840](https://github.com/jgong5/ATOM/issues/1840)). A data-parallel group
runs every rank at the rung of its widest rank's batch (`ForwardMode.decide`),
so the record takes N sessions a rank, N times the group's width. The rungs
are the capture ladder's, then the eager rung where the scheduler's bound
passes the widest captured size, taken at the bound.

The prefill vector fits from trace runs alone, and its held-out band needs two
(`step_linear_fit.fit`): one at concurrency 16 and one at 4, as the quickstart
takes them, where `compass record --trace` takes one; where it refuses, none
is planned and the law's prefill stays unfitted. The plan reads no law: a law
short of a rung is fitted again on the whole list, since a published law takes
no record.

A designed-load calibration plans its design records from a served real
record instead: its engine files, its steps and the calibration traces its
client run was given, each design record by its own seed.

    python -m atom.compass.backends.step_linear_plan design|trace REAL SEED

prints a decode design plan (`design`) or a prefill point plan
(`trace_design`) as JSON. A session is one conversation of the calibration
traces cut at three lengths, its first message opening with a marker of its
own, so no two sessions share a KV block: each length drawn within `BAND`
percentiles of its level, the real record's rows at `LEVELS` on that DP rank,
and the conversation one whose longest prompt passes the top length. A rung or
point of n rows scales the three by min(1, `CAP` P / T), P the rank's KV pool
and T the top lengths of its first n sessions. A hit resumes at the last whole
KV block a prompt shares with what its session has cached, short of the
prompt's last block, and on a model whose hit rebuilds its state from the block
below (`_hit_replay`) at that block: each plan counts so the tokens its fills
compute, beside the rows it sends.
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from itertools import pairwise, zip_longest
from pathlib import Path

import numpy as np

from atom.compass.backends.json_file import read_jsonl
from atom.compass.backends.step_linear import EAGER
from atom.compass.backends.step_linear_fit import _stamps

#: aiperf's arguments that start every trajectory at its first turn.
FIRST_TURN = "--trajectory-start-min-ratio 0 --trajectory-start-max-ratio 0"
#: The trace runs, each a `compass record --trace` of its concurrency's record.
TRACE_RUNS = ("trace 16", "trace 4")
#: The percentiles of the real record's rows the three levels sit at, bottom
#: first, and the percentiles either side of a level its draws fall within.
LEVELS, BAND = (10, 50, 90), 10
#: The share of a rank's KV pool a rung's or a point's top cuts may hold.
CAP = 0.8
#: The decode steps a design group's rows run.
DECODE_STEPS = 40
#: The output tokens a prefill point's request asks for.
POINT_TOKENS = 2
#: The bytes ATOM's tokenizer cache keeps at most of a prompt, a character's
#: and an id's: its text, its ids with their block hashes and its segments' ids.
CHAR_BYTES, ID_BYTES = 4, 12
#: The model types whose prefix hit, under encoder bounded replay, rebuilds its
#: window state from the KV block below where it resumes.
REPLAYING = ("deepseek_v41", "deepseek_v41_text")
#: The HTTP intake stamps a real server writes beside its steps
#: (`atom.compass.run.INTAKE_RECORD`).
INTAKE = "intake.stamps"
#: A decode design's level groups: each kind's first row's level, then the
#: rest's, 0 the bottom level and 2 the top.
LEVEL_GROUPS = (("bottom", 0, 0), ("middle", 1, 1), ("top", 2, 2), ("one_long", 2, 0))


def scheduled_rungs(ladder: list[int], bound: int) -> list:
    """Each decode rung a configuration schedules whose scheduler's widest
    decode batch is `bound`, and that captures `ladder`, ascending, each at
    most `bound`: each of `ladder`, then `EAGER` where `bound` passes the
    widest, as on an eager runner, which captures none."""
    return [*ladder, *([EAGER] if bound > max(ladder, default=0) else [])]


def records(ladder: list[int], bound: int, width: int, traced: bool) -> list[str]:
    """`calibrate.sh`'s RECORD arguments for a law with a vector at every rung
    `scheduled_rungs` gives: for each, ascending, a first-turn record of its
    batch, `bound` at the eager rung, at each of `width` data-parallel ranks;
    then, when `traced`, the trace runs."""
    rows = [bound if r == EAGER else r for r in scheduled_rungs(ladder, bound)]
    first_turn = [f"{n * width} {FIRST_TURN}" for n in rows]
    return first_turn + (list(TRACE_RUNS) if traced else [])


def design(real, seed: int) -> dict:
    """The decode design plan of the real record at `real`, its draws by `seed`.

    Per DP rank: `levels`, the decode rows' contexts at `LEVELS`; and
    `sessions`, each ``{"trace", "cuts"}``, the cuts its prompts' tokens
    unscaled, as many as the widest group's rows. `groups` in the order they
    run: per scheduled rung, ascending, the bottom, middle and top level's
    group, the one-long group (one top row over bottom rows) and, from four
    rows, the half-long pair (half top, half bottom, alternating then in halves,
    the second sent on the first's fill), each at the rung's rows on every rank,
    `bound` at the eager rung, and the first four kinds' gap batch after each at
    the fewest rows that replay the rung; then under DP the middle level's group
    with the full rank, moving from group to group and rung to rung, beside
    the other ranks at half its rows, at one row, with the next rank empty and
    every other rank empty. A group is ``{"rung", "kind", "gap", "fill",
    "rows"}``, `rows` per rank in the order sent, each ``[session, prompt
    tokens]``, row i of a group session i but in the half-long pair. Each
    group's prompts are sent first, one output token each, when `fill`, then
    again as its rows at once, each asking `max_tokens`, `DECODE_STEPS` at the
    real record's tokens a decode step. `tokenize_cache_mb` holds every
    prompt sent, at `CHAR_BYTES` a character, taken as the real record's
    request body bytes a prompt token, and `ID_BYTES` an id; `rows` and
    `fill_tokens` count, per rank, the rows sent and the tokens the fills
    compute."""
    real = Path(real)
    stated, served, advance, prompts = _served(real)
    first = stated[0]
    block, ranks = first["kv_cache_block_size"], len(stated)
    drafts = (first.get("speculative") or (None, 0))[1]
    bound = min(first["max_num_seqs"], first["max_num_batched_tokens"] // (1 + drafts))
    rungs = scheduled_rungs(first["ladder"], bound)
    width = {r: bound if r == EAGER else r for r in rungs}
    max_tokens = round(DECODE_STEPS * advance)
    limit = first["max_model_len"] - max_tokens - drafts
    rng = random.Random(seed)
    traces = _traces(real)
    sessions = [
        _sessions(
            rng, served[k][0], traces, max(width.values()), lambda c: min(c, limit), k
        )
        for k in range(ranks)
    ]
    pools = [stated[k]["num_kvcache_blocks"] * block for k in range(ranks)]
    groups, prev = [], 0
    for index, rung in enumerate(rungs):
        r = width[rung]
        scale = [_scale(pools[k], sessions[k][:r]) for k in range(ranks)]
        # Each group's kind, whether a gap batch, whether filled, and its rows
        # on each rank, each a (session, level).
        batches = [(r, False)] + ([(prev + 1, True)] if r > prev + 1 else [])
        made = [
            (kind, gap, True, [list(enumerate([head] + [rest] * (n - 1)))] * ranks)
            for kind, head, rest in LEVEL_GROUPS
            for n, gap in batches
        ]
        if r >= 4:
            tops = [(i, 2) for i in range(r // 2)]
            bottoms = [(i, 0) for i in range(r // 2, r)]
            alternating = [x for pair in zip_longest(tops, bottoms) for x in pair if x]
            made.append(("half_alternating", False, True, [alternating] * ranks))
            made.append(("half_halves", False, False, [tops + bottoms] * ranks))
        # The other ranks' rows beside the full rank's, the full rank moving
        # from group to group and from rung to rung.
        others = {"dp_half": r // 2, "dp_one": 1, "dp_one_empty": r, "dp_empty": 0}
        for j, (kind, n) in enumerate(others.items() if ranks > 1 else ()):
            at = (index + j) % ranks
            counts = [n] * ranks
            counts[at] = r
            if kind == "dp_one_empty":
                counts[(at + 1) % ranks] = 0
            made.append(
                (kind, False, True, [[(i, 1) for i in range(c)] for c in counts])
            )
        for kind, gap, fill, placed in made:
            tokens = [
                [
                    [i, max(1, int(scale[k] * sessions[k][i]["cuts"][level]))]
                    for i, level in own
                ]
                for k, own in enumerate(placed)
            ]
            groups.append(
                {"rung": rung, "kind": kind, "gap": gap, "fill": fill, "rows": tokens}
            )
        prev = r
    replay = _hit_replay(real, block)
    cached = [defaultdict(int) for _ in range(ranks)]
    fill_tokens, rows, sent = [0] * ranks, [0] * ranks, set()
    for g in groups:
        for k, own in enumerate(g["rows"]):
            rows[k] += len(own)
            for i, tokens in own:
                sent.add((k, i, tokens))
                if g["fill"]:
                    fill_tokens[k] += _computed(cached[k][i], tokens, block, replay)
                    cached[k][i] = max(cached[k][i], tokens)
    body = sum(e[3] for e in _stamps(real / "steps" / INTAKE) if e[2] == "body")
    per_token = CHAR_BYTES * body / prompts + ID_BYTES
    return {
        "seed": seed,
        "block": block,
        "hit_replay": replay,
        "max_tokens": max_tokens,
        "tokenize_cache_mb": math.ceil(
            per_token * sum(t for *_, t in sorted(sent)) / 2**20
        ),
        "levels": [_levels(served[k][0]) for k in range(ranks)],
        "sessions": sessions,
        "groups": groups,
        "rows": rows,
        "fill_tokens": fill_tokens,
    }


def trace_design(real, seed: int) -> dict:
    """The prefill point plan of the real record at `real`, at DP 1 and PP 1,
    where `compass record --trace` runs, its draws by `seed`.

    `levels` are the prefill rows' cached histories, N_KV - N_Q, at `LEVELS`
    over the rows that have one, and `sessions` each ``{"trace", "cuts"}``, its
    histories at whole KV blocks, unscaled. A point is n rows of q new tokens,
    n doubling from 1 while the n rows beside its one parked decode fit
    `max_num_seqs`: with no history, q doubling from 1 to the budget b while n
    q fits it, and one row at each prefill token-graph bucket and one past it;
    on each history level, the same while each row's computed tokens fit b
    n times, and one row computing b and one b + 1, whose first chunk is a
    middle chunk. `points` run with no history first, then list by list, the
    points of one scaling sharing one, bottom level first, n and q ascending,
    each ``{"level", "rows", "record"}``: `level` None with no history, `rows`
    each ``[session, history tokens, q]``, session None with no history, a row
    on a history sent after its fill, the history and one more of its trace's
    tokens, and `record` each row as the step running it states it, its first
    chunk, ``[query tokens, context tokens]``. Every request asks
    `POINT_TOKENS`. A row's new tokens start with tokens no other row sent
    after that history, and a row with no history opens with a marker of its
    own, so no row hits past its history. `rows` and `fill_tokens` count the
    rows sent and the tokens the fills compute."""
    real = Path(real)
    stated, served, _, _ = _served(real)
    if len(stated) > 1 or any((real / "steps").glob("pp*")):
        raise ValueError(
            f"{real} holds a DP or pipeline group, and a trace design record runs "
            "at DP 1 and PP 1, where `compass record --trace` runs"
        )
    engine = stated[0]
    block, b, bound = (
        engine["kv_cache_block_size"],
        engine["max_num_batched_tokens"],
        engine["max_num_seqs"],
    )
    replay = _hit_replay(real, block)
    tallest = (engine["max_model_len"] - POINT_TOKENS - b - 1) // block * block
    ns = [1 << j for j in range(bound.bit_length()) if (1 << j) + 1 <= bound]
    if not ns:
        raise ValueError(
            f"max_num_seqs {bound} leaves no row beside a point's parked decode"
        )
    qs = sorted({*(1 << j for j in range(b.bit_length())), b})

    def whole(h: int) -> int:
        return max(block, min(h, tallest) // block * block)

    sessions = _sessions(
        random.Random(seed), served[0][1], _traces(real), max(ns), whole, 0
    )
    pool = engine["num_kvcache_blocks"] * block
    cold = {(n, q) for n in ns for q in qs if n * q <= b}
    cold |= {
        (1, t + d)
        for t in engine.get("prefill_buckets", ())
        for d in (0, 1)
        if t + d <= b
    }
    warm = {(n, q) for n in ns for q in qs if n * (q + replay) <= b}
    warm |= {(1, b - replay), (1, b - replay + 1)}
    points = [
        {"level": None, "rows": [[None, 0, q]] * n, "record": [[q, q]] * n}
        for n, q in sorted(cold)
    ]
    cached, fill_tokens = defaultdict(int), 0
    for less, level, n, q in sorted(
        (-_scale(pool, sessions[:n]), level, n, q)
        for level in range(3)
        for n, q in sorted(warm)
    ):
        heights = [whole(int(-less * s["cuts"][level])) for s in sessions[:n]]
        for i, h in enumerate(heights):
            fill_tokens += _computed(cached[i], h + 1, block, replay)
            cached[i] = max(cached[i], h + 1)
        start = [max(0, h - replay) for h in heights]
        first = [min(h + q - s, b) for h, s in zip(heights, start)]
        points.append(
            {
                "level": level,
                "rows": [[i, h, q] for i, h in enumerate(heights)],
                "record": [[c, s + c] for c, s in zip(first, start)],
            }
        )
    return {
        "seed": seed,
        "block": block,
        "hit_replay": replay,
        "budget": b,
        "levels": _levels(served[0][1]),
        "sessions": sessions,
        "points": points,
        "rows": sum(len(p["rows"]) for p in points),
        "fill_tokens": fill_tokens,
    }


def _hit_replay(real: Path, block: int) -> int:
    """The tokens a prefix hit recomputes below where it resumes on the server
    of the real record at `real`: `block`, its KV block, where the model's type
    is one of `REPLAYING` and the ATOM arguments after -- in its run.json leave
    its encoder bounded replay on (`BlockManager.replay_start`); else none."""
    argv = json.loads((real / "run.json").read_text())["argv"]
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model")
    parser.add_argument(
        "--encoder-swa-bounded-replay",
        "--enable-encoder-swa-bounded-replay",
        dest="replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    atom = parser.parse_known_args(argv[argv.index("--") + 1 :])[0]
    model = json.loads((Path(atom.model) / "config.json").read_text())
    return block if model.get("model_type") in REPLAYING and atom.replay else 0


def _served(real: Path) -> tuple[dict, dict, float, int]:
    """The real record at `real`: each DP rank's engine file; per rank, its
    decode rows' contexts and its prefill rows' cached histories, each sorted;
    the tokens a decode step adds to a request's context, on average; and the
    prompt tokens of every request it names at its final chunk, summed."""
    stated, served, contexts, prompts = {}, {}, defaultdict(list), 0
    for path in sorted((real / "steps").glob("dp*[0-9].jsonl")):
        rank = int(path.stem[2:])
        stated[rank] = json.loads(path.with_suffix(".engine.json").read_text())
        decode, history = [], []
        for step in read_jsonl(path):
            named = [request for request, *_ in step["named"]]
            for (request, _, context), (q, kv, decoding) in zip(
                step["batch"], step["rows"]
            ):
                prompts += context if request in named else 0
                if decoding:
                    decode.append(kv)
                    contexts[rank, request].append(kv)
                elif kv > q:
                    history.append(kv - q)
        served[rank] = sorted(decode), sorted(history)
    added = [b - a for kv in contexts.values() for a, b in pairwise(kv)]
    advance = float(np.mean(added)) if added else 1.0
    return stated, served, advance, prompts


def _traces(real: Path) -> dict[str, int]:
    """Each calibration trace the real record's client run was given, by id:
    its conversation's longest prompt."""
    given = sorted((real / "client" / "traces").glob("*.json"))
    if not given:
        raise ValueError(
            f"{real} holds no client/traces, the traces its client run was given"
        )
    traces = [json.loads(path.read_text()) for path in given]
    return {t["id"]: max(r["in"] for r in t["requests"] if "in" in r) for t in traces}


def _sessions(
    rng: random.Random, rows: list, traces: dict, count: int, cut, rank: int
) -> list[dict]:
    """`count` sessions, each three draws from `rows`, sorted, one a level,
    bottom first, and a trace of `traces` whose longest prompt passes the top
    one, else the longest trace, the draws held under its longest prompt and
    then `cut`."""
    if not rows:
        raise ValueError(
            f"DP rank {rank} of the real record holds no row to draw levels from"
        )
    drawn = []
    for _ in range(count):
        draws = [_draw(rng, rows, level) for level in LEVELS]
        fits = [t for t, n in traces.items() if n > draws[2]] or [
            max(traces, key=traces.get)
        ]
        trace = rng.choice(fits)
        drawn.append(
            {"trace": trace, "cuts": [cut(min(c, traces[trace] - 1)) for c in draws]}
        )
    return drawn


def _draw(rng: random.Random, rows: list, level: int) -> int:
    """A row of `rows`, sorted, drawn within `BAND` percentiles of `level`."""
    lo = len(rows) * (level - BAND) // 100
    hi = -(-len(rows) * (level + BAND) // 100)
    return rows[rng.randrange(lo, max(lo + 1, min(hi, len(rows))))]


def _levels(rows: list) -> list[float]:
    return [float(x) for x in np.percentile(rows, LEVELS)]


def _scale(pool: int, sessions: list[dict]) -> float:
    """What the cuts of `sessions` scale by, so their top ones hold at most
    `CAP` of a KV pool of `pool` tokens."""
    return min(1.0, CAP * pool / sum(s["cuts"][2] for s in sessions))


def _computed(cached: int, tokens: int, block: int, replay: int) -> int:
    """The tokens a prompt of `tokens` computes on a session that has cached
    `cached` of its tokens: from the last whole `block` the two share short of
    the prompt's last block, `replay` before it at a hit."""
    hit = min(cached // block, (tokens - 1) // block) * block
    return tokens - (max(0, hit - replay) if hit else 0)


def main(argv=None) -> None:
    args = sys.argv[1:] if argv is None else argv
    plans = {"design": design, "trace": trace_design}
    if len(args) != 3 or args[0] not in plans or not args[2].isdigit():
        sys.exit("usage: step_linear_plan design|trace REAL SEED")
    kind, real, seed = args
    print(json.dumps(plans[kind](real, int(seed))))


if __name__ == "__main__":
    main()
