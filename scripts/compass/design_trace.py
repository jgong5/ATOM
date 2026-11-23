#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Recipe C's designed workload R_d as cc-traces (weka_trace) for `client.sh --traces`.

    design_trace.py OUT [--rows 56] [--dwell 40] [--tail 10] [--step-ms JSON]
                        [--spread 0.55,0.8,1.2,1.45] [--period 7.5] [--waves N]
                        [--s3 32] [--s3-turns 40]

Writes OUT/s0.jsonl, s1.jsonl, s2.jsonl and s3.jsonl, one trace a line as TRACES
holds them, and OUT/plan.json: each segment, in the order the server runs them, with
its traces (its concurrency and --num-conversations),
requests and S2's decode-step schedule. Every prompt is whole 64-token blocks and a
follow-up turn's hash_ids extend its previous turn's, so it hits the prefix cache.
Under aiperf's base mode with trajectory start ratios 0 a trace's first turn is sent
at the client run's start whatever its `t`; a later turn `t - t_prev` seconds after
the previous turn's response ends (no `api_time` is written), as the dry run checks.

  S0  4 lanes, one 100k request each (out 64): warm-up, left out of the fit.
  S1  1 lane, 6 sessions of 8 turns: first prompt 2k..700k, deltas 128..16k, out
      32/512, 1 s apart: lone arrivals on one busy rank.
  S2a ROWS groups of 4 lanes, one request each, a group's prompt P (40/40/20 %
      log-uniform over 8-64k, 64-256k, 256-600k) times SPREAD's member m, at most
      880k, members smallest first; a group's lanes end together. Under
      cal_record.sh's routing (a new session to the rank with the fewest requests,
      ties to the fewest prompt tokens queued) the members land one a DP rank,
      member m always on the same rank: the ranks hold one row count and four
      context mixes. A group's end takes each rank down a row, so the decode rung
      sweeps 64..1 holding each rung DWELL seconds (its decode steps at
      --step-ms), then the last group's lanes end one by one (3, 2, 1 busy ranks,
      TAIL seconds each). out = 1 + ACCEPT x decode steps to its end.
  S2b 8 groups of 4 probe lanes (first prompt 16k/96k/256k/512k, out 16) on deep
      history, sent with S2a; then waves PERIOD apart: even waves every lane, odd
      ones k = 1..4 members of one group (k ranks prefilling), each turn a delta
      of 128..32k (64k more every 4th turn of a 16k lane) and out 16/128.
  S3  S3 lanes (8 a rank), first prompt 8k..256k, then S3_TURNS turns of delta
      256..4k, out 64..1024, 0.5..3 s apart: rungs 1..8 under ever other mixes,
      as low-concurrency agentic runs hold them, which the ladder passes once.
"""

import argparse
import json
import math
from pathlib import Path

BLOCK, MODEL, ACCEPT, K, RANKS = 64, "claude-opus-4-8", 3.51, 1024, 4
RUNGS = (64, 48, 40, 32, 24, 16, 8, 4, 2, 1)  # args agg's capture sizes, widest first
#: Decode group step seconds per rung, ms: the dry run's (store_c128's law), 2026-11-19.
STEP_MS = {
    64: 48,
    48: 40,
    40: 36,
    32: 31,
    24: 28,
    16: 24,
    8: 19,
    4: 14.3,
    2: 13.9,
    1: 12,
}
MIX = ((0.4, 8 * K, 64 * K), (0.4, 64 * K, 256 * K), (0.2, 256 * K, 600 * K))
CAP = 880 * K  # a prompt, under the model's 1M context with the longest output
S1_FIRST, S1_DELTA, S1_OUT = (
    (2, 16, 64, 200, 400, 700),
    (128, K, 4 * K, 16 * K),
    (32, 512),
)
PROBE_FIRST, PROBE_OUT = (16, 96, 256, 512), (16, 128)
PROBE_DELTA = (128, 512, 2 * K, 8 * K, 32 * K)
S3_DELTA, S3_OUT, S3_GAP = (
    (256, K, 4 * K, 2 * K, 512),
    (64, 256, 1024, 128),
    (0.5, 1.5, 3.0, 1.0),
)


def blocks(n: int) -> int:
    return -(-n // BLOCK)


def trace(name: str, turns) -> dict:
    """A trace of `turns`, (t, prompt, out) each; prompt i's blocks are 0..n-1,
    so each turn's hash_ids extend the one before."""
    reqs = [
        {
            "t": round(t, 3),
            "type": "s",
            "model": MODEL,
            "in": n,
            "out": out,
            "hash_ids": list(range(n // BLOCK)),
        }
        for t, n, out in turns
    ]
    return {
        "id": name,
        "models": [MODEL],
        "block_size": BLOCK,
        "hash_id_scope": "local",
        "requests": reqs,
    }


def session(first: int, follow, out0: int) -> list:
    """(t, prompt, out) turns: `first` tokens, then per (gap, delta, out) of
    `follow` the previous prompt, its output in whole blocks and `delta`."""
    t, n, out = 0.0, BLOCK * blocks(first), out0
    turns = [(t, n, out)]
    for gap, delta, nxt in follow:
        t, n = t + gap, n + BLOCK * blocks(out) + BLOCK * blocks(delta)
        turns.append((t, n, nxt))
        out = nxt
    return turns


def prompts(groups: int) -> list[int]:
    """`MIX`'s stratified log-uniform quantiles, spread over the end order by
    the van der Corput sequence, so any run of groups spans the mix."""
    counts = [round(f * groups) for f, _, _ in MIX]
    counts[1] += groups - sum(counts)
    values = sorted(
        BLOCK * round(lo * (hi / lo) ** ((i + 0.5) / c) / BLOCK)
        for c, (_, lo, hi) in zip(counts, MIX)
        for i in range(c)
    )
    vdc = lambda i: int(f"{i:016b}"[::-1], 2)
    order = sorted(range(groups), key=vdc)
    return [values[order.index(i)] for i in range(groups)]


def ladder(rows: int, dwell: float, step_ms: dict, tail: float):
    """(per-rung decode steps, each group's end step) for ROWS groups: a row
    count n dwells its rung's steps over the counts it holds."""
    rung = {n: min(r for r in RUNGS if r >= n) for n in range(1, rows + 1)}
    held = {r: sum(1 for n in rung if rung[n] == r) for r in RUNGS}
    steps = {r: round(dwell * 1000 / step_ms[r]) for r in RUNGS if held[r]}
    ends, at = [], 0
    for n in range(rows, 0, -1):
        at += steps[rung[n]] / held[rung[n]]
        ends.append(round(at))
    lone = round(tail * 1000 / step_ms[1])
    last = [ends[-1] + j * lone for j in range(RANKS)]
    return steps, ends, last


def s2(rows, dwell, step_ms, tail, spread, period, waves):
    steps, ends, last = ladder(rows, dwell, step_ms, tail)
    lanes = []
    for g, (n, end) in enumerate(zip(prompts(rows), ends)):
        for m, f in enumerate(spread):
            stop = last[m] if g == rows - 1 else end
            size = min(CAP, BLOCK * round(n * f / BLOCK))
            lanes.append(
                trace(
                    f"rc-s2a-g{g:02d}-m{m}", [(0.0, size, 1 + math.ceil(ACCEPT * stop))]
                )
            )
    ramp = sum(dwell for r in steps) + (RANKS - 1) * tail
    waves = waves or int(ramp / period)
    for g in range(8):
        for m in range(RANKS):
            lane, mine = g * RANKS + m, []
            for w in range(waves):
                j = w // 2
                if w % 2 == 0 or (j % 8 == g and m < 1 + (j // 8) % RANKS):
                    mine.append(w)
            follow, prev, est = [], None, 0.0
            for i, w in enumerate(mine):
                delta = PROBE_DELTA[(i + lane) % len(PROBE_DELTA)]
                delta += 64 * K if g % 4 == 0 and i % 4 == 3 else 0
                out = PROBE_OUT[(w // 2 + w % 2) % 2]
                gap = 2.0 if prev is None else max(0.5, (w - prev) * period - est)
                follow.append((gap, delta, out))
                est = 0.1 + delta / 50000 + math.ceil(out / ACCEPT) * 0.03
                prev = w
            first = PROBE_FIRST[g % 4] * K
            lanes.append(
                trace(f"rc-s2b-g{g}-m{m}", session(first, follow, PROBE_OUT[0]))
            )
    plan = {
        "decode_steps": steps,
        "group_end_steps": ends,
        "last_group_end_steps": last,
        "waves": waves,
        "period": period,
        "est_ramp_s": ramp,
    }
    return lanes, plan


def s3(lanes: int, turns: int) -> list:
    """S3's sessions: lane i's first prompt the (i + 1/2)/lanes quantile of
    log-uniform 8k..256k, its turns' delta, out and gap cycling apart."""
    return [
        trace(
            f"rc-s3-{i:02d}",
            session(
                BLOCK * round(8 * K * 32 ** ((i + 0.5) / lanes) / BLOCK),
                [
                    (
                        S3_GAP[(i + j) % 4],
                        S3_DELTA[(i + j) % 5],
                        S3_OUT[(i + 2 * j) % 4],
                    )
                    for j in range(1, turns)
                ],
                S3_OUT[i % 4],
            ),
        )
        for i in range(lanes)
    ]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--rows", type=int, default=56, help="S2a lanes a DP rank (groups)")
    p.add_argument("--dwell", type=float, default=40.0, help="seconds a decode rung")
    p.add_argument(
        "--tail", type=float, default=10.0, help="seconds at 3, 2, 1 busy ranks"
    )
    p.add_argument(
        "--spread", default="0.55,0.8,1.2,1.45", help="a group's member sizes"
    )
    p.add_argument(
        "--step-ms", type=json.loads, default=None, help="{rung: ms} over STEP_MS"
    )
    p.add_argument(
        "--period", type=float, default=7.5, help="seconds between probe waves"
    )
    p.add_argument("--waves", type=int, default=0, help="probe waves (0: the ramp's)")
    p.add_argument("--s3", type=int, default=32, help="S3 lanes (0: no S3)")
    p.add_argument("--s3-turns", type=int, default=40, help="turns an S3 session")
    a = p.parse_args(argv)
    step_ms = STEP_MS | {int(k): float(v) for k, v in (a.step_ms or {}).items()}
    s1 = [
        trace(
            f"rc-s1-{i}",
            session(
                f * K,
                [(1.0, S1_DELTA[(j - 1) % 4], S1_OUT[j % 2]) for j in range(1, 8)],
                S1_OUT[0],
            ),
        )
        for i, f in enumerate(S1_FIRST)
    ]
    s0 = [
        trace(f"rc-s0-{i}", [(0.0, BLOCK * blocks(100_000), 64)]) for i in range(RANKS)
    ]
    spread = sorted(map(float, a.spread.split(",")))
    if len(spread) != RANKS:
        p.error(f"--spread names {len(spread)} sizes, one a member of {RANKS}")
    lanes, plan = s2(a.rows, a.dwell, step_ms, a.tail, spread, a.period, a.waves)
    a.out.mkdir(parents=True, exist_ok=True)
    segments = {"s0": (s0, RANKS), "s1": (s1, 1), "s2": (lanes, len(lanes))}
    if a.s3:
        segments["s3"] = (s3(a.s3, a.s3_turns), a.s3)
    for name, (traces, _) in segments.items():
        (a.out / f"{name}.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in traces)
        )
    plan |= {"step_ms": step_ms, "args": {k: str(v) for k, v in vars(a).items()}}
    plan["segments"] = {
        name: {
            "traces": len(t),
            "concurrency": c,
            "requests": sum(len(x["requests"]) for x in t),
            "prompt_tokens": sum(r["in"] for x in t for r in x["requests"]),
        }
        for name, (t, c) in segments.items()
    }
    (a.out / "plan.json").write_text(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
