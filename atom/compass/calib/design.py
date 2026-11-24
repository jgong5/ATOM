# SPDX-License-Identifier: MIT
"""A design's segments, generated from the deployment (#739 sections 2.1 and 3).

A segment is one or more lanes run at once, each a client run of its own: a
lane's sessions are weka traces as `client.sh --traces` sends them under
aiperf's base mode with trajectory start ratios 0 (a session's first turn at
the lane's start, a later turn `gap` seconds after the previous turn's response
ends), at the lane's concurrency. A lane with a `pin` sends every request with
`data_parallel_rank` set to it (aiperf `--extra-inputs`, which the server's
chat request takes), so its requests run on that DP rank whatever the router
does; an unpinned lane's sessions are placed by the router, which under
`ATOM_DP_LB_REQ_EQUIV=1e9` puts a new session on the rank holding the fewest
requests, and under `ATOM_DP_SESSION_AFFINITY=1` keeps a session's later turns
there: `dp` sessions started at once land one a rank. A later turn's prompt
extends the previous one's (its history, output and delta in whole trace
blocks), so its history is a prefix-cache hit on that rank.

Kinds (`GENERATORS`):

- ``drain``: decode drains. Each lane's rows start together and leave on the
  common departure schedule (`departures`): with N rows on a rank, the
  `rung_steps` decode steps a rung needs are spread over its row counts, so
  every rung the drain passes holds at least `rung_steps` group steps. A row's
  output is `1 + floor(acceptance x steps)` for the step it leaves at. Groups
  marked `last` leave last (an anchor stays to the end); a lane with `hold`
  keeps every row until the segment's longest drain ends (a peer that holds
  the rung or the group's tokens while another rank drains). An unpinned lane
  drains `rows / dp` rows a rank, `dp` rows leaving at each departure. A
  group with `then` sends each prompt with one output token, then a
  `then`-token turn on it that carries the drain's output: the second turns
  queue behind every first turn's chunks, so they prefill together, a step of
  many rows on deep histories, before the drain. A rank whose prefill ends
  before a peer's decodes beside it meanwhile: its schedule starts that many
  steps later (the peer's extra prefill chunks of the step budget).
- ``quads``: prefill quads, one session a rank: each scale and peer share a
  quad, the busiest session (turning over the ranks) prefilling `scale`
  tokens on its history and the others `share x scale`; quads alternate dense
  (one output token, the next turn at once) and sparse (`sparse_steps` decode
  steps between, which lowers every rank's prefill duty without idling).
- ``backlog``: one session back to back on long prompts growing to `long`, the
  others (`short_sessions`, one a peer rank by default) back to back on short
  deltas: an unbalanced, dense prefill (on one rank, long chunks sharing the
  step budget with short prompts).
- ``trickle``: one session a rank on a long first prompt, then short turns on
  that history (`deltas`, `outs`, `gaps` cycled): short prefills and middle
  chunks on deep histories.
- ``rows``: many short single-turn requests at high concurrency (`out1` of
  them one output token): prefill steps of many rows and every token-graph
  bucket.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from atom.compass.calib.deploy import Config, Deployment


@dataclass
class Turn:
    gap: float  # seconds after the previous turn's response ends; 0 on a first turn
    prompt: int  # the whole prompt, whole trace blocks
    out: int


@dataclass
class Session:
    name: str
    turns: list[Turn]
    role: str = ""


@dataclass
class Lane:
    name: str
    pin: int | None
    concurrency: int
    sessions: list[Session]


@dataclass
class Segment:
    name: str
    kind: str
    purpose: str
    lanes: list[Lane]
    plan: dict = field(default_factory=dict)


def blocks(tokens: int, dep: Deployment) -> int:
    """Whole trace blocks holding `tokens`, at least one."""
    return max(1, -(-int(tokens) // dep.trace_block))


def whole(tokens: float, dep: Deployment) -> int:
    return dep.trace_block * max(1, round(tokens / dep.trace_block))


def session(
    name: str, first: int, follow, dep: Deployment, out0: int, role=""
) -> Session:
    """A session: `first` tokens with output `out0`, then per (gap, delta, out)
    of `follow` the previous prompt, its output and `delta`, in whole blocks."""
    n = dep.trace_block * blocks(first, dep)
    turns, out = [Turn(0.0, n, out0)], out0
    for gap, delta, nxt in follow:
        n += dep.trace_block * (blocks(out, dep) + blocks(delta, dep))
        if n + nxt > dep.max_model_len:
            raise ValueError(
                f"session {name}: a {n}-token prompt passes the model length"
            )
        turns.append(Turn(float(gap), n, int(nxt)))
        out = nxt
    return Session(name, turns, role)


def spread(lo: float, hi: float, count: int, dep: Deployment) -> list[int]:
    """`count` log-uniform quantiles of [lo, hi], whole blocks, ascending."""
    if count == 1 or lo == hi:
        return [whole(math.sqrt(lo * hi), dep)] * count
    return [whole(lo * (hi / lo) ** ((i + 0.5) / count), dep) for i in range(count)]


def interleave(values: list, count: int) -> list:
    """`values` reordered by the van der Corput sequence, so any run spans them."""
    order = sorted(range(count), key=lambda i: int(f"{i:016b}"[::-1], 2))
    return [values[order.index(i)] for i in range(count)]


def dwell(dep: Deployment, rung_steps: int) -> dict[int, int]:
    """The common departure schedule: the decode steps a rank holds at each
    row count, `rung_steps` spread over the row counts of its rung."""
    out = {}
    for n in range(1, dep.max_num_seqs + 1):
        rung = dep.rung(n)
        rows = (
            dep.rows_of(rung)
            if rung
            else range(dep.rungs[-1] + 1, dep.max_num_seqs + 1)
        )
        out[n] = math.ceil(rung_steps / len(rows))
    return out


def departures(rows: int, dep: Deployment, rung_steps: int) -> list[int]:
    """The decode step each of a rank's `rows` rows leaves at, in leaving order."""
    hold = dwell(dep, rung_steps)
    steps, at = [], 0
    for n in range(rows, 0, -1):
        at += hold[n]
        steps.append(at)
    return steps


def out_for(steps: int, dep: Deployment) -> int:
    """The output that leaves after `steps` decode steps: the prefill's token
    and `acceptance` a step, rounded down so it leaves at that step."""
    return 1 + math.floor(dep.per_step * steps)


def _prompts(group: dict, rows: int, dep: Deployment) -> list[int]:
    p = group["prompt"]
    if isinstance(p, dict):  # a share of the rank's KV pool, split over the rows
        tokens = min(
            p["kv_share"] * dep.kv_tokens / p.get("over", rows), dep.max_model_len * 0.9
        )
        return [whole(tokens, dep)] * rows
    if isinstance(p, list):
        return interleave(spread(p[0], p[1], rows, dep), rows)
    return [whole(p, dep)] * rows


def drain(seg: dict, cfg: Config) -> Segment:
    dep, steps_per_rung = cfg.deployment, cfg.acceptance["rung_steps"]
    lanes, ends, plan = [], [], {"lanes": {}}
    specs = seg["lanes"]
    load = [0.0] * dep.dp  # prompt tokens each rank prefills before its drain
    for k, lane in enumerate(specs):
        rows = sum(g["rows"] for g in lane["groups"])
        width = 1 if lane.get("pin") is not None else dep.dp
        if rows % width:
            raise ValueError(f"{seg['name']} lane {k}: {rows} rows over {width} ranks")
        if rows // width > dep.max_num_seqs:
            raise ValueError(f"{seg['name']} lane {k}: past max_num_seqs a rank")
        tokens = sum(
            sum(_prompts(g, g["rows"], dep)) + g["rows"] * g.get("then", 0)
            for g in lane["groups"]
        )
        for r in [lane["pin"]] if width == 1 else range(dep.dp):
            load[r] += tokens / width
        if not lane.get("hold"):
            ends.append(departures(rows // width, dep, steps_per_rung)[-1])
    end = max(ends) if ends else departures(1, dep, steps_per_rung)[-1]
    # A rank whose prefill ends first decodes (eagerly) while a peer still
    # prefills: its rows start their schedule that many steps later, counted
    # in prefill chunks of the step budget, so the drain proper starts with
    # every rank decoding.
    chunks = [math.ceil(x / dep.max_num_batched_tokens) for x in load]
    for k, lane in enumerate(specs):
        pin = lane.get("pin")
        width = 1 if pin is not None else dep.dp
        lag = max(chunks) - (chunks[pin] if width == 1 else min(chunks))
        # leaving order: groups as listed, `last` ones after the rest
        order = [g for g in lane["groups"] if not g.get("last")]
        order += [g for g in lane["groups"] if g.get("last")]
        rows = sum(g["rows"] for g in order)
        leave = (
            [end] * (rows // width)
            if lane.get("hold")
            else departures(rows // width, dep, steps_per_rung)
        )
        leave = [lag + x for x in leave]
        sessions, j = [], 0
        for g, group in enumerate(order):
            for p in _prompts(group, group["rows"], dep):
                at = leave[j // width]
                role = group.get("role", "anchor" if group.get("last") else "row")
                out = out_for(at, dep)
                if group.get("then"):  # one token, then a short turn on it
                    first, follow, out0 = p, [(0.0, group["then"], out)], 1
                else:
                    first, follow, out0 = p, [], out
                sessions.append(
                    session(f"g{g}r{j:03d}", first, follow, dep, out0, role)
                )
                j += 1
        name = lane.get("name", f"l{k}")
        lanes.append(Lane(name, pin, len(sessions), sessions))
        plan["lanes"][name] = {
            "pin": pin,
            "rows_a_rank": rows // width,
            "leave_steps": sorted(set(leave)),
            "hold": bool(lane.get("hold")),
            "lag_steps": lag,
        }
    plan["drain_steps"] = end
    plan["prefill_chunks"] = chunks
    return Segment(seg["name"], "drain", seg.get("purpose", ""), lanes, plan)


def quads(seg: dict, cfg: Config) -> Segment:
    dep = cfg.deployment
    scales, shares = seg["scales"], seg["shares"]
    sparse = seg.get("sparse_steps", [0])
    small = dep.trace_block * seg.get("floor_blocks", 1)
    follow = [[] for _ in range(dep.dp)]
    firsts, levels = [None] * dep.dp, []
    q = 0
    for scale in scales:
        for share in shares:
            busy = q % dep.dp
            steps = sparse[q % len(sparse)]
            out = out_for(steps, dep) if steps else 1
            for r in range(dep.dp):
                tokens = scale if r == busy else max(small, share * scale)
                if firsts[r] is None:
                    firsts[r] = (tokens, out)
                else:
                    follow[r].append((0.0, tokens, out))
            levels.append({"scale": scale, "share": share, "decode_steps": steps})
            q += 1
    sessions = [
        session(f"q{r}", firsts[r][0], follow[r], dep, firsts[r][1], "quad")
        for r in range(dep.dp)
    ]
    lane = Lane("l0", None, dep.dp, sessions)
    plan = {"quads": levels}
    return Segment(seg["name"], "quads", seg.get("purpose", ""), [lane], plan)


def backlog(seg: dict, cfg: Config) -> Segment:
    dep = cfg.deployment
    long = seg["long"]
    follow = [(0.0, b - a, 1) for a, b in itertools.pairwise(long)]
    sessions = [session("long", long[0], follow, dep, 1, "long")]
    lo, hi = seg["short"]
    turns = seg["short_turns"]
    for r in range(1, 1 + seg.get("short_sessions", dep.dp - 1)):
        deltas = interleave(spread(lo, hi, turns, dep), turns)
        deltas = deltas[r % turns :] + deltas[: r % turns]
        sessions.append(
            session(
                f"short{r}",
                deltas[0],
                [(0.0, d, 1) for d in deltas[1:]],
                dep,
                1,
                "short",
            )
        )
    lane = Lane("l0", None, len(sessions), sessions)
    plan = {"long": long, "short": [lo, hi], "short_turns": turns}
    return Segment(seg["name"], "backlog", seg.get("purpose", ""), [lane], plan)


def trickle(seg: dict, cfg: Config) -> Segment:
    dep = cfg.deployment
    deltas, outs, gaps = seg["deltas"], seg["outs"], seg["gaps"]
    sessions = []
    for r, first in enumerate(seg["firsts"]):
        follow = [
            (
                gaps[(r + j) % len(gaps)],
                deltas[(r + 2 * j) % len(deltas)],
                outs[(r + j) % len(outs)],
            )
            for j in range(1, seg["turns"])
        ]
        sessions.append(
            session(f"t{r}", first, follow, dep, outs[r % len(outs)], "trickle")
        )
    lane = Lane("l0", None, len(sessions), sessions)
    plan = {k: seg[k] for k in ("firsts", "deltas", "outs", "gaps", "turns")}
    return Segment(seg["name"], "trickle", seg.get("purpose", ""), [lane], plan)


def rows(seg: dict, cfg: Config) -> Segment:
    dep = cfg.deployment
    n = seg["requests"]
    prompts = interleave(spread(seg["prompt"][0], seg["prompt"][1], n, dep), n)
    outs = seg.get("outs", [1, 8, 32])
    tail = [o for o in outs if o > 1] or [1]
    sessions = []
    for i, p in enumerate(prompts):
        one = (i * 997) % 1000 < seg.get("out1", 0.8) * 1000
        out = 1 if one else tail[i % len(tail)]
        sessions.append(session(f"r{i:05d}", p, [], dep, out, "row"))
    lane = Lane("l0", None, seg["concurrency"], sessions)
    plan = {"requests": n, "prompt": seg["prompt"], "out1": seg.get("out1", 0.8)}
    return Segment(seg["name"], "rows", seg.get("purpose", ""), [lane], plan)


GENERATORS = {
    "drain": drain,
    "quads": quads,
    "backlog": backlog,
    "trickle": trickle,
    "rows": rows,
}


def generate(cfg: Config) -> list[Segment]:
    out = []
    specs = {s["name"]: s for s in cfg.segments}
    for seg in cfg.segments:
        if seg["kind"] == "copy":  # the same segment again, as an anchor
            src = specs[seg["of"]]
            if src["kind"] == "copy":
                raise ValueError(f"segment {seg['name']}: copies a copy, {seg['of']}")
            purpose = seg.get("purpose", f"anchor: {seg['of']} again")
            seg = {**src, "name": seg["name"], "purpose": purpose}
        if seg["kind"] not in GENERATORS:
            raise ValueError(f"segment {seg['name']}: no generator {seg['kind']!r}")
        made = GENERATORS[seg["kind"]](seg, cfg)
        for lane in made.lanes:
            if lane.pin is not None and not 0 <= lane.pin < cfg.deployment.dp:
                raise ValueError(f"{made.name} {lane.name}: pin {lane.pin} past dp")
        out.append(made)
    return out


def trace(seg: Segment, lane: Lane, s: Session, dep: Deployment) -> dict:
    """`s` as a weka trace: turn t's hash ids extend turn t-1's, so a later turn
    hits the prefix cache on the rank its session stays on."""
    t, reqs = 0.0, []
    for turn in s.turns:
        t += turn.gap
        reqs.append(
            {
                "t": round(t, 3),
                "type": "s",
                "model": dep.trace_model,
                "in": turn.prompt,
                "out": turn.out,
                "hash_ids": list(range(turn.prompt // dep.trace_block)),
            }
        )
    return {
        "id": f"{seg.name}.{lane.name}.{s.name}",
        "models": [dep.trace_model],
        "block_size": dep.trace_block,
        "hash_id_scope": "local",
        "requests": reqs,
    }


def lane_file(seg: Segment, lane: Lane) -> str:
    return f"{seg.name}.{lane.name}.jsonl"


def write(segments: list[Segment], cfg: Config, out: Path) -> dict:
    """Each lane's traces as OUT/<segment>.<lane>.jsonl, and OUT/design.json:
    the config, and per segment its lanes (file, traces, requests, prompt
    tokens, concurrency, pin) and what its generator planned."""
    out.mkdir(parents=True, exist_ok=True)
    dep = cfg.deployment
    doc = {"config": cfg.raw, "segments": []}
    for seg in segments:
        lanes = []
        for lane in seg.lanes:
            traces = [trace(seg, lane, s, dep) for s in lane.sessions]
            (out / lane_file(seg, lane)).write_text(
                "".join(json.dumps(t) + "\n" for t in traces)
            )
            lanes.append(
                {
                    "name": lane.name,
                    "file": lane_file(seg, lane),
                    "pin": lane.pin,
                    "concurrency": lane.concurrency,
                    "traces": len(traces),
                    "requests": sum(len(t["requests"]) for t in traces),
                    "prompt_tokens": sum(
                        r["in"] for t in traces for r in t["requests"]
                    ),
                    "output_tokens": sum(
                        r["out"] for t in traces for r in t["requests"]
                    ),
                }
            )
        doc["segments"].append(
            {
                "name": seg.name,
                "kind": seg.kind,
                "purpose": seg.purpose,
                "lanes": lanes,
                "plan": seg.plan,
            }
        )
    (out / "design.json").write_text(json.dumps(doc, indent=1))
    return doc


def read(directory: Path, cfg: Config) -> list[Segment]:
    """The segments `write` wrote to `directory`, back from its traces."""
    doc = json.loads((Path(directory) / "design.json").read_text())
    segments = []
    for seg in doc["segments"]:
        lanes = []
        for lane in seg["lanes"]:
            sessions = []
            for line in (Path(directory) / lane["file"]).read_text().splitlines():
                t = json.loads(line)
                prev, turns = 0.0, []
                for r in t["requests"]:
                    turns.append(Turn(round(r["t"] - prev, 3), r["in"], r["out"]))
                    prev = r["t"]
                sessions.append(Session(t["id"].rsplit(".", 1)[1], turns))
            lanes.append(Lane(lane["name"], lane["pin"], lane["concurrency"], sessions))
        segments.append(
            Segment(seg["name"], seg["kind"], seg["purpose"], lanes, seg["plan"])
        )
    return segments
