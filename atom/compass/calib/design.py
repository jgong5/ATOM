# SPDX-License-Identifier: MIT
"""A design's segments, generated from the deployment.

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
blocks), so its history is a prefix-cache hit on that rank. A trace's hash ids
are local to it, so a session hits no other session's cache.

A segment's sizes are plain numbers or relative to the deployment, resolved
before its generator runs (`resolved`): a count (a group's rows, requests,
concurrency) as a share of `max_num_seqs`, a rank's or the group's
(`deploy.count`); a token size (a prompt, a scale, a history) as a share of
the KV pool, the model length, the step budget or the largest token-graph
bucket (`deploy.tokens`). A design stated that way follows a deployment change
with no edit.

Kinds (`GENERATORS`):

- ``drain``: decode drains. Each lane's rows start together and leave on the
  departure schedule every drain shares (`departures`): a rank holds each row
  count for the decode steps `dwell` gives it, `rung_steps` spread over the
  reachable row counts of its rung, so every rung a drain passes from its top
  holds at least `rung_steps` steps. A row's output is
  `1 + floor(acceptance x steps)` for the step it leaves at. Groups marked
  `last` leave last (an anchor stays to the end); a lane with `hold` keeps
  every row until the segment's longest drain ends (a peer that holds the rung
  or the group's tokens while another rank drains). An unpinned lane drains
  `rows / dp` rows a rank, `dp` rows leaving at each departure. A rank holds
  one lane of a segment, an unpinned lane every rank, so `max_num_seqs` bounds
  its rows; lanes sharing a rank are refused by name. A group with
  `then` sends each prompt with one output token, then a `then`-token turn on
  it that carries the drain's output: the second turns queue behind every
  first turn's chunks, so they prefill together, a step of many rows on deep
  histories, before the drain. A rank whose prefill ends before a peer's
  decodes beside it meanwhile: its schedule starts that many steps later (the
  peer's extra prefill chunks of the step budget).
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

A ``copy`` segment is the segment it names (`of`) again under its own name,
an anchor: its traces are new, so it hits none of the original's cache. A
segment stating `min_dp` is skipped on a deployment with fewer DP ranks
(`skipped`).
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from atom.compass.calib.deploy import Config, Deployment, count, tokens, whole


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


def blocks(n: int, dep: Deployment) -> int:
    """Whole trace blocks holding `n` tokens, at least one."""
    return max(1, -(-int(n) // dep.trace_block))


def session(
    name: str, first: int, follow, dep: Deployment, out0: int, role=""
) -> Session:
    """A session: `first` tokens with output `out0`, then per (gap, delta, out)
    of `follow` the previous prompt, its output and `delta`, in whole blocks.
    No turn's prompt and output pass the model length."""
    n = dep.trace_block * blocks(first, dep)
    turns = [Turn(0.0, n, int(out0))]
    for gap, delta, out in follow:
        n += dep.trace_block * (blocks(turns[-1].out, dep) + blocks(delta, dep))
        turns.append(Turn(float(gap), n, int(out)))
    for turn in turns:
        if turn.prompt + turn.out > dep.max_model_len:
            raise ValueError(
                f"session {name}: a {turn.prompt}-token prompt and its "
                f"{turn.out} output tokens pass the model length"
            )
    return Session(name, turns, role)


def spread(lo: float, hi: float, n: int, dep: Deployment) -> list[int]:
    """`n` log-uniform quantiles of [lo, hi], whole blocks, ascending."""
    if n == 1 or lo == hi:
        return [whole(math.sqrt(lo * hi), dep)] * n
    return [whole(lo * (hi / lo) ** ((i + 0.5) / n), dep) for i in range(n)]


def interleave(values: list) -> list:
    """`values` (ascending) reordered by the van der Corput sequence, so any
    run of consecutive entries spans them."""
    order = sorted(range(len(values)), key=lambda i: int(f"{i:016b}"[::-1], 2))
    out = [None] * len(values)
    for rank, i in enumerate(order):
        out[i] = values[rank]
    return out


def dwell(dep: Deployment, rung_steps: int) -> dict[int, int]:
    """The departure schedule every drain shares: the decode steps a rank
    holds at each row count, `rung_steps` spread over the row counts of its
    rung that `max_num_seqs` reaches (past the widest rung: the eager ones)."""
    out = {}
    for n in range(1, dep.max_num_seqs + 1):
        rung = dep.rung(n)
        if rung is None:
            reach = range(dep.rungs[-1] + 1, dep.max_num_seqs + 1)
        else:
            rows = dep.rows_of(rung)
            reach = range(rows.start, min(rows.stop, dep.max_num_seqs + 1))
        out[n] = math.ceil(rung_steps / len(reach))
    return out


def departures(rows: int, dep: Deployment, rung_steps: int) -> list[int]:
    """The decode step each of a rank's `rows` rows leaves at, in leaving order."""
    hold = dwell(dep, rung_steps)
    return list(itertools.accumulate(hold[n] for n in range(rows, 0, -1)))


def out_for(steps: int, dep: Deployment) -> int:
    """The output that leaves after `steps` decode steps: the prefill's token
    and `acceptance` a step, rounded down so it leaves at that step."""
    return 1 + math.floor(dep.per_step * steps)


#: The segment entries that are counts (`deploy.count`), and those that are
#: token sizes (`deploy.tokens`, a number or a list of them); a drain group's
#: `rows` and `prompt` besides.
COUNT_KEYS = (
    "rows",
    "requests",
    "concurrency",
    "short_sessions",
    "short_turns",
    "turns",
)
TOKEN_KEYS = ("prompt", "scales", "long", "short", "firsts", "deltas", "then")


def resolved(seg: dict, dep: Deployment) -> dict:
    """`seg` with every relative size a number at `dep`; a drain group's
    prompt stays, `_prompts` sizes it over its rows."""
    out = {}
    for key, value in seg.items():
        if key == "lanes":
            out[key] = [
                {**lane, "groups": [_group(g, dep) for g in lane["groups"]]}
                for lane in value
            ]
        elif key in COUNT_KEYS:
            out[key] = count(value, dep)
        elif key in TOKEN_KEYS:
            if isinstance(value, list):
                out[key] = [tokens(v, dep) for v in value]
            else:
                out[key] = tokens(value, dep)
        else:
            out[key] = value
    return out


def _group(group: dict, dep: Deployment) -> dict:
    out = {**group, "rows": count(group["rows"], dep)}
    if "then" in group:
        out["then"] = tokens(group["then"], dep)
    return out


def _prompts(group: dict, rows: int, dep: Deployment) -> list[int]:
    p = group["prompt"]
    if isinstance(p, dict):  # a share of the KV pool or the model, over the rows
        return [tokens(p, dep, rows)] * rows
    if isinstance(p, list):
        lo, hi = (tokens(x, dep, rows) for x in p)
        return interleave(spread(lo, hi, rows, dep))
    return [whole(p, dep)] * rows


def drain(seg: dict, cfg: Config) -> Segment:
    dep, steps_per_rung = cfg.deployment, cfg.acceptance["rung_steps"]
    lanes, ends, plan = [], [], {"lanes": {}}
    specs = seg["lanes"]
    names = [lane.get("name", f"l{k}") for k, lane in enumerate(specs)]
    if len(set(names)) != len(names):
        raise ValueError(f"{seg['name']}: lane names repeat: {names}")
    load = [0.0] * dep.dp  # prompt tokens each rank prefills before its drain
    held: dict[int, str] = {}  # the lane on each rank
    for k, (lane, name) in enumerate(zip(specs, names)):
        pin = lane.get("pin")
        if pin is not None and not 0 <= pin < dep.dp:
            raise ValueError(f"{seg['name']} lane {k}: pin {pin} past dp {dep.dp}")
        # An unpinned lane is on every rank, even the one of a one-rank
        # deployment; a rank holds one lane, so `max_num_seqs` bounds its rows
        # and its rows leave on the one schedule.
        ranks = [pin] if pin is not None else list(range(dep.dp))
        for r in ranks:
            if r in held:
                raise ValueError(
                    f"{seg['name']}: lanes {held[r]} and {name} share rank {r} "
                    "(an unpinned lane is on every rank); a rank holds one lane"
                )
            held[r] = name
        rows = sum(g["rows"] for g in lane["groups"])
        width = len(ranks)
        if rows % width:
            raise ValueError(f"{seg['name']} lane {k}: {rows} rows over {width} ranks")
        if rows // width > dep.max_num_seqs:
            raise ValueError(f"{seg['name']} lane {k}: past max_num_seqs a rank")
        prefill = sum(
            sum(_prompts(g, g["rows"], dep)) + g["rows"] * g.get("then", 0)
            for g in lane["groups"]
        )
        for r in ranks:
            load[r] += prefill / width
        if not lane.get("hold"):
            ends.append(departures(rows // width, dep, steps_per_rung)[-1])
    end = max(ends) if ends else departures(1, dep, steps_per_rung)[-1]
    # A rank whose prefill ends first decodes (eagerly) while a peer still
    # prefills: its rows start their schedule that many steps later, counted
    # in prefill chunks of the step budget, so the drain proper starts with
    # every rank decoding.
    chunks = [math.ceil(x / dep.max_num_batched_tokens) for x in load]
    for lane, name in zip(specs, names):
        pin = lane.get("pin")
        width = 1 if pin is not None else dep.dp
        lag = max(chunks) - (chunks[pin] if pin is not None else min(chunks))
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
                size = scale if r == busy else max(small, share * scale)
                if firsts[r] is None:
                    firsts[r] = (size, out)
                else:
                    follow[r].append((0.0, size, out))
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
        deltas = interleave(spread(lo, hi, turns, dep))
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
    prompts = interleave(spread(seg["prompt"][0], seg["prompt"][1], n, dep))
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


def _expanded(cfg: Config) -> list[dict]:
    """The config's segments, a `copy` as the segment it copies."""
    specs, out = {s["name"]: s for s in cfg.segments}, []
    for seg in cfg.segments:
        if seg["kind"] == "copy":  # the same segment again, as an anchor
            src = specs.get(seg["of"])
            if src is None:
                raise ValueError(
                    f"segment {seg['name']}: copies no segment {seg['of']}"
                )
            if src["kind"] == "copy":
                raise ValueError(f"segment {seg['name']}: copies a copy, {seg['of']}")
            purpose = seg.get("purpose", f"anchor: {seg['of']} again")
            seg = {**src, "name": seg["name"], "purpose": purpose}
        out.append(seg)
    return out


def skipped(cfg: Config) -> dict[str, int]:
    """The segments the deployment has too few DP ranks for, by name: each
    states the ranks it needs, `min_dp` (its lanes pin ranks past a smaller
    group's, or it decodes on several ranks at once)."""
    dp = cfg.deployment.dp
    return {s["name"]: s["min_dp"] for s in _expanded(cfg) if s.get("min_dp", 1) > dp}


def generate(cfg: Config) -> list[Segment]:
    """Every segment the deployment runs, generated (`skipped`: those it has
    too few DP ranks for)."""
    out, skip = [], skipped(cfg)
    for seg in _expanded(cfg):
        if seg["name"] in skip:
            continue
        if seg["kind"] not in GENERATORS:
            raise ValueError(f"segment {seg['name']}: no generator {seg['kind']!r}")
        out.append(GENERATORS[seg["kind"]](resolved(seg, cfg.deployment), cfg))
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
    the config, the segments it skips, and per segment its lanes (file,
    traces, requests, prompt and output tokens, concurrency, pin) and what its
    generator planned."""
    out.mkdir(parents=True, exist_ok=True)
    dep = cfg.deployment
    doc = {"config": cfg.raw, "skipped": skipped(cfg), "segments": []}
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


def read(directory: Path) -> list[Segment]:
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
