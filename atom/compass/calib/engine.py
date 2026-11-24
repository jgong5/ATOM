# SPDX-License-Identifier: MIT
"""A CPU rehearsal of a design: the steps a DP engine forms from its
segments, priced by a step law and a host law (#739 section 2.4, rules 1-3).

The engine is a simplified ATOM scheduler, deterministic and fast (seconds a
design), not `compass run`: it is what a design is checked against before a
record, and its per-segment combinations are what `check` compares a record
with. What it holds:

- Segments run one after another, chained by events: a segment starts when
  every request of the one before has ended, plus `segment_overhead_s`
  (the client's start and the drain poll), never by predicted time.
- Routing: a pinned lane's request goes to its rank; an unpinned session's
  first turn to the rank with the fewest requests in flight, then fewest
  prompt tokens, then the lowest index (`ATOM_DP_LB_REQ_EQUIV=1e9`), its
  later turns to the same rank (`ATOM_DP_SESSION_AFFINITY=1`), hitting the
  prefix cache on its history in whole `prefix_unit`s.
- A step is prefill when any rank has prompt tokens waiting: such a rank
  schedules its waiting requests first come first served, up to
  `max_num_batched_tokens` and `max_num_seqs` (chunked), and its decode rows
  wait; a rank with only decode rows decodes eagerly beside it (which the step
  law prices 0 s); an idle rank runs a dummy forward (`dummy_view`).
  Otherwise every rank with rows decodes at the rung of the widest rank, an
  idle rank running a dummy forward. A decode row verifies `drafts + 1`
  tokens and emits `acceptance` tokens a step on average; it leaves once its
  output is reached.
- A step's device seconds are the max over its ranks of the law's target and
  draft price; its host seconds the host law's per-step segments at the
  costliest rank; it lasts the larger. A decode is GPU-bound when its device
  outlasts the host's step (otherwise its launch is exposed, which a fit
  cuts); a prefill when its device outlasts its host launch by more than
  `step_labels.PACED_S` (otherwise its launch paces it, which a fit cuts).

Not modelled: DP prefill cost balancing (`PrefillDelayer`), which trims a
long-context rank's chunk; KV eviction (a design stays far below the pool);
the drafter's acceptance varying around its forced mean; host intake and
tokenizer time.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field, replace

from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import BatchView, PrefillDuty, RequestShape
from atom.compass.backends.step_labels import PACED_S
from atom.compass.backends.step_linear import (
    StepLinearBackend,
    StepLinearLaw,
    dummy_view,
)
from atom.compass.calib.deploy import Config
from atom.compass.calib.design import Lane, Segment, Session

#: The router's weight of one request in flight against prompt tokens.
REQ_EQUIV = 10**9
HOST_STEP = ("E", "W_pre", "W_inputs", "W_launch", "W_post")


class Law:
    """A step law (`params` of `step_linear_law.json`, or law.json) and a host
    law (`params` of host_laws.json), pricing one rank's view."""

    def __init__(self, step_law: dict, host_law: dict | None, dp: int) -> None:
        params = step_law.get("params", step_law)
        self.params = params
        self.backend = StepLinearBackend(StepLinearLaw(params), dp=dp)
        self.host = (host_law or {}).get("params", host_law or {}).get("laws", {})
        self.refused = 0

    def price(self, view: BatchView) -> tuple[float, float]:
        """(target, draft) seconds of `view`; (0, 0) where the law refuses it."""
        try:
            cost = self.backend.price(view)
        except CostRefused:
            self.refused += 1
            return 0.0, 0.0
        return cost.target_seconds, cost.draft_seconds

    def host_s(self, kind: str, tokens: int, seqs: int, segments=HOST_STEP) -> float:
        laws = self.host.get(kind, {})
        x = (1.0, float(tokens), float(seqs))
        return sum(
            sum(c * v for c, v in zip(laws[s], x)) for s in segments if s in laws
        )


@dataclass
class Step:
    seg: str
    kind: str  # prefill or decode
    rung: int | None
    t: float
    wall: float
    device: float
    host: float
    gpu_bound: bool
    #: (rank, view, target s, draft s) of every rank that ran a forward that
    #: prices: rows, or a dummy at a prefill step
    ranks: list
    priced: int  # the index in `ranks` of the costliest rank (the fit's row)

    @property
    def view(self) -> BatchView:
        return self.ranks[self.priced][1]


@dataclass
class _Req:
    seg: int
    sess: _Sess
    turn: int
    rank: int
    prompt: int
    cached: int
    out: int
    start: float
    done: int = 0
    gen: float = 0.0

    @property
    def left(self) -> int:
        return self.prompt - self.cached - self.done


@dataclass
class _Sess:
    lane: _Lane
    session: Session
    turn: int = 0
    rank: int | None = None
    history: int = 0  # the tokens the last turn left in its rank's KV


@dataclass
class _Lane:
    lane: Lane
    queue: deque
    active: int = 0


@dataclass
class _Rank:
    waiting: deque = field(default_factory=deque)
    running: list = field(default_factory=list)
    duty: PrefillDuty = field(default_factory=PrefillDuty)
    inflight: int = 0
    tokens: int = 0


@dataclass
class SegmentRun:
    name: str
    start: float
    end: float
    requests: int


class Engine:
    def __init__(self, cfg: Config, law: Law) -> None:
        self.cfg, self.dep, self.law = cfg, cfg.deployment, law
        self.ranks = [_Rank() for _ in range(self.dep.dp)]
        self.steps: list[Step] = []
        self.runs: list[SegmentRun] = []
        self._seq = itertools.count()

    # -- arrivals -------------------------------------------------------------
    def _route(self, sess: _Sess) -> int:
        if sess.lane.lane.pin is not None:
            return sess.lane.lane.pin
        if sess.rank is None:
            sess.rank = min(
                range(self.dep.dp),
                key=lambda r: (
                    self.ranks[r].inflight * REQ_EQUIV + self.ranks[r].tokens,
                    r,
                ),
            )
        return sess.rank

    def _arrive(self, seg: int, sess: _Sess, t: float) -> None:
        turn = sess.session.turns[sess.turn]
        before = sess.rank
        rank = self._route(sess)
        cached = 0
        if sess.turn and before == rank:  # the history is in this rank's prefix cache
            cached = self.dep.cached(min(sess.history, turn.prompt - 1))
        sess.rank = rank
        req = _Req(seg, sess, sess.turn, rank, turn.prompt, cached, turn.out, t)
        r = self.ranks[rank]
        r.waiting.append(req)
        r.inflight += 1
        r.tokens += turn.prompt

    def _end(self, req: _Req, t: float, due: list) -> None:
        r = self.ranks[req.rank]
        r.inflight -= 1
        r.tokens -= req.prompt
        sess = req.sess
        sess.history = req.prompt + req.out
        sess.turn += 1
        self._open -= 1
        if sess.turn < len(sess.session.turns):
            gap = sess.session.turns[sess.turn].gap
            heapq.heappush(due, (t + gap, next(self._seq), req.seg, sess))
            self._open += 1
            return
        lane = sess.lane
        lane.active -= 1
        if lane.queue:
            nxt = _Sess(lane, lane.queue.popleft())
            lane.active += 1
            heapq.heappush(due, (t, next(self._seq), req.seg, nxt))
            self._open += 1

    # -- steps ----------------------------------------------------------------
    def _prefill_step(self, seg: Segment, t: float, due: list) -> float:
        dep = self.dep
        plan = {}
        for k, r in enumerate(self.ranks):
            if not r.waiting:
                continue
            budget, rows = dep.max_num_batched_tokens, []
            for req in r.waiting:
                if budget <= 0:
                    break
                if not req.done and len(r.running) + len(rows) >= dep.max_num_seqs:
                    break
                q = min(req.left, budget)
                rows.append((req, q))
                budget -= q
            if rows:
                plan[k] = rows
        if not plan:  # every waiting request is held by max_num_seqs: decode instead
            return self._decode_step(seg, t, due)
        tokens = {k: sum(q for _, q in rows) for k, rows in plan.items()}
        dp = dep.dp > 1
        dummies = [
            k for k, r in enumerate(self.ranks) if k not in plan and not r.running
        ]
        tail = any(q == req.left for rows in plan.values() for req, q in rows)
        moe = sum(tokens.values())  # a dummy's row reaches no expert (#3682)
        ranks = []
        for k, rows in plan.items():
            view = BatchView(
                tuple(
                    RequestShape(q, req.cached + req.done + q, False) for req, q in rows
                ),
                moe_rows=moe if dp else None,
                moe_tokens=moe if dp else None,
                produces_output=any(q == req.left for req, q in rows),
                dp_rank=k if dp else None,
                group_dummies=len(dummies),
                group_prefills=dp,
                group_max_tokens=max(tokens.values()) if dp else None,
                group_tail=tail and dp,
                prefill_duty=self.ranks[k].duty.at(t),
            )
            ranks.append((k, view, *self.law.price(view)))
        peer = ranks[0][1]
        for k in dummies:
            view = replace(
                dummy_view(peer), dp_rank=k, prefill_duty=self.ranks[k].duty.at(t)
            )
            ranks.append((k, view, *self.law.price(view)))
        priced = max(range(len(ranks)), key=lambda i: (ranks[i][2], i))
        device = max(x[2] + x[3] for x in ranks)
        top = ranks[priced][1]
        kind = "prefill" if top.produces_output else "middle"
        n_tok = sum(r.query_tokens for r in top.requests)
        host = self.law.host_s(kind, n_tok, len(top.requests))
        launch = self.law.host_s(kind, n_tok, len(top.requests), ("W_launch",))
        wall = max(device, host)
        gpu_bound = device > launch + PACED_S
        self._record(
            Step(
                seg.name,
                "prefill",
                None,
                t,
                wall,
                device,
                host,
                gpu_bound,
                ranks,
                priced,
            )
        )
        end = t + wall
        # ranks with only decode rows decode eagerly beside the prefill
        for k, r in enumerate(self.ranks):
            if k not in plan and r.running:
                self._advance(r, end, due)
        for k, rows in plan.items():
            r = self.ranks[k]
            for req, q in rows:
                finished = q == req.left
                req.done += q
                if finished:
                    r.waiting.remove(req)
                    req.gen = 1.0
                    if req.gen >= req.out:
                        self._end(req, end, due)
                    else:
                        r.running.append(req)
        self._tick(t, end, True)
        return end

    def _advance(self, r: _Rank, end: float, due: list) -> None:
        keep = []
        for req in r.running:
            req.gen += self.dep.per_step
            if req.gen >= req.out:
                self._end(req, end, due)
            else:
                keep.append(req)
        r.running = keep

    def _decode_step(self, seg: Segment, t: float, due: list) -> float:
        dep = self.dep
        widest = max(len(r.running) for r in self.ranks)
        rung = dep.rung(widest)
        dp = dep.dp > 1
        busy = [k for k, r in enumerate(self.ranks) if r.running]
        dummies = dep.dp - len(busy)
        q = dep.query
        moe = sum(len(self.ranks[k].running) for k in busy) * q  # dummies add none
        ranks = []
        for k in busy:
            rows = tuple(
                RequestShape(q, req.prompt + int(req.gen) - 1 + q, True)
                for req in self.ranks[k].running
            )
            view = BatchView(
                rows,
                capture_rung=rung,
                moe_rows=(rung or widest) * dep.dp * q if dp else None,
                moe_tokens=moe if dp else None,
                produces_output=True,
                dp_rank=k if dp else None,
                group_dummies=dummies,
                prefill_duty=self.ranks[k].duty.at(t),
            )
            ranks.append((k, view, *self.law.price(view)))
        priced = max(range(len(ranks)), key=lambda i: (ranks[i][2], i))
        device = max(x[2] + x[3] for x in ranks)
        top = ranks[priced][1]
        host = self.law.host_s(
            "decode", sum(r.query_tokens for r in top.requests), len(top.requests)
        )
        wall = max(device, host)
        self._record(
            Step(
                seg.name,
                "decode",
                rung,
                t,
                wall,
                device,
                host,
                device >= host,
                ranks,
                priced,
            )
        )
        end = t + wall
        for k in busy:
            self._advance(self.ranks[k], end, due)
        self._tick(t, end, False)
        return end

    def _tick(self, t: float, end: float, prefill: bool) -> None:
        for r in self.ranks:
            r.duty.add(t, end, prefill)

    def _record(self, step: Step) -> None:
        self.steps.append(step)

    # -- segments -------------------------------------------------------------
    def run_segment(self, i: int, seg: Segment, t: float) -> float:
        due: list = []
        self._open = 0
        start = t
        for lane in seg.lanes:
            state = _Lane(lane, deque(lane.sessions))
            for _ in range(min(lane.concurrency, len(lane.sessions))):
                sess = _Sess(state, state.queue.popleft())
                state.active += 1
                heapq.heappush(due, (t, next(self._seq), i, sess))
                self._open += 1
        n = sum(len(s.turns) for lane in seg.lanes for s in lane.sessions)
        while self._open:
            while due and due[0][0] <= t:
                _, _, _, sess = heapq.heappop(due)
                self._arrive(i, sess, t)
            if not any(r.waiting or r.running for r in self.ranks):
                t = due[0][0]
                continue
            if any(r.waiting for r in self.ranks):
                t = self._prefill_step(seg, t, due)
            else:
                t = self._decode_step(seg, t, due)
        self.runs.append(SegmentRun(seg.name, start, t, n))
        return t

    def run(self, segments: list[Segment]) -> Rehearsal:
        t = self.cfg.timing["server_start_s"]
        for i, seg in enumerate(segments):
            t = self.run_segment(i, seg, t) + self.cfg.timing["segment_overhead_s"]
        return Rehearsal(self.cfg, self.steps, self.runs, t, self.law.refused)


@dataclass
class Rehearsal:
    cfg: Config
    steps: list[Step]
    runs: list[SegmentRun]
    wall_s: float  # server start to stop
    refused: int

    @property
    def gpu_minutes(self) -> float:
        return self.wall_s / 60.0


def rehearse(cfg: Config, segments: list[Segment], law: Law) -> Rehearsal:
    return Engine(cfg, law).run(segments)
