"""Select the cc-traces acceptance workloads by a rule, not by hand.

The long workload this PoC has reported against so far -- `cc_pilot.jsonl`, the
first twenty requests of the corpus's first session -- is development data. The
cost model was iterated against it, so a number produced on it is a fit
statistic (`POC_STATUS.md` section 3 says so in those words). The final
acceptance therefore needs cc-traces requests that no model was developed
against, chosen in a way that cannot be steered by what the answer turns out
to be.

So the selection is a *rule with constants fixed in this file*, not a set of
command-line knobs. `emit` names a class and a corpus and gets whatever the rule
returns; there is no flag that would let a re-run quietly take an easier slice.
The rule's constants were set from the corpus's structure alone -- what a row
is, how many turns a session has, how long its inputs are, how far apart its
arrivals sit -- and from no measurement of any engine.

What a corpus row is, checked rather than assumed
-------------------------------------------------

* A session line carries `id`, `block_size` (64 everywhere), `hash_id_scope`,
  `models` and `requests`.
* A **top-level** request row's `type` is `s` (28173 rows in the whole corpus),
  `n` (271) or `subagent` (1697). **A `subagent` row is not a request**: it
  carries no `in`, no `out` and no `ttft`, only a summary of a delegated agent
  (`duration_ms`, `total_tokens`) *and a nested `requests` list*. Reading the
  wrapper itself as a request is what produced the zero-length rows in the
  first draft of this file.
* Those nested lists hold **real inference requests** -- 39822 of them across
  the corpus, `type` `n`, block-aligned like any other, and **not** duplicated
  at the top level (no nested row shares a `t` with a top-level servable row).
  Their `t` is on the **session's own clock**, not the wrapper's: for all 1697
  wrappers the first nested `t` equals the wrapper's `t` exactly. They are
  therefore concurrent inference load that a top-level-only replay would
  silently drop. This selector does not replay them -- in the trace they are a
  delegated agent's calls, to a different model, and not part of the session's
  own turn sequence -- so instead it **refuses any window or opening segment
  that overlaps one**. `nested_requests_in_scope` is an eligibility check, and
  the manifest records that it is zero for what was selected: the registered
  workloads are provably free of concealed concurrency rather than assumed to
  be.
* `in` is a **token count, in tokens**. It is block-aligned, which is not the
  same as being a block count: for all 28444 top-level servable rows in the
  corpus `in == len(hash_ids) * block_size` exactly, and the smallest is 128
  tokens. Nothing here multiplies `in` by anything. The selector re-checks that
  identity on every request it takes and refuses the session if it fails, so
  the unit is observed rather than believed.
* `t` is seconds **since that session's own first turn**. The corpus carries no
  absolute clock and no session start date, so there is no chronology across
  sessions to preserve -- a point that decides how the short class is built.

Two classes, and they are not symmetric, because the corpus is not:

* **long** -- one whole session's opening window, replayed on that session's own
  timeline. Raw inter-arrival intervals, nothing clipped, nothing compressed;
  the only transform is the common origin shift that puts the window's first
  turn at zero.
* **short** -- the corpus contains **no short-input session at all**. Across all
  393 sessions the median count of requests at or under 4096 tokens is 1, the
  longest run of such requests at the start of a session is 2, and the median
  request is 135k tokens. A short-ISL class drawn from this corpus is therefore
  necessarily a *pool of session openings*, and pooling means placing sessions
  on a timeline the corpus does not have. That placement is **declared**: every
  selected session starts at zero. It is stated as a constructed alignment, not
  offered as chronology, and it is the one invented quantity in either class.

What the corpus does and does not carry, which bounds what either class can
claim:

* No prompt text exists. Lengths are source-provided token counts and the
  driver synthesises prompts of exactly that many tokens. Content is not
  reproduced and is not claimed to be.
* `hash_ids` records which blocks a request shares with earlier ones -- the
  reuse that makes an agentic trace agentic. Prefix caching is off in every
  acceptance cell, so that reuse is deliberately not exercised: this is a test
  of step cost and scheduling under a real length and arrival process, not a
  test of prefix caching. Nothing here synthesises a shared prefix.
* `out` is what the session actually produced and is carried across
  **unchanged**. A turn that produced nothing has no TPOT to compare, so the
  rule rejects the whole session rather than raise a zero to a one: no output
  length in either workload differs from its source.
* Replay is open loop. Arrivals are the trace's, not a function of how fast the
  server answers; no think-time feedback is modelled.

    python scripts/compass/cc_traces_workload.py describe
    python scripts/compass/cc_traces_workload.py emit --class long \\
        --corpus /workspace/hf_cache/cc-traces-256k/traces.jsonl \\
        --out cc_long.jsonl --manifest cc_long.manifest.json --at 2026-09-11T00:00:00Z
    python scripts/compass/cc_traces_workload.py verify --manifest cc_long.manifest.json \\
        --corpus /workspace/hf_cache/cc-traces-256k/traces.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# the registered constants


#: The corpus these workloads are drawn from. The digest is the identity: a
#: path names a file on one machine, and the acceptance runs happen on another.
CORPUS = {
    "dataset": "semianalysisai/cc-traces-weka-062126-256k",
    "file": "traces.jsonl",
    "bytes": 568864747,
    "sha256": "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef",
    "sessions": 393,
}

#: Sessions the cost model was developed against, excluded from both classes.
#: `cc_pilot.jsonl` (sha256 bf4049f84be161df...) is the first twenty requests of
#: the first of these, and every G4 long-input result to date is on it.
DEVELOPMENT_SESSIONS = ("002001296e8a8c38ad9d7cc436d691afc602",)

#: Row types that are actual API requests. `subagent` rows are summaries and
#: carry no `in`/`out`; they are excluded by type and counted, never served.
SERVABLE_TYPES = ("s", "n")

#: The served model's context. A request above it cannot be served at all, so a
#: session containing one is not eligible -- rather than being silently pruned,
#: which would change the session's turn structure without saying so.
CONTEXT_TOKENS = 262144

#: One KV block. No servable row in the corpus is below 128 tokens, so this
#: floor never fires; it is checked on every taken request anyway, and the
#: manifest records that it fired zero times.
MIN_INPUT_TOKENS = 64


@dataclass(frozen=True)
class LongRule:
    """One whole session's opening window, on its own raw timeline.

    `window` matches the development slice's twenty requests so the two cost
    comparable GPU time; `volume_band` keeps the chosen session's window within
    a factor of the development slice's 1.68M input tokens, so the acceptance
    run is neither cheaper nor far more expensive than the run it supersedes.

    `max_window_span_s` is an **eligibility** criterion, not a transform. A
    session's turns are minutes or hours apart because a human is thinking in
    between; replaying that dead time costs wall clock on a leased GPU and
    exercises nothing. An earlier draft of this rule instead clipped every gap
    above 60 s, which is a compression of the arrival process and would have
    made the clipping constant, rather than the trace, set the spacing of a
    quarter of the arrivals. That is gone. A session is taken with its raw
    intervals or it is not taken: eligible sessions are those whose opening
    window already fits in `max_window_span_s`, and the first one in corpus
    order wins -- not the largest, not the smallest, and not one picked after
    seeing a result.

    The cost of that choice, stated rather than hidden: this class is a session
    whose twenty opening turns arrive inside fifteen minutes. Sessions with
    hour-long human pauses are out of the acceptance's scope, and the workload
    is therefore denser than the corpus median. The idle structure inside the
    window is the session's own, up to a gap of `max_window_span_s`.
    """

    kind: str = "long"
    window: int = 20
    min_input: int = MIN_INPUT_TOKENS
    max_input: int = CONTEXT_TOKENS
    volume_band: tuple = (1_000_000, 2_500_000)
    max_window_span_s: float = 900.0


@dataclass(frozen=True)
class ShortRule:
    """A pool of session *openings*, because no short session exists.

    Sessions are taken in corpus order and each contributes its **leading run**
    of requests at or under `max_input` -- turn 0, then turn 1, stopping at the
    first turn that is longer -- until `sessions` sessions have contributed. A
    whole leading run is taken or the session is skipped; a run is never cut in
    half to hit a request count.

    Leading, not "every short request anywhere in the session": a short turn an
    hour into a session is not an opening, and taking it would need an
    inter-arrival interval that the pool, not the trace, supplies.

    **The inter-session alignment is declared, and it is simultaneous**: every
    selected session starts at zero, so the pool is a cohort of `sessions`
    sessions opening at once and then following their own intervals. The corpus
    times each request relative to its own session's start and never says when a
    session began, so any pool must choose something and no choice is the
    chronology. An earlier draft spaced sessions one second apart, which reads
    as a chronology, is equally invented, and produced an arrival-bound
    ~1 req/s workload that could not tell TP1 from TP4 -- a cohort with no
    contention measures the arrival rate, not the engine. Simultaneous start is
    the honest version of an invented placement: it is obviously constructed,
    it is stated in the manifest as `session_start_at_zero`, and it puts real
    opening lengths under real concurrency. Within a session the intervals are
    the session's own, raw.
    """

    kind: str = "short"
    min_input: int = MIN_INPUT_TOKENS
    max_input: int = 4096
    sessions: int = 64
    alignment: str = "session_start_at_zero"


RULES = {"long": LongRule(), "short": ShortRule()}

#: How arrivals in each class relate to the source. Recorded in the manifest so
#: a reader of a result never has to infer it from the numbers.
ARRIVAL_MODELS = {
    "long": {
        "model": "source_paced_open_loop",
        "within_session": "raw source intervals, nothing clipped or compressed",
        "across_sessions": "not applicable, one session",
        "transform": "one common origin shift: the window's first turn at 0.0",
    },
    "short": {
        "model": "declared_session_start_alignment_open_loop",
        "within_session": "raw source intervals, nothing clipped or compressed",
        "across_sessions": "constructed: every session starts at 0.0, which the "
        "corpus does not state and which is not chronology",
        "transform": "per-session origin shift to 0.0",
    },
}


# --------------------------------------------------------------------------
# reading the corpus


class Ineligible(Exception):
    """This session cannot be taken, with the reason a reader would want."""


def sessions(path: str):
    """Sessions in corpus order.

    Yields `(index, blob, requests, wrappers, nested)`: the session's top-level
    servable requests in arrival order, its `subagent` wrapper rows, and the
    real requests nested inside those wrappers -- kept rather than thrown away,
    because a scope that overlaps one is a scope with concurrent load in it.

    The type filter happens here and nowhere else, so no later step can see a
    `subagent` summary and mistake it for a request with no tokens in it.
    """
    with open(path, encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            blob = json.loads(line)
            rows = blob.get("requests") or []
            requests = sorted(
                (r for r in rows if r.get("type") in SERVABLE_TYPES),
                key=lambda r: float(r.get("t", 0.0)),
            )
            wrappers = [r for r in rows if r.get("type") == "subagent"]
            nested = [n for w in wrappers for n in (w.get("requests") or [])]
            if requests:
                yield index, blob, requests, wrappers, nested


def scope_end(last_request) -> float:
    """When the last selected turn is finished with the server, on the source clock.

    The trace's own `api_time` for that turn: a nested request that starts
    before this instant was in flight while the selected segment was being
    served, whatever the engine here would do with it.
    """
    return float(last_request.get("t", 0.0)) + float(
        last_request.get("api_time") or 0.0
    )


def nested_in_scope(nested, first_t: float, end_t: float) -> list:
    """The subagent requests that overlap `[first_t, end_t]` on the session clock."""
    inside = []
    for request in nested:
        t = request.get("t")
        if t is None:
            continue
        if first_t <= float(t) <= end_t:
            inside.append(request)
    return inside


def checked_tokens(request, block_size: int) -> tuple:
    """(tokens, blocks) for one request, with the corpus's own identity checked.

    `in` is a token count and `hash_ids` is one entry per block, so
    `in == len(hash_ids) * block_size` must hold. It does for every servable row
    in the registered corpus. If it ever does not, the row's length means
    something other than what this file says it means, and the session is
    refused rather than served at a guessed length.
    """
    if "in" not in request or "out" not in request:
        raise Ineligible(f"a {request.get('type')!r} row carries no in/out")
    tokens = int(request["in"])
    hashes = request.get("hash_ids")
    if hashes is None:
        raise Ineligible("a request carries no hash_ids to check its length against")
    blocks = len(hashes)
    if blocks * block_size != tokens:
        raise Ineligible(f"in={tokens} is not {blocks} blocks of {block_size}")
    return tokens, blocks


def _eligible_window(requests, rule, block_size):
    """Check a run of requests against a rule; raise `Ineligible` with the why."""
    for request in requests:
        tokens, _ = checked_tokens(request, block_size)
        if tokens < rule.min_input:
            raise Ineligible(f"a request is {tokens} tokens, below one KV block")
        if tokens > rule.max_input:
            raise Ineligible(
                f"a request is {tokens} tokens, above the limit " f"{rule.max_input}"
            )
        if int(request["out"]) < 1:
            # Raising a zero to a one would be an edit to a source output
            # length. Refusing the session is not.
            raise Ineligible("a request produced no output tokens")


def _row(session_id, index_in_session, at, origin_shift, request, block_size):
    """One workload row, carrying where in the corpus it came from.

    `replay.py` reads only `arrival_s`, `input_tokens` and `output_tokens` and
    writes only those three into its artifact, so the identity of a request
    survives a run as the position it holds in this file and as this file's
    digest -- which is why the digest is what the protocol registers. The
    corpus gives a request no id of its own; `(session, request_index,
    source_t_s)` is its identity here and in the manifest.
    """
    tokens, blocks = checked_tokens(request, block_size)
    return {
        "arrival_s": round(at, 6),
        "input_tokens": tokens,
        "output_tokens": int(request["out"]),
        "session": session_id,
        "request_index": index_in_session,
        "source_t_s": float(request.get("t", 0.0)),
        "origin_shift_s": round(origin_shift, 6),
        "input_blocks": blocks,
        "model": request.get("model"),
        "api_time_s": request.get("api_time"),
        "ttft_s": request.get("ttft"),
    }


def select_long(path: str, rule: LongRule):
    """The first eligible session's opening window, on its raw timeline."""
    considered, counts = [], {"sessions": 0, "wrappers": 0, "nested": 0}
    for index, blob, requests, wrappers, nested in sessions(path):
        session_id = blob.get("id", "")
        counts["sessions"] += 1
        counts["wrappers"] += len(wrappers)
        counts["nested"] += len(nested)
        block_size = int(blob.get("block_size", 64))
        window = requests[: rule.window]
        why, overlap = None, []
        if session_id in DEVELOPMENT_SESSIONS:
            why = "development session"
        elif len(requests) < rule.window:
            why = (
                f"only {len(requests)} servable requests, fewer than the "
                f"{rule.window} window"
            )
        else:
            try:
                _eligible_window(window, rule, block_size)
            except Ineligible as bad:
                why = str(bad)
            else:
                volume = sum(int(r["in"]) for r in window)
                span = float(window[-1]["t"]) - float(window[0]["t"])
                overlap = nested_in_scope(
                    nested, float(window[0]["t"]), scope_end(window[-1])
                )
                if not (rule.volume_band[0] <= volume <= rule.volume_band[1]):
                    why = f"window volume {volume} outside the band"
                elif span > rule.max_window_span_s:
                    why = (
                        f"window spans {span:.1f}s, above "
                        f"{rule.max_window_span_s:.0f}s"
                    )
                elif overlap:
                    why = (
                        f"{len(overlap)} subagent requests run inside this "
                        f"window and this workload does not replay them"
                    )
        considered.append({"index": index, "id": session_id, "rejected": why})
        if why:
            continue
        origin = float(window[0]["t"])
        rows = [
            _row(session_id, i, float(r["t"]) - origin, -origin, r, block_size)
            for i, r in enumerate(window)
        ]
        chosen = [
            {
                "index": index,
                "id": session_id,
                "requests": len(rows),
                "input_tokens": sum(r["input_tokens"] for r in rows),
                "turns_in_session": len(requests),
                "turns_taken": len(window),
                "source_t_first_s": origin,
                "source_t_last_s": float(window[-1]["t"]),
                "scope_end_s": scope_end(window[-1]),
                "origin_shift_s": -origin,
                "starts_at_s": 0.0,
                "subagent_wrappers_in_session": len(wrappers),
                "nested_requests_in_session": len(nested),
                "nested_requests_in_scope": len(overlap),
            }
        ]
        return rows, chosen, considered, counts
    raise SystemExit("no session satisfies the long rule")


def _leading_run(requests, rule: ShortRule, block_size):
    """The session's opening turns, up to the first one that is too long.

    A run is taken whole. If any turn inside it fails a check -- a length the
    corpus contradicts, an output of zero -- the session contributes nothing,
    because dropping the offending turn would leave a hole in the session's own
    timeline that nothing records.
    """
    run = []
    for i, request in enumerate(requests):
        try:
            tokens, _ = checked_tokens(request, block_size)
        except Ineligible:
            break
        if tokens > rule.max_input:
            break
        if tokens < rule.min_input:
            raise Ineligible(f"an opening turn is {tokens} tokens")
        if int(request["out"]) < 1:
            raise Ineligible("an opening turn produced no output tokens")
        run.append((i, request))
    return run


def select_short(path: str, rule: ShortRule):
    """Session openings, pooled, every session started at zero by declaration."""
    rows, chosen, considered = [], [], []
    counts = {"sessions": 0, "wrappers": 0, "nested": 0}
    for index, blob, requests, wrappers, nested in sessions(path):
        session_id = blob.get("id", "")
        counts["sessions"] += 1
        counts["wrappers"] += len(wrappers)
        counts["nested"] += len(nested)
        block_size = int(blob.get("block_size", 64))
        if session_id in DEVELOPMENT_SESSIONS:
            considered.append(
                {"index": index, "id": session_id, "rejected": "development session"}
            )
            continue
        try:
            run = _leading_run(requests, rule, block_size)
        except Ineligible as bad:
            considered.append({"index": index, "id": session_id, "rejected": str(bad)})
            continue
        if not run:
            considered.append(
                {
                    "index": index,
                    "id": session_id,
                    "rejected": "opens above the short limit",
                }
            )
            continue
        origin = float(run[0][1]["t"])
        overlap = nested_in_scope(nested, origin, scope_end(run[-1][1]))
        if overlap:
            considered.append(
                {
                    "index": index,
                    "id": session_id,
                    "rejected": (
                        f"{len(overlap)} subagent requests run while this "
                        f"opening is being served and this workload does not "
                        f"replay them"
                    ),
                }
            )
            continue
        taken = [
            _row(session_id, i, float(r["t"]) - origin, -origin, r, block_size)
            for i, r in run
        ]
        rows.extend(taken)
        chosen.append(
            {
                "index": index,
                "id": session_id,
                "requests": len(taken),
                "input_tokens": sum(r["input_tokens"] for r in taken),
                "turns_in_session": len(requests),
                "turns_taken": len(taken),
                "source_t_first_s": origin,
                "source_t_last_s": float(run[-1][1]["t"]),
                "scope_end_s": scope_end(run[-1][1]),
                "origin_shift_s": -origin,
                "starts_at_s": 0.0,
                "subagent_wrappers_in_session": len(wrappers),
                "nested_requests_in_session": len(nested),
                "nested_requests_in_scope": 0,
            }
        )
        if len(chosen) >= rule.sessions:
            break
    if len(chosen) < rule.sessions:
        raise SystemExit(
            f"only {len(chosen)} sessions satisfy the short rule, "
            f"fewer than the {rule.sessions} it asks for"
        )
    rows.sort(key=lambda r: (r["arrival_s"], r["session"], r["request_index"]))
    return rows, chosen, considered, counts


SELECTORS = {"long": select_long, "short": select_short}


# --------------------------------------------------------------------------
# emitting and checking


def render(rows) -> str:
    """The workload file's exact bytes, so re-emission is byte-comparable."""
    return "".join(json.dumps(r) + "\n" for r in rows)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path) -> str:
    return digest_bytes(Path(path).read_bytes())


def _quantiles(values):
    ordered = sorted(values)

    def at(p):
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {
        "min": ordered[0],
        "p10": at(0.1),
        "median": at(0.5),
        "p90": at(0.9),
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def build(corpus: str, klass: str):
    """Rows and manifest body for a class, with no clock and no randomness."""
    rule = RULES[klass]
    rows, chosen, considered, counts = SELECTORS[klass](corpus, rule)
    text = render(rows)
    manifest = {
        "class": klass,
        "rule": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(rule).items()
        },
        "corpus": dict(CORPUS),
        "corpus_sha256_observed": digest_file(corpus),
        "development_sessions_excluded": list(DEVELOPMENT_SESSIONS),
        "context_tokens": CONTEXT_TOKENS,
        "servable_types": list(SERVABLE_TYPES),
        "generator_sha256": digest_file(__file__),
        "arrivals": dict(ARRIVAL_MODELS[klass]),
        "sessions": chosen,
        "sessions_rejected": [c for c in considered if c.get("rejected")][:10],
        "requests": len(rows),
        "input_tokens": _quantiles([r["input_tokens"] for r in rows]),
        "output_tokens": _quantiles([r["output_tokens"] for r in rows]),
        "arrival_span_s": rows[-1]["arrival_s"] if rows else 0.0,
        # Accounting a reader should not have to take on trust. `selection_*`
        # counts are about what was taken; `scanned_*` counts are about the
        # sessions this rule had to read before it stopped, which is not the
        # whole corpus for the short class. Neither is a corpus-wide figure:
        # those live in `corpus` and in this file's module docstring.
        "gaps_clipped": 0,
        "outputs_altered": sum(1 for r in rows if r["output_tokens"] < 1),
        "selection_nested_requests_in_scope": sum(
            s["nested_requests_in_scope"] for s in chosen
        ),
        "selection_nested_requests_in_selected_sessions": sum(
            s["nested_requests_in_session"] for s in chosen
        ),
        "scanned_sessions": counts["sessions"],
        "scanned_subagent_wrapper_rows": counts["wrappers"],
        "scanned_nested_requests": counts["nested"],
        "input_token_units": (
            "tokens; corpus `in` is already a token count and is block-aligned "
            "(in == len(hash_ids) * block_size), never multiplied here"
        ),
        "lengths_checked": "in == len(hash_ids) * block_size on every request",
        "provenance": [
            {
                "session": r["session"],
                "request_index": r["request_index"],
                "source_t_s": r["source_t_s"],
                "origin_shift_s": r["origin_shift_s"],
                "arrival_s": r["arrival_s"],
                "input_tokens": r["input_tokens"],
                "input_blocks": r["input_blocks"],
                "output_tokens": r["output_tokens"],
            }
            for r in rows
        ],
        "sha256": digest_bytes(text.encode()),
    }
    return text, manifest


def emit(args) -> int:
    text, manifest = build(args.corpus, getattr(args, "class"))
    if manifest["corpus_sha256_observed"] != CORPUS["sha256"]:
        print(
            f"ATOMCompass WARNING: this corpus is not the registered one "
            f"({manifest['corpus_sha256_observed'][:16]} against "
            f"{CORPUS['sha256'][:16]}); the workload it would produce is not "
            f"the registered workload",
            file=sys.stderr,
        )
        return 2
    manifest["emitted_at"] = args.at
    manifest["file"] = str(Path(args.out).name)
    Path(args.out).write_text(text)
    Path(args.manifest).write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        f"{manifest['class']}: {manifest['requests']} requests from "
        f"{len(manifest['sessions'])} session(s), input median "
        f"{manifest['input_tokens']['median']} max "
        f"{manifest['input_tokens']['max']}, arrivals over "
        f"{manifest['arrival_span_s']:.1f}s ({manifest['arrivals']['model']}) "
        f"-> {args.out}"
    )
    print(f"  sha256 {manifest['sha256']}")
    return 0


def verify(args) -> int:
    """Is this the registered workload, and does the rule still produce it?

    Two different questions. The digest says the file has not changed since it
    was emitted; re-emitting says the *rule* still produces those bytes, which
    is the one that catches an edit to the selection itself.
    """
    manifest = json.loads(Path(args.manifest).read_text())
    bad = []
    if args.workload or manifest.get("file"):
        path = args.workload or (Path(args.manifest).parent / manifest["file"])
        have = digest_file(path) if Path(path).exists() else None
        if have is None:
            bad.append(f"{path} does not exist")
        elif have != manifest["sha256"]:
            bad.append(
                f"{path} is not the workload this manifest describes "
                f"({have[:16]} against {manifest['sha256'][:16]})"
            )
    rule = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in asdict(RULES[manifest["class"]]).items()
    }
    if manifest.get("rule") != rule:
        bad.append(
            f"the manifest was emitted under a different selection rule "
            f"({manifest.get('rule')} against {rule})"
        )
    if manifest.get("arrivals") != ARRIVAL_MODELS[manifest["class"]]:
        bad.append(
            "the manifest declares a different arrival model than this "
            "selector builds"
        )
    if manifest.get("generator_sha256") != digest_file(__file__):
        bad.append(
            "this selector's bytes differ from the one that emitted the "
            "manifest; re-emit rather than assume the rule is unchanged"
        )
    if args.corpus:
        if digest_file(args.corpus) != manifest["corpus"]["sha256"]:
            bad.append("this corpus is not the one the manifest names")
        else:
            _text, fresh = build(args.corpus, manifest["class"])
            if fresh["sha256"] != manifest["sha256"]:
                bad.append(
                    f"re-emitting the rule gives {fresh['sha256'][:16]}, "
                    f"not the registered {manifest['sha256'][:16]}"
                )
    for reason in bad:
        print(f"  FAIL: {reason}")
    if bad:
        return 1
    print(
        f"{manifest['class']} workload {manifest['sha256'][:16]} verified"
        + (" against the corpus" if args.corpus else " by digest only")
    )
    return 0


def describe(_args) -> int:
    print(__doc__.strip().splitlines()[0])
    print(
        f"corpus: {CORPUS['dataset']} {CORPUS['sha256'][:16]} "
        f"({CORPUS['sessions']} sessions)"
    )
    print(f"excluded as development data: {', '.join(DEVELOPMENT_SESSIONS)}")
    for name, rule in RULES.items():
        print(f"{name}: {json.dumps(asdict(rule), default=list)}")
        print(f"  arrivals: {json.dumps(ARRIVAL_MODELS[name])}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument("--class", required=True, choices=sorted(RULES))
    e.add_argument("--corpus", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument(
        "--at",
        required=True,
        help="the instant of emission, UTC ISO-8601; stated rather "
        "than read, so a re-emission cannot be backdated",
    )
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--workload", default=None)
    v.add_argument(
        "--corpus", default=None, help="re-emit from this corpus and compare bytes"
    )
    sub.add_parser("describe")
    args = ap.parse_args(argv)
    return {"emit": emit, "verify": verify, "describe": describe}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
