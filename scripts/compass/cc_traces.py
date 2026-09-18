"""Turn the cc-traces corpus into a replay trace, keeping its prefix reuse.

`semianalysisai/cc-traces-*` is an agentic corpus: one line per *session*, and a
session re-sends its whole conversation on every turn. Two things about that
shape decide whether a replay is the same workload or merely a workload with
the same histogram.

**Sub-agents nest.** A session's `requests` list mixes leaf requests
(`type` `"s"` or `"n"`) with `{"type": "subagent", "requests": [...]}` wrappers
that hold their own. A reader that iterates the top level and takes what it
finds sees 28,444 of the 68,266 requests in the 256k corpus -- it silently drops
58%, and drops them non-uniformly, because a sub-agent fan-out is exactly the
moment several requests are in the server at once. So this walks the tree.

**`hash_ids` is the workload.** Each id names a 64-token block of the prompt,
scoped to its session, and consecutive turns share their leading ids. Emitting
only `in`/`out` gives the right arrival process and the right length multiset
with none of the reuse, which on this corpus is most of the prefill work. The
ids are carried through so `replay.py` can build prompts that share blocks
exactly where the trace says they do (`atom.compass.workload.prompt_of_hash_ids`).

`in` is a block count times 64, checked on all 68,266 rows, so it is quantised
up to a block boundary: accurate in distribution, approximate per request.

    python scripts/compass/cc_traces.py \
        --traces ~/.cache/huggingface/cc-traces-256k/traces.jsonl \
        --session-min-peak-tokens 200000 --sessions 4 --out trace.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

#: Tokens per `hash_id`, fixed by the corpus (`block_size` is 64 on every
#: session, checked). Not the engine's `--block-size`, which is 16 and stays
#: there: 64 is a multiple of 16, so a shared run of whole source blocks is
#: automatically a whole number of native blocks.
BLOCK_TOKENS = 64

#: Requests shorter than this are dropped by default. Not squeamishness about
#: small numbers: take2's calibrated oracle has no hull *refusal* below its
#: sampled range -- it warns and extrapolates -- so a short prompt comes back
#: with a plausible-looking time nobody measured. 412 of 68,266 rows (0.6%) are
#: under 640 tokens. Pass --min-input-tokens 0 to keep them and read the warning.
DEFAULT_MIN_INPUT_TOKENS = 640

def _leaves(requests, counter=None, stream=0):
    """Every actual LLM request under `requests`, and which chain it belongs to.

    Nested leaves carry absolute session-clock timestamps, not offsets from
    their wrapper -- checked: no nested `t` is below its wrapper's on any of the
    1,697 wrappers -- so nothing needs rebasing here.

    The chain id matters for ordering. Golden replays a session as a set of
    concurrent *streams*: the root conversation is one, every `subagent`
    wrapper is another, and within a stream turn k+1 waits for turn k to come
    back whatever the recorded clocks say. Between streams the ordering is the
    recorded one. Without the id the two cannot be told apart, and a session
    whose root turns happen to overlap in the recording replays as a fan-out it
    never had. Root is 0; wrappers are numbered in document order, so the id is
    stable for a given session and means nothing across sessions.
    """
    if counter is None:
        counter = [0]
    for r in requests:
        if r.get("type") == "subagent":
            counter[0] += 1
            yield from _leaves(r.get("requests", ()), counter, counter[0])
        else:
            yield r, stream


def _sessions(path):
    """The corpus, one decoded session per line, with its index.

    The index is what namespaces `hash_ids`: `hash_id_scope` is `"local"`, so
    the same number in two sessions means different text, and merging them
    would invent reuse neither session had.
    """
    with open(path, encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if line:
                yield index, json.loads(line)


def extract(path, *, min_input_tokens=DEFAULT_MIN_INPUT_TOKENS,
            max_input_tokens=None, max_total_tokens=None, min_output_tokens=1,
            session_min_peak_tokens=0, max_session_span_s=None,
            sessions=None, max_requests=None, session_offset_s=0.0,
            max_requests_per_session=None):
    """`(rows, stats)` -- the replay trace, and what was left out of it.

    Every drop is counted and returned rather than logged and forgotten. A
    filtered replay is still a legitimate measurement; one that cannot say what
    it filtered is not, and the counts are what tell you whether a fall in cache
    hits is the workload or is this function.
    """
    stats = {"sessions_in_corpus": 0, "sessions_kept": 0, "leaves_in_corpus": 0,
             "dropped_short": 0, "dropped_long": 0, "dropped_total_tokens": 0,
             "dropped_no_output": 0,
             "dropped_by_session_filter": 0, "dropped_by_max_requests": 0,
             "dropped_past_session_cap": 0,
             "gaps_clipped": 0}
    rows, kept_sessions = [], 0
    for index, session in _sessions(path):
        stats["sessions_in_corpus"] += 1
        leaves = list(_leaves(session.get("requests", ())))
        stats["leaves_in_corpus"] += len(leaves)
        if session.get("block_size") != BLOCK_TOKENS:
            raise ValueError(
                f"session {session.get('id')} has block_size "
                f"{session.get('block_size')}, not {BLOCK_TOKENS}; the prompt "
                f"builder assumes 64-token blocks and would build the wrong "
                f"sharing silently")
        if session.get("hash_id_scope") != "local":
            raise ValueError(
                f"session {session.get('id')} has hash_id_scope "
                f"{session.get('hash_id_scope')!r}; this namespaces ids per "
                f"session, which is only correct for 'local'")
        peak = max((int(r["in"]) for r, _ in leaves), default=0)
        ts = [float(r["t"]) for r, _ in leaves] or [0.0]
        span = max(ts) - min(ts)
        if (peak < session_min_peak_tokens
                or (max_session_span_s is not None and span > max_session_span_s)
                or (sessions is not None and kept_sessions >= sessions)):
            stats["dropped_by_session_filter"] += len(leaves)
            continue
        kept_sessions += 1
        offset = float(session_offset_s) * (kept_sessions - 1)
        if max_requests_per_session is not None:
            # An earliest-first prefix of the conversation, not a sample of it.
            # Sessions in this corpus run from 31 to 866 requests, so a
            # closed-loop sweep that gives every client one whole session has
            # its wall clock set by the longest one; capping equalises them.
            #
            # It costs reuse, and not a little: the turns removed are the late
            # ones, which are the ones that hit. Over the first 16 sessions,
            # 96.2% of input tokens are a re-send at full length and 85.1% at a
            # cap of 16. `summarise` reports what survived, so the run's own
            # artifact states the reuse it should see.
            leaves = sorted(leaves, key=lambda rs: float(rs[0]["t"]))
            if len(leaves) > max_requests_per_session:
                stats["dropped_past_session_cap"] += (
                    len(leaves) - max_requests_per_session)
                leaves = leaves[:max_requests_per_session]
        for r, stream in leaves:
            n_in, n_out = int(r["in"]), int(r["out"])
            if n_in < min_input_tokens:
                stats["dropped_short"] += 1
                continue
            if n_out < min_output_tokens:
                # 28 rows in the corpus record out=0. A request that generates
                # nothing is a completions call the server rejects, and keeping
                # it would turn a workload property into "N failed".
                stats["dropped_no_output"] += 1
                continue
            if max_input_tokens is not None and n_in > max_input_tokens:
                stats["dropped_long"] += 1
                continue
            if max_total_tokens is not None and n_in + n_out > max_total_tokens:
                stats["dropped_total_tokens"] += 1
                continue
            row = {"arrival_s": round(float(r["t"]) + offset, 6),
                   "input_tokens": n_in,
                   "output_tokens": n_out,
                   "hash_ids": [int(h) for h in r["hash_ids"]],
                   "session": index,
                   # Which chain of the session tree this turn belongs to.
                   # 0 is the root conversation; the rest are sub-agents.
                   "stream": int(stream),
                   "session_id": session.get("id"),
                   "api_time_s": float(r["api_time"])}
            if r.get("ttft") is not None:
                # What the *source* system took, recorded for provenance. It is
                # not a target: it came off different hardware serving a
                # different model, so comparing our TTFT to it measures the gap
                # between two deployments, not this engine's error.
                row["source_ttft_s"] = float(r["ttft"])
            rows.append(row)
    stats["sessions_kept"] = kept_sessions
    rows.sort(key=lambda r: (r["arrival_s"], r["session"]))
    if max_requests is not None and len(rows) > max_requests:
        # Truncating by arrival keeps a prefix of the timeline rather than a
        # sample of it, so what survives is a real interval of the workload. It
        # does cut sessions mid-conversation, which lowers reuse -- the later
        # turns that would have hit are the ones removed.
        stats["dropped_by_max_requests"] = len(rows) - max_requests
        rows = rows[:max_requests]
    return rows, stats


def summarise(rows):
    """Shape of what came out, including the reuse it should produce."""
    if not rows:
        return {"requests": 0}
    ins = sorted(r["input_tokens"] for r in rows)
    outs = sorted(r["output_tokens"] for r in rows)

    def at(seq, q):
        return seq[min(len(seq) - 1, int(q * len(seq)))]

    # Blocks a perfect cache would serve: every id after the first time its
    # (session, id) pair is seen. This is the number verification step 4 checks
    # `cached_tokens` against -- if the engine reports far fewer, the prompts
    # are not sharing and the replay is not this workload.
    seen, shared = set(), 0
    for r in rows:
        for h in r["hash_ids"]:
            key = (r["session"], h)
            if key in seen:
                shared += 1
            else:
                seen.add(key)
    return {
        "requests": len(rows),
        "sessions": len({r["session"] for r in rows}),
        "input_tokens_total": sum(ins),
        "output_tokens_total": sum(outs),
        "input_tokens": {"min": ins[0], "p50": at(ins, 0.5), "p90": at(ins, 0.9),
                         "max": ins[-1]},
        "output_tokens": {"min": outs[0], "p50": at(outs, 0.5),
                          "p90": at(outs, 0.9), "max": outs[-1]},
        "arrival_span_s": round(rows[-1]["arrival_s"] - rows[0]["arrival_s"], 3),
        "reusable_blocks": shared,
        "reusable_tokens": shared * BLOCK_TOKENS,
        "distinct_blocks": len(seen),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", required=True, help="corpus traces.jsonl")
    p.add_argument("--out", required=True, help="replay trace JSONL to write")
    p.add_argument("--min-input-tokens", type=int, default=DEFAULT_MIN_INPUT_TOKENS)
    p.add_argument("--max-input-tokens", type=int, default=None)
    p.add_argument("--min-output-tokens", type=int, default=1,
                   help="drop rows asking for fewer output tokens than this; "
                        "28 corpus rows ask for 0, which the server refuses")
    p.add_argument("--max-total-tokens", type=int, default=None,
                   help="drop rows whose input+output exceeds the served "
                        "--max_model_len, which the server would reject anyway")
    p.add_argument("--session-min-peak-tokens", type=int, default=0,
                   help="keep only sessions whose longest request reaches this, "
                        "which is how you select the long-context end without "
                        "cutting sessions apart")
    p.add_argument("--max-session-span-s", type=float, default=None,
                   help="drop sessions whose own timeline is longer than this. "
                        "A paced run against a real engine takes the longest "
                        "session's span in wall clock -- the corpus reaches 917,239s "
                        "(10.6 days) -- so this is what bounds a real-side run "
                        "without --time-scale, which would change the queueing")
    p.add_argument("--sessions", type=int, default=None,
                   help="keep at most this many surviving sessions, whole")
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--max-requests-per-session", type=int, default=None,
                   help="keep only each session's first N requests. Sessions "
                        "here run from 31 to 866 requests, so a closed-loop "
                        "sweep that hands each client one whole session has "
                        "its wall clock set by the longest; this equalises "
                        "them. It lowers reuse, because the turns it removes "
                        "are the late ones that hit")
    p.add_argument("--session-offset-s", type=float, default=0.0,
                   help="stagger session k by k times this. Every session's "
                        "clock starts at t=0 in the corpus, so the default of "
                        "0 runs them concurrently -- which is a choice, not a "
                        "neutral reading of the trace")
    args = p.parse_args()

    rows, stats = extract(
        args.traces, min_input_tokens=args.min_input_tokens,
        max_input_tokens=args.max_input_tokens,
        max_total_tokens=args.max_total_tokens,
        min_output_tokens=args.min_output_tokens,
        session_min_peak_tokens=args.session_min_peak_tokens,
        max_session_span_s=args.max_session_span_s,
        sessions=args.sessions, max_requests=args.max_requests,
        session_offset_s=args.session_offset_s,
        max_requests_per_session=args.max_requests_per_session)

    # Made here rather than left to the caller: a missing directory
    # otherwise ends an extraction over 543MB of corpus at the last line.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = {"source": args.traces, "selection": vars(args),
            "dropped": stats, "workload": summarise(rows)}
    with open(args.out + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    print(f"{len(rows)} requests from {stats['sessions_kept']} sessions "
          f"-> {args.out}")
    print(f"  corpus: {stats['leaves_in_corpus']} leaves in "
          f"{stats['sessions_in_corpus']} sessions")
    for key in ("dropped_short", "dropped_long", "dropped_total_tokens",
                "dropped_no_output",
                "dropped_by_session_filter", "dropped_by_max_requests"):
        if stats[key]:
            print(f"  {key}: {stats[key]}")
    w = meta["workload"]
    if rows:
        print(f"  in p50/p90/max {w['input_tokens']['p50']}/"
              f"{w['input_tokens']['p90']}/{w['input_tokens']['max']}, "
              f"span {w['arrival_span_s']}s, "
              f"{w['reusable_tokens']} of {w['input_tokens_total']} input "
              f"tokens reusable "
              f"({100.0 * w['reusable_tokens'] / w['input_tokens_total']:.1f}%)")
    return 0 if rows else 2


if __name__ == "__main__":
    sys.exit(main())
