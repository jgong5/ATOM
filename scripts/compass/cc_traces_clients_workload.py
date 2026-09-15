"""Select the cc-traces *client* workloads: whole busy episodes, descendants and all.

`cc_traces_workload.py` registered two classes that deliberately refuse any
window overlapping a subagent -- `selection_nested_requests_in_scope: 0` in both
manifests. That was the right call for what it was measuring and it is the
wrong corpus slice for the question this matrix asks, which is what happens to
a served width when one top-level session has several branches in flight at
once. Those registrations and their evidence stand unchanged; this is a second,
separately versioned rule beside them, and it does not touch their files.

A **client** is one top-level root session. It is not a cap on in-flight
requests: a root can and does have several descendants running at the same
instant, and every eligible descendant is offered to the server.

What an episode is
------------------

Take every API leaf in a session's tree -- the root's own `s`/`n` turns and the
`n` rows nested inside `subagent` wrappers -- as the interval
`[t, t + api_time)` on the root's clock. A `subagent` row is not a request: it
carries no `in`/`out`, only a summary and the nested list, so it contributes
ancestry and no load. An **episode** is a maximal run of those intervals
connected by gaps of no more than `EPISODE_GAP_S`, which is zero: an episode is
a complete busy period of the root, with no idle instant inside it and no idle
time replayed. That bound comes from the session's own structure rather than
from a wall-clock budget, and it is what keeps a many-hour session out of a
leased GPU hour without compressing anything that is inside.

Episodes are cut from **all** leaves and then accepted or rejected entire. An
earlier draft dropped unservable leaves first and cut episodes from what was
left, which splices two bursts the source separated and shortens a branch
without saying so. An episode holding a request this engine cannot serve -- a
zero-output turn, a length that does not match its blocks, a prompt above the
context -- is refused whole, and the count is in the manifest.

What the corpus documents, and what it does not
-----------------------------------------------

The dataset card for `semianalysisai/cc-traces-weka-062126-256k` documents the
timeline and nothing about causality. Under "Timeline preservation" it says
surviving request `t` values "keep their original relative offsets, including
sub-agent overlap", and that if the first request was filtered "all surviving
timestamps are shifted by one uniform offset so the earliest survivor starts at
`t = 0`". Its stats block gives `traces: 393`, `main_turns: 28,444`,
`subagent_groups: 1,697`, `subagent_inner_requests: 39,822` -- all four of
which this file's own pass over the corpus reproduces exactly.

The card does **not** define `api_time`, and `think_time` does not appear on it
at all. So there is no documented contract that a child was issued because a
parent finished, and none is invented here: the replay stays **open-loop**,
every request firing at its recorded offset whether or not anything it might
depend on has completed. Source `api_time` is used for two things only --
cutting episodes, and reporting the overlap the source exhibits as evidence
that the slice contains real concurrency. It is not a latency reference. The
TTFT and TPOT this acceptance grades are measured on the real Qwen server and
on the modelled one, for every replayed request including descendants.

Identity
--------

A nested row carries no id of its own, no parent id and no root id -- only a
`model`. A request's identity here is therefore `(root session, JSON path from
that session, actor ancestry)`: `/requests/5/requests/1` under `agent_id`
`subagent_001`. Paths are unique within a root, so ordering by
`(arrival, client index, path)` is total and the flatten has no tie to break by
chance. The source `model` label is carried on every row as provenance and
**every request is replayed against the one PoC target model**; a descendant is
never dropped for having been served by something else in the trace.

Alignment
---------

Each selected episode is shifted so its own first arrival is 0, and all C roots
start together. The corpus has no absolute start times, so any placement of
independent roots is constructed; this one extends the registered
`session_start_at_zero` declaration from one session to C. It is stated as a
construction and is not offered as chronology.

The pool is **fixed and ordered**: eight roots per class, in corpus order, and
client count C takes the first C of them. So `clients=1` is a subset of
`clients=2` is a subset of `clients=4` is a subset of `clients=8`, and the four
cells of a row differ in the number of concurrent roots and in nothing else.

    python scripts/compass/cc_traces_clients_workload.py describe
    python scripts/compass/cc_traces_clients_workload.py volumes --corpus traces.jsonl
    python scripts/compass/cc_traces_clients_workload.py emit --class clients_short \\
        --clients 4 --corpus traces.jsonl --out cc_traces_clients_short_c4.jsonl \\
        --manifest cc_traces_clients_short_c4.manifest.json --at 2026-09-12
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """A sibling script as a module, the way the other cc-traces scripts do it."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: The first rule's registered constants are reused rather than restated: one
#: corpus identity, one development exclusion, one servable-type set. A second
#: copy of any of them is a second thing to keep in step.
base = _load("cc_traces_workload")

CORPUS = base.CORPUS
DEVELOPMENT_SESSIONS = base.DEVELOPMENT_SESSIONS
SERVABLE_TYPES = base.SERVABLE_TYPES
CONTEXT_TOKENS = base.CONTEXT_TOKENS
MIN_INPUT_TOKENS = base.MIN_INPUT_TOKENS
Ineligible = base.Ineligible
checked_tokens = base.checked_tokens
render = base.render
digest_bytes = base.digest_bytes
digest_file = base.digest_file

#: The model every row is replayed against, whatever the trace says served it.
TARGET_MODEL = "Qwen/Qwen3.8-27B"

#: Zero. An episode is a complete busy period: joined only where the source
#: leaves no idle instant. Any larger tolerance would be this file choosing how
#: much dead time to replay, and the constant rather than the trace would then
#: set where an episode ends.
EPISODE_GAP_S = 0.0

#: Client counts, and the size of the ordered pool each class draws from. The
#: counts are nested by construction, so the pool is exactly `max(CLIENT_COUNTS)`
#: roots and cell `c=4` is cell `c=8`'s first four roots.
CLIENT_COUNTS = (1, 2, 4, 8)
POOL_SIZE = max(CLIENT_COUNTS)


@dataclass(frozen=True)
class ClientsShortRule:
    """Short prompts, kept to the registered meaning of short.

    The first registration's `short` class is 4096-token session openings, and
    that meaning is preserved here rather than redefined: every prompt in a
    selected episode is at or under `max_input`.

    What that costs, stated plainly because the manifest cannot infer it: in
    this corpus a concurrent episode of short prompts is **always** a pair or
    triple of siblings inside a single subagent branch. Every qualifying
    episode has no root turn in it at all and exactly one branch. So this class
    exercises small-prompt requests arriving together and being served
    together; it does not exercise a root turn overlapping a delegate, and it
    does not exercise multi-branch fan-out. The large-prompt class does both.
    """

    kind: str = "clients_short"
    gap_s: float = EPISODE_GAP_S
    min_input: int = MIN_INPUT_TOKENS
    max_input: int = 4096
    min_peak: int = 2
    min_descendants: int = 1


@dataclass(frozen=True)
class ClientsLargeRule:
    """Large prompts under real descendant concurrency, bounded to a burst.

    Not the first registration's `long` volume band. That band is 1.0-2.5M
    input tokens for one session's twenty opening turns, and eight roots of it
    is 11M tokens of prefill in one cell -- a cost this PoC cannot spend four
    times a row. This class instead keeps the *burst* structure: one complete
    busy episode per root, `max_span_s` seconds of arrivals, at most
    `max_total_input` tokens from that root, and at least one prompt of
    `min_large_input` tokens so the class is genuinely about long prefill. It
    is a new pre-registered rule with its own name, not a re-registration of
    `long`, whose manifest and evidence are untouched.

    `min_peak` and `min_descendants` are what make the class about clients at
    all: an episode with no overlap, or with no subagent request in it, cannot
    show what happens when one session has several calls outstanding.
    """

    kind: str = "clients_large"
    gap_s: float = EPISODE_GAP_S
    min_input: int = MIN_INPUT_TOKENS
    max_input: int = CONTEXT_TOKENS
    max_span_s: float = 60.0
    max_total_input: int = 400_000
    min_large_input: int = 32768
    min_peak: int = 2
    min_descendants: int = 1


RULES = {
    "clients_short": ClientsShortRule(),
    "clients_large": ClientsLargeRule(),
}

ARRIVALS = {
    "model": "source_paced_open_loop",
    "within_root": (
        "raw source offsets on the root's own clock, descendants included; "
        "nothing clipped, compressed or reordered"
    ),
    "across_roots": (
        "declared: every selected episode's first arrival at 0.0, so the C "
        "roots start together. The corpus carries no absolute start time, so "
        "this is a construction and not a chronology"
    ),
    "causality": (
        "none assumed. The dataset card defines no completion dependency and "
        "does not define api_time or think_time at all, so a descendant fires "
        "at its recorded offset whether or not its parent turn has finished"
    ),
}


# --------------------------------------------------------------------------
# the tree


def leaves(blob) -> list:
    """Every API leaf of one session, with its path and actor ancestry.

    Depth is one: the corpus has no wrapper inside a wrapper, checked over all
    393 sessions. A `subagent` row is ancestry, never load.
    """
    out = []
    for i, row in enumerate(blob.get("requests") or []):
        kind = row.get("type")
        if kind in SERVABLE_TYPES:
            out.append(
                {
                    "path": f"/requests/{i}",
                    "actor": "root",
                    "agent_id": None,
                    "subagent_type": None,
                    "row": row,
                }
            )
        elif kind == "subagent":
            for j, kid in enumerate(row.get("requests") or []):
                if kid.get("type") in SERVABLE_TYPES:
                    out.append(
                        {
                            "path": f"/requests/{i}/requests/{j}",
                            "actor": "subagent",
                            "agent_id": row.get("agent_id"),
                            "subagent_type": row.get("subagent_type"),
                            "row": kid,
                        }
                    )
    for leaf in out:
        leaf["t"] = float(leaf["row"].get("t", 0.0))
        leaf["end"] = leaf["t"] + float(leaf["row"].get("api_time") or 0.0)
    out.sort(key=lambda x: (x["t"], x["path"]))
    return out


def busy_episodes(items, gap: float) -> list:
    """Maximal runs of leaves connected by gaps of at most `gap`."""
    if not items:
        return []
    runs, current, reach = [], [items[0]], items[0]["end"]
    for leaf in items[1:]:
        if leaf["t"] <= reach + gap:
            current.append(leaf)
            reach = max(reach, leaf["end"])
        else:
            runs.append(current)
            current, reach = [leaf], leaf["end"]
    runs.append(current)
    return runs


def peak_overlap(run) -> int:
    """The most requests the source had in flight at once inside this episode.

    Diagnostic only. It says the slice contains real concurrency; it is not a
    bound on what the served engine will have outstanding, because Qwen's
    service times are not the trace's and the same arrivals can queue past
    `max_num_seqs` or never reach it.
    """
    events = []
    for leaf in run:
        events.append((leaf["t"], 1))
        events.append((leaf["end"], -1))
    events.sort()
    live = peak = 0
    for _, delta in events:
        live += delta
        peak = max(peak, live)
    return peak


class Rejected(Ineligible):
    """Why an episode was refused: a stable category, and what tripped it.

    The category carries no interpolated measurement. A reason string that
    embeds the offending token count gives every episode its own census key,
    which turns a census into a transcript -- so the number travels separately
    in `value`, and the census reports the range of values seen per category.
    """

    def __init__(self, category, value=None):
        super().__init__(category)
        self.category = category
        self.value = value


def _servable(run, rule, block_size):
    """Raise `Rejected` if any request in this episode cannot be served."""
    for leaf in run:
        tokens, _ = checked_tokens(leaf["row"], block_size)
        if tokens < rule.min_input:
            raise Rejected(
                f"a request is below the {rule.min_input}-token floor "
                f"of one KV block",
                tokens,
            )
        if tokens > rule.max_input:
            raise Rejected(
                f"a request is above the {rule.max_input}-token prompt limit",
                tokens,
            )
        if int(leaf["row"]["out"]) < 1:
            raise Rejected("a request produced no output tokens")


def _wanted(run, rule) -> None:
    """Raise `Rejected` unless this episode is what the class is about."""
    descendants = sum(1 for x in run if x["actor"] == "subagent")
    if descendants < rule.min_descendants:
        raise Rejected("no subagent request in the episode", descendants)
    peak = peak_overlap(run)
    if peak < rule.min_peak:
        raise Rejected(
            f"the source never has {rule.min_peak} of these in flight", peak
        )
    if isinstance(rule, ClientsLargeRule):
        span = max(x["t"] for x in run) - min(x["t"] for x in run)
        if span > rule.max_span_s:
            raise Rejected(
                f"arrivals span more than {rule.max_span_s}s", round(span, 1)
            )
        total = sum(int(x["row"]["in"]) for x in run)
        if total > rule.max_total_input:
            raise Rejected(
                f"more than {rule.max_total_input} input tokens in the episode",
                total,
            )
        largest = max(int(x["row"]["in"]) for x in run)
        if largest < rule.min_large_input:
            raise Rejected(
                f"no prompt reaches {rule.min_large_input} tokens, so this is "
                f"not a large-prompt episode",
                largest,
            )


def _episode_record(line, blob, run) -> dict:
    """What the manifest says about one selected root's episode."""
    starts = [x["t"] for x in run]
    descendants = [x for x in run if x["actor"] == "subagent"]
    return {
        "corpus_line_1based": line,
        "corpus_index_0based": line - 1,
        "id": blob.get("id"),
        "requests": len(run),
        "root_requests": len(run) - len(descendants),
        "descendant_requests": len(descendants),
        "branches": sorted({x["agent_id"] for x in descendants if x["agent_id"]}),
        "subagent_types": sorted(
            {x["subagent_type"] for x in descendants if x["subagent_type"]}
        ),
        "source_peak_overlap": peak_overlap(run),
        "source_t_first_s": min(starts),
        "source_t_last_s": max(starts),
        "arrival_span_s": round(max(starts) - min(starts), 6),
        "input_tokens": sum(int(x["row"]["in"]) for x in run),
        "output_tokens": sum(int(x["row"]["out"]) for x in run),
        "max_input_tokens": max(int(x["row"]["in"]) for x in run),
        "source_models": dict(
            sorted(Counter(str(x["row"].get("model")) for x in run).items())
        ),
        "paths": [x["path"] for x in run],
    }


def select_pool(corpus: str, rule) -> tuple:
    """The class's fixed ordered pool of `POOL_SIZE` roots, in corpus order.

    One episode per root -- its first qualifying one -- so a root contributes
    one contiguous burst and no session is represented twice.
    """
    pool, rejected = [], Counter()
    values: dict = {}
    counts = Counter()
    with open(corpus, encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            counts["sessions"] += 1
            blob = json.loads(line)
            rows = blob.get("requests") or []
            wrappers = [r for r in rows if r.get("type") == "subagent"]
            counts["wrappers"] += len(wrappers)
            counts["nested"] += sum(len(w.get("requests") or []) for w in wrappers)
            if blob.get("id") in DEVELOPMENT_SESSIONS:
                rejected["a development session"] += 1
                continue
            if len(pool) >= POOL_SIZE:
                continue
            block_size = int(blob.get("block_size") or 0)
            for run in busy_episodes(leaves(blob), rule.gap_s):
                try:
                    _servable(run, rule, block_size)
                    _wanted(run, rule)
                except Ineligible as why:
                    # `checked_tokens` raises the base class, so catch that and
                    # read a category off it only when one was supplied.
                    category = getattr(why, "category", None) or str(why)
                    value = getattr(why, "value", None)
                    rejected[category] += 1
                    if value is not None:
                        seen = values.get(category)
                        values[category] = (
                            (value, value)
                            if seen is None
                            else (min(seen[0], value), max(seen[1], value))
                        )
                    continue
                pool.append((index + 1, blob, run))
                break
    if len(pool) < POOL_SIZE:
        raise SystemExit(
            f"{rule.kind}: only {len(pool)} of the {POOL_SIZE} roots this class "
            f"needs qualify in the registered corpus, so client counts "
            f"{CLIENT_COUNTS} cannot all be built from one nested pool. This is "
            f"reported rather than worked around: changing a constant to reach "
            f"eight would be choosing the slice after seeing what it costs."
        )
    return pool, rejected, values, counts


# --------------------------------------------------------------------------
# the workload file


def _row(client_index: int, episode, leaf, origin_shift: float, block_size: int):
    """One workload row: what is served, and exactly where it came from.

    `replay.py` reads `arrival_s`, `input_tokens` and `output_tokens`. The rest
    is the provenance that lets a row be traced back to a corpus record without
    the corpus -- which matters more here than in the first registration,
    because a descendant has no id in the source and its position in a
    flattened file is otherwise all it has.
    """
    tokens, blocks = checked_tokens(leaf["row"], block_size)
    return {
        "arrival_s": round(leaf["t"] - origin_shift, 6),
        "input_tokens": tokens,
        "output_tokens": int(leaf["row"]["out"]),
        "client_index": client_index,
        "session": episode["id"],
        "corpus_line_1based": episode["corpus_line_1based"],
        "json_path": leaf["path"],
        "actor": leaf["actor"],
        "agent_id": leaf["agent_id"],
        "subagent_type": leaf["subagent_type"],
        "source_t_s": leaf["t"],
        "origin_shift_s": round(origin_shift, 6),
        "input_blocks": blocks,
        # The label the trace was served by, kept as provenance. Every row is
        # replayed against TARGET_MODEL regardless; see `target_model` in the
        # manifest.
        "source_model": leaf["row"].get("model"),
        "api_time_s": leaf["row"].get("api_time"),
        # Present on top-level `s` rows and absent on every nested row. It is
        # carried where it exists and is never a gate input: the acceptance
        # compares the real Qwen server against the modelled one.
        "source_ttft_s": leaf["row"].get("ttft"),
    }


def build(corpus: str, klass: str, clients: int):
    """Rows and manifest body for one (class, client count). No clock, no randomness."""
    if klass not in RULES:
        raise SystemExit(f"unknown class {klass!r}; this file registers {sorted(RULES)}")
    if clients not in CLIENT_COUNTS:
        raise SystemExit(f"clients must be one of {CLIENT_COUNTS}, not {clients!r}")
    rule = RULES[klass]
    pool, rejected, rejected_values, counts = select_pool(corpus, rule)

    rows, episodes = [], []
    for client_index, (line, blob, run) in enumerate(pool):
        record = _episode_record(line, blob, run)
        episodes.append(record)
        if client_index >= clients:
            continue
        block_size = int(blob.get("block_size") or 0)
        origin = min(x["t"] for x in run)
        for leaf in run:
            rows.append(_row(client_index, record, leaf, origin, block_size))
    # Total by construction: two rows can share an arrival and a client, never
    # a path within one client.
    rows.sort(key=lambda r: (r["arrival_s"], r["client_index"], r["json_path"]))

    text = render(rows)
    used = episodes[:clients]
    manifest = {
        "class": klass,
        "clients": clients,
        "client_counts": list(CLIENT_COUNTS),
        "pool_size": POOL_SIZE,
        "nested_pool": (
            f"the first {clients} roots of one fixed ordered pool of "
            f"{POOL_SIZE}, so this cell's requests are a subset of the next "
            f"client count's"
        ),
        "rule": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(rule).items()
        },
        "episode_definition": (
            "a maximal run of API-leaf intervals [t, t+api_time) on the root's "
            f"clock joined by gaps of at most {rule.gap_s:.1f}s; cut from all "
            f"leaves and accepted or rejected entire"
        ),
        "corpus": dict(CORPUS),
        "corpus_sha256_observed": digest_file(corpus),
        "development_sessions_excluded": list(DEVELOPMENT_SESSIONS),
        "context_tokens": CONTEXT_TOKENS,
        "servable_types": list(SERVABLE_TYPES),
        "generator_sha256": digest_file(__file__),
        "base_generator_sha256": digest_file(base.__file__),
        "arrivals": dict(ARRIVALS),
        "target_model": TARGET_MODEL,
        "source_models_declared": (
            "the corpus served these requests with several models; every row "
            "carries its source label and every row is replayed against "
            "target_model. No descendant is dropped for a differing label"
        ),
        "source_models": dict(
            sorted(Counter(str(r["source_model"]) for r in rows).items())
        ),
        "roots_used": used,
        "roots_pool": episodes,
        "requests": len(rows),
        "root_requests": sum(1 for r in rows if r["actor"] == "root"),
        "descendant_requests": sum(1 for r in rows if r["actor"] == "subagent"),
        "branches": sum(len(e["branches"]) for e in used),
        "input_tokens": base._quantiles([r["input_tokens"] for r in rows]),
        "output_tokens": base._quantiles([r["output_tokens"] for r in rows]),
        "arrival_span_s": max(r["arrival_s"] for r in rows) if rows else 0.0,
        "source_peak_overlap": max(e["source_peak_overlap"] for e in used),
        "source_peak_overlap_means": (
            "the most requests the source had in flight at once, as evidence "
            "that this slice contains concurrency. It is not a bound on the "
            "served engine's outstanding requests: Qwen's service times are "
            "not the trace's, so these arrivals may reach max_num_seqs or "
            "queue past it"
        ),
        "episodes_rejected": dict(sorted(rejected.items())),
        "episodes_rejected_value_range": {
            k: {"min": v[0], "max": v[1]} for k, v in sorted(rejected_values.items())
        },
        "episodes_rejected_scope": (
            "every episode examined while filling this class's pool of "
            f"{POOL_SIZE} roots, in corpus order. Scanning stops once the pool "
            "is full, so this is a census of what was refused on the way to "
            "the selection, not of the whole corpus"
        ),
        "gaps_clipped": 0,
        "outputs_altered": sum(1 for r in rows if r["output_tokens"] < 1),
        "subagents_pruned": 0,
        "requests_serialised": 0,
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
                "client_index": r["client_index"],
                "session": r["session"],
                "json_path": r["json_path"],
                "actor": r["actor"],
                "agent_id": r["agent_id"],
                "source_t_s": r["source_t_s"],
                "origin_shift_s": r["origin_shift_s"],
                "arrival_s": r["arrival_s"],
                "input_tokens": r["input_tokens"],
                "input_blocks": r["input_blocks"],
                "output_tokens": r["output_tokens"],
                "source_model": r["source_model"],
            }
            for r in rows
        ],
        "sha256": digest_bytes(text.encode()),
    }
    return text, manifest


# --------------------------------------------------------------------------
# commands


def emit(args) -> int:
    text, manifest = build(args.corpus, getattr(args, "class"), args.clients)
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
        f"{manifest['class']} c{manifest['clients']}: {manifest['requests']} "
        f"requests ({manifest['root_requests']} root, "
        f"{manifest['descendant_requests']} descendant) from "
        f"{len(manifest['roots_used'])} root session(s), input median "
        f"{manifest['input_tokens']['median']} max "
        f"{manifest['input_tokens']['max']}, arrivals over "
        f"{manifest['arrival_span_s']:.1f}s -> {args.out}"
    )
    print(f"  sha256 {manifest['sha256']}")
    return 0


def verify(args) -> int:
    """Is this the registered workload, and does the rule still produce it?"""
    manifest = json.loads(Path(args.manifest).read_text())
    bad = []
    path = args.workload or (Path(args.manifest).parent / manifest["file"])
    if digest_file(path) != manifest["sha256"]:
        bad.append(f"{path}: bytes do not match the manifest's sha256")
    if args.corpus:
        _text, rebuilt = build(args.corpus, manifest["class"], manifest["clients"])
        if rebuilt["sha256"] != manifest["sha256"]:
            bad.append(
                "re-emitting the rule does not reproduce these bytes: the "
                "selection itself has changed"
            )
    for reason in bad:
        print(f"  REFUSED: {reason}")
    if not bad:
        print(f"{manifest['class']} c{manifest['clients']}: {manifest['sha256']} ok")
    return 1 if bad else 0


def volumes(args) -> int:
    """What each cell would offer, printed before anything is run."""
    for klass in sorted(RULES):
        rule = RULES[klass]
        pool, _rejected, _values, _counts = select_pool(args.corpus, rule)
        records = [_episode_record(line, blob, run) for line, blob, run in pool]
        print(f"== {klass} ==")
        print(
            f"   {'line':<5} {'session':<24} {'reqs':>5} {'desc':>5} {'br':>4} "
            f"{'peak':>4} {'span_s':>8} {'in_tok':>11} {'out_tok':>10}"
        )
        for record in records:
            print(
                f"   {record['corpus_line_1based']:<5d} "
                f"{str(record['id'])[:24]:<24} "
                f"{record['requests']:>5d} {record['descendant_requests']:>5d} "
                f"{len(record['branches']):>4d} "
                f"{record['source_peak_overlap']:>4d} "
                f"{record['arrival_span_s']:>8.1f} "
                f"{record['input_tokens']:>11d} {record['output_tokens']:>10d}"
            )
        print(
            f"   {'clients':<8} {'reqs':>8} {'root':>8} {'desc':>8} "
            f"{'in_tok':>13} {'out_tok':>12} {'span_s':>9} {'src_peak':>9}"
        )
        for count in CLIENT_COUNTS:
            used = records[:count]
            print(
                f"   {count:<8d} "
                f"{sum(e['requests'] for e in used):>8d} "
                f"{sum(e['root_requests'] for e in used):>8d} "
                f"{sum(e['descendant_requests'] for e in used):>8d} "
                f"{sum(e['input_tokens'] for e in used):>13d} "
                f"{sum(e['output_tokens'] for e in used):>12d} "
                f"{max(e['arrival_span_s'] for e in used):>9.1f} "
                f"{max(e['source_peak_overlap'] for e in used):>9d}"
            )
        print()
    return 0


def describe(_args) -> int:
    print(__doc__.strip())
    for klass in sorted(RULES):
        print()
        print(f"--- {klass}")
        print((RULES[klass].__doc__ or "").strip())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("emit", help="write one class's workload and manifest")
    e.add_argument("--class", required=True, choices=sorted(RULES))
    e.add_argument("--clients", required=True, type=int, choices=list(CLIENT_COUNTS))
    e.add_argument("--corpus", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--at", required=True, help="the date this was emitted")
    e.set_defaults(func=emit)

    v = sub.add_parser("verify", help="check a workload against its manifest")
    v.add_argument("--manifest", required=True)
    v.add_argument("--workload")
    v.add_argument("--corpus", help="also re-run the rule and compare bytes")
    v.set_defaults(func=verify)

    n = sub.add_parser("volumes", help="what every client count would offer")
    n.add_argument("--corpus", required=True)
    n.set_defaults(func=volumes)

    d = sub.add_parser("describe", help="the rules, in words")
    d.set_defaults(func=describe)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
