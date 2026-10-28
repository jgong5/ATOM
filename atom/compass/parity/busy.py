# SPDX-License-Identifier: MIT
"""Each forward's device busy time per pass, reduced from a trace run's traces.

A trace run is `compass record --trace`: its server ran under ATOM's kineto
profiler on every TP rank, without Python stacks, from before the first
request to its stop, every rank but 0 paused for each drafter pass, and each
rank wrote ``OUT/trace/rank_<r>/`` one gzipped
chrome trace. `reduce` writes ``OUT/steps/dp0.busy.jsonl``: per line of the
record, its ``step`` and ``b_target_ns``, and on an MTP record ``b_draft_ns``
and, for each drafter pass `PASSES` names that the line times, its busy time
under that pass's ``b_`` name. `step_labels` labels a prefill pass by them.

The n-th ``ATOM::forward`` range on each rank is the record's n-th line, its
``tokens=`` the line's; the ranks and the record must agree on both. A
forward's kernels, memcpys and memsets are those whose launch call opens
between its ``ATOM::run_model`` start and its end. The target's are launched
before the recorder's `TARGET_DONE` range opens, on a forward with none before
its first drafter pass, and outside every drafter pass range; the rest are the
draft's, and those launched inside a drafter pass's range that pass's too.

A pass's busy time is the union of its kernels' device intervals plus each gap
before one of them on the forward's device timeline: whole where that kernel's
launch call had returned when the gap opened, so the device idled on its own,
else capped at the rank's median such queued gap, since the device then waited
for the host's launch. A TP collective's kernel on one rank includes its wait
for the peers: the n-th target collective kernel (`COLLECTIVE`) of a forward
counts on every rank from its latest rank's start, and as queued once its
latest rank's launch call returned. A rank that paused its profiler for a
drafter pass holds none of its kernels, so the gap before the first kernel
launched after its `PAUSED` range opened holds that pass's device time and
counts nothing. The target's busy time is its largest over the ranks; the
draft's and each drafter pass's are rank 0's, the one rank that profiled them.

A trace directory not named ``rank_<r>``, a DP or a pipeline run's, refuses:
per-rank busy time under DP is not reduced.
"""

import gzip
import json
import re
import statistics
from bisect import bisect_right
from pathlib import Path

from atom.compass.backends.json_file import read_jsonl
from atom.compass.backends.step_labels import TRACE
from atom.compass.parity import MARK, PAUSED, TARGET_DONE

RECORD = "steps/dp0.jsonl"
GPU = ("kernel", "gpu_memcpy", "gpu_memset")
RUNTIME = ("cuda_runtime", "cuda_driver")
#: A kernel whose name matches is a TP collective: aiter's custom all-reduce
#: and all-gather, RCCL's, and the mono runner's step-begin peer fence, which
#: holds each rank until every rank has launched it.
COLLECTIVE = re.compile(
    r"cross_device_reduce|allgather|all_gather|all_?reduce|nccl|mono_step_begin"
)
#: The drafter passes a busy line times apart, by their record line field.
PASSES = ("d_propose_ns", "d_draft_kv_ns")
#: Characters read from a trace at a time.
CHUNK = 1 << 24
_SKIP = re.compile(r"[\s,]*")


def events(path):
    """A kineto trace's events, streamed: the trace is one JSON object whose
    ``traceEvents`` list holds them, too large to load whole."""
    decode = json.JSONDecoder().raw_decode
    with gzip.open(path, "rt") as f:
        buf = ""
        while (at := buf.find('"traceEvents"')) < 0:
            if not (chunk := f.read(CHUNK)):
                raise ValueError(f"{path} holds no traceEvents")
            buf += chunk
        buf, pos = buf[buf.index("[", at) + 1 :], 0
        while True:
            pos = _SKIP.match(buf, pos).end()
            if buf.startswith("]", pos):
                return
            try:
                event, pos = decode(buf, pos)
            except json.JSONDecodeError:
                if not (chunk := f.read(CHUNK)):
                    raise
                buf, pos = buf[pos:] + chunk, 0
                continue
            yield event


def forwards(path) -> list[dict]:
    """One rank's forwards: each its ``tokens`` and its ``kernels``, each
    (start, end, launch return, pass, its drafter pass's line field or None,
    name, whether it is the first launched after a `PAUSED` range), in start
    order."""
    ranges, runs, marks, launches, kernels = [], [], [], {}, []
    for e in events(path):
        cat, ts = e.get("cat"), e.get("ts")
        if cat == "user_annotation":
            span = (ts, ts + e["dur"], e["name"])
            if e["name"].startswith("ATOM::forward"):
                ranges.append(span)
            elif e["name"] == "ATOM::run_model":
                runs.append(ts)
            elif e["name"].startswith(MARK):
                marks.append(span)
        elif cat in RUNTIME and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = (ts, ts + e["dur"])
        elif cat in GPU:
            kernels.append((ts, ts + e["dur"], e["args"]["correlation"], e["name"]))
    ranges.sort()
    starts = [r[0] for r in ranges]
    out = []
    for f0, f1, name in ranges:
        mine = [m for m in marks if f0 <= m[0] <= f1]
        done = [m[0] for m in mine if m[2] == TARGET_DONE]
        drafts = [m for m in mine if m[2] not in (TARGET_DONE, PAUSED)]
        out.append(
            {
                "tokens": int(re.search(r"tokens=(\d+)", name).group(1)),
                "run": min((t for t in runs if f0 <= t <= f1), default=f0),
                "end": min(done or [m[0] for m in drafts] or [f1 + 1]),
                "drafts": drafts,
                "pauses": [m[0] for m in mine if m[2] == PAUSED],
                "kernels": [],
            }
        )
    for start, stop, corr, name in kernels:
        launch, queued = launches.get(corr, (None, None))
        i = bisect_right(starts, launch) - 1 if launch is not None else -1
        if i < 0 or not out[i]["run"] <= launch <= ranges[i][1]:
            continue
        f = out[i]
        inside = [m[2] for m in f["drafts"] if m[0] <= launch <= m[1]]
        part = "target" if launch < f["end"] and not inside else "draft"
        key = next((m[len(MARK) :] for m in inside), None)
        f["kernels"].append((launch, start, stop, queued, part, key, name))
    for f in out:
        resumed = {
            min((k for k in f["kernels"] if k[0] > p), default=None)
            for p in f.pop("pauses")
        }
        f["kernels"] = sorted((*k[1:], k in resumed) for k in f["kernels"])
    return out


def _align(ranks: list[list[dict]]) -> None:
    """Start each forward's n-th target collective kernel on every rank no
    earlier than its latest rank's start, queued at its latest launch return."""
    for fs in zip(*ranks):
        calls = [
            [
                j
                for j, k in enumerate(f["kernels"])
                if k[3] == "target" and COLLECTIVE.search(k[5])
            ]
            for f in fs
        ]
        if len({len(c) for c in calls}) > 1:
            raise ValueError(
                f"the ranks' forward of {fs[0]['tokens']} tokens runs "
                f"{[len(c) for c in calls]} target collective kernels"
            )
        for n in range(len(calls[0])):
            ks = [f["kernels"][c[n]] for f, c in zip(fs, calls)]
            last, queued = max(k[0] for k in ks), max(k[2] for k in ks)
            for f, c, k in zip(fs, calls, ks):
                f["kernels"][c[n]] = (min(max(k[0], last), k[1]), k[1], queued, *k[3:])
        for f in fs:
            f["kernels"].sort()


def _walk(kernels):
    """Per kernel in start order: its passes, the union it adds, and the gap
    before it, none after a paused pass, with whether it was queued when the
    gap opened."""
    end = None
    for start, stop, queued, part, key, _, resumed in kernels:
        gap = start - end if end is not None and start > end and not resumed else 0.0
        was = end is not None and queued is not None and queued <= end
        add = stop - max(start, end) if end is not None else stop - start
        end = stop if end is None else max(end, stop)
        yield (part, key) if key else (part,), max(add, 0.0), gap, was


def busy(fs: list[dict]) -> list[dict]:
    """One rank's busy time per forward and pass, in ns."""
    queued = [g for f in fs for _, _, g, q in _walk(f["kernels"]) if q and g > 0]
    if not queued:
        raise ValueError("no kernel was queued at a gap, so no gap measures the cap")
    cap = statistics.median(queued)
    out = []
    for f in fs:
        got = {}
        for parts, add, gap, was in _walk(f["kernels"]):
            for p in parts:
                got[p] = got.get(p, 0.0) + add + (gap if was else min(gap, cap))
        out.append({p: round(us * 1e3) for p, us in got.items()})
    return out


def reduce(out) -> Path:
    """Write the trace run in `out`'s busy lines beside its record; return
    their path."""
    out = Path(out)
    dirs = sorted((out / TRACE).iterdir())
    if not dirs or any(not re.fullmatch(r"rank_\d+", d.name) for d in dirs):
        raise ValueError(
            f"{out / TRACE} holds {[d.name for d in dirs]}, not rank_<r> alone: a "
            "trace run under DP or pipeline stages is not reduced"
        )
    ranks = []
    for d in dirs:
        traces = sorted(d.glob("*.pt.trace.json.gz"))
        if len(traces) != 1:
            raise ValueError(f"{d} holds {len(traces)} traces, not one")
        ranks.append(forwards(traces[0]))
    lines = read_jsonl(out / RECORD)
    tokens = [sum(r[0] for r in s["rows"]) for s in lines]
    if any([f["tokens"] for f in fs] != tokens for fs in ranks):
        raise ValueError(
            f"the ranks' forwards ({[len(fs) for fs in ranks]}) are not the "
            f"record's {len(lines)} lines with their tokens"
        )
    _align(ranks)
    per_rank = [busy(fs) for fs in ranks]
    rows = []
    for i, s in enumerate(lines):
        got = {p: per_rank[0][i].get(p, 0) for p in ("draft", *PASSES)}
        got["target"] = max(r[i].get("target", 0) for r in per_rank)
        row = {"step": s["step"], "b_target_ns": got["target"]}
        if "d_target_ns" in s:
            row["b_draft_ns"] = got["draft"]
        for key in PASSES:
            if key in s:
                row["b" + key[1:]] = got[key]
        rows.append(row)
    path = (out / RECORD).with_suffix(".busy.jsonl")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path
