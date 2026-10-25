# SPDX-License-Identifier: MIT
"""A real run's recorded steps, labelled one way for every reader.

A record is a real run's step record (`steps/dp<r>.jsonl`, written with
`ATOM_COMPASS_PARITY_RECORD`), read as JSON lines. `label` makes each forward
that is not a dummy run a `Sample`: its batch view, its device label, its host
labels and its provenance. `step_linear_fit` fits samples and `validate.cost`
prices them; neither reads a step's stamps. Neither takes a record whose
run.json names `BUILDS`, aiter JIT modules its server built while it served;
`refusal` says why.

A data-parallel step's view takes its line's `moe_tokens`, `tbo_tokens` and
`dp_rank`. An older record's lines state neither count: `group_tokens`
rebuilds `moe_tokens` from every rank's line, and, when its host key's engine
states `enable_tbo`, the caller's `split_rule`, ATOM's two-batch-overlap split
over those lines, rebuilds `tbo_tokens`; with no rule such a line refuses.

A step's device label is its device busy time, `d_busy_ns`, plus for a decode
step the device work queued since the previous forward's exit, `d_gap_ns`. A
decode step whose device start precedes its host launch end is launch-exposed
and cut. A prefill step is cut as host-paced when its exit event ran less than
`PACED_SHARE` of its device label after the host recorded it: the device had
caught up with the host's launch, so the window holds device idle between
launches and only bounds the device time. A device event runs no earlier than
the host records it, so the largest launch record less its device start, over
the record, is the host clock's lead there. A host bubble in a prefill whose
device stays behind the host stays in its label. On an MTP record, one with
`d_target_ns`, the device label is the target's, `d_target_ns` with no gap,
and the draft label is the rest of
`d_busy_ns`. On a prefill step that produces output, the draft label's
`propose` pass, `d_propose_ns`, is also its propose label. A decode's
draft-KV pass, `d_draft_kv_ns`, runs before `forward_done_event`: its device
label holds it, as its draft-KV label too, and its draft label does not. The exit of each
wait for the previous output's token copy (`t_recv_ns`)
bounds the host clock's lead over the device's (that forward had finished by
then); the tightest of the last `ANCHORS` bounds is the lead. The copy itself
lands on its own stream after the exit event the bound reads, by 14 to 29 us
measured on one host (median 17 us), so each bound carries that much more lead
than the clocks actually have; on the step linear law's records the gap never
moves a decode step across the launch-exposed cut, so it is left in the bound
rather than corrected by a host-specific number with no record or law of its
own to carry it. A record without device windows is refused. A step mixing
prefill and decode rows is cut.
A step of an MTP record without `d_target_ns` is cut as `unsplit`: its device
label would hold a draft the law prices again.

A step's host labels are its segments: `E` from the previous forward's exit,
cut as idle when the two batches share no request; `W_pre` to the launch, less
the staging gate's wait; `W_launch` from the launch to the token copy wait,
the sampler's launch included, or to the launch's end on a forward with no
such wait; `W_post` from that wait's end to the exit, less the draft copy wait,
and a middle chunk has none. With an MTP status wait,
`W_pre` ends at it and `W_inputs` runs from it to the launch. On a later
pipeline stage, `W_recv` runs from the launch to the receive of its input.
Where the record stamps input pulls (`dp<r>.pulls.jsonl`), `E` runs from the
pull after the previous exit, and `E_out` from the exit to the pull after it.
`L_mtp` is `mtp_copy`'s time on a forward whose MTP status wait began before
the event it waits for.
`label_run` labels a run's DP ranks together: there `E` runs from the latest
pull, else exit, over the ranks before the same forward, where the group's
`_sync_dp_state` releases each of them.

`write_table` writes a record's labels beside it, `dp<r>.cost.jsonl`, for
people: a first line `{"version": TABLE_VERSION}`, then per step `step`,
`d_busy_s`, `d_gap_s`, `launch_exposed`, `idle`, the segments, `label_s`,
`draft_s` and `cut`, the reason its device label was cut; seconds throughout, null where
the step has none.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.backends.shape import BatchView, RequestShape

#: Token copy wait exits whose tightest bound is a step's host clock lead.
ANCHORS = 32
#: Why a device label is cut: a launch-exposed decode, mixed rows, no MTP target
#: end, a host-paced prefill.
CUTS = ("exposed", "mixed", "unsplit", "paced")
#: A prefill's exit event running less than this share of its device label after
#: its host record makes it host-paced: between the two clusters the DSpark TP2
#: records show, host-paced output prefills at 0 to 4.3%, device-paced at 11% up.
PACED_SHARE = 0.08
HOST_SEGMENTS = ("E", "W_pre", "W_launch", "W_post")
TABLE_VERSION = 1
#: run.json's entry naming the aiter JIT modules its server built while it
#: served (`atom.compass.drive.aiter_jit`); a server that built none has none.
BUILDS = "aiter_jit_builds"


def refusal(run: dict) -> str | None:
    """Why grading, a fit or `validate.cost` refuses a record whose run.json is
    `run`, naming the modules of its `BUILDS`; None when it names none."""
    if builds := run.get(BUILDS):
        return (
            f"aiter ran a JIT build of {builds} while its server served, "
            "so a step may have waited on it"
        )
    return None


@dataclass(frozen=True)
class Sample:
    """One real forward, labelled.

    `device_s` is its device label, None when cut, and `cut` names why (`CUTS`).
    `draft_s` is its draft label, None when cut or on a step with no draft.
    `propose_s` is its propose label, the part of `draft_s` that is the
    `propose` pass, on a prefill step that produces output; None elsewhere.
    `draft_kv_s` is its draft-KV label, the part of `device_s` that is the
    drafter's draft-KV pass, on a decode step; None elsewhere.
    `host` holds the segments it has, in seconds: no `W_post` on a middle chunk,
    no `E` on the record's first forward or after an idle engine (`idle`).
    `provenance` holds `source`, `run` (the run's digest), `requests` (the
    request ids the step served) and `config` (the host key the record states,
    or None).
    `dummies` holds the steps its rank ran as a DP dummy forward, which give no
    sample: the same set on each of the rank's samples.
    """

    step: int
    rank: int
    view: BatchView
    device_s: float | None
    cut: str | None
    host: dict
    idle: bool
    provenance: dict
    draft_s: float | None = None
    propose_s: float | None = None
    dummies: frozenset = frozenset()
    draft_kv_s: float | None = None


def group_tokens(lines) -> int | None:
    """The real tokens among a data-parallel step's `moe_rows`: ATOM's
    `num_tokens_across_dp` summed over every rank's line at that step, a dummy
    rank's one row. None when a rank's line is missing."""
    return None if None in lines else sum(q for s in lines for q, _, _ in s["rows"])


def digest(record) -> str:
    """The record's file name and the head of its bytes' SHA-256."""
    path = Path(record)
    return f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"


def host_kind(view: BatchView) -> str:
    """`decode`, `prefill`, or `middle` for a step that produces no output."""
    if not view.produces_output:
        return "middle"
    return "prefill" if view.prefill else "decode"


def _timed(lines) -> tuple[list[tuple[dict, int]], int]:
    """Each line with a device window and its device start, gaps and busy
    times summed; and the host clock's lead over that device clock."""
    device, timed = 0, []
    for s in lines:
        if "d_busy_ns" in s:
            start = device + s.get("d_gap_ns", 0)
            device = start + s["d_busy_ns"]
            timed.append((s, start))
    return timed, max(s["t_launch_ns"][0] - start for s, start in timed)


def mtp_copy(lines) -> dict[int, tuple[int, bool]]:
    """Per forward with an MTP status wait after an output forward that marks
    its target: the nanoseconds from that forward's `forward_done_event`, after
    its target and draft-KV pass, to the wait's return, and whether the wait
    began before the event. The lead places the event no later than it ran, so
    the time is an upper bound."""
    timed, lead = _timed(lines)
    copy, done = {}, None
    for s, start in timed:
        if "t_mtp_ns" in s and done is not None:
            m0, m1 = s["t_mtp_ns"]
            copy[s["step"]] = (m1 - done - lead, m0 < done + lead)
        if s["produces_output"]:
            done = None
            if "d_target_ns" in s:
                done = start + s["d_target_ns"] + s.get("d_draft_kv_ns", 0)
    return copy


def _host_cut(lines) -> tuple[set[int], set[int]]:
    """The steps whose device start precedes their host launch end, and the
    prefill steps the host paced."""
    (lines_timed, lead), output_end, timed = _timed(lines), None, []
    for s, start in lines_timed:
        bound = None
        if "t_recv_ns" in s and output_end is not None:
            bound = s["t_recv_ns"][1] - output_end
        if s["produces_output"]:
            output_end = start + s["d_busy_ns"]
        timed.append((s, start, bound))
    bounds = [b for _, _, b in timed if b is not None]
    if not bounds:
        raise ValueError(
            "no token copy wait exit (`t_recv_ns`) ties the record's host and device clocks"
        )
    # Steps before the first anchor read the first ones.
    first, recent, exposed = min(bounds[:ANCHORS]), deque(maxlen=ANCHORS), set()
    for s, start, bound in timed:
        if bound is not None:
            recent.append(bound)
        if start + min(recent, default=first) < s["t_launch_ns"][1]:
            exposed.add(s["step"])
    paced = {
        s["step"]
        for s, start, _ in timed
        if not all(r[2] for r in s["rows"])
        and start + s["d_busy_ns"] + lead - s["t_exit_ns"]
        < PACED_SHARE * s.get("d_target_ns", s["d_busy_ns"])
    }
    return exposed, paced


def join_stamps(steps: list[dict], path) -> None:
    """Put into each step the first stamp of each key in `path` at or after
    its exit and before the next step's entry."""
    times = defaultdict(list)
    for stamps in read_jsonl(path):
        for key, ns in stamps.items():
            times[key].append(ns)
    ends = [s["t_enter_ns"] for s in steps[1:]] + [float("inf")]
    for step, end in zip(steps, ends):
        for key, ts in times.items():
            i = bisect.bisect_left(ts, step["t_exit_ns"])
            if i < len(ts) and ts[i] < end:
                step[key] = ts[i]


def _shared(step, other) -> bool:
    """Whether two steps' batches hold a request in common."""
    return bool({r[0] for r in step["batch"]} & {r[0] for r in other["batch"]})


def _read(record) -> list[dict]:
    path = Path(record)
    lines = read_jsonl(path)
    if (pulls := path.with_suffix(".pulls.jsonl")).exists():
        join_stamps(lines, pulls)
    return lines


def _pull(step: dict) -> int:
    return step.get("t_pull_ns", step["t_exit_ns"])


def label(
    record, run: str | None = None, split_rule=None
) -> tuple[list[Sample], list[dict]]:
    """The record's samples, in step order, and its table's rows.

    `run` names the run in each sample's provenance, the record's `digest` by
    default; `split_rule` rebuilds an older record's split, as above.
    """
    lines = _read(record)
    return _label(record, lines, run, [_pull(s) for s in lines], split_rule)


def label_run(
    records, run: str | None = None, split_rule=None
) -> list[tuple[list[Sample], list[dict]]]:
    """`label` of each of `records`, one run's DP ranks, with each `E` from the
    latest pull over the ranks before the same forward. A rank that ran only
    DP's dummy forwards gives no sample and no row. Ranks whose line counts
    differ by more than the forward a stopped server leaves on some refuse."""
    ranks = [_read(r) for r in records]
    counts = {Path(r).name: len(x) for r, x in zip(records, ranks) if x}
    if counts and max(counts.values()) - min(counts.values()) > 1:
        raise ValueError(
            f"the run's rank records hold {counts} lines, so their forwards "
            "do not pair up"
        )
    n = max(map(len, ranks), default=0)
    starts = [max(_pull(x[i]) for x in ranks if i < len(x)) for i in range(n)]
    return [
        (
            ([], [])
            if x and all(s["is_dummy_run"] for s in x)
            else _label(r, x, run, starts, split_rule)
        )
        for r, x in zip(records, ranks)
    ]


def _label(record, lines, run, starts, split_rule) -> tuple[list[Sample], list[dict]]:
    """`label` of `record`'s `lines`, each `E` from `starts` at the line before."""
    path = Path(record)
    steps = [s for s in lines if not s["is_dummy_run"]]
    dummies = frozenset(s["step"] for s in lines if s["is_dummy_run"])
    if not steps or "t_enter_ns" not in steps[0]:
        raise ValueError(
            f"{record} has no timed steps; a fit needs a real run's record"
        )
    if "d_busy_ns" not in steps[0]:
        raise ValueError(
            f"{record} has no device windows (`d_busy_ns`); a fit labels a step "
            "by its device time"
        )
    exposed, paced = _host_cut(lines)
    copy = mtp_copy(lines)
    mtp = any("d_target_ns" in s for s in steps)
    key = path.with_suffix(".key.json")
    config = read_json(key) if key.exists() else None
    tbo = (config or {}).get("engine", {}).get("enable_tbo")
    run = run or digest(record)
    samples, table, prev, ranks = [], [], None, None
    for i, s in enumerate(lines):
        if s["is_dummy_run"]:
            prev = s
            continue
        dp, tokens = "moe_rows" in s, s.get("moe_tokens")
        split = s.get("tbo_tokens")
        if dp and tokens is None:
            if ranks is None:
                peers = path.parent.glob(path.name.split("dp")[0] + "dp*[0-9].jsonl")
                ranks = [{t["step"]: t for t in read_jsonl(p)} for p in sorted(peers)]
            group = [r.get(s["step"]) for r in ranks]
            tokens = group_tokens(group)
            if tokens is not None and tbo:
                if split_rule is None:
                    raise ValueError(
                        f"{record} step {s['step']} states no two-batch-overlap "
                        "split under enable_tbo, and no split rule was given to "
                        "rebuild it"
                    )
                split = split_rule(group)
        view = BatchView(
            tuple(RequestShape(q, c, bool(d)) for q, c, d in s["rows"]),
            capture_rung=s["rung"],
            moe_rows=s.get("moe_rows"),
            produces_output=s["produces_output"],
            tbo_tokens=None if split is None else tuple(map(tuple, split)),
            moe_tokens=tokens,
            dp_rank=s["dp_rank"] if dp else None,
        )
        cut = None
        if view.prefill and view.decode:
            cut = "mixed"
        elif not view.prefill and s["step"] in exposed:
            cut = "exposed"
        elif view.prefill and s["step"] in paced:
            cut = "paced"
        elif mtp and "d_target_ns" not in s:
            cut = "unsplit"
        gap = 0 if view.prefill else s.get("d_gap_ns", 0)
        device = None if cut else (gap + s["d_busy_ns"]) * 1e-9
        draft = propose = draft_kv = None
        if "d_target_ns" in s:  # an MTP record: the target alone, no gap
            target = s["d_target_ns"]
            if not view.prefill:
                draft_kv = s.get("d_draft_kv_ns")
                target += draft_kv or 0
            device = None if cut else target * 1e-9
            draft = (s["d_busy_ns"] - target) * 1e-9
            if view.prefill and view.produces_output:
                propose = s.get("d_propose_ns")
        enter, (l0, l1) = s["t_enter_ns"], s["t_launch_ns"]
        g0, g1 = s.get("t_gate_ns", (l0, l0))
        r0, r1 = s.get("t_recv_ns", (l1, l1))
        m0, m1 = s.get("t_mtp_ns", (l0, l0))
        host = {"W_pre": g0 - enter + m0 - g1, "W_launch": r0 - l0}
        if "t_mtp_ns" in s:
            host["W_inputs"] = l0 - m1
        if "t_pp_recv_ns" in s:
            host["W_recv"] = s["t_pp_recv_ns"][0] - l0
        if host_kind(view) != "middle":
            d0, d1 = s.get("t_draft_recv_ns", (0, 0))
            host["W_post"] = s["t_exit_ns"] - r1 - (d1 - d0)
        if "t_pull_ns" in s:
            host["E_out"] = s["t_pull_ns"] - s["t_exit_ns"]
        if copy.get(s["step"], (0, False))[1]:
            host["L_mtp"] = copy[s["step"]][0]
        idle = prev is not None and not _shared(s, prev)
        if prev is not None and not idle:
            host["E"] = enter - starts[i - 1]
        host = {k: v * 1e-9 for k, v in host.items()}
        provenance = {
            "source": "record",
            "run": run,
            "requests": [r[0] for r in s["batch"]],
            "config": config,
        }
        samples.append(
            Sample(
                s["step"],
                s["dp_rank"],
                view,
                device,
                cut,
                host,
                idle,
                provenance,
                None if cut else draft,
                None if cut or propose is None else propose * 1e-9,
                dummies,
                None if cut or draft_kv is None else draft_kv * 1e-9,
            )
        )
        gap_ns = s.get("d_gap_ns")
        table.append(
            {
                "step": s["step"],
                "d_busy_s": s["d_busy_ns"] * 1e-9,
                "d_gap_s": None if gap_ns is None else gap_ns * 1e-9,
                "launch_exposed": s["step"] in exposed,
                "idle": idle,
                **{k: host.get(k) for k in (*HOST_SEGMENTS, "W_inputs", "E_out")},
                "label_s": device,
                "draft_s": draft,
                "cut": cut,
            }
        )
        prev = s
    return samples, table


def write_table(record, table: list[dict]) -> Path:
    """Write `table`, `label`'s rows for `record`, beside it; return its path."""
    path = Path(record).with_suffix(".cost.jsonl")
    lines = [{"version": TABLE_VERSION}, *table]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path
