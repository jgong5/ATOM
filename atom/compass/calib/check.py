# SPDX-License-Identifier: MIT
"""After a record (#739 section 4): its steps per segment, each segment's
actual combinations against the plan's, and the design's identifiability on
the record's own steps.

A record is `cal_v1.sh`'s: one server's `real/steps/` (`dp<r>.jsonl`), and per
segment and lane a client run whose export (`<segment>/<lane>/artifacts/
client_run.json`) names the requests it sent, which `parity.served` places
each rank's steps by. A step's views are built as `cal_coverage.py` and the
fit build them: its lines' rows, rung and MoE rows and tokens, the group's
dummies, peak tokens and tail from every rank's lines (`step_labels`), each
rank's prefill duty over its earlier lines. A step is GPU-bound where the
record's labels keep its costliest rank's device label (`step_labels.label_run`:
not cut as launch-exposed or paced); a record without device windows (a
simulated run's) is judged as the rehearsal judges a step (`engine`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atom.compass.backends import step_labels
from atom.compass.backends.json_file import read_jsonl
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_labels import PACED_S
from atom.compass.calib import columns as cols
from atom.compass.calib.engine import Law, Step
from atom.compass.parity.served import served


def exports(record: Path, design: dict) -> list[tuple[str, str, Path]]:
    """(segment, lane, export) of every lane `cal_v1.sh` ran, in plan order."""
    out = []
    for seg in design["segments"]:
        for lane in seg["lanes"]:
            path = record / seg["name"] / lane["name"] / "artifacts" / "client_run.json"
            if path.exists():
                out.append((seg["name"], lane["name"], path))
    return out


def _labels(paths) -> dict | None:
    """{(rank, step): device label or None where cut} from the record's own
    device windows; None for a record without them."""
    try:
        got = step_labels.label_run([str(p) for p in paths])
    except (KeyError, ValueError, TypeError, OSError):  # no device windows, no host key
        return None
    out = {}
    for samples, _ in got:
        for s in samples:
            out[(s.rank, s.step)] = s.device_s
    return out or None


def steps_dir(record) -> Path:
    """A record's step files: `cal_v1.sh`'s server --out (``real/steps``), a
    record directory's ``steps``, or the directory itself."""
    record = Path(record)
    for d in (record / "real" / "steps", record / "steps", record):
        if (d / "dp0.jsonl").exists():
            return d
    raise FileNotFoundError(f"no dp0.jsonl under {record}")


def record_steps(record: Path, design: dict | None, cfg, law: Law, segment=None):
    """The record's steps as `engine.Step`s, each under its segment, and
    per step its measured device seconds at the priced rank (None where none).
    With `segment`, every step of the record is that segment's and no client
    run is read: a target workload's record (`ident.vectors`' `target`)."""
    dep = cfg.deployment
    sdir = steps_dir(record)
    paths = sorted(sdir.glob("dp*[0-9].jsonl"), key=lambda p: int(p.stem[2:]))
    lanes = [] if segment else exports(Path(record), design)
    owner = {} if segment else served(sdir, [p for *_, p in lanes])
    seg_of = {k: lanes[k][0] for k in range(len(lanes))}
    ranks = [read_jsonl(p) for p in paths]
    dummies, peaks = step_labels.group_dummies(ranks), step_labels.group_peaks(ranks)
    tails, prefills = step_labels.group_tails(ranks), step_labels.group_prefills(ranks)
    duties = [step_labels.prefill_duties(lines, prefills) for lines in ranks]
    at = [
        {s["step"]: (s, d) for s, d in zip(lines, du)}
        for lines, du in zip(ranks, duties)
    ]
    whose = {
        (r, s["step"]): s["client_run"] for r, lines in owner.items() for s in lines
    }
    labels = _labels(paths)
    dp = dep.dp > 1
    out, measured = [], []
    for n in sorted(set().union(*at)):
        group = {r: x[n] for r, x in enumerate(at) if n in x}
        real = {r: g for r, g in group.items() if not g[0]["is_dummy_run"]}
        runs = {whose.get((r, n)) for r in real} - {None}
        if not real or (len(runs) != 1 and not segment):
            continue
        seg = segment or seg_of[min(runs)]  # the one client run
        kinds = {bool(row[2]) for s, _ in real.values() for row in s["rows"]}
        if len(kinds) != 1:
            continue  # mixed rows: a fit cuts it
        decode = min(kinds)  # the one kind
        views = []
        for r, (s, duty) in sorted(group.items()):
            if s["is_dummy_run"] and decode:
                continue
            rows = tuple(RequestShape(q, c, bool(d)) for q, c, d in s["rows"])
            view = BatchView(
                rows,
                capture_rung=s.get("rung"),
                moe_rows=s.get("moe_rows") if dp else None,
                moe_tokens=(
                    (
                        s.get("moe_tokens")
                        or step_labels.group_tokens([g[0] for g in group.values()])
                    )
                    if dp
                    else None
                ),
                produces_output=s.get("produces_output"),
                draft_bs=s.get("draft_bs"),
                dp_rank=r if dp else None,
                dp_dummy=bool(s["is_dummy_run"]),
                group_dummies=dummies.get(n, 0),
                group_prefills=dp and not decode,
                group_max_tokens=peaks.get(n) if dp else None,
                group_tail=dp and n in tails,
                group_needs_eager=s.get("group_needs_eager") or False,
                prefill_duty=duty,
            )
            views.append((r, view, *law.price(view)))
        if not views:
            continue
        priced = max(range(len(views)), key=lambda i: (views[i][2], i))
        top_rank, top = views[priced][0], views[priced][1]
        line = group[top_rank][0]
        device_ns = line.get("d_target_ns", line.get("d_busy_ns"))
        device = None if device_ns is None else device_ns * 1e-9
        law_device = max(v[2] + v[3] for v in views)
        if labels is not None:
            label = labels.get((top_rank, n))
            bound = label is not None
            device = label if label is not None else device
        else:
            kind = (
                "decode" if decode else ("prefill" if top.produces_output else "middle")
            )
            tokens = sum(r.query_tokens for r in top.requests)
            if decode:
                bound = law_device >= law.host_s(kind, tokens, len(top.requests))
            else:
                launch = law.host_s(kind, tokens, len(top.requests), ("W_launch",))
                bound = law_device > launch + PACED_S
        rung = top.capture_rung if decode else None
        out.append(
            Step(
                seg,
                "decode" if decode else "prefill",
                rung,
                0.0,
                0.0,
                law_device,
                0.0,
                bound,
                views,
                priced,
            )
        )
        measured.append(device)
    return out, measured


def compare(planned: dict, actual: dict, floor: int = 20) -> list[dict]:
    """Per segment and combination the plan gives `floor` or more GPU-bound
    steps: planned, actual and their ratio; and every combination the record
    holds that the plan does not (`unplanned`)."""
    out = []
    for seg in sorted(set(planned) | set(actual)):
        p, a = planned.get(seg, {}), actual.get(seg, {})
        for key in sorted(set(p) | set(a)):
            want, got = p.get(key, 0), a.get(key, 0)
            if want < floor and got < floor:
                continue
            out.append(
                {
                    "segment": seg,
                    "combo": key,
                    "planned": want,
                    "actual": got,
                    "ratio": None if not want else round(got / want, 2),
                    "short": bool(want) and got < 0.5 * want,
                    "unplanned": not want,
                }
            )
    return out


def residuals(cfg, steps, measured) -> list[dict]:
    """I5: each vector's base form fitted (relative least squares) on the
    GPU-bound steps with a measured label, and per segment the median and
    scaled MAD of its relative residuals."""
    dep, out = cfg.deployment, []
    keys = sorted({(s.kind, s.rung) for s in steps}, key=lambda k: (k[0], k[1] or 0))
    for kind, rung in keys:
        rows = [
            (s, m)
            for s, m in zip(steps, measured)
            if s.kind == kind and s.rung == rung and s.gpu_bound and m
        ]
        names = cfg.columns[kind]
        if len(rows) <= len(names):
            continue
        spread = cfg.columns.get("min_spread", 0.0)
        X = np.array([cols.values(s.view, kind, names, dep, spread) for s, _ in rows])
        y = np.array([m for _, m in rows])
        coef, *_ = np.linalg.lstsq(X / y[:, None], np.ones(len(y)), rcond=None)
        rel = X @ coef / y - 1.0
        segs = {}
        for (s, _), e in zip(rows, rel):
            segs.setdefault(s.seg, []).append(e)
        mad = lambda e: float(1.4826 * np.median(np.abs(np.asarray(e) - np.median(e))))
        out.append(
            {
                "vector": "prefill" if kind == "prefill" else f"decode rung {rung}",
                "steps": len(rows),
                "mad": mad(rel),
                "segments": {
                    k: {"steps": len(e), "median": float(np.median(e)), "mad": mad(e)}
                    for k, e in segs.items()
                },
            }
        )
    return out


def load_design(directory) -> dict:
    return json.loads((Path(directory) / "design.json").read_text())
