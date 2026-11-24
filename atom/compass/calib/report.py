# SPDX-License-Identifier: MIT
"""A design's step combinations per segment, and the reports `plan_v1.py`,
`dry_v1.sh` and `check_v1.py` print.

A step's combination (`combo`) is what its segment is designed to produce:
a decode step's rung, whether the rank it is priced at holds the rung itself
(`own`) or a peer does (`peer`, rule 2.4.3) and how many ranks decode; a
prefill step's ranks prefilling, the priced rank's tokens (within the largest
token-graph bucket, `graph`; up to half the step budget, `half`; past it,
`full`), its rows (`1`, `2-8`, `9+`), its history (`h0`, below 128k `h<128k`,
`h>=128k`) and its prefill duty (`cold` below 0.3, `hot` above 0.7, else
`warm`). Only GPU-bound steps count toward a combination's design steps.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from atom.compass.backends.shape import sum_context, sum_tokens


def combo(step, dep) -> str:
    view = step.view
    if step.kind == "decode":
        n = len(view.decode)
        held = (
            "own" if step.rung is None or n >= dep.rows_of(step.rung).start else "peer"
        )
        busy = sum(1 for _, v, *_ in step.ranks if v.decode and not v.dp_dummy)
        return f"decode r{step.rung} {held} {busy}rk"
    busy = sum(1 for _, v, *_ in step.ranks if v.prefill and not v.dp_dummy)
    tokens = sum_tokens(view.requests)
    biggest = max(dep.prefill_buckets or (0,))
    band = (
        "graph"
        if tokens <= biggest
        else "half" if tokens <= dep.max_num_batched_tokens // 2 else "full"
    )
    n = len(view.requests)
    rows = "1" if n == 1 else "2-8" if n <= 8 else "9+"
    history = sum_context(view.requests) - tokens
    hist = "h0" if history == 0 else "h<128k" if history < 131072 else "h>=128k"
    duty = view.prefill_duty or 0.0
    heat = "cold" if duty < 0.3 else "hot" if duty > 0.7 else "warm"
    return f"prefill {busy}rk {band} {rows} {hist} {heat}"


def combos(steps, dep) -> dict[str, dict[str, int]]:
    """Per segment, GPU-bound steps per combination."""
    out: dict = defaultdict(Counter)
    for s in steps:
        if s.gpu_bound:
            out[s.seg][combo(s, dep)] += 1
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def segment_rows(steps, runs) -> list[dict]:
    """Per segment: its span, steps by kind, GPU-bound and not, decode steps per rung."""
    by = defaultdict(list)
    for s in steps:
        by[s.seg].append(s)
    out = []
    for run in runs:
        got = by.get(run.name, [])
        rungs = Counter(s.rung for s in got if s.kind == "decode" and s.gpu_bound)
        out.append(
            {
                "segment": run.name,
                "start_s": round(run.start, 1),
                "seconds": round(run.end - run.start, 1),
                "requests": run.requests,
                "decode": sum(1 for s in got if s.kind == "decode"),
                "prefill": sum(1 for s in got if s.kind == "prefill"),
                "not_gpu_bound": sum(1 for s in got if not s.gpu_bound),
                "decode_rungs": {
                    str(k): v for k, v in sorted(rungs.items(), key=lambda x: x[0] or 0)
                },
            }
        )
    return out


def _g(x) -> str:
    return "inf" if x is None or not math.isfinite(x) else f"{x:.2%}"


def render_vectors(vectors: list[dict]) -> list[str]:
    """A markdown table: each vector's variants, their rank, worst VIF, G and verdict."""
    lines = [
        "| vector | variant | steps (GPU-bound) | rank | max VIF | G | range vertices (paced) | verdict |",
        "|---|---|--:|--:|--:|--:|--:|---|",
    ]
    for v in vectors:
        for i, x in enumerate(v["variants"]):
            head = v["vector"] if i == 0 else ""
            steps = f"{v['steps']} ({v['gpu_bound']})" if i == 0 else ""
            if "error" in x:
                lines.append(
                    f"| {head} | {x['variant']} | {steps} | | | | | {x['error']} |"
                )
                continue
            vifs = x["vif"]
            worst = max(vifs, key=vifs.get) if vifs else None
            vif = f"{vifs[worst]:.3g} ({worst})" if worst else "-"
            notes = ["pass" if x["pass"] else "; ".join(x["fails"])]
            if x.get("rule6"):
                notes.append(x["rule6"])
            if x.get("structural"):
                notes.append("structural: " + ", ".join(x["structural"]))
            if x.get("provisional"):
                notes.append("provisional: " + ", ".join(x["provisional"]))
            lines.append(
                f"| {head} | {x['variant']} | {steps} | {x['rank']}/{len(x['columns'])} | "
                f"{vif} | {_g(x['g'])} | {v['vertices']} ({v['vertices_paced']}) | "
                f"{'; '.join(notes)} |"
            )
    return lines


def render_vif(vectors: list[dict]) -> list[str]:
    """Each vector's first passing variant's (else first variant's) VIF per column."""
    lines = []
    for v in vectors:
        ok = [x for x in v["variants"] if x.get("pass")] or [
            x for x in v["variants"] if "vif" in x
        ]
        if not ok:
            continue
        x = ok[0]
        cells = ", ".join(
            f"{c} {f:.3g}"
            + (f"/{x['vif_weighted'][c]:.3g}w" if "vif_weighted" in x else "")
            for c, f in x["vif"].items()
        )
        lines.append(f"- {v['vector']} ({x.get('variant', 'base')}): {cells}")
    return lines
