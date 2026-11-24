#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A design's run plan, segments chained by events, and its CPU rehearsal on a law.

    PYTHONPATH=$T python3 scripts/compass/plan_v1.py DESIGN [--law LAW --host HOST]

DESIGN is `gen_v1.py`'s OUT. Writes DESIGN/plan.json, which `cal_v1.sh` runs:
the segments in config order, each started by an event, never by a predicted
time (#739 2.4 rule 1): the server's start for the first; for each later one,
every lane of the one before has exited and the server holds no request
(`atom:requests_running` plus `atom:requests_waiting` at 0). A segment's lanes
start together, each a client run of its own: its trace file, concurrency,
aiperf seed and, for a pinned lane, `--extra-inputs data_parallel_rank:R`.
The server runs under `ATOM_DP_SESSION_AFFINITY=1` and
`ATOM_DP_LB_REQ_EQUIV=1000000000` (`server_env`), the routing the design
places unpinned sessions by.

With LAW (a step linear law: `step_linear_law.json` or its `params` as
law.json) and HOST (`host_laws.json`), the design is rehearsed on them
(`atom.compass.calib.engine`): per segment its rehearsed span, steps and
GPU-bound steps and its combinations (`report.combos`, what `check_v1.py`
compares a record against); the predicted GPU minutes, the server's start
and every segment's overhead included; and each vector's identifiability over
the GPU-bound steps (`atom.compass.calib.ident`: rank, VIF, G, levels) against
the config's acceptance criteria. DESIGN/plan.md holds the same as tables.
"""

import argparse
import json
import time
from pathlib import Path

from atom.compass.calib import deploy, design, engine, ident, report

SERVER_ENV = {"ATOM_DP_SESSION_AFFINITY": "1", "ATOM_DP_LB_REQ_EQUIV": "1000000000"}
EVENT = "every lane of the segment before has exited and the server holds no request"


def sequence(doc: dict) -> list[dict]:
    out = []
    for k, seg in enumerate(doc["segments"]):
        out.append(
            {
                "segment": seg["name"],
                "after": (
                    "server start"
                    if not k
                    else f"{doc['segments'][k - 1]['name']}: {EVENT}"
                ),
                "lanes": [
                    {
                        **{
                            x: lane[x]
                            for x in (
                                "name",
                                "file",
                                "pin",
                                "concurrency",
                                "traces",
                                "requests",
                            )
                        },
                        "seed": 200 + 10 * k + j,
                    }
                    for j, lane in enumerate(seg["lanes"])
                ],
            }
        )
    return out


def read_law(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    return doc.get("params", doc)


def rehearse(cfg, segments, law_path: Path, host_path: Path | None) -> dict:
    law = engine.Law(
        read_law(law_path),
        json.loads(Path(host_path).read_text()) if host_path else None,
        cfg.deployment.dp,
    )
    t0 = time.time()
    got = engine.rehearse(cfg, segments, law)
    vectors = ident.vectors(cfg, got.steps, law)
    return {
        "law": str(law_path),
        "host": str(host_path) if host_path else None,
        "cpu_seconds": round(time.time() - t0, 1),
        "gpu_minutes": round(got.gpu_minutes, 2),
        "wall_s": round(got.wall_s, 1),
        "refused": got.refused,
        "segments": report.segment_rows(got.steps, got.runs),
        "combos": report.combos(got.steps, cfg.deployment),
        "vectors": vectors,
        "pass": all(v["pass"] for v in vectors),
    }


def _pin(pin) -> str:
    return "router" if pin is None else f"rank {pin}"


def markdown(cfg, plan: dict) -> str:
    lines = [
        f"# {cfg.name}: calibration plan",
        "",
        cfg.about,
        "",
        "## Sequence (chained by events)",
        "",
    ]
    lines += [
        "| # | segment | starts after | lanes (pin, concurrency, traces) |",
        "|--:|---|---|---|",
    ]
    for k, s in enumerate(plan["sequence"]):
        lanes = "; ".join(
            f"{x['name']} ({_pin(x['pin'])}, c={x['concurrency']}, {x['traces']})"
            for x in s["lanes"]
        )
        lines.append(f"| {k} | {s['segment']} | {s['after'].split(':')[0]} | {lanes} |")
    r = plan.get("rehearsal")
    if not r:
        return "\n".join(lines) + "\n"
    acc = cfg.acceptance
    lines += [
        "",
        f"## Rehearsal on {r['law']}",
        "",
        (
            f"Predicted GPU time: **{r['gpu_minutes']:.1f} min** (server start "
            f"{cfg.timing['server_start_s']:.0f} s, {cfg.timing['segment_overhead_s']:.0f} s a "
            f"segment for its client's start and the drain); {r['refused']} views the law "
            f"refused; rehearsed in {r['cpu_seconds']} CPU s."
        ),
        "",
        "| segment | starts at s | lasts s | requests | decode steps | prefill steps | not GPU-bound |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for s in r["segments"]:
        lines.append(
            f"| {s['segment']} | {s['start_s']} | {s['seconds']} | {s['requests']} | {s['decode']} | "
            f"{s['prefill']} | {s['not_gpu_bound']} |"
        )
    rungs = [str(x) for x in cfg.deployment.rungs]
    lines += [
        "",
        (
            "GPU-bound decode steps per rung and segment (each drain needs "
            f"{acc['rung_steps']} a rung it passes):"
        ),
        "",
    ]
    lines += ["| segment | " + " | ".join(rungs) + " |", "|---|" + "--:|" * len(rungs)]
    for s in r["segments"]:
        if s["decode"]:
            lines.append(
                f"| {s['segment']} | "
                + " | ".join(str(s["decode_rungs"].get(x, "")) for x in rungs)
                + " |"
            )
    lines += [
        "",
        (
            f"## Identifiability (GPU-bound steps; VIF <= {acc['vif']:g}, G <= "
            f"{acc['g']['decode']:.0%} decode / {acc['g']['prefill']:.0%} prefill at "
            f"{acc['g_z']:g} sigma, noise {cfg.noise['decode']:.2%} / "
            f"{cfg.noise['prefill']:.2%} a step, levels >= {acc['levels']})"
        ),
        "",
    ]
    lines += report.render_vectors(r["vectors"])
    lines += [
        "",
        "VIF per column (the first passing variant; `w`: weighted as the fit weighs steps):",
        "",
    ]
    lines += report.render_vif(r["vectors"])
    lines += [
        "",
        f"Verdict: **{'pass' if r['pass'] else 'fail'}** (a vector passes when one of its variants does).",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("design", type=Path)
    p.add_argument(
        "--law", type=Path, help="step linear law (step_linear_law.json or law.json)"
    )
    p.add_argument("--host", type=Path, help="host_laws.json")
    a = p.parse_args(argv)
    doc = json.loads((a.design / "design.json").read_text())
    cfg = deploy.parse(doc["config"])
    plan = {
        "config": cfg.name,
        "server_env": SERVER_ENV,
        "event": EVENT,
        "sequence": sequence(doc),
    }
    if a.law:
        plan["rehearsal"] = rehearse(cfg, design.read(a.design, cfg), a.law, a.host)
    (a.design / "plan.json").write_text(json.dumps(plan, indent=1))
    text = markdown(cfg, plan)
    (a.design / "plan.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
