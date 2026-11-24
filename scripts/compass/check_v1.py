#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A calibration record against its plan (#739 section 4): per segment the
combinations it produced against the planned ones, and its identifiability.

    PYTHONPATH=$T python3 scripts/compass/check_v1.py DESIGN RECORD --law LAW [--host HOST]

DESIGN is the `gen_v1.py` OUT whose plan.json (`plan_v1.py --law`) holds the
rehearsed combinations; RECORD the directory `cal_v1.sh` wrote: the server's
record (`steps/`) and per segment and lane a client run's export
(`<segment>/<lane>/artifacts/client_run.json`). LAW prices each rank's view,
which names the rank a step is priced at; HOST judges a record without device
windows (a simulated run) as the rehearsal does.

Prints, and writes RECORD/check.json and RECORD/check.md: each segment's
combinations with `floor` or more planned or actual GPU-bound steps, planned
and actual, those under half the plan (`short`) and those not planned
(`unplanned`); each vector's identifiability on the record's own GPU-bound
steps (`atom.compass.calib.ident`); and per vector the relative residuals of
its base form fitted on the record, overall and per segment (I5). Exits 1
when a vector fails or a planned combination falls short.
"""

import argparse
import json
import sys
from pathlib import Path

from atom.compass.calib import check, deploy, engine, ident, report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("design", type=Path)
    p.add_argument("record", type=Path)
    p.add_argument("--law", type=Path, required=True)
    p.add_argument("--host", type=Path)
    p.add_argument("--floor", type=int, default=20)
    a = p.parse_args(argv)
    doc = check.load_design(a.design)
    cfg = deploy.parse(doc["config"])
    plan = json.loads((a.design / "plan.json").read_text())
    law_doc = json.loads(a.law.read_text())
    law = engine.Law(
        law_doc.get("params", law_doc),
        json.loads(a.host.read_text()) if a.host else None,
        cfg.deployment.dp,
    )
    steps, measured = check.record_steps(a.record, doc, cfg, law)
    actual = report.combos(steps, cfg.deployment)
    planned = plan.get("rehearsal", {}).get("combos", {})
    rows = check.compare(planned, actual, a.floor)
    vectors = ident.vectors(cfg, steps, law)
    fits = check.residuals(cfg, steps, measured)
    out = {
        "record": str(a.record),
        "steps": len(steps),
        "combos": rows,
        "vectors": vectors,
        "residuals": fits,
    }
    short = [r for r in rows if r["short"]]
    out["pass"] = all(v["pass"] for v in vectors) and not short
    lines = [
        f"# {cfg.name}: record {a.record} against its plan",
        "",
        f"{len(steps)} steps.",
        "",
    ]
    lines += [
        "| segment | combination | planned | actual | ratio |",
        "|---|---|--:|--:|--:|",
    ]
    for r in rows:
        flag = " (short)" if r["short"] else " (unplanned)" if r["unplanned"] else ""
        lines.append(
            f"| {r['segment']} | {r['combo']} | {r['planned']} | {r['actual']} | {r['ratio']}{flag} |"
        )
    lines += ["", "## Identifiability on the record", ""] + report.render_vectors(
        vectors
    )
    lines += ["", "## Residuals of each base form fitted on the record (relative)", ""]
    for f in fits:
        segs = ", ".join(
            f"{k} {v['median']:+.2%} ({v['steps']})" for k, v in f["segments"].items()
        )
        lines.append(
            f"- {f['vector']}: {f['steps']} steps, MAD {f['mad']:.2%}; per segment median: {segs}"
        )
    lines += ["", f"Verdict: **{'pass' if out['pass'] else 'fail'}**"]
    text = "\n".join(lines) + "\n"
    (a.record / "check.json").write_text(json.dumps(out, indent=1))
    (a.record / "check.md").write_text(text)
    print(text)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
