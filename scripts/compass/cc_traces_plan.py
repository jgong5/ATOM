"""The exact command sequence each cc-traces acceptance cell needs, written down.

A GPU lease is bought in hours, and the way to waste one is to arrive with a
protocol and improvise the commands. `CC_TRACES_PROTOCOL.md` says what a cell
must be; this says what to type, in order, per cell, with the artifacts each
step produces and the ones the next step reads. It runs nothing: it prints a
plan, and the plan is checkable before anyone is paying for a device.

Three places a step can run, and they are not interchangeable:

* `gpu` -- the leased node, under the isolation audit. Only the real side.
* `device_free` -- a container with no `/dev/kfd` and no `/dev/dri`. The
  modelled side and its probe. The protocol's speedup claim is about what a
  prediction costs where no device could have helped, so this is a property of
  the container and not of `HIP_VISIBLE_DEVICES` (§5).
* `cpu` -- anywhere; reads artifacts only. Validation.

What this file will not do is guess. Where the protocol needs something the
repository does not yet provide -- the `rocm-smi` sampler that writes
`gpu.jsonl`, the oracle's own option wiring -- the step says so in `gaps`
rather than emitting a command that would fail at 3 a.m. on a leased node.

    python scripts/compass/cc_traces_plan.py --root /workspace/results/cc_acceptance
    python scripts/compass/cc_traces_plan.py --root ... --shell
    python scripts/compass/cc_traces_plan.py --root ... --out plan.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: The matrix. TP=1 is the source width and its cells are a fit statistic, not
#: a prediction; they are here because the residual is reported, not because
#: they test the bet.
TPS = (1, 2, 4)
CLASSES = ("short", "long")

#: Repeats per side, from the protocol's §3. Three on the modelled side too: a
#: single simulated run is not a distribution.
REPEATS = 3

MODEL = "Qwen/Qwen3.8-27B"
PORT = 8000

#: The engine configuration, unchanged from the registered matrix. Every cell
#: is served with these, and the validator reads them back out of the server's
#: own provenance rather than trusting this list.
ENGINE_ARGS = (
    "--gpu-memory-utilization",
    "0.90",
    "--max-model-len",
    "262144",
    "--no-enable_prefix_caching",
    "--max-num-seqs",
    "32",
)

#: What each cell keeps, and which step writes it. The validator refuses a cell
#: that is missing any of it, so the plan names it rather than leaving it to be
#: noticed afterwards.
EVIDENCE = {
    "real.r{n}.json": "the real side's repeat n",
    "modelled.r{n}.json": "the modelled side's repeat n",
    "real.r{n}.prepare.json": "the drained preparation records, real side",
    "real_steps.jsonl": "the real side's step table",
    "gpu.jsonl": "rocm-smi samples across the whole window",
    "isolation.json": "the isolation audit over those samples",
    "costs.json": "every cost term of §5, in seconds",
    "gpu_free.json": "the device-free observation, taken in the modelled container",
    "registry.json": "the calibration registry of §4",
    "cc_traces_protocol.json": "the protocol stamp, written by cc_traces_protocol.py",
    "cc_traces_cell.json": "the validator's verdict, kept pass or fail",
}

#: Things the protocol asks for that this repository does not yet hand over.
#: Listed here so a plan cannot read as more complete than it is.
GAPS = (
    (
        "no sampler in this tree writes gpu.jsonl; isolation.py reads it but "
        "nothing here produces it, so the sampling step is the run harness's"
    ),
    (
        "the oracle's option names are a property of the oracle chosen at run "
        "time; --oracle-option here is passed through verbatim and is not "
        "validated against the oracle's constructor"
    ),
    (
        "warmup_seconds is not in the server's /compass/provenance, so 'the "
        "modelled side was given no warmup' is checked from the replay "
        "artifact's own manifest and not from the server"
    ),
    (
        "costs.json is filled in by whoever runs the cell; no step here "
        "measures capture, calibration or derivation seconds for it"
    ),
)


def _serve(tp: int, *, modelled: bool, oracle: str | None, options, port: int):
    """The server command for one side of one cell."""
    cmd = [
        "python",
        "-m",
        "atom.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--port",
        str(port),
        "-tp",
        str(tp),
        *ENGINE_ARGS,
        "--compass",
        "--compass-mode",
        "predict" if modelled else "measure",
    ]
    if modelled:
        if oracle:
            cmd += ["--compass-oracle", oracle]
        for option in options:
            cmd += ["--compass-oracle-option", option]
    else:
        cmd += ["--compass-measure-out", "real_steps.jsonl"]
    return cmd


def _replay(klass: str, out: str, *, paced: bool, prepare: int, port: int):
    cmd = [
        "python",
        "scripts/compass/replay.py",
        "--port",
        str(port),
        "--model",
        MODEL,
        "--trace",
        f"atom/compass/cc_traces_{klass}.jsonl",
        "--out",
        out,
        "--check-lengths",
    ]
    if paced:
        # The real engine stamps arrivals on receipt, so a declared arrival is
        # discarded there: without --pace the real side answers a burst while
        # the modelled side answers the trace, and the two are not comparable.
        cmd += ["--pace"]
    if prepare:
        cmd += [
            "--prepare",
            str(prepare),
            "--prepare-out",
            out.replace(".json", ".prepare.json"),
        ]
    return cmd


def cell_steps(
    tp: int, klass: str, *, root: str, oracle, options, port: int, repeats: int
):
    """Every step of one cell, in the order it has to happen."""
    cell = f"{root.rstrip('/')}/tp{tp}_{klass}"
    steps = [
        {
            "id": "stamp",
            "where": "cpu",
            "why": "a cell that does not say what it ran under is not a cell",
            "command": [
                "python",
                "scripts/compass/cc_traces_protocol.py",
                "stamp",
                cell,
            ],
            "produces": ["cc_traces_protocol.json"],
        },
        {
            "id": "verify-workload",
            "where": "cpu",
            "why": "the registered bytes, re-derived from the rule, before the lease is spent",
            "command": [
                "python",
                "scripts/compass/cc_traces_workload.py",
                "verify",
                "--manifest",
                f"atom/compass/cc_traces_{klass}.manifest.json",
                "--corpus",
                "$CC_TRACES_CORPUS",
            ],
            "produces": [],
        },
        {
            "id": "sample-devices",
            "where": "gpu",
            "why": "isolation is a property of the whole window, not of two instants",
            "command": None,
            "provided_by": "the run harness; see gaps",
            "produces": ["gpu.jsonl"],
        },
        {
            "id": "serve-real",
            "where": "gpu",
            "why": "the measured side, at this cell's width",
            "command": _serve(tp, modelled=False, oracle=None, options=(), port=port),
            "produces": ["real_steps.jsonl", "server.real.log"],
            "background": True,
        },
    ]
    for n in range(1, repeats + 1):
        steps.append(
            {
                "id": f"replay-real-{n}",
                "where": "gpu",
                "why": "one repeat, from a prepared and drained engine, paced to the trace",
                "command": _replay(
                    klass,
                    f"{cell}/real.r{n}.json",
                    paced=True,
                    prepare=3,
                    port=port,
                ),
                "produces": [f"real.r{n}.json", f"real.r{n}.prepare.json"],
            }
        )
    steps += [
        {
            "id": "stop-real",
            "where": "gpu",
            "why": "the real side's window ends before the audit is read",
            "command": None,
            "provided_by": "the run harness; see gaps",
            "produces": [],
        },
        {
            "id": "isolation",
            "where": "cpu",
            "why": "who else was on the node while this was measured",
            "command": [
                "python",
                "scripts/compass/isolation.py",
                f"{cell}/gpu.jsonl",
                "--json",
                f"{cell}/isolation.json",
            ],
            "produces": ["isolation.json"],
        },
        {
            "id": "serve-modelled",
            "where": "device_free",
            "why": "the prediction, where no device could have helped it",
            "command": _serve(
                tp, modelled=True, oracle=oracle, options=options, port=port
            ),
            "produces": ["server.modelled.log"],
            "background": True,
        },
    ]
    for n in range(1, repeats + 1):
        steps.append(
            {
                "id": f"replay-modelled-{n}",
                "where": "device_free",
                "why": "the same trace, declared arrivals, no pacing and no preparation",
                "command": _replay(
                    klass,
                    f"{cell}/modelled.r{n}.json",
                    paced=False,
                    prepare=0,
                    port=port,
                ),
                "produces": [f"modelled.r{n}.json"],
            }
        )
    steps += [
        {
            "id": "gpu-free",
            "where": "device_free",
            "why": "the device-free claim, observed in the container that made the prediction",
            "command": [
                "python",
                "scripts/compass/cc_traces_validate.py",
                "gpu-free",
                cell,
            ],
            "produces": ["gpu_free.json"],
        },
        {
            "id": "costs",
            "where": "cpu",
            "why": "every term of §5, separately; the gate reads two of them and reports the rest",
            "command": None,
            "provided_by": "whoever ran the cell; see gaps",
            "produces": ["costs.json"],
        },
        {
            "id": "validate",
            "where": "cpu",
            "why": "the verdict, kept whether it passes or not",
            "command": [
                "python",
                "scripts/compass/cc_traces_validate.py",
                "cell",
                cell,
                "--class",
                klass,
                "--tp",
                str(tp),
                "--repeats",
                str(repeats),
                "--calibration-registry",
                f"{cell}/registry.json",
            ],
            "produces": ["cc_traces_cell.json"],
        },
    ]
    return {"cell": cell, "tp": tp, "class": klass, "steps": steps}


def build(args) -> dict:
    cells = [
        cell_steps(
            tp,
            klass,
            root=args.root,
            oracle=args.oracle,
            options=args.oracle_option,
            port=args.port,
            repeats=args.repeats,
        )
        for tp in TPS
        for klass in CLASSES
    ]
    return {
        "protocol": "atom/compass/CC_TRACES_PROTOCOL.md",
        "root": args.root,
        "model": MODEL,
        "engine_args": list(ENGINE_ARGS),
        "repeats": args.repeats,
        "cells": cells,
        "matrix": [
            "python",
            "scripts/compass/cc_traces_validate.py",
            "matrix",
            *[c["cell"] for c in cells],
            "--out",
            f"{args.root.rstrip('/')}/verdict.json",
        ],
        "evidence": dict(EVIDENCE),
        "gaps": list(GAPS),
        "source_width": 1,
        "means": (
            "the commands a cell needs, in order; running them is not what this "
            "does and no result here is claimed"
        ),
    }


def render(plan: dict) -> str:
    """The plan as something a person can read next to a terminal."""
    out = [
        f"# {plan['protocol']}: {len(plan['cells'])} cells, {plan['repeats']} repeats a side",
        "",
    ]
    for cell in plan["cells"]:
        out.append(f"## {cell['cell']}  (TP={cell['tp']}, {cell['class']})")
        for step in cell["steps"]:
            where = step["where"]
            if step.get("command"):
                line = " ".join(shlex.quote(part) for part in step["command"])
                if step.get("background"):
                    line += "   &   # leave running for this cell"
                out.append(f"  [{where}] {line}")
            else:
                out.append(f"  [{where}] ({step['id']}: {step['provided_by']})")
        out.append("")
    out.append("## the matrix, once every cell has a verdict")
    out.append("  [cpu] " + " ".join(shlex.quote(p) for p in plan["matrix"]))
    out.append("")
    out.append("## what this plan does not provide")
    for gap in plan["gaps"]:
        out.append(f"  - {gap}")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="where the cells' directories live")
    ap.add_argument(
        "--oracle", default=None, help="the cost oracle the modelled side runs"
    )
    ap.add_argument(
        "--oracle-option",
        action="append",
        default=[],
        help="passed through to --compass-oracle-option, repeatable",
    )
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--out", default=None, help="write the plan as JSON here")
    ap.add_argument(
        "--shell", action="store_true", help="print the commands instead of JSON"
    )
    args = ap.parse_args(argv)
    plan = build(args)
    if args.out:
        Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    if args.shell:
        print(render(plan), end="")
    elif not args.out:
        print(json.dumps(plan, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
