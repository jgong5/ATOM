"""The exact command sequence each cc-traces acceptance cell needs, written down.

A GPU lease is bought in hours, and the way to waste one is to arrive with a
protocol and improvise the commands. `CC_TRACES_PROTOCOL.md` says what a cell
must be; this says what to type, in order, per cell, with the artifacts each
step produces and the ones the next step reads. It runs nothing itself --
`cc_traces_run.py` executes this plan, and both read the same `cell_steps()`, so
what is printed for review is what is run.

Three places a step can run, and they are not interchangeable:

* `gpu` -- the leased node, under the isolation audit. Only the real side.
* `device_free` -- a container with no `/dev/kfd` and no `/dev/dri`. The
  modelled side and its probe. The protocol's speedup claim is about what a
  prediction costs where no device could have helped, so this is a property of
  the container and not of `HIP_VISIBLE_DEVICES` (§5).
* `cpu` -- anywhere; reads artifacts only. Validation.

**A repeat is a process, not a request batch.** §3 registers three real and
three modelled repeats *each from a fresh process*, so every repeat here starts
its own server and stops it again. On the modelled side that is not bookkeeping:
a predicting server holds a virtual clock frozen at its epoch, and a second
replay against the same process is stamped against an origin the first replay
has already moved -- the +71% TTFT error that `replay.py` refuses a warmed
predictor for. Reuse is only correct against an engine-owned run boundary that
resets that origin, and this tree has none.

What this file will not do is guess. Where the protocol needs something the
repository does not yet provide -- the oracle's own option wiring, the
acquisition seconds -- the step says so in `gaps` rather than emitting a command
that would fail at 3 a.m. on a leased node.

    python scripts/compass/cc_traces_plan.py --root /workspace/results/cc_acceptance
    python scripts/compass/cc_traces_plan.py --root ... --shell
    python scripts/compass/cc_traces_plan.py --root ... --out plan.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """A sibling script as a module, the way `cc_traces_run.py` loads them."""
    path = ROOT / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The six cells' configuration as data. The plan used to take the modelled
#: side's oracle and its dozen options as free text on the command line, which
#: is the part of a matrix run most expensive to mistype and the part no test
#: could reach. Naming `--artifact-root` resolves them from here instead.
registry = _load("cc_traces_registry")


def _registry_oracle(args):
    return registry.ORACLE if getattr(args, "artifact_root", None) else None


def _registry_options(args, tp: int) -> list:
    root = getattr(args, "artifact_root", None)
    return registry.options(tp, root) if root else []


def _registry_target(args, tp: int):
    root = getattr(args, "artifact_root", None)
    return registry.replay_target(tp, root) if root else None


def _registry_profile(args, tp: int):
    root = getattr(args, "artifact_root", None)
    return registry.memory_model(tp, root) if root else None


def width_map(values, flag: str) -> dict:
    """`TP=PATH` pairs read into a dict keyed by width.

    A bare path is refused. Neither of these files is shared across the
    matrix: a target record is of the width it was captured or derived at --
    the replay runner takes the block and state capacities straight out of it
    and refuses a width it was not made for -- and a memory profile is of the
    width it was taken at. One path here would therefore name the wrong file
    for two of the three widths, which is exactly the mistake this plan
    exists to stop someone making by hand.
    """
    out = {}
    for value in values or ():
        width, sep, path = value.partition("=")
        if not sep or not path or not width.isdigit():
            raise SystemExit(
                f"{flag} takes TP=PATH, not {value!r}: the three widths do "
                f"not share one file, so a bare path would be handed to all "
                f"of them"
            )
        out[int(width)] = path
    return out


#: The matrix. TP=1 is the source width and its cells are a fit statistic, not
#: a prediction; they are here because the residual is reported, not because
#: they test the bet.
TPS = (1, 2, 4)
CLASSES = ("short", "long")

#: Repeats per side, from the protocol's §3. Three on the modelled side too: a
#: single simulated run is not a distribution.
REPEATS = 3

MODEL = "Qwen/Qwen3.8-27B"

#: The HTTP listener: what the health check polls and the replay client dials.
#: `--server-port` on the entry point's parser.
PORT = 8000

#: The engine's internal rendezvous port -- `--port` on that same parser, and
#: what `model_runner.py` exports as `MASTER_PORT` for torch.distributed. It is
#: not the listener and it is not free for the taking either: two servers alive
#: at once on one host collide on it even when their listeners differ, and at
#: TP>1 the collision is between rendezvous groups, so the second server can
#: join the first one's. Chosen explicitly per run rather than left to the
#: engine's default, and checked for a conflict beside the listener.
ENGINE_PORT = 8006

#: Seconds between `rocm-smi` samples across the real side's whole window. Five
#: is short enough that `isolation.py`'s two-sample sustained rule needs ten
#: seconds of a neighbour to call it busy, and long enough that the sampler is
#: not itself load.
SAMPLE_INTERVAL = 5.0

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
    "real.r{n}.json": "the real side's repeat n, from its own server process",
    "modelled.r{n}.json": "the modelled side's repeat n, from its own process",
    "real.r{n}.prepare.json": "the drained preparation records, real side",
    "provenance.real.r{n}.json": "what that server said it was, read when it came up",
    "provenance.modelled.r{n}.json": "the same for the modelled repeat",
    "real.r{n}_steps.jsonl": "that repeat's step table, from that repeat's server",
    "gpu.jsonl": "rocm-smi samples across the whole window, baseline first",
    "isolation.json": "the isolation audit over those samples",
    "costs.real.json": "the seconds the real side measured of itself",
    "costs.modelled.json": "the seconds the modelled side measured of itself",
    "costs.json": "every cost term of §5, in seconds, merged",
    "gpu_free.json": "the device-free observation, taken in the modelled container",
    "registry.json": "the calibration registry of §4",
    "cc_traces_protocol.json": "the protocol stamp, written by cc_traces_protocol.py",
    "run.real.json": "the real side's journal: every command, pid and exit status",
    "run.modelled.json": "the same for the modelled side",
    "cc_traces_cell.json": "the validator's verdict, kept pass or fail",
}

#: Things the protocol asks for that this repository does not yet hand over.
#: Listed here so a plan cannot read as more complete than it is.
GAPS = (
    (
        "the oracle's option names are a property of the oracle chosen at run "
        "time; --oracle-option here is passed through verbatim and is not "
        "validated against the oracle's constructor"
    ),
    (
        "the modelled side needs a target record of its own width to build a "
        "Config without a device, and at TP=2 and TP=4 a memory profile to "
        "size the pool from; both are inputs here, derived by the memory "
        "model, and no step in this plan creates either"
    ),
    (
        "warmup_seconds is not in the server's /compass/provenance, so 'the "
        "modelled side was given no warmup' is checked from the replay "
        "artifact's own manifest and not from the server"
    ),
    (
        "capture, calibration, derivation and load seconds are inputs to the "
        "costs step: nothing here measures them, and the cell is refused until "
        "they are supplied"
    ),
)


def _serve(
    tp: int,
    n: int,
    *,
    modelled: bool,
    oracle,
    options,
    port: int,
    engine_port: int,
    cell: str,
    target,
    memory_model=None,
):
    """The server command for one side of one cell.

    The modelled side goes through `replay_server.py` rather than the module
    entry point: AITER asks the driver for the chip at import, which on a
    machine with no device fails before any flag could be parsed, so the
    target record has to answer that question first.
    """
    if modelled:
        cmd = [
            "python",
            "scripts/compass/replay_server.py",
            "--compass-replay-target",
            target or "$CC_TRACES_REPLAY_TARGET",
        ]
        if memory_model:
            # What the pool is sized from at a width with no captured record
            # of its own. The replay runner publishes the budget it used as
            # `source-derived` when this is given and `captured` when it is
            # not, so the flag is also what makes the capacity attributable.
            cmd += ["--compass-memory-model", memory_model]
    else:
        cmd = ["python", "-m", "atom.entrypoints.openai.api_server"]
    cmd += [
        "--model",
        MODEL,
        # The HTTP listener, which is what the health check and the replay
        # client talk to. `--port` on this parser is a different thing -- the
        # engine's internal port -- so naming it here left the listener on its
        # own default and the harness waiting on a port nothing ever bound.
        "--server-port",
        str(port),
        # The engine's internal rendezvous port, named because leaving it at
        # the engine's default means every server on this host asks for 8006.
        "--port",
        str(engine_port),
        "-tp",
        str(tp),
        *ENGINE_ARGS,
        "--compass",
        "--compass-mode",
        "predict" if modelled else "measure",
    ]
    if modelled:
        # Named here rather than left at the parser default. One process stands
        # in for the whole group, so `rank0` would price the rank it calls
        # itself and quietly drop the TP4 rank-1 outlier -- the ranks are not
        # symmetric, and nothing in this plan establishes that they are. This
        # side is therefore evaluated on every logical rank; `rank_aggregation`
        # in each row says what the number is and in which direction it
        # approximates.
        cmd += ["--compass-rank-aggregation", registry.RANK_AGGREGATION]
        if oracle:
            cmd += ["--compass-oracle", oracle]
        for option in options:
            cmd += ["--compass-oracle-option", option]
    else:
        # Per repeat: the engine opens this path with "w", so three repeats
        # sharing one name would leave one table and two overwritten ones.
        cmd += ["--compass-measure-out", f"{cell}/real.r{n}_steps.jsonl"]
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


def _lifecycle(
    side: str,
    n: int,
    *,
    tp: int,
    klass: str,
    cell: str,
    where: str,
    port: int,
    engine_port: int,
    oracle,
    options,
    target,
    memory_model=None,
):
    """One repeat: its own server, its replay, and the end of that process."""
    modelled = side == "modelled"
    return [
        {
            "id": f"serve-{side}-{n}",
            "role": "serve",
            "side": side,
            "repeat": n,
            "where": where,
            "why": (
                "the prediction, from a process that has predicted nothing yet"
                if modelled
                else "the measured side, at this cell's width"
            ),
            "command": _serve(
                tp,
                n,
                modelled=modelled,
                oracle=oracle,
                options=options,
                port=port,
                engine_port=engine_port,
                cell=cell,
                target=target,
                memory_model=memory_model,
            ),
            "background": True,
            "health": f"http://127.0.0.1:{port}/health",
            # Named by scope, and by the flag that carries each one, so the
            # harness checks the right port for a conflict and the manifest
            # does not have to guess which of the two "port" means.
            "ports": {
                "http_listener": {"port": port, "flag": "--server-port"},
                "engine_rendezvous": {"port": engine_port, "flag": "--port"},
            },
            "produces": (
                [f"server.{side}.r{n}.log", f"provenance.{side}.r{n}.json"]
                + ([] if modelled else [f"real.r{n}_steps.jsonl"])
            ),
        },
        {
            "id": f"replay-{side}-{n}",
            "role": "replay",
            "side": side,
            "repeat": n,
            "where": where,
            "why": (
                "the same trace, declared arrivals, no pacing and no preparation"
                if modelled
                else "one repeat, from a prepared and drained engine, paced to the trace"
            ),
            "command": _replay(
                klass,
                f"{cell}/{side}.r{n}.json",
                paced=not modelled,
                prepare=0 if modelled else 3,
                port=port,
            ),
            "produces": (
                [f"{side}.r{n}.json"]
                + ([] if modelled else [f"{side}.r{n}.prepare.json"])
            ),
        },
        {
            "id": f"stop-{side}-{n}",
            "role": "stop",
            "side": side,
            "repeat": n,
            "where": where,
            "stops": f"serve-{side}-{n}",
            "why": (
                "the next repeat is a fresh process: a predicting server's "
                "virtual epoch is fixed when it starts, so a second replay "
                "against it is stamped against an origin the first one moved"
                if modelled
                else "the next repeat is a fresh process, as §3 registers"
            ),
            "command": None,
            "provided_by": "the run harness, by signalling the process it started",
            "produces": [],
        },
    ]


def cell_steps(
    tp: int,
    klass: str,
    *,
    root: str,
    oracle,
    options,
    port: int,
    repeats: int,
    engine_port: int = ENGINE_PORT,
    target=None,
    memory_model=None,
    corpus: str = "$CC_TRACES_CORPUS",
):
    """Every step of one cell, in the order it has to happen."""
    if port == engine_port:
        raise SystemExit(
            f"the HTTP listener and the engine's rendezvous port are both "
            f"{port}: they are two different sockets on one host and the "
            f"server cannot bind one of them"
        )
    cell = f"{root.rstrip('/')}/tp{tp}_{klass}"
    steps = [
        {
            "id": "stamp",
            "role": "command",
            "side": "both",
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
            "role": "command",
            "side": "both",
            "where": "cpu",
            "why": "the registered bytes, re-derived from the rule, before the lease is spent",
            "command": [
                "python",
                "scripts/compass/cc_traces_workload.py",
                "verify",
                "--manifest",
                f"atom/compass/cc_traces_{klass}.manifest.json",
                "--workload",
                f"atom/compass/cc_traces_{klass}.jsonl",
                "--corpus",
                corpus,
            ],
            "produces": [],
        },
        {
            "id": "sample-baseline",
            "role": "command",
            "side": "real",
            "where": "gpu",
            "why": (
                "the one sample that can show a card was already somebody "
                "else's: after our server starts, every byte on our cards is "
                "ours"
            ),
            "command": [
                "python",
                "scripts/compass/gpu_sampler.py",
                f"{cell}/gpu.jsonl",
                "--once",
                "--phase",
                "baseline",
            ],
            "produces": ["gpu.jsonl"],
        },
        {
            "id": "sample",
            "role": "sample",
            "side": "real",
            "where": "gpu",
            "why": "isolation is a property of the whole window, not of two instants",
            "command": [
                "python",
                "scripts/compass/gpu_sampler.py",
                f"{cell}/gpu.jsonl",
                "--interval",
                str(SAMPLE_INTERVAL),
                "--phase-file",
                f"{cell}/phase.json",
            ],
            "background": True,
            "produces": ["gpu.jsonl"],
        },
    ]
    for n in range(1, repeats + 1):
        steps += _lifecycle(
            "real",
            n,
            tp=tp,
            klass=klass,
            cell=cell,
            where="gpu",
            port=port,
            engine_port=engine_port,
            oracle=None,
            options=(),
            target=None,
            memory_model=None,
        )
    steps += [
        {
            "id": "stop-sample",
            "role": "stop",
            "side": "real",
            "where": "gpu",
            "stops": "sample",
            "why": "the window the audit covers ends with the last real repeat",
            "command": None,
            "provided_by": "the run harness, by signalling the sampler it started",
            "produces": [],
        },
        {
            "id": "isolation",
            "role": "command",
            "side": "real",
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
    ]
    for n in range(1, repeats + 1):
        steps += _lifecycle(
            "modelled",
            n,
            tp=tp,
            klass=klass,
            cell=cell,
            where="device_free",
            port=port,
            engine_port=engine_port,
            oracle=oracle,
            options=options,
            target=target,
            memory_model=memory_model,
        )
    steps += [
        {
            "id": "gpu-free",
            "role": "command",
            "side": "modelled",
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
            "role": "command",
            "side": "both",
            "where": "cpu",
            "why": "every term of §5, separately; the gate reads two of them and reports the rest",
            "command": [
                "python",
                "scripts/compass/cc_traces_run.py",
                "costs",
                cell,
                "--capture",
                "$CC_TRACES_CAPTURE_S",
                "--calibration",
                "$CC_TRACES_CALIBRATION_S",
                "--derivation",
                "$CC_TRACES_DERIVATION_S",
                "--load",
                "$CC_TRACES_LOAD_S",
            ],
            "produces": ["costs.json"],
        },
        {
            "id": "validate",
            "role": "command",
            "side": "both",
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
    # Resolved per width, not once: what is handed over on the command line
    # overrides the registry, and neither can cover a width it was not named
    # for.
    targets = width_map(getattr(args, "replay_target", None), "--replay-target")
    profiles = width_map(getattr(args, "memory_model", None), "--memory-model")
    cells = [
        cell_steps(
            tp,
            klass,
            root=args.root,
            oracle=args.oracle or _registry_oracle(args),
            options=args.oracle_option or _registry_options(args, tp),
            port=args.port,
            engine_port=args.engine_port,
            repeats=args.repeats,
            target=targets.get(tp) or _registry_target(args, tp),
            memory_model=profiles.get(tp) or _registry_profile(args, tp),
            corpus=getattr(args, "corpus", None) or "$CC_TRACES_CORPUS",
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
        "processes_per_side": args.repeats,
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
            "the commands a cell needs, in order; `cc_traces_run.py` is what "
            "runs them, and no result is claimed here"
        ),
    }


def render(plan: dict) -> str:
    """The plan as something a person can read next to a terminal."""
    out = [
        (
            f"# {plan['protocol']}: {len(plan['cells'])} cells, "
            f"{plan['repeats']} repeats a side, one process each"
        ),
        "",
    ]
    for cell in plan["cells"]:
        out.append(f"## {cell['cell']}  (TP={cell['tp']}, {cell['class']})")
        for step in cell["steps"]:
            where = step["where"]
            if step.get("command"):
                line = " ".join(shlex.quote(part) for part in step["command"])
                if step.get("background"):
                    line += (
                        "   &   # the audit window's sampler"
                        if step["role"] == "sample"
                        else "   &   # this repeat's own process"
                    )
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
    ap.add_argument(
        "--replay-target",
        action="append",
        default=[],
        metavar="TP=PATH",
        help=("the target record the modelled server of that width builds "
              "its Config from; repeatable, one per width, overrides the "
              "registry"),
    )
    ap.add_argument(
        "--memory-model",
        action="append",
        default=[],
        metavar="TP=PATH",
        help=("the memory profile that width sizes its pool from; "
              "repeatable, one per width, overrides the registry"),
    )
    ap.add_argument(
        "--corpus", default=None, help="the cc-traces corpus to verify against"
    )
    ap.add_argument(
        "--port", type=int, default=PORT, help="the HTTP listener (--server-port)"
    )
    ap.add_argument(
        "--engine-port",
        type=int,
        default=ENGINE_PORT,
        help="the engine's internal rendezvous port (--port on the engine)",
    )
    ap.add_argument(
        "--artifact-root",
        default=None,
        help=("resolve the modelled side's oracle and its options from "
              "cc_traces_registry.py against this directory, instead of "
              "typing them; --oracle/--oracle-option still override"),
    )
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
