"""What each of the six cc-traces cells is configured with, in one place.

`cc_traces_plan.py` prints the commands a cell needs and `cc_traces_run.py`
executes them, and both have so far taken the modelled side's oracle
configuration as free text on the command line: a factory qualname and a list of
`KEY=VALUE` strings the operator types. That is the part of the matrix most
expensive to get wrong -- an artifact resolved from the wrong width prices a
different deployment and nothing in the run says so -- and it is also the part
no test could reach, because it did not exist anywhere in the tree.

So the configuration is written down here, per width, resolved against a root
directory, and the plan reads it. `SOURCE_ONLY_SERVING.md` remains the prose
account of why each option is what it is; this is the same set as data, and
`tests/compass/test_cc_traces_registry.py` holds the two against each other.

Two things are deliberately *not* defaulted:

* **Costs.** Four of the protocol's §5 terms -- `capture`, `calibration`,
  `derivation`, `load` -- are inputs no step of the harness measures. They are
  listed here as owed, with who owes them, rather than carried as a zero that
  would make `replay_ratio` read better than it is.
* **Artifacts that do not exist yet.** `required_artifacts` names every file a
  width needs; `check` reports the ones absent from the root as absent. A cell
  whose artifacts are incomplete is not runnable, and saying so before a lease
  is bought is the entire point.

    python scripts/compass/cc_traces_registry.py --root /workspace/results/poc
    python scripts/compass/cc_traces_registry.py --root ... --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atom.compass.core.artifacts import resolve_rank_path

#: The matrix, in the order a report should read.
TPS = (1, 2, 4)
CLASSES = ("short", "long")
CELLS = tuple(f"tp{tp}-{klass}" for tp in TPS for klass in CLASSES)

MODEL = "Qwen/Qwen3.8-27B"

#: The factory a served modelled run names. `_build_oracle` imports this
#: qualname and offers it `rank_coords` because its signature asks for them.
ORACLE = "atom.compass.runtime.source_oracle.source_cost_oracle"

#: Selected on the server command line rather than through the oracle: the
#: aggregation is a property of how one process stands in for a group, not of
#: the price composition. `rank0` is the parser default and would price the
#: rank that calls itself, so acceptance names the other one. What the maximum
#: is and is not is in `atom/compass/runtime/predict.py`.
RANK_AGGREGATION = "slowest"

#: Every width shares these. `allocation=native` takes the CPU scheduler's own
#: block and state assignment for the step being priced; `carry_allocation=1`
#: reuses a capture's and is inadmissible for acceptance, so it is absent here
#: and `check` refuses a configuration that reintroduces it.
SHARED_OPTIONS = (
    ("model", MODEL),
    ("device", "meta"),
    ("replay_target", "{root}/poc/g5_27b/target.json"),
    ("block_size", "16"),
    ("max_model_len", "262144"),
    ("position_rows", "3"),
    ("cudagraph_mode", "full"),
    ("head", "1"),
    ("regions", "source-27b-tp1-conc-v2"),
    ("require_complete", "1"),
    ("allocation", "native"),
    ("derive", "1"),
)

#: Options no acceptance cell may carry, with why. Checked rather than trusted:
#: a diagnostic option that survives into a matrix run is the failure mode this
#: registry exists to catch.
INADMISSIBLE = {
    "carry_allocation": (
        "reuses the template's block and state assignment and declares it "
        "unmeasured; acceptance needs the scheduler's own"
    ),
}

#: Where each width's artifacts live under the root. TP1 reads the source-width
#: files directly; TP2 and TP4 read a directory of links in
#: `resolve_rank_path`'s own convention, because the pricing artifacts are
#: named `<stem>.tp<width>.r<rank>.json` and that convention never produces it.
#: `SOURCE_ONLY_SERVING.md` has the loop that builds the links.
_TP1 = "{root}/g4/src1"
_WIDE = "{root}/serving/src_tp{tp}"


def per_width_options(tp: int) -> tuple:
    """The options this width adds to `SHARED_OPTIONS`, unresolved."""
    if tp == 1:
        body = (f"{_TP1}/p27bdec32.tp1.r0.json"
                f":{_TP1}/b27dec32.tp1.r0.json:unregistered")
        head = (f"{_TP1}/p27hdec32.tp1.r0.json"
                f":{_TP1}/h27dec32.tp1.r0.json:unregistered")
        return (
            ("tp", "1"),
            ("price", f"{body},{head}"),
            ("template", f"{_TP1}/b27dec32.tp1.r0.json"),
            ("head_template", f"{_TP1}/h27dec32.tp1.r0.json"),
        )
    body = f"{_WIDE}/p27bdec32.json:{_WIDE}/b27dec32.json:unregistered"
    head = f"{_WIDE}/p27hdec32.json:{_WIDE}/h27dec32.json:unregistered"
    # Both all-reduce lists are loaded on purpose: they hold the same signature
    # under different registration regimes and the graph selects between them,
    # so which one answers must not depend on load order.
    prices = (body, head, f"{_WIDE}/ar_capture.json",
              f"{_WIDE}/ar_plain.json", f"{_WIDE}/ag_prices.json")
    return (
        ("tp", str(tp)),
        ("price", ",".join(prices)),
        ("template", f"{_WIDE}/b27dec32.json"),
        ("head_template", f"{_WIDE}/h27dec32.json"),
    )


def options(tp: int, root) -> list:
    """The `KEY=VALUE` strings for this width, resolved against `root`."""
    root = str(Path(root))
    return [f"{key}={value.format(root=root, tp=tp)}"
            for key, value in SHARED_OPTIONS + per_width_options(tp)]


def option_paths(tp: int, root) -> dict:
    """Every file this width's options name, by the role it plays.

    Read back out of `options` rather than listed again, so the two cannot
    drift: a path that stops being an option stops being required in the same
    edit. These are the paths as *written*; what a rank actually opens is
    `resolution`, below.
    """
    found = {}
    for item in options(tp, root):
        key, _, value = item.partition("=")
        if key == "price":
            for n, spec in enumerate(value.split(",")):
                parts = spec.split(":")
                found[f"price[{n}].prices"] = parts[0]
                if len(parts) > 1 and parts[1]:
                    found[f"price[{n}].graph"] = parts[1]
        elif key in ("template", "head_template", "replay_target"):
            found[key] = value
    return found


#: Kept under the old name: `required_artifacts` is what the plan's tests and
#: the first readers of this module called it.
required_artifacts = option_paths


#: Roles whose single file answers for the whole group rather than for a rank.
#: One captured target builds the same Config everywhere, and a collective's
#: price list is a measurement of the group, not of a member: nothing writes
#: `ar_capture.tp2.json`, so asking for one and then reporting the unsuffixed
#: file as a fallback would file a claim against a file that is correct.
_GROUP_STEMS = ("ar_capture.json", "ar_plain.json", "ag_prices.json")


def _group_level(role: str, path: str) -> bool:
    return role == "replay_target" or path.endswith(_GROUP_STEMS)


def resolution(tp: int, root) -> dict:
    """What each rank of this width actually opens, and whether it is its own.

    An option names `b27dec32.json`; the rank appends its own coordinates and
    reads `b27dec32.tp2.json`, falling back to the unsuffixed file when it has
    none of its own. The fallback is legitimate -- a symmetric group's ranks
    time within a fraction of a percent -- but it is a claim, and the whole
    reason to report it here is that a run in which every rank silently read
    rank 0's artifacts looks identical to one in which each read its own.

    Group-level roles and TP=1 are excluded from that: neither can confuse one
    rank's measurement for another's, so reporting them would bury the case
    that can under dozens of lines that cannot.
    """
    out = {}
    for rank in range(tp):
        per = {}
        for role, path in option_paths(tp, root).items():
            if _group_level(role, path):
                per[role] = {"path": path, "own": True,
                             "exists": Path(path).exists()}
                continue
            if tp == 1:
                # The source-width files carry `.tp1.r0` in the written name,
                # and a group of one has no other rank to be confused with.
                per[role] = {"path": path, "own": True,
                             "exists": Path(path).exists()}
                continue
            resolved, own = resolve_rank_path(path, {"tp": rank})
            per[role] = {"path": resolved, "own": own,
                         "exists": Path(resolved).exists()}
        out[rank] = per
    return out


#: What a cc-traces run would still refuse with these artifacts in place, from
#: `SOURCE_ONLY_SERVING.md`'s own list. Every artifact resolving is necessary
#: and not sufficient: a cell with nothing absent is *configured*, not ready,
#: and these are the reasons. `closes` names what would retire each one, so a
#: cell's readiness is a question with an answer rather than a run that fails
#: at the first decode step with a running-request count of 31.
OPEN_REFUSALS = {
    "head_rows": {
        "what": "the head is priced at 32 rows and nowhere else; any other "
                "running-request count is refused, and one measured point "
                "fits nothing",
        "cells": CELLS,
        "closes": "a head price sweep over the running-request counts the "
                  "registered workloads actually reach",
        "first_met": True,
    },
    "seeded_structure": {
        "what": "only the decode-32 structure is seeded; every prefill, "
                "chunked step and other bucket needs derivation",
        "cells": CELLS,
        "closes": "derivation of the remaining structures, ~10.5 s each, "
                  "device-free",
        "first_met": False,
    },
    "derivation_is_rank0_s": {
        "what": "derivation produces rank 0's shard whatever rank asks, so "
                "at TP>1 the other ranks are served the representative",
        "cells": tuple(c for c in CELLS if not c.startswith("tp1-")),
        "closes": "per-rank derivation, or a measured bound on the "
                  "rank-to-rank spread at each width",
        "first_met": False,
    },
    "region_model_held_out": {
        "what": "the region model is held out at TP2 and TP4: it was fitted "
                "at the source width and its transfer is what G4 tests",
        "cells": tuple(c for c in CELLS if not c.startswith("tp1-")),
        "closes": "the G4 held-out transfer result, or a width-local fit "
                  "declared as such",
        "first_met": False,
    },
    "unallocatable_steps": {
        "what": "a step nobody offers an allocation for is refused by name: "
                "a mixed prefill/decode batch, a row with no state slot, and "
                "the padded tail below the capture bucket",
        "cells": CELLS,
        "closes": "nothing, for cc-traces: with TBO off the scheduler emits "
                  "no mixed batch, and the other two are refusals on purpose",
        "first_met": False,
    },
}


COST_TERMS = {
    "capture": {
        "what": "the TP=1 tracing pass that produced the graphs",
        "state": "owed",
        "from": "the source-width capture run's own log",
        "in_gate": False,
    },
    "calibration": {
        "what": "the source measurement pass (sweep / primitive pricing)",
        "state": "owed",
        "from": "the pricing acquisition's log, summed over its passes",
        "in_gate": False,
    },
    "derivation": {
        "what": "CPU derivation of this width's graphs from the TP=1 capture",
        "state": "owed",
        "from": "the device-free derivation run, per width",
        # In the denominator: deriving this candidate is work that asking the
        # question costs, unlike capture and calibration which are paid once.
        "in_gate": True,
    },
    "load": {
        "what": "weight load and graph capture inside startup",
        "state": "owed",
        "from": "the real side's startup log",
        "in_gate": False,
    },
    "startup_real": {
        "what": "process start to healthy, real side",
        "state": "measured", "from": "cc_traces_run.py", "in_gate": False,
    },
    "startup_modelled": {
        "what": "process start to healthy, modelled side",
        "state": "measured", "from": "cc_traces_run.py", "in_gate": False,
    },
    "execution_real": {
        "what": "the measured window itself, real side",
        "state": "measured", "from": "cc_traces_run.py", "in_gate": True,
    },
    "execution_modelled": {
        "what": "the measured window itself, modelled side",
        "state": "measured", "from": "cc_traces_run.py", "in_gate": True,
    },
}

OWED_TERMS = tuple(k for k, v in COST_TERMS.items() if v["state"] == "owed")


def cell_config(tp: int, klass: str, root) -> dict:
    """One cell, whole: what it is served with and what it still owes."""
    return {
        "cell": f"tp{tp}-{klass}",
        "tp": tp,
        "class": klass,
        "oracle": ORACLE,
        "oracle_options": options(tp, root),
        "rank_aggregation": RANK_AGGREGATION,
        "allocation": "native",
        "artifacts": option_paths(tp, root),
        "resolution": resolution(tp, root),
        "owed_costs": list(OWED_TERMS),
        "open_refusals": sorted(k for k, v in OPEN_REFUSALS.items()
                                if f"tp{tp}-{klass}" in v["cells"]),
    }


def check(root) -> dict:
    """Resolve every cell and say what is missing, without running anything."""
    cells = []
    for tp in TPS:
        for klass in CLASSES:
            config = cell_config(tp, klass, root)
            absent, shared = set(), set()
            for rank, roles in config["resolution"].items():
                for role, found in roles.items():
                    if not found["exists"]:
                        absent.add(f"{role}@tp{rank}")
                    elif not found["own"]:
                        # Exists, and is not this rank's: one rank's file
                        # answering for another is admissible and is a claim,
                        # so it is named rather than counted as present.
                        shared.add(f"{role}@tp{rank}")
            bad = sorted(key for key in INADMISSIBLE
                         if any(o.startswith(f"{key}=")
                                for o in config["oracle_options"]))
            config["absent_artifacts"] = sorted(absent)
            config["shared_artifacts"] = sorted(shared)
            config["inadmissible_options"] = bad
            config["runnable"] = not absent and not bad
            # Configured is not ready. A cell with every artifact resolved
            # still refuses the steps below, and the difference between the
            # two is the whole content of this report.
            config["ready"] = config["runnable"] and not config["open_refusals"]
            cells.append(config)
    return {
        "root": str(Path(root)),
        "cells": cells,
        "owed_costs": {k: COST_TERMS[k] for k in OWED_TERMS},
        "means": (
            "the configuration each cell would be served with, what is "
            "absent, and which ranks would read another rank's file; nothing "
            "was run and no cell is claimed ready"
        ),
    }


def _resolved(cell, key: str) -> str:
    role, _, rank = key.partition("@tp")
    return cell["resolution"][int(rank)][role]["path"]


def render(report: dict) -> str:
    out = [f"# six cc-traces cells under {report['root']}", ""]
    for cell in report["cells"]:
        state = ("ready" if cell["ready"] else
                 "configured, NOT ready" if cell["runnable"] else
                 "NOT runnable")
        out.append(f"## {cell['cell']} -- {state}")
        out.append(f"  oracle            {cell['oracle']}")
        out.append(f"  rank_aggregation  {cell['rank_aggregation']}")
        out.append(f"  allocation        {cell['allocation']}")
        for key in cell["absent_artifacts"]:
            out.append(f"  ABSENT  {key}: {_resolved(cell, key)}")
        for key in cell["shared_artifacts"]:
            out.append(f"  NOT THIS RANK'S  {key}: {_resolved(cell, key)}")
        for key in cell["inadmissible_options"]:
            out.append(f"  INADMISSIBLE  {key}: {INADMISSIBLE[key]}")
        for key in cell["open_refusals"]:
            term = OPEN_REFUSALS[key]
            first = " (met first)" if term["first_met"] else ""
            out.append(f"  REFUSES{first}  {key}: {term['what']}")
            out.append(f"    closed by: {term['closes']}")
        out.append("")
    out.append("## costs no step of this harness measures")
    for name, term in report["owed_costs"].items():
        where = ("in the gate denominator" if term["in_gate"]
                 else "reported beside it")
        out.append(f"  {name}: {term['what']} -- from {term['from']}, {where}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True,
                    help="directory the artifact paths resolve against")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    report = check(args.root)
    print(json.dumps(report, indent=2) if args.json else render(report))
    # Non-zero while anything is absent, so this can gate a lease rather than
    # only describe one.
    return 0 if all(c["runnable"] for c in report["cells"]) else 1


if __name__ == "__main__":
    sys.exit(main())
