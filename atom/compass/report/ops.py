# SPDX-License-Identifier: MIT
"""An op-level run's time by op family, per step phase, from the prices its
cost lines hold.

A run `compass run --op-graph` priced writes, beside each DP rank's step
record, its cost lines (`dp<r>.cost.jsonl`): each step's terms, one per op
kind, ``op.<kind>``, ``.price_list`` added where its price list priced the
kind, and ``draft `` or ``draft_kv `` before a drafter's (`backends.op_graph`).
`split` sums them as they are, recomputing and estimating nothing, for each
side of the run (`validate.runs.sides`), by the phase of the step on its
record line: ``DP dummy`` for a DP dummy forward, else ``decode`` when every
row decodes and ``prefill`` otherwise, with ``, no output`` for a forward
that samples nothing. `rows` puts each kind in the first of `FAMILIES` naming
part of it, or in ``other``. The terms `WHOLE` names hold no one kind and stay
whole, each with why.

A phase reconciles with its predicted time, its steps' ``group_s``: each
step's terms fold to its ``device_s`` and ``draft_s``, or `split` refuses the
run, and `rows` lists the rest with its cause (`HOST`, `GROUP`). The refused
steps, by reason, are the phase's coverage: one the cost backend refused holds
no term, and one the host law refused keeps the terms it was priced at. A real
run and a step-level run have no op split, and `split` says so.
"""

import collections
import dataclasses
import json
from pathlib import Path

from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.json_file import read_json
from atom.compass.backends.op_graph import DRAFT, DRAFT_KV, TBO
from atom.compass.ir.nodes import JoinPolicy
from atom.compass.run import OP_GRAPH
from atom.compass.validate.runs import sides

#: Each family, and the parts of an op kind's name that put the kind in it;
#: the first family naming part of a kind takes it.
FAMILIES = (
    (
        "collectives",
        ("all_reduce", "allreduce", "all_gather", "reduce_scatter", "all_to_all")
        + ("broadcast", "dispatch_combine", "_c10d_functional."),
    ),
    ("MoE", ("moe", "topk_gating", "topk_softmax", "hash_topk", "router")),
    (
        "attention",
        ("attention", "attn", "mla", ".pa_", "paged", "mqa", "indexer")
        + ("kv_indices", "compress", "csa_", "swa_"),
    ),
    (
        "dense GEMM",
        ("gemm", "aten.mm.", "aten.addmm.", "aten.bmm.", "aten.baddbmm.")
        + ("aten._scaled_mm.", "aten.matmul.", "aten.linear."),
    ),
    (
        "norms and elementwise",
        ("norm", "rms", "quant", "rope", "silu", "gelu", "act_mul", "softmax")
        + ("embedding", "aten."),
    ),
)
OTHER = "other"
#: The terms that hold no one op kind, and why.
WHOLE = {
    TBO: "a two-batch-overlap step: its two micro-batches overlap on two"
    " streams and are priced as one finish time, which no op kind holds",
    f"par.{JoinPolicy.RESOURCE_BOUND.value}": "a Par whose branches share the"
    " device, priced as one roofline over their summed work, which no op kind"
    " holds",
}
#: Why a phase's predicted time holds more than its terms: the host time, its
#: steps' ``charge_s`` less their priced device time, and the DP group's slowest
#: rank, their ``group_s`` less their ``charge_s``.
HOST = (
    "the host time the run's host law composed with the device time, under DP"
    " to the group's slowest device end; below 0 where forwards return before"
    " their device work ends"
)
GROUP = "a rank whose step samples leaves at its DP group's slowest charge"


@dataclasses.dataclass
class Phase:
    """A phase's steps, its refused steps by reason, its terms' seconds by
    name, a price list's under its kind's, and its steps' ``device_s`` and
    ``draft_s`` (`priced`), ``charge_s`` (`charged`) and ``group_s``
    (`predicted`), each summed."""

    steps: int = 0
    refused: collections.Counter = dataclasses.field(
        default_factory=collections.Counter
    )
    kinds: dict = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    priced: float = 0.0
    charged: float = 0.0
    predicted: float = 0.0


def _phase(step: dict) -> str:
    if step["is_dummy_run"]:
        return "DP dummy"
    phase = "decode" if all(d for *_, d in step["rows"]) else "prefill"
    return phase if step["produces_output"] else f"{phase}, no output"


def _add(record: Path, phases: dict) -> None:
    """Add each step of `record`, a DP rank's, to its phase in `phases`."""
    cost_path = record.with_suffix(".cost.jsonl")
    with open(record, "rb") as lines, open(cost_path, "rb") as costs:
        next(costs, None)  # its version line
        for line, cost in zip(lines, costs):
            step, cost = json.loads(line), json.loads(cost)
            if cost["step"] != step["step"]:
                raise ValueError(
                    f"{cost_path}'s line of step {cost['step']} stands beside"
                    f" step {step['step']}"
                )
            target, draft = [], []
            for t in cost["terms"]:
                (draft if t["name"].startswith(DRAFT) else target).append(t["seconds"])
            folded = (fold_seconds(target), fold_seconds(draft))
            if folded != (cost["device_s"], cost["draft_s"]):
                raise ValueError(
                    f"{cost_path}'s step {step['step']}: its terms fold to {folded},"
                    f" not its device_s and draft_s"
                )
            p = phases[_phase(step)]
            p.steps += 1
            if cost["refused"]:
                p.refused[cost["refused"]] += 1
            for t in cost["terms"]:
                p.kinds[t["name"].removesuffix(".price_list")] += t["seconds"]
            p.priced += cost["device_s"] + cost["draft_s"]
            p.charged += cost["charge_s"]
            p.predicted += cost["group_s"]


def split(run) -> list[tuple[str, int, dict]]:
    """Each side of `run`: its name (none for an instance), its DP ranks, and
    its `Phase`s by phase; or ValueError, why it has no op split."""
    run = Path(run)
    run_file = read_json(run / "run.json").get("run_file")
    if run_file is None:
        raise ValueError("a real run is measured, not priced: it has no op split")
    if not run_file.get(OP_GRAPH):
        raise ValueError("a step-level run prices whole steps: it has no op split")
    out = []
    for side in sides(run):
        records = sorted((side / "steps").glob("*dp*[0-9].jsonl"))
        if not records:
            raise ValueError(f"{side / 'steps'} holds no DP rank's step record")
        phases = collections.defaultdict(Phase)
        for record in records:
            _add(record, phases)
        out.append(("" if side == run else side.name, len(records), dict(phases)))
    return out


def _bare(kind: str) -> str:
    """The term `kind`, a drafter's as its target's would be named."""
    return kind.removeprefix(DRAFT).removeprefix(DRAFT_KV)


def family(kind: str) -> str:
    """The family of the term `kind`, a drafter's as its op kind's."""
    name = _bare(kind)
    return next((f for f, parts in FAMILIES if any(p in name for p in parts)), OTHER)


def rows(phase: Phase) -> list[tuple[str, float, str]]:
    """`phase`'s parts, each its name, seconds, and its kinds, most seconds
    first, or why: the families, the `WHOLE` terms, the host time and the DP
    group's slowest rank where not 0; then its predicted seconds, which they
    sum to."""
    held, whole = collections.defaultdict(list), []
    for kind, seconds in sorted(phase.kinds.items(), key=lambda k: -k[1]):
        if _bare(kind) in WHOLE:
            whole.append((kind, seconds, WHOLE[_bare(kind)]))
        else:
            held[family(kind)].append((kind, seconds))
    out = [
        (f, sum(s for _, s in held[f]), "; ".join(f"{k} {s:.4g}" for k, s in held[f]))
        for f in (*(f for f, _ in FAMILIES), OTHER)
        if f in held
    ]
    rest = (
        ("host time", phase.charged - phase.priced, HOST),
        ("the DP group's slowest rank", phase.predicted - phase.charged, GROUP),
    )
    summed = "group_s, summed over the side's DP ranks"
    out += whole + [r for r in rest if r[1]]
    return out + [("predicted", phase.predicted, summed)]
