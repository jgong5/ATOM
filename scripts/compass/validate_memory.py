"""Each memory term against its own ground truth, one row per term.

The derived budget was first checked as a sum: weights plus activations against
`peak_torch`, which came out +13.8%. A sum of terms validated only in total is
the shape of error this project has already been caught by twice -- two terms
wrong in opposite directions read as one term slightly wrong. So each term gets
its own recorded counterpart here, and none of them is a subtraction of the
others.

What each row compares:

* **weights** -- derived from the checkpoint headers, against the allocator
  read once the weights were resident and before any forward ran.
* **persistent** -- not modelled. The forward buffers and anything else that
  outlives a step; reported so it is visible rather than absorbed into a
  neighbour. Its recorded value is `current_torch - weights_torch`.
* **activations** -- derived by walking the traced graph for liveness, against
  `peak_torch - current_torch`. Only comparable when the trace and the run's
  warmup prefill are the same shape, which the script checks and says.
* **non-torch** -- not modelled. Recorded directly; the table over several
  topologies is what a model would have to fit.
* **graph pool** -- derived from the engine's own formula without a device,
  against the recorded estimate, and separately against the pool the capture
  loop measured if a log is given. The first says the derivation reproduces the
  engine's decision; only the second says the decision was right.

    python scripts/compass/validate_memory.py compass_ops/mem_*.json \
        [--graph compass_ops/g.prefill.json] [--checkpoint DIR] [--log run.log]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from atom.compass.core.memory_model import (  # noqa: E402
    activation_curve, graph_pool_bytes, peak_activation_bytes, weight_bytes)

GB = float(1 << 30)


def warmup_tokens(config: dict, max_num_batched_tokens: int) -> int:
    """The token count of the prefill that sets `peak_torch`.

    `ModelRunner.warmup_model` resets the peak, then runs one dummy prefill --
    so the peak activation reading belongs to that shape and no other. Mirrored
    here at data parallel one, which is every configuration recorded so far.
    """
    max_model_len = int(config.get("max_model_len") or 0)
    max_num_seqs = int(config.get("max_num_seqs") or 1)
    if not (max_model_len and max_num_batched_tokens):
        return 0
    num_seqs = max(1, min(max_num_batched_tokens // max_model_len, max_num_seqs))
    seq_len = max(1, min(max_model_len, max_num_batched_tokens // num_seqs))
    return num_seqs * seq_len


def graph_tokens(graph: dict) -> int:
    """How many tokens the traced step ran.

    `batch_signature` is the per-sequence scheduled-token count kept exact, so
    its sum is the step's token count.
    """
    key = graph.get("key") or {}
    return sum(int(n) for n in (key.get("batch_signature") or ()))


def measured_pool(log_path: str) -> int:
    """The pool the capture loop measured, off the line it logs."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return 0
    found = re.findall(r"pool\(reserved\)=([0-9.]+)GB", text)
    return int(float(found[-1]) * GB) if found else 0


def row(name: str, derived, recorded, note: str = "") -> None:
    def show(value):
        return "       -" if value is None else "%7.3fG" % (value / GB)
    if derived is None or not recorded:
        error = "     -"
    else:
        error = "%+5.1f%%" % ((derived - recorded) / recorded * 100)
    print("  %-14s %8s %8s  %6s  %s" % (name, show(derived), show(recorded),
                                        error, note))


def write_calibration(records, path: str) -> None:
    """The collective constants this box actually shows, per world size.

    The minimum across the ranks of one run, not the mean: `non_torch` is
    device-wide used memory minus torch's reserve, so a neighbour inflates it
    and never deflates it. The least contaminated rank is the closest thing to
    the configuration's own share.

    These do not transfer between boxes -- which is the whole reason for
    writing them rather than shipping a table -- so a calibration is only worth
    what the box it was taken on is worth.
    """
    per_width: dict = {}
    for _, config, readings, tp in records:
        parameters = readings.get("parameter_bytes")
        allocated = readings.get("weights_torch")
        if parameters is None or allocated is None:
            continue
        seen = per_width.setdefault(tp, {"non_torch": [], "load_residue": [],
                                         "persistent": []})
        seen["non_torch"].append(int(readings.get("non_torch") or 0))
        seen["load_residue"].append(int(allocated - parameters))
        current = readings.get("current_torch")
        if current is not None:
            seen["persistent"].append(int(current - allocated))
    blob = {
        "non_torch": {str(w): min(v["non_torch"]) for w, v in per_width.items()
                      if v["non_torch"]},
        "load_residue": {str(w): min(v["load_residue"])
                         for w, v in per_width.items() if v["load_residue"]},
        # Flat in width -- 85.2 MiB on the 0.6B at widths 1, 2, 4 and 8, and
        # 117.2 MiB on the 27B at 2 and 4 -- so one number per model, taken as
        # the largest seen rather than the smallest: this one is the engine's
        # own buffers, uncontaminated, and under-reserving it over-allocates KV.
        "persistent": max((max(v["persistent"]) for v in per_width.values()
                           if v["persistent"]), default=0),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1)
    print("\ncalibration written to %s" % path)
    for width in sorted(int(w) for w in blob["non_torch"]):
        print("  world size %-2d  non_torch %8.1f MiB   residue %8.1f MiB"
              % (width, blob["non_torch"][str(width)] / (1 << 20),
                 blob["load_residue"].get(str(width), 0) / (1 << 20)))
    print("  persistent      %8.1f MiB (flat in width)"
          % (blob["persistent"] / (1 << 20)))


def show_curve(graph: dict, worst: int = 12) -> None:
    """Where the walk and the allocator part company, operator by operator.

    Comparing peaks says the activation model is wrong. Comparing curves says
    where -- and the operators worth looking at are the ones that *introduce*
    a divergence, not the ones that inherit it, so this ranks by the step
    change rather than by the running difference.
    """
    provenance = graph.get("provenance") or {}
    measured = provenance.get("allocated_after_bytes") or []
    baseline = provenance.get("allocated_before_bytes")
    if not measured or baseline is None:
        print("  no allocator curve in this graph; re-trace to record one")
        return
    ops = graph["ops"]
    derived = activation_curve(graph)
    n = min(len(ops), len(derived), len(measured))
    if not n:
        return

    # A reading per operator the dispatch tracer recorded; the Triton tracer
    # adds operators of its own to the same graph and takes no reading, so
    # those positions are empty and simply have nothing to compare.
    steps = []
    previous = 0
    for index in range(n):
        if measured[index] is None:
            continue
        divergence = derived[index] - (measured[index] - baseline)
        steps.append((divergence - previous, index, divergence))
        previous = divergence
    if not steps:
        print("  no comparable readings in this graph")
        return

    print("\n  the walk against the allocator, operator by operator")
    print("  %5s %10s %10s  %s" % ("op", "introduced", "running", "operator"))
    for change, index, divergence in sorted(steps, key=lambda s: -abs(s[0]))[:worst]:
        op = ops[index]
        print("  %5d %+9.3fM %+9.3fM  %-38s out=%s"
              % (index, change / (1 << 20), divergence / (1 << 20),
                 op.get("name"), op.get("output_shapes")))
    print("  %5s %10s %+9.3fM  at the last operator"
          % ("", "", steps[-1][2] / (1 << 20)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", help="memory records, globs allowed")
    ap.add_argument("--graph", help="traced op graph, for the activation term")
    ap.add_argument("--checkpoint", help="model directory, for the weight term")
    ap.add_argument("--log", help="run log, for the measured graph pool")
    ap.add_argument("--max-num-batched-tokens", type=int, default=0,
                    help="not in the record; needed to know the warmup shape")
    ap.add_argument("--curve", action="store_true",
                    help="show where the walk and the allocator diverge")
    ap.add_argument("--calibrate",
                    help="write the collective constants measured from these "
                         "records to this path, for memory_model to read")
    args = ap.parse_args()

    paths = [q for p in args.records for q in sorted(glob.glob(p))] or args.records
    graph = json.load(open(args.graph)) if args.graph else None
    pool_seen = measured_pool(args.log) if args.log else 0

    non_torch_seen = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        readings, config = blob.get("readings") or {}, blob.get("config") or {}
        if not readings:
            continue
        tp = int((config.get("topology") or {}).get("tp", 1) or 1)
        non_torch_seen.append((os.path.basename(path), config, readings, tp))

    print("  %-14s %8s %8s  %6s  %s"
          % ("term", "derived", "recorded", "error", "note"))
    for name, config, readings, tp in non_torch_seen:
        print("\n%s  --  %s tp=%d max_model_len=%s"
              % (name, config.get("model"), tp, config.get("max_model_len")))

        allocated = readings.get("weights_torch")
        parameters = readings.get("parameter_bytes")
        current = readings.get("current_torch")
        peak = readings.get("peak_torch")

        checkpoint = args.checkpoint
        derived_weights = weight_bytes(checkpoint, tp) if checkpoint else None
        # Against the model's own parameters, which is what the term claims to
        # be -- not against the allocator after loading, which is that plus
        # whatever the loader still holds, and not against the buffers either,
        # which the checkpoint does not contain.
        buffers = readings.get("buffer_bytes")
        weights_seen = (parameters - buffers
                        if parameters is not None and buffers is not None
                        else parameters)
        row("weights", derived_weights, weights_seen,
            "" if parameters else "record predates the split")
        row("model buffers", None, buffers,
            "not modelled; built at init, absent from the checkpoint")

        residue = (allocated - parameters
                   if allocated is not None and parameters is not None else None)
        row("load residue", None, residue,
            "not modelled; held after load, beyond the parameters")

        # Everything resident at sizing time that is neither the parameters nor
        # a step's activations: forward buffers, and any residue still held.
        persistent = (current - parameters
                      if current is not None and parameters is not None else None)
        row("persistent", None, persistent, "not modelled")

        derived_act = peak_activation_bytes(graph) if graph is not None else None

        # Preferred ground truth: what the allocator went above its baseline
        # for the very step the graph describes. Same shape, same work, no
        # inference. Written into the graph's provenance by the tracing run.
        traced_peak = ((graph or {}).get("provenance") or {}).get(
            "activation_peak_bytes")
        if traced_peak:
            row("activations", derived_act, int(traced_peak),
                "vs the traced step's own peak (%d tokens)" % graph_tokens(graph))

        # Fallback, and a weaker one: the warmup prefill's peak. Only a check
        # at all when the warmup and the trace ran the same number of tokens.
        budget = (args.max_num_batched_tokens
                  or int(config.get("max_num_batched_tokens") or 0))
        if current is not None and peak is not None:  # noqa: SIM108
            warmup_act, note = max(peak - current, 0), "vs the warmup peak"  # noqa: E501
        else:
            # `_estimate_cudagraph_overhead` is 0.2 x peak activations under
            # manual capture, so a record predating the split still says what
            # its activation term was -- inverted, and flagged as inverted.
            overhead = readings.get("cudagraph_overhead") or 0
            warmup_act = int(overhead / 0.2) if overhead else 0
            note = "vs the warmup peak, inverted from the pool estimate"
        # The held-out test the traced-step comparison cannot be: the warmup
        # prefill is a different shape, measured independently, and the walk
        # has to reach it by scaling. Linear in tokens is the claim being
        # tested, and it is the whole reason this term is analytical rather
        # than a recording -- a term that only answers at the shape it was
        # taken on answers nothing worth asking.
        want, got = warmup_tokens(config, budget), graph_tokens(graph or {})
        scaled = None
        if derived_act is not None and want and got:
            scaled = int(derived_act * want / got)
            note += " (walk scaled %d -> %d tokens)" % (got, want)
        elif not want:
            note += "; warmup shape unknown (pass --max-num-batched-tokens)"
        row("activations", scaled, warmup_act, note)

        row("non-torch", None, readings.get("non_torch"), "not modelled")

        derived_pool = graph_pool_bytes(warmup_act) if warmup_act else None
        row("graph pool", derived_pool, readings.get("cudagraph_overhead"),
            "vs the engine's estimate")
        if pool_seen:
            row("graph pool", derived_pool, pool_seen, "vs the measured pool")

    if args.calibrate:
        write_calibration(non_torch_seen, args.calibrate)

    if args.curve and graph is not None:
        show_curve(graph)

    if len(non_torch_seen) > 1:
        print("\nthe terms with no model yet, across configurations")
        print("  %-22s %-12s %3s %9s %9s %9s"
              % ("record", "model", "tp", "params", "residue", "non_torch"))
        for name, config, readings, tp in non_torch_seen:
            parameters = readings.get("parameter_bytes")
            allocated = readings.get("weights_torch")
            residue = (allocated - parameters
                       if allocated is not None and parameters is not None
                       else None)
            print("  %-22s %-12s %3d %8s %8s %8.3fG"
                  % (name, str(config.get("model")).split("/")[-1], tp,
                     "-" if parameters is None else "%.3fG" % (parameters / GB),
                     "-" if residue is None else "%.3fG" % (residue / GB),
                     (readings.get("non_torch") or 0) / GB))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
