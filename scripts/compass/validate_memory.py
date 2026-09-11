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
* **pool estimate** -- the mirror of `_estimate_cudagraph_overhead` against
  that estimator's own output. An identity, and labelled as one: it read
  `+0.0%` at every width because both sides are `0.2 x (peak_torch -
  current_torch)` computed from the same record. Worth keeping as a drift check
  on the mirror, worth nothing as evidence about the pool.
* **graph pool** -- the term itself: what capture is predicted to reserve,
  against what capture did reserve. That measurement has been in every record
  since the terms were split and nothing was reading it.
* **kv blocks** -- the block count, derived from the checkpoint's own
  `config.json` and sized by ATOM's own `plan_pools`. The only row here with no
  fitted constant anywhere in it, which is what makes a disagreement
  attributable: it can only be in the readings.

    python scripts/compass/validate_memory.py compass_ops/mem_*.json \
        [--graph compass_ops/g.prefill.json] [--checkpoint DIR] \
        [--model-config DIR/config.json] [--log run.log]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from atom.compass.core.kv_geometry import (  # noqa: E402
    InsufficientPoolBudget, blocks_from_readings, gdn_state_bytes,
    layer_types_disagree, paged_block_bytes)
from atom.compass.core.memory import MemoryReadings  # noqa: E402
from atom.compass.core.memory_calibration import for_model  # noqa: E402
from atom.compass.core.memory_model import (  # noqa: E402
    DEFAULT_PERSISTENT, activation_curve, graph_pool_bytes,
    load_residue_bytes, measured_graph_pool_bytes, non_torch_bytes,
    peak_activation_bytes, weight_bytes)

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


def recorded_pool(blob: dict) -> tuple:
    """What capture reserved, allocated and captured, out of the record itself.

    `_measured_graph_pool` has been writing this into every record since the
    terms were split, and nothing read it: the graph-pool row compared the
    derivation against `cudagraph_overhead`, which is the *engine's estimate*
    of the pool and not the pool. Both sides of that comparison are
    `0.2 x (peak_torch - current_torch)`, so it reported +0.0% at every width
    and could not have reported anything else.

    Returns ``(reserved, allocated, capture_sizes)``, all zero or empty for a
    record written before the measurement existed.
    """
    pool = blob.get("graph_pool") or {}
    return (int(pool.get("reserved") or 0), int(pool.get("allocated") or 0),
            tuple(int(s) for s in (pool.get("capture_sizes") or ())))


#: How many bytes an element of the KV cache takes, by the name the record
#: keeps. Enough to price a block; a quantized cache also carries a scale,
#: which `paged_block_bytes` adds in fp32 regardless.
KV_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "float16": 2, "bfloat16": 2,
                  "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "int8": 1}


def kv_rows(config: dict, readings: dict, tp: int, world: int, blob: dict,
            model_config: str) -> None:
    """The block count, derived from the model's own geometry.

    The last term, and the one every other term exists to serve: a
    configuration's capacity is its block count. It is also the only term whose
    derivation carries no fitted constant at all -- `config.json` gives the
    layer split and the head geometry, `plan_pools` is ATOM's own, and the five
    readings come from the record. So when this disagrees with a run, the
    disagreement is in the readings and nowhere else.

    Silent without `--model-config`, because guessing the checkpoint from the
    record's model name would be a download.
    """
    if not model_config:
        return
    try:
        with open(model_config, encoding="utf-8") as fh:
            native = json.load(fh)
    except (OSError, ValueError) as exc:
        print("  %-14s %s" % ("kv blocks", "cannot read %s: %s"
                              % (model_config, exc)))
        return

    disagreement = layer_types_disagree(native)
    if disagreement:
        print("  %-14s %s" % ("", "WARNING: " + disagreement))

    block_size = int(config.get("block_size") or 0)
    kv_bytes = KV_DTYPE_BYTES.get(str(config.get("kv_cache_dtype")), 2)
    recorded_blocks = ((blob.get("blocks") or {}).get("num_kvcache_blocks"))
    try:
        plan = blocks_from_readings(
            native, MemoryReadings(
                total=int(readings["total"]), free=int(readings["free"]),
                peak_torch=int(readings["peak_torch"]),
                non_torch=int(readings["non_torch"]),
                cudagraph_overhead=int(readings["cudagraph_overhead"])),
            utilization=float(config.get("gpu_memory_utilization") or 0),
            max_num_seqs=int(config.get("max_num_seqs") or 0),
            tensor_parallel=tp, block_size=block_size,
            kv_dtype_bytes=kv_bytes)
    except InsufficientPoolBudget as exc:
        print("  %-14s %s" % ("kv blocks", "INFEASIBLE: the state floor needs "
                              "%.2fGB of %.2fGB"
                              % (exc.reserved_bytes / GB,
                                 exc.available_bytes / GB)))
        return
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        print("  %-14s %s" % ("kv blocks", "not derivable: %s" % exc))
        return

    block_bytes = paged_block_bytes(native, tensor_parallel=tp,
                                    block_size=block_size,
                                    kv_dtype_bytes=kv_bytes)
    state_bytes = gdn_state_bytes(native, tensor_parallel=tp)
    if recorded_blocks:
        error = (plan.paged_entries - recorded_blocks) / recorded_blocks * 100
        print("  %-14s %8d %8d  %+5.2f%% %7s  %s"
              % ("kv blocks", plan.paged_entries, recorded_blocks, error, "",
                 "derived from config.json; %d B a block, %.1f MiB a request "
                 "of state" % (block_bytes, state_bytes / (1 << 20))))
    else:
        print("  %-14s %8d %8s  %6s %7s  %s"
              % ("kv blocks", plan.paged_entries, "-", "-", "",
                 "derived; no recorded count in this record"))


def record_sha(path: str) -> str:
    """The record's own hash, so a rerun is not mistaken for the source run."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _cal_note(calib, cal_map: dict, config: dict, sha: str, term: str,
              default: str) -> str:
    """The row note, saying what the comparison is worth.

    Bound per record with `partial`, because the answer depends on which run is
    being read: one constant is a residual against the record it was fitted on
    and a validation against any other.
    """
    if calib is None or term not in (cal_map or {}):
        return default
    kind = calib.classify(term, config, record_sha256=sha)
    return "%s; source-calibrated, this record is its %s" % (default, kind)


def row(name: str, derived, recorded, note: str = "", budget: int = 0) -> None:
    """One term, its own error, and what that error is worth.

    The relative column alone ranks a 14 MiB term missed by 93% alongside a
    54 GB term missed by 2%. Only one of those can move the block count, so
    the miss is also shown against the sizing budget it competes for.
    """
    def show(value):
        return "       -" if value is None else "%7.3fG" % (value / GB)
    if derived is None or recorded is None or not recorded:
        error, share = "     -", "      -"
    else:
        error = "%+5.1f%%" % ((derived - recorded) / recorded * 100)
        share = ("%+6.2f%%" % ((derived - recorded) / budget * 100)
                 if budget else "      -")
    print("  %-14s %8s %8s  %6s %7s  %s" % (name, show(derived), show(recorded),
                                            error, share, note))


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
    for _, config, readings, tp, _blob, _sha in records:
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
    ap.add_argument("--log", help="run log, for the measured graph pool "
                                  "(records now carry it; only needed for one "
                                  "written before they did)")
    ap.add_argument("--model-config",
                    help="the checkpoint's own config.json, for the KV block "
                         "geometry. Not the checkpoint directory: this reads "
                         "one file and downloads nothing")
    ap.add_argument("--max-num-batched-tokens", type=int, default=0,
                    help="not in the record; needed to know the warmup shape")
    ap.add_argument("--curve", action="store_true",
                    help="show where the walk and the allocator diverge")
    ap.add_argument("--source-calibration", action="store_true",
                    help="use the model's constants calibrated at its declared "
                         "source configuration, where it has any. Rows say "
                         "whether each comparison is a validation or the "
                         "calibration's own residual.")
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
        non_torch_seen.append(
            (os.path.basename(path), config, readings, tp, blob, record_sha(path)))

    print("  %-14s %8s %8s  %6s %7s  %s"
          % ("term", "derived", "recorded", "error", "of bgt", "note"))
    for name, config, readings, tp, blob, sha in non_torch_seen:
        print("\n%s  --  %s tp=%d max_model_len=%s"
              % (name, config.get("model"), tp, config.get("max_model_len")))

        allocated = readings.get("weights_torch")
        parameters = readings.get("parameter_bytes")
        current = readings.get("current_torch")
        peak = readings.get("peak_torch")

        # What the terms are competing for: the fraction of the card the
        # engine may spend. An error only matters as a share of this.
        world = 1
        for size in (config.get("topology") or {"tp": tp}).values():
            world *= max(1, int(size or 1))
        sizing_budget = int((readings.get("total") or 0)
                            * float(config.get("gpu_memory_utilization") or 0))

        calib = for_model(config.get("model")) if args.source_calibration else None
        cal_map = calib.mapping(world) if calib else None

        cal_note = partial(_cal_note, calib, cal_map, config, sha)

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
            "" if parameters else "record predates the split", sizing_budget)
        row("model buffers", None, buffers,
            "not modelled; built at init, absent from the checkpoint", sizing_budget)

        residue = (allocated - parameters
                   if allocated is not None and parameters is not None else None)
        row("load residue", load_residue_bytes(world, cal_map), residue,
            cal_note("load_residue", "collective pools held through the "
                     "allocator"), sizing_budget)

        # The engine's own forward buffers, and nothing else: the residue above
        # is already resident and counting it twice would make this row a sum
        # of two terms rather than a term.
        persistent = (current - allocated
                      if current is not None and allocated is not None else None)
        row("persistent", int((cal_map or {}).get("persistent") or DEFAULT_PERSISTENT),
            persistent,
            cal_note("persistent", "engine forward buffers; flat in width"),
            sizing_budget)

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

        # `non_torch` is device-wide used memory minus this process's reserve,
        # so a neighbour on the same card is charged here. A disagreement is
        # therefore ambiguous between a wrong model and a busy box, and the
        # ranks' spread is the tell: they hold the same thing, so where they
        # differ, something outside the run does not.
        #
        # The calibration is handed over only at a width it was measured at.
        # `non_torch_bytes` drops `MODEL_HEADROOM` whenever it is given one,
        # which is right where the calibrated run already contains the headroom
        # and wrong everywhere else: an uncalibrated width would lose the term.
        nt_cal = (cal_map if cal_map and world in (cal_map.get("non_torch") or {})
                  else None)
        row("non-torch", non_torch_bytes(world, nt_cal), readings.get("non_torch"),
            cal_note("non_torch",
                     "device-wide reading; a neighbour is charged here"),
            sizing_budget)

        # Two questions that were being asked as one. `graph_pool_bytes`
        # mirrors `_estimate_cudagraph_overhead`, and `cudagraph_overhead` *is*
        # that estimator's output, so the two agree by construction: both are
        # 0.2 x (peak_torch - current_torch) computed from the same record.
        # That row is an identity check -- worth keeping, because a drift in
        # the mirror would show here first -- and it is now labelled as one
        # rather than read as a validated term.
        derived_pool = graph_pool_bytes(warmup_act) if warmup_act else None
        estimate = readings.get("cudagraph_overhead")
        identity = (derived_pool is not None and derived_pool == estimate)
        row("pool estimate", derived_pool, estimate,
            "identity: both sides are the engine's own estimator"
            if identity else "the mirror has drifted from the engine's "
                             "estimator", sizing_budget)

        # The term itself: what capture reserved, against what capture was
        # predicted to reserve. `graph_pool.reserved` is in the record; a
        # `--log` is only needed for a record written before it was.
        reserved, allocated, capture_sizes = recorded_pool(blob)
        seen = reserved or pool_seen
        if seen:
            row("graph pool", measured_graph_pool_bytes(capture_sizes, world),
                seen,
                "vs the %d MiB capture actually reserved over %d buckets"
                % (seen / (1 << 20), len(capture_sizes)), sizing_budget)
            if estimate:
                # Which way the estimator is wrong is not fixed. It is 0.2x the
                # peak activations, so it scales with the model while the pool
                # does not: on the 0.6B it was 19x under, and here it is over.
                # A term whose error changes sign with the model is not a term
                # that is merely mis-calibrated.
                over = estimate > seen
                print("  %-14s %8s %8s  %6s %7s  %s"
                      % ("", "", "", "", "",
                         "the engine budgeted %.3fG; capture took %.3fG -- "
                         "%.1fx %s"
                         % (estimate / GB, seen / GB,
                            (estimate / seen) if over else (seen / estimate),
                            "over" if over else "under")))
        if allocated:
            print("  %-14s %8s %8s  %6s %7s  %s"
                  % ("", "", "", "", "",
                     "of which %.1f MiB allocated; the rest is segment "
                     "bookkeeping" % (allocated / (1 << 20))))

        kv_rows(config, readings, tp, world, blob, args.model_config)

    if args.calibrate:
        write_calibration(non_torch_seen, args.calibrate)

    if args.curve and graph is not None:
        show_curve(graph)

    if len(non_torch_seen) > 1:
        print("\nthe terms with no model yet, across configurations")
        print("  %-22s %-12s %3s %9s %9s %9s"
              % ("record", "model", "tp", "params", "residue", "non_torch"))
        for name, config, readings, tp, _blob, _sha in non_torch_seen:
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
