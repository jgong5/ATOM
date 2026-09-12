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
* **reservation** -- the engine's *policy*, predicted rather than restated:
  `0.2 x` the modelled peak activations, against the overhead the run actually
  reserved. This is the number that leaves the KV budget, so it is the pool
  quantity the block count is exposed to, and it is gated as its own term.
* **graph pool** -- what capture actually costs, which is a different question
  from what the engine sets aside for it. The derived side is the source-only
  prediction published in the profile's own calibration -- the TP=1 capture
  request stream transformed to this width and replayed through the allocator
  (`memory_capture.capture_stream` -> `capture_reserved_parts`) -- against the
  reserved delta in the record. **Not** `measured_graph_pool_bytes`: that is a
  superseded width-constant (104 MiB flat above TP=1, its own docstring says
  so), no predictor in the tree calls it, and gating against it charged the
  model 26.8% at TP=4 for a formula the run never used. Where the profile
  publishes no capture prediction the row does not compare, and under `--gate`
  an uncompared term fails.
* **kv blocks** -- the block count, derived from the checkpoint's own
  `config.json` and sized by ATOM's own `plan_pools`. The only row here with no
  fitted constant anywhere in it, which is what makes a disagreement
  attributable: it can only be in the readings.

    python scripts/compass/validate_memory.py compass_ops/mem_*.json \
        [--graph compass_ops/g.prefill.json] [--checkpoint DIR] \
        [--model-config DIR/config.json] [--log run.log]

Acceptance runs it once more, with the gate:

    python scripts/compass/validate_memory.py memory_out.json \
        --budget-source budget_source.json --model-config DIR/config.json \
        --max-num-batched-tokens N --gate

`--budget-source` is the record the run published when it chose its budget;
the profile is taken out of its input manifest, so the derived column is the
prediction that actually sized the run rather than one re-derived afterwards.
`--gate` then requires every non-KV term within 10% and the block count within
5%, and fails on a term nothing compared as readily as on one that disagreed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from atom.compass.core.kv_geometry import (  # noqa: E402
    InsufficientPoolBudget, blocks_from_readings, gdn_state_bytes,
    layer_types_disagree, paged_block_bytes, text_config)
from atom.compass.core.loaded_input import load_json  # noqa: E402
from atom.compass.core.memory import MemoryReadings  # noqa: E402
from atom.compass.core.memory_blocks import PROFILE_ROLE  # noqa: E402
from atom.compass.core.memory_calibration import for_model  # noqa: E402
from atom.compass.core.memory_model import (  # noqa: E402
    DEFAULT_PERSISTENT, UnfoundedPrediction, activation_curve,
    capture_pinned_bytes, derived_readings, graph_pool_bytes,
    load_residue_bytes, non_torch_bytes,
    peak_activation_bytes, liveness_is_recorded, liveness_instrumentation,
    traced_shape, weight_bytes, LIVENESS_INSTRUMENTATION)

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


def warmup_shape(config: dict, max_num_batched_tokens: int) -> tuple:
    """The prefill `warmup_model` actually runs, per request and in full.

    `ModelRunner.warmup_model` builds `num_seqs` sequences of `seq_len` tokens
    each, with no history -- they are fresh `Sequence` objects, so every token
    is scheduled and nothing is cached. The peak reading belongs to *that*
    shape: this many requests, this many query tokens each, zero context
    behind them.

    Returned rather than summed because the sum is not the shape. See
    `warmup_mismatch`.
    """
    max_model_len = int(config.get("max_model_len") or 0)
    max_num_seqs = int(config.get("max_num_seqs") or 1)
    if not (max_model_len and max_num_batched_tokens):
        return ((), ())
    num_seqs = max(1, min(max_num_batched_tokens // max_model_len, max_num_seqs))
    seq_len = max(1, min(max_model_len, max_num_batched_tokens // num_seqs))
    return ((seq_len,) * num_seqs, (seq_len,) * num_seqs)


def warmup_mismatch(graph: dict, config: dict,
                    max_num_batched_tokens: int) -> str:
    """Why this graph cannot be scaled to the warmup peak, or "".

    The check `graph_tokens` cannot make. The 27B's two 16 384-token prefill
    graphs have *identical* keys -- `batch_signature [16384]` both -- and
    differ only in history: the head chunk starts cold, the deep chunk carries
    98 304 cached tokens and reads 7x the KV. A token-total match takes either
    one, and comparing the deep chunk's walk against the warmup peak would be
    comparing two different steps and calling the difference model error.

    A *different number of tokens* is not a mismatch -- scaling across token
    counts is the claim the row exists to test, and the 0.6B's 3494-token trace
    reaching its 4096-token warmup peak is the one held-out result the
    activation term has. History is the disqualifier: warmup runs fresh
    sequences, so any graph with cached tokens behind its queries is a
    different step, not the same step at another size.

    Context is counted inclusive of the query, the way a batch spec writes it,
    so a cold request has `context_lens == query_lens`.
    """
    want_q, _ = warmup_shape(config, max_num_batched_tokens)
    if not want_q:
        return "warmup shape unknown (pass --max-num-batched-tokens)"
    got_q, got_c = traced_shape(graph)
    if not got_q:
        return "traced shape unknown"
    if not got_c:
        return "traced history unrecorded, so tokens are all there is to match"
    history = [c - q for c, q in zip(got_c, got_q)]
    if any(h for h in history):
        return "traced %s tokens of history, warmup runs none" % history
    return ""


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


def published_capture(cal_map, world: int):
    """The source-derived capture prediction the profile carries, at `world`.

    The prediction is a replay of the TP=1 capture request stream, transformed
    to this width and run through the allocator's own segment rules
    (`memory_capture.capture_stream` -> `memory_model.capture_reserved_parts`).
    It needs the recorded TP=1 allocation history, which is a probe artifact
    and not something a validator can hold, so it is computed once where that
    history lives and published per width in the calibration the profile names.
    That makes it input-bound: the run digested the calibration when it loaded
    the profile, so what is compared here is a number the run itself was
    carrying, not one this script chose.

    Returns ``(bytes, provenance)`` or ``(None, reason)``. Answering from a
    neighbouring width is exactly the carry-forward failure the topology module
    refuses, so an unpublished width returns nothing rather than the closest
    one.
    """
    table = (cal_map or {}).get("capture_reserved")
    if not isinstance(table, Mapping):
        return None, ("the profile's calibration publishes no source-derived "
                      "capture prediction")
    keyed = {}
    for key, value in table.items():
        try:
            keyed[int(key)] = value
        except (TypeError, ValueError):
            continue
    entry = keyed.get(int(world))
    if not isinstance(entry, Mapping) or not entry.get("total"):
        return None, ("the profile's calibration publishes no capture "
                      "prediction at TP=%d" % world)
    return int(entry["total"]), str(entry.get("provenance") or "")


#: How many bytes an element of the KV cache takes, by the name the record
#: keeps. Enough to price a block; a quantized cache also carries a scale,
#: which `paged_block_bytes` adds in fp32 regardless.
KV_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "float16": 2, "bfloat16": 2,
                  "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "int8": 1}


def kv_rows(config: dict, readings: dict, tp: int, world: int, blob: dict,
            model_config: str, predicted: dict = None, served=None) -> None:
    """The block count, derived from the model's own geometry.

    The last term, and the one every other term exists to serve: a
    configuration's capacity is its block count. It is also the only term whose
    derivation carries no fitted constant at all -- `config.json` gives the
    layer split and the head geometry, `plan_pools` is ATOM's own, and the five
    readings come from the record. So when this disagrees with a run, the
    disagreement is in the readings and nowhere else.

    `predicted` is the five readings the *model* states. Given one, the plan is
    made from those instead, and the row stops being an arithmetic check and
    becomes the question acceptance is actually asking: how far off is the
    capacity this configuration would have been given. Run off the recorded
    readings the row reads +0.00% on every record by construction -- the
    engine planned from exactly those numbers -- so a 5% gate over it would
    pass without testing anything.

    `served` is the count the run published in its budget source, and that is a
    *modelled* number, not a measured one -- it is what the prediction sized the
    deployment at. So it answers identity, never accuracy: re-deriving the
    plan here has to reproduce it, and if it does not, this is not the forecast
    that sized the run. It is never compared to the recorded count and never
    stands in for it. Both mistakes were live: equality against the record
    rejected a perfectly good 2% forecast as an inconsistency, and falling back
    to it when the record had no count compared the prediction with itself and
    read +0.00%.

    The recorded count is the independent half, and without it there is nothing
    to be accurate against -- the row goes uncovered rather than borrowing the
    forecast.

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
    plan_from = predicted or readings
    try:
        plan = blocks_from_readings(
            native, MemoryReadings(
                total=int(plan_from["total"]), free=int(plan_from["free"]),
                peak_torch=int(plan_from["peak_torch"]),
                non_torch=int(plan_from["non_torch"]),
                cudagraph_overhead=int(plan_from["cudagraph_overhead"])),
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
    # Identity, not accuracy: both sides here are the model's. The run stored
    # the capacity its forecast arrived at, and re-deriving that forecast from
    # the same profile has to land on the same block count. A difference means
    # the plan being gated is not the plan the run was sized by -- a different
    # revision, a different cell -- and the comparison below would be about
    # some other forecast than the one that shipped.
    if served is not None and predicted and int(served) != plan.paged_entries:
        PROBLEMS.append(
            "the run published a modelled %d KV blocks and re-deriving its own "
            "plan from the same profile gives %d" % (int(served),
                                                     plan.paged_entries))
    if recorded_blocks:
        error = (plan.paged_entries - recorded_blocks) / recorded_blocks * 100
        # Only a plan made from the model's own readings is evidence about the
        # model. The arithmetic-only row is left uncovered rather than counted,
        # so a gate run without a profile fails loudly instead of passing on an
        # identity.
        if predicted:
            note_term("kv blocks", error)
        print("  %-14s %8d %8d  %+5.2f%% %7s  %s"
              % ("kv blocks", plan.paged_entries, recorded_blocks, error, "",
                 "%s readings; %d B a block, %.1f MiB a request of state"
                 % ("the profile's" if predicted else
                    "the record's own (arithmetic only)",
                    block_bytes, state_bytes / (1 << 20))))
    else:
        # No independent count, so nothing here is evidence about the model.
        # The published forecast is not borrowed to fill the column: that would
        # compare the prediction with itself.
        note_term("kv blocks", None)
        print("  %-14s %8d %8s  %6s %7s  %s"
              % ("kv blocks", plan.paged_entries, "-", "-", "",
                 "derived; this record states no measured block count, so the "
                 "term is uncovered"))


def record_sha(path: str) -> str:
    """The record's own hash: integrity, and only integrity.

    It says whether these are the bytes a constant was read off. It does not
    say which run wrote them -- the source calibration's three runs wrote
    identical bytes -- so it is not what tells a repeat from a residual.
    """
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _cal_note(calib, cal_map: dict, config: dict, sha: str, producer,
              term: str, default: str) -> str:
    """The row note, saying what the comparison is worth.

    Bound per record with `partial`, because the answer depends on which run is
    being read: one constant is a residual against the record it was fitted on
    and a validation against any other.

    `residual` is also what an unidentified run gets. Records do not yet carry
    a producer, so that is every record today -- the honest reading of the row
    is "this record cannot show the constant reproduces", not "this record is
    the fit".

    Integrity is only reported where the producer says this *is* the fitted
    run. The first cut printed it whenever the record's hash differed from the
    fitted one, which fired on every row of every other record -- the exclusive
    capture is not the historical record and never claimed to be. `altered` has
    to mean the bytes moved under a run, not "you are reading another file".
    """
    if calib is None or term not in (cal_map or {}):
        return default
    kind = calib.classify(term, config, producer=producer)
    note = "%s; source-calibrated, this record is its %s" % (default, kind)
    if producer is None and kind == "residual":
        note += " (run unidentified)"
    elif calib.identifies(term, producer) and calib.integrity(term, sha) == "altered":
        note += ", re-serialised since it was fitted"
    return note


#: Where a saved budget source sits inside the provenance artifacts that carry
#: it. Nothing writes a standalone budget file: the runner publishes the object
#: as a runtime attribute and provenance saves it nested, so the readback reads
#: the same object out of the artifact rather than anyone adding a second
#: writer and a second schema for the same bytes.
BUDGET_CONTAINERS = (
    ("compass", "loaded_inputs", "ranks"),
    ("run", "server", "compass", "loaded_inputs", "ranks"),
)


def _dig(blob, keys):
    for key in keys:
        if not isinstance(blob, Mapping):
            return None
        blob = blob.get(key)
    return blob


def budget_source_object(blob, path: str, world=None, coords=None):
    """The budget object itself, wherever it was saved.

    Takes either the object (a `compass.memory.budget_source/1` blob) or the
    provenance artifact that contains one, and returns the object.

    The index into `ranks` is a *physical predictor record*, not a TP rank: a
    GPU-free replay writes one physical record and that record carries the
    whole width's budget, so there are widths where `ranks` has one entry and
    the deployment has four. Selection is therefore by what the record says
    about itself -- its coordinates, its width -- and an artifact holding
    several that cannot be told apart is refused. Picking one, or spreading one
    budget across the ranks that did not write it, would be inventing per-rank
    evidence out of a single reading.
    """
    if not isinstance(blob, Mapping):
        raise SystemExit("%s is not a JSON object" % path)
    if blob.get("inputs") or str(blob.get("schema") or "").startswith(
            "compass.memory.budget_source"):
        return blob
    for keys in BUDGET_CONTAINERS:
        ranks = _dig(blob, keys)
        if not isinstance(ranks, (list, tuple)) or not ranks:
            continue
        found = [(i, entry.get("budget_source")) for i, entry in enumerate(ranks)
                 if isinstance(entry, Mapping) and entry.get("budget_source")]
        if not found:
            raise SystemExit(
                "%s carries %s but no entry in it saved a budget_source, so "
                "nothing there says what this run was sized from."
                % (path, ".".join(keys)))
        if len(found) > 1:
            wanted = [(i, obj) for i, obj in found
                      if _budget_matches(obj, world, coords)]
            if len(wanted) != 1:
                raise SystemExit(
                    "%s saved %d budget sources and %s. The index into `ranks` "
                    "is a physical predictor record, not a TP rank, so this "
                    "script will not guess which one describes this record -- "
                    "name the record's own coordinates, or point "
                    "--budget-source at the entry."
                    % (path, len(found),
                       "none of them matches this record"
                       if not wanted else "several of them match this record"))
            found = wanted
        return found[0][1]
    raise SystemExit(
        "%s is neither a budget source nor a saved provenance artifact "
        "carrying one (looked under %s)"
        % (path, " and ".join(".".join(k) for k in BUDGET_CONTAINERS)))


def _budget_matches(obj, world, coords) -> bool:
    """Whether a saved budget describes the record being read back.

    Deliberately narrow. A budget states the width it sized and the deployment
    coordinates it was written at; anything it does not state cannot be
    matched, and a near-miss is not a match.
    """
    if not isinstance(obj, Mapping):
        return False
    lineage = obj.get("lineage") or {}
    stated = lineage.get("world_size", (obj.get("deployment") or {}).get(
        "world_size"))
    if world is not None and stated is not None and int(stated) != int(world):
        return False
    if coords:
        got = (obj.get("deployment") or {}).get("coords") or obj.get("coords")
        if got is not None and dict(got) != dict(coords):
            return False
    return True


def profile_from_budget_source(path: str, world=None, coords=None) -> dict:
    """Everything the run attested about the budget it chose.

    Both runners publish `compass.memory.budget_source/1` at the branch that
    *chose* the budget, and its `inputs` manifest carries the digest of the
    bytes the runner parsed, taken at the read. Taking the profile from there
    rather than from this script's own command line is the whole difference
    between "the file the run used" and "a file with a similar name that
    exists now" -- and the second is not evidence about the run.

    The manifest is kept **whole**. An earlier cut pulled out the profile row
    and dropped the rest, which meant the profile's digest was checked and the
    calibration and model config it names -- where every collective constant
    and the whole KV geometry live -- were re-opened unverified. A nested file
    can be edited without touching the profile's bytes, so an outer digest that
    still matches proves nothing about what `derived_readings` will read.

    Returns the profile's path and digest, a path -> digest map covering every
    attested input, and the run's own stored prediction (`lineage`) and served
    block count, so the comparison can be held to what the run published rather
    than only to what it recorded.
    """
    with open(path, encoding="utf-8") as fh:
        saved = json.load(fh)
    blob = budget_source_object(saved, path, world=world, coords=coords)
    rows = list((blob.get("inputs") or {}).get("inputs") or ())
    profiles = [r for r in rows if r.get("role") == PROFILE_ROLE]
    if not profiles:
        raise SystemExit(
            "%s is a %r budget and names no %s input, so this run was not "
            "sized from a profile and there is no prediction to compare its "
            "readings against." % (path, blob.get("kind"), PROFILE_ROLE))
    if len(profiles) > 1:
        raise SystemExit("%s names %d memory profiles; a rank reads one"
                         % (path, len(profiles)))
    attested: dict = {}
    for row in rows:
        where, digest = str(row.get("path") or ""), str(row.get("sha256") or "")
        if not where or not digest:
            continue
        key = os.path.abspath(where)
        if attested.get(key, digest) != digest:
            # The same path read twice with different bytes is not something to
            # pick a winner from: one of the two reads is being compared here
            # and there is no way to tell which.
            raise SystemExit(
                "%s attests %s at two different digests, %s and %s"
                % (path, where, attested[key][:12], digest[:12]))
        attested[key] = digest
    return {
        "budget_source": path,
        "profile": str(profiles[0].get("path") or ""),
        "profile_sha256": str(profiles[0].get("sha256") or ""),
        "attested": attested,
        "lineage": blob.get("lineage") or {},
        "num_kvcache_blocks": blob.get("num_kvcache_blocks"),
        "kind": blob.get("kind"),
    }


def attesting_loader(attested, source: str):
    """A `load` for `derived_readings` that reads each input once, verified.

    `derived_readings` opens whatever the profile names -- its calibration and
    its model config -- and those are inputs to the prediction exactly as the
    profile is. This reads each one time, digests the same bytes it parses, and
    refuses any file the run's manifest does not attest or whose bytes have
    moved since it did. Caching is not an optimisation here: the calibration
    was being opened twice, so the two reads could disagree and the second
    would silently win.
    """
    cache: dict = {}

    def load(where):
        key = os.path.abspath(str(where))
        if key in cache:
            return cache[key]
        if attested is None:
            # No budget source was saved, so there is nothing to attest
            # against. The caller has already said so; this reads plainly
            # rather than inventing an expectation.
            with open(str(where), encoding="utf-8") as fh:
                cache[key] = json.load(fh)
            return cache[key]
        expect = attested.get(key)
        if expect is None:
            raise SystemExit(
                "the prediction reads %s, which %s does not attest. An input "
                "the run never published cannot be compared against the run."
                % (where, source))
        blob, loaded = load_json(str(where), role="validate.nested_input")
        if loaded.sha256 != expect:
            raise SystemExit(
                "%s has moved under the comparison: the run read %s, these "
                "bytes are %s. The profile's own digest still matching says "
                "nothing about the files it names."
                % (where, expect[:12], loaded.sha256[:12]))
        cache[key] = blob
        return blob

    return load


def predicted_terms(attest: dict, config: dict, tokens: int,
                    world: int) -> dict:
    """Every non-KV term the predictor states, at this record's width.

    The derived column for `weights`, `model buffers` and `activations` is the
    profile's own -- the same numbers `derived_readings` hands the runner --
    so what the gate compares is the prediction that actually sized the run,
    not a re-derivation that might differ from it.

    Every file the derivation touches is verified against the run's manifest,
    not just the profile, and each is read once. Where the run also published
    its own readings (`lineage`), re-deriving them has to reproduce them: a
    disagreement means this is not the prediction that sized the run, whatever
    the digests say, and that is refused rather than reported as model error.

    `world_size` is passed so the profile's own cross-width guard fires: a
    TP=2 profile read against a TP=4 record is refused rather than quietly
    sized at two and reported as four.
    """
    path, expect_sha = attest["profile"], attest["profile_sha256"]
    profile, loaded = load_json(path, role=PROFILE_ROLE)
    if expect_sha and loaded.sha256 != expect_sha:
        raise SystemExit(
            "%s is not the profile the run read: the run's manifest says "
            "%s, these bytes are %s. The file moved under the comparison."
            % (path, expect_sha[:12], loaded.sha256[:12]))
    # A profile for another model priced against this record's readings is not
    # a failing comparison, it is a comparison of two unrelated runs -- the
    # 27B's parameters against the 0.6B's allocator read at +4560%. The gate
    # would report it as a model error, which is the one thing it must never
    # do, so it is refused here instead.
    stated = str((profile.get("provenance") or {}).get("model") or "")
    recorded = str(config.get("model") or "")
    if stated and recorded and stated != recorded:
        raise SystemExit(
            "%s is the profile for %s and this record is %s. Every term would "
            "compare two different models." % (path, stated, recorded))

    load = attesting_loader(attest.get("attested"),
                            attest.get("budget_source") or "the manifest")
    readings, activation = derived_readings(
        profile, warmup_tokens=tokens, load=load, world_size=world,
        source=path,
        enforce_eager=bool(config.get("enforce_eager")))

    # The run stored what it derived. Re-deriving it here has to land on the
    # same numbers, or this is some other prediction wearing the right digests
    # -- a different code revision, a different warmup shape -- and comparing
    # it to the record would charge the model for a gap it never produced.
    lineage = attest.get("lineage") or {}
    stated_world = lineage.get("world_size")
    if stated_world is not None and int(stated_world) != int(world):
        raise SystemExit(
            "the run published its budget at TP=%s and this record is TP=%d"
            % (stated_world, world))
    if lineage.get("profile") and os.path.abspath(str(lineage["profile"])) \
            != os.path.abspath(path):
        raise SystemExit(
            "the run's lineage says it modelled from %s and its manifest says "
            "it read %s" % (lineage["profile"], path))
    for term, got in (("peak_torch", readings.get("peak_torch")),
                      ("non_torch", readings.get("non_torch")),
                      ("cudagraph_overhead",
                       readings.get("cudagraph_overhead")),
                      ("activation_bytes", activation)):
        want = lineage.get(term)
        if want is None or got is None:
            continue
        if int(want) != int(got):
            raise SystemExit(
                "re-deriving the run's own prediction gives %s = %d and the "
                "run published %d. Same inputs, different answer: this is not "
                "the prediction that sized the run." % (term, int(got),
                                                        int(want)))
    return {
        "path": loaded.path,
        "sha256": loaded.sha256,
        "parameters": int(profile.get("parameters") or 0),
        "buffers": int(profile.get("buffers") or 0),
        "activation": int(activation),
        "readings": readings,
        # Already read and verified by the loader above; taking it from the
        # cache is what keeps one file from being parsed twice into two
        # possibly different mappings.
        "calibration": load(str(profile["calibration"])),
    }


#: What the comparison has to come out at. KV is the tighter one because it
#: *is* the capacity: every other term exists to size it, so their errors
#: reach a scheduler only through this one, and only after competing with each
#: other for the same budget.
KV_TOLERANCE = 5.0
TERM_TOLERANCE = 10.0

#: Every term the comparison has to have actually compared. A term that
#: printed no error is not a term that passed -- it is one nobody looked at,
#: and a sum of terms validated only in total is the shape of error this
#: project has already been caught by twice. So absence fails exactly as a
#: breach does.
#:
#: `pool estimate` and `capture pinned` are deliberately not here. The first is
#: an identity (both sides are the engine's own estimator) and the second is a
#: mechanism check on the pinned half of a term `graph pool` already gates.
#:
#: The pool is two terms, not one, because the engine asks two questions about
#: it. `reservation` is policy -- `0.2 x` the modelled peak activations, which
#: is what actually leaves the KV budget -- and `graph pool` is cost, what
#: capture goes on to reserve. They differ by 4-19x and neither substitutes
#: for the other: gating only the first would leave the capture prediction
#: unchecked, and gating only the second would leave the number the scheduler
#: is exposed to unchecked.
REQUIRED_TERMS = ("weights", "model buffers", "load residue", "persistent",
                  "activations", "non-torch", "reservation", "graph pool",
                  "kv blocks")

#: term -> error percent, or None where the row could not compare. Filled by
#: `row` and `kv_rows` as they print, reset per record, read by `gate`.
SEEN: dict = {}

#: Reasons this comparison is not complete, as opposed to not within tolerance.
#: A record that was asked for and could not be compared is the failure mode
#: with no row to show for it: the terms that did print all pass and the run
#: whose numbers are missing is the one nobody looked at. Rank 1 going quiet
#: while rank 0 reads clean is exactly that shape, so it fails here.
PROBLEMS: list = []


def note_term(name: str, error) -> None:
    """Remember a term's error for the gate, keeping the worst seen.

    Worst rather than last: `activations` prints twice on a record that
    carries a traced graph, and a gate that took the second would let the
    first disagreement through.
    """
    if name not in SEEN or SEEN[name] is None:
        SEEN[name] = error
    elif error is not None and abs(error) > abs(SEEN[name]):
        SEEN[name] = error


def gate(label: str) -> bool:
    """Whether this record's comparison passes, term by term."""
    print("\n  gate  --  every non-KV term within %.0f%%, KV within %.0f%%"
          % (TERM_TOLERANCE, KV_TOLERANCE))
    ok = not PROBLEMS
    for problem in PROBLEMS:
        print("  %-14s %9s             %s" % ("completeness", "-", problem))
    for name in REQUIRED_TERMS:
        limit = KV_TOLERANCE if name == "kv blocks" else TERM_TOLERANCE
        error = SEEN.get(name)
        if error is None:
            shown, verdict, ok = "       -", "UNCOVERED", False
        else:
            within = abs(error) <= limit
            shown, verdict = "%+7.2f%%" % error, "pass" if within else "FAIL"
            ok = ok and within
        print("  %-14s %9s  <= %4.1f%%   %s" % (name, shown, limit, verdict))
    print("  %s  %s" % ("GATE PASS" if ok else "GATE FAIL", label))
    return ok


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
        note_term(name, None)
    else:
        note_term(name, (derived - recorded) / recorded * 100)
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
    ap.add_argument("--budget-source",
                    help="the run's saved compass.memory.budget_source/1, or "
                         "the provenance artifact that contains it "
                         "(provenance.modelled.rN.json, modelled.rN.json). The "
                         "profile is taken from its input manifest, so the "
                         "derived column is the prediction that actually "
                         "sized the run")
    ap.add_argument("--budget-width", type=int,
                    help="which saved budget to read, by the width it sized, "
                         "when the artifact holds several. The index into "
                         "`ranks` is a physical predictor record and not a TP "
                         "rank -- a GPU-free replay writes one record for the "
                         "whole width -- so the selection is by what the "
                         "budget says about itself")
    ap.add_argument("--budget-coords",
                    help="which saved budget to read, by the deployment "
                         "coordinates it was written at, as JSON")
    ap.add_argument("--profile",
                    help="the memory profile, where no budget source was "
                         "saved. Names the file but cannot attest the run "
                         "read it; prefer --budget-source")
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero unless every non-KV term is within "
                         "%.0f%% and the block count within %.0f%%, on every "
                         "record. An uncompared term fails."
                         % (TERM_TOLERANCE, KV_TOLERANCE))
    args = ap.parse_args()

    attest = None
    if args.budget_source:
        coords = json.loads(args.budget_coords) if args.budget_coords else None
        attest = profile_from_budget_source(args.budget_source,
                                            world=args.budget_width,
                                            coords=coords)
        if args.profile and (os.path.abspath(args.profile)
                             != os.path.abspath(attest["profile"])):
            raise SystemExit(
                "--profile is %s but the run's budget source says it read %s. "
                "The run decides which it was."
                % (args.profile, attest["profile"]))
    elif args.profile:
        # Named but not attested: nothing says the run read these bytes, and
        # the rows will say so.
        attest = {"budget_source": "", "profile": args.profile,
                  "profile_sha256": "", "attested": None, "lineage": {},
                  "num_kvcache_blocks": None, "kind": None}
    profile_path = attest["profile"] if attest else None

    # A pattern that matched nothing is a record that was asked for and is not
    # here. Falling back to the literal pattern turned that into a file-not-
    # found much later, or -- when other patterns did match -- into a clean
    # report over whatever happened to exist.
    paths = []
    for pattern in args.records:
        found = sorted(glob.glob(pattern))
        if not found:
            if os.path.exists(pattern):
                found = [pattern]
            else:
                PROBLEMS.append("%s was asked for and matched no file" % pattern)
                continue
        paths.extend(found)
    graph = json.load(open(args.graph)) if args.graph else None
    pool_seen = measured_pool(args.log) if args.log else 0

    non_torch_seen = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        readings, config = blob.get("readings") or {}, blob.get("config") or {}
        if not readings:
            # A record that exists and states nothing is not a record that
            # agrees. Skipping it let a clean rank 0 carry a silent rank 1.
            PROBLEMS.append("%s carries no readings, so nothing in it was "
                            "compared" % os.path.basename(path))
            continue
        tp = int((config.get("topology") or {}).get("tp", 1) or 1)
        non_torch_seen.append(
            (os.path.basename(path), config, readings, tp, blob, record_sha(path)))

    print("  %-14s %8s %8s  %6s %7s  %s"
          % ("term", "derived", "recorded", "error", "of bgt", "note"))
    passed = True
    for name, config, readings, tp, blob, sha in non_torch_seen:
        SEEN.clear()
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

        # The prediction that sized this run, where the run said which one it
        # was. Its calibration then supplies the width-dependent terms too, so
        # every derived figure below comes from one profile rather than from a
        # profile and a separately chosen table.
        predicted = None
        if profile_path:
            predicted = predicted_terms(
                attest, config,
                warmup_tokens(config, args.max_num_batched_tokens
                              or int(config.get("max_num_batched_tokens") or 0)),
                world)
            calib, cal_map = None, predicted["calibration"]
            print("  predicted by    %s  %s"
                  % (predicted["sha256"][:12], predicted["path"]))

        # The producer block if the record carries one. None today: the writer
        # does not emit it, which is why every row reads "run unidentified".
        cal_note = partial(_cal_note, calib, cal_map, config, sha,
                           blob.get("run"))

        checkpoint = args.checkpoint
        derived_weights = weight_bytes(checkpoint, tp) if checkpoint else None
        weights_note = "" if parameters else "record predates the split"
        # The profile states both, so where there is one it is the derived
        # side of both rows: the profile's `parameters` are what the run was
        # sized with, and a checkpoint re-read would be a second derivation
        # answering a question the run did not ask.
        derived_buffers = None
        if predicted is not None:
            derived_weights = predicted["parameters"]
            derived_buffers = predicted["buffers"]
            weights_note = "the profile's own parameters, sharded at TP=%d" % tp
        # Against the model's own parameters, which is what the term claims to
        # be -- not against the allocator after loading, which is that plus
        # whatever the loader still holds, and not against the buffers either,
        # which the checkpoint does not contain.
        buffers = readings.get("buffer_bytes")
        weights_seen = (parameters - buffers
                        if parameters is not None and buffers is not None
                        else parameters)
        row("weights", derived_weights, weights_seen, weights_note, sizing_budget)
        row("model buffers", derived_buffers, buffers,
            "the profile's own buffers" if derived_buffers is not None else
            "not modelled; built at init, absent from the checkpoint",
            sizing_budget)

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

        # A walk is only liveness where liveness was recorded. Without
        # `dies_at` the walk falls back to last-read, which on the 27B's
        # meta-derived prefill graphs returns 19.3% of the measured term out of
        # allocations it never saw freed. That number is not a model of
        # anything and is not to be read as one.
        walked = graph is not None and liveness_is_recorded(graph)
        if graph is not None and not walked:
            derived_act = None

        # And which producer wrote the fields the walk read. A graph derived
        # before 2026-09-11 is version 1, where every meta storage keyed to the
        # same address: weights read as activations, every output as an alias,
        # no out-variant destination as unseen. Those graphs carried no
        # `dies_at` either, so `walked` is already False for them -- this says
        # so out loud, because "no liveness recorded" and "liveness recorded by
        # a producer that could not tell two tensors apart" are different
        # findings and only one of them is fixed by re-deriving.
        if graph is not None:
            version = liveness_instrumentation(graph)
            print("  liveness instrumentation : v%d%s" % (
                version,
                "" if version >= LIVENESS_INSTRUMENTATION else
                "  (pre-fix producer; re-derive before trusting a walk)"))

        # Preferred ground truth: what the allocator went above its baseline
        # for the very step the graph describes. Same shape, same work, no
        # inference. Written into the graph's provenance by the tracing run.
        traced_peak = ((graph or {}).get("provenance") or {}).get(
            "activation_peak_bytes")
        if traced_peak:
            # Named apart from the gated term where a profile is in play: the
            # walk against its own traced step and the profile's term at the
            # warmup shape are two different claims, and the gate is about the
            # one the run was sized by.
            row("activations (walk)" if predicted is not None else "activations",
                derived_act, int(traced_peak),
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
        #
        # Scaled only from the graph that *is* the warmup step. Same token
        # total is not the same step: the 27B's head and deep 16 384-token
        # chunks have identical keys and 98 304 tokens of history between them.
        want, got = warmup_tokens(config, budget), graph_tokens(graph or {})
        mismatch = warmup_mismatch(graph, config, budget) if graph else ""
        scaled = None
        if graph is not None and not walked:
            note += "; no liveness recorded in this graph, nothing to walk"
        elif mismatch:
            note += "; not the warmup step -- %s" % mismatch
        elif derived_act is not None and want and got:
            scaled = int(derived_act * want / got)
            note += " (walk scaled %d -> %d tokens)" % (got, want)
        elif not want:
            note += "; warmup shape unknown (pass --max-num-batched-tokens)"
        # The term is calibrated even where it cannot be walked, and saying so
        # is the difference between "no derived figure here" and "no evidence
        # for this term anywhere". It is not offered as a model input --
        # `mapping` withholds it -- because it is a peak at a shape, so the
        # note has to name the shape it holds at rather than the width.
        if calib is not None and "activations" in calib.terms:
            kind = calib.classify("activations", config, producer=blob.get("run"))
            if kind == "validation":
                note += ("; the source calibration is a peak at another "
                         "configuration and is not stretched to this one")
            else:
                note += ("; source-calibrated at this warmup shape (%d B), "
                         "this record is its %s"
                         % (calib.terms["activations"].value, kind))
                if blob.get("run") is None and kind == "residual":
                    note += " (run unidentified)"
        if predicted is not None:
            # The profile's own term, at this record's warmup shape -- the
            # figure that went into `peak_torch` and sized the run, rather than
            # a walk re-scaled here.
            scaled = predicted["activation"]
            note = ("the profile's term at the warmup shape (%d tokens), vs "
                    "the warmup peak" % warmup_tokens(config, budget))
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
        widths = {int(w) for w in (cal_map or {}).get("non_torch") or {}}
        nt_cal = cal_map if world in widths else None
        derived_nt = (predicted["readings"]["non_torch"] if predicted is not None
                      else non_torch_bytes(world, nt_cal))
        row("non-torch", derived_nt, readings.get("non_torch"),
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

        # The same estimator run forward instead of restated: 0.2 x the
        # *modelled* peak activations, against what the run reserved. This is
        # the pool number the KV budget is exposed to, so it is gated even
        # though it is policy rather than cost -- an engine that sets aside the
        # wrong amount misprices the cache whether or not capture then fits.
        if predicted is not None:
            row("reservation", predicted["readings"].get("cudagraph_overhead"),
                estimate, "the engine's reservation policy at the modelled "
                          "activation peak", sizing_budget)

        # The cost, which is a different question. The derived side is the
        # source-only capture replay published in the profile's calibration --
        # never `measured_graph_pool_bytes`, whose width constant no predictor
        # uses. `graph_pool.reserved` is in the record; a `--log` is only
        # needed for a record written before it was.
        reserved, allocated, capture_sizes = recorded_pool(blob)
        seen = reserved or pool_seen
        if seen:
            capture_pred, capture_note = published_capture(cal_map, world)
            row("graph pool", capture_pred, seen,
                ("vs the %d MiB capture actually reserved over %d buckets; %s"
                 % (seen / (1 << 20), len(capture_sizes),
                    capture_note or "source-derived capture replay"))
                if capture_pred is not None else capture_note,
                sizing_budget)
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
            # The pinned half, against a mechanism rather than a fitted line.
            # Silent without `--model-config` for the same reason the KV rows
            # are: the vocabulary is the checkpoint's to state.
            vocab = 0
            if args.model_config:
                try:
                    with open(args.model_config, encoding="utf-8") as fh:
                        native = json.load(fh) or {}
                    vocab = int(text_config(native).get("vocab_size") or 0)
                except (OSError, ValueError, TypeError):
                    vocab = 0
            if vocab or world > 1:
                try:
                    pinned = capture_pinned_bytes(
                        capture_sizes, vocab_size=vocab, world_size=world,
                        tbo=bool(config.get("enable_tbo")))
                except UnfoundedPrediction as exc:
                    print("  %-14s %s" % ("capture pinned", exc))
                else:
                    row("capture pinned", pinned, allocated,
                        "fixed residue %s the LM head, which the runner "
                        "captures only at width one"
                        % ("plus" if world == 1 else "without"),
                        sizing_budget)

        kv_rows(config, readings, tp, world, blob, args.model_config,
                predicted["readings"] if predicted is not None else None,
                served=(attest or {}).get("num_kvcache_blocks"))

        if args.gate:
            passed = gate(name) and passed

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

    if args.gate:
        if not non_torch_seen:
            print("\nGATE FAIL  no record carried readings, so nothing was "
                  "compared")
            for problem in PROBLEMS:
                print("  %s" % problem)
            return 1
        if PROBLEMS:
            # Already shown per record; restated here because a completeness
            # failure is the one that leaves no failing row behind it.
            print("\nincomplete: %d requested record(s) were not compared"
                  % len(PROBLEMS))
            passed = False
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
