"""**analytical** -- the startup block reply a GPU-free replay never measured.

`memory_model` says what a configuration spends on everything that is not KV
and `kv_geometry` turns what is left into a pool plan. Both already run without
a device. What was missing is the last hop: the engine does not ask a runner
for readings or for a plan, it asks it for the reply to `get_num_blocks`, and
until now only the device-backed runner could produce one from a model.

So a GPU-free replay accepted ``--compass-memory-model`` and then answered
``get_num_blocks`` out of the captured target anyway. The flags composed, the
run came up, the profile changed nothing, and a scheduler sized by a *capture*
was read as a forecast of the modelled deployment -- the same failure
`derived_readings` refuses in the term it owns, one call further out.

This module is that hop and nothing else. It owns no arithmetic: the readings
are `derived_readings`', the geometry is `kv_geometry`'s, the sizing is ATOM's
own `plan_pools`, and the wire form is the one `model_runner.get_num_blocks`
returns. What it adds is the refusals that belong at the boundary -- a profile
that will not load, a width the deployment is not running at, a topology the
profile cannot describe, a budget with no room to page -- and one deliberate
borrow from the capture, `state_runtime`, which is layout and not capacity.

**Never a fallback.** Every path out of `derived_block_info` is either the
modelled reply or an exception. Returning the captured counts when the model
refuses would put the two kinds of number back in the same channel.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, MutableSequence, Optional

from atom.compass.core.kv_geometry import (
    GDN_HYBRID_MODEL_TYPES,
    blocks_from_readings,
    layer_types_disagree,
    text_config,
)
from atom.compass.core.loaded_input import load_json
from atom.compass.core.memory import MemoryReadings
from atom.compass.core.memory_model import UnfoundedPrediction, derived_readings

logger = logging.getLogger(__name__)

__all__ = ["warmup_tokens", "derived_block_info", "profile_reader",
           "capacity_context", "derivation_lineage", "PROFILE_ROLE",
           "BUDGET_SCHEMA", "DEVICE_MEASURED", "CAPTURED", "RECORDED",
           "SOURCE_DERIVED", "budget_source"]

#: The record `budget_source` produces. Versioned: a validator reads it to
#: decide whether a run is a hardware reference, and a field that quietly
#: changes meaning is worse than one that is absent.
BUDGET_SCHEMA = "compass.memory.budget_source/1"

#: Where the block count a run actually served came from. Decided at the
#: branch that produced it, never inferred from the flags afterwards: a
#: profile is honoured by the device-backed runner in *measure* mode too, so a
#: run whose mode and clock look like a measurement can still have been sized
#: analytically. That is a legitimate diagnostic; what is not legitimate is it
#: being unreadable from the outside.
DEVICE_MEASURED = "device-measured"   # the card was asked, on this run
CAPTURED = "captured"                 # replayed from a target a device wrote
RECORDED = "recorded"                 # replayed from a memory record (memory_in)
SOURCE_DERIVED = "source-derived"     # computed from a profile; no device

#: What a memory profile is to a run: an input that decides the deployment's
#: *actual capacity*, not one that constructs the cost oracle. The files the
#: profile itself names are nested under it -- `runtime.memory_model.
#: calibration`, `runtime.memory_model.model_config` -- so a validator can see
#: that a calibration was read without it being flattened into its parent.
PROFILE_ROLE = "runtime.memory_model"

#: What a `kv_cache_dtype` costs per element. Mirrors the table
#: `scripts/compass/validate_memory.py` reads records with; both exist because
#: the flag names a dtype and the geometry wants a width.
KV_DTYPE_BYTES = {
    "bf16": 2, "fp16": 2, "float16": 2, "bfloat16": 2,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "auto": 2, "None": 2, "": 2,
}




def budget_source(
    kind: str,
    *,
    inputs=(),
    served: bool = True,
    num_kvcache_blocks: Optional[int] = None,
    deployment: Optional[Mapping[str, Any]] = None,
    lineage: Optional[Mapping[str, Any]] = None,
    coords: Optional[Mapping[str, int]] = None,
) -> dict:
    """What the budget this run served actually came from.

    Published at the branch that chose it -- `get_num_blocks` -- and not
    reconstructed later from the options, because the options do not decide it:
    `--compass-memory-model` is honoured by the device-backed runner in every
    mode, so a measure-mode run with a real clock can serve an analytically
    derived capacity, and nothing in the request says so.

    `hardware_reference` is the field a validator wants and it is true for one
    kind only: a budget this run took off the card. A captured, recorded or
    derived budget may be entirely legitimate -- a GPU-free replay sized from a
    source-derived profile is the point of the exercise -- but it is not
    evidence about *this* run's hardware, and final acceptance's real-hardware
    side is the only place that distinction has to be enforced.

    `lineage` is how a derivation says what it was derived *from*. A
    source-derived budget is not disqualified for having read an artifact;
    what disqualifies it is being unable to say which one and at what width.
    """
    from atom.compass.core.loaded_input import manifest as _manifest

    record = {
        "schema": BUDGET_SCHEMA,
        "kind": kind,
        "hardware_reference": kind == DEVICE_MEASURED,
        "served": bool(served),
        "inputs": _manifest(inputs, coords=coords),
    }
    if num_kvcache_blocks is not None:
        record["num_kvcache_blocks"] = int(num_kvcache_blocks)
    if deployment is not None:
        record["deployment"] = dict(deployment)
    if lineage is not None:
        record["lineage"] = dict(lineage)
    return record


def warmup_tokens(config) -> int:
    """The token count of the prefill that would have set `peak_torch`.

    The same shape `runtime/runner.py::_warmup_tokens` derives, because it is
    the same question: `warmup_model` resets the high-water mark and runs one
    dummy prefill, so the peak belongs to that shape. Restated here rather than
    imported because importing it means importing the device-backed runner,
    which cannot be imported without a device -- and this module exists for the
    machine that has none. `tests/compass/test_replay_memory.py` holds the two
    against each other so the restatement cannot drift quietly.
    """
    budget = int(getattr(config, "max_num_batched_tokens", 0) or 0)
    length = int(getattr(config, "max_model_len", 0) or 0)
    if not (budget and length):
        return 0
    seqs = max(1, min(budget // length,
                      int(getattr(config, "max_num_seqs", 1) or 1)))
    return seqs * max(1, min(length, budget // seqs))


def _refuse(what: str) -> None:
    raise UnfoundedPrediction(
        "ATOMCompass: %s. This run asked for a modelled KV budget, so the "
        "captured block count is not an answer to fall back to -- it is a "
        "measurement of the configuration the model was supposed to replace."
        % what)


def _world_size(config) -> int:
    """The width the deployment is about to run at, or a refusal.

    Pipeline parallelism is refused rather than folded in: the engine reduces
    the per-stage block counts to their minimum with a collective, a profile
    describes one stage's memory, and there is no second stage here to disagree
    with. Silently sizing one stage and publishing it as the pool is the exact
    shape of error this module is for.
    """
    stages = int(getattr(config, "pipeline_parallel_size", 1) or 1)
    if stages > 1:
        _refuse("a modelled budget was asked for at pipeline_parallel_size=%d. "
                "The engine takes the minimum block count across stages with a "
                "collective and a profile describes one stage" % stages)
    return max(1, int(getattr(config, "tensor_parallel_size", 1) or 1))


def capacity_context(config, compass_config, memory_model: str = "") -> dict:
    """The deployment terms a block count cannot be checked without.

    The digests say which bytes were read; these say what they were read
    *for*. Both are needed to say a number was founded: the same profile sizes
    a different pool at another width, block size or utilization. One shape for
    both runners, because the question a reader asks of either is the same.
    """
    return {
        "modelled": bool(memory_model),
        "memory_model": memory_model or None,
        "deployment": {
            # The mode as well as the flags: the pair is the thing a reader has
            # been guessing from. A measure-mode run can serve a derived
            # budget, so neither field alone says what was served.
            "mode": str(getattr(compass_config, "mode", "")),
            "model": str(getattr(config, "model", "")),
            "tensor_parallel_size": int(
                getattr(config, "tensor_parallel_size", 1) or 1),
            "pipeline_parallel_size": int(
                getattr(config, "pipeline_parallel_size", 1) or 1),
            "max_model_len": int(getattr(config, "max_model_len", 0) or 0),
            "max_num_seqs": int(getattr(config, "max_num_seqs", 0) or 0),
            "max_num_batched_tokens": int(
                getattr(config, "max_num_batched_tokens", 0) or 0),
            "gpu_memory_utilization": float(
                getattr(config, "gpu_memory_utilization", 0.0) or 0.0),
            "kv_cache_block_size": int(
                getattr(config, "kv_cache_block_size", 0) or 0),
            "kv_cache_dtype": str(getattr(config, "kv_cache_dtype", "auto")),
            "enforce_eager": bool(getattr(config, "enforce_eager", False)),
        },
    }


def derivation_lineage(profile, calibration, *, path, world, activation,
                       readings=None) -> dict:
    """What a derived budget was derived *from*.

    One shape for both sizing paths, so a modelled budget describes itself the
    same way whether a card was present or not. A source-derived budget is not
    disqualified for having read an artifact; what disqualifies it is being
    unable to say which one, at what width, and on whose calibration.
    """
    profile = profile if isinstance(profile, Mapping) else {}
    calibration = calibration if isinstance(calibration, Mapping) else {}
    readings = readings if isinstance(readings, Mapping) else {}
    return {
        "kind": SOURCE_DERIVED,
        "profile": path,
        "world_size": int(world),
        "compile_mode": profile.get("compile_mode"),
        "total_source": profile.get("total_source") or profile.get("source"),
        # Per term, in the calibration's own words: which run each number came
        # off and what was composed onto it. This is what makes a derived
        # budget auditable rather than merely undevice.
        "calibration_provenance": dict(calibration.get("provenance") or {}),
        "activation_bytes": int(activation),
        # The derived terms themselves, so a record built from this budget
        # cannot state a graph pool the budget was not computed with.
        "peak_torch": int(readings.get("peak_torch") or 0),
        "non_torch": int(readings.get("non_torch") or 0),
        "cudagraph_overhead": int(readings.get("cudagraph_overhead") or 0),
    }


def profile_reader(collect: MutableSequence, *, coords=None):
    """The readers a modelled budget loads its profile through.

    Both sizing paths -- the GPU-free replay and the device-backed runner in
    modelled mode -- read the same artifacts for the same reason, so they read
    them the same way: one open, the digest of those bytes, `json.loads` of
    those same bytes, appended to `collect` as it happens. A caller that only
    needs the readings still leaves behind a record of what produced them.

    Returns `(read, load_referenced, payloads)`. `read(requested, role)` takes
    a role; `load_referenced(where)` is the one-argument loader
    `derived_readings` calls for the files a profile names, and it nests their
    roles under the profile -- a calibration is an input in its own right, and
    a validator that saw only the profile could not tell that the numbers
    behind it were read at all. `payloads` keeps the first payload per role for
    a caller that wants what it said as well as that it was read.
    """
    payloads: dict = {}

    def read(requested, role):
        """One open, one digest, one parse -- the shared helper's contract."""
        payload, record = load_json(requested, role=role, coords=coords)
        collect.append(record)
        payloads.setdefault(role, payload)
        return payload

    def load_referenced(where):
        profile = payloads.get(PROFILE_ROLE) or {}
        role = PROFILE_ROLE + ".referenced"
        if isinstance(where, str) and isinstance(profile, Mapping):
            for field in ("calibration", "model_config"):
                if where == profile.get(field):
                    role = "%s.%s" % (PROFILE_ROLE, field)
                    break
        return read(where, role)

    return read, load_referenced, payloads


def derived_block_info(
    path: str,
    config,
    *,
    state_runtime: Optional[Mapping[str, Any]] = None,
    captured: Optional[Mapping[str, Any]] = None,
    coords: Optional[Mapping[str, int]] = None,
    collect: Optional[MutableSequence] = None,
    lineage: Optional[dict] = None,
) -> dict:
    """`get_num_blocks`' reply, derived from a profile instead of measured.

    `config` is the deployment's own configuration -- the live one, not the
    captured one, because the point is to size the run being replayed.
    `state_runtime` is carried through from the capture: it describes how state
    is transferred and checkpointed, which is a property of the architecture
    and the flags rather than of the budget, and nothing here can derive it.
    `captured` is used only to say when the derived pool has a different shape
    than the recorded one.

    `collect` is a list the caller keeps. Every file this reads -- the profile
    and the calibration and model config it names -- is appended to it as a
    `LoadedInput`: the digest of the exact bytes that were parsed, taken at the
    read. Nothing is reopened afterwards, so what the caller retains is what
    this process loaded and not what is at those paths later. The list is
    appended to as the reads happen, so a refusal leaves behind what had been
    read before the run stopped, which is evidence about the run that stopped.

    `coords` is passed through to the loader's rank resolution unchanged; the
    caller decides whether these artifacts are per-rank (see the note at the
    runner's call site -- they are not).

    `lineage` is a dict this fills in on success with what the derivation was
    derived *from*: the width it is for, how the profile was compiled, and the
    calibration's own per-term provenance strings. A source-derived budget is
    a legitimate answer -- it is the whole point of a GPU-free replay -- but
    only if it can say this much about itself.

    Raises `UnfoundedPrediction` when the profile cannot answer, and lets
    ATOM's own `InsufficientPoolBudget` through when the budget leaves nothing
    to page with -- a configuration Compass calls infeasible has to be refused
    by the engine's arithmetic and carry the engine's error.
    """
    collect = collect if collect is not None else []
    read, load_referenced, payloads = profile_reader(collect, coords=coords)

    if not (path or "").strip():
        _refuse("no memory profile was named")

    try:
        profile = read(path, PROFILE_ROLE)
    except FileNotFoundError:
        _refuse("no memory profile at %r" % path)
    except (OSError, ValueError) as exc:
        _refuse("the memory profile at %r could not be read (%s)" % (path, exc))
    if not isinstance(profile, Mapping):
        _refuse("the memory profile at %r is not an object" % path)

    world = _world_size(config)

    # Before the readings, not after: the activation term reads this same
    # config for GDN widths, so a checkpoint the geometry does not describe
    # raises a `KeyError` from inside the walk rather than saying which
    # architecture it is. Checked here, the run is told.
    native_path = profile.get("model_config")
    if not native_path:
        _refuse("the memory profile at %r names no `model_config`, so the KV "
                "geometry has no checkpoint to read" % path)
    try:
        # A profile may carry the checkpoint geometry inline. Then there is no
        # file to digest and nothing to record separately: those bytes are the
        # profile's own, already covered by its digest.
        native = (native_path if isinstance(native_path, Mapping)
                  else read(native_path, PROFILE_ROLE + ".model_config"))
    except (OSError, ValueError) as exc:
        _refuse("the model config the profile names (%r) could not be read (%s)"
                % (native_path, exc))

    model_type = str(text_config(native).get("model_type") or "")
    if model_type not in GDN_HYBRID_MODEL_TYPES:
        _refuse("the KV geometry here describes the GDN hybrid and this "
                "checkpoint is %r. A dense model has no state pool and an MLA "
                "block is a different shape, so this would size a pool the "
                "deployment does not have" % model_type)
    disagreement = layer_types_disagree(native)
    if disagreement:
        logger.warning("ATOMCompass WARNING: %s", disagreement)

    readings, activation = derived_readings(
        profile,
        warmup_tokens=warmup_tokens(config),
        load=load_referenced,
        enforce_eager=bool(getattr(config, "enforce_eager", False)),
        world_size=world,
        source=path,
    )

    block_size = int(getattr(config, "kv_cache_block_size", 0)
                     or getattr(config, "block_size", 0) or 16)
    kv_bytes = KV_DTYPE_BYTES.get(
        str(getattr(config, "kv_cache_dtype", "auto")), 2)
    spec = getattr(config, "speculative_config", None)
    num_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0

    # Not caught: `InsufficientPoolBudget` is the refusal for an infeasible
    # budget and it is the engine's own.
    plan = blocks_from_readings(
        native,
        MemoryReadings(total=readings["total"], free=readings["free"],
                       peak_torch=readings["peak_torch"],
                       non_torch=readings["non_torch"],
                       cudagraph_overhead=readings["cudagraph_overhead"]),
        utilization=float(getattr(config, "gpu_memory_utilization", 0.0) or 0.0),
        max_num_seqs=int(getattr(config, "max_num_seqs", 0) or 0),
        tensor_parallel=world,
        block_size=block_size,
        kv_dtype_bytes=kv_bytes,
        num_spec=num_spec,
    )
    if int(plan.paged_entries) <= 0:
        _refuse("the modelled budget leaves 0 KV blocks (peak_torch %.2fGB, "
                "non_torch %.2fGB, cudagraph %.2fGB of %.2fGB at utilization "
                "%s)" % (readings["peak_torch"] / 2**30,
                         readings["non_torch"] / 2**30,
                         readings["cudagraph_overhead"] / 2**30,
                         readings["total"] / 2**30,
                         getattr(config, "gpu_memory_utilization", None)))

    if state_runtime is None:
        _refuse("the replay target carries no `state_runtime`, and how state "
                "is transferred and checkpointed is a property of the build "
                "rather than of the budget -- there is nothing to derive it "
                "from here")

    entries = dict(plan.entries)
    if captured:
        was = set((captured.get("pool_entries") or {}))
        if was and was != set(entries):
            logger.warning(
                "ATOMCompass WARNING: the modelled pool has entry classes %s "
                "and the captured target has %s. `state_runtime` is the "
                "capture's, so it may name a class this pool does not have.",
                sorted(entries), sorted(was))

    logger.info(
        "ATOMCompass: KV capacity modelled, not replayed: %d blocks from %s "
        "(TP=%d, peak_torch %.2fGB, non_torch %.2fGB, activations %.2fGB over "
        "%d warmup tokens)", plan.paged_entries, path, world,
        readings["peak_torch"] / 2**30, readings["non_torch"] / 2**30,
        activation / 2**30, warmup_tokens(config))

    if lineage is not None:
        lineage.update(derivation_lineage(
            profile, payloads.get(PROFILE_ROLE + ".calibration"),
            path=path, world=world, activation=activation, readings=readings))

    return {
        "num_kvcache_blocks": int(plan.paged_entries),
        "pool_entries": entries,
        "pool_entries_per_req": dict(plan.entries_per_req),
        "state_runtime": dict(state_runtime),
    }
