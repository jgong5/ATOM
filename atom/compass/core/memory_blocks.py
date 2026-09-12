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

import hashlib
import json
import logging
import os
from typing import Any, Callable, Mapping, Optional

from atom.compass.core.kv_geometry import (
    GDN_HYBRID_MODEL_TYPES,
    blocks_from_readings,
    layer_types_disagree,
    text_config,
)
from atom.compass.core.memory import MemoryReadings
from atom.compass.core.memory_model import UnfoundedPrediction, derived_readings

logger = logging.getLogger(__name__)

__all__ = ["warmup_tokens", "derived_block_info", "load_json", "LoadedInputs"]

#: The manifest `LoadedInputs.manifest()` produces. Versioned because a
#: validator will read it and a field that quietly changes meaning is worse
#: than one that is absent.
LOADED_INPUTS_SCHEMA = "compass.memory.loaded_inputs/1"

#: What a `kv_cache_dtype` costs per element. Mirrors the table
#: `scripts/compass/validate_memory.py` reads records with; both exist because
#: the flag names a dtype and the geometry wants a width.
KV_DTYPE_BYTES = {
    "bf16": 2, "fp16": 2, "float16": 2, "bfloat16": 2,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "auto": 2, "None": 2, "": 2,
}


def load_json(path: str):
    """Read a path and parse it. The file-system half, kept injectable."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class LoadedInputs:
    """Every file a sizing actually read, digested at the moment it was read.

    The run's own flags are hashed, but `replay_target` and `memory_model` name
    *files* and the names are not the inputs -- the bytes are. A path can be
    rewritten between the run and the report, can be a symlink, can be named by
    one option and read through another. So this digests what the reader
    consumed, in the order it consumed it, and never re-opens anything: a
    manifest built by re-reading at report time attests to whatever is on disk
    then, which is exactly the claim it appears to rule out.

    A file read twice is recorded twice, in order, rather than deduplicated:
    two reads of one path are two events, and a validator that saw one entry
    could not tell that the bytes were the same both times.

    One instance covers one sizing. It seals when that sizing finishes, so a
    later read cannot append to a manifest that has already been published.
    The record survives a refusal on purpose -- what was read before a run
    stopped is evidence about the run that stopped.
    """

    def __init__(self, load: Optional[Callable[[str], Any]] = None) -> None:
        #: A caller-supplied reader (tests, and callers with their own file
        #: system). Its bytes are not ours to digest, and that is recorded
        #: rather than papered over with a second read of our own.
        self._load = load
        self._reads: list = []
        self._sealed = False

    def read(self, where: str, role: str = "referenced"):
        if self._sealed:
            raise RuntimeError(
                "ATOMCompass: this loaded-input manifest is sealed. It "
                "describes one sizing, and a read after that sizing belongs "
                "to a different one.")
        entry = {"role": role, "path": str(where), "order": len(self._reads)}
        try:
            entry["abspath"] = os.path.abspath(str(where))
        except (OSError, ValueError):                            # noqa: BLE001
            entry["abspath"] = None
        if self._load is not None:
            value = self._load(where)
            entry["sha256"] = None
            entry["bytes"] = None
            entry["note"] = ("read through a caller-supplied loader; the "
                             "bytes were never in this process")
        else:
            with open(where, "rb") as fh:
                raw = fh.read()
            value = json.loads(raw.decode("utf-8"))
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
            entry["bytes"] = len(raw)
        self._reads.append(entry)
        return value

    def note(self, role: str, **fields) -> None:
        """Record an input that was not a file -- carried inline, or absent."""
        if self._sealed:
            raise RuntimeError("ATOMCompass: this manifest is sealed")
        entry = {"role": role, "path": None, "abspath": None,
                 "sha256": None, "bytes": None, "order": len(self._reads)}
        entry.update(fields)
        self._reads.append(entry)

    def seal(self) -> None:
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def digest_of(self, role: str) -> Optional[str]:
        for entry in self._reads:
            if entry["role"] == role:
                return entry["sha256"]
        return None

    def manifest(self, **context) -> dict:
        """The record, copied out. Callers cannot edit what is kept here."""
        return {
            "schema": LOADED_INPUTS_SCHEMA,
            "sealed": self._sealed,
            "reads": [dict(entry) for entry in self._reads],
            **context,
        }


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


def derived_block_info(
    path: str,
    config,
    *,
    state_runtime: Optional[Mapping[str, Any]] = None,
    captured: Optional[Mapping[str, Any]] = None,
    load: Optional[Callable[[str], Any]] = None,
    inputs: Optional["LoadedInputs"] = None,
) -> dict:
    """`get_num_blocks`' reply, derived from a profile instead of measured.

    `config` is the deployment's own configuration -- the live one, not the
    captured one, because the point is to size the run being replayed.
    `state_runtime` is carried through from the capture: it describes how state
    is transferred and checkpointed, which is a property of the architecture
    and the flags rather than of the budget, and nothing here can derive it.
    `captured` is used only to say when the derived pool has a different shape
    than the recorded one.

    `inputs` is a `LoadedInputs` the caller keeps. Every file this reads --
    the profile, the calibration and the model config it names -- is digested
    here, as it is read, and the collector is sealed before this returns. The
    caller then owns an immutable record of the bytes that produced the number,
    which is not the same thing as the paths the flags named.

    Raises `UnfoundedPrediction` when the profile cannot answer, and lets
    ATOM's own `InsufficientPoolBudget` through when the budget leaves nothing
    to page with -- a configuration Compass calls infeasible has to be refused
    by the engine's arithmetic and carry the engine's error.
    """
    inputs = inputs if inputs is not None else LoadedInputs(load)
    try:
        return _derive(path, config, state_runtime=state_runtime,
                       captured=captured, load=load, inputs=inputs)
    finally:
        # Sealed on the way out of either exit. A refusal keeps what it had
        # already read -- that is evidence about the run that stopped -- but
        # nothing may be appended to it afterwards.
        inputs.seal()


def _derive(path, config, *, state_runtime, captured, load, inputs) -> dict:
    """`derived_block_info` without the sealing. See it for the contract."""
    # Everything downstream -- including `derived_readings`, which opens the
    # calibration itself -- reads through the collector, so nothing can be
    # loaded without being digested.
    read = inputs.read

    def load_referenced(where):
        role = "referenced"
        if isinstance(where, str):
            if where == profile.get("calibration"):
                role = "calibration"
            elif where == profile.get("model_config"):
                role = "model_config"
        return read(where, role)

    if not (path or "").strip():
        _refuse("no memory profile was named")

    try:
        profile = read(path, "profile")
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
        if isinstance(native_path, Mapping):
            native = native_path
            inputs.note("model_config", note="carried inline in the profile")
        else:
            native = read(native_path, "model_config")
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

    return {
        "num_kvcache_blocks": int(plan.paged_entries),
        "pool_entries": entries,
        "pool_entries_per_req": dict(plan.entries_per_req),
        "state_runtime": dict(state_runtime),
    }
