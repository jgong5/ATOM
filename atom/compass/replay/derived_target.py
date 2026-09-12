"""The startup answers of a width no device has ever run.

A GPU-free replay needs a `TargetRecord`: the reply to each startup RPC, and
the configuration those replies belong to. Until now the only way to get one
was to capture it off a device running that configuration -- which is exactly
what a wider deployment cannot do, and exactly what a prediction of a wider
deployment must not be founded on.

So this builds the record instead. The block count, the sub-pool entries and
the per-request entries come from `derived_block_info`, which is the memory
model: no term in them came off a card, and none came from a target-engine
capture at the width being derived. What cannot be modelled is borrowed
*explicitly* from a TP=1 source record and named in the lineage -- the state
transfer layout, the cudagraph capture sizes, and the card the deployment is
declared to run on. Those are properties of the architecture, the flags and
the hardware, not of the width; the capacity, which is a property of the
width, is derived.

**The borrow is the whole risk, so it is the whole disclosure.** Every field
this does not derive appears in `lineage["borrowed"]` with the file it came
from. A reader who disagrees that a borrowed field is width-independent can
see precisely which one to argue about, which is not true of a record that
merely looks captured.
"""

from __future__ import annotations

import logging
from typing import Optional

from atom.compass.core.memory_blocks import derived_block_info
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.state_runtime import StateRuntime
from atom.compass.replay.runner import TARGET_VERSION, TargetRecord

logger = logging.getLogger(__name__)

__all__ = ["derive_target", "DERIVED_SCHEMA"]

#: Stamped into the record so a reader never has to infer that a target was
#: modelled. A captured record has no `derivation` block at all.
DERIVED_SCHEMA = "compass.memory.derived_target/1"


def _check_source(layout: TargetRecord, config) -> None:
    """Refuse any layout that is not the POC's TP=1 source record.

    Without this the function is a laundry: point `--source-target` at a TP=4
    *target-engine* capture and every field of it -- a measured block count
    included -- comes back out stamped `source-derived` at TP=4, which is
    precisely the reading the replay's own width check exists to stop. The
    borrow is only defensible from the width the calibration was taken at, so
    that is the only width accepted.

    Two checks, both about the POC's input contract and nothing wider: the
    record is TP=1, and it is a record of the same model. A layout for another
    checkpoint describes another architecture's recurrent state.
    """
    captured = int(layout.config.get("tensor_parallel_size", 0) or 0)
    if captured != 1:
        raise ValueError(
            "ATOMCompass: %s is a TP%d record and only a TP1 source record may "
            "be borrowed from. Its numbers came off a target engine running "
            "TP%d, so deriving from it would stamp a measurement as "
            "source-derived and hand it to a scheduler at a width the replay's "
            "own check would otherwise have refused."
            % (layout.source, captured, captured))
    want = str(getattr(config, "model", "") or "")
    got = str(layout.config.get("model", "") or "")
    if not got:
        raise ValueError(
            "ATOMCompass: %s names no model, so there is nothing to check the "
            "borrowed layout against. An unattributed record is not a source: "
            "the state transfer layout being taken from it is a property of "
            "the architecture that produced it." % layout.source)
    if want and want != got:
        raise ValueError(
            "ATOMCompass: %s records %s and this derivation is for %s. The "
            "state transfer layout and capture shapes being borrowed are "
            "properties of the model that produced them, not of the width."
            % (layout.source, got, want))


def derive_target(
    profile: str,
    config,
    *,
    layout: TargetRecord,
    collect: Optional[list] = None,
) -> dict:
    """A replay target for `config`'s width, derived from `profile`.

    `layout` is the TP=1 *source* record the width-independent fields are
    borrowed from. It is a source-class artifact (the full engine at the
    calibration width), never a target-engine capture of the width being
    derived -- that is the reading this whole path exists to keep out of a
    prediction.

    `collect` is a list every file read is appended to as a `LoadedInput`, so
    the caller can say which bytes produced the record.
    """
    lineage: dict = {}
    _check_source(layout, config)
    state_runtime = layout.blocks.get("state_runtime")
    if not state_runtime:
        raise ValueError(
            "ATOMCompass: %s carries no state_runtime, so there is no state "
            "transfer layout to derive a wider record with. A hybrid model's "
            "recurrent slots are transferred, forked and checkpointed by rules "
            "the memory model does not describe." % layout.source)
    try:
        StateRuntime.from_wire(state_runtime)
    except Exception as exc:  # noqa: BLE001 - the message names the file
        raise ValueError(
            "ATOMCompass: %s carries a state_runtime the engine's own wire "
            "contract will not read back (%s). A layout the target engine "
            "cannot rebuild is not a layout to borrow."
            % (layout.source, exc)) from None

    blocks = derived_block_info(
        profile, config,
        state_runtime=state_runtime,
        coords=None,
        collect=collect if collect is not None else [],
        lineage=lineage)

    width = int(getattr(config, "tensor_parallel_size", 1) or 1)
    capture_sizes = list(layout.graph.get("capture_sizes") or [])
    record = {
        "version": TARGET_VERSION,
        "blocks": blocks,
        "config": {
            "model": str(getattr(config, "model", "")),
            "tensor_parallel_size": width,
            "max_model_len": int(getattr(config, "max_model_len", 0) or 0),
            "max_num_seqs": int(getattr(config, "max_num_seqs", 0) or 0),
            "gpu_memory_utilization": float(
                getattr(config, "gpu_memory_utilization", 0.0) or 0.0),
        },
        "graph": {
            # No device captured graphs for this width, so there is no capture
            # time to state. Zero is what a replay pays, and the field says so
            # rather than carrying a TP=1 duration into a TP=4 timeline.
            "capture_seconds": 0.0,
            "capture_sizes": capture_sizes,
            # The graph pool, as the memory model sizes it at this width --
            # the same term the budget above was computed with, so the record
            # cannot disagree with the budget it came from.
            "pool_bytes": int(lineage.get("cudagraph_overhead") or 0),
        },
        "hardware": dict(layout.hardware or {}),
        "derivation": {
            "schema": DERIVED_SCHEMA,
            "lineage": lineage,
            "borrowed": {
                "from": layout.source,
                # Layout, not capacity. The state transfer is the backend's
                # rule for handing one request's recurrent slot to another and
                # the capture sizes are the batch shapes the compiler was told
                # to graph; neither is a function of tensor-parallel width.
                # The card is a declaration about where this runs.
                "state_runtime": "state transfer and checkpoint layout",
                "capture_sizes": "cudagraph batch shapes",
                "hardware": "the declared card",
            },
        },
    }
    logger.info(
        "ATOMCompass: derived a TP%d replay target from %s -- %d KV blocks, "
        "%d state slots, nothing measured at this width",
        width, profile, int(blocks.get("num_kvcache_blocks") or 0),
        int((blocks.get("pool_entries") or {}).get(STATE_SLOT_CLASS, 0)))
    return record
