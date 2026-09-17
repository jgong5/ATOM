# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The derived KV geometry against block counts a real run produced.

`kv_geometry` mirrors two formulas out of `gdn_attn.py` because reaching the
originals needs a ModelRunner and therefore a device. A mirror can drift, so it
is checked against artifacts rather than against the source it copies: the
records under `memory_records/` are what the 27B actually sized at TP=1, 2 and
4 on node 18 on 2026-09-10, written by `--compass-memory-out`.

**Which way the artifacts are used.** The block counts and the five device
readings are *assertion targets and measurement inputs* -- never calibration.
No constant in `kv_geometry` was fitted to them; the geometry comes from the
checkpoint's own `config.json`, which is beside them here as the vendor shipped
it. That distinction is the whole point of the module: if a number in it had
been tuned to make these tests pass, the tests would prove nothing.

Record provenance (sha256 of the files as copied from
`hjbog-srdc-18:/tmp/xiaobizh-compass/ATOM/agent_scratch/poc/stage0/`):

    27b.tp1.memory.json        6233290081fde7197b221f489145dc74997e2788e19bc...
    27b.tp2.rank0.memory.json  1a9a55b32abe8bb2a2d5248c4b998ff3ef40cbac1ac2...
    27b.tp4.rank0.memory.json  e3ffeeb2766aa0bb8018c751927c47e8343c08e4500b...
    27b.tp4.rank1.memory.json  311848806d2821b994bf74b68d18fbd4240faf910c1e...
    qwen3_5_27b.config.json    191e0af232104ed8b65258cf3fb2b842e288008baca7...
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atom.compass.core.kv_geometry import (
    InsufficientPoolBudget,
    blocks_from_readings,
    gdn_hybrid_specs,
    gdn_state_bytes,
    layer_counts,
    layer_types_disagree,
    paged_block_bytes,
    plan_from_specs,
    text_config,
)
from atom.compass.core.memory import MemoryReadings

RECORDS = Path(__file__).parent / "memory_records"


def _config() -> dict:
    with open(RECORDS / "qwen3_5_27b.config.json", encoding="utf-8") as fh:
        return json.load(fh)


def _record(name: str) -> dict:
    with open(RECORDS / name, encoding="utf-8") as fh:
        return json.load(fh)


def _readings(record: dict) -> MemoryReadings:
    got = record["readings"]
    return MemoryReadings(
        total=got["total"],
        free=got["free"],
        peak_torch=got["peak_torch"],
        non_torch=got["non_torch"],
        cudagraph_overhead=got["cudagraph_overhead"],
    )


ALL_RECORDS = [
    "27b.tp1.memory.json",
    "27b.tp2.rank0.memory.json",
    "27b.tp4.rank0.memory.json",
    "27b.tp4.rank1.memory.json",
]


# ── geometry, from the checkpoint alone ───────────────────────────────────


def test_layer_split_matches_the_checkpoints_own_layer_types():
    """The interval rule and the shipped layout agree for this checkpoint.

    They need not: the engine sizes from the interval and the checkpoint also
    lists every layer's type, and nothing makes them consistent. Here they are,
    which is why sizing from the interval is safe for this model and why the
    check exists for the one where it is not.
    """
    config = _config()
    assert layer_counts(config) == (16, 48)
    assert layer_types_disagree(config) is None

    listed = text_config(config)["layer_types"]
    assert sum(1 for t in listed if t == "full_attention") == 16


def test_layer_types_disagreement_is_reported_not_absorbed():
    config = _config()
    text = dict(text_config(config))
    text["layer_types"] = ["full_attention"] * 64
    assert layer_types_disagree({"text_config": text})


@pytest.mark.parametrize(
    "tp,block,state",
    [
        (1, 1056768, 78446592),
        (2, 528384, 39223296),
        (4, 264192, 19611648),
    ],
)
def test_entry_sizes_halve_with_width(tp, block, state):
    """Both entry classes shard cleanly, and to the byte the run recorded.

    Worth pinning separately from the block count: the two errors that would
    hide in a block count -- a block priced wrong and a state floor priced
    wrong -- move it in the same direction and are not separable from it.
    """
    config = _config()
    assert paged_block_bytes(config, tensor_parallel=tp) == block
    assert gdn_state_bytes(config, tensor_parallel=tp) == state


def test_the_fp32_scale_is_not_optional():
    """The per-block scale is 8 KiB of a 1 MiB block and was once left out.

    0.8% looks like rounding and is not: at TP=1 it is the difference between
    112 740 blocks and 113 620, which is 880 blocks of capacity the engine does
    not have.
    """
    config = _config()
    text = text_config(config)
    cache_only = 2 * 16 * 16 * text["num_key_value_heads"] * text["head_dim"] * 2
    assert paged_block_bytes(config, tensor_parallel=1) - cache_only == 8192


# ── the block count a real run produced ───────────────────────────────────


@pytest.mark.parametrize("name", ALL_RECORDS)
def test_block_count_is_exact_at_every_width(name):
    """Derived geometry plus the engine's own `plan_pools` reproduces the run.

    Exactly -- not within a tolerance. Every input is an integer and every step
    is integer arithmetic, so there is no reason for this to be approximate,
    and a tolerance here would hide the day it stops being exact.
    """
    record = _record(name)
    config, got = _config(), record["blocks"]
    deployed = record["config"]

    plan = blocks_from_readings(
        config,
        _readings(record),
        utilization=deployed["gpu_memory_utilization"],
        max_num_seqs=deployed["max_num_seqs"],
        tensor_parallel=deployed["topology"]["tp"],
        block_size=deployed["block_size"],
    )

    assert plan.paged_entries == got["num_kvcache_blocks"]
    assert plan.entries == got["pool_entries"]
    assert plan.entries_per_req == got["pool_entries_per_req"]


def test_the_ranks_of_one_run_disagree_and_the_geometry_is_not_why():
    """At TP=4 two ranks sized 191 blocks apart, and both are reproduced.

    The geometry is identical on both -- same checkpoint, same width -- so the
    spread is entirely in the readings, and `non_torch` is where it lives: it
    is device-wide used memory, so a neighbour on one card and not another
    moves the budget. Reproducing both counts from one geometry is what makes
    that attribution a measurement rather than a story.
    """
    config = _config()
    counts = {}
    for name in ("27b.tp4.rank0.memory.json", "27b.tp4.rank1.memory.json"):
        record = _record(name)
        deployed = record["config"]
        plan = blocks_from_readings(
            config,
            _readings(record),
            utilization=deployed["gpu_memory_utilization"],
            max_num_seqs=deployed["max_num_seqs"],
            tensor_parallel=4,
            block_size=deployed["block_size"],
        )
        counts[name] = plan.paged_entries
        assert plan.paged_entries == record["blocks"]["num_kvcache_blocks"]

    spread = max(counts.values()) - min(counts.values())
    assert spread == 191

    non_torch = [_record(n)["readings"]["non_torch"] for n in counts]
    assert max(non_torch) - min(non_torch) > 0


# ── infeasibility, refused by ATOM's own arithmetic ───────────────────────


def test_a_state_floor_that_eats_the_budget_raises_atoms_own_error():
    """The rejection has to be the engine's, and it is.

    `plan_pools` is imported, not reimplemented, so this is the same exception
    object `ModelRunner.get_num_blocks` catches and turns into its start-up
    error. A configuration Compass calls infeasible is therefore refused by the
    engine's arithmetic rather than by a rule of Compass's own.
    """
    config = _config()
    specs = gdn_hybrid_specs(config, tensor_parallel=1)
    state_bytes = gdn_state_bytes(config, tensor_parallel=1)

    # 512 concurrent requests at TP=1: 38.2 GB of recurrent state before a
    # single block is paged. Give it a budget one byte short of that floor.
    with pytest.raises(InsufficientPoolBudget) as caught:
        plan_from_specs(specs, 512 * state_bytes - 1, 512)
    assert caught.value.reserved_bytes == 512 * state_bytes
    assert caught.value.entries == 512


def test_a_budget_that_only_just_covers_the_floor_pages_nothing():
    """The boundary is `remaining <= 0`, so covering the floor exactly fails.

    Correct, and worth pinning: a pool of zero blocks is not a deployment that
    runs slowly, it is one that cannot admit a single token.
    """
    config = _config()
    specs = gdn_hybrid_specs(config, tensor_parallel=1)
    floor = 32 * gdn_state_bytes(config, tensor_parallel=1)

    with pytest.raises(InsufficientPoolBudget):
        plan_from_specs(specs, floor, 32)

    plan = plan_from_specs(specs, floor + paged_block_bytes(config), 32)
    assert plan.paged_entries == 1
