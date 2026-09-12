"""**analytical** -- the KV pool's geometry, from the model's own config file.

The four terms in `memory_model` say what a configuration spends on everything
that is *not* KV. What is left is the KV budget, and turning a byte budget into
a block count is the last step of sizing -- the one the whole decision rests on,
because a configuration's capacity is its block count and nothing else.

That step is already pure arithmetic in ATOM: `plan_pools` takes entry-class
declarations and a byte budget and needs no device, no config and no env. So
the only thing missing on a box with no GPU is the *declarations*, and a
declaration is two numbers: what one paged block costs, and what one in-flight
request's state costs. Both are geometry. `config.json` has every input.

**Why this is not the engine's own code.** `GDNAttentionMetadataBuilder.
sub_pool_specs` computes the same two numbers, and reaching it means building a
ModelRunner, which means a device. So the two formulas are mirrored here, from

    atom/model_ops/attentions/gdn_attn.py
        GDNAttentionMetadataBuilder.sub_pool_specs   -- the paged block
        GDNStateMixin.state_spec / _state_shape      -- the state slot

and the sizing itself is *not* mirrored: `plan_pools` is imported and called,
because it is the part that is already portable. A mirror can drift from its
source, which is why `tests/compass/test_kv_geometry.py` checks these against
recorded artifacts rather than against the formulas they came from.

**What this deliberately does not read.** Nothing from a capture. The block
counts a run recorded are what this is checked *against*; none of them, and no
constant fitted to them, is an input. The inputs are `config.json`, the
deployment's own flags, and -- for `blocks_from_readings` -- the five device
readings, which are the configuration's measurement and not its answer.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_ops.attentions.sub_pool_spec import (
    InsufficientPoolBudget,
    PoolPlan,
    page_pool,
    plan_pools,
    state_pool,
)

__all__ = [
    "text_config",
    "layer_counts",
    "layer_types_disagree",
    "paged_block_bytes",
    "gdn_state_bytes",
    "gdn_hybrid_specs",
    "plan_from_specs",
    "blocks_from_readings",
    "InsufficientPoolBudget",
    "GDN_HYBRID_MODEL_TYPES",
]

#: Model types whose attention is the GDN hybrid these formulas describe: a
#: paged pool over the full-attention layers only, and a per-request recurrent
#: state for the linear-attention ones. Named rather than assumed -- a dense
#: model has no state pool and an MLA model's block is a different shape, and
#: silently applying this to either would produce a plausible wrong number.
GDN_HYBRID_MODEL_TYPES = frozenset({"qwen3_5_text", "qwen3_5", "qwen3_next"})

#: The engine holds this back before anything else (`model_runner.py`), and
#: `core/memory.py` says the same. Repeated rather than imported to keep the
#: two sides of the project from depending on each other.
SAFETY_FRACTION = 0.02


def text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The language model's own config, for a checkpoint that nests one.

    A multimodal checkpoint keeps the decoder under `text_config` and puts the
    vision tower beside it. Every geometry term here is the decoder's, and the
    engine reads the same nested block as its `hf_text_config`.
    """
    nested = config.get("text_config")
    return nested if isinstance(nested, Mapping) else config


def layer_counts(config: Mapping[str, Any]) -> tuple:
    """``(full_attention, linear_attention)`` layer counts.

    `GDNStateMixin.__init__` takes the full-attention count as
    `num_hidden_layers // full_attention_interval` and the linear count as the
    rest -- an interval, not the `layer_types` list, even though the checkpoint
    carries both. The same rule is used here so the two agree; `layer_types` is
    read only to say when it does not, which would mean the engine is sizing
    for a layout the checkpoint does not have.
    """
    text = text_config(config)
    layers = int(text.get("num_hidden_layers") or 0)
    interval = int(text.get("full_attention_interval") or 0)
    if not (layers and interval):
        return 0, 0
    full = layers // interval
    return full, layers - full


def layer_types_disagree(config: Mapping[str, Any]) -> Optional[str]:
    """Why the interval rule and the checkpoint's `layer_types` differ, or None.

    Cheap, and worth having: the interval rule is what the engine sizes with,
    so if a checkpoint ever ships a layout that is not strictly every-nth, the
    pool is sized for a model that is not there and every term downstream of it
    is wrong in a way no memory reading would reveal.
    """
    text = text_config(config)
    listed = text.get("layer_types")
    if not isinstance(listed, (list, tuple)) or not listed:
        return None
    full_listed = sum(1 for t in listed if t == "full_attention")
    full_rule, _ = layer_counts(config)
    if full_listed == full_rule and len(listed) == int(
        text.get("num_hidden_layers") or 0
    ):
        return None
    return (
        "checkpoint lists %d full-attention layers of %d, the interval "
        "rule sizes for %d" % (full_listed, len(listed), full_rule)
    )


def paged_block_bytes(
    config: Mapping[str, Any],
    *,
    tensor_parallel: int = 1,
    block_size: int = 16,
    kv_dtype_bytes: int = 2,
    num_draft_layers: int = 0,
) -> int:
    """What one paged KV block costs, on one rank.

    Two tensors, not one. The cache itself is
    ``[2, n_full, blocks, block_size, kv_heads, head_dim]`` at the KV dtype,
    and beside it a per-block scale of ``[2, n_full, blocks, kv_heads,
    block_size]`` in fp32 -- which is 0.8% of the block at bf16 and was the
    whole of the gap when the scale was left out.

    Only the full-attention layers are paged: a linear-attention layer keeps a
    recurrent state per request instead, which is `gdn_state_bytes`.
    """
    text = text_config(config)
    full, _ = layer_counts(config)
    n_full = full + int(num_draft_layers)
    kv_heads = int(text.get("num_key_value_heads") or 0) // max(1, tensor_parallel)
    head_dim = int(text.get("head_dim") or 0)
    if not (n_full and kv_heads and head_dim and block_size):
        return 0
    cache = 2 * n_full * block_size * kv_heads * head_dim * int(kv_dtype_bytes)
    scale = 2 * n_full * kv_heads * block_size * 4
    return cache + scale


def gdn_state_bytes(
    config: Mapping[str, Any],
    *,
    tensor_parallel: int = 1,
    state_dtype_bytes: int = 2,
    num_spec: int = 0,
) -> int:
    """What one in-flight request's recurrent state costs, on one rank.

    Per linear-attention layer, a convolution window and a temporal state:

        conv     ``[conv_kernel_dim - 1 + num_spec, conv_dim / tp]``
        temporal ``[v_heads / tp, v_head_dim, k_head_dim]``

    where ``conv_dim = k_head_dim * k_heads * 2 + v_head_dim * v_heads`` -- the
    two being the query and key halves of the gated delta rule, which is why
    the key term is doubled and the value term is not.

    Both at the model's own dtype. `kimi_linear` keeps the temporal half in
    fp32 and is not one of these model types, so the single `state_dtype_bytes`
    is enough here; a second dtype would be needed to extend this to it.

    This is the term that makes a hybrid's floor large. At TP=1 on the 27B it
    is 74.8 MiB a request, so 32 concurrent requests reserve 2.34 GB before a
    single block is paged -- and it is reserved whether or not the traffic ever
    reaches that concurrency.
    """
    text = text_config(config)
    _, linear = layer_counts(config)
    tp = max(1, tensor_parallel)
    k_heads = int(text.get("linear_num_key_heads") or 0)
    v_heads = int(text.get("linear_num_value_heads") or 0)
    k_dim = int(text.get("linear_key_head_dim") or 0)
    v_dim = int(text.get("linear_value_head_dim") or 0)
    kernel = int(text.get("linear_conv_kernel_dim") or 0)
    if not (linear and k_heads and v_heads and k_dim and v_dim and kernel):
        return 0
    conv_dim = k_dim * k_heads * 2 + v_dim * v_heads
    conv = (kernel - 1 + int(num_spec)) * (conv_dim // tp) * int(state_dtype_bytes)
    temporal = (v_heads // tp) * v_dim * k_dim * int(state_dtype_bytes)
    return linear * (conv + temporal)


def gdn_hybrid_specs(
    config: Mapping[str, Any],
    *,
    tensor_parallel: int = 1,
    block_size: int = 16,
    kv_dtype_bytes: int = 2,
    state_dtype_bytes: int = 2,
    num_spec: int = 0,
    num_draft_layers: int = 0,
) -> list:
    """The entry-class declarations `plan_pools` sizes from.

    `entries_per_req` is `1 + num_spec` for the baseline GDN state, which keeps
    a rollback slot per speculated token (`GDNStateMixin.slots_per_req`).
    ReplaySSM drops it back to 1 and adds a record buffer instead; that
    configuration is not derived here, and passing `num_spec` for a ReplaySSM
    deployment would over-reserve rather than under-reserve.
    """
    return [
        page_pool(
            paged_block_bytes(
                config,
                tensor_parallel=tensor_parallel,
                block_size=block_size,
                kv_dtype_bytes=kv_dtype_bytes,
                num_draft_layers=num_draft_layers,
            )
        ),
        state_pool(
            STATE_SLOT_CLASS,
            gdn_state_bytes(
                config,
                tensor_parallel=tensor_parallel,
                state_dtype_bytes=state_dtype_bytes,
                num_spec=num_spec,
            ),
            entries_per_req=1 + int(num_spec),
        ),
    ]


def plan_from_specs(specs: list, available_bytes: int, max_num_seqs: int) -> PoolPlan:
    """ATOM's own sizing, called rather than reimplemented.

    Raises ATOM's own `InsufficientPoolBudget` when the state floor leaves
    nothing to page with -- which is the point: a configuration Compass calls
    infeasible has to be refused by the engine's arithmetic and carry the
    engine's error, not a verdict of Compass's own.
    """
    return plan_pools(specs, int(available_bytes), int(max_num_seqs))


def blocks_from_readings(
    config: Mapping[str, Any],
    readings,
    *,
    utilization: float,
    max_num_seqs: int,
    tensor_parallel: int = 1,
    block_size: int = 16,
    kv_dtype_bytes: int = 2,
    state_dtype_bytes: int = 2,
    num_spec: int = 0,
    extra_reserve: int = 0,
) -> PoolPlan:
    """The block count a set of readings and this geometry produce together.

    `readings` is anything with the five attributes `MemoryReadings` has, so a
    recorded record and a modelled one are both accepted and neither is
    preferred. The split is the useful part: the geometry half carries no
    fitted constant at all, so when this disagrees with a run, the disagreement
    is in the readings.
    """
    overheads = (
        readings.peak_torch
        + readings.non_torch
        + readings.cudagraph_overhead
        + int(readings.total * SAFETY_FRACTION)
    )
    budget = int(readings.total * utilization) - overheads - int(extra_reserve)
    available = min(budget, readings.free)
    specs = gdn_hybrid_specs(
        config,
        tensor_parallel=tensor_parallel,
        block_size=block_size,
        kv_dtype_bytes=kv_dtype_bytes,
        state_dtype_bytes=state_dtype_bytes,
        num_spec=num_spec,
    )
    return plan_from_specs(specs, available, max_num_seqs)
