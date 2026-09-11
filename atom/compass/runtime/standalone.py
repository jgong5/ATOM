"""Stand up the layers a graph's operators reach, and nothing else.

Attention is the one operator family that cannot be called from its arguments.
``unified_attention_with_output_base`` takes a *layer name* and looks the layer
up in ``compilation_config.static_forward_context``; the layer's implementation
then reads a KV region that is not an argument either. So pricing attention has
so far meant standing up the deployment it came from -- weights loaded, pool
sized for ``max_num_seqs``, scheduler running -- which is exactly the cost the
PoC claims to avoid, and which at TP=1 on one MI308X is not merely expensive but
impossible: the 27B DeltaNet state pool does not fit, so the operators cannot be
measured at all on a card that runs every one of them comfortably.

What the layer actually needs is much narrower:

* to exist, under the name the graph recorded. Layers self-register from their
  own ``__init__``, so *building the model* is what registers them -- and a model
  built on ``meta`` registers exactly the same names as one built on a card.
* its own small parameters: the rope tables, the q/k norms, the DeltaNet
  convolution, ``dt_bias`` and ``A_log``. Not the 27 billion in the projections.
* somewhere to address. A KV region and a state pool sized to *this graph's own
  batch* -- four requests and their block tables -- rather than to the 512
  concurrent requests a deployment reserves for.

So this module builds the model on meta, materialises the attention subtrees
alone, and asks ATOM's own attention builder to lay out and bind the caches at
the footprint the graph records. The sizing is not reimplemented here: the
builder is the real one, driven through a runner shim that exposes the handful
of attributes it reads. What is deliberately different is how *large* the pools
are, and that is the point -- a price is a per-call cost, and the calls are the
ones the graph made.

Weights are random, as they are under ``--load-dummy`` in the collector this
replaces. Nothing here produces an output anyone reads; only times.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Cache class the GDN state pool is allocated under. Imported lazily at use.
_STATE_SLOT_CLASS = "state"


def _real_like(tensor, device):
    """A device tensor with ``tensor``'s shape and dtype, plausibly filled.

    Values never leave this process -- only times do -- but they cannot be
    arbitrary: uninitialised memory reads as denormals or NaNs often enough to
    move a kernel's cost, and a scale is a scale. Multi-element float tensors get
    small normals, scalars get one, integers get zero.
    """
    import torch

    out = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
    if out.dtype.is_floating_point:
        if out.numel() > 1:
            out.normal_(0.0, 0.02)
        else:
            out.fill_(1.0)
    else:
        out.zero_()
    return out


def _materialise(module, device) -> int:
    """Give every meta tensor under ``module`` real storage. Returns how many.

    Three kinds, because a layer holds all three: registered parameters,
    registered buffers, and plain tensor attributes. The last is not a detail --
    ``PagedAttentionImpl`` keeps its dequant scale as an ordinary attribute, and
    a module whose parameters are real and whose scale is still on meta fails
    inside the kernel rather than here.
    """
    import torch

    moved = 0
    for sub in module.modules():
        for name, param in list(sub._parameters.items()):
            if param is not None and param.is_meta:
                sub._parameters[name] = torch.nn.Parameter(
                    _real_like(param, device), requires_grad=False)
                moved += 1
        for name, buf in list(sub._buffers.items()):
            if buf is not None and buf.is_meta:
                sub._buffers[name] = _real_like(buf, device)
                moved += 1
        for name, value in list(sub.__dict__.items()):
            if isinstance(value, torch.Tensor) and value.is_meta:
                setattr(sub, name, _real_like(value, device))
                moved += 1
        # Modules that recorded the device they were built on recorded "meta".
        built = getattr(sub, "device", None)
        if isinstance(built, str) and built.startswith("meta"):
            sub.device = str(device)
        elif isinstance(built, torch.device) and built.type == "meta":
            sub.device = device
    return moved


def _census(model, attn) -> dict[str, Any]:
    """Count what got storage and what stayed on meta, in tensors and bytes.

    The claim this collector makes is not "no parameters were materialised" --
    attention needs its own -- but "the target network was not". That is only
    checkable if both sides of the split are counted, so both are: everything
    under an attention module, and everything else. A tensor still on meta has
    ``numel * itemsize`` of shape but no storage, and it is reported as the
    storage *avoided*, not as storage held.
    """
    import torch

    inside = {id(sub) for module in attn for sub in module.modules()}
    tally: dict[str, Any] = {
        "attention": {"tensors": 0, "on_device": 0, "bytes": 0},
        "rest": {"tensors": 0, "on_meta": 0, "bytes_avoided": 0},
    }
    names: set[str] = set()
    for path, sub in model.named_modules():
        where = "attention" if id(sub) in inside else "rest"
        held = list(sub._parameters.items()) + list(sub._buffers.items())
        held += [(n, v) for n, v in sub.__dict__.items()
                 if isinstance(v, torch.Tensor)]
        for name, tensor in held:
            if tensor is None:
                continue
            size = tensor.numel() * tensor.element_size()
            tally[where]["tensors"] += 1
            if tensor.is_meta:
                if where == "rest":
                    tally["rest"]["on_meta"] += 1
                    tally["rest"]["bytes_avoided"] += size
            elif where == "attention":
                tally["attention"]["on_device"] += 1
                tally["attention"]["bytes"] += size
                names.add(f"{path.split('.')[-1]}.{name}" if path else name)
    tally["attention"]["names"] = sorted(names)
    return tally


def _footprint(ops) -> dict[str, int]:
    """How much cache the recorded calls address: blocks, and state slots.

    Read from the contexts the graph carries, not assumed. A block table that
    tops out at 65 needs 66 blocks and not one more, and the state indices say
    the same thing for the DeltaNet pool. Pricing rotates attention over
    ``KV_VARIANTS`` disjoint copies of that footprint so a repeated call does not
    run entirely out of cache (see ``forward_ctx._install_attention``), so the
    blocks are multiplied by it here -- the rotation is clamped to what is
    allocated, and clamping it to one would silently price attention warm.
    """
    from atom.compass.runtime import forward_ctx
    from atom.compass.runtime.microbench import KV_VARIANTS

    blocks, slots = 0, 0
    for op in ops:
        name = op.get("name")
        recorded = op.get("context")
        if not recorded or not forward_ctx.is_context_dependent(name):
            continue
        ctx = {k: v for k, v in (tuple(x) for x in recorded)}
        flat = ctx.get("block_tables") or []
        if flat:
            blocks = max(blocks, max(flat) + 1)
        for field in ("spec_state_indices_tensor",
                      "non_spec_state_indices_tensor",
                      "non_spec_state_indices_in_tensor"):
            held = ctx.get(field)
            if not held:
                continue
            values = held[0]
            if values:
                slots = max(slots, max(int(v) for v in values) + 1)
    return {"blocks": blocks * max(1, KV_VARIANTS), "slots": slots,
            "blocks_per_variant": blocks, "variants": max(1, KV_VARIANTS)}


class _RunnerShim:
    """The attributes ATOM's attention builder reads off a ModelRunner.

    Borrowed rather than reimplemented: the four derivations that matter --
    per-rank KV heads, total layers, which hybrid family this is, and the
    sub-pool declarations -- are ModelRunner's own functions, bound to this
    object. Everything else is a scalar copied from the config, which is where
    the runner reads it from too.

    What this object is NOT is a runner. It has no scheduler, no batch, no
    forward, and no pool plan; anything the builder reaches for beyond the list
    below will raise, which is the intended failure -- a silent default here
    would be a sizing decision nobody made.
    """

    def __init__(self, config, device, blocks: int):
        import torch

        from atom.model_engine.model_runner import ModelRunner
        from atom.utils import get_hf_text_config

        self.config = config
        self.device = torch.device(device)
        self.block_size = config.kv_cache_block_size
        self.kv_cache_dtype = config.kv_cache_dtype
        self.world_size = config.tensor_parallel_size
        self.hf_text_config = get_hf_text_config(config.hf_config)
        self.max_bs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.forward_vars: dict[str, Any] = {}
        self.num_spec_tokens = 0
        self.tokenID_processor = None
        self.state_runtime = None
        # No draft model, so every module's layer_id is a target layer.
        self.mtp_start_layer_idx = config.hf_config.num_hidden_layers
        self._sparse_attention_cache_next = 0
        self._kv_layer_cache_store = []
        # Sized to the graph, not to a deployment. Set before the builder is
        # constructed because `allocate_kv_cache_tensors` reads it directly.
        self.num_physical_kvcache_blocks = blocks
        self.num_kv_heads = None  # filled once the builder can be asked

        self._runner_cls = ModelRunner

    # ModelRunner's own, unbound. If one of them grows a dependency this shim
    # does not carry, it raises here rather than quietly answering differently
    # from the deployment.
    def _get_num_kv_heads(self):
        return self._runner_cls._get_num_kv_heads(self)

    def _get_total_num_layers(self):
        return self._runner_cls._get_total_num_layers(self)

    def is_qwen_next(self):
        return self._runner_cls.is_qwen_next(self)

    def is_kimi_linear(self):
        return self._runner_cls.is_kimi_linear(self)

    def is_mimo_v2(self):
        return self._runner_cls.is_mimo_v2(self)

    def is_deepseek_v4(self):
        return self._runner_cls.is_deepseek_v4(self)

    def is_deepseek_mla(self):
        return self._runner_cls.is_deepseek_mla(self)


def build_model_on_meta(config):
    """Build the target model on meta, which is what registers its layers.

    The registration is a side effect of every attention layer's ``__init__``
    (`base_attention.py`, `paged_attention.py`), so there is no lighter way to
    populate ``static_forward_context`` than to construct the model -- and no
    need for a heavier one, because meta allocates nothing. This is the same
    build `scripts/compass/graph_diff.py` derives a graph from.
    """
    import torch

    from atom.model_engine.model_runner import support_model_arch_dict
    from atom.utils import resolve_obj_by_qualname

    arch = config.hf_config.architectures[0]
    model_class = resolve_obj_by_qualname(support_model_arch_dict[arch])
    # The runner remaps quantised layer names before constructing, and the
    # constructor reads the result; skipping it builds a different model.
    config.quant_config.remap_layer_name(
        config.hf_config,
        packed_modules_mapping=getattr(model_class, "packed_modules_mapping", {}),
        quant_exclude_name_mapping=getattr(
            model_class, "quant_exclude_name_mapping", {}),
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype)
    try:
        with torch.device("meta"):
            model = model_class(config)
    finally:
        torch.set_default_dtype(prev_dtype)
    return model


def _attention_modules(model) -> list:
    """Every module the two attention custom ops can be dispatched to."""
    return [m for m in model.modules()
            if hasattr(m, "base_attention") or hasattr(m, "base_linear_attention")]


def _bind_caches(shim, model, builder, slots: int) -> dict[str, int]:
    """Allocate and bind the caches, through the builder that owns their layout.

    A transcription of `ModelRunner.allocate_kv_cache`'s binding loop with the
    deployment removed: same builder, same `build_kv_cache_tensor` per module,
    same `layer_{layer_num}` keys the attention reads back. The pool sizes are
    this graph's; the layouts are ATOM's.
    """
    from atom.utils.forward_context import set_kv_cache_data

    import torch

    def _install(pools: dict, tally: dict, shapes: dict) -> None:
        for name, value in pools.items():
            setattr(shim, name, value)
            if isinstance(value, torch.Tensor):
                tally[name] = value.numel() * value.element_size()
                # The shape, not a byte count someone would have to factorise
                # back into layers and planes. A KV tensor's leading 2 is the
                # k/v pair and the next axis is the full-attention layers; the
                # two are easy to conflate and impossible to conflate here.
                shapes[name] = [list(value.shape), str(value.dtype)]

    bytes_by_pool: dict[str, int] = {}
    shape_by_pool: dict[str, list] = {}
    num_kv_heads = shim._get_num_kv_heads()
    shim.num_kv_heads = num_kv_heads
    _install(builder.allocate_kv_cache_tensors(num_kv_heads, 0), bytes_by_pool,
             shape_by_pool)
    if slots:
        entries = {_STATE_SLOT_CLASS: slots}
        _install(builder.allocate_per_req_cache(entries), bytes_by_pool,
                 shape_by_pool)

    tensors, keys, layer_id = [], [], 0
    for module in model.modules():
        bound = builder.build_kv_cache_tensor(layer_id, module)
        if bound is not None:
            tensors.append(bound)
            keys.append(getattr(module, "layer_num", layer_id))
            layer_id += 1
    set_kv_cache_data({f"layer_{k}": t for k, t in zip(keys, tensors)})
    return {"bound_layers": len(tensors), "kv_heads": num_kv_heads,
            "pool_bytes": bytes_by_pool, "pool_shapes": shape_by_pool,
            "pool_bytes_total": sum(bytes_by_pool.values())}


def stand_up_layers(config, graph_path: str, mode: str) -> dict[str, Any]:
    """Make the graph's attention operators callable. Returns what was done.

    ``mode='meta'`` registers the layers and stops: enough for the custom op to
    find its layer, not enough for the layer to run, which is a useful halfway
    point when the question is whether registration alone is the blocker.
    ``mode='attention'`` also materialises the attention subtrees and binds
    caches at the graph's own footprint.

    The dict it returns goes verbatim into the price list's provenance, so a
    reader never has to infer what stood behind a number: a one-line summary,
    and under it the census and the device bytes that make the summary
    checkable rather than merely asserted.
    """
    import time

    import torch

    from atom.compass.runtime.microbench import load_ops

    if mode not in ("meta", "attention"):
        raise ValueError(f"unknown stand-up mode {mode!r}")

    # From here, peak allocation is this collector's own. Anything already held
    # belongs to the process group, and is reported separately below.
    device = torch.device("cuda", torch.cuda.current_device())
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    model = build_model_on_meta(config)
    registered = len(config.compilation_config.static_forward_context)
    build_s = time.perf_counter() - t0
    logger.info("ATOMCompass: registered %d attention layers from a meta build "
                "in %.1fs", registered, build_s)
    if mode == "meta":
        return {
            "summary": (f"{registered} layers registered from a meta build, "
                        f"no parameters and no caches"),
            "registered_layers": registered,
            "census": _census(model, []),
            "device_bytes": {"before_standup": before,
                             "after_standup": torch.cuda.memory_allocated(device),
                             "peak_standup": torch.cuda.max_memory_allocated(device)},
            "seconds": {"build": round(build_s, 2)},
        }

    ops, paths, _ = load_ops(graph_path)
    foot = _footprint(ops)

    attn = _attention_modules(model)
    moved = sum(_materialise(m, device) for m in attn)

    shim = _RunnerShim(config, device, foot["blocks"])
    from atom.utils.selector import get_attn_backend

    backend = get_attn_backend(
        shim.block_size,
        use_mla=shim.is_deepseek_mla(),
        use_gdn=shim.is_qwen_next(),
        use_v4=shim.is_deepseek_v4(),
        use_kimi_mla=shim.is_kimi_linear(),
    )
    builder = backend.get_builder_cls()(model_runner=shim)
    shim.physical_block_size = builder.block_size
    shim.num_physical_kvcache_blocks = foot["blocks"] * builder.block_ratio
    config.num_kvcache_blocks = foot["blocks"]
    bound = _bind_caches(shim, model, builder, foot["slots"])

    # Kept alive for the process's lifetime: the caches are reached through the
    # module attributes and the KV context, and a shim that fell out of scope
    # would take the only reference to the pools with it.
    globals()["_STANDING"] = (model, shim, builder)

    census = _census(model, attn)
    summary = (f"{registered} layers registered, {len(attn)} attention modules "
               f"materialised ({moved} tensors), {bound['bound_layers']} bound "
               f"to {shim.num_physical_kvcache_blocks} blocks "
               f"({foot['blocks_per_variant']} per variant x {foot['variants']})"
               f" and {foot['slots']} state slots, from {len(paths)} graph(s)")
    return {
        "summary": summary,
        "registered_layers": registered,
        "attention_modules": len(attn),
        "tensors_materialised": moved,
        "census": census,
        "caches": {**{k: v for k, v in bound.items() if k != "pool_bytes"},
                   "pool_bytes": bound["pool_bytes"],
                   "blocks": shim.num_physical_kvcache_blocks,
                   "blocks_per_variant": foot["blocks_per_variant"],
                   "variants": foot["variants"], "state_slots": foot["slots"],
                   "variant_policy": (
                       "attention priced against N disjoint copies of the "
                       "recorded footprint, rotated; N from COMPASS_KV_VARIANTS"
                       if foot["variants"] > 1 else
                       "attention priced against a single copy of the recorded "
                       "footprint: no rotation, so a repeated call may find the "
                       "region resident that a deployment would not"),},
        "device_bytes": {
            "before_standup": before,
            "after_standup": torch.cuda.memory_allocated(device),
            "peak_standup": torch.cuda.max_memory_allocated(device),
        },
        "graphs": [str(p) for p in paths],
        "seconds": {"build": round(build_s, 2)},
    }
