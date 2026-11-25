# SPDX-License-Identifier: MIT
"""One rank's meta build, read as the modules the step roofline prices.

`walk(tree)` goes down the tree from its root by ATOM module class, matched by
name so the walk imports none of them, and hands
back one `Module` per module that works on a step, its dims read off the
module's own per-rank parameter shapes and attributes:

- a linear class (`LinearBase`, `torch.nn.Linear`): a GEMM of its weight,
  joining an all-reduce over its TP group when it reduces its result, its
  input width counting both elements of each byte an fp4 weight packs. A
  weight that is not a matrix is the kernel of a depthwise causal convolution
  (GDN keeps its `conv1d` in a column-parallel linear): one pass over its
  channels.
- `VocabParallelEmbedding`, `ParallelLMHead`: a lookup and its all-reduce; a
  GEMM and its all-gather.
- `RMSNorm`, `GemmaRMSNorm`, `RMSNormGated`: one pass over the weight's
  width; an `RMSNorm` built with `fused_allreduce` joins its all-reduce
  inside the norm, and so do the norms MiniMax-M3's decoder reduces the
  sub-layer before them in: a decoder layer's two, but the first layer's
  input norm, and the model's final norm.
- `aiter`'s `RotaryEmbedding`: one pass over `rotary_dim`, once for every
  module that holds it.
- `Attention` (not MLA), `LinearAttention`: attention over the step's rows,
  at their own heads and dims; neither is gone into. An `Attention` with a
  sliding window is unclassified, and so is one whose model plugs in an impl
  other than ATOM's dense `PagedAttentionImpl` (MiniMax-M3's sparse layers).
- `Attention` over MLA: its attention kernels, at the rank's heads, the
  heads the decode kernel pads them to, and the latent, rope, no-rope and
  value widths, the KV at the cache's format. Its up-projection linear
  (`kv_b_proj`) is priced as that projection, not as a GEMM: absorbed into
  two batched GEMMs on a decode step, over each request's context on a
  prefill one, gathered out of the cache where the prefill holds cached
  tokens, the cached ones alone past ATOM's `attn_prefill_chunk_size`.
  Sparse MLA, MLA whose KV is sharded over a decode
  context-parallel group, and its fp8 forms switched on by
  `ATOM_USE_FLYDSL_FP8_PREFILL_ATTN` and `ATOM_USE_TRITON_MXFP4_BMM`, are
  unclassified.
- `SiluAndMul`: the input width of the one row-parallel linear beside it.
- a torch or transformers activation module, the vision tower, and V4's
  rotary tables: nothing. ATOM's text stacks call no bare activation module
  (GDN holds one it never calls), the backend refuses a step with images, and
  V4 reads its tables inside the attention and compressor kernels.
- `PPMissingLayer`, which a pipeline stage holds where another stage runs the
  layer, and a `VocabParallelEmbedding` on any stage but the first, which a
  tied last stage holds only to load its LM head from: nothing.

DeepSeek-V4's modules, each priced and then gone into:

- `Block`: two hyper-connection mixes, one per sub-layer, each folding the
  previous sub-layer's output into the widened residual and reducing it to the
  next input; the block's two norms run inside them.
- `DeepseekV4Attention`: sparse attention over the sliding window and the
  compressed KV, with the indexer's scoring where the layer has one, at fp4
  where the `Indexer` runs its fp4 path, else fp8; under an fp8 KV cache its
  kernels take Q in fp8 as well. Its linears and the
  compressors' are priced as linears, `wo_a` at fp8 activations where it runs
  the fp8 batched GEMM.
- `Expert` (the shared expert): its activation, between its two linears.
- `FusedMoE`, not gone into: the routed experts, at the activation format
  its quant method runs them at, and the exchange around them: the
  data-parallel group's gather and reduce-scatter, MoRI's dispatch and
  combine under expert parallelism, at the formats MoRI sends, or the
  tensor-parallel all-reduce on one data-parallel rank unless its block
  leaves that to the next norm (`reduce_results`), as MiniMax-M3's always
  does. Shared experts the
  kernel runs with them (`num_fused_shared_experts`) take every row, or
  under expert parallelism on one data-parallel rank, where aiter's top-k
  table deals them, a rank's share. An
  expert-parallel MoE over a data-parallel group whose exchange is not MoRI's
  (`--all2all-backend rccl` or `none`, `ATOM_DISABLE_MORI_EP`), one over such
  a group that fuses its shared experts, the Mega kernel, and experts picked
  by phase, are unclassified.

DeepSeek-V4.1's, each priced and then gone into, the KV and index caches at
the formats ATOM's config states; its attention under a KV cache neither
bf16 nor fp4 is unclassified:

- a linear with native group-32 weights (`native_a8_group_rows`): its
  product quantizes its input to fp8, so it reads fp8 activations.
- `Block`: two hyper-connection mixes, as V4's; its two norms run inside
  them.
- `Attention`: its Q and KV norms in one launch, writing Q in fp8; sparse
  attention over the window and, where its layer compresses, the top k of
  the compressed KV, which its indexer scores where it has one, at most its
  candidates where an earlier layer chose them; and where it compresses,
  the index keys it writes. Its compressor's norm runs inside the pooling,
  its indexer's key projection and norm in those index keys.
- the rotary tables, the vision tower and the runtime's compiled stages,
  which hold the model outside the tree: nothing.

On a prefill that replays its decoder (`replay`), its layers after the first
decoder layer run on each finishing request's last window of rows, the last
layer on its last row, and the first decoder layer on that window but for its
compression.

`walk` also reads a block drafter's model, `drafter` builds it: DSpark's, as
V4.1 ships it. Its modules carry its block width (`Module.draft`), and the
pass that writes its KV over the target's rows is marked apart (`CONTEXT`):
the stages' KV projections, one fp8 product, and the first stage's
`main_proj` and `main_norm`. `DraftAttention` attends its block and the
window before it; the block's LM head, the target's, runs on its vocabulary
shard, and the Markov sampler on that shard, once a position.

A module of any other class is gone into when it has children. One without,
or a listed one in a format or layout not priced here, is unclassified: the
first of each class comes back as an `unclassified` module named by class
and path, which the backend refuses. Each module carries the index of the
decoder layer it sits in, read off its path.
"""

import dataclasses
import re

import torch

from atom.compass.backends.step_roofline import CONTEXT, UNCLASSIFIED, Module

#: Weight, KV and state formats a module is priced at, by dtype.
FORMATS = {
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "fp8",
    torch.float8_e4m3fnuz: "fp8",
    torch.float4_e2m1fn_x2: "fp4",
    "bf16": "bf16",
    "fp8": "fp8",
}
#: What MoRI's combine sends back, by `ATOM_MORI_COMBINE_QUANT`.
MORI_COMBINE = {
    "none": "bf16",
    "fp8_direct_cast": "fp8",
    "fp8_blockwise": "fp8",
    "fp4_blockwise": "fp4",
}
#: A module that prices nothing; one to go into.
NOTHING, DESCEND = object(), object()
OPS = "atom.model_ops."
V4 = "atom.models.deepseek_v4."
ROW = OPS + "linear.RowParallelLinear"
#: The impl a non-MLA `Attention` runs unless its model plugs in another.
DENSE_MHA = OPS + "attention_mha.PagedAttentionImpl"
FP8_MOE = OPS + "moe.Fp8MoEMethod"
NORMS = tuple(
    OPS + "layernorm." + n for n in ("RMSNorm", "GemmaRMSNorm", "RMSNormGated")
)
#: Modules that price nothing: activation modules ATOM's text stacks hold and
#: never call, and the vision tower.
SILENT = ("torch.nn.modules.activation", "transformers.activations")
VISION = "atom.models.qwen3_5_vl.Qwen3VisionTransformer"
#: A pipeline stage's stand-in for a layer another stage runs.
PP_MISSING = "atom.models.utils.PPMissingLayer"
M3 = "atom.models.minimax_m3.model."
#: V4's rotary tables, read inside the attention and compressor kernels.
V4_ROPE = V4 + "_V4RoPE"
V41 = "atom.models.deepseek_v41."
V41_BLOCK, DRAFT_BLOCK = V41 + "model.Block", V41 + "dspark.DraftBlock"
V41_ATTENTION, V41_INDEXER = V41 + "attention.Attention", V41 + "attention.Indexer"
DRAFT_ATTENTION = V41 + "dspark.DraftAttention"
COMPRESSOR = "atom.model_ops.deepseek_v41.compressor.Compressor"
#: The block drafter whose passes the step roofline lays out.
DSPARK = V41 + "dspark.DeepseekV41DSpark"
#: V4.1's rotary tables, read inside its attention and index kernels, its
#: vision tower, the runtime's compiled stages, and DSpark's Markov tables,
#: priced at the drafter's root.
V41_SILENT = (
    "atom.model_ops.deepseek_v41.rotary.RotaryEmbedding",
    V41 + "vision.ViT",
    V41 + "vision.Aligner",
    V41 + "runtime._Stage",
    "atom.models.dspark_draft.DSparkMarkovHead",
)
#: What the routed experts' kernel reads once for each expert it runs.
HELD = tuple(f"w{w}_{p}" for w in (13, 2) for p in ("weight", "weight_scale", "bias"))
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def walk(tree: torch.nn.Module, drafted=None) -> list[Module]:
    """The modules of `tree` the step roofline prices, in visiting order, then
    those of `drafted`, the model its block drafter runs (`drafter`)."""
    found: list[Module] = []
    seen: set[type] = set()
    replay = _replay(tree)

    def visit(module, name, parent, draft):
        m = _classify(module, name, parent)
        *priced, then = m if isinstance(m, tuple) else (m,)
        if isinstance(then, Module):
            priced, then = [then], NOTHING
        layer = LAYER.search(name)
        layer = int(layer.group(1)) if layer else -1
        for p in priced:
            p = dataclasses.replace(p, layer=layer, draft=p.draft or draft)
            rows = _replayed(replay, layer, name, p.kind) if not p.draft else 0
            if p.draft == CONTEXT and replay:
                rows = replay[2]
            found.append(dataclasses.replace(p, replay=rows))
        if then is DESCEND:
            for child, sub in module.named_children():
                inner = CONTEXT if _drafts_kv(sub, module) else draft
                visit(sub, f"{name}.{child}" if name else child, module, inner)
        elif then is None and type(module) not in seen:
            seen.add(type(module))
            where = f"{type(module).__name__} at {name or 'the root'}"
            found.append(Module(where, UNCLASSIFIED))

    visit(tree, "", None, 0)
    if drafted is not None:
        visit(drafted, "drafter", None, drafted.block_size)
    return found


def drafter(config):
    """The model `config`'s drafter runs, built on the meta device as a
    simulated run builds its target, its post-load hooks run; None without a
    drafter, or for one whose passes the step roofline does not lay out: any
    but DeepSeek-V4.1's DSpark, whose weights ship with the target's.

    It registers its layers in a copy of ATOM's forward context: ATOM's own
    drafter build registers the same names, and refuses one twice."""
    spec = getattr(config, "speculative_config", None)
    if spec is None or spec.use_dspark_with_draft():
        return None
    from atom.spec_decode.drafter import support_draft_model_arch_dict

    arch = spec.draft_model_hf_config.architectures[0]
    if support_draft_model_arch_dict.get(arch) != DSPARK:
        return None
    import copy

    from atom.compass.runner.overrides import (
        FactoriesOnMeta,
        process_weights_after_loading,
        simulated_arch,
    )
    from atom.config import use_custom_atom_config
    from atom.utils import resolve_obj_by_qualname

    own = copy.copy(config)
    own.compilation_config = copy.copy(config.compilation_config)
    own.compilation_config.static_forward_context = {}
    with use_custom_atom_config(own), FactoriesOnMeta(), simulated_arch():
        model = resolve_obj_by_qualname(DSPARK)(own)
        process_weights_after_loading(model)
    return model


def _replay(tree):
    """The first decoder layer, the last layer and the window of a V4.1
    `tree` that replays its decoder on a prefill's last rows, else None."""
    if not (_is(tree, V41 + "runtime.DeepseekV41RuntimeModel") and tree.replay):
        return None
    return tree.decoder_start, tree.last_layer, tree.config.sliding_window


def _replayed(replay, layer: int, name: str, kind: str) -> int:
    """`Module.replay` for a module of `layer` named `name`: the decoder's
    window, the last layer's last row, and every row before the decoder and
    for the first decoder layer's compression."""
    if replay is None:
        return 0
    start, last, window = replay
    if layer == last:
        return 1
    compresses = ".compressor" in name or kind == "index_keys"
    return window if layer > start or layer == start and not compresses else 0


def _drafts_kv(module, parent) -> bool:
    """Whether `module` runs in its drafter's pass over the target's rows."""
    held = ("main_proj", "main_norm") if _is(parent, DRAFT_BLOCK) else ()
    return any(module is getattr(parent, h, None) for h in held)


def _inside(module, parent) -> bool:
    """Whether `module` runs inside a kernel its V4.1 parent is priced as: a
    block's norms in its mixes, an attention's Q and KV norms in their one
    launch, a compressor's norm in its pooling, an indexer's key projection
    and norm in its index keys."""
    held = ()
    if _is(parent, V41_BLOCK):
        held = ("attn_norm", "ffn_norm")
    elif _is(parent, V41_ATTENTION):
        held = ("q_norm", "kv_norm")
    elif _is(parent, COMPRESSOR):
        held = ("norm",)
    elif _is(parent, V41_INDEXER):
        held = ("wk", "k_norm")
    return any(module is getattr(parent, h, None) for h in held)


def _m3_reduces(module, name, parent) -> bool:
    """Whether `module` is a MiniMax-M3 norm that all-reduces the sub-layer
    output before it: a decoder layer's two, but the first layer's input
    norm, and the model's final norm."""
    if _is(parent, M3 + "MiniMaxM3Model"):
        return module is parent.norm
    if not _is(parent, M3 + "MiniMaxM3DecoderLayer"):
        return False
    if module is parent.input_layernorm:
        return LAYER.search(name).group(1) != "0"
    return module is parent.post_attention_layernorm


def _tp_width() -> int:
    """The TP group's width."""
    from aiter.dist.parallel_state import get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_world_size()


def _first_stage() -> bool:
    """Whether this rank is its pipeline's first stage, the one that embeds."""
    from aiter.dist.parallel_state import get_pp_group

    return get_pp_group().is_first_rank


def _is(module, *names: str) -> bool:
    """Whether `module`'s class is, or derives from, a class of `names`."""
    return any(
        f"{c.__module__}.{c.__qualname__}" in names for c in type(module).__mro__
    )


def _classify(module, name, parent):
    """`module`'s `Module`, `NOTHING`, `DESCEND`, or None if unclassified."""
    if parent is not None and _inside(module, parent):
        return NOTHING
    weight = getattr(module, "weight", None)
    fmt = FORMATS.get(getattr(weight, "dtype", None))
    if _is(module, V4 + "Block"):
        mix = Module(name, "mhc", (module.hc_mult, module.attn_norm.weight.shape[0]))
        return mix, mix, DESCEND
    if _is(module, V41_BLOCK):
        width = module.attn_norm.weight.shape[0]
        mix = Module(name, "mhc", (module.hc_attn_fn.shape[1] // width, width))
        return mix, mix, DESCEND
    if _is(module, V41_ATTENTION):
        found = _csa2(module, name)
        return found and (*found, DESCEND)
    if _is(module, DSPARK):
        return (*_dspark(module, name), DESCEND)
    if _is(module, V4 + "DeepseekV4Attention"):
        ix = module.indexer
        dims = (module.n_local_heads, module.head_dim, module.window_size)
        dims += (module.compress_ratio,)
        dims += (ix.index_topk, ix.n_heads, ix.head_dim) if ix else (0, 0, 0)
        # ponytail: the KV is read at fp8, its narrowest format; the window
        # rows held in bf16 are left out of this lower bound.
        q = "fp8" if module.kv_fp8 else "bf16"
        index = "fp4" if ix and ix._indexer_fp4 else "fp8"
        attn = Module(name, "sparse_attention", dims, "fp8", act=q, index=index)
        return attn, DESCEND
    if _is(module, V4 + "Expert"):
        return Module(name, "act", (_in_width(module.w2),)), DESCEND
    if _is(module, OPS + "moe.FusedMoE"):
        return _moe(module, name, parent)
    if _is(module, OPS + "embed_head.ParallelLMHead"):
        return fmt and Module(name, "lm_head", tuple(weight.shape), fmt, module.tp_size)
    if _is(module, OPS + "embed_head.VocabParallelEmbedding"):
        if not _first_stage():
            return NOTHING
        return Module(name, "embedding", (weight.shape[1],), ranks=module.tp_size)
    if _is(module, OPS + "linear.LinearBase", "torch.nn.modules.linear.Linear"):
        if not fmt:
            return None
        if weight.dim() != 2:
            return Module(name, "conv", (weight.shape[0],), fmt)
        reduces = _is(module, ROW) and module.reduce_results
        ranks = module.tp_size if reduces else 1
        quant = getattr(getattr(module, "quant_type", None), "name", "No")
        # A quantized linear quantizes its input to its weight's format.
        act = "bf16" if quant == "No" else fmt
        if getattr(module, "native_a8_group_rows", None) is not None:
            act = "fp8"
        if getattr(parent, "_wo_a_mxscale", False) and module is parent.wo_a:
            # V4 quantizes the input to the weight's fp8 for its batched GEMM.
            act = FORMATS.get(parent._wo_a_fp8_dtype)
        if attention := _mla_up_of(module, parent):
            from atom.config import get_current_atom_config
            from atom.utils import envs

            if envs.ATOM_USE_TRITON_MXFP4_BMM:
                return None
            mla, kv = attention.impl, FORMATS.get(attention.kv_cache_dtype)
            dims = (mla.num_heads, mla.qk_nope_head_dim, mla.v_head_dim)
            dims += (_in_width(module),)
            chunk = get_current_atom_config().attn_prefill_chunk_size
            return kv and Module(name, "mla_up", dims, fmt, act=act, kv=kv, chunk=chunk)
        dims = (weight.shape[0], _in_width(module))
        return Module(name, "gemm", dims, fmt, ranks, act=act)
    if _is(module, *NORMS):
        if parent is not None and _is(parent, V4 + "Block"):
            return NOTHING
        tp = module.tp_size if getattr(module, "fused_allreduce", False) else 1
        tp = _tp_width() if _m3_reduces(module, name, parent) else tp
        return Module(name, "norm", (weight.shape[0],), ranks=tp, fused=tp > 1)
    if _is(module, "aiter.rotary_embedding.RotaryEmbedding"):
        return Module(name, "rotary", (module.rotary_dim,))
    if _is(module, OPS + "paged_attention.Attention"):
        kv = FORMATS.get(module.kv_cache_dtype)
        if module.use_mla:
            return kv and _mla(module.impl, name, kv)
        # An impl a model plugs in, even one derived from the dense one (as
        # MiniMax-M3's sparse attention is), runs a kernel of its own.
        impl = type(module.impl)
        if f"{impl.__module__}.{impl.__qualname__}" != DENSE_MHA:
            return None
        if getattr(module.impl, "sliding_window", -1) > 0:
            return None
        dims = (module.num_heads, module.num_kv_heads, module.head_dim)
        return kv and Module(name, "attention", dims, kv) or None
    if _is(module, OPS + "base_attention.LinearAttention"):
        tp, state = module.impl.tp_size, FORMATS.get(module.A_log.dtype)
        dims = (module.num_v_heads // tp, module.num_k_heads // tp)
        dims += (module.head_k_dim, module.head_v_dim)
        return state and Module(name, "linear_attention", dims, state)
    if _is(module, OPS + "activation.SiluAndMul"):
        down = [c for c in parent.children() if _is(c, ROW)]
        return len(down) == 1 and Module(name, "act", (_in_width(down[0]),)) or None
    if _is(module, VISION, V4_ROPE, PP_MISSING, *V41_SILENT):
        return NOTHING
    if type(module).__module__.startswith(SILENT):
        return NOTHING
    return DESCEND if next(module.children(), None) is not None else None


def _in_width(linear) -> int:
    """`linear`'s input width: an fp4 weight packs two of it to a byte."""
    weight = linear.weight
    return weight.shape[1] * (2 if weight.dtype == torch.float4_e2m1fn_x2 else 1)


def _csa2(module, name):
    """V4.1's attention: its Q and KV norms' one launch; a draft's block
    attention, or the sparse attention of its layer's mode; and where it
    compresses, the index keys it writes. The caches are at the formats
    ATOM's config states: the KV pool bf16 or fp4, a draft's window bf16 or
    beside an fp4 pool fp8, the index plane fp8 or fp4. None under a KV
    cache of another format."""
    from atom.config import get_current_atom_config

    config = get_current_atom_config()
    if config.kv_cache_dtype not in ("bf16", "fp4"):
        return None
    hf, spec, ix = config.hf_config, module.spec, module.indexer
    q, kv = (n.weight.shape[0] for n in (module.q_norm, module.kv_norm))
    found = [Module(f"{name}.qk_norm", "norm", (q + kv, q))]
    dims = (module.heads, module.head_dim, hf.sliding_window)
    fp4 = config.kv_cache_dtype == "fp4"
    if _is(module, DRAFT_ATTENTION):
        return found + [Module(name, "block_attention", dims, "fp8" if fp4 else "bf16")]
    index = "fp4" if config.index_cache_dtype == "fp4" else "fp8"
    dims += (spec.ratio, hf.index_topk if spec.ratio else 0)
    dims += (ix.heads, ix.head_dim) if ix else (0, 0)
    if ix and spec.candidate_source is not None:
        dims += (ix.topk_blocks * ix.block_size,)
    pool = "fp4" if fp4 else "bf16"
    found.append(Module(name, "sparse_attention", dims, pool, index=index))
    if module.compressor is not None:
        keys = (ix.head_dim, module.head_dim, spec.ratio)
        found.append(Module(f"{name}.indexer", "index_keys", keys, index))
    return found


def _dspark(module, name):
    """DSpark's work outside its blocks: the pass over the target's rows that
    writes each stage's KV, one fp8 product of the stages' KV projections;
    the block's LM head, the target's, on this rank's vocabulary shard; and
    the Markov sampler over that shard."""
    stages = [block.attn for block in module.mtp]
    tp, hidden = stages[0].wo_b.tp_size, module.config.hidden_size
    vocab = -(-module.vocab_size // tp)
    kv = (sum(a.head_dim for a in stages), hidden)
    markov = (vocab, module.mtp[-1].markov_head.rank, module.block_size)
    return (
        Module(f"{name}.context_kv", "gemm", kv, "fp8", act="fp8", draft=CONTEXT),
        Module(f"{name}.head", "gemm", (vocab, hidden)),
        Module(f"{name}.markov", "markov", markov, ranks=tp),
    )


def _mla(impl, name, kv):
    """MLA attention's `Module` from its `MLAAttention` impl, or None where it
    runs sparse, its KV sharded over a decode context-parallel group, or its
    prefill attention in fp8."""
    if impl.is_sparse_mla or impl.dcp_world_size > 1:
        return None
    if impl.use_flydsl_fp8_prefill_attn:
        return None
    dims = (impl.num_heads, impl.padded_num_heads, impl.kv_lora_rank)
    dims += (impl.qk_rope_head_dim, impl.qk_nope_head_dim, impl.v_head_dim)
    return Module(name, "mla_attention", dims, kv)


def _mla_up_of(linear, parent):
    """The MLA `Attention` beside `linear` that up-projects its latent KV
    through it, or None."""
    for sibling in parent.children() if parent is not None else ():
        if getattr(getattr(sibling, "mla_modules", None), "kv_b_proj", None) is linear:
            return sibling
    return None


def _moe(module, name, parent):
    """The routed experts and their exchange, or None if unclassified.

    `dims` are (experts, local experts, top k, intermediate per rank, hidden,
    group ranks, data-parallel ranks, MoRI's split send and receive, the bytes
    of one local expert's weights, scales and biases), then, where the kernel
    runs shared experts beside the routed ones, how many; the local experts
    count the routed ones only. The data-parallel
    width is the MoE group's over the attention's TP width: the parent MoE
    block's, or the TP group's where the block holds none (MiniMax-M3's,
    Mixtral's). Under expert parallelism over that group, `wire` is what
    MoRI's dispatch sends, as ATOM resolves it, and what its combine sends
    back.

    The products read activations at the format of the kernel the method runs.
    Its Triton experts under expert parallelism, and its fused SiLU ones
    without, run fp8 or, under `ATOM_USE_TRITON_MOE_A4W4`, fp4 ones; its other
    Triton experts run `act_quant`'s. aiter's run fp8 ones when the method
    asks for them or is the FP8 method, and otherwise `act_quant`'s, or the
    weights' format when it has a quant type: the kernel quantizes them to it.
    The unquantized method's experts read its weights' format.
    A method that picks its experts by phase (`ATOM_USE_TRITON_MOE_DECODE`) is
    unclassified. Under expert parallelism with no all-to-all, on one
    data-parallel rank, aiter's top-k table deals each row's fused shared
    experts to rank `row mod ranks`; fused shared experts under expert
    parallelism in any other layout are unclassified.
    """
    from atom.model_ops.fused_moe.expert_layout import SharedExpertMode
    from atom.utils import envs

    method, w13 = module.quant_method, module.w13_weight
    act = getattr(getattr(method, "act_quant", None), "value", None)
    fmt = FORMATS.get(w13.dtype)
    attention_tp = getattr(parent, "tp_size", None) or _tp_width()
    inter = w13.shape[1] // 2
    tp, ep = (getattr(method, f, False) for f in ("use_triton", "use_triton_ep"))
    if getattr(method, "use_triton_decode", False):
        return None
    if tp or ep:
        # Its Triton prep re-lays the weights as bytes, in a layout of its own.
        fmt, inter = FORMATS.get(method.quant_dtype), method.intermediate_size
    if ep:
        act = "fp4" if envs.ATOM_USE_TRITON_MOE_A4W4 else "fp8"
    elif tp:
        a4w4 = envs.ATOM_USE_TRITON_MOE_A4W4 or act == "fp4"
        act = ("fp4" if a4w4 else "fp8") if module.use_fused_silu_gugu else act
    elif getattr(method, "fp8_activations", False) or _is(method, FP8_MOE):
        act = "fp8"
    elif getattr(getattr(method, "quant_type", None), "value", 0):
        act = act and fmt
    elif _is(method, OPS + "moe.UnquantizedFusedMoEMethod"):
        act = fmt
    held = sum(t.nbytes for p in HELD if (t := getattr(module, p, None)) is not None)
    mega = _is(method, OPS + "moe.MegaMxfp4MoEMethod")
    if mega or not (act and fmt and attention_tp):
        return None
    config = module.moe_parallel_config
    shared = getattr(module, "num_fused_shared_experts", 0)
    ranks = config.ep_size if config.use_ep else config.tp_size
    dp = ranks // attention_tp
    legacy = SharedExpertMode.LEGACY_AITER
    if shared and config.use_ep and (dp > 1 or module.expert_layout.mode != legacy):
        return None
    split = int(config.use_ep and config.low_latency)
    dims = (module.global_num_experts, w13.shape[0] - shared, module.top_k)
    dims += (inter, module.hidden_size, ranks, dp, split)
    dims += (held // w13.shape[0],) + ((shared,) if shared else ())
    fused = module._comm_fused_moe is not None
    if not (config.use_ep and dims[6] > 1):
        # A block that does not reduce its output leaves it to the next
        # norm's fused all-reduce; MiniMax-M3's always does.
        leaves = _is(parent, M3 + "MiniMaxM3MoE")
        reduces = dims[6] > 1 or getattr(parent, "reduce_results", not leaves)
        return Module(name, "moe", dims, fmt, ranks if reduces else 1, fused, act=act)
    from atom.model_ops.fused_moe.routed_rows import resolve_routed_rows_wire

    mori = config.requested_all2all_backend in ("auto", "mori") and not (
        envs.ATOM_DISABLE_MORI_EP or envs.ATOM_MORI_V2
    )
    # The method's routed-rows wire, as `_maybe_make_prepare_finalize` takes it.
    sent = getattr(method, "routed_rows_wire", None) or resolve_routed_rows_wire(
        module.moe_config.in_dtype, module.hidden_size
    )
    wire = (FORMATS.get(sent.dtype), MORI_COMBINE.get(envs.ATOM_MORI_COMBINE_QUANT))
    if not (mori and all(wire)):
        return None
    return Module(name, "moe", dims, fmt, ranks, fused, act=act, wire=wire)
