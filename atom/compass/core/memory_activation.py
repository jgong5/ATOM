"""The warmup prefill activation peak, derived from the allocation sites.

The activation term is the only one in the memory budget that is *walked*
rather than looked up, and on this model the walk was wrong by 5x. This module
is why. It does not walk anything: it reads the model's own geometry out of the
config and adds up the tensors the code allocates, naming the line that
allocates each one.

Two facts make that possible here and would not make it possible everywhere.

**The forward runs under `torch.inference_mode`** (`model_runner.forward`), so
nothing is saved for a backward pass and every intermediate is dead the instant
its last Python reference goes. Liveness is dataflow and scope, not autograd.

**The peak is inside one operator.** On a GDN hybrid the linear-attention layer
hands its whole body to `aiter::linear_attention_with_output_base`, a custom op
whose internals a dispatch trace cannot see -- it records the op and its output
and nothing in between. Inside it, `chunk_gated_delta_rule` allocates a
per-chunk state `h` of `[1, T/64, Hv, 128, 128]` and half a dozen tensors of
`[1, T, Hv, *]` beside it, and they are all live at once because they are all
still bound to locals in `chunk_gated_delta_rule_fwd` when `h` is allocated.
That is the 2.4 GB the walk missed at T=16 384, and no amount of tracing at the
torch level will ever find it.

The point of writing it down this way is the **width**. Every one of those
tensors is indexed by a head count, and head counts divide by the
tensor-parallel width. The two that do not -- the residual stream and the
normalised hidden state -- are full `hidden_size` and do not divide at all. So
the peak is `replicated + sharded / tp` with a mechanism behind each side,
rather than a curve fitted through measurements at two widths.

What this module does **not** do is fit anything to a measurement at the target
width. The TP=2 and TP=4 peaks exist and are evaluation-only: they are read
after a prediction is frozen, never before, and never as an input.
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

__all__ = [
    "ACTIVATION_SCHEMA",
    "CHUNK_SIZE",
    "PeakTerm",
    "WarmupPeak",
    "gdn_prefill_terms",
    "warmup_prefill_peak",
]

#: Bump on any change to the term set, the way an id schema is bumped: a
#: prediction carries this, and two numbers derived under different term sets
#: are not comparable.
ACTIVATION_SCHEMA = "compass.activation.gdn_prefill/1"

#: `chunk_gated_delta_rule_fwd` passes `chunk_size=64` to every kernel it
#: calls, and the HIP fast path is entered at `T >= 64`. The number of chunks
#: is what sizes `h`, so this is not a tuning knob here -- it is geometry.
CHUNK_SIZE = 64

_BF16 = 2
_FP32 = 4


@dataclass(frozen=True)
class PeakTerm:
    """One tensor live at the peak, and whether it divides by the width."""

    name: str
    bytes_at_one: int
    shards: bool
    where: str
    why: str

    def at(self, tensor_parallel: int) -> int:
        tp = max(1, int(tensor_parallel))
        return self.bytes_at_one // tp if self.shards else self.bytes_at_one


@dataclass(frozen=True)
class WarmupPeak:
    """A derived peak, with the two halves kept apart on purpose.

    `residue` is whatever a source measurement at TP=1 saw that the terms below
    do not account for. It is carried separately because its *mechanism* is
    unknown, and a term whose mechanism is unknown cannot be told to shard.
    """

    schema: str
    tokens: int
    tensor_parallel: int
    terms: Sequence[PeakTerm]
    residue: int = 0
    residue_shards: bool = False

    @property
    def replicated(self) -> int:
        return sum(t.bytes_at_one for t in self.terms if not t.shards)

    @property
    def sharded_at_one(self) -> int:
        return sum(t.bytes_at_one for t in self.terms if t.shards)

    @property
    def derived(self) -> int:
        """The enumerated terms at this width, before any residue."""
        return sum(t.at(self.tensor_parallel) for t in self.terms)

    @property
    def residue_at(self) -> int:
        tp = max(1, int(self.tensor_parallel))
        return self.residue // tp if self.residue_shards else self.residue

    @property
    def total(self) -> int:
        return self.derived + self.residue_at


def _text(config: Mapping) -> Mapping:
    return config.get("text_config") or config


def gdn_prefill_terms(config: Mapping, *, tokens: int) -> list:
    """Every tensor live when `chunk_gated_delta_rule` allocates `h`.

    One linear-attention layer at the warmup shape, at TP=1. The layer is the
    peak of the step: the full-attention layers hold a fraction of this, and
    the MLP's `[T, 2 x intermediate / tp]` is smaller than what the gated delta
    rule has live at the same moment.

    The order is the order the code allocates them, so the list reads as the
    forward does.
    """
    text = _text(config)
    hidden = int(text["hidden_size"])
    k_heads = int(text["linear_num_key_heads"])
    v_heads = int(text["linear_num_value_heads"])
    k_dim = int(text["linear_key_head_dim"])
    v_dim = int(text["linear_value_head_dim"])
    tokens = int(tokens)

    key_dim = k_heads * k_dim
    value_dim = v_heads * v_dim
    conv_dim = 2 * key_dim + value_dim
    # `[q | k | v | z | b | a]`, the merged in-projection's output widths.
    qkvzba = 2 * key_dim + 2 * value_dim + 2 * v_heads
    chunks = -(-tokens // CHUNK_SIZE)

    return [
        PeakTerm(
            "residual", tokens * hidden * _BF16, False,
            "models/qwen3_next.py Qwen3NextDecoderLayer.forward",
            "the residual stream, carried across every layer at full width. "
            "There is no sequence parallelism here, so neither the token count "
            "nor the hidden size divides by the width."),
        PeakTerm(
            "hidden_states", tokens * hidden * _BF16, False,
            "models/qwen3_next.py Qwen3NextDecoderLayer.forward",
            "the normalised input to the attention, live beside the residual "
            "for the whole of the attention call. Replicated for the same "
            "reason."),
        PeakTerm(
            "in_proj_qkvzba", tokens * qkvzba * _BF16, True,
            "models/qwen3_5.py Qwen3_5GatedDeltaNet.forward",
            "one allocation, not six: `torch.split` returns views, so the "
            "whole projection stays alive as long as any of q, k, v, z, b or a "
            "is still needed -- and `z` is needed after the core attention "
            "returns. A column-parallel projection, so every output width is "
            "already per-rank."),
        PeakTerm(
            "core_attn_out", tokens * value_dim * _BF16, True,
            "models/qwen3_5.py Qwen3_5GatedDeltaNet.forward",
            "`torch.empty_like(z)`, allocated before the op and written by it."),
        PeakTerm(
            "conv_qkv", tokens * conv_dim * _BF16, True,
            "model_ops/mamba_ops/causal_conv1d.py causal_conv1d_fn",
            "three fresh tensors of `k_dim_size`, `k_dim_size` and "
            "`v_dim_size`, and the call site passes those as "
            "`num_heads * head_dim // tp_size`."),
        PeakTerm(
            "gating_g", tokens * v_heads * _FP32, True,
            "model_ops/attention_gdn.py fused_gdn_gating",
            "fp32 by construction -- `A_log.float().exp()` would be -inf in "
            "bf16."),
        PeakTerm(
            "gating_beta", tokens * v_heads * _BF16, True,
            "model_ops/attention_gdn.py fused_gdn_gating", "b.sigmoid()."),
        PeakTerm(
            "g_cumsum", tokens * v_heads * _FP32, True,
            "model_ops/fla_ops/fused_cumsum_kkt.py",
            "the running log-gate, fp32, a second tensor rather than an "
            "in-place update of the gate above."),
        PeakTerm(
            "A", tokens * v_heads * CHUNK_SIZE * _FP32, True,
            "model_ops/fla_ops/fused_cumsum_kkt.py",
            "`[B, T, H, 64]` fp32 -- one column per position in the chunk. It "
            "stays live to the end of the enclosing function because the local "
            "still refers to it, long after the kernel that read it."),
        PeakTerm(
            "Ai16", tokens * v_heads * 16 * _FP32, True,
            "model_ops/fla_ops/chunk.py chunk_gated_delta_rule_fwd",
            "the 16x16 diagonal inverse, fp32, same scope and same lifetime."),
        PeakTerm(
            "w", tokens * v_heads * k_dim * _BF16, True,
            "model_ops/fla_ops/fused_merge_recompute.py",
            "the WY representation."),
        PeakTerm(
            "u", tokens * v_heads * v_dim * _BF16, True,
            "model_ops/fla_ops/fused_merge_recompute.py",
            "`torch.empty_like(v)` -- the recomputed values."),
        PeakTerm(
            "h", chunks * v_heads * k_dim * v_dim * _BF16, True,
            "model_ops/fla_ops/chunk_delta_h.py chunk_gated_delta_rule_fwd_h",
            "`k.new_empty(B, NT, H, K, V)`: the recurrent state at every chunk "
            "boundary. This is the single largest transient in the step and "
            "the one a dispatch trace has no way to see. It grows with tokens "
            "like everything else -- one chunk per 64 -- and divides by the "
            "width like everything else, because NT is chunks and H is heads."),
        PeakTerm(
            "v_new", tokens * v_heads * v_dim * _BF16, True,
            "model_ops/fla_ops/chunk_delta_h.py chunk_gated_delta_rule_fwd_h",
            "`torch.empty_like(u)`, allocated in the same call as `h`."),
        PeakTerm(
            "recurrent_states", 2 * v_heads * k_dim * v_dim * _FP32, True,
            "model_ops/attention_gdn.py GatedDeltaNet.forward",
            "the initial state gathered out of the SSM cache and the final "
            "state written back. Independent of the token count, which is why "
            "they are the only terms here that do not grow with the shape."),
    ]


def warmup_prefill_peak(config: Mapping, *, tokens: int,
                        tensor_parallel: int = 1, residue: int = 0,
                        residue_shards: bool = False) -> WarmupPeak:
    """The activation peak of the warmup prefill, at a width nobody has run.

    `tokens` is the warmup shape: `warmup_model` schedules one sequence of
    `max_num_batched_tokens` with no history, so this is that number and the
    history is zero. A different shape is a different peak and this is not the
    function to ask about it.

    `residue` is a measured TP=1 peak's excess over the enumerated terms, and
    `residue_shards` is the claim about it. The default is `False`, which
    over-predicts the peak at TP>1 and therefore under-sizes the KV pool --
    the safe direction, and the honest one while its mechanism is unknown.
    """
    return WarmupPeak(
        schema=ACTIVATION_SCHEMA, tokens=int(tokens),
        tensor_parallel=max(1, int(tensor_parallel)),
        terms=gdn_prefill_terms(config, tokens=tokens),
        residue=int(residue), residue_shards=bool(residue_shards))
