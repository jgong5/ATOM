"""Ambient state an operator reads that its arguments do not describe.

Most operators are functions of their arguments, so a recorded ``(name, shapes,
dtypes, scalars)`` is enough to call one again and find out what it costs.
Attention is not. It reads its metadata -- which blocks of KV cache to walk, how
long each sequence is -- from a module-global forward context, and its arguments
say nothing about any of it.

The obvious repair was to give the operator those arguments and let it stand up
its own context when there is no live forward. That was tried, in
``c337ee3a``, and it does not work: ``torch.compile`` traces
``md = get_forward_context().attn_metadata`` and only the *tensor* reads survive
as graph inputs. ``md.max_seqlen_q`` is an int, so it is constant-folded, and
``md.block_tables`` was ``None`` in the forward that compiled -- the warmup
dummy -- so it is baked in as the constant ``None``. The recorded call then
claims ``block_tables=None, max_seqlen_q=16384`` where the live step has a
``(4, 2560)`` tensor and ``1``. Production never notices, because it ignores
those arguments and reads the context; a benchmark that honours them attends the
wrong thing or crashes.

So the context is recorded alongside the operator instead of through it. The
tracer reads it from the live forward, where it is true, and the benchmark
installs it before calling. The operator is untouched and stays exactly what
production runs.

The cost of this design is that Compass has to know something about attention
specifically, which the rest of the op graph avoids. It is confined to this
module, and to operators that genuinely read ambient state -- currently one.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["capture", "install", "is_context_dependent"]

#: Operators whose cost depends on state their arguments do not carry.
_ATTENTION = "aiter::unified_attention_with_output_base"


def _values(tensor) -> list[int] | None:
    if tensor is None:
        return None
    return [int(x) for x in tensor.flatten().tolist()]


def _capture_attention() -> tuple[tuple[str, Any], ...]:
    """Attention's metadata, as the live forward has it.

    ``block_tables`` is recorded by its full shape but only the columns the
    kernel reads. The full width is ``max_model_len / block_size`` -- 2560 here
    -- and is part of the call because it is the row stride the kernel indexes
    with, but a decode at 315 tokens of context touches 20 columns of it. Keeping
    the used prefix and the shape reproduces both the addressing and the traffic
    without an artifact that grows with the model's maximum context.
    """
    from atom.config import get_current_atom_config
    from atom.utils.forward_context import get_forward_context

    fwd = get_forward_context()
    md = getattr(fwd, "attn_metadata", None)
    if md is None or fwd.context is None or md.context_lens is None:
        return ()

    recorded: list[tuple[str, Any]] = [
        ("context_lens", _values(md.context_lens)),
        ("slot_mapping", _values(md.slot_mapping)),
        ("cu_seqlens_q", _values(md.cu_seqlens_q)),
        ("max_seqlen_q", int(md.max_seqlen_q)),
        ("max_seqlen_k", int(md.max_seqlen_k)),
        ("state", md.state.value),
        ("is_prefill", bool(fwd.context.is_prefill)),
        ("positions", _values(fwd.context.positions)),
    ]

    table = md.block_tables
    if table is not None and table.dim() == 2:
        block_size = int(get_current_atom_config().kv_cache_block_size)
        longest = max(_values(md.context_lens) or [0])
        used = min(table.shape[1], -(-longest // max(block_size, 1)) or 1)
        recorded += [
            ("block_tables_shape", [int(d) for d in table.shape]),
            ("block_tables", _values(table[:, :used])),
        ]
    return tuple(recorded)


def _install_attention(recorded: dict[str, Any]) -> None:
    import torch

    from atom.config import get_current_atom_config
    from atom.utils.forward_context import (
        AttentionMetaData,
        AttnState,
        Context,
        set_forward_context,
    )

    def tensor(name, dtype):
        values = recorded.get(name)
        if values is None:
            return None
        return torch.tensor(values, dtype=dtype, device="cuda")

    table = None
    shape = recorded.get("block_tables_shape")
    if shape is not None:
        table = torch.zeros(tuple(shape), dtype=torch.int32, device="cuda")
        flat = recorded.get("block_tables") or []
        used = len(flat) // max(shape[0], 1)
        if used:
            table[:, :used] = torch.tensor(
                flat, dtype=torch.int32, device="cuda").reshape(shape[0], used)

    metadata = AttentionMetaData(
        block_tables=table,
        context_lens=tensor("context_lens", torch.int32),
        slot_mapping=tensor("slot_mapping", torch.int64),
        cu_seqlens_q=tensor("cu_seqlens_q", torch.int32),
        max_seqlen_q=int(recorded.get("max_seqlen_q", 0)),
        max_seqlen_k=int(recorded.get("max_seqlen_k", 0)),
        state=AttnState(recorded.get("state", AttnState.DECODE.value)),
    )
    set_forward_context(
        attn_metadata=metadata,
        atom_config=get_current_atom_config(),
        context=Context(
            positions=tensor("positions", torch.int64),
            is_prefill=bool(recorded.get("is_prefill", False)),
        ),
    )


_CAPTURE = {_ATTENTION: _capture_attention}
_INSTALL = {_ATTENTION: _install_attention}


def is_context_dependent(name: str) -> bool:
    """Whether ``name`` cannot be called from its arguments alone."""
    return name in _INSTALL


def capture(name: str) -> tuple[tuple[str, Any], ...]:
    """What the live forward context holds for ``name``, or ``()``.

    Each read is a device-to-host copy, so this belongs to trace mode, which
    already runs eagerly and exists to produce an artifact rather than to serve.
    """
    recorder = _CAPTURE.get(name)
    if recorder is None:
        return ()
    try:
        return recorder()
    except Exception as exc:  # noqa: BLE001 - never fail a trace over this
        logger.warning("ATOMCompass WARNING: could not record the forward "
                       "context for %s (%s); it will be unpriceable", name, exc)
        return ()


def install(name: str, recorded) -> bool:
    """Stand up the forward context ``name`` was recorded with.

    Returns whether it could, so a caller can report the operator unpriced
    rather than price it against whatever context happened to be installed --
    which is the failure this whole module exists to prevent.
    """
    installer = _INSTALL.get(name)
    if installer is None or not recorded:
        return False
    installer({k: v for k, v in (tuple(x) for x in recorded)})
    return True
