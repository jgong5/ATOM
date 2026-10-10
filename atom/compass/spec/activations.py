# SPDX-License-Identifier: MIT
"""Activation bytes per token, measured on a real run per model and width.

An entry is keyed the way a tokenizer entry is: `applies_to` lists the model
architectures it answers for, and `fingerprint` is the sha256 of the
`config.json` that was run. The fingerprint is part of the key rather than a
check beside it, because one architecture class serves every size of a family
and a coefficient measured on one size is wrong for the next.

`bytes_per_token` holds, per tensor-parallel width, one real run's
`peak - current` allocated bytes after warmup divided by the warmup tokens. That
is the reading ATOM's `ModelRunner._estimate_cudagraph_overhead` takes, so the
kernel scratch the allocator saw during the warmup forward is inside it.

A model, a fingerprint or a width with no entry resolves to None, and the memory
model falls back to its declared geometry form, labelled declared.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .fields import Field, Kind, check
from .rules import Rule, SpecRefusal

ENTRY_FIELDS = (
    Field("id", Kind.TEXT),
    Field("fingerprint", Kind.TEXT),
    Field("applies_to", Kind.NAMES),
    Field("bytes_per_token", Kind.WIDTH_TABLE),
)


@dataclass(frozen=True, slots=True)
class ActivationEntry:
    """One measured model: its identity and its bytes per token at each width."""

    id: str
    fingerprint: str
    applies_to: tuple[str, ...]
    bytes_per_token: Mapping[int, float]


def _entry(raw: object, index: int) -> ActivationEntry:
    where = f"device.activations[{index}]"
    declared = {field.path: field for field in ENTRY_FIELDS}
    if not isinstance(raw, Mapping) or set(raw) != set(declared):
        raise SpecRefusal(
            Rule.SHAPE,
            f"`{where}` holds {raw!r}, which is not an activation entry",
            f"write a mapping of exactly {', '.join(declared)} there",
        )
    return ActivationEntry(
        **{
            name: check(field, raw[name], f"{where}.{name}")
            for name, field in declared.items()
        }
    )


def table(raw: object) -> tuple[ActivationEntry, ...]:
    """Read the entry list, refusing a model that two entries both answer for."""
    entries = tuple(_entry(item, index) for index, item in enumerate(raw))
    seen: dict[tuple[str, str], str] = {}
    for entry in entries:
        for architecture in entry.applies_to:
            claim = (architecture, entry.fingerprint)
            if claim in seen:
                raise SpecRefusal(
                    Rule.SHAPE,
                    f"{architecture!r} with config {entry.fingerprint} resolves "
                    f"to both {seen[claim]!r} and {entry.id!r}",
                    "one model resolves to one entry; merge the widths into it",
                )
            seen[claim] = entry.id
    return entries


def resolve(
    entries: tuple[ActivationEntry, ...],
    architecture: str,
    fingerprint: str | None,
    tp_width: int,
) -> tuple[ActivationEntry, float] | None:
    """The entry measured for this model at this width and its bytes per token,
    or None when nothing was measured for it."""
    for entry in entries:
        if (
            architecture in entry.applies_to
            and fingerprint == entry.fingerprint
            and tp_width in entry.bytes_per_token
        ):
            return entry, entry.bytes_per_token[tp_width]
    return None
