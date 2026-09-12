"""Cost identity: the same work, wherever the allocator happened to put it.

``signature_of`` is an *observation* identity. It answers "which call was
this", and for that job every field it carries is right: two calls that
differed in any of them were not the same call. A price library asks a
different question -- "does this measurement answer for this operator" -- and
some of what distinguishes two calls does not distinguish two amounts of work.

The concrete case. A decode batch of eight rows at history 1151 was priced with

    slot_mapping=[9102, 9118, 9134, 9150, 9166, 9182, 9198, 9214]

and the same eight rows at the same history, scheduled by a replay, arrive with

    slot_mapping=[1150, 2302, 3454, 4606, 5758, 6910, 8062, 9214]

Everything else in the key -- shapes, dtypes, ``context_lens``,
``cu_seqlens_q``, ``max_seqlen_k``, state, layout -- is identical. The kernel
writes eight KV rows either way. Keyed on the addresses, the measurement can
only ever answer for the one allocation it was taken under, so no replay whose
scheduler assigns blocks differently can be priced at all, however many
captures are taken. That is a key that cannot generalise, not a gap in the
data.

What is normalised, and what is emphatically not
------------------------------------------------

Only absolute addresses: where the allocator put this batch's KV rows and
linear-attention state, and which absolute token positions the rows sit at.
For each, two things survive, because both are work:

* **how many** entries there are -- one KV write per row;
* **how many of them are void** -- a padded row carries ``-1``, meaning "no
  slot". A batch of 32 with 8 real rows and 24 padded ones is not a batch of
  32, and collapsing those two would be exactly the error this module exists
  to avoid making in the other direction.

Everything else stays in the key: ``context_lens`` and ``max_seqlen_k`` (the
history each row reads), ``cu_seqlens_q``/``cu_seqlens_k`` (the ragged
structure), ``has_cached``, ``state``, ``is_prefill``, ``block_tables_shape``
(how many blocks are walked), operand shapes, dtypes, layout, scalars, Triton
grid, and the scope a price was measured in (topology, registration). A price
still refuses across any of those.

Why this is a string function
-----------------------------

Prices already on disk are keyed by the signature string they were measured
under. Applying one function to both the stored key and the freshly computed
signature reindexes every retained artifact without recollecting anything, and
makes it impossible for the two sides to normalise differently -- a second
implementation is how prices keyed one way and looked up another end up
agreeing on most operators and disagreeing on whichever detail drifted.

The raw signature is never discarded. The library keeps it on the record it
came from, so a lookup answered under a shifted allocation can be told from one
answered under the measured allocation, and counted separately.
"""

from __future__ import annotations

import ast

__all__ = ["ADDRESS_COMPONENTS", "cost_key", "normalize_component",
           "normalized_context", "describe_normalization"]

#: Context components that name absolute physical locations rather than work.
#: ``block_tables`` is absent because ``signature_of`` already drops it, and
#: ``block_tables_shape`` is absent because it is a count of blocks walked.
ADDRESS_COMPONENTS = (
    "slot_mapping",
    "positions",
    "non_spec_state_indices_tensor",
    "non_spec_state_indices_in_tensor",
    "state_indices",
)

#: How a normalised component is written into the key. Distinct enough from any
#: real value that an un-normalised key and a normalised one cannot collide.
_SHAPE = "<n={n},void={void}>"


def _summarise(items) -> str:
    total = 0
    void = 0
    for item in items:
        total += 1
        text = item.strip() if isinstance(item, str) else item
        if isinstance(text, str):
            if text.startswith("-"):
                void += 1
        elif isinstance(text, (int, float)) and text < 0:
            void += 1
    return _SHAPE.format(n=total, void=void)


def _normalize_value(value):
    """The structural rule, applied to a Python value from either path.

    One rule, one place. The string path parses first and then lands here, so
    a stored key and a freshly computed one cannot be summarised differently
    -- which they were when each path did its own splitting: the state-index
    tensors are recorded as ``[values, dtype]``, and splitting that text on
    commas counted the dtype and the brackets as entries while the operator
    path counted two.
    """
    if isinstance(value, (list, tuple)):
        if (len(value) == 2 and isinstance(value[0], (list, tuple))
                and isinstance(value[1], str)):
            # ``[values, dtype]``. The dtype is kept: it is what the kernel
            # reads, not where the allocator put it.
            return f"[{_summarise(value[0])}, {value[1]!r}]"
        return _summarise(value)
    return str(value)


def normalize_component(name: str, value) -> str:
    """The work-bearing summary of one component's value.

    Takes the value either as the text a signature carries or as the list an
    operator record holds, so the string path and the op path cannot drift.
    Anything that is not a sequence is returned unchanged: a scalar in one of
    these fields is not an address list and this module has nothing to say
    about it.
    """
    if name not in ADDRESS_COMPONENTS:
        return value if isinstance(value, str) else str(value)
    if not isinstance(value, str):
        return _normalize_value(value)
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        # Already normalised, or never an address list. Either way, untouched:
        # this is what makes the function idempotent, so a library may be
        # reindexed more than once.
        return text
    inner = text[1:-1]
    if "[" not in inner:
        # The common case by far -- a flat list of a few thousand integers.
        # Split rather than parse: `literal_eval` on a 16384-entry prefill
        # `slot_mapping`, 2439 operators to a graph, is minutes of parsing for
        # a count.
        stripped = inner.strip()
        return _summarise(stripped.split(",") if stripped else ())
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    return _normalize_value(parsed)


def _normalize_segment(segment: str) -> str:
    """One ``k=v;k=v`` segment of a signature, addresses summarised."""
    out = []
    for item in segment.split(";"):
        name, sep, value = item.partition("=")
        if not sep or name not in ADDRESS_COMPONENTS:
            out.append(item)
            continue
        out.append(f"{name}={normalize_component(name, value)}")
    return ";".join(out)


def cost_key(signature: str) -> str:
    """A signature reduced to what a price is a price of.

    Applied to a freshly computed signature and to a signature read from a
    price file alike. Idempotent: normalising an already-normalised key leaves
    it alone, so a library may be reindexed more than once.
    """
    if not signature:
        return signature
    return "|".join(_normalize_segment(part) if "=" in part else part
                    for part in signature.split("|"))


def normalized_context(op: dict) -> tuple:
    """This operator's ``context`` with addresses summarised, as pairs.

    For the family path, which compares operator records rather than signature
    strings and must reach the same verdict.
    """
    return tuple(
        (k, normalize_component(k, v) if k in ADDRESS_COMPONENTS else v)
        for k, v in (tuple(x) for x in op.get("context") or ()))


def describe_normalization() -> str:
    """One line per normalised component, for a provenance record."""
    return ("cost identity normalises absolute allocator addresses -- "
            + ", ".join(ADDRESS_COMPONENTS)
            + " -- to (count, void count); work shape, history, ragged "
              "structure, layout, scope and scalars are unchanged")
