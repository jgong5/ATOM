"""What each family's price may depend on, and what it may not.

A price measured for one operator answers for another only if the two differ
solely in things the price is a known function of. There are three ways a key
component can be treated here and they are not interchangeable:

``FEATURE``
    The price depends on it and the dependence is read off measurements. Token
    rows is the only feature this module carries, because rows is the only one
    the measured points vary over enough to read anything from.

``NUISANCE, measured``
    The price is declared independent of it, and the declaration cites the
    measurement showing the spread across its values. The layer index is the
    case: sixteen attention layers in one step span 1.4% and forty-eight
    linear-attention layers span 1.1%, so collapsing the index costs a known
    band rather than an assumed one.

``NUISANCE, unmeasured``
    Nobody has measured whether the price depends on it. ``slot_mapping`` and
    ``positions`` are here. They may not be dropped to gain key coverage: an
    operator differing from the measured one in an unmeasured nuisance is
    refused, and the refusal names the component. Turning one of these into a
    measured nuisance needs a probe that varies it at a fixed feature point --
    a GPU acquisition, not a decision taken in this file.

Matching two widths of the same operator
----------------------------------------

For a shape-only family the key is the operator name, the operand shapes and
dtypes, and architectural scalars. Between two steps of different width the
only thing that moves is the token-row count. The obvious way to exploit that
-- abstract the row count out of one key and call the result a template --
misreads its own evidence: at 32 rows the architectural constant 17408 is an
exact multiple of 32, so a template built from one measurement cannot tell a
constant from a row-proportional dimension, and it silently decides that
``5120x17408`` scales with the batch.

So nothing is abstracted from a single measurement. Two operators measured at
different row counts are recognised as the same operator at two widths by
comparing them *against each other*, position by position: at each integer
position the two values must either be equal -- an architectural constant, and
17408 equals 17408 whatever the batch was -- or be the same multiple of their
respective row counts, as a Triton grid of ``rows * heads`` is. Equality is
tested first, so a coincidental divisibility never gets the chance to be
mistaken for scaling. At equal row counts the rule degenerates to exact key
equality, which is what it should be.

See :func:`aligns`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from atom.compass.core.cost.identity import normalized_context

__all__ = [
    "FAMILY_CONTRACTS",
    "FamilyContract",
    "Nuisance",
    "aligns",
    "contract_for",
    "grouping_key",
    "row_features",
]


@dataclass(frozen=True)
class Nuisance:
    """A key component the price is declared independent of."""

    #: the component's name as it appears in the operator's context or scalars
    component: str
    #: ``True`` only when a measurement shows the spread across its values
    measured: bool
    #: the relative spread the measurement showed, when there is one
    spread: Optional[float] = None
    #: where that reading comes from, so a reader can check it
    evidence: str = ""
    #: whether the cost key normalises this component away. Independent of
    #: ``measured``: a normalised component is one the key no longer
    #: distinguishes, which is a decision about identity; a measured one is a
    #: component whose spread somebody has read off a measurement. Recording
    #: both is what keeps "we collapsed this" from reading as "we measured it".
    normalized: bool = False

    def describe(self) -> str:
        if not self.measured:
            how = ("normalised out of the cost key on a structural argument, "
                   "spread not yet measured" if self.normalized else
                   "so operators differing in it are refused rather than "
                   "priced")
            return (f"{self.component}: unmeasured -- no measurement varies it "
                    f"at a fixed feature point, {how}")
        band = f" (spread {self.spread:.1%})" if self.spread is not None else ""
        return f"{self.component}: measured independent{band}, {self.evidence}"


@dataclass(frozen=True)
class FamilyContract:
    """The pricing contract for one operator family."""

    family: str
    #: how the price is parameterised: ``"rows"`` or ``"ragged"``
    kind: str
    #: components the price is declared independent of
    nuisances: tuple[Nuisance, ...] = ()
    #: why this family is parameterised the way it is
    rationale: str = ""

    @property
    def unmeasured_nuisances(self) -> tuple[str, ...]:
        return tuple(n.component for n in self.nuisances if not n.measured)

    @property
    def nuisance_spread(self) -> float:
        """The band collapsing this family's measured nuisances costs."""
        spreads = [n.spread or 0.0 for n in self.nuisances if n.measured]
        return max(spreads) if spreads else 0.0

    def describe(self) -> str:
        lines = [f"{self.family}: parameterised by {self.kind}"]
        if self.rationale:
            lines.append(f"    {self.rationale}")
        for n in self.nuisances:
            lines.append(f"    nuisance {n.describe()}")
        return "\n".join(lines)


# The layer index appears in the attention families' scalars as the module path
# of the layer that ran. Sixteen unified-attention layers priced in one step
# (g4/src1/p27bdec32.tp1.r0.json, 8c1fb4bbb2c3c7da) span 1.161e-04 to 1.177e-04
# seconds and forty-eight linear-attention layers in the same file span
# 8.906e-05 to 9.000e-05. Those are measurements of the same work repeated, so
# the index is a nuisance whose cost is the band, not zero.
_LAYER_UNIFIED = Nuisance(
    component="layer",
    measured=True,
    spread=(1.177527e-04 - 1.161958e-04) / 1.161958e-04,
    evidence=("16 layers priced in one step, g4/src1/p27bdec32.tp1.r0.json "
              "(8c1fb4bbb2c3c7da)"),
)
_LAYER_LINEAR = Nuisance(
    component="layer",
    measured=True,
    spread=(8.999375e-05 - 8.906312e-05) / 8.906312e-05,
    evidence=("48 layers priced in one step, g4/src1/p27bdec32.tp1.r0.json "
              "(8c1fb4bbb2c3c7da)"),
)

# Physical allocator and position state. Nothing under g4/ varies either of
# these at a fixed (rows, history) point, so nothing still licenses calling the
# price independent of them *on the evidence*.
#
# They are nevertheless normalised out of the cost key, by
# `atom.compass.core.cost.identity`, on a structural argument rather than a
# measured one: these fields say where the allocator put this batch, and the
# kernel does one write per row wherever the row lands. The count of rows and
# the count of void (-1, padded) entries are kept, because those are work.
#
# `normalized` records that separately from `measured`, so the distinction
# survives: this is a declared equivalence awaiting the probe that would make
# it a measured band, not a measurement. The probe is a fixed (rows, history)
# point priced twice under two allocations; until it runs, the spread is
# unknown and is reported as unknown.
_SLOT_MAPPING = Nuisance(component="slot_mapping", measured=False,
                         normalized=True)
_POSITIONS = Nuisance(component="positions", measured=False, normalized=True)
_BLOCK_TABLES = Nuisance(component="block_tables_shape", measured=False)


#: Families whose key is shapes, dtypes and architectural scalars only. Their
#: price moves with the token rows and with nothing else that changes between
#: steps. Read off the 22 families present in g4/src1/p27bdec32.tp1.r0.json and
#: g4/v2/p27pref{head,deep,tail}.tp1.r0.json.
_ROW_FAMILIES = (
    "aiter::gemm_a16w16",
    "aiter::silu_and_mul",
    "aiter::_fused_qk_rmsnorm_group_quant_kernel",
    "triton::_fused_qk_norm_single_kernel",
    "triton::_mrope_qk_kernel",
    "triton::_mrope_qk_tiled_kernel",
    "aten::add.Tensor",
    "aten::detach",
    "aten::embedding",
    "aten::empty.memory_format",
    "aten::empty_like",
    "aten::mean.dim",
    "aten::min",
    "aten::mul.Tensor",
    "aten::pow.Tensor_Scalar",
    "aten::reshape",
    "aten::rsqrt",
    "aten::sigmoid",
    "aten::silu",
    "aten::split_with_sizes",
    "aten::view",
)

FAMILY_CONTRACTS: dict[str, FamilyContract] = {
    name: FamilyContract(
        family=name,
        kind="rows",
        rationale=("key carries shapes, dtypes and architectural scalars only; "
                   "token rows is the sole component that moves between steps"),
    )
    for name in _ROW_FAMILIES
}

FAMILY_CONTRACTS["aiter::unified_attention_with_output_base"] = FamilyContract(
    family="aiter::unified_attention_with_output_base",
    kind="ragged",
    nuisances=(_LAYER_UNIFIED, _SLOT_MAPPING, _POSITIONS, _BLOCK_TABLES),
    rationale=(
        "price is set by the joint (query rows, history) structure, which the "
        "operand shapes do not carry: g4/v2/p27prefhead and p27prefdeep share "
        "the shapes 16384,6144;16384,1024;16384,1024 and differ 13x "
        "(3.02e-02 vs 3.92e-01 s) on history alone"),
)

FAMILY_CONTRACTS["aiter::linear_attention_with_output_base"] = FamilyContract(
    family="aiter::linear_attention_with_output_base",
    kind="ragged",
    nuisances=(_LAYER_LINEAR,),
    rationale=("key carries num_prefills, num_decodes, num_actual_tokens and "
               "the query start offsets; the prefill/decode split is part of "
               "the feature, not a nuisance"),
)


def contract_for(family: str) -> Optional[FamilyContract]:
    return FAMILY_CONTRACTS.get(family)


def grouping_key(op: dict) -> tuple:
    """A cheap key that two widths of one operator are guaranteed to share.

    Only structure: the name, the dtypes, the arity and rank of the operands,
    and which operands are views rather than dense. Never a value, because
    values are exactly what differs between widths. Its job is to keep
    :func:`aligns` from being asked about obviously unrelated pairs, not to
    decide anything.
    """
    shapes = op.get("input_shapes") or ()
    return (
        op.get("name", ""),
        tuple(op.get("dtypes") or ()),
        tuple(len(s) if isinstance(s, (list, tuple)) else 0 for s in shapes),
        tuple(sorted(k for k, _ in (tuple(x) for x in op.get("scalars") or ()))),
        tuple(sorted(k for k, _ in (tuple(x) for x in op.get("context") or ()))),
        # Which positions carry a recorded layout, and how many fields each
        # records -- structure, not the stride or the offset themselves. A
        # dense rebuild and a strided view of the same shapes are not two
        # widths of one operator, so they do not share a group.
        tuple(sorted((int(pos), len(tuple(value))) for pos, value in
                     (tuple(x) for x in op.get("layouts") or ()))),
    )


def _values(op: dict) -> list:
    """Every value in the operator's key, in a fixed order.

    The same traversal for both operators being compared, so position ``i`` on
    one side means the same thing as position ``i`` on the other. Anything that
    is not an integer is carried through and compared for equality.
    """
    out: list = []

    def walk(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            out.append(("<seq>", len(value)))
            for v in value:
                walk(v)
        else:
            out.append(value)

    walk(op.get("input_shapes") or ())
    walk(tuple(op.get("dtypes") or ()))
    # Addresses summarised to (count, void count) first, on the same rule the
    # cost key uses. The parametric path has to reach the cost key's verdict:
    # if it compared raw `slot_mapping` values it would separate two operators
    # the library has already agreed are the same work, and a family law would
    # then refuse a width the key itself accepts.
    walk(normalized_context(op))
    walk(tuple((i, tuple(v)) for i, v in
               (tuple(x) for x in op.get("int_values") or ())))
    walk(tuple((k, v) for k, v in (tuple(x) for x in op.get("scalars") or ())))
    for key, value in (tuple(x) for x in op.get("launch") or ()):
        if key == "grid":
            walk(tuple(value))
    # How the operands sit in memory, on the same footing as the shapes.
    #
    # `signature_of` excludes layout deliberately, so without this a strided
    # view and a dense rebuild of the same shapes are indistinguishable here --
    # and a width the dense ladder never measured would be interpolated for a
    # strided request and then reported as covered.
    #
    # Carried as values rather than compared separately, so the row-adjustment
    # rule reaches them too: an extent that scales with the rows is the same
    # view at another width, while a stride or an offset that moves under it is
    # a different operator. Sorted by position so two graphs that recorded the
    # same layouts in a different order still line up.
    walk(tuple((int(pos), tuple(value)) for pos, value in
               sorted((tuple(x) for x in op.get("layouts") or ()),
                      key=lambda entry: int(entry[0]))))
    return out


def aligns(op_a: dict, rows_a: int, op_b: dict, rows_b: int) -> bool:
    """Whether these are the same operator run at two widths.

    Position by position, each pair of values must either be equal, or be the
    same positive multiple of their own row counts. Equality is tested first,
    so an architectural constant that happens to divide one of the row counts
    is never mistaken for a dimension that scales with it.

    At ``rows_a == rows_b`` this is exact key equality: equal values pass, and
    unequal ones cannot share a multiple of the same divisor.
    """
    if rows_a <= 0 or rows_b <= 0:
        return False
    va, vb = _values(op_a), _values(op_b)
    if len(va) != len(vb):
        return False
    for x, y in zip(va, vb):
        if x == y:
            continue
        if (isinstance(x, bool) or isinstance(y, bool)
                or not isinstance(x, int) or not isinstance(y, int)):
            return False
        if x <= 0 or y <= 0 or x % rows_a or y % rows_b:
            return False
        if x // rows_a != y // rows_b:
            return False
    return True


def infer_rows(op: dict, measured_op: dict, measured_rows: int) -> Optional[int]:
    """The row count ``op`` must be at to be ``measured_op`` at another width.

    Solved rather than searched. The first position where the two keys differ
    fixes the answer: if the measured side holds ``c * measured_rows`` there,
    the target side holds ``c * rows``, so ``rows`` is that value divided by
    ``c``. The candidate is then put back through :func:`aligns`, so a value
    that happens to divide correctly at one position but not the rest is
    rejected rather than believed.

    Returns ``None`` when no row count makes the two the same operator.
    """
    if measured_rows <= 0:
        return None
    va, vb = _values(op), _values(measured_op)
    if len(va) != len(vb):
        return None
    candidate = None
    for x, y in zip(va, vb):
        if x == y:
            continue
        if (isinstance(x, bool) or isinstance(y, bool)
                or not isinstance(x, int) or not isinstance(y, int)):
            return None
        if y <= 0 or x <= 0 or y % measured_rows:
            return None
        coefficient = y // measured_rows
        if coefficient <= 0 or x % coefficient:
            return None
        solved = x // coefficient
        if candidate is None:
            candidate = solved
        elif candidate != solved:
            return None
    if candidate is None:
        # Nothing differs: the same operator at the same width.
        return measured_rows
    if not aligns(op, candidate, measured_op, measured_rows):
        return None
    return candidate


def row_features(op: dict) -> tuple:
    """The part of a ragged family's key that is a feature rather than state.

    Returned for reporting and for the ragged families' exact-match path; the
    unmeasured nuisances are deliberately absent, which is why an operator that
    differs only in them is refused rather than answered from this.
    """
    wanted = {"num_prefills", "num_prefill_tokens", "num_decodes",
              "num_decode_tokens", "num_actual_tokens", "context_lens",
              "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k",
              "min_seqlen_q", "has_cached", "state", "is_prefill"}
    context = {k: v for k, v in (tuple(x) for x in op.get("context") or ())
               if k in wanted}
    return tuple(sorted(context.items(), key=lambda kv: kv[0]))


def unmeasured_differences(op: dict, other: dict) -> tuple[str, ...]:
    """Unmeasured-nuisance components on which these two operators differ.

    Named so a refusal can say which one blocked it rather than refusing
    anonymously.
    """
    contract = contract_for(op.get("name", ""))
    if contract is None:
        return ()
    mine = dict((k, v) for k, v in (tuple(x) for x in op.get("context") or ()))
    theirs = dict((k, v) for k, v in
                  (tuple(x) for x in other.get("context") or ()))
    return tuple(c for c in contract.unmeasured_nuisances
                 if mine.get(c) != theirs.get(c))
