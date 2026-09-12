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
    "ValueContract",
    "Nuisance",
    "aligns",
    "contract_for",
    "grouping_key",
    "executed_rows",
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
    #: where this family's executed width is read off its own operator, as
    #: ``(operand position, dimension)``. ``None`` leaves the width to the file
    #: the measurement came from, which is what every family did before the
    #: head region needed otherwise. See :func:`executed_rows`.
    rows_from: Optional[tuple[int, int]] = None
    #: operand dimensions that must match exactly rather than scale with the
    #: row count, as ``(operand position, dimension)``. Empty for almost every
    #: family, and load-bearing for the few whose key holds a second extent
    #: that is independent of the width: :func:`aligns` accepts any integer
    #: that is the same multiple of its own row count, so without a declaration
    #: here a training pair that moved BOTH extents together lets
    #: :func:`infer_rows` solve the second one along with the rows. See
    #: :func:`_values`, which is where the declaration takes effect.
    fixed_dims: tuple[tuple[int, int], ...] = ()
    #: operand positions whose integer CONTENTS the price is declared
    #: independent of, each with the validation that must hold before the
    #: declaration applies. Empty for every family but the two head selectors.
    #: See :class:`ValueContract`, and :func:`_abstracted_int_values`, which is
    #: where a declaration takes effect.
    values: tuple["ValueContract", ...] = ()
    #: why this family is parameterised the way it is
    rationale: str = ""

    @property
    def all_nuisances(self) -> tuple[Nuisance, ...]:
        """Every declared nuisance, wherever it was declared.

        A family declares independence in two places. `nuisances` holds the
        components that are not operands at all -- the layer index, the slot
        mapping -- and `values[i].nuisance` holds the band that abstracting a
        declared integer payload costs. Both are the same kind of claim and
        both have to reach the same two consumers, or a band gets promised in
        a contract and then dropped on the way to the price: the two head
        selectors declare their whole uncertainty through `values`, and
        `nuisances` is empty for them, so reading only `nuisances` returned a
        spread of 0.0 for the two families that have one.
        """
        return self.nuisances + tuple(v.nuisance for v in self.values)

    @property
    def unmeasured_nuisances(self) -> tuple[str, ...]:
        return tuple(n.component for n in self.all_nuisances if not n.measured)

    @property
    def nuisance_spread(self) -> float:
        """The band collapsing this family's measured nuisances costs.

        The widest, not the sum: these are alternative readings of the same
        price, not independent error terms to accumulate. It is combined in
        quadrature with the measurement spread downstream, which is where the
        independence assumption belongs.
        """
        spreads = [n.spread or 0.0 for n in self.all_nuisances if n.measured]
        return max(spreads) if spreads else 0.0

    def describe(self) -> str:
        lines = [f"{self.family}: parameterised by {self.kind}"]
        if self.rationale:
            lines.append(f"    {self.rationale}")
        for v in self.values:
            lines.append(f"    values[{v.position}] {v.nuisance.describe()}")
        for n in self.nuisances:
            lines.append(f"    nuisance {n.describe()}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ValueContract:
    """When an integer operand's CONTENTS may stand for their extent alone.

    ``int_values`` is recorded because for a data-dependent kernel the shapes
    do not say how much work is done: attention walks as much KV cache as
    ``context_lens`` says, and a benchmark handed a zero-filled tensor of the
    right shape priced one decode step's attention above the whole step. That
    argument is real and it is why the values are in the key. It does not reach
    every operator carrying an integer operand, and where it does not reach,
    holding the literal vector costs every width: the head's gather carries the
    selected row numbers, which differ at every width AND at every mix of
    request lengths, so :func:`aligns` compares 4095 against 8191 and refuses
    two measurements of the same operator.

    A declaration here says, for one family at one operand position: the price
    is a function of how many values there are, not of which values they are.
    It is admissible only when :attr:`validate` confirms the operand really is
    the thing the declaration describes -- an operand that fails validation
    keeps its literal values and goes on refusing exactly as before, which is
    the fail-closed direction. And it must carry a :attr:`nuisance` whose
    measurement varies the values at a fixed width, on the same footing this
    module demands of every other nuisance.

    The values are never removed from the operator. This changes what the
    comparison looks at, not what the recording holds: the raw vector, its
    range and its layout association all stay on the ``OpSpec``, and a reader
    or a later probe can still reach them.
    """

    #: operand position whose contents may be abstracted
    position: int
    #: name in :data:`_VALUE_VALIDATORS`: what must hold of the operand before
    #: the abstraction applies
    validate: str
    #: the declaration that the price does not depend on the values, with the
    #: measurement that varied them at a fixed width behind it
    nuisance: Nuisance
    #: why the abstraction is admissible for this family
    rationale: str = ""


_INT_DTYPES = ("int8", "int16", "int32", "int64",
               "uint8", "uint16", "uint32", "uint64", "bool")


def _dtype_at(op: dict, position: int) -> str:
    dtypes = op.get("dtypes") or ()
    return str(dtypes[position]) if position < len(dtypes) else ""


def _vector_extent(op: dict, position: int) -> Optional[int]:
    """The length of a one-dimensional operand, or ``None`` if it is not one."""
    shapes = op.get("input_shapes") or ()
    if position >= len(shapes):
        return None
    shape = shapes[position]
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        return None
    return int(shape[0])


def _validates_int_elementwise(op: dict, position: int, values) -> bool:
    """An integer vector whose kernel touches each element once.

    What is checked is that the operand is what the declaration assumes: a
    one-dimensional integer tensor, of the length the recording says, and that
    the operator's declared width is that length. The claim being licensed is
    narrow -- elementwise integer work at a fixed dtype and extent does the
    same work whatever the integers are -- and it is not a claim that the
    operator is cheap.
    """
    if _dtype_at(op, position) not in _INT_DTYPES:
        return False
    extent = _vector_extent(op, position)
    if extent is None or extent != len(values) or extent <= 0:
        return False
    return executed_rows(op) == extent


def _validates_row_selector(op: dict, position: int, values) -> bool:
    """A gather index that selects distinct existing rows, once each.

    Three things must hold before row numbers may stand for their count, and
    each rules out a different way the price could depend on the values:

    * every index inside the source height -- an out-of-bounds or negative
      entry is not a row of this tensor, and a vector holding one is not the
      access pattern this contract describes;
    * no row selected twice -- the same count reading one row repeatedly is a
      different amount of memory traffic from one reading distinct rows, and
      the second is the only one measured;
    * as many indices as the operand says, and as many as the operator's
      declared width -- which is what ties the count to the rows rather than
      leaving two independent extents.

    Clustering is deliberately NOT checked. Whether the selected rows sit
    together or spread across the tensor is a property the measurement varies
    (see the selector probe behind the nuisance) rather than one this function
    asserts away.
    """
    shapes = op.get("input_shapes") or ()
    if position >= len(shapes) or not shapes or not shapes[0]:
        return False
    if _dtype_at(op, position) not in _INT_DTYPES:
        return False
    height = int(shapes[0][0])
    extent = _vector_extent(op, position)
    if extent is None or extent != len(values) or extent <= 0:
        return False
    numbers = [int(v) for v in values]
    if any(v < 0 or v >= height for v in numbers):
        return False
    if len(set(numbers)) != len(numbers):
        return False
    return executed_rows(op) == extent


#: The validations a :class:`ValueContract` may name. Keyed by name rather than
#: holding the function itself so a contract stays a plain data literal.
_VALUE_VALIDATORS = {
    "int_elementwise": _validates_int_elementwise,
    "row_selector": _validates_row_selector,
}


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

#: Where each row family's executed width is read off its own operator. Absent
#: means "the file's width", which is what every family used before the head
#: region showed the two can differ. Each entry here is a checked reading, not
#: a convention -- see :func:`executed_rows` for the counts.
_ROWS_FROM: dict[str, tuple[int, int]] = {
    "aiter::gemm_a16w16": (0, 0),
    "aten::embedding": (1, 0),
}

FAMILY_CONTRACTS: dict[str, FamilyContract] = {
    name: FamilyContract(
        family=name,
        kind="rows",
        rows_from=_ROWS_FROM.get(name),
        rationale=("key carries shapes, dtypes and architectural scalars only; "
                   "token rows is the sole component that moves between steps"),
    )
    for name in _ROW_FAMILIES
}

# The head selectors' VALUES, varied at a fixed width.
#
# Both metadata operators carry an integer vector whose contents change with
# every batch: the subtraction's operand is the cumulative token offsets and
# the gather's is the row numbers those offsets point at. Holding the literal
# vectors in the comparison key is why both refused at every width -- not
# because the family was unmeasured, but because no two measurements could ever
# be recognised as the same operator.
#
# So the values were varied at a fixed width, which is the probe this module
# demands before any component may be called a nuisance. Eight requests,
# H=16384, five length distributions from equal-split to seven single-token
# requests and one covering the rest (agent_scratch/headgrid/specs_native,
# head_sel_m8_v1..v5), each priced three times under the ladder's own protocol.
# The spread below is across those five value-sets at one width, so it is a
# measured band rather than an assumed independence.
_SELECTOR_SUB = Nuisance(
    component="int_values[0]",
    measured=True,
    spread=0.0077,
    evidence=("5 value-sets at M=8, headgrid/prices_native/head_sel_m8_v*, "
              "1.956057e-06 to 1.971052e-06 s; the band is narrower than the "
              "1.28% repeat spread of the widest single point, so nothing in "
              "the measurement separates the value-sets from each other"),
)
# The gather's band is dominated by one value-set, v5 = [1]*7 + [16377], the
# pathological one chosen to stress locality. Its own repeat spread is 51.66%
# -- larger than the 18.64% band across all five -- which says that point is a
# reading of the event-timing floor at ~4 us, not of a locality effect the
# measurement can resolve. The larger number is recorded anyway: a nuisance
# band that is bounded by resolution rather than by demonstrated independence
# should be declared at its resolution, not narrowed by dropping the point
# that made it wide. Excluding v5 would give 3.0% across v1..v4, and that
# figure is deliberately NOT the one used here.
_SELECTOR_INDEX = Nuisance(
    component="int_values[1]",
    measured=True,
    spread=0.1864,
    evidence=("5 value-sets at M=8, headgrid/prices_native/head_sel_m8_v*, "
              "4.204401e-06 to 4.988156e-06 s; widest at v5=[1]*7+[16377], "
              "whose own 51.66% repeat spread exceeds the band"),
)

# The two head metadata operators. Both ran in run 5's head graph, both were
# refused with "has no declared family contract", and neither is free: the
# gather is 4.3e-06 s and the subtraction 2.0e-06 s at M=2, measured on the
# standalone head grid. Declaring them is what lets a head step be priced
# without treating a gather as zero.
FAMILY_CONTRACTS["aten::sub.Tensor"] = FamilyContract(
    family="aten::sub.Tensor",
    kind="rows",
    rows_from=(0, 0),
    values=(
        ValueContract(
            position=0,
            validate="int_elementwise",
            nuisance=_SELECTOR_SUB,
            rationale=("the operand is the cumulative offset vector and the "
                       "kernel subtracts a scalar from each entry once; at a "
                       "fixed int32 dtype and extent that is the same work "
                       "whatever the offsets are, and the measured band above "
                       "is what the declaration costs"),
        ),
    ),
    rationale=("elementwise over the selected last-token indices; operand 0 is "
               "that index vector, so its length is the executed width"),
)

# The gather that selects the last token of each request out of the hidden
# state. Its width is the number selected -- operand 1's length -- while the
# height it selects FROM is operand 0 dimension 0 and is a second, independent
# extent. It is declared `fixed_dims` rather than left to the grouping key,
# which records only each operand's rank and so says nothing about the value:
# two measurements that moved the selected count and the source height
# together -- 2 out of 8192 and 4 out of 16384 -- are otherwise read as one
# operator at two widths with a coefficient of 4096, and a request gathering 3
# out of 12288 is then solved and interpolated from them, against a source
# height nobody measured. With the declaration those two are not the same
# operator at all, which is the honest answer: the height is not the width.
# A fixed height with a moving selected count still interpolates, which is the
# case the head region actually needs.
#
# The feature width, operand 0 dimension 1, is fixed for the same reason and
# is not covered by fixing the height. The bytes the gather moves are the
# selected count times the feature width, so the two can trade against each
# other inside one law: measurements at (M=2, N=256) and (M=4, N=512) sit on a
# straight line through a query at (M=3, N=384) even though no measurement
# ever held N still, and the interpolated price would be reported as covered.
# Fixing the width makes those two different operators, so the only thing a
# curve can be read along is the selected count -- which is what this narrow
# head contract claims and all it claims. This is scoped to the head's gather;
# the conflicting-width guard the body already applies is a separate mechanism
# and is untouched.
FAMILY_CONTRACTS["aten::index.Tensor"] = FamilyContract(
    family="aten::index.Tensor",
    kind="rows",
    rows_from=(1, 0),
    fixed_dims=((0, 0), (0, 1)),
    values=(
        ValueContract(
            position=1,
            validate="row_selector",
            nuisance=_SELECTOR_INDEX,
            rationale=("the operand is the last-token row number of each "
                       "request: validated in bounds, distinct, and as many as "
                       "the width. Under those conditions the gather reads that "
                       "many distinct rows of the source whatever their "
                       "numbers, which the five value-sets confirm; a vector "
                       "that fails any of the three keeps its literal values "
                       "and is refused"),
        ),
    ),
    rationale=("gathers operand 1's rows out of operand 0; the selected count "
               "is the width and the source height is fixed, never scaled"),
)

# `aten::slice.Tensor` is priced by its ALIAS, not by a row law. Its operand is
# the cumulative sequence-offset vector, whose length is requests + 1: affine
# in the width, not a multiple of it, so `aligns` cannot recognise two of its
# widths as the same operator and a "rows" contract would be a false
# declaration. That much was already established and is unchanged.
#
# What is new is that the family does not need one. The recording says the
# operator allocated nothing -- `output_aliases` holds an index rather than
# None, decided by whether the output's storage is one of the operator's own
# inputs -- so it returned a view: the same storage at an offset, with no
# kernel dispatched. A price of zero here is a structural fact about that
# recording and not a small measurement rounded down, which is the distinction
# `ZERO_WORK_FLAG` exists to keep. The standalone head grid corroborates it
# without being the basis for it: 9.3e-08 s, four orders below the LM-head
# GEMM in the same graph and at the timing floor.
#
# A slice whose recording shows an allocation is a copy, and is refused. A
# graph too old to record `output_aliases` says nothing about which it was, and
# nothing is not zero -- so that is refused too. See `_view_price`.
FAMILY_CONTRACTS["aten::slice.Tensor"] = FamilyContract(
    family="aten::slice.Tensor",
    kind="view",
    rationale=("priced at zero only on a recording that proves the output "
               "aliases an operand's storage, so no kernel ran; a slice that "
               "allocated is a copy and is refused"),
)

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




def executed_rows(op: dict) -> Optional[int]:
    """The width THIS operator ran at, read off the operator itself.

    A file-level width answers "how many rows did the step this measurement
    came from schedule". For most families that is also the width every
    operator in the file ran at, and reading it once per file is both cheaper
    and the only reading available -- an ``aten::view`` carries no operand that
    says which of its dimensions the batch is.

    The head region is where the two readings part company. A head graph is
    traced over the hidden state handed to ``compute_logits`` -- 16384 rows at
    the 16384-token prefill -- but ``compute_logits`` first selects the last
    token of each request, so the LM-head GEMM runs at the *request* count.
    Run 5 executed ``[2,5120] x [248320,5120]``: M=2 inside a file whose
    traced width is 16384. Pricing that GEMM as a 16384-row measurement is not
    an approximation, it is a measurement of different work.

    So the reading is declared per family rather than inferred, and only where
    the operand position carrying the width is known:

    * ``aiter::gemm_a16w16`` -- operand 0, dimension 0 is M. Checked against
      every existing measurement: on the 40 ``gemm_a16w16`` observations in the
      run-5 price list this reading reproduces the file width 40 times and
      disagrees 0 times, so declaring it changes no price already taken.
    * ``aten::embedding`` -- operand 1, dimension 0 is the token index, which
      is the same tensor ``_traced_rows`` reads the file width from. Declaring
      it is a restatement, not a new claim.
    * ``aten::sub.Tensor`` and ``aten::index.Tensor`` -- the two head metadata
      operators whose width is the request count. Neither had a contract at
      all, so both refused with "no declared family contract" however the
      library was built.

    Every other family keeps ``rows_from=None`` deliberately. The same check
    that licenses the GEMM refuses a blanket rule: reading operand 0 dimension
    0 on ``aten::embedding`` hits the weight table (0 of 8 agree), on
    ``aten::min`` 0 of 8, on ``aten::empty.memory_format`` 0 of 9, on
    ``aten::reshape`` 8 of 16. A global "rows are operand 0 dimension 0" would
    silently refile two thirds of the existing library.

    ``None`` when the family declares no reading, or when the operator does not
    carry the declared position.
    """
    contract = contract_for(op.get("name", ""))
    if contract is None or contract.rows_from is None:
        return None
    position, dimension = contract.rows_from
    shapes = op.get("input_shapes") or ()
    if position >= len(shapes):
        return None
    shape = shapes[position]
    if dimension >= len(shape):
        return None
    rows = int(shape[dimension])
    return rows if rows > 0 else None


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


def _fixed_shapes(op: dict) -> Any:
    """``input_shapes`` with the family's fixed dimensions marked as fixed.

    A marked dimension is carried as a string rather than an integer, so both
    :func:`aligns` and :func:`infer_rows` reach it on their equality branch and
    never on their multiple-of-the-rows branch. That is the whole mechanism:
    no separate comparison pass, no position arithmetic on the flattened key,
    and nothing to keep in step between the two functions.
    """
    shapes = op.get("input_shapes") or ()
    contract = contract_for(op.get("name", ""))
    fixed = contract.fixed_dims if contract is not None else ()
    if not fixed:
        return shapes
    out = [list(s) if isinstance(s, (list, tuple)) else s for s in shapes]
    for position, dimension in fixed:
        if position >= len(out) or not isinstance(out[position], list):
            continue
        if dimension < len(out[position]):
            out[position][dimension] = f"fixed:{out[position][dimension]}"
    return out


def _abstracted_int_values(op: dict) -> tuple:
    """``int_values`` with declared-and-validated contents reduced to extent.

    The operator is not modified: this is the comparison's view of it, built
    fresh on each call, in the same spirit as :func:`_fixed_shapes`. A position
    with no declaration, or one whose contents fail the declared validation,
    is carried through with its values intact and so goes on matching nothing
    but an identical vector.

    A validated position becomes ``(position, count)``. The count is an integer
    and it equals the operator's width, so the row-adjustment rule in
    :func:`aligns` reaches it: n values at one width and 2n at twice the width
    are recognised as one operator at two widths, which is the whole point, and
    a count that did NOT move with the rows would still be refused.
    """
    raw = tuple((int(i), tuple(v)) for i, v in
                (tuple(x) for x in op.get("int_values") or ()))
    contract = contract_for(op.get("name", ""))
    declared = {v.position: v for v in (contract.values if contract else ())}
    if not declared:
        return raw
    out = []
    for position, values in raw:
        rule = declared.get(position)
        validator = _VALUE_VALIDATORS.get(rule.validate) if rule else None
        if validator is None or not validator(op, position, values):
            out.append((position, values))
            continue
        out.append((position, len(values)))
    return tuple(out)


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

    walk(_fixed_shapes(op))
    walk(tuple(op.get("dtypes") or ()))
    # Addresses summarised to (count, void count) first, on the same rule the
    # cost key uses. The parametric path has to reach the cost key's verdict:
    # if it compared raw `slot_mapping` values it would separate two operators
    # the library has already agreed are the same work, and a family law would
    # then refuse a width the key itself accepts.
    walk(normalized_context(op))
    walk(_abstracted_int_values(op))
    # Scalar VALUES, not just their names.
    #
    # `grouping_key` carries scalar *names* only, so this walk is the sole
    # place a scalar's value is compared at all. Two operators with the same
    # scalar names and different values -- a different Triton `num_warps`, a
    # different stride, a different epsilon -- share a grouping key and must
    # be separated here or they would align, and a curve fitted on one
    # configuration would interpolate a width for the other.
    #
    # Deliberately outside the nuisance abstraction above: that abstraction
    # applies only to declared integer payload positions, which are entries of
    # `int_values`, never scalars. A scalar is never abstracted by any
    # contract.
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
