# SPDX-License-Identifier: MIT
"""`atom.compass.ir.grouping`: proving that a detected grouping was free.

A detector proposes repeats by comparing what blocks are. Nothing it compares
is a cost, so every repeat it emits is a claim waiting to be checked, and this
is where it is checked: price the sequence, price the repeat that would replace
it, and keep the repeat only if the two numbers are the same number.

The named result is one stack accepted with both prices agreeing and one stack
rejected with the disagreement reported, both built to order. Around it, a test
for each of the three ways a real stack refuses to group:

* **The first layer differs from the rest.** It has to stay a sibling while the
  rest still groups, and it must never be priced as if it were one of them.
* **Per-layer quantisation scales differ without differing in cost.** A
  signature difference is not a cost difference. Told which attribute to leave
  out, the stack groups and the grouping proves free; told nothing, it does not
  group at all -- compression lost, correctness kept.
* **A hybrid interleaves its layer classes rather than blocking them.** Three
  of one class then one of another, that period twenty times. Both levels have
  to group, and every one of the eighty blocks has to be priced at the layer it
  actually sits at -- which under two levels of binding is the sum of the
  values bound above it plus its offset in the body that holds it.

Three more, each of which is the difference between a check and a formality:

* **The equality is bitwise and the summation order is fixed.** The grouped
  form folds the same prices in the same order as the flat form, so the two
  agree bit for bit exactly when every instance prices as the first one. One
  bit of difference at one instance is refused, and a body priced once and
  multiplied by its count is a different number from the same body folded --
  that difference is pinned here rather than described.
* **A price is asked for with the key, never with the shape alone.** Every
  instance of one body has the same shapes by construction, so a shape-keyed
  price reports agreement for every grouping ever proposed, including the ones
  that are wrong.
* **The structural evidence is recomputed from the body beside it.** A
  mandatory field nobody recomputes records whatever was written into it.

And throughout: nothing reads a symbolic dimension. Every shape here carries a
stand-in that raises on comparison, on conversion and on hashing outside the
window in which a shape is built, so a proof that reached one -- by comparing
two regions, by putting one in a set, by rebuilding an operator -- fails here
rather than silently installing a guard on the record it was reading.
"""

import math
from contextlib import contextmanager

import pytest

from atom.compass import ir
from atom.compass.ir import (
    ContextRef,
    EqualPrice,
    IdenticalStructure,
    IndexBinding,
    JoinPolicy,
    NodeKind,
    Op,
    Par,
    Repeat,
    Seq,
    Ungrouped,
    detect_repeats,
    prove_grouping,
    signature_of,
)

_BUILDING = False


@contextmanager
def _building():
    """The one window in which a dimension may be hashed.

    Building a shape hashes every dimension handed to it, so the stand-in below
    has to survive that much or nothing here could be constructed at all.
    Outside it the stand-in refuses, which is what turns "the proof compares
    prices and signature text, and never two regions" into something a test can
    fail.
    """
    global _BUILDING
    previous = _BUILDING
    _BUILDING = True
    try:
        yield
    finally:
        _BUILDING = previous


class _Symbol:
    """A symbolic dimension that fails the moment anything reads it.

    Its hash is the identity hash and not a constant: a constant forces an
    equality comparison the moment two of them meet in one dict, and a test
    that passes because of that is testing the stand-in.
    """

    def __repr__(self):
        return "s52"

    def __hash__(self):
        if not _BUILDING:
            raise AssertionError("a symbolic dimension was hashed")
        return object.__hash__(self)

    def _refuse(self, *_args):
        raise AssertionError("a symbolic dimension was compared or converted")

    __eq__ = __lt__ = __le__ = __gt__ = __ge__ = _refuse
    __int__ = __index__ = __bool__ = __float__ = _refuse


TOKENS = _Symbol()

#: What each operator costs, in seconds. A tenth of a second is nobody's kernel
#: and is here for one test: six of them folded and six of them multiplied
#: are different numbers, which is the whole of why the grouped form folds.
SECONDS = {
    "aiter::rmsnorm": 1e-6,
    "aiter::gemm_a16w16": 2e-6,
    "aiter::linear_attention": 3e-6,
    "aiter::full_attention": 5e-6,
    "aiter::tenth": 0.1,
}

BLOCK_SECONDS = 1e-6 + 3e-6 + 2e-6


def _op(name="aiter::gemm_a16w16", **kwargs):
    kwargs.setdefault("kind", NodeKind.CAPTURED)
    kwargs.setdefault("in_shapes", ((TOKENS, 4096),))
    kwargs.setdefault("out_shapes", ((TOKENS, 4096),))
    with _building():
        return Op(name=name, **kwargs)


def _block(variant="linear_attention", **attrs):
    """One layer as a tracer records it.

    The key carries the repeat index as a placeholder rather than a number,
    which is what makes the same layer class at layer 3 and at layer 47 one
    body -- and what leaves pricing an instance needing to know where that
    instance sits.
    """
    return Seq(
        (
            _op("aiter::rmsnorm"),
            _op(
                f"aiter::{variant}",
                kind=NodeKind.OPAQUE_LEAF,
                context_ref=ContextRef(f"model.layers.{{layer}}.{variant}"),
                attrs=attrs,
            ),
            _op("aiter::gemm_a16w16"),
        )
    )


def _dense_stack(layers=8):
    return [_block("linear_attention") for _ in range(layers)]


def _hybrid_stack(periods=20):
    """`AAAB` repeated: three of one layer class then one of another."""
    blocks = []
    for _ in range(periods):
        blocks += [_block("linear_attention") for _ in range(3)]
        blocks.append(_block("full_attention"))
    return blocks


def _layer_of(key):
    return int(key.split(".")[2])


class _Pricer:
    """A price for each operator, and the record of what it was asked.

    A price is a function of the operator *and* of the key it reads its ambient
    state through. `by_layer` is how a test says that one layer really does
    cost something else -- the case a price taken from the shapes alone cannot
    see.
    """

    def __init__(self, by_layer=None):
        self.asked = []
        self.by_layer = by_layer

    def __call__(self, op, key):
        self.asked.append((op.name, key))
        seconds = SECONDS[op.name]
        if self.by_layer is not None and key is not None:
            seconds = self.by_layer(_layer_of(key), seconds)
        return seconds

    @property
    def keys(self):
        return [key for _, key in self.asked if key is not None]

    @property
    def layers(self):
        return [_layer_of(key) for key in self.keys]


# --- the named result --------------------------------------------------------


def test_a_uniform_stack_groups_and_both_prices_agree():
    """Eight identical layers: the repeat is emitted, carrying the two prices.

    The price calls are counted because the count is what grouping buys. Eight
    bodies are priced to derive the flat number and one to derive the grouped
    one, and after that a replay asks for one body's prices however many layers
    the stack has.
    """
    blocks = _dense_stack(8)
    price = _Pricer()

    proved, refused = prove_grouping(detect_repeats(blocks), price)

    assert refused == ()
    assert isinstance(proved, Repeat)
    assert proved.count == 8
    assert isinstance(proved.evidence, EqualPrice)
    assert proved.evidence.flat_seconds == proved.evidence.grouped_seconds
    assert proved.evidence.flat_seconds == pytest.approx(8 * BLOCK_SECONDS)
    assert len(price.asked) == 8 * 3 + 1 * 3
    assert price.layers == [*range(8), 0]


def test_a_stack_with_one_costlier_layer_is_refused_and_the_difference_is_named():
    """The same eight layers, one of which costs twice as much.

    Nothing about the blocks differs -- same operators, same shapes, same
    attributes -- so the detector proposes the same repeat. It is the price
    through the key that differs, and the repeat does not survive it.
    """
    blocks = _dense_stack(8)
    price = _Pricer(by_layer=lambda layer, s: s * 2 if layer == 5 else s)

    proved, refused = prove_grouping(detect_repeats(blocks), price)

    assert len(refused) == 1
    (only,) = refused
    assert isinstance(only, Ungrouped)
    assert (only.at, only.count) == (0, 8)
    assert only.flat_seconds != only.grouped_seconds
    assert only.flat_seconds == pytest.approx(8 * BLOCK_SECONDS + 3e-6)
    assert only.grouped_seconds == pytest.approx(8 * BLOCK_SECONDS)
    assert "at 3e-06 s where the sequence itself prices it at 6e-06 s" in only.reason
    assert "multiplied by the count" in only.reason
    assert "8 instances at 0" in str(only)

    assert isinstance(proved, Seq)
    assert len(proved.items) == 8
    assert not any(isinstance(item, Repeat) for item in proved.items)


# --- the three ways a real stack refuses to group ----------------------------


def test_a_first_layer_that_differs_stays_a_sibling_and_is_never_priced():
    blocks = [_block("full_attention"), *_dense_stack(7)]
    price = _Pricer()

    proved, refused = prove_grouping(detect_repeats(blocks), price)

    assert refused == ()
    assert isinstance(proved, Seq)
    prologue, grouped = proved.items
    assert prologue is blocks[0]
    assert isinstance(grouped, Repeat)
    assert (grouped.count, grouped.index.start) == (7, 1)
    assert isinstance(grouped.evidence, EqualPrice)
    assert price.layers == [*range(1, 8), 1]


def test_quantisation_scales_that_differ_without_costing_differently_still_group():
    """A signature difference is not a cost difference.

    Left in, the per-layer scale gives every block its own signature and the
    stack does not group at all. Left out, the stack groups and the grouping
    proves free -- which it is, because nothing prices the scale.
    """
    blocks = [_block("linear_attention", weight_scale=0.5 + n) for n in range(8)]

    bare = detect_repeats(blocks)
    assert len(bare.items) == 8
    assert not any(isinstance(item, Repeat) for item in bare.items)

    tree = detect_repeats(blocks, ignore_attrs={"weight_scale"})
    proposed = tree.items[0]
    assert isinstance(proposed, Repeat)
    assert proposed.evidence.signature.startswith("less(weight_scale)|")

    proved, refused = prove_grouping(tree, _Pricer(), ignore_attrs={"weight_scale"})
    assert refused == ()
    assert isinstance(proved, Repeat)
    assert proved.count == 8
    assert isinstance(proved.evidence, EqualPrice)


def test_a_hybrid_stack_groups_at_both_levels_and_every_key_resolves():
    """Three of one class then one of another, that period twenty times.

    The check that matters is the key list. Under two levels of binding a
    block's layer is the sum of the values bound above it plus its offset in
    the body that holds it, and the eighty keys the flat pass asked about are
    that sum, once per block, in order.
    """
    blocks = _hybrid_stack(20)
    price = _Pricer()

    proved, refused = prove_grouping(detect_repeats(blocks), price)

    assert refused == ()
    assert isinstance(proved, Repeat)
    assert proved.count == 20
    assert isinstance(proved.evidence, EqualPrice)
    inner = proved.body.items[0]
    assert isinstance(inner, Repeat)
    assert inner.count == 3
    assert isinstance(inner.evidence, EqualPrice)

    variant = lambda n: "full_attention" if n % 4 == 3 else "linear_attention"
    flat = [f"model.layers.{n}.{variant(n)}" for n in range(80)]
    # the inner repeat is proved first, at the first period; then the outer's
    # flat pass over all eighty blocks; then the outer's grouped pass, which
    # asks for one period's prices and reuses them nineteen times.
    assert price.layers == [0, 1, 2, 0, *range(80), 0, 3]
    assert price.keys[4:84] == flat


def test_a_refused_outer_repeat_is_taken_apart_completely():
    """One block of the eleventh period costs more, so the period does not repeat.

    The inner run of three was proved free at the first period and is given up
    anyway: its index is written relative to a period that no longer exists, so
    keeping it would price the later runs at the wrong layer.
    """
    blocks = _hybrid_stack(20)
    price = _Pricer(by_layer=lambda layer, s: s * 2 if layer == 43 else s)

    proved, refused = prove_grouping(detect_repeats(blocks), price)

    assert len(refused) == 1
    assert refused[0].count == 20
    assert refused[0].flat_seconds != refused[0].grouped_seconds
    assert isinstance(proved, Seq)
    assert len(proved.items) == 80
    assert not any(isinstance(item, Repeat) for item in proved.items)


# --- the equality criterion --------------------------------------------------


def test_the_grouped_total_folds_the_prices_and_does_not_multiply_them():
    """Six instances of a tenth of a second, which is where the two differ.

    Folded from zero the six come to 0.6 and multiplied they come to
    0.6000000000000001. Both are arithmetic on the same six numbers and neither
    is wrong; they differ because addition is not associative. The grouped form
    folds, so the two forms are one number and no difference of association can
    be mistaken for an instance that costs something else.
    """
    key = ContextRef("model.layers.{layer}")
    blocks = [_op("aiter::tenth", context_ref=key) for _ in range(6)]

    proved, refused = prove_grouping(detect_repeats(blocks), _Pricer())

    assert refused == ()
    assert proved.evidence.flat_seconds == 0.6
    assert proved.evidence.grouped_seconds == proved.evidence.flat_seconds
    assert 6 * 0.1 == 0.6000000000000001
    assert proved.evidence.grouped_seconds != 6 * 0.1


def test_one_bit_of_difference_at_one_instance_refuses_the_grouping():
    """The smallest difference there is, absorbed by the total, still refused.

    One layer is priced one bit above the others. Added into a running total
    five orders of magnitude larger the bit disappears, so the two forms total
    to the same number -- and the grouping is refused anyway, because the
    prices themselves differ and this total is not the only sum they will enter.
    A check that compared totals alone would accept this, and would be
    accepting a body that is not the body it stands for.
    """
    blocks = _dense_stack(8)
    moved = lambda layer, s: math.nextafter(s, 1.0) if layer == 5 else s

    _, refused = prove_grouping(detect_repeats(blocks), _Pricer(by_layer=moved))

    assert len(refused) == 1
    assert refused[0].flat_seconds == refused[0].grouped_seconds
    assert "prices it at 3.0000000000000005e-06 s" in refused[0].reason


def test_a_price_read_from_the_shapes_alone_agrees_where_the_keyed_price_does_not():
    """Why the pricing function is handed the key and not just the operator.

    Both runs are the same stack and the same proposal. The keyed price sees a
    layer that costs twice as much and refuses; a price that never reads the
    key compares a number with itself, and would report every grouping ever
    proposed as free.
    """
    tree = detect_repeats(_dense_stack(8))
    keyed = _Pricer(by_layer=lambda layer, s: s * 2 if layer == 5 else s)
    blind = _Pricer()

    _, refused_keyed = prove_grouping(tree, keyed)
    _, refused_blind = prove_grouping(tree, blind)

    assert len(refused_keyed) == 1
    assert refused_blind == ()
    assert blind.layers == keyed.layers


# --- the evidence, recomputed ------------------------------------------------


def test_the_structural_evidence_is_recomputed_from_the_body_beside_it():
    """A repeat carrying a signature that is not its body's does not survive.

    Signing a body reads nothing but the body, so the string can be taken
    again wherever the record is read. A field that is never recomputed is a
    field that records whatever was written into it.
    """
    body = _block()
    proposal = Seq(
        (
            Repeat(
                body=body,
                count=4,
                index=IndexBinding("layer"),
                evidence=IdenticalStructure("seq(op|aiter::something_else)"),
            ),
        )
    )
    price = _Pricer()

    proved, refused = prove_grouping(proposal, price)

    assert len(refused) == 1
    assert "signs as" in refused[0].reason
    assert signature_of(body) in refused[0].reason
    assert price.asked == []
    assert isinstance(proved, Seq)
    assert len(proved.items) == 4


def test_evidence_taken_without_an_attribute_is_refused_when_taken_with_it():
    """The recomputation uses what it was told to leave out, and says so.

    The signature names the attributes it was taken without, so asking for it
    again with a different set is a mismatch rather than a silent pass.
    """
    blocks = [_block("linear_attention", weight_scale=0.5 + n) for n in range(8)]
    tree = detect_repeats(blocks, ignore_attrs={"weight_scale"})

    _, refused = prove_grouping(tree, _Pricer())

    assert len(refused) == 1
    assert "less(weight_scale)|" in refused[0].reason


# --- where an instance sits --------------------------------------------------


def test_a_block_in_a_nested_period_resolves_to_the_sum_and_not_to_one_binding():
    """The tenth block of the hybrid stack is layer 9, and it is a sum.

    It sits in the third period, second instance of the inner run: the outer
    repeat binds 8 for that period and the inner binds 1 for that instance.
    A resolver that substituted the binding nearest the block would write
    layer 1, and one that substituted the outermost would write layer 8. Both
    name a block that exists somewhere else in the stack, so neither would
    raise and both would price against another layer's ambient state.

    The offset a period advances by is its count times its step, which is what
    lets the two levels add up rather than overlap; the names differ per level
    so that neither binding shadows the other.
    """
    blocks = _hybrid_stack(20)
    tree = detect_repeats(blocks)
    outer = tree.items[0]
    inner = outer.body.items[0]

    assert outer.index.name != inner.index.name
    assert outer.index.step == inner.count * inner.index.step + 1
    assert (outer.index.value_at(2), inner.index.value_at(1)) == (8, 1)

    price = _Pricer()
    prove_grouping(tree, price)

    assert price.keys[4 + 9] == "model.layers.9.linear_attention"
    assert price.keys[4 + 8] == "model.layers.8.linear_attention"
    assert price.keys[4 + 11] == "model.layers.11.full_attention"


def test_a_repeat_bound_somewhere_other_than_where_it_sits_is_not_priced():
    """An index that does not agree with the position is refused, not resolved.

    Pricing it would substitute 7 into a key for the block at 0 and read some
    other layer's state, which is a wrong price that looks like a right one.
    Handed the offset the binding was written for, the same repeat proves free.
    """
    body = _block()
    proposal = Seq(
        (
            Repeat(
                body=body,
                count=4,
                index=IndexBinding("layer", start=7),
                evidence=IdenticalStructure(signature_of(body)),
            ),
        )
    )
    price = _Pricer()

    _, refused = prove_grouping(proposal, price)
    assert len(refused) == 1
    assert "binds 7" in refused[0].reason
    assert refused[0].flat_seconds is None
    assert price.asked == []

    proved, refused = prove_grouping(proposal, price, index_start=7)
    assert refused == ()
    assert isinstance(proved, Repeat)
    assert price.layers == [7, 8, 9, 10, 7]


def test_a_key_naming_two_indices_is_refused_rather_than_guessed():
    """A block sits at one position and cannot say which of two indices it is."""
    key = ContextRef("model.layers.{layer}.experts.{expert}")
    blocks = [_op("aiter::tenth", context_ref=key) for _ in range(4)]

    _, refused = prove_grouping(detect_repeats(blocks), _Pricer())

    assert len(refused) == 1
    assert "cannot say which" in refused[0].reason


def test_a_body_holding_an_overlap_is_not_grouped():
    """Overlapping branches combine by a join policy, so their costs do not add."""
    blocks = [
        Seq(
            (
                _op("aiter::rmsnorm"),
                Par((_op("aiter::gemm_a16w16"), _op("aiter::rmsnorm")), JoinPolicy.MAX),
            )
        )
        for _ in range(4)
    ]

    _, refused = prove_grouping(detect_repeats(blocks), _Pricer())

    assert len(refused) == 1
    assert "join policy" in refused[0].reason


# --- what it refuses to be handed, and what it never reads -------------------


def test_what_prove_grouping_refuses_to_be_handed():
    with pytest.raises(TypeError, match="what the detector returned"):
        prove_grouping([_block()], _Pricer())
    with pytest.raises(TypeError, match="position of the first block"):
        prove_grouping(detect_repeats(_dense_stack(4)), _Pricer(), index_start=True)


def test_nothing_in_the_proof_reads_a_symbolic_dimension():
    """The stand-in is armed, and the whole proof runs over it.

    Every shape in this file carries it, so this is a statement about all of
    the above as much as about the run below: comparing two regions, keying
    anything by one, or rebuilding an operator to carry a resolved key would
    each reach it and fail.
    """
    with pytest.raises(AssertionError, match="hashed"):
        hash(TOKENS)
    with pytest.raises(AssertionError, match="compared or converted"):
        TOKENS.__eq__(TOKENS)

    proved, refused = prove_grouping(detect_repeats(_hybrid_stack(5)), _Pricer())

    assert refused == ()
    assert isinstance(proved, Repeat)


def test_the_package_exports_the_rule_and_its_refusal():
    assert ir.prove_grouping is prove_grouping
    assert ir.Ungrouped is Ungrouped
    assert {"prove_grouping", "Ungrouped"} <= set(ir.__all__)
