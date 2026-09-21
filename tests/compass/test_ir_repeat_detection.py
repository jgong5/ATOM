# SPDX-License-Identifier: MIT
"""`atom.compass.ir.repeats`: naming the repetition in a flat block sequence.

A capture is flat, and a flat capture of an eighty-layer stack is eighty bodies
to price. The tree that prices two of them instead only exists if something
builds it, and a builder that handles only the easy stack is worth very little,
because real stacks are not the easy stack. Two ways they are not, and a test
here defends each.

* **The ends differ from the middle.** A dense first layer, a different
  attention variant on layer zero, a final norm folded into the last block. The
  prologue and the epilogue have to come out as plain siblings while the middle
  still groups.
* **The middle is not one class repeated.** Read as block classes, a hybrid
  stack is `AAABAAAB...`, and there are exactly two ways to get that wrong:
  eighty groups of one, which names no repetition at all, and twenty groups of
  four, whose body is four different things and therefore prices nothing that
  ran. The named result here is that eighty blocks produce neither.

Three more, each of which is the difference between a detector and a hash of a
module name:

* **The signature ignores the layer a block sits at and nothing else.** Blocks
  that differ in what they do, on what shapes, with what attributes or on which
  stream are different blocks and do not group.
* **The index bindings are carried and they mean something.** With nesting no
  one binding is a block's absolute position: the outer repeat steps by the
  period and the inner one by one, and the two add.
* **Nothing reads a symbolic dimension.** Comparing one, or converting one,
  installs a guard -- so reading the record would change what the record says
  about where it is valid. Region equality compares dimensions and region
  hashing hashes them, so "the detector never compares two blocks as regions"
  is the same claim. A dimension that refuses every way of being read stands in
  for a symbol here, including hashing outside the window in which a shape is
  built, and the whole detector runs over it.

The evidence every repeat carries is structural and never a price. Pricing the
flat form against the grouped one is the check that a grouping was free to take;
this module cannot price anything, so it claims only what it compared.
"""

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
    detect_repeats,
    signature_of,
)

_BUILDING = False


@contextmanager
def _building():
    """The one window in which a dimension may be hashed.

    Building a shape hashes every dimension handed to it, so the symbol below
    has to survive that much or nothing here could be constructed at all.
    Outside this window it refuses, which is what turns "the detector compares
    signature strings and never regions" into something a test can fail: asking
    whether two operators are equal compares their dimensions, putting one in a
    set or a dict hashes them, and either would be a guard installed by the act
    of reading the record.
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

    Every way of reading one raises: comparison, ordering, `int()`, `bool()`,
    `float()`, and hashing outside the construction window. A stand-in that
    answered any of them would differ from the thing it stands in for in
    exactly the property being tested -- which is also why its hash is the
    identity hash rather than a constant: a constant forces an equality
    comparison the moment two of them meet in one dict, and a test that passes
    because of that is testing the stand-in.

    It is a stand-in and not the real thing: a size that is not known yet
    arrives from the tracer as the toolchain's own type, and putting one of
    those into a shape is not possible today.
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


def _op(name="aiter::gemm_a16w16", **kwargs):
    kwargs.setdefault("kind", NodeKind.CAPTURED)
    kwargs.setdefault("in_shapes", ((TOKENS, 4096),))
    kwargs.setdefault("out_shapes", ((TOKENS, 4096),))
    with _building():
        return Op(name=name, **kwargs)


def _block(variant="linear_attention", **kwargs):
    """One layer as a tracer records it: its operators, and where each reads
    the ambient state that decides its cost.

    The key carries the repeat index as a placeholder rather than a number,
    which is what makes the same layer class at layer 3 and at layer 47 one
    signature without anything here stripping an index out of a name.
    """
    return Seq(
        (
            _op("aiter::rmsnorm"),
            _op(
                f"aiter::{variant}",
                kind=NodeKind.OPAQUE_LEAF,
                context_ref=ContextRef(f"model.layers.{{layer}}.{variant}"),
                **kwargs,
            ),
            _op("aiter::gemm_a16w16"),
        )
    )


def _hybrid_stack(periods=20):
    """`AAAB` repeated: three of one layer class then one of another.

    Every block is built fresh, so what groups them is their signature and not
    that the test handed the same object in eighty times.
    """
    blocks = []
    for _ in range(periods):
        blocks += [_block("linear_attention") for _ in range(3)]
        blocks.append(_block("full_attention"))
    return blocks


def _expand(region, blocks):
    """The blocks `region` stands for, in the order they run.

    A group keeps one body and a count, so the instances after the first are
    the same object rather than the objects handed in; what the grouped form
    claims is that they are interchangeable, which is a claim about signatures.
    Membership is by identity all the same, because asking whether two regions
    are equal would compare their shapes, and comparing a symbolic shape is the
    thing these tests exist to catch.
    """
    known = {id(block) for block in blocks}
    if id(region) in known:
        return [region]
    if isinstance(region, Repeat):
        return _expand(region.body, blocks) * region.count
    if isinstance(region, Seq):
        return [b for item in region.items for b in _expand(item, blocks)]
    return [region]


def _signatures(regions):
    return [signature_of(region) for region in regions]


# --- the named result: eighty blocks of three-then-one -----------------------


def test_eighty_blocks_of_three_then_one_produce_the_nested_form():
    blocks = _hybrid_stack()
    region = detect_repeats(blocks)

    (outer,) = region.items
    assert isinstance(outer, Repeat)
    assert outer.count == 20

    inner, tail = outer.body.items
    assert isinstance(inner, Repeat)
    assert inner.count == 3
    assert signature_of(inner.body) == signature_of(blocks[0])
    assert signature_of(tail) == signature_of(blocks[3])


def test_the_nested_form_is_neither_flat_encoding():
    # The two a run-length scan over single symbols produces, and why neither
    # is usable: eighty groups of one names no repetition, and twenty groups of
    # four names a body that is four different things, so pricing it once and
    # multiplying prices nothing that ran.
    blocks = _hybrid_stack()
    region = detect_repeats(blocks)

    assert len(region.items) == 1, "eighty singleton siblings is the first"
    outer = region.items[0]
    assert len(outer.body.items) == 2, "a four-block flat body is the second"
    assert any(isinstance(item, Repeat) for item in outer.body.items)


def test_the_nested_form_stands_for_the_blocks_it_was_built_from():
    blocks = _hybrid_stack()
    region = detect_repeats(blocks)
    expanded = _expand(region, blocks)
    assert len(expanded) == len(blocks)
    assert _signatures(expanded) == _signatures(blocks)
    assert expanded[0] is blocks[0]


def test_the_blocks_come_back_as_the_objects_that_were_handed_in():
    blocks = _hybrid_stack(periods=3)
    before = list(blocks)
    detect_repeats(blocks)
    assert len(blocks) == len(before)
    assert all(a is b for a, b in zip(blocks, before))


# --- way one: the ends differ from the middle --------------------------------


def _stack_with_ends():
    """Embedding, a dense first layer, the hybrid middle, a fused last layer,
    a final norm and the head."""
    return (
        [_block("embedding"), _block("dense_mlp")]
        + _hybrid_stack()
        + [_block("fused_norm_attention"), _block("norm"), _block("lm_head")]
    )


def test_a_prologue_and_an_epilogue_stay_as_siblings():
    blocks = _stack_with_ends()
    region = detect_repeats(blocks)

    kinds = [type(item) for item in region.items]
    assert kinds == [Seq, Seq, Repeat, Seq, Seq, Seq]
    assert region.items[0] is blocks[0]
    assert region.items[1] is blocks[1]
    assert region.items[-1] is blocks[-1]


def test_the_middle_still_groups_with_the_ends_around_it():
    blocks = _stack_with_ends()
    region = detect_repeats(blocks)
    outer = region.items[2]
    assert outer.count == 20
    assert isinstance(outer.body.items[0], Repeat)


def test_the_whole_stack_expands_back_in_order():
    blocks = _stack_with_ends()
    expanded = _expand(detect_repeats(blocks), blocks)
    assert len(expanded) == len(blocks)
    assert _signatures(expanded) == _signatures(blocks)


def test_a_single_instance_is_a_sibling_and_never_a_group_of_one():
    blocks = [_block("embedding"), _block("dense_mlp"), _block("lm_head")]
    region = detect_repeats(blocks)
    assert [item is block for item, block in zip(region.items, blocks)] == [True] * 3


# --- the contiguous stack, and deeper nesting --------------------------------


def test_a_dense_stack_is_one_repeat_and_is_not_nested():
    blocks = [_block("full_attention") for _ in range(64)]
    (repeat,) = detect_repeats(blocks).items
    assert repeat.count == 64
    assert repeat.body is blocks[0]


def test_a_run_of_six_is_six_of_one_and_not_three_of_a_pair():
    blocks = [_block("full_attention") for _ in range(6)]
    (repeat,) = detect_repeats(blocks).items
    assert repeat.count == 6


def test_a_period_that_itself_repeats_a_period_nests_three_deep():
    # ((A A B) x3  C) x4: the pass re-encodes what the previous pass produced,
    # so depth follows the model rather than a limit written here.
    blocks = []
    for _ in range(4):
        for _ in range(3):
            blocks += [_block("linear_attention"), _block("linear_attention")]
            blocks.append(_block("full_attention"))
        blocks.append(_block("dense_mlp"))

    (outermost,) = detect_repeats(blocks).items
    assert outermost.count == 4
    middle, _tail = outermost.body.items
    assert middle.count == 3
    innermost, _b = middle.body.items
    assert innermost.count == 2
    assert len(_expand(detect_repeats(blocks), blocks)) == len(blocks)


def test_two_stacks_of_the_same_shape_group_the_same_way():
    blocks = _hybrid_stack(periods=4)
    first = detect_repeats(blocks)
    second = detect_repeats(list(blocks))
    assert signature_of(first) == signature_of(second)


# --- the signature: what it ignores, and what it must not --------------------


def test_the_layer_a_block_sits_at_is_not_part_of_its_signature():
    # Nothing here strips an index: the traced key names the index as a
    # placeholder, so the two are already the same text.
    at_three = _block("full_attention")
    at_forty_seven = _block("full_attention")
    assert signature_of(at_three) == signature_of(at_forty_seven)
    assert "{layer}" in signature_of(at_three)


def test_a_key_that_names_one_layer_rather_than_the_index_does_not_group():
    # The safe direction: a capture that wrote a number where the placeholder
    # belongs loses the compression and says nothing untrue.
    blocks = [
        Seq((_op("aiter::attn", context_ref=ContextRef(f"model.layers.{i}.attn")),))
        for i in range(8)
    ]
    region = detect_repeats(blocks)
    assert len(region.items) == 8
    assert not any(isinstance(item, Repeat) for item in region.items)


@pytest.mark.parametrize(
    "changed",
    [
        {"name": "aiter::v4_mqa"},
        {"kind": NodeKind.DECLARED},
        {"in_shapes": ((512, 8192),)},
        {"out_shapes": ((512, 8192),)},
        {"attrs": (("dtype", "fp8"),)},
        {"stream_id": 1},
        {"context_ref": ContextRef("model.layers.{layer}.mlp")},
    ],
    ids=["name", "kind", "in_shapes", "out_shapes", "attrs", "stream", "context"],
)
def test_blocks_that_genuinely_differ_have_different_signatures(changed):
    plain = _op("aiter::attn", context_ref=ContextRef("model.layers.{layer}.attn"))
    other = _op(
        **{
            "name": "aiter::attn",
            "context_ref": ContextRef("model.layers.{layer}.attn"),
            **changed,
        }
    )
    assert signature_of(plain) != signature_of(other)


def test_blocks_that_genuinely_differ_do_not_group():
    blocks = [_block("linear_attention"), _block("full_attention")] * 4
    region = detect_repeats(blocks)
    (repeat,) = region.items
    assert repeat.count == 4
    assert len(repeat.body.items) == 2


def test_the_shape_of_a_sequence_is_part_of_its_signature():
    two = Seq((_op("a"), _op("b")))
    nested = Seq((_op("a"), Seq((_op("b"),))))
    assert signature_of(two) != signature_of(nested)


def test_an_overlap_signs_its_branches_and_how_they_combine():
    branches = (_op("routed_expert"), _op("shared_expert"))
    assert signature_of(Par(branches, JoinPolicy.MAX)) != signature_of(
        Par(branches, JoinPolicy.EXCLUSIVE)
    )
    assert signature_of(Par(branches, JoinPolicy.MAX)) == signature_of(
        Par(branches, JoinPolicy.MAX)
    )


def test_a_repeat_signs_its_body_and_count_and_not_where_it_sits():
    # Two runs of one body are the same run wherever they are, which is what
    # lets the second pass see a period made of a run and a block.
    body = _op("aiter::attn")
    here = Repeat(body, 3, IndexBinding("layer", start=0), IdenticalStructure("a"))
    there = Repeat(body, 3, IndexBinding("other", start=40), IdenticalStructure("b"))
    assert signature_of(here) == signature_of(there)
    assert signature_of(here) != signature_of(
        Repeat(body, 4, IndexBinding("layer"), IdenticalStructure("a"))
    )


def test_a_per_layer_constant_that_costs_nothing_can_be_left_out():
    # A quantisation scale differs at every layer and does not differ in cost.
    # A signature that keeps it splits a group that was valid; the caller is
    # the one who knows, so the caller says so.
    blocks = [
        Seq((_op("aiter::gemm_a16w16", attrs=(("weight_scale", i), ("dtype", "fp8"))),))
        for i in range(8)
    ]
    assert len(detect_repeats(blocks).items) == 8
    (repeat,) = detect_repeats(blocks, ignore_attrs=("weight_scale",)).items
    assert repeat.count == 8
    assert "dtype" in signature_of(blocks[0], ignore_attrs=("weight_scale",))


def test_a_signature_is_taken_of_a_region():
    with pytest.raises(TypeError, match="signature is taken of a region"):
        signature_of("model.layers.0")


@pytest.mark.parametrize("bad", ["weight_scale", 7], ids=["str", "int"])
def test_attributes_to_leave_out_are_named_one_by_one(bad):
    with pytest.raises(TypeError):
        signature_of(_op(), ignore_attrs=bad)


# --- the index bindings -------------------------------------------------------


def test_the_outer_repeat_steps_by_the_period_and_the_inner_one_by_one():
    blocks = _hybrid_stack()
    (outer,) = detect_repeats(blocks).items
    inner = outer.body.items[0]

    assert (outer.index.start, outer.index.step) == (0, 4)
    assert outer.index_values() == tuple(range(0, 80, 4))
    assert (inner.index.start, inner.index.step) == (0, 1)
    assert inner.index_values() == (0, 1, 2)


def test_a_blocks_position_is_the_sum_of_what_the_repeats_above_it_bind():
    # The third instance of the period, second block inside it: 8 + 1 = 9.
    blocks = _hybrid_stack()
    (outer,) = detect_repeats(blocks).items
    inner = outer.body.items[0]
    assert outer.index.value_at(2) + inner.index.value_at(1) == 9
    # The block after the run is not in it, so its position is the period's
    # index plus the fixed offset it sits at.
    assert outer.index.value_at(2) + 3 == 11


def test_the_index_counts_positions_in_what_the_caller_handed_over():
    blocks = _stack_with_ends()
    outer = detect_repeats(blocks).items[2]
    assert outer.index.start == 2, "two blocks come before the stack"
    assert outer.index_values()[-1] == 2 + 76


def test_the_first_block_can_be_told_which_index_it_is():
    blocks = _hybrid_stack(periods=4)
    (outer,) = detect_repeats(blocks, index_start=100).items
    assert outer.index_values() == (100, 104, 108, 112)


def test_each_level_binds_its_own_name():
    blocks = _hybrid_stack()
    (outer,) = detect_repeats(blocks, index_name="depth").items
    inner = outer.body.items[0]
    assert outer.index.name == "depth"
    assert inner.index.name == "depth_1"
    assert outer.index.name not in inner.bound_indices


def test_a_body_that_already_binds_the_name_is_not_shadowed():
    inner = Repeat(
        body=_op("aiter::attn"),
        count=2,
        index=IndexBinding("layer"),
        evidence=IdenticalStructure("attn"),
    )
    blocks = [inner, inner, inner]
    (outer,) = detect_repeats(blocks).items
    assert outer.count == 3
    assert outer.index.name != "layer"
    assert "layer" in outer.body.bound_indices


@pytest.mark.parametrize("bad", ["not an identifier", "", 7], ids=["spaces", "", "int"])
def test_an_index_name_that_cannot_be_a_placeholder_is_refused(bad):
    with pytest.raises(ValueError, match="identifier"):
        detect_repeats([_op()], index_name=bad)


@pytest.mark.parametrize("bad", [1.0, "4", True], ids=["float", "str", "bool"])
def test_an_index_start_that_is_not_an_int_is_refused(bad):
    with pytest.raises(TypeError, match="index_start"):
        detect_repeats([_op()], index_start=bad)


# --- the evidence -------------------------------------------------------------


def test_every_repeat_carries_the_signature_its_instances_share():
    blocks = _hybrid_stack()
    (outer,) = detect_repeats(blocks).items
    inner = outer.body.items[0]
    for repeat in (outer, inner):
        assert isinstance(repeat.evidence, IdenticalStructure)
        assert repeat.evidence.signature == signature_of(repeat.body)


def test_the_evidence_can_be_taken_again_from_the_body_beside_it():
    # A signature a reader cannot reproduce is a field it cannot check. What a
    # signature was taken without is part of the string, so the body is enough
    # to take it again.
    blocks = [
        Seq((_op("aiter::gemm_a16w16", attrs=(("weight_scale", i),)),))
        for i in range(6)
    ]
    (repeat,) = detect_repeats(blocks, ignore_attrs=("weight_scale",)).items
    assert repeat.evidence.signature.startswith("less(weight_scale)|")
    assert repeat.evidence.signature == signature_of(
        repeat.body, ignore_attrs=("weight_scale",)
    )
    assert repeat.evidence.signature != signature_of(repeat.body)


def test_no_repeat_claims_a_price_was_compared():
    # There is no price here to compare. Structural identity is what was
    # checked, so structural identity is what is claimed; whether the grouping
    # is free is decided elsewhere, by pricing both forms.
    blocks = _stack_with_ends()
    found = []

    def walk(region):
        if isinstance(region, Repeat):
            found.append(region.evidence)
            walk(region.body)
        elif isinstance(region, Seq):
            for item in region.items:
                walk(item)

    walk(detect_repeats(blocks))
    assert found
    assert not any(isinstance(evidence, EqualPrice) for evidence in found)


# --- nothing compares a symbolic dimension ------------------------------------


@pytest.mark.parametrize(
    "read",
    [
        lambda a, b: a == b,
        lambda a, b: a < b,
        lambda a, b: hash(a),
        lambda a, b: int(a),
        lambda a, b: bool(a),
        lambda a, b: float(a),
        lambda a, b: [a][0:1] == [b],
        lambda a, b: {a, b},
        lambda a, b: {a: 1}[b],
    ],
    ids=["eq", "lt", "hash", "int", "bool", "float", "in_list", "set", "dict"],
)
def test_the_stand_in_refuses_every_way_of_being_read(read):
    # The stand-in is the whole of the proof below, so it is itself under test:
    # one that answered any of these would pass the tests that follow without
    # them meaning anything.
    with pytest.raises(AssertionError):
        read(_Symbol(), _Symbol())


def test_the_whole_detector_runs_over_a_dimension_that_refuses_comparison():
    # Every block above already carries one; this states it as the claim rather
    # than leaving it implicit in the fixtures.
    blocks = _hybrid_stack()
    assert any(
        any(dim is TOKENS for dim in shape) for shape in blocks[0].items[0].in_shapes
    )
    (outer,) = detect_repeats(blocks).items
    assert outer.count == 20


def test_a_symbolic_dimension_reaches_the_signature_without_being_read():
    text = signature_of(_op(in_shapes=((TOKENS, 4096),), out_shapes=((TOKENS,),)))
    assert "s52" in text
    assert "4096" in text


def test_a_symbol_that_prints_as_a_number_does_not_read_as_that_number():
    class _PrintsAsFour(_Symbol):
        def __repr__(self):
            return "4"

    assert signature_of(_op(in_shapes=((_PrintsAsFour(),),))) != signature_of(
        _op(in_shapes=((4,),))
    )


# --- what the caller passed ---------------------------------------------------


def test_a_forward_pass_with_no_blocks_is_refused():
    with pytest.raises(ValueError, match="no blocks"):
        detect_repeats([])


def test_one_region_on_its_own_is_not_a_sequence_of_blocks():
    with pytest.raises(TypeError, match="sequence of regions"):
        detect_repeats(_op())


@pytest.mark.parametrize("bad", [7, "layer", None], ids=["int", "str", "none"])
def test_a_block_that_is_not_a_region_is_refused(bad):
    with pytest.raises(TypeError, match="a block is a region"):
        detect_repeats([_op(), bad])


def test_something_that_is_not_a_sequence_at_all_is_refused():
    with pytest.raises(TypeError, match="sequence of regions"):
        detect_repeats(7)


def test_one_block_is_a_sequence_of_one():
    block = _block("lm_head")
    region = detect_repeats([block])
    assert len(region.items) == 1
    assert region.items[0] is block


# --- the package ---------------------------------------------------------------


def test_the_detector_is_reachable_from_the_package():
    assert ir.detect_repeats is detect_repeats
    assert ir.signature_of is signature_of
    assert {"detect_repeats", "signature_of"} <= set(ir.__all__)
