# SPDX-License-Identifier: MIT
"""`atom.compass.ir`: the cost graph's data model.

The structure exists because a flat operator list fails: 2,999 nodes for a 27B
model, priced at ~39 ms per step against a 32.7 ms modelled step, answering only
for the one shape it was traced at. A tree fixes that only if three things hold,
and each of them is what a test here is defending.

* **A repeat body is any region.** Real models interleave two layer classes on a
  period and repeat the period, so the body has to be allowed to be a sequence
  or another repeat, at any depth. A type that permitted only a leaf body would
  flatten the nested form back to a linear sequence -- silently, and with the
  compression gone.
* **A repeat carries the index it varies over.** Otherwise a repeat of twenty
  asserts twenty identical bodies, and a body whose cost moves with the layer is
  priced as if it did not. The binding is mandatory, and a body that rebinds the
  same name is refused, because the inner binding would shadow the outer one and
  the node reading it would be priced at the wrong instance.
* **Grouping is a claim, not a default.** A repeat stands in for the sequence it
  replaces only if the two cost the same. Validating that is not this package's
  job; refusing to let the unvalidated version be built by accident is, so the
  evidence is a mandatory field and prices that disagree are refused.

Two more, both from what the operators themselves turned out to be. The tensors
that decide an opaque operator's cost are ambient rather than arguments, so
`attrs` refuses to hold a copy of one -- a copy is whatever the tracing forward
saw, and one such capture priced attention at 163.6 us against a true 23.0 us.
And a symbolic dimension is never compared or converted here, because either
installs a guard: the act of checking the record would change what the record
says about where it is valid.
"""

import ast
from pathlib import Path

import pytest

from atom.compass import ir
from atom.compass.ir import (
    AMBIENT_READINGS,
    Applicability,
    ContextRef,
    EqualPrice,
    Graph,
    IdenticalStructure,
    IndexBinding,
    JoinPolicy,
    NodeKind,
    Op,
    Par,
    Region,
    Repeat,
    Seq,
    as_shape,
    is_symbolic,
)

IR_PACKAGE = Path(__file__).resolve().parents[2] / "atom" / "compass" / "ir"

STRUCTURE = IdenticalStructure("linear_attention_block")


def _op(name="aiter::gemm_a16w16", **kwargs):
    kwargs.setdefault("kind", NodeKind.CAPTURED)
    kwargs.setdefault("in_shapes", ((4, 8),))
    kwargs.setdefault("out_shapes", ((4, 16),))
    return Op(name=name, **kwargs)


class _Tripwire:
    """A symbolic dimension that fails the moment anything compares it.

    Comparing a symbolic size, or calling `int()` on one, installs a guard --
    which is why this stands in for one here: any code path that did either
    would be mutating the artifact it was reading.
    """

    def __hash__(self):
        return 17

    def __repr__(self):
        return "s52"

    def _refuse(self, *_args):
        raise AssertionError("a symbolic dimension was compared or converted")

    __eq__ = __lt__ = __le__ = __gt__ = __ge__ = _refuse
    __int__ = __index__ = __bool__ = __float__ = _refuse


class _Everywhere(Applicability):
    """Stand-in for the statement the guard evaluator produces."""


# --- a repeat body is any region ---------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        _op(),
        Seq((_op("embed"), _op("norm"))),
        Repeat(body=_op(), count=3, index=IndexBinding("inner"), evidence=STRUCTURE),
        Par((_op("routed"), _op("shared")), JoinPolicy.MAX),
    ],
    ids=["op", "seq", "repeat", "par"],
)
def test_a_repeat_body_is_any_region(body):
    repeat = Repeat(
        body=body, count=20, index=IndexBinding("period"), evidence=STRUCTURE
    )
    assert repeat.body is body


def test_a_repeat_nests_without_a_depth_limit():
    region = _op()
    for depth in range(6):
        region = Repeat(
            body=region,
            count=2,
            index=IndexBinding(f"level_{depth}"),
            evidence=STRUCTURE,
        )
    assert len(region.bound_indices) == 6


def test_the_nested_non_contiguous_form_is_expressible():
    # Three of one layer class then one of another, that period repeated: the
    # shape a flat run-length encoder cannot produce, and the reason the body
    # has to be allowed to be a composite.
    period = Seq(
        (
            Repeat(
                body=_op("linear_attention"),
                count=3,
                index=IndexBinding("sub"),
                evidence=STRUCTURE,
            ),
            _op("full_attention"),
        )
    )
    model = Seq(
        (
            _op("embed"),
            _op("layer_0_dense"),
            Repeat(
                body=period,
                count=20,
                index=IndexBinding("period", step=4),
                evidence=STRUCTURE,
            ),
            _op("lm_head"),
        )
    )
    assert model.bound_indices == frozenset({"period", "sub"})


@pytest.mark.parametrize("body", [[_op()], (_op(),), "gemm", None, 3])
def test_a_repeat_body_that_is_not_a_region_is_refused(body):
    with pytest.raises(TypeError, match="repeat body is any region"):
        Repeat(body=body, count=2, index=IndexBinding("layer"), evidence=STRUCTURE)


# --- a repeat carries the index it varies over -------------------------------


def test_a_repeat_cannot_be_built_without_its_index():
    with pytest.raises(TypeError):
        Repeat(body=_op(), count=20, evidence=STRUCTURE)


@pytest.mark.parametrize("index", ["layer", 0, None])
def test_an_index_that_is_not_a_binding_is_refused(index):
    with pytest.raises(TypeError, match="index it varies its body over"):
        Repeat(body=_op(), count=20, index=index, evidence=STRUCTURE)


def test_an_inner_repeat_may_not_rebind_the_outer_index():
    inner = Repeat(body=_op(), count=3, index=IndexBinding("layer"), evidence=STRUCTURE)
    with pytest.raises(ValueError, match="'layer' is already bound"):
        Repeat(
            body=Seq((inner, _op("mlp"))),
            count=20,
            index=IndexBinding("layer"),
            evidence=STRUCTURE,
        )


def test_each_instance_binds_its_own_index_value():
    outer = Repeat(
        body=_op(), count=4, index=IndexBinding("layer", step=4), evidence=STRUCTURE
    )
    assert outer.index_values() == (0, 4, 8, 12)
    assert IndexBinding("layer", start=1).value_at(5) == 6


def test_a_body_reaches_its_instance_through_the_context_key():
    ref = ContextRef("model.layers.{layer}.self_attn")
    assert ref.index_names == ("layer",)
    assert str(ref.bind(layer=7)) == "model.layers.7.self_attn"
    with pytest.raises(KeyError, match="missing"):
        ref.bind()
    with pytest.raises(KeyError, match="not used"):
        ref.bind(layer=1, period=2)


def test_an_index_that_never_varies_is_refused():
    with pytest.raises(ValueError, match="step of zero"):
        IndexBinding("layer", step=0)


@pytest.mark.parametrize("count", [1, 0, -3])
def test_a_repeat_of_fewer_than_two_is_refused(count):
    with pytest.raises(ValueError, match="at least twice"):
        Repeat(body=_op(), count=count, index=IndexBinding("layer"), evidence=STRUCTURE)


# --- grouping is a claim, not a default --------------------------------------


def test_a_repeat_cannot_be_built_without_evidence():
    with pytest.raises(TypeError):
        Repeat(body=_op(), count=20, index=IndexBinding("layer"))


@pytest.mark.parametrize("evidence", ["checked", True, None, 0.0])
def test_evidence_that_names_no_comparison_is_refused(evidence):
    with pytest.raises(TypeError, match="what was compared"):
        Repeat(body=_op(), count=20, index=IndexBinding("layer"), evidence=evidence)


def test_prices_that_disagree_refuse_the_grouping():
    assert EqualPrice(2.5, 2.5).flat_seconds == 2.5
    with pytest.raises(ValueError, match="Grouping has to be free"):
        EqualPrice(flat_seconds=32.667e-3, grouped_seconds=28.360e-3)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_price_that_is_not_a_number_of_seconds_is_refused(bad):
    with pytest.raises(ValueError, match="not a number of seconds"):
        EqualPrice(bad, bad)


@pytest.mark.parametrize("signature", ["", "   "])
def test_an_empty_signature_compares_nothing(signature):
    with pytest.raises(ValueError, match="nothing was compared"):
        IdenticalStructure(signature)


# --- overlapping branches ----------------------------------------------------


def test_the_join_policy_has_exactly_three_values():
    # Taking the maximum is right only when the branches are small and the
    # device is not saturated. Two compute-heavy branches contend, and a branch
    # holding the whole device overlaps with nothing, so its cost adds.
    assert [str(policy) for policy in JoinPolicy] == [
        "max",
        "resource_bound",
        "exclusive",
    ]


def test_overlap_needs_two_branches():
    with pytest.raises(ValueError, match="at least two branches"):
        Par((_op(),), JoinPolicy.MAX)


@pytest.mark.parametrize("join", ["max", None, 0])
def test_a_parallel_region_states_how_its_branches_combine(join):
    with pytest.raises(TypeError, match="join must be a JoinPolicy"):
        Par((_op("a"), _op("b")), join)


def test_a_sequence_with_nothing_in_it_is_refused():
    with pytest.raises(ValueError, match="costs zero"):
        Seq(())


# --- shapes that stay symbolic -----------------------------------------------


def test_a_symbolic_dimension_is_never_compared_or_converted():
    tokens = _Tripwire()
    op = _op(in_shapes=((tokens, 4096),), out_shapes=((tokens, 11008),))
    assert op.in_shapes[0][0] is tokens
    assert is_symbolic(tokens) and not is_symbolic(4096)


def test_a_concrete_dimension_is_bounded():
    assert as_shape([0, 8]) == (0, 8)
    with pytest.raises(ValueError, match="cannot be negative"):
        as_shape([-1])


@pytest.mark.parametrize("bad", [True, 1.5, "8", None, b"8"])
def test_a_dimension_that_only_arrives_by_mistake_is_refused(bad):
    with pytest.raises(TypeError, match="concrete int or an object"):
        as_shape([bad])


def test_an_unhashable_symbolic_dimension_is_refused():
    with pytest.raises(TypeError, match="must be hashable"):
        as_shape([["s52"]])


def test_shapes_are_one_per_operand():
    op = _op(in_shapes=[[4, 8], [8, 16]], out_shapes=[[4, 16]])
    assert op.in_shapes == ((4, 8), (8, 16))
    assert op.out_shapes == ((4, 16),)


# --- cost is not a function of a node's arguments ----------------------------


@pytest.mark.parametrize("reading", sorted(AMBIENT_READINGS))
def test_a_per_step_ambient_reading_cannot_be_frozen_into_a_node(reading):
    # These change every step and are read through the context reference when a
    # price is asked for. Stored on the node they are whichever value the
    # tracing forward saw -- the failure that priced attention at 163.6 us
    # against a true 23.0 us, from a maximum sequence length left behind by a
    # warm-up run.
    with pytest.raises(ValueError, match="changes every step"):
        _op(kind=NodeKind.OPAQUE_LEAF, attrs={reading: 16384})


def test_two_nodes_differing_only_in_where_they_read_state_are_different_nodes():
    shared = {
        "kind": NodeKind.OPAQUE_LEAF,
        "in_shapes": ((4, 8),),
        "out_shapes": ((4, 8),),
    }
    first = Op(name="unified_attention", context_ref=ContextRef("layers.0"), **shared)
    second = Op(name="unified_attention", context_ref=ContextRef("layers.1"), **shared)
    assert first != second
    assert Op(name="unified_attention", **shared) != first


def test_attributes_do_not_depend_on_the_order_they_were_recorded_in():
    forwards = _op(attrs={"dtype": "bf16", "transpose_b": True})
    backwards = _op(attrs=(("transpose_b", True), ("dtype", "bf16")))
    assert forwards == backwards
    assert forwards.attrs == (("dtype", "bf16"), ("transpose_b", True))


def test_an_attribute_a_node_cannot_keep_is_refused():
    with pytest.raises(TypeError, match="must be hashable"):
        _op(attrs={"scales": [1.0, 2.0]})


def test_the_three_node_kinds_say_where_a_price_comes_from():
    assert [str(kind) for kind in NodeKind] == [
        "captured",
        "opaque_leaf",
        "declared",
    ]


def test_a_stream_id_is_carried_by_every_node():
    assert _op().stream_id == 0
    assert _op(stream_id=1).stream_id == 1
    with pytest.raises(TypeError, match="stream the operator runs on"):
        _op(stream_id="alt_stream")


# --- a graph states where it is valid ----------------------------------------


def test_a_graph_pairs_a_region_with_where_it_applies():
    region = Seq((_op("embed"), _op("lm_head")))
    graph = Graph(applicability=_Everywhere(), region=region)
    assert graph.region is region


@pytest.mark.parametrize("applicability", [None, "prefill", True])
def test_a_graph_without_a_validity_statement_is_refused(applicability):
    # `None` is not an empty predicate. It is a graph claiming to hold for every
    # step, which is exactly what the flat list did while holding for one.
    with pytest.raises(TypeError, match="states where its record is valid"):
        Graph(applicability=applicability, region=_op())


def test_a_graph_is_not_a_region():
    graph = Graph(applicability=_Everywhere(), region=_op())
    assert not isinstance(graph, Region)
    with pytest.raises(TypeError, match="holds regions"):
        Seq((graph,))


# --- the package reaches nothing ---------------------------------------------


def _ir_modules():
    return sorted(IR_PACKAGE.rglob("*.py"))


def test_the_package_was_found():
    assert _ir_modules(), f"no modules under {IR_PACKAGE}"


@pytest.mark.parametrize("module", _ir_modules(), ids=lambda p: p.name)
def test_the_package_imports_only_the_standard_library_it_names(module):
    # An allowlist, listing only what the package actually imports. The claim
    # being kept is that a recorded graph can be built and checked on a machine
    # with no device runtime and no symbolic-algebra library -- and that what a
    # symbolic dimension is stays the capture side's choice rather than
    # something decided here.
    allowed = {"abc", "collections", "dataclasses", "enum", "math", "string", "typing"}
    tree = ast.parse(module.read_text())
    roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.append((node.module or "").split(".")[0])
    strays = sorted({root for root in roots if root not in allowed})
    assert not strays, f"{module.name} imports {strays}; allowed: {sorted(allowed)}"


def test_everything_the_package_exports_is_reachable_by_name():
    for name in ir.__all__:
        assert getattr(ir, name, None) is not None, name
