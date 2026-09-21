# SPDX-License-Identifier: MIT
"""Finding the repetition in a flat sequence of blocks.

A capture arrives flat: the blocks of one forward pass in the order they ran,
one entry per layer plus whatever sits either side of the stack. Priced that
way an eighty-layer stack is eighty bodies. The tree that prices two of them
instead is only worth having if something builds it, and building it is the
whole of what this module does -- it reads a flat sequence and returns the same
sequence with its repetition named. It does not price anything, and it does not
decide whether a grouping was free to take.

Real stacks break a naive scan in two ways, and a detector that handles only the
first gives up the compression on every model that is not a plain dense decoder.

**The ends differ from the middle.** A dense first layer in an otherwise sparse
stack, a different attention variant on layer zero, a final norm folded into the
last block. A scan that expects one class from end to end finds nothing at all.

**The middle is not one class repeated.** A hybrid attention stack, a
mixture-of-experts schedule that fires every k layers, a shared-expert variant:
each interleaves two layer classes on a period and repeats the period. Read as a
string of block classes that is `AAABAAAB...`, and a run-length scan over single
symbols produces either eighty groups of one or twenty groups of four. Neither
is wrong about the order of the blocks and both are useless: the first names no
repetition, and the second names a body that is four different things, so
pricing it once and multiplying prices nothing that ran.

Both fall out of one algorithm, run bottom-up over a canonical block signature
rather than over a module name:

1. Give every block a signature -- what it does, on what shapes, with the layer
   it sits at left out. Two blocks with one signature are interchangeable for
   pricing.
2. Over the signature string, find the stretch from each position that is one
   period repeated. A period of one symbol is a contiguous run; a longer period
   is the interleaved case. The ends of the stack are periods that repeat once,
   which is to say they are not periods, and they stay as plain siblings -- so
   the first way needs no special case.
3. Do the same inside every period found, and repeat the whole pass until one
   changes nothing. `AAABAAAB...` becomes twenty of `AAAB`, then twenty of
   `A` three times followed by `B`. Each pass either replaces two or more nodes
   with one or leaves the level alone, so the passes stop.

**The signature is a string and not a digest.** A digest would be shorter and
would hide the one thing worth reading when a group that should have formed did
not: which part of two blocks differed. Equality is all the algorithm asks of a
signature, and two strings compare as cheaply as two digests at these sizes.

**Nothing here reads a symbolic dimension.** Comparing one or converting one
installs a guard, which would make reading a record change what the record says
about where it is valid. That rules out far more than the obvious: a region is a
value whose generated equality compares its shapes one dimension at a time, so
`a == b`, `a in [b]`, `{a, b}` and `{a: 1}[b]` all reach a comparison on a
dimension, and hashing a region hashes its dimensions too. Two blocks are
therefore never compared as regions here. The signature is built by reading the
fields of the leaf operators directly -- names, kinds, attributes, streams,
context keys -- and a dimension enters it as text and nothing else; the
algorithm then compares signature strings, and keys nothing by a region, a
shape or a dimension.

A dimension's text is the canonical form the shapes module settles on, which
today is its `repr`. That is the one thing in this module that depends on what a
symbolic dimension turns out to be, and it is the one line to change if that
answer changes.

**What the index bindings mean.** A repeat carries the index it varies its body
over, and with nesting no single binding can be the absolute position of a
block: the outer repeat steps by the length of its period and the inner one by
one, so a block's position is the sum of the values bound by the repeats above
it, plus its own fixed offset in the body that holds it. The outermost repeat
carries the name the caller asked for and each level inside it carries that name
with its depth, so a body already binding a name is never shadowed.

**What a signature difference costs.** Two blocks that differ only in something
pricing ignores -- a per-layer quantisation scale is the standing example -- get
different signatures and fail to group. That costs compression and never
correctness, which is the right way round, and `ignore_attrs` is the way to say
that a named attribute is not part of what a block is.
"""

from dataclasses import dataclass

from .nodes import (
    IdenticalStructure,
    IndexBinding,
    Op,
    Par,
    Region,
    Repeat,
    Seq,
)
from .shapes import is_symbolic


def signature_of(region: Region, *, ignore_attrs: object = ()) -> str:
    """What `region` does, canonically, with the layer it sits at left out.

    Two regions with one signature are interchangeable for pricing. The layer
    index is left out for free rather than stripped: a captured body names the
    module it reads its ambient state from with the repeat index as a
    placeholder, `model.layers.{layer}.self_attn`, so the key is already the
    same text at every layer. A capture that wrote a concrete layer into that
    key instead gives every block its own signature and the stack does not
    group -- less compression, and nothing said that was not true.

    A repeat contributes its count and its body and not the index it varies
    over, for the same reason: two runs of one body are the same run wherever
    they sit.
    """
    ignored = _as_ignored(ignore_attrs)
    return _prefix(ignored) + _signature(region, ignored)


def detect_repeats(
    blocks: object,
    *,
    index_name: str = "layer",
    index_start: int = 0,
    ignore_attrs: object = (),
) -> Seq:
    """The blocks of one forward pass, with their repetition named.

    Returns a sequence holding the same blocks in the same order, with every
    stretch that is one period repeated two or more times replaced by a repeat
    over that period, nested where the period itself repeats something.

    The index a repeat binds counts positions in `blocks`, offset by
    `index_start`: hand over the whole forward pass and the repeats are written
    in terms of positions in it; hand over just the layer stack and they are
    written in terms of layer numbers.

    Every repeat is emitted with structural evidence -- the signature its
    instances share -- and never with a price. Whether a grouping is free is
    decided by pricing the two forms, which this module cannot do and does not
    claim to have done.
    """
    blocks = _as_blocks(blocks)
    if not isinstance(index_name, str) or not index_name.isidentifier():
        raise ValueError(
            "the index name is what a repeat binds and a body names it as a "
            f"placeholder, so it has to be an identifier, got {index_name!r}"
        )
    if isinstance(index_start, bool) or not isinstance(index_start, int):
        raise TypeError(
            "index_start is the index of the first block, as an int, got "
            f"{type(index_start).__name__}"
        )
    ignored = _as_ignored(ignore_attrs)
    nodes = tuple(_Leaf(block, _signature(block, ignored)) for block in blocks)
    nodes = _encode_until_settled(nodes)
    return Seq(tuple(_materialise(nodes, index_start, 0, index_name, ignored)))


# --- the signature -----------------------------------------------------------


def _prefix(ignored: frozenset) -> str:
    """What a signature was taken without, so that it can be taken again.

    A repeat carries the signature its instances share, and a reader that cannot
    reproduce that string from the body beside it is holding a field it cannot
    check. Leaving an attribute out changes the string, so the string says which
    ones were left out. Nothing is said when nothing was left out, which keeps
    the ordinary signature the ordinary text.
    """
    if not ignored:
        return ""
    return "less(" + ",".join(sorted(ignored)) + ")|"


def _signature(region: Region, ignored: frozenset) -> str:
    if isinstance(region, Op):
        return _op_signature(region, ignored)
    if isinstance(region, Seq):
        return _seq_signature([_signature(item, ignored) for item in region.items])
    if isinstance(region, Repeat):
        return f"repeat({region.count},{_signature(region.body, ignored)})"
    if isinstance(region, Par):
        branches = ",".join(_signature(branch, ignored) for branch in region.branches)
        return f"par({region.join},{branches})"
    raise TypeError(
        "a signature is taken of a region -- an operator, a sequence, a repeat "
        f"or an overlap -- got {type(region).__name__}"
    )


def _op_signature(op: Op, ignored: frozenset) -> str:
    attrs = ";".join(
        f"{key}={value!r}" for key, value in op.attrs if key not in ignored
    )
    return "|".join(
        (
            "op",
            op.name,
            str(op.kind),
            _shapes_text(op.in_shapes),
            _shapes_text(op.out_shapes),
            attrs,
            str(op.stream_id),
            "" if op.context_ref is None else op.context_ref.key,
        )
    )


def _shapes_text(shapes: tuple) -> str:
    return "/".join(",".join(_dim_text(dim) for dim in shape) for shape in shapes)


def _dim_text(dim: object) -> str:
    # `repr` and not `str(int(...))`, and not a comparison against anything: a
    # symbolic dimension that is compared or converted has a guard installed on
    # it, and the guard is part of the record of where the trace is valid. The
    # mark keeps a symbol that prints as a number from reading as that number.
    return f"?{dim!r}" if is_symbolic(dim) else repr(dim)


def _seq_signature(signatures: list) -> str:
    return "seq(" + ",".join(signatures) + ")"


def _body_signature(nodes: tuple) -> str:
    """The signature of the region a period materialises into."""
    if len(nodes) == 1:
        return nodes[0].signature
    return _seq_signature([node.signature for node in nodes])


def _evidence_signature(nodes: tuple, ignored: frozenset) -> str:
    """What a reader gets by signing the body this period materialises into."""
    return _prefix(ignored) + _body_signature(nodes)


# --- the encoding ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Leaf:
    """One block of the flat sequence, and what it does."""

    region: Region
    signature: str

    @property
    def span(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class _Run:
    """A stretch of the sequence that is one period repeated `count` times."""

    nodes: tuple
    count: int

    @property
    def period_span(self) -> int:
        """Blocks of the flat sequence one instance of the period covers."""
        return sum(node.span for node in self.nodes)

    @property
    def span(self) -> int:
        return self.count * self.period_span

    @property
    def signature(self) -> str:
        return f"repeat({self.count},{_body_signature(self.nodes)})"


def _encode_until_settled(nodes: tuple) -> tuple:
    """Encode, and encode the result, until a pass changes nothing.

    A productive pass replaces two or more nodes with one somewhere in the tree,
    so the node count falls and the passes stop. The count is the measure and
    the comparison is over signatures, which a pass changes exactly when it
    changed the structure.
    """
    settled = _level_signature(nodes)
    while True:
        nodes = _encode(nodes)
        signature = _level_signature(nodes)
        if signature == settled:
            return nodes
        settled = signature


def _level_signature(nodes: tuple) -> str:
    return ",".join(node.signature for node in nodes)


def _encode(nodes: tuple) -> tuple:
    """One pass: inside every period first, then across this level."""
    inner = tuple(
        _Run(_encode(node.nodes), node.count) if isinstance(node, _Run) else node
        for node in nodes
    )
    return _encode_level(inner)


def _encode_level(nodes: tuple) -> tuple:
    signatures = [node.signature for node in nodes]
    out = []
    at = 0
    while at < len(nodes):
        period, count = _longest_repetition(signatures, at)
        if count > 1:
            out.append(_Run(tuple(nodes[at : at + period]), count))
            at += period * count
        else:
            out.append(nodes[at])
            at += 1
    return tuple(out)


def _longest_repetition(signatures: list, at: int) -> tuple:
    """The period and count that cover most of `signatures` from `at`.

    Most covered rather than shortest period, so a period made of several
    classes is found in the same pass as a run of one; the shortest wins a tie,
    so six of one block is six repeats of it and not three of a pair. A count of
    one means nothing repeats here, and the node stays where it is.
    """
    best = (1, 1)
    for period in range(1, (len(signatures) - at) // 2 + 1):
        head = signatures[at : at + period]
        count = 1
        while signatures[at + period * count : at + period * (count + 1)] == head:
            count += 1
        if count > 1 and period * count > best[0] * best[1]:
            best = (period, count)
    return best


# --- the regions -------------------------------------------------------------


def _materialise(
    nodes: tuple, base: int, depth: int, index_name: str, ignored: frozenset
) -> list:
    """The regions for one level, with the index each repeat varies over.

    `base` is the index of the first block of this level. At the top that is
    where the caller said the sequence starts; inside a period it is zero,
    because the enclosing repeat already binds where the instance begins and the
    two values add.
    """
    regions = []
    offset = 0
    for node in nodes:
        if isinstance(node, _Leaf):
            regions.append(node.region)
        else:
            inner = _materialise(node.nodes, 0, depth + 1, index_name, ignored)
            body = inner[0] if len(inner) == 1 else Seq(tuple(inner))
            regions.append(
                Repeat(
                    body=body,
                    count=node.count,
                    index=IndexBinding(
                        name=_index_name(index_name, depth, body),
                        start=base + offset,
                        step=node.period_span,
                    ),
                    evidence=IdenticalStructure(
                        _evidence_signature(node.nodes, ignored)
                    ),
                )
            )
        offset += node.span
    return regions


def _index_name(index_name: str, depth: int, body: Region) -> str:
    """A name for this level's index that the body does not already bind.

    The outermost repeat carries the name the caller asked for and each level
    inside it carries that name with its depth. A body that already binds the
    chosen name -- a caller handing over blocks that are themselves repeats --
    takes the next number rather than failing, since shadowing is what the
    numbering exists to avoid.
    """
    name = index_name if depth == 0 else f"{index_name}_{depth}"
    bound = body.bound_indices
    candidate, taken = name, 1
    while candidate in bound:
        taken += 1
        candidate = f"{name}_{taken}"
    return candidate


# --- what the caller passed --------------------------------------------------


def _as_blocks(blocks: object) -> tuple:
    if isinstance(blocks, Region):
        raise TypeError(
            "the blocks of a forward pass are a sequence of regions; one region "
            "on its own has no repetition to find"
        )
    try:
        items = tuple(blocks)
    except TypeError:
        raise TypeError(
            f"expected a sequence of regions, got {type(blocks).__name__}"
        ) from None
    for item in items:
        if not isinstance(item, Region):
            raise TypeError(f"a block is a region, got {type(item).__name__}: {item!r}")
    if not items:
        raise ValueError(
            "a forward pass with no blocks in it is a claim that nothing ran, "
            "not a sequence whose repetition is yet to be found"
        )
    return items


def _as_ignored(ignore_attrs: object) -> frozenset:
    if isinstance(ignore_attrs, str):
        raise TypeError(
            "ignore_attrs names the attributes a signature leaves out; one "
            f"string reads as a set of its letters, got {ignore_attrs!r}"
        )
    names = tuple(ignore_attrs)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"an attribute name is a str, got {type(name).__name__}")
    return frozenset(names)
