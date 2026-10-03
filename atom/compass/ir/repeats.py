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

A dimension enters as the whole of its identity: for a size that is known, the
number; for one that is not, what it renders to together with the capture that
rendering is read against, because symbols are numbered per capture and two
unrelated traces both produce `s26`. That is the one thing in this module that
depends on what a dimension turns out to be, and `_dim_text` is the one place to
change if that answer changes.

**What the index bindings mean, and how a reader resolves one.** A repeat
carries the index it varies its body over, and an index name is in scope for
everything below the repeat that binds it. With nesting, no single binding is a
block's position: the outer repeat steps by the length of its period and the
inner one by one, so

    position of a block = the sum of the values bound by every repeat above it
                        + its own offset in the body that holds it

and both terms are read off the emitted tree. The offset of an item within a
body is the sum, over the items before it, of `count * index.step` for a repeat
and one for a block -- how many of the flat sequence's blocks each of them
stands for.

Reading only the nearest binding, or only the outermost, is wrong in a way that
looks right. In a stack of three of one class then one of another, repeated:
the second block of the third period is at `8 + 1 = 9`. The outer binding alone
says 8, and 8 is a block that exists, so nothing announces the mistake.

The outermost repeat carries the name the caller asked for and each level inside
carries that name with its depth, so a body already binding a name is never
shadowed. Which level gets the plain name is not arbitrary: a block that is a
plain sibling of an inner run -- the `B` in a period of `AAAB` -- sits directly
in the outer body, so only an enclosing binding can supply the index its context
key names. Putting the caller's name on the innermost repeat instead would leave
that sibling's key naming something nothing binds.

**What a signature difference costs.** Two blocks that differ only in something
pricing ignores -- a per-layer quantisation scale is the standing example -- get
different signatures and fail to group. That costs compression and never
correctness, which is the right way round, and `ignore_attrs` is the way to say
that a named attribute is not part of what a block is. A signature taken that
way says so in its own text, so two signatures taken under different rules are
different signatures and a reader can see which rule produced the one it holds.
The names go in as a comma-separated list, which is why a name carrying one of
that list's own marks is refused: `("a,b",)` and `("a", "b")` would otherwise
write the same text, and two different rules would produce one signature.
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


#: How a signature says what it was taken without. The names may not contain
#: either mark, nor the comma between them; see `_as_ignored`.
_LESS_OPEN = "less("
_LESS_CLOSE = ")|"


def _prefix(ignored: frozenset) -> str:
    """What a signature was taken without, so that two rules cannot collide.

    A repeat carries the signature its instances share, and a reader
    reproduces it by signing the body beside it under the same rule. Leaving an
    attribute out changes what a block is for this purpose, so the string says
    which ones were left out and a signature taken one way never equals one
    taken another. Nothing is said when nothing was left out, which keeps the
    ordinary signature the ordinary text.
    """
    if not ignored:
        return ""
    return _LESS_OPEN + ",".join(sorted(ignored)) + _LESS_CLOSE


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
    # A size that is not known yet enters as the whole of its identity: what it
    # renders to, and the capture that rendering is read against. A shape
    # environment numbers symbols per capture, so two unrelated traces both
    # produce `s26`; on the rendering alone, two bodies from two captures would
    # sign the same and group as one -- a false equal, which is the direction
    # that does not announce itself. The scope is quoted so that neither half
    # can run into the other.
    #
    # Read field by field, and never `repr` of the dimension: that spells the
    # class holding the identity as well, so every signature would carry a type
    # name and would move the day that type were renamed. Never a comparison or
    # a conversion either -- on a size that is not known yet, either installs a
    # guard, and the guard is part of the record of where the trace is valid.
    if not is_symbolic(dim):
        return str(dim)
    return f"?{dim.scope!r}:{dim}"


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


@dataclass(frozen=True, slots=True, eq=False)
class _Leaf:
    """One block of the flat sequence, and what it does.

    Compared by identity, never by value: generated equality would compare the
    region it holds, and comparing two regions compares their dimensions one at
    a time. Everything this module decides, it decides on the signature string.
    """

    region: Region
    signature: str

    @property
    def span(self) -> int:
        return 1


@dataclass(frozen=True, slots=True, eq=False)
class _Run:
    """A stretch of the sequence that is one period repeated `count` times.

    Compared by identity, for the reason `_Leaf` is.
    """

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

    What falls is the number of leaves, not the number of nodes. A productive
    pass turns `k` copies of a period of `L` leaves into one run holding `L`,
    and `k >= 2` and `L >= 1` make `k * L > L`, so every pass that changes
    anything strictly reduces the leaves and the passes stop. The node count is
    not the measure and does not always fall: two identical blocks become one
    run holding one leaf, which is two nodes before and two after.

    The comparison is over signatures, which a pass changes exactly when it
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
    classes is found in the same pass as a run of one. A count of one means
    nothing repeats here, and the node stays where it is.

    **The shortest period wins a tie, and that is not a preference.** Six of one
    block can be written as six repeats of it or as three repeats of a pair, and
    the two are not equally true. A repeat binds one index value per instance,
    so the pair form gives one value to two blocks and both resolve the same
    layer -- the uniformity a mandatory index binding exists to refuse. The
    longer period can never buy anything a price cares about, since the blocks
    and their order are the same either way; it can only lose the distinction
    between two of them.
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
        stray = [mark for mark in (",", ")", "|") if mark in name]
        if stray:
            raise ValueError(
                f"{name!r} contains {stray}, and a signature lists what it was "
                "taken without as a comma-separated run of names between "
                f"{_LESS_OPEN!r} and {_LESS_CLOSE!r}. A name carrying one of "
                "those would write the text two names write, so two different "
                "rules would produce one signature."
            )
    return frozenset(names)
