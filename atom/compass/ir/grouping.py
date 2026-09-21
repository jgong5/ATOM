# SPDX-License-Identifier: MIT
"""Whether a detected grouping was free to take, and the proof that it was.

A detector reads a flat block sequence and names its repetition. It compares
what the blocks *are* and cannot compare what they *cost*, so every repeat it
emits is a proposal: twenty instances of one body, on the strength of twenty
identical signatures. This module is where that proposal is tested. A repeat
survives only where the sequence it stands for and the repeat itself price
to the same prices, term for term; where they do not, the repeat is taken
apart and the blocks stay siblings in the sequence that held it.

Refusing is cheap and accepting is not. A repeat of twenty that is wrong by one
microsecond is wrong by twenty, every step, in one direction -- so the failure
is a bias in the total and not a spread around it, and no amount of averaging
finds it. Losing a grouping that was in fact free costs replay time and nothing
else.

**What a grouping saves, which decides what it may change.** Pricing an
operator is a lookup and a law evaluation; adding two floats is neither. A
flat step of a 27B model was measured at 41.7 ms to replay, of which ~39 ms was
obtaining ~2,440 operator prices and the rest everything else. So the saving
worth having is asking for each price once per *body* instead of once per
*instance*; the additions are noise beside it. This module therefore defines
the grouped form as **the same additions in the same order, over prices that
were obtained once and reused**, and not as a body price multiplied by a count.

That is not a detail. Float addition is not associative, so a total is a
property of the order it was summed in, and a run that has to be reproducible
cannot have that order depend on which form of the tree it happened to hold.
Multiplying re-associates: folding six copies of 0.1 from zero gives 0.6 and
`6 * 0.1` gives 0.6000000000000001. Those differ, they differ for a
reason that has nothing to do with the model, and a criterion loose enough to
call them equal is loose enough to hide a body that really does cost more at
one instance than another. Keeping the order fixed removes the question: the
two forms are bit-identical exactly when every instance prices as the first
one, which is the claim a repeat makes and the only thing worth checking.

**And the comparison is of the prices, not only of their total.** A region's
cost is not only what it sums to; it is what it contributes to the sum that
contains it. Two different sequences of prices can fold to one number from zero
and to two different numbers from a running total, so a check that compared
totals alone would admit a grouping that moves the step it sits in. The two
forms are therefore compared term by term, in order; the totals are then equal
by construction, and are what the surviving repeat records. One consequence is
worth stating rather than discovering: a difference too small to survive being
added to the running total is refused all the same. It is a real difference
between two instances, and the sum it vanishes into here is not the only sum it
will ever be part of.

So **equality here is bitwise, with no tolerance**, and it is affordable
because nothing re-associates. A price is one law evaluated on one set of
readings, so two instances that cost the same produce the same bits; a bit
that moved means the law was handed something else. A tolerance would have to
be a number, the number would have to be defended per model and per device,
and what it would buy is acceptance of groupings whose instances genuinely
differ -- multiplied by the count.

**A price is asked for with the key, never with the shape alone.** Two
instances of one body have identical shapes and attributes by construction:
that is what made them one body. What differs is the module each reads its
ambient state through, and that state is what decides the cost. A check that
priced by shape would compare a number with itself and report agreement for
every grouping ever proposed. Measured: two equally valid allocations of one
step, identical shapes throughout, moved 64 of 2,439 operator signatures and
summed to 32.667 ms against 28.360 ms. The pricing function here is therefore
handed the operator *and* the context key with this instance's index
substituted in, and a body whose instances price differently through their keys
is refused.

**Where an instance sits, under nesting.** A context key is written with the
repeat index as a placeholder, so pricing an instance means resolving that
placeholder to the block's absolute position. One repeat makes that the value
it binds. Two do not: the outer repeat steps by the length of the period and
the inner one by one, so a block's position is the sum of the values bound
above it plus its own offset in the body that holds it. Nothing below a repeat
knows what is above it, and a single key knows neither, so the sum cannot be
taken by either of them -- it is taken here, by the walk that is already
visiting the instances in order. The walk carries the position, and every
repeat it enters is checked against it: an instance that binds a value other
than the position it sits at is refused rather than priced, because a key
resolved from a binding that does not agree with the walk names some other
layer's state.

**A refused repeat is taken apart completely.** Its body may hold repeats of
its own, and those were proved against the positions of the first instance; at
the second instance the positions differ, and a binding written relative to a
period that no longer exists names the wrong block. Rewriting those bindings to
keep an inner grouping is possible and is not worth it: it would preserve
compression inside a region that has just been shown not to compress, and
compression is the thing this module is allowed to lose.

**Recomputing the structural evidence is possible because signing a body is a
pure function of that body.** A repeat arrives carrying the signature its
instances were found to share, and a field nobody recomputes records whatever
was written into it. So it is recomputed here, from the body beside it, with
the same attributes left out; a body that signs as something else is refused.
That works because nothing in a signature is read from outside the body -- no
live tracing state, no environment that has to still exist -- so the string is
reproducible long after the trace that produced it.

**Nothing here reads a symbolic dimension, and nothing here builds an
operator.** Two regions are never compared or hashed as values: a region's
equality compares its shapes one dimension at a time, and a signature compared
as text asks nothing of a dimension beyond how it renders. Two renderings of
one expression are two dimensions, which costs a grouping and never a wrong
price, and it is settled before a region reaches here. No operator is rebuilt
to carry a resolved key either -- the key is resolved to a string and handed
alongside the operator, so pricing a recorded trace cannot add a guard to it.
"""

from collections.abc import Callable
from dataclasses import dataclass

from .nodes import EqualPrice, IdenticalStructure, Op, Par, Region, Repeat, Seq
from .repeats import signature_of

#: What pricing one operator needs: the operator, and the key it reads its
#: ambient state through with this instance's index substituted in. `None` is
#: an operator that named no context, not a key that could not be resolved --
#: an unresolvable key is refused and never reaches here.
Price = Callable[[Op, str | None], float]


@dataclass(frozen=True, slots=True)
class Ungrouped:
    """A proposed repeat that was not proved free, and what was compared.

    The prices are present when the two forms were priced and disagreed, and
    absent when the repeat was refused before pricing -- a signature that does
    not reproduce, a binding that does not agree with the position, a body that
    holds work whose costs do not add.
    """

    at: int
    count: int
    signature: str
    reason: str
    flat_seconds: float | None = None
    grouped_seconds: float | None = None

    def __str__(self) -> str:
        return f"{self.count} instances at {self.at}: {self.reason}"


def prove_grouping(
    region: Region,
    price: Price,
    *,
    index_start: int = 0,
    ignore_attrs: object = (),
) -> tuple[Region, tuple[Ungrouped, ...]]:
    """`region` with every grouping either proved free or taken apart.

    Hand over what the detector returned, the same `index_start` it was given
    and the same `ignore_attrs`. Every repeat that survives carries the two
    prices that were compared in place of the structural evidence it arrived
    with; every repeat that does not is replaced by the blocks it stood for,
    and is named in the second half of the result together with the reason.
    """
    if not isinstance(region, Region):
        raise TypeError(
            "a grouping is proved over a region -- what the detector returned "
            f"-- got {type(region).__name__}"
        )
    if isinstance(index_start, bool) or not isinstance(index_start, int):
        raise TypeError(
            "index_start is the position of the first block, as an int, got "
            f"{type(index_start).__name__}"
        )
    refusals: list[Ungrouped] = []
    items = region.items if isinstance(region, Seq) else (region,)
    proved, _ = _prove(items, index_start, 0, price, ignore_attrs, refusals)
    kept = proved[0] if len(proved) == 1 else Seq(tuple(proved))
    return kept, tuple(refusals)


class _Refused(Exception):
    """Why one proposed repeat may not be emitted."""

    def __init__(
        self,
        reason: str,
        flat: float | None = None,
        grouped: float | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.flat = flat
        self.grouped = grouped


def _prove(
    items: tuple,
    pos: int,
    base: int,
    price: Price,
    ignored: object,
    refusals: list,
) -> tuple[list, int]:
    """One level of the tree, and the position after it."""
    out: list[Region] = []
    for item in items:
        if not isinstance(item, Repeat):
            out.append(item)
            pos += 1
            continue
        try:
            kept, pos = _prove_repeat(item, pos, base, price, ignored, refusals)
        except _Refused as why:
            refusals.append(
                Ungrouped(
                    at=pos,
                    count=item.count,
                    signature=signature_of(item.body, ignore_attrs=ignored),
                    reason=why.reason,
                    flat_seconds=why.flat,
                    grouped_seconds=why.grouped,
                )
            )
            taken_apart = _expand((item,))
            out.extend(taken_apart)
            pos += len(taken_apart)
        else:
            out.append(kept)
    return out, pos


def _prove_repeat(
    repeat: Repeat,
    pos: int,
    base: int,
    price: Price,
    ignored: object,
    refusals: list,
) -> tuple[Repeat, int]:
    """The repeat with the prices that justify it, or the reason it has none."""
    _check_signature(repeat, ignored)
    step = repeat.index.step
    if step < 1:
        raise _Refused(
            f"the index steps by {step}, so one instance of the body covers no "
            "blocks of the sequence it stands for"
        )
    first = base + repeat.index.value_at(0)
    if first != pos:
        raise _Refused(
            f"the index binds {first} at the first instance, which is the "
            f"block at {pos}"
        )
    items = _body_items(repeat.body, step)
    inner, end = _prove(items, pos, first, price, ignored, refusals)
    if end - pos != step:
        raise _Refused(
            f"one instance of the body covers {end - pos} blocks and the index "
            f"steps by {step}, so the instances do not sit where they are bound"
        )
    apart = _instances(inner, repeat, pos, base, price, reuse=False)
    together = _instances(inner, repeat, pos, base, price, reuse=True)
    flat, grouped = _fold(apart), _fold(together)
    if apart != together:
        where = next(i for i, (a, b) in enumerate(zip(apart, together)) if a != b)
        raise _Refused(
            f"the repeat prices operator {where} of the sequence at "
            f"{together[where]!r} s where the sequence itself prices it at "
            f"{apart[where]!r} s; the two forms total {flat!r} s and "
            f"{grouped!r} s, and a difference at one instance is multiplied by "
            "the count",
            flat,
            grouped,
        )
    body = inner[0] if len(inner) == 1 else Seq(tuple(inner))
    kept = Repeat(
        body=body,
        count=repeat.count,
        index=repeat.index,
        evidence=EqualPrice(flat, grouped),
    )
    return kept, pos + repeat.count * step


def _check_signature(repeat: Repeat, ignored: object) -> None:
    """Sign the body again, and read the evidence against it.

    The evidence names a signature its instances share. Taken from the body
    beside it with the same attributes left out, it has to be that string; a
    field nobody recomputes is a field that records whatever was written into
    it.
    """
    if not isinstance(repeat.evidence, IdenticalStructure):
        return
    taken = signature_of(repeat.body, ignore_attrs=ignored)
    if taken != repeat.evidence.signature:
        raise _Refused(
            f"the body signs as {taken!r} and the evidence beside it names "
            f"{repeat.evidence.signature!r}"
        )


def _instances(
    items: tuple, repeat: Repeat, pos: int, base: int, price: Price, reuse: bool
) -> list:
    """Every price of every instance, in the order the blocks run.

    With `reuse`, the first instance's prices stand for all of them -- the
    grouped form, asking for each price once per body rather than once per
    instance. The two lists are the same length and in the same order either
    way, so they can be read against each other term by term, and they differ
    exactly at the operators an instance prices differently from the first.
    """
    values: list[float] = []
    first: list[float] | None = None
    for instance in range(repeat.count):
        at = pos + instance * repeat.index.step
        bound = base + repeat.index.value_at(instance)
        if bound != at:
            raise _Refused(
                f"instance {instance} binds {bound} and is the block at {at}, "
                "so its context key would name another layer's state"
            )
        if reuse and first is not None:
            values.extend(first)
            continue
        priced, _ = _values(items, at, bound, price, reuse)
        if first is None:
            first = priced
        values.extend(priced)
    return values


def _values(
    items: tuple, pos: int, base: int, price: Price, reuse: bool
) -> tuple[list, int]:
    """The prices of one level, and the position after it."""
    values: list[float] = []
    for item in items:
        if isinstance(item, Repeat):
            inner = _body_items(item.body, item.index.step)
            values.extend(_instances(inner, item, pos, base, price, reuse))
            pos += item.count * item.index.step
        else:
            values.extend(price(op, _resolve(op, pos)) for op in _operators(item))
            pos += 1
    return values, pos


def _operators(block: Region):
    """The operators of one block, in the order they run."""
    if isinstance(block, Op):
        yield block
    elif isinstance(block, Seq):
        for item in block.items:
            yield from _operators(item)
    elif isinstance(block, Par):
        raise _Refused(
            "the body holds branches that overlap on the device, whose costs "
            "combine by a join policy and not by addition"
        )
    else:
        raise _Refused(
            f"the body holds a {type(block).__name__} inside a block, which "
            "has no position of its own to resolve a context key at"
        )


def _resolve(op: Op, at: int) -> str | None:
    """The key this operator reads its ambient state through, at position `at`.

    One placeholder is the position. Two are not resolvable from a position at
    all -- a block sits at one, and substituting it for both would name a
    module that is somewhere else entirely -- so the grouping is refused rather
    than priced through a key that was guessed at.
    """
    ref = op.context_ref
    if ref is None:
        return None
    names = ref.index_names
    if not names:
        return ref.key
    if len(names) > 1:
        raise _Refused(
            f"{ref.key!r} is written in terms of {list(names)}; a block sits at "
            "one position and cannot say which of them it is"
        )
    return ref.bind(**{names[0]: at}).key


def _fold(values: list) -> float:
    """Sum in the given order, left fold from zero. The only summation here."""
    total = 0.0
    for value in values:
        total += value
    return total


def _body_items(body: Region, step: int) -> tuple:
    """The blocks and sub-repeats one instance of `body` is made of.

    A body that covers one block is that block, whatever it holds inside. A
    body that covers several is a sequence of them, and the step the repeat
    takes is what says which of the two this is.
    """
    if step <= 1 or not isinstance(body, Seq):
        return (body,)
    return body.items


def _expand(items: tuple) -> list:
    """The blocks `items` stand for, with every repeat taken apart."""
    out: list[Region] = []
    for item in items:
        if isinstance(item, Repeat):
            inner = _expand(_body_items(item.body, item.index.step))
            out.extend(inner * item.count)
        else:
            out.append(item)
    return out
