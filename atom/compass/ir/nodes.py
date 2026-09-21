# SPDX-License-Identifier: MIT
"""The nodes a cost graph is built from, and the rules that make one well formed.

A forward pass is a prologue, a great many near-identical layer bodies, and an
epilogue. Recorded flat, one node per dispatched operator, a 27B model is 2,999
nodes that answer for exactly the batch they were traced at and for nothing
else. Recorded as a tree -- a sequence of regions, with repetition named rather
than written out -- the same model is a handful of bodies, each priced once.

So there are four kinds of region and they compose:

* `Op`      one operator. The only leaf.
* `Seq`     regions one after another.
* `Repeat`  one body, run a stated number of times, over a named index.
* `Par`     regions that overlap on the device, combined by a join policy.

Three rules make a tree of these sound rather than merely expressible, and each
is carried by the types rather than by a convention a caller is asked to follow.

1. **A `Repeat` body is any region**, a `Seq` or another `Repeat` included.
   There is no arity limit and no depth limit, because a body is typed as a
   region and a region is any of the four. A model that interleaves two layer
   classes on a period and repeats that period needs two levels; a type that
   only allowed a leaf body would flatten it back to a linear sequence and give
   up the compression that makes the tree worth having.

2. **A `Repeat` carries the index it varies over.** The binding is a mandatory
   field with no default, so a `Repeat` cannot be built without saying what
   distinguishes one instance from the next. Without it, a repeat of twenty
   asserts twenty identical bodies, and a body whose cost moves with the layer
   -- a cache offset, a per-layer expert count, a window period -- is priced as
   if it did not. A body that binds the same name again is refused: the inner
   binding would shadow the outer one, and the node that reads the name would
   silently get the wrong instance.

3. **Grouping is an optimisation, and has to be free.** A `Repeat` stands in
   for the sequence it replaces only if the two cost the same; where they do
   not, the region stays a `Seq`. Checking that is not this module's job, but
   making the unchecked version awkward is: `Repeat` takes a mandatory
   `GroupingEvidence` naming what was compared, and that type is abstract, so
   neither a bare instance nor a flag will do.

Every region reports two sets of index names, each composed from its immediate
children rather than by searching the tree. `bound_indices` is what the repeats
at or below it bind. `free_indices` is what the context keys below it name and
nothing binds -- an unresolved reference, which `Graph` refuses, because a body
that names an index no enclosing repeat supplies cannot be priced per instance.

**What an `Op` carries, and why cost is not a function of it.** ATOM registers
whole subsystems as single dispatcher operators -- attention, mixture-of-experts
routing, a tuned GEMM family. Those take a layer name and look the module up in
a context the caller established; the tensors that actually decide their cost
(the block tables, the context lengths, the slot mapping, the cumulative
sequence lengths) are ambient, not arguments. An operator signature widened to
carry them was tried and reverted, and the reason generalises: arguments cannot
carry non-tensor ambient state through a compiled graph, so no signature change
would have worked. `context_ref` is the consequence -- the handle by which that
state is found again at predict time -- and `attrs` refuses to hold a snapshot
of one, because a snapshot records whichever value the tracing forward happened
to see. One such capture took a maximum sequence length from a warm-up dummy and
priced attention at 163.6 us against a true 23.0 us.

`attrs` refuses such a snapshot two ways, and only the first is a guarantee: no
value that looks like a tensor, and no name on `AMBIENT_READINGS`. The name list
is a tripwire for the readings seen in the metadata so far, not a proof -- it
cannot know the next field somebody adds, and two of its entries were added only
after a review found them missing while one-character-different spellings were
already there. What carries the weight is that ambient state is reached through
`context_ref` when a price is asked for, not copied into the node at all.

For the same reason this module offers no price key built out of a node. A key
made from shapes and static attributes alone has been measured unsound: two
equally valid allocations of one step, same shapes throughout, moved 64 of 2,439
operator signatures, and the step summed to 32.667 ms under one and 28.360 ms
under the other. Node equality here is structural identity -- useful for
deciding that two blocks are interchangeable -- and is not a statement that two
nodes cost the same.
"""

import abc
import enum
import math
import string
from dataclasses import dataclass
from typing import Any

from .shapes import Shape, as_shapes

#: Readings that decide an operator's cost and change every step. They are
#: reached through the context reference when a price is asked for; a copy taken
#: at trace time is a value from whichever forward did the tracing. A tripwire
#: for the names seen so far, not a proof -- see the module docstring.
AMBIENT_READINGS = frozenset(
    {
        "batch_ptr",
        "block_table_tensor",
        "block_tables",
        "context_lens",
        "cu_seqlens_k",
        "cu_seqlens_q",
        "kv_indices",
        "kv_indptr",
        "kv_last_page_lens",
        "max_query_len",
        "max_seq_len",
        "max_seqlen_k",
        "max_seqlen_q",
        "num_actual_tokens",
        "num_reqs",
        "nums_dict",
        "query_start_loc",
        "seq_lens",
        "slot_mapping",
        "token_chunk_offset_ptr",
    }
)


class NodeKind(enum.Enum):
    """Where an operator's cost comes from.

    The three differ in what has to exist for the node to be priced at all, so
    the distinction is not bookkeeping: a captured operator is priced from what
    was recorded, an opaque one from a hand-written extractor for its family,
    and a declared one from an assumption that is in the artifact precisely so
    that it can be read and disputed.
    """

    #: Recorded from a dispatcher event, internals visible, priced from its own
    #: shapes and attributes.
    CAPTURED = "captured"

    #: Recorded as one node whose internal kernels are invisible, because a
    #: whole subsystem is registered as a single operator. It is priced whole,
    #: never decomposed -- roughly two thirds of a step sits in a dozen of them.
    OPAQUE_LEAF = "opaque_leaf"

    #: Not observed by the tracer. Present so that work the trace cannot see is
    #: an explicit assumption rather than a silent absence.
    DECLARED = "declared"

    def __str__(self) -> str:
        return self.value


class JoinPolicy(enum.Enum):
    """How the cost of overlapping branches combines.

    Three values, because taking the maximum is right in only one of the three
    regimes and wrong by a wide margin in the other two.
    """

    #: Branches are small and the device is not saturated, so the shorter one
    #: hides entirely inside the longer: the join costs nothing.
    MAX = "max"

    #: Both branches are compute-heavy, so they contend: the pair takes at least
    #: as long as their combined work divided by what the device can sustain.
    RESOURCE_BOUND = "resource_bound"

    #: One branch occupies the whole device -- a grid-wide barrier needs every
    #: block resident at once -- so nothing overlaps it and the costs add.
    EXCLUSIVE = "exclusive"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ContextRef:
    """Where a node's ambient, cost-deciding state is found again.

    The key is the name the operator was given to look its module up with. It
    may carry `{name}` placeholders naming repeat indices, which is what lets
    one recorded body serve every instance of a repeat: the body records
    `model.layers.{layer}.self_attn`, and pricing instance *i* binds `layer` to
    that instance's index value. Written without placeholders, a body would name
    one layer and every other instance would be priced against it.

    A placeholder has to be a name a repeat index could carry, so `{0}`, `{}`
    and `{x.y}` are refused here rather than at the point somebody tries to bind
    them, and a key that does not parse is refused with a reason rather than
    raising out of the formatter later.
    """

    key: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str):
            raise TypeError(f"a context key is a str, got {type(self.key).__name__}")
        if not self.key.strip():
            raise ValueError("a context key must not be empty")
        for name in self._fields():
            if not name.isidentifier():
                raise ValueError(
                    f"{self.key!r} has the placeholder {{{name}}}, which no "
                    "repeat index can be named; a placeholder is a plain name. "
                    "An automatically numbered one is refused for the same "
                    "reason: nothing can bind it."
                )

    def _fields(self) -> tuple[str, ...]:
        """Every placeholder in the key, as written, including unusable ones."""
        try:
            fields = [field for _, field, _, _ in string.Formatter().parse(self.key)]
        except ValueError as exc:
            raise ValueError(
                f"{self.key!r} is not a usable context key: {exc}"
            ) from None
        return tuple(field for field in fields if field is not None)

    @property
    def index_names(self) -> tuple[str, ...]:
        """The repeat indices this key is written in terms of, in order."""
        return tuple(dict.fromkeys(self._fields()))

    def bind(self, **values: int) -> "ContextRef":
        """This key with its indices substituted, for one instance of a repeat."""
        names = self.index_names
        missing = [name for name in names if name not in values]
        unknown = [name for name in values if name not in names]
        if missing or unknown:
            raise KeyError(
                f"cannot bind {self.key!r}: it is written in terms of "
                f"{list(names)}; missing {missing}, not used {unknown}"
            )
        return ContextRef(self.key.format(**values))

    def __str__(self) -> str:
        return self.key


class Region:
    """One node of a cost graph. `Op`, `Seq`, `Repeat` and `Par` are the four."""

    __slots__ = ()

    @property
    def bound_indices(self) -> frozenset[str]:
        """Every repeat index bound at or below this region."""
        raise NotImplementedError

    @property
    def free_indices(self) -> frozenset[str]:
        """Index names used below this region that nothing at or below it binds."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Op(Region):
    """One operator: what ran, on what shapes, on which stream.

    `attrs` holds the non-tensor arguments and static configuration that are
    part of what the operator is -- a data type, a transpose flag, a quantisation
    scheme. It is part of the node's identity and it is not enough to price the
    node by; see the module docstring.
    """

    name: str
    kind: NodeKind
    in_shapes: tuple[Shape, ...]
    out_shapes: tuple[Shape, ...]
    attrs: tuple[tuple[str, Any], ...] = ()
    stream_id: int = 0
    context_ref: ContextRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"an operator needs a name, got {self.name!r}")
        if not isinstance(self.kind, NodeKind):
            raise TypeError(f"kind must be a NodeKind, got {type(self.kind).__name__}")
        object.__setattr__(self, "in_shapes", as_shapes(self.in_shapes))
        object.__setattr__(self, "out_shapes", as_shapes(self.out_shapes))
        object.__setattr__(self, "attrs", _as_attrs(self.attrs))
        if isinstance(self.stream_id, bool) or not isinstance(self.stream_id, int):
            raise TypeError(
                "stream_id is the stream the operator runs on, as an int, got "
                f"{type(self.stream_id).__name__}"
            )
        if self.stream_id < 0:
            raise ValueError(f"stream_id cannot be negative, got {self.stream_id}")
        if self.context_ref is not None and not isinstance(
            self.context_ref, ContextRef
        ):
            raise TypeError(
                "context_ref must be a ContextRef, got "
                f"{type(self.context_ref).__name__}"
            )

    @property
    def bound_indices(self) -> frozenset[str]:
        return frozenset()

    @property
    def free_indices(self) -> frozenset[str]:
        if self.context_ref is None:
            return frozenset()
        return frozenset(self.context_ref.index_names)


def _as_attrs(attrs: Any) -> tuple[tuple[str, Any], ...]:
    """Normalise attributes to a sorted tuple of pairs, refusing what cannot keep.

    Sorted, so two recordings of one operator compare equal whatever order the
    tracer visited the arguments in -- which is why a name appearing twice is
    refused rather than resolved: the sort would silently pick one of them and
    the invariant would be a coin toss.

    Hashable values only, for the same reason the shapes are. Nothing
    tensor-shaped, and no name on `AMBIENT_READINGS`: both are per-step state
    copied out of one forward, and a later step priced against the copy is
    priced against a batch that is not the one being asked about. A tensor
    hashes by identity, so the value check has to look at what it is rather than
    ask whether it can be kept.
    """
    if isinstance(attrs, (str, bytes)):
        raise TypeError(f"attributes are name/value pairs, got {attrs!r}")
    items = attrs.items() if hasattr(attrs, "items") else attrs
    pairs: list[tuple[str, Any]] = []
    seen: dict[str, None] = {}
    for item in items:
        if isinstance(item, (str, bytes)) or not isinstance(item, (tuple, list)):
            raise TypeError(f"each attribute is a (name, value) pair, got {item!r}")
        if len(item) != 2:
            raise ValueError(
                f"each attribute is a (name, value) pair, got {len(item)} "
                f"items: {item!r}"
            )
        key, value = item
        if not isinstance(key, str):
            raise TypeError(f"an attribute name is a str, got {key!r}")
        if key in seen:
            raise ValueError(
                f"{key!r} is given twice. Attributes are sorted by name so that "
                "two recordings of one operator compare equal, and a repeated "
                "name would make which value survives depend on the order."
            )
        seen[key] = None
        if key in AMBIENT_READINGS:
            raise ValueError(
                f"{key!r} changes every step and is read through the node's "
                "context reference when a price is asked for. Stored here it "
                "would be whatever value the tracing forward happened to see."
            )
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            raise ValueError(
                f"attribute {key!r} is a tensor. Its contents are this step's "
                "state, not what the operator is; reach it through the node's "
                "context reference when a price is asked for."
            )
        try:
            hash(value)
        except TypeError:
            raise TypeError(
                f"attribute {key!r} must be hashable; {type(value).__name__} "
                "is not, and the node holding it is a value"
            ) from None
        pairs.append((key, value))
    return tuple(sorted(pairs, key=lambda pair: pair[0]))


@dataclass(frozen=True, slots=True)
class Seq(Region):
    """Regions one after another, in the order they run."""

    items: tuple[Region, ...]

    def __post_init__(self) -> None:
        items = _as_regions(self.items, "a sequence")
        if not items:
            raise ValueError(
                "a sequence with nothing in it costs zero, which is a claim "
                "about the model rather than a way of saying nothing happened"
            )
        object.__setattr__(self, "items", items)

    @property
    def bound_indices(self) -> frozenset[str]:
        return frozenset().union(*(item.bound_indices for item in self.items))

    @property
    def free_indices(self) -> frozenset[str]:
        return frozenset().union(*(item.free_indices for item in self.items))


@dataclass(frozen=True, slots=True)
class IndexBinding:
    """The index a repeat varies its body over.

    Instance *i* of the repeat binds `name` to `start + step * i`, so a body
    nested two deep can still name an absolute layer: the outer repeat steps by
    the length of its period and the inner one by one.
    """

    name: str
    start: int = 0
    step: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(
                "an index name has to be usable as a placeholder in a context "
                f"key, got {self.name!r}"
            )
        for field_name in ("start", "step"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} is an int, got {type(value).__name__}")
        if self.step == 0:
            raise ValueError(
                "a step of zero binds every instance to the same value, which "
                "is the uniform body this binding exists to distinguish"
            )

    def value_at(self, position: int) -> int:
        """The value this index takes at `position`, counting from zero."""
        return self.start + self.step * position


class GroupingEvidence(abc.ABC):
    """What was compared before a repeat was allowed to replace a sequence.

    Collapsing *n* instances into one body is only free if pricing the body once
    and multiplying gives what pricing the instances separately would give. The
    two subclasses are the two things that can be compared: the instances'
    structure, and their price. A repeat takes one of them, mandatorily.

    Abstract, and not merely a base class, because an instance of this by itself
    would be a grouping justified by nothing -- cheaper to write than the flag
    the mandatory field exists to refuse.
    """

    __slots__ = ()

    @abc.abstractmethod
    def describe(self) -> str:
        """What was compared, in one line, for a record or a refusal to quote."""


@dataclass(frozen=True, slots=True)
class IdenticalStructure(GroupingEvidence):
    """Every instance carries the same canonical signature.

    The signature has to be one a later reader can recompute from the repeat's
    body. A signature nobody can reproduce is a claim that cannot be rechecked,
    which is the same as no claim.
    """

    signature: str

    def __post_init__(self) -> None:
        if not isinstance(self.signature, str) or not self.signature.strip():
            raise ValueError(
                "name the signature the instances share; an empty one says "
                f"nothing was compared, got {self.signature!r}"
            )

    def describe(self) -> str:
        return f"every instance carries the signature {self.signature}"


@dataclass(frozen=True, slots=True)
class EqualPrice(GroupingEvidence):
    """The sequence and the repeat that replaces it were priced, and agree."""

    flat_seconds: float
    grouped_seconds: float

    def __post_init__(self) -> None:
        for name in ("flat_seconds", "grouped_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} is a duration in seconds, got {type(value).__name__}"
                )
            if not math.isfinite(value):
                raise ValueError(f"{name} is not a number of seconds: {value!r}")
            if value < 0:
                raise ValueError(
                    f"{name} is a duration and cannot be negative, got {value!r}"
                )
            object.__setattr__(self, name, float(value))
        flat, grouped = self.flat_seconds, self.grouped_seconds
        if flat != grouped:
            raise ValueError(
                f"the sequence prices at {flat!r} s and the repeat replacing it "
                f"at {grouped!r} s. Grouping has to be free, so a region that "
                "prices differently grouped stays a sequence; the difference "
                "would otherwise be multiplied by the repeat count."
            )

    def describe(self) -> str:
        return f"flat and grouped both price at {self.flat_seconds!r} s"


@dataclass(frozen=True, slots=True)
class Repeat(Region):
    """One body, run `count` times, over a named index.

    Every field is mandatory. The index says what distinguishes one instance
    from the next, and the evidence says what was compared before the grouping
    was taken; a repeat built without either would be asserting uniformity that
    nobody checked.
    """

    body: Region
    count: int
    index: IndexBinding
    evidence: GroupingEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.body, Region):
            raise TypeError(
                "a repeat body is any region -- an operator, a sequence or "
                f"another repeat -- got {type(self.body).__name__}"
            )
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError(f"count is an int, got {type(self.count).__name__}")
        if self.count < 2:
            raise ValueError(
                f"a repeat runs its body at least twice, got {self.count}; a "
                "single instance is a sibling in the enclosing sequence, not a "
                "group of one"
            )
        if not isinstance(self.index, IndexBinding):
            raise TypeError(
                "a repeat carries the index it varies its body over, as an "
                f"IndexBinding, got {type(self.index).__name__}"
            )
        if not isinstance(self.evidence, GroupingEvidence):
            raise TypeError(
                "a repeat carries what was compared before the grouping was "
                f"taken, got {type(self.evidence).__name__}"
            )
        if self.index.name in self.body.bound_indices:
            raise ValueError(
                f"{self.index.name!r} is already bound inside this body. The "
                "inner binding would shadow this one, so a node naming it would "
                "read the inner instance and be priced at the wrong index."
            )

    @property
    def bound_indices(self) -> frozenset[str]:
        return frozenset({self.index.name}) | self.body.bound_indices

    @property
    def free_indices(self) -> frozenset[str]:
        return self.body.free_indices - {self.index.name}

    def index_values(self) -> tuple[int, ...]:
        """The index value each instance binds, in order."""
        return tuple(self.index.value_at(i) for i in range(self.count))


@dataclass(frozen=True, slots=True)
class Par(Region):
    """Regions that overlap on the device, and how their costs combine."""

    branches: tuple[Region, ...]
    join: JoinPolicy

    def __post_init__(self) -> None:
        branches = _as_regions(self.branches, "a parallel region")
        if len(branches) < 2:
            raise ValueError(
                f"overlap needs at least two branches, got {len(branches)}; one "
                "branch runs on its own and is a sequence"
            )
        if not isinstance(self.join, JoinPolicy):
            raise TypeError(
                "join must be a JoinPolicy: taking the maximum is right in only "
                f"one of the three regimes, got {type(self.join).__name__}"
            )
        object.__setattr__(self, "branches", branches)

    @property
    def bound_indices(self) -> frozenset[str]:
        return frozenset().union(*(branch.bound_indices for branch in self.branches))

    @property
    def free_indices(self) -> frozenset[str]:
        return frozenset().union(*(branch.free_indices for branch in self.branches))


def _as_regions(regions: Any, what: str) -> tuple[Region, ...]:
    if isinstance(regions, Region):
        raise TypeError(f"{what} holds several regions; pass them in a sequence")
    try:
        items = tuple(regions)
    except TypeError:
        raise TypeError(
            f"{what} holds a sequence of regions, got {type(regions).__name__}"
        ) from None
    for item in items:
        if not isinstance(item, Region):
            raise TypeError(
                f"{what} holds regions; got {type(item).__name__}: {item!r}"
            )
    return items
