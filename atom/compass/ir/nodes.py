# SPDX-License-Identifier: MIT
r"""The nodes a cost graph is built from, and the rules that make one well formed.

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

`attrs` refuses such a snapshot two ways, and the two cover different halves.

The first is a type allowlist, and it is a guarantee. An attribute value is a
number, a string, an enum member, `None`, or a tuple of those, and nothing else
-- so a tensor cannot be stored, and neither can a live symbolic value. Stating
it positively is the whole of why it holds. The earlier version asked whether a
value could be hashed, which reads as a safe question and is not one: measured
on torch 2.10.0+rocm7.2.4, that test refuses a `SymInt` loudly, accepts a
`SymBool` and a `SymFloat` silently -- specialising them, and pinning a
`SymFloat` to its trace-time hint with a guard nobody asked for -- and raises
`GuardOnDataDependentSymNode` on an unbacked one. `isinstance` against `bool`,
`int`, `float`, `str`, `bytes` and `tuple` answers False for all three symbolic
types and moves the guard list not at all. Asking a value a question is the
leak; asking its type is not.

The second is `AMBIENT_READINGS`, a list of names, and it is not a guarantee. It
exists because the type rule cannot see the difference between a width that is
part of what the operator is and a token count copied out of this step's
metadata: both are `int`. The names are the ones ATOM reads off its attention
metadata, counted from the tree with

    grep -rhoE '\b(attn_metadata|metadata|md|gdn_metadata|attn_md)\.[a-z_][a-z0-9_]*' \
        atom/ --include=*.py | sed 's/.*\.//' | sort | uniq -c | sort -rn

and matched case-insensitively, since a recorded name may arrive in any case.
Regenerate it the same way when the metadata grows: it is incomplete the moment
somebody adds a field, and it has already been extended twice after review. What
carries the weight is that ambient state is reached through `context_ref` when a
price is asked for, not copied into the node at all.

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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .shapes import Shape, as_shapes

#: Readings that decide an operator's cost and change every step. They are
#: reached through the context reference when a price is asked for; a copy taken
#: at trace time is a value from whichever forward did the tracing. Counted from
#: the tree, matched case-insensitively, and incomplete by construction: a
#: tripwire for the readings seen so far, not a proof. See the module docstring.
AMBIENT_READINGS = frozenset(
    {
        "batch_id_per_token",
        "batch_ptr",
        "block_table_tensor",
        "block_tables",
        "context_lens",
        "cu_seqlen_ks",
        "cu_seqlens_k",
        "cu_seqlens_q",
        "index_topk",
        "kv_indices",
        "kv_indices_csa",
        "kv_indptr",
        "kv_last_page_lens",
        "max_query_len",
        "max_seq_len",
        "max_seqlen_k",
        "max_seqlen_q",
        "n_committed_csa_per_seq",
        "n_committed_csa_per_seq_cpu",
        "n_committed_hca_per_seq",
        "n_committed_hca_per_seq_cpu",
        "n_committed_per_token",
        "num_actual_tokens",
        "num_decodes",
        "num_prefills",
        "num_reqs",
        "nums_dict",
        "qo_indptr",
        "query_start_loc",
        "reduce_final_map",
        "reduce_indptr",
        "reduce_partial_map",
        "seq_lens",
        "skip_prefix_len_csa",
        "slot_mapping",
        "sparse_cu_seqlens_q",
        "sparse_kv_indptr",
        "sparse_kv_last_page_lens",
        "state_slot_mapping",
        "state_slot_out",
        "state_slot_out_cpu",
        "token_chunk_offset_ptr",
        "token_to_seq_idxs",
        "work_indptr",
        "work_info_set",
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

    Values are checked by type and never by what they can do; the module
    docstring says why that distinction is the whole of the guarantee. Names are
    checked against `AMBIENT_READINGS`, which catches the per-step readings that
    arrive as plain numbers and that no type rule could tell from a width.
    """
    if isinstance(attrs, (str, bytes)):
        raise TypeError(f"attributes are name/value pairs, got {attrs!r}")
    items = attrs.items() if isinstance(attrs, Mapping) else attrs
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
        if key.lower() in AMBIENT_READINGS:
            raise ValueError(
                f"{key!r} changes every step and is read through the node's "
                "context reference when a price is asked for. Stored here it "
                "would be whatever value the tracing forward happened to see."
            )
        pairs.append((key, _as_attr_value(key, value)))
    return tuple(sorted(pairs, key=lambda pair: pair[0]))


#: What an attribute value may be: what the operator is, never what this step
#: is. Checked with `isinstance` and nothing else -- a live symbolic value
#: answers False to every one of these and is refused without being asked a
#: question it would answer by installing a guard.
ATTR_VALUE_TYPES = (bool, int, float, str, bytes, enum.Enum)


def _as_attr_value(key: str, value: Any) -> Any:
    """Return `value` if it is one an attribute may hold, else refuse by type."""
    if value is None or isinstance(value, ATTR_VALUE_TYPES):
        return value
    if isinstance(value, tuple):
        return tuple(_as_attr_value(key, item) for item in value)
    raise TypeError(
        f"attribute {key!r} is a {type(value).__name__}. An attribute is a "
        "number, a string, an enum member, None, or a tuple of those -- what "
        "the operator is, not what this step is. A tensor and a size that is "
        "not known yet are both refused here: their contents are this step's "
        "state, reached through the node's context reference when a price is "
        "asked for, and a size that is not known yet cannot be stored anywhere "
        "without being asked a question that resolves it."
    )


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

    Collapsing *n* instances into one body is only free if pricing the body's
    operators once and *reusing* those prices for every instance gives what
    pricing the instances separately would give -- the same prices, in the same
    order, added the same way. What is reused is the body's sequence of
    per-operator prices, re-emitted in order once per instance, and not a body
    total. Multiplying is a different sum, because it re-associates. Reuse
    reproduces the recorded price exactly in all three shapes measured and
    multiplying in none of them: eight identical layers, 3.2e-05 s multiplied
    against 3.200000000000001e-05 s recorded; a four-block pattern repeated
    twenty times, 0.00036 against 0.0003600000000000009; six instances of 0.1 s,
    0.6000000000000001 against 0.6.

    The two subclasses are the two things that can be compared: the instances'
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
