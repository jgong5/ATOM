# SPDX-License-Identifier: MIT
"""Deciding whether a recorded graph describes the step in front of us.

A trace is a straight-line record of one path. Where that record is valid is
stated in two parts, because one part cannot cover the other:

* a **discrete key**, matched by equality -- the things a step either is or is
  not, which leave no trace in the shapes: the forward mode, whether the batch
  has cached context, whether the step produces output, the speculation width,
  replayed or eager, two-batch overlap, the attention backend, and whether the
  step is a dummy;
* a **symbolic domain**, evaluated -- the guards the tracing run installed when
  it branched on a size, plus the range each symbol was left with.

Guards record only branches taken on a symbolic size. A branch on a flag leaves
nothing behind, so a graph carrying guards alone would claim to cover the eager
path it never ran down. That is what the key is for.

**Both parts are produced, not authored.** `from_shape_env` reads the guards
off the tracing run, so the predicate cannot drift away from the control flow it
describes, and a step outside every recorded domain is refused by name rather
than priced against a record of some other path.

**What ATOM already states, and what it does not.** `atom/model_engine/
run_labels.py` builds the profiler label for every forward pass, and its field
taxonomy is the key's: the label's prefix separates `prefill` from `decode` from
`eager_decode`, which is both the forward mode and replayed-versus-eager; a
`dummy_` prefix marks a dummy run; `tbo=1` marks a step that ran two-batch
overlap. Four things it does not state, and a key built only from it would be
silently blind to each:

1. whether the batch has cached context -- the eager label carries `ctx=` per
   sequence, truncated to the first three past five sequences, and the replayed
   decode label carries none at all;
2. whether the step produces output -- `ScheduledBatch.produces_output()` is
   what the head is skipped on, and no label field reports it;
3. the attention backend;
4. the speculation width on any path but the replayed one, and there only when
   it is above zero, so a width of zero and a step that is not speculating are
   one absence.

A label is also a string built for a profiler: lossy, truncated, and parsed back
out by a regex. The key here is the values, not the rendering of them, and which
fields it carries is the recording side's to state -- what this module fixes is
that the two sides must carry the same fields, since a field one side omits is a
condition nobody checked.

**Nothing here evaluates a symbolic value, on either half.** `int()` on a
symbolic size, or a Python comparison against one, installs a guard -- so a
predicate that did either would change the very artifact it was asked about, and
the second reading of a graph would differ from the first. Guards are evaluated
by substituting the step's sizes into the recorded expression, which is a
rewrite and records nothing; a size offered as a symbol is refused, with the
tracing run's own hint lookup named as the way to resolve it.

The key half needs a different defence, and finding that out is the reason this
paragraph is two paragraphs. The obvious gate on a key's value -- ask whether it
hashes, since a key is held in a value and matched by equality -- is itself an
operation on the value, and the two symbolic types answer it in opposite
directions. A symbolic size raises, which is safe by accident. A symbolic
boolean answers, by forcing itself concrete first: `hash()` on one installs a
guard and returns a number, so a graph would have been narrowed by the act of
recording where it applies, silently and with nothing raised. The same gate also
lets through a value whose `==` or truth test raises rather than answering,
which would turn a mismatch into an exception instead of a verdict. So a key's
values come from a stated list of types instead, and nothing asks a value
anything. `_CONDITION_TYPES` carries the reasoning.

As in the rest of this package, no tensor library and no symbolic-algebra
library is imported: a guard is anything that can report its free symbols and
substitute them.
"""

import enum
from dataclasses import dataclass
from typing import Any

from .graph import Applicability
from .shapes import is_symbolic

#: What a discrete condition's value may be. A stated list of types, not a test
#: that the value behaves: asking a value whether it hashes is itself an
#: operation, and the two symbolic types answer that question in opposite
#: directions -- a symbolic size raises, a symbolic boolean answers silently by
#: forcing itself concrete, which installs a guard. A gate that can install a
#: guard is the leak it was meant to close, so nothing here asks; a value is
#: either one of these types or it is refused. Equality between two of these is
#: a plain bool, which is also what keeps a mismatch a verdict rather than an
#: exception.
_CONDITION_TYPES = (bool, int, str, enum.Enum, type(None))


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a recorded graph describes a step, and if not, why not.

    A refusal carries its reason because the reason is the result: a step no
    recorded graph admits has been detected, and what has to be readable is
    which condition it failed and what it offered.
    """

    applies: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if self.applies and self.reason:
            raise ValueError(
                f"an admitted step has nothing to explain, got {self.reason!r}"
            )
        if not self.applies and not self.reason.strip():
            raise ValueError(
                "a refusal states which condition the step failed; an unexplained "
                "one cannot be acted on and reads as an error path"
            )


@dataclass(frozen=True, slots=True)
class GuardedApplicability(Applicability):
    """A discrete key and a symbolic domain, together deciding one step.

    `key` is the conditions matched by equality, `guards` the expressions the
    tracing run branched on, and `ranges` the interval each symbol was left
    inside, by symbol name. Both halves of the domain are kept: a guard records
    a branch that was taken, while a range also carries the bound no branch
    asked about -- a size recorded as at least two describes a graph that says
    nothing about a batch of one.
    """

    key: tuple[tuple[str, Any], ...]
    guards: tuple[Any, ...] = ()
    ranges: tuple[tuple[str, tuple[Any, Any]], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _as_key(self.key))
        object.__setattr__(self, "guards", _as_guards(self.guards))
        object.__setattr__(self, "ranges", _as_ranges(self.ranges))

    @classmethod
    def from_shape_env(cls, shape_env: Any, key: Any) -> "GuardedApplicability":
        """Read the domain off the run that traced the graph.

        Both attributes are read, never evaluated, so recording the predicate
        installs no guard of its own -- which matters, because the tracing run
        is still live and one extra guard would narrow the graph being recorded.

        Only the symbols the run actually branched on are recorded. The rest of
        `var_to_range` is the run's own bookkeeping about sizes whose value
        never changed which path was taken, so a range over one of them is not
        a condition the record earned. Carried as one it would be unanswerable
        rather than merely unnecessary: a size the router decides is left there
        bounded by nothing at all, with no guard naming it, and a step has no
        way to bind a count that had not been produced when the trace ran. Every
        step would then be refused, naming a symbol nobody could have supplied.
        The range of a symbol a guard does name is kept, because that one says
        how much room the branch left -- a decode path recorded as at least two
        tokens says nothing about a batch of one, and no guard says that.
        """
        guards = tuple(guard.expr for guard in shape_env.guards)
        branched = {str(symbol) for guard in guards for symbol in guard.free_symbols}
        return cls(
            key=key,
            guards=guards,
            ranges={
                str(symbol): (value_range.lower, value_range.upper)
                for symbol, value_range in shape_env.var_to_range.items()
                if str(symbol) in branched
            },
        )

    def describe(self) -> str:
        """Where this graph is valid, in one line, for a record or a refusal."""
        conditions = " ".join(f"{name}={value!r}" for name, value in self.key)
        stated = [str(guard) for guard in self.guards]
        stated += [f"{name} in [{low}, {high}]" for name, (low, high) in self.ranges]
        if not stated:
            return f"{conditions}, at any size"
        return f"{conditions}, where {', '.join(stated)}"

    def decide(self, *, key: Any, sizes: Any) -> Verdict:
        """Whether this graph describes a step stating `key` at `sizes`.

        `sizes` binds symbol names to concrete sizes. Every symbol the domain
        names has to be bound: a graph whose domain mentions a size the step
        cannot state has not been shown to describe it.

        The guards are taken first because a guard names the branch the
        tracing run actually took, which is the more useful thing to be told;
        the ranges then catch what no branch asked about.
        """
        offered = _as_key(key)
        if offered != self.key:
            return Verdict(False, _key_refusal(dict(self.key), dict(offered)))
        bindings = _as_bindings(sizes)
        for guard in self.guards:
            symbols = {str(symbol): symbol for symbol in guard.free_symbols}
            unbound = sorted(name for name in symbols if name not in bindings)
            if unbound:
                return Verdict(
                    False,
                    f"guard `{guard}` is over {unbound}, which the step does "
                    f"not bind; it binds {sorted(bindings)}",
                )
            resolved = guard.subs(
                {symbol: bindings[name] for name, symbol in symbols.items()}
            )
            if getattr(resolved, "free_symbols", frozenset()):
                return Verdict(
                    False,
                    f"guard `{guard}` did not resolve to true or false under "
                    f"{_shown(bindings, symbols)}; it left `{resolved}`",
                )
            if not bool(resolved):
                return Verdict(
                    False,
                    f"guard `{guard}` is false at {_shown(bindings, symbols)}",
                )
        for name, (low, high) in self.ranges:
            if name not in bindings:
                return Verdict(
                    False,
                    f"the record holds {name} in [{low}, {high}] and the step "
                    f"binds {sorted(bindings)}, which does not include {name}",
                )
            size = bindings[name]
            if not (bool(low <= size) and bool(size <= high)):
                return Verdict(
                    False,
                    f"the record holds {name} in [{low}, {high}]; the step "
                    f"binds {name}={size}",
                )
        return Verdict(True)


def _shown(bindings: dict[str, int], symbols: Any) -> str:
    """The bindings a guard was evaluated at, named and in order."""
    return ", ".join(f"{name}={bindings[name]}" for name in sorted(symbols))


def _key_refusal(recorded: dict[str, Any], offered: dict[str, Any]) -> str:
    """Why two discrete keys are not the same key."""
    parts = []
    absent = sorted(set(recorded) - set(offered))
    extra = sorted(set(offered) - set(recorded))
    if absent:
        parts.append(f"the step states nothing about {absent}")
    if extra:
        parts.append(f"the record states nothing about {extra}")
    parts += [
        f"{name} was recorded as {recorded[name]!r} and the step states "
        f"{offered[name]!r}"
        for name in sorted(set(recorded) & set(offered))
        if recorded[name] != offered[name]
    ]
    return "; ".join(parts)


def _as_key(key: Any) -> tuple[tuple[str, Any], ...]:
    """Normalise a discrete key to a sorted tuple of pairs.

    Sorted, so two statements of one key compare equal whatever order they were
    written in. Values drawn from a stated list of types, for the reason below.
    And not empty: a key naming no condition is a graph claiming to describe
    every forward mode, every backend and the dummy runs, which is the claim
    the key exists to stop being made by default.
    """
    if isinstance(key, (str, bytes)):
        raise TypeError(f"a key is named conditions and their values, got {key!r}")
    items = key.items() if hasattr(key, "items") else key
    pairs = []
    for name, value in items:
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"a key names its conditions; got {name!r}")
        if not isinstance(value, _CONDITION_TYPES):
            raise TypeError(
                f"the value of {name!r} is a {type(value).__name__}. A "
                "condition matched by equality is a bool, an int, a str, an "
                "enum member or None. Nothing else is admitted, because the "
                "alternative is to ask the value whether it behaves -- and on "
                "a symbolic boolean that question is answered by installing a "
                "guard, which would make the check itself the thing that "
                "changed the record."
            )
        pairs.append((name, value))
    if not pairs:
        raise ValueError(
            "a key with no conditions in it admits every step -- every forward "
            "mode, every backend, the dummy runs -- which is what the key is for"
        )
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"a condition is stated twice: {sorted(names)}")
    return tuple(sorted(pairs))


def _as_guards(guards: Any) -> tuple[Any, ...]:
    """Check each guard can be substituted into and can name its symbols."""
    if isinstance(guards, (str, bytes)) or not hasattr(guards, "__iter__"):
        raise TypeError(f"guards are a sequence, got {type(guards).__name__}")
    checked = tuple(guards)
    for guard in checked:
        if not hasattr(guard, "subs") or not hasattr(guard, "free_symbols"):
            raise TypeError(
                "a guard is an expression that can report its free symbols and "
                f"substitute them, got {type(guard).__name__}: {guard!r}"
            )
    return checked


def _as_ranges(ranges: Any) -> tuple[tuple[str, tuple[Any, Any]], ...]:
    """Normalise symbol ranges to a sorted tuple of name/bounds pairs."""
    if isinstance(ranges, (str, bytes)):
        raise TypeError(f"ranges are named bounds, got {ranges!r}")
    items = ranges.items() if hasattr(ranges, "items") else ranges
    pairs = []
    for name, bounds in items:
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"a range names its symbol; got {name!r}")
        low, high = bounds
        pairs.append((name, (low, high)))
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"a symbol is bounded twice: {sorted(names)}")
    return tuple(sorted(pairs, key=lambda pair: pair[0]))


def _as_bindings(sizes: Any) -> dict[str, int]:
    """Normalise a step's sizes, refusing one that is still symbolic.

    The refusal is the rule this module is built around. A size that is still a
    symbol resolves only by asking for it -- `int()` on it, or comparing it,
    installs a guard on the tracing run, and the act of deciding a step would
    then narrow the graph it was deciding against. The tracing run's own hint
    lookup answers without recording anything, so the caller resolves the size
    there and offers the number.
    """
    if isinstance(sizes, (str, bytes)):
        raise TypeError(f"sizes are named sizes, got {sizes!r}")
    items = sizes.items() if hasattr(sizes, "items") else sizes
    bindings: dict[str, int] = {}
    for name, size in items:
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"a size names its symbol; got {name!r}")
        if name in bindings:
            raise ValueError(f"{name!r} is bound twice")
        if isinstance(size, bool) or is_symbolic(size):
            raise TypeError(
                f"{name} was offered as {type(size).__name__}; a size is a "
                "concrete int, and a type that merely stands in for one is not "
                "the same thing. If it is a size that is still a symbol, ask "
                "the tracing run for its hint and offer that int: converting "
                "or comparing the symbol here installs a guard, so deciding a "
                "step would change the record it is being decided against."
            )
        bindings[name] = size
    return bindings
