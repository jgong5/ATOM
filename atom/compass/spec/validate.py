# SPDX-License-Identifier: MIT
"""What a spec is checked for, and why every reason is reported at once.

A spec is authored by hand against a schema most of whose terms cannot be looked
up anywhere, so the document a person first writes is nearly always incomplete.
The whole value of checking it is that an incomplete one is refused rather than
quietly completed: a term nobody measured has no default, because a default here
is a guess with a number's authority, and it will be discovered by a run rather
than by the person who could still go and measure it.

Refusals are collected, not raised one at a time. Reading a document stops at
the first thing wrong with it, which is right for a reader that has to produce a
value and wrong for a check whose reader is filling in a form: told one missing
field per attempt, they measure one term, run again, and find the next. So the
answer is the list, in the schema's own order, and a caller that wants an
exception raises the first.

The check runs in two phases and cannot be collapsed into one. The first asks
whether the document is a complete instance of the schema -- every required
field present and every value the shape its field declares. The second asks
whether a complete spec is consistent: whether the widths the deployment will
actually use were measured, whether the stack the constants were taken on is the
stack now loaded, and whether constants carried over from another spec came from
one pinned to the same stack. Consistency questions are asked of a resolved
spec, so they wait until there is one; when the first phase has refusals the
second has nothing to run against, and saying so is more honest than checking
half of it.

Two of those questions need something the document does not carry.

**The widths a deployment will use are ATOM's, not the spec's.** The spec
describes a machine and says nothing about how the engine was launched, so the
widths come in as an argument, from the engine's own configuration. A spec that
measured widths 1 and 2 is complete and correct, and is still the wrong spec for
a run at width 8 -- which is a fact about the pair, not about the document.

**A transferred constant is checked against its source's stack pin.** The
fragment that carried it declares the stack the source spec was pinned to, which
is why the merge keeps that declaration aside instead of contributing it. Here
the two are compared, and a transfer out of a differently-pinned spec is refused
by name. A transfer that declares no stack at all is refused too: the evidence
that these constants move with the compute stack is the whole reason a transfer
is allowed to be cheap, and a transfer that does not say what it was measured
against cannot be checked by anyone, ever.

A stack mismatch on the running machine is the one finding that is not fatal by
default. The numbers are still measurements, taken on a stack that has since
moved, and whether that matters is a judgement about how far it moved -- so it
warns and names both versions, and refuses only when the caller asks for that.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .fields import PINNED, SCHEMA, Kind, check
from .machine import MachineSpec, _missing, _walk
from .merge import PINNED_BLOCK, Merge
from .rules import Rule, SpecRefusal


@dataclass(frozen=True, slots=True)
class Validation:
    """Everything wrong with a spec, rather than the first thing wrong with it."""

    spec: MachineSpec | None
    refusals: tuple[SpecRefusal, ...]
    stack_differences: tuple[tuple[str, str, Any], ...]

    @property
    def ok(self) -> bool:
        """Whether the spec may be used, which is not whether it was silent."""
        return not self.refusals

    def raise_first(self) -> MachineSpec:
        """The resolved spec, or the first refusal, for a caller that wants one."""
        if self.refusals:
            raise self.refusals[0]
        return self.spec


def _complete(document: object):
    if not isinstance(document, Mapping):
        yield SpecRefusal(
            Rule.SHAPE,
            f"a spec is a mapping of its sections, not {type(document).__name__}",
            "load the document before checking it as a spec",
        )
        return
    found: dict[str, Any] = {}
    try:
        _walk(document, "", found)
    except SpecRefusal as refusal:
        yield refusal
        return
    for field in SCHEMA:
        try:
            if field.path not in found:
                if field.required:
                    _missing(field)
            else:
                check(field, found[field.path], field.path)
        except SpecRefusal as refusal:
            yield refusal


def _widths(spec: MachineSpec, tp_widths: Sequence[int]):
    for field in SCHEMA:
        if field.kind is not Kind.WIDTH_TABLE:
            continue
        for width in tp_widths:
            try:
                spec.runtime_constant(field.path.rsplit(".", 1)[-1], width)
            except SpecRefusal as refusal:
                yield refusal


def _transfers(spec: MachineSpec, merged: Merge):
    for fragment in merged.transfers:
        declared = {
            component: fragment.values[f"{PINNED_BLOCK}.{component}"]
            for component in PINNED
            if f"{PINNED_BLOCK}.{component}" in fragment.values
        }
        if not declared:
            yield SpecRefusal(
                Rule.PINNED_STACK,
                f"{fragment.stanza()} carried constants over from "
                f"{fragment.transferred_from!r} without saying which stack they "
                "were measured against",
                "a transfer carries both specs' stack pins; these constants "
                "move with the compute stack, so one that cannot name its own "
                "stack cannot be checked against this machine's",
            )
            continue
        for component, version in declared.items():
            here = spec.value(f"{PINNED_BLOCK}.{component}")
            if version != here:
                yield SpecRefusal(
                    Rule.PINNED_STACK,
                    f"{fragment.stanza()} carried constants over from "
                    f"{fragment.transferred_from!r}, measured against "
                    f"{component} {version!r}, into a spec pinned to "
                    f"{component} {here!r}",
                    "transfer from a spec pinned to this stack, or measure the "
                    "constants on it; the untested assumption is transfer "
                    "within a software generation, not across one",
                )


def validate(
    subject: object,
    *,
    tp_widths: Sequence[int] = (),
    observed_stack: Mapping[str, str] | None = None,
    strict: bool = False,
) -> Validation:
    """Check a document, or a merge of fragments, and report every refusal."""
    merged = subject if isinstance(subject, Merge) else None
    document = merged.document if merged is not None else subject
    refusals = list(_complete(document))
    spec = None
    if not refusals:
        try:
            spec = MachineSpec.from_mapping(document)
        except SpecRefusal as refusal:
            refusals.append(refusal)
    if spec is None:
        return Validation(None, tuple(refusals), ())
    refusals += _widths(spec, tp_widths)
    differences = ()
    if observed_stack is not None:
        differences = spec.check_stack(observed_stack)
        if differences and strict:
            refusals.append(
                SpecRefusal(
                    Rule.PINNED_STACK,
                    "this spec's constants were measured against "
                    + ", ".join(
                        f"{component} {pinned!r} (now {seen!r})"
                        for component, pinned, seen in differences
                    ),
                    "measure them on the stack that is loaded, or drop the "
                    "strict check and take responsibility for the difference",
                )
            )
    if merged is not None:
        refusals += _transfers(spec, merged)
    return Validation(spec, tuple(refusals), differences)
