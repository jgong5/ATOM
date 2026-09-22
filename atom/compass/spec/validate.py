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

The check has two halves. The first asks whether the document is a complete
instance of the schema -- every required field present and every value the shape
its field declares. The second asks whether what it describes is consistent:
whether the widths the deployment will actually use were measured, whether the
stack the constants were taken on is the stack now loaded, and whether constants
carried over from another spec came from one pinned to the same stack.

**The second half is asked of whatever resolved, and not only of a complete
spec.** Those questions are not intrinsically about a resolved document, only
their implementation was: both width tables are present and schema-valid in a
document that the first half refuses for a single missing derate, and that
derate is a value its author types in at a desk. Running the second half only
when the first came back clear therefore hid the expensive refusal -- a width
nobody measured, which costs an eight-GPU reservation to fix -- behind the cheap
one, and reintroduced at the halfway line exactly the one-refusal-per-attempt
this module opens by rejecting. So each question is asked of the fields that did
check out, and a question that reads a field which did not resolve says so by
name rather than being passed over in silence.

**The same record makes the opt-in conditions legible.** Three of the five in
`CONDITIONS` need something from the caller -- `tp_widths=`, `observed_stack=`,
and a `Merge` rather than a document -- and a clear result that does not say what
it declined to ask is the shape this package exists to refuse. The sharpest case
is the transfer: its source's pin is in no field of the merged document, by the
decision below, so `validate(document)` can never ask that condition however the
document was built, and the same spec is refused as a `Merge` and clear as a
document. That is not a wrong number, but it must be visible, because the verb
the design writes -- `compass spec validate machine.yaml` -- is the form that
cannot ask it, and so reaches four of the five. `CONDITIONS` is what that count
is a count of; a condition added to the check set and not to it is one no result
can report on.

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
from .tokenizers import TokenizerTable

#: The conditions a spec is checked for. The first two are asked of any
#: document; each of the rest names itself in `not_asked` when a run could not
#: reach it.
MISSING = "whether every required field is present and every value its declared shape"
DERATES = "whether a spec-peak number carries the derate it obliges"
WIDTHS = "whether the widths this deployment will use were measured"
STACK = "whether the constants' stack pin is the stack now loaded"
TRANSFERS = "whether a transferred constant came from a spec pinned to this stack"
#: The whole check set, in the order a run asks it. What `not_asked` is measured
#: against: a condition missing from here is one no result can report on.
CONDITIONS = (MISSING, DERATES, WIDTHS, STACK, TRANSFERS)
#: The fields each consistency question reads before it can be asked at all.
WIDTH_TABLES = tuple(field.path for field in SCHEMA if field.kind is Kind.WIDTH_TABLE)
STACK_PINS = tuple(f"{PINNED_BLOCK}.{component}" for component in PINNED)


@dataclass(frozen=True, slots=True)
class Validation:
    """Everything wrong with a spec, rather than the first thing wrong with it."""

    spec: MachineSpec | None
    refusals: tuple[SpecRefusal, ...]
    stack_differences: tuple[tuple[str, str, Any], ...]
    not_asked: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the spec may be used, which is not whether it was silent."""
        return not self.refusals

    def raise_first(self) -> MachineSpec:
        """The resolved spec, or the first refusal, for a caller that wants one."""
        if self.refusals:
            raise self.refusals[0]
        return self.spec

    def __str__(self) -> str:
        """The verdict, every reason for it, every stack difference it found,
        and every question left unasked."""
        return "\n".join(
            [f"{'ok' if self.ok else 'refused'}: {len(self.refusals)} refusal(s)"]
            + [f"  {refusal}" for refusal in self.refusals]
            + [
                f"  stack moved: the constants were measured against "
                f"{component} {pinned!r} (now {seen!r})"
                for component, pinned, seen in self.stack_differences
            ]
            + [f"  not asked: {condition}" for condition in self.not_asked]
        )


def _unreached(
    condition: str, reads: Sequence[str], resolved: Mapping[str, Any]
) -> tuple[str, ...]:
    """What a condition was not asked of, because a field it reads did not resolve."""
    absent = [path for path in reads if path not in resolved]
    if not absent:
        return ()
    named = ", ".join(f"`{path}`" for path in absent)
    reach = (
        "so it could not be asked at all"
        if len(absent) == len(reads)
        else "so it was asked of the rest and not of those"
    )
    return (f"{condition} -- {named} did not resolve, {reach}",)


def _not_asked(
    resolved: Mapping[str, Any],
    tp_widths: Sequence[int],
    observed_stack: Mapping[str, str] | None,
    merged: Merge | None,
) -> tuple[str, ...]:
    """The consistency questions this run did not ask, each with its reason."""
    unasked: list[str] = []
    if not tp_widths:
        unasked.append(f"{WIDTHS} -- no `tp_widths=` was given")
    else:
        unasked.extend(_unreached(WIDTHS, WIDTH_TABLES, resolved))
    if observed_stack is None:
        unasked.append(f"{STACK} -- no `observed_stack=` was given")
    else:
        unasked.extend(_unreached(STACK, STACK_PINS, resolved))
    if merged is None:
        unasked.append(
            f"{TRANSFERS} -- the subject is a document, and a transfer's source "
            "pin is in no field of one; ask this of the `Merge` while the "
            "fragments are still in hand"
        )
    elif merged.transfers:
        unasked.extend(_unreached(TRANSFERS, STACK_PINS, resolved))
    return tuple(unasked)


def _complete(document: object, resolved: dict[str, Any]):
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
                resolved[field.path] = check(field, found[field.path], field.path)
        except SpecRefusal as refusal:
            yield refusal


def _widths(spec: MachineSpec, tp_widths: Sequence[int], resolved: Mapping[str, Any]):
    for path in WIDTH_TABLES:
        if path not in resolved:
            continue
        for width in tp_widths:
            try:
                spec.runtime_constant(path.rsplit(".", 1)[-1], width)
            except SpecRefusal as refusal:
                yield refusal


def _transfers(resolved: Mapping[str, Any], merged: Merge):
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
            if f"{PINNED_BLOCK}.{component}" not in resolved:
                continue
            here = resolved[f"{PINNED_BLOCK}.{component}"]
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
    resolved: dict[str, Any] = {}
    refusals = list(_complete(document, resolved))
    spec = None
    if not refusals:
        try:
            spec = MachineSpec.from_mapping(document)
        except SpecRefusal as refusal:
            refusals.append(refusal)
    # The consistency questions go through the fields that checked out, whether
    # or not they add up to a spec. Where there is no spec this stands in for
    # one, and it never leaves this function: a partial document is not a
    # machine, and `Validation.spec` must not offer it as one.
    asked_of = spec if spec is not None else MachineSpec(resolved, TokenizerTable(()))
    refusals += _widths(asked_of, tp_widths, resolved)
    differences: tuple = ()
    if observed_stack is not None:
        differences = asked_of.check_stack(observed_stack)
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
        refusals += _transfers(resolved, merged)
    return Validation(
        spec,
        tuple(refusals),
        differences,
        _not_asked(resolved, tp_widths, observed_stack, merged),
    )
