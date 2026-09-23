# SPDX-License-Identifier: MIT
"""The machine spec: one document read against the field table, and its readers.

A spec is authored outside Compass, so the only thing that stands between a
mistyped document and a run built on it is this reader. It holds the document as
a flat map from dotted path to checked value. That shape is deliberate and it is
chosen for what comes next: combining two documents, reporting on one, and
explaining where a number came from are all per-field operations, and a nested
mapping makes every one of them a recursive walk written again each time.

Four properties are enforced here rather than described.

**The schema is closed.** A key with no row in the field table is refused where
it sits. That, and not a list of excluded knobs, is how the spec stays a
description of the machine: a tensor-parallel width or a block size is refused
because it is not a machine property, with no rule needed that names it.

**A runtime constant has no default.** The widths that were measured are the
widths that can be asked for; a width that was not measured is refused by name,
because these terms fit no law and there is nothing to interpolate along.

**A stack pin is checked.** `check_stack` compares the versions the constants
were measured against with the versions now loaded and warns, naming both, for
every component that differs. The evidence says these constants track the
compute stack at least as much as the die, so a spec transferred across a stack
upgrade is a plausible-looking wrong answer.

**The resolved spec is total, and it is echoed.** `echo` rebuilds the whole
document from the checked values, and `digest` fingerprints it. A run artifact
carries the echo, so every number it reports can be traced back to the spec that
produced it; a partial echo would break that, so the echo is built from the
field table rather than from whatever the reader happened to keep. Totality is
`from_mapping`'s doing rather than the dataclass's: the checking verb builds one
out of the fields a document did resolve, deliberately, so that it can ask its
consistency questions of a document that is not yet a spec.

The walk over the document has two forms for the same reason. `from_mapping` is
a reader and raises the first thing wrong; `_survey` under it yields every
refusal and keeps walking, so a checking caller still gets the fields that sit
beside a mistyped key. One unrecognised key would otherwise empty the whole
document of resolved values, and the questions asked over them with it.
"""

import hashlib
import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .fields import (
    BLOCKS,
    BY_PATH,
    PINNED,
    RUNTIME_CONSTANTS,
    SCHEMA,
    SCHEMA_VERSION,
    Field,
    Kind,
    check,
)
from .rules import Rule, SpecRefusal, StackMismatch, refuse_unknown_key
from .tokenizers import Backend, TokenizerEntry, TokenizerTable, table


def _missing(field: Field) -> None:
    if field.path.startswith(RUNTIME_CONSTANTS):
        raise SpecRefusal(
            Rule.NO_DEFAULTS,
            f"`{field.path}` is missing",
            "measure it on the engine and write it down; there is no law "
            "behind this term, so a default would be a guess wearing a number",
        )
    if field.kind is Kind.DERATE:
        raise SpecRefusal(
            Rule.DERATE,
            f"`{field.path}` is missing, and its block states a spec peak",
            "declare the fraction of that peak a kernel actually reaches, so "
            "nobody spends a datasheet number as an achievable one",
        )
    raise SpecRefusal(
        Rule.SHAPE,
        f"`{field.path}` is missing",
        "the spec describes one machine completely; a term left out would be "
        "discovered by a run rather than by the person writing the document",
    )


def _survey(node: Mapping, prefix: str, found: dict):
    """Every key of a document, keeping what the table declares and yielding a
    refusal for each key that it does not.

    The walk carries on past a key it refuses, because one unrecognised key
    says nothing about the rest of the document: the block beside it holds the
    width tables and the stack pin, and stopping here would take those out of
    reach of every question asked further on. A reader that has to produce a
    value wants the first refusal and `_walk` below raises it; a check wants
    them all.
    """
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if path in BY_PATH:
            found[path] = value
        elif path in BLOCKS:
            if isinstance(value, Mapping):
                yield from _survey(value, path, found)
            else:
                yield SpecRefusal(
                    Rule.SHAPE,
                    f"`{path}` holds {value!r}, where the schema has a block",
                    "write the block's fields there",
                )
        else:
            try:
                refuse_unknown_key(path)
            except SpecRefusal as refusal:
                yield refusal


def _walk(node: Mapping, prefix: str, found: dict) -> None:
    for refusal in _survey(node, prefix, found):
        raise refusal


@dataclass(frozen=True, slots=True)
class MachineSpec:
    """One machine, as a document checked against the schema."""

    values: Mapping[str, Any]
    tokenizers: TokenizerTable

    @classmethod
    def from_mapping(cls, document: object) -> "MachineSpec":
        """Read a document, or refuse with the rule that declined and the fix."""
        if not isinstance(document, Mapping):
            raise SpecRefusal(
                Rule.SHAPE,
                f"a spec is a mapping of its sections, not {type(document).__name__}",
                "load the document before reading it as a spec",
            )
        found: dict[str, Any] = {}
        _walk(document, "", found)
        values = {}
        for field in SCHEMA:
            if field.path not in found:
                if field.required:
                    _missing(field)
                continue
            values[field.path] = check(field, found[field.path], field.path)
        version = values["schema_version"]
        if version != SCHEMA_VERSION:
            raise SpecRefusal(
                Rule.SHAPE,
                f"this document states schema_version {version}",
                f"this reader understands version {SCHEMA_VERSION}",
            )
        return cls(values, table(values["host.tokenizers"]))

    def value(self, path: str) -> Any:
        """One checked field by its dotted path."""
        if path not in self.values:
            refuse_unknown_key(path)
        return self.values[path]

    def echo(self) -> dict:
        """The whole resolved spec, for a run artifact to carry verbatim."""
        document: dict = {}
        for field in SCHEMA:
            if field.path not in self.values:
                continue
            *blocks, leaf = field.path.split(".")
            node = document
            for block in blocks:
                node = node.setdefault(block, {})
            value = self.values[field.path]
            node[leaf] = list(value) if isinstance(value, tuple) else value
        return document

    def digest(self) -> str:
        """A fingerprint of the echo, so an artifact can name the spec it used."""
        canonical = json.dumps(self.echo(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    def runtime_constant(self, name: str, tp_width: int | None = None) -> Any:
        """A measured constant, refusing a width nobody measured."""
        path = f"{RUNTIME_CONSTANTS}.{name}"
        field = BY_PATH.get(path)
        if field is None:
            refuse_unknown_key(path)
        if field.kind is not Kind.WIDTH_TABLE:
            return self.values[path]
        measured = self.values[path]
        if tp_width is None:
            raise SpecRefusal(
                Rule.NO_DEFAULTS,
                f"`{name}` is measured per tensor-parallel width and none was given",
                f"ask for one of the measured widths {sorted(measured)}",
            )
        if tp_width not in measured:
            raise SpecRefusal(
                Rule.NO_DEFAULTS,
                f"`{name}` was not measured at tensor-parallel width {tp_width}; "
                f"this spec has {sorted(measured)}",
                "measure the engine at that width and add the entry; the "
                "measured widths fit no fixed-plus-per-peer form, so there is "
                "nothing here to interpolate along",
            )
        return measured[tp_width]

    def tokenizer_for(
        self, architecture: str, backend: Backend, fingerprint: str | None = None
    ) -> TokenizerEntry:
        """The measured tokenizer for a model architecture, or a refusal."""
        return self.tokenizers.resolve(architecture, backend, fingerprint)

    def check_stack(self, observed: Mapping[str, str]) -> tuple:
        """Compare the pinned stack with the loaded one; warn on every difference.

        A component this document does not carry is not compared, so a partial
        one can still be asked this much of the question.
        """
        found = []
        for component in PINNED:
            pinned = self.values.get(f"device.software_pinned_to.{component}")
            if pinned is None:
                continue
            seen = observed.get(component)
            if seen != pinned:
                found.append((component, pinned, seen))
        differences = tuple(found)
        if differences:
            warnings.warn(
                "this spec's constants were measured against "
                + ", ".join(
                    f"{component} {pinned} (now {seen!r})"
                    for component, pinned, seen in differences
                )
                + "; they track the compute stack as much as the die",
                StackMismatch,
                stacklevel=2,
            )
        return differences
