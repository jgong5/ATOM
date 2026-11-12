# SPDX-License-Identifier: MIT
"""What names an artifact: the artifact keys, as tuples that cannot be paths.

The table below is the whole of artifact identity. An entry is addressed by the
values that decide what it contains and by nothing else -- not by where someone
put the file, and not by a name a script composed. `price_list` is the row that
earns the rule: it is keyed by **(model, parallel layout, platform,
source-root digest)**, and a bare or merged price file states no layout, so it
silently prices nothing while looking layered.

The platform (`PLATFORM`) is the device, the software stack, the PTL state and
torch, each by value. A kind keys each of them its row of the invalidation
matrix marks and no other (`KEY_FIELDS`), so an entry measured on one platform
does not answer on another, a host law keys no PTL state, and nothing outside
those four moves a key.

The parallel layout (`LAYOUT`) is the rank widths (`AXES`) plus the settings
that change kernels and groups at a fixed rank count: the switches
(`SWITCHES`) and `decode_context_parallel_size`. A `price_list` or
`memory_readings` entry answers only at the layout it was
keyed at: a TP2 price list does not answer for `-tp 2 -dp 2`. A width is a
whole number of at least one and a switch is a `bool`. A `bool` is refused in
every other field, because `True == 1` would make two keys equal whose
canonical texts, and so directories, differ.

Two refusals follow from that, and between them they are why this module
exists rather than a dict of strings.

**A field the kind does not have is refused with the kind's whole tuple.** The
most likely mistake is reaching for `path=`, and the message that helps is not
"unknown field" but the whole tuple a `price_list` is keyed by.

**A value that is a path is refused where it sits.** Deterministically, without
asking the filesystem: an `os.PathLike`, an absolute path, anything with a
`..` segment, or anything whose last segment ends in a data-file suffix. A
model name like `Qwen/Qwen3-32B` contains a separator and is not a path, so the
separator alone cannot be the test.

A key renders to a directory name, and the rendering is deliberately lossy --
it is a label plus a digest of the canonical key, and the entry restates its
key inside itself. The path is a place, never the authority on what lives
there.
"""

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .matrix import MATRIX, ROWS_OF, Axis, Kind
from .naming import AXES
from .rules import ArtifactRefusal, Rule

#: The layout switches: a `bool` each, and no other field takes one.
SWITCHES = ("enable_expert_parallel", "enable_dp_attention")
#: The parallel layout a key states: the rank widths, then what changes kernels
#: and groups at a fixed rank count.
LAYOUT = (*AXES, *SWITCHES, "decode_context_parallel_size")

#: The platform axes a key states by value, in the order a key reads them.
PLATFORM = (Axis.DEVICE, Axis.SOFTWARE_STACK, Axis.PTL, Axis.TORCH)

#: What keys each artifact besides its platform, in the order a key reads.
OWN_FIELDS: Mapping[Kind, tuple[str, ...]] = {
    Kind.MACHINE_SPEC: (),
    # `workload` is the digest of the requests its steps name; `backend` the
    # pricing backend's digest, so another backend addresses another entry.
    Kind.SHAPE_POPULATION: ("model", "workload", "engine_config", "backend"),
    # The ATOM source it was traced under, and torch from its row, so a graph
    # traced under another of either is another entry.
    Kind.OP_GRAPH: ("structure", "shape", "source_root"),
    # `point` is the digest of the shape, context and leaf one measurement took.
    Kind.OP_MEASUREMENT: ("model", *LAYOUT, "source_root", "point"),
    Kind.PRICE_LIST: ("model", *LAYOUT, "source_root"),
    # The engine config as `run.memory.key_fields` reads it, as sorted JSON.
    Kind.MEMORY_READINGS: ("model", *LAYOUT, "engine_config"),
    # `view` is the digest of the batch view the step was measured as.
    Kind.STEP_MEASUREMENT: (
        "model",
        "engine_config",
        "source_root",
        "sampling",
        "step_kind",
        "view",
    ),
    Kind.STEP_LINEAR_LAW: ("model", "engine_config", "source_root", "form"),
    # The values `run.host_key` reads, the engine config as its sorted JSON
    # since a key holds only strings and numbers.
    Kind.HOST_LAWS: ("host_cpu", "atom_src", "model", "engine_config"),
    # `engine_config` names the one the grid was planned for; `grid` names
    # the grid definition with its margin.
    Kind.STEP_GRID_BUNDLE: ("model", "engine_config", "grid", "source_root"),
    # `spec` is the whole machine spec's digest, which moves with any value it
    # prices by; `cells` the digest of each cell's step records and engine.
    Kind.PROJECTION: ("model", "spec", "source_root", "cells"),
    # The node's host name, so one node publishes one record per platform.
    Kind.PROBE_RECORD: ("host",),
}

#: What keys each artifact: its own fields, then each platform axis a row of
#: the kind's marks.
KEY_FIELDS: Mapping[Kind, tuple[str, ...]] = {
    kind: (
        *own,
        *(
            axis.field
            for axis in PLATFORM
            if any(MATRIX[row][axis].depends for row in ROWS_OF[kind])
        ),
    )
    for kind, own in OWN_FIELDS.items()
}

#: The incident each kind's refusal quotes, where there is one to quote.
WHY: Mapping[Kind, str] = {
    Kind.PRICE_LIST: (
        "a bare or merged price file states no width and silently prices nothing"
    ),
    Kind.OP_GRAPH: "an op_graph holds one shape: a decode width or a prefill (T, B)",
}

#: Suffixes that make a string a file name rather than a value.
DATA_SUFFIXES = frozenset(
    {".json", ".jsonl", ".yaml", ".yml", ".csv", ".txt", ".tar", ".pt", ".parquet"}
)
_SLUG = re.compile(r"[^0-9A-Za-z]+")


def _tuple_text(kind: Kind) -> str:
    return "(" + ", ".join(KEY_FIELDS[kind]) + ")"


def _remedy(kind: Kind) -> str:
    why = WHY.get(kind)
    keyed = f"a {kind} is keyed by {_tuple_text(kind)}"
    return f"{keyed}; {why}" if why else keyed


def _looks_like_a_path(value: object) -> bool:
    if isinstance(value, os.PathLike):
        return True
    if not isinstance(value, str):
        return False
    if os.path.isabs(value) or value.startswith("~"):
        return True
    parts = value.replace("\\", "/").split("/")
    if any(part in ("..", ".") for part in parts):
        return True
    return os.path.splitext(parts[-1])[1].lower() in DATA_SUFFIXES


@dataclass(frozen=True, slots=True)
class Key:
    """One artifact's identity: its kind and the values that key it."""

    kind: Kind
    fields: tuple[tuple[str, object], ...]

    @classmethod
    def of(cls, kind: Kind, **values: object) -> "Key":
        """A key, or a refusal naming the tuple the kind is keyed by."""
        if not isinstance(kind, Kind):
            raise ArtifactRefusal(
                Rule.KEY_IS_A_TUPLE,
                f"{kind!r} is not a Kind member, and only a Kind names an artifact",
                "pass one of " + ", ".join(f"Kind.{k.name}" for k in Kind),
            )
        declared = KEY_FIELDS[kind]
        unknown = sorted(set(values) - set(declared))
        if unknown:
            raise ArtifactRefusal(
                Rule.KEY_IS_A_TUPLE,
                f"{kind} has no key field {', '.join(repr(u) for u in unknown)}",
                _remedy(kind),
            )
        missing = [name for name in declared if name not in values]
        if missing:
            raise ArtifactRefusal(
                Rule.KEY_IS_A_TUPLE,
                f"this {kind} key states no {', '.join(missing)}",
                _remedy(kind),
            )
        fields = []
        for name in declared:
            value = values[name]
            if _looks_like_a_path(value):
                raise ArtifactRefusal(
                    Rule.KEY_IS_A_TUPLE,
                    f"`{kind}.{name}` holds {str(value)!r}, which is a path",
                    _remedy(kind) + "; write the value itself there, not the "
                    "path of a file holding it -- the file can change after the "
                    "key is written, and the key would then no longer say what "
                    "the entry holds",
                )
            if name in SWITCHES:
                ok, want = isinstance(value, bool), "write True or False there"
            elif name in LAYOUT:
                ok = type(value) is int and value >= 1
                want = "write a width of at least one there"
            else:
                ok = isinstance(value, (str, int, float)) and not isinstance(
                    value, bool
                )
                ok = ok and not (isinstance(value, str) and not value.strip())
                want = "write a non-empty string or a number there"
            if not ok:
                raise ArtifactRefusal(
                    Rule.KEY_IS_A_TUPLE,
                    f"`{kind}.{name}` holds {value!r}, which a key cannot compare by",
                    want,
                )
            fields.append((name, value))
        return cls(kind, tuple(fields))

    @property
    def values(self) -> dict[str, object]:
        """The key's fields as a mapping, in declared order."""
        return dict(self.fields)

    def canonical(self) -> str:
        """The key as one line of text, which is what is digested and compared."""
        return json.dumps(
            {"kind": self.kind.value, "key": self.values},
            sort_keys=True,
            separators=(",", ":"),
        )

    def token(self) -> str:
        """A short digest of the canonical key: the part of a directory name that
        is unambiguous."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:12]

    def dirname(self) -> str:
        """Where this key's entry lives, relative to its kind's directory.

        A readable label for a person reading `ls`, and a digest so that two
        keys can never land on one directory. The label is lossy on purpose;
        the entry restates its key, so the name is never asked to be the
        authority.
        """
        label = "-".join(
            _SLUG.sub("-", str(value)).strip("-") for _, value in self.fields
        )
        return f"{label[:64].strip('-')}-{self.token()}" if label else self.token()

    def __str__(self) -> str:
        stated = ", ".join(f"{name}={value!r}" for name, value in self.fields)
        return f"{self.kind}({stated})"


#: The PTL state of a card with no `ptl` directory, and of a spec that states none.
NO_PTL = "disabled"


def platform_of(device: str, stack: Mapping[str, str], ptl: str) -> dict:
    """The platform of `device`, `<name> <arch>`, running `stack`, {component:
    version} as a spec pins it, with its cards' PTL state `ptl`, by field: the
    device, the software stack as its ROCm, AITER and RCCL in that order, the
    PTL state and torch."""
    from atom.compass.spec.fields import PINNED

    pins = " / ".join(f"{c} {stack[c]}" for c in PINNED if c != "torch")
    return dict(zip((a.field for a in PLATFORM), (device, pins, ptl, stack["torch"])))


def machine_platform(machine) -> dict:
    """The platform a `MachineSpec` states, `NO_PTL` where it states no PTL state."""
    from atom.compass.spec.fields import PINNED

    return platform_of(
        f"{machine.value('device.name')} {machine.value('device.arch')}",
        {c: machine.value(f"device.software_pinned_to.{c}") for c in PINNED},
        machine.values.get("device.ptl", NO_PTL),
    )


def keyed_platform(kind: Kind, platform: Mapping[str, object]) -> dict:
    """The fields of `platform` a `kind` key states: the axes its row marks."""
    return {a.field: platform[a.field] for a in PLATFORM if a.field in KEY_FIELDS[kind]}


def refuse_not_a_key(given: object) -> None:
    """Decline something used where a key belongs, saying what a key is."""
    raise ArtifactRefusal(
        Rule.KEY_IS_A_TUPLE,
        f"{given!r} is not a key",
        "address an entry by Key.of(kind, ...); an API that accepts a path "
        "where a key belongs reproduces the price file that stated no width",
    )
