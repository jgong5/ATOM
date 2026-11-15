# SPDX-License-Identifier: MIT
"""The invalidation matrix, as a table rather than a chain of conditionals.

A global fingerprint would force re-measuring everything on a torch bump. Each
artifact records the fingerprint of **its own dependency row**, so a bump
invalidates the rows that name it and leaves the rest alone.

`MATRIX` below holds the dependency rows, and it is deliberately a table: the
deliverable is that a reader can check it cell by cell.

**A uniform matrix has lost its point**, and this cell is the one a tidying
hand would take away:

* `price_list` does **not** depend on the model. Its leaves are
  shape-parametric, and that is exactly what lets one campaign serve many
  shapes. A row that also depended on the model would make every new model a
  new pricing campaign.

The host CPU invalidates `host_laws`, which time the host, and nothing that
holds device time. A `shape_population`'s pricing backend is in its key rather
than its row: another backend addresses another entry instead of invalidating
this one.
"""

import enum
from collections.abc import Mapping
from dataclasses import dataclass

from .rules import ArtifactRefusal, Rule


class Kind(enum.Enum):
    """The artifacts this store holds. There are no others."""

    SHAPE_POPULATION = "shape_population"
    OP_GRAPH = "op_graph"
    OP_MEASUREMENT = "op_measurement"
    PRICE_LIST = "price_list"
    MEMORY_READINGS = "memory_readings"
    STEP_MEASUREMENT = "step_measurement"
    STEP_LINEAR_LAW = "step_linear_law"
    HOST_LAWS = "host_laws"
    STEP_GRID_BUNDLE = "step_grid_bundle"
    PROJECTION = "projection"
    PROBE_RECORD = "probe_record"
    PLATFORM_READINGS = "platform_readings"

    def __str__(self) -> str:
        return self.value


class Axis(enum.Enum):
    """The things a change to which can invalidate an artifact.

    The value is the document's column header verbatim so a refusal quotes it;
    `field` is the stable name the same axis is recorded and asked for under.
    """

    SOFTWARE_STACK = "ROCm / AITER / RCCL"
    TORCH = "torch"
    ATOM_SRC = "ATOM src"
    MODEL = "model"
    DEVICE = "device"
    PTL = "PTL"
    HOST_CPU = "host CPU"
    ENGINE_CONFIG = "engine cfg"

    @property
    def field(self) -> str:
        """The axis as it is written in JSON and passed to `Conditions.of`."""
        return self.name.lower()

    def __str__(self) -> str:
        return self.value


class Row(enum.Enum):
    OP_GRAPH = "op_graph"
    PRICE_LIST = "price_list"
    MEMORY_READINGS = "memory_readings"
    STEP_MEASUREMENT = "step_measurement"
    STEP_LINEAR_LAW = "step_linear_law"
    HOST_LAWS = "host_laws"
    STEP_GRID_BUNDLE = "step grid bundle"
    SHAPE_POPULATION = "shape_population"
    PROJECTION = "projection"
    PROBE_RECORD = "probe_record"

    @property
    def field(self) -> str:
        """The row as it is written in JSON."""
        return self.name.lower()

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Cell:
    """One cell of the table: the mark, and the parenthesis beside it.

    The note is carried rather than dropped because each row that has one
    says something the mark alone does not -- which part of the engine config
    an `op_graph` turns on, why a `price_list` is free of the model, and what
    a `probe_record` reads torch through.
    """

    depends: bool
    note: str = ""

    def __str__(self) -> str:
        mark = "X" if self.depends else "-"
        return f"{mark} ({self.note})" if self.note else mark


#: The document's `X` and `-`, so the table below transcribes rather than translates.
DEPENDS = Cell(True)
INDEPENDENT = Cell(False)

#: The invalidation matrix, row by row and column by column.
MATRIX: Mapping[Row, Mapping[Axis, Cell]] = {
    Row.OP_GRAPH: {
        Axis.SOFTWARE_STACK: INDEPENDENT,
        Axis.TORCH: DEPENDS,
        Axis.ATOM_SRC: DEPENDS,
        Axis.MODEL: DEPENDS,
        Axis.DEVICE: INDEPENDENT,
        Axis.PTL: INDEPENDENT,
        Axis.HOST_CPU: INDEPENDENT,
        Axis.ENGINE_CONFIG: Cell(True, "level, cudagraph mode"),
    },
    Row.PRICE_LIST: {
        Axis.SOFTWARE_STACK: DEPENDS,
        Axis.TORCH: DEPENDS,
        Axis.ATOM_SRC: DEPENDS,
        Axis.MODEL: Cell(False, "shape-parametric"),
        Axis.DEVICE: DEPENDS,
        Axis.PTL: DEPENDS,
        Axis.HOST_CPU: INDEPENDENT,
        Axis.ENGINE_CONFIG: INDEPENDENT,
    },
    Row.MEMORY_READINGS: {
        Axis.SOFTWARE_STACK: DEPENDS,
        Axis.TORCH: DEPENDS,
        Axis.ATOM_SRC: DEPENDS,
        Axis.MODEL: DEPENDS,
        Axis.DEVICE: DEPENDS,
        Axis.PTL: DEPENDS,
        Axis.HOST_CPU: INDEPENDENT,
        Axis.ENGINE_CONFIG: DEPENDS,
    },
    Row.STEP_MEASUREMENT: {
        Axis.SOFTWARE_STACK: DEPENDS,
        Axis.TORCH: DEPENDS,
        Axis.ATOM_SRC: DEPENDS,
        Axis.MODEL: DEPENDS,
        Axis.DEVICE: DEPENDS,
        Axis.PTL: DEPENDS,
        Axis.HOST_CPU: INDEPENDENT,
        Axis.ENGINE_CONFIG: DEPENDS,
    },
    Row.STEP_LINEAR_LAW: {
        Axis.SOFTWARE_STACK: DEPENDS,
        Axis.TORCH: DEPENDS,
        Axis.ATOM_SRC: DEPENDS,
        Axis.MODEL: DEPENDS,
        Axis.DEVICE: DEPENDS,
        Axis.PTL: DEPENDS,
        Axis.HOST_CPU: INDEPENDENT,
        Axis.ENGINE_CONFIG: DEPENDS,
    },
    Row.HOST_LAWS: {
        Axis.SOFTWARE_STACK: DEPENDS,
        Axis.TORCH: DEPENDS,
        Axis.ATOM_SRC: DEPENDS,
        Axis.MODEL: DEPENDS,
        Axis.DEVICE: DEPENDS,
        Axis.PTL: INDEPENDENT,
        Axis.HOST_CPU: DEPENDS,
        Axis.ENGINE_CONFIG: DEPENDS,
    },
    Row.STEP_GRID_BUNDLE: {
        Axis.SOFTWARE_STACK: DEPENDS,
        Axis.TORCH: DEPENDS,
        Axis.ATOM_SRC: DEPENDS,
        Axis.MODEL: DEPENDS,
        Axis.DEVICE: DEPENDS,
        Axis.PTL: DEPENDS,
        Axis.HOST_CPU: INDEPENDENT,
        Axis.ENGINE_CONFIG: DEPENDS,
    },
    Row.SHAPE_POPULATION: {
        Axis.SOFTWARE_STACK: INDEPENDENT,
        Axis.TORCH: INDEPENDENT,
        Axis.ATOM_SRC: DEPENDS,
        Axis.MODEL: DEPENDS,
        Axis.DEVICE: INDEPENDENT,
        Axis.PTL: INDEPENDENT,
        Axis.HOST_CPU: INDEPENDENT,
        Axis.ENGINE_CONFIG: DEPENDS,
    },
    Row.PROJECTION: {
        Axis.SOFTWARE_STACK: Cell(False, "analytical: no kernel timed"),
        Axis.TORCH: Cell(True, "op_graph capture"),
        Axis.ATOM_SRC: DEPENDS,
        Axis.MODEL: DEPENDS,
        Axis.DEVICE: Cell(False, "the spec priced is in the key"),
        Axis.PTL: INDEPENDENT,
        Axis.HOST_CPU: INDEPENDENT,
        Axis.ENGINE_CONFIG: DEPENDS,
    },
    # Vendor kernels timed on the cards; no model, engine or ATOM engine code.
    Row.PROBE_RECORD: {
        Axis.SOFTWARE_STACK: DEPENDS,
        Axis.TORCH: Cell(True, "Triton, torch._scaled_mm"),
        Axis.ATOM_SRC: INDEPENDENT,
        Axis.MODEL: INDEPENDENT,
        Axis.DEVICE: DEPENDS,
        Axis.PTL: DEPENDS,
        Axis.HOST_CPU: INDEPENDENT,
        Axis.ENGINE_CONFIG: INDEPENDENT,
    },
}

#: Which rows decide a kind's validity.
ROWS_OF: Mapping[Kind, tuple[Row, ...]] = {
    Kind.OP_GRAPH: (Row.OP_GRAPH,),
    Kind.PRICE_LIST: (Row.PRICE_LIST,),
    Kind.OP_MEASUREMENT: (Row.PRICE_LIST,),
    Kind.MEMORY_READINGS: (Row.MEMORY_READINGS,),
    Kind.STEP_MEASUREMENT: (Row.STEP_MEASUREMENT,),
    Kind.STEP_LINEAR_LAW: (Row.STEP_LINEAR_LAW,),
    Kind.HOST_LAWS: (Row.HOST_LAWS,),
    Kind.STEP_GRID_BUNDLE: (Row.STEP_GRID_BUNDLE,),
    Kind.SHAPE_POPULATION: (Row.SHAPE_POPULATION,),
    Kind.PROJECTION: (Row.PROJECTION,),
    Kind.PROBE_RECORD: (Row.PROBE_RECORD,),
    # Folded from probe records, or another device's entry: the records' row.
    Kind.PLATFORM_READINGS: (Row.PROBE_RECORD,),
}


def rows_for(kind: Kind) -> tuple[Row, ...]:
    """The rows of the matrix a kind is checked against, or a refusal."""
    rows = ROWS_OF.get(kind, ())
    if not rows:
        raise ArtifactRefusal(
            Rule.INVALIDATED,
            f"the invalidation matrix states no dependency row for {kind}",
            "the artifact key table and the matrix do not hold the same "
            f"artifacts: {kind} is in the first and not the second. Add a row "
            "to the matrix, or say there why the kind needs none; a row "
            "guessed here would answer under conditions nobody checked",
        )
    return rows


def axes_of(row: Row) -> tuple[Axis, ...]:
    """The axes a row depends on, in the matrix's column order."""
    return tuple(axis for axis in Axis if MATRIX[row][axis].depends)
