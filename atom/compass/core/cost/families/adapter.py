"""A price library that answers an unmeasured width from measured ones.

:class:`ParametricPriceLibrary` is a :class:`PriceLibrary` that overrides
``lookup`` and nothing else, so ``PriceLibrary.body`` and ``LibraryCostOracle``
use it without being changed. It answers exactly as its base class does, except
in one case.

Where it does and does not step in
----------------------------------

``PriceLibrary.lookup`` refuses for three distinct reasons and they are not
interchangeable:

``no entry for this signature``
    Nobody priced this operator. An open question, and the only one this class
    answers.

``refused when priced: ...``
    A pricing run met this operator and declined to price it. That is a
    finding. Answering it with an interpolation would replace a known-bad case
    with a guess that looks like a price.

``priced under a different operand layout (...)``
    A measurement of this operator exists and is a measurement of a different
    arrangement of memory. Same reasoning: the library already knows something
    specific here, and it is not "unknown".

So the fallback fires behind the first reason only. The other two are returned
untouched, which is also why this class cannot be used to make the
``_fused_qk_norm_single_kernel`` stride refusal go away: whether that guard is
right is a question about the guard.

Measured and interpolated are kept apart
----------------------------------------

An interpolated record carries ``interpolated: True`` and an ``interpolation``
block naming the family, the measured points it came from with their files, the
support it was evaluated inside, and the uncertainty those measurements imply.
Its source string starts with ``interpolated://``, so it is distinguishable from
a measured source by inspection and not by convention.

That is deliberately not sufficient. ``Coverage`` counts ``priced`` against
``operators`` and calls the step complete when they match; an interpolated
record increments ``priced`` like any other, and a step summed entirely from
interpolations would report itself complete. Keeping the two apart in the
*counts* is a change to ``Coverage`` in ``library.py``, which is not this
module's file. :func:`coverage_split` computes the split from a graph so the
caller can assert on it today, and ``tests/compass/test_family_prices.py``
carries the test that fails until ``Coverage`` itself carries the third count.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from atom.compass.core.cost.families.features import (
    contract_for,
    grouping_key,
    infer_rows,
)
from atom.compass.core.cost.families.support import (
    MeasuredCurve,
    Refusal,
    RowSupport,
)
from atom.compass.core.cost.library import PriceLibrary

logger = logging.getLogger(__name__)

__all__ = ["ParametricPriceLibrary", "coverage_split", "INTERPOLATED_SCHEME"]

#: Prefix marking a source string as not a measurement.
INTERPOLATED_SCHEME = "interpolated://"

#: The one refusal a family module is allowed to answer.
_OPEN_QUESTION = "no entry for this signature"


class ParametricPriceLibrary(PriceLibrary):
    """Exact prices first; a measured curve in rows behind the open question.

    ``max_gap_ratio`` is the declared sampling density: the widest ratio
    between two adjacent measured row counts that may be interpolated across.
    It is a statement about what the evidence supports, so it is set once and
    reported, never tuned against a prediction.
    """

    def __init__(self, max_gap_ratio: float = 2.0) -> None:
        super().__init__()
        self.max_gap_ratio = max_gap_ratio
        #: signature -> the graph operator dict it was priced from. The price
        #: file gives a signature string; the structured operator is what a
        #: feature map can be read off, and only a graph supplies it.
        self._ops: dict[str, dict] = {}
        #: price file -> the row count that run was measured at
        self._rows: dict[str, int] = {}
        #: grouping key -> [(op, rows, seconds, source, kernels)]
        self._observations: dict[tuple, list] = {}
        self._curves_built = False
        #: reasons a family could not be assembled, for describe()
        self.unbuildable: dict[str, str] = {}
        #: price file -> (executed rows, scheduled tokens) where a capture
        #: bucket makes the two differ, so the padding stays visible
        self.padded: dict[str, tuple[int, int]] = {}

    # -- assembly -------------------------------------------------------

    def add(self, price_path: str, graph_path: Optional[str] = None,
            registration: Optional[str] = None) -> None:
        """Load a price file, and the graph that says what its keys mean.

        Without a graph this behaves exactly as the base class: prices are
        keyed by signature and there is no structured operator to read a
        feature off, so the file contributes nothing to any curve. That is a
        silent loss of capability rather than of correctness, so it is
        recorded.
        """
        super().add(price_path, graph_path, registration)
        if not graph_path:
            self.unbuildable[price_path] = (
                "no graph supplied, so its operators have no structure to "
                "read a feature from; exact-signature use only")
            return
        with open(graph_path, encoding="utf-8") as fh:
            graph = json.load(fh)
        reading = _traced_rows(graph)
        if isinstance(reading, tuple):
            self.unbuildable[price_path] = f"{graph_path}: {reading[1]}"
            return
        rows = reading
        scheduled = sum((graph.get("key") or {}).get("batch_signature") or ())
        if scheduled and scheduled != rows:
            # Legitimate under a capture bucket, and worth saying out loud: the
            # prices are of the padded width, not of the scheduled one.
            self.padded[price_path] = (rows, scheduled)
        self._rows[price_path] = rows
        from atom.compass.runtime.microbench import signature_of

        for op in graph.get("ops") or ():
            self._ops.setdefault(signature_of(op), op)
        self._curves_built = False

    def _build(self) -> None:
        """Group every measured price by the operator it priced."""
        if self._curves_built:
            return
        self._observations.clear()
        for sig, records in self._prices.items():
            op = self._ops.get(sig)
            if op is None:
                continue
            contract = contract_for(op.get("name", ""))
            if contract is None or contract.kind != "rows":
                continue
            for record in records:
                rows = self._rows.get(record.get("source"))
                seconds = record.get("seconds")
                if rows is None or seconds is None:
                    continue
                self._observations.setdefault(grouping_key(op), []).append(
                    (op, rows, float(seconds), record.get("source", "?"),
                     tuple(record.get("kernels") or ())))
        self._curves_built = True

    # -- lookup ---------------------------------------------------------

    def lookup(self, op: dict, topology=None, registration=None):
        record, detail = super().lookup(op, topology, registration)
        if record is not None or detail != _OPEN_QUESTION:
            # Either answered, or refused for a reason that is a finding rather
            # than a gap. Both are returned as they came.
            return record, detail
        return self._parametric(op, detail)

    def _parametric(self, op: dict, original: str):
        contract = contract_for(op.get("name", ""))
        if contract is None:
            return None, (f"{original}; and {op.get('name', '?')} has no "
                          "declared family contract, so there is no statement "
                          "of what its price may depend on")
        if contract.kind != "rows":
            missing = ", ".join(contract.unmeasured_nuisances)
            return None, (
                f"{original}; {contract.family} is parameterised by its ragged "
                "(query, history) structure and no measurement varies that at "
                "fixed " + (f"{missing}" if missing else "state") +
                ", so a price here would be an assumption about components "
                "nobody has measured")

        self._build()
        curve, verified_rows = self._curve_for(op)
        if curve is None:
            return None, (f"{original}; and no measured operator matches this "
                          "one at any width")
        support = RowSupport(curve, max_gap_ratio=self.max_gap_ratio)
        answer = support.price(verified_rows)
        if isinstance(answer, Refusal):
            return None, (f"{original}; {answer.reason} "
                          f"(measured at {curve.measured_rows})")
        if answer.basis == "measured":
            # The curve already holds this width from another file; that is a
            # measurement, and it is reported as one.
            return (dict(_record(answer, curve, verified_rows),
                         interpolated=False),
                    answer.sources[0])
        return (dict(_record(answer, curve, verified_rows), interpolated=True),
                f"{INTERPOLATED_SCHEME}{contract.family}/rows={verified_rows}")

    def _curve_for(self, op: dict):
        """The measured curve for this operator, and the width it sits at."""
        observations = self._observations.get(grouping_key(op)) or []
        matched: list = []
        rows_here = None
        for measured_op, rows, seconds, source, kernels in observations:
            candidate = infer_rows(op, measured_op, rows)
            if candidate is None:
                continue
            if rows_here is None:
                rows_here = candidate
            elif rows_here != candidate:
                # Two measured operators imply different widths for the same
                # target. They cannot both be it, and choosing would be
                # arbitrary.
                return None, None
            matched.append((rows, seconds, source, kernels))
        if rows_here is None or not matched:
            return None, None
        contract = contract_for(op.get("name", ""))
        curve = MeasuredCurve(
            template_key=repr(grouping_key(op)),
            family=op.get("name", ""),
            nuisance_spread=contract.nuisance_spread if contract else 0.0,
        )
        for rows, seconds, source, kernels in matched:
            curve.add(rows, seconds, source, kernels)
        return curve, rows_here

    def describe(self) -> str:
        self._build()
        base = super().describe()
        curves = len(self._observations)
        widths = sorted({rows for obs in self._observations.values()
                         for _, rows, _, _, _ in obs})
        note = (f"; {curves} operator groups with measured widths {widths}, "
                f"max gap ratio {self.max_gap_ratio}")
        if self.unbuildable:
            note += f"; {len(self.unbuildable)} file(s) exact-signature only"
        return base + note


def _traced_rows(graph: dict) -> Optional[int] | tuple[None, str]:
    """The width this graph's operators actually ran at.

    Three readings can be present and they are not the same number:

    ``provenance.execution.body_rows_traced``
        What the deriver recorded. Authoritative where it exists, and the
        cross-check where something else is used.

    the token-index operand of ``aten::embedding``
        How many rows the operators in this graph were handed. This is what a
        price in this file is a price *of*, which is why it is what gets used:
        a replayed step runs its padded bucket and every operator in it is
        priced at the padded width.

    ``key.batch_signature``
        How many tokens the step *scheduled*. Under a capture bucket this is
        smaller than the executed width, legitimately. It is therefore never
        used as the width -- reading it as one is exactly the actual-for-padded
        substitution that has to stay visible -- but a difference is recorded.

    A disagreement between the first two is a real conflict and refuses the
    file rather than picking one.
    """
    declared = ((graph.get("provenance") or {})
                .get("execution", {})
                .get("body_rows_traced"))
    declared = int(declared) if isinstance(declared, int) else None

    executed = None
    for op in graph.get("ops") or ():
        if op.get("name") != "aten::embedding":
            continue
        shapes = op.get("input_shapes") or ()
        if len(shapes) >= 2 and len(shapes[1]) == 1:
            executed = int(shapes[1][0])
            break

    if declared is not None and executed is not None and declared != executed:
        return None, (
            f"provenance says the body was traced over {declared} rows and its "
            f"own embedding runs {executed}; those are different widths and "
            "choosing between them would be a guess")
    rows = declared if declared is not None else executed
    if rows is None or rows <= 0:
        return None, (
            "the graph records neither provenance.execution.body_rows_traced "
            "nor an embedding whose token operand states the executed width")
    return rows


def _record(answer, curve: MeasuredCurve, rows: int) -> dict:
    """A record shaped like a measured one, plus what it actually is."""
    return {
        "seconds": answer.seconds,
        "kernels": {name: answer.seconds / max(1, len(answer.kernels))
                    for name in answer.kernels},
        "occurrences": 1,
        "name": curve.family,
        "interpolation": {
            "family": curve.family,
            "rows": rows,
            "basis": answer.basis,
            "detail": answer.detail,
            "uncertainty": answer.uncertainty,
            "measured_rows": curve.measured_rows,
            "measured_sources": sorted(set(answer.sources)),
        },
    }


def coverage_split(library: PriceLibrary, graph_blob: dict,
                   registration: Optional[str] = None) -> dict:
    """How a graph's operators divide into measured, interpolated and refused.

    ``PriceLibrary.body`` returns a ``Coverage`` that counts ``priced`` without
    asking how each price was arrived at, so a step summed entirely from
    interpolations reports itself complete. Until ``Coverage`` carries the
    distinction itself, this computes it beside the sum, in operators and in
    occurrence-weighted seconds, so a caller can assert on it.
    """
    from atom.compass.core.cost.priced import HOST_SYNC

    topology = dict(((graph_blob.get("key") or {}).get("topology") or []))
    if registration is None:
        declared = (graph_blob.get("provenance") or {}).get(
            "collective_registration_required")
        registration = declared

    split = {
        "operators": 0,
        "measured": 0, "interpolated": 0, "zero_work": 0, "refused": 0,
        "measured_seconds": 0.0, "interpolated_seconds": 0.0,
        "interpolated_families": {},
        "refusal_reasons": {},
    }
    for op in (graph_blob.get("ops") or ()):
        if op.get("name", "") in HOST_SYNC:
            continue
        split["operators"] += 1
        record, detail = library.lookup(op, topology, registration)
        if record is None:
            split["refused"] += 1
            split["refusal_reasons"].setdefault(op.get("name", "?"), detail)
            continue
        seconds = float(record["seconds"])
        if record.get("zero_work"):
            # Not a cheap measurement and not a gap: a case the engine is known
            # not to run at all, such as a collective at group width one or the
            # head on a chunk that produces no token. It is fully accounted for
            # and it is not a measurement, so it gets its own count.
            split["zero_work"] += 1
        elif record.get("interpolated"):
            split["interpolated"] += 1
            split["interpolated_seconds"] += seconds
            family = op.get("name", "?")
            split["interpolated_families"][family] = (
                split["interpolated_families"].get(family, 0) + 1)
        else:
            split["measured"] += 1
            split["measured_seconds"] += seconds
    # Every operator is accounted for by something. That is the property an
    # acceptance run needs, and it is not the same as every operator being
    # measured -- which is why both are reported and neither is called
    # "complete" on its own.
    split["accounted"] = (split["measured"] + split["interpolated"]
                          + split["zero_work"])
    split["complete_accounted"] = (split["accounted"] == split["operators"]
                                   and split["operators"] > 0)
    split["complete_measured"] = (
        split["measured"] == split["operators"] and split["operators"] > 0)
    return split
