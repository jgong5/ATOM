"""**empirical/library** -- a step costs what its operators cost, where the
prices come from a library rather than from a pricing run of that step's graph.

`priced` holds graphs it was handed, already summed, and says so when asked
about anything else: "What this oracle cannot do yet is price a shape it has no
graph for." A serving run does not stay on the shapes that were measured --
chunked prefill moves the boundary every step and decode batches change size --
so that oracle answers the rung and the workload is elsewhere.

This one separates the two halves of the question. A *graph* says which
operators a step runs and how often, and it can be derived for any shape on the
meta device with no GPU present. A *price* says what one operator costs, and is
a property of the hardware and the operator, not of the step it appeared in. So
the library is measured once, on the source configuration, and every later step
is a lookup:

    body = sum over the derived graph of price(signature) x occurrences

What makes this usable rather than merely arithmetic is the part that says what
it could not price. A step summed from 96% of its operators is not the same
evidence as a step summed from all of them, and the two must not print the same
number with nothing to tell them apart. Every cost here comes with a
:class:`Coverage`, and a caller that ignores it gets a number whose provenance it
cannot state.

Three ways a lookup fails, all of them recorded rather than silently zeroed:

* **missing** -- no entry for that signature. The shape was never measured.
* **refused at pricing time** -- the library's own run could not price it, and
  the reason it recorded is carried through verbatim.
* **layout mismatch** -- there is an entry, and it was measured against a
  different arrangement of memory than this graph describes. A signature is
  name, shapes, dtypes, scalars and context; it does **not** carry layout. So a
  dense reconstruction and a strided window into a fused buffer share a key, and
  a library merged on that key alone would answer one with the other's price and
  say nothing. Where the graph a price list was measured from is available, this
  compares the two and refuses the mismatch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

from atom.compass.core.cost.base import StepCost, StepShape

logger = logging.getLogger(__name__)

__all__ = ["Coverage", "PriceLibrary", "GraphSource", "StaticGraphs",
           "LibraryCostOracle"]


def _signature_of(op: dict) -> str:
    """The one definition of a signature, imported rather than restated.

    Imported here rather than at module scope because `microbench` reaches
    `torch` through the Triton tracer, and this module is meant to be usable
    wherever a replay is -- the import is paid once, on the first lookup.

    A second copy of this function would be the kind of bug that produces
    plausible numbers: prices keyed one way and looked up another agree on most
    operators and disagree on whichever detail drifted.
    """
    from atom.compass.runtime.microbench import signature_of

    return signature_of(op)


def _layout_fingerprint(op: dict) -> str:
    """How this operator's tensor arguments sit in memory, as a comparable key.

    Empty when the graph records no layout, which is the common case: a plain
    contiguous tensor is not recorded because there is nothing to say about it.
    Two empty fingerprints therefore match, and that match is meaningful -- both
    describe a dense rebuild.
    """
    layouts = op.get("layouts") or ()
    return json.dumps([[int(pos), list(tuple(value))]
                       for pos, value in (tuple(x) for x in layouts)],
                      sort_keys=True)


@dataclass(frozen=True)
class Coverage:
    """What a summed cost is a sum *of*.

    Counted in operators and in occurrence-weighted seconds, not in signatures.
    A price list has one entry per signature and a step does not run one of
    each: this model's step has five GEMM signatures launched 256 times beside
    sixteen attention signatures launched once, so a count of signatures
    describes the library and a count of operators describes the step. Only the
    second is what a coverage claim is about.
    """

    operators: int
    priced: int
    seconds: float
    #: Operator counts by name for what could not be priced, with one reason
    #: each -- named, because "4% unpriced" and "4% unpriced, all of it
    #: attention" are different situations.
    refused: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    #: Which price list answered, by operator count. A library assembled from
    #: more than one run is legitimate; one whose composition cannot be stated
    #: is not.
    sources: dict[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.priced == self.operators

    def merged(self, other: "Coverage") -> "Coverage":
        """One record over two regions of the same step.

        A step summed from a body graph and a head graph is still one step, and
        its coverage has to read as one: a head whose GEMM is unpriced must
        make the step incomplete, not sit in a second record nobody prints.
        Reasons collide only when the same operator name is refused in both,
        and then either reason is the truth about that name.
        """
        refused = dict(self.refused)
        for name, n in other.refused.items():
            refused[name] = refused.get(name, 0) + n
        sources = dict(self.sources)
        for name, n in other.sources.items():
            sources[name] = sources.get(name, 0) + n
        return Coverage(
            operators=self.operators + other.operators,
            priced=self.priced + other.priced,
            seconds=self.seconds + other.seconds,
            refused=refused,
            reasons={**self.reasons, **other.reasons},
            sources=sources,
        )

    def describe(self) -> str:
        head = (f"{self.priced}/{self.operators} operators, "
                f"{self.seconds * 1e3:.3f} ms")
        if self.complete:
            return head
        missing = ", ".join(f"{n}x {name}"
                            for name, n in sorted(self.refused.items(),
                                                  key=lambda kv: -kv[1])[:5])
        return f"{head}; UNPRICED {self.operators - self.priced}: {missing}"


class PriceLibrary:
    """Signature-keyed operator prices, with where each one came from.

    Assembled from one or more pricing runs. Merging is allowed and is not
    silent: a signature priced by two runs keeps the first and records the
    disagreement, and any run that was narrowed with ``--only`` marks the whole
    library partial, because a narrowed run's coverage counts describe what was
    asked for and not what a graph contains.
    """

    def __init__(self) -> None:
        self._prices: dict[str, dict] = {}
        self._refusals: dict[str, str] = {}
        #: signature -> layout fingerprint of the operator it was measured from,
        #: where the graph that was priced is available to say.
        self._layouts: dict[str, str] = {}
        self.sources: list[str] = []
        self.partial: list[str] = []
        self.conflicts: dict[str, list[float]] = {}
        #: signature -> the parallel width its price was measured at. A
        #: collective's signature carries its message and not its group width,
        #: so a TP4 all-reduce price matches a TP2 call exactly and would be
        #: spent silently. `priced` found this and refuses it; the same refusal
        #: belongs here, per signature, because this library is assembled from
        #: runs at more than one width on purpose.
        self._topology: dict[str, dict] = {}

    @classmethod
    def load(cls, pairs) -> "PriceLibrary":
        """Build from ``(price_path, graph_path or None)`` pairs.

        The graph is optional and worth supplying. Without it a price can only
        be matched by signature, and a signature does not carry layout; with it
        the library knows which arrangement of memory each price was measured
        against and can refuse a lookup that would answer a strided call with a
        dense call's price.
        """
        lib = cls()
        for entry in pairs:
            price_path, graph_path = (entry if isinstance(entry, (tuple, list))
                                      else (entry, None))
            lib.add(price_path, graph_path)
        return lib

    def add(self, price_path: str, graph_path: Optional[str] = None) -> None:
        from atom.compass.core.cost.priced import _declared_topology

        with open(price_path, encoding="utf-8") as fh:
            blob = json.load(fh)
        self.sources.append(price_path)
        provenance = blob.get("provenance") or {}
        topology = _declared_topology(blob, price_path)
        if provenance.get("only"):
            # A narrowed run priced one family and counted its coverage over
            # that family. Read as a library it is a library of one family, and
            # anything assembled from it is partial until a full-graph run says
            # otherwise.
            self.partial.append(f"{price_path} (--only {provenance['only']})")
        for sig, record in (blob.get("prices") or {}).items():
            if sig in self._prices:
                was = float(self._prices[sig]["seconds"])
                now = float(record["seconds"])
                if was and abs(now - was) / was > 0.05:
                    self.conflicts.setdefault(sig, [was]).append(now)
                continue
            self._prices[sig] = dict(record, source=price_path)
            self._topology[sig] = topology
        for sig, why in (blob.get("unpriced") or {}).items():
            self._refusals.setdefault(sig, why)
        if graph_path:
            with open(graph_path, encoding="utf-8") as fh:
                for op in json.load(fh)["ops"]:
                    self._layouts.setdefault(_signature_of(op),
                                             _layout_fingerprint(op))

    def lookup(self, op: dict, topology=None):
        """``(record, source)`` for one operator, or ``(None, reason)``.

        ``topology`` is the width the *graph* is of, which a collective must be
        paid for at.
        """
        from atom.compass.core.cost.priced import _collectives_transferable
        from atom.compass.runtime.microbench import _is_collective_op

        sig = _signature_of(op)
        record = self._prices.get(sig)
        if record is None:
            refusal = self._refusals.get(sig)
            return None, (f"refused when priced: {refusal}" if refusal
                          else "no entry for this signature")
        if _is_collective_op(op) and not _collectives_transferable(
                topology, self._topology.get(sig)):
            return None, ("its price was measured at a different group width, "
                          "which a collective's signature does not carry")
        measured = self._layouts.get(sig)
        if measured is not None:
            mine = _layout_fingerprint(op)
            if mine != measured:
                # Same key, different memory. The price is real and it is a
                # price of something else.
                return None, ("priced under a different operand layout "
                              f"({measured or 'dense'} vs {mine or 'dense'})")
        return record, record.get("source", "?")

    def body(self, graph_blob: dict) -> tuple[float, Coverage, int]:
        """Sum a graph's operators against the library.

        Returns the seconds, the coverage, and the launch count -- the last
        because an operator is not one kernel (attention launches three) and the
        execution term is paid per launch. The count comes from what the
        benchmark saw the operator launch, so it is only known for operators
        that were priced.
        """
        from atom.compass.core.cost.priced import HOST_SYNC

        topology = dict(((graph_blob.get("key") or {}).get("topology") or []))
        ops = [op for op in (graph_blob.get("ops") or [])
               if op.get("name", "") not in HOST_SYNC]
        total, priced, launches = 0.0, 0, 0
        refused: dict[str, int] = {}
        reasons: dict[str, str] = {}
        sources: dict[str, int] = {}
        for op in ops:
            record, detail = self.lookup(op, topology)
            if record is None:
                refused[op["name"]] = refused.get(op["name"], 0) + 1
                reasons.setdefault(op["name"], detail)
                continue
            total += float(record["seconds"])
            launches += max(1, len(record.get("kernels") or {}))
            priced += 1
            sources[detail] = sources.get(detail, 0) + 1
        return (total,
                Coverage(operators=len(ops), priced=priced, seconds=total,
                         refused=refused, reasons=reasons, sources=sources),
                launches)

    def describe(self) -> str:
        partial = f", PARTIAL: {'; '.join(self.partial)}" if self.partial else ""
        conflict = (f", {len(self.conflicts)} signatures disagree across runs"
                    if self.conflicts else "")
        return (f"PriceLibrary({len(self._prices)} signatures from "
                f"{len(self.sources)} runs, {len(self._layouts)} with a "
                f"recorded layout{partial}{conflict})")


class GraphSource(Protocol):
    """Where the oracle gets the operator graph for a step it is asked about."""

    def graph_for(self, shape: StepShape) -> Optional[dict]:
        ...

    def describe(self) -> str:
        ...


class StaticGraphs:
    """A fixed set of graphs, keyed the way a derivation cache keys them.

    The key is a multiset of **rows**, where a row is one request's
    ``(query_len, cached_len)`` pair kept together. Sorting the two lists
    independently would be wrong and quietly so: a batch of
    ``(1, cached 0) + (64, cached 1024)`` and one of
    ``(64, cached 0) + (1, cached 1024)`` produce the same pair of sorted
    lists and nothing like the same attention work. Only whole rows are
    interchangeable, so only whole rows are canonicalised.

    The rest of the key is everything else that changes the graph and is not a
    row: how much of the batch is prefill (which selects the attention branch,
    and with it whether a row is a fresh or a cached-prefix chunk), the group
    widths and this rank's coordinates, the replay bucket the step was padded
    to, and whether it ran compiled. Model identity, block size and layout are
    not in the key because they are properties of the *instance* -- a
    ``StaticGraphs`` holds graphs for one model at one block size, and mixing
    two into one cache is a caller error this class cannot detect.

    What this equivalence is and is not: it is a claim about the derived
    operator graph being identical under permutation of whole rows, which is
    what makes the cache sound. It is not a claim that two serving runs with
    permuted requests are interchangeable -- serving output keeps request
    identity and order.
    """

    def __init__(self, graphs: dict) -> None:
        self._graphs = dict(graphs)

    @staticmethod
    def key(shape: StepShape):
        rows = tuple(sorted(
            (int(q), int(max(0, c - q)))
            for q, c in zip(shape.num_scheduled_tokens, shape.context_lens)))
        groups = tuple(sorted((str(k), int(v))
                              for k, v in (shape.topology or {}).items()))
        coords = tuple(sorted((str(k), int(v))
                              for k, v in (shape.rank_coords or {}).items()))
        return (rows, int(shape.num_prefill_tokens), groups, coords,
                shape.capture_bucket, shape.compiled)

    def graph_for(self, shape: StepShape) -> Optional[dict]:
        return self._graphs.get(self.key(shape))

    def describe(self) -> str:
        return f"StaticGraphs({len(self._graphs)} shapes)"


class LibraryCostOracle:
    """A step's cost from its derived graph and the library, plus coverage.

    The execution term -- what a step pays beyond its kernels -- is deliberately
    the same one `PricedGraphCostOracle` already measured and argued for, passed
    in rather than refitted here. This class's subject is the body and what is
    missing from it.
    """

    def __init__(self, library: PriceLibrary, graphs: GraphSource,
                 seconds_per_launch: float = 0.0,
                 extra_seconds: float = 0.0,
                 floor_seconds: float = 0.0,
                 require_complete: bool = False) -> None:
        self.library = library
        self.graphs = graphs
        self.seconds_per_launch = seconds_per_launch
        #: Work a production step does that the traced body never contained --
        #: the LM head and the runner's own device work. Zero until it is
        #: measured, and zero is wrong; it is left visible here rather than
        #: folded into a fitted constant, so that a prediction carrying none of
        #: it can be told from one that does.
        self.extra_seconds = extra_seconds
        self.floor_seconds = floor_seconds
        #: Refuse rather than answer from a partial sum. Off by default because
        #: a partial answer with its coverage attached is useful; on where a
        #: gate requires a complete one.
        self.require_complete = require_complete
        self.last_coverage: Optional[Coverage] = None

    def estimate(self, shape: StepShape) -> StepCost:
        graph = self.graphs.graph_for(shape)
        if graph is None:
            raise KeyError(
                f"no graph for {len(shape.num_scheduled_tokens)} requests, "
                f"{shape.total_tokens} tokens: derive one rather than "
                "answering from a neighbouring shape")
        body, coverage, launches = self.library.body(graph)
        self.last_coverage = coverage
        if self.require_complete and not coverage.complete:
            raise ValueError("incomplete: " + coverage.describe())
        overhead = launches * self.seconds_per_launch
        total = max(body + overhead + self.extra_seconds, self.floor_seconds)
        breakdown = {"<body>": body, "<overhead>": overhead}
        if self.extra_seconds:
            breakdown["<runner>"] = self.extra_seconds
        return StepCost(seconds=total, breakdown=breakdown)

    def describe(self) -> str:
        return (f"LibraryCostOracle({self.library.describe()}, "
                f"{self.graphs.describe()}, "
                f"+{self.seconds_per_launch * 1e6:.2f}us/launch, "
                f"runner {self.extra_seconds * 1e3:.3f}ms)")
