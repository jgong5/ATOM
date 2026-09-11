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
           "LibraryCostOracle", "head_placement", "executed_body_rows"]


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


def head_placement(graph: dict) -> str:
    """Whether this graph's operators already include the LM head.

    Three answers, and the third is not the same as the second.

    ``"inside"`` -- the graph contains ``compute_logits``. One kind of step is:
    a decode replayed through a manually captured whole-forward graph at TP1,
    where ``ModelRunner.logits_in_graph = world_size == 1 and not is_tbo`` is
    true and the runner takes its logits from ``graph_logits`` rather than
    calling the head (model_runner.py:3238-3241). Adding a head term to such a
    graph charges the projection, the sampling gather and their launches twice.

    ``"outside"`` -- the graph declares it does not. Most steps are here, and
    TP1 does not exclude a step from this list: a prefill calls
    ``compute_logits`` eagerly at every width (model_runner.py:3182), and so
    does a piecewise-compiled decode (:3230). Neither consults
    ``logits_in_graph``, so width alone never decides this. A head term is then
    required, not optional: the work is real and this sum does not contain it.

    ``"unknown"`` -- the graph predates the field or was not written by our
    deriver. Not treated as "outside": an unstated placement is the case where
    a double count is invisible, so the caller refuses instead of guessing.
    """
    placement = (graph.get("provenance") or {}).get("head_placement")
    if not isinstance(placement, dict) or "in_this_graph" not in placement:
        return "unknown"
    return "inside" if placement["in_this_graph"] else "outside"


def _max_q_len(shape: StepShape) -> int:
    return max(shape.num_scheduled_tokens) if shape.num_scheduled_tokens else 0


def executed_body_rows(shape: StepShape) -> int:
    """How many rows this step's body actually runs.

    Not the token count. A replayed step runs its padded bucket: the runner
    computes ``num_tokens_pad = running_bs * max_q_len`` and the captured graph
    executes all of it, the real count being used only to slice the result
    afterwards (model_runner.py:3189-3192, 3841-3843). An eager step has no
    bucket and runs the tokens it was given.
    """
    if shape.capture_bucket is None:
        return shape.total_tokens
    return shape.capture_bucket * _max_q_len(shape)


def _traced_body_rows(graph: dict) -> Optional[int]:
    """The row count a body graph was traced over, or None if it does not say."""
    rows = ((graph.get("provenance") or {})
            .get("execution", {})
            .get("body_rows_traced"))
    return int(rows) if isinstance(rows, int) else None


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


#: The two data paths one collective operator runs over, which its signature
#: does not carry. On the ``registered`` path the peers read the input buffer
#: directly, because its address was registered with them when the graph was
#: captured; on the ``unregistered`` path the input is copied into a
#: pre-registered IPC pool buffer first and reduced from there.
#: `CustomAllreduce.all_reduce` picks between them from `_IS_CAPTURING`, which
#: only `CustomAllreduce.capture()` sets -- entered by
#: `parallel_state.graph_capture()`, which `model_runner.py:4158` uses, and
#: *not* by a bare `torch.cuda.graph`, which is how `microbench` times. Same
#: operator, same signature, 6.29 us against 9.06 us for the same 4x5120
#: bfloat16 reduction at TP2 (`agent_scratch/g4/ar_probe/README.md`).
REGISTERED = "registered"
UNREGISTERED = "unregistered"
_REGIMES = (REGISTERED, UNREGISTERED)


#: Two fields, deliberately not one name. A *graph* records the path its
#: region requires in production; a *price list* records the path its benchmark
#: was observed to take. They are different claims about different artifacts,
#: and a generic benchmark run against a graph that requires the registered
#: path still measures whichever path it actually ran. Sharing one key would
#: let a requirement be copied forward as though it were an observation, which
#: is the failure this whole distinction exists to prevent -- so the reader of
#: each side accepts only its own key.
REQUIRED_KEY = "collective_registration_required"
MEASURED_KEY = "collective_registration_measured"


def _declared_registration(price_blob: dict, prices_path: str):
    """Which data path a price list's collectives were *observed* on, or None.

    ``None`` means the list does not say, and a list that does not say cannot
    have its collectives spent on a step that requires a named path -- the same
    rule the group width already follows, for the same reason: the signature
    carries neither, so a price from the other path matches exactly and is
    spent silently.

    A price list is read only for what it observed. If it carries the graph
    side's requirement key that is a copied field, not a measurement, and it is
    an error rather than an answer: `microbench` captures into a bare
    `torch.cuda.graph` and does not arm the communicator, so a run priced
    against a registered-path graph has still measured the copy path.
    """
    provenance = price_blob.get("provenance") or {}
    if REQUIRED_KEY in provenance:
        raise ValueError(
            f"{prices_path} carries {REQUIRED_KEY}, which is a graph's "
            "requirement and not a measurement of this run. A benchmark "
            "records the path it was observed to take; it does not inherit the "
            f"path its subject needs. Use {MEASURED_KEY}, or leave it unstated "
            "and name the scope at load.")
    declared = provenance.get(MEASURED_KEY)
    if declared in _REGIMES:
        return declared
    if declared is not None:
        raise ValueError(
            f"{prices_path}: {MEASURED_KEY} is {declared!r}, not one of "
            f"{list(_REGIMES)}")
    # A probe that watched the communicator per call rather than declaring a
    # label: `registered_input` as the communicator actually received it.
    observed = provenance.get("observed_registered_input")
    if isinstance(observed, dict) and observed:
        seen = {bool(v) for v in observed.values()}
        if len(seen) == 1:
            return REGISTERED if seen.pop() else UNREGISTERED
    return None


def _same_scope(a: dict, b: dict) -> bool:
    return (a.get("registration") == b.get("registration")
            and (a.get("topology") or None) == (b.get("topology") or None))


def _scope_note(scope: dict) -> str:
    width = scope.get("topology")
    return (f"{scope.get('registration') or 'undeclared path'} at "
            f"{width or 'undeclared width'}")


class PriceLibrary:
    """Signature-keyed operator prices, with the scope each one is valid in.

    Assembled from one or more pricing runs. For a collective the signature is
    not a key on its own: it carries the message and neither the group width
    nor which of the communicator's two data paths ran. Two measurements that
    differ in either are measurements of different things that hash the same,
    so they are kept side by side under one signature and selected between
    explicitly. Keeping the first and dropping the rest would make correctness
    depend on the order the files were loaded in.

    Within one scope the older rule stands: the first price wins and a later
    one that differs by more than 5% is recorded as a conflict. Any run
    narrowed with ``--only`` marks the whole library partial, because a
    narrowed run's coverage counts describe what was asked for and not what a
    graph contains.
    """

    def __init__(self) -> None:
        #: signature -> the records priced under it, each with the scope it was
        #: measured in: ``{"topology": ..., "registration": ...}``. A list and
        #: not a record, for the reason in the class docstring.
        self._prices: dict[str, list] = {}
        self._refusals: dict[str, str] = {}
        #: signature -> layout fingerprint of the operator it was measured from,
        #: where the graph that was priced is available to say.
        self._layouts: dict[str, str] = {}
        self.sources: list[str] = []
        self.partial: list[str] = []
        self.conflicts: dict[str, list[float]] = {}

    @classmethod
    def load(cls, pairs) -> "PriceLibrary":
        """Build from ``(price_path, graph_path or None)`` pairs.

        The graph is optional and worth supplying. Without it a price can only
        be matched by signature, and a signature does not carry layout; with it
        the library knows which arrangement of memory each price was measured
        against and can refuse a lookup that would answer a strided call with a
        dense call's price.

        A third element names the collective registration regime for a list
        whose own provenance predates the field. It states the scope the
        measurement already had; it cannot change it, and disagreeing with what
        the file says is an error.
        """
        lib = cls()
        for entry in pairs:
            if isinstance(entry, (tuple, list)):
                price_path, graph_path = entry[0], entry[1]
                registration = entry[2] if len(entry) > 2 else None
            else:
                price_path, graph_path, registration = entry, None, None
            lib.add(price_path, graph_path, registration)
        return lib

    def add(self, price_path: str, graph_path: Optional[str] = None,
            registration: Optional[str] = None) -> None:
        from atom.compass.core.cost.priced import _declared_topology

        with open(price_path, encoding="utf-8") as fh:
            blob = json.load(fh)
        self.sources.append(price_path)
        provenance = blob.get("provenance") or {}
        topology = _declared_topology(blob, price_path)
        declared = _declared_registration(blob, price_path)
        if registration is not None:
            if registration not in _REGIMES:
                raise ValueError(f"registration must be one of "
                                 f"{list(_REGIMES)}, not {registration!r}")
            if declared is not None and declared != registration:
                raise ValueError(
                    f"{price_path} was measured on the {declared} path and is "
                    f"being loaded as {registration}: a caller may name the "
                    "scope a file leaves unstated, not overrule the one it "
                    "states")
        scope = {"topology": topology,
                 "registration": declared or registration}
        if provenance.get("only"):
            # A narrowed run priced one family and counted its coverage over
            # that family. Read as a library it is a library of one family, and
            # anything assembled from it is partial until a full-graph run says
            # otherwise.
            self.partial.append(f"{price_path} (--only {provenance['only']})")
        for sig, record in (blob.get("prices") or {}).items():
            kept = self._prices.setdefault(sig, [])
            same = [r for r in kept if _same_scope(r["scope"], scope)]
            if same:
                was = float(same[0]["seconds"])
                now = float(record["seconds"])
                if was and abs(now - was) / was > 0.05:
                    self.conflicts.setdefault(sig, [was]).append(now)
                continue
            kept.append(dict(record, source=price_path, scope=scope))
        for sig, why in (blob.get("unpriced") or {}).items():
            self._refusals.setdefault(sig, why)
        if graph_path:
            with open(graph_path, encoding="utf-8") as fh:
                for op in json.load(fh)["ops"]:
                    self._layouts.setdefault(_signature_of(op),
                                             _layout_fingerprint(op))

    def lookup(self, op: dict, topology=None, registration=None):
        """``(record, source)`` for one operator, or ``(None, reason)``.

        ``topology`` is the width the *graph* is of and ``registration`` is the
        data path this region's collectives take. Those are the two things a
        collective must be paid for at and the two things its signature does
        not carry, so both are required of one and neither is inferred: a
        lookup that cannot name the path it needs is refused, because the only
        alternative is spending a price measured on the other one.
        """
        from atom.compass.runtime.microbench import _is_collective_op

        sig = _signature_of(op)
        candidates = self._prices.get(sig) or []
        if not candidates:
            refusal = self._refusals.get(sig)
            return None, (f"refused when priced: {refusal}" if refusal
                          else "no entry for this signature")
        if _is_collective_op(op):
            record, why = self._collective(candidates, topology, registration)
            if record is None:
                return None, why
        else:
            record = candidates[0]
        measured = self._layouts.get(sig)
        if measured is not None:
            mine = _layout_fingerprint(op)
            if mine != measured:
                # Same key, different memory. The price is real and it is a
                # price of something else.
                return None, ("priced under a different operand layout "
                              f"({measured or 'dense'} vs {mine or 'dense'})")
        return record, record.get("source", "?")

    @staticmethod
    def _collective(candidates, topology, registration):
        """Select among same-signature collective prices, or say why not.

        Width first, then path. Both must match and the caller must have named
        the path: an unnamed one is a question the library cannot answer for
        it, not a licence to pick.
        """
        from atom.compass.core.cost.priced import _collectives_transferable

        fits = [r for r in candidates
                if _collectives_transferable(topology, r["scope"]["topology"])]
        if not fits:
            return None, ("its price was measured at a different group width, "
                          "which a collective's signature does not carry "
                          f"(have: {'; '.join(sorted({_scope_note(r['scope']) for r in candidates}))})")
        if registration is None:
            return None, (
                "the registration regime this region's collectives run on was "
                "not declared, and the signature does not carry it: a "
                "registered and an unregistered price are prices of different "
                "work under the same key "
                f"(have: {'; '.join(sorted({_scope_note(r['scope']) for r in fits}))})")
        matched = [r for r in fits if r["scope"]["registration"] == registration]
        if not matched:
            return None, (
                f"this region reduces on the {registration} path and no price "
                "under this signature was measured on it "
                f"(have: {'; '.join(sorted({_scope_note(r['scope']) for r in fits}))})")
        return matched[0], ""

    def body(self, graph_blob: dict,
             registration: Optional[str] = None) -> tuple[float, Coverage, int]:
        """Sum a graph's operators against the library.

        Returns the seconds, the coverage, and the launch count -- the last
        because an operator is not one kernel (attention launches three) and the
        execution term is paid per launch. The count comes from what the
        benchmark saw the operator launch, so it is only known for operators
        that were priced.

        ``registration`` is the data path this region's collectives take in
        production, from the caller when it knows and otherwise from the
        graph's own ``provenance.collective_registration_required``. A graph
        that states neither and contains collectives has those collectives
        refused by name rather than priced from whichever measurement was
        loaded. This is a *requirement*: what the region needs, never what any
        benchmark was observed to do.
        """
        from atom.compass.core.cost.priced import HOST_SYNC

        topology = dict(((graph_blob.get("key") or {}).get("topology") or []))
        if registration is None:
            declared = (graph_blob.get("provenance") or {}).get(REQUIRED_KEY)
            registration = declared if declared in _REGIMES else None
        ops = [op for op in (graph_blob.get("ops") or [])
               if op.get("name", "") not in HOST_SYNC]
        total, priced, launches = 0.0, 0, 0
        refused: dict[str, int] = {}
        reasons: dict[str, str] = {}
        sources: dict[str, int] = {}
        for op in ops:
            record, detail = self.lookup(op, topology, registration)
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
        scoped = sum(1 for recs in self._prices.values() if len(recs) > 1)
        multi = (f", {scoped} held in more than one scope" if scoped else "")
        return (f"PriceLibrary({len(self._prices)} signatures from "
                f"{len(self.sources)} runs, {len(self._layouts)} with a "
                f"recorded layout{multi}{partial}{conflict})")


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
                 head_graphs: Optional[GraphSource] = None,
                 floor_seconds: float = 0.0,
                 require_complete: bool = False,
                 body_registration: Optional[str] = None,
                 head_registration: Optional[str] = None,
                 regions=None) -> None:
        self.library = library
        self.graphs = graphs
        self.seconds_per_launch = seconds_per_launch
        #: The LM head as a second region of the same step -- its own graph,
        #: priced through the same library -- rather than a scalar correction.
        #: It has to be a graph: the head's cost is a projection whose N is the
        #: number of output-producing rows, and at TP>1 a collective whose
        #: price is a real two-rank measurement keyed on its own signature.
        #: Neither survives being averaged into one number.
        #:
        #: Kept apart from `extra_seconds` because where the head runs is a
        #: property of the width: at TP1 a level-3 graph replays it inside the
        #: body, at TP>1 the runner computes it eagerly afterwards. A head
        #: folded into an undifferentiated correction cannot be checked against
        #: the body graph that may already contain it.
        self.head_graphs = head_graphs
        #: Work a production step does that neither region contains -- the
        #: runner's own device work. Zero until it is measured, and zero is
        #: wrong; it is left visible here rather than folded into a fitted
        #: constant, so that a prediction carrying none of it can be told from
        #: one that does.
        self.extra_seconds = extra_seconds
        self.floor_seconds = floor_seconds
        #: Refuse rather than answer from a partial sum. Off by default because
        #: a partial answer with its coverage attached is useful; on where a
        #: gate requires a complete one.
        self.require_complete = require_complete
        #: Which of the communicator's two data paths each region's collectives
        #: run on, per region because the answer differs between them: a
        #: replayed body was captured under `graph_capture()` and reduces on the
        #: registered path, while the TP>1 head runs eagerly after the replay
        #: and takes the copy path. Left None where the graph's own provenance
        #: states it; a graph that states neither has its collectives refused.
        self.body_registration = body_registration
        self.head_registration = head_registration
        #: The runner's own regions -- postprocess, input preparation, and the
        #: TP broadcast of the sampled ids -- as a `RunnerRegions` measured on
        #: the source and refusing outside its domain. It supersedes
        #: `extra_seconds`, which is the same quantity as one number for a
        #: caller that has only that; setting both would charge it twice, so
        #: it is refused.
        if regions is not None and extra_seconds:
            raise ValueError(
                "extra_seconds and regions are the same term measured two "
                "ways; supplying both charges the runner's work twice. Pass "
                "the region model alone.")
        self.regions = regions
        self.last_coverage: Optional[Coverage] = None

    def estimate(self, shape: StepShape) -> StepCost:
        # Whether a shape is inside the region model's calibrated domain
        # depends on the shape alone -- sequence count, prefill extent, width.
        # Asking that first costs a few microseconds and refuses for exactly
        # the same reason, with exactly the same message, as asking it last;
        # asking it last means a shape outside the domain pays a bind, and on
        # a miss a full trace, for an answer that is then thrown away. On the
        # twenty-shape diagnostic that was 805 of 922 ms, two derivations of
        # 470 and 237 ms among them. No work is dropped and no answer moves:
        # the refusal is hoisted, not weakened.
        if self.regions is not None:
            why = self.regions.refusal(shape)
            if why is not None:
                raise ValueError(f"no measured region for this shape: {why}")
        graph = self.graphs.graph_for(shape)
        if graph is None:
            raise KeyError(
                f"no graph for {len(shape.num_scheduled_tokens)} requests, "
                f"{shape.total_tokens} tokens: derive one rather than "
                "answering from a neighbouring shape")
        self._check_body_rows(graph, shape)
        body, coverage, launches = self.library.body(
            graph, self.body_registration)
        head, head_coverage, head_launches = self._head_for(graph, shape)
        if head_coverage is not None:
            # One step, one coverage record: a head whose all-gather is unpriced
            # has to make the *step* incomplete, not sit in a second record that
            # `require_complete` never reads.
            coverage = coverage.merged(head_coverage)
        self.last_coverage = coverage
        if self.require_complete and not coverage.complete:
            raise ValueError("incomplete: " + coverage.describe())
        overhead = (launches + head_launches) * self.seconds_per_launch
        breakdown = {"<body>": body, "<overhead>": overhead}
        if head_coverage is not None:
            # Recorded even at zero seconds, because "the head ran and priced
            # to nothing" and "no head was added" are different claims.
            breakdown["<head>"] = head
        if self.regions is not None:
            # Named per region rather than summed, so a reader can see which
            # measured region each part of the step came from. Outside the
            # calibrated domain this raises rather than answering.
            breakdown.update(self.regions.breakdown(shape))
            runner = sum(v for k, v in breakdown.items()
                         if k not in ("<body>", "<overhead>", "<head>"))
        else:
            runner = self.extra_seconds
            if runner:
                breakdown["<runner>"] = runner
        total = max(body + overhead + head + runner, self.floor_seconds)
        return StepCost(seconds=total, breakdown=breakdown)

    def _check_body_rows(self, graph: dict, shape: StepShape) -> None:
        """Refuse a body graph traced over a different number of rows.

        A replayed step does not execute its batch; it executes its bucket. The
        runner pads to `running_bs * max_q_len` and runs every one of those rows
        through the body (model_runner.py:3192, 3841-3843), then slices the
        result back down. So a graph traced with twenty rows describes twenty
        rows of work, and charging it for a step that replays a thirty-two-row
        bucket understates the body by the padding -- silently, because both are
        "the decode-20 graph" by name.

        Only graphs that state their traced row count are checked. An older
        graph without the field is priced as before rather than refused: this
        guard exists to catch a mismatch it can see, not to invalidate every
        artifact derived before the field existed. Which of those two a caller
        wants is `require_complete`'s question, not this one's.
        """
        traced = _traced_body_rows(graph)
        if traced is None:
            return
        executed = executed_body_rows(shape)
        if traced == executed:
            return
        raise ValueError(
            f"this body graph was traced over {traced} rows but the step "
            f"executes {executed}"
            + (f" ({shape.capture_bucket} bucket x "
               f"{_max_q_len(shape)} query length)"
               if shape.capture_bucket is not None else " (eager, unpadded)")
            + ": a replay runs the padded bucket, not the batch, so pricing "
              "the narrower trace would drop the padding's work. Derive the "
              "graph at the shape the step actually runs.")

    def _head_for(self, graph: dict, shape: StepShape):
        """The head region of this step: seconds, coverage and launches.

        Absent -- `(0.0, None, 0)` -- for two reasons that are not gaps: no
        head graphs were supplied, or the step produces no output position, so
        the runner skips `compute_logits` entirely.

        Two refusals, both about composition rather than about the head itself.
        A body graph that already contains the head cannot also be given one:
        at TP1 `ModelRunner.logits_in_graph` is true and a level-3 replay runs
        the projection and its sampling inside the body, so adding a second
        region charges them twice. And a graph that does not say where its head
        is gets a refusal rather than a guess -- it is the one case where either
        answer is silently wrong, because a prediction quietly missing an LM
        head reads exactly like one quietly charging two.
        """
        if self.head_graphs is None:
            return 0.0, None, 0
        if not shape.produces_output:
            # `is_pure_middle_chunk(batch)` -> `logits = None`
            # (model_runner.py:3174). Nothing is projected and nothing is
            # sampled, so there is no head to charge.
            return 0.0, None, 0
        placement = head_placement(graph)
        if placement == "inside":
            raise ValueError(
                "this body graph already contains the LM head, so adding the "
                "head region would count it twice: it was derived at a width "
                "where ModelRunner.logits_in_graph is true (TP1), which puts "
                "compute_logits inside the replayed body. Price the step from "
                "that graph alone, or use a body graph that excludes the head.")
        if placement == "unknown":
            raise ValueError(
                "this body graph does not say whether it contains the LM head, "
                "and a head region cannot be added to a graph whose head "
                "placement is unstated -- both answers are plausible and both "
                "failures are silent. Re-derive it so provenance carries "
                "'head_placement'.")
        head_graph = self.head_graphs.graph_for(shape)
        if head_graph is None:
            raise KeyError(
                f"no head graph for {len(shape.num_scheduled_tokens)} "
                f"requests, {shape.total_tokens} tokens: this step samples, so "
                "its head is real work. Derive one rather than dropping it.")
        if head_placement(head_graph) == "outside":
            raise ValueError(
                "the head graph says it does not contain the head; it is a "
                "body graph. Derive it with --region head.")
        padded = ((head_graph.get("provenance") or {})
                  .get("head_placement", {})
                  .get("rows_padded_to_capture_bucket"))
        if padded and shape.capture_bucket is None:
            # A head traced over a capture bucket's padded rows is a wider GEMM
            # than the one an eager step runs: the capture projects
            # `outputs[:bs * max_q_len]` and slices `graph_logits` afterwards
            # (model_runner.py:4297, 3239), while the eager head is handed
            # hidden states already cut to the real count (model_runner.py:
            # 3237-3241). Twenty requests are twenty rows in one and a bucket's
            # worth in the other.
            raise ValueError(
                "this head graph projects a capture bucket's padded rows, but "
                "the step runs eagerly and projects only its real ones. "
                "Derive the head for the eager shape rather than charging the "
                "padded GEMM.")
        return self.library.body(head_graph, self.head_registration)

    def describe(self) -> str:
        head = ("no head region" if self.head_graphs is None
                else f"head {self.head_graphs.describe()}")
        return (f"LibraryCostOracle({self.library.describe()}, "
                f"{self.graphs.describe()}, "
                f"+{self.seconds_per_launch * 1e6:.2f}us/launch, "
                f"{head}, "
                f"runner {self.extra_seconds * 1e3:.3f}ms)")
