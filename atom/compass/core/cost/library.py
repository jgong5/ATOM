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
from atom.compass.core.cost.identity import cost_key
from atom.compass.core.loaded_input import load_json

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


def _cost_key_of(op: dict) -> str:
    """The key a price is filed and found under.

    `_signature_of` identifies the call; this identifies the work. They differ
    only in the allocator's absolute addresses, and the rule lives in
    `atom.compass.core.cost.identity` so that the key a file was written with
    and the key a lookup computes cannot drift apart.
    """
    from atom.compass.runtime.microbench import cost_key_of

    return cost_key_of(op)


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
    bucket and runs the tokens it was given -- and so does a prefill that
    declares one, because `ForwardMode.decide` sends any batch holding a
    prefill token down the eager path (forward_context.py:196-204). The same
    expression as `BatchSpec.padded_rows`, prefill guard included: this is the
    number `_check_body_rows` holds a derivation to, and a guard is only worth
    having if both sides compute it the same way.
    """
    if shape.capture_bucket is None or int(
            getattr(shape, "num_prefill_tokens", 0) or 0):
        return shape.total_tokens
    return shape.capture_bucket * _max_q_len(shape)


def _traced_body_rows(graph: dict) -> Optional[int]:
    """The row count a body graph was traced over, or None if it does not say."""
    rows = ((graph.get("provenance") or {})
            .get("execution", {})
            .get("body_rows_traced"))
    return int(rows) if isinstance(rows, int) else None


#: The one marker that makes a derived price distinguishable from a measured
#: one by inspection. A record carrying ``"interpolated": True`` was fitted
#: from neighbouring points; a measured record carries no such field at all,
#: so absence is the measured case and nothing has to be back-filled onto the
#: measurements. Deliberately a plain field rather than a type or an import:
#: the module that derives prices depends on this one, not the other way
#: round, and a flag on the record survives the record being serialised.
INTERPOLATED_FLAG = "interpolated"
#: The prefix that same module puts on its source string, so the split is also
#: legible in `Coverage.sources` and in any report that prints it.
INTERPOLATED_SOURCE_PREFIX = "interpolated://"
#: The separate declaration that an operator does no work in this
#: configuration -- a collective at group width one, a head on a chunk that
#: produces no token. Fully accounted for and not a measurement, so it is its
#: own count. Declared, never inferred from a zero time: a real measurement can
#: round to zero, and reading that as "known not to run" would turn a timing
#: floor into a structural claim.
ZERO_WORK_FLAG = "zero_work"


@dataclass(frozen=True)
class Coverage:
    """What a summed cost is a sum *of*.

    Counted in operators and in occurrence-weighted seconds, not in signatures.
    A price list has one entry per signature and a step does not run one of
    each: this model's step has five GEMM signatures launched 256 times beside
    sixteen attention signatures launched once, so a count of signatures
    describes the library and a count of operators describes the step. Only the
    second is what a coverage claim is about.

    An answered operator is answered in one of three ways, counted apart and
    none of them recoverable by subtraction. ``measured`` is a timing of this
    operator at this signature. ``interpolated`` is a value derived from
    timings of neighbouring points in the same family -- legitimate inside its
    support, and still not a measurement of this point. ``zero_work`` is an
    operator the configuration is known not to run at all -- a collective at
    group width one, a head on a chunk producing no token. It is fully
    accounted for and it is not a timing, so a coverage claim resting on it is
    a different claim from one resting on a measurement.

    ``complete`` means nothing was refused: the step has a predicted cost for
    every operator in it, and validated in-support interpolation may be part
    of that. ``complete_measured`` is the stricter claim that none of it was
    fitted. Both are published, because the PoC needs the first and may only
    call the second direct measurement.
    """

    operators: int
    #: Priced from a timing of this operator at this signature.
    measured: int
    seconds: float
    #: Priced from neighbouring measurements in the same family. The seam is a
    #: plain field on the record -- ``record["interpolated"] is True`` -- so
    #: this module reads the marker without importing the module that derives
    #: it; `atom/compass/core/cost/families/adapter.py` sets it and labels its
    #: source string ``interpolated://<family>/...``.
    interpolated: int = 0
    #: Declared not to run in this configuration, by the record's own
    #: ``zero_work`` field. Never inferred from a zero time.
    zero_work: int = 0
    #: Operator counts by name for what could not be priced, with one reason
    #: each -- named, because "4% unpriced" and "4% unpriced, all of it
    #: attention" are different situations.
    refused: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    #: One entry per refused *signature*, not per name. A name collapses every
    #: shape an operator was called at, and a name is not something anyone can
    #: measure or model: the counts above answer "how much is missing", this
    #: answers "missing at what", which is the question a price acquisition or a
    #: family model is actually given. Held as a tuple of plain dicts, and
    #: deliberately without the operator's context entries -- those carry whole
    #: block tables and slot maps, and a refusal record that large stops being
    #: something a log line or a handoff can hold.
    refused_signatures: tuple = ()
    #: Which price list answered, by operator count. A library assembled from
    #: more than one run is legitimate; one whose composition cannot be stated
    #: is not.
    sources: dict[str, int] = field(default_factory=dict)

    @property
    def priced(self) -> int:
        """Answered, however answered. Kept as the total the reports already read."""
        return self.measured + self.interpolated + self.zero_work

    @property
    def complete(self) -> bool:
        """Every operator has a price. Some of them may be fitted."""
        return self.priced == self.operators

    @property
    def complete_measured(self) -> bool:
        """Every operator has a price and none of it was interpolated."""
        return self.complete and self.interpolated == 0

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
            measured=self.measured + other.measured,
            seconds=self.seconds + other.seconds,
            interpolated=self.interpolated + other.interpolated,
            zero_work=self.zero_work + other.zero_work,
            refused=refused,
            reasons={**self.reasons, **other.reasons},
            # Concatenated, not merged by key: body and head are two graphs,
            # and the same signature refused in both is two refusals at two
            # widths. Summing them would report one call site where there are
            # two, which is the thing the per-name counts already do.
            refused_signatures=(tuple(self.refused_signatures)
                                + tuple(other.refused_signatures)),
            sources=sources,
        )

    def as_dict(self) -> dict:
        """The same record as data, for a report that will be read by machine.

        `describe` is for a person reading a log line and truncates the refusals
        to the five commonest; a saved report is the thing an acceptance run is
        graded from, and a category that only ever appeared inside a sentence
        cannot be checked. The four ways an operator is accounted for are each
        their own key, and the two derived answers are written out rather than
        left to be recomputed from them by a reader who might recompute them
        differently.
        """
        return {
            "operators": self.operators,
            "measured": self.measured,
            "interpolated": self.interpolated,
            "zero_work": self.zero_work,
            "refused": self.operators - self.priced,
            "priced": self.priced,
            "complete": self.complete,
            "complete_measured": self.complete_measured,
            "seconds": self.seconds,
            "refused_operators": dict(self.refused),
            "refusal_reasons": dict(self.reasons),
            "refused_signatures": [dict(entry)
                                   for entry in self.refused_signatures],
            "sources": dict(self.sources),
        }

    def describe(self) -> str:
        head = (f"{self.priced}/{self.operators} operators, "
                f"{self.seconds * 1e3:.3f} ms")
        # Stated whenever any of it is fitted or free, and stated in the same
        # line as the total: a reader who sees only "1/1 operators" of an
        # interpolated step has been told the step is covered and not told
        # what by.
        if self.interpolated or self.zero_work:
            parts = [f"{self.measured} measured"]
            if self.interpolated:
                parts.append(f"{self.interpolated} interpolated")
            if self.zero_work:
                parts.append(f"{self.zero_work} zero-work")
            head = f"{head} ({', '.join(parts)})"
        if self.complete:
            return head
        ranked = sorted(self.refused.items(), key=lambda kv: -kv[1])
        missing = ", ".join(f"{n}x {name}" for name, n in ranked[:5])
        # What the five shown do not account for, stated rather than left for a
        # reader to discover by adding them up. Run 4's batch 5 printed five
        # names summing to 67 against an UNPRICED 68: the sixth name was real,
        # and the line gave no sign it existed. A truncation that cannot be
        # detected from the line it truncates is worse than a longer line.
        rest = sum(n for _, n in ranked[5:])
        if rest:
            missing = (f"{missing}, and {rest} more in "
                       f"{len(ranked) - 5} further names")
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
        #: cost key -> the records priced under it, each with the scope it was
        #: measured in: ``{"topology": ..., "registration": ...}``. A list and
        #: not a record, for the reason in the class docstring.
        #:
        #: The key is the *cost* key, not the signature: prices on disk were
        #: written under the signature they were measured with, and are
        #: reindexed through the same normalisation a lookup goes through, so
        #: an artifact does not have to be recollected to be found. Each record
        #: keeps its own ``signature``, which is the observation identity and
        #: is never rewritten.
        self._prices: dict[str, list] = {}
        self._refusals: dict[str, str] = {}
        #: cost key -> how many lookups this library answered from a record
        #: measured under a different allocation. Not an error and not a
        #: separate kind of price -- the same measurement, found through the
        #: normalised key -- but it is the thing this normalisation buys, so it
        #: is counted rather than assumed.
        self.address_shifted: dict[str, int] = {}
        #: Every artifact this library parsed, in the order it parsed them,
        #: each carrying the digest of the exact bytes that were parsed. Empty
        #: on a library assembled by hand, which is an honest statement that
        #: there is no file to identify rather than a missing record.
        self.loaded_inputs: tuple = ()
        self.sources: list[str] = []
        self.partial: list[str] = []
        self.conflicts: dict[str, list[float]] = {}

    @classmethod
    def load(cls, pairs, *, coords=None) -> "PriceLibrary":
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

        ``coords`` is handed down to every ``add`` unresolved, for the reason
        given there.
        """
        lib = cls()
        for entry in pairs:
            if isinstance(entry, (tuple, list)):
                price_path, graph_path = entry[0], entry[1]
                registration = entry[2] if len(entry) > 2 else None
            else:
                price_path, graph_path, registration = entry, None, None
            lib.add(price_path, graph_path, registration, coords=coords)
        return lib

    def add(self, price_path: str, graph_path: str | None = None,
            registration: str | None = None, *, coords=None) -> None:
        """Load a price file and, where given, the graph it was priced from.

        ``coords`` names this rank so the loader can resolve a per-rank
        artifact. It is passed *unresolved*: resolution happens once, inside
        the loader, which is the only place that can report both the stem a
        caller asked for and the file this rank was actually served. Omitting
        it is exactly today's behaviour -- no resolution, ``rank_own`` false.
        """
        self._ingest(price_path, graph_path, registration, coords)

    def _ingest(self, price_path: str, graph_path: str | None,
                registration: Optional[str], coords):
        """Read both artifacts once and absorb them. Returns the payloads.

        Subclasses that need the graph take it from here rather than calling
        ``super().add`` and opening the file a second time. Two reads of one
        path are not only wasted work: they are two chances to see different
        bytes, and the digest retained for provenance would then describe
        whichever read happened to be hashed.
        """
        from atom.compass.core.cost.priced import _declared_topology

        blob, loaded = load_json(price_path, role="oracle.price", coords=coords)
        records = [loaded]
        graph = None
        if graph_path:
            graph, graph_loaded = load_json(
                graph_path, role="oracle.price_graph", coords=coords)
            records.append(graph_loaded)
        # Immutable, and replaced rather than mutated, so a holder that has
        # already read it cannot be changed under by a later `add`.
        self.loaded_inputs = self.loaded_inputs + tuple(records)
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
        # The layouts this run's own graph recorded, read before the prices so
        # each record can carry the one it was measured under.
        #
        # Keeping these in one signature-keyed table instead made the check
        # load-order dependent: a signature holds several scoped records, and
        # the first file read fixed the layout every later scope was validated
        # against. Two files priced at different widths and different layouts
        # then produced both errors at once -- the scope whose layout lost the
        # race had its own measurement refused, and the other scope's request
        # was answered from a price measured on a layout it does not have.
        #
        # Keyed by the RAW signature, not by the cost key. The cost key is a
        # many-to-one map: two observations whose allocator addresses differ
        # collapse onto one key, and nothing says they share a layout. Keying
        # this table by the cost key reintroduces the same load-order bug one
        # level down -- `setdefault` keeps whichever raw observation the graph
        # listed first, while the record finally chosen may be a different raw
        # one (a different order in the JSON, or the first raw record refused
        # a price), so a price gets stamped with a layout it was not measured
        # under. The cost key is a reusable lookup index. It is not an
        # association, and anything per-record stays on the observation it
        # belongs to.
        layouts = {}
        if graph is not None:
            for op in graph["ops"]:
                layouts.setdefault(_signature_of(op), _layout_fingerprint(op))
        for sig, record in (blob.get("prices") or {}).items():
            # Reindex on the way in. The file names the signature it was
            # measured under; the library files it under the cost key, through
            # the same normalisation a lookup goes through. Nothing on disk is
            # rewritten and nothing is remeasured -- an artifact collected
            # before this rule existed is found by it.
            key = cost_key(sig)
            kept = self._prices.setdefault(key, [])
            same = [r for r in kept if _same_scope(r["scope"], scope)]
            if same:
                was = float(same[0]["seconds"])
                now = float(record["seconds"])
                if was and abs(now - was) / was > 0.05:
                    # Two measurements the cost key says are of the same work,
                    # disagreeing. Recorded rather than averaged or quietly
                    # kept: if the normalisation has collapsed something that
                    # is not a nuisance, this is the line that says so.
                    self.conflicts.setdefault(key, [was]).append(now)
                continue
            # ``signature`` is the observation identity, kept verbatim. The
            # cost key says which measurements answer for an operator; this
            # says which call was actually measured, and a lookup answered
            # under a shifted allocation is told apart by comparing the two.
            entry = dict(record, source=price_path, scope=scope, signature=sig)
            if sig in layouts:
                # Looked up by this record's own raw signature, so the layout
                # stored is the one this measurement was taken under and not
                # whichever collapsed sibling the graph happened to list first.
                #
                # Absent is not dense: a price loaded without its graph has
                # nothing to say about layout, and must not be read as having
                # said "dense". Only a recorded one is stored, and only a
                # recorded one is checked.
                entry["layout"] = layouts[sig]
            kept.append(entry)
        for sig, why in (blob.get("unpriced") or {}).items():
            self._refusals.setdefault(cost_key(sig), why)
        return blob, graph

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

        key = _cost_key_of(op)
        candidates = self._prices.get(key) or []
        if not candidates:
            refusal = self._refusals.get(key)
            return None, (f"refused when priced: {refusal}" if refusal
                          else "no entry for this signature")
        if _is_collective_op(op):
            record, why = self._collective(candidates, topology, registration)
            if record is None:
                return None, why
        else:
            record = candidates[0]
        # The measurement answers for this operator, and it was taken under a
        # different allocation. That is the whole point of the cost key, and it
        # is still worth counting: a coverage report that cannot separate
        # "priced exactly as measured" from "priced through the normalisation"
        # cannot be audited if the normalisation later turns out to be wrong.
        measured_signature = record.get("signature")
        if (measured_signature is not None
                and measured_signature != _signature_of(op)):
            self.address_shifted[key] = self.address_shifted.get(key, 0) + 1
        # Against the layout of the record that was *selected*, which for a
        # collective is chosen by width and path above. A price and the layout
        # it was measured under are one measurement and travel together.
        measured = record.get("layout")
        if measured is not None:
            mine = _layout_fingerprint(op)
            if mine != measured:
                # Same key, different memory. The price is real and it is a
                # price of something else.
                return None, (
                    "priced under a different operand layout at "
                    f"{_scope_note(record['scope'])} "
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
        total, launches = 0.0, 0
        measured = interpolated = zero_work = 0
        refused: dict[str, int] = {}
        reasons: dict[str, str] = {}
        sources: dict[str, int] = {}
        #: Refused calls gathered by cost key, in the order the graph lists
        #: them, so the record a reader gets back is the step's own order and
        #: not a dict's.
        by_key: dict[str, dict] = {}
        for op in ops:
            record, detail = self.lookup(op, topology, registration)
            if record is None:
                refused[op["name"]] = refused.get(op["name"], 0) + 1
                reasons.setdefault(op["name"], detail)
                key = _cost_key_of(op)
                entry = by_key.get(key)
                if entry is None:
                    by_key[key] = {
                        "name": op.get("name", ""),
                        "cost_key": key,
                        "signature": _signature_of(op),
                        "input_shapes": op.get("input_shapes"),
                        # The graph's own field name is `dtypes`, inputs only;
                        # reading `input_dtypes` here would record None for
                        # every refusal and look like a graph that lost them.
                        "dtypes": op.get("dtypes"),
                        "output_shapes": op.get("output_shapes"),
                        "output_dtypes": op.get("output_dtypes"),
                        # In the signature already, repeated as a field because
                        # this is where a decode attention says max_qlen=1 and a
                        # prefill says 16384 -- the same operator on the same
                        # shapes at two unrelated amounts of work.
                        "scalars": op.get("scalars"),
                        "occurrences": 1,
                        "reason": detail,
                    }
                else:
                    entry["occurrences"] += 1
                continue
            seconds = float(record["seconds"])
            total += seconds
            launches += max(1, len(record.get("kernels") or {}))
            # Three counts, decided here and not by subtraction downstream, and
            # both markers are read rather than inferred. A zero *time* is not
            # a zero-work operator -- a measurement can round to zero and a fit
            # can land on it -- so only the record's own declaration promotes
            # one, in the same order `coverage_split` uses.
            if record.get(ZERO_WORK_FLAG):
                zero_work += 1
            elif record.get(INTERPOLATED_FLAG):
                interpolated += 1
            else:
                measured += 1
            sources[detail] = sources.get(detail, 0) + 1
        return (total,
                Coverage(operators=len(ops), measured=measured, seconds=total,
                         interpolated=interpolated, zero_work=zero_work,
                         refused=refused, reasons=reasons, sources=sources,
                         refused_signatures=tuple(by_key.values())),
                launches)

    def describe(self) -> str:
        partial = f", PARTIAL: {'; '.join(self.partial)}" if self.partial else ""
        conflict = (f", {len(self.conflicts)} signatures disagree across runs"
                    if self.conflicts else "")
        scoped = sum(1 for recs in self._prices.values() if len(recs) > 1)
        multi = (f", {scoped} held in more than one scope" if scoped else "")
        with_layout = sum(1 for recs in self._prices.values()
                          if any("layout" in r for r in recs))
        shifted = sum(self.address_shifted.values())
        moved = (f", {shifted} lookups answered under a shifted allocation"
                 if shifted else "")
        return (f"PriceLibrary({len(self._prices)} cost keys from "
                f"{len(self.sources)} runs, {with_layout} with a "
                f"recorded layout{multi}{moved}{partial}{conflict})")


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


def _frozen(value):
    """A hashable copy of a JSON value, for use inside a cache key."""
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _frozen(v)) for k, v in value.items()))
    return value


def _binding_key(graph):
    """The part of a bound graph's price key that binding can move.

    `BOUND_FIELDS` are reproduced exactly from the derived graph, so the only
    component of `signature_of` a bind rewrites is ``context`` -- and that is
    where the allocator's `slot_mapping` and state indices land. Measured on
    the 27B decode-32 graph: a second valid allocation for the same shape moves
    64 of 2439 signatures and takes the priced step from 32.667 ms over 2424
    priced operators to 28.360 ms over 2376. A cache keyed on the shape alone
    would have answered 32.667 ms, with complete coverage, for a step that is
    neither.

    `block_tables` is left out for the same reason `signature_of` leaves it
    out: it says which blocks are walked, not how many. Keeping it in would
    move the key every single step and cost the cache its whole purpose.

    Only operators that carry a context are looked at -- a handful per graph --
    so this is cheap enough to compute on every step, which is what makes it
    safe to reuse the arithmetic over the other ~2375.
    """
    if not graph:
        return ()
    out = []
    for index, op in enumerate(graph.get("ops") or ()):
        context = op.get("context")
        if not context:
            continue
        entries = tuple(
            (k, _frozen(v)) for k, v in (tuple(x) for x in context)
            if k != "block_tables")
        if entries:
            out.append((index, entries))
    return tuple(out)


#: How many refusals one process will write before it stops writing. A refusal
#: raises, so a served run normally produces exactly one -- but a caller that
#: catches and continues would otherwise fill a disk with the same graph, and
#: a diagnostic that can take a node down is not one anybody will leave on.
_DUMPS_WRITTEN = 0
_DUMP_LIMIT = 4


def _dump_refusal(shape: StepShape, graph, head_graph, coverage) -> None:
    """Write down the step that was refused, if anybody asked to see it.

    Off unless ``COMPASS_REFUSAL_DUMP`` names a directory, so the served path
    is byte-identical to before for every run that does not want this. What it
    preserves is the evidence a refusal destroys today: `describe()` keeps five
    names and a count, and the graph the names came from -- the actual derived
    body, at the actual widths this step's own allocation bound into it -- is
    dropped on the way out. Re-deriving it later from a log line is guesswork,
    and a family cannot be modelled from a name.

    Failures here are swallowed on purpose. This runs one line before a
    ``ValueError`` that the caller is expecting; turning an unwritable
    directory into a different exception would hide the refusal behind the
    diagnostic meant to explain it.
    """
    global _DUMPS_WRITTEN

    import os

    target = os.environ.get("COMPASS_REFUSAL_DUMP", "")
    if not target or _DUMPS_WRITTEN >= _DUMP_LIMIT:
        return
    try:
        from dataclasses import asdict

        _DUMPS_WRITTEN += 1
        os.makedirs(target, exist_ok=True)
        path = os.path.join(
            target,
            "refusal_b%d_t%d_%d.json" % (
                shape.batch_size, shape.total_tokens, _DUMPS_WRITTEN))
        payload = {
            # The step as the oracle was given it -- lengths per request, not
            # reduced, because which request is long is the whole question at
            # a mixed batch.
            "shape": asdict(shape),
            "coverage": coverage.as_dict(),
            # Both regions, whole. The refused operators are in here at their
            # real shapes and dtypes, which is what a price acquisition or a
            # family model is actually written against.
            "body_graph": graph,
            "head_graph": head_graph,
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, default=str)
        logger.warning(
            "ATOMCompass WARNING: refusal evidence written to %s", path)
    except Exception as error:  # noqa: BLE001 - see docstring
        logger.warning(
            "ATOMCompass WARNING: could not write refusal evidence: %s",
            error)


def _price_key(shape: StepShape, graph, head_graph):
    """What two steps must share for their priced body and head to be equal.

    `StaticGraphs.key` says what two steps must share for their derived *graph*
    to be equal, and a price is keyed on operator names, tensor shapes and
    dtypes -- never on tensor values -- so equal derived graphs price equally.

    Two things are added. `produces_output` decides whether the LM head runs at
    all, so it changes the priced step where the body graph is identical. And
    `_binding_key` carries what this step's own allocation wrote into the
    graph, for both regions, because that is in the price key too.
    """
    return (StaticGraphs.key(shape) + (bool(shape.produces_output),)
            + (_binding_key(graph), _binding_key(head_graph)))


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
        #: Priced body and head per shape, because summing a price over every
        #: operator is where a replayed step's CPU time actually goes: measured
        #: on the 27B decode-32 shape, 39ms of a 41.7ms estimate against a
        #: 32.7ms modelled GPU step, over ~2440 operators. A serving replay asks
        #: the same few bucketed shapes thousands of times, and pricing them
        #: again each time is the whole gap to the GPU-free speedup gate.
        #:
        #: The bind is *not* cached: every step still derives or binds its own
        #: graph, so every refusal -- a stale allocation, a shape outside the
        #: template's rows, a rank mismatch -- still fires per step. What is
        #: reused is only the arithmetic over an identical graph.
        self._priced: dict = {}
        self.price_cache_limit = 4096
        self.price_cache_hits = 0
        self.price_cache_misses = 0

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
        # Every step: the head's composition refusals and its own bind are not
        # arithmetic, so they cannot sit behind the cache. A head graph that
        # this step cannot be given has to say so on the thousandth step as
        # loudly as on the first.
        head_graph = self._head_graph_for(graph, shape)
        key = _price_key(shape, graph, head_graph)
        priced = self._priced.get(key)
        if priced is None:
            body, coverage, launches = self.library.body(
                graph, self.body_registration)
            if head_graph is None:
                head, head_coverage, head_launches = 0.0, None, 0
            else:
                head, head_coverage, head_launches = self.library.body(
                    head_graph, self.head_registration)
            if head_coverage is not None:
                # One step, one coverage record: a head whose all-gather is
                # unpriced has to make the *step* incomplete, not sit in a
                # second record that `require_complete` never reads.
                coverage = coverage.merged(head_coverage)
            self.price_cache_misses += 1
            if len(self._priced) < self.price_cache_limit:
                self._priced[key] = (body, coverage, launches, head,
                                     head_coverage, head_launches)
        else:
            body, coverage, launches, head, head_coverage, head_launches = priced
            self.price_cache_hits += 1
        self.last_coverage = coverage
        if self.require_complete and not coverage.complete:
            _dump_refusal(shape, graph, head_graph, coverage)
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
        """The head region of this step: seconds, coverage and launches."""
        head_graph = self._head_graph_for(graph, shape)
        if head_graph is None:
            return 0.0, None, 0
        return self.library.body(head_graph, self.head_registration)

    def _head_graph_for(self, graph: dict, shape: StepShape):
        """This step's head graph, or None if it has no head to charge.

        Split out from pricing it so that the checks below run on every step
        while the arithmetic over the head's operators can be reused. They are
        different kinds of claim: one is about whether this step may be given
        a head at all, the other is about what that head costs.

        Absent -- `None` -- for two reasons that are not gaps: no
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
            return None
        if not shape.produces_output:
            # `is_pure_middle_chunk(batch)` -> `logits = None`
            # (model_runner.py:3174). Nothing is projected and nothing is
            # sampled, so there is no head to charge.
            return None
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
        return head_graph

    def describe(self) -> str:
        head = ("no head region" if self.head_graphs is None
                else f"head {self.head_graphs.describe()}")
        return (f"LibraryCostOracle({self.library.describe()}, "
                f"{self.graphs.describe()}, "
                f"+{self.seconds_per_launch * 1e6:.2f}us/launch, "
                f"{head}, "
                f"runner {self.extra_seconds * 1e3:.3f}ms)")
