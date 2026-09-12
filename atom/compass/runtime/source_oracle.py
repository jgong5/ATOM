"""Build the frozen source composition from arguments a command line can carry.

`LibraryCostOracle` takes a `PriceLibrary`, a `GraphSource` and a
`RunnerRegions` -- three live objects. `CompassConfig.oracle_options` is a dict
parsed out of repeated `--compass-oracle-option KEY=VALUE` flags, whose values
are strings that `arg_utils` turns into ints and floats where they look like
one. There is no value a served run can put on that command line that becomes
any of those three objects, so the composition the frozen diagnostics use has
been reachable from `predict_step.py` and from nothing else.

This is the seam between the two. `source_cost_oracle` names only parameters a
command line can carry -- paths, names, numbers, flags -- and returns the same
`LibraryCostOracle` `predict_step.py` builds, from the same helpers. It is a
factory rather than a subclass because what the CLI cannot express is the
*construction*, not the behaviour: an oracle that behaved differently here
would make a served prediction a different claim from a frozen one, which is
the thing the composition exists to prevent.

    --compass-oracle atom.compass.runtime.source_oracle.source_cost_oracle
    --compass-oracle-option model=/models/Qwen3.8-27B
    --compass-oracle-option tp=1
    --compass-oracle-option block_size=16
    --compass-oracle-option max_model_len=262144
    --compass-oracle-option position_rows=3
    --compass-oracle-option price=prices.json:graph.json:unregistered
    --compass-oracle-option template=graph.json,hgraph.json
    --compass-oracle-option regions=source-27b-tp1-conc-v2

Repeatable arguments arrive as one comma-separated string, because the flag
that carries them is `KEY=VALUE` and a repeated key would overwrite rather than
append. A programmatic caller may pass a real list instead; both are read the
same way, and a path containing a comma is refused rather than split wrongly.

What it refuses, rather than defaulting:

  * an unknown region model name -- `region_model` raises, because running
    without the runner's regions is a different prediction, not a lesser one;
  * a flag-valued option that is neither a boolean nor `0`/`1`/`true`/`false`
    -- `head=yes` must not silently mean the head was priced;
  * `derive=0` together with no template, which can only ever refuse every
    shape, reported here rather than as a hundred identical refusals later.

It does not accept `extra_seconds`. That term and a region model are the same
quantity measured two ways, and `LibraryCostOracle` already refuses both at
once; a served run asking for the source composition wants the measured one.
"""

from typing import NamedTuple, Optional

__all__ = ["source_cost_oracle", "build_source_oracle", "build_source_group",
           "SourceComposition", "SourceGroup", "RankGroupOracle",
           "price_specs", "template_shape", "seeded_graphs", "gap_ratio",
           "region_snapshot", "region_values", "REGION_SNAPSHOT_SCHEMA"]


def _entries(value, what: str):
    """A repeatable argument, as a list or as one comma-separated string."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if not isinstance(value, str):
        raise ValueError(f"{what}: expected a path or a comma-separated list, "
                         f"got {type(value).__name__}")
    return [part.strip() for part in value.split(",") if part.strip()]


def _flag(value, what: str) -> bool:
    """A boolean from what a `KEY=VALUE` command line can actually carry.

    `arg_utils` converts a value that parses as a number, so `head=1` arrives
    as `int` and `head=true` arrives as `str`. Both are accepted; anything else
    is refused, because a flag that silently reads false is a prediction
    missing a region with nothing in the record to say so.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
    elif isinstance(value, str):
        if value.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if value.strip().lower() in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"{what}: expected a boolean, 0/1 or true/false, "
                     f"got {value!r}")


#: "interpolate at whatever density the provider itself declares". A sentinel
#: rather than the number, so turning interpolation on from a command line does
#: not restate a constant that lives in the provider and could drift from it.
_DEFAULT_GAP_RATIO = "provider default"


def gap_ratio(value, what: str = "interpolate"):
    """The declared sampling density, or `None` for exact prices only.

    A ratio and not a flag, because that is the thing the caller is actually
    asserting: the widest ratio between two adjacent measured row counts their
    evidence supports interpolating across. ``true`` is accepted as the
    family default so a command line can turn it on without also choosing a
    number, and a ratio below 1 is refused rather than clamped -- it would name
    a gap narrower than no gap at all.
    """
    if value in (None, "", False, 0):
        return None
    if value is True:
        return _DEFAULT_GAP_RATIO
    if isinstance(value, str) and value.strip().lower() in (
            "1", "true", "yes", "on"):
        return _DEFAULT_GAP_RATIO
    if isinstance(value, str) and value.strip().lower() in (
            "false", "no", "off"):
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{what}: expected a max gap ratio, true, or off, "
            f"got {value!r}") from None
    if ratio < 1.0:
        raise ValueError(f"{what}: {ratio} is narrower than adjacent measured "
                         "points, so nothing could ever be interpolated")
    return ratio


#: The record :func:`region_snapshot` produces. Versioned for the same reason
#: the budget record is: a validator reads it, and a field that quietly changes
#: meaning is worse than one that is absent.
REGION_SNAPSHOT_SCHEMA = "compass.regions.selected/1"


def region_values(value):
    """Every value a region preset holds, in a form JSON keeps whole.

    Dataclasses become their fields and tuples become lists. A mapping whose
    keys are not strings -- a ``(capture_bucket, padded) -> Measured`` table,
    say -- becomes a sorted list of key/value pairs rather than being coerced:
    `json.dumps` cannot write a tuple key, so it either raises or a careless
    serialiser flattens it to a string, and both lose the coefficient. A
    coefficient outside the digest is a number nobody is holding the run to.

    Read off the object rather than from a list of fields kept here, so a
    preset that grows a table -- a prefill cell map, a new region -- is
    carried without this having to learn about it first.
    """
    import dataclasses
    import json as _json

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: region_values(item) for key, item in value.items()}
        return {"__pairs__": sorted(
            ([region_values(key), region_values(item)]
             for key, item in value.items()),
            key=lambda pair: _json.dumps(pair[0], sort_keys=True))}
    if isinstance(value, (list, tuple)):
        return [region_values(item) for item in value]
    return value


def region_snapshot(name: str, model) -> dict:
    """The region preset a run selected, as a value rather than as a name.

    ``model`` is the object the factory is holding -- the one it will price
    with -- and not a name to look up again. Resolving the name a second time
    at the end of construction would be the very pattern this work removes: a
    record derived from an option rather than from the thing that was used, so
    that a preset swapped in between would be priced from and not reported.
    It is snapshotted where it is selected, for the same reason a file's
    digest is taken where its bytes are parsed.

    A region model supplies preparation and postprocess -- everything in the
    step that is not the body and not the head -- from measured coefficients.
    Those coefficients go straight into every predicted duration, and until
    now the record said only which *name* was asked for. A name is not a
    measurement: the preset behind it can be edited, and two runs quoting the
    same name can have been priced from different numbers with nothing to show
    it.

    So the numbers are snapshotted where they are selected, and digested. This
    is deliberately **not** a `LoadedInput` and must not be filed as one: no
    file was read. The preset is built into the code, `sha256` here is over a
    canonical serialisation of its own values, and a validator that treated it
    as a file read would go looking for bytes on disk that never existed. The
    distinction is why this has its own schema and its own key in the manifest
    rather than joining `inputs`.

    ``"none"`` snapshots as a selection of nothing. That is a real choice --
    body plus head with no runner term -- and it contributes no coefficients,
    so there is nothing to attribute and nothing is required of it. Whether it
    is *allowed* in an acceptance cell is `check_source_factory`'s question,
    and it already answers no.

    Every name that selects this preset is recorded beside the one that was
    asked for, so an alias cannot make two records of one preset look like
    records of two.
    """
    import hashlib
    import json as _json

    from atom.compass.core.cost.regions import REGION_MODELS

    # Aliases by identity against the object in hand, so two names for one
    # preset cannot read as records of two.
    aliases = sorted(key for key, value in REGION_MODELS.items()
                     if value is model and value is not None)
    snapshot = {
        "schema": REGION_SNAPSHOT_SCHEMA,
        "requested": str(name),
        "aliases": aliases,
        "version": str(getattr(model, "version", "") or ""),
        "provenance": str(getattr(model, "provenance", "") or ""),
        # Every field of the preset, coefficients and calibrated domain alike.
        # The domain is part of what was selected: the same numbers over a
        # wider domain is a different claim about where they hold.
        # Read off the dataclass rather than listed here, so a preset that
        # grows a field -- a prefill cell table, a new region -- is carried
        # without this needing to know about it.
        "parameters": None if model is None else region_values(model),
    }
    body = _json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    snapshot["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return snapshot
def _attention_request_scope(value, coords=None, what="attention_scope"):
    """The deployment the request is asking for a ragged attention price IN.

    A ragged attention law is identified by the deployment it was measured
    under -- KV dtype and layout, block size, sliding window, the backend the
    dispatcher took, and for linear attention the state geometry. None of that
    is derivable from the operator, so a request that does not declare it is
    refused rather than answered by whichever law happens to be fitted. This
    is where that declaration enters: a mapping, or a path to the JSON file
    whoever resolved the deployment wrote.

    The file is read through `loaded_input.load_json`, the same reader the
    price lists go through, and for the same reasons: this scope decides which
    laws a run is allowed to price from, so what it was is part of what the
    run was. That reader resolves the rank's own file from the stem, digests
    the exact bytes it parsed, and hands back a record of both -- which is
    returned here so the composition can carry it beside the prices instead of
    the manifest describing a deployment nobody can check.

    Nothing is inferred and nothing is defaulted. A missing file is an error,
    because a run that asked to price attention under a named deployment and
    silently got no deployment at all would read as an honest refusal of the
    whole family.
    """
    from atom.compass.core.cost.families.attention_scope import declaration_of

    if not value:
        return None, None
    if isinstance(value, dict):
        return declaration_of(value, where="the %s mapping" % what), None
    requested = str(value).strip()
    if not requested:
        return None, None
    from atom.compass.core.loaded_input import load_json

    try:
        payload, loaded = load_json(requested, role="oracle." + what,
                                    coords=coords)
    except FileNotFoundError as exc:
        # Named with the resolution, not with the stem alone: at TP>1 the
        # stem is what the option said and the resolved name is what this
        # rank went looking for, and a reader who cannot see the second
        # cannot tell a missing file from a rank suffix nobody wrote.
        raise ValueError(
            "%s names %r, which resolved to %s and is not a file "
            "that exists. The deployment a price is asked for has to come "
            "from something that recorded it." % (what, requested, exc.filename)
        ) from exc
    return declaration_of(payload, where=loaded.path), loaded




def _price_library(entries, gap_ratio, coords=None, attention_scope=None,
                   measured_attention_scope=None):
    """The exact-signature library, or the family provider in front of it.

    The provider is a subclass that overrides `lookup` alone, so everything
    downstream -- `PriceLibrary.body`, the `Coverage` split, `LibraryCostOracle`
    -- is the same code either way, and a derived price arrives carrying its own
    ``interpolated`` marker rather than being counted as a measurement.

    Off by default. An interpolated price is a claim about a row count nobody
    ran, and a run that did not ask for one should not silently get one.

    ``coords`` goes to the library unresolved, and the library resolves as it
    reads. Both branches go through ``add``, so there is one call site
    carrying it rather than two.

    ``measured_attention_scope`` is the other end of the same question:
    ``attention_scope`` says what deployment a price is being ASKED for, and
    this says what deployment a price list that does not record one was TAKEN
    in. A primitive price file written by `scripts/compass/primitives.py`
    records the pools it allocated and the kernel its dispatch probe saw, but
    in sections the family adapter does not read as a scope, so without this
    every ragged observation in it is refused for want of a declared backend
    and the family fits over nothing at all. It fills silences only -- a file
    that states a fact and a declaration that contradicts it raise -- and the
    declaration is carried into the manifest as a loaded input, because a law
    fitted under declared facts is worth exactly what the declaration is.
    """
    from atom.compass.core.cost.library import PriceLibrary

    if gap_ratio is None:
        library = PriceLibrary()
    else:
        from atom.compass.core.cost.families import ParametricPriceLibrary

        library = (ParametricPriceLibrary()
                   if gap_ratio is _DEFAULT_GAP_RATIO
                   else ParametricPriceLibrary(max_gap_ratio=gap_ratio))
    declaration, loaded = _attention_request_scope(attention_scope, coords)
    if declaration is not None:
        if gap_ratio is None:
            raise ValueError(
                "attention_scope declares the deployment a MODELLED attention "
                "price would be asked for, and modelling is off. Turn the "
                "family provider on with interpolate, or drop the scope.")
        # The per-family declaration, not one flattened mapping: the families
        # are identified by different facts, and a linear attention call
        # refused over a KV layout it never reads would be refused for a
        # reason that does not apply to it.
        library.request_attention_scope = declaration
        if loaded is not None:
            # Recorded beside the prices in the same manifest. A run that
            # answered from a law depended on these bytes as much as on the
            # price list, and a manifest that omits them describes a
            # deployment nobody can check afterwards.
            library.loaded_inputs = library.loaded_inputs + (loaded,)
    measured, measured_loaded = _attention_request_scope(
        measured_attention_scope, coords, what="measured_attention_scope")
    if measured is not None:
        if gap_ratio is None:
            raise ValueError(
                "measured_attention_scope declares the deployment these price "
                "lists were MEASURED under, which only a fitted family reads. "
                "Turn the family provider on with interpolate, or drop it.")
        library.declared_attention_scope = measured
        if measured_loaded is not None:
            library.loaded_inputs = library.loaded_inputs + (measured_loaded,)
    extra = {"coords": coords} if coords else {}
    for entry in entries:
        if isinstance(entry, (tuple, list)):
            library.add(entry[0], entry[1] if len(entry) > 1 else None,
                        entry[2] if len(entry) > 2 else None, **extra)
        else:
            library.add(entry, None, None, **extra)
    return library


def _rank_coords(value):
    """This rank's coordinates, from what a `KEY=VALUE` command line carries.

    ``tp:2`` or ``tp:2,dp:1``; a programmatic caller may pass the dict
    directly. Refused rather than guessed at, because a coordinate map read
    wrongly is worse than none at all: it sends every artifact lookup to a
    rank's file that is not this rank's, and both the read and the prediction
    succeed.
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    coords = {}
    for entry in _entries(value, "rank_coords"):
        group, sep, index = entry.partition(":")
        if not sep:
            group, sep, index = entry.partition("=")
        if not sep or not group.strip():
            raise ValueError(
                f"rank_coords {entry!r}: expected GROUP:INDEX, as in tp:2")
        try:
            coords[group.strip()] = int(index)
        except ValueError:
            raise ValueError(
                f"rank_coords {entry!r}: {index!r} is not a rank index") from None
    return coords


def price_specs(entries, coords=None):
    """``prices.json[:graph.json[:regime]]`` triples, as `PriceLibrary` takes.

    The graph is optional and worth supplying: without it a price matches by
    signature alone, and a signature does not carry layout. The third element
    names the collective registration regime for a list whose provenance
    predates the field.

    With ``coords``, each path is resolved through
    :func:`~atom.compass.core.artifacts.resolve_rank_path` -- the same
    convention the write side uses and the calibrated oracle already reads, so
    a rank's own price list is preferred where one was written and the
    unsuffixed one is used where it was not. That fallback is the predeclared
    semantics of a symmetric TP group and is not changed here.
    """
    from atom.compass.core.artifacts import resolve_rank_path

    def resolved(path):
        return resolve_rank_path(path, coords)[0] if path and coords else path

    loaded = []
    for entry in entries or ():
        parts = entry.split(":")
        if len(parts) == 1:
            loaded.append((resolved(parts[0]), None))
        elif len(parts) == 2:
            loaded.append((resolved(parts[0]), resolved(parts[1]) or None))
        elif len(parts) == 3:
            loaded.append((resolved(parts[0]), resolved(parts[1]) or None,
                           parts[2]))
        else:
            raise ValueError(f"price {entry!r}: expected at most "
                             "prices.json:graph.json:regime")
    return loaded


def _coords(saved):
    """A coordinate map as saved: a dict, or the pair list JSON turned it into."""
    if not saved:
        return {}
    if isinstance(saved, dict):
        return dict(saved)
    return {k: v for k, v in saved}


def template_shape(graph: dict):
    """The structure a graph on disk is a graph of, from its own provenance.

    A derived artifact records the batch it was traced over
    (`provenance.batch_spec`). Reading it back is the only way to key a
    pre-derived graph that does not involve the caller restating the batch and
    getting it wrong -- and a graph whose provenance does not carry one cannot
    be used as a template at all, because nothing says what it is a template
    *for*.
    """
    from atom.compass.core.cost.base import StepShape

    prov = graph.get("provenance") or {}
    spec = prov.get("batch_spec")
    if not spec:
        raise ValueError(
            "this graph records no batch_spec, so nothing says which structure "
            "it is a template for. Derive it with --batch-spec.")
    key = graph.get("key") or {}
    # A saved key writes its coordinate dicts as pair lists, because JSON has
    # no dict-of-tuples. Reading one back as a dict is not a conversion, it is
    # the inverse of how it was written.
    topology = _coords(key.get("topology"))
    rank_coords = _coords(key.get("rank_coords"))
    queries = tuple(spec["query_lens"])
    contexts = tuple(spec["context_lens"])
    prefill = 0 if spec.get("kind") == "decode" else sum(queries)
    # Which bucket a graph was captured at is an execution fact, and that is
    # where the tracer records it -- not in the batch spec, which describes the
    # requests. A graph derived with no bucket declared keys as None: it is a
    # template for the uncaptured structure, and claiming a bucket it was not
    # traced at would make it answer for a padded graph nobody derived.
    execution = prov.get("execution") or {}
    return StepShape(
        num_scheduled_tokens=queries,
        context_lens=contexts,
        num_prefill_tokens=prefill,
        topology=topology, rank_coords=rank_coords,
        capture_bucket=execution.get("capture_bucket"),
        compiled=None,
        produces_output=True,
    )


def seeded_graphs(paths, derive, allocation, coords=None, *,
                  role="oracle.template", collect=None, cudagraph_mode=None):
    """A `TemplateGraphs` holding the graphs already on disk, keyed by spec.

    ``coords`` resolves each path to this rank's file where one was written,
    by the same rule as the prices. It does *not* rewrite the key: a template
    is keyed by the coordinates in its own provenance, because that is the rank
    the graph is a graph of. Re-keying one to the rank that happens to be
    reading it would erase the difference between a graph derived for this rank
    and one borrowed from the representative. `TemplateGraphs` serves the
    borrow deliberately, on a miss, and counts it as a representative hit.

    ``collect``, where given, is a list each template's `LoadedInput` is
    appended to under ``role``. The identity is taken here because here is
    where the bytes are parsed: a caller that digested these paths afterwards
    would describe whatever is at them *then*, and would be digesting the stem
    the option carried rather than the per-rank file that was served.

    ``cudagraph_mode`` is the deployment's declared mode and goes to the cache,
    not to the templates: it is the same value the deriver is built with, and
    handing it to only one of the two would let a warm bind resolve a FULL
    step's launch extent differently from the derivation that produced its
    graph. A template read off disk says nothing about it -- the mode belongs
    to the run being priced, not to the run that traced the graph.
    """
    from atom.compass.core.loaded_input import load_json
    from atom.compass.runtime.templates import TemplateGraphs, template_key

    graphs = {}
    for requested in paths:
        graph, loaded = load_json(requested, role=role, coords=coords)
        if collect is not None:
            collect.append(loaded)
        graphs[template_key(template_shape(graph))] = graph
    return TemplateGraphs(graphs, derive=derive, allocation=allocation,
                          cudagraph_mode=cudagraph_mode)


class SourceComposition(NamedTuple):
    """The oracle, and the parts a report needs to describe how it answered.

    `LibraryCostOracle` holds its own graph sources, so most of this is
    reachable from the oracle alone -- but the deriver is held privately by the
    cache it feeds, and how long the model took to load is not held anywhere.
    A caller that wants to report derivation cost builds through
    :func:`build_source_oracle`; one that only wants to price steps calls
    :func:`source_cost_oracle` and gets the oracle by itself.
    """

    oracle: object
    body_graphs: object
    head_graphs: object
    deriver: object
    build_seconds: float
    #: The carried-allocation approximation, if one was asked for. Kept here
    #: rather than read back off the cache, because it is an explicitly
    #: unmeasured assumption and a report has to be able to say it was made.
    allocation: object = None
    #: Which rank this composition was built to serve, as the runner supplied
    #: it. :func:`build_source_oracle` always fills this in, empty under TP1
    #: where the runner passes nothing and there is one rank to be. ``None``
    #: means a composition assembled by hand rather than built.
    rank_coords: Optional[dict] = None
    #: Whether each resolved artifact was this rank's own file or the shared
    #: unsuffixed one, by path. A report that says "rank 3" has to be able to
    #: say which of rank 3's files actually existed.
    rank_artifacts: Optional[dict] = None
    #: The widest ratio between adjacent measured row counts a fitted price was
    #: allowed to span, read back off the provider that holds it rather than
    #: off the argument that asked for it -- ``interpolate=true`` names no
    #: number, and the number is the part a reader has to be able to check.
    #: ``None`` means no price could be fitted at all.
    interpolation_limit: float | None = None
    #: Every artifact this composition actually loaded, as the reader that
    #: parsed it described it -- a `LoadedInput` per file, carrying the digest
    #: of the bytes that were parsed and which rank's file they came from.
    #:
    #: Distinct from `rank_artifacts`, which answers "was this rank's own file
    #: there?" from the paths alone and can be computed without opening
    #: anything. This answers "what did this rank load?", which no later
    #: reader can reconstruct: the option is a DSL over stems, and the files
    #: it names can change after the load.
    loaded_inputs: tuple = ()


class RankGroupOracle:
    """One composition per rank of the group, selected by the shape's rank.

    `rank_aggregation="slowest"` prices a step on every logical rank and keeps
    the maximum, because a deployment's step ends when its slowest rank ends
    and the ranks are not interchangeable -- the TP4 head measurements run
    13.510 / 16.158 / 13.459 / 13.452 ms. It did that by moving
    ``StepShape.rank_coords`` and asking *one* oracle, whose price library had
    been loaded once, at the executor's own coordinates. `LibraryCostOracle`
    does not select prices by rank: it looks a signature up in the library it
    holds. So every rank was priced from rank 0's tables and the outlier that
    motivates the policy could not appear in the answer.

    This holds one real oracle per rank, each built from that rank's own
    resolved artifacts, and dispatches on the coordinate the shape carries.
    The expensive part is not repeated: the model is traced once and every
    rank's oracle shares those derivers.

    Everything that is not `estimate` delegates to the representative rank, so
    `describe`, coverage and the reporting surface behave as they did --
    including `native_allocation`, and that one is worth being exact about.

    The predict mixin does not *set* `native_allocation`; it reads the provider
    off the oracle and calls `offer()` on the provider itself, once per step.
    The provider is held inside each rank's `TemplateGraphs`, so a wrapper that
    fanned out an assignment would be fanning out the wrong thing -- the
    objects already inside the other ranks' template sources would never see
    the batch, and every rank above the representative would refuse the step
    for want of an allocation. So the ranks are built sharing *one* provider.
    `NativeAllocation.allocation_for` already indexes by the shape's own
    coordinates, so one object serves the whole group correctly, and one
    `offer()` reaches all of it.
    """

    def __init__(self, by_rank: dict, representative: int = 0) -> None:
        self._by_rank = dict(by_rank)
        self._representative = representative
        #: Which ranks `estimate` has been answered from. A set, not a log:
        #: this is asked once by a report and would otherwise grow by one
        #: entry per rank per step for the length of a run. What it
        #: distinguishes is "every rank was asked" from "rank 0 was asked
        #: four times", and that needs the ranks, not their order.
        self.selected_ranks: set = set()

    @property
    def representative(self):
        return self._by_rank[self._representative]

    def oracle_for(self, rank: int):
        """This rank's own oracle, or an error naming the rank that has none.

        Fails closed. The shared-file fallback for a rank that wrote no
        artifacts of its own already happened, during construction, in
        `resolve_rank_path` -- that rank has a real composition built from the
        shared tables. A rank with no composition at all is a rank outside the
        group this oracle was built for, and answering it from another rank's
        prices is precisely the substitution this class exists to stop.
        """
        rank = int(rank)
        if rank not in self._by_rank:
            raise LookupError(
                f"this oracle was built for ranks {sorted(self._by_rank)} and "
                f"was asked to price rank {rank}. A rank with no composition "
                f"has no prices of its own, and another rank's are prices of "
                f"different work.")
        return self._by_rank[rank]

    def estimate(self, shape):
        rank = int((shape.rank_coords or {}).get("tp", self._representative))
        oracle = self.oracle_for(rank)
        self.selected_ranks.add(rank)
        return oracle.estimate(shape)

    def __getattr__(self, name):
        # Reached only for names this class does not define, so `estimate`
        # never lands here. Keeps `describe`, the coverage split, the
        # allocation provider and anything else a report reads working
        # unchanged, answered by the rank the composition represents.
        return getattr(self.representative, name)


class SourceGroup(NamedTuple):
    """Every rank's composition, and the oracle that selects between them."""

    oracle: object
    #: rank index -> that rank's `SourceComposition`. Empty at TP1, where the
    #: group is one rank and `oracle` is that rank's own oracle.
    by_rank: dict
    #: The union of what every rank loaded. Each `LoadedInput` carries the
    #: coordinates of the rank that read it, so this stays per-rank detail
    #: rather than becoming an undifferentiated set.
    loaded_inputs: tuple = ()
    #: What the derivation cost, counted once for the group rather than once
    #: per rank: the model is traced once and the derivers are shared.
    build_seconds: float = 0.0


def build_source_group(*, tp: int = 1, rank_coords=None, head=False, **kwargs):
    """The group's oracle: one real composition per rank, one derivation.

    At TP1 this is exactly :func:`build_source_oracle` and returns that
    composition's own oracle -- there is one rank, and wrapping it would add a
    layer with nothing to select between.

    Above TP1 every rank gets its own composition, so each resolves its own
    price list and its own templates through `resolve_rank_path`. Only the
    artifact reads are repeated; the model is traced once, by rank 0's build,
    and every other rank is handed those derivers. That matters: a whole-model
    build per rank would be four builds of a 27B model to read four small JSON
    files, and the build cost would then be counted four times in a record
    whose whole purpose is to be accountable.
    """
    width = int(tp or 1)
    coords = _rank_coords(rank_coords)
    representative = int(coords.get("tp", 0))
    if width <= 1:
        built = build_source_oracle(tp=width, rank_coords=rank_coords,
                                    head=head, **kwargs)
        return SourceGroup(built.oracle, {}, built.loaded_inputs,
                           built.build_seconds)

    first = build_source_oracle(tp=width, rank_coords={"tp": representative},
                                head=head, **kwargs)
    shared = (first.deriver, _head_deriver_of(first))
    by_rank = {representative: first}
    for rank in range(width):
        if rank == representative:
            continue
        by_rank[rank] = build_source_oracle(
            tp=width, rank_coords={"tp": rank}, head=head,
            _shared_derivers=shared,
            # One provider for the group, not one per rank. The runner offers
            # this step's assignment by calling `offer()` on the object
            # itself, and that object lives inside each rank's
            # `TemplateGraphs` -- so ranks holding their own copies would
            # never be offered anything and would refuse every step.
            _shared_allocation=first.allocation, **kwargs)

    loaded: list = []
    for rank in sorted(by_rank):
        loaded.extend(by_rank[rank].loaded_inputs)
    oracle = RankGroupOracle({r: c.oracle for r, c in by_rank.items()},
                             representative)
    oracle.compass_loaded_inputs = tuple(loaded)
    return SourceGroup(oracle, by_rank, tuple(loaded), first.build_seconds)


def _head_deriver_of(composition):
    """The head deriver a composition built, where it built one.

    `SourceComposition` carries the body deriver by name and the head one only
    inside `head_graphs`, which is where `TemplateGraphs` keeps it.
    """
    return getattr(composition.head_graphs, "_derive", None)


def source_cost_oracle(*, rank_coords=None, **kwargs):
    """The frozen composition as a plain oracle, for `oracle_qualname`.

    This is the entry point a served run names. It takes exactly the arguments
    :func:`build_source_oracle` documents and returns only the oracle, because
    that is what `_build_oracle` expects to get back.

    Above TP1 that oracle is the group's, not one rank's. A served run has one
    executor standing in for the whole group, and `rank_aggregation="slowest"`
    asks it to price every rank; it can only answer that honestly if every
    rank's artifacts were actually loaded.

    ``rank_coords`` is named explicitly rather than swept into ``**kwargs``,
    and that is the whole reason it is in this signature: `_build_oracle`
    offers the rank only to an oracle that names it, by
    ``inspect.signature(...).parameters``, and a bare ``**kwargs`` names
    nothing. Under TP>1 the injection silently did not fire, so every rank of
    a served run built the same composition and resolved the same artifacts --
    rank 0's, wherever a per-rank file existed.
    """
    return build_source_group(rank_coords=rank_coords, **kwargs).oracle


def build_source_oracle(
    *,
    model: Optional[str] = None,
    tp: int = 1,
    device: str = "meta",
    replay_target: Optional[str] = None,
    block_size: int = 0,
    max_model_len: int = 0,
    position_rows: int = 1,
    block_policy: str = "rounds",
    cudagraph_mode: Optional[str] = None,
    price=None,
    template=None,
    head_template=None,
    head=False,
    regions: str = "source-27b-tp1",
    seconds_per_launch: float = 0.0,
    require_complete: bool = True,
    carry_allocation: bool = False,
    allocation: str = "",
    derive: bool = True,
    interpolate=None,
    attention_scope=None,
    measured_attention_scope=None,
    rank_coords=None,
    _shared_derivers=None,
    _shared_allocation=None,
):
    """The frozen composition, from names and paths alone.

    ``model``/``tp``/``device``/``replay_target`` and the five derivation
    settings are what building a deriver costs; they are required only when
    ``derive`` is on, so a run that seeds every template it needs can leave the
    model out and never load one.

    ``interpolate`` selects the per-family price provider and declares the
    sampling density it may work over: the widest ratio between two adjacent
    measured row counts it may interpolate across. ``true`` takes the
    provider's own declared default; leaving it out means exact prices only,
    and an operator at a row count nobody measured is refused with the reason
    rather than answered from a fit. What it changes is which prices exist, not
    how they are counted: a fitted price arrives marked, so `Coverage` reports
    it as interpolated and `complete_measured` goes false.

    ``rank_coords`` is this rank's coordinates, as ``tp:2`` or a dict. It
    selects artifacts and nothing else, and the distinction is worth being
    exact about, because two different things are per-rank here:

    * *Artifacts* -- price lists and seeded templates -- resolve through
      `resolve_rank_path`, preferring a file written for this rank and falling
      back to the shared one. That fallback is the predeclared semantics for a
      symmetric TP group, unchanged.
    * *Derivation* does not. `ModelTracer.build` initialises a one-rank group
      and calls `simulate_group_width`, which raises the reported ``world_size``
      to the logical width and leaves ``rank_in_group`` at 0 -- its own log line
      says "deriving rank 0 of a TP%d deployment". Every graph this deriver
      produces is rank 0's shard, whatever rank the shape asks for, and the
      shape's coordinates reach only the graph's key. For a uniformly sharded
      dense model the two coincide (`num_embeddings // tp_size` is the same on
      every rank) but that is a property of the model, not a guarantee of the
      derivation, so it is declared here rather than assumed.

    The consequence a caller has to know: `template_key` includes the rank
    coordinates, and a seeded template carries the rank its provenance names.
    So at TP>1 a shape from rank 1 does not match a template derived at rank 0.
    It falls through to derivation, or -- with ``derive=0`` -- is refused. That
    is reported at build time rather than as a hundred identical misses later.

    ``attention_scope`` and ``measured_attention_scope`` are the two ends of
    the ragged-attention deployment question, and neither is defaulted: the
    first is the deployment this run is ASKING for a price in, the second the
    deployment the price lists were TAKEN in, for lists whose collector wrote
    no scope down. Both are a mapping or a path to the JSON whoever resolved
    the deployment wrote, both go through the same reader, and both are
    carried into the manifest as loaded inputs. See `_price_library`.
    """
    from atom.compass.core.cost.library import LibraryCostOracle
    from atom.compass.core.cost.regions import region_model
    from atom.compass.runtime.templates import (CarriedAllocation,
                                                 NativeAllocation)

    head = _flag(head, "head")
    require_complete = _flag(require_complete, "require_complete")
    carry_allocation = _flag(carry_allocation, "carry_allocation")
    derive = _flag(derive, "derive")
    coords = _rank_coords(rank_coords)

    templates = _entries(template, "template")
    head_templates = _entries(head_template, "head_template")
    if not derive and not templates:
        raise ValueError(
            "derive is off and no template was given, so every shape would be "
            "refused for want of a graph. Seed a template or turn derivation "
            "on.")

    requested_prices = _entries(price, "price")
    # Unresolved on the way in, resolved once by the reader as it opens the
    # file. That is what lets the record hold both ends of it: the stem the
    # option carried, and the per-rank file this rank was actually served.
    # Resolving here as well would hand the library a suffixed path it had no
    # way to recognise as a rank's own, and every record would read
    # `rank_own: false` under a name nothing asked for.
    price_entries = price_specs(requested_prices)
    library = _price_library(price_entries, gap_ratio(interpolate), coords,
                             attention_scope, measured_attention_scope)
    regions_model = region_model(regions)
    # Immediately, off the object just selected -- not from the name again.
    regions_taken = region_snapshot(regions, regions_model)
    rank_artifacts = _rank_artifacts(
        coords, requested_prices, templates, head_templates)

    allocation_choice = str(allocation or "").strip().lower()
    if allocation_choice and carry_allocation:
        raise ValueError(
            "allocation and carry_allocation both name where the block "
            "assignment comes from, and they disagree. Pass one.")
    allocation = None
    if allocation_choice == "native":
        # The bridge a served run takes: the oracle holds the source,
        # and the runner offers it the scheduler's own assignment
        # before asking for a cost. Offline -- a CLI, a diagnostic --
        # nothing offers, and every shape whose template carries
        # allocator fields is refused with that as the reason. That is
        # the intended behaviour and not a misconfiguration: an
        # unattended run stops rather than reusing a template's blocks.
        if not block_size or not max_model_len:
            raise ValueError(
                "allocation=native encodes the scheduler's block table "
                "through BatchSpec, so it needs block_size and "
                "max_model_len")
        allocation = NativeAllocation(
            block_size=int(block_size), max_model_len=int(max_model_len),
            position_rows=int(position_rows),
            # Same declared mode the derivers and the template cache get: the
            # state tail this source supplies is mode-specific.
            cudagraph_mode=cudagraph_mode)
    elif allocation_choice not in ("", "carry", "none"):
        raise ValueError(
            f"allocation must be native, carry or none, not "
            f"{allocation_choice!r}")
    if allocation_choice == "carry":
        carry_allocation = True
    if carry_allocation:
        # Names the approximation rather than the caller that asked for it:
        # the same assumption is the same assumption whether a CLI flag or an
        # oracle option turned it on, and a report that names one of the two
        # cannot be compared with a report that names the other.
        allocation = CarriedAllocation(
            "carry_allocation: the template's block assignment is reused for "
            "every cohort bound to it")

    if _shared_allocation is not None:
        # Another rank of this group built it. Deliberately the same object
        # and not an equal one: the runner offers a step's assignment by
        # calling `offer()` on the provider, and `allocation_for` indexes what
        # was offered by the shape's own coordinates -- so one provider serves
        # the whole group, and a per-rank copy would serve only the rank whose
        # copy the runner happened to hold.
        allocation = _shared_allocation

    body_deriver = head_deriver = None
    build_seconds = 0.0
    if _shared_derivers is not None:
        # Another rank of this group already paid for the trace. `ModelTracer`
        # builds one model per process and refuses a second, and a rank's
        # graphs are produced from the shape's own coordinates rather than
        # from a per-rank build, so sharing is what the derivation already
        # assumed. `build_seconds` stays zero here: the cost was real once and
        # a record that counted it per rank would report four builds of a
        # model that was built once.
        body_deriver, head_deriver = _shared_derivers
    elif derive:
        import time

        from atom.compass.runtime.tracer import ModelTracer, ShapeDeriver

        if not model:
            raise ValueError("derive is on, so a model path is needed to "
                             "trace the shapes no template covers")
        if not block_size or not max_model_len:
            raise ValueError("derive is on, so block_size and max_model_len "
                             "are needed: they decide the block table a "
                             "derived graph is traced against")
        from atom.compass.runtime import derivation_log

        # Recorded on the same wall clock as the on-demand derivations in
        # `TemplateGraphs.graph_for`, so one journal holds every derivation a
        # run paid for and the cost record can place each by interval. This
        # one lands inside the server's startup; a first-seen structure lands
        # inside the served window. Neither phase is asserted here.
        began = derivation_log.now()
        started = time.perf_counter()
        tracer = ModelTracer.build(model, int(tp), device,
                                   replay_target=replay_target)
        build_seconds = time.perf_counter() - started
        derivation_log.record(began, derivation_log.now(),
                              what="model_tracer_build", on_demand=False)
        common = dict(block_size=int(block_size),
                      max_model_len=int(max_model_len),
                      position_rows=int(position_rows),
                      block_policy=block_policy,
                      cudagraph_mode=cudagraph_mode)
        body_deriver = ShapeDeriver(tracer, region="body", **common)
        if head:
            head_deriver = ShapeDeriver(tracer, region="head", **common)

    seeded_inputs: list = []
    # The same declared mode the derivers were built with just above, so a
    # warm bind and the derivation behind it resolve one step's launch extent
    # identically -- see `TemplateGraphs`.
    body_graphs = seeded_graphs(templates, body_deriver, allocation, coords,
                                role="oracle.template",
                                collect=seeded_inputs,
                                cudagraph_mode=cudagraph_mode)
    head_graphs = (seeded_graphs(head_templates, head_deriver, allocation,
                                 coords, role="oracle.head_template",
                                 collect=seeded_inputs,
                                 cudagraph_mode=cudagraph_mode)
                   if head else None)
    _report_rank_binding(coords, body_graphs, head_graphs, derive)
    # What a launch costs in THIS composition, told to the library that has to
    # decide whether a missing launch composition matters. The oracle charges
    # `launches * seconds_per_launch`, so where that rate is zero a price whose
    # kernel composition nobody recorded costs exactly what a price with one
    # would; where it is nonzero the count is a cost and the absence is a
    # refusal. Published rather than inferred: the library reads the configured
    # rate and never chooses it.
    library.launch_charge_seconds = float(seconds_per_launch)
    oracle = LibraryCostOracle(
        library, body_graphs,
        seconds_per_launch=float(seconds_per_launch),
        head_graphs=head_graphs,
        regions=regions_model,
        require_complete=require_complete,
    )
    # How a served run reaches the allocation source without knowing how
    # the composition was assembled: `source_cost_oracle` hands back the
    # oracle alone, and the runner has to be able to offer this step's
    # assignment to it. Attached only when it is the native source --
    # there is nothing to offer a carried one, and a runner that finds
    # the attribute absent knows the oracle is not taking allocations.
    if allocation is not None and getattr(allocation, "measured", False):
        oracle.native_allocation = allocation
    loaded_inputs = _loaded_inputs(library, seeded_inputs, derive)
    # Attached to the oracle, and not only returned in the composition, for
    # the same reason `native_allocation` is: `source_cost_oracle` is what a
    # served run names, and it hands back the oracle alone. Everything else
    # here would be discarded before `CompassPredictMixin` ever saw it -- so a
    # record that lived only in the composition would describe the diagnostic
    # path and never the served one, which is the path that has to be
    # attributable.
    #
    # A tuple, so what the worker later exposes cannot be edited by anything
    # that gets a reference to the oracle.
    oracle.compass_loaded_inputs = loaded_inputs
    # Rides beside them and stays separate from them. The coefficients this
    # run will price every step's preparation and postprocess from, as values,
    # taken where they were selected -- not a file, and not filed as one.
    oracle.compass_region_snapshot = regions_taken
    return SourceComposition(oracle, body_graphs, head_graphs, body_deriver,
                             build_seconds, allocation, coords, rank_artifacts,
                             getattr(library, "max_gap_ratio", None),
                             loaded_inputs)


def _loaded_inputs(library, seeded, derive) -> tuple:
    """Every artifact this composition loaded, as its reader described it.

    Collected from the readers rather than re-derived from the options, which
    is the whole point: the option is a DSL over stems and the files it names
    can change after the load, so only the reader can say what was read.

    Three sources, because there are three readers:

    * the price library, which reports its own reads once it takes identities.
      Until then it reports none, and a price is absent from this record
      rather than misdescribed in it -- an absent input reads as "unrecorded"
      downstream, where a guessed one would read as evidence.
    * the seeded templates, collected as `seeded_graphs` parses them.
    * the replay target the derivation read to answer its architecture query,
      taken from `bootstrap.state()` rather than reopened. Filtered to the
      oracle role: the same process may also have read the deployment's own
      target to bootstrap itself, and that is a different input belonging to
      the runtime side of the record.
    """
    from atom.compass.core.loaded_input import LoadedInput

    found = list(getattr(library, "loaded_inputs", ()) or ())
    found.extend(seeded)
    if derive:
        from atom.compass.replay import bootstrap

        for row in bootstrap.state().get("inputs") or ():
            if row.get("role") == "oracle.replay_target":
                found.append(LoadedInput.from_dict(row))
    return tuple(found)


def _rank_artifacts(coords, prices, templates, head_templates) -> dict:
    """Which of the artifacts asked for were this rank's own file.

    Recorded per requested path, not per resolved one, so a report can say
    "rank 3 asked for prices.json and got prices.json" -- which is the shared
    list, and a legitimate thing to do in a symmetric group, but not the same
    claim as having measured rank 3.
    """
    if not coords:
        return {}
    from atom.compass.core.artifacts import resolve_rank_path

    found = {}
    for entry in list(prices or ()) + list(templates or ()) + list(
            head_templates or ()):
        # At most the price list and its graph. A third field is the
        # collective registration regime, not a path, and resolving it as one
        # put `unregistered.tp0` in the record as an artifact this rank owned.
        for path in str(entry).split(":")[:2]:
            if not path:
                continue
            resolved, own = resolve_rank_path(path, coords)
            found[path] = {"resolved": resolved, "rank_own": bool(own)}
    return found


def _report_rank_binding(coords, body_graphs, head_graphs, derive) -> None:
    """Say once, at build time, which rank's graphs this rank will be served.

    `template_key` carries the rank coordinates, and every template frozen so
    far was derived at rank 0 -- derivation simulates the group's width from
    one gloo rank and leaves ``rank_in_group`` there. So a rank-1 shape matches
    none of them by its own key and is served by `TemplateGraphs`'
    representative fallback instead. That is the predeclared aggregation and
    not a defect, but it should be stated once here rather than inferred later
    from a hit count.
    """
    rank = int((coords or {}).get("tp", 0))
    if not rank:
        return
    import logging

    logger = logging.getLogger(__name__)
    seeded = []
    for source in (body_graphs, head_graphs):
        for key in list(getattr(source, "_templates", {}) or {}):
            # `template_key` puts the coordinate pairs fourth.
            seeded.extend(index for name, index in (key[3] or ())
                          if name == "tp")
    elsewhere = sorted({index for index in seeded if index != rank})
    if elsewhere:
        logger.info(
            "ATOMCompass: serving tp rank %d from templates derived at tp "
            "rank(s) %s. Uniform sharding makes those graphs this rank's "
            "graphs in every field that prices; they are counted as "
            "representative hits, not as this rank's own.%s",
            rank, elsewhere,
            "" if derive else " derive is off, so nothing else is available.")
