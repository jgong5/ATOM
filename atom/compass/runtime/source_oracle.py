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

__all__ = ["source_cost_oracle", "build_source_oracle", "SourceComposition",
           "price_specs", "template_shape", "seeded_graphs", "gap_ratio"]


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


def _price_library(entries, gap_ratio):
    """The exact-signature library, or the family provider in front of it.

    The provider is a subclass that overrides `lookup` alone, so everything
    downstream -- `PriceLibrary.body`, the `Coverage` split, `LibraryCostOracle`
    -- is the same code either way, and a derived price arrives carrying its own
    ``interpolated`` marker rather than being counted as a measurement.

    Off by default. An interpolated price is a claim about a row count nobody
    ran, and a run that did not ask for one should not silently get one.
    """
    from atom.compass.core.cost.library import PriceLibrary

    if gap_ratio is None:
        return PriceLibrary.load(entries)

    from atom.compass.core.cost.families import ParametricPriceLibrary

    library = (ParametricPriceLibrary()
               if gap_ratio is _DEFAULT_GAP_RATIO
               else ParametricPriceLibrary(max_gap_ratio=gap_ratio))
    for entry in entries:
        if isinstance(entry, (tuple, list)):
            library.add(entry[0], entry[1] if len(entry) > 1 else None,
                        entry[2] if len(entry) > 2 else None)
        else:
            library.add(entry, None, None)
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


def seeded_graphs(paths, derive, allocation, coords=None):
    """A `TemplateGraphs` holding the graphs already on disk, keyed by spec.

    ``coords`` resolves each path to this rank's file where one was written,
    by the same rule as the prices. It does *not* rewrite the key: a template
    is keyed by the coordinates in its own provenance, because that is the rank
    the graph is a graph of. Re-keying one to the rank that happens to be
    reading it would erase the difference between a graph derived for this rank
    and one borrowed from the representative. `TemplateGraphs` serves the
    borrow deliberately, on a miss, and counts it as a representative hit.
    """
    import json

    from atom.compass.core.artifacts import resolve_rank_path
    from atom.compass.runtime.templates import TemplateGraphs, template_key

    graphs = {}
    for path in paths:
        if coords:
            path = resolve_rank_path(path, coords)[0]
        with open(path, encoding="utf-8") as fh:
            graph = json.load(fh)
        graphs[template_key(template_shape(graph))] = graph
    return TemplateGraphs(graphs, derive=derive, allocation=allocation)


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
    interpolation_limit: Optional[float] = None


def source_cost_oracle(*, rank_coords=None, **kwargs):
    """The frozen composition as a plain oracle, for `oracle_qualname`.

    This is the entry point a served run names. It takes exactly the arguments
    :func:`build_source_oracle` documents and returns only the oracle, because
    that is what `_build_oracle` expects to get back.

    ``rank_coords`` is named explicitly rather than swept into ``**kwargs``,
    and that is the whole reason it is in this signature: `_build_oracle`
    offers the rank only to an oracle that names it, by
    ``inspect.signature(...).parameters``, and a bare ``**kwargs`` names
    nothing. Under TP>1 the injection silently did not fire, so every rank of
    a served run built the same composition and resolved the same artifacts --
    rank 0's, wherever a per-rank file existed.
    """
    return build_source_oracle(rank_coords=rank_coords, **kwargs).oracle


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
    derive: bool = True,
    interpolate=None,
    rank_coords=None,
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
    """
    from atom.compass.core.cost.library import LibraryCostOracle
    from atom.compass.core.cost.regions import region_model
    from atom.compass.runtime.templates import CarriedAllocation

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

    price_entries = price_specs(_entries(price, "price"), coords)
    library = _price_library(price_entries, gap_ratio(interpolate))
    regions_model = region_model(regions)
    rank_artifacts = _rank_artifacts(
        coords, _entries(price, "price"), templates, head_templates)

    allocation = None
    if carry_allocation:
        # Names the approximation rather than the caller that asked for it:
        # the same assumption is the same assumption whether a CLI flag or an
        # oracle option turned it on, and a report that names one of the two
        # cannot be compared with a report that names the other.
        allocation = CarriedAllocation(
            "carry_allocation: the template's block assignment is reused for "
            "every cohort bound to it")

    body_deriver = head_deriver = None
    build_seconds = 0.0
    if derive:
        import time

        from atom.compass.runtime.tracer import ModelTracer, ShapeDeriver

        if not model:
            raise ValueError("derive is on, so a model path is needed to "
                             "trace the shapes no template covers")
        if not block_size or not max_model_len:
            raise ValueError("derive is on, so block_size and max_model_len "
                             "are needed: they decide the block table a "
                             "derived graph is traced against")
        started = time.perf_counter()
        tracer = ModelTracer.build(model, int(tp), device,
                                   replay_target=replay_target)
        build_seconds = time.perf_counter() - started
        common = dict(block_size=int(block_size),
                      max_model_len=int(max_model_len),
                      position_rows=int(position_rows),
                      block_policy=block_policy,
                      cudagraph_mode=cudagraph_mode)
        body_deriver = ShapeDeriver(tracer, region="body", **common)
        if head:
            head_deriver = ShapeDeriver(tracer, region="head", **common)

    body_graphs = seeded_graphs(templates, body_deriver, allocation, coords)
    head_graphs = (seeded_graphs(head_templates, head_deriver, allocation,
                                 coords)
                   if head else None)
    _report_rank_binding(coords, body_graphs, head_graphs, derive)
    oracle = LibraryCostOracle(
        library, body_graphs,
        seconds_per_launch=float(seconds_per_launch),
        head_graphs=head_graphs,
        regions=regions_model,
        require_complete=require_complete,
    )
    return SourceComposition(oracle, body_graphs, head_graphs, body_deriver,
                             build_seconds, allocation, coords, rank_artifacts,
                             getattr(library, "max_gap_ratio", None))


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
        for path in str(entry).split(":"):
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
