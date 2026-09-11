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
           "price_specs", "template_shape", "seeded_graphs"]


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


def price_specs(entries):
    """``prices.json[:graph.json[:regime]]`` triples, as `PriceLibrary` takes.

    The graph is optional and worth supplying: without it a price matches by
    signature alone, and a signature does not carry layout. The third element
    names the collective registration regime for a list whose provenance
    predates the field.
    """
    loaded = []
    for entry in entries or ():
        parts = entry.split(":")
        if len(parts) == 1:
            loaded.append((parts[0], None))
        elif len(parts) == 2:
            loaded.append((parts[0], parts[1] or None))
        elif len(parts) == 3:
            loaded.append((parts[0], parts[1] or None, parts[2]))
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


def seeded_graphs(paths, derive, allocation):
    """A `TemplateGraphs` holding the graphs already on disk, keyed by spec."""
    import json

    from atom.compass.runtime.templates import TemplateGraphs, template_key

    graphs = {}
    for path in paths:
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


def source_cost_oracle(**kwargs):
    """The frozen composition as a plain oracle, for `oracle_qualname`.

    This is the entry point a served run names. It takes exactly the arguments
    :func:`build_source_oracle` documents and returns only the oracle, because
    that is what `_build_oracle` expects to get back.
    """
    return build_source_oracle(**kwargs).oracle


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
):
    """The frozen composition, from names and paths alone.

    ``model``/``tp``/``device``/``replay_target`` and the five derivation
    settings are what building a deriver costs; they are required only when
    ``derive`` is on, so a run that seeds every template it needs can leave the
    model out and never load one.
    """
    from atom.compass.core.cost.library import LibraryCostOracle, PriceLibrary
    from atom.compass.core.cost.regions import region_model
    from atom.compass.runtime.templates import CarriedAllocation

    head = _flag(head, "head")
    require_complete = _flag(require_complete, "require_complete")
    carry_allocation = _flag(carry_allocation, "carry_allocation")
    derive = _flag(derive, "derive")

    templates = _entries(template, "template")
    head_templates = _entries(head_template, "head_template")
    if not derive and not templates:
        raise ValueError(
            "derive is off and no template was given, so every shape would be "
            "refused for want of a graph. Seed a template or turn derivation "
            "on.")

    library = PriceLibrary.load(price_specs(_entries(price, "price")))
    regions_model = region_model(regions)

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

    body_graphs = seeded_graphs(templates, body_deriver, allocation)
    head_graphs = (seeded_graphs(head_templates, head_deriver, allocation)
                   if head else None)
    oracle = LibraryCostOracle(
        library, body_graphs,
        seconds_per_launch=float(seconds_per_launch),
        head_graphs=head_graphs,
        regions=regions_model,
        require_complete=require_complete,
    )
    return SourceComposition(oracle, body_graphs, head_graphs, body_deriver,
                             build_seconds, allocation)
