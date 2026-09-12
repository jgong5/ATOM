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

That is deliberately not sufficient on its own. The markers only matter if the
*counts* keep them apart, which is ``Coverage`` in ``library.py`` -- not this
module's file. That split has been agreed and specified in
``agent_scratch/COVERAGE_SEAM.md``: ``Coverage`` classifies each record as
zero-work, then interpolated, then measured, in that order, and reports
``complete`` (nothing refused) separately from ``complete_measured`` (nothing
fitted either).

Both questions are real and neither subsumes the other. A validated in-support
interpolation is allowed to make a step ``complete``, because predictive
coverage is the claim; it is not allowed to make it ``complete_measured``,
because that would overstate the evidence. :func:`coverage_split` computes the
same four-way split directly from a graph, so a caller can assert on it
regardless of which half of the seam it is running against.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from atom.compass.core.cost.families import attention, attention_scope
from atom.compass.core.cost.families.features import (
    contract_for,
    grouping_key,
    executed_rows,
    infer_rows,
)
from atom.compass.core.cost.families.support import (
    MeasuredCurve,
    Refusal,
    RowSupport,
)
from atom.compass.core.cost.library import (
    INTERPOLATED_FLAG,
    INTERPOLATED_SOURCE_PREFIX,
    ZERO_WORK_FLAG,
    PriceLibrary,
)

logger = logging.getLogger(__name__)

__all__ = ["ParametricPriceLibrary", "coverage_split", "INTERPOLATED_SCHEME"]

#: The record markers and source prefix the consumer classifies on. They belong
#: to ``library.py``, which is the file that reads them, and are imported from
#: there rather than restated: a literal kept in two files that must agree is
#: how they stop agreeing.
INTERPOLATED_SCHEME = INTERPOLATED_SOURCE_PREFIX

#: The one refusal a family module is allowed to answer.
_OPEN_QUESTION = "no entry for this signature"


#: Where a RESOLVED attention scope may be written in a price artifact. Only
#: sections whose values are facts about the process that ran: `config` is not
#: among them, because a config states what was *asked for*. `kv_cache_dtype:
#: "auto"` is a request the engine then resolves to something concrete, and a
#: backend named in a config is a preference the dispatcher may not have taken.
#: Reading either as a runtime fact is how a law gets attributed to a kernel
#: that never ran.
_ATTENTION_SCOPE_SECTIONS = ("attention_scope", "resolved_scope")

#: Every key some attention regime is identified by. Read as a union: a
#: unified observation simply will not carry `gdn_state_geometry`, and a GDN
#: one will not carry `kv_cache_dtype`, and neither absence is filled in here.
#: Taken from the regimes themselves rather than from the two family scopes,
#: because a regime may be identified by more than its family is -- the paged
#: decode law turns on `num_kv_heads` and `compute_units` as well, and a key
#: missing from this tuple is a fact the adapter drops on the floor, leaving
#: the law that needs it permanently refused for want of it.
_ATTENTION_SCOPE_KEYS = tuple(sorted(
    set(attention.UNIFIED_SCOPE) | set(attention.GDN_SCOPE)
    | {key for regime in attention.REGIMES.values()
       for key in regime.required_scope}))

#: Conditions of the measurement itself, as the collector writes them. Not
#: kernel-selecting, and carried anyway: `attention._scope_of` requires
#: agreement on every declared key, so two files that differ in how many
#: repeats they averaged, whether the first call was included, whether the
#: capture rotated its cache residency, or which of them a later collection
#: superseded, refuse to pool rather than averaging measurements taken under
#: different conditions. Dropping them here is what would make that refusal
#: unreachable.
_MEASUREMENT_CONDITIONS = (
    "cache_state", "compile_mode", "cudagraph_mode", "dtype", "enforce_eager",
    "first_call", "max_model_len", "model", "quantization",
    "repeats", "rotation", "superseded", "tensor_parallel_size", "timing",
    "timing_method", "visited", "warmup",
)

#: Acquisition metadata: true of how a batch was collected, and not a property
#: an inference request could have or fail to have. `iters` is how many times
#: the harness replayed a capture to get a stable number, and `only` is the
#: family filter the operator was selected by. A served step does not choose an
#: iteration count, so requiring a request to state one before a law applies
#: would make every law unmatchable for a reason nobody could act on. They are
#: kept as qualification -- a reader still sees how many iterations stood
#: behind a number -- and taken out of scope EQUALITY, which is about which
#: deployment a price is a price for.
_ACQUISITION_CONTEXT = ("iters", "only")

#: Config fields worth carrying as conditions, under a name that cannot be
#: mistaken for a resolved fact. Two files that asked for different things are
#: not obviously one deployment, and `requested.` says plainly that this is the
#: request and not what the engine did with it.
_REQUESTED_CONDITIONS = ("attention_backend", "block_size", "kv_cache_dtype",
                         "kv_cache_layout", "sliding_window")


def _hashable(value):
    """A declared value in a form two scopes can be compared by."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _hashable(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return value


def _attention_scope(blob: dict, registration: Optional[str]) -> dict:
    """What a price file DECLARES about the kernel and the conditions.

    Nothing is inferred. A pool allocation proves storage, not the view a
    backend takes over it, so no key here is derived from operand shapes:
    every one comes from something the collector wrote down, in a section that
    records what ran rather than what was requested.

    Everything declared is carried, not only the kernel-selecting keys. The
    scope is the identity a fit is filed under, and a condition left out here
    is a difference two observations are allowed to disagree on silently.
    """
    provenance = blob.get("provenance") or {}
    scope: dict = {}
    for field in _ATTENTION_SCOPE_KEYS:
        for section in _ATTENTION_SCOPE_SECTIONS:
            where = provenance.get(section) or {}
            if isinstance(where, dict) and field in where:
                scope[field] = _hashable(where[field])
                break
        else:
            if field in provenance:
                scope[field] = _hashable(provenance[field])
    for field in _MEASUREMENT_CONDITIONS:
        if field in provenance:
            scope[field] = _hashable(provenance[field])
    config = provenance.get("config") or {}
    if isinstance(config, dict):
        for field in _REQUESTED_CONDITIONS:
            if field in config:
                scope["requested." + field] = _hashable(config[field])
    declared_registration = provenance.get("registration", registration)
    if declared_registration is not None:
        scope["registration"] = declared_registration
    topology = provenance.get("topology")
    if topology is not None:
        scope["topology"] = _hashable(topology)
    return scope


def _resolved_declaration(blob: dict, price_path: str):
    """What the measuring process itself resolved, read as a per-family scope.

    New price records carry `resolved_scope`: the environment, the pools and
    the per-layer backends the SAME process stood up, written down before it
    priced anything. That is the strongest evidence a scope can have -- it is
    not a later reading of a config, it is what ran -- so it is taken here in
    preference to anything the request declares about the same file.

    A record whose resolution cannot be read is not quietly treated as a
    record without one. It keeps a scope key naming the failure, so it can
    never pool with a record whose resolution WAS read, and a reader is told
    which file to go and look at.
    """
    payload = (blob or {}).get("resolved_scope")
    if not isinstance(payload, dict):
        return None, None
    try:
        return attention_scope.read_resolved(
            payload, where="%s:resolved_scope" % price_path), None
    except ValueError as exc:
        return None, str(exc)


#: Key components that identify WHICH layer a call was, rather than what it
#: cost. A layer name and the blocks and slots that layer's cache occupies
#: differ across the copies of one step and are the only things that do; every
#: other component stays in the identity, so anything else that differs makes
#: two observations separate design points rather than replicates.
_LAYER_IDENTITY = ("layer_name", "layer", "layer_idx", "block_tables",
                   "slot_mapping", "kv_cache", "kv_cache_ptr")


#: The TREATMENT a price record was taken under: the conditions imposed on the
#: measurement, as opposed to what came out of it. Not provenance --
#: `provenance.cache` states the cache mode that was requested, while
#: `record["cache"]` is what that record was actually taken under, and only the
#: second is a fact about the measurement. `kv_regions` says which regions were
#: rebuilt, and the argument rotation says how cold the operands were kept: two
#: records that differ in either are measurements of different residency, not
#: repeats of one. The version qualifiers are here because a record taken by a
#: different collector is not a repeat of one taken by this one.
#:
#: The operand rotation enters as the POLICY that produced it, not as the count
#: it produced. `arg_sets` is `max(2, min(64, COLD_WORKING_SET_BYTES //
#: per_set))` -- see `microbench._build_arg_sets` -- so it is a shape-derived
#: output of one residency policy, a function of the design point rather than a
#: knob anybody set. Keying on it makes the treatment vary with the very thing
#: being fitted: every size gets its own singleton law and nothing is left to
#: fit. What distinguishes two genuinely different measurement policies is the
#: code that decided the residency, and the collector records exactly that --
#: see `_acquisition_policy`. So records taken under one policy pool, records
#: taken under a different one do not, and the realized count is kept as
#: qualification on the point instead of as its scope.
#:
#: The treatment rides into the FIT scope, not only into the replicate key. A
#: cold-cache point and a warm-cache point are not two points on one law, and
#: separating them only at collapse time would let them rejoin as independent
#: points of the same fit -- which is the same averaging, moved one step later.
_TREATMENT_FIELDS = ("cache", "kv_regions", "version",
                     "collector_version", "schema_version")

#: Shape-derived outputs of the acquisition policy. Kept with the point as
#: evidence -- a reader still sees how many operand sets a record actually
#: rotated over, and so how large its footprint was -- and kept OUT of the
#: identity, because a value the design point determines cannot also be a
#: condition the design point is compared under.
_QUALIFICATION_FIELDS = ("arg_sets",)

#: The module whose code decides the residency policy: the byte budget, the
#: clamp, and the rotation `arg_sets` is the output of. Its content hash is
#: what separates two measurement policies from one policy applied to two
#: shapes.
_POLICY_MODULE = "atom.compass.runtime.microbench"


def _acquisition_policy(blob: dict) -> tuple:
    """The measurement policy a price file was taken under, from its own record.

    The collector writes the identity of every module it imported, by path and
    content hash, before it measures anything. The hash of the module that
    builds the operand rotation IS the policy: the same hash is the same byte
    budget and the same clamp, and a different hash is a different policy whose
    numbers were not taken the same way.

    A file that does not record it gets ``("unevidenced",)``, which equals no
    recorded policy and pools with nothing that has one. That is a refusal to
    assume, not a default: an unrecorded policy might be this one or might not,
    and the difference is the whole question.
    """
    modules = (((blob.get("provenance") or {}).get("collector") or {})
               .get("source_identity") or {}).get("modules") or {}
    entry = modules.get(_POLICY_MODULE) if isinstance(modules, dict) else None
    if isinstance(entry, dict):
        digest = entry.get("sha256")
    else:
        digest = entry if isinstance(entry, str) else None
    if not digest:
        return ("unevidenced",)
    return (_POLICY_MODULE, str(digest))

#: Measured OUTCOMES. Deliberately not part of any identity: `host_seconds` is
#: a number that came out of the measurement and carries ordinary timing
#: noise, so keying on it would make every repeat and every layer copy its own
#: design point -- the exact inflation the replicate collapse exists to
#: prevent. It is kept as a distribution, with its missingness, for a reader
#: who needs to know whether a record was measuring the launch or the kernel.
_OUTCOME_FIELDS = ("host_seconds",)

#: Kernel symbols whose integer template arguments are a FUNCTION of the launch
#: geometry a law already models, not a condition the measurement was taken
#: under. The paged decode reduce kernel is instantiated per split count, and
#: the split count is decided by the launch itself --
#: `min(DECODE_MAX_SPLITS, ceil(compute_units * DECODE_OCCUPANCY /
#: (rows * num_kv_heads)))` -- so a batch width and a specialization are the
#: same fact stated twice. Keeping the specialization in the identity is not
#: conservative here, it is fatal: every row width lands in its own group, and
#: within one row width `crit_waves` is exactly proportional to
#: `max_cta_tiles`, so the makespan law is unidentifiable by construction and
#: no amount of further measurement can separate its two bounds. The other two
#: integer arguments are head dimensions and are NOT canonicalised; neither are
#: the type arguments, because a bfloat16 instantiation and an fp8 one are
#: different work.
_GEOMETRY_SPECIALIZED_KERNELS = ("pa_decode_ps_reduce_hip_kernel",)

#: The *last* integer template argument, and only that one.
#:
#: The resolved declaration (`csrc/cpp_itfs/pa/pa_ps.cuh`:183-190) is
#: `<output_t, logits_t, sink_t, USE_SINKS, HEAD_SIZE, QUERY_GROUP_SIZE,
#: CONTEXT_PARTITION_NUM>`: three integers, split count last. The campaign
#: observed `<__hip_bfloat16, __hip_bfloat16, __hip_bfloat16, false, 256, 6,
#: N>` for N in {2, 3, 5, 8} and no other instantiation.
#:
#: Only `CONTEXT_PARTITION_NUM` is pooled, because only it is a modelled
#: launch feature -- it is the split count `Structure` already reads, and
#: leaving it in the measurement identity is what makes the law
#: unidentifiable. `HEAD_SIZE` (256) and `QUERY_GROUP_SIZE` (6) stay in the
#: symbol: they are head dimensions, not launch features, and an earlier
#: blanket `\d+` substitution erased them as well -- which would have let a
#: differently-shaped head or query grouping pool silently into this law.
#: Both were constant across every observed symbol, so narrowing changes no
#: existing design point. It removes a way for a future one to be wrong.
_TEMPLATE_SPLIT_COUNT = re.compile(r"(?<=,) ?\d+ ?(?=>)")


def _canonical_kernel(name: str) -> str:
    """A split-specialized kernel symbol, with the split count taken out.

    Returns the name unchanged for every kernel not named in
    `_GEOMETRY_SPECIALIZED_KERNELS`, which is all but one of them. Where it
    does substitute, the substitution is recorded and reported, because
    pooling two symbols that the compiler kept apart is a modelling claim and
    a reader is entitled to see it made.
    """
    if not any(symbol in name for symbol in _GEOMETRY_SPECIALIZED_KERNELS):
        return name
    return _TEMPLATE_SPLIT_COUNT.sub(" *", name)


def _measurement_identity(record: dict, policy: tuple = ("unevidenced",)) -> tuple:
    """The treatment this record was taken under, as a comparable key.

    Part of both the design identity and the fitted law's scope, so two
    records taken under different cache modes, different region rebuilds or
    different collector versions are never collapsed into one design point's
    median and never pooled into one law -- either of which would average a
    cold measurement with a warm one and report the result as a repeat.
    """
    kernels = tuple(sorted(_canonical_kernel(name)
                           for name in (record.get("kernels") or {})))
    return (kernels, ("acquisition_policy", policy)) + tuple(
        (field, _hashable(record[field]))
        for field in _TREATMENT_FIELDS if field in record)


def _qualification(record: dict) -> tuple:
    """The shape-derived facts kept with a record but out of its identity."""
    return tuple((field, _hashable(record[field]))
                 for field in _QUALIFICATION_FIELDS if field in record)


def _acquisition_context(blob: dict) -> tuple:
    """How a batch was collected, as qualification rather than as scope.

    `iters` is how many times the harness replayed a capture, `only` the
    family filter the operator was selected by -- see `_ACQUISITION_CONTEXT`.
    Both are true of the acquisition and neither is a property a served
    request could have, so they are recorded and reported and never compared.
    """
    provenance = blob.get("provenance") or {}
    config = provenance.get("config") or {}
    found = []
    for field in _ACQUISITION_CONTEXT:
        for where in (provenance, config if isinstance(config, dict) else {}):
            if field in where:
                found.append((field, _hashable(where[field])))
                break
    return tuple(found)


def _host_seconds(record: dict):
    """The record's host time, or None where it does not say."""
    value = (record or {}).get("host_seconds")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def kernels_of(measurement: tuple) -> tuple:
    """The kernel names a measurement identity carries."""
    return measurement[0] if measurement else ()


#: The layer index inside a bound module path, e.g. the `3` in
#: `language_model.model.layers.3.self_attn`. A real graph records that path as
#: a SCALAR under a positional key (`#5`), so a layer cannot be recognised by
#: its key the way `layer_name` can -- and without this every one of the 64
#: bound modules in a captured step reads as its own design point, which would
#: report 64 independent measurements where one step was measured 64 times.
_LAYER_INDEX = re.compile(r"(?<=\.layers\.)\d+(?=\.)")


def _delayered(value):
    """``value`` with any bound-module layer index blanked.

    The module KIND survives: `self_attn` and `linear_attn` stay distinct, so
    an MHA layer and a GDN layer are never collapsed into one another. Only
    the index -- the thing that differs between replicates of one step and
    nothing else -- is removed.
    """
    if isinstance(value, str):
        return _LAYER_INDEX.sub("*", value) if ".layers." in value else value
    if isinstance(value, (list, tuple)):
        return [_delayered(item) for item in value]
    return value


def _design_identity(op: dict) -> tuple:
    """Everything about this call except which layer it was.

    Operand shapes, operand dtypes and the recorded operand views, because a
    strided call and a dense one are different work on the same numbers. Every
    context and scalar the key carries, because that is where the native
    branches live -- `is_prefill`, `has_cached`, `state`, `replayssm`, the
    speculative offsets, `num_actual_tokens`. The launch grid, because a
    Triton launch over a different grid is a different amount of work.
    """
    context = tuple((k, repr(_delayered(v))) for k, v in
                    (tuple(x) for x in op.get("context") or ())
                    if k not in _LAYER_IDENTITY)
    scalars = tuple((k, repr(_delayered(v))) for k, v in
                    (tuple(x) for x in op.get("scalars") or ())
                    if k not in _LAYER_IDENTITY)
    grid = tuple((k, repr(v)) for k, v in
                 (tuple(x) for x in op.get("launch") or ()) if k == "grid")
    values = _hashable(op.get("int_values") or ())
    # `_hashable`, not a shallow conversion: a recorded operand view is a
    # nested list of strides, and a one-level tuple() leaves inner lists
    # unhashable -- which on a real strided operator is a crash, not a
    # subtlety.
    return (op.get("name"),
            _hashable(op.get("input_shapes") or ()),
            tuple(op.get("dtypes") or ()),
            _hashable(op.get("layouts") or ()),
            context, scalars, grid, values)


def _scope_key(scope) -> tuple:
    """A scope as a hashable key. Same two fields ``_same_scope`` compares."""
    scope = scope or {}
    topology = scope.get("topology") or {}
    return (tuple(sorted(topology.items())) if isinstance(topology, dict)
            else tuple(topology), scope.get("registration"))


def _scope_note(key: tuple) -> str:
    topology, registration = key
    return (f"{dict(topology) or 'undeclared width'} on "
            f"{registration or 'an undeclared path'}")


def _topology_key(topology):
    """A declared topology as the hashable value a scope carries."""
    return (tuple(sorted(topology.items())) if isinstance(topology, dict)
            else tuple(topology))


def _shown_to_match(obs_scope, requested) -> bool:
    """Whether an observation was SHOWN to be under the requested scope.

    Only the keys the request declares are judged, and the observation has to
    declare every one of them with the same value. Silence on a requested key
    is not a match: an entry that never said which backend, KV dtype or state
    it was measured under was not shown to be this deployment, so it does not
    get a say in which treatment this deployment is priced under.

    The observation's own extra keys are deliberately not judged here. At this
    point the request is still the raw declaration -- registration and
    topology are added after -- so demanding agreement in that direction would
    reject every well-scoped observation for keys the request has not been
    given yet. `attention.Model._fit_for` compares both directions once the
    request is complete, and that remains the check that decides the price.

    Both sides are normalised through `_hashable` before they are compared.
    An observation's scope was made hashable when it was read and a caller's
    declaration need not have been, so a state geometry declared as a list
    would otherwise differ from the identical tuple that was measured -- a
    mismatch invented by the reader, not by the deployments.
    """
    obs_scope = obs_scope or {}
    for field, value in (requested or {}).items():
        if field not in obs_scope:
            return False
        if _hashable(obs_scope[field]) != _hashable(value):
            return False
    return True


#: The scope key for an operator whose cost does not depend on the group. Not
#: `None`, so it cannot be confused with "scope not recorded".
_LOCAL = ("local",)


def _scope_of(op: dict, scope) -> tuple:
    """The scope a measurement of this operator is valid in.

    Only a collective's cost depends on the group it runs in, and
    ``PriceLibrary.lookup`` scopes exact prices for collectives alone. A GEMM
    is a GEMM: the same shapes on the same layout cost the same whether the
    deployment around it is TP1 or TP2, and whether the collectives elsewhere
    in that graph take the registered path or not. The body hands the graph's
    registration to *every* lookup, so scoping local families on it would
    refuse an in-support interpolation of a GEMM measured in an unregistered
    price list purely because the graph asking has registered collectives --
    while the exact-width lookup of that same GEMM succeeds. Two paths
    disagreeing about the same operator is the bug, not the scoping.

    So local families share one scope and collectives keep theirs.
    """
    from atom.compass.runtime.microbench import _is_collective_op

    return _scope_key(scope) if _is_collective_op(op) else _LOCAL


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
        #: price file -> the row count that run was measured at, or None when
        #: the file states no width of its own and its operators must
        self._rows: dict[str, Optional[int]] = {}
        #: price file -> {signature -> the operator THAT file's graph recorded}
        #: Needed beside `_ops` because a key does not carry layout: two
        #: files can price the same key on differently arranged memory, and
        #: only the per-file map says which price is which.
        self._source_ops: dict[str, dict] = {}
        #: (grouping key, scope key) -> [(op, rows, seconds, source, kernels)]
        self._observations: dict[tuple, list] = {}
        self._curves_built = False
        #: reasons a family could not be assembled, for describe()
        self.unbuildable: dict[str, str] = {}
        #: price file -> (executed rows, scheduled tokens) where a capture
        #: bucket makes the two differ, so the padding stays visible
        self.padded: dict[str, tuple[int, int]] = {}
        #: price file -> why it states no width of its own, when it does not.
        #: Not the same as `unbuildable`: a head graph has no embedding and no
        #: `body_rows_traced`, so it states no file width at all, and yet every
        #: operator in it whose family declares `rows_from` still says what
        #: width IT ran at. Those files are usable per operator and refused
        #: only for the families that have no such reading.
        #:
        #: A graph whose two width statements CONTRADICT each other is not
        #: here -- it stays in `unbuildable`. Silence about the width and a
        #: contradiction about it are different facts, and only the first one
        #: leaves the operators trustworthy.
        self.no_file_width: dict[str, str] = {}
        #: Every ragged attention price as it was written, before the library
        #: reindexes it: (op, seconds, source, scope). Collected here rather
        #: than read back out of `self._prices` because `_ingest` files records
        #: under the cost key and keeps one per scope -- which is right for an
        #: exact lookup and wrong for a fit, where two captures of the same key
        #: at different ragged structures are two design points and the second
        #: would be dropped as a duplicate.
        self._attention_obs: list = []
        #: canonical kernel symbol -> the specializations pooled under it.
        #: Empty unless `_canonical_kernel` actually substituted something.
        #: Reported by `attention_coverage`, because pooling two symbols the
        #: compiler kept apart is a modelling claim, not bookkeeping.
        self._pooled_kernels: dict = {}
        #: What a launch costs where this library is being priced, published by
        #: whoever builds the cost oracle -- see `source_oracle`. `None` means
        #: nobody said, which is not the same as zero and is never read as it.
        self.launch_charge_seconds = None
        #: Per price file, how it was acquired: the policy its numbers were
        #: taken under and the acquisition metadata that qualifies them
        #: without identifying a deployment. Reported, never compared.
        self._attention_acquisition: dict = {}
        #: the fitted attention model, built once from those observations
        self._attention_model = None
        #: The asking deployment's own resolved attention scope -- its KV
        #: dtype, layout, block size, window, resolved backend, GDN state
        #: geometry. Set by whoever resolves it; left empty here, because
        #: guessing it is how a law measured under one deployment ends up
        #: answering for another. An empty request matches only an unscoped
        #: fit, which under `strict` does not exist.
        self.request_attention_scope: dict = {}
        #: The deployment a price file that is SILENT about its own was
        #: measured under, declared by whoever resolved it -- an
        #: `attention_scope.Declaration`, carrying the facts behind every
        #: member it names. `None` means nobody declared one, which is not the
        #: same as an empty declaration: a silent file then stays silent and
        #: its observations are refused for want of a backend, which is the
        #: honest outcome rather than a loss.
        #:
        #: It may only fill a silence. A key the file states is the measuring
        #: process speaking about its own run, and a caller disagreeing with
        #: it is a contradiction to be fixed, not a preference to be applied.
        #: That is the rule `add` already keeps for `registration` -- "a
        #: caller may name the scope a file leaves unstated, not overrule the
        #: one it states" -- and this is the same rule for the same reason.
        self.declared_attention_scope = None

    # -- assembly -------------------------------------------------------

    def add(self, price_path: str, graph_path: Optional[str] = None,
            registration: Optional[str] = None, *, coords=None) -> None:
        """Load a price file, and the graph that says what its keys mean.

        Without a graph this behaves exactly as the base class: prices are
        keyed by signature and there is no structured operator to read a
        feature off, so the file contributes nothing to any curve. That is a
        silent loss of capability rather than of correctness, so it is
        recorded.

        ``coords`` is forwarded unresolved, exactly as the base class takes
        it, so this override cannot become a second place that resolves a
        rank's path.
        """
        # `_ingest` rather than `super().add`, so the graph is parsed once and
        # this override reads the payload the base class already has. Opening
        # it again here would give the retained digest a second set of bytes
        # to be a digest of.
        _blob, graph = self._ingest(price_path, graph_path, registration,
                                    coords)
        if graph is None:
            self.unbuildable[price_path] = (
                "no graph supplied, so its operators have no structure to "
                "read a feature from; exact-signature use only")
            return
        # Before the row-family eligibility check below, deliberately. A
        # standalone attention primitive graph need contain no embedding and
        # carry no `body_rows_traced` -- it is one operator measured on its
        # own, not a body -- and `_traced_rows` refuses such a file. That
        # refusal is correct for a row curve, which is a curve in the width the
        # body ran at, and irrelevant to an attention fit, which reads its
        # features off the operator's own ragged structure. Collecting after it
        # would throw away exactly the primitive measurements this model needs.
        self._collect_attention(price_path, _blob, graph, registration)
        reading = _traced_rows(graph)
        if isinstance(reading, tuple):
            _, why, conflicting = reading
            if conflicting:
                # The graph states a width twice and the two disagree. That is
                # not a missing reading, it is an untrustworthy one, and the
                # per-operator readings come off the same graph: a body whose
                # provenance and embedding contradict each other is not a
                # source an interpolated price may be built from just because
                # its GEMM happens to declare `rows_from`. Refused as a whole,
                # exactly as it was before any operator could state a width.
                self.unbuildable[price_path] = f"{graph_path}: {why}"
                return
            # No width for the file as a whole, and nothing contradicting it
            # either. That used to end the file's usefulness, which is why
            # every head price file was exact-key only: a head graph carries
            # neither an embedding nor `body_rows_traced`, so it could never
            # state one. It is only fatal for families that have no operand
            # reading of their own; the operators that do are still
            # measurements of a known width, and `_build` reads them one at a
            # time below.
            rows = None
            self.no_file_width[price_path] = f"{graph_path}: {why}"
        else:
            rows = reading
            scheduled = sum((graph.get("key") or {}).get("batch_signature")
                            or ())
            if scheduled and scheduled != rows:
                # Legitimate under a capture bucket, and worth saying out loud:
                # the prices are of the padded width, not of the scheduled one.
                self.padded[price_path] = (rows, scheduled)
        self._rows[price_path] = rows
        from atom.compass.runtime.microbench import cost_key_of, signature_of

        for op in graph.get("ops") or ():
            # `_ops` is an index -- "what structure does this key have".
            # Keyed by the cost key so it still answers for a request whose
            # allocator moved; keying it by the raw signature would miss on
            # every operator carrying an address, silently, leaving the curve
            # empty and the family refusing widths it can price.
            self._ops.setdefault(cost_key_of(op), op)
            # `_source_ops` is an association -- "what did THIS record price".
            # Keyed by the RAW signature, because the cost key is many-to-one
            # and the observations it collapses need not share a layout; the
            # layout is exactly what this per-file map exists to keep straight.
            # A cost-key join would hand a record whichever collapsed sibling
            # the graph happened to list first, which is R5 again by another
            # route. `_build` therefore joins on `record["signature"]`.
            #
            # First occurrence within one raw signature, matching the two
            # readers that already choose: `microbench` keys its example
            # operator with `example.setdefault`, and `PriceLibrary._ingest`
            # captures the measured layout with `layouts.setdefault`. A graph
            # holding a dense and a strided call under one raw signature is
            # PRICED as the dense one, so labelling it strided here would
            # disagree with the measurement.
            self._source_ops.setdefault(price_path, {}).setdefault(
                signature_of(op), op)
        self._curves_built = False

    def _collect_attention(self, price_path: str, blob: dict, graph: dict,
                           registration: Optional[str]) -> None:
        """Keep every ragged attention price joined to its own graph operator.

        The join is on the record's own RAW signature against the graph from
        the same file, which is the association `_source_ops` exists to hold;
        it is populated here too, so a file that `_traced_rows` later refuses
        still keeps the link between what was priced and what it was a price
        of. The record's kernels are kept with it: which kernels served a call
        is evidence about what ran, and two records served by different kernels
        are not replicates of one design point.

        Nothing is deduplicated at this stage and nothing is filed under a cost
        key. Two captures of one signature at different ragged structures are
        two design points, and the library's own index -- keyed by cost key,
        one record per scope -- would keep only the first of them.
        """
        from atom.compass.runtime.microbench import signature_of

        families = (attention.UNIFIED, attention.GDN)
        by_sig: dict = {}
        for op in graph.get("ops") or ():
            if op.get("name") not in families:
                continue
            sig = signature_of(op)
            by_sig.setdefault(sig, op)
            self._source_ops.setdefault(price_path, {}).setdefault(sig, op)
        if not by_sig:
            return
        declared = _attention_scope(blob, registration)
        policy = _acquisition_policy(blob)
        self._attention_acquisition[price_path] = {
            "policy": policy,
            "context": _acquisition_context(blob),
        }
        resolved, unreadable = _resolved_declaration(blob, price_path)
        if unreadable:
            # Never silently pooled with a file whose resolution was read:
            # this key differs from every readable one, so the two are
            # different deployments until somebody fixes the record.
            declared["resolved_scope_unreadable"] = unreadable
        for sig, record in (blob.get("prices") or {}).items():
            op = by_sig.get(sig)
            seconds = (record or {}).get("seconds")
            if op is None or seconds is None:
                continue
            # Per operator, because the file's resolution covers both
            # families and the two are identified by different facts: the
            # linear attention kernel never reads the paged KV cache, so a KV
            # layout is not a fact about it.
            scope = dict(declared)
            if resolved is not None:
                scope.update(resolved.for_op(op))
            scope = self._with_declared_scope(scope, op, price_path)
            for name in (record or {}).get("kernels") or {}:
                canonical = _canonical_kernel(name)
                if canonical != name:
                    self._pooled_kernels.setdefault(
                        canonical, set()).add(name)
            self._attention_obs.append(
                (op, float(seconds), price_path, scope,
                 _measurement_identity(record, policy), _host_seconds(record),
                 _qualification(record)))
        self._attention_model = None

    def _with_declared_scope(self, scope: dict, op: dict,
                             price_path: str) -> dict:
        """The file's own scope, with a caller's declaration filling silences.

        Only silences. Where the file and the declaration both state a key and
        disagree, this raises: one of them is wrong about what ran, and which
        one is not something this can decide. Answering from either would file
        the measurement under a deployment it may not have been taken in, and
        a wrong scope answers where an absent one refuses.

        The keys it may fill are the two families' own scope keys and nothing
        else. A declaration is a reading of which kernel-selecting facts held;
        it is not a place to write measurement conditions the collector did
        not record, and letting it reach `_MEASUREMENT_CONDITIONS` would make
        a caller able to claim two files were captured alike.
        """
        declaration = self.declared_attention_scope
        if declaration is None:
            return scope
        offered = declaration.for_op(op)
        if not offered:
            return scope
        filled = dict(scope)
        for key in _ATTENTION_SCOPE_KEYS:
            if key not in offered:
                continue
            value = _hashable(offered[key])
            if key in filled:
                if filled[key] != value:
                    raise ValueError(
                        "%s was measured with %s=%r recorded in the file, and "
                        "the declared measured scope says %r. A caller may "
                        "name a fact the file leaves unstated, not overrule "
                        "one it states: fix whichever is wrong about what ran."
                        % (price_path, key, filled[key], value))
                continue
            filled[key] = value
        return filled

    def attention_design_points(self) -> list:
        """The collected observations with layer replicates collapsed.

        Every layer in one captured step shares that step's ragged structure,
        so 16 or 48 of them are 16 or 48 measurements of one design point, not
        16 or 48 points. Fitting them as points would inflate the residual
        degrees of freedom by an order of magnitude and report a law as checked
        when nothing independent ever checked it, so they are collapsed to
        their median here and the replicate count and spread kept with it.

        What decides that two observations are replicates is their FULL source
        identity, not their ragged structure: the operand shapes and dtypes,
        the recorded operand views, every context and scalar the key carries --
        which includes the native state and speculative branches -- the launch
        grid, the kernels that served them, and the declared scope. Two
        observations that differ in any of those are measurements of different
        work, and collapsing them to a median before the model ever sees them
        would hide that difference inside a single number.
        """
        groups: dict = {}
        for obs in self._attention_obs:
            op, seconds, source, scope, measurement, _host, _qual = obs
            key = (_design_identity(op), measurement,
                   attention.scope_key(scope))
            groups.setdefault(key, []).append(obs)
        points = []
        for members in groups.values():
            members.sort(key=lambda m: m[1])
            op, seconds, source, scope, measurement, _host, _qual = \
                members[len(members) // 2]
            # The treatment travels with the point into the fit, so a law is
            # identified by the conditions its measurements were taken under
            # as well as by the deployment. Without this a cold-cache point
            # and a warm-cache point, correctly kept apart here, would rejoin
            # as two independent points of one fit.
            scope = dict(scope or {})
            scope["measurement_treatment"] = measurement
            note = source
            if len(members) > 1:
                low, high = members[0][1], members[-1][1]
                sources = sorted({m[2] for m in members})
                note = ("%s (%d replicates, spread %.1f%%%s)"
                        % (source, len(members),
                           (high - low) / seconds * 100 if seconds else 0.0,
                           "" if len(sources) == 1
                           else ", from %d files" % len(sources)))
            note += self._host_note(members) + self._qualification_note(members)
            points.append((op, seconds, note, scope))
        return points

    @staticmethod
    def _qualification_note(members) -> str:
        """The shape-derived facts behind a point, kept where a reader sees them.

        The operand rotation a record realized is evidence about how cold its
        operands were held, and it is a function of this point's own footprint.
        It qualifies the number; it does not identify the law -- see
        `_QUALIFICATION_FIELDS` -- so it is written here rather than into the
        scope.
        """
        seen: dict = {}
        for member in members:
            for field, value in member[6] or ():
                seen.setdefault(field, set()).add(value)
        if not seen:
            return ""
        return " (" + ", ".join(
            "%s %s" % (field, "/".join(str(v) for v in sorted(
                values, key=str)))
            for field, values in sorted(seen.items())) + ")"

    @staticmethod
    def _host_note(members) -> str:
        """The host-time distribution behind a design point, and what is missing.

        Diagnostic only: host time is an outcome, so it never decides whether
        two measurements are the same design. A reader still needs it, because
        a point whose host time dominates its device time was measuring the
        launch rather than the kernel.
        """
        hosts = [m[5] for m in members]
        known = [value for value in hosts if value is not None]
        if not known:
            return " (host time: not recorded on any of %d)" % len(hosts)
        note = " (host time %.3g-%.3gs" % (min(known), max(known))
        if len(known) != len(hosts):
            note += ", absent on %d of %d" % (len(hosts) - len(known),
                                              len(hosts))
        return note + ")"

    def attention_model(self, *, strict: bool = True):
        """The fitted ragged attention model, built once from what was added.

        ``strict`` is the acceptance setting: a fit whose observations do not
        declare the scope keys its kernel turns on is refused rather than
        assumed. It is a keyword here only so a diagnostic caller can ask for
        the undeclared fits by name and get them stamped as such.
        """
        if self._attention_model is None or not strict:
            model = attention.Model.from_priced(
                self.attention_design_points(), strict=strict)
            if not strict:
                return model
            self._attention_model = model
        return self._attention_model

    def attention_coverage(self) -> dict:
        """What the ragged model can and cannot price, to be reported."""
        coverage = dict(self.attention_model().coverage())
        coverage["observations"] = len(self._attention_obs)
        coverage["design_points"] = len(self.attention_design_points())
        # How the numbers were acquired, reported beside what they cover. A
        # reader deciding whether to trust a modelled price needs to see that
        # two policies are present, or that one file recorded none -- neither
        # of which is visible from the laws, because the first is why two laws
        # exist and the second is why one does not.
        coverage["acquisition_policies"] = sorted(
            {tuple(entry["policy"])
             for entry in self._attention_acquisition.values()})
        coverage["acquisition_context"] = {
            path: dict(entry["context"])
            for path, entry in sorted(self._attention_acquisition.items())}
        coverage["launch_charge_seconds"] = self.launch_charge_seconds
        # Whether any of this was filed under a scope the FILES stated or one
        # a caller declared for them. A law fitted over declared facts is
        # exactly as good as the declaration, and a reader cannot weigh it
        # without being told the declaration was there.
        coverage["declared_measured_scope"] = (
            None if self.declared_attention_scope is None
            else self.declared_attention_scope.as_dict())
        # Which compiler specializations were pooled into one treatment, and
        # what they were. Empty for every library that priced nothing
        # geometry-specialized, which is most of them.
        coverage["pooled_kernel_specializations"] = {
            canonical: sorted(names)
            for canonical, names in sorted(self._pooled_kernels.items())}
        return coverage

    def _build(self) -> None:
        """Group every measured price by the operator *it* priced.

        Each record is paired with the operator from its own file's graph and
        grouped under its own scope. Two things were collapsing here:

        * **layout** -- a strided measurement was attached to the dense
          operator, so its seconds joined the dense curve (contaminating the
          median at every width it shared) and no strided curve existed at all,
          so a strided request at an unmeasured width had no support to sit in.
        * **scope** -- prices measured at different group widths or on
          different registration paths landed on one curve, which averages
          measurements of different work.
        """
        if self._curves_built:
            return
        self._observations.clear()
        for _cost_key, records in self._prices.items():
            for record in records:
                source = record.get("source")
                # Joined on the record's OWN raw signature, not on the cost key
                # it is filed under. The cost key collapses observations that
                # may differ in layout, and this join is what decides which
                # operator -- and therefore which layout -- a measurement is
                # attributed to. A record from before this field existed has no
                # signature and is left to exact-key use, as an unpaired one
                # always was.
                sig = record.get("signature")
                op = (None if sig is None
                      else (self._source_ops.get(source) or {}).get(sig))
                if op is None:
                    # No graph from this file, so nothing says what this price
                    # is a price of. Exact-signature use only; `unbuildable`
                    # already records why.
                    continue
                contract = contract_for(op.get("name", ""))
                if contract is None or contract.kind != "rows":
                    continue
                # The width this observation is a measurement OF. Read off the
                # operator first where its family declares where to look,
                # because the two readings are not the same number in the head
                # region: a head graph traced over 16384 hidden rows contains
                # an LM-head GEMM that ran at the request count, because
                # `compute_logits` selects each request's last token before
                # multiplying. Filing that GEMM at 16384 rows would put a
                # measurement of 2 rows of work on the 16384-row curve.
                #
                # For families with no declared reading the file's width is
                # still the only statement available, and it is used unchanged
                # -- this is a per-family refinement, not a new default. On the
                # existing library the two agree wherever both exist: 40 of 40
                # `gemm_a16w16` observations in run 5's price list.
                rows = executed_rows(op)
                if rows is None:
                    rows = self._rows.get(source)
                seconds = record.get("seconds")
                if rows is None or seconds is None:
                    continue
                key = (grouping_key(op), _scope_of(op, record.get("scope")))
                self._observations.setdefault(key, []).append(
                    (op, rows, float(seconds), source or "?",
                     tuple(record.get("kernels") or ())))
        self._curves_built = True

    # -- lookup ---------------------------------------------------------

    def lookup(self, op: dict, topology=None, registration=None):
        record, detail = super().lookup(op, topology, registration)
        if record is not None or detail != _OPEN_QUESTION:
            # Either answered, or refused for a reason that is a finding rather
            # than a gap. Both are returned as they came.
            return record, detail
        return self._parametric(op, detail, topology, registration)

    def _parametric(self, op: dict, original: str, topology=None,
                    registration=None):
        contract = contract_for(op.get("name", ""))
        if contract is None:
            return None, (f"{original}; and {op.get('name', '?')} has no "
                          "declared family contract, so there is no statement "
                          "of what its price may depend on")
        if contract.kind == "view":
            # Not a curve: a family whose price is structurally absent when the
            # recording proves it, and refused when the recording does not.
            return self._view_price(op, original, contract)
        if contract.kind != "rows":
            return self._modelled(op, original, contract, topology,
                                  registration)

        self._build()
        curve, verified_rows = self._curve_for(op, topology, registration)
        if curve is None:
            # Scope first: "this family is measured, but not here" is a
            # different gap from "nothing matches this operator", and only one
            # of them is closed by measuring a new width.
            _groups, scope_why = self._groups_for(op, topology, registration)
            if scope_why:
                return None, f"{original}; {scope_why}"
            return None, (f"{original}; and no measured operator matches this "
                          "one at any width" + self._layout_note(op))
        support = RowSupport(curve, max_gap_ratio=self.max_gap_ratio)
        answer = support.price(verified_rows)
        if isinstance(answer, Refusal):
            return None, (f"{original}; {answer.reason} "
                          f"(measured at {curve.measured_rows})")
        if answer.basis == "measured":
            # The curve already holds this width from another file; that is a
            # measurement, and it is reported as one.
            return (dict(_record(answer, curve, verified_rows),
                         **{INTERPOLATED_FLAG: False}),
                    answer.sources[0])
        return (dict(_record(answer, curve, verified_rows),
                     **{INTERPOLATED_FLAG: True}),
                f"{INTERPOLATED_SCHEME}{contract.family}/rows={verified_rows}")

    def _view_price(self, op: dict, original: str, contract):
        """Zero seconds, but only where the recording proves nothing ran.

        A slice that returns a view dispatches no kernel: the output is the
        input's storage at an offset, and producing it is host bookkeeping.
        That is not something to be measured and found small -- it is work that
        does not exist -- so it is declared, and it is declared from evidence
        the graph carries rather than from the operator's name.

        `output_aliases` is that evidence. It records, per output, whether the
        operator allocated it, decided as the trace ran by whether the output's
        storage is one of the operator's own inputs. An index means it wrote
        into a tensor that already existed; `None` means it allocated.

        Three outcomes, and the two refusals matter as much as the price:

        * no `output_aliases` at all -- a graph written before the field was
          recorded. Empty means *not known*, which the field's own docstring is
          careful to distinguish from *the same as the input*. Refused.
        * an output the operator allocated -- then it copied rather than
          viewed, and a copy of an arbitrary extent is real work that no
          measurement here covers. Refused.
        * every output an alias -- no kernel, and the price is zero.

        Reported under `ZERO_WORK_FLAG`, which exists for exactly this: fully
        accounted for, not a measurement, and never inferred from a zero time.
        """
        aliases = op.get("output_aliases")
        if not aliases:
            return None, (
                f"{original}; {contract.family} is priced at zero only where "
                "the recording shows it allocated nothing, and this graph "
                "records no output_aliases -- which is not known, not the same "
                "as not allocated")
        if any(alias is None for alias in aliases):
            return None, (
                f"{original}; this {contract.family} allocated its output, so "
                "it copied rather than viewed, and a copy is work no "
                "measurement here covers")
        return ({"seconds": 0.0,
                 "kernels": {},
                 "occurrences": 1,
                 "name": contract.family,
                 ZERO_WORK_FLAG: True,
                 "structural": {
                     "family": contract.family,
                     "basis": "alias",
                     "output_aliases": list(aliases),
                     "detail": ("output aliases an operand's storage, so no "
                                "kernel is dispatched"),
                 }},
                f"structural://{contract.family}/alias")
    def _modelled(self, op: dict, original: str, contract, topology=None,
                  registration=None):
        """A ragged family's price from its regime's law, or why there is none.

        Reached only behind the open question -- an operator nobody priced --
        so an exact measurement of this call, when one exists, has already been
        returned by the base class and is never displaced by a law.

        The result is marked `interpolated` and carries an `interpolated://`
        source. It is a prediction: honest coverage counts it as covered and
        never as measured, and that distinction is the reason this does not
        return a bare number.

        It also carries the launch composition the evidence behind that law
        shows, because a record's kernel count is not decoration: `body` adds
        ``max(1, len(record["kernels"]))`` launches per occurrence and the
        oracle charges per-launch overhead on top of the seconds. An empty
        kernels map would quietly count a multi-kernel attention wrapper as one
        launch, so a price whose composition the evidence does not establish is
        refused instead. The names are carried with no seconds attributed to
        them: how a modelled total divides between kernels is not something
        this knows, and splitting it evenly would be inventing the split.
        """
        if not self._attention_obs:
            # Nothing was collected at all, which is a different statement
            # from "this call is outside the law": there is no law. Said
            # first, because a regime refusal here would describe the key
            # when the answer is that nobody has measured this family.
            return None, (f"{original}; this is a ragged attention family and "
                          "nobody has measured it in this library. Its price "
                          "depends on how the batch pairs queries with "
                          "histories, so no row count stands in for one")
        scope = self._request_scope(op, topology, registration)
        model = self.attention_model()
        answer = model.price(op, scope)
        if isinstance(answer, attention.Refusal):
            # The family is named in the refusal, not only the reason: a
            # reader of "no entry for this signature" needs to know this is a
            # ragged family whose price depends on the batch structure rather
            # than on a row count, or the gap looks like an ordinary
            # unmeasured width.
            return None, (f"{original}; this is a ragged attention family, "
                          f"and {answer.reason}")
        scope = attention.scoped(op, scope)
        regime = attention.regime_of(op, None, scope)
        name = regime.name
        fit, _why = model.fit_for(name, scope)
        kernels, unevidenced, state = self._launch_composition(name, fit)
        attribution = "unknown: modelled total, not measured per kernel"
        if state == "unrecorded":
            # An unrecorded composition matters exactly as much as a launch
            # costs. `PriceLibrary.body` charges `launches *
            # seconds_per_launch`, so where that rate is zero the composition
            # buys nothing: no launch charge is inferred from it, and refusing
            # would be withholding a price over a number that could not have
            # changed it. Where the rate is nonzero -- or is not stated at all
            # -- the count IS a cost, and no count is invented to get past
            # this. The charge is read as configured; it is never set to zero
            # here.
            charge = getattr(self, "launch_charge_seconds", None)
            if isinstance(charge, (int, float)) \
                    and not isinstance(charge, bool) and float(charge) == 0.0:
                kernels, unevidenced = (), None
                attribution = (
                    "unknown and uncharged: no measurement behind this law "
                    "records which kernels served it, and this library "
                    "charges nothing per launch, so no launch count is "
                    "inferred from the absence")
            else:
                unevidenced = (
                    "%s -- and this library charges %s per launch, so the "
                    "count it would need is a cost, not a label"
                    % (unevidenced, "an unstated amount" if charge is None
                       else "%.3gs" % float(charge)))
        if unevidenced:
            return None, (f"{original}; this is a ragged attention family, "
                          f"and {unevidenced}")
        return ({
            "seconds": answer,
            # Names, not an attribution. `None` rather than a share, so
            # anything that reads a per-kernel number finds an absence instead
            # of a plausible fabrication; the launch COUNT is what the evidence
            # establishes and what `body` reads.
            "kernels": {kernel: None for kernel in kernels},
            "kernel_attribution": attribution,
            "occurrences": 1,
            "name": contract.family,
            INTERPOLATED_FLAG: True,
            "interpolation": {
                "family": contract.family,
                "basis": "modelled",
                "regime": name,
                "detail": fit.describe(),
                "measured_sources": sorted({
                    source for _op, _s, source, _scope, _how, _host, _q
                    in self._attention_obs}),
            },
        }, f"{INTERPOLATED_SCHEME}{contract.family}/{name}")

    def _launch_composition(self, regime_name: str, fit):
        """The kernels every measurement behind this law was served by.

        Returns ``(names, reason, state)``. The state separates the two ways
        this can come up empty, because they are different facts and only one
        of them is ever survivable. ``"disagree"`` means two records behind one
        law were served by different kernel sets: that is positive evidence
        that the regime has no fixed composition, and picking one would invent
        a launch count. ``"unrecorded"`` means nobody wrote the composition
        down -- the collector was run without kernel-id collection -- which
        says nothing about what ran and leaves the decision to what a launch
        actually costs. The caller resolves that; this only reports it.

        In practice `"unrecorded"` is the state that arrives here: the kernels
        a record was served by are part of its measurement identity, so two
        records served differently are separated into two laws before either
        is fitted. The disagreement branch is the backstop for a scope that
        stops carrying the composition, and it is kept because the cost of
        being wrong about it is a silently altered launch count.
        """
        compositions = set()
        wanted = attention.scope_key(fit.scope)
        for op, _seconds, _source, obs_scope, measurement, _host, _q in \
                self._attention_obs:
            obs_scope = dict(obs_scope or {})
            obs_scope["measurement_treatment"] = measurement
            obs_scope = attention.scoped(op, obs_scope)
            regime = attention.regime_of(op, None, obs_scope)
            if isinstance(regime, attention.Refusal) \
                    or regime.name != regime_name:
                continue
            if attention.scope_key(obs_scope) != wanted:
                continue
            compositions.add(kernels_of(measurement))
        compositions.discard(())
        if not compositions:
            return None, (
                "no measurement behind the %s law records which kernels "
                "served it, so how many launches this call is cannot be "
                "stated" % regime_name), "unrecorded"
        if len(compositions) > 1:
            shown = "; ".join(sorted(", ".join(c) for c in compositions))
            return None, (
                "the measurements behind the %s law were served by different "
                "kernel sets (%s), so this regime has no established launch "
                "composition to carry" % (regime_name, shown)), "disagree"
        return sorted(compositions.pop()), None, "known"

    def _request_scope(self, op: dict, topology, registration):
        """The static scope the asking deployment declares, or None.

        A request that declares nothing is not silently given a measured
        scope: `attention.Model.price` treats an undeclared request as not
        matching a scoped fit, which is the refusal that keeps a law measured
        under one deployment from answering for another.
        """
        scope = self._declared_scope(op)
        if "measurement_treatment" not in scope:
            treatment = self._treatment_for(op, scope)
            if treatment is not None:
                # One treatment across the domain this call belongs to, so
                # pricing under it states a fact rather than choosing between
                # laws. With several present and none declared, nothing is
                # filled in and the scope comparison refuses by name -- which
                # is the point: a warm-cache law is not a price for a cold
                # request.
                scope["measurement_treatment"] = treatment
        if registration is not None:
            scope.setdefault("registration", registration)
        if topology:
            scope.setdefault("topology", _topology_key(topology))
        return scope or None

    def _declared_scope(self, op: dict) -> dict:
        """What the asking deployment declares, for THIS call's family.

        A declaration may be per family -- which is what reading a resolved
        deployment produces, because the two families are identified by
        different facts -- or one mapping meant for whichever family asks.
        Both are accepted; a per-family declaration is selected by the
        operator's own name rather than by anything the caller passes, so a
        GDN call is never handed the unified family's KV facts.
        """
        declared = self.request_attention_scope
        if declared is None:
            return {}
        if hasattr(declared, "for_op"):
            return declared.for_op(op)
        if not isinstance(declared, dict) or not declared:
            return {}
        labels = set(attention_scope.FAMILY_LABELS.values())
        if set(declared) <= labels and all(
                isinstance(value, dict) for value in declared.values()):
            label = attention_scope.FAMILY_LABELS.get(op.get("name"))
            return dict(declared.get(label) or {})
        return dict(declared)

    def _treatment_for(self, op: dict, scope):
        """The treatment resolved WITHIN the domain this request asks about.

        Not across the library. A mixed library holds both families, and they
        are naturally measured under different treatments -- different region
        rebuilds, different rotations, and under a collector that records
        kernel ids for one run and not another. Asking whether the whole
        library shares one treatment answers "no" for a perfectly ordinary
        library and refuses every price in it.

        The domain is this call's family, its regime and its static operand
        geometry: the observations that could be points of the law this call
        would be priced from, and no others. Ambiguity inside that domain is
        still refused -- a cold-cache and a warm-cache measurement of the same
        law are two laws, and picking one would be choosing which to charge.

        The regime filter is dropped when this call's own regime cannot be
        read, which happens when the scope is not yet complete enough to say.
        Family and geometry still hold, so the answer is drawn from the same
        operator on the same heads either way.

        Family, regime and geometry are not the whole domain, and taking them
        for it is how an ordinary mixed library goes wrong. A real one holds
        the entries a registry already had beside the new family records, and
        a legacy attention entry may declare no static scope at all. Filtered
        only by family and geometry it is still a candidate here, so its
        treatment joins the set, the set has two members, and a request whose
        own scope matches exactly one deployment is told the treatment is
        ambiguous -- a valid modelled fallback lost to an entry that was never
        shown to be the same deployment. So the requested static scope filters
        too: see `_shown_to_match`. Those legacy entries keep answering
        exactly, by signature, through the path above this one; what they lose
        is the vote on a fallback that is not theirs.

        What is deliberately NOT done is to break a remaining tie by asking
        which candidate happens to have a fitted law. It is tempting -- an
        unfittable domain cannot answer anyway -- but the candidates left at
        that point are treatments that this request's own declared scope
        matches, and choosing among them by fittability charges the request
        under a treatment it never named because the alternative had too few
        points to check it. Two treatments inside the asked deployment are two
        laws, and which one a call was under is a fact about the call. So the
        ambiguity stands and the refusal names it.
        """
        family = op.get("name")
        geometry = attention.geometry_of(op)
        asked = attention.regime_of(op, None, attention.scoped(op, scope))
        wanted = None if isinstance(asked, attention.Refusal) else asked.name
        requested = dict(scope or {})
        requested.pop("measurement_treatment", None)
        treatments = set()
        for obs_op, _s, _src, obs_scope, measurement, _host, _q in \
                self._attention_obs:
            if obs_op.get("name") != family:
                continue
            if attention.geometry_of(obs_op) != geometry:
                continue
            if not _shown_to_match(obs_scope, requested):
                continue
            if wanted is not None:
                full = dict(obs_scope or {})
                full["measurement_treatment"] = measurement
                regime = attention.regime_of(
                    obs_op, None, attention.scoped(obs_op, full))
                if isinstance(regime, attention.Refusal) \
                        or regime.name != wanted:
                    continue
            treatments.add(measurement)
        return treatments.pop() if len(treatments) == 1 else None

    def _layout_note(self, op: dict) -> str:
        """Say so when operand layout is why nothing matched.

        Without this the refusal reads "no measured operator matches this one
        at any width", which is true and unhelpful: the family *was* measured,
        at widths that bracket this one, on operands in a different memory
        arrangement. Naming that is the difference between a gap somebody can
        close and a gap somebody re-measures the wrong thing to close.
        """
        mine = _layout_note_for(op)
        others = {_layout_note_for(measured)
                  for key, obs in self._observations.items()
                  if key[0][0] == op.get("name", "")
                  for measured, *_ in obs}
        others.discard(mine)
        if not others:
            return ""
        return (f". This request's operands are {mine} while this family was "
                f"measured on {', '.join(sorted(others))}, which is a "
                "different operator rather than another width of this one")

    def _groups_for(self, op: dict, topology, registration):
        """The observation groups this request may be answered from.

        Keyed by structure *and* scope, so a request is only ever answered from
        measurements taken at its own group width on its own registration path.
        A caller that names neither is allowed through only while there is one
        scope to be: with several, picking would be the silent spend the base
        class already refuses for collectives, so this refuses too.
        """
        structural = grouping_key(op)
        present = {key[1]: obs for key, obs in self._observations.items()
                   if key[0] == structural}
        if not present:
            return [], None
        if present.keys() == {_LOCAL}:
            # A local family: one scope by construction, and the request's
            # topology and registration say nothing about its cost.
            return [present[_LOCAL]], None
        if topology is None and registration is None:
            if len(present) == 1:
                return list(present.values()), None
            return [], ("this family is measured in more than one scope and "
                        "the request named none: "
                        + "; ".join(sorted(_scope_note(k) for k in present)))
        wanted = _scope_key({"topology": topology,
                             "registration": registration})
        exact = present.get(wanted)
        if exact is not None:
            return [exact], None
        return [], (f"no measurement of this family at {_scope_note(wanted)} "
                    "(have: "
                    + "; ".join(sorted(_scope_note(k) for k in present)) + ")")

    def _curve_for(self, op: dict, topology=None, registration=None):
        """The measured curve for this operator, and the width it sits at."""
        groups, _why = self._groups_for(op, topology, registration)
        observations = [entry for group in groups for entry in group]
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
        if self.no_file_width:
            note += (f"; {len(self.no_file_width)} file(s) state no width of "
                     "their own and are read per operator")
        if self._attention_obs:
            note += ("; %d ragged attention observation(s) over %d design "
                     "point(s)" % (len(self._attention_obs),
                                   len(self.attention_design_points())))
        return base + note


def _traced_rows(graph: dict) -> Optional[int] | tuple[None, str, bool]:
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

    A disagreement between the first two is a real conflict. It is returned
    distinctly from an absent reading, because the two are not the same loss:
    a graph that states no width can still hold operators that state their
    own, while a graph whose two statements contradict each other is not a
    trustworthy source for any of them.

    Returns the width, or ``(None, why, conflicting)``.
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
            "choosing between them would be a guess"), True
    rows = declared if declared is not None else executed
    if rows is None or rows <= 0:
        return None, (
            "the graph records neither provenance.execution.body_rows_traced "
            "nor an embedding whose token operand states the executed width"), False
    return rows
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


def _layout_note_for(op: dict) -> str:
    """This operator's operand layout, named for a refusal message."""
    positions = sorted(int(pos) for pos, _ in
                       (tuple(x) for x in op.get("layouts") or ()))
    if not positions:
        return "a dense rebuild"
    return f"a recorded view at operand {positions}"


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
        if record.get(ZERO_WORK_FLAG):
            # Not a cheap measurement and not a gap: a case the engine is known
            # not to run at all, such as a collective at group width one or the
            # head on a chunk that produces no token. It is fully accounted for
            # and it is not a measurement, so it gets its own count.
            split["zero_work"] += 1
        elif record.get(INTERPOLATED_FLAG):
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
