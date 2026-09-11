"""One derived graph, re-pointed at a different cohort.

Deriving a graph costs 0.39-0.71 s of tracing on a CPU. A decode step costs
about 13 ms on the device. So a cost model that derives a graph per step is
fifty times slower than the thing it predicts, and "derivation is cheap" is only
true per *distinct structure*. This module is what makes the structures repeat.

What licenses it is a measurement. Two decode graphs at the same capture rung,
one with every request at context 1151 and one with 32 requests spread over
eight different contexts, were compared field by field across all 2439
operators: identical operator count, identical order, identical
``input_shapes``, ``output_shapes``, ``dtypes``, ``scalars``, ``launch``,
``layouts``, ``param_names``, ``abi``, ``inputs_from``, ``output_aliases`` and
``dies_at``. Exactly one field differed, on exactly 16 operators: the attention
``context``. Changing the rung instead moves six fields across up to 2117
operators, which is why a rung is a template and a context is a binding.

So a **template** is a graph derived once for a structure -- a group width, a
rank, a query-length vector, a prefill split, a capture rung -- and **binding**
re-points its per-request metadata at another cohort of the same structure.

Allocation is not part of the cohort and is not invented here
-------------------------------------------------------------

``slot_mapping``, ``block_tables`` and the two ``non_spec_state_indices``
tensors say which physical KV blocks and which linear-attention state slots this
batch occupies. They are the block manager's output, not a function of the
request lengths, and they are neither cosmetic nor nondeterministic
bookkeeping: a kernel that walks scattered blocks and one that walks contiguous
blocks do the same arithmetic over different memory. ``signature_of`` already
treats them as cost-relevant in one direction -- ``slot_mapping`` is in the
price key, ``block_tables`` deliberately is not.

Binding therefore **refuses** unless the caller supplies an allocation for the
cohort. Three answers are admissible and the module makes the caller pick one:

* :class:`NativeAllocation` -- wrap the real scheduler/block manager and take
  its actual assignment for this batch. Correct by construction; needs the
  engine's own CPU-side allocator, which is why it is a protocol here rather
  than an implementation.
* an abstraction that has been **measured** against the native allocator and
  carries the scope that measurement covers;
* :class:`CarriedAllocation` -- keep the template's own allocation and say so.
  Explicitly *unmeasured*: it is a stated approximation for CPU-side structural
  work, it is recorded in the bound graph's provenance, and it is not evidence
  for any claim about a price keyed on ``slot_mapping``.

Passing nothing gets no graph. Stripping the fields to make two price keys agree
is not one of the options.
"""

from __future__ import annotations

from typing import Optional, Protocol

from atom.compass.core.cost.base import StepShape

__all__ = ["BindRefusal", "AllocationSource", "CarriedAllocation",
           "template_key", "bind_cohort", "TemplateGraphs",
           "ALLOCATOR_FIELDS", "BOUND_FIELDS", "CARRIED_CONSTANTS"]


class BindRefusal(Exception):
    """Binding would have to guess. It does not guess."""


#: Per-request metadata that belongs to an allocator, not to the cohort.
#: ``slot_mapping`` and ``block_tables`` are the KV block manager's; the two
#: ``non_spec_state_indices`` tensors are the same thing for the
#: linear-attention state slots, which the GDN layers index instead of a block
#: table.
ALLOCATOR_FIELDS = ("slot_mapping", "block_tables",
                    "non_spec_state_indices_tensor",
                    "non_spec_state_indices_in_tensor")

#: Context fields the native producer sets to a literal, so the cohort cannot
#: change them, with the producer named. Traced, not inferred from samples.
CARRIED_CONSTANTS = {
    "min_seqlen_q": ("literal 0 in "
                     "atom/model_ops/attentions/aiter_attention.py:1071 "
                     "(prepare_decode) and "
                     "atom/model_ops/attentions/backends.py:532; "
                     "atom/plugin/sglang/kimi_k3_bridge.py:478 computes it "
                     "from extend_lens instead, so this rule is scoped to the "
                     "aiter attention path and refuses elsewhere"),
}


class AllocationSource(Protocol):
    """Where a bound graph's block and state assignment comes from."""

    def allocation_for(self, shape: StepShape) -> dict:
        """The allocator fields for this cohort, by name."""

    def describe(self) -> str:
        ...

    @property
    def measured(self) -> bool:
        """Whether this assignment is the real allocator's or an abstraction."""


class CarriedAllocation:
    """Keep the template's allocation. An approximation, declared as one.

    Use when the work is structural -- counting shapes, sizing a cache, timing
    derivation -- and the allocation is not what is being studied. It is not a
    basis for a price that is keyed on ``slot_mapping``, and
    :attr:`measured` says so to anything that asks.
    """

    measured = False

    def __init__(self, why: str) -> None:
        self.why = why

    def allocation_for(self, shape: StepShape) -> dict:
        return {}

    def describe(self) -> str:
        return f"CarriedAllocation(unmeasured; {self.why})"


def _rows(shape: StepShape):
    """``(query_len, context_len)`` per request, in the batch's own order.

    Order is kept. ``StaticGraphs.key`` canonicalises by sorting whole rows
    because two permutations derive the same graph; binding is not a cache key
    and writes the metadata the cohort actually has.
    """
    return tuple((int(q), int(c))
                 for q, c in zip(shape.num_scheduled_tokens, shape.context_lens))


def template_key(shape: StepShape):
    """What makes two cohorts share a template.

    Everything ``StaticGraphs.key`` uses except the context lengths: the query
    lengths row by row (which set every tensor dimension and the attention
    branch), how much of the batch is prefill, the group widths and this rank's
    coordinates, the replay bucket, and whether it ran compiled. Context lengths
    are exactly what binding supplies, so they are exactly what the template key
    drops.
    """
    queries = tuple(int(q) for q in shape.num_scheduled_tokens)
    groups = tuple(sorted((str(k), int(v))
                          for k, v in (shape.topology or {}).items()))
    coords = tuple(sorted((str(k), int(v))
                          for k, v in (shape.rank_coords or {}).items()))
    return (queries, int(shape.num_prefill_tokens), groups, coords,
            shape.capture_bucket, shape.compiled)


def _cu_seqlens(queries):
    out, total = [0], 0
    for q in queries:
        total += q
        out.append(total)
    return out


def _bind(key, template_value, rows):
    """One context entry, recomputed for ``rows``, or :class:`BindRefusal`.

    Every formula here is a property of the batch that the runner also computes
    from the batch. None reads the template's value except to keep a constant
    the cohort cannot change, or to learn a section count.
    """
    queries = [q for q, _ in rows]
    contexts = [c for _, c in rows]
    if key == "context_lens":
        return contexts
    if key == "positions":
        # A request's next position is the last index of its history; a
        # multi-token query runs to the end of its chunk. The tensor is M-RoPE,
        # so it is `position_rows` identical sections laid end to end -- 96
        # entries for a 32-token decode at three rows. The section count comes
        # from the template's own length rather than a constant, because the
        # template and the cohort share a query-length vector by construction
        # and so share the token count.
        per_token = []
        for q, c in rows:
            per_token.extend(range(c - q, c))
        tokens = len(per_token)
        if not tokens or len(template_value) % tokens:
            raise BindRefusal(
                f"positions has {len(template_value)} entries for {tokens} "
                "tokens; the section layout is not what this rule assumes")
        return per_token * (len(template_value) // tokens)
    if key == "max_seqlen_k":
        return max(contexts) if contexts else 0
    if key == "max_seqlen_q":
        return max(queries) if queries else 0
    if key == "cu_seqlens_q":
        return _cu_seqlens(queries)
    if key == "cu_seqlens_k":
        # Recorded as null on the decode path; a template that carries one is a
        # structure this function has not been shown.
        if template_value is None:
            return None
        raise BindRefusal("cu_seqlens_k is set in the template")
    if key == "non_spec_query_start_loc":
        # The linear-attention layers take the same cumulative query offsets
        # over the non-speculative requests, recorded as a ``[values, dtype]``
        # pair. Speculative decoding is off in this deployment, so every
        # request is non-spec; a template whose non-spec count differs from its
        # batch is a structure this function has not been shown.
        values, dtype = template_value
        if len(values) != len(queries) + 1:
            raise BindRefusal("non_spec_query_start_loc is not over the whole "
                              "batch; speculative decoding changes the "
                              "structure, not the cohort")
        return [_cu_seqlens(queries), dtype]
    if key in CARRIED_CONSTANTS:
        return template_value
    if key in ("num_prefills", "num_prefill_tokens", "num_decodes",
               "num_decode_tokens", "num_spec_decodes",
               "num_spec_decode_tokens", "num_actual_tokens", "is_prefill",
               "has_cached", "state", "replayssm", "block_tables_shape"):
        # Structure, not cohort: fixed by the template key, and a cohort that
        # changed one of them would need its own template.
        return template_value
    raise BindRefusal(f"no rule for context field {key!r}")


def bind_cohort(template: dict, shape: StepShape,
                allocation: Optional[AllocationSource] = None) -> dict:
    """A copy of ``template`` whose per-request metadata describes ``shape``.

    ``allocation`` is required whenever the template carries allocator fields.
    Refuses rather than guesses, in both directions: an operator with a context
    field this module has no rule for raises, and so does a missing allocation.
    """
    rows = _rows(shape)
    has_allocator = any(
        tuple(entry)[0] in ALLOCATOR_FIELDS
        for op in template["ops"] for entry in (op.get("context") or ()))
    if has_allocator and allocation is None:
        raise BindRefusal(
            "this template carries block/state allocation "
            f"({', '.join(ALLOCATOR_FIELDS)}) and no AllocationSource was "
            "given. Supply the engine's own allocator, or a measured "
            "abstraction, or CarriedAllocation to say explicitly that the "
            "template's assignment is being reused unmeasured.")
    supplied = allocation.allocation_for(shape) if allocation else {}

    ops, rebound = [], 0
    for op in template["ops"]:
        context = op.get("context")
        if not context:
            ops.append(op)
            continue
        new, changed = [], False
        for entry in context:
            key, value = tuple(entry)
            if key in ALLOCATOR_FIELDS:
                bound = supplied.get(key, value)
            else:
                bound = _bind(key, value, rows)
            new.append([key, bound])
            changed = changed or bound != value
        rebound += bool(changed)
        op = dict(op)
        op["context"] = new
        ops.append(op)

    bound_graph = dict(template)
    bound_graph["ops"] = ops
    provenance = dict(template.get("provenance") or {})
    provenance["binding"] = {
        "source": "template",
        "rows": list(rows),
        "operators_rebound": rebound,
        "allocation": allocation.describe() if allocation else "none needed",
        "allocation_measured": bool(allocation and allocation.measured),
        "allocation_fields": [k for k in ALLOCATOR_FIELDS
                              if k in supplied] if supplied else [],
        "carried_constants": {k: why for k, why in CARRIED_CONSTANTS.items()},
    }
    bound_graph["provenance"] = provenance
    return bound_graph


#: Fields a bound graph reproduces exactly. Everything a derived graph carries
#: outside ``context``.
BOUND_FIELDS = ("name", "input_shapes", "output_shapes", "dtypes", "group",
                "scalars", "int_values", "launch", "int_ranges", "layouts",
                "param_names", "abi", "inputs_from", "output_aliases",
                "dies_at")


class TemplateGraphs:
    """A ``GraphSource`` that binds rather than derives, when it can.

    Holds templates by :func:`template_key` and an optional deriver for the
    structures it has not seen. Counts hits, binds, derivations and refusals,
    because the only claim worth making about a cache is one measured on a real
    schedule.

    ``representative_rank`` is the rank whose graph stands for the group.
    :func:`template_key` carries this rank's coordinates, and derivation
    produces rank 0's shard whatever rank is asked for -- it builds a one-rank
    gloo group and tells it to report the wider width, leaving
    ``rank_in_group`` at 0. So every template frozen so far is keyed at rank 0,
    and at TP>1 a shape from rank 1 matches none of them. Falling back to the
    representative is the predeclared aggregation: one uniformly sharded rank
    stands for its peers, which is what the derivation was already doing
    silently. ``representative_hits`` counts the subset of ``hits`` served that
    way, because "rank 3 was priced from rank 0's graph" is a different claim
    from "rank 3 was priced from rank 3's graph" and a report has to be able to
    tell them apart. A template keyed at the asking rank always wins; the
    fallback only fires on a miss.
    """

    def __init__(self, templates=None, derive=None, allocation=None,
                 representative_rank: int = 0) -> None:
        self._templates = dict(templates or {})
        self._derive = derive
        self._allocation = allocation
        self._representative = int(representative_rank)
        self.hits = 0
        self.binds = 0
        self.derivations = 0
        self.representative_hits = 0
        self.refusals = {}

    def add(self, shape: StepShape, graph: dict) -> None:
        self._templates[template_key(shape)] = graph

    def _representative_key(self, key):
        """``key`` with every rank coordinate moved to the representative."""
        coords = tuple((group, self._representative)
                       for group, _ in (key[3] or ()))
        return key[:3] + (coords,) + key[4:]

    def graph_for(self, shape: StepShape) -> Optional[dict]:
        key = template_key(shape)
        template = self._templates.get(key)
        if template is None:
            standin = self._representative_key(key)
            if standin != key:
                template = self._templates.get(standin)
                if template is not None:
                    self.representative_hits += 1
        if template is None:
            if self._derive is None:
                self.refusals[key] = "no template and no deriver"
                return None
            template = self._derive(shape)
            if template is None:
                self.refusals[key] = "deriver produced nothing"
                return None
            self.derivations += 1
            self._templates[key] = template
        else:
            self.hits += 1
        try:
            bound = bind_cohort(template, shape, self._allocation)
        except BindRefusal as exc:
            self.refusals[key] = str(exc)
            return None
        self.binds += 1
        return bound

    def describe(self) -> str:
        where = (self._allocation.describe() if self._allocation
                 else "no allocation source")
        stood_in = (f", {self.representative_hits} of them from rank "
                    f"{self._representative}'s graph"
                    if self.representative_hits else "")
        return (f"TemplateGraphs({len(self._templates)} templates, "
                f"{self.hits} hits{stood_in}, {self.derivations} derivations, "
                f"{self.binds} binds, {len(self.refusals)} refused; {where})")
