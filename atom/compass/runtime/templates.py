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

* :class:`NativeAllocation` -- take the scheduler's own assignment for this
  batch, offered per step by whatever drives the step. Correct by
  construction, and encoded through :class:`BatchSpec` so that the same
  code turns a block table into a ``slot_mapping`` here and in a capture.
  It refuses when nothing was offered, when the record is a previous
  step's, when the batch mixes prefill and decode, and when a
  state-bearing batch carries no state slots.
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
           "NativeAllocation", "NativeStepAllocation",
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


class NativeStepAllocation:
    """One step's assignment, exactly as the scheduler already holds it.

    Not a model of an allocator: nothing here computes a block id or a slot.
    Every field is read off `ScheduledBatch`, which the scheduler has already
    filled in before the runner is called:

    * ``block_tables`` -- one row per request in batch order, each the
      request's own list of physical block ids
      (`ScheduledBatch.block_tables`, itself `[seq.block_table for seq ...]`).
    * ``state_slots`` with ``state_rows`` -- the committed state slot of each
      state-bearing request (`state_slots_committed`) and *which batch row each
      one belongs to*. The two are separate because the scheduler's list is
      already filtered: it holds one entry per seq with
      ``has_per_req_cache and state_slot >= 0``, so its index is not the batch
      index and reading it positionally would hand row 5's slot to row 3 in any
      batch that mixes state-bearing requests with requests that hold none.
    * ``num_prefill_seqs`` -- how many leading rows are doing prefill, as
      `ScheduledBatch.total_seqs_num_prefill` counts them. Prefills lead the
      batch: the scheduler indexes them as ``batch.req_ids[:num_prefill]`` and
      `CommonAttentionBuilder.prepare_prefill` walks ``range(bs)`` with
      ``bs = batch.total_seqs_num_prefill``. It is carried rather than derived
      from the token count, because "this batch contains prefill tokens" and
      "every request in it is prefilling" are different statements and only the
      scheduler knows the second.

    ``rows`` is carried so the record can be checked against the shape it is
    offered for rather than trusted: a record from the previous step describes
    a different batch and would otherwise be spliced in silently.

    ``shared_across_ranks`` is true for the ATOM scheduler because the
    assignment is not a rank's: `Sequence.block_table` and `Sequence.state_slot`
    live on the request, one scheduler owns them, and every rank in the TP
    group is handed the same batch. It is a field rather than an assumption so
    that a producer for which it is false can say so and be refused.
    """

    __slots__ = ("rows", "block_tables", "state_slots", "state_rows",
                 "num_prefill_seqs", "source", "rank_coords",
                 "shared_across_ranks")

    def __init__(self, *, rows, block_tables, state_slots, num_prefill_seqs,
                 state_rows=None, source="", rank_coords=None,
                 shared_across_ranks=True) -> None:
        self.rows = tuple((int(q), int(c)) for q, c in rows)
        self.block_tables = tuple(tuple(int(b) for b in row)
                                  for row in block_tables)
        self.state_slots = (None if state_slots is None
                            else tuple(int(s) for s in state_slots))
        #: Batch row per entry of ``state_slots``. Defaults to the leading rows
        #: only when the two lengths already agree, which is the case where the
        #: filtered list and the batch coincide; otherwise it is required.
        if state_rows is not None:
            self.state_rows = tuple(int(r) for r in state_rows)
        elif self.state_slots is not None and len(self.state_slots) == len(
                self.rows):
            self.state_rows = tuple(range(len(self.rows)))
        else:
            self.state_rows = None
        self.num_prefill_seqs = int(num_prefill_seqs)
        self.source = str(source)
        self.rank_coords = dict(rank_coords or {})
        self.shared_across_ranks = bool(shared_across_ranks)


class NativeAllocation:
    """The scheduler's own assignment, encoded the way a capture records it.

    Holds no allocator and reimplements none. The step's record is *offered*
    by whatever is driving the step -- the runner, before it asks for a cost --
    and this turns it into the recorded metadata fields by handing it to
    :class:`BatchSpec`, the same object that describes a captured batch. So the
    encoding of ``slot_mapping`` from a block table, and of the padded
    ``block_tables`` row, is written once and is the same on both sides.

    It refuses in every direction rather than filling in:

    * no record offered for this step -- the offline case, and the one that
      makes an unattended run stop instead of quietly reusing a template's
      blocks;
    * a record whose per-request rows are not the shape's, which is a stale
      record;
    * a record read at another rank, unless its producer declared the
      assignment shared;
    * a batch that is part prefill and part decode, which has no single
      :class:`BatchSpec` kind and so no single encoding here;
    * a batch with state-bearing layers and no state slots.

    :attr:`measured` is true: the fields are the allocator's own output, not an
    abstraction of it. What that does *not* claim is that the price keyed on
    them was measured at this assignment -- coverage says that separately.
    """

    measured = True

    def __init__(self, *, block_size: int, max_model_len: int,
                 position_rows: int = 1, num_spec_step: int = 0,
                 needs_state: bool = True) -> None:
        self.block_size = int(block_size)
        self.max_model_len = int(max_model_len)
        self.position_rows = int(position_rows)
        self.num_spec_step = int(num_spec_step)
        #: Whether a record without state slots is a refusal. True for this
        #: deployment, whose GDN layers index a state pool every step.
        self.needs_state = bool(needs_state)
        self._record = None
        self.offered = 0
        self.answered = 0

    def offer(self, record: NativeStepAllocation) -> None:
        """Hand this step's assignment over. Replaces any previous one."""
        self._record = record
        self.offered += 1

    def clear(self) -> None:
        self._record = None

    def allocation_for(self, shape: StepShape) -> dict:
        record = self._record
        if record is None:
            raise BindRefusal(
                "no native allocation was offered for this step. The block "
                "and state assignment is the scheduler's, and nothing here "
                "invents one: drive this oracle from a runner that offers the "
                "batch's own allocation, or say explicitly with "
                "CarriedAllocation that a template's assignment is being "
                "reused unmeasured.")
        rows = _rows(shape)
        if record.rows != rows:
            raise BindRefusal(
                f"the offered allocation is for {len(record.rows)} requests "
                f"{record.rows[:3]}... and this shape has {len(rows)} "
                f"{rows[:3]}...; a record from another step is not this "
                "step's assignment")
        coords = {str(k): int(v) for k, v in (shape.rank_coords or {}).items()}
        if coords != record.rank_coords and not record.shared_across_ranks:
            raise BindRefusal(
                f"the allocation was read at {record.rank_coords} and this "
                f"shape is rank {coords}; its producer did not declare the "
                "assignment shared across ranks")
        prefilling = record.num_prefill_seqs
        if prefilling and prefilling != len(rows):
            # The gap, named rather than papered over. `BatchSpec` has one
            # `kind` for the whole batch, and so does the engine: at
            # `atom/model_ops/attentions/backends.py:613` a batch with any
            # prefill token goes down `prepare_prefill`, which walks
            # `range(batch.total_seqs_num_prefill)` and writes metadata for the
            # leading prefill rows alone. Encoding all of the rows here would
            # produce a `slot_mapping` longer than the one the runner builds,
            # and encoding the leading rows would drop the decode rows'
            # allocation entirely. Neither is this step's assignment, so it is
            # refused. Closing it needs a per-request kind in `BatchSpec` and a
            # deriver that can trace such a batch -- and first, evidence from
            # the engine that it produces one, since the backend as written
            # would not serve it either.
            raise BindRefusal(
                f"this batch has {prefilling} prefill rows and "
                f"{len(rows) - prefilling} decode rows. BatchSpec carries one "
                "kind for the batch, and the attention backend takes the "
                "whole batch down the prefill path while preparing metadata "
                "for the prefill rows only, so neither encoding is this "
                "step's assignment.")

        from atom.compass.runtime.batch_spec import BatchSpec

        spec = BatchSpec(
            # The engine's own rule, not an inference from the token count:
            # a batch whose scheduler says no request is prefilling is a
            # decode batch, and one where every request is prefilling is a
            # prefill batch. The mixed case never reaches here.
            kind="prefill" if prefilling else "decode",
            query_lens=tuple(q for q, _ in rows),
            context_lens=tuple(c for _, c in rows),
            block_size=self.block_size,
            max_model_len=self.max_model_len,
            capture_bucket=shape.capture_bucket,
            num_spec_step=self.num_spec_step,
            block_tables=record.block_tables,
            position_rows=self.position_rows,
        )
        try:
            attention = dict(spec.attention_context())
        except ValueError as exc:
            raise BindRefusal(
                f"the scheduler's assignment does not describe a runnable "
                f"batch: {exc}") from exc
        supplied = {"slot_mapping": attention["slot_mapping"],
                    "block_tables": attention["block_tables"]}
        if record.state_slots is not None:
            if record.state_rows is None:
                raise BindRefusal(
                    f"the allocation carries {len(record.state_slots)} state "
                    f"slots for {len(rows)} requests and does not say which "
                    "row each belongs to. The scheduler's own list holds only "
                    "the state-bearing requests, so its index is not the "
                    "batch index, and this module will not guess which row is "
                    "which.")
            if len(record.state_rows) != len(record.state_slots):
                raise BindRefusal(
                    "the allocation carries "
                    f"{len(record.state_rows)} state rows for "
                    f"{len(record.state_slots)} state slots")
            by_row = dict(zip(record.state_rows, record.state_slots))
            missing = [i for i in range(len(rows)) if i not in by_row]
            if missing:
                raise BindRefusal(
                    f"rows {missing[:4]} of {len(rows)} hold no state slot. "
                    "The recorded metadata is one index per request, so a "
                    "batch that mixes state-bearing requests with requests "
                    "that hold none has no encoding here -- and filling the "
                    "gaps with row numbers is the fresh-pool assumption under "
                    "another name.")
            ordered = [by_row[i] for i in range(len(rows))]
            gdn = dict(spec.gdn_context(state_slots=ordered))
            supplied["non_spec_state_indices_tensor"] = gdn[
                "non_spec_state_indices_tensor"]
            supplied["non_spec_state_indices_in_tensor"] = gdn[
                "non_spec_state_indices_in_tensor"]
        elif self.needs_state:
            raise BindRefusal(
                "the offered allocation carries no state slots, and this "
                "deployment's linear-attention layers index a state pool "
                "every step. A missing slot set is refused rather than "
                "defaulted to the batch order, which is only what a fresh "
                "pool happens to hand out.")
        self.answered += 1
        return supplied

    def describe(self) -> str:
        record = self._record
        held = ("none offered" if record is None
                else f"{len(record.rows)} requests from {record.source}")
        return (f"NativeAllocation(block_size={self.block_size}, "
                f"state={'required' if self.needs_state else 'optional'}; "
                f"{held}; offered {self.offered}, answered {self.answered})")

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
    if key in ("group", "group_world_size"):
        # The process group a collective runs in, and how wide it is, written
        # by `derive.py` where it synthesizes the operator. Topology, not
        # cohort: `template_key` already carries the topology, so a template
        # keyed at this width is never bound to a cohort at another one, and
        # rebinding the name would name a group the pricer then has to perform
        # the real collective in.
        return template_value
    raise BindRefusal(f"no rule for context field {key!r}")


def _fit_allocation(key, template_value, native_value, padding):
    """The allocator's value, in the layout the template recorded.

    A capture at bucket 32 running twenty requests writes a padded buffer, and
    what `forward_ctx` records is the buffer, not the twenty active entries.
    The scheduler's record is the active batch and nothing else. So where the
    template is longer, the active entries are written over its head and its
    own tail -- the capture's pad, whatever the runner filled it with -- is
    kept and counted. Where the native value is longer, binding refuses: an
    allocation that does not fit the buffer it is being written into is not
    this step's allocation.

    ``block_tables`` is excepted and replaced whole. Its length is
    ``rows * used``, where ``used`` is the column count the longest context
    needs, so the two sides differing in length is the cohort changing, which
    is the entire point of binding. It is also the one allocator field
    `signature_of` deliberately keeps out of the price key.
    """
    if key == "block_tables":
        return list(native_value)
    if (isinstance(template_value, list) and len(template_value) == 2
            and isinstance(template_value[0], list)):
        # ``[values, dtype]``, as the state-index tensors are recorded.
        if native_value[1] != template_value[1]:
            raise BindRefusal(
                f"{key} is {native_value[1]} in the allocation and "
                f"{template_value[1]} in the template")
        inner = _fit_allocation(key, template_value[0], native_value[0],
                                padding)
        return [inner, template_value[1]]
    if not isinstance(template_value, list):
        raise BindRefusal(
            f"{key} is not a list in the template; this module has no rule "
            f"for writing an allocation into a {type(template_value).__name__}")
    if len(native_value) == len(template_value):
        return list(native_value)
    if len(native_value) < len(template_value):
        padding[key] = len(template_value) - len(native_value)
        return list(native_value) + list(template_value[len(native_value):])
    raise BindRefusal(
        f"the allocation supplies {len(native_value)} entries for {key} and "
        f"the template records {len(template_value)}; a longer allocation "
        "does not fit the buffer the capture recorded")


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
    padding: dict = {}

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
                if key in supplied:
                    bound = _fit_allocation(key, value, supplied[key],
                                            padding)
                elif allocation.measured:
                    # A measured source that answered without this
                    # field does not know where this batch's blocks
                    # or state slots are. Falling back to the
                    # template's is the carried approximation under
                    # another name, and it would arrive stamped
                    # `allocation_measured: true`.
                    raise BindRefusal(
                        f"the allocation source supplied no {key!r}, and "
                        "it reports itself measured. The template's own "
                        "value is this cohort's only if something says so "
                        "explicitly.")
                else:
                    bound = value
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
        # Where the template's buffer was wider than the batch, and by
        # how much: those entries are the capture's pad, not this
        # step's assignment, and a reader has to be able to tell.
        "allocation_padding": dict(padding),
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
