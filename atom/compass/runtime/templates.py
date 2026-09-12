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
from atom.compass.runtime.batch_spec import extent_scope_of

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
                 needs_state: bool = True, cudagraph_mode=None) -> None:
        self.block_size = int(block_size)
        self.max_model_len = int(max_model_len)
        self.position_rows = int(position_rows)
        self.num_spec_step = int(num_spec_step)
        #: The deployment's declared capture mode. Held here for the same
        #: reason `TemplateGraphs` holds it: the state tail this class encodes
        #: is mode-specific. A FULL replay writes the bucket's counts with a
        #: PAD-filled state tail; a PIECEWISE one writes the active counts.
        #: Left ``None``, a bucketed decode is refused rather than encoded
        #: under whichever rule happened to be the default.
        self.cudagraph_mode = cudagraph_mode
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
            # The declared mode, so the state tail this class supplies is the
            # one the engine writes: `gdn_context` refuses a bucketed decode
            # without it rather than pick a rule.
            cudagraph_mode=self.cudagraph_mode,
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
            try:
                gdn = dict(spec.gdn_context(state_slots=ordered))
            except ValueError as exc:
                # An undeclared capture mode, most often. A refusal here is
                # this module's own vocabulary; a bare ValueError out of a
                # binding reads as a bug rather than as a missing declaration.
                raise BindRefusal(
                    f"the state tail cannot be encoded for this step: "
                    f"{exc}") from exc
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
                f"cudagraph_mode={self.cudagraph_mode}, "
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


def _reads_cache(shape: StepShape) -> bool:
    """Whether this batch is a prefill that reads KV it did not compute.

    `BatchSpec.has_cached`, computed from the shape: prefill, and some row
    whose context is longer than its query. It is in the template key and not
    in the bound fields because it is not a value a cohort can be given. The
    backend records three fields on the cached branch that it does not record
    otherwise -- `total_kv`, `seq_starts`, `num_cached_tokens` -- and binding
    rewrites values, it does not grow a context entry a trace never held. It
    also selects the attention kernel, so the two are not the same graph.
    """
    return bool(shape.num_prefill_tokens) and any(
        int(c) > int(q) for q, c in zip(shape.num_scheduled_tokens,
                                        shape.context_lens))


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
            shape.capture_bucket, shape.compiled, _reads_cache(shape))


def _cu_seqlens(queries):
    out, total = [0], 0
    for q in queries:
        total += q
        out.append(total)
    return out


def _padding_of(shape: StepShape) -> tuple[int, int]:
    """``(pad_rows, pad_tokens)``: how much of this step's buffers is padding.

    A replayed decode runs ``running_bs`` rows and ``running_bs * max_q_len``
    tokens while the batch has fewer of each, so every per-request and
    per-token buffer the kernels read has a tail. Both numbers are zero without
    a bucket, zero again for a batch that lands exactly on one, and zero for a
    prefill at any bucket -- `ForwardMode.decide` sends a batch holding a
    prefill token down the eager path (forward_context.py:196-204), so nothing
    is replayed and nothing is padded.

    ``template_key`` carries the bucket and the query-length vector, so a
    template and any cohort bound to it agree on these two numbers by
    construction: the padding is structure, and a cohort that changed it would
    key its own template.
    """
    bucket = shape.capture_bucket
    queries = [int(q) for q in shape.num_scheduled_tokens]
    if bucket is None or int(getattr(shape, "num_prefill_tokens", 0) or 0):
        return 0, 0
    if not queries:
        return 0, 0
    return (int(bucket) - len(queries),
            int(bucket) * max(queries) - sum(queries))


def _bind(key, template_value, rows, pad_rows: int = 0, pad_tokens: int = 0,
          extent_scope: str = "batch"):
    """One context entry, recomputed for ``rows``, or :class:`BindRefusal`.

    Every formula here is a property of the batch that the runner also computes
    from the batch. None reads the template's value except to keep a constant
    the cohort cannot change, or to learn a section count.

    ``pad_rows`` and ``pad_tokens`` are the replay's padding, and each field
    that has a tail gets its own -- the same tails
    ``BatchSpec.attention_context`` derives, since both are reproducing what
    the runner writes into the buffers a captured graph reads. Recomputing
    these at the batch's width instead is how a padded template comes back
    unpadded from a warm hit: the derivation is right and the binding narrows
    it again, one cohort later.
    """
    queries = [q for q, _ in rows]
    contexts = [c for _, c in rows]
    if key == "context_lens":
        # Zero for a padded row: it holds no history to walk
        # (aiter_attention.py:1115).
        return contexts + [0] * pad_rows
    if key == "positions":
        # A request's next position is the last index of its history; a
        # multi-token query runs to the end of its chunk. The tensor is M-RoPE,
        # so it is `position_rows` identical sections laid end to end -- 96
        # entries for a 32-token decode at three rows. The section count comes
        # from the template's own length rather than a constant, because the
        # template and the cohort share a query-length vector by construction
        # and so share the token count.
        #
        # Padded before the sections are counted, not after: the buffer is
        # [position_rows, num_tokens_pad], so the pad is inside each section.
        # Counting sections against the unpadded length would read a 3-request
        # decode in a bucket of 4 as four sections of three.
        per_token = []
        for q, c in rows:
            per_token.extend(range(c - q, c))
        per_token += [0] * pad_tokens
        tokens = len(per_token)
        if not tokens or len(template_value) % tokens:
            raise BindRefusal(
                f"positions has {len(template_value)} entries for {tokens} "
                "tokens; the section layout is not what this rule assumes")
        return per_token * (len(template_value) // tokens)
    if key == "max_seqlen_k":
        # Two rules, and which one applies is the deployment's to say.
        #
        # Eager and PIECEWISE: the real rows', not the padded ones'. A padded
        # row's context is zero, and the runner's own `max_seqlen_q` /
        # `max_seqlen_k` come off the scheduled batch (model_runner.py:3187,
        # aiter_attention.py:1100, :1139).
        #
        # FULL: the capture pinned the field to the engine's `max_model_len`
        # (aiter_attention.py:1367, :1331) and the replay runs the buffer the
        # capture holds, so the extent does not follow this cohort at all.
        # Recomputing it from the contexts is the same failure `pad_rows`
        # exists for one paragraph up: the derivation is right and the binding
        # narrows it again, one cohort later.
        if extent_scope == "captured":
            return template_value
        if extent_scope == "undeclared":
            raise BindRefusal(
                "this template replays a capture bucket but the deployment's "
                "--cudagraph-mode was not declared, so whether max_seqlen_k "
                "follows the batch or the capture's max_model_len is unknown; "
                "both answers bind without complaint and one of them is a "
                "graph the native run never had")
        return max(contexts) if contexts else 0
    if key == "max_seqlen_q":
        return max(queries) if queries else 0
    if key == "cu_seqlens_q":
        # The padded rows repeat the last real offset, which is what makes each
        # of them an empty sequence for attention rather than a fifth request
        # (model_runner.py:2649-2652): three in a bucket of four is
        # [0, 1, 2, 3, 3].
        out = _cu_seqlens(queries)
        return out + [out[-1]] * pad_rows
    if key == "cu_seqlens_k":
        # Cumulative key lengths, prefill only. The keys are the whole context,
        # cached prefix included: `prepare_prefill` accumulates
        # `seqlen_k = context_lens[i]`
        # (atom/model_ops/attentions/backends.py:449), and
        # `BatchSpec.attention_context` records that same sum. Decode leaves it
        # None and so does binding -- it does not invent an empty tensor.
        if template_value is None:
            return None
        if len(template_value) != len(rows) + 1:
            raise BindRefusal(
                f"cu_seqlens_k has {len(template_value)} entries and this "
                f"batch has {len(rows)} requests; it is one offset per prefill "
                "row after a leading zero, and a batch whose prefill rows are "
                "not all of it is a structure this function has not been shown")
        return _cu_seqlens(contexts)
    if key == "non_spec_query_start_loc":
        # The linear-attention layers take the same cumulative query offsets
        # over the non-speculative requests, recorded as a ``[values, dtype]``
        # pair. Speculative decoding is off in this deployment, so every
        # request is non-spec; a template whose non-spec count differs from its
        # batch is a structure this function has not been shown.
        #
        # Unlike `cu_seqlens_q`, this one is padded only under FULL. That is
        # not a choice made here: `_build_gdn_capture_metadata` bakes the
        # bucket's counts into the graph and the replay refills the tail of the
        # buffer with the last real offset (gdn_attn.py:1189-1235, :1264-1281),
        # while PIECEWISE leaves this metadata eager and rebuilds it from the
        # active batch every step -- `A + 1` offsets, no tail
        # (model_runner.py:4019-4031). `BatchSpec.gdn_context` derives it under
        # exactly that split, so binding it under the other one refuses a
        # correctly derived PIECEWISE template. ``extent_scope`` is the same
        # declared mode, resolved once in `batch_spec.extent_scope_of`.
        values, dtype = template_value
        if extent_scope == "undeclared":
            raise BindRefusal(
                "this template replays a capture bucket but the deployment's "
                "--cudagraph-mode was not declared, so whether "
                "non_spec_query_start_loc carries the bucket's padded offsets "
                "or the active batch's is unknown; the two differ by exactly "
                "the padding and both bind without complaint")
        pad = pad_rows if extent_scope == "captured" else 0
        if len(values) != len(queries) + 1 + pad:
            raise BindRefusal(
                f"non_spec_query_start_loc has {len(values)} offsets and this "
                f"batch has {len(queries)} requests with {pad} padded rows "
                f"under the {extent_scope!r} rule; it is one offset per row "
                "of the whole batch after a leading zero. Speculative decoding "
                "changes the structure, not the cohort")
        out = _cu_seqlens(queries)
        return [out + [out[-1]] * pad, dtype]
    if key == "has_initial_state":
        # Which prefill rows continue a sequence whose convolution state is
        # already in the pool. Cohort, not structure: `template_key` drops the
        # context lengths, so one template serves a first chunk, a continuation
        # and a batch of both at once, and those differ row by row -- a row
        # with an incoming state reads it and a row without does not.
        #
        # The formula is the recording rule's, `batch_spec.py::gdn_context`:
        # per prefill row, `num_cached_tokens > 0`, which is context minus
        # query. Decode records None, as the backend does, so a template
        # carrying None binds to None rather than to an empty tensor.
        if template_value is None:
            return None
        values, dtype = template_value
        if len(values) != len(rows):
            raise BindRefusal(
                f"has_initial_state has {len(values)} entries and this batch "
                f"has {len(rows)} requests; the field is recorded over the "
                "prefill rows, and a batch whose prefill rows are not all of "
                "it is a structure this function has not been shown")
        return [[1 if c - q > 0 else 0 for q, c in rows], dtype]
    if key in ("total_kv", "seq_starts", "num_cached_tokens"):
        # The cached-prefix branch of `BatchSpec.attention_context`, which
        # `template_key` now separates, so a template carrying these is bound
        # only to a cohort that also reads a cached prefix. Each is the
        # recording rule verbatim: the total keys attention walks, the start of
        # each request's cached prefix, and how many tokens of it were already
        # there -- `cached_lens`, which is context minus query.
        cached = [c - q for q, c in rows]
        if key == "total_kv":
            return sum(contexts)
        if key == "num_cached_tokens":
            return cached
        starts, run = [], 0
        for n in cached:
            starts.append(run)
            run += n
        if len(template_value) != len(rows):
            raise BindRefusal(
                f"seq_starts has {len(template_value)} entries and this batch "
                f"has {len(rows)} requests; it is one start per prefill row, "
                "and a batch whose prefill rows are not all of it is a "
                "structure this function has not been shown")
        return starts
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


def _check_traced_as_captured(template: dict) -> None:
    """Refuse a template that was not itself traced as a FULL capture.

    Under ``"captured"`` the binder keeps the template's ``max_seqlen_k``
    rather than recomputing it, so the value is only right if the template was
    derived under the mode being replayed. A template read off disk was traced
    by some other run: `seeded_graphs` keys it by structure, and structure does
    not include the mode. So a PIECEWISE-traced graph served to a FULL
    deployment would be kept verbatim and would carry that run's longest
    history as a captured extent -- a wrong number with a right-looking
    provenance, which is worse than a refusal.

    The declared mode is necessary and not sufficient. A seed can say ``full``
    and still carry the extent the *eager* rule produced, because that is
    exactly what every graph derived before this rule was fixed does. So the
    seed is also checked against itself: its provenance records the whole
    `BatchSpec`, `BatchSpec.launch_max_seqlen_k` is the one rule that says what
    the extent should be under the mode that spec declares, and a recorded
    value that disagrees with it is a stale derivation whatever its label
    reads. Refusing is the only safe answer -- under ``"captured"`` this value
    is kept verbatim, so a wrong one is preserved rather than corrected, and
    `max_seqlen_k` is part of the operator identity key.
    """
    spec = ((template.get("provenance") or {}).get("batch_spec") or {})
    traced = spec.get("cudagraph_mode")
    if traced is None:
        # A seed written before the derivation recorded its mode. Silence is
        # not a FULL trace: the extent this binder is about to keep verbatim
        # is exactly the field the eager rule got wrong, so an unqualified
        # seed carrying an eager `max_seqlen_k` would be preserved as a
        # captured one. Re-derive it under the mode being priced -- the
        # derivation path stamps `provenance.batch_spec.cudagraph_mode` -- or
        # bind under PIECEWISE/eager, where the value is recomputed anyway.
        raise BindRefusal(
            "this template's provenance does not say which cudagraph_mode it "
            "was derived under, and it is being bound for a FULL replay, "
            "which keeps the template's own max_seqlen_k rather than "
            "recomputing it. An unqualified seed is not evidence of a FULL "
            "trace. Re-derive it through the source derivation path, which "
            "records the mode.")
    if str(traced).strip().lower() != "full":
        raise BindRefusal(
            f"this template was traced under cudagraph_mode {traced!r} and is "
            "being bound for a FULL replay, which keeps the template's own "
            "max_seqlen_k. That value is the traced run's longest history, "
            "not this deployment's max_model_len. Derive the template under "
            "the mode being priced.")
    from atom.compass.runtime.batch_spec import BatchSpec

    try:
        expected = BatchSpec.from_dict(spec).launch_max_seqlen_k
    except Exception as exc:
        raise BindRefusal(
            "this template's provenance carries a batch_spec that will not "
            f"rebuild ({exc}), so the extent it records cannot be checked "
            "against the mode it declares") from exc
    for op in template.get("ops") or ():
        for entry in op.get("context") or ():
            key, value = tuple(entry)
            if key != "max_seqlen_k" or value == expected:
                continue
            raise BindRefusal(
                f"{op.get('name')} records max_seqlen_k {value} and the "
                f"batch_spec in its own provenance -- cudagraph_mode "
                f"{traced!r}, max_model_len {spec.get('max_model_len')}, "
                f"capture_bucket {spec.get('capture_bucket')} -- makes it "
                f"{expected}. The seed is labelled FULL and carries the eager "
                "rule's value, which this bind would keep verbatim. "
                "Re-derive it through the source derivation path.")


def bind_cohort(template: dict, shape: StepShape,
                allocation: Optional[AllocationSource] = None,
                extent_scope: str = "batch") -> dict:
    """A copy of ``template`` whose per-request metadata describes ``shape``.

    ``allocation`` is required whenever the template carries allocator fields.
    Refuses rather than guesses, in both directions: an operator with a context
    field this module has no rule for raises, and so does a missing allocation.

    ``extent_scope`` is which rule the attention launch extent follows --
    ``"batch"``, ``"captured"`` or ``"undeclared"``, the three
    :attr:`BatchSpec.launch_extent_scope` gives. It defaults to ``"batch"``,
    which is every eager and PIECEWISE step, and a caller replaying a FULL
    capture has to say so: a captured extent is a constant of the graph and
    rebinding it to the cohort produces a graph the native run never had.
    """
    if extent_scope not in ("batch", "captured", "undeclared"):
        raise BindRefusal(
            f"extent_scope {extent_scope!r} is not one of 'batch', "
            "'captured', 'undeclared'")
    if extent_scope == "captured":
        _check_traced_as_captured(template)
    rows = _rows(shape)
    pad_rows, pad_tokens = _padding_of(shape)
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
                bound = _bind(key, value, rows, pad_rows, pad_tokens,
                              extent_scope)
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
        # The replay's padding, which is a different thing from the line above.
        # That one is buffer entries the allocation did not reach and the
        # template's own values were kept for; these are rows the captured
        # graph really forwards, derived rather than carried.
        "replay_pad_rows": pad_rows,
        "replay_pad_tokens": pad_tokens,
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

    ``cudagraph_mode`` is the deployment's declared mode, and it is held here
    rather than read off the shape because it is a property of the deployment
    and not of a step: `StepShape` says which bucket ran, never which mode
    captured it. Every bind goes through :func:`extent_scope_of` with it, so a
    warm reuse and the derivation that filled the cache resolve one step's
    launch extent the same way. Left ``None``, a bucketed step binds to
    ``"undeclared"`` and is refused -- which is the intended answer, not a
    gap: the deriver takes the same mode, and a composition that declares it
    to one and not the other would bind FULL graphs against the batch rule.
    """

    def __init__(self, templates=None, derive=None, allocation=None,
                 representative_rank: int = 0, cudagraph_mode=None) -> None:
        self._templates = dict(templates or {})
        self._derive = derive
        self._allocation = allocation
        self._representative = int(representative_rank)
        self._cudagraph_mode = cudagraph_mode
        self.hits = 0
        self.binds = 0
        self.derivations = 0
        #: Wall seconds spent in the deriver, including the misses that ended
        #: in a refusal -- a refused derivation still cost the time it took.
        self.derivation_seconds = 0.0
        self.representative_hits = 0
        self.refusals = {}
        # A hit leaves no interval in the journal, so without this a run's
        # derivation rows cannot be read as a rate. Off unless the journal is.
        from atom.compass.runtime import derivation_log

        derivation_log.watch(self, "template_graphs")

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
            # Timed, not only counted. Whether these seconds are already
            # inside the served window depends on *when* the miss happened,
            # and nothing in this process knows when the harness decided
            # startup ended -- so the interval is recorded on the wall clock,
            # the one cc_traces_run.py stamps its own windows on, and the cost
            # record places it by intersection instead of asserting a phase.
            # Off unless ATOM_COMPASS_DERIVATION_LOG names a file.
            from atom.compass.runtime import derivation_log

            began = derivation_log.now()
            try:
                template = self._derive(shape)
            except (BindRefusal, ValueError) as exc:
                # `BatchSpec.gdn_context` raises where a bucketed decode has no
                # declared mode: the FULL and PIECEWISE shapes differ by
                # exactly the padding and neither is right under both. That is
                # the same answer as a bind refusal and belongs in the same
                # place -- a served step asks this cache a question and gets
                # None with a reason, rather than an exception out of the
                # provider for one operator family and a refusal for another.
                self.refusals[key] = str(exc)
                return None
            ended = derivation_log.now()
            self.derivation_seconds += ended - began
            if template is None:
                self.refusals[key] = "deriver produced nothing"
                return None
            derivation_log.record(began, ended, key=str(key), on_demand=True)
            self.derivations += 1
            self._templates[key] = template
        else:
            self.hits += 1
        # The same scope on the cold return and every warm one after it. A
        # derivation builds its graph from a `BatchSpec` that already knows
        # the mode, so binding that graph back under the default batch rule
        # would overwrite a captured `max_seqlen_k` the moment it was
        # produced -- and then again on every hit.
        scope = extent_scope_of(shape.capture_bucket,
                                shape.num_prefill_tokens > 0,
                                self._cudagraph_mode)
        try:
            bound = bind_cohort(template, shape, self._allocation, scope)
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
        mode = (f"cudagraph_mode {self._cudagraph_mode}"
                if self._cudagraph_mode else "no cudagraph_mode declared")
        return (f"TemplateGraphs({len(self._templates)} templates, "
                f"{self.hits} hits{stood_in}, {self.derivations} derivations "
                f"in {self.derivation_seconds:.3f}s, "
                f"{self.binds} binds, {len(self.refusals)} refused; "
                f"{mode}; {where})")
