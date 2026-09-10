"""The batch a derivation is deriving, stated rather than inferred.

A derived graph is a graph of *some* forward, and until now the trace said which
one only by a token count. Four tokens through the model body is not four decode
requests at context 66 -- the Q/K/V shapes agree, and nothing else does. Decode
reads 66 tokens of KV per sequence and prefill reads none; a chunked prefill
reads a gathered prefix; the kernels differ, the traffic differs by two orders of
magnitude, and the operator signature that prices them is the same. So the batch
is an input to derivation, written down, and the metadata attention reads is
computed from it.

Computed, not copied. The values here are the ones
``aiter_attention.prepare_decode`` and ``CommonAttentionBuilder.prepare_prefill``
compute from a ``ScheduledBatch``, reproduced from the same quantities a
scheduler would have: per-request query and context lengths, the block size, the
capture bucket. Nothing is read off a meta tensor -- there is nothing in one to
read -- and nothing is borrowed from a capture of another configuration. What
cannot be derived is declared: the block allocation is a policy, named in the
spec and reproducible, because which block ids a request holds is the block
manager's history and not a property of the batch.

The recipe is validated by replaying it against a real capture:
``tests/compass/test_batch_spec.py`` rebuilds every field the tracer recorded
from the shape provenance that capture also wrote, and compares.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any, Optional

__all__ = ["BatchSpec", "allocate_blocks"]


def allocate_blocks(prompt_lens, context_lens, block_size: int,
                    policy: str = "rounds") -> list[list[int]]:
    """Which KV blocks each request holds, under a named policy.

    Block identity is not a property of the batch. It is what the block manager
    happened to hand out, and two runs of the same workload can differ. It still
    reaches cost -- through which pages the attention kernel walks and how they
    sit in cache -- so it cannot simply be invented per request as 0..n.

    ``rounds`` is the policy a first-come scheduler produces and the one the 27B
    capture shows: every request's prompt blocks are allocated in request order,
    then one growth block per request per round, again in request order. The
    captured 4x66 decode holds exactly
    ``[[0,1,2,3,16],[4,5,6,7,17],[8,9,10,11,18],[12,13,14,15,19]]`` -- four
    prompt blocks each from a 64-token prompt, then a fifth for the tokens past
    64, in a second round.

    ``packed`` is the degenerate alternative: every request's blocks contiguous.
    It is offered because a long single-sequence batch has no rounds to speak of
    and packing is then both true and simpler, not because it is interchangeable.
    """
    prompt_lens = list(prompt_lens)
    context_lens = list(context_lens)
    if len(prompt_lens) != len(context_lens):
        raise ValueError("one prompt length per request")
    need = [max(1, -(-c // block_size)) for c in context_lens]
    have = [max(0, -(-p // block_size)) for p in prompt_lens]
    if any(h > n for h, n in zip(have, need)):
        raise ValueError("a prompt cannot occupy more blocks than the context")

    tables: list[list[int]] = [[] for _ in need]
    nxt = 0
    if policy == "packed":
        for i, n in enumerate(need):
            tables[i] = list(range(nxt, nxt + n))
            nxt += n
        return tables
    if policy != "rounds":
        raise ValueError(f"unknown block allocation policy {policy!r}")
    # Round zero: the prompts, in request order.
    for i, n in enumerate(have):
        tables[i] = list(range(nxt, nxt + n))
        nxt += n
    # Then one growth block per request per round, until every request has what
    # its context needs. A request that finished growing is skipped, which is
    # what leaves the ids of a longer request non-contiguous.
    while any(len(t) < n for t, n in zip(tables, need)):
        for i, n in enumerate(need):
            if len(tables[i]) < n:
                tables[i].append(nxt)
                nxt += 1
    return tables


@dataclass(frozen=True)
class BatchSpec:
    """One forward, as a scheduler would have described it.

    ``query_lens`` is how many tokens of each request this step computes and
    ``context_lens`` how many the KV cache holds for it *including* those -- the
    two names ATOM's own batch uses. A decode step is ``query_lens=[1,...]``; a
    native prefill is ``query_lens == context_lens``; a chunked prefill is
    neither, and sets ``has_cached``.

    ``capture_bucket`` is the padded batch size a CUDA-graph replay would run,
    or None for eager. It is recorded because it changes the size of every
    metadata buffer and therefore the work, not because the model reads it.
    """

    kind: str                      # "decode" | "prefill"
    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    block_size: int
    max_model_len: int
    capture_bucket: Optional[int] = None
    num_spec_step: int = 0
    block_policy: str = "rounds"
    #: How long each request was at admission. Only the block allocation reads
    #: it, and only to know where the first growth round starts.
    prompt_lens: Optional[tuple[int, ...]] = None
    #: Explicit block tables, when the batch is being replayed from a capture
    #: that recorded them rather than described by a policy.
    block_tables: Optional[tuple[tuple[int, ...], ...]] = None
    #: MRoPE models lay positions out as [3, N]. Recorded because the tracer
    #: reads the flattened tensor and its length is otherwise unexplainable.
    position_rows: int = 1
    notes: dict = field(default_factory=dict)

    # -- reading one off disk ------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict) -> "BatchSpec":
        """Build from JSON, strictly.

        JSON has lists where this has tuples, and no way to say "frozen". It
        also has no way to say "you misspelled a field", which is why an
        unknown key is an error rather than a default quietly winning: a spec
        with ``prompt_len`` instead of ``prompt_lens`` would otherwise derive a
        different batch than the one written down, and say nothing.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown batch spec field(s) {sorted(unknown)}; "
                f"known fields are {sorted(known)}")
        kw = dict(raw)
        for name in ("query_lens", "context_lens", "prompt_lens"):
            if kw.get(name) is not None:
                kw[name] = tuple(int(v) for v in kw[name])
        if kw.get("block_tables") is not None:
            kw["block_tables"] = tuple(tuple(int(b) for b in row)
                                       for row in kw["block_tables"])
        spec = cls(**kw)
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: str) -> "BatchSpec":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict:
        """The spec as JSON, with the derived block table made explicit.

        What goes in a graph's provenance is what was actually derived, so the
        table is written out even when a policy produced it: the policy is
        reproducible, but only if its output is there to check against.
        """
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["block_tables"] = self.tables()
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in out.items() if v is not None and v != {}}

    # -- derived quantities, all of them pure -------------------------------

    @property
    def batch_size(self) -> int:
        return len(self.query_lens)

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)

    @property
    def has_cached(self) -> bool:
        """A prefill that reads KV it did not compute this step: chunked."""
        return (self.kind == "prefill"
                and any(c > q for q, c in zip(self.query_lens,
                                              self.context_lens)))

    @property
    def cached_lens(self) -> tuple[int, ...]:
        """How many tokens of each request the KV cache already held.

        Not the prompt. A request admitted with 64 tokens that has generated one
        has a 64-token prompt and 65 cached tokens, and the two are read by
        different things: chunked prefill's `num_cached_tokens` means this one,
        the block allocation below means the other.
        """
        return tuple(c - q for q, c in zip(self.query_lens, self.context_lens))

    @property
    def admitted_lens(self) -> tuple[int, ...]:
        """How long each request was when the scheduler admitted it.

        The block manager allocates a prompt's blocks in one go and then grows
        one block at a time, so where the first growth round falls is a fact
        about the prompt and is not recoverable from the context. Defaulted to
        the cached length, which is right for a native prefill and for any
        request that has not grown past its prompt.
        """
        return self.prompt_lens or self.cached_lens

    def tables(self) -> list[list[int]]:
        if self.block_tables is not None:
            return [list(t) for t in self.block_tables]
        return allocate_blocks(self.admitted_lens, self.context_lens,
                               self.block_size, self.block_policy)

    def validate(self) -> None:
        """Everything that can be checked without a device, checked.

        The failure this prevents is a spec that prices happily and describes a
        step no engine could run: a context longer than the model's, a query
        longer than its context, a block table too narrow for the tokens it must
        address. Each of those produces a plausible number.
        """
        if self.kind not in ("decode", "prefill"):
            raise ValueError(f"kind must be decode or prefill, not {self.kind!r}")
        if not self.query_lens:
            raise ValueError("a batch has at least one request")
        if len(self.query_lens) != len(self.context_lens):
            raise ValueError("one context length per request")
        if self.block_size <= 0:
            raise ValueError("block size must be positive")
        for i, (q, c) in enumerate(zip(self.query_lens, self.context_lens)):
            if q <= 0:
                raise ValueError(f"request {i} computes no token")
            if q > c:
                raise ValueError(
                    f"request {i} computes {q} tokens into a context of {c}")
            if c > self.max_model_len:
                raise ValueError(
                    f"request {i} context {c} exceeds max_model_len "
                    f"{self.max_model_len}")
        if self.kind == "decode" and set(self.query_lens) != {
                self.num_spec_step + 1}:
            raise ValueError(
                "a decode step computes num_spec_step+1 tokens per request; "
                f"got {sorted(set(self.query_lens))}")
        width = self.max_model_len // self.block_size
        for i, table in enumerate(self.tables()):
            need = -(-self.context_lens[i] // self.block_size)
            if len(table) < need:
                raise ValueError(
                    f"request {i} needs {need} blocks for {self.context_lens[i]}"
                    f" tokens and its table holds {len(table)}")
            if len(table) > width:
                raise ValueError(
                    f"request {i} holds {len(table)} blocks and a table row is "
                    f"{width} wide at max_model_len {self.max_model_len}")
        if self.capture_bucket is not None and self.capture_bucket < self.batch_size:
            raise ValueError(
                f"capture bucket {self.capture_bucket} is narrower than the "
                f"batch of {self.batch_size}")

    # -- the metadata the tracer records ------------------------------------

    def attention_context(self) -> tuple[tuple[str, Any], ...]:
        """What ``forward_ctx._capture_attention`` would record for this batch.

        Field for field, in the order that function writes them, computed the
        way the backend computes them:

        * ``slot_mapping`` -- decode takes the last block and the count of
          tokens in it (`aiter_attention.prepare_decode`, the ``max_seqlen_q==1``
          branch); prefill walks every block from the cached prefix to the end
          (`CommonAttentionBuilder.prepare_prefill`).
        * ``cu_seqlens_k`` -- prefill only. Decode leaves it unset and the
          recorded value is None.
        * ``state`` -- ``prefill_prefix`` under a cached prefix, else
          ``prefill_native``. Decode does not set it and inherits the same
          default, which is why a decode capture reads ``prefill_native``.
        * ``positions`` -- the last ``query_lens[i]`` positions of each request,
          tiled over ``position_rows`` for MRoPE.
        """
        self.validate()
        tables = self.tables()
        block = self.block_size
        cu_q = [0]
        for q in self.query_lens:
            cu_q.append(cu_q[-1] + q)

        slots: list[int] = []
        for i, (q, c) in enumerate(zip(self.query_lens, self.context_lens)):
            for pos in range(c - q, c):
                slots.append(tables[i][pos // block] * block + pos % block)

        positions: list[int] = []
        for q, c in zip(self.query_lens, self.context_lens):
            positions.extend(range(c - q, c))
        positions = positions * self.position_rows

        recorded: list[tuple[str, Any]] = [
            ("context_lens", list(self.context_lens)),
            ("slot_mapping", slots),
            ("cu_seqlens_q", cu_q),
        ]
        if self.kind == "prefill":
            cu_k = [0]
            for c in self.context_lens:
                cu_k.append(cu_k[-1] + c)
            recorded.append(("cu_seqlens_k", cu_k))
        else:
            recorded.append(("cu_seqlens_k", None))
        recorded += [
            ("max_seqlen_q", max(self.query_lens)),
            ("max_seqlen_k", max(self.context_lens)),
            ("min_seqlen_q", 0),
            ("has_cached", self.has_cached),
            ("state", "prefill_prefix" if self.has_cached else "prefill_native"),
            ("is_prefill", self.kind == "prefill"),
            ("positions", positions),
        ]
        if self.has_cached:
            recorded += [
                ("total_kv", sum(self.context_lens)),
                ("seq_starts", list(_prefix_starts(self.cached_lens))),
                ("num_cached_tokens", list(self.cached_lens)),
            ]

        width = self.max_model_len // self.block_size
        used = min(width, max(1, -(-max(self.context_lens) // self.block_size)))
        flat: list[int] = []
        for table in tables:
            row = list(table[:used])
            row += [0] * (used - len(row))
            flat.extend(row)
        recorded += [
            ("block_tables_shape", [self.batch_size, width]),
            ("block_tables", flat),
        ]
        return tuple(recorded)

    def gdn_context(self, state_slots=None) -> tuple[tuple[str, Any], ...]:
        """What ``_capture_linear_attention`` would record for this batch.

        DeltaNet counts prefills and decodes separately and indexes a
        per-request recurrent state pool rather than a paged KV cache, so its
        metadata is start offsets and state indices, not block tables. The
        recurrent and convolution state itself is not here for the same reason
        it is not in a capture: it lives in ``kv_cache_data``, which the engine
        installs at start-up and which is already real wherever pricing runs.

        ``state_slots`` is which per-request state entry each request occupies;
        the default is the batch order, which is what a fresh pool hands out.
        """
        self.validate()
        n = self.batch_size
        slots = list(state_slots if state_slots is not None else range(n))
        if len(slots) != n:
            raise ValueError("one state slot per request")
        starts = [0]
        for q in self.query_lens:
            starts.append(starts[-1] + q)

        prefill = self.kind == "prefill"
        recorded: list[tuple[str, Any]] = [
            ("num_prefills", n if prefill else 0),
            ("num_prefill_tokens", self.num_tokens if prefill else 0),
            ("num_decodes", 0 if prefill else n),
            ("num_decode_tokens", 0 if prefill else self.num_tokens),
            ("num_spec_decodes", 0),
            ("num_spec_decode_tokens", 0),
            ("num_actual_tokens", self.num_tokens),
            ("replayssm", False),
            ("non_spec_query_start_loc", [starts, "int32"]),
            ("non_spec_state_indices_tensor", [slots, "int32"]),
            ("non_spec_state_indices_in_tensor", [slots, "int32"]),
        ]
        return tuple(recorded)


def _prefix_starts(lengths):
    total = 0
    for n in lengths:
        yield total
        total += n



def install(spec: "BatchSpec", device: str = "cpu"):
    """Put the batch's metadata on the forward context, for a derivation.

    Derivation traces a bare ``model(input_ids, positions)`` with no engine
    around it, so nothing installs a forward context and attention records none
    -- which is why 64 attention operators per step arrived at the price list
    marked "reads a forward context and the graph recorded none". This installs
    one, built from the spec.

    The tensors are real and live on ``device``, not on meta. That is the point:
    a meta tensor holds no values, so a context made of them would record
    nothing, and the recorder would be reading its own emptiness back. The
    metadata is small -- a few hundred integers -- and computing it on the host
    costs nothing next to the model it describes.

    It goes through ATOM's own ``AttentionMetaData`` and ``Context``, so a field
    this recipe does not set takes the engine's default rather than one invented
    here, and ``forward_ctx.capture`` reads it back exactly as it reads a live
    step's.
    """
    import contextlib

    import torch

    from atom.config import get_current_atom_config
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata
    from atom.utils.forward_context import (
        AttentionMetaData, AttnState, Context, set_forward_context)

    fields = dict(spec.attention_context())
    gdn_fields = dict(spec.gdn_context())

    def tensor(values, dtype):
        if values is None:
            return None
        return torch.tensor(values, dtype=dtype, device=device)

    shape = fields["block_tables_shape"]
    table = torch.zeros(tuple(shape), dtype=torch.int32, device=device)
    used = len(fields["block_tables"]) // max(shape[0], 1)
    if used:
        table[:, :used] = torch.tensor(
            fields["block_tables"], dtype=torch.int32,
            device=device).reshape(shape[0], used)

    metadata = AttentionMetaData(
        block_tables=table,
        context_lens=tensor(fields["context_lens"], torch.int32),
        slot_mapping=tensor(fields["slot_mapping"], torch.int64),
        cu_seqlens_q=tensor(fields["cu_seqlens_q"], torch.int32),
        cu_seqlens_k=tensor(fields["cu_seqlens_k"], torch.int32),
        max_seqlen_q=fields["max_seqlen_q"],
        max_seqlen_k=fields["max_seqlen_k"],
        min_seqlen_q=fields["min_seqlen_q"],
        has_cached=fields["has_cached"],
        state=AttnState(fields["state"]),
        total_kv=fields.get("total_kv"),
        seq_starts=tensor(fields.get("seq_starts"), torch.int32),
        num_cached_tokens=tensor(fields.get("num_cached_tokens"), torch.int32),
    )

    gdn = GDNAttentionMetadata(**{
        name: (value if not isinstance(value, list)
               else tensor(value[0], getattr(torch, value[1])))
        for name, value in gdn_fields.items() if name != "replayssm"})
    # The convolution reads three more fields off the same object, and they are
    # a pure function of the query start offsets -- recomputed with the engine's
    # own helper rather than recorded, exactly as the pricing installer does.
    from atom.model_ops.attentions.gdn_attn import compute_causal_conv1d_metadata

    (gdn.nums_dict, gdn.batch_ptr,
     gdn.token_chunk_offset_ptr) = compute_causal_conv1d_metadata(
        gdn.non_spec_query_start_loc)
    metadata.gdn_metadata = gdn

    context = Context(
        positions=tensor(fields["positions"], torch.int64),
        is_prefill=fields["is_prefill"],
    )

    @contextlib.contextmanager
    def installed():
        set_forward_context(attn_metadata=metadata,
                            atom_config=get_current_atom_config(),
                            context=context)
        try:
            yield metadata
        finally:
            from atom.utils.forward_context import reset_forward_context

            reset_forward_context()

    return installed()


def model_inputs(spec: "BatchSpec", device="meta"):
    """The token and position tensors the runner would hand the model.

    ``derived_inputs`` produced ``arange(tokens)`` -- one sequence starting at
    position zero -- which is a prefill of ``tokens`` tokens and nothing else.
    A decode batch's positions are the last token of each request, and the two
    graphs differ from the first RoPE onward. Dtypes as ``derived_inputs``
    documents them: ``int32`` ids, ``int64`` positions.

    Under MRoPE the runner hands the model a ``[3, N]`` view of its position
    buffer, not the flat one (`model_runner._mrope_positions_view`), and each
    of the three rows holds the same values for text-only requests. The shape
    reaches the graph -- RoPE indexes it -- so it is reproduced here rather
    than left flat.
    """
    import torch

    positions = []
    for q, c in zip(spec.query_lens, spec.context_lens):
        positions.extend(range(c - q, c))
    pos = torch.tensor(positions * spec.position_rows, dtype=torch.int64,
                       device=device)
    if spec.position_rows > 1:
        pos = pos.view(spec.position_rows, spec.num_tokens)
    return (torch.zeros(spec.num_tokens, dtype=torch.int32, device=device),
            pos)
