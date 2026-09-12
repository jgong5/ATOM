"""A price for an attention call at a (query, history) structure nobody ran.

The row families move in one component -- token rows -- so one curve and an
interpolation between measured row counts answers them. The attention families
do not. `unified_attention_with_output_base` at fixed operand shapes differs
13x on history alone, so its price is set by the joint structure of the batch:
which rows are new queries, how much history each of them reads, and which
native branch that combination takes. There is no single axis to interpolate
along, which is why `FamilyPriceLibrary` refuses these outright today.

This module is the model that replaces that refusal -- and only where the
refusal was a gap rather than a finding. Exact lookup still comes first; a
modelled price is what an *honest miss* falls through to.

What it is not
--------------
It does not fit whole-step timings, and it does not reuse the calibrated
oracle's paired coefficients: those were fitted against the target engine's
own step durations, which is the thing a source-only prediction may not touch.
Every number here comes from a source primitive measurement of one operator.

It also does not interpolate blindly. A structure whose regime has no
identifiable support is refused by name, and the refusal says which points
would close it.

Regimes
-------
A regime is a native branch, not a convenience grouping. `attention_mha.py`
dispatches prefill and decode through different kernels, and the cached-prefix
path reads KV that the cold path does not, so their costs are different
functions of the same structure rather than one function with a parameter.

  ``unified.prefill.cold``    new queries only, no cached prefix to read
  ``unified.prefill.cached``  queries over a prefix already in the KV cache
  ``unified.decode.paged_gluon``   one query row per sequence, gluon paged
                                   decode tiling each context separately
  ``unified.decode.unified_attn``  the same batch through aiter unified
                                   attention, which does not tile that way
  ``gdn.prefill``             chunked scan over fresh sequences
  ``gdn.decode``              conv update plus recurrence over fixed state

Observations from two regimes are never pooled, and a structure is priced only
from its own regime's fit.

Scope
-----
A per-layer wrapper is not one kernel. `_dispatch_decode` alone turns on
`sliding_window`, on `ATOM_USE_UNIFIED_ATTN` and `ATOM_FORCE_ATTN_TRITON`, on
`kv_cache_block_size`, and on the flash-versus-shuffle layout -- and none of
that is recoverable from the operand dtypes, because a BF16 query says nothing
about whether the cached KV it reads is FP8.

The two families do not turn on the same facts, so they do not carry the same
required scope. Unified attention reads the paged KV cache and its dtype,
layout, window, block size and resolved backend all select the kernel. GDN
reads no KV cache at all -- its state is the conv and recurrent pool the engine
stood up -- so a KV dtype would be a scope key that means nothing here, and
requiring it would refuse honest observations for a fact that does not apply.
What does apply to GDN is the fixed-state geometry and whether the lossy fast
decode path was enabled.

A fit is therefore identified by its regime **and** its scope, never by regime
alone, and a request is priced only from a fit whose scope its own declared
scope matches key for key. Under `strict` -- the default, and what an
acceptance run gets -- a fit whose observations do not declare the required
keys is refused rather than assumed. `strict=False` permits an undeclared fit
for diagnostic use and stamps `scope_undeclared` on the result, so a number
produced that way can never be mistaken for one that was scoped.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["REQUIRED_SCOPE", "UNIFIED_SCOPE", "GDN_SCOPE", "UNIFIED", "GDN",
           "Structure", "Regime", "REGIMES", "DECODE_KERNELS",
           "UNPROVEN_DECODE_KERNELS",
           "Refusal", "Fit", "Model", "structure_of", "regime_of",
           "features_for", "fit_regime", "CHUNK_SIZE", "scope_key",
           "geometry_of", "scoped", "TOKEN_AXIS"]

UNIFIED = "aiter::unified_attention_with_output_base"
GDN = "aiter::linear_attention_with_output_base"

#: The native chunk width the GDN scan is written against. Declared here so a
#: chunk-count feature can be stated; it is not a fitted quantity, and a
#: deployment that changes it invalidates the chunk term rather than rescaling
#: it.
CHUNK_SIZE = 64

#: What a padded state lane holds, which is no lane. ``gdn_attn.py``:1189-1225
#: fills ``non_spec_state_indices_tensor[num_decodes:]`` with it at replay,
#: and the convolution skips those entries rather than touching slot 0's
#: checkpoint. Repeated here rather than imported from
#: `atom.compass.runtime.batch_spec`, which defines the same constant for the
#: same reason: this module reads recorded keys and must not depend on the
#: runtime that produces them.
PAD_SLOT_ID = -1

#: Static deployment facts that decide which kernel a *unified attention* call
#: takes. Two observations that disagree on any of these are measurements of
#: different work, and one that declares none of them cannot be shown to be in
#: any regime at all.
#:
#: `kv_cache_dtype` is resolvable today from the collector's own record of the
#: pool it stood up. The rest are not: an allocation geometry proves the
#: storage, not the view the backend takes over it, so layout, sliding window,
#: block size and backend selection still have to be declared by whoever
#: resolves them. `kv_cache_block_size` is here because `_dispatch_decode`
#: reads it directly -- at 256 under unified attention it takes the persistent
#: ASM kernel and otherwise Triton.
UNIFIED_SCOPE = ("kv_cache_dtype", "kv_cache_layout", "kv_cache_block_size",
                 "sliding_window", "attention_backend")

#: What the gluon paged decode branch needs on top of the unified facts.
#:
#: Its split count is not a constant and not a property of the batch alone:
#: `attention_mha.py`:552 asks `get_recommended_splits(num_seqs,
#: num_kv_heads)`, which is
#: ``min(8, ceil(compute_units * 2 / (num_seqs * num_kv_heads)))``
#: (`pa_decode_gluon.py`:111-118). The sequence count is in the key; the head
#: count and the device's CU count are not, so they are declared. Two
#: measurements taken on parts with different CU counts are measurements of
#: different grids, which is the other reason this belongs in the scope.
PAGED_GLUON_SCOPE = UNIFIED_SCOPE + ("num_kv_heads", "compute_units")

#: The static facts a *linear attention* call turns on. No KV cache appears
#: here: GDN reads the conv and recurrent state pool, never the paged KV cache,
#: so a KV dtype or layout is not a fact about this kernel and requiring one
#: would refuse honest observations over something that does not apply.
#:
#: `gdn_decode_lossy_fast` is `ATOM_ENABLE_GDN_DECODE_LOSSY_FAST` as the
#: measured process resolved it; the guarded branch in `attention_gdn.py` is a
#: different kernel, not a faster setting of the same one.
#: `gdn_state_geometry` is the conv width and head geometry the fixed state
#: has, which sizes every decode step regardless of the batch.
GDN_SCOPE = ("gdn_decode_lossy_fast", "gdn_state_geometry")

#: Back-compatible name: the unified family's scope, which is what the module
#: required when it modelled only that family.
REQUIRED_SCOPE = UNIFIED_SCOPE

#: Bytes of KV one history row costs, per MHA layer, at the measured
#: deployment: heads 4 x head_dim 256 x 2 bytes (BF16) x K and V = 4 KiB.
#:
#: Stated because it is easy to get wrong by an order of magnitude and the
#: wrong number has been written down before. The pool record is a *block* of
#: 16 rows -- 16 x 4 x 256 x 2 x 2 = 64 KiB per MHA layer -- and the model has
#: 16 MHA layers, not 64: the other 48 bound modules are GDN and hold no KV.
#: Dividing the pool across all 64 gives 16 KiB and is the error to avoid.
#:
#: It is deliberately NOT a separate feature. Per-layer bytes are this
#: constant times `history_rows`, so a byte column would be exactly collinear
#: with the row column and the fit would refuse it. `history_rows` *is* the
#: gather term; this is the scale a reader needs to interpret its coefficient.
KV_BYTES_PER_HISTORY_ROW_PER_MHA_LAYER = 4 * 256 * 2 * 2
MHA_LAYERS = 16


class Refusal:
    """Why no price was produced. Carries the reason, never a number."""

    __slots__ = ("reason", "missing")

    def __init__(self, reason: str, missing: tuple = ()) -> None:
        self.reason = reason
        self.missing = tuple(missing)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Refusal({self.reason!r})"


class Structure:
    """The ragged shape of one attention call, as the key records it.

    ``queries`` and ``histories`` are per request and stay per request. The
    product of their sums is a different quantity from the sum of their
    products, and only the second is work: one long query over a short history
    and one short query over a long history do not cost what their totals
    suggest.
    """

    __slots__ = ("queries", "histories", "is_prefill", "has_cached", "state",
                 "bucket", "num_prefills", "num_decodes", "num_actual_tokens",
                 "num_spec_decodes", "num_spec_decode_tokens", "replayssm",
                 "spec_masked", "has_initial_state", "executed_rows",
                 "state_indices")

    def __init__(self, queries=(), histories=(), *, is_prefill=None,
                 has_cached=None, state=None, bucket=None,
                 num_prefills=None, num_decodes=None, num_actual_tokens=None,
                 num_spec_decodes=None, num_spec_decode_tokens=None,
                 replayssm=None, spec_masked=None, has_initial_state=None,
                 executed_rows=None, state_indices=None):
        self.queries = tuple(int(q) for q in queries)
        self.histories = tuple(int(h) for h in histories)
        self.is_prefill = is_prefill
        self.has_cached = has_cached
        self.state = state
        self.bucket = bucket
        self.num_prefills = num_prefills
        self.num_decodes = num_decodes
        self.num_actual_tokens = num_actual_tokens
        self.num_spec_decodes = num_spec_decodes
        self.num_spec_decode_tokens = num_spec_decode_tokens
        self.replayssm = replayssm
        self.spec_masked = spec_masked
        self.has_initial_state = has_initial_state
        self.executed_rows = executed_rows
        self.state_indices = (None if state_indices is None
                              else tuple(int(s) for s in state_indices))

    @property
    def sequences(self) -> int:
        return len(self.queries) or int(self.num_decodes or 0)

    @property
    def active_sequences(self) -> Optional[int]:
        """How many of the recorded rows carry a request.

        A padded decode's offsets repeat the last real one, so a bucket of four
        holding three requests records `cu_seqlens_q=[0,1,2,3,3]` -- four rows,
        lengths `(1,1,1,0)`. `sequences` counts all four, because all four are
        launched; this counts the three that have a query to compute.

        `None` when the call records no offsets at all, which is the one case
        where the two cannot be told apart.
        """
        if not self.queries:
            return None
        return sum(1 for q in self.queries if q > 0)

    @property
    def state_lanes(self) -> Optional[int]:
        """How many recurrent-state lanes this GDN call actually works on.

        Not the launched lane count. A FULL capture bakes ``num_decodes = bs``
        into the graph and the replay refills the buffers around it: the state
        index tail is filled with ``PAD_SLOT_ID`` and the query offsets repeat
        the last real one (gdn_attn.py:1224-1235, :1264-1281). The convolution
        skips a PAD index and the recurrence skips a zero-length lane, so the
        state work is the *unpadded* count while the gating and the output copy
        still run the bucket's width. Counting the launched lanes for both --
        which ``sequences`` does, because for attention they are the same
        number -- makes three-of-four and four-of-four the same vector.

        Read off the state index tensor, which is where the padding is
        explicit. The offsets give the same count and are used to contradict
        it, never to supply it: if the two disagree the metadata does not
        describe one step, and :func:`features_for` refuses rather than
        picking the more convenient one. ``None`` when neither is recorded.
        """
        if self.state_indices is None:
            return self.active_sequences
        return sum(1 for s in self.state_indices if s != PAD_SLOT_ID)

    @property
    def query_total(self) -> int:
        return sum(self.queries) if self.queries else int(
            self.num_actual_tokens or 0)

    @property
    def history_total(self) -> int:
        return sum(self.histories)

    def paired_work(self) -> int:
        """``sum_i [ q_i*h_i + q_i(q_i+1)/2 ]`` -- the attended pairs.

        Per request, deliberately. The first term is every new query row
        reading every cached history row; the second is the causal triangle
        among the new rows themselves.
        """
        total = 0
        for q, h in zip(self.queries, self.histories):
            total += q * h + q * (q + 1) // 2
        return total

    def chunks(self) -> int:
        """Per-sequence ceil(q/CHUNK_SIZE), summed. Not ceil of the total."""
        return sum(-(-q // CHUNK_SIZE) for q in self.queries)

    def contexts(self) -> tuple:
        """Per request, what the decode kernel reads: history plus own query."""
        if self.queries and len(self.queries) == len(self.histories):
            return tuple(q + h for q, h in zip(self.queries, self.histories))
        return self.histories

    def split_tiles(self, partition: int, splits: int,
                    window: Optional[int] = None) -> int:
        """Partition tiles the gluon paged decode kernel actually iterates.

        Not ``sum(ceil(C / partition))``. `run_pa_decode_gluon` launches a grid
        of ``(sequences, kv_heads, splits)`` and each split program covers one
        contiguous *page* of its own sequence
        (`pa_decode_gluon.py`:1481-1491)::

            page  = ceil(C / splits)
            start = (page * j) // partition
            end   = ceil(min(C, page * (j + 1)) / partition)

        Each split rounds to whole partitions separately, so a page boundary
        falling inside a partition makes both neighbouring splits load that
        partition. The summed form misses it exactly where it matters: at eight
        splits, contexts ``(2048, 2305)`` and ``(2176, 2177)`` both sum to 18
        partitions, and the kernel runs 25 and 32 tile iterations per KV head.

        ``window`` is the deployment's sliding window where it sets one. That
        branch (:1453-1478) tiles only the window and hands each split a whole
        number of partitions, so nothing there is covered twice -- and the
        caller pins the split count to one for it anyway
        (`attention_mha.py`:554-556).

        Per KV head: the head is the grid's second axis and every head repeats
        this work, so the head count belongs to the law's coefficient rather
        than to the count.
        """
        splits = max(1, int(splits))
        total = 0
        for context in self.contexts():
            context = int(context)
            if context <= 0:
                continue
            if window is not None and window > 0:
                start = max(0, (context - window) // partition)
                total += max(0, -(-context // partition) - start)
                continue
            page = -(-context // splits)
            for index in range(splits):
                low = page * index
                if low >= context:
                    break
                high = min(context, low + page)
                total += -(-high // partition) - low // partition
        return total

    def continued(self) -> Optional[int]:
        """Sequences whose scan resumes from a recurrent state already held.

        `has_initial_state` is a per-sequence boolean mask the engine builds,
        and it is a branch, not a detail: a fresh sequence starts its chunked
        scan from a zero state, a continued one loads the state the pool holds
        and carries it in. Returns None where no mask was recorded, so a caller
        can refuse rather than read an absence as "all fresh".
        """
        mask = self.has_initial_state
        if mask is None:
            return None
        if self.queries:
            mask = list(mask)[:len(self.queries)]
        return sum(1 for flag in mask if flag)


def _context(op: dict) -> dict:
    return {k: v for k, v in (tuple(x) for x in op.get("context") or ())}


def _serialized(value):
    """Values out of one of `forward_ctx`'s captured tensors.

    `_capture_linear_attention` writes each GDN tensor as ``[values,
    dtype_name]``, because these are a mix of index tensors and boolean masks
    and a mask rebuilt as int32 selects nothing. Read in that shape, so a
    fixture serialized by the source collector parses without a second
    convention. A bare list is accepted too -- the unified family's
    `cu_seqlens_q` is written that way -- and anything else is None rather than
    a guess.
    """
    if isinstance(value, (list, tuple)):
        if (len(value) == 2 and isinstance(value[0], (list, tuple))
                and isinstance(value[1], str)):
            return list(value[0])
        if all(isinstance(v, (int, bool)) for v in value):
            return list(value)
    return None


def _starts_to_lengths(starts) -> tuple:
    values = _serialized(starts)
    if not values or len(values) < 2:
        return ()
    return tuple(int(values[i + 1]) - int(values[i])
                 for i in range(len(values) - 1))


def structure_of(op: dict) -> Optional[Structure]:
    """The ragged structure an operator records, or None if it records none."""
    ctx = _context(op)
    queries = _starts_to_lengths(ctx.get("cu_seqlens_q"))
    if not queries:
        # GDN records its offsets under its own names. The non-spec tensor is
        # the one that describes the sequences this model prices; the spec one
        # belongs to a branch `regime_of` refuses.
        queries = _starts_to_lengths(ctx.get("non_spec_query_start_loc"))
    context = _serialized(ctx.get("context_lens"))
    histories: tuple = ()
    if context and queries:
        histories = tuple(int(c) - q for c, q in zip(context, queries))
    elif context:
        histories = tuple(int(c) for c in context)
    return Structure(
        queries, histories,
        is_prefill=ctx.get("is_prefill"),
        has_cached=ctx.get("has_cached"),
        state=ctx.get("state"),
        bucket=ctx.get("capture_bucket"),
        num_prefills=ctx.get("num_prefills"),
        num_decodes=ctx.get("num_decodes"),
        num_actual_tokens=ctx.get("num_actual_tokens"),
        num_spec_decodes=ctx.get("num_spec_decodes"),
        num_spec_decode_tokens=ctx.get("num_spec_decode_tokens"),
        replayssm=ctx.get("replayssm"),
        spec_masked=(_serialized(ctx.get("spec_sequence_masks")) is not None
                     or _serialized(ctx.get("spec_query_start_loc"))
                     is not None),
        has_initial_state=_serialized(ctx.get("has_initial_state")),
        executed_rows=_output_rows(op),
        state_indices=_serialized(ctx.get("non_spec_state_indices_tensor")),
    )


#: `core_attn_out` is operand 3 of `linear_attention_with_output_base(mixed_qkv,
#: b, a, core_attn_out, layer_name)`. Its row count is the width the call was
#: given, which is not the width it computed over: the wrapper slices its
#: operands to `num_actual_tokens` and then zeros `core_attn_out` from there to
#: the end. Both numbers are recorded facts, and the difference between them is
#: work.
_GDN_OUTPUT_OPERAND = 3


#: `q` is operand 0 of `unified_attention_with_output_base(q, q_scale, k, v,
#: positions, layer_name, use_mla, qkv)`, shaped `[rows, num_q_heads,
#: head_dim]`. The Gluon decode kernel takes its launch extent straight off it
#: -- `batch_size = query.shape[0] // query_length`, and `grid = (batch_size,
#: num_kv_heads, max_context_partition_num)` (pa_decode_gluon.py:5342, :5356)
#: -- so how many rows the kernel executed is a property of the operand, not a
#: deployment field somebody has to declare.
_UNIFIED_QUERY_OPERAND = 0


def _output_rows(op: dict):
    """Rows the kernel was launched over, where the key records them.

    For GDN, the width `core_attn_out` was allocated with. For the unified
    wrapper, the query rows divided by the per-row query length -- the same
    expression the kernel itself uses. Both are read off the recorded call;
    neither needs `capture_bucket`, which no operator context carries.
    """
    name = op.get("name")
    shapes = op.get("input_shapes") or ()
    if name == GDN:
        if len(shapes) <= _GDN_OUTPUT_OPERAND:
            return None
        shape = shapes[_GDN_OUTPUT_OPERAND]
        if not isinstance(shape, (list, tuple)) or not shape:
            return None
        return int(shape[0])
    if name == UNIFIED:
        if len(shapes) <= _UNIFIED_QUERY_OPERAND:
            return None
        shape = shapes[_UNIFIED_QUERY_OPERAND]
        if not isinstance(shape, (list, tuple)) or not shape:
            return None
        per_row = _context(op).get("max_seqlen_q")
        try:
            per_row = int(per_row)
        except (TypeError, ValueError):
            return None
        if per_row <= 0 or int(shape[0]) % per_row:
            # A row count that is not a whole number of query lengths is not
            # this kernel's batch. Unknown beats a floor division that would
            # come back as a plausible extent.
            return None
        return int(shape[0]) // per_row
    return None


def _hashable(value):
    """A recorded value in a form two records can be compared by.

    Recursive, because the things being compared are not flat: a recorded
    operand view is a nested list of strides, and a shallow conversion leaves
    inner lists unhashable -- which on real strided operators is not a subtle
    inaccuracy but a crash at the first grouping.
    """
    if isinstance(value, dict):
        return tuple(sorted((str(k), _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


#: The axis a ragged structure varies. Substituted for a recorded extent only
#: where that extent equals a token count the key itself records, so it is the
#: key that says an axis is the token axis and never a position convention.
TOKEN_AXIS = "*tokens"


def geometry_of(op: dict, structure: Optional["Structure"] = None) -> tuple:
    """The static operand geometry of this call: everything but the tokens.

    Heads, head dimension, conv width, state rank, the operand dtypes and the
    recorded operand views. All of it selects the kernel and sets its cost per
    row, so two calls that differ in any of it are not points on one law --
    and a law fitted over 4 KV heads must not answer a request with 8.

    The token extent is abstracted away, because varying it is precisely what
    a ragged fit is a fit over. Only extents the key itself states are token
    counts are abstracted; an axis that merely happens to equal one at this
    batch size keeps its number, so nothing is generalised on a coincidence.
    """
    structure = structure_of(op) if structure is None else structure
    counts = set()
    if structure is not None:
        for value in (structure.query_total, structure.num_actual_tokens,
                      structure.executed_rows):
            if isinstance(value, int) and value > 0:
                counts.add(value)
    shapes = []
    for shape in op.get("input_shapes") or ():
        if isinstance(shape, (list, tuple)) and shape and shape[0] in counts:
            shapes.append((TOKEN_AXIS,) + tuple(shape[1:]))
        else:
            shapes.append(_hashable(shape))
    return (tuple(shapes), tuple(op.get("dtypes") or ()),
            _hashable(op.get("layouts") or ()))


def scoped(op: dict, scope, structure: Optional["Structure"] = None) -> dict:
    """``scope`` with this call's static operand geometry folded in.

    Both a fit and a request go through here, so a law is identified by its
    geometry as well as its regime and its deployment, and a request whose
    geometry differs is refused by the same scope machinery that refuses a
    different KV dtype -- rather than being priced by a law fitted on other
    heads.
    """
    combined = dict(scope or {})
    combined["operand_geometry"] = geometry_of(op, structure)
    return combined


class Regime:
    """One native branch: the features its cost depends on, and its scope."""

    __slots__ = ("name", "features", "required_scope")

    def __init__(self, name: str, features: tuple,
                 required_scope: tuple = UNIFIED_SCOPE) -> None:
        self.name = name
        self.features = tuple(features)
        self.required_scope = tuple(required_scope)

    def __eq__(self, other):
        return isinstance(other, Regime) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Regime({self.name!r})"


#: Features per regime. Kept minimal and checked for collinearity at fit time
#: rather than assumed independent: `tokens` and `chunks` coincide exactly when
#: every query is a multiple of CHUNK_SIZE, which is true of every GDN prefill
#: measured so far.
REGIMES = {
    # `calls` first, for the same reason it leads `gdn.prefill`: the wrapper
    # launches once per batch and that launch is not free. P1 is the training
    # point that says so -- 2080 paired positions over 64 query rows, as close
    # to no work as a measured call gets, and it still costs 28.41us. A law
    # with only proportional terms charges P1 essentially nothing and has to
    # absorb that floor into the slopes of the points that do work.
    "unified.prefill.cold": Regime("unified.prefill.cold",
                                   ("calls", "paired_work", "query_rows")),
    "unified.prefill.cached": Regime("unified.prefill.cached",
                                     ("calls", "paired_work", "query_rows",
                                      "history_rows")),
    # Decode is two regimes, not one, because `paged_attention_triton` is a
    # fork: with `ATOM_USE_UNIFIED_ATTN` or the flash layout it calls aiter's
    # `unified_attention` over the paged cache, and otherwise it runs the
    # gluon paged decode, which partitions each sequence into
    # `context_partition_size` tiles and reduces across them. Those are
    # different kernels with different cost laws, and imposing either one's law
    # on the other is the mistake this split exists to prevent. Which ran is a
    # scope fact, so `regime_of` refuses a decode whose backend is undeclared.
    #
    # On the partitioned branch the tile count is the ragged term: a batch pays
    # for a part-full tail tile per sequence, so two batches with the same
    # summed context and different raggedness do not cost the same.
    #
    # `split_tiles`, not a summed `ceil(C / 256)`. The launcher splits each
    # sequence into `splits` contiguous pages and each page rounds to whole
    # partitions on its own (pa_decode_gluon.py:1481-1491), so the boundaries
    # are paid for. The two names are not the same number: at eight splits,
    # contexts (2048, 2305) and (2176, 2177) both sum to 18 partitions and run
    # 25 and 32 tile iterations. The feature was renamed rather than
    # redefined -- a law fitted against the summed count is a law about a
    # different quantity, and should not be silently reinterpreted.
    "unified.decode.paged_gluon": Regime(
        "unified.decode.paged_gluon",
        ("context_rows", "split_tiles", "active", "bucket_pad"),
        PAGED_GLUON_SCOPE),
    # On the unified/flash branch there is no per-sequence partition to count.
    # What the measurements show instead is that raggedness dominates: a
    # 32-sequence mixed batch summing 394164 context rows costs ~3.70ms while a
    # balanced 32x16384 batch summing 524288 costs ~0.998ms -- more rows, less
    # than a third the time. No law in summed rows can produce that, so the
    # ragged term here is the grid the longest sequence forces every sequence
    # to be covered by, and `grid_pad_rows` is what that grid covers beyond the
    # rows that exist. It is a candidate law and nothing more until a holdout
    # at a structure it was not fitted on says otherwise.
    "unified.decode.unified_attn": Regime(
        "unified.decode.unified_attn",
        ("context_rows", "grid_pad_rows", "active", "bucket_pad")),
    # Three widths, and a padded decode makes them three different numbers.
    #
    # `state_lanes` -- the lanes that do recurrent work. A FULL capture bakes
    # `num_decodes = bs` into the graph, and the replay refills the state index
    # tail with PAD and repeats the last query offset (gdn_attn.py:1224-1235,
    # :1264-1281). The convolution skips a PAD index and the recurrence skips a
    # zero-length lane, so three-of-four does less state work than four-of-four
    # at the same launch. A single `active` term read off the offsets counted
    # all four for both and made them the same vector.
    #
    # `actual_rows` -- the rows the gating and the output copy actually
    # process, which is `num_actual_tokens` and not the allocated width. The
    # two are the same number under FULL, where the capture pinned
    # `num_actual_tokens` to `bs`; under PIECEWISE they are not.
    # `attention_gdn.py` slices `a` and `b` to `num_actual_tokens` and copies
    # `output[:num_actual_tokens]`, so a PIECEWISE step at bucket four with
    # three active rows gates three rows, not four. Charging the allocated
    # width there would order a PIECEWISE step above a FULL one that does
    # strictly more work.
    #
    # `tail_pad_rows` -- the rows `attention_gdn.py` zeros above
    # `num_actual_tokens` for replay safety, which is the rest of the
    # allocation. Under FULL this is *zero*: the capture pinned
    # `num_actual_tokens` to `bs`, so there is no tail to zero and charging one
    # would be inventing work. Under PIECEWISE the counts are the batch's while
    # the allocation is still the bucket's, so the branch is real and the term
    # is what tells the two modes apart. The allocated width is still
    # recoverable as `actual_rows + tail_pad_rows`; it is split because the
    # zeroing and the gating are different work at different rates.
    "gdn.decode": Regime("gdn.decode",
                         ("calls", "state_lanes", "actual_rows",
                          "tail_pad_rows"),
                         GDN_SCOPE),
    # `continued_sequences`, because `has_initial_state` is a branch the
    # native scan takes per sequence: a fresh one starts from a zero state, a
    # continued one loads the recurrent state the pool holds and carries it
    # into the first chunk. Every GDN prefill measured so far is all-fresh, so
    # the column pins to zero and the fit states the subdomain it covers --
    # which is what makes a mixed fresh/continued batch a named refusal here
    # instead of a price with no evidence under it.
    # `calls` because the wrapper runs once per batch whatever the batch
    # holds, and the measurements say that term is not zero. Two priced GDN
    # prefills settle it on their own: 96 query rows in 2 chunks over 1
    # sequence costs 156.07us, and 224 rows in 4 chunks over 2 sequences --
    # more of every structural term, on the same graph, the same operand
    # rotation and the same region count -- costs 201.46us. A law with no
    # intercept and no negative coefficients has to charge the second at least
    # twice the first, so it cannot come within 21% of both; the per-call term
    # is what the evidence requires, not what makes a number fit. It is the
    # same term `gdn.decode` already carries, for the same reason: the launch
    # and the fixed conv and recurrent state a step touches regardless of its
    # rows. Identified from training designs; nothing here is fitted to a
    # holdout.
    "gdn.prefill": Regime("gdn.prefill",
                          ("calls", "query_rows", "chunks", "sequences",
                           "continued_sequences", "tail_pad_rows"),
                          GDN_SCOPE),
}

#: `context_partition_size` in the gluon paged decode path, and *only* there.
#: The sliding-window case uses 128 and one partition, which is why the window
#: is in the required scope and this is read through it rather than assumed.
DECODE_PARTITION_SIZE = 256
DECODE_PARTITION_SIZE_SLIDING = 128

#: The resolved decode kernel, as whoever resolves the scope must name it, to
#: the regime whose law it takes. Exact values, refused when unknown: a decode
#: priced under the wrong branch's law is the failure this table prevents.
DECODE_KERNELS = {
    "unified_attention": "unified.decode.unified_attn",
    "paged_gluon": "unified.decode.paged_gluon",
}

#: Decode kernels that exist and have no law here. They are listed rather than
#: aliased onto one that does: `paged_attention_persistent_asm` and
#: `paged_attention_asm` are separate implementations, and nothing measured so
#: far shows either of them following the gluon path's tile law. Mapping them
#: onto it would be a price with no evidence behind it, so they refuse by name
#: until a source primitive measurement says which law they take.
UNPROVEN_DECODE_KERNELS = ("paged_attention_persistent_asm",
                           "paged_attention_asm", "paged_attention_triton")


def _partition_size(scope) -> int:
    window = (scope or {}).get("sliding_window")
    if isinstance(window, int) and window > 0:
        return DECODE_PARTITION_SIZE_SLIDING
    return DECODE_PARTITION_SIZE


#: `get_recommended_splits` caps the split count here (pa_decode_gluon.py:118)
#: and assumes two workgroups per CU (`get_occupancy`, :107-108).
DECODE_MAX_SPLITS = 8
DECODE_OCCUPANCY = 2


def _decode_splits(scope, sequences):
    """How many context splits the launcher asks for, or a `Refusal`.

    `attention_mha.py`:552 computes it per call:
    ``min(8, ceil(compute_units * occupancy / (num_seqs * num_kv_heads)))``.
    It is not fixed across a batch-size sweep -- a batch of 8 and a batch of
    64 on the same part take different split counts and so different tile
    geometry -- which is why it is derived here rather than declared whole.

    The sliding-window branch is pinned to one split by the caller
    (`attention_mha.py`:554-556), and that is a source fact rather than an
    arithmetic one, so it is returned before anything else is read.
    """
    window = (scope or {}).get("sliding_window")
    if isinstance(window, int) and window > 0:
        return 1
    heads = (scope or {}).get("num_kv_heads")
    units = (scope or {}).get("compute_units")
    missing = tuple(name for name, value in (("num_kv_heads", heads),
                                             ("compute_units", units))
                    if not value)
    if missing:
        return Refusal(
            "the gluon decode grid's split count is "
            "min(8, ceil(compute_units * 2 / (sequences * num_kv_heads))), and "
            "this scope declares neither %s. The tile geometry follows from "
            "it, so a tile count without it is not this kernel's work"
            % " nor ".join(missing),
            missing=missing)
    if not sequences:
        return Refusal(
            "the call records no sequence count, and the split count the "
            "launcher asks for is computed from it")
    lanes = int(sequences) * int(heads)
    splits = -(-int(units) * DECODE_OCCUPANCY // lanes)
    return max(1, min(DECODE_MAX_SPLITS, splits))


def regime_of(op: dict, structure: Optional[Structure] = None, scope=None):
    """Which native branch this call takes, or a `Refusal` naming the gap.

    ``scope`` is the declared static scope. Decode needs it: which decode
    kernel ran is not in the key, and the two do not share a cost law.
    """
    name = op.get("name", "")
    structure = structure_of(op) if structure is None else structure
    if structure is None:
        return Refusal("the operator records no ragged structure")
    if name == UNIFIED:
        if structure.is_prefill is None:
            return Refusal(
                "the key does not say whether this is a prefill, and the "
                "prefill and decode kernels are different work")
        if not structure.is_prefill:
            backend = (scope or {}).get("attention_backend")
            if backend is None:
                return Refusal(
                    "which decode kernel ran is not declared. "
                    "`paged_attention_triton` forks on ATOM_USE_UNIFIED_ATTN "
                    "and the flash layout into aiter's unified_attention and "
                    "the gluon paged decode, and those partition the context "
                    "differently; one law imposed on the other is a wrong "
                    "price, not an approximate one",
                    missing=("attention_backend",))
            regime = DECODE_KERNELS.get(str(backend))
            if regime is None:
                known = (" It is a kernel this knows of and has no law for; "
                         "nothing measured shows it follows another's."
                         if str(backend) in UNPROVEN_DECODE_KERNELS else "")
                return Refusal(
                    "%r is not a decode kernel this has a law for.%s Declare "
                    "one of %s" % (backend, known,
                                   ", ".join(sorted(DECODE_KERNELS))),
                    missing=("attention_backend",))
            return REGIMES[regime]
        if structure.has_cached is None:
            return Refusal(
                "the key does not say whether a cached prefix was read, and "
                "the cold and cached prefill paths read different KV")
        return REGIMES["unified.prefill.cached" if structure.has_cached
                       else "unified.prefill.cold"]
    if name == GDN:
        prefills = structure.num_prefills
        decodes = structure.num_decodes
        if structure.replayssm:
            return Refusal(
                "this call ran under ReplaySSM, where the state pool holds one "
                "checkpoint per request and the per-draft states are "
                "reconstructed on demand; that is a different kernel and this "
                "models the plain one",
                missing=("replayssm",))
        if structure.spec_masked or int(structure.num_spec_decodes or 0) \
                or int(structure.num_spec_decode_tokens or 0):
            return Refusal(
                "this call carries speculative sequences, which take the "
                "multi-query conv update and a verify window rather than the "
                "single-token path; no measurement here separates their share",
                missing=("spec_sequence_masks",))
        if structure.num_actual_tokens is None:
            return Refusal(
                "the key does not carry num_actual_tokens, and the wrapper "
                "slices its operands to that before the convolution, so how "
                "many rows the kernel ran over is unknown",
                missing=("num_actual_tokens",))
        if prefills is None or decodes is None:
            return Refusal(
                "the key does not carry the prefill/decode split, which is "
                "the branch the linear-attention kernel takes")
        if int(prefills) and int(decodes):
            return Refusal(
                "this call mixes fresh prefill sequences with continued "
                "decode ones; the two run different kernels in one call and "
                "no measurement separates their share")
        if int(prefills):
            if structure.has_initial_state is None:
                return Refusal(
                    "the key does not carry has_initial_state, and that mask "
                    "is the branch the chunked scan takes per sequence: a "
                    "fresh sequence starts from a zero state, a continued one "
                    "loads the state the pool holds. Reading its absence as "
                    "all-fresh would be assuming the branch",
                    missing=("has_initial_state",))
            return REGIMES["gdn.prefill"]
        return REGIMES["gdn.decode"]
    return Refusal(f"{name} is not an attention family this models")


def features_for(regime: Regime, structure: Structure, scope=None):
    """The feature vector for one call, or a `Refusal` for what it lacks.

    ``scope`` is the declared static scope the call ran under. It is read for
    the one feature that needs it -- the decode partition width, which the
    sliding-window case halves -- and never to fill in a structural fact.
    """
    partition = _partition_size(scope)
    values = []
    for feature in regime.features:
        if feature == "calls":
            # A per-call cost: the launch, and the fixed conv and recurrent
            # state a GDN step touches whatever the batch holds.
            values.append(1.0)
        elif feature == "split_tiles":
            if not structure.contexts():
                return Refusal(
                    "the call records no per-sequence context, so the tiles "
                    "its splits cover cannot be counted")
            splits = _decode_splits(scope, structure.sequences)
            if isinstance(splits, Refusal):
                return splits
            window = (scope or {}).get("sliding_window")
            values.append(float(structure.split_tiles(
                partition, splits,
                window if isinstance(window, int) and window > 0 else None)))
        elif feature == "grid_pad_rows":
            contexts = structure.contexts()
            if not contexts:
                return Refusal("the call records no per-sequence context, so "
                               "how ragged the batch was is unknown")
            values.append(float(max(contexts) * len(contexts)
                                - sum(contexts)))
        elif feature == "continued_sequences":
            continued = structure.continued()
            if continued is None:
                return Refusal(
                    "the call records no has_initial_state mask, so how many "
                    "of its sequences resume a held recurrent state is "
                    "unknown; that is a branch, not a zero",
                    missing=("has_initial_state",))
            values.append(float(continued))
        elif feature == "state_lanes":
            lanes = structure.state_lanes
            if lanes is None:
                return Refusal(
                    "the call records neither a state index tensor nor query "
                    "offsets, so how many lanes carry a live recurrent state "
                    "is unknown; a padded lane is skipped, not cheap",
                    missing=("non_spec_state_indices_tensor",))
            # The offsets say the same thing, and where both are recorded they
            # have to agree: they are two views of one padding decision
            # (gdn_attn.py:1224-1235 writes them together). A disagreement
            # means the key does not describe one step.
            by_offset = structure.active_sequences
            if (structure.state_indices is not None and by_offset is not None
                    and by_offset != lanes):
                return Refusal(
                    f"{lanes} state lanes are unpadded but {by_offset} rows "
                    "carry a query; the state index tensor and the query "
                    "offsets describe different steps")
            values.append(float(lanes))
        elif feature == "tail_pad_rows":
            if structure.executed_rows is None:
                return Refusal(
                    "the key does not record the width the output tensor was "
                    "allocated with, so the padding tail the kernel zeroes "
                    "cannot be counted; underfill is not shown to be free",
                    missing=("output_rows",))
            if structure.num_actual_tokens is None:
                return Refusal(
                    "the key does not record num_actual_tokens, so the sliced "
                    "width the kernel ran over is unknown",
                    missing=("num_actual_tokens",))
            values.append(float(max(int(structure.executed_rows)
                                    - int(structure.num_actual_tokens), 0)))
        elif feature == "actual_rows":
            if structure.num_actual_tokens is None:
                return Refusal(
                    "the call does not record num_actual_tokens, which is "
                    "what the wrapper sliced its operands to",
                    missing=("num_actual_tokens",))
            values.append(float(structure.num_actual_tokens))
        elif feature == "paired_work":
            if len(structure.queries) != len(structure.histories):
                return Refusal("queries and histories are not paired per "
                               "request, so the attended pairs are unknown")
            values.append(float(structure.paired_work()))
        elif feature == "query_rows":
            values.append(float(structure.query_total))
        elif feature == "history_rows":
            values.append(float(structure.history_total))
        elif feature == "context_rows":
            values.append(float(structure.history_total + structure.query_total))
        elif feature == "active":
            values.append(float(structure.sequences))
        elif feature == "sequences":
            values.append(float(len(structure.queries)))
        elif feature == "chunks":
            values.append(float(structure.chunks()))
        elif feature == "bucket_pad":
            # Derived, never guessed -- but derived from the call's own rows
            # rather than from a declared bucket. `capture_bucket` is a field
            # of `StepShape` and `BatchSpec` that no operator context carries,
            # so requiring it here refused every decode vector this family can
            # otherwise build, including the ones already recorded.
            #
            # What the kernel launches is on the operands: the Gluon decode
            # takes `batch_size = query.shape[0] // query_length` and the
            # padded rows are the ones whose query length is zero. The recorded
            # bucket is used only to contradict that, never to supply it.
            rows = structure.executed_rows
            if rows is None:
                rows = structure.sequences
            active = structure.active_sequences
            if active is None:
                return Refusal(
                    "the call records no per-request query offsets, so how "
                    "many of its launched rows were padding is unknown; that "
                    "is a question for the padding owner, not a zero",
                    missing=("cu_seqlens_q",))
            if structure.bucket is not None and int(structure.bucket) != rows:
                return Refusal(
                    f"the call was launched over {rows} rows but declares a "
                    f"capture bucket of {structure.bucket}; one of the two "
                    "does not describe this step")
            values.append(float(max(int(rows) - int(active), 0)))
        else:  # pragma: no cover - guarded by REGIMES
            return Refusal(f"unknown feature {feature!r}")
    return values


# -- fitting ---------------------------------------------------------------


def _solve(matrix, rhs):
    """Least squares by normal equations, with a rank check. Pure stdlib.

    Small systems -- a handful of features -- so the normal equations are
    adequate and keeping this dependency-free matters more: this module is
    imported wherever a price is looked up.
    """
    n = len(matrix[0])
    ata = [[sum(row[i] * row[j] for row in matrix) for j in range(n)]
           for i in range(n)]
    atb = [sum(row[i] * y for row, y in zip(matrix, rhs)) for i in range(n)]
    # Gaussian elimination with partial pivoting.
    aug = [ata[i][:] + [atb[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None  # rank deficient
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


#: Active sets are enumerated, so the work is exponential in the column count.
#: Every regime here has at most six terms; the cap is a guard against a
#: regime that grows one day, not a limit anything currently meets.
_MAX_ENUMERATED_COLUMNS = 12


def _solve_nonnegative(matrix, rhs):
    """Least squares subject to every coefficient being at least zero.

    A cost is not negative, so the constraint is not a preference -- it is the
    only region of the parameter space that means anything. The unconstrained
    solution leaving it is what a near-collinear design looks like: with two
    columns that move together, the residual is flat along their difference
    and the split between them is decided by noise, which routinely puts one
    of them below zero. Refusing there throws away a law the evidence does
    support over the region that is meaningful.

    Solved by enumerating the active sets. The optimum of a nonnegative least
    squares problem is the unconstrained solution of SOME subset of columns
    with the rest held at zero, so with a handful of columns every subset can
    be tried and the best feasible one taken. That is the global optimum by
    construction, and it satisfies the KKT conditions because it is.

    Enumeration rather than a greedy descent, because the greedy version is
    wrong in exactly the case this exists for. Dropping the most negative
    column and refitting never reconsiders a column it dropped, and a column
    that is negative alongside its correlated partner can be positive once
    that partner is the one held at zero. On ``A = [[6,2,1],[4,8,1],[5,7,1],
    [5,3,1]]`` with ``y = [7,5,4,8]`` the free solution is ``(-2,-1,21)``;
    dropping greedily gives ``(0,0,6)`` at SSE 10, while ``(1,0,1)`` is
    feasible at SSE 8. Returns ``(coefficients, bounded)`` where ``bounded``
    are the column indices held at zero, or ``None`` when no subset of the
    columns is solvable.

    A column bounded at zero is NOT a column shown to be free. It is a column
    this design cannot separate from the ones it moves with, and the caller
    has to say so -- see `Fit.bounded`.
    """
    width = len(matrix[0])
    if width > _MAX_ENUMERATED_COLUMNS:  # pragma: no cover - no such regime
        return None
    best = None
    for mask in range(1, 1 << width):
        free = [i for i in range(width) if mask & (1 << i)]
        if len(free) > len(matrix):
            continue
        sub = [[row[i] for i in free] for row in matrix]
        solved = _solve(sub, rhs)
        if solved is None or any(c < 0.0 for c in solved):
            continue
        predicted = [sum(c * v for c, v in zip(solved, row)) for row in sub]
        sse = sum((p - y) ** 2 for p, y in zip(predicted, rhs))
        if best is None or sse < best[0]:
            out = [0.0] * width
            for k, i in enumerate(free):
                out[i] = solved[k]
            best = (sse, out, tuple(i for i in range(width)
                                    if not mask & (1 << i)))
    if best is None:
        return None
    return best[1], best[2]


class Fit:
    """A regime's law, and everything a reader needs to distrust it.

    ``features`` are the terms this law actually carries. They can be fewer
    than its regime's: a column that is zero in every measurement carries no
    information about its own coefficient, and fitting it anyway makes the
    whole design rank deficient and refuses a law that the evidence otherwise
    supports. Those columns are ``pinned`` instead -- the fit is a fit of the
    subdomain where they are zero, and :meth:`Model.price` refuses a structure
    where any of them is not, rather than extrapolating a coefficient nobody
    measured.
    """

    __slots__ = ("regime", "features", "pinned", "coefficients", "scales",
                 "points", "residual_df", "relative_error", "domain", "scope",
                 "scope_undeclared", "bounded")

    def __init__(self, regime, features, pinned, coefficients, scales, points,
                 residual_df, relative_error, domain, scope,
                 scope_undeclared=(), bounded=()):
        self.regime = regime
        self.features = tuple(features)
        self.pinned = tuple(pinned)
        self.coefficients = tuple(coefficients)
        self.scales = tuple(scales)
        self.points = points
        self.residual_df = residual_df
        self.relative_error = relative_error
        self.domain = domain
        self.scope = scope
        self.scope_undeclared = tuple(scope_undeclared)
        #: Terms the nonnegativity constraint holds at zero. Carried and
        #: reported, never silently dropped: a zero here is not evidence that
        #: the work is free, it is this design's inability to separate that
        #: term from the ones it moves with.
        self.bounded = tuple(bounded)

    def predict(self, values):
        """``values`` in this fit's own feature order, pinned ones removed."""
        total = 0.0
        for value, coefficient, scale in zip(values, self.coefficients,
                                             self.scales):
            total += coefficient * (value / scale if scale else 0.0)
        return total

    def describe(self) -> str:
        terms = ", ".join(
            "%s=%.4e" % (name, coefficient / scale if scale else 0.0)
            for name, coefficient, scale in zip(self.features,
                                                self.coefficients, self.scales))
        note = ("" if not self.scope_undeclared else
                "; scope undeclared: " + ", ".join(self.scope_undeclared))
        if self.pinned:
            note += ("; measured only where %s is zero"
                     % ", ".join(self.pinned))
        if self.bounded:
            note += ("; %s held at zero by the nonnegativity bound, which is "
                     "active there -- the free solve wanted a negative cost "
                     "for them, either because this design cannot separate "
                     "them from the terms they move with or because a term "
                     "is missing. The law charges nothing for them and does "
                     "not claim they are free" % ", ".join(self.bounded))
        return ("%s from %d point(s), %d residual df, in-sample %.1f%% [%s]%s"
                % (self.regime.name, self.points, self.residual_df,
                   self.relative_error * 100, terms, note))


class _Absent:
    """Distinct from a declared ``None``.

    `sliding_window: None` is a statement -- there is no window. Not carrying
    the key at all is the absence of a statement. Collapsing the two would let
    an observation that declares nothing pool with one that declares a window
    is off.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<undeclared>"


ABSENT = _Absent()


def _distinct(values):
    """The distinct values among these, by equality, tolerating lists."""
    out = []
    for value in values:
        if any((value is ABSENT) == (other is ABSENT)
               and (value is ABSENT or value == other) for other in out):
            continue
        out.append(value)
    return out


def _scope_of(observations, required=UNIFIED_SCOPE):
    """The static scope these observations share, or a refusal to pool them.

    Agreement is required on **every** key any observation declares, not only
    on :data:`REQUIRED_SCOPE`. The required ones name the kernel; the rest name
    the conditions -- tensor-parallel geometry, which rotation the capture used,
    whether a file is superseded -- and two measurements that differ in any of
    them are measurements of different things. A key one observation carries and
    another does not is a disagreement too: the second has not said it matches,
    and reading its silence as agreement is the failure this is closed against.
    """
    keys = set(required)
    for obs in observations:
        keys.update(obs[3] or {})
    declared, undeclared = {}, []
    for field in sorted(keys):
        seen = _distinct([(obs[3] or {}).get(field, ABSENT)
                          for obs in observations])
        if len(seen) == 1 and seen[0] is ABSENT:
            undeclared.append(field)
            continue
        if len(seen) > 1:
            shown = ", ".join(sorted(repr(s) for s in seen))
            if field in required:
                return Refusal(
                    "these observations disagree on %s (%s); they are "
                    "measurements of different kernels and pooling them would "
                    "average work that is not the same work" % (field, shown))
            return Refusal(
                "these observations disagree on %s (%s); they were taken "
                "under different conditions, and pooling them into one "
                "training observation would average measurements of "
                "different deployments" % (field, shown))
        declared[field] = seen[0]
    return declared, tuple(f for f in undeclared if f in required)


def fit_regime(regime, observations, *, strict=True, min_residual_df=1):
    """Fit one regime, or refuse and say what would close the gap.

    ``observations`` are ``(structure, seconds, source, scope)`` tuples, each
    from a source primitive measurement of this one operator.

    Refuses, rather than producing a number, when: the observations do not
    declare the static scope (under ``strict``); they disagree on it; there are
    fewer points than features plus the required residual degrees of freedom;
    the design is rank deficient, which is what perfectly collinear features
    look like; or the fit wants a negative cost for some work.
    """
    scope = _scope_of(observations, regime.required_scope)
    if isinstance(scope, Refusal):
        return scope
    declared, undeclared = scope
    if strict and undeclared:
        return Refusal(
            "these measurements do not declare %s, so nothing says they were "
            "taken on one kernel; a price from them would be pooling regimes "
            "that may differ" % ", ".join(undeclared),
            missing=undeclared)

    full, rhs = [], []
    for structure, seconds, _source, obs_scope in observations:
        values = features_for(regime, structure, obs_scope)
        if isinstance(values, Refusal):
            return values
        full.append(values)
        rhs.append(float(seconds))

    # A column that is zero at every measured point says nothing about its own
    # coefficient, and carrying it makes the whole design rank deficient -- so
    # a mandatory padding term would refuse every fit over a set of full-bucket
    # captures, which is most of what exists. Pin it instead: the law is a law
    # of the subdomain where it is zero, and `price` refuses a structure where
    # it is not.
    kept = [i for i in range(len(regime.features))
            if any(row[i] for row in full)]
    pinned = tuple(name for i, name in enumerate(regime.features)
                   if i not in kept)
    if not kept:
        return Refusal(
            "%s: every one of %s is zero at every measured point, so there is "
            "no work here to attribute a cost to"
            % (regime.name, ", ".join(regime.features)))
    features = tuple(regime.features[i] for i in kept)
    rows = [[row[i] for i in kept] for row in full]
    domain = rows

    wanted = len(features) + min_residual_df
    if len(rows) < wanted:
        return Refusal(
            "%s has %d independent point(s) and needs at least %d to fit %d "
            "term(s) with anything left over to check them against"
            % (regime.name, len(rows), wanted, len(features)))

    # Scale each column by its largest value: the features differ by many
    # orders of magnitude -- attended pairs against sequence counts -- and the
    # normal equations would otherwise be conditioned by the units.
    scales = [max((abs(row[i]) for row in rows), default=0.0) or 1.0
              for i in range(len(features))]
    scaled = [[row[i] / scales[i] for i in range(len(row))] for row in rows]
    solved = _solve(scaled, rhs)
    if solved is None:
        return Refusal(
            "%s: the design is rank deficient -- two or more of %s do not "
            "vary independently across these points, so their coefficients "
            "cannot be told apart. Measure a point that separates them."
            % (regime.name, ", ".join(features)))
    bounded = ()
    if any(c < 0 for c in solved):
        # A cost is not negative, so an unconstrained solution outside that
        # region is not a law. It is a hint -- usually of a near-collinear
        # design, where noise decides the split between columns that move
        # together, sometimes of physics this regime's features are missing.
        # Either way the answer is the constrained optimum and an honest
        # statement of which terms its constraint holds at zero. Whether that
        # predictor is usable is decided by what it costs in error, below and
        # in cross-validation, not by the sign of the free solve.
        constrained = _solve_nonnegative(scaled, rhs)
        if constrained is None:
            return Refusal(
                "%s: the fit wants a negative cost for %s, and the design "
                "left after holding it at zero is rank deficient, so there is "
                "no nonnegative law these points identify. Measure a point "
                "that separates them."
                % (regime.name,
                   ", ".join(name for name, c in zip(features, solved)
                             if c < 0)))
        solved, at_zero = constrained
        bounded = tuple(features[i] for i in at_zero)
        if not any(solved):
            return Refusal(
                "%s: every term goes to zero under nonnegativity, so these "
                "points identify no cost at all" % regime.name)

    predicted = [sum(c * v for c, v in zip(solved, row)) for row in scaled]

    # A law has to beat the dullest rival there is: charging every point the
    # same number. If it does not, it has found no structure -- the seconds
    # move, and not with the work -- and whatever came out of the solve is a
    # shape fitted to scatter. Stated as a comparison rather than as a
    # tolerance so there is no threshold to tune: the mean is a competitor,
    # not a number somebody chose.
    mean = sum(rhs) / len(rhs)
    total = sum((y - mean) ** 2 for y in rhs)
    residual = sum((p - y) ** 2 for p, y in zip(predicted, rhs))
    if total > 0.0 and residual >= total:
        return Refusal(
            "%s: the law fits these points no better than charging every one "
            "of them the same number, so nothing here says the cost follows "
            "the work. Either a term is missing or these points are not one "
            "regime." % regime.name)

    errors = [abs(p - y) / y for p, y in zip(predicted, rhs) if y]
    # Residual df counts every fitted term, including the ones the constraint
    # holds at zero. Counting only the free ones would report more degrees of
    # freedom on a design that identified less, which is backwards.
    return Fit(regime, features, pinned, solved, scales, len(rows),
               len(rows) - len(solved), max(errors) if errors else 0.0,
               domain, declared, undeclared, bounded)


def scope_key(scope) -> tuple:
    """A declared scope as a hashable key. Every key it carries, sorted."""
    return tuple(sorted((str(k), repr(v)) for k, v in (scope or {}).items()))


def _by_scope(observations) -> dict:
    """Observations split by their own declared scope, in first-seen order."""
    groups: dict = {}
    for obs in observations:
        groups.setdefault(scope_key(obs[3]), []).append(obs)
    return groups


def _label(regime_name: str, key: tuple) -> str:
    if not key:
        return f"{regime_name} @ undeclared scope"
    return "%s @ %s" % (regime_name,
                        ", ".join("%s=%s" % (k, v) for k, v in key))


def _scope_matches(fit_scope, request_scope) -> Optional[str]:
    """The first key on which a fit and a request differ, or None.

    Every key either side declares has to agree. A request that is silent
    where the fit is specific has not said it matches, and a fit that is silent
    where the request is specific was not shown to have been measured there --
    which is the known-against-undeclared pooling this refuses.
    """
    fit_scope = fit_scope or {}
    request_scope = request_scope or {}
    for field in sorted(set(fit_scope) | set(request_scope)):
        mine = fit_scope.get(field, ABSENT)
        theirs = request_scope.get(field, ABSENT)
        if (mine is ABSENT) != (theirs is ABSENT):
            return field
        if mine is not ABSENT and mine != theirs:
            return field
    return None


class Model:
    """Every regime that could be fitted, and a price for a structure."""

    __slots__ = ("fits", "refusals", "strict")

    def __init__(self, strict=True):
        self.fits: dict = {}
        self.refusals: dict = {}
        self.strict = strict

    @classmethod
    def from_observations(cls, grouped, *, strict=True, min_residual_df=1):
        """``grouped`` maps a regime name to its observation list.

        Each regime's observations are split by their own declared scope before
        anything is fitted, so a law is identified by regime **and** scope. Two
        TP geometries, or a superseded collection beside a current one, become
        two fits or two refusals -- never one law averaged over both.
        """
        model = cls(strict=strict)
        for name, observations in grouped.items():
            regime = REGIMES.get(name)
            if regime is None:
                model.refusals[name] = Refusal(f"{name} is not a known regime")
                continue
            for key, group in _by_scope(observations).items():
                label = _label(name, key)
                outcome = fit_regime(regime, group, strict=strict,
                                     min_residual_df=min_residual_df)
                if isinstance(outcome, Refusal):
                    model.refusals[label] = outcome
                else:
                    model.fits[label] = outcome
        return model

    @classmethod
    def from_priced(cls, observations, *, strict=True, min_residual_df=1):
        """Build from ``(op, seconds, source, scope)`` measurements.

        The adapter's entry point: it holds priced operators, not structures,
        and which regime each one is in is a question about the operator and
        its scope rather than something a caller should decide. Operators whose
        regime cannot be established are recorded as refusals against their own
        name, so they are reported rather than dropped.
        """
        grouped: dict = {}
        model_refusals: dict = {}
        for op, seconds, source, scope in observations:
            structure = structure_of(op)
            # Geometry folded in before anything is grouped: a law over 4 KV
            # heads and a law over 8 are two laws, and grouping them by regime
            # and deployment alone would fit one curve through both.
            scope = scoped(op, scope, structure)
            regime = regime_of(op, structure, scope)
            if isinstance(regime, Refusal):
                model_refusals.setdefault(op.get("name", "?"), regime)
                continue
            grouped.setdefault(regime.name, []).append(
                (structure, seconds, source, scope))
        model = cls.from_observations(grouped, strict=strict,
                                      min_residual_df=min_residual_df)
        for name, refusal in model_refusals.items():
            model.refusals.setdefault(name, refusal)
        return model

    def price(self, op: dict, scope=None):
        """Seconds for this call, or a `Refusal` naming what is missing.

        ``scope`` is the requesting deployment's own declared scope. It selects
        the fit: a law measured under one static scope does not answer for
        another, and an undeclared request does not match a known fit.
        """
        structure = structure_of(op)
        if structure is None:
            return Refusal("this operator records no ragged structure")
        # The same fold as at fit time, so the geometry this call actually has
        # is what selects the law rather than being ignored at prediction.
        scope = scoped(op, scope, structure)
        regime = regime_of(op, structure, scope)
        if isinstance(regime, Refusal):
            return regime
        fit, why = self._fit_for(regime, scope)
        if fit is None:
            return why
        values = features_for(regime, structure, scope)
        if isinstance(values, Refusal):
            return values
        index = {name: i for i, name in enumerate(regime.features)}
        for name in fit.pinned:
            if values[index[name]]:
                return Refusal(
                    "%s: this call has %g %s and every measurement behind this "
                    "law had none, so what that work costs has not been "
                    "measured. It is unsupported rather than free"
                    % (regime.name, values[index[name]], name),
                    missing=(name,))
        values = [values[index[name]] for name in fit.features]
        outside = _outside_domain(fit, values)
        if outside:
            return Refusal(
                "%s: %s is outside the measured range %s, and this law is a "
                "fit over what was measured rather than a claim about "
                "everywhere" % (regime.name, outside[0], outside[1]))
        seconds = fit.predict(values)
        if not (seconds > 0 and math.isfinite(seconds)):
            return Refusal(
                "%s: the law returns %r for this structure, which is not a "
                "duration" % (regime.name, seconds))
        return seconds



    def fit_for(self, regime_name: str, scope=None, op: Optional[dict] = None):
        """``(fit, None)`` for this regime and scope, or ``(None, Refusal)``.

        ``op``, when given, folds that call's static operand geometry into the
        scope exactly as `price` does. A caller that asks about a law for a
        particular call has to ask under the same key the price is selected
        by, or it is told the law is unidentifiable when the truth is that it
        asked about a different deployment's heads.

        Public because a caller that has to say more about a modelled price
        than the number -- which measurements are behind it, what they were
        served by -- needs the fit itself, and reaching into the private
        selector to get it is how two callers end up selecting differently.
        """
        regime = REGIMES.get(regime_name)
        if regime is None:
            return None, Refusal("%s is not a known regime" % regime_name)
        if op is not None:
            scope = scoped(op, scope)
        return self._fit_for(regime, scope)

    def describe_fit(self, regime_name: str, scope=None) -> str:
        """The law a price at this regime and scope came from, in words.

        Attached to every modelled record, so a reader of a prediction can see
        how many independent points are behind it, how far it misses them, and
        which subdomain it was measured in -- without going back to the fit.
        """
        fit, why = self.fit_for(regime_name, scope)
        if fit is None:
            return why.reason
        return fit.describe()

    def _fit_for(self, regime, scope):
        """The fit for this regime whose scope the request matches."""
        candidates = [(label, fit) for label, fit in self.fits.items()
                      if fit.regime.name == regime.name]
        if not candidates:
            known = [r for label, r in self.refusals.items()
                     if label.split(" @ ")[0] == regime.name]
            return None, Refusal(
                "%s has no fitted law: %s" % (
                    regime.name,
                    known[0].reason if known
                    else "no measurement reached it"))
        matched = [(label, fit) for label, fit in candidates
                   if _scope_matches(fit.scope, scope) is None]
        if len(matched) == 1:
            return matched[0][1], None
        if not matched:
            differs = {label: _scope_matches(fit.scope, scope)
                       for label, fit in candidates}
            return None, Refusal(
                "%s is fitted, but not for this deployment: %s. A law "
                "measured under one static scope is not a price under "
                "another" % (regime.name,
                             "; ".join("%s differs on %s" % (label, field)
                                       for label, field in
                                       sorted(differs.items()))))
        return None, Refusal(
            "%s has %d fits whose scope this request matches (%s); the "
            "request does not say which deployment it is"
            % (regime.name, len(matched),
               ", ".join(sorted(label for label, _fit in matched))))

    def coverage(self) -> dict:
        """What is modelled and what is not, for a report to state plainly."""
        return {
            "fitted": {name: fit.describe() for name, fit in
                       sorted(self.fits.items())},
            "refused": {name: refusal.reason for name, refusal in
                        sorted(self.refusals.items())},
            "strict": self.strict,
        }


def _outside_domain(fit, values):
    """Which feature, if any, sits outside the measured hull, and its range."""
    for index, name in enumerate(fit.features):
        column = [row[index] for row in fit.domain]
        low, high = min(column), max(column)
        if values[index] < low or values[index] > high:
            return name, "[%g, %g]" % (low, high)
    return None
