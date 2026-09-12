"""The parts of a production step that are not the body and not the head.

`ModelRunner.forward` is four things (model_runner.py:3409): advancing forward
variables, `prepare_model` -- input preparation and its H2D staging copies --
`run_model`, which is the body and `compute_logits`, and `postprocess`, which
is the sampler, any logprobs, the TP broadcast of the sampled ids and the
sampled-id enqueue. A cost model composed of a body graph and a head graph
covers `run_model` and nothing else, so a prediction made from those two alone
is a prediction of a strict subset of the step.

This supplies the rest, and supplies it the way the rest is actually shaped:

* **Postprocess** is dominated by the sampler over `[sequences, padded_vocab]`,
  and at this vocabulary the region looks latency-bound rather than size-bound:
  the four prefill rows sit at 0.1045 ms over one sequence and 0.1496 ms over
  sixteen while the 255 decode rows sit at 0.1019 ms over thirty-two.

  That is why a constant is used, and it is also the limit of what supports it.
  The capture holds decode steps at **one** sequence count, 32. Flatness across
  decode sequence counts is not measured here; it is inferred from the prefill
  rows, which are a different step kind. `decode_sequences` is therefore (32,)
  and every other decode batch is refused -- not because the constant is
  believed to fail there, but because nothing has looked. Extending it needs a
  TP1 source capture driven at several concurrencies, not a wider domain
  tuple.

* **Preparation** is `prepare_model` plus whatever device idle the host leaves
  between the regions, which the outer event pair contains and the inner ones
  do not. The two are not separable by event pairs at all -- an idle device
  records nothing -- so they are carried together, named as a sum, rather than
  split on an assumption.

* **The TP broadcast** is the one thing postprocess does at TP>1 that it does
  not do at TP1: `get_tp_group().broadcast(sampled_tokens, src=0)` at
  model_runner.py:3313, taken whenever `is_deferred_out`, which is
  `pipeline_parallel_size == 1` (model_runner.py:189) and therefore always
  here. It is an independently measured standalone primitive, not a fitted
  residual: `agent_scratch/g4/bcast_probe.py` on the real group.

Everything else in postprocess runs on shapes that do not change with TP,
because `compute_logits` has already all-gathered the vocabulary shards back to
full width before postprocess sees them. That is why a region calibrated at TP1
transfers to TP2/TP4 with one added term rather than being re-fitted -- and it
is a claim about these regions at these widths, not a general one.

Outside the calibrated domain this refuses. A region model that answers
anywhere is a region model that has stopped being a measurement.

Two profiles live here, and the older one is not superseded.
`SOURCE_27B_TP1` is what the frozen TP2 and TP4 transfer reports were computed
from, so editing it would silently move predictions that have already been
made and checked; it stays as it was measured. `SOURCE_27B_TP1_CONC` is a
second, separately versioned profile from a second pair of captures, covering
fourteen decode concurrencies instead of one and keyed by the capture rung
rather than the sequence count. New work should ask the newer one and get a
refusal where it has not looked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: The sequence slot a pooled prefill anchor group is keyed under. Negative so
#: it can never collide with a real sequence count, and rendered by
#: `_seqs_label` so a refusal never reads "minus-one sequences".
POOLED_SEQS = -1


@dataclass(frozen=True)
class Measured:
    """One region's measured cost, with the spread it was measured over.

    `low` and `high` are not error bars derived from a model; they are the
    percentiles actually observed, so a prediction can be quoted as a band
    without inventing one.
    """

    seconds: float
    low: float
    high: float
    samples: int
    how: str

    @classmethod
    def zero_work(cls, why: str) -> "Measured":
        """A region the engine is known not to run for this shape.

        Not a measurement that came out small, and not a gap: a declaration
        that the work does not happen. A middle chunk of a long prompt samples
        no token, so the runner skips postprocess entirely.
        """
        return cls(seconds=0.0, low=0.0, high=0.0, samples=0, how=why)

    def describe(self) -> str:
        return (f"{self.seconds * 1e3:.4f} ms "
                f"[{self.low * 1e3:.4f}, {self.high * 1e3:.4f}] "
                f"n={self.samples}, {self.how}")


@dataclass(frozen=True)
class RunnerRegions:
    """Postprocess and preparation for one deployment's calibrated domain."""

    postprocess_decode: Measured
    postprocess_prefill: Measured
    prepare_decode: Measured
    prepare_prefill: Measured
    tp_broadcast: Measured
    #: Sequence counts and token extents the numbers above were measured at.
    #: A shape outside these is refused rather than extrapolated to.
    decode_sequences: tuple
    prefill_sequences: tuple
    prefill_tokens: tuple
    topologies: tuple
    provenance: str = ""

    def refusal(self, shape) -> Optional[str]:
        """Why this shape is outside the calibration, or None if it is inside."""
        tp = int((dict(shape.topology) if shape.topology else {}).get("tp", 1))
        if tp not in self.topologies:
            return (f"tp={tp} is outside the measured widths "
                    f"{list(self.topologies)}")
        seqs = len(shape.num_scheduled_tokens)
        if shape.num_prefill_tokens:
            if seqs not in self.prefill_sequences:
                return (f"prefill over {seqs} sequences, measured only at "
                        f"{list(self.prefill_sequences)}")
            lo, hi = self.prefill_tokens
            if not lo <= shape.total_tokens <= hi:
                return (f"prefill of {shape.total_tokens} tokens, measured "
                        f"only over [{lo}, {hi}]")
            return None
        if seqs not in self.decode_sequences:
            return (f"decode over {seqs} sequences, measured only at "
                    f"{list(self.decode_sequences)}")
        return None

    def breakdown(self, shape) -> dict:
        """Each region's seconds for this shape, by name.

        Raises if the shape is outside the domain: a caller that wanted a
        partial answer should not have supplied a region model.
        """
        why = self.refusal(shape)
        if why is not None:
            raise ValueError(f"no measured region for this shape: {why}")
        prefill = bool(shape.num_prefill_tokens)
        tp = int((dict(shape.topology) if shape.topology else {}).get("tp", 1))
        out = {
            "<postprocess>": (self.postprocess_prefill if prefill
                              else self.postprocess_decode).seconds,
            "<prepare>": (self.prepare_prefill if prefill
                          else self.prepare_decode).seconds,
        }
        if tp > 1:
            out["<tp-broadcast>"] = self.tp_broadcast.seconds
        return out

    def seconds(self, shape) -> float:
        return sum(self.breakdown(shape).values())

    def band(self, shape) -> tuple:
        """(low, high) over the same regions, from the observed spreads."""
        why = self.refusal(shape)
        if why is not None:
            raise ValueError(f"no measured region for this shape: {why}")
        prefill = bool(shape.num_prefill_tokens)
        tp = int((dict(shape.topology) if shape.topology else {}).get("tp", 1))
        parts = [self.postprocess_prefill if prefill
                 else self.postprocess_decode,
                 self.prepare_prefill if prefill else self.prepare_decode]
        if tp > 1:
            parts.append(self.tp_broadcast)
        return (sum(p.low for p in parts), sum(p.high for p in parts))

    def describe(self) -> str:
        return "\n".join([
            f"postprocess decode : {self.postprocess_decode.describe()}",
            f"postprocess prefill: {self.postprocess_prefill.describe()}",
            f"prepare+idle decode : {self.prepare_decode.describe()}",
            f"prepare+idle prefill: {self.prepare_prefill.describe()}",
            f"tp broadcast        : {self.tp_broadcast.describe()}",
            f"domain              : decode {list(self.decode_sequences)} seqs, "
            f"prefill {list(self.prefill_sequences)} seqs over "
            f"{list(self.prefill_tokens)} tokens, tp {list(self.topologies)}",
            f"provenance          : {self.provenance}",
        ])


#: The 27B deployment's regions, from its own capture.
#:
#: `agent_scratch/g4/cap_subspan/capture_steps.jsonl`: 260 steps of the
#: Qwen3.8-27B TP1 server -- 255 decodes of 32 sequences and 5 prefills -- with
#: `span_seconds.run_model` and `span_seconds.postprocess` recorded by event
#: pairs inside the outer one. Preparation is the remainder, `seconds` minus
#: those two.
#:
#: The first prefill row is excluded. It is the first use of that shape and
#: reads 7.071 s against 4.47-4.98 s for the others, with a 2.047 ms remainder
#: against 0.759-1.211 ms; the acceptance protocol is explicitly warmed, so a
#: cold first-use row is not part of a warm model and is not relabelled as one.
#:
#: The broadcast is not from this capture at all -- TP1 never executes it. It is
#: `agent_scratch/g4/bcast_probe.py` on the real two- and four-rank groups,
#: which measured 26.5-29.0 us flat across 1-32 sequences, int32 and int64, and
#: both widths; the single figure is the TP2 int64 32-sequence case.
SOURCE_27B_TP1 = RunnerRegions(
    postprocess_decode=Measured(
        seconds=1.019e-4, low=1.007e-4, high=1.038e-4, samples=255,
        how="p50 [p10,p90] of span_seconds.postprocess, 32-sequence decodes"),
    postprocess_prefill=Measured(
        seconds=1.496e-4, low=1.127e-4, high=1.502e-4, samples=4,
        how="p50 [min,max] of the four warm prefill rows"),
    prepare_decode=Measured(
        seconds=1.314e-4, low=1.288e-4, high=1.348e-4, samples=255,
        how="p50 [p10,p90] of seconds - run_model - postprocess"),
    prepare_prefill=Measured(
        seconds=1.1851e-3, low=7.590e-4, high=1.2108e-3, samples=4,
        how="p50 [min,max] of the four warm prefill rows"),
    tp_broadcast=Measured(
        seconds=2.86e-5, low=2.65e-5, high=2.90e-5, samples=8,
        how="bcast_probe.py medians over {int32,int64} x {1,4,16,32} at tp2/tp4"),
    decode_sequences=(32,),
    prefill_sequences=(15, 16),
    prefill_tokens=(15360, 16384),
    topologies=(1, 2, 4),
    provenance=("cap_subspan capture of the Qwen3.8-27B TP1 server on node 18 "
                "(runner.py 82800aad3fa9, predict.py a913f289b09c), plus "
                "bcast_probe.py standalone group measurements; no logprobs "
                "requested, no speculative decoding, PP=1"),
)


@dataclass(frozen=True)
class BucketedRunnerRegions:
    """The same two regions, keyed by what the engine actually ran.

    `RunnerRegions` above carries one decode number because its capture held
    one decode concurrency. Driven at fourteen instead, preparation is not one
    number: it is not even monotone in the batch. Two sequences cost 0.1138 ms
    and eight cost 0.1072 ms, so a model rising with batch size would be wrong
    in the wrong direction over a range serving traffic actually visits.

    What it does track is the pair `(capture_bucket, padded)` -- the rung the
    step replayed at, and whether that rung was wider than the batch. That
    pairing is not a fit: `ForwardMode.decide` selects the smallest rung at
    least the unified batch size, attention metadata is built at the rung, and
    the engine stamped its own `capture_bucket` on all 5 375 measured decode
    steps in agreement with that rule at every one of the fourteen sizes.

    Padded cells read above their exact neighbour at the two wide rungs --
    0.1239 against 0.1148 at sixteen, 0.1381 against 0.1256 at thirty-two --
    and that is *consistent with* the padded-row contract costing something to
    fill (`context_lens=0`, `slot_mapping=-1`, repeated `kv_indptr`, zeroed
    `input_ids`). It is not a measurement of padding-fill cost: nothing here
    separated that work from the rest of preparation, and at rung four the
    padded cell reads marginally *below* the exact one. The cells are a lookup
    over what was observed, and the ordering between them is reported, not
    explained.

    Within a padded cell the active size is pooled across the sizes measured in
    it, spreading 3.9% at rung sixteen and 2.0% at rung thirty-two, in neither
    case monotone in the size. The quoted band covers that spread. Pooling is
    the claim that the cell is the grouping the data supports -- not that the
    active size provably cannot matter at all inside one.

    Two boundaries this refuses at, both of them real:

    * **Context.** Every burst ran 1024-token prompts to 128 outputs, so the
      whole table sits between 1025 and 1152 tokens of history. Preparation has
      a term that grows with history -- `pack_rows` copies
      `ceil(context/block_size)` int32 per sequence -- but at 65-72 int32 a row
      it is below the ~500-int32 threshold that function's own docstring gives
      as where the bytes start to matter, so this calibration never exercised
      it. The cc-traces acceptance corpus reaches 109 741 tokens, about 6 859
      int32 a row, an order of magnitude past that threshold. The other two
      terms -- the `dst[:n_rows] = 0` memset and the `copy_to_gpu(bs)` upload
      -- are `rows x block_table_cols`, and `block_table_cols` is
      `max_num_blocks_per_seq // block_ratio`, a configuration constant rather
      than the live context. So the rise measured here is row-count driven and
      the history-driven term is the one left unmeasured. Extending past 1152
      needs a TP1 source capture at long context, not a wider tuple here.

    * **Eager.** `capture_bucket=None` means nothing was replayed. Every row in
      this table is a captured replay, so a step that ran eager is outside the
      calibration rather than at its first rung.
    """

    #: Postprocess over decode, one constant: measured flat from one sequence
    #: to thirty-two, which `RunnerRegions` could only infer from prefill rows.
    postprocess_decode: Measured
    #: `(capture_bucket, padded) -> Measured` for preparation over decode.
    prepare_decode_cells: tuple
    #: Prefill and the broadcast are unchanged measurements, carried from the
    #: profile below rather than re-declared, so one capture backs one number.
    #: The scalars are what a source uses when it has no prefill cells; where
    #: it has them they are the fallback for nothing, and the cells decide.
    postprocess_prefill: Measured
    prepare_prefill: Measured
    tp_broadcast: Measured
    #: Inclusive bounds on every request's history, in tokens.
    decode_context: tuple
    prefill_sequences: tuple
    prefill_tokens: tuple
    topologies: tuple
    #: The engine's capture ladder, for callers resolving a rung themselves.
    #: Not used to fill in a shape's missing bucket -- see `refusal`.
    capture_sizes: tuple = ()
    #: ``(sequences, total_tokens) -> Measured``, the prefill analogue of
    #: `prepare_decode_cells`. Empty means this source has none and the scalar
    #: above is used with the `prefill_sequences`/`prefill_tokens` domain, which
    #: is every source written before these existed.
    #:
    #: They exist for the same reason the decode cells do: the term is not a
    #: function of one number and is not monotone in the batch. Preparation over
    #: one sequence of 640 tokens is 0.39 ms; over two sequences of a full
    #: prompt each it is 3.9 ms; over fifteen and sixteen it is 1.19 ms again.
    #: A single published scalar cannot represent a tenfold difference between
    #: measured cells, and a range widened to span them would claim every token
    #: count in between on the strength of a term measured at neither end.
    #:
    #: A populated source is therefore a **lookup, not a domain**: a shape is
    #: priced when its exact cell was measured and refused otherwise. The
    #: intermediate cells are not unknown-but-probably-fine, they are unknown.
    prepare_prefill_cells: tuple = ()
    #: The same keys, for postprocess. Declared separately rather than paired
    #: so a source can carry one and not the other without a placeholder.
    postprocess_prefill_cells: tuple = ()
    #: Additional measured points, same key shape as the cells above, that a
    #: source offers as **interpolation anchors** rather than as published
    #: cells.
    #:
    #: The distinction is provenance, not arithmetic. The cells are the stanzas
    #: that have been reviewed and quoted elsewhere, and they are preserved
    #: byte for byte -- where a cell and an anchor share a key the cell wins,
    #: so adding anchors can never move a number that has already been
    #: reported. The anchors are the rest of the same captures, admitted under
    #: the same protocol, and they exist because a chunked prefill's final
    #: chunk is an arbitrary remainder: run 7 died on `(1, 9216, True)` after
    #: 104 priced steps, and no campaign will enumerate every tail.
    #:
    #: With anchors present a shape is answered by **linear interpolation in
    #: token count, within one `(sequences, produces_output)` group, and only
    #: inside that group's measured span**. Never across sequence counts:
    #: preparation over two sequences totalling 15360 tokens is 3.9 ms against
    #: 0.78 ms over one sequence of 15232, five times apart at the same token
    #: count, so the sequence count is a separate model rather than an axis to
    #: slide along. Never across `produces_output`: a middle chunk skips
    #: postprocess entirely, so the two are different work.
    #:
    #: An interpolated value is an APPROXIMATION between two measurements. Its
    #: band is not interpolated -- it is the union of the two bracketing
    #: anchors' observed [min, max], so a quoted band is never tighter than the
    #: evidence on either side of it. A group holding one anchor interpolates
    #: nothing and answers that token count alone.
    prepare_prefill_anchors: tuple = ()
    #: The same, for postprocess.
    postprocess_prefill_anchors: tuple = ()
    #: Inclusive `(lo, hi)` sequence counts whose prefill anchors share ONE
    #: token axis, keyed under `POOLED_SEQS`. Empty -- every source published
    #: before this existed -- keeps the strict per-sequence-count grouping.
    #:
    #: This is a deliberate, narrow relaxation of the rule three fields above,
    #: and it is not a general licence to slide along the sequence axis. That
    #: rule was written on the one- and two-sequence evidence, where 15360
    #: tokens over two sequences is five times 15232 over one; pooling those
    #: would be indefensible and this field does not, because a pooled range
    #: starting at 3 leaves 1 and 2 as their own groups.
    #:
    #: What licenses pooling above that is measurement, not convenience: at a
    #: fixed token count the spread ACROSS sequence counts is 2.4% at 16384
    #: (4 and 32 sequences) and 9.6% at 12288 (3, 12 and 24), while across
    #: token counts the same term moves 6.8x. Covering 3..32 as separate
    #: groups would need roughly sixty campaigns to say something the evidence
    #: says is not there.
    prefill_pooled_sequences: tuple = ()
    version: str = ""
    provenance: str = ""

    def _cells(self) -> dict:
        return {(int(b), bool(p)): m for (b, p), m in self.prepare_decode_cells}

    def _prefill_cells(self, which: tuple) -> dict:
        return {(int(s), int(t), bool(o)): m for (s, t, o), m in which}

    def _prefill_table(self, cells: tuple, anchors: tuple) -> dict:
        """The anchors, then the published cells written over them.

        The order is the guarantee: a key that is both an anchor and a cell
        keeps the cell's Measured exactly, so a source can gain anchors without
        any already-reported number moving.
        """
        table = self._prefill_cells(anchors)
        table.update(self._prefill_cells(cells))
        return table

    @staticmethod
    def _prefill_span(table: dict, seqs: int, produces: bool) -> list:
        """The token counts measured at this sequence count and output status."""
        return sorted(t for (s, t, o) in table
                      if s == seqs and o == produces)

    def _pool(self, seqs: int) -> int:
        """The group `seqs` is looked up under: itself, or the pooled slot."""
        if self.prefill_pooled_sequences:
            lo, hi = self.prefill_pooled_sequences
            if lo <= seqs <= hi:
                return POOLED_SEQS
        return seqs

    def _seqs_label(self, seqs: int) -> str:
        """How a group is named in a refusal or a `how` string."""
        if seqs == POOLED_SEQS and self.prefill_pooled_sequences:
            lo, hi = self.prefill_pooled_sequences
            return f"the pooled {lo}..{hi}-sequence group"
        return f"{seqs} sequence(s)"

    def _interpolates(self) -> bool:
        """Whether this source asked for interpolation at all.

        A source that declares only cells keeps the exact-lookup behaviour it
        was published with, to the number: `source-27b-tp1-prefill-cells` holds
        640 and 15232 at one sequence, and the arrival of an interpolating
        sibling must not quietly turn its 9216 refusal into an answer. Opting
        in is declaring an anchor.
        """
        return bool(self.prepare_prefill_anchors
                    or self.postprocess_prefill_anchors)

    def _prefill_at(self, table: dict, key: tuple) -> Measured:
        """This cell, exactly if measured, otherwise between its neighbours.

        Callers reach here only after `refusal` has passed, so either the key
        is present or this source interpolates and the key is inside its span.
        """
        if key in table:
            return table[key]
        if not self._interpolates():
            raise KeyError(
                f"prefill cell {key} is absent and this source declares no "
                "interpolation anchors; `refusal` should have rejected it")
        seqs, tokens, produces = key
        span = self._prefill_span(table, seqs, produces)
        lo = max(t for t in span if t <= tokens)
        hi = min(t for t in span if t >= tokens)
        left, right = table[(seqs, lo, produces)], table[(seqs, hi, produces)]
        frac = (tokens - lo) / float(hi - lo)
        return Measured(
            seconds=left.seconds + frac * (right.seconds - left.seconds),
            # The union, not an interpolated band. Both neighbours were
            # measured and neither bounds the other, so the honest interval
            # around a point between them is the one that contains both.
            low=min(left.low, right.low),
            high=max(left.high, right.high),
            samples=left.samples + right.samples,
            how=(f"INTERPOLATED linearly in token count between the measured "
                 f"{lo} ({left.seconds:.6e} s) and {hi} "
                 f"({right.seconds:.6e} s) at {self._seqs_label(seqs)} "
                 f"{'producing a token' if produces else 'producing none'}; "
                 f"the band is the union of both anchors' observed ranges, "
                 f"not a narrower interpolated one. Approximation between two "
                 f"measurements, not a measurement"),
        )

    @staticmethod
    def _produces_output(shape) -> bool:
        """Whether this prefill step samples a token.

        A middle chunk of a long prompt does not: the runner skips postprocess
        for it entirely, so its postprocess is *zero work* and not a small
        measurement. Averaging a middle chunk with a final one would put a
        term on a step that never ran it, which is why this is part of the key
        rather than a footnote -- the same role `padded` plays for decode.
        """
        produces = getattr(shape, "produces_output", None)
        if produces is not None:
            return bool(produces)
        # Older shapes do not carry it. A prefill whose scheduled tokens finish
        # every sequence's prompt is a final chunk.
        ctx = tuple(shape.context_lens or ())
        sched = tuple(shape.num_scheduled_tokens or ())
        return bool(ctx) and len(ctx) == len(sched)

    def _prefill_key(self, shape) -> tuple:
        # Every lookup and every refusal goes through here, so pooling is
        # applied once, at the key, rather than at each of the five sites that
        # build a span or quote one.
        return (self._pool(len(shape.num_scheduled_tokens)),
                int(shape.total_tokens), self._produces_output(shape))

    def _prefill_term(self, shape, cells: tuple, scalar: Measured,
                      anchors: tuple = ()) -> Measured:
        """The cell for this shape, or the scalar where a source has no cells.

        With anchors the exact cell still wins; only a token count nobody
        measured is interpolated, and `refusal` has already established that it
        lies between two that were.
        """
        if not cells and not anchors:
            return scalar
        return self._prefill_at(self._prefill_table(cells, anchors),
                                self._prefill_key(shape))

    def bucket_for(self, sequences: int):
        """The rung `ForwardMode.decide` would pick, or None above the ladder.

        Offered for callers that resolve the ladder themselves -- a ladder
        derivation, a shape list being written. `refusal` uses it only to check
        a rung a shape has already declared, never to supply one it has not: a
        shape says which rung it ran at, and a region model that guesses one
        has stopped reading the capture.
        """
        for size in self.capture_sizes:
            if size >= sequences:
                return size
        return None

    def refusal(self, shape) -> Optional[str]:
        """Why this shape is outside the calibration, or None if it is inside."""
        tp = int((dict(shape.topology) if shape.topology else {}).get("tp", 1))
        if tp not in self.topologies:
            return (f"tp={tp} is outside the measured widths "
                    f"{list(self.topologies)}")
        seqs = len(shape.num_scheduled_tokens)
        if shape.num_prefill_tokens:
            if self.prepare_prefill_cells or self.prepare_prefill_anchors:
                # A lookup where there are only cells; a lookup with bounded
                # interpolation where the source also supplies anchors.
                # Preparation is not monotone in the batch, so what is claimed
                # between two anchors is claimed only between *adjacent*
                # measurements of the same sequence count and output status,
                # and never outside their span.
                cells = self._prefill_table(self.prepare_prefill_cells,
                                            self.prepare_prefill_anchors)
                key = self._prefill_key(shape)
                span = self._prefill_span(cells, key[0], key[2])
                produced = 'a token' if key[2] else 'no token'
                if not self._interpolates():
                    # Cells only: the published lookup, cell by cell. This is
                    # the branch run 7 died in, and it must keep dying there.
                    if key not in cells:
                        return (f"prefill of {key[1]} tokens over "
                                f"{self._seqs_label(key[0])}, producing "
                                f"{produced}, was not "
                                "measured; the measured token counts for that "
                                f"group are {span}")
                    post = self._prefill_cells(self.postprocess_prefill_cells)
                    if key[2] and key not in post:
                        return (f"prefill cell {key} has a preparation "
                                "measurement and no postprocess one, and this "
                                "step samples a token, so its postprocess is "
                                "missing rather than zero")
                    return None
                if not span:
                    groups = sorted({(s, o) for (s, _t, o) in cells})
                    return (f"prefill over {self._seqs_label(key[0])} producing "
                            f"{produced} was not measured at any token count; "
                            f"the measured (sequences, produces_output) "
                            f"groups are {groups}")
                if not span[0] <= key[1] <= span[-1]:
                    return (f"prefill of {key[1]} tokens over "
                            f"{self._seqs_label(key[0])}, producing "
                            f"{produced}, is outside "
                            f"the measured token span [{span[0]}, {span[-1]}] "
                            f"for that group; the anchors there are {span}")
                if key[2]:
                    # This step samples a token, so postprocess runs and must
                    # have been measured. Absent is a GAP here, not zero work:
                    # only an explicit `produces_output=False` licenses zero,
                    # and defaulting a sampling step to zero would silently
                    # drop a region the step really pays.
                    post = self._prefill_table(
                        self.postprocess_prefill_cells,
                        self.postprocess_prefill_anchors)
                    post_span = self._prefill_span(post, key[0], key[2])
                    if not post_span or not (post_span[0] <= key[1]
                                             <= post_span[-1]):
                        return (f"prefill cell {key} has a preparation "
                                "measurement and no postprocess one, and this "
                                "step samples a token, so its postprocess is "
                                "missing rather than zero; the postprocess "
                                f"anchors for that group are {post_span}")
                return None
            if seqs not in self.prefill_sequences:
                return (f"prefill over {seqs} sequences, measured only at "
                        f"{list(self.prefill_sequences)}")
            lo, hi = self.prefill_tokens
            if not lo <= shape.total_tokens <= hi:
                return (f"prefill of {shape.total_tokens} tokens, measured "
                        f"only over [{lo}, {hi}]")
            return None
        bucket = shape.capture_bucket
        if bucket is None:
            return ("this decode step replayed no captured graph; every "
                    "measured row here is a replay, so an eager step is "
                    "outside the calibration rather than at its first rung")
        if bucket < seqs:
            return (f"capture bucket {bucket} is narrower than the {seqs} "
                    "sequences scheduled, which the engine does not do")
        rung = self.bucket_for(seqs)
        if rung is not None and int(bucket) != rung:
            return (f"{seqs} sequences at capture bucket {bucket}: the ladder "
                    f"{list(self.capture_sizes)} replays {seqs} at {rung}, so "
                    f"this pairing did not occur in the capture and the cell "
                    f"for bucket {bucket} was filled by the batches that do "
                    "reach it, padded a different distance")
        cell = (int(bucket), bucket != seqs)
        if cell not in self._cells():
            measured = sorted((b, "padded" if p else "exact")
                              for b, p in self._cells())
            return (f"{seqs} sequences at capture bucket {bucket} is the "
                    f"{'padded' if cell[1] else 'exact'} cell of bucket "
                    f"{bucket}, measured only at {measured}")
        lo, hi = self.decode_context
        histories = [int(c) for c in shape.context_lens]
        if histories and not (lo <= min(histories) and max(histories) <= hi):
            return (f"decode over histories {min(histories)}-{max(histories)} "
                    f"tokens, measured only over [{lo}, {hi}]; preparation has "
                    "a term that grows with history and this capture did not "
                    "reach it")
        return None

    def _decode_prepare(self, shape) -> Measured:
        seqs = len(shape.num_scheduled_tokens)
        return self._cells()[(int(shape.capture_bucket),
                              shape.capture_bucket != seqs)]

    def _parts(self, shape) -> list:
        why = self.refusal(shape)
        if why is not None:
            raise ValueError(f"no measured region for this shape: {why}")
        prefill = bool(shape.num_prefill_tokens)
        tp = int((dict(shape.topology) if shape.topology else {}).get("tp", 1))
        if prefill:
            prepare = self._prefill_term(shape, self.prepare_prefill_cells,
                                         self.prepare_prefill,
                                         self.prepare_prefill_anchors)
            if self.prepare_prefill_cells or self.prepare_prefill_anchors:
                # Only an explicit "produces no token" licenses zero here.
                # `refusal` has already rejected a sampling cell whose
                # postprocess was never measured, so reaching this branch with
                # no postprocess coverage means the step genuinely samples
                # nothing.
                cells = self._prefill_table(self.postprocess_prefill_cells,
                                            self.postprocess_prefill_anchors)
                key = self._prefill_key(shape)
                covered = (key in cells if not self._interpolates()
                           else bool(self._prefill_span(cells, key[0], key[2])))
                if covered:
                    postprocess = self._prefill_at(cells, key)
                else:
                    postprocess = Measured.zero_work(
                        "this chunk samples no token, so the runner skips "
                        "postprocess entirely")
            else:
                postprocess = self.postprocess_prefill
        else:
            prepare = self._decode_prepare(shape)
            postprocess = self.postprocess_decode
        parts = [("<postprocess>", postprocess), ("<prepare>", prepare)]
        # The broadcast is of the SAMPLED TOKEN
        # (`get_tp_group().broadcast(sampled_tokens, src=0)`), so a step that
        # samples nothing does not run it. A middle chunk of a long prompt is
        # exactly that, and charging it a broadcast would bill a collective
        # that never happened. Decode always samples.
        samples = (self._produces_output(shape) if prefill else True)
        if tp > 1 and samples:
            parts.append(("<tp-broadcast>", self.tp_broadcast))
        return parts

    def breakdown(self, shape) -> dict:
        """Each region's seconds for this shape, by name.

        Raises if the shape is outside the domain: a caller that wanted a
        partial answer should not have supplied a region model.
        """
        return {name: m.seconds for name, m in self._parts(shape)}

    def seconds(self, shape) -> float:
        return sum(self.breakdown(shape).values())

    def band(self, shape) -> tuple:
        """(low, high) over the same regions, from the observed spreads."""
        parts = [m for _, m in self._parts(shape)]
        return (sum(m.low for m in parts), sum(m.high for m in parts))

    def describe(self) -> str:
        lines = [f"version             : {self.version}",
                 f"postprocess decode : {self.postprocess_decode.describe()}",
                 f"postprocess prefill: {self.postprocess_prefill.describe()}"]
        for (bucket, padded), m in sorted(self.prepare_decode_cells,
                                          key=lambda kv: (kv[0][0], kv[0][1])):
            tag = f"bucket {bucket} {'padded' if padded else 'exact'}"
            lines.append(f"prepare+idle {tag:<18}: {m.describe()}")
        lines += [
            f"prepare+idle prefill: {self.prepare_prefill.describe()}",
            f"tp broadcast        : {self.tp_broadcast.describe()}",
            f"domain              : decode over histories "
            f"{list(self.decode_context)} tokens at the cells above, prefill "
            f"{list(self.prefill_sequences)} seqs over "
            f"{list(self.prefill_tokens)} tokens, tp {list(self.topologies)}",
        ]
        if self.prefill_pooled_sequences:
            lo, hi = self.prefill_pooled_sequences
            lines.append(
                f"pooled prefill group: sequence counts {lo}..{hi} share one "
                "token axis; 1 and 2 remain separate groups")
        lines.append(f"provenance          : {self.provenance}")
        return "\n".join(lines)


#: The 27B deployment's regions over fourteen decode concurrencies.
#:
#: `SOURCE_27B_TP1` above is **not** superseded and must not be edited: the
#: frozen TP2 and TP4 transfer reports were produced from it, and a prediction
#: already made does not improve by being recomputed. This is a second,
#: separately versioned profile from a second pair of captures.
#:
#: `agent_scratch/g4/cap_conc` and `agent_scratch/g4/cap_conc2`: the same TP1
#: Qwen3.8-27B server, knob for knob, driven at N in
#: [1,2,3,4,5,8,9,12,15,16,17,20,31,32]. Each N ran as three independent bursts
#: in a fixed interleaved order, so drift across a session cannot alias onto N;
#: the statistic is the p50 over all decode steps of the three, with p10/p90 as
#: the band; and a concurrency was admitted only if the three per-burst p50s
#: agreed within 5% of their median. All fourteen admitted, worst 0.86%.
#:
#: Steps were assigned to bursts by request cohort -- the engine's own request
#: counter, minted once per admitted request for the life of the server -- not
#: by file-line windows, which the measure file's lazy flush moves. Both
#: partitions were checked complete and disjoint before any statistic was
#: taken, and the preparation burst of each session was excluded as a set fact
#: about its request ids rather than by a row number.
#:
#: Prefill and the broadcast are `SOURCE_27B_TP1`'s own `Measured` objects.
#: Neither capture here ran a prefill shape in that domain and neither ran at
#: TP>1, so re-stating them would be copying, not measuring.
SOURCE_27B_TP1_CONC = BucketedRunnerRegions(
    postprocess_decode=Measured(
        seconds=1.026e-4, low=1.005e-4, high=1.061e-4, samples=5375,
        how="p50 [p10,p90] of span_seconds.postprocess over every decode step "
            "of all fourteen concurrencies -- flat from 1 to 32 sequences "
            "(0.1011-0.1046 ms by N), which the single-concurrency profile "
            "could only infer from prefill rows"),
    prepare_decode_cells=(
        ((1, False), Measured(
            seconds=1.037e-4, low=1.012e-4, high=1.069e-4, samples=384,
            how="p50 [p10,p90] of seconds - run_model - postprocess, N=1")),
        ((2, False), Measured(
            seconds=1.138e-4, low=1.112e-4, high=1.177e-4, samples=384,
            how="p50 [p10,p90], N=2")),
        ((4, True), Measured(
            seconds=1.074e-4, low=1.050e-4, high=1.105e-4, samples=384,
            how="p50 [p10,p90], N=3 padded to 4")),
        ((4, False), Measured(
            seconds=1.080e-4, low=1.052e-4, high=1.120e-4, samples=384,
            how="p50 [p10,p90], N=4")),
        ((8, True), Measured(
            seconds=1.100e-4, low=1.078e-4, high=1.134e-4, samples=384,
            how="p50 [p10,p90], N=5 padded to 8")),
        ((8, False), Measured(
            seconds=1.072e-4, low=1.046e-4, high=1.106e-4, samples=384,
            how="p50 [p10,p90], N=8")),
        ((16, True), Measured(
            seconds=1.239e-4, low=1.193e-4, high=1.281e-4, samples=1152,
            how="p50 [p10,p90] pooled over N=9,12,15 padded to 16; the three "
                "per-N p50s spread 3.9% and not monotone in N")),
        ((16, False), Measured(
            seconds=1.148e-4, low=1.126e-4, high=1.184e-4, samples=384,
            how="p50 [p10,p90], N=16")),
        ((32, True), Measured(
            seconds=1.381e-4, low=1.350e-4, high=1.414e-4, samples=1152,
            how="p50 [p10,p90] pooled over N=17,20,31 padded to 32; the three "
                "per-N p50s spread 2.0% and not monotone in N")),
        ((32, False), Measured(
            seconds=1.256e-4, low=1.240e-4, high=1.281e-4, samples=383,
            how="p50 [p10,p90], N=32; the last burst's final decode step was "
                "lost to the measure file's lazy flush, hence 383 not 384")),
    ),
    postprocess_prefill=SOURCE_27B_TP1.postprocess_prefill,
    prepare_prefill=SOURCE_27B_TP1.prepare_prefill,
    tp_broadcast=SOURCE_27B_TP1.tp_broadcast,
    decode_context=(1025, 1152),
    prefill_sequences=(15, 16),
    prefill_tokens=(15360, 16384),
    topologies=(1, 2, 4),
    capture_sizes=(1, 2, 4, 8, 16, 32),
    version="source-27b-tp1-conc/1",
    provenance=("cap_conc + cap_conc2 captures of the Qwen3.8-27B TP1 server "
                "on node 18 (runner.py 474ec80e0554c412, predict.py "
                "a913f289b09ca9dc, replay.py 02736b4601c4e0c7); prefill and "
                "broadcast terms carried unchanged from SOURCE_27B_TP1; no "
                "logprobs requested, no speculative decoding, PP=1, prefix "
                "caching off"),
)


#: The same ten cells, re-measured on a second server, with bands that survive
#: the restart.
#:
#: `/1` above is preserved exactly as published, including its truncated
#: `(32, exact)` cell. It is not corrected here and its numbers are not
#: adjusted: a profile that rewrites what it recorded is a profile no later
#: reader can check.
#:
#: What produced this one: `cap_conc3` re-ran `cap_conc`'s entire ladder --
#: N in [1,3,5,8,16,17,20,31,32], three bursts each, same interleaving, same
#: 5%-agreement admission rule, same preparation burst, same card, same
#: weights, same capture code -- on a second server process, and ended with a
#: short drain burst so the lazy flush could not eat the last step of the last
#: measured burst. It did not. All nine concurrencies were admitted (worst
#: per-burst spread 0.69%), every measurement cohort holds its full 128 decode
#: steps, and each one's context ladder runs 1025..1152 with no gaps.
#:
#: **The reason this exists is not the one lost row.** Two independent server
#: instances measuring the same seven cells disagree about prepare by +4.0% to
#: +5.0%, uniformly, across every cell and both padding states -- while
#: agreeing about postprocess to within 0.43%. Neither instance's own [p10,p90]
#: is wide enough to contain the other's median, so a band taken from one
#: server is a band that will be wrong the next time a server starts. Every
#: two-instance cell below carries the union of both instances' bands, which is
#: 3.7%-5.3% half-width, and says which instances measured it.
#:
#: The three cells cap_conc3 did not re-run -- `(2, exact)`, `(4, exact)` and
#: `(16, padded)`, which only `cap_conc2` measured -- keep their
#: single-instance bands and are marked. Their true between-instance width is
#: unmeasured, and on the evidence of the other seven it is likely wider than
#: what they state.
#:
#: What the second instance does reproduce is the shape. The level shifts
#: uniformly; the structure does not move. In particular the drop at the top
#: rung -- a batch of exactly 32 preparing faster than a batch of 17, 20 or 31
#: padded up to 32 -- is -9.01% on the first instance and -9.18% on the second.
SOURCE_27B_TP1_CONC_V2 = BucketedRunnerRegions(
    postprocess_decode=Measured(
        seconds=1.026e-4, low=1.006e-4, high=1.060e-4, samples=5376,
        how="p50 [p10,p90] of span_seconds.postprocess over every decode step "
            "of both capture sessions, fourteen concurrencies. The one region "
            "that reproduces across a restart: the first instance's own pooled "
            "p50 was 1.0216e-4, this is 1.0260e-4, +0.43%"),
    prepare_decode_cells=(
        ((1, False), Measured(
            seconds=1.079e-4, low=1.012e-4, high=1.121e-4, samples=384,
            how="p50 of seconds - run_model - postprocess at N=1 on the second "
                "instance; band is the union of both instances' [p10,p90], "
                "whose p50s differ by +4.09%")),
        ((2, False), Measured(
            seconds=1.138e-4, low=1.112e-4, high=1.177e-4, samples=384,
            how="p50 [p10,p90], N=2. ONE INSTANCE (cap_conc2 only): the "
                "between-instance width the other cells show is not in this "
                "band")),
        ((4, True), Measured(
            seconds=1.127e-4, low=1.050e-4, high=1.168e-4, samples=384,
            how="p50 at N=3 padded to 4, second instance; union band, "
                "+4.96% between instances")),
        ((4, False), Measured(
            seconds=1.080e-4, low=1.052e-4, high=1.120e-4, samples=384,
            how="p50 [p10,p90], N=4. ONE INSTANCE (cap_conc2 only)")),
        ((8, True), Measured(
            seconds=1.154e-4, low=1.078e-4, high=1.196e-4, samples=384,
            how="p50 at N=5 padded to 8, second instance; union band, "
                "+4.98% between instances")),
        ((8, False), Measured(
            seconds=1.116e-4, low=1.046e-4, high=1.152e-4, samples=384,
            how="p50 at N=8, second instance; union band, +4.18%")),
        ((16, True), Measured(
            seconds=1.239e-4, low=1.193e-4, high=1.281e-4, samples=1152,
            how="p50 pooled over N=9,12,15 padded to 16. ONE INSTANCE "
                "(cap_conc2 only); the three per-N p50s spread 3.9% and are "
                "not monotone in N")),
        ((16, False), Measured(
            seconds=1.201e-4, low=1.126e-4, high=1.242e-4, samples=384,
            how="p50 at N=16, second instance; union band, +4.64%")),
        ((32, True), Measured(
            seconds=1.442e-4, low=1.350e-4, high=1.485e-4, samples=1152,
            how="p50 pooled over N=17,20,31 padded to 32, second instance; "
                "union band, +4.40%")),
        ((32, False), Measured(
            seconds=1.309e-4, low=1.240e-4, high=1.336e-4, samples=384,
            how="p50 at N=32, second instance; union band, +4.21%. The full "
                "384 steps: this is the cell /1 recorded at 383, and the drain "
                "burst is why")),
    ),
    postprocess_prefill=SOURCE_27B_TP1.postprocess_prefill,
    prepare_prefill=SOURCE_27B_TP1.prepare_prefill,
    tp_broadcast=SOURCE_27B_TP1.tp_broadcast,
    decode_context=(1025, 1152),
    prefill_sequences=(15, 16),
    prefill_tokens=(15360, 16384),
    topologies=(1, 2, 4),
    capture_sizes=(1, 2, 4, 8, 16, 32),
    version="source-27b-tp1-conc/2",
    provenance=("cap_conc3 (the seven cells it re-ran, and the second instance "
                "of each) + cap_conc2 (the three it did not) + cap_conc (the "
                "first instance, for the union bands), Qwen3.8-27B TP1 on node "
                "18, runner.py 474ec80e0554c412 byte-identical across all "
                "three sessions; prefill and broadcast terms carried unchanged "
                "from SOURCE_27B_TP1; no logprobs requested, no speculative "
                "decoding, PP=1, prefix caching off"),
)


#: The region models a caller may ask for by name, and the only place that
#: mapping lives.
#:
#: A name rather than an import path because the callers are command lines --
#: `predict_step.py --regions`, and `--compass-oracle-option regions=...` on a
#: served run -- and a command line that can name any importable object can
#: name the wrong one. A typo here is an error; a typo in an import path is a
#: different region model, silently.
#:
#: `"none"` maps to `None` and means *no region model*, which is a real and
#: different claim: body plus head alone, priced without the runner's own
#: work. It is spelled out for the same reason the rest are -- so a prediction
#: that carries no prepare or postprocess term says so in the same field that
#: would have named the model.
#: The dev run's first step, and only that step.
#:
#: The authoritative unpaced run opens with one sequence of 640 tokens: all 62
#: requests arrived within 3 s, the barrier completed, and the first scheduled
#: batch was ``1 reqs, 640 new tokens``. Every source above refuses it twice --
#: ``prefill_sequences`` is membership-tested and holds (15, 16), and
#: ``prefill_tokens`` is range-tested over [15360, 16384] -- so the step the
#: run begins with had no region price while its body and head were fully
#: measured.
#:
#: This carries its own measured terms and declares support for that one cell.
#: It is deliberately NOT a widening of the profile above. ``prepare_prefill``
#: is one scalar per source, and the measurements do not permit one:
#:
#:     1 sequence,  640 tokens        3.9168e-4 s   single640, bulk-enqueued
#:     2 sequences, 15360 tokens      3.8609e-3 s   calib2seq lower
#:     2 sequences, 16384 tokens      3.8754e-3 s   calib2seq even
#:     2 sequences, 16384 tokens      3.9316e-3 s   calib2seq observed
#:     fifteen/sixteen, 15360-16384   1.1851e-3 s   cap_subspan, profile above
#:
#: Ten times between one sequence and two, and back down again at fifteen. A
#: source spanning those cells would have to carry a term per cell the way
#: ``prepare_decode_cells`` already does for decode, and there is no
#: ``prepare_prefill_cells``. Until there is, one source per measured prefill
#: cell is the honest shape: widening the tuples on the profile above would
#: claim support at token counts nobody measured, priced with a term measured
#: somewhere else.
#:
#: Decode, the broadcast and the context bound are carried from
#: ``SOURCE_27B_TP1_CONC_V2`` unchanged, so a run that mixes this first step
#: with the decodes that follow is priced by the same decode evidence as
#: before.
SOURCE_27B_TP1_PREFILL_1X640 = BucketedRunnerRegions(
    postprocess_decode=SOURCE_27B_TP1_CONC_V2.postprocess_decode,
    prepare_decode_cells=SOURCE_27B_TP1_CONC_V2.prepare_decode_cells,
    postprocess_prefill=Measured(
        seconds=1.0052e-4, low=9.956e-5, high=1.0092e-4, samples=3,
        how="upper median [min,max] of the 3 retained warm 1x640 prefill rows "
            "of pricing_coverage/single640, matching-shape warmup discarded "
            "by position. Prefill-region method, not the operator gate"),
    prepare_prefill=Measured(
        seconds=3.916815e-4, low=3.85378e-4, high=3.92064e-4, samples=3,
        how="upper median [min,max] of seconds - run_model - postprocess over "
            "the same 3 rows. The remainder carries prepare_model and "
            "whatever device idle the host leaves inside forward, which is "
            "why it is pacing sensitive: the same cell under HTTP pacing "
            "reads 1.1675e-3 s"),
    tp_broadcast=SOURCE_27B_TP1_CONC_V2.tp_broadcast,
    decode_context=SOURCE_27B_TP1_CONC_V2.decode_context,
    prefill_sequences=(1,),
    prefill_tokens=(640, 640),
    topologies=(1,),
    capture_sizes=SOURCE_27B_TP1_CONC_V2.capture_sizes,
    version="1x640-2026-09-12",
    provenance="pricing_coverage/single640: bulk-enqueued, matching-shape "
               "warmup, 3 retained repeats. Decode and broadcast carried from "
               "source-27b-tp1-conc-v2",
)


#: The prefill cells that have actually been measured, as a lookup.
#:
#: Keyed `(sequences, total_tokens, produces_output)`, the prefill analogue of
#: `prepare_decode_cells`. The third field is not decoration: a middle chunk of
#: a long prompt samples no token, so the runner skips postprocess for it
#: entirely, and pairing a middle chunk with a final chunk's postprocess would
#: charge a step for work it never did.
#:
#: Every cell below is three retained repeats of one shape from one campaign.
#: Nothing here is fitted and nothing between the cells is claimed.
#:
#: Where two cohorts share a key they are POOLED, not chosen between. The
#: ragged 2x16384 case and the two-full-prompt one differ by 1.4%; the 1x16384
#: middle chunks differ by 73.6% across the two histories they were measured
#: at. Both pooled ranges are under 0.008% of their step, and acceptance is
#: end-to-end throughput, TPOT and TTFT -- so a relative spread on a term worth
#: a thousandth of the step is not grounds for an indefinite per-history table,
#: and silently preferring one cohort would be worse than carrying both. Every
#: pooled cell names the histories and shapes it spans and its range covers
#: them all. The pooled value is an APPROXIMATION generalising over the pooled
#: cohorts, and low/high are the observed six-sample range -- neither is a
#: bound on anything unmeasured.
#:
#: Deliberately absent: **1 sequence x 1024**. Excluded on ROLE, not on spread
#: -- those three rows are the campaign.s own preparation and flush bursts
#: rather than a measured cell.
SOURCE_27B_TP1_PREFILL_CELLS = BucketedRunnerRegions(
    postprocess_decode=SOURCE_27B_TP1_CONC_V2.postprocess_decode,
    prepare_decode_cells=SOURCE_27B_TP1_CONC_V2.prepare_decode_cells,
    # Unused while the cells below are populated; kept so the dataclass is
    # complete and a reader sees which scalar was superseded.
    postprocess_prefill=SOURCE_27B_TP1_CONC_V2.postprocess_prefill,
    prepare_prefill=SOURCE_27B_TP1_CONC_V2.prepare_prefill,
    prepare_prefill_cells=(
        ((1, 640, True), Measured(
            seconds=3.916815e-4, low=3.85378e-4, high=3.92064e-4, samples=3,
            how="upper median [min,max] of seconds - run_model - postprocess, "
                "pricing_coverage/single640, bulk-enqueued with a "
                "matching-shape warmup discarded by position")),
        ((1, 15232, True), Measured(
            seconds=7.823394e-4, low=7.60528e-4, high=8.30608e-4, samples=3,
            how="the tail chunk that finishes the 63744-token prompt, one per "
                "burst of calib2seq observed; upper median [min,max]")),
        ((2, 15360, True), Measured(
            seconds=3.860857e-3, low=3.82871e-3, high=4.03999e-3, samples=3,
            how="calib2seq lower, two full 7680-token prompts co-scheduled; "
                "upper median [min,max]")),
        ((2, 16384, True), Measured(
            seconds=3.931546e-3, low=3.819904e-3, high=4.209002e-3, samples=6,
            how="calib2seq even and observed POOLED: two full 8192-token "
                "prompts, and a full 640-token prompt beside a 15744-token "
                "chunk. Same key, 1.4% apart, and the pooled range is 3.891e-4 "
                "s on a 5.3310 s step -- 0.0073%. Pooled rather than choosing "
                "one, so no cohort is silently preferred; upper median "
                "[min,max] over all six rows")),
        ((1, 16384, False), Measured(
            seconds=1.086426e-3, low=7.377930e-4, high=1.280762e-3, samples=6,
            how="the middle chunks of the 63744-token prompt, POOLED over the "
                "two histories they were measured at, 32128 and 48512 cached. "
                "73.6% apart relative and 5.430e-4 s on a 7.0196 s step, "
                "0.0077% of it: acceptance is end-to-end throughput, TPOT and "
                "TTFT, so a relative spread on a term worth a thousandth of "
                "the step is not grounds for a per-history table. Both "
                "histories are named and the range spans both")),
    ),
    postprocess_prefill_cells=(
        ((1, 640, True), Measured(
            seconds=1.0052e-4, low=9.956e-5, high=1.0092e-4, samples=3,
            how="span_seconds.postprocess, pricing_coverage/single640")),
        ((1, 15232, True), Measured(
            seconds=9.908e-5, low=9.852e-5, high=1.0080e-4, samples=3,
            how="span_seconds.postprocess, calib2seq observed tail chunk")),
        ((2, 15360, True), Measured(
            seconds=1.590810e-4, low=1.56721e-4, high=1.62161e-4, samples=3,
            how="span_seconds.postprocess, calib2seq lower")),
        ((2, 16384, True), Measured(
            seconds=1.100000e-4, low=1.09441e-4, high=1.13121e-4, samples=3,
            how="span_seconds.postprocess, calib2seq even")),
    ),
    tp_broadcast=SOURCE_27B_TP1_CONC_V2.tp_broadcast,
    decode_context=SOURCE_27B_TP1_CONC_V2.decode_context,
    # Unused while cells are populated; `refusal` consults the cells instead.
    prefill_sequences=(1, 2),
    prefill_tokens=(640, 16384),
    topologies=(1,),
    capture_sizes=SOURCE_27B_TP1_CONC_V2.capture_sizes,
    version="prefill-cells-2026-09-12",
    provenance="pricing_coverage/single640 and pricing_coverage/calib2seq, "
               "three retained repeats per cell. Decode, broadcast and the "
               "context bound carried from source-27b-tp1-conc-v2. TP1 only: "
               "nothing here is fitted to a TP2 or TP4 engine",
)


#: The same five cells, plus the rest of the same captures as anchors.
#:
#: Run 7 is why this exists. The replay priced 104 steps of the first-20
#: cc_pilot slice -- 84 at (1 req, 16384), 18 at (2 reqs, 16384), one at
#: (1 req, 640) -- and then died on request 19's final chunk, `(1, 9216,
#: True)`, with `no measured region for this shape`. A chunked prefill's tail
#: is `prompt mod chunk_budget`, an arbitrary remainder, so a five-cell lookup
#: does not merely have a hole: it refuses the last step of nearly every long
#: request, and the registered short workload (max 2560 tokens) matches no cell
#: at all. No campaign fixes that by adding cells.
#:
#: **Nothing here was re-measured and no published number moved.** The five
#: `prepare_prefill_cells` and four `postprocess_prefill_cells` above are
#: carried over byte for byte and still win at their own keys. What is added is
#: the rest of the TP1 source captures, extracted by
#: `agent_scratch/stage/region_anchors.py` under three stated admissions:
#:
#: 1. **TP1 source captures only.** A TP>1 postprocess contains the sampled-id
#:    broadcast this model prices separately, so pooling one would double count
#:    it. No target-engine observation is read, and no final workload is read.
#: 2. **Unpaced rows only.** Preparation is the remainder `seconds -
#:    run_model - postprocess` and carries whatever device idle the host leaves
#:    inside `forward`, so it moves with pacing -- ~3x at 1x640 between the
#:    HTTP campaign and the bulk one. `devstep1` settles that this is a
#:    per-row property rather than a campaign label: it alternates, one paced
#:    row then one unpaced row, on the same 1x8192 shape, 2.790911e-3 against
#:    1.124574e-3. The regime is therefore read from `gap_seconds` per row, cut
#:    at 0.10 s, and the populations are nowhere near that line -- unpaced rows
#:    run 0.0007..0.05 s and paced ones 0.17..3.2 s. The served replay enqueues
#:    back to back, so unpaced is the regime it runs in.
#: 3. **Cold first use excluded by position**, which is the prefill-region
#:    method's own rule.
#:
#: That yields six one-sequence final-chunk anchors, not two: 640, 1024, 7680,
#: 8192, 15232 and 15744. They are not monotone -- preparation climbs to
#: 1.15 ms at 7680 and falls back to 0.78 ms at 15232 -- which is exactly why
#: interpolation is restricted to *adjacent* anchors and never extrapolates.
#:
#: **What is still refused, deliberately.** Tails below 640 tokens: nothing in
#: any TP1 source capture is shorter, so the registered short workload's own
#: small final chunks have no evidence and must refuse until a campaign
#: measures them. One sequence above 15744 producing a token. Two sequences
#: below 2048. Three or more sequences at any token count -- those captures do
#: exist (3x3072 through 16x16384, every one at 1024 tokens per sequence) but
#: each is a single point that no trace shape will land on, so admitting them
#: would widen what this model claims without covering anything asked of it.
#:
#: An interpolated answer is an approximation between two measurements and says
#: so in its own `how`. For the shape that killed run 7 it is 1.049 ms, between
#: the 8192 and 15232 anchors, against a step of several seconds -- about 0.03%
#: of it. That is the honest size of this term and the reason a bounded
#: interpolation is preferable to an indefinite acquisition.
SOURCE_27B_TP1_PREFILL_INTERP = BucketedRunnerRegions(
    postprocess_decode=SOURCE_27B_TP1_CONC_V2.postprocess_decode,
    prepare_decode_cells=SOURCE_27B_TP1_CONC_V2.prepare_decode_cells,
    postprocess_prefill=SOURCE_27B_TP1_CONC_V2.postprocess_prefill,
    prepare_prefill=SOURCE_27B_TP1_CONC_V2.prepare_prefill,
    # Unchanged, and authoritative wherever they apply.
    prepare_prefill_cells=SOURCE_27B_TP1_PREFILL_CELLS.prepare_prefill_cells,
    postprocess_prefill_cells=(
        SOURCE_27B_TP1_PREFILL_CELLS.postprocess_prefill_cells),
    prepare_prefill_anchors=(
        ((1, 1024, True), Measured(
            seconds=5.223719e-4, low=4.24425e-4, high=8.820927e-4, samples=7,
            how="upper median [min,max] of the 7 unpaced 1x1024 final-chunk "
                "rows of cap_conc2, calib2seq (all three cases) and "
                "single640. POOLED across four captures, which is why the "
                "range is 2.1x wide: these are the same cell measured by "
                "different servers, and a band that cannot contain the other "
                "instance is a band that will be wrong next start")),
        ((1, 7680, True), Measured(
            seconds=1.145359e-3, low=1.0484e-3, high=1.168497e-3, samples=3,
            how="upper median [min,max] of the 3 unpaced 1x7680 rows of "
                "devstep1, each the second of a paced/unpaced pair on the "
                "same shape; the paced partners read 2.72-2.76e-3 and are "
                "excluded as a different pacing regime, not discarded")),
        ((1, 8192, True), Measured(
            seconds=1.094724e-3, low=9.154264e-4, high=1.124574e-3, samples=3,
            how="upper median [min,max] of the 3 unpaced 1x8192 rows of "
                "devstep1; paced partners 2.79-2.84e-3, excluded by regime")),
        ((1, 15744, True), Measured(
            seconds=7.853395e-4, low=7.739877e-4, high=7.893327e-4, samples=3,
            how="upper median [min,max] of the 3 unpaced 1x15744 rows of "
                "devstep1. These three were already unpaced as measured, gaps "
                "0.0023-0.0024 s, and have no paced partner")),
        ((2, 2048, True), Measured(
            seconds=6.163755e-4, low=5.538445e-4, high=7.502701e-4, samples=6,
            how="upper median [min,max] of the 6 unpaced 2x1024 cohort "
                "prefills of cap_conc and cap_conc3. The one paced 2x2048 row "
                "reads 1.981141e-3 and is excluded by regime. This is the "
                "only two-sequence evidence below 15360 and it is 6x smaller, "
                "so the two-sequence group is steep and its interpolation "
                "between 2048 and 15360 spans a wide gap on two anchors")),
    ),
    postprocess_prefill_anchors=(
        ((1, 1024, True), Measured(
            seconds=1.0156e-4, low=9.92e-5, high=1.0496e-4, samples=7,
            how="span_seconds.postprocess of the same 7 unpaced rows")),
        ((1, 7680, True), Measured(
            seconds=1.0372e-4, low=1.022e-4, high=1.0492e-4, samples=3,
            how="span_seconds.postprocess of the same 3 unpaced rows")),
        ((1, 8192, True), Measured(
            seconds=1.0028e-4, low=9.888e-5, high=1.0264e-4, samples=3,
            how="span_seconds.postprocess of the same 3 unpaced rows")),
        ((1, 15744, True), Measured(
            seconds=1.0764e-4, low=1.0736e-4, high=1.0968e-4, samples=3,
            how="span_seconds.postprocess of the same 3 unpaced rows")),
        ((2, 2048, True), Measured(
            seconds=1.0056e-4, low=9.972e-5, high=1.0152e-4, samples=6,
            how="span_seconds.postprocess of the same 6 unpaced rows")),
    ),
    tp_broadcast=SOURCE_27B_TP1_CONC_V2.tp_broadcast,
    decode_context=SOURCE_27B_TP1_CONC_V2.decode_context,
    prefill_sequences=(1, 2),
    prefill_tokens=(640, 16384),
    topologies=(1,),
    capture_sizes=SOURCE_27B_TP1_CONC_V2.capture_sizes,
    version="prefill-interp-2026-09-12",
    provenance="source-27b-tp1-prefill-cells unchanged, plus unpaced TP1 "
               "prefill rows of pricing_coverage/single640, "
               "pricing_coverage/calib2seq, pricing_coverage/devstep1, "
               "g4/cap_conc and g4/cap_conc3 as interpolation anchors, "
               "extracted by agent_scratch/stage/region_anchors.py. Decode, "
               "broadcast and the context bound carried from "
               "source-27b-tp1-conc-v2. TP1 only",
)


#: The same model, extended to the sequence counts the scheduler can reach.
#:
#: `SOURCE_27B_TP1_PREFILL_INTERP` above stops at two sequences, and its own
#: note says three or more is refused deliberately because the captures there
#: were single points no trace shape would land on. The corrected client
#: workload changes that premise: clients are top-level sessions, not an
#: in-flight cap, so eight clients can put far more than eight requests in
#: flight and the scheduler will batch up to `max_num_seqs=32`. A model that
#: refuses everything above two sequences cannot complete the development or
#: client workloads at all.
#:
#: **The published stanzas above are untouched.** Both anchor tuples and both
#: cell tuples are carried over by reference; the one- and two-sequence groups
#: answer exactly what they answered before, byte for byte. What is added is a
#: third group.
#:
#: WHY ONE GROUP AND NOT THIRTY. `agent_scratch/stage/job_regionseqs.sh`
#: measured eight shapes on one TP1 engine -- 32x512, 32x256, 28x512, 24x512,
#: 20x512, 12x1024, 5x1024, and 16x1024 as a control -- joining the 3x512,
#: 4x512, 3x4096 and 4x4096 cells of `job_regiongaps`. Across those, at a fixed
#: token count the term barely moves with the sequence count: 12288 tokens over
#: 3, 12 and 24 sequences spread 9.6%, and 16384 over 4 and 32 spread 2.4%.
#: Across token counts it moves 6.8x. So the evidence says the axis is tokens,
#: and thirty per-sequence-count groups would be roughly sixty campaigns spent
#: resolving a difference smaller than the repeat spread. The single/two-
#: sequence distinction is NOT pooled away -- there the sequence count really
#: does matter, five times over at 15232 vs 15360 tokens -- so `_pool` starts
#: the pooled range at 3.
#:
#: THE 4.6x CONTROL, ANSWERED. `cap_conc` published (16, 16384) at 8.458e-04 s
#: while `regiongaps` published (4, 16384) at 3.891e-03 -- 4.6x apart at the
#: same token count, which would have destroyed the pooling argument if it were
#: a sequence-count effect. Re-measuring `cap_conc`'s own 16x1024 shape inside
#: this campaign returned 4.177e-03. It is a difference between campaigns, not
#: between shapes, so the `cap_conc` rows are excluded here rather than pooled
#: -- the standing rule against pooling the two cached populations -- and they
#: remain published and unedited where they already are.
#:
#: STATED UNCERTAINTY, three items, none of them hidden:
#:
#: 1. **The flat mode is reachable on this tree.** Three of today's own rows
#:    fall into the ~6e-04 population at scattered positions, which is why the
#:    bands below are wide at 5120, 8192 and 10240 rather than tight: the low
#:    edge is a real observation, not a percentile artefact. A caller reading
#:    the band rather than the central value is reading the honest range.
#: 2. **5120 to 8192 crosses the two modes.** The 5120 anchor's centre sits in
#:    the flat population and the 8192 one does not, so an interpolated answer
#:    in between is the weakest claim in this model. It is also where the term
#:    is smallest in absolute size.
#: 3. **`waiting: 0` only.** Every burst was enqueued into an empty waiting
#:    queue, so these rows are steps scheduled with nothing else waiting. A
#:    served step with a backlog is a different regime and is not claimed.
#:
#: Below 1536 tokens and above 16384 the pooled group refuses, as it should:
#: 16384 is the token budget, so no step exceeds it, and nothing multi-sequence
#: shorter than 1536 was measured.
#:
#: THE DECLARED WIDTHS, 2026-09-13. This profile shipped saying `topologies=(1,)`
#: while carrying its decode, broadcast and scalar prefill terms from
#: `SOURCE_27B_TP1_CONC_V2`, which says `(1, 2, 4)` -- and says it over prefill
#: terms measured at TP1 too, carried in turn from `SOURCE_27B_TP1`. So the
#: narrowing was not a stronger claim about where these seconds hold; it was
#: the prefill-era descendants dropping their parents' declaration while
#: inheriting their numbers. The cost of that was real: the TP2 and TP4 cells
#: of the client matrix had to pass `regions=none`, which does not widen a
#: prediction, it drops preparation and postprocess out of it and leaves body
#: plus head. The declaration is now carried from the parent like every other
#: transferred field, and `version` moves with it so a run stamped against the
#: old declaration is not confusable with one stamped against this.
#:
#: **No coefficient changed.** Every cell, anchor, band and sample count below
#: is what it was; at TP1 this profile answers exactly what `prefill-seqs-
#: 2026-09-12` answered. What changed is which widths it will answer for, and
#: `region_snapshot` digests the domain along with the numbers, so the change
#: is visible in the digest rather than silent.
#:
#: WHY THE TRANSFER HOLDS HERE. Not a claim that runner regions are
#: width-insensitive in general. A claim that on THIS engine at THIS
#: configuration, the code `prepare_model` and `postprocess` execute is the
#: same code over the same shapes at every width, plus one collective that was
#: measured separately. Read off the source, not fitted:
#:
#: 1. **The runner has exactly two TP-conditional statements outside the
#:    model.** `get_tp_group()` appears in the forward path at
#:    model_runner.py:3309-3312, the broadcast of the sampled ids, and at
#:    :3323-3324, a second broadcast taken only when a request asked for
#:    logprobs. There is no third, and `prepare_model` contains none.
#:
#: 2. **Postprocess sees full-width logits at every TP.** `ParallelLMHead`
#:    shards the vocabulary and all-gathers it back inside `compute_logits`
#:    (embed_head.py:255-257), and the shard divides exactly --
#:    `assert num_embeddings % self.tp_size == 0` (embed_head.py:150) with
#:    `num_embeddings = config.vocab_size`, unpadded (qwen3_5.py:522). The
#:    sampler's `[sequences, vocab]` input is therefore identically shaped at
#:    TP1, TP2 and TP4. The all-gather itself is inside `run_model`, so it is
#:    the head's cost and not a region's; `LibraryCostOracle.estimate` merges
#:    head coverage into the step's, so an unpriced all-gather makes the step
#:    incomplete rather than silently free.
#:
#: 3. **Preparation is sized by the batch and by configuration constants.**
#:    `prepare_sample` writes `[bs]` rows (model_runner.py:2726-2772);
#:    `block_table_cols` is `max_num_blocks_per_seq // block_ratio`, from
#:    `max_model_len` and `block_size` (backends.py:341); `pack_rows` writes
#:    one row per sequence. None of them carries a width.
#:
#: 4. **The builder's one per-rank head count does no per-step work here.**
#:    `CommonAttentionBuilder.num_attention_heads` is `heads // world_size`
#:    (backends.py:346) and gates eagle's mid-step path; this deployment has
#:    no drafter.
#:
#: 5. **The one genuinely width-sized preparation term is off this
#:    configuration's path.** See `KV_HEAD_SIZED_PREPARE_BLOCK_SIZES`:
#:    `block_size=16` never calls it, and `wide_tp_precondition` refuses a
#:    build that pairs this profile with a block size that would. That is the
#:    one material width-sensitive term found in preparation, and it is
#:    excluded by configuration rather than by argument.
#:
#: 6. **The cross-rank collectives in preparation are DP's, not TP's.**
#:    `_preprocess` gates its packed all_gather on `data_parallel_size`,
#:    `enable_tbo` and `prefill_context_parallel_size`
#:    (model_runner.py:2140-2180), all 1 here.
#:
#: WHAT IS ADDED AT TP>1: `tp_broadcast`, the collective of point 1, carried
#: from `SOURCE_27B_TP1` where it was measured by
#: `agent_scratch/g4/bcast_probe.py` as a standalone primitive on the real two-
#: and four-rank groups (26.5-29.0 us, flat across 1-32 sequences, both
#: dtypes, both widths). `_parts` charges it only when the step samples a
#: token, so a middle chunk of a chunked prefill is not billed a collective it
#: never runs. Nothing else is scaled by the width; no number here is fitted
#: to any TP2 or TP4 engine observation.
#:
#: DECLARED LIMITATIONS, none of them closed by this change:
#:
#: * **Inter-rank skew is not modelled.** Preparation here is
#:   `seconds - run_model - postprocess` from a TP1 capture, where there was
#:   no peer to wait for. At TP>1 a rank reaching the body's first collective
#:   early waits there, and that wait lands inside `run_model` -- the body's
#:   span, not a region's. This profile neither models nor claims it;
#:   `rank_aggregation="slowest"` is what the step-level answer leans on.
#: * **Process control is outside this span entirely.** At TP>1 the engine
#:   core dispatches a step through `rpc_broadcast_mq.enqueue`
#:   (async_proc.py:429) and the writer waits until `read_count == n_reader`,
#:   `n_reader` being the width (aiter `shm_broadcast.acquire_write`). That is
#:   host-side and outside `ModelRunner.forward`, so it is neither included
#:   here nor double-counted -- it is an unmeasured term of the served step,
#:   reported as such rather than folded into a region. Measuring it is a
#:   standalone primitive campaign, to be coordinated rather than assumed.
#: * **Logprobs are out of scope.** The second broadcast of point 1 fires when
#:   any request asks for logprobs. The captures requested none and neither
#:   does the acceptance protocol, so nothing here measures it.
#: * **Every bound above still binds at every width**: `waiting=0` steps,
#:   decode histories 1025..1152 tokens, the pooled 3..32-sequence prefill
#:   group on the token axis, captured replays only.
SOURCE_27B_TP1_PREFILL_SEQS = BucketedRunnerRegions(
    postprocess_decode=SOURCE_27B_TP1_CONC_V2.postprocess_decode,
    prepare_decode_cells=SOURCE_27B_TP1_CONC_V2.prepare_decode_cells,
    postprocess_prefill=SOURCE_27B_TP1_CONC_V2.postprocess_prefill,
    prepare_prefill=SOURCE_27B_TP1_CONC_V2.prepare_prefill,
    prepare_prefill_cells=SOURCE_27B_TP1_PREFILL_CELLS.prepare_prefill_cells,
    postprocess_prefill_cells=(
        SOURCE_27B_TP1_PREFILL_CELLS.postprocess_prefill_cells),
    prepare_prefill_anchors=(
        SOURCE_27B_TP1_PREFILL_INTERP.prepare_prefill_anchors + (
            ((POOLED_SEQS, 1536, True), Measured(
                seconds=9.857996e-4, low=4.5342e-4, high=1.0493e-3, samples=3,
                how="upper median [min,max] of the 3 retained rows of the "
                    "3x512 cell, pricing_coverage/regiongaps")),
            ((POOLED_SEQS, 2048, True), Measured(
                seconds=4.802168e-4, low=4.6448e-4, high=5.5421e-4, samples=3,
                how="upper median [min,max] of the 3 retained rows of the "
                    "4x512 cell, pricing_coverage/regiongaps")),
            ((POOLED_SEQS, 5120, True), Measured(
                seconds=6.021354e-4, low=5.5550e-4, high=1.8053e-3, samples=3,
                how="upper median [min,max] of the 3 retained 5x1024 rows, "
                    "pricing_coverage/regionseqs. The centre is in the flat "
                    "population and the high edge is not, which is the widest "
                    "band here and the reason item 2 above is stated")),
            ((POOLED_SEQS, 8192, True), Measured(
                seconds=2.344933e-3, low=6.4582e-4, high=2.4911e-3, samples=3,
                how="upper median [min,max] of the 3 retained 32x256 rows, "
                    "pricing_coverage/regionseqs. One of the three fell into "
                    "the flat mode and sets the low edge; it is kept, not "
                    "dropped")),
            ((POOLED_SEQS, 10240, True), Measured(
                seconds=2.871009e-3, low=7.9247e-4, high=2.8926e-3, samples=3,
                how="upper median [min,max] of the 3 retained 20x512 rows, "
                    "pricing_coverage/regionseqs; low edge a flat-mode row")),
            ((POOLED_SEQS, 12288, True), Measured(
                seconds=3.241286e-3, low=2.6858e-3, high=3.5118e-3, samples=9,
                how="upper median [min,max] of 9 retained rows POOLED over 3, "
                    "12 and 24 sequences -- 3x4096 from regiongaps, 12x1024 "
                    "and 24x512 from regionseqs. Their three centres are "
                    "2.964e-3, 3.241e-3 and 3.248e-3, 9.6% apart, which is "
                    "the measurement this group's existence rests on")),
            ((POOLED_SEQS, 14336, True), Measured(
                seconds=3.602950e-3, low=3.4989e-3, high=3.7258e-3, samples=3,
                how="upper median [min,max] of the 3 retained 28x512 rows, "
                    "pricing_coverage/regionseqs")),
            ((POOLED_SEQS, 16384, True), Measured(
                seconds=3.985405e-3, low=3.8541e-3, high=4.3725e-3, samples=6,
                how="upper median [min,max] of 6 retained rows POOLED over 4 "
                    "and 32 sequences -- 4x4096 from regiongaps and 32x512 "
                    "from regionseqs, centres 3.891e-3 and 3.985e-3, 2.4% "
                    "apart at an eightfold difference in sequence count. "
                    "32x512 is the largest step the scheduler can build: 32 "
                    "is max_num_seqs and 16384 is the token budget")),
        )),
    postprocess_prefill_anchors=(
        SOURCE_27B_TP1_PREFILL_INTERP.postprocess_prefill_anchors + (
            ((POOLED_SEQS, 1536, True), Measured(
                seconds=1.0380e-4, low=1.0188e-4, high=1.0448e-4, samples=3,
                how="span_seconds.postprocess of the same 3 rows")),
            ((POOLED_SEQS, 2048, True), Measured(
                seconds=1.0096e-4, low=1.0064e-4, high=1.0176e-4, samples=3,
                how="span_seconds.postprocess of the same 3 rows")),
            ((POOLED_SEQS, 5120, True), Measured(
                seconds=1.0160e-4, low=1.0092e-4, high=1.0844e-4, samples=3,
                how="span_seconds.postprocess of the same 3 rows")),
            ((POOLED_SEQS, 8192, True), Measured(
                seconds=1.0160e-4, low=1.0076e-4, high=1.0320e-4, samples=3,
                how="span_seconds.postprocess of the same 3 rows")),
            ((POOLED_SEQS, 10240, True), Measured(
                seconds=1.0060e-4, low=1.0016e-4, high=1.0360e-4, samples=3,
                how="span_seconds.postprocess of the same 3 rows")),
            ((POOLED_SEQS, 12288, True), Measured(
                seconds=1.534410e-4, low=1.4944e-4, high=1.5896e-4, samples=9,
                how="span_seconds.postprocess of the same 9 rows. Postprocess "
                    "is bimodal across this group -- ~1.01e-4 at 1536, 2048, "
                    "5120, 8192, 10240 and 14336, ~1.50e-4 at 12288 and 16384 "
                    "-- and the split follows neither tokens nor sequences "
                    "monotonically. It is 5e-5 on a millisecond term, so each "
                    "token count is quoted from its own rows rather than "
                    "modelled, and no cause is claimed")),
            ((POOLED_SEQS, 14336, True), Measured(
                seconds=1.0136e-4, low=1.0020e-4, high=1.0408e-4, samples=3,
                how="span_seconds.postprocess of the same 3 rows")),
            ((POOLED_SEQS, 16384, True), Measured(
                seconds=1.500810e-4, low=1.4640e-4, high=1.5256e-4, samples=6,
                how="span_seconds.postprocess of the same 6 rows")),
        )),
    prefill_pooled_sequences=(3, 32),
    tp_broadcast=SOURCE_27B_TP1_CONC_V2.tp_broadcast,
    decode_context=SOURCE_27B_TP1_CONC_V2.decode_context,
    prefill_sequences=(1, 2),
    prefill_tokens=(640, 16384),
    # Carried from the parent, like the decode cells and the broadcast above.
    # See "THE DECLARED WIDTHS" in the note: the numbers under this are the
    # same numbers, and what licenses them at 2 and 4 is a source-code
    # argument plus the separately measured broadcast, not a fit.
    topologies=SOURCE_27B_TP1_CONC_V2.topologies,
    capture_sizes=SOURCE_27B_TP1_CONC_V2.capture_sizes,
    version="prefill-seqs-2026-09-13",
    provenance="source-27b-tp1-prefill-interp unchanged, plus the retained "
               "rows of pricing_coverage/regiongaps and "
               "pricing_coverage/regionseqs as one pooled 3..32-sequence "
               "anchor group, acquired on GPU0 by "
               "agent_scratch/stage/job_regionseqs.sh and "
               "agent_scratch/stage/job_regiongaps.sh, tabulated by "
               "agent_scratch/stage/multiseq_anchors.py. g4/cap_conc rows "
               "above four sequences are excluded as a separate cached "
               "population, not pooled. Steps scheduled with waiting=0 only. "
               "Every coefficient measured at TP1; declared for tp 1, 2 and 4 "
               "as SOURCE_27B_TP1 and source-27b-tp1-conc-v2 already are, on "
               "the source-code argument in the note above (the runner's only "
               "TP-conditional statements outside the model are "
               "model_runner.py:3309-3312 and :3323-3324; compute_logits "
               "all-gathers the vocabulary shards back to config.vocab_size "
               "before postprocess sees them, embed_head.py:150 and :255-257; "
               "preparation is sized by the batch and by max_model_len / "
               "block_size, backends.py:341; the kv-head-sized metadata build "
               "is block_size 256/1024 only, aiter_attention.py:1133 and "
               ":1370, and this deployment runs 16) plus tp_broadcast, a "
               "standalone bcast_probe.py primitive on the real 2- and 4-rank "
               "groups. NOT COVERED: inter-rank skew (lands in the body's "
               "span), engine-core process control (outside "
               "ModelRunner.forward), logprobs broadcasts. No full-engine or "
               "serving measurement at TP2 or TP4 contributed any number here",
)


#: Block sizes whose per-step attention metadata build is sized by the rank's
#: own KV head count, and therefore is NOT width-invariant.
#:
#: `AiterAttentionMetadataBuilder` calls `set_aiter_persistent_worker_buffers`
#: only `if self.block_size in (256, 1024)` (aiter_attention.py:1133 on the
#: served path, :1370 on the capture path), and that function passes
#: `num_key_value_heads // get_tp_group().world_size` to
#: `aiter.get_pa_metadata_v1` (aiter_attention.py:354-396). Its output tables
#: -- `work_meta_data`, `work_info_set`, the reduce maps -- are sized by that
#: per-rank head count, so on those block sizes preparation carries a term
#: that shrinks as TP grows, and a TP1 measurement of it is a measurement of
#: different work.
#:
#: At `block_size=16`, which is what the cc-traces deployment runs, the branch
#: is not taken and the question does not arise. That is a
#: configuration-conditional exemption rather than a general result, so it is
#: named here and checked at build time by `wide_tp_precondition`.
KV_HEAD_SIZED_PREPARE_BLOCK_SIZES = (256, 1024)


def wide_tp_precondition(model, tp: int, block_size: int) -> Optional[str]:
    """Why this preset must not be used at this width and block size, or None.

    A widened preset's transfer argument rests on a branch that
    `block_size=16` does not take. A deployment that moved the block size to
    256 or 1024 would take it, and preparation would then carry a term sized
    by the rank's own KV head count -- so the TP1 numbers would be
    measurements of different work, quietly.

    Checked at build time because that is where both facts are known: a shape
    does not carry the block size, so `refusal` cannot see it. Fails closed,
    and only for a preset that actually claims a width above one.
    """
    if model is None or int(tp or 1) <= 1:
        return None
    if len(getattr(model, "topologies", (1,))) <= 1:
        return None
    if int(block_size or 0) not in KV_HEAD_SIZED_PREPARE_BLOCK_SIZES:
        return None
    return (
        f"region preset {getattr(model, 'version', '')!r} is declared for tp "
        f"{list(model.topologies)}, and that declaration rests on the "
        f"per-step attention metadata build being width-invariant. At "
        f"block_size={int(block_size)} it is not: "
        f"set_aiter_persistent_worker_buffers runs for block sizes "
        f"{list(KV_HEAD_SIZED_PREPARE_BLOCK_SIZES)} and sizes its tables by "
        f"num_key_value_heads // tp, so preparation measured at TP1 is a "
        f"measurement of different work. Measure that configuration, or run "
        f"this width with regions=none.")


REGION_MODELS = {
    "source-27b-tp1": SOURCE_27B_TP1,
    "source-27b-tp1-conc": SOURCE_27B_TP1_CONC,
    "source-27b-tp1-conc-v2": SOURCE_27B_TP1_CONC_V2,
    "source-27b-tp1-prefill-1x640": SOURCE_27B_TP1_PREFILL_1X640,
    "source-27b-tp1-prefill-cells": SOURCE_27B_TP1_PREFILL_CELLS,
    "source-27b-tp1-prefill-interp": SOURCE_27B_TP1_PREFILL_INTERP,
    "source-27b-tp1-prefill-seqs": SOURCE_27B_TP1_PREFILL_SEQS,
    "none": None,
}


def region_model(name):
    """The named region model, or ``None`` for ``"none"``.

    Raises rather than falling back to a default: running with the source's
    region model and running without one differ by tens of percent in the step,
    so an unrecognised name must not resolve to either.
    """
    try:
        return REGION_MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown region model {name!r}; known: "
            f"{', '.join(sorted(REGION_MODELS))}") from None
