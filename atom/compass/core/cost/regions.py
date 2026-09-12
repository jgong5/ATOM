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
    version: str = ""
    provenance: str = ""

    def _cells(self) -> dict:
        return {(int(b), bool(p)): m for (b, p), m in self.prepare_decode_cells}

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
        parts = [("<postprocess>", self.postprocess_prefill if prefill
                  else self.postprocess_decode),
                 ("<prepare>", self.prepare_prefill if prefill
                  else self._decode_prepare(shape))]
        if tp > 1:
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
            f"provenance          : {self.provenance}",
        ]
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


REGION_MODELS = {
    "source-27b-tp1": SOURCE_27B_TP1,
    "source-27b-tp1-conc": SOURCE_27B_TP1_CONC,
    "source-27b-tp1-conc-v2": SOURCE_27B_TP1_CONC_V2,
    "source-27b-tp1-prefill-1x640": SOURCE_27B_TP1_PREFILL_1X640,
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
