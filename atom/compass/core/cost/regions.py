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
