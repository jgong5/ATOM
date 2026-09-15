# ATOMCompass — PoC summary and handover

A wrap-up of the proof of concept. It states what works, what does not, what was
tried, and what is left. It is written to be read by someone who has not seen
the work.

**How this relates to the other documents.**

| file | what it is |
| --- | --- |
| `POC_SUMMARY.md` (this) | conclusions. The single source of truth for status. |
| `DESIGN_NOTES.md` | the working log, ~5000 lines, written as each thing was found. Every claim here is traceable to it. Detailed, and parts of it are stale. |
| `ATOM_DEFECTS.md` | bugs found in ATOM itself, not in Compass. |

Where the two disagree, this file is newer. Where you need the evidence for a
claim, go to `DESIGN_NOTES.md`.

---

## 1. What ATOMCompass is

A performance simulator for ATOM. It answers: **for this model, this
parallelism, and this workload — what latency and throughput, and does it fit in
memory?**

The central design choice: **Compass replaces only the forward pass.** ATOM's
real scheduler, real block manager, and real admission logic all run unchanged.
A simulated run therefore makes the same scheduling decisions as a real one; it
only substitutes a predicted duration for the work. Compass does not model
serving *decisions*, only the time they consume.

Two quantities are modelled:

* **time** — what a step costs, and what serving adds around it.
* **memory** — what a configuration consumes, and so whether it fits and how
  many KV blocks it gets.

Memory is in scope because it decides which configurations exist at all.
Predicting the speed of a configuration that cannot start is answering the wrong
question.

---

## 2. Vocabulary

Needed to read the rest. Two axes.

**Subject** — what is being costed:

* **model step** — the forward, for a step of a given shape.
* **serving** — admission, scheduling, KV cache management.

**Provenance** — how the cost was obtained:

* **analytical** — computed without measuring the subject. *Nothing in Compass
  is analytical today.* The word is reserved, not aspirational.
* **empirical** — derived from measurement. A genus, written
  `empirical/<species>`:

| species | meaning |
| --- | --- |
| `empirical/measured` | the unit itself was timed |
| `empirical/fitted` | a form was chosen and coefficients regressed over measured steps |
| `empirical/interpolated` | no form assumed; nearby measurements looked up |
| `empirical/extrapolated` | asked outside the measured range |

**"Priced" is not a species.** *Pricing* is measuring one operator at one shape.
A price is `empirical/measured` like any other measurement. What differs is the
*unit*: say **measured (op-level)** or **measured (step-level)**, and say
"summed" when op-level measurements are added to stand for a step.

Reading a measurement back from a file is still `empirical/measured`. *When* a
measurement was taken is not provenance. What changes is whether the key matched:
exact is `measured`, nearest is `interpolated`, outside is `extrapolated`.

The same vocabulary applies to bytes. Say **time cost** or **memory cost** where
it matters.

**The four oracles built:**

| oracle | subject | provenance |
| --- | --- | --- |
| `constant` | model step | declared — a stub for proving the plumbing |
| `calibrated` | model step | `empirical/fitted` |
| `interpolated` | model step | `empirical/interpolated` |
| `priced` | model step | `empirical/measured`, summed from operators |
| admission | serving | `empirical/measured` |

---

## 3. Status

### 3.1 What has been demonstrated

All on MI300-series (MI308X and similar), 8 GPUs per node, **shared with other
tenants**. Read section 7 on noise before trusting any single number.

| what | result | oracle | reading |
| --- | --- | --- | --- |
| **Qwen3-0.6B, TP=1, offline serving** | TTFT +2.0 ± 7.0%, TPOT −1.0 ± 3.3%, latency +0.0 ± 1.8% | calibrated | **inside run-to-run noise.** The honest ceiling for this model on a shared box. |
| **Qwen3-0.6B, TP=2** | TTFT −32.3%, TPOT −13.6%, latency −20.8% | calibrated | one run. Worse than noise — a real gap, see §4.5 |
| **Qwen3.8-27B, TP=2, offline** | TTFT +2.45%, TPOT 0.00%, latency +1.37% | priced | **the strongest result.** Both machine constants came from a model 45× smaller. Nothing was fitted on the 27B except admission. |
| same, admission held out | TTFT +6.58%, TPOT −0.11%, latency +2.37% | priced | admission is worth ~4 points of TTFT and must be measured per deployment |
| **Qwen3.8-27B, memory** | KV block count **−0.09%** | analytical-ish memory model | see §5.5; the terms are measured constants, not a formula |
| Qwen3-0.6B, memory, TP=1/2/4 | block count −0.02% / −0.02% / +0.07% | same | |
| **HTTP serving, varied shapes** | TTFT +47.9%, latency +25.3% | priced | **the main failure.** A fixed-shape oracle does not survive serving. |
| **cc-traces (real agentic trace)** | latency −5.0%, but TTFT +51.6% and decode −30.3% | calibrated | latency is right **by cancellation**. See §5.8. |

### 3.2 The one-line summary

**Offline, at a fixed shape, prediction is good — within a few percent on a
27B with constants transferred from a 0.6B. Under real serving, with varied
shapes and real arrival patterns, it is not yet.** The gap between those two
sentences is the remaining work.

### 3.3 What is built

| path | what |
| --- | --- |
| `atom/compass/core/cost/` | the four oracles |
| `atom/compass/core/graph.py` | the op-graph data model |
| `atom/compass/core/memory.py`, `memory_model.py` | memory readings and the analytical model |
| `atom/compass/runtime/runner.py` | the hook into `ModelRunner`; trace / measure / predict modes |
| `atom/compass/runtime/meta.py` | op tracing on meta tensors, liveness by `weakref.finalize` |
| `atom/compass/runtime/microbench.py` | operator pricing |
| `atom/compass/runtime/derive.py`, `triton_trace.py`, `forward_ctx.py` | graph derivation, Triton interception, forward-context capture |
| `atom/compass/workload.py` | exact-length synthetic prompts |
| `scripts/compass/validate.py` | the end-to-end comparison (calibrate → real → modelled → compare) |
| `scripts/compass/run.py` | fixed-workload and calibration-sweep driver |
| `scripts/compass/replay.py` | HTTP client, trace replay, arrival pacing |
| `scripts/compass/residual.py` | splits a step's residual into boundary and pricing error |
| `scripts/compass/step_accounting.py` | where a step's time goes, from a profile |
| `scripts/compass/validate_memory.py` | per-term memory validation |

264 tests, all passing.

---

## 4. Key challenges

The hard problems. These are properties of the domain, not bugs.

### 4.1 A step is not the sum of its kernels

Summed operator prices came to **0.740** of a real decode step at 98.8%
operator coverage. The missing quarter is real. Understanding it took most of
the project (§5.4).

### 4.2 An operator priced alone is not the same operator in a step

A kernel measured on its own differs from the same kernel inside a step, and the
sign is not fixed:

* on the 0.6B, the priced sum falls ~28% **below** in-situ kernel time
* on the 27B, it lands ~2% **above**

This bounds everything operator-level pricing can deliver. It is the single
largest source of remaining error.

### 4.3 A client cannot time a simulated engine

Under simulation the engine advances a *virtual* clock and never runs the
forward. A benchmark timing an HTTP socket measures how fast the simulator ran —
about 3× faster than the system it stands for. **All timings must come from the
engine** (`GET /compass/requests`), which reports simulated time on a simulated
run and wall time on a real one, making the two comparable.

### 4.4 Serving is not a fixed shape

A priced oracle predicts one shape well. A deployment produces a continuous
range of shapes — prefill, chunked prefill, mixed batches, long-context decode.
Prices do not interpolate across shapes (tried; §5.3).

### 4.5 Coverage must bracket the evaluation in every dimension

An oracle asked outside its evidence extrapolates silently. This has caused real
errors more than once. Worse, the current guard checks each feature
*separately*, so it cannot see a hole in the joint distribution.

### 4.6 The machine is shared

GPUs are not partitioned and about twenty containers share the box. Five runs of
one unchanged command gave TPOT from −5.4% to +2.4%. **The standard deviation
exceeds the mean.** Any single number is a draw from a distribution.

A second consequence: `non_torch` memory is a *device-wide* reading, so
neighbours' memory is charged to your configuration. This routinely prevents
runs from starting at all.

### 4.7 Some configurations cannot be simulated on fewer devices

TP is symmetric and can be simulated on one device. Pipeline parallel, context
parallel, data parallel, DP-attention, and disaggregated prefill cannot: ranks
diverge and virtual time would have to be coordinated across processes. Nobody
in the field has published a solution.

### 4.8 Memory decides which configurations exist

Sizing the 27B at TP=1 failed before any timing could be taken. A tool asked
"which configuration should I deploy" must answer with a set whose members all
fit.

---

## 5. Features, design options, and evidence

For each area: what was tried, what works, what does not.

### 5.1 Cost oracles

| option | status | evidence |
| --- | --- | --- |
| `constant` | works, stub only | proves the plumbing |
| `calibrated` (fit a form to measured steps) | **works offline**, best general-purpose choice today | 0.6B TP=1 inside noise |
| `interpolated` (look up nearby measurements) | works, limited | needs dense coverage |
| `priced` (sum measured operator costs) | **works at a fixed shape**, fails at serving | 27B ±2.5% offline; +47.9% TTFT served |

**Not explored:** anything `analytical`. No cost is derived from first
principles today. This is the biggest unexplored branch, and it is what would
let Compass predict a configuration nobody has run.

**Fitting lessons that hold:**

* Fit **relative** error, not absolute seconds. A fit minimising seconds is
  decided by its largest samples.
* Fit prefill and decode **separately**.
* Fit decode **per CUDA-graph rung**. With graphs, a decode step replays a
  padded bucket, so cost steps at the ladder.
* Decode cost **falls** as batch grows (12 sequences cost less per step than 1).
  An early model had this sign backwards.

### 5.2 Op-graph capture and derivation

Works. A graph is captured from the real runner, per rank, with shapes, dtypes,
forward context, and liveness.

| option | status |
| --- | --- |
| capture on hardware | **works**, is the basis of pricing |
| derive on meta tensors (no GPU) | **works** and reproduces hardware; lets one process derive any TP width |
| derive under compilation | **not done.** Derivation is eager; production is compiled at `--level 3`. A derived graph and a captured one differ by construction (386 operators vs 330). |

**Known gaps:** trace mode does not observe the CUDA-graph replay path; only one
decode step is traced by default; a custom op's inner Triton kernel is recorded
only on hardware; derivation takes ~0.16 s for the 27B, far too slow to run per
step.

### 5.3 Pricing operators

Works, at 98–99% operator coverage, including collectives.

| option | status | evidence |
| --- | --- | --- |
| time each operator in a CUDA graph | **works** — the right method | a per-call loop cannot price a kernel smaller than its own ~30 µs call overhead |
| time each operator back-to-back | works only for uncapturable operators | carries launch overhead |
| interpolate a price across shapes | **does not work** | tried and abandoned |
| price attention from a shape signature | **does not work** | attention reads its metadata from the forward context, not its arguments — priced 7× over until the context was captured too |

**Pricing is not repeatable to better than ~1%.** Two runs of the same graph on
the same machine moved the summed total by 0.96%, with a p90 per-signature swing
of 32%. High-occupancy signatures repeat to ~1–3%; a few are genuinely unstable.
**Any residual quoted below ~2% is quoting the instrument.**

**Known holes in the price list:**

* **inductor-generated kernels are unpriced** — 128 of 130 such operators. They
  are **4.8% of a 27B decode step's kernel time**. Loading them faults the device
  under tensor parallelism, so the default is off.
* per-kernel breakdowns were sparse until recently; an operator whose signature
  fragments per layer was invisible to the threshold that decides them.

### 5.4 The overhead terms — the "missing quarter"

This was the central question and the answer changed twice.

| term | status |
| --- | --- |
| `DEFAULT_BOUNDARY_SECONDS = 2.25 µs` per launch, replayed steps | **the value works; its explanation is wrong** |
| `DEFAULT_DISPATCH_SECONDS = 130 µs` per operator, eager steps | works; does **not** transfer between models (91.72 vs 132.70 µs) |
| `DEFAULT_HOST_SECONDS_PER_LAUNCH = 95.8 µs` — a host floor | works; `step = max(kernel time, launches × h)` holds within 2.6% on eight shapes |

**The boundary is not a boundary.** Measured directly from the device timeline
on four configurations, the gap between consecutive kernels on a replayed step
has a **median of 1 ns and a p90 of 2 ns**. If every launch paid 2.25 µs, each
step would show ~950 gaps in the 1–5 µs band; across 15,200 gaps there are
**zero**, while the same traces record gaps above 5 µs. The device runs a
replayed step back to back.

What the constant really absorbs is **pricing error** (§4.2), multiplied by
launch count into a shape that happens to fit. That is why it works and why it
does not transfer. It is kept at its current value because removing it would
make predictions worse without making them righter.

**There is no collective boundary either.** Measured at TP=2, 4 and 8: the gap
after a collective launch never exceeds **2 ns**. The earlier "collective
boundary constants" (4.20 µs at TP=2, 9.72 µs at TP=4) do not reproduce; they
were residual artifacts. What does scale with width is the collective kernel's
own duration, and it is nearly flat: `cross_device_reduce_1stage` runs 9.08,
10.40, 10.95 µs at TP=2/4/8 — **+20.6%** where linear-in-group predicts +300%
and linear-in-log2 +200%. That is a property of this fully-connected
single-node fabric and should be re-measured multi-node.

**Synchronising operators are not kernel time.** `aten::item` and
`aten::is_nonzero` (72 per 27B decode graph) run no kernel; their price is the
wait. Counting them as kernel time overstated a step by 8.4%. They are now in
`HOST_SYNC` and excluded.

### 5.5 Memory model

The most complete part of the project.

**Result: KV block count predicted to −0.09% (27B TP=2) and −0.02% / −0.02% /
+0.07% (0.6B TP=1/2/4).**

Terms and how each is obtained:

| term | method | status |
| --- | --- | --- |
| weights | ask the built model (including a meta build) | **works.** Reading the checkpoint instead is exact on dense models but 3.3% low on the hybrid 27B at TP=4 |
| activations | def-use walk over the traced op graph | **works** once compared at the right shape |
| scratch not visible to the tracer | one measured number per token | needed for the hybrid; takes the 27B from +0.91% to −0.09% |
| graph pool | measured floor + per-token | **works.** The engine's own estimate is 8–19× under what capture costs |
| `non_torch` | measured constant per topology | **not modelled.** A device-wide reading — it includes the neighbours |
| load residue | measured constant | **not modelled.** 2.02 GB per rank at TP>1, seven times the weights at TP=4 |

**Validation must be per term.** A summed check reported +13.8%; splitting it
found three errors, two of which cancelled — the largest was 25% of a term.

**Two limits:** the model generalises to a *shape* the trace was not taken at,
but **not** to a model that was never traced (the hybrid needs a real graph).
And fragmentation is not modelled and is not planned.

### 5.6 Parallelism

| what | status |
| --- | --- |
| TP, any width, from one process (`simulate_group_width`) | **works** — ranks of a symmetric group are interchangeable, measured |
| per-rank artifacts | **works** — each rank writes and reads its own graph |
| collectives priced | **works** at 98.3% coverage |
| Expert parallelism | **does not work meaningfully** |
| pipeline / context / data parallel, DP-attention, disagg | **refused**, by design (§4.7) |

**On EP specifically** — this was first written up as a success and is not one.
At `ep_size == tp_size` EP is close to a **no-op**: without EP each rank holds
all experts sharded along the intermediate dimension, with EP each rank holds
half the experts whole. Same FLOPs, different decomposition; graphs identical,
timings within 3–8%. At a real degree (`ep_size=4` via DP-attention) it runs, but
**the expert communication never reaches the graph**: dispatch and combine are
MORI kernels called inside `moe_forward`, so the dispatcher never sees them.
Re-costing EP across degrees needs `moe_forward` to expose them or a model of
them.

Note the 27B target model is a **hybrid** — 48 gated-DeltaNet (linear attention)
layers and 16 full-attention layers — not a dense transformer. It is not an MoE,
so EP needs a different model (Qwen3-30B-A3B).

### 5.7 Serving harness and arrivals

| option | status |
| --- | --- |
| time requests from the client | **wrong** for a simulated engine (§4.3) |
| read timings from `/compass/requests` | **correct**, and used |
| declare arrivals to the engine (`compass_arrival`) | **works on a simulated run** — the virtual clock honours them and jumps over idle |
| declare arrivals on a *real* run | **does not work.** A real clock has no start-of-run; the arrival is discarded and the real side answers a burst |
| server-side pacing (give the real clock an origin) | **tried and reverted.** The origin cannot be inferred from whichever request lands first under concurrent posting |
| **client-side pacing (`replay.py --pace`)** | **works, and is the right answer.** No origin is needed, and the engine's queue is genuinely empty between arrivals — which server-side pacing cannot reproduce |

The simulation's advantage shows here: on the same workload the real run took
309 s and the simulated one **3 s**, because the discrete-event jump skips idle
the real engine must sit through.

### 5.8 Realistic workloads (cc-traces)

Validated against `semianalysisai/cc-traces-weka-062126-256k` — 393 real Claude
Code sessions, 30,141 requests, 3.90 G input tokens, median **129,664** input
tokens per request.

**First result, 20 requests at median 163,584 tokens:**

| | real | modelled | on totals |
| --- | --- | --- | --- |
| TTFT | 28.35 s | 50.58 s | **+51.6%** |
| decode | 56.16 s | 25.44 s | **−30.3%** |
| per output token | 120.7 ms | 53.5 ms | **−71.1%** |
| **latency** | **76.28 s** | **75.91 s** | **−5.0%** |

**Latency agrees to 1.3% per request and that is not a result.** Too much time
before the first token, too little after, and the two nearly cancel.

The decode miss has a cause in the calibration, not the model: the sweep sampled
decode almost entirely at short context (median 258 tokens) and was then asked
about 164 k, where attention over the history dominates the step.

Two harness defects had to be fixed before any of this meant anything:

* the replay client sent **5–8× the requested tokens**, superlinearly — a
  length-dependent distortion reshapes a length distribution rather than
  shifting it.
* the calibration sweep topped out at ~1 k tokens against a 256 k workload.

---

## 6. Open issues

Ranked by what is known about their size. **S** = hours, **M** = days, **L** =
weeks or unknown.

### Blocking a production-quality result

| # | issue | size |
| --- | --- | --- |
| 1 | **Pricing error (§4.2)** — an operator priced alone is not the same in a step, and the sign flips between models. This bounds everything operator-level pricing can achieve. | **L** |
| 2 | **A fixed-shape oracle does not survive serving** — +47.9% TTFT on varied shapes. Prices do not interpolate. | **L** |
| 3 | **Decode is not sampled at long context**, so long-context TPOT is under-predicted by 71%. Fix is in the calibration sweep, not the model. | **S/M** |
| 4 | **Inductor-generated kernels are unpriced** — 4.8% of a 27B decode step. Loading them faults the device at TP>1. | **M** |
| 5 | **Nothing is analytical.** Every cost needs the configuration to have been run. This is the difference between a simulator and an interpolator. | **L** |

### Correctness and coverage

| # | issue | size |
| --- | --- | --- |
| 6 | Derivation is uncompiled; production is compiled. Derived and captured graphs are not comparable. | **L** |
| 7 | Trace mode does not observe the CUDA-graph path. | **M** |
| 8 | Only one decode step is traced. Prefill, chunked prefill, mixed batches, speculative decoding and MTP are untraced. | **M** |
| 9 | The extrapolation guard is per-feature and cannot see a hole in the joint distribution. | **S** |
| 10 | Derivation (~0.16 s) is too slow to run per step; needs a shape-keyed cache. | **M** |
| 11 | No way to tell the runner when to start measuring — warmup steps land in the table. | **M** |
| 12 | `non_torch` and load residue are measured constants, not models. Three topologies of one model is interpolation. | **M** |
| 13 | Only one communication group is resolvable; with TP and EP together the group is recorded as `"?"`. | **M** |
| 14 | Shape-changing collectives (`all_gather`, `reduce_scatter`) have no meta stand-in. | **S/M** |
| 15 | `silu_and_mul`'s price is **unstable**, not merely wrong: 3.08 → 4.51 → 3.06 µs across three runs. Investigate as instability. | **S** |

### Out of scope, recorded

| # | issue | size |
| --- | --- | --- |
| 16 | Asymmetric parallelism (PP, CP, DP, disagg) — refused by design. | **L** |
| 17 | Expert parallelism — the graph cannot see expert communication at any degree. | **L** |
| 18 | Fragmentation — not modelled, not planned, nobody models it. |  |

---

## 7. Traps — read before trusting a number

These cost real time and will cost it again.

1. **Read the spread before the mean.** Five identical runs spanned −5.4% to
   +2.4% TPOT. This document reported one run as the project's result for some
   time. Nothing was wrong with the run; quoting it was wrong.

2. **Aggregates hide compensating errors.** This has happened **four times**:
   * a +13.8% memory sum hid three errors, two cancelling, the largest 25% of a term;
   * a priced sum matched a step while over-charging its own coverage by 5.6% and omitting 4.8%;
   * the boundary constant was pricing error wearing a mechanism;
   * cc-traces latency landed within 5% by cancelling +52% TTFT against −30% decode.

   **Always validate per term and per request, never on the total.**

3. **Being profiled costs ~0.7–1.05 µs per kernel** — 8% of a 27B decode step.
   A price list gathered without a profiler cannot be compared against an
   in-situ figure gathered with one. Idle *is* comparable, because profiling
   inflates the kernel sum and the window equally.

4. **Never mix machines.** The same unprofiled 27B TP=4 decode is 9.774 ms on one
   box and 12.365 ms on another — a 26% difference, larger than the difference
   between TP=4 and TP=8 on one box.

5. **A dead gate is worse than no gate.** A guard that reads an environment
   variable the engine never sets is trusted and does nothing. Five runs were
   misdiagnosed this way.

6. **Turn the flag off and see what changes.** This is what showed EP was a
   no-op at `ep_size == tp_size`, and it should have been the first check.

7. **The instrument changes what it measures.** Synchronising to time each step
   made the run 33% slower and produced a table describing a machine that only
   exists while being measured. Measure with CUDA events read back later.

8. **Unit tests cannot catch harness-level races.** Server-side arrival pacing
   passed twelve unit tests and failed the first end-to-end smoke run, because
   which request establishes the run's origin depends on a race between posting
   threads.

---

## 8. How to reproduce

End-to-end comparison (calibrate → real → modelled → compare):

    python scripts/compass/validate.py --model Qwen/Qwen3-0.6B --num-prompts 8

Trace, then price, then predict:

    # 1. capture the op graph
    python scripts/compass/run.py --model M -tp N --compass \
        --compass-mode trace --compass-graph-out out/g.json --out out/t.json

    # 2. price its operators, and record real step times in the same run
    python scripts/compass/run.py --model M -tp N --compass \
        --compass-mode measure --compass-measure-out out/steps.jsonl \
        --compass-bench-graph "out/g.tp*.json" --compass-bench-out out/prices.json \
        --compass-bench-cache graph --out out/m.json

    # 3. split the residual into boundary and pricing error (one machine only)
    python scripts/compass/residual.py <trace-dir> --graph out/g.tp0.json \
        --prices out/prices.tp0.json --steps out/steps.tp0.jsonl --by-kernel

Calibration sweep, including long context:

    python scripts/compass/run.py --model M -tp N --sweep --sweep-long \
        --compass --compass-mode measure --compass-measure-out out/sweep.jsonl \
        --out out/s.json

Replay a recorded trace against a real engine:

    python scripts/compass/replay.py --port P --trace w.jsonl --pace \
        --check-lengths --out real.json

Memory, per term:

    python scripts/compass/validate_memory.py ...

**Always pass `--check-lengths` when replaying.** It costs nothing and catches a
class of error that silently invalidates everything downstream.

---

## 9. What a next stage should do first

In order, by value for effort:

1. **Sample decode at long context in the sweep** (§6.3). Small, and it unblocks
   the only realistic-workload result there is.
2. **Attack pricing error directly** (§6.1). Everything else is bounded by it.
   The tool exists (`residual.py`); what is missing is a model of why an
   operator costs more in a step than alone.
3. **Make the price list cover generated kernels** (§6.4), or state the 4.8%
   hole as a known bias.
4. **Decide whether to pursue `analytical`** (§6.5). Without it, Compass
   interpolates between configurations that were run. With it, Compass answers
   the question it was built for.
5. **Repeat every serving number** before believing it. None of the serving
   comparisons in §3.1 has repeats.

A note on the goal, from the memory work but true generally: **the gate that
matters is not the error in percent — it is whether the ranking of
configurations survives.** An error of a few percent that never changes which
configuration wins is a better outcome than a tighter one that does.
