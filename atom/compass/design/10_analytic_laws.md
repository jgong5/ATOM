# ATOM Compass — Design Point 10: Analytic Laws (Tier 0)

**Status:** draft for review, and **more speculative than every other document here.**
Docs 01–09 are grounded in measurements, most of which were got wrong once before they
were got right. This one has a single existence proof and four recorded failures. Treat
its numbers as targets, not findings.

**Depends on:** `05_machine_spec_and_probes.md` (the device parameters), `04` (the IR whose
shapes feed a law), `07` D36 (the tier and the resolver ladder).

**Scope.** Cost and memory derived from device parameters and model geometry, with **no
measurement of the subject**. Serves three purposes: a standalone roofline tier usable on
day zero, rung 4 of the resolver ladder where empirical data is missing, and the only route
to a device that has never been run.

---

## D63. What "analytic" means, and the three classes

### The word is reserved

From the prior taxonomy: `analytical` means **computed without measuring the subject**.
*"Nothing in Compass is analytical today. The word is reserved, not aspirational."* A value
read back from a file is still `measured`; a fitted coefficient is `fitted`. Only a
quantity derived from declared geometry and declared device parameters is `analytical`.

### The organising principle, from what worked and what didn't

**The one existence proof.** Analytic **weight bytes** — `resident_bytes(model, tied_head)`
asked of a meta build — came out exact at every width: **−0.00 / +0.00 / −0.02 / +0.01%**
at TP=1/2/4/8 on the 0.6B, and 0.00% at TP=2 and TP=4 on the 27B.

**The four failures, and they share a shape.**

| Attempt | Result |
|---|---|
| rotary buffer size from `max_position_embeddings × head_dim × 2` | matched the 0.6B **exactly**, **4× wrong** on the 27B (partial rotary). *Tested on a second model, failed, did not ship.* |
| weight bytes from checkpoint headers (2-D shards, 1-D replicates, tied head dropped) | exact on the dense 0.6B at every width and on the hybrid 27B at TP=2 — **3.3% low** at TP=4 |
| `non_torch` as fixed-plus-per-peer | **no form fits** 5980 / 6340 / 9138 MiB at widths 2/4/8 |
| collective scaling as linear-in-group or linear-in-log₂ | predicts **+300%** / **+200%**; measured **+20.6%** TP2→TP8, and one collective went **down 43.5%** |

> **Analytic works where the quantity is a pure function of declared geometry. It fails
> where the quantity is decided by a runtime, library or kernel-selection policy.**

### The three classes

| Class | Derivable from | Examples | Tier 0 treatment |
|---|---|---|---|
| **A — exact** | model geometry + dtype | weight bytes, KV bytes/token, FLOPs per operator, bytes moved per operator, collective message sizes | computed |
| **B — device-parameterised** | A + device spec | GEMM time, norm/elementwise time, attention time | roofline with derates |
| **C — policy-determined** | nothing | which tuned kernel a shape dispatches to, `non_torch`, load residue, launch overhead, the in-situ gap, fabric scaling | **declared constants** (doc 05) or accepted error |

Class A comes free from the symbolic IR: doc 04's shapes are sympy expressions, so FLOPs
and bytes-moved are closed-form expressions in `T`, `B`, `Ctx`, `TP` without any extra
machinery. The experiment already demonstrated this — `FLOPs(T) = 256·T² + 98304·T`,
exact against a fresh concrete trace at T = 2 / 17 / 64 / 512 / 4096.

---

## D64. The roofline leaf model

For a leaf with `F` flops and `M` bytes moved, both Class-A expressions over the IR's
symbols:

```
  t_compute = F / (peak_flops  × derate_compute)
  t_memory  = M / (bandwidth   × derate_memory)
  t_leaf    = max(t_compute, t_memory)
```

and at step level, the host floor that the prior work established as a **`max`, not an
addend**:

```
  t_step = max( Σ t_leaf ,  launches × h )
```

All four parameters come from the machine spec (doc 05 D25): `compute.*_flops`,
`memory.bandwidth_bytes_per_s`, their `derate`s, and — for `h` — `host.*`.

### Why the host floor is not optional

`step = max(kernel time, launches × h)` at h = **95.8 µs** held four warm prefill shapes
within 2% at a fixed 391 launches:

| tokens | kernels | idle | device window |
|---|---|---|---|
| 794 | 13.384 ms | **23.360 ms (63.6%)** | 36.745 ms |
| 2,294 | 27.037 | 11.195 (29.3%) | 38.232 |
| 6,594 | 110.046 | **0.033 (0.0%)** | 110.080 |
| 15,694 | 418.164 | 0.009 (0.0%) | 418.173 |

The 6,594 row was a **held-out prediction before it ran** and came back at 0.033 ms idle. A
roofline that ignores this term is **63% wrong at small shapes** and right at large ones —
which is exactly the regime a decode step lives in.

And the additive form fails: fitted to the smallest shape it over-predicts the largest by
**+5.6%**; fitted to the largest it under-predicts the smallest by **−63.6%**.

**`h` is a Class-C constant.** It survives a machine change (95.8 vs 95.7 µs on a box whose
GPUs are ~20% faster — *"a floor that ignores a 20% swing in device speed is a floor on the
host side"*) and a width change exactly (TP=4 costs what TP=2 costs: same 505 launches,
same 45.1 ms). It does **not** survive a model change: **95.7 µs on the 0.6B against 125.0
µs on the 27B — 40% apart.**

---

## D65. Four places roofline is known to be wrong, with numbers

These are not speculative. Each was measured, and each is a smooth model meeting a
discontinuous reality.

**1. GEMM tile cliffs.** A 2.4× jump between M=768 (58.07 µs) and M=1024 (137.45 µs) where
the trend says ~77 µs. Twelve distinct tile configurations across twenty values of M. A
roofline is smooth through all of them.

**2. Attention plateaus.** Cost per token sits on flat plateaus quantised in sequence
length — 3.67, 6.28, 6.10, 5.90, 6.01, 8.31, 8.45, 8.17, 8.22, 10.58, 10.41, 10.47, 10.50
µs/token across a 13-point ladder — with the plateau index `ceil(L / 2530)` and steps of
~2.27 µs/token. *"The quadratic term is computed in whole chunks of keys."*

**3. Decode cost falls as batch grows.** At matched total context, batch 1 → 12 costs
3.640 → 3.532 ms (TP=1) and 3.636 → 3.310 ms (TP=2). Twelve sequences cost **less** per
step than one. A roofline predicts growth; the sign is wrong.

**4. Prefill is non-monotone in token count.** 2×400 at 4,588 tokens costs **52.14 ms**
where 4×200 at 4,376 tokens costs **63.16 ms** — fewer tokens, more time, reproducible to
0.03 ms over five repeats.

### Decision

**Tier 0 is a smooth model and is allowed to be.** It must not pretend otherwise:

- **It reports a band, not a point**, wherever a known discontinuity lies inside its input
  range. The dispatch-band boundaries are already recorded in the price list (doc 09 D60)
  when one exists; when it doesn't, the band comes from the geometry (tile size, chunk
  size) rather than from measurement.
- **It is never promoted to rung 4 for a leaf whose known discontinuities fall inside the
  queried range**, unless a measured band table exists to place them.

---

## D66. Collectives

The textbook models are refuted on this fabric. Ring all-reduce predicts `2(N−1)/N · n/BW`,
i.e. roughly +75% from TP2 to TP8; linear-in-group predicts +300%; linear-in-log₂ +200%.
**Measured: `cross_device_reduce_1stage` 9.08 → 10.40 → 10.95 µs at TP 2/4/8 (+20.6%), and
`allgather_lastdim` 29.00 → 23.73 → 16.39 µs (−43.5%).**

The explanation on record: *"a one-stage reduce on a fully-connected fabric is latency-bound
at these sizes ... which is a claim about this interconnect, not about collectives, and
should be re-measured on a multi-node deployment before being relied on there."*

### Decision

A two-regime model keyed off the spec's `interconnect` section:

```
  t = max( link_latency_s × hops(topology, N),
           message_bytes(N, algorithm) / (link_bandwidth × derate) )
```

with the **algorithm named in the spec**, not assumed — because AITER's one-stage reduce,
its quick-reduce, the custom all-gather and RCCL all have different `message_bytes(N)`, and
the default path is `ATOM_USE_CUSTOM_ALL_GATHER=1`, which is **entirely invisible to a
dispatch trace** (doc 07 D40).

Expected accuracy: poor at the sizes ATOM actually uses, because those are latency-bound
and the latency term is the one least constrained by a datasheet. Collectives are ~6.6% of
kernel time at TP=2, 10.4% at TP=4, 13.4% at TP=8 — so a 2× error here is 7–27% of a step
at TP=8.

---

## D67. Analytic memory

This is where tier 0 is strongest, because most terms are Class A.

| Term | Class | Analytic form | Evidence |
|---|---|---|---|
| **weights** | A | ask a meta build, dedupe by storage | **exact at every width** — the existence proof |
| **KV bytes/token** | A | layers × kv_heads/TP × head_dim × 2 × dtype | arithmetic |
| **buffers** | A-ish | **recorded, not formula'd** | the formula matched the 0.6B and was **4× wrong** on the 27B |
| **activations** | B | `k_model × tokens`, k from geometry | linear scaling is **validated** — a 3,494-token trace scaled to an independently measured 4,096-token peak at **+0.0% at TP=1/2/4**. Whether `k` itself is derivable from geometry is **untested**. |
| **invisible scratch** | **C** | — | kernel-internal. **0.1 KB/token on the 0.6B, 39.6 KB/token on the 27B** — worth the difference between −35.0% and +3.4% held out |
| **`non_torch`, load residue** | **C** | — | no form fits; declared per width in the spec |
| **graph pool** | C | — | measured line at W=1, flat 104 MiB above |

So an analytic memory model gets weights and KV **exactly**, activations **approximately**,
and needs Class-C constants from the machine spec for the rest — which is precisely the
arrangement docs 03 and 05 already describe. Tier 0 memory is therefore *not* a new design;
it is the existing memory model with the activation coefficient derived rather than walked.

**The one thing to test:** whether `k_model` can be derived from geometry (hidden,
intermediate, dtype, layer types, TP) rather than measured from a liveness walk. If yes,
memory sizing needs no graph at all. If no, tier 0 memory needs one graph per (model,
width) — cheap, but not zero.

---

## D68. Validation: the empirical campaign *is* the analytic model's validation set

### The structural argument

For every leaf where **both** a measured price and an analytic law exist, the ratio is free
evidence about whether that law can be trusted where no measurement exists. Build tier 0
after the empirical campaign and there is no way to check it except by measuring again.

**So tier 0's laws are authored during the empirical campaign, not after it**, even though
they are not needed until later.

### The promotion rule

A leaf's analytic law earns rung 4 of the resolver ladder when it tracks its measured price
**within the instrument's own repeatability** — 0.96% summed, 1.18% median per signature,
p90 32%, so anything under about **2%** is quoting the instrument (doc 07 D41).

Promotion is **per leaf**, recorded in the artifact, and revocable when a new measurement
contradicts it.

### What is reported

Per leaf family: `analytic / measured` ratio distribution, and the fraction of leaves
promoted. A tier-0-only run reports 100% `analytical` in its `provenance_mix`; a tier-b run
with gaps reports the mixture, and doc 08 D52 refuses an acceptance cell whose analytic
fraction exceeds the declared threshold.

---

## D69. Device transfer, and the derate problem

Tier 0 is the only route to a device never measured. It has a specific weakness that must
be stated rather than discovered.

**Peak FLOPs and bandwidth are datasheet numbers. Achieved performance is
`peak × derate`, and the derate is a property of the kernel library's maturity on that
architecture — not of the silicon.** For a device you have never run, the derate is a
guess, and it is the dominant term.

> **Tier 0 on an unmeasured device is as good as its assumed derate, and nothing in the
> method validates that assumption.**

Two honest mitigations, neither of which removes the caveat:

1. **Report sensitivity.** Every tier-0 prediction on an unmeasured device is reported with
   its derate, and with the prediction recomputed at ±20% derate. A ranking that survives
   that band is more trustworthy than one that doesn't.
2. **Rank, don't predict.** The gate that matters is whether the top-1 configuration choice
   survives (doc 08 D48). A derate error that applies uniformly across configurations may
   leave the ranking intact while moving every absolute number.

---

## D70. Expected accuracy, stated in advance

So the result cannot be graded against an expectation invented afterwards:

| Use | Expected error | Why |
|---|---|---|
| tier 0, measured device, ranking configurations | **plausible** | derates are measured; discontinuities affect configurations similarly |
| tier 0, measured device, absolute latency | **tens of percent** | tile cliffs, plateaus, the decode sign, non-monotone prefill |
| tier 0, unmeasured device | **unbounded, derate-dominated** | D69 |
| rung 4 gap-filling, promoted leaf | **~2%** | the promotion rule requires it |
| rung 4 gap-filling, unpromoted leaf | **unknown** — that is why the fraction is capped | |
| analytic memory: weights, KV | **exact** | Class A |
| analytic memory: block count end to end | **unknown** | depends on activations and the Class-C constants |

Single digits on latency are not a target for tier 0 and should not be claimed. That is
what tier b exists for.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D63 | Three classes: A exact from geometry, B device-parameterised, C policy-determined. Analytic works where the quantity is a pure function of declared geometry and fails where a runtime or library policy decides it. | 2026-09-19 |
| D64 | Roofline leaf `max(F/flops, M/bandwidth)` with derates, plus the host floor as a step-level `max`, not an addend. | 2026-09-19 |
| D65 | Tier 0 is a smooth model and says so: it reports a band where a known discontinuity lies in range, and is not promoted for such leaves without a measured band table. | 2026-09-19 |
| D66 | Two-regime collective model with the algorithm named in the spec. Textbook ring/log models are refuted on this fabric. | 2026-09-19 |
| D67 | Analytic memory is the existing memory model with the activation coefficient derived rather than walked. Weights and KV exact; scratch, `non_torch` and load residue stay declared constants. | 2026-09-19 |
| D68 | Tier 0's laws are authored **during** the empirical campaign, because that campaign is their only validation set. Promotion to rung 4 is per leaf, requires tracking the measured price within ~2%, and is revocable. | 2026-09-19 |
| D69 | On an unmeasured device, tier 0 is derate-dominated. Report the derate and a ±20% sensitivity band; prefer ranking claims to absolute ones. | 2026-09-19 |
| D70 | Expected accuracy is declared in advance, per use. Single-digit latency error is not a tier-0 target. | 2026-09-19 |

---

## TODO register

| # | Item | Why deferred |
|---|---|---|
| T34 | Test whether the activation coefficient `k_model` is derivable from geometry | decides whether tier-0 memory needs a graph at all |
| T35 | Derive FLOPs and bytes-moved expressions for the ~20 opaque leaves | Class A is free from the IR for aten ops, but an opaque leaf's internal work must be described by hand (doc 04 D20's parameter extractor, extended) |
| T36 | Name the collective algorithm per code path in the machine spec schema | doc 05 D25 has no field for it yet |
| T37 | Decide whether `h` (the host floor) is derivable or stays a per-model measured constant | 40% apart between two models; it is the largest single Class-C term |
| T38 | Build the analytic-vs-measured ratio report as part of the empirical campaign | it must exist before the campaign, not after |
| T39 | Establish a dispatch-band table from geometry for leaves with no measured bands | tile sizes and chunk sizes are knowable; whether the *selection rule* is, is not |
