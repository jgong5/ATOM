# ATOM Compass — Design Point 7: The Calibration and Benchmarking Toolchain

**Status:** draft for review. Drafted by an AI assistant during a design interview; not
yet reviewed or approved. No code has been written against it.

**Depends on:** `04_model_capture_and_cost_ir.md` (what is traced and priced),
`05_machine_spec_and_probes.md` (the machine artifact and its probes).

**Scope.** The offline toolchain that produces the data an online prediction consumes:
what each phase measures, how its output is keyed and digested, when it must be re-run,
and what the user actually types. Empirical modelling only — analytic laws are a separate
design point (D43).

**Terminology, fixed here because the prior effort's taxonomy warns that *"priced is not a
species"*:**

| Term | Unit | Obtained by |
|---|---|---|
| **op price** | one operator signature | microbenchmark, CUDA-graph replay |
| **step measurement** | one whole forward | CUDA events in situ |
| **region term** | one runner-side region (`prepare_*`, sampler, logprobs) | in situ, upper median over warm rows |
| **calibration** | the *act* of deriving model terms from any of the above | — |

"Calibration" must not swallow the other three. A prior conflation applied a 2% kernel
stability gate to a prefill region whose published value carries a **59.5% relative range**
as its declared uncertainty, and concluded the region was unmeasurable.

---

## D36. Three tiers, and what each needs

| Tier | Needs | Used for |
|---|---|---|
| **0 — analytic / roofline** | machine spec + model geometry **only** | day one, before any campaign; a device never measured; fast configuration sweeps |
| **a — coarse empirical** | fitted over features `ScheduledBatch` already carries | structure discovery, plumbing, sweeps |
| **b — op-level empirical** | symbolic IR + priced leaves + region terms | acceptance cells |

Two orthogonal axes: **which tier was asked**, and **how each answer inside it was
obtained** (the provenance ladder of doc 02 D11, extended). This document covers what tier
b (and the empirical half of tier a) needs to exist.

Scope boundary, stated once: **Compass models a device it has been measured on.**
Configurability — dialling the capacity, bandwidth or interconnect of a measured device —
stays in scope. Predicting an unmeasured architecture is tier 0's job. Accordingly, doc 05
D26's `transfer` probe is **deferred**: there is no cross-hardware transfer of empirical
data.

---

## D37. `compass plan` — the unified entry point

### Problem

The toolchain is six artifacts, four phases, three hardware tiers and an invalidation
matrix. Expecting a user to know which of those they need is how coverage rots quietly.

### Decision

**The tool tells the user what to measure. The user does not tell the tool.**

```
compass plan --model M --spec machine.yaml --workload W [--width 1,2,4]
```

It reads the artifact store, recomputes every fingerprint, and emits a **measurement
plan**: which artifacts are valid, which are stale and *why*, which are missing, which
structures will JIT-trace, which shape ranges the sweep must reach, and an estimated GPU
time. Everything else in the toolchain is a step the plan names.

### The flows it consolidates

```
 Flow A  new model, on hardware I have
   spec probes -> plan -> trace -> price -> in-situ -> memory -> validate -> predict

 Flow B  new workload, same model + hardware
   plan -> "all artifacts valid; 2 new structures will JIT-trace" -> predict   [NO GPU]

 Flow C  new parallel width
   plan -> "need: memory readings @TP8, collectives @TP8 (~6 min GPU)" -> predict

 Flow D  software upgrade
   plan -> "price_list stale (aiter changed); graphs stale (torch changed);
            spec capacity still valid" -> re-run only the invalidated subset

 Flow E  configuration sweep
   plan -> predict x N -> rank

 Flow Z  day zero, nothing measured
   plan -> "tier 0 (roofline) available now; tier b needs ~3 h GPU" -> predict@tier0
```

Flow D is the one the invalidation matrix (D41) exists to make cheap. Flow Z is why tier 0
is a tier and not a fallback.

### Preparing a measurement so the empirical model works

This is the question users actually have, and the answer is two tool behaviours rather
than user knowledge:

- **Pre-flight.** `plan` names the shapes the sweep must reach, derived from Phase 0's
  observed population plus a declared margin.
- **Post-hoc.** After an acceptance run, a coverage check names any step that fell outside
  what was measured.

Both must use a **convex hull or k-NN distance, never a per-feature bounding box.** The
bounding box cost three iterations to abandon. The worst case: rung 16 was covered on both
axes *separately* and still came out **22.6% low**, because it held 64 samples at
raggedness exactly 1.00 against a run at 1.18-1.32. And the dimensions are
workload-dependent — 0.6B batches ran at raggedness 2.82-3.85, 27B batches at 1.11-1.42 —
so *"'the sweep covers this' is a statement about a pair, never about a sweep alone."*

### Open issues

- `plan`'s GPU-time estimate has no basis yet; it needs one campaign to calibrate itself.
- What `plan` should do when the store holds an artifact whose fingerprint cannot be
  recomputed (e.g. the source root is gone) is unresolved. Refusing is safe but may be
  unusable.

---

## D38. The phases

```
 Phase 0   discovery            device-free   shapes ONLY, never timings
 Phase 1a  trace                device-free   one per structure, JIT-able
 Phase 1b  price ops            GPU, TP1      compute leaves, shape-parametric
           price collectives    GPU, each W   real process groups, minutes
 Phase 1c  in-situ calibrate    GPU, TP1      full-engine steps
           ...plus ONCE         GPU, TP2      not to calibrate, but to TEST transfer
 Phase 2   memory readings      GPU, each W   one engine startup, no workload
 Phase 3   nothing per client count or workload slice
```

### Phase 0 — discovery

Runs ATOM's real scheduler device-free with tier a, logging the step table:
`num_scheduled_tokens`, `context_lens`, `req_ids`, `block_tables`, plus the scheduler's own
decision. It yields the distinct **structure keys**, the **shape population** actually
reached (batch widths, context multisets, raggedness, rung histogram), and which structures
exist only at width.

**It reads the target workload's shapes and never its timings.** That separation is what
keeps the acceptance corpus held out, and it is auditable — the artifact records exactly
what Phase 0 consumed.

**Never reimplement admission to derive shapes.** A prior attempt re-derived
`Scheduler.schedule()`'s Phase 1/Phase 2 plus `_chunked_prefill_size` and omitted
`_finalize_prefill_chunk`'s `checkpoint_cut`, the `BlockManager`, preemption and the pool —
*"which change which requests coexist, not just chunk sizes"* — so it was not conservative
in either direction. Against the real classes it under-counted prefill steps **3 vs 8** on
one cell and over-counted width **5 seqs vs 4** on another, while landing close in
aggregate.

**Circularity, and how it is closed.** Phase 0's schedule depends on the cost model Phase 1
produces. Rather than iterate: **over-cover with a declared margin and verify the hull
post-hoc.** Cheaper, and it fails closed.

### Phase 1b — op pricing, and why it cannot leave the worker

Two hard placement constraints:

- **AITER registers kernels lazily on first call *in the worker process*.** Importing the
  defining module does not put `gemm_a16w16` in `torch.ops.aiter`; even a process that
  built an engine and generated tokens still reports it missing, because ATOM runs the
  model in a worker subprocess.
- **An operator that walks paged KV cannot be called before the KV cache exists.**

So pricing is a mode inside the runner process, **after CUDA-graph capture**. That is a
feature: the kernels priced are the deployment's own, already autotuned for its shapes.

Method, settled by elimination:

| Route | Verdict |
|---|---|
| per-operator CUDA events in line | **rejected** — 3.9x instrumentation overhead; 327 operators summed to 45.664 ms against a 3.946 ms replayed step |
| per-call back-to-back loop | **rejected** — measures the host. A flat ~30 µs floor from M=1 to M=256; 298 operators x 30 µs = 8.9 ms against a priced total of 9.24 ms, i.e. the price list was the operator count times the floor |
| one event pair per call | **worst** — the device idles through the host dispatch; 52.20 µs at M=4 against ~9 µs of kernel |
| **capture N calls into a CUDA graph and replay** | **the method.** Falls to B=8 and is flat after; `kernel + overhead/B` with a graph-replay overhead of ~5.2-5.7 µs independent of kernel size |

### Phase 1c — and the one extra run

Full-engine steps, TP1. Produces region terms, the host floor, and the residual correction
that makes op prices usable — because **an operator priced alone is not the same operator
in a step**: priced kernels at 98.8% coverage summed to **0.749 of the step**; per kernel
the ratio clusters 0.83-0.93, across all kernels it runs **0.434 to 0.989**.

**Plus one full-engine run at TP2**, not to calibrate but to **test whether 1c transfers**.
It is the cheapest possible test of the one term with no transfer evidence: between TP=1
and TP=2 on the same GPUs, the two overhead constants moved **12% and 7% in opposite
directions** (2.25 -> 1.97 µs/launch, 86.35 -> 92.48 µs/op). If it transfers, TP4/TP8 are
predicted; if not, that is the finding and the recipe grows.

### What is NOT a calibration axis

Client count and workload slice. They change **which shapes occur**, not what a shape
costs, and Phase 0 already told us which.

### Open issues

- Under the old flat-graph design the per-launch constant was absorbing pricing error
  multiplied by launch count — which is why it worked and why it did not transfer. Whether
  the symbolic parametric-leaf design shrinks that residual is a **hope, not evidence**.
- Phase 1c's sensitivity: prefill priced against a real run's own steps was **+0.10%**
  overall but **+3.06% excluding step 0** — a single cold-start error of **-6.75 s** almost
  exactly offsetting **+6.99 s** spread over 105 chunks. The scheduler never sees the
  total; it sees the running sum.

---

## D39. The standalone measurement bench

### Problem

Driving a whole serving stack to take a measurement is slow to iterate and couples the
measurement to the scheduler. But calling the model directly was tried and failed:
`model(input_ids, positions)` dies at `fwd_ctx.context.is_dummy_run` — attention reads a
forward context only the runner establishes, and building it by hand means reimplementing
`prepare_inputs`.

### Decision

**A single-process `ModelRunner` bench, fed *replayed* batches from Phase 0's step table.**

The constraints point at the *runner*, not the server — a single-process `ModelRunner` **is**
the worker, so Phase 1b's placement requirement is satisfied. ATOM already shows how to
drive one without a scheduler: `dummy_execution` (`model_runner.py:1177-1217`) and
`warmup_model` (`:1219-1284`) both fabricate `ScheduledBatch`es by hand.

```
  Phase 0 (device-free)            standalone bench (GPU, one process)
  +--------------+  step table     +------------------------------+
  | real engine  |---------------->| ModelRunner                  |
  | + scheduler  |  num_tokens     |   .forward(replayed batch)   |--> step measurement
  +--------------+  context_lens   |   .prepare_model/postprocess |--> region terms
                    block_tables   |   microbench(graph)          |--> op prices
                    req_ids        +------------------------------+
```

**Batches must be replayed, not synthesised.** Hand-built batches get the composition
wrong in ways that matter:

- Row order alone moves measured decode attention across **1.77x** at one fixed 32-row
  context multiset (grouped-descending 276.1 µs, grouped-ascending 292.2 µs, alternating
  376.5 µs, ladder-interleaved 489.5 µs). Hand-built ladders are grouped by construction.
  **Explicit `block_tables` are mandatory**, not optional, or the treatment is
  unreproducible.
- Raggedness is workload-dependent and a fit is blind to a dimension it never varied: every
  calibration round running one sequence length left raggedness at exactly 1.00 in **all
  2,997 samples** — a column of zeros where a feature should be.

### What each measurement may use

| Measurement | Bench with replayed batches | Synthetic shapes |
|---|---|---|
| op price | yes | **yes** — arguments come from the graph, so the signature is self-contained |
| step measurement | **required** | no |
| region term | **required** | no |

Op pricing on synthetic shapes is what keeps cc-traces off a GPU during calibration.

### Open issues

- A standalone runner must reproduce enough of `EngineCore`'s setup (`get_num_blocks`,
  `allocate_kv_cache`, `capture_cudagraph`) to be faithful. The boundary is not yet drawn.
- Whether replaying a step table reproduces the *forward context* faithfully, including the
  attention metadata built by `prepare_inputs`, is unverified.

---

## D40. Pricing communication

Comms split four ways, and only one is straightforward.

**(a) Dispatcher-visible collectives** — `aiter::all_reduce_` and the fused
`fused_allreduce_rmsnorm_*` family, `qr_all_reduce`. Priced like any operator, **in the
worker where the process group is live**. Measured: `[4,1024]` decode **10.38 µs**,
`[1256,1024]` prefill **71.33 µs**; a TP=2 decode graph holds 58 grouped operators, 57 of
them `all_reduce_`, coverage **98.3%**. One known failure: `c10d::broadcast_` takes a
`ProcessGroup` object no JSON artifact can hold — one operator per step.

**(b) Invisible collectives, and the default path is the invisible one.**
`ATOM_USE_CUSTOM_ALL_GATHER` defaults to **1** (`atom/utils/envs.py:354-355`); its
registration is *commented out* at `aiter/dist/parallel_state.py:467-468` and
`all_gather_reg` / `all_gather_unreg` are in `NONE_WRAPPED_OP`. Zero dispatcher events end
to end — while the RCCL path **is** visible (`c10d::allgather_into_tensor_`). These need
doc 04 D21's **declared node** plus a dedicated standalone benchmark calling the same entry
point.

**(c) MoE all-to-all (EP)** — MORI, called inside `moe_forward`, invisible at any degree.
It also has a property no generic model captures: `_get_dispatch_config` caps `block_num`
at `get_cu_num()` because dispatch and combine use a **grid-wide spin barrier requiring all
blocks co-resident**, so the kernel occupies the entire device. That is doc 04 D19's
`exclusive` join policy, and it is a measured fact rather than a modelling choice. (On an
80-CU MI308X, launching 128 blocks deadlocks.)

**(d) p2p.** PP is `isend`/`recv` on NCCL (`atom/distributed/pp_comm.py:127-164`),
priceable like (a). **KV transfer is not priced at all** — doc 01 D6 simulates it as
`latency + bytes/bandwidth` from the machine spec, which is what makes interconnect
configurable.

### Three rules, all learned by hanging or by being wrong

1. **Price the union of every rank's graphs, not each rank's own.** Signature lists
   differing by a single entry **deadlock rather than fail** — a hang with no error, inside
   the worker.
2. **Anything optional in the pricing loop is optional on every rank or on none.** Pricing
   at TP=4 died in a distributed `recvBytes` with no error of its own, because a
   per-signature kernel breakdown made two extra calls per signature — and for a
   collective, two extra calls are two extra collectives.
3. **Scaling is a table, not a law.** Measured in situ: `cross_device_reduce_1stage`
   9.08 / 10.40 / 10.95 µs at TP 2/4/8 (**+20.6%** across the range) while
   `allgather_lastdim` went 29.00 / 23.73 / **16.39 µs — down 43.5%**. Linear-in-group
   predicts +300%; linear-in-log2 predicts +200%. Both badly wrong. Measure per width.

### The honesty constraint

**A collective's measured duration includes waiting for its peer.** Same shape, two passes
in a fresh container: `cross_device_reduce_1stage` x129 read 14554.5 µs then 3400.9 µs
(**0.23x**) while every compute kernel matched within 4% and the window did not move at
all. So a collective price at TP>1 is an **upper bound on device work**, and any
in-situ-versus-isolated comparison at width carries the same contamination on its in-situ
side.

Also recorded: *"a one-stage reduce on a fully-connected fabric is latency-bound at these
sizes ... which is a claim about this interconnect, not about collectives, and should be
re-measured on a multi-node deployment before being relied on there."*

---

## D41. The artifact store

### Problem

The benchmarking mechanics are largely settled by prior evidence. **Artifact identity is
not**, and it is where this project has repeatedly bled. Eight distinct incidents, all the
same shape — the number was fine, the question of *which artifact answered* was not:

- a price list that could not name the code that produced it — *"not an observation"*
- a bare or merged price file that **states no width and silently prices nothing**
- a hard-coded `sys.path` letting a **stale regions module answer under a current
  registry's name**
- a correct registry digest sitting beside a **stale ATOM package**, because the snapshot
  was `rsync` rather than `git archive`
- **two copies of every script** on one node — a silent divergence that faked two
  modelling bugs
- a head-only price list reporting **2,568 refused operators** when the real number was
  **17**
- a summary field with no range that **dropped a 50.88 µs excursion** from a record
  claiming 2.3% spread
- a notes-only regeneration that **destroyed a reviewed digest**

### The six artifacts

Every entry carries a **key**, a content **digest**, a validity **fingerprint**, and a
**provenance** stanza naming the executed source root.

| Artifact | Produced by | Keyed by |
|---|---|---|
| `machine_spec` | doc 05 probes | (device, software stack) |
| `shape_population` | Phase 0 | (model, workload, engine config) |
| `op_graph` | Phase 1a | **structure** — not shape |
| `price_list` | Phase 1b | **(model, width, source-root digest)** — a triple, never a path |
| `region_terms` | Phase 1c | (model, width) |
| `memory_readings` | Phase 2 | (model, width, utilization, max_num_seqs, max_model_len, kv dtype, block size) |
| `coverage_hull` | derived | the price list it was built from |

### Four rules

1. **A key is a tuple, never a path.** *"A bare or merged price file states no width and
   silently prices nothing."*
2. **The source root is a `git archive` digest, not an rsync.** And the audit must digest
   **every** module that can answer, not just the one named — a hard-coded `sys.path` once
   let a stale regions module answer under a current registry's name.
3. **Resolution names its answer.** Every predicted step records which artifacts answered
   it, and a refusal says *which key missed*. "Price refusal is not graph absence" was a
   day's confusion on its own, and an `incomplete: N/2570` line proves less than it looks.
4. **Handed-off artifacts are immutable.** A regeneration that changes only the notes still
   destroys the reviewed digest. Publish a new one.

### Caching keys must carry the binding, not just the shape

A shape-keyed price cache is **unsound**. `signature_of` reads each operator's `context`,
and `context` is where the binder writes this step's `slot_mapping` and state indices. A
second valid allocation for the same shape moved **64 of 2,439** signatures and took a step
from **32.667 ms over 2,424 priced operators to 28.360 ms over 2,376** — 48 operators
priced under one allocation are unpriced under the other. A shape-only cache answers the
first number, with a complete-coverage claim, for a step that is neither.

Measured cost of getting the key right, device-free, against a 32.7 ms modelled step:
**41.7 ms** uncached (~39 ms of it summing prices), **2.3 ms** shape-keyed and unsound,
**4.3 ms** with the allocation carried.

### Cold costs are accounted once, not twice

Factory build is **9.5-11 s** once; the first `graph_for` on a new shape is **0.27 s**.
These must be journalled, not folded into a per-step average.

**Beware the cold-loop artefact:** an average over a loop whose first iteration is a cache
miss reads **3-7x higher** than steady state. Measure a second, warm loop.

### Capture hygiene, as refusals rather than conventions

- **Trace at step >= 2.** Triton autotunes on a kernel's first launch. Step one recorded
  **90,838 operators** against step two's **101**; `chunk_fwd_kernel_o` alone appeared
  **34,269 times**.
- **Warm with identical *token* counts**, plus `--no-enable_prefix_caching` so the second
  pass really re-prefills. Warming with the same *word* count instead produced **50,451
  tuning launches in a prefill graph of 51,179 operators**, ranks disagreeing (51,179 vs
  50,549), coverage reading **2.7%**. The fix took prefill from 51,179 to **1,280**
  operators and coverage to 62.1%.
- **A failed forward writes no graph.** A 101-operator capture of a 64-layer model was a
  crashed forward written from a `finally` block — *"a well-formed artifact from a failed
  run, structurally valid and merely wrong."*
- **What is written is checked against model depth**, since attention runs once per layer.
- **Per-rank artifacts carry every group's coordinates.** A rank-coordinate function that
  reported only `{"tp": rank}` while the topology was TP=2 x DP=2 made **all four ranks
  resolve to one `graph.tp0.json` and write it in turn** — three quarters of the run's
  evidence discarded, the surviving file looking complete at 807 operators with no error.
  *"A single-writer path is indistinguishable from a correct one by inspection."*
- **Write and read sides go through one naming function.** At TP=2 the calibration wrote
  `steps.tp0.jsonl` / `steps.tp1.jsonl` and the run looked for `steps.jsonl`; both workers
  died on a bare `FileNotFoundError` and the manager reported *"Received unexpected
  SHUTDOWN signal from DP rank 0 during initialization"* — no DP in the run, nothing to do
  with initialization. **A single rank cannot expose this**, because at width one no suffix
  is applied.

### Instrument note, to be carried on every artifact

Pricing repeatability, two runs of one graph on one box back to back with identical code:
summed contribution moved **0.96%**, median per signature **1.18%**, **p90 32%**.
High-occupancy signatures repeat to ~1-3%.

**Any residual quoted below about 2% is quoting the instrument, not the model.**

---

## D42. When: offline, JIT, and `--measure`

| Activity | When | GPU? |
|---|---|---|
| machine spec probes | once per (device, software stack) | tiers 1-3 yes |
| Phase 0 discovery | every run | **no** |
| Phase 1a trace | **JIT on first use of a structure**, then cached | **no** |
| Phase 1b op pricing | offline campaign, or opt-in `--measure` | yes |
| Phase 1c in-situ | offline campaign | yes |
| Phase 2 memory readings | offline, one startup per width | yes |

This follows the project's own note: the runner has no modes, every run simulates, trace
happens on demand at first use and is cached, and `--measure` is a separate flag that does
a real GPU run and caches the result.

**New shape is fine; new operator kind refuses.** Because leaves are shape-parametric, a
shape never traced is priced from the existing law. Only an operator *kind* with no samples
refuses. Under the prior flat design a new shape meant a new graph meant a refusal, which
is why a fixed-shape oracle cost **+47.9% TTFT** the moment serving varied shapes (3
prefill steps of 16 / 256 / 16,128 tokens costing 38.6 / 151.0 / 177.6 ms, all answered as
~47 ms).

### Two refusals that belong with `--measure`

1. **A `--measure` run can never be an acceptance run.** The instrument changes what it
   measures: measure mode cost ~**11 ms of TTFT on the 27B (4%)**, and an earlier version
   that synchronised around each forward made the run **33% slower** (TPOT 3.26 -> 4.33 ms;
   fixed by recording CUDA events on the stream and draining with `query()` rather than
   `synchronize()`, which took perturbation from 1.33x to 0.97x). The standing rule
   generalises: *"admission must come from a run instrumented like the one the prediction
   is judged against."*
2. **A price measured during a simulated run is marked as such**, so it is distinguishable
   from one measured in a dedicated campaign.

### And two hygiene rules for any measurement run

- **Never mix profiled and unprofiled numbers.** Being profiled costs ~**0.7-1.05 µs per
  kernel** — **8.1% of a 27B decode step** (951 kernels, +1.002 ms), 8.5% on the 0.6B. A
  profiled run is ~16% slower end to end. A profile is for attribution *within* a step and
  never for an absolute.
- **Never mix machines.** The same unprofiled 27B TP=4 decode step is **9.774 ms on one box
  and 12.365 ms on another — 26%**, larger than the difference between TP=4 and TP=8 on one
  box.

---

## D43. Invalidation

A global fingerprint would force re-measuring everything on a torch bump. Each artifact
records the fingerprint of **its own dependency row**; on load, recompute and compare;
mismatch **refuses** (warn only under an explicit flag).

| artifact \ invalidated by | ROCm / AITER / RCCL | torch | ATOM src | model | device | engine cfg |
|---|---|---|---|---|---|---|
| `op_graph` | - | **X** | **X** | **X** | - | X (level, cudagraph mode) |
| `price_list` | **X** | **X** | X | - *(shape-parametric)* | **X** | - |
| `region_terms` | X | X | **X** | X | X | X |
| `memory_readings` | **X** | X | X | X | **X** | **X** |
| `machine_spec`: capacity | - | - | - | - | **X** | - |
| `machine_spec`: runtime constants | **X** | - | - | - | **X** | - |
| `machine_spec`: tokenizer terms | - | - | - | **X** *(tokenizer)* | - *(host CPU)* | - |

Two consequences worth noting:

- `price_list` does **not** depend on the model once leaves are shape-parametric. That is
  what lets one campaign serve many shapes.
- The device runtime constants depend on the **library build** as much as the silicon,
  which is why doc 05's schema carries `software_pinned_to` and why a mismatch warns
  loudly. The +5980 MiB appearing the moment width exceeds one is collective buffer sizing;
  the 926 MiB at TP1 is HIP context plus libraries.

### The gate must be verifiable from the artifact, not from the flag

*"A dead gate is worse than no gate."* A prior `PRICE_KERNELS` gate read `WORLD_SIZE`,
which the engine never sets — it spawns its own ranks — so the gate was false in every
worker and per-kernel breakdowns were taken at TP=2 all along. **The proof was in the
artifact: the price list written under the supposedly-off gate carried breakdowns for 164
of its 237 entries.** Worse, an inductor-kernel device fault had been gated "off under
parallelism" on the strength of that gate; five runs then faulted, and each fault was read
as evidence about whatever had changed most recently, because an HSA fault surfaces at the
next synchronise and the signature moves between runs.

So: every gate's state is written into the artifact and checked there.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D36 | Three tiers: analytic/roofline, coarse empirical, op-level empirical. Compass models a device it has been measured on; cross-hardware transfer of empirical data is out of scope, and doc 05 D26's `transfer` probe is deferred. | 2026-09-18 |
| D37 | `compass plan` is the unified entry point. The tool tells the user what to measure. Coverage checks use a convex hull or k-NN distance, never a bounding box. | 2026-09-18 |
| D38 | Phase 0 discovery (shapes only) -> 1a trace -> 1b price -> 1c in-situ (+ one TP2 transfer test) -> Phase 2 memory. Client count and workload are not calibration axes. | 2026-09-18 |
| D39 | A single-process `ModelRunner` bench fed **replayed** batches from Phase 0's step table. Op prices may use synthetic shapes; step and region measurements may not. | 2026-09-18 |
| D40 | Four classes of communication, priced separately. Scaling is measured per width, never modelled. A collective's measured duration is an upper bound on device work. | 2026-09-18 |
| D41 | Six artifacts, each with key, digest, fingerprint and provenance. Keys are tuples; source roots are `git archive` digests; resolution names its answer; handed-off artifacts are immutable. | 2026-09-18 |
| D42 | Trace is JIT and cached; pricing is offline or opt-in `--measure`. A `--measure` run is never an acceptance run. New shape is fine; new operator kind refuses. | 2026-09-18 |
| D43 | Per-artifact invalidation matrix, not a global fingerprint. Every gate's state is written into the artifact and checked there. | 2026-09-18 |

---

## TODO register

| # | Item | Why deferred |
|---|---|---|
| T16 | Calibrate `compass plan`'s GPU-time estimates | needs one campaign |
| T17 | Draw the boundary of the standalone `ModelRunner` bench | needs a first implementation |
| T18 | Verify a replayed step table reproduces the forward context faithfully | unverified, and it gates D39 |
| T19 | Decide the artifact store's physical form: directory convention, manifest, or indexed store | no decision taken |
| T20 | Build the declared-node + standalone benchmark for invisible collectives | custom all-gather is the default path and is entirely invisible |
| T21 | Establish whether Phase 1c transfers across width (the TP2 test run) | it is the recipe's one unproven assumption |
| T22 | Analytic laws as their own design point | deferred by decision; rung 4 of the resolver ladder reserves its slot |
