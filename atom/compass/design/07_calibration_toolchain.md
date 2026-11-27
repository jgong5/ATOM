# ATOM Compass — Design Topic 7: The Calibration and Benchmarking Toolchain

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Revised 2026-10-13 by the owner's rulings on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791) and
[#882](https://github.com/jgong5/ATOM/issues/882#issuecomment-6140285611):** the
toolchain's commands are `13` D80's, and D38 says which phase each runs; planning belongs
to the backend that needs the measurements (D37); the bench is drawn and its window
defined (D39); new artifact kinds and invalidation rows (D41, D43).

**Revised 2026-10-14 by the owner's
[ruling on #937](https://github.com/jgong5/ATOM/pull/937#issuecomment-6161726104), which
[#851](https://github.com/jgong5/ATOM/issues/851)'s decisions carry:** Phase 2 is
`compass measure memory`, whose `memory_readings` entry a run reads, so the readings leave
the machine spec (D38, D41, D43); the step grid's planner runs inside `measure steps` (D37); no
command starts the client, and grading decides whether a run is gradeable (D38.1).

**Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952):** two granularities, step-level and
op-level, and a method per term replace the earlier three-way split of cost models (D36);
the backends, artifacts and flags carry the step stub, step linear and step grid names.

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489), amended the same day by its
[bench, band and capture ruling](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252):**
the op-level backend is empirical on a measured device. Capture is concrete, one graph
per decode width and prefill (T, B) (Phase 1a). `measure ops` times each IR leaf cold
from its contract, at points planned from the steps of record directories (#1527),
recording the kernels each point launched (D38,
D39); collectives are leaves measured across every rank of the layout (D40); `fit ops`
publishes a `price_list` that interpolates only between same-kernel neighbours, as `09`
D60 requires, and nothing else, so region terms and the op-level residual are retired
(D38.1, D41, D43); a leaf the price list declines falls to its analytical law, counted
(D36, D42).

**Revised 2026-10-21 by the owner's
[decision on #1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6308315167),
recorded by [#1868](https://github.com/jgong5/ATOM/issues/1868):** `measure ops` runs no
forward (Phase 1b, D39).

**Revised 2026-11-02 by the
[design session on #2640](https://github.com/jgong5/ATOM/issues/2640#issuecomment-6568381628):**
a constructed JIT sweep and a per-stack cache library warm a container before a graded
run (D106).

**Revised 2026-11-02 by the owner's
[Ruling on #2650](https://github.com/jgong5/ATOM/issues/2650#issuecomment-6579531822) and
[Ruling on #2687](https://github.com/jgong5/ATOM/issues/2687#issuecomment-6591321600):**
the sweep's ranks share one walk of the token counts, ATOM's PrefillDelayer coalesces each
designed step with no blocker, and the sweep server differs from the cell's deployment
only in pacing variables that select no kernel or shape (D106).

**Revised 2026-11-03 by [#2716](https://github.com/jgong5/ATOM/issues/2716) and the
[ruling on #2724](https://github.com/jgong5/ATOM/pull/2724#issuecomment-6596632012):** the
`machine_spec` kind gives way to a `probe_record` per node and a `platform_readings` entry
per (device, stack, PTL, torch), and every kind keys the device, stack, PTL state and
torch its invalidation row marks, read from its records, in place of the machine-spec
digest (D41, D43, `05` D24).

**Revised 2026-11-03 by
[#2309's Ruling (e)](https://github.com/jgong5/ATOM/issues/2309#issuecomment-6498578963),
as amended that day:** `record` times the served tokenizer and stamps the HTTP intake, and
`fit host` fits both into the `host_laws` entry, keyed by model as every host law is
(D38, Phase 1c; D43).

**Revised 2026-11-10 by [#2899](https://github.com/jgong5/ATOM/issues/2899):** a law file
carries its memory readings, assumed: the commits that re-key its host key re-key them, and
one that moves what a start under the law's engine config holds takes them again (D41,
D43); a run predicts the host its host laws timed (D41).

**Revised 2026-11-14 by the owner's
[Ruling on #1500](https://github.com/jgong5/ATOM/issues/1500#issuecomment-6870338230)
([#3408](https://github.com/jgong5/ATOM/issues/3408)):** an `op_measurement` keys the
parallel layout only for a collective leaf; any other leaf's point answers at every
layout, and `fit ops` keys a price list at the layout of the runs that listed its points
(D41).

**Revised 2026-11-19 by [#2678](https://github.com/jgong5/ATOM/issues/2678):** a change
that cuts calibration time meets one of three bars (D118); the step bench runs on several
cards once they agree (D39), the op planner plans the decode rungs its steps replay and one
more (Phase 1b), and a step grid prediction with no real run iterates on its own simulated
runs (D37, Phase 0).

**Revised 2026-11-18 by [#3668](https://github.com/jgong5/ATOM/issues/3668) and the
owner's [Ruling on it](https://github.com/jgong5/ATOM/issues/3668#issuecomment-6974348636):** a step linear law may be calibrated from designed load
on the real server, every feature it fits moved apart in each scenario it supports, a
candidate beside `19` D113's record list, which `compass advice` keeps printing (D119).

**Depends on:** `04_model_capture_and_cost_ir.md` (what is traced and priced),
`05_machine_spec_and_probes.md` (the machine artifact and its probes).

**Scope.** The offline toolchain that produces the data an online prediction consumes:
what each phase measures, how its output is keyed and digested, when it must be re-run,
and what the user actually types. Empirical modelling only — analytical laws are a separate
design topic (`10`).

**Terminology, fixed here because "priced" is not a method (doc 02 D11):**

| Term | Unit | Produced by |
|---|---|---|
| **op price** | one IR leaf at one point | `measure ops`: the leaf from its contract at a point ATOM's runner prepared, CUDA-graph replay over cold copies (D38) |
| **step measurement** | one whole forward | CUDA events in situ |
| **calibration** | the *act* of deriving model terms from either of the above | — |

"Calibration" must not swallow the other two. A prior conflation applied a 2% kernel
stability gate to a prefill region whose published value carries a **59.5% relative range**
as its stated uncertainty, and concluded the region was unmeasurable.

---

## D36. Two granularities, and what each needs

Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952), which retires the earlier three-way
split of cost models: it mixed what a backend prices as one unit with where each number
comes from. The op-level row revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489).

| Granularity | Backends | Needs | Used for |
|---|---|---|---|
| **step-level** | the step stub, linear law and grid (run-file keys `step_stub`, `step_linear`, `step_grid`); the step roofline | measured steps, fitted over features `ScheduledBatch` already carries; the roofline only the machine spec and the modules of the meta build | structure discovery, plumbing, sweeps; day zero, a device never measured, fast configuration sweeps |
| **op-level** | the op graph (`OpGraphBackend`, not built) | a concrete IR graph per decode width and prefill (T, B); each leaf priced from a `price_list`, else by its analytical law; no residual | acceptance cells; on a measured device, a new engine config, layout or shape by measuring only the leaf signatures the price list lacks, with no step recapture |

Two orthogonal axes: **the granularity of the backend asked**, and **the method of each
term it returns** (doc 02 D11): analytical, declared, measured or estimated. Fallback runs
only along method inside one granularity: an op-level backend answers an unpriced leaf by
its analytical law, always on; a step-level run falls to the step roofline only where its
run file composes `StepFallbackBackend`; nothing falls from op-level to step-level. This
document covers what the empirical terms of both need to exist.

**One use beyond the table, by ruling.** Topic `17`'s step grid, a step-level backend, is
graded as a candidate for #627's stage 2 by the owner's ruling on #696 (`17` D96). The
"used for" column is otherwise unchanged.

Scope boundary, stated once: **Compass models a device it has been measured on.**
Configurability — dialling the capacity, bandwidth or interconnect of a measured device —
stays in scope. An unmeasured architecture is predicted only by analytical laws (`10`),
declared while their derates are assumed, and a term with an input observed on another
subject (another platform, width, model or engine config) is declared. Accordingly, doc 05
D26 has no `transfer` probe: no empirical data crosses hardware as measured.

---

## D37. The backend that needs a measurement plans it

### Problem

The toolchain is several artifacts, phases, hardware classes and an invalidation matrix.
Expecting a user to know which of those they need is how coverage rots quietly.

### Decision

**The tool tells the user what to measure. The user does not tell the tool.**

**Revised 2026-10-13 by the owner's ruling on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791):** no single
command plans the whole campaign. A backend that needs measurements plans them itself,
beside its own code, and owns the coverage check over them. The step grid's planner and
coverage check are the first (`17` D98, D99): `measure steps --grid` runs the planner
over a `shape_population`, and `compass run --coverage` calls the check before the run
starts, refuses a run that fails it, and records the result for grading; a backend with
no check refuses `--coverage`. The commands, with their inputs and outputs, are `13`
D80's.

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489):** the op-level backend's planner is
the second. `measure ops --plan` reads each step of the record directories it is given (#1527) against the
`op_graph` captured at its shape and places each leaf's points over them (D38, Phase 1b),
and its coverage check answers which steps have every leaf at a measured point or between
two same-kernel neighbours.

### The cases the commands cover

| Case | What runs |
|---|---|
| new model, on hardware I have | D39's recommended flow: its steps half for the step-level backends, and its ops half too for the op-level one |
| new workload, same model and hardware | `trace steps` and the coverage check; a GPU only for steps outside what was measured. A step grid prediction with no real run of the workload runs them round by round, the loop below |
| new parallel width | `measure memory` and `measure` at that width, since memory readings and collectives are per width (D40); `measure ops` there takes only the collectives and the leaf signatures no layout measured (D41) |
| software upgrade | each entry whose D43 row the change invalidates refuses at load, naming its key; only those are re-measured |
| configuration sweep | `measure memory` and `run` per configuration, then rank |
| day zero, nothing measured | the step roofline (`10`), with no measurement, its terms declared while the derates are assumed; memory from a `memory_readings` entry marked assumed (Phase 2) |

The upgrade case is the one the invalidation matrix (D43) exists to make cheap. Day zero
is why the roofline is designed as a backend and not only as a fallback; today a run
reaches it only as the fallback its run file composes (`StepFallbackBackend`).

**A step grid prediction plans from its own simulated runs** (revised 2026-11-19 by
[#2678](https://github.com/jgong5/ATOM/issues/2678), D118, bar 3). A prediction has no real
run of its target to plan from or to check Phase 0's margin against, only its own
simulated run, whose schedule moves with the bundle. Round by round,
`trace steps` simulates the target under the current bundle, the step grid's planner
(`measure steps --grid`) plans over the simulated steps and measures what the store lacks,
`fit steps` runs again, and the next round simulates under the new bundle; the loop ends
when a round's planner places no new point. The last round's run is the prediction. Its
gate is the coverage check over its own steps: no step falls to the step roofline but
where no point can be placed, past a limit (`17` D98), which stays counted apart (`17`
D102). Real runs, where grading has them, stay a planning input, since cost grading prices
their steps (`08` D47), their labels unchanged: under DP a prefill step has none until
[#3509](https://github.com/jgong5/ATOM/issues/3509)'s gated records. On #1233's graded
traces 0 to 255 at c1 to c64 (Qwen3.8-27B TP 1, MI308X), after bundle v2 from the
populations' 421 and 85 points, the loop placed 17 points under v2 and none under v3, and
the steps on the roofline at c1/c4/c16/c64 went from 0/2/3/0 to none: about 0.7 GPU hours,
against about 4.7 to run the workload for real first and plan from its steps
([#2678](https://github.com/jgong5/ATOM/issues/2678)). The op level keeps the coverage
check and one top-up: its planner places points at the steps' own contexts and token
counts (`atom/compass/bench/ops.py::plan`), so a round whose schedule moved places new
ones, and no stop is shown there (T119). The step linear law has no loop: it fits real
records only, and its coverage is `19` D112's requirement 3. What the loop has not shown
is T115 to T117.

### Preparing a measurement so the empirical model works

This is the question users actually have, and the answer is two tool behaviours rather
than user knowledge:

- **Pre-flight.** The backend's planner names the shapes the sweep must reach, derived
  from Phase 0's observed population plus a stated margin.
- **Before a graded run.** The coverage check covers every real step and every Phase 0
  step of the case set, from `record` directories and, for the step grid's, `shape_population` entries, and
  names any step that falls outside what was measured.

Both must use a **convex hull or k-NN distance, never a per-feature bounding box.** The
reasoning, the three iterations it took to abandon the bounding box, and the measured
failures are **doc `09` D58**, which owns coverage geometry because that is a fitting
question. Two consequences the toolchain has to honour, without restating the argument:
the planner's pre-flight and the coverage check both call the same hull test that `09`
specifies, and neither reports coverage for a sweep alone — coverage is a statement about
a **(sweep, workload) pair**.

### Open issues

- What a command should do when the store holds an artifact whose fingerprint cannot be
  recomputed (e.g. the source root is gone) is unresolved. Refusing is safe but may be
  unusable.

---

## D38. The phases

```
 Phase 0   discovery            device-free   shapes ONLY, never timings
 Phase 1a  trace                device-free   one per decode width / prefill (T, B), JIT-able
 Phase 1b  price ops            GPU, each     every leaf from its contract, cold, at planned points
           price collectives    layout        every rank of the layout, the union of signatures
 Phase 1c  in-situ calibrate    GPU, TP1      full-engine steps
           ...plus ONCE         GPU, TP2      not to calibrate, but to TEST transfer
 Phase 2   memory readings      GPU, each cfg one engine start, no workload
 (there is no Phase 3 - see "What is NOT a calibration axis" below)
```

`13` D80's commands run them. Phase 0 is `trace steps`; 1a is `trace ops`; 1b is
`measure ops`, collectives included (D40), then `fit ops`; 1c is `record` for real runs
and `measure steps` on the bench, then `fit steps` and `fit host`; 2 is `measure memory`.
A `measure` command writes measurements per point, and a `fit` command makes the laws,
bundles and leaves from them.

Every phase gets a section. Phases 1a and 2 were named but not elaborated in the first
draft; they are below, after 0, 1b and 1c.

### Phase 0 — discovery

`trace steps` runs ATOM's real scheduler device-free, serving as `run` does, priced by
the backend it names and driven by a client started apart, and writes the step record
(`13` D80): each step's batch, its rows in order with their query and context tokens, and
its rung. It yields the distinct **structure keys**, the
**shape population** actually reached (batch widths, context multisets, raggedness, rung
histogram), and which structures exist only at width, published as one `shape_population`
entry (D41).

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
produces. Where real runs of the target exist, as in a calibration or a validation
campaign, rather than iterate: **over-cover with a stated margin and verify the hull
post-hoc** against the real runs' steps, topping up what they reach outside it (`17`
D98's order). Cheaper, and it fails closed. A step grid prediction has no real run to verify
against, so it iterates on its own simulated runs until a round's planner places no new
point; an op-level one keeps the margin and one top-up (D37; revised 2026-11-19 by
[#2678](https://github.com/jgong5/ATOM/issues/2678), D118, bar 3). The population's key
names the backend that priced it, so populations from
two backends coexist and a planner names the one it read.

### Phase 1a — trace capture

**Device-free.** `trace ops` captures one concrete IR graph per decode width and per
prefill (T, B) it is given or that a record directory's steps hold (`13` D80), using the `TorchDispatchMode` + `FakeTensorMode` mechanism of
doc `04` D18, each leaf with its contract (`04` D20). Revised 2026-10-18 by the owner's
ruling on [#1489](https://github.com/jgong5/ATOM/issues/1489): capture is concrete, not
one symbolic graph per structure. Nothing is allocated on a GPU and no kernel runs, which
is what makes this phase schedulable anywhere, and what makes it the one phase that may
also run lazily inside a simulated run (doc `02`).

Under DP a capture runs one rank. A shape also carries G, the rows the step's MoE
gathers over the DP group, its default `dp x T`; a capture gives the peers G less this
rank's T, split evenly, and a record's steps give each shape its line's `moe_rows`. A
decode gathers other than its rung's default only beside a prefilling peer, which turns
ATOM's MoE to the variable-length gather, so at G a capture's first peer prefills.
Revised 2026-10-20 by the owner's ruling on
[#1731](https://github.com/jgong5/ATOM/issues/1731#issuecomment-6284509620): a
data-parallel graph is keyed by its gathered rows, and the op-level backend reads a DP
step, and each two-batch-overlap micro-batch, at (T, B, G), refused only when no graph is
captured there; a decode's G since
[#1802](https://github.com/jgong5/ATOM/issues/1802), 2026-10-22. Revised 2026-11-09 by the
[ruling on #2956](https://github.com/jgong5/ATOM/issues/2956#issuecomment-6742702610): where
no graph is captured at a DP prefill's own G, a run's lazy trace takes its (T, B), at the
width its LM head pads to, once with G a symbol, at both ends of G's range, and the step reads that graph put at its G
when the two ends agree and the graph's guards admit G, else the graph traced at its G
(`04` D18). `trace ops --symbolic` takes each prefill (T, B) of its shapes so ahead of a
run, beside each G it captures concretely as before.

| | |
|---|---|
| **Input** | the decode widths and prefill (T, B) it is given, and the steps of the record directories it is given, from `record` or `trace steps --out`; the model config; the engine config |
| **Output** | one `op_graph` artifact per shape, keyed by the structure's discrete key, the shape, the torch it was traced under and the digest of the files its trace ran and read (`04` D19, D41) |
| **Cost** | 7.6 s per shape on Qwen3.8-27B TP1, 2.2 s of it the traced forward ([#1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252)'s PoC); a decode shape traces two forwards, one per context its integer arguments are read at (`04` D20) |
| **Needs a GPU** | **no** — but see the caveat below |

**The caveat that makes this phase less free than it looks.** "Device-free" means no GPU
*compute* and no GPU *allocation*. It does not mean no ROCm. `FakeTensorMode.__enter__`
probes the driver to choose the fake device, so a wedged driver hangs a trace that
touches no GPU — which has happened, and reads exactly like a tracing bug. The probe is
stubbable (claim a device exists so both `is_available` and
`_ensureCUDADeviceGuardSet` paths are skipped), but stubbing has its own cost: anything
that trusts `is_available()` and then really touches the device will hang instead. So
Phase 1a is *portable*, not *hermetic*, and a hang here is diagnosed with the 30-second
`rocminfo` check before it is diagnosed as a capture defect.

**What a trace at TP>1 requires** is the open question, not a detail: `T5` asks whether
ATOM's model classes trace cleanly under `FakeTensorMode` at width, and `T52` is the
mode-induced 8-rank hang that gates it. Phase 1a is device-free at TP1 and **unproven at
TP>1**. If it turns out to need a live process group, this phase stops being schedulable
on a CPU box and the campaign's shape changes — which is why both are gating tasks.

**What it does not produce.** A graph is a structure, not a price. Every leaf in it is
unpriced until Phase 1b. A run with complete Phase 1a coverage and no Phase 1b has no
measured leaf: the op-level backend answers every leaf by its analytical law, and the step
reports those seconds by method (`08` D50.1). That is the intended behaviour and a useful
diagnostic rather than a failure.

### Phase 1b — op pricing: each leaf from its contract, cold

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489) and its
[bench, band and capture ruling](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252);
the bench revised 2026-10-21 by the owner's
[decision on #1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6308315167)
([#1868](https://github.com/jgong5/ATOM/issues/1868)).**
`measure ops` times each IR leaf alone, from its contract (`04` D20), at a point ATOM's
runner prepared, and runs no forward: not the model's, not its LM head's, not ATOM's
warmup. The op bench (D39) builds the runner with its parameters on the meta device
(`model_runner.py::RapidServeModelRunner._init_weight_params_on_meta`) and no warmup.
Each placement of a planned point goes through `ModelRunner.prepare_model`, which builds
the metadata and forward context. The bench then calls each leaf with activations random
by its contract's spec and stride; a weight argument built from its contract, random at
its spec, copy c taking the c-th of as many such tensors as the leaf occurs in the step,
or as it has cold copies when those are more ([#1902](https://github.com/jgong5/ATOM/issues/1902));
and metadata, forward context, KV and module tensors from the runner, but an argument the
contract states on the host (`04` D20), which is built on the host, random by its spec as
an activation is, in pinned memory, and bound to no buffer of the runner's
([#2857](https://github.com/jgong5/ATOM/issues/2857)): a CUDA graph replays a copy from
pinned memory only, so a copy with a host tensor the contract states pageable, such as
DeepSeek-V4-Pro's prefill indptrs (`upload_numpy`), refuses by name, as a copy to the host
does ([#3172](https://github.com/jgong5/ATOM/issues/3172)); a 0-dim one another operator
reads as a number is built as any host argument. A store captured before the host mark
keeps timing DeepSeek-V4.1's staged copies device to device, and one captured between the
two marks V4-Pro's indptrs as pinned copies; a store publishes an entry once, so timing
either as captured takes a capture into a fresh store. Before the first
point a runner that defers its output (`tokenIDProcessor.is_deferred_out`) makes one
sampled-id handoff to the host on its token processor's copy stream, as
`ModelRunner.postprocess` does after every forward of such a runner: in a runner that
had made none, Qwen3.8-27B's GDN decode leaf read 5% under one that had (#1902).
`_init_weight_params_on_meta` makes every parameter meta as it is registered and leaves
buffers and other tensor attributes on the device (`02` D10.1), so such a tensor a leaf
reads through a module reference is the model's own. A module a leaf references, by its
path or through its layer's forward-context key, has its meta parameters moved to the
device and filled as ATOM's `--load_dummy xavier` fills a model's
(`loader.py::initialize_dummy_weights`): GDN reads its layer's `conv1d.weight`, `A_log`
and `dt_bias` so, 3.95 MB over Qwen3.8-27B's 48 GDN layers
([ruling on #1869](https://github.com/jgong5/ATOM/issues/1869#issuecomment-6321134450)).
Attribution from profiled steps is not built.

Two placement constraints put pricing inside the runner process, and the op bench is
that process:

- **AITER registers kernels lazily on first call *in the calling process*.** Importing the
  defining module does not put `gemm_a16w16` in `torch.ops.aiter`; even a process that
  built an engine and generated tokens still reports it missing, because ATOM runs the
  model in a worker subprocess. The bench calls each leaf in the process that holds
  ATOM's runner; a leaf whose operator cannot run from its contract refuses by name.
- **An operator that walks paged KV needs a KV cache.** The runner's, with the block
  table `prepare_model` builds for the point's batch.

Weights are random by spec, so the runner holds no device memory for the model's
weights but the referenced modules' parameters. The [#1489 PoC](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252)
read one GEMM signature 5% high on random weights against ATOM's dummy weights;
[#1869](https://github.com/jgong5/ATOM/issues/1869)'s named result compares every leaf
with the bench it replaces, which ran ATOM's warmup on ATOM's dummy weights.

Method, settled by elimination:

| Route | Verdict |
|---|---|
| per-operator CUDA events in line | **rejected** — 3.9x instrumentation overhead; 327 operators summed to 45.664 ms against a 3.946 ms replayed step |
| per-call back-to-back loop | **rejected** — measures the host. A flat ~30 µs floor from M=1 to M=256; 298 operators x 30 µs = 8.9 ms against a priced total of 9.24 ms, i.e. the price list was the operator count times the floor |
| one event pair per call | **worst** — the device idles through the host dispatch; 52.20 µs at M=4 against ~9 µs of kernel |
| **capture N calls into a CUDA graph and replay** | **the method.** A replay of K calls reads `kernel + F/K` a call, F the replay's fixed cost: 18.7 and 19.4 µs for the mrope leaf on MI308X between K = 32 and 256 ([#1513](https://github.com/jgong5/ATOM/pull/1513)), so a 2.7 µs leaf still reads 16% high at K = 32. A graph holds the same K (`atom/compass/bench/ops.py::CALLS`), which adds F/K, about 0.1 µs, to every price, unless K calls pass 10 ms of device time (`GRAPH_NS`): then it holds as many calls as fill 10 ms, at least one, so a capped graph holds at least 5 ms and F is under 0.4% of the price. After each replay the host idles a tenth of its device time (`REST`): replayed back to back, the lm_head GEMM at decode B = 16 ran an MI308X at a 1100 MHz memory and 1550 MHz fabric clock, against 1300 and 1800 MHz with an idle after each call, at the same socket power (a power limit is the suspected cause), and read up to 2.55% over ATOM's decode step by block median; with the idle it reads 0.22% to 0.58% under the step ([#1684](https://github.com/jgong5/ATOM/issues/1684)). A cached prefill attention call runs up to about 1 s, and at a fixed K its grid cost about 350 GPU-hours ([#1620](https://github.com/jgong5/ATOM/issues/1620)) |

**Cold, by rotating copies.** The replay rotates over copies of the leaf's inputs and
ambient state whose total passes twice the device's last-level cache, at most one copy a
call (`ops.py::CALLS`), so a call reads data no call of the same replay brought into the
cache. Before each replay of the cold graph, warmups included, the bench reads a device
buffer of twice the last-level cache (`ops.py::COLD`), outside the timed events and before
the ranks start together ([ruling on #1927](https://github.com/jgong5/ATOM/issues/1927#issuecomment-6355887860)),
so no call reads what the replay before it left there either. The buffer is read, not
written, so the timed replay writes none of it back
([ruling on #2019](https://github.com/jgong5/ATOM/issues/2019#issuecomment-6368138779)).
Read or written, it also evicts the graph's kernel arguments, which a replayed graph keeps
in a device-memory pool that stays in L2 or the MALL between replays: on MI355X each
kernel then waits about 0.3 µs longer for its arguments, about 80 µs on a replay of 256
calls, +19% on a 0.5 ms graph and about +1% on GEMMs already past twice the cache; on
MI308X the read leaves them cached
([root cause](https://github.com/jgong5/ATOM/issues/2019#issuecomment-6378825038)).
ATOM's own decode step leaves them cold as well: DeepSeek-V4-Pro on eight MI355X moved
+16.4, -1.3 and -6.9 µs at B = 1, 16 and 128 with the same read before each step, inside
its repeat spread, against 0.72 to 0.79 ms had they stayed cached
([(d)](https://github.com/jgong5/ATOM/pull/2038#issuecomment-6398675137)), so a cold price
includes the wait.
Without the eviction a leaf whose
`CALLS` copies stay under twice the cache read warm: `indexer_score_topk` at
DeepSeek-V4-Pro decode B=16 on MI355X, 90 MB of copies, read `warm_over_cold` 0.939
([#1927](https://github.com/jgong5/ATOM/issues/1927#issuecomment-6353682488)). The
warm graph gets no eviction. The copies include tensors the leaf reads through module attributes: the mrope
leaf's cos/sin caches are `rotary_emb` attributes, counted in its bytes. A weight leaf
also cycles over one weight of its spec per instance, the population its decode step
reads: one weight can read a few percent off another of the same spec
([#1580](https://github.com/jgong5/ATOM/issues/1580)). Copies past the cache-sized count
reuse that count's activations, so they hold no memory beyond the weights. The prototype's
mrope reading, 3.3% to 11.5% under the same call timed where ATOM's decode step makes it,
was the fixed replay cost at two call counts (above), not warm caches. The warm-over-cold ratio, the same point replayed on one copy, one an occurrence where ATOM passes later occurrences a buffer of their own, is reported per point
as a diagnostic and prices nothing. A graph under K calls has no copy floor and no warm
graph: its 10 ms of calls pass the last-level cache many times over, so warm over cold
reads about 1. The card's PTL state is read and keyed as
[#1434](https://github.com/jgong5/ATOM/issues/1434) rules.

**Each release cleared before the next capture
([ruling on #1932](https://github.com/jgong5/ATOM/issues/1932)).** The driver clears VRAM
a process returns in the background, about 31 GiB/s on MI308X, and `torch.cuda.graph`
returns the caching allocator's free memory before each capture: the previous leaf's
weights, up to 21 GiB for Qwen3.8-27B. A memory-bound leaf timed during that clear read up
to 1.20 of the same leaf timed after it ([#1947](https://github.com/jgong5/ATOM/issues/1947)).
So before each capture, and after the runner frees the hold that sized its KV pool, the
bench returns that memory itself and waits until the card's VRAM in use
(`mem_info_vram_used` in the card's PCI sysfs directory) falls by what it returned, at
most 3 s (`ops.py::_release`). Each point records the seconds waited since the point before
it and how many waits ended at the cap, which leaves a clear possibly still running while
it was timed.

**Points, planned from the steps of record directories (D37).** Per leaf: each decode rung
the steps replay and the ladder's next rung above the widest of them, a stated margin as
Phase 0's (revised 2026-11-19 by [#2678](https://github.com/jgong5/ATOM/issues/2678),
D118, bar 3: on #1233's c1, c4 and c16 records, 1,980 of the first plan's 4,884
context-dependent decode points, 41%, sat on rungs 48 to 512, which no step reached; the
rung above costs about 66 of them, an even share of those 30 rungs). A decode leaf's family
is its width (`atom/compass/bench/ops.py::_family`), so a step at a rung the plan left out
prices its leaves by their analytical laws: `compass run --op-graph --coverage` counts such
steps per leaf kind before a graded run, which grading requires (`13` D80), and the plan
extended over that run's records measures only the points the store lacks.
Prefill rows on
a ladder of chunk remainders, a kind that depends on context also over the same ladder of
the rows' cached tokens, once its points at the most cached tokens still differ from its
fresh ones in kernels or time ([#1706](https://github.com/jgong5/ATOM/issues/1706)): a ladder twice as wide missed a same-kernel attention midpoint by
1.0%, its repeats spreading 0.4% ([#1620](https://github.com/jgong5/ATOM/issues/1620)); attention's context features chosen per kind by data, under
[#941](https://github.com/jgong5/ATOM/issues/941)'s rule. MoE leaves run on independent
uniform routing. A padded decode's MoE runs real rows and pad rows that ATOM routes to no
expert, 13 real of 32 gathered at p50 on a 1P1D DPA-TP8 decode side at rung 4, where the
rung-4 leaf prices 0.55 of its every-row time
([#3239](https://github.com/jgong5/ATOM/issues/3239#issuecomment-6861371586)). So each
rung's routed leaf, whose ids are every row its MoE runs, is measured too at each count of
real rows its steps' MoE ran and at one row's tokens, not on a ladder, since that leaf's time
bends between counts 3 or 4 apart by up to 9% under one kernel
([#3399](https://github.com/jgong5/ATOM/pull/3399)): a point states them as `R`, routes
each row past them to no expert, the real ones spread evenly over the DP ranks and first in
each rank's block of the gather, and on one rank its batch holds them alone, padded to the
rung. `fit ops` interpolates such a family along its pad rows, the rows its ids route less
`R`, 0 at every row, and a run asks it at the step's real rows: the DP group's tokens with its
dummy forwards' rows as one, or one rank's own. A
drafter's routed leaves stay at every row. Each point records the kernels the leaf launched there, its kernel
identity, and each one's grid, by which a trace run's drafter kernel finds its own length
(`09` D54, [#2415](https://github.com/jgong5/ATOM/issues/2415)).

**The price.** `fit ops` publishes a `price_list`: per leaf, piecewise-linear
interpolation between same-kernel neighbours only, as `09` D60 requires. A query at a
measured point takes that point's price; one between two points that launched the same
kernel is interpolated; any other query, outside the measured range, between two
kernels, or along a line whose held-out points show a hidden band or an error over 5%
(`09` D60), is declined with its reason, and the leaf falls to its analytical law, its
share counted (D42, `08` D50.3). Which points are not held out, as adjacent to both
neighbours, is [`09` D60](09_fitting_and_law_selection.md#d60-cost-is-piecewise-interpolation-is-not-sound-but-imprecise)'s rule.

### Phase 1c — and the one extra run

Full-engine steps, TP1. Produces the step-level backends' measurements and the real runs
the host laws are fitted on.

**The trace run** ([#2072](https://github.com/jgong5/ATOM/issues/2072)): a second run of
a record's recipe and client, `compass record --trace`, under ATOM's kineto profiler on
every TP rank without Python stacks, from before the first request to the server's stop,
no rank pausing (`09` D54).
`atom/compass/parity/busy.py` reduces its traces to each forward's busy time per pass,
which labels prefill (`09` D54); its decodes and host segments are profiled and label
nothing. On every rank the recorder marks where a forward's target ends and where each
drafter pass runs with profiler ranges, opened only while the profiler runs.

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489):** Phase 1c produces nothing the
op-level backend reads. That backend has no region terms and no residual correction: it
composes its leaves by the IR's joins, and the gap between a composed step and the
measured one is reported per phase and op kind (`08` D50.3), never fitted away. That
**an operator priced alone is not the same operator in a step** is why the gap is
reported: priced kernels at 98.8% coverage summed to **0.749 of the step**; per kernel the ratio clusters 0.83-0.93,
across all kernels it runs **0.434 to 0.989**.

**Revised 2026-10-11 by the owner's ruling on
[#815](https://github.com/jgong5/ATOM/issues/815):** every backend prices device time only
(`02` D11), so a step measurement is device busy time, and the host floor is no op-level
term. Host time is `09` D54's host laws, composed by `01` D4's recurrence.

**The tokenizer and the HTTP intake are host laws too** (revised 2026-11-03 by
[#2309's Ruling (e)](https://github.com/jgong5/ATOM/issues/2309#issuecomment-6498578963)
and the [#2681 transition](https://github.com/jgong5/ATOM/issues/2309#issuecomment-6591319181)).
A `record` server times each tokenizer call it serves, `encode`, `decode` and the Rust
backend's `encode_batch`, and ATOM's segment tokenizer call around its `encode_batch`,
`segment`, with the characters its stored-prefix search compared, each with its tokens,
its items, its thread, the backend and the loaded
`tokenizer.json`'s fingerprint, and stamps each HTTP request's intake where
`scripts/compass/http_intake.py serve` does: `app`, `body`, `tmpl`, `prep` and `send`, with
the body's bytes, each on the wall clock and on its thread's CPU clock, and on the loop
`wait`, from the tokenizer pool job's submit to the request's resume, and `recv`, each
receive of the body's read; and `out`, each row of a streamed engine output the server's
output thread takes in, and `head`, each SSE head sent. The lines reach the record at the
server's stop, its exit or the SIGTERM a stop sends: until then they are in files in the
server's temp directory, which `steps/<name>.local` names, since on NFS a line written right
after another waits for the first's round trip inside the stage it stamps
([#2870](https://github.com/jgong5/ATOM/issues/2870)); a temp directory on NFS or CIFS
refuses. A server that died first leaves the `.local` file, naming a path inside its
container, and `fit host` refuses the record: copy the file into `steps/` with
`podman cp` and delete the `.local` file while that container runs. A tmpfs `/tmp`, as
QUICKSTART's container mounts, goes with the container's stop, and the record is taken
again. `fit host` fits both
into the entry's `tokenizer` and `http_intake` terms (`05` D25) with the host laws' solver,
drops and hulls (`09` D53, D58). Compass does not model the GIL, so no fitted term may carry
a GIL wait
([#2751's Ruling (e)](https://github.com/jgong5/ATOM/issues/2751#issuecomment-6620615130),
[and the tokenizer Ruling](https://github.com/jgong5/ATOM/issues/2751#issuecomment-6655175759)):

- `tokenizer`: on the wall time of lone calls, stating the records' backend and
  fingerprint. A lone call has no other tokenizer call and no other request's intake in
  flight between its start and end, so it holds no other request's GIL wait; the waits the
  server's own threads add to it are a stated residual. A one-item call, `encode` or a
  one-item `encode_batch`, reads the same on the wall and on its thread's CPU clock, and
  gives `encode`, `fixed_s + tokens / tokens_per_s`. A several-item `encode_batch` runs its
  items on the library's pool, where no thread clock sees them, and gives `encode_batch`,
  in the same form with its own coefficients: a term per item adds nothing, since nine in
  ten such calls carry two items ([#2751](https://github.com/jgong5/ATOM/issues/2751)).
  `decode` is fitted apart, so one
  kind's refusal leaves the others fitted; with no positive slope over its tokens it is
  `fixed_s` alone, priced inside its token hull. `segment`, the segment tokenizer call's
  own work, is its wall less the calls it makes, those on its thread that start inside it,
  over the ids they return and, where every call's record holds them, the characters its
  stored-prefix search compared, else listed unfitted as `tokenizer.segment.chars_per_s`;
  it is lone among the calls it does not make
  ([#2618](https://github.com/jgong5/ATOM/issues/2618), [#3083](https://github.com/jgong5/ATOM/issues/3083)).
- `http_intake`: on the loop thread's CPU time. `per_byte_s` and `idle_s` from requests
  that found no other intake on the loop, each the loop's CPU time from its app call to
  its send's end less its `wait`, which holds the loop's work for other streams while the
  pool job runs
  ([the exits Ruling](https://github.com/jgong5/ATOM/issues/2751#issuecomment-6686324280)).
  A request is lone when every request whose body was read before its app call was sent,
  counted by bodies, not app calls: two intakes that overlap share one app call, and a count
  of those fell behind the sends for the rest of the record, so no record past c=1 had a
  lone request and every run charged no intake ([#3748](https://github.com/jgong5/ATOM/issues/3748)). Where the API server renders
  the template on a pool thread, the loop awaits it and the template job charges only its
  encode calls, so a lone request's seconds add that render's wall, and only templates
  rendered on the loop's thread make bursts.
  Its `recv` stretches stay in: the transport parses the request's own body as the read
  awaits it, and on a loaded loop serves other streams there too
  ([#2919](https://github.com/jgong5/ATOM/issues/2919)). `fixed_s` is the mean, over every
  burst's later requests in the records, of the loop's CPU time from the mark before each
  to its own, less that `per_byte_s` times its body; none is dropped, so a
  fresh server's cold gap stays in, and the report gives the values' p10 and p90
  ([#3020](https://github.com/jgong5/ATOM/issues/3020)). A request's mark is its template
  start or, on a server that renders none, such as a 1P1D decode side, its pool job's
  submit, the first stamp after its body's parse
  ([#3229](https://github.com/jgong5/ATOM/issues/3229)). With no
  burst, or bursts whose requests take less of the loop than that `per_byte_s` charges their
  bodies, which would leave `fixed_s` negative, `fixed_s` is `idle_s` and the report says
  why ([#3368](https://github.com/jgong5/ATOM/issues/3368), [#3384](https://github.com/jgong5/ATOM/issues/3384)). No first-turn cell is recorded: cc-traces sessions
  begin mid-conversation, so a burst is a client run's start. A burst alone would
  underprice later turns: on mi355-gpu-40 a V4.1-Flash
  first-turn body takes 5.64 to 5.80 ns a byte to parse and template, and the c=16 cell's
  later turns, with more messages, 7.88
  ([#2681](https://github.com/jgong5/ATOM/issues/2681)). The
  hull spans body bytes, so a law fitted on 47 KB first-turn bodies refuses a 388 KB body,
  naming the hull, rather than extrapolating.
- `hops`: `admission_s`, `ipc_s`, `ipc_out_s` and `stream_s` (`01`, Lookahead sources),
  each the p10 of the messages alone on its hop, sent once every message ahead of it
  arrived and arrived before the next was sent, so none queues behind another; a client
  misses the requests in flight at its stop, and the pairing allows for them. The pairs,
  on one CLOCK_MONOTONIC: a client send, aiperf's `start_perf_ns`, to its request's app
  call; an engine send call's return to the engine input thread's receipt
  (`dp<r>.handoff.jsonl`); an output's put on the engine's output queue to the server's
  `out` of its first row; a `head` send call's return to the client's ack, aiperf's
  `request_ack_ns` less its `request_start_ns` from that `start_perf_ns`. A message that
  arrived before its send call returned is left out. The client's side is the aiperf raw
  records beside each `--client-run` export. `ipc_s` counts only receipts inside a
  forward of the receiving DP rank, where the step loop waits on the device: an idle
  engine's loop holds the GIL its input thread waits on, 5.25 ms on node 18 against
  0.107 ms stepping ([#2600](https://github.com/jgong5/ATOM/issues/2600)), and between
  forwards the step loop runs Python. The other three hops end in the server or the
  client, which that loop does not hold.
  `kv_write_req_s` is not fitted and stays
  [#2603](https://github.com/jgong5/ATOM/issues/2603)'s stated p10: fitting it needs
  decode's and prefill's records on one clock
  ([#2714](https://github.com/jgong5/ATOM/issues/2714)).
- A term whose calls or requests the records lack is listed unfitted; nothing falls back
  to a spec's value. A committed law file states, until `fit host` over its cell's records
  replaces them, the terms its machine's spec stated for its model, gpu-40's `per_byte_s`
  being #2681's lone-run fit ([#2751](https://github.com/jgong5/ATOM/issues/2751)).

**Why every host-law term stays keyed by model.** The key is D41's: host CPU, ATOM
source, model, engine config, device, stack and torch.

- Step host laws: input preparation varies with the architecture, sampling with the
  vocabulary.
- HTTP intake: the parse and the chat template vary with the model's bodies.
- Tokenizer: each model ships its own `tokenizer.json` (gpu-40: V4-Pro 928k, V4.1-Flash
  912k tokens/s). The model's digest does not cover that file; the entry's fingerprint
  does.

The tokenizer is not split off: one record yields all three terms at no extra cost, the
engine config exposes contention at high concurrency, and a tokenizer-only key needs a
store kind of its own, which the Ruling excludes. A tokenizer estimate for a
configuration no record ran on is [#2483](https://github.com/jgong5/ATOM/issues/2483)'s.
The host CPU in the key retires T53: a rate fitted on one host answers for no other.

**Plus one full-engine run at TP2**, not to calibrate but to **test whether 1c transfers**.
It is the cheapest possible test of the one term with no transfer evidence: between TP=1
and TP=2 on the same GPUs, the two overhead constants moved **12% and 7% in opposite
directions** (2.25 -> 1.97 µs/launch, 86.35 -> 92.48 µs/op). If it transfers, TP4/TP8 are
predicted; if not, that is the finding and the recipe grows.

### Phase 2 — memory readings

**One engine start per configuration, no workload at all.**
`compass measure memory --gpu G --store S --out D -- <ATOM args>` starts the
real engine, reads the readings below on every rank once the KV cache is allocated and
the graphs are captured, publishes one `memory_readings` entry (D41), and exits (`13`
D80); `record` takes none. The entry is keyed by the model, the device, stack and PTL
state its cards read and torch, the parallel layout and the engine config, its KV offload connector and resolved
`index_cache_dtype` included (any `kv_transfer_config` resolves DeepSeek-V4's
indexer to fp8, so a PD start reads a plain fp8-indexer start's entry)
(`atom/compass/run/memory.py::key_fields`). A run reads its rank's
member, or the law file's it prices by (D41), and refuses with neither, so the readings leave the machine spec (doc
`05` D25). This phase exists because these memory quantities cannot be derived from
geometry and have no law behind them. It is the cheapest phase in the
campaign and the one with the tightest acceptance gate behind it — KV block count at
**≤5%**.

| | |
|---|---|
| **Input** | ATOM's arguments: the model, its widths, and the engine config that fixes the capture ladder |
| **Output** | one `memory_readings` entry for that configuration (D41) |
| **Cost** | one start per configuration, two on a cold inductor cache. Minutes, no workload, no traffic |
| **Needs a GPU** | **yes**, and a **quiet** one |

**What is read, and where it comes from:**

| Reading | Source |
|---|---|
| `capacity_bytes` | the device, once; the capacity a run sizes against stays the device datasheet's configured one (`13` D78) |
| `driver_and_collective_reserve_bytes` | ATOM's own `get_num_blocks` `non_torch` term, from a start in which inductor autotuned no kernel and its FX graph cache missed no graph: the configs autotuning tries stay loaded, +12 to 26 MiB a rank on a cold cache (Qwen3.8-27B TP2), and a start that compiled every graph afresh on a cache a Compass-off start keyed, autotuning nothing, read 2 MiB more and sized 1 to 2 KV blocks fewer (Qwen3-0.6B TP1, [#1517](https://github.com/jgong5/ATOM/issues/1517)), so such a start is started again on the cache it warmed, under a free `ATOM_DP_MASTER_PORT`, since the first start to load ATOM's compiled-graph cache autotunes again ([#1428](https://github.com/jgong5/ATOM/issues/1428)) |
| `allocator_retained_after_load_bytes` | torch allocator, after weight load completes, less the model's parameter and buffer storages |
| `persistent_forward_buffer_bytes` | allocated bytes where `get_num_blocks` reads the device, less those after load: `allocate_forward_vars`, attention metadata setup |
| `persistent_at_page_bound_bytes` | allocated bytes where the attention builder bounds its PAGEs by them, less those after load: the drafter, `allocate_forward_vars` and the builder's own so far. Only a start whose builder reads it states one (DeepSeek-V4.1's); a simulated start answers it there under `--index-cache-dtype fp4` and refuses without it ([#3501](https://github.com/jgong5/ATOM/issues/3501)) |
| `activation_bytes` over `warmup_tokens` | ATOM's one warmup forward's allocator peak above what stays allocated, as a serving start reads it, from a start in which inductor autotuned no kernel: on a cold compile cache the peak sat in inductor's autotuning tensors, +15.8% (Qwen3-0.6B) and +65.2% (Qwen3-30B-A3B) ([#705](https://github.com/jgong5/ATOM/issues/705)). A second warmup in the same start reads higher than the first, which is the one that sizes the KV pool ([#1481](https://github.com/jgong5/ATOM/issues/1481)) |
| `cudagraph_pool_bytes` | reserved delta across capture, at the configuration's own ladder, as `capture_cudagraph` returns it |

**Three rules that turn this from a reading into evidence:**

1. **Quiet machine, checked before *and* after.** `non_torch` is a **device-wide**
   reading, so a neighbour's allocation lands in it and is indistinguishable from ours.
   Three of the last five prior pilot attempts were lost or degraded this way — two
   refused to start against 141 GB of neighbour, one ran 64% slow. A check that only
   runs before the measurement does not catch a tenant that arrives during it.
2. **Reset the allocator high-water mark around the measured step.** Otherwise
   `peak_torch` belongs to the warmup prefill, whose shape is nobody's choice — measured
   at **−14.7%** on one configuration.
3. **A configuration with no entry, and no law file's readings (D41), refuses rather than defaults.** `run` names it. This
   is the one place with no fallback, because a silently defaulted memory constant
   produces a plausible KV block count against a 5% gate.

**The checkpoint is loaded.** ATOM's `--load_dummy` skips it, and `measure memory`
refuses it until its readings match a real load's on Qwen3.8-27B TP1 on MI308X (T100).

**A device never measured** has no engine to start. Developer tooling outside the CLI
publishes an entry for its configuration, every reading marked assumed, as the frozen MI455
projection's entry is (`16` S3c.5); `run` reads it as any entry, and every term built on it
is declared, its note naming the assumption.

**Why it is per width and not fitted across widths:** because it does not fit. The
measured `non_torch` sequence is 926 / 6906 / 7266 / 10704 MiB at widths 1/2/4/8, and
**no fixed-plus-per-peer form reproduces** the deltas 5980 / 6340 / 9138. A table, not
a law — which is also why Phase 2 is per configuration rather than a one-off.

### What is NOT a calibration axis — and why there is no Phase 3

Client count and workload slice. They change **which shapes occur**, not what a shape
costs, and Phase 0 already told us which. There is deliberately no Phase 3: the numbered
list stops at 2 because adding a per-client-count or per-slice phase would multiply the
campaign by a factor that buys nothing. The place client count *does* matter is
validation (doc `08`), not calibration.

### Open issues

- Under the old flat-graph design the per-launch constant was absorbing pricing error
  multiplied by launch count — which is why it worked and why it did not transfer. With no
  residual now, the composed-over-measured gap (`08` D50.3) shows that error; whether
  pricing each leaf at its own points shrinks it is a **hope, not evidence**.
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

**The bench is `EngineCore` without its scheduler, fed *replayed* batches.** Revised
2026-10-13 by the owner's ruling on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791), which draws
the boundary this decision first left open (T17).

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252): this is
the step bench, and the op bench is apart from it. Revised 2026-10-21 by the owner's
[decision on #1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6308315167)
([#1868](https://github.com/jgong5/ATOM/issues/1868)): the op bench runs no forward.**
It builds ATOM's runner for the whole model with its parameters on the meta device
(`RapidServeModelRunner._init_weight_params_on_meta`), skips ATOM's warmup, and sizes
the KV pool without a forward. Each placement of a planned point reaches the
runner as a batch through `ModelRunner.prepare_model`, so ATOM builds the metadata and
forward context as for any step, and the bench snapshots them; it then times each leaf of
the point's `op_graph` from its contract, cold (D38, Phase 1b). No forward of the
model, its LM head or its sampler runs. Nothing of ATOM's input preparation is copied.
At width it starts every rank of the layout through ATOM's own group initialisation
(D40). On one GPU at a DP-attention layout it builds the runner at one DP rank's own
layout, TP 1 and DP 1, while the `op_graph` it reads and every record key keep the
deployed layout; a leaf's weights come from its contract, so it runs at its deployed
shard, and a collective leaf refuses by name, measured across the layout's ranks instead
([#1870](https://github.com/jgong5/ATOM/issues/1870)). The step bench is not
reused for ops: it times whole forwards, and it refuses DP.

Measured on [#1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252)'s
PoC, Qwen3.8-27B TP1: a leaf timed from its contract with no model forward, its
ambient state from ATOM's input preparation, came within 1.6% of the same call timed in
ATOM's decode step for attention, GDN, six GEMM signatures and `silu_and_mul` at B = 2
and 16, and within 1.1% for nine prefill leaves at two points. That PoC held
ATOM's dummy weights and ran ATOM's warmup. The runner is the whole model's: a 4-layer
runner read GDN 12.6% low at B = 16, its state staying in the last-level cache. Without
ATOM's warmup the PoC read GDN about 5% low, cause not found. This bench reads GDN about
5% under the bench it replaced at decode B = 2 and 16 and within 0.1% at prefill; the gap is
diagnosed on [#1902](https://github.com/jgong5/ATOM/issues/1902) and set up in the
leaf's own preparation, never by a forward.

The constraints point at the *runner*, not the server — a `ModelRunner` **is** the worker.
ATOM already shows how to drive one
without a scheduler: `model_runner.py::ModelRunner.dummy_execution` and
`model_runner.py::ModelRunner.warmup_model` both fabricate `ScheduledBatch`es by hand.

- **Every rank's runner, started as `EngineCore` starts it.** The bench starts each TP
  rank's `RecordingModelRunner` (`atom/compass/parity/runner.py`) through `EngineCore`'s
  runner manager, in `EngineCore`'s order — `get_num_blocks`, `allocate_kv_cache`,
  `capture_cudagraph` — and then calls `forward` with replayed batches, with no scheduler.
  Any TP width: every rank measures, and a step is charged to its slowest rank (`15`
  D90). DP above one, expert parallelism and PP refuse until a task builds them.
- **The runner a real run records with.** A measured step and a recorded one share one
  timing method, so a bench point and a real step compare.
- **One window for both step kinds.** A step's value, decode or prefill, is device time
  from an event at the forward's entry to one at its exit, with device work queued ahead
  so the host's launch cannot show in it. A repeat counts only if the entry event is still
  pending when the forward returns; one that is not is discarded, and more work is queued
  ahead. A forward with a host synchronisation inside never qualifies, and its point
  refuses. For decode the window covers the device work the step linear law's label covers in a
  recorded step, device gap and busy window (`09` D54); for prefill it adds, to that label's
  busy window, the device work the step's input preparation queues.
- **The value kind is stated.** A step grid bundle states that value kind beside its
  interval (`02` D10), and loading refuses another (`17` D98).
- **Replayed by rule.** A batch arrives as a step-record line (`13` D80), its rows in
  order with their query and context tokens, and is rebuilt by a fixed, versioned rule:
  row order kept, each row's KV blocks contiguous from a fresh pool in row order, filler
  tokens, and a row that keeps state the fresh state slots ATOM's block manager gives a
  request: one, or 1 + K under a GDN model's drafter of K drafts without ReplaySSM. Each
  measurement records the rule's version. Whether that reproduces the recorded step's forward context is T18.
- **Several cards, once they agree** (revised 2026-11-19 by
  [#2678](https://github.com/jgong5/ATOM/issues/2678), D118, bar 2). `measure steps --gpu`
  naming k times the TP width's cards runs k benches, each on the next TP width of them,
  over the invocation's points into one store: each bench takes the next point from one
  queue, and runs all of that point's repeats (`17` D98). A split first checks its cards:
  a fixed sample of at least ten of its points, every step kind among them, is measured
  once on every bench in one pass, and a second time, a pass later, on the first. A bench
  is named and measures nothing when the median of its differences from the first bench's
  first pass exceeds the largest difference between the first bench's two passes: its
  offset from the first card is then past the first card's whole pass-to-pass spread over
  the same points. A median is set against a largest because a bench agreeing with the
  first as well as the first agrees with itself draws its differences from the first
  bench's own distribution, so the same statistic of each exceeds the other's about half
  the time; such a bench is left out only when half its differences exceed every one of
  the first bench's, under 2% of the time at ten independent points and less with more.
  The evidence passes it: two of node 18's cards measured #1233's views 0.23% apart at
  p50 ([#2678](https://github.com/jgong5/ATOM/issues/2678)), against one card's largest
  spread of 1.1% over three passes ([#1881](https://github.com/jgong5/ATOM/pull/1881)),
  while D41 puts PTL-on MI308X cards 3 to 5% apart, which it leaves out; no pair answers
  for another node. It replaces a split of the inputs over GPUs, each run reading the store
  once, at its start (`atom/compass/bench/measure.py::_todo`), which spent 13.6 of #1233's
  27.2 device hours on views another GPU measured. The op bench stays one bench an
  invocation (T118).

```
  record, trace steps;              the step bench (GPU): EngineCore's
  step grid's planner, in measure   runner manager, every TP rank, no scheduler
  +-------------------+  step-     +--------------------------------+
  | batch: its rows   |  record    | RecordingModelRunner, per rank |
  | in order, their   |  lines     |  .forward(replayed by rule)    |--> step measurement
  | tokens, its rung  |----------->|                                |
  +-------------------+            +--------------------------------+
```

**Batches must be replayed, not synthesised.** Hand-built batches get the composition
wrong in ways that matter:

- Row order alone moves measured decode attention across **1.77x** at one fixed 32-row
  context multiset (grouped-descending 276.1 µs, grouped-ascending 292.2 µs, alternating
  376.5 µs, ladder-interleaved 489.5 µs). Hand-built ladders are grouped by construction.
  **Explicit `block_tables` are mandatory**, not optional, or the treatment is
  unreproducible; the replay rule builds them from the recorded row order.
- Raggedness is workload-dependent and a fit is blind to a dimension it never varied: every
  calibration round running one sequence length left raggedness at exactly 1.00 in **all
  2,997 samples** — a column of zeros where a feature should be.

### What each measurement may use

| Measurement | Step bench with replayed batches | Synthetic shapes |
|---|---|---|
| op price | no: the op bench | **yes** — a planned point is a batch ATOM prepares; activations and weights come from the leaf's contract, the rest from the runner |
| step measurement | **required** | no |

Op pricing on synthetic shapes is what keeps cc-traces off a GPU during calibration.

**One departure, by ruling.** Topic `17`'s step grid measures its baseline grid on synthetic
homogeneous shapes (`17` D98). The residual that carries row order and raggedness is
fitted to real-run step times, to replayed bench measurements, or to both (`17` D100,
[#882](https://github.com/jgong5/ATOM/issues/882#issuecomment-6140285611)). The rule
above holds for every other step measurement.

### The benches do not change D38's phases — they are where two of them run

A bench is a **host**, not a phase. D38's phase list is unchanged; what the benches
settle is *which process* each phase runs in, which was previously implicit.

| Phase | Command (`13` D80) | Runs in | Why there |
|---|---|---|---|
| 0 discovery | `trace steps` | a simulated serve, device-free | drives ATOM's real `Scheduler`, priced by the backend it names |
| 1a trace | `trace ops` | any CPU process, **or** lazily inside a sim run | device-free (with the ROCm caveat above) |
| 1b price ops | `measure ops` | **the op bench**: ATOM's runner on meta weights, running no forward | `prepare_model` builds each point's metadata and forward context, and the bench times each leaf from its contract; at width every rank of the layout (D40) |
| 1c in-situ | `measure steps`; `record` for real runs | **the step bench**; a real serve | needs whole real steps; the step bench replays recorded and planned batches into them |
| 2 memory | `measure memory` | **a plain engine startup**, not a bench | it measures startup, so a bench that skips parts of startup would measure the wrong thing |

So: the op bench covers 1b and the step bench 1c; Phase 2 is an ordinary startup
`measure memory` runs; Phases 0 and 1a need neither.

### The recommended flow

One canonical sequence per half of the toolchain. For a new (model, machine) pair, where
each server in steps 2, 3 and 6 serves the client runs a client started apart sends
(`13` D80):

```
 Platform: once per node of a new device, stack, PTL state or torch (05 D26)
 0.  compass measure device --gpu ... on each node; compass fit device <records>
                                                           [GPU, every card; then CPU]
       -> a probe_record per node; platform_readings for the platform

 Steps half: the step-level backends (16 S2.7)
 1.  compass measure memory -- <ATOM args>                 [GPU, one start]
       -> memory_readings for this configuration
 2.  compass record ... -- <ATOM args>                     [GPU, a server]
       -> step records, split per client run; run.json with the engine's bounds
     compass record --trace ... -- <ATOM args>             [GPU, a server, the same client]
       -> a trace run: the record's prefill passes' busy time (D38, Phase 1c)
 3.  compass trace steps <backend> --device DEV --ptl PTL ... -- <ATOM args>  [CPU, a server]
       -> shape_population, keyed by the backend that priced it
 4.  compass measure steps <records> --grid <populations> ...       [GPU, k benches]
       -> the step grid's planner over 3 (17 D98); one step_measurement per distinct batch,
          on every bench the agreement check admits (D39)
 5.  compass fit host <records>; compass fit steps --backend step-linear|step-grid ... [CPU]
       -> host_laws; a step_linear_law or a step grid bundle
 6.  compass run <backend> --device DEV --ptl PTL [--coverage <populations, records>] ...  [CPU, a server]
       -> the step grid's coverage check reports before it starts, refusing
          no step (17 D99); run.json records what grading reads
 7.  python -m atom.compass.validate; python -m atom.compass.memory.check,
     outside the CLI                                       [CPU]
       -> whether each client run is gradeable, the separable results (08 D44);
          the memory terms, one at a time

 Ops half: the op-level backend (16 Stage 3d), after the steps half's 1, 2, 3 and 5
 1.  compass trace ops --population <records> ...          [CPU, seconds per shape]
       -> op_graph per decode width and prefill (T, B, G) of the steps in the
          steps half's 3 --out, or 2's records
 2.  compass measure ops --plan <record> [--plan <record>]... ...  [GPU, every rank]
       -> the op-level planner over the steps in the steps half's 3 --out,
          or 2's records (D37); one cold
          measurement per planned point, with the kernels each leaf launched,
          collectives included (D40)
 3.  compass fit ops ...                                   [CPU]
       -> price_list
 then the steps half's 6 and 7 with the op-level backend
```

The ops half is not self-contained. `run` with the op-level backend reads the steps half's
step 1 (`memory_readings`) and step 5 (`host_laws`, which `01` D4 composes with the device
time the backend returns), and grading reads step 2's records.

Only `record` and `measure` need a card. One `record` at TP2 is the whole transfer claim:
it does not calibrate anything, it tests whether Phase 1c generalises across width (T21).
Steps 6 and 7 are what turn a pile of artifacts into a statement about what can and cannot
be predicted.

**Two things the flow is arranged to make hard to get wrong.** `trace steps` runs with the
engine config and prefix caching that the target workload will use, because a shape
population discovered without prefix caching does not contain the high-`N_KV`/low-`N_Q`
corner a cache hit produces (doc `03`), and every hit step will then fall outside the
hull. And the coverage check runs *before* a graded run, not after a prediction looks
wrong.

**Iterating.** When the coverage check or a run reports refusals, the fix is to extend the
shape population and re-run from `trace steps` — not to widen the hull. Re-measuring for
an extended population is incremental: the artifact store is keyed per entry (D41), and a
step measurement by its batch, so only new batches and signatures are measured.

### Open issues

- Whether a batch replayed by rule reproduces the recorded step's *forward context*
  faithfully, including the attention metadata built by `prepare_inputs`, is unverified.
  MoE routing follows token content, so a replayed MoE step may route otherwise than the
  recorded one.

---

## D40. Pricing communication

Comms split four ways, and only one is straightforward.

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489):** collectives are leaves like any
other, interpolated over message bytes. They are measured on a multi-rank op bench that
starts every rank of the deployed layout through ATOM's own group initialisation, takes a
barrier before each timed replay, and measures the union of all ranks' signatures (rule 1
below). The step bench, which refuses DP, is not reused.

**(a) Dispatcher-visible collectives** — `aiter::all_reduce_` and the fused
`fused_allreduce_rmsnorm_*` family, `qr_all_reduce`. Priced like any leaf, **on the
multi-rank op bench, where the process group is live**. Measured: `[4,1024]` decode
**10.38 µs**, `[1256,1024]` prefill **71.33 µs**; a TP=2 decode graph holds 58 grouped
operators, 57 of them `all_reduce_`, coverage **98.3%**. One known failure: `c10d::broadcast_` takes a
`ProcessGroup` object no JSON artifact can hold — one operator per step.

**(b) Invisible collectives, and the default path is the invisible one.**
`ATOM_USE_CUSTOM_ALL_GATHER` defaults to **1** (`atom/utils/envs.py::environment_variables`); its
registration is *commented out* at `aiter/dist/parallel_state.py:467-468` and
`all_gather_reg` / `all_gather_unreg` are in `NONE_WRAPPED_OP`. Zero dispatcher events end
to end — while the RCCL path **is** visible (`c10d::allgather_into_tensor_`). These need
doc 04 D21's **stated node** plus a dedicated standalone benchmark calling the same entry
point.

**(c) MoE all-to-all (EP)** — MORI, called inside `moe_forward`, invisible at any degree.
Its block counts, ATOM's two TBO paths and the overlap are doc 04 D19's.

**(d) p2p.** PP is `isend`/`recv` on NCCL (`atom/distributed/pp_comm.py::async_send_intermediate_tensors`, `commit_pp_send_work`),
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

### The recommended flow for comms — and what it reuses

Only class (a) reuses the ordinary op-pricing flow unchanged. The other three need
something, and the "something" differs per class. This is the answer to whether a
dedicated flow has to be built: **one new tool, shared by (b) and (c); nothing new for
(a) or (d).**

| Class | Command (`13` D80) | Reuses | New work |
|---|---|---|---|
| **(a)** dispatcher-visible | `measure ops` at each layout — the multi-rank op bench, the same graph-capture-and-replay method as any leaf | **everything**, including the CUDA-graph replay method and the artifact schema | none; it is an op with a process group |
| **(b)** invisible collectives | `measure ops` at each width, driven from the **stated-node list** rather than from the captured graph | the op bench, the timing method, the artifact schema | a **stated-entry-point benchmark**: call `all_gather_reg`/`all_gather_unreg` directly at the shapes the graphs imply |
| **(c)** MoE all-to-all | `measure ops` at each width and EP degree | the same stated-entry-point benchmark as (b) | the block count per phase: `min(128, CU)` in prefill, `min(64, CU)` in decode |
| **(d)** p2p / PP | `measure ops` at each width and PP degree | same as (a) — `isend`/`recv` are dispatcher-visible | none |
| **(d)** KV transfer | **not priced** | — | nothing: doc `01` D6 computes it from the machine spec, which is what keeps interconnect configurable |

**So the recommended flow is:** classes (a) and (d)-p2p ride `measure ops` and need no
separate step at all — they appear in the captured graph and are priced with everything
else. Classes (b) and (c) ride `measure ops` too, in the **same multi-rank op bench, in
the same processes**, differing only in where their work list comes from: a stated list of
entry points beside the captured graph. That is why it is one command and not two.

Three constraints carry over into the flow and are not optional:

1. **`measure ops` prices collectives at every deployed width**, not at TP1 with a
   scaling law — rule 3 above, where two collectives moved **+20.6%** and **−43.5%**
   across the same width range.
2. **It prices the union of all ranks' signatures** — rule 1 above, where a
   one-entry difference deadlocks silently.
3. **Its output is labelled an upper bound**, not a price — the honesty constraint above,
   where the same collective read **0.23x** between two passes because the number includes
   peer wait. Doc `02`'s method vocabulary tags these `measured (collective, upper
   bound)` so nothing downstream treats them as device work.

**The open piece:** `c10d::broadcast_` takes a `ProcessGroup` object no JSON artifact can
hold, so it cannot be replayed from a price list — one operator per step, issued in
`ModelRunner.postprocess` after the sampler. Since the owner's ruling on
[#815](https://github.com/jgong5/ATOM/issues/815) (2026-10-11) removed the Phase 1c host
floor, its device time is inside the forward's device busy time, the label the step linear law and
Phase 1c are fitted to (`09` D54), and its launch is inside the `W_launch` host law (`01`
D4). Recorded as **T55**.

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

### The artifacts

Every entry carries a **key**, a content **digest**, a validity **fingerprint**, and a
**provenance** stanza naming the executed source root. Revised 2026-10-13 on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791): `measure`
writes measurements per point and `fit` makes laws and bundles from them, each a kind of
its own, and a population is keyed by the backend that priced it. Revised 2026-10-18 on
[#1489](https://github.com/jgong5/ATOM/issues/1489): the region-term artifact is retired,
`measure ops` writes an `op_measurement` per point, `fit ops` publishes a `price_list`
only, and an `op_graph` is keyed by its shape.

| Artifact | Produced by | Keyed by |
|---|---|---|
| `probe_record` | `measure device`, one per node (`05` D26) | (device, software stack, PTL state, torch, host name) |
| `platform_readings` | `fit device`, from probe records, or `--assume-from` another device's entry for a device never run (`05` D26) | (device, software stack, PTL state, torch) |
| `shape_population` | Phase 0, `trace steps` | (model, workload, engine config, the pricing backend's digest) |
| `op_graph` | Phase 1a, `trace ops`; a run's lazy trace | (**structure**, shape, source-root digest, torch): the structure's discrete key; a decode width or a prefill (T, B), with its gathered rows G when they are not `dp x T`, or `G=G` for a DP prefill's graph that holds G as a symbol (`04` D18); the torch it was traced under; and the digest of its files, each file under ATOM's and aiter's roots its process had run or read when it traced the graph, with its own digest (D43). A graph traced under another torch, or a file of which reads otherwise, is a miss, which a run's lazy trace publishes its own beside ([ruling on #3124](https://github.com/jgong5/ATOM/pull/3124#issuecomment-6768936508), #3156) |
| `op_measurement` | Phase 1b, `measure ops` | (model, platform, source-root digest, the point's digest): of its shape, context and leaf signature, and of the parallel layout for a collective leaf alone |
| `price_list` | Phase 1b, `fit ops` | **(model, parallel layout, platform, source-root digest)** — a tuple, never a path |
| `memory_readings` | Phase 2, `measure memory`; published with every reading assumed for a device never measured; or carried by a law file, assumed (below) | (model, platform, parallel layout, engine config) |
| `step_measurement` | Phase 1c, `measure steps` | (model, platform, engine config, source-root digest, sampling, step kind, the batch's digest) |
| `step_linear_law` | `fit steps --backend step-linear` | (model, platform, engine config, source-root digest, form) |
| `host_laws` | `fit host` | (host CPU, ATOM source, model, engine config, device, software stack, torch), as the run reads its host key, but for the host CPU, which it takes from the laws: the host they timed is the host it predicts (`09` D54) |
| step grid bundle (`step_grid_bundle`) | `fit steps --backend step-grid` | `17` D101, its `device` field the platform |
| `projection` | `compass project`, real runs' steps priced on a device none of them ran on | (model, torch, the whole composed machine spec's digest, source-root digest, the cells' step records and engines) |

**Platform**, in a key, is the device, the software stack, the PTL state and torch, each by
value, as the kind's row in D43 marks them; a kind keys each it depends on and no other.
A datasheet and a network file are files, not entries: a run names them, and its composed
spec echoes them (`05` D25).

The **parallel layout** of a `price_list` and `memory_readings` key, and of a collective
leaf's `op_measurement`, is
`dp`, `pp`, `pcp`, `tp`, `enable_expert_parallel`, `enable_dp_attention` and
`decode_context_parallel_size` (owner ruling on
[#165](https://github.com/jgong5/ATOM/issues/165)), held in
`atom/compass/artifacts/keys.py::LAYOUT`. No such entry answers for another layout: a TP2
price list does not answer for `-tp 2 -dp 2`, and a transfer between layouts, if one is
wanted, is an explicit record backed by a measurement. A read miss names the fields in
which the nearest entry of its kind differs.

**An `op_measurement` keys the layout only for a collective leaf** (owner's
[Ruling on #1500](https://github.com/jgong5/ATOM/issues/1500#issuecomment-6870338230)):
a leaf that runs on a process group: one `measure ops` families by message bytes
(`exchange`), one whose arguments name its group (`group_name`), as aiter's all-reduces
and fused all-reduce norms do, or ATOM's all-reduce under two-batch overlap
(`tbo_all_reduce`), which runs on the TP group naming none
(`atom/compass/bench/ops.py::ON_GROUP`). A signature names a group by role (`tp:0`) or
not at all, never by its ranks, so the leaf's time depends on the layout beyond its
signature, and its point's digest holds the layout. Any other leaf's signature holds the shapes the layout gives it, a weight at its
shard and attention at its heads, and under DP attention `measure ops` already times such
a leaf on a runner at one rank's own width (D39). So its point is keyed by its signature
alone and answers at every layout: a new layout measures only its collectives and the leaf
signatures no layout measured. The point records the rows its run's KV pool held, which
`fit ops` reads. `fit ops` keys the price list at the one layout every line it reads
states, the layout of the `measure ops` run that listed the point; a `--plan` run lists
the points it found in the store too, so its `--out` alone names a layout's points.
`scripts/compass/migrate_keys.py` moves a store keyed by the layout before 2026-11-14,
folding a collective's layout into its point's digest: its points are re-keyed, not
re-measured.

**The platform is read from the records,** never from a spec file: an entry keys the
device, stack, PTL state and torch the records it was made from read
(`atom/compass/drive/identity.py::identity`), so an entry measured on one platform does
not answer on another, and nothing outside those four moves a key. It replaces the
machine-spec digest, a hash of one spec file's key that carried PTL into `host_laws`,
whose row marks PTL independent. The host name, the HBM vendor and the PTL format stay out
of the platform (owner ruling on
[#1434](https://github.com/jgong5/ATOM/issues/1434#issuecomment-6245078946)); only a
`probe_record` keys its host name, so one node publishes one record per platform. Torch
is the platform's fourth field: a kind whose row marks it keys it, so after a torch bump
a fit publishes under a new key, where a fit to the old one refuses (rule 4), and a fold
refuses records of two torches (`05` D26).

The **PTL state** is the platform's third field, `disabled` for a card that has none. A
record reads it at capture and at run time from
`/sys/bus/pci/devices/<bus>/ptl/ptl_enable`, with each GPU's bus as `rocm-smi
--showbus` names it (not `/sys/class/drm/card<n>`, whose numbering is not the GPU
index: DRM `card0` can be the BMC's VGA), a card with no `ptl` directory reading
`disabled`, and refuses when the cards of one run disagree, naming each
(`atom/compass/drive/identity.py::_ptl`). A fit refuses records that disagree on their PTL
state, as it does on their device and stack, and keys the one they share. A record that
carries no PTL state, captured before the field existed, refuses
at a fit unless `--ptl enabled|disabled` states it; the fit records the stated value in
its report, and refuses `--ptl` for records that carry a state of their own. PTL is the
one static field that separated hjbog-srdc-39 from node 18 once both pinned one stack,
and it accounts for the gap a synthetic decode step shows between them, about 10%.
[#828](https://github.com/jgong5/ATOM/issues/828)'s gap on real runs, 18 to 20%, is
larger and stays open. The band one key
leaves: same-key cards agree within 0.6% with PTL off; with PTL on, 3 to 5% per card
plus 1 to 2% run-to-run remains. That band is MI308X's, on a synthetic decode step.
MI355X's cards spread wider with PTL disabled: crsuse2-m2m-v3-005's eight run a
DeepSeek-V4-Pro 16384-token prefill chunk at 0.6% under to 1.1% over the crs005
prefill law across 1P1D rounds of [#975](https://github.com/jgong5/ATOM/issues/975)'s cell ([#2992](https://github.com/jgong5/ATOM/issues/2992)),
and crsuse2-m2m-v3-013's ranks run a one-row decode step 0.6 to 1.4% apart, part of
it their contexts' mix ([#3048](https://github.com/jgong5/ATOM/issues/3048)). Neither enters a key or a fit:
a law prices the card mix of its records, and `validate.cost` states each DP rank's
error with its cards' PCI buses, which `run.json` records (README, principle 12;
[#3048 Ruling](https://github.com/jgong5/ATOM/issues/3048#issuecomment-6851366020)).

**A law file's readings.** A law file under `scripts/compass/laws/` carries the
`memory_readings` members `measure memory` took on its law's platform under its engine
config, each stating `assumed`: where, and at which tree. `run` reads them when its store
holds none (`13` D80). They are keyed as a store entry is, by a key that holds no ATOM
digest (`atom/compass/artifacts/keys.py::OWN_FIELDS`), and checked as one is against
ATOM's source, their `atom_src` the `atom` digest of the law's host key, which every kind
reads (rule 2 below), so the commits that re-key the law to a new digest re-key them, and
no commit outside ATOM's own Python moves either: one engine start serves every tree the
law reaches, a reuse the README's principle 10 allows where it is stated, here as
`assumed`, which makes their terms declared. A commit that moves what a start under the
law's engine config holds, through the code a reading reads (Phase 2), takes them again on
that platform. The measured cases are DeepSeek-V4.1-Flash's graph pool at TP4 EP4 DSpark5
on MI355X, its KV block count unmoved.
[#2861](https://github.com/jgong5/ATOM/pull/2861) raised ATOM's default prefill token
buckets from 3072 to 7168 tokens, taking a start at the default from 2.94 to 7.78 GB per
GPU; the key reads the buckets only off their default
(`atom/compass/run/step_grid.py::keyed`), so that commit stated the old ones in the
V4.1-Flash laws' engine config, as their readings' key would, and a start under it holds
what it did. [#2873](https://github.com/jgong5/ATOM/pull/2873)'s recordings, keeping no
argument their regions take live, took a start at those buckets from 2.94 to 2.08 GB: such
a commit takes the readings again. The first law to carry them is the Qwen3.8-27B TP1
MI308X law, its readings taken once on one card of hjbog-srdc-39, whose cards run with PTL
disabled as the law's records did, under the stack the law is keyed to.

### Four rules

1. **A key is a tuple, never a path.** *"A bare or merged price file states no width and
   silently prices nothing."*
2. **The source root is a `git archive` digest, not an rsync.** And the audit must digest
   **every** module that can answer, not just the one named — a hard-coded `sys.path` once
   let a stale regions module answer under a current registry's name. A dirty tree may run
   but not publish (owner [ruling on #168](https://github.com/jgong5/ATOM/issues/168#issuecomment-6399062318)): a checkout's root is keyed by its
   HEAD's tree, so an entry published from uncommitted files would answer for code that
   never ran. A publish refuses a root that counts uncommitted files, and a fit refuses a
   record whose `run.json` `source` counts any; a snapshot's stamp states no count, since
   `snapshot.sh` refuses a dirty tree, and publishes. An aiter bump misses through the
   software stack in the key, which states aiter's commit.

   **ATOM's source, as an artifact reads it, is the digest host laws use, not the git
   tree** (owner [ruling on #2310](https://github.com/jgong5/ATOM/issues/2310#issuecomment-6497009949),
   2026-10-29). Every row's ATOM src cell (D43) and every key's `source_root` field read
   `atom/compass/run/__init__.py::atom_digest`: the sha256 of every `atom/**/*.py` file,
   less the Compass code `COMPASS_ONLY` names and the `atom/_version.py` that
   `pip install -e .` writes, the `atom` value of the host key. It is a
   reading kind of its own, `atom_py`, and a record's value is its `run.json`
   `host_key.atom`. A commit that edits no Python file under `atom/` outside Compass's
   code, a doc or a test, moves no reading and no key; in the week to `671e2ac9d`, 250 integration commits moved
   the tree and 8 moved this digest. Compass's measuring code stays outside it, as it has
   for host laws since [#935](https://github.com/jgong5/ATOM/issues/935): a change to
   what a measurement yields republishes its entries by hand. Provenance keeps the tree,
   so the uncommitted-files refusal above still holds. An entry published under a tree or
   commit reading refuses once, as not comparable, and is republished; checked-in laws
   carry only `host.key.atom`, which keeps its value.
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
| `measure device`, `fit device` | once per node, then once per (device, stack, PTL, torch) | `measure device` yes, every card of the node; `fit device` no |
| Phase 0 discovery | every run | **no** |
| Phase 1a trace | **JIT on first use of a shape**, then cached | **no** |
| Phase 1b op pricing | offline campaign, or opt-in `--measure` | yes |
| Phase 1c in-situ | offline campaign | yes |
| Phase 2 memory readings | offline, `measure memory`, one start per configuration | yes |

This follows the project's own note: the runner has no modes, every run simulates, trace
happens on demand at first use and is cached, and `--measure` is a separate flag that does
a real GPU run and caches the result.

**A new shape is priced between same-kernel neighbours; anywhere else, and for a new
operator kind, the leaf falls to its analytical law.** Revised 2026-10-18 by the owner's
ruling on [#1489](https://github.com/jgong5/ATOM/issues/1489) and its
[band ruling](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252). A shape
never measured is captured (Phase 1a) and each leaf in it priced by interpolating the
leaf's measured points when its two neighbours launched the same kernel (`09` D60). A
query outside the measured range or between two kernels, or an operator *kind* with no
measurement, is answered by its analytical law (D36), stating that source, and refuses
only without one; the analytical share of priced seconds is counted and reported (`08`
D50.3). Under the prior flat design a new shape meant a new graph meant a refusal, which
is why a fixed-shape oracle cost **+47.9% TTFT** the moment serving varied shapes (3
prefill steps of 16 / 256 / 16,128 tokens costing 38.6 / 151.0 / 177.6 ms, all answered as
~47 ms).

### Two refusals that belong with `--measure`

1. **A `--measure` run can never be an acceptance run.** The instrument changes what it
   measures: measure mode cost ~**11 ms of TTFT on the 27B (4%)**, and a variant that
   synchronised around each forward made the run **33% slower** (TPOT 3.26 -> 4.33 ms;
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

Every GPU command holds both as refusals (`13` D80): it names its GPU and records whether
it profiled and which machine it ran on, and `fit` refuses profiled inputs and inputs from
two machines.

---

## D43. Invalidation

A global fingerprint would force re-measuring everything on a torch bump. Each artifact
records the fingerprint of **its own dependency row**; on load, recompute and compare;
mismatch **refuses** (warn only under an explicit flag).

| artifact \ invalidated by | ROCm / AITER / RCCL | torch | ATOM src | model | device | PTL | host CPU | engine cfg |
|---|---|---|---|---|---|---|---|---|
| `shape_population` | - | - | **X** | **X** | - | - | - | **X** |
| `op_graph` | - *(aiter's Python and data: in its files)* | **X** | **X** *(the files its trace ran and read)* | **X** | - | - | - | X (a step law's less its ladder and level, and the cudagraph mode) |
| `op_measurement`, `price_list` | **X** | **X** | X | - *(shape-parametric)* | **X** | X | - | - |
| `step_measurement`, `step_linear_law` | X | X | X | X | X | X | - | X |
| step grid bundle (`17` D101) | X | X | X | X | X | X | - | X |
| `host_laws` | X | X | **X** | **X** | X | - | **X** | **X** |
| `projection` | - *(analytical)* | X *(op_graph capture)* | X | X | - *(the spec is in its key)* | - | - | X |
| `memory_readings` | **X** | X | X | X | **X** | X | - | **X** |
| `probe_record`, `platform_readings` | **X** | X *(Triton, `torch._scaled_mm`)* | - | - | **X** | **X** | - | - |

Consequences worth noting:

- `price_list` does **not** depend on the model once leaves are shape-parametric. That is
  what lets one campaign serve many shapes.
- A `shape_population`'s pricing backend is in its key (D41), not its row: another
  backend addresses another entry rather than invalidating this one.
- PTL invalidates what the device does, except the capacity: it moves the clock, not the
  memory. `memory_readings` still refuse under another PTL state, because their row marks
  it and so their key states it; a parity run re-measures them.
- The `machine_spec` rows gave way on 2026-11-03 (#2716): its capacity, which only the
  device moves, is a datasheet's, a file no fingerprint guards; its tokenizer terms are
  the host laws' (#2309). A `probe_record` and the `platform_readings` folded from it time
  vendor kernels on the cards, so the stack, the device and PTL invalidate them, and torch
  through the Triton and `torch._scaled_mm` kernels the probes launch; no model, engine
  config or ATOM engine code runs in a probe, and a record's provenance names the tree it
  ran from.
- `host_laws` time the host, so the host CPU invalidates them. The device and the
  software stack invalidate them too, through the device and stack in their key:
  `W_launch` is time spent launching into the HIP runtime, and `W_post` and `L_mtp` wait
  on a device-to-host copy. PTL does not: it moves the device clock, not host time, so
  their key does not carry it. Torch invalidates them, and their key carries it:
  `W_pre`, `W_launch` and `W_post` run through torch. A run checks the torch of their
  host key as it checks the rest of it ([#2717](https://github.com/jgong5/ATOM/issues/2717)). Their tokenizer and
  HTTP intake terms time the host too, so its CPU invalidates them, where the spec's
  tokenizer row left the host CPU out on T53's assumption, now retired (D38, Phase 1c).
  `step_measurement` and `step_linear_law` hold device time, so the host CPU does not
  invalidate them.
- The memory readings depend on the **library build** as much as the silicon, which is
  why ROCm, AITER and RCCL invalidate them. The +5980 MiB appearing the moment width
  exceeds one is collective buffer sizing; the 926 MiB at TP1 is HIP context plus
  libraries.
- The ATOM src column reads the host key's `atom` digest, not the git tree (D41 rule 2,
  2026-10-29 on [#2310](https://github.com/jgong5/ATOM/issues/2310#issuecomment-6497009949)):
  a doc, a test or Compass's own code moves no cell, and any other Python file under
  `atom/` does.
- An `op_graph`'s ATOM src cell reads **its files**, not the whole source (#3156): each
  file under ATOM's and aiter's roots its trace's process had imported, whose body ran,
  or opened to read by the time it traced the graph, each with its own digest, ATOM's
  read as the host key reads it. A lookup reads a graph while each of them reads as it
  did, checked once a process, so an edit to a file no trace ran, another model or the
  server, misses no graph. A module imported counts whether or not a function of it ran:
  its body defines what code elsewhere reads, an aiter op's schema among them. What the
  files cannot hold stays keyed by version: torch's C++ meta and fake registrations, by
  the torch in the key. aiter registers no op in C++, and no compiled aiter code runs in
  a capture: a V4-Pro prefill capture made no call into a compiled aiter module, and
  about 90,000 into torch's (#3156). So the software stack stays `-`. A graph keyed
  by the whole ATOM source still answers while that source is this one's.
- ATOM's source invalidates a store entry's readings, which `measure memory` takes again; a
  law file's are re-keyed or taken again as D41's *A law file's readings* says.
- The region-term artifact's row is retired with its kind, 2026-10-18 on
  [#1489](https://github.com/jgong5/ATOM/issues/1489). An entry that still records a
  fingerprint for it refuses at load, as any row the matrix no longer names does.
  `op_measurement` shares `price_list`'s row, since a price list is fitted from them.

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

## D106. Warming a container: a constructed JIT sweep and a per-stack cache library

Decided 2026-11-02 by the
[design session on #2640](https://github.com/jgong5/ATOM/issues/2640#issuecomment-6568381628).
Revised the same day by the owner's
[Ruling on #2650](https://github.com/jgong5/ATOM/issues/2650#issuecomment-6579531822)
and [Ruling on #2687](https://github.com/jgong5/ATOM/issues/2687#issuecomment-6591321600),
on [#2687's measurements](https://github.com/jgong5/ATOM/issues/2687#issuecomment-6581166120):
the ranks share one walk of the token counts, ATOM's PrefillDelayer coalesces each
designed step, and no point needs a blocker. Revised 2026-11-03 by the
[Ruling on #2650](https://github.com/jgong5/ATOM/issues/2650#issuecomment-6607750609),
after that sweep's acceptance run left one chunked step unaligned and two served records
built 13 keys on swept axes: chunked rows take the full product of query size, its residue
mod 16 and context level up to the longest prompt, a step's chunked rows need as many
earlier chunks each, and TBO points split inside a long first row too. Revised
2026-11-04 by the
[Ruling on #2650](https://github.com/jgong5/ATOM/issues/2650#issuecomment-6619270702) and
2026-11-06 by the
[Ruling on #2689](https://github.com/jgong5/ATOM/issues/2689#issuecomment-6686630693):
TBO split steps cut other ranks' rows into halves of 256 tokens and on, and chunked query
sizes from the 256-token alignment up take each multiple of 16 below it, so a row's KV
length varies off 64 at a query on 16.

### Problem

A real record whose `run.json` names `aiter_jit_builds` was refused whole (`08` D52,
[#1704's ruling](https://github.com/jgong5/ATOM/issues/1704#issuecomment-6405004644)), so
a graded run had to compile nothing its container could have compiled before it. Since
[#3739](https://github.com/jgong5/ATOM/issues/3739) the fits and `validate.cost` cut the
steps that overlap a timed compile instead, up to a bound, while grading still refuses a
run that compiled, since the stall delays every later request (`08` D52). Each JIT
compiler in play (aiter's JIT modules, `compile_hsaco`, FlyDSL, Triton and Gluon,
Inductor) caches by its own key, and a key mixes code identity with a runtime
specialization: shapes, constexprs, argument divisibility by 16, pointer alignment,
`None` arguments, dtypes, a tuned config a token count selects.

A container was warmed by serving the workload until no record built
(`scripts/compass/README.md`). Serving reaches the key space by sampling: the tail is
reached rarely and never certainly, and under DP attention and TBO no client controls a
rank's batch. DeepSeek-V4-Pro's prefill still built new keys at its eighth load
([#2493](https://github.com/jgong5/ATOM/issues/2493),
[#2634's reading](https://github.com/jgong5/ATOM/issues/2634#issuecomment-6560767690)):
the FlyDSL A16W16 GEMM, keyed by the tuned config a forward's token count selects, whose
uncompiled configs sit in narrow token-count windows, the Gluon `fp8_mqa_logits` and
`_hash_topk`. A per-kernel list is no answer: models, kernels and images change faster
than one.

The first driver walked every count on every rank and queued each step's rows behind a
blocker prefill. Its first round on #975's cell
([#2650](https://github.com/jgong5/ATOM/issues/2650#issuecomment-6577201027)) projected
31 h a round, 78% of the prompt tokens blockers, and covered 292 of about 2,230 points
sent: rank 0 covered 231 of 278, ranks 1 to 7 at most 38 each. The delayer's alignment
gate held their prefills while rank 0 had none, so their rows joined the held requests'
step.

### Decision

**Construct the inputs that select keys instead of sampling them, and keep what is
compiled per stack, not per container.** The goal is to reach the JIT key space faster
than serving samples it. A compile in a graded run still voids that run; the sweep
changes when keys are compiled, not what a graded run may contain.

**The driver.** A constructed client against the real server, and ATOM gains no hook
point. The deployment is the cell's own: a changed deployment compiles another
configuration's keys. Only pacing variables that select no kernel or shape may differ:
they decide when a prefill fires and how much of a row a step admits, so every step they
allow is one a graded run of the cell reaches. `compass record --sweep` starts the server
with them, and `client.sh --sweep` checks the server has them before it sends anything.
They are:

| Variable | Value | Why |
|---|---|---|
| `ATOM_ENABLE_PREFILL_DELAYER` | `1` | the delayer coalesces a designed step (below) |
| `ATOM_PREFILL_DELAYER_COST_GATE` | `0` | the cost gate releases an unaligned step when it judges the wait cannot pay off; off, only the stall releases one |
| `ATOM_PREFILL_DELAYER_TARGET_FILL` | `1.0` | the fill goal then fires only when the prefilling ranks hold whole budgets, not on a step's first rows before its last arrives |
| `ATOM_PREFILL_DELAYER_STALL_TICKS` | `16` | a step's rows arrived within 4 ticks in all of #2687's runs (below); a step fired early is a missing point, resent |
| `ATOM_PREFILL_DECODE_INTERVAL` | `1`, at DP 1 and PP 1 only | at DP 1 `EngineCore.__init__` builds the delayer only when the interval is above 0 |
| `ATOM_DP_PREFILL_BALANCE_MIN_TOKENS` | the cell's batched-token budget | DP prefill cost balancing would cut the designed rows (below) |

**The balancing floor.** Under DP attention at more than one rank, the scheduler's DP
prefill cost balancing (`Scheduler._balance_prefill_step`) sets each prefill step a cost
target of min(costliest rank, 1.1 times the mean) and cuts a rank above it to the target,
never below the floor of 4096 tokens (`Scheduler._prefill_room`). On #2687's probe it cut
178 of 256 designed points, and a resend sends the same lengths, so the target cuts them
again. With the floor at the budget, `_prefill_room` grants every rank its whole chunk.
The floor decides only how much of a row a step admits, as the delayer's variables decide
when it fires: every step it allows is one balanced ranks reach in a graded run, and the
cost exchange stays on, so the delayer's reduce keeps its shape. The floor reaches
`_prefill_room` only: admission still stops taking new rows once a rank's step cost
reaches the target (the `_tick_cost < _tick_cost_limit` guard in `Scheduler`'s prefill
admission), so a designed step puts its costly row last, or gives every rank a first row
of about one cost, which leaves each rank under the target after it. Balancing's
truncation of a fractional cost ([#2695](https://github.com/jgong5/ATOM/issues/2695)) is
ATOM's to fix; the sweep does not need it.

**Two segments.** The ranks share one walk: each token count runs once per container,
its rank chosen by the count. The ranks write one set of cache roots, so one compile
serves all of them, and a per-rank walk compiles nothing more.

- **A: every prefill token count** from 1 to the batched-token budget, as
  `ModelRunner.warmup_max_tokens` sizes it, one single-row request per count and no
  parked decode, one count to each rank a round. A tuned config is selected by the token
  count, and only a walk over every count reaches every config with certainty. Count t is
  covered by any rank's prefill step of t tokens that TBO did not split, since a split
  step runs its halves, not t. A round TBO would split leaves one rank without a row,
  since a rank with no prefill vetoes the split
  ([Ruling on #2689](https://github.com/jgong5/ATOM/issues/2689#issuecomment-6595541171)):
  on #975's cell a round past `ATOM_TBO_PREFILL_MIN_TOKENS` carries seven counts, not
  eight. With no decode running the delayer fires for whichever
  ranks have arrived and the rest follow a few steps later (`FIRE (nodecode)`): 1 to 3
  prefill groups a round on #2687's probe.
- **B: designed compositions**, because a token count does not select the alignment and
  divisibility keys. Lockstep steps, one designed point per rank, over the full product
  of:
  - the second row's offset mod 16, from a first row of 1 to 16 tokens, for the
    divisibility and pointer-alignment specializations;
  - 1 to 16 held requests a rank, parked decodes the step's rows queue behind. State
    slots come off a min-heap (`atom/model_engine/state_pool.py::StateSlotPool`), so the
    requests held ahead of a probe decide which slot it takes, and the parked decodes keep
    the delayer's alignment gate armed (below);
  - TBO on or off where the engine enables TBO: on, one rank's rows reach
    `ATOM_TBO_PREFILL_MIN_TOKENS` and every rank's step splits. Under ATOM's default
    `ATOM_TBO_PREFILL_TOKEN_SPLIT=1` a rank's prefill is cut at its token midpoint,
    inside a row as readily as between two, so one row of two tokens or more splits too;
    a rank with no prefill vetoes the split (`atom/utils/tbo/ubatching.py`). The plan
    predicts every step's split, a chunked row's included, by that rule
    (`atom/compass/validate/bound.py::_split`). With the short row first the cut falls
    inside the second row, so the second micro-batch starts on the second sequence. TBO
    on therefore takes two orders: the short row first, and swapped, every rank's
    first row past the minimum, so the cut falls inside it at every offset mod 16 and the
    second micro-batch starts on the first sequence's state slot: an aligned slot with
    unaligned positions is the specialization #2650's served records built. Every rank's
    first row then costs about the same, which admits each second row (above). A third
    set of split steps gives one rank a row past the minimum and every other rank one
    row of 512 tokens and on, cut into halves of 256 and on, the second at every offset
    mod 16: a micro-batch of 256 to 511 tokens at an offset off 16 is the
    `_csa_translate_pack_kernel` specialization a TBO split built in #2650's served
    records, which no other point gives;

  plus chunked rows, for the keys a context or KV length selects, over the full product
  of:
  - query size 1, 2, 4 and so on up to the budget;
  - its residue mod 16: each size and the 15 past it, or the 16 up to the budget;
  - from the alignment (below) up, each size plus each multiple of 16 below the
    alignment. Earlier chunks and cached prefixes sit on the alignment, so a row's KV
    length moves only with its query; these vary the KV length's residue mod the
    alignment at a query on 16. The indexer's key count is a quarter of the KV length,
    and a count off 16 at a query on 16 is the `_gluon_fp8_mqa_logits_kernel`
    specialization #2650's c=64 record built, which no other point gives;
  - context level: the cuts below the budget that balancing makes in a graded run,
    aligned as the scheduler aligns a cut chunk (`Scheduler._chunk_alignment`, 256
    tokens on #975's cell), doubling from the alignment; the budget doubling; and the top,
    the last aligned level a whole budget fits behind in the workload's longest prompt.
    With the floor the sweep no longer makes the cut points itself, and an aligned level
    keeps the KV length's residue the query's.

  A step's chunked rows each need the same number of earlier chunks, so their last
  chunks run in one step: a row whose prompt needs more runs its earlier chunks beside
  the others' last, a step TBO splits. Where prefix caching is on, a shared prefix and a
  second turn start a row at a cached offset too, over the same product. A B point is
  covered by its exact rows and its split.

Every request asks for two output tokens: the second is its one decode step, so decode is
reached and the request leaves a step later.

**Coalescing, with no blocker.** The parked decodes keep the delayer's alignment gate
armed: it holds while any rank lacks prefill, and when pending tokens stop growing for the
stall it releases the step's rows together on every rank. On #2687's probe of #975's cell,
with the variables above at stall 80, every designed step fired once on all eight ranks,
80 ticks after its last row's admission, and 32 of 32 steps and 256 of 256 points ran
right first time, TBO steps split. The stall comes from #2687's tick: a decode step with
1 to 16 parked decodes a rank takes 23.8 ms median (p10 20.3, p90 36.6); a step's rows
were all admitted, across the eight ranks, within 96 ms of the first send (median 43 ms),
4 ticks; and the stall counter restarts whenever pending tokens grow, so it has to cover
the largest gap between arrivals. 16 ticks is four times that spread, 0.38 s. At stall
16, #2650's first acceptance round covered 17,038 of 17,046 points right first time; six
of the rest were one chunked step whose rows needed four and eight earlier chunks.

**What a sweep costs** on #975's cell (DP 8, budget 16,384, longest prompt 255,744 of
traces 0 to 255), projected from that round's step times
([#2650](https://github.com/jgong5/ATOM/issues/2650#issuecomment-6607095908)):

| Segment | Points | Steps | Step time | Total |
|---|---|---|---|---|
| server start | | | | 125 s |
| A | 16,384 | 2,195 rounds | 1.70 s | 3,732 s |
| B, two rows | 768 | 96, 64 split | 3.97 s unsplit, 5.10 s split (medians) | 453 s |
| B, TBO halves | 24 | 3 split | 5.10 s | 15 s |
| B, chunked | 3,091 | 390: 246 on one earlier chunk, 36 each on 2, 4, 8 and 15 | 8.31 s on one, 2.1 s more for each further chunk | 5,131 s |
| first round | 20,267 | | | 9,456 s, 2.63 h |

The 2.1 s is a 16,384-token forward on a 32k to 131k context, flat across that range;
past 131k it is unmeasured. Rounds 2 and 3 resent their few points in about 3 min each,
so the three rounds project to about 2.7 h against acceptance's 3 h.

**Cells the delayer does not coalesce** run A only and are not marked swept: ATOM builds
no delayer at PP above 1 (`EngineCore.__init__`) or under rapidserve
(`EngineCore._init_prefill_delayer`), so nothing makes a designed step's rows one batch
there (T105).

**A PD cell's decode side is outside the sweep.** The sweep covers the prefill side, whose
steps the client designs. The decode side and the KV connector path are warmed by one
unscored short run of the cell and by the library, which that run's stop writes back to as
any real server's does.

**The axes come from the cell.** Each axis and its range is read from the cell's own
engine (its budget, DP, TBO, prefix caching and drafter state) and workload (its longest
context): a point outside them compiles a key its graded run never meets. Sampling
arguments are copied from the workload, never varied, since they select the sampling path
the graded run takes.

**A cache library per stack.** Keyed by what a container can read of its stack: ROCm,
aiter's commit, RCCL, torch, Triton, FlyDSL and the gfx arch, because an artifact is
reusable only on the stack that compiled it, and each compiler's own key separates the
rest. It lives at `<store>/jit`, under the artifact store's root (D41), because it must
outlive the container whose caches it collects; `compass record` takes an optional
`--store`, and a record given none behaves as before. A real server's start merges the
library into the container's cache roots, the ones `atom/compass/drive/aiter_jit.py`
reads, never overwriting a file the container has. Its stop writes back only finished
artifacts, judged per compiler, each staged and renamed into place: a compiler writes one
key as several files (an aiter module and its `.so`, a Triton directory and its `__grp__`
json, a FlyDSL pkl and its lock), so a build the stop cut short would reach every later
container as finished, and two servers stopping at once must leave the union. Each key is
then compiled once per stack, not once per container.

**Closed by coverage.** `client.sh --sweep` drives the sweep against a running server.
Coverage is read from the step record after the server stops, a point covered when a line
states its batch as its segment requires (above): the scheduler, not the client, decides
what each rank ran, and the record states it. The coverage is kept in the library, per
stack and cell, since a cell swept on one stack is unswept on the next. A later sweep of
the cell sends only the points still missing, so the rounds converge instead of
repeating. It resends them unchanged, since a point missed by an early fire needs only to
be sent again and there is no blocker to lengthen, and three rounds at most, the bound
acceptance sets. A cell is marked swept only when every point is covered.

**Not swept: preemption and recompute, and error paths**, which the client cannot
construct without changing the deployment, and a PD cell's decode side and KV connector
path (above). A build there is a residual, recorded with the batch that compiled it, not
a defect of the sweep; the record naming it is still refused (`08` D52) and taken again
(T104).

**Deferred: streaming the real record** to the client, so that one round reads its own
coverage and closes its own loop before the stop (T103). Until then a round's missing
points wait for the next round.

**The estimate.** Read against #2634's keys, A's token-count walk alone covers about two
thirds of them with certainty, the GEMM's included; B's designed compositions cover the
alignment keys.

**Acceptance**, on #975's DeepSeek-V4-Pro cell
([#2650](https://github.com/jgong5/ATOM/issues/2650)), all of: a fresh container on an
empty library sweeps the cell to swept within three rounds, compiling every key #2634
names; two real records of #975's workload on its prefill side after it name no
`aiter_jit_builds`, and a build that does appear is traced to the batch that compiled it
and classed: in a path left unswept above it is recorded, on a swept axis it fails the
acceptance; a second fresh container on the same stack returns from `--sweep` at once,
and its record names no build; and the first criterion's rounds take 3 h or less in all,
server starts included, the time #975's warm pass took. The fallback the Ruling on #2650
set, a short burst plus the library if under 90% of B steps ran right first time or any
were left after one resend, is not taken: with the floor every step ran right first time.

**What stays.** `06` D27's *Warmup is not a protocol feature*: sweep traffic is
ordinary requests to a server whose record nothing grades. Detection stays as well: the
sweep stops warmth recurring on the axes it covers, and `08` D52's refusal and T54 catch
what it leaves.

### Open issues

- T103: streaming the real record, deferred above.
- T104: the unswept paths, preemption and recompute, error paths, and a PD cell's decode
  side and KV connector path.
- T105: cells the delayer does not coalesce, PP above 1 and rapidserve, run A only and
  are not marked swept.

---

## D118. Calibration in less time at the same accuracy: three bars

Decided 2026-11-19 by [#2678](https://github.com/jgong5/ATOM/issues/2678). It changes D37,
D38's Phase 1b, D39, `13` D80 and `17` D98, each in place, naming there the bar the change
meets.

### Problem

Calibration time goes to views measured twice, to 45 repeats a point where fewer settle, to
points on decode rungs no step reaches, and to cards left idle: on stage 2's step grid,
13.6 of 27.2 device hours were repeats of views another GPU had measured, and an op plan at
TP 1 takes 2.5 to 4.6 hours on one of eight cards
([#2678](https://github.com/jgong5/ATOM/issues/2678)). Each change below cites its own
figure.

### Decision

**Three bars.** A change that cuts calibration time meets one of them, which the decision
making the change names, and the named result of the code task building it shows it:

1. **Same bytes.** Every artifact it publishes is byte-identical, or every point's record
   is unchanged.
2. **Within the measurement's own repeatability.** A changed value stays within the same
   card's pass-to-pass spread, shown on the same points under the old way and the new. The
   spread is measured there, on the cards and points the change moves, never carried from
   another platform (README principle 10): the one on record, 1.1% over three passes on one
   MI308X ([#1881](https://github.com/jgong5/ATOM/pull/1881)), is that card's, and no MI355X
   step bench has one. It is shown as a median against a largest: the median of the new
   values' differences from the old within the largest difference between two passes of
   the old, since one statistic of two samples from one distribution exceeds the other's
   about half the time (D39, `17` D98).
3. **No lost coverage.** No step of the workloads a plan was made from loses its measured
   price, and a step outside them is counted by its backend's coverage check before a
   graded run, so it is measured, not priced by a fallback.

**An end-to-end delta tests no cut.** Graded against the same real runs, the prediction
loop's runs on #1233 under bundles v2 and v3 (D37) differed in cost on totals by at most
0.41 points, while single metrics moved a few points either way as the schedule diverged
from step 110, c4's TTFT p50 error from 4.5% to 9.1% (README finding 1).

**Not proposed, as no bar holds:** pricing a prefill chunk with output and one without from
one point, which `17` D98 measures apart since ATOM skips logits and sampling on a chunk
that samples nothing (`02` D10); and real steps standing in for grid points, which `17`
D100 keeps out, a real step's time carrying the row order and raggedness the grid cannot
see.

**The cuts**, each decided where it applies, with its strongest rejected alternative:

| Cut | Where | Bar | Rejected |
|---|---|---|---|
| `measure steps` on several cards, each admitted by an agreement check | D39, `13` D80 | 2 | `--part K/N`, N invocations into one store: a new flag and a split made by hand |
| a step point's repeats: two bursts of five a pass apart, kept when they agree, else 45 | `17` D98 | 2 | repeats until the kept windows sum to 10 s: 10 s is chosen, not measured, and no point checks itself |
| `measure ops --plan` at the decode rungs its steps replay and the next one up | Phase 1b | 3 | the replayed rungs alone: no margin, where Phase 0 states one, so a workload one rung wider waits on a top-up before it is graded; the rung above costs about 66 points on #1233's plan |
| an op-level run graded only after its coverage check | `13` D80 | 3 | the analytical share counted (`08` D50.3): a whole rung priced analytically, uncounted before the run |
| a step grid prediction planned from its own simulated runs | D37, Phase 0, `17` D98 | 3 | real runs dropped as a planning input everywhere: cost grading then declines the steps at the 52 points only #1233's real runs reach |
| byte-identical speed-ups of `measure ops`, `trace ops` and the fits (Summary) | none | 1 | none |

`measure ops` keeps one bench an invocation: `--gpu` names the layout's ranks, or one card
at a DP-attention layout (`atom/compass/bench/ops.py::main`), and the ranks take a barrier
before each replay (`atom/compass/bench/ranks.py`), so eight cards at DPA8 are one bench,
not eight. A multi-rank plan keeps each rank's cold time beside rank 0's record
(`atom/compass/bench/ranks.py::merge`), but no one has set an op point's times on two cards
against one card's two passes (T118).

### Scenarios

A step grid calibration on a node's eight cards, TP 1:

```
 compass measure steps --grid POP... REAL... --gpu 0,1,...,7 ... -- <ATOM args>
   the points the store lacks (_todo) -> one queue
   check: ten or more points, every step kind, on each bench in one pass,
          and on bench 0 a pass later
     bench k's median difference from bench 0's first pass
       <= the largest between bench 0's two passes?  yes: k joins
                                                     no: k named, idle
   pass 1: each point's first burst of 5, on the bench that takes it
   pass 2: its second burst, on the same bench
     medians within 0.23% -> its value, the median of its 10 repeats
     otherwise            -> on to 45 on that bench, the median of 45
   -> one step_measurement per point, protocol 2, in one store
```

An op plan and the run graded on it:

```
 compass measure ops --plan PLAN_RECORDS ...
   decode rungs: those PLAN_RECORDS' steps replay, and the ladder's next above them
   -> op_measurements; compass fit ops -> price_list
 compass run --op-graph --coverage GRADED_RECORDS ...
   steps at a rung no point holds -> counted per leaf kind before the run starts
     -> measure ops --plan PLAN_RECORDS GRADED_RECORDS: the points the store lacks; refit
 python -m atom.compass.validate: an op-level run with no coverage check is not graded
```

A step grid prediction of a new workload W on a calibrated configuration, no real run of W:

```
 round k: compass trace steps <bundle k> ...; W's client
            -> W's simulated steps: a shape_population and a record
          measure steps --grid over them
            new points -> measured; fit steps -> bundle k+1; next round
            none       -> stop
 the prediction: the last round's run; its coverage check sends no step to the roofline
   but past a limit, counted apart
 #1233, c1 to c64: v2 from the populations' 421 and 85 points; the loop: 17 under v2,
   0 under v3; steps on the roofline 0/2/3/0 under v2, none under v3
```

### Simpler alternative

The same cuts with no agreement check: a split measures on whatever cards `--gpu` names.
It fails bar 2 wherever two cards differ by more than one card's passes, as D41 states
PTL-on MI308X cards do, and README principle 6. Simpler still, the repeats and the op
plan's rungs alone keep #1233's 13.6 device hours of repeated views and leave a prediction
with no rule for what it measures, both of which #2678 asks for.

### Low confidence

None.

### Dependencies

No spike and no sync issue.

### Summary

**Deliverable.** The three bars and the cuts, each written where it applies. **Not in
scope:** `19` D113's record list, and its designed-load candidate (D119), which the
[Ruling on #3668](https://github.com/jgong5/ATOM/issues/3668#issuecomment-7008540288)
grades end to end against real runs, not by bar 2: no record-list law is its bar, and `08`
D47's figures on the same runs assign each miss; D106's sweep
([#3738](https://github.com/jgong5/ATOM/issues/3738)); `measure ops` on several cards
(T118); a prediction loop at the op level (T119); rank-local op measurement above TP 1 and
a parallel `trace ops`
([#1849](https://github.com/jgong5/ATOM/issues/1849)), with no evidence yet. The ATOM
source key's churn ended with [#2517](https://github.com/jgong5/ATOM/pull/2517) (D41 rule
2).

| Task | Where | Bar, and the named result |
|---|---|---|
| 1 | `atom/compass/bench/measure.py`, `atom/compass/bench/engine.py`: k benches from one queue, the agreement check; `scripts/compass/README.md` step 7 | 2: the check's reading, and a plan's hours on k cards against one [GPU] |
| 2 | `atom/compass/bench/window.py`: the bursts, `PROTOCOL` 2, under [#3287](https://github.com/jgong5/ATOM/issues/3287), whose brief takes the rule | 2: before the rule ships, about 40 prefill points on one card of the platform the task runs on (#3287's: MI355X) under both protocols, the new values within the old protocol's two-pass spread there, as `17` D98 states the test [GPU] |
| 3 | `atom/compass/bench/ops.py::plan`; `atom/compass/validate/runs.py::_not_graded` | 3: a plan over a record holds no rung past its widest and the next; an op-level run with no coverage check is refused; CPU tests |
| 4 | `scripts/compass/README.md`: the step grid's prediction loop, its stop and its gate | 3: the recipe, no code |
| 5 | `measure ops`: no copies or graphs for a leaf that launches no kernel, which times 0 ([#1822](https://github.com/jgong5/ATOM/issues/1822)); `trace ops`: shapes its store holds skipped, not refused; `fit ops`: its tokens resolved from one listing of the store; `fit host`: its JIT cut built once, not per sample; `fit steps --backend step-grid`: nodes found by bisection, each target priced once; every fit: each record parsed once, BLAS on one thread | 1: the same bytes on #2317's cal2 records or the store measured |

**Acceptance.** Each task shows its bar as its row states; T115 to T119 hold what is open.

---

## D119. A step linear law from designed load on the real server, a candidate

Decided 2026-11-18 by [#3668](https://github.com/jgong5/ATOM/issues/3668): a candidate
beside [`19` D113](19_adoption.md#d113-calibrating-what-the-search-lacks)'s record list,
which `compass advice` keeps printing until the validation below says otherwise (T120).
Widened the same day by the owner's
[Ruling on #3668](https://github.com/jgong5/ATOM/issues/3668#issuecomment-6974348636):
it covers every feature the law fits in each scenario the law supports, prefill and
decode, target and draft, each parallel layout, and settles the levels' cap. Its
validation was set on 2026-11-19 by the owner's
[Ruling on #3668](https://github.com/jgong5/ATOM/issues/3668#issuecomment-7008540288):
runs priced by its law are graded end to end against real runs, DeepSeek-V4.1-Flash's cell
A first ([#3720](https://github.com/jgong5/ATOM/issues/3720)), and D118's bar 2 does not
hold it.

### Problem

D113's list records each decode rung on a server of its own, an 1800 s client run starting
every trajectory at its first turn (`atom/compass/backends/step_linear_plan.py::records`,
`scripts/compass/calibrate.sh`), which reaches its rung only while those sessions decode
together at the start; #3668 tabulates the hours a law takes so. Its prefill comes from
trace runs of real traffic (`step_linear_plan.py::TRACE_RUNS`), which move the prefill
features as the traffic happens to. Each of the law's vectors is linear in a few features (`atom/compass/backends/step_linear.py`),
and a few steps that move each feature apart from the rest settle it.

### Decision

**A calibration** takes the warm record and `measure memory` as today, then these records:

| Record | What it is | What the fits take from it |
|---|---|---|
| the real record | an ordinary record at the space's top concurrency, sampled starts, 600 s | decode and host labels on real load; the levels |
| `design 1`, `design 2` | a decode design record each, by its own seed | decode, draft decode and host labels |
| `trace 16` | as today, the c=16 record under the profiler | prefill and draft prefill labels on real load |
| `trace design 1`, `trace design 2` | a prefill design record each under the profiler, by its own seed | prefill and draft prefill labels |

Every record replays the calibration traces `calibrate.sh`'s `CAL_TRACES` names, apart from
those a law is graded on (`13` D80), and the design records cut their prompts from them.
Where the top concurrency passes their count, the real record's RECORD names `--cache-bust
first_turn_suffix`: aiperf 0.12.0 refuses more lanes than traces otherwise
(`aiperf/timing/trajectory_source.py::validate_dataset_wrap_policy`). Each is a `compass
record` directory, so `fit steps` and `fit host` read them unchanged, and the
leave-one-run-out folds hold the real record and `trace 16`: `19` D112's 5% gate is judged
on real load too. `fit host` takes the untraced records, as `calibrate.sh` passes them.

**The levels.** Three context levels per rank, the real record's decode rows' p10, p50 and
p90 there, each session's context drawn within ten percentiles of its level by the record's
own seed. At a rung of r rows a rank, its gap batch included, the three are scaled by
min(1, 0.8 P / T), P the rank's KV pool in tokens and T the top-level contexts the rung's
first r sessions drew: the top-level group holds at most 80% of the pool, and the others
keep their ratio to it. A draw reaches past p90 up to p100, so the group's total, not r
times p90, is what the pool must hold. A trace design record's
history levels are the real record's prefill rows' cached history, N_KV - N_Q, at p10, p50
and p90 over the rows that have one, scaled the same way over a point's n rows.

**Sessions.** A rank's session i is one conversation of the calibration traces cut at three
lengths, its draw at each level. Each session's first message opens with a marker of its
own, where aiperf's `--cache-bust system_prefix` puts one
(`aiperf/workers/worker.py::_apply_system_target_cache_bust`), so no two sessions share a
KV block, two cut from one trace included, at a cut inside its first turn as anywhere. A
group's rows are distinct sessions, the first ones of the rung's.

A hit resumes as `atom/model_engine/block_manager.py` allows: over a cached prefix of whole
blocks short of the prompt's last block (`BlockManager._walk_prefix`), and on DeepSeek-V4
only at a state checkpoint (`_gated_hit`). A prefill forward keeps one interval
checkpoint, the last multiple of `--state-checkpoint-interval-tokens` in its chunk
(`checkpoint_cut`), so at most one a chunk; a prompt keeps one more, its anchor, at its
last whole block, or a block before its end when it ends on a block boundary
(`_record_checkpoint_end`, `ATOM_STATE_CHECKPOINT_END_MARGIN` unset). So a session's cuts
fill bottom first: each extends the one below and resumes at that cut's anchor, unless the
lower cut's generation prompt, which the cut above renders otherwise, starts before it, and
then at the checkpoint before. DeepSeek-V4.1 under its default encoder SWA bounded replay,
as cell A runs it, keeps no state checkpoint: a hit resumes at any whole block the walk
reaches, and its forward recomputes the block below to rebuild the sliding window
(`BlockManager.replay_start`, under the `replay` transfer of
`atom/model_ops/attentions/deepseek_v41/backend.py::state_transfer`). A forward finishing a
prompt with fewer rows than the window starts that many earlier, on the block grid
(`Config.min_final_prefill_rows`, the window under decoder SWA bounded replay), but a block
holds at least the window less one row
(`atom/models/deepseek_v41/config.py::validate_runtime_config`), so that rule never moves
a hit's resume. On DeepSeek-V4.1 a cut therefore resumes at the start of the last whole
block it shares with the cut below, whichever fills first, and on cell A's 128-token blocks
a resent prompt recomputes 129 to 256 tokens. A chat template that renders an earlier turn
otherwise once a later user turn follows, as one dropping earlier turns' reasoning does,
parts the two renderings at the first turn it changes instead, which can sit far below the
anchor (T123). A cut an earlier group filled fills again at a hit while the pool keeps it,
and the pool evicts its least recently freed cached block, and a vacant one before any
(`atom/model_engine/block_pool.py::BlockPool`). Unscaled levels are the same at every
rung, so one list of sessions serves every rung below the cap, each filled once; a rung
past the cap fills its own.

**A design record** is one server at the configuration's arguments and the arm's widest
`--max-num-seqs`, since a law's key holds neither the scheduler's bounds nor
`enable_prefix_caching` (`atom/compass/backends/step_grid_plan.py::UNKEYED`), its tokenizer
cache holding every prompt the record sends (step 2):

1. *Groups.* At each rung `step_linear_plan.py::scheduled_rungs` gives, the eager one at the
   bound, every rank at the rung's rows:
   - each level alone, every row at it;
   - one row at the top level over the rest at the bottom;
   - at a rung of four rows or more, half at the top level and half at the bottom, twice
     from one fill: alternating, and in halves, the top half first.

   On a gap in the ladder, the first two kinds again at the smallest batch that replays the
   rung.
2. *Fill, then send.* Before each group the driver sends its sessions' prompts, each with
   one output token, and waits for all of them. Then the group's rows go at once, in the
   group's order, each its fill's prompt again with `max_tokens` about 40 decode steps at
   the forced acceptance length, so a row computes the last block of its fill's prompt, and
   on DeepSeek-V4.1 the block below it too.
   A row's prompt of 100,000 characters or more is then served by ATOM's segment tokenizer
   from the ids its fill stored, at the cost of comparing the two
   (`atom/model_engine/segment_tokenizer.py::SegmentTokenizer`, `06` D33), while its cache
   keeps them. So the server's `ATOM_TOKENIZE_CACHE_MB` holds every prompt the record sends,
   at four bytes a character and twelve an id, more than the cache keeps of a prompt: its
   text, its ids with their block hashes, and its segments' ids
   (`segment_tokenizer.py::_Prefix.payload_bytes`). The variable selects no kernel or
   shape, and no law's key holds it (`atom/compass/run/step_grid.py::keyed_env`).
   A rung runs its bottom level's group first, then the middle level's and the top level's,
   then the groups holding top rows, each kind's gap batch after it.
3. *DP.* Every request names its rank by `data_parallel_rank`, which the chat route takes
   ahead of session affinity
   (`atom/model_engine/engine_core_mgr.py::CoreManager._select_dp_rank_for_seq_locked`).
   At each rung the middle level's group runs again with the other ranks at half its rows,
   at one row, with one rank empty and with every other rank empty; the full rank rotates
   from group to group.
4. *Requests* are chat completions, each carrying the real record's request parameters, the
   fields client.sh's aiperf adds to every request (`ignore_eos` and each `--extra-inputs`
   of its RECORD, `atom/compass/drive/sweep.py::_sampling`), and under DP its rank. The
   driver writes the record's client-run export (`atom/compass/parity/served.py`): each
   request by its response id, with the trace its session was cut from and those
   parameters as its `sampling`, neither its rank nor its prompt. Beside it go each
   request's send and acknowledgement as aiperf's raw records state them, which `fit
   host`'s hops read (`atom/compass/fit/host.py::_client`), and each phase's and group's
   wall time. A fit refuses a record no export names (`atom/compass/fit/publish.py::inputs`),
   one whose requests carry two parameter sets, and inputs whose sets differ
   (`atom/compass/fit/samples.py::_record`, `load`).
5. *Cover.* Once the server stops, steps per rung and group are read from the record; a
   group short of half its planned steps is resent in a new record, three rounds at most,
   as D106 resends.

A design record's pacing is the deployment's own.

**A trace design record** is a `compass record --trace` server at the same arguments, at DP
1 and PP 1 only, where `--trace` runs. A point is n rows of q new tokens, each on a cached
history of h tokens:

- q doubling from 1 to the batched-token budget, h at 0 and at each history level, and n
  doubling from 1 while n q fits the budget and n rows beside the point's parked decode fit
  `--max-num-seqs`: a prefill step admits no row once the running sequences, parked
  decodes included, reach it (`atom/model_engine/scheduler.py::Scheduler.schedule`), the
  bound `atom/compass/drive/sweep_plan.py::_fits` holds a JIT sweep's point to;
- at each history level, one row of the budget, and one past it, whose first chunk is a
  middle chunk;
- where the runner records prefill token graphs (`PREFILL_BUCKETS`), one row at each
  bucket's tokens and one past it.

Histories are sessions as the decode design's, drawn the same way, in token ids cut from
the calibration traces' tokenized messages at whole KV blocks. A point's rows take distinct
histories, the first ones of its level, and points of one scaling share one list: each
list's points run bottom level first, then the middle and the top, n and q ascending, so a
history fills once, from the one below, while the pool keeps it. A history's fill is its
tokens and the trace's next one, which puts the fill's anchor at the history's end, so a
row on that history computes exactly its q tokens. A point's rows are planned as ATOM
admits a row on its history, its query from the last position a hit resumes from, as the
step record states it.

The points go as D106 sends a designed step (`atom/compass/drive/sweep.py::send`): under
D106's pacing variables, the prompts a point's rows reuse sent first and read to their end,
then one parked decode, then the rows. Every request asks for two output tokens with the
real record's request parameters, and the record's export is a design record's without the
raw records, since `fit host` takes no traced record. A point is covered when a step ran
its rows (`atom/compass/drive/sweep_plan.py::covered`); missing points are resent in a new
record, three rounds at most.

`calibrate.sh` takes `design K` and `trace design K` as it takes `trace C`; the planner sits
in `step_linear_plan.py` (D37), the decode driver beside `drive/sweep.py`, the prefill
points on its sender, and the law's provenance names the design records.

### Coverage

Each feature the law's vectors fit and the steps that move it apart from the rest, or the
item that holds it. A row holds for the target and, under a drafter, the draft, unless it
names one.

| Vector and feature | Scenario | Steps that move it apart |
|---|---|---|
| decode, each rung and `eager`: `step`, `context` (Sum N_KV) | TP, DP, EP | the level groups: Sum N_KV at three values, `graph_padding` near 0 |
| `graph_padding` (r.max N_KV - Sum N_KV) | TP, DP, EP | the one-long group, and the gap batch's groups |
| `moe_tokens`, `moe_experts` | one rank: TP, or EP at DP 1 | the gap batch's groups; at a rung with no gap the tokens are fixed under any load |
| `moe_tokens`, `moe_experts` | DP, or EP above DP 1 | the middle level's group with its other ranks at half, one row and empty: the group's tokens move at the full rank's own rows, and each empty rank's dummy moves `moe_experts` apart from `moe_tokens` |
| `rowmod(k)` | TP, DP, EP | the half-long pair, one multiset in two orders, which moves rowmod(k) at fixed Sum N_KV and max N_KV for every power of two k from 2 to r/2, the k `step_linear_fit.py::_rowmod` tries; T127 |
| the draft's decode vectors | MTP, DSpark | the target's groups; the draft law adopts no MoE column (`step_linear_fit.py::_draft_options`) |
| `draft_kv`: `step`, `seqs` | MTP, DSpark | the rungs and the gap batches |
| `forward` per pass: `step`, `tokens`, `query_square`, `query_cached`; `propose`: `step`, `seqs` | EAGLE3 | the target's groups: rows and levels per pass, and the first pass's K+1-token rows against the later passes' one-token rows, which tell `tokens` from `query_square` (`step_linear.py::serial_passes`) |
| prefill: `step`, `tokens`, `query_square` | DP 1, PP 1 | the points at h = 0: q alone, and n rows at one total |
| `query_cached`, `history` | DP 1, PP 1 | the points over q and h |
| `cached_seqs`, in the `cached_seqs` form | DP 1, PP 1 | the points over n and h: it moves with n, `history` with h too |
| `seqs`, `out` | a drafter | every point an output prefill, against the budget rows' middle chunks |
| `padding` | the target, where the runner records buckets | the bucket points and the points one past |
| the draft's prefill vector, `propose`, `output_prefill_kv` | MTP, DSpark | the same points' draft, propose and draft-KV passes (`step_labels.py`) |
| `short_propose` | | none: no fit writes it, T130 |
| prefill under DP: `moe_rows`, `serial_moe_rows`; the draft's prefill and `propose` | DP | no designed point: `--trace` refuses DP ([#3010](https://github.com/jgong5/ATOM/issues/3010)), T128. A fit that labels an untraced prefill from its own window, as `COMPASS_PREFILL_LABEL=record` does (`atom/compass/backends/step_labels.py::PREFILL_LABEL_ENV`), takes every record's prefill steps, the design records' fills among them, as cell A's laws take their records' ([#3720](https://github.com/jgong5/ATOM/issues/3720)) |
| host laws: each kind's `[1, tokens, seqs]` per segment; `admit`; tokenizer; HTTP intake; hops | untraced records | decode from the rungs and gap batches; prefill and middle from the fills; `admit` from fills, uncached, against rows, cached; lone calls and requests from the real record and the one-row groups; hops from the client-run exports |

The layouts and the step kinds no vector fits:

| Scenario | Coverage |
|---|---|
| TP | every row above, at one rank |
| DP attention | decode as above; prefill, T128 |
| EP | as DP above DP 1 and as TP at DP 1, its law keyed apart by `enable_expert_parallel` (`15` D92.1); validated on cell A |
| a DSpark drafter | the rows above that name it; validated on cell A |
| two-batch overlap | a decode splits only under `--enable-tbo all`, a bare `--enable-tbo` being prefill only (`atom/model_engine/arg_utils.py`), and only where every rank runs more than two rows and none a dummy (`atom/utils/tbo/ubatching.py::local_tbo_precompute`, `ubatch_splitting.py::decode_bucket_splits`): then a group splits by request count into halves in scheduled order, as a real batch of its size, and the half-long pair gives it balanced and unbalanced halves; its validation, T125. Prefill splits, T128 |
| DP dummy forwards, and decodes beside a prefilling peer | no record fits either: a dummy rank gives no row, and a step whose ranks fall to different vectors is cut (`step_linear_fit.py::_groups`); T131 |
| a pipeline stage | prefill from the record list's trace runs, each stage reduced on its own ([`09` D54.1](09_fitting_and_law_selection.md#d541-a-step-linear-law-of-a-pipeline-stage)); designed load at a stage and a stage's decode, T129 |
| a 1P1D side | T121 |
| context parallel | decode: the law's engine config states the width and the DCP knobs (`atom/compass/run/step_grid.py::keyed`), so a run at another refuses it (`run/__init__.py::refuse_unfitted_law`); prefill: out, the fit refuses it (`step_linear_fit.py::refuse_context_parallel`) |

### Reason

- **The levels.** A rung's decode vector tells its intercept from its Sum N_KV slope only
  where its groups hold Sum N_KV apart. Capped row by row, a level sat at 0.8 P / n once
  n times it passed 0.8 P: on DeepSeek-V4-Pro at TP8 MTP3
  ([#3344](https://github.com/jgong5/ATOM/issues/3344)'s c=16 calibration record: 36,579
  blocks of 256 tokens, decode rows at p10 94k, p50 152k, p90 230k), p50 met p90 from rung
  64 and all three met from about rung 128. Scaled together by the top level's drawn total,
  a wide rung's top-level group holds 0.8 of the pool and the others their ratio to it,
  about 0.33 and 0.53 at #3344's percentiles, which keeps a group resident, since ATOM's
  pool preempts a row it cannot hold (`03` D13). On cell A a rank's top-level draws average
  1.09 to 1.38 times its p90, and its 64 take 19.7M to 26.6M of the 28.4M a rank allows, so
  the cap does not bind on their mean (Summary).
- **The groups.** A level group moves Sum N_KV with `graph_padding` near 0, the one-long
  group `graph_padding`. On a full level or one-long group, rowmod(k) is about Sum N_KV / k
  plus (1 - 1/k) / (r - 1) times `graph_padding`, so those groups cannot tell the column from
  the base features; the half-long pair moves it alone, and shuffling its rows undoes that,
  the test its rule applies (`step_linear_fit.py::_rowmod`). At one rank a rung's MoE
  tokens move only with its rows; under DP they are the group's, so peers' rows move them
  at a rank's fixed features (`step_linear.py::moe_tokens`).
- **Two design records** give every rung a held-out error: a fold prices no step at a rung
  its law has no vector for, counting it refused (`step_linear_fit.py::_held_out`,
  `fold_figures`), and a rung one record reaches adopts no MoE column (`_moe_column`).
  Drawn contexts make the fold a held-out test, not a repeat.
- **Levels from a served record.** A decode row's context is what admission and its turn's
  output make of a trace's messages, which only a step record states; replaying the
  calibration traces, it carries their contexts, which may sit apart from the graded
  traces' (T124).
- **Fill group by group, a session one conversation, from its bottom cut.** A rung's three
  level groups take r (p10 + p50 + p90) tokens a rank, past DeepSeek-V4-Pro TP8's pool from
  about r = 20, so they cannot all stay cached. A session's lower cuts are prefixes of its
  top cut and a group's rows are distinct sessions, so the fills compute about one top cut
  a session while no row of a step reads another row's KV; filled group by group and point
  by point instead, DeepSeek-V4-Pro TP8's fills prefill about twelve times as much
  (Summary). On DeepSeek-V4, from the bottom, each cut resumes within a block of the cut
  below's end; from the top, a lower cut would resume at the top cut's last interval
  checkpoint below it, up to a chunk back. On DeepSeek-V4.1 both orders resume at the start
  of the last whole block two cuts share. A row resending its fill's prompt computes one
  block or two, so up to 64 rows prefill in one step of a 16,384-token budget and 512 in
  four to eight; whether a wide group's rows then enter decode together at the
  deployment's pacing is T126.
- **The tokenizer cache.** At its default 512 MB, one LRU for every rank's prompts
  (`atom/model_engine/byte_lru.py::ByteLRU`), a rung-64 top-level group's fills on cell A,
  about 94M ids, keep some 0.75 GB at a byte a character, so the oldest are evicted before
  their rows come. A row sent in fill order then misses, re-encodes whole and evicts the
  next row's ids: the top-level groups at rungs 48 and 64 and the 49-row gap batch
  re-encode every row, about 90 s of `encode_batch` at cell A's 1.07M tokens/s for the
  widest, against its 40 decode steps of about 2 s. Each tokenizer term prices a call by
  the ids it encodes (`06` D33), so a cache that holds every prompt moves which calls hit,
  not their price.
- **Chat requests.** The record states each row's tokens, which the fit reads, so a decode
  row needs no exact count, and chat requests feed `fit host`'s tokenizer and intake terms
  (Phase 1c). An explicit rank chooses rows that load-aware affinity would not, and
  rotation spreads the card mix a law prices (README principle 12). D106's parked decodes
  would join the very batches a group designs, so the decode groups keep the deployment's
  pacing.
- **Prefill under the profiler.** A prefill label is a trace run's busy time alone, since
  a prefill window holds the device's idle between launches the host paced
  (`step_labels.py`), so the prefill points run in trace records, whose decodes are cut
  and which give no host label. A point's rows must share one step: under D106's pacing
  variables, which select no kernel or shape, #2650's first round ran 17,038 of 17,046
  designed points right first time, and the parked decodes touch no label in a trace
  record. Token ids give each point its exact tokens, bucket edges included, and cut from
  the traces they route the MoE as served text does.
- **Two trace design records**, each the other's fold, as the design records are;
  `trace 16` checks both on real prefill, and `trace 4` leaves the list.
- **The validation, end to end.** The owner's Ruling makes the real runs, end to end, the
  final acceptance, and no record-list law its bar. An end-to-end miss alone does not say
  whose it is, as D118 finds, since a schedule that diverges moves single metrics; `08`
  D47's three figures price each graded run's own steps under the law, so a miss past
  #627's 5% gate on them is the law's, and within it the schedule's or a host term's,
  which `08` D44 reports apart.

### Rejected

- Each level capped alone: from rung 64 on DeepSeek-V4-Pro TP8 the levels merge, and a
  wide rung's intercept and slope do not separate.
- Fixed fractions of the pool at wide rungs: they place the levels where the workload may
  never run, where scaling keeps its spread.
- The levels scaled by r times p90, or each draw capped at 0.8 P / r: draws reach p100, so
  under the first a top group whose draws average past 1.25 times p90 passes the pool, and
  the second clips the top level's tail where the pool needs only the group's total held.
- One prefix shared by a group's rows, filled once: every row would read one row's KV,
  from cache, where real rows read their own.
- Token-id completions for decode rows, as `drive/sweep.py` sends: no tokenizer or
  template runs, so a design record feeds neither term.
- The default tokenizer cache, each group's fills sent in the reverse of its rows' order:
  the cache then keeps the rows sent first, but the rest still re-encode, about a third of a
  rung-64 top-level group on cell A at a byte a character and more at wider ones, some 30 s
  at 1.07M tokens/s against its 40 decode steps.
- Fresh token ids for prefill points, as D106 sends them: they route the MoE as no served
  text does.
- Each level's sessions as conversations of their own: every level fills whole, about
  twice the fill of cuts of one conversation on DeepSeek-V4-Pro TP8 and 1.3 times on cell
  A (Summary).
- A session's top cut filled first, its lower cuts at a hit on it: on DeepSeek-V4 a lower
  cut resumes at the top cut's last interval checkpoint below it, one a forward, and
  computes up to a chunk, on DeepSeek-V4-Pro TP8 about 1.35 times the fill (Summary); on
  DeepSeek-V4.1 it saves nothing.
- Each session's marker at the end of its first turn, where the real record's
  `first_turn_suffix` puts one (`aiperf/workers/worker.py::_inject_marker_at_first_user`):
  sessions cut from one trace share that turn, and the whole prompt at a cut inside it, as
  a cut at DeepSeek-V4-Pro TP8's widest rungs, 6k to 15k tokens, can be.
- An export of each request's payload less `NOT_SAMPLING`, as
  `atom/compass/parity/client_run.py` writes aiperf's: it keeps each request's rank and
  prompt token ids, so a record holds as many parameter sets as prompts, which `fit steps`
  refuses.
- One trace design record beside `trace 16`: its fold is priced by `trace 16` alone,
  extrapolating over the designed extremes, and one figure past 5% fails `19` D112.
- Both design records at the percentiles exactly: each fold re-prices the other's points.
- Dataset wrap for the real record in place of a cache-bust target: wrapped lanes replay
  one trace's prompts byte for byte, each lane's prefill a cache hit no graded run gets.

**D39 is not amended.** Its replay rule binds the step bench's measurements, and a design
record's steps are a real server's. Its reason still bears: hand-built rows may miss real
row order and raggedness, which T124 holds until the real record's fold and the
validation show them.

**Unchanged.** The law's key; `19` D112 and D113's list. A kernel variant the warm record
leaves uncompiled is built in a design record, which the fit then refuses whole, every rung
it holds ([#3618](https://github.com/jgong5/ATOM/issues/3618)).

### Scenarios

**A calibration with MTP at DP 1**, DeepSeek-V4-Pro at TP8 MTP3 as
[#3344](https://github.com/jgong5/ATOM/issues/3344) recorded it:

```
 calibrate.sh OUT '256 --cache-bust first_turn_suffix' 'design 1' 'design 2'
                  'trace 16' 'trace design 1' 'trace design 2'
   warm, memory      as today
   real record       c=256, sampled starts, 600 s, CAL_TRACES
                       -> decode rows' contexts, prefill rows' histories
   design K          plan: pool + real record -> levels, scaled per rung;
                       sessions, each one conversation cut at its three levels,
                       its own marker at its head
                     one server, its tokenizer cache holding every prompt;
                       per rung, bottom level first; per group: its
                       fill (a cut from the anchor of the one below, a hit where
                       filled before), waited for -> its rows at once, each its
                       fill again, in order, ~40 steps; the next fill
                     each request: the real record's parameters; the export
                       names its trace, its sends and acks beside it
                     stop; cover: a group under half its steps -> a new record,
                       <= 3 rounds
   trace 16          as today: real prefill under the profiler
   trace design K    one --trace server under D106's pacing; points list by
                       list, bottom level first; per point: its histories'
                       prompts, each one token past its history (a hit where
                       filled) -> one parked decode -> its rows, at most
                       --max-num-seqs less one; its export
                     stop; cover by rows; missing points -> a new record
   fit steps         folds: real, design 1 and 2, trace 16, trace design 1 and 2
                       -> decode and draft decode vectors at every rung, a MoE
                          column or rowmod where its rule adopts one, draft_kv;
                          prefill, draft prefill, propose, output_prefill_kv;
                          a held-out error at every rung and on prefill
   fit host          real, design 1 and 2: each kind's segments, admit,
                     tokenizer, intake, hops
   status            each step's wall time (#3467); each phase's and group's
   -> a step_linear_law naming its design records; compass advice reuses it as
      any law (D112) and still prints D113's list
```

**Under DP, with EP and a DSpark drafter**, cell A
([#3720](https://github.com/jgong5/ATOM/issues/3720): DeepSeek-V4.1-Flash on one MI355X
server, DPA4-EP4, DSpark5, a bound of 64 rows a rank), its law keyed by
`enable_expert_parallel` (`15` D92.1):

```
 calibrate.sh OUT '256 --cache-bust first_turn_suffix' 'design 1' 'design 2'
                                   no trace record: --trace refuses DP (#3010)
   real record     c=256, cell A's top: decode rows' contexts, rank by rank
   design K        no rung past the cap at the draws' mean: one list of 64
                     sessions a rank, each filled once; a tokenizer cache
                     holding every prompt; a hit resumes at the start of the
                     last whole block two cuts share (no state checkpoint)
                   per group: each rank's fill, data_parallel_rank on each,
                     waited for on every rank -> the group's rows at once
                   per rung: the groups, every rank at the rung's rows; the
                     middle level again, peers at half, one row, one empty,
                     all empty; the full rank rotating
                   each decode times its draft passes and draft KV
   fit steps       decode and draft decode vectors at each rank's rows,
                   priced at the costliest; a MoE column over the group's
                   tokens and dummies where its rule adopts one; draft_kv.
                   Prefill: no designed point (T128); under
                   COMPASS_PREFILL_LABEL=record, every record's
                   prefill steps, the fills among them
   fit host        real, design 1 and 2, as at DP 1
```

**An EAGLE3 drafter**, MiniMax-M3 TP4 (`19` D117):

```
   design K          the groups; each decode times its draft's passes
   trace design K    middle chunks: the draft's prefill vector
   fit steps         the draft's forward per pass, the first at K+1 tokens a
                     row and the rest at one; propose over [1, seqs]
```

**Acceptance**, once the planner and drivers exist:

```
 mechanism [GPU]   QUICKSTART's Qwen3.8-27B on one card: a group holds its
                   rung, its fill stays cached, a point's rows share a step,
                   the trace record stays readable (T122, T123, T126, T132);
                   group time against the estimate
 validation [GPU]  cell A first, on crsuse2-m2m-v3-005 from dsv41/dev: a
                   designed-load calibration at the ATOM digest of its real
                   runs (#3720: c = 1 to 256, agentx, one run each)
   at each c: a simulated run priced by the law and its host laws
     -> against the real run: p90 TTFT within 10%, median TPOT and
        throughput per GPU within 5%, or inside the real runs' spread
     each miss -> 08 D47's three figures on that real run's own steps:
        past 5%   the law's, and the candidate fails
        within    its cause named, as #3720 names cell A's
   beside, not gated: a law from D113's list at the same digest, where one
     exists, and cell A's laws fitted on those runs themselves (#3720)
   then each configuration the GPU plan runs; held: decode at DP 1, MTP,
     EAGLE3 and the trace design records (T133), decode TBO (T125)
   -> T120: whether compass advice prints the designed-load list
```

The metrics are `08` D45.1's for a graded cell. The real runs' spread is `08` D45's, across
three or more spaced real runs; with one at a concurrency, as cell A has, it is that run's
own 95% interval over resamples of its sends' 60 s blocks, which
`scripts/compass/one_real.py` on `dsv41/dev` states for p90 TTFT (`TAILS`, #3720) and for
no other of the three, so median TPOT and throughput meet their 5% alone. Cell A's real
runs replay every trace (`client.sh --agentx`), so the calibration traces the design
records cut from are not apart from them as `13` D80's process keeps a graded cc-traces
run's; no graded run's step enters a fit, and T112 asks the same of MiniMax-M3's agentx
runs.

### Simpler alternative

The decode design alone, with prefill from today's `trace 16` and `trace 4`: one start
fewer and no prefill plan. It fails the Ruling on #3668, which has the design cover every
feature the law fits: real traffic moves the prefill features as it happens to, so for
`padding`, `cached_seqs` or `seqs` no step can be named that moves it apart. No fill, each
row a fresh prompt prefilled once its group is sent, departs from #3668's candidate, whose
rows are cached sessions' prompts, and costs each group its rows' whole contexts: on cell A
some 390M tokens a rank a design record, 2.6 h or more at a rank's 40k tokens/s, against 10
to 11 min of fills. Its rows would still enter decode together there, since a runner that
records prefill token graphs keeps an idle DP rank on the group's dummy prefill
(`atom/model_engine/scheduler.py::Scheduler._group_prefill_tick`).

### Low confidence

None. T122 to T124, T126 and T132 are what the mechanism check and the validation measure,
each resting on a decision above; none chooses between designs before the planner and
drivers exist. T125 and T133 wait for a configuration the GPU plan runs with decode TBO, on
a model other than DeepSeek-V4.1, which refuses it, or at DP 1.

### Dependencies

No spike and no sync issue. Cell A's validation needs the code tasks on `dsv41/dev`, from
which DeepSeek-V4.1-Flash runs ([#3670](https://github.com/jgong5/ATOM/issues/3670)), and
nothing of [#3468](https://github.com/jgong5/ATOM/issues/3468)'s
([Ruling](https://github.com/jgong5/ATOM/issues/3468#issuecomment-7009504788)). Designed
prefill under DP waits on [#3509](https://github.com/jgong5/ATOM/issues/3509)'s gated
record (T128), and the validation of decode TBO and of the scenarios cell A does not run on
a configuration the GPU plan runs (T125, T133).

### Summary

**Deliverable.** Laws from designed load beside D113's list, every feature the law fits
moved apart in each scenario above. **Not in scope:** the printed list (T120); prefill under
DP (T128); a pipeline stage (T129); a 1P1D side (T121); context parallel; `short_propose`
(T130); the validation of decode TBO and of the scenarios cell A does not run (T125, T133).

**Estimate**, derived and unmeasured. On cell A a law takes about 1.4 to 2 h of its four
cards, against about 6.8 to 7 h for D113's list there, ten first-turn records at c = 4 to
256 and no trace run (`atom/compass/backends/step_linear_plan.py::records`), and 4.8 h for
the six real records cell A's laws are fitted on (#3720). At one concurrency it costs more
than cell A's in-sample law, one real record of 41 to 61 min: what it adds is a law that
holds every rung, priced apart from the runs it is graded on. On DeepSeek-V4-Pro at TP8
MTP3, with the state checkpoints and the trace design records cell A does not run, about
3.6 to 5 h against 10.9 to 12.5 h, at ATOM's default capture sizes, 1 to 8, 16, 32, 48,
64, 128, 256 and 512 (`atom/model_engine/arg_utils.py::EngineArgs.cudagraph_capture_sizes`).
D113's list counts the records past the calibration traces' count, which #3455's handoff
reads as unrunnable. MiniMax-M3's waits on its pool and contexts.

<details><summary>Cell A</summary>

- Basis: cell A's c=256 record (#3720), standing for the real record: a rank's pool of
  277,491 blocks of 128 tokens (35.5M); decode rows in its graded window at p10 10k to 28k,
  p50 81k to 92k and p90 257k to 364k, by rank; a step of 32,768 tokens on every rank in
  about 0.8 s at p50, some 40k tokens/s a rank; a decode step at 57 to 64 rows and about 11M
  tokens a rank in 40 ms at p50. Cell A's records took 41 to 61 min of four cards at c = 1
  to 256, a start 7 to 8 min to a served endpoint and the c=256 run's warmup 21 min;
  `measure memory` took 6.3 min.
- Fills: no rung past the cap on the draws' mean, so 64 sessions a rank, each its top cut
  once, drawn within ten percentiles of p90: 19.7M to 26.6M tokens a rank, 10 to 11 min a
  design record at the slowest rank. Each level a conversation of its own: 1.3 times that.
- Groups: 128 a design record, each of ten rungs' level, one-long and half-long groups and
  four DP groups, and four at each of eight gap batches; 3,112 rows a rank, each
  recomputing 129 to 256 tokens: 3 to 6 min.
- The rest: a start and a stop for warm, memory and each of three records; warm's and the
  real record's 600 s client runs, and the real record's warmup, at most the c=256 run's.
- D113's list: a start and a stop for warm, memory and each record, a first-turn record's
  1800 s. The six real records: their own times.
- Not counted: a resend of uncovered groups; a refill an eviction forces (T123); the
  frontend's, about 4 min a design record one request at a time on cell A's host law: the
  event loop's intake of every fill and row, the chat template's render among it, some 2.9G
  tokens of prompt at about 4 bytes a token and 5.7 ns a body byte, 1 min; each session's
  first encode, about 0.1G tokens at 1.07M tokens/s, 2 min; the bottom level's prompts under
  100,000 characters, which encode whole at each send, 25M tokens at 853k tokens/s, half a
  minute. A longer prompt resent comes from stored ids, about 1 ms a top-level row. How the
  loop spreads a wide group's rows is T126.
</details>

<details><summary>DeepSeek-V4-Pro at TP8 MTP3</summary>

- Basis: #3344's c=16 calibration record, a pool of 36,579 blocks of 256 tokens (9.36M) and
  decode rows at p10 94k, p50 152k, p90 230k, which stand for the prefill histories too;
  prefill at the 24.5k to 32.6k tokens/s of its prefill-side 16,384-token chunks (medians by
  context); a start at 10 to 15 min (#3668); a session's fill its top cut and half a block
  past each lower cut's anchor, a row's and a refill's half a block.
- Fills, each session once: 47M tokens a decode design record, its rows and refills 2M of
  them, 30M a trace design record, at points of up to 256 rows; all four records 155M, 1.3
  to 1.8 h. The top cut first, at a rung every 16,384 tokens: 207M. Each level a
  conversation of its own: 313M. Each group and point filling its own: 1.9G.
- The rest: starts of warm, memory and six records, 1.3 to 2.0 h; client runs 22 min (warm
  and real 600 s each, `trace 16` 120 s); groups 3 to 5 min a decode design record, points
  15 to 20 min a trace design record.
- D113's list: a start for warm, memory and each record, a first-turn record's 1800 s, a
  trace run's 120 s.
- Not counted: a resend of uncovered groups or points, a record and a start of its own; a
  cut that resumes before the anchor below it, the generation prompt or a re-rendered turn
  starting first; the frontend's time on a decode design record's chat requests, about
  0.54G tokens with their fills.
</details>

| Where | Change | Task |
|---|---|---|
| `atom/compass/backends/step_linear_plan.py` | the design plans: configuration, memory reading and real record in; levels, sessions and their cuts, groups and prefill points, in the order they run, and the tokenizer cache that holds a design record's prompts, out; CPU tests | 1 |
| beside `atom/compass/drive/sweep.py` | each group's fill and send, stop and cover; the client-run export, its sends and acks beside it; phase and group times | 2 |
| `atom/compass/drive/sweep.py` | the prefill points sent and covered on its sender, in a trace record, with its export | 3 |
| `scripts/compass/calibrate.sh` | `design K` and `trace design K`, a design record's server at the planner's tokenizer cache; the law's provenance names them | 4 |
| one card [GPU] | the mechanism check | 5 |
| crsuse2-m2m-v3-005 [GPU] | cell A's validation, from `dsv41/dev`; then each configuration the GPU plan runs | 6 |

**Acceptance.** The validation passes end to end on cell A, then on each configuration the
GPU plan runs; every scheduled rung has a vector and a held-out error, and at DP 1 the
prefill vector one too; each step's wall time is recorded
([#3467](https://github.com/jgong5/ATOM/issues/3467)); the mechanism holds on one card.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D36 | Two granularities, step-level and op-level, and a method per term (analytical, declared, measured, estimated) replace the three-way split of cost models; fallback runs only along method inside one granularity. Compass models a device it has been measured on; an input observed on another subject makes a term declared, and doc 05 D26 has no `transfer` probe. The op-level backend is empirical on a measured device, over a concrete IR graph per decode width and prefill (T, B): each leaf priced from a `price_list`, else by its analytical law, with no residual, so a new engine config, layout or shape needs only its missing leaf signatures measured. | 2026-09-18; probe dropped 2026-10-14 (#937); two granularities 2026-10-14 (#952); op level empirical 2026-10-18 (#1489) |
| D37 | The tool tells the user what to measure: a backend that needs measurements plans them beside its own code and owns the coverage check that `compass run --coverage` calls before the run starts; the step grid's planner runs inside `measure steps`, over `shape_population` entries and `record` directories, and the op-level backend's inside `measure ops --plan`, over the steps of record directories from `record` or `trace steps --out`; no single command plans the campaign. Coverage checks use a convex hull or k-NN distance, never a bounding box. On the step grid, a prediction with no real run of its target plans from its own simulated runs, round by round until the planner places no new point, its last run gated on no step falling to the roofline where a point can be placed; real runs, where grading has them, stay a planning input. The op level keeps the coverage check and one top-up. | 2026-09-18; revised 2026-10-13 and 2026-10-14 (#851); op-level planner 2026-10-18 (#1489); its input, record directories, 2026-10-18 (#1527); a step grid prediction planned from its simulated runs 2026-11-19 ([#2678](https://github.com/jgong5/ATOM/issues/2678)) |
| D38 | Phase 0 discovery (shapes only) -> 1a trace -> 1b price -> 1c in-situ (+ one TP2 transfer test) -> Phase 2 memory. Client count and workload are not calibration axes. `13` D80's commands run them: `trace steps`, `trace ops`, `measure ops` and `fit ops`, `record` and `measure steps` then `fit steps\|host`, and `measure memory` for Phase 2, one engine start per configuration whose `memory_readings`, or a law file's, a run reads and refuses without; `measure` writes per point, `fit` makes laws, bundles and leaves. Phase 1a captures one concrete graph per decode width and prefill (T, B). Phase 1b times each IR leaf alone from its contract at a point `ModelRunner.prepare_model` prepared, on a runner whose parameters are on the meta device, running no forward, warmup included: activations random by spec, weights random by spec over as many tensors as the leaf has instances, metadata, forward context, KV and module tensors from the runner, a host argument built on the host, pinned, a copy with one ATOM keeps pageable refused, and a referenced module's parameters filled on the device as `--load_dummy xavier` fills them, by CUDA-graph replay over rotating copies, module-attribute tensors included, whose total exceeds the last-level cache, each cold replay after a read of twice the cache outside its timing, at points planned from the steps of record directories from `record` or `trace steps --out`, at the decode rungs they replay and the ladder's next above the widest, each point recording the kernels its leaf launched, with the warm-over-cold ratio reported and the card's PTL state keyed; MoE leaves on uniform routing; `fit ops` interpolates only between same-kernel neighbours (`09` D60) and declines any other query, which the leaf's analytical law answers, counted. Phase 1c feeds the op-level backend nothing. Phase 1c's records also time each served tokenizer call and stamp each HTTP request's intake, which `fit host` fits into the `host_laws` entry's `tokenizer` and `http_intake` terms, the intake on its loop thread's CPU time, its `per_byte_s` and `idle_s` from lone requests and its hull over body bytes, the tokenizer on lone calls' wall time, one-item and several-item encodes and decode as three kinds, and the segment tokenizer call's own work over the ids it encodes as a fourth; an unfitted term or kind is listed; every host-law term stays keyed by model. | 2026-09-18; commands 2026-10-13 (#851); Phase 2 2026-10-14 (#851); Phases 1b and 1c 2026-10-18 (#1489); 1b's points planned from record directories 2026-10-18 (#1527); 1b without a forward 2026-10-21 ([#1868](https://github.com/jgong5/ATOM/issues/1868)); referenced modules' parameters filled 2026-10-21 ([#1869](https://github.com/jgong5/ATOM/issues/1869#issuecomment-6321134450)); the cache evicted before each cold replay 2026-10-23 ([#2019](https://github.com/jgong5/ATOM/issues/2019)); tokenizer and HTTP intake fitted from records 2026-11-03 ([#2309](https://github.com/jgong5/ATOM/issues/2309#issuecomment-6498578963)); intake on CPU time 2026-11-04, tokenizer on lone calls by kind 2026-11-05 ([#2751](https://github.com/jgong5/ATOM/issues/2751#issuecomment-6655175759)); the segment tokenizer call 2026-11-09 ([#2618](https://github.com/jgong5/ATOM/issues/2618)); host arguments 2026-11-10 ([#2857](https://github.com/jgong5/ATOM/issues/2857)); pageable host memory refused 2026-11-12 ([#3172](https://github.com/jgong5/ATOM/issues/3172)); the rungs the steps replay and the next, and Phase 0's margin checked after the fact only against real runs, 2026-11-19 ([#2678](https://github.com/jgong5/ATOM/issues/2678)) |
| D39 | The step bench is `EngineCore` without its scheduler: every TP rank's `RecordingModelRunner` started through its runner manager in its order, at any TP width, a step charged to its slowest rank, fed **replayed** batches rebuilt from step records by a versioned rule. One window for both step kinds: device time from an event at the forward's entry to one at its exit, with device work queued ahead, a repeat counting only if the entry event is still pending when the forward returns; whatever is fitted on it states that value kind. Step measurements may not use synthetic shapes, the step grid's excepted (`17` D98). The op bench, apart from the step bench, runs no forward: ATOM's runner with its parameters on the meta device and no warmup, `ModelRunner.prepare_model` building each placement's metadata and forward context, and the bench timing each leaf from its contract; on one GPU at a DP-attention layout, the runner at one rank's own width, collectives measured across the layout's ranks. `measure steps --gpu` naming k times the TP width's cards runs k benches over one queue of points into one store, a bench joining unless its median difference from the first bench over a sample of points exceeds the largest between the first bench's two passes over them; the op bench stays one bench an invocation. | 2026-09-18; revised 2026-10-13 (#851); op bench 2026-10-18 (#1489); op bench without a forward 2026-10-21 ([#1868](https://github.com/jgong5/ATOM/issues/1868)); several cards for the step bench 2026-11-19 ([#2678](https://github.com/jgong5/ATOM/issues/2678)) |
| D40 | Four classes of communication, priced separately. Scaling is measured per width, never modelled. A collective's measured duration is an upper bound on device work. Collectives are leaves, interpolated over message bytes, measured on a multi-rank op bench that starts every rank of the layout through ATOM's own group initialisation, takes a barrier before each timed replay and measures the union of all ranks' signatures; the step bench is not reused. | 2026-09-18; collectives as leaves 2026-10-18 (#1489) |
| D41 | Artifacts, each with key, digest, fingerprint and provenance. Keys are tuples; source roots are `git archive` digests; a source root with uncommitted files does not publish; resolution names its answer; handed-off artifacts are immutable. `step_measurement`, `step_linear_law` and `host_laws` added, and `shape_population` keyed by its pricing backend. `memory_readings` from `measure memory`, keyed as `step_linear_law` is, or published with every reading assumed for a device never measured. `price_list` and `memory_readings` keyed by the parallel layout, not a scalar width. The region-term artifact retired; `op_measurement` added, one per `measure ops` point, keyed as `price_list` is plus the point, the layout only for a collective leaf; `op_graph` keyed by structure and shape, and by the torch it was traced under and the files its trace ran and read. A law file carries its readings, assumed, which `run` reads when its store holds none, keyed as a store entry is and checked by their `atom_src`, the host key's `atom` digest; a run takes the host CPU of host laws from the laws. | 2026-09-18; kinds 2026-10-13 (#851); `memory_readings` 2026-10-14 (#851); parallel layout 2026-10-15 (#165, #537); region terms retired, `op_measurement`, `op_graph` per shape 2026-10-18 (#1489); a dirty tree publishes nothing 2026-10-25 (#168); ATOM's source read as the host key's `atom` digest, not the git tree, 2026-10-29 ([#2310](https://github.com/jgong5/ATOM/issues/2310#issuecomment-6497009949)); `probe_record` and `platform_readings` replace `machine_spec`, and each kind keys the device, stack, PTL and torch its row marks, read from its records, in place of the machine-spec digest, 2026-11-03 (#2716, [ruling on #2724](https://github.com/jgong5/ATOM/pull/2724#issuecomment-6596632012)); `op_graph` keyed by the ATOM source and torch it was traced under, 2026-11-11 (#3155, [ruling on #3124](https://github.com/jgong5/ATOM/pull/3124#issuecomment-6768936508)), and by the files its trace ran and read in place of the whole source, 2026-11-17 (#3156); `op_measurement` keys the layout only for a collective leaf, 2026-11-14 ([#3408](https://github.com/jgong5/ATOM/issues/3408), [Ruling on #1500](https://github.com/jgong5/ATOM/issues/1500#issuecomment-6870338230)); a law file's readings, and the host CPU from the laws, 2026-11-10 ([#2899](https://github.com/jgong5/ATOM/issues/2899)) |
| D42 | Trace is JIT and cached; pricing is offline or opt-in `--measure`. A `--measure` run is never an acceptance run. A new shape is captured, and each leaf in it interpolated between same-kernel neighbours; anywhere else, and for a new operator kind, the leaf falls to its analytical law, counted, and refuses without one. | 2026-09-18; same-kernel interpolation 2026-10-18 (#1489) |
| D43 | Per-artifact invalidation matrix, not a global fingerprint. Every gate's state is written into the artifact and checked there. Rows for `shape_population`, `step_measurement`, `step_linear_law` and `host_laws`, and a host-CPU axis. The region-term row retired; `op_measurement` on `price_list`'s row. The tokenizer terms leave the spec for `host_laws`, whose row the host CPU invalidates. A law file's readings re-key with its host key, and are taken again by a commit that moves what a start under the law's engine config holds. | 2026-09-18; rows 2026-10-13 (#851); region-term row retired, `op_measurement` 2026-10-18 (#1489); `host_laws` row gains the device, stack and PTL its machine-spec digest carries, and torch through its fingerprint, 2026-10-29 (#2484); the ATOM src column reads the host key's `atom` digest 2026-10-29 ([#2310](https://github.com/jgong5/ATOM/issues/2310#issuecomment-6497009949)); PTL back to `-`, carried only because the digest is one hash, and a run does not yet check torch for host laws, 2026-10-30 (#2308, #2475); the `machine_spec` rows give way to `probe_record` and `platform_readings`, and `host_laws` keys no PTL, 2026-11-03 (#2716); torch is keyed wherever its row marks it, 2026-11-03 ([ruling on #2724](https://github.com/jgong5/ATOM/pull/2724#issuecomment-6596632012)); tokenizer terms to `host_laws` 2026-11-03 ([#2309](https://github.com/jgong5/ATOM/issues/2309#issuecomment-6498578963)); a law file's readings re-keyed, or taken again, 2026-11-10 ([#2899](https://github.com/jgong5/ATOM/issues/2899)); the `op_graph` row's engine config is a step law's less its ladder, as its key has held it since 2026-10-20 ([#1811](https://github.com/jgong5/ATOM/issues/1811)), and less its compilation level 2026-11-17 ([ruling on #3052](https://github.com/jgong5/ATOM/issues/3052#issuecomment-6942839264)) |
| D38.1 | One recommended flow per half: steps (`measure memory`, `record`, `trace steps`, `measure steps` with the step grid's planner, `fit host` and `fit steps`, `run --coverage`, then grading outside the CLI), each server driven by a client started apart, and ops, after the steps half's `measure memory`, `record`, `trace steps` and `fit host` (`trace ops --population`, `measure ops --plan` over the record directories of `record` or `trace steps --out`, `fit ops` publishing a `price_list` only, then `run` and grading). Phases unchanged; the op bench, ATOM's runner preparing each point and running no forward, hosts 1b and the step bench 1c, Phase 2 is a plain startup `measure memory` runs. | 2026-09-19; revised 2026-10-13 and 2026-10-14 (#851); ops half 2026-10-18 (#1489); `measure ops --plan` reads record directories 2026-10-18 (#1527); op bench without a forward 2026-10-21 ([#1868](https://github.com/jgong5/ATOM/issues/1868)) |
| D40.1 | Classes (a) and p2p ride `measure ops` unchanged. Classes (b) and (c) share one new stated-entry-point benchmark inside the same multi-rank op bench, also run by `measure ops`. KV transfer is never priced. Collective prices are labelled upper bounds. | 2026-09-19; one command 2026-10-13 (#851); the multi-rank op bench 2026-10-18 (#1489) |
| D106 | A container is warmed for a graded run by a constructed JIT sweep and a per-stack cache library; a compile in a graded run still voids it. A client against the cell's own deployment, no ATOM hook point; only pacing variables that select no kernel or shape differ, set by `compass record --sweep` and checked by `client.sh --sweep`: `ATOM_ENABLE_PREFILL_DELAYER=1`, `ATOM_PREFILL_DELAYER_COST_GATE=0`, `ATOM_PREFILL_DELAYER_TARGET_FILL=1.0`, `ATOM_PREFILL_DELAYER_STALL_TICKS=16`, `ATOM_PREFILL_DECODE_INTERVAL=1` at DP 1 and PP 1, and `ATOM_DP_PREFILL_BALANCE_MIN_TOKENS` at the batched-token budget, so balancing cuts no designed row. The ranks share one walk, each count once per container on a rank the count chooses. A: one single-row request per prefill token count from 1 to the budget, no parked decode, covered by any rank's step of that count TBO did not split. B: lockstep steps, a designed point per rank, the product of the second row's offset mod 16, 1 to 16 parked decodes and TBO on or off, plus chunked rows over the product of query size, its residue mod 16 (and, from the chunk alignment up, each multiple of 16 below it) and context level up to the longest prompt, balancing's aligned cuts below the budget included, a step's rows on as many earlier chunks; TBO points split inside a long first row too, and in split steps other ranks' rows cut into halves of 256 to 511 tokens at every offset mod 16; the costly row last or every rank's first alike; covered by exact rows and split. Parked decodes keep the PrefillDelayer's alignment gate armed, so a step's rows release together; no blocker. Two output tokens a request; axes and ranges from the cell's engine and workload, sampling arguments copied. Missing points resent unchanged, three rounds at most. PP above 1 and rapidserve run A only, not marked swept; a PD cell's decode side and KV connector path are warmed by one unscored run and the library. One library per stack (ROCm, aiter's commit, RCCL, torch, Triton, FlyDSL, gfx arch) at `<store>/jit`, `compass record --store` optional, merged in at a real server's start and written back at its stop, finished artifacts only, per compiler, atomically. Coverage is read from the step record after the stop and kept in the library per stack and cell; a cell is swept only when all its points are covered. Preemption, recompute and error paths are not swept: a build there is recorded, not a defect. Streaming the real record is deferred. | 2026-11-02 ([#2640](https://github.com/jgong5/ATOM/issues/2640#issuecomment-6568381628)); shared walk, delayer coalescing, pacing variables 2026-11-02 ([#2650](https://github.com/jgong5/ATOM/issues/2650#issuecomment-6579531822)); balancing floor, stall 16 2026-11-02 ([#2687](https://github.com/jgong5/ATOM/issues/2687#issuecomment-6591321600)); chunked grid, split inside a first row 2026-11-03 ([#2650](https://github.com/jgong5/ATOM/issues/2650#issuecomment-6607750609)); TBO halves 2026-11-04 ([#2650](https://github.com/jgong5/ATOM/issues/2650#issuecomment-6619270702)); KV residues 2026-11-06 ([#2689](https://github.com/jgong5/ATOM/issues/2689#issuecomment-6686630693)) |
| D118 | A change that cuts calibration time meets one of three bars, named where it is decided and shown by its code task's named result: the same bytes; within the measurement's own repeatability, the same card's pass-to-pass spread measured on the same points under the old way and the new, no number of it in code, the new values' median difference from the old within the largest between two passes of the old; no lost coverage, every step of the planning workloads keeping its measured price and a step outside counted by its backend's coverage check before a graded run. An end-to-end delta tests no cut. Not proposed, as no bar holds: a prefill chunk with and without output priced from one point, and real steps as grid points. The cuts: the step bench on several cards that agree (D39), a step point's repeats in two bursts a pass apart (`17` D98), the op plan's decode rungs (Phase 1b), an op-level run graded only after its coverage check (`13` D80), a step grid prediction planned from its simulated runs (D37), and byte-identical speed-ups of `measure ops`, `trace ops` and the fits. | 2026-11-19 ([#2678](https://github.com/jgong5/ATOM/issues/2678)) |
| D119 | A candidate beside `19` D113's record list, which `compass advice` keeps printing: a step linear law from the warm record and `measure memory`, a real record at the space's top concurrency (sampled starts, 600 s), two decode design records, `trace 16` and two trace design records, fitted and gated as any law's records; every feature the law fits is moved apart in each scenario it supports, or held in topic 12. The levels are the real record's decode-row p10, p50 and p90 per rank, drawn by each record's seed and at a rung of r rows scaled together by min(1, 0.8 P over the top-level contexts its first r sessions drew); a session is one conversation cut at its three levels, its own marker at its head, its cuts filling bottom first, each at a hit from the prompt-end checkpoint of the one below on DeepSeek-V4 and from the start of the last whole block they share on DeepSeek-V4.1, and a cut fills once while the pool keeps it. A design record is one server at the arm's widest bound, its tokenizer cache holding every prompt it sends: at every scheduled rung, bottom level first, each level alone, one long row over short ones and a half-long pair in two orders, and a gap batch's groups; each group's sessions filled and waited for, then its rows at once, each its fill's prompt again, about 40 decode steps; under DP every request names its rank, the middle level again over uneven and empty peers, the full rank rotated; chat completions carrying the real record's request parameters, and a client-run export naming each request's trace and those parameters, its sends and acks beside it; the deployment's pacing; groups under half their steps resent, three rounds at most. A trace design record, at DP 1 and PP 1, sends prefill points of n rows of q tokens on cached histories, each history filled one token past its end, n rows beside one parked decode within `--max-num-seqs`, the budget's middle chunks and the recorded buckets' edges, as D106 sends a designed step, its token ids cut from the calibration traces, with an export as a design record's. The real record names an aiperf cache-bust target past the calibration traces' count, and every record replays those traces. `calibrate.sh` takes `design K` and `trace design K`; the law's provenance names them. Its validation grades runs priced by the law end to end against real runs, DeepSeek-V4.1-Flash's cell A first: p90 TTFT within 10%, median TPOT and throughput per GPU within 5%, or inside the real runs' spread, each miss assigned by `08` D47's three figures on the same runs, past 5% the law's; no record-list law is its bar. | 2026-11-18 ([#3668](https://github.com/jgong5/ATOM/issues/3668), [Ruling](https://github.com/jgong5/ATOM/issues/3668#issuecomment-6974348636)); validation 2026-11-19 ([Ruling](https://github.com/jgong5/ATOM/issues/3668#issuecomment-7008540288)) |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| ~~T16~~ | ~~Calibrate the campaign planner's GPU-time estimates~~ — retired 2026-10-13 on #851: no command plans the whole campaign (D37) | needs one campaign |
| ~~T17~~ | ~~Draw the boundary of the standalone `ModelRunner` bench~~ — drawn 2026-10-13 on #851: `EngineCore` without its scheduler (D39) | needs a first implementation |
| T18 | Verify a batch replayed by rule reproduces the recorded step's forward context faithfully, MoE routing included | unverified, and it gates D39 |
| T19 | Decide the artifact store's physical form: directory convention, manifest, or indexed store | no decision taken |
| T20 | Build the stated-node + standalone benchmark for invisible collectives | custom all-gather is the default path and is entirely invisible |
| T21 | Establish whether Phase 1c transfers across width (the TP2 test run) | it is the recipe's one unproven assumption |
| ~~T22~~ | ~~Analytical laws as their own design topic~~ — **done**, now `10` | — |
| T100 | Compare `measure memory`'s readings under ATOM's `--load_dummy` with a real load's on Qwen3.8-27B TP1 on MI308X; until they match, `measure memory` refuses `--load_dummy` (D38, Phase 2) | needs `measure memory` built |
| T103 | Stream the real record to the JIT sweep's client, so one round reads its own coverage and sends its missing points before the stop (D106) | deferred by #2640's design session; coverage is read after the stop meanwhile, and missing points are resent unchanged in the next round, three rounds at most |
| T104 | The paths the JIT sweep does not construct: preemption and recompute, error paths, and a PD cell's decode side and KV connector path, which one unscored run and the library warm. A build there is recorded with its batch, not a defect, and the record naming it is refused and taken again (D106) | the client cannot reach the first two without changing the deployment, and designs only the prefill side's steps |
| T105 | Sweep B on cells the PrefillDelayer does not coalesce: PP above 1 and rapidserve run A only and are not marked swept (D106) | ATOM builds no delayer there, so nothing makes a designed step's rows one batch |
| T115 | Plan a step grid prediction from its simulated runs (D37) on a new engine configuration | its host laws need a real run, since their key holds the engine config (D41); the loop is shown only on a calibrated configuration ([#2678](https://github.com/jgong5/ATOM/issues/2678)) |
| T116 | Plan a step grid prediction from its simulated runs (D37) on day zero, where the step roofline alone prices the first round's populations ([#2661](https://github.com/jgong5/ATOM/issues/2661)) | not yet run |
| T118 | `measure ops` on several cards (D39, D118): the same op points on two cards against one card's two passes, and how a split meets a DP-attention layout, whose cards are one bench, its ranks taking a barrier before each replay (`atom/compass/bench/ops.py::main`, `atom/compass/bench/ranks.py`). Until then one bench an invocation; `scripts/compass/README.md`'s op-level cell splits an op plan's decode rungs over one-card runs with no such check | a multi-rank plan keeps each rank's cold time beside rank 0's record (`atom/compass/bench/ranks.py::merge`), but no one has set them against one card's two passes |
| T119 | A prediction loop at the op level (D37): a stop rule, since its planner places points at the steps' own contexts and token counts (`atom/compass/bench/ops.py::plan`), so a round whose schedule moved places new ones. Until then the coverage check and one top-up | [#2678](https://github.com/jgong5/ATOM/issues/2678) ran the loop on the step grid only |
| T121 | A designed-load calibration of a 1P1D side (D119) | a 1P1D law stays on `deploy.sh record`, which `19` D113 defers until a 1P1D configuration needs one |
| T122 | A design group's tail: random forced acceptance ends its rows over several steps, at lower rungs (D119) | the mechanism check measures it |
| T123 | Whether a design group's fill, or a cut an earlier group or point filled, is still cached when its rows arrive: the cap leaves 20% of the pool for output, the lower cuts' own blocks and a model's other pools, the state checkpoints a cut resumes from among them (D119) | the mechanism check measures it; an evicted session's row prefills its whole context, an evicted cut fills again, and a cut whose checkpoint is gone fills from the one before, which the estimate does not count |
| T124 | Whether designed rows stand for real raggedness and row order, which alone move decode attention 1.77x (D39), and the calibration traces for the graded ones' contexts (D119) | the real record's fold and the validation check it |
| T125 | Decode two-batch overlap's validation: whether the law's features carry how a group's rows divide between a decode's halves, its scheduled order's first and second half, under `--enable-tbo all`, where a group with an empty rank, or a rank at two rows or fewer, runs unsplit on every rank; the half-long pair gives balanced and unbalanced halves (D119) | cell A runs no two-batch overlap, its records stating `enable_tbo` off ([#3720](https://github.com/jgong5/ATOM/issues/3720)), so it waits for a configuration the GPU plan runs under `--enable-tbo all`, another model's, since DeepSeek-V4.1 refuses decode TBO (`atom/models/deepseek_v41/config.py::validate_runtime_config`) |
| T126 | Pacing: whether a design group's rows enter decode together at the deployment's own pacing, a wide group's taking several prefill steps, else D106's parked decodes, and whether a trace design point's rows share one step at DP 1 under D106's pacing variables; and the frontend's spread of a group's rows: the event loop takes requests in one at a time, the chat template's render among it (`api_server.py::chat_completions`), on cell A some 2.2 s for a rung-64 top-level group's 256 bodies against their 40 decode steps of about 2 s, and a row under 100,000 characters, most of the bottom level's, encodes whole, up to 30 ms on the default executor's threads; rows that do not enter together decode until the last has (D119) | the mechanism check measures the first two; cell A's validation the frontend's spread, from each group's sends and acknowledgements; D106 measured the second at DP 8 only |
| T127 | The `rowmod` column at a rung the real record reaches with fewer steps than `ROWMOD_CELL_STEPS`: its rule holds every record to that many there, so the column stays off at that rung (D119) | a change to the rule of its own, if the validation shows the column missing where it pays |
| T128 | Designed prefill under DP: `moe_rows`, `serial_moe_rows`, and a DP drafter's prefill and `propose`, from the trace design points in #3509's gated record once it lands (D119) | `compass record --trace` refuses DP (#3010), so a DP law's prefill stays unfitted, as a DP list's is, but where `COMPASS_PREFILL_LABEL=record` labels untraced prefill steps from their windows, as cell A's laws are fitted |
| T129 | Designed load at a pipeline stage (D119), and a stage's decode (`09` D54.1) | ATOM builds no PrefillDelayer above one stage, so a designed point's rows are not one batch (T105); a stage's decode window holds the last send's stream hold and its wait for the stage before, which no label rule parts, so the fit cuts it; both wait until a configuration that decodes on a pipeline needs a law |
| T130 | `short_propose`: once a fit writes it, the trace design's output prefills under its `below` move it (D119) | the law reads it, and no fit writes it |
| T131 | DP dummy forwards and decodes beside a prefilling peer: no record fits either, so each stays at 0 s under its treatment or, under a law with an `eager` vector, on that vector (D119) | a dummy rank gives no row, and a step whose ranks fall to different vectors is cut; checking either needs DP prefill labels (T128) |
| T132 | Whether a trace design record's profiler trace reduces to busy lines at the points' length, else its points split over more records (D119) | the mechanism check measures it on one card, a validation at DP 1 (T133) at full width |
| T133 | The designed-load validation of what cell A does not run: decode at DP 1, the MTP and EAGLE3 drafters, and the trace design records' prefill (D119) | cell A runs at DP 4 with DSpark; it waits for a configuration the GPU plan runs, with real runs at its law's ATOM digest |
