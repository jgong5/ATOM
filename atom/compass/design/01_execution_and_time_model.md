# ATOM Compass — Design Topic 1: Execution and Time Model

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Branch:** `feature/atomcompass_new` (clean fork of upstream `main` at `0b4f1ddb`).

**Scope of this document.** The execution architecture: what a simulated run *is*,
which processes exist, how virtual time advances, and what contract the rest of the
system must honour. It does **not** cover the cost model, the memory model, model
capture, or the workload harness. Those are separate design topics.

---

## 0. Ground rules inherited from the task

- Compass replaces **only the forward pass**. ATOM's real scheduler, real block
  manager, real admission logic run unchanged. A simulated run makes the same
  scheduling decisions as a real one; it substitutes a predicted duration for the work.
- Simulated execution must not compute on a GPU and must not allocate GPU memory.
  Compute capability, memory size, bandwidth and interconnect are **configured**, not
  read from a device runtime.
- Prefer clean abstractions and refactoring over ad-hoc changes. Add only what is
  necessary.
- Final proof is paired simulation and real execution of cc-traces proper.

---

## Relationship to the prior PoC branches

Not a design point — a one-paragraph statement of provenance, recorded here so the
evidence cited throughout this document has an origin. The fuller account is
**Development history** in `README.md`.

Two prior attempts exist (`feature/atomcompass_take2`, `feature/atomcompass`; the
history is linear, the second is the first plus ~350 commits). This design is **fresh,
referring to the prior work at the level of *design* only**. No code-port plan is
defined up front. Where a prior mechanism is the right answer it is described here on
its merits and re-derived; where it is not, it is not carried. Every quantitative claim
below that is attributed to "the prior work" or "a prior run" is a measurement taken on
one of those branches, on this hardware.

### What the prior work established that this design relies on

These are measurements, not opinions, and they are load-bearing below:

- **The seam needs no ATOM change.** `Config.runner_qualname` (`atom/config.py:1595`),
  consumed at `engine_core.py:129` and `async_proc.py:166`. Two in-tree precedents:
  `RLHFModelRunner` (`atom/rollout/async_engine.py:26-32`) and `RapidServeModelRunner`
  (`config.py:1729-1736`).
- **Rank-0 single-sourcing of the clock is correct for symmetric TP.** TP=2 over 1727
  steps: per-step rank difference median 0.03%, worst 0.82%, rank 1 slower on 51% of
  steps. TP=4 over 2295 steps: rank totals within ±0.02%; charging every step to its
  **slowest** rank adds **0.06%** to the total.
- **Speedup has two independent sources, and only one of them survives saturation.**
  (i) *Not doing the compute* — the forward pass evaluates a cost model instead of
  running kernels, so every step is cheaper by construction, at every load. (ii) *The
  idle jump* — when every LP is blocked, the clock leaps to the next event instead of
  sleeping through real seconds. Measured: 64 requests Poisson 8/s, 9,426 ms real vs
  639 ms simulated (14.8x); cc-traces 20 requests 27B TP=4, 309 s vs 3 s (~103x). Under
  saturation, where there is no idle to skip, only (i) is left — and it was not enough:
  **122 s vs 36 s, i.e. 0.30x, slower than the system it simulates.**

  **Why source (i) is worth so much less than it sounds, which is the whole explanation
  of that 0.30x.** Removing the kernels does not remove the step. A real decode step runs
  its kernels on the device *while the host is busy* with `prepare_inputs`, block-table
  marshalling, sampling setup and the scheduler's own bookkeeping — the two overlap, and
  on small shapes the host side is the longer of the two. Doc `10` D64 measures exactly
  this from the other direction: at 794 tokens the device window is 36.745 ms of which
  **23.360 ms (63.6%) is idle**, waiting on the host. So deleting the kernel time deletes
  the *shorter* of two overlapped costs on precisely the steps a decode-heavy workload is
  made of, and the host work that remains is not replaced — Compass still runs all of it,
  plus the cost model on top.

  Two consequences: the ≥5x target is a statement about **the arrival process**, not only
  about the cost backend; and the thing worth optimising for speed is the per-step host
  path, which is what doc `08`'s "cost of a simulated step" work measures.
- **Aggregate cost accuracy does not bound schedule accuracy when the scheduler has a
  discontinuity.** The prior run was within 1.0% on prefill seconds, 1.1% on decode,
  1.0% on run length — and **90% wrong on median TTFT**, because TTFT was decided by two
  comparisons with 1.5 s and 1.2 s of slack. A counterfactual at x0.97 on the prefill
  price split a 63-chunk streak back into the real run's exact 42+7+15.
  **Consequence for this design: schedule agreement must be a first-class, separately
  reported result, never inferred from latency error.**
- **The real machine is not on the knife edge; only the simulator is.** All four real
  repeats produced the identical streak structure (36, 6, 42, 7, 15) breaking at the
  same two places, margins varying by at most 0.1 s.

---

## D1. Process and thread model

### Problem

ATOM is a multi-process system. A minimal TP1/DP1 serving run is **3 processes**: one
API server, one `EngineCore`, one `ModelRunner` worker. Wider configurations multiply:
`dp_size x pp_size` engine cores, each with `tp_world_size x pcp_size` workers.

A discrete-event simulator needs a coherent notion of "now". The question is whether to
keep that topology or collapse it.

Verified topology (file:line):

| Link | Mechanism |
|---|---|
| API server <-> CoreManager | in-process |
| CoreManager -> EngineCore | ZMQ ROUTER/DEALER (`engine_core_mgr.py:441-449`, `engine_core.py:516-578`) |
| EngineCore -> CoreManager | ZMQ PUSH/PULL (`engine_core.py:579-634`) |
| EngineCore -> workers | `aiter.dist.shm_broadcast.MessageQueue`, POSIX shm, 16 MiB chunks (`async_proc.py:28,288-291,425-434`) |
| worker rank0 -> EngineCore | ZMQ PUSH/PULL (`async_proc.py:310-311,391-406`) |
| every worker -> EngineCore | per-rank ZMQ PUSH/PULL for KV status (`async_proc.py:302-306,408-423`) |
| cross-DP | `torch.distributed` Gloo CPU group (`engine_core.py:664-686,751-815`) |
| PP stage <-> stage | ZMQ metadata + NCCL tensors (`atom/distributed/pp_comm.py`, `pp_transport.py`) |

`multiprocessing` start method is forced to `spawn` (`engine_core_mgr.py:337-338`,
`:1423-1424`, `atom/utils/__init__.py:192-194`). **There is no in-process / single-process
mode**: `LLMEngine.__init__` unconditionally constructs a `CoreManager`
(`llm_engine.py:139-142`) and `EngineCore.__init__` unconditionally constructs an
`AsyncIOProcManager` (`engine_core.py:125`). Offline examples still go through the full
stack.

### Options

**A. Single process, virtual-time asyncio loop.** Replace `CoreManager` and
`AsyncIOProcManager` with in-process equivalents; every engine core, worker and the
traffic source become coroutines on one loop whose `time()` is virtual. IPC becomes
in-process queues with modelled latency.

- *Pros:* one clock, no races, no distributed time, trivially reproducible. Scales to
  PP/DP/EP/PD because "two nodes" is two coroutine groups. Fastest wall clock.
- *Cons:* large, invasive refactor of ATOM's process layer. Diverges from the design
  principle of reusing ATOM's structure. A simulated run would no longer be "the same
  program with a different forward".

**B. Keep ATOM's multi-process topology; give it coordinated virtual time.**

- *Pros:* minimal intrusion. The simulated deployment *is* the real deployment. Every
  ATOM change is additive (a clock module, clock-read substitutions, status
  annotations, config flags).
- *Cons:* requires a time-coordination protocol between processes, and causality
  violations are silent.

**C. Hybrid:** single-process for milestones 1-3, multi-process later.

- *Cons:* defers the hard problem while designing the kernel around assumptions that
  break when it arrives.

### Decision

**Option B — keep ATOM's process and thread design.** No process is removed, no thread
is removed. The changes are:

1. an injectable clock module,
2. clock-read substitution at business-logic sites,
3. status annotation around cross-process blocking waits,
4. configuration to disable failure detectors,
5. a simulated KV connector and a simulated model runner.

Rationale: the design principle is explicit that ATOM's api server and scheduling
modules are reused and only the model layer and its dependencies are replaced. Option A
replaces the process layer, which is neither the model layer nor a dependency of it.
Option B's stated cost — a coordination protocol — is addressed in D3 and turns out to
be small **because ATOM's own structure supplies most of the synchronization already**
(see D3's logical-process collapse).

### Mitigating option B's real cost: causality violations are silent

Option A's safety is structural — one thread cannot race itself. Option B has to *earn*
the same property, and a violation under option B produces a plausible number rather
than a crash. Accepting option B without a detector would mean accepting that class of
error into the acceptance evidence. So three always-on detectors, sized to be cheap
enough that none of them is a mode anyone can forget to enable.

**What class of error these detect — and it matters which.** All three catch
**implementation defects, not holes in the PDES algorithm.** The grant rule
`T_grant(i) = min over j≠i of (now[j] + L[j→i])` is conservative by construction: given
correct inputs it *cannot* produce a violation, which is the standard Chandy–Misra–Bryant
guarantee and is not in question here. What the detectors watch for is the inputs being
wrong — a lookahead constant declared too large, a call site nobody annotated, a clock
read nobody substituted. Those are bugs in *our* code and configuration.

This distinction is worth stating plainly because the two cases warrant opposite
responses. If a detector fires, the fix is local: correct the constant, add the
annotation, substitute the clock read. **If one fired and none of those explained it, the
mechanism itself would be in question** — and that would be a much larger problem than a
detector, because it would mean the conservative rule is not conservative on this
topology. Nothing observed so far suggests that, and the grant rule is standard rather
than invented here. But the detectors are also the only thing that would *tell us*, which
is a second reason to have them.

**Where violations actually come from.** There are exactly three ways the inputs go
wrong, and each maps to one detector.

| Failure | What it looks like | Detector |
|---|---|---|
| **Declared lookahead is larger than reality** — an LP claims it cannot affect another for 50 µs and then does it in 10 | Receiver has already advanced past the send time. Silent; the message lands "in the past" | **(1) Straggler check** |
| **A wait is not annotated** — an LP blocks for real while the CA believes it is running | Either a deadlock (loud) or, if a timeout rescues it, an LP that consumed no virtual time while real time passed. Silent | **(2) Annotation-coverage audit** |
| **A clock read was missed** — business logic still calls `time.monotonic()` | Two timestamps on one timeline disagree; durations mix scales. Silent | **(3) Clock-source audit** |

**(1) Straggler check — receive side, always on, one comparison.** Every cross-LP
message already passes through a small number of send/recv wrappers (the ZMQ hops of
D4 category B). Each carries the sender's virtual send time `t_s`. On receipt the LP
asserts `t_s >= now_self`. A violation means this LP has already simulated past the
moment the message was sent, i.e. the local-causality constraint is broken. The check
costs one float comparison per message and is the direct test of the property the whole
protocol exists to provide. On failure: record `(sender, receiver, t_s, now_self,
declared lookahead)` and **fail the run** — not a warning, because a straggler
invalidates every number downstream of it.

This is also what makes a wrong lookahead *findable*: the report names the pair, so
raising `L[j→i]` to the observed violation plus margin is a mechanical fix.

**(2) Annotation-coverage audit — a watchdog on "running" LPs.** A background thread
per LP samples its own state. If an LP has been `declare_running()` for more than a
wall-clock threshold (~200 ms is far above any real simulated step and far below any
real blocking wait) *without* its virtual clock advancing, it is blocked on something
nobody annotated. Log the LP, the stack, and continue — this one is a warning rather
than a failure, because it costs correctness only when it also produces a straggler,
and detector (1) catches that. Its value is that it names the missing annotation
**during development**, when D4's ~55-site audit is still being worked through, rather
than leaving a category-B site to be discovered by a wrong result.

**(3) Clock-source audit — static, run in CI.** D4's categorisation rests on a
by-hand audit of clock-read sites. A grep-level lint over the simulated-path modules
that flags `time.time`, `time.monotonic`, `time.perf_counter`, `datetime.now` and
`asyncio.sleep` outside an allow-list keeps that audit from rotting as ATOM's main
branch moves. The allow-list is the set doc `11` D72 establishes as deliberately real
(metrics push cadence, transport). Anything new lands as a CI failure on the day it is
added, not at validation time.

**What this costs.** One comparison per cross-LP message, one sampling thread per LP,
one CI lint. None of it is on the per-step path. **What it buys:** the statement "no
causality violation occurred" becomes a reported result of every run rather than an
assumption, which is what doc `08` needs in order to treat a simulated number as
evidence at all.

**What it does not cover.** A lookahead that is wrong but *never exercised* by the
workload is not detected — the run is correct, and a different workload may not be.
Recorded as **T47**.

### Open issues

- Simulation wall-clock cost is higher than Option A's. The >=5x target is stated as
  negotiable with a bottom line of "faster than real runs". Under saturation the prior
  design was 0.30x. This must be measured early, not assumed.
- `torch.cuda.set_device` (in `model_runner.py::ModelRunner._setup_device_and_distributed`) and
  `torch.cuda.mem_get_info` (in `model_runner.py::ModelRunner._read_device_memory`) are the two
  hard GPU dependencies a simulated runner must not inherit.

---

## D2. What "PD disaggregation on two nodes" means

### Problem

ATOM contains **three unrelated mechanisms** that are all called prefill/decode
disaggregation. Conflating them invalidates any performance model.

| Name | Scope | Processes | How KV moves |
|---|---|---|---|
| **RapidServe** (`--enable-rapidserve`) | one node, one GPU set | 2 `EngineCore` procs on the **same** devices | **Never moves.** Decode owns the KV tensor; prefill imported it by CUDA IPC and writes directly into decode's buffer. Only the sampled first token crosses ZMQ. `PrefillScheduler.block_manager = None` (`scheduler.py:3141`). |
| **True PD disagg** (`--kv-transfer-config`) | two node groups | independent full servers | RDMA — MoRI-IO (read/pull) or Mooncake (write/push) |
| **Atomesh** (`atom/mesh/`) | fleet router | Rust binary, **or** `libmesh.so` in-process via PyO3 | Never touches KV. HTTP relay only. |

### Decision

**True PD disaggregation**, with two qualifications from the project owner:

1. The two "nodes" are **two containers on one physical node**. This removes the real
   network, lets both sides share a filesystem, and lets the time-coordination channel
   be local ZMQ/shm rather than a network protocol.
2. **KV transfer is simulated**, not carried by real RDMA. See D6.

### Consequences

- Atomesh is in the loop. Its treatment is D7.
- ATOM's CI-benchmarked multi-node PD path exists (`.github/scripts/atomesh/pd_server_atom.sh`),
  so real-run baselines for pairing are obtainable.
- The ATOM relay is **strictly sequential and blocking**
  (`http_pd_router.rs:969-1192`): POST prefill, await the full JSON, extract
  `kv_transfer_params` (hard error if absent), enrich the decode body, POST decode.
  Unlike the SGLang path (`tokio::join!` on both, `:1463`) and the vLLM path (detached
  `tokio::spawn`, `:732`). **This sequentiality is a gift: the prefill->decode causal
  edge is explicit, one-way, and per-request.**

### Open issues

- Both containers on one physical node contend for the same CPUs. This does not affect
  fidelity (no GPU work) but it does affect simulation wall time, and it means the
  simulator's own CPU cost is on the critical path twice.
- `MORI_SHMEM_MODE=ISOLATION` is required for DP+TP so MoRI (MoE all-to-all) and MoRI-IO
  (KV) heaps stay apart. With a simulated connector this may become moot; confirm.

---

## D3. Virtual time coordination protocol

### Problem

With ATOM's topology kept (D1) and two engine deployments (D2), more than one process
advances time. The local-causality constraint must hold:

> A logical process may not advance its clock past the timestamp of any message it has
> not yet received.

Violating it does not crash. It produces a plausible latency table. Every historically
expensive failure in this project has had exactly that shape.

### Survey of what the field does

Verified against source, not only papers:

- **SimAI** is the only true parallel DES in the LLM/ML simulator space. It achieves it
  by vendoring **UNISON** (EuroSys'24) — conservative, lookahead-based, barrier-windowed
  YAWNS. The safe-window computation, verbatim from
  `ns-3-alibabacloud/simulation/src/mtp/model/logical-process.cc:155`:
  `grantedTime = Min(MtpInterface::GetSmallestTime() + m_lookAhead, MtpInterface::GetNextPublicTime())`.
  Its auto-partitioner cuts p2p links whose delay >= the median — i.e. **it partitions on
  where the lookahead is**.
- **ASTRA-sim** avoids the problem entirely: it has no second clock. `Sys::boostedTick()`
  is a *read* of `comm_NI->sim_get_time()`, and every rank reads rank 0's network
  interface. The system layer never advances time; it only registers callbacks.
- **LLMServingSim** is more extreme still: its frontend parses ASTRA-sim's cycle count
  out of stdout and adopts it.
- **LLMCompass** is not a DES at all — analytical tile-recurrence plus a brute-force
  mapper; its only cycle-level component is memoized into a shipped CSV.
- **No optimism, no rollback, no GVT anywhere in the space.** This is the right call
  here: rolling back a live Python `Scheduler` + `BlockManager` + prefix-cache index is
  not cheap and probably not sound.

Summary: **every system that parallelizes uses conservative barrier windows; every
system that avoids the problem collapses to one clock.**

### The move that makes this small: collapse already-barriered groups into one LP

A *logical process* (LP) is the unit the protocol coordinates. It need not be an OS
process. ATOM already contains hardware barriers that make several processes behave as
one LP:

| ATOM group | OS processes | LPs | Why |
|---|---|---|---|
| TP group | 1 EngineCore + N workers | **1** | Workers are slaved by a blocking RPC (`async_proc.py:431`) and hold no clock. Rank-0 authority validated at 0.06% (D0). |
| DP group | N EngineCores | **1** | Already `all_reduce`s every step for lockstep (`engine_core.py:751-781`) and runs `dummy_execution` on idle ranks (`:748-749`). Carry `max(step_seconds)` on the collective that already runs. |
| Prefill container | — | **1** | |
| Decode container | — | **1** | |
| Traffic source | — | **1** | |
| PP stages | N EngineCores | **N** | The only group the collapse does not cover. |

LP counts for the milestones:

| Milestone | LPs |
|---|---|
| M1-M3 (single deployment, TP1/2/4) | **2** — traffic, engine |
| M4 (two containers, PD disagg) | **3** — traffic, prefill, decode |
| M5-M6 (Kimi-K3, TP8) | **3** |
| M7 (PP) | +1 per stage |

So a TP4 x DP2 deployment is **one** LP, not eight.

### Options for the protocol

**A. Central Clock Authority.** One small process owns global virtual time; LPs request
grants.

- *Pros:* simplest to reason about and to audit. Trivially reproducible. Makes causality
  violations detectable at a single choke point. Degenerates gracefully — with one LP it
  is the prior design's local clock, with zero lookahead it is a single global event loop.
- *Cons:* one RPC round trip per advance; a new component; a single point of failure
  (which is acceptable — its failure is loud).

**B. Conservative barrier window (SimAI/UNISON style).** No central process; each LP
computes its own safe window from a distributed LBTS reduction plus a lookahead matrix.

- *Pros:* what the only true PDES in this field does. No bottleneck.
- *Cons:* more machinery; the LBTS reduction still needs a collective, so at 3-10 LPs it
  buys nothing over A.

**C. Fixed-quantum lockstep.** Barrier every fixed quantum of virtual time.

- *Cons:* the quantum must be smaller than the shortest modelled delay or fidelity is
  lost; at 100 us quanta a 300 s cc-traces run is 3M barriers.

### Decision

**Option A — a central Clock Authority (CA).**

#### State

Per LP: `now[i]`, `next[i]` (earliest future event this LP knows of, or `+inf`),
`status[i] in {running, granted, blocked-on-message}`. Plus a static lookahead matrix
`L[j->i]`.

#### Grant rule

```
T_grant(i) = min over all j != i of ( now[j] + L[j->i] )
LP i advances to min( T_grant(i), next[i] )
```

**It is `now[j]`, not `next[j]`.** This matters and the naive version looks correct:

- With `now[j]`, no LP is ever granted past the globally-earliest LP's current time, so
  no event that LP generates can land in anyone's past. Correct at **any** lookahead,
  including zero.
- With `next[j]`, LP A can be behind LP B (A had the minimum, B ran ahead earlier), and
  an event A generates at `A.now + 0` lands in `[A.now, B.now)` — B's past.

At zero lookahead the rule degenerates to `T_grant(i) = min_j now[j]`, i.e. a single
global event loop across processes: correct, serialized, no parallelism. **That is
acceptable here, because the speedup comes from skipping idle, not from running LPs
concurrently in wall time.**

#### Invariants

- **Safety (asserted, always on):** an event LP *i* schedules on LP *j* must satisfy
  `ts >= now[i] + L[i->j]`. A backdated event **aborts the run with a full LP state
  dump**. It must not be a warning and must not be behind a flag.
- **Deadlock:** all LPs blocked and none holding a finite `next` -> abort loudly with the
  LP table. Never a quiet timeout. (The prior arrival barrier's 120 s timeout released
  on a run that was invalid, the client printed "0 failed", and a day's conclusions came
  off it.)
- **Determinism:** ties at equal timestamps broken by LP id. Without this the
  126-vs-189-decode-steps nondeterminism returns.

#### Lookahead sources

Every one is physical and configurable, which is also a project requirement
(interconnect must be configurable and not read from a device):

| Link | Lookahead |
|---|---|
| traffic -> engine | modelled admission delay. Prior work measured 13.7 ms end-to-end, worth ~4 points of TTFT. Path-specific: 13 ms offline batch, 9 ms serving. |
| prefill -> decode | Atomesh relay + simulated KV transfer. Millisecond scale. Comfortable. |
| PP stage -> stage | modelled NCCL send/recv of intermediate tensors. Microsecond scale. The only tight one. |

**Declaring a lookahead floor on every inter-LP link is a design commitment, not a
constant to tune later.** Zero lookahead is correct under the grant rule above but
serializes everything.

### Sizing

PP8, 27B, ~10 ms step, 300 s modelled run: ~30k steps x 8 stages ~= **240k grants**. At
~50 us per local IPC round trip, ~12 s of overhead against a 300 s real run — still
~25x. For M1-M4 with 2-3 LPs the grant traffic is negligible.

**PP is therefore an efficiency concern, not a correctness concern.** It is also
single-node only (every PP address is ZMQ IPC, `engine_core_mgr.py:327-329`), and ATOM
*rejects* PP+DP (`:298-300`) and multi-node DP+PP (`:272-279`). Defer it on scope
grounds.

### Open issues

- Grant RPC latency has not been measured on this box. The 50 us figure is an estimate.
- Whether the CA should be a process or a thread in the API-server process is open. A
  separate process is cleaner for the PD case where the API server is not obviously the
  right host.
- PP's microsecond lookahead will make the CA the bottleneck. When M7 arrives,
  re-evaluate option B for the PP sub-graph specifically.

---

## D3.1. Clock Authority deployment and scalability

### Problem

The CA is a logical service. Where it runs, and whether one instance can serve a
deployment that grows from one container to many nodes, is a separate decision from the
protocol itself.

### The finding that settles the sizing: LP count does not scale with hardware

A DP group is **one** LP however many ranks it holds, because it already
`all_reduce`s every step (`engine_core.py:751-781`). A TP group is **one** LP however
wide, because its workers are slaved by a blocking RPC (`async_proc.py:431`) and hold no
clock. Adding GPUs to either does not create a time domain.

Only three things create an LP:

1. a **PD role boundary** — prefill fleet vs decode fleet
2. a **PP stage**
3. an **independent replica** behind the router. `.github/scripts/atomesh/pd_server_atom.sh`
   deploys *xP* prefill servers and *yD* decode servers, each an independent deployment
   with its own `Scheduler`, none synchronizing with the others.

| Deployment | GPUs | LPs |
|---|---|---|
| M1-M3: TP4, one server | 4 | **2** |
| M4: TP4 prefill + TP4 decode, two containers | 8 | **3** |
| M5-M6: Kimi-K3 TP8, PD disagg | 16 | **3** |
| M7: Kimi-K3 TP8 + PP4 | 32 | **6** |
| 8 prefill + 8 decode replicas, each TP8 | 128 | **17** |
| ... the same with PP4 | 512 | **65** |

### Sizing against a measured workload

The prior 27B cc-traces run executed 106 prefill + 4,346 decode steps over 267 s of
modelled time — about **4,450 events per LP**.

| LPs | Grants per run | CA cost at 50 us (local IPC) | at 500 us (cross-node TCP) |
|---|---|---|---|
| 3 | 13k | 0.7 s | 7 s |
| 17 | 76k | 4 s | 38 s |
| 65 | 289k | 14 s | 145 s |

Against the **cost model on the same steps**: previously measured at **4.3 ms per step**
with the bound allocation carried in the cache key (2.3 ms with a shape-only key, but
that key is unsound — a second valid allocation for the same shape moves 64 of 2,439
operator signatures; 41.7 ms with no cache at all). 4,450 steps x 4.3 ms = **~19 s per
LP, running in parallel across LPs.**

**A grant is therefore ~1% of the per-step cost at local IPC and ~10% cross-node.** The
simulator's own pricing dominates by two orders of magnitude. Optimising the time
protocol before the cost model would be optimising the wrong thing.

Throughput headroom: a Python ZMQ ROUTER sustains roughly 50-200k msg/s; the largest case
above is ~10k/s. The CA's own work per grant is a `min` over at most 65 entries, and RTTs
pipeline across LPs.

### Options

**A. Single CA with a hierarchy-ready interface.** One instance. Deployed as a thread in
the API-server process for the single-node case, or as a standalone process addressed by
`host:port` for multi-node — the same shape as `--data-parallel-master-ip`. The LP-facing
interface is identical in both.

- *Pros:* zero deployment cost single-node; one address to configure multi-node; one
  choke point that knows global state, which is where the safety assertion, the deadlock
  dump, and a "who is holding up the simulation" query naturally live. A node-CA can
  later slot between an LP and a root CA without either side changing, because a node-CA
  runs the same algorithm as the root — the CA is recursive.
- *Cons:* a single point of failure. Acceptable: its failure is a hang, which is loud,
  and a simulation is a batch job rather than a service.

**B. Hierarchical CA built now.** Node-local CA per physical node, root CA over them.

- *Pros:* cross-node grant traffic drops from O(LPs) to O(nodes). The correct end state
  beyond ~100 LPs across many nodes.
- *Cons:* two levels to debug and a global-min protocol to get right, at a scale no
  milestone reaches. The sizing above says this is premature.

**C. No CA — carry time on a Compass-owned `torch.distributed` collective.**
`all_reduce(MIN)` over a Gloo group built with ATOM's existing
`stateless_init_torch_distributed_process_group` (`utils/distributed/utils.py:75-130`).

- *Pros:* no new process; scales exactly as `torch.distributed` scales; the collective
  **is** the barrier, so the safety property is structural rather than asserted.
- *Cons, and the first is probably disqualifying:* a Gloo `all_reduce` requires every
  rank to call it, and **a blocked LP cannot**. An LP parked in `zmq.recv` awaiting a
  peer's message would need a separate time-service thread sharing mutable `now`/`next`
  with its main thread under a lock — a concurrency hazard placed exactly where silent
  failure is most expensive. It also gives up the single choke point for the safety
  assertion and the deadlock dump.

**D. Single CA with no hierarchy provision.** Least code now; if a hierarchy is ever
needed the LP-facing interface changes, which touches every LP.

### Decision

**Option A — a single CA with a hierarchy-ready interface.**

Requirements that follow, and they are cheap only if honoured from the start:

1. The LP-facing interface must not name the CA's location or its level. An LP knows an
   endpoint, nothing more.
2. The CA's own interface to *its* peers must be the same as its interface to LPs, so a
   node-CA is a CA whose "LPs" are other CAs.
3. The lookahead matrix must be addressable by LP identity, not by index, so inserting a
   level does not renumber anything.
4. Partition guidance for a future hierarchy is already implied by the topology, and
   matches what SimAI's auto-partitioner does (it cuts p2p links whose delay is at or
   above the median): **PP stages have microsecond lookahead and must stay under one
   node-local CA; PD role boundaries have millisecond lookahead and are the cheap links
   to cross a node.** The hardware already lays out that way.

### Where the CA runs: both, selected by one flag

Settled: **two deployment forms of one implementation, not two implementations.**

| Form | When | How it is reached |
|---|---|---|
| **Co-hosted in the API-server process** (default) | single-container runs — M1 through M3, M5, and every calibration or debug run | the LP's endpoint resolves to an in-process transport; no socket, no extra process to start or reap |
| **Standalone CA server** | multi-container and multi-node runs — M4, M6, and anything with an Atomesh router between roles | `--compass-clock-endpoint tcp://host:port`; the CA is its own process, started before the engines |

This is cheap precisely because requirement 1 above already forbids the LP-facing
interface from naming the CA's location. The two forms differ only in which transport
the endpoint resolves to. The in-process form is *not* a shortcut that skips the
protocol — the same grant rule, the same lookahead matrix, the same straggler stamps —
so a bug found in one form is a bug in the other, and the cheap single-container runs
are a real test of the expensive multi-container path.

Default co-hosted rather than always-standalone because the overwhelming majority of
runs are single-container, and a standalone CA there is one more process to start,
supervise and leak. Standalone for M4/M6 because neither container is obviously the
right host and making one of them the clock owner would give the two roles asymmetric
failure behaviour that the real deployment does not have.

**What must be true for this to stay one implementation:** the co-hosted form must not
acquire an in-process fast path that bypasses the message stamps detector (1) above
depends on. If grant traffic ever stops carrying virtual send times in-process, the
single-container runs stop testing the property they are supposed to be testing.

### Open issues

- Grant RPC latency has not been measured on this hardware. The 50 us figure is an
  estimate and should be measured before it is quoted. Note this matters only for the
  standalone form; co-hosted grants are function calls.
- The CA is the only component that knows every LP's virtual time. It should therefore
  own the global timeline log and the deadlock dump. That makes it an observability
  component as well as a coordination one, which is a benefit but also means its output
  format is part of the acceptance evidence and should be designed, not improvised.

---


---

## D3.4. Determinism: what a simulated run must reproduce, and what it need not

### Problem

`08` T26 asks for bit-reproducibility as a test, and the whole paired-comparison protocol
leans on it: a simulated side that disagrees with *itself* between runs cannot be compared
with a real side at all. But a multi-process simulator has several genuine sources of
non-determinism, and demanding that all of them vanish would mean rebuilding the process
model that D1 deliberately kept.

### The distinction that makes this tractable

> **Wall-clock interleaving may vary between runs. The sequence of
> `(LP, virtual time, event)` may not.**

Nothing downstream reads wall time. Every graded number, every step-table row and every
artifact is a function of virtual time, so reproducibility is a property of the *virtual*
schedule and of nothing else. Two runs may take different real durations, issue grants in
different real orders, and schedule their threads differently, and still be identical
runs.

### The sources, and what each needs

| Source | Deterministic? | What makes it so |
|---|---|---|
| **Grant order at the CA** when two LPs are eligible at the same virtual time | **not by default** — real arrival order decides | **Tie-break by LP identity, never by arrival order.** The CA holds a total order over LP ids and grants in it. This is the single most important rule here, and it costs one comparison. |
| **Cost model output** | yes, if the backend is pure | no iteration over a `dict` or `set` whose order depends on insertion or on object identity; a fixed summation order over IR nodes, since float addition is not associative |
| **Speculative acceptance draw** | already handled | `14` D83: one host draw seeded from the step counter, not `world_size` draws that must agree |
| **`dict` iteration** over request or block ids | **yes** | insertion-ordered since Python 3.7, and insertion order is the schedule's order, which is itself deterministic |
| **`set` iteration** | **NO — and this is the trap** | see below |
| **Thread scheduling inside an LP** | irrelevant | by D4, waits inside one LP are invisible to modelled time |
| **Deliberately-real clock reads** (metrics push cadence, transport) | irrelevant | `11` D72 — they affect when a scrape lands, not what it says |

### `set` is the trap, and it is worse than "unordered"

`dict` preserves insertion order; **`set` does not, and its order is not even stable
between processes.** Iteration order follows hash values and the insertion history that
produced the table's layout. For the ids Compass actually holds:

- **string ids** — request ids, block hashes, LP names — are hashed with
  **`PYTHONHASHSEED` randomisation**, on by default. So a `set` of request ids iterates
  in a *different order in every process*, and two runs of one configuration diverge with
  no code change at all.
- **integer ids** hash to themselves, so a small-int set looks stable — which is worse,
  because it works in testing and then reorders the moment the values spread out or the
  table resizes.

This is the one source in the table that produces a divergence with nothing to point at:
no error, no warning, and a diff between two runs of identical code.

**Rule: no `set` iteration on the simulated path.** Where set *semantics* are wanted, use
a `dict` with `None` values as an ordered set, or sort explicitly at the point of
iteration. Membership tests against a `set` are fine — it is only iteration that leaks
order.

This is mechanically checkable, so it joins the CI clock-source lint of D3.2 as a second
rule in the same check rather than a convention anyone has to remember.

### The rule, and the test

**Rule:** the CA's grant order is a total order over LP identity; the cost backend is a
pure function of its `batch_view`; and no simulated-path code iterates a `set` or any
container whose order depends on object identity.

**Test** (this is `08` T26, now with a mechanism): run the same configuration twice and
diff the step tables byte for byte. It is CPU-only, it needs no GPU, and it belongs in CI
from the first stage that produces a step table — because the failure it catches is one
that gets *much* harder to localise once several tracks are contributing.

**What the test does not cover:** a run that is reproducible and wrong in the same way
twice. Determinism is a precondition for comparison, not evidence of fidelity.

---

## D3.5. Observability: the Clock Authority owns the timeline

### Problem

D3.1 records that the CA is the only component that knows every LP's virtual time, and
that it should therefore own the global timeline log and the deadlock dump — and that
*"its output format is part of the acceptance evidence and should be designed, not
improvised."* This is that design. It is deliberately small.

Note the boundary with `11`: that topic covers ATOM's **Prometheus engine metrics**, which
describe the *simulated system*. This covers the **simulator itself** — whether the run was
valid, not what the modelled engine did. Different consumers, different lifetimes, no
overlap.

### Three outputs, and nothing else

**1. The timeline log.** One append-only record per granted advance:

```
  lp_id . virtual_time_from . virtual_time_to . event . detail
```

Written by the CA because it is the only place with a consistent global view, and the
only place where the ordering is authoritative. It is what makes a causality report
actionable: when the straggler check (D3.2) fails, the log already contains both LPs'
histories up to the violation.

**2. The deadlock dump.** When no LP can be granted, the CA dumps, for every LP: its
current virtual time, its declared state (`running` / `blocked`), what it declared itself
blocked on, and the lookahead row that produced its grant bound. A deadlock is the *loud*
failure D4 deliberately engineered for — this is what makes it diagnosable rather than
merely noisy.

**3. The run summary**, written once at the end and carried in the run artifact:

| Field | Why |
|---|---|
| grants issued, per LP | the protocol's own cost; the number that says whether PP degree is affordable |
| wall seconds vs simulated seconds | the speed result (`08`), and the only place the ≥5x target is measured |
| lazy traces: count and wall seconds | `02` — they consume real time inside a simulated run and must not silently degrade the speed result |
| causality detector state | straggler count (must be 0), watchdog warnings, clock-lint status |
| refusals: count, fraction of steps, **fraction of predicted seconds**, distinct reasons | `08` D50.1's admissibility gate reads this |

### Two rules

1. **The timeline log is off by default and costs nothing when off.** It is a debugging
   and evidence artifact, not a per-step tax; at millions of grants it would dominate a
   fast run. The *summary* is always written — it is small and it is what the acceptance
   gate reads.
2. **The summary is part of the run artifact, not a log line.** `13` D81's rule applies:
   recorded by value, so a result can be audited without the machine that produced it.

### Open issue

- The timeline log's volume at PP degree > 1 is unestimated. Grants scale with PP stages
  and with the microsecond lookahead between them, so the log could be very large exactly
  where it is most wanted. Recorded as **T70**.
## D4. The interception contract: which waits must be touched

### Problem

ATOM's serving path contains 212 distinct synchronization points — "roughly 55" until
they were counted, see the contract below: blocking ZMQ
recvs, bounded pollers, queue gets with timeouts, Gloo and NCCL collectives,
`multiprocessing` barriers and joins, busy-waits, and literal sleeps. "Intercept every
blocking call" is the obvious reading of what a virtual clock demands. It is also the
reading that turns this project into reimplementing a scheduler on top of the OS
scheduler.

### The contract

A wait matters to virtual time **only if its duration is observable in the simulated
result.** That yields four categories.

**The counts below are measured, not estimated.** The classified list is
`atom/compass/audit/sync_sites.json`, produced by the scanner beside it and held to the
tree by `tests/compass/test_sync_inventory.py` — which parses **this table too**, so it
cannot drift from the rows. The estimates this table carried until 2026-09-21 are kept
in the last column so the diff stays visible.

| Cat. | What it is | What you do | Count | Est. |
|---|---|---|---|---|
| A | Wait whose duration **is** modelled time, or a reading the result reports | **Rewrite.** Do not wait — `advance_to(now + d)` and continue. | **23** | ~6 |
| B | Wait for a message another LP will send | **Annotate only.** `declare_blocked()` / `declare_running()` around the existing call. Leave the call itself alone. | **36** | ~10 |
| C1 | Timeout that is a failure detector | **Disable or raise.** No virtual semantics needed. | **11** | ~20 |
| C2 | Timeout that is pacing | **Virtual timer.** Declare next event at `now+d`; keep a short real poll so the thread stays responsive. | **3** | ~3 |
| ignore | Wait *inside* one LP; startup; shutdown; OS-level | **Ignore.** Invisible to modelled time. | **137** | ~20 |
| undecided | Reading depends on a decision not yet made | **Decide before building on it.** | **2** | — |

**212 sites, not ~55**, over the serving-path directories named in the scanner's
`SCANNED_ROOTS` — 194 call sites plus 18 pinned points that are not a call. The
category totals move less than the grand total does: A, B and C2 are within a factor of
four of the estimates and C1 is *below* its estimate. Almost all of the growth is in
`ignore`, and counting each site once, in this order, the 134 ignored *call sites* are:
the module the runner seam replaces (21), the two real RDMA transfer backends the
simulated connector replaces (17), the send half of a cross-process message (24),
startup and shutdown (35), collectives inside the real forward pass (4), text scanners
whose loops park on nothing (4), and 29 others carrying their own reasons.
**Deliberately left alone: 137 of 212.**

Two rules settle the boundaries the estimate left implicit, and both are in the
artifact's own README rather than only here:

- **B against C1/C2 is the bound, not the peer.** An unbounded wait for another process
  is B; the same wait with a finite bound is C1 or C2 by what the bound is for. This is
  the rule the B list below already used ("no timeout argument").
- **B against ignore is the process.** A thread parked on a queue its *own* process
  fills does not make that process idle — its step loop is running — so declaring it
  blocked there is wrong rather than merely redundant.

#### Category A — the short list, as measured

Sixteen call sites and seven pinned points:

- the forward pass, at **five** call sites, not one: `engine_core.py:386` (the main
  step), `:992` and `:1264` (the two halves of RapidServe), `pp_engine_core.py:118` and
  `:379` (the PP head and a downstream stage)
- the idle rank's empty batch, `engine_core.py:749` — it consumes a step and is charged
  like any other
- KV-transfer completion (D6), at **three** call sites: `engine_core.py:488`,
  `pp_engine_core.py:252` and `:406`
- **tokenization**, at the five `run_in_executor` hand-offs `06` D33 names
  (`api_server.py:890`, `:1004`, `:1126`, `:1258`, `:1480`). D33 charges their service
  time from the machine spec, which makes them category A by this table's own
  definition; this list omitted them.
- **the idle jump, which is a real site in ATOM even though the name this list gave it
  is not.** `Scheduler._advance_to_next_arrival` does not exist here — but the loops it
  would have served do, and all three spin rather than wait when there is nothing to
  run, because `pull_and_process_input_queue` drains with `get_nowait` and nothing else
  in the turn blocks: `EngineCore.busy_loop` (`engine_core.py:315`),
  `DPEngineCoreProc.busy_loop` (`:696`), and `PPEngineCoreProc._head_busy_loop`
  (`pp_engine_core.py:67`). D8's measurement of the prior design — first real step is
  tick 1, first simulated step is tick **89,336** — is this loop counted. Simulated time
  has to jump to the next declared event at each of the three.
- `Scheduler._passed_delay` / `--scheduler-delay-factor` (`scheduler.py:3108`)
- `Scheduler._oldest_waiting_prefill_age_ms` feeding `PrefillDelayer` (`scheduler.py:1194`)
- the four stamps the result reports: `llm_engine.py:745`, `:777`, `scheduler.py:2688`
  and `:3364` — D5 already lists these as business logic, and they are carried in the
  inventory because a category applies to them

**One entry of the original list is not ATOM code at all.** The arrival gate (D8) is
something Compass adds: `_arrival_barrier_unmet`, `compass_workload_size` and
`ARRIVAL_BARRIER_TIMEOUT_S` return nothing on the whole `atom/` tree, so there is no
site to intercept, only a mechanism to build.
#### Category B — annotate, do not intercept

The important instance: **`call_func(..., wait_out=True)` is not intercepted.** It has
no timeout and every forward goes through it. Two lines of status annotation are enough,
because its duration was already charged by the *sender* when it called `advance_to`.
The receiver is merely idle in wall time; the CA routes grants elsewhere meanwhile.
The blocking call is `self.outputs_queue.get()` at **`async_proc.py:431`**; this section
said `:439`, which is inside the docstring of the *other* RPC entry point, and D1's own
`:425-434` was right.

Same treatment for `engine_core.py:544/546` (the engine's input thread, no timeout
argument — this section called it "the CoreManager poller", which is the peer, not the
thread), `engine_core_mgr.py:534/544/576/586`, and the RapidServe recvs, which are now
listed rather than deferred: `engine_core.py:946` (block assignments) and `:1204`
(prefill completion).

**Two groups this list never reached**, both invisible until the scanner's roots became
directories rather than a list of files:

- the four streaming endpoints' own collector reads — `serving_chat.py:289` and `:596`,
  `serving_completion.py:91` and `:252`. `api_server.py:2147` is the *same call* on the
  Anthropic endpoint and was listed; these two are the endpoints a replay drives.
- the nine out-of-band control commands in `engine_utility.py` (`:126`, `:147`, `:174`,
  `:194`, `:203`, `:213`, `:236`, `:252`, `:264`). Each parks the engine's **step loop**
  with no bound, and unlike the startup calls they can arrive at any point in a run.
  They wait for **rank 0 only**: `async_proc.py:310` gives the primary output address to
  rank 0 and `None` to every other rank, and `:332` builds one `outputs_queue` for it,
  so no other worker has a thread on that channel. The form that does wait for all of
  them is `call_func_with_aggregation`, which has a queue per rank.

**On the send side, "it cannot park" is true of most of them and has to be said per
socket, not once.** Of the 25 sends, the ones built by `make_zmq_socket`
(`atom/utils/__init__.py:541-543`) carry `SNDHWM=0` and provably cannot park. The rest
are bare `ctx.socket(...)` and keep ZeroMQ's default thousand-message bound: the three
`pp_transport.py` sends, the disagg bootstrap sends, and — the two that matter at
runtime — `engine_core.py:1014` on the **prefill step loop** and `:1229` on the decode
one, whose sockets are created bare at `:925` and `:1182`. Their protocol is one message
per sequence and the peer drains it every tick, so a thousand-message backlog is not
reachable in a run; the rows say that rather than claiming the bound does not exist.

**Two of the sites named here are classified otherwise in the inventory, with reasons:**

- `engine_core.py:587` — the line is `self.output_queue.get()` in the engine's *output*
  thread, fed by this same process's step loop. It is `ignore`: annotating it would
  declare the engine blocked while its step loop is running, which is the one direction
  the annotation must not be wrong in.
- `engine_core_mgr.py:534/544` — reached only from `CoreManager.__init__`, i.e. startup,
  which this table's last row says to ignore. It is kept as **B** because the peer is
  another process and the annotation is harmless; the contradiction between the two rows
  is recorded rather than resolved by fiat.

#### Why I4 is safe rather than merely careful

An LP that forgets to declare `blocked` leaves the CA waiting for a grant response that
never arrives. That is a **hang**, which is loud. The opposite failure — advancing past
a message — is silent. The protocol is deliberately arranged so that mistakes fall on
the loud side.

### The case that is not a "wait" problem at all

`_recv_prefill_done` (`engine_core.py:1204`) blocks in a background thread and calls
`DecodeScheduler.on_prefill_done`, which stamps `seq.first_token_time = time.time()`
(`scheduler.py:3364`). Intercepting the block fixes nothing. The rule is about who
stamps:

> **Background threads enqueue. The main loop stamps, at a granted time.**

The recv itself is category B and needs only the annotation. **But "ATOM's queueing is
already correct here" is half right, and the half that is wrong is worth stating.**
Measured at `7fc7a5ddd`:

- The `prefill_done` deque *is* correct: `schedule()` pops it in the step loop
  (`scheduler.py:3381-3383`), so promotion order follows message arrival order and
  nothing else.
- `on_prefill_done` does **more than enqueue**. In the background thread it also pops
  `prefill_waiting`, sets `num_cached_tokens`, appends the sampled first token, and
  stamps `first_token_time` — all before the deque append.
- It takes **no lock** while doing so, although `_prefill_lock` was created for exactly
  this (`scheduler.py:3302-3304`: *"Protects prefill_waiting and running: on_prefill_done
  is called from the _recv_prefill_done background thread"*). The lock's two users are
  `allocate_waiting` and `schedule`, both on the step-loop thread; the thread the comment
  names never takes it.
- The `_pending_assignments` half of the claim is about the **prefill** side
  (`engine_core.py:952/956/968` under `PrefillScheduler._pending_lock`) and is correct
  there. The two locks are different objects on different schedulers in different
  processes, and only the prefill one is used as its comment says.

So "only the stamping moves" understates it by three mutations. Moving the stamp alone
would leave the sequence's token and cached-token count set at an ungranted moment by a
thread holding no lock. Not a correctness defect in CPython today — the individual dict
and deque operations are atomic — but it is not the shape the rule describes, and a task
that assumes the enqueue is the only thing in that function will be surprised.

### Open issues

- ~~The counts above are estimates from a synchronization inventory, not from a completed
  pass over the code. The first implementation task should be to produce the exact
  classified list and check it in.~~ **Done, 2026-09-21.** `atom/compass/audit/`.
- ~~`engine_core.py:1156` is a literal `time.sleep(2)` ... it must be *checked*, not
  assumed.~~ **Checked, 2026-09-21, and both halves hold.** It is reached only from
  `DecodeEngineCore._post_model_load_hook`, and `DecodeEngineCore` is constructed only by
  `DisaggCoreManager`, which `LLMEngine.__init__` selects only under
  `config.enable_rapidserve` (`llm_engine.py:140-142`) — so "RapidServe-only" is exact.
  Its purpose is verified by the code around it: the sleep sits between importing
  decode's weight IPC handles and acknowledging to prefill, and prefill measures free
  VRAM for KV sizing only after that ACK. It must stay on the real clock. Nothing in ATOM
  couples it to whether weights are real, but `Config` keeps a simulated runner from it:
  `--enable-rapidserve` selects `RapidServeModelRunner` only when `runner_qualname` is
  still the default (`config.py:1727-1736`), and otherwise `Config` raises `ValueError`
  unless `runner_qualname` is in `RAPIDSERVE_RUNNERS` (`config.py:1737-1745`), before
  `LLMEngine.__init__` constructs any engine core. The cost, for a runner that list names,
  is two real seconds of startup and no modelled time, because
  it runs before READY and therefore before any arrival.
- The scanner's boundary is a list of directories, not a graph. It reads every `.py`
  file under `SCANNED_ROOTS`, so a module added beside a scanned one is caught; but a
  blocking call under one of the three directories `UNSCANNED_ROOTS` names is invisible
  to the test, and so is one reached through a call shape the scanner does not know. The
  offload connectors are the largest of the three exclusions. The earlier version of
  this scanner listed **files**, which declared its blind spot at a finer grain than it
  excluded: 24 candidate sites sat inside scanned directories and outside both lists,
  four of them the same streaming-collector read already classified B on another
  endpoint.

---

## D5. Failure detectors vs business logic

### Problem

Some time-dependent behaviour in ATOM changes what the scheduler does; some only decides
when to declare something broken. Treating them the same way breaks one or the other.

### The sorting rule

> **Does it change which batch gets scheduled?**
> Yes -> business logic -> must read the virtual clock.
> No -> failure detector -> disable it.

This rule is deliberately ATOM-aware. A general PDES framework cannot make this
distinction; a simulator built for ATOM can.

### Disable (configuration where possible, a new flag where not)

| Thing | Where | Today |
|---|---|---|
| Atomesh health check + circuit breaker | `--disable-health-check --disable-circuit-breaker` | **flags exist**; already used in ATOM's own CI launcher |
| Anthropic SSE ping, 5.0 s | `api_server.py:336`, clock at `:2175` | needs a flag |
| Stream silence warning, 30.0 s | `streaming_dispatch.py:27`, clock at `:176,182,203` | logging + a Prometheus gauge only; needs a flag |
| uvicorn `--timeout-keep-alive` 5 s | `api_server.py:2479-2489`, applied `:2648` | **flag exists**; raise it |
| Atomesh worker HTTP 30 s | `core/worker.rs:28-36` | **hardcoded in Rust**; one-line change |
| Atomesh `WorkerManager::REQUEST_TIMEOUT` 5 s | `worker_manager.rs:24` | **hardcoded in Rust**; one-line change |

Note both Rust constants only bite when virtual time runs **slower** than wall, which it
does under saturation (0.30x measured). They are not optional.

### Make a virtual timer, not disabled

- ~~`METRICS_PUSH_INTERVAL_S`~~ — **corrected by `11` D72: it stays on the real clock.** The
  simulated timeline comes from `11` D74 sampling, not from this cadence.
- `KV_IDLE_DRAIN_INTERVAL_S = 0.001` (`engine_core.py:55`), gate at `:454-456`.

### Must be virtualized, never disabled — these three change scheduling

- **`PrefillDelayer`** (`atom/model_engine/prefill_delayer.py`, gated by
  `ATOM_ENABLE_PREFILL_DELAYER`). Its `decide()` performs a cross-DP `all_reduce(SUM)`
  every tick on every rank (`prefill_delayer.py:309-313`). Its input
  `oldest_waiting_age_ms` reads `time.time()` at `scheduler.py:1194`. The delayer's own
  module docstring says it is deliberately *tick*-based for determinism; **this is its
  one wall-clock leak, and skew between DP ranks would make them decide differently and
  desynchronize.** Fix the leak, keep the delayer.
- **`Scheduler._passed_delay`** / `--scheduler-delay-factor` (`scheduler.py:3108-3126`).
- **`seq.arrive_time` / `first_token_time` / `finish_time`** stamps
  (`llm_engine.py:745,777`, `scheduler.py:2688`, `scheduler.py:3364`).

### Leave on real time, deliberately

- `monitor_procs` (`async_proc.py:475-496`) — `multiprocessing.connection.wait` on
  process sentinels. OS-clock, unreachable from Python, and it is a **death detector**,
  not a timeout. Keeping it real is correct.
- Shutdown joins (`engine_core_mgr.py:781-785`, `async_proc.py:363-368`).

The only requirement is that these stay on the *same* clock as each other:
`engine_core_mgr.py:781`'s `time.monotonic() + 5` is patchable while
`async_proc.py:482` is not, so virtualizing only the former makes the two halves of
shutdown disagree.

### Deleted for free by D6

- the MoRI-IO bare-`continue` spin with no sleep and no deadline
  (`moriio_connector.py:330-343`)
- the Mooncake staging-pool busy-wait (`mooncake_connector.py:1083-1087`)
- Mooncake's `PREFILL_LOOKUP_TIMEOUT = 60` blocking `Condition.wait_for`
  (`mooncake_connector.py:76`, `:1596-1608`) and its 2.0 s doubling RDMA retry
  (`:1665-1668`)

### Open issues

- **All 34 `file:line` cites in this section were re-checked against `7fc7a5ddd`** while
  the synchronization inventory was built, and every one holds — including the four
  sites "deleted for free by D6", the two Rust constants, and the three clock reads in
  `streaming_dispatch.py`. The drifted cites are elsewhere: three distinct ones, one of
  which appears twice more in D3, for five occurrences in all. The rows this section
  owns are carried in `atom/compass/audit/sync_sites.json` as pinned lines of text
  rather than as call sites, so a rename or a move fails the inventory test instead of
  rotting quietly.
- Disabling a failure detector removes a safety net from a long unattended run. The
  simulator should log, once at startup, exactly which detectors it disabled, so a
  hung run is diagnosable.

---

## D6. KV transfer: simulated, not carried

### Problem

True PD disaggregation (D2) moves KV between deployments by RDMA. The project
requirement is that simulated execution must not depend on real devices, and that
bandwidth and interconnect are configurable rather than read from hardware. Real RDMA is
therefore out.

### Connector landscape (verified)

`KVConnectorFactory` (`atom/kv_transfer/disaggregation/factory.py:27-171`) registers:

| Name | Model | Classes |
|---|---|---|
| `moriio` (**default** when unset, `factory.py:151`) | RDMA **read**, pull: decode reads from prefill | `moriio_connector.py` |
| `mooncake` | RDMA **write**, push: producer writes into consumer | `mooncake_connector.py` |
| `multi` | fan-out wrapper | |
| `lmcache_offload` | CPU/NVMe offload, **not** P/D | `offload/connector.py` |

There is **no NIXL connector** in this tree.

The ABC is small and has **no `send_kv` / `recv_kv` verb** to fake
(`disaggregation/base.py`):

- worker: `register_kv_caches` (`:33`), `start_load_kv` (`:49`), `get_finished` (`:57`),
  `get_finished_recv_blocks` (`:67`)
- scheduler: `get_num_new_matched_tokens` (`:83`), `build_connector_meta` (`:92`),
  `update_state_after_alloc` (`:97`), `request_finished` (`:102`)

### The seam

Every connector's completion reaches the scheduler through **one** method:

```
model_runner.py::ModelRunner.async_proc_aggregation
  -> EngineCore._poll_kv_transfer_progress   engine_core.py:485-489
     -> Scheduler._update_from_kv_xfer_finished   scheduler.py:2989-3053
```

That single funnel covers any backend, which is why a simulated connector is cheap.

### Design

A `SimulatedKVConnector` registered through the existing factory:

- `get_finished()` releases a request when the **virtual** clock passes
  `issue_time + latency + bytes / bandwidth`.
- `bytes` is exactly computable from block count x per-block bytes — no measurement
  needed.
- `latency` and `bandwidth` are configuration, satisfying the "interconnect is
  configurable" requirement directly.
- It must still emit the `kv_transfer_params` blob that Atomesh relays, so
  `AtomAdapter` works unmodified. The router hard-errors if it is absent
  (`http_pd_router.rs:1073-1078`). The two backends emit **different shapes** —
  thirteen fields and seventeen — so the connector it stands in for decides which;
  see *The blob, per backend* below.
- The consumer side must still return `(len(prompt), True)` from
  `get_num_new_matched_tokens` when `do_remote_prefill` is set, i.e. park the request
  (`moriio_connector.py:904-917`), so `Scheduler._park_for_remote_load`
  (`scheduler.py:2207-2212`) and the `WAITING_FOR_REMOTE_KVS` state behave identically.

### The blob, per backend

Each connector assigns `seq.kv_transfer_params_output` exactly once, so there is no
second site either row below could be describing. Each row's field set is the key
list of that one dict literal, walked out of the AST at `92f1fdafe`.

| Backend | Assignment | Keys | Field set, in source order |
|---|---|---|---|
| `moriio` (pull, the default) | `moriio_connector.py:983-997` | 13 | `do_remote_prefill`, `do_remote_decode`, `remote_block_ids`, `remote_engine_id`, `remote_host`, `remote_port`, `remote_handshake_port`, `tp_size`, `dp_rank`, `transfer_id`, `first_token_id`, `draft_token_ids`, `prefix_cache_hit_tokens` |
| `mooncake` (push) | `mooncake_connector.py:432-452` | 17 | `do_remote_prefill`, `do_remote_decode`, `remote_block_ids`, `remote_swa_block_ids`, `remote_engine_id`, `remote_host`, `remote_port`, `remote_handshake_port`, `tp_size`, `dp_rank`, `remote_pp_size`, `hash_block_size`, `transfer_id`, `first_token_id`, `draft_token_ids`, `local_slot_index`, `prefix_cache_hit_tokens` |

The push shape is the pull shape plus four: `remote_swa_block_ids`, `remote_pp_size`,
`hash_block_size`, `local_slot_index`. They are a second backend's blob, not optional
fields of one, and a simulated connector standing in for `moriio` emits the thirteen.

One of the four is load-bearing rather than descriptive. The push consumer compares
the producer's `hash_block_size` against its own and falls back to a full transfer —
`num_computed_blocks = 0` — whenever it is absent or differs
(`mooncake_connector.py:388-401`), so a blob carrying only the thirteen can never
take the incremental path.

`tests/compass/test_kv_blob_doc_table.py` re-derives both sets from the connectors and
fails naming the field that differs, so this table cannot drift from the source the
way its twelve-field predecessor did.

### Pros

- Removes RDMA, drivers and the handshake entirely.
- Makes interconnect a first-class configured parameter, which a real RDMA run could not.
- Deletes three busy-waits and two long blocking timeouts (D5).
- The parked-duration gauge `_num_parked_remote_kv` (`scheduler.py:2216`, logged at
  `:1648-1656`) is already almost the instrumentation needed to validate it.

### Cons / open issues

- The transfer model is now **unvalidated** — a real RDMA baseline is needed at least
  once to fit `latency` and `bandwidth`, or the numbers are declared rather than
  measured. This is a calibration task, not a simulator task, but it must be named.
- MoRI-IO's `_pop_done_transfers` (`moriio_connector.py:818-837`) polls only
  `status_list[-1].Succeeded()` — the *last* status in the list. If the simulated
  connector is ever compared against the real one, this is a semantic difference to
  watch.
- Mooncake requires **all** `(pp_rank, tp_rank)` pairs to report before a request
  completes (`mooncake_connector.py:1763-1832`); MoRI-IO does not. The simulated
  connector must pick one and declare it.

---

## D7. Atomesh: no virtual clock, three flags and a constant

### Problem

Atomesh is Rust. Every timing primitive in it is `tokio::time` or `std::time::Instant`,
unreachable from a Python virtual clock. In standalone mode it runs a tokio runtime
**inside the Python process** via PyO3 (`src/python.rs:59-84`), so the engine and the
router share an OS process and run on two different clocks.

### Analysis

The ATOM relay is strictly sequential and blocking (D2). Atomesh therefore contributes
exactly two things to a request: a **routing decision**, and **two HTTP round trips of
overhead**. It never overlaps prefill and decode.

So its only time-dependent behaviours are **failure detectors**, which by the D5 rule
get disabled — not virtualized.

### Design

1. **Run mesh-only mode, not standalone.** In standalone, the SSE drain at
   `atom_standalone.rs:172-189` is a `loop { ... if chunks.is_empty() { continue } }`
   paced only by a Python `queue.get(timeout=0.05)`. Under a fast virtual clock that
   degenerates into a GIL-hammering spin. A separate binary keeps the two clocks from
   touching.
2. **Disable the detectors** (D5 table). `--request-timeout-secs` is already 1800.
3. **Model the relay overhead as a declared per-request delay**, exactly as the prior
   work modelled admission (13.7 ms measured; worth ~4 points of TTFT; applied as a
   delay on the request, *not* as time the engine consumes — advancing a global clock
   per admission double-counts concurrent arrivals).
4. Atomesh's routing **decision** is business logic and is reused unchanged. Only its
   latency is modelled.

### An unused hook worth knowing about

`mocker/virtual_workers/mock_case.rs:69-75` declares
`SimulationFixture { ttft_ms, chunk_interval_ms }`, threads it through
`MockCase.simulation` (`:92`), and ships it in all 8 fixtures — with **zero consumers**
anywhere in the crate. A purpose-built latency-injection point that nobody wired up. If
router-side latency injection is ever wanted, this is where it goes.

### Open issues

- The two hardcoded Rust timeouts (`worker.rs:29`, `worker_manager.rs:24`) require
  touching the Rust crate, which means `ATOM_MESH_BUILD=1` and a Rust toolchain in the
  loop. Confirm the container has one.
- Mesh-only mode has not been exercised by this project. Confirm it serves the ATOM
  relay path correctly against two ATOM servers before depending on it.

---

## D8. Arrivals

### Problem

A discrete-event clock may only advance when it knows no earlier event will still turn
up. An HTTP client posts concurrently, so requests reach the engine in a different order
from the one they were declared in.

Measured cost of getting this wrong (64 requests, Poisson 8/s, Qwen3-0.6B TP=1):

| | real | declared arrivals, no barrier | + barrier |
|---|---|---|---|
| TTFT mean | 42.23 ms | **213.50 ms** | 37.39 ms |
| TTFT max | 82.10 ms | **1653.09 ms** | 67.63 ms |
| wall clock | 9,417 ms | 495 ms | 847 ms |

### The prior mechanism and why it is replaced

take2 solved it with an **arrival barrier**: the engine runs nothing until
`len(waiting) >= compass_workload_size`. It works, and it has four costs:

1. It needs a **count**, so it cannot serve open-ended arrival.
2. It is bounded by `ARRIVAL_BARRIER_TIMEOUT_S = 120.0` on the **real** clock, and on
   timeout the run's latencies are invalid. In one case the client still reported
   "0 failed" and a day's conclusions came off it.
3. It forces a client design of one connection per request, hard-bounded at
   `MAX_IN_FLIGHT = 1024` — a 64-thread pool against a 300-request declared workload
   deadlocks, because the server produces no response until all 300 have arrived.
4. It spins: the real run's first step is scheduling tick 1; the simulated run's is
   tick **89,336**. Virtual time is frozen so it costs no fidelity, but it is real CPU
   and it scales with the workload.

### Design

With a CA, the traffic source is simply **an LP that publishes one thing: "no arrival
before time T."** The engine LP can never be granted past `T + L[traffic->engine]`, so
it cannot miss an arrival.

This is the same mechanism every other LP uses. Consequences:

- `_arrival_barrier_unmet`, `compass_workload_size`, the 120 s timeout, and the 89,336
  spinning ticks are all removed.
- **Open-ended serving works**, because the CA needs the next arrival time, never the
  total count.
- The one-connection-per-request constraint relaxes, because the server no longer
  withholds every response until the whole workload has landed.

### The delivery-lag wrinkle

A POST travels uvicorn -> `LLMEngine.preprocess` -> ZMQ -> EngineCore input thread ->
`scheduler.waiting`. The traffic LP must not publish a bound past an arrival that has not
landed in `waiting`.

- **Closed / pre-declared workload** (the cc-traces case): post everything up front, then
  publish bounds. Trivially correct. take2 already has both halves —
  `CompletionRequest.compass_arrival` as an offset into the run
  (`protocol.py`, `llm_engine._stamp_arrival`) and `Scheduler._declared_arrival_pending`
  putting a not-yet-arrived sequence back on the waiting queue rather than routing it
  through `_unschedulable_reason`, which would finish it.
- **Open-ended:** publish the bound only after an enqueue acknowledgement. One extra hop,
  well-defined, build it when needed.

### What the corpus cannot supply

The requirement mentions "faithful agentic behaviour: branching, joins, delays,
completion-driven recycling, cancellation". Verified against the corpus:

- **Joins are not represented.** No field says a parent resumed because a child finished.
- **Cancellation is not represented.** `status` is `"completed"` on all 1,697 subagent
  wrappers; `tool_use_count` is `null` on all 1,697; `subagent_type` is `"Subagent"` on
  all 1,697. There is no other value in either corpus.
- **Branching is represented** — 1,697 wrappers across 175 of 393 sessions, up to 10
  concurrent branches, max 153 branches in one session.
- **Delays are present but undefined on the dataset card** (`think_time`, p50 1.17 s,
  p90 10.45 s).
- Arrivals are therefore **open-loop** by necessity, and the prior tooling deliberately
  keeps them so.

**This must be declared in the acceptance scope rather than claimed.** It belongs to the
workload design topic, but it is recorded here because it is what licenses the
declared-arrival model.

### Open issues

- Whether the workload schedule is owned by the client (declared arrivals, take2's
  model) or handed to the engine as an input file. The prior notes concluded the general
  serving case "needs the schedule handed over up front, with HTTP demoted to fetching
  results". The CA makes the client-owned version viable, so this is now a preference
  rather than a forced move — but it should be decided, not drifted into.
- A client stopwatch cannot time a simulated engine. Measurement egress must come from
  the engine's own readings. take2 used `GET /compass/requests` returning
  `{request_id, seq_id, arrive_time, first_token_time, finish_time, ttft, latency}` plus
  a top-level `clock: "virtual"|"wall"`. Something equivalent is required.

---

## D9. The ATOM diff, and how completeness is enforced

### Expected change set

1. **`atom/utils/clock.py`** — an injectable clock. take2 designed this in 148 lines:
   a `Clock` protocol, `WallClock`, `VirtualClock`, `get/set/reset_clock`. One detail
   worth preserving: `WallClock.epoch` returns `None`, deliberately not `0.0`, because a
   caller offsetting from the Unix epoch would get a 1970 timestamp and a duration in the
   billions; `epoch is None` then becomes the engine-wide discriminator for "real clock,
   none of this applies". Add an `AuthorityClock` that talks to the CA.
2. **Clock-read substitution** at the business-logic sites of D5. take2 did four in
   `scheduler.py`; the full set is ~15.
3. **`EngineCore._process_engine_step_inner`** (`engine_core.py:352`): `declare_next`
   before the forward, `advance_to(now + predicted)` after. take2's `_stamp_step_start`
   and `_advance_clock_for` are exactly these two hooks. The predicted duration rides
   back on a new field of `ScheduledBatchOutput`, which is `None` on a real run so the
   advance is a no-op and a real run is untouched.
4. **`Scheduler._advance_to_next_arrival`** becomes `declare_next(next_arrival)`; the CA
   performs the jump.
5. **Status annotation** at the Category-B sites of D4.
6. **Category-C2 pacing** converted to virtual timers.
7. **Configuration** for the D5 disable list, plus two one-line Rust constants.
8. **`SimulatedKVConnector`** registered through the existing factory (D6).
9. **A simulated `ModelRunner`** injected via `--runner-qualname` (no ATOM change needed
   for the injection itself).

### Size estimate

take2's whole clock integration was **+148 new lines and ~771 modified across 11 files,
with 13 deletions** — almost pure addition, no import-time monkeypatching of any ATOM
class. The CA adds perhaps 400 (authority process plus client). Estimate **~1,200 lines
across ~12 files**, no process removed and no thread removed.

### The enforcement mechanism

Clock-site completeness **cannot be verified by reading.** take2 solved the narrow
version with an AST-walking test (`tests/compass/test_serving_timings.py`) asserting that
no assignment to `first_token_time` / `finish_time` / `arrive_time` in `scheduler.py`
calls `time.*`.

Generalize it: a test that walks the serving path and **fails on any un-allowlisted
`time.time` / `time.monotonic` / `time.perf_counter`**, where the allowlist is exactly
the failure-detector set from D5. That test is what makes this design safe rather than
merely careful, and it should exist before the substitution work starts, not after.

A second test should assert the CA's safety invariant fires: construct a backdated
cross-LP event and assert the run aborts.

---

## Cross-cutting open issues

Ordered by how much they could cost.

1. **Silent failure is the dominant risk mode.** Every failure in D3-D5 produces a
   plausible latency table rather than an exception. The mitigations — always-on
   assertions, loud deadlock abort, the AST test — are the design, not decoration. This
   project's history contains at least four instances of a plausible artifact from a
   broken run being read as a result for a day or more.
2. **Simulation speed is unmeasured under this architecture.** The prior design was
   0.30x under saturation. Multi-process with per-advance RPCs is not obviously faster.
   Measure a saturated cell early, before the architecture is load-bearing.
3. **Per-step replay CPU cost.** Previously measured device-free against a 32.7 ms
   modelled step: 41.7 ms uncached, 2.3 ms with a shape-keyed cache (**unsound** — a
   second valid allocation for the same shape moves 64 of 2439 operator signatures), and
   4.3 ms with the allocation carried in the key. Cold costs are separate: ~9.5-11 s for
   a factory build, 0.27 s for the first graph of a new shape. Beware a cold-loop
   artefact: an average over a loop whose first iteration is a cache miss reads 3-7x
   higher than steady state.
4. **Schedule agreement must be reported separately from latency.** See D0. The prior
   best result reproduced five prefill streaks with identical chunk counts, every break
   within half a second across a 267-second run — and that agreement was established two
   changes *before* the latency numbers were right. It is the more diagnostic metric.
5. **Scheduling fidelity is unobservable on a saturated workload.** Every loaded prior
   experiment ran at essentially 100% utilisation, where the queue term swamps
   everything. A scheduler comparison needs a workload with slack, or controlled step
   durations that remove the cost error by construction.
6. **Multi-node DP is implemented but not hardware-validated** — `docs/distributed_guide.md`
   §9 carries an explicit banner, and the MoRI `InterNodeV1` path and RDMA behaviour are
   unverified. If paired real-vs-simulated evidence is needed there, the real side may
   not exist yet.
7. **CA placement and grant latency** are unresolved (D3).
8. **PP will make the CA the bottleneck** when M7 arrives (D3).

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D0 | Fresh design; prior branches referenced at the design level only, not as a code-port plan | 2026-09-17 |
| D1 | Keep ATOM's multi-process / multi-thread topology; additive changes only | 2026-09-17 |
| D2 | "Two nodes" means true PD disaggregation, realised as two containers on one physical node | 2026-09-17 |
| D3 | Central Clock Authority, grant rule `min_j(now[j] + L[j->i])`, LPs collapsed on existing hardware barriers | 2026-09-18 |
| D3.1 | Single CA with a hierarchy-ready interface; LP count scales with replicas and PP stages, not with GPUs | 2026-09-18 |
| D3.2 | Three always-on causality detectors: receive-side straggler check (fails the run), annotation-coverage watchdog (warns), clock-source CI lint | 2026-09-19 |
| D3.3 | CA deploys two ways from one implementation: co-hosted in the API-server process by default, standalone server via `--compass-clock-endpoint` for M4/M6 multi-container runs | 2026-09-19 |
| D3.4 | Wall-clock interleaving may vary between runs; the `(LP, virtual time, event)` sequence may not. CA grants tie-break by **LP identity, never arrival order**; the cost backend is a pure function of its batch view; **no `set` iteration on the simulated path** - string ids are hashed under `PYTHONHASHSEED` randomisation, so a set of request ids iterates differently in every process. Test is a byte-diff of two step tables, CPU-only, in CI. | 2026-09-20 |
| D3.5 | The CA owns three outputs: an opt-in timeline log, a deadlock dump naming every LP's state and lookahead row, and an always-written run summary carrying grants, speed ratio, lazy-trace cost, detector state and the refusal fractions `08` D50.1 gates on. | 2026-09-20 |
| D4 | Four-category interception contract; annotate rather than intercept cross-LP blocking waits | 2026-09-18 |
| D5 | Disable failure detectors, virtualize business logic; sorting rule is "does it change which batch gets scheduled" | 2026-09-18 |
| D6 | KV transfer is simulated through a connector registered in the existing factory | 2026-09-18 |
| D7 | Atomesh gets no virtual clock: mesh-only mode, detectors disabled, relay latency declared | 2026-09-18 |
| D8 | Arrivals via the CA as a next-arrival lower bound; the take2 arrival barrier is replaced | 2026-09-18 |
| D9 | The ATOM diff is enumerated and completeness enforced by test rather than by review; ATOM's own suite is the gate (`08` D43.1) | 2026-09-18 |

---

## Appendix: verified reference points

Facts this design leans on, with their source, so a later reader can re-check rather than
re-derive.

**The seam**
- `Config.runner_qualname` — `atom/config.py:1595`; consumed `engine_core.py:129`,
  `async_proc.py:166-169`
- `model_runner.py::ModelRunner.forward`, whose signature is
  `forward(batch: ScheduledBatch) -> ScheduledBatchOutput`
- the RPC boundary — `engine_core.py:386-388`
- `ScheduledBatch` fields — `scheduler.py:579-820`; notably `detailed_sqsq` /
  `detailed_sqsk` / `detailed_sk` at `:790-792`, which are sum(N_Q^2), sum(N_Q * N_KV),
  sum(N_KV) per batch, computed by `compute_detailed_aggregates` (`:2788-2842`) and
  currently gated on `profile_active and ATOM_ENABLE_DETAILED_ANNOTATION`
- `ScheduledBatchOutput` — `scheduler.py:841-885`; `produces_output()` at `:823-840`

**Existing simulation-shaped hooks in ATOM**
- `--load_dummy {empty,zero,xavier}` — `config.py:1556`, `arg_utils.py:260`,
  `loader.py:179-227,309-310`, `loading_core.py:266-291`
- meta-device model construction —
  `model_runner.py::RapidServeModelRunner._init_weight_params_on_meta`
- a working non-allocating runner template — `model_runner.py::RapidServeModelRunner`,
  which overrides `_build_and_load_model`, `_maybe_warmup`, `_kv_budget_extra_reserve`,
  `get_num_blocks`, `allocate_kv_cache` and `forward`
- `model_runner.py::ModelRunner.dummy_execution` shows how to hand-build
  a `ScheduledBatch`
- `ScheduledBatch.is_dummy_run` — `scheduler.py:589,781`
- simulated TP (`--fake-eplb`) — `atom/distributed/simulated_tp.py`; explicit precedent
  for a shape-accurate, value-meaningless run
- synthetic speculative acceptance — `config.py:1064-1190`,
  `atom/model_ops/rejection_sampler.py:20-224`; the closest existing behaviour simulator
- profiler label taxonomy — `atom/model_engine/run_labels.py`; consumed by
  `tools/parse_trace.py`

**Memory sizing (needed because it decides which configurations exist)**
- `model_runner.py::ModelRunner.get_num_blocks`, with its four `torch.cuda`
  reads in `model_runner.py::ModelRunner._read_device_memory`. Five device readings plus
  arithmetic: `mem_get_info`, `allocated_bytes.all.peak`,
  `(total - free) - memory_reserved()`, `_estimate_cudagraph_overhead()`, a 2% safety
  margin, then `min(budget - ..., free)` and `plan_pools`. Consumed
  `engine_core.py:132-145`.
- `BlockManager.__init__` asserts `num_blocks > 0` (`block_manager.py:77`), so a simulated
  runner must return a plausible count.
- The prior work's rule: **substitute the readings, never the arithmetic.**

**Prefix caching** (default on, `config.py:1546`)
- hash: `BlockManager.compute_hash` — `block_manager.py:233-245`, xxhash xxh64 chained
  with the parent hash
- hit scan: `can_allocate` — `block_manager.py:469-561`
- publish, deferred until after the forward computed the KV: `hash_blocks` —
  `block_manager.py:696-771`, called from `Scheduler.postprocess` at `scheduler.py:2404,2420`

**Timing call sites in the serving path**
- request latency uses **`time.time()`** (wall) because `arrive_time` is stamped in the
  API process and `first_token_time` in the EngineCore process: `llm_engine.py:745`,
  `scheduler.py:2688`, `scheduler.py:3364`, `llm_engine.py:777`; TTFT/TPOT computed at
  `llm_engine.py:781-799`
- loop pacing uses `time.monotonic()`; benchmark clients use `time.perf_counter()`
- **the main `EngineCore._process_engine_step_inner` does not time the forward.** Only the
  RapidServe prefill/decode cores do (`engine_core.py:991-1001`, `:1263-1274`).

**Test harness that a compatible seam inherits**
- `tests/conftest.py:49-78` — `MockConfig`, a GPU-free, download-free stand-in giving
  `BlockManager` / `Scheduler` exactly the fields they read
- `tests/aiter_stub.py:11` — `stubbed_aiter()` so `async_proc` imports on a CPU runner
- directly relevant existing tests: `test_scheduler.py`, `test_block_manager.py`,
  `test_block_pool.py`, `test_prefill_scheduler.py`, `test_scheduled_batch_marshal.py`,
  `test_forward_mode.py`, `test_prefill_delayer.py`, `test_dp_load_balance.py`,
  `test_simulated_tp.py`, `test_kv_drain_liveness.py`
