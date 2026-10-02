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
  consumed at `engine_core.py:128` and `async_proc.py:166`. Two in-tree precedents:
  `RLHFModelRunner` (`atom/rollout/async_engine.py:26-32`) and `RapidServeModelRunner`
  (`Config.__post_init__`, `config.py:1730-1736`).
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

ATOM's serving path contains many distinct synchronization points: blocking ZMQ
recvs, bounded pollers, queue gets with timeouts, Gloo and NCCL collectives,
`multiprocessing` barriers and joins, busy-waits, and literal sleeps. "Intercept every
blocking call" is the obvious reading of what a virtual clock demands. It is also the
reading that turns this project into reimplementing a scheduler on top of the OS
scheduler.

### The contract

Each site is answered by **the PDES mechanism a simulated run applies there**, not by the
shape of the wait. What decides it is which LP owns the waiting thread and what the wait
means for that LP's clock. Each LP has one clock owner — its step loop or its asyncio
event loop — and only the clock owner moves the LP's clock (D3); every other thread in the
LP is transport. Every site maps to one of the mechanisms K1–K9, and **none is left
undecided**.

| Class | Mechanism | PDES rule | Representative sites |
|---|---|---|---|
| K1 | Event cost | TAR: `advance_to(now + d)`, `d` from the cost model or a resource station | the forward, the idle DP rank's dummy batch, KV-transfer completion, tokenization, the DP lockstep `all_reduce`, multimodal preprocessing (refused today) |
| K2 | Clock read | returns the LP's logical time | the arrive / leave / first-token stamps, `_passed_delay`, the waiting-prefill age |
| K3 | Idle point | NER: `next_event(t)` | the step loops' spin, the PP bounded polls, offline `get_output` |
| K4 | Channel send | timestamped send, logged on the clock owner | PD direct sends, PP sends, output sends through the relay queue |
| K5 | Channel receive | counted at the wait point, then TSO delivery | the engine input thread, the frontend output thread, the PD handler threads, the PP receives and zero-bound drains, the PP `flush_pp_send` |
| K6 | Wait inside one LP | LP aggregation: never reported to the CA | TP worker RPC and barriers, frontend coroutines awaiting their own process's data, control commands' calls to workers, the DP group's reduction |
| K7 | Virtual timer | the timer runs on the LP clock | idle KV drain, metrics push and refresh, Anthropic ping, keep-alive, the silence warning, the control-command reply timeout |
| K8 | Real bound | the CA cannot reach it, or it stays real on purpose: configuration | the Rust router's bounds and health check, process-death detection |
| K9 | Outside the model | outside the simulation window, inside replaced code, or cannot park | startup and shutdown, the replaced runner and RDMA backends, collectives inside the real forward, text scanners |

Each site's mechanism is the `mechanism` field #476 adds to its row in
`atom/compass/audit/sync_sites.json`, produced by the scanner beside it. A row's
first classification (#53) maps naturally to A→K1, B→K5, C1→K8, C2→K7, ignore→K9; the
rules below say where a site moved off that map.

The rules that settle the boundaries #53's categories left in the wrong place:

- **Inside or across an LP, not the same or another process.** A wait whose peer is in
  the same LP is K6: the LP never reports it to the CA, and its duration belongs to the
  event the clock owner is pricing. The TP group is one LP, so the engine's RPC to its
  workers — `call_func(..., wait_out=True)`, which parks in `self.outputs_queue.get()`
  in `AsyncIOProcManager.call_func` (`atom/model_engine/async_proc.py`) and carries every
  forward — is K6 although its peer is another process; the forward's duration is
  charged by the clock owner's TAR. #53 drew this line at the process and put such waits
  in B. K5 holds only the receives that cross an LP: those, the PP zero-bound drains, and
  the PP idle flushes.
- **Zero-lookahead couplings go inside one LP.** A barrier or collective, a synchronous
  RPC, or a live read of the peer's state lets one side's action at *t* change the other
  at *t*. Split across LPs, every round of such a coupling serializes through the CA, and
  ATOM's real collectives would bypass it ([D3](#d3-virtual-time-coordination-protocol)
  sizes both). So the TP workers and every rank of a lockstep DP group sit in the engine
  LP, and a startup assertion refuses a process group whose ranks map to more than one
  LP.
- **A DP step is one TAR, priced by its per-layer critical path** (D3). The ranks exchange
  their batch descriptions on the lockstep `all_reduce` they already run
  (`DPEngineCoreProc._sync_dp_state`, `atom/model_engine/engine_core.py`), each computes
  the same path, and one rank talks to the CA — which is why that `all_reduce` is K1
  rather than ignored: it is where the DP LP's step length is decided. **Second-order
  bound:** a fast rank's output is stamped at the end of the whole step, late by at most
  the difference between the ranks' tail times; the step is split with in-step TAR only
  if that is measured to affect TPOT.
- **A bound in Python is a virtual timer; only what the CA cannot reach is a real
  bound.** #53 split bounded waits by purpose, failure detector (C1) against pacing (C2).
  Both run on the LP clock now (K7, D5). K8 is the Rust router's bounds and health check,
  and OS-level process-death detection. The C1 bounds inside the TP group
  (`AsyncIOProcManager.process_output_sockets`, `process_kv_output_sockets` and
  `call_func_with_aggregation`) are K6: real-time bounds on in-LP transport that the
  simulated runner answers in real milliseconds, so they never fire and stay as they are.
- **Sends are counted.** #53 ignored sends because they do not park. Under D3 a send is
  one end of the in-transit message count: it carries the timestamp and is logged on the
  clock owner, so the sends are K4. A direct send (the clock owner calls `send`:
  `PrefillEngineCore._process_engine_step` and `DecodeEngineCore._send_block_assignment`
  in `engine_core.py`, the `PPStageTransport` sends in `atom/distributed/pp_transport.py`)
  is stamped by a socket wrapper. The engine's output sends go through its output thread
  (`EngineCore.process_output_sockets`), so `output_queue`, created in
  `EngineCore.__init__`, becomes a relay queue that stamps at `put`, on the step loop:
  the timestamp is the step's, not the moment the output thread wakes. Inside the
  simulation window every item put is sent — the one skip is a list of `EXIT_ENGINE`
  sequences, which exist only at shutdown — so there are no null messages, and an item
  never sent raises `UnsentRelayItem`.

#### K1–K3 — what #53 called category A, as measured

- **K1**, the forward pass: `EngineCore._process_engine_step_inner` (the main step),
  `PrefillEngineCore._process_engine_step` and `DecodeEngineCore._process_engine_step`
  (the two halves of RapidServe) in `atom/model_engine/engine_core.py`, and
  `PPEngineCoreProc._pp_head_step` and `_downstream_busy_loop` (the PP head and a
  downstream stage) in `atom/model_engine/pp_engine_core.py`
- **K1**, the idle rank's empty batch, `DPEngineCoreProc._execute_dummy_batch` — it
  consumes a step and is charged like any other
- **K1**, KV-transfer completion (D6): `EngineCore._poll_kv_transfer_progress`,
  `PPEngineCoreProc._poll_kv_transfer_progress` and `_poll_and_send_kv_status`
- **K1**, **tokenization**, at the `run_in_executor` hand-offs `06` D33 names, in
  `generate_async`, `generate_async_multimodal`, `generate_async_fanout`,
  `setup_streaming_request` and `setup_streaming_request_fanout`
  (`atom/entrypoints/openai/api_server.py`): a multi-server resource station in the
  frontend LP whose completion times come from D33's service model. The multimodal
  preprocessing hand-off in `chat_completions` is K1 too, and is refused today because no
  service time is modelled for it.
- **K1**, the DP lockstep `all_reduce` (`DPEngineCoreProc._sync_dp_state`), per the DP
  rule above
- **K3**, **the idle jump, which is a real site in ATOM even though the name this list
  once gave it is not.** `Scheduler._advance_to_next_arrival` does not exist here — but
  the loops it would have served do, and each spins rather than waits when there is
  nothing to run, because `pull_and_process_input_queue` drains with `get_nowait` and
  nothing else in the turn blocks: `EngineCore.busy_loop`, `DPEngineCoreProc.busy_loop`
  and `PPEngineCoreProc._head_busy_loop`. D8's measurement of the prior design — first
  real step is tick 1, first simulated step is tick **89,336** — is this loop counted.
  Each becomes `next_event(t)`, `t` the LP's earliest local event. The PP bounded polls
  (`PPStageTransport.recv_tokens` and `recv_metadata`) are `next_event(now + bound)`, and
  the offline driver's `CoreManager.get_output` (`atom/model_engine/engine_core_mgr.py`)
  is `next_event(inf)`.
- **K2**, `Scheduler._passed_delay` / `--scheduler-delay-factor`,
  `Scheduler._oldest_waiting_prefill_age_ms` feeding `PrefillDelayer`, and the stamps the
  result reports: `seq.arrive_time` in `InputOutputProcessor.preprocess_fanout` and
  `req.leave_time` in `InputOutputProcessor.postprocess` (`atom/model_engine/llm_engine.py`),
  `seq.first_token_time` in `Scheduler.postprocess` and `DecodeScheduler.on_prefill_done`
  (`atom/model_engine/scheduler.py`) — D5 lists these as business logic

**One entry of the original list is not ATOM code at all.** The arrival gate (D8) is
something Compass adds: `_arrival_barrier_unmet`, `compass_workload_size` and
`ARRIVAL_BARRIER_TIMEOUT_S` return nothing on the whole `atom/` tree, so there is no
site to intercept, only a mechanism to build.

#### K5 and K6 — what #53 called category B

**Nothing is annotated.** #53's category B wrapped each cross-process wait in
`declare_blocked()` / `declare_running()`. Under D3 an LP's idleness is inferred by the
CA from its clock owner parking at an idle point with nothing left to release, so no
thread declares anything:

- **K6**, `call_func(..., wait_out=True)`, per the first rule above, and the out-of-band
  control commands' calls to the worker in the `EngineUtilityHandler._handle_*` methods
  (`atom/model_engine/engine_utility.py`); D5 says which of those commands a simulated
  run refuses.
- **K6**, the streaming endpoints' collector reads in `stream_chat_response` and
  `stream_chat_response_fanout` (`serving_chat.py`), `stream_completion_response` and
  `stream_completion_response_fanout` (`serving_completion.py`), and the same call in
  `anthropic_messages` (`api_server.py`): a frontend coroutine awaiting data its own
  process produces.
- **K5**, the engine's input thread (`EngineCore.process_input_sockets`), the frontend's
  output thread (`CoreManager._create_output_thread`), and the RapidServe handler threads
  (`PrefillEngineCore._recv_block_assignments`, `DecodeEngineCore._recv_prefill_done`).
  A receiving thread is counted when it returns to its wait point, so every branch is
  counted, including a frame dropped on a decode error; the poller is wrapped where it is
  created, and a handler thread's socket is wrapped. Then **TSO delivery**: a received
  message is held at the wait point until the clock owner's `advance_to` walks past its
  arrival time, and released one at a time in arrival order.
- **K5**, the PP idle flush, `flush_pp_send` in `PPEngineCoreProc._pp_head_step` and
  `_downstream_busy_loop`: it waits for the stage's previous send, whose completion a
  rendezvous (large) send learns from the next stage. The stage loop receives that send's
  `stage(k+1)->stage(k):pp_ack#dp0` inline before the call; an eager (small) send has a
  local completion and no ack, and the loop advances to it by K1's rule. The forward's
  own wait for the previous send is the same receive, made at the forward call site after
  its compute TAR. The shutdown call in `_downstream_busy_loop` is the same call as the
  idle one and shares its answer; it runs outside the simulation window, so sharing
  changes nothing. The head's shutdown call in `_head_busy_loop` is outside the window
  too and stays K9. The real `isend` and `wait()` inside the replaced runner are K9
  (`15` D91 Q3).
- **K9**, `CoreManager._wait_for_all_ready_signals`, reached only from
  `CoreManager.__init__`: waiting for READY is outside the simulation window. #53 kept
  these as B with the contradiction recorded; the LP rule resolves it.

**On the send side (K4), "it cannot park" is true of most of them and has to be said per
socket, not once.** The PP stage loop also registers its `pp_data` and `pp_ack` sends at
the forward call sites (`15` D91). Of the sends the scanner finds, the ones built by
`make_zmq_socket` (`atom/utils/__init__.py`) carry `SNDHWM=0` and provably cannot park.
The rest are bare `ctx.socket(...)` and keep ZeroMQ's default thousand-message bound: the
`PPStageTransport` sends, the disagg bootstrap sends, and — the two that matter at
runtime — `PrefillEngineCore._process_engine_step` on the **prefill step loop** and
`DecodeEngineCore._send_block_assignment` on the decode one, whose sockets each class's
`_init_disagg` creates bare. Their protocol is one message per sequence and the peer
drains it every tick, so a thousand-message backlog is not reachable in a run; the rows
say that rather than claiming the bound does not exist.

#### Which mistakes fall on the loud side

A channel site this table got wrong — a send or a receive — leaves an LP waiting: a
message logged but never delivered, or an LP stuck in a primitive the CA cannot see. Such
a wait produces no simulated time, so the run waits for it, prints one stall diagnostic
naming the channel and sequence number after `DIAG_S` wall seconds, and keeps waiting
(D3). The opposite failure — advancing past a message — is what the send count and the
strict grant rule rule out. So for sends and receives, mistakes fall on the loud side.
The other mechanisms do not: a missed event cost charges no modelled time, and a clock
read or timer left real stamps wall time, both silent at run time. The clock-source lint
(D9) catches a clock read left real; validation against real runs (`08`) catches a
missed event cost.

### The case that is not a "wait" problem at all

`DecodeEngineCore._recv_prefill_done` blocks in a background thread and calls
`DecodeScheduler.on_prefill_done`, which stamps `seq.first_token_time = time.time()`.
Intercepting the block fixes nothing. The rule is about **when the handler runs**:

> **Handler threads stay where they are. TSO delivery runs each one at its message's
> timestamp, while the clock owner waits.**

Moving `on_prefill_done`'s logic into the step loop instead would change ATOM's thread
model (D1).

Left alone, the handler runs whenever its thread wins the race, so which step's
`schedule()` sees the sequence changes from run to run. TSO delivery fixes that without
touching the handler: the receive wrapper holds the frame, and the clock owner's
`advance_to` walks the pending arrivals in order — set the LP clock to the arrival time,
release one message, wait until the handler thread is back at its wait point, release
the next. `time.time()` inside the handler then reads the arrival time, and `schedule()`
never sees a message from its future. While the handler runs, the clock owner is parked
in its CA call, so one executor handles events in the LP at a time, as with HLA's
single-threaded callbacks. **`on_prefill_done` needs no change.** The cost is that a
simulated run never reproduces the real lock-free race between this handler and the step
loop.

### Open issues

- ~~The counts above are estimates from a synchronization inventory, not from a completed
  pass over the code. The first implementation task should be to produce the exact
  classified list and check it in.~~ **Done, 2026-09-21.** `atom/compass/audit/`.
- ~~The literal `time.sleep(2)` in `DecodeEngineCore._post_model_load_hook` ... it must
  be *checked*, not assumed.~~ **Checked, 2026-09-21, and both halves hold.** It is
  reached only from that hook, and `DecodeEngineCore` is constructed only by
  `DisaggCoreManager`, which `LLMEngine.__init__` selects only under
  `config.enable_rapidserve` — so "RapidServe-only" is exact.
  Its purpose is verified by the code around it: the sleep sits between importing
  decode's weight IPC handles and acknowledging to prefill, and prefill measures free
  VRAM for KV sizing only after that ACK. It must stay on the real clock. Nothing in ATOM
  couples it to whether weights are real, but `Config` keeps a simulated runner from it:
  `--enable-rapidserve` selects `RapidServeModelRunner` only when `runner_qualname` is
  still the default (`Config.__post_init__`), and otherwise `Config` raises `ValueError`
  unless `runner_qualname` is in `RAPIDSERVE_RUNNERS`. The cost, for a runner that list
  names, is two real seconds of startup and no modelled time, because it runs before
  READY and therefore before any arrival.
- The scanner's boundary is a list of directories, not a graph. It reads every `.py`
  file under `SCANNED_ROOTS`, so a module added beside a scanned one is caught; but a
  blocking call under a directory `UNSCANNED_ROOTS` names is invisible to the test, and
  so is one reached through a call shape the scanner does not know. The offload
  connectors are the largest exclusion.

---

## D5. Timers, clock reads and control commands

### Problem

Some time-dependent behaviour in ATOM changes what the scheduler does; some only decides
when to declare something broken; some is a command from outside the request path. Each
needs a stated answer, and a wrong one either changes the schedule or hides a cost.

### The sorting rule

> **Can the CA reach it?**
> Yes (a Python `time.*` read, an asyncio timer) -> it runs on the LP clock.
> No (a compiled Rust constant, an OS-level wait) -> configuration, or left real on purpose.

This replaces the earlier rule, *"does it change which batch gets scheduled? — if not,
disable it"*. Deterministic-simulation testing (FoundationDB, madsim, turmoil) puts every
timer on the virtual clock and stops sorting them by purpose: a failure detector then
fires in simulated time, only when the simulated system really is that slow, which is
the right behaviour. Disabling it was a second mechanism for no gain.

### Configured, because the CA cannot reach them (K8)

| Thing | Where | Simulated run |
|---|---|---|
| Atomesh health check + circuit breaker | `--disable-health-check --disable-circuit-breaker` | switched off; the flags are already used in ATOM's own CI launcher |
| Atomesh fan-out request bound, default 5 s (`request_timeout` in `atom/mesh/src/core/worker_manager.rs`) | `--worker-request-timeout-secs` (#478) | set large |

That is one Rust bound. The 30 s `DEFAULT_WORKER_HTTP_TIMEOUT_SECS`
(`atom/mesh/src/core/worker.rs`) is never reached: it is the default of a client whose
only request is the health check, which a simulated launch switches off and which sets
its own timeout (#477). The fan-out bound bites whenever the CA holds an LP while wall
time passes, which includes virtual time running **slower** than wall under saturation
(0.30x measured). It is not optional.

### On the LP clock (K7), and why nothing needs a flag

Every Python-side `time.*` read goes through the clock substitution of D9 item 2 and
returns LP time. Every asyncio timer becomes a virtual timer with no code change, because
the frontend's event loop itself runs on LP time (D5.1): the Anthropic SSE ping
(`_ANTHROPIC_PING_INTERVAL_SECONDS`, 5.0 s, read in `anthropic_messages` in
`atom/entrypoints/openai/api_server.py`), uvicorn's `--timeout-keep-alive` (parsed and
applied in that file's `main`), and the stream silence warning (`SILENCE_LOG_SECONDS`,
30.0 s, clock reads in `FrameWait` and `longest_silence_seconds` in
`atom/entrypoints/openai/streaming_dispatch.py`). The control-command reply timeout
(`CoreManager.broadcast_utility_command_sync`) is measured on the LP clock the same way.
Each fires only if the simulated system is really that slow, which is when a real one
would fire it too.

The pacing timers are local next events. Each joins the `t` the clock owner hands to
`next_event(t)` (D4, K3):

- `KV_IDLE_DRAIN_INTERVAL_S = 0.001` (`atom/model_engine/engine_core.py`), gated in
  `EngineCore._advance_idle_kv_transfer`.
- `METRICS_PUSH_INTERVAL_S`, the engine's metrics push (`EngineCore.busy_loop` and
  `DPEngineCoreProc.busy_loop`), and the API server's refresh loop
  (`_metrics_refresh_loop` in `api_server.py`). **Metric cadence is virtual time again**,
  which revises `11` D72's decision to keep it on the real clock: an observer in the
  traffic LP scrapes `/metrics` every `scrape_interval` of simulated time. `11` D72
  carries the observer and the reasons.

### Must read the LP clock — these three change scheduling (K2)

- **`PrefillDelayer`** (`atom/model_engine/prefill_delayer.py`, gated by
  `ATOM_ENABLE_PREFILL_DELAYER`). Its `should_allow_prefill` performs a cross-DP
  `all_reduce(SUM)` every tick on every rank. Its input `oldest_waiting_age_ms` reads
  `time.time()` in `Scheduler._oldest_waiting_prefill_age_ms`. The delayer's own module
  docstring says it is deliberately *tick*-based for determinism; **this is its one
  wall-clock leak, and skew between DP ranks would make them decide differently and
  desynchronize.** Fix the leak, keep the delayer.
- **`Scheduler._passed_delay`** / `--scheduler-delay-factor`.
- **The arrive, leave and first-token stamps** D4 lists under K2.

### Leave on real time, deliberately

- `AsyncIOProcManager.monitor_procs` (`atom/model_engine/async_proc.py`) —
  `multiprocessing.connection.wait` on process sentinels. OS-clock, unreachable from
  Python, and it is a **death detector**, not a timeout. Keeping it real is correct (K8).
- Shutdown joins (`CoreManager.close`, `AsyncIOProcManager.exit`), outside the
  simulation window (K9).

The only requirement is that these stay on the *same* clock as each other:
`CoreManager.close`'s `time.monotonic() + 5` is patchable while `monitor_procs`'s wait
is not, so virtualizing only the former makes the two halves of shutdown disagree.

### D5.1. The simulated run uses stdlib asyncio, not uvloop

The frontend's clock owner is its event loop, and hosting LP time needs two things from
it: timers that expire on the loop's `time()`, and a Python-level point where the loop
blocks, to call `next_event`. The stdlib loop has both — `_run_once` judges timer expiry
and the select timeout from `self.time()`, and blocks in `self._selector.select(timeout)`,
whose selector is replaceable. uvloop has neither: `call_at` becomes a libuv timer counted
in real milliseconds, and the loop blocks inside `uv_run` in C. The only way to keep
uvloop is to intercept `clock_gettime` and `epoll_wait` for the whole process, which is not
an additive change at an existing site. So a simulated run passes uvicorn a stdlib loop
class through `loop="module:Class"`, where `main` in `api_server.py` already chooses the
loop. Only the loop implementation changes; processes, threads and the serving topology
do not (D1), and the frontend's service times come from the model, not from the wall.

### Control commands

The engine registers its out-of-band commands in `EngineUtilityHandler._UTILITY_HANDLERS`
(`atom/model_engine/engine_utility.py`). Each arrives on the control channel, runs on the
step loop between steps, reaches a worker, if at all, through the in-LP RPC (K6), and
replies on the output channel.

| Commands | Simulated run | Why |
|---|---|---|
| `abort_request`, `get_mtp_stats`, `get_mtp_statistics`, `get_cache_statistics` — call no worker | **run as usual**, unchanged | scheduler state only, inside the LP. `abort_request` fires when a client disconnects and changes scheduling, so it must stay |
| `start_profile`, `stop_profile` (HTTP routes `start_profile` and `stop_profile` in `api_server.py`) | **refused** | a real profiler slows every later step; charging it zero is a silent simplification, and a profiled system is not what is simulated |
| `update_weights`, `update_weights_shm`, `update_weights_ipc`, `release_memory`, `resume_memory`, `clear_kv_cache`, `configure_hidden_states` | **refused** | RL weight sync and sleep/wake. The API server has no route to them; the only senders in the tree are in `atom/rollout/` (`async_engine.py`, `weight_sync.py`), a Python interface for external RL frameworks. Simulating them needs cost models for weight transfer and memory release that do not exist |

A command is refused by the **simulated runner**, in the worker method it reaches, with an
exception naming the command, and is counted in the run summary's refusals (D3.5).
ATOM's handlers are unchanged, the command travels its real channel, and a refusal is
visible rather than a silent zero.

### Deleted for free by D6

In `atom/kv_transfer/disaggregation/`:

- the MoRI-IO bare-`continue` spin with no sleep and no deadline
  (`MoRIIOConnector.start_load_kv`)
- the Mooncake staging-pool busy-wait (`MooncakeConnector._acquire_staging_slot`)
- Mooncake's `PREFILL_LOOKUP_TIMEOUT = 60` blocking `Condition.wait_for`
  (`MooncakeConnector._wait_for_prefill_data`) and its 2.0 s doubling RDMA retry
  (`_rdma_write_with_retry`)

### Open issues

- A real bound configured off removes a safety net from a long unattended run. The
  simulator should log, once at startup, exactly which K8 bounds it configured, so a hung
  run is diagnosable.

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

A fifth scheduler-side hook is not on the ABC: `process_completions`, which the scheduler
calls on each completion report when the connector defines it (`scheduler.py:3002-3004`).
The offload and `multi` connectors define it; Mooncake and MoRI-IO do not.

### The seam

Every connector's completion reaches the scheduler through **one** method:

```
model_runner.py::ModelRunner.async_proc_aggregation
  -> EngineCore._poll_kv_transfer_progress   engine_core.py:485-489
     -> Scheduler._update_from_kv_xfer_finished   scheduler.py:2989-3053
```

That single funnel covers any backend, which is why a simulated connector is cheap.

### Design

A `SimulatedKVConnector` is registered through the existing factory. It reproduces
**Mooncake**, the backend ATOM's PD CI deploys
(`.github/scripts/atomesh/pd_server_atom.sh:560`, `:612`), and
implements **scheduler-side hooks only**: sends and receives both run on the engine's
step loop, and no worker takes part.

- `T = latency + bytes / bandwidth`. `bytes` is exactly computable from block count x
  per-block bytes — no measurement needed. `latency` and `bandwidth` are
  configuration, satisfying the "interconnect is configurable" requirement directly.
- **Decode** posts the write request in `update_state_after_alloc`
  (`scheduler.py:2196`): channel `engine-D->engine-P:kv_write_req`, sent at `t`,
  arriving at `a = t + request latency`. It computes its own ready time
  `a + T + notify latency` and, once the clock reaches it, reports the request in
  `finished_recving` from `process_completions`. A local event of engine-D.
- **Prefill** records the data-ready time `r` in `request_finished`
  (`scheduler.py:2708`, `:2768`) and takes the write request in `process_completions`,
  which the step loop receives inline. At `max(a, r) + T` it reports
  `finished_sending`, and the scheduler frees the blocks it parked in
  `deferred_free_blocks` (`:2772-2774`, freed at `:3038-3046`). A local event of
  engine-P. Scheduler code is unchanged.
- **Write-done is not a message.** Real Mooncake sends `MSG_WRITE_DONE` when the write
  ends (`mooncake_connector.py:1313`). Both ends compute that time instead, for two
  reasons: the router forwards strictly in sequence (D2), so the decode request, and
  with it the write request, exists only after prefill has returned, i.e. `r <= a`,
  which prefill asserts and refuses otherwise; and `T` depends only on this request's
  bytes and the configuration, not on other writes in flight. No channel is added; one
  message whose time both ends already know is not simulated.
- **Why this is the real channel.** A simulated component keeps the real one's
  channel: endpoints, message and trigger. The write request is Mooncake's
  `MSG_WRITE_REQUEST`, decode to prefill, triggered by decode's block allocation. The
  real one leaves a decode worker (`mooncake_connector.py:1048`) for a prefill worker's
  listener thread (`:1139`); the simulated one leaves and reaches the engine process of
  the same two LPs, so the LP-level channel is the same and no worker ever sends or
  receives across LPs. A prefill-side local timer would not do: when decode posts
  depends on its own admission, which prefill cannot know, and assuming "at once"
  frees prefill's blocks early exactly when decode is congested.
- **Ceiling.** Pricing bandwidth contention between concurrent writes would make `T`
  depend on prefill's other writes, which decode cannot see. That model needs the
  write-done message back, plus protocol support for a thread that handles a write
  request sending a message; in this protocol only an LP's clock owner registers a
  send (D3).
- It must still emit the `kv_transfer_params` blob that Atomesh relays, so
  `AtomAdapter` works unmodified. The router hard-errors if it is absent
  (`http_pd_router.rs:1067-1078`). The two backends emit **different shapes** —
  thirteen fields and seventeen — and Mooncake's is the seventeen; see *The blob, per
  backend* below.
- The consumer side must still return `(len(prompt), True)` from
  `get_num_new_matched_tokens` when `do_remote_prefill` is set, i.e. park the request
  (`mooncake_connector.py:320-328`), so `Scheduler._park_for_remote_load`
  (`scheduler.py:2207-2212`) and the `WAITING_FOR_REMOTE_KVS` state behave identically.
  It also clears the flag in `update_state_after_alloc`, as Mooncake does (`:383`), so
  every call that posts a write request is followed by that park.

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
fields of one, and the simulated connector, standing in for `mooncake`, emits the
seventeen.

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
- Every cross-LP send and receive stays on the engine's step loop, so no LP state has
  to be shared with a worker process.
- The parked-duration gauge `_num_parked_remote_kv` (`scheduler.py:2216`, logged at
  `:1648-1656`) is already almost the instrumentation needed to validate it.

### Cons / open issues

- The transfer model is now **unvalidated** — a real RDMA baseline is needed at least
  once to fit `latency` and `bandwidth`, or the numbers are declared rather than
  measured. This is a calibration task, not a simulator task, but it must be named.
- Mooncake requires **all** `(pp_rank, tp_rank)` pairs to report before a request
  completes (`mooncake_connector.py:1761-1832`). The simulated write is one event per
  request, priced on the whole block table, so ranks with unequal shares are not
  modelled. Under PP, Mooncake adds a decode-to-prefill `MSG_RELEASE` channel
  (`:1759`) that this section does not cover.

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

M4 is **1P1D**: one prefill and one decode instance, the only layout ATOM's single-node
PD launch supports (`.github/scripts/atomesh/pd_server_atom.sh:271`). The routing
decision then has nothing to choose between, so no load count, random draw or clock
read reaches its result, and the router is **one segment of a channel**, not an LP. Each
of the three channels through it has a real LP at both ends:

| Channel | Registered by |
|---|---|
| `traffic->frontend-P:http` | traffic, when it posts |
| `frontend-P->frontend-D:relay` (prefill's JSON out, the decode request in) | frontend-P, when it writes prefill's JSON |
| `frontend-D->traffic:stream` | frontend-D, per streamed event |

The router's per-request overhead is part of these channels' lookahead. The wall time
it really takes only lengthens a message's time in flight, which the `(channel, seq)`
counting covers (D3, D8). FIFO is not required: concurrent requests through the router's
tokio tasks can arrive out of order, and messages are released by `(channel, seq)`.

This holds only while the router sees one worker per role. A `--dp-aware` launch, which
CI uses with DP attention (`pd_server_atom.sh:649-655`), registers every DP rank as a
separate worker, and the policy then chooses among them; that is the xPyD case below.

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
   per admission double-counts concurrent arrivals). The delay is the lookahead of the
   three channels above.
4. Atomesh's routing **decision** is business logic and is reused unchanged. Only its
   latency is modelled.
5. **Timestamps ride carriers the router already forwards**, so the router is not
   changed. Verified against `http_pd_router.rs` and `openai-protocol 1.0.0`, the crate
   it pins:

   | Channel | What the router does | Carrier |
   |---|---|---|
   | `traffic->frontend-P:http` | Parses the body into a typed request and re-serializes it. `CompletionRequest` keeps unknown fields (`#[serde(flatten)]`, `completion.rs:146`); `ChatCompletionRequest` has no such field (`chat.rs:151`), so **an unknown top-level chat field is dropped**. Request headers are forwarded by allow-list, `tracestate` included (`header_utils.rs:51-63`), to decode as well. | The `compass` entry of the W3C `tracestate` header, e.g. `tracestate: compass=a:12.345;s:17`, appended after any existing entry so real tracing is unaffected. The same for chat and completion. |
   | `frontend-P->frontend-D:relay` | Takes `kv_transfer_params` from prefill's JSON, lets the ATOM adapter insert fields, and writes it into the decode body (`http_pd_router.rs:1067-1114`). | A field inside `kv_transfer_params`. frontend-D also receives the original `tracestate`, so a request carrying `kv_transfer_params` takes its stamp from there (relay channel), any other from `tracestate` (traffic channel). |
   | `frontend-D->traffic:stream` | Passes the ATOM stream through byte for byte: `create_streaming_response` (`http_pd_router.rs:1587`) rewrites only with `return_logprob` and prefill logprobs, and the ATOM path has neither. | An SSE comment line before each event, e.g. `: compass a=12.345 s=18`. SSE clients ignore lines starting with a colon. |

   A single deployment (M1-M3, no router) uses the same carriers.

**xPyD is deferred.** With several instances per role, or `--dp-aware`, the policies
this router ships read state that changes over time — `power_of_two` and `cache_aware`
compare `load()`, which moves as requests start and finish; `random` draws from an
unseeded generator; `dp_sticky` reads `Instant::now()` (`atom/mesh/src/policies/`). A
component whose time-varying state decides its output is an LP, so the router would have
to become one, instrumented in its own process with the policy code untouched. That
design is not done here; M4 stays 1P1D.

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
before time T."** No LP it feeds can be granted past `T` plus its lookahead distance
from traffic (D3), so none can miss an arrival.

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

This is not special to arrivals: it is the general **transient message**, sent but not
yet received, and the send/receive counting of D3 already covers it (PDES term:
Fujimoto/Mattern message counting under an HLA time-advance grant). The traffic LP
registers each POST as a `(channel, seq)` with its arrival time, reported with its next
advance request; a grant past that time carries the `(channel, seq)` set it releases,
and the receiver does not return from the grant until each has arrived. The frontend's
send to the engine is counted the same way, and a frame is held at the receive wrapper
until its grant, so ATOM never sees a request before its arrival time. There is no
enqueue acknowledgement, and **closed and open-ended workloads are handled alike**.

The arrival time rides the `compass` entry of the `tracestate` request header (D7). take2
put it in a body field, `CompletionRequest.compass_arrival`, which the router drops from a
chat request because the typed chat request keeps no unknown top-level field.

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

Every item is additive at an existing site; no process and no thread is removed (D1).

1. **`atom/utils/clock.py`** — an injectable clock. take2 designed this in 148 lines:
   a `Clock` protocol, `WallClock`, `VirtualClock`, `get/set/reset_clock`. One detail
   worth preserving: `WallClock.epoch` returns `None`, deliberately not `0.0`, because a
   caller offsetting from the Unix epoch would get a 1970 timestamp and a duration in the
   billions; `epoch is None` then becomes the engine-wide discriminator for "real clock,
   none of this applies". Add an `AuthorityClock` that talks to the CA, and with it the
   LP runtime and the CA's side of D3: the send log, the expected counts, the strict
   grant, and walking pending arrivals in order. **~550**
2. **Clock-read substitution** at the K2 and K7 sites of D4 and D5, including the
   `_last_refresh` stamp in `AtomMetricsExporter.update`
   (`atom/entrypoints/openai/metrics.py`, `11` D72). take2 did four in `scheduler.py`.
   **~30**
3. **TAR and NER hooks.** `EngineCore._process_engine_step_inner` calls
   `advance_to(now + predicted)` after the forward; take2's `_advance_clock_for` is
   this hook. The predicted duration rides back on a new field of
   `ScheduledBatchOutput`, which is `None` on a real run so the advance is a no-op and a
   real run is untouched. The same at the other forward sites and the idle rank's
   dummy batch (K1).
4. **`Scheduler._advance_to_next_arrival`**, which does not exist, is not added: the step
   loops' spin and the offline `get_output` become `next_event(t)` (K3), and the CA
   performs the jump. Items 3 and 4 together: **~100**
5. **Socket wrappers** at the channel creation points, one line each. The message
   header is (channel, arrival time, sequence number); on a ROUTER socket it goes after
   the identity frame (K4, K5). **~150**
6. **The relay queue**, one line replacing `output_queue` in `EngineCore.__init__` (K4).
   **~80**
7. **Poller wrappers** where the receiving threads create their pollers,
   `EngineCore.process_input_sockets` and `CoreManager._create_output_thread` (K5).
   **~40**
8. **Frontend virtual time**: the stdlib event loop class and its selector hook (D5.1);
   the tokenizer thread pool as a resource station and detokenization as one per output
   thread; wrappers on `tokenizer.encode` / `decode`; an HTTP middleware that counts and
   releases requests (`06` D33). **~350**
9. **Configuration** for the K8 bounds of D5 — atomesh's two disable flags and
   `--worker-request-timeout-secs` set large, no Rust change — and uvloop off (D5.1).
   **~20**
10. **The process-group assertion** (D4): a collective group never spans two LPs.
    **~30**
11. **`SimulatedKVConnector`** registered through the existing factory (D6). **~80**
12. **Control-command refusal** in the simulated runner's worker methods that the refused
    commands reach (D5). **~40**
13. **The metrics observer and the end of a run**: a periodic `/metrics` scrape in the
    traffic LP, the engine's push gate on a virtual timer (D5, `11` D72), and the CA's
    end-of-workload request (D3). **~60**
14. **A simulated `ModelRunner`** injected via `--runner-qualname` (no ATOM change needed
    for the injection itself).

### Size estimate

Items 1-13 come to ~1,530 lines, and the tests below to ~450: **~1,980 lines**, against
the ~1,200 this section estimated while the change set was status annotation at the
category-B sites. take2's whole clock integration was **+148 new lines and ~771 modified
across 11 files, with 13 deletions** — almost pure addition, no import-time
monkeypatching of any ATOM class.

The tests, all CPU-only:

- **In-transit messages**: two fake LPs with 5 ms of added transport delay; the drain at
  10.2 sees a message sent at 9.6, and two runs choose the same batches.
- **TSO delivery**: a handler thread reads its message's arrival time as the clock, and
  `schedule()` never sees a message from its future.
- **Relay queue contract**: three puts are all sent, received in order, each stamped with
  the LP time of its `put`; an output thread made to skip one raises `UnsentRelayItem` on
  the next `get()`; a `put` from another thread raises.
- **Inventory completeness**: `test_sync_inventory.py` extended so every row carries a
  mechanism consistent with its peer and bound fields.
- **Determinism** (D3.4): two runs of one configuration, byte-diffed step tables.
- **PD release**: two fake engine LPs and the simulated connector, with decode admission
  delayed: prefill frees its KV blocks at the write's completion and decode is ready a
  notify latency later (D6); prefill finishing after the write request arrives raises.
- **I2 guard**: an AST test that no `advance_to` / `next_event` call sits lexically inside
  a `with ...lock` block, so a clock owner never parks in the CA holding a lock another
  thread in its LP needs.

### The enforcement mechanism

Clock-site completeness **cannot be verified by reading.** take2 solved the narrow
version with an AST-walking test (`tests/compass/test_serving_timings.py`) asserting that
no assignment to `first_token_time` / `finish_time` / `arrive_time` in `scheduler.py`
calls `time.*`.

Generalize it: a test that walks the serving path and **fails on any un-allowlisted
`time.time` / `time.monotonic` / `time.perf_counter`**, where the allowlist is exactly
what D5 leaves on real time. That test is what makes this design safe rather than
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
| D3 | Central Clock Authority acting as the HLA RTI (`advance_to` = TAR, idle point = NER, grant = TAG); an LP is a process group with one clock owner, and zero-lookahead couplings collapse into one LP; a grant is strictly below the lookahead-distance LBTS and waits for the messages already in transit (Fujimoto counters) | 2026-09-18; revised 2026-09-28 |
| D3.1 | Single CA with a hierarchy-ready interface; LP count scales with replicas and PP stages, not with GPUs | 2026-09-18 |
| D3.2 | Always-on causality detectors: a straggler check against the receiver's last drain (fails the run); a stall the CA cannot see prints one diagnostic after 30 wall seconds and keeps waiting, never aborting; clock-source CI lint | 2026-09-19; revised 2026-09-28 |
| D3.3 | CA deploys two ways from one implementation: co-hosted in the API-server process by default, standalone server via `--compass-clock-endpoint` for M4/M6 multi-container runs | 2026-09-19 |
| D3.4 | Wall-clock interleaving may vary between runs; the `(LP, virtual time, event)` sequence may not. Thread scheduling inside an LP is irrelevant only under in-transit counting and TSO delivery; ties break by **(time, LP id, channel, sequence number), never arrival order**; the cost backend is a pure function of its batch view; **no `set` iteration on the simulated path** - string ids are hashed under `PYTHONHASHSEED` randomisation, so a set of request ids iterates differently in every process. Test is a byte-diff of two step tables, CPU-only, in CI. | 2026-09-20; revised 2026-09-28 |
| D3.5 | The CA owns three outputs: an opt-in timeline log, a stall diagnostic (a stall the CA can see is resolved by deadlock detection and recovery; one it cannot see gets a diagnostic, never an abort), and an always-written run summary carrying grants, speed ratio, lazy-trace cost, detector state and the refusal fractions `08` D50.1 gates on. | 2026-09-20; revised 2026-09-28 |
| D4 | Every synchronization site maps to one of the PDES mechanisms K1-K9, none undecided; the boundary is the LP, not the process; a DP step is one compound event priced by its per-layer critical path; handler threads stay and TSO delivery runs them at their message's timestamp | 2026-09-18; revised 2026-09-28 |
| D5 | Every timer and clock read the CA can reach runs on the LP clock; only what it cannot reach is configured or left real; metric cadence is virtual time (revising `11` D72); the profiler and RL control commands are refused by the simulated runner and counted | 2026-09-18; revised 2026-09-28 |
| D5.1 | The simulated run uses stdlib asyncio, not uvloop: only a Python event loop can host virtual time | 2026-09-28 |
| D6 | KV transfer is simulated through a connector registered in the existing factory, reproducing Mooncake (what ATOM's PD CI deploys) through scheduler-side hooks only | 2026-09-18; revised 2026-09-28 |
| D7 | Atomesh gets no virtual clock: mesh-only mode; in 1P1D the router is one segment of a channel, not an LP, and its relay latency is that channel's lookahead; its bounds are configured (health check and circuit breaker off, `--worker-request-timeout-secs` set large); timestamps ride carriers it already relays; xPyD is deferred | 2026-09-18; revised 2026-09-30 |
| D8 | Arrivals are messages in transit, covered by the send and receive counts for closed and open workloads alike; the arrival timestamp rides the `tracestate` header | 2026-09-18; revised 2026-09-28 |
| D9 | The ATOM diff is enumerated (~1,980 lines with tests) and completeness enforced by test rather than by review; ATOM's own suite is the gate (`08` D43.1) | 2026-09-18; revised 2026-09-28 |

---

## Appendix: verified reference points

Facts this design leans on, with their source, so a later reader can re-check rather than
re-derive.

**The seam**
- `Config.runner_qualname` — `atom/config.py:1595`; consumed `engine_core.py:128`,
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
