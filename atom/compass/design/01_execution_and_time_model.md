# ATOM Compass — Design Point 1: Execution and Time Model

**Status:** draft for review. Drafted by an AI assistant during a design interview; not
yet reviewed or approved. No code has been written against it.

**Branch:** `feature/atomcompass_new` (clean fork of upstream `main` at `0b4f1ddb`).

**Scope of this document.** The execution architecture: what a simulated run *is*,
which processes exist, how virtual time advances, and what contract the rest of the
system must honour. It does **not** cover the cost model, the memory model, model
capture, or the workload harness. Those are separate design points.

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

## D0. Relationship to the two prior branches

### Problem

Two prior attempts exist in the repository. Restarting on a clean branch raises the
question of what, if anything, is inherited.

Verified facts:

| Branch | Commits over `fork/main` | Diff | Note |
|---|---|---|---|
| `feature/atomcompass_take2` | 177 | 75 files, +23,878 / -13 | The PoC baseline. Head `66ae9d87`. Equals GitHub PR jgong5/ATOM#2. |
| `feature/atomcompass` | 527 | 263 files, +118,921 / -37 | **Not** an abandoned earlier attempt — `git merge-base --is-ancestor take2 atomcompass` is true. It is take2 **plus 350 more commits**. |

So the history is linear: take2 is a deliberately pruned baseline, `feature/atomcompass`
is its continuation.

### Options

1. **Rebase on take2.** Fastest to a running thing. Carries a wall-clock-shaped
   architecture into a virtual-clock design.
2. **Parts bin.** Design fresh; port only what the new design names.
3. **Pure from-scratch.** Reuse nothing.

### Decision

**Fresh design, referring to the prior work at the level of *design* only.** No
code-port plan is defined up front. Where a prior mechanism is the right answer it is
described here on its merits and re-derived; where it is not, it is not carried.

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
- **The idle jump is where the speedup comes from.** 64 requests Poisson 8/s: 9,426 ms
  real vs 639 ms simulated (14.8x). cc-traces 20 requests 27B TP=4: 309 s vs 3 s
  (~103x). Under saturation, where there is no idle to skip: **122 s vs 36 s, i.e.
  0.30x — slower than the system it simulates.**
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

### Open issues

- The two handed-off artifacts (`atom/compass/DESIGN_NOTES.html`, `POC_SUMMARY.html`)
  disagree about which is newer. Content settles it: DESIGN_NOTES is newer;
  POC_SUMMARY's cc-traces and HTTP-serving rows are stale by roughly five successive
  states. Anyone reading POC_SUMMARY alone inherits an out-of-date picture.
- Nine further design documents referenced by those two (`RETROSPECTIVE.md`,
  `PROTOCOL.md`, `CC_TRACES_PROTOCOL.md`, `POC_STATUS.md`, `G4_TRANSFER.md`,
  `MEMORY_EVIDENCE.md`, `DECODE_CALIBRATION_SCOPE.md`, `SOURCE_ONLY_SERVING.md`,
  `PREFIX_CACHE_REPLAY.md`) exist only on `feature/atomcompass`. They are readable via
  `git show`, and several are load-bearing.

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

### Open issues

- Simulation wall-clock cost is higher than Option A's. The >=5x target is stated as
  negotiable with a bottom line of "faster than real runs". Under saturation the prior
  design was 0.30x. This must be measured early, not assumed.
- `torch.cuda.set_device` (`model_runner.py:958`) and `torch.cuda.mem_get_info`
  (`model_runner.py:1659`) are the two hard GPU dependencies a simulated runner must not
  inherit.

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
| TP group | 1 EngineCore + N workers | **1** | Workers are slaved by a blocking RPC (`async_proc.py:439`) and hold no clock. Rank-0 authority validated at 0.06% (D0). |
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
wide, because its workers are slaved by a blocking RPC (`async_proc.py:439`) and hold no
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

### Open issues

- Grant RPC latency has not been measured on this hardware. The 50 us figure is an
  estimate and should be measured before it is quoted.
- Whether the CA lives in the API-server process or standalone for the M4 two-container
  case is unresolved. Standalone is probably cleaner there, since neither container is
  obviously the right host.
- The CA is the only component that knows every LP's virtual time. It should therefore
  own the global timeline log and the deadlock dump. That makes it an observability
  component as well as a coordination one, which is a benefit but also means its output
  format is part of the acceptance evidence and should be designed, not improvised.

---

## D4. The interception contract: which waits must be touched

### Problem

ATOM's serving path contains roughly 55 distinct synchronization points: blocking ZMQ
recvs, bounded pollers, queue gets with timeouts, Gloo and NCCL collectives,
`multiprocessing` barriers and joins, busy-waits, and literal sleeps. "Intercept every
blocking call" is the obvious reading of what a virtual clock demands. It is also the
reading that turns this project into reimplementing a scheduler on top of the OS
scheduler.

### The contract

A wait matters to virtual time **only if its duration is observable in the simulated
result.** That yields four categories.

| Cat. | What it is | What you do | Approx. count |
|---|---|---|---|
| **A** | Wait whose duration **is** modelled time | **Rewrite.** Do not wait — `advance_to(now + d)` and continue. | ~6 |
| **B** | Wait for a message another LP will send | **Annotate only.** `declare_blocked()` / `declare_running()` around the existing call. Leave the call itself alone. | ~10 |
| **C1** | Timeout that is a failure detector | **Disable or raise.** No virtual semantics needed. | ~20 |
| **C2** | Timeout that is pacing | **Virtual timer.** Declare next event at `now+d`; keep a short real poll so the thread stays responsive. | ~3 |
| **—** | Wait *inside* one LP; startup; shutdown; OS-level | **Ignore.** Invisible to modelled time. | ~20 |

#### Category A — the short list

- the forward pass (`engine_core.py:386-388`)
- KV-transfer completion (D6)
- the arrival gate (D8)
- `Scheduler._passed_delay` / `--scheduler-delay-factor` (`scheduler.py:3108-3126`)
- `Scheduler._oldest_waiting_prefill_age_ms` feeding `PrefillDelayer` (`scheduler.py:1194`)
- the idle jump (`Scheduler._advance_to_next_arrival`, replaced by `declare_next`)

#### Category B — annotate, do not intercept

The important instance: **`call_func(..., wait_out=True)` (`async_proc.py:439`) is not
intercepted.** It has no timeout and every forward goes through it. Two lines of status
annotation are enough, because its duration was already charged by the *sender* when it
called `advance_to`. The receiver is merely idle in wall time; the CA routes grants
elsewhere meanwhile.

Same treatment for `engine_core.py:546` (the CoreManager poller, no timeout argument),
`engine_core.py:587`, `engine_core_mgr.py:534/544/576/586`, and the RapidServe recvs
if that path is ever used.

#### Why I4 is safe rather than merely careful

An LP that forgets to declare `blocked` leaves the CA waiting for a grant response that
never arrives. That is a **hang**, which is loud. The opposite failure — advancing past
a message — is silent. The protocol is deliberately arranged so that mistakes fall on
the loud side.

### The case that is not a "wait" problem at all

`_recv_prefill_done` (`engine_core.py:1201`) blocks in a background thread and calls
`Scheduler.on_prefill_done`, which stamps `seq.first_token_time = time.time()`
(`scheduler.py:3357`). Intercepting the block fixes nothing. The rule is about who
stamps:

> **Background threads enqueue. The main loop stamps, at a granted time.**

ATOM's queueing is already correct here (`_pending_assignments` under
`scheduler._pending_lock`; the `prefill_done` deque). Only the stamping moves.

### Open issues

- The counts above are estimates from a synchronization inventory, not from a completed
  pass over the code. The first implementation task should be to produce the exact
  classified list and check it in.
- `engine_core.py:1156` is a literal `time.sleep(2)` whose purpose is GPU-allocator
  settling after weight-handle import. It must stay on real time. It is RapidServe-only
  and Compass loads no real weights, so it is probably moot — but it must be *checked*,
  not assumed.

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
ModelRunner.async_proc_aggregation      model_runner.py:3351-3372
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
  `AtomAdapter` works unmodified. The real shape is at `moriio_connector.py:970-1001`:
  `{do_remote_prefill, remote_block_ids, remote_engine_id, remote_host, remote_port,
  remote_handshake_port, tp_size, dp_rank, transfer_id, first_token_id,
  draft_token_ids, prefix_cache_hit_tokens}`. The router hard-errors if it is absent
  (`http_pd_router.rs:1073-1078`).
- The consumer side must still return `(len(prompt), True)` from
  `get_num_new_matched_tokens` when `do_remote_prefill` is set, i.e. park the request
  (`moriio_connector.py:904-917`), so `Scheduler._park_for_remote_load`
  (`scheduler.py:2207-2212`) and the `WAITING_FOR_REMOTE_KVS` state behave identically.

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
workload design point, but it is recorded here because it is what licenses the
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
| D4 | Four-category interception contract; annotate rather than intercept cross-LP blocking waits | 2026-09-18 |
| D5 | Disable failure detectors, virtualize business logic; sorting rule is "does it change which batch gets scheduled" | 2026-09-18 |
| D6 | KV transfer is simulated through a connector registered in the existing factory | 2026-09-18 |
| D7 | Atomesh gets no virtual clock: mesh-only mode, detectors disabled, relay latency declared | 2026-09-18 |
| D8 | Arrivals via the CA as a next-arrival lower bound; the take2 arrival barrier is replaced | 2026-09-18 |

---

## Appendix: verified reference points

Facts this design leans on, with their source, so a later reader can re-check rather than
re-derive.

**The seam**
- `Config.runner_qualname` — `atom/config.py:1595`; consumed `engine_core.py:129`,
  `async_proc.py:166-169`
- `ModelRunner.forward(batch: ScheduledBatch) -> ScheduledBatchOutput` —
  `model_runner.py:3233-3320`
- the RPC boundary — `engine_core.py:386-388`
- `ScheduledBatch` fields — `scheduler.py:579-820`; notably `detailed_sqsq` /
  `detailed_sqsk` / `detailed_sk` at `:790-792`, which are sum(N_Q^2), sum(N_Q * N_KV),
  sum(N_KV) per batch, computed by `compute_detailed_aggregates` (`:2788-2842`) and
  currently gated on `profile_active and ATOM_ENABLE_DETAILED_ANNOTATION`
- `ScheduledBatchOutput` — `scheduler.py:841-885`; `produces_output()` at `:823-840`

**Existing simulation-shaped hooks in ATOM**
- `--load_dummy {empty,zero,xavier}` — `config.py:1556`, `arg_utils.py:260`,
  `loader.py:179-227,309-310`, `loading_core.py:266-291`
- meta-device model construction — `RapidServeModelRunner._init_weight_params_on_meta`,
  `model_runner.py:4188-4211`
- a working non-allocating runner template — `RapidServeModelRunner` overrides at
  `model_runner.py:4218,4232,4239,4245,4261,4269`
- `ModelRunner.dummy_execution()` — `model_runner.py:1177-1217`, shows how to hand-build
  a `ScheduledBatch`
- `ScheduledBatch.is_dummy_run` — `scheduler.py:589,781`
- simulated TP (`--fake-eplb`) — `atom/distributed/simulated_tp.py`; explicit precedent
  for a shape-accurate, value-meaningless run
- synthetic speculative acceptance — `config.py:1064-1190`,
  `atom/model_ops/rejection_sampler.py:20-224`; the closest existing behaviour simulator
- profiler label taxonomy — `atom/model_engine/run_labels.py`; consumed by
  `tools/parse_trace.py`

**Memory sizing (needed because it decides which configurations exist)**
- `ModelRunner.get_num_blocks()` — `model_runner.py:1652-1873`. Five device readings plus
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
