# ATOM Compass — Design Point 5: The Machine Specification and its Probes

**Status:** draft for review. Drafted by an AI assistant during a design interview; not
yet reviewed or approved. No code has been written against it.

**Depends on:** `03_memory_and_kv_model.md` D14/D15 (which named the artifact), and feeds
`04_model_capture_and_cost_ir.md` (cost) and `01_execution_and_time_model.md` D3
(lookahead floors).

**Scope.** The single input artifact that tells Compass what machine it is simulating,
its schema, and the tools that fill it in. A project requirement is that compute, memory
size, bandwidth and interconnect are **configured and not read from a device runtime**, so
this artifact is load-bearing rather than convenient.

---

## D24. The separation rule

### Problem

A simulated run needs numbers describing a GPU, a host CPU, a fabric, and a software
stack. Some of those are properties of hardware; some are properties of the deployment
ATOM was launched with. Mixing them produces a spec that silently contradicts the engine
it is meant to describe.

### Decision

> **The machine spec describes the machine. ATOM's config describes the deployment.
> Anything ATOM could configure belongs in ATOM's config, and Compass reads it.**

Consequences in both directions:

**Stays out of the spec, because ATOM already owns it:** `gpu_memory_utilization`,
`max_num_seqs`, `max_model_len`, `kv_cache_dtype`, `block_size`,
`cudagraph_capture_sizes`, `tensor_parallel_size`, compilation level, cudagraph mode.

**Stays out of the spec, and ATOM does not own it yet — so ATOM should.** Verified on
`feature/atomcompass_new`: there is **no thread configuration anywhere** in `atom/`. No
`set_default_executor`, no `ThreadPoolExecutor`, no `max_workers` under `entrypoints/` or
`model_engine/`; `arg_utils.py` has no thread or worker argument; and
`EngineCore.output_thread` / `input_thread` are hardcoded singletons
(`engine_core.py:93`, `:105`). Tokenization runs on Python's *implicit* default executor
via `await loop.run_in_executor(None, do_preprocess)` (`api_server.py:890`, `:1004`,
`:1126`, `:1258`, `:1480`), whose width is `min(32, cpu_count + 4)` and is named nowhere.

| Add to ATOM | Why |
|---|---|
| `--preprocess-pool-width`, applied via `loop.set_default_executor(ThreadPoolExecutor(max_workers=N))` | an unnamed implicit default cannot be tuned or reproduced; this is useful to ATOM independent of Compass |
| expose the resolved value on `Config` | Compass reads it rather than re-deriving `min(32, cpu+4)` |

The engine output-thread count stays at 1 — changing it changes ordering semantics — but
it must be **readable**, not inferred.

### Open issues

- Whether `--preprocess-pool-width` belongs on `EngineArgs` or is server-only. It is
  consumed in the API-server process, so server-only is defensible.

---

## D25. Schema

One artifact, four sections, authored outside Compass and **echoed verbatim into every run
artifact**.

```yaml
schema_version: 1
name: mi355x-8gpu-2node

provenance:                      # one block, not per-field tags
  authored_by: <who>
  date: 2026-09-18
  method: datasheet | probed | transferred-from:<spec-name> | mixed
  fragments: [...]               # filled by `merge`, see D26
  notes: free text

host:
  cpu:
    cores_physical: 96
    cores_logical: 192
  tokenizer:                     # per (tokenizer, host) PAIR - re-probe if either changes
    encode_fixed_s:            3.0e-4
    encode_tokens_per_s:       2.0e6
    decode_fixed_s:            1.5e-4
    decode_tokens_per_s:       3.0e6
    derate:                    0.85
  ipc:
    zmq_roundtrip_s:           5.0e-5
    shm_broadcast_s:           2.0e-5
  admission_fixed_s:           9.0e-3   # HTTP arrival -> `waiting`, EXCLUDING tokenize

device:
  name: MI355X
  arch: gfx950
  count_per_node: 8
  memory:
    capacity_bytes:            288.0e9
    bandwidth_bytes_per_s:     8.0e12
    derate:                    0.85
  compute:
    bf16_flops:                2.5e15
    fp8_flops:                 5.0e15
    derate:                    0.70
  runtime_constants:           # the "table, not a law" terms, keyed by TP width
    non_torch_bytes:     {1: 970.0e6, 2: 7.2e9, 4: 7.6e9, 8: 11.2e9}
    load_residue_bytes:  {1: 1.1e6,   2: 2.17e9, 4: 2.17e9, 8: 2.17e9}
    persistent_bytes:    124.0e6
    graph_pool:
      w1_floor_bytes:                  95.5e6
      w1_per_captured_token_bytes:     0.318e6
      w_gt1_bytes:                     109.0e6
  software_pinned_to:          # these constants belong to the stack as much as the silicon
    rocm:  "7.2.4"
    aiter: "<commit>"
    rccl:  "<version>"

interconnect:
  intra_node:
    topology: fully_connected
    link_bandwidth_bytes_per_s: 1.0e12
    link_latency_s:             2.0e-6
    derate:                     0.80
  inter_node:
    link_bandwidth_bytes_per_s: 5.0e10
    link_latency_s:             5.0e-6
    derate:                     0.80
  router_relay_s:               1.5e-3   # Atomesh per-hop, PD only
```

### Why each term exists

| Term | Consumer | Evidence it is needed |
|---|---|---|
| `memory.capacity_bytes` | `get_num_blocks` `total` | the one reading that cannot be derived |
| `runtime_constants.non_torch_bytes` | `get_num_blocks` `non_torch` | 926 / 6906 / 7266 / 10704 MiB at widths 1/2/4/8. **No fixed-plus-per-peer form fits** 5980 / 6340 / 9138. A table, not a law. |
| `runtime_constants.load_residue_bytes` | `peak_torch` | AITER `CustomAllreduce` 1 GiB pool plus the two-stage kernel's: 1.1 MiB at TP1, **2069 MiB flat** at TP2/4/8 |
| `runtime_constants.persistent_bytes` | `peak_torch` | `allocate_forward_vars` + attention metadata; ~118 MiB, flat in width, differs only by model |
| `runtime_constants.graph_pool` | `cudagraph_overhead` | measured `91.1 MiB + 0.3033 MiB per captured token` at W=1; flat **104 MiB** above W=1, where the allocated delta was byte-identical (79,692,800) across three widths and three ladders |
| `tokenizer.*` | admission queue | cc-traces p50 input is **88,768 tokens**; at 2 M tok/s that is ~44 ms, **3x take2's entire admission constant**, and it scales with prompt length while a constant does not |
| `ipc.*`, `interconnect.*_latency_s` | **Clock Authority lookahead floors** (doc 01 D3) | a zero lookahead serializes the whole simulation |
| every `derate` | cost model | no kernel reaches spec peak; the gap between datasheet and achievable is the user's to declare |

### Schema rules

1. **No defaults for `runtime_constants`.** A missing TP width **refuses and names it**.
   These are the terms with no law behind them; a silent default is the worst kind of
   wrong.
2. **`derate` is mandatory wherever a spec-peak number appears**, so nobody can quietly
   use a datasheet FLOP as an achievable one.
3. **`software_pinned_to` is checked, not decorative.** The plausible reading of the
   evidence is that these constants track the ROCm/RCCL/AITER build more than the die —
   the +5980 MiB at width > 1 is collective buffer sizing, the 926 MiB at TP1 is HIP
   context plus libraries. A stack mismatch warns loudly.
4. **The whole resolved spec is echoed into every run artifact.** The KV gate is **≤5%**;
   a number whose spec cannot be recovered from the artifact is unattributable.

### Open issues

- **Whether the runtime constants transfer across devices is untested.** If the reading
  above is right they transfer across cards of one software generation and **not** across
  software upgrades. That is a testable claim and nobody has tested it.
- Three topologies of one model is interpolation, not a law. The graph-pool width scaling
  rests on **one** point above W=1.
- `host.tokenizer` is keyed by neither model nor tokenizer in this schema. If two models
  with different tokenizers are compared on one spec, the terms silently apply to both.
  Either key them, or require one spec per model.

---

## D26. Probe tools

### Problem

The schema asks users for numbers most of them cannot obtain by reading a datasheet — the
runtime constants in particular are measurable only by starting an engine. Shipping a
schema without the tools to fill it is shipping a blank form.

### Design

A probe **emits a fragment**: a partial spec plus a provenance stanza saying what it
measured, how, and on what. `merge` combines fragments into a spec, recording which
fragment supplied each field. `validate` refuses an incomplete or inconsistent one.

```
compass spec probe <what>  -> fragment.yaml
compass spec merge f1 f2 ... -> machine.yaml
compass spec validate machine.yaml
compass spec explain machine.yaml --term kv_blocks
```

### The probes, by what hardware they need

**Tier 0 — no GPU. Runs anywhere, including the CPU container.**

| Probe | Fills | Method |
|---|---|---|
| `tokenizer --model M` | `host.tokenizer.*` | encode and decode a length sweep with **ATOM's own loaded tokenizer** (`_load_tokenizer`, `llm_engine.py:23`), fit `a + b*n`. Must sweep past 200k tokens: the workload's p90 input is 204,288. |
| `ipc` | `host.ipc.*` | round-trip over ATOM's own `make_zmq_socket` and `aiter.dist.shm_broadcast.MessageQueue`, so it measures the transports ATOM actually uses |

**Tier 1 — one GPU of the target type.**

| Probe | Fills | Notes |
|---|---|---|
| `device-memory --tp 1` | `capacity_bytes`, `non_torch_bytes[1]`, `persistent_bytes`, `graph_pool.w1_*` | start an engine, read the five readings. Sweep >=5 capture ladders for the graph-pool fit; the prior fit predicted a held-out sixth at **+6.4%** |
| `device-compute` | `bandwidth_bytes_per_s`, `*_flops`, their `derate`s | GEMM and bandwidth sweeps; derate is achieved/datasheet |

**Tier 2 — N GPUs of the target type.**

| Probe | Fills | Notes |
|---|---|---|
| `device-runtime-constants --tp 2,4,8` | `non_torch_bytes`, `load_residue_bytes`, `graph_pool.w_gt1_bytes` | one engine start per width |
| `interconnect-intra` | `intra_node.*` | collectives in **real groups**, several sizes |

**Tier 3 — two nodes.**

| Probe | Fills |
|---|---|
| `interconnect-inter` | `inter_node.*`, `router_relay_s` |

**Tier 4 — no hardware at all.**

| Probe | Fills | Provenance stamped |
|---|---|---|
| `from-datasheet` | `capacity_bytes`, `bandwidth`, `flops` | `method: datasheet`, derates left blank so `validate` refuses until declared |
| `transfer --from <spec>` | `runtime_constants.*` | `method: transferred-from:<spec>`, carrying **both** specs' `software_pinned_to` so a later mismatch is visible |

### Two refusals every hardware probe must implement

Both come from failures that cost real time:

1. **`non_torch` is a device-wide reading.** It is `(total - free) - reserved`, and
   `total - free` counts every process on the card. Six prior runs died at start-up with
   `available_for_kv = -103667.58 MB (budget=57.60GB, peak_torch=2.94GB,
   non_torch=152.01GB, ...)` because a neighbour held 152 GB while the rank had reserved
   2.9 GB. **A probe must refuse a reading whose `non_torch` far exceeds what the
   collective terms predict for its width** — the case worth catching is 50x, not 50%.
2. **Rank disagreement is a direct, single-run measurement of contamination.** Ranks of a
   symmetric group do identical work, so a spread in `non_torch` across ranks needs no
   model to interpret: 0 MiB at widths 1-2, **192 MiB at width 4, 640 MiB at width 8**.
   Take the **minimum** across ranks and report the spread; refuse above a threshold.

Also carried over: a probe must record `free` and `total` **separately** and refuse a
reading where `free` was binding in `min(budget, free)` — otherwise the captured budget is
a property of which neighbours happened to be on the box, and the admission cliff moves
with them.

### `validate` refuses on

- a missing required field
- a TP width absent from `runtime_constants` that the deployment will use
- a `derate` missing beside a spec-peak number
- `software_pinned_to` not matching the running stack (warn, or refuse under a strict flag)
- a `transfer` fragment whose source spec pinned a different stack

### `explain`

Given a spec and a predicted quantity, print which spec fields contributed and by how
much. This is not a nicety: the KV gate is 5%, memory validation is **per term and never
as a sum**, and a prior summed check read **+13.8%** while hiding three errors two of
which cancelled — the largest being 25% of a single term.

### Open issues

- The tokenizer probe measures *this* host; if the simulated host differs from the probing
  host, a derate or a transfer is needed and there is no evidence yet on whether tokenizer
  throughput transfers across CPUs.
- Tier 2 and 3 probes need a quiet machine. Three of the last five prior pilot attempts
  were lost or degraded by other tenants — two refused to start with a negative KV budget
  at 141 GB of neighbour, one ran 64% slow. **Any probe run needs the machine checked
  before and after, not only before.**
- Nothing yet decides what a user does when they have no card of the target type and no
  spec to transfer from. Today the answer is "declare it and accept the error", which is
  honest but unguided.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D24 | The spec describes the machine; ATOM's config describes the deployment. Thread-pool width becomes an ATOM config option, not a Compass one. | 2026-09-18 |
| D25 | Four-section schema: `provenance`, `host`, `device`, `interconnect`. Mandatory derates, no defaults for `runtime_constants`, `software_pinned_to` checked, whole spec echoed into every artifact. | 2026-09-18 |
| D26 | Probes emit fragments; `merge` / `validate` / `explain` combine and check them. Four hardware tiers plus datasheet and transfer paths. Every hardware probe implements the contamination refusals. | 2026-09-18 |
