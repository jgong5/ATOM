# ATOM Compass — Design Topic 5: The Machine Specification and its Probes

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** `03_memory_and_kv_model.md` D14/D15 (which named the artifact), and feeds
`04_model_capture_and_cost_ir.md` (cost) and `01_execution_and_time_model.md` D3
(lookahead floors).

**Revised 2026-10-10 for `16` D95:** the schema holds what the MI455 roofline projection
needs: an FP4 matrix-core rate, the size of the domain the in-node fabric joins, and a
*sourced* or *assumed* status on every field (D25).

**Revised 2026-10-13 by the owner's ruling on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791):** the probes,
`merge`, `validate` and `explain` are the `spec` package's functions, run by per-machine
scripts, with no `compass` command (D26, `13` D80); `software_pinned_to` states torch, and
a step grid bundle's stack is checked against it (D25).

**Revised 2026-10-14 by the owner's
[ruling on #937](https://github.com/jgong5/ATOM/pull/937#issuecomment-6161726104), which
[#851](https://github.com/jgong5/ATOM/issues/851)'s decisions carry:** machine specs are
developers', one per supported machine, and a user passes one by `--machine`; the memory
readings of an engine start leave the spec for the `memory_readings` entries
`compass measure memory` takes per configuration (D25, D26, `07` D38).

**Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952):** a spec states where its values came from,
`provenance.origin`, not how they were produced, and each field's origin, with a probe's
subject, is recorded and kept across a save (D25); probes are grouped by probe class (D26).

**Revised 2026-10-16 by the owner's
[ruling on #1103](https://github.com/jgong5/ATOM/issues/1103#issuecomment-6211106002):** an
optional `host.http_intake` pair, the frontend event loop's charge per HTTP request taken in
(D25), filled by a class-1 `http-intake` probe (D26).

**Revised 2026-10-18 by [#1230](https://github.com/jgong5/ATOM/issues/1230):** the
`host.http_intake` pair is required, now that every committed spec states one probed by
`scripts/compass/http_intake.py`.

**Revised 2026-10-20 by [#1775](https://github.com/jgong5/ATOM/issues/1775):** an optional
`host.http_intake.idle_s`, the intake of a request that finds no other on the loop (D25),
filled by the `http-intake` probe from lone requests (D26).

**Revised 2026-10-16 by [#1226](https://github.com/jgong5/ATOM/issues/1226):** the schema
drops `device.activations` and the `transferred-from:<spec>` origin with the runtime
constants; the activation coefficient is a `memory_readings` reading (D25).

**Revised 2026-10-17 by the owner's ruling on
[#1392](https://github.com/jgong5/ATOM/issues/1392):** the analytical method reads
attainable peaks probed on the device: one compute rate per format, a bandwidth curve
`BW(bytes)`, an empty-kernel floor and link latency and bandwidth (D25), filled by the
class-1 and class-2 probes (D26). A `derate` is attainable over datasheet, probed; a device
with no hardware carries a probed device's, assumed.

**Revised 2026-10-18 by the owner's
[ruling on #1482](https://github.com/jgong5/ATOM/pull/1482#issuecomment-6248221485):** a
probed number carries the p10 and p90 its probe measured, beside its subject in
`provenance.origins`, and a spec that states one without them is refused by name before it
is written (D25 rule 5, [#1518](https://github.com/jgong5/ATOM/issues/1518)).

**Revised 2026-10-25 by the owner's ruling on
[#2138](https://github.com/jgong5/ATOM/issues/2138), and 2026-10-28 by the
[ruling on #2141](https://github.com/jgong5/ATOM/issues/2141#issuecomment-6467960187):**
MI455 has two specs. The A0 sample node's is probed there, a value not probed staying
assumed; the B0 performance machines' carries MI355X's probed derates, assumed, as the
frozen stage-3c projection took them, until a B0 node is probed (D25, rule 5).

**Revised 2026-11-03 by [#2716](https://github.com/jgong5/ATOM/issues/2716), which
supersedes [#2708](https://github.com/jgong5/ATOM/issues/2708), and the owner's
[ruling on #2724](https://github.com/jgong5/ATOM/pull/2724#issuecomment-6596632012):** the
machine spec splits by what each value varies with, into a device datasheet per SKU,
platform readings per (device, stack, PTL, torch) and a network file per scale-out
network, composed for a run; the host block is #2309's, and the router's hop an assumed
host transport constant (D24, D25). A measured readings entry holds attainable rates and
an assumed one another device's ratios, each composed with the datasheet as it stands
(D25 rule 2). `compass measure device` publishes one node's probe record and `compass fit
device` folds a platform's records into its readings, so a new node, stack or device
needs no developer and no per-machine script (D26). Each artifact kind keys the platform
axes its invalidation row marks, torch included, in place of the machine-spec digest
(`07` D41, D43).

**Revised 2026-11-03 by
[#2309's Ruling (e)](https://github.com/jgong5/ATOM/issues/2309#issuecomment-6498578963),
as amended that day, and the
[#2681 transition](https://github.com/jgong5/ATOM/issues/2309#issuecomment-6591319181):**
the host block leaves the spec. The tokenizer rates and the HTTP intake are terms of the
`host_laws` entry, keyed by model and fitted by `compass fit host` from what a `compass
record` server served (D25, `07` D38); `host.cpu`, `host.ipc`, the class 0 probes and the
`http-intake` probe go (D26).

**Scope.** The inputs that tell Compass what machine it is simulating, the layers they
split into, their schema, and the commands that fill them in. A project requirement is
that compute, memory size, bandwidth and interconnect are **configured and not read from
a device runtime**, so these inputs are load-bearing rather than convenient.

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
(`engine_core.py::EngineCore.__init__`). Tokenization runs on Python's *implicit* default executor
via `await loop.run_in_executor(None, do_preprocess)` (`api_server.py::generate_async`, `generate_async_multimodal`,
`generate_async_fanout`, `setup_streaming_request`, `setup_streaming_request_fanout`), whose width is `min(32, cpu_count + 4)` and is named nowhere.

| Add to ATOM | Why |
|---|---|
| `--preprocess-pool-width`, applied via `loop.set_default_executor(ThreadPoolExecutor(max_workers=N))` | an unnamed implicit default cannot be tuned or reproduced; this is useful to ATOM independent of Compass |
| expose the resolved value on `Config` | Compass reads it rather than re-deriving `min(32, cpu+4)` |

The engine output-thread count stays at 1 — changing it changes ordering semantics — but
it must be **readable**, not inferred.

### The second rule: a value lives where what it varies with is named

> **A value is stated once, in the layer that names everything it varies with and
> nothing else.**

One spec per node stated every value once per node: a SKU's datasheet peaks were copied
into each node's file, and a platform's readings were whichever node a developer probed.
The three MI355X nodes probe bf16 derates of 0.615 to 0.651 and pin two aiter commits
([#2708](https://github.com/jgong5/ATOM/issues/2708)), so a run answered for the SKU at one
node's value and stated no band across the others.

| Layer | Varies with | What it is | Written by | Read by |
|---|---|---|---|---|
| 1. device datasheet | the SKU | a file, `atom/compass/devices/<sku>.json`, named by SKU or by path | a person, from the vendor's datasheet | `fit device --assume-from` and every command that prices |
| 2. platform readings | device, stack, PTL and torch, its key | a `platform_readings` store entry, one per key, never replaced | `compass fit device`, from the probe records `compass measure device` publishes per node, or from another device's entry, assumed (D26) | every command that prices |
| 3. network file | the scale-out network | a file, `atom/compass/networks/<name>.json`, named by name or by path | a person, or the class 3 probe (D26) | a run across nodes |
| 4. host block | the host CPU, ATOM source, model, engine config, device, stack and torch, the host laws' key | the `host_laws` entry's `tokenizer` and `http_intake` terms (D25); the router's hop a host transport constant | `compass fit host`, from `compass record`'s records (`07` D38); the router's hop [#2714](https://github.com/jgong5/ATOM/issues/2714) | `run`'s tokenizer and HTTP intake charges (`06` D33) |
| 5. keys | — | the platform fields of every artifact key (`07` D41) | every publisher, from what its inputs read | every reader |

**Stack** here is ROCm's version, aiter's commit and RCCL's version, the matrix's
`ROCm / AITER / RCCL` axis (`07` D43); torch is an axis of its own.

**The keys (layer 5).** Each artifact kind keys, by value, each of device, software
stack, PTL state and torch that its invalidation row marks (`07` D43), read from the
records it was made from, where it stated the machine-spec digest of one spec file; its
other key fields stay as `07` D41 lists them. A host law then keys no PTL state, which
its row marks independent, and a kind whose row marks torch keys it, so a torch bump
publishes under a new key where a fit to the old one would refuse.

**A run's platform is read from the entries it prices with.** A user names the device and
its PTL state, `--stack`, which names a stack and a torch, only when the store holds
readings of several for them and the entries it prices with state none, and `--network`
for a run across nodes (`13` D80). The run composes the datasheet, the readings entry
and, across nodes, the network file into one machine spec, with the field paths of D25,
and echoes it into every run artifact (D25 rule 4). A command whose inputs carry their
platform, a GPU command from its cards and a fit from its records, names none: it keys
what it read.

### Open issues

- Whether `--preprocess-pool-width` belongs on `EngineArgs` or is server-only. It is
  consumed in the API-server process, so server-only is defensible.

---

## D25. Schema

Three documents, one per layer of D24, each holding the fields below at the paths shown,
and composed for a run into **one machine spec, echoed verbatim into every run artifact**.
A field sits in exactly one layer, and a layer document holding another layer's field is
refused by name. A readings entry's rates and ratios are the one change of path: the
composition turns them into the spec's derates (rule 2). The keys of layer 5 hold
`device.software_pinned_to` and `device.ptl`, which no document states any more: a
readings entry's key states its stack, torch and PTL state (`07` D41).

Every document states its own `schema_version`, `name` and `provenance`: who wrote it and
when, the document's origin, each field's origin with a probed field's subject and band
(rule 5), and notes. These describe the document, not the machine; the composed spec
states the three documents it was composed from.

```yaml
# 1. device datasheet, atom/compass/devices/mi355x.json
device:
  name: MI355X
  arch: gfx950
  count_per_node: 8
  memory:
    capacity_bytes:            288.0e9
    bandwidth_bytes_per_s:     8.0e12
  compute:
    bf16_flops:                2.5e15
    fp8_flops:                 5.0e15
    fp4_flops:                 10.0e15  # optional: absent where there is no FP4 matrix core
interconnect:
  intra_node:
    topology: fully_connected
    domain_devices:             8        # devices the fabric joins; may exceed count_per_node
    link_bandwidth_bytes_per_s: 1.0e12

# 2. platform readings, a measured store entry keyed (MI355X gfx950, rocm 7.2.4 /
#    aiter <commit> / rccl <version>, disabled, torch <version>)
device:
  memory:
    bandwidth_curve:                     # optional: bytes -> attainable bytes/s, fastest of read, write, copy; monotone
      - [4096,        1.1e11]
      - [4294967296,  6.4e12]
  compute:
    attainable_flops:          # the best large-GEMM rate, one per format
      bf16:                    1.75e15
      fp8:                     3.25e15
      fp4:                     6.0e15
  kernel_floor_s:              2.0e-6   # optional: an empty kernel, replayed in a graph
  # no memory readings, activations included: they are `compass measure memory`'s
  # entries (07 D38, Phase 2)
interconnect:
  intra_node:
    link_latency_s:             2.0e-6
    # link_setup_s: optional, once on a rank pair's first KV write; absent is zero
    attainable_bandwidth_bytes_per_s: 8.0e11   # a large peer copy
    pp: {rccl: {...}, gloo: {...}}       # optional: a pipeline stage's send per carrier
  host_link: {bandwidth_bytes_per_s, latency_s}   # optional: the device-to-host copy

# 3. network file, atom/compass/networks/<name>.json
interconnect:
  inter_node:
    link_bandwidth_bytes_per_s: 5.0e10
    link_latency_s:             5.0e-6
    # link_setup_s: optional, once on a rank pair's first KV write; absent is zero
    derate:                     0.80
```

`pp` holds, per carrier, `eager_threshold_bytes`, `latency_s`, `overhead_s`, `per_byte_s`
and `rendezvous_fixed_s`, and for `gloo` also `wake_s` and `wake_rise_s`
(`atom/compass/spec/fields.py::FIELDS`).

An assumed readings entry, for a device never run (D26), holds the source's ratios where a
measured one holds rates: `compute.derate` per format, `intra_node.derate`, and
`memory.bandwidth_derate_curve`, the source's curve over its datasheet's bandwidth. Its
other fields are as above.

A run across nodes reads the network file, and no other run does: today that is a
prefill-decode deployment whose two instances sit on two nodes, which prices its KV
transfers on `inter_node` (`atom/compass/drive/runfile.py::prefill_decode`). The router's
per-request hop, `router_relay_s`, is no network's: every prefill-decode deployment
charges it, one node or two. It is a host transport cost, layer 4's: an assumed constant
beside `atom/compass/drive/runfile.py::KV_WRITE_REQ_S`, among the host transport
constants [#2714](https://github.com/jgong5/ATOM/issues/2714) moves into host laws.

Layer 4, the host block, is two terms of the `host_laws` entry (`09` D54), beside its
segment laws, under the entry's key (`07` D41); no layer document holds them and the
composed spec has no `host` section:

```yaml
# 4. host laws: the tokenizer and HTTP intake terms of a host_laws entry
tokenizer:
  backend:     fast                     # fast (Rust) and slow (Python) differ by >10x
  fingerprint: sha256:<tokenizer.json>  # the file the timed calls loaded
  encode: {fixed_s: 4.0e-4, tokens_per_s: 3.6e5}        # one-item calls: encode, one-item encode_batch
  encode_batch: {fixed_s: 3.6e-4, tokens_per_s: 4.3e5}  # several-item encode_batch
  decode: {fixed_s: 2.8e-5, tokens: [0, 9]}             # no slope: fixed_s alone, inside tokens
http_intake:                  # the event loop's time to take in one HTTP request
  fixed_s:    4.3e-3          # per request of a burst
  per_byte_s: 5.6e-9          # per body byte
  idle_s:     1.1e-3          # optional: a request that finds no other intake on the loop
  bytes:      [3.0e4, 4.6e5]  # optional: the body bytes it was fitted over; outside, refused
```

A run reads them from the entry `--host-laws` names for its engine config (`13` D80). The
entry's `tokenizer` resolves against the tokenizer the server loaded: another backend
refuses, naming both, since the two run an order of magnitude apart on one
`tokenizer.json`; another fingerprint warns, naming both, since the rates are still that
model's tokenizer's. Each kind of call is optional and priced on its own: `encode`, a call of
one item, `encode` or a one-item `encode_batch`; `encode_batch`, a Rust-backend call of
several items, whose work the library's pool runs in parallel; `decode`, `fixed_s` and
`tokens_per_s`, or for calls whose time shows no slope in their tokens `fixed_s` alone,
priced only inside `tokens`, the `[lo, hi]` token counts it was fitted over: a decode
outside them is refused, naming them. **An entry with no `tokenizer` or no `http_intake`,
a `tokenizer` with no term for a kind of call, and a run with no host laws, charges that
term 0 s, and the run summary names the term, a kind as `tokenizer.<kind>`.** An `http_intake`
with no `idle_s` charges `fixed_s` to every request. Neither term carries a derate: the
entry is keyed by the host CPU it was fitted on and answers for no other (`09` D54). Why
every term stays keyed by model, and how `fit host` fits them, is `07` D38, Phase 1c.

### Why each term exists

Names are chosen to say what the quantity *is*, not where it was first observed. Several
of these feed ATOM variables whose own names are terser; the third column records that
mapping so a reader can follow the term into ATOM's code without guessing. The
`memory_readings` rows are fields of `07` D41's entry, not of the spec.

| Term | Feeds (ATOM name) | Evidence it is needed |
|---|---|---|
| `memory.capacity_bytes` | `get_num_blocks` → `total` | the one reading that cannot be derived |
| `memory_readings.driver_and_collective_reserve_bytes` | `get_num_blocks` → `non_torch` | 926 / 6906 / 7266 / 10704 MiB at widths 1/2/4/8. **No fixed-plus-per-peer form fits** 5980 / 6340 / 9138. A table, not a law. Note ATOM's `non_torch` is a **device-wide** reading, so on a shared box it includes neighbours — see D14's dedicated-device scope boundary. |
| `memory_readings.allocator_retained_after_load_bytes` | `peak_torch` | AITER `CustomAllreduce` 1 GiB pool plus the two-stage kernel's: 1.1 MiB at TP1, **2069 MiB flat** at TP2/4/8 |
| `memory_readings.persistent_forward_buffer_bytes` | `peak_torch` | `allocate_forward_vars` + attention metadata; ~118 MiB, flat in width, differs only by model |
| `memory_readings.cudagraph_pool_bytes` | `graph_pool.predicts`, the per-term check of the pool's real cost; not `cudagraph_overhead`, which mirrors ATOM's estimator (`03` D16) | measured `91.1 MiB + 0.3033 MiB per captured token` at W=1; flat **104 MiB** above W=1, where the allocated delta was byte-identical (79,692,800) across three widths and three ladders; read at each configuration's own ladder |
| `memory_readings.activation_bytes` over `warmup_tokens` | `peak_torch`, `cudagraph_overhead` | Qwen3.8-27B TP1 on MI308X: **180,480 B/token** fitted at 16384 warmup tokens predicts the held-out 8192 at **+0.000%**; the geometry form gave 98,304 (-45.5%) and is not derivable within 25% ([#705](https://github.com/jgong5/ATOM/issues/705)) |
| host laws' `tokenizer.*` | admission queue | cc-traces p50 input is **88,768 tokens**; at 2 M tok/s that is ~44 ms, **3x take2's entire admission constant**, and it scales with prompt length while a constant does not |
| host laws' `http_intake.*` | the frontend event loop's charge per request taken in (`06` D33) | at c=256 first turn the real loop takes a burst in one request at a time over 744 ms; taken in at one instant, step 1 admits 225 against the real 109 ([#1103](https://github.com/jgong5/ATOM/issues/1103)); a lone request waits for its own path alone, and at the burst's `fixed` two of #1328's lone requests join one step late ([#1775](https://github.com/jgong5/ATOM/issues/1775)) |
| `interconnect.*_latency_s` | **Clock Authority lookahead floors** (doc 01 D3) | a zero lookahead serializes the whole simulation |
| `compute.fp4_flops` | the roofline (`10` D64) for an FP4 leaf | DeepSeek-V4-Pro's routed experts are FP4 (`16` D95), and MI455X's dense FP4 peak is twice its FP8 peak ([#504](https://github.com/jgong5/ATOM/issues/504#issuecomment-6080974531)); priced at the FP8 rate they would read twice as slow |
| `intra_node.domain_devices` | the link a collective group rides (`10` D66) | MI455X's UALoE fabric joins 72 GPUs across compute trays of 4 (#504), so a TP8 or EP8 group spans two trays without touching the NICs; `count_per_node` alone would send it over `inter_node` |
| `memory.bandwidth_curve` | `BW(M)` in the analytical price (`10` D64) | the attainable bandwidth depends on the bytes moved; one rate would price a small transfer at the large-transfer rate (#1392) |
| `kernel_floor_s` | `t0` in the analytical price (`10` D64) | the device's floor on one kernel, below which no pricing unit runs (#1392); the host's launch time is a host law, not this (`10` D64) |
| `*.link_setup_s` | a rank pair's first KV write (`01` D6) | on node 18 Mooncake TCP a pair's first write of 2 to 7 blocks took 9.0 to 28.1 ms longer than its later writes ([#1172](https://github.com/jgong5/ATOM/issues/1172)); absent is zero |
| every `derate` | the analytical price (`10` D64) | no kernel reaches spec peak; the attainable fraction is probed on the device (D26), compute per format, and assumed only where carried from a probed device (`10` D69) |

**On `derate`.** The name was kept deliberately after considering `achievable_fraction`
and `efficiency`. It is a fraction in `(0, 1]` multiplying a spec-peak number, and it is
the term the schema rules make *mandatory* so that nobody quietly uses a datasheet FLOP as
an attainable one. On a device it is attainable over datasheet, probed where the device was
run (D26). It is not ATOM's implementation efficiency, which is never an input (`10` D68);
`efficiency` would read as that, and `derate` reads like what it is.

**Precisions and links.** Each `*_flops` is the dense matrix-core peak at one operand
precision, never a structured-sparsity figure: `fp8_flops` covers OCP FP8 and MXFP8,
`fp4_flops` is OCP MXFP4. `fp4_flops` is the one optional rate. A device with no FP4
matrix-core path (gfx942) omits it, and a law asked to price an FP4 leaf on that spec
refuses by name rather than borrowing another precision's rate. `intra_node` is the
scale-up fabric: it carries TP collectives, the DP group's gather and MORI's all-to-all,
and `topology` says how it joins its `domain_devices`, `fully_connected` (a link per pair)
or `switched` (MI455X's UALoE). `inter_node` is the scale-out network, a network file's:
it carries Mooncake's KV transfer between the prefill and the decode node (`01` D6).

### Schema rules

1. **No default for a memory reading.** A configuration with no `memory_readings` entry
   **refuses and names it** (`07` D38). These are the terms with no law behind them; a
   silent default is the worst kind of wrong.
2. **`derate` is mandatory wherever a spec-peak number appears**, so nobody can quietly
   use a datasheet FLOP as an achievable one. The peak is the datasheet's, and the
   composition makes its derate from the readings entry and the datasheet as it stands:
   a measured entry's attainable rate over the peak, an assumed entry's ratio as it is.
   The bandwidth curve is a measured entry's as it is, or an assumed entry's
   `bandwidth_derate_curve` times the datasheet's bandwidth, and `memory.derate` is
   derived, not probed: the composed curve's largest point over `bandwidth_bytes_per_s`.
   No readings entry names a datasheet, so correcting one re-prices every run that reads
   it and refuses none. A stated peak the entry gives no derate for is left out of the
   composed spec, as `atom/compass/validate/project.py::projected` leaves it: a price
   that reads that rate refuses naming it, and a required peak refuses as a missing
   field.
3. **Entries one run reads must agree.** Device numbers track the ROCm/RCCL/AITER build
   as well as the die (`07` D43), so every entry a command reads states the platform it
   was made on: its key the device, stack, PTL state and torch its row marks (D24, `07`
   D41). Entries that disagree on an axis two of them state refuse,
   naming each entry and both values; so do probe records of two devices, stacks,
   torches or PTL states folded together (D26). It is the target's platform: a step grid
   bundle's stack, torch included, is checked against the entries the run reads, never
   against the stack of the process that simulates (`17` D101).
4. **The whole composed spec is echoed into every run artifact**, with the documents it
   was composed from. The KV gate is **≤5%**; a number whose spec cannot be recovered
   from the artifact is unattributable.
5. **Every field has an origin, and its document records it** (`spec/fields.py::ORIGINS`,
   `MachineSpec.origin`): `datasheet`, `probed` or `assumed`. A datasheet's fields are
   `datasheet`, its notes citing the document and the date it was read, or `probed` for a
   capacity read on a card, or `assumed`; a readings entry's are `probed`, or all
   `assumed` when taken from another device (D26); a network file's are `assumed` until
   the class 3 probe runs. A probed field also records its subject, what the probe ran
   on: in a readings entry, the probe records it folded. A probed number, a quantity or a
   byte count, also records its band, `p10` and `p90`: in a probe record the spread over
   the probe's repeated samples, in a readings entry the span of every folded record's
   band. Readers use the value, never the band. A document that states a probed number
   without its band is refused by name before it is written
   (`spec/bands.py::refuse_unbanded`). A count or a text is read, not sampled, and so is
   a field the table marks `exact`, the device's capacity: those carry no band. An
   `assumed` field's notes say why it was chosen, and a document that states a method in
   place of an origin is refused by name. The origin is not the method: the term that
   reads a field decides that (`02` D11, #952), so a datasheet value used as a parameter
   is analytical, a probed one used as the answer is measured, and an assumed one makes
   the term declared. A derate for a device never run is assumed, as the frozen stage-3c
   projection took MI455's from the ratios probed on MI355X (`10` D69, #1392), and as
   MI455X B0's readings take MI355X's until a B0 node is probed (#2438). The MI455 A0
   sample node's readings are probed there, and key to A0, never to B0 (#2141); its
   datasheet states a value it did not probe as assumed. `08` D50.2 says how the
   projection reports its assumed fields.

### The tokenizer terms are keyed by model, not by tokenizer identity

D25.1 keyed the spec's tokenizer rates by the tokenizer, with the architectures that
resolve to it, so models sharing a tokenizer shared one probe. The host laws key them by
model, as they key every term (`07` D38, Phase 1c, says why). The key's model does not
cover `tokenizer.json`, so the entry's `fingerprint` and `backend` check it (above).

### Open issues

- **Memory readings carried across dies are untested** (**T50**). An entry assumed for a
  device never run copies another device's of the same configuration, and every term
  built on it is declared (`07` D38). The working assumption — the readings transfer
  within a software generation, not across one — is recorded in doc `03` and has not
  been measured on a second card type. One engine startup settles it.

---

## D26. Probe tools

### Problem

The schema asks for numbers a datasheet does not give — achievable rates, link
latencies. Shipping a schema without the tools to fill it is shipping a
blank form.

### Design

Two commands, one per node and one per platform, so a user measures a platform with no
developer script (`13` D80):

```
compass measure device --gpu G... [--pp] --store S --out D     [GPU, every card of a node]
    -> one probe record per node
compass fit device     RECORD... --store S --out D   [no GPU]
    -> one platform_readings entry per (device, stack, PTL, torch)
compass fit device     --device DATASHEET --assume-from DEVICE --ptl P --store S --out D
    -> one platform_readings entry, every field assumed
```

**`measure device` publishes one probe record per node.** It runs the class 1 and 2
probes below on the cards it names, and with `--pp` the pipeline carriers', and publishes
a `probe_record` entry keyed by the device, stack, PTL state and torch the cards read and
the node's host name (`07` D41). A record holds raw rates, never derates, each with its
band, the cards and the contamination refusals below; a second record on the same node
and key refuses.

**`fit device` folds a platform's records into its readings.** It reads the probe
records it names, each by the token its key prints, and publishes one `platform_readings`
entry: each field the median over the records' nodes, a curve point by point, its p10 and
p90 spanning every record's band, every rate held as a rate. The entry names each record
it folded and no datasheet, since a run divides its rates by the datasheet as it stands
(D25 rule 2). Records of two devices, stacks, torches or PTL states refuse together,
naming both values; a fit to a key already published refuses, since handed-off artifacts
are immutable (`07` D41 rule 4). A node probed after the fit adds a record and leaves the
entry as it is.

**A device never run gets an assumed entry.** `fit device --assume-from DEVICE` takes the
one readings entry the store holds for DEVICE at `--ptl`, divides it by DEVICE's
datasheet, and states the ratios for the device `--device` names, every field assumed:
each derate, the bandwidth curve over DEVICE's bandwidth, and the kernel floor and the
link and carrier terms as they are; a compute format the source states no rate for is
left out, so the composition leaves its peak out (D25 rule 2). This is the rule `compass
project` carries a probed device's ratios by (`atom/compass/validate/project.py::projected`),
and it moves here. The entry is keyed by the target device and the source's stack, torch
and PTL state, the stack stated as assumed from the source's. Several source entries
refuse, listing them. MI455X B0's readings are MI355X's taken this
way, as the frozen MI455 projection took them (`10` D69).

The fragment merge, `validate`'s fragment checks, `explain` and the per-machine scripts
go with the specs they built: a run composes the layers and checks the composition
(D25 rules 2 and 3). The readings of an engine start are not a layer: `compass measure
memory` takes them per configuration into a `memory_readings` entry (`07` D38, Phase 2).

### The probes, by what hardware they need

**Class 0, the host's probes, is retired.** Its tokenizer sweep and IPC round trips, and
class 1's `http-intake` burst, filled the spec's host block. The tokenizer and the HTTP
intake are now timed on what a `compass record` server serves and fitted by `compass fit
host` (D25, `07` D38); nothing read the IPC round trips or the host's core counts.
`scripts/compass/http_intake.py` stays as a check of the hops (`12` T47).

**Class 1 — one GPU of the target type.** `measure device` runs each but
`device-memory`, whose capacity a datasheet states once per SKU.

| Probe | Fills | Notes |
|---|---|---|
| `device-memory`, at TP1 | the datasheet's `capacity_bytes` | the device's capacity, `probed` and exact (D25 rule 5); the readings of an engine start are `measure memory`'s |
| `device-compute` | `compute.attainable_flops`, one per format | per format (BF16, FP8, FP4 where the device has it), the best large-GEMM rate the vendor library reaches, recorded as a rate, which a run divides by the datasheet's `*_flops` (#1392) |
| `device-bandwidth` | `memory.bandwidth_curve` | read-only, write-only and STREAM copy sweeps over transfer sizes; the curve is the per-size maximum of the three, so it bounds any mix of reads and writes ([#1575](https://github.com/jgong5/ATOM/issues/1575)); `memory.derate` is derived from it (schema rule 2) |
| `kernel-floor` | `kernel_floor_s` | an empty kernel replayed in a graph, so the host's launch time is not in it |
| `host-link` | `host_link.*` | the device-to-host copy an lmcache server's stores and loads ride, its latency and the bandwidth a copy reaches |

**Class 2 — N GPUs of the target type.** `measure device` runs both, `pp-carriers` under
`--pp`.

| Probe | Fills | Notes |
|---|---|---|
| `interconnect-intra` | `intra_node.link_latency_s`, `intra_node.attainable_bandwidth_bytes_per_s` | link latency as half a small-message round trip (one hop), the link's attainable bandwidth from a large peer copy, which a run divides by the datasheet's (#1392); collectives in **real groups**, several sizes, are `measure ops`'s, for the empirical pricing of `07` D40 |
| `pp-carriers` | `intra_node.pp.*` | a pipeline stage's send per carrier, RCCL and gloo, between two cards |

`intra_node.link_setup_s` has no probe; absent is zero.

**Class 3 — two nodes.**

| Probe | Fills |
|---|---|
| `interconnect-inter` | a network file's `inter_node.*` |

**Class 4 — no hardware at all.**

| Probe | Fills | Provenance stamped |
|---|---|---|
| `from-datasheet` | a datasheet: `capacity_bytes`, `bandwidth`, `flops`, the intra-node fabric | `origin: datasheet`; a datasheet states no derate, so a run refuses without a readings entry for it |

### Two refusals every hardware probe must implement

Both come from failures that cost real time, and `measure memory` holds them too (`07`
D38):

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

### The composition refuses on

- a missing required field, in whichever layer holds it
- a field in a layer that does not hold it
- no readings entry for the device and PTL state, or several the run cannot choose between (D24)
- entries that disagree on the platform (D25 rule 3)

The composed spec in `run.json` states each field with its value and origin, one entry per
field and never a sum, which is what `explain` printed. This is not a nicety: the KV gate
is 5%, memory validation is **per term and never as a sum**, and a prior summed check read
**+13.8%** while hiding three errors two of which cancelled — the largest being 25% of a
single term.

### Open issues

- Class 2 and 3 probes need a quiet machine. Three of the last five prior pilot attempts
  were lost or degraded by other tenants — two refused to start with a negative KV budget
  at 141 GB of neighbour, one ran 64% slow. **Any probe run needs the machine checked
  before and after, not only before.**
- With no card of the target type, the peaks are its datasheet times the ratios probed on
  a card that was run, assumed, as the frozen MI455 projection took MI355X's and MI455X
  B0's readings take them by `fit device --assume-from` until a B0 node is probed (`10`
  D69, #1392); MI455's A0 samples are
  probed on their node (`16` *Stage 4*). For the memory readings nothing yet decides it:
  the answer is "assume it and accept the error", an entry marked assumed (`07` D38),
  which is honest but unguided.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D24 | The spec describes the machine; ATOM's config describes the deployment. Thread-pool width becomes an ATOM config option, not a Compass one. Revised: a value is stated once, in the layer that names what it varies with: a device datasheet per SKU, platform readings per (device, stack, PTL, torch), a network file per scale-out network that only a run across nodes reads, the host block in host laws (#2309) and the router's hop an assumed host transport constant beside `KV_WRITE_REQ_S` (#2714); each artifact kind keys the device, stack, PTL and torch its invalidation row marks, read from its records, in place of the machine-spec digest; a run's platform is read from the entries it prices with, the user naming device and PTL, the stack only when the store holds several, and the network for a run across nodes. | 2026-09-18; layers 2026-11-03 (#2716, [ruling on #2724](https://github.com/jgong5/ATOM/pull/2724#issuecomment-6596632012)) |
| D25 | Four-section schema: `provenance`, `host`, `device`, `interconnect`. Mandatory derates, no memory readings (`measure memory`'s entries, 2026-10-14), activations included (2026-10-16), `software_pinned_to` checked, torch included, as the target's stack, whole spec echoed into every artifact. Revised for the MI455 projection: an optional dense `fp4_flops`, `intra_node.domain_devices` for a scale-up fabric wider than a node, and every field given an origin, `datasheet`, `probed` or `assumed`, `provenance.origin` and per field `provenance.origins` with a probe's subject, kept across a save. Revised: a `host.http_intake` pair, the event loop's charge per HTTP request taken in, required once every committed spec stated it. An optional `host.http_intake.idle_s`, a lone request's intake. An optional per-link `link_setup_s`, charged once on a rank pair's first KV write. Revised for the attainable-peak roofline: a `derate` is attainable over datasheet, probed, one per compute format; an optional `memory.bandwidth_curve` and an optional `kernel_floor_s`; a device never run carries a probed device's derates, assumed; MI455 has two specs, the A0 samples' probed on their node and the B0 machines' carrying MI355X's derates, assumed, as the frozen projection took them, until a B0 node is probed. A probed number carries its measured p10 and p90 beside its subject, and a spec stating one without them is refused by name before it is written. Revised: three documents, a datasheet, a readings entry and a network file, each holding its own fields and their origins, composed for a run into one spec with the same field paths; `software_pinned_to` and `device.ptl` leave the documents for the readings key; a measured readings entry holds attainable rates and an assumed one the source's ratios, composed into derates with the datasheet as it stands, so no entry names a datasheet, and a peak with no derate is left out of the composition; `router_relay_s` leaves the network file for an assumed host transport constant; the stack check becomes: entries one run reads must agree. Revised: the host block leaves the spec; the tokenizer rates and the HTTP intake are the `tokenizer` and `http_intake` terms of the `host_laws` entry, keyed by model with the tokenizer's backend and fingerprint checked and no derate, and a missing term charges 0 s, named in the run summary; `host.cpu` and `host.ipc` go with no reader. Revised: the `tokenizer` term prices one-item and several-item encodes as two kinds and a decode with no slope as `fixed_s` alone inside its token hull, each kind optional. | 2026-09-18; revised 2026-10-10; torch 2026-10-13 (#851); origin 2026-10-14 (#952); intake 2026-10-16 (#1103), required 2026-10-18 (#1230); idle intake 2026-10-20 (#1775); link setup 2026-10-16 (#1172); activations and the transfer origin dropped 2026-10-16 (#1226); attainable peaks 2026-10-17 (#1392); `host.admission_fixed_s` dropped 2026-10-17 (#1000); bands 2026-10-18 (#1482, #1518); MI455 probed 2026-10-25 (#2138), split into A0 and B0 2026-10-28 (#2141); layers 2026-11-03 (#2716, [ruling on #2724](https://github.com/jgong5/ATOM/pull/2724#issuecomment-6596632012)); host block to host laws 2026-11-03 ([#2309](https://github.com/jgong5/ATOM/issues/2309#issuecomment-6498578963)); tokenizer kinds 2026-11-05 ([#2751](https://github.com/jgong5/ATOM/issues/2751#issuecomment-6655175759)) |
| D26 | Probes emit fragments; `merge` / `validate` / `explain` combine and check them. Four hardware probe classes plus a datasheet path; the transfer path dropped 2026-10-14 (#937). Every hardware probe implements the contamination refusals. All are the `spec` package's functions, run by per-machine scripts, with no `compass` command. Machine specs are developers', one per supported machine, passed by `--machine`; an engine start's memory readings are `measure memory`'s, not the spec's. Revised: class 1 probes the per-format GEMM rate, the bandwidth curve and the kernel floor; class 2 the link latency and bandwidth. Revised: the bandwidth curve is the per-size maximum of a read-only, a write-only and a copy sweep. Revised: `compass measure device` publishes one probe record per node, raw rates with their bands; `compass fit device` folds a platform's records into one readings entry, each field the median over nodes with a band spanning theirs, rates held as rates and naming no datasheet, one entry per key, never replaced; `--assume-from` states another device's ratios, assumed, for a device never run; the fragment merge, `explain` and the per-machine scripts go. Revised: the class 0 probes and the `http-intake` probe are retired; `compass record` times the served tokenizer and stamps the intake, and `compass fit host` fits both. | 2026-09-18; revised 2026-10-13 and 2026-10-14 (#851); 2026-10-17 (#1392); 2026-10-19 (#1575); 2026-11-03 (#2716, [ruling on #2724](https://github.com/jgong5/ATOM/pull/2724#issuecomment-6596632012)); 2026-11-03 ([#2309](https://github.com/jgong5/ATOM/issues/2309#issuecomment-6498578963)) |
| D25.1 | Runtime-constant fields renamed to say what they are; ATOM's own term recorded as a mapping column. `host.tokenizer` becomes `host.tokenizers[]`, keyed by tokenizer identity (id + backend + fingerprint) with `applies_to` model architectures. Superseded for the tokenizer: its rates are a `host_laws` term keyed by model, the entry's backend and fingerprint checked. | 2026-09-19; tokenizer superseded 2026-11-03 ([#2309](https://github.com/jgong5/ATOM/issues/2309#issuecomment-6498578963)) |
