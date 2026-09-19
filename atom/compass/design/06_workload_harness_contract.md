# ATOM Compass — Design Topic 6: The Workload Harness Contract

**Status:** draft for review. Drafted by an AI assistant during a design interview; not
yet reviewed or approved. No code has been written against it.

**Depends on:** `01_execution_and_time_model.md` (the Clock Authority and the Category-A/B
wait contract), `05_machine_spec_and_probes.md` (the host terms the tokenizer model needs).

**Scope.** How traffic reaches a simulated ATOM: the contract any harness must satisfy,
the wire protocol, the per-harness adapter, and the serving-side costs the harness makes
visible. Acceptance requires paired simulated and real execution of cc-traces proper, so
the *same* harness must drive both sides.

---

## D27. A contract, not a harness

### Problem

The obvious move is to write a bespoke client. The prior effort did (`scripts/compass/replay.py`,
325 lines) and paid for it: a client that sent 5-8x the requested tokens superlinearly; a
64-connection pool that **deadlocked** against a 300-request declared workload and was
released only by a 120-second barrier timeout, after which the client still printed
`0 failed` and a day's conclusions were drawn from the result.

Meanwhile SemiAnalysis ship the official replayer for this corpus, and reimplementing
their trace interpretation is the single highest-risk way to invalidate everything
downstream.

### Decision

**Define a contract. Keep every harness outside ATOM. Support more than one.**

Three parts:

```
   +--------------+   1. clock: now() / advance_to(T) / blocked() / running()
   |   harness    |<----------------------------------+
   | (any vendor) |                                   |
   +------+-------+                            +------+------+
          |  2. wire:  compass.arrival_s  -->  |    Clock    |
          |            <-- compass.{arrival_s,|  Authority  |
          |                 first_token_s,    +------+------+
          |                 finish_s}                |
          v                                           |
   +--------------+                                   |
   | ATOM api_srv |-----------------------------------+
   +--------------+
```

**Part 1 - clock client.** `now()`, `advance_to(T)`, `declare_blocked()`,
`declare_running()`. This is doc 01 D4's Category-A (rewrite: do not wait, advance) and
Category-B (annotate: leave the blocking call alone) contract, already specified. A thin
library per language.

**Part 2 - wire protocol.** Additive optional fields on the existing OpenAI-compatible
endpoint, ignored by a real server. See D28.

**Part 3 - a per-harness adapter**, living outside both ATOM and the harness repo. It
redirects pacing to the clock client and stamps the harness's latency anchors from the
response fields.

### What a second harness must implement

Exactly three things: replace its inter-request sleep with a clock call; attach one field
to an outgoing request; take its latency numbers from response fields rather than its own
stopwatch. A harness that cannot do the third is still usable for throughput-only studies.

### Warmup is not a protocol feature

Settled: **warmup requests are ordinary requests sent early.** They carry `compass.arrival_s`
like any other, go through the same endpoint, and the contract gains nothing. The
earlier draft treated warmup as an open contract question; it is not one, because
nothing about a warmup *request* differs from a normal one.

What does differ is the **engine's state when it serves them**, and that is where the
actual hazard lives. A prior sweep's first three prefill steps cost 47.5 / 19.5 / 6.1 s
at 8 / 32 / 96 tokens, where the same shapes later cost **0.11 s** — 430x on the first
one. That is JIT compilation, autotuning and allocator growth, not a property of the
request.

So one guardrail, and it is a measurement rule rather than a protocol one:

> **The cost model does not model warmth, so the validation window must exclude it on
> both sides, by request id, agreed before the run.**

Three consequences worth being explicit about:

1. **Both sides exclude the same requests.** Excluding the first N on the real side and
   the first N on the simulated side is not the same thing if the two runs admit in a
   different order. The harness declares the warmup request ids; both sides honour that
   list. This is the same rule doc `08` applies to every paired comparison.
2. **Simulated warmup is cheap and real warmup is not.** A simulated run will serve
   those requests at steady-state prices, so the two runs' *wall* behaviour during
   warmup diverges wildly. That is expected and harmless as long as rule 1 holds — and
   it is the reason warmup cannot simply be left in and averaged over.
3. **Warmth is measured, never charged** (doc `09` D62). If a future model wants to
   predict the cold steps, that is a separate term with its own evidence, not a fudge on
   the steady-state price.

### Open issues

- Nothing here handles a workload where warmth *recurs* mid-run — a new shape reaching
  autotune for the first time at minute 10. Doc `09` D62 measures leading warmth; it
  does not detect a late one. Recorded as **T54**.

---

## D28. Endpoint: additive fields on the real endpoint

### Problem

Simulation needs one declared value in and three readings out. They can ride the production endpoint or a
dedicated one.

### Options

| | share `/v1/chat/completions` | dedicated `/compass/*` |
|---|---|---|
| Harness change | **none** (aiperf has `--extra-inputs`) | must retarget the harness |
| Paired comparison | **identical client code on both sides** | the simulated path diverges from the real one |
| Exercises real protocol parse + tokenizer | **yes** | duplicated or skipped |
| Production schema | polluted with simulation fields | clean |
| A real server with `extra="forbid"` | 422s unless the fields are optional-and-ignored | unaffected |

### Decision

**Additive optional fields on the real endpoint, in both directions.** The request path
*is* the thing being measured; diverging it breaks the pairing.

| Direction | Field | Meaning |
|---|---|---|
| request -> | `compass.arrival_s` | when this request counts as arriving, on the run's simulated timeline |
| <- response | `compass.{arrival_s, first_token_s, finish_s}` | the engine's own readings |

#### Minimality audit: why four values, and why not three or five

The requirement is that these are the minimum — nothing more, nothing less. Audited in
both directions.

**One nested object per direction, not four flat keys.** The earlier draft used four
top-level names (`sim_arrival` in; `sim_arrive`/`sim_first_token`/`sim_finish` out).
Two problems: `sim_arrival` and `sim_arrive` differ by two characters and mean different
things, which is a bug waiting to happen; and four new top-level keys is four collisions
with a schema ATOM does not own. One `compass` object is **one** additive key each way,
namespaced, and trivially ignorable by a server that does not know it.

**What is deliberately NOT a Compass field, because the OpenAI schema already has it:**

| Need | Existing field | Note |
|---|---|---|
| declared output length | `max_tokens` / `max_completion_tokens` | the harness must always set it |
| never stop early | `ignore_eos` | always true under simulation; the output is garbage, so an EOS would be an artifact of the filler, not the model |
| request identity | `id` on the response | the join key for the exclusion list and the step table |
| streaming first-token timing | SSE first chunk | the harness's *wall* reading — kept, but not the graded one |

Adding a Compass field for any of these would duplicate state that can disagree.

**Why each of the four is irreducible:**

- `compass.arrival_s` **(in)** — the only field with no existing home. The server cannot
  infer it: under simulation the harness sends requests as fast as the socket allows and
  the *declared* arrival is the whole point. Without it there is no simulated timeline.
- `compass.arrival_s` **(out)** — not an echo. The engine may clamp a declared arrival
  (it cannot be earlier than the run's epoch, and the arrival gate of doc `01` D8 may
  defer it). Returning what the engine *used* is how the harness detects that its
  timeline was not honoured. Dropping this makes a clamped run silently misreport TTFT.
- `compass.first_token_s` **(out)** — TTFT is a graded acceptance metric at ≤10%. The
  harness's own stopwatch measures HTTP and wall time, which under simulation is
  unrelated to simulated time. There is no other source.
- `compass.finish_s` **(out)** — same argument for TPOT and throughput; and
  `finish − first_token` over the generated count is TPOT, so this is not derivable from
  the other three.

**What was considered and rejected as a fifth field:** a per-step or per-token timeline.
It would make TPOT *distribution* gradeable rather than just its mean. Rejected because
the same information is already in the engine's own step table (doc `08`), which both
sides emit, and putting it on the wire would grow every response by the output length.
If a later result needs per-token simulated stamps, it comes from the step table join,
not from the endpoint.

take2 built the request half: `CompletionRequest.compass_arrival` as an offset into the
run, `llm_engine._stamp_arrival` turning it into `epoch + offset` and warning plus falling
back to `now()` on a real clock, and `Scheduler._declared_arrival_pending` putting a
not-yet-arrived sequence back on the waiting queue — deliberately **not** via
`_unschedulable_reason`, which would finish it.

**The response half is a change from take2, and an improvement.** take2 used
`GET /compass/requests` as a bulk drain. Response-carried fields are better for a
streaming client: numbers arrive per request, in band, with no separate drain and **no
id-space join**. take2 lost a day to exactly that join — the step table recorded internal
sequence ids (`0, 1, 10`), `/compass/requests` recorded external completion ids
(`cmpl-...`), the engine held the mapping in `_internal_to_external` and exposed neither
side. Worse, an ordinal join is unsound on the simulated side: it gave -7.2 s where the
true join gave +7.0 s, because the virtual clock's origin is not the workload's first
arrival.

Keep `GET /compass/requests` as a bulk diagnostic, not the primary path.

### Open issues

- Whether upstream ATOM will accept simulation fields in the public request schema, or
  whether they must be namespaced or gated.

---

## D29. Transport: real HTTP to ATOM's real server

### Decision

The adapter's transport speaks **real HTTP to ATOM's actual uvicorn socket**, in both PD
topologies, using the same client code as a real run.

Reasons, in order of weight:

1. **D30's timeline piggyback only works over the real relay.** Bypassing HTTP bypasses
   Atomesh, and then PD-disaggregated runs cannot be driven by the same harness.
2. It keeps `api_server.py` in the loop — protocol parse, request handling and
   **tokenization**, which is on the modelling list and turns out to matter (D33).
3. A real run and a simulated run then differ in exactly one thing: what the forward pass
   costs.

The round trip is a doc 01 **Category-B** blocking wait: annotate `declare_blocked()` /
`declare_running()` around the existing call, leave the call itself alone. Real wall time
passes; virtual time does not advance; the Clock Authority routes grants elsewhere.

Rejected: aiperf's own `FakeTransport` (bypasses HTTP entirely) and an in-process ASGI
call. Both are faster and both break reason 1.

---

## D30. PD disaggregation: piggyback the timeline, change nothing in Atomesh

### Problem

Under PD disaggregation the **first token comes from prefill** — the router rewrites the
prefill request with `max_tokens=1` — so TTFT spans prefill + KV transfer + decode start.
A naive design makes the router merge two simulated timelines. Atomesh is Rust, its clocks
are `tokio::time` / `std::time::Instant`, and we want to touch it as little as possible.

### The mechanism that already exists

The ATOM relay is strictly sequential (`http_pd_router.rs:969-1192`), unlike the SGLang
path (`tokio::join!` on both, `:1463`) and the vLLM path (detached `spawn`, `:732`):

```
client --compass.arrival_s--> router --inject_prefill_fields--> PREFILL
                                  (sets kv_transfer_params, stream=false,
                                   max_tokens=1; other fields pass through)
                        router <-- prefill response
                                  extracts kv_transfer_params ONLY
                        router --enrich_decode_kv--------> DECODE
                                  (adds remote_*; strips nothing)
client <--------------  router <-- decode response (streamed)
```

`AtomAdapter::enrich_decode_kv` (`placement/backend/atom.rs:37-59`) only *adds* fields
(`remote_dp_size`, `remote_tp_size`, renames `dp_rank` -> `remote_dp_rank`). The blob
itself is produced by `MoRIIOConnectorScheduler.request_finished`
(`moriio_connector.py:970-1001`) and the router hard-errors if it is absent
(`http_pd_router.rs:1073-1078`).

### Decision

**Prefill writes `arrival_s` and `first_token_s` into `kv_transfer_params`. Decode reads
them, adds `finish_s`, and emits the merged timeline in its response.**

Consequences:

- **Atomesh needs zero changes** on the timeline path. (It still needs its failure
  detectors disabled per doc 01 D5, and two hardcoded Rust timeouts raised.)
- **The harness cannot tell PD-aggregated from PD-disaggregated** — one uniform response
  shape in both topologies, which is exactly the property a fair comparison needs.

### Open issues

- The simulated KV connector (doc 01 D6) must produce a `kv_transfer_params` blob of the
  same shape as MoRI-IO's, or the router refuses.
- Mooncake requires **all** `(pp_rank, tp_rank)` pairs to report before a request
  completes; MoRI-IO does not. The simulated connector must pick one semantic and declare
  it.

---

## D31. Output semantics: declared length, forced continuation, and the filler token

### Problem

The simulation runs no real inference, so the tokens it returns are meaningless. That is
fine for the harness — it ignores content — but the engine's own stop logic and the
serving path's detokenizer both consume those tokens.

### What the harness already does

The AgentX scenario sets `require_ignore_eos=True`
(`common/scenario/inferencex_agentx_mvp.py:10`) and takes `max_tokens` per turn from the
recorded `out` (`weka_trace.py:1686`, `_cap_output` at `:1029-1041`). `--synthesis-max-osl`
applies to parent turns only; **subagent turns are deliberately uncapped**. So the client
always declares the output length, and `ignore_eos=false` is a hard config conflict.

### Decision

The simulated forward returns a **filler token id** with two properties:

1. **Not EOS, and not part of any stop string.** Otherwise a random token terminates the
   sequence early and the run length drifts from the trace.
2. **Decodes to complete, standalone ASCII.** This is not cosmetic — see D33. ATOM's
   `IncrementalStreamDetokenizer` only advances its sliding window when the decoded text
   does **not** end in `�`. Random token ids produce random byte sequences, incomplete
   UTF-8 would be frequent, the window would grow without bound, and detokenization would
   go **O(n^2) in output length** — nothing like a real run.
3. **Derived from the request id**, not a global constant. See D32.

A small vetted set of such tokens, chosen per tokenizer, indexed by request id.

### Open issues

- The vetted set is per tokenizer and nobody has built one.
- Whether ATOM honours `ignore_eos` on every path, or whether the filler token is the only
  guarantee, is unverified.

---

## D32. Prefix caching with garbage outputs

### Problem

In a real agentic trace, turn N's prompt contains turn N-1's assistant output, so the
prefix chain crosses generated tokens. If the simulation generates garbage, that chain
appears to break.

### Finding: the chain is already broken, for real servers too

The harness sends the **recorded** assistant output in the history, not the server's
generation — `faq.md:448-453`, deliberate, so block structure is identical across servers.
A real server generates different text from the recorded history it is later handed, so
**its** decode-cached blocks do not match the next turn's prompt either.

So the simulation is **not at a disadvantage**: both sides of the paired comparison have
the same property. That is what makes the comparison valid.

### The real hazard is a false hit

If every request decodes the same filler id, two requests sharing a prompt prefix produce
identically-chained decode blocks (`BlockManager.compute_hash` is xxhash chained with the
parent hash, `block_manager.py:233-245`), and the second would **hit** the first. A real
run would not. That inflates cache hits — and since `num_cached_tokens` feeds chunked
prefill sizing and admission, **it changes the schedule**, which is the thing this project
is most sensitive to.

Hence the per-request filler id of D31.

### A free oracle

aiperf computes `theoretical_prefix_cache_hit`
(`metrics/theoretical_prefix_cache.py:125-134`) as an **infinite-cache simulation over the
trace's own `hash_ids`**: `100.0 * hit_blocks / total_blocks`. Our simulated block-manager
hit rate should sit *below* it by exactly the amount finite capacity explains. Two
independent computations of the same quantity, from different data, is the kind of check
this project has repeatedly needed.

Corpus scale for calibration: **95.1% of blocks issued corpus-wide are reuse** (107.7 M
issued vs 5.27 M distinct), and **86.5% of consecutive root turns share the entire previous
prompt as prefix**; `lcp / current_prompt_blocks` is p10 0.880, p50 0.988, p90 0.998.

### Open issues

- Cache-busting is **locked on** under the scenario (`require_cache_bust=FIRST_TURN_PREFIX`),
  injecting `[rid:8a3f2c1b9e7d]` derived from benchmark id, recycle count, lane and trace
  id. It deliberately suppresses cross-lane sharing. Any cache-hit number must say whether
  it was on.
- `hash_id_scope` is `Literal["local"]` only (`weka_trace_models.py:150-157`); global scope
  is rejected at schema level because it *"would require synthesis-time coordination across
  files"*. So cross-trace cache sharing is out of scope by the corpus's own construction.

---

## D33. The cost of tokenization and detokenization

### Problem

The harness sends **text**, as a real run does, so ATOM tokenizes it. With a corpus whose
p50 input is 88,768 tokens and p90 is 204,288, this is not a rounding error.

### Where it happens

**Encode** — `self.tokenizer.encode(prompt_or_tokens)` at `llm_engine.py:690`, inside
`InputOutputProcessor.preprocess`, reached from the API server via
`await loop.run_in_executor(None, do_preprocess)` at `api_server.py:890`, `:1004`, `:1126`,
`:1258`, `:1480`. That is Python's **implicit default executor**, width
`min(32, cpu_count + 4)` — see doc 05 D24, which makes it an ATOM config option.

**Decode** — `IncrementalStreamDetokenizer.update` at
`atom/entrypoints/openai/streaming_dispatch.py:40-69`:

```python
self.tokens.extend(token_ids)
prefix_text = self.tokenizer.decode(self.tokens[self.prefix_offset : self.read_offset], ...)
new_text    = self.tokenizer.decode(self.tokens[self.prefix_offset :], ...)
if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
    delta = new_text[len(prefix_text):]
    self.prefix_offset = self.read_offset
    self.read_offset   = len(self.tokens)
```

Four properties that shape the model:

1. **Two `decode` calls per update**, not one.
2. **Cost is O(sliding window), not O(total output).**
3. It runs on the **engine output threads, batched per engine step** — the module
   docstring: *"buffers a whole engine step, detokenizes it, and schedules a single
   callback per event loop."* One thread, all streams: a serialization point at high
   concurrency.
4. The window only advances when the decode does not end in `�`. Hence D31's ASCII
   constraint.

Note also the terminal `self.tokenizer.decode(req.completion_token_ids)` at
`llm_engine.py:776` — but ATOM defect #6 records that `InputOutputProcessor.postprocess` is
**never called from `api_server.py`**, so that path does not run under serving. It matters
only for the offline `generate()` path.

### Decision

**Run the real tokenizer for its effect; charge a modelled duration for its time.** Same
rule as the forward pass.

| Stage | Where | Concurrency | Service time |
|---|---|---|---|
| encode | default `ThreadPoolExecutor` | width from ATOM config | `encode_fixed_s + tokens / encode_tokens_per_s` |
| decode | engine output thread, per stream per step | **1** | `2 x (decode_fixed_s + window / decode_tokens_per_s)` |

Terms come from `host.tokenizers[]` in the machine spec (doc 05 D25), populated by the
`compass spec probe tokenizer` Tier-0 probe.

**This is a queue, not a constant.** At the corpus p50 and a plausible 2 M tokens/s,
encode is ~44 ms — roughly **3x take2's entire admission constant** (13.7 ms) — and it
scales with prompt length while a constant does not. At 256 clients with sub-agent fan-out,
requests will queue for the executor, and that queueing is a real serving effect worth
reproducing rather than averaging away.

`host.admission_fixed_s` covers what is left: everything between HTTP arrival and the
request appearing in `scheduler.waiting`, **excluding** tokenize.

### Open issues

- take2's admission constant was measured six times at 8.25 / 11.52 / 13.71 / 13.91 /
  17.88 / 18.25 ms, **unrelated to request count or prompt length** — spread +-5 ms on
  13 ms. Some of that spread is probably the tokenizer term now being modelled separately;
  the rest bounds TTFT accuracy at roughly +-8%.
- Admission is **path-specific**: 13 ms on the offline batch path, 9 ms on the serving
  path, same machine and model. And **instrumentation-specific**: measure mode cost ~11 ms
  of TTFT on the 27B (4%).

---

## D34. The aiperf adapter package

### What it plugs into

agentx-harness is a fork of **NVIDIA AIPerf v0.12.0**, Apache-2.0, ~203,654 lines under
`src/aiperf/`. Two facts make the adapter small:

1. **34 plugin categories**, discovered via an `aiperf.plugins` setuptools entry point
   (`pyproject.toml:84`). An out-of-tree package can register a transport, service manager,
   communication backend, timing strategy, dataset loader, metric or exporter with **zero
   edits to their repo**. Highest priority wins.
2. **All arrival pacing funnels through one abstraction** — `common/loop_scheduler.py`,
   37 call sites, all but 3 inside `timing/`. The single line that matters is
   `timing/strategies/agentic_replay.py:1559`:

```python
if next_meta.delay_ms is not None and next_meta.delay_ms > 0:
    self.scheduler.schedule_later(next_meta.delay_ms / MILLIS_PER_SECOND, coro,
                                  group_id=credit.effective_root_correlation_id)
else:
    await coro
```

3. **Metrics are stamped in the transport**, so our transport controls them.
   `ttft_metric.py:49-56` is `content_responses[0].perf_ns - request.start_perf_ns`;
   `request_latency_metric.py:38-49` is `content_responses[-1].perf_ns - start_perf_ns`;
   ITL is `(request_latency - ttft) / (osl - 1)`. A transport that sets `start_perf_ns`,
   each SSE chunk's `perf_ns` and `end_perf_ns` from the response's `sim_*` fields makes
   the **entire existing metric and export stack correct by construction** — TTFT, ITL,
   ICL, latency, throughput, percentiles, goodput, the JSON/CSV exporters, the swim-lane
   plot and `submission_valid`. Roughly 15,000 lines of metrics code for free.

### Contents and size

| Component | Lines |
|---|---|
| Transport plugin — real HTTP, re-stamp anchors from `sim_*` | 150-250 |
| Timing-strategy override — redirect `schedule_later` to the clock client | 80-150 **if subclassing works** |
| Clock client library | 100-150 |
| Plugin manifest, bootstrap, config glue | ~100 |
| Tests | 200-400 |

**~450-650 lines of production code**, plus tests. For scale, aiperf's own
`fake_transport.py` is 478 lines and does *more* — it simulates a whole server.

### What we do NOT need

**`TimeTraveler` is probably unnecessary.** aiperf ships one (`tests/harness/time_traveler.py`,
261 lines, patching all six `time.*` functions) and depends on `looptime>=0.5`. That route
would also require ~25 line changes inside their repo: 10 `default_factory=time.*` sites
bound at import (including `RequestRecord.timestamp_ns` and `start_perf_ns`), a
`from time import` binding at `common/models/trace_models.py:4`, and
`common/event_loop_monitor.py:80-82` opting *out* to the real clock.

If pacing goes through the clock client and metrics come from the transport, **the
harness's own `time.*` reads stop mattering**, and the edit count goes to zero. That is the
design.

### The risk that could blow the estimate up

If `AgenticReplayStrategy` (1,952 lines) cannot be subclassed cleanly and must be vendored
into the adapter package. Still zero edits to their repo, but 2,000 lines to keep in sync
with upstream. **Testable in about an hour, and worth testing before committing.**

Second risk, shared by any route: **32 `asyncio.wait_for(..., timeout=T)` sites across 21
files.** Under virtual time these can fire instantly. This is where the debugging will go.

### Open issues

- `--use-think-time-only` (`docs/cli-options.md:512`) exists specifically for *"zero-latency
  mocks"*, dropping the recorded `api_time` from each gap. It is the wrong choice here —
  our engine supplies a simulated `api_time`, so the default end-to-start derivation is
  correct — but a run using it is `submission_valid: false`, which is worth knowing.
- Dataset reconstruction is **minutes of real CPU** (tokenizing and decoding the whole
  corpus). It must run on the real clock before the measured window opens.

---

## D35. What this harness does and does not reproduce

Acceptance asks for *"faithful agentic behavior: branching, joins, delays,
completion-driven recycling, cancellation, cache reuse and the resulting batch shapes"*.
Stated against the evidence:

| Property | Status |
|---|---|
| **Branching** | **Reproduced.** 1,697 subagent wrappers across 175 of 393 sessions, up to 10 concurrent branches, max 153 in one session. The loader does not even trust the nesting: an LCP-over-`hash_ids` pass finds fan-outs recorded as flat interleaved top-level requests — *"615 subagent entries hide ~3.1k distinct context chains"*. |
| **Joins** | **Reconstructed, not recorded.** No field in the corpus says a parent resumed because a child finished. The harness imposes it: subagent entries become child conversations linked SPAWN/JOIN, and `handle_credit_return` gates the next parent turn on blocking children returning. |
| **Delays** | **Reproduced**, end-to-start. `_end_to_start_delay_ms` (`weka_trace.py:131-157`) subtracts the previous turn's recorded `api_time` from the start-to-start gap, because *"the replay dispatches turn k after turn k-1 completes, so adding the full gap double-counts api_{k-1} -- each turn drifts later, compounding per stream and fabricating cross-stream concurrency."* |
| **Completion-driven recycling** | **Reproduced.** A lane replays a tree, and on drain immediately draws the next trace from turn 0. |
| **Cancellation** | **NOT reproduced, and not available.** `status` is `"completed"` on **all 1,697** wrappers, `tool_use_count` is `null` on all of them, `subagent_type` is `"Subagent"` on all of them, in **both** the 1M and 256k corpora. There is no other value to replay. |
| **Cache reuse** | **Reproduced structurally.** `hash_id -> deterministic token block`, memoized, reseeded per `(trace_id, hash_id)`, hard-erroring if one `hash_id` ever gets two sizes (`dataset/generator/prompt.py:417-451`). Their phrasing: *"Synthesized -- but token-count-exact and cache-structure-exact."* |
| **Closed-loop arrivals** | **Reproduced.** There is no request-rate knob in this mode at all. |
| **Batch shapes** | Follow from the above plus ATOM's real scheduler. |

Other declared non-reproductions, from the harness's own docs:

- Prompt **text** is synthetic; prompt **structure** is real. *"Don't read the generated
  text as meaningful."*
- The assistant history is the **recorded** output, block-aligned, not the server's
  generation.
- Recorded model names are silently rewritten to `--model`; *"no warning is emitted when
  the counts differ"*.
- Tool calls are not real schemas — plain user messages by default.
- Recorded `ttft` on `type: "s"` requests is parsed into the model and **never read**.
- Fork mode is unexercised: *"In the SemiAnalysis corpora every subagent is a spawn."*
- Latencies are per request only; *"there is no built-in per-session roll-up."*
- The tutorial carries a status banner: *"Work-in-progress MVP ... may change as the spec
  stabilizes."*

**This table belongs in the acceptance scope, declared up front.** The prior effort's
standing lesson is that each of these is easy to leave unstated and each one silently
narrows what "validated on real traces" means.

### Client-count note

`--concurrency N` means **N agent session trees**, not N requests — *"Each unit of
concurrency is one replay lane ... When a session finishes, its lane immediately
recycles"* — and subagents run inside the parent's slot, so instantaneous in-flight
requests exceed N during fan-out. That matches the requirement's own wording. 1/4/16/64/256
is five invocations; the scenario **rejects** comma-separated sweeps.

One corpus limit to plan around: only **175 of 393** sessions contain any subagent, and
only **144** offer a multi-request episode containing a descendant. A 256-client cell that
requires genuine fan-out in every root is not constructible without reusing sessions.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D27 | A three-part contract (clock client, wire fields, per-harness adapter), not a bespoke client. Harnesses live outside ATOM; more than one is supported. | 2026-09-18 |
| D28 | Additive optional fields on the real endpoint, both directions. Response-carried timings replace take2's bulk drain. | 2026-09-18 |
| D29 | Real HTTP to ATOM's real uvicorn server, both PD topologies. Round trip is a Category-B blocking wait. | 2026-09-18 |
| D30 | Piggyback the simulated timeline on `kv_transfer_params`; Atomesh needs zero changes on that path. | 2026-09-18 |
| D31 | Filler token is non-EOS, decodes to complete standalone ASCII, and is derived from the request id. | 2026-09-18 |
| D32 | The decode->prefill cache chain is already broken by the harness for real servers too; guard only against false hits. `theoretical_prefix_cache_hit` is the oracle. | 2026-09-18 |
| D33 | Run the real tokenizer for its effect, charge a modelled duration for its time. Encode is a bounded-width queue; decode is a single-threaded per-step stage. | 2026-09-18 |
| D34 | The aiperf adapter is an out-of-tree plugin package, ~450-650 lines, with zero edits to agentx-harness. | 2026-09-18 |
| D35 | Declare what the harness reproduces and what it cannot; cancellation is not available from this corpus. | 2026-09-18 |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T10 | Verify `AgenticReplayStrategy` can be subclassed rather than vendored | ~1 hour to test; changes the adapter estimate by 2,000 lines |
| T11 | Build the per-tokenizer vetted filler-token set | needs a tokenizer in hand |
| T12 | Chase the 32 `asyncio.wait_for` sites under virtual time | only reachable once the adapter runs |
| T13 | Decide the simulated KV connector's completion semantic (MoRI-IO's last-status vs Mooncake's all-ranks) | doc 01 D6 open issue, surfaces here |
| T14 | Build the client-count matrix given only 144 fan-out-capable sessions | affects the 256-client cell |
| ~~T15~~ | ~~Warmup handling in the contract~~ — **done**, D27: warmup requests are ordinary requests; the rule is an exclusion window agreed by request id |
