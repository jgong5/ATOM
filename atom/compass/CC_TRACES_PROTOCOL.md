# The cc-traces acceptance protocol

The final acceptance of this PoC is an end-to-end serving comparison on requests
from the cc-traces corpus. Everything else — the synthetic short workload, the
single-forward-step checks, the one-step TP2 and TP4 freezes — is a diagnostic.
They stay in the record and keep their own registrations; none of them is this.

Registered before the runs it governs, and stamped into every cell those runs
produce (`scripts/compass/cc_traces_protocol.py`). A cell whose stamp does not
match the registration is not an acceptance run.

## 0. What this replaces, and what it leaves alone

`atom/compass/PROTOCOL.md` — prepare-then-measure — is **not** superseded. Its
sections 1 through 7 and 10 through 12 are how an acceptance cell is run, and
this protocol inherits them unchanged: the compile-cache regime, the preparation
batch, the drain and its proof, virtual time, the refusal to warm a predictor,
and the two enforcement layers in `replay.py` and `compare.py`.

What this replaces is its **section 8**, and only for the cells registered here.
Section 8 registers a synthetic short workload, the first twenty requests of
`cc_pilot.jsonl` as the long one, and one engine calibration per (tp, workload
class). All three are development quantities, and the third is the reason a new
protocol is needed rather than a new workload file: calibrating the engine at
the width being evaluated is what made every long-input result a fit statistic.

Cells already run under `PROTOCOL.md` keep their lock, their stamps and their
labels. Nothing here relabels them.

## 1. The original two classes, registered by digest

**These are the workloads of the six-cell matrix.** They remain registered, and
the cells already run against them keep their meaning. The acceptance matrix
this protocol now governs is §1B's, which adds an offered-load axis and replays
the delegated-agent concurrency §1 deliberately refused. Both registrations are
live; a cell says which one it ran.

Two classes, both drawn from `semianalysisai/cc-traces-weka-062126-256k`
(`traces.jsonl`, 568 864 747 bytes, sha256
`e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`, 393
sessions), selected by the rule in `scripts/compass/cc_traces_workload.py`
(sha256 `d0a1dc1be2ae777408266d8a18996bd55145cd688edddb1482a179a0ad8b09ff`).

The rule's constants live in that file, not on its command line. There is no
flag that takes a different slice, so a re-run cannot quietly take an easier
one, and `verify --corpus` re-runs the rule and compares bytes.

| | long | short |
| --- | --- | --- |
| file | `atom/compass/cc_traces_long.jsonl` | `atom/compass/cc_traces_short.jsonl` |
| sha256 | `68326792ead8da5f52478ba111cfae84746c24a2f23a3cb394d026c99b018e8a` | `d76694982caceae322d6dcb81229f95cb294fccd434d8f5d3ea4be6b20e2a0b0` |
| manifest | `cc_traces_long.manifest.json` | `cc_traces_short.manifest.json` |
| requests | 20 | 64 |
| sessions | 1 (corpus index 6, `0470d446a451…`, 414 turns) | 64, corpus indices 0–85, each contributing its whole leading short run |
| input tokens | min 448, median 91 008, p90 105 920, max 107 328, sum 1 594 624 | min 256, median 448, p90 768, max 2 560, sum 36 480 |
| output tokens | min 24, median 608, p90 1 661, max 2 413, sum 15 833 | min 14, median 21, max 39, sum 1 377 |
| longest prompt + output | 109 741 (107 328 + 2 413, the window's last turn) | 2 581 (2 560 + 21) |
| arrivals | 262.828 s, the session's own spacing, **no gap clipped or compressed** (min 0.125 s, median 9.858 s, max 26.670 s) | every session starts at 0.0 by declaration; within a session the source's own intervals |
| rewrites | `gaps_clipped: 0`, `outputs_altered: 0` | `gaps_clipped: 0`, `outputs_altered: 0` |

**Long** is one whole session's opening window of twenty requests, replayed on
that session's own timeline. The only transform is a common origin shift that
puts the window's first turn at 0.0; every interval between arrivals is the
trace's. Eligibility, not editing, is what keeps the cell affordable: a session
qualifies if its opening twenty turns *already* fit inside 900 s, and the first
qualifying session in corpus order is taken. 285 sessions fall in the volume
band and 150 of those fit in 900 s raw, so no clipping is needed to find one.
The cost of that criterion is stated in §9.6: this class is a session whose
opening turns arrive inside fifteen minutes, which is denser than the corpus
median.

**Short** is a pool of 64 session *openings*. Each selected session contributes
its **leading run** of turns at or under 4 096 tokens — turn 0, then turn 1,
stopping at the first longer turn — taken whole or not at all. On this corpus
every selected session's leading run is one turn long, so the class is 64
requests from 64 sessions; the rule would take a two-turn opening, with its
source interval intact, if a session had one.

The **inter-session alignment is the one invented quantity** in either class,
and it is declared rather than dressed up: all 64 sessions start at 0.0
(`session_start_at_zero`). The corpus times every request relative to its own
session's first turn and never records when a session began, so a pool must
choose something and no choice is chronology. A simultaneous start is visibly
constructed, and it puts 64 real opening lengths into one admission decision
against `max_num_seqs=32`. An earlier draft spaced the sessions one second
apart; that reads as a chronology it is not, and produced a ~1 req/s workload
whose throughput was the arrival rate at every width. §9.2 records the change.

### What the corpus carries, and what it therefore cannot claim

* `in` is a **token count, in tokens**, and it is block-aligned: for all 28 444
  top-level servable rows `in == len(hash_ids) * block_size` with
  `block_size = 64`, and the smallest is 128 tokens. Block-aligned is not the
  same as a block count, and nothing in the selector multiplies `in` by
  anything; the identity is re-checked on every request taken and a session
  that breaks it is refused. No prompt text exists in the corpus at all:
  lengths are **source-provided token counts** and the driver synthesises a
  prompt of exactly that many tokens (`atom/compass/workload.py`, verified per
  run by `--check-lengths`). Content is not reproduced and is not claimed to be.
* A row's `type` is `s` (28 173 rows), `n` (271) or `subagent` (1 697). **A
  `subagent` row is not a request** — no `in`, no `out`, only a summary of a
  delegated agent — but it wraps a list of **real nested requests**: 39 822
  across the corpus, on the session's own clock (for all 1 697 wrappers the
  first nested `t` equals the wrapper's `t`), never duplicated at top level.
  They are concurrent inference load that a top-level-only replay would drop
  silently. Neither class replays them, so instead the rule **refuses any
  window or opening segment that overlaps one**, and each manifest records
  `selection_nested_requests_in_scope: 0` beside the count in the selected
  sessions (long: 131 in the session, 0 in the window — all 131 begin at
  114 083.0 s, 113 783.4 s past the window's scope end; short: 3 016 in the 64
  sessions, 0 in any opening segment — the nearest starts 3.593 s after a
  segment ends). Overlap is measured against `t + api_time` of the last
  selected turn. §9.7 states what this excludes.
* `hash_ids` records the prefix blocks a turn shares with earlier ones — the
  reuse that makes an agentic trace agentic. **Prefix caching is off** in every
  cell, so that reuse is deliberately not exercised. This is a test of step cost
  and scheduling under a real length and arrival process. It is not a test of
  prefix caching, and a result here says nothing about one.
* `out` is what the session actually produced and is carried across unchanged.
  A turn that produced nothing has no TPOT to compare, so the rule rejects the
  whole session rather than raise a zero to a one: no output length in either
  file differs from its source, and both manifests record `outputs_altered: 0`.
* `t` is seconds since that session's own first turn. There is no absolute
  clock and no session start date anywhere in the corpus, which is why §1's
  short class has to declare an alignment.
* Request identity is position: `replay.py` writes only `arrival_s`,
  `input_tokens` and `output_tokens` into its artifact, so the session and turn
  each request came from survive in the registered file — and in each
  manifest's `provenance` list, which carries `(session, request_index,
  source_t_s, origin_shift_s)` per row — and are joined back by index. The
  file's digest is what ties a result to a request set.
* Replay is **open loop**. Arrivals are the trace's, not a function of how fast
  the server answers; no think-time feedback is modelled and none is claimed.

## 1B. The clients matrix, registered by digest

§1's two classes each offer one fixed amount of work, and both refuse to overlap
the delegated-agent requests the corpus contains (§9.7). This registration adds
an **offered-load axis** and stops refusing that concurrency: it replays it.

A **client** is one top-level root session. It is not a cap on in-flight
requests: every eligible descendant of a root is replayed at its own recorded
offset and contributes load, so a single client can have several requests in the
server at once. There is no request semaphore anywhere in the driver.

Selected by the rule in `scripts/compass/cc_traces_clients_workload.py`
(sha256 `3f54059888d51f01d7c2ff282231f4d385d9f3bbd6073fbf4b4d0321b557c37a`),
which loads `cc_traces_workload.py`
(sha256 `d0a1dc1be2ae777408266d8a18996bd55145cd688edddb1482a179a0ad8b09ff`) for
the corpus reader, the token identity check and the eligibility vocabulary it
already registered. Same corpus, same digest, same development-session
exclusion (`002001296e8a8c38ad9d7cc436d691afc602`). The rule's constants live in
the file; the command line takes only the class, the client count and the paths,
so a re-run cannot take a different slice, and `verify --corpus` re-runs the
rule and compares bytes.

### The episode, and how an episode is refused

An **episode** is a maximal run of API-leaf intervals `[t, t + api_time)` on the
root's clock joined by gaps of at most `G`. Both classes use **G = 0**: the
complete busy period, the interval during which the source had at least one
request outstanding, cut at the first instant it had none.

Episodes are cut from **all** API leaves — root turns and nested requests alike
— and then an episode is taken **entire or not at all**: if any request in it is
unservable (below one KV block, above the class's prompt limit, or `out < 1`)
the whole episode is refused. Dropping the bad request first would splice two
bursts the source separated and would silently shorten a branch, which is the
failure §9.7 was written about. Each manifest carries the census of what was
refused on the way to the selection, by category, with the range of values that
tripped each one.

One episode per root — its first qualifying one — so no session appears twice.

### The two classes

| | `clients_short` | `clients_large` |
| --- | --- | --- |
| gap `G` | 0 s (complete busy period) | 0 s (complete busy period) |
| prompt limit | every prompt ≤ 4 096 tokens | every prompt ≤ 262 144 tokens |
| also requires | — | span ≤ 60 s, ≤ 400 k input tokens, ≥ 1 prompt ≥ 32 768 tokens |
| concurrency | ≥ 1 descendant, source peak ≥ 2 | ≥ 1 descendant, source peak ≥ 2 |
| qualifying sessions in the corpus | 21 | 127 |

`clients_short` keeps the registered meaning of "short" — every prompt at or
under 4 096 tokens — and pays for it with what §9.1 already established: this
corpus has no short-input *session*. At G = 0 there are 99 concurrent episodes
under a 4 096-token ceiling across 21 sessions, and **every one of them is a
burst of sibling requests inside a single subagent branch, with no root turn in
it**. That is disclosed, not worked around (§9.8).

`clients_large` is the bounded large-prompt class. It is not §1's `long`: the
volume registration there was one session's twenty opening turns at 1.59 M input
tokens, and multiplying that by eight roots is not a PoC. This class instead
bounds each root — one complete busy episode, ≤ 60 s of arrivals, ≤ 400 k input
tokens — and requires a genuinely large prompt in it.

### The fixed ordered pool of eight roots

Client counts 1/2/4/8 take the **first C roots of one fixed ordered pool of 8**
per class, in corpus order. The counts are therefore nested — c1 ⊂ c2 ⊂ c4 ⊂ c8,
checked by request identity — so a comparison across client counts adds load
without changing which requests were already there.

`clients_large` (corpus line, session, requests, descendants, branches, source
peak, arrival span, input, output):

| line | session | reqs | desc | br | peak | span s | input | output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | `00ca01c4aaec…` | 7 | 6 | 3 | 3 | 38.5 | 379 840 | 13 887 |
| 4 | `0196085d85d2…` | 7 | 6 | 3 | 3 | 33.8 | 373 248 | 9 800 |
| 5 | `02bc0afb13f7…` | 3 | 2 | 1 | 2 | 36.1 | 86 592 | 2 963 |
| 7 | `0470d446a451…` | 4 | 3 | 2 | 3 | 5.5 | 294 912 | 1 380 |
| 8 | `04dba6fe6213…` | 6 | 6 | 1 | 5 | 11.0 | 48 384 | 13 624 |
| 12 | `069d7bf5f1ef…` | 3 | 3 | 2 | 2 | 5.5 | 302 592 | 3 773 |
| 14 | `07dd40536557…` | 3 | 3 | 3 | 3 | 0.6 | 157 952 | 557 |
| 15 | `085411a4cc4b…` | 4 | 4 | 3 | 3 | 3.3 | 205 120 | 1 353 |

`clients_short`:

| line | session | reqs | desc | br | peak | span s | input | output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 58 | `208d6e28928b…` | 2 | 2 | 1 | 2 | 0.5 | 4 864 | 2 061 |
| 60 | `212e46e294bd…` | 2 | 2 | 1 | 2 | 1.2 | 6 400 | 697 |
| 70 | `2654d8b687d9…` | 2 | 2 | 1 | 2 | 0.4 | 6 016 | 2 216 |
| 94 | `3cb1c6dc71a2…` | 2 | 2 | 1 | 2 | 0.5 | 5 888 | 2 444 |
| 106 | `46350c0355fa…` | 3 | 3 | 1 | 3 | 0.0 | 8 832 | 2 705 |
| 112 | `4b433e21f638…` | 2 | 2 | 1 | 2 | 0.0 | 4 864 | 1 330 |
| 121 | `4dcf5b262778…` | 2 | 2 | 1 | 2 | 1.7 | 5 888 | 2 024 |
| 125 | `509ad65c576a…` | 2 | 2 | 1 | 2 | 0.0 | 5 888 | 2 539 |

Corpus lines are 1-based. §1's manifests index the same sessions 0-based; each
clients manifest records both.

### The eight registered files

| class | clients | file (`atom/compass/`) | sha256 | reqs | root | desc | input | output | span s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short | 1 | `cc_traces_clients_short_c1.jsonl` | `fcc116645f4887b45742b7a0074b9d60692fabf2bfa80e021122bafa70188c18` | 2 | 0 | 2 | 4 864 | 2 061 | 0.5 |
| short | 2 | `cc_traces_clients_short_c2.jsonl` | `d808bb5e25e04c965e5c54753efb13e70977e36ef78b7aaf178fb6bfd407e189` | 4 | 0 | 4 | 11 264 | 2 758 | 1.2 |
| short | 4 | `cc_traces_clients_short_c4.jsonl` | `85ecf96ebbfeabc76d3398e6661ed6b3bae2f09d297fe2f2feabbf990f1567ea` | 8 | 0 | 8 | 23 168 | 7 418 | 1.2 |
| short | 8 | `cc_traces_clients_short_c8.jsonl` | `2b6f03c3c4791d80c5399d2a02a16e94c9bf3f05e3d153718723d62595df6980` | 17 | 0 | 17 | 48 640 | 16 016 | 1.7 |
| large | 1 | `cc_traces_clients_large_c1.jsonl` | `e232e42255e2bfaa9764d79093f1ea3ebed627e5dba4469e6d8a6845b3c65ab3` | 7 | 1 | 6 | 379 840 | 13 887 | 38.5 |
| large | 2 | `cc_traces_clients_large_c2.jsonl` | `cb4967ec0757ed478e95c664cad48269d8c53439aec2c94e1176036724dc38f9` | 14 | 2 | 12 | 753 088 | 23 687 | 38.5 |
| large | 4 | `cc_traces_clients_large_c4.jsonl` | `59e1a616fcdceda500d79bad7caf44a124633a2517ed40f6f9c50c7511d0e4c7` | 21 | 4 | 17 | 1 134 592 | 28 030 | 38.5 |
| large | 8 | `cc_traces_clients_large_c8.jsonl` | `3ba365adc2c39bf0fb56a5e9154b372f9d44291d4d4a0d2bdc1ed67a5ca7df95` | 37 | 4 | 33 | 1 848 640 | 47 337 | 38.5 |

Each has a `.manifest.json` beside it carrying the rule, the pool, the per-root
episode record, the refusal census, and a `provenance` list that keeps every
request's original JSON path (`/requests/5/requests/1`), its ancestry (root or
subagent, `agent_id`, `subagent_type`), its source model label and its
root-relative timestamp. `gaps_clipped: 0`, `outputs_altered: 0`,
`subagents_pruned: 0`, `requests_serialised: 0` in all eight.

### What is invented here, and what is not

**Within a root, nothing.** Every arrival keeps its source offset, including
descendant overlap; no gap is clipped, no branch is pruned, no request is
serialised, and no output length is changed.

**Across roots, one thing, declared:** each episode is rebased so its own first
arrival is 0.0, and all C roots start together. The corpus times a request only
against its own session's first turn and records no absolute clock, so a pool
must choose something and no choice is chronology. This is the same declared
alignment §1's short class uses (`session_start_at_zero`), extended to C roots.

**No causal contract is assumed.** The dataset card documents that surviving
`t` values keep their original relative offsets *including sub-agent overlap*,
and documents no parent/child completion dependency; it does not define
`api_time`, and `think_time` does not appear on it. So replay stays open loop: a
descendant fires at its recorded offset whether or not its parent has finished.
`api_time` is used for two things only — cutting the busy period, and reporting
the source's peak overlap as evidence that the slice contains concurrency.

**Source peak overlap is not a bound on the served engine.** Qwen's service
times are not the trace's, so the same arrivals may reach `max_num_seqs = 32` or
queue past it. The manifests say so at the field.

**Source TTFT is not a reference.** It exists only on top-level `s` rows, so
most replayed requests have none. It is not a gate input: §6's TTFT and TPOT
gates compare the **real Qwen serving run against the modelled Qwen serving
run**, over every replayed request including descendants. The corpus supplies
lengths and arrival structure, not a reference latency.

## 2. What is held out

The PoC's bet is: capture at TP=1, derive TP=2 and TP=4, predict. A cell that
calibrates at the width it then predicts does not test that bet, and every
long-input number reported so far did exactly that (`POC_STATUS.md` §3).

So for these cells:

* **Source calibration is TP=1 only.** The full-engine capture, the region model
  and the execution constants come from the TP=1 configuration.
* **Standalone primitive measurements at TP=2 and TP=4 are permitted** — a
  collective's price is a property of the hardware and the rank count and cannot
  be derived from a single rank. They are measured outside the target engine,
  on their own, and are declared as such in the calibration registry (§4).
* **No full-step or serving observation of the evaluated configuration may reach
  the predictor.** Not a step table, not a TTFT, not an admission constant
  fitted on the target run.

| axis | held out? | honest label |
| --- | --- | --- |
| configuration (TP=2, TP=4) | **yes** | a genuine prediction: nothing measured at that width inside the target engine is an input |
| configuration (TP=1) | **no** | the source configuration; its cells are a fit statistic and are reported as the calibration's own residual |
| workload (every class, §1 and §1B) | **yes, on the corpus slice** — no model was fitted or iterated on these requests | a prediction on unseen requests, from the same corpus and the same generating process as the development slice. It is not a new workload *family* |
| offered load (clients 1/2/4/8) | **yes** — the counts and the pool were fixed before any cell ran, and the pool is nested so load is added without changing the requests already there | an offered-load axis, reported as scaling; not a configuration and never pooled into a rank (§6) |
| model, hardware | no | one model, one device class throughout |

The workload axis is stated this narrowly on purpose. cc_pilot's twenty requests
are development data and this protocol does not pretend otherwise; what it adds
is that the acceptance requests are disjoint from them, drawn by a rule fixed in
advance, and that the configuration axis — the one the whole bet rests on — is
genuinely held out.

## 3. The cells

**Twenty-four cells: TP ∈ {1, 2, 4} × class ∈ {`clients_short`,
`clients_large`} × clients ∈ {1, 2, 4, 8}.** A cell is identified by all three,
and its directory is named `tp{tp}_{class}_c{clients}`.

The six cells of §1's matrix — TP ∈ {1, 2, 4} × {short, long} — keep their
registration, their stamps and their labels. They are not re-run and they are
not relabelled; they are simply not the matrix this section governs.

Each cell is run as `PROTOCOL.md` §§2–6 describe, with the engine configuration
unchanged from the matrix already registered there:

```
--model Qwen/Qwen3.8-27B --gpu-memory-utilization 0.90 --max-model-len 262144
--no-enable_prefix_caching --max-num-seqs 32 -tp {1,2,4}
```

Compilation stays at the engine's default level 3 (piecewise + CUDAGraph) and
chunked prefill at the engine's own default, which is what produced the
16 384-token prefill chunks the long cells report. No acceptance cell overrides
either. Prefix caching is off, on both sides, in every cell.

**Repeats: three real and three modelled per cell**, each from a fresh process.
Three on the modelled side too, and that is new — E1 reported four real runs
against one simulated one, and one simulated run is not a distribution. A
modelled side that is deterministic will show it by producing three identical
results, which is a finding and costs a few CPU minutes to establish.

Three is what **acceptance** is graded against, and it is not a caller's
number. `--repeats` below three still builds and runs — a one-repeat pass over
the matrix is a useful thing to want — but the plan says it is a diagnostic,
`cc_traces_run.py` stamps every execution record and artifact `diagnostic`
whatever `--purpose` was asked for, the cell verdict is written for
`diagnostic`, and `cc_traces_validate.py matrix` refuses any cell that does not
carry three repeats. A short run is evidence about the harness, never a cell of
the matrix.

Cells are run one at a time on an otherwise idle node, with the isolation audit
of `scripts/compass/isolation.py` over the whole window.

## 4. Calibration provenance, and the leakage rule

Every artifact the modelled server reads is declared in a **calibration
registry** — a JSON file listing, per artifact, its sha256, what it is, what the
server actually loaded for it, what measurement it came from, and which code
derived it:

```json
{"artifacts": [
  {"sha256": "…", "role": "table", "kind": "source_calibration",
   "measured_at_tp": 1, "produced_by": "run.py --sweep", "workload_sha256": null,
   "contents": {"table.json": "…"},
   "sources": [{"path": "/m/sweep_tp1.json", "sha256": "…"}],
   "code": {"scripts/compass/run.py": "…", "atom/compass/oracle.py": "…"}},
  {"sha256": "…", "role": "prices", "kind": "standalone_primitive",
   "measured_at_tp": 4, "produced_by": "primitives.py", "workload_sha256": null,
   "contents": {"prices.json": "…", "prices.tp4.json": "…"},
   "sources": [{"path": "/m/primitive_sweep_tp4.json", "sha256": "…"}],
   "code": {"scripts/compass/primitives.py": "…"}}
]}
```

`kind` is one of `source_calibration`, `region_model`, `overhead_constant`,
`derived_graph` (all of which must declare `measured_at_tp: 1` or null) and
`standalone_primitive` (which may declare any width, because that is the one
measurement a single rank cannot produce). An `overhead_constant` must also
write its `value` down: a constant nobody stated cannot be checked against
anything.

A digest identifies a file. It does not say what is inside it, and it does not
say what produced it — and an oracle option can name a *directory* or a file
that stands for several (`_artifact_digests` in the server expands
`prices.json` to `prices.*.json` and reports one rolled digest over the set). So
the declaration has to be transitive, and the parts of it that the server also
observes are cross-checked rather than believed.

The rule the validator enforces, fail-closed:

1. every digest the server reports in `oracle_option_sha256` appears in the
   registry — an artifact nobody declared is a refusal, not a warning; and
   every option the server reports *files* for reports a digest;
2. no artifact of a non-`standalone_primitive` kind was measured at a width
   other than the source width;
3. no artifact declares the acceptance workload's own digest as an input;
4. no artifact digest equals the real side's own step table in the same cell;
5. the modelled server reports `mode=predict` with a virtual clock, and the real
   server reports `mode=measure` on the wall clock;
6. **every file the server loaded for an option is enumerated** in that
   artifact's `contents`, name by name and digest by digest — a member the
   server read and the registry omits is a refusal, and so is one the registry
   claims and the server did not read. Where an option stands for more than one
   file, the declared members are rolled up the way the server rolls them and
   must reproduce the digest it reported;
7. **every artifact names its `sources` and its `code`**, each with a sha256 —
   a registry entry that asserts its own provenance and nothing else is
   metadata, not evidence;
8. rules 3 and 4 are applied **transitively**: a source or a loaded member whose
   digest is the acceptance workload, or this cell's own step table, is a
   refusal whether the predictor reaches it in one hop or two.

## 5. Cost, and what the ≥ 5× claim is measured against

A replay that is fast because someone else paid for the capture is not 5×
anything. Each cell keeps a `costs.json` recording, in seconds and separately:

| term | what |
| --- | --- |
| `capture` | the TP=1 tracing pass that produced the graphs |
| `calibration` | the source measurement pass (sweep / primitive pricing) |
| `derivation` | CPU derivation of this width's graphs from the TP=1 capture |
| `startup_real`, `startup_modelled` | process start to healthy |
| `load` | weight load and graph capture inside that startup |
| `execution_real`, `execution_modelled` | the measured window itself |

**The gate is the PoC's own and has not been moved**: a GPU-free replay of this
workload, after the capture exists, against serving it for real, at least 5×
faster —

```
replay_ratio = execution_real / (execution_modelled + derivation)  >=  5
```

`derivation` is inside the denominator because deriving *this* candidate's
graphs is work that asking this question costs. `capture` and `calibration` are
not: they are paid once, before any question is asked. They are not hidden
either — every cell's verdict reports `acquisition_s` and its two terms, the
`startup_inclusive_ratio` for a reader who wants end-to-end wall clock, the
`amortised_ratio` over a stated `amortised_over_cells`, and the
`break_even_cells` count at which the modelled path has repaid its acquisition.
An amortised ratio is a different and weaker claim than the gate, so it is never
what the pass criterion reads.

**GPU-free means device-free, not masked.** An empty `HIP_VISIBLE_DEVICES` says
what a library will enumerate; it does not say the process could not have opened
`/dev/kfd` itself. So the modelled side runs in a container with no `/dev/kfd`,
no `/dev/dri` and no NVIDIA nodes, and each cell carries a `gpu_free.json`
written *inside that container* by

```
python scripts/compass/cc_traces_validate.py gpu-free <cell-dir>
```

which records the device nodes it could see, every open driver handle held by
any process in the container, what torch reports, any masking variables (for the
reader — they are never what makes it pass), and the digest of each modelled
artifact it covers. The validator refuses a cell whose probe is missing, reports
a reachable device node or an open handle, could not ask the runtime, covers
none of the modelled runs, or covers one whose digest has since changed. What
the probe does *not* prove is stated in the file itself: that those artifacts
were produced by that exact process — only that they existed, with those
digests, in a device-free container when it was asked.

## 6. Gates

The numbers are those already registered (`PROTOCOL.md` §9); this protocol does
not relax any of them, and adds no new one.

* Throughput ≤ 10 %; TPOT/ITL ≤ 10 %; TTFT ≤ 15 %. The TTFT and TPOT gates
  compare the real and the modelled Qwen serving runs over **every replayed
  request, descendants included** — not root requests only, and not against any
  source latency.
* Spearman ρ ≥ 0.90 within comparable groups, over the TP axis.
* Top-1 configuration agreement with hardware, per objective.

**The comparable group is one (class, client count).** Client count is an
offered-workload axis, not a configuration: c1 and c8 contain different numbers
and mixtures of requests, so a rank computed over both pooled would be ranking
workloads as well as widths. Ranking, top-1 and regret are therefore computed
**within each fixed (class, clients) group**, comparing TP1/TP2/TP4 over the
same replayed request set — eight groups of three. Behaviour across client
counts is reported separately, as scaling, and is never pooled into a rank.

Both sides of a cell replay the **identical selected request identities** under
the identical timing policy: the same registered workload file, the same
arrivals, open loop, no semaphore.
* Non-KV memory terms ≤ 10 %, KV block count ≤ 5 %, and the deliberately
  infeasible configuration rejected for the same reason as the real engine.
* ≥ 5× replay speedup, `execution_real / (execution_modelled + derivation)`,
  after capture, produced in a container with no device in it (§5). Acquisition,
  startup-inclusive and amortised figures are reported beside it and are not the
  pass criterion.

**Ties are part of the gate, not an escape from it.** Two configurations are
tied on a metric when the real side's repeats overlap on it: the spread across
the three real repeats of one cell bounds what a difference between cells can
mean. A model that picks either member of a real tie agrees at top-1; a model
that *invents* a separation the hardware does not show fails, and so does one
that flattens a separation the hardware does show. **Regret** — the real metric
of the configuration the model chose, against the real metric of the best one —
is reported for every objective, in percent, alongside top-1.

## 7. Validation

`scripts/compass/cc_traces_validate.py` runs on CPU, reads artifacts only, and
is the thing that says whether a cell counts. It fails closed.

```
python scripts/compass/cc_traces_clients_workload.py verify --manifest <m> --corpus <c>
python scripts/compass/cc_traces_validate.py gpu-free <cell-dir>   # in the device-free container
python scripts/compass/cc_traces_validate.py cell   <cell-dir> \
    --class <clients_short|clients_large> --clients <1|2|4|8> \
    --calibration-registry <registry.json>
python scripts/compass/cc_traces_validate.py matrix <cell-dir>... --out verdict.json
```

`--class` and `--clients` together name the registered workload the cell must
have sent; a cell that sent another file is refused, not reported.

`cell` refuses, rather than reports, when: the workload sent was not the
registered one; a request has no engine record, no `usage.completion_tokens`, or
a token count that differs between the sides; a timestamp is missing, out of
order, non-finite, or inconsistent with its own derived `ttft`; an arrival is
negative or reorders the registered workload; the arrival barrier timed out; the
preparation drain is unproven or its boundary lies after a measured arrival; the
server build, model or calibration is unattributable; the modelled side was
paced, prepared, or given a warmup; any calibration artifact breaks §4; or the
cell carries no device-free evidence, or evidence showing a reachable device, an
open driver handle, or modelled artifacts it does not cover (§5).

`matrix` computes the ranking gates of §6 from the cells it is given, states
which cells it used, and refuses to report a ranking over an incomplete matrix —
all twenty-four `(tp, class, clients)` cells must be present, exactly once. It
ranks within each `(class, clients)` group and reports the client-count scaling
beside those groups, never as one of them.

## 8. Evidence each cell keeps

Both server logs and the cache-regime line read from them; the drained
preparation records and the drain assertion; both step tables with the
preparation boundary recorded; the `modelled warm state` line; both result
JSONs per repeat; `costs.json`; the isolation audit; the calibration registry;
`cc_traces_protocol.json` (this file's digest and every registered workload's
digest — §1's two and §1B's eight);
`gpu_free.json` from the modelled side's own container; and the validator's own
output, including for cells that failed. Failed artifacts
are kept and labelled, never deleted.

## 9. Findings that bound what this protocol can claim

Stated here because they were found while building it, and each one limits a
reading of the result. Every count below was measured on the registered corpus,
not assumed.

1. **The corpus contains no short-input session.** Across all 393 sessions the
   median count of requests at or under 4 096 tokens is 1, the longest run of
   such requests at the start of a session is 2, and the median request is about
   135 k tokens. A short-ISL class from this corpus is therefore necessarily a
   pool of openings across sessions, and pooling requires inventing when each
   session began — the corpus times a request only relative to its own
   session's start.
2. **The short class's inter-session alignment is declared, not observed.** All
   64 openings start at 0.0. An earlier draft spaced them one second apart;
   that was equally invented, read as a chronology it is not, and made the class
   arrival-bound — 64 requests of ~450 tokens over 63 s is ~1 req/s, at which
   rate the engine idles between arrivals at every width and throughput is the
   arrival rate at TP=1, 2 and 4 alike. A simultaneous start is the honest form
   of an invented placement and puts real opening lengths under real
   contention against `max_num_seqs=32`. It was changed before any acceptance
   run, on the argument above and on no measurement; the source lengths,
   outputs and within-session intervals were not touched. Changing it again
   after seeing a result would be the steering the rule exists to prevent:
   raising or lowering the load is a re-registration, not an edit.
3. **The short class's outputs are tiny** — median 21 tokens, max 39 — so its
   TPOT is measured over a handful of decode steps per request. That is what
   the trace says those turns are; it also means the short TPOT number is
   noisier than the long one.
4. **The long class is one session.** Cross-session mixing at long context
   would need the same invented placement as §9.1, and at 1.59 M input tokens a
   second session would double an already expensive cell. So this protocol does
   not claim anything about multi-tenant long-context interleaving.
5. **Lengths are source-provided token counts.** The corpus README says they
   track the real prompt size at about 1.00× on average and overcount by as
   much as 260 k tokens in the heavy-cache-write tail. They are block-aligned
   (§1), which is a property of how they were recorded and not a change of
   unit. Lengths here are accurate in distribution and approximate
   individually.
6. **The long class is denser than the corpus median, by construction.** A
   session qualifies only if its twenty opening turns already arrive inside
   900 s, so sessions with hour-long human pauses are outside the acceptance's
   scope. The alternative — clipping the gaps — would have made a constant in
   the selector, rather than the trace, set a quarter of the arrival spacings,
   and it was removed for that reason. Within the window the idle structure is
   the session's own: gaps run from 0.125 s to 26.670 s.
7. **Delegated-agent load is excluded by refusing to overlap it, not by
   dropping it.** 1 697 `subagent` wrappers hold 39 822 real nested requests on
   the sessions' own clocks. Neither class replays them — in the trace they are
   a different agent's calls, often to a different model, and not part of the
   session's own turn sequence — so both selectors refuse any window or opening
   segment that overlaps one, and both manifests record that the selected scope
   contains none. Two separate things are being said here and they should not
   be read as one: the wrapper *rows* are omitted because they are not requests
   — no `in`, no `out`, nothing to serve — whereas the nested requests inside
   them are real load, omitted only because no selected scope reaches them. In
   the long session all 131 begin 114 083.0 s in, 31.6 h after the window ends;
   in the short pool the nearest is 3.593 s past a segment's end. What this
   costs: neither class exercises the main-agent-plus-subagent concurrency that
   the corpus does contain. A result here is about a single agent's turn
   sequence, at these lengths and this spacing, and says nothing about a
   session running its delegates alongside it. **§1B's classes replay that
   concurrency rather than refuse it**, which is why they exist; this finding
   still bounds §1's cells.
8. **`clients_short` is subagent-only, and that is disclosed, not fixed.** Under
   a 4 096-token ceiling this corpus has no concurrent episode containing a root
   turn: at G = 0 all 99 qualifying episodes across 21 sessions are bursts of
   sibling requests inside one subagent branch, so all 17 requests of the c8
   cell are descendants and none is a root turn. The class keeps the registered
   meaning of "short" — every prompt ≤ 4 096 tokens — and the cost of keeping it
   is that this class measures a delegated agent's burst, not a user turn. A
   third class was considered and rejected: it would have added a workload
   family to an axis §2 already states narrowly.
9. **Short-class concurrency is only just concurrency.** Seven of the eight
   short roots have a source peak of 2, one has 3, and the spans are 0.0–1.7 s.
   These are real simultaneous requests from the trace, not a construction, but
   the per-root contention is small; what makes the c8 cell interesting is the
   eight roots aligned at 0.0, which is the declared alignment of §1B, not an
   observation.
10. **The large class's bounds are constants, and they select.** ≤ 60 s of
    arrivals, ≤ 400 k input tokens and ≥ 1 prompt of ≥ 32 768 tokens were fixed
    before the pool was read, and the refusal census in each manifest reports
    what they cost on the way to eight roots: 5 episodes refused for span
    (65.0–196.3 s), 3 for volume (533 312–919 424 tokens), 616 for containing no
    subagent request, 92 for never having two requests in flight. 127 sessions
    qualify, so the pool is the first 8 of many rather than all that exist.
11. **Client count is offered load, not a concurrency cap.** A client is a root
    session; its descendants are replayed at their own offsets and can be in
    flight beside it. So c8 does not mean eight outstanding requests — it means
    eight roots' worth of real arrivals, 17 requests in the short class and 37
    in the large one. The server's own admission (`max_num_seqs = 32`) is what
    bounds concurrency, on both sides.

## 10. Immutability

`scripts/compass/cc_traces_protocol.py register` writes
`atom/compass/cc_traces_protocol.lock.json` with this file's SHA-256; `verify`
recomputes it; `stamp` writes a cell's copy, including every registered workload
digest, so
a cell run against a re-emitted workload cannot read as a cell run against this
one. The superseded lock is kept inside its replacement.

Changing this file invalidates the registration. The correct response is a new
registration and a re-run of the affected cells. Editing it and re-registering
without re-running is not a way to make an existing result compliant.

## 11. What this protocol does not claim

It does not claim prefix caching works, or that agentic reuse is modelled: the
one feature that most distinguishes this corpus is switched off.

It does not claim a workload-family generalisation. The acceptance requests are
unseen, from the same corpus as the development slice.

It does not make TP=1 a prediction. TP=1 is the source configuration; its cells
report the calibration's own residual and are labelled that way.

It does not claim cold start is predictable, and it does not fold startup into a
serving result. Startup, capture and calibration are costs reported beside the
gate in §5, never inside it; per-candidate derivation is inside it.

It does not claim the delegated-agent concurrency the corpus contains for §1's
two classes: those refuse to overlap it rather than replay it (§9.7). §1B's
classes do replay it, and what they claim is bounded by §9.8–§9.11 — a
subagent-only short class, small per-root contention, and a large class whose
bounds are constants that select.

It does not claim a scaling law over client counts. Four points, nested by
construction, on two classes, are a scaling *report* beside eight independent
rank comparisons — not a model of how this engine behaves as load grows.
