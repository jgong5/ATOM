# ATOM Compass — Design Topic 18: One Pass Record and One Rule for Host-Paced Device Time

**Status:** drafted 2026-11-13 by an AI assistant from the owner's
[ruling on #2986](https://github.com/jgong5/ATOM/issues/2986#issuecomment-6841897073),
which fixes the scheme this topic states; the points the ruling left open are settled
here against the code at `67a59f2fc`. No code implements it yet: the implementation is
cut from *Summary*. Until it lands, `09` D54 and `01` D4 describe what the code does.

---

## D107. One pass record and one label rule for host-paced device time

### The problem

A device window holds every wait between its two events, including the device's wait
for a host that has not yet queued the work. A label holding such a wait prices host
time as device time, and the recurrence of `01` D4, which adds the host segments beside
it, charges that time twice. Today a separate rule keeps it out of each part of a
forward:

- a decode whose device start precedes its launch end is cut as launch-exposed, placed
  by the token copy wait's bound (`09` D54, `step_labels._host_cut`), which sits at
  least 781 us above the lead on #2072's MTP records
  ([#2986](https://github.com/jgong5/ATOM/issues/2986#issuecomment-6813130089));
- a decode's `propose` label is cut where the device reached the pass before the host's
  draft copy wait (`step_labels.host_paced_drafts`);
- a trace run's decodes take no label (`09` D54, *Traced prefill*);
- an MTP decode's label leaves out the device gap before its launch, which a plain
  decode's keeps, and the recurrence's check feeds that gap back as work queued ahead
  of the launch (`compass_queued_s`, `01` D4, *Accepted approximations*).

Each stood in for when the host enqueued its pass (the Ruling). A trace run's busy time
does not replace them on decode: it reads decode 1 to 6% over unprofiled time
([#3303's result](https://github.com/jgong5/ATOM/issues/2986#issuecomment-6825410534)).
So the record states, per forward and pass, when the host enqueued it, and one rule
labels every pass of every model from that.

### The pass registry

A **pass** is device work ATOM enqueues in one host call. The registry, on the Compass
side, lists per drafter flavor each pass's key, the line key timing it, the ATOM method
that enqueues it, whether it runs once per draft step, and its anchor on the
recurrence's host timeline. The recorder, `parity.busy`, the labels and pricing
iterate it; a new flavor adds its entries, with no ATOM change while ATOM enqueues each
pass from a method the recorder can wrap.

The flavors are ATOM's drafter classes, told apart by `Drafter.is_block_drafter`:
**block**, `DSparkProposer`, with the `MonoDSparkProposer` a mono deployment installs
over it, and **serial**, `EagleProposer`, which runs EAGLE3 and serial MTP. A model
without a drafter runs the target alone.

| flavor | pass | line key | enqueued by | per draft step | anchor |
|---|---|---|---|---|---|
| every model | `target` | `d_target_ns` | `ModelRunner.run_model`, then on an output forward `ModelRunner.postprocess` up to `forward_done_event` | no | entry |
| block, serial | `draft_kv` | `d_draft_kv_ns` | `Drafter.compute_draft_kv`, which `ModelRunner.forward` calls once `run_model` returns | no | entry |
| block | `propose` | `d_propose_ns` | `ModelRunner.propose_draft_token_ids`, up to its draft copy wait | no | exit |
| serial | `propose` | `d_propose_ns` less the summed `d_draft_passes_ns` | `ModelRunner.propose_draft_token_ids`, up to `EagleProposer.propose`'s first draft step | no | exit |
| serial | `draft_step` | `d_draft_passes_ns` | `EagleProposer.propose`, one step from its `_publish_draft_shape` call to the next step's, the last up to the draft copy wait | yes | exit |

An **entry** pass is enqueued before the forward reads its tokens, so the recurrence
places it among the boundaries the forward reaches from its entry: the launch, the
target's graph launch, the launch end. An **exit** pass is enqueued in `W_post`, after
the token read, so the recurrence places it at the forward's exit (*The recurrence*).

The sampler is no pass of its own, and on a middle chunk the `draft_kv` pass is the
draft's: both under *The draft-KV pass and the sampler*.

### The pass record

Per forward and registered pass, the recorder writes:

- `h0`, `h1`: the host times, on the step clock, at which it records the pass's start
  and end events;
- `g`: how many `hipGraphLaunch` calls the thread made between them, and the last one's
  entry and return;
- `d0`, `d1`: the pass's start and end events, as offsets on the device from the
  forward's launch event;
- `blocked`: the thread's time between them in each blocking HIP call that waited on
  the device (G, below).

Where the events sit:

- `target`: from `run_model`'s entry, where the launch event is recorded today, to the
  target's end: where `postprocess` has recorded `forward_done_event`, just before
  `prepare_sampled_ids` waits for the previous output's token copy; on a forward that
  reads no tokens, the first drafter pass's start, or the forward's exit without a
  drafter. The recorder records that end event on every output forward, a plain
  model's too: today `StepRecording.__init__` records it only under a drafter, so a
  plain line's window runs on to the exit event, past `W_post`.
- `draft_kv`: around `compute_draft_kv`, as `StepRecording._draft_pass` times it today.
- `propose`: from `propose_draft_token_ids`'s entry to the draft copy wait's entry
  (`recv_draft_token_ids`, `t_draft_recv_ns`). The wait is for the previous forward's
  draft copy (`tokenIDProcessor.prepare_draft_ids`); this pass's device work is all on
  the stream before it, so its end event moves there from the call's return. A forward
  that reaches no such wait ends it at the call's return. `d_propose_ns` keeps the
  whole draft under every flavor; under the serial flavor the `propose` pass ends at
  the first draft step's start, and its label is `d_propose_ns` less the summed
  `d_draft_passes_ns`, as `step_linear_fit` derives that prologue today (`_rest`).
- `draft_step`: at each `_publish_draft_shape` entry (`StepRecording._serial_pass`),
  under every serial flavor; today the recorder times them under EAGLE3 alone.

So every pass's work is queued by its `h1`, and its `L` below never comes early.

`g` comes from the blocking-call tool (`atom/compass/parity/blocked.cpp`), which also
hooks `hipGraphLaunch` and `hipGraphLaunch_spt`: per thread, a count and the last
call's entry and return on `CLOCK_MONOTONIC`, the step clock's
(`RecordingModelRunner.step_clock_ns`). They stay out of its blocked sums, which
`W_launch` subtracts. torch's `CUDAGraph.replay` calls `hipGraphLaunch` for every graph
ATOM replays: a `ReqGraph` bucket, each `TokenGraph` segment, a `DraftGraph` pass and the
mono runners' own captures. A worker that did not load the tool records no `g`; its
lines are read as old records.

### The rule

Per decode step of an ordinary record, and per registered pass:

- **Kind.** *Single* when the call made exactly one graph launch, *multi* otherwise.
- **E and L.** `E = L = g.entry` when single; `E, L = h0, h1` when multi.
- **F**, the device end of the work before the pass: for the `target`, the previous
  forward's exit event (this launch event less `d_gap_ns`); for any other pass, its own
  start event `d0`, which completes once everything queued before it has.
- **The lead** places a host time `t` on the device clock at `t - lead`.
- **Label**: `d1 - max(F, E - lead)` when single, or when `F >= L - lead`; otherwise the
  pass is excluded on that step. A pass's label leaves out the passes inside it: the
  `target`'s, on an output forward, its `draft_kv` window.
- **G.** A pass whose call waited on the device in a blocking HIP call is excluded, and
  so is every pass inside it.
- **The single bound.** A single pass reports `max(0, g.return - max(F + lead,
  g.entry))`, the most its device waited inside the graph launch call. Its label keeps
  that wait.

An excluded pass leaves the others on its step standing. A prefill step takes no window
label (*Trace runs*). A later pipeline stage's `target` still leaves out its receive's
device wait (`d_recv_ns`), and under TP and DP a window still holds its collectives'
waits for the slowest rank, which `09` D54's max-affine fit models.

The rule replaces the decode cuts. On a multi `target`, `F >= L - lead` asks what the
launch-exposed cut asked, whether the device was still busy once the host had queued the
pass, over the pass's whole call and by the one lead. A single `target` is never cut:
its label starts where the host's graph launch let the device start, so the device gap
before it is counted where the device was busy (`F >= E - lead`) and dropped where it
idled, on MTP and plain records alike. On an old record the `propose` rule is
`host_paced_drafts` exactly (*Old records*).

#### Settled: a call of several graph launches is multi

The ruling makes a step single when its call made a graph launch, `g` the last one. A
`TokenGraph` replay (`atom/utils/region_graph.py`) runs one graph per segment and an
eager region between segments, which the host launches as it walks them: a prefill
bucket, the token-graph decode `--cudagraph-mode` TR_HYBRID runs
(`ModelRunner._tr_hybrid_active`), and DeepSeek-V4.1's prefill, its body on a token
graph and its decoder on a request graph (`launch_features.DeepseekV41`). Taking the
last launch's entry as `E` there would drop from a host-paced label every segment the
device ran before it. So only a call of one launch is single: a `ReqGraph` bucket,
recorded as one graph, or a `DraftGraph` pass.

#### Settled: one lead

Today `_host_cut` places the device by the token read exits and `_timed` by the launch
bounds and MTP status waits. One lead serves every record: `_timed`'s, the largest
launch bound moved along the tightest recent upper bounds, those being the MTP status
wait returns where the record stamps them and else the token read exits `_host_cut`
reads. A device event runs no earlier than the host records it, so the lead is the low
end of the clocks' bracket: it places `E` and `L` at their latest on the device clock,
giving the smallest label and the strictest test. No margin widens either (C).

#### Settled: "waited on the device" (G)

The tool times every call to a blocking HIP API, whether it waited for the device or
found its work done. On #2256's DPA8 cell every launch TBO does not split spends 2 to
3 us in `hipEventSynchronize`
([#3069's handoff](https://github.com/jgong5/ATOM/issues/3069#issuecomment-6755273142));
counting every call would exclude each of those decodes. So the tool asks first:
before it forwards `hipEventSynchronize` or `hipStreamSynchronize` (and its `_spt`
form), it queries the same event or stream (`hipEventQuery`, `hipStreamQuery`) and
counts the call under `blocked` only where the device had not finished. A device or
context synchronize, a synchronous copy and a free count always. `blocked_ns`, which
`W_launch` subtracts, keeps every call's time as today.

Under the mono DSpark flavor this excludes every output forward's `target`, `draft_kv`
and `propose`: `V4MonoSession.write_draft_context`, inside the draft-KV call, reads the
request segments to the host (`cu_seqlens.tolist()`), a synchronization its bring-up
pays on purpose, and `consume_draft_context` and `forward_draft_block`, inside the
propose call, read anchors the same way. Each read waits for the target's graph, after
which the device idles while the host queues the rest, so the windows hold host time.
Those passes take labels once ATOM drops the reads, an ATOM change filed under
`atom-issue` like any wait in a launch ([principle 9](README.md#design-principles)).

#### Settled: the draft-KV pass and the sampler

The ruling's open point. `ModelRunner.forward` calls `Drafter.compute_draft_kv` once
`run_model` returns; `postprocess` then launches the sampler (the `Sampler`, under
speculation the `RejectionSampler`, with TP above 1 the token broadcasts, and the
logprobs) and records `forward_done_event`. Both run inside the target's window, which
today runs to that event less the draft-KV window (`StepRecording.step_stamps`).

- **The sampler is the `target`'s tail.** The target's vectors price its device time,
  as today, and `W_launch` holds its host time, after the target's graph launch. So the
  `target`'s call runs to `forward_done_event`. Splitting it off would need a law of
  its own for work no backend prices apart; under A it waits for nothing.
- **On an output forward, `draft_kv` is a pass inside the `target`'s call.** Its label
  is its window wherever the target's label stands, and it is not tested apart: on a
  single `target`, A puts no wait anywhere after the graph launch, the draft-KV pass
  included; on a multi one, `F >= L - lead` with the target's `L` after the draft-KV
  call already rules a wait out. Its own `E` and `L` feed A's report. In the
  recurrence it sits inside `D_target`, priced by the draft law's `draft_kv` vector
  among the target's terms, as `01` D4 has it today, with no start of its own.
- **On a middle chunk, `draft_kv` is the draft's first pass.** The forward returns
  before `postprocess`, so the target's window ends at the draft-KV pass's start and the
  pass is the draft label, as `09` D54 has it. A middle chunk is a prefill, labelled by
  its trace (`b_draft_kv_ns`). The recurrence starts it at `max(target_end, exit)`, the
  exit of a middle chunk being its launch end: it reads no tokens and has no `W_post`.
- Under the serial flavor, `EagleProposer.compute_draft_kv` returns at once on a forward
  that samples, since `propose` redoes that pass, so its decode's `draft_kv` is empty.

### The recurrence

Each pass ends at `end = max(max(prev_end, E') + D, L')`, `D` its price, `E'` and `L'`
the first boundaries of the recurrence's host timeline at or after `E` and `L` from the
pass's anchor. `01` D4's boundaries are the launch (after `W_pre`, and `W_inputs` under
MTP), the launch end (after `W_launch`) and the exit (after `W_post`); D107 adds one:

- **`W_graph`**, the part of `W_launch` from `run_model`'s entry to the target's graph
  launch's entry. Like `W_recv`, it is inside `W_launch` and never charged beside it. A
  host law fits it, per kind and on `(1, tokens, seqs)`, from the single targets of
  records that carry the pass record.

Per pass:

| pass | `E'` | `L'` | start | end |
|---|---|---|---|---|
| `target`, single | launch + `W_graph` | the same | `max(ends[-1], launch + W_graph)` | start + `D_target` |
| `target`, multi | launch | launch end | `max(ends[-1], launch)` | `max(start + D_target, launch end)` |
| `draft_kv`, output forward | inside the `target` | — | — | inside `D_target` |
| `draft_kv`, middle chunk; `propose` | exit | exit | `max(target_end, exit)` | start + `D` |
| `draft_step` | exit | exit | the previous pass's end | start + `D` |

**The target's kind is predicted, and a fit checks it.** A simulated run counts no
launch, so the target's kind comes from the step's view, by the model's launch provider
(`backends.launch_features`), as the provider predicts its launch features. By default a
step is single where its view replays a request graph (`capture_rung` set) and the
record's graph sets (`parity.recorded_graphs`) hold no decode token-graph bucket, which
a token-graph decode records; multi otherwise. A model whose request graph runs
otherwise states it in its provider: DeepSeek-V4's mono request graph
(`deepseek_v4/mono/engine.py`, its graph's `replay`) runs `engine.session.forward`
eagerly, or one graph per lane, where `V4MonoEngine.prepare_step` settles `use_graph`
false on host state no view carries. On a record with the pass record, a fit refuses a
step whose counted kind differs from the predicted one, naming the step and both kinds,
as `step_linear_fit.fit_host` refuses a step whose launch features differ from `predict`.
So no step is labelled multi and charged single, nor started after a `W_graph` no record
measured.

A drafter pass starts at or after the exit, so its kind moves nothing there: the serial
steps sum to the draft's `D`, and the draft starts no earlier than the exit as `01` D4
rules ([#2917](https://github.com/jgong5/ATOM/issues/2917)).

Against today's recurrence, a single target starts `W_graph` later, a multi target ends
no earlier than its launch end, and the check stops feeding an MTP decode's `d_gap_ns`
as `compass_queued_s`: it feeds each pass the rule's label, which holds that gap where
the device was busy. Host laws without `W_graph`, fitted on old records, start every
target at the launch, as their labels' `E` does.

### Trace runs

A trace run labels its prefill steps only, every pass of them, and its decodes not
(#3303). Each kernel, memory copy or memory set of a pass is a pass of one call, its
launch call: its own `hipLaunchKernel`, or the `hipGraphLaunch` that replays it. The gap
before it counts whole when `F >= L`, `F` the end of the rows before it on the rank's
kernel streams and `L` that call's return, both on the trace's clock; otherwise it
counts up to the cap, the rank's median such gap past one rounding step
([#3312](https://github.com/jgong5/ATOM/issues/3312)), for the device's own share of a
gap the host paced. A gap between two kernels of one graph launch that opened before
the launch returned is such a gap: the trace cannot tell a gap the device ran from a
wait on the launch's submission (E). This is `parity.busy`'s rule today, with #3312's
cap; collectives are aligned across ranks and each pass takes its largest rank, as
there.

### Old records

A record without the pass record keeps every law fitted on it valid:

- **`target`**: its kind from `rung`, single where it is set; `E = h0`, the launch's
  entry, single or multi; `L` the same when single, and when multi the token copy wait's
  entry, or `run_model`'s return on a forward that reads none. `d1` is `d_target_ns`'s
  end, or on a plain record `d_busy_ns`'s, at the exit event. Host laws fitted on such
  records carry no `W_graph`.
- **G** reads `blocked_ns`, which spans the `target`'s call but sums every hooked call,
  whether it waited or found its work done. So on an old record G counts only the calls
  that count always: device and context synchronizes, synchronous copies and frees. An
  event or stream synchronize there stays uncounted, as it would where its query found
  the device done, and G's report gives their count and time per record.
- **`draft_kv`**: its window `d_draft_kv_ns`, standing with the target's.
- **`propose`**: multi, `L = t_draft_recv_ns[0]`, `F` the target's end with its draft-KV
  pass (`start + d_target_ns + d_draft_kv_ns`). The rule is then `host_paced_drafts`.
- **`draft_step`**: no host stamps; each stands or falls with its `propose`, as today.

An old record's kind is the default prediction, with no count to check it: `rung` is
also set on a token-graph decode, and an old record states its graph sets only under a
provider that declares launch features, so one run under TR_HYBRID reads its decodes
single; a record with the pass record has its count checked. Each law's provenance
names the `E` its labels took (`launch` or `graph`), and a fit given records of both
kinds refuses, naming them.

### Assumptions and their report items

The fit's report gives each per record and pass.

| | Assumption | Report item |
|---|---|---|
| A | No wait after a pass's last graph launch: the work a single pass enqueues after it (`compute_logits` when the graph returns none, the draft-KV pass, the sampler, a `DraftGraph` epilogue) finds the device still busy. | The single bound, p50 and p90; the share of single targets whose draft-KV pass fails `F >= L - lead` on its own call, where the device waited before it; and the tail, the share of single passes whose end event completed no later than the host recorded it, `d1 + lead <= h1`, where the label's end holds the device's wait for the host. |
| B | The recurrence anchors a pass no earlier than the host enqueued it. | `E' - E` per pass on the record, the exit read as the recurrence places it (less the draft copy wait), p50 and p90, and the share of steps where the device reached the pass before `E'`, so that the lateness moves its start. Zero by construction on the `target`. |
| C | The lead's error is small, and no margin is taken. | The bracket: the tightest recent upper bound less the lead, p50 and p90; and the steps whose label start or exclusion changes with the lead anywhere in its bracket. |
| D | A multi pass the host paced ends by `max(start + D, L')`. | On each multi pass the rule excludes: `d1 + lead - L'`, how far past the host's boundary its device ended, p50 and p90. |
| E | A trace's graph-internal gaps are worth the cap. | Per trace run and pass: the gaps inside a graph launch opened before it returned, their count and their sum whole and at the cap, as shares of the pass's busy time. |
| F | A trace labels prefill only, its profiler error small there. | `09`'s profiler check, per pass, on prefill. |
| G | A call that waits on the device excludes its pass. | The share G excludes, the APIs that waited, and their time p50 and p90; on an old record, the event and stream synchronizes it left uncounted. |

Beside them, the **excluded share**: per record and pass, steps, single, multi,
labelled, excluded by `F < L` and by G, on MTP and plain records.

### The per-pass table

Each pass on an ordinary record's decode, with the record that carries the pass record
first and an old record after. `F`, `E` and `L` as in *The rule*; *rec. start* is the
recurrence's.

| pass, case | kind | anchor | `E` | `L` | label | rec. start |
|---|---|---|---|---|---|---|
| `target`, a request-graph decode (`ReqGraph`, one launch) | single | entry | `g.entry` | `g.entry` | `d1 - max(F, E)` less `draft_kv`, `F` the previous forward's end | `max(ends[-1], launch + W_graph)` |
| `target`, an eager or token-graph decode | multi | entry | `run_model`'s entry | the target's end record | the same if `F >= L`, else excluded | `max(ends[-1], launch)`, ending no earlier than the launch end |
| `target`, any prefill | by its launches | entry | — | — | trace busy, per kernel | as its kind |
| `draft_kv`, output forward (empty under serial) | multi in every flavor today | entry, inside `target` | its call's `h0` | its call's `h1` | its window, standing with the `target`'s | inside `D_target` |
| `draft_kv`, middle chunk | multi | entry, whose next boundary is the exit | — | — | trace busy, the draft's | `max(target_end, exit)` |
| sampler | the `target`'s tail | entry | — | — | in the `target`'s | inside `D_target` |
| `propose`, block, replayed (`DraftGraph.run`) | single | exit | `g.entry` | `g.entry` | `d1 - max(F, E)`, `F` its start event | `max(target_end, exit)` |
| `propose`, block, eager (no graph at its batch) | multi | exit | its `h0` | the draft copy wait's entry | `d1 - F` if `F >= L`, else excluded | `max(target_end, exit)` |
| `propose`, block, mono (`MonoDSparkProposer`) | by its launches | exit | as its kind | as its kind | excluded by G on every step, as are the output forward's `target` and `draft_kv` (*Settled: G*) | `max(target_end, exit)` |
| `propose`, serial, to the first step | multi | exit | its `h0` | step 0's `h0` | the same | `max(target_end, exit)` |
| `draft_step` 0, serial (the draft model, eager, over the target's stream) | multi | exit | its `h0` | step 1's `h0`, or the draft copy wait's entry | the same | the previous pass's end |
| `draft_step` 1 on, serial | single where `DraftGraph.run` replays, else multi | exit | `g.entry`, else its `h0` | `g.entry`, else the next `h0` or the draft copy wait's entry | by its kind | the previous pass's end |
| old `target` | single where `rung` is set, else multi | entry | `t_launch_ns[0]` | `E` if single, else `t_recv_ns[0]` or `t_launch_ns[1]` | by its kind, `d1` at the exit event on a plain record | `max(ends[-1], launch)`, no `W_graph` |
| old `draft_kv` | — | entry, inside `target` | — | — | `d_draft_kv_ns`, standing with the `target`'s | inside `D_target` |
| old `propose` | multi | exit | — | `t_draft_recv_ns[0]` | `d_propose_ns` if `F >= L`, `F` the target's end with `draft_kv` | `max(target_end, exit)` |
| old `draft_step` | — | exit | — | — | standing with the `propose` | the previous pass's end |

### What D107 replaces

- `09` D54, *The label*: the decode's launch-exposed cut and the token copy bound that
  placed it; the MTP decode label without its gap; the `propose` label's host-paced cut;
  the trace-run decode cut, which "a trace run labels prefill only" restates; the
  traced prefill's gap rule, which *Trace runs* restates with #3312's cap.
- `01` D4, *A forward's charge*: the target's start, now at its graph launch for a single
  target, and a multi target's end, now no earlier than its launch end; and the check's
  `compass_queued_s` under *Accepted approximations*.
- `step_labels`: `_host_cut`, `host_paced_drafts`, the `exposed` cut and the table's
  `launch_exposed`; `parity.recurrence`'s `waited` and `compass_queued_s` feed.

### Summary

What `plan` cuts. Old records stay labelled, by their rule, at every step.

1. **The registry**: one Compass module holding the table under *The pass registry*,
   which the rest read. CPU only.
2. **The recorder**: `parity/blocked.cpp` counts graph launches and keeps the last one's
   entry and return, and queries before a synchronize (G); `parity/blocked.py` reads
   them; `parity.StepRecording` writes the pass record per registered pass, records the
   target's end on every output forward, moves the `propose` end event to the draft copy
   wait, and stamps draft steps under every serial flavor; `parity._record` writes the
   graph sets (`recorded_graphs`) on every record, not only under a provider that
   declares launch features; `parity/README.md`. One GPU session: the overhead against
   a record without it, then new records of #2072's and #2415's DeepSeek-V4.1-Flash
   DSpark TP2 recipes, mono on and off, of
   [#3256](https://github.com/jgong5/ATOM/issues/3256)'s MiniMax-M3 EAGLE3 recipe for
   the serial flavor, and of one without a drafter, which every later item's named
   result reads.
3. **The labels**: `step_labels` applies *The rule* with one lead and reads old records
   by theirs; `launch_features` predicts the target's kind; `step_linear_fit` reports
   *Assumptions and their report items* and the excluded share, and refuses a fit mixing
   old and new records or holding a step whose counted kind differs from the predicted
   one. CPU only, on item 2's records.
4. **Trace runs**: `parity.busy` iterates the registry's passes and takes #3312's cap,
   folding #3312 in; it reports E. CPU only.
5. **The recurrence and the host laws**: `step_linear_fit.fit_host` fits `W_graph`;
   `HostDeviceRecurrence.device_end` starts a single target at its graph launch, ends a
   multi one no earlier than its launch end, and drops `compass_queued_s`;
   `parity.recurrence` feeds each pass its label; each law's provenance names its `E`.
   CPU only, then `01` D4's check on item 2's records.
6. **The design text**: `09` D54 and `01` D4 state the rule once items 3 and 5 land, and
   this topic's status says so.

Items 2 to 5 follow item 1; items 3 and 5 read item 2's records for their named results.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D107 | Every device label and the recurrence count each host-paced wait once, by one rule over one record per pass. A Compass registry lists per drafter flavor each pass's line key, enqueuing method, per-step flag and anchor (entry or exit). The record gives each pass its host call's `h0` and `h1` at its events, its graph launches (a count, the last's entry and return) and its device events. A call of one graph launch is single, `E = L = g.entry`; any other is multi, `E, L = h0, h1`; `F` the device end before it. The label is `d1 - max(F, E)` when single or `F >= L`, else excluded, and a call that waited on the device excludes its pass; a single pass reports its in-call wait bound. The target's kind is predicted from the step's view by the model's launch provider, and a fit refuses a step whose counted kind differs; on an old record G counts only the calls that always wait. The sampler is the target's tail; on an output forward the draft-KV pass sits inside the target, standing with it; on a middle chunk it is the draft's first pass. One lead, the launch bound moved along the tightest upper bounds. The recurrence ends each pass at `max(max(prev_end, E') + D, L')`, adding `W_graph`; trace runs label prefill only, per kernel, graph-internal gaps before the launch's return at #3312's cap; old records keep their rule. Assumptions A to G each carry a report item. | 2026-11-13 ([#2986](https://github.com/jgong5/ATOM/issues/2986#issuecomment-6841897073)) |
