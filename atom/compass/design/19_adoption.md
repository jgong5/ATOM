# ATOM Compass — Design Topic 19: Adoption, a User's Configuration Space Searched

**Status:** drafted 2026-11-13 by an AI assistant from the decisions of the design issue
[#3355](https://github.com/jgong5/ATOM/issues/3355), which the owner took from 2026-11-10
to 2026-11-13, and the owner's ruling on its design PR
[#3363](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6845852318). No code
implements it yet: its tasks are cut from #3355 after v1.

**Depends on:** `07` (the store, D37), `08` (D44, D45, D45.1, D50, D50.1), `13` (D79,
D80), `16` (D95.1), and the M3-Agentic stage's `03` D13.1, `06` D35.1 and `14` D87.1.

**Scope.** How a user finds the configuration to deploy: one command searches a space
they provide, simulates every candidate whose law exists and recommends one, while the
calibration a candidate lacks stays apart; what each run writes for a person to read;
where the tool is built and shown. The development nodes are crsuse2-m2m-v3-005 and -013.
What this round leaves out is #3355's *Deferred* list; upstream work the search meets,
such as DeepSeek-V4's P/D issues and `inferencex_export` labelling every row single-node,
is tracked apart.

| Touches | How | Decision |
|---|---|---|
| `13` D79: precedence has no level but CLI and artifact, and no Compass config file | not amended, as argued: the space file is the search's input, as traces are a client's, and each run takes its flags from `compass advice` | D110 |
| `13` D80: no command starts the client or takes a workload | `compass advice` does both | D109 |
| `13` D80: a prefill or decode `run` refuses without `--join` | not under `--check`, which resolves the side as `deploy.sh` does | D109 |
| `13` D80: what `run`, `record` and `deploy.sh` write | each writes its per-request split and `report.html` when its run ends | D114, D115 |
| `08` D50: the cell matrix | a MiniMax-M3 showcase set: the recommended configuration and the runner-up | D117 |

---

## D108. Fitting apart from searching

**Rule.** A candidate, one configuration of the user's space, is simulated once its law
exists; fitting and searching meet only through the artifact store and a calibration's
`status` file (D113). The search consumes laws and calls the reuse check (D112); it owns
no calibration, fitting, reuse judgement or version management, and reaches profiling
(D114) and the views (D115) only through files in each run's directory. A user guide
comes first, and no skill: `scripts/compass/ADVICE.md`, linked both ways with
`QUICKSTART.md` and #2893's page, gives each step one command and its expected output,
has a section on filling the space file (D110) and states D109's limits, exact enough
that an agent following it fills the DeepSeek-V4-Pro template's space file from the
recipes D116 names. An agent shows the user a printed calibration's GPU time, starts it
in the background only on the user's yes, reruns `compass advice` later rather than
waiting, and adds no judgement of its own.

**Reason.** A calibration spends GPU hours and a search CPU minutes: apart, each half is
built and replaced alone, and a rerun takes up whatever calibration has finished; a guide
an agent follows saves the user the split's round trips. **Basis.** The owner on #3355 and
#3363; `07` D37; #2317 (`calibrate.sh` plans nothing). A skill needs its own ruling: the
[#2722 Ruling](https://github.com/jgong5/ATOM/issues/2722#issuecomment-6678010091)
removed `compass-onboard`.

---

## D109. `compass advice`: one command, three steps

**Rule.** `compass advice SPACE OUT` is a `compass` subcommand whose steps are its own:

1. **Check.** Expand SPACE (D110) and give each candidate one state: *searchable*;
   *missing a law*, with the `calibrate.sh` command of D113's record list;
   *calibrating*, with its `status` file's progress; or *refused*, with ATOM's or
   Compass's refusal verbatim. It runs the reuse check (D112), `compass run --check`,
   which on 1P1D resolves each side as `scripts/compass/deploy.sh` does, and a start with
   no traffic, for what only a start shows: arguments ATOM rejects, a model that does not
   fit memory, a configuration Compass refuses, such as `--enable-dynamic-chunking`. It
   never calibrates.
2. **Simulate.** Each searchable candidate at each concurrency, a fresh server per point
   (`13` D80): `compass run`, or `deploy.sh run` on 1P1D, driven by one `client.sh` run,
   points in parallel over the host's cores and ports, a failed one retried and then
   listed with its reason, a finished one skipped. No `--step-roofline`: a refusal means
   a calibration is missing, and the law is completed, as stages 3 (#975, #2660) and 4
   (#2544) completed theirs.
3. **Results.** D111's, with D115's dashboard; a changed SLO reruns this step only.

OUT holds `dashboard.html`, the recommendation, the result table and
`runs/<config>-c<concurrency>/`, each point's run directory. The search runs at the
integration head (D112); on the target machine type, since host laws key on the host's
CPU and another CPU is refused by name until #3122's code task; in a GPU container with
no GPU compute, since a simulation imports aiter, which needs a ROCm driver (no GPU at
all is #2893's). It wraps the commands it runs, so no user step needs `python -m` (the
owner's ruling on #1108, recorded in #1527).

**Reason.** A command rerun until every candidate is searchable leaves the user nothing
to compute; with no roofline every point is priced by its own law; a later client run
meets the prefix cache an earlier one left (`03` D13). **Basis.** The owner on #3355; the
[ruling on #2312](https://github.com/jgong5/ATOM/issues/2312#issuecomment-6462166589);
`08` D50.1. `run --check` refuses a P/D side's arguments, and without
`--kv-transfer-config` DeepSeek-V4's index cache resolves to fp4, not fp8
(`atom/config.py`), a value both the law's and the readings' keys hold.

---

## D110. The space file

**Rule.** The user provides the space, by hand or through an agent that fills it from a
recipe or from the user's real-run arguments by following the guide, not a skill (D108);
Compass ships none as a default. It states the model, the machine, the shared store
(D113), the workload (traces and concurrencies), the SLO, the ATOM arguments every
candidate shares, the dimensions searched with their values, exclusions, and arguments
that vary with the concurrency, as MiniMax-M3's `--max-num-seqs` is twice it. It expands
into the product of the dimensions less the exclusions, always into the same candidates
in the same order, and refuses by name an unknown field, contradictory values or an empty
expansion. Each candidate names its ATOM basis, the recipe its arguments follow, and each
departure with its reason. Its format is T108's. It is the search's input, as traces are
a client's: each candidate's run takes `compass run`'s flags and ATOM's arguments from
`compass advice`, so the file adds no level to `13` D79's precedence and is not the
Compass config file D79 rejects.

**Reason.** The user, not Compass, knows what they would deploy; a template (D116, D117)
shows the shape, and an agent saves the user copying arguments by hand. **Basis.** The
owner on #3355 and #3363; `13` D79.

---

## D111. Results: the frontier, the recommendation, the runner-up

**Rule.** A point's metrics come from its client export. InferenceX's are computed by
#3264's code (`atom/compass/validate/curves.py`), which follows InferenceX's
`request_metrics.py` and grades the M3-Agentic stage (`08` D45.1): throughput per GPU of
the input and output tokens, cache hits included; P90 interactivity, one over the P90 ITL,
and the normalised one, one over the P90 of each request's end-to-end latency per output
token; P90 TTFT. The search adds two of its own from the same export: throughput per GPU
of the tokens the engine computed, the input and output tokens less
`usage_prompt_cache_read_tokens`; P90 end-to-end latency, from `request_latency`. A 1P1D
point counts both sides' GPUs (T110).

- **Frontier**: throughput per GPU, InferenceX's, against P90 interactivity; the points
  no other dominates on both.
- **Recommendation**: the most throughput per GPU, InferenceX's, among points that meet
  the SLO, its interactivity bound on P90 interactivity, and have no refusal.
  **Runner-up**: at its concurrency, another configuration's point chosen by the same
  rule.
- **A point with a refusal** is a coverage report (`08` D50.1), out of the frontier,
  recommendation and runner-up, listed with what it lacks and the command that
  calibrates it.
- **The result table**, a row per point: configuration, concurrency, both throughputs,
  both interactivities, P90 TTFT, P90 end-to-end latency, SLO met, on the frontier,
  grading state, pricing-method shares under `08` D50.1's names, refusal reason.
- **The recommendation's text**: its metrics, the runner-up and the gap, its confidence
  (grading state, method shares, extrapolated share of steps), its laws' sources, every
  point left out with its reason; each number in it is in the table.

**Reason.** InferenceX's definitions, so a user reads the curves as InferenceX's, from the
code that grades the M3-Agentic stage, so a search and a grade share them; the search's
own add the engine's work net of prefix-cache hits and a request's whole wait; a refused
step charges less than it ran by an amount nothing measured (`08` D50.1). **Basis.** The
owner on #3355; #3264, landed by #3325; `08` D45.1, D50.1.

---

## D112. Calibrated data is reused only when every requirement holds

**Rule.** Laws, memory readings and platform readings, in a store or under
`scripts/compass/laws/`, serve a candidate only when every requirement below holds. The
reuse check gives each candidate one verdict, *reuse*, *calibrate* or *refused*, naming
the requirement that failed; `compass advice` calls it and judges nothing. A reused input
is stated in the run's provenance.

1. **Same key**, as `07` D41 keys each kind, at the integration head's tree. No entry at
   the candidate's key: *calibrate*. Another tree, an older checkout, a release or a local
   patch, is *refused* by name, the only requirement that refuses: only the head is
   supported, other trees waiting for after v1 (#3122), and the showcase's frozen commit
   (D117) stands for the head.
2. **Not invalidated**: no ATOM change since the calibration moves step costs or memory,
   and only then is a law re-keyed onto the new tree (README principle 10, #2594), its
   readings with it under #3122's design; a change that moves memory takes the readings
   again, as #2861's token buckets moved DeepSeek-V4.1's graph pool. A search may mix laws
   fitted at different commits and so re-keyed, each point's provenance naming the commit
   its law was fitted at and saying re-keyed, not re-validated; the showcase fits all its
   laws at one frozen commit (D117) and measures again after a later source change,
   until #2311 settles who judges "moves step costs" and how. Failing: *calibrate*.
3. **Enough coverage**: the law's calibrated range holds the batch shapes the search
   needs, a vector for every decode rung of the capture ladder included, judged by
   `compass run --coverage`, which gains a step linear check beside the step grid's and
   the op graph's. Failing: *calibrate*, the missing shapes and rungs listed.
4. **Accurate enough**: the law's held-out error is within the acceptance targets (T114).
   Failing: *calibrate*.

**Reason.** Each requirement guards one way a reused law stops matching the subject it was
fitted on, and one check outside the search gives every caller one answer. **Basis.** The
owner on #3355 and #3363; `07` D41, D43; README principle 10; #2594; #2899's head-only
rule; `13` D80, under which a backend fitted under another engine configuration refuses to
run.

---

## D113. Calibrating what the search lacks

**Rule.**

- **The record list.** For a configuration missing a law or lacking coverage, the step
  linear backend derives the records from the configuration alone: from its capture
  ladder, the concurrencies and the decode rungs, each recorded from every trajectory's
  first turn (#1840), and the trace runs. `compass advice` prints them as one
  `calibrate.sh` command with every argument and its expected time, unknown for a
  configuration never calibrated. Planning sits beside the backend (`07` D37).
- **`calibrate.sh`** publishes into the shared store `STORE` names, or, with none named,
  its own `store/` as today; each step appends its start, end or failure to `status` in
  its output directory (`cal/<config>/` under the search's OUT), which `compass advice`
  reads; a failed calibration resumes where it stopped. The guide runs it under `setsid`
  or `nohup`, its exit trap unchanged. A 1P1D calibration (`deploy.sh record`, a fit per
  side) waits until a 1P1D configuration needs a new law.
- **No top-up in place.** A store entry is immutable, a second publish at its key
  refused, and `calibrate.sh` takes no record once a fit has run: a configuration lacking
  coverage is calibrated again, old records with new, in a new output directory and into
  that calibration's own store. Every configuration that borrows its law, as S5 borrows
  S2's (D116) and MiniMax-M3's LMCache arms their base arms' (D117), has its readings
  measured into that store too, `compass advice` printing them with the calibration
  command. The reuse check looks a candidate up in the latest store holding all its
  inputs, a new calibration's before the shared one, and the candidate's runs read that
  store; `compass run` keeps its one `--store`.
- **Timing.** Each calibration records each step's wall time (the warm record, `measure
  memory`, each record, the trace runs, `fit steps`, `fit host`) and its node-hours.

**Reason.** The backend whose law prices the steps knows which it lacks; an immutable
entry cannot change under a run that read it, so a gap is filled by a new law beside the
old one, and a borrower's readings go with it, or its one store would keep it on the old
law; the timing sets fitting-efficiency targets (#2678). **Basis.** The owner on #3355 and
#3363; `07` D37, D41; #2317; #1840; `atom/compass/artifacts/store.py`, whose entries
refuse a second publish.

---

## D114. Profiling: the per-request split and the op-family split

**Rule.**

- **The per-request split**: each request's queue, prefill, KV transfer and decode time,
  both 1P1D sides mapped to one request, and its LMCache store and load, read from the
  step record (`dp<r>.jsonl`), a P/D run's `transfers.jsonl` and aiperf's per-request
  export, which simulated and real runs both have, so one code path serves both. Every
  run writes it when it ends, into its own file in its run directory, never
  `summary.json` (`01` D3.5, rule 4), changing none of the run's results: `compass run`
  and `compass record` for an instance, `deploy.sh` for a deployment, both sides joined.
  A request's segments sum to its end-to-end latency within the clock's resolution; on a
  real run its TTFT matches aiperf's within a tolerance measured first. Its format is
  T109's.
- **The op-family split**, op level only: the prices per op kind a `compass run
  --op-graph` run holds per step (`op.<kind>` in `dp<r>.cost.jsonl`), summed per step
  phase into MoE, attention, dense GEMM, collectives, norms and elementwise, and other,
  with no recomputation or estimate. Each phase's families reconcile with its predicted
  time, a remainder (the DP group's slowest rank) listed with its cause; a refused op is
  reported as coverage; a TBO step shows its one `op.tbo` term with the reason. A
  step-level run says it has no op split.

**Reason.** The split says where a request's time went, the same way for a simulated and
a real run, outside a summary that copies no latency; the op split adds no estimate.
**Basis.** The owner on #3355; `01` D3.5; `08` D50.1. #2763's one-off script split TTFT
only and is not in the tree.

---

## D115. The dashboard and the per-run report

**Rule.** Two static HTML views, served by no server, reading files and never calling the
search's or profiling's code.

- **The dashboard**, the search's `dashboard.html`: the frontier curves, then the
  recommendation and its trust, then each point opening its run's report. The main chart
  draws throughput per GPU against interactivity, both definitions of each switchable,
  with TTFT, total throughput, GPU count, versions and law sources. Each point carries two
  marks, its grading state (graded, ungraded, coverage report) and its pricing-method
  shares under `08` D50.1's names; a point with a refusal is grey with its reason. An
  optional reference layer, labelled with its source: InferenceX's DeepSeek-V4-Pro curves,
  frozen since it retired the model on 2026-10-29, as another ATOM version with
  unverified configurations; on MiniMax-M3, the M3-Agentic stage's per-point verdicts on
  mi355-gpu-40, as another platform; never InferenceX's MiniMax-M3 points. ATOM's
  Benchmark Dashboard (`.github/dashboard/index.html`) is tried first (T111).
- **The per-run report**, `report.html`, which every run, simulated or real, writes into
  its directory when it ends: the run (configuration, machine, ATOM tree, law sources from
  `run.json`, workload); headline metrics by #3264's code, as the dashboard computes them,
  and the speed ratio; trust, from `summary.json` (method shares of seconds, refusals,
  extrapolated and fallback steps as shares of steps, whether it is a coverage report, a
  graded cell's grades); the per-request split, beside the real run's when the cell has
  one; the op-family split on op-level runs. Each number comes from a file in the run
  directory, a part with no data says so, and the page changes none of the run's results.
  ATOM's own report (`.github/scripts/atomesh/observability/export_report.py`) is not
  used: it plots Prometheus series, which need `/metrics` sampled on the simulated clock,
  deferred.

**Reason.** A static page travels with its run; reading files keeps the views apart from
the code they show (D108); two marks keep an ungraded or guessed number from reading as
measured. **Basis.** The owner on #3355; `08` D50.1; `11`, which owns `/metrics`.

---

## D116. The development template: DeepSeek-V4-Pro on MI355X, ungraded

**Rule.** The search, profiling and views are developed on DeepSeek-V4-Pro (the original
checkpoint) on MI355X under the agentic cc-traces workload, on the development nodes, from
their store. The template is a vehicle, not evidence: none of its configurations is graded
or entered in `08` D50, whose M2 set is unchanged, and every point reads ungraded. #975's
graded cell differs from P3 only in prefix caching, off there, which matters on an agentic
workload, so its grade is not cited.

| # | Deployment | Parallelism | MTP | TBO | LMCache | ATOM basis | Law |
|---|---|---|---|---|---|---|---|
| S1 | one node, 8 GPUs | TP8 | MTP3 | off | off | `recipes/DeepSeek-V4-Agentic-InferenceX.md`, TP | calibrated anew (D113) |
| S2 | one node | DPA8 | MTP3 | on | off | the same, DP attention | calibrated anew (D113) |
| S5 | one node | DPA8 | MTP3 | on | on | the same, DP attention with LMCache | S2's step and host laws; its own readings, whose key holds the KV offload |
| P1 | 1P1D, 16 GPUs | TP8 / TP8 | off | off | off | `recipes/mesh/DeepSeek-V4.md`, pure TP; nightly `ds-v4-1p1d-tp8` | M2's TP law in the crsuse2-m2m-v3-005 store, if it passes D112 |
| P3 | 1P1D | DPA8 / DPA8 | off | prefill | off | `recipes/mesh/DeepSeek-V4.md`, DPA | the shipped crs005 and gpu45 pairs under `scripts/compass/laws/`, if they pass D112 |

The configurations cut from it, each one switch alone (S3, S4, P2, P4 to P7), can come
back. Arguments and environment follow the basis recipe, except:

- `--max-num-seqs 512` everywhere, not the single-node recipe's twice the concurrency:
  one value keeps one memory-readings entry per configuration.
- Prefix caching on everywhere, as the single-node recipe and the agentic workload have
  it; the 1P1D recipe turns it off. It is in no law's key, so whether M2's laws, fitted
  with it off, serve is D112's accuracy requirement.
- `--gpu-memory-utilization` 0.9 on one node; 0.75 on each decode side, the development
  nodes' Mooncake dmabuf limit; on the prefill side 0.72 for P3, whose eight DPA ranks'
  gathered MoE rows pass 65,536 at 0.75 and run aiter's MoE workspace out of memory
  ([#975 Ruling](https://github.com/jgong5/ATOM/issues/975#issuecomment-6565403459)), and
  0.75 for P1 unless its calibration runs out of memory. No law keys on it.
- MTP3's acceptance length stated as 2.49, InferenceX's golden
  `deepseek-v4-pro__mtp__native__default-default.yaml` (thinking on) and the single-node
  recipe's, by `--spec-decode-acceptance-length` on real and simulated runs alike
  ([#2683 Ruling](https://github.com/jgong5/ATOM/pull/2683#issuecomment-6578751421)).
- No `--block-size`: ATOM forces 256 on any DeepSeek-V4 checkpoint (`atom/config.py`).
- Capture sizes: ATOM's default on one node; on 1P1D the shipped DPA laws' ladders, so
  those laws can pass the key check.

Each configuration with a law runs at 1, 2, 4, 8, 16, 32, 64, 128 and 256 clients, no
higher: the held-out traces are 0 to 255, and reusing sessions is T27's. S5 offloads
through `lmcache_mp`, its two stations declared (`01` D6.1) at the run-file values of
`scripts/compass/parity_one_node.sh`, so its points show a declared share. A fractional
stated length has three known gaps, ungraded here: a law's key holds only the method and
the draft count; `14` D87's gates on the mean length and the accepted-count histogram are
not built; real DP ranks draw the same uniforms each step where simulated ones draw apart
(read in the code, not measured).

**Reason.** DeepSeek-V4-Pro has a store and shipped laws on the development nodes, so the
tool is built while MiniMax-M3's support lands (D117); the template runs every path of the
tool, a reused law and a new one included, and S1 against S2 puts the agentic workload's
best-known trade-off on its frontier. **Basis.** The owner on #3355; `16` D95; the #975
and #2683 Rulings.

---

## D117. MiniMax-M3: the move, its template and the showcase

**Rule.**

- **The move**, once the Compass features MiniMax-M3 needs have landed and the M3-Agentic
  stage's small case (#3265) passes, whatever the DeepSeek-V4-Pro work has reached; #3256
  (`16` D95.1) tracks that support.
- **Our platform.** Laws, readings, offload terms and real runs are taken on the
  development nodes at one frozen commit: every candidate's law is fitted there, a
  recalibration included, and the showcase's simulations and validation runs run there,
  since a law's version is the commit it was fitted at, not only the code a simulation
  runs. The M3-Agentic stage runs on mi355-gpu-40 at a frozen commit of its own, its spec
  pinning another aiter (`09d256420` against crs005's `41d4d8d7e`) and so another
  platform key, on another host CPU: its code and method are reused, not its laws,
  readings or real runs.
- **The environment check**, first: each of #3256's arms served once with the
  submission's arguments, as #3258 did on gpu-40: whether it comes up, fits memory at
  `--gpu-memory-utilization 0.95`, has the host memory its LMCache tier pins, places
  across NUMA nodes and reads its checkpoints; an arm that does not leaves the template.
- **The template**: #3256's arms with the submission's arguments as #3256 records them,
  `--max-num-seqs` twice the concurrency, EAGLE3 at `14` D87.1's forced length and the
  `agentx` workload (`06` D35.1), each arm at every concurrency InferenceX ran; a point
  that cannot be priced is a grey coverage report. How one law serves points whose
  `--max-num-seqs` differs is #3275's ruling (T113).
- **Calibration** as `16` D95.1's, after the environment check: a base law each for TP4,
  TP4 with indexer CP and TP2, the last two pooling their LMCache arms' records, with `03`
  D13.1's offload terms, only if those arms come up; a readings entry per point; each step
  timed (D113).
- **The showcase run**: `compass advice` on that template, the model differing only in
  the space file, no search, profiling or view code branching on it; whatever MiniMax-M3
  forces to change is filed as a defect; an op-family split only where the op-level
  backend prices the cell (#3126 records MiniMax-M3's sparse layers as missing at op
  level). Its CPU time, the machine stated, is set against an InferenceX-style sweep's GPU
  time, 3600 s a point plus the measured start cost.
- **The showcase validation**, an `08` D50 set registered and frozen before it runs: at
  the recommended point's concurrency, the recommended configuration and the runner-up
  (D111) each run real three times on the development nodes, spaced (`08` D45), 3600 s
  each as the `agentx` workload runs. Each cell is graded by `08` D44, and their real
  order by throughput per GPU, InferenceX's, is compared with the simulated one: it
  matches, or the gap lies within the real runs' spread and is reported undetermined. `08`
  D45.1 is not the criterion: it judges hit rates, which a simulated run lacks (its *Open
  issues*). The M3-Agentic stage's verdict on the same arm at the same concurrency stands
  beside, labelled as another platform. The simulation is not compared with InferenceX's
  published results. How calibration and grading divide the traces is T112.

**Reason.** MiniMax-M3 is ATOM's AgentX submission to InferenceX, for which the M3-Agentic
stage builds what Compass needs, so the search is shown on a space users deploy; a law is
keyed to its platform and host; the recommendation against its runner-up is the choice a
user acts on, and their order is what real runs confirm. **Basis.** The owner on #3355 and
#3363; `16` D95.1; `08` D44, D45, D45.1; `14` D87.1; `03` D13.1; `06` D35.1.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D108 | Fitting apart from searching: a candidate is simulated once its law exists; the halves meet only through the store and a calibration's `status` file, the search owning no calibration, fitting, reuse judgement or version management. A user guide, `scripts/compass/ADVICE.md`, exact enough that an agent following it fills the DeepSeek-V4-Pro template's space file, and no skill; an agent starts a calibration only on the user's yes. | 2026-11-13 ([#3355](https://github.com/jgong5/ATOM/issues/3355), [#3363](https://github.com/jgong5/ATOM/pull/3363#issuecomment-6845852318)) |
| D109 | `compass advice SPACE OUT` checks each candidate (searchable, missing a law, calibrating, refused), simulates each searchable one at each concurrency on a fresh server with no step roofline, and writes the results; it never calibrates, and runs at the integration head, on the target machine type, in a GPU container. Amends `13` D80: it starts the client and takes a workload, and `run --check` resolves a prefill or decode side. | 2026-11-13 (#3355) |
| D110 | The user provides the space file, by hand or through an agent following the guide, and it names the shared store; it expands deterministically into candidates, each naming its ATOM basis and departures; it is the search's input, not a level of `13` D79's precedence. | 2026-11-13 (#3355, #3363) |
| D111 | InferenceX's metrics by #3264's code, and the search's computed-token throughput and P90 end-to-end latency from the same export; the frontier of throughput per GPU against P90 interactivity; the recommendation, the most throughput per GPU meeting the SLO, which bounds P90 interactivity, with no refusal; the runner-up, another configuration's best at its concurrency; the result table and the recommendation's text. | 2026-11-13 (#3355) |
| D112 | Calibrated data serves a candidate only with the same key, not invalidated, enough coverage and held-out error within the acceptance targets; the verdict is reuse, calibrate, or refused for a tree other than the integration head. A search may mix laws fitted at different commits and re-keyed to the head, each point's provenance naming its law's commit; the showcase's are all fitted at its one frozen commit. | 2026-11-13 (#3355, #3363) |
| D113 | The step linear backend lists a configuration's missing records as one `calibrate.sh` command, which publishes into `STORE` and reports each step in `status`; with no top-up in place, a configuration lacking coverage is calibrated again into a new store, the readings of every configuration borrowing its law with it, and each candidate runs on the one store holding its inputs; each calibration is timed per step. | 2026-11-13 (#3355, #3363) |
| D114 | Every run writes its per-request split when it ends, apart from `summary.json`; an op-level run's op-family split sums pricing's prices per step phase. Amends `13` D80's outputs of `run`, `record` and `deploy.sh`. | 2026-11-13 (#3355) |
| D115 | Two static views reading files only: the search's dashboard, each point marked by grading state and method shares, and every run's `report.html`. | 2026-11-13 (#3355) |
| D116 | Developed on DeepSeek-V4-Pro on MI355X on the development nodes, ungraded and outside `08` D50: S1, S2, S5, P1 and P3 at 1 to 256 clients, each departure from its recipe stated. | 2026-11-13 (#3355) |
| D117 | Moved to MiniMax-M3 once its Compass features land and #3265 passes, calibrated on the development nodes from #3256's arms at one frozen commit, where its simulations and validation runs run too; the showcase validation runs the recommended configuration and the runner-up real, graded by `08` D44, their real order compared with the simulated one. Amends `08` D50. | 2026-11-13 (#3355, #3363) |

---

## Open items

T108 to T114, registered in [`12_open_items.md`](12_open_items.md) under *Topic 19*.
