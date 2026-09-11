# ATOMCompass — PoC gate status

**Maintained document. One row per completion gate, and nothing in it may be
loosened.** `POC_SUMMARY.md` is the narrative handover, `DESIGN_NOTES.md` the
working log, `RETROSPECTIVE.md` a dated audit. This file is the score.

Scope, verbatim from `presentation/POC_SCOPE.md`: **Qwen3.8-27B** (hybrid — 48
gated-DeltaNet + 16 full-attention layers), MI308X-class single node, **TP ∈
{1, 2, 4}**, one short-input and one long-input serving workload. Qwen3-0.6B is
the development model, not a gate subject. Out of scope and not counted against
any gate: EP/MoE, PP/DP/PD disaggregation, speculative decoding, an analytical
SOL oracle.

Production contracts that must survive: `--level 3` compilation, CUDA-graph
capture and replay, chunked prefill. **Cache policy for every gate run:
`--no-enable_prefix_caching`.** Prefix caching changes which tokens are
computed, so a real run with it on and a simulated run with it off are not the
same experiment; it is off on both sides, always, and any gate run that turns it
on is a different row.

## Final acceptance is end-to-end cc-traces — registered 2026-09-11

**Every gate in this file is finally proved on an end-to-end replay of the
cc-traces corpus, and on nothing else.** Registered 2026-09-11 at the user's
direction. What that changes:

* Throughput, TPOT/ITL and TTFT accuracy (G2a–c), feasible-configuration
  selection, ranking, ties and regret (G1, G1b, G1c), prediction outside the
  calibration configurations (G4) and the integrated GPU-free ≥ 5× timing with
  acquisition and amortisation (G5a, G5c) are end-to-end cc-traces quantities.
  Independent per-term memory and KV validation (G3a, G3b) and the deliberate
  infeasible rejection (G3c) must correspond to **those same deployment
  configurations**.
* The synthetic 64 × 1024/128 workload, the standalone primitive prices and the
  covered single-step checks (E7 at TP2, E8 at TP4) are **diagnostics**. They
  are retained in full as dated history and none of them may be cited as gate
  evidence. The rows below that still rest on them are marked accordingly.
* `agent_scratch/cc_pilot.jsonl` — the first 20 requests of the corpus — is a
  **development and regression workload**. The cost model was iterated against
  it; §3 already says so about E1. It is never held out on workload.
* The held-out axis the PoC is asked for is **configuration**. Every held-out
  axis is named per prediction, and evaluated TP2/TP4 full-engine measurements
  stay out of calibration.
* Before final evaluation the dataset version, workload scope, short/long regime
  definitions, selection rules, request identities and order, token lengths and
  actual usage, arrival pacing, preparation protocol and hashes are frozen.
  Cases are not selected or shrunk after errors are seen, and real arrival
  semantics are respected — a heterogeneous paced trace is not silently turned
  into a homogeneous burst.

`PROTOCOL.md` §8 still declares the legacy diagnostic cells (the synthetic short
workload and the first-20 `cc_pilot` long workload); its stamped results and lock
history are preserved as they stand and are not relabelled. The end-to-end
cc-traces acceptance registration that supersedes it is being written
separately. **Until that registration exists and is stamped, no row in §1 may be
moved on cc-traces evidence.**

Three properties of the corpus do not survive the current replay path, and they
bound what any cc-traces cell can claim (`agent_scratch/cctraces.py`): prefix
reuse is dropped — the trace's `hash_ids` carry 64-token block sharing, while
`replay.py` sends synthetic prompts that share no prefix and the engine runs with
prefix caching off; `in` is a **block count**, not a tokenizer count, accurate in
distribution and approximate per request; and recorded arrivals are
**open-loop**, so a faster engine sees the same arrivals a slower one did. The
source is `semianalysisai/cc-traces-weka-062126-256k`, cached at
`/md1/users/jgong5/hf_cache/cc-traces-256k/traces.jsonl` (568 864 747 B; full
corpus 1 847 151 435 B), 393 sessions / 98 827 requests.

---

## 0. Metric definitions — fixed here, before evaluation

All timings come from the **engine's own clock** (`GET /compass/requests`), wall
under a real run and virtual under a simulated one. A client stopwatch measures
how fast the simulator ran and is never a gate number
(`POC_SUMMARY.md` §4.3).

| symbol | definition |
| --- | --- |
| `arrival_i` | when the engine recorded request *i* as arrived |
| `first_i` | when its first output token was published |
| `finish_i` | when its last output token was published |
| `n_i` | output tokens **produced** for *i*, read from the server's `usage.completion_tokens` and never from the length the workload asked for. `compare.py` refuses a run whose responses do not carry it, and refuses a pair whose two sides produced different counts — those two runs have no comparable throughput or TPOT |
| **TTFT_i** | `first_i − arrival_i` |
| **TPOT_i** | `(finish_i − first_i) / (n_i − 1)`, over requests with `n_i ≥ 2` |
| **latency_i** | `finish_i − arrival_i` |
| **throughput** | `Σ n_i / (max finish − min arrival)`, output tokens per second, on the engine clock |
| **error** | signed, `(modelled − real) / real`, computed **separately on the median, the mean and the p90** of the per-request distribution |
| **median, p90** | plain order statistics of the sorted sample, `s[min(n−1, ⌊q·n⌋)]`, no interpolation. For n = 10 the median is the 6th value and p90 the 10th, so both are *upper* picks — not `numpy.percentile` defaults, and not `statistics.median`, which averages the middle two on an even sample. Only `mean` is an average. Every summary `compare.py` writes stamps this convention in `quantile_convention` |

ITL is not separately recorded: the engine stamps first and last token, not
every token, so **TPOT stands in for ITL and the gate is read against TPOT
alone.** Reporting a per-token interval distribution would need a per-token
timestamp the engine does not currently emit; that is a named gap, not a pass.

An aggregate that agrees is not a result. Every gate row must be accompanied by
the per-request distribution and by the components (prefill work and cost,
decode work and cost, schedule structure) so that two errors cancelling cannot
read as success — this has happened four times (`POC_SUMMARY.md` §7.2).

**Held-out vocabulary.** A prediction is *held out with respect to an axis* only
if no measurement of the evaluated point on that axis entered the model. A
reusable deployment constant measured once elsewhere may be a calibrated input
and must be named as one. **A per-target measurement does not make the target
held out**, however small the measurement.

---

## 1. Gate matrix

Status values: **PASS** (evidence exists and is preserved), **FAIL** (measured
and outside the bar), **PARTIAL** (some cells pass, the gate as stated does
not), **UNPROVEN** (no valid measurement yet).

| # | Gate | Bar | Status | Evidence |
| --- | --- | --- | --- | --- |
| **G1** | Correct feasible-configuration selection | top-1 matches hardware | **UNPROVEN** | no TP × workload matrix has been run |
| **G1b** | Ranking correlation within comparable groups | Spearman ρ ≥ 0.90 | **UNPROVEN** | — |
| **G1c** | Ties reported honestly, selection regret reported | stated, not assumed | **UNPROVEN** | — |
| **G2a** | Throughput error | ≤ 10% | **FAIL** | E2a, 27B short-input: **+27.3% / +37.0% / +85.9%** at TP=1/2/4. Measured, not missing — the modelled run finishes the same 64 requests in 27.5/16.3/8.5 s against 35.1/22.3/15.9 s real. The error grows with TP, which is the signature of a fixed per-step cost the model does not carry |
| **G2b** | TPOT / ITL error | ≤ 10% | **FAIL on short, PARTIAL on long** | E2a short-input median **−49.6% / −45.9% / −39.7%** at TP=1/2/4, and the modelled TPOT is near-constant across requests (p90 ≈ median) where the real one is not. E1 long-input passes at TP=4 |
| **G2c** | TTFT error | ≤ 15% | **FAIL on short, PARTIAL on long** | E2a short-input median **−10.4% / −20.5% / −46.1%** at TP=1/2/4 — within bar at TP=1 only. E1 long-input at TP=4: mean −0.14%, p90 −0.27%, median −6.4%, on the workload the model was developed against |
| **G3a** | Non-KV memory terms | each within 10% | **PARTIAL** | E3b: 27B at TP=1/2/4. weights +1.6/−0.1/−3.3%, non-torch +4.9/+1.2/+1.3%, graph pool +0.0% everywhere, load residue −3.3/−6.7% at TP≥2. Three terms outside: load residue at TP=1 (−93%, 0.01% of budget), persistent (−51%, 0.07% of budget), activations (no prefill-shaped graph for this model) |
| **G3b** | KV block count | within 5% | **PARTIAL** | E3: −0.09% (27B TP=2), −0.02/−0.02/+0.07% (0.6B TP=1/2/4). Not yet predicted from a profile at 27B TP=1/4 — the measured ground truth now exists (112 740 / 265 520 / 584 880 blocks) |
| **G3c** | One infeasible configuration rejected for the right reason | same error as the engine | **UNPROVEN** | — |
| **G4** | Prediction outside the calibration configurations | stated per prediction | **UNPROVEN on the acceptance source; the two step-level diagnostics split, one within and one outside** | No end-to-end cc-traces transfer has been run, and the gate is an end-to-end quantity. Diagnostics, both `ModelRunner.forward` decode steps frozen before measurement with no target timing among their inputs: **E7, TP2, 20.970 ms frozen against 19.845 ms, +5.7%** — within 10% (`G4_TRANSFER.md` §12). **E8, TP4, 16.157 ms frozen against 12.807 ms, +26.2% — outside 10%, a failed prediction** (§13). E8's miss is localised: rank 1's body price was measured 17.1% high by one contaminated pricing process, reproduced as such by a repeat run into a separate directory, and the other three ranks predicted +5.1 to +5.5%. That diagnosis explains the number and does not license a corrected one — the frozen prediction stands as it was frozen. Both share the shape `bucket=32, cohort=32, tokens_each=1, context=1151`. E6 is their coverage precondition — a priced body at TP=1 (23.122 ms, 2423/2439 operators) and TP=2 (15.467 ms, 2552/2568) |
| **G5a** | Replay speedup | ≥ 5× | **PARTIAL** | Observed, device-free, at 27B: 36 s of serving → 1.46 s (24.7×), or 133 s → 23 s (5.8×) including server startup. That is the replay speed and it is measured. It is not yet a gate pass because G5c — capture and calibration cost reported separately and amortised — is a distinct gate and is unmeasured; an earlier 309 s → 3 s figure is superseded, its simulated half still held the model on a GPU |
| **G5b** | GPU-free replay after capture | no device | **PASS at 0.6B and 27B** | E5: served 32/32 (0.6B) and 64/64 (27B) in `xiaobizh_n18_cpu`, a container with **no `/dev/kfd` and no `/dev/dri`** — zero driver handles and no KFD process registration in any process of the tree. Both reproduce the GPU-resident simulator's schedule step for step and its TTFT/TPOT/latency distributions exactly; at 27B ten of 64 requests sit in a different slot of that same schedule, which is the burst's admission order, not the GPU-free path (E5). This is the no-device gate only; G5a and G5c are separate and still open |
| **G5c** | Capture / calibration / startup / load / execution costs reported separately, with amortisation | reported | **UNPROVEN** | — |

**One gate is a pass; the rest are not.** G5b is PASS at both models — GPU-free
replay in a container with no `/dev/kfd` and no `/dev/dri` — and the earlier
blanket "nothing here is a pass yet" no longer described the matrix. Corrected
2026-09-11. Everything else stands: the two PARTIAL accuracy rows are one
workload at one width, G5a and G5c are separate from G5b and open, and the
honest reading of the whole matrix is still that the pilot is closed and the
gates are open.

**G4 was moved to PARTIAL on 2026-09-11 morning and back to UNPROVEN the same
afternoon.** Two things moved it back, and both matter. The first is the
acceptance registration above: G4 is an end-to-end quantity, and a step-level
check — passing or failing — cannot carry it. The second is E8. The TP4 step,
frozen at 07:22:04Z and measured an hour later, came in at **+26.2%, outside the
criterion**. The morning's PARTIAL rested on E7 alone; by evening the same method
at the next width had produced a miss five times larger, and a row that reads
PARTIAL on the strength of one of two diagnostics while the other fails is a row
that flatters itself.

**E8 is preserved as a failure, not as a pending item.** Its cause is understood
— one of twenty-three frozen inputs was measured wrong — and understanding the
cause does not retroactively make the prediction right. No corrected TP4 number
may be computed against that capture by anyone, because the capture has now been
seen. A second TP4 transfer claim needs a fresh freeze over re-measured inputs
and a fresh capture.

What E7 and E8 do establish jointly, at two points: no target-width step or
serving time entered either prediction, and the composition reproduces itself
byte-for-byte across widths (`tpN_frozen.py` at TP=2 regenerates
`tp2_frozen.txt` exactly, which is the precondition that licensed the TP4 run).
What they do not establish is any accuracy gate in this matrix (G2a/G2b/G2c) or
any ranking gate (G1/G1b/G1c): those are measured over a whole serving run, and
none of them moves on either experiment.

**Method gap E8 exposed, now open as item 10 in §7.** Nothing in the freeze path
checked a price against a second measurement of itself. `PriceLibrary` already
has the mechanism — a 5% conflict band, `library.py:373` — and it would have
caught rank 1 had the body been priced twice. The repeat run cost about five
minutes at TP4 against the GPU-hours a wasted capture costs. **No future freeze
should accept a price measured once.**

---

## 2. Experiment register

Each experiment is defined once, here, so that a gate row names a procedure
rather than a run.

### E1 — long-input serving, 27B, TP=4 (the cc-traces pilot; closed)

* **Workload** `agent_scratch/cc_pilot.jsonl`, the first 20 requests of
  `semianalysisai/cc-traces-weka-062126-256k`. Statistics of **those twenty**:
  input tokens min 640, median 84,800, max 119,360, sum 1,681,024; output
  tokens min 20, median 525, max 3,693; declared arrivals spanning 244.5 s,
  client-paced (`replay.py --pace --check-lengths`). An earlier revision quoted
  163,584 as the median: that is the median of the *whole trace*, not of the
  slice actually sent, and the number here is recomputed from the file.
* **Engine** `Qwen/Qwen3.8-27B`, `-tp 4`, `--gpu-memory-utilization 0.90`,
  `--max-model-len 262144`, `--no-enable_prefix_caching`, `--max-num-seqs 32`.
* **Calibration** `CalibratedCostOracle` fitted on `big_sweep.tp0.jsonl`, a
  measured sweep of the *same* engine configuration, including long-context and
  ragged rounds. **Held out on nothing.** An earlier revision called this "held
  out on the workload axis": it is not. `cc_pilot.jsonl` is the workload the
  cost model was developed against and iterated on, repeatedly, and a result on
  it is a fit statistic rather than a prediction. The configuration axis was
  never held out either — the sweep is the same engine configuration. G4 needs
  a prediction on something this run has not seen.
* **Result** mean TTFT −0.14%, p90 −0.27%, median −6.4%, latency median −0.51%,
  prefill total −0.29% over 106 steps, decode −0.09%, schedule structure
  identical (5 streaks, longest 42 steps).
* **Repeats** four real runs, one simulated. Real TTFT median range is 0.287% of
  its median. **One simulated run is not a distribution**; the simulated side is
  unrepeated.

### E2 — the decision matrix (to run)

27B × TP ∈ {1,2,4} × {short-input, long-input}, real and modelled, ranked by
throughput and by TTFT under the same objective. Defined in
`agent_scratch/poc/` and reported into G1/G2. Feasibility of each cell is
established before it is ranked (§E4).

### E3 — memory, per term

`--compass-memory-out` records the four terms ATOM's budget is built from;
`scripts/compass/validate_memory.py` compares them one at a time. A summed check
is not evidence: one previously reported +13.8% sum hid three errors of which
two cancelled.

### E3b — the 27B budget at all three widths (run 2026-09-10)

`agent_scratch/poc/stage0_feasibility.sh`, sequentially at TP=1, 2 and 4 on
devices `0..N-1` of node 18, with `rocm-smi --showuse --showmemuse --showid`
before and after each width. Sequential deliberately: `non_torch` is
`(total − free) − reserved` and `total − free` is device-wide, so a
configuration running beside another is charged the other's bytes.

Measured budget (all at `total 191.98 GB`, `utilization 0.90`,
`budget 172.79 GB`):

| TP | free | peak_torch | non_torch | cudagraph | available_for_kv | block bytes | blocks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 139.43 | 53.99 | 1.11 | 0.55 | 113.30 | 1 056 768 | 112 740 |
| 2 | 156.69 | 29.88 | 6.92 | 0.32 | 131.83 | 528 384 | 265 520 |
| 4 | 168.66–168.85 | 16.92 | 7.12–7.31 | 0.22 | 144.49–144.68 | 264 192 | 584 880–585 642 |

Derived against recorded, per term, from `scripts/compass/validate_memory.py`
(which now calls the model for the three terms it used to print as "not
modelled", and shows each error as a share of the sizing budget as well as of
its own term):

| term | TP=1 | TP=2 | TP=4 | worst as share of budget |
| --- | --- | --- | --- | --- |
| weights | +1.6% | −0.1% | −3.3% | 0.26% |
| load residue | −93.0% | −3.3% | −6.7% | 0.08% |
| persistent | −51.0% | −51.0% | −51.0% | 0.07% |
| non-torch | +4.9% | +1.2% | +1.3% | 0.06% |
| graph pool | +0.0% | +0.0% | +0.0% | — |
| activations | — | — | — | — |

Three honest qualifications:

1. **`persistent` is wrong for this model.** `DEFAULT_PERSISTENT` is 118 MiB,
   from the 0.6B; the 27B holds 252 MiB at every width. It is a per-model
   constant and the calibration path (`validate_memory.py --calibrate`) exists
   to replace it. Not rewritten from one campaign.
2. **`load residue` at TP=1 is −93% of 14 MiB.** The relative bar and the term's
   weight point opposite ways, which is why the budget-share column was added.
   A gate stated only in relative terms ranks this alongside the weights.
3. **`activations` is still underived for the 27B.** The only traced 27B graph
   on the box is a 4-token decode step; scaling it to the 16 384-token warmup
   prefill reads +767%, which is a statement about the scaling, not about the
   term. A prefill-shaped trace (`--compass-trace-prefill`) is the missing
   capture.

`non_torch` agreeing to within 5% at all three widths is the load-bearing
result: it is the largest unmodelled-looking term, the model for it
(`DEFAULT_NON_TORCH`) was fitted on the **0.6B**, and it was applied here to a
27B nobody had sized at TP=1 or TP=4.

### E3c — deployment constants, held out across models (invalid, to re-run)

Intended to show `non_torch` and `load_residue` are properties of the runtime
rather than of the model, by measuring a 0.6B on the same box. Run on devices
4–7 while the matrix used 0–3; devices 4 and 5 were taken by another tenant at
100% use and 58% VRAM partway through, and the probe recorded `non_torch` of
112.9 GB and 118.8 GB — the neighbour, exactly as the device-wide definition
says it must. **The records are contaminated and prove nothing.** Kept because
the failure is the cleanest demonstration on file that a before/after pair does
not establish isolation: `stage0b/tp1.smi_before.json` already showed cards 4
and 5 busy, and nothing was watching.

### E4 — feasibility

A configuration is *feasible* when the engine starts, sizes a non-zero KV pool
and serves the workload's longest request. Rejection must carry ATOM's own
error (`InsufficientPoolBudget`), not a Compass-invented one.

**Result (2026-09-10):** all three widths are feasible for the 27B at
`max_model_len 262144`. TP=1 in particular now sizes, which an earlier attempt
did not — that attempt was contaminated by neighbours' bytes landing in
`non_torch`. Evidence `agent_scratch/poc/stage0/tp{1,2,4}.*`, run rc=0 at every
width, `### STAGE0 DONE 2026-09-10T10:57:03Z`.

---

### E5 — GPU-free serving replay, including in a device-free container (run 2026-09-10)

`agent_scratch/poc/g5_slice.sh` and `agent_scratch/poc/g5_replay_only.sh`,
artifacts under `agent_scratch/poc/g5_slice5/` (**Qwen3-0.6B**, 32 requests,
512 in / 64 out) and `agent_scratch/poc/g5_27b/` (**Qwen3.8-27B**, 64 requests,
1024 in / 128 out). This is evidence about the *mechanism*; the calibration is
deliberately in-sample (each oracle is fitted to its own capture run's steps),
so nothing here is an accuracy result.

The 0.6B walkthrough below is the smaller of the two. The device-free container
runs and the equivalence check cover both.

* **Capture.** One real deployment on card 1, `--compass-mode measure`, writing
  `capture_steps.jsonl` and `target.json`. The target now also records the
  machine: `arch gfx942:sramecc+:xnack-`, `AMD Instinct MI308X`, torch
  `2.10.0+rocm7.2.4`.
* **Replay.** `HIP_VISIBLE_DEVICES=""`, `--compass-mode predict`,
  `--compass-replay-target target.json`, started through
  `scripts/compass/replay_server.py`. Served 32/32 requests. **65 step rows on
  each side** — the GPU-free run took the same number of scheduler steps as the
  real one.

#### What "no GPU" means here, precisely

Open handles on `/dev/kfd` and `/dev/dri` are **not** the test, and the replay
holds nine of them per process. The HSA runtime opens every render node during
`torch.cuda.is_available()` to enumerate agents; with `HIP_VISIBLE_DEVICES`
empty it then reports none. Those handles are how the process was *told* there
is no GPU.

What using a GPU looks like is a KFD process registration —
`/sys/class/kfd/kfd/proc/<pid>`, created when a process acquires a context,
queues or device memory. Verified separately that enumeration alone does not
create it: `import torch; torch.cuda.is_available()` under
`HIP_VISIBLE_DEVICES=""` leaves it absent.

| pid | role | driver nodes open | KFD process registration |
| --- | --- | --- | --- |
| 558036 | API server | 9 | absent |
| 558566 | engine-core manager | 9 | absent |
| 558567 | engine core (runner in-process) | 9 | absent |

No process in the tree held a context, a queue or a byte of device memory.

#### What ran, and what was bypassed

Ran, unmodified and off ATOM: the OpenAI API server, `LLMEngine`, the engine
core, the **scheduler**, the block manager, KV admission, chunked prefill, the
sequence lifecycle and the output path. `ReplayModelRunner` answers the startup
RPCs (`get_num_blocks`, `allocate_kv_cache`, `capture_cudagraph`) from the
captured target instead of from a device.

Bypassed, and named: `AsyncIOProcManager` is replaced by `LocalProcManager`, so
the runner runs in the engine-core process rather than in worker processes over
zmq — the transport, not the policy. The forward itself is priced by the cost
oracle, which is what `--compass-mode predict` does on a GPU too.

#### The software dependency is separate from the hardware one

The replay still needs Torch, Triton and AITER **installed**. It no longer needs
them to find a device. AITER resolves an architecture name at import time, in
three places, and raises without one:

| query | how it asks | how the replay answers |
| --- | --- | --- |
| `ops/triton/utils/_triton/arch_info` | Triton driver, then `jax._src.lib.gpu_triton.get_arch_details` | AITER's own authored fallback branch, supplied with the captured arch |
| `jit/utils/chip_info.get_gfx_custom_op_core` | `GPU_ARCHS`, else shells out to `rocminfo` | `GPU_ARCHS` — AITER's own documented env seam |
| `jit/utils/chip_info.get_gfx_runtime` | always `rocminfo`, documented to ignore `GPU_ARCHS` | an import hook that loads AITER's own `chip_info` unmodified, then replaces `_detect_native` with the captured arch |

The third has no env seam by design and runs during `import aiter` itself
(`utility/dtypes.py` picks the fp8 dtype from it), so there is no post-import
moment at which to answer. All three live in
`atom/compass/replay/bootstrap.py`; `atom/compass/replay/_sitedir/sitecustomize.py`
carries the same answers into the processes ATOM spawns. Nothing installed is
patched on disk. Only a name is supplied — no kernel, no allocator, no
numerical result — and the whole bootstrap is a no-op whenever a real driver can
answer, so it cannot be used to dress a GPU run as a GPU-free one. Any device
query that is not one of these three still reaches the driver, and in a
device-free container still fails.

#### The decisive test: a container with no device nodes at all

Hiding a device with `HIP_VISIBLE_DEVICES=""` leaves the driver nodes present
and openable. So the same launcher was run in `xiaobizh_n18_cpu`, a separate
container created from the same image with **no `--device` flags** — `ls /dev/kfd
/dev/dri` returns "No such file or directory", and `torch.cuda.is_available()`
is `False` with `device_count() == 0`. The GPU container was not touched.

Both models served their whole workload there:

| | 0.6B (`g5_slice5`) | 27B (`g5_27b`) |
| --- | --- | --- |
| device nodes in container | 0 | 0 |
| requests served | 32/32, rc=0 | 64/64, rc=0 |
| scheduler steps | 65 | 260 |
| server up | 23 s | 22 s |
| serve wall | 0.52 s | 1.46 s |
| peak RSS (api / mgr / core) | 1.41 / 0.87 / 1.30 GB | 1.52 / 0.86 / 1.31 GB |
| CPU time (api / mgr / core) | 26.9 / 9.6 / 27.3 s | 26.8 / 9.6 / 27.8 s |
| driver nodes open, every pid | **0** | **0** |
| KFD process registration, every pid | absent | absent |

Peak RSS barely moves between a 0.6B and a 27B replay because no weights are
materialised: `ReplayModelRunner` answers the startup RPCs from the captured
target. The cost of a replay is a property of the *schedule*, not of the model.

#### Equivalence with the GPU-resident simulator

Same target, same oracle table, same workload, run both ways — in the GPU
container and in the device-free one. Per-request timings are compared on the
engine clock with only the run epoch removed (each run's own minimum arrival
subtracted); nothing else is normalised.

| | 0.6B | 27B |
| --- | --- | --- |
| step rows, both sides | 65 | 260 |
| steps whose scheduler decision differs in anything but `tick` | **0** | **0** |
| summed modelled step seconds | 0.329384 vs 0.329384 | 28.598431 vs 28.598431 |
| per-request `(index, ok, completion_tokens, finish_reason)` | identical | identical |
| per-request TTFT / TPOT / latency, epoch-normalised | **identical to 0.0 s on all 32** | **differs on 10 of 64** |
| TTFT / TPOT / latency *distributions* | identical | **identical multisets, to 9 dp** |
| throughput, window | identical | 286.449279976 tok/s, 28.598431110 s — bit-identical |

**The claim this supports is equivalence up to permutation of interchangeable
requests -- not per-request identity.** Two requests are interchangeable here
when they declare the same arrival instant and the same input and output
lengths; in this workload all 64 do, so they form one equivalence class and any
permutation within it is the same workload presented in a different order. The
label is available only because the workload is homogeneous. It is *not*
available for the heterogeneous long workload, where requests differ in length
and no permutation of them is the same workload: there the comparison keeps
per-request identity, and any ties are named as an explicit equivalence class
before the comparison rather than inferred from the result.

All 64 requests are stamped with a
*single* arrival instant — it is a burst, not a paced trace — and the order in
which the server accepts simultaneous arrivals is not fixed. Ten client requests
therefore occupy a different position in the same schedule between the two runs:
requests 21/22/23 and 37/38/39 exchange the first and second decode wave
(TTFT 10.006 s ↔ 24.272 s), and 12/32/33/34 exchange two adjacent slots within
the first wave (9.972 s ↔ 10.006 s). The set of positions, and everything about
them, is the same on both sides; only which client sits in which is not.

This is worth stating precisely because it is *not* evidence about the GPU-free
path. The same permutation is available to two GPU-resident runs of the same
burst. What the GPU-free path reproduces exactly is the schedule and the
distribution it produces; what neither side fixes is the admission order among
requests that arrive at the same instant. The 0.6B pair happens to agree
per-request as well, which is a smaller burst getting the same order twice, not
a stronger guarantee.

So the per-request row above should be read as: within the single equivalence class this
workload defines, the two runs are the same run; per-request identity is not
claimed and, on this workload, is not the property being tested.

`decision` records `kind`, `batched_tokens`, `running`, `waiting`,
`token_budget`, `waiting_held_for_arrival` and `waiting_prefill_outstanding`;
those agree step for step.

`tick`, the engine's process-local step counter, differs by a constant offset
(5525 at 0.6B, −21224 at 27B). **Recorded as a diagnostic, not explained**: the
offset is the number of engine steps each process took before it began serving,
and what those steps were has not been traced. Attributing them to graph capture
would be a guess, and a five-figure step count is large enough that the guess
should not be made without witnessing it.

#### What this does not yet show

* **The 0.6B slice does not close the 27B PoC gate**, and neither does the 27B
  run above on its own. G5b — *no GPU access during replay* — is now shown at
  both sizes. G5a (≥5× speedup) and G5c (costs reported separately plus
  amortisation) are not.
* **Observed replay speed, measured (this is G5a's numerator, on its own).**
  The 27B pair is above the clock's resolution: **36 s** of real serving against
  **1.46 s** device-free, a factor of **24.7**; including server startup,
  **133 s** against **23 s**, a factor of **5.8**. Both are measurements of this
  workload on this target, reported as such.
* **Amortisation is a separate gate and is not measured (G5c).** What a replay
  cost to *make* — the capture run and the calibration sweep — is not in the
  numbers above, and G5c asks for it to be reported separately and amortised
  over the candidate configurations a replay then saves. Nothing here quantifies
  that, and the speedup figures must not be read as if it did.
* **Accuracy is not the claim here.** In-sample, on the development model: TTFT
  median −49.1%, TPOT median −0.14%, throughput +33.0%. The 27B compare gives
  TTFT −6.25%, TPOT −49.97%, throughput +22.85%. Those errors are the same
  family as E2a — the first-use component and the virtual-clock question, both
  open — and the GPU-free path did not create or change them: it reproduces the
  GPU-resident simulator's decisions exactly.

---

### E6 — the stated batch, and a priced body at two widths (run 2026-09-11)

* **What changed** `graph_diff.py trace` installed no forward context, so all
  64 attention operators were recorded unpriceable. `--batch-spec` now installs
  one derived from a written-down batch
  (`atom/compass/runtime/batch_spec.py`, `tests/compass/batch_specs/decode4_c66.json`:
  four requests admitted at 64 prompt tokens, two tokens into generation).
* **Validation of the context** all 13 full-attention and 11 DeltaNet fields
  equal those recorded by the level-3 27B capture `compass_ops/silu27_graph.tp0.json`
  — capture at TP=4, derivation at TP=1 on meta. Nothing read from a meta tensor.
* **Result** TP=1 rank 0: 106/107 signatures, 2423/2439 operators (99.3%), body
  **23.122 ms**. TP=2 rank 0: 107/108 signatures, 2552/2568 (99.4%), body
  **15.467 ms** including `aiter::all_reduce_` at 1.18 ms. One family unpriced
  at both widths: `triton::_fused_qk_norm_single_kernel`, 16 operators, refused
  by the stride guard.
* **Not a step time.** The traced graph is the model body: the LM head, sampling
  and the runner's per-step work are outside it. The 16 refused operators are
  unpriced and unbounded, the derivation is level 0 against a level-3
  production, and the domain is one decode batch at context 66. Reported as
  predicted body cost with coverage; step-level acceptance is separate and open.
  Full record in `atom/compass/G4_TRANSFER.md` §10.
* **Artifacts** `agent_scratch/g4/s27decode4.tp{1,2,4}.r*.json` (graphs),
  `agent_scratch/g4/sprices_decode4_tp{1,2}*.json` (prices),
  `agent_scratch/g4/sdecode4_tp2.run` (run log, `rc=0`).

### E7 — a frozen TP2 forward step, then measured (run 2026-09-11)

* **Procedure** Predict one TP2 decode step from the G4 ledger's allowed inputs
  only; write the number and its full report to `dec32/tp2_frozen.txt`; hash the
  fifteen inputs into `dec32/tp2_frozen.sha256`, timestamped **06:21:54Z**; only
  then stand up a TP2 deployment and capture. The comparison re-hashes all
  fifteen before reading a number, and reads the frozen terms rather than
  recomputing them.
* **Span and shape** the whole of `ModelRunner.forward` — prepare_model +
  run_model + postprocess, which is what a capture row's `seconds` records —
  at `kind=decode, tp=2, bucket=32, cohort=32, tokens_each=1, context=1151`.
* **Result** frozen **20.970 ms** against a measured **19.845 ms**, **+5.7%**;
  per rank 20.970/+5.7% and 20.814/+4.9%. Four matching rows (ticks 130 and 260
  × two ranks), all retained. Rank alignment read from each row's `rank_coords`,
  not assumed from filenames.
* **Reproducibility** `checkpoint_tp2.py` re-executes the frozen script and
  regenerates the text byte-identically, hashing the **20 tree modules** the
  composition actually imported, plus torch/Python versions.
* **Scope, stated narrowly.** A `ModelRunner.forward` decode-step check against
  a 10% criterion. **Not** a passed serving TPOT gate; TPOT, throughput, TTFT
  and ranking are end-to-end quantities and still need E2's matrix. The named
  +0.223 ms copy-path sensitivity is reported separately: applied on its own it
  would worsen this comparison, and it identifies no component of the residual.
* **Artifacts** `agent_scratch/g4/archive/tp2_2026-09-11/` — 33 files under
  `MANIFEST.sha256`, tarball
  `40c1b3caf9e021314efe9e543c1150a2caba1c30bbc1c1028280d02b1eb54134`, verified
  equal on node 18 and the host. Full record in `G4_TRANSFER.md` §12.

### E8 — a frozen TP4 forward step, then measured — **it misses** (run 2026-09-11)

* **Procedure** E7's, at the next width, with one precondition added. The frozen
  script was generalised to either width (`tpN_frozen.py`) and `freeze_tp4.sh`
  **refuses to freeze** unless that script first reproduces `tp2_frozen.txt`
  byte for byte; it did, RC=0, recorded in `dec32/tpN_equivalence_at_tp2.txt`.
  Twenty-three inputs hashed into `dec32/tp4_frozen.sha256`, timestamped
  **07:22:04Z**, both outputs then `chmod a-w`. The TP4 deployment was stood up
  and captured afterwards.
* **Span and shape** as E7: the whole of `ModelRunner.forward`, at
  `kind=decode, tp=4, bucket=32, cohort=32, tokens_each=1, context=1151`. Eight
  matching rows — two scheduler ticks × four ranks — all retained.
* **Result** frozen **16.157 ms** against a measured **12.807 ms**, **+26.2%**.
  **Outside the 10% criterion. The prediction fails.** Per rank: r0 +5.5%,
  **r1 +26.2%**, r2 +5.1%, r3 +5.1%.
* **Where it went wrong, and where it did not.** The frozen prediction carried a
  21% rank asymmetry (rank 1 at 16.157 ms, the others at 13.45–13.51); that
  asymmetry was visible in the frozen text and was flagged before the capture
  ran. The hardware shows none — all four ranks measured 12.792–12.807 ms, a
  0.1% spread. 2.671 ms of rank 1's 2.692 ms excess is `aiter::gemm_a16w16`
  (6.398 → 9.070 ms, +42%). A repeat of the body pricing into a separate
  directory (`dec32/repeat_2026-09-11/`, rc=0, 07:33:52Z) reproduced rank 1 at
  **13.106 ms, 17.1% below the frozen input**, with the other three ranks within
  0.5%. One pricing process was disturbed; the transfer method was not refuted
  at this width, and neither was it confirmed.
* **What this does not license.** No corrected TP4 number. The capture has been
  seen, so any re-run of the composition against it would be fitting. E8 stands
  as a **failed prediction**; a second TP4 claim needs a fresh freeze over
  re-measured inputs and a fresh capture. The repeat is filed under
  `diagnostic/` in the archive, apart from the frozen inputs, for that reason.
* **Held-out region terms, reported.** postprocess −9.0 to −11.4%, prepare +
  remainder −22.5 to −23.7% — together about 0.06 ms of a 12.8 ms step. A real
  residual in the TP1-calibrated region model, and far too small to be the miss.
* **Method sensitivity: none at this width.** The two copy-path measurements of
  the body's all-reduce agreed to 0.83%, inside `PriceLibrary`'s 5% conflict
  band, so no conflict was recorded and the frozen text names no sensitivity.
  That is a result about the two methods at TP4, not an omission; the question of
  which path matches production remains open.
* **Reproducibility** `checkpoint_tpN.py` regenerates the frozen text
  byte-identically (`regenerated == frozen True`), 23 inputs unchanged, **21 tree
  modules** hashed, torch 2.10.0+rocm7.2.4.git3d3aa833.
* **Artifacts** `agent_scratch/g4/archive/tp4_2026-09-11/` — 53 files, 7.0 M,
  under `MANIFEST.sha256`, tarball
  `834732b042c372b1699f87b6454c2469d092e43edeb3ff2cd74601e0f2edb7a0`. Full
  record in `G4_TRANSFER.md` §13.
* **Status under the acceptance registration.** A diagnostic, like E7. It moves
  no gate row in either direction; it is recorded here because a failed
  prediction is evidence and deleting it would be the only way to lose it.

## 3. What is and is not held out (G4)

| prediction | measured on the evaluated point? | honest label |
| --- | --- | --- |
| E1 step cost, 27B TP=4 | **yes** — the sweep is the same engine configuration, and `cc_pilot.jsonl` is the workload the model was developed against | **held out on nothing**; a fit statistic, not a prediction |
| E1 admission constant | yes, per deployment | a calibrated per-deployment input |
| `warmup_seconds` | yes, first-use, per deployment | a calibrated first-use input — §4 |
| 27B TP=2 priced oracle (`POC_SUMMARY` §3.1) | overhead constants came from the 0.6B | held out on model for those two constants only; the price list was measured on the 27B at TP=2 |
| **E7 frozen TP2 decode step** (`G4_TRANSFER` §12) | **no** — frozen at 06:21:54Z with its input hashes, captured afterwards; no TP2 step or serving time among the fifteen inputs | **a genuine prediction, on one forward step.** +5.7% against a 10% criterion. Its region term (`SOURCE_27B_TP1`) is TP1-calibrated and applied unchanged, so that part is held out on width too |
| **E8 frozen TP4 decode step** (`G4_TRANSFER` §13) | **no** — frozen at 07:22:04Z with twenty-three input hashes, captured afterwards; no TP4 step or serving time among them | **a genuine prediction, on one forward step, and it missed.** +26.2% against a 10% criterion. Same held-out axes as E7 (width for the region term, target timings entirely). Diagnosed to one contaminated input, which does not un-fail it |

**Naming every held-out axis, because "held out" without an axis is a claim
about nothing.** E7 and E8 are held out on: the evaluated width's full-engine
step and serving timings (none entered), and — for the region term
`SOURCE_27B_TP1` — width itself. They are **not** held out on: the model, the
hardware, the operator set, or the standalone primitive prices, which were
measured at the evaluated width and are named as calibrated hardware-library
inputs. They are held out on workload only in the trivial sense that a single
decode step has no workload.

**The technical bet — capture at TP=1, derive TP=2/4 — has now been evaluated at
two points, both on one quantity, and it split.** E7 landed +5.7% at TP2; E8
landed +26.2% at TP4 and failed. Both are forward-step results. Neither is a
serving result: TPOT, throughput, TTFT and the TP ranking are end-to-end
quantities and, under the acceptance registration above, are proved on cc-traces
and on nothing else. Every prefill shape is still unmeasured, as is the
transfer's behaviour over a whole serving run at any width.

---

## 4. Startup, as its own scenario — and why it is not an input to the matrix

Registered decision, 2026-09-10: the acceptance matrix (G1, G2, G4) measures an
**explicitly warmed** server under `atom/compass/PROTOCOL.md`, and startup is
measured separately as its own scenario with its own numbers. One scalar was
not going to predict every compiler and cache regime, and making it try would
have spent the PoC on a question the PoC is not about — whether limited
measurements plus ATOM's own serving logic pick the right configuration for a
*running* deployment.

This is a change of scope, not a reprieve. Runs already collected under a cold
start stay labelled as cold runs; none of them becomes a pass.

### 4.1 The protocol

`atom/compass/PROTOCOL.md`, SHA-256 `19ead7bac949dcb9…`, registered
`2026-09-10T15:03:02Z` against tree `66ae9d87` +35 uncommitted. The lock is
`atom/compass/protocol.lock.json`; `scripts/compass/protocol.py verify`
recomputes it and every cell stamps its own `protocol.json`, so a cell carries
the protocol it actually ran under and a later edit cannot reach backwards.

What a cell now does, in order: verify and stamp the protocol; start servers
until one loads the compile cache *and* its predecessor inside that cell also
loaded it, which puts the measured process in the steady state by construction
rather than by trusting the machine's history; send a preparation batch of the
measured workload's own shapes and wait for all of it; drain
`GET /compass/requests`, which clears as it reads, and confirm the store is
empty; only then measure. The drained preparation rows are kept as
`*.prepare.json`.

Two properties make the drain sufficient rather than merely tidy. Every
reported metric is a within-request difference or a difference against the
measured window's own first arrival (`compare.py::metrics`), so a
preparation-shifted virtual-time origin cancels. And preparation is sent
**undeclared** — no `compass_workload_size` — because the scheduler's arrival
barrier latches open the first time a declared workload fully arrives and never
re-arms; a declared preparation batch would spend the latch and leave the
measured workload unheld, which is the exact defect the barrier exists to
prevent. Asserted against `Scheduler._arrival_barrier_unmet` itself in
`tests/compass/test_protocol.py`.

A warmed cell charges no `warmup_seconds`. The first forward of both servers
happens inside preparation, whose records are discarded, so the constant would
land in a window nothing is reported from.

### 4.2 The cold-start scenario's own result

Measured by `agent_scratch/poc/first_use_probe.sh`: one server start, two
identical 1024-token requests 30 s apart, the difference between their prefill
steps. Serialized TP 1 → 2 → 4 on 2026-09-10, nothing else of this campaign on
the devices; all three probes ran with foreign tenants on cards this campaign
did not own (isolation `node_busy`), own cards clean at every baseline.

Reported per regime, because one number cannot stand for three:

| regime | TP=1 | TP=2 | TP=4 |
| --- | --- | --- | --- |
| `compiled` — the process compiled the graph | not sampled | +6.7205 s | +6.6813 s |
| `cache-first` — first to load a cache just written | not sampled | **+45.1166 s** | **+47.3954 s** |
| `cache-steady` — the state a deployment serves in | +6.6985 s (n=5) | +6.8421 s (n=1) | +6.6170 s (n=1) |
| warm prefill, cache-steady | 0.3759 s | 0.2858 s | 0.1174 s |

TP=1 never showed `cache-first` because an earlier probe had already written
its cache; that is why its five repeats are all steady and why it alone has a
usable n. At TP=2 and TP=4 the steady figure rests on a single repeat.

The `cache-first` regime is **not** averaged into anything. It is about six
times the steady constant, it appeared once at each of TP=2 and TP=4, and it is
untraced: the cache load itself only went from 3.8 s to 8.5 s, so the other
~38 s is observed and unexplained. It is reported at full size here and it is
represented in no warmed result.

Obligations that remain, unmet: validate reuse of the steady constant on an
independent run of the same deployment before any result depends on it, and
report warm and first-use requests as separate distributions wherever both
appear. If the value ever has to move to make a workload fit, it has stopped
being a first-use constant and the change must be rejected. Preparation,
calibration and startup costs are accounted in G5c, separately and with their
amortisation stated; they are never netted out of a serving result.

---

## 5. Harness validity — checks a gate run must pass

Promoted from the retrospective's findings. A gate row may not cite a run that
fails any of these, and the check must run at the real client/consumer boundary
rather than over source text.

**Table reconciled 2026-09-11.** The states below were written against the pilot
scripts — `agent_scratch/cc_compare.py`, `spread.py`, `queue_wait.py` — which
printed warnings and continued. Those scripts still exist, are not on the gate
path, and are not what a gate row may cite. The gate path is
`scripts/compass/compare.py`, whose `check_run` and `check_pair` return a list of
reasons and whose caller refuses the run when the list is non-empty; it is
covered by `tests/compass/test_run_validity.py`. Each row now names the check
that enforces it.

| check | state | enforced by |
| --- | --- | --- |
| every request in the workload joins an engine record | **refuses** — a missing record is a refusal, not a printed count | `check_run`, the `missing` list |
| declared request count matches the saved workload, and the expected count | **refuses** | `check_run`, `--expect-requests` |
| `arrival ≤ first token ≤ finish` for every request | **refuses**, and separately refuses a reported `ttft`/`latency` that disagrees with its own timestamps by > 1 ms | `check_run`, per-request ordering loop |
| arrival-barrier timeout reaches the result | **exported and refuses** — `scheduler.py:1271` sets it, the manifest carries it, `compare.py:171` refuses on it | `check_run` |
| workload / input lengths / time scale / model match on both sides | **refuses** — on `trace_sha256`, `time_scale`, `model`, request count and per-request lengths | `check_pair` |
| produced token counts equal on both sides | **refuses** — different `usage.completion_tokens` means throughput and TPOT are not comparable quantities | `check_pair` |
| every response carries `usage.completion_tokens` | **refuses** — the requested length is never substituted | `check_run` |
| the final trace is fully drained | **refuses** — `engine_records != len(workload)` | `check_run` |
| preparation finished before the earliest measured arrival, on the engine clock | **refuses** | `check_run`, `prepare.boundary_engine_time` |
| prompt lengths verified against the workload | **refuses** unless the run was replayed `--check-lengths` | `check_run` |
| metadata identifies server code, model and calibration | **refuses** — `server_revision` **or** `server_code_sha256` (the GPU nodes are rsync copies, not checkouts), `model_revision`, and a digest for every file-valued oracle option | `check_run`, `require_provenance` |
| clock domains not mixed | **structurally closed** — there is no `perf_counter` path left in `replay.py` or `compare.py`; all timings come from the engine's `/compass/requests` records | — |
| artifacts preserved per run, with hashes | **in use** — `agent_scratch/poc/preserve.py` | — |

**Two of these are still weaker than they read.** `require_provenance` is a
parameter, so a caller may switch the provenance block off; no gate row may cite
a run compared with it off, and that is a convention rather than an enforced
property. And the length check accepts the manifest's own `"passed"` string —
it verifies that the replay ran the verification, not the verification itself.

### 5.1 The test suite, and what it takes to collect it

`tests/compass`, run 2026-09-10 in the GPU container on node 18 with one device
visible: **426 passed, nothing excluded, nothing skipped.**

The environment requirement is part of the record, because an earlier run of
mine reported it wrongly. `tests/compass/test_graph_alignment.py` — which
covers graph/capture alignment and is required coverage for G4 — cannot be
*collected* unless triton has an active driver: aiter's `arch_info` resolves the
architecture from `triton.runtime.driver.active`, and on `RuntimeError` falls
back to `from jax._src.lib import gpu_triton`, which this image does not have.
Two ways to reach that fallback: run in the device-free CPU container, or pass
an empty `HIP_VISIBLE_DEVICES` in the GPU one. I did the latter, and reported
the module as broken and pre-existing. It is neither: with a device visible it
collects and its 28 tests pass. No dependency was added and the ROCm stack was
not touched — the module needs a visible device at import, which is what a G4
capture run has anyway.

The device-free CPU container therefore does not collect that module. That is a
constraint on where the module runs, not a gap in coverage, and it is stated
here so a future suite run in that container is not read as a full pass.

**Run 2026-09-11, current source (99 files hash-verified host-to-node):** GPU
container on node 18, one device visible — **502 passed, rc=0**, nothing
excluded and nothing skipped, including `test_graph_alignment.py`. Re-run after
the formatting fixes below: 502 passed, rc=0 again. Logs and exit codes on node 18 at
`/tmp/xiaobizh-compass/ATOM/agent_scratch/g4/`: `pytest_gpu.log` / `.rc` and
`pytest_gpu2.log` / `.rc` (GPU container `xiaobizh_n18`), `pytest_20260911.log`
/ `.rc` and `pytest2.log` / `.rc` (device-free `xiaobizh_n18_cpu`). Each `.rc`
holds pytest's own exit status, written by the runner rather than inferred
from a pipeline. The source manifest checked host-to-node is
`agent_scratch/edits/manifest.txt` plus `mf2.txt`, 99 files, sha256 equal on
both sides.

The same source in the **device-free CPU container** does not pass and is not
reported as one: collection of `test_graph_alignment.py` fails with
`RuntimeError: Get GPU arch from rocminfo failed` (rocminfo returns 1, "Unable
to open /dev/kfd"), `rc=2`. Excluding that one module, `rc=1` with **18 failed,
456 passed** — and all 18 are the same aiter hardware-discovery call, not test
logic, which is what the GPU run passing all 502 on identical source
establishes. The device-free container is a valid environment for the replay
gate (G5b) and is not a valid environment for the suite.

Formatting, checked on the 45 changed/new Python files only, counts kept
separate because new files have no HEAD version to compare against:

* **22 modified tracked files.** At HEAD, 19 would be reformatted by black and
  3 were clean (`atom/config.py`, `atom/model_engine/model_runner.py`,
  `tests/compass/test_clock_advance.py`). This work left `model_runner.py`
  clean and broke the other two, which were reformatted back to clean. The 19
  fail both before and after and were not touched.
* **23 new files.** 22 would be reformatted, 1 is clean. No before-state exists
  for them; they follow the compass subtree's existing hand-formatted style,
  which is not black-clean either (`atom/compass/core/graph.py` at HEAD fails
  the same check).
* Current totals across the 45: 41 would be reformatted, 4 clean.

Ruff is advisory in CI (`--exit-zero`, reviewdog on diff context). 13 F541
findings introduced by this work were fixed; the remaining per-file counts were
recorded rather than mass-fixed, so no unrelated edit was swept in.

---

## 6. Evidence register

Paths are on the node-18 measurement host under `/workspace/ATOM/`; the packet
directory holds a copy with a `MANIFEST.json` of SHA-256 digests.

### `agent_scratch/poc/evidence/cc27_20260910/` — E1

| file | sha256 (first 16) | what |
| --- | --- | --- |
| `big_real.json` | `5f0bd3be98027d28` | real run, 27B TP=4, engine-clock records |
| `d27_final.json` | `6546b0ff18062356` | simulated run at `66ae9d87` |
| `big_sweep.tp0.jsonl` | `4bae38442eba46c3` | the calibration table both sides used |
| `cc_pilot.jsonl` | `bf4049f84be161df` | the workload |
| `big_real_steps.tp0.jsonl` | `739cc6e5dcbd94bc` | real per-step table |
| `d27_final_steps.tp0.jsonl` | `ed3b39dc9a1c636c` | simulated per-step table |

Code revision for the simulated side: `66ae9d87` on `feature/atomcompass`.
The node-18 tree is an rsync copy, not a checkout, so the revision is recorded
here rather than read from it — which is exactly the provenance gap §5 names.

---

### E2a — the short-input decision matrix (run 2026-09-10; **G2 fails**)

64 requests, 1024 input / 128 output tokens, 27B at TP 1/2/4, one cell at a
time on an otherwise idle node, every cell served by the same `atom` bytes
(`server_code_sha256` `57aa468b…`, staged by `agent_scratch/poc/freeze.sh` as
`frozen/poc-gate-a`). Artifacts under `agent_scratch/poc/matrix/tp{1,2,4}_short_idle/`.

| cell | TTFT med | TTFT p90 | TPOT med | latency med | throughput | isolation |
| --- | --- | --- | --- | --- | --- | --- |
| TP=1 | −10.4% | −24.5% | −49.6% | −21.4% | +27.3% | clean |
| TP=2 | −20.5% | −30.4% | −45.9% | −26.9% | +37.0% | clean |
| TP=4 | −46.1% | −51.3% | −39.7% | −46.1% | +85.9% | node not quiet¹ |

Gates are TTFT ≤ 15%, TPOT and throughput ≤ 10%. Every cell fails TPOT and
throughput; TP=2 and TP=4 also fail TTFT. This is a material G2 failure and it
is a different regime from the E1 long-input pilot, which does not cover it.

¹ card 4 busy in 2 of 31 samples, cards 5–7 one unexplained sample each. Card 4
is not one of this cell's devices, so its bytes cannot enter this cell's
`mem_get_info`; the finding downgrades the timings to advisory, and the TP=4
numbers above are reported under that caveat rather than as a clean measurement.

#### What the failure is, before any coefficient is touched

`scripts/compass/price_steps.py` prices the real run's *own* step sequence with
the same frozen oracle the simulated run used. That holds the schedule fixed,
so the oracle is measured on its own and not through the scheduler it drives.

| | TP=1 | TP=2 | TP=4 |
| --- | --- | --- | --- |
| decode steps, median error (n=255) | +0.63% | +0.25% | −0.52% |
| batched prefill, median error (n=4, 15–16k tokens) | −5.6% | +6.9% | −13.9% |
| **first forward** (1024 tokens, batch 1): real | **7.005s** | **7.026s** | **6.813s** |
| the same step, priced | 0.358s | 0.243s | 0.163s |
| all forwards, total error | −21.4% | −26.7% | −45.7% |
| forwards *excluding the first step* | **−3.0%** | **+5.4%** | **−6.7%** |
| host time between forwards (259 gaps) | 0.220s | 0.213s | 0.211s |

Observed elapsed (first step start to last step end) is 35.02 / 22.28 / 15.86s
against forward sums of 35.06 / 22.30 / 15.87s, so host work outside the
forwards is under 1% and is **not** the missing time. Forward seconds and host
gaps are measured against different pairs of instants and are reported apart,
not added.

Two separate defects, neither of them a cost coefficient:

**(a) The cold first forward is unmodelled.** ~6.8–7.0s, and *invariant in TP* —
it does not fall by 4× from TP=1 to TP=4 the way the same shape's compute does.
A cost that does not shard is not compute. That is evidence for an unmodelled
**first-use component** — it is not, on its own, proof of any particular cause;
compile, autotune and graph capture are candidates, and none has been witnessed.
Whatever it is, the real server pays it inside the timed window and the
simulated one never does. It is 89–114% of each cell's whole forward-time
shortfall. With it excluded, the steady-state forward error is −3.0/+5.4/−6.7%,
inside the 10% gate. §4's `warmup_seconds` is the input this belongs in, and it
has never been calibrated for this deployment. First-use and steady-state
results are kept apart here on purpose; the decode fit is already within ±1% and
must not be moved to absorb a one-off.

**(b) HYPOTHESIS — predict mode may attribute its own virtual time
differently.** Not established. The walk below advances a clock by step
durations and credits each request in a step's `req_ids` with one token at that
step's completion. In an engine with deferred output processing that assumption
is wrong, and a discrepancy it produces is a discrepancy in the assumption, not
a proven timestamp bug. Settling it needs the token lifecycle witnessed on both
sides — produced and returned request IDs, final-chunk state, token append, and
first-token publication — not a step table. Recorded here as the leading
hypothesis and as the reason to instrument those events. Walking the
*simulated* run's own step table gives TPOT medians of 0.0708 / 0.0419 / 0.0230s
and first-token medians of 18.47 / 10.93 / 5.59s. The same run *reported* TPOT
0.0343 / 0.0203 / 0.0128s and TTFT 23.18 / 13.71 / 6.91s. Total latency matches
the step table exactly (27.537 / 16.287 / 8.541s), so time is being moved from
the decode phase into TTFT rather than lost: reported TPOT is 0.48–0.56× what
the engine's own steps imply. The real run has no such discrepancy — its
reported TTFT (25.86s) and TPOT (0.0681s) match its step table (25.77s, 0.0688s)
to under 1%. So *if* the walk's publication assumption held on both sides, the
TPOT failure would be an output-contract difference rather than a prediction
error — but the assumption is exactly what is untested, and the real side
agreeing with it is consistent with either the simulated side deferring outputs
or the simulated side stamping wrongly. As a
counterfactual, substituting priced costs into the real step sequence gives TPOT
errors of −0.3 / +5.8 / +1.5%.

Neither finding was obtained by fitting to the reported TTFT or TPOT.

---

### E2b — the long half stopped on a device fault (2026-09-10)

The TP=1 long calibration sweep, from the frozen `poc-gate-a` tree, died on
card 0 at 12:00:43 partway through its 65 536-token prefill buckets:

```
[atom 12:00:43] Scheduled prefill batch: 1 reqs, 16384 new tokens (done: [16384], new: [16384]), req_ids: (490,)
Memory access fault by GPU node-2 (Agent handle: 0x10a61d50) on address 0x7fe605869000. Reason: Unknown.
[atom 12:19:54] AsyncIOProcManager(CompassModelRunner): [ModelRunner0/1] proc died unexpectedly (exitcode=-6)
```

605 rows had been written. They are kept as
`sweep_tp1_long.CRASHED-partial-605rows.jsonl` and the log as
`tp1_long_gatea/sweep.CRASHED.log` — renamed away from the name a *finished*
sweep has, because `cell.sh` reuses a non-empty sweep table without asking
whether it is complete, and a partial calibration reused silently is worse than
no calibration at all.

**Where it happened, from the preserved table.** The sweep's single-request
long rounds all *succeeded* first: the partial table's largest prefill context
is 258 048 and its largest decode total context 258 080 at rung 32. The fault
came in the first round with several long sequences at once — `(65536, 8)`,
eight 64 k requests — after requests 488 and 489 had each completed all four of
their 16 384-token chunks, on request 490's second chunk. Resident KV at that
instant was roughly 163 840 tokens across three sequences.

So it is not per-sequence context length: a single 258 k sequence was fine and a
third 64 k one was not. Whatever it is involves several long sequences resident
together. There is no Python traceback to read — the worker took SIGABRT from
an HSA memory-access fault, which is a kernel reading or writing outside its
allocation, so the cause is below ATOM's Python layer.

**It is not an over-extreme calibration round.** The registered long gate
workload (`cc_pilot.jsonl`, first 20 requests) is 640–119 360 input tokens,
1.68 M tokens in total, arriving over 244.5 s. Prefilling that much at the
measured ~7–8 s per 16 k chunk takes far longer than the arrival span, so the
requests accumulate: the gate workload enters a *larger* multi-sequence region
than the round that faulted, not a smaller one. The sweep did not ask for
something the gate run will not.

(The rounds that *would* be arguably beyond this gate workload — `(196608, 1)`
and `(258048, 1)`, above the workload's 119 360-token maximum — are precisely
the ones that completed.)

Still not known: whether the fault is reproducible, and whether it is specific
to the frozen tree. The next experiment is the minimal one — the `(65536, 8)`
round alone, on a free card — rather than another four-hour sweep.

The 27B **short** half is unaffected — it was measured before this and is
reported in E2a. The long half of the decision matrix is blocked until this is
diagnosed. The parent process wedged after the worker died (EngineCore zombie,
no GPU held, 0% CPU) and is left to exit on its own timeout.

---

## 7. Open items, ranked by which gate they block

Re-ranked 2026-09-11 under the cc-traces acceptance registration. Items that
block only a diagnostic are marked as such and no longer compete for priority
with items that block a gate.

| # | item | blocks |
| --- | --- | --- |
| 1 | register the end-to-end cc-traces acceptance protocol: dataset version, scope, short/long regime definitions, selection rules, request identities and order, lengths, arrival pacing, preparation protocol, hashes | **every gate** — until this is stamped, no gate row can move |
| 2 | cc-traces coverage: what fraction of the corpus's shapes the price library covers, and the structural cache's cost and hit rate at that coverage | every gate, via the oracle the matrix runs on |
| 3 | the full short/long × TP{1,2,4} cc-traces serving matrix | G1, G1b, G1c, G2a, G2b, G2c, G4 |
| 4 | a timed cc-traces workload with capture, calibration and startup costs separated and amortised | G5a, G5c |
| 5 | per-term memory and KV validation *at the matrix's own deployments*, and an infeasible configuration rejected for ATOM's own reason | G3a, G3b, G3c |
| 6 | diagnose the long-sweep device fault (E2b) — it blocks any long cc-traces cell that needs a calibration sweep | items 2 and 3 |
| 7 | witness the token lifecycle on both sides; settle E2a-b | G2, if any synthetic diagnostic is to stay interpretable |
| 8 | calibrate `warmup_seconds` for this deployment; the cold first forward is ~6.9 s and unmodelled (E2a-a) | G2, and the preparation protocol in item 1 |
| 9 | prefix-reuse replay from `hash_ids`, so a cc-traces cell can be run with native cache policy rather than with prefix caching off | the scope boundary of item 1, not a gate as currently registered |
| 10 | **require a repeat pricing run before any freeze** — E8's miss was one price measured once (§2 E8) | the credibility of every future frozen prediction |
| 11 | outlier rejection by region (the 4-MAD pass drops 83% of sub-1024-token prefill rows) | fit quality, not a gate unless it moves one |

**Closed since the last ranking.** Item 5 of the old list — "predict TP=2/4 from
a TP=1 capture" — is done as a *diagnostic* and is not coming back in that form:
E7 answered it at TP2 (+5.7%) and E8 answered it at TP4 (+26.2%, failed). The
gate version of the question is item 3, over a serving run. Old item 8 —
"promote the §5 validity checks and test them at the client boundary" — is done;
§5's table above names the check that enforces each row and
`tests/compass/test_run_validity.py` covers them. Two residual weaknesses in
that work are stated at the end of §5 rather than left implicit here.
