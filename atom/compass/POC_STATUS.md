# ATOMCompass — PoC gate status

**Current user priority (2026-09-13):** the 5x speed target is advisory and may
be compromised. Accuracy, memory, feasibility, configuration selection and
ranking retain their requirements. Report observed speed and derivation costs
without calling a missed 5x target a pass; `cc_traces_validate.py matrix
--speed-advisory` records this policy explicitly. Existing strict protocol
registrations and their historical verdicts remain unchanged.

**Current execution priority (2026-09-15): prefix caching enabled for the
main cc-traces completion path.** The [coverage and confirmation
review](CC_TRACES_COVERAGE_REVIEW.md) controls coverage and case selection.
The original cache-disabled registration and its **9/24** passing cells
(including **8/8 original TP1**) retain their historical meaning. They do not
prove the cache-enabled configuration, whole-corpus accuracy or **99%
confidence**. No cache-enabled paired accuracy result is established yet.
The earlier short-c4 campaign and staged cache-disabled long real run are held.

**Cache-on implementation is integrated at `2782333ab`.** The new diagnostic
path binds an explicit cache policy and source-hash prompt encoding; both
engines use prefix caching, checkpoint interval 8192, demand checkpoints and
native one-token GDN fork semantics. It retains the original cache-off plan.
The native/replay allocation bridge now carries distinct fork source and
destination slots. A quiescent cache reset fences every worker, clears both
indexes and records policy/pool/counter snapshots; a modelled run must remain
fresh at its original virtual epoch. Future requests do not publish prefix
demand before readiness. Captured and derived target/memory provenance retain
the source policy, with cache-on memory reuse explicitly an unvalidated
derived candidate. **648 targeted CPU regressions pass** on exact source
`2782333abba2135050072f5966f41c79472c2142`, with zero failures, errors or skips
in a container with no GPU device nodes. These contracts alone prove neither
cache hits, pool pressure, native memory terms, region coverage nor E2E accuracy.
Cache-aware diagnostics explicitly call the policy-aware engine and memory
checks plus `check_cache_policy_evidence`; the registered `cell` checker
retains its historical cache-off contract.

**Latest cache-disabled diagnostic evidence, preserved as history:**

| Case | Completed evidence and remaining qualification |
| --- | --- |
| Near-limit 250,048 input / 39 output | Fresh pair on `4b109a968` completed 1/1 on both sides. TTFT **+5.70%**, TPOT **+4.30%**, throughput **−5.38%**; all maintained provenance/capacity/memory checks pass. One paired diagnostic, not a registered cell or confidence sample. |
| Tiny 128 input / 16 output | The finite gather-price overlay completed a fresh pair. TTFT **−46.73% fails** the 15% bar; TPOT **+5.65%** and throughput **+8.16%** are within their 10% bars. Non-Torch memory **2,046,820,352 real / 1,157,627,904 modelled bytes** fails its component bar; KV **111,930 / 112,773**, error **0.753%**, is within 5%. The smaller KV error does not erase the component failure. |
| Long decode 195,840 input / 40,339 output | Modelled side on `74bb2d128` completed 1/1 and all 13 maintained checks. Predicted TTFT **133.296848 s**, TPOT **0.035998853 s**, latency **1,585.418576 s**. The real side is staged with an explicit diagnostic warmup-output cap of 32 and unchanged measured output, but is **HOLD** after the cache-policy change. No paired accuracy result. |
| Intact seven-request root `1493faff…` | Its cache-disabled modelled run refused at the opening 448-token prefill: 16 unpriced `triton::_fused_qk_norm_single_kernel` occurrences. The refusal and cleanup are preserved; the real side was not launched. Further cache-disabled source collection/retry is held. |

The tiny non-Torch discrepancy remains a measured failure with unresolved
attribution. Native allocation and its recorder use the current PyTorch
device's `mem_get_info` and `memory_reserved`, not an SMI device-index path;
the proposed SMI-index mismatch is unsupported by that code. No cause or
corrected constant follows from the observation, and no retuning was done.
All owned diagnostic serving processes above are closed; the queued long
real plan remains unexecuted.

**Historical cache-disabled corpus diagnostics (2026-09-14): first fresh sustained-arrivals pair;
no new registered acceptance cell.** The TP1 pair on `b27d0e7f2` completed
**21/21 requests and 15,979 output tokens on each side**, using the unchanged
`source-27b-tp1-history-2m` preset and diagnostic readiness profile `6e26d4cb`.
One root client reached **21 in-flight requests and a scheduled batch of 21**
on both native and modelled engines; source overlap was only five. Measured
steps were 1,538 real (30 prefill / 1,508 decode) and 1,539 modelled
(30 / 1,509), with maximum context 54,393. This is one intact source episode
under warm/empty replay semantics, not full-root continuity.

| Sustained diagnostic metric | Signed error, modelled versus real |
| --- | --- |
| TTFT median / mean / p90 | −3.38% / +7.30% / −0.12% |
| TPOT median / mean / p90 | +1.84% / **+10.74%** / +8.18% |
| Throughput | −7.12% (63.396 versus 68.258 output tokens/s) |

Individual absolute TTFT/TPOT errors reach **32.31% / 38.23%**. The 15% TTFT
and 10% TPOT reference bars contain 15/21 and 12/21 requests, respectively,
and only 6/21 jointly; these descriptive counts establish no confidence claim.
All eight non-KV components pass 10% (maximum 1.075%); KV blocks are
112,760 real / 112,773 modelled, **0.01153%** error. Request, pair, preparation
drain, serving identity, calibration and GPU-free provenance checks pass.
Real/modelled measured wall windows are 234.857/119.615 seconds (**1.963×**,
advisory). Isolation is `own_clean=true`, `node_busy`; one unknown zero-use
foreign PID was preserved. Native cache key `28d953bf35` and preparation logs
are retained. GPU1 and ports were released; one resource-tracker zombie under
PID1 has no command or FDs. This single pair adds no three-repeat registered
cell and leaves the original **9/24** count unchanged.

**Earlier region-domain refusals remain recorded.** Low-byte readiness support
passed, but the selected region model
originally refused compiled output-producing N1 prefill at 128 tokens against
`[640,16384]`. The seven-request full root begins at 448 and was held by explicit
preflight without a run. The near-limit 250,048/39 case priced all 16 prefills
(15 × 16,384 plus 4,288), then refused first decode history **250,049** against
`[128,196608]`. Its 39-token response did not complete. The “2M history” source
extended summed history at selected 16/32-request cells, not the per-row bound.

Tiny used the old wrapper v1: its reconstructed source-path defect remains in
the original record, while actual `--trace` and separate identity pins survive.
Near-limit used wrapper v2 with correct source identity and passing provenance
checks. Both owned waiting replays were stopped after worker refusal; original
journals remain `exit=-15`, `ok=false`, `refused=false` (near-limit wrapper exit
1). They are domain witnesses and incomplete diagnostics, not valid paired
accuracy results. Both CPU lanes are closed. The maintained diagnostic CLI and
owned-refusal monitor are now integrated in `60dc92c84`; their future exit-5
model-refusal classification does not rewrite those old outcomes.

The separate TP1 preset `source-27b-tp1-history-256k` is integrated in
**`4b109a968`**. All **56/56**
frozen source heldouts pass the fixed 110 µs preparation-increment criterion
(maximum **12.560 µs**). The per-row domain reaches 262,143; 11 legacy
snapshots and 411 old breakdown/band comparisons remain exact, with 209 tests
passing. The sustained pair used the old preset; the fresh near-limit pair
on the new preset subsequently completed with the passing diagnostic results
above. This does not establish cache-enabled E2E coverage.

Low-prefill native source validation **failed** its fixed 110 µs preparation
criterion: 17/30 pass, 13 fail, maximum **427.143 µs**. Postprocess 14/14 and
structural-zero 16/16 checks pass. Explicit TP1 diagnostic candidate `e43976eb`
uses already-frozen anchors; it changes no defaults or acceptance selection,
does not relabel the failed criterion, and adds no N2/pool support. Its tiny
CPU follow-on loaded successfully but exited **5** on the missing head-row
gather price `aten::index.Tensor|128,5120;1|bfloat16,int32|1:127`
(2,442/2,443 operators priced). That pre-overlay attempt completed no tiny
response. The later finite-overlay pair and intact-root QK-norm refusal are
recorded above; the failed low-region criterion remains a failure.
**TP2/TP4 remain paused.** Next useful evidence is a cache-enabled native
source/oracle preflight and a fresh paired exposed-root diagnostic.
N1 low-prefill
support must not stand in for multi-client N2/pool support; the coverage review
records a minimal exposed-root C2/C4 cold-opening proposal, with actual batching
to be witnessed and no manufactured C8 case.

The seven original cache-disabled diagnostic manifests are unchanged; new
cache-aware fixtures receive separate identities. **249 roots remain exposed and
at most 144 potentially untouched**, with no new reserve allocation. The
[confirmation next step](CC_TRACES_COVERAGE_REVIEW.md#6-confidence-limitation-and-next-step)
requires a frozen probability design spanning both partitions, cluster-aware
uncertainty and an inconclusive stop; detailed sampling is deferred. It does not claim that a small suite,
static coverage or a bootstrap establishes literal joint 99% over all 393 roots.
The original paired count below remains **9/24, including 8/8 original TP1**.

**Historical 2026-09-14 cache-disabled checkpoint: nine paired cc-traces cells pass their registered
aggregate checks, including all 8/8 original TP1 cells.** The original TP1
execution sweep is closed. Its eight cells use the frozen candidate at commit
`2feea392`; the retained first TP2 cell uses `8a637eff`. Each has three fresh
real and three fresh GPU-free modelled runs. Independent review reproduced
aggregate metrics, per-rank memory, exact requests, provenance, process
identities, derivation accounting and sampler coverage from the raw artifacts.
Each combined validator reports `accepted=true`, `passed=true`, and no
failures under the current contract.

| Paired cell | Requests per repeat | Throughput error | TTFT median error | TPOT median error | Replay ratio |
| --- | --- | --- | --- | --- | --- |
| `tp1_clients_short_c1` | 2 | −3.542386% | +9.796520% | +4.475799% | 1.36249× |
| `tp1_clients_short_c2` | 4 | −3.067558% | +3.686786% | +3.270906% | 1.30190× |
| `tp1_clients_short_c4` | 8 | −2.363219% | −1.069261% | +3.665594% | 1.22940× |
| `tp1_clients_short_c8` | 17 | −2.843880% | +3.461302% | +9.147776% | 1.12427× |
| `tp1_clients_large_c1` | 7 | −6.113405% | +12.660296% | +5.035067% | 2.29158× |
| `tp1_clients_large_c2` | 14 | −6.046615% | +9.740734% | +6.159878% | 2.48310× |
| `tp1_clients_large_c4` | 21 | −6.510275% | +9.598291% | +8.711996% | 2.49948× |
| `tp1_clients_large_c8` | 37 | −6.776582% | +9.143126% | +5.991701% | 2.53304× |
| `tp2_clients_short_c1` | 2 | −6.742557% | +3.791065% | +6.445119% | 0.61627× |

The registered aggregate tolerances are 10% throughput, 15% TTFT and 10%
TPOT. All eight non-KV components were compared on every rank in every
repeat. TP1 maximum component error is **0.035461%**; TP2 maximum is
**7.859768%**, both within 10%. TP1 has **112,772 real / 112,773 modelled KV
blocks** (**0.000887%** error). Both TP2 ranks have **265,540 real / 266,768
modelled blocks** (**0.462454%** error), within the 5% bar.

**Short-workload tail/admission discrepancies remain open.** The existing
median-based gates pass; the following p90 diagnostics remain separate.

| TP1 cell | TTFT p90 error | TPOT p90 error |
| --- | --- | --- |
| `clients_short_c2` | **+30.164484%** | **−10.013389%** |
| `clients_short_c4` | +8.283975% | **−22.096999%** |
| `clients_short_c8` | +5.849286% | **−14.687403%** |

The highlighted tails exceed the corresponding 15% TTFT or 10% TPOT
reference bars. They are not silently substituted for the existing aggregate
gate or hidden by its passing medians. Short-c8 median TPOT has only
**0.852224 percentage points** of margin. Large-c1 p90 TTFT/TPOT errors are
**+12.261761% / +8.321208%**; large-c2 values are
**+10.087576% / +8.219798%**; large-c4 values are
**+9.751630% / +9.273788%**; large-c8 values are
**+9.370755% / +8.892671%**, within those reference bars. No acceptance
coefficient, workload or threshold was changed to obtain these comparisons.

Native TP1 host-enqueue source probes v2 and v3 completed with the accepted
FULL compilation configuration and cache `dd263e63bc`, preserving native
waits and natural drains. V3 adds prepare-model and postprocess host
boundaries. The separate CPU handoff source collection passed structural
checks, and its endpoint-only models passed all 24 heldout component
comparisons under the predeclared 100-microsecond absolute service-error
budget after predictions were frozen. The subsequent low-payload source
extension collected **7,668 frames / 6,816 measured frames** after the GPU
timing window closed. Its endpoint-only fit and predictions were frozen before
excluded timings were released. **All 84 heldout component checks pass** the
unchanged 100-microsecond limit; the maximum error is **49.154504 microseconds**.
Candidate profile `6e26d4cb` accepts all **532 previously refused requests /
1,064 byte variants across 340 roots** in the positive canary; **56 CPU
regressions pass**. The candidate was subsequently selected **only for the new
modelled corpus diagnostics**, with registry `94ef3d0f`; no registered acceptance
cell was changed. These CPU source/profile checks add no E2E acceptance cell and do not prove
whole-corpus accuracy. Representative fresh E2E confirmation remains required.
Probe v1's failure and all earlier acceptance artifacts remain preserved.

The reviewed asynchronous HTTP client (`0075d21f`) submits requests without
allocating a separate thread for each one. Later requests proceed independently
of earlier responses.
CPU transport tests registered all **1,025 / 3,551 / 16,913 requests** before
any response; large-payload, pacing, timeout and cancellation checks also pass.
These are transport tests, not GPU serving results. Socket/file-descriptor and
explicit client-memory limits remain; insufficient resources refuse the whole
workload. Preparing request JSON before pacing changes send jitter, so this
client requires **fresh paired E2E runs**; retained real references are not
interchangeable with new transport runs.

**Corrected readiness/preparation-fence diagnostics improve short-c4 tails,
with residuals still explicit.** One fresh GPU-free modelled repeat for each
of short-c4 and short-c1 was compared with its three retained real references.
These are diagnostics, not fresh paired acceptance. The profile and successor
registry remain frozen; no coefficient was fitted to these E2E outcomes.

| Corrected TP1 diagnostic | Throughput error range | TTFT median / p90 error range | TPOT median / p90 error range |
| --- | --- | --- | --- |
| `clients_short_c4` | −3.91% to −3.61% | +5.47–5.79% / +8.66–8.90% | +3.67–3.91% / +7.43–7.83% |
| `clients_short_c1` | −3.90% to −3.51% | +8.96–10.21% / +8.96–10.21% | +4.24–4.80% / +4.24–4.80% |

Corrected short-c4 reproduces native prefill group membership and sizes
**1, 3, 3, 1** in all three references; one reference differs only in order
within a group. Its first-request TTFT changes from 3.822214 s to
**0.613801 s**, versus **0.743536–0.802088 s** observed: a remaining
**17.45–23.47% underprediction**. Dispatch after a following prefill remains
unpriced; extra origin/transport are zero assumptions. Short-c1 TTFT and
latency change by only **+62.94 microseconds**, with TPOT unchanged.

The corrected large-c4 diagnostic also passes aggregate and p90 reference
bars, while exposing an arrival-order approximation. Its first-request TTFT
is **150.797 s**, versus about **60 s** in real repeats 1/2 and **139.736 s**
in repeat 3. After the first request's matching five full prefill chunks,
the 2,048-token tail shares its next batch with a 51,328-token request in
real repeats 1/2, but with a 170,368-token request in the model and real
repeat 3. Real initial arrival order is 0,2,3,1 versus 0,1,2,3; the model
uses the declared equal arrival times and fixed index order. Deferred first
output therefore waits through different neighbouring prefill chunks.
The bounded audit attributes this difference to the explicit arrival/order
approximation and native variability; it does not justify fitting a new
kernel cost. These diagnostics retain their own evidence and do not replace
the original accepted pairs. The corrected short-c4 paired bundle remains
frozen at `e7e4d2588` and held; no acceptance server was launched from it.

**Scope: 9 of 24 positive cells pass the current aggregate and memory
checks; 15 remain.** All eight original TP1 cells have three fresh real and
three fresh modelled repeats, covering C=1/2/4/8 in both selected workload
classes. **The large-c8 real side and its cleanup are closed.** Its 37 requests
per repeat are preserved; both sides reached 37 in flight with scheduled
batches capped at 32. Client count is not an in-flight-request bound.
TP2/TP4 expansion and the earlier corrected short-c4 campaign are held while
whole-corpus functionality, coverage and confirmation design control priorities. The retained TP2 cell
establishes one E2E transfer result; broader configuration selection, transfer
and ranking remain incomplete. The tail discrepancies are not closed by this
count. G3c retains its separate frozen negative cc-traces witness below.
Earlier sampler, source and serving failures remain preserved.

**Shared-node qualification:** all nine assigned-device audits report
`own_clean=true`, with sampler coverage through final server exit. Foreign
processes and activity outside the assigned set leave the isolation verdict
`node_busy`, so timing results remain advisory. Closing samples and later
release observations remain separate evidence; the first cell's original
closing sample and subsequent idle-baseline check are intact. Large-c2
recorded unassigned GPU6 activity in 43 of 616 observations, up to 97% use;
this qualification remains attached to the passing aggregate result.
Large-c4 recorded bursts on unassigned devices in up to 18 of 720 samples,
with a peak of 76% on GPU4. Its original closing sample saw 16% GPU1 VRAM
with no owned process remaining; the later release check separately proved
0% use and 297,689,088 bytes before large-c8 started. Neither observation is
relabelled as the other. Large-c8 recorded GPU6 activity in 73 of 948 samples,
up to 100% use, and retained a separate unexplained single GPU6 observation.
Its closing GPU1 sample and later release check both show 0% use and
297,689,088 bytes; all three original servers are gone and their ports are
free. These assigned-device checks do not establish a quiet node.
TP2 also observed neighbour activity up to 51%. Its final sample covers server
exit and still reported 72% VRAM on GPU4. Separate allocation release and
process-identity checks found no live original process; three defunct children
without file descriptors remain recorded. New foreign GPU work after release
is separate from the accepted timing window; no idle-node claim is made.
Local modelled-lane and bounded factory-check overlap is disclosed in
`codex_tp1_remaining_v1/LOCAL_CPU_OVERLAP_DISCLOSURE.json`.

All replay ratios in the table are below the advisory 5× target; none is
reported as a speed pass. Startup, execution and derivation windows are
recorded. Full historical capture/calibration durations and a separate load
duration remain unknown, so acquisition amortisation is not established.

The first-cell evidence remains in the local `jgong5_compass_cpu` stage at
`/workspace/ATOM/agent_scratch/codex_tp1_first_pair_v2/`.
`results/tp1_clients_short_c1/cc_traces_cell.json` has SHA256
`9f1b5ce16cf68ec658520a48f1a0f055f65e9d2664b53b27a624e9b5d6924a23`.

`FIRST_CELL_EXECUTION_REVIEW.json` records the execution qualification;
`INDEPENDENT_FIRST_CELL_REVIEW.json` preserves the accepted independent
review, and `CLOSE_AFTER_RUN.json` preserves the separate release check.
The seven paired TP1 continuation cells are under
`/workspace/ATOM/agent_scratch/codex_tp1_remaining_v1/results/`.
Their `cc_traces_cell.json` hashes are:

| TP1 continuation cell | SHA256 |
| --- | --- |
| `clients_short_c2` | `883f8d992b262b17bd19fd042239a114d2648613e78f8344b3592e37162d6dd6` |
| `clients_short_c4` | `9c6bc5fab5a291b2ba02f4d106a097065ff288d5ce1bc035b4d11b5f9a898917` |
| `clients_short_c8` | `44c3d33dcdc6ae871bea4ab2a91c14d76a73b1bd4a49893ae516e3c43e28c220` |
| `clients_large_c1` | `70400e1aab1f43b1d592cc7a061271d5050fdb6026fb07f8e6f4885b6985410a` |
| `clients_large_c2` | `ffa075ae4531c2f10c8f485297899d8d961ea1b6b02c2d583fc4ee7c4d57cce7` |
| `clients_large_c4` | `ca31201b576e21a46d090b31939fc1c7f83b2b628f21157bf4d3afdad5b27497` |
| `clients_large_c8` | `8d23af0f83424d57fdcb6795b013823e8cfefdc65547d43884d6ad6028f3e635` |

Each `<cell>.EXECUTION_REVIEW.json` in the continuation root retains median,
p90, memory and closing-sampler evidence; `TP1_CONTINUATION_COMPLETION.json`
records the completed modelled sweep. Source probe evidence is under
`codex_decode_domain_v1/native_host_enqueue_v3/target_gpu1_v1/`; its
`COMPLETE.json` has SHA256
`39ed2cb3a6ae4d82ce0fa1b63be1e46e6e8830718ad93b33d8d5a6a5d9ad661d`.
The CPU contract, endpoint prediction freeze and heldout verdict remain under
`codex_decode_domain_v1/ingress_handoff_support_v2/`. Corrected diagnostic
analyses are under `codex_tp1_readiness_candidate_v1/diagnostics_v1/`.
The short-c4 and short-c1 `DIAGNOSTIC_ANALYSIS.json` hashes are respectively
`992d3d36637bfd500be35be49df5d036909b7f691f5e006fd68ea830fd231a8c` and
`82d97b967f8dbc24e44870b3dafadef9b0e5c99108d53a0c8c29e7012f2b3457`.
The low-payload source extension is under
`codex_decode_domain_v1/ingress_handoff_low_extension_v1/`.
`HELDOUT_SUPPORT_VERDICT_V1.json` has SHA256
`a3e082f4f65d0b13abeaeadeb1a2f0eed7f846300ee73529ca37f97d5f680c6a`;
its raw `target_host_v1/handoff/RESULT.json` has SHA256
`bba73a22bb64824bb466921d329b13259a55efd61e8044c3186cdda2f16fbff9`.
Raw traces, source probe outputs and temporary reviews remain ignored
scratch artifacts.

New refusal evidence is under `codex_corpus_diagnostics_v1/results/` in the
tiny and near-limit case directories. Their `REFUSAL_HANDOFF_V1.json` hashes
are `38f9ba230d252dbd3821d2725d3f903bad981ade28d2c1cdc9fd07bec5d881a8`
and `b2f44ec02f84db1b9425a98b4f300ead3a51793d1fff06c7b562470b64e2a059`.
The source audit's `REGION_SUPPORT_HANDOFF_V1.json` hash is
`59870d4c9dbc5668fd71927f4f56447fdaba84a8011365d6103c5a59f3779b66`.

The sustained pair's `codex_sustained_arrivals_v1/PAIRED_HANDOFF.json` has SHA256
`45052b75d5db0fa4cec2257b22be1d3ba9017df9ae249a3da3e88590f34595e1`.
It pins every artifact, remote-copy hash check, validation result and cleanup
observation in the local CPU evidence tree. Execution IDs are
`cx-4c1e447bc8f14546` (real) and `cx-d28d93604c2d5e8a` (modelled).

Later cache-disabled diagnostic receipts are retained separately:

| Artifact | SHA-256 |
| --- | --- |
| `codex_near_limit_high_v1/PAIR_VALIDATION_V1.json` | `aadf9f70a21616efaa3c0150261bc8b8b1d53bacc66707b093f835e9d68e8cbc` |
| `codex_long_decode_high_v1/MODELLED_VALIDATION_V1.json` | `6fb8a2595ea310441303bdb12d8dfb4f6b79b83d481df8e2f42bde86cd6e7116` |
| `codex_low_gather_source_v1/finite_overlay_v1/TINY_NO_CACHE_PAIR.json` | `799b73abc5c05e92eca8953696c56f2514412c76cb47e8f4670ede856db8bbe4` |
| Intact-root first QK-norm refusal | `b3d2ff0e4217be77d8dbac60b4bb601be633c562f13113cc785add8c9767e794` |
| `codex_cache_on_integration_2782333_v1/CPU_REGRESSION_V1.json` (CPU contracts only) | `b485bb197addca09c8431b32d3174cceceeda83211ac334f96d2e7de10cb3e53` |

The intact-root closeout is under
`codex_low_gather_source_v1/finite_overlay_v1/root_modelled_execution/REFUSAL_CLOSEOUT.json`.
The tiny registry's metadata-only price-rollup addendum is
`REGISTRY_WITH_PRICE_ROLLUP.json`, SHA256
`0329181fd83f01c08195eb1884cbea6b9f817b44128c05a7e8797b860fc9b814`;
the original omission and all timing failures remain preserved.

**First TP2 E2E transfer cell verified.** The frozen `8a637eff` candidate's
three real/modelled pairs preserve both registered requests and pass all
48 non-KV component and six KV comparisons. Non-Torch memory is
**7,417,626,624 B observed / 6,834,618,368 B predicted** on both ranks in all
repeats, a 556 MiB underprediction within the 10% component bar. The primer's
transient 7.74 GiB reading did not persist; no calibration was changed to
absorb either observation.

Evidence is under `codex_tp2_first_pair_v1/` in the same local stage.
`results/tp2_clients_short_c1/cc_traces_cell.json` has SHA256
`c0949a7290a89ef65fabd595823cb68c6aaf52e8f5600c8e8460c802e1c2ef11`.
`TP2_FIRST_CELL_EXECUTION_REVIEW.json`, `CLOSE_AFTER_RUN.json` and
`OWNED_PROCESS_CLOSURE_V2.json` retain the timing, memory and release
qualifications. `BUNDLE_MANIFEST.json` preserves all eight workloads in one
source snapshot, verified reader/provenance reuse, and the identical absolute
source path/cache environment used by primer and real execution. TP4 remains
a separate unresolved validation task.

At commit `6e5596dfaeab0b682bcbf9f7cdd7672435ef0b32`, TP1 with
`gpu_memory_utilization=0.34` rejected the first registered `clients_large_c8`
request (83,968 input tokens, 3,008 requested output tokens) on both the real
and device-free servers. Both returned HTTP 400 because the prompt exceeded
the single-request KV capacity. The model's prediction was frozen before
execution: 3,535 blocks versus 3,534 on the real server after a prior process
had loaded its compiled cache (0.0283% difference). The API performs this
capacity check before scheduler admission. One unservable member suffices to
reject this deployment for that workload; this is a negative witness, not a
shortened positive acceptance cell.

The first cache-loading native attempt is also retained: it reported 3,105
blocks and 1.50 GiB non-Torch memory versus the warm model's 1.08 GiB. Its
rejection was correct, but it is not capacity-accuracy evidence. The subsequent
cache-steady process reported 1.08 GiB and 3,534 blocks. No memory coefficient
was fitted to either observation. Prediction, original request identity,
result hashes, both attempts and qualifications are under node18's
`/workspace/ATOM/agent_scratch/codex_supervision/infeasible_admission_v1/`:
`PREDICTION_FROZEN.json`, `OBSERVED_FIRST.json`, and `OBSERVED_STEADY.json`.
The parent workload SHA256 is
`3ba365adc2c39bf0fb56a5e9154b372f9d44291d4d4a0d2bdc1ed67a5ca7df95`.

**Positive v1 progress — numerical agreement, not acceptance.** Artifacts are
retained under node18's
`/workspace/ATOM/agent_scratch/codex_final_20260913_v1/`. In
`results/tp1_clients_short_c1/`, all three real/modelled pairs completed the
same two requests per repeat. `all_pairs_numeric_comparison.json` records
throughput errors **−3.80%, −3.25%, −3.23%**, TTFT aggregate errors **≤5.48%**,
TPOT aggregate errors **≤4.03%**, non-KV component errors **≤0.036%**, and a
one-block KV difference (**112,773 predicted / 112,772 real**). These numerical
comparisons are within their error bars; the cell is **not acceptance**:
`run.real.json` and `gpu.jsonl` retain a failed sampler endpoint, with the last
sample 0.574 s before the last server exit. The v1 command also used
`interpolate=1`, leaving the supported interpolation ratio implicit.

`results/tp2_clients_short_c1/run.modelled.json` records three completed
modelled repeats. Its old output lacks memory predictions for every target
rank, so it cannot establish the TP2 memory gate. The v1 artifacts remain
unchanged. Three fixes are integrated for the next frozen source snapshot:

| Commit | Integrated correction |
| --- | --- |
| `b5c9d412` | Publish and validate memory predictions for every target TP rank. |
| `9b65a370` | Align acceptance with the actual source factory contract and explicitly record `interpolate=2.0` support. |
| `1e95b1ca` | Take a fresh closing GPU observation and publish phase/owned-process provenance without claiming foreign processes. |

The local device-free lane, `jgong5_compass_cpu` on `hjbog-srdc-39`, is ready
for execution: no GPU device nodes, 1,548 runtime files aligned, and all 1,710
frozen inputs verified. Its TP2 factory canary reported 675 reads and 266,768
KV blocks; the local receipt is
`/workspace/ATOM/agent_scratch/local_cpu_readiness.json`. This establishes lane
readiness only. The first local TP1 `clients_short_c2` repeat, retained in the
v1 tree at `results/tp1_clients_short_c2/run.modelled.json`, stopped at the
**5,760-token GEMM dispatch support gap** (M=5760, K=5120, N=14336, bf16,
between the 5440/5824 kernel-switch anchors). That gap is being addressed
separately. Corrected end-to-end runs and the complete configuration matrix
remain required; no positive gate is promoted by these checks.

**Subsequent source and C8 checks (2026-09-13).** Provisional supplemental
sources now pass all 8,960 ordinary GEMM lookups: five native weight geometries,
256 prefill row counts, and seven rank libraries. All 931 earlier exact
measurements retain their values; new books remain exact-only. This establishes
functional coverage on the 64-token grid, not numerical acceptance. Source
stability and reference-condition holds remain explicit while longer measurements
repair the affected rows (`codex_gemm_campaign_20260913_v1/`).

The local TP1 `clients_large_c8` diagnostic preserved all 37 requests from
eight root clients. It completed 99 prefill steps and reached 32 running requests
before refusing summed decode history 1,618,144, above the previous preparation
limit 1,572,864. No preemption or non-64-aligned prefill was observed before
that refusal. Evidence is under `provisional_tp1_c8_v1/` on node18; the run
failed and is not acceptance.

Commit `99c6af0c` adds the independently measured
`source-27b-tp1-history-2m` profile through total history 2,097,152, preserving
all old-domain predictions and bands. Its 56 source cases include 16 heldouts;
maximum incremental error is 6.081 microseconds against the declared
110-microsecond component-impact criterion. All 223 relevant region and oracle
tests pass with the production replay bootstrap. Source evidence is under
`codex_regions/prepare_history_2m_v1/`; the larger source-only KV pool does not
change the target's allocation settings.

A separate structural probe at the same C8 decode contexts prices every GDN
and other body operator, and the head, but refuses all 16 unified MHA calls:
`work_waves=159.3` exceeds measured support 128.725. Its block/state assignment
is explicitly reconstructed, not observed. The independent MHA source-domain
extension remains required before another C8 replay. No positive E2E gate has
been promoted by these source checks.


**Qualified registry and MHA diagnosis (2026-09-14).** Commit `e8b94244`
selects the qualified GEMM repairs and the measured 2M preparation profile.
All 8,960 ordinary GEMM lookups pass, all 931 legacy exact measurements retain
precedence, and all 85 repair records are selected at their intended widths.
The source-quality qualifications remain explicit. The successor calibration
registry (`codex_decode_domain_v1/calibration_registry_v7_e8b94244/`) passes
all 24 provenance checks and retains all 4,994 historical index records.

The TP1 MHA extension completed five independent heldouts. Four meet the
frozen 10% component criterion; the reordered 32-request case misses it at
**+14.714%** (7.350762 ms predicted versus 6.407904 ms measured), with repeat
ranges below 0.30%. The failed validation is preserved in
`codex_regions/mha_decode_2m_v1/VALIDATION.json` (SHA256
`1a52017b0e08f1a4bab294f1de2589314d1057e967ffa1d2c9ad79f9295e92f7`).
That failed fit remains unselected. A CPU reproduction
refits only the 33 training books and reproduces the miss in about 2.6 seconds.
The subsequent hardware-ID probe observed 16 dispatch groups with five active
CUs each. The successor descriptor models scheduling within each group and
was fitted only to the original training books. All nine fresh TP1/TP2
confirmation cases passed; those confirmation timings and the instrumented
probe never entered fitting. Commit `0ffcc4f6` adds the explicitly scoped
descriptor, and `2feea392` selects the confirmed TP1 sources used by the three
paired TP1 cells above. `8a637eff` selects the verified per-rank TP2 sources
used by the first passing TP2 aggregate/memory comparison. TP4 retains source
residuals near 20% and 17%; its separate
`head_1` investigation remains open. These component results do not establish
the remaining serving cells or any matrix/ranking gate.

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
capture and replay, chunked prefill. **Current main-path cache policy:
prefix caching enabled on both engines**, with explicit checkpoint and initial
cache-state semantics. The original registration used
`--no-enable_prefix_caching` on both sides; its results remain historical
cache-disabled rows. Cache-on validation needs fresh matched evidence and a
separate configuration/workload identity, without relabelling those rows.

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
* `agent_scratch/cc_pilot.jsonl` (`bf4049f84be161df`) is a **development and
  regression workload**. The cost model was iterated against it; §3 already says
  so about E1. It is never held out on workload. Two facts about the file that
  its one-line description used to hide:
  * It holds **62 requests**, every one of session
    `002001296e8a8c38ad9d7cc436d691afc602` (the file abbreviates it to
    `002001296e8a`), which is excluded from both frozen cc-traces workloads
    — disjointness is by session identity, not by sampling. E1 sent only its
    **first 20**; the current development replays send all 62. "First 20" is
    a description of an historical run, not of the file, and where it appears
    below it means the run.
  * Its arrival column is **file time, not raw corpus time**. 25 of the 61 gaps
    were clipped to 30 s when the pilot was cut, compressing a raw source span of
    88 934 s into a file span of 1 098.9 s. Replays pace to the file, so a
    development run reproduces the clipped pacing and not the corpus's real idle
    structure. Nothing in the final protocol or the two frozen workloads depends
    on this file, and none of them is changed by saying it.
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
workload and the first-20 `cc_pilot` long workload); it, its lock and its stamped
results are preserved as they stand and are not relabelled. The acceptance
registration is a **separate, new pair** — `atom/compass/CC_TRACES_PROTOCOL.md`
and `atom/compass/cc_traces_protocol.lock.json` — in preparation on the
`compass/cc-acceptance` task branch. **Until that pair exists here and is
stamped, no row in §1 may be moved on cc-traces evidence, and no workload is
locked.** Two drafting decisions are recorded as open at the time of writing:
within-session request timing is taken raw rather than clipped at a 60 s idle
ceiling, and short-session start alignment is declared explicitly before
registration rather than by assigning an arbitrary 1 s arrival.

A **provisional** selection exists as of 2026-09-11 and is recorded here so that
CPU-side coverage work has something real to aim at, not because it is settled.
Long: session `0470d446a4514dfe0c6ad0be92853bd13287`, window index 6, 20
requests, 1 594 624 input tokens (median 91 008, max 107 328), 15 833 output
tokens, native span 262.828 s. Short: 64 source session openings with declared
start alignment at 0, 36 480 input / 1 377 output, source lengths unmodified.
**Neither is locked.** CC is auditing whether nested subagent requests inside a
selected window belong in it — a wrapper event is not a request, but its
children can be — and that audit can change both manifests. Nothing derived from
these numbers is gate evidence, and no §1 row moves on them.

**`cc_pilot.jsonl` is 62 requests, not 20, and its arrivals are clipped.**
Audited 2026-09-12 against the corpus itself (`traces.jsonl`, sha256
`e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`, the digest
both acceptance manifests record). The file at digest `bf4049f84be161df` holds
**62** rows, and they are all 62 top-level servable requests of corpus session
`002001296e8a8c38ad9d7cc436d691afc602` -- matched positionally on `in`, `out`,
`api_time` and `ttft`, not on token counts alone, with that 12-character session
prefix unique among the corpus's 393 sessions. That session is named in
`development_sessions_excluded` and `sessions_rejected` in **both** acceptance
manifests and appears in neither `sessions` list, so the development workload is
disjoint from the frozen long and short workloads for all 62 requests, at
whole-session granularity rather than request by request.

Two numbers differ from the frozen long workload by more than scale:

* **Volume.** All 62: 9 510 208 input / 83 702 output tokens. The first 20:
  1 681 024 input, which is the "1.68M development slice" the long rule's
  `volume_band` was sized against. Frozen long is 20 requests / 1 594 624 input
  on raw source arrivals.
* **Arrivals.** 25 of the 61 inter-arrival gaps -- exactly those above 30 s in
  the source -- were clipped to 30.000 s when the file was written, collapsing a
  source span of 88 934 s to 1 098.9 s (largest single clip 63 372.5 s -> 30 s).
  The replay preserves the **file's** timestamps faithfully; the file does not
  preserve the corpus's intervals. The frozen workloads declare the opposite
  (`"raw source intervals, nothing clipped or compressed"`), so development and
  acceptance runs do not share an arrival process. On the long rule's own
  `max_window_span_s` of 900 s, the 62-request window would not be eligible.

The digest does not say which slice a run consumed; the replay manifest's
`requests` field does. Both exist in the record under the same digest:
`poc/evidence/cc27_20260910/big_real.json` and
`poc/matrix/tp1_long_gatea/real.json` replayed **20**;
`memval/tp1ref/out/real.r1.json` and `diag_tp1_replay3.json` replayed **62**.
The "first 20 requests" wording elsewhere in this file, in
`CC_TRACES_PROTOCOL.md` and in `cc_traces_workload.py` is therefore a correct
historical reference to the slice those earlier runs took, and the statistics
recorded with it (min 640, median 84 800, max 119 360, sum 1 681 024) are the
first-20 statistics. Over all 62 the range is 448 to 249 344 input tokens. None
of this is a reason to alter the registered acceptance scope, and no earlier
result is relabelled by it: the qualification is that a cc_pilot figure must
carry its request count, and that no cc_pilot run -- at either count -- is
acceptance evidence.

Three properties qualify the historical replay evidence and its cache-on
successor (`agent_scratch/cctraces.py`). Historical replay dropped prefix
reuse: the trace's `hash_ids` carry 64-token block sharing, while its synthetic
prompts shared no prefix and caching was off. The new codec preserves declared
local hash-block prefix identity using synthetic token IDs; it does not recover
original text or establish generated-output reuse. `in` is a
**block-derived token count**, not a tokenizer
count — a CC source audit on 2026-09-11 established that it is already in token
units, equal to `len(hash_ids) × block_size` across 28 444 top-level real
request rows, so it is a true length quantised up to a 64-token boundary rather
than a count of blocks awaiting multiplication. An earlier sentence here called
it a block count; that was wrong, and any conversion that multiplied it would
have been 64× too long. It stays accurate in distribution and approximate per
request, because the quantisation is real; and recorded arrivals are
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

The table retains the original cache-disabled registration's results and
**9/24** count. Cache-enabled main-path gate evidence is still pending; none
of these historical passes is automatically transferred to that configuration.

| # | Gate | Bar | Status | Evidence |
| --- | --- | --- | --- | --- |
| **G1** | Correct feasible-configuration selection | top-1 matches hardware | **UNPROVEN** | Nine positive cells pass current aggregate checks; 15 registered cells and cross-configuration selection remain unestablished; whole-corpus confirmation is a separate requirement |
| **G1b** | Ranking correlation within comparable groups | Spearman ρ ≥ 0.90 | **UNPROVEN** | — |
| **G1c** | Ties reported honestly, selection regret reported | stated, not assumed | **UNPROVEN** | — |
| **G2a** | Throughput error | ≤ 10% | **PARTIAL — 9/24 aggregate checks** | The nine current errors range from **−6.776582% to −2.363219%**; the checkpoint table gives every cell. Shared-node timing qualification applies. Historical E2a, 27B short-input: **+27.3% / +37.0% / +85.9%** at TP=1/2/4. Measured, not missing — the modelled run finishes the same 64 requests in 27.5/16.3/8.5 s against 35.1/22.3/15.9 s real. Those failed diagnostic predictions remain preserved |
| **G2b** | TPOT / ITL error | ≤ 10% | **PARTIAL — 9/24 median checks; tail gap open** | The nine current medians range from **+3.270906% to +9.147776%**; the checkpoint table gives every cell. Original short-c2/c4/c8 p90 and per-request discrepancies remain separate open limitations. Shared-node timing qualification applies. Historical E2a short-input median **−49.6% / −45.9% / −39.7%** at TP=1/2/4, and the modelled TPOT is near-constant across requests (p90 ≈ median) where the real one is not. E1 long-input passes at TP=4 |
| **G2c** | TTFT error | ≤ 15% | **PARTIAL — 9/24 median checks; tail gap open** | The nine current medians range from **−1.069261% to +12.660296%**; the checkpoint table gives every cell. Short-c2 p90 is **+30.164484%**; tail/admission and per-request order discrepancies remain open. Shared-node timing qualification applies. Historical E2a short-input median **−10.4% / −20.5% / −46.1%** at TP=1/2/4 — within bar at TP=1 only. E1 long-input at TP=4: mean −0.14%, p90 −0.27%, median −6.4%, on the workload the model was developed against |
| **G3a** | Non-KV memory terms | each within 10% | **PARTIAL — 9/24 paired cells** | All eight components are compared on every rank/repeat. Maximum error is **0.035461% at TP1**, **7.859768% at TP2** (non-Torch). Historical E3b: 27B at TP=1/2/4. weights +1.6/−0.1/−3.3%, non-torch +4.9/+1.2/+1.3%, load residue −3.3/−6.7% at TP≥2. Terms outside: load residue at TP=1 (−93%, 0.01% of budget), persistent (−51%, 0.07% of budget), activations (no prefill-shaped graph for this model). The historical "graph pool +0.0% everywhere" claim remains retracted: the independent TP4 comparison was **+26.8%**, outside the bar. TP1/TP2 results do not resolve TP4 |
| **G3b** | KV block count | within 5% | **PARTIAL — 9/24 paired cells** | TP1: **112,772 real / 112,773 modelled**, error **0.000887%**. Both TP2 ranks: **265,540 real / 266,768 modelled**, error **0.462454%**. Historical E3: −0.09% (27B TP=2), −0.02/−0.02/+0.07% (0.6B TP=1/2/4); its 27B ground truth was 112,740 / 265,520 / 584,880 blocks. Remaining cc-traces configurations still require matched evidence |
| **G3c** | One infeasible configuration rejected for the right reason | same error as the engine | **PASS — negative cc-traces witness** | TP1, utilization 0.34; the same 83,968-token registered request is rejected for insufficient KV capacity on both servers; frozen prediction and retained attempts documented above |
| **G4** | Prediction outside the calibration configurations | stated per prediction | **PARTIAL — first TP2 E2E transfer cell passes current checks** | `tp2_clients_short_c1` passes three paired aggregate/memory comparisons with no target timing used for fitting; remaining TP2/TP4 cells are unproven. Historical diagnostics, both `ModelRunner.forward` decode steps frozen before measurement with no target timing among their inputs: **E7, TP2, 20.970 ms frozen against 19.845 ms, +5.7%** — within 10% (`G4_TRANSFER.md` §12). **E8, TP4, 16.157 ms frozen against 12.807 ms, +26.2% — outside 10%, a failed prediction** (§13). E8's miss is localised to one input: rank 1's frozen body price is 17.1% above what a later repeat measurement of the same thing produced, and the other three ranks predicted +5.1 to +5.5%. **A repeat that does not reproduce a value establishes that the value is unstable; it does not establish which of the two is right, nor what made them differ.** No corrected number follows from it either way — the frozen prediction stands as it was frozen. Both share the shape `bucket=32, cohort=32, tokens_each=1, context=1151`. E6 is their coverage precondition — a priced body at TP=1 (23.122 ms, 2423/2439 operators) and TP=2 (15.467 ms, 2552/2568) |
| **G5a** | Replay speedup, **after capture** | ≥ 5× | **FAIL against 5×; advisory** | The nine current paired ratios span **0.61627× to 2.53304×**, including derivation not already counted in execution; the checkpoint table gives every cell. The target is advisory under the user policy above, and shared-node timings remain advisory. Full acquisition amortisation is separate (G5c). Historical device-free 27B diagnostics reported 36 s → 1.46 s (**24.7×**) and 133 s → 23 s (5.8×) including startup; they are not these cc-traces results |
| **G5b** | GPU-free replay after capture | no device | **PASS at 0.6B and 27B; confirmed on nine paired cells** | All 27 modelled repeats of the nine paired cells pass device-free evidence checks; all eight original TP1 paired cells are complete. Historical E5: served 32/32 (0.6B) and 64/64 (27B) in `xiaobizh_n18_cpu`, a container with **no `/dev/kfd` and no `/dev/dri`** — zero driver handles and no KFD process registration in any process of the tree. Both reproduce the GPU-resident simulator's schedule step for step and its TTFT/TPOT/latency distributions exactly; at 27B ten of 64 requests sit in a different slot of that same schedule, which is the burst's admission order, not the GPU-free path (E5). This is the no-device gate only; G5a and G5c are separate |
| **G5c** | Capture / calibration / startup / load / execution costs reported separately, with amortisation | reported | **PARTIAL — acquisition durations unknown** | Current paired startup/execution windows are measured and derivation is assigned to its containing windows. Full historical capture/calibration totals and a separate load duration remain unknown; amortised ratio and break-even remain unproven |

**Nine positive cells pass current aggregate checks; the full PoC remains
open.** Their median/throughput and memory evidence is matched, subject to
shared-node qualification. Original tail, per-request and admission/order
discrepancies remain open. G3c retains its negative witness and G5b its no-device
proof. The remaining 15 registered cells, configuration selection, ranking and
broader transfer are not established. Whole-corpus confidence follows the
[coverage review](CC_TRACES_COVERAGE_REVIEW.md), not these cell counts. The first TP2 transfer cell passes; its 7.859768%
non-Torch residual remains reported without recalibration. The 5× target is
missed and advisory, and acquisition costs/amortisation remain incomplete.
The dated diagnoses below preserve earlier evidence and failures.

**The graph-pool memory term, flagged 2026-09-11, not yet integrated.** E3b
reported graph pool at +0.0% at every width, and an independent comparison being
built on the `compass/memory-validation` task branch reports **+26.8% at TP4**.
The two disagree, one of them is wrong, and until the comparison is integrated
here with its source and import hashes there is no basis in this file for
choosing. The G3a row therefore no longer claims that term. It is recorded now,
before integration, because a retracted claim that sits unmarked until the
correction lands is the same failure this document was written to prevent —
`RETROSPECTIVE.md` has four instances of it. The provenance is explicit: branch
`compass/memory-validation`, audit in progress, no ready hash reported, nothing
from it merged. **G3a stays PARTIAL and G3 stays open.**

**G4 was moved to PARTIAL on 2026-09-11 morning and back to UNPROVEN the same
afternoon.** Two things moved it back, and both matter. The first is the
acceptance registration above: G4 is an end-to-end quantity, and a step-level
check — passing or failing — cannot carry it. The second is E8. The TP4 step,
frozen at 07:22:04Z and measured an hour later, came in at **+26.2%, outside the
criterion**. The morning's PARTIAL rested on E7 alone; by evening the same method
at the next width had produced a miss five times larger, and a row that reads
PARTIAL on the strength of one of two diagnostics while the other fails is a row
that flatters itself.

**E8 is preserved as a failure, not as a pending item.** One of its
twenty-three inputs is now known to be unstable — a later repeat of the same
measurement came out 17.1% lower — and *that is the whole of what is known*. The
cause is not established: an unreproduced measurement says the quantity does not
hold still, not which value is the right one and not what made them differ.
Whatever the explanation turns out to be, it cannot make the prediction right
after the fact. No corrected TP4 number may be computed against that capture by
anyone, because the capture has now been seen. A second TP4 transfer claim needs
a fresh freeze over inputs measured under a declared stability method, and a
fresh capture.

What E7 and E8 do establish jointly, at two points: no target-width step or
serving time entered either prediction, and the composition reproduces itself
byte-for-byte across widths (`tpN_frozen.py` at TP=2 regenerates
`tp2_frozen.txt` exactly, which is the precondition that licensed the TP4 run).
What they do not establish is any accuracy gate in this matrix (G2a/G2b/G2c) or
any ranking gate (G1/G1b/G1c): those are measured over a whole serving run, and
none of them moves on either experiment.

**Method gap E8 exposed, now open as item 10 in §7.** Nothing in the freeze path
established that a price holds still before freezing it. `PriceLibrary` already
has a mechanism — a 5% conflict band, `library.py:373` — and it would have
flagged rank 1 had the body been priced twice. The requirement that follows has
to be stated carefully, because the obvious version of it is a hole:

* **Predeclared.** The repeat count, the statistic taken over the repeats and
  the band a spread must fall inside are fixed *before* the measurements, in the
  freeze script, and recorded with the prices. A price whose spread exceeds the
  band is a refusal to freeze, not a choice between values.
* **Never residual-driven.** A price may not be re-measured, re-selected or
  preferred because it moves the prediction toward a target observation. The
  repeat that followed E8 was run after the comparison, which is exactly why it
  is filed as a diagnostic and cannot be substituted into the frozen input set —
  and why "the repeat agrees with the measurement better" is not an argument
  that would have been allowed to pick it.
* **Cheap relative to what it protects.** Re-pricing the body cost about five
  minutes at TP4; the capture it would have spared cost four cards for the
  better part of an hour.

Improved primitive measurement under such a method is available to the fresh
end-to-end cc-traces checks. It does not rewrite E8, whose numbers stand.

---

## 2. Experiment register

Each experiment is defined once, here, so that a gate row names a procedure
rather than a run.

### E1 — long-input serving, 27B, TP=4 (the cc-traces pilot; closed)

* **Workload** the first 20 rows of `agent_scratch/cc_pilot.jsonl`, a 62-row
  single-session development file cut from
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
  directory (`dec32/repeat_2026-09-11/`, rc=0, 07:33:52Z) did **not** reproduce
  that +42%: rank 1 came out at **13.106 ms, 17.1% below the frozen input**,
  with the other three ranks within 0.5%. **What that establishes is that the
  quantity is unstable at rank 1 — not that either value is the correct one, and
  not what made them differ.** Naming a cause would need a third measurement
  under a declared method, which has not been taken. On the evidence that does
  exist, the transfer method was neither refuted at this width nor confirmed.
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
| **E8 frozen TP4 decode step** (`G4_TRANSFER` §13) | **no** — frozen at 07:22:04Z with twenty-three input hashes, captured afterwards; no TP4 step or serving time among them | **a genuine prediction, on one forward step, and it missed.** +26.2% against a 10% criterion. Same held-out axes as E7 (width for the region term, target timings entirely). Localised to one input later shown to be unstable, which neither explains it nor un-fails it |

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

Re-ranked 2026-09-15 for the cache-enabled main path. Historical cache-disabled
results remain evidence about their registered configuration, not completion
of these successor tasks.

| # | item | blocks |
| --- | --- | --- |
| 1 | Complete combined CPU verification and registration of cache-aware prompt identity, policy, native fork binding, quiescent reset and readiness-governed demand | every cache-on gate |
| 2 | Independent cache-on source/oracle preflight: low checkpoint-cut shapes, resume/fork work, region scope and actual native memory terms; retain the tiny TTFT/non-Torch failures | functionality, G2 and G3 |
| 3 | Fresh intact exposed-root TP1 pair with actual admitted cache hits, checkpoint/pool evidence and all requests/arrivals preserved | first cache-on E2E accuracy evidence |
| 4 | Cache-aware short/long × C{1,2,4,8} × TP{1,2,4} paired matrix, with clients kept distinct from in-flight requests | G1, G1b, G1c, G2 and G4; historical 9/24 remains unchanged |
| 5 | Per-term memory/KV checks and native infeasible rejection at the same cache-on deployments, plus ranking/ties/regret evidence | G1 and G3 |
| 6 | Freeze a probability design across exposed and potentially untouched roots; preserve cluster dependence and an inconclusive stop | whole-corpus and 99% confidence claim, currently unproved |
| 7 | Record execution, derivation and acquisition/amortisation costs; keep a missed 5× target explicit and advisory | G5a and G5c |
| 8 | Keep source statistics, repeat counts and bands fixed before measurement; refuse uncovered work and never fit evaluated cc-trace timings | credibility of every prediction |

**Closed since the last ranking.** Item 5 of the old list — "predict TP=2/4 from
a TP=1 capture" — is done as a *diagnostic* and is not coming back in that form:
E7 answered it at TP2 (+5.7%) and E8 answered it at TP4 (+26.2%, failed). The
gate version of the question is item 3, over a serving run. Old item 8 —
"promote the §5 validity checks and test them at the client boundary" — is done;
§5's table above names the check that enforces each row and
`tests/compass/test_run_validity.py` covers them. Two residual weaknesses in
that work are stated at the end of §5 rather than left implicit here.
