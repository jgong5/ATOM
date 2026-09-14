# cc-traces coverage and confirmation review

This review separates the current registered burst suite from a claim about the
whole pinned cc-traces corpus. **The present evidence does not establish
full-corpus accuracy or 99% statistical confidence.** The audit identified two
functionality limits: the readiness profile refused small native requests, and
the replay transport could not submit some intact roots. Their remediation
status is recorded below; the original findings and evidence are preserved.
Coverage also needs sustained and multi-episode continuity, since success on
isolated source busy episodes does not imply success on their concatenation.

This is a coverage review and a proposed confirmation design, not a replacement
registration. Existing workload hashes, run registrations and results retain
their historical meaning. The initial audit performed no GPU run, fit or
workload rewrite; subsequent execution results are dated below.

## Current main path — 2026-09-15

**Prefix caching enabled is now the main cc-traces validation path.** The
original cache-disabled suite stays at **9/24** passing paired cells, including
8/8 original TP1. Those historical results do not establish cache-on accuracy,
whole-corpus generalization or **99% confidence**. The first bounded cache-on
surrogate pair is complete; it adds no registered acceptance cell.

**First cache-on paired milestone: the two-request `4b433` token surrogate.**
The fresh pair on `aa8bd80e2` completed 2/2 requests and 366 output tokens per
side, preserving source requests 919/920, 92,928/92,672 input lengths,
12/354 output lengths and the original 114.735-second gap. Independent review
reconstructed both prompt digests from the pinned codec and checked request,
clock, process and GPU-free identities. TTFT median **+8.59%**, TPOT median
**+4.92%** and throughput **−1.88%** meet the numerical reference bars.

Both sides reproduce the prefill query sequence, a 32,768-token cache hit
against 38,400 wanted, 13 retained checkpoints and no evictions. Empty-cache
reset receipts and cleanup pass. All five memory budget terms and eight
non-KV components pass; KV counts are 112,760 real / 112,773 modelled,
**0.01153%** error. The physical pool remains 112,760 KV blocks plus 32 state
slots and 121,671,450,624 bytes. These observations do not establish pool
pressure or full-root cache continuity.

The real harness's original **exit 1** is preserved: the 93-sample isolation
audit says GPU1 was owned exclusively but the node was not quiet because of
unrelated PID1685833. Timings remain advisory. The raw replay wall-window
ratio is 155.844406/28.253012 = **5.516×**, excluding preparation/drain and
post-run reporting; it is not a full-cost or amortized speedup. No
`costs.real.json` was emitted after the failed lifecycle, so the normal cost
merge remains unavailable.

The original registry `5f04fa7e…` failed the unchanged calibration check because
it omitted the aggregate digest for the actual 413-file price bundle, despite
declaring every member individually. Post-run metadata registry `42329052…`
adds exactly that bundle (`3c6ccf75…`): the prior 401 members plus 12 finite-GEMM
and q16 graph/price files. All 452 prior registry entries, source bytes and
coefficients are unchanged. The same calibration checks pass against the
labelled successor; the original refusal remains recorded. The actual wrapper,
overlay, q16 scope and selected region snapshot also pass explicit source
checks, while the **FAILED outputless** source remains diagnostic-only.
Closeout `PAIRED_CLOSEOUT_V3_POSTRUN_METADATA.json` (`8f871739…`) and independent
`INDEPENDENT_REVIEW_V1.json` (`cfb20e7f…`), under
`agent_scratch/codex_cache_pair_4b433_v1/` in the artifact container, retain
both outcomes. This is one bounded surrogate pair,
not faithful/default AIPerf replay, statistical confirmation or new matrix credit.

**The actual published replay adds role and release semantics.** AIPerf defaults
to synthetic assistant history, so original text is not an additional blocker.
Its reconstructed chat tokens and response-gated continuations differ from our
completion/open-loop mechanism diagnostics. The seeded `4b433` chat inputs are
93,986/92,701 with native remainders 2/13; final-16 surrogate coverage does not
prove these shapes. The next faithful candidate is the complete opening of root
`72d021…`: inputs 38,240/39,982, output maxima 338/186, no prior cached history
and second release `max(run origin + 21.437 seconds, first target response)`.
Later branches are excluded. The maintained opening adapter and wrapper checks
are reviewed and integrated (`4078a5bae`, `9950bb532`). All 524 inventoried
decode shapes are covered; the bounded q16 primitive source is qualified.
The remaining primitive dependencies have now been measured: source v3
completed **249 references and 480 heldouts**, with predictions sealed before
heldout release. Every numerical/kernel gate passes, while five reference and
five heldout repeat-spread failures leave the full domain **unqualified**.
The strict opening dependencies have 22/22 qualified references and 82/83
qualified controls; the required `heldout_gdn_q1_l62_fork2_3` control has
**5.21917495%** spread against 5%, although its prediction error is 0.032394%.
No dependency is removed and no threshold is relaxed. A separately reviewed
diagnostic bundle must retain this failure through explicit
`low_q_allow_failed_spread` and `diagnostic_only` selection.

Final-region transfer remains **FAILED**, with 21/24 checks passing and all
q16 controls/postprocess checks passing. The q8/q9/q15 preparation misses at
history 33,792 are 145–160 µs, about 0.13–0.15% of their native forwards;
the unchanged 110 µs verdict is retained. The explicit diagnostic q1–15
selection reuses the frozen q16 formula only over cached history
`[33792,66560]`, with no fit or default/acceptance activation. The completed
region collection was not repeated in v3.

The source verdict is `e00b20ad…`, closeout `da8817a4…`, and final-region
result `108a18a2…`; [PoC status](POC_STATUS.md) records their full paths and
digests. Final provider input closure and aggregate bundle registration are
still required before the next modelled-first opening. No faithful opening
pair, new accepted cell or confidence claim follows from these acquisitions.
[AIPerf replay alignment](AIPERF_REPLAY_ALIGNMENT.md) pins the valid
local-file fixed profile, exact token evidence and its branch-planner limitation;
AgentX gap compression is a separate profile, not an implicit replacement.

Implementation is integrated at `2782333ab`: explicit TP1 cache/checkpoint
policy; a pinned codec for local 64-token source hash blocks with native
16-token prefix-match safeguards; distinct native GDN fork source/destination
slots; a quiescent reset with all-worker completion and fresh modelled epoch
checks; and readiness-gated prefix demand. The current target policy is cache
on, checkpoint interval 8192, demand enabled and one-token GDN forks. Legacy
plans stay cache off. Source memory geometry remains an explicit derived
candidate with the bounded native memory/hit/checkpoint validation above.
Pool-pressure behavior, remaining region coverage and faithful paired E2E
accuracy still require evidence. Combined
CPU verification passed **648 tests**, with zero failures, errors or skips on
exact source `2782333abba2135050072f5966f41c79472c2142` in a container without GPU
device nodes. This cannot substitute for those observations.

Cache-on diagnostic validation explicitly uses
`check_engine(..., expected_cache_policy=...)`, `check_cache_policy_evidence`
and `check_memory_terms(..., expected_cache_policy=...)`. The registered
`cell` checker retains its historical cache-off contract. Final snapshot
policy binding is not an automatic verdict on cache-hit or pool-pressure
semantics; those observations must be assessed in the experiment evidence.

| Latest historical cache-disabled diagnostic | Outcome |
| --- | --- |
| Near-limit 250,048 / 39 | Fresh pair completed 1/1 on each side; TTFT +5.70%, TPOT +4.30%, throughput −5.38%; maintained provenance/capacity/memory checks pass. One diagnostic, not a new registered cell. |
| Tiny 128 / 16, finite gather overlay | Pair completed, but TTFT −46.73% fails 15%. TPOT +5.65% and throughput +8.16% pass their 10% bars. Non-Torch memory fails: 2,046,820,352 real versus 1,157,627,904 modelled bytes. KV 111,930 versus 112,773 differs 0.753%; that does not remove the component failure. |
| Long decode 195,840 / 40,339 | Modelled run completed with all 13 maintained checks. Prediction: TTFT 133.296848 s, TPOT 0.035998853 s, latency 1,585.418576 s. Staged real run is **HOLD**, with its diagnostic cap-32 warmup plan preserved and no measured-output truncation. No paired accuracy result. |
| Seven-request root `1493faff…` | Modelled run refused 16 unpriced QK-norm occurrences at the opening 448-token prefill. The real side was not launched; further cache-disabled collection/retry is held. |

The non-Torch discrepancy is preserved as a failure with unresolved
attribution. Native sizing/recording uses the current PyTorch device's memory
APIs, not an SMI device index; that specific mapping hypothesis is unsupported.
No cause, corrected constant or cache-on conclusion follows from this result.
The next execution evidence is the response-gated opening above, explicitly
diagnostic with its retained spread/region failures and zero response-delivery
approximation. Qualified full-domain support and broader E2E validation remain
gaps; the completed surrogate pair does not close them.

## Remediation update — 2026-09-14

* **Async submission is integrated and pushed** in feature commit `0cb6c2f7f`.
  It replaces the audited 1,024-thread ceiling while retaining native
  `/v1/completions`, singleton ADD messages and the fatal registration-barrier
  timeout. Thirteen CPU transport tests pass, including 3,551 and 16,913
  registrations before any response, 32 full-context request bodies, native ADD
  byte agreement, cancellation artifacts and an established peer that stops
  reading uploads. A finite network-attempt deadline covers that backpressure.
  Declared replay still needs O(N) sockets and memory for prepared request bytes;
  explicit FD checks and a conservative client memory budget refuse insufficient
  resources. This does not establish unlimited capacity. JSON encoding now
  precedes the pacing epoch, so **fresh paired E2E runs remain required**; old real
  references are not interchangeable. Recorded aiohttp callbacks are pre-write
  events, not wire-completion or target-ingress timestamps.
* **The low-range candidate `6e26d4cb…` passed source-support validation.** It was
  unactivated at the source handoff and was subsequently selected only for the
  corpus diagnostics below, with registry `94ef3d0f…`. No registered acceptance
  cell was changed. Its serialized-byte domain is 2,033–1,050,371.
  All 84 heldout source-component median checks pass the unchanged 100 µs limit
  (maximum error 49.155 µs); all 532 formerly refused requests pass both native
  layout variants, giving 1,064 accepted canaries, and 56 CPU regressions pass.
  These source-median and layout checks do not bound individual source-frame
  tails, establish E2E accuracy, or provide 99% confidence. The pinned handoff and
  candidate identities are added to §7's evidence record.

### Subsequent corpus diagnostics and source work — 2026-09-14

Readiness-byte support did not establish native region support. The two initial cache-disabled
modelled attempts exposed separate region bounds; neither completed its request
or produced a paired accuracy result. The original nine passing paired cells
retain their previous scope and `node_busy` qualification.

| Diagnostic | Observed result and original status |
| --- | --- |
| Tiny 128-token episode | Compiled TP1 output-producing prefill `[128]`, context `[128]`, refused the selected N1 span `[640,16384]`. Wrapper v1 recorded the wrong reconstructed source path; actual `--trace` and separate workload/manifest pins survive. Its owned replay was stopped after the exception; journal remains `exit=-15`, `ok=false`, `refused=false`. This is a domain witness with a provenance defect, not a valid E2E pair. |
| Near-limit 250,048/39 request | Wrapper v2 has the actual source identity and passing provenance checks. It priced 15 prefills of 16,384 tokens plus the 4,288-token tail, then refused the first decode at history 250,049 against `[128,196608]`. The owned replay was stopped; its journal also remains `exit=-15`, `ok=false`, `refused=false` (wrapper exit 1). No completed 39-token response or accuracy claim. |
| Complete seven-request root `1493faff…` | Held by explicit preflight: its opening 448-token prefill is below the selected N1 lower bound. No execution occurred. |
| Sustained arrivals, episode 284 of root `509ad65c…` | First fresh TP1 diagnostic pair on `b27d0e7f2`: 21/21 requests and 15,979 output tokens per side. One root client reached 21 in-flight requests and native/modelled scheduled batch 21; source peak overlap was five. Uses the existing `history-2m` preset, not the subsequent high-history extension. |
| Other three emitted cases | Long decode, fanout and opening continuity remain separate development cases; no completed paired result is claimed here. |

The sustained pair's TTFT median/mean/p90 errors are **−3.38/+7.30/−0.12%**;
TPOT errors are **+1.84/+10.74/+8.18%**; throughput error is **−7.12%**.
Individual absolute TTFT/TPOT errors reach 32.31/38.23%; only 6/21 requests
satisfy both reference bars. All eight non-KV components pass 10% (maximum
1.075%), and KV block error is 0.01153% (112,760 real / 112,773 modelled).
Completeness, preparation drain, pair identity, calibration and GPU-free
provenance checks pass. Measured wall speedup is 1.963×, advisory. The native
sampler is `own_clean=true`, `node_busy`; timings remain advisory. GPU1 and
ports were released, with one harmless resource-tracker zombie under PID1
retained in the cleanup record. This is one paired intact-episode diagnostic
under warm/empty replay semantics, not full-root continuity, a three-repeat
registered cell or statistical confirmation. The original count stays **9/24**.

Both attempted CPU lanes are closed. The maintained diagnostic entry point and
owned-refusal watcher are now integrated in `60dc92c84`; their exit-5 model-refusal
classification applies to future executions and does not rewrite those old
wrapper journals. All seven manifests are unchanged and use exposed roots;
the exposure inventory remains **249 exposed / at most 144 potentially untouched**.

The source audit recovered cold 128/256/512 and N2-total 512/1,536 captures for
review, not automatic selection. The existing “2M history” source extended
summed history for selected 16/32-request cells; it did not raise the 196,608
per-request bound. The disjoint-interval representation is integrated in
`b27d0e7f2`, with 11 legacy snapshots unchanged and 125 CPU tests passing.
The high-history source acquisition completed: all **56/56** frozen heldouts
pass the fixed 110 µs preparation-increment criterion, maximum **12.560 µs**.
Separate TP1 preset `source-27b-tp1-history-256k` is integrated in **`4b109a968`**,
with per-row support through 262,143, all 11 old snapshots and 411 old
breakdown/band comparisons unchanged, and 209 tests passing. The later
near-limit pair on this preset completed with the passing diagnostic results
above; neither source validation nor this pair adds a registered acceptance cell.

Low-prefill source preparation **failed** the unchanged 110 µs criterion:
17/30 pass, 13 fail, maximum **427.143 µs**. Postprocess 14/14 and structural
zeros 16/16 pass. TP1 candidate `e43976eb` is explicitly **diagnostic-only**,
using already-frozen anchors without default/acceptance activation or N2/pool
extension. Its tiny follow-on loaded correctly, then exited 5 on missing
head-row gather price `aten::index.Tensor|128,5120;1|bfloat16,int32|1:127`
(2,442/2,443 operators priced). That pre-overlay attempt completed no request;
the later finite-overlay tiny pair failed TTFT/non-Torch as recorded above.
The original low-source criterion remains failed. TP2/TP4 work remains paused.

## 1. Population and meaning of replay

The initial population is `semianalysisai/cc-traces-weka-062126-256k`, exactly
`traces.jsonl`, 568,864,747 bytes, SHA-256
`e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`.
The [filtered dataset card][filtered-card] and its pinned repository tree agree
with this artifact. It has 393 root sessions, 28,444 root API requests, 39,822
descendant API requests, and 1,697 subagent wrappers. Wrappers are ancestry, not
additional API requests.

| Population | API requests | Input proxy tokens | Output tokens | Published cap |
| --- | ---: | ---: | ---: | --- |
| Pinned filtered corpus | 68,266 | 6,891,228,864 | 58,728,807 | Per request, input + output ≤ **256,000** |
| [Parent release][parent-card] | 98,827 | 21,635,381,376 | 106,474,498 | Input ≤ 990,016 |

The decimal 256,000 filter is distinct from the deployment's 262,144-token
maximum. It applies independently to root and nested requests. A partially
filtered wrapper keeps its surviving children, and an empty wrapper disappears.
Survivor timestamps retain their relative offsets, with one common origin shift
if necessary. An intact episode in this review is therefore intact **within the
filtered survivor data**, not necessarily within the parent session.

The parent card explains that `in` is the number of 64-token KV hash blocks
times 64, **not a true tokenizer count**; heavy cache-write tails can overcount
the real prompt by approximately 260k tokens. The selector's
`in == len(hash_ids) * block_size` check establishes proxy consistency. Exact
synthetic-token replay reproduces that proxy length, not original prompt content
or true billed token length. The parent release is itself filtered and must not
be called raw, unfiltered production traffic. Both cards describe v7, CLI-version,
minimum-request, subagent-overlap, image, classifier, deduplication and dynamic
workflow filters; the published invocation uses `--sampling top`. No random
sample of all Claude Code activity is established.

The historical evidence is conditional on Qwen/Qwen3.8-27B, TP ∈ {1,2,4},
its declared memory/scheduler settings, synthetic content, **prefix caching
disabled**, fixed source output counts and open-loop arrivals. The current
target semantics instead enable prefix caching and preserve declared local
source hash-block relationships through a separately pinned synthetic-token
codec. That reconstructs prefix identity, not original text or true billed
token lengths; generated-output reuse is not inferred from output counts.
Fresh cache-on runs must bind that encoding, checkpoint policy and matching
initial cache state on both engines. Historical cache-off results cannot be
relabelled as evidence for it.
All source model labels map to the same Qwen target. Root-relative arrival
intervals are available; cross-root wall-clock chronology and a causal
parent/tool-completion DAG are not. C ∈ {1,2,4,8} is the number of
root clients selected for a scenario. It does not cap requests in flight:
descendants and the target's service times can produce more than C outstanding
requests. Zero-aligning roots is a declared scenario, not recovered chronology.

## 2. What the selected suite represents

The census distinguishes three denominators:

* 68,266 published API leaves across 393 roots.
* 68,109 individually selector-eligible leaves across 392 roots after excluding
  28 zero-output leaves and the explicit pilot root's 129 leaves. Here
  “non-pilot” is more precise than “untouched”: many of these roots were later
  exposed during development.
* 67,716 requests in 40,225 complete eligible source busy episodes. Rejecting a
  whole episode with an ineligible leaf also excludes eligible neighbours: the
  28 zero-output leaves occupy 28 rejected episodes across 14 roots; those
  episodes contain 393 otherwise-eligible neighbours across three roots.

The original registered cache-disabled short/large C=1/2/4/8 workloads contain **54 unique requests from 16
roots**. Nested C workloads produce 110 appearances; these are not 110
independent workload observations. The selection is the first qualifying
episode of each of the first eight qualifying roots in corpus order per class.
It is deterministic and reproducible, but not a probability sample. Once a
pool is full, later roots are not examined for selection eligibility, so the
selector's refusal counters are not a full-corpus census.

Both classes require at least one descendant, source peak overlap ≥2 and whole
eligible episodes. The short class requires every prompt ≤4,096. The large
class allows prompts through 262,144, limits the source arrival span to
60 seconds and total input to 400,000, and requires a prompt ≥32,768.
These rules favor compact bursts. The selected short
episodes are descendant sibling bursts from one branch. Across both classes,
50 of 54 unique requests are descendants; the corpus's individually eligible
set has 39,742 descendants and 28,367 root requests.

| Gap relative to selected requests | Eligible requests | Share of requests | Share of input volume | Share of output volume |
| --- | ---: | ---: | ---: | ---: |
| Input > selected maximum 170,368 | 12,287 | 18.04% | 37.51% | 25.23% |
| Output > selected maximum 6,903 | 909 | 1.33% | 2.00% | 17.73% |
| Input <1,024 | 532 | 0.78% | 0.0034% | 0.0625% |

The selected requests themselves account for 0.0793% of eligible requests,
0.0276% of input volume and 0.1081% of output volume. Sparse sampling alone does
not disprove representativeness, but this selection has known structural gaps:
no prompt from 4,097 through 16,384, none above 170,368, no output below 100,
and no selected adjacent root-turn transition. The census records 27,960
adjacent root transitions, including 25,663 growth transitions and 1,739
shrinks; descendant transitions add 32,209 growths and 4,165 shrinks. Shared
prefix hashes describe these histories even though the historical suite disabled
prefix reuse. The cache-on successor must preserve their declared scope and
must not expose future-request demand before logical readiness.

Coarse input/output-length × actor bins represented by at least one selected
request contain 42.41% of eligible request mass and 23.53% of input volume.
Adding source concurrency lowers these to 22.18% and 12.79%. These are generous
descriptor-bin coverage statistics, **not performance guarantees or confidence
levels**; one request does not validate the bin containing it.

| Whole eligible episode regime | Episodes | Requests in those episodes | Current selected episodes |
| --- | ---: | ---: | ---: |
| Single request | 36,771 | 36,771 | 0 |
| Root requests only | 26,729 | 27,790 | 0 |
| More than seven requests | 609 | 22,757 | 0 |
| Arrival span >60 seconds | 332 | 18,724 | 0 |
| Input volume >400,000 | 903 | 23,640 | 0 |
| Source peak overlap >8 | 46 | 7,200 | 0 |
| Mixed short/large input | 1,012 | 14,201 | 1 |

These regimes overlap and must not be added together. Source overlap and live
input are source API proxies, not observations of target concurrency, resident
KV, or preemption. The largest complete eligible episode has 969 requests;
the largest input volume per episode is 66,735,744 proxy tokens.

## 3. Functionality and continuity come first

**The initial small-request support failure is preserved.** The readiness profile
`8317c718…` accepts serialized ADD payloads starting at 5,875 bytes. Native
`Sequence`/`CoreManager.add_request` serialization of all 532 non-pilot eligible
requests below 1,024 tokens produces 2,329–5,668 bytes under the standard replay
layout. A separate descriptor calculation agrees for 1,064 ordinary/full-corpus
ordinal variants; that frozen profile refuses all of them. This affects **340/392 roots (86.7%)**,
including 267 first root turns. Small request mass is therefore not evidence
that the issue is harmless for complete-session support. These are deterministic
native serialization checks, not GPU or HTTP timing observations. A source
measurement or justified source model extension is needed; silently widening a
bound would not provide that evidence. Even among the 378 non-pilot roots with
every leaf selector-servable, 328 contain one of these confirmed refusals.
The later low-byte source extension addresses this serialized-layout bound;
cache-on checkpoint cuts and native region support remain separate checks.

**The audited submission predecessor had a fixed thread ceiling.** Its
`replay.py` pretokenized every prompt and used one blocking thread/HTTP request
per workload row. It rejected workloads over `MAX_IN_FLIGHT=1024`; the limit was
checked after optional preparation and pretokenization. Thirteen published roots
already exceed 1,024 API leaves,
and combining C roots can exceed it sooner. The maximum root has 3,551 API
requests. A 969-request episode is below the nominal limit but is not evidence
that thread, connection, memory or timeout behavior is adequate near the limit.

The modelled server waits for all declared workload rows before advancing
virtual time. Replacing the client with K blocking workers while N>K requests
must register deadlocks behind responses the server cannot yet produce; the
existing comments document this failure. The 120-second barrier fallback makes
that replay invalid, and the readiness path now refuses an incomplete barrier.
The audit therefore required a bulk/streamed registration protocol or another
explicitly proven transport for larger workloads; the async successor above
addresses that requirement within its tested bounds. `--num-requests`, a
client-count semaphore, or a
longer timeout must not silently substitute for an intact workload.
Preserve the measured per-request admission path when changing transport, or
requalify source readiness evidence if message layout or admission work changes.

**Source-idle boundaries do not prove target drain.** G=0 busy episodes are cut
using source `t + api_time`. The Qwen target may still be decoding or backlogged
when the next source episode begins. Restarting every episode from empty removes
cross-episode queue buildup, live KV and mixed prefill/decode. Even small
per-step errors can accumulate near service capacity and change queueing,
batching, memory pressure and TTFT substantially. Episode successes compose
only with a witnessed target drain and equivalent state at every boundary;
the source timestamps alone do not supply that proof. Validation needs intact
multi-episode trajectories and complete roots, with gaps preserved.

## 4. What existing outcomes do and do not prove

The fixed metrics and bars remain those in [CC_TRACES_PROTOCOL.md](CC_TRACES_PROTOCOL.md)
§6: throughput and TPOT ≤10% error, TTFT ≤15%, non-KV memory terms ≤10%, KV block
count ≤5%, matched infeasibility rejection, configuration ranking/top-1 within
comparable workloads. The protocol's ≥5× replay speedup target is
**advisory/compromisable under the user's explicit override**, not a mandatory
acceptance gate. Keep reporting speedup with its stated denominator and costs.
TP1 remains source-configuration residual evidence; TP2/TP4 test the separate
source-only configuration prediction claim. Accuracy and memory bars are unchanged.

`compare.metrics` defines TTFT as first token minus arrival, TPOT as
`(finish-first)/(output-1)` for output ≥2, and throughput as output tokens divided
by the complete arrival-to-final-finish window. `compare._quantiles` uses the
upper order statistic `s[min(n-1,floor(q*n))]`, without interpolation.
`cc_traces_validate._across_repeats` compares median repeat summaries and records
their ranges; for TTFT/TPOT its gated statistic is the median, while p90 remains
a diagnostic. It does not compute a confidence interval. Errors of aggregate
medians/p90s are different quantities from per-request errors. The existing
`per_request_error_pct` quantiles are signed, not absolute.
The validator rejects failed/missing/mismatched requests and barrier/drain
violations before a run counts; an uncertainty calculation must preserve that
validity contract rather than summarize only surviving successful responses.

Retained baseline outcomes illustrate the distinction. In short_c8, only 27/51
repeated TTFT observations were within 15% and 30/51 TPOT observations within
10%, despite an aggregate cell pass; worst absolute TTFT error was 959.1%.
The provisional large_c8 first pair had 32/37 TTFT values within 15%, with
absolute-error p90 39.16% and worst 149.93%. These are correlated observations
from old baseline outcomes, not corrected-profile results or confidence bounds.

Corrected short_c4 and short_c1 diagnostics each compare one new modelled run
against three retained real references, not fresh paired acceptance runs. In
short_c4 the group membership matches the references and aggregate median TTFT
error is about +5.5–5.8%; its first-request TTFT remains about −17.5% to −23.5%.
Source-component validation can support a model correction without converting
this diagnostic into an untouched workload evaluation.

**All current real timing outcomes remain `node_busy`/advisory.** A clean assigned
GPU does not establish an isolated node. These results must not be upgraded into
unqualified isolated-performance evidence, at 99% confidence or otherwise.
Environmental qualification applies to the eventual statistical claim as well
as to individual runs.

### Root exposure

Current selected outcomes informed readiness and fence changes. Their roots
are development data for the corrected model even where the new calibration
inputs were measured independently at source. Source calibration provenance
does not restore workload holdout status. The old protocol's §2 claim that no
model was iterated on the selected requests is historical, not an accurate label
for evaluating the corrected model on the same roots.

The outcome-linked exposure ledger identifies **249 development/exposed roots**,
leaving **at most 144 potentially untouched roots**. Fourteen retained
`cc_small.jsonl` outcomes match exact 60/300-row source prefixes; the 300-row
prefix contains 248 distinct session IDs. Retained `cc_pilot.jsonl` outcomes
match exact 20/62-row prefixes from one root already in that set. These source
IDs are 12-character prefixes: each maps uniquely to one full ID in the pinned
393-root corpus. Union with the 17 current/pilot development roots adds one
more root, `07dd40536557a1d6440a923557c3129dc929`. All 64 legacy-short registered
roots are already covered by this historical outcome evidence; no additional
registration-only quarantine remains in this inventory.

The earlier registration-only audit's 74-root union and 319-root reserve are
superseded. The anonymous older 64-row runs have synthetic-looking requests and
missing workload hashes; they are not the basis of this exposure conclusion.
Exposure applies at root level even when only an opening was executed: it does
not mean later long-decode or fanout mechanisms in that root were tested.
Additional retained-outcome searches can only reduce the potentially untouched
reserve. Inspecting corpus metadata alone does not expose target outcomes.

Fresh hardware repeats of the same roots estimate repeat noise, not independent
workload generalization. Nested C=1/2/4/8 workloads, TP comparisons and requests
from one root must remain linked in any uncertainty calculation.

## 5. Minimal next work, in order

1. **Finish the cache-aware contract and deterministic checks.** Keep the census,
   exposure ledger and legacy registrations. Bind source-local hash identities,
   exact synthetic token IDs, cache/checkpoint policy, native fork allocation and
   a matching empty-cache boundary. Keep future requests' prefix demand hidden
   until native readiness. The explicit zero-output surrogate is documented in
   `ZERO_OUTPUT_CONTRACT.md`; fresh diagnostics must declare it and leave TTFT/TPOT
   undefined where no output token exists, rather than silently dropping leaves.
   The historical selector's 28 excluded zero-output leaves remain part of its
   recorded scope. Preserve all source arrivals, descendants and outputs.
2. **Run an independent cache-on source/oracle preflight.** Check actual allocated
   KV/state memory separately from occupancy, checkpoint cuts and one-token fork
   work, cached-prefill dispatch and preparation regions. Reuse existing primitive
   prices only where signatures and source scope match. Cache-on can cut cold
   128-token prompts into 112+16 and 448 into 432+16; resolving a cache-off 448-token
   price does not establish that path. Preserve failed source criteria and the
   tiny TTFT/non-Torch failures; do not fit their residuals or widen thresholds.
3. **Obtain fresh cache-aware E2E evidence from exposed roots.** Start with a
   paired run of complete consecutive episodes from an exposed root and assess actual admitted hits,
   compressed/wanted/reusable tokens, checkpoint fates and pool pressure alongside
   timing/memory. Policy tests and source overlap alone prove none of those
   mechanisms. Keep the completed cache-off near-limit and sustained pairs as
   history; the old long real run stays held. Expand to complete continuity and
   C=1/2/4/8 root bundles without request throttles, dropped descendants or altered
   gaps. N1 support does not establish N2/pool support, and no source proxy alone
   proves target saturation or preemption.
4. **Freeze a separate confirmation design.** After model, transport and
   diagnostics stabilize, preregister untouched roots/bundles, exact workloads,
   weighting, configuration gates, environmental scope, sample count, repeats,
   multiplicity and stop rule before looking at confirmation outcomes. Keep
   current-suite reruns as regression evidence alongside this new evaluation.

Concrete native witnesses already present in the census:

| Mechanism | Root and native path | Cost/shape and limitation |
| --- | --- | --- |
| Low-prompt readiness | Current short root `3cb1c6dc71a21714587287306ebe52748568`, `/requests/8/requests/31` | 128 input, 171 output; use its whole source episode for E2E, the leaf for the CPU domain canary |
| Long decode | Historically exposed root `2a2da059b7425d9dc1f999fca1177bc1cdb9`, `/requests/6/requests/4` | 42,880 input, 59,903 output; another descendant has 59,725 output; preserve the containing episode |
| Near context limit | `ac5f7a7c9662c03f757eba1142136bd244d0`, `/requests/84` | 255,808 input +191 output =255,999; metadata-only witness, not yet a designated development root |
| Complete-root capacity | `d5654f5758cb492c6ad55ad95bc7e268b7a4` | 3,551 requests; 93,310.424-second arrival span; 532,458,048 input proxy tokens; CPU capacity witness first |
| Current-root continuity | Current short root `4b433e21f63822412e0f27d5473da9ea1322` | 2,141 API leaves, 2,139 individually eligible, 548,931.661-second span; current workload retains only a two-request episode |
| Cheap complete-root continuity | Historically exposed root `1493faffdc8942e99daa22277f44871a40b8` | Seven requests, 194,368 input proxy tokens, 4,097 output tokens, 373.459-second arrival span; use as the first intact-root E2E witness after deterministic support passes |

The final compact-case inventory (`ADDITIONAL_CASES_V4.json`) prefers roots
already exposed through the outcome-prefix audit. It can exercise additional
mechanisms without spending an untouched root:

| Mechanism | Exposed root, source episode | Requests | Input proxy/output tokens | Native arrival span |
| --- | --- | ---: | ---: | ---: |
| Smallest native request | `a26302776b9ed6524bac18bdf9a9430f4170`, episode 22, `/requests/22` | 1 | 128 / 16 | 0 s, then generation |
| Near context limit | `65c7b96990e2e7f206a2c01ed90474f56c57`, episode 163, `/requests/173` | 1 | 250,048 / 39 | 0 s, then generation |
| Long decode | `6c6be4bc5a49a5d1062b8becdbf188480843`, episode 22, `/requests/31` | 1 | 195,840 / 40,339 | 0 s, then generation |
| Sustained variable arrivals | Current short root `509ad65c576a007df9c0cf1e9863874da21d`, episode 284 | 21 | 459,072 / 15,979 | 68.342 s; source peak 5 |
| Fanout/target-pressure candidate | `5fc8495d01a2ad301e8ca7879579f2529029`, episode 10 | 95 | 2,019,392 / 92,183 | 333.885 s; source peak 12 |
| Opening growth and continuity | `7ff48bb238572dba164f441fc274d0cb6e52`, opening episodes 0–2 | 4 | 50,176 / 730 | 11.832 s; root input 384→384→33,408, then a descendant |
| Complete-root continuity | `1493faffdc8942e99daa22277f44871a40b8`, all five episodes | 7 | 194,368 / 4,097 | 373.459 s; longest source-idle gap 180.652 s |

Episode indices and complete native paths are retained in `ADDITIONAL_CASES_V4.json`.
Selection here is for mechanism discovery, not statistical certification.
Arrival span excludes final generation time; input+output volume is a work proxy,
not predicted GPU runtime. Fanout and volume nominate target-pressure tests but
do not establish saturation or preemption. The four-request opening preserves
its 0.864- and 0.068-second source-idle gaps with no artificial target drain; it
does not stand for the remainder of that root. A 95-request case is a useful
mechanism witness but does not qualify the 969-request episode or larger root
submission limits. Prefer the cheap complete-root case
and a minimal complementary subset first, and do not make the largest extrema
mandatory early GPU runs merely because they are extrema.

A minimal cold-opening follow-on is available without using reserve roots:
`0bcd99353218b9a386b1c4aec13d77abecdb` and
`4b88101f5ed7507b73138a73dfbf55f972de` each has one complete initial source
episode containing only `/requests/0` at time zero, with inputs 320/320 and
outputs 21/16. Their zero-aligned C2 offer can exercise a small N2 group, but
**actual batching must be witnessed, never forced or inferred from C**. C4 can
add the intact 384- and 448-token openings from
`48f9644a9b17cade09142b3d8f964bea35be` and
`6d5c6b957d5f634a7f820ddff7389e161def`. Only six eligible exposed complete first
episodes have every input below 640; do not manufacture C8 by duplicating roots
or dropping larger neighbours. This is a metadata-only development proposal,
not a full-session or confidence sample; no new workload or run was created.

## 6. Confidence limitation and next step

**No currently practical small suite establishes literal joint 99% confidence
over all 393 roots.** Static coverage and repeatability do not establish workload
generalization. Confidence in configuration estimates is also different from a
claim that at least 99% of requests/workloads pass; no such pass-rate gate is
added here.

Prioritize the cache-on source/oracle preflight and a paired run of complete
consecutive episodes from an exposed root. Retain historical cache-off results; add cache-aware continuity, fanout
and multi-client evidence where each exercises a remaining mechanism.
Before confirmation, freeze the predictor and a probability design covering
both the 249 exposed and at-most-144 untouched partitions, with known inclusion
probabilities. The untouched partition cannot stand in for all 393 roots.
Specify the C=1/2/4/8 root-bundle construction, preserve complete within-root
structure and gaps, and keep each root's requests, nested C scenarios and
hardware repeats together in uncertainty calculations. Three paired repeats
remain a useful run-noise check, not three independent workload samples.

A later registration must fix sample size, repeat count, budget, population
aggregation and a validated simultaneous confidence method before outcomes are
seen. Preserve the existing metric functionals/bars and mandatory memory,
feasibility and ranking checks; allocate total alpha 0.01 across the claimed
family rather than calling separate intervals jointly 99%. Insufficient
clusters, uncertain interval coverage, bounds crossing a bar, incomplete runs
or budget exhaustion mean inconclusive. Do not reroll expensive/failing roots
or add repeats until a nominal bootstrap interval passes. Resolve zero-output
semantics before retaining an all-393-root claim. All current timing evidence
remains node-busy/advisory; TP2/TP4 remain paused.

The feasible cost/precision decision and detailed sampling registration are
deferred until mechanism results justify them. This review allocates no reserve
roots and launches no confirmation experiment.

## 7. Cost and evidence record

A full single-C=1 sweep starts with 6.89 billion input proxy tokens and 58.73
million output tokens before configuration variants and repeats. Full-root
arrival spans range from 69.146 to 917,239.434 seconds (10.6 days), with median
6,607 seconds (1.84 hours). Their summed arrival span is 18,506,239.930 seconds,
or **214.2 days** of sequential C=1 pacing before final response time,
configuration variants and repeats. This is a cost floor for that schedule,
not a claim that all runs must execute sequentially. Thus CPU census and deterministic support checks are
cheap compared with raw-paced confirmation. The predictor can advance virtual
idle time, but the real reference cannot silently compress gaps while retaining
the same registration. Future idle-skipping would need a separately qualified
protocol, paired evidence of target drain and equivalent native state/time
semantics at each skipped gap, and hardware wake/cooling effects accounted for.
Source G=0 boundaries alone do not supply that evidence; no gap is compressed
in the proposed confirmation. Runtime budgets must be based on
preserved arrivals, target execution and preparation/startup/derivation costs,
not input+output volume alone.

The immediate feasible E2E next step is the seven-request, 373.459-second
complete-root witness above plus a few compact exposed-root mechanism cases,
after their deterministic support checks. These can falsify functionality or
continuity quickly; they cannot certify 99% confidence cheaply. The full-root
probability design remains conditional on a frozen precision/cost plan that the
available clean-root reserve and hardware budget can actually support.

The audit artifacts are ignored under `agent_scratch/codex_corpus_coverage_v1`
in the CPU evidence tree, with request/episode/transition inventories retained.
The host tree is
`/md1/users/jgong5/atomcompass-worktrees/codex-regions-e2e/agent_scratch/local_cpu_stage_v1/ATOM`;
the device-free CPU container exposes it as `/workspace/ATOM`. Historical
accuracy/exposure inventories are under its sibling
`agent_scratch/codex_cc_trace_request_accuracy_audit_v1` directory.

| Artifact | SHA-256 |
| --- | --- |
| `EARLY_CENSUS.json` | `cf99ddd5234fead662d335fc88593f5ccbdb237cb0c2d3f3813d51637f81f615` |
| `ROOT_INVENTORY.json` | `43b571ce3422463c65b32e60ac9aab86efe0224625f5131800fff550d0117279` |
| `COVERAGE_REPORT.json` | `796d790d8663faecf5f43b7eb9c018c72ffb82e041b3e3df69de2d863925f8c3` |
| `LOW_PROMPT_BYTE_CANARY.json` | `90c3d1ea9f5a4448824bf56d74ee72e02d2c0d3fc1b1f85156d2cdce7742d0f3` |
| `ADDITIONAL_CASES.json` (initial, before exposure preference) | `3c2538559dd22a061daa17c951c22f39510ea35e2fbed2ed32a2f79136289a12` |
| `ADDITIONAL_CASES_V4.json` (final exposed-root preference) | `29eb56aa0f0b1801194cb75a57e94c3da1bf49410a50a74f22c7084528fcd51f` |
| Baseline request-accuracy `INVENTORY.json` | `e930c135ac15b07e3e7dcf97faf716a3fd8856bfe63fabed9dd108d9fd9809f6` |
| Historical outcome-prefix `LEGACY_EMBEDDED_IDENTITY_V1.json` | `45f6964243dbfd8fa1d9fae6911ec6079251247b7a765b4450984fa76a450660` |
| Low-range `SOURCE_SUPPORT_HANDOFF_V1.json` | `e5ddcc22a7452f7580cade6c7dafdda9ba2a4e6f7c4a2705aaa9ec63d19ead55` |
| Low-range `CANDIDATE_PROFILE_V1.json` (diagnostic-only selection) | `6e26d4cb7304b9d6868308deca808dd0334bf30747146ce4f3682a7adf38e3f9` |
| Tiny `REFUSAL_HANDOFF_V1.json` | `38f9ba230d252dbd3821d2725d3f903bad981ade28d2c1cdc9fd07bec5d881a8` |
| Near-limit `REFUSAL_HANDOFF_V1.json` | `b2f44ec02f84db1b9425a98b4f300ead3a51793d1fff06c7b562470b64e2a059` |
| `REGION_SUPPORT_HANDOFF_V1.json` | `59870d4c9dbc5668fd71927f4f56447fdaba84a8011365d6103c5a59f3779b66` |
| Sustained `PAIRED_HANDOFF.json` | `45052b75d5db0fa4cec2257b22be1d3ba9017df9ae249a3da3e88590f34595e1` |
| High-history `HELDOUT_VERDICT_V1.json` | `ed6896d490a1ad1e327fd2009990b72fba4f475c1faa97f3b3188910e58ea445` |
| Near-limit `PAIR_VALIDATION_V1.json` | `aadf9f70a21616efaa3c0150261bc8b8b1d53bacc66707b093f835e9d68e8cbc` |
| Long-decode `MODELLED_VALIDATION_V1.json` | `6fb8a2595ea310441303bdb12d8dfb4f6b79b83d481df8e2f42bde86cd6e7116` |
| Tiny `TINY_NO_CACHE_PAIR.json` | `799b73abc5c05e92eca8953696c56f2514412c76cb47e8f4670ede856db8bbe4` |
| Intact-root first QK-norm refusal | `b3d2ff0e4217be77d8dbac60b4bb601be633c562f13113cc785add8c9767e794` |
| Integrated cache-on `CPU_REGRESSION_V1.json` (CPU contracts only) | `b485bb197addca09c8431b32d3174cceceeda83211ac334f96d2e7de10cb3e53` |

The low-range handoff and candidate are under
`agent_scratch/codex_decode_domain_v1/ingress_handoff_low_extension_v1` in the
same CPU evidence tree. Earlier profiles, canaries and audit findings remain
historical evidence; the candidate does not relabel them.
The refusal handoffs are under the respective case directories in
`agent_scratch/codex_corpus_diagnostics_v1/results/`; the source audit is under
`agent_scratch/codex_small_prefill_regions_v1/`. The metadata-only C2/C4 proposal
is retained in the confidence worktree's
`agent_scratch/corpus_confidence_v1/COLD_OPENING_BUNDLE_PROPOSAL.json`.
The sustained handoff is under `agent_scratch/codex_sustained_arrivals_v1/`
in the same CPU evidence tree; it pins all real/modelled artifacts, remote-copy
hash checks and the separate final sampler and later release observations.
Low-source failure and diagnostic candidate evidence remain under
`codex_small_prefill_regions_v1/` and `codex_low_prefill_diagnostic_v1/`.
The high-history verdict is under
`codex_high_per_sequence_source_v1/acquisition_v4/`.
The later near-limit and long-decode receipts are under
`codex_near_limit_high_v1/` and `codex_long_decode_high_v1/`. The tiny pair is
under `codex_low_gather_source_v1/finite_overlay_v1/`, with intact-root refusal
details in `root_modelled_execution/REFUSAL_CLOSEOUT.json`. These are historical
cache-disabled diagnostic artifacts, not cache-on acceptance or confidence samples.
The integrated cache-on CPU receipt and JUnit output are under
`codex_cache_on_integration_2782333_v1/`; they record code-contract checks only.

The current root exposure ledger is
`agent_scratch/corpus_review_v1/EXPOSURE_LEDGER_V2.json` in the review worktree.
It records the historical outcome-prefix/full-root identity map and union with
current development roots, superseding the preliminary registration ledger.
Raw traces, generated census outputs and scratch scripts stay
ignored; this maintained review records their identities and conclusions.

Code witnesses: `cc_traces_clients_workload.py` (`busy_episodes`, `select_pool`
and `build`); `cc_traces_workload.py` (`checked_tokens` and
legacy selection); `replay.py` (`MAX_IN_FLIGHT`, pretokenization and submission);
`Scheduler._arrival_barrier_unmet`; `compare.py` (`metrics`, `_quantiles`,
`compare`); `cc_traces_validate.py` (`_across_repeats`). Source cards are pinned
to the commits below; saved copies were hash-checked against those versions.

[filtered-card]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/blob/8fecd2fc56694469f758f0afbbb6335ad3043740/README.md
[parent-card]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/blob/23f152f6f0f9399a85901b89a6458def0ef16729/README.md
