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

Readiness-byte support did not establish native region support. The two new
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
breakdown/band comparisons unchanged, and 209 tests passing. A fresh near-limit
run on this preset is pending; source validation adds no E2E acceptance.

Low-prefill source preparation **failed** the unchanged 110 µs criterion:
17/30 pass, 13 fail, maximum **427.143 µs**. Postprocess 14/14 and structural
zeros 16/16 pass. TP1 candidate `e43976eb` is explicitly **diagnostic-only**,
using already-frozen anchors without default/acceptance activation or N2/pool
extension. Its tiny follow-on loaded correctly, then exited 5 on missing
head-row gather price `aten::index.Tensor|128,5120;1|bfloat16,int32|1:127`
(2,442/2,443 operators priced). No tiny request completed; the original source
failure remains a failure. TP2/TP4 work remains paused.

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

The claim remains conditional on the declared deployment and replay semantics:
Qwen/Qwen3.8-27B, TP ∈ {1,2,4}, existing memory and scheduler settings, synthetic
content, prefix caching disabled, fixed source outputs, and open-loop arrivals.
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

The current short/large C=1/2/4/8 workloads contain **54 unique requests from 16
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
prefix hashes describe these histories even though prefix reuse is disabled.

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

**Small requests are a demonstrated support failure.** The readiness profile
`8317c718…` accepts serialized ADD payloads starting at 5,875 bytes. Native
`Sequence`/`CoreManager.add_request` serialization of all 532 non-pilot eligible
requests below 1,024 tokens produces 2,329–5,668 bytes under the standard replay
layout. A separate descriptor calculation agrees for 1,064 ordinary/full-corpus
ordinal variants; all are refused. This affects **340/392 roots (86.7%)**,
including 267 first root turns. Small request mass is therefore not evidence
that the issue is harmless for complete-session support. These are deterministic
native serialization checks, not GPU or HTTP timing observations. A source
measurement or justified source model extension is needed; silently widening a
bound would not provide that evidence. Even among the 378 non-pilot roots with
every leaf selector-servable, 328 contain one of these confirmed refusals.

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

1. **Finish deterministic support checks and freeze exposure.** Keep the complete
   census and current registrations. Resolve the 28 zero-output leaves explicitly
   for a full-corpus claim: define their replay/outcome semantics and undefined
   token-latency metrics where no token exists, or declare the exclusion and
   reduced population. Validate standard serialized
   descriptors, token/context limits, model-domain bounds and request accounting
   over every retained leaf, including output-producing low prefill, cached tails
   and decode history reached during generation. Retain the completed async
   transport qualification and its socket/memory limits.
   Check intact roots, not just individually eligible leaves. Keep these checks
   separate from target timing claims.
2. **Use a few corpus-native development cases to expose mechanisms.** Prefer
   already-exposed roots wherever they provide equivalent coverage. Retain every
   leaf in the selected native episode or continuous trajectory, including all
   descendants. The compact witness list below specifies mechanisms and costs;
   source overlap/volume nominate cases but do not prove target saturation or
   preemption. Require actual target scheduler/KV witnesses for those claims.
   First close the tiny-prefill and near-limit decode refusals with source-only
   evidence, then obtain fresh pairs for those cases and the seven-request root.
   Retain the completed 21-request sustained pair and its per-request misses;
   use fanout/long-decode cases when their extra mechanism is needed, not merely
   to fill a grid. Keep multi-client cold small groups open: N1 evidence does
   not establish N2/pool support. Add complete-root continuity and keep the
   C=1/2/4/8 axis as roots rather than a request throttle.
3. **Freeze a separate confirmation design.** After model, transport and
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

Prioritize the source-price gap and fresh tiny, near-limit and intact-root pairs;
retain the sustained pair and add fanout or multi-client cold-opening evidence
where it exercises a remaining mechanism.
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
