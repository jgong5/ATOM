# cc-traces coverage and confirmation review

This review separates the current registered burst suite from a claim about the
whole pinned cc-traces corpus. **The present evidence does not establish
full-corpus accuracy or 99% statistical confidence.** Two functionality limits
come before additional timing campaigns: the readiness profile refuses small
native requests, and the replay transport cannot submit some intact roots.
Coverage also needs sustained and multi-episode continuity, since success on
isolated source busy episodes does not imply success on their concatenation.

This is a coverage review and a proposed confirmation design, not a replacement
registration. Existing workload hashes, run registrations and results retain
their historical meaning. No GPU run, fit or workload rewrite was performed for
this review.

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

**Submission has a finite workload limit.** `replay.py` pretokenizes every prompt
and uses one blocking thread/HTTP request per workload row. It rejects workloads
over `MAX_IN_FLIGHT=1024`; the limit is checked after optional preparation and
pretokenization. Thirteen published roots already exceed 1,024 API leaves,
and combining C roots can exceed it sooner. The maximum root has 3,551 API
requests. A 969-request episode is below the nominal limit but is not evidence
that thread, connection, memory or timeout behavior is adequate near the limit.

The modelled server waits for all declared workload rows before advancing
virtual time. Replacing the client with K blocking workers while N>K requests
must register deadlocks behind responses the server cannot yet produce; the
existing comments document this failure. The 120-second barrier fallback makes
that replay invalid, and the readiness path now refuses an incomplete barrier.
A bulk/streamed registration protocol or another explicitly proven transport is
needed for larger workloads. `--num-requests`, a client-count semaphore, or a
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
   over every retained leaf. Confirm native replay transport above 1,024 with CPU
   response-holding tests: complete registration, stable ordinal/timestamp
   semantics, no lost rows, no barrier timeout and bounded client resources.
   Check intact roots, not just individually eligible leaves. Keep these checks
   separate from target timing claims.
2. **Use a few corpus-native development cases to expose mechanisms.** Prefer
   already-exposed roots wherever they provide equivalent coverage. Retain every
   leaf in the selected native episode or continuous trajectory, including all
   descendants. The compact witness list below specifies mechanisms and costs;
   source overlap/volume nominate cases but do not prove target saturation or
   preemption. Require actual target scheduler/KV witnesses for those claims.
   Add a complete-root continuity check after transport support, and keep the
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

Compact cases from the initial episode inventory are now known to belong to
historically exposed roots through the outcome-prefix audit. They can exercise
additional mechanisms without spending an untouched root:

| Mechanism | Exposed root, source episode | Requests | Input proxy/output tokens | Native arrival span |
| --- | --- | ---: | ---: | ---: |
| Mixed near-limit/short inputs | `27de9ca964aa7905c2f5438db8bdd72d8b02`, episode 63, root paths 69–71 | 3 | 248,640 / 2,779 | 19.938 s |
| Long decode | `6c6be4bc5a49a5d1062b8becdbf188480843`, episode 22, `/requests/31` | 1 | 195,840 / 40,339 | 0 s, then generation |
| Native fanout | `5fc70a9165e6a1fd4bfd1add7ef5452205b2`, episode 50 | 13 | 345,984 / 11,284 | 40.341 s; source peak 9 |
| Sustained input-volume candidate | `21cde366f5bd2f0ff2031b0538496c7e2a05`, episode 63 | 34 | 2,001,280 / 32,360 | 59.893 s; source peak 4 |
| Idle continuity | `94055207ca7ba1e62a4bdaf29652e35fdf14`, adjacent episodes 100–101 | 2 | 4,736 / 793 | 90.119 s; preserve the 82.498 s source-idle gap |

Episode indices and complete native paths are retained in `ADDITIONAL_CASES.json`.
Selection here is for mechanism discovery, not statistical certification.
Arrival span excludes final generation time; input+output volume is a work proxy,
not predicted GPU runtime. Fanout and volume nominate target-pressure tests but
do not establish saturation or preemption. Prefer the cheap complete-root case
and a minimal complementary subset first, and do not make the largest extrema
mandatory early GPU runs merely because they are extrema.

## 6. What a defensible 99% statement requires

**99% confidence about fixed configuration-level gate estimates is distinct from
“at least 99% of requests/workloads pass.”** The latter is a separate reliability
estimand and has not been added as a PoC requirement. Nor does 99% descriptor
coverage imply either statement. Exhaustively evaluating a declared finite
scenario set establishes facts about that set, with remaining hardware
repeatability uncertainty; it does not create independent samples of deployment
workloads. Even replaying every individual root at C=1 does not exhaust the
possible C=2/4/8 root combinations; the scenario distribution must remain explicit.

Use the root as the minimum workload cluster. For the C axis, a practical common
statistical design, conditional on a feasible execution budget, is randomly
ordered, disjoint bundles of eight untouched roots, preserving
each root's full trajectory. C=1/2/4/8 use nested prefixes within each bundle;
all TP variants and repeats share those identities. The bundle is the outer
statistical unit, and requests, C values, TP values and repeats are dependent
measurements within it. With the current conservative reserve there can be at
most **18 disjoint eight-root bundles**, before any further quarantine or new
development use. This is a ceiling, not a justified sample size. Other valid
root-sampling designs are possible, but must state their dependence explicitly.

Root strata can be chosen from census-only properties such as size, input/output
tails, fanout and trajectory span. A probability sample must give every member of
its declared population a known nonzero inclusion probability; use fixed stratum
weights for oversampling. Pure greedy mechanism selection belongs in development.
Sampling only the untouched partition supports that partition directly. An
all-corpus estimate must also account for the exposed/quarantined partition with
explicit weights and evidence; it cannot silently drop it and retain the
393-root label. Fresh evaluation of exposed roots can be labelled regression or
finite-corpus verification, but does not restore unseen-workload status.

The registration must define the **estimand before the interval**. Preserve the
existing per-scenario metric functionals and tolerance bars, and say how scenarios
aggregate into each configuration estimate. For example, request-weighted TTFT
quantiles require a root-sampling-weighted request distribution, whereas an
equal-root quantile is different. Throughput of a synchronized root cohort is
tokens divided by its own end-to-end window; a mean of root throughputs or one
giant pooled clock is a different estimand. Do not substitute median or mean
per-request error for error between the registered aggregate summaries. Keep
request-level absolute error and failures as visible diagnostics beside the
configuration gates. Idle-dominated throughput can obscure service-time error;
retain queue/latency and sustained-load diagnostics alongside it without silently
changing the gated functional.

For uncertainty, keep real/modelled observations paired and propagate workload
cluster and hardware-repeat variation separately, preserving strata and all
within-bundle dependence. Use simultaneous intervals for the predeclared family
of configuration/metric claims: independent 99% intervals per gate do not give
99% joint confidence. A Bonferroni allocation of total alpha 0.01 across the
fixed family is a simple conservative option; a justified simultaneous method
can be more efficient. Ranking/top-1, memory, infeasibility and speedup retain
their stated roles: ranking/top-1, memory and infeasibility remain mandatory,
while speedup is advisory under the user's override. Mandatory claims need
explicit treatment if included in the joint 99% statement; a speed estimate can
be reported separately. Bootstrap intervals from a small bundle sample are approximate; validate
the chosen design's assumptions and label that approximation rather than claim
an exact guarantee.

**Stop rule:** use exposed-root development measurements to plan cost and
precision, then fix the confirmation sample size and hardware-repeat count
within a written budget. Pass only if all mandatory functionality checks pass
and the simultaneous confidence intervals for every claimed error gate lie
wholly within its unchanged tolerance. An interval crossing a tolerance is
inconclusive, even when its point estimate passes. Exhausting the budget or
reserve is also inconclusive. Do not repeatedly add roots until a naive interval
passes. If sequential confirmation is necessary, register a valid alpha-spending
or confidence-sequence procedure first. Any model change after inspecting a
confirmation outcome moves those roots to development and requires a new frozen
confirmation set.

As an illustration of why a different reliability claim is demanding, with n
independent IID Bernoulli trials and all successes, the one-sided exact 99%
lower bound is `0.01**(1/n)`. Requiring that bound to reach 0.99 needs at least
459 independent trials. The current correlated requests/repeats are not such
trials, and that calculation is **not** a sample-size prescription for the
configuration-error estimands above or for this finite corpus.

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
the same registration. An accelerated-idle protocol would require a separate
scope and evidence of state equivalence. Runtime budgets must be based on
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
| Baseline request-accuracy `INVENTORY.json` | `e930c135ac15b07e3e7dcf97faf716a3fd8856bfe63fabed9dd108d9fd9809f6` |
| Historical outcome-prefix `LEGACY_EMBEDDED_IDENTITY_V1.json` | `45f6964243dbfd8fa1d9fae6911ec6079251247b7a765b4450984fa76a450660` |

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
