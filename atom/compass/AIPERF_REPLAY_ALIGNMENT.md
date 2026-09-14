# Alignment with published cc-traces replay

The current `4b433` and `31dd` completion workloads are **prefix-mechanics
diagnostics**. They preserve declared source prompt-prefix identities and exact
proxy lengths, but do not yet establish AIPerf chat, full-history cache, or
dependency-scheduling fidelity. `4b433` now has a completed paired result within
that narrower scope: TTFT median +8.59%, TPOT median +4.92%, throughput −1.88%.
Its original isolation failure keeps timings advisory; the post-run aggregate
registry correction and unavailable normal cost merge remain explicit in
[PoC status](POC_STATUS.md). No faithful AIPerf opening pair is established.

The reviewed `4b433` native structural closeout now establishes a true 32,768-token
hit against 38,400 wanted tokens, 13 retained checkpoints and no evictions. Its
physical pool contains 112,760 KV blocks plus 32 state slots, totaling
121,671,450,624 bytes. All memory gates pass against the unchanged source memory model;
the preparation/final resets and cleanup pass. This structural receipt establishes
bounded native cache mechanics and memory checks; the subsequent paired
surrogate result above still does not establish default-chat fidelity. The receipt
and the superseding seeded chat audit are pinned below.

This audit pins SemiAnalysisAI/aiperf commit
`0d2aa0572ac685943d38c580675c4a61023581d3` and the [published dataset card][card]
at revision `23f152f6f0f9399a85901b89a6458def0ef16729` (card SHA256
`c5358865e47321e50b89e7c04168cb3e7601307c48541149462bc217853e9fcf`). Our input
is the 393-root **062126-256k** sibling, corpus SHA256
`e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`.
The card specifies a loader flag, not a complete execution profile. The rolling
with-subagents alias points to the full-context parent; the explicit 256k alias
and local-file loader must not be confused with it ([registry][registry]).

## What the published surrogate preserves

The default loader synthesizes coding-text blocks from `(root trace ID, local
hash ID)`. Root, inferred-agent and explicit-subagent conversations share that
root namespace. It reconstructs chat role segments from hash-prefix geometry and
recorded previous output lengths; **live generated assistant responses are off
by default**. The optional live-response mode trades recorded hash fidelity for
reuse of generated-output KV. Unreleased original text is therefore not an
additional PoC blocker ([defaults][defaults], [loader][loader], [roles][roles]).

The source `in` count is a proxy block count times 64, not a true tokenizer count
([card][card]). AIPerf decodes synthesized content and sends chat messages, so
retokenization and the server's chat template change both length and prefix
alignment. Default local ISL reporting joins message contents with spaces;
`--apply-chat-template` is optional. Pin actual server token usage and consumed
token digests for acceptance, not just that reporting flag ([token-count][count]).
Our direct-token codec deliberately avoids these transformations and is not
byte-identical to AIPerf's coding-content surrogate.

## Explicit execution profile

The [Weka tutorial][tutorial] adds `--endpoint-type chat --streaming
--fixed-schedule` to the dataset flag. At the audited commit, HF plus
`--fixed-schedule` without a file fails `InputConfig` validation. The loader flag
alone selects generic concurrency-burst timing (default concurrency 1, request
count 10), not the tutorial's fixed schedule. CPU configuration checks confirmed
both outcomes. Use an explicit local-file profile ([input-config][input-config],
[timing-config][timing-config]). For the bounded opening below:

```bash
aiperf profile \
    --url localhost:8000 --model Qwen/Qwen3.8-27B \
    --tokenizer Qwen/Qwen3.8-27B --endpoint-type chat --streaming \
    --input-file agent_scratch/codex_prefix_cache_v1/AIPERF_OPENING_72d0_TWO_TURNS.json \
    --custom-dataset-type weka_trace --fixed-schedule --fixed-schedule-auto-offset \
    --random-seed 42 --concurrency 1 --request-count 2 --use-server-token-count
```

Pin the code, tokenizer, chat-template kwargs and input bytes below. Leave generic
warmup unset and establish the PoC's acknowledged empty cache boundary before
measurement. Ordinary AIPerf sets maximum output counts; it does not force them.
If a PoC pair requires the recorded output lengths exactly, add
`--extra-inputs ignore_eos:true` as an **explicit profile variant**, and verify
actual counts on both sides. That is not the generic fixed-schedule default.

For ordinary same-chain turns, fixed scheduling waits for the prior target
response, then schedules at the next absolute source timestamp. A timestamp
already in the past dispatches immediately: release is
`max(target prior return, run origin + source timestamp)`, before transport
overhead. The loader also records source start-to-start `delay_ms`; fixed mode
uses the timestamp in preference to that delay. Generic request-rate mode instead
adds this delay after target return ([fixed][fixed], [clock][clock], [rate][rate]).

Branch behavior needs separate qualification. Return callbacks spawn children;
their offsets are relative to the branch start. An unsatisfied `SPAWN_JOIN`
blocks the parent; its eventual release goes directly through the issuer, without
the ordinary fixed-timestamp path. Already-satisfied joins fall through to that
ordinary path ([callback][callback], [branches][branches], [issuer][issuer]).
A bounded CPU execution of `FixedScheduleStrategy.setup_phase/execute_phase`
also produced independent depth-0 first credits for **both** root metadata and
branch-referenced child metadata. This is planner/issuer-capture evidence, not
an observed duplicate HTTP send. Use a branch-free opening for the first fixed
profile pair; do not infer correct fan-out behavior from the tutorial alone.

AgentX is a different, scenario-locked profile: response-relative end-to-start
delays after per-root start-gap compression, trajectory cache warmup, recycle
markers, and tree-slot accounting. Only that mode installs the extra recorded
cross-stream completion barriers and holds each slot until the whole tree drains
([scenario][scenario], [phase][phase], [trees][trees]). Future C1/C2/C4/C8 fan-out
coverage must name its profile, count root trees rather than requests, and allow
more than C simultaneous requests from their descendants.

## Measured chat shapes and first faithful opening

These are CPU reconstructions using the actual pinned loader/coding generator,
Qwen tokenizer, `add_generation_prompt=True`, default template kwargs, and seed
42 initialized exactly as [AIPerf's bootstrap][bootstrap] does:
`rng.reset(); rng.init(42)`.
The earlier unbootstrapped audit draws are superseded. Canonical tokenizer backend
SHA256 `3173693f511156ffb4ad1922cef28eeea31bb7d1bd264346d1fe10e684e23059`
matches the n18 preflight; template SHA256 is
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.

| Source requests | Source inputs | Output maxima | Chat segments | Rendered inputs | Input modulo 16 | Rendered LCP |
|---|---|---|---|---|---|---|
| `4b433… /919, /920` | 92,928 / 92,672 | 12 / 354 | 160 / 3 | 93,986 / 92,701 | 2 / 13 | 38,424 |
| `72d021… /0, /1` | 38,208 / 39,936 | 338 / 186 | 1 / 3 | 38,240 / 39,982 | 0 / 14 | 38,239 |

The full `4b433e21f63822412e0f27d5473da9ea1322` root maps the selected requests
to turns 161/162 of inferred child `::fa:011`; the second resets chat context.
The two-row diagnostic omits that earlier lineage and cache state. Its original
114.735-second gap matches fixed release only if the first target response
returns in time and no other prerequisite delays it; this has not been proved.
The rendered tails above also show why final-16 source coverage alone cannot
close chat-prefill coverage: collect actual native scheduler/kernel shapes.
Modulo zero can imply a full final 16-token chunk; it does not mean zero work.
The other modulo values (2, 13 and 14) also require native scheduling evidence.

Recommend root `72d021d543c045e303c44b3c478f626cd21b`, corpus line 182, requests
0/1, as the first faithful opening. It is the smallest by summed input in the
restricted exposed-root shortlist with no explicit subagents and no inferred
child active through the second source completion; it is not a global-minimum
claim. Full-root planning confirms these are the first root-chain turns, and
future role-planning caps do not alter them. The complete root has 21 requests
and three later inferred branches; those later requests are explicitly excluded.
No earlier request/cache state is omitted. The second release is
`max(run origin + 21.437 seconds, first target return)`; its 338-token predecessor
can therefore make gating materially different from open-loop replay. Rendered
common-prefix length floors to 38,224 native tokens; actual reusable KV/state
must still be measured, not inferred from this LCP.

## Minimal implementation implications and evidence

Export the pinned loader's reconstructed messages and dependency metadata once,
then use the same payloads and release rules on real and modelled clocks. Preserve
source IDs, role resets, output/EOS policy, root membership and token digests;
do not feed measured real arrivals or cache hits into the predictor. The opening
needs one response-gated continuation. Fan-out requires a corrected bounded
scheduler adapter or a separately pinned AgentX profile before broader claims.
Check server-consumed tokens, native shape/pricing support, empty-start receipts,
observed cache/state reuse, and paired timing before promoting either case.

The bounded adapter now has explicit entry points:

- `scripts/compass/export_aiperf_opening.py` exports the first two source turns
  after checking their full-root inferred ancestry and role-planning constraints.
  It pins the audited AIPerf Python source, exact exporter bytes, environment,
  tokenizer/template, coding pool, messages and rendered token arrays.
  Effective Weka reconstruction settings are recorded and required to match
  the pinned defaults, including synthetic rather than live assistant history.
- The predictor takes `--compass-opening-plan PATH
  --compass-opening-plan-sha256 SHA` with its source-backed
  `--compass-request-readiness-profile`. `ReleaseCalendar` separates registration
  from causal release; only released requests enter the qualified serial ingress
  service. Intermediate streaming output does not release the continuation.
- `scripts/compass/replay.py --opening-plan PATH --opening-plan-sha256 SHA
  --port PORT --out RESULT` selects the chat-opening path. The real clock waits
  for final SSE/DONE and EOF; the predictor preregisters both requests. The
  predictor must be fresh and unprepared. Native preparation can use the existing
  synthetic prompts at the exported rendered lengths, followed by the required
  empty-cache reset; measured chat payloads remain frozen.

Use the maintained `cc_traces_run.py opening-side` entry point for a measured
diagnostic, with `--case-id`, `--opening-plan`, `--opening-plan-sha256` and the
usual side/source options. `--plan-only` verifies the concrete argv without
launching a process. This case has schema `compass.aiperf_opening_case/1`; it
does not claim the source-proxy/prefix-codec contract of `diagnostic-side`.
Both use the same owned-process lifecycle, isolation, preparation, refusal and
execution-identity machinery. `opening-pair` reuses the maintained metric,
provenance, calibration and policy-aware memory checks, then writes
`opening_diagnostic.json` with `purpose=diagnostic` and `accepted=false`.

The opening route also checks the exact optional
`atom.compass.runtime.cache_region_oracle.source_cost_oracle` wrapper. It
retains the ordinary completeness, head, native-allocation, topology and
option checks on the base composition. Pair validation requires the pinned
overlay and optional q16 bundle at their configured paths: their bytes must
match the worker's original loaded inputs. It rebuilds the selected region
snapshot with the recorded flags and checks the selected coefficients against
the source registry. q16 source files and the deployment request scope must
match their loaded identities. The result retains the observed wrapper name;
the base-factory check is a separately labelled validation view. Selecting the
FAILED outputless source remains explicit in the diagnostic notes and never
gains acceptance credit. The registered acceptance route remains unchanged.

The opening deployment pins bf16 KV, block size 16, batch budget 16,384,
level 3/FULL and the declared capture ladder through 256. Core policy and
scheduler limits come from the core cache utility; worker configuration and
graph facts come from the worker's input-manifest RPC. Native capture sizes
are published only after native capture completes. The predictor separately
reports effective decode buckets and its borrowed target ladder, with no
claim that it captured graphs. The same producer fix supplies full cache
policy to the existing `4b433` mechanism path. Legacy validators retain their
historical rules; the opening path adds its full owned-configuration checks.

The initial profile is **“AIPerf-aligned opening, zero-response-delivery
approximation.”** It explicitly approximates modelled client-response
availability by native engine finish plus zero delivery time, not a calibrated
coefficient. Real engine finish, SSE terminal frames, EOF, client return and
the next preprocessing boundary are retained separately. A material gap needs
independent source calibration, not fitting the evaluated pair's residual.
Post-run checks require matching consumed tokens and cache policy, the loaded
core plan, both terminal completions/releases, and readiness service starting at
the causal releases. Client/server runtime revision remains separate from the
export's producer identity. These contracts have CPU validation; they do not
establish native shape support or paired timing.

Ignored evidence resides in `jgong5_compass_cpu` under
`/workspace/ATOM/agent_scratch/codex_prefix_cache_v1/`:

- `AIPERF_CHAT_AUDIT_SEED42_V2.json`, SHA256
  `50d5d71d7b5801d3bba37b72f5aba800f9d2869329d6ad5422cb3a605903180d`;
  includes exact script, tokenizer, coding-pool, source and token digests.
- `AIPERF_FIXED_SCHEDULE_PLAN_AUDIT.json`, SHA256
  `302544dc7767f488065a3f32c192089a0c276f9f33b26889f73b6618b87f594c`.
- `AIPERF_OPENING_72d0_TWO_TURNS.json`, SHA256
  `1bcbd296f5c5b5a48889ec6cc7b29b60c41b034243879896812d0759e323ca0e`.

The original frozen adapter export is retained in the sibling
`codex_aiperf_opening_v1/OPENING_PLAN.json`, SHA256
`16182a101d0a9579c1b03e48625ee792852aabd403ac47910e653a5ccaa14fb4`.
Maintained opening diagnostics use `OPENING_PLAN_V2.json`, SHA256
`dca12eef53e0d1fa00b0742e58558ebbfcae4e2548f0a69f8b8080182c5306f5`.
It adds the verified 14-setting Weka producer policy; every request and every
non-producer field equals the original. The unchanged shape inventory applies.
`PRODUCER_POLICY_SUPERSESSION_V2.json`, SHA256
`72c8507fe0a4076233c1fea19de276e54d2489ba2498b72a72755e9f962fb286`,
records that comparison. Concrete plan-only commands and both lifecycle plans
are in `maintained_handoff_v1/HANDOFF.json`, SHA256
`a6a8facc24e373e0e40719002c47bab99866dc5f002f12580d1e9074fd1b635b`.
Those plans retain the existing source registry for review; plan construction
does not establish primitive/region support or authorize a server launch.
They predate wrapper selection and must be superseded with complete source
pins before an opening run. The currently qualified q16 bundle for the
separate `4b433` mechanism fixture does not cover the opening's final q14 or
its other newly inventoried shapes.

The sibling `codex_cache_on_preflight_v1/STRUCTURAL_CLOSEOUT.json` has SHA256
`eb120c52550c310b177d1c012bf231baeb4f38cbe45c600cab792afbf523b74d`;
`NATIVE_CLEANUP.json` has SHA256
`1740893a91d6dcd263577f487fdc8d54db4ad422717b8cf7f725fda293e5c7a9`.
That closeout's preliminary chat-tail numbers are superseded by the seeded V2
audit above; its native surrogate observations remain unchanged.

The AIPerf audit receipts establish reconstruction and planning only. The native
closeout adds bounded structural/cache evidence. No AIPerf-aligned paired GPU
result, whole-root cache validation, or whole-corpus confidence follows yet.

[card]: https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/blob/23f152f6f0f9399a85901b89a6458def0ef16729/README.md
[registry]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/plugin/plugins.yaml#L2057
[defaults]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/common/environment.py#L295
[loader]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/dataset/loader/weka_trace.py#L1527
[roles]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/dataset/loader/weka_synth_buf.py#L261
[count]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/records/inference_result_parser.py#L314
[tutorial]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/docs/tutorials/weka-trace.md#L78
[input-config]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/common/config/input_config.py#L67
[timing-config]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/common/config/user_config.py#L110
[fixed]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/timing/strategies/fixed_schedule.py#L76
[clock]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/common/loop_scheduler.py#L177
[rate]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/timing/strategies/request_rate.py#L210
[callback]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/credit/callback_handler.py#L410
[branches]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/timing/branch_orchestrator.py#L723
[issuer]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/credit/issuer.py#L450
[scenario]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/common/scenario/inferencex_agentx_mvp.py#L7
[phase]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/timing/phase/runner.py#L177
[trees]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/timing/phase_orchestrator.py#L151
[bootstrap]: https://github.com/SemiAnalysisAI/aiperf/blob/0d2aa0572ac685943d38c580675c4a61023581d3/src/aiperf/common/bootstrap.py#L169
