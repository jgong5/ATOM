# Adjusting a calibration when the deployment changes

A calibration design is a config in this directory, generated into traces by
`scripts/compass/gen_v1.py`, rehearsed and judged on a law by `scripts/compass/plan_v1.py`
(both through `scripts/compass/dry_v1.sh`), recorded by `scripts/compass/cal_v1.sh` and
checked against its plan by `scripts/compass/check_v1.py`. The library is
`atom/compass/calib/`: `deploy` (the config and the deployment), `design` (the segment
generators), `engine` (the rehearsal), `ident` (identifiability), `columns` (a step's column
values), `check` and `report`. The method is wuhuikx/LLM-Booster#739 sections 2 to 4.

A design follows its deployment. Its deployment block is read from a record
(`deploy.from_record`, `deploy.deployment_of`), and its segments state their sizes relative to
that deployment (`deploy.count`, `deploy.tokens`). A change to the deployment is then a new
smoke record and a regenerated design, with no edit to the config. What follows says when a
change also needs a new calibration record, which variables move which parts of a design,
the steps with their commands, what each dry-run failure asks for, and what never changes.

## 1. Which variables re-key the laws and the readings

Three artifacts depend on the deployment, each through its key
(`atom/compass/artifacts/keys.py`, `OWN_FIELDS`):

- **The step law** (`Kind.STEP_LINEAR_LAW`) is keyed by the model, an engine config, the ATOM
  source and the form. The engine config is `atom/compass/run/step_grid.py` `keyed`: the
  speculative method and its draft width (and `DRAFT`, a drafter checkpoint of its own), the
  capture ladder (`atom/compass/runner/overrides.py` `schedulable_ladder`), each of
  `OFF_DEFAULT` off its default (MoE backend, all-to-all backend, EPLB, bounded SWA replay,
  index cache dtype), `LEVEL`, indexer-only context parallelism, `hf_overrides`, the KV block
  size and dtype, the quantization, the TP and DP widths, DP attention, expert parallelism,
  two-batch overlap, and the performance environment (`keyed_env`).
- **The host laws** (`Kind.HOST_LAWS`) are keyed by the host CPU, the ATOM source, the model
  and the same engine config (`atom/compass/run/__init__.py` `host_key_of`).
- **The memory readings** (`Kind.MEMORY_READINGS`) are keyed by the model, the layout
  (`LAYOUT`: the rank widths, `SWITCHES` and `decode_context_parallel_size`) and an engine
  config that adds to `keyed` the scheduler's bounds (`atom/compass/backends/step_grid_plan.py`
  `BOUNDS`: max-num-seqs, max-num-batched-tokens, max-model-len) and the KV offload
  (`atom/compass/run/memory.py` `key_fields`).

So:

- A change to anything `keyed` states needs a new calibration record under the new
  deployment, new step and host laws fitted from it, and new memory readings.
- A change only to `BOUNDS` or `LAYOUT` needs new memory readings. The laws still answer,
  but the design moves with it: regenerate and dry-run it, and record again if a vector the
  old record could not reach (a new widest rung) now has to be fitted.
- A change to neither (the GPU memory fraction, the forced acceptance length, the prefill
  chunk, the state checkpoint interval) re-keys nothing, but the design reads it: regenerate
  and dry-run, and record again only if the dry run on the current law fails.

## 2. Each variable: what is derived, what to edit, what it moves

"Derived" means `deploy.from_record` reads it from a record (its run.json engine over its
`steps/dp0.engine.json`, its serve command line, its model's config.json). A field stated in
the config's deployment block beside `from_record` overrides the record; `gen_v1.py --record`
(or `RECORD=` for `dry_v1.sh`) instead replaces every field the record states. "Edit" names
the config field to set by hand when no record states it.

| Variable | Derived | Edit | What it moves in the design | GPU time | Re-keys |
|---|---|---|---|---|---|
| Capture sizes (`--cudagraph-capture-sizes`) | `rungs`, from the engine's `ladder` | `deployment.rungs` | One decode vector a rung (`ident.vectors`); every drain dwells at each rung it passes for `acceptance.rung_steps` steps (`design.dwell`); a rung the law lacks is rehearsed on its nearest rung's prices (`engine.proxied`, named in plan.md) | Up with each rung added, down with each removed | Step law, host laws, readings |
| `--max-num-seqs` | `max_num_seqs` | `deployment.max_num_seqs` | Every `seqs_share` count: a drain's rows, a group's total, a `rows` segment's concurrency (`deploy.count`); the widest rung a drain reaches; the eager rows past the widest rung | Up with more rows to prefill before each drain and with the rungs they reach | Readings (`BOUNDS`); the laws only through the ladder |
| TP, DP, EP (`-tp`, `--enable-dp-attention`, `--enable-expert-parallel`) | `dp`, `tp` | `deployment.dp`, `deployment.tp` | `"dp": true` counts (group totals) and unpinned lanes, which split over the ranks; a segment whose `min_dp` exceeds `dp` is skipped (`design.skipped`, listed in plan.md); the MoE gather columns (`moe_rows`, `moe_tokens`, `peer_tokens`, `cold_peer_rows`), which read zero at one rank; at one rank `graph_padding` and `row_padding` alias (section 4). TP moves nothing the rehearsal forms: the law prices it | Up with more DP ranks (group segments grow), down to one rank (multi-rank segments skip) | Everything |
| `--decode-context-parallel-size` | `dcp`, from the serve command line | `deployment.dcp` | Not modelled by the rehearsal. Its per-sequence cost is the decode column `rows` (`columns.PROVISIONAL`), a decode variant the config lists | None from the design | Readings (`LAYOUT`); the law through `keyed`'s indexer-only DCP alone |
| `--max-num-batched-tokens`, `--attn-prefill-chunk-size` | `max_num_batched_tokens`, `chunk` | `deployment.max_num_batched_tokens`, `deployment.chunk` | Every `budget_share` size (quad scales, backlog short prompts, single-row prefills); how long prompts chunk (`engine.Engine._prefill_step`) and how far a drain lags a peer still prefilling (`design.drain`); the levels of `tokens` and `query_square`. The chunk is stated but not modelled | A larger budget means fewer, longer prefill steps | Readings (`BOUNDS`) |
| KV capacity (`--kv_cache_dtype`, `--gpu-memory-utilization`, `--max-model-len`) | `kv_tokens` (the run's KV blocks times the block), `max_model_len` | `deployment.kv_tokens` (a number or `{"run_json": PATH}`), `deployment.max_model_len` | Every `kv_share` and `len_share` size: the long-context drain, the anchors, the backlog and trickle histories (`deploy.tokens`); the decode context range the vertices span (`ident.decode_vertices`) | Up with a larger pool (longer contexts prefill longer) | KV dtype: everything. Model length: readings. GPU memory fraction: nothing |
| `--block-size`, `--state-checkpoint-interval-tokens` | `block_size`; `prefix_unit`, the checkpoint interval for a model with KDA layers, else the block | `deployment.block_size`, `deployment.prefix_unit` | Where a later turn's prefix hit ends (`Deployment.cached`), and with it `query_cached` and `history` | Up with a coarser hit unit (more of each history recomputed) | Block: everything. Interval: nothing |
| Speculative decoding (`--method`, `--num-speculative-tokens`, `--spec-decode-acceptance-length`) | `drafts`, from the engine's `speculative`; `acceptance`, from the command line, else one plus the mean of `dp0.acceptance.json` | `deployment.drafts`, `deployment.acceptance` | The tokens a decode row verifies (`Deployment.query`, so `moe_tokens`); each row's output for its departure step (`design.out_for`); the draft's prices. Off is `drafts` zero | Drafts: up with each draft (each step's device time). Acceptance: drains keep their steps, since departures are counted in steps | Method and drafts: everything. Acceptance: nothing |
| MoE routed experts, top-k | `moe_routing`, from the model's config.json (`deploy.EXPERTS`, `deploy.TOP_K`) | `deployment.moe_routing` | The `moe_experts` column (a decode variant) and how early it saturates across the rungs | None | Everything (the model) |
| A new layer kind (KDA, ...) | Only its prefix unit | `columns.decode`, `columns.prefill`, `columns.<kind>_variants` | The columns the backend prices for it, each checked as a variant. A column neither this tree's backend prices (`columns.backend_prices`) nor `columns.PROVISIONAL` defines is reported as an error for that variant until it does; the rehearsal's views carry only the counts `BatchView` states | By the segments that exercise it (for state checkpoints: quad scales and trickle deltas around `prefix_unit`) | Everything (the model) |

## 3. The steps, with their commands

`$T` is this tree, `$S` the store, `SERVE_ARGS` the new deployment's ATOM arguments.

1. **Smoke record under the new deployment.** It holds the engine state the design reads.
   GPU work holds a gpool lease for its duration (`gpool acquire -g N -p "<what>"`, then
   `gpool release LEASE --note "<result>"`):
   ```
   $T/scripts/compass/compass record --gpu G --store $S --out SMOKE -- SERVE_ARGS &
   HARNESS_PYTHON=... TRACES=... MODEL=... $T/scripts/compass/client.sh SMOKE --traces 0-3 --concurrency 4 --duration 300 --stop
   ```
   If the memory readings re-key (section 1), measure them into the store under the same
   lease (as `scripts/compass/calibrate.sh` does):
   `$T/scripts/compass/compass measure memory --gpu G --store $S --out MEM -- SERVE_ARGS`.
2. **Generate.** The config is unchanged. The deployment comes from the smoke record:
   ```
   PYTHONPATH=$T python3 $T/scripts/compass/gen_v1.py CONFIG DESIGN --record SMOKE
   ```
3. **Dry run** on the nearest law, meaning the current deployment's or one fitted on the
   smoke record, and optionally on the target workloads' records:
   ```
   RECORD=SMOKE TARGET="REAL1 REAL2" $T/scripts/compass/dry_v1.sh CONFIG LAW_DIR OUT
   ```
   It exits 0 when every vector passes and 1 when one fails.
4. **Read the report**, `OUT/design/plan.md`: the verdict, each failing vector and why, the
   rungs priced by proxy, the skipped segments, and the predicted GPU minutes. Fix a failure
   as section 4 says, in the config's segments, never in the acceptance block, and dry-run
   again.
5. **Freeze the criteria.** Commit the config as it passed. Its acceptance block and column
   set are fixed from here until the fit is graded. `DESIGN/design.json` holds the deployment
   as plain fields.
6. **Record** on one server, under a gpool lease:
   ```
   MODEL=... HARNESS_PYTHON=... $T/scripts/compass/cal_v1.sh OUT/design REC record --gpu G --store $S -- SERVE_ARGS
   ```
7. **Check** the record against its plan:
   ```
   COMPASS_PREFILL_LABEL=record PYTHONPATH=$T python3 $T/scripts/compass/check_v1.py OUT/design REC --law LAW --host HOST
   ```
   A combination far short of the plan, or a vector that fails on the record's own steps,
   goes back to step 4.
8. **Fit**, with every lane's client run:
   ```
   COMPASS_PREFILL_LABEL=record $T/scripts/compass/compass fit steps --backend step-linear --form history \
       --model MODEL REC/real --client-run REC/real REC/<segment>/<lane>/artifacts/client_run.json ... --store $S --out L/steps
   $T/scripts/compass/compass fit host REC/real --client-run ... --store $S --out L/host
   ```
9. **Graded simulations** of the target workloads on the new laws, then grading:
   ```
   $T/scripts/compass/compass run --step-linear LAW --host-laws HOST --device D --ptl P --store $S --out SIM -- SERVE_ARGS &
   HARNESS_PYTHON=... $T/scripts/compass/client.sh SIM --agentx --concurrency C --duration D --stop
   PYTHONPATH=$T python -m atom.compass.validate CELL...
   ```

## 4. Each dry-run failure and what answers it

plan.md names a failing variant's reasons (`ident._verdict`); a vector passes when one of
its variants does.

- **A rung under `acceptance.rung_steps` GPU-bound steps.** Each drain dwells at every
  rung it passes, so a short rung means its steps were cut. Either the host's step outlasted
  the device's (a launch-exposed decode), or the drains never reach the rung (rows below
  it). For the first, lengthen the drains' contexts (raise a `kv_share` or `len_share`). For
  the second, raise the drains' `seqs_share`, or check `deployment.rungs` against the
  ladder.
- **VIF over `acceptance.vif`.** The `vif` entry names the columns; find the pair that
  moves together and add steps where one moves without the other. Examples:
  - `query_square` against `tokens`: single-row prefills across the budget (a `rows`
    segment at `per_rank` concurrency with `budget_share` prompts).
  - `history` against `query_cached`: prefills on deep histories with and without a hit
    (`then` turns in the long drains, trickle sessions).
  - A MoE column against `peer_tokens`: drains where peers hold rows apart from the busy
    rank (`hold` lanes, `min_dp`).

  A variant whose columns no design separates is dropped from `<kind>_variants`. The
  form-selection rule then never chooses it.
- **G over `acceptance.g`.** `g_vertex` names the worst point of the reachable range. Add
  steps near it: a long-context decode vertex calls for a larger `kv_share` in the
  long-context drain; a prefill vertex on a deep history, for trickle or backlog histories
  nearer it (`len_share`); a many-row prefill, for a `rows` segment. More steps of the same
  kind (more `requests`, more `turns`) lower G slowly. Steps nearer the vertex lower it
  fast.
- **Fewer than `acceptance.levels` levels.** The column takes too few values. Spread it:
  give a prompt a `[lo, hi]` range, add quad scales, vary `then`. An indicator, which the
  range holds at 0 and 1 alone, is exempt. A MoE column that saturates at the high rungs
  fails there by construction: that rung's other variants carry it.
- **One DP rank: `graph_padding` and `row_padding` aliased.** Reported as aliased (rule
  2.4.6). Estimability is checked and the rung is judged by G, so it is no failure.
  Dropping `row_padding` from `columns.decode` fits the two as one column.
- **A vertex or variant reported as an error.** A column neither the backend nor
  `columns.PROVISIONAL` counts. Price it in the backend or define it provisionally, or drop
  the variant.
- **Views the law refused, or rungs priced by proxy.** The law lacks a vector the
  deployment needs. Proxy prices (`engine.proxied`) leave rank, VIF and levels exact and
  make GPU minutes and the GPU-bound cut approximate. For a firm prediction, fit a law on
  the smoke record first.
- **Over the GPU budget.** plan.md gives each segment's span. Shorten the costliest:
  - fewer `requests` in a `rows` segment, fewer `turns` in a trickle;
  - a smaller `kv_share` in the long-context drain (which raises G at long context: re-check
    it);
  - drop an anchor `copy` or a segment whose vectors pass without it (dry-run without it).

  Never lower `rung_steps`.

## 5. What never changes

- The acceptance criteria, `deploy.ACCEPTANCE` (`vif`, `G_PERCENT`, `g_z`, `rung_steps`,
  `levels`), are frozen before a record (#739 section 2.2). A config may tighten them,
  never loosen them to pass.
- The form-selection rule (#739 section 2.3). The config lists the candidates
  (`<kind>_variants`, `ident.variants`); the fit chooses among them by the frozen rule, never
  by which one the dry run liked.
- The noise a step (`noise`) is measured on the hardware, not tuned to pass G.
- Segments are chained by events, never by a predicted time (`plan_v1.py` `EVENT`), and the
  server's routing environment is `plan_v1.py` `SERVER_ENV`.

## 6. What a deployment change does not make mechanical

- **The model.** A new model needs its own config: its column set, its noise, its
  `trace_model`, and the segments its layer kinds call for.
- **A pinned lane names a rank.** `min_dp` skips a segment the group is too small for, but
  a group larger than any pin leaves the extra ranks idle in that segment. Spreading a
  segment over them is an edit.
- **Plain numbers stay plain.** The short decode prompts, the anchors' row counts, the
  short-prompt ranges and the request counts do not scale with the deployment. They are
  sized for the columns' ranges, not for the deployment's bounds.
- **The dry run needs a law.** One fitted for another deployment prices new rungs by proxy
  and refuses what it has no vector for. A smoke record needs GPUs.
