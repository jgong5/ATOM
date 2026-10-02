**Developer halt: the standalone fix does not meet the exit criterion. Owner ruling needed.**

**Reproduced on the unfixed tip `f87413a7a`.** A real `Scheduler` with the `compass` connector as producer, driven in `EngineCore._process_engine_step_inner`'s order (schedule, then dispatch `connector_meta_output` to the worker, then poll `get_finished`, then postprocess; when `schedule()` returns `None`, poll only). There are 3 prefills of 40 tokens each (10 blocks each, block size 4), `max_tokens=1`, a 100-block pool, and the clock set 3600 s past issue so every priced deadline has passed. Run on node 18 `xiaobizh_n18_cpu` with the tree staged by `git archive`:

| run | tree | after 3 prefills: `len(deferred_free_blocks)` / free blocks | after all deadlines |
|---|---|---|---|
| idle (nothing else arrives) | `f87413a7a` (before) | 3 / 70 | **3 / 70** |
| idle | fix (below) | 3 / 70 | **3 / 70** |
| follower (one more request arrives on the next step) | `f87413a7a` (before) | 3 / 70 | 4 / 60 |
| follower | fix (below) | 3 / 70 | 1 / 90 |

**The fix tried** is the one the issue describes. The scheduler half's `request_finished` queues `seq.id -> list(seq.block_table)` when it is the producer, keyed by request so the second call replaces the first. `build_connector_meta` then announces each queued request through `add_new_req_to_save` and clears the queue. It frees the earlier batch as soon as a later step schedules something (follower row). It frees nothing when the finishing batch is the last work (idle row), and the follower request is itself held for the same reason.

**Why.** `build_connector_meta` reaches the worker only with a scheduled batch. `Scheduler.schedule()` returns `None` once `waiting` and `running` are both empty (`scheduler.py:1437-1438`; line numbers at `f87413a7a`), and EngineCore's idle path dispatches connector metadata only for offload connectors (`engine_core.py` `_dispatch_idle_offload_work`, gated on `is_offload`; `pp_engine_core.py` has the same gate). So an announcement queued in the last postprocess before idle is never sent. `deferred_free_blocks` keeps `is_finished()` false. The shutdown drain gives up after `KV_SHUTDOWN_DRAIN_TIMEOUT_S` (2.0 s) with a warning. Mooncake does not hit this because it announces the save at allocation (`update_state_after_alloc`). This connector cannot do the same, because it prices from the moment of announcement and would report `finished_sending` before the request is in `deferred_free_blocks`, which trips the assert at `scheduler.py:3044`.

**Candidate answers:**
1. **Land the fix as a partial repair and state the limit.** It frees every finished producer request once a later step schedules a batch. The last batch before idle stays held until the next request arrives, or until the shutdown drain times out. #466 would close the gap. File set unchanged (`connector.py` plus one test). The test would assert the follower row and pin the idle row as the known limit.
2. **Widen the file set to `engine_core.py` and `pp_engine_core.py`.** The idle path would dispatch connector metadata for any connector, not only offload ones. `connector_metadata_has_work` already drops empty metadata. This changes ATOM's engine loop for the simulation's sake (principle 1) and also changes the idle behaviour of Mooncake and MoRI-IO.
3. **Close #465 as superseded by #466.** Freeing from `process_completions` on the scheduler half runs on every poll, idle included (`_poll_kv_transfer_progress`, then `_update_from_kv_xfer_finished`), so #466 does not have this gap. That design needs the clock on the scheduler half, which is #466's rework and depends on #455 and #456.

The fix is committed on `compass/issue-465-producer-frees-blocks`, locally only. It has not been pushed, and no PR is open.
