Needs owner ruling: land a partial fix, widen the file set, or close as superseded by #466? `need human` applied.

The fix announces finished producer requests in `reqs_to_save` from `build_connector_meta`. That metadata only reaches the worker with a scheduled batch, so the last batch before idle is never freed. Repro at `f87413a7a` (3 prefills, all deadlines passed): free blocks 70 before and after the fix when idle; 60 before, 90 after when one more request follows.

(a) Land as partial repair; test pins the idle gap as a known limit. File set unchanged.
(b) Add `engine_core.py` and `pp_engine_core.py`: dispatch connector metadata on idle for every connector. Changes ATOM's engine loop and Mooncake/MoRI-IO idle behaviour.
(c) Close as superseded by #466, which frees from `process_completions` on every poll, idle included.

Recommend (c). Branch `compass/issue-465-producer-frees-blocks` is local only; no PR.
