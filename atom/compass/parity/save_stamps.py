# SPDX-License-Identifier: MIT
"""``COMPASS_SAVE_STAMPS=1`` (opt-in; a real run recording its steps): each worker stamps every KV offload copy its
connector runs, a save or a load (`OffloadWorkerMixin`'s save and load executors), as a line of
``<record>/save_stamps/<pid>.jsonl``: the kind, the request, the connector's ranks, and three monotonic
nanoseconds: ``t_submit_ns`` (the step handed it to the executor), ``t_start_ns`` (a worker thread began it) and
``t_end_ns`` (it returned or raised). Where a save waited behind another (one save worker), which a 1 s
OFFLOAD-SAVE-PROF line cannot tell, is then read off them. Each stamp wraps the submit and the job only: the job,
its arguments and its result are unchanged, and any failure to log is swallowed."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

ENV = "COMPASS_SAVE_STAMPS"
RANKS = ("rank", "tp_rank", "pp_rank", "dp_rank")


def _stamped_submit(owner, executor, path: Path, lock: threading.Lock):
    """`executor`'s ``submit``, whose job ``owner._guard(kind, fn, req, ...)`` is stamped."""
    submit = executor.submit
    ranks = {k: getattr(owner, k) for k in RANKS if isinstance(getattr(owner, k, None), int)}

    def stamped(fn, *args, **kwargs):
        t_submit = time.monotonic_ns()

        def job():
            t_start = time.monotonic_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                t_end = time.monotonic_ns()
                try:
                    kind = args[0] if args and isinstance(args[0], str) else getattr(fn, "__name__", "?")
                    req = args[2] if len(args) > 2 else None
                    line = {
                        "kind": kind,
                        "req": str(getattr(req, "req_id", req)),
                        "pid": os.getpid(),
                        **ranks,
                        "t_submit_ns": t_submit,
                        "t_start_ns": t_start,
                        "t_end_ns": t_end,
                    }
                    with lock, path.open("a") as out:
                        out.write(json.dumps(line) + "\n")
                except Exception:  # noqa: BLE001  the job never pays for the log
                    pass

        return submit(job)

    return stamped


def install(record, mixin=None) -> bool:
    """Under `ENV`, make each offload connector built from now on stamp its copies into `record` (the parity
    record's root); False when off. `mixin` is ATOM's `OffloadWorkerMixin` unless given."""
    if os.environ.get(ENV) != "1":
        return False
    if mixin is None:
        from atom.kv_transfer.offload._offload_common import OffloadWorkerMixin as mixin
    init = mixin._init_worker_common
    if getattr(init, "_compass_stamps", False):
        return True
    path = Path(record) / "save_stamps" / f"{os.getpid()}.jsonl"
    lock = threading.Lock()

    def stamped_init(self, *args, **kwargs):
        out = init(self, *args, **kwargs)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            for name in ("_save_executor", "_load_executor"):
                executor = getattr(self, name, None)
                if executor is not None:
                    executor.submit = _stamped_submit(self, executor, path, lock)
        except Exception:  # noqa: BLE001  a connector never pays for the stamps
            pass
        return out

    stamped_init._compass_stamps = True
    mixin._init_worker_common = stamped_init
    return True
