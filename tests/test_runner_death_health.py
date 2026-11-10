# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A model runner killed by a signal is reported within seconds (#2507).

`AsyncIOProcManager.monitor_procs` waits on its runners' process sentinels.
Here two real child processes stand in for one EngineCore's runners; one is
killed with SIGKILL, as a device fault or the kernel's OOM killer kills one.
The report (`EngineCore._send_engine_dead` in a server, the SHUTDOWN frame
that turns the frontend's `/health` to 503) must come before `exit()`, which
in #2507 blocked on the surviving ranks while `/health` kept answering OK.
"""

import multiprocessing
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.model_engine.async_proc import AsyncIOProcManager


def test_a_sigkilled_runner_is_reported_before_cleanup_within_seconds():
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=time.sleep, args=(600,), name=f"ModelRunner{i}")
        for i in range(2)
    ]
    for proc in procs:
        proc.start()
    try:
        mgr = object.__new__(AsyncIOProcManager)
        mgr.procs = procs
        mgr._dead_ranks = {}
        mgr.label = mgr.runner_label = "test"
        reported = threading.Event()
        cleaned_up = threading.Event()
        order = []
        mgr.on_worker_failure = lambda: (order.append("report"), reported.set())
        # The cleanup that blocked in #2507: it never returns here.
        mgr.exit = lambda: (order.append("exit"), cleaned_up.set())
        mgr.monitor_procs()

        killed_at = time.monotonic()
        os.kill(procs[1].pid, signal.SIGKILL)

        assert reported.wait(timeout=30), "no report 30 s after the SIGKILL"
        assert time.monotonic() - killed_at < 30
        assert cleaned_up.wait(timeout=30)
        assert order == ["report", "exit"]
        assert mgr._dead_ranks == {1: -signal.SIGKILL}
        assert procs[0].is_alive(), "only the killed rank is reported"
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=10)
