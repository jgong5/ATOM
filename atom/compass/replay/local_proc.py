"""The runner pool, collapsed into this process.

``AsyncIOProcManager`` spawns one OS process per rank, hands each a shared
memory queue, and dispatches RPCs by broadcasting a ``(name, *args)`` tuple.
Every caller in the engine reaches it through one method -- ``call_func`` -- so
that is the whole seam, and this is the same seam with the transport removed.

**What this bypasses, explicitly.** Worker processes, the shared-memory RPC
broadcast, the ZMQ output sockets, and the KV-output aggregation across ranks.
Those are transport, not policy: the scheduler still schedules, the block
manager still allocates, the sequence lifecycle and output contract are
untouched. But they are not free in a real deployment, and a replay that
silently drops them under-reports serving overhead. A replayed step therefore
costs what the oracle says the forward costs and nothing for the crossing;
whatever the crossing is worth has to be modelled, not assumed to be zero.
`CompassModelRunner` already records `gap_seconds` on a measured run for exactly
this comparison.

**One rank.** Collapsing N ranks into one process would mean either running N
runners with no collectives between them -- which deadlocks the moment anything
synchronises -- or pretending a rank's work is the group's. Neither is honest,
so this refuses. A multi-rank configuration is predicted from a rank-0 capture
plus a modelled collective, which is a cost-model question, not a transport one.
"""

from __future__ import annotations

import logging

from atom.utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)

__all__ = ["LocalProcManager"]


class LocalProcManager:
    """``AsyncIOProcManager``'s interface, served from this process."""

    def __init__(self, finalizer, proc_num: int, runner: str, *args, **kwargs):
        if proc_num != 1:
            raise ValueError(
                f"ATOMCompass: GPU-free replay runs one rank in one process, "
                f"and this deployment asks for {proc_num}. Ranks of a parallel "
                f"group synchronise with each other; there is no honest way to "
                f"run them in a single thread. Replay TP=1 and predict the "
                f"wider configuration from it."
            )
        self.parent_finalizer = finalizer
        self.proc_num = proc_num
        self.runner_label = runner.split(".")[-1]
        self.label = f"LocalProcManager({self.runner_label})"
        self.still_running = True
        self.keep_monitoring = False
        # Named `procs` because EngineCore's monitor reads it. Empty because
        # there are none, which is the property being demonstrated.
        self.procs: list = []
        runner_class = resolve_obj_by_qualname(runner)
        self.runner = runner_class(0, *args, **kwargs)
        logger.info("%s: runner constructed in-process, no workers spawned",
                    self.label)

    def call_func(self, func_name: str, *args, wait_out: bool = False):
        """Dispatch by name, exactly as the broadcast queue does.

        ``wait_out`` is honoured by returning rather than by waiting: the call
        already happened. Callers that pass ``wait_out=False`` are firing and
        forgetting, and get the same behaviour they would get from a queue --
        the work is done, the result is dropped.
        """
        fn = getattr(self.runner, func_name, None)
        if fn is None:
            raise AttributeError(
                f"{self.label}: the engine called {func_name!r} and the runner "
                f"does not implement it. A replay runner has to answer every "
                f"RPC the engine makes, or the engine hangs on a reply that "
                f"never comes."
            )
        result = fn(*args)
        return result if wait_out else None

    def call_func_with_aggregation(self, func_name: str, *args,
                                   timeout: float = 10.0):
        raise NotImplementedError(
            f"{self.label}: KV-transfer aggregation is a multi-rank, "
            f"multi-process path and PD disaggregation is out of scope for the "
            f"PoC. Nothing should reach here with one rank and no connector."
        )

    def process_output_sockets(self, output_address: str):
        """No socket to drain -- ``call_func`` returns the result directly."""

    def monitor_procs(self):
        """Nothing to monitor. A worker that cannot die cannot die unnoticed."""

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        exit_fn = getattr(self.runner, "exit", None)
        if exit_fn is not None:
            try:
                exit_fn()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.exception("%s: runner exit failed", self.label)
        logger.info("%s: shutdown", self.label)
        self.parent_finalizer()
