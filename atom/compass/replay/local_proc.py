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

**One executor, any logical width.** These are two numbers and the replay keeps
them apart:

- the *logical target TP*, ``config.tensor_parallel_size``, which is how wide
  the deployment being predicted is. It shards every weight, sizes every
  collective, places the LM head, decides the pool and block specs, and is what
  the run reports as the configuration it predicted. It comes from the target
  record, which came from a device, so it is source-derived and not a label.
- the *physical executor count*, ``config.tp_world_size``, which is how many
  processes hold a runner. For a GPU-free replay it is 1 -- ``Config`` collapses
  it there -- because there is no device for a second one to occupy and no
  forward for it to run: the step is priced, not executed.

Collapsing them the other way round is what must not happen. Running the one
executor and calling its work the group's would be a TP1 result wearing a TP4
label; so the single executor is rank 0 of the logical group, and the group's
cost is the oracle's answer at those rank coordinates plus the modelled
collectives, which is a cost-model question rather than a transport one. Nothing
here weakens that: the rank-aggregation and head-placement semantics are the
same ones a device-backed run uses, reached with the same logical width.

What this does refuse is a configuration whose parallelism genuinely needs more
than one *process* for reasons the cost model does not cover -- prefill context
parallel, pipeline stages -- because there the second process is not an executor
of the same step but a different stage of it.
"""

from __future__ import annotations

import logging

from atom.utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)

__all__ = ["LocalProcManager"]


class LocalProcManager:
    """``AsyncIOProcManager``'s interface, served from this process."""

    def __init__(self, finalizer, proc_num: int, runner: str, *args, **kwargs):
        config = args[0] if args else None
        self.logical_tp = int(
            getattr(config, "tensor_parallel_size", 1) or 1)
        if proc_num != 1:
            # `Config.tp_world_size` is already 1 under a replay, so the only
            # way to arrive here is a parallelism that multiplies it: prefill
            # context parallel, in `EngineCore`. Name what it actually is
            # rather than blaming the logical width, which is legitimately
            # wide and is not the problem.
            pcp = int(getattr(config, "prefill_context_parallel_size", 1) or 1)
            raise ValueError(
                f"ATOMCompass: a GPU-free replay runs one executor, and this "
                f"deployment asks for {proc_num} "
                f"(prefill_context_parallel_size={pcp}). A logical TP of "
                f"{self.logical_tp} is fine and is predicted from rank 0 plus "
                f"modelled collectives; context-parallel ranks are not, "
                f"because each holds a different slice of the same prefill and "
                f"no cost model here composes them. Replay with "
                f"prefill_context_parallel_size=1."
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
        logger.info(
            "%s: 1 executor for a logical TP%d deployment, runner constructed "
            "in-process as rank 0, no workers spawned",
            self.label, self.logical_tp)

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
