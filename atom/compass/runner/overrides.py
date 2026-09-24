# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The model-runner methods that keep construction off the device.

These are the memory-owning methods of ATOM's `ModelRunner`: the ones that read
a checkpoint, warm the model, size the KV pool, allocate it, capture graphs, and
run a step. They live here rather than beside the subclass because this module
imports nothing from the engine at module scope, and so can be executed on a
machine with no driver.

Two of them run during construction (`_build_and_load_model`, `_maybe_warmup`)
and decline to do their work. The rest run afterwards, over the worker's RPC
channel; `allocate_kv_cache` does the arithmetic and none of the allocation,
`capture_cudagraph` captures nothing and says so in the shape its caller
unpacks, `forward` reports what a step produced without running one, and
`get_num_blocks` runs ATOM's own budget arithmetic against readings taken off a
machine spec instead of off a card.

That last one replaces no arithmetic, and the two methods beside it are the
reason it does not have to. `ModelRunner.get_num_blocks` touches the device in
exactly two calls -- `_read_device_memory` and `_estimate_cudagraph_overhead`
-- and both are override points, so replacing them leaves every line of the
budget formula, the clamp and `plan_pools` to run unchanged. A second copy of
that formula would drift from it with no test failing; a substituted reading
cannot.

Order matters when mixing this in: `NonAllocatingRunner` must precede
`ModelRunner` in the bases so these definitions win. The class deliberately
defines no `__init__`. The base class runs the whole of its own `__init__`
before a subclass body would get control, so anything these methods read has to
come from `self.config`, which the base sets first.

## The reply contract, and why it is not a matter of taste

A worker runs `AsyncIOProc.busy_loop`, which takes a name off a shared-memory
ring, resolves it with `getattr(runner, name, None)`, calls it, and forwards the
result **only when it is not None** (`async_proc.py:231-252`). The caller's side
is `AsyncIOProcManager.call_func`, whose `wait_out=True` form blocks on
`self.outputs_queue.get()` with no timeout. Three consequences, none of which a
single-process test can show:

* **A name the runner does not have is skipped, not raised.** `getattr` returns
  None, the loop moves on, the worker stays healthy, and a caller that asked for
  the reply waits for the life of the process. Absence is the quietest failure
  on this surface, which is why `RPC_SURFACE` below names every dispatched
  method rather than leaving it to whatever the class happens to inherit.
* **A method that is present and answers None parks its caller in exactly the
  same way.** This is one line below the skip, not a separate mechanism:
  `async_proc.py:243` is `if out is not None:`, and **both** of the loop's
  `put_nowait` calls -- the primary output queue at `:248` and the KV queue at
  `:250` -- are inside it, while the `getattr` skip is at `:237-239`. So from
  the caller's side an answer of None and a method that was never defined are
  one event: nothing is queued, nothing logs, and no timeout ends either wait.
  Naming only absence is worse than saying nothing, because it sends whoever
  is debugging the hang to check whether the method is there, find that it is,
  and stop. The shapes here are therefore checked against what the call site
  unpacks, not against what reads well -- returning None is the case a
  plausible stub falls into by accident, which is why every name on the
  surface is either replaced here or left to a base implementation that is
  known to end in a value.
* **Raising is the loud option, and it is louder in the worker than in the
  parent.** An exception leaves `busy_loop`, kills the worker, and the
  manager's process monitor turns that into a `SystemExit` on the output queue,
  which `call_func` re-raises in the caller. What crosses the boundary is the
  type and nothing else: the parent gets a bare `SystemExit()` with empty args,
  no message and no `__cause__`, so a refusal's name and its reason exist only
  on the worker's stderr. `SystemExit()` also carries `code=None`, and
  `engine_core.py:132` sits in a `try/finally` with no `except`, so letting it
  propagate ends the parent process with status 0 -- a clean shutdown to any
  supervisor reading exit codes. So a refusal reaches the caller and a silence
  never does, but "reaches" means the fact of it and not the reason for it: a
  successor that wants its refusal diagnosable in the engine's own log has to
  put it there itself, on the worker side, before it raises.

`RPC_SURFACE` records, for each dispatched name, whether its caller waits for
the reply. The names themselves are not a list anyone typed: they are the ones
`engine_core`, `pp_engine_core` and `engine_utility` broadcast that a
`ModelRunner` answers, and the tests derive that intersection from ATOM's own
source and compare it with this table.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from atom.compass.memory import EAGER_SOURCE, DeviceReadings, SizedKVPool
from atom.compass.runner.step_output import (
    DeferredTokenStream,
    reported_token_id,
    reports_previous_step,
)

logger = logging.getLogger(__name__)

# Every name broadcast to a worker whose runner is a `ModelRunner`, and whether
# the caller blocks on the reply. True means a None return, or a missing method,
# parks that caller; False means a reply is not read and lands on the output
# queue for whoever asks next. A call site is named beside each one -- but what
# a site fixes varies, and the difference matters: where the comment says the
# caller unpacks, subscripts or reads the reply, the site fixes the shape; where
# it says the reply is forwarded or dropped, the site fixes only that something
# non-None and picklable has to arrive, and any richer shape is a convention
# inherited from the base implementation rather than a requirement.
#
# Scope: a runner reached through `EngineCore`. Nothing else in the tree says
# so, and it is load-bearing. Fourteen dispatched names are absent from this
# table because `ModelRunner` does not define them -- seven belong to
# `RapidServeModelRunner`, seven to the rollout extension. The rollout seven are
# unreachable here. The RapidServe seven are kept out by `Config`, not by this
# module: `enable_rapidserve` picks `PrefillEngineCore` / `DecodeEngineCore`
# (`llm_engine.py:140`), those classes broadcast all seven with `wait_out=True`
# without consulting `runner_qualname`, and the substitution that installs a
# RapidServe runner (`config.py:1730-1736`) fires only while `runner_qualname`
# is still ATOM's default -- which Compass overwrites. `Config` therefore raises
# `ValueError` for `enable_rapidserve=True` with any runner not in
# `RAPIDSERVE_RUNNERS`, this one included.
#
# Also outside the table, and outside anything a broadcast-derived enumeration
# can see: three of these twelve are called in-process on the runner itself,
# reached over the `resume_memory` RPC. `rollout/memory_manager.py:176`
# subscripts one key of `get_num_blocks`, `:183` discards `allocate_kv_cache`,
# and `:209` calls `capture_cudagraph` without unpacking it, inside a `try` that
# degrades to `enforce_eager=True`. Different arities, and the one place in the
# tree where a refusal from this module would be caught rather than fatal.
RPC_SURFACE: dict[str, bool] = {
    # Replaced here, in this module.
    "get_num_blocks": True,  # engine_core.py:132-141 reads four keys off a dict
    "allocate_kv_cache": True,  # engine_core.py:142-145 asserts the reply
    "capture_cudagraph": True,  # engine_core.py:149 unpacks three values
    "forward": True,  # scheduler.py:2435-2542 reads nine attributes off it
    # Answered by ATOM's own, which needs neither weights nor a device for them.
    "dummy_execution": True,  # engine_core.py:749 returns it to its own caller
    "exit": False,  # engine_core.py:260, the last call of the process's life
    "freeze_gc_heap": True,  # engine_core.py:203-206 catches a raise, not a None
    "process_kvconnector_output": False,  # engine_core.py:500 does not wait
    "async_proc_aggregation": True,  # engine_core.py:488, the one bounded wait
    "start_profiler": True,  # engine_utility.py:252 forwards the reply unread
    "stop_profiler": True,  # engine_utility.py:264 forwards the reply unread
    "flush_pp_send": True,  # pp_engine_core.py:81 waits before the next send
}


def unanswered_rpc_names(runner: Any) -> tuple[str, ...]:
    """The dispatched names `getattr` would answer with None for this runner.

    Drawn from all twelve, which do not fail the same way, so this return
    carries no one story about its names. `RPC_SURFACE[name]` records whether
    any caller reads the reply, and this function's only caller in the package
    partitions the result on exactly that before reporting it: a waited name
    parks its caller on an unbounded queue read for the life of the process,
    while `exit` and `process_kvconnector_output` are read by nobody and a
    hole in either parks no one.

    Checked where the class is composed instead of being discovered by a
    deployment, because the worker raises on none of them.
    """
    return tuple(name for name in RPC_SURFACE if getattr(runner, name, None) is None)


class RunnerRefusal(RuntimeError):
    """Raised where this runner has no answer and will not invent one."""


def install_device_readings(runner: Any, readings: DeviceReadings) -> None:
    """Give a runner the readings its KV budget will be computed from.

    Set on the instance rather than taken in an `__init__`, because the base
    class runs the whole of its own before a subclass body would get control.
    It is a function rather than a method so that every method on
    `NonAllocatingRunner` is a replacement for one of ATOM's, which is a
    property the tests check rather than a preference.

    Reading a machine spec off a flag or a config field, and building the
    readings from it, is the configuration surface's job and is not wired yet.
    Until it is, whoever has the spec installs the readings here, and
    `get_num_blocks` refuses by name in the meantime rather than sizing a pool
    from nothing.
    """
    if not isinstance(readings, DeviceReadings):
        raise RunnerRefusal(
            "the KV budget is sized from the five substituted readings, and "
            f"{type(readings).__name__} is not them; install what "
            "`atom.compass.memory.device_readings` returned for this width"
        )
    runner.compass_readings = readings


def _installed_readings(runner: Any) -> DeviceReadings:
    """The installed readings, or a refusal that says what would supply them."""
    readings = getattr(runner, "compass_readings", None)
    if readings is None:
        raise RunnerRefusal(
            "no device readings are installed on this runner, so there is "
            "nothing for ATOM's KV budget arithmetic to run against; call "
            "`install_device_readings` with what "
            "`atom.compass.memory.device_readings` produced from a machine "
            "spec before the engine asks this worker for a block count."
        )
    return readings


def _config_field(runner: Any, name: str) -> Any:
    """A config field read with no default; a missing one is refused by name."""
    try:
        return getattr(runner.config, name)
    except AttributeError:
        raise RunnerRefusal(
            f"ATOM's config has no field {name!r}; this runner reads it with "
            "no default, so a config that lacks it is refused by name rather "
            "than read as if the field were unset."
        ) from None


class UnbuiltModel(torch.nn.Module):
    """Stands in for the module tree a non-allocating runner never builds.

    It registers no parameter and no buffer, so it costs nothing on any device,
    and calling it raises instead of returning something a caller could mistake
    for a forward pass.
    """

    def __init__(self, model_class: Any) -> None:
        super().__init__()
        self.model_class_name = getattr(model_class, "__name__", repr(model_class))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RunnerRefusal(
            f"{self.model_class_name} was never built: this runner predicts what a "
            "forward pass costs instead of running one, so there are no weights "
            "to call."
        )


class NonAllocatingRunner:
    """Overrides that construct a model runner without touching device memory."""

    def _build_and_load_model(self, model_class: Any) -> None:
        """Build nothing and read no checkpoint.

        The base class constructs the model with the default device set to this
        rank's GPU and then fills it from disk. Both halves are skipped here, so
        no weight byte reaches the device and no checkpoint is read.

        A speculative config is refused here, because on the last pipeline rank
        the base goes on to build the drafter on this rank's GPU and load its
        checkpoint. On any rank, it cannot be modelled: a predicted step's reply has
        `draft_token_ids` None and zero `num_rejected`/`num_bonus`, which the
        scheduler accepts as a step that drafted nothing (`scheduler.py:2579`
        never fills `seq.spec_token_ids`, and `:2521-2522` reads the zeros), so
        a caller that asked for speculation would get a prediction with it off
        and no error.
        """
        self.model = UnbuiltModel(model_class)
        # Cleared on the way out, as both of ATOM's own implementations do: the
        # caller set the default device to this rank's GPU before calling, and
        # the code that runs next is written against a cleared default.
        torch.set_default_device(None)
        if _config_field(self, "speculative_config") is not None:
            raise RunnerRefusal(
                "a predicted step drafts no tokens, so reporting one under a "
                "speculative config would model speculation as off and say "
                "nothing about it; speculative decoding has no step semantics "
                "here yet."
            )
        logger.info(
            "%s not built and no checkpoint read; no weight bytes on the device.",
            self.model.model_class_name,
        )

    def _maybe_warmup(self) -> None:
        """Skip warmup.

        Warmup runs a forward over a dummy batch against real weights. There are
        none here, so there is nothing to warm. Skipping is also what lets
        construction finish at all: the base class warms the model from inside
        its own `__init__`, and the forward that warmup drives is this class's,
        which refuses.
        """
        return

    def _read_device_memory(self) -> Any:
        """Answer the four device figures off the machine spec.

        ATOM's own reads this rank's allocator and the driver. These four come
        from the device model instead, so every line of budget arithmetic below
        the call runs unchanged against a card this host does not have -- which
        is the whole of the substitution, and the reason no formula is copied
        anywhere in this package.

        The named tuple is ATOM's, imported at call time so this module stays
        importable where there is no driver, and built by keyword: four
        same-typed integers handed over positionally are a reordering nobody
        would see.
        """
        from atom.model_engine.model_runner import DeviceMemoryReadings

        readings = _installed_readings(self)
        return DeviceMemoryReadings(
            free=readings.free.total,
            total=readings.total.total,
            peak_torch=readings.peak_torch.total,
            non_torch=readings.non_torch.total,
        )

    def _estimate_cudagraph_overhead(self) -> int:
        """The graph-pool bytes the budget holds back, off the machine spec.

        ATOM's own derives this from the gap between the allocator's warmup
        peak and its steady state, which is two more device reads, so it is the
        second of the two places `get_num_blocks` touches the device and the
        second thing replaced here.

        The reading installed for it is the one `memory.graph_pool.reserves()`
        produces -- the mirror of ATOM's estimator rather than the measured
        pool, because this is the number that actually reserves and the two
        disagree; `memory.device_readings` refuses the other one by name at the
        call site. Two adjustments inside ATOM's estimator are not mirrored and
        move the block count in opposite directions, and `reserves()`' own
        docstring names both. Both need a drafter, and this runner never has
        one: `_build_and_load_model` refuses a speculative config before the
        base can build a drafter, so construction fails before
        `get_num_blocks` can run.

        `enforce_eager` is reconciled here rather than trusted. ATOM's own
        returns zero under that flag (`model_runner.py:1570-1571`) and this
        returns whatever was installed, so correctness would otherwise rest on
        whoever built the reading having passed the same flag this runner is
        configured with. A reading built for a capturing deployment, installed
        on a runner told not to capture, holds back bytes ATOM would not and
        moves the block count with nothing to show for it. The eager branch
        names the config field as the source of its only term, so the reading
        says which deployment it was built for and a disagreement declines.
        """
        reading = _installed_readings(self).cudagraph_overhead
        built_eager = any(term.source == EAGER_SOURCE for term in reading.terms)
        configured_eager = bool(_config_field(self, "enforce_eager"))
        if built_eager != configured_eager:
            raise RunnerRefusal(
                "the installed graph-pool reading was built for a deployment "
                f"with enforce_eager={built_eager} and this runner is "
                f"configured with enforce_eager={configured_eager}; ATOM "
                "reserves nothing for a graph it never captures, so one of "
                "the two would move the block count by the whole of the "
                "reservation without saying so."
            )
        return reading.total

    def get_num_blocks(self) -> dict[str, object]:
        """Size the KV pool with ATOM's own arithmetic over substituted readings.

        The base method is readings and then arithmetic: the utilisation
        budget, the 2% safety margin, `_kv_budget_extra_reserve`, the
        `min(budget, free)` clamp, `plan_pools`, and under pipeline parallelism
        a minimum across stages. None of that is replaced, and none of it is
        written down anywhere in this package. The two methods above are, and
        `super()` does the rest.

        The reply is the base's, forwarded unaltered, and
        `engine_core.py:132-141` fixes its four keys: `num_kvcache_blocks`
        (`:133`) and `state_runtime` (`:141`) are subscripted, `pool_entries`
        (`:139`) and `pool_entries_per_req` (`:140`) are taken with a `{}`
        default. The block count goes on to `BlockManager`, which asserts it is
        greater than zero (`block_manager.py:78`); the base asserts the same
        thing first and prints the whole budget with it. `state_runtime` is not
        an opaque value: `:141` hands it to `StateRuntime.from_wire`
        (`state_runtime.py:159-166`), which raises `TypeError` unless it is a
        `Mapping` and `ValueError` unless its key set is exactly
        `{"transfer", "checkpoint_spec"}`. Building that dict here instead of
        forwarding the base's would put all of those failures on the first RPC
        of the engine's life, so it is not built here.

        What this adds is one record and no arithmetic. The count the scheduler
        receives is an integer with nothing attached, and most of what produced
        it is a coefficient somebody wrote down rather than a measurement of
        the card being modelled; `kv_pool_sizing` keeps the count beside the
        readings, so which terms those were is recoverable from the runner and
        not only from a log line somebody kept.

        The decode process of intra-GPU prefill/decode disaggregation is
        refused rather than sized. ATOM's own runner for that answers a block
        count of zero there, because decode imports the pool from prefill and
        owns no device memory (`model_runner.py:4272-4286`), and it holds back
        four safety margins on the prefill side because two processes share the
        card (`:4266-4270`). Substituting readings into the base method reaches
        neither: this path would hand a decode process a pool it does not own.
        The base's `_kv_budget_extra_reserve` of zero is left alone for the
        same reason it is right -- the card these readings describe is a
        dedicated one, and a shared-card reservation would hold bytes back for a
        second tenant that those readings do not have.
        """
        if _config_field(self, "disagg_is_decode"):
            raise RunnerRefusal(
                "this runner does not model the decode process of intra-GPU "
                "prefill/decode disaggregation: that process owns no device "
                "memory and imports the pool from prefill, so its block count "
                "is zero rather than a sizing, and the substituted readings "
                "describe a card with one tenant on it."
            )
        readings = _installed_readings(self)
        reply = super().get_num_blocks()
        self.kv_pool_sizing = SizedKVPool(
            num_kvcache_blocks=int(reply["num_kvcache_blocks"]),
            entries=dict(reply["pool_entries"]),
            readings=readings,
        )
        logger.info("%s", self.kv_pool_sizing.table())
        return reply

    def allocate_kv_cache(self, num_kvcache_blocks: int) -> bool:
        """Record the block count and allocate nothing.

        The block accounting above this -- the pool, the prefix index, eviction,
        preemption -- is integer arithmetic and runs unmodified against the count
        recorded here. Only the tensors behind the blocks are absent.

        An empty registry still goes through ATOM's `set_kv_cache_data`: it is
        the one call that builds the worker-side KV connector, without which no
        transfer the scheduler announces ever starts or finishes.
        """
        from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
        from atom.utils.forward_context import set_kv_cache_data

        kv = _config_field(self, "kv_transfer_config")
        if kv:
            name = KVConnectorFactory.canonical_name(kv.get("kv_connector", "moriio"))
            if name != "compass":
                raise RunnerRefusal(
                    f"kv_connector {name!r} is a real transfer backend; this "
                    "runner simulates only 'compass', and an unset kv_connector "
                    "means 'moriio'."
                )
        set_kv_cache_data({}, self.config, num_blocks=num_kvcache_blocks)
        self.config.num_kvcache_blocks = num_kvcache_blocks
        logger.info(
            "kv cache: %d blocks accounted, 0 bytes allocated.",
            num_kvcache_blocks,
        )
        return True

    def capture_cudagraph(self) -> tuple[float, list[int], int]:
        """Capture nothing, and say so in the three values the caller unpacks.

        The base reaches the model only after it has zeroed device buffers,
        opened a graph memory pool and entered a capture context, so inheriting
        it would put bytes on a device before it discovered there are no
        weights to trace. Nothing is captured here, and the reply says exactly
        that: no seconds spent, no batch sizes captured, no pool bytes held.

        The shape is fixed by `engine_core.py:149-151` and `1109-1111`, which
        both write `cap_cost, bs, pool_bytes = ...` and then format the first
        and third as numbers. Returning two values, or None, is not a test
        failure over there: the worker dies on the unpack, or never replies, and
        the caller waits for the rest of its life either way.

        `capture_sizes` and `capture_sizes_np` are deliberately left at the
        eager-fallback values the base set during construction, because the
        attention metadata builder reads them on every step. The reply
        deliberately disagrees with the attribute it preserves: the base sends
        `self.capture_sizes` -- the same `[0]` -- as the second element
        (`model_runner.py:4112`), and this sends `[]`. Both are only ever
        formatted into a log line, and `[]` is the truthful one.

        `engine_core.py:148` reaches this method only when `not enforce_eager
        and not disagg_is_decode`, so on an eager deployment the override never
        runs at all. Replacing it anyway is the point: a runner must not depend
        on a caller-side flag to stay off the device.
        """
        logger.info("no cudagraph captured: there are no weights to trace.")
        return 0.0, [], 0

    @torch.inference_mode()
    def forward(self, batch: Any) -> Any:
        """Report what a step produced, without running one.

        The reply is the whole of what the scheduler learns from a step, and
        `step_output` holds the three rules that decide it: the tokens belong
        to the previous output-producing batch, the lag is counted in
        output-producing steps rather than in steps, and a batch that samples
        nothing reports its requests with no tokens. Each is wrong in a way the
        scheduler accepts without complaining, so they are stated in one place
        with their reasons rather than inlined here.

        What a replacement owes its callers. Four sites broadcast this name --
        `engine_core.py:386` and `:1264`, and `pp_engine_core.py:118` (which
        waits and discards the reply) and `:379`. None of them unpacks it; the
        reply is one object, read for its attributes, and the reads are these:

        * `.req_ids`, an iterable of request ids in the batch's own order --
          `pp_engine_core.py:144`. That read is on the head's side of the
          transport, on what `recv_tokens()` returns at `:139`, not at the
          `:379` call site, which only passes the object on.
        * `.token_ids`, `.draft_token_ids`, `.is_deferred_out`, `.logprobs`
          (None or a `dict[int, float]`) -- `scheduler.py:2435-2438`.
        * `.get_idx(req_id)`, returning a row index or None --
          `scheduler.py:2454`.
        * `.num_rejected[idx]` and `.num_bonus[idx]`, indexable by that row and
          castable to `int` -- `scheduler.py:2521-2522`.
        * `.dspark_ell`, either None or a mapping answering `.get(seq.id)` --
          `scheduler.py:2541-2542`.

        Nine attributes, and picklable on top: under pipeline parallelism the
        last stage sends the object whole (`pp_engine_core.py:385`) and the head
        reads it back (`:139`), each hop through `pickle` in
        `atom/distributed/pp_transport.py:141` and `:114`. ATOM's own answer is
        a `ScheduledBatchOutput`; the contract a replacement has to meet is the
        nine reads and the pickle, not that class.

        One decorator, not the base's two. ATOM's own carries
        `torch.inference_mode`, which is kept: the body no longer raises, and
        whatever a cost model eventually evaluates here should build in the
        same context as the step it stands for. It also carries
        `with_eplb_forward_monitor`, which is not. That monitor exists to
        observe how a real forward routed tokens across experts and commits a
        load window from what it saw; a predicted step routes nothing, so the
        window would be fabricated, and acting on one can move experts on a
        device this runner owns none of. A decorator is applied while the
        class body runs, so it would also have to be imported from the engine
        at module scope, which this module does not do.
        `RapidServeModelRunner`, the other runner in this tree that declines
        to own its memory, carries the same one of the two.

        No duration is reported. The reply has nowhere to put one: the engine
        times the call itself, and the batch output it reads carries tokens.
        """
        if not hasattr(batch, "produces_output"):
            # Not reachable from the engine, which only ever passes a scheduled
            # batch. Kept because it is the one shape this method cannot report
            # from, and because a refusal that did cross the worker boundary
            # would reach the parent as a bare SystemExit -- so the loud
            # failure has to happen on this side of it.
            raise RunnerRefusal(
                "a step is reported from what the scheduler scheduled, and "
                f"{type(batch).__name__} cannot say whether its batch produces "
                "output; there is nothing here to report from."
            )
        # Imported at call time, not at module scope, so this module stays
        # importable where there is no driver. By the time a step is reported
        # the worker has imported the engine anyway.
        from atom.model_engine.scheduler import ScheduledBatchOutput

        stream = getattr(self, "_token_stream", None)
        if stream is None:
            # Built on first use rather than in an `__init__`: the base class
            # runs the whole of its own before a subclass body would get
            # control, and `self.config` is what this reads.
            stream = DeferredTokenStream(
                reported_token_id(
                    _config_field(self, "eos_token_id"),
                    _config_field(self, "stop_token_ids"),
                ),
                deferred=reports_previous_step(
                    _config_field(self, "pipeline_parallel_size")
                ),
            )
            self._token_stream = stream
        return ScheduledBatchOutput(**stream.step(batch))
