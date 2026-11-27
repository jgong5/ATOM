# SPDX-License-Identifier: MIT
"""Each step's scheduling decision, recorded on a real or a simulated runner.

A simulated run reuses ATOM's scheduler, so it should build the batches a real
run builds. This records them where both runs can be read the same way, at the
model runner's `forward`: one JSON line per call, built before the call runs,
because a real forward rewrites some batch fields in place. Each line carries
the batch as the `BatchView` a simulated runner prices it from (`rows`, `rung`,
and above DP width 1 `moe_rows` and `moe_tokens`, with an MTP drafter `draft_moe_rows` too, `draft_bs` where its later passes pad, `draft_rows` where they run at its rows, `target_bs` where its target pads, on a split step `tbo_tokens`,
`group_needs_eager`, true, on a step an attention backend refuses a graph for,
and `kda` for a model with KDA layers); a
real runner settles that view from the `ForwardMode` its own forward settled,
so recording adds no DP collective. Its line also carries `t_enter_ns` and `t_exit_ns`, the worker's
monotonic clock around its forward, and is finished once the forward returns.
Inside the forward it carries `[enter, exit]` on that clock around the staging
gate (`t_gate_ns`), the launch (`t_launch_ns`) and the wait for the previous
output's token copy (`t_recv_ns`, only on a forward that reaches it; off a
pipeline, a runner that does not defer its output reads its own tokens under
`t_read_ns` instead), and with
a drafter its MTP status (`t_mtp_ns`) and draft copy (`t_draft_recv_ns`; reading its own, `t_draft_read_ns`). Its device
window is `d_busy_ns`, from an event recorded at the launch's entry, the clock
read under `t_launch_event_ns` once the event exists and just before its record,
to one recorded at the forward's exit (`d_target_ns` to one recorded just after
`postprocess` records `forward_done_event`, less the drafter's `compute_draft_kv`
pass before it, or on a forward that samples nothing, to the start of its first
drafter pass, that one or `propose_draft_token_ids`); `d_draft_kv_ns` and
`d_propose_ns` time those two drafter passes, each between device events at its
entry and exit, on a forward that runs it; under an EAGLE3 drafter
`d_draft_passes_ns` times each draft forward `propose_draft_token_ids` runs, one
a pass, from a device event where the drafter points its context at that pass
(`_publish_draft_shape`) to the next one's, the last to the propose's exit;
and `d_gap_ns`, from the previous forward's
exit event to this launch's entry event, absent on the first line; summed in
order they lay every forward on one device timeline. A later pipeline stage's
receive of its input is `t_pp_recv_ns`, its device part `d_recv_ns`; a stage
before the last starts sending its output at `d_send_ns`, an event before it
settles its previous send, whose wait holds the stream until the next stage
takes that send; an idle
`flush_pp_send` since the last forward that settled a pending send is `t_flush_ns`. The events are polled,
never waited on, so a line is finished once its exit event has completed. A
real runner's lines are kept in memory and written at its `exit`, so no file
call or JSON encoding runs between its forwards; a worker that never reaches
`exit` (SIGKILL, a native crash) leaves an empty record. With `WRITE_THROUGH`
set to 1, as `compass record --sweep` sets it for a record nothing grades, a
real runner writes and flushes its finished lines at each forward instead,
never waiting on a window still open, so a worker that dies keeps every line
finished by its last forward. A real line also
carries `launch`, where its model's launch provider declares features
(`backends.launch_features`), and `blocked_ns`, where the worker loaded the
blocking-call tool (`blocked`): the nanoseconds its thread spent in each HIP
call that blocked, from the launch's entry to its token read's, or to
the launch's exit on a forward with no such wait. A simulated runner's line is
written as its step is recorded, and flushed to the file the record keeps open
from its start, so a killed simulated worker keeps every line. Under `kv` a line carries the lmcache offload loads
and saves the scheduler handed the worker since the previous line
(`StepRecord.note_kv`), where any were. `compare` reads two records and names, per DP rank,
the first step where they part; timestamps are not part of a step's decision.

Setting `ATOM_COMPASS_PARITY_RECORD` to a directory turns recording on: TP rank 0
of each DP rank writes `dp<rank>.jsonl` there, replacing any earlier file
(pipeline stage `k` above 0: `pp<k>.dp<rank>.jsonl`, its other files alike;
a RapidServe engine in its `record_dir`).
Unset, nothing is recorded. On its first step it also writes the engine its
runner resolved to `dp<rank>.engine.json` (`README.md` says what it holds), and
a runner with a `host_key` writes that to `dp<rank>.key.json`, with the
graph sets ATOM recorded (`recorded_graphs`, and its launch provider's
`recorded`) where `launch` is written. The simulated
runner records from `NonAllocatingRunner.forward`; a real one records through
`StepRecording`, composed into `RUNNER`, which the frontend names on a real run
with the variable set.

A request is named by its key, because request ids are numbered per run: a
digest of its whole prompt as its prefill windows carry it, taken when its final
chunk is scheduled, so it does not move with chunking. A line records request
ids and names each request whose final chunk it holds; `read` puts the keys in.
A named request's response id follows its key if it has one (`carry_response_ids`).
A request in a batch marking no final chunk (a decode instance's, its KV
received, or a RapidServe engine's) has no digest: its first step names it,
its request id its key, so its response id still follows it. A 1P1D
deployment's two records join on prefill's request id (`transfers`).
Prefix caching is not covered: the worker never sees a cached prefix, so parity
runs need caching off or the same cache hits in both runs. Nor is an lmcache
offload load: a prefill window that starts where the last load handed for its
request ended, and the request's last window ended where that load began, is
digested on from there, the loaded tokens left out as a cached prefix's are. Requests of one
record with one prompt are named in the order their final chunks are scheduled:
the first by the digest, the n-th after it by `<digest>.<n>`. A run that
schedules them in the other order names them the other way round, and `compare`
may then report a step where the twins' rows differ.
"""

import collections
import contextlib
import copy
import functools
import hashlib
import importlib
import json
import os
import pathlib
import sys
import time
from itertools import pairwise, zip_longest

import numpy as np

from atom.compass.backends.json_file import read_json, read_jsonl
from atom.compass.backends.launch_features import GRAPHS, architecture, provider
from atom.compass.backends.step_labels import DRAFT_PASSES, join_stamps

ENV = "ATOM_COMPASS_PARITY_RECORD"
#: Set to 1, a real runner's record is written as its lines finish, not at `exit`.
WRITE_THROUGH = "ATOM_COMPASS_PARITY_WRITE_THROUGH"
#: The file in a real run's record directory holding when its workload was sent
#: and when the server admitted each request.
ARRIVALS = "arrivals.json"
#: The file in a decode instance's record directory joining each request's
#: response id to the prefill request its KV came from (`carry_response_ids`).
TRANSFERS = "transfers.jsonl"
#: A transfer's fields naming the prefill instance it came from: a simulated
#: one's engine id (`kv.connector.engine_id`), a real one's host and Mooncake
#: handshake port.
INSTANCE_FIELDS = ("remote_engine_id", "remote_host", "remote_handshake_port")
#: The file in a record directory with a line ``{key, rank, reqs, tokens}`` per
#: request the frontend routed across DP ranks, in routing order: its whole
#: prompt's digest, its rank, and each rank's in-flight requests and queued
#: prefill tokens the router weighed (`atom.compass.run._record_routes`).
ROUTES = "routes.jsonl"
RUNNER = "atom.compass.parity.runner.RecordingModelRunner"
#: The recording runner of a real RapidServe run.
RAPID = "atom.compass.parity.runner.RecordingRapidServeModelRunner"
#: Line keys that time a step rather than describe its decision.
TIMESTAMPS = (
    "t_enter_ns",
    "t_exit_ns",
    "t_gate_ns",
    "t_launch_ns",
    "t_launch_event_ns",
    "t_recv_ns",
    "t_read_ns",
    "t_mtp_ns",
    "t_draft_recv_ns",
    "t_draft_read_ns",
    "t_pp_recv_ns",
    "t_flush_ns",
    "d_busy_ns",
    "d_gap_ns",
    "d_target_ns",
    "d_recv_ns",
    "d_send_ns",
    "d_draft_kv_ns",
    "d_propose_ns",
    DRAFT_PASSES,
    "t_pull_ns",
    "t_put_ns",
    "blocked_ns",
)
#: Profiler ranges a trace run (`busy`) splits a forward's kernels at, on every
#: TP rank: the target ends where `TARGET_DONE` opens, each drafter pass is the
#: range named `MARK` and its line field, and a pipeline stage's receive of its
#: input and send of its output are `PP_RECV` and `PP_SEND`, its handoff.
MARK = "compass::"
TARGET_DONE = MARK + "target_done"
PP_RECV = MARK + "pp_recv"
PP_SEND = MARK + "pp_send"
#: ATOM's runner module, and the calls by which its `run_model` sends a stage's
#: output: the previous send settled, then this one started.
SENDER = "atom.model_engine.model_runner"
SENDS = ("commit_pp_send_work", "async_send_intermediate_tensors")
#: The stamps the blocking-call tool's sums are read at: a line's `blocked_ns`
#: runs from the launch's entry to its token read's, as `W_launch` does.
BLOCKED_AT = ("t_launch_ns", "t_recv_ns", "t_read_ns")


class StepRecord:
    """One DP rank's record: a JSON line per forward call. A real runner's
    line is written at `write`, or with `write_through` at the first forward
    that finds it finished, as a simulated runner's always is."""

    def __init__(
        self, directory, dp_rank: int, stage: int = 0, write_through: bool = False
    ) -> None:
        name = f"pp{stage}.dp{dp_rank}" if stage else f"dp{dp_rank}"
        self.path = pathlib.Path(directory) / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Opened once, replacing any earlier file, and kept open: on a network
        # file system an open and a close per simulated step took most of its
        # wall time.
        self.file = self.path.open("w")
        # Finished lines, not yet written. Built of tuples, not lists, so a
        # full collection untracks them: a million list-built lines held made
        # each full collection 0.8 s. ponytail: unless `write_through`, the
        # whole record is held in the worker's memory until `exit`; bound it if
        # a run outgrows that.
        self.lines: list[dict] = []
        self.write_through = write_through
        self.dp_rank = dp_rank
        self.step = 0
        self.keys: dict[int, str] = {}  # request id -> key
        self.prompt_count: collections.Counter[str] = collections.Counter()
        self.prompts: dict[int, tuple] = {}  # request id -> (digest, next position)
        # (line, its device window: a callable taking `wait`, or None once closed)
        self.pending: collections.deque[tuple] = collections.deque()
        # Offload decisions handed to the worker, not yet on a line.
        self.kv: list[tuple] = []
        # Request id -> (start, end) of the last offload load handed for it.
        self.loaded: dict[int, tuple[int, int]] = {}

    def note_kv(self, meta) -> None:
        """Keep each lmcache offload load and save `meta` hands the worker:
        ``[request id, "load", HBM-cached tokens, offload-cached tokens]`` or
        ``[request id, "save", tokens already saved, tokens computed]``."""
        from atom.kv_transfer.offload.mp.token_shipping import ShippedTokens

        for req in getattr(meta, "requests", ()):
            if (ls := req.load_spec) is not None:
                span = (ls.hbm_cached_tokens, ls.lmcache_cached_tokens)
                self.kv.append((req.req_id, "load", *span))
                self.loaded[req.req_id] = span
            if (ss := req.save_spec) is not None:
                # A real worker is handed the scheduler's shipment, which names
                # the prefix's end, before it rebuilds the prefix.
                tokens = req.token_ids
                end = tokens.end if isinstance(tokens, ShippedTokens) else len(tokens)
                self.kv.append((req.req_id, "save", ss.skip_leading_tokens, end))

    def add(self, batch, runner, forward=None):
        """Append `batch`'s line; with `forward`, run it inside the line's timestamps.

        `batch` is recorded as (request id, scheduled tokens, context length)
        per request, and as the `BatchView` `runner` prices it from: `rows`
        (query tokens, context tokens, decode), `rung`, the graph width the
        step replays or None, and `moe_rows`, the rows the DP group's MoE
        gather carries, absent at DP width 1, and `draft_moe_rows`, `draft_bs`, `draft_rows`, `target_bs`, `moe_tokens` and `tbo_tokens`, each where the
        view has it, `group_needs_eager` where it is true, and `kda`, a KDA
        model's state counts (`projection.kda_counts`). The view's mode is the one `forward` settled, stamped by
        `run_model` under `forward_mode`; without one it is `forward_mode`'s,
        which runs the DP collective itself. Each prefill window is added to its request's
        digest; the line holding a request's final chunk names it under
        `named`, as does the first line holding a request of a batch marking
        no final chunk, under its request id. A dummy batch is fabricated for DP
        synchronisation and names no request, so its `batch` is empty; its
        `rows` are what it runs.

        The line is built from the batch as handed in, since a real forward
        rewrites batch fields in place, and written after it returns, with `t_enter_ns`
        and `t_exit_ns` read off `runner.step_clock_ns()` around it and
        `runner.step_stamps()` added; a line with a device window waits in
        `pending` until the window closes, then in `lines` until `write`,
        which a write-through record runs at each step. Returns `forward`'s reply, or
        without one the mode the view was built from and the view, so a caller
        pricing the step need not run the collective or build the view again.
        """
        from atom.compass.runner.projection import batch_view, forward_mode, is_dummy

        # The view is settled after `forward`, from the fields it was handed;
        # with no forward to rewrite them, the batch is its own.
        before = batch
        if forward is not None:
            before = copy.copy(batch)
            before.num_scheduled_tokens = np.copy(batch.num_scheduled_tokens)
            before.context_lens = np.copy(batch.context_lens)
        rows, named = [], []
        if not is_dummy(batch):
            ends = np.cumsum(batch.num_scheduled_tokens)
            for i, req_id in enumerate(batch.req_ids):
                num = int(batch.num_scheduled_tokens[i])
                rows.append((req_id, num, int(batch.context_lens[i])))
                if req_id in self.keys:
                    continue
                if batch.is_final_chunk is None:
                    # No final chunk marked (a decode instance's request, or any
                    # request of a RapidServe engine): no prompt digest, so the
                    # key is its request id, the same in another run of the
                    # workload, unlike its response id.
                    response_id = batch.external_request_ids[i]
                    if response_id is not None:
                        self.keys[req_id] = req_id
                        named.append((req_id, req_id, response_id))
                    continue
                start = int(batch.num_cached_tokens[i])
                if start == 0:  # a preempted prefill restarts its prompt
                    self.prompts.pop(req_id, None)
                digest, end = self.prompts.pop(
                    req_id, (hashlib.blake2b(digest_size=8), start)
                )
                # A load filled [end, start) from the offload tier.
                span = self.loaded.get(req_id)
                if span is not None and span[0] <= end < start <= span[1]:
                    end = start
                    del self.loaded[req_id]
                # A forward the scheduler backed off below where the prefill
                # resumed recomputes the rows between: digested once already.
                replay = batch.replaying.get(i)
                skip = end - start if replay is not None and replay.until == end else 0
                if start + skip != end:
                    raise ValueError(
                        f"request {req_id}'s prefill window starts at token "
                        f"{start}, not at {end} where its last one ended, so its "
                        "prompt cannot be digested whole."
                    )
                window = batch.scheduled_tokens[ends[i] - num + skip : ends[i]]
                digest.update(np.asarray(window, dtype=np.int32).tobytes())
                if not batch.is_final_chunk[i]:
                    self.prompts[req_id] = (digest, start + num)
                    continue
                key = digest.hexdigest()
                n = self.prompt_count[key]
                self.prompt_count[key] += 1
                if n:
                    key = f"{key}.{n}"
                self.keys[req_id] = key
                response_id = batch.external_request_ids[i]
                named.append(
                    (req_id, key) + (() if response_id is None else (response_id,))
                )
        line = {
            "step": self.step,
            "dp_rank": self.dp_rank,
            "is_dummy_run": is_dummy(batch),
            "produces_output": bool(batch.produces_output()),
            "batch": tuple(rows),
            "named": tuple(named),
        }
        if self.kv:
            line["kv"], self.kv = tuple(self.kv), []
        reply, window, mode = None, None, None
        if forward is not None:
            line["t_enter_ns"] = runner.step_clock_ns()
            reply = forward(batch)
            line["t_exit_ns"] = runner.step_clock_ns()
            stamps, window = runner.step_stamps()
            mode = stamps.pop("forward_mode", None)
            line.update(stamps)
        mode = mode or forward_mode(before, runner)
        view = batch_view(before, mode, runner)
        line["rows"] = tuple(
            (r.query_tokens, r.context_tokens, r.decode) for r in view.requests
        )
        line["rung"] = view.capture_rung
        if view.moe_rows is not None:
            line["moe_rows"] = view.moe_rows
        if view.draft_moe_rows is not None:
            line["draft_moe_rows"] = view.draft_moe_rows
        for field in (
            "draft_bs",
            "draft_rows",
            "target_bs",
            "moe_tokens",
            "tbo_tokens",
        ):
            if getattr(view, field) is not None:
                line[field] = getattr(view, field)
        if view.group_needs_eager:
            line["group_needs_eager"] = True
        if view.kda is not None:
            line["kda"] = view.kda
        self.pending.append((line, window))
        # A simulated step is priced, not timed, and nothing grades a
        # write-through record's time: their finished lines are written now.
        if forward is None or self.write_through:
            self.write(wait=False)
        else:
            self.flush()
        self.step += 1
        return (mode, view) if forward is None else reply

    def flush(self, wait: bool = False) -> None:
        """Finish the lines whose device windows have closed, in step order.

        A line stops the finishing while its window is open, unless `wait`.
        """
        while self.pending:
            line, window = self.pending[0]
            if window is not None:
                fields = window(wait)
                if fields is None:
                    return
                line.update(fields)
            self.lines.append(line)
            self.pending.popleft()

    def write(self, wait: bool = True) -> None:
        """Append the finished lines to the file, flushed, so a worker killed
        after this keeps them; with `wait`, first wait for every open window."""
        self.flush(wait)
        self.file.write("".join(json.dumps(line) + "\n" for line in self.lines))
        self.file.flush()
        self.lines.clear()


def record_step(runner, batch, forward=None):
    """Record `batch` if `ENV` names a directory and `runner` is TP rank 0.

    With `forward`, it runs on `batch` whether or not the step is recorded,
    inside the line's timestamps when it is, and its reply is returned.
    Without `forward`, a recorded step returns the `ForwardMode` it settled and
    the `BatchView` it built from it, and an unrecorded one None.
    """
    if _record(runner) is not None:
        return runner._parity_record.add(batch, runner, forward)
    return None if forward is None else forward(batch)


def record_kv(runner, meta) -> None:
    """Keep the offload decisions in `meta`, dispatched to `runner`'s worker,
    for its next recorded step's line, as `record_step` would record it."""
    if _record(runner) is not None:
        runner._parity_record.note_kv(meta)


def record_dir(directory, config) -> pathlib.Path:
    """Where the engine of `config` records under `directory`: RapidServe's
    prefill and decode engines in ``prefill/`` and ``decode/``, any other there."""
    if not getattr(config, "enable_rapidserve", False):
        return pathlib.Path(directory)
    return pathlib.Path(directory) / (
        "decode" if config.disagg_is_decode else "prefill"
    )


def _record(runner):
    """`runner`'s record, made on its first call; None unless recorded."""
    if not hasattr(runner, "_parity_record"):
        directory = os.environ.get(ENV)
        record = None
        if directory and runner.rank == 0:
            directory = record_dir(directory, runner.config)
            from atom.compass.backends.step_grid_plan import (
                DEFERS_OUTPUT,
                MIN_FINAL_ROWS,
                PREFILL_BUCKETS,
                REPLAYS_STATE,
                UNKEYED,
            )
            from atom.compass.backends.step_linear_fit import CONTEXT_PARALLEL
            from atom.compass.run.step_grid import keyed
            from atom.utils.region_graph import TokenGraph

            pc = runner.config.parallel_config
            pp = runner.config.pipeline_parallel_size > 1
            stage = pc.pipeline_parallel_rank if pp else 0
            through = os.environ.get(WRITE_THROUGH) == "1"
            record = StepRecord(directory, pc.data_parallel_rank, stage, through)
            # By its first step the runner has resolved its engine.
            engine = keyed(runner.config) | {
                name: getattr(runner.config, name) for name in UNKEYED
            }
            for name in CONTEXT_PARALLEL:
                if (width := getattr(runner.config, name, 1)) > 1:
                    engine[name] = width
            if (rows := getattr(runner.config, MIN_FINAL_ROWS, 0)) > 0:
                engine[MIN_FINAL_ROWS] = rows
            # The transfer the runner hands `BlockManager`, which keeps a
            # replaying one: it carries no checkpoint spec (`StateRuntime`).
            state = getattr(runner, "state_runtime", None)
            if state is not None and state.transfer.replays:
                engine[REPLAYS_STATE] = True
            engine[PREFILL_BUCKETS] = list(TokenGraph.captured_sizes("prefill"))
            processor = getattr(runner, "tokenID_processor", None)
            engine[DEFERS_OUTPUT] = getattr(processor, "is_deferred_out", True)
            record.path.with_suffix(".engine.json").write_text(json.dumps(engine))
            key = getattr(runner, "host_key", None)
            if key is not None:
                if getattr(runner, "_launch", None) and runner._launch.features:
                    sets = recorded_graphs(runner.config)
                    graphs = {GRAPHS: sets | runner._launch.recorded(runner)}
                    key = {**key, "engine": {**key["engine"], **graphs}}
                record.path.with_suffix(".key.json").write_text(json.dumps(key))
        runner._parity_record = record
    return runner._parity_record


def recorded_graphs(config) -> dict:
    """`GRAPHS`' mapping for a runner of `config` whose ATOM has recorded its
    graphs: its model's architecture, each phase's token-graph buckets
    (`TokenGraph.captured_sizes`) and each request graph's keys, sorted: the
    graphs by their keys' text, since one keys by a tuple (a decode's request
    bucket) and another by an int (a DSpark draft's `running_bs`)."""
    from atom.utils.region_graph import ReqGraph, TokenGraph

    token = {p: TokenGraph.captured_sizes(p) for p in ("prefill", "decode")}
    keys = (sorted(g.keys()) for g in ReqGraph._live if g.keys())
    request = sorted(keys, key=repr)
    return {"architecture": architecture(config), "token": token, "request": request}


#: How many frames above `add_request` `_caller_request_id` looks for the API
#: server function's `request_id`: since the upstream merge its handlers reach
#: `add_request` through `_submit_engine_requests`, one frame further up.
CALLER_DEPTH = 4


def _caller_request_id():
    """The `request_id` local of the nearest of `CALLER_DEPTH` frames above
    the patched `add_request` that has one, or None (a thread's own stack,
    such as `asyncio.to_thread`'s, holds none)."""
    frame = sys._getframe(2)
    for _ in range(CALLER_DEPTH):
        if frame is None:
            return None
        if "request_id" in frame.f_locals:
            return frame.f_locals["request_id"]
        frame = frame.f_back
    return None


def carry_response_ids() -> None:
    """Set a one-sequence request's `external_request_id` to its response id,
    the `request_id` of the API server function calling `add_request`, read
    off its frame, or off its caller's when `_submit_engine_requests` makes
    the call. A sequence with an id keeps it; a fan-out's get none.

    A sequence whose `kv_transfer_params` carry a `transfer_id`, a decode
    instance's request, also gets a line ``{response_id, remote_dp_rank,
    transfer_id}``, with each of `INSTANCE_FIELDS` the parameters hold, in
    `TRANSFERS` of the record directory, which this replaces."""
    from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager

    path = pathlib.Path(os.environ[ENV]) / TRANSFERS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    for cls in (CoreManager, DisaggCoreManager):

        def add_request(self, seqs, add=cls.add_request):
            if len(seqs) == 1 and seqs[0].external_request_id is None:
                seqs[0].external_request_id = _caller_request_id()
            params = (seqs[0].kv_transfer_params or {}) if len(seqs) == 1 else {}
            if params.get("transfer_id") is not None:
                line = {
                    "response_id": seqs[0].external_request_id,
                    "remote_dp_rank": params.get("remote_dp_rank", 0),
                    "transfer_id": params["transfer_id"],
                }
                # Which prefill instance sent it, where a deployment runs
                # several: their transfer ids, each its own sequence ids, meet.
                for field in INSTANCE_FIELDS:
                    if params.get(field) is not None:
                        line[field] = params[field]
                with path.open("a") as f:
                    f.write(json.dumps(line) + "\n")
            return add(self, seqs)

        cls.add_request = add_request


def _memory_format(name: str):
    import torch

    return getattr(torch, name)


def key_memory_formats_by_name() -> None:
    """Pickle a `torch.memory_format` as its name in `torch`.

    Pickle otherwise saves one under the first module in `sys.modules` that
    reaches it, so inductor's FX graph cache keys a weight by the process's
    import order: an API server's engine and `compass measure memory`'s keyed
    the same graphs apart, and a server started after the measurement compiled
    them again, reading `non_torch` above it. Each real runner calls this
    before its model compiles."""
    import copyreg

    import torch

    copyreg.pickle(
        torch.memory_format,
        lambda f: (_memory_format, (str(f).removeprefix("torch."),)),
    )


class StepRecording:
    """Records each forward call; put it before `ModelRunner` in the bases.

    ATOM's `ModelRunner.__init__` warms up with a forward over a fabricated
    batch, which the simulated runner declines, so recording starts once
    construction returns. The class it is composed into supplies
    `step_clock_ns`, the clock its lines are timed by, and
    `device_event(event=None)`, which records `event`, or a new timing event,
    on the current stream and returns it.

    A recorded forward is also stamped inside: around `_gate_staging_reuse`,
    `run_model` and the instance's `tokenID_processor.recv_async_output`, with
    device events at `run_model` entry and at the forward's exit. At
    `run_model` entry it also keeps the `ForwardMode` the forward settled,
    which the line's view is built from.

    On a pipeline stage it wraps the PP group's receive of its input and, on a
    stage before the last, the `SENDS` calls of `SENDER` that send its output:
    each runs under its profiler range (`PP_RECV`, `PP_SEND`) on every TP
    rank, and a recorded forward records a device event where the received
    tensors are in and one where its send starts.

    Once `allocate_kv_cache` has built a Mooncake connector, every worker
    records that connector's KV writes (`writes`), written at `exit`. A
    launch provider that declares features is installed once built, and its
    `observe` is the line's `launch`. The class it is composed into may set
    `_blocked` to the blocking-call tool (`blocked.loaded`) for `blocked_ns`.
    """

    _recording = False
    _blocked = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._launch = provider(architecture(getattr(self, "config", None)))
        if self._launch.features:
            self._launch.install(self)
        self._stamps: dict = {}
        self._last_exit = None
        processor = getattr(self, "tokenID_processor", None)
        if processor is not None:
            recv = processor.recv_async_output
            processor.recv_async_output = functools.partial(
                self._stamped, "t_recv_ns", recv
            )
        pp = getattr(getattr(self, "config", None), "pipeline_parallel_size", 1) > 1
        if not getattr(processor, "is_deferred_out", True):
            # The forward reads its own tokens off the device: the wait is the
            # call. Off a pipeline the gate releases that wait at the forward's
            # own target end, and the labels end `W_launch` and start `W_post` there.
            sampled = processor.prepare_sampled_ids
            processor.prepare_sampled_ids = functools.partial(
                self._stamped, "t_recv_ns" if pp else "t_read_ns", sampled
            )
            if hasattr(self, "drafter"):  # and its own draft ids, once its draft ends
                drafts = processor.prepare_draft_ids
                processor.prepare_draft_ids = functools.partial(
                    self._stamped, "t_draft_read_ns", drafts
                )
        if pp:
            from aiter.dist.parallel_state import get_pp_group

            group = get_pp_group()
            if not group.is_first_rank:
                recv = group.recv_tensor_dict
                group.recv_tensor_dict = functools.partial(self._pp_received, recv)
            last = self.config.pipeline_parallel_size - 1
            if self.config.parallel_config.pipeline_parallel_rank < last:
                # `run_model` calls them by name, and no method of the group
                # or the runner holds them alone.
                module = importlib.import_module(SENDER)
                for name in SENDS:
                    call = functools.partial(self._pp_sending, getattr(module, name))
                    setattr(module, name, call)
        if processor is not None and hasattr(self, "drafter"):
            for key, name in (
                ("t_mtp_ns", "recv_spec_status"),
                ("t_draft_recv_ns", "recv_draft_token_ids"),
            ):
                call = getattr(processor, name)
                setattr(processor, name, functools.partial(self._stamped, key, call))
            sampled = processor.prepare_sampled_ids
            processor.prepare_sampled_ids = functools.partial(
                self._target_done, sampled
            )
            draft_kv = self.drafter.compute_draft_kv
            self.drafter.compute_draft_kv = functools.partial(
                self._draft_pass, "d_draft_kv_ns", draft_kv
            )
            spec = getattr(self.drafter, "speculative_config", None)
            publish = getattr(self.drafter, "_publish_draft_shape", None)
            if publish is not None and getattr(spec, "method", None) == "eagle3":
                self.drafter._publish_draft_shape = functools.partial(
                    self._serial_pass, publish
                )
        self._recording = True

    def forward(self, batch):
        if self._recording:
            return record_step(self, batch, super().forward)
        return super().forward(batch)

    def process_kvconnector_output(self, meta):
        if self._recording:
            record_kv(self, meta)
        return super().process_kvconnector_output(meta)

    def allocate_kv_cache(self, num_kvcache_blocks):
        reply = super().allocate_kv_cache(num_kvcache_blocks)
        from atom.compass.parity import writes

        # The KV connector is built here, before any request can reach it.
        self._writes = writes.install(self)
        return reply

    def flush_pp_send(self):
        """ATOM's idle flush, stamped for the next forward when a send is pending."""
        if getattr(self, "_pp_pending_send", None):
            return self._stamped("t_flush_ns", super().flush_pp_send)
        return super().flush_pp_send()

    def _stamped(self, key, call, *args, **kwargs):
        """`call(*args, **kwargs)`, stamped under `key` when this forward is
        recorded, with the blocking-call tool's sums around it at `BLOCKED_AT`."""
        if getattr(self, "_parity_record", None) is None:
            return call(*args, **kwargs)
        tool = self._blocked if key in BLOCKED_AT else None
        enter = self.step_clock_ns()
        before = None if tool is None else tool()
        reply = call(*args, **kwargs)
        self._stamps[key] = (enter, self.step_clock_ns())
        if tool is not None:
            self._stamps[(key, "blocked")] = (before, tool())
        return reply

    def _pp_received(self, call, *args, **kwargs):
        """The stage before's output, received under `t_pp_recv_ns` inside
        `PP_RECV`, then a device event where the received tensors are in."""
        with _traced(PP_RECV):
            reply = self._stamped("t_pp_recv_ns", lambda: call(*args, **kwargs))
        if getattr(self, "_parity_record", None) is not None:
            self._stamps["recv_event"] = self.device_event()
        return reply

    def _pp_sending(self, call, *args, **kwargs):
        """`call`, a part of this stage's send of its output, inside `PP_SEND`;
        in a recorded forward's launch the first part is preceded by a device
        event, the send's start. An idle `flush_pp_send` settles a send outside
        any launch and records none."""
        if (
            getattr(self, "_parity_record", None) is not None
            and "launch_event" in self._stamps
            and "send_event" not in self._stamps
        ):
            self._stamps["send_event"] = self.device_event()
        with _traced(PP_SEND):
            return call(*args, **kwargs)

    def _target_done(self, call, *args, **kwargs):
        """`call(*args, **kwargs)` after a device event where `postprocess` has
        just recorded `forward_done_event`: the target's end, sampling
        included. Keywords pass through: since the upstream merge
        `prepare_sampled_ids` takes `channel_states=`."""
        if getattr(self, "_parity_record", None) is not None:
            self._stamps["target_event"] = self.device_event()
        with _traced(TARGET_DONE):
            return call(*args, **kwargs)

    def _draft_pass(self, key, call, *args, **kwargs):
        """`call`, a drafter pass, between two device events kept under `key`,
        the line field that times it. The first pass's start also ends the
        target of a forward that samples nothing. Under ATOM's profiler every TP
        rank opens the pass's range, recording or not."""
        if getattr(self, "_parity_record", None) is None:
            with _traced(MARK + key):
                return call(*args, **kwargs)
        start = self.device_event()
        self._stamps.setdefault("draft_event", start)
        with _traced(MARK + key):
            reply = call(*args, **kwargs)
        self._stamps[key] = [start, self.device_event()]
        return reply

    def _serial_pass(self, call, *args, **kwargs):
        """`call`, an EAGLE3 drafter pointing its context at its next pass,
        after a device event where that pass starts."""
        if getattr(self, "_parity_record", None) is not None:
            self._stamps.setdefault("pass_events", []).append(self.device_event())
        return call(*args, **kwargs)

    def propose_draft_token_ids(self, *args, **kwargs):
        call = super().propose_draft_token_ids
        return self._draft_pass("d_propose_ns", call, *args, **kwargs)

    def _gate_staging_reuse(self):
        return self._stamped("t_gate_ns", super()._gate_staging_reuse)

    def run_model(self, *args):
        if getattr(self, "_parity_record", None) is not None:
            from atom.utils.forward_context import get_forward_context

            # The event is recorded once to create it, then again just after
            # the read: neither a stall nor the event's creation comes between
            # the read and the record that places the launch.
            event = self.device_event()
            self._stamps["t_launch_event_ns"] = self.step_clock_ns()
            self._stamps["launch_event"] = self.device_event(event)
            context = get_forward_context().context
            self._stamps["forward_mode"] = getattr(context, "forward_mode", None)
        return self._stamped("t_launch_ns", super().run_model, *args)

    def step_stamps(self) -> tuple:
        """At a recorded forward's exit: its stamps, and its device window callable or None."""
        stamps, self._stamps = self._stamps, {}
        if self._launch.features:
            stamps["launch"] = self._launch.observe(self)
        sums = {key: stamps.pop((key, "blocked"), None) for key in BLOCKED_AT}
        if sums["t_launch_ns"]:
            before, end = sums["t_launch_ns"]
            end = (sums["t_recv_ns"] or sums["t_read_ns"] or (end,))[0]
            stamps["blocked_ns"] = self._blocked.between(before, end)
        launch = stamps.pop("launch_event", None)
        names = {
            "d_target_ns": "target_event",
            "d_recv_ns": "recv_event",
            "d_send_ns": "send_event",
        }
        marks = {k: [stamps.pop(e)] for k, e in names.items() if e in stamps}
        first = stamps.pop("draft_event", None)
        starts = stamps.pop("pass_events", [])
        passes = {
            k: stamps.pop(k) for k in ("d_draft_kv_ns", "d_propose_ns") if k in stamps
        }
        # Each draft pass runs to the next one's start, the last to the propose's exit.
        propose = passes.get("d_propose_ns")
        bounds = [*starts, propose[1]] if starts and propose else []
        if "d_target_ns" in marks and "d_draft_kv_ns" in passes:
            # It sampled after its draft-KV pass, the first drafter pass a
            # forward runs, and that pass is not the target's work.
            marks["d_target_ns"] += passes["d_draft_kv_ns"]
        elif first is not None:
            marks.setdefault("d_target_ns", [first])
        window = None
        if launch is not None:
            for key, (start, end) in passes.items():
                marks[key] = [end, launch, start]
            done, previous = self.device_event(), self._last_exit
            self._last_exit = done
            window = functools.partial(_window, previous, launch, done, marks, bounds)
        return stamps, window

    def exit(self):
        # Reached on a clean stop and on a SIGTERM (ATOM's exit handler).
        if getattr(self, "_parity_record", None) is not None:
            self._parity_record.write()
        if getattr(self, "_writes", None) is not None:
            self._writes.write()
        return super().exit()


def _traced(name: str):
    """A profiler range named `name` while ATOM's profiler runs, else none, so
    an untraced forward's host time pays for no range."""
    import torch

    if torch.autograd._profiler_enabled():
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


def _window(previous, launch, done, marks, bounds, wait: bool):
    """The device window's fields once `done` has completed, polled, else None;
    with `bounds`, `DRAFT_PASSES`, the time between each two of them."""
    while not done.query():
        if not wait:
            return None
        time.sleep(1e-3)
    fields = {"d_busy_ns": round(launch.elapsed_time(done) * 1e6)}
    for key, (event, *less) in marks.items():
        ms = launch.elapsed_time(event) - (less[0].elapsed_time(less[1]) if less else 0)
        fields[key] = round(ms * 1e6)
    if bounds:
        fields[DRAFT_PASSES] = [
            round(a.elapsed_time(b) * 1e6) for a, b in pairwise(bounds)
        ]
    if previous is not None:
        fields["d_gap_ns"] = round(previous.elapsed_time(launch) * 1e6)
    return fields


def read(directory, stage: int = 0) -> dict[int, list[dict]]:
    """A record directory's `stage` as {DP rank: steps}, each named request under its key.

    A request the record never named keeps its request id, so a record that
    ends inside a request's prefill parts from a full one at that request's
    first step. With `ARRIVALS`, the worker-clock times a workload was sent,
    the steps entered before the first send are the engine's start, not the
    workload's: they are left out and the rest renumbered from 0. A step
    naming requests with response ids holds them by key in `response_ids`,
    and their request ids by key in `request_ids`.
    A real rank's ``dp<rank>.pulls.jsonl`` (`clock.PullStamps`) gives each
    step the first ``t_pull_ns`` and ``t_put_ns`` at or after its
    ``t_exit_ns`` and before the next step's ``t_enter_ns``, where there is one.
    """
    directory, record, start = pathlib.Path(directory), {}, None
    if (directory / ARRIVALS).exists():
        start = min(read_json(directory / ARRIVALS)["sent_ns"])
    # Each rank's record, and not the cost lines a simulated run writes beside it.
    prefix = f"pp{stage}." if stage else ""
    for path in sorted(directory.glob(f"{prefix}dp*[0-9].jsonl")):
        steps = read_jsonl(path)
        if (pulls := path.with_suffix(".pulls.jsonl")).exists():
            join_stamps(steps, pulls)
        keys = {}
        for step in steps:
            for req_id, key, *response_id in step.pop("named"):
                keys[req_id] = key
                if response_id:
                    step.setdefault("response_ids", {})[key] = response_id[0]
                    step.setdefault("request_ids", {})[key] = req_id
        if start is not None:
            steps = [s for s in steps if s["t_enter_ns"] >= start]
            for i, step in enumerate(steps):
                step["step"] = i
        for step in steps:
            for row in (*step["batch"], *step.get("kv", ())):
                row[0] = keys.get(row[0], row[0])
        record[int(path.stem[len(prefix) + 2 :])] = steps
    return record


def transfers(directory, fields=None) -> dict[tuple[int, int], str]:
    """`TRANSFERS` in `directory` as {(prefill DP rank, prefill request id):
    decode response id}; empty without it. With `fields`, one prefill
    instance's `INSTANCE_FIELDS` values, only the transfers that came from it:
    each of a deployment's prefill instances numbers its transfer ids from its
    own sequence ids, so two instances' ids meet."""
    path = pathlib.Path(directory) / TRANSFERS
    if not path.exists():
        return {}
    return {
        (t["remote_dp_rank"], t["transfer_id"]): t["response_id"]
        for t in read_jsonl(path)
        if all(t.get(f) == v for f, v in (fields or {}).items())
    }


def _request_ranks(name: str, record: dict[int, list[dict]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, steps in record.items():
        for step in steps:
            for key, _num, _context in step["batch"]:
                if ranks.setdefault(key, rank) != rank:
                    raise ValueError(
                        f"the {name} record schedules request {key} on DP ranks "
                        f"{ranks[key]} and {rank}; a request lives on one rank, "
                        "so the key joins two requests."
                    )
    return ranks


def compare_routes(real, simulated) -> dict | None:
    """Two record directories' `ROUTES`, None unless both hold it: requests
    per DP rank in each; of the requests both routed, keyed as `read` keys
    twins, how many took the same rank; and the first of those, in the real
    run's order, that did not, with each run's line for it."""
    runs = {}
    for name, directory in (("real", real), ("simulated", simulated)):
        if not (path := pathlib.Path(directory) / ROUTES).exists():
            return None
        runs[name], seen = read_jsonl(path), collections.Counter()
        for line in runs[name]:
            n = seen[line["key"]]
            seen[line["key"]] += 1
            line["key"] += f".{n}" if n else ""
    other = {line["key"]: line for line in runs["simulated"]}
    both = [(a, other[a["key"]]) for a in runs["real"] if a["key"] in other]
    unlike = [{"real": a, "simulated": b} for a, b in both if a["rank"] != b["rank"]]
    width = max((len(x["reqs"]) for lines in runs.values() for x in lines), default=0)
    return {
        "requests_per_rank": {
            name: [sum(line["rank"] == r for line in lines) for r in range(width)]
            for name, lines in runs.items()
        },
        "routed_by_both": len(both),
        "routed_alike": len(both) - len(unlike),
        "first_unlike": unlike[0] if unlike else None,
    }


def compare(real, simulated, stage: int = 0) -> dict:
    """Two record directories: per DP rank of pipeline `stage`, the first step
    whose decision differs.

    A rank's entry is None where its steps all agree, else the step index and
    both steps, None for a record that has already ended. A record with no
    steps is refused: two runs that scheduled nothing agree, and that means
    nothing. Steps are compared and reported without timestamps, request
    or response ids or a real line's `launch`. Under `steps`, each rank's count of step pairs compared
    and of identical ones. Where both hold `ROUTES`, `routes` holds
    `compare_routes`'s answer.
    """
    records = {"real": read(real, stage), "simulated": read(simulated, stage)}
    for name, record in records.items():
        if not any(record.values()):
            raise ValueError(f"the {name} record has no steps to compare.")
        for steps in record.values():
            for step in steps:
                for key in (*TIMESTAMPS, "request_ids", "response_ids", "launch"):
                    step.pop(key, None)
    first, counts = {}, {}
    for rank in sorted(records["real"].keys() | records["simulated"].keys()):
        pairs = list(
            zip_longest(
                records["real"].get(rank, []), records["simulated"].get(rank, [])
            )
        )
        first[rank] = next(
            (
                {"step": i, "real": a, "simulated": b}
                for i, (a, b) in enumerate(pairs)
                if a != b
            ),
            None,
        )
        counts[rank] = {
            "compared": len(pairs),
            "identical": sum(a == b for a, b in pairs),
        }
    out = {
        "first_divergence": first,
        "steps": counts,
        "request_dp_rank": {
            name: _request_ranks(name, record) for name, record in records.items()
        },
    }
    if (routed := compare_routes(real, simulated)) is not None:
        out["routes"] = routed
    return out
