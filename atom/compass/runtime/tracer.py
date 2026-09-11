"""Build a model once, trace many batches through it.

``scripts/compass/graph_diff.py trace`` was one graph per process, and said so:
ATOM registers attention layers in a global table, so a process builds a model
once and a second build corrupts the first. That constraint is real and stays.
What was accidental is that the *process* was the unit of work -- every graph
paid the model build again, and, off a GPU box, the ~9.5s of imports before it.

Measured on the 27B at TP1 on a CPU node: 0.22s to build, 0.39-0.71s to trace,
inside a process that takes 10.3s wall from launch. A template cache that misses
therefore costs ten seconds if a miss means a new process and six tenths of a
second if it does not. This module is the difference: `ModelTracer.build` pays
the once-per-process part, `ModelTracer.trace` pays the per-batch part, and the
global-registration constraint is enforced rather than merely documented.

The script keeps its CLI and its behaviour; it imports the functions below
instead of defining them, so there is one definition of what a derived graph
contains and what its provenance says. A refactor that changed either would be
a silent change to every artifact derived afterwards, so the check that this
did not is to re-derive a frozen graph and diff it against the artifact -- not
to read the diff of this commit.

Nothing here touches a device unless `device` says so. Meta derivation is the
intended use.
"""

import contextlib
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

#: What ``--tokens`` means when nobody said otherwise. Named so a batch spec can
#: tell "the default was left alone" from "the caller asked for this many".
TRACE_TOKENS_DEFAULT = 8


class BuildRefusal(Exception):
    """A second model build in a process that already has one.

    Not an optimisation guard. ATOM registers attention layers into a global
    table at construction, so the second model's layers land beside the first's
    and both are traced wrong -- quietly, with plausible shapes. Refusing is the
    only honest answer; the caller wants a new process.
    """


def _free_port() -> str:
    """Ask the OS for a port nobody is using.

    A fixed port is wrong here. The container runs with host networking on a
    shared machine, so a hardcoded number collides with whatever else happens to
    hold it -- including a previous run of this same script -- and the failure
    (``EADDRINUSE`` from the rendezvous store) says nothing about tracing.
    """
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


# Named per region so a graph says what it holds, and a sum over two graphs
# can be checked for double counting or for a hole. `sampler` and
# `input preparation` are excluded everywhere because no region traces them;
# they are the remaining named gap in the step, not a covered term.
REGION_INCLUDES = {
    "body": ["model forward"],
    "head": ["compute_logits"],
    "both": ["model forward", "compute_logits"],
}
REGION_EXCLUDES = {
    "body": ["compute_logits", "sampler", "input preparation"],
    "head": ["model forward", "sampler", "input preparation"],
    "both": ["sampler", "input preparation"],
}


@dataclass(frozen=True)
class TraceRequest:
    """The per-batch inputs to one trace.

    Field names match what the provenance helpers read, because those helpers
    were written against an ``argparse`` namespace and moved here unchanged. A
    rename would be a rewrite of code whose correctness is established by the
    artifacts it has already produced.

    ``cudagraph_mode`` and ``capture_bucket`` are declared deployment inputs, not
    observations. Left ``None`` they stay ``None`` in the provenance, and a
    consumer must refuse rather than read that as a default.
    """

    tp: int = 1
    rank: int = 0
    region: str = "body"
    cudagraph_mode: Optional[str] = None
    capture_bucket: Optional[int] = None
    tokens: int = TRACE_TOKENS_DEFAULT
    model: str = ""


def head_placement(args, spec, notes) -> dict:
    """Whether production runs this step's LM head inside the replayed body.

    Not a function of TP. `run_model` reaches `compute_logits` down three
    different branches, and only one of them consults `logits_in_graph`:

      * `if not forward_mode.use_cudagraph:` -- prefill, or a decode forced
        eager -- calls `compute_logits(hidden_states)` directly
        (model_runner.py:3182). Eager at every width, TP1 included.
      * the PIECEWISE decode path calls it directly too
        (model_runner.py:3230), again at every width: the dense pieces
        self-capture, the head is not among them.
      * only the manual whole-forward (FULL) decode replay asks
        `if self.logits_in_graph` (model_runner.py:3238), and
        `logits_in_graph = world_size == 1 and not is_tbo` (:4104).

    So a prefill at TP1 has an eager head, and so does a PIECEWISE decode at
    TP1, and classifying either from TP alone puts the head in a graph that
    does not contain it. Four conditions, all required: decode, FULL cudagraph
    mode, TP1, TBO off.

    `cudagraph_mode` is a declared deployment input, not something a derivation
    can observe, so without it the answer is `None` -- recorded as unanswered.
    A consumer must refuse rather than read `None` as `False`.
    """
    region = getattr(args, "region", "body")
    mode = getattr(args, "cudagraph_mode", None)
    kind = step_kind(args, spec)
    if mode is None or kind is None:
        in_replay = None
        why = ("undetermined: needs both the step kind and --cudagraph-mode; "
               "TP alone does not decide this")
    else:
        in_replay = (kind == "decode" and mode == "full" and args.tp == 1)
        why = (f"{kind} step, cudagraph mode {mode}, tp={args.tp}, TBO assumed "
               "off")
    return {
        # What this graph holds, which follows `--region` and nothing else.
        "in_this_graph": region in ("head", "both"),
        # What production does, which is the question a composition needs.
        "in_replayed_body_graph": in_replay,
        "why": why,
        "rule": ("decode AND cudagraph_mode==FULL AND tensor_parallel_size==1 "
                 "AND not is_tbo; model_runner.py:3182 (eager branch), :3230 "
                 "(piecewise branch), :3238-3241 (full-replay branch), :4104 "
                 "(logits_in_graph)"),
        "assumes": "two-batch overlap off",
        # The head's rows are not the body's. A FULL capture projects the whole
        # padded bucket, `compute_logits(outputs[:num_tokens])` with
        # `num_tokens = bs * max_q_len` (:4297), and the replay slices
        # `graph_logits[key][:num_tokens]` afterwards (:3239). Every eager head
        # instead receives hidden states already cut to `scheduled_bs *
        # max_q_len` (:3189, :3228-3230). This graph is the eager shape.
        "rows_padded_to_capture_bucket": False,
        "rows_into_compute_logits": notes.get("hidden_rows"),
    }


def collective_registration(args, spec) -> dict:
    """Which data path this region's collectives take in production.

    `CustomAllreduce` runs one operator over two of them and the recorded
    signature carries neither: with `registered_input=True` the peers read the
    input buffer directly, with False it is copied into the IPC pool first and
    reduced from there. The branch is `CustomAllreduce._IS_CAPTURING`
    (custom_all_reduce.py:1231), which only `CustomAllreduce.capture()` sets --
    entered by `parallel_state.graph_capture()`, which `model_runner.py:4158`
    wraps its capture in, and not by a bare `torch.cuda.graph`. Measured rather
    than read off the source: `agent_scratch/g4/ar_probe/README.md`, where the
    same 4x5120 reduction costs 6.29 us on one path and 9.06 us on the other.

    So a body production replays from a FULL decode capture reduces on the
    registered path, and an eager region -- an eager body, and the TP>1 head,
    which the runner computes after the replay -- takes the copy path. The
    PIECEWISE case is left unanswered: its pieces are captured by the compiler's
    wrapper rather than by `model_runner`'s own `graph_capture()`, and that has
    not been audited. Unanswered is also what an undeclared `--cudagraph-mode`
    gets, for the reason `head_placement` gives: a derivation cannot observe a
    deployment input, and a consumer must refuse rather than read `None` as
    either path.
    """
    region = getattr(args, "region", "body")
    mode = getattr(args, "cudagraph_mode", None)
    kind = step_kind(args, spec)
    if region == "head":
        # `compute_logits` runs outside any replay at TP>1 (model_runner.py:
        # 3182, 3230); at TP1 the head holds no collective to price.
        return {"regime": "unregistered",
                "why": "the head runs eagerly after the replay, and an eager "
                       "call takes the copy path"}
    if mode is None or kind is None:
        return {"regime": None,
                "why": "undetermined: needs both the step kind and "
                       "--cudagraph-mode; neither is observable from a trace"}
    if mode == "eager":
        return {"regime": "unregistered",
                "why": "an eager step captures nothing, so the communicator is "
                       "never armed"}
    if mode == "full":
        if kind == "decode":
            return {"regime": "registered",
                    "why": "a FULL decode replays a graph captured inside "
                           "parallel_state.graph_capture() "
                           "(model_runner.py:4158), which arms the "
                           "communicator"}
        return {"regime": "unregistered",
                "why": f"a {kind} step runs eagerly even under FULL cudagraph "
                       "mode, which captures decode buckets only"}
    return {"regime": None,
            "why": "PIECEWISE pieces are captured by the compiler's wrapper "
                   "rather than model_runner's graph_capture(); not audited, "
                   "so not claimed"}


def step_kind(args, spec):
    """``"prefill"``, ``"decode"``, or ``None`` when nothing says.

    The branch `run_model` takes turns on this, so a graph that cannot state it
    cannot state where its head runs either.
    """
    if spec is None:
        return None
    if getattr(spec, "num_prefill_tokens", 0):
        return "prefill"
    lens = tuple(getattr(spec, "query_lens", ()) or ())
    if not lens:
        return None
    return "decode" if max(lens) == 1 else "prefill"


def execution_record(args, spec, notes) -> dict:
    """What this graph's operators were traced over, against what production
    executes.

    Both counts, because they differ and the difference is not visible in the
    operators. A decode of 20 sequences whose graph was captured at
    `running_bs` 32 forwards 32 rows -- `self.model(input_ids[:num_tokens_pad])`
    with `num_tokens_pad = running_bs * max_q_len` (model_runner.py:3192, 3841)
    -- and slices to `scheduled_bs * max_q_len` only afterwards (:3189, :3234).
    A body graph traced at 20 rows is therefore the *eager* body; charging it
    for a bucketed replay undercounts every dense operator in the step.

    `capture_bucket` is a declared input for the same reason `cudagraph_mode`
    is: a derivation cannot see which bucket a deployment snapped to. Absent,
    it is recorded as unknown rather than assumed equal to the real count.
    """
    bucket = getattr(args, "capture_bucket", None)
    return {
        "step_kind": step_kind(args, spec),
        "cudagraph_mode": getattr(args, "cudagraph_mode", None),
        # Rows this trace put through the model.
        "body_rows_traced": notes.get("body_rows"),
        # Rows the step really has, before any padding.
        "rows_real": spec.num_tokens if spec is not None else args.tokens,
        # Rows a replay would forward, when the step replays one.
        "body_rows_executed": bucket,
        "capture_bucket": bucket,
        "head_rows_traced": notes.get("hidden_rows"),
        "regions_traced": REGION_INCLUDES[getattr(args, "region", "body")],
        "regions_not_traced": REGION_EXCLUDES[getattr(args, "region", "body")],
        # How many operators the trace watched an output die from. Recorded
        # because its absence is what a reader needs to see: a graph with no
        # deaths is walked for memory by a last-read rule that returns a number
        # either way, so nothing downstream ever says the liveness was guessed.
        "operators_with_observed_deaths": notes.get("deaths_stamped"),
    }


def init_env(tp: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT") or _free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def trace_regions(model, input_ids, positions, topology=None, on_meta=False,
                  spec=None, region="body"):
    """Run one forward under the tracers, returning the combined graph.

    ``region`` says what is recorded.

    ``body`` is the model forward, and is what every graph before this one
    held. ``head`` is the runner's ``compute_logits``: the LM-head projection
    and, at TP>1, the all-gather over its vocab shard. No body graph contains
    either, so both were priced at zero -- the failure mode this project keeps
    naming, an unmeasured family summed as nothing and the step called
    complete. ``both`` records the two together.

    For ``head`` the forward still runs, because the head needs its hidden
    states, but outside the tracers so its operators are not counted twice.

    The head is not a projection of the whole chunk. `ParallelLMHead.forward`
    reads the forward context and, on prefill, keeps one row per sequence
    (`x[cu_seqlens_q[1:] - 1]`); on decode it keeps every row, because decode
    already has one row per sequence. So this must be traced with the batch
    spec installed, exactly as attention is, or a 16384-token prefill would
    record a head 16384 rows wide against the 1 row production computes.

    Still outside every region: sampling, input preparation, and the runner's
    own host work. Those are named in the graph's provenance rather than left
    for a reader to assume covered.
    """
    from atom.compass.runtime.derive import (record_collectives,
                                             redirect_device_factories)
    from atom.compass.runtime.meta import MetaOpTracer
    from atom.compass.runtime.triton_trace import TritonLaunchTracer

    ops = MetaOpTracer(topology=topology)
    triton = TritonLaunchTracer(graph=ops.graph)
    collectives = record_collectives(ops.graph, tracer=ops)
    # Only on meta, and reported: see redirect_device_factories.
    factories = (redirect_device_factories() if on_meta
                 else contextlib.nullcontext())
    # The batch's metadata, on the forward context, where attention reads it.
    # Without this the model still traces -- attention runs, its shapes are
    # right -- and every attention operator is recorded with no context, which
    # makes it unpriceable. See atom/compass/runtime/batch_spec.py.
    if spec is not None:
        from atom.compass.runtime import batch_spec as bs

        installed = bs.install(spec)
    else:
        installed = contextlib.nullcontext()
    if region not in ("body", "head", "both"):
        raise ValueError(f"unknown region {region!r}")

    # Only when the body's operators are this graph's. A head region runs a
    # forward too, but into a graph that is thrown away, so reporting its rows
    # here would describe work the artifact does not contain.
    notes: dict = ({} if region == "head"
                   else {"body_rows": int(input_ids.shape[0])})
    with factories, installed, torch.inference_mode():
        if region == "head":
            # The forward still has to run, because the head needs its hidden
            # states -- but its operators are not this graph's, so they go
            # into a graph that is thrown away.
            #
            # It cannot simply run untraced. On meta a Triton launch reaches
            # the real Triton runtime and dies there ("0 active drivers"),
            # so the launch tracer is not only a recorder: it is what makes a
            # device-free forward possible at all. Hence a second, discarded
            # tracer rather than no tracer.
            scratch = MetaOpTracer(topology=topology)
            with record_collectives(scratch.graph, tracer=scratch), \
                    TritonLaunchTracer(graph=scratch.graph), scratch:
                hidden = model(input_ids, positions)
            notes["hidden_rows"] = int(hidden.shape[0])
            t0 = time.perf_counter()
            with collectives, triton, ops:
                model.compute_logits(hidden)
        elif region == "both":
            t0 = time.perf_counter()
            with collectives, triton, ops:
                model.compute_logits(model(input_ids, positions))
        else:
            t0 = time.perf_counter()
            with collectives, triton, ops:
                model(input_ids, positions)
        trace_s = time.perf_counter() - t0
    # The same stamp the capture path applies, from the same tracer, and for
    # the same reason: the tracer watched every output die and nothing wrote
    # that down. Only the capture path stamped, and every template on disk came
    # down this one -- so every derived graph reached the memory walk with no
    # `dies_at` at all, and the walk fell back to a last-read rule silently.
    # After the forward has returned, which is when the finalizers fire.
    notes["deaths_stamped"] = ops.stamp_deaths()
    return ops.graph, trace_s, getattr(factories, "redirected", 0), notes


@dataclass
class TracedGraph:
    """One derived graph and what it cost to derive.

    The seconds are here rather than printed because the caller that matters is
    a cache: a template miss has to be able to say what the miss cost without
    parsing stdout.
    """

    graph: Any
    trace_s: float
    redirected: int
    notes: dict


#: One model per process, and which one. ATOM's attention registry is global,
#: so the second build is not slower, it is wrong. Module-level because that is
#: the scope of the registry it guards.
_BUILT: Optional[str] = None


class ModelTracer:
    """A built model, reusable across batches.

    Construct through :meth:`build`. Once built, :meth:`trace` is the whole
    per-batch cost: a forward on meta with the tracers installed, plus the
    provenance assembly, and nothing else.

    One per process. A second :meth:`build` raises :class:`BuildRefusal` --
    see its docstring for why that is correctness and not tidiness.
    """

    def __init__(self, *, model, config, arch, device, tp, model_path,
                 build_s, bootstrap=None, rank: int = 0):
        self.model = model
        self.config = config
        self.arch = arch
        self.device = device
        self.tp = tp
        self.model_path = model_path
        self.build_s = build_s
        #: Which logical rank of the TP group the built modules currently hold.
        #: Moved by `set_rank`, which `trace` calls when a request asks for a
        #: rank this model is not on.
        self.rank = rank
        #: How many times that move has happened, and how many modules the last
        #: one touched -- recorded so a graph says whether its rank was built
        #: or rebound.
        self.rank_rebinds = 0
        self.rank_modules_rebound = 0
        #: What `install_from_target` reported, when a replay target was used
        #: to answer the chip question off a GPU box. `None` when the machine
        #: answered for itself.
        self.bootstrap = bootstrap
        self.traces = 0
        self.trace_seconds = 0.0

    @classmethod
    def build(cls, model_path: str, tp: int, device: str = "meta", *,
              replay_target: Optional[str] = None,
              on_duplicate: str = "refuse", rank: int = 0) -> "ModelTracer":
        """Pay the once-per-process cost: env, process group, config, weights.

        ``rank`` is which logical rank of the TP group the layers are built at.
        It is not a second build axis: one process builds once, and `trace`
        moves the built model between ranks through `set_rank`.

        ``replay_target`` answers the chip question off a GPU box. AITER
        resolves the architecture at import time by shelling out to
        ``rocminfo``, which fails where there is no GPU -- and derivation's
        whole point is to run there. The replay path already answers that
        question from a captured record, so derivation uses the same seam
        rather than a second one.

        ``on_duplicate="allow"`` exists for tests that build a stub, never for
        a real model.
        """
        global _BUILT
        if _BUILT is not None and on_duplicate != "allow":
            raise BuildRefusal(
                f"this process already built {_BUILT!r}. ATOM registers "
                "attention layers in a global table at construction, so a "
                "second model's layers land beside the first's and both trace "
                "wrong with plausible shapes. Use a new process.")

        init_env(tp)
        bootstrap = None
        if replay_target:
            # Before the `aiter` import below, which is what triggers the query.
            from atom.compass.replay.bootstrap import install_from_target

            bootstrap = install_from_target(replay_target)
        from aiter import init_dist_env

        # Derivation builds the group at world size ONE, whatever TP width is
        # being derived, and then tells the group to report the wider width.
        #
        # A real group of size N needs N processes to arrive. That defeats the
        # whole purpose here: the point of derivation is to produce a sharded
        # rank's graph from one process, on no GPUs, for a configuration nobody
        # has run. Asking gloo for a world of 2 from a single process simply
        # waits forever.
        #
        # ATOM already solves this for benchmarking -- `apply_simulated_tp`
        # reports a logical width while the real group stays physical, all the
        # way down to one rank -- and the reason it works there is the reason it
        # works here: every shard-size computation bottoms out at
        # `get_tp_group().world_size`. Its caveat, that collectives over absent
        # ranks make the *output* meaningless, costs derivation nothing, because
        # derivation never looks at the output. Only shapes are recorded, and
        # shapes stay right.
        init_dist_env(1, rankID=0, backend="gloo",
                      distributed_init_method="env://", local_rank=0)

        from atom.config import Config, set_current_atom_config
        from atom.model_engine.model_runner import support_model_arch_dict
        from atom.utils import resolve_obj_by_qualname

        config = Config(model=model_path, tensor_parallel_size=tp,
                        load_dummy=True)
        set_current_atom_config(config)
        if tp > 1:
            from atom.compass.runtime.derive import simulate_group_width

            # The rank the layers are *constructed* at; `set_rank` moves it.
            simulate_group_width(tp, rank=rank)
        elif rank:
            raise BuildRefusal(
                f"tp=1 has only rank 0; a tracer cannot build rank {rank}.")
        arch = config.hf_config.architectures[0]
        model_class = resolve_obj_by_qualname(support_model_arch_dict[arch])

        dev = torch.device(device)
        # Build in the model's own dtype. Left at the fp32 default, meta happily
        # traces kernels that real hardware rejects -- AITER's fused qk-rmsnorm
        # takes only fp16/bf16 -- so the two graphs would not be comparable in
        # the one way that matters.
        build_t0 = time.perf_counter()
        with torch.device(dev):
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(config.torch_dtype)
            try:
                model = model_class(config)
            finally:
                torch.set_default_dtype(prev_dtype)
        if dev.type != "meta":
            model = model.to(dev)
        build_s = time.perf_counter() - build_t0

        _BUILT = model_path
        return cls(model=model, config=config, arch=arch, device=dev, tp=tp,
                   model_path=model_path, build_s=build_s, bootstrap=bootstrap,
                   rank=rank)

    def set_rank(self, rank: int) -> int:
        """Move the built model onto logical ``rank``. Returns modules moved.

        A tracer cannot build a second model (see :meth:`build`), so a process
        that prices every rank of a TP4 deployment moves one model between them
        instead. What moves is enumerated and refusable in
        :func:`~atom.compass.runtime.derive.rebind_logical_rank`; what does not
        move -- every shard *shape* -- is rank-independent by construction.
        """
        if rank == self.rank:
            return 0
        if self.tp == 1:
            raise BuildRefusal(
                f"tp=1 has only rank 0; this tracer cannot serve rank {rank}.")
        from atom.compass.runtime.derive import rebind_logical_rank

        moved = rebind_logical_rank(self.model, rank, logical=self.tp)
        self.rank = rank
        self.rank_rebinds += 1
        self.rank_modules_rebound = moved
        return moved

    def trace(self, spec=None, *, request: Optional[TraceRequest] = None,
              rank: int = 0, region: str = "body",
              cudagraph_mode: Optional[str] = None,
              capture_bucket: Optional[int] = None,
              tokens: Optional[int] = None) -> TracedGraph:
        """Trace one batch. Everything the built model is not.

        ``spec`` is a :class:`~atom.compass.runtime.batch_spec.BatchSpec`, or
        ``None`` for a bare forward with no forward context installed -- which
        traces, but records every attention operator with no context and so
        unpriceable.
        """
        from atom.compass.core.graph import GraphKey

        if request is None:
            request = TraceRequest(
                tp=self.tp, rank=rank, region=region,
                cudagraph_mode=cudagraph_mode, capture_bucket=capture_bucket,
                tokens=(spec.num_tokens if spec is not None
                        else (tokens if tokens is not None
                              else TRACE_TOKENS_DEFAULT)),
                model=self.model_path)
        if request.tp != self.tp:
            raise BuildRefusal(
                f"this tracer built the model at tp={self.tp}; a request at "
                f"tp={request.tp} would key a graph the shards do not match. "
                "Build a tracer per width.")
        # Before the forward, not after the key is stamped. `rank_coords` was
        # previously a label written onto rank 0's graph whatever rank asked.
        self.set_rank(request.rank)

        if spec is None:
            from atom.compass.runtime.meta import derived_inputs

            inputs = derived_inputs(request.tokens, self.device)
        else:
            from atom.compass.runtime import batch_spec as bs

            inputs = bs.model_inputs(spec, self.device)

        # The trace runs in the model's dtype too, not only the build. A
        # library that creates a tensor without naming one gets the ambient
        # default, and AITER's dispatch dummy is exactly that: restored to
        # fp32 it is recorded as `...|1;4,5120;5120;4,5120|float32,bfloat16,
        # bfloat16,bfloat16`, against the capture's all-bfloat16, and every
        # fused qk-rmsnorm in the graph then misses its price by dtype alone.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.config.torch_dtype)
        try:
            graph, trace_s, redirected, notes = trace_regions(
                self.model, *inputs, topology={"tp": self.tp},
                on_meta=self.device.type == "meta", spec=spec,
                region=request.region)
        finally:
            torch.set_default_dtype(prev_dtype)

        graph.key = GraphKey.of(
            model_id=f"{self.arch}@{os.path.basename(self.model_path.rstrip('/'))}",
            topology={"tp": self.tp},
            rank_coords={"tp": request.rank},
            # Per-request scheduled tokens, which is what the runner records
            # (`runner.py`: `shape.num_scheduled_tokens`). A bare body pass is
            # one sequence of that many. The key still does not carry context
            # length, so a decode and a chunked prefill of the same query
            # lengths key alike -- the spec below is what tells them apart.
            batch_signature=spec.query_lens if spec else (request.tokens,),
        )
        graph.provenance = self.provenance(request, spec, notes, redirected)
        self.traces += 1
        self.trace_seconds += trace_s
        return TracedGraph(graph=graph, trace_s=trace_s, redirected=redirected,
                           notes=notes)

    def provenance(self, request: TraceRequest, spec, notes,
                   redirected: int) -> dict:
        """What the graph says about itself.

        Separate from :meth:`trace` so a reader can see the whole record in one
        place, and so a test can check a field without building a 27B model.
        """
        from atom.compass.runtime.meta import LIVENESS_INSTRUMENTATION

        registration = collective_registration(request, spec)
        prov = {
            "source": "derivation" if self.device.type == "meta" else "capture",
            "device": self.device.type,
            # Which revision of the tracer produced this graph's liveness
            # fields -- `inputs_from`, `output_aliases`, `dies_at`. A graph on
            # disk without this key is version 1: derived before 2026-09-11,
            # when every meta storage keyed to the same address and all three
            # were wrong in the same direction. Prices are unaffected at either
            # version. Old artifacts are left exactly as they are and read as 1;
            # see `atom.compass.runtime.meta.LIVENESS_INSTRUMENTATION`.
            "liveness_instrumentation": LIVENESS_INSTRUMENTATION,
            "compilation_level": 0,  # a bare model call is never compiled
            "tokens": request.tokens,
            # Which logical rank of the TP group this graph is of, and whether
            # the model was constructed on that rank or moved onto it after the
            # fact. A reader comparing two ranks' graphs from one process needs
            # to know the second is a rebind and what the rebind touched.
            "logical_rank": self.rank,
            "rank_binding": "built" if self.rank_rebinds == 0 else "rebound",
            "rank_rebinds": self.rank_rebinds,
            "rank_modules_rebound": self.rank_modules_rebound,
            # How many device-typed factory calls were sent to meta so the
            # operator after them could dispatch. Recorded, not hidden.
            "device_factories_redirected": redirected,
            # What this graph is a graph of, and -- as important -- what it is
            # still not. A reader summing it must be able to see the gap without
            # reading this script.
            "region": request.region,
            "includes": REGION_INCLUDES[request.region],
            "excludes": REGION_EXCLUDES[request.region],
            # How each collective's identity was established, because "verified"
            # means two different things here. The all-reduce is a real custom
            # op and a capture records it, so its name is observed. The
            # all-gather is not: the decorator above it is commented out in the
            # installed aiter, so nothing reaches the dispatcher and no capture
            # -- of this graph or any other -- will contain it. Its name is a
            # chosen label for a call confirmed by running two real ranks, and a
            # reader comparing this graph against a captured one should expect
            # that operator to be present here and absent there. See
            # `atom.compass.runtime.derive.record_collectives`.
            "collectives": {
                "aiter::all_reduce_": "dispatcher-recorded",
                "aiter::all_gather_unreg": "two-rank probe; not "
                                           "dispatcher-visible",
            },
            # Which of the communicator's two data paths those collectives run
            # on in production, and why. The signature carries the message and
            # not the path, so a price measured on the other one matches
            # exactly; a library lookup reads this field and refuses when it is
            # null rather than spending whichever measurement was loaded.
            #
            # "required" is the whole of the name. This is what the region
            # needs, not what any benchmark did: a generic pricing run measures
            # the path it actually took, which for `microbench`'s bare
            # `torch.cuda.graph` is the copy path whatever the graph it was
            # pricing requires. The price side records
            # `collective_registration_measured` and the library refuses to read
            # this key from a price list at all.
            "collective_registration_required": registration["regime"],
            "collective_registration_required_why": registration["why"],
            # Where the LM head runs relative to a *graph*, which is not the
            # same answer at every width and is how a step gets counted twice.
            # `ModelRunner.logits_in_graph = self.world_size == 1 and not
            # is_tbo` (model_runner.py:4104), with `self.world_size` the
            # tensor-parallel size (model_runner.py:642). At TP1 a level-3
            # capture replays `compute_logits` inside the body graph; at TP>1
            # the runner computes logits eagerly after the replay and the body
            # graph genuinely does not contain them.
            #
            # This graph is a level-0 eager derivation and traces one region at
            # a time, so `in_this_graph` follows `region` and nothing else. The
            # second field is about the level-3 graph of the same configuration
            # -- the thing a transfer is measured against -- and it is recorded
            # here because a source step whose body already contains the head
            # must not then be given a head term, nor have one folded into a
            # body correction.
            "head_placement": head_placement(request, spec, notes),
            "execution": execution_record(request, spec, notes),
        }
        if request.region in ("head", "both"):
            # Whether production runs this head at all, which is not a property
            # of the shape. A batch of pure middle chunks samples nothing and
            # the runner skips `compute_logits` entirely, so the graph below is
            # then a template for the step that *would* run it -- correct to
            # derive, wrong to charge. Recorded so a reader of the artifact
            # cannot sum it without meeting the question.
            prov["head_runs"] = (spec.produces_output()
                                 if spec is not None else None)
        if spec is not None:
            # The batch this graph is a graph of, written down in full, block
            # table included. Without it "4 tokens" is all a reader gets, and
            # four decodes at context 66 and a four-token prefill produce the
            # same number with two orders of magnitude between their KV traffic.
            prov["batch_spec"] = spec.to_dict()
        else:
            prov["forward_context"] = "none installed"
        return prov

    def describe(self) -> str:
        return (f"ModelTracer({self.arch} tp={self.tp} on {self.device.type}; "
                f"built in {self.build_s:.2f}s, {self.traces} traces in "
                f"{self.trace_seconds:.3f}s)")


class DeriveRefusal(Exception):
    """A step shape that cannot be turned into a batch without inventing one.

    A ``StepShape`` is what the oracle is asked about; a ``BatchSpec`` is what a
    trace needs. The gap between them is deployment configuration -- block size,
    maximum context, MRoPE layout -- and per-request facts a shape does not
    carry. Filling the gap with plausible defaults produces a graph of a batch
    nobody scheduled.
    """


class ShapeDeriver:
    """Turn a ``StepShape`` into a derived graph, for a declared deployment.

    This is the ``derive`` callable
    :class:`~atom.compass.runtime.templates.TemplateGraphs` takes: what a
    template cache does on a miss. It is deliberately the *only* thing that
    turns a shape into a batch, so the assumptions are in one place and each one
    is either declared here or refused.

    Declared, not inferred: ``block_size``, ``max_model_len`` and
    ``position_rows`` are properties of the deployment. A shape does not carry
    them and a default would be a guess with two orders of magnitude in it --
    MRoPE's ``position_rows=3`` alone triples the positions tensor.

    Refused, not guessed: a prefill shape whose ``produces_output`` is ``None``.
    Whether a chunk is a request's last is the scheduler's to know, it decides
    whether the LM head runs at all, and neither answer is safe to assume.

    Left to the block policy: ``prompt_lens``. A shape says how long each
    request's history is, not how much of it was prompt, and only the block
    allocation reads the difference. ``BatchSpec.admitted_lens`` already falls
    back to ``cached_lens`` (batch_spec.py:222), which is that policy's own
    documented default rather than something invented here. It still moves the
    allocator fields, which is why a bound graph refuses to reuse them without
    an ``AllocationSource``.
    """

    def __init__(self, tracer: ModelTracer, *, block_size: int,
                 max_model_len: int, position_rows: int = 1,
                 block_policy: str = "rounds", region: str = "body",
                 cudagraph_mode: Optional[str] = None) -> None:
        self.tracer = tracer
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.position_rows = position_rows
        self.block_policy = block_policy
        self.region = region
        self.cudagraph_mode = cudagraph_mode
        self.derivations = 0
        self.seconds = 0.0
        #: Why a shape was not derived, keyed the way the template cache keys
        #: it. Per instance, not per class: two deployments refuse different
        #: shapes and a shared dict would attribute one's refusal to the other.
        self.refusals: dict = {}

    def spec_for(self, shape):
        """The batch a trace of ``shape`` would run, or a refusal."""
        from atom.compass.runtime.batch_spec import BatchSpec

        queries = tuple(int(q) for q in shape.num_scheduled_tokens)
        contexts = tuple(int(c) for c in shape.context_lens)
        if len(queries) != len(contexts):
            raise DeriveRefusal(
                f"{len(queries)} query lengths against {len(contexts)} context "
                "lengths; a row is a pair and this shape has no rows")
        if not queries:
            raise DeriveRefusal("an empty batch has no graph")
        if max(contexts) > self.max_model_len:
            raise DeriveRefusal(
                f"context {max(contexts)} exceeds the declared max_model_len "
                f"{self.max_model_len}; this deployment could not run this step")

        prefill = int(getattr(shape, "num_prefill_tokens", 0) or 0)
        kind = "decode" if (prefill == 0 and max(queries) == 1) else "prefill"

        final = None
        if kind == "prefill":
            produces = getattr(shape, "produces_output", None)
            if callable(produces):
                produces = produces()
            if produces is None:
                raise DeriveRefusal(
                    "a prefill shape that does not say whether it produces "
                    "output. Whether a chunk is a request's last decides "
                    "whether the LM head runs at all, and it cannot be read "
                    "off the lengths -- the scheduler has to say.")
            # One bool for the batch is what a StepShape carries, so it is
            # spread over the rows rather than invented per request. The head's
            # row count is then right for an all-or-nothing batch and wrong for
            # a mixed one; a mixed batch needs per-request truth, which this
            # class cannot manufacture.
            final = tuple([bool(produces)] * len(queries))

        spec = BatchSpec(
            kind=kind, query_lens=queries, context_lens=contexts,
            block_size=self.block_size, max_model_len=self.max_model_len,
            capture_bucket=shape.capture_bucket,
            block_policy=self.block_policy,
            position_rows=self.position_rows,
            is_final_chunk=final,
            notes={"why": "derived from a StepShape by ShapeDeriver; "
                          "prompt_lens left to the block policy's "
                          "cached_lens default",
                   "declared": {"block_size": self.block_size,
                                "max_model_len": self.max_model_len,
                                "position_rows": self.position_rows,
                                "block_policy": self.block_policy}})
        spec.validate()
        return spec

    def __call__(self, shape) -> Optional[dict]:
        """Derive, and return the graph as a dict -- what a cache stores.

        Returns ``None`` on a refusal rather than raising, because the caller is
        a cache whose whole job is to record why a shape was not served. The
        reason is on :attr:`refusals`.
        """
        try:
            spec = self.spec_for(shape)
        except (DeriveRefusal, ValueError) as exc:
            self.refusals[template_key_of(shape)] = str(exc)
            return None
        traced = self.tracer.trace(
            spec, request=TraceRequest(
                tp=self.tracer.tp,
                rank=int((shape.rank_coords or {}).get("tp", 0)),
                region=self.region, cudagraph_mode=self.cudagraph_mode,
                capture_bucket=shape.capture_bucket,
                tokens=spec.num_tokens, model=self.tracer.model_path))
        self.derivations += 1
        self.seconds += traced.trace_s
        return traced.graph.to_dict()

    def describe(self) -> str:
        return (f"ShapeDeriver(block={self.block_size}, "
                f"max_model_len={self.max_model_len}, "
                f"position_rows={self.position_rows}, region={self.region}; "
                f"{self.derivations} derivations in {self.seconds:.3f}s)")


def template_key_of(shape):
    """The template cache's key, imported late to keep this module importable.

    ``templates`` has no torch dependency and this module does; keeping the
    import inside the call means a test can key a shape without a build.
    """
    from atom.compass.runtime.templates import template_key

    return template_key(shape)
