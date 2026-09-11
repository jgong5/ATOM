"""Build a sharded rank's model in one process, on no GPUs.

Derivation exists to produce a graph for a configuration nobody has run — a TP
width there are no devices for, a rank whose peers were never started. A real
process group cannot do that: asking gloo for a world of N from one process
waits forever for peers that will never arrive.

ATOM already has the mechanism. `atom/distributed/simulated_tp.py` makes the TP
group *report* a logical width while the real group stays smaller, down to a
single rank, because every shard-size computation in the tree bottoms out at
`get_tp_group().world_size`. Its documented caveat — that collectives covering
absent ranks make the model's output meaningless — costs derivation nothing:
derivation never reads the output. It records shapes, and shapes stay right.

What this module adds is only the entry point. `apply_simulated_tp` decides how
wide the real group is from `torch.cuda.device_count()`, which is the right
question when the caller is a worker process holding a device and the wrong one
here: derivation's real group is always exactly one rank, however many GPUs the
machine happens to have. On an 8-GPU box that heuristic concludes a TP2
derivation needs no simulation at all, and the model then builds unsharded —
silently, and looking entirely normal.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["simulate_group_width", "record_collectives",
           "redirect_device_factories"]


def simulate_group_width(logical: int, physical: int = 1) -> None:
    """Make the TP group report ``logical`` ranks over a ``physical``-rank group.

    Call after the process group exists and before the model is built: layers
    read the width while they are being constructed, so patching afterwards
    changes nothing that has already been sized.
    """
    if logical <= physical:
        return

    from aiter.dist.parallel_state import get_tp_group

    from atom.distributed.simulated_tp import _patch_group

    group = get_tp_group()
    if group.world_size != physical:
        raise RuntimeError(
            f"TP group has {group.world_size} ranks, expected {physical}. "
            "Derivation builds the group at world size one and simulates the "
            "rest; something else initialised it."
        )
    _patch_group(group, logical, physical)
    logger.info(
        "ATOMCompass: deriving rank 0 of a TP%d deployment from a %d-rank group. "
        "Shapes match TP%d; no collective is performed and no output is read.",
        logical, physical, logical,
    )


def head_gather_opspec(group, input_, dim: int = -1, group_name: str = "tp"):
    """The head all-gather as one operator, for both recording and pricing.

    One definition, because the two uses have to agree exactly. A derivation
    records this operator into a graph; a two-rank harness performs the real
    collective and writes a price under this operator's signature. If those
    signatures are built by two pieces of code they will agree on most fields
    and drift on one, and the symptom is not an error -- it is a lookup miss,
    which prices tensor-parallel communication at zero.

    ``dim`` is the axis production gathers along, and the output shape follows
    from it and the group's width: that axis grows by the width, every other is
    untouched. The width comes from ``group.world_size``, which under simulated
    TP is already the logical deployment width even though one rank is running.

    Two of the operator's six declared arguments -- ``(_fa, inp, reg_buffer,
    out, reg_bytes, dim)`` -- are raw addresses, and they are deliberately not
    recorded. An address is not a property of the step, and a signature
    carrying one would never match twice. That is also why ``abi`` is
    ``"live-state"``: this operator is priced by being performed, never by
    being rebuilt.
    """
    from atom.compass.core.graph import OpSpec

    width = int(group.world_size)
    shape = list(int(d) for d in input_.shape)
    axis = dim % len(shape)
    shape[axis] *= width
    return OpSpec(
        name="aiter::all_gather_unreg",
        input_shapes=(tuple(int(d) for d in input_.shape),),
        output_shapes=(tuple(shape),),
        dtypes=(str(input_.dtype).replace("torch.", ""),),
        # A gather concatenates its inputs; it does not convert them. Recorded
        # rather than left empty, because here it is known and not inferred.
        output_dtypes=(str(input_.dtype).replace("torch.", ""),),
        group=group_name,
        # The group, explicitly, rather than left to be inferred from the ratio
        # of the two shapes. A pricer for this operator does not rebuild a call
        # -- it performs the real collective -- so it needs to know which group
        # and how wide, and a width recovered by dividing shapes is a guess
        # that happens to be right.
        context=(("group", str(group.unique_name)),
                 ("group_world_size", width)),
        scalars=(("#0", str(group.unique_name)),
                 ("#5", int(axis))),
        # A fresh allocation. The output is the caller's `out` argument, sized
        # for the whole group, and it cannot be the input's storage at any
        # width above one -- but it is stamped rather than left to a default,
        # because an absent `output_aliases` and a recorded "allocates" read
        # the same in the memory walk and are not the same claim. `reg_buffer`
        # is the persistent IPC pool the transfer passes through; it belongs to
        # the process, is the same buffer for every step, and is not this
        # operator's activation.
        output_aliases=(None,),
        abi="live-state",
    )


class record_collectives:
    """Record collectives that simulated TP performs no operation for.

    At a physical world size of one there is no communicator, so simulated TP
    replaces ``all_reduce`` with a passthrough. That is right for its own
    purpose — benchmarking kernels, where a collective over absent ranks is
    meaningless anyway — and wrong for derivation, where the collective is a
    large part of what is being modelled. A TP graph derived through the
    passthrough contains no communication at all, and would cost out as though
    tensor parallelism were free.

    So the collective is recorded rather than performed: the graph gets the
    operator, its shapes, and the group it ran on, which is what a cost model
    needs. Nothing is sent, and no peer has to exist.

    The recorded name matches what the dispatcher records on real hardware
    (``aiter::all_reduce_``), so a derived graph and a captured one can be
    compared operator for operator rather than merely in spirit.

    ``all_gather`` needs the same treatment for a different reason. Simulated TP
    does not pass it through -- it cannot, because an all-gather grows its
    output -- so it *reimplements* it locally: allocate the full buffer, copy
    this rank's shard in, movedim, reshape. Those are real operators and the
    tracer records them, so the graph looks covered. It is not: what it holds is
    the cost of a local copy where production has a cross-rank transfer, and the
    LM head at TP2 or TP4 would come out near free. The local reimplementation
    is therefore replaced rather than recorded, and the collective recorded in
    its place.

    Which collective, and on which branch, was settled by running one -- two
    ranks on real devices through the real ``GroupCoordinator``, no model, no
    replayed handles (``agent_scratch/g4/ag_probe.py``). Four things came out of
    it, and the first is the one that matters most:

    *The dispatcher sees no all-gather.* ``outplace_all_gather``'s
    ``@torch.library.custom_op`` decorator is commented out in the installed
    aiter, and the kernel below it is a plain pybind function, so a
    ``TorchDispatchMode`` wrapped around a real two-rank call records exactly
    ``view.dtype``, ``detach``, ``empty``, ``view.dtype``, ``detach`` -- the
    output's allocation and nothing else. A *captured* TP2 graph therefore has
    no head collective in it either. This is not a defect of derivation that
    capture would fix; it is the same hole in both, and recording the operator
    here is what closes it. The name is consequently chosen rather than
    observed, and no captured graph will ever contain it.

    *The tuple is confirmed.* ``all_gather_unreg`` was handed
    ``(_fa=293897648, inp[1,124160]bf16, reg_buffer=139848155398144,
    out[1,248320]bf16, reg_bytes=1073741824, dim=1)`` -- the declaration's
    order exactly, with ``dim`` the resolved positive axis.

    *The transfer is real.* Each rank filled its shard with its own rank+1 and
    the gathered row read ``[1.0, 2.0]``, so the output is peers' data and not
    a right-shaped buffer of zeros.

    *The head is eager at every width that has a collective.*
    ``ModelRunner.logits_in_graph`` is ``world_size == 1 and not is_tbo``, so
    the head is only ever inside a CUDA graph at TP1 -- where ``all_gather``
    returns early and there is no collective at all. At TP>1 it is always the
    eager branch, which is ``all_gather_unreg``. The capture branch
    (``all_gather_reg``, four arguments) is unreachable for this call.

    So ``abi`` is ``"live-state"`` rather than ``"unverified"``: the tuple is
    known, and knowing it is what shows the call cannot be rebuilt from a graph.
    Two of those six arguments are a live communicator handle and a live IPC
    pool address. They belong to the process, not to the step, and no artifact
    carries them. The microbench refuses to reconstruct it and says so; a price
    for it has to come from performing the real collective in a real group.
    """

    def __init__(self, graph, group_name: str = "tp", tracer=None) -> None:
        self.graph = graph
        self.group_name = group_name
        #: The dispatch tracer recording into the same graph, where there is
        #: one. A synthesized operator needs its bookkeeping as much as a
        #: dispatched one does -- see `MetaOpTracer.note_operator` -- and only
        #: the tracer holds the producer map that supplies it. Optional so a
        #: test can record into a bare graph and read the spec back.
        self.tracer = tracer
        self._group = None
        self._original = None
        self._original_gather = None

    def _record(self, spec, inputs=(), outputs=()) -> None:
        if self.tracer is not None:
            self.tracer.note_operator(spec, inputs=inputs, outputs=outputs)
        else:
            self.graph.add(spec)

    def __enter__(self) -> "record_collectives":
        try:
            from aiter.dist.parallel_state import get_tp_group
        except ImportError:  # pragma: no cover - aiter absent
            return self
        group = get_tp_group()
        if getattr(group, "simulated_tp_physical_world_size", None) is None:
            # Not simulated: the real collective will dispatch and be recorded
            # by the op tracer, and wrapping it here would double-count.
            return self

        from atom.compass.core.graph import OpSpec
        from atom.compass.runtime.meta import fresh_like, fresh_shape

        import inspect

        record, name = self._record, self.group_name
        original = group.all_reduce
        # The dispatcher does not record the wrapper's arguments, it
        # records the custom op's: `all_reduce_(tensor, group_name,
        # ca_use_new, ca_fp8_quant, prefill_support=False)`. A derived
        # collective that omits them gets the signature
        # `aiter::all_reduce_|4,5120|bfloat16`, and the captured one is
        # `...|#1=tp:0;#2=True;#3=False` -- so an exact price lookup
        # misses every all-reduce in the graph, and tensor-parallel
        # communication silently costs nothing. Bind against the real
        # method so an omitted argument gets its real default rather
        # than a guess.
        # Against the *class*, not `original`: at a physical world size of one
        # the instance attribute is already simulated TP's passthrough, whose
        # signature is `(input_, *args, **kwargs)` and carries none of the
        # defaults this needs.
        sig = inspect.signature(type(group).all_reduce)

        def all_reduce(input_, *args, **kwargs):
            bound = sig.bind(group, input_, *args, **kwargs)
            bound.apply_defaults()
            flags = bound.arguments
            spec = OpSpec(
                name="aiter::all_reduce_",
                input_shapes=(tuple(int(d) for d in input_.shape),),
                output_shapes=(tuple(int(d) for d in input_.shape),),
                dtypes=(str(input_.dtype).replace("torch.", ""),),
                # A sum over ranks, in the dtype it reduces. Every native path
                # allocates with `empty_like`/`zeros_like`/`clone`, all of
                # which take the input's dtype.
                output_dtypes=(str(input_.dtype).replace("torch.", ""),),
                group=name,
                scalars=(("#1", str(group.unique_name)),
                         ("#2", bool(flags["ca_use_new"])),
                         ("#3", bool(flags["ca_fp8_quant"]))),
                # A fresh allocation, read off the native implementation rather
                # than off the name: the trailing underscore is a naming
                # artifact and every path this call can take is out of place.
                # `all_reduce_` defers to `_all_reduce_out_place`, whose fake
                # is `torch.empty_like`; `CustomAllreduce.all_reduce` and
                # `quick_all_reduce` each open with `out = torch.empty_like`;
                # pynccl allocates `out_tensor` likewise; the torch.distributed
                # fallback is `input_.clone()`; and the capture warm-up branch
                # returns `torch.zeros_like`. `registered_input` decides
                # whether the *input* is copied into the persistent IPC pool,
                # not where the output lives, so there is no path on which this
                # writes into input 0.
                output_aliases=(None,),
            )
            out = original(input_, *args, **kwargs)
            if out is input_ and input_.device.type == "meta":
                # Simulated TP's passthrough returned the input, because at a
                # physical world size of one that is what
                # `GroupCoordinator.all_reduce` does. Production does not: every
                # path with a peer allocates its own output. Handed back as it
                # came, the collective has no tensor of its own -- nothing is
                # watched, no death is recorded, and the input's death at the
                # call site is hidden behind it. `linear.py` does
                # `y = tensor_model_parallel_all_reduce(y)`, which drops the
                # last reference to the matmul's output right there; with one
                # storage for two tensors that release is invisible and the
                # buffer reads as immortal. 128 of them at TP2.
                #
                # Meta only. On a device, a physical-world-size-one microbench
                # under simulated TP would then really allocate, which is a cost
                # it does not pay today; there the passthrough stands and the
                # aliasing above is what the graph records.
                out = fresh_like(input_)
            record(spec, inputs=(input_,), outputs=(out,))
            return out

        group.all_reduce = all_reduce
        self._group, self._original = group, original

        original_gather = group.all_gather

        def all_gather(input_, use_custom: bool = False, dim: int = -1):
            spec = head_gather_opspec(group, input_, dim, group_name=name)
            out = fresh_shape(spec.output_shapes[0], input_)
            record(spec, inputs=(input_,), outputs=(out,))
            return out

        group.all_gather = all_gather
        self._original_gather = original_gather
        return self

    def __exit__(self, *exc) -> None:
        if self._group is not None and self._original is not None:
            self._group.all_reduce = self._original
            if self._original_gather is not None:
                self._group.all_gather = self._original_gather
            self._group = self._original = self._original_gather = None


class redirect_device_factories:
    """Send a library's device-typed *factory* calls to meta, and count them.

    AITER dispatches some custom operators by handing them a dummy tensor whose
    device selects the kernel::

        # aiter/jit/utils/torch_guard.py
        getattr(torch.ops.aiter, loadName)(torch.empty(1, device=device), ...)

    ``device`` there is the decorator's, a literal ``"cuda"`` for kernels that
    only exist on a card. An explicit ``device=`` beats the ambient
    ``torch.device("meta")`` a derivation builds under, so that one-element
    tensor is the single thing in the whole forward that still demands a GPU --
    and on a machine without one the trace stops before the operator it was
    about to name.

    So the *factory* is redirected, and only the factory: an allowlist of
    tensor-creating functions, and only when the caller asked for cuda. The
    operator then dispatches on meta, where the tracer records it, which is the
    entire purpose of the pass. Nothing in the installed library is patched.

    What this deliberately does not cover is movement: ``.to("cuda")``,
    ``.cuda()``, ``Tensor.copy_`` to a card. Those are a derivation reaching for
    real memory rather than selecting a dispatch key, and they still fail --
    which on a device-free machine they would have done anyway.

    ``redirected`` is reported by the caller rather than kept quiet: a graph
    derived through this seam should say so.
    """

    def __init__(self) -> None:
        self.redirected = 0
        self._mode = None

    def __enter__(self) -> "redirect_device_factories":
        import torch
        from torch.overrides import TorchFunctionMode

        factories = {
            torch.empty, torch.zeros, torch.ones, torch.full, torch.tensor,
            torch.arange, torch.rand, torch.randn, torch.eye, torch.linspace,
            torch.empty_like, torch.zeros_like, torch.ones_like,
            torch.full_like,
        }
        outer = self

        class _Mode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = {} if kwargs is None else kwargs
                device = kwargs.get("device")
                if device is not None and func in factories:
                    if torch.device(device).type == "cuda":
                        kwargs = dict(kwargs, device="meta")
                        outer.redirected += 1
                return func(*args, **kwargs)

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._mode is not None:
            self._mode.__exit__(*exc)
            self._mode = None
