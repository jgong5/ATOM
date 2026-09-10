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
    """

    def __init__(self, graph, group_name: str = "tp") -> None:
        self.graph = graph
        self.group_name = group_name
        self._group = None
        self._original = None

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

        import inspect

        recorder, name = self.graph, self.group_name
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
            recorder.add(
                OpSpec(
                    name="aiter::all_reduce_",
                    input_shapes=(tuple(int(d) for d in input_.shape),),
                    output_shapes=(tuple(int(d) for d in input_.shape),),
                    dtypes=(str(input_.dtype).replace("torch.", ""),),
                    group=name,
                    scalars=(("#1", str(group.unique_name)),
                             ("#2", bool(flags["ca_use_new"])),
                             ("#3", bool(flags["ca_fp8_quant"]))),
                )
            )
            return original(input_, *args, **kwargs)

        group.all_reduce = all_reduce
        self._group, self._original = group, original
        return self

    def __exit__(self, *exc) -> None:
        if self._group is not None and self._original is not None:
            self._group.all_reduce = self._original
            self._group = self._original = None


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
