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

__all__ = ["simulate_group_width", "record_collectives"]


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

        recorder, name = self.graph, self.group_name
        original = group.all_reduce

        def all_reduce(input_, *args, **kwargs):
            recorder.add(
                OpSpec(
                    name="aiter::all_reduce_",
                    input_shapes=(tuple(int(d) for d in input_.shape),),
                    output_shapes=(tuple(int(d) for d in input_.shape),),
                    dtypes=(str(input_.dtype).replace("torch.", ""),),
                    group=name,
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
