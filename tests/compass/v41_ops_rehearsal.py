# SPDX-License-Identifier: MIT
"""`compass measure ops` rehearsed off the card, each leaf up to its launch.

    python v41_ops_rehearsal.py OUT [--kinds] <measure ops args> -- <ATOM args>

`measure ops`' own arguments but `--gpu` and `--out`. The process declares
card 0 of the MI455 A0 node as it read (`test_capture_mi455_cells.A0`) and
ATOM's TP group in itself (`driverless.group`), then runs `ops.measure` on
the graphs of the store, one rank, in a capture's fake mode
(`driverless_mode`, the capture's `_patches`): ATOM's runner is built as the
bench builds it, its KV pool sized from that card's memory, all free past the
parameters' bytes, and each point's batch prepared by ATOM; each leaf's call is
built by `Stub.point` and its operator found, and there it stops: the call
would launch next. With `--kinds` a graph keeps its first leaf of each
operator, as a card run of one leaf an op kind takes them.

The fake mode stands in for the device where the bench reads it: the
runner's checkpoint is ATOM's empty dummy, as a capture's is; a router's
expert ids are drawn on the host and their loads not counted; no tensor
equals another, so each placement clones its metadata; DeepSeek-V4.1's state
cursors read back zeros and its check of them has no values to judge, as in
a capture. OUT gets one record a leaf: its point, its verdict (`LAUNCH`,
`REFUSED` with the bench's reason, or `UNTOLD`, which a fake tensor's
missing values stopped) and the buffers its KV and metadata arguments bound.
"""

import contextlib
import json
import pathlib
import sys
import tempfile
from unittest.mock import patch

LAUNCH, REFUSED, UNTOLD = "reaches launch", "refused", "cannot be told off the card"
#: What a fake tensor raises where a value would be read: the device's to say.
VALUELESS = (
    "DataDependentOutputException",
    "DynamicOutputShapeException",
    "GuardOnDataDependentSymNode",
)


class Launched(Exception):
    """The leaf's call is built and its operator found: it would launch."""


def first_of_each(found):
    """`found` with its first leaf of each operator."""
    kept, seen = {}, set()
    for sig, (op, instances) in found.items():
        if op["op"] not in seen:
            seen.add(op["op"])
            kept[sig] = op, instances
    return kept


def verdict(record) -> tuple[str, str]:
    why = record.get("refused", "")
    if why.startswith(f"{Launched.__name__}:"):
        return LAUNCH, ""
    return (UNTOLD if why.split(":")[0] in VALUELESS else REFUSED), why


def main(out, argv) -> None:
    here = pathlib.Path(__file__).parent
    sys.path[:0] = [str(here), str(here.parent)]  # its flat imports
    from test_capture_mi455_cells import A0

    from atom.compass.capture import driverless, forward

    driverless.install(tempfile.mkdtemp(), A0["arch"], A0)

    import torch
    from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from atom.compass.artifacts import ArtifactStore
    from atom.compass.bench import ops, ranks
    from atom.compass.bench.measure import atom_config
    from atom.compass.capture.forward import decode, prefill
    from atom.config import set_current_atom_config
    from atom.model_engine.model_runner import DeviceMemoryReadings, ModelRunner
    from atom.utils import h2d

    kinds = "--kinds" in argv
    argv = [a for a in argv if a != "--kinds"]
    cut = argv.index("--")
    args = ops._parser().parse_args([*argv[:cut], "--gpu", "0", "--out", out])
    shapes = [decode(w) for w in args.width] + [prefill(*s) for s in args.prefill]
    config = atom_config([*argv[cut + 1 :], "--load_dummy", "empty"])
    ranks.layout(config)
    store, keep = ArtifactStore(args.store), first_of_each if kinds else dict
    todo = [
        (s, args.context, keep(ops.leaves(ops.graph(store, config, s)["region"])))
        for s in shapes
    ]
    pc = config.parallel_config
    driverless.group(config.tensor_parallel_size, pc.data_parallel_size)
    set_current_atom_config(config)
    fake = driverless.driverless_mode(ShapeEnv())
    contexts = config.compilation_config.static_forward_context
    recorder = forward._Recorder(contexts, fake)
    recorder.lands = True
    driverless.on_wait = recorder.wait
    import torch._inductor.decomposition  # noqa: F401  as a capture imports it

    function, draw, bincount = ops._function, ops.uniform_ids, torch.bincount

    def launching(name):
        function(name)  # found, or the leaf refuses

        def launch(**kwargs):
            raise Launched(name)

        return launch

    def on_device(self, rank, config):
        self.device = torch.device("cuda:0")

    def readings(self):
        total = torch.cuda.get_device_properties(0).total_memory
        held = sum(p.nbytes for p in self.model.parameters())
        return DeviceMemoryReadings(total, total, held, 0)

    def host_draw(rows, experts, k, seed, device):
        return draw(rows, experts, k, seed, "cpu").to(device)

    def uncounted(x, weights=None, minlength=0):
        if not isinstance(x, FakeTensor):
            return bincount(x, weights, minlength)
        with unset_fake_temporarily():
            return torch.zeros(minlength, dtype=torch.int64)

    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache

    with contextlib.ExitStack() as stack:
        stack.enter_context(forward._patches(recorder))
        for owner, name, value in (
            (ModelRunner, "_setup_device_and_distributed", on_device),
            (ModelRunner, "_read_device_memory", readings),
            (ops, "_function", launching),
            (ops, "uniform_ids", host_draw),
            (ranks, "cpu_group", lambda: None),
            (torch, "bincount", uncounted),
            (torch, "equal", lambda a, b: False),
            (h2d, "_range", lambda t: (0, 0)),
            (PagedAttentionCache, "_report_stale_state", lambda *a: None),
        ):
            stack.enter_context(patch.object(owner, name, value))
        stack.enter_context(fake)
        stack.enter_context(torch.inference_mode())
        got = ops.measure(config, todo, args.llc_mib << 20)
    records = []
    for r in got:
        state, why = verdict(r)
        records.append(
            {k: r[k] for k in ("shape", "op", "leaf")}
            | {"verdict": state, "why": why, "bound": r.get("metadata", {})}
        )
    pathlib.Path(out).write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
