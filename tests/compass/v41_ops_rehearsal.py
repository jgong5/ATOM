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
built by `Stub.point`, each index, slot, pointer and count it binds checked
against what it indexes on the values ATOM staged (`out_of_range`), and its
operator called in the fake mode up to its kernels: a copy with pageable host
memory, which no graph capture copies, refuses, and each Triton launch types
and keys its arguments as Triton does, a tensor as an aligned pointer, and
launches nothing. A FlyDSL launch and a compiled operator's are not typed.
With `--kinds` a graph keeps its first leaf of each operator, as a card run of
one leaf an op kind takes them.

The fake mode stands in for the device where the bench reads it: the
runner's checkpoint is ATOM's empty dummy, as a capture's is; a router's
expert ids are drawn on the host and their loads not counted; no tensor
equals another, so each placement clones its metadata; DeepSeek-V4.1's state
cursors read back zeros and its check of them has no values to judge, as in
a capture. OUT gets one record a leaf: its point, its verdict (`LAUNCH` with
the kernels it launched, `REFUSED` with the reason, or `UNTOLD`, which a fake
tensor's missing values stopped), the buffers its KV and metadata arguments
bound, and each one's identity (`identity`).
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


#: The integer dtypes of a leaf's indices, slots, pointers and counts.
INTEGERS = ("torch.int32", "torch.int64")
#: aten's index operators, by the slot of their indices: the dimension of
#: `self` each indexes (`dim`, or the index's own position in the list), and
#: the least value it may take, as a share of that extent.
ATEN_INDEX = {
    ("aten.index_select.default", "index"): ("dim", 0),
    ("aten.index_fill_.int_Scalar", "index"): ("dim", 0),
    ("aten.gather.default", "index"): ("dim", 0),
    ("aten.index.Tensor", "indices"): (None, -1),
    ("aten.index_put_.default", "indices"): (None, -1),
}


class Launched(Exception):
    """The leaf's call is built and its operator found: it would launch."""


#: A step's tables its forward's first reader fills and a later reader takes
#: from it, views of buffers the cache and workspace hold, which a step the
#: bench prepares holds none of: a tensor is named by those buffers.
PER_FORWARD = (
    "group_indices",
    "indptrs",
    "tiles",
    "selected",
    "candidates",
    "candidate_blocks",
    "window_rows",
)


def named(runner) -> list:
    """Each device tensor a V4.1 step can pass a leaf, by a name the capture's
    runner and the bench's both give it: the forward variables, the forward
    context's and its metadata's own, and what the builder's paged cache holds,
    its step, its decoder plan (`decoder`, the forward's `replay`) and its
    scorer's workspace among them, but its forward's tables (`PER_FORWARD`)."""
    import torch

    from atom.compass.bench import ops
    from atom.utils.forward_context import get_forward_context

    fc, device = get_forward_context(), runner.device
    out = [(f"fv.{k}", getattr(v, "gpu", v)) for k, v in runner.forward_vars.items()]
    for prefix, owner in (("md", fc.attn_metadata), ("ctx", fc.context)):
        out += [(f"{prefix}.{k}", v) for k, v in getattr(owner, "__dict__", {}).items()]
    cache = getattr(runner.attn_metadata_builder, "cache", None)
    step = getattr(cache, "current_step", None)
    plan = getattr(step, "decoder", None)
    replay = ".current_step.replay." if getattr(step, "replay", None) is plan else None
    for name, t in ops._bound("cache", cache, set(), device).items():
        if not any(f".{table}" in name for table in PER_FORWARD):
            out.append(
                (name.replace(replay, ".current_step.decoder.") if replay else name, t)
            )
    return [
        (n, t, _region(t))
        for n, t in out
        if torch.is_tensor(t) and t.device.type == device.type
    ]


def _region(t):
    """`t`'s storage and the bytes it spans there."""
    from torch.multiprocessing.reductions import StorageWeakRef

    from atom.compass.bench import ops

    start = t.storage_offset() * t.element_size()
    end = start + ops._span(t.shape, t.stride()) * t.element_size()
    return StorageWeakRef(t.untyped_storage()), start, end


def identity(t, pool) -> str:
    """The smallest of `pool`'s tensors that holds `t`'s bytes, an exact view
    of them first, by name and byte offset; `?` when none does: a host tensor
    or one the step made and holds nowhere."""
    storage, lo, hi = _region(t)
    best = None
    for name, c, (where, clo, chi) in pool:
        if where == storage and clo <= lo and hi <= chi:
            exact = (c.shape, c.stride(), c.dtype, clo) == (
                t.shape,
                t.stride(),
                t.dtype,
                lo,
            )
            key = (not exact, chi - clo, name)
            best = min(best or (key, name, lo - clo), (key, name, lo - clo))
    return f"{best[1]}+{best[2]}" if best else "?"


def staged(runner) -> dict:
    """The host side of each buffer ATOM stages for the step (`CpuGpuBuffer`),
    by its device side's storage: what the device side holds once published."""
    from torch.multiprocessing.reductions import StorageWeakRef

    from atom.utils import CpuGpuBuffer

    cache = getattr(runner.attn_metadata_builder, "cache", None)
    held = [*runner.forward_vars.values(), *getattr(cache, "__dict__", {}).values()]
    return {
        StorageWeakRef(b.gpu.untyped_storage()): b.cpu
        for b in held
        if isinstance(b, CpuGpuBuffer)
    }


def values(t, hosts):
    """The values ATOM staged for the device tensor `t`, or None."""
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.multiprocessing.reductions import StorageWeakRef

    host = hosts.get(StorageWeakRef(t.untyped_storage()))
    if host is None:
        return None
    with unset_fake_temporarily():
        return host.as_strided(t.shape, t.stride(), t.storage_offset()).tolist()


def out_of_range(op, held, known) -> str | None:
    """Why an index, slot, pointer or count `op` takes, of the values `held`
    by slot path, reaches past what it indexes, or None: aten's index
    operators against `self`'s dimension (`ATEN_INDEX`), a gather's rows
    against its tensors' token axis, a state slot against the slots and a
    block table against the blocks `known` states, a position against the
    model's length, a row bound against its logits' width, a count against
    the rows the leaf runs, and a pointer table never falling."""
    from atom.compass.bench import ops

    args, specs = op["args"], dict(ops._specs(op["args"]))
    first = next((s for s in specs.values() if s["role"] == "activation"), None)
    logits = specs.get("logits", specs.get("out_logits"))
    for path, got in held.items():
        flat = [v for v in _flat(got)]
        if not flat:
            continue
        slot, lo, hi = path.split(".")[0], min(flat), max(flat)
        index = ATEN_INDEX.get((op["op"], slot))
        if index is not None:
            dim = args[index[0]] if index[0] else int(path.split(".")[1])
            size = specs["self"]["shape"][dim]
            bounds = index[1] * size, size - 1
        elif slot == "rows" and "tensors" in args:
            bounds = 0, specs["tensors.0"]["shape"][args["dim"]] - 1
        elif "slot" in slot:
            bounds = -1, known["slots"] - 1
        elif "block_table" in slot:
            bounds = -1, known["blocks"] - 1
        elif slot == "positions":
            bounds = 0, known["max_model_len"] - 1
        elif slot in ("context_lens", "row_starts", "row_ends", "visible") and logits:
            bounds = 0, logits["shape"][-1]
        elif slot.startswith("num_") and first is not None:
            bounds = 0, first["shape"][0]
        elif "ptr" in slot or slot.startswith("cu_seqlens"):
            if any(b < a for a, b in zip(flat, flat[1:])) or lo < 0:
                return f"{path} falls or is negative: {flat[:8]}"
            continue
        else:
            continue
        if lo < bounds[0] or hi > bounds[1]:
            return f"{path} holds {lo} to {hi}, past {bounds[0]} to {bounds[1]}"
    return None


class _Pointer:
    """A tensor as Triton's argument typing reads one: its dtype and an
    aligned address."""

    def __init__(self, dtype):
        self.dtype = dtype

    def data_ptr(self):
        return 0


def _pointer(value):
    import torch

    if torch.is_tensor(value):
        return _Pointer(value.dtype)
    return tuple(map(_pointer, value)) if isinstance(value, tuple) else value


def _pageable(value) -> bool:
    """Whether `value` is a host tensor in pageable memory, which no copy in a
    graph capture reads or writes."""
    import torch

    if not torch.is_tensor(value) or value.device.type != "cpu":
        return False
    try:
        return not value.is_pinned()
    except RuntimeError:  # a fake host tensor holds no memory to ask
        return True


def _flat(value):
    if isinstance(value, list):
        for v in value:
            yield from _flat(v)
    else:
        yield value


def first_of_each(found):
    """`found` with its first leaf of each operator."""
    kept, seen = {}, set()
    for sig, (op, instances) in found.items():
        if op["op"] not in seen:
            seen.add(op["op"])
            kept[sig] = op, instances
    return kept


def verdict(record) -> tuple[str, str]:
    """Its verdict and why; the kernels its call launched when it reaches its
    launch."""
    why = record.get("refused", "")
    if why.startswith(f"{Launched.__name__}:"):
        return LAUNCH, why.split(": ", 1)[1]
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
    from triton.runtime import jit as triton_jit

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
    deep, prepare, time_ = ops._deep, ops.Stub._prepare, ops.Stub._time
    journal = ops._journal
    # by the id of each metadata view's clone, the live view it cloned; by
    # leaf and slot, the identity and staged values of what placement 0 bound
    sources, bound, seen = {}, {}, {}

    def cloning(x, memo, clones=None):
        out = deep(x, memo, clones)
        if torch.is_tensor(x):
            sources[id(out)] = out, x
        return out

    def preparing(stub, batch):
        prepare(stub, batch)
        if len(stub.placed) > 1:
            return
        bound.clear()
        pool, hosts = named(stub.runner), staged(stub.runner)
        for (sig, path), (_, view) in stub.placed[0][1].items():
            if view is None:
                continue
            live = sources.get(id(view), (None, view))[1]
            spec = dict(ops._specs(stub.leaves[sig][0]["args"]))[path]
            if spec["role"] == "metadata" or str(live.dtype) in INTEGERS:
                bound[sig, path] = identity(live, pool), values(live, hosts)

    def timing(stub, op, sig, instances, n):
        held = {p: v for (s, p), (_, v) in bound.items() if s == sig and v}
        seen[sig] = {p: i for (s, p), (i, _) in bound.items() if s == sig}
        known = {
            "slots": stub.slots,
            "blocks": stub.runner.config.num_kvcache_blocks,
            "max_model_len": stub.runner.config.max_model_len,
        }
        if why := out_of_range(op, held, known):
            raise IndexError(why)
        return time_(stub, op, sig, instances, n)

    def journaled(path, records):
        for r in records:
            r["identity"] = seen.get(r["leaf"], {})
        return journal(path, records)

    def launching(name):
        fn = function(name)  # found, or the leaf refuses

        def launch(**kwargs):
            pageable = kwargs.get("device") in ("cpu", torch.device("cpu"))
            pageable &= not kwargs.get("pin_memory")
            if pageable or any(_pageable(v) for v in kwargs.values()):
                raise RuntimeError(
                    "a copy with pageable host memory in a graph capture"
                )
            recorder.replaying = []  # each kernel told the recorder, not launched
            try:
                fn(**kwargs)
            finally:
                kernels, recorder.replaying = recorder.replaying, None
            raise Launched(json.dumps(kernels))

        return launch

    def typed(kernel, *args, grid=None, warmup=False, **kwargs):
        # Triton's run up to its compile: each argument typed and the kernel
        # keyed as Triton types and keys them, a tensor as an aligned pointer
        caches = kernel.device_caches[0]
        named = {k: _pointer(v) for k, v in kwargs.items()}
        _, specialization, options = caches[4](*map(_pointer, args), **named)
        triton_jit.compute_cache_key(caches[1], specialization, options)
        recorder.launch(f"triton.{kernel.__name__}", kernel)

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
            (triton_jit.JITFunction, "run", typed),
            (ops, "_deep", cloning),
            (ops, "_journal", journaled),
            (ops.Stub, "_prepare", preparing),
            (ops.Stub, "_time", timing),
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
        kernels = json.loads(why) if state == LAUNCH else []
        why = "" if state == LAUNCH else why
        records.append(
            {k: r[k] for k in ("shape", "op", "leaf")}
            | {"verdict": state, "why": why, "kernels": kernels}
            | {"bound": r.get("metadata", {})}
            | {"identity": r.get("identity", {})}
        )
    pathlib.Path(out).write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
