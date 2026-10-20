# SPDX-License-Identifier: MIT
"""Steps of a model, traced with no device into `ir.Graph`s, one per `Shape`.

ATOM's `ModelRunner` runs its own step through the model's own classes under
`TorchDispatchMode` over `FakeTensorMode`, the Python dispatcher on, at one
concrete shape: a decode width, or a prefill of T tokens over B sequences. One
runner steps through every shape of a capture, as it steps through a run.
A decode shape is traced twice, its rows holding each of `CONTEXTS` tokens: an
integer argument that differs between the two is affine in the context
(`ir.Affine`), or the shape is refused by operator and slot.

Each leaf carries its contract: its arguments by schema slot, a tensor as its
shape, dtype, stride and role. A tensor's role is where its storage came from:
the model's parameters, buffers and tensor attributes are weights, what
`allocate_kv_cache` allocated is KV, what the model's forward and the sampler
after it allocated is an activation, and anything else (the runner's buffers,
what its input preparation allocated) is metadata.

A raw Triton or FlyDSL launch lifts to its innermost ATOM function: one leaf,
that function by its module and name, its parameters as the arguments, with
what else the function dispatched folded in. A function that reassigned a
parameter before the launch, or that dispatched a leaf of its own other than a
host query (`HOST_QUERIES`), is refused by name.

The engine is ATOM's own `Config` for ATOM's engine arguments, on a declared
device of one arch. Under DP the capture runs one rank and every peer runs its
shape (`peers`).
"""

import ast
import contextlib
import dataclasses
import inspect
import pathlib
import re
import sys
import textwrap
from typing import NamedTuple
from unittest.mock import patch

import numpy
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode

import atom
from atom.compass import ir

from . import driverless, peers
from .graph import LAYER, Fork, Join, Traced, _json, build

#: The fewest KV blocks a capture allocates; a shape that needs more gets more.
KV_BLOCKS = 64
#: The two contexts a decode shape is traced at, each row holding that many
#: tokens. An integer argument that differs between them is affine in the
#: context or refused; 997 apart, a prime, so a count rounded to blocks or
#: divided moves by less than the context and is refused.
CONTEXTS = (1, 998)
#: Torch's own namespaces; an operator in any other hides its kernels.
TORCH_NAMESPACES = ("aten", "prim", "prims", "_c10d_functional", "c10d")
#: ATOM's H2D publication copies the runner's host-staged inputs to the device:
#: input preparation, not the step's device work, so its copies are left out.
H2D = "atom/utils/h2d.py"
#: Host queries a lifted function may dispatch: they launch nothing.
HOST_QUERIES = ("aiter.get_gfx_custom_op.default",)
#: Operators that ask a fake tensor where it is and run nothing on a device.
UNDISPATCHED = ("profiler.", "prim.device")
#: The hf config fields, or its text config's, that count a model's routed experts.
ROUTED = ("n_routed_experts", "num_experts", "num_local_experts")
#: DeepSeek-V4's persistent decode kernel and its indexer's score and top-k.
V4_PS_DECODE = "aiter.mla_decode_v4_ps_asm.default"
V4_INDEXER = "aiter.indexer_score_topk.default"

_ROOT = pathlib.Path(atom.__file__).resolve().parent.parent
_IS_ATOM: dict = {}


class Shape(NamedTuple):
    """A step's shape: `tokens` query tokens over `seqs` sequences."""

    phase: str  # "decode" or "prefill"
    tokens: int
    seqs: int

    def __str__(self) -> str:
        if self.phase == "prefill":
            return f"prefill T={self.tokens} B={self.seqs}"
        return f"decode B={self.seqs}"


def decode(width: int) -> Shape:
    if width < 1:
        raise ValueError(f"a decode width is at least 1, got {width}")
    return Shape("decode", width, width)


def prefill(tokens: int, seqs: int) -> Shape:
    if not 1 <= seqs <= tokens:
        raise ValueError(
            f"a prefill of {tokens} tokens over {seqs} sequences: each sequence "
            "takes at least one token"
        )
    return Shape("prefill", tokens, seqs)


def device_arch() -> str:
    """The visible device's arch, its target features dropped."""
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def structure(
    config,
    phase: str,
    eager: bool | None = None,
    device: str | None = None,
    output: bool = True,
) -> str:
    """The structure an `op_graph` of `config` at a `phase` step is keyed by: the
    architecture, the config.json digest, the TP width, the DP width and the EP
    group's when they are set, the phase and the mode, `config`'s unless `eager`
    names it, the arch of `device`, else `device_arch()`, the routed experts'
    Triton path when it is taken (`experts`), and ``no output`` unless the step
    produces output (`output`)."""
    from atom.compass.run import _file_digest

    digest = _file_digest(config.model, "config.json")
    mode = "eager" if (config.enforce_eager if eager is None else eager) else "graph"
    arch, device = config.hf_config.architectures[0], device or device_arch()
    tp, dp = config.tensor_parallel_size, config.parallel_config.data_parallel_size
    widths = f"tp={tp}" + (f" dp={dp}" if dp > 1 else "")
    widths += f" ep={tp * dp}" if config.enable_expert_parallel else ""
    key = f"{arch} {device} {digest} {widths}{experts(config, device)} {phase} {mode}"
    return key if output else f"{key} no output"


def experts(config, device: str) -> str:
    """`` experts=triton`` when the MXFP4 routed experts run Triton's kernels on
    a `device` arch, from what `Mxfp4MoEMethod` reads to pick them, then
    ``+a4w4`` and ``+decode`` (only decode steps run them) as set. Empty for
    aiter's path and for a model with no routed experts."""
    from atom.utils import envs

    tp, dp = config.tensor_parallel_size, config.parallel_config.data_parallel_size
    hf = (config.hf_config, getattr(config.hf_config, "text_config", None))
    if not any(getattr(c, n, None) for c in hf for n in ROUTED):
        return ""
    ep = config.enable_expert_parallel and tp * dp > 1
    arches = ("gfx94", "gfx95") if envs.ATOM_USE_TRITON_GEMM else ("gfx94",)
    on = not ep and device.startswith(arches)
    if envs.is_set("ATOM_USE_TRITON_MOE"):
        on = envs.ATOM_USE_TRITON_MOE
    # MegaMoE's method keeps Triton's EP path alone, and no phase split.
    mega = getattr(config, "moe_backend", "") == "mega"
    on = on and not getattr(config, "eplb_enable", False) and not (mega and not ep)
    if not on or ep and not device.startswith(("gfx95", "gfx125")):
        return ""
    a4w4, decode = envs.ATOM_USE_TRITON_MOE_A4W4, envs.ATOM_USE_TRITON_MOE_DECODE
    return " experts=triton" + "+a4w4" * a4w4 + "+decode" * (decode and not mega)


def _check_experts(model, named: str) -> None:
    """Refuses a model whose MXFP4 experts take another path than `named`."""
    for name, module in model.named_modules():
        m = getattr(module, "quant_method", None)
        if hasattr(m, "use_triton_ep"):
            took = (m.use_triton or m.use_triton_ep, m.use_triton_decode)
            if took != ("triton" in named, "+decode" in named):
                raise ValueError(
                    f"ATOM's experts at {name} take Triton's, decode only: {took}; "
                    f"the op_graph key names {named.strip() or 'aiter'}"
                )


def _atom_frames():
    """The ATOM frames on the stack, innermost first, Compass's own excluded."""
    frames, frame = [], sys._getframe(1)
    while frame:
        name = frame.f_code.co_filename
        if name not in _IS_ATOM:
            path = pathlib.Path(name)
            _IS_ATOM[name] = path.is_relative_to(
                _ROOT / "atom"
            ) and not path.is_relative_to(_ROOT / "atom" / "compass")
        if _IS_ATOM[name]:
            frames.append(frame)
        frame = frame.f_back
    return frames


def _where(frame) -> str:
    path = pathlib.Path(frame.f_code.co_filename).relative_to(_ROOT)
    return f"{path}:{frame.f_lineno} in {frame.f_code.co_qualname}"


def _storage(tensor) -> StorageWeakRef:
    return StorageWeakRef(tensor.untyped_storage())


def _reassigned(frame, names) -> list[str]:
    """The parameters `frame`'s function stores to before its current line."""
    lines, first = inspect.getsourcelines(frame.f_code)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    return sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id in names
            and node.lineno + first - 1 < frame.f_lineno
        }
    )


class _Recorder(TorchDispatchMode):
    """Every dispatched operator and launched kernel, as `ir.Op`, by layer.

    What is not ATOM's work at the traced shape goes to `left_out`, with why.
    `creating` is the role a storage allocated now takes; `roles` holds every
    role given so far. `stated` is what an op of each kind recorded now
    states beside its arguments (`ir.Op.stated`), by its name.
    """

    def __init__(self, contexts, fake_mode=None):
        super().__init__()
        self.contexts, self.modules, self.roles = contexts, {}, {}
        self.creating, self.on, self.fake_mode = None, False, fake_mode
        self.stated = {}
        self.start()

    def start(self):
        self.layer, self.items, self.stacks, self.left_out = None, [], [], []
        self.lifting = None  # the frame the last item lifted to

    @contextlib.contextmanager
    def quiet(self):
        """Record nothing inside: a stand-in's own work is not ATOM's."""
        on, self.on = self.on, False
        try:
            yield
        finally:
            self.on = on

    def _fake(self, tensor):
        if not isinstance(tensor, FakeTensor):
            return self.fake_mode.from_tensor(tensor)
        tensor.constant = None
        return tensor

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        device = kwargs.get("device")
        if device is not None and torch.device(device).type != "cpu":
            # A host array headed for the device, real or a fake holding its
            # values, is faked without them first, or the fake mode would copy
            # the values onto a device that is not there.
            args = _pytree.tree_map_only(torch.Tensor, self._fake, args)
        out = func(*args, **kwargs)
        leaves = _pytree.tree_leaves((args, kwargs))
        ins = [t for t in leaves if isinstance(t, torch.Tensor)]
        outs = [t for t in _pytree.tree_leaves(out) if isinstance(t, torch.Tensor)]
        if self.creating is not None:
            read = set(map(_storage, ins))
            for storage in map(_storage, outs):
                if storage not in read:
                    self.roles[storage] = self.creating
        name = str(func)
        if self.on and not name.startswith(UNDISPATCHED):
            torch_own = name.split(".")[0] in TORCH_NAMESPACES
            kind = ir.NodeKind.CAPTURED if torch_own else ir.NodeKind.OPAQUE_LEAF
            stack = _atom_frames()
            why = "h2d" if stack and _where(stack[0]).startswith(H2D) else None
            if ins + outs and all(t.device.type == "cpu" for t in ins + outs):
                why = "host"  # host arithmetic on host tensors runs no kernel
            if driverless.standing_in[0] and not name.startswith("_c10d_functional."):
                why = "stand-in"
            slots = {a.name: v for a, v in zip(func._schema.arguments, args)}
            self.add(name, kind, {**slots, **kwargs}, outs, why, stack)
        return out

    def add(self, name, kind, slots, outs, why, stack):
        op = self._op(name, kind, slots, [tuple(t.shape) for t in outs])
        if why:
            self.left_out.append((why, op))
        elif self.lifting is not None and any(f is self.lifting for f in stack):
            self._fold(op, self.lifting)
        else:
            self.lifting = None
            self.items.append((self.layer, op))
            self.stacks.append(stack)

    def launch(self, name):
        """A raw Triton or FlyDSL launch, lifted to its innermost ATOM function."""
        if not self.on:
            return
        stack = _atom_frames()
        if not stack:
            raise ValueError(f"{name} launches from no ATOM function to lift it to")
        frame = stack[0]
        if frame is self.lifting:  # another kernel of the function being lifted
            return
        code = frame.f_code
        count = code.co_argcount + code.co_kwonlyargcount
        count += bool(code.co_flags & inspect.CO_VARARGS)
        count += bool(code.co_flags & inspect.CO_VARKEYWORDS)
        params = code.co_varnames[:count]
        if reassigned := _reassigned(frame, params):
            raise ValueError(
                f"{name} lifts to {_where(frame)}, which reassigns its parameters "
                f"{reassigned} before the launch: they are not what it was called with"
            )
        while (
            self.items
            and self.items[-1][0] == self.layer
            and any(f is frame for f in self.stacks[-1])
        ):
            self.stacks.pop()
            self._fold(self.items.pop()[1], frame)
        lifted = f"{frame.f_globals['__name__']}:{code.co_qualname}"
        slots = {p: frame.f_locals[p] for p in params}
        op = self._op(lifted, ir.NodeKind.OPAQUE_LEAF, slots, ())
        self.items.append((self.layer, op))
        self.stacks.append([])
        self.lifting = frame

    def _fold(self, op, frame):
        if op.kind is ir.NodeKind.OPAQUE_LEAF and op.name not in HOST_QUERIES:
            raise ValueError(
                f"a launch lifts to {_where(frame)}, which also dispatches the leaf "
                f"{op.name}: one leaf cannot hold another"
            )

    def wait(self, waiter, waited):
        """A side stream waiting on the main one forks; the reverse joins."""
        if not self.on:
            return
        if waited is driverless.main_stream and waiter is not waited:
            item = Fork(waiter.stream_id)
        elif waiter is driverless.main_stream and waited not in (None, waiter):
            item = Join(waited.stream_id)
        else:
            return
        self.items.append((self.layer, item))
        self.stacks.append([])
        self.lifting = None

    def _op(self, name, kind, slots, out_shapes):
        """`name` as an `Op`, or a refusal naming it and what it cannot hold."""
        try:
            args = tuple((slot, self._arg(value)) for slot, value in slots.items())
            ref = next((v for _, v in args if type(v) is ir.ContextRef), None)
            stream = driverless.current[0].stream_id
            stated = self.stated.get(name, ())
            return ir.Op(name, kind, args, out_shapes, stream, ref, stated)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} has no contract a graph can hold: {exc}"
            ) from None

    def _arg(self, value):
        if isinstance(value, torch.Tensor):
            role = self.roles.get(_storage(value), ir.Role.METADATA)
            shape, stride = tuple(value.shape), value.stride()
            return ir.Tensor(shape, str(value.dtype), stride, role)
        if isinstance(value, torch.nn.Module):
            if id(value) not in self.modules:
                raise TypeError(f"a {type(value).__name__} outside the model")
            return ir.ModuleRef(self._layered(self.modules[id(value)]))
        if isinstance(value, str) and value in self.contexts:
            return ir.ContextRef(self._layered(value))
        if isinstance(value, (list, tuple)):
            return tuple(self._arg(v) for v in value)
        if isinstance(value, (torch.dtype, torch.device, torch.layout)):
            return str(value)
        if isinstance(value, numpy.generic):
            return value.item()
        if isinstance(value, dict):
            return tuple((k, self._arg(v)) for k, v in value.items())
        if hasattr(value, "__dict__") and not callable(value):
            # A host object the engine built from its configuration, which the
            # graph's key names: its type stands for it.
            return f"<{type(value).__qualname__}>"
        return value

    def _layered(self, path):
        if self.layer is None:
            return path
        return re.sub(rf"\blayers\.{self.layer}\b", f"layers.{{{LAYER}}}", path)


def _patches(recorder):
    """What the step runs under besides the modes, as one stack of patches.

    * A raw Triton or FlyDSL launch goes to the driver below the dispatcher: it
      is lifted to its ATOM function and not launched.
    * `CpuGpuBuffer.__init__` (ATOM's own) allocates its host side and that
      side's numpy view outside the fake mode, unpinned (pinning is a driver
      allocation); its device side is ATOM's `zeros_like`, faked.
    * GDN's causal-conv1d metadata is host arithmetic over the staged
      `cu_seqlens`, and it runs outside the fake mode too.
    * A real host tensor's values are read on the host, `tolist` as `numpy`.
    * ATOM's MoE wrappers run their bodies, not their fakes: inside are the
      collectives, the streams and the expert kernels.
    * The checkpoint's weights iterator yields nothing: ATOM's dummy load still
      runs it, which would open every shard, or download them for a hub name.
    * A step ATOM splits into two micro-batches (`--enable-tbo`) is refused by
      name before its threads start: one graph cannot hold the two.
    """
    import flydsl.compiler
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from triton.runtime.jit import JITFunction

    from atom.model_loader import loader
    from atom.model_ops import module_dispatch_ops, moe
    from atom.model_ops.attentions import gdn_attn
    from atom.utils import CpuGpuBuffer
    from atom.utils.tbo.ubatch_wrapper import UBatchWrapper

    def run(kernel, *args, **kwargs):
        recorder.launch(f"triton.{kernel.__name__}")

    def launch(kernel, *args):
        recorder.launch(f"flydsl.{getattr(kernel, 'func', kernel).__name__}")

    init, zeros, numpy = CpuGpuBuffer.__init__, torch.zeros, torch.Tensor.numpy
    conv1d, tolist = gdn_attn.compute_causal_conv1d_metadata, torch.Tensor.tolist

    def host_tolist(tensor):
        if isinstance(tensor, FakeTensor):
            return tolist(tensor)
        with unset_fake_temporarily():
            return tolist(tensor)

    def host_zeros(*size, pin_memory=False, **kwargs):
        if torch.device(kwargs.get("device") or "cpu").type != "cpu":
            return zeros(*size, **kwargs)
        with unset_fake_temporarily():
            return zeros(*size, **kwargs)

    def host_numpy(tensor, *args, **kwargs):
        with unset_fake_temporarily():
            return numpy(tensor, *args, **kwargs)

    def staged_init(self, *args, **kwargs):
        with (
            patch.object(torch, "zeros", host_zeros),
            patch.object(torch.Tensor, "numpy", host_numpy),
        ):
            init(self, *args, **kwargs)

    def micro_batches(*args, **kwargs):
        raise ValueError(
            f"{recorder.shape} runs two micro-batches under --enable-tbo, which "
            "one op_graph cannot hold"
        )

    def host_conv1d(*args, **kwargs):
        with unset_fake_temporarily():
            return conv1d(*args, **kwargs)

    stack = contextlib.ExitStack()
    for owner, name, value in (
        (JITFunction, "run", run),
        (flydsl.compiler, "compile", launch),
        (CpuGpuBuffer, "__init__", staged_init),
        (gdn_attn, "compute_causal_conv1d_metadata", host_conv1d),
        (torch.Tensor, "tolist", host_tolist),
        (loader, "safetensors_weights_iterator", lambda *a, **k: iter(())),
        (UBatchWrapper, "_run_ubatches", micro_batches),
        (torch.ops.aiter, "moe_forward", moe.moe_forward),
        (
            torch.ops.aiter,
            "maybe_dual_stream_forward",
            module_dispatch_ops.maybe_dual_stream_forward,
        ),
    ):
        stack.enter_context(patch.object(owner, name, value))
    return stack


def _v4_attention(v4, recorder):
    """DeepSeek-V4's attention op runs its body, not its fake, so its kernels
    are leaves of their own, as a real step runs it: the capture's batch is a
    dummy run, which the body would skip. The indexer's score and top-k stays
    one leaf: its FP4 kernels launch through aiter's own launcher, which no
    patch here intercepts.

    Two of its functions are one leaf each at their call, what they run inside
    folded in: before its raw launches, `fused_compress_attn` derives its
    `quant` parameter, and the decode kernel's aiter entry dispatches a leaf,
    either of which a lift refuses.

    The two decode kernels' leaves state the layer's `window_size`,
    `compress_ratio` and `index_topk` (None when it has no indexer), and the
    indexer's its `compress_ratio`, none of which they are passed: read off
    the layer `layer_name` names.
    """
    from atom.model_ops.v4_kernels import paged_decode
    from atom.utils.forward_context import get_forward_context

    decode = paged_decode._sparse_attn_v4_paged_decode_asm
    decodes = (f"{decode.__module__}:{decode.__qualname__}", V4_PS_DECODE)

    def body(x, positions, layer_name):
        layer = recorder.contexts[layer_name]
        top = layer.indexer and layer.indexer.index_topk
        ratio = (("compress_ratio", layer.compress_ratio),)
        geometry = (("window_size", layer.window_size), *ratio, ("index_topk", top))
        stated = {**dict.fromkeys(decodes, geometry), V4_INDEXER: ratio}
        with (
            patch.object(recorder, "stated", stated),
            patch.object(get_forward_context().context, "is_dummy_run", False),
        ):
            return v4.v4_attention_with_output(x, positions, layer_name)

    def at_call(owner, fn):
        name = f"{fn.__module__}:{fn.__qualname__}"
        leaf = peers._declared(recorder, name, fn, ir.NodeKind.OPAQUE_LEAF)
        return patch.object(owner, fn.__name__, leaf)

    stack = contextlib.ExitStack()
    for each in (
        patch.object(torch.ops.aiter, "v4_attention_with_output", body),
        at_call(v4, v4.fused_compress_attn),
        at_call(paged_decode, decode),
    ):
        stack.enter_context(each)
    return stack


def _build_runner(config, fake_mode):
    """ATOM's `ModelRunner`, built in the mode in the model's dtype, which it
    leaves torch's default for the steps, as ATOM's runner does.

    The group already exists at its width, warmup would run a forward of its
    own, the checkpoint is ATOM's empty dummy (built and post-processed, no
    shard opened, under `_patches`), and the declared device's memory is all
    free.
    """
    from atom.model_engine.model_runner import DeviceMemoryReadings, ModelRunner
    from atom.utils import h2d

    class _CapturedRunner(ModelRunner):
        def _setup_device_and_distributed(self, rank, config):
            self.device = torch.device("cuda:0")

        def _maybe_warmup(self):
            return

        def _read_device_memory(self):
            total = torch.cuda.get_device_properties(0).total_memory
            return DeviceMemoryReadings(total, total, 0, 0)

    # A fake tensor has no address: the alias check sees each buffer empty.
    with fake_mode, patch.object(h2d, "_range", lambda t: (0, 0)):
        return _CapturedRunner(0, config)


def _weights(model):
    """Every tensor the model holds: parameters, buffers and tensor attributes."""
    for module in model.modules():
        held = [*module._parameters.values(), *module._buffers.values()]
        for value in [*held, *vars(module).values()]:
            if isinstance(value, torch.Tensor):
                yield value


def _batch(shape, block_size, context=1, output=True):
    """`shape.seqs` sequences of `shape.tokens` new tokens between them, the
    first `tokens % seqs` one longer, each in blocks of its own; and the
    blocks they take. A decode sequence holds `context` tokens. Unless
    `output`, every prefill sequence is a middle chunk, so the step samples
    nothing."""
    from atom.model_engine import sequence
    from atom.model_engine.scheduler import ScheduledBatch

    per, extra = divmod(shape.tokens, shape.seqs)
    counts = [per + (i < extra) for i in range(shape.seqs)]
    seqs, blocks = {}, 0
    for index, count in enumerate(counts):
        held = context if shape.phase == "decode" else count
        # An id of 0 would draw one from the class counter instead.
        seq = sequence.Sequence([0] * held, block_size=block_size, id=index + 1)
        seq.status = sequence.SequenceStatus.RUNNING
        seq.type = sequence.SequenceType[shape.phase.upper()]
        seq.num_cached_tokens = held - count
        taken = -(-held // block_size)
        seq.block_table = sequence.new_block_table(list(range(blocks, blocks + taken)))
        seqs[seq.id], blocks = seq, blocks + taken
    batch = ScheduledBatch(
        seqs=seqs,
        num_scheduled_tokens=numpy.array(counts, dtype=numpy.int32),
        total_tokens_num=shape.tokens,
        total_seqs_num=shape.seqs,
        **{f"total_tokens_num_{shape.phase}": shape.tokens},
        **{f"total_seqs_num_{shape.phase}": shape.seqs},
        is_dummy_run=True,
        is_final_chunk=None if output else [False] * shape.seqs,
    )
    return batch, blocks


def _affine(shape: str, first, second):
    """`first`, a decode step traced at the first of `CONTEXTS`, each integer
    argument that differs in `second`, the step at the second, as an
    `ir.Affine`; refused by operator and slot when it is not affine, and when
    the two differ otherwise."""
    (c0, c1), out = CONTEXTS, []

    def bind(name, slot, u, v):
        if u == v:
            return u
        slope, rest = divmod(v - u, c1 - c0) if type(u) is type(v) is int else (0, 1)
        if rest:
            raise ValueError(
                f"{name} at {shape} takes {slot}={u!r} at context {c0} and {v!r} "
                f"at {c1}: not affine in the context"
            )
        return ir.Affine(slope, u - slope * c0)

    if len(first) != len(second):
        raise ValueError(
            f"{shape} dispatches {len(first)} operators at context {c0} and "
            f"{len(second)} at {c1}"
        )
    for (layer, a), (other, b) in zip(first, second):
        ops = isinstance(a, ir.Op) and isinstance(b, ir.Op)
        if ops and [s for s, _ in a.args] == [s for s, _ in b.args]:
            args = [
                (s, bind(a.name, s, u, v)) for (s, u), (_, v) in zip(a.args, b.args)
            ]
            a, b = (dataclasses.replace(op, args=args) for op in (a, b))
        if (layer, a) != (other, b):
            raise ValueError(
                f"{shape} differs at contexts {c0} and {c1} past an integer "
                f"argument: {getattr(a, 'name', a)} against {getattr(b, 'name', b)}"
            )
        out.append((layer, a))
    return out


def _config(model, tp, atom_args):
    """ATOM's `Config` for its engine arguments `atom_args`, eager, at level 0,
    the checkpoint an empty dummy.

    Under `--enable-dp-attention` each TP rank is a DP rank of TP 1, as
    `CoreManager` rewrites it.
    """
    import argparse
    import dataclasses

    from atom.config import Config
    from atom.model_engine.arg_utils import EngineArgs

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser(prog="trace ops --"))
    forced = ["--enforce-eager", "--level", "0", "--load_dummy", "empty"]
    args = parser.parse_args(["--model", model, "-tp", str(tp), *atom_args, *forced])
    engine = EngineArgs.from_cli_args(args)
    names = {field.name for field in dataclasses.fields(Config)}
    kwargs = engine._get_engine_kwargs().items()
    config = Config(engine.model, **{k: v for k, v in kwargs if k in names})
    if config.enable_dp_attention:
        parallel = config.parallel_config
        parallel.data_parallel_size *= config.tensor_parallel_size
        parallel.data_parallel_size_local = parallel.data_parallel_size
        config.tensor_parallel_size = 1
    return config


def capture(model: str, tp: int, shapes, tmpdir: str, atom_args=(), arch="gfx942"):
    """Trace one step of `model` at TP `tp` at each of `shapes`, in this process,
    under ATOM's engine arguments `atom_args` on a declared `arch`.

    Yields, per shape, the graph, the model's config digest, what the capture
    declared, the operators it left out of the graph among it; per prefill
    shape, the middle chunk's and then the final one's.
    """
    declared = driverless.install(tmpdir, arch)

    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from atom.compass.run import _file_digest
    from atom.config import set_current_atom_config

    config = _config(model, tp, atom_args)
    tp, dp = config.tensor_parallel_size, config.parallel_config.data_parallel_size
    ep = int(config.enable_expert_parallel)
    declared["group"] = driverless.group(tp, dp)
    set_current_atom_config(config)
    digest = _file_digest(config.model, "config.json")
    fake_mode = driverless.driverless_mode(ShapeEnv())
    contexts = config.compilation_config.static_forward_context
    recorder, block_size = _Recorder(contexts, fake_mode), config.kv_cache_block_size
    driverless.on_wait = recorder.wait
    stack = _patches(recorder)
    if dp > 1:
        for owner, name, value in peers.patches(recorder):
            stack.enter_context(patch.object(owner, name, value))
        if ep:
            peers.install_mori(recorder)
    with stack:
        runner = _build_runner(config, fake_mode)
        _check_experts(runner.model, experts(config, arch))
        if v4 := sys.modules.get("atom.models.deepseek_v4"):
            stack.enter_context(_v4_attention(v4, recorder))
        stacks = [
            module
            for name, module in runner.model.named_modules()
            if isinstance(module, torch.nn.ModuleList) and name.endswith("layers")
        ]
        if not stacks:
            raise ValueError(f"{type(runner.model).__name__} has no `layers` to fold")
        for index, layer in enumerate(max(stacks, key=len)):
            layer.register_forward_pre_hook(
                lambda m, a, i=index: setattr(recorder, "layer", i)
            )
            layer.register_forward_hook(
                lambda m, a, o: setattr(recorder, "layer", None)
            )
        runner.model.register_forward_pre_hook(
            lambda m, a: setattr(recorder, "creating", ir.Role.ACTIVATION)
        )
        recorder.modules = {id(m): n for n, m in runner.model.named_modules()}
        weights = map(_storage, _weights(runner.model))
        recorder.roles = dict.fromkeys(weights, ir.Role.WEIGHT)
        recorder.creating = ir.Role.KV
        blocks = max(_batch(s, block_size, max(CONTEXTS))[1] for s in shapes)
        with fake_mode:
            runner.get_num_blocks()  # sizes the pools, as `EngineCore` does
        # A warmup copy whose bytes nobody reads, like `_maybe_warmup`.
        runner.attn_metadata_builder.warmup_per_req_cache = lambda: None
        with fake_mode, recorder:
            runner.allocate_kv_cache(max(KV_BLOCKS, blocks))
        # A prefill is traced as a middle chunk, which samples nothing, then as
        # a final one; a decode step always samples.
        variants = [
            (shape, output)
            for shape in shapes
            for output in ((False, True) if shape.phase == "prefill" else (True,))
        ]
        for shape, output in variants:
            recorder.shape = shape
            traces = []
            for context in CONTEXTS if shape.phase == "decode" else CONTEXTS[:1]:
                recorder.creating = ir.Role.METADATA
                with fake_mode, torch._C._EnablePythonDispatcher(), recorder:
                    # A runner's first step takes its own path, so the traced
                    # step follows one of its own shape: no graph depends on
                    # another's.
                    runner.forward(_batch(shape, block_size, context, output)[0])
                    recorder.start()
                    recorder.creating, recorder.on = ir.Role.METADATA, True
                    runner.forward(_batch(shape, block_size, context, output)[0])
                recorder.creating, recorder.on = None, False
                traces.append((recorder.items, recorder.left_out))
            items, left_out = traces[0]
            if len(traces) > 1:
                items = _affine(str(shape), items, traces[1][0])
            key = structure(config, shape.phase, output=output)
            crossing, traced = [], Traced(key, str(shape))
            graph = build(items, traced, crossing)
            left_out = [*left_out, *(("step-crossing", op) for op in crossing)]
            declared["left_out"] = [{"why": why, **_json(op)} for why, op in left_out]
            yield graph, digest, dict(declared)
