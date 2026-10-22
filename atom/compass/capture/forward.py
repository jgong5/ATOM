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
what its input preparation allocated) is metadata. A functional collective's
group is the aiter coordinator it belongs to (`_coordinator`).

A raw Triton or FlyDSL launch lifts to the function that issues it
(`_Recorder`): one leaf, that function by its module and name, its parameters at
call entry as the arguments, with whatever else the call ran folded in. An
argument that is an attribute of the runner is a `ModuleRef` `runner.<attr>`;
any other object is held as pickle reduces it (`ir.nodes.Reduced`), or refused
by its type.

The engine is ATOM's own `Config` for ATOM's engine arguments, on a declared
device of one arch. Under DP the capture runs one rank, and its peers the rest of
the rows the group's MoE gathers (`peers`).

Under a drafter a decode row verifies the drafter's tokens, a real step's
batch so the rejection sampler runs, and a step that samples is cut where ATOM
records `forward_done_event`: the target's work before it, the drafter's after
it (its passes, the shared embedding and LM head among them), each its own
graph. A block drafter writes its draft KV before the cut and is refused.
"""

import contextlib
import copyreg
import dataclasses
import enum
import importlib
import inspect
import json
import pathlib
import re
import sys
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
from atom.compass.ir.nodes import Reduced

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
#: Operators that ask a fake tensor where it is and run nothing on a device.
UNDISPATCHED = ("profiler.", "prim.device")
#: DeepSeek-V4's decode attention leaves: the fp8 KV cache's, the bf16 one's and the
#: persistent kernel; and its indexer's score and top-k.
V4_DECODES = (
    "aiter.mla:mla_decode_fwd_v4_nm",
    "atom.model_ops.v4_kernels.paged_decode:_sparse_attn_v4_paged_decode_triton",
    "aiter.mla_decode_v4_ps_asm.default",
)
V4_INDEXER = "aiter.indexer_score_topk.default"
#: Where a drafter's step ends its target's work: ATOM records `forward_done_event`.
CUT = "forward_done_event"

_ROOT = pathlib.Path(atom.__file__).resolve().parent.parent
_IS_ATOM: dict = {}


class Shape(NamedTuple):
    """A step's shape: `tokens` query tokens over `seqs` sequences, and under DP
    `gathered`, the rows the group's MoE gathers, None at every rank's `tokens`."""

    phase: str  # "decode" or "prefill"
    tokens: int
    seqs: int
    gathered: int | None = None

    def __str__(self) -> str:
        text = f"prefill T={self.tokens} B={self.seqs}"
        text = text if self.phase == "prefill" else f"decode B={self.seqs}"
        return text if self.gathered is None else f"{text} G={self.gathered}"

    def at(self, dp: int) -> "Shape":
        """This shape in a DP group of `dp`: `gathered` only off `dp x tokens`."""
        return (
            self._replace(gathered=None) if self.gathered == dp * self.tokens else self
        )


def decode(width: int, gathered: int | None = None) -> Shape:
    if width < 1:
        raise ValueError(f"a decode width is at least 1, got {width}")
    return _gathering(Shape("decode", width, width, gathered))


def prefill(tokens: int, seqs: int, gathered: int | None = None) -> Shape:
    if not 1 <= seqs <= tokens:
        raise ValueError(
            f"a prefill of {tokens} tokens over {seqs} sequences: each sequence "
            "takes at least one token"
        )
    return _gathering(Shape("prefill", tokens, seqs, gathered))


def _gathering(shape: Shape) -> Shape:
    if shape.gathered is not None and shape.gathered < shape.tokens:
        raise ValueError(
            f"a {shape.phase} of {shape.tokens} tokens whose MoE gathers "
            f"{shape.gathered} rows: the gather holds this rank's"
        )
    return shape


def device_arch() -> str:
    """The visible device's arch, its target features dropped."""
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def structure(
    config,
    phase: str,
    eager: bool | None = None,
    device: str | None = None,
    output: bool = True,
    draft: bool = False,
) -> str:
    """The structure an `op_graph` of `config` at a `phase` step is keyed by: the
    architecture, the config.json digest, the arch of `device`, else
    `device_arch()`, the engine config a step law is keyed by less its ladder
    (`run.step_grid.keyed`), which names every variable that picks a kernel and
    a drafter's method and width, the TP width, the DP width and the EP group's
    when they are set, the phase and the mode, `config`'s unless `eager` names
    it, ``no output`` unless the step produces output (`output`), and ``draft``
    for the drafter's part of the step (`draft`)."""
    from atom.compass.run import _file_digest
    from atom.compass.run.step_grid import keyed

    digest = _file_digest(config.model, "config.json")
    engine = keyed(config, ladder=False)
    engine = json.dumps(engine, sort_keys=True, separators=(",", ":"))
    mode = "eager" if (config.enforce_eager if eager is None else eager) else "graph"
    arch, device = config.hf_config.architectures[0], device or device_arch()
    tp, dp = config.tensor_parallel_size, config.parallel_config.data_parallel_size
    widths = f"tp={tp}" + (f" dp={dp}" if dp > 1 else "")
    widths += f" ep={tp * dp}" if config.enable_expert_parallel else ""
    key = f"{arch} {device} {digest} {engine} {widths} {phase} {mode}"
    key = key if output else f"{key} no output"
    return f"{key} draft" if draft else key


def _engine(text: str):
    """A structure's `text` before its engine config, the config, and the text
    after; None for a structure that states none."""
    head = " ".join(text.split(" ", 3)[:3])
    try:
        engine, end = json.JSONDecoder().raw_decode(text, len(head) + 1)
    except ValueError:
        return None
    return isinstance(engine, dict) and (head, engine, text[end:])


def missed(store, key) -> str:
    """For an `op_graph` `key` `store` holds no entry at, each engine field
    (`step_grid.differ`) in which a graph it holds at the key's shape and
    otherwise the key's structure differs, as text; "" when it holds none."""
    from atom.compass.artifacts.store import ENTRY_FILE
    from atom.compass.run.step_grid import differ

    head, here, tail = _engine(key.values["structure"])
    named = []
    for path in sorted((store.root / key.kind.value).glob(f"[!.]*/{ENTRY_FILE}")):
        there = json.loads(path.read_bytes())["key"]
        if there.get("shape") == key.values["shape"]:
            at = _engine(there.get("structure", ""))
            if at and (at[0], at[2]) == (head, tail):
                named.append(", ".join(differ(at[1], here, "the graph")))
    return "".join(
        f"; a graph the store holds at this shape and structure differs in engine "
        f"config: {n}"
        for n in named
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
    path = pathlib.Path(frame.f_code.co_filename)
    path = path.relative_to(_ROOT) if path.is_relative_to(_ROOT) else path
    return f"{path}:{frame.f_lineno} in {frame.f_code.co_qualname}"


def _storage(tensor) -> StorageWeakRef:
    return StorageWeakRef(tensor.untyped_storage())


def _params(code) -> tuple[str, ...]:
    """A function's parameters, by name, as its code object lists them."""
    count = code.co_argcount + code.co_kwonlyargcount
    count += bool(code.co_flags & inspect.CO_VARARGS)
    count += bool(code.co_flags & inspect.CO_VARKEYWORDS)
    return code.co_varnames[:count]


def _function(name: str):
    """The function a lifted leaf names, `module:qualname`."""
    module, qualname = name.split(":")
    fn = importlib.import_module(module)
    for attr in qualname.split("."):
        fn = getattr(fn, attr)
    return fn


def _rebuilt(value, live):
    """An argument value as a call takes it: a `Reduced` object rebuilt (an
    enum member from its value, a dict from its items, any other object from
    its class and state), tuples item by item, and `Tensor` and `ContextRef`
    values through `live`."""
    if isinstance(value, tuple):
        return tuple(_rebuilt(item, live) for item in value)
    if isinstance(value, (ir.Tensor, ir.ContextRef)):
        return live(value)
    if not isinstance(value, Reduced):
        return value
    return rebuild_object(value.cls, _rebuilt(value.state, live))


def rebuild_object(cls: str, state):
    """The object pickle reduced to class `cls`, `module:qualname`, and `state`:
    an enum member from its value, a dict from its (key, value) pairs, any other
    object from its (attribute, value) pairs."""
    cls = _function(cls)
    if issubclass(cls, enum.Enum):
        return cls(state)
    state = dict(state)
    if cls is dict:
        return state
    obj = cls.__new__(cls)
    if setstate := getattr(obj, "__setstate__", None):
        setstate(state)
    else:
        obj.__dict__.update(state)
    return obj


def referenced(runner, key: str):
    """What a `ModuleRef` key names on `runner`: an attribute of the runner
    itself, `runner.<attr>`, or a module of its model by path."""
    if key.startswith("runner."):
        return getattr(runner, key.removeprefix("runner."))
    return runner.model.get_submodule(key)


def _addressable(frame) -> bool:
    """Whether the function `frame` runs is the one its `module:qualname` names
    (through `functools.wraps`), so a replay can call it again: a closure, a
    lambda or a function a decorator hides is not."""
    code = frame.f_code
    if "<" in code.co_qualname:
        return False
    try:
        fn = _function(f"{frame.f_globals['__name__']}:{code.co_qualname}")
        fn = inspect.unwrap(getattr(fn, "__func__", fn))
    except (AttributeError, ImportError, KeyError, ValueError):
        return False
    return getattr(fn, "__code__", None) is code


class _Recorder(TorchDispatchMode):
    """Every dispatched operator and launched kernel, as `ir.Op`, by layer.

    What is not ATOM's work at the traced shape goes to `left_out`, with why.
    `creating` is the role a storage allocated now takes; `roles` holds every
    role given so far. `stated` is what an op of each kind recorded now
    states beside its arguments (`ir.Op.stated`), by its name.

    A raw Triton or FlyDSL launch lifts to the function that issues it: past
    the kernel languages' own frames, closures and generic launchers (the
    kernel is one of the frame's arguments or an attribute of its `self`), the
    first function its `module:qualname` names. An untraced step finds these
    functions (`launchers`); in a traced step each call of one that launches
    a kernel or dispatches an operator outside torch's is a leaf, its
    arguments taken at entry, what it runs inside (another launching function,
    an operator) folded into it. It is called again, rebuilt from its contract
    under the fake mode, to check it launches those kernels and dispatches
    those operators again. A launch in a traced step outside every such call
    is refused, and so is a leaf that raises after it launched, or is still
    open when the step ends; one that raises having launched nothing gives
    back what it ran.
    """

    def __init__(self, contexts, fake_mode=None):
        super().__init__()
        self.contexts, self.modules, self.roles = contexts, {}, {}
        self.creating, self.on, self.fake_mode = None, False, fake_mode
        self.stated, self.runner, self.replayed = {}, None, set()
        self.launchers, self.languages = set(), ()
        self.replaying = None  # the kernels a replay launched, while one runs
        self.start()

    def start(self):
        self.layer, self.items, self.left_out = None, [], []
        self.leaf = None  # the frame of the leaf call in progress

    def __exit__(self, *exc):
        super().__exit__(*exc)
        if self.leaf is not None and exc[0] is None:
            raise ValueError(
                f"{self.items[-1][1].name} is still open when the step ends: "
                "a call that neither returned nor raised has no end to fold to"
            )

    def cut(self):
        """ATOM records `forward_done_event`: the target's work ends here, and
        what follows is the drafter's."""
        if self.on:
            self.items.append((None, CUT))

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
        torch_own = name.split(".")[0] in TORCH_NAMESPACES
        if not torch_own and not name.startswith(UNDISPATCHED):
            if self.replaying is not None:
                self.replaying.append(name)
            elif self.on and self.leaf is not None:
                self.kernels.append(name)
        if self.on and not name.startswith(UNDISPATCHED):
            kind = ir.NodeKind.CAPTURED if torch_own else ir.NodeKind.OPAQUE_LEAF
            stack = _atom_frames()
            why = "h2d" if stack and _where(stack[0]).startswith(H2D) else None
            if ins + outs and all(t.device.type == "cpu" for t in ins + outs):
                why = "host"  # host arithmetic on host tensors runs no kernel
            if driverless.standing_in[0] and not name.startswith("_c10d_functional."):
                why = "stand-in"
            slots = {a.name: v for a, v in zip(func._schema.arguments, args)}
            slots |= kwargs
            if name.startswith("_c10d_functional.") and "group_name" in slots:
                slots["group_name"] = _coordinator(slots["group_name"])
            self.add(name, kind, slots, outs, why)
        return out

    def add(self, name, kind, slots, outs, why):
        if self.leaf is not None:  # part of the leaf call in progress
            self.inner.append(lambda: self.add(name, kind, slots, outs, why))
            return
        op = self._op(name, kind, slots, [tuple(t.shape) for t in outs])
        if why:
            self.left_out.append((why, op))
        else:
            self.items.append((self.layer, op))

    def launch(self, name, *held):
        """A raw Triton or FlyDSL launch of `name`, whose kernel objects are
        `held`: found in an untraced step, part of a leaf call in a traced one."""
        if self.replaying is not None:
            self.replaying.append(name)
        elif self.on and self.leaf is not None:
            self.kernels.append(name)
        else:
            frame = self._lift({id(k) for k in held})
            if self.on:
                where = f"lifts to {_where(frame)}" if frame else "lifts to nothing"
                raise ValueError(
                    f"{name} {where}, which launched nothing in the untraced "
                    "step: no call entry took its arguments"
                )
            if frame and frame.f_code not in self.launchers:
                self.launchers.add(frame.f_code)
                events = sys.monitoring.events
                sys.monitoring.set_local_events(
                    self.tool, frame.f_code, events.PY_START | events.PY_RETURN
                )

    def _lift(self, held):
        """The frame of the function a launch of the kernel objects `held`
        lifts to, or None."""
        frame = sys._getframe(1)
        while frame is not None:
            local = frame.f_locals
            if frame.f_code.co_filename.startswith(self.languages):
                if "self" in local:  # a wrapper of the kernel: an autotuner
                    held.add(id(local["self"]))
            elif frame.f_code.co_filename != __file__ and _addressable(frame):
                values = [local.get(p) for p in _params(frame.f_code)]
                owner = getattr(values[0], "__dict__", {}) if values else {}
                if not any(id(v) in held for v in [*values, *owner.values()]):
                    return frame
            frame = frame.f_back
        return None

    def enter(self, code, offset):
        """A launching function's call begins: a leaf, unless one is open."""
        if not self.on or self.leaf is not None:
            return
        frame = sys._getframe(1)
        name = f"{frame.f_globals['__name__']}:{code.co_qualname}"
        slots = {p: frame.f_locals[p] for p in _params(code)}
        self.items.append(
            (self.layer, self._op(name, ir.NodeKind.OPAQUE_LEAF, slots, ()))
        )
        self.leaf, self.kernels, self.inner = frame, [], []

    def leave(self, code, offset, value):
        """A launching function's call ends: its leaf holds what ran inside,
        or, when it launched nothing, gives it back."""
        if self.leaf is sys._getframe(1):
            self._close()

    def unwind(self, code, offset, exc):
        """A launching function's call raises: refused once it launched, as no
        replay can check a call cut short; else it gives back what it ran."""
        if self.leaf is not sys._getframe(1):
            return
        if self.kernels:
            raise ValueError(
                f"{self.items[-1][1].name} raised {exc!r} after it launched "
                f"{self.kernels}: a call cut short has no contract to replay"
            )
        self._close()

    def _close(self):
        self.leaf, op = None, self.items[-1][1]
        if not self.kernels:
            self.items.pop()
            for each in self.inner:
                each()
        elif op not in self.replayed:
            self.replayed.add(op)
            if (again := self._replay(op)) != self.kernels:
                raise ValueError(
                    f"{op.name}, called again from its contract, launches {again}, "
                    f"not {self.kernels}"
                )

    def _replay(self, op):
        """The kernels `op` launches, called again on fresh fakes of its
        tensors and its other arguments rebuilt."""
        on, creating, self.on, self.creating = self.on, self.creating, False, None
        self.replaying = []
        try:
            kwargs = {slot: _rebuilt(value, self._live) for slot, value in op.args}
            for slot, value in kwargs.items():  # a dtype is held as its text
                held = isinstance(value, str) and value.startswith("torch.")
                held = held and getattr(torch, value[6:], None)
                kwargs[slot] = held if isinstance(held, torch.dtype) else value
            _function(op.name)(**kwargs)
            return self.replaying
        except Exception as exc:  # noqa: BLE001 - refused with what stopped it
            raise ValueError(
                f"{op.name} does not rebuild from its contract: {exc!r}"
            ) from None
        finally:
            self.on, self.creating, self.replaying = on, creating, None

    def _live(self, value):
        if isinstance(value, ir.Tensor):
            dtype = getattr(torch, value.dtype.removeprefix("torch."))
            return torch.empty_strided(
                value.shape, value.stride, dtype=dtype, device="cuda"
            )
        if isinstance(value, ir.ContextRef) and LAYER in value.index_names:
            value = value.bind(**{LAYER: self.layer})
        if not isinstance(value, ir.ModuleRef):
            return value.key
        return referenced(self.runner, value.key)

    def wait(self, waiter, waited):
        """A side stream waiting on the main one forks; the reverse joins."""
        if not self.on:
            return
        if self.leaf is not None:
            self.inner.append(lambda: self.wait(waiter, waited))
            return
        if waited is driverless.main_stream and waiter is not waited:
            item = Fork(waiter.stream_id)
        elif waiter is driverless.main_stream and waited not in (None, waiter):
            item = Join(waited.stream_id)
        else:
            return
        self.items.append((self.layer, item))

    def _op(self, name, kind, slots, out_shapes):
        """`name` as an `Op`, or a refusal naming it and what it cannot hold."""
        try:
            args = tuple((slot, self._arg(value)) for slot, value in slots.items())
            ref = next((v for _, v in args if type(v) is ir.ContextRef), None)
            stream = driverless.current[0].stream_id
            stated = self.stated.get(name, ())
            return ir.Op(name, kind, args, out_shapes, stream, ref, stated)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(
                f"{name} has no contract a graph can hold: {exc}"
            ) from None

    def _arg(self, value):
        if isinstance(value, torch.Tensor):
            role = self.roles.get(_storage(value), ir.Role.METADATA)
            shape, stride = tuple(value.shape), value.stride()
            return ir.Tensor(shape, str(value.dtype), stride, role)
        if isinstance(value, torch.nn.Module) and id(value) in self.modules:
            return ir.ModuleRef(self._layered(self.modules[id(value)]))
        if isinstance(value, str) and value in self.contexts:
            return ir.ContextRef(self._layered(value))
        if isinstance(value, (list, tuple)):
            return tuple(self._arg(v) for v in value)
        if isinstance(value, (torch.dtype, torch.device, torch.layout)):
            return str(value)
        if isinstance(value, numpy.generic):
            return value.item()
        if type(value) is dict:
            items = tuple((self._arg(k), self._arg(v)) for k, v in value.items())
            return Reduced("builtins:dict", items)
        if isinstance(value, enum.Enum):
            return Reduced(_class(value), value.value)
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        held = vars(self.runner).items() if self.runner is not None else ()
        if attr := next((a for a, v in held if v is value), None):
            return ir.ModuleRef(f"runner.{attr}")
        return self._reduced(value)

    def _reduced(self, value):
        """An object as pickle reduces it, its state a dict of attributes each
        contract-legal; anything else is refused by its type."""
        try:
            reduced = () if callable(value) else value.__reduce_ex__(2)
        except TypeError:  # an object pickle refuses
            reduced = ()
        if (
            len(reduced) < 3
            or reduced[0] is not copyreg.__newobj__
            or reduced[1] != (type(value),)
            or not isinstance(reduced[2] or {}, dict)
            or any(item is not None for item in reduced[3:])
        ):
            raise TypeError(
                f"a {type(value).__qualname__} is neither an argument value nor "
                "an object pickle reduces to its class and a dict of attributes"
            )
        state = reduced[2] or {}
        return Reduced(
            _class(value), tuple((k, self._arg(v)) for k, v in state.items())
        )

    def _layered(self, path):
        if self.layer is None:
            return path
        return re.sub(rf"\blayers\.{self.layer}\b", f"layers.{{{LAYER}}}", path)


def _coordinator(group_name: str) -> str:
    """The aiter coordinator (`tp:0`) whose device group torch names
    `group_name`, or the name as it is when none is: torch numbers groups in
    the order a process creates them, which the bench's ranks need not share."""
    from aiter.dist.parallel_state import _groups

    held = {g.device_group.group_name: k for k, r in _groups.items() if (g := r())}
    return held.get(group_name, group_name)


def _class(value) -> str:
    cls = type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def _patches(recorder):
    """What the step runs under besides the modes, as one stack of patches.

    * A raw Triton or FlyDSL launch goes to the driver below the dispatcher: it
      is told to the recorder and not launched. Triton's is taken at
      `JITFunction.run`, a warmup aside, and a direct `CompiledKernel` launch
      is refused; FlyDSL's at `JitFunction.__call__` and at a
      `CompiledFunction`'s call, which `flyc.compile` maps to its jit function.
    * Each call of a launching function, its entry and its return, is told to
      the recorder through `sys.monitoring`, on those functions alone; a raise
      that unwinds a frame, on every function.
    * `CpuGpuBuffer.__init__` (ATOM's own) allocates its host side and that
      side's numpy view outside the fake mode, unpinned (pinning is a driver
      allocation); its device side is ATOM's `zeros_like`, faked.
    * GDN's causal-conv1d metadata is host arithmetic over the staged
      `cu_seqlens`, and it runs outside the fake mode too.
    * A real host tensor's values are read on the host, `tolist` as `numpy`;
      under a drafter a device value copied to the host, which nothing
      computed, reads as zeros (its runner reads back the last step's MTP
      status and draft ids). Without one, such a read raises.
    * ATOM's MoE wrappers run their bodies, not their fakes: inside are the
      collectives, the streams and the expert kernels.
    * The checkpoint's weights iterator yields nothing: ATOM's dummy load still
      runs it, which would open every shard, or download them for a hub name.
    * A step ATOM splits into two micro-batches (`--enable-tbo`) is refused by
      name before its threads start: one graph cannot hold the two.
    """
    import flydsl
    import triton
    from flydsl.compiler import jit_function as fly
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from triton.compiler.compiler import CompiledKernel
    from triton.runtime.jit import JITFunction

    from atom.model_loader import loader
    from atom.model_ops import module_dispatch_ops, moe
    from atom.model_ops.attentions import gdn_attn
    from atom.utils import CpuGpuBuffer
    from atom.utils.tbo.ubatch_wrapper import UBatchWrapper

    recorder.languages = tuple(
        f"{pathlib.Path(m.__file__).parent}/" for m in (triton, flydsl)
    )

    def run(kernel, *args, grid=None, warmup=False, **kwargs):
        if not warmup:  # a warmup compiles and launches nothing
            recorder.launch(f"triton.{kernel.__name__}", kernel)

    def compiled_kernel(kernel, *args):
        raise ValueError(
            f"triton.{kernel.name} launches through CompiledKernel.run, past "
            "JITFunction.run: a call that took no arguments to lift"
        )

    def jit_call(jit, *args, **kwargs):
        recorder.launch(f"flydsl.{jit.func.__name__}", jit)

    def compile_impl(jit, *args):
        jit(*args)  # flyc.compile launches the kernel as it compiles it
        return fly.CompiledFunction(None, jit)

    def compiled_call(compiled, *args):
        jit = compiled._keepalive
        if not isinstance(jit, fly.JitFunction):  # compiled before the capture
            raise ValueError(  # noqa: TRY004 - a refusal the CLI names
                f"a FlyDSL CompiledFunction of {getattr(jit, '_entry', jit)!r} "
                "compiled outside the capture: no jit function names its kernel"
            )
        recorder.launch(f"flydsl.{jit.func.__name__}", compiled, jit)

    init, zeros, numpy = CpuGpuBuffer.__init__, torch.zeros, torch.Tensor.numpy
    conv1d, tolist = gdn_attn.compute_causal_conv1d_metadata, torch.Tensor.tolist
    fake_tolist = FakeTensor.tolist

    def landed(tensor):
        if not getattr(recorder.runner, "drafter", None):
            return tensor
        if isinstance(tensor, FakeTensor) and tensor.device.type == "cpu":
            with unset_fake_temporarily():
                return zeros(tensor.shape, dtype=tensor.dtype)
        return tensor

    def host_tolist(tensor):
        tensor = landed(tensor)
        if isinstance(tensor, FakeTensor):
            return fake_tolist(tensor)
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
        (CompiledKernel, "run", property(compiled_kernel)),
        (CompiledKernel, "__getitem__", compiled_kernel),
        (fly.JitFunction, "__call__", jit_call),
        (fly, "_compile_impl", compile_impl),
        (fly.CompiledFunction, "__call__", compiled_call),
        (CpuGpuBuffer, "__init__", staged_init),
        (gdn_attn, "compute_causal_conv1d_metadata", host_conv1d),
        (torch.Tensor, "tolist", host_tolist),
        (FakeTensor, "tolist", host_tolist),
        (torch.Tensor, "numpy", lambda t, *a, **k: host_numpy(landed(t), *a, **k)),
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
    stack.enter_context(_entries(recorder))
    return stack


@contextlib.contextmanager
def _entries(recorder):
    """`recorder.enter` and `recorder.leave` on each call of the functions it
    turns `sys.monitoring`'s events on for, and `recorder.unwind` on every
    frame a raise unwinds (an event no single function can take), under a
    free tool id."""
    monitoring = sys.monitoring
    events = monitoring.events
    recorder.tool = next(i for i in range(6) if monitoring.get_tool(i) is None)
    monitoring.use_tool_id(recorder.tool, "compass capture")
    monitoring.register_callback(recorder.tool, events.PY_START, recorder.enter)
    monitoring.register_callback(recorder.tool, events.PY_RETURN, recorder.leave)
    monitoring.register_callback(recorder.tool, events.PY_UNWIND, recorder.unwind)
    monitoring.set_events(recorder.tool, events.PY_UNWIND)
    try:
        yield
    finally:
        for code in recorder.launchers:
            monitoring.set_local_events(recorder.tool, code, 0)
        monitoring.set_events(recorder.tool, 0)
        for event in (events.PY_START, events.PY_RETURN, events.PY_UNWIND):
            monitoring.register_callback(recorder.tool, event, None)
        monitoring.free_tool_id(recorder.tool)


def _v4_attention(v4, recorder):
    """DeepSeek-V4's attention op runs its body, not its fake, so its kernels
    are leaves of their own, as a real step runs it: the capture's batch is a
    dummy run, which the body would skip. The indexer's score and top-k stays
    one leaf: its FP4 kernels launch through aiter's own launcher, which no
    patch here intercepts.

    The decode leaves (`V4_DECODES`) state the layer's `window_size`,
    `compress_ratio` and `index_topk` (None when it has no indexer), and the
    indexer's its `compress_ratio`, none of which they are passed: read off
    the layer `layer_name` names.
    """
    from atom.utils.forward_context import get_forward_context

    def body(x, positions, layer_name):
        layer = recorder.contexts[layer_name]
        top = layer.indexer and layer.indexer.index_topk
        ratio = (("compress_ratio", layer.compress_ratio),)
        geometry = (("window_size", layer.window_size), *ratio, ("index_topk", top))
        stated = {**dict.fromkeys(V4_DECODES, geometry), V4_INDEXER: ratio}
        with (
            patch.object(recorder, "stated", stated),
            patch.object(get_forward_context().context, "is_dummy_run", False),
        ):
            return v4.v4_attention_with_output(x, positions, layer_name)

    return patch.object(torch.ops.aiter, "v4_attention_with_output", body)


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


def _batch(shape, block_size, context=1, output=True, drafts=0):
    """`shape.seqs` sequences of `shape.tokens` new tokens between them, the
    first `tokens % seqs` one longer, each in blocks of its own; and the
    blocks they take. A decode sequence holds `context` tokens, and under a
    drafter of `drafts` tokens verifies that many more, as ATOM's scheduler
    schedules it. Unless `output`, every prefill sequence is a middle chunk,
    so the step samples nothing."""
    from atom.model_engine import sequence
    from atom.model_engine.scheduler import ScheduledBatch

    drafts = drafts if shape.phase == "decode" else 0
    per, extra = divmod(shape.tokens, shape.seqs)
    counts = [per + (i < extra) + drafts for i in range(shape.seqs)]
    seqs, blocks = {}, 0
    for index, count in enumerate(counts):
        held = context + drafts if shape.phase == "decode" else count
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
        total_tokens_num=sum(counts),
        total_seqs_num=shape.seqs,
        **{f"total_tokens_num_{shape.phase}": sum(counts)},
        **{f"total_seqs_num_{shape.phase}": shape.seqs},
        # ATOM scores drafts only on a step that is not a dummy run.
        is_dummy_run=not drafts,
        is_final_chunk=None if output else [False] * shape.seqs,
        num_spec_step=drafts,
        scheduled_spec_decode_tokens={
            i: numpy.zeros(drafts, dtype=numpy.int32) for i in seqs
        },
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
    shape, the middle chunk's and then the final one's. Under a drafter a step
    that samples is two graphs, its target's and then its drafter's, cut where
    ATOM records `forward_done_event`; a decode step verifies the drafter's
    tokens in each row.
    """
    declared = driverless.install(tmpdir, arch)

    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from atom.compass.run import _file_digest
    from atom.config import set_current_atom_config

    config = _config(model, tp, atom_args)
    tp, dp = config.tensor_parallel_size, config.parallel_config.data_parallel_size
    ep = int(config.enable_expert_parallel)
    shapes = list(dict.fromkeys(s.at(dp) for s in shapes))
    if dp == 1 and (gathering := [s for s in shapes if s.gathered is not None]):
        raise ValueError(f"{gathering[0]} gathers a DP group's rows on one rank")
    declared["group"] = driverless.group(tp, dp)
    set_current_atom_config(config)
    digest = _file_digest(config.model, "config.json")
    fake_mode = driverless.driverless_mode(ShapeEnv())
    contexts = config.compilation_config.static_forward_context
    recorder, block_size = _Recorder(contexts, fake_mode), config.kv_cache_block_size
    driverless.on_wait = recorder.wait
    # A drafter's runner imports inductor, whose decomposition table walks
    # every `torch.ops.aiter` op, which the patches below replace by functions.
    import torch._inductor.decomposition

    stack = _patches(recorder)
    if dp > 1:
        for owner, name, value in peers.patches(recorder):
            stack.enter_context(patch.object(owner, name, value))
        if ep:
            peers.install_mori(recorder)
    with stack:
        runner = _build_runner(config, fake_mode)
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
        recorder.runner = runner
        recorder.modules = {id(m): n for n, m in runner.model.named_modules() if n}
        models, drafts = [runner.model], 0
        if drafter := getattr(runner, "drafter", None):
            if drafter.is_block_drafter:
                raise ValueError(
                    f"{type(drafter).__name__} writes its draft KV before "
                    "forward_done_event on a step that samples: no cut parts its "
                    "target from its draft"
                )
            drafts, done = drafter.mtp_k, runner.forward_done_event
            models.append(drafter.model)
            done.record = lambda *a: (recorder.cut(), driverless.Event.record(done, *a))
        weights = map(_storage, (w for m in models for w in _weights(m)))
        recorder.roles = dict.fromkeys(weights, ir.Role.WEIGHT)
        recorder.creating = ir.Role.KV
        blocks = max(
            _batch(s, block_size, max(CONTEXTS), True, drafts)[1] for s in shapes
        )
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
                batch = (shape, block_size, context, output, drafts)
                with fake_mode, torch._C._EnablePythonDispatcher(), recorder:
                    # A runner's first step takes its own path (a decode
                    # fills no deferred ids), so the traced step follows two
                    # of its own shape, which find its launching functions:
                    # no graph depends on another's.
                    for _ in range(2):
                        runner.forward(_batch(*batch)[0])
                    recorder.start()
                    recorder.creating, recorder.on = ir.Role.METADATA, True
                    runner.forward(_batch(*batch)[0])
                recorder.creating, recorder.on = None, False
                traces.append((recorder.items, recorder.left_out))
            items, left_out = traces[0]
            if len(traces) > 1:
                items = _affine(str(shape), items, traces[1][0])
            # A drafter's work follows its target's: its own graph.
            cut = [i for i, (_, item) in enumerate(items) if item is CUT][:1]
            for draft, part in enumerate(
                (items[: cut[0]], items[cut[0] + 1 :]) if cut else (items,)
            ):
                key = structure(config, shape.phase, output=output, draft=bool(draft))
                crossing, traced = [], Traced(key, str(shape))
                graph = build(part, traced, crossing)
                gone = [*left_out, *(("step-crossing", op) for op in crossing)]
                declared["left_out"] = [{"why": why, **_json(op)} for why, op in gone]
                yield graph, digest, dict(declared)
