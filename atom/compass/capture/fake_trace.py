# SPDX-License-Identifier: MIT
"""Route A capture: `TorchDispatchMode` + `FakeTensorMode(ShapeEnv)` + the Python
dispatcher (`04` D18), applied to ATOM's real model classes.

Scope is the P0.4 spike: build the module tree device-free at a logical TP width,
run one forward through it, record the operator list, and *refuse loudly* rather
than record a plausible-looking partial one. Nothing here fits a cost model; the
artifact is an inventory.

Three things in here are not in `04` D18 and were found by running it:

* D18's five `torch.cuda` stubs are not enough to import ATOM. `get_device_properties`
  is read at *import* time by aiter's Triton attention kernels
  (`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py:111`, via
  `bwd.py:224 get_bwd_configs`) and the failure is a `NameError` from torch's own
  `get_device_properties` once `_lazy_init` is stubbed out. It is a *declared*
  device reading in the sense of `03` D14, so it is passed in, never read.
* Attention arrives in the inventory as ONE opaque leaf and nothing below it.
  `aiter.unified_attention_with_output_base` and
  `aiter.linear_attention_with_output_base` are registered custom ops, so under
  FakeTensorMode the registered *fake impl* answers and the Python body never
  runs. One consequence is worth stating because it is the opposite of what it
  looks like: `attention_mha.py:178` short-circuits attention when
  `context.is_dummy_run` is set, and that branch is NEVER REACHED under a fake
  trace -- so a dummy-batch capture is not missing attention. Measured: the
  warmup capture of the 27B records 16 unified + 48 linear attention
  dispatches, identical to the non-dummy decode capture.
* A logical TP width wider than the number of ranks present is ATOM's own
  `apply_simulated_tp` (`atom/distributed/simulated_tp.py`), reused rather than
  reimplemented (principle 1). Its documented caveat is usually read as "the
  values are meaningless, the shapes are right", which would cost a capture
  nothing. It costs a capture something else: at one physical rank
  `_patch_group` replaces `all_reduce` with the IDENTITY, so a TP2 capture
  taken this way contains NO COLLECTIVE OPERATORS AT ALL. Measured: the TP2
  decode capture of the 27B has zero `c10d.*` dispatches inside the model. Any
  cost model fed from it would price TP2 as collective-free.
"""

from __future__ import annotations

import contextlib
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._python_dispatch import TorchDispatchMode


class CaptureRefusal(Exception):
    """A declined capture with a named reason (principle 6)."""


# --------------------------------------------------------------------------
# device readings, declared


@dataclass(frozen=True)
class DeviceReadings:
    """The device facts ATOM's import and construction paths read.

    Declared, never queried (`03` D14). Defaults describe node 18's MI308X;
    a caller modelling another device passes its own.
    """

    cu_count: int = 80
    gcn_arch: str = "gfx942"
    capability: tuple = (9, 4)
    total_memory_bytes: int = 192 * (1 << 30)


def install_device_stubs(readings: DeviceReadings | None = None) -> list:
    """Stub `torch.cuda`. Returns the names stubbed, so a run can report them.

    The first five are `04` D18's list verbatim. The rest are named BEYOND-D18
    because they were added here and D18 should say so.
    """
    readings = readings or DeviceReadings()
    installed = []

    def put(name: str, value: Callable, beyond: bool = False) -> None:
        setattr(torch.cuda, name, value)
        installed.append(name + ("(BEYOND-D18)" if beyond else ""))

    put("is_available", lambda: True)
    put("device_count", lambda: 1)
    put("_lazy_init", lambda *a, **k: None)
    put("get_rng_state", lambda *a, **k: torch.zeros(16, dtype=torch.uint8))
    put("set_rng_state", lambda *a, **k: None)

    class _Props:
        multi_processor_count = readings.cu_count
        gcnArchName = readings.gcn_arch
        name = readings.gcn_arch
        major, minor = readings.capability
        total_memory = readings.total_memory_bytes
        warp_size = 64
        max_threads_per_multi_processor = 2048
        L2_cache_size = 8 << 20
        regs_per_multiprocessor = 65536
        shared_memory_per_block = 65536

    put("get_device_properties", lambda *a, **k: _Props(), beyond=True)
    put("current_device", lambda: 0, beyond=True)
    put("get_device_capability", lambda *a, **k: readings.capability, beyond=True)
    return installed


def _free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


# --------------------------------------------------------------------------
# the module tree


def init_single_rank_group(backend: str = "gloo") -> None:
    """One physical rank, no device, through aiter's own initialiser.

    A real group of width N needs N processes to arrive; asking gloo for a
    world of 2 from one process waits forever. Width comes from
    `apply_simulated_tp` instead, which is how ATOM already runs the first N
    ranks of a wider deployment.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        # Idempotent on purpose. `apply_simulated_tp` has by then set the tp
        # group's reported world_size to the LOGICAL width, and aiter's
        # `ensure_model_parallel_initialized` asserts that reported width
        # against the physical one it is asked for -- so a second call fails
        # with "already initialized, but of unexpected size", which reads like
        # a TP bug and is not one.
        return
    from aiter import init_dist_env

    init_dist_env(
        1,
        rankID=0,
        backend=backend,
        distributed_init_method="env://",
        local_rank=0,
    )


def build_config(model: str, tp: int):
    """ATOM's own Config, at a logical TP width, reading no checkpoint."""
    from atom.config import Config, set_current_atom_config
    from atom.distributed.simulated_tp import apply_simulated_tp

    config = Config(
        model=model,
        tensor_parallel_size=tp,
        load_dummy="empty",  # `02` D10.1: no checkpoint bytes
        fake_eplb=(tp > 1),  # what `apply_simulated_tp` is gated on
    )
    set_current_atom_config(config)
    if config.tp_world_size != 1:
        raise CaptureRefusal(
            f"tp_world_size resolved to {config.tp_world_size}, not 1: this "
            "process holds one rank and the logical width must come from "
            "apply_simulated_tp, not from a real group."
        )
    apply_simulated_tp(config)
    return config


def build_model(config, fake_mode: FakeTensorMode):
    """Construct ATOM's real model class inside the fake mode.

    Built in the model's own dtype: at torch's fp32 default, fake tensors trace
    kernels real hardware rejects (AITER's fused qk-rmsnorm is fp16/bf16 only),
    so the graph would not be the one that runs.
    """
    from atom.model_engine.model_runner import support_model_arch_dict
    from atom.utils import resolve_obj_by_qualname

    arch = config.hf_config.architectures[0]
    if arch not in support_model_arch_dict:
        raise CaptureRefusal(f"{arch} is not in ATOM's support_model_arch_dict")
    model_class = resolve_obj_by_qualname(support_model_arch_dict[arch])
    prev = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype)
    try:
        with fake_mode, torch.device("cuda"):
            model = model_class(config)
    finally:
        torch.set_default_dtype(prev)
    return arch, model_class, model


# --------------------------------------------------------------------------
# the recorder


@dataclass
class OpRecord:
    op: str
    in_shapes: list
    out_shapes: list
    dtypes: list
    stream: Any = None


@dataclass
class Recorder(TorchDispatchMode):
    """Every dispatched operator, with its shapes as the ShapeEnv reports them.

    Shapes are stringified rather than kept as `SymInt`s: an inventory is
    compared across widths and across runs, and a `SymInt` compares by
    identity of its symbol, which differs between two ShapeEnvs that agree.
    """

    ops: list = field(default_factory=list)
    _depth: int = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.ops.append(
            OpRecord(
                op=str(func),
                in_shapes=[_shape_of(a) for a in args if _is_tensor(a)],
                out_shapes=[_shape_of(o) for o in _flatten(out) if _is_tensor(o)],
                dtypes=sorted({str(a.dtype) for a in args if _is_tensor(a)}),
            )
        )
        return out


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


def _flatten(x):
    if isinstance(x, (list, tuple)):
        for i in x:
            yield from _flatten(i)
    else:
        yield x


def _shape_of(t) -> list:
    return [str(s) for s in t.shape]


@contextlib.contextmanager
def capture(shape_env: ShapeEnv, recorder: Recorder):
    """The three disciplines of `04` D18, with the post-trace assertions.

    Discipline 1 is the `_EnablePythonDispatcher` below. Without it
    `torch.matmul` on ndim>2 takes a C++ composite that calls non-symbolic
    `sizes()`; measured on this stack under a dispatch mode it raises
    `RuntimeError: Cannot call numel() on tensor with symbolic sizes/strides`
    rather than specialising silently, which is the louder of the two
    documented failures but still a failure.
    """
    with torch._C._EnablePythonDispatcher(), recorder:
        yield recorder
    if shape_env.replacements:
        raise CaptureRefusal(
            "the trace specialised: shape_env.replacements = "
            f"{dict(shape_env.replacements)}. A specialised graph prices as a "
            "constant where the model is not one (`04` D18 records 8.44x at "
            "T=512, 46x at T=4096)."
        )
    shape_env.freeze()


# --------------------------------------------------------------------------
# stubs the RUNNER path needs on top of the construction path


class _NullEvent:
    def record(self, *a, **k):
        return None

    def wait(self, *a, **k):
        return None

    def synchronize(self, *a, **k):
        return None

    def query(self, *a, **k):
        return True

    def elapsed_time(self, *a, **k):
        return 0.0


class _NullStream:
    def __init__(self, *a, **k):
        self.cuda_stream = 0

    def record_event(self, *a, **k):
        return _NullEvent()

    def wait_event(self, *a, **k):
        return None

    def wait_stream(self, *a, **k):
        return None

    def synchronize(self, *a, **k):
        return None

    def query(self, *a, **k):
        return True


def install_runner_stubs() -> list:
    """`ModelRunner.__init__` reads more of `torch.cuda` than construction does.

    All BEYOND-D18. Streams and events are the interesting entry: they are not
    *readings* in the `03` D14 sense -- they are ordering primitives, and a
    capture has one order by construction -- so a null object is the whole of
    their content here. `memory_stats` IS a reading, and returning zeros is a
    declaration that the capture measures no memory; `03`'s memory terms are
    modelled elsewhere and must not be back-filled from a fake trace.
    """
    installed = []

    def put(name, value):
        setattr(torch.cuda, name, value)
        installed.append(name + "(BEYOND-D18)")

    put("Stream", _NullStream)
    # NOT a lambda: `model_runner.py:216` evaluates `torch.cuda.Event | None`
    # in a class body at import, and `function | None` is a TypeError. A stub
    # for a name used in an annotation has to stay a type.
    put("Event", _NullEvent)
    put("current_stream", lambda *a, **k: _NullStream())
    put("default_stream", lambda *a, **k: _NullStream())
    put("set_device", lambda *a, **k: None)
    put("synchronize", lambda *a, **k: None)
    put("empty_cache", lambda *a, **k: None)
    put("reset_peak_memory_stats", lambda *a, **k: None)
    put(
        "memory_stats",
        lambda *a, **k: {
            "allocated_bytes.all.current": 0,
            "allocated_bytes.all.peak": 0,
            "reserved_bytes.all.current": 0,
        },
    )
    put("mem_get_info", lambda *a, **k: (0, 0))
    put("max_memory_allocated", lambda *a, **k: 0)
    put("memory_allocated", lambda *a, **k: 0)
    put("memory_reserved", lambda *a, **k: 0)
    put("stream", lambda s: contextlib.nullcontext())
    return installed


# --------------------------------------------------------------------------
# raw Triton launches -- the thing `04` D18 does not cover


@dataclass
class TritonLaunch:
    kernel: str
    where: str
    n: int = 1


class TritonLaunchRecorder:
    """Record `triton.jit` kernel launches and do NOT launch them.

    DIAGNOSTIC, not a capture mechanism. A raw `kernel[grid](...)` launch goes
    straight to the AMD driver: it never enters the torch dispatcher, so
    `TorchDispatchMode` cannot see it and `FakeTensorMode` cannot fake it. The
    first one reached ends a fake trace with

        ValueError: Pointer argument (at 0) cannot be accessed from Triton
        (cpu tensor?)                     triton/backends/amd/driver.py:369

    because the launcher asks a FakeTensor for its `data_ptr`. Skipping the
    launch keeps the trace alive long enough to *enumerate* which of ATOM's
    154 `@triton.jit` kernels a given forward reaches -- which is the input
    the escalation needs. The operator inventory produced with this installed
    is NOT a cost model input: a skipped kernel writes nothing, so anything
    downstream that branches on its output is being fed uninitialised fake
    memory. Runs that use it must say so.
    """

    def __init__(self) -> None:
        self.launches: dict = {}
        self._orig = None

    def __enter__(self) -> Self:  # noqa: F821
        import inspect

        from triton.runtime.jit import JITFunction

        self._orig = JITFunction.run
        launches = self.launches

        def run(jf, *args, **kwargs):
            try:
                src = inspect.getsourcefile(jf.fn) or "?"
                line = inspect.getsourcelines(jf.fn)[1]
                where = f"{src}:{line}"
            except Exception:  # noqa: BLE001  # pragma: no cover
                where = "?"
            key = f"{getattr(jf, 'fn', jf).__module__}.{jf.__name__}"
            rec = launches.get(key)
            if rec is None:
                launches[key] = TritonLaunch(kernel=jf.__name__, where=where)
            else:
                rec.n += 1

        JITFunction.run = run
        return self

    def __exit__(self, *exc) -> None:
        from triton.runtime.jit import JITFunction

        JITFunction.run = self._orig

    def report(self) -> dict:
        items = sorted(self.launches.items(), key=lambda kv: (-kv[1].n, kv[0]))
        return {
            "n_distinct_triton_kernels": len(items),
            "n_triton_launches": sum(v.n for _, v in items),
            "triton_kernels": [
                {"kernel": k, "defined_at": v.where, "launches": v.n} for k, v in items
            ],
        }
