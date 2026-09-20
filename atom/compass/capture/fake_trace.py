# SPDX-License-Identifier: MIT
"""Device-free operator capture for ATOM's real model classes.

`TorchDispatchMode` + `FakeTensorMode(ShapeEnv)` + the Python dispatcher. Builds
the module tree at a logical TP width, runs one forward through it, and records
every dispatched operator with its shapes. The artifact is an inventory of what
a step executes -- not a cost model -- and anything the mechanism cannot record
faithfully is refused rather than recorded partially.

Four properties of the inventories this has produced, each found by running it,
each of which a reader of one needs:

* **The traces are CONCRETE, not symbolic.** Across the four committed forward
  records, 0 shape entries out of 12,425 / 12,455 / 12,490 / 12,520 are
  non-numeric. The cause is that every tensor is allocated inside
  `with fake_mode:`, which yields an already-fake *static* tensor with plain
  `int` shapes and no indication anything went wrong. A concrete inventory is
  only valid at the shapes it was taken at, so `capture()` below refuses one
  unless the caller passes `concrete_ok=True`.

* **Importing ATOM needs more `torch.cuda` stubs than construction does.**
  `get_device_properties` is read at *import* time by aiter's Triton attention
  kernels (`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py:111`,
  via `bwd.py:224 get_bwd_configs`); the failure without it is a `NameError`
  from torch's own `get_device_properties` once `_lazy_init` is stubbed out.

* **Attention arrives as ONE opaque leaf with nothing below it.**
  `aiter.unified_attention_with_output_base` and
  `aiter.linear_attention_with_output_base` are registered custom ops, so under
  FakeTensorMode the registered fake impl answers and the Python body never
  runs. One consequence is the opposite of what it looks like:
  `attention_mha.py:178` short-circuits attention when `context.is_dummy_run` is
  set, and that branch is never reached under a fake trace -- so a dummy-batch
  capture is not missing attention. Measured: the warmup capture of the 27B
  records 16 unified + 48 linear attention dispatches, identical to the
  non-dummy decode capture.

* **A TP>1 capture at one physical rank both erases and fabricates
  collectives.** The logical width comes from ATOM's `apply_simulated_tp`
  (`atom/distributed/simulated_tp.py`), whose `_patch_group` replaces
  `all_reduce` with the identity at one rank, so no collective operator reaches
  the inventory. Counted by instrumenting the patched group rather than by
  reading the inventory: **129 `all_reduce` calls per forward at TP2** -- 128
  from `model_ops/communication_op.py:58 tensor_model_parallel_all_reduce` (the
  row-parallel linears: 64 `mlp.down_proj` + 48 `linear_attn.out_proj` + 16
  `self_attn.o_proj`, zero in the vision tower) and 1 from
  `model_ops/embed_head.py:175` (the vocab-parallel embedding reduce), identical
  at warmup and at decode. Against that 129 the inventory records **0** `c10d.*`
  dispatches, plus one erased sampler `broadcast`. In the other direction
  `_all_gather`, `_gather` and `_reduce_scatter_tensor` are NOT identities at
  one rank -- they build the absent ranks out of zeros, and every op they use is
  dispatched and recorded. The single `all_gather` this forward makes
  (`model_ops/embed_head.py:257`, the vocab-parallel lm_head) arrives as six ops
  a real TP2 forward never issues -- `aten.zeros` -> `aten.slice` ->
  `aten.copy_` -> `aten.view` -> `aten.movedim` -> `aten.reshape`, at indices
  2458-2466 of `real_tp2.json` -- and half of the `[2,248320]` logits tensor
  they produce is zeros. So a cost model fed from a TP2 inventory would price
  TP2 as collective-free and also pay for a gather that exists only because of
  the substitution. `simulated_tp.py`'s own docstring says the model output is
  meaningless under `--fake-eplb`; that caveat travels with any structural
  reading of this inventory.
"""

from __future__ import annotations

import contextlib
import os
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._python_dispatch import TorchDispatchMode


class CaptureRefusal(Exception):
    """A capture that was declined, carrying the reason it was declined."""


# --------------------------------------------------------------------------
# device readings, declared


@dataclass(frozen=True)
class DeviceReadings:
    """The device facts ATOM's import and construction paths read.

    Supplied by the caller and never queried from a runtime, so a capture can
    describe a device the host is not. Defaults describe node 18's MI308X.
    """

    cu_count: int = 80
    gcn_arch: str = "gfx942"
    capability: tuple = (9, 4)
    total_memory_bytes: int = 192 * (1 << 30)


def install_device_stubs(readings: DeviceReadings | None = None) -> list[dict]:
    """Stub the `torch.cuda` names reached while importing ATOM and building a model.

    Install these *before* importing ATOM: aiter's Triton attention kernels read
    `get_device_properties` at import time. `is_available` must report True --
    on False, `FakeTensorMode.__enter__` takes its `avoid_device_init` path and
    probes the driver anyway.

    Returns one `{"name": ..., "needed_for": "import"}` entry per stubbed name,
    so a run record can state exactly what was substituted and when it was
    needed.
    """
    readings = readings or DeviceReadings()
    installed = []

    def put(name: str, value: Callable) -> None:
        setattr(torch.cuda, name, value)
        installed.append({"name": name, "needed_for": "import"})

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

    put("get_device_properties", lambda *a, **k: _Props())
    put("current_device", lambda: 0)
    put("get_device_capability", lambda *a, **k: readings.capability)
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
        load_dummy="empty",  # construct the module tree, read no checkpoint bytes
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


_NUMERIC = re.compile(r"^-?\d+$")


def shape_entry_census(recorder: Recorder) -> tuple[int, int]:
    """`(shape entries recorded, entries that are not a plain integer)`.

    The second number is what separates a symbolic inventory from a concrete
    one: if every shape is an integer, the inventory describes only the shapes
    it was taken at and cannot be evaluated anywhere else.
    """
    total = non_numeric = 0
    for rec in recorder.ops:
        for shape in (*rec.in_shapes, *rec.out_shapes):
            for entry in shape:
                total += 1
                if not _NUMERIC.match(str(entry)):
                    non_numeric += 1
    return total, non_numeric


@contextlib.contextmanager
def capture(shape_env: ShapeEnv, recorder: Recorder, concrete_ok: bool = False):
    """Run a trace with the Python dispatcher enabled, then check what came out.

    `_EnablePythonDispatcher` is not optional: without it `torch.matmul` on
    ndim>2 takes a C++ composite that calls non-symbolic `sizes()`. Measured on
    this stack under a dispatch mode it raises `RuntimeError: Cannot call
    numel() on tensor with symbolic sizes/strides`; on stacks where it does not
    raise, it specialises the graph silently.

    Two post-trace checks, because they catch different failures.
    `shape_env.replacements` catches a symbol that was created and then
    specialised. It says nothing about a trace where no symbol was ever created
    -- there an empty `replacements` means "nothing was checked", not "clean" --
    so the free-symbol census runs as well, and a trace with no free symbol is
    refused unless the caller declared it wanted a concrete one.

    `concrete_ok=True` does not soften the refusal into a fallback: it changes
    what is being asked for, and the result is recorded as a concrete trace
    rather than reported as a symbolic one.
    """
    with torch._C._EnablePythonDispatcher(), recorder:
        yield recorder
    if shape_env.replacements:
        raise CaptureRefusal(
            "the trace specialised: shape_env.replacements = "
            f"{dict(shape_env.replacements)}. A specialised graph prices as a "
            "constant where the model is not one -- a graph that went linear "
            "where the model is quadratic is 8.44x wrong at T=512 and 46x "
            "wrong at T=4096."
        )
    entries, free = shape_entry_census(recorder)
    if free == 0 and not concrete_ok:
        raise CaptureRefusal(
            f"the trace carries no free symbol: 0 of {entries} recorded shape "
            "entries are non-numeric, so every shape in this inventory is a "
            "constant and the inventory is valid only at the shapes it was "
            "taken at. An empty `shape_env.replacements` on a trace that "
            "created no symbol means NOTHING WAS CHECKED, not that the trace "
            "is clean. Either ask for a concrete trace deliberately "
            "(concrete_ok=True), or keep the symbols: build the input tensors "
            "outside the fake mode and convert them with an explicit "
            "`StatelessSymbolicContext`."
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
        # NOT 0.0. `model_runner.py:4144` does
        # `times_ms.append(start.elapsed_time(end))`, so a zero here is a
        # confident, precise, entirely fictional duration. A capture measures
        # no time; say so. Nothing on the current capture path reaches this.
        raise CaptureRefusal(
            "torch.cuda.Event.elapsed_time was read under a fake capture. A "
            "capture runs no kernel, so it has no elapsed time to report and "
            "will not invent one. Timing belongs to the cost model, not to "
            "the trace."
        )


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


def install_runner_stubs(readings: DeviceReadings | None = None) -> list[dict]:
    """Stub the extra `torch.cuda` names `ModelRunner.__init__` reads.

    On top of `install_device_stubs`; returns
    `{"name": ..., "needed_for": "model_runner"}` entries.

    Streams and events are not device *readings* -- they are ordering
    primitives, and a capture has one order by construction -- so a null object
    is the whole of their content here. `memory_stats` IS a reading, and
    returning zeros declares that the capture measures no memory; memory is
    modelled elsewhere and must not be back-filled from a fake trace.

    `mem_get_info` is a reading too, and it is the one that must not be zero.
    ATOM sizes the KV budget from it at `model_runner.py:1590`
    (`budget = gpu_memory_utilization * torch.cuda.mem_get_info()[1]`), so a
    `(0, 0)` there is a KV budget of exactly zero -- precise, confident and
    fictional. It comes from `DeviceReadings` like every other declared fact,
    and free == total because a capture has allocated nothing.
    """
    readings = readings or DeviceReadings()
    installed = []

    def put(name, value):
        setattr(torch.cuda, name, value)
        installed.append({"name": name, "needed_for": "model_runner"})

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
    put(
        "mem_get_info",
        lambda *a, **k: (readings.total_memory_bytes, readings.total_memory_bytes),
    )
    put("max_memory_allocated", lambda *a, **k: 0)
    put("memory_allocated", lambda *a, **k: 0)
    put("memory_reserved", lambda *a, **k: 0)
    put("stream", lambda s: contextlib.nullcontext())
    return installed


# --------------------------------------------------------------------------
# raw Triton launches -- the traffic the dispatcher never sees


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
    154 `@triton.jit` kernels a given forward reaches, which is what deciding
    how to price them needs. The operator inventory produced with this
    installed is NOT a cost model input: a skipped kernel writes nothing, so
    anything downstream that branches on its output is reading uninitialised
    fake memory. Runs that use it must say so.
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
