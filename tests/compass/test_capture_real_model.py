# SPDX-License-Identifier: MIT
"""One forward of the published Qwen3.8-27B, traced with no device, at TP1 and TP2.

This file is both the test and the capture driver. `pytest` runs the tests at
the bottom; each of them runs this same file as a script in a fresh interpreter
and reads the JSON record it prints. The subprocess is not isolation for its own
sake -- three pieces of the capture are process-global and one-shot, so two
widths cannot share an interpreter:

* `torch.distributed` is initialised once, at one world size;
* aiter's model-parallel state is module-global and asserts when re-entered;
* the `torch.cuda` and Triton substitutions below must be installed *before*
  `import atom`, and an import happens once per process.

Running it as a script also keeps the substitutions off every other test in the
tier: nothing here mutates `torch.cuda` in the pytest process.

What the capture is
-------------------
`TorchDispatchMode` over `FakeTensorMode(ShapeEnv())` with the Python dispatcher
enabled, around ATOM's own `ModelRunner` driving its own decode step through
ATOM's own model classes. No stub model and no synthetic config: the config is
the published `Qwen/Qwen3.8-27B` `config.json`, vendored beside this file and
checked byte-for-byte by sha256 before it is used.

**The inventories are DIAGNOSTIC, not captures.** Raw `@triton.jit` launches go
straight to the AMD driver and never enter the torch dispatcher, so
`FakeTensorMode` cannot fake them: the launcher asks a `FakeTensor` for its
`data_ptr` and the first kernel reached ends the trace. They are recorded and
not executed, which keeps the trace alive long enough to enumerate what a step
reaches -- but anything downstream of a skipped kernel read uninitialised fake
memory. The record says so in `diagnostic_inventory`, and these operator counts
are not a cost-model input at any width.

What is substituted, and what is not
------------------------------------
Substituted, all of it device facts declared by the caller rather than read from
a runtime: the `torch.cuda` namespace (8 names to import ATOM, 14 more to build
a `ModelRunner`); `rocminfo`, which aiter shells out to at import and which
needs `/dev/kfd`; Triton's active device target; the collective *transport*, so
a group of width N needs no peer; and the three primitives `CpuGpuBuffer`
reaches for, because its two allocations and its numpy view have to straddle
the mode -- see `_staged_allocators`. What is *not* substituted there is the
constructor: ATOM's own `CpuGpuBuffer.__init__` executes, and the record
carries which one ran and what it allocated, so a change inside it cannot be
silent here.

Not substituted: the process group's width. `apply_simulated_tp` is never
called -- observed by a sentinel over both bindings of it, not asserted against
a constant. At one physical rank it both erases and fabricates -- row-parallel
`all_reduce` becomes the identity and appears nowhere, while one `all_gather`
becomes six dispatched ops over a half-zeros tensor -- so a TP2 inventory taken
through it is not a TP2 inventory. The group here reports width 2 because it has
width 2, and every collective ATOM issues is dispatched and recorded.

Where the shapes specialise
---------------------------
Three sites, in an order rather than a set. A free symbol is solved by whichever
line reaches it first, so repairing one does not always close what it solved:
closing the first moves the caller's bound to a third site fourteen lines later,
in an ATOM assertion helper. The plain symbolic pass records the first two; a
third pass simulates the first closed, from outside ATOM, and records what is
behind it. All three are pinned, and there is no claim that three is all there
are -- only that these three are what this instrument can reach today.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import traceback

import numpy
import torch
from torch.utils._python_dispatch import TorchDispatchMode

# The published checkpoint this config came from: `Qwen/Qwen3.8-27B` at
# revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. The file beside this one is
# that revision's `config.json`, byte for byte -- 4,312 bytes, sha256 below,
# which is also its git blob id 706cebd746c4b6f2b1d1f892630867acfdfd3df8. Only
# the config is vendored; no weight byte is read by anything here.
CONFIG_JSON = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
CONFIG_SHA256 = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
CONFIG_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

# The decode step that is traced. Two sequences of one token each: two, not one,
# because a dimension whose trace-time hint is 1 is silently specialised to a
# constant, and a decode step has one token per sequence.
DECODE_SEQS = 2

# ATOM's own default, restated here because the second specialisation depends
# on it: the block-table buffer is `max_num_seqs` by `max_model_len //
# block_size`, and the published config's 262,144 positions over 16-token
# blocks is the 16,384 that dimension takes.
BLOCK_SIZE = 16

# The KV pool the step indexes into. A block count, not a measurement: the
# device readings are another task's, and a fixed number is what makes this
# reproducible.
KV_BLOCKS = 64

# What counts as a collective in the inventory. ATOM issues them two ways: as
# aiter custom operators, which carry a registered fake implementation, and
# through `c10d`'s own entry points.
COLLECTIVE_OPS = r"c10d|aiter\.(all_reduce|all_gather|reduce_scatter|gather)"

# Declared device readings -- node 18's MI308X. Nothing here reads a device.
ARCH = "gfx942"
CU_COUNT = 80
CAPABILITY = (9, 4)
TOTAL_MEMORY_BYTES = 192 * (1 << 30)


# ── what the capture is expected to find ──────────────────────────────────
# Every figure below was measured by running this file. The distinct-operator
# counts are asserted and the totals are not: a total moves with any change to
# ATOM's forward -- a fused kernel, one more view -- while the distinct set
# moves when the kinds of work change, which is the thing worth holding.

TP1_DISTINCT_OPS = 33
TP2_DISTINCT_OPS = 38

# What the width changes, by name. It swaps the embedding for its
# vocab-parallel form, adds the collectives, and adds the `movedim` the
# non-custom gather uses to rearrange what it gathered. Nothing else.
TP1_ONLY_OPS = ("aten.embedding.default",)
TP2_ONLY_OPS = (
    "_c10d_functional.all_gather_into_tensor.default",
    "_c10d_functional.broadcast.default",
    "_c10d_functional.wait_tensor.default",
    "aiter.all_reduce_.default",
    "aiter.masked_embedding.default",
    "aten.movedim.int",
)

# The lines a TP2 decode step issues a collective from. Named, because a
# collective's operator says what ran and only its call site says which of
# ATOM's communication paths ran it.
ROW_PARALLEL = (
    "atom/model_ops/communication_op.py:58 in tensor_model_parallel_all_reduce"
)
VOCAB_EMBEDDING = "atom/model_ops/embed_head.py:175 in forward"
VOCAB_LM_HEAD = "atom/model_ops/embed_head.py:257 in forward"
SAMPLER = "atom/model_engine/model_runner.py:3138 in postprocess"

# Where a free symbol stops being free, each as the value it takes and the
# innermost ATOM frames it takes it through. Three, not two, and in this order:
# a symbol is solved by whichever line reaches it first, so which sites are
# visible is a property of the order and not only of the code. Sites one and
# two are what a plain symbolic pass records; site three is what site one hides,
# and it shows up in the pass that simulates site one closed.
SITE_ONE = (
    "2",
    ("atom/model_ops/attentions/aiter_attention.py:1115 in prepare_decode",),
)
SITE_TWO = (
    "16384",
    (
        "atom/model_ops/attentions/aiter_attention.py:1142 in prepare_decode",
        "atom/utils/__init__.py:725 in copy_to_gpu",
    ),
)
# `int(t.shape[0])` in the `_rows` helper of `assert_shape_contract` -- an ATOM
# assertion helper, reached with the bound still free once `:1115` no longer
# takes `__index__` of it.
SITE_THREE = (
    "2",
    (
        "atom/utils/forward_context.py:437 in assert_shape_contract",
        "atom/utils/forward_context.py:424 in _rows",
    ),
)

# `CpuGpuBuffer.__init__` as it is on this tree, and how many buffers one
# runner builds through it. ATOM's own `__init__` executes under this capture,
# so these say which constructor answered and how often -- the half of the
# second site's pin that no specialisation site covers, because a repair inside
# `__init__` changes what it allocates rather than where a symbol is solved.
# Measured: 19 buffers, each one host allocation and one numpy view.
BUFFER_INIT = "atom/utils/__init__.py:700"
BUFFER_COUNT = 19

# ---------------------------------------------------------------------------
# the capture driver -- everything below runs in the subprocess


def _declare_cuda():
    """Stub the `torch.cuda` names ATOM reads, and report what was stubbed.

    Two groups, because they are needed at different moments and a count that
    merges them hides which. The `import` group must be in place before ATOM is
    imported at all: aiter's Triton attention configs read
    `get_device_properties` at module scope. `is_available` has to report True
    -- ATOM branches on the device throughout, and on False takes paths nobody
    runs. `FakeTensorMode` needs the opposite answer; `_driverless_mode` says
    why, and how both are told what they need.

    `mem_get_info` is a reading, not an ordering primitive, and it is the one
    that must not be zero: ATOM sizes the KV budget from it, so a `(0, 0)` here
    is a budget of exactly zero, precise and fictional. Streams and events are
    not readings at all -- a capture has one order by construction -- so a null
    object is the whole of their content.
    """

    class _Props:
        multi_processor_count = CU_COUNT
        gcnArchName = ARCH
        name = ARCH
        major, minor = CAPABILITY
        total_memory = TOTAL_MEMORY_BYTES
        warp_size = 64
        max_threads_per_multi_processor = 2048
        L2_cache_size = 8 << 20
        regs_per_multiprocessor = 65536
        shared_memory_per_block = 65536

    class _Event:
        def record(self, *a, **k):
            return None

        def wait(self, *a, **k):
            return None

        def synchronize(self, *a, **k):
            return None

        def query(self, *a, **k):
            return True

        def elapsed_time(self, *a, **k):
            # Not 0.0: the caller appends this to a list of step durations, so a
            # zero is a confident, precise, entirely fictional measurement. A
            # capture runs no kernel and has no elapsed time to report.
            raise RuntimeError(
                "elapsed_time was read under a capture, which runs no kernel "
                "and will not invent a duration"
            )

    class _Stream:
        def __init__(self, *a, **k):
            self.cuda_stream = 0

        def record_event(self, *a, **k):
            return _Event()

        def wait_event(self, *a, **k):
            return None

        def wait_stream(self, *a, **k):
            return None

        def synchronize(self, *a, **k):
            return None

        def query(self, *a, **k):
            return True

    for_import = {
        "is_available": lambda: True,
        "device_count": lambda: 1,
        "_lazy_init": lambda *a, **k: None,
        "get_rng_state": lambda *a, **k: torch.zeros(16, dtype=torch.uint8),
        "set_rng_state": lambda *a, **k: None,
        "get_device_properties": lambda *a, **k: _Props(),
        "current_device": lambda: 0,
        "get_device_capability": lambda *a, **k: CAPABILITY,
    }
    for_runner = {
        # `Event` stays a type, not a lambda: `ModelRunner` evaluates
        # `torch.cuda.Event | None` in a class body at import, and
        # `function | None` is a TypeError.
        "Stream": _Stream,
        "Event": _Event,
        "current_stream": lambda *a, **k: _Stream(),
        "default_stream": lambda *a, **k: _Stream(),
        "set_device": lambda *a, **k: None,
        "synchronize": lambda *a, **k: None,
        "empty_cache": lambda *a, **k: None,
        "reset_peak_memory_stats": lambda *a, **k: None,
        "memory_stats": lambda *a, **k: {
            "allocated_bytes.all.current": 0,
            "allocated_bytes.all.peak": 0,
            "reserved_bytes.all.current": 0,
        },
        "mem_get_info": lambda *a, **k: (TOTAL_MEMORY_BYTES, TOTAL_MEMORY_BYTES),
        "max_memory_allocated": lambda *a, **k: 0,
        "memory_allocated": lambda *a, **k: 0,
        "memory_reserved": lambda *a, **k: 0,
        "stream": lambda s: contextlib.nullcontext(),
    }
    for group in (for_import, for_runner):
        for name, value in group.items():
            setattr(torch.cuda, name, value)
    return {"import": sorted(for_import), "model_runner": sorted(for_runner)}


def _decline_initialisers():
    """Decline the in-place RNG fills `nn.Module.reset_parameters` performs.

    A layer built on a CUDA device initialises its parameters through
    `Tensor.uniform_` / `Tensor.normal_`, and under the mode those reach a
    decomposition that asks the device for a generator -- `HIP error: no
    ROCm-capable device is detected`, from inside `nn.Conv3d.__init__` in the
    vision tower.

    Declining them costs nothing that is measured here. This capture reads no
    checkpoint, so every parameter value is arbitrary before it is arbitrary;
    what the inventory records is shape, dtype and device, and the fill changes
    none of the three. Returning `self` keeps the initialiser's contract.
    """
    filled = []
    for name in ("uniform_", "normal_"):
        filled.append(name)
        setattr(torch.Tensor, name, lambda self, *a, **k: self)
    return filled


def _driverless_mode(shape_env):
    """`FakeTensorMode` told the device is absent, while ATOM is told it is there.

    These two readers of `torch.cuda.is_available()` need opposite answers, and
    the same function cannot give both.

    ATOM needs True. ATOM branches on the device throughout, registers its
    operators at `dispatch_key="CUDA"`, and on False takes paths nobody runs.

    `FakeTensorMode` needs False, on a host with no driver at all. That one flag
    gates three accommodations it makes for an absent device, and every one of
    them is load-bearing here:

    * `_only_lift_cpu_tensors(True)`, which keeps `torch.tensor` on the host and
      moves it afterwards. `torch.tensor` reads its device eagerly, in C++,
      below anything the mode can intercept, so ATOM's own
      `torch.tensor([])` under a CUDA default device is otherwise
      `No HIP GPUs are available`;
    * `_ensureCUDADeviceGuardSet()`, which swaps the CUDA device guard for a
      no-op so CUDA kernels can be traced at all;
    * skipping constant propagation across a device conversion. Fake tensors
      small enough to carry their constant otherwise have their next operator
      run **for real** on the destination device.

    Overriding the property says the second without changing the first. The
    original measurement of this stub set was taken against a host whose driver
    was wedged rather than absent, where reporting False hangs inside
    `_ensureCUDADeviceGuardSet` and reporting True completes; with no driver at
    all the dependency runs the other way.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    class _DriverlessFakeTensorMode(FakeTensorMode):
        @property
        def avoid_device_init(self):
            return True

    return _DriverlessFakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)


def _decline_custom_all_gather():
    """Select the non-custom vocab-parallel gather, and say so.

    ATOM's default sends `embed_head.py`'s vocab-parallel gather down a custom
    path that reads `device_communicator.ca_comm` -- the device communicator
    this capture declines, because it opens a rendezvous that waits for ranks
    that do not exist. Left on, a TP2 forward dies with `'NoneType' object has
    no attribute 'ca_comm'` after roughly 2,500 operators; the non-custom form
    gathers the same shapes and is recorded.

    This is a configuration of the capture, not a stub, and belongs in the
    record beside the device readings: it is the whole difference between the
    run that refuses part-way and the one that completes.
    """
    os.environ["ATOM_USE_CUSTOM_ALL_GATHER"] = "0"
    return "ATOM_USE_CUSTOM_ALL_GATHER=0"


def _decline_context_priming():
    """Decline the one real allocation the fake-tensor machinery makes.

    `FakeTensor.__new__` primes a CUDA context for any device-tagged fake it
    builds -- a real `torch.zeros(1, device=...)`, guarded only by
    `torch.cuda.is_available()`, which the stubs above must report True. It
    exists so that backward through a CUDA fake does not error; nothing here
    runs backward, and there is no context to prime. Left in place it is the
    first thing the capture does and the first thing that fails.
    """
    from torch._subclasses import fake_tensor

    fake_tensor.init_gpu_context = lambda device: None
    return "fake_tensor.init_gpu_context"


def _declare_arch(tmpdir):
    """Declare the GPU architecture to the two things that go looking for it.

    aiter resolves it twice and only one of those reads `GPU_ARCHS`: the other,
    `get_gfx_runtime`, shells out to `rocminfo` unconditionally and raises when
    there is no `/dev/kfd`. A `rocminfo` of our own on PATH answers it. Triton's
    side is the same fact through a different door -- `triton.runtime.driver`
    asks the live device for its target, and aiter falls back to a jax import
    that is not installed when that raises.

    Both are the arch, configured. Neither is read from a device here, and the
    proxy answers nothing but the target: any other attribute raises rather than
    quietly standing in for a driver.
    """
    import triton
    from triton.backends.compiler import GPUTarget

    binpath = pathlib.Path(tmpdir) / "declared-bin"
    binpath.mkdir(parents=True, exist_ok=True)
    rocminfo = binpath / "rocminfo"
    rocminfo.write_text(f'#!/bin/sh\necho "  Name:                    {ARCH}"\n')
    rocminfo.chmod(0o755)
    os.environ["PATH"] = f"{binpath}:{os.environ.get('PATH', '')}"
    os.environ["GPU_ARCHS"] = ARCH

    class _DeclaredTarget:
        def get_current_target(self):
            return GPUTarget("hip", ARCH, 64)

        def __getattr__(self, name):
            raise RuntimeError(
                f"a device-free capture has no Triton driver; {name} was read"
            )

    triton.runtime.driver.set_active(_DeclaredTarget())
    return {"rocminfo": str(rocminfo), "triton_target": ARCH}


def _functional_collectives():
    """Route `c10d`'s legacy in-place collectives to their functional forms.

    ATOM's collective at TP>1 is aiter's `all_reduce_`, a registered custom
    operator with a registered fake implementation, and it needs nothing here.
    A handful of call sites reach `torch.distributed`'s own entry points
    instead, and on this stack those `c10d::*` operators carry a backend kernel
    and **neither a Meta nor a CompositeExplicitAutograd one**, so under the
    mode they raise `NotImplementedError` rather than producing a meta
    operation. Measured to be general rather than particular to one of them:
    `c10d::barrier` and `c10d::_allgather_base_` fail the same way.

    The functional forms do carry a kernel the mode can run and give the same
    shapes, so the collective is still recorded, with its real shapes, having
    communicated nothing. The output tensor is filled by the caller's own
    contract, so the shapes a reader sees are the shapes the legacy call would
    have produced.

    The one arrangement the substitution does change is recorded here rather
    than inferred anywhere. ATOM hands `all_gather_into_tensor` an output
    buffer of `(world_size,) + input_size`; the functional form concatenates
    along dim 0 into `(world_size * rows,) + rest` and the shim reshapes. Both
    are measured off the live call, so a reader of the record never has to
    guess which of the two a shape belongs to.
    """
    import torch.distributed as dist
    from torch.distributed._functional_collectives import (
        all_gather_tensor,
    )
    from torch.distributed._functional_collectives import (
        broadcast as functional_broadcast,
    )

    buffers = []

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        gathered = all_gather_tensor(input, 0, group or dist.group.WORLD)
        # aiter stages the gather into a `(world_size,) + input_size` buffer and
        # reshapes afterwards; the functional form concatenates along dim 0.
        # Same elements, same order, different arrangement.
        buffers.append(
            {
                "op": "all_gather_into_tensor",
                "input": [str(dim) for dim in input.shape],
                "atom_output": [str(dim) for dim in output.shape],
                "functional_output": [str(dim) for dim in gathered.shape],
            }
        )
        output.copy_(gathered.reshape(output.shape))

    def broadcast(tensor, src=0, group=None, async_op=False):
        tensor.copy_(functional_broadcast(tensor, src, group or dist.group.WORLD))

    dist.all_gather_into_tensor = all_gather_into_tensor
    dist.broadcast = broadcast
    return ["all_gather_into_tensor", "broadcast"], buffers


def _build_group(tp):
    """A process group of width `tp` inside one process, transport declined.

    The width is real: `get_tp_group().world_size` is `tp` because the group has
    `tp` ranks, so every shard size in the tree is the one a `tp`-way deployment
    computes, and every collective ATOM issues is dispatched and recorded. What
    one process cannot build is the transport -- a device communicator opens a
    rendezvous that waits for ranks that do not exist, as does the message-queue
    broadcaster, and a gloo sub-group's own rendezvous does the same. All three
    are declined here; none of them carries a shape.

    The collective ATOM issues at TP>1 does not need any of them. aiter's
    `all_reduce_` is a registered custom operator with a registered fake
    implementation, so under the mode the fake answers and the body that wants a
    communicator is unreachable: the operator is recorded with its real shapes
    having allocated and communicated nothing.
    """
    import torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeStore

    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(tp)
    os.environ["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp))
    dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=tp)

    from aiter.dist import parallel_state

    original_group = parallel_state.init_model_parallel_group
    original_new_group = dist.new_group

    def decline_transport(*args, **kwargs):
        kwargs["use_device_communicator"] = False
        kwargs["use_message_queue_broadcaster"] = False
        return original_group(*args, **kwargs)

    def decline_backend(ranks=None, **kwargs):
        # Every sub-group, including the `gloo` one the coordinator builds for
        # host-side coordination, takes the same peerless backend as the world.
        kwargs["backend"] = "fake"
        kwargs.pop("pg_options", None)
        return original_new_group(ranks, **kwargs)

    # The barrier in `allocate_kv_cache` is transport too, and it is the one
    # that has no fake implementation to answer with: `c10d::barrier` carries a
    # backend kernel and neither a Meta nor a CompositeExplicitAutograd one, so
    # under the mode it raises rather than producing a meta operation. It
    # carries no shape and moves no bytes; there is nothing for an inventory to
    # record and nobody to wait for.
    dist.barrier = lambda *args, **kwargs: None

    parallel_state.init_model_parallel_group = decline_transport
    dist.new_group = decline_backend
    try:
        parallel_state.init_distributed_environment(
            world_size=tp, rank=0, backend="fake", local_rank=0
        )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp, backend="fake"
        )
    finally:
        parallel_state.init_model_parallel_group = original_group
        dist.new_group = original_new_group
    functional, gather_buffers = _functional_collectives()
    return (
        parallel_state.get_tp_group().world_size,
        {
            "declined": [
                "new_group backend",
                "device_communicator",
                "message_queue_broadcaster",
                "barrier",
            ],
            "functional": functional,
        },
        gather_buffers,
    )


def _watch_simulated_tp(tree_root):
    """A sentinel on `apply_simulated_tp`, in place of a hard-coded `False`.

    That `apply_simulated_tp` never runs is the hardest claim this file makes:
    a TP>1 inventory taken through it both erases and fabricates, so a record
    that came through it is not a TP>1 record at all. It used to be carried by
    a literal written into the record and asserted against itself, which
    cannot fail and cannot go stale -- the defect principle 8 exists for.

    The sentinel records every call, with the ATOM frames that made it, and
    does not call through: a run in which it fires produces a record naming
    the site rather than a number nobody can check. Both bindings are taken,
    because `model_runner` imports the name rather than the module.
    """
    from atom.distributed import simulated_tp
    from atom.model_engine import model_runner

    calls = []

    def sentinel(config):
        calls.append({"frames": _atom_frames(tree_root)})

    simulated_tp.apply_simulated_tp = sentinel
    model_runner.apply_simulated_tp = sentinel
    return calls


def _hint(value):
    """A `SymInt`'s trace-time hint, read without solving it.

    `int(sym)` and `sym.__index__()` both *specialise*: they record the hint as
    the symbol's value and the symbol stops being free. `sym.node.hint` reads
    the same number and records nothing, which is the whole difference between
    standing in for a repair and being the thing a repair is about.
    """
    if isinstance(value, torch.SymInt):
        hint = value.node.hint
        if hint is None:
            raise RuntimeError(f"{value} carries no hint to read")
        return int(hint)
    return value


def _hinted(key):
    """The same subscript with every `SymInt` in it replaced by its hint."""
    if isinstance(key, tuple):
        return tuple(_hinted(item) for item in key)
    if isinstance(key, slice):
        return slice(_hint(key.start), _hint(key.stop), _hint(key.step))
    return _hint(key)


class _HintSlicedView(numpy.ndarray):
    """A staging buffer's numpy view that does not solve a `SymInt` bound.

    This is the **simulated site-one repair**, and it is a probe rather than
    part of any capture. `prepare_decode` fills each staging buffer's numpy
    view with the row count it was handed -- `var["slot_mapping"].np[:running_
    tokens]` -- and numpy takes `__index__` of whatever it is given, which
    solves the symbol. A staging buffer that reads the bound's hint instead
    leaves it free. That is the shape of one of the two repair routes the
    design record carries for this, applied from outside ATOM rather than by
    editing it.

    It exists to measure what happens *next*, because the answer is not what
    the two-site account assumed: the bound is not closed by repairing the
    site that solves it first. It is solved somewhere else, and that somewhere
    is in a third module.

    A `numpy.ndarray` subclass rather than a wrapper, because ATOM's own
    `pack_rows` takes `memoryview()` of the staging view -- a delegating
    wrapper is `a bytes-like object is required` there, and swapping the
    buffer protocol out is a change to the thing being measured rather than
    to the one line under test. `.view(_HintSlicedView)` shares the storage.
    """

    def __setitem__(self, key, value):
        super().__setitem__(_hinted(key), value)

    def __getitem__(self, key):
        return super().__getitem__(_hinted(key))


@contextlib.contextmanager
def _staged_allocators(fake_mode, symbolic, observed):
    """The three primitives `CpuGpuBuffer.__init__` needs staged, and no more.

    Substituted for the duration of one `__init__` body, so that what runs is
    ATOM's own body. The first version of this file replaced the method
    wholesale instead. The buffer built, but every line of ATOM's `__init__`
    was then unreachable: a `raise` as its first statement changed nothing
    anywhere in this file, and the repair route that makes `CpuGpuBuffer`
    symbolic -- an allocation change, and the
    allocation is `__init__` -- was the one route this test could not see.

    * `torch.zeros` for the host side runs outside the mode, so `self.cpu` is
      a real, numpy-backed, concrete tensor, which is the concrete half of
      the straddle. `pin_memory` is dropped: pinning is a real host allocation
      through the driver (`hipHostMalloc`), a property of the transfer rather
      than of the shape, and nothing traced here can observe it.
    * `torch.zeros_like` for the device side is substituted **only** in the
      symbolic pass. `static_shapes=False` is what puts a free symbol on each
      dimension; a tensor allocated inside the mode is already fake and
      *static*, with plain `int` shapes and no sign that anything was lost.
      The symbolic side is converted from a template that is then dropped, not
      from `self.cpu`: converting `self.cpu` memoises it as a symbolic fake,
      and the first operator ATOM performs on the CPU side directly then asks
      the converter for a concrete view of a tensor it has already given a
      symbolic meta storage -- `Trying to resize storage that is not
      resizable`. In the concrete pass ATOM's own `zeros_like` runs unaltered.
    * `Tensor.numpy` runs outside the mode. A `.numpy()` taken while the mode
      is active leaves the real storage marked not resizable, and the next
      operator on the CPU side then fails inside the converter with a
      deprecation warning about reading a FakeTensor data pointer as the only
      clue. The numpy view is a host alias, not an operation worth tracing.

    Each substitution counts its calls, and the counts go in the record. They
    are what says ATOM's body ran, and they move if its allocations are added
    to, removed or re-routed -- which is the other half of closing the hole.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    real_zeros = torch.zeros
    real_zeros_like = torch.zeros_like
    real_numpy = torch.Tensor.numpy
    reentered: list[bool] = []

    def zeros(*size, **kwargs):
        kwargs.pop("pin_memory", None)
        device = kwargs.get("device")
        if device is not None and torch.device(device).type != "cpu":
            return real_zeros(*size, **kwargs)
        observed["host_allocations"] += 1
        with unset_fake_temporarily():
            return real_zeros(*size, **kwargs)

    def zeros_like(tensor, **kwargs):
        if not symbolic:
            return real_zeros_like(tensor, **kwargs)
        device = kwargs.pop("device", None)
        with unset_fake_temporarily():
            template = real_zeros_like(tensor, device="cpu", **kwargs)
        observed["symbolic_device_allocations"] += 1
        staged = fake_mode.from_tensor(template, static_shapes=False)
        return staged if device is None else staged.to(device)

    def numpy(tensor, *args, **kwargs):
        # One view, counted once. `Tensor.numpy` is dispatched through the
        # torch-function mode `set_default_device` installs, which calls the
        # bound name again with dispatch off -- so an unguarded counter reads
        # two per buffer and the number stops meaning what it says.
        if not reentered:
            observed["numpy_views"] += 1
        reentered.append(True)
        try:
            with unset_fake_temporarily():
                return real_numpy(tensor, *args, **kwargs)
        finally:
            reentered.pop()

    torch.zeros = zeros
    torch.zeros_like = zeros_like
    torch.Tensor.numpy = numpy
    try:
        yield
    finally:
        torch.zeros = real_zeros
        torch.zeros_like = real_zeros_like
        torch.Tensor.numpy = real_numpy


def _stage_buffers(fake_mode, symbolic, tree_root, repair_site_one=False):
    """Run ATOM's own `CpuGpuBuffer.__init__`, staging only what has no device.

    `CpuGpuBuffer.__init__` allocates a CPU tensor, allocates the device side
    `zeros_like` it, and takes `.numpy()` of the first. Under the mode all
    three are faked and `.numpy()` raises -- `.numpy() is not supported for
    tensor subclasses` -- so no runner constructs without something being
    done here. What is done is the three substitutions in `_staged_allocators`
    above; the body between them is ATOM's, executed.

    That the body executes is what this test pins, and the record carries the
    evidence: which `__init__` ran, how many buffers it built, and how many of
    each staged allocation it asked for. `copy_to_gpu` is pinned by the
    specialisation site it produces; `__init__` is pinned by these counts,
    because a repair there changes what it allocates and not where a symbol is
    solved.

    `repair_site_one` wraps each numpy view in `_HintSlicedView` afterwards.
    That is the probe, not the capture: see the class.
    """
    from atom.utils import CpuGpuBuffer

    original = CpuGpuBuffer.__init__
    code = original.__code__
    path = pathlib.Path(code.co_filename).resolve()
    try:
        source = str(path.relative_to(tree_root))
    except ValueError:
        source = str(path)
    observed = {
        "source": f"{source}:{code.co_firstlineno}",
        "constructed": 0,
        "host_allocations": 0,
        "symbolic_device_allocations": 0,
        "numpy_views": 0,
        "hint_sliced_views": 0,
    }

    def staged_init(self, *size, dtype, device, pin_memory=True, with_numpy=True):
        observed["constructed"] += 1
        with _staged_allocators(fake_mode, symbolic, observed):
            original(
                self,
                *size,
                dtype=dtype,
                device=device,
                pin_memory=pin_memory,
                with_numpy=with_numpy,
            )
        if repair_site_one and with_numpy:
            observed["hint_sliced_views"] += 1
            self.np = self.np.view(_HintSlicedView)

    CpuGpuBuffer.__init__ = staged_init
    return original, observed


def _atom_frames(tree_root):
    """The ATOM source frames on the live stack, innermost last.

    A collective's name says which operator ran; only its call site says which
    of ATOM's twenty-odd communication paths issued it, and that is the half
    that goes stale silently when a file is edited.
    """
    frames = []
    for frame in traceback.extract_stack():
        path = pathlib.Path(frame.filename)
        try:
            rel = path.relative_to(tree_root)
        except ValueError:
            continue
        if rel.parts[0] != "atom":
            continue
        frames.append(f"{rel}:{frame.lineno} in {frame.name}")
    return frames


class _Recorder(TorchDispatchMode):
    """Every dispatched operator, with its shapes as the ShapeEnv reports them.

    Shapes are stringified rather than kept as `SymInt`s: an inventory is
    compared across widths and across runs, and a `SymInt` compares by the
    identity of its symbol, which differs between two ShapeEnvs that agree.
    """

    def __init__(self, tree_root, collective_pattern):
        super().__init__()
        self._tree_root = tree_root
        self._collective = re.compile(collective_pattern)
        self.ops = []
        self.collectives = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)
        shapes_in = [self._shape(a) for a in self._tensors(args)]
        shapes_out = [self._shape(o) for o in self._tensors(self._flat(out))]
        self.ops.append((name, shapes_in, shapes_out))
        if self._collective.search(name):
            frames = _atom_frames(self._tree_root)
            self.collectives.append(
                {
                    "op": name,
                    "call_site": frames[-1] if frames else "outside atom",
                    "shapes": shapes_in,
                }
            )
        return out

    def _tensors(self, items):
        return [x for x in items if isinstance(x, torch.Tensor)]

    def _flat(self, value):
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                out.extend(self._flat(item))
            return out
        return [value]

    @staticmethod
    def _shape(tensor):
        return [str(dim) for dim in tensor.shape]

    def distinct_ops(self):
        return sorted({name for name, _, _ in self.ops})

    def non_numeric_ops(self):
        """The operators carrying a shape entry that is not a plain integer."""
        carrying = set()
        for name, shapes_in, shapes_out in self.ops:
            for shape in (*shapes_in, *shapes_out):
                if any(not re.fullmatch(r"-?\d+", dim) for dim in shape):
                    carrying.add(name)
        return sorted(carrying)

    def shape_census(self):
        """`(shape entries recorded, entries that are not a plain integer)`.

        The second number separates a symbolic inventory from a concrete one: if
        every shape is an integer, the inventory describes only the shapes it was
        taken at and can be evaluated nowhere else.
        """
        entries = non_numeric = 0
        for _, shapes_in, shapes_out in self.ops:
            for shape in (*shapes_in, *shapes_out):
                for dim in shape:
                    entries += 1
                    if not re.fullmatch(r"-?\d+", dim):
                        non_numeric += 1
        return entries, non_numeric


class _Specialisations:
    """Where each free symbol stopped being free, and what it became.

    `ShapeEnv._set_replacement` is the single funnel every replacement goes
    through -- its own docstring says to use it rather than assigning into
    `replacements` -- so wrapping it catches each one at the moment it happens,
    with the live stack still standing. Reading `replacements` afterwards gives
    the same symbols and none of the sites, and the site is the claim: a symbol
    solved somewhere else is a different finding about a different line.
    """

    def __init__(self, shape_env, tree_root):
        self.events = []
        self._shape_env = shape_env
        self._tree_root = tree_root
        self._original = type(shape_env)._set_replacement

    def __enter__(self):
        recorder = self

        def watched(shape_env, symbol, target, msg):
            before = shape_env.replacements.get(symbol)
            result = recorder._original(shape_env, symbol, target, msg)
            after = shape_env.replacements.get(symbol)
            if after is not None and after != before:
                recorder.events.append(
                    {
                        "symbol": str(symbol),
                        "value": str(after),
                        "frames": _atom_frames(recorder._tree_root),
                    }
                )
            return result

        type(self._shape_env)._set_replacement = watched
        return self

    def __exit__(self, *exc):
        type(self._shape_env)._set_replacement = self._original
        return False


class _TritonLaunches:
    """Record `@triton.jit` launches and do not launch them -- a diagnostic.

    A raw `kernel[grid](...)` goes straight to the AMD driver. It never enters
    the torch dispatcher, so `TorchDispatchMode` cannot see it and
    `FakeTensorMode` cannot fake it: the launcher asks a `FakeTensor` for its
    `data_ptr` and the first one reached ends the trace. Skipping keeps the
    trace alive long enough to enumerate which kernels a step reaches, which is
    what deciding how to price them needs. It is not a capture: a skipped kernel
    writes nothing, so everything downstream of one reads uninitialised fake
    memory.
    """

    def __init__(self):
        self.launches = {}
        self._original = None

    def __enter__(self):
        from triton.runtime.jit import JITFunction

        self._original = JITFunction.run
        launches = self.launches

        def run(jit_function, *args, **kwargs):
            launches[jit_function.__name__] = launches.get(jit_function.__name__, 0) + 1

        JITFunction.run = run
        return self

    def __exit__(self, *exc):
        from triton.runtime.jit import JITFunction

        JITFunction.run = self._original
        self._original = None
        return False


def _decode_batch():
    """Two sequences of one token each, built the way ATOM builds a dummy step.

    Two, not one: a dimension whose trace-time hint is 1 is silently specialised
    to a constant, and a decode step has exactly one token per sequence, so a
    one-sequence trace yields a fully constant graph with no warning.
    """
    import numpy as np

    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import (
        Sequence,
        SequenceStatus,
        SequenceType,
        new_block_table,
    )

    seqs = {}
    for index in range(DECODE_SEQS):
        seq = Sequence([0], block_size=BLOCK_SIZE, id=index)
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.DECODE
        seq.block_table = new_block_table([index])
        seqs[seq.id] = seq
    return ScheduledBatch(
        seqs=seqs,
        num_scheduled_tokens=np.ones(DECODE_SEQS, dtype=np.int32),
        total_tokens_num=DECODE_SEQS,
        total_tokens_num_decode=DECODE_SEQS,
        total_seqs_num=DECODE_SEQS,
        total_seqs_num_decode=DECODE_SEQS,
        is_dummy_run=True,
    )


def _build_runner(config, fake_mode):
    """ATOM's own `ModelRunner`, constructed inside the mode, reading nothing.

    Two of the base class's own override points do all of the work. The model is
    built -- the real class, from `support_model_arch_dict`, in the model's own
    dtype -- and no checkpoint is read; at ATOM's fp32 default the fake tensors
    would trace kernels real hardware rejects, so the graph would not be the one
    that runs. Warmup is skipped because it runs a forward from inside
    `__init__`, and this capture drives its own.

    The distributed setup is skipped too: the group already exists, built at the
    honest width with its transport declined, and ATOM's own path would rebuild
    it through `init_dist_env` and then reach for `apply_simulated_tp`.
    """
    from atom.model_engine.model_runner import ModelRunner, support_model_arch_dict
    from atom.utils import resolve_obj_by_qualname

    architecture = config.hf_config.architectures[0]
    if architecture not in support_model_arch_dict:
        raise RuntimeError(
            f"{architecture} is not in ATOM's support_model_arch_dict, so this "
            "capture has no model class to build"
        )
    model_class = resolve_obj_by_qualname(support_model_arch_dict[architecture])

    class _CapturedRunner(ModelRunner):
        def _setup_device_and_distributed(self, rank, config):
            self.device = torch.device("cuda:0")

        def _build_and_load_model(self, built_class):
            self.model = built_class(config)
            torch.set_default_device(None)

        def _maybe_warmup(self):
            return

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype)
    try:
        with fake_mode:
            runner = _CapturedRunner(0, config)
    finally:
        torch.set_default_dtype(previous_dtype)
    return architecture, model_class.__name__, runner


def _symbolic_bound(runner, fake_mode):
    """Hand `prepare_decode` its row count as a free symbol instead of an int.

    This is the caller's half of the specialisation. `prepare_decode` takes one
    count per staged buffer, uses it to fill the buffer's numpy view and then
    again as the copy's bound, and anything that needs an `int` takes
    `__index__` of a `SymInt` and gets its hint -- recording the symbol as a
    constant with no error and no warning. It is not a numpy behaviour: a bare
    `__index__()` and a plain list slice do the same.

    The symbol is made from a tensor of exactly the count the caller passed, so
    its hint is that count and the step traced is the step ATOM asked for.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    builder = runner.attn_metadata_builder
    original = builder.build
    injected = []

    def build(batch, running_bs, running_tokens, max_seqlen_q):
        # Outside the mode. A tensor allocated inside it is already fake and
        # *static*, `from_tensor` is then a no-op, and the bound comes back a
        # plain `int` with nothing to say it was meant to be a symbol.
        with unset_fake_temporarily():
            template = torch.zeros(int(running_tokens), dtype=torch.int32)
        bound = fake_mode.from_tensor(template, static_shapes=False).shape[0]
        injected.append({"symbol": str(bound), "hint": int(running_tokens)})
        return original(batch, running_bs, bound, max_seqlen_q)

    builder.build = build
    return injected


def _capture(tp, tmpdir, symbolic, repair_site_one=False):
    """Trace one decode step of the published model at width `tp`.

    Three passes, because they answer different questions and cannot be one
    run. The capture ATOM produces is `symbolic=False`: the staged buffers are
    concrete on both sides, as ATOM builds them, and what comes out is the
    inventory -- the operators, the collectives, and a shape census that is the
    claim about the inventory rather than about the mechanism.

    `symbolic=True` is the probe. It gives every staged buffer's device side a
    free symbol per dimension and hands `prepare_decode` its row count as a
    symbol too, then records where each one stopped being free. Symbols that
    survive are not evidence of anything here: a handful reach `as_strided`,
    `reshape` and `slice` on views that no compute operator consumes.

    `repair_site_one=True` is the same probe with the first of those places
    simulated closed, from outside ATOM (`_HintSlicedView`). It exists because
    the specialisation sites are **ordered**, not independent: closing the one
    that solves the bound first does not close the bound, it moves it. The
    record that pass produces is the source for the third site."""
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    declared_cuda = _declare_cuda()
    declared_arch = _declare_arch(tmpdir)
    declared_priming = _decline_context_priming()
    declared_init = _decline_initialisers()
    declared_gather = _decline_custom_all_gather()

    tree_root = pathlib.Path(__file__).resolve().parents[2]
    model_dir = pathlib.Path(tmpdir) / "published-config"
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = CONFIG_JSON.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CONFIG_SHA256:
        raise RuntimeError(
            f"{CONFIG_JSON.name} is not the published config: sha256 {digest}, "
            f"expected {CONFIG_SHA256} for Qwen/Qwen3.8-27B at {CONFIG_REVISION}"
        )
    (model_dir / "config.json").write_bytes(payload)

    group_width, declared_transport, gather_buffers = _build_group(tp)

    from atom.config import Config, set_current_atom_config

    config = Config(
        model=str(model_dir),
        tensor_parallel_size=tp,
        load_dummy="empty",
        enforce_eager=True,
        kv_cache_block_size=BLOCK_SIZE,
    )
    set_current_atom_config(config)
    if config.tp_world_size != tp:
        raise RuntimeError(
            f"tp_world_size is {config.tp_world_size} at tensor_parallel_size "
            f"{tp}: the width has to be the group's, not a simulated one"
        )

    simulated_tp_calls = _watch_simulated_tp(tree_root)
    shape_env = ShapeEnv()
    fake_mode = _driverless_mode(shape_env)
    original_buffer_init, buffer_init = _stage_buffers(
        fake_mode, symbolic, tree_root, repair_site_one
    )
    try:
        architecture, model_class_name, runner = _build_runner(config, fake_mode)
        bounds = _symbolic_bound(runner, fake_mode) if symbolic else []
        recorder = _Recorder(tree_root, COLLECTIVE_OPS)
        triton = _TritonLaunches()
        batch = _decode_batch()
        with fake_mode, _Specialisations(shape_env, tree_root) as specialised:
            runner.allocate_kv_cache(KV_BLOCKS)
            with torch._C._EnablePythonDispatcher(), triton, recorder:
                runner.forward(batch)
    finally:
        from atom.utils import CpuGpuBuffer

        CpuGpuBuffer.__init__ = original_buffer_init

    import atom

    entries, non_numeric = recorder.shape_census()
    return {
        # Which `atom` package this record came from. A capture that cannot
        # name the tree it traced is not an observation about that tree, and
        # an `atom` resolved from somewhere else fails silently wherever both
        # trees have the symbol.
        "atom_package": atom.__file__,
        "tp": tp,
        "tp_group_world_size": group_width,
        "symbolic_staging": symbolic,
        "site_one_repair_simulated": repair_site_one,
        # Observed, not declared: every call the sentinel saw, with its ATOM
        # frames. Empty is the claim; a non-empty list names the site.
        "apply_simulated_tp_calls": simulated_tp_calls,
        # Which `CpuGpuBuffer.__init__` ran, and what it asked for. ATOM's own
        # body executes here, so a change to it moves one of these counts.
        "buffer_init": buffer_init,
        "diagnostic_inventory": True,
        "model": {
            "config_sha256": digest,
            "config_revision": CONFIG_REVISION,
            "architecture": architecture,
            "model_class": model_class_name,
            "dtype": str(config.torch_dtype),
        },
        "declared": {
            "cuda": declared_cuda,
            "arch": declared_arch,
            "context_priming": declared_priming,
            "initialisers": declared_init,
            "transport": declared_transport,
            "config": declared_gather,
        },
        "ops": len(recorder.ops),
        "distinct_ops": recorder.distinct_ops(),
        "collectives": recorder.collectives,
        # The two arrangements of the vocab gather, both measured off the live
        # call: the buffer ATOM passes, and the one the functional substitute
        # produces before the shim reshapes into it.
        "gather_buffers": gather_buffers,
        "shape_entries": entries,
        "non_numeric_shape_entries": non_numeric,
        "non_numeric_ops": recorder.non_numeric_ops(),
        "injected_bounds": bounds,
        "specialisations": specialised.events,
        "triton_launches": triton.launches,
    }


def main(argv):
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--symbolic", action="store_true")
    parser.add_argument("--repair-site-one", action="store_true")
    args = parser.parse_args(argv)
    if args.repair_site_one and not args.symbolic:
        parser.error("--repair-site-one is a probe on the symbolic pass")
    with tempfile.TemporaryDirectory(prefix="compass-capture-") as tmpdir:
        record = _capture(args.tp, tmpdir, args.symbolic, args.repair_site_one)
    sys.stdout.write(RECORD_MARKER + json.dumps(record) + "\n")
    return 0


# ---------------------------------------------------------------------------
# the tests -- everything below runs under pytest, in the parent process


RECORD_MARKER = "CAPTURE-RECORD "


_RECORDS: dict[tuple, dict] = {}


def capture(tp, symbolic=False, repair_site_one=False):
    """Run this file as a script at width `tp` and read back its record.

    Memoised: six of these would otherwise be nine, and each one builds the
    64-layer module tree.
    """
    import pytest

    key = (tp, symbolic, repair_site_one)
    if key in _RECORDS:
        return _RECORDS[key]
    tree_root = pathlib.Path(__file__).resolve().parents[2]
    argv = [sys.executable, str(pathlib.Path(__file__).resolve()), "--tp", str(tp)]
    if symbolic:
        argv.append("--symbolic")
    if repair_site_one:
        argv.append("--repair-site-one")
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tree_root)},
        cwd=str(tree_root),
    )
    for line in completed.stdout.splitlines():
        if line.startswith(RECORD_MARKER):
            _RECORDS[key] = json.loads(line[len(RECORD_MARKER) :])
            return _RECORDS[key]
    pytest.fail(
        f"the capture at TP{tp} (symbolic={symbolic}, "
        f"repair_site_one={repair_site_one}) produced no record; the "
        f"subprocess exited {completed.returncode}.\n"
        f"--- stderr tail ---\n{completed.stderr[-4000:]}"
    )


def row_parallel_reduces():
    """How many row-parallel reduces one forward owes, from the config alone.

    Derived from the published config's geometry rather than read off the
    inventory, so the count has a source that is not the thing it checks. Every
    weight sharded along its input dimension reduces once per forward through
    `tensor_model_parallel_all_reduce`: each layer's `mlp.down_proj`, each
    full-attention layer's `self_attn.o_proj`, and each linear-attention
    layer's `linear_attn.out_proj`. The vision tower shards none.

    What the expression is sensitive to, stated rather than implied: given the
    assertion below that the two attention kinds account for every layer, the
    sum is `2 x num_hidden_layers` and does **not** depend on how the layers
    divide between them. The reduce per layer is the `mlp.down_proj`; the
    second is one attention output projection whichever kind the layer is. The
    split is asserted because a third layer kind would break the identity, not
    because the total counts it.
    """
    text = json.loads(CONFIG_JSON.read_bytes())["text_config"]
    layer_types = text["layer_types"]
    assert len(layer_types) == text["num_hidden_layers"]
    full = layer_types.count("full_attention")
    linear = layer_types.count("linear_attention")
    assert full + linear == len(layer_types)
    return len(layer_types) + full + linear


def site(event, depth):
    """One specialisation as `(value, the innermost `depth` ATOM frames)`.

    Only the innermost frames are the site. The path that reaches it runs
    through the runner and the eplb wrapper, whose line numbers move for
    reasons that have nothing to do with where a symbol was solved.
    """
    return event["value"], tuple(event["frames"][-depth:])


def test_the_published_config_is_the_one_that_was_published():
    """The fixture is the checkpoint's own file, not a description of it.

    A synthetic config is the thing this tree deleted once already, and the
    difference between one and the published file is invisible in every number
    downstream of it.
    """
    payload = CONFIG_JSON.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == CONFIG_SHA256
    assert json.loads(payload)["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert json.loads(payload)["text_config"]["max_position_embeddings"] == 262144


def test_a_decode_step_traces_at_both_widths():
    """The forward completes at TP1 and TP2 and yields an inventory.

    The assertion is the **distinct**-operator count, not the total. A total
    moves with any change to ATOM's forward -- a fused kernel, one more view --
    and a test that pins one is a test that is edited every time it fails. The
    distinct set moves when the *kinds* of work change, which is the thing worth
    holding.
    """
    tp1 = capture(1)
    tp2 = capture(2)
    tree_root = str(pathlib.Path(__file__).resolve().parents[2])
    for record in (tp1, tp2):
        assert record["atom_package"].startswith(tree_root)
        assert record["ops"] > 0
        assert record["diagnostic_inventory"] is True
        assert record["model"]["architecture"] == "Qwen3_5ForConditionalGeneration"
    assert len(tp1["distinct_ops"]) == TP1_DISTINCT_OPS
    assert len(tp2["distinct_ops"]) == TP2_DISTINCT_OPS
    assert set(tp1["distinct_ops"]) - set(tp2["distinct_ops"]) == set(TP1_ONLY_OPS)
    assert set(tp2["distinct_ops"]) - set(tp1["distinct_ops"]) == set(TP2_ONLY_OPS)


def test_the_width_is_the_group_s_and_nothing_simulated_it():
    """The two halves of "this is an honest TP2", both as measurements.

    Neither is a constant. `tp_group_world_size` is read from
    `get_tp_group().world_size`, so it is what the group has rather than what
    the config asked for -- the one width figure in the record that a
    substitution could not fake. `apply_simulated_tp_calls` is what a sentinel
    installed over both bindings of the function observed; an empty list is the
    claim, and a non-empty one carries the ATOM frames that called it.

    This used to be `assert record["apply_simulated_tp"] is False` against a
    literal `False` written into the record four hundred lines earlier, which
    could not fail and could not go stale. It matters because a TP>1 inventory
    taken through `apply_simulated_tp` both erases and fabricates: 129
    `all_reduce` become the identity and appear nowhere, and one `all_gather`
    becomes six dispatched operators over a half-zeros tensor.
    """
    for record in (capture(1), capture(2), capture(1, symbolic=True)):
        assert record["tp_group_world_size"] == record["tp"]
        assert record["apply_simulated_tp_calls"] == []


def test_atom_s_own_buffer_constructor_is_what_runs():
    """`CpuGpuBuffer.__init__` executes here, and the record says how.

    The pin on the second specialisation site is worth only as much as the
    code it lets run. An earlier version of this file replaced `__init__`
    wholesale, and a `raise` as its first statement then changed nothing
    anywhere in this file -- so the repair route T81 names for site two, a
    symbolic `CpuGpuBuffer`, could have landed and this test would still have
    reported the site unrepaired. It is ATOM's body that runs now, with three
    primitives staged around it, and these counts are what says so.

    The counts are also the pin on the body itself, which no specialisation
    site covers: one host allocation and one numpy view per buffer, and in the
    symbolic pass one device-side allocation per buffer through
    `torch.zeros_like`. A repair that allocates differently moves one of them.
    """
    tree_root = pathlib.Path(__file__).resolve().parents[2]
    concrete = capture(1)
    symbolic = capture(1, symbolic=True)
    for record in (concrete, symbolic):
        init = record["buffer_init"]
        assert init["source"] == BUFFER_INIT
        assert (tree_root / init["source"].split(":")[0]).exists()
        assert init["constructed"] == BUFFER_COUNT
        # ATOM allocates the host side once per buffer and takes one numpy
        # view of it; both run outside the mode. A buffer that stopped doing
        # either, or a twentieth buffer, moves one of these.
        assert init["host_allocations"] == BUFFER_COUNT
        assert init["numpy_views"] == BUFFER_COUNT
    # The device side is staged only in the symbolic pass; in the concrete one
    # ATOM's own `torch.zeros_like` runs unaltered and nothing counts it.
    assert concrete["buffer_init"]["symbolic_device_allocations"] == 0
    symbolic_init = symbolic["buffer_init"]
    assert symbolic_init["symbolic_device_allocations"] == symbolic_init["constructed"]


def test_the_collectives_at_tp2_are_recorded_by_name_and_call_site():
    """TP2 records ATOM's real collectives, at the lines that issue them.

    By name and by site, never by a total. A count is the figure that goes stale
    first and says least: it cannot distinguish one collective moving to a
    different call site from the layer count changing, and it has gone stale
    twice on this project already.

    Six rows, not four: the two `wait_tensor` entries are the functional
    substitution's own operators rather than ATOM's, which is a reason to
    label them and not a reason to leave them out of a decomposition.

    TP1 issues none, which is the control: a group of width 1 shortcuts every
    reduce, so a collective appearing there would mean the recorder was
    counting something else.
    """
    assert capture(1)["collectives"] == []

    by_site = collections.Counter(
        (entry["op"], entry["call_site"]) for entry in capture(2)["collectives"]
    )
    assert dict(by_site) == {
        ("aiter.all_reduce_.default", ROW_PARALLEL): row_parallel_reduces(),
        ("aiter.all_reduce_.default", VOCAB_EMBEDDING): 1,
        ("_c10d_functional.all_gather_into_tensor.default", VOCAB_LM_HEAD): 1,
        ("_c10d_functional.wait_tensor.default", VOCAB_LM_HEAD): 1,
        ("_c10d_functional.broadcast.default", SAMPLER): 1,
        ("_c10d_functional.wait_tensor.default", SAMPLER): 1,
    }
    # The gather shapes say what the width did: one rank's vocab slice arrives
    # as the whole vocabulary.
    gathered = [
        entry["shapes"][0]
        for entry in capture(2)["collectives"]
        if entry["op"].endswith("all_gather_into_tensor.default")
    ]
    assert gathered == [["2", "124160"]]


def test_the_two_arrangements_of_the_vocab_gather_are_both_measured():
    """ATOM's gather buffer and the substitute's, neither one inferred.

    The dispatched operator carries only its input, so the destination shape
    is not in the inventory at all and stating one from the width would be
    arithmetic wearing a measurement's clothes. Both are read off the live
    call instead: ATOM stages the gather into `(world_size,) + input_size`,
    and the functional form the substitution routes to concatenates along
    dim 0 before the shim reshapes into ATOM's buffer.

    Which arrangement a shape belongs to is the distinction, because only the
    first is a fact about ATOM at TP2 and only the second is a fact about this
    capture's substitution.
    """
    assert capture(1)["gather_buffers"] == []
    assert capture(2)["gather_buffers"] == [
        {
            "op": "all_gather_into_tensor",
            "input": ["2", "124160"],
            "atom_output": ["2", "2", "124160"],
            "functional_output": ["4", "124160"],
        }
    ]


def test_the_inventory_is_concrete_at_both_widths():
    """No shape entry in either inventory is anything but a plain integer.

    That is the finding, not a defect in the tracing: the mechanism keeps free
    symbols perfectly well, and the symbolic probe below puts one on every
    staged dimension and watches ATOM's own path solve them. A concrete
    inventory is valid only at the shapes it was taken at, so this is the
    sentence that stops anyone evaluating one anywhere else.
    """
    for tp in (1, 2):
        record = capture(tp)
        assert record["shape_entries"] > 10000
        assert record["non_numeric_shape_entries"] == 0
        assert record["non_numeric_ops"] == []


def test_the_first_two_specialisation_sites_are_where_they_were_measured():
    """The two places a free symbol stops being free first.

    Not two independent sites. They are the first two in an order: the bound
    the caller passes is solved at `:1115` because that is the first line to
    take `__index__` of it, and the buffer's own dimension is solved in
    `copy_to_gpu` because that is the first copy whose slice does not cover
    it. Both are pinned by value and by the frames they happened through, so a
    repair to either one fails here -- which is the point, because the repair
    is the next task and this is how it will be known to have worked. What
    lies behind the first of them is the test below.
    """
    record = capture(1, symbolic=True)
    injected = record["injected_bounds"]
    assert len(injected) == 1 and injected[0]["hint"] == DECODE_SEQS
    # A bound that arrived as a plain `int` would specialise nothing and this
    # test would pass by measuring the absence of a question.
    assert re.fullmatch(r"s\d+", injected[0]["symbol"])
    assert record["site_one_repair_simulated"] is False

    one, two = record["specialisations"]
    assert site(one, 1) == SITE_ONE
    assert site(two, 2) == SITE_TWO
    # Two symbols solved at two places: the buffer's dimension is not the
    # caller's bound, so repairing the bound leaves the buffer as it is.
    assert one["symbol"] == injected[0]["symbol"]
    assert two["symbol"] != one["symbol"]


def test_closing_site_one_moves_the_bound_to_a_third_site():
    """Repairing the first site does not close the bound; it relocates it.

    T81 recorded the two sites as independent and said a symbolic bound closes
    the first and leaves the second as it is. Half of that is true. This is
    the other half, and it is the reason the sites are an order rather than a
    set: with the numpy view reading the bound's hint instead of solving it --
    the simulated site-one repair, applied from outside ATOM -- the bound
    survives `:1115` and is solved fourteen lines later, inside an ATOM
    assertion helper that takes `int()` of a dimension.

    The second site is untouched by the repair, exactly as recorded. The third
    is the one CAP-2 meets the moment its site-one repair lands, and it is in
    a module nothing in the design record named before this test.
    """
    record = capture(1, symbolic=True, repair_site_one=True)
    assert record["site_one_repair_simulated"] is True
    assert record["buffer_init"]["hint_sliced_views"] > 0

    two, three = record["specialisations"]
    # Site two, unchanged by the repair -- which is the half of T81's sentence
    # that holds.
    assert site(two, 2) == SITE_TWO
    assert site(three, 2) == SITE_THREE
    # Still the caller's bound, solved somewhere else: the value is the same
    # and the site is not.
    assert three["value"] == SITE_ONE[0]
    assert record["injected_bounds"][0]["symbol"] == three["symbol"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
