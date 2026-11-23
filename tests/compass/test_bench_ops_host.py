# SPDX-License-Identifier: MIT
"""A tensor argument ATOM holds on the host: the capture states its device, and
`compass measure ops` builds it there rather than binding ATOM's metadata; a
tensor stating none is on the device, as in a graph captured before. The
capture also states a host tensor ATOM did not ask to pin, and a copy with one
refuses by name.
"""

import contextlib
import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy
import torch
from test_bench_ops import _graph, _leaf, _Runner, _t
from test_capture_lift import recorder  # noqa: F401  the recorder fixture
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from atom.compass.bench import ops
from atom.compass.capture import driverless, forward, graph

I32 = "torch.int32"
QWEN3 = pathlib.Path(__file__).with_name("qwen3_0_6b_config.json")
COPY = "aten.copy_.default"


def test_a_copy_s_host_source_states_its_device_and_its_destination_none(recorder):
    with recorder.fake_mode, recorder:
        out = torch.empty(1, 4, dtype=torch.int64, device="cuda")
        staged = torch.zeros(1, 4, dtype=torch.int64)
        recorder.start()
        recorder.on = True
        out.copy_(staged, non_blocking=True)
    [(_, op)] = recorder.items
    args = graph._json(op)["args"]
    assert args["src"].get("device") == "cpu" and "device" not in args["self"]


def test_a_host_source_is_built_apart_and_one_stating_no_device_binds_atom_s():
    stub = ops.Stub(_Runner(), 10_000)
    out, src = _t([2, 8], "activation", dtype=I32), _t([2, 8], "metadata", dtype=I32)
    host = dataclasses.replace(src, host=True)
    copies = [_leaf(COPY, ("self", out), ("src", s)) for s in (host, src)]
    stub.leaves = ops.leaves(_graph(*copies, layers=1)["region"])

    stub._prepare((4, 0))  # ATOM's block tables hold block 4

    names = {sig: name for (sig, _), (name, _) in stub.placed[0][1].items()}
    for sig, (op, instances) in stub.leaves.items():
        built = stub._copy(op, sig, instances, 0)[0]["src"]
        if op["args"]["src"].get("device") == "cpu":
            assert sig not in names and built.device.type == "cpu" and not built.any()
        else:
            assert names[sig] == "block_tables" and set(built.flatten().tolist()) == {4}


def test_a_host_tensor_atom_did_not_pin_states_it(request, tmp_path):
    """A copy's host side ATOM asked to pin, a `CpuGpuBuffer`'s, a `pin_memory`
    allocation or a `pin_memory()` copy, states no `pageable`; one it keeps
    pageable, a `CpuGpuBuffer`'s unpinned or a numpy array's, states it."""
    if "COMPASS_ARCH_DECLARED" not in os.environ:
        # The patches import aiter, which needs an arch declared: once per
        # process, so in a process of its own.
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            + [f"{__file__}::{request.node.name}"],
            env={**os.environ, "COMPASS_ARCH_DECLARED": "1"},
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert done.returncode == 0, done.stdout[-4000:]
        return
    driverless.install(str(tmp_path), "gfx942")
    from atom.utils import CpuGpuBuffer

    capture = forward._Recorder({}, driverless.driverless_mode(ShapeEnv()))
    with forward._patches(capture), capture.fake_mode, capture:
        buffers = [
            CpuGpuBuffer(4, dtype=torch.int64, device="cuda", pin_memory=pin)
            for pin in (True, False)
        ]
        allocated = torch.zeros(4, dtype=torch.int64, pin_memory=True)
        array = numpy.arange(4)
        capture.start()
        capture.on = True
        for buffer in buffers:
            buffer.copy_to_gpu(2)
        buffers[0].gpu.copy_(allocated)
        torch.from_numpy(array).pin_memory().to("cuda")
        torch.from_numpy(array).to("cuda")
    held = [
        spec
        for _, op in capture.items
        for spec in graph._json(op)["args"].values()
        if isinstance(spec, dict) and spec.get("device") == "cpu"
    ]
    assert [s.get("pageable", False) for s in held] == [
        False,
        True,
        False,
        False,
        True,
    ]


def test_a_copy_with_host_memory_atom_keeps_pageable_refuses_by_name():
    """A copy from pageable host memory refuses, naming its argument; one from
    pinned memory is timed, and so is a 0-dim pageable value another operator
    reads as a number."""
    stub = ops.Stub(_Runner(), 10_000)
    out = _t([2, 8], "activation", dtype=I32)
    pinned = dataclasses.replace(_t([2, 8], "metadata", dtype=I32), host=True)
    pageable = dataclasses.replace(pinned, pageable=True)
    scalar = dataclasses.replace(
        _t([], "metadata", dtype=I32), host=True, pageable=True
    )
    leaves = [_leaf(COPY, ("self", out), ("src", s)) for s in (pinned, pageable)]
    leaves.append(_leaf("aten.fill_.Tensor", ("self", out), ("value", scalar)))
    stub.leaves = ops.leaves(_graph(*leaves, layers=1)["region"])
    stub._prepare((4, 0))

    got = {}
    with (
        patch.object(ops, "_function", return_value=lambda **kwargs: None),
        patch.object(ops, "_kernels", return_value=(["k"], [None], 1.0)),
        patch.object(ops, "_replay", return_value=[1.0] * ops.REPEATS),
    ):
        for sig, (op, instances) in stub.leaves.items():
            try:
                got[sig] = stub._time(op, sig, instances, 1)["kernels"]
            except ValueError as error:
                got[sig] = str(error)
    assert list(got.values()) == [
        ["k"],
        "a copy with pageable host memory, src, which ATOM allocates and no CUDA "
        "graph captures",
        ["k"],
    ]


def test_a_tensor_pinned_as_the_runner_is_built_states_no_pageable(request, tmp_path):
    """A `pin_memory=True` host tensor Qwen3-0.6B's model allocates in its
    constructor, or its runner as it sizes its pools, states no `pageable`
    where a captured step copies it."""
    if "COMPASS_ARCH_DECLARED" not in os.environ:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            + ["--show-capture=no", f"{__file__}::{request.node.name}"],
            env={**os.environ, "COMPASS_ARCH_DECLARED": "1"},
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert done.returncode == 0, done.stdout[-4000:]
        return
    (tmp_path / "model").mkdir()
    shutil.copy(QWEN3, tmp_path / "model" / "config.json")
    tracer = forward.Tracer(str(tmp_path / "model"), 1, str(tmp_path))
    from atom.model_engine.model_runner import ModelRunner
    from atom.models.qwen3 import Qwen3ForCausalLM as Model

    held, plan = [], ModelRunner.get_num_blocks
    init, step = Model.__init__, Model.forward

    def pinned(out):
        held.append(torch.zeros(3, dtype=torch.int16, device="cpu", pin_memory=True))
        return out

    def copied(model, *args, **kwargs):
        for host in held:
            torch.empty(3, dtype=torch.int16, device="cuda").copy_(host)
        return step(model, *args, **kwargs)

    with (
        patch.object(Model, "__init__", lambda *a, **k: pinned(init(*a, **k))),
        patch.object(ModelRunner, "get_num_blocks", lambda r: pinned(plan(r))),
        patch.object(Model, "forward", copied),
        contextlib.closing(tracer),
    ):
        list(tracer.trace([forward.Shape("decode", 1, 1)]))
    copies = [op for _, op in tracer.recorder.items if getattr(op, "name", 0) == COPY]
    sources = [graph._json(op)["args"]["src"] for op in copies]
    staged = [s for s in sources if s["dtype"] == "torch.int16"]
    assert [(s.get("device"), s.get("pageable")) for s in staged] == [("cpu", None)] * 2


def _apart(request) -> bool:
    """Whether this process runs the test's body: the patches import aiter,
    which needs an arch declared, once per process, so the test runs again in
    a process of its own, which does."""
    if "COMPASS_ARCH_DECLARED" in os.environ:
        return True
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        + ["--show-capture=no", f"{__file__}::{request.node.name}"],
        env={**os.environ, "COMPASS_ARCH_DECLARED": "1"},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-4000:]
    return False


def test_a_host_tensor_built_over_pinned_bytes_states_no_pageable(request, tmp_path):
    """`torch.from_numpy` of a pinned `CpuGpuBuffer`'s numpy view, and a
    `torch.tensor` or `torch.arange` with `pin_memory=True`, a `torch.tensor`
    built for the device among them, each built and copied in a step, state no
    `pageable`; `torch.from_numpy` of an unpinned buffer's view or of an array
    numpy allocated still does."""
    if not _apart(request):
        return
    driverless.install(str(tmp_path), "gfx942")
    from atom.utils import CpuGpuBuffer

    capture = forward._Recorder({}, driverless.driverless_mode(ShapeEnv()))
    with forward._patches(capture), capture.fake_mode, capture:
        pinned, pageable = (
            CpuGpuBuffer(4, dtype=torch.int64, device="cuda", pin_memory=pin)
            for pin in (True, False)
        )
        capture.start()
        capture.on = True
        for host in (
            torch.from_numpy(pinned.np[:2]),
            torch.tensor([1, 2], pin_memory=True),
            torch.tensor([1, 2], device="cuda", pin_memory=True),  # copied here
            torch.arange(2, pin_memory=True),
            torch.from_numpy(pageable.np[:2]),
            torch.from_numpy(numpy.arange(2)),
        ):
            host.to("cuda")
    held = [
        spec
        for _, op in capture.items
        for spec in graph._json(op)["args"].values()
        if isinstance(spec, dict) and spec.get("device") == "cpu"
    ]
    assert [s.get("pageable", False) for s in held] == [False] * 4 + [True] * 2


def test_a_pinned_tensor_or_arange_records_what_torch_records(request, tmp_path):
    """A `torch.tensor` or `torch.arange` with `pin_memory=True`, built in a
    step and copied to the device, lands where the same call without it does
    and records its ops, each host source pinned where torch pins it: a
    `torch.tensor` for a device by index or for a default device without one,
    and one that requires grad; not an `arange` for a device, which torch
    builds there, nor a pageable `out`, beside which torch ignores the pin."""
    if not _apart(request):
        return
    driverless.install(str(tmp_path), "gfx942")
    capture = forward._Recorder({}, driverless.driverless_mode(ShapeEnv()))

    def defaulted(**pin):
        with torch.device("cuda"):
            return torch.tensor([1, 2], **pin)

    def step(build, **pin):
        capture.start()
        capture.on = True
        out = build(**pin)
        out.to("cuda")
        capture.on = False
        ops = [graph._json(op) for _, op in capture.items]
        marks = [  # each host source's, taken off its op
            spec.pop("pageable", False)
            for op in ops
            for spec in op["args"].values()
            if isinstance(spec, dict) and spec.get("device") == "cpu"
        ]
        return (str(out.device), out.is_leaf, out.requires_grad, ops), marks

    with forward._patches(capture), capture.fake_mode, capture:
        buf = torch.zeros(2, dtype=torch.int64, device="cpu")
        builds = (
            lambda **pin: torch.tensor([1.0], device=0, requires_grad=True, **pin),
            defaulted,
            lambda **pin: torch.arange(2, device="cuda", **pin),
            lambda **pin: torch.arange(2, out=buf, **pin),
        )
        steps = [(step(b), step(b, pin_memory=True)) for b in builds]
    assert [held for (held, _), _ in steps] == [held for _, (held, _) in steps]
    assert [marks for _, (_, marks) in steps] == [[False], [False], [], [True]]
