# SPDX-License-Identifier: MIT
"""A tensor argument ATOM holds on the host: the capture states its device, and
`compass measure ops` builds it there rather than binding ATOM's metadata; a
tensor stating none is on the device, as in a graph captured before. The
capture also states a host tensor ATOM did not ask to pin, and a copy with one
refuses by name.
"""

import dataclasses
import os
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
