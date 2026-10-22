# SPDX-License-Identifier: MIT
"""`compass measure ops` calls a leaf that takes an object, as the capture
recorded it: a runner attribute by name, any other object as pickle reduces it.
"""

import importlib
import json
import os
import pathlib
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from test_bench_ops import _graph, _leaf, _Runner, _t
from test_capture_v4_pro import DPA_TP8, SHAPES, capture

from atom.compass import ir
from atom.compass.bench import ops
from atom.compass.capture import driverless
from atom.compass.capture.forward import decode, prefill
from atom.compass.ir.nodes import Reduced
from atom.utils import forward_context as fctx

I32 = "torch.int32"
V4 = "atom.model_ops.v4_kernels."
#: The variable naming the file of the leaf a test's own process calls.
FOUND = "COMPASS_TEST_FOUND"


def test_a_runner_attribute_is_the_runner_s_own_and_an_object_is_rebuilt():
    runner = _Runner()
    runner.pool = object()
    rows = _t([2, 8], "metadata", dtype=I32)
    held = Reduced(
        "types:SimpleNamespace",
        (("rows", rows), ("role", Reduced("atom.compass.ir.nodes:Role", "kv"))),
    )
    leaf = _leaf(
        "aten.nowhere.default",
        ("pool", ir.ModuleRef("runner.pool")),
        ("held", held),
        ("by_ratio", Reduced("builtins:dict", ((4, rows),))),
    )
    stub = ops.Stub(runner, 10_000)
    stub.leaves = ops.leaves(_graph(leaf, layers=1)["region"])
    [(sig, (op, instances))] = stub.leaves.items()
    stub._prepare((3, 0))

    kwargs, _ = stub._copy(op, sig, instances, 0)

    assert kwargs["pool"] is runner.pool
    assert type(kwargs["held"]) is types.SimpleNamespace
    assert kwargs["held"].role is ir.Role.KV
    # An object's tensor is a view of live metadata, as any other argument's.
    assert set(kwargs["held"].rows.flatten().tolist()) == {3}
    assert set(kwargs["by_ratio"]) == {4}
    assert set(kwargs["by_ratio"][4].flatten().tolist()) == {3}


class _Kernel:
    """A Triton kernel that records each launch's grid and keyword arguments
    and runs nothing."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.launches.append((grid, kwargs))


@pytest.mark.parametrize(
    "shape, context, leaf",
    [
        (decode(2), 1024, "paged_decode_indices:write_v4_paged_decode_indices"),
        (prefill(512, 2), None, "paged_prefill_indices:write_v4_paged_prefill_indices"),
    ],
    ids=["decode", "prefill"],
)
def test_v4_pro_s_index_writing_leaves_rebuild_their_arguments_and_launch(
    shape, context, leaf, monkeypatch, request, tmp_path, tmp_path_factory
):
    found = tmp_path / "found.json"
    if "COMPASS_ARCH_DECLARED" not in os.environ:
        # The leaves import aiter, which needs an arch declared: once per
        # process, so in a process of its own, on the leaf this one captured.
        graph, _ = capture(tmp_path_factory, SHAPES + DPA_TP8)[str(shape)]
        leaves = ops.leaves(graph["region"]).items()
        found.write_text(
            json.dumps({s: v for s, v in leaves if v[0]["op"] == V4 + leaf})
        )
        env = {**os.environ, "COMPASS_ARCH_DECLARED": "1", FOUND: str(found)}
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            + [f"{__file__}::{request.node.name}"],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert done.returncode == 0, done.stdout[-4000:]
        return
    driverless.install(str(tmp_path), "gfx950")
    found = json.loads(pathlib.Path(os.environ[FOUND]).read_text())
    [(op, _)] = found.values()
    module, function = leaf.split(":")
    kernel = _Kernel()
    name = f"_v4_{function.removeprefix('write_v4_')}_kernel"
    monkeypatch.setattr(importlib.import_module(V4 + module), name, kernel)
    monkeypatch.setattr(fctx._forward_kv_cache_context, "kv_cache_data", {})

    def call(stub, op, sig, instances, n):
        kwargs, _ = stub._copy(op, sig, instances, 0)
        ops._function(op["op"])(**kwargs)
        return {}

    monkeypatch.setattr(ops.Stub, "_time", call)
    runner = _Runner()
    runner.forward_vars["indices"] = torch.zeros(1 << 23, dtype=torch.int32)
    runner.positions = torch.zeros(1 << 14, dtype=torch.int64)
    placed = SimpleNamespace(block_size=16, batch=lambda view, block, slot: (1, slot))

    [record] = ops.Stub(runner, 10_000).point(placed, shape, context, found)

    assert "refused" not in record, record["refused"]
    # The rebuilt geometry gives the kernel the contract's envelope stride.
    [(grid, launched)] = kernel.launches
    assert grid == (op["args"]["T"],)
    rows = launched.get("HCA_ENVELOPE_ROWS", launched.get("ENVELOPE_ROWS"))
    assert rows == dict(op["args"]["geometry"]["state"])["envelope_rows"]
