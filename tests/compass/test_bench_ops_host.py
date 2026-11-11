# SPDX-License-Identifier: MIT
"""A tensor argument ATOM holds on the host: the capture states its device, and
`compass measure ops` builds it there rather than binding ATOM's metadata; a
tensor stating none is on the device, as in a graph captured before.
"""

import dataclasses

import torch
from test_bench_ops import _graph, _leaf, _Runner, _t
from test_capture_lift import recorder  # noqa: F401  the recorder fixture

from atom.compass.bench import ops
from atom.compass.capture import graph

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
