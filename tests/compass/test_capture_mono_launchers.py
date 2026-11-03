# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's mono decode launches its kernels through module-level
launchers of what each reads: K1 and K2 of their build key and their tensors
and scalars by ABI name, the step's beginning of its peer buffer's fields, the
step metadata of the step's. A launcher packs each tensor as its address, so
Compass off a kernel takes what its call site packed; under a driverless
capture each is one leaf of tensors and scalars, replaying to its kernel.
"""

import collections
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import torch
from test_capture_ir import _ops
from test_capture_v41_flash import LAYERS, MONO, published, trace

from atom.models.deepseek_v41.mono import step_meta
from atom.mono.plan.execution import BLOCKS

V41 = "atom.models.deepseek_v41.mono"
#: Each launcher, its leaves in a decode step and the kernel each replay
#: launches, by the start of its name.
LAUNCHERS = {
    "atom.mono.runtime.step_begin:_launch_step_begin": (1, "flydsl.launch_mono_step_begin_"),
    f"{V41}.step_meta:_launch_step_meta": (1, "triton._step_meta_kernel"),
    f"{V41}.kernels.attn_pre:launch_attn_pre": (LAYERS, "flydsl.launch_v41_attn_pre_s16_"),
    f"{V41}.kernels.layer_post:launch_layer_post": (LAYERS, "flydsl.launch_v41_layer_post_s16_"),
}  # fmt: skip
#: Before the capture: what each replay launched, printed.
REPLAYS = (
    "from atom.compass.capture import forward\n"
    "replay = forward._Recorder._replay\n"
    "def printed(self, op):\n"
    "    kernels = replay(self, op)\n"
    "    print('replayed', op.name, *kernels, flush=True)\n"
    "    return kernels\n"
    "forward._Recorder._replay = printed\n"
)


def _held(value):
    """Whether `value` is a tensor, a scalar, or a list or dict of them: the
    same at both contexts by being itself."""
    if isinstance(value, list):
        return all(map(_held, value))
    if isinstance(value, dict):
        dict_ = value.get("object") == "builtins:dict" and _held(value["state"])
        return "shape" in value or dict_
    return value is None or isinstance(value, (bool, int, float, str))


def test_each_mono_launch_is_one_leaf_of_tensors_and_scalars_replaying_to_its_kernel(
    tmp_path,
):
    done = trace(tmp_path, MONO, "gfx950", REPLAYS)

    graph, _ = published(tmp_path, done)["decode B=16"]
    leaves = [op for op in _ops(graph["region"]) if op["op"] in LAUNCHERS]
    counts = collections.Counter(op["op"] for op in leaves)
    assert counts == {name: count for name, (count, _) in LAUNCHERS.items()}
    for op in leaves:
        args = dict(op["args"])
        key = args.pop("key", None)
        # A build key by its fields; nothing else is an object.
        assert key is None or key["object"].endswith("Build") and _held(key["state"])
        assert all(map(_held, args.values())), op
    replayed = re.findall(r"^replayed (\S+)(.*)$", done.stdout, re.MULTILINE)
    for name, (_, kernel) in LAUNCHERS.items():
        # Each replay launches its one kernel.
        kernels = [launched.split() for leaf, launched in replayed if leaf == name]
        assert kernels and all(len(k) == 1 and k[0].startswith(kernel) for k in kernels)


def _launched(calls):
    """A recorder of each launch's positional arguments and build key or grid."""
    return lambda first: lambda *args, **kw: calls.append((first, args, kw))


def _is(got, want):
    if isinstance(want, tuple):
        return len(got) == len(want) and all(map(_is, got, want))
    return got is want if isinstance(want, torch.Tensor) else got == want


def test_the_step_metadata_kernel_takes_what_write_step_meta_passed(monkeypatch):
    calls = []

    class Kernel:  # a launch's grid, then its arguments
        def __getitem__(self, grid):
            return _launched(calls)(grid)

    monkeypatch.setattr(step_meta, "_step_meta_kernel", Kernel())
    rows = 16

    def ints(*shape):
        return torch.zeros(*shape, dtype=torch.int32)

    step = SimpleNamespace(batch_ids=ints(rows), slots=ints(rows), positions=ints(rows))
    step.block_tables = ints(rows, 64)
    ring_rel, key_meta = ints(rows), ints(rows, step_meta.KMETA)
    plans = {1: ints(BLOCKS, 4), 4: ints(BLOCKS, 4)}
    step_meta.write_step_meta(
        step,
        SimpleNamespace(window_size=128),
        SimpleNamespace(slot_rows=5153, ring_slots=129),
        ring_rel,
        key_meta,
        plans,
        (4, 8),
    )

    (grid, args, constexprs), *more = calls
    assert not more and grid == (3,) and constexprs["RATIOS"] == (1, 4)
    want = (step.batch_ids, step.slots, step.positions, ring_rel, key_meta)
    want += (tuple(plans.values()), 4, 12, 64, 5153, 129, 128, rows)
    assert _is(args, want)


def test_the_step_begin_kernel_takes_the_peer_buffer_s_fields(monkeypatch):
    # Here, not at the top: it imports aiter, which the K1 and K2 test's own
    # process imports only once it declared an arch.
    from atom.mono.runtime import step_begin

    calls = []
    monkeypatch.setattr(step_begin, "_build", _launched(calls))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "the caller's")
    scratch = torch.zeros(64, dtype=torch.uint8)
    peers = SimpleNamespace(
        addresses=torch.zeros(4, dtype=torch.int64),
        bytes=torch.zeros(256, dtype=torch.uint8),
        rank=2,
    )
    state = torch.zeros(step_begin.STATE_WORDS, dtype=torch.int64)

    step_begin.step_begin((scratch,), peers, state, debug=True)

    (key, args, kw), *more = calls
    assert not more and key == step_begin.StepBeginBuild(tp=4, debug=True)
    p = torch.Tensor.data_ptr
    assert args == (p(scratch), 64, 0, 0, p(peers.addresses), 256, 2, p(state))
    assert kw == {"stream": "the caller's"}


def test_k1_and_k2_take_each_tensor_as_its_address_in_abi_order(
    request, tmp_path, monkeypatch
):
    if "COMPASS_ARCH_DECLARED" not in os.environ:
        # Their modules import aiter, which needs an arch declared: once per
        # process, so in a process of its own.
        env = {**os.environ, "COMPASS_ARCH_DECLARED": "1"}
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
    from atom.compass.capture import driverless

    driverless.install(str(tmp_path), "gfx950")
    from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
    from atom.models.deepseek_v41.mono.kernels import layer_post as k2

    for module, launch, built, key in (
        (k1, k1.launch_attn_pre, "built_attn_pre", k1.AttnPreBuild(tokens=16)),
        (k2, k2.launch_layer_post, "built_layer_post", k2.LayerPostBuild(tokens=16)),
    ):
        calls = []
        monkeypatch.setattr(module, built, _launched(calls))
        # A host tensor of its own a name, but the integers.
        values = {n: torch.zeros(1, dtype=torch.uint8) for n in module.ABI.names}
        values.update(rank=3, layer=7, tl=0)

        launch(key, **values)

        p = torch.Tensor.data_ptr
        want = [values[n] for n in module.ABI.names]
        want = tuple(p(v) if isinstance(v, torch.Tensor) else v for v in want)
        assert calls == [(key, want, {"stream": torch.cuda.current_stream()})]
