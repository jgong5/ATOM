# SPDX-License-Identifier: MIT
"""The capture's fake mode answers a named aiter operator whose schema mutates
nothing from its dispatch cache; any other aiter operator goes past the cache.

The declarations are process-global, so the calls run in a process of their own
and report, for each operator, what its second call with the same arguments
did to torch's cache counters.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

TREE = pathlib.Path(__file__).resolve().parents[2]

CALLS = """
import json, sys
from unittest.mock import patch
from atom.compass.capture import driverless
driverless.install(sys.argv[1], "gfx942")
import aiter, torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

def second_call(op, *shapes):
    with driverless.driverless_mode(ShapeEnv()):
        args = [torch.empty(s, dtype=torch.bfloat16, device="cuda") for s in shapes]
        op(*args)
        hits, bypasses = FakeTensorMode.cache_hits, dict(FakeTensorMode.cache_bypasses)
        op(*args)
    after = FakeTensorMode.cache_bypasses
    return {
        "hits": FakeTensorMode.cache_hits - hits,
        "bypasses": {r: n - bypasses.get(r, 0) for r, n in after.items() if n != bypasses.get(r, 0)},
    }

norm, silu = torch.ops.aiter.rmsnorm2d_fwd.default, torch.ops.aiter.silu_and_mul.default
out = {
    "admitted": second_call(lambda x, w: norm(x, w, 1e-6), (4, 512), (512,)),
    "mutating": second_call(silu, (4, 256), (4, 512)),
}
with patch.object(driverless, "CACHED", driverless.CACHED | {str(silu)}):
    out["mutating, named"] = second_call(silu, (8, 256), (8, 512))
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def calls(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fake-cache")
    done = subprocess.run(
        [sys.executable, "-c", CALLS, str(tmp)],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE), "AITER_LOG_LEVEL": "WARNING"},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.splitlines()[-1])


def test_an_admitted_aiter_op_is_answered_from_the_cache(calls):
    assert calls["admitted"] == {"hits": 1, "bypasses": {}}


def test_a_mutating_aiter_op_goes_past_the_cache_even_when_named(calls):
    assert calls["mutating"] == {"hits": 0, "bypasses": {"non-builtin": 1}}
    assert calls["mutating, named"] == {"hits": 0, "bypasses": {"non-builtin": 1}}
