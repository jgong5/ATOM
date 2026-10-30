# SPDX-License-Identifier: MIT
"""`run._graph_pool` reserves what ATOM's `_estimate_cudagraph_overhead` does.

The real side is the estimator and the runner methods it calls, compiled from
this checkout's `model_runner.py` as `test_tbo_forward_mode.py` compiles
`prepare_model`: importing the real runner needs a driver. Its device reads are
planted: the warmup's activation gap is the one the simulated side is given and
the card's total memory is the machine's capacity. Both sides run on the same
runner, so `_graph_pool` asks ATOM's own graph-mode check which branch applies.
"""

import ast
import logging
from types import SimpleNamespace

import pytest
import torch
from test_runner_non_allocating import ATOM_RUNNER, _classes

from atom.compass.backends.provenance import Method
from atom.compass.memory.terms import Term
from atom.compass.run import _graph_pool
from atom.config import CUDAGraphMode
from atom.utils import envs

#: The `ModelRunner` methods `_estimate_cudagraph_overhead` runs.
REAL = {
    "_estimate_cudagraph_overhead",
    "_tr_hybrid_active",
    "_token_graph_bytes_per_token",
    "_dspark_capture_q_buckets",
}
CAPACITY = 309_220_868_096
ACTIVATIONS = 8_225_887_232


def _atom_runner():
    cls = _classes(ATOM_RUNNER)["ModelRunner"]
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert REAL <= methods.keys(), REAL - methods.keys()
    cls.body = [methods[name] for name in sorted(REAL)]
    stats = {
        "allocated_bytes.all.peak": ACTIVATIONS + 1,
        "allocated_bytes.all.current": 1,
    }
    cuda = SimpleNamespace(
        memory_stats=lambda: stats, mem_get_info=lambda: (CAPACITY // 2, CAPACITY)
    )
    namespace = {
        "torch": SimpleNamespace(
            inference_mode=torch.inference_mode, finfo=torch.finfo, cuda=cuda
        ),
        "logger": logging.getLogger("atom"),
        "envs": envs,
    }
    exec(ast.unparse(cls), namespace)  # noqa: S102
    return namespace["ModelRunner"]


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize(
    "eager, mode",
    [
        (True, CUDAGraphMode.FULL),
        (False, None),
        (False, CUDAGraphMode.FULL),
        (False, CUDAGraphMode.TR_HYBRID),
        (False, CUDAGraphMode.FULL_AND_TR_HYBRID),
    ],
)
def test_a_simulated_start_reserves_what_atoms_estimator_does(eager, mode, dp):
    r = object.__new__(_atom_runner())
    r.enforce_eager = eager
    r.config = SimpleNamespace(
        enforce_eager=eager,
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode, cudagraph_capture_sizes=[2**i for i in range(15)]
        ),
        max_num_seqs=512,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.9,
        hf_config=SimpleNamespace(hidden_size=5120, num_hidden_layers=64),
        torch_dtype=torch.bfloat16,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
        speculative_config=None,
    )
    machine = SimpleNamespace(value={"device.memory.capacity_bytes": CAPACITY}.get)
    model = SimpleNamespace(
        activations=Term("activations", ACTIVATIONS, Method.MEASURED, "planted")
    )
    reserved = _graph_pool(r, machine, model).total
    assert reserved == r._estimate_cudagraph_overhead()
