# SPDX-License-Identifier: MIT
"""A simulated runner states the prefill token-graph buckets ATOM's capture
records, recording none, and ATOM's scheduler, prefill delayer and dummy
forward act on them as on a real capture's: a DP rank with no prefill of its
own runs the group's dummy prefill beside a peer's prefill, not a dummy decode.

ATOM's `prefill_token_graphs`, `warmup_max_tokens` and `dummy_execution` run
from its source, as `test_dp_step_max.py` runs `dummy_execution`.
"""

import ast
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig
from test_dp_step_max import Recording, dp2  # noqa: F401
from test_runner_non_allocating import ATOM_RUNNER, PACKAGE, _classes

from atom.compass.backends.shape import RequestShape
from atom.compass.runner import projection
from atom.compass.runner.overrides import NonAllocatingRunner, install_cost_backend
from atom.compass.runner.step_output import DeferredTokenStream
from atom.model_engine.prefill_delayer import PrefillDelayer
from atom.model_engine.scheduler import ScheduledBatch, Scheduler
from atom.model_engine.sequence import (
    Sequence,
    SequenceStatus,
    SequenceType,
    new_block_table,
)
from atom.sampling_params import SamplingParams
from atom.utils.graph_holders import release_registered_graphs
from atom.utils.region_graph import TokenGraph

BLOCK, BUDGET = 16, 1024
ATOM = ("prefill_token_graphs", "warmup_max_tokens", "dummy_execution")


def _runner_class():
    atom_runner = _classes(ATOM_RUNNER)["ModelRunner"]
    atom_runner.body = [
        n for n in atom_runner.body if isinstance(n, ast.FunctionDef) and n.name in ATOM
    ]
    namespace = {
        **{c.__name__: c for c in (Sequence, SequenceStatus, SequenceType)},
        **{c.__name__: c for c in (ScheduledBatch, TokenGraph, NonAllocatingRunner)},
        "np": np,
        "new_block_table": new_block_table,
    }
    exec(ast.unparse(atom_runner), namespace)  # noqa: S102
    compass_runner = _classes(PACKAGE / "model_runner.py")["CompassModelRunner"]
    exec(ast.unparse(compass_runner), namespace)  # noqa: S102
    return namespace["CompassModelRunner"]


CompassModelRunner = _runner_class()


@pytest.fixture(autouse=True)
def released():
    """`TokenGraph`'s buckets are the process's: each test's go with it."""
    yield
    release_registered_graphs()


def runner(dp_rank=0, token_graphs=True, decode_only=False):
    r = object.__new__(CompassModelRunner)
    r.config = MockConfig(
        pipeline_parallel_size=1,
        parallel_config=SimpleNamespace(
            data_parallel_size=2, data_parallel_rank=dp_rank
        ),
        max_num_batched_tokens=BUDGET,
        enable_dp_attention=True,
        prefill_context_parallel_size=1,
        disagg_is_decode=decode_only,
        capture_sizes=[1, 2, 4],
    )
    r.rank, r.block_size, r.enforce_eager = 0, BLOCK, False
    r.capture_sizes_np = np.asarray([0], dtype=np.int32)
    r.running_tokens_per_dp_rank = np.zeros(2, dtype=np.int32)
    r.tbo_decode_graphs = False
    r.attn_metadata_builder = SimpleNamespace(
        step_needs_eager=lambda batch: False, supports_token_graph=token_graphs
    )
    r._token_stream = DeferredTokenStream([0])
    install_cost_backend(r, Recording())
    return r


@pytest.mark.parametrize(
    "token_graphs, decode_only, stated",
    [(True, False, (256, 512, 1024)), (False, False, ()), (True, True, ())],
    ids=["token-graphs", "no-token-graph-builder", "decode-only"],
)
def test_capture_states_the_buckets_atom_records(
    monkeypatch, token_graphs, decode_only, stated
):
    """`ATOM_TOKEN_GRAPH_SIZES` up to `warmup_max_tokens`, the batch budget."""
    monkeypatch.setenv("ATOM_TOKEN_GRAPH_SIZES", "256,512,1024,2048")
    r = runner(token_graphs=token_graphs, decode_only=decode_only)
    assert r.capture_cudagraph() == (0.0, [], 0)
    assert TokenGraph.captured_sizes("prefill") == stated
    assert TokenGraph.captured_sizes("decode") == ()
    assert r.prefill_token_graphs() is bool(stated)


@pytest.mark.parametrize("eager", [False, True], ids=["graphs", "eager"])
def test_an_idle_rank_dummy_prefills_beside_a_peer_s_prefill(
    dp2, monkeypatch, eager  # noqa: F811
):
    """The scheduler of a rank with nothing to prefill takes the runner's
    answer: with buckets stated it gives the tick to the group's dummy
    prefill, the smallest bucket's tokens, which `decide` pads to the bucket
    of the group's 300 tokens; an eager deployment, whose engine never asks
    for a capture, keeps the one-token dummy decode."""
    monkeypatch.setenv("ATOM_TOKEN_GRAPH_SIZES", "256,512,1024")
    ranks = [runner(r) for r in (0, 1)]
    for r in ranks:
        r.enforce_eager = eager
        if not eager:
            r.capture_cudagraph()
    schedulers = [
        Scheduler(
            MockConfig(
                kv_cache_block_size=BLOCK,
                num_kvcache_blocks=64,
                max_model_len=BUDGET,
                max_num_batched_tokens=BUDGET,
            ),
            prefill_token_graphs=r.prefill_token_graphs(),
        )
        for r in ranks
    ]
    prompt = Sequence(list(range(5, 305)), BLOCK, sampling_params=SamplingParams())
    schedulers[0].add(prompt)

    def tick(rank):
        def run():
            scheduler = schedulers[rank]
            scheduler.set_prefill_delayer(
                PrefillDelayer(2, projection.dp_group(), BUDGET)
            )
            scheduled = scheduler.schedule()
            if scheduled is not None:
                return ranks[rank].forward(scheduled[0])
            return ranks[rank].dummy_execution(scheduler.dummy_prefill)

        return run

    dp2(tick(0), tick(1))

    views = [r.compass_backend.views[0] for r in ranks]
    assert views[0].requests == (RequestShape(300, 300, False),)
    assert views[1].dp_dummy and views[1].group_prefills
    dummy = RequestShape(1, 1, True) if eager else RequestShape(256, 256, False)
    assert views[1].requests == (dummy,)
    assert schedulers[1].dummy_prefill is not eager
    padded = [300, 1] if eager else [512, 512]
    assert [r.running_tokens_per_dp_rank.tolist() for r in ranks] == [padded] * 2
