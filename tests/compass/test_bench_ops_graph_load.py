# SPDX-License-Identifier: MIT
"""`compass measure ops` loads each `op_graph` it reads as a run loads it,
through the invalidation matrix under this process's readings and the engine
config the entry records.

A graph at the key the bench looks up whose row states another torch, ATOM
source or model refuses, naming the axis: the target's, a drafter's beside it,
and on `--plan` one the store holds, which the plan refuses rather than trace
its shape again, since a capture of it would be keyed as it is. A shape the
store lacks is still traced. No device.
"""

import json
import re
from types import SimpleNamespace

import pytest
from test_artifact_invalidation import moved
from test_artifact_store import (
    clean_checkout,  # noqa: F401  autouse
    stanza,
)
from test_bench_ops import _graph, _leaf, _t
from test_op_graph_arch import ENGINE, run_readings

from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    Axis,
    GateState,
    Row,
    Topology,
    axes_of,
    member_name,
)
from atom.compass.bench import ops
from atom.compass.capture import graph_key
from atom.compass.capture.forward import decode, structure

ARCH = "gfx942"
#: The axes of the op_graph row the bench reads off this process; the engine
#: config it takes from the entry, the tracer's own arguments.
READ = [a for a in axes_of(Row.OP_GRAPH) if a is not Axis.ENGINE_CONFIG]


def config(tmp_path, drafter=None):
    """A stand-in config of a model whose config.json is `{}`, with
    `drafter` as its speculative config."""
    (tmp_path / "model").mkdir(exist_ok=True)
    (tmp_path / "model" / "config.json").write_text("{}")
    return SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_expert_parallel=False,
        enforce_eager=True,
        speculative_config=drafter,
        **ENGINE,
    )


def traced(store, c, conditions, leaf, draft=False):
    """`c`'s decode B=2 graph of one leaf `leaf`, its target's or with `draft`
    its drafter's on a step that samples, at the key the bench looks it up at,
    handed off under `conditions`."""
    at = structure(c, "decode", True, ARCH, True, draft)
    x = ("x", _t([2, 8], "activation"))
    store.publish(
        graph_key(at, "decode B=2"),
        provenance=stanza("compass trace ops"),
        topology=Topology(),
        conditions=conditions,
        gates=GateState.of(),
        members={
            member_name("graph", ops.RANK, "json"): json.dumps(
                _graph(_leaf(leaf, x), layers=1)
            ).encode()
        },
    )


def named(axis) -> str:
    return re.escape(f"`{Row.OP_GRAPH}` x `{axis}`")


@pytest.mark.parametrize("axis", READ, ids=str)
def test_measure_ops_refuses_a_graph_whose_row_states_another_reading(tmp_path, axis):
    c, store = config(tmp_path), ArtifactStore(tmp_path / "store")
    traced(store, c, moved(run_readings(c.model), axis), "aiter.held")

    with pytest.raises(ArtifactRefusal, match=named(axis)):
        ops.graph(store, c, decode(2), ARCH)


@pytest.mark.parametrize("axis", READ, ids=str)
def test_a_drafter_s_graph_that_refuses_here_refuses_its_step_s_leaves(tmp_path, axis):
    c = config(tmp_path, SimpleNamespace(method="mtp", num_speculative_tokens=1))
    store = ArtifactStore(tmp_path / "store")
    traced(store, c, run_readings(c.model), "aiter.target")
    traced(store, c, moved(run_readings(c.model), axis), "aiter.draft", draft=True)

    with pytest.raises(ArtifactRefusal, match=named(axis)):
        ops.leaves_at(store, c, decode(2), ARCH)


@pytest.mark.parametrize("axis", READ, ids=str)
def test_a_plan_refuses_a_held_graph_that_refuses_here_and_traces_nothing(
    tmp_path, monkeypatch, axis
):
    c, store = config(tmp_path), ArtifactStore(tmp_path / "store")
    traced(store, c, moved(run_readings(c.model), axis), "aiter.held")
    captured = []
    monkeypatch.setattr(ops, "_capture", lambda c, shapes, *a: captured.append(shapes))

    with pytest.raises(ArtifactRefusal, match=named(axis)):
        ops._load(store, c, str(store.root), {decode(2)}, {}, ARCH)
    assert captured == []


def test_a_plan_traces_a_shape_the_store_lacks_and_reads_it_back(tmp_path, monkeypatch):
    c, store = config(tmp_path), ArtifactStore(tmp_path / "store")
    captured = []

    def capture(config, shapes, *args):
        captured.append(shapes)
        traced(store, config, run_readings(config.model), "aiter.traced")

    monkeypatch.setattr(ops, "_capture", capture)
    graphs = {}

    ops._load(store, c, str(store.root), {decode(2)}, graphs, ARCH)

    assert captured == [[decode(2)]]
    assert [op["op"] for op, _ in graphs[decode(2)].values()] == ["aiter.traced"]
