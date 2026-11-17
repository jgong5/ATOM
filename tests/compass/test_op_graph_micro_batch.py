# SPDX-License-Identifier: MIT
"""A two-batch-overlap micro-batch's `op_graph` is traced as ATOM runs it under
overlap, its MoE on one stream, and keyed apart from a whole step of its
shape, which forks each layer's shared expert beside the MoE at up to
`ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD` tokens: DeepSeek-V4-Pro's DPA-TP8
prefill side at T=512.
"""

import json
import re

from test_capture_v4_pro import DPA_TP8, LAYERS, _pars, trace
from test_op_graph import _run_config
from test_op_graph_gathered import DP, RANK, _record
from test_op_work_v4 import SPEC

from atom.compass.artifacts import ArtifactStore, Kind
from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.bench import ops
from atom.compass.capture.forward import Shape
from atom.compass.run import _op_graphs


def _split(peers: int) -> BatchView:
    """Rank 0's 1024 tokens of one sequence, split 512 and 512, each peer's
    `peers` and `peers`."""
    halves = ((512,) + (peers,) * (DP - 1),) * 2
    one = (RequestShape(1024, 1024, False),)
    rows = sum(map(sum, halves))
    return BatchView(one, moe_rows=rows, produces_output=True, tbo_tokens=halves)


def test_a_micro_batch_runs_its_moe_on_one_stream_and_prices_where_a_whole_step_forks(
    monkeypatch, tmp_path_factory
):
    # Each rank's 1024 tokens split 512 and 512; `--prefill 512 1`, one forward,
    # is a whole step of the micro-batches' shape.
    monkeypatch.setenv("ATOM_TBO_PREFILL_MIN_TOKENS", "512")
    ranks = [([[1024, 1024, False]], DP * 1024)] * DP
    record = _record(tmp_path_factory.mktemp("record"), ranks)
    root = tmp_path_factory.mktemp("v4pro")
    argv = f"--prefill 512 1 --whole --symbolic --population {record}"
    done = trace(root, argv + DPA_TP8 + " --enable-tbo")
    store = ArtifactStore(str(root / "store"))
    graphs = {}
    for token in re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE):
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph = json.loads(entry.read_member("graph", RANK, "json"))
        part = graph["structure"].rpartition(" prefill eager")[2]
        graphs[graph["shape"], part] = graph
    assert len(graphs) == 8, done.stderr[-4000:]

    # The run's lookup reads each micro-batch's graph, at the default G and,
    # from the graph that holds G, at another.
    whole = graphs["prefill T=512 B=1", ""]
    config = _run_config(whole["structure"], root / "model")
    lookup = _op_graphs(store.root, config, "gfx950")
    backend = OpGraphBackend(SPEC, lookup, dp=DP, kv="fp8")
    for peers in (512, 64):
        cost = backend.price(_split(peers))
        assert [t.name for t in cost.terms] == ["op.tbo"] and cost.seconds > 0
    one = (RequestShape(512, 512, False),)
    whole_step = BatchView(one, moe_rows=DP * 512, produces_output=True)
    assert backend.price(whole_step).seconds > 0

    # Under overlap ATOM runs a micro-batch's MoE on one stream; a whole step of
    # its shape forks each layer's shared expert.
    for (shape, part), graph in graphs.items():
        assert len(_pars(graph["region"])) == (0 if "micro-batch" in part else LAYERS)
    # The bench reads the micro-batch's graph by the same key.
    micro = Shape("prefill", 512, 1, micro_batch=True)
    at = ("prefill T=512 B=1", " micro-batch no output")
    assert ops.graph(store, config, micro, "gfx950", False) == graphs[at]
