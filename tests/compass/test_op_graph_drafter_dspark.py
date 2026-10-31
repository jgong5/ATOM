# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1-Flash's DSpark5 drafter under `compass trace ops` with no
device, one rank of TP4 EP4 (of DP4 for a middle chunk under DP), priced by
the op-level backend a run builds.

A block drafter runs its draft-KV pass between its target's forward and its
sampling, before `forward_done_event`: a step that samples is three graphs,
the target's, the pass's and the drafter's propose, and the pass prices among
the target's terms, the propose as draft terms. A middle chunk is cut where
the pass starts, so the pass is its drafter's.
"""

import json
import re

import pytest
from test_capture_ir import _ops
from test_capture_v41_flash import ARGS, EP4, TP, trace
from test_op_graph import _engine, g
from test_op_work_v41 import MI455
from test_work import SPEC

from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.op_graph import DRAFT, DRAFT_KV, KV, OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.work import price
from atom.compass.capture import forward

DSPARK = ARGS + EP4 + " --method dspark --num-speculative-tokens 5"
SHAPES = "--width 16 --prefill 512 1"
DECODE, PREFILL = "decode B=16", "prefill T=512 B=1"
#: The draft-KV pass's epilogue, the block's attention and its Markov sampler.
TAIL = "atom.model_ops.deepseek_v41.dspark:fused_draft_kv_tail"
BLOCK = "atom.model_ops.sparse_attn_v4:_sparse_attn_triton"
MARKOV = "atom.model_ops.dspark_markov_sample:_launch"
DRAFTER = {TAIL, BLOCK, MARKOV}
#: The target's sampler, which runs after the pass.
SAMPLER = "aiter.mixed_sample_outer_exponential.default"
#: A decode row verifies its 5 drafts and one over 1024 tokens of context.
VERIFY = BatchView((RequestShape(6, 1024, True),) * 16, produces_output=True)
FINAL = BatchView((RequestShape(512, 512, False),), produces_output=True)
MIDDLE = BatchView((RequestShape(512, 512, False),), produces_output=False)


def _capture(root, argv) -> dict:
    """`trace` over `argv` into `root`: each graph by what its structure adds
    to the step's mode (``draft kv``, ``no output draft``, ...) and its shape."""
    done = trace(root, argv, "gfx1250")
    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert done.returncode == 0 and tokens, done.stderr[-4000:]
    store, rank = ArtifactStore(str(root / "store")), Topology().ranks()[0]
    graphs = {}
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph = json.loads(entry.read_member("graph", rank, "json"))
        part = graph["structure"].split(" eager", 1)[1].strip()
        graphs[part, graph["shape"]] = graph
    return graphs


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    """The capture's directory and its graphs (`_capture`)."""
    root = tmp_path_factory.mktemp("dspark")
    return root, _capture(root, SHAPES + DSPARK)


def _kinds(graph) -> set:
    return {op["op"] for op in _ops(graph["region"])}


@pytest.mark.parametrize("shape", [DECODE, PREFILL])
def test_a_sampling_step_is_its_target_its_draft_kv_pass_and_its_propose(
    captured, shape
):
    _, graphs = captured
    target, kv, propose = (graphs[p, shape] for p in ("", "draft kv", "draft"))

    # The pass writes the target's rows into the draft's window, and only it.
    assert _kinds(kv) & DRAFTER == {TAIL}
    # The target runs none of the drafter's kernels, and keeps the sampling
    # that follows the pass.
    assert not _kinds(target) & DRAFTER
    assert SAMPLER in _kinds(target) - _kinds(kv)
    # The propose drafts the block and samples it, and writes no context.
    assert _kinds(propose) & DRAFTER == {BLOCK, MARKOV}


def test_a_middle_chunk_s_drafter_starts_where_its_draft_kv_pass_starts(captured):
    _, graphs = captured

    parts = {part for part, shape in graphs if shape == PREFILL}
    assert parts == {"", "draft kv", "draft", "no output", "no output draft"}
    # A chunk that finishes no prompt writes no draft context, and proposes
    # nothing: its drafter's graph is the pass's gather of its taps.
    assert not _kinds(graphs["no output draft", PREFILL]) & DRAFTER


def test_a_dp_middle_chunk_s_drafter_is_its_draft_kv_pass_and_its_propose(tmp_path):
    graphs = _capture(tmp_path, "--prefill 512 1" + DSPARK + " --enable-dp-attention")

    # Under DP the chunk proposes after its pass, for the group's collectives:
    # one graph past the first cut.
    assert BLOCK in _kinds(graphs["no output draft", PREFILL])
    assert BLOCK not in _kinds(graphs["no output", PREFILL])


def _run(root, graphs):
    """The op-level backend a run builds for the capture's engine arguments,
    at the quantization hash its built model keyed the graphs by."""
    config = forward._config(str(root / "model"), TP, DSPARK.split()[1:])
    config.quant_config = _engine(graphs["", DECODE])["quant_config"]
    store = {"store": str(root / "store")}
    return compass_run.op_graph(store, config, MI455, ep=TP)


def _terms(graph, prefix, step) -> dict:
    """Each op kind of `graph` as `prefix` names its term, its leaves priced
    at `step` and folded in run order."""
    flat: dict = {}
    for op in _ops(graph["region"]):
        if (w := work(op, step, TP, "bf16", TP)) is not None:
            flat.setdefault(f"{prefix}op.{op['op']}", []).append(price(w, MI455))
    return {name: fold_seconds(s) for name, s in flat.items()}


@pytest.mark.parametrize("step, shape", [(VERIFY, DECODE), (FINAL, PREFILL)])
def test_a_run_prices_the_draft_kv_pass_among_the_target_s_terms(captured, step, shape):
    root, graphs = captured
    cost = _run(root, graphs).price(step)

    passed = {t.name: t.seconds for t in cost.terms if t.name.startswith(DRAFT_KV)}
    # The step's output waits for the pass: it prices among the target's terms.
    assert passed == _terms(graphs["draft kv", shape], DRAFT_KV, step)
    drafted = {t.name for t in cost.draft_terms}
    assert all(name.startswith(DRAFT) for name in drafted)
    assert {f"{DRAFT}op.{BLOCK}", f"{DRAFT}op.{MARKOV}"} <= drafted
    assert 0 < cost.draft_seconds < cost.target_seconds


def test_a_run_prices_a_middle_chunk_s_draft_kv_pass_as_its_draft(captured):
    root, graphs = captured
    cost = _run(root, graphs).price(MIDDLE)

    assert not [t for t in cost.terms if t.name.startswith(DRAFT_KV)]
    assert {t.name: t.seconds for t in cost.draft_terms} == _terms(
        graphs["no output draft", PREFILL], DRAFT, MIDDLE
    )


def test_a_block_drafter_s_sampling_step_with_no_draft_kv_graph_is_refused():
    def graph(*at, draft=False):
        return None if draft == KV else g(1, 8, 8)

    step = BatchView((RequestShape(2, 9, True),), produces_output=True)
    backend = OpGraphBackend(SPEC, graph, kv="bf16", drafts=1, block=True)
    with pytest.raises(CostRefused, match="for its drafter's draft-KV pass"):
        backend.price(step)
