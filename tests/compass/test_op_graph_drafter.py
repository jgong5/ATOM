# SPDX-License-Identifier: MIT
"""A drafter's step, traced and priced apart from its target's.

Under ``--method mtp`` `compass trace ops` traces a decode step whose rows
verify the drafter's tokens, and cuts each step that samples where ATOM records
`forward_done_event`: the target's work before the cut, the drafter's after
it, each an `op_graph` of its own. `OpGraphBackend` prices the drafter's graph
as terms tagged draft. Traced on Qwen3.5-35B-A3B-FP8's published config.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_backend_step_linear import view
from test_capture_ir import TREE, _ops, _record
from test_op_graph import MIDDLE, g
from test_work import SPEC

from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.backends import BatchView, CostRefused, RequestShape
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.work import gemm, price
from atom.compass.capture import population_shapes
from atom.compass.run import _op_graphs
from atom.compass.spec import MachineSpec

CONFIG = pathlib.Path(__file__).with_name("qwen3_5_35b_a3b_fp8_config.json")
MACHINE = TREE / "scripts" / "compass" / "machines" / "mi308x-hjbog39.json"
ATTENTION = "aiter.unified_attention_with_output_base.default"
SAMPLER = "atom.model_ops.rejection_sampler:rejection_sample"
ARGMAX = "aiter.ops.flydsl.topk.topk_per_row_argmax:topk_per_row_argmax"
#: Two rows, each verifying one draft and its bonus token.
VERIFY = view((2, 90, True), (2, 9, True), rung=2)

_CAPTURES: dict = {}


def capture(drafts, tmp_path_factory):
    """`compass trace ops --width 2` under an MTP drafter of `drafts` tokens:
    the capture's directory and its graphs by structure."""
    if drafts in _CAPTURES:
        return _CAPTURES[drafts]
    root = tmp_path_factory.mktemp(f"mtp{drafts}")
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    argv = ["trace", "ops", "--model", str(root / "model"), "--width", "2"]
    argv += ["--store", str(root / "store"), "--", "--method", "mtp"]
    argv += ["--num-speculative-tokens", str(drafts)]
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from atom.compass.cli import main; main(sys.argv[1:])",
            *argv,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert tokens, f"rc {done.returncode}\n{done.stderr[-4000:]}"
    store, rank = ArtifactStore(str(root / "store")), Topology().ranks()[0]
    graphs = {}
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph = json.loads(entry.read_member("graph", rank, "json"))
        graphs[graph["structure"]] = graph
    _CAPTURES[drafts] = root, graphs
    return root, graphs


@pytest.mark.parametrize("drafts", [1, 2])
def test_a_verify_step_is_cut_at_forward_done_event_into_target_and_drafter(
    drafts, tmp_path_factory
):
    _, graphs = capture(drafts, tmp_path_factory)
    [target] = [key for key in graphs if not key.endswith(" draft")]
    assert set(graphs) == {target, f"{target} draft"}
    assert f" mtp={drafts} " in target
    ours, theirs = (_ops(graphs[key]["region"]) for key in (target, f"{target} draft"))
    vocab = json.loads(CONFIG.read_text())["text_config"]["vocab_size"]

    # The target verifies each row's drafts and its bonus token, and samples.
    assert {op["args"]["q"]["shape"][0] for op in ours if op["op"] == ATTENTION} == {
        2 * (drafts + 1)
    }
    assert SAMPLER in {op["op"] for op in ours} - {op["op"] for op in theirs}
    # The drafter runs its layer once per draft, each through the shared
    # embedding and LM head.
    names = [op["op"] for op in theirs]
    assert names.count(ATTENTION) == names.count(ARGMAX) == drafts
    tables = [op for op in theirs if op["op"] == "aten.embedding.default"]
    assert [op["args"]["weight"]["shape"][0] for op in tables] == [vocab] * drafts


def test_a_run_prices_a_verify_step_s_drafter_from_its_own_graph(tmp_path_factory):
    root, graphs = capture(1, tmp_path_factory)
    text = json.loads(CONFIG.read_text())
    config = SimpleNamespace(
        model=str(root / "model"),
        hf_config=SimpleNamespace(
            architectures=text["architectures"],
            text_config=SimpleNamespace(num_experts=text["text_config"]["num_experts"]),
        ),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_expert_parallel=False,
        enforce_eager=True,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
    )
    machine = MachineSpec.from_mapping(json.loads(MACHINE.read_text()))
    lookup = _op_graphs(str(root / "store"), config, "gfx942")
    cost = OpGraphBackend(machine, lookup, kv="bf16", drafts=1).price(VERIFY)
    [drafter] = [graph for key, graph in graphs.items() if key.endswith(" draft")]
    flat: dict = {}
    for op in _ops(drafter["region"]):
        if (w := work(op, VERIFY, 1, "bf16")) is not None:
            flat.setdefault(f"draft op.{op['op']}", []).append(price(w, machine))

    assert {t.name: t.seconds for t in cost.terms if t.draft} == {
        name: fold_seconds(s) for name, s in flat.items()
    }
    assert f"op.{SAMPLER}" in {t.name for t in cost.terms if not t.draft}


FINAL = BatchView((RequestShape(9, 9, False),), produces_output=True)


@pytest.mark.parametrize(
    "batch,shape",
    [(VERIFY, ("decode", 2, 2, True)), (FINAL, ("prefill", 9, 1, True))],
    ids=["verify", "final-chunk"],
)
def test_a_step_that_samples_reads_its_drafter_s_graph_as_draft_terms(batch, shape):
    asked = []

    def graph(*at, draft=False):
        asked.append((*at, draft))
        return g(16, 64, 64) if draft else g(4, 8, 8)

    cost = OpGraphBackend(SPEC, graph, kv="bf16", drafts=1).price(batch)

    assert asked == [(*shape, False), (*shape, True)]
    assert (cost.target_seconds, cost.draft_seconds) == (
        price(gemm(4, 8, 8), SPEC),
        price(gemm(16, 64, 64), SPEC),
    )


@pytest.mark.parametrize(
    "batch,drafted,named",
    [
        (view((1, 9, True)), g(1, 8, 8), "a step that decodes a row of other than 2"),
        (MIDDLE, g(1, 8, 8), "a step under a drafter that produces no output"),
        (VERIFY, None, "over 2 sequences for its drafter"),
    ],
    ids=["unverified-row", "middle-chunk", "uncaptured-drafter"],
)
def test_a_step_under_a_drafter_with_no_draft_to_price_is_refused_by_name(
    batch, drafted, named
):
    def graph(*at, draft=False):
        return drafted if draft else g(1, 8, 8)

    with pytest.raises(CostRefused, match=named):
        OpGraphBackend(SPEC, graph, kv="bf16", drafts=1).price(batch)


def test_a_population_captures_a_verify_step_at_its_width(tmp_path):
    record = _record(tmp_path, [{"rows": [[2, 90, True]] * 2, "rung": 2}])

    assert list(map(str, population_shapes([record]))) == ["decode B=2"]
