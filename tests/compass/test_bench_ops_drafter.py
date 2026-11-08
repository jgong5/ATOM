# SPDX-License-Identifier: MIT
"""`compass measure ops` on a speculative drafter's graphs, with no device.

With the stand-in bench and graphs of `test_bench_ops_plan.py`, each part of a
drafter's step holds a leaf of its own: `--plan` measures them beside the
target's leaves, and `--coverage` reads each step's parts as a run prices them.
Beside the stand-in runner of `test_bench_ops.py`, a leaf of the drafter's
alone counts the rows of its own pass as ATOM's `batch_info` states them, and
a drafter's MoE draws its expert ids over its own experts. DeepSeek-V4.1-Flash's
DSpark5 drafter, traced at #2086's trace run b's arguments where a capture
holds them, reaches each leaf's launch off the card (`v41_ops_rehearsal.py`).
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops import TREE, _graph, _leaf, _Runner, _t
from test_bench_ops_plan import CONFIG, _graph_at, bench  # noqa: F401  a fixture
from test_capture_ir import MAIN
from test_capture_v41_flash import CONFIG as V41
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture

from atom.compass.artifacts import ArtifactRefusal, Rule
from atom.compass.backends.op_graph import KV
from atom.compass.bench import ops
from atom.utils import forward_context as fctx

#: A block drafter of one draft: a decode row verifies two tokens.
DRAFTING = SimpleNamespace(**vars(CONFIG))
DRAFTING.speculative_config = SimpleNamespace(
    num_speculative_tokens=1, use_dspark=lambda: True
)
#: Each part's own leaf: its propose, and its draft-KV pass, which is a middle
#: chunk's drafter's part too.
OWN = {(True, True): "aiter.propose", (True, KV): "aiter.write_kv"}
OWN[False, True] = OWN[True, KV]


def _parted(store, config, shape, arch=None, output=True, draft=False):
    """The target's graph, or a part of the drafter's, as `trace ops` keeps
    them: a decode has no middle chunk's."""
    if not draft:
        return _graph_at(store, config, shape, arch)
    if shape.phase == "decode" and not output:
        raise ArtifactRefusal(Rule.RESOLUTION, f"no op_graph holds {shape}", "")
    rows = shape.tokens if shape.phase == "prefill" else shape.seqs
    own = _leaf(OWN[output, draft], ("x", _t([rows, 8], "activation")))
    return _graph(own, layers=1)


def _records(path, *steps):
    """A record directory of `steps`, each `(rows, rung, produces_output)`."""
    (path / "steps").mkdir(parents=True)
    (path / "run.json").write_text("{}")
    lines = [
        {"step": i, "is_dummy_run": False, "rows": rows, "rung": rung}
        | {"produces_output": output}
        for i, (rows, rung, output) in enumerate(steps)
    ]
    (path / "steps" / "dp0.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    return str(path)


def test_a_plan_measures_a_drafter_s_parts_and_coverage_reads_each_step_s(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(ops, "graph", _parted)
    steps = _records(
        tmp_path / "steps",
        ([[2, 1000, True], [2, 3000, True]], 2, True),
        ([[2, 12000, True]], 1, True),
        ([[40, 40, False]], None, True),  # a final chunk
        ([[150, 150, False]], None, False),  # a middle chunk
    )
    store, root, key, run = bench
    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(plan=[steps], store=root, llc_mib=256)

    ops.plan(store, key, DRAFTING, run, args, "compass measure ops --plan", out)

    records = [json.loads(x) for x in (out / "plan.jsonl").read_text().splitlines()]
    shapes = {r["shape"] for r in records if r["op"] == "aiter.propose"}
    assert shapes == {r["shape"] for r in records if r["op"] == "aiter.gemm"}
    assert {r["op"] for r in records} == {"aiter.attn", "aiter.gemm", *OWN.values()}
    report = ops.coverage(store, root, DRAFTING, [steps], records, "gfx942")
    # Each step reads its target's and its drafter's graphs; one that samples
    # also its draft-KV pass's; a middle chunk its drafter's alone.
    assert report["by part"] == {
        "target": {"steps": 4, "covered": 4},
        "draft": {"steps": 4, "covered": 4},
        "draft_kv": {"steps": 3, "covered": 3},
    }
    assert report["covered"] == 4


class _Drafting(_Runner):
    """`_Runner` whose model routes 6 of 384 experts, whose drafter's routes 3
    of 128, and whose step states its rows as ATOM's `batch_info` does: 4
    sequences of 6 tokens, 20 drafted."""

    def __init__(self):
        super().__init__()
        self.model.add_module("moe", _Moe(384, 6))
        self.drafter = SimpleNamespace(model=_Moe(128, 3))
        info = torch.tensor([4, 24, 20], dtype=torch.int32)
        self.forward_vars["batch_info"] = SimpleNamespace(np=info.numpy(), gpu=info)

    def prepare_model(self, batch):
        prepared = super().prepare_model(batch)
        context = fctx.get_forward_context().context
        context.running_tokens = 24
        context.scheduled_tokens_gpu = self.forward_vars["batch_info"].gpu[1:2]
        return prepared


class _Moe(torch.nn.Module):
    def __init__(self, experts, k):
        super().__init__()
        self.global_num_experts, self.top_k = experts, k


def test_a_drafter_s_pass_of_its_own_rows_counts_them_as_batch_info_states():
    stub = ops.Stub(_Drafting(), 10_000)
    count = ("num_local_tokens", _t([1], "metadata", dtype="torch.int32"))
    moe = {
        rows: _leaf(
            "aiter.fused_moe_.default", ("x", _t([rows, 8], "activation")), count
        )
        for rows in (24, 20)
    }
    stub.leaves = ops.leaves(_graph(*moe.values(), layers=1)["region"])
    sigs = {op["args"]["x"]["shape"][0]: sig for sig, (op, _) in stub.leaves.items()}
    stub.leaves[sigs[20]][0]["drafted"] = True

    stub._prepare((0, 0))

    bound = {sig: view for (sig, _), (_, view) in stub.placed[0][1].items()}
    # the drafted rows' count, not the target's tokens
    assert {rows: bound[sig].item() for rows, sig in sigs.items()} == {24: 24, 20: 20}


def test_a_drafter_s_moe_draws_its_ids_over_its_own_experts():
    stub = ops.Stub(_Drafting(), 10_000)

    assert stub.moe == [(128, 3), (384, 6)]
    for experts, k in stub.moe:
        spec = {"shape": [20, k], "stride": [k, 1], "dtype": "torch.int32"}
        ids = stub._route(spec | {"role": "activation"}, 0)
        assert ids.shape == (20, k) and int(ids.max()) < experts


#: Trace run b's engine arguments past the model and TP, in the setup a capture
#: holds: a BF16 KV cache, an FP8 index plane, and V4.1's mono decode off.
CELL = (
    "--kv_cache_dtype bf16 --index-cache-dtype fp8 --block-size 16"
    " --max-num-seqs 128 --max-num-batched-tokens 16384"
    " --attn-prefill-chunk-size 16384 --enable_prefix_caching"
    " --state-checkpoint-interval-tokens 8192 --long-prefill-token-threshold 8192"
    " --method dspark --num-speculative-tokens 5 --spec-decode-acceptance-length 3.51"
)
SHAPES = ["--width", "4", "--prefill", "124", "1"]
#: Kernels only the drafter launches: its draft-KV pass's tail, its block's
#: attention and its Markov sampler.
OWN_KERNELS = {
    "atom.model_ops.deepseek_v41.dspark:fused_draft_kv_tail",
    "atom.model_ops.sparse_attn_v4:_sparse_attn_triton",
    "atom.model_ops.dspark_markov_sample:_launch",
}


def _run(*argv):
    env = {**os.environ, "PYTHONPATH": str(TREE), "ATOM_MONO_ENABLE": "0"}
    done = subprocess.run(
        [sys.executable, *map(str, argv)],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        cwd=TREE,
        env=env,
    )
    assert done.returncode == 0, done.stderr[-4000:]


@pytest.fixture(scope="module")
def rehearsed(model_dir, tmp_path_factory):  # noqa: F811
    """Each leaf of the drafter's graphs alone at a decode of 4 rows of 1024
    tokens and a prefill of 124 past as many, traced at TP2 on gfx950, as
    the rehearsal records it."""
    root = tmp_path_factory.mktemp("drafter")
    (model := root / "model").mkdir()
    shutil.copy(V41, model / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(model_dir / name, model / name)
    store, tp = root / "store", ["--tp", "2"]
    trace = ["trace", "ops", "--model", model, *tp, "--arch", "gfx950"]
    _run("-c", MAIN, *trace, "--store", store, *SHAPES, "--", *CELL.split())
    harness = pathlib.Path(__file__).with_name("v41_ops_rehearsal.py")
    point = [*SHAPES, "--context", "1024", "--llc-mib", "256", "--store", store]
    atom = ["--model", model, "-tp", "2", "--enforce-eager", "--level", "0"]
    flags = ["--drafter", "--arch", "gfx950"]
    _run(harness, root / "drafter.json", *flags, *point, "--", *atom, *CELL.split())
    return json.loads((root / "drafter.json").read_text())


def test_each_leaf_of_a_drafter_s_graphs_reaches_its_launch(rehearsed):
    """Each leaf of the drafter's propose and draft-KV pass, and of a middle
    chunk's drafter, at both shapes, its kernels its own among them, each
    index, slot and count it binds inside what it indexes."""
    assert {r["shape"] for r in rehearsed} == {"decode B=4", "prefill T=124 B=1"}
    assert OWN_KERNELS <= {r["op"] for r in rehearsed}
    assert [r for r in rehearsed if r["verdict"] != "reaches launch"] == []
