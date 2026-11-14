# SPDX-License-Identifier: MIT
"""aiter's fused-MoE leaf of a layer that fuses its shared expert into aiter's
top-k table: the capture states the layer's top k and fused shared count on
it, and the op-level work routes the top k on a padded gather's real rows and
the shared expert on every row, under expert parallelism the rows dealt to
the rank. Each price is checked against a count by hand, and the capture is a
DeepSeek-V3 of four layers traced under DP attention, its experts unsharded
and over eight ranks.
"""

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
from test_capture_ir import MAIN, _ops
from test_moe_gather_pad_rows import gathered
from test_op_work import I32, t
from test_op_work_v4 import F32, close, moe

from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.backends.op_work import work
from atom.compass.backends.work import Work, activated_experts, max_rank_load

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).with_name("deepseek_v3_config.json")
#: The layer's top 2 and its one fused shared expert, as the capture states them.
COUNTS = {"top_k": 2, "fused_shared_experts": 1}


def fused(rows, local, width, mask=None):
    """`moe`'s leaf of `local` experts a rank, the shared one among them, with
    `width` top-k ids and weights a row, stating `COUNTS`."""
    op = moe(rows, local, mask)
    ids = {"topk_weight": t([rows, width], F32), "topk_ids": t([rows, width], I32)}
    return op | {"args": op["args"] | ids, "stated": COUNTS}


def priced(rows, picks, groups, ran, width, mask=0):
    """`moe`'s gate-up and down products over `picks` routes at fp4, each of
    `groups` experts' weights and scales read once, and the bf16 rows in and
    out, their `width` ids and weights and the mask at the share `ran` of the
    `rows`."""
    flops = 2 * picks * 64 * 64 + 2 * picks * 64 * 32
    io = (rows * (64 * 2 + width * 8 + 64 * 2) + mask * 4) * ran / rows
    return Work({"fp4": flops}, io + groups * ((64 * 64 + 32 * 64) * 0.5 + 64 * 3))


def test_the_shared_expert_takes_every_gathered_row_and_the_top_k_the_real_ones():
    # 16 gathered rows, 5 real: their 10 routes and all 16 rows through the
    # shared expert, 26, over the experts the 5 reach and the shared one.
    hit = activated_experts(16, 2, 5) + 1

    got = work(fused(16, 17, 3), gathered(16, 5), 1, "bf16")

    close(got, priced(16, 26, hit, 16, 3))


def test_under_expert_parallelism_a_rank_runs_the_shared_expert_on_its_dealt_rows():
    # EP4, four routed experts and the shared one a rank, the ids one wider
    # for the column that reaches no expert: the busiest rank's share of the 5
    # real rows' routes, and 4 of the 16 rows dealt round robin.
    load, hit = max_rank_load(5, 2, 16, 4), activated_experts(16, 2, 5) / 4 + 1

    got = work(fused(16, 5, 4, mask=18), gathered(16, 5), 1, "bf16", 4)

    close(got, priced(16, load + 4, hit, max(load / 2, 4), 4, mask=18))


#: Expert parallelism over the DP group's gather: MoRI off, so no all-to-all.
EP8_GATHERED = ("--enable-expert-parallel",), {"ATOM_DISABLE_MORI_EP": "1"}
_CAPTURES: dict = {}


def _decode_moes(tmp_path_factory, atom_args=(), env=None):
    """The fused-MoE leaves of a DeepSeek-V3 of four layers, the first dense,
    traced at a decode of 4 rows a rank under DP attention over 8 ranks with
    ATOM's `atom_args`, its shared expert fused into aiter's top-k table."""
    if atom_args in _CAPTURES:
        return _CAPTURES[atom_args]
    root = tmp_path_factory.mktemp("v3")
    (root / "model").mkdir()
    config = json.loads(CONFIG.read_text())
    config["num_hidden_layers"], config["first_k_dense_replace"] = 4, 1
    (root / "model" / "config.json").write_text(json.dumps(config))
    (root / "model" / "model-00001-of-00001.safetensors").write_bytes(b"not a shard")
    command = ["trace", "ops", "--model", str(root / "model"), "--tp", "8"]
    command += ["--width", "4", "--arch", "gfx950", "--store", str(root / "store")]
    command += ["--", "--kv_cache_dtype", "fp8", "--enable-dp-attention", *atom_args]
    done = subprocess.run(
        [sys.executable, "-c", MAIN, *command],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={
            **os.environ,
            "PYTHONPATH": str(TREE),
            "ATOM_FUSE_SHARED_EXPERT": "1",
            **(env or {}),
        },
    )
    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert tokens, f"rc {done.returncode}\n{done.stderr[-4000:]}"
    [token] = tokens
    store = ArtifactStore(str(root / "store"))
    entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
    rank = next(iter(Topology().ranks()))
    graph = json.loads(entry.read_member("graph", rank, "json"))
    moes = [op for op in _ops(graph["region"]) if "fused_moe" in op["op"]]
    _CAPTURES[atom_args] = moes
    return moes


@pytest.mark.parametrize(
    "atom_args, env, width",
    [((), None, 9), (*EP8_GATHERED, 10)],
    ids=["ep_off", "ep8_gathered"],
)
def test_a_traced_fused_shared_layer_states_its_top_k_and_shared_count(
    tmp_path_factory, atom_args, env, width
):
    # The top 8 and the shared expert, and under EP the column the rows
    # dealt to another rank take, which reaches no expert here.
    moes = _decode_moes(tmp_path_factory, atom_args, env)

    assert len(moes) == 3
    for op in moes:
        assert op["stated"] == {"top_k": 8, "fused_shared_experts": 1}
        assert op["args"]["topk_ids"]["shape"] == [32, width]


def test_a_traced_leaf_runs_its_real_rows_top_k_and_every_gathered_row_s_shared(
    tmp_path_factory,
):
    # 32 gathered rows, 9 of them real: 9 x 8 top-k routes and 32 shared ones,
    # each a gate-up of `up` and a down of half of it over the hidden width.
    [op, *_] = _decode_moes(tmp_path_factory)
    _, up, hidden = op["args"]["w1"]["shape"]

    got = work(op, gathered(32, 9), 1, "fp8").flops_by_format

    assert got == {"fp8": pytest.approx((9 * 8 + 32) * 3 * up * hidden)}
