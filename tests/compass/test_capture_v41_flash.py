# SPDX-License-Identifier: MIT
"""`compass trace ops` on DeepSeek-V4.1-Flash with no device: one rank of TP4,
with and without whole-expert EP4, decode and prefill at gfx1250; and decode
at gfx950, where its indexer's FP8 scorer launches a compiled Triton kernel
directly.

V4.1's attention, Engram and the step's staging run their bodies on a real
step's batch, so their kernels are leaves; its sparse attention states its
layer's window, compression ratio and top k, and its indexer's scorer the
ratio and the candidates it scores among.
"""

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
from test_capture_ir import MAIN, _decoder, _ops

from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.capture.forward import V41_ATTENTION, V41_OPENED, V41_SCORERS

TREE = pathlib.Path(__file__).resolve().parents[2]
# `deepseek-ai/DeepSeek-V4.1-Flash` at the revision its `reference_manifest.json`
# names, its config.json byte for byte.
CONFIG = TREE / "tests" / "models" / "deepseek_v41" / "fixtures" / "config.json"
CONFIG_SHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
TEXT = json.loads(CONFIG.read_text())["text_config"]
LAYERS = TEXT["num_hidden_layers"]
RATIOS = TEXT["compress_ratios"]
SHAPES = "--width 16 --prefill 512 1"
ARGS = " -- --kv_cache_dtype bf16 --index-cache-dtype fp8"
EP4 = " --enable-expert-parallel"
TP = 4
MOE = "aiter.fused_moe_.default"
#: gfx950's FP8 scorer: aiter's function that launches its compiled kernel.
COMPILED = "aiter.ops.triton.attention.pa_mqa_logits:deepgemm_fp8_paged_mqa_logits"

_CAPTURES: dict = {}


def trace(root, argv, arch):
    """`trace ops --tp 4 --arch <arch>` over `argv`, in a process of its own, on
    a model directory holding V4.1-Flash's config.json and a shard of junk:
    nothing may open the checkpoint."""
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    junk = root / "model" / "model-00001-of-00001.safetensors"
    junk.write_bytes(b"not a safetensors shard")
    command = ["trace", "ops", "--model", str(root / "model"), "--tp", str(TP)]
    command += ["--arch", arch, "--store", str(root / "store"), *argv.split()]
    return subprocess.run(
        [sys.executable, "-c", MAIN, *command],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )


def capture(tmp_path_factory, argv, arch="gfx1250"):
    """`trace` over `argv` at `arch`: each shape's published graph and
    declarations, by shape, a prefill's final chunk."""
    if (argv, arch) in _CAPTURES:
        return _CAPTURES[argv, arch]
    root = tmp_path_factory.mktemp("v41")
    done = trace(root, argv, arch)
    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert tokens, f"rc {done.returncode}\n{done.stderr[-4000:]}"
    store = ArtifactStore(str(root / "store"))
    rank = next(iter(Topology().ranks()))
    graphs = {}
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph, declared = (
            json.loads(entry.read_member(stem, rank, "json"))
            for stem in ("graph", "declared")
        )
        graphs[graph["shape"]] = graph, declared
    _CAPTURES[argv, arch] = graphs
    return graphs


def geometry(layer):
    """What layer `layer`'s sparse attention states."""
    ratio = RATIOS[layer]
    top = TEXT["index_topk"] if ratio else None
    return {
        "window_size": TEXT["sliding_window"],
        "compress_ratio": ratio,
        "index_topk": top,
    }


def scored(layer):
    """What layer `layer`'s indexer scorer states: past the candidate source,
    its candidates."""
    cap = TEXT["candidate_topk_blocks"] * TEXT["candidate_block_size"]
    past = layer > TEXT["candidate_source_layer_id"]
    return {"compress_ratio": RATIOS[layer], "index_candidates": cap if past else None}


@pytest.mark.parametrize("ep", [False, True], ids=["ep_off", "ep4"])
def test_each_decode_layer_attends_through_its_own_leaves_stating_its_geometry(
    ep, tmp_path_factory
):
    graph, _ = capture(tmp_path_factory, SHAPES + ARGS + (EP4 if ep else ""))[
        "decode B=16"
    ]

    widths = f"tp={TP}" + (f" ep={TP}" if ep else "")
    assert graph["structure"].endswith(f" {widths} decode eager")
    names = {op["op"] for op in _ops(graph["region"])}
    # The guarded ops ran their bodies: none is a leaf of its own.
    assert not names & {f"aiter.{name}.default" for name in V41_OPENED}
    layers = _decoder(graph)
    assert len(layers) == LAYERS
    for index, layer in enumerate(layers):
        attention = [op for op in layer if op["op"] == V41_ATTENTION]
        assert [op["stated"] for op in attention] == [geometry(index)]
        scorers = [op for op in layer if op["op"] in V41_SCORERS]
        if index in TEXT["index_source_layer_ids"]:
            assert [op["stated"] for op in scorers] == [scored(index)]
        else:
            assert not scorers


@pytest.mark.parametrize("ep", [False, True], ids=["ep_off", "ep4"])
def test_a_prefill_attends_every_layer_and_its_tail_through_their_own_leaves(
    ep, tmp_path_factory
):
    graph, _ = capture(tmp_path_factory, SHAPES + ARGS + (EP4 if ep else ""))[
        "prefill T=512 B=1"
    ]

    ops = _ops(graph["region"])
    attention = [op for op in ops if op["op"] == V41_ATTENTION]
    # The first decoder layers on every row, the rest on the replayed tail.
    assert len(attention) == LAYERS
    assert min(op["args"]["q"]["shape"][0] for op in attention) < 512
    assert all(
        op["stated"]["window_size"] == TEXT["sliding_window"] for op in attention
    )
    # The tail is gathered by one launch of every tensor it takes.
    gathers = [op for op in ops if op["op"].endswith("gather:gather_rows")]
    assert gathers and all(len(op["args"]["tensors"]) > 1 for op in gathers)


def test_under_ep4_a_rank_holds_a_quarter_of_the_experts_whole(tmp_path_factory):
    experts = TEXT["n_routed_experts"]
    held = {}
    for ep in ("", EP4):
        graph, _ = capture(tmp_path_factory, SHAPES + ARGS + ep)["decode B=16"]
        moes = [op for op in _ops(graph["region"]) if op["op"] == MOE]
        assert len(moes) == LAYERS
        held[ep] = {op["args"]["w1"]["shape"][0] for op in moes}

    assert held == {"": {experts}, EP4: {experts // TP}}


def test_gfx950_s_fp8_scorer_lifts_its_compiled_kernel_launch_to_its_function(
    tmp_path_factory,
):
    graph, _ = capture(tmp_path_factory, "--width 16" + ARGS + EP4, "gfx950")[
        "decode B=16"
    ]

    scorers = [op for op in _ops(graph["region"]) if op["op"] == COMPILED]
    assert len(scorers) == len(TEXT["index_source_layer_ids"])
    for op in scorers:
        # Its arguments at call entry: the query, the plane and the weights.
        assert op["args"]["q_fp8"]["shape"][0] == 16
        assert op["args"]["q_fp8"]["dtype"] == "torch.float8_e4m3fn"
        assert {"kv_cache", "weights", "out_logits"} <= set(op["args"])
