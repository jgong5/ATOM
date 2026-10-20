# SPDX-License-Identifier: MIT
"""`compass trace ops` on DeepSeek-V4-Pro with no device: one rank of the
nightly DPA-TP8 deployment at EP off and EP8, and TP8 without DP attention,
decode and prefill.

Under DP every peer runs this rank's shape, so what the MoE sees is eight
ranks' rows.
"""

import functools
import hashlib
import importlib
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import types

import pytest
from test_capture_ir import _decoder, _ops

from atom.compass.artifacts import ArtifactStore, Kind, Topology

TREE = pathlib.Path(__file__).resolve().parents[2]
# `deepseek-ai/DeepSeek-V4-Pro` at revision
# b5968e9190ef611bbf34a7229255be88a0e937c1, its config.json byte for byte.
CONFIG = pathlib.Path(__file__).with_name("deepseek_v4_pro_config.json")
CONFIG_SHA256 = "5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f569c810e"
LAYERS = json.loads(CONFIG.read_text())["num_hidden_layers"]
SHAPES = "--width 1 --width 2 --prefill 512 1 --prefill 512 2"
DPA_TP8 = " -- --kv_cache_dtype fp8 --enable-dp-attention"
EP8 = " --enable-expert-parallel"
DP = 8
MOE = "atom.model_ops.moe"
MORI = "mori.ops.dispatch_combine:EpDispatchCombineOp"
V4 = "atom.model_ops.v4_kernels."
DECODE = V4 + "paged_decode:_sparse_attn_v4_paged_decode_asm"
#: The attention's core kernel, one a layer, by phase.
CORE = {"decode": DECODE, "prefill": "aiter.pa_sparse_prefill_fp8_opus.default"}

_CAPTURES: dict = {}


def trace(root, argv):
    """`trace ops --tp 8 --arch gfx950` over `argv`, run in a process of its own
    on a model directory holding V4-Pro's config.json and a shard of junk:
    nothing may open the checkpoint."""
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    junk = root / "model" / "model-00001-of-00001.safetensors"
    junk.write_bytes(b"not a safetensors shard")
    command = ["trace", "ops", "--model", str(root / "model"), "--tp", "8"]
    command += ["--arch", "gfx950", "--store", str(root / "store"), *argv.split()]
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from atom.compass.cli import main; main(sys.argv[1:])",
            *command,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )


def capture(tmp_path_factory, argv):
    """`trace` over `argv`: each shape's published graph and declarations, by
    shape."""
    if argv in _CAPTURES:
        return _CAPTURES[argv]
    root = tmp_path_factory.mktemp("v4pro")
    done = trace(root, argv)
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
    _CAPTURES[argv] = graphs
    return graphs


def _pars(region):
    """Every `Par` the step runs, repeats expanded, each as its branches' op names."""
    if "par" in region:
        return [[[op["op"] for op in _ops(b)] for b in region["par"]]]
    if "repeat" in region:
        return _pars(region["repeat"]) * region["count"]
    return [par for item in region.get("seq", []) for par in _pars(item)]


def _rows(op):
    """The rows an exchange takes in and gives back."""
    sent = next(v for v in op["args"].values() if isinstance(v, dict) and "shape" in v)
    return sent["shape"][0], op["out"][0][0]


@pytest.mark.parametrize("ep", [False, True], ids=["ep_off", "ep8"])
@pytest.mark.parametrize(
    "shape, tokens",
    [("decode B=2", 2), ("prefill T=512 B=1", 512), ("prefill T=512 B=2", 512)],
)
def test_each_layer_runs_its_shared_expert_beside_the_moe_on_every_rank_s_rows(
    shape, tokens, ep, tmp_path_factory
):
    graph, _ = capture(tmp_path_factory, SHAPES + DPA_TP8 + (EP8 if ep else ""))[shape]

    widths = f"tp=1 dp={DP}" + (f" ep={DP}" if ep else "")
    assert graph["structure"].endswith(f" {widths} {shape.split()[0]} eager")
    # Every layer's shared expert runs on a side stream beside its routed MoE.
    pars = _pars(graph["region"])
    assert len(pars) == LAYERS
    for routed, shared in pars:
        assert "aiter.fused_moe_.default" in routed
        assert "aiter.silu_and_mul.default" in shared
    layers = _decoder(graph)
    assert len(layers) == LAYERS
    for layer in layers:
        names = [op["op"] for op in layer]
        # The attention runs its body: its kernels are the layer's own leaves.
        assert "aiter.v4_attention_with_output.default" not in names
        assert names.count(CORE[shape.split()[0]]) == 1
        assert names.count("aiter.fused_moe_.default") == 1
        exchanged = [op for op in layer if op["kind"] == "declared"]
        if ep:
            assert [op["op"] for op in exchanged] == [
                f"{MORI}.dispatch",
                f"{MORI}.combine",
            ]
            # Each names MORI's method, and its slots are that method's parameters
            # and the EP width its handle was built for.
            for op in exchanged:
                module, _, qualname = op["op"].partition(":")
                fn = functools.reduce(
                    getattr, qualname.split("."), importlib.import_module(module)
                )
                params = set(inspect.signature(fn).parameters)
                assert set(op["args"]) <= params | {"num_ep_ranks"}
        else:
            assert {op["op"].split(":")[0] for op in exchanged} == {MOE}
        # In: this rank's rows. Out of the dispatch or gather: every rank's.
        assert _rows(exchanged[0]) == (tokens, DP * tokens)


def test_a_dp_decode_at_width_1_gathers_one_row_from_each_rank(tmp_path_factory):
    graph, _ = capture(tmp_path_factory, SHAPES + DPA_TP8)["decode B=1"]

    rows = {
        (op["op"], *_rows(op))
        for layer in _decoder(graph)
        for op in layer
        if op["kind"] == "declared"
    }
    assert rows == {
        (f"{MOE}:all_gather_with_padding", 1, DP),
        (f"{MOE}:reduce_scatter_with_unpadding", DP, 1),
    }


def test_a_prefill_s_head_takes_each_sequence_s_last_token(tmp_path_factory):
    graphs = capture(tmp_path_factory, SHAPES + DPA_TP8)
    for seqs in (1, 2):
        graph, _ = graphs[f"prefill T=512 B={seqs}"]
        gemms = [
            op
            for op in _ops(graph["region"])
            if op["op"] == "aiter.gemm_a16w16.default"
        ]
        # Each layer's router and attention take every token; the head, after
        # them, one per sequence.
        taken = [op["args"]["A"]["shape"][0] for op in gemms]
        assert len(taken) > LAYERS
        assert taken == [512] * (len(taken) - 1) + [seqs]


def test_a_lifted_function_holds_a_mapping_and_a_host_object(tmp_path_factory):
    graph, _ = capture(tmp_path_factory, SHAPES + DPA_TP8)["decode B=2"]

    [write] = [
        op
        for op in _ops(graph["region"])
        if op["op"].endswith("paged_decode_indices:write_v4_paged_decode_indices")
    ]
    assert write["kind"] == "opaque_leaf"
    assert write["args"]["geometry"] == "<UnifiedPoolGeometry>"
    rows = write["args"]["dest_rows"]
    assert rows and all(isinstance(k, int) and "shape" in t for k, t in rows)


def test_two_attention_functions_are_leaves_at_their_call(tmp_path_factory):
    graph, _ = capture(tmp_path_factory, SHAPES + DPA_TP8)["decode B=2"]

    ops = _ops(graph["region"])
    names = {op["op"] for op in ops}
    assert V4 + "fused_compress:fused_compress_attn" in names
    # The aiter kernel the decode function dispatches is folded into it.
    assert "aiter.mla_decode_v4_asm.default" not in names
    for op in ops:
        if op["op"] == DECODE:
            assert op["kind"] == "opaque_leaf"
            # As called: the function trims it to the step's rows before its
            # launches.
            assert op["args"]["split_indptr"]["shape"][0] > 2 + 1


def test_tp8_without_dp_attention_reduces_in_each_layer_and_declares_nothing(
    tmp_path_factory,
):
    graphs = capture(
        tmp_path_factory, "--width 2 --prefill 512 1 -- --kv_cache_dtype fp8"
    )

    for shape in ("decode B=2", "prefill T=512 B=1"):
        graph, _ = graphs[shape]
        assert graph["structure"].endswith(f" tp=8 {shape.split()[0]} eager")
        layers = _decoder(graph)
        assert len(layers) == LAYERS
        for layer in layers:
            names = [op["op"] for op in layer]
            # The attention's output projection and the MoE each reduce.
            assert names.count("aiter.all_reduce_.default") == 2
            assert not [op for op in layer if op["kind"] == "declared"]


def test_a_step_split_into_two_micro_batches_is_refused_by_name(tmp_path):
    # ATOM splits a prefill under --enable-tbo from 8192 tokens on, its default.
    done = trace(
        tmp_path, "--prefill 512 1 --prefill 8192 1" + DPA_TP8 + EP8 + " --enable-tbo"
    )

    captured = re.findall(r"^captured (.*) in ", done.stdout, re.MULTILINE)
    # A prefill shape is captured twice: as a middle chunk, then a final one.
    assert captured == ["prefill T=512 B=1"] * 2, done.stderr[-4000:]
    assert done.returncode != 0
    assert (
        "prefill T=8192 B=1 runs two micro-batches under --enable-tbo"
        in done.stderr.splitlines()[-1]
    )


def test_the_attention_body_runs_as_a_real_step_then_the_batch_is_dummy_again(
    monkeypatch,
):
    import torch

    from atom.compass.capture.forward import _v4_attention
    from atom.utils.forward_context import Context, get_forward_context

    def fused_compress_attn():
        pass

    def _sparse_attn_v4_paged_decode_asm():
        pass

    # The model and its kernels import aiter, which needs a device arch: they
    # are stood in for, and so is the op the model registers.
    decode = types.SimpleNamespace(
        _sparse_attn_v4_paged_decode_asm=_sparse_attn_v4_paged_decode_asm
    )
    kernels = types.SimpleNamespace(paged_decode=decode)
    monkeypatch.setitem(sys.modules, "atom.model_ops.v4_kernels", kernels)
    monkeypatch.setattr(torch.ops.aiter, "v4_attention_with_output", None, False)
    context = Context(positions=None, is_dummy_run=True)
    monkeypatch.setattr(get_forward_context(), "context", context)
    seen = []

    def attention(x, positions, layer_name):
        seen.append(context.is_dummy_run)
        if layer_name == "raises":
            raise RuntimeError(layer_name)

    v4 = types.SimpleNamespace(
        v4_attention_with_output=attention, fused_compress_attn=fused_compress_attn
    )
    with _v4_attention(v4, recorder=None):
        torch.ops.aiter.v4_attention_with_output(None, None, "returns")
        assert context.is_dummy_run
        with pytest.raises(RuntimeError):
            torch.ops.aiter.v4_attention_with_output(None, None, "raises")
        assert context.is_dummy_run
    assert seen == [False, False]
