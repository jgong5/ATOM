# SPDX-License-Identifier: MIT
"""`compass trace ops` on DeepSeek-V4.1-Flash with no device: one rank of TP4,
with and without whole-expert EP4, decode and prefill at gfx1250; and decode
at gfx950, where its indexer's FP8 scorer launches a compiled Triton kernel
directly, and where with expert parallelism off V4.1's mono decode binds and
builds its width as its deployment does.

V4.1's attention, Engram and the step's staging run their bodies on a real
step's batch, so their kernels are leaves; its sparse attention states its
layer's window, compression ratio and top k, and its indexer's scorer the
ratio and the candidates it scores among; the mono decode's K2, which runs
both, states them all and its experts' unpadded width.
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
from atom.compass.capture.forward import (
    V41_ATTENTION,
    V41_LAYER_POST,
    V41_OPENED,
    V41_SCORERS,
)

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
#: A decode V4.1's mono decode serves: gfx950, expert parallelism off.
MONO = "--width 16" + ARGS
#: Two such decodes and a prefill, which mono never serves.
MONOS = "--width 16 --width 8 --prefill 512 1" + ARGS
#: Each run before a capture: the mono runtime's bind check refuses, its bind
#: fails, its build or its kernel list fails, or a built kernel needs scratch.
BIND = (
    "from atom.mono.runtime import consensus\n"
    "agreed = consensus.bind_agreed\n"
    "def bind():\n"
    "    raise {}('an injected bind {}')\n"
    "consensus.bind_agreed = lambda _, group: agreed(bind, group)\n"
)
BIND_REFUSAL = BIND.format("consensus.MonoUnsupported", "refusal")
BIND_FAILURE = BIND.format("RuntimeError", "failure")
BUILD_FAILURE = (
    "from atom.mono.runtime import compile as built\n"
    "def compile_kernels(kernels):\n"
    "    raise RuntimeError('an injected build failure')\n"
    "built.compile_kernels = compile_kernels\n"
)
KERNELS = (
    "from atom.mono.runtime import widths\n"
    "init = widths.WidthBuilds.__init__\n"
    "def kernels(built):\n"
    "    raise RuntimeError('an injected kernel-list failure')\n"
    "widths.WidthBuilds.__init__ = lambda s, b, _, g, w: init(s, b, kernels, g, w)\n"
)
SCRATCH = (
    "from atom.mono.runtime import compile as built\n"
    "built.scratch_bytes = lambda launcher: 64\n"
)
#: Each mono launcher, by its arguments that hold a buffer its runner keeps for
#: the run: the scratch, the mailboxes' peer table and step state, the step
#: metadata and the row buffers K1 and K2 write.
HELD = {
    "atom.mono.runtime.step_begin:_launch_step_begin": "scratch peers state",
    "atom.models.deepseek_v41.mono.step_meta:_launch_step_meta": "ring_rel key_meta",
    "atom.models.deepseek_v41.mono.kernels.attn_pre:launch_attn_pre": (
        "scratch ring_rows q_out normed qr_out qrs_out"
    ),
    "atom.models.deepseek_v41.mono.kernels.layer_post:launch_layer_post": (
        "scratch kmeta q ilog ibmax normed out"
    ),
}

_CAPTURES: dict = {}


def trace(root, argv, arch, prelude=""):
    """`trace ops --tp 4 --arch <arch>` over `argv`, in a process of its own
    that runs `prelude` first, on a model directory holding V4.1-Flash's
    config.json and a shard of junk: nothing may open the checkpoint."""
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    junk = root / "model" / "model-00001-of-00001.safetensors"
    junk.write_bytes(b"not a safetensors shard")
    command = ["trace", "ops", "--model", str(root / "model"), "--tp", str(TP)]
    command += ["--arch", arch, "--store", str(root / "store"), *argv.split()]
    return subprocess.run(
        [sys.executable, "-c", prelude + MAIN, *command],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )


def capture(tmp_path_factory, argv, arch="gfx1250", prelude=""):
    """`trace` over `argv` at `arch` after `prelude`: each shape's published
    graph and declarations, by shape, a prefill's final chunk."""
    if (argv, arch, prelude) in _CAPTURES:
        return _CAPTURES[argv, arch, prelude]
    root = tmp_path_factory.mktemp("v41")
    graphs = published(root, trace(root, argv, arch, prelude))
    _CAPTURES[argv, arch, prelude] = graphs
    return graphs


def published(root, done):
    """What `done`, a `trace` under `root`, published: each shape's graph and
    declarations, by shape."""
    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert (
        done.returncode == 0 and tokens
    ), f"rc {done.returncode}\n{done.stderr[-4000:]}"
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


def test_gfx950_s_mono_decode_binds_builds_its_width_and_meets_its_kernels(tmp_path):
    done = trace(tmp_path, MONO, "gfx950")

    log = done.stdout + done.stderr
    # The width's kernels compiled for the declared arch, as the deployment's.
    assert re.search(r"V4\.1 mono: \d+ width-16 kernels compiled at gfx950,", log), log
    # The step ran mono, not the original path: its leaves launch mono kernels.
    graph, _ = published(tmp_path, done)["decode B=16"]
    mono = r"atom\.(mono|models\.deepseek_v41\.mono)\.[\w.]+:\w"
    assert [op for op in _ops(graph["region"]) if re.match(mono, op["op"])]


def roles(value):
    """The roles of the tensors argument `value` holds."""
    if isinstance(value, dict):
        return {value["role"]} if "role" in value else roles([*value.values()])
    return set().union(*map(roles, value)) if isinstance(value, list) else set()


def test_gfx950_s_mono_leaves_hold_the_buffers_their_runner_keeps_as_metadata(
    tmp_path_factory,
):
    graph, _ = capture(tmp_path_factory, MONO, "gfx950")["decode B=16"]

    leaves = [op for op in _ops(graph["region"]) if op["op"] in HELD]
    assert {op["op"] for op in leaves} == set(HELD)
    for op in leaves:
        # The runner is made in the model's first forward and lives the run:
        # its buffers are metadata, never activations a bench fills at random.
        held = {slot: roles(op["args"][slot]) for slot in HELD[op["op"]].split()}
        assert held == {slot: {"metadata"} for slot in held}, op["op"]


def test_gfx950_s_mono_k2_states_its_layer_s_geometry(tmp_path_factory):
    graph, _ = capture(tmp_path_factory, MONO, "gfx950")["decode B=16"]

    posts = [op for op in _ops(graph["region"]) if op["op"] == V41_LAYER_POST]
    assert len(posts) == LAYERS
    for index, op in enumerate(posts):
        # Mono calls no layer's forward: its runner's call states the layer's.
        indexed = index in TEXT["index_source_layer_ids"]
        cap = scored(index)["index_candidates"] if indexed else None
        width = TEXT["moe_intermediate_size"] // TP
        stated = {"index_candidates": cap, "expert_width": width}
        assert op["stated"] == {**geometry(index), **stated}
        assert op["args"]["iact"] == indexed


@pytest.mark.parametrize(
    "prelude, refusal",
    [
        (
            BIND_FAILURE,
            "V4.1 mono decode failed to bind: RuntimeError: an injected bind failure",
        ),
        (
            BUILD_FAILURE,
            (
                "V4.1 mono declined its width-16 kernels: "
                "RuntimeError('an injected build failure')"
            ),
        ),
        (
            KERNELS,
            (
                "V4.1 mono declined its width-16 kernels: "
                "RuntimeError('an injected kernel-list failure')"
            ),
        ),
    ],
    ids=["bind", "build", "kernels"],
)
def test_a_mono_failure_the_capture_makes_stops_it_by_name(prelude, refusal, tmp_path):
    done = trace(tmp_path, MONO, "gfx950", prelude)

    assert done.returncode and "published" not in done.stdout
    assert done.stderr.strip().splitlines()[-1] == f"compass trace ops: {refusal}"


@pytest.mark.parametrize(
    "prelude, declined",
    [
        (
            BIND_REFUSAL,
            lambda w: ("V4.1 mono decode", None, "an injected bind refusal"),
        ),
        (SCRATCH, lambda w: ("V4.1 mono", w, "a kernel needs 64 B of scratch a lane")),
    ],
    ids=["bind_check", "scratch"],
)
def test_a_mono_refusal_the_deployment_makes_too_is_declared_by_the_graphs_it_turns_away(
    prelude, declined, tmp_path_factory
):
    graphs = capture(tmp_path_factory, MONOS, "gfx950", prelude)

    assert len(graphs) == 3
    for shape, (graph, declared) in graphs.items():
        # Each the original path; a prefill is never mono's, so it declares none.
        ops = _ops(graph["region"])
        assert not [op for op in ops if ".mono." in op["op"]]
        assert len([op for op in ops if op["op"] == V41_ATTENTION]) == LAYERS
        width = int(shape.split("B=")[1]) if shape.startswith("decode") else None
        why = [dict(zip(("runner", "width", "why"), declined(width)))]
        assert declared.get("mono") == (why if width else None), shape
