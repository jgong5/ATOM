# SPDX-License-Identifier: MIT
"""The MI455 cells under `compass trace ops`: DeepSeek-V4.1-Flash at TP4 with
DSpark5 and a forced acceptance, with and without whole-expert EP4, in the
cells' environment, priced by the op-level backend a run builds; and a card
declared as it reads in place of its arch's device.
"""

import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_capture_ir import MAIN, _ops
from test_capture_v41_flash import ARGS, CONFIG, EP4, TP, TREE
from test_op_work_v41 import MI455

from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.backends import BatchView, RequestShape
from atom.compass.capture import driverless, forward

CELL = ARGS + " --method dspark --num-speculative-tokens 5"
CELL += " --spec-decode-acceptance-length 3"
FORCED = "atom.model_ops.rejection_sampler:rejection_sample"
STEPS = [
    BatchView((RequestShape(6, 1024, True),) * 4, produces_output=True),
    BatchView((RequestShape(512, 512, False),), produces_output=True),
    BatchView((RequestShape(512, 512, False),), produces_output=False),
]
#: MI455 A0 as #2141's probe read it, its arch with the card's features.
A0 = {"index": 0, "arch": "gfx1250", "revision": 0}
A0["props"] = vars(driverless._props("gfx1250")) | {
    "multi_processor_count": 192,
    "total_memory": 309237645312,
    "gcnArchName": "gfx1250:sramecc+:xnack-",
}


def _trace(root, argv, where=("--arch", "gfx1250"), prelude="") -> dict:
    """`trace ops --tp 4` at `where` over `argv` into `root`, after `prelude`,
    in a process of its own on V4.1-Flash's config.json: each graph and its
    declarations by key."""
    (root / "model").mkdir(parents=True)
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    command = ["trace", "ops", "--model", str(root / "model"), "--tp", str(TP)]
    command += [*where, "--store", str(root / "store"), *argv.split()]
    done = subprocess.run(
        [sys.executable, "-c", prelude + MAIN, *command],
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert done.returncode == 0 and tokens, done.stderr[-4000:]
    store, rank = ArtifactStore(str(root / "store")), Topology().ranks()[0]
    graphs = {}
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph, declared = (
            json.loads(entry.read_member(stem, rank, "json"))
            for stem in ("graph", "declared")
        )
        graphs[graph["structure"], graph["shape"]] = graph, declared
    return graphs


@pytest.mark.parametrize("ep", ["", EP4], ids=["cell_b", "cell_a"])
def test_a_cell_s_forced_acceptance_traces_and_prices(ep, tmp_path, monkeypatch):
    # The cells' environment: aiter shares a rank set's communicators.
    monkeypatch.setenv("AITER_REUSE_IDENTICAL_COMM_GROUPS", "1")
    graphs = _trace(tmp_path, "--width 4 --prefill 512 1" + CELL + ep)

    decode = next(k for k in graphs if k[0].endswith("decode eager"))
    target = graphs[decode][0]
    # A decode samples its rows by the forced draw, at one seed.
    forced = [op for op in _ops(target["region"]) if op["op"] == FORCED]
    assert [op["args"]["synthetic_step"] for op in forced] == [0]
    config = forward._config(str(tmp_path / "model"), TP, (CELL + ep).split()[1:])
    quantization = json.loads(target["structure"].split()[3])["quantization"]
    config.quant_config = SimpleNamespace(compute_hash=lambda: quantization)
    store = {"store": str(tmp_path / "store")}
    backend = compass_run.op_graph(store, config, MI455, ep=TP if ep else 1)
    for step in STEPS:
        cost = backend.price(step)
        assert cost.target_seconds > 0 and cost.draft_seconds > 0


def test_a_card_is_declared_as_it_reads_at_its_arch_s_key(tmp_path):
    read = "from atom.compass.capture import driverless\n"
    read += f"driverless.card = {{0: {A0!r}}}.get\n"
    on_card = _trace(tmp_path / "card", "--width 4" + CELL, ["--gpu", "0"], read)
    declared = _trace(tmp_path / "arch", "--width 4" + CELL)

    # One key: the card's arch, its features dropped.
    assert on_card.keys() == declared.keys()
    for key, (graph, stated) in on_card.items():
        assert stated["card"] == A0 and stated["flags"]["CU_NUM"] == "192"
        # What a stepping and a CU count change in this stack: nothing.
        assert graph["region"] == declared[key][0]["region"]


def test_a_card_s_stepping_is_its_own_gpu_agent_s(monkeypatch):
    rocminfo = (
        "Agent 1\n  Name:                    AMD EPYC\n"
        "  Device Type:             CPU\n  ASIC Revision:           1(0x1)\n"
        "Agent 2\n  Name:                    gfx1250\n"
        "  Device Type:             GPU\n  ASIC Revision:           0(0x0)\n"
        "  ISA Info:\n    ISA 1\n      Name:  amdgcn-amd-amdhsa--gfx1250\n"
    )
    seen = []

    def run(argv, env, **kwargs):
        seen.append(env["ROCR_VISIBLE_DEVICES"])
        out = rocminfo if argv == ["rocminfo"] else json.dumps(A0["props"])
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(driverless.subprocess, "run", run)

    assert driverless.card(2) == {**A0, "index": 2}
    assert seen == ["2", "2"]


#: aiter's top-k in the cells' stack asks `torch._C` for the device and raw
#: stream, under the stream a capture has made current.
RAW = """
import sys, torch
from atom.compass.capture import driverless
driverless.install(sys.argv[1], "gfx1250")
side = driverless.Stream()
with torch.cuda.stream(side):
    raw = torch._C._cuda_getCurrentRawStream(0)
    print(torch._C._cuda_getDevice(), raw == side.cuda_stream)
"""


def test_a_capture_answers_the_raw_device_queries_on_the_declared_device(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(TREE)}
    done = subprocess.run(
        [sys.executable, "-c", RAW, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=TREE,
        env=env,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    assert done.stdout.splitlines()[-1] == "0 True"
