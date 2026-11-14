# SPDX-License-Identifier: MIT
"""`pd_sim.sh`, run directly or through `cctraces_sim.sh`, starts each side
under its own flags and environment, and from what each server was started
with, `keyed` picks that side's entry of DeepSeek-V4-Pro's crs005 DPA8 law
pair, keyed to the fp8 index cache a P/D side runs, and refuses both gpu-45
laws, which state the fp4 their lone servers ran, naming `index_cache_dtype`.
Under fp4 the same flags pick the gpu-45 pair, and the crs005 pair refuses
them, naming it too.

`python3` is stubbed on `PATH`: an API server prints its arguments and its
environment to its log and then answers `/health`, and the clock authority only
listens. The router binary is `false`: the script exits 1 once both servers have
answered, never on a timeout. Nothing needs a driver.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.run.step_grid import keyed
from tests.compass.held_ports import held_ports

TREE = Path(__file__).resolve().parents[2]
#: DeepSeek-V4-Pro's two DPA8 law pairs, prefill then decode, by the index
#: cache dtype each is keyed to.
LAWS = {
    index: [
        json.loads(
            (
                TREE
                / f"scripts/compass/laws/deepseek-v4-pro-dpa8-mi355x-{machine}-{side}.json"
            ).read_text()
        )
        for side in ("prefill", "decode")
    ]
    for index, machine in (("fp8", "crs005"), ("fp4", "gpu45"))
}
PREFILL_ENV = {"ATOM_NUMA_BIND": "1", "GPU_MAX_HW_QUEUES": "5"}
#: Outside `PERFORMANCE_KEYS`, so `keyed` reads none of it.
DECODE_ENV = {"PD_SIM_DECODE_ONLY": "7"}
SHARED_ENV = {
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "AITER_LOG_LEVEL": "WARNING",
    "ATOM_MOE_GU_ITLV": "1",
}
DECODE_SIZES = "[1,2," + ",".join(map(str, range(4, 257, 4))) + "]"
#: `EngineArgs.cudagraph_capture_sizes`, the ladder without the flag.
DEFAULT_SIZES = "[1,2,3,4,5,6,7,8,16,32,48,64,128,256,512]"
PYTHON3 = f"""#!/bin/sh
case "$2" in
atom.entrypoints.openai_server) echo "$@"; env; exec {sys.executable} -c 'import sys
from http.server import BaseHTTPRequestHandler as B, HTTPServer
B.do_GET = lambda h: (h.send_response(200), h.end_headers())
port = int(sys.argv[sys.argv.index("--server-port") + 1])
HTTPServer(("127.0.0.1", port), B).serve_forever()' "$@" ;;
atom.compass.run) exec {sys.executable} -c 'import socket, sys, time
s = socket.create_server(("127.0.0.1", int(sys.argv[1].rsplit(":", 1)[1])))
time.sleep(60)' "$6" ;;
esac
exec {sys.executable} "$@"
"""


def started(tmp_path, script) -> dict:
    """Each side's (arguments, environment) as `script` started its server."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text(PYTHON3)
    # The import probe: this tree's compass_harness, as cctraces_sim.sh requires.
    (bin_ / "harness").write_text(
        '#!/bin/sh\necho "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    out = tmp_path / "out"
    out.mkdir()
    (out / "run.json").write_text(json.dumps({"router_s": 1e-5}))
    (tmp_path / "traces.jsonl").write_text('{"id": 0}\n')
    env = {k: v for k, v in os.environ.items() if k not in PERFORMANCE_KEYS}
    env.update(
        SHARED_ENV,
        PATH=f"{bin_}:{os.environ['PATH']}",
        MODEL=str(tmp_path),
        TRACES=str(tmp_path / "traces.jsonl"),
        HARNESS_PYTHON=str(bin_ / "harness"),
        ATOMESH="false",
        WAIT_S="60",
        PREFILL_ARGS="--enable-tbo",
        DECODE_ARGS=f"--cudagraph-capture-sizes {DECODE_SIZES}",
        PREFILL_ENV=" ".join(f"{k}={v}" for k, v in PREFILL_ENV.items()),
        DECODE_ENV=" ".join(f"{k}={v}" for k, v in DECODE_ENV.items()),
    )
    args = [out / "run.json"] + ([out] if script == "cctraces_sim.sh" else [])
    with held_ports(4) as ports:
        names = ("CLOCK_PORT", "PREFILL_PORT", "DECODE_PORT", "ROUTER_PORT")
        env.update(zip(names, map(str, ports)))
        done = subprocess.run(
            ["bash", str(TREE / "scripts/compass" / script), *map(str, args)],
            env=env,
            cwd=TREE,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    assert done.returncode == 1, done.stdout + done.stderr
    # cctraces_sim.sh writes pd_sim.sh's output to pd_sim.log.
    why = done.stderr + "".join(p.read_text() for p in out.glob("pd_sim.log"))
    sides = {}
    for side in ("prefill", "decode"):
        assert (out / f"{side}.log").exists(), why
        argv, *lines = (out / f"{side}.log").read_text().splitlines()
        sides[side] = argv.split(), dict(e.split("=", 1) for e in lines if "=" in e)
    return sides


def side_config(argv, index: str) -> SimpleNamespace:
    """The engine config `keyed` reads: the side's own flags over the layout
    every law shares, its index cache `index`."""
    shared = LAWS["fp8"][1]["engine_config"]
    flag = "--cudagraph-capture-sizes"
    sizes = argv[argv.index(flag) + 1] if flag in argv else DEFAULT_SIZES
    return SimpleNamespace(
        enforce_eager=False,
        max_num_seqs=512,
        max_num_batched_tokens=16384,
        capture_sizes=ast.literal_eval(sizes),
        kv_cache_block_size=shared["kv_cache_block_size"],
        kv_cache_dtype=shared["kv_cache_dtype"],
        index_cache_dtype=index,
        quant_config=SimpleNamespace(compute_hash=lambda: shared["quantization"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=8),
        enable_dp_attention=True,
        enable_expert_parallel=False,
        enable_tbo="--enable-tbo" in argv,
        enable_tbo_decode=False,
    )


@pytest.mark.parametrize("script", ["pd_sim.sh", "cctraces_sim.sh"])
def test_each_side_starts_under_its_own_flags_and_picks_its_law_by_the_indexer(
    monkeypatch, tmp_path, script
):
    sides = started(tmp_path, script)
    (p_argv, p_env), (d_argv, d_env) = sides["prefill"], sides["decode"]
    assert "--enable-tbo" in p_argv and "--cudagraph-capture-sizes" not in p_argv
    assert "--enable-tbo" not in d_argv and DECODE_SIZES in d_argv
    assert (
        PREFILL_ENV.items() <= p_env.items() and not PREFILL_ENV.keys() & d_env.keys()
    )
    assert DECODE_ENV.items() <= d_env.items() and not DECODE_ENV.keys() & p_env.keys()

    for i, (argv, env) in enumerate(sides.values()):
        for name in PERFORMANCE_KEYS:
            monkeypatch.delenv(name, raising=False)
        for name in PERFORMANCE_KEYS & env.keys():
            monkeypatch.setenv(name, env[name])
        # A P/D side resolves fp8 under --kv-transfer-config, a lone server fp4.
        for index, other in (("fp8", "fp4"), ("fp4", "fp8")):
            here = keyed(side_config(argv, index))
            for entry, engine, what in (
                (lambda law: law, lambda e: e["engine_config"], "law"),
                (lambda law: law["host"], lambda e: e["key"]["engine"], "host"),
            ):
                own, refusing = (
                    [entry(law) for law in LAWS[k]] for k in (index, other)
                )
                assert compass_run._fitted_here(own, here, engine, what) is own[i]
                with pytest.raises(
                    ValueError, match=f"no {what} entry was fitted"
                ) as no:
                    compass_run._fitted_here(refusing, here, engine, what)
                assert f"entry {i} in ['index_cache_dtype']" in str(no.value)
