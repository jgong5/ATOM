# SPDX-License-Identifier: MIT
"""`parity_one_node.sh`: what each CONFIG starts both servers with, the DP
width the simulated run file names, and the memory measurement it starts first
when STORE is unset.

`python3` is stubbed for the measurement, the servers, the traffic and the
compare, so these run with no GPU and no model; the rest of the script runs as written.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_drive_gpu import GPU45, SHOWHW

from atom.compass.artifacts import machine_spec_key
from atom.compass.runner.pp_send import carrier_models
from atom.compass.spec import MachineSpec

ROOT = Path(__file__).resolve().parents[2]
STUB = """#!/bin/bash
await() { for _ in {1..600}; do [ -e "$1" ] && return; sleep 0.1; done; exit 1; }
case "$1 $2" in
"-m atom.entrypoints.openai.api_server")
    # As a real DP pair's group handshake orders them: rank 1 starts after rank 0,
    # and rank 0 listens only once rank 1 is up.
    dp=$STUB_LOG.dp${ATOM_DP_MASTER_PORT-}
    case "$*" in *"--data-parallel-rank 1"*) await "$dp.0" ;; esac
    echo "$* GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES-} ATOM_NUMA_BIND=${ATOM_NUMA_BIND-}" >> "$STUB_LOG"
    echo "${HIP_VISIBLE_DEVICES-} ${ATOM_DP_MASTER_PORT-}" >> "$STUB_LOG.env"
    case "$*" in
    *"--data-parallel-rank 0"*) touch "$dp.0" && await "$dp.1" ;;
    *"--data-parallel-rank 1"*) touch "$dp.1" ;;
    esac
    if [ -n "${TREE1_EXITS-}" ]; then
        case "$*" in *"--data-parallel-rank 1"*) exit 1 ;; esac
        exec sleep 60
    fi
    echo "Uvicorn running"
    # A simulated server's run ends after its traffic.
    case "$*" in *--compass-run*) await "$STUB_LOG.sent"; exit 0 ;; esac
    exec sleep 60 ;;
"-m atom.compass.parity.traffic")
    [ -n "${ATOM_COMPASS_RUN-}" ] && touch "$STUB_LOG.sent" && exit 0
    mkdir -p "$5" && echo '{"sent_ns": [0], "admitted_ns": [2000000]}' > "$5/arrivals.json"
    exit 0 ;;
"-m atom.compass.parity")
    echo "$*" >> "$STUB_LOG.compare"
    echo '{"steps": {"0": {"compared": 3, "identical": 3}}, "first_divergence": {"0": null}}'
    exit 0 ;;
"-c from atom.compass.cli import main; main()")
    echo "${*//$'\\n'/ } GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES-} ATOM_NUMA_BIND=${ATOM_NUMA_BIND-}" >> "$STUB_LOG"
    echo "${ATOM_KV_OFFLOAD-}" > "$STUB_LOG.offload"
    exit "${MEASURE_RC:-0}" ;;
"-m atom.compass.run")
    echo "$*" >> "$STUB_LOG"
    "$REAL_PYTHON" -c 'import socket, sys
socket.create_server(("127.0.0.1", int(sys.argv[1].rpartition(":")[2]))).accept()' "$6"
    exec sleep "${AUTHORITY_S:-0}" ;;
esac
exec "$REAL_PYTHON" "$@"
"""


def _parity(tmp_path, showhw=SHOWHW, **env):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").write_text(STUB)
    (bindir / "python3").chmod(0o755)
    (tmp_path / "showhw").write_text(showhw)
    (bindir / "rocm-smi").write_text(f"#!/bin/sh\ncat {tmp_path}/showhw\n")
    (bindir / "rocm-smi").chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"architectures": ["DeepseekV4ForCausalLM"]}')
    inherited = (
        "CONFIG",
        "MACHINE",
        "SERVER_ARGS",
        "GPU_MAX_HW_QUEUES",
        "ATOM_NUMA_BIND",
        "HIP_VISIBLE_DEVICES",
    )
    env = {k: v for k, v in os.environ.items() if k not in inherited} | dict(
        PATH=f"{bindir}:{os.environ['PATH']}",
        REAL_PYTHON=sys.executable,
        STUB_LOG=str(tmp_path / "servers.log"),
        MODEL=str(model),
        **{"STORE": str(tmp_path / "store"), "WAIT_S": "30"} | env,
    )
    env = {k: v for k, v in env.items() if v is not None}
    r = subprocess.run(
        ["bash", "scripts/compass/parity_one_node.sh", str(tmp_path / "out")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    log = tmp_path / "servers.log"
    servers = log.read_text().splitlines() if log.exists() else []
    return r, servers, tmp_path / "out" / "sim" / "run.json"


def test_tp2_is_the_default(tmp_path):
    r, servers, run = _parity(tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(servers) == 2 and "--compass-run" in servers[1]
    for line in servers:
        assert " -tp 2 --enable-dp-attention --enforce-eager " in line
        assert line.endswith("GPU_MAX_HW_QUEUES= ATOM_NUMA_BIND=")
    assert json.loads(run.read_text())["data_parallel_size"] == 2
    assert r.stdout.splitlines()[-1] == (
        "parity_one_node: dp0 compared=3 identical=3 first_divergence=None"
    )


def test_tp2_plain_drops_dp_attention_everywhere(tmp_path):
    r, servers, run = _parity(tmp_path, CONFIG="tp2-plain", STORE=None)
    assert r.returncode == 0, r.stderr
    assert len(servers) == 3 and " measure memory " in servers[0]
    for line in servers:
        assert " -tp 2 --enforce-eager " in line
        assert "--enable-dp-attention" not in line
    assert json.loads(run.read_text())["data_parallel_size"] == 1


def test_dp_two_trees_runs_two_rank_pairs_on_their_own_gpus_and_one_authority(
    tmp_path,
):
    r, servers, run = _parity(
        tmp_path, CONFIG="dp-two-trees", STORE=None, HIP_VISIBLE_DEVICES="4,5,6,7"
    )
    assert r.returncode == 0, r.stderr
    measure, *real, authority, sim0, sim1 = servers
    assert " measure memory --gpu 4,5,6,7 " in measure
    assert " -tp 2 --enable-dp-attention --data-parallel-size 2 --enforce-eager" in (
        measure
    )
    endpoint = json.loads(run.read_text())["clock_endpoint"]
    flags = f"--compass-run {run} --compass-clock-endpoint {endpoint}"
    assert authority == f"-m atom.compass.run {flags}"
    for side in (real, [sim0, sim1]):
        trees = [line.split(" --data-parallel-size ", 1)[1].split() for line in side]
        for k, tree in enumerate(trees):
            dp = ["2", "--data-parallel-size-local", "1", "--data-parallel-rank"]
            assert tree[:6] == [*dp, str(k), "--data-parallel-base-port"]
        assert trees[0][6] == trees[1][6]
    assert all(f" {flags} " in line for line in (sim0, sim1))
    # Each tree on its own GPU pair; a side's trees share one DP master port.
    env = [e.split() for e in (tmp_path / "servers.log.env").read_text().splitlines()]
    assert [e[0] for e in env] == ["4,5", "6,7"] * 2
    assert env[0][1] == env[1][1] != env[2][1] == env[3][1]
    # Below the ephemeral range, with the base port and engine sockets above it.
    low = Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0]
    assert int(env[2][1]) + 112 == int(trees[0][6]) < int(low)  # the simulated side
    assert json.loads(run.read_text())["data_parallel_size"] == 4


def test_dp_two_trees_stops_once_the_worker_tree_exits(tmp_path):
    r, _, _ = _parity(tmp_path, CONFIG="dp-two-trees", TREE1_EXITS="1")
    assert r.returncode == 1
    assert r.stderr == "parity_one_node: a real server exited\n"


def test_dp_two_trees_stops_an_authority_still_up_after_wait_s(tmp_path):
    r, _, _ = _parity(tmp_path, CONFIG="dp-two-trees", AUTHORITY_S="60", WAIT_S="5")
    assert r.returncode == 1
    assert r.stderr == "parity_one_node: the authority is still up\n"


def test_dsv4_dpa8_runs_the_nightly_prefill_server_at_eight_ranks(tmp_path):
    machine = ROOT / "scripts/compass/machines/mi308x-hjbog18.json"
    r, servers, run = _parity(tmp_path, CONFIG="dsv4-dpa8", MACHINE=str(machine))
    assert r.returncode == 0, r.stderr
    assert len(servers) == 2 and "--compass-run" in servers[1]
    for line in servers:
        assert " -tp 8 --enable-dp-attention --trust-remote-code " in line
        assert (
            "--kv_cache_dtype fp8 --block-size 16 --gpu-memory-utilization 0.85" in line
        )
        assert "--max-num-seqs 512 --enable-tbo" in line
        assert "--enforce-eager" not in line and "--enable-expert-parallel" not in line
        assert line.endswith("GPU_MAX_HW_QUEUES=5 ATOM_NUMA_BIND=1")
    spec = json.loads(run.read_text())
    assert spec["data_parallel_size"] == 8
    assert spec["store"] == str(tmp_path / "store")
    assert spec["machine"]["name"] == json.loads(machine.read_text())["name"]


def test_dsv4_dpa8_without_a_machine_starts_nothing(tmp_path):
    r, servers, _ = _parity(tmp_path, CONFIG="dsv4-dpa8")
    assert r.returncode != 0
    assert "needs MACHINE" in r.stderr
    assert servers == []


def test_a_spec_that_measured_the_models_tokenizer_names_it_once(tmp_path):
    arch = "DeepseekV4ForCausalLM"
    machine = json.loads(
        (ROOT / "scripts/compass/machines/mi308x-hjbog18.json").read_text()
    )
    machine["host"]["tokenizers"][0]["applies_to"].append(arch)
    (tmp_path / "spec.json").write_text(json.dumps(machine))
    r, _, run = _parity(
        tmp_path, CONFIG="dsv4-dpa8", MACHINE=str(tmp_path / "spec.json")
    )
    assert r.returncode == 0, r.stderr
    tokenizers = json.loads(run.read_text())["machine"]["host"]["tokenizers"]
    assert [a for t in tokenizers for a in t["applies_to"]].count(arch) == 1


def test_unset_store_measures_first_with_the_servers_flags_and_environment(tmp_path):
    machine = ROOT / "scripts/compass/machines/mi308x-hjbog18.json"
    r, servers, run = _parity(
        tmp_path, CONFIG="dsv4-dpa8", MACHINE=str(machine), STORE=None
    )
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out"
    measure, real, sim = servers
    assert (
        f" measure memory --gpu 0,1,2,3,4,5,6,7 --machine {machine} "
        f"--store {out}/store --out {out}/memory -- --model {tmp_path}/model -tp 8 "
    ) in measure
    flags = measure.split(" -tp ", 1)[1]
    assert real.split(" -tp ", 1)[1] == flags
    assert sim.split(" -tp ", 1)[1].replace(f" --compass-run {run}", "") == flags
    assert json.loads(run.read_text())["store"] == f"{out}/store"


def test_the_measurement_takes_the_real_runs_gpus(tmp_path):
    r, servers, _ = _parity(tmp_path, STORE=None, HIP_VISIBLE_DEVICES="4,5")
    assert r.returncode == 0, r.stderr
    assert " measure memory --gpu 4,5 " in servers[0]
    assert len(servers) == 3


def test_the_measurement_names_the_real_runs_hip_devices_as_rocm_smi_gpus(tmp_path):
    r, servers, _ = _parity(tmp_path, GPU45, STORE=None, HIP_VISIBLE_DEVICES="0,1")
    assert r.returncode == 0, r.stderr
    assert " measure memory --gpu 3,0 " in servers[0]


def test_a_failed_measurement_starts_no_server(tmp_path):
    r, servers, run = _parity(tmp_path, STORE=None, MEASURE_RC="3")
    assert r.returncode != 0
    assert len(servers) == 1 and " measure memory " in servers[0]
    assert not run.exists()


@pytest.mark.parametrize("config, writers", [("tp2", 1), ("tp2-plain", 2)])
def test_lmcache_writes_from_each_dp_rank_under_dp_attention_else_each_tp_rank(
    tmp_path, monkeypatch, config, writers
):
    lm = tmp_path / "lm"
    lm.mkdir()
    (lm / "lmcache").write_text("#!/bin/bash\nexec sleep 60\n")
    (lm / "lmcache").chmod(0o755)
    monkeypatch.setenv("PATH", f"{lm}:{os.environ['PATH']}")
    r, _, run = _parity(tmp_path, CONFIG=config, LMCACHE="1")
    assert r.returncode == 0, r.stderr
    assert json.loads(run.read_text())["lmcache"]["writers"] == writers


def test_under_lmcache_the_measurement_offloads_to_a_server_of_its_own(
    tmp_path, monkeypatch
):
    lm = tmp_path / "lm"
    lm.mkdir()
    (lm / "lmcache").write_text("#!/bin/bash\nexec sleep 60\n")
    (lm / "lmcache").chmod(0o755)
    monkeypatch.setenv("PATH", f"{lm}:{os.environ['PATH']}")
    r, servers, run = _parity(tmp_path, LMCACHE="1", STORE=None)
    assert r.returncode == 0, r.stderr
    measure = servers[0]
    beside = measure.split(" --beside ", 1)[1].split(" --store ", 1)[0].split()
    server = json.loads(run.read_text())["lmcache"]["server"]
    assert beside == ["lmcache", "server", *server]
    assert (tmp_path / "servers.log.offload").read_text() == "lmcache_mp\n"


def test_rapidserve_measures_a_rapidserve_engine_and_compares_each_engine(tmp_path):
    r, servers, run = _parity(tmp_path, CONFIG="rapidserve", STORE=None)
    assert r.returncode == 0, r.stderr
    measure, real, sim = servers
    flags = "-tp 1 --enforce-eager --level 0 --max-model-len 8192"
    for line in (measure, real, sim):
        assert f" {flags} " in line and " --enable-rapidserve" in line
    assert json.loads(run.read_text())["rapidserve"] is True
    record = tmp_path / "out" / "real" / "record"
    assert all((record / e / "arrivals.json").exists() for e in ("prefill", "decode"))
    assert r.stdout.splitlines()[-2:] == [
        f"parity_one_node: {e} dp0 compared=3 identical=3 first_divergence=None"
        for e in ("decode", "prefill")
    ]


def test_pp2_runs_two_stages_on_two_gpus_and_compares_each_stage(tmp_path):
    r, servers, run = _parity(tmp_path, CONFIG="pp2", STORE=None)
    assert r.returncode == 0, r.stderr
    machine = ROOT / "scripts/compass/machines/mi308x-hjbog18.json"
    assert f" measure memory --gpu 0,1 --machine {machine} " in servers[0]
    assert len(servers) == 3
    for line in servers:
        assert " -tp 1 -pp 2 --enforce-eager --level 0 --max-model-len 8192 " in line
    spec = json.loads(run.read_text())
    assert spec["pipeline_parallel_size"] == 2 and spec["data_parallel_size"] == 1
    assert spec["machine"]["name"] == json.loads(machine.read_text())["name"]
    # Node 18's cards read PTL enabled, and a stage send is priced on each carrier.
    stated = MachineSpec.from_mapping(spec["machine"])
    assert machine_spec_key(stated).values["ptl"] == "enabled"
    assert set(carrier_models(stated)) == {"rccl", "gloo"}
    compares = (tmp_path / "servers.log.compare").read_text().splitlines()
    assert [c.rsplit(" --stage ", 1)[1] for c in compares] == ["0", "1"]
    assert r.stdout.splitlines()[-2:] == [
        f"parity_one_node: pp{k} dp0 compared=3 identical=3 first_divergence=None"
        for k in (0, 1)
    ]
