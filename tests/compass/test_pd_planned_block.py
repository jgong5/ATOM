# SPDX-License-Identifier: MIT
"""A simulated prefill-decode transfer moves the paged KV block each engine's
runner planned.

Each engine's runner publishes one entry of the paged class ATOM's
`get_num_blocks` planned; the engine's connector, built inside
`compass_run.engine` as `Scheduler` builds it, prices a block at it once every
engine of the run has published, and refuses deployments that plan different
blocks. DeepSeek-V4-Pro's compressed cache and DeepSeek-R1-0528's MLA latent
(`deepseek_r1_0528_config.json`, its config.json byte for byte at revision
4236a6af538feda4548eca9ab308586007567f52) are planned by ATOM's own runner,
built driverless in a process of its own as `compass trace ops` builds one.
"""

import concurrent.futures
import copy
import json
import os
import pathlib
import shutil
import subprocess
import sys
import uuid
from types import SimpleNamespace

import pytest
from conftest import atom_config_double
from test_dp_slice import DP_MACHINE
from test_lmcache_lp import CHUNK, OFFLOAD, _run
from test_memory_readings import DOCUMENT

from atom.compass import run as compass_run
from atom.compass.clock import ClockAuthority
from atom.compass.clock_transport import serve
from atom.compass.kv import WRITE_REQ_KEY
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.utils import clock
from atom.utils.distributed import utils as dist_utils

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIGS = {
    "deepseek-v4-pro": pathlib.Path(__file__).with_name("deepseek_v4_pro_config.json"),
    "deepseek-r1-0528": pathlib.Path(__file__).with_name(
        "deepseek_r1_0528_config.json"
    ),
}
#: Both sides of the stage-4 1P1D cell: DPA-TP8, an fp8 cache, 16-token blocks.
CELL = ["--kv_cache_dtype", "fp8", "--enable-dp-attention", "--block-size", "16"]

PLAN = """
import json, pathlib, sys, torch
root, out, *atom_args = sys.argv[1:]
from atom.compass.capture import driverless
declared = driverless.install(root, "gfx950")
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from atom.compass.capture import forward
from atom.compass.runner.overrides import publish_paged_block
from atom.config import set_current_atom_config
config = forward._config(root + "/model", 8, atom_args)
parallel = config.parallel_config
declared["group"] = driverless.group(1, parallel.data_parallel_size)
set_current_atom_config(config)
fake = driverless.driverless_mode(ShapeEnv())
recorder = forward._Recorder(config.compilation_config.static_forward_context, fake)
driverless.on_wait = recorder.wait
with forward._patches(recorder):
    runner = forward._build_runner(config, fake)
    with fake, torch.inference_mode():
        runner.get_num_blocks()
runner.compass_block_file = pathlib.Path(out)
publish_paged_block(runner)
plan = runner.pool_plan
print(json.dumps({"entry_bytes": plan.entry_bytes[plan.paged_class],
                  "block_size": config.kv_cache_block_size}))
"""


def publish(out_dir, name: str, block: int, tp: int = 1) -> None:
    """Publish engine `name`'s planned block as its runner does."""
    now = pathlib.Path(out_dir) / compass_run.NOW_FILE.format(name)
    compass_run.publish_block(compass_run._block_file(now), block, tp)


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Serve a prefill-decode run's authority; `served(**more)` writes its run
    file with `more` and returns it."""
    servers = []

    def run_file(**more):
        run = {
            "clock_endpoint": f"inproc:test-pd-block-{uuid.uuid4().hex}",
            "bound_s": 60.0,
            "admission_path": "serving",
            "ipc_s": 2.0**-14,
            "stream_s": 2.0**-12,
            "router_s": 2.0**-12,
            "kv_write_req_s": 2.0**-10,
            "kv_link": "intra_node",
            "machine": copy.deepcopy(DOCUMENT),
            "out_dir": str(tmp_path),
            **more,
        }
        (tmp_path / "run.json").write_text(json.dumps(run))
        monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
        table = compass_run.channel_table(run)
        servers.append(
            serve(ClockAuthority(table, bound_s=60.0), run["clock_endpoint"])
        )
        return run

    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(dist_utils, "LP_OF_RANK", None)
    yield run_file
    for server in servers:
        server.close()


def _connector(role: str, **config):
    """The scheduler half the engine of `role` builds, its sockets closed."""
    kv = {
        "kv_connector": "compass",
        "kv_role": role,
        WRITE_REQ_KEY: f"inproc://kv-write-req-{uuid.uuid4().hex}",
    }
    config = atom_config_double(kv_transfer_config=kv, **config)
    with compass_run.engine(config):
        half = KVConnectorFactory.create_connector(config, role="scheduler")
    for sock in half._socks.values():
        sock.raw.close(linger=0)
    return half


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    """By model, the block ATOM's runner plans for the cell and the file it
    published, each built in a process of its own."""
    root = tmp_path_factory.mktemp("planned")

    def build(name):
        (root / name / "model").mkdir(parents=True)
        shutil.copy(CONFIGS[name], root / name / "model" / "config.json")
        out = root / name / "now-engine-P.kv_block"
        done = subprocess.run(
            [sys.executable, "-c", PLAN, str(root / name), str(out), *CELL],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
            cwd=TREE,
            env={**os.environ, "PYTHONPATH": str(TREE)},
        )
        assert done.returncode == 0, done.stderr[-4000:]
        return json.loads(done.stdout.splitlines()[-1]) | {"published": out}

    with concurrent.futures.ThreadPoolExecutor(len(CONFIGS)) as pool:
        return dict(zip(CONFIGS, pool.map(build, CONFIGS)))


@pytest.mark.parametrize("model", CONFIGS)
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_a_transferred_block_is_the_paged_entry_atom_planned(
    served, tmp_path, planned, model, role
):
    run = served()
    shutil.copy(planned[model]["published"], tmp_path / "now-engine-P.kv_block")
    shutil.copy(planned[model]["published"], tmp_path / "now-engine-D.kv_block")
    raw = json.loads(CONFIGS[model].read_text())
    half = _connector(
        role,
        hf_config=SimpleNamespace(**raw),
        kv_cache_block_size=planned[model]["block_size"],
        kv_cache_dtype="fp8",
    )
    (transfer,) = half._transfer
    link = run["machine"]["interconnect"]["intra_node"]
    assert transfer.bytes_per_block == planned[model]["entry_bytes"]
    assert transfer.bandwidth_bytes_per_s == pytest.approx(
        link["link_bandwidth_bytes_per_s"] * link["derate"]
    )


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_sides_that_plan_different_blocks_refuse_naming_both(served, tmp_path, role):
    served()
    publish(tmp_path, "engine-P", 1835008, tp=1)
    publish(tmp_path, "engine-D", 917504, tp=2)
    with pytest.raises(
        ValueError,
        match="engine-D at TP 2 plans a KV block of 917504 B and engine-P at TP 1 "
        "one of 1835008 B",
    ):
        _connector(role)


@pytest.mark.parametrize("decode, refused", [(1000, None), (900, "900 B and ")])
def test_decode_prices_each_prefill_stage_at_the_block_that_stage_planned(
    served, tmp_path, decode, refused
):
    served(machine=DP_MACHINE, **{compass_run.PREFILL_PP: 2})
    publish(tmp_path, "engine-P-pp0", 600)
    publish(tmp_path, "engine-P-pp1", 400)
    publish(tmp_path, "engine-D", decode)
    if refused:
        with pytest.raises(ValueError, match=f"engine-D at TP 1 .* {refused}engine-P"):
            _connector("kv_consumer")
        return
    stages = _connector("kv_consumer")._transfer
    assert [m.bytes_per_block for m in stages] == [600, 400]


def test_an_engine_waits_a_bounded_time_for_the_others_blocks(
    served, tmp_path, monkeypatch
):
    served(decode_instances=2, router_in_s=2.0**-14)
    publish(tmp_path, "engine-P", 1000)
    publish(tmp_path, "engine-D0", 1000)
    monkeypatch.setattr(compass_run, "HAND_IN_WAIT_S", 0.2)
    with pytest.raises(TimeoutError, match="^engine-D1 published no planned KV"):
        _connector("kv_producer")


def test_an_engine_whose_runner_published_nothing_refuses_at_once(served):
    served()
    with pytest.raises(RuntimeError, match="runner published no planned KV block"):
        _connector("kv_consumer")


@pytest.mark.parametrize(
    "model, why",
    [
        ("deepseek-v4-pro", "DeepseekV4ForCausalLM states compress_ratios, "),
        ("deepseek-r1-0528", "DeepseekV3ForCausalLM states kv_lora_rank: "),
    ],
)
def test_the_lmcache_station_refuses_a_cache_the_geometry_cannot_size(
    tmp_path, monkeypatch, model, why
):
    started = []
    monkeypatch.setattr(
        compass_run.subprocess, "Popen", lambda *a, **k: started.append(a)
    )
    raw = json.loads(CONFIGS[model].read_text())
    config = atom_config_double(
        kv_transfer_config=OFFLOAD
        | {"kv_connector_extra_config": {"lmcache.chunk_size": CHUNK}},
        hf_config=SimpleNamespace(**raw),
        kv_cache_block_size=CHUNK,
        kv_cache_dtype="fp8",
    )
    with pytest.raises(ValueError, match=why):
        compass_run._launch_lmcache(config, _run(tmp_path), None)
    assert started == []
