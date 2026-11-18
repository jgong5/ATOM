# SPDX-License-Identifier: MIT
"""LP lmcache sizes a chunk and a rank's state image from what the engine it
serves planned, as a prefill-decode connector prices its transfer.

Each member's runner publishes its planned KV block and the state image its
lmcache saves carry (`publish_planned_transfer`); LP lmcache reads every
member's at its first stamped request (`run._lmcache_plan`), and its launch
takes only the deployment's KV block tokens (`run._launch_lmcache`). The
template's S5 point, DeepSeek-V4-Pro at DP attention offloading to an lmcache
server of its own, launches on them.
"""

import argparse
import dataclasses
import json
from types import SimpleNamespace

import platforms
import pytest
from conftest import atom_config_double
from test_advice_lmcache import TEMPLATE, served
from test_lmcache_lp import CHUNK, _config, _run
from test_pd_planned_block import CONFIGS, publish

from atom.compass import run as compass_run
from atom.compass.advice import check, space
from atom.compass.backends.step_stub import Coefficients
from atom.compass.drive import runfile
from atom.compass.runner.overrides import _saved_image_bytes
from atom.model_engine.arg_utils import kv_offload_connector_config
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec

BLOCK, IMAGE = 2**20, 2**18
#: A prefill-decode run whose prefill side runs two DP ranks.
PD = {"router_s": 2.0**-12, "kv_write_req_s": 2.0**-10, "prefill_data_parallel_size": 2}


def test_the_template_s5_point_launches_lp_lmcache_on_its_kv_block(
    tmp_path, monkeypatch
):
    command, env = served(space.expand(TEMPLATE), "S5-c1", tmp_path / "S5-c1")
    entry = command[command.index("--lmcache") + 1]
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))
    store = tmp_path / "store"
    platform = platforms.flags(store, _run(tmp_path)["machine"], tmp_path)
    parser = argparse.ArgumentParser()
    runfile.add_arguments(parser)
    args = parser.parse_args(
        ["--step-stub", str(stub), *platform, "--store", str(store)]
        + ["--lmcache", str(entry)]
    )
    run = runfile.build(args, ["--model", str(tmp_path)])
    started = []
    monkeypatch.setattr(
        compass_run.subprocess, "Popen", lambda a, **k: started.append(a)
    )
    raw = json.loads(CONFIGS["deepseek-v4-pro"].read_text())
    config = atom_config_double(
        kv_transfer_config=kv_offload_connector_config(
            env["ATOM_KV_OFFLOAD"], env[check.EXTRA]
        ),
        hf_config=SimpleNamespace(**raw),
        tensor_parallel_size=8,
        enable_dp_attention=True,
        kv_cache_dtype="fp8",
        kv_cache_block_size=256,  # as `Config` fixes it for DeepSeek-V4
    )

    compass_run._launch_lmcache(config, run, None)

    [argv] = started
    assert argv[-1] == "256"


def test_lp_lmcache_sizes_a_chunk_and_the_image_every_member_planned(tmp_path):
    run = _run(tmp_path) | {"data_parallel_size": 8}
    for r in range(8):
        publish(tmp_path, f"engine.dp{r}", BLOCK, image=IMAGE)

    assert compass_run._lmcache_plan(run, 4) == (4 * BLOCK, IMAGE)


def test_a_prefill_decode_run_sizes_lp_lmcache_from_its_prefill_side_alone(
    tmp_path,
):
    publish(tmp_path, "engine-P.dp0", BLOCK, image=IMAGE)
    publish(tmp_path, "engine-P.dp1", BLOCK, image=IMAGE)

    # The decode side has published nothing, and LP lmcache does not wait for it.
    assert compass_run._lmcache_plan(_run(tmp_path) | PD, 1) == (BLOCK, IMAGE)


@pytest.mark.parametrize("key, what", [("block", "KV block"), ("image", "state image")])
def test_members_that_plan_another_block_or_image_are_refused_naming_both(
    tmp_path, key, what
):
    run = _run(tmp_path) | {"data_parallel_size": 2}
    publish(tmp_path, "engine.dp0", BLOCK, image=IMAGE)
    publish(tmp_path, "engine.dp1", **{"block": BLOCK, "image": IMAGE, key: 7})

    with pytest.raises(
        ValueError, match=f"^engine.dp1 plans a {what} of 7 B and engine.dp0 one of "
    ):
        compass_run._lmcache_plan(run, 1)


def test_a_member_that_publishes_no_plan_is_refused_by_name(tmp_path, monkeypatch):
    publish(tmp_path, "engine-P.dp1", BLOCK)
    monkeypatch.setattr(compass_run, "PLANNED_BLOCK_WAIT_S", 0.2)

    with pytest.raises(TimeoutError, match="^engine-P.dp0 published no planned KV"):
        compass_run._lmcache_plan(_run(tmp_path) | PD, 1)


def test_a_server_chunk_of_no_whole_number_of_blocks_is_refused_at_launch(
    tmp_path, monkeypatch
):
    started = []
    monkeypatch.setattr(
        compass_run.subprocess, "Popen", lambda *a, **k: started.append(a)
    )

    with pytest.raises(
        ValueError,
        match=f"--chunk-size is {CHUNK} tokens, not a whole number of this "
        "deployment's 3-token KV blocks",
    ):
        compass_run._launch_lmcache(
            _config(kv_cache_block_size=3), _run(tmp_path), None
        )
    assert started == []


SPEC = PagedStateCheckpointSpec(BLOCK, 2 * IMAGE, "native", IMAGE)


@pytest.mark.parametrize(
    "spec, replay, image",
    [(SPEC, 0, IMAGE), (SPEC, 512, 0), (None, 0, 0)],
    ids=["saves-carry-it", "bounded-replay", "no-checkpoints"],
)
def test_a_runner_publishes_the_image_its_lmcache_saves_carry(spec, replay, image):
    runner = SimpleNamespace(
        state_runtime=SimpleNamespace(checkpoint_spec=spec),
        config=SimpleNamespace(state_replay_tokens=replay),
    )

    assert _saved_image_bytes(runner) == image
