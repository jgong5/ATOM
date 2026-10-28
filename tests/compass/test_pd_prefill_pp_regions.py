# SPDX-License-Identifier: MIT
"""A simulated prefill-decode transfer from a pipeline stage is refused where
ATOM refuses it, naming ATOM's reason.

Each model's stage 0 is built driverless at ``-pp 2`` in a process of its own,
as `test_qwen3_pp_stages` builds a stage, on the simulated connector's config,
and plans its pool with ATOM's `get_num_blocks`, as a simulated run does. Its
runner then publishes what its connector prices, or refuses; its fake KV cache
is allocated and ATOM's own `get_kv_transfer_tensors` is asked for the regions
on Mooncake's config. Kimi-K3's regions refuse the stage; DeepSeek-V4-Pro's
pool plan refuses it first; DeepSeek-R1-0528's MLA regions take it.
"""

import json
import pathlib

import pytest
from test_pd_planned_block import build_each

HERE = pathlib.Path(__file__).parent
CONFIGS = {
    "kimi-k3": HERE / "kimi_k3_config.json",
    "deepseek-v4-pro": HERE / "deepseek_v4_pro_config.json",
    "deepseek-r1-0528": HERE / "deepseek_r1_0528_config.json",
}
#: By model, what refuses its stage: ATOM's pool plan (``plan``), its
#: builder's regions (``atom``), the simulated runner (``compass``).
REFUSED = {
    "kimi-k3": ["atom", "compass"],
    "deepseek-v4-pro": ["plan"],
    "deepseek-r1-0528": [],
}
CELL = [
    "--kv_cache_dtype",
    "fp8",
    "--block-size",
    "128",
    "-pp",
    "2",
    "--kv-transfer-config",
    json.dumps({"kv_connector": "compass", "kv_role": "kv_producer"}),
]

#: Builds stage 0 of ATOM's runner for the model under argv's root on argv's
#: cell, its pipeline group fake, and prints what refused it and why.
STAGE = """
import json, pathlib, sys, torch
from types import SimpleNamespace
root, out, *atom_args = sys.argv[1:]
from atom.compass.capture import driverless
driverless.install(root, "gfx950")
import torch.distributed as dist
from aiter.dist import parallel_state as ps
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore
from atom.compass.capture import forward
from atom.compass.runner.overrides import RunnerRefusal, publish_planned_transfer
from atom.config import set_current_atom_config
config = forward._config(root + "/model", 1, atom_args)
raw = json.loads(pathlib.Path(root, "model", "config.json").read_text())
if "vision_config" in raw:
    config.multimodal_config = SimpleNamespace(
        vision_config=SimpleNamespace(**raw["vision_config"])
    )
pp = config.pipeline_parallel_size
dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=pp)
group, new_group = ps.init_model_parallel_group, dist.new_group
off = dict(use_device_communicator=False, use_message_queue_broadcaster=False)
ps.init_model_parallel_group = lambda *a, **k: group(*a, **k | off)
dist.new_group = lambda ranks=None, **k: new_group(ranks, backend="fake")
ps.init_distributed_environment(world_size=pp, rank=0, backend="fake", local_rank=0)
ps.initialize_model_parallel(pipeline_model_parallel_size=pp, backend="fake")
set_current_atom_config(config)
fake = driverless.driverless_mode(ShapeEnv())
recorder = forward._Recorder(config.compilation_config.static_forward_context, fake)
driverless.on_wait = recorder.wait
with forward._patches(recorder):
    runner = forward._build_runner(config, fake)
    # This stage stands for every stage: its block count is their least, and
    # there is no other stage to wait for.
    dist.is_initialized = lambda: False
    try:
        with fake, torch.inference_mode():
            runner.get_num_blocks()
    except RuntimeError as e:
        print(json.dumps({"plan": str(e)}))
        sys.exit()
refused = {}
runner.compass_block_file = pathlib.Path(out)
try:
    publish_planned_transfer(runner)
except RunnerRefusal as e:
    refused["compass"] = str(e)
try:
    with forward._patches(recorder), fake, torch.inference_mode():
        runner.attn_metadata_builder.warmup_per_req_cache = lambda: None
        runner.allocate_kv_cache(runner.pool_plan.paged_entries)
        config.kv_transfer_config = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
        runner.attn_metadata_builder.get_kv_transfer_tensors()
except NotImplementedError as e:
    refused["atom"] = str(e)
print(json.dumps(refused))
"""


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """By model, what refused its stage 0 and why, and the file its runner
    published."""
    return build_each(tmp_path_factory.mktemp("pp"), CONFIGS, STAGE, CELL)


@pytest.mark.parametrize("model", CONFIGS)
def test_a_prefill_stage_refuses_its_transfer_where_atom_does(built, model):
    refused = built[model]
    assert sorted(refused.keys() - {"published"}) == REFUSED[model]
    assert refused["published"].exists() == (not REFUSED[model])
    if "plan" in refused:
        assert "do not yet support pipeline parallelism" in refused["plan"]
    if "compass" in refused:
        assert refused["compass"].startswith("ATOM's `")
        assert refused["compass"].partition(": ")[2] in refused["atom"]
