# SPDX-License-Identifier: MIT
"""The step roofline's module walk over each stage of Qwen3-0.6B at `-pp 2`
leaves nothing unclassified that the whole model's walk does not, and prices
each stage's own decoder layers.

Each runner is built driverless in a process of its own, as
`test_qwen3_pp_stages` builds one, on a fake pipeline group standing at its
stage. The model is Qwen3-0.6B (`qwen3_0_6b_config.json`).
"""

import concurrent.futures
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from atom.models.utils import get_pp_indices

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).with_name("qwen3_0_6b_config.json")

#: Builds ATOM's runner for the model under argv's root at pipeline rank argv's
#: rank, its group fake, and prints the modules the walk leaves unclassified
#: and the decoder layers of the attentions it prices.
STAGE = """
import json, sys
root, rank, *atom_args = sys.argv[1:]
from atom.compass.capture import driverless
driverless.install(root, "gfx950")
import torch.distributed as dist
from aiter.dist import parallel_state as ps
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore
from atom.compass.backends.step_roofline import UNCLASSIFIED
from atom.compass.capture import forward
from atom.compass.run.modules import walk
from atom.config import set_current_atom_config
config = forward._config(root + "/model", 1, atom_args)
pp, rank = config.pipeline_parallel_size, int(rank)
dist.init_process_group(backend="fake", store=FakeStore(), rank=rank, world_size=pp)
group, new_group = ps.init_model_parallel_group, dist.new_group
off = dict(use_device_communicator=False, use_message_queue_broadcaster=False)
ps.init_model_parallel_group = lambda *a, **k: group(*a, **k | off)
dist.new_group = lambda ranks=None, **k: new_group(ranks, backend="fake")
ps.init_distributed_environment(world_size=pp, rank=rank, backend="fake", local_rank=0)
ps.initialize_model_parallel(pipeline_model_parallel_size=pp, backend="fake")
set_current_atom_config(config)
fake = driverless.driverless_mode(ShapeEnv())
recorder = forward._Recorder(config.compilation_config.static_forward_context, fake)
with forward._patches(recorder):
    runner = forward._build_runner(config, fake)
found = walk(runner.model)
print(json.dumps({
    "unclassified": [m.name for m in found if m.kind == UNCLASSIFIED],
    "attention": [m.layer for m in found if m.kind == "attention"],
}))
"""


@pytest.fixture(scope="module")
def walked(tmp_path_factory):
    """By (pipeline width, rank), what that stage's walk printed."""
    root = tmp_path_factory.mktemp("qwen3")

    def build(stage):
        pp, rank = stage
        (root / f"{pp}-{rank}" / "model").mkdir(parents=True)
        shutil.copy(CONFIG, root / f"{pp}-{rank}" / "model" / "config.json")
        argv = [str(root / f"{pp}-{rank}"), str(rank), "-pp", str(pp)]
        done = subprocess.run(
            [sys.executable, "-c", STAGE, *argv],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
            cwd=TREE,
            env={**os.environ, "PYTHONPATH": str(TREE)},
        )
        assert done.returncode == 0, done.stderr[-4000:]
        return json.loads(done.stdout.splitlines()[-1])

    stages = [(1, 0), (2, 0), (2, 1)]
    with concurrent.futures.ThreadPoolExecutor(len(stages)) as pool:
        return dict(zip(stages, pool.map(build, stages)))


def test_a_stage_leaves_unclassified_only_what_the_whole_model_does(walked):
    whole = set(walked[1, 0]["unclassified"])
    stages = [set(walked[2, rank]["unclassified"]) - whole for rank in (0, 1)]
    assert stages == [set(), set()]


def test_each_stage_prices_the_attention_of_its_own_decoder_layers(walked):
    total = len(walked[1, 0]["attention"])
    spans = [list(range(*get_pp_indices(total, rank, 2))) for rank in (0, 1)]
    assert [walked[2, rank]["attention"] for rank in (0, 1)] == spans
