# SPDX-License-Identifier: MIT
"""ATOM's Qwen3 at `-pp 2`: each stage builds only the embedding, final norm
and LM head it uses, so the weights a simulated run reads off its tree are its
share of the model's.

Each runner is built driverless in a process of its own, as
`test_qwen3_pp_stages` builds one, on a fake pipeline group standing at its
stage. The model is Qwen3-0.6B (`qwen3_0_6b_config.json`), whose LM head is its
embedding, and the same config untied.
"""

import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys

import pytest

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).with_name("qwen3_0_6b_config.json")

#: Builds ATOM's runner for the model under argv's root at pipeline rank argv's
#: rank, its group fake, and prints which of the embedding, final norm and LM
#: head it built, its weight bytes as a simulated run reads them, and its
#: embedding's bytes.
STAGE = """
import json, sys
root, rank, *atom_args = sys.argv[1:]
from atom.compass.capture import driverless
driverless.install(root, "gfx950")
import torch.distributed as dist
from aiter.dist import parallel_state as ps
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore
from atom.compass.capture import forward
from atom.compass.memory import ModelTerms
from atom.config import set_current_atom_config
from atom.models.utils import PPMissingLayer
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
model = runner.model
ends = {"embed_tokens": model.model.embed_tokens, "norm": model.model.norm}
ends["lm_head"] = model.lm_head
built = sorted(name for name, m in ends.items() if not isinstance(m, PPMissingLayer))
embedding = getattr(model.model.embed_tokens, "weight", None)
print(json.dumps({
    "built": built,
    "weights": ModelTerms.from_tree(model, activations=None).weights.nbytes,
    "embedding": embedding.untyped_storage().nbytes() if embedding is not None else 0,
}))
"""

TIED = (True, False)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """By (tied, pipeline width, rank), what that stage's runner printed."""
    root = tmp_path_factory.mktemp("qwen3")
    hf = json.loads(CONFIG.read_text())

    def build(stage):
        tied, pp, rank = stage
        model = root / f"{tied}-{pp}-{rank}" / "model"
        model.mkdir(parents=True)
        (model / "config.json").write_text(
            json.dumps({**hf, "tie_word_embeddings": tied})
        )
        argv = [str(model.parent), str(rank), "-pp", str(pp)]
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

    stages = [(tied, *s) for tied in TIED for s in ((1, 0), (2, 0), (2, 1))]
    with concurrent.futures.ThreadPoolExecutor(len(stages)) as pool:
        return dict(zip(stages, pool.map(build, stages)))


@pytest.mark.parametrize("tied", TIED, ids=("tied", "untied"))
def test_each_stage_builds_only_the_embedding_norm_and_head_it_uses(built, tied):
    last = ["lm_head", "norm"] + ["embed_tokens"] * tied
    assert built[tied, 1, 0]["built"] == ["embed_tokens", "lm_head", "norm"]
    assert [built[tied, 2, rank]["built"] for rank in (0, 1)] == [
        ["embed_tokens"],
        sorted(last),
    ]


@pytest.mark.parametrize("tied", TIED, ids=("tied", "untied"))
def test_the_stages_hold_the_models_weights_once_and_a_tied_table_on_each_end(
    built, tied
):
    whole = built[tied, 1, 0]
    stages = sum(built[tied, 2, rank]["weights"] for rank in (0, 1))
    assert stages == whole["weights"] + whole["embedding"] * tied
