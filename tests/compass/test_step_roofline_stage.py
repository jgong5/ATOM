# SPDX-License-Identifier: MIT
"""The step roofline prices one pipeline stage.

`run.runner` installs it alone on each stage of Qwen3-0.6B at `-pp 2`, at TP 1
and at TP 2 with `ATOM_PP_SEND_ALLGATHER` on, which it refuses for a backend
that prices no all-gather. Each stage states its share of the stack, as the
step stub's is stated, and prices the TP all-gather it runs on each sharded
tensor it receives; its forward then prices through `_group_step_seconds`,
which refuses a backend that states no share. A fallback states the share of
the backend it asks first, and over one that prices a whole step it is refused
naming that backend.

Each stage is built on the meta device in a process of its own, on a fake
pipeline group standing at the stage's TP rank 0, as `test_run_modules_pp`
builds one, and installed as `test_run_step_roofline_alone` installs the whole
model, on `tests/compass/machines/mi308x-hjbog18.json`; the installed backend
comes back pickled. A stage's forward runs on a runner seated as the last
stage, which sends on nothing, as `test_stage_price` seats one.
"""

import concurrent.futures
import json
import os
import pathlib
import pickle
import subprocess
import sys

import pytest
from test_backend_step_linear import law
from test_dp_step_max import prefill, runner

from atom.compass.backends import StepLinearBackend, StepLinearLaw
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_roofline import (
    StepFallbackBackend,
    StepRooflineBackend,
)
from atom.compass.backends.step_stub import Coefficients, StepStubBackend
from atom.compass.backends.work import attention, elementwise, gemm, price
from atom.compass.runner.overrides import RunnerRefusal
from atom.compass.spec import MachineSpec

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).with_name("qwen3_0_6b_config.json")
MI308X = TREE / "tests" / "compass" / "machines" / "mi308x-hjbog18.json"
#: (pipeline width, TP width, stage) of each build: the whole model, then
#: both stages at TP 1 and at TP 2.
STAGES = [(1, 1, 0), (2, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1)]
#: A decode step of eight rows at 600 to 607 tokens of context, eager.
DECODE = BatchView(tuple(RequestShape(1, 600 + i, True) for i in range(8)))

#: Builds the model under argv's root on the meta device of argv's machine's
#: arch at argv's pipeline width, TP width and stage, on a fake group standing
#: at the stage's TP rank 0, installs the step roofline alone through
#: `run.runner` with the stage's memory readings published, and pickles the
#: installed backend to ``backend.pkl``.
STAGE = """
import json, os, pathlib, pickle, sys
root, machine = pathlib.Path(sys.argv[1]), json.loads(pathlib.Path(sys.argv[2]).read_text())
sys.path.insert(0, sys.argv[3])
pp, tp, stage = map(int, sys.argv[4:7])
from atom.compass.capture import driverless
driverless.install(str(root), machine["device"]["arch"])
import torch
import torch.distributed as dist
from aiter.dist import parallel_state as ps
from torch.testing._internal.distributed.fake_pg import FakeStore
rank, world = stage * tp, pp * tp
dist.init_process_group(backend="fake", store=FakeStore(), rank=rank, world_size=world)
group, new_group = ps.init_model_parallel_group, dist.new_group
off = dict(use_device_communicator=False, use_message_queue_broadcaster=False)
ps.init_model_parallel_group = lambda *a, **k: group(*a, **k | off)
dist.new_group = lambda ranks=None, **k: new_group(ranks, backend="fake")
ps.init_distributed_environment(world_size=world, rank=rank, backend="fake", local_rank=0)
ps.initialize_model_parallel(
    tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, backend="fake"
)
from types import SimpleNamespace
from atom.config import Config, set_current_atom_config
config = Config(model=str(root / "model"), tensor_parallel_size=tp,
                pipeline_parallel_size=pp, load_dummy="empty", enforce_eager=True)
config.parallel_config.pipeline_parallel_rank = stage
set_current_atom_config(config)
from atom.model_engine.model_runner import support_model_arch_dict
from atom.utils import resolve_obj_by_qualname
from atom.compass.runner.overrides import FactoriesOnMeta, process_weights_after_loading
torch.set_default_dtype(config.torch_dtype)
with FactoriesOnMeta():
    tree = resolve_obj_by_qualname(support_model_arch_dict[config.hf_config.architectures[0]])(config)
    process_weights_after_loading(tree)
from test_memory_readings import publish_readings
from atom.compass import run as compass_run
from atom.compass.runner.overrides import _installed_backend
publish_readings(root / "store", config, document=machine)
run = {"bound_s": 600.0, "step_roofline": True, "machine": machine, "store": str(root / "store")}
(root / "run.json").write_text(json.dumps(run))
os.environ[compass_run.ENV] = str(root / "run.json")
runner = SimpleNamespace(config=config, rank=0, _tr_hybrid_active=lambda: False)
compass_run.runner(runner, tree)
(root / "backend.pkl").write_bytes(pickle.dumps(_installed_backend(runner)))
"""


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    """By (pipeline width, TP width, stage), the backend `run.runner` installed."""
    root = tmp_path_factory.mktemp("qwen3")
    here = str(pathlib.Path(__file__).parent)

    def build(stage):
        out = root / "-".join(map(str, stage))
        (out / "model").mkdir(parents=True)
        (out / "model" / "config.json").write_text(CONFIG.read_text())
        argv = [str(out), str(MI308X), here, *map(str, stage)]
        done = subprocess.run(
            [sys.executable, "-c", STAGE, *argv],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
            cwd=TREE,
            env={**os.environ, "PYTHONPATH": str(TREE), "ATOM_PP_SEND_ALLGATHER": "1"},
        )
        assert done.returncode == 0, done.stderr[-4000:]
        return pickle.loads((out / "backend.pkl").read_bytes())

    with concurrent.futures.ThreadPoolExecutor(len(STAGES)) as pool:
        return dict(zip(STAGES, pool.map(build, STAGES)))


def last_stage(backend):
    """A runner seated as the last of two stages, `backend` installed."""
    r = runner(backend)
    r.config.pipeline_parallel_size = 2
    r.config.parallel_config.pipeline_parallel_rank = 1
    return r


def by_hand(stage):
    """Qwen3-0.6B's decode price at `DECODE` on stage `stage` of two at TP 1,
    module by module from its config: each of the stage's 14 decoder layers,
    then the embedding on the first stage, the final norm and LM head on the
    last, each module one roofline (`work.price`), with no collective."""
    hf, machine = json.loads(CONFIG.read_text()), MachineSpec.from_mapping(
        json.loads(MI308X.read_text())
    )
    h, d, inter = hf["hidden_size"], hf["head_dim"], hf["intermediate_size"]
    heads, kv, rows = hf["num_attention_heads"], hf["num_key_value_heads"], 8
    layer = [
        gemm(rows, (heads + 2 * kv) * d, h),  # qkv_proj
        gemm(rows, h, heads * d),  # o_proj
        gemm(rows, 2 * inter, h),  # gate_up_proj
        gemm(rows, h, inter),  # down_proj
        elementwise(rows * h),  # input_layernorm
        elementwise(rows * h),  # post_attention_layernorm
        elementwise(rows * d),  # q_norm, at its head's width
        elementwise(rows * d),  # k_norm
        elementwise(rows * d),  # rotary_emb, over its rotary_dim
        attention([1] * rows, [600 + i for i in range(rows)], heads, d, kv),
        elementwise(rows * inter, reads=2),  # SiluAndMul
    ]
    ends = [
        [elementwise(rows * h)],  # embed_tokens
        [elementwise(rows * h), gemm(rows, hf["vocab_size"], h)],  # norm, lm_head
    ][stage]
    layers = hf["num_hidden_layers"] // 2
    return layers * sum(price(w, machine) for w in layer) + sum(
        price(w, machine) for w in ends
    )


@pytest.mark.parametrize("tp", [1, 2])
def test_each_stage_installs_the_roofline_alone_and_prices_its_forward(installed, tp):
    for stage in (0, 1):
        backend = installed[2, tp, stage]
        assert type(backend) is StepRooflineBackend
        assert backend.unclassified == []
        # get_pp_indices(28, k, 2): 14 layers each, every one paged.
        assert backend.stage_share == (0.5, 0.5)
        assert "stage share 0.5 of layers" in backend.describe()
        r = last_stage(backend)
        assert r.forward(prefill(8)).predicted_s > 0
        assert not vars(r).get("_refused_commands")
    assert installed[1, 1, 0].stage_share is None


def test_the_two_stages_decode_prices_sum_to_the_whole_stacks(installed):
    """The named result: Qwen3-0.6B at PP2 TP1, decode at batch 8."""
    whole = installed[1, 1, 0].price(DECODE)
    stages = [installed[2, 1, k].price(DECODE) for k in (0, 1)]
    ends = {t.name: t.seconds for t in whole.terms}
    ends = ends["roofline.embedding"] + ends["roofline.lm_head"]
    print(
        f"\nwhole {whole.seconds:.9g} s; stages "
        f"{' + '.join(f'{s.seconds:.9g}' for s in stages)} s; "
        f"by hand {by_hand(0):.9g} + {by_hand(1):.9g} s; "
        f"embedding and LM head {ends:.9g} s"
    )
    total = sum(s.seconds for s in stages)
    assert abs(total - whole.seconds) <= ends
    assert total == pytest.approx(whole.seconds, rel=1e-12)
    for k, s in enumerate(stages):
        assert s.seconds == pytest.approx(by_hand(k), rel=1e-12), k


def test_a_stage_at_tp2_prices_its_all_gather_by_the_link():
    """One all-gather of eight rows of 1024 bf16 elements over two ranks: a
    rank sends half of 16 KiB over its one link to the other, at the spec's
    link bandwidth and derate, after one hop of link latency."""
    machine = MachineSpec.from_mapping(json.loads(MI308X.read_text()))
    link = "interconnect.intra_node."
    rate = machine.value(link + "link_bandwidth_bytes_per_s")
    rate *= machine.value(link + "derate")
    hop = machine.value(link + "link_latency_s")
    stage = StepRooflineBackend(machine, (), stage_share=(0.5, 0.5), tp=2, hidden=1024)

    assert stage.all_gather_s(8) == pytest.approx(hop + 8 * 1024 * 2 / 2 / rate)
    assert StepRooflineBackend(machine, (), tp=1, hidden=1024).all_gather_s(8) == 0


def test_an_installed_tp2_stage_prices_the_same_all_gather(installed):
    machine = MachineSpec.from_mapping(json.loads(MI308X.read_text()))
    hand = StepRooflineBackend(machine, (), tp=2, hidden=1024).all_gather_s(8)
    assert [installed[2, 2, k].all_gather_s(8) for k in (0, 1)] == [hand, hand]


@pytest.mark.parametrize(
    "first",
    [StepStubBackend(), StepLinearBackend(StepLinearLaw(law()))],
    ids=["stub", "law"],
)
def test_a_fallback_over_a_backend_with_no_stage_share_is_refused_naming_it(
    installed, first
):
    fallback = StepFallbackBackend(first, installed[2, 1, 1])

    assert fallback.stage_share is None
    named = f"{type(first).__name__} prices a whole step"
    with pytest.raises(RunnerRefusal, match=named):
        last_stage(fallback).forward(prefill(8))


def test_a_fallback_over_a_stage_backend_states_its_share_and_all_gather(
    installed,
):
    stub = StepStubBackend(Coefficients(), stage_share=(0.5, 0.5))
    roofline = installed[2, 2, 1]
    fallback = StepFallbackBackend(stub, roofline)

    assert fallback.stage_share == (0.5, 0.5)
    assert fallback.all_gather_s(8) == stub.all_gather_s(8)
    assert last_stage(fallback).forward(prefill(8)).predicted_s > 0
    # A first backend that prices no all-gather leaves it to the roofline.
    linear = StepFallbackBackend(StepLinearBackend(StepLinearLaw(law())), roofline)
    assert linear.all_gather_s(8) == roofline.all_gather_s(8)
