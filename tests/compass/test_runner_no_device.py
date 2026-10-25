# SPDX-License-Identifier: MIT
"""The simulated runner's start sets no device and builds nothing on one.

`start_on_host` runs with aiter stood in, because importing aiter needs a
driver; what it is checked against is ATOM's own `CpuGpuBuffer` and torch's own
`torch.cuda.Stream`, the two things `ModelRunner.__init__` builds after it.
"""

import ast
import copy
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
from test_memory_readings import CONFIG_JSON, DOCUMENT, EXPECTED, readings_at
from transformers import PretrainedConfig

from atom.compass.runner.overrides import (
    HostEvent,
    HostStream,
    RunnerRefusal,
    install_device_readings,
    start_on_host,
)
from atom.compass.spec import MachineSpec
from atom.model_ops.attentions.deepseek_v41.backend import DeepseekV41MetadataBuilder
from atom.utils import CpuGpuBuffer
from atom.utils.h2d import PublicationOwner

REPO = pathlib.Path(__file__).resolve().parents[2]


def _config(tp=1, dp=1, simulated_tp=None, pp=1):
    return SimpleNamespace(
        tp_world_size=tp,
        tensor_parallel_size=tp if simulated_tp is None else simulated_tp,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        pipeline_parallel_size=pp,
        master_addr="127.0.0.1",
        port=29500,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            data_parallel_rank=0,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_base_port=29501,
            _managed_distributed_store=False,
        ),
    )


@pytest.fixture
def started(monkeypatch):
    """Run `start_on_host`, recording the group call; undo what it replaces."""
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(
            init_distributed_environment=lambda *a, **k: calls.append((a, k)),
            initialize_model_parallel=lambda *a, **k: calls.append((a, k)),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.utils",
        SimpleNamespace(
            get_distributed_init_method=lambda ip, port: f"tcp://{ip}:{port}"
        ),
    )
    monkeypatch.setattr(torch.cuda, "Stream", torch.cuda.Stream)
    monkeypatch.setattr(torch.cuda, "Event", torch.cuda.Event)
    monkeypatch.setattr(CpuGpuBuffer, "__init__", CpuGpuBuffer.__init__)
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)

    def no_device(*a, **k):
        raise AssertionError("the start set a device")

    monkeypatch.setattr(torch.cuda, "set_device", no_device)

    def start(config):
        runner = SimpleNamespace()
        start_on_host(runner, 0, config)
        return runner, calls

    return start


def test_the_start_sets_the_host_as_the_device_and_builds_groups_on_gloo(started):
    runner, calls = started(_config())
    assert runner.device == torch.device("cpu")
    (_, world), (args, groups) = calls
    assert world["backend"] == "gloo"
    assert world["distributed_init_method"] == "tcp://127.0.0.1:29501"
    assert args == (1, 1)
    assert groups["custom_group_config"] == {}


def test_atoms_buffers_built_after_it_are_unpinned_and_on_the_host(started):
    started(_config())
    buffer = CpuGpuBuffer(8, dtype=torch.int32, device=torch.device("cpu"))
    assert not buffer.cpu.is_pinned()
    assert buffer.gpu.device == torch.device("cpu")
    assert not buffer.clone().cpu.is_pinned()


def test_a_stream_built_after_it_is_a_host_stand_in_that_cannot_queue_work(started):
    started(_config())
    stream = torch.cuda.Stream(torch.device("cpu"))
    assert isinstance(stream, HostStream)
    with pytest.raises(AttributeError):
        stream.synchronize()


def test_an_event_built_after_it_records_and_waits_on_the_host(started):
    """`ModelRunner._init_h2d_publication` records each slot's event in
    `__init__`, and ATOM's publication owner records and waits on it per step."""
    started(_config())
    event = torch.cuda.Event()
    assert isinstance(event, HostEvent)
    event.record()
    owner = PublicationOwner(torch.device("cpu"), event)
    owner.begin()
    owner.finish()
    owner.begin()
    owner._wait_sources()


@pytest.mark.parametrize(
    "width, refusal",
    [({"simulated_tp": 2}, "-tp 2 starts 1 workers")],
)
def test_a_start_the_host_groups_cannot_hold_is_refused_before_anything_is_replaced(
    started, width, refusal
):
    stream = torch.cuda.Stream
    event = torch.cuda.Event
    with pytest.raises(RunnerRefusal, match=refusal):
        started(_config(**width))
    assert torch.cuda.Stream is stream
    assert torch.cuda.Event is event


def test_atoms_init_builds_streams_and_buffers_only_after_the_replaced_setup():
    """What the start relies on, read from ATOM's `ModelRunner.__init__`."""
    tree = ast.parse((REPO / "atom/model_engine/model_runner.py").read_text())
    init = next(
        f
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == "ModelRunner"
        for f in c.body
        if isinstance(f, ast.FunctionDef) and f.name == "__init__"
    )
    body = [ast.unparse(s) for s in init.body]

    def first(text):
        return next(i for i, s in enumerate(body) if text in s)

    setup = first("self._setup_device_and_distributed(")
    for later in ("tokenIDProcessor(", "torch.cuda.Stream(", "allocate_forward_vars("):
        assert setup < first(later), later


def test_asked_about_its_own_device_a_host_start_answers_from_its_readings(
    started, monkeypatch
):
    """DeepSeek-V4.1's attention builder bounds its pages by `mem_get_info` and
    `memory_allocated` of the runner's device, the host after this start: the
    two answer from the installed readings, what a real start holds once its
    target loads and before the builder allocates, and of a card as before."""

    def card(device=None):
        return "card", device

    monkeypatch.setattr(torch.cuda, "mem_get_info", card)
    monkeypatch.setattr(torch.cuda, "memory_allocated", card)
    runner, _ = started(_config())
    cpu = torch.device("cpu")
    with pytest.raises(RunnerRefusal, match="install_device_readings"):
        torch.cuda.mem_get_info(cpu)
    qwen = PretrainedConfig.from_dict(
        json.loads(CONFIG_JSON.read_text())["text_config"]
    )
    spec = MachineSpec.from_mapping(copy.deepcopy(DOCUMENT))
    install_device_readings(runner, readings_at(spec, qwen, 2))
    expected, peak = EXPECTED[2], EXPECTED[2]["peak_torch"]
    free = sum(expected["free"].values())
    assert torch.cuda.mem_get_info(cpu) == (free, expected["total"]["capacity"])
    held = peak["weights"] + peak["buffers"] + peak["load residue"]
    assert torch.cuda.memory_allocated(cpu) == held
    assert torch.cuda.mem_get_info(0) == ("card", 0)
    assert torch.cuda.memory_allocated() == ("card", None)


@pytest.mark.parametrize("index", ["fp8", "fp4"])
def test_a_host_start_refuses_the_v41_page_bound_only_on_the_fp4_index_plane(
    started, monkeypatch, index
):
    """The fp4 plane sizes V4.1's packed score workspace by the PAGEs left
    after `memory_allocated`, which a real start reads with the drafter and
    forward buffers held too; the fp8 plane's workspace ignores the bound."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", torch.cuda.mem_get_info)
    monkeypatch.setattr(torch.cuda, "memory_allocated", torch.cuda.memory_allocated)
    config = _config()
    config.index_cache_dtype = index
    runner, _ = started(config)
    qwen = PretrainedConfig.from_dict(
        json.loads(CONFIG_JSON.read_text())["text_config"]
    )
    spec = MachineSpec.from_mapping(copy.deepcopy(DOCUMENT))
    install_device_readings(runner, readings_at(spec, qwen, 2))
    builder = SimpleNamespace(
        device=runner.device, geometry=SimpleNamespace(paged_bytes=1 << 20)
    )
    if index == "fp4":
        with pytest.raises(RunnerRefusal, match="--index-cache-dtype fp4"):
            DeepseekV41MetadataBuilder._page_bound(builder, 0.9)
        return
    expected, peak = EXPECTED[2], EXPECTED[2]["peak_torch"]
    held = peak["weights"] + peak["buffers"] + peak["load residue"]
    left = int(expected["total"]["capacity"] * 0.9) - held
    assert DeepseekV41MetadataBuilder._page_bound(builder, 0.9) == left >> 20
