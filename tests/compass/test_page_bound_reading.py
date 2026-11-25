# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's PAGE bound under --index-cache-dtype fp4, simulated.

The V4.1 attention builder bounds its PAGEs by what `memory_allocated` leaves
of its device's budget, and on the fp4 index plane packs its score workspace
by that bound. A real start holds the drafter, the forward buffers and the
builder's own there besides the load; `compass measure memory` reads it where
the builder does, the run's member carries it, and a host start answers it.
ATOM's `_page_bound` is run from its source: importing the backend needs a
driver.
"""

import ast
import copy
import json
import pathlib
import shutil
from types import SimpleNamespace

import pytest
import torch
from test_drive_run import stub
from test_measure_memory import reading_runner, tp2
from test_memory_readings import (
    CONFIG_JSON,
    DOCUMENT,
    EXPECTED,
    PARAMETERS,
    READINGS,
    measured,
    meta_tree,
    model_terms,
    publish_readings,
    reserved,
)
from test_run_lacking import checked, lacked, served
from test_run_step_grid import deployment
from test_runner_no_device import _config, started  # noqa: F401  fixture
from transformers import PretrainedConfig

from atom.compass.artifacts import Kind
from atom.compass.backends.geometry import dtype_bytes
from atom.compass.drive import measure_memory
from atom.compass.memory import MemoryReadings, MemoryRefusal, device_readings
from atom.compass.memory.readings import PAGE_BOUND
from atom.compass.runner.overrides import RunnerRefusal, install_device_readings
from atom.compass.spec import MachineSpec

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "atom/model_ops/attentions/deepseek_v41/backend.py"
#: What a real DTP2 start holds above its load where the builder asks.
HELD_THERE = 3_210_000_000
#: The bytes of one PAGE across the paged layers.
PAGED_BYTES = 1_234_567
V41 = SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])


def atoms_page_bound():
    """`DeepseekV41MetadataBuilder._page_bound`, compiled from ATOM's source."""
    tree = ast.parse(BACKEND.read_text())
    (builder,) = (
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41MetadataBuilder"
    )
    (bound,) = (
        n
        for n in builder.body
        if isinstance(n, ast.FunctionDef) and n.name == "_page_bound"
    )
    scope = {"torch": torch}
    exec(ast.unparse(bound), scope)  # noqa: S102
    return scope["_page_bound"]


def host_start(started, index, member):  # noqa: F811
    """A TP2 host start on the `index` plane, its readings off `member`."""
    config = _config()
    config.index_cache_dtype = index
    runner, _ = started(config)
    qwen = PretrainedConfig.from_dict(
        json.loads(CONFIG_JSON.read_text())["text_config"]
    )
    spec = MachineSpec.from_mapping(copy.deepcopy(DOCUMENT))
    capacity = int(spec.value("device.memory.capacity_bytes"))
    readings = device_readings(
        spec,
        memory=MemoryReadings.from_member(member, "page bound test entry"),
        tp_width=2,
        model=model_terms(meta_tree(PARAMETERS * dtype_bytes(qwen.dtype) // 2)),
        cudagraph_overhead=reserved(qwen, capacity),
    )
    install_device_readings(runner, readings)
    return runner


def loaded():
    """What the TP2 start holds once its target loads."""
    peak = EXPECTED[2]["peak_torch"]
    return peak["weights"] + peak["buffers"] + peak["load residue"]


def test_an_fp4_host_start_bounds_the_pages_a_real_start_bounds(
    started, monkeypatch  # noqa: F811
):
    """ATOM's bound on the host start counts the PAGEs it counts on a card
    holding what the real start held there: its load and the reading."""
    capacity = EXPECTED[2]["total"]["capacity"]
    card = torch.device("cuda", 0)

    def card_free(device=None):
        assert torch.device(device) == card
        return 0, capacity

    def card_allocated(device=None):
        assert torch.device(device) == card
        return loaded() + HELD_THERE

    monkeypatch.setattr(torch.cuda, "mem_get_info", card_free)
    monkeypatch.setattr(torch.cuda, "memory_allocated", card_allocated)
    member = {**READINGS[2], PAGE_BOUND: HELD_THERE}
    runner = host_start(started, "fp4", member)
    bound = atoms_page_bound()
    geometry = SimpleNamespace(paged_bytes=PAGED_BYTES)

    simulated = bound(SimpleNamespace(device=runner.device, geometry=geometry), 0.9)
    real = bound(SimpleNamespace(device=card, geometry=geometry), 0.9)

    assert torch.cuda.memory_allocated(runner.device) == loaded() + HELD_THERE
    assert simulated == real


def test_an_fp4_host_start_whose_readings_lack_it_refuses_naming_the_reading(
    started, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(torch.cuda, "memory_allocated", torch.cuda.memory_allocated)
    runner = host_start(started, "fp4", READINGS[2])

    with pytest.raises(RunnerRefusal) as refused:
        torch.cuda.memory_allocated(runner.device)

    assert f"state no {PAGE_BOUND}" in str(refused.value)
    assert "compass measure memory" in str(refused.value)


@pytest.mark.parametrize("member", [READINGS[2], {**READINGS[2], PAGE_BOUND: 7}])
def test_the_fp8_plane_answers_the_load_alone(
    started, monkeypatch, member  # noqa: F811
):
    monkeypatch.setattr(torch.cuda, "memory_allocated", torch.cuda.memory_allocated)
    runner = host_start(started, "fp8", member)

    assert torch.cuda.memory_allocated(runner.device) == loaded()


@pytest.mark.parametrize("value", [-1, 1.5, None, True])
def test_a_member_s_reading_at_the_page_bound_is_whole_bytes(value):
    with pytest.raises(MemoryRefusal, match=PAGE_BOUND):
        measured(1, **{PAGE_BOUND: value})
    read = measured(1, **{PAGE_BOUND: 9})
    assert read.term("held", PAGE_BOUND).nbytes == 9


class Builder:
    """An attention builder that bounds its PAGEs as V4.1's does, as it builds."""

    def __init__(self, device):
        self.device = device
        self.pages = self._page_bound(0.9)

    def _page_bound(self, utilization):
        return torch.cuda.memory_allocated(self.device) // 100


@pytest.mark.parametrize("bounds", [True, False])
def test_the_reading_runner_writes_what_is_allocated_where_the_builder_bounds(
    tmp_path, monkeypatch, bounds
):
    built = type("Built", (Builder,), {}) if bounds else object
    allocated = {"cuda:0": 1_000}

    class ModelRunner:
        attn_backend = SimpleNamespace(get_builder_cls=lambda: built)

        def allocate_forward_vars(self):
            allocated["cuda:0"] += 4_000

    monkeypatch.setattr(torch.cuda, "memory_allocated", allocated.get)
    runner = reading_runner(monkeypatch, ModelRunner).MemoryReadingModelRunner()
    runner.config, runner.rank = tp2(tmp_path).config, 0
    runner._read = dict.fromkeys(["total", "non_torch", "weights", "parameters"], 0)
    runner._read |= {"loaded": 1_000, "current": 9_000, "peak": 9_000}
    runner._read |= {"warmup_tokens": 0, "num_kvcache_blocks": 1}
    monkeypatch.setenv(measure_memory.OUT_ENV, str(tmp_path))

    runner.allocate_forward_vars()
    if bounds:
        assert built("cuda:0").pages == 50
    runner._write(None)

    (written,) = tmp_path.glob("readings.*.json")
    held = json.loads(written.read_text())["readings"]
    assert held.get(PAGE_BOUND) == (4_000 if bounds else None)
    member = MemoryReadings.from_member({**held, "record": "sha256:" + "e" * 64}, "")
    assert (PAGE_BOUND in member.values) is bounds


@pytest.mark.parametrize(
    "index, hf, reading, lacks",
    [
        ("fp4", V41, None, ["memory readings: stale"]),
        ("fp4", V41, 3, []),
        ("fp8", V41, None, []),
        ("fp4", None, None, []),
    ],
    ids=["fp4-v41-without", "fp4-v41-with", "fp8-v41", "fp4-other-model"],
)
def test_check_lists_an_fp4_v41_entry_without_the_page_bound_reading(
    tmp_path, monkeypatch, capsys, index, hf, reading, lacks
):
    fields = {"index_cache_dtype": index} | ({"hf_config": hf} if hf else {})
    r = deployment(tmp_path, **fields)
    if reading is not None:
        shutil.rmtree(tmp_path / "store" / Kind.MEMORY_READINGS.value)
        member = {**READINGS[1], PAGE_BOUND: reading}
        publish_readings(tmp_path / "store", r.config, readings=member)
    served(monkeypatch, r.config)

    assert checked(tmp_path, "--step-stub", stub(tmp_path)) == (1 if lacks else 0)

    out = capsys.readouterr().out
    assert lacked(out) == lacks
    if lacks:
        assert f"states no {PAGE_BOUND}" in out
        assert "made by: compass measure memory --gpu G --out D --store " in out
