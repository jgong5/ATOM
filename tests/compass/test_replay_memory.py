"""A GPU-free replay sized by a model, and the refusals that keep it honest.

`--compass-memory-model` and `--compass-replay-target` have always composed on
the command line. Until the seam these guard existed, only the device-backed
runner read the profile: the replay answered `get_num_blocks` out of the
capture, so the flag changed nothing and a scheduler planning from a measured
block count was read as a forecast of the modelled deployment.

So these tests are about capacity, not about a dict. Each one takes the reply
the replay runner gives the engine and builds ATOM's own `BlockManager` from
it -- the thing the scheduler actually asks "can this request be admitted" --
and asks whether the profile moved it. The refusal half is the same claim from
the other side: when the model cannot answer, nothing downstream may quietly
be planned from the capture instead.

The numbers here are the operational TP1/2/4 profiles' own. They are composed
from the TP=1 source calibration plus the standalone topology delta, so the
TP=2 and TP=4 cases are derived with no target-engine capture anywhere in
their lineage -- which is the property the widths were wanted for.
"""

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import pytest

from atom.compass.config import CompassConfig
from atom.compass.core.kv_geometry import InsufficientPoolBudget
from atom.compass.core.memory_blocks import derived_block_info, warmup_tokens
from atom.compass.core.memory_model import UnfoundedPrediction
from atom.compass.core.memory_topology import compose_calibration
from atom.compass.replay.runner import ReplayModelRunner
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.state_runtime import StateRuntime

ROOT = Path(__file__).resolve().parents[2]
NATIVE_CONFIG = ROOT / "tests" / "compass" / "memory_records" / "qwen3_5_27b.config.json"

#: MI308X. The target card's capacity is stated, never read off the box a
#: prediction happens to run on.
TOTAL = 206141652992
BUFFERS = 33554432

#: The TP=1 full-engine calibration (class S27) the topology delta composes on.
BASE = {
    "persistent": 252339712,
    "non_torch": {1: 1157627904},
    "load_residue": {1: 14924832},
    "provenance": {
        "persistent": "S27: 27B full engine, the source config",
        "non_torch": "S27: 27B full engine, the source config",
        "load_residue": "S27: 27B full engine, the source config",
    },
}

#: Read from the checkpoint's own safetensors headers at each width by
#: `emit_profile.py`; restated here so the suite needs no 55 GB checkpoint.
PARAMETERS = {1: 55562855904, 2: 27782542816, 4: 13892386272}

#: What the emitted profiles size at the registered acceptance cell. Pinned
#: rather than recomputed: a test that derives its own expectation from the
#: same code cannot notice the code moving.
BLOCKS = {1: 111969, 2: 266835, 4: 590328}

CAPTURED_BLOCKS = 4096


def _profile(tmp_path, width, *, total=TOTAL, calibration=None, **over):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    calibration = calibration or compose_calibration(BASE, width, env={})
    cal_path = tmp_path / ("calibration.tp%d.json" % width)
    cal_path.write_text(json.dumps(calibration))
    profile = {
        "total": total,
        "world_size": width,
        "parameters": PARAMETERS[width],
        "buffers": BUFFERS,
        "model_config": str(NATIVE_CONFIG),
        "compile_mode": "inductor",
        "dtype_bytes": 2,
        "calibration": str(cal_path),
    }
    for key, value in over.items():
        if value is None:
            profile.pop(key, None)
        else:
            profile[key] = value
    path = tmp_path / ("profile.tp%d.json" % width)
    path.write_text(json.dumps(profile))
    return str(path)


def _target(tmp_path):
    blob = {
        "version": 1,
        "blocks": {"num_kvcache_blocks": CAPTURED_BLOCKS,
                   "pool_entries": {"paged": CAPTURED_BLOCKS,
                                    STATE_SLOT_CLASS: 32},
                   "pool_entries_per_req": {STATE_SLOT_CLASS: 1},
                   "state_runtime": StateRuntime().to_wire()},
        "config": {"model": "Qwen/Qwen3.8-27B", "tensor_parallel_size": 1,
                   "max_model_len": 262144, "max_num_seqs": 32,
                   "gpu_memory_utilization": 0.9},
        "graph": {"capture_seconds": 12.5, "capture_sizes": [1, 8, 32],
                  "pool_bytes": 1 << 30},
    }
    path = tmp_path / "target.json"
    path.write_text(json.dumps(blob))
    return str(path)


def _config(tmp_path, *, profile="", width=1, **over):
    """The deployment being replayed -- the registered acceptance cell."""
    compass = CompassConfig(enabled=True, mode="predict",
                            replay_target=_target(tmp_path),
                            memory_model=profile or None)
    config = types.SimpleNamespace(
        compass_config=compass, model="Qwen/Qwen3.8-27B",
        tensor_parallel_size=width, pipeline_parallel_size=1,
        max_model_len=262144, max_num_seqs=32,
        max_num_batched_tokens=16384, gpu_memory_utilization=0.9,
        enforce_eager=False, kv_cache_block_size=16, kv_cache_dtype="auto",
        speculative_config=None,
        prefill_context_parallel_size=1, decode_context_parallel_size=1,
        parallel_config=None, compilation_config=None)
    for key, value in over.items():
        setattr(config, key, value)
    return config


def _planner(blocks):
    """ATOM's own block manager, built from the reply the engine would get.

    `engine_core` does exactly this: the per-class entry table travels over,
    `StateRuntime` is rebuilt from its wire form, and the scheduler plans from
    what comes out. Reproduced rather than asserted against so that what these
    tests measure is admission capacity and not a dictionary.
    """
    config = types.SimpleNamespace(
        kv_cache_block_size=16,
        num_kvcache_blocks=blocks["num_kvcache_blocks"],
        enable_prefix_caching=False,
        decode_context_parallel_size=1,
        pool_entries=dict(blocks["pool_entries"]),
        pool_entries_per_req=dict(blocks["pool_entries_per_req"]),
        state_checkpoint_interval_tokens=0,
        state_checkpoint_demand=False,
    )
    return BlockManager(
        config, state_runtime=StateRuntime.from_wire(blocks["state_runtime"]))


class TestAReplayWithNoProfileIsStillAReplay:
    """The flag that was always honoured has to keep behaving identically."""

    def test_the_captured_reply_comes_through_verbatim(self, tmp_path):
        runner = ReplayModelRunner(0, _config(tmp_path))
        blocks = runner.get_num_blocks()
        assert blocks["num_kvcache_blocks"] == CAPTURED_BLOCKS
        assert blocks["pool_entries"] == {"paged": CAPTURED_BLOCKS,
                                          STATE_SLOT_CLASS: 32}

    def test_the_planner_is_the_captured_size(self, tmp_path):
        runner = ReplayModelRunner(0, _config(tmp_path))
        planner = _planner(runner.get_num_blocks())
        assert planner.kv.num_free == CAPTURED_BLOCKS


class TestAProfileChangesWhatTheSchedulerMayHandOut:
    def test_the_pool_is_the_modelled_one_and_not_the_captured_one(self, tmp_path):
        runner = ReplayModelRunner(0, _config(tmp_path,
                                              profile=_profile(tmp_path, 1)))
        blocks = runner.get_num_blocks()
        assert blocks["num_kvcache_blocks"] == BLOCKS[1]
        assert _planner(blocks).kv.num_free == BLOCKS[1] != CAPTURED_BLOCKS

    def test_changing_the_profile_changes_the_capacity(self, tmp_path):
        """The claim the whole seam exists for, measured at the planner."""
        lean = _profile(tmp_path / "a", 1)
        heavy = _profile(
            tmp_path / "b", 1,
            calibration={"persistent": 252339712,
                         "non_torch": {"1": 1157627904 + (8 << 30)},
                         "load_residue": {"1": 14924832},
                         "provenance": dict(BASE["provenance"])})
        first = _planner(ReplayModelRunner(
            0, _config(tmp_path, profile=lean)).get_num_blocks()).kv.num_free
        second = _planner(ReplayModelRunner(
            0, _config(tmp_path, profile=heavy)).get_num_blocks()).kv.num_free
        assert first == BLOCKS[1]
        # 8 GiB more held outside torch is 8 GiB the pool does not get.
        assert 0 < second < first

    def test_the_wider_deployments_need_no_capture_of_their_own(self, tmp_path):
        """TP=2 and TP=4 from the TP=1 source plus the standalone delta.

        The capture in play is a TP=1 one and no target-engine reading at
        either width exists in this lineage -- that is the point of composing
        the widths rather than measuring them.
        """
        seen = {}
        for width in (2, 4):
            (tmp_path / str(width)).mkdir(parents=True, exist_ok=True)
            config = _config(tmp_path, width=width,
                             profile=_profile(tmp_path / str(width), width))
            seen[width] = ReplayModelRunner(0, config).get_num_blocks()
        assert seen[2]["num_kvcache_blocks"] == BLOCKS[2]
        assert seen[4]["num_kvcache_blocks"] == BLOCKS[4]
        assert _planner(seen[4]).kv.num_free == BLOCKS[4]

    def test_the_state_floor_reaches_the_planner_too(self, tmp_path):
        """A hybrid's admission is slot-bound before it is block-bound."""
        blocks = ReplayModelRunner(
            0, _config(tmp_path, profile=_profile(tmp_path, 1))).get_num_blocks()
        planner = _planner(blocks)
        assert planner.num_state_slots == blocks["pool_entries"][STATE_SLOT_CLASS]
        assert planner.num_state_slots > 0


class TestARefusalIsNeverAFallback:
    """Every one of these has a captured block count sitting right there."""

    def _refused(self, tmp_path, **over):
        runner = ReplayModelRunner(0, _config(tmp_path, **over))
        with pytest.raises(UnfoundedPrediction) as refusal:
            runner.get_num_blocks()
        assert str(CAPTURED_BLOCKS) not in str(refusal.value)
        return str(refusal.value)

    def test_a_profile_that_is_not_there(self, tmp_path):
        assert "nope.json" in self._refused(
            tmp_path, profile=str(tmp_path / "nope.json"))

    def test_a_profile_that_will_not_parse(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json")
        assert "could not be read" in self._refused(tmp_path, profile=str(path))

    def test_a_profile_missing_a_term(self, tmp_path):
        """`derived_readings` owns this refusal; the seam must not swallow it."""
        assert "parameter bytes" in self._refused(
            tmp_path, profile=_profile(tmp_path, 1, parameters=None))

    def test_a_profile_for_another_width(self, tmp_path):
        message = self._refused(tmp_path, width=4,
                                profile=_profile(tmp_path, 2))
        assert "TP=4" in message

    def test_a_pipeline_parallel_deployment(self, tmp_path):
        message = self._refused(tmp_path, pipeline_parallel_size=2,
                                profile=_profile(tmp_path, 1))
        assert "pipeline_parallel_size" in message

    def test_a_checkpoint_this_geometry_does_not_describe(self, tmp_path):
        native = json.loads(NATIVE_CONFIG.read_text())
        native["model_type"] = "llama"
        native.pop("text_config", None)
        path = tmp_path / "dense.json"
        path.write_text(json.dumps(native))
        message = self._refused(
            tmp_path, profile=_profile(tmp_path, 1, model_config=str(path)))
        assert "GDN hybrid" in message

    def test_an_infeasible_budget_carries_the_engines_own_error(self, tmp_path):
        """Compass calling a configuration infeasible is not Compass's verdict.

        `plan_pools` raises when the per-request state floor leaves nothing to
        page with. Letting that through rather than translating it keeps the
        refusal the engine's -- and keeps it from being caught as "the model
        did not answer, so use the capture".
        """
        profile = _profile(tmp_path, 1, total=16 << 30)
        runner = ReplayModelRunner(0, _config(tmp_path, profile=profile))
        with pytest.raises(InsufficientPoolBudget):
            runner.get_num_blocks()


class TestTheModelledPathTouchesNoDevice:
    def test_sizing_from_a_profile_reaches_no_cuda_call(self, tmp_path,
                                                        monkeypatch):
        """The gate claim, re-asserted on the path that now does arithmetic."""
        import torch

        def refuse(name):
            def go(*_a, **_k):
                raise AssertionError("replay touched torch.cuda.%s" % name)
            return go

        for name in ("is_available", "current_device", "set_device",
                     "mem_get_info", "memory_stats", "memory_allocated",
                     "memory_reserved", "synchronize", "init"):
            if hasattr(torch.cuda, name):
                monkeypatch.setattr(torch.cuda, name, refuse(name))

        runner = ReplayModelRunner(0, _config(tmp_path,
                                              profile=_profile(tmp_path, 1)))
        assert runner.get_num_blocks()["num_kvcache_blocks"] == BLOCKS[1]


class TestTheWarmupShapeIsTheOneTheRunnerUses:
    """`peak_torch` belongs to one prefill shape, and two files derive it.

    `memory_blocks.warmup_tokens` cannot import the device-backed runner's
    `_warmup_tokens` -- importing that module means importing `ModelRunner`,
    which needs a GPU, and this is the path for the machine without one. So
    the arithmetic is restated, and a restatement can drift. Compared as
    source text rather than by calling both, for the same reason.
    """

    @staticmethod
    def _arithmetic(path, name):
        source = Path(path).read_text()
        body = re.search(r"def %s\(self[^)]*\) -> int:(.*?)\n    def " % name,
                         source, re.S)
        if body is None:
            body = re.search(r"def %s\(config\) -> int:(.*?)\n\ndef " % name,
                             source, re.S)
        text = re.sub(r'""".*?"""', "", body.group(1), count=1, flags=re.S)
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # The runner reads `self.config`; this module is handed one.
            if line == "config = self.config":
                continue
            lines.append(line)
        return lines

    def test_the_two_derivations_are_the_same_arithmetic(self):
        runner = self._arithmetic(
            ROOT / "atom" / "compass" / "runtime" / "runner.py",
            "_warmup_tokens")
        shared = self._arithmetic(
            ROOT / "atom" / "compass" / "core" / "memory_blocks.py",
            "warmup_tokens")
        assert runner and runner == shared

    def test_the_acceptance_cell_warms_on_the_whole_token_budget(self, tmp_path):
        assert warmup_tokens(_config(tmp_path)) == 16384


class TestTheSeamIsCallableWithoutARunner:
    """The helper is the shared half; the runner owns only where it is called."""

    def test_it_answers_the_same_reply_the_runner_returns(self, tmp_path):
        config = _config(tmp_path, profile=_profile(tmp_path, 1))
        direct = derived_block_info(
            config.compass_config.memory_model, config,
            state_runtime=StateRuntime().to_wire())
        assert direct == ReplayModelRunner(0, config).get_num_blocks()

    def test_a_target_with_no_state_runtime_says_what_is_missing(self, tmp_path):
        config = _config(tmp_path, profile=_profile(tmp_path, 1))
        with pytest.raises(UnfoundedPrediction, match="state_runtime"):
            derived_block_info(config.compass_config.memory_model, config,
                               state_runtime=None)
