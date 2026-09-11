"""GPU-free serving replay: the seam, and the promise that no device is touched.

These build the smallest thing that satisfies each contract rather than a real
deployment. What they are guarding is narrow and specific: that the replay
runner answers every startup RPC the engine makes, that it answers them from a
record rather than from hardware, and that constructing and stepping it reaches
no CUDA call at all. The last one is the claim the PoC gate rests on, so it is
asserted rather than assumed.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from atom.compass.config import CompassConfig
from atom.compass.replay.local_proc import LocalProcManager
from atom.compass.replay.runner import ReplayModelRunner, TargetRecord


def _load_script(name: str):
    """Import one of `scripts/compass/*.py`, which is not a package."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "compass" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compass_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _target(tmp_path, **over):
    blob = {
        "version": 1,
        "blocks": {"num_kvcache_blocks": 4096,
                   "pool_entries": {"full": 4096},
                   "pool_entries_per_req": {"full": 8},
                   "state_runtime": {"transfer": {"copies": []}}},
        "config": {"model": "Qwen/Qwen3.8-27B", "tensor_parallel_size": 1,
                   "max_model_len": 262144, "max_num_seqs": 32,
                   "gpu_memory_utilization": 0.9},
        "graph": {"capture_seconds": 12.5, "capture_sizes": [1, 8, 32],
                  "pool_bytes": 1 << 30},
    }
    blob.update(over)
    path = tmp_path / "target.json"
    path.write_text(json.dumps(blob))
    return str(path)


def _config(target: str, mode: str = "predict", **over):
    # `measure_out` only to satisfy CompassConfig's own validation for the
    # not-predict case; nothing here writes a table.
    compass = CompassConfig(enabled=True, mode=mode, replay_target=target,
                            measure_out="" if mode == "predict" else "/dev/null")
    config = types.SimpleNamespace(
        compass_config=compass, model="Qwen/Qwen3.8-27B",
        tensor_parallel_size=1, max_model_len=262144, max_num_seqs=32,
        gpu_memory_utilization=0.9, enforce_eager=False,
        prefill_context_parallel_size=1, decode_context_parallel_size=1,
        parallel_config=None, compilation_config=None)
    for key, value in over.items():
        setattr(config, key, value)
    return config


class TestTheRecordIsTheBoundary:
    def test_a_missing_target_says_how_to_get_one(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="replay-target-out"):
            TargetRecord.load(str(tmp_path / "nope.json"))

    def test_an_older_target_is_recaptured_not_reinterpreted(self, tmp_path):
        with pytest.raises(ValueError, match="version 0"):
            TargetRecord.load(_target(tmp_path, version=0))

    def test_a_target_that_never_sized_is_not_a_target(self, tmp_path):
        path = _target(tmp_path, blocks={"num_kvcache_blocks": 0})
        with pytest.raises(ValueError, match="no KV block count"):
            TargetRecord.load(path)

    def test_replaying_another_configuration_is_reported_not_refused(
            self, tmp_path, caplog):
        """Predicting a configuration the record was not captured from is the
        point; silence about the difference is not.

        This used to be asserted with `tensor_parallel_size`, and that one case
        has since moved to a refusal -- see
        `TestTheLogicalWidthAndTheExecutorCount`. The block count and pool
        entries here *were* measured at the record's width and move with it, so
        carrying them into a wider replay sizes the wrong deployment; a
        difference in how many sequences the scheduler may admit does not.
        The reporting contract is unchanged and is what this still guards.
        """
        config = _config(_target(tmp_path), max_num_seqs=64)
        with caplog.at_level("WARNING"):
            runner = ReplayModelRunner(0, config)
        assert "max_num_seqs: captured 32, replaying 64" in caplog.text
        assert runner.get_num_blocks()["num_kvcache_blocks"] == 4096


class TestTheStartupRPCsAreAnswered:
    def test_the_block_layout_comes_through_verbatim(self, tmp_path):
        runner = ReplayModelRunner(0, _config(_target(tmp_path)))
        blocks = runner.get_num_blocks()
        assert blocks["pool_entries"] == {"full": 4096}
        assert blocks["state_runtime"] == {"transfer": {"copies": []}}

    def test_the_cache_is_accounted_for_and_not_allocated(self, tmp_path):
        runner = ReplayModelRunner(0, _config(_target(tmp_path)))
        assert runner.allocate_kv_cache(4096) is True

    def test_capture_reports_what_it_cost_the_real_run(self, tmp_path):
        """Zero here would silently drop a startup cost from amortisation."""
        runner = ReplayModelRunner(0, _config(_target(tmp_path)))
        seconds, sizes, pool = runner.capture_cudagraph()
        assert (seconds, sizes, pool) == (12.5, [1, 8, 32], 1 << 30)

    def test_a_replay_of_a_traced_run_is_a_configuration_error(self, tmp_path):
        with pytest.raises(ValueError, match="predict mode"):
            ReplayModelRunner(0, _config(_target(tmp_path), mode="measure"))

    def test_an_unmodelled_rpc_names_itself(self, tmp_path):
        runner = ReplayModelRunner(0, _config(_target(tmp_path)))
        with pytest.raises(AttributeError, match="load_weights"):
            runner.load_weights


class TestNoDeviceIsTouched:
    def test_construction_reaches_no_cuda_call(self, tmp_path, monkeypatch):
        """The gate claim, asserted.

        Every CUDA entry point the runner could plausibly reach is replaced
        with something that fails the test rather than returning a plausible
        answer, so a device touch cannot pass by being harmless.
        """
        import torch

        touched = []

        def refuse(name):
            def go(*_a, **_k):
                touched.append(name)
                raise AssertionError(f"replay touched torch.cuda.{name}")
            return go

        for name in ("is_available", "current_device", "set_device",
                     "mem_get_info", "memory_stats", "memory_allocated",
                     "memory_reserved", "synchronize", "init"):
            if hasattr(torch.cuda, name):
                monkeypatch.setattr(torch.cuda, name, refuse(name))

        runner = ReplayModelRunner(0, _config(_target(tmp_path)))
        runner.get_num_blocks()
        runner.allocate_kv_cache(4096)
        runner.capture_cudagraph()
        runner.warmup_model()
        assert touched == []


class TestThePoolWithoutProcesses:
    def test_it_spawns_nothing(self, tmp_path):
        mgr = LocalProcManager(lambda: None, 1,
                               "atom.compass.replay.runner.ReplayModelRunner",
                               _config(_target(tmp_path)))
        assert mgr.procs == []
        assert mgr.call_func("get_num_blocks",
                             wait_out=True)["num_kvcache_blocks"] == 4096

    def test_fire_and_forget_still_runs_the_call(self, tmp_path):
        mgr = LocalProcManager(lambda: None, 1,
                               "atom.compass.replay.runner.ReplayModelRunner",
                               _config(_target(tmp_path)))
        assert mgr.call_func("allocate_kv_cache", 4096) is None

    def test_an_rpc_the_runner_lacks_is_named_not_hung(self, tmp_path):
        mgr = LocalProcManager(lambda: None, 1,
                               "atom.compass.replay.runner.ReplayModelRunner",
                               _config(_target(tmp_path)))
        with pytest.raises(AttributeError, match="start_profiler"):
            mgr.call_func("start_profiler", wait_out=True)

    def test_more_than_one_executor_is_refused_with_the_reason(self, tmp_path):
        # The message names the executor count, not the logical width: see
        # `TestTheLogicalWidthAndTheExecutorCount`, where the two are pulled
        # apart. A wide deployment is replayable; a second executor is not.
        with pytest.raises(ValueError, match="runs one executor"):
            LocalProcManager(lambda: None, 4,
                             "atom.compass.replay.runner.ReplayModelRunner",
                             _config(_target(tmp_path)))

    def test_exit_closes_the_runner_and_calls_the_finalizer(self, tmp_path):
        called = []
        mgr = LocalProcManager(lambda: called.append(True), 1,
                               "atom.compass.replay.runner.ReplayModelRunner",
                               _config(_target(tmp_path)))
        mgr.exit()
        mgr.exit()
        assert called == [True]


class TestTheMixinSitsInFrontOfTheRealRunner:
    """The split has one sharp edge, and it cost a run to find.

    ``CompassModelRunner(CompassPredictMixin, ModelRunner)`` puts the mixin
    *between* the class and the real runner, and the mixin defines ``forward``.
    So ``super().forward(batch)`` written inside ``CompassModelRunner`` no
    longer means "the real forward pass" -- it lands back on the mixin's
    dispatcher, which calls ``_forward_measured``, which calls it again. The
    server came up, served nothing, and died with a recursion trace.

    Checked on the source rather than by importing, so it runs on a machine
    with no GPU -- ``ModelRunner`` cannot be imported without one.
    """

    def _runner_source(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        return (root / "atom" / "compass" / "runtime" / "runner.py").read_text()

    def test_the_real_forward_is_named_not_deferred_to_super(self):
        assert "super().forward(" not in self._runner_source()
        assert "ModelRunner.forward(self, batch)" in self._runner_source()

    def test_the_mixin_is_still_what_defines_forward(self):
        """If the bases were reordered to dodge the above, predict would die."""
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2]
        source = (root / "atom" / "compass" / "runtime" / "runner.py").read_text()
        bases = re.search(r"class CompassModelRunner\(([^)]*)\)", source).group(1)
        names = [b.strip() for b in bases.split(",")]
        assert names.index("CompassPredictMixin") < names.index("ModelRunner")


class TestTheArchComesFromTheCaptureNotTheHost:
    """The one thing a GPU-free replay is told about the hardware.

    AITER resolves an architecture name at import time and cannot be imported
    without one. These pin what the bootstrap may answer with, and -- more
    importantly -- what it may not.
    """

    @staticmethod
    def _clean(monkeypatch):
        from atom.compass.replay import bootstrap

        for name in list(sys.modules):
            if name == "jax" or name.startswith("jax."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        # The bootstrap refuses once AITER is imported, which is correct and is
        # tested below. Another module in this suite may have imported it on a
        # GPU host, so these tests restore the situation they are about: the
        # bootstrap running first.
        monkeypatch.delitem(sys.modules, "aiter", raising=False)
        for finder in list(sys.meta_path):
            if type(finder).__name__ == "_ChipInfoFinder":
                sys.meta_path.remove(finder)
        monkeypatch.setattr(bootstrap, "_STATE",
                            {"installed": False, "arch": None,
                             "gpu_archs": None, "source": None, "reason": None,
                             "calls": 0, "chip_info_hook": False,
                             "chip_info_calls": 0})
        monkeypatch.setattr(bootstrap, "_live_arch", lambda: None)
        monkeypatch.setattr(bootstrap, "_jax_installed", lambda: False)
        return bootstrap

    def test_the_captured_arch_answers_aiters_fallback(self, monkeypatch):
        bootstrap = self._clean(monkeypatch)
        bootstrap.install("gfx942:sramecc+:xnack-", source="test")
        from jax._src.lib import gpu_triton

        # Verbatim, suffixes and all: AITER does the `.split(":")[0]` itself,
        # and trimming it here would make this a second opinion about the arch.
        assert gpu_triton.get_arch_details("0") == "gfx942:sramecc+:xnack-"
        assert bootstrap.state()["installed"] is True
        assert bootstrap.state()["calls"] == 1

    def test_the_jit_is_told_the_bare_chip(self, monkeypatch):
        """AITER's JIT shells out to `rocminfo` unless `GPU_ARCHS` names it.

        Bare, because `GPU_ARCHS` is matched against a table of chip names and
        the suffixes are not part of one.
        """
        bootstrap = self._clean(monkeypatch)
        monkeypatch.delenv("GPU_ARCHS", raising=False)
        state = bootstrap.install("gfx942:sramecc+:xnack-", source="test")
        assert os.environ["GPU_ARCHS"] == "gfx942"
        assert state["gpu_archs"] == "gfx942"

    def test_the_runtime_query_is_answered_too(self, monkeypatch):
        """`get_gfx_runtime` ignores `GPU_ARCHS` by design, and runs on import."""
        bootstrap = self._clean(monkeypatch)
        state = bootstrap.install("gfx942:sramecc+:xnack-", source="test")
        assert state["chip_info_hook"] is True
        finders = [f for f in sys.meta_path
                   if type(f).__name__ == "_ChipInfoFinder"]
        assert len(finders) == 1
        assert finders[0].find_spec("aiter.jit.utils.something_else") is None

    def test_a_visible_device_keeps_native_detection(self, monkeypatch):
        bootstrap = self._clean(monkeypatch)
        monkeypatch.setattr(bootstrap, "_live_arch", lambda: "gfx942")
        monkeypatch.delenv("GPU_ARCHS", raising=False)
        state = bootstrap.install("gfx942", source="test")
        assert "GPU_ARCHS" not in os.environ
        assert state["chip_info_hook"] is False

    def test_only_a_name_is_supplied(self, monkeypatch):
        """No fake kernels. The stub has exactly one function on it."""
        bootstrap = self._clean(monkeypatch)
        bootstrap.install("gfx942", source="test")
        from jax._src.lib import gpu_triton

        assert [n for n in vars(gpu_triton) if not n.startswith("__")] == [
            "get_arch_details"]
        assert "aiter" not in sys.modules

    def test_a_visible_device_answers_first_and_nothing_is_installed(
            self, monkeypatch):
        """The bootstrap cannot be used to claim a GPU run was GPU-free."""
        bootstrap = self._clean(monkeypatch)
        monkeypatch.setattr(bootstrap, "_live_arch", lambda: "gfx942")
        state = bootstrap.install("gfx942", source="test")
        assert state["installed"] is False
        assert state["source"] == "live-triton-driver"
        assert "jax" not in sys.modules

    def test_a_host_device_that_disagrees_with_the_capture_is_reported(
            self, monkeypatch, caplog):
        bootstrap = self._clean(monkeypatch)
        monkeypatch.setattr(bootstrap, "_live_arch", lambda: "gfx950")
        with caplog.at_level(logging.WARNING):
            bootstrap.install("gfx942", source="test")
        assert "this host's GPU is gfx950" in caplog.text

    def test_installed_jax_is_left_alone(self, monkeypatch, caplog):
        bootstrap = self._clean(monkeypatch)
        monkeypatch.setattr(bootstrap, "_jax_installed", lambda: True)
        with caplog.at_level(logging.WARNING):
            state = bootstrap.install("gfx942", source="test")
        assert state["installed"] is False and state["source"] == "installed-jax"
        assert "jax" not in sys.modules

    def test_arriving_after_aiter_refuses(self, monkeypatch):
        """`_CACHED_ARCH` is resolved once; a later answer is never read."""
        bootstrap = self._clean(monkeypatch)
        monkeypatch.setitem(sys.modules, "aiter", types.ModuleType("aiter"))
        with pytest.raises(RuntimeError, match="already imported"):
            bootstrap.install("gfx942", source="test")

    def test_a_target_without_hardware_says_to_recapture(self, monkeypatch,
                                                         tmp_path):
        bootstrap = self._clean(monkeypatch)
        path = _target(tmp_path)   # written by the pre-`hardware` shape
        with pytest.raises(bootstrap.ArchUnavailable, match="hardware.arch"):
            bootstrap.install_from_target(path)

    def test_a_target_with_hardware_is_used_and_names_its_source(
            self, monkeypatch, tmp_path):
        bootstrap = self._clean(monkeypatch)
        path = _target(tmp_path, hardware={"arch": "gfx942",
                                           "device_name": "MI308X"})
        state = bootstrap.install_from_target(path)
        assert state["arch"] == "gfx942" and "MI308X" in state["source"]

    def test_the_record_carries_hardware_through(self, tmp_path):
        from atom.compass.replay.runner import TargetRecord

        record = TargetRecord.load(_target(tmp_path,
                                           hardware={"arch": "gfx942"}))
        assert record.hardware["arch"] == "gfx942"

    def test_an_old_target_still_loads(self, tmp_path):
        """`hardware` is additive: it must not invalidate what came before."""
        from atom.compass.replay.runner import TargetRecord

        assert TargetRecord.load(_target(tmp_path)).hardware == {}


class TestTheLauncherRefusesToGuess:
    def test_no_target_is_an_error_not_a_default(self):
        module = _load_script("replay_server")
        with pytest.raises(SystemExit, match="--compass-replay-target"):
            module._target_from_argv(["--model", "x"])

    def test_both_spellings_of_the_flag_are_read(self):
        module = _load_script("replay_server")
        assert module._target_from_argv(
            ["--compass-replay-target", "a.json"]) == "a.json"
        assert module._target_from_argv(
            ["--compass-replay-target=b.json"]) == "b.json"


class TestTheChildProcessesAreToldToo:
    """`spawn` starts a fresh interpreter; `sys.modules` does not survive it."""

    @staticmethod
    def _blind() -> dict:
        """A child environment with no device, whatever this host has.

        The hook is only reached when Triton finds no driver, so a test run on
        a GPU machine has to hide the GPU or it is testing the other branch.
        """
        return {**os.environ, "HIP_VISIBLE_DEVICES": "",
                "CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}

    @staticmethod
    def _sitedir() -> Path:
        return (Path(__file__).resolve().parents[2] / "atom" / "compass"
                / "replay" / "_sitedir")

    def test_the_hook_is_inert_without_the_variable(self, tmp_path):
        out = subprocess.run(
            [sys.executable, "-c", "import sys; print('jax' in sys.modules)"],
            env={**self._blind(), "PYTHONPATH": str(self._sitedir()),
                 "ATOM_COMPASS_REPLAY_ARCH": ""},
            capture_output=True, text=True, timeout=120)
        assert out.stdout.strip() == "False", out.stderr

    def test_a_fresh_interpreter_gets_the_arch(self):
        out = subprocess.run(
            [sys.executable, "-c",
             "from jax._src.lib import gpu_triton as g;"
             " print(g.get_arch_details('0'))"],
            env={**self._blind(), "PYTHONPATH": str(self._sitedir()),
                 "ATOM_COMPASS_REPLAY_ARCH": "gfx942:sramecc+:xnack-"},
            capture_output=True, text=True, timeout=120)
        assert out.stdout.strip() == "gfx942:sramecc+:xnack-", out.stderr

    def test_the_hook_holds_only_itself(self):
        """Anything else here would be on every child's path, unannounced."""
        assert sorted(p.name for p in self._sitedir().iterdir()
                      if p.name != "__pycache__") == ["sitecustomize.py"]

    def test_the_real_sitecustomize_still_runs(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        (real / "sitecustomize.py").write_text(
            "import os; os.environ['ATOM_TEST_CHAINED'] = 'yes'\n")
        out = subprocess.run(
            [sys.executable, "-c",
             "import os; print(os.environ.get('ATOM_TEST_CHAINED'))"],
            env={**self._blind(),
                 "PYTHONPATH": os.pathsep.join([str(self._sitedir()),
                                                str(real)]),
                 "ATOM_COMPASS_REPLAY_ARCH": "gfx942"},
            capture_output=True, text=True, timeout=120)
        assert out.stdout.strip() == "yes", out.stderr


def _tp4(tmp_path):
    """A target and a config that agree on a logical width of four."""
    target = _target(tmp_path, config={
        "model": "Qwen/Qwen3.8-27B", "tensor_parallel_size": 4,
        "max_model_len": 262144, "max_num_seqs": 32,
        "gpu_memory_utilization": 0.9})
    return _config(target, tensor_parallel_size=4)


class TestTheLogicalWidthAndTheExecutorCount:
    """Two numbers that a single-process replay makes it tempting to conflate.

    The logical width is how wide the deployment under evaluation is: it shards
    the weights, sizes the collectives, places the head, and is what the run
    says it predicted. The executor count is how many processes hold a runner,
    and here it is one, because the step is priced rather than run.

    Collapsing them in the direction that matters -- running one executor and
    reporting its work as a TP4 group's -- is a TP1 result wearing a TP4 label.
    So the one executor is rank 0 *of the logical group*, and the group's cost
    is the oracle's answer at those coordinates plus the modelled collectives.
    """

    def test_a_wide_deployment_replays_on_one_executor(self, tmp_path):
        mgr = LocalProcManager(lambda: None, 1,
                               "atom.compass.replay.runner.ReplayModelRunner",
                               _tp4(tmp_path))
        assert mgr.logical_tp == 4
        assert mgr.procs == []
        assert mgr.runner.logical_tp == 4
        assert mgr.runner.physical_executors == 1

    def test_the_oracle_is_asked_about_the_group_not_the_executor(self, tmp_path):
        """The seam the whole contract rests on.

        `StepShape` carries `topology` and `rank_coords` to the oracle. If the
        topology reported the executor count, every price would be a TP1 price
        and the collectives would vanish -- the label would say TP4 and nothing
        underneath it would.
        """
        runner = ReplayModelRunner(0, _tp4(tmp_path))
        assert runner._topology() == {"tp": 4}
        # Rank 0 of four, not the whole of one.
        assert runner._rank_coords() == {"tp": 0}

    def test_a_record_sized_at_another_width_is_refused(self, tmp_path):
        """Refused rather than warned, because block count moves with width.

        Everything else in `disagreements` is reported and carried; this one is
        not, because handing a TP1 block count to a TP4 scheduler lets it admit
        a workload the target cannot hold and then reports the throughput of
        the schedule that followed.
        """
        config = _config(_target(tmp_path), tensor_parallel_size=4)
        with pytest.raises(ValueError, match="TP1 deployment.*logical TP4"):
            ReplayModelRunner(0, config)

    def test_pipeline_stages_are_refused_by_name(self, tmp_path):
        config = _config(_target(tmp_path), pipeline_parallel_size=2)
        with pytest.raises(ValueError, match="pipeline stages"):
            ReplayModelRunner(0, config)

    def test_more_than_one_executor_names_what_asked_for_it(self, tmp_path):
        """`tp_world_size` is already 1 under a replay, so TP is not the cause.

        Blaming the logical width here would send a reader to lower `-tp`,
        which is the one thing that is legitimately wide.
        """
        config = _config(_target(tmp_path), prefill_context_parallel_size=2)
        with pytest.raises(ValueError, match="one executor.*context-parallel"):
            LocalProcManager(lambda: None, 2,
                             "atom.compass.replay.runner.ReplayModelRunner",
                             config)


class TestConfigCollapsesTheExecutorCountNotTheWidth:
    """`Config.tp_world_size`, exercised as the function it is.

    Building a real `Config` needs a model on disk and a device to count, and
    neither is available to a unit test. So the two properties are borrowed --
    both of them, because `tp_world_size` reads the other one, and a stub
    carrying only plain attributes raises on the read rather than exercising
    it.
    """

    @staticmethod
    def _fget(**attrs):
        from atom.config import Config

        stub = type("_ConfigStub", (), {
            "tp_world_size": Config.tp_world_size,
            "_compass_replay_active": Config._compass_replay_active,
        })()
        for name, value in attrs.items():
            setattr(stub, name, value)
        return stub.tp_world_size

    def test_a_replay_gets_one_executor_at_any_width(self):
        compass = CompassConfig(enabled=True, mode="predict",
                                replay_target="/tmp/t.json", measure_out="")
        assert self._fget(tensor_parallel_size=4, fake_eplb=False,
                          compass_config=compass) == 1

    def test_a_compass_run_without_a_replay_target_is_untouched(self):
        # Predict on real hardware: the forward is priced, but there are still
        # four ranks each holding a quarter of the model, and each still needs
        # a process. Only the replay has no device for them to hold.
        compass = CompassConfig(enabled=True, mode="predict")
        assert self._fget(tensor_parallel_size=4, fake_eplb=False,
                          compass_config=compass) == 4

    def test_a_run_with_no_compass_at_all_is_untouched(self):
        assert self._fget(tensor_parallel_size=4, fake_eplb=False,
                          compass_config=None) == 4
