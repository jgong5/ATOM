# SPDX-License-Identifier: MIT
"""COMPASS_KEEP_COMPILED=1 (#3720): a real server's compiled-graph cache kept per configuration across starts."""

import hashlib
import json
import types
from pathlib import Path

from atom.compass import run
from atom.compass.drive import serve


class _Config:
    """ATOM's Config as `kept_cache_dir` reads it: a hash over the DP master's address and port, among the rest."""

    def __init__(self, port, tp=1):
        self.parallel_config = types.SimpleNamespace(
            data_parallel_master_ip="10.0.0.1", data_parallel_master_port=port, pipeline_parallel_rank=0
        )
        self.tensor_parallel_size, self.pipeline_parallel_size, self.model = tp, 4, "/m"
        self.compilation_config = types.SimpleNamespace(cache_dir="")

    def compute_hash(self):
        pc = self.parallel_config
        factors = [pc.data_parallel_master_ip, pc.data_parallel_master_port, self.tensor_parallel_size]
        return hashlib.sha256(str(factors).encode()).hexdigest()


def test_a_configuration_keeps_one_directory_whatever_its_ports(monkeypatch, tmp_path):
    a, inputs = run.kept_cache_dir(_Config(29500))
    b, _ = run.kept_cache_dir(_Config(31234))
    c, _ = run.kept_cache_dir(_Config(29500, tp=4))
    assert a == b != c and Path(a).name.startswith("kept-")
    assert inputs["blanked"]["data_parallel_master_port"] == 29500
    config = _Config(29500)
    assert config.parallel_config.data_parallel_master_port == 29500  # left as it was
    run._keep_compiled(config, str(tmp_path))
    assert config.compilation_config.cache_dir == a
    (note,) = tmp_path.glob("kept_cache.*.json")
    assert json.loads(note.read_text())["key"] == Path(a).name.removeprefix("kept-")


def test_the_stop_keeps_a_kept_directory_under_the_switch_only(monkeypatch, tmp_path):
    root = tmp_path / "torch_compile_cache"
    for d in ("kept-abc", "1234567890"):
        (root / d / "rank_0").mkdir(parents=True)
    log = tmp_path / "server.log"
    log.write_text("".join(f"Using cache directory: {root / d / 'rank_0'} for vLLM's torch.compile\n" for d in ("kept-abc", "1234567890")))
    monkeypatch.setattr(serve, "CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv(serve.KEEP_ENV, "1")
    serve.drop_compiled(log)
    assert sorted(p.name for p in root.iterdir()) == ["kept-abc"]
    monkeypatch.delenv(serve.KEEP_ENV)
    serve.drop_compiled(log)
    assert not list(root.iterdir())


def test_the_jit_gate_counts_no_write_in_atoms_cache(monkeypatch, tmp_path):
    from atom.compass.drive import aiter_jit

    home = tmp_path / "home"
    kept = home / ".cache" / "atom" / "torch_compile_cache" / "kept-abc" / "triton_cache"
    other = tmp_path / "triton"
    for d in (kept, other):
        d.mkdir(parents=True)
        (d / "k.hsaco").write_text("x")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(aiter_jit, "_roots", lambda: {kept: "triton", other: "triton"})
    assert aiter_jit.builds_since(0) == {aiter_jit.BUILDS: ["k"]}  # the other root's only
    monkeypatch.setattr(aiter_jit, "_roots", lambda: {kept: "triton"})
    assert aiter_jit.builds_since(0) == {}
