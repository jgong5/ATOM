# SPDX-License-Identifier: MIT
"""``ATOM_MONO_ENABLE`` picks between a mono model's fused decode kernels and
its original path, so a run with it off keys its `op_graph` and its step law
apart from one at ATOM's default, and a run at the default keys both as before.

No device: the graph key names its arch.
"""

from types import SimpleNamespace

import pytest
from test_op_graph_arch import ENGINE

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass.capture.forward import structure
from atom.compass.run.step_grid import keyed

FLAG = "ATOM_MONO_ENABLE"


@pytest.fixture
def config(tmp_path, monkeypatch):
    """DeepSeek-V4.1-Flash at TP 4 with expert parallelism off, no performance
    variable set."""
    for name in PERFORMANCE_KEYS:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    return SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"]),
        tensor_parallel_size=4,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_expert_parallel=False,
        enforce_eager=True,
        **ENGINE,
    )


def test_the_mono_flag_keys_the_graph_and_the_step_law(config, monkeypatch):
    on = structure(config, "decode", device="gfx950"), keyed(config, ladder=False)
    assert on[1]["env"] == {}
    monkeypatch.setenv(FLAG, "0")
    off = structure(config, "decode", device="gfx950"), keyed(config, ladder=False)
    assert off[1]["env"] == {FLAG: "0"}
    assert off[0] != on[0]
