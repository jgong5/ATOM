# SPDX-License-Identifier: MIT
"""``ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION`` picks the TP all-reduce and norm
kernels, so a run with it off keys its `op_graph` and its step law apart from
one at ATOM's default, and a run at the default keys both as before.

No device: the graph key names its arch.
"""

from types import SimpleNamespace

import pytest
from test_op_graph_arch import ENGINE

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass.capture.forward import structure
from atom.compass.run.step_grid import keyed

FLAG = "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION"


@pytest.fixture
def config(tmp_path, monkeypatch):
    """DeepSeek-V3 at TP 8, no performance variable set."""
    for name in PERFORMANCE_KEYS:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    return SimpleNamespace(
        model=str(tmp_path / "model"),
        hf_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
        tensor_parallel_size=8,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        enable_expert_parallel=False,
        enforce_eager=True,
        **ENGINE,
    )


def test_the_fusion_flag_keys_the_graph_and_the_step_law(config, monkeypatch):
    on = structure(config, "decode", device="gfx942"), keyed(config, ladder=False)
    assert on[1]["env"] == {}
    monkeypatch.setenv(FLAG, "0")
    off = structure(config, "decode", device="gfx942"), keyed(config, ladder=False)
    assert off[1]["env"] == {FLAG: "0"}
    assert off[0] != on[0]
