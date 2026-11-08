# SPDX-License-Identifier: MIT
"""DeepSeek-V4's indexer cache picks its own kernels: at gfx950 ATOM resolves
fp4 for a plain DPA-TP8 V4-Pro server and fp8 under a Mooncake producer's
``--kv-transfer-config``. `keyed` names the `index_cache_dtype` ATOM resolved
where it differs from `kv_cache_dtype`, so the two key their `op_graph`s apart
and a step law fitted on one refuses the other, naming it; a config whose
index cache follows its KV cache keys as before, and a V4.1-Flash MI455 server
keys the engine config both of its laws state.

No device: aiter's arch probe answers a declared arch.
"""

import json
import pathlib

import pytest
from test_capture_v4_pro import CONFIG as V4_PRO
from test_capture_v4_pro import PRODUCER
from test_drive_runfile import build, install, step_linear
from test_run_step_grid import deployment

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass.capture.forward import structure
from atom.compass.drive.run import _config
from atom.compass.run.step_grid import BUCKETS, keyed

TREE = pathlib.Path(__file__).resolve().parents[2]
QWEN = pathlib.Path(__file__).with_name("qwen3_0_6b_config.json")
V41 = TREE / "tests/models/deepseek_v41/fixtures/config.json"
DPA_TP8 = ["-tp", "8", "--kv_cache_dtype", "fp8", "--enable-dp-attention"]
#: The server arguments and environment of the V4.1-Flash MI455 laws' records,
#: as scripts/compass/README.md states them.
V41_SERVER = (
    "--trust-remote-code -tp 4 --kv_cache_dtype bf16 --index-cache-dtype fp8 "
    "--method dspark --num-speculative-tokens 5 --gpu-memory-utilization 0.9 "
    "--max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 --block-size 16 "
    "--level 0 --max-num-seqs 64 --max-model-len 262144 "
    "--cudagraph-capture-sizes [1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64]"
)
V41_ENV = {
    "AITER_LOG_LEVEL": "WARNING",
    "OMP_NUM_THREADS": "4",
    BUCKETS: "256,512,1024,2048,3072",
}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in (*PERFORMANCE_KEYS, BUCKETS):
        monkeypatch.delenv(name, raising=False)


def served(tmp_path, monkeypatch, gfx, config_json, *args):
    """ATOM's config of a server of the model `config_json` describes, under
    ATOM's arguments `args`, on a `gfx` device, as `compass run` builds it."""
    monkeypatch.setattr("aiter.jit.utils.chip_info.get_gfx", lambda: gfx)
    model = tmp_path / config_json.stem
    model.mkdir(exist_ok=True)
    (model / "config.json").write_bytes(config_json.read_bytes())
    return _config(["--model", str(model), *args])


@pytest.fixture
def sides(tmp_path, monkeypatch):
    """DPA-TP8 V4-Pro on gfx950: a plain server's config, a Mooncake producer's."""
    return [
        served(tmp_path, monkeypatch, "gfx950", V4_PRO, *DPA_TP8, *transfer)
        for transfer in ((), PRODUCER.split())
    ]


def test_a_mooncake_producer_keys_apart_from_a_plain_server_at_gfx950(sides):
    plain, producer = sides

    assert keyed(plain) == keyed(producer) | {"index_cache_dtype": "fp4"}
    graphs = [structure(c, "decode", device="gfx950") for c in sides]
    assert graphs[0] != graphs[1]


@pytest.mark.parametrize(
    "fitted, named",
    [
        (1, "index_cache_dtype None in the law and 'fp4' here"),
        (0, "index_cache_dtype 'fp4' in the law and None here"),
    ],
    ids=["producer-law", "plain-law"],
)
def test_a_step_law_fitted_on_one_side_refuses_the_other_naming_it(
    sides, tmp_path, monkeypatch, fitted, named
):
    runners = [
        deployment(
            tmp_path, kv_cache_dtype="fp8", index_cache_dtype=c.index_cache_dtype
        )
        for c in sides
    ]
    run = build(
        tmp_path, "--step-linear", step_linear(tmp_path, runners[fitted]).key.token()
    )

    with pytest.raises(ValueError, match="fitted under another engine config") as e:
        install(tmp_path, monkeypatch, run, runners[1 - fitted])
    assert named in str(e.value)


@pytest.mark.parametrize(
    "gfx, config_json, args",
    [("gfx942", V4_PRO, DPA_TP8), ("gfx950", QWEN, [])],
    ids=["v4-pro-gfx942", "qwen3"],
)
def test_an_index_cache_that_follows_the_kv_cache_keys_as_before(
    tmp_path, monkeypatch, gfx, config_json, args
):
    config = served(tmp_path, monkeypatch, gfx, config_json, *args)

    assert config.index_cache_dtype == config.kv_cache_dtype
    assert "index_cache_dtype" not in keyed(config)


@pytest.mark.parametrize("ep", [[], ["--enable-expert-parallel"]], ids=["tp4", "ep4"])
def test_a_v41_flash_mi455_server_keys_the_engine_config_its_law_states(
    tmp_path, monkeypatch, ep
):
    law = f"deepseek-v4.1-flash-tp4-{'ep4-' * bool(ep)}dspark5-mi455-a0.json"
    stated = json.loads((TREE / "scripts/compass/laws" / law).read_text())
    for name, value in V41_ENV.items():
        monkeypatch.setenv(name, value)
    config = served(tmp_path, monkeypatch, "gfx1250", V41, *V41_SERVER.split(), *ep)
    # With no device aiter's fp8 dtype is uint8, which the quantization hash reads.
    quantization = {"quantization": stated["engine_config"]["quantization"]}

    engine = keyed(config) | quantization
    assert engine == stated["engine_config"] == stated["host"]["key"]["engine"]
