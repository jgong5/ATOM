# SPDX-License-Identifier: MIT
"""The step roofline over Kimi-K3's KDA layer: the recurrence, chunked on a
step with a prefill row and recurrent on a decode one, a snapshot a verified
position or, under ReplaySSM, a record a token; its gated output norm; and
what the walk reads off a KDA layer for them.

Each price is checked against a count by hand, on a machine with no kernel
floor, once memory-bound and once compute-bound.
"""

import copy
from types import SimpleNamespace

import pytest
from test_backend_step_linear import view
from test_backend_step_roofline import ROOFLINE

import atom.config
from atom.compass.backends.step_roofline import Module, StepRooflineBackend
from atom.compass.run import modules
from atom.compass.spec import MachineSpec

#: Two heads of 4, the state in fp32: 2 x 4 x 4 x 4 bytes a row.
KDA = (2, 4)
STATE = 2 * 4 * 4 * 4
#: A token reads q, k, the gate (a channel) and v, and beta (a head), and
#: writes o, in bf16; the gate's A_log and dt_bias are read once, in fp32.
TOKEN = 2 * (5 * 4 + 1) * 2
ONCE = 2 * (4 + 1) * 4
#: Three products with a 4 x 4 state a token and head.
PRODUCTS = 6 * 2 * 4 * 4


def _priced(bf16, priced, step):
    document = copy.deepcopy(ROOFLINE)
    document["device"]["kernel_floor_s"] = 1e-12
    document["device"]["memory"]["bandwidth_curve"] = [[1, 1e12], [2**40, 1e12]]
    document["device"]["compute"]["bf16_flops"] = bf16
    backend = StepRooflineBackend(MachineSpec.from_mapping(document), priced)
    return {t.name: t.seconds for t in backend.price(step).terms}


BOUND = pytest.mark.parametrize(
    "bf16", [2.5e15, 1e9], ids=["memory-bound", "compute-bound"]
)


@BOUND
def test_a_step_with_a_prefill_row_runs_the_chunked_kernel_by_hand(bf16):
    recurrence = Module("kda", "kda", KDA + (0,), "fp32")
    # A fresh 40-token prefill, 3 tokens over 7 cached, and a decode row.
    step = view((40, 40, False), (3, 10, False), (1, 20, True))
    by_name = _priced(bf16, [recurrence], step)
    # 44 tokens; the 2 rows with history read their state and all 3 write it.
    moved = 44 * TOKEN + ONCE + (2 + 3) * STATE
    # Chunks of 32: 40 runs one of 32 and one of 8, then 3 and 1; each chunk
    # of n runs 2 n^2 (3 x 4 + 2 x 4) in each of the 2 heads.
    squares = 32 * 32 + 8 * 8 + 3 * 3 + 1 * 1
    flops = 44 * PRODUCTS + 2 * 2 * squares * (3 * 4 + 2 * 4)

    assert (moved, flops) == (4376, 96288)
    assert by_name["roofline.kda"] == pytest.approx(
        max(moved / 1e12, flops / (bf16 * 0.70))
    )


@BOUND
@pytest.mark.parametrize("records", [0, 1], ids=["snapshots", "replayssm"])
def test_a_decode_step_writes_a_snapshot_a_position_or_records_by_hand(bf16, records):
    layer = [
        Module("kda", "kda", KDA + (records,), "fp32"),
        Module("o_norm", "gated_norm", (8,)),
        Module("o_norm_fp8", "gated_norm", (8,), "fp8"),
    ]
    # A decode row and a three-token verify, in a rung-4 graph.
    step = view((1, 30, True), (3, 50, True), rung=4)
    by_name = _priced(bf16, layer, step)
    # 4 real tokens; both rows read their state. The fused kernel writes it
    # after each of the 4 tokens; the replay kernel writes each token's k and
    # u in bf16 and g in fp32 instead, 2 heads of 4 each.
    moved = 4 * TOKEN + ONCE + 2 * STATE
    moved += 4 * 2 * 4 * (2 + 2 + 4) if records else 4 * STATE
    # The norm runs on the 4 tokens and the graph's 2 pad rows of 3: its
    # input and gate read in bf16, its output written in bf16, or fp8.
    bf16_out, fp8_out = 10 * 8 * (2 + 2 + 2), 10 * 8 * (2 + 2 + 1)

    assert moved == (888 if records else 1144)
    assert by_name["roofline.kda"] == pytest.approx(
        max(moved / 1e12, 4 * PRODUCTS / (bf16 * 0.70))
    )
    assert by_name["roofline.gated_norm"] == pytest.approx((bf16_out + fp8_out) / 1e12)


@pytest.mark.parametrize(
    "spec, override, dtype, expected",
    [
        (None, None, "fp32", ((12, 128, 0), "fp32")),
        (3, None, "fp32", ((12, 128, 1), "fp32")),
        (3, "0", "bf16", ((12, 128, 0), "bf16")),
        (None, "1", "fp16", ((12, 128, 1), "bf16")),
    ],
    ids=["decode", "speculative", "replayssm-off", "replayssm-on"],
)
def test_a_kda_layer_is_its_convolution_and_its_recurrence(
    monkeypatch, spec, override, dtype, expected
):
    """`GDNStateMixin`: ReplaySSM under speculative decoding unless
    `ATOM_ENABLE_REPLAYSSM` says otherwise, the state at `ATOM_GDN_SSM_DTYPE`."""
    speculative = spec and SimpleNamespace(num_speculative_tokens=spec)
    config = SimpleNamespace(speculative_config=speculative)
    monkeypatch.setattr(atom.config, "get_current_atom_config", lambda: config)
    monkeypatch.delenv("ATOM_ENABLE_REPLAYSSM", raising=False)
    if override is not None:
        monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", override)
    monkeypatch.setenv("ATOM_GDN_SSM_DTYPE", dtype)
    layer = SimpleNamespace(num_local_heads=12, head_dim=128, local_proj_size=1536)
    conv, recurrence = modules._kda(layer, "layers.0.self_attn")

    assert (conv.name, conv.kind, conv.dims) == (
        "layers.0.self_attn.conv",
        "conv",
        (4608,),
    )
    assert (recurrence.kind, (recurrence.dims, recurrence.fmt)) == ("kda", expected)
