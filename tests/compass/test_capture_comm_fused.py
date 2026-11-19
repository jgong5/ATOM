# SPDX-License-Identifier: MIT
"""aiter's comm-fused MoE in a device-free capture of DeepSeek-V4-Pro: the
declared MI355X names its product, so ATOM takes the comm-fused MoE wherever
aiter holds a winner for the layer's rows, and the capture traces it as one
leaf. The op-level backend prices that leaf from its work.
"""

import os
from unittest.mock import patch

import pytest
from test_capture_ir import _decoder
from test_capture_v4_pro import DP, DPA_TP8, LAYERS, MOE, _pars, _rows, capture
from test_moe_gather_pad_rows import gathered
from test_op_work_v4 import close, moe

from atom.compass.backends.op_work import work
from atom.compass.backends.work import allreduce, reduce_scatter
from atom.compass.capture.peers import COMM_FUSED

#: The interleaved gate-up layout, the only one ATOM takes aiter's comm-fused
#: MoE under.
INTERLEAVED = {"ATOM_MOE_GU_ITLV": "1"}
#: The standard path's routed experts.
FUSED_MOE = "aiter.fused_moe_.default"


def _layers(tmp_path_factory, argv, shape):
    """Each decoder layer's leaves of `shape`, captured over `argv` with the
    interleaved layout."""
    with patch.dict(os.environ, INTERLEAVED):
        graph, _ = capture(tmp_path_factory, argv)[shape]
    layers = _decoder(graph)
    assert len(layers) == LAYERS
    return graph, layers


#: DP-attention shapes: 8, 64 and 4096 gathered rows.
DPA = "--width 1 --width 8 --prefill 512 1" + DPA_TP8


@pytest.mark.parametrize(
    "shape, rows, sizes",
    # From 64 gathered rows aiter holds reduce-scatter winners; at 4096 its
    # runner takes each rank's rows as they are, sized.
    [("decode B=8", 8, None), ("prefill T=512 B=1", 512, [512] * DP)],
)
def test_dpa_steps_aiter_holds_a_winner_for_run_the_comm_fused_moe(
    shape, rows, sizes, tmp_path_factory
):
    graph, layers = _layers(tmp_path_factory, DPA, shape)

    for layer in layers:
        names = [op["op"] for op in layer]
        assert FUSED_MOE not in names
        assert f"{MOE}:reduce_scatter_with_unpadding" not in names
        [gather] = [op for op in layer if op["op"] == f"{MOE}:all_gatherv"]
        [fused] = [op for op in layer if op["op"] == COMM_FUSED]
        assert names.index(gather["op"]) < names.index(COMM_FUSED)
        assert _rows(gather) == (rows, DP * rows)
        # Every rank's rows in, this rank's back, reduced across the DP group.
        assert _rows(fused) == (DP * rows, rows)
        args = fused["args"]
        assert (fused["kind"], args["tp"], args["comm"]) == ("declared", DP, "rs")
        assert args["reduce_scatter_sizes"] == sizes
    # The shared expert still runs on its own stream beside it.
    for routed, shared in _pars(graph["region"]):
        assert COMM_FUSED in routed and "aiter.silu_and_mul.default" in shared


def test_a_dpa_decode_of_one_row_a_rank_keeps_the_standard_path(tmp_path_factory):
    # Eight gathered rows: aiter's winners hold no bucket that small.
    _, layers = _layers(tmp_path_factory, DPA, "decode B=1")

    for layer in layers:
        names = [op["op"] for op in layer]
        assert COMM_FUSED not in names
        assert names.count(FUSED_MOE) == 1
        assert names.count(f"{MOE}:reduce_scatter_with_unpadding") == 1


def test_under_two_batch_overlap_the_moe_keeps_the_standard_path(tmp_path_factory):
    argv = "--width 8" + DPA_TP8 + " --enable-tbo"
    _, layers = _layers(tmp_path_factory, argv, "decode B=8")

    for layer in layers:
        names = [op["op"] for op in layer]
        assert COMM_FUSED not in names
        assert names.count(FUSED_MOE) == 1


def test_tp8_reduces_its_moe_inside_the_comm_fused_leaf(tmp_path_factory):
    # aiter's all-reduce winners hold two rows, and 4096, a prefill above the
    # dual-stream threshold, which runs ATOM's custom op.
    argv = "--width 2 --prefill 4096 1 -- --kv_cache_dtype fp8"
    for shape, rows in (("decode B=2", 2), ("prefill T=4096 B=1", 4096)):
        _, layers = _layers(tmp_path_factory, argv, shape)

        for layer in layers:
            names = [op["op"] for op in layer]
            # The attention's output projection reduces alone.
            assert names.count("aiter.all_reduce_.default") == 1
            [fused] = [op for op in layer if op["op"] == COMM_FUSED]
            assert _rows(fused) == (rows, rows)
            assert (fused["args"]["tp"], fused["args"]["comm"]) == (8, "ar")


@pytest.mark.parametrize(
    "comm, reduce", [("rs", reduce_scatter), ("ar", allreduce)], ids=["rs", "ar"]
)
def test_the_comm_fused_leaf_prices_its_experts_and_the_reduction_it_fuses(
    comm, reduce
):
    # 16 rows in, this rank's 2 of eight ranks' back: the experts as aiter's
    # fused-MoE leaf of the same arguments prices them, and the rows' bf16
    # bytes reduced across the eight.
    experts = moe(16, 16) | {"out": [[2, 64]]}
    fused = experts | {"op": COMM_FUSED}
    fused["args"] = experts["args"] | {"tp": 8, "comm": comm}
    view = gathered(16, 16)

    got = work(fused, view, 1, "bf16")

    close(got, work(experts, view, 1, "bf16") + reduce(8, 16 * 64 * 2))
