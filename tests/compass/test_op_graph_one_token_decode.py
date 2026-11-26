# SPDX-License-Identifier: MIT
"""A decode step of one token a row under an MTP drafter (#4210).

A prefill instance under an MTP drafter runs decode steps whose rows each
decode one token, verifying no draft. `OpGraphBackend` reads such a step's
graph, its target's and its drafter's, at its shape verifying none (``V=0``),
and `compass trace ops --population` traces it there.
"""

import pytest
from test_backend_step_linear import view
from test_capture_ir import _record
from test_op_graph import g
from test_work import SPEC

from atom.compass.backends import CostRefused
from atom.compass.backends.op_graph import OpGraphBackend, step_shape, verified
from atom.compass.backends.work import gemm, price
from atom.compass.capture import population_shapes
from atom.compass.capture.forward import decode

#: Two rows decoding one token each, replayed at rung 2.
ONE_A_ROW = view((1, 90, True), (1, 9, True), rung=2)


@pytest.mark.parametrize("drafts", [1, 3])
def test_a_decode_of_one_token_a_row_under_a_drafter_reads_graphs_verifying_none(
    drafts,
):
    asked = []

    def graph(*at, draft=False, verifies=None):
        asked.append((*at, draft, verifies))
        return g(16, 64, 64) if draft else g(4, 8, 8)

    cost = OpGraphBackend(SPEC, graph, kv="bf16", drafts=drafts).price(ONE_A_ROW)

    assert asked == [("decode", 2, 2, True, False, 0), ("decode", 2, 2, True, True, 0)]
    assert (cost.target_seconds, cost.draft_seconds) == (
        price(gemm(4, 8, 8), SPEC),
        price(gemm(16, 64, 64), SPEC),
    )


def test_a_verify_step_reads_its_graphs_at_its_drafter_s_drafts():
    asked = []

    def graph(*at, **kwargs):
        asked.append(kwargs)
        return g(4, 8, 8)

    batch = view((2, 90, True), (2, 9, True), rung=2)
    OpGraphBackend(SPEC, graph, kv="bf16", drafts=1).price(batch)

    assert asked == [{}, {"draft": True}]


def test_a_decode_mixing_one_token_rows_and_verify_rows_is_refused_by_name():
    batch = view((1, 90, True), (2, 9, True), rung=2)
    with pytest.raises(CostRefused, match="decodes a row of other than 2 tokens"):
        OpGraphBackend(SPEC, lambda *a, **k: g(4, 8, 8), kv="bf16", drafts=1).price(
            batch
        )


def test_a_graph_missing_at_a_shape_verifying_none_is_named_so():
    with pytest.raises(CostRefused, match="verifying 0 drafts a row"):
        OpGraphBackend(SPEC, lambda *a, **k: None, kv="bf16", drafts=1).price(ONE_A_ROW)


def test_verified_and_step_shape_place_a_one_token_decode_under_a_drafter():
    rows = [(1, True), (1, True)]
    assert verified(rows, 3) == 0
    assert verified([(4, True)], 3) == 3
    assert verified(rows, 0) == 0
    assert step_shape(rows, 4, 3) == ("decode", 4, 4)


def test_a_population_under_a_drafter_traces_a_one_token_decode_verifying_none(
    tmp_path,
):
    record = _record(
        tmp_path,
        [
            {"rows": [[1, 90, True]] * 3, "rung": 4, "draft_bs": 4},
            {"rows": [[2, 90, True]] * 2, "rung": 2, "draft_bs": 2},
        ],
    )

    shapes = population_shapes([record], run_drafts=1)

    assert list(map(str, shapes)) == [
        "decode B=4 V=0",
        "decode T=3 B=4 V=0",
        "decode B=2",
    ]
    assert shapes[0] == decode(4)._replace(verifies=0)


def test_a_population_with_no_drafter_traces_a_one_token_decode_as_before(tmp_path):
    record = _record(tmp_path, [{"rows": [[1, 90, True]] * 3, "rung": 4}])
    assert list(map(str, population_shapes([record]))) == ["decode B=4"]
