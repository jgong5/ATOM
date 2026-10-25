# SPDX-License-Identifier: MIT
"""A roofline term's seconds by the part of its prices that set them.

At op level each price goes to its largest bound, a comm bound's split into
its hop and its bytes; at step level each module's kernel goes to its largest
bound and its collectives' hops and bytes to their own, a fused module's to
the largest of all. A term no roofline part sets, a price list's or an
overlapped step's, holds none. Priced seconds are convex and nondecreasing in
the kernel floor and the link latency, so the floor's and the latency's
seconds, the price's slope in each, lie between the difference quotients of a
re-pricing at 0.8 and 1.2 times.
"""

from dataclasses import replace

import pytest
from test_backend_step_roofline import LAYER, MACHINE, STACK
from test_op_graph import DECODE, FLAT, GEMM, SECOND, SPLIT, backend, g, layers
from test_op_work import leaf, t
from test_validate_bound import BACKENDS, LINES, record

from atom.compass.backends import BatchView, RequestShape
from atom.compass.backends.cost import CostTerm, StepCost, fold_seconds
from atom.compass.backends.op_graph import OpGraphBackend
from atom.compass.backends.provenance import Method, Provenance
from atom.compass.backends.step_roofline import Module, StepRooflineBackend
from atom.compass.backends.work import attention, bounds, gemm
from atom.compass.validate import bound, project

REDUCE = "aiter.all_reduce_.default"
HOP = "interconnect.intra_node.link_latency_s"
#: A floor-bound GEMM, a compute-bound one and a TP2 all-reduce.
REGION = {
    "seq": [
        g(2, 64, 64),
        g(4096, 4096, 4096),
        leaf(REDUCE, [[2, 4096]], tensor=t([2, 4096])),
    ]
}
ROWS = BatchView((RequestShape(1, 900, True), RequestShape(1, 300, True)))


def by_bound(cost):
    return {term.name: term.by_bound for term in cost.terms}


def test_an_op_level_price_goes_to_its_largest_bound_and_comm_to_hop_and_bytes():
    cost = backend(REGION, MACHINE, tp=2).price(DECODE)
    big = bounds(gemm(4096, 4096, 4096), MACHINE)
    # The all-reduce: 2 (N-1)/N of 2 rows of 4096 bf16 over one link.
    sent = 2 * 4096 * 2

    assert by_bound(cost) == {
        f"op.{GEMM}": {"floor": 2e-6, "compute": big["compute"]},
        f"op.{REDUCE}": {
            "comm.latency": 2e-6,
            "comm.transfer": pytest.approx(sent / (1.0e12 * 0.80)),
        },
    }
    for term in cost.terms:
        assert fold_seconds(term.by_bound.values()) == pytest.approx(term.seconds)


def test_a_step_level_module_s_kernel_takes_its_largest_bound_beside_its_collective():
    cost = StepRooflineBackend(MACHINE, LAYER).price(ROWS)
    proj = bounds(gemm(2, 4096, 2048), MACHINE)
    attn = bounds(attention([1, 1], [900, 300], 16, 128, 4), MACHINE)
    comm = {"comm.latency": 2e-6, "comm.transfer": 2 * 4096 * 2 / (1.0e12 * 0.80)}

    assert by_bound(cost) == {
        "roofline.gemm": {"memory": proj["memory"], **comm},
        # Fused: the all-reduce is the largest bound of the norm's one max.
        "roofline.norm": comm,
        "roofline.attention": {"floor": attn["floor"]},
    }
    for term in cost.terms:
        assert fold_seconds(term.by_bound.values()) == pytest.approx(term.seconds)
    # Modules of one shape: one price times their count.
    (term,) = StepRooflineBackend(MACHINE, STACK).price(ROWS).terms
    one = bounds(gemm(2, 4096, 4096), MACHINE)["memory"]
    assert term.by_bound == {"memory": pytest.approx(8 * one)}


def test_a_listed_price_and_an_overlapped_step_hold_no_part():
    listed = backend(g(2, 64, 64), price_list=FLAT).price(DECODE)
    tbo = OpGraphBackend(SECOND, lambda *shape: layers(), kv="bf16").price(SPLIT)
    split = BatchView(
        (RequestShape(4, 4, False), RequestShape(3, 10, False)),
        produces_output=True,
        tbo_tokens=((3, 5), (4, 5)),
    )
    modules = [
        Module("layers.0.proj", "gemm", (32, 32), layer=0),
        Module("head", "lm_head", (64, 32)),
    ]
    step = StepRooflineBackend(MACHINE, modules).price(split)

    assert by_bound(listed) == {f"op.{GEMM}.price_list": {}}
    assert by_bound(tbo) == {"op.tbo": {}}
    assert by_bound(step)["roofline.tbo"] == {}
    assert by_bound(step)["roofline.lm_head"]


@pytest.mark.parametrize("part,path", [("floor", project.FLOOR), ("comm.latency", HOP)])
@pytest.mark.parametrize("level", ["op", "step"])
def test_a_part_s_seconds_lie_between_its_band_s_difference_quotients(
    level, part, path
):
    def priced(machine):
        if level == "op":
            return backend(REGION, machine, tp=2).price(DECODE)
        return StepRooflineBackend(machine, LAYER).price(ROWS)

    base = priced(MACHINE)
    at = {k: priced(project.scaled(MACHINE, path, k)[0]).seconds for k in (0.8, 1.2)}
    slope = fold_seconds(term.by_bound.get(part, 0.0) for term in base.terms)
    tol = 1e-9 * base.seconds

    assert slope > 0
    assert (base.seconds - at[0.8]) / 0.2 - tol <= slope
    assert slope <= (at[1.2] - base.seconds) / 0.2 + tol


def test_a_term_whose_parts_do_not_sum_to_its_seconds_is_refused():
    (term,) = backend(g(2, 64, 64), MACHINE).price(DECODE).terms

    with pytest.raises(ValueError, match="sum to"):
        replace(term, by_bound={"floor": term.seconds / 2})


class Measured:
    """A backend whose one term is a measured second."""

    def price(self, view):
        said = Provenance(Method.MEASURED, "1 s", "a record")
        return StepCost([CostTerm("measured", 1.0, said)])


def test_the_gate_sums_each_phase_s_seconds_by_part_and_those_no_part_set(tmp_path):
    rec = record(tmp_path / "rec", LINES[:3])
    backends = {**BACKENDS, "measured": Measured()}

    report = bound.gate([rec], backends)

    for level, r in report.items():
        assert list(r["by_bound"]) == list(r["analytical_s"])
        for phase, parts in r["by_bound"].items():
            assert list(parts) == list(bound.BY_BOUND)
            total = r["analytical_s"][phase]["total"]
            assert fold_seconds(parts.values()) == pytest.approx(total)
            unset = total if level == "measured" else 0.0
            assert parts[bound.UNATTRIBUTED] == unset
