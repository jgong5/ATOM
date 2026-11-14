# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` over a decode leaf that launches nothing, as a
view does, timed at 0 ns at every context, whose profiles at one context named
a kernel another call launched, as a TP rank's profile can keep the all-reduce
that lines the ranks up before it: the leaf does not depend on context, round
0's two points are all the plan measures of it, and `fit ops` prices it at
0 ns. The bench and graphs are `test_bench_ops_plan`'s stand-ins.
"""

from test_artifact_store import clean_checkout  # noqa: F401  autouse
from test_bench_ops_plan import (  # noqa: F401  a fixture
    _bench,
    _plan,
    _records,
    bench,
)

from atom.compass.bench import ops
from atom.compass.fit import ops as fit_ops

#: The RCCL kernel a profile kept from the all-reduce before the call.
STRAY = "ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)"
#: The longest context the steps below hold, round 0's far end.
LONGEST = 12000


def _stray_bench(fn, config, todo, llc):
    """`_bench` with the decode GEMM launching nothing, timed at 0 ns, its
    profiles at the longest context naming `STRAY`."""
    got = _bench(fn, config, todo, llc)
    for r in got:
        if r["op"] == "aiter.gemm" and r["shape"].startswith("decode"):
            kernels = [STRAY] if r["context"] == LONGEST else []
            r |= {"kernels": kernels, "cold_ns": 0.0}
    return got


def test_a_leaf_timed_at_0_ns_launched_no_kernel_its_profiles_named(
    bench, tmp_path, monkeypatch  # noqa: F811
):
    monkeypatch.setattr(ops, "_child", _stray_bench)
    population = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[1, LONGEST, True]], 1),
    )

    records = _plan(bench, tmp_path, population)

    assert ops._dependent(records)["aiter.gemm"] is False
    gemm = [r for r in records if r["op"] == "aiter.gemm"]
    assert {r["context"] for r in gemm} == {2000, LONGEST}
    assert any(r["kernels"] == [STRAY] for r in gemm)
    families = fit_ops.fit(records)["families"].values()
    priced = [p for f in families if f["op"] == "aiter.gemm" for p in f["points"]]
    assert priced and all(p[1:] == [0.0, []] for p in priced)
