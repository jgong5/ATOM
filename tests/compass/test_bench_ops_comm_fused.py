# SPDX-License-Identifier: MIT
"""`compass measure ops` on aiter's comm-fused MoE: the leaf is called on the
runtime ATOM's loader builds, built before the runner's KV pool is sized, with
the leaf's arguments and the shared expert's output its stage 2 asks for; its
point keys the layout, and `fit ops` prices the leaf from its points."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_bench_ops import _Runner
from test_bench_ops_ranks import _atom_runner, _AtomRunner, _dpa_config

from atom.compass.bench import ops, ranks
from atom.compass.capture.peers import COMM_FUSED
from atom.compass.fit import ops as fit
from atom.utils import forward_context as fctx

#: The leaf's device time on the fake device, in ns, and the rows its stage 2
#: gives back: this rank's of a reduce-scatter.
NS, ROWS = 1000.0, 2


def _t(shape, dtype="torch.bfloat16", role="activation"):
    stride = list(torch.empty(shape, device="meta").stride())
    return {"shape": shape, "dtype": dtype, "stride": stride, "role": role}


def _leaf(rows, hidden=8, experts=4, out=ROWS):
    """The leaf as the capture states it (`peers.comm_fused`): every rank's
    rows in, this rank's out, reduced across eight ranks."""
    return {
        "op": COMM_FUSED,
        "args": {
            "hidden_states": _t([rows, hidden]),
            "w1": _t([experts, 2 * hidden, hidden // 2], "torch.uint8", "weight"),
            "topk_weight": _t([rows, 2], "torch.float32"),
            "topk_ids": _t([rows, 2], "torch.int32"),
            "activation": 0,
            "quant_type": 3,
            "gate_mode": "interleave",
            "routes_may_be_missing": True,
            "reuse_is_synchronized": True,
            "shared_partial": None,
            "tp": 8,
            "comm": "rs",
            "reduce_scatter_sizes": None,
        },
        "out": [[out, hidden]],
        "occurrences": 2,
    }


class _Runtime:
    """aiter's comm-fused runtime as ATOM builds it: runners for `tp` ranks
    reducing by `comm`; a run asks the shared expert's output of its rows and
    takes NS on the device."""

    def __init__(self, tp=8, comm="rs"):
        self.runners = SimpleNamespace(shape=SimpleNamespace(tp=tp, comm=comm))
        self.time, self.shared, self.args = 0.0, set(), None

    def run(self, *, shared_partial, before_stage2_for_rows=None, **moe_args):
        assert shared_partial is None
        self.shared.add(before_stage2_for_rows(ROWS))
        self.args, self.time = moe_args, self.time + NS
        return moe_args["hidden_states"][:ROWS]


def _time(runner):
    def replay(fn, calls, flush=None):
        fn(**calls[0][0])
        times = []
        for _ in range(ops.WARMUPS + ops.REPEATS):
            start = runtime.time
            for kwargs, _ in calls:
                fn(**kwargs)
            times.append((runtime.time - start) / len(calls))
        return times[ops.WARMUPS :]

    runtime, stub = runner.comm_fused, ops.Stub(runner, 10_000)
    stub.placed = [(fctx.ForwardContext(), {})]
    with (
        patch.object(ops, "_replay", replay),
        patch.object(ops, "_identity", lambda fn, kw, timed=False: (["k"], [0], NS)),
    ):
        return stub._time(_leaf(16), "sig", [{}], 4)


def test_the_leaf_is_timed_on_atom_s_runtime_with_the_shared_output_it_asks_for():
    runner = _Runner()
    runner.comm_fused = _Runtime()

    record = _time(runner)

    assert record["cold_ns"] == NS and record["copies"] == 4
    args = runner.comm_fused.args
    assert not {"tp", "comm", "shared_partial"} & args.keys()
    assert (args["activation"], args["gate_mode"]) == (0, "interleave")
    assert args["reuse_is_synchronized"] and args["hidden_states"].shape == (16, 8)
    # one buffer of zeros at the rows asked, read by every call
    [shared] = runner.comm_fused.shared
    assert (shared.shape, shared.dtype) == ((ROWS, 8), torch.bfloat16)
    assert not shared.any()


@pytest.mark.parametrize(
    "runtime, alone, why",
    [
        (None, 0, "ATOM builds no comm-fused MoE"),
        (_Runtime(comm="ar"), 0, r"\(8, 'ar'\), and the leaf states \(8, 'rs'\)"),
        (_Runtime(), 8, "a collective over the layout's 8 ranks"),
    ],
    ids=["none", "other reduction", "one rank alone"],
)
def test_a_leaf_the_runner_cannot_run_refuses_naming_why(runtime, alone, why):
    runner = _Runner()
    runner.comm_fused, runner.config.dp_logical_size = runtime, alone

    with pytest.raises(ValueError, match=why):
        _time(runner)


def test_the_runner_builds_the_runtime_before_atom_sizes_its_pool(monkeypatch):
    # ATOM's loader builds the runners, their peer memory among them, before
    # it sizes the pool, which then leaves that memory to them
    sized, built = [], []
    backend = SimpleNamespace(runtime=None)

    def initialize(layer):
        built.append((layer, len(sized)))
        backend.runtime = _Runtime()

    backend.initialize = initialize
    layer = torch.nn.Module()
    layer._comm_fused_moe = backend

    class Runner(_AtomRunner):
        def __init__(self, rank, config):
            super().__init__(rank, config)
            self.model.append(layer)

        def get_num_blocks(self):
            sized.append(True)
            return super().get_num_blocks()

    _atom_runner(monkeypatch)
    engine = SimpleNamespace(ModelRunner=Runner, RapidServeModelRunner=None)
    monkeypatch.setitem(sys.modules, "atom.model_engine.model_runner", engine)
    config = _dpa_config(2)
    ranks.layout(config)
    config.parallel_config._managed_distributed_store = True

    runner = ops._runner(config, 0)

    assert built == [(layer, 0)] and runner.comm_fused is backend.runtime


class _Backend:
    """ATOM's comm-fused backend (`CommFusedMoeBackend`) of a layer the layout
    gives it over eight ranks: the runtime its loader builds refuses a group of
    one."""

    runtime = None

    def initialize(self, layer):
        raise ValueError(
            "Communication-fused MoE TP all-reduce layout mismatch: group=1, "
            "moe_tp=8, dp=1, use_ep=False"
        )


@pytest.mark.parametrize("drafter", [False, True], ids=["target", "drafter"])
def test_one_dp_rank_alone_builds_no_runtime_for_its_target_or_drafter(
    drafter, monkeypatch
):
    # DP-attention MTP1 decode on one card: the target's parameters load
    # nothing, and a drafter's load builds each MoE layer's runtime
    comm = SimpleNamespace(CommFusedMoeBackend=_Backend)
    monkeypatch.setitem(sys.modules, "atom.model_ops.fused_moe.comm_fused_moe", comm)
    draft = torch.nn.Module()
    draft._comm_fused_moe = _Backend()

    class Runner(_AtomRunner):
        def __init__(self, rank, config):
            super().__init__(rank, config)
            self.model.append(torch.nn.Module())
            self.model[-1]._comm_fused_moe = _Backend()
            if config.speculative_config:  # as ATOM's loader, after the weights
                self.drafter = SimpleNamespace(model=torch.nn.Sequential(draft))
                draft._comm_fused_moe.initialize(draft)

    _atom_runner(monkeypatch)
    engine = SimpleNamespace(ModelRunner=Runner, RapidServeModelRunner=None)
    monkeypatch.setitem(sys.modules, "atom.model_engine.model_runner", engine)
    config = _dpa_config(8)
    config.speculative_config = SimpleNamespace() if drafter else None
    ranks.layout(config)

    runner = ops._runner(config, 0)

    assert runner.config.dp_logical_size == 8 and runner.comm_fused is None
    # and ATOM's own build is back once the runner is built
    with pytest.raises(ValueError, match="group=1"):
        draft._comm_fused_moe.initialize(draft)


def test_a_rank_of_the_layout_s_eight_loads_its_drafter_as_atom_does(monkeypatch):
    # each rank in a process of its own: the drafter's load builds its runtime
    built = []

    class Backend(_Backend):
        def initialize(self, layer):
            built.append(layer)

    comm = SimpleNamespace(CommFusedMoeBackend=Backend)
    monkeypatch.setitem(sys.modules, "atom.model_ops.fused_moe.comm_fused_moe", comm)
    draft = torch.nn.Module()
    draft._comm_fused_moe = Backend()

    class Runner(_AtomRunner):
        def __init__(self, rank, config):
            super().__init__(rank, config)
            self.drafter = SimpleNamespace(model=torch.nn.Sequential(draft))
            draft._comm_fused_moe.initialize(draft)  # as ATOM's loader

    _atom_runner(monkeypatch)
    engine = SimpleNamespace(ModelRunner=Runner, RapidServeModelRunner=None)
    monkeypatch.setitem(sys.modules, "atom.model_engine.model_runner", engine)
    config = _dpa_config(8)
    config.speculative_config = SimpleNamespace()
    ranks.layout(config)
    config.parallel_config._managed_distributed_store = True

    runner = ops._runner(config, 0)

    assert runner.config.dp_logical_size == 0 and built == [draft]


#: The leaf's cold ns at three DP-attention decode widths, eight ranks' rows in.
COLD_NS = {8: 61000.0, 16: 64000.0, 32: 70000.0}


def test_its_point_keys_the_layout_and_fit_ops_prices_it_at_each_width():
    leaves = {b: _leaf(8 * b, 7168, 384, b) for b in COLD_NS}
    points = [
        {"shape": f"decode B={b}", "context": 8192, "op": COMM_FUSED}
        | {"leaf": ops.signature(leaves[b]), "cold_ns": ns, "kernels": ["k"]}
        for b, ns in COLD_NS.items()
    ]
    params = fit.fit(points)
    fit.held_out(params)

    got = [
        fit.PriceList(params).price(
            (leaves[b], ("decode", None, b, None, 8192, 8192), 0)
        )
        for b in COLD_NS
    ]

    assert all(ops.on_group(p["leaf"]) for p in points)
    assert [g.seconds for g in got] == pytest.approx(
        [ns * 1e-9 for ns in COLD_NS.values()]
    )
