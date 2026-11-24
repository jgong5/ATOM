# SPDX-License-Identifier: MIT
"""A recorded simulated step builds its `BatchView` once: the step record's
view is the one the runner prices, with the DP group's dummy count put in.

Two DP ranks run as two threads on a gloo group (`test_dp_step_max.dp2`), one
busy and one running its dummy forward, each recording its steps.
"""

from collections import Counter
from types import SimpleNamespace

from test_dp_step_max import Recording, dp2, prefill, runner  # noqa: F401

from atom.compass import parity
from atom.compass.backends.shape import RequestShape
from atom.compass.runner import projection


def _recorded(backend, dp_rank):
    r = runner(backend, dp_size=2, dp_rank=dp_rank)
    quant = SimpleNamespace(compute_hash=lambda: "q")
    fields = {"kv_cache_dtype": "bf16", "quant_config": quant}
    vars(r.config).update(fields, enforce_eager=True, tensor_parallel_size=1)
    return r


def test_a_recorded_step_prices_the_view_its_record_built(
    monkeypatch, tmp_path, dp2  # noqa: F811
):
    monkeypatch.setenv(parity.ENV, str(tmp_path))
    built = Counter()
    build = projection.batch_view

    def counted(batch, mode, runner, *args, **kwargs):
        built[runner.config.parallel_config.data_parallel_rank] += 1
        return build(batch, mode, runner, *args, **kwargs)

    monkeypatch.setattr(projection, "batch_view", counted)
    backends = [Recording(), Recording()]
    ranks = [_recorded(b, rank) for rank, b in enumerate(backends)]

    dp2(lambda: ranks[0].forward(prefill(40)), ranks[1].dummy_execution)

    assert built == {0: 1, 1: 1}
    assert [b.views[0].group_dummies for b in backends] == [1, 1]
    assert backends[0].views[0].requests == (RequestShape(40, 40, False),)
    assert backends[1].views[0].requests == (RequestShape(1, 1, True),)
    lines = [(tmp_path / f"dp{r}.jsonl").read_text().splitlines() for r in (0, 1)]
    assert [len(rank) for rank in lines] == [1, 1]
