# SPDX-License-Identifier: MIT
"""The step linear law's prefill `padding`: ATOM replays a prefill's first layers
as a token graph at the smallest recorded bucket holding the most tokens any
rank of its data-parallel group scheduled, so a step pays for the bucket's
rows, not its own.

Records are written from a known law by `test_backend_step_linear_fit.py`'s
writer, their engine files stating the buckets the runner recorded. The single
rank record holds both groups the trace runs of #2976 mispriced, one-request
output prefills of 1033 to 1314 tokens and of 124 to 248, beside steps of the
other buckets and past the largest; its law is the one #2976 fitted there.

No device.
"""

import dataclasses
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
from test_backend_step_linear_fit import (
    ENGINE,
    SPEC,
    _engine_file,
    _record,
    platform_arg,
)
from test_fit_steps import LINEAR, clients, fit, law_key, member, record
from test_memory_readings import DOCUMENT
from test_parity_record import _config
from test_step_linear_draft import _mtp_record

from atom.compass.backends import (
    BatchView,
    CostRefused,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
    step_linear_fit,
)
from atom.compass.backends.step_labels import digest
from atom.compass.parity import _record as parity_record
from atom.compass.runner import projection
from atom.utils.region_graph import TokenGraph

#: ATOM's default prefill token-graph buckets, which the trace runs recorded.
BUCKETS = [256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168]
#: The prefill law #2976 fitted with `padding` on the trace runs, in seconds.
PADDED = {
    "step": 4.197e-3,
    "tokens": 1.75e-5,
    "query_square": 1.476e-11,
    "query_cached": 2.616e-12,
    "padding": 1.141e-5,
}
#: One-request output prefills of 1033 to 1314 tokens, and those of 124 to 248.
G1 = [[(n, n, False)] for n in (1033, 1070, 1105, 1140, 1176, 1211, 1246, 1281, 1314)]
G2 = [[(n, n, False)] for n in (124, 157, 189, 226)]
G2 += [[(100, 100, False), (105, 105, False)], [(124, 124, False)] * 2]


def _bucket(tokens):
    return next((b for b in BUCKETS if b >= tokens), None)


def _priced(rows, peak=None):
    """`PADDED`'s price of a rank's `rows`, its group's largest rank at `peak`
    tokens, its own by default."""
    q = np.array([r[0] for r in rows])
    n = np.array([r[1] for r in rows])
    bucket = _bucket(peak or q.sum())
    padding = bucket - q.sum() if bucket else 0
    x = (1, q.sum(), (q * q).sum(), (q * (n - q)).sum(), padding)
    return float(np.dot(list(PADDED.values()), x))


def _trace_steps(seed):
    """Both groups, and prefills of 1 to 8 requests up to 16384 tokens, some on
    a cached history; each with its price."""
    rng = np.random.default_rng(seed)
    out = [rows for rows in G1 + G2]
    for _ in range(120):
        rows = []
        for _ in range(int(rng.integers(1, 9))):
            q = int(rng.integers(16, 2048))
            rows.append(
                (q, q + int(rng.integers(0, 4)) * int(rng.integers(0, 20000)), False)
            )
        out.append(rows)
    return [(rows, None, _priced(rows)) for rows in out]


def _view(rows, **fields):
    return BatchView(tuple(RequestShape(*r) for r in rows), **fields)


def _law_of(tmp_path, records, *args):
    out = tmp_path / "law.json"
    step_linear_fit.main([*map(str, records), "--out", str(out), *SPEC, *args])
    return json.loads(out.read_text())


def _path(directory):
    directory.mkdir()
    return directory / "dp0.jsonl"


def test_a_trace_run_record_holding_both_groups_fits_and_prices_its_buckets(tmp_path):
    steps = _trace_steps(0)
    path = _record(_path(tmp_path / "padded"), steps)
    _engine_file(path, {**ENGINE, "prefill_buckets": BUCKETS})
    law = _law_of(tmp_path, [path])

    prefill = law["prefill"]
    assert prefill["coefficients"] == pytest.approx(PADDED, rel=1e-6)
    assert prefill["buckets"] == BUCKETS
    assert prefill["fit"]["dropped"] == 0
    backend = StepLinearBackend(StepLinearLaw(law))
    for rows in G1 + G2:
        cost = backend.price(_view(rows, produces_output=True))
        assert cost.seconds == pytest.approx(_priced(rows), rel=1e-6)
        [padding] = [t for t in cost.terms if t.name == "prefill.padding"]
        tokens = sum(r[0] for r in rows)
        assert padding.seconds == pytest.approx(
            PADDED["padding"] * (_bucket(tokens) - tokens)
        )
    past = backend.price(_view([(8000, 8000, False)]))
    assert [t.seconds for t in past.terms if t.name == "prefill.padding"] == [0.0]

    # The same record, its engine file stating no buckets: the law prices the
    # 2048 bucket's short prompts low and the 256 bucket's high.
    _engine_file(path)
    plain = _law_of(tmp_path, [path])
    assert "padding" not in plain["prefill"]["coefficients"]
    assert "buckets" not in plain["prefill"]
    backend = StepLinearBackend(StepLinearLaw(plain))
    error = [backend.price(_view(rows)).seconds / _priced(rows) - 1 for rows in G1 + G2]
    assert max(error[: len(G1)]) < -0.05
    assert min(error[len(G1) :]) > 0.0


#: A rank's step under the data-parallel law: `PADDED` and the MoE gather's rows.
DP_PADDED = {**PADDED, "moe_rows": 2e-6}


def _dp_priced(rows, moe, peak):
    return _priced(rows, peak) + DP_PADDED["moe_rows"] * moe


#: The data-parallel group's ranks.
DP = 4


def _dp_groups(seed, n=200):
    """Prefill steps of `DP` ranks, each with its group's step, the max over
    its ranks' prices: rank `i % DP` the slowest, at 1000 to 7000 tokens, the
    others at 30 to 90% of its tokens, padded to its bucket."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ranks, slow = [], int(rng.integers(1000, 7000))
        for r in range(DP):
            q = slow if r == i % DP else int(slow * rng.uniform(0.3, 0.9))
            history = int(rng.integers(0, 2)) * int(rng.integers(0, 9000))
            ranks.append([(q, q + history, False)])
        moe = sum(rows[0][0] for rows in ranks)
        peak = max(rows[0][0] for rows in ranks)
        out.append((ranks, moe, peak, max(_dp_priced(r, moe, peak) for r in ranks)))
    return out


def _dp_records(tmp_path, groups):
    paths = []
    for r in range(DP):
        steps = [(ranks[r], None, seconds) for ranks, _, _, seconds in groups]
        path = _record(tmp_path / f"dp{r}.jsonl", steps)
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        for line, (_, moe, _, _) in zip(lines, groups):
            line |= {"dp_rank": r, "moe_rows": moe, "moe_tokens": moe}
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        _engine_file(path, {"data_parallel_size": DP, "prefill_buckets": BUCKETS})
        paths.append(path)
    return paths


def test_a_data_parallel_step_pads_every_rank_to_its_groups_largest(tmp_path):
    groups = _dp_groups(0)
    paths = _dp_records(tmp_path, groups)
    # No line states its group's largest rank; the labels read it off every rank's.
    samples = step_linear_fit.label_records(paths, DP)[0]
    peaks = [peak for _, _, peak, _ in groups]
    assert [s.view.group_max_tokens for s in samples] == peaks * DP
    law = _law_of(tmp_path, paths)

    expected = {k: DP_PADDED[k] for k in law["prefill"]["coefficients"]}
    assert law["prefill"]["coefficients"] == pytest.approx(expected, rel=1e-6)
    backend = StepLinearBackend(StepLinearLaw(law), DP)
    for ranks, moe, peak, seconds in groups:
        views = [
            _view(rows, moe_rows=moe, moe_tokens=moe, group_max_tokens=peak)
            for rows in ranks
        ]
        assert max(backend.price(v).seconds for v in views) == pytest.approx(
            seconds, rel=1e-6
        )

    # Rank 1's 100 tokens replay rank 0's 3000-token bucket.
    small = _view([(100, 100, False)], moe_rows=3100, group_max_tokens=3000)
    [padding] = [t for t in backend.price(small).terms if t.name == "prefill.padding"]
    assert padding.seconds == pytest.approx(DP_PADDED["padding"] * 2972)
    split = _view(
        [(100, 100, False)],
        moe_rows=3100,
        group_max_tokens=3000,
        tbo_tokens=((50, 50),),
    )
    terms = backend.price(split).terms
    assert [t.seconds for t in terms if t.name == "prefill.padding"] == [0.0]
    unknown = _view([(100, 100, False)], moe_rows=3100)
    with pytest.raises(CostRefused, match="no group_max_tokens on this data-parallel"):
        backend.price(unknown)


def test_an_mtp_record_s_draft_law_adds_no_padding(tmp_path):
    path = _mtp_record(_path(tmp_path / "mtp"), _trace_steps(0))
    _engine_file(path, {**ENGINE, "prefill_buckets": BUCKETS})
    law = _law_of(tmp_path, [path])

    assert law["prefill"]["buckets"] == BUCKETS
    assert "padding" in law["prefill"]["coefficients"]
    draft = law["draft"]["prefill"]
    assert "padding" not in draft["coefficients"]
    assert "buckets" not in draft


def test_a_data_parallel_step_stating_no_largest_rank_refuses_a_padded_fit(
    tmp_path,
):
    samples = step_linear_fit.label_records(
        _dp_records(tmp_path, _dp_groups(1, n=20)), DP
    )[0]
    view = dataclasses.replace(samples[3].view, group_max_tokens=None)
    samples[3] = dataclasses.replace(samples[3], view=view)

    with pytest.raises(ValueError, match="step 3 has no group_max_tokens"):
        step_linear_fit.fit(samples, {"dp": DP, "buckets": BUCKETS})


def test_a_law_carrying_padding_states_its_buckets():
    vector = {
        "coefficients": PADDED,
        "hull": {"equal": dict.fromkeys(list(PADDED)[1:], 0), "facets": []},
        "buckets": BUCKETS,
    }
    law = {"provenance": "planted", "prefill": vector, "decode": {}}
    StepLinearLaw(law)
    del vector["buckets"]
    with pytest.raises(ValueError, match="carries padding and states no buckets"):
        StepLinearLaw(law)


def test_records_stating_other_buckets_refuse_and_an_engine_file_states_them(
    tmp_path,
):
    a = _record(_path(tmp_path / "a"), _trace_steps(1))
    b = _record(_path(tmp_path / "b"), _trace_steps(2))
    _engine_file(a, {**ENGINE, "prefill_buckets": BUCKETS})
    with pytest.raises(ValueError, match="state other prefill buckets: .* none in"):
        _law_of(tmp_path, [a, b])

    stated = tmp_path / "engine.json"
    stated.write_text(json.dumps({**ENGINE, "prefill_buckets": BUCKETS}))
    law = _law_of(tmp_path, [a, b], "--engine", str(stated))
    assert law["prefill"]["buckets"] == BUCKETS
    assert law["prefill"]["coefficients"] == pytest.approx(PADDED, rel=1e-6)

    _engine_file(b, {**ENGINE, "prefill_buckets": BUCKETS[:4]})
    with pytest.raises(ValueError, match="state other prefill buckets"):
        _law_of(tmp_path, [a, b], "--engine", str(stated))


def test_the_published_law_pads_to_the_records_buckets_as_the_command_does(tmp_path):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    for d in (a, b):
        engine = d / "steps/dp0.engine.json"
        held = json.loads(engine.read_text())
        engine.write_text(json.dumps({**held, "prefill_buckets": BUCKETS}))
    fit(tmp_path, "fit steps", *LINEAR, a, b, *clients(a, b))

    law = member(tmp_path / "store", law_key(), "params")
    assert law["prefill"]["buckets"] == BUCKETS
    paths = sorted((d / "steps/dp0.jsonl" for d in (a, b)), key=digest)
    out = tmp_path / "law.json"
    step_linear_fit.main(
        [*map(str, paths), "--form", "history", *platform_arg(DOCUMENT)]
        + ["--out", str(out)]
    )
    written = json.loads(out.read_text())
    assert law == {k: v for k, v in written.items() if k not in ("host", "host_fit")}


def test_fit_steps_refuses_records_of_which_only_some_state_buckets(tmp_path):
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    engine = a / "steps/dp0.engine.json"
    held = json.loads(engine.read_text())
    engine.write_text(json.dumps({**held, "prefill_buckets": BUCKETS}))

    named = f"prefill_buckets: {json.dumps(BUCKETS)} in {a}; null in {b}"
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit steps", *LINEAR, a, b, *clients(a, b))
    assert not (tmp_path / "store").exists()


def test_a_record_states_the_prefill_buckets_its_runner_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOM_COMPASS_PARITY_RECORD", str(tmp_path / "none"))
    path = parity_record(SimpleNamespace(rank=0, config=_config())).path
    engine = json.loads(path.with_suffix(".engine.json").read_text())
    assert engine["prefill_buckets"] == []

    monkeypatch.setattr(TokenGraph, "captured_sizes", lambda phase=None: (256, 512))
    monkeypatch.setenv("ATOM_COMPASS_PARITY_RECORD", str(tmp_path / "some"))
    path = parity_record(SimpleNamespace(rank=0, config=_config())).path
    engine = json.loads(path.with_suffix(".engine.json").read_text())
    assert engine["prefill_buckets"] == [256, 512]


@pytest.mark.parametrize("dp", [1, 2])
def test_the_projection_states_the_groups_largest_rank(dp):
    batch = SimpleNamespace(
        total_seqs_num_prefill=1,
        num_scheduled_tokens=np.array([100]),
        context_lens=np.array([100]),
        is_dummy_run=False,
        produces_output=lambda: True,
    )
    sync = SimpleNamespace(
        scheduled_tokens_per_dp_rank=np.array([100, 600][:dp]),
        ub_tokens_across_dp=None,
        max_bs_across_dp=1,
    )
    mode = SimpleNamespace(
        use_cudagraph=False,
        running_tokens_are_unified=dp == 1,
        running_bs=1,
        running_tokens=100,
        sync=sync,
    )
    parallel = SimpleNamespace(data_parallel_size=dp, data_parallel_rank=0)
    runner = SimpleNamespace(config=SimpleNamespace(parallel_config=parallel))

    view = projection.batch_view(batch, mode, runner)
    assert view.group_max_tokens == (600 if dp == 2 else None)


def test_a_largest_rank_below_the_ranks_own_tokens_is_refused():
    with pytest.raises(ValueError, match="group max tokens 99 is fewer than the 100"):
        _view([(100, 100, False)], moe_rows=200, group_max_tokens=99)
