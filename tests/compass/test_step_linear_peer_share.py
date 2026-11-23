# SPDX-License-Identifier: MIT
"""A prefill step's `peer_small` and `peer_large`: on a data-parallel rank's
own prefill, output or middle chunk, whose peers run real rows, 1 in the first
where those rows are under `PEER_SMALL` of its own and 1 in the second where
they are not (#3720), and the fit option that keeps the slower steps they mark.

Each rank's forward waits on its peers' at every layer's MoE collective; on
cell B's P a peer's small real prefill cost the group 21-28 ms past the law,
a one-sided tail its MAD rounds dropped. The law prices the columns
(`StepLinearBackend`); the fit, asked, fits them and, asked again, keeps that
tail; unasked, nothing changes.

No device.
"""

import json

import numpy as np
import pytest
from test_backend_step_linear_fit import _engine_file, traced

from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_linear import (
    PEER_SHARE,
    dummy_view,
    hull_axes,
    peer_share,
)
from atom.compass.backends.step_linear_fit import fit, label_records

LAW = {
    "step": 0.02,
    "tokens": 6e-5,
    "query_square": 2e-10,
    "query_cached": 3e-10,
    "moe_rows": 2e-5,
    "peer_small": 0.025,
    "peer_large": 0.007,
}
DP = 2


def _rows(rng, lo, hi):
    rows = []
    for _ in range(int(rng.integers(1, 4))):
        q = int(rng.integers(lo, hi))
        rows.append(RequestShape(q, q + int(rng.integers(0, 30000)), False))
    return tuple(rows)


def _price(view, law=LAW):
    q = np.array([r.query_tokens for r in view.prefill])
    n = np.array([r.context_tokens for r in view.prefill])
    x = (1, q.sum(), (q * q).sum(), (q * (n - q)).sum(), view.moe_rows)
    x += peer_share(view)
    return float(np.dot([law[k] for k in LAW], x))


def _line(r, i, view, enter, seconds, dummy=False):
    rows = (
        [[1, 1, False]]
        if dummy
        else [[x.query_tokens, x.context_tokens, x.decode] for x in view.requests]
    )
    return traced(
        {
            "step": i,
            "dp_rank": r,
            "is_dummy_run": dummy,
            "produces_output": view.produces_output,
            "batch": (
                []
                if dummy
                else [
                    [r * 100_000 + i * 10 + j, q, c] for j, (q, c, _) in enumerate(rows)
                ]
            ),
            "named": [],
            "rows": rows,
            "rung": None,
            "moe_rows": view.moe_rows,
            "t_enter_ns": round(enter * 1e9),
            "t_launch_ns": [round(enter * 1e9)] * 2,
            "t_recv_ns": [round(enter * 1e9)] * 2,
            "d_busy_ns": round(seconds * 1e9),
            "t_exit_ns": round((enter + seconds) * 1e9),
        }
    )


def _record(tmp_path, seed=0, n=420, slow=None):
    """A DP2 record of prefill steps, rank 0 the busiest, beside a peer of
    under a quarter of its tokens, a larger one, or a dummy, a random half of
    them producing output, each labelled with the group's step under `LAW`, the
    max over its busy ranks' prices; with `slow`, every `slow`-th step a small
    peer marks runs 30% over it. Returns the record and each step's (kind,
    label, the law's price)."""
    rng = np.random.default_rng(seed)
    lines, steps, t = ([], []), [], 0.0
    for i in range(n):
        output = bool(rng.integers(0, 2))
        kind = ("small", "large", "dummy")[i % 3]
        big = _rows(rng, 6000, 8192)
        own = sum(r.query_tokens for r in big)
        if kind == "small":
            peer = _rows(rng, 16, 400)
        elif kind == "large":
            peer = _rows(rng, 2100, 4096)
        else:
            peer = ()
        moe = own + sum(r.query_tokens for r in peer)
        views = [BatchView(big, moe_rows=moe, produces_output=output)]
        if peer:
            views.append(BatchView(peer, moe_rows=moe, produces_output=output))
        price = max(map(_price, views))
        seconds = price
        if slow and kind == "small" and i % slow == 0:
            seconds *= 1.3
        steps.append((kind, seconds, price))
        lines[0].append(_line(0, i, views[0], t, seconds))
        if peer:
            lines[1].append(_line(1, i, views[1], t, seconds))
        else:
            dummy = BatchView(
                (RequestShape(1, 1, False),), moe_rows=moe, produces_output=output
            )
            lines[1].append(_line(1, i, dummy, t, seconds, dummy=True))
        t += seconds
    paths = []
    for r in range(DP):
        path = tmp_path / f"dp{r}.jsonl"
        path.write_text("".join(json.dumps(x) + "\n" for x in lines[r]))
        _engine_file(path)
        paths.append(path)
    return paths, steps


def test_peer_share_marks_a_prefill_beside_a_peer_s_real_rows_by_their_share():
    rows = (RequestShape(4000, 9000, False),)
    for moe, output, want in (
        (4999, True, (1, 0)),  # a peer of 999 rows, under a quarter of 4000
        (5000, False, (0, 1)),  # a quarter exactly is not under it; middle too
        (9000, True, (0, 1)),
        (4000, True, (0, 0)),  # a peer's dummy row is no row of the gather
    ):
        view = BatchView(rows, moe_rows=moe, produces_output=output)
        assert peer_share(view) == want
    # One rank, and a dummy forward, carry neither.
    assert peer_share(BatchView(rows, produces_output=True)) == (0, 0)
    assert peer_share(dummy_view(BatchView(rows, moe_rows=9000))) == (0, 0)
    # They are indicators, not hull coordinates.
    features = ("step", "tokens", *PEER_SHARE, "padding")
    assert hull_axes(features) == (1, 4)


def _law():
    hull = {"facets": [[-1, 0, 0, 0, 0]]}
    prefill = {"coefficients": dict(LAW), "hull": hull}
    eager = {"coefficients": {"step": 0.0, "context": 0.0, "graph_padding": 0.0}}
    eager["hull"] = {"equal": {"graph_padding": 0}, "facets": [[-1, 0]]}
    return {"provenance": "a known law", "prefill": prefill, "decode": {"eager": eager}}


def test_the_law_prices_peer_share_on_output_and_middle_chunks():
    backend = StepLinearBackend(StepLinearLaw(_law()), dp=DP)
    rows = (RequestShape(4000, 9000, False),)
    for moe, output, small, large in (
        (4500, True, 0.025, 0.0),
        (4500, False, 0.025, 0.0),
        (9000, False, 0.0, 0.007),
        (4000, True, 0.0, 0.0),
    ):
        view = BatchView(rows, moe_rows=moe, dp_rank=0, produces_output=output)
        terms = {t.name: t.seconds for t in backend.price(view).terms}
        assert terms["prefill.peer_small"] == pytest.approx(small)
        assert terms["prefill.peer_large"] == pytest.approx(large)


def test_the_fit_asked_for_peer_share_recovers_its_coefficients(tmp_path):
    paths, _ = _record(tmp_path)
    samples = label_records(paths, DP)[0]
    law = fit(samples, {"form": "base", "dp": DP, "peer_share": True})[0]

    got = law["prefill"]["coefficients"]
    assert {k: got[k] for k in LAW} == pytest.approx(LAW, rel=1e-4)
    assert list(got)[: len(LAW)] == list(LAW)
    # Unasked the vector has no such column, and the law is the same law
    # whether the options are absent or off; at one rank they refuse.
    plain = fit(samples, {"form": "base", "dp": DP})[0]
    assert not set(PEER_SHARE) & set(plain["prefill"]["coefficients"])
    off = {"form": "base", "dp": DP, "peer_share": False, "keep_peer_tail": False}
    assert json.dumps(fit(samples, off)[0]) == json.dumps(plain)
    with pytest.raises(ValueError, match="these records run one rank"):
        fit(samples, {"form": "base", "dp": 1, "peer_share": True})
    with pytest.raises(ValueError, match="without peer_share"):
        fit(samples, {"form": "base", "dp": DP, "keep_peer_tail": True})


def test_keeping_the_peer_tail_prices_the_small_peer_class_at_its_mean(tmp_path):
    paths, steps = _record(tmp_path, slow=4)
    samples = label_records(paths, DP)[0]
    asked = {"form": "base", "dp": DP, "peer_share": True}
    dropped = fit(samples, asked)[0]["prefill"]
    kept = fit(samples, {**asked, "keep_peer_tail": True})[0]["prefill"]

    # Dropped, the slow small-peer steps leave the column at its median, the
    # planted 25 ms; kept, it carries their time too.
    assert dropped["coefficients"]["peer_small"] == pytest.approx(0.025, rel=0.05)
    assert kept["coefficients"]["peer_small"] > 0.035
    assert kept["fit"]["peer_tail"] > 0 and "peer_tail" not in dropped["fit"]
    small = [(label, price) for kind, label, price in steps if kind == "small"]
    real = sum(label for label, _ in small)
    gap = {}
    for name, vector in (("dropped", dropped), ("kept", kept)):
        c = vector["coefficients"]["peer_small"] - LAW["peer_small"]
        gap[name] = abs(sum(price + c for _, price in small) / real - 1)
    assert gap["kept"] < gap["dropped"] / 2
