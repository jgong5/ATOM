# SPDX-License-Identifier: MIT
"""A DP rank's dummy forward at a step its group prefills: labelled apart from
the samples (`step_labels.Sample.dummy_host`), fitted as the host laws'
`dummy` kind with every launch feature a column (`step_linear_fit._fit_dummy`),
and priced by that kind, `decode`'s where it states no segment
(`HostDeviceRecurrence`)."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_backend_step_labels import _pair

from atom.compass import run as compass_run
from atom.compass.backends.launch_features import (
    GRAPHS,
    REGISTRY,
    DeepseekV41,
    LaunchFeatures,
    stated,
)
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_labels import DUMMY_SEGMENTS, Sample, label_run
from atom.compass.backends.step_linear import dummy_view
from atom.compass.backends.step_linear_fit import DUMMY, fit_host
from atom.compass.runner.overrides import DUMMY_KIND, HostDeviceRecurrence

ARCH = "PlantedDummyForCausalLM"
#: The planted group peak past which a step's dummies launch eagerly.
BUCKET = 4000
SETS = {"architecture": ARCH, "token": {"prefill": [BUCKET], "decode": []}}
#: A dummy's planted W_launch: 2 ms replayed, 25 ms more eagerly.
REPLAYED_S, EAGER_S = 2e-3, 25e-3


class Planted(LaunchFeatures):
    """Each step eager past `BUCKET` of its group's most tokens."""

    features = ("eager",)

    def predict(self, view, graphs):
        peak = max(
            sum(r.query_tokens for r in view.requests), view.group_max_tokens or 0
        )
        return {"eager": float(peak > max(graphs["token"]["prefill"]))}


@pytest.fixture(autouse=True)
def planted(monkeypatch):
    monkeypatch.setitem(REGISTRY, ARCH, Planted())


def _dummy_lines(records):
    """Make rank 1 of `_pair`'s records a dummy at every step rank 0
    prefills, as ATOM's lines state one: no request, its one-token prefill row,
    producing output; both ranks' lines with their group's MoE rows."""
    ranks = [[json.loads(x) for x in r.open()] for r in records]
    for a, b in zip(*ranks):
        a["moe_rows"] = b["moe_rows"] = 1
        if not a["rows"][0][2]:
            b.update(is_dummy_run=True, batch=[], named=[], rows=[[1, 1, False]])
            b["produces_output"] = True
    for r, lines in zip(records, ranks):
        r.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return ranks


def test_a_prefill_step_s_dummy_gives_no_sample_and_its_rank_s_dummy_host(tmp_path):
    records = _pair(tmp_path, ((3,), (2,)))[0]
    ranks = _dummy_lines(records)
    (zero, _), (one, _) = label_run(records)

    dummies = [s for s in ranks[1] if s["is_dummy_run"]]
    assert dummies and all(s.dummy_host == () for s in zero)
    assert {s.step for s in one}.isdisjoint(d["step"] for d in dummies)
    held = one[0].dummy_host
    assert all(s.dummy_host is held for s in one)
    assert [h[0] for h in held] == [d["step"] for d in dummies]
    peaks = {s["step"]: sum(q for q, _, _ in s["rows"]) for s in ranks[0]}
    for (step, view, host, launch), line in zip(held, dummies):
        assert view.dp_dummy and view.prefill and view.produces_output
        assert view.group_max_tokens == peaks[step] and not view.group_tail
        l0, l1 = line["t_launch_ns"]
        r0, r1 = line.get("t_recv_ns", (l1, l1))
        g0, g1 = line["t_gate_ns"]
        assert host["W_launch"] == pytest.approx((r0 - l0) * 1e-9, rel=1e-12)
        assert host["W_pre"] == pytest.approx(
            (g0 - line["t_enter_ns"] + l0 - g1) * 1e-9, rel=1e-12
        )
        assert host["W_post"] == pytest.approx((line["t_exit_ns"] - r1) * 1e-9)
        assert set(host) <= set(DUMMY_SEGMENTS) and launch == {}


def _samples(rng, dummies, observed=None):
    """Planted decodes and output prefills of one run, the first carrying
    `dummies` dummy forwards whose `W_launch` is `REPLAYED_S` plus `EAGER_S`
    eagerly; `observed` overrides one dummy's launch features."""
    key = {"engine": {GRAPHS: SETS}}
    rows = []
    for i in range(dummies):
        peak = int(rng.integers(256, 2 * BUCKET))
        view = dummy_view(
            BatchView((RequestShape(1, 1, False),), dp_rank=1, group_max_tokens=peak)
        )
        launch = Planted().predict(view, SETS)
        host = {
            "W_pre": 1e-3 * rng.uniform(0.99, 1.01),
            "W_inputs": 5e-4 * rng.uniform(0.99, 1.01),
            "W_launch": (REPLAYED_S + EAGER_S * launch["eager"])
            * rng.uniform(0.99, 1.01),
            "W_post": 4e-4 * rng.uniform(0.99, 1.01),
        }
        rows.append((2 * i, view, host, launch))
    if observed is not None:
        rows[3] = (*rows[3][:3], observed)
    out = []
    for i in range(60):
        tokens = int(rng.integers(256, BUCKET)) if i % 2 else 1
        view = BatchView(
            (RequestShape(tokens, tokens, i % 2 == 0),), produces_output=True
        )
        host = {g: 1e-4 * rng.uniform(0.99, 1.01) for g in ("E", "W_pre", "W_post")}
        host["W_launch"] = (2e-3 + 1e-6 * tokens) * rng.uniform(0.99, 1.01)
        provenance = {"source": "record", "run": "a", "requests": [i], "config": key}
        sample = Sample(i, 0, view, 0.01, None, host, False, provenance)
        out.append(
            dataclasses.replace(sample, launch={"eager": 0.0}, dummy_host=tuple(rows))
        )
    return out


def test_the_dummy_kind_is_fitted_with_every_launch_feature_a_column():
    host, report = fit_host(_samples(np.random.default_rng(3746), 80))

    law, columns = host["laws"][DUMMY], host["columns"][DUMMY]
    assert set(law) == set(DUMMY_SEGMENTS) == set(columns)
    assert law["W_launch"] == pytest.approx([REPLAYED_S, 0, 0], abs=1e-4)
    assert columns["W_launch"] == pytest.approx({"eager": EAGER_S}, rel=0.02)
    assert law["W_pre"] == pytest.approx([1e-3, 0, 0], rel=0.02)
    assert report[f"{DUMMY}.W_launch"]["forwards"] == 80


def test_too_few_dummies_are_named_unfitted_and_none_fits_no_kind():
    host, report = fit_host(_samples(np.random.default_rng(1), 4))
    assert DUMMY not in host["laws"]
    assert report["unfitted"][f"{DUMMY}.W_launch"] == "4 forwards for 4 coefficients"
    host, report = fit_host(_samples(np.random.default_rng(1), 0))
    assert DUMMY not in host["laws"] and not any(DUMMY in k for k in report)


def test_a_dummy_launched_otherwise_than_predicted_refuses():
    samples = _samples(np.random.default_rng(5), 20, observed={"eager": 7.0})
    with pytest.raises(ValueError, match="a dummy forward at step 6 launched"):
        fit_host(samples)


DECODE = {"E": [1e-3, 0, 0], "W_pre": [6e-4, 0, 0], "W_launch": [2.3e-3, 0, 0]}
DECODE |= {"W_post": [4e-4, 0, 0]}


def _dummy_batch(dummy, eager, output=True):
    return SimpleNamespace(
        total_seqs_num_prefill=0,
        total_tokens_num=1,
        total_seqs_num=1,
        produces_output=lambda: output,
        compass_dummy=dummy,
        compass_launch={"eager": eager},
    )


@pytest.mark.parametrize("eager", [0.0, 1.0])
def test_a_dummy_is_priced_by_the_dummy_kind_and_decode_s_other_segments(eager):
    laws = {"decode": DECODE, DUMMY_KIND: {"W_launch": [REPLAYED_S, 0, 0]}}
    columns = {DUMMY_KIND: {"W_launch": {"eager": EAGER_S}}}
    launch = stated({"engine": {GRAPHS: SETS}})
    rec = HostDeviceRecurrence(laws, (), lambda: 0.0, columns=columns, launch=launch)

    rec.charge(_dummy_batch(True, eager), 0.01, 0.0)
    assert rec.segments["W_launch"] == pytest.approx(REPLAYED_S + EAGER_S * eager)
    for segment in ("E", "W_pre", "W_post"):
        assert rec.segments[segment] == pytest.approx(DECODE[segment][0])
    # A decode, and a dummy under laws that state no dummy kind, are decode's.
    rec.charge(_dummy_batch(False, eager), 0.01, 0.0)
    assert rec.segments["W_launch"] == pytest.approx(DECODE["W_launch"][0])
    plain = HostDeviceRecurrence({"decode": DECODE}, (), lambda: 0.0)
    plain.charge(_dummy_batch(True, eager), 0.01, 0.0)
    assert plain.segments["W_launch"] == pytest.approx(DECODE["W_launch"][0])
    # A dummy that produces no output is a middle chunk, whatever its laws.
    laws["middle"] = {g: DECODE[g] for g in ("E", "W_pre", "W_launch")}
    rec.charge(_dummy_batch(True, eager, output=False), 0.01, 0.0)
    assert rec.segments["W_launch"] == pytest.approx(DECODE["W_launch"][0])


def test_v41_runs_a_dummy_s_tail_where_its_group_s_runs():
    graphs = {"token": {"prefill": [256, 7168]}, "decoder": []}
    v41 = DeepseekV41()
    for peak, tail in ((9000, False), (9000, True), (300, True), (300, False)):
        view = dummy_view(
            BatchView(
                (RequestShape(1, 1, False),),
                dp_rank=1,
                group_max_tokens=peak,
                group_tail=tail,
            )
        )
        assert v41.predict(view, graphs) == {
            "eager_body": float(peak > 7168),
            "eager_tail": float(tail),
        }
    # A rank's own output prefill runs its tail whatever its group's.
    own = BatchView((RequestShape(300, 300, False),), produces_output=True)
    assert v41.predict(own, graphs) == {"eager_body": 0.0, "eager_tail": 1.0}


def test_a_run_file_s_host_laws_may_state_the_dummy_kind(tmp_path):
    segment = {"W_launch": [REPLAYED_S, 0, 0]}
    compass_run._check_host(tmp_path, {"laws": {DUMMY_KIND: segment}})
    compass_run._check_host(
        tmp_path, {"columns": {DUMMY_KIND: {"W_launch": {"eager": EAGER_S}}}}
    )
    with pytest.raises(ValueError, match="host is"):
        compass_run._check_host(tmp_path, {"laws": {"dummies": segment}})


def test_a_dummy_reads_no_token_so_leaves_at_its_launch_not_its_group_s_end():
    """A producer that does not defer its output waits for its own target; a
    dummy beside a peer's middle chunk, whose device runs until the group's
    end (the DP charge's `all_reduce`), reads no token and leaves at its
    launch and `W_post`, so the group's next step enters beside that device
    work, as a real dummy leaves (#3720 b3_B3: 2.6 ms beside a 162 ms chunk)."""
    laws = {"decode": DECODE}
    host = sum(DECODE[s][0] for s in ("E", "W_pre", "W_launch", "W_post"))
    late = 0.172  # the group's device end: the peer's middle chunk's
    dummy = HostDeviceRecurrence(laws, (), lambda: 0.0, defer=False)
    dummy.device_end(_dummy_batch(True, 0.0), 0.0, 0.0)
    assert dummy.release(late, late) == pytest.approx(host)
    assert dummy.ends[-1] == late
    own = HostDeviceRecurrence(laws, (), lambda: 0.0, defer=False)
    own.device_end(_dummy_batch(False, 0.0), 0.0, 0.0)
    assert own.release(late, late) == pytest.approx(late + DECODE["W_post"][0])
