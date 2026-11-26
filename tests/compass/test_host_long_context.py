# SPDX-License-Identifier: MIT
"""The `long_context` launch feature under ``COMPASS_FIT_HOST_LONG_CONTEXT=T``.

A host law prices ``(1, tokens, seqs)``. Where a forward's host work steps with
its context, a host fit under the switch adds `launch_features.LONG_CONTEXT`
to each sample's launch features: 1 where its longest prefill row reaches `T`
tokens of context. Its laws may adopt it as a column by their held-out choice
(`step_linear_fit._launch_columns`). The host entry states the bound
(`launch_features.LONG_TOKENS`), and a run's provider then predicts the feature
from the view it prices (`launch_features.stated`). Unset, no sample, provider
or entry changes.

No device.
"""

import dataclasses
from types import SimpleNamespace

import pytest

from atom.compass import run as compass_run
from atom.compass.backends import launch_features as lf
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.fit import host as fit_host
from atom.compass.runner.overrides import HostDeviceRecurrence

T = 131072


def _view(*rows, output=True):
    return BatchView(tuple(RequestShape(*r) for r in rows), produces_output=output)


def test_the_bound_is_a_whole_number_of_tokens(monkeypatch):
    monkeypatch.delenv(lf.LONG_ENV, raising=False)
    assert lf.long_tokens_fit() is None
    monkeypatch.setenv(lf.LONG_ENV, "0")
    assert lf.long_tokens_fit() is None
    monkeypatch.setenv(lf.LONG_ENV, str(T))
    assert lf.long_tokens_fit() == T
    monkeypatch.setenv(lf.LONG_ENV, "128k")
    with pytest.raises(ValueError, match=lf.LONG_ENV):
        lf.long_tokens_fit()


def test_a_stated_bound_adds_the_feature_to_the_provider():
    base, graphs = lf.stated(None)
    assert type(base) is lf.LaunchFeatures and base.features == ()
    wrapped, _ = lf.stated(None, T)
    assert wrapped.features == (lf.LONG_CONTEXT,)
    # The longest prefill row's context, the chunk included, against the bound.
    assert wrapped.predict(_view((1024, T, False)), graphs) == {lf.LONG_CONTEXT: 1.0}
    assert wrapped.predict(_view((1024, T - 1, False)), graphs) == {
        lf.LONG_CONTEXT: 0.0
    }
    assert (
        wrapped.predict(_view((1024, 9000, False), (512, 400000, False)), graphs)[
            lf.LONG_CONTEXT
        ]
        == 1.0
    )
    # A decode's rows are no prefill: it never takes the feature.
    assert wrapped.predict(_view((1, 900000, True)), graphs) == {lf.LONG_CONTEXT: 0.0}
    # A line states no such feature, so the path of a short forward is as before.
    assert (
        lf.path(
            _view((1024, 9000, False)),
            wrapped.predict(_view((1024, 9000, False)), graphs),
        )
        == "replay"
    )


@dataclasses.dataclass(frozen=True)
class _Sample:
    view: BatchView
    launch: dict
    provenance: dict = dataclasses.field(default_factory=lambda: {"config": {}})


def test_samples_take_the_feature_as_a_run_predicts_it():
    got = lf.long_samples(
        [
            _Sample(_view((800, 300000, False)), {}),
            _Sample(_view((800, 3000, False)), {}),
        ],
        T,
    )
    assert [s.launch for s in got] == [{lf.LONG_CONTEXT: 1.0}, {lf.LONG_CONTEXT: 0.0}]


@pytest.mark.parametrize("value", [None, "0", str(T)])
def test_a_host_fit_adds_the_feature_under_the_switch_only(monkeypatch, value):
    seen = []

    def fit(samples):
        seen.append([s.launch for s in samples])
        return {"laws": {}}, {"unfitted": {}}

    monkeypatch.setattr(fit_host, "fit_host", fit)
    monkeypatch.setattr(fit_host.publish, "runs", lambda samples: {})
    monkeypatch.delenv("COMPASS_FIT_LAUNCH_WAIT", raising=False)
    if value is None:
        monkeypatch.delenv(lf.LONG_ENV, raising=False)
    else:
        monkeypatch.setenv(lf.LONG_ENV, value)
    fit_host.host_laws([_Sample(_view((800, 300000, False)), {})])
    assert seen[0] == ([{lf.LONG_CONTEXT: 1.0}] if value == str(T) else [{}])


LAWS = {
    k: {
        "E": [0, 0, 0],
        "W_pre": [0.002, 0, 0],
        "W_launch": [0.18, 0, 0],
        "W_post": [0, 0, 0],
    }
    for k in ("decode", "prefill", "middle")
}


def _batch(long):
    return SimpleNamespace(
        total_seqs_num_prefill=1,
        produces_output=lambda: True,
        total_tokens_num=800,
        total_seqs_num=1,
        compass_launch={lf.LONG_CONTEXT: float(long)},
    )


def test_a_run_prices_an_adopted_column_on_the_predicted_feature():
    columns = {"prefill": {"W_pre": {lf.LONG_CONTEXT: 0.045}}}
    rec = HostDeviceRecurrence(
        LAWS, (), lambda: 0.0, columns=columns, launch=lf.stated(None, T)
    )
    rec.device_end(_batch(True), 0.05, 0.0)
    assert rec.segments["W_pre"] == pytest.approx(0.047)
    rec.device_end(_batch(False), 0.05, 0.0)
    assert rec.segments["W_pre"] == pytest.approx(0.002)
    # Without the bound the provider declares no such feature: the laws refuse.
    with pytest.raises(ValueError, match="long_context"):
        HostDeviceRecurrence(
            LAWS, (), lambda: 0.0, columns=columns, launch=lf.stated(None)
        )


def test_a_run_file_s_host_entry_may_state_the_bound(tmp_path):
    host = {"laws": LAWS, lf.LONG_TOKENS: T}
    compass_run._check_host(tmp_path / "run.json", host)
    for bad in (0, "128k"):
        with pytest.raises(ValueError, match=lf.LONG_TOKENS):
            compass_run._check_host(
                tmp_path / "run.json", {**host, lf.LONG_TOKENS: bad}
            )
