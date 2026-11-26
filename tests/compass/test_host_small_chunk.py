# SPDX-License-Identifier: MIT
"""The `small_chunk` launch feature under ``COMPASS_FIT_HOST_SMALL_CHUNK=S``.

A host law prices ``(1, tokens, seqs)``, linear in tokens. Where a prefill's
host launch is longer for a small chunk than the line through the larger ones
gives, a host fit under the switch adds `launch_features.SMALL_CHUNK` to each
sample's launch features: 1 where the forward's prefill rows compute fewer than
`S` new tokens. Its laws may adopt it as a column by their held-out choice
(`step_linear_fit._launch_columns`). The host entry states the bound
(`launch_features.SMALL_TOKENS`), and a run's provider then predicts the feature
from the view it prices (`launch_features.stated`), beside `LONG_CONTEXT` when
the entry states that bound too. Unset, no sample, provider or entry changes.

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

S, T = 2048, 131072


def _view(*rows, output=True):
    return BatchView(tuple(RequestShape(*r) for r in rows), produces_output=output)


def test_the_bound_is_a_whole_number_of_tokens(monkeypatch):
    monkeypatch.delenv(lf.SMALL_ENV, raising=False)
    assert lf.small_tokens_fit() is None
    monkeypatch.setenv(lf.SMALL_ENV, "0")
    assert lf.small_tokens_fit() is None
    monkeypatch.setenv(lf.SMALL_ENV, str(S))
    assert lf.small_tokens_fit() == S
    monkeypatch.setenv(lf.SMALL_ENV, "2k")
    with pytest.raises(ValueError, match=lf.SMALL_ENV):
        lf.small_tokens_fit()


def test_a_stated_bound_adds_the_feature_to_the_provider():
    base, graphs = lf.stated(None)
    assert type(base) is lf.LaunchFeatures and base.features == ()
    wrapped, _ = lf.stated(None, small_tokens=S)
    assert wrapped.features == (lf.SMALL_CHUNK,)
    # The forward's new prefill tokens, summed over its rows, against the bound.
    assert wrapped.predict(_view((1024, 90000, False)), graphs) == {lf.SMALL_CHUNK: 1.0}
    assert wrapped.predict(_view((S, 90000, False)), graphs) == {lf.SMALL_CHUNK: 0.0}
    two = _view((1024, 9000, False), (1024, 400000, False))
    assert wrapped.predict(two, graphs) == {lf.SMALL_CHUNK: 0.0}
    # A decode, holding no prefill row, never takes the feature.
    assert wrapped.predict(_view((1, 900, True)), graphs) == {lf.SMALL_CHUNK: 0.0}
    # Beside the long-context bound, both features, each its own.
    both, _ = lf.stated(None, T, S)
    assert both.features == (lf.LONG_CONTEXT, lf.SMALL_CHUNK)
    assert both.predict(_view((800, 300000, False)), graphs) == {
        lf.LONG_CONTEXT: 1.0,
        lf.SMALL_CHUNK: 1.0,
    }
    # A short forward's path reads as before where the feature is 0.
    view = _view((S, 9000, False))
    assert lf.path(view, wrapped.predict(view, graphs)) == "replay"


@dataclasses.dataclass(frozen=True)
class _Sample:
    view: BatchView
    launch: dict
    provenance: dict = dataclasses.field(default_factory=lambda: {"config": {}})


def test_samples_take_the_feature_as_a_run_predicts_it():
    got = lf.small_samples(
        [
            _Sample(_view((800, 300000, False)), {}),
            _Sample(_view((8000, 30000, False)), {}),
        ],
        S,
    )
    assert [s.launch for s in got] == [{lf.SMALL_CHUNK: 1.0}, {lf.SMALL_CHUNK: 0.0}]


@pytest.mark.parametrize("value", [None, "0", str(S)])
def test_a_host_fit_adds_the_feature_under_the_switch_only(monkeypatch, value):
    seen = []

    def fit(samples):
        seen.append([s.launch for s in samples])
        return {"laws": {}}, {"unfitted": {}}

    monkeypatch.setattr(fit_host, "fit_host", fit)
    monkeypatch.setattr(fit_host.publish, "runs", lambda samples: {})
    for env in (lf.LONG_ENV, "COMPASS_FIT_LAUNCH_WAIT", "COMPASS_FIT_PACED_TAIL"):
        monkeypatch.delenv(env, raising=False)
    if value is None:
        monkeypatch.delenv(lf.SMALL_ENV, raising=False)
    else:
        monkeypatch.setenv(lf.SMALL_ENV, value)
    fit_host.host_laws([_Sample(_view((800, 3000, False)), {})])
    assert seen[0] == ([{lf.SMALL_CHUNK: 1.0}] if value == str(S) else [{}])


LAWS = {
    k: {
        "E": [0, 0, 0],
        "W_pre": [0.002, 0, 0],
        "W_launch": [0.17, 0, 0],
        "W_post": [0, 0, 0],
    }
    for k in ("decode", "prefill", "middle")
}


def _batch(small):
    return SimpleNamespace(
        total_seqs_num_prefill=1,
        produces_output=lambda: True,
        total_tokens_num=800,
        total_seqs_num=1,
        compass_launch={lf.SMALL_CHUNK: float(small)},
    )


def test_a_run_prices_an_adopted_column_on_the_predicted_feature():
    columns = {"prefill": {"W_launch": {lf.SMALL_CHUNK: 0.008}}}
    launch = lf.stated(None, small_tokens=S)
    rec = HostDeviceRecurrence(LAWS, (), lambda: 0.0, columns=columns, launch=launch)
    rec.device_end(_batch(True), 0.05, 0.0)
    assert rec.segments["W_launch"] == pytest.approx(0.178)
    rec.device_end(_batch(False), 0.05, 0.0)
    assert rec.segments["W_launch"] == pytest.approx(0.17)
    # Without the bound the provider declares no such feature: the laws refuse.
    with pytest.raises(ValueError, match="small_chunk"):
        HostDeviceRecurrence(
            LAWS, (), lambda: 0.0, columns=columns, launch=lf.stated(None)
        )


def test_a_run_file_s_host_entry_may_state_the_bound(tmp_path):
    host = {"laws": LAWS, lf.SMALL_TOKENS: S}
    compass_run._check_host(tmp_path / "run.json", host)
    for bad in (0, "2k"):
        with pytest.raises(ValueError, match=lf.SMALL_TOKENS):
            compass_run._check_host(
                tmp_path / "run.json", {**host, lf.SMALL_TOKENS: bad}
            )
