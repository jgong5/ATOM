# SPDX-License-Identifier: MIT
"""A host law adopts a launch column on its held-out error per launch path.

`Tails` says a prefill of a request count ATOM recorded no decoder graph for,
two here, runs its decoder tail eagerly, `TAIL_S` dearer: a fifth of each run's
prefills. Each run's launches sit their own share off the law, as runs do:
three runs 1% under, one 3% over. Held out, each of the three is priced high,
and without the column its eager tails are priced far low, below all its other
launches, so one p50 over every launch is a lower quantile of the rest, nearer
zero than their median: scored that way, the column loses, though without it
a fifth of the launches read about 90% low. Scored per path, it wins.
"""

import numpy as np
import pytest

from atom.compass.backends.launch_features import GRAPHS, REGISTRY, LaunchFeatures
from atom.compass.backends.shape import BatchView, RequestShape, sum_tokens
from atom.compass.backends.step_labels import Sample
from atom.compass.backends.step_linear_fit import fit_host

ARCH = "TailsForCausalLM"
#: The request counts whose decoder tail ATOM recorded a graph for.
SETS = {"architecture": ARCH, "decoder": [1, 3]}
TAIL_S = 0.04
#: Each run's launches, off the law by its own share.
OFFSETS = {"a": -0.01, "b": -0.01, "c": -0.01, "d": 0.03}


class Tails(LaunchFeatures):
    features = ("eager_tail",)

    def predict(self, view, graphs):
        eager = view.produces_output and len(view.requests) not in graphs["decoder"]
        return {"eager_tail": float(eager)}


@pytest.fixture(autouse=True)
def tails(monkeypatch):
    monkeypatch.setitem(REGISTRY, ARCH, Tails())


def _samples(seed=3088):
    """Each run's output prefills of one, two or three requests, two every
    fifth; `W_launch` 2 ms plus 1 us a token, and `TAIL_S` more for an eager
    tail, off by the run's offset, within 2%."""
    rng, out = np.random.default_rng(seed), []
    key = {"engine": {GRAPHS: SETS}}
    for run, offset in OFFSETS.items():
        for i in range(60):
            seqs = 2 if i % 5 == 0 else 1 + 2 * (i % 2)
            tokens = int(rng.integers(256, 2048))
            view = BatchView(
                (RequestShape(tokens, tokens, False),) * seqs, produces_output=True
            )
            launch = Tails().predict(view, SETS)
            w = 2e-3 + 1e-6 * sum_tokens(view.requests)
            w = (w + TAIL_S * launch["eager_tail"]) * (1 + offset)
            host = {"E": 1e-4, "W_pre": 1e-4, "W_post": 1e-4}
            host["W_launch"] = w * rng.uniform(0.98, 1.02)
            provenance = {
                "source": "record",
                "run": run,
                "requests": [i],
                "config": key,
            }
            out.append(
                Sample(i, 0, view, 0.01, None, host, False, provenance, launch=launch)
            )
    return out


def test_a_column_a_minority_path_needs_is_adopted_on_its_path_s_held_out_error():
    host, report = fit_host(_samples())
    assert report["jit_steps"] == {}
    columns = report["prefill.W_launch"]["launch_columns"]
    assert columns["adopted"] == ["eager_tail"]
    # Without the column, the eager tails' path reads about 90% low held out.
    none, tail = columns["held_out"][0]["none"], columns["held_out"][1]["eager_tail"]
    assert none > 0.1 and tail < 0.03
    column = host["columns"]["prefill"]["W_launch"]["eager_tail"]
    assert column == pytest.approx(TAIL_S, rel=0.05)
