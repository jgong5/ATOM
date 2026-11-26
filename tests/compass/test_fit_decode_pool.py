# SPDX-License-Identifier: MIT
"""`COMPASS_FIT_DECODE_POOL`: the decode rungs from a rung up share one
`context` coefficient, keep a `step` each, and price no `graph_padding`."""

import numpy as np
import pytest

from atom.compass.backends import step_linear_fit as slf


def _rows(step, ctx_coef, contexts, padding=1e6):
    return [((1.0, float(c), float(padding)), step + ctx_coef * c) for c in contexts]


def _vector(**coefficients):
    return {"coefficients": coefficients, "hull": "kept", "fit": {"samples": 3}}


def test_pooled_rungs_share_their_context_and_keep_a_step_each():
    law = {
        "decode": {
            "32": _vector(step=0.019, context=1.4e-10, graph_padding=1e-11),
            "80": _vector(step=0.029, context=0.0, graph_padding=8e-10),
            "96": _vector(step=0.032, context=0.0, graph_padding=3.8e-10),
        }
    }
    rows = {
        "80": _rows(0.030, 2e-10, np.linspace(3e5, 1.1e6, 20)),
        "96": _rows(0.033, 2e-10, np.linspace(5e5, 1.3e6, 20)),
    }
    slf._pool_decode(law, rows, 64)
    for k, step in (("80", 0.030), ("96", 0.033)):
        c = law["decode"][k]["coefficients"]
        assert c["step"] == pytest.approx(step, rel=1e-6)
        assert c["context"] == pytest.approx(2e-10, rel=1e-6)
        assert c["graph_padding"] == 0.0
        pooled = law["decode"][k]["fit"]["pooled"]
        assert pooled["rungs"] == [80, 96] and pooled["from"] == 64
        assert pooled["samples"] == 20 and abs(pooled["centre"]) < 1e-9
        assert law["decode"][k]["hull"] == "kept"
    # a rung below the pool keeps its own vector
    assert law["decode"]["32"]["coefficients"]["graph_padding"] == 1e-11


def test_a_rung_with_another_column_or_unfitted_keeps_its_own_vector():
    law = {
        "decode": {
            "80": _vector(step=0.029, context=0.0, graph_padding=8e-10),
            "96": _vector(step=0.03, context=1e-10, graph_padding=0.0, rowmod=1e-9),
        }
    }
    rows = {
        "80": _rows(0.030, 2e-10, np.linspace(3e5, 1.1e6, 20)),
        "96": _rows(0.033, 2e-10, np.linspace(5e5, 1.3e6, 20)),
        "112": _rows(0.036, 2e-10, np.linspace(6e5, 1.4e6, 20)),  # unfitted: no vector
    }
    slf._pool_decode(law, rows, 64)
    assert law["decode"]["80"]["fit"]["pooled"]["rungs"] == [80]
    assert "pooled" not in law["decode"]["96"]["fit"]
    assert "112" not in law["decode"]


def test_the_switch_is_a_rung_and_off_by_default():
    assert "decode_pool" not in slf.env_options({})
    assert "decode_pool" not in slf.env_options({slf.DECODE_POOL_ENV: "0"})
    assert slf.env_options({slf.DECODE_POOL_ENV: "40"})["decode_pool"] == 40
    with pytest.raises(ValueError, match="not a whole number"):
        slf.env_options({slf.DECODE_POOL_ENV: "x"})
