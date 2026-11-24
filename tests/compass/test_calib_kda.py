# SPDX-License-Identifier: MIT
"""A design for a KDA model (`kda_state`) counts the law's KDA prefill columns.

The step linear backend prices a KDA model's prefill by its state work
(`BatchView.kda`: state rows, saves, forks, moves; the chunk padding off the
rows). A deployment stating `kda_state` makes the rehearsal's views and the
reachable range's vertices state those counts (`Deployment.kda`), and a
ones-law vector over KDA columns states the recurrence's chunk, so the
identifiability check counts the columns with the backend itself instead of
refusing them. Off, nothing a view states changes.
"""

import copy

from test_calib_v1 import _config, _law, _vector

from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.step_linear import KDA, KDA_CHUNK
from atom.compass.calib import columns, deploy, design, engine, ident

GLM = {"kda_state": True, "prefix_unit": 64}


def _kda_config(**deployment):
    raw = _config()
    raw = copy.deepcopy(raw)
    raw["deployment"] |= deployment
    raw["columns"]["prefill_variants"] = [[], list(KDA)]
    return deploy.parse(raw)


def _kda_law(cfg) -> dict:
    law = _law(cfg.deployment.rungs, cfg.deployment.dp)
    coef = law["prefill"]["coefficients"] | {k: 1e-4 for k in KDA}
    law["prefill"] = _vector(coef, kda_chunk=KDA_CHUNK)
    return law


def test_a_kda_deployment_states_each_rows_state_work():
    dep = _kda_config(**GLM).deployment
    rows = (
        RequestShape(8192, 8192, False),  # a first chunk: no state to restore
        RequestShape(512, 40960, False),  # resumes past its first token: a move
        RequestShape(4, 9000, True),  # a decode row holds a state too
    )
    assert dep.kda(rows) == (3, 0, 0, 1)
    assert _kda_config().deployment.kda(rows) is None


def test_the_kda_columns_are_counted_by_the_backend_with_the_rows_padding():
    dep = _kda_config(**GLM).deployment
    rows = (RequestShape(100, 100, False), RequestShape(64, 8256, False))
    view = BatchView(rows, produces_output=True, kda=dep.kda(rows))
    assert columns.backend_prices("prefill", "kda_moves")
    got = columns.values(view, "prefill", ["step", *KDA], dep)
    assert got == (1.0, 2.0, 0.0, 0.0, 1.0, 28 / KDA_CHUNK)


def test_a_kda_design_rehearses_and_assesses_its_kda_variant():
    cfg = _kda_config(**GLM)
    law = engine.Law(_kda_law(cfg), None, cfg.deployment.dp)
    got = engine.rehearse(cfg, design.generate(cfg), law)
    assert got.refused == 0
    prefill = [s for s in got.steps if s.kind == "prefill"]
    assert prefill and all(v.kda is not None for s in prefill for _, v, *_ in s.ranks)
    assert all(v.kda is not None for v in ident.prefill_vertices(cfg))
    vector = next(
        v for v in ident.vectors(cfg, got.steps, law) if v["kind"] == "prefill"
    )
    kda = next(r for r in vector["variants"] if r["variant"] != "base")
    assert "error" not in kda
    # no save and no fork on these views: the range holds them at 0
    assert {"kda_saves", "kda_forks"} <= set(kda["structural"])


def test_without_kda_state_the_views_state_none_and_the_variant_is_not_counted():
    cfg = _kda_config()
    law = engine.Law(
        _law(cfg.deployment.rungs, cfg.deployment.dp), None, cfg.deployment.dp
    )
    got = engine.rehearse(cfg, design.generate(cfg), law)
    assert all(v.kda is None for s in got.steps for _, v, *_ in s.ranks)
    vector = next(
        v for v in ident.vectors(cfg, got.steps, law) if v["kind"] == "prefill"
    )
    kda = next(r for r in vector["variants"] if r["variant"] != "base")
    assert "error" in kda
