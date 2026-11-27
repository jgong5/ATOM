# SPDX-License-Identifier: MIT
"""Above one pipeline stage `keyed` states the stages and each one's span of
the layers as ATOM splits them (`get_pp_indices`), so a step law fitted at one
stage refuses a pipeline's run, and one fitted at a split refuses a run at
another, naming what differs; ``VLLM_PP_LAYER_PARTITION`` keys by the split it
gives, not by being set. A run at one stage states neither, so its key does
not move.

No device.
"""

import pytest
from test_drive_runfile import build, step_linear
from test_run_step_grid import deployment

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.run.step_grid import BUCKETS, PP_LAYERS, keyed

#: ATOM's override of the pipeline's layer split (`get_pp_indices`).
PARTITION = "VLLM_PP_LAYER_PARTITION"
#: The deployment's 64 layers over two stages, as ATOM splits them by default.
HALVES = [[0, 32], [32, 64]]


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in (*PERFORMANCE_KEYS, BUCKETS, PARTITION):
        monkeypatch.delenv(name, raising=False)


def staged(tmp_path, pp: int):
    r = deployment(tmp_path)
    r.config.pipeline_parallel_size = pp
    return r


def test_one_stage_states_neither_the_stages_nor_their_spans(tmp_path):
    engine = keyed(staged(tmp_path, 1).config)

    assert "pipeline_parallel_size" not in engine and PP_LAYERS not in engine


def test_a_law_fitted_at_one_stage_refuses_a_pipeline_naming_its_stages_and_spans(
    tmp_path,
):
    law = step_linear(tmp_path, staged(tmp_path, 1))
    run = build(tmp_path, "--step-linear", law.key.token())

    with pytest.raises(ValueError, match="fitted under another engine config") as e:
        compass_run.refuse_unfitted_law(run, staged(tmp_path, 2).config)
    assert "pipeline_parallel_size None in the law and 2 here" in str(e.value)
    assert f"{PP_LAYERS} None in the law and {HALVES} here" in str(e.value)


def test_a_pipeline_law_holds_at_its_split_however_stated_and_refuses_another(
    tmp_path, monkeypatch
):
    r = staged(tmp_path, 2)
    law = step_linear(tmp_path, r)
    run = build(tmp_path, "--step-linear", law.key.token())

    assert keyed(r.config)[PP_LAYERS] == HALVES
    monkeypatch.setenv(PARTITION, "32,32")
    compass_run.refuse_unfitted_law(run, r.config)
    monkeypatch.setenv(PARTITION, "30,34")
    with pytest.raises(ValueError, match="fitted under another engine config") as e:
        compass_run.refuse_unfitted_law(run, r.config)
    moved = [[0, 30], [30, 64]]
    assert f"{PP_LAYERS} {HALVES} in the law and {moved} here" in str(e.value)
