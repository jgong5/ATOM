# SPDX-License-Identifier: MIT
"""A model with windowed attention, one request end to end on the simulated
clock: under the step stub (level 0) and under the step roofline alone
(level 1), as `test_vertical_slice` runs the slice model.

The model is the slice model, `ATOM_COMPASS_SLICE_MODEL`, built as a Llama of
its own widths whose odd layers attend a sliding window (`llama.py` reads it
off `layer_types`) narrower than the request's prompt; its tokenizer is the
slice model's, and no weight is read. The runs need a driver, as the slice's do.
"""

import json
from pathlib import Path

import pytest
import test_vertical_slice as vertical
from capabilities import ROOFLINE, ROOFLINE_MACHINE, capability
from test_vertical_slice import MAX_TOKENS, NEEDS_A_RUN, run_slice

from atom.compass.detect.scenario import check_run

#: Narrower than the request's prompt, so its later rows and every decode
#: attend a window shorter than their context.
WINDOW = 4


def _windowed(root: Path) -> Path:
    """The slice model's directory but its weights, its config a Llama of its
    widths whose odd layers attend `WINDOW` keys."""
    slice_model, model = Path(vertical.MODEL), root / "model"
    model.mkdir()
    for f in slice_model.iterdir():
        if f.name != "config.json" and f.suffix not in (".safetensors", ".bin"):
            (model / f.name).symlink_to(f.resolve())
    config = json.loads((slice_model / "config.json").read_text())
    kinds = ("full_attention", "sliding_attention")
    config.update(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        sliding_window=WINDOW,
        layer_types=[kinds[i % 2] for i in range(config["num_hidden_layers"])],
    )
    (model / "config.json").write_text(json.dumps(config))
    return model


def _runs(root: Path, *pricing) -> list[dict]:
    """Two runs of the windowed model, priced by `pricing` (`run_slice`)."""
    model = _windowed(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(vertical, "MODEL", str(model))
        return [run_slice(root / f"seed{s}", s, *pricing) for s in ("1", "2")]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return _runs(tmp_path_factory.mktemp("windowed"))


@pytest.fixture(scope="module")
def roofline_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("windowed_roofline")
    return _runs(root, ROOFLINE, ROOFLINE_MACHINE)


@NEEDS_A_RUN
@capability("windowed_attention")
def test_a_windowed_model_meets_every_scenario_check(runs):
    run = runs[0]
    print(f"\n{run['table']}")
    failed = check_run(
        [r["out"] for r in runs],
        [MAX_TOKENS],
        [line for _, line in run["events"]],
        {"engine": run["out"] / "parity"},
    )
    assert not failed, "\n".join(failed)


@NEEDS_A_RUN
@capability("windowed_attention", 1)
def test_a_windowed_model_meets_every_scenario_check_under_the_step_roofline(
    roofline_runs,
):
    """The level-0 case's checks but one: the engine's hand-computed
    timeline, `check_run`'s records, which it computes only for a constant
    step stub."""
    run = roofline_runs[0]
    print(f"\n{run['summary']['schedule']['seconds_by_method']}\n{run['table']}")
    failed = check_run(
        [r["out"] for r in roofline_runs],
        [MAX_TOKENS],
        [line for _, line in run["events"]],
    )
    assert not failed, "\n".join(failed)
