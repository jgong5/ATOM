# SPDX-License-Identifier: MIT
"""`compass fit steps --backend step-grid` refusing a `--grid` input whose
engine or sampling is not its targets', naming the targets', which it compares
against, and not `measure steps`' flags."""

import json
import re

import pytest
from test_backend_step_grid_fit import (  # noqa: F401
    clients,
    fit,
    fitted,
    measured,
    mixed,
    nodes,
    planted_roots,
    record,
    run,
)


def sampled(directory):
    export = json.loads((directory / "client.json").read_text())
    for request in export["requests"].values():
        request["sampling"] = {"temperature": 0.25, "top_p": 1.0}
    (directory / "client.json").write_text(json.dumps(export))


def engined(directory):
    path = directory / "steps" / "dp0.engine.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | {"kv_cache_dtype": "x"}))


@pytest.mark.parametrize(
    "edit, refusal",
    [
        (
            sampled,
            (
                '{g} served requests under {{"temperature": 0.25, "top_p": 1.0}}, and '
                'the targets\' sampling is {{"temperature": 0.0, "top_p": 1.0}}'
            ),
        ),
        (
            engined,
            "{g} ran under another engine than the targets': kv_cache_dtype 'x' there",
        ),
    ],
)
def test_a_grid_input_off_its_targets_refuses_naming_the_targets(
    tmp_path, run, edit, refusal  # noqa: F811
):
    g = record(tmp_path, run, nodes() + mixed(seed=2), name="grid")
    edit(g)
    for line in nodes():
        measured(tmp_path, run, line)
    a = record(tmp_path, run, mixed(seed=3))

    with pytest.raises(SystemExit, match=re.escape(refusal.format(g=g))) as refused:
        fitted(tmp_path, a, *clients(g), *clients(a), grid="grid")
    assert "--sampling" not in str(refused.value)
    assert "ATOM's arguments" not in str(refused.value)
    assert not (tmp_path / "store" / "step_grid_bundle").exists()
