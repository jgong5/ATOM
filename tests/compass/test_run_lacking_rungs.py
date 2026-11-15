# SPDX-License-Identifier: MIT
"""`compass run --check` names each decode rung a configuration schedules that
its step linear law has no vector for, the eager rung where the scheduler's
bound passes the widest captured size, as stale coverage of the law.

The laws are checked in under `scripts/compass/laws/`, each held against
`test_run_step_grid`'s deployment, at the ladder and bounds a test sets, with
platform readings published at the law's stack.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import platforms
import pytest
from test_memory_readings import DOCUMENT
from test_run_lacking import checked, served
from test_run_step_grid import deployment

from atom.model_engine.arg_utils import EngineArgs

LAWS = Path(__file__).resolve().parents[2] / "scripts" / "compass" / "laws"
#: Decode rungs 1, 2, 4, 8, 12, 16, 20, 24 and 28; no eager vector, no draft law.
DPA8 = LAWS / "deepseek-v4-pro-dpa8-mi355x-crs005-decode.json"
#: Decode rungs 1 to 8, 12, 16, 24, 32, 40, 48, 56 and 64, its draft law's too.
DSPARK5 = LAWS / "deepseek-v4.1-flash-tp4-dspark5-mi455-a0.json"
#: The capture sizes ATOM's server declares when its arguments name none.
DEFAULT_SIZES = json.loads(EngineArgs.cudagraph_capture_sizes)


def mtp(drafts: int) -> SimpleNamespace:
    return SimpleNamespace(method="mtp", num_speculative_tokens=drafts, model=None)


def named(tmp_path, monkeypatch, capsys, law: Path, **config):
    """The rungs, and the line naming them, `--check` lists as lacking for the
    law file `law` on the deployment with `config`; None when it lists none."""
    served(monkeypatch, deployment(tmp_path, **config).config)
    at = json.loads(law.read_text())["platform"]
    platform = platforms.flags(tmp_path / "store", DOCUMENT, tmp_path, at=at)
    checked(tmp_path, "--step-linear", str(law), platform=platform)
    out = capsys.readouterr().out
    line = re.search(
        r"^lacking step linear law: stale \(coverage\)\n  refused: (no decode "
        r"vector.* for rungs? (.*), which this configuration schedules; .*)$",
        out,
        re.MULTILINE,
    )
    return line and (line[2].split(", "), line[1])


def test_a_ladder_the_law_covers_lists_no_rung_however_long_a_context(
    tmp_path, monkeypatch, capsys
):
    # A context past every record the law was fitted on is priced past its
    # hull, not refused, so the configuration's longest is no rung to list.
    sizes = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    covered = {"capture_sizes": sizes, "max_num_seqs": 28, "max_model_len": 10**7}

    assert named(tmp_path, monkeypatch, capsys, DPA8, **covered) is None


def test_every_rung_of_the_default_ladder_without_a_vector_is_listed(
    tmp_path, monkeypatch, capsys
):
    # The bounds of the DeepSeek-V4-Pro template's single-node configurations.
    s2 = {"capture_sizes": DEFAULT_SIZES, "max_num_seqs": 512}
    s2 |= {"max_num_batched_tokens": 16384, "speculative_config": mtp(3)}

    rungs, line = named(tmp_path, monkeypatch, capsys, DPA8, **s2)

    assert rungs == ["3", "5", "6", "7", "32", "48", "64", "128", "256", "512"]
    assert "in the law or its draft law" not in line


@pytest.mark.parametrize(
    "bounds, eager",
    [
        ({"max_num_seqs": 12}, True),
        ({"max_num_seqs": 8}, False),
        # Four tokens a sequence: 35 tokens hold 8 sequences, the widest size.
        ({"max_num_batched_tokens": 35, "speculative_config": mtp(3)}, False),
        ({"max_num_batched_tokens": 36, "speculative_config": mtp(3)}, True),
        ({"max_num_seqs": 8, "enforce_eager": True}, True),
    ],
)
def test_the_eager_rung_is_listed_where_the_bound_passes_the_widest_size(
    tmp_path, monkeypatch, capsys, bounds, eager
):
    config = {"capture_sizes": [1, 2, 4, 8, 16], "max_num_seqs": 48, **bounds}

    found = named(tmp_path, monkeypatch, capsys, DPA8, **config)

    assert (found and found[0]) == (["eager"] if eager else None)


@pytest.mark.parametrize("serial", [False, True])
def test_a_rung_its_draft_law_lacks_is_listed_unless_its_draft_is_serial(
    tmp_path, monkeypatch, capsys, serial
):
    law = json.loads(DSPARK5.read_text())
    ladder = sorted(map(int, law["engine_config"]["ladder"]))
    config = {"capture_sizes": ladder, "max_num_seqs": ladder[-1]}
    config |= {"speculative_config": mtp(5)}
    assert named(tmp_path, monkeypatch, capsys, DSPARK5, **config) is None

    del law["draft"]["decode"]["24"]
    if serial:
        law["draft"]["forward"] = {**law["draft"]["prefill"], "passes": 5}
    (tmp_path / "law.json").write_text(json.dumps(law))
    found = named(tmp_path, monkeypatch, capsys, tmp_path / "law.json", **config)

    if serial:
        assert found is None
    else:
        assert found[0] == ["24"] and "in the law or its draft law for rung 24" in (
            found[1]
        )
