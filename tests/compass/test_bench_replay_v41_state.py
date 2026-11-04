# SPDX-License-Identifier: MIT
"""A replayed DeepSeek-V4.1 row's state, seated at its cached context.

V4.1's `PagedAttentionCache` keeps each state slot's position and refuses a
request whose state is not where its forward starts. Each replayed row is a
new request in a fresh pool, so unseated, every decode row, and every prefill
row past a cached context, is refused at its first forward. The batches are
the replay rule's, over `test_filler_tokens`'s tokenizer, for a V4.1 engine
under a DSpark drafter of five drafts, as cell A runs; ATOM's cache judges them
in a process of its own (`v41_state_probe.py`), since its module needs a
declared device to import.
"""

import json
import os
import pathlib
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_bench_replay_drafter import spec
from test_filler_tokens import model_dir  # noqa: F401  the model_dir fixture
from test_parity_record import _config

from atom.compass.bench import replay

TREE = pathlib.Path(__file__).resolve().parents[2]
PROBE = pathlib.Path(__file__).with_name("v41_state_probe.py")
#: DSpark5's verify width.
WIDTH = 6
#: Cell A's c16 step 126877, four verify rows, its contexts scaled down; the
#: prefill rows: a chunk past a cached prefix, two middle chunks, a fresh prompt.
LINES = {
    "decode": [
        [WIDTH, 208, True],
        [WIDTH, 129, True],
        [WIDTH, 57, True],
        [WIDTH, 35, True],
    ],
    "prefix": [[40, 104, False]],
    "chunks": [[32, 96, False], [8, 40, False]],
    "prompt": [[50, 50, False]],
}
OUTPUT = {"decode": True, "prefix": True, "chunks": False, "prompt": True}


@pytest.fixture(scope="module")
def probed(model_dir, tmp_path_factory):  # noqa: F811
    """Each line's batch, and what V4.1's cache saw of it (`v41_state_probe`)."""
    built = _config(
        blocks=64,
        speculative_config=spec(WIDTH - 1, True),
        pool_entries_per_req={"state": 1},
    )
    built.model, built.trust_remote_code = str(model_dir), False
    built.kv_cache_block_size = 32
    built.hf_config = SimpleNamespace(model_type="deepseek_v41_text")
    rule = replay.ReplayRule(built)
    batches = {
        name: rule.batch({"rows": rows, "produces_output": OUTPUT[name]})
        for name, rows in LINES.items()
    }
    root = tmp_path_factory.mktemp("v41_state")
    (root / "batches.pkl").write_bytes(pickle.dumps(batches))
    done = subprocess.run(
        [sys.executable, str(PROBE), str(root)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode == 0, f"rc {done.returncode}\n{done.stderr[-4000:]}"
    return batches, json.loads((root / "probed.json").read_text())


def cached(batch):
    return (batch.context_lens - batch.num_scheduled_tokens).tolist()


def test_an_unseated_verify_row_is_refused_where_its_forward_starts(probed):
    """The issue's failure: the first row's state is wanted at 208 - 6."""
    _, seen = probed
    assert seen["unseated"] == (
        f"Request {_first_id(probed)} needs state at 202, found 0; "
        "replay from a recoverable boundary"
    )


def test_a_seated_verify_step_passes_v41s_check_with_its_history_untouched(probed):
    batches, seen = probed
    assert seen["seated"] == {
        "refused": None,
        "positions": cached(batches["decode"]),
        "history_kept": True,
    }
    assert cached(batches["decode"]) == [202, 123, 51, 29]


def test_a_seated_row_is_not_reset_so_its_forward_runs_no_more_than_a_served_one(
    probed,
):
    """Seated, a row is not a fresh request: the cache leaves its slot as it
    stands, as for a request that reached that position, and does none of the
    zeroing a fresh slot gets inside the forward."""
    _, seen = probed
    assert seen["untouched"] == {"refused": None, "state_kept": True}


def test_every_repeat_is_seated_again_after_a_forward_moves_the_state(probed):
    """A forward leaves each state where its row ended, and the next repeat's
    rows start where the first's did: each repeat is seated before its forward,
    and the check a decode step defers to the next one passes too."""
    _, seen = probed
    repeats = seen["repeats"]
    assert repeats["refused"] is None
    assert repeats["moved"] == [context for _, context, _ in LINES["decode"]]
    assert "needs state at 202, found 208" in repeats["unseated"]
    assert repeats["seated"] == [None] * 4


@pytest.mark.parametrize("name", ["prefix", "chunks", "prompt"])
def test_a_replayed_prefill_row_starts_where_its_cached_context_ends(probed, name):
    """A fresh prompt is seated at 0 and reset there, as a new request is."""
    batches, seen = probed
    assert seen[name] == {"refused": None, "positions": cached(batches[name])}


def test_a_dummy_batch_and_a_cache_that_keeps_no_position_are_left_alone(probed):
    _, seen = probed
    assert seen["left_alone"] is True


def _first_id(probed):
    batches, _ = probed
    return batches["decode"].req_ids[0]
