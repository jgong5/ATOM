# SPDX-License-Identifier: MIT
"""A replayed DeepSeek-V4.1 row's state, seated at its cached context.

V4.1's `PagedAttentionCache` keeps each state slot's position and refuses a
request whose state is not where its forward starts. Each replayed row is a
new request in a fresh pool, so unseated, every decode row, and every prefill
row past a cached context, is refused at its first forward: by the step
bench, and by the op bench at each placement it prepares. The batches are the
replay rule's, over `test_filler_tokens`'s tokenizer, for a V4.1 engine under
a DSpark drafter of five drafts, as cell A runs, and, for the op bench, with
no drafter; ATOM's cache judges them in a process of its own
(`v41_state_probe.py`), since its module needs a declared device to import.
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


def v41_rule(model_dir, drafter):  # noqa: F811
    """The replay rule for a V4.1 engine over 64 blocks of 32 tokens, one state
    slot a request, under `drafter`."""
    built = _config(
        blocks=64, speculative_config=drafter, pool_entries_per_req={"state": 1}
    )
    built.model, built.trust_remote_code = str(model_dir), False
    built.kv_cache_block_size = 32
    built.hf_config = SimpleNamespace(model_type="deepseek_v41_text")
    return replay.ReplayRule(built)


@pytest.fixture(scope="module")
def probed(model_dir, tmp_path_factory):  # noqa: F811
    """Each line's batch, and what V4.1's cache saw of it (`v41_state_probe`)."""
    rule = v41_rule(model_dir, spec(WIDTH - 1, True))
    batches = {
        name: rule.batch({"rows": rows, "produces_output": OUTPUT[name]})
        for name, rows in LINES.items()
    }
    root = tmp_path_factory.mktemp("v41_state")
    plain = v41_rule(model_dir, None)
    pickled = batches | {"rule": plain, "drafted": rule}
    (root / "batches.pkl").write_bytes(pickle.dumps(pickled))
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
    """Past a cached prefix a row's slot is kept as it stands; a fresh prompt
    starts at 0 and its slot is reset there, as a new request's is, whatever
    was seated."""
    batches, seen = probed
    assert seen[name] == {
        "refused": None,
        "positions": cached(batches[name]),
        "reset": name == "prompt",
    }


def test_v41s_own_builder_starts_each_seated_verify_row_where_it_was_seated(
    probed,
):
    """Prepared as ATOM's runner prepares a decode, each verify row starts at
    `context_lens - (K + 1)`, where the bench seated it."""
    _, seen = probed
    assert "needs state at 202, found 0" in seen["builder"]["unseated"]
    assert seen["builder"]["seated"] is None


def test_every_op_bench_placement_is_seated_in_the_slots_it_is_given(probed):
    """`Stub.point` gives each placement the next two of 8 state slots: past
    the first, a row's slot is not its batch row, and every placement passes
    the check only when its own slots are seated."""
    _, seen = probed
    ops = seen["ops"]
    assert ops["refused"] is None
    assert ops["slots"][:4] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    unseated = seen["ops_unseated"]
    assert unseated["slots"] == [[0, 1]]
    assert "needs state at 19, found 0" in unseated["refused"]


def test_the_step_bench_seats_a_batch_before_the_forward_below_it(probed):
    """`BenchModelRunner.forward` seats the rows, then calls the recording
    runner's forward, which opens the window."""
    batches, seen = probed
    assert seen["runner"] == [cached(batches["decode"])]


def test_a_dummy_batch_and_a_cache_that_keeps_no_position_are_left_alone(probed):
    _, seen = probed
    assert seen["left_alone"] is True


def _first_id(probed):
    batches, _ = probed
    return batches["decode"].req_ids[0]


@pytest.mark.parametrize("point", ["ops_verify", "ops_prefill"])
def test_every_op_bench_placement_of_a_verify_or_cached_prefill_point_prepares(
    probed, point
):
    """`Stub.point`'s placements of DSpark5's tentative verify rows, each
    beginning over the last one's uncommitted state, and of a chunk past a
    cached prefix, through V4.1's own builder: every one prepares."""
    _, seen = probed
    placed = seen[point]
    assert placed["refused"] is None
    assert len(placed["slots"]) > 1


def test_a_leaf_of_v41s_state_counts_it_once_a_slot_its_placements_hold(probed):
    """V4.1's compressor states, bound on its builder: the leaf's 7 placements
    of two rows take the 4 slot pairs 8 slots hold, its states counted once a
    pair, its activation once a copy."""
    _, seen = probed
    placed = seen["ops_v41_state"]
    [record] = placed["got"]
    assert placed["refused"] is None and record["placements"] == 7
    pairs = [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert placed["slots"] == pairs + pairs[:3]
    assert set(record["metadata"].values()) == {
        f"attn_metadata_builder.cache.state.{f}.0"
        for f in ("compress_kv", "compress_score")
    }
    assert record["footprint_bytes"] == 7 * 32 + 4 * placed["read"]


def test_a_v41_state_field_taken_whole_counts_each_row_s_slot_once_a_layer(probed):
    """`aten.select.int` takes a layer of V4.1's state field from the whole
    field, `[layers, slots, ...]`, its slots strided past its layers: each row's
    slot counts once a layer, at the 4 slot pairs 8 slots hold, and no slot
    past the rows'."""
    _, seen = probed
    placed = seen["ops_v41_field"]
    [record] = placed["got"]
    assert placed["refused"] is None and record["placements"] == 7
    assert set(record["metadata"].values()) == {
        f"attn_metadata_builder.cache.state.{f}" for f in ("window", "compress_score")
    }
    assert record["footprint_bytes"] == 7 * 32 + 4 * placed["read"]


def test_a_v41_page_field_taken_whole_binds_it_and_counts_once_a_placement(probed):
    """`aten.select.int` takes a layer of V4.1's page field from the whole
    field, `[layers, pages, rows, dim]`, of the 4 PAGEs a capture's pool held:
    it binds that field, not the state's buffer, so it counts once a placement."""
    _, seen = probed
    placed = seen["ops_v41_page"]
    [record] = placed["got"]
    assert placed["refused"] is None and record["placements"] == 7
    assert record["metadata"] == {"main_1": "attn_metadata_builder.cache.pages.main_1"}
    assert record["footprint_bytes"] == 7 * 32 + 7 * placed["read"]


@pytest.mark.parametrize("pool", ["whole", "ring"])
def test_a_v41_window_ring_reached_through_its_pool_counts_once_a_slot(probed, pool):
    """V4.1's pool, `[rows, dim]` over its PAGEs and its state arena, as a
    capture's 4 PAGEs held it. Taken whole, a row a request is the window ring
    row its decode writes, in the arena: it counts once a slot pair of the 4
    that 8 slots hold. Beside a ring cut from it (`ring_view`), at a leaf
    stating its window and compress ratio, the whole pool's plane rows count
    once a placement and the ring's once a slot pair."""
    _, seen = probed
    placed = seen[f"ops_v41_{pool}_pool"]
    [record] = placed["got"]
    assert placed["refused"] is None and record["placements"] == 7
    assert record["metadata"]["pool"] == "attn_metadata_builder.cache.pool"
    pages = 0 if pool == "whole" else placed["read"] // 2
    ring = placed["read"] - pages
    assert record["footprint_bytes"] == 7 * 32 + 7 * pages + 4 * ring
