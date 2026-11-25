# SPDX-License-Identifier: MIT
"""LP `mooncake_store` and its shell under a pipelined prefill: the 1P1D
recipe's MiniMax-M3.1 PP4 x TP1 prefill offloads to the Store, each stage's
rank putting its own layers' chunk. An entry states the fields its ranks
differ in a rank (`PER_RANK`): each object takes its rank's bytes, each stage
plans its own chunk, and each rank's copies run on its own laws."""

import copy
import re

import pytest
from test_mooncake_store_lp import (
    ENTRY,
    STORE,
    _config,
    _entry,
    _run,
    started,  # noqa: F401  a fixture
)

from atom.compass import run as compass_run
from atom.compass.kv import mooncake_store_server as store
from atom.compass.kv.lmcache_offload import Lanes
from atom.compass.kv.mooncake_store_server import (
    Master,
    MooncakeStoreRefusal,
    check_entry,
    check_plans,
    of_rank,
)

#: A PP4 x TP1 prefill's chunks: three 16-layer stages and the last's 12
#: layers beside the draft's KV (bytes for a 256-token chunk, made up).
BYTES = [3, 3, 3, 5]


def _pp_entry(**changed):
    return _entry(**{"world": 4, "chunk_bytes": list(BYTES), **changed})


def _key(rank, chunk, world=4):
    return f"ns/w{rank}of{world}/{chunk:032x}"


def test_an_entry_may_state_its_per_rank_fields_a_rank():
    e = _pp_entry(store_s=[[1e-3, 1e-4]] * 3 + [[2e-3, 2e-4]])
    check_entry(e)
    assert [of_rank(e, "chunk_bytes", r) for r in range(4)] == BYTES
    assert of_rank(e, "store_s", 3) == [2e-3, 2e-4]
    assert of_rank(e, "load_s", 3) == ENTRY["load_s"]
    assert of_rank(e, "save_slots", 2) == ENTRY["save_slots"]
    assert of_rank(_entry(), "chunk_bytes", 0) == 1


@pytest.mark.parametrize(
    "field, value",
    [
        ("chunk_bytes", [3, 3, 3]),  # not one a rank
        ("chunk_bytes", [3, 3, 3, 0]),
        ("save_workers", [1, 1, 1, 1]),  # not a field a rank may state
        ("store_s", [[1e-3, 1e-4]] * 3),
        ("lookup_s", [[1e-3, 1e-4]] * 4),
    ],
)
def test_an_entry_s_per_rank_field_is_one_valid_value_a_rank(field, value):
    with pytest.raises(MooncakeStoreRefusal, match=f"{store.ENTRY}.{field} is"):
        check_entry(_pp_entry(**{field: value}))


def test_each_object_takes_its_rank_s_bytes_from_put_to_eviction():
    m = Master(_pp_entry(capacity_bytes=10**6))
    keys = [_key(r, 0) for r in range(4)]
    m.publish(m.put(keys, ["g0"] * 4, "engine-P-pp0", 1.0), 1.0)
    assert m.used == sum(BYTES)
    assert [m.size(k) for k in keys] == BYTES
    flat = Master(_entry(capacity_bytes=10**6))
    assert flat.size(_key(3, 0, 1)) == 1


def test_a_put_past_the_capacity_counts_each_rank_s_bytes():
    m = Master(_pp_entry(capacity_bytes=sum(BYTES) - 1))
    with pytest.raises(
        MooncakeStoreRefusal, match=re.escape(f"4 objects ({sum(BYTES)} B)")
    ):
        m.put([_key(r, 0) for r in range(4)], ["g0"] * 4, "engine-P-pp0", 1.0)


def test_each_stage_plans_the_chunk_its_ranks_state():
    blocks = ENTRY["chunk_tokens"] // 128
    e = _pp_entry(chunk_bytes=[b * blocks for b in BYTES])
    plans = {
        f"engine-P-pp{k}": {"block_bytes": b, "tp": 1} for k, b in enumerate(BYTES)
    }
    check_plans(e, plans, 128)
    wrong = {**plans, "engine-P-pp3": {"block_bytes": 3, "tp": 1}}
    with pytest.raises(MooncakeStoreRefusal, match="engine-P-pp3 plans a 6 B chunk"):
        check_plans(e, wrong, 128)
    # One stage's plans against a PP4 entry: the world is short.
    with pytest.raises(MooncakeStoreRefusal, match="TP 1 of 1 stage"):
        check_plans(e, {"engine-P": {"block_bytes": 3, "tp": 1}}, 128)


def test_a_pp4_prefill_launches_against_an_entry_of_world_4(
    tmp_path, started  # noqa: F811
):
    run = _run(tmp_path)
    run[store.ENTRY] = _pp_entry()
    config = _config(STORE, pipeline_parallel_size=4)
    compass_run._launch_mooncake_store(config, run, None)
    assert len(started) == 1


def test_a_pp4_prefill_names_the_rank_whose_slots_differ(
    tmp_path, started  # noqa: F811
):
    run = _run(tmp_path)
    run[store.ENTRY] = _pp_entry(save_slots=[ENTRY["save_slots"]] * 3 + [7])
    config = _config(STORE, pipeline_parallel_size=4)
    with pytest.raises(ValueError, match=r"rank 3's save_slots \d+ \(the entry's 7\)"):
        compass_run._launch_mooncake_store(config, run, None)
    assert started == []


class _Clock:
    def __init__(self):
        self.t = 0.0

    def read_clock(self):
        return self.t


def test_each_rank_s_copies_run_on_its_own_laws():
    rt = copy.copy(_Clock())
    ranks = [type("R", (), {"load_workers": 1, "save_workers": 1})()] * 2
    lanes = Lanes(
        rt, ranks, {"store_s": [[1.0, 0.5], [2.0, 1.0]], "load_s": [0.0, 1.0]}, "s"
    )
    req = type("Q", (), {"req_id": "r"})()
    assert lanes.queue(0, "save", req, 2) == (0.0, 2.0)
    assert lanes.queue(1, "save", req, 2) == (0.0, 4.0)
    assert lanes.queue(1, "load", req, 3) == (0.0, 3.0)
    flat = Lanes(rt, ranks, {"store_s": [1.0, 0.5], "load_s": [0.0, 1.0]}, "s")
    assert flat.queue(1, "save", req, 2) == (0.0, 2.0)


def test_a_pp4_prefill_s_store_logs_give_its_per_rank_fields_and_laws(
    tmp_path, monkeypatch, caplog
):
    """One log of a PP4 x TP1 prefill: each stage's rank logs its own chunk
    bytes and slots, and its copies (`rank=`) take its own seconds a chunk."""
    import json

    from test_mooncake_store_fit import CAPACITY, FLAGS, START, copy, lookups
    from test_mooncake_store_fit import started as started_line

    from atom.compass.backends.step_linear_fit import offload_main

    chunk = [3 << 20, 3 << 20, 3 << 20, 5 << 20]
    slots = [26, 26, 26, 15]
    lines = [
        started_line(
            START
            | {"rank": r, "world": 4, "chunk_bytes": chunk[r], "save_slots": slots[r]}
        )
        for r in range(4)
    ]
    for r in range(4):
        for n in range(1, 5):
            ms = {"total_ms": (1.0 + r) + (2.0 + r) * n, "pack_ms": 1, "put_ms": 1}
            lines.append(copy("[OFFLOAD-SAVE-PROF]", n, rank=r, **ms))
            ms = {"total_ms": 0.5 + 1.5 * n, "get_ms": 1, "unpack_ms": 1}
            lines.append(copy("[OFFLOAD-LOAD-PROF]", n, rank=r, **ms))
    lines += lookups(monkeypatch, caplog, range(1, 4))
    log = tmp_path / "p.log"
    log.write_text("\n".join(lines) + "\n")
    out = tmp_path / "entry.json"
    offload_main(
        [
            str(log),
            "--out",
            str(out),
            "--master-flags",
            FLAGS,
            "--capacity-bytes",
            str(CAPACITY),
        ]
    )
    entry = json.loads(out.read_text())
    check_entry(entry)
    assert (entry["world"], entry["chunk_bytes"], entry["save_slots"]) == (
        4,
        chunk,
        slots,
    )
    assert entry["load_slots"] == START["load_slots"]
    for r in range(4):
        assert of_rank(entry, "store_s", r) == pytest.approx(
            [(1 + r) * 1e-3, (2 + r) * 1e-3]
        )
        assert of_rank(entry, "load_s", r) == pytest.approx([0.5e-3, 1.5e-3])
    assert len(entry["fit"]["store_s"]["ranks"]) == 4
