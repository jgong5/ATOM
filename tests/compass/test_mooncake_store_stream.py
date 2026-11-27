# SPDX-License-Identifier: MIT
"""``COMPASS_STORE_ON_STREAM=1``: ATOM's Store worker stages each window's pack
or unpack on its stage's forward stream and waits for it there, so a copy
waits for its stage's device backlog once a window, beside its floor
(`mooncake_store_server.STREAM_FIELDS`, `lmcache_offload.Lanes`,
`mooncake_store.StoreShell`). Unset, nothing changes."""

import functools
from types import SimpleNamespace

import pytest
from test_mooncake_store_fit import copy as copy_line
from test_mooncake_store_lp import _entry

from atom.compass.kv import connector
from atom.compass.kv.lmcache_offload import Lanes
from atom.compass.kv.mooncake_store import StandInPool, StoreShell
from atom.compass.kv.mooncake_store_server import (
    MooncakeStoreRefusal,
    check_entry,
    of_rank,
    stream_terms,
)
from atom.compass.kv.transfer import DeviceQueue

STREAM = {"store_floor_s": [1e-3, 1e-5], "load_floor_s": [2e-3, 2e-5], "stream_wait": 1.0}


class _Clock:
    def __init__(self):
        self.t = 0.0

    def read_clock(self):
        return self.t


def _rank(save=1, load=1):
    return type("R", (), {"load_workers": load, "save_workers": save})()


# --- the entry --------------------------------------------------------------------


def test_an_entry_may_state_its_stream_fields_and_a_rank_s_floors():
    check_entry(_entry(**STREAM))
    e = _entry(
        world=2,
        **STREAM | {"load_floor_s": [[1e-3, 1e-5], [3e-3, 3e-5]]},
    )
    check_entry(e)
    assert of_rank(e, "load_floor_s", 1) == [3e-3, 3e-5]
    assert of_rank(e, "store_floor_s", 1) == [1e-3, 1e-5]
    assert stream_terms(e) == {
        "store_s": [1e-3, 1e-5],
        "load_s": [[1e-3, 1e-5], [3e-3, 3e-5]],
        "stream_wait": [1.0, 1.0],
    }
    # A factor on the head stage's backlog, one a rank: a later stage waits longer.
    ranked = _entry(world=2, **STREAM | {"stream_wait": [1.0, 1.75]})
    check_entry(ranked)
    assert stream_terms(ranked)["stream_wait"] == [1.0, 1.75]


@pytest.mark.parametrize(
    "changed, match",
    [
        ({"store_floor_s": [1e-3, 1e-5]}, "and not all of"),
        ({**STREAM, "stream_wait": -0.5}, "stream_wait is"),
        ({**STREAM, "stream_wait": [1.0, 2.0]}, "stream_wait is"),  # world 1
        ({**STREAM, "load_floor_s": [1e-3]}, "load_floor_s is"),
    ],
)
def test_an_entry_s_stream_fields_are_all_or_none_and_valid(changed, match):
    with pytest.raises(MooncakeStoreRefusal, match=match):
        check_entry(_entry(**changed))


def test_the_switch_refuses_an_entry_without_its_floors():
    check_entry(_entry())
    with pytest.raises(MooncakeStoreRefusal, match="lacks store_floor_s"):
        stream_terms(_entry())


# --- the lanes --------------------------------------------------------------------


def test_lanes_without_a_stream_price_and_count_as_before():
    rt = _Clock()
    lanes = Lanes(rt, [_rank()], {"store_s": [1.0, 0.5], "load_s": [0.0, 1.0]}, "s")
    req = SimpleNamespace(req_id="r")
    assert lanes.queue(0, "save", req, 2) == (0.0, 2.0)
    assert set(lanes.copies()["save"]) == {"copies", "chunks", "seconds", "waited"}


def test_each_window_of_a_copy_waits_its_stage_s_backlog_at_the_copy_s_start():
    rt = _Clock()
    asked = []

    def windows(k, kind, req, chunks):
        return -(-chunks // 4)

    def wait(k, t, n):
        asked.append((k, t, n))
        return 0.25 * n

    lanes = Lanes(
        rt, [_rank()], {"store_s": [1.0, 0.5], "load_s": [0.0, 1.0]}, "s", (windows, wait)
    )
    req = SimpleNamespace(req_id="r")
    # 9 chunks: 3 windows of 4, each 0.25 s behind the stream, beside 1 + 0.5 x 9.
    assert lanes.queue(0, "save", req, 9) == (0.0, 5.5 + 0.75)
    # The next save waits for the lane, then meets the backlog at its start.
    assert lanes.queue(0, "save", req, 1) == (6.25, 6.25 + 1.5 + 0.25)
    assert asked == [(0, 0.0, 3), (0, 6.25, 1)]
    assert lanes.queue(0, "load", req, 0) == (0.0, 0.0)  # no chunk, no window
    got = lanes.copies()["save"]
    assert (got["windows"], got["stream_s"]) == (4, 1.0)


# --- the shell's windows and backlog ----------------------------------------------


def _shell(present=None, *, chunk=256, slots=8):
    from atom.kv_transfer.offload.mooncake_store.keys import DIGEST_BYTES

    pool = StandInPool(_entry(save_slots=slots, load_slots=2 * slots))
    rank = SimpleNamespace(_pool=pool, save_workers=1, load_workers=1)
    fake = SimpleNamespace(
        _ranks=[rank],
        _leases={} if present is None else {"r": (9.0, present)},
        _namespace="ns",
        _world=1,
        chunk=chunk,
        _device=DeviceQueue(),
        _stream_wait=[1.0],
        _busy_backlog=0.0,
        _last_launch=float("-inf"),
        _rt=SimpleNamespace(read_clock=lambda: 1.0),
    )
    fake._chunks = functools.partial(StoreShell._chunks, fake)
    req = SimpleNamespace(
        req_id="r",
        chunk_hashes=b"".join(bytes([i]) * DIGEST_BYTES for i in range(1, 13)),
        save_spec=SimpleNamespace(skip_leading_tokens=0),
    )
    return fake, req


def test_a_load_waits_once_each_window_its_pool_cuts():
    fake, req = _shell()
    # A load region of 16 slots: windows of 16 // 4 = 4 chunks.
    assert StoreShell._stream_windows(fake, 0, "load", req, 9) == 3


def test_a_save_waits_only_at_the_windows_it_packs():
    from atom.kv_transfer.offload.mooncake_store.keys import chunk_keys

    fake, req = _shell()
    # Save windows of 8 // 4 = 2 chunks: 12 chunks, 6 windows, none held.
    assert StoreShell._stream_windows(fake, 0, "save", req, 12) == 6
    keys = chunk_keys("ns", 0, 1, req.chunk_hashes, 0, 12)
    held = {key: "engine-P0" for key in keys[:7]}  # windows 0-2 held, 3 half
    fake, req = _shell(held)
    assert StoreShell._stream_windows(fake, 0, "save", req, 12) == 3


def test_the_backlog_is_the_head_stage_s_device_reach_of_what_it_enqueued():
    fake, _ = _shell()
    shell_forward = functools.partial(StoreShell.device_forward, fake)
    shell_forward((0.0, 1.0, 0.0, 3.0))  # launched over [0, 1], runs to 3
    shell_forward(None)  # a forward with no recurrence adds nothing
    backlog = functools.partial(StoreShell._stream_backlog, fake)
    assert backlog(0, 1.0) == pytest.approx(2.0)
    assert backlog(0, 0.5) == pytest.approx(1.0)  # half launched: half its run
    assert backlog(0, 5.0) == 0.0  # the device caught up
    fake._stream_wait = [0.5]
    assert backlog(0, 1.0) == pytest.approx(1.0)
    # Later windows meet the backlog the busy stage keeps (each forward's, as
    # its launch ends, in a moving mean), and none once it has gone idle.
    fake._stream_wait = [1.0]
    assert fake._busy_backlog == pytest.approx(0.1 * 2.0)  # 3 - 1, weight 0.1
    assert backlog(0, 1.0, 3) == pytest.approx(2.0 + 2 * 0.2)
    fake._rt = SimpleNamespace(read_clock=lambda: 9.0)
    assert backlog(0, 1.0, 3) == pytest.approx(2.0)


def test_a_returned_forward_reaches_each_device_watcher():
    got = []
    holder = type("W", (), {"device_forward": lambda self, w: got.append(w)})()
    connector.DEVICE_WATCHERS.add(holder)
    try:
        connector.forward_returned(SimpleNamespace(device_window=(0.0, 1.0, 0.5, 2.0)))
        connector.forward_returned(SimpleNamespace())
    finally:
        connector.DEVICE_WATCHERS.discard(holder)
    assert got == [(0.0, 1.0, 0.5, 2.0), None]


# --- the fit ----------------------------------------------------------------------


def test_the_floors_take_each_copy_s_stream_wait_off_its_seconds(tmp_path):
    from atom.compass.backends.step_linear_fit import fit_stream_floors

    lines = []
    for n in range(1, 9):
        # Each save: 1 + 2 ms a chunk, its pack 0.5 ms a segment plus a wait
        # that grows with n; each load: 3 + 1 ms a chunk, its unpack likewise.
        wait = 40.0 * (n % 3)
        segs = -(-n // 2)
        lines.append(
            copy_line(
                "[OFFLOAD-SAVE-PROF]",
                n,
                rank=0,
                segments=segs,
                pack_ms=0.5 * segs + wait,
                total_ms=1.0 + 2.0 * n + 0.5 * segs + wait,
            )
        )
        lines.append(
            copy_line(
                "[OFFLOAD-LOAD-PROF]",
                n,
                rank=0,
                in_place_windows=segs,
                unpack_ms=0.5 * segs + wait,
                total_ms=3.0 + 1.0 * n + 0.5 * segs + wait,
            )
        )
    log = tmp_path / "p.log"
    log.write_text("\n".join(lines) + "\n")
    floors, report = fit_stream_floors([log], [0])
    # The floor puts back 0.5 ms a window, which the fit takes as per chunk
    # and fixed shares: the wait, up to 80 ms a copy, is gone.
    for law, (fixed, per) in (("store_floor_s", (1e-3, 2e-3)), ("load_floor_s", (3e-3, 1e-3))):
        got = floors[law]
        assert got[1] == pytest.approx(per + 0.25e-3, abs=0.3e-3), law
        assert got[0] < 5e-3, law
        assert report[law]["ranks"][0]["copy_floor_ms_a_window"] == pytest.approx(0.5)
    assert floors["stream_wait"] == 1.0
    check_entry(_entry(**floors))


def test_a_later_rank_s_factor_is_its_mean_wait_over_the_first_rank_s(tmp_path):
    from atom.compass.backends.step_linear_fit import fit_stream_floors

    lines = []
    for r, wait in ((0, 10.0), (1, 25.0)):
        for n in range(1, 9):
            w = n * (n % 2)  # every other copy waits, a window at a time
            lines.append(
                copy_line("[OFFLOAD-SAVE-PROF]", n, rank=r, segments=n,
                          pack_ms=0.5 * n + wait * w, total_ms=1.0 + 2.0 * n + 0.5 * n + wait * w)
            )
            lines.append(
                copy_line("[OFFLOAD-LOAD-PROF]", n, rank=r, in_place_windows=n,
                          unpack_ms=0.5 * n + wait * w, total_ms=3.0 + 1.0 * n + 0.5 * n + wait * w)
            )
    log = tmp_path / "p.log"
    log.write_text("\n".join(lines) + "\n")
    floors, report = fit_stream_floors([log], [0, 1])
    assert floors["stream_wait"][0] == 1.0
    assert floors["stream_wait"][1] == pytest.approx(2.5, rel=0.05)
    check_entry(_entry(world=2, **floors))
