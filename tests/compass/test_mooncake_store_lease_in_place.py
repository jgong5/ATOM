# SPDX-License-Identifier: MIT
"""LP `mooncake_store` with `COMPASS_STORE_LEASE_IN_PLACE`: a lease keeps a
group where it was filed, so a prefix a lookup found leaves at the first round
after its lease, and the load that relied on the lookup fails its get.

The master's index is driven directly, with the switch off and on. The engine
test drives ATOM's `MooncakeStoreOffloadScheduler` through `StoreShell` on the
step loop against LP mooncake_store in a process of its own, as
test_mooncake_store_shell does: a prefix is saved, another prompt saved after
it, a request looks the prefix up, its lease runs out while it waits, a third
save takes the Store past its mark, and the request's load renews and gets at
its dispatch. Kept in place, the prefix's oldest chunk is the round's; refiled
at the lookup's deadline, the later prompt's chunk is, and the load succeeds.
"""

from __future__ import annotations

import array

import pytest
from test_mooncake_store_lp import (
    MASTER,
    _entry,
    _group,
    _keys,
    _publish,
)
from test_mooncake_store_shell import (  # noqa: F401  (factory is a fixture)
    CHUNK_BYTES,
    CHUNKS,
    TOKENS,
    Engine,
    _answered,
    _save,
    _seq,
    factory,
)
from test_mooncake_store_shell import _entry as _shell_entry

from atom.compass.kv import mooncake_store_server as store
from atom.compass.kv.mooncake_store_server import LEASE_IN_PLACE, Master


@pytest.fixture(params=[False, True], ids=["refiled", "in_place"])
def in_place(request, monkeypatch):
    if request.param:
        monkeypatch.setenv(LEASE_IN_PLACE, "1")
    else:
        monkeypatch.delenv(LEASE_IN_PLACE, raising=False)
    return request.param


@pytest.mark.parametrize("value, on", [("", False), ("0", False), ("1", True)])
def test_the_switch_reads_as_set_unless_empty_or_0(monkeypatch, value, on):
    monkeypatch.setenv(LEASE_IN_PLACE, value)
    assert store.lease_in_place() is on and Master(_entry()).in_place is on
    monkeypatch.delenv(LEASE_IN_PLACE)
    assert not store.lease_in_place()


def test_a_leased_group_is_passed_in_place_and_leaves_from_its_bucket(in_place):
    """The design's round (test_mooncake_store_lp): group B, chunk 1, leased by
    a lookup at 32 s to 42 s, is the round's at 38 s either way, passed: filed
    again at its deadline when refiled, kept in its bucket in place. The next
    round, at 43 s, takes the oldest filed groups: B first in place, its lease
    run out, and B survives when refiled, behind every group filed before 42 s."""
    m = Master(_entry())
    for chunk in range(90):
        _publish(m, chunk, 8.0 + chunk / 10)
    assert sorted(m.exists(_keys(1)[0], 32.0)) == _keys(1)[0]
    _publish(m, 90, 37.995)
    r = m.evict(m.next_tick(37.995))
    assert r.evicted == [_group(0), *map(_group, range(2, 7))] and r.objects == 12
    assert r.skipped == [(_group(1), "lease" if in_place else "moved")]
    for chunk in range(91, 97):  # 170 + 12 of 200 bytes: past the 0.90 mark
        _publish(m, chunk, 43.0)
    r = m.evict(m.next_tick(43.0))
    assert r.goal == 11 and r.skipped == []  # ceil(182 objects x 0.06)
    first = [1, *range(7, 12)] if in_place else list(range(7, 13))
    assert r.evicted == list(map(_group, first))
    assert bool(m.exists(_keys(1)[0], 44.0)) is not in_place


def test_in_place_a_running_lease_still_keeps_its_group_through_a_round(monkeypatch):
    """In place, as refiled, a group whose lease runs at the round is passed,
    and the first round after its lease takes it."""
    monkeypatch.setenv(LEASE_IN_PLACE, "1")
    flags = ["--eviction_high_watermark_ratio=0.5", "--eviction_ratio=1"]
    m = Master(_entry(capacity_bytes=10, master=MASTER + flags))
    _publish(m, 1, 0.5)
    _publish(m, 0, 1.0)
    m.exists(_keys(1)[0], 0.52)  # leased to 10.52
    r = m.evict(10.5)
    assert r.evicted == [_group(0)] and r.skipped == [(_group(1), "lease")]
    assert m.evict(10.53).evicted == [_group(1)]


def _prompt(req_id: int, base: int):
    """A prompt of `TOKENS` tokens no other prompt here shares a chunk with."""
    seq = _seq(req_id)
    seq.token_ids = array.array("i", range(base, base + TOKENS))
    return seq


def test_a_load_whose_prefix_left_after_its_lookup_fails_and_prefills(
    tmp_path, monkeypatch, factory, in_place  # noqa: F811
):
    """A Store of 12.5 chunks, mark 0.90, ratio 0.05. Prompt 1 is saved, then
    prompt 3; request 2 looks prompt 1 up (a lease to its reply plus 10 s) and
    waits 11.5 s; prompt 4's save takes the Store to 12 of 12.5 chunks, and
    its round evicts one object (ceil(8 x 0.11)). Kept in place it is prompt
    1's chunk 0, filed first, its lease run out: request 2's load renews at its
    dispatch, three of its four chunks present, its all-or-nothing get fails,
    and ATOM counts the failed load and prefills it. Refiled, prompt 1's
    groups sit at the lookup's deadline, prompt 3's chunk 0 leaves, and the
    load succeeds."""
    entry = _shell_entry(capacity_bytes=CHUNK_BYTES * 25 // 2)
    engine = Engine(tmp_path, monkeypatch, **{store.ENTRY: entry})
    shell, rt = engine.shell, engine.rt
    client = shell._inner._lookup_client
    _save(shell, _prompt(1, 0))
    _save(shell, _prompt(3, 10_000))
    while rt.now < 0.2:
        engine.step()
    b = _prompt(2, 0)
    assert client.submit(b, "2")
    _answered(engine, client, "2")
    looked_up = rt.now
    assert shell.get_num_new_matched_tokens(b) == (CHUNKS * 8, True)
    while rt.now < looked_up + 11.5:
        engine.step()
    _save(shell, _prompt(4, 20_000))
    for _ in range(10):
        engine.step()
    shell.update_state_after_alloc(b)
    [load] = shell.build_connector_meta().requests
    assert load.load_spec.lmcache_cached_tokens == CHUNKS * 8
    assert shell.counts["renewals"] == 1
    present = shell._leases["2"][1]
    assert len(present) == (CHUNKS - 1 if in_place else CHUNKS)
    while not any("2" in done for _, done in engine.reported):
        engine.step()
    engine.step()
    stats = shell._inner.get_statistics()
    assert (stats["load_failures"], stats["load_requests"]) == (
        (1, 0) if in_place else (0, 1)
    )
    assert shell.counts["past_lease_loads"] == 0 and rt.refusals == []
    handed = engine.finish()
    assert handed["refusals"] == [] and handed["store"]["evicted"] == 1
    assert handed["store"]["renewals"] == 1
