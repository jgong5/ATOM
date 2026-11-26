# SPDX-License-Identifier: MIT
"""Under ATOM_MOONCAKE_STORE_DRAFT_WINDOW the simulated producer tells the decode
where its blocks stop lacking draft rows, as Mooncake's producer does: the
block manager's mark (`draft_rows_from`) rides the transfer params, and the
decode's block manager marks its received blocks below it."""

from types import SimpleNamespace

from atom.compass.kv import connector as connector_module
from atom.compass.kv.connector import SimulatedKVConnectorScheduler


def _producer(monkeypatch, manager):
    monkeypatch.setattr(connector_module, "transfer_params", lambda seq, **_: {"x": 1})
    monkeypatch.setattr(connector_module, "engine_id", lambda name: name)
    p = object.__new__(SimulatedKVConnectorScheduler)
    p.__dict__.update(
        is_producer=True,
        _rt=SimpleNamespace(
            read_clock=lambda: 0.0, me=SimpleNamespace(name="engine-P")
        ),
        _ready={},
        _awaiting_send=set(),
        _tp_size=1,
        _dp_rank=0,
        _pp_size=4,
        _block_size=128,
        _dcp_size=1,
    )
    if manager is not None:
        p.bind_block_manager(manager)
    return p


def _seq():
    return SimpleNamespace(
        id=7, kv_transfer_params={"do_remote_decode": True}, leave_reason=None
    )


def test_a_producer_tells_the_decode_where_draft_rows_start(monkeypatch):
    p = _producer(monkeypatch, SimpleNamespace(draft_rows_from=lambda seq: 2048))
    seq = _seq()
    p.request_finished(seq)
    assert seq.kv_transfer_params_output == {"x": 1, "draft_rows_from": 2048}


def test_no_mark_and_no_manager_say_nothing(monkeypatch):
    for manager in (SimpleNamespace(draft_rows_from=lambda seq: 0), None):
        p = _producer(monkeypatch, manager)
        seq = _seq()
        p.request_finished(seq)
        assert seq.kv_transfer_params_output == {"x": 1}
