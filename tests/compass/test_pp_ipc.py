# SPDX-License-Identifier: MIT
"""A pipeline stage boundary priced as ATOM's HIP IPC one (`pp_send.PP_IPC`):
the sender's copy stream pushes the tensors into the receiver's ring, the
receiver's device waits for the first two only, no stream is held by a send,
and a sender waits for the ack freeing a slot `SLOTS` sends back.

The stage loops are real `StageSends` on an `LPRuntime` whose clock authority
is the scripted fake connection of the LP runtime tests; the runner calls are
a stand-in that charges each forward `C` seconds.
"""

import math
from functools import partial
from types import SimpleNamespace

import pytest

from atom.compass.clock import LpId, single_engine_table
from atom.compass.runner import pp_send
from atom.compass.runner.pp_send import CarrierModel, StageSends
from atom.utils.clock import LPRuntime
from tests.compass.clock.test_lp_runtime import FakeConn, _owner

CARRIER = CarrierModel(1024, 2.0**-10, 2.0**-12, 2.0**-20, 2.0**-8)
#: The gloo carrier rendezvous from its first byte, as on the M3.1 CAL machine.
GLOO = CarrierModel(0, 2.0**-10, 2.0**-12, 2.0**-20, 2.0**-8)
MODELS = {"rccl": CARRIER, "gloo": GLOO}
L_DATA, L_ACK = pp_send.lookaheads(MODELS)
C = 2.0**-6
DATA, ACK = "engine-pp0->engine-pp1:pp_data#dp0", "engine-pp1->engine-pp0:pp_ack#dp0"
#: Size and metadata on gloo, then hidden states, residual and an aux block.
PARTS = [("gloo", 8), ("gloo", 64), ("rccl", 4096), ("rccl", 4096), ("rccl", 8192)]


@pytest.fixture(autouse=True)
def ipc(monkeypatch):
    monkeypatch.setenv(pp_send.PP_IPC, "1")


def _stage(rank, *grants, frames=(), ready=C, end=None, parts=PARTS, pp=2):
    """Stage `rank` of `pp`, whose forward takes `C` of host time, ends its
    device work `ready` after the call (`end`, absolute, its reply's
    ``device_end``); its runner calls by name and arguments."""
    table = single_engine_table(
        admission_s=0.54e-3, ipc_s=0.5, stream_s=0.5, pp=pp, pp_links=(L_DATA, L_ACK)
    )
    thread, on_owner = _owner()
    rt = LPRuntime(LpId(f"engine-pp{rank}"), table, FakeConn(*grants), owner=thread)
    rt.start_run()
    frames, calls = list(frames), []

    def call(name, *args, wait_out=False):
        calls.append((name, *args))
        if name == "forward":
            rt.advance_to(rt.now + C)
            return SimpleNamespace(pp_send=(ready, 0.0, parts), device_end=end)
        if name == "pp_recv_posted":
            return args[0], 99.0  # its device is busy long after: nothing waits on it
        return frames.pop(0) if name == "recv_pp_frame" else True

    sends = StageSends(rt, call, rank, pp, MODELS)
    return rt, partial(on_owner, sends), calls, sends


def _batch():
    return SimpleNamespace(total_seqs_num_prefill=1, total_tokens_num=8, total_seqs_num=1,
                           produces_output=lambda: True)


def _frames(calls, kind="pp_data"):
    return [c[2] for c in calls if c[0] == "send_pp_frame" and c[1] == kind]


def test_the_ring_and_its_lead_are_atoms():
    from atom.distributed import pp_ipc

    assert pp_send.SLOTS == pp_ipc.SLOTS
    assert pp_send.IPC_LEAD == len(pp_ipc._KEYS)


def test_the_copies_run_on_one_stream_the_lead_first():
    lead, end = pp_send.ipc_copies(CARRIER, 1.0, PARTS[2:])
    each = 2.0**-12
    assert lead == 1.0 + 2.0**-10 + 2 * each + 8192 * 2.0**-20
    assert end == lead + each + 8192 * 2.0**-20


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(pp_send.PP_IPC)
    assert not pp_send.ipc_on()
    monkeypatch.setenv(pp_send.PP_IPC, "0")
    assert not pp_send.ipc_on()


def test_an_ipc_send_holds_no_stream_and_carries_its_copies():
    """Stage 0: each send's metadata leaves at the host's launch end, its copies
    at the device end, the second's after the first's; no forward's device
    work is held by a send, and no ack is waited for."""
    ready = 4 * C
    rt, run, calls, _ = _stage(0, (C, {}), (2 * C, {}), ready=ready)
    batches = [_batch(), _batch()]
    for b in batches:
        run("forward", b)
    first, second = _frames(calls)
    lead0, end0 = pp_send.ipc_copies(CARRIER, ready, PARTS[2:])
    assert (first["t_send"], first["t_ready"], first["ipc"]) == (C, ready, (lead0, end0))
    lead1, end1 = pp_send.ipc_copies(CARRIER, max(C + ready, end0), PARTS[2:])
    assert second["ipc"] == (lead1, end1)
    assert not any(hasattr(b, "compass_pp_release") for b in batches)
    assert [m[:2] for m in rt.conn.sent] == [("TAR", C), ("TAR", 2 * C)]
    assert rt.send_log[-1] == (DATA, 1, 2 * C + L_DATA)


def test_an_ipc_receiver_takes_the_lead_copy_posting_nothing_and_frees_its_slot():
    """Stage 1, its device busy to 99: the input's lead copy, not its device,
    bounds its forward's device work; the metadata is taken at its prepare
    end; at the forward's end it frees the slot at its device end."""
    t_send, lead, end = 0.25, 0.5, 0.75
    frame = {"seq": 0, "t_send": t_send, "t_ready": 0.4, "parts": PARTS, "ipc": (lead, end)}
    arrival = t_send + L_DATA
    rt, run, calls, sends = _stage(
        1, (arrival, {DATA: [(0, arrival)]}), (arrival + C, {}), frames=[frame], end=7.0
    )
    batch = _batch()
    run("forward", batch)
    meta = pp_send.recv_done(MODELS, t_send, 0.0, PARTS[:2])[1]
    assert batch.compass_pp_host == (0.0, meta)
    assert batch.compass_pp_release == lead
    assert sends.aux_in == end
    assert _frames(calls, "pp_ack") == [{"seq": 0, "freed": 7.0}]
    assert rt.send_log == [(ACK, 0, arrival + C + L_ACK)]


def test_a_middle_stage_sends_once_its_input_rest_is_in():
    """Stage 1 of three: its own send's copies wait for the aux block it relays."""
    frame = {"seq": 0, "t_send": 0.25, "t_ready": 0.4, "parts": PARTS, "ipc": (0.5, 3.0)}
    arrival = 0.25 + L_DATA
    rt, run, calls, _ = _stage(
        1, (arrival, {DATA: [(0, arrival)]}), (arrival + C, {}), frames=[frame], pp=3
    )
    run("forward", _batch())
    [sent] = _frames(calls)
    assert sent["t_ready"] == 3.0
    assert sent["ipc"] == pp_send.ipc_copies(CARRIER, 3.0, PARTS[2:])


def test_the_fourth_ipc_send_waits_for_the_ack_freeing_its_slot():
    """Stage 0: sends 0-2 fill the ring; send 3 waits on its host for the ack
    of send 0, released at `acked`, and copies no earlier than that slot's
    `freed`."""
    acked, freed = 1.0, 1.5
    rt, run, calls, _ = _stage(
        0,
        (C, {}),
        (2 * C, {}),
        (3 * C, {}),
        (4 * C, {}),
        (acked, {ACK: [(0, acked)]}),
        frames=[{"seq": 0, "freed": freed}],
    )
    for _ in range(4):
        run("forward", _batch())
    assert [m[:2] for m in rt.conn.sent] == [
        ("TAR", C),
        ("TAR", 2 * C),
        ("TAR", 3 * C),
        ("TAR", 4 * C),
        ("NER", math.inf),
    ]
    sent = _frames(calls)
    assert sent[3]["t_send"] == acked
    assert sent[3]["ipc"][0] >= freed
    # Its forward, called at 3C, ends its device work C later.
    assert sent[3]["ipc"] == pp_send.ipc_copies(
        CARRIER, max(4 * C, sent[2]["ipc"][1], freed), PARTS[2:]
    )


def test_an_ack_released_early_is_read_without_waiting():
    """Stage 0: an ack released before a send it is not needed for is read
    there, at no clock cost, and frees its slot."""
    rt, run, calls, sends = _stage(
        0, (C, {}), (2 * C, {ACK: [(0, 1.5 * C)]}), frames=[{"seq": 0, "freed": 0.1}]
    )
    run("forward", _batch())
    run("forward", _batch())
    assert sends.acks == 1 and sends.slot_free[0] == 0.1
    assert [m[:2] for m in rt.conn.sent] == [("TAR", C), ("TAR", 2 * C)]


def test_a_sharded_tensor_across_an_ipc_boundary_is_refused():
    parts = [("gloo", 8), ("rccl", 512, True)]
    frame = {"seq": 0, "t_send": 0.25, "t_ready": 0.4, "parts": parts, "ipc": (0.5, 0.5)}
    arrival = 0.25 + L_DATA
    _, run, _, _ = _stage(1, (arrival, {DATA: [(0, arrival)]}), frames=[frame])
    with pytest.raises(RuntimeError, match="ATOM runs at TP 1 only"):
        run("forward", _batch())
