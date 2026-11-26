# SPDX-License-Identifier: MIT
"""A pipelined prefill stage serving its own KV writes under load (opt-in).

A link that states the ``serve_*`` fields (`ServeModel`) prices each stage's
share of a transfer where Mooncake's stage worker serves it: one write a
decode rank, each in calls of `chunk_bytes`, each call gathered behind the
work its stage's host had enqueued on the device (`DeviceQueue`), the first
after a gather. Each stage tells decode when its writes are done; decode is
ready once every stage's notice is in. A link without the fields prices as
before, and its tables carry no new channel.
"""

import copy
import math
import socket
from types import SimpleNamespace

import pytest
from conftest import atom_config_double
from test_kv_remote_prefill import L, Script
from test_spec_layers import _document, _written
from test_spec_schema import DOCUMENT

from atom.compass.clock import prefill_decode_table
from atom.compass.kv import TRANSFER_KEY, WRITE_REQ_KEY
from atom.compass.kv.connector import WRITE_DONE_KEY
from atom.compass.kv.transfer import (
    SERVE_FIELDS,
    DeviceQueue,
    Scope,
    ServeModel,
    StageServe,
    TransferModel,
)
from atom.compass.spec import MachineSpec
from atom.compass.spec.layers import NETWORK, network
from atom.utils import clock
from atom.utils.clock import LPRuntime

#: Calls of 4 bytes at 2 bytes a second, 0.1 s a call, a 1 s gather and 2 s
#: more over the calls, writes 0.5 s apart over the calls, the whole backlog.
SERVE = ServeModel(
    chunk_bytes=4,
    call_bandwidth_bytes_per_s=2.0,
    call_s=0.1,
    gather_s=1.0,
    gather_one_s=2.0,
    stagger_s=0.5,
    device_wait=1.0,
    writes=2,
)
FIELDS = {
    "serve_chunk_bytes": 8388608,
    "serve_call_bandwidth_bytes_per_s": 8.0e9,
    "serve_call_s": 3.0e-4,
    "serve_gather_s": 2.5e-3,
    "serve_gather_one_s": 9.0e-3,
    "serve_stagger_s": 1.4e-3,
    "serve_device_wait": 0.7,
}


def _spec(**fields):
    document = copy.deepcopy(DOCUMENT)
    document["interconnect"].setdefault(
        "inter_node",
        {"link_bandwidth_bytes_per_s": 5.0e10, "link_latency_s": 2e-4, "derate": 0.3},
    )
    document["interconnect"]["inter_node"].update(fields)
    return MachineSpec.from_mapping(document)


# -- the link fields ---------------------------------------------------------


def test_a_link_without_serve_fields_prices_as_before():
    spec = _spec()
    assert ServeModel.from_spec(spec, Scope.INTER_NODE) is None
    model = TransferModel.from_spec(spec, 100, Scope.INTER_NODE)
    assert model.serve is None
    link = spec.values
    assert model == TransferModel(
        link["interconnect.inter_node.link_latency_s"],
        link["interconnect.inter_node.link_bandwidth_bytes_per_s"]
        * link["interconnect.inter_node.derate"],
        100,
    )


def test_a_link_states_every_serve_field_or_none():
    with pytest.raises(ValueError, match="serve_stagger_s"):
        ServeModel.from_spec(
            _spec(**{k: v for k, v in FIELDS.items() if k != "serve_stagger_s"}),
            Scope.INTER_NODE,
        )
    model = TransferModel.from_spec(_spec(**FIELDS), 100, Scope.INTER_NODE)
    assert model.serve == ServeModel(8388608, 8.0e9, 3.0e-4, 2.5e-3, 9.0e-3, 1.4e-3, 0.7)
    assert model.serve.writes == 1  # the run sets decode's TP


def test_a_network_file_states_the_serve_fields(tmp_path):
    assert {f"interconnect.inter_node.{f}" for f in SERVE_FIELDS} <= set(NETWORK)
    extra = {f"interconnect.inter_node.{k}": v for k, v in FIELDS.items()}
    layer = network(_written(tmp_path, _document(**extra)))
    assert layer.values["interconnect.inter_node.serve_device_wait"] == 0.7
    with pytest.raises(Exception, match="serve_device_wait"):
        bad = {**extra, "interconnect.inter_node.serve_device_wait": -0.5}
        network(_written(tmp_path, _document(**bad)))
    with pytest.raises(ValueError, match="share of the device's backlog|fraction"):
        ServeModel(4, 1.0, 0.0, 0.0, 0.0, 0.0, 1.5)


# -- the device queue and a stage's writes -----------------------------------


def test_the_device_reaches_what_its_host_enqueued_by_then():
    q = DeviceQueue()
    assert q.reached(1.0) == 1.0  # no forward: nothing to wait for
    q.add(1.0, 2.0, 1.5, 4.0)  # enqueued over [1, 2], run over [1.5, 4]
    assert q.reached(0.5) == 0.5
    assert q.reached(1.5) == pytest.approx(1.5 + 0.5 * 2.5)  # half enqueued
    assert q.reached(3.0) == 4.0  # all of it
    assert q.reached(5.0) == 5.0  # the device has caught up
    q.add(6.0, 6.0, 6.0, 7.0)  # a launch of no length enqueues it all
    assert q.reached(6.0) == 7.0
    with pytest.raises(ValueError, match="in order"):
        q.add(5.0, 5.5, 7.0, 8.0)
    q.forget_before(6.5)
    assert q.reached(3.0) == 3.0 and q.reached(6.2) == 7.0


def test_a_stage_s_writes_on_an_idle_device():
    """Two writes of 3 bytes, one call each: each waits the whole gather, the
    second asked 0.5 s after the first, both moving 1.5 s."""
    job = StageServe(SERVE, 10.0, 6)
    assert job.n == 1
    done = job.advance(math.inf, DeviceQueue())
    assert done == pytest.approx(10.0 + 0.5 + 1.0 + 2.0 + 0.1 + 1.5)
    # Two calls each (8 bytes a write): the gather's per-call part halves,
    # and so does the stagger.
    job = StageServe(SERVE, 0.0, 16)
    assert job.n == 2
    each = 0.1 + 2.0
    assert job.advance(math.inf, DeviceQueue()) == pytest.approx(
        0.25 + 1.0 + 1.0 + 2 * each
    )
    assert StageServe(SERVE, 3.0, 0).advance(0.0, DeviceQueue()) == 3.0


def test_each_call_waits_for_what_the_host_had_enqueued_when_it_was_asked():
    """A forward enqueued over [0, 1] runs on the device until 9: the first
    call, asked at 0.5, waits until the device has run half of it, 4.5; the
    second, asked once the launch is over, until 9."""
    serve = ServeModel(4, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, writes=1)
    q = DeviceQueue()
    q.add(0.0, 1.0, 0.0, 9.0)
    job = StageServe(serve, 0.5, 8)
    first = 4.5 + 2.0  # gathered at 4.5, moved in 2 s
    assert job.advance(math.inf, q) == pytest.approx(max(first, 9.0) + 2.0)
    half = ServeModel(4, 2.0, 0.0, 0.0, 0.0, 0.0, 0.5, writes=1)
    job = StageServe(half, 0.5, 4)
    assert job.advance(math.inf, q) == pytest.approx(0.5 + 0.5 * 4.0 + 2.0)


def test_a_call_is_decided_only_once_its_ask_is_past():
    """Decided at 1, the first call (asked at 0) is fixed; the second, asked
    at 2, waits for the forward launched at 1.5, which a decision at 1 had
    not seen."""
    serve = ServeModel(4, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, writes=1)
    q, job = DeviceQueue(), StageServe(serve, 0.0, 8)
    assert job.advance(1.0, q) is None and job.next_ask() == 2.0
    q.add(1.5, 1.8, 1.5, 10.0)
    assert job.advance(3.0, q) == pytest.approx(10.0 + 2.0)


# -- the channels -----------------------------------------------------------

HOPS = dict(admission_s=0.54e-3, ipc_s=L, stream_s=L, router_s=L, kv_write_req_s=L)


def _kv(table):
    return sorted(
        c.name
        for lp in table.registry.ids()
        for c in table.channels_from(lp)
        if ":kv_write" in c.name
    )


def test_a_serving_prefill_takes_a_write_request_and_tells_decode_on_each_stage():
    off = prefill_decode_table(**HOPS, prefill_pp=2)
    assert _kv(off) == _kv(prefill_decode_table(**HOPS, prefill_pp=2, serve=False))
    assert _kv(off) == ["engine-D->engine-P-pp0:kv_write_req"]
    on = prefill_decode_table(**HOPS, prefill_pp=2, serve=True)
    assert _kv(on) == [
        "engine-D->engine-P-pp0:kv_write_req",
        "engine-D->engine-P-pp1:kv_write_req",
        "engine-P-pp0->engine-D:kv_write_done",
        "engine-P-pp1->engine-D:kv_write_done",
    ]
    with pytest.raises(ValueError, match="chunked"):
        prefill_decode_table(**HOPS, prefill_pp=2, serve=True, chunked=True)


# -- the connector ------------------------------------------------------------


def _ports(n):
    """`n` consecutive free ports' first."""
    for _ in range(50):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            base = probe.getsockname()[1]
        socks = []
        try:
            for p in range(base, base + n):
                s = socket.socket()
                socks.append(s)
                s.bind(("127.0.0.1", p))
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free ports")


@pytest.fixture
def served(monkeypatch):
    """Decode, and stages 1 and 2 of a three-stage prefill whose link serves
    on stages; the head's write-request end is a bare socket."""
    table = prefill_decode_table(**HOPS, prefill_pp=3, serve=True)
    lps = ("engine-D", "engine-P-pp1", "engine-P-pp2")
    rts = {}
    for lp in lps:
        rts[lp] = LPRuntime(clock.LpId(lp), table, Script())
        rts[lp].start_run()
    req, done = _ports(3), _ports(3)
    model = TransferModel(L, 1.0, 4, serve=SERVE)
    kv = {
        "kv_connector": "compass",
        TRANSFER_KEY: (model, model, model),
        WRITE_REQ_KEY: f"tcp://127.0.0.1:{req}",
        WRITE_DONE_KEY: f"tcp://127.0.0.1:{done}",
    }
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    halves = {}
    # Decode binds a notice socket per stage, so it is built first.
    for lp in lps:
        monkeypatch.setattr(clock, "_installed", rts[lp])
        parallel = SimpleNamespace(data_parallel_rank=0, data_parallel_size=1)
        producer = lp != "engine-D"
        if producer:
            parallel.pipeline_parallel_rank = int(lp[-1])
        halves[lp] = KVConnectorFactory.create_connector(
            atom_config_double(
                kv_transfer_config={
                    **kv, "kv_role": "kv_producer" if producer else "kv_consumer"
                },
                kv_cache_block_size=4,
                pipeline_parallel_size=3 if producer else 1,
                parallel_config=parallel,
            ),
            role="scheduler",
        )  # fmt: skip
    monkeypatch.setattr(clock, "_installed", None)
    import zmq

    head = zmq.Context.instance().socket(zmq.PULL)
    head.bind(f"tcp://127.0.0.1:{req}")
    yield SimpleNamespace(
        rts=rts, d=halves["engine-D"], mid=halves["engine-P-pp1"],
        last=halves["engine-P-pp2"],
    )  # fmt: skip
    head.close(linger=0)


def _at(rt, t):
    rt.conn.grants.append((t, {}))
    rt.advance_to(t)


def test_a_middle_stage_serves_its_writes_and_tells_decode(served):
    """Two blocks of 4 bytes, a call a write: write 0, asked at 0.5, waits
    for the half of a forward the host had enqueued by then (device until
    10), write 1, asked at 1, for all of it (until 20); each then pays the
    gather, its call and 2 s of bytes. Both asked by 1, the stage decides at
    1 and its notice is stamped at its end."""
    stage, rt = served.mid, served.rts["engine-P-pp1"]
    assert stage.enable_chunked_transfer, "it polls at the head's idle cadence"
    assert set(stage._socks) == {("engine-D", 0)}
    pair = ("engine-D", 0)
    stage.served_forward((0.0, 1.0, 0.0, 20.0))
    stage._serve_take(pair, "t", 0.5, 2)
    assert stage._jobs and stage.has_pending_work()
    _at(rt, 1.0)
    assert stage.has_pending_work() is False
    end = 1.0 + 19.0 + (1.0 + 2.0) + 0.1 + 2.0
    (notice,) = [e for e in rt.send_log if ":kv_write_done" in e[0]]
    assert notice[0] == "engine-P-pp1->engine-D:kv_write_done"
    assert notice[2] == pytest.approx(end + L)
    # No block to write (decode holds the prompt): done at its arrival.
    stage._serve_take(pair, "u", 2.0, 0)
    last = [e for e in rt.send_log if ":kv_write_done" in e[0]][-1]
    assert last[2] == pytest.approx(2.0 + L) and not stage._jobs


def test_the_last_stage_s_writes_wait_for_no_forward(served):
    """The same forward and write on the last stage: no backlog is waited."""
    stage, rt = served.last, served.rts["engine-P-pp2"]
    stage.served_forward((0.0, 1.0, 0.0, 20.0))
    stage._serve_take(("engine-D", 0), "t", 0.5, 2)
    _at(rt, 1.0)
    assert not stage.has_pending_work()
    (notice,) = [e for e in rt.send_log if ":kv_write_done" in e[0]]
    assert notice[2] == pytest.approx(1.0 + (1.0 + 2.0) + 0.1 + 2.0 + L)


def test_decode_posts_to_every_stage_and_waits_for_each_one_s_notice(served):
    d, rt = served.d, served.rts["engine-D"]
    assert set(d._socks) == {(f"engine-P-pp{k}", 0) for k in range(3)}
    assert list(d._peers.values()) == ["engine-P-pp0"], "a release goes to the head"
    seq = SimpleNamespace(id="y", prompt_token_ids=[1, 2, 3])
    params = {"transfer_id": "t", "remote_pp_size": 3, "dp_rank": 0}
    seq.kv_transfer_params = params
    d._listening["t"] = [seq, 3, -math.inf]
    d._queued.append(("y", params, 3))
    d._post(1.0)
    sent = sorted(e[0] for e in rt.send_log)
    assert sent == [f"engine-D->engine-P-pp{k}:kv_write_req" for k in range(3)]
    assert not d._due, "decode times nothing itself"
    assert d.has_pending_work()
