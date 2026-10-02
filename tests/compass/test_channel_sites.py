# SPDX-License-Identifier: MIT
"""ATOM's channel sockets, pollers and output queue, wrapped where they are made.

Every channel send and receive row of the sync inventory (K4, K5) is mapped to
the function that opens its socket, and that function is read for the wrap.
Then ATOM's own threads and transport run over real zmq sockets with a runtime
installed: the PP stages, the engine output thread and the frontend's READY
wait and output thread. No clock authority is needed: frames outside the run
pass through, and a frame put inside it carries its stamp.
"""

import ast
import pickle
import queue
import threading
import time

import pytest
import zmq
from aiter_stub import stubbed_aiter

from atom.compass.audit import sync_scan
from atom.compass.clock import ChannelTable, LpId, LpRegistry, single_engine_table
from atom.distributed.pp_transport import PPStageTransport
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.sequence import get_exit_sequence
from atom.utils import clock
from atom.utils.clock import LPRuntime, WrappedSocket

with stubbed_aiter():
    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.engine_core_mgr import CoreManager

IPC = 0.001
OUT = "engine->frontend:output#dp0"
MGR = ("CoreManager.__init__", "DisaggCoreManager.__init__._connect_proc")
PP = ("PPStageTransport.__init__",)

#: Row symbol -> (the kind of each channel its call carries, the functions that
#: open those sockets). `pp_ack` is received by the stage loop before the call,
#: not on an ATOM socket, so no function here opens it.
SITES = {
    "PPStageTransport.send_metadata": (("meta",), PP),
    "PPStageTransport.recv_metadata": (("meta",), PP),
    "PPStageTransport.send_tokens": (("tokens",), PP),
    "PPStageTransport.recv_tokens": (("tokens",), PP),
    "PPStageTransport.send_kv_status": (("kv_status",), PP),
    "PPStageTransport.recv_kv_status": (("kv_status",), PP),
    "EngineCore.process_input_sockets": (
        ("request", "control"),
        ("EngineCore.process_input_sockets",),
    ),
    "EngineCore.process_output_sockets": (("output",), ("EngineCore.__init__",)),
    "PrefillEngineCore._process_engine_step": (
        ("prefill_done",),
        ("PrefillEngineCore._init_disagg",),
    ),
    "DecodeEngineCore._recv_prefill_done": (
        ("prefill_done",),
        ("DecodeEngineCore._init_disagg",),
    ),
    "DecodeEngineCore._send_block_assignment": (
        ("block_assignment",),
        ("DecodeEngineCore._init_disagg",),
    ),
    "PrefillEngineCore._recv_block_assignments": (
        ("block_assignment",),
        ("PrefillEngineCore._init_disagg",),
    ),
    "CoreManager._send_request": (("request",), MGR),
    "CoreManager._send_control": (("control",), MGR),
    "CoreManager._create_output_thread.process_outputs_socket": (("output",), MGR),
    "PPEngineCoreProc._pp_head_step": (("pp_ack",), ()),
    "PPEngineCoreProc._downstream_busy_loop": (("pp_ack",), ()),
}


class _Opened(ast.NodeVisitor):
    """Function qualname -> the kinds it wraps, plus ``poller`` for `clock.poller()`."""

    def __init__(self):
        self.scope, self.kinds = [], {}

    def _push(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_ClassDef = _push

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and getattr(f.value, "id", None) == "clock":
            mine = self.kinds.setdefault(".".join(self.scope), set())
            if f.attr == "poller":
                mine.add("poller")
            elif f.attr in ("wrap", "relay_queue"):
                arg = node.args[-1]
                head = getattr(arg, "left", None) or getattr(arg, "values", [arg])[0]
                mine.add(head.value.split("#")[0])
        self.generic_visit(node)


def test_every_channel_row_maps_to_the_site_that_wraps_it():
    rows = [
        r for r in sync_scan.load_inventory()["sites"] if r["mechanism"] in ("K4", "K5")
    ]
    opened = _Opened()
    for rel in {r["file"] for r in rows}:
        opened.visit(ast.parse((sync_scan.repo_root_from_here() / rel).read_text()))
    for r in rows:
        print(r["mechanism"], r["id"], "->", *SITES.get(r["symbol"], ("?",))[0])
    assert rows
    assert sorted({r["symbol"] for r in rows}) == sorted(SITES), "unmapped or stale"
    unwrapped = [
        (symbol, kind, fn)
        for symbol, (kinds, fns) in SITES.items()
        for fn in fns
        for kind in kinds
        if kind not in opened.kinds.get(fn, ())
    ] + [
        (r["symbol"], "poller")
        for r in rows
        if r["call"] == "poller.poll" and "poller" not in opened.kinds[r["symbol"]]
    ]
    assert not unwrapped


@pytest.fixture
def ctx():
    ctx = zmq.Context()
    yield ctx
    ctx.destroy(linger=0)


def _engine_table():
    return single_engine_table(admission_path="serving", ipc_s=IPC, stream_s=IPC)


def _pp_table(stages):
    registry = LpRegistry()
    for k in range(stages):
        registry.register(LpId(f"stage{k}"))
    table = ChannelTable(registry)
    for k in range(1, stages):
        for src, dst, kind in ((0, k, "meta"), (k, 0, "tokens"), (k, 0, "kv_status")):
            name = f"stage{src}->stage{dst}:{kind}"
            table.declare(name, LpId(f"stage{src}"), LpId(f"stage{dst}"), IPC, "inline")
    return table


def _stage(monkeypatch, table, rank, stages, ctx):
    monkeypatch.setattr(
        clock, "_installed", LPRuntime(LpId(f"stage{rank}"), table, None)
    )
    addrs = [""] + [f"inproc://meta{k}" for k in range(1, stages)]
    return PPStageTransport(rank, stages, addrs, "inproc://tok", "inproc://kv", ctx=ctx)


def test_a_real_run_gets_atoms_own_objects(ctx):
    raw = ctx.socket(zmq.PULL)
    q = clock.relay_queue("output#dp0")
    assert clock.wrap(raw, "request#dp0") is raw and clock.relay_socket(q, raw) is raw
    assert type(clock.poller()) is zmq.Poller and type(q) is queue.Queue


def test_pp_stages_talk_through_their_wrapped_sockets(monkeypatch, ctx):
    table = _pp_table(2)
    last, head = (_stage(monkeypatch, table, rank, 2, ctx) for rank in (1, 0))
    head.send_metadata("batch")
    assert last.recv_metadata() == "batch"
    last.send_tokens("out")
    assert head.recv_tokens(timeout_ms=5000) == "out"
    last.send_kv_status("kv")
    assert head.recv_kv_status(timeout_ms=5000) == [(1, "kv")]
    socks = (*head._meta_send, head._token_recv, head._kv_status_recv)
    socks += (last._meta_recv, last._token_send, last._kv_status_send)
    assert [s.ch.split(":")[1] for s in socks] == ["meta", "tokens", "kv_status"] * 2
    head.close()
    last.close()


def test_a_head_with_two_downstream_stages_refuses_by_name(monkeypatch, ctx):
    with pytest.raises(KeyError, match="stage0 has 2 channels of kind 'meta'"):
        _stage(monkeypatch, _pp_table(3), 0, 3, ctx)


def test_the_engine_output_thread_sends_each_item_with_its_put_stamp(monkeypatch, ctx):
    rt = LPRuntime(LpId("engine"), _engine_table(), None)
    monkeypatch.setattr(clock, "_installed", rt)
    engine = EngineCore.__new__(EngineCore)
    engine.label, engine.output_queue = "engine", clock.relay_queue("output#dp0")
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    thread = threading.Thread(
        target=engine.process_output_sockets, args=(f"tcp://127.0.0.1:{port}",)
    )
    thread.start()
    frames = []
    rt.start_run()
    for item in (("METRICS", {"m": 1}), [get_exit_sequence()]):
        engine.output_queue.put_nowait(item)
        if pull.poll(5000):
            frames.append(pull.recv_multipart())
        rt.end_run()  # the run closes only once every stamped item has gone out
    thread.join(5)
    assert [pickle.loads(h) for h, _ in frames] == [(OUT, IPC, 0), (OUT, None, None)]
    assert pickle.loads(frames[0][1]) == (EngineCoreRequestType.METRICS, {"m": 1})


def test_the_frontend_waits_for_ready_and_reads_output_through_the_wrapper(
    monkeypatch, ctx
):
    table = _engine_table()
    monkeypatch.setattr(clock, "_installed", LPRuntime(LpId("frontend"), table, None))
    mgr = CoreManager.__new__(CoreManager)
    mgr.ctx, mgr.label, mgr.max_pool_tokens, mgr.latest_metrics = ctx, "fe", None, {}
    pull, push = ctx.socket(zmq.PULL), ctx.socket(zmq.PUSH)
    pull.bind("inproc://out")
    push.connect("inproc://out")
    mgr.output_sockets = [clock.wrap(pull, "output#dp0")]
    engine = WrappedSocket(LPRuntime(LpId("engine"), table, None), push, OUT)
    engine.send(pickle.dumps((EngineCoreRequestType.READY, {"max_pool_tokens": 7})))
    mgr._wait_for_all_ready_signals()
    assert mgr.max_pool_tokens == 7
    thread = mgr._create_output_thread(0, mgr.output_sockets[0], "inproc://stop")
    thread.start()
    engine.send(pickle.dumps((EngineCoreRequestType.METRICS, {"m": 2})))
    deadline = time.monotonic() + 5
    while 0 not in mgr.latest_metrics and time.monotonic() < deadline:
        time.sleep(0.01)
    stop = ctx.socket(zmq.PAIR)
    stop.connect("inproc://stop")
    stop.send(b"")
    thread.join(5)
    assert mgr.latest_metrics == {0: {"m": 2}} and mgr.output_sockets[0].closed
