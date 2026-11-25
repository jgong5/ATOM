"""Diagnostic only (jgong5/ATOM#4151): wall-time stamps of a simulated decode engine step's hand-offs.

run.sh prep copies this file to atom/compass/hopstamps.py and hooks.patch calls it, with COMPASS_TRACE_DIR set:
`install_engine` from `atom.compass.run.engine` in each decode engine's DP member, `install_worker` from
`atom.compass.runner.model_runner` in each worker. Each process appends (time.time(), code) float64 pairs to
<COMPASS_TRACE_DIR>/<pid>.hop and names itself in <pid>.hop.json; a code is 100 * event + detail.

Engine member (the clock owner, unless noted): S clock request sent, R grant taken and R1 its release walk done,
Y0/Y1 a settle of owed rounds, A0/A1 `advance_group`, D0/D1 `_sync_dp_state`, X0/X1 its shared-memory
exchange, P0/P1 `_process_engine_step`, H0/H1 `Scheduler.schedule`, C0/C1 `call_func`, Q0/Q1 the ring's enqueue
and Qw its write done (before the doorbell rings), O1 the output thread's put of a worker reply, G1 the owner's
get of it, K0/K1 `call_func_with_aggregation`, Z0/Z1 `_postprocess`.
Worker: W0 woken by its doorbell, W1 a call dequeued, F0/F1 the forward stand-in, c0/c1 a torch.distributed
collective, wget the output thread's get of a reply and wsent its next get (the reply sent).
"""

import array
import atexit
import functools
import json
import os
import queue
import threading
import time

NAMES = ("S R R1 Y0 Y1 A0 A1 D0 D1 X0 X1 P0 P1 H0 H1 C0 C1 Q0 Qw Q1 O1 G1 K0 K1 Z0 Z1 "
         "W0 W1 F0 F1 c0 c1 wget wsent").split()
EV = {n: i for i, n in enumerate(NAMES)}
DETAIL = {"TAR": 1, "NER": 2, "PARK": 3, "REPORT": 4, "forward": 11, "dummy_execution": 12,
          "async_proc_aggregation": 13, "all_reduce": 21, "all_gather": 22, "all_gather_into_tensor": 23,
          "broadcast": 24, "barrier": 25}
_st = None


class _Stamps:
    def __init__(self, tag, role):
        path = f"{os.environ['COMPASS_TRACE_DIR']}/{os.getpid()}.hop"
        with open(path + ".json", "w") as f:
            json.dump({"tag": tag, "role": role, "pid": os.getpid(), "ppid": os.getppid(), "events": NAMES,
                       "details": DETAIL}, f)
        self.f, self.buf, self.lock, self.flushed = open(path, "ab"), array.array("d"), threading.Lock(), time.time()
        atexit.register(self.flush)

    def __call__(self, ev, detail=0):
        t = time.time()
        with self.lock:
            self.buf.append(t)
            self.buf.append(100 * ev + detail)
            if len(self.buf) >= 1 << 15 or t - self.flushed > 5:
                self._flush(t)

    def _flush(self, t):
        self.buf.tofile(self.f)
        self.f.flush()
        self.buf, self.flushed = array.array("d"), t

    def flush(self):
        with self.lock:
            self._flush(time.time())


def _wrap(owner, name, before=None, after=None, detail=None):
    fn = getattr(owner, name)
    b, a = EV.get(before), EV.get(after)

    @functools.wraps(fn)
    def stamped(*args, **kw):
        if b is not None:
            _st(b, detail(args) if detail else 0)
        try:
            return fn(*args, **kw)
        finally:
            if a is not None:
                _st(a)

    setattr(owner, name, stamped)


def install_engine(tag):
    """In a decode engine's DP member, inside `atom.compass.run.engine`, after its own wrappers."""
    global _st
    from aiter.dist.shm_broadcast import MessageQueue

    from atom.compass.runner import doorbell, shm_collectives
    from atom.model_engine import async_proc, engine_core, scheduler
    from atom.utils import clock

    _st = _Stamps(tag, "engine")
    rt = clock.LPRuntime
    _wrap(rt, "_ca_send", "S", detail=lambda a: DETAIL.get(a[1], 0))
    _wrap(rt, "_grant", "R", "R1")
    _wrap(rt, "advance_group", "A0", "A1")
    settle = rt.settle

    @functools.wraps(settle)
    def settling(self):
        if not self.owed:
            return settle(self)
        _st(EV["Y0"])
        try:
            return settle(self)
        finally:
            _st(EV["Y1"])

    rt.settle = settling
    _wrap(shm_collectives._Segment, "exchange", "X0", "X1")
    _wrap(engine_core.DPEngineCoreProc, "_sync_dp_state", "D0", "D1")
    _wrap(engine_core.EngineCore, "_process_engine_step", "P0", "P1")
    _wrap(engine_core.EngineCore, "_postprocess", "Z0", "Z1")
    _wrap(scheduler.Scheduler, "schedule", "H0", "H1")
    mgr = async_proc.AsyncIOProcManager
    _wrap(mgr, "call_func", "C0", "C1", detail=lambda a: DETAIL.get(a[1], 0))
    _wrap(mgr, "call_func_with_aggregation", "K0", "K1")
    _wrap(doorbell.RingingQueue, "enqueue", "Q0", "Q1",
          detail=lambda a: DETAIL.get(a[1][0], 0) if isinstance(a[1], tuple) and a[1] else 0)
    _wrap(MessageQueue, "enqueue", after="Qw")
    init = mgr.__init__

    @functools.wraps(init)
    def built(self, *args, **kw):
        init(self, *args, **kw)
        self.outputs_queue.__class__ = _Replies

    mgr.__init__ = built


class _Replies(queue.Queue):
    """The engine's queue of worker replies: the output thread's put, the owner's get."""

    def put_nowait(self, item):
        _st(EV["O1"])
        return super().put_nowait(item)

    def get(self, block=True, timeout=None):
        item = super().get(block, timeout)
        _st(EV["G1"])
        return item


class _Outputs(queue.Queue):
    """A worker's output queue: its output thread gets a reply, sends it, and gets the next."""

    def get(self, block=True, timeout=None):
        _st(EV["wsent"])
        item = super().get(block, timeout)
        _st(EV["wget"])
        return item


def install_worker(runner_class):
    """In a worker, as `atom.compass.runner.model_runner` is imported: a decode engine's only."""
    global _st
    name = os.path.basename(os.environ.get("ATOM_COMPASS_NOW", ""))  # now-<lp>.<member>.bin, the engine's
    if not (name.startswith("now-engine-D") and name.endswith(".bin")):
        return
    import torch.distributed as dist

    from atom.compass.runner import doorbell
    from atom.model_engine import async_proc

    _st = _Stamps(name[4:-4].replace(".", "/"), "worker")
    _wrap(doorbell.Doorbell, "spin", after="W0")
    proc = async_proc.AsyncIOProc
    get_func = proc.get_func

    @functools.wraps(get_func)
    def got(self):
        out = get_func(self)
        _st(EV["W1"], DETAIL.get(out[0], 0))
        return out

    proc.get_func = got
    loop = proc.busy_loop

    @functools.wraps(loop)
    def looping(self):
        if self.io_addrs[1] is not None:
            self.io_queues[1].__class__ = _Outputs
        return loop(self)

    proc.busy_loop = looping
    _wrap(runner_class, "forward", "F0", "F1")
    for c in ("all_reduce", "all_gather", "all_gather_into_tensor", "broadcast", "barrier"):
        _wrap(dist, c, "c0", "c1", detail=lambda a, d=DETAIL[c]: d)
