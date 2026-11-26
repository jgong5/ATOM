# SPDX-License-Identifier: MIT
"""Each Mooncake KV write of a recorded P/D run, stamped on the calls that serve it.

Once the recording runner's `allocate_kv_cache` has built a Mooncake connector,
`install` wraps the connector's methods on the instance, or those of the one a
multi connector holds (`kv_connector: multi`, as a prefill instance with an
offload tier runs): ATOM's code is unchanged, and a run that does not record
wraps nothing. Every worker, each TP rank included, records its own writes in
``dp<rank>.tp<rank>.writes.jsonl`` of the record directory (pipeline stage `k`
above 0: ``pp<k>.dp<rank>.tp<rank>.writes.jsonl``), named by its connector's
ranks and replaced at install. Its lines are kept in memory and written at the runner's
`exit`, so no file call or JSON encoding runs on the transfer path. Times are
the runner's step clock, the worker's monotonic clock.

A producer rank writes a line per write request a send worker serves
(`_serve_write`): the request's ``request_id`` (decode's), ``transfer_id``
(prefill's) and ``write_nonce``; ``t_receipt_ns``, when the rank's listener
took the request in; ``t_serve_ns``, the send worker's ``[start, end]``;
``rdma``, ``[start, end, bytes]`` per `_rdma_write` call, the bytes it was
handed; ``hca``, the HCA of the engine those calls wrote through, the rail
`RailEnginePool` started it for or else the rank's own, absent where no call
ran; and ``t_notify_ns``, ``[enter, exit]``, and ``success`` of its first
write-done (`_notify_transfer_result`).

A consumer rank writes a line per write-done it receives
(`_record_write_done`), the first of each producer's repeats: ``request_id``,
the producer's ``pp_rank`` and ``tp_rank``, ``write_nonce``, ``success`` and
``t_write_done_ns``, when it arrived. A write's line joins its write-done's on
``request_id``, ``write_nonce`` and the producer's ranks.

The write-done's line also says when its write request left: ``t_load_ns``,
when `start_load_kv` took the step's metadata that asked for it, the engine
having admitted the request; ``t_post_ns``, when it was queued for the
write-request thread (`_enqueue_write_requests`); ``t_sync_ns``, when that
thread took it off the queue, with the requests queued beside it, and began
the device synchronize it posts them after (`_post_queued_write_requests`:
the return of its wait on the queue's condition, the one wait that condition
serves); and ``t_send_ns``, ``[enter, exit]`` of each `_send_on_socket` call
that sent it, one per producer stage, after that synchronize. Between
``t_sync_ns`` and the first send's enter the device ran every forward already
launched, so which forwards a write request waited for is read off these
stamps and the step record (#3720: on r256 a request's send came a forward
later than on r128b). A write request that no write-done answered by the
record's end has a line of its own with those stamps, ``request_id`` and
``write_nonce``; ``t_sync_ns`` is None on one the thread never took.
"""

import functools
import json
import os
import threading

#: A consumer's stamps of a write request, on its write-done's line.
REQUEST_STAMPS = ("t_load_ns", "t_post_ns", "t_sync_ns", "t_send_ns")


class _TakeStamped:
    """A consumer's write-request condition, `cv`, stamping on `record` when
    each `wait_for` returns on the thread that called it: the write-request
    thread taking the queue, right before its device synchronize. Everything
    else is the condition's own."""

    def __init__(self, cv, record) -> None:
        self._cv, self._record = cv, record

    def __enter__(self):
        return self._cv.__enter__()

    def __exit__(self, *exc):
        return self._cv.__exit__(*exc)

    def wait_for(self, predicate, timeout=None):
        got = self._cv.wait_for(predicate, timeout)
        self._record.taken.t = self._record.clock()
        return got

    def __getattr__(self, name):
        return getattr(self._cv, name)


class WriteRecord:
    """One worker's write lines, stamped by its connector's wrapped methods."""

    def __init__(self, connector, path, clock) -> None:
        from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import (
            PD_TRANSFER_DEADLINE_S,
        )

        self.connector, self.clock = connector, clock
        self.deadline_s = PD_TRANSFER_DEADLINE_S
        self.file = path.open("w")
        self.lines: list[dict] = []
        self.serving = threading.local()  # the line of the write a send worker serves
        self.heard: set[tuple] = set()  # the write-dones already stamped
        # When this thread last took the write-request queue (`_TakeStamped`).
        self.taken = threading.local()
        c = connector
        if c.is_producer:
            c._serve_write = functools.partial(self.serve_write, c._serve_write)
            c._rdma_write = functools.partial(self.rdma_write, c._rdma_write)
            c._notify_transfer_result = functools.partial(
                self.notify_transfer_result, c._notify_transfer_result
            )
        else:
            c._record_write_done = functools.partial(
                self.record_write_done, c._record_write_done
            )
            c.start_load_kv = functools.partial(self.start_load_kv, c.start_load_kv)
            c._enqueue_write_requests = functools.partial(
                self.enqueue_write_requests, c._enqueue_write_requests
            )
            c._send_on_socket = functools.partial(
                self.send_on_socket, c._send_on_socket
            )
            # Built in the connector's constructor, before any write request
            # starts the thread that waits on it.
            if getattr(c, "_write_cv", None) is not None:
                c._write_cv = _TakeStamped(c._write_cv, self)
        # The worker's `start_load_kv` in progress: its receives, in the order
        # it walks them, how many it has queued, and its entry.
        self.loading = threading.local()
        self.posted: dict = {}  # id(parts) -> (parts, request stamps), until sent
        self.requested: dict = {}  # request id -> its write request's stamps

    def serve_write(self, call, request_data):
        keys = ("request_id", "transfer_id", "write_nonce")
        line = {k: request_data.get(k) for k in keys}
        # The listener stamps the transfer deadline as it takes the request in.
        deadline = request_data.get("_deadline")
        if deadline is not None:
            line["t_receipt_ns"] = round((deadline - self.deadline_s) * 1e9)
        line["rdma"] = []
        self.serving.line = line
        start = self.clock()
        try:
            return call(request_data)
        finally:
            line["t_serve_ns"] = [start, self.clock()]
            self.serving.line = None
            self.lines.append(line)

    def rdma_write(self, call, target, src_addrs, dst_addrs, sizes, *args, **kwargs):
        line = getattr(self.serving, "line", None)
        start = self.clock()
        reply = call(target, src_addrs, dst_addrs, sizes, *args, **kwargs)
        if line is not None:
            line["rdma"].append([start, self.clock(), int(sum(sizes))])
            line.setdefault("hca", self.hca(kwargs.get("engine")))
        return reply

    def hca(self, engine):
        """The HCA `engine` writes through: the rail the pool started it for,
        else the rank's own (None over TCP)."""
        # Copied: another send worker may start a rail meanwhile.
        rails = dict(getattr(self.connector._rail_pool, "_engines", {}))
        own = ",".join(self.connector.ib_devices) or None
        return next((d for d, e in rails.items() if e is engine), own)

    def notify_transfer_result(self, call, request_data, *, success):
        line = getattr(self.serving, "line", None)
        enter = self.clock()
        reply = call(request_data, success=success)
        if line is not None and "t_notify_ns" not in line:
            line["t_notify_ns"] = [enter, self.clock()]
            line["success"] = success
        return reply

    def record_write_done(
        self, call, req_id, pp_rank, tp_rank=0, write_nonce=0, **kwargs
    ):
        received = self.clock()
        key = (req_id, pp_rank, tp_rank, write_nonce)
        if key not in self.heard:  # the producer resends each write-done
            self.heard.add(key)
            line = {
                "request_id": req_id,
                "pp_rank": pp_rank,
                "tp_rank": tp_rank,
                "write_nonce": write_nonce,
                "success": kwargs.get("success", True),
                "t_write_done_ns": received,
            }
            asked = self.requested.get(req_id)
            # None: a landing request, whose nonce the receiver keeps.
            if asked is not None and asked["write_nonce"] in (None, write_nonce):
                asked["answered"] = True
                line.update((k, asked[k]) for k in REQUEST_STAMPS)
            self.lines.append(line)
        return call(req_id, pp_rank, tp_rank, write_nonce, **kwargs)

    def start_load_kv(self, call, metadata):
        receives = getattr(metadata, "reqs_to_recv", None)
        if not receives:
            return call(metadata)
        loading = self.loading
        loading.ids, loading.next, loading.t = list(receives), 0, self.clock()
        try:
            return call(metadata)
        finally:
            loading.ids = None

    def enqueue_write_requests(self, call, write_requests):
        loading = self.loading
        ids = getattr(loading, "ids", None)
        if ids is not None:
            # `start_load_kv` queues each receive it takes, in its order, once
            # it is pending; one it refuses is never pending.
            pending = self.connector._pending_recv
            while loading.next < len(ids) and ids[loading.next] not in pending:
                loading.next += 1
            if loading.next < len(ids):
                req_id = ids[loading.next]
                loading.next += 1
                stamps = {
                    "request_id": req_id,
                    "write_nonce": self.connector._pending_recv_nonce.get(req_id),
                    "t_load_ns": loading.t,
                    "t_post_ns": self.clock(),
                    "t_sync_ns": None,
                    "t_send_ns": [],
                    "answered": False,
                }
                self.requested[req_id] = stamps
                for _, parts in write_requests:
                    self.posted[id(parts)] = (parts, stamps)
        return call(write_requests)

    def send_on_socket(self, call, addr, parts, *args, **kwargs):
        posted = self.posted.pop(id(parts), None)
        enter = self.clock()
        reply = call(addr, parts, *args, **kwargs)
        if posted is not None:
            stamps = posted[1]
            stamps["t_send_ns"].append([enter, self.clock()])
            if stamps["t_sync_ns"] is None:
                stamps["t_sync_ns"] = getattr(self.taken, "t", None)
        return reply

    def write(self) -> None:
        """Append the lines kept so far to the file, flushed, and a line for
        each write request no write-done has answered."""
        for req_id, asked in list(self.requested.items()):
            if not asked["answered"]:
                self.lines.append(
                    {"request_id": req_id, "write_nonce": asked["write_nonce"]}
                    | {k: asked[k] for k in REQUEST_STAMPS}
                )
            del self.requested[req_id]
        lines, self.lines = self.lines, []
        self.file.write("".join(json.dumps(line) + "\n" for line in lines))
        self.file.flush()


def mooncake(connector):
    """`connector` if it is Mooncake's P/D connector, else the one a multi
    connector holds among its subs (a prefill instance writing KV to decode
    beside an offload tier, the Mooncake Store's), else None."""
    subs = getattr(connector, "_connectors", None) or [connector]
    return next((c for c in subs if hasattr(c, "_serve_write")), None)


def install(runner):
    """The `WriteRecord` of `runner`'s Mooncake connector, its own or a multi
    connector's, if the run records its steps (`ENV`); else None, and nothing
    is wrapped."""
    from atom.compass.parity import ENV, record_dir
    from atom.utils.forward_context import get_kvconnector

    directory = os.environ.get(ENV)
    c = mooncake(get_kvconnector()) if directory else None
    if c is None:
        return None
    name = (f"pp{c.pp_rank}." if c.pp_rank else "") + f"dp{c.dp_rank}.tp{c.tp_rank}"
    path = record_dir(directory, runner.config) / f"{name}.writes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return WriteRecord(c, path, runner.step_clock_ns)
