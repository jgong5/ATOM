# SPDX-License-Identifier: MIT
"""Each Mooncake KV write of a recorded P/D run, stamped on the calls that serve it.

Once the recording runner's `allocate_kv_cache` has built a Mooncake connector,
`install` wraps the connector's methods on the instance: ATOM's code is
unchanged, and a run that does not record wraps nothing. Every worker, each TP
rank included, records its own writes in ``dp<rank>.tp<rank>.writes.jsonl`` of
the record directory (pipeline stage `k` above 0:
``pp<k>.dp<rank>.tp<rank>.writes.jsonl``), named by its connector's ranks and
replaced at install. Its lines are kept in memory and written at the runner's
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
"""

import functools
import json
import os
import threading


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
        if key not in self.heard:  # the producer sends each write-done thrice
            self.heard.add(key)
            self.lines.append(
                {
                    "request_id": req_id,
                    "pp_rank": pp_rank,
                    "tp_rank": tp_rank,
                    "write_nonce": write_nonce,
                    "success": kwargs.get("success", True),
                    "t_write_done_ns": received,
                }
            )
        return call(req_id, pp_rank, tp_rank, write_nonce, **kwargs)

    def write(self) -> None:
        """Append the lines kept so far to the file, flushed."""
        lines, self.lines = self.lines, []
        self.file.write("".join(json.dumps(line) + "\n" for line in lines))
        self.file.flush()


def install(runner):
    """The `WriteRecord` of `runner`'s Mooncake connector if the run records its
    steps (`ENV`); else None, and nothing is wrapped."""
    from atom.compass.parity import ENV, record_dir
    from atom.utils.forward_context import get_kvconnector

    directory, connector = os.environ.get(ENV), get_kvconnector()
    if not directory or not hasattr(connector, "_serve_write"):  # Mooncake's
        return None
    c = connector
    name = (f"pp{c.pp_rank}." if c.pp_rank else "") + f"dp{c.dp_rank}.tp{c.tp_rank}"
    path = record_dir(directory, runner.config) / f"{name}.writes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return WriteRecord(connector, path, runner.step_clock_ns)
