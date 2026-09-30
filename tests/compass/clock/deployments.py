# SPDX-License-Identifier: MIT
"""The two deployments a synthetic run drives, on the shipped channel tables.

A deployment is its channel table: the LPs are the table's registry, and an LP
sends only on the table's channels. A tensor- or data-parallel group is one LP
however wide, so nothing here has a width. The tables declare data-parallel
rank 0 only. The router of the prefill-decode deployment is not an LP: its
forward cost is part of the lookahead of the channels that cross it.
"""

import functools

from atom.compass.clock import prefill_decode_table, single_engine_table

from .participants import Engine, Frontend, Traffic

# Declared, not measured: the lookaheads the tables leave to the caller.
IPC_S = 1.0e-4
STREAM_S = 2.0e-3
ROUTER_S = 1.0e-3
KV_WRITE_REQ_S = 1.0e-4

DEPLOYMENTS = {
    "single-deployment": functools.partial(
        single_engine_table, admission_path="serving", ipc_s=IPC_S, stream_s=STREAM_S
    ),
    "prefill-decode-1p1d": functools.partial(
        prefill_decode_table,
        admission_path="serving",
        ipc_s=IPC_S,
        stream_s=STREAM_S,
        router_s=ROUTER_S,
        kv_write_req_s=KV_WRITE_REQ_S,
    ),
}


def build(table, workload):
    """Every LP of the deployment `table` describes, by `LpId`."""
    prefill = workload.prefill_steps * (workload.prefill_step_seconds,)
    decode = workload.decode_steps * (workload.decode_step_seconds,)
    if len(table.registry) == 3:
        lps = [
            Traffic(
                table, workload, "traffic->frontend:http", "traffic->frontend:http"
            ),
            Frontend(
                "frontend",
                table,
                "frontend->engine:request#dp0",
                stream="frontend->traffic:stream",
            ),
            Engine("engine", table, "engine->frontend:output#dp0", prefill + decode),
        ]
    else:
        lps = [
            Traffic(table, workload, "traffic->frontend-P:http", None),
            Frontend(
                "frontend-P",
                table,
                "frontend-P->engine-P:request#dp0",
                relay="frontend-P->frontend-D:relay",
            ),
            Frontend(
                "frontend-D",
                table,
                "frontend-D->engine-D:request#dp0",
                stream="frontend-D->traffic:stream",
            ),
            Engine(
                "engine-P",
                table,
                "engine-P->frontend-P:output#dp0",
                prefill,
                producer=True,
            ),
            Engine(
                "engine-D",
                table,
                "engine-D->frontend-D:output#dp0",
                decode,
                write_req="engine-D->engine-P:kv_write_req",
            ),
        ]
    return {lp.name: lp for lp in lps}
