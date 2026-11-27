# SPDX-License-Identifier: MIT
"""The per-process bootstrap of a simulated run: which LP a process is, and its clock.

A run is on when `ENV` names a run file, a JSON object every process of the
deployment reads; spawned processes inherit it. The API server's
``--compass-run`` sets it. With it unset, every function here returns at once
and ATOM runs as it always does.

One engine LP at one stage, one frontend LP, and the traffic LP the harness
drives, over `single_engine_table`. At data-parallel width above one, each DP
rank's engine process is a member ``dp<rank>`` of the one engine LP and owns
its rank's channels. Each pipeline stage's engine process is an engine LP of
its own, ``engine-pp<stage>``, and each stage's forward waits for the stage
before it on ``pp_data`` (`atom.compass.runner.pp_send`). Under RapidServe,
which a run file states as ``rapidserve: true``, the prefill and decode engine
processes are engine LPs ``engine-P`` and ``engine-D``, which send each other
``block_assignment`` and ``prefill_done``. The Clock Authority is co-hosted in the frontend's
process and served at the run file's ``clock_endpoint``, which every LP and
member, the frontend's included, connects to.

- `frontend(config)` wraps the `CoreManager` construction in `LLMEngine`: it
  selects the simulated runner, serves the authority and installs the
  frontend LP's runtime; leaving it, after READY from every engine, starts
  the run. It refuses an `LLMEngine` the API server is not building and the
  Atomesh entrypoints, and makes a sync utility command on the frontend's
  clock owner refuse. Each image a chat request's media preparation loads
  charges that tokenizer-station job the run file's ``image_preprocess_s``;
  an image named by a remote URL is refused. After each metrics refresh
  the frontend samples its Prometheus series (`metrics.sample_frontend`).
  On a worker node of a multi-node DP group it only selects the simulated
  runner.
- `engine(config)` wraps the engine's construction in its own process the
  same way, and maps every DP rank to the engine LP for the process-group
  check. The engine ignores SIGTERM from just before it says READY. The
  workers it starts sleep between its calls (`atom.compass.runner.doorbell`).
  Under DP the group its engine core builds runs its small collectives over
  shared memory when its ranks share a host
  (`atom.compass.runner.shm_collectives`).
- `runner(runner, tree)` installs the cost backend and the device readings on a
  `CompassModelRunner` where it would build its model, in the worker process.
- `tokenizer(tokenizer, config)` charges the served tokenizer's ``encode`` and
  ``decode`` to the frontend's clock, at the rates of its ``host`` entry's
  ``tokenizer`` term. A term that refuses the tokenizer stops the run as a
  signal would, so the engines leave and the server exits on the refusal.
  On a real run recording its steps it records each such call, and stamps
  each HTTP request's intake, beside the step record instead.
- `http_intake()` is what `atom.utils.compass_loop.HttpChannel` charges the
  frontend's event loop for each request it takes in, from the ``host``
  entry's ``http_intake`` term; `http_stream()`, for each SSE event a
  streamed response sends, from its ``http_stream`` term.
- `engine_done(engine)` and `frontend_done(engine)` close each side after the
  finish, each handing in its LP's refusals; the frontend's, when it co-hosts
  the authority, then writes the step table and the run summary. A SIGINT or
  SIGTERM that stopped the frontend's event loop
  (`atom.utils.compass_loop.CompassEventLoop.stopped`) is a stop: the
  frontend's runtime first ends every LP's run through the authority, so each
  engine leaves its loop at its ``+inf`` grant and hands in, and a client
  still bound gets ``+inf`` too, each grant releasing every message sent its
  LP; `frontend_done` hands in the signal, which the summary counts as an
  ``ended:`` reason, and the server leaves once the summary is written from
  every hand-in. A server whose ``client_runs`` is ``null`` serves until it is
  stopped: a stop with no client run open is its run's end, summarised as the
  finish would be.
- `engine_loop(engine)` wraps the engine's loop: under ``step_grid`` it asks the
  workers for the bundle before it, `engine_done` after it, and if either
  raises, the engine hands in the exception as an ``ended:`` reason, with the
  bundle; `engine_done` raises for a loop that left before the finish. The
  engine ignores SIGTERM until it has handed in. A
  hand-in before the finish is an early end: the authority's host writes the
  step table and the run summary at once, each LP's clock as it stands, from
  what has been handed in. Before each metrics push the engine samples its
  Prometheus series (`metrics.sample_engine`).
- An engine or frontend LP's last hand-in first samples its Prometheus
  series once more at its last finite clock (`metrics.Sampler.final`), then
  writes the series it sampled, `metrics.FILE` beside its hand-in, whole or
  not at all.

Run file keys: ``clock_endpoint``, ``bound_s`` (finite), the hops `HOPS` (each
optional, the lookahead of the channels it names, `HOP_FLOOR_S` where absent,
which the run summary names under ``unmeasured_hops``), one of ``step_stub`` (a
`Coefficients` mapping,
priced by `StepStubBackend`), ``step_linear`` (a `StepLinearLaw` mapping,
priced by `StepLinearBackend`; its ``platform``, the platform fields of its key,
and its ``engine_config``, `step_grid.keyed` as it was fitted, refuse a run under
any other, a run whose ladder is a prefix of the law's read at the law's, and a
law missing either refuses; a runner that records prefill token-graph buckets
refuses a law whose prefill vector, or else engine config, states none or others, and keeps
one with no prefill vector, which prices no prefill step (`refuse_other_buckets`); a list of them is one per engine config, a process taking its own), ``step_grid`` (an artifact store's root and a
bundle's key, priced by `StepGridBackend`; see `step_grid`), ``op_graph``
(``true``: `OpGraphBackend` prices each step from the `op_graph` ``store``
holds at its shape and output, traced under this process's torch, each file it
ran and read reading here as it did (`capture.lookup`), and loaded under them and
its model (`_op_graphs`), the run's TP and EP widths and KV cache format; capture traces
eager forwards, so a graph-mode run reads the eager graphs too; with
``price_list``, the fields of a `price_list` key ``store`` holds, loaded
under this run's readings, it prices each leaf from it first; with
``atom_args``, ATOM's arguments the server was started with, which `compass
run` writes, a shape the store holds no graph at is traced first, in the
one process each engine spawns for its traces, the model built once there,
and published there, or from a tree with uncommitted files kept for this run
alone, unpublished, and a trace that fails ends the run, naming the shape),
``step_roofline`` (optional, ``true`` or ``false``: with ``true`` a
`StepRooflineBackend` prices the steps the backend refuses, through
`StepFallbackBackend`, as it always does under ``step_grid``, from the
modules `run.modules.walk` reads off the meta build and its block drafter's,
`run.modules.drafter`), ``padding`` (optional:
``none`` prices a replayed graph's real rows in place of the rows it pads to,
which ``executed``, the default, prices),
``on_refusal`` (optional: ``mark``, the default, charges a step the cost
backend refuses no device time, its answered terms included, and a segment the
host laws refuse 0 s, hands in its ``cost:`` reason and continues; ``abort``
ends the run there, which still writes the summary),
``client_runs`` (optional: how many client runs the server serves, one traffic
LP after another, 1 when absent and ``null`` for no limit; each ends when no
essential work is left, and the last one's end finishes the run),
``local_grants`` (optional, ``true``, the default, or ``false``: with ``true``
the authority gives each grant a ceiling, below which the LP answers its own
clock calls and reports them with its next one, a DP member at its settle,
`ClockAuthority`; the step table is the same),
``image_preprocess_s`` (optional: seconds the frontend takes to prepare one
image of a chat request; without it a request with an image is refused),
``host`` (optional: ``laws``, per forward kind and host segment, coefficients
on ``(1, tokens, seqs)``, and per segment of ``dummy``, a DP rank's dummy
forward at a step its group prefills, priced by ``decode``'s where it states
none; with any laws, a forward whose kind has none for a
segment it runs and ``skip`` does not name is refused, that segment costing 0 s
(`HostDeviceRecurrence.device_end`); ``key``, the `host_key` they were
fitted under and the platform fields of their store key, which a run with
laws must match; ``admit``, coefficients on
a forward's admissions, which add to its ``E``, or null to refuse a forward
that admits a request, and absent from an entry whose ``E`` laws hold them;
``skip``, segments zeroed for a what-if run, which the graders refuse;
``columns``, per kind and segment, coefficients on launch features
(`backends.launch_features`), predicted under the graph sets ``key``'s engine
states, which the run takes as its own, since it records no graphs; a key stating
other prefill token-graph buckets than its runner's refuses (`refuse_other_buckets`);
``tokenizer``, the served tokenizer's ``backend`` (``fast`` or ``slow``), the
``fingerprint`` of the ``tokenizer.json`` it was fitted on (null for a slow
tokenizer, which loads none, as Kimi-K3.1's tiktoken one: `_fingerprint`), and, each
optional, ``encode``, one-item calls, ``encode_batch``, a backend
``encode_batch`` of several items, ``decode``, and ``segment``, the segment
tokenizer call's own work over the ids its ``encode_batch`` calls return, each ``fixed_s`` and a
positive ``tokens_per_s``, a ``segment`` optionally also a positive ``chars_per_s`` over
the characters its stored-prefix search compared, and a ``decode`` in their place
``fixed_s`` alone over ``tokens``, the ``[lo, hi]``
token counts outside which a decode is refused and charged 0 s; ``http_intake``,
``fixed_s`` and ``per_byte_s``, with ``idle_s`` optional and ``bytes``, the
``[lo, hi]`` body bytes it was fitted over, outside which an intake is refused
and charged 0 s; a prefill-decode run's prefill entry's ``relay``,
``fixed_s`` and ``per_byte_s``, with ``bytes`` optional, the time a response
takes from prefill's server to decode's app call through the router, whose
part past ``router_s`` decode's API server loop serves each relayed response,
queued with the others in flight there (`relay`), and which a
prefill-decode run with no entry stating it names under
``uncharged_host_terms``; a term an entry lacks, or every term with no entry, is
charged 0 s and the run summary names it, a kind of tokenizer call as
``tokenizer.<kind>`` and a ``segment`` without ``chars_per_s`` as
``tokenizer.segment.chars_per_s``, under ``uncharged_host_terms``; ``hops``, the
`HOPS` `runfile` writes to the run file; a list of them as ``step_linear``'s),
or in its place ``replay`` (a real run's parity record directory, or for a
prefill-decode run one holding each deployment's as ``prefill/`` and
``decode/``, ``prefill<k>/`` and ``decode<k>/`` above one of a side, and for a
RapidServe run each engine's as ``prefill/`` and ``decode/``: each step
ends when the real one did, by `atom.compass.parity.replay`, and the cost
backend's price reaches only the cost record),
``machine`` (a machine spec mapping), ``store`` (the artifact
store's root, where `memory` reads the configuration's memory readings) and
``backend`` (the backend's ``digest``
and its fit ``provenance``, which nothing here reads), both written by
`atom.compass.drive.runfile`, and ``out_dir``,
where the step table, the run summary and each server LP's hand-in go, one
file written as the LP leaves its loop. It holds the LP's refusals: an
engine's are the worker's refused commands and the engine's refused clock
calls, a frontend's its executor's refused jobs and its refused clock calls.
An engine's also holds what its worker's steps cost, for the summary's
predicted seconds and their shares by method (`backends.cost.GROUPS`), and
its counts of steps extrapolated and of steps a fallback priced, and of its
lazy traces with their wall seconds, which the summary's cost half totals,
with each DP prefill whose G it traced one by one (`_op_graphs`).
It holds the memory the worker predicts too, for
``python -m atom.compass.memory.check``, with the assumption a
`memory_readings` member it read states, which `compass run` records, and
under ``step_grid`` the bundle's key, digest and ``describe()`` line, which the
summary carries. A speculative engine's holds its acceptance input and its
scheduler's accepted-draft histogram (`_acceptance`), which the summary carries
by LP over its steps, with their mean acceptance length. The authority
removes an earlier run's files there as it starts. At the end of client run
``k`` every server LP hands in to ``client-<k>/`` there, and the authority's
host writes the step table and that client run's summary, its steps taken as
each hand-in less the one at client run ``k - 1``'s end; the finish writes the
last client run's beside the run summary. An optional
``data_parallel_size``, 1 when absent, is the deployment's DP width, which
under DP-attention is its TP size times its DP size, and an optional
``pipeline_parallel_size``, 1 when absent, its pipeline width, above which the
machine states each stage send carrier under ``interconnect.intra_node.pp``; the traffic LP reads the
channel table from the run file alone. Each DP rank's engine process hands in
its own refusals, as ``<lp>.<member>``. A key `RENAMED_KEYS` lists refuses,
naming its name now.

A run file that also has ``router_s``, the router's per-request forward or
null where no entry states it, describes a prefill-decode run over
`prefill_decode_table`: its hops are prefill's, ``router_s`` among them, its
``decode_hops`` decode's ``ipc_s``, ``ipc_out_s`` and ``stream_s``, and
``kv_link`` (``intra_node``
or ``inter_node``) names the link its KV transfers are priced on. An optional
``router_in_s`` makes the router the ``router`` LP, with that lookahead on each
hop into it; it prices no work, so it is no server LP and hands nothing in.
Its ``router`` entry, `router_binary`'s record of the binary that ran it, is
per-run and goes to the summary as it is. A deployment
whose ``kv_connector`` is ``compass`` is its prefill (``kv_producer``) or
decode side: `engine(config)` makes it the ``engine-P`` or ``engine-D`` LP and
binds the connector's transfer model, which prices a block at the paged KV
entry each engine's runner planned, and each transfer's state slot and staging
row at the ones it planned, when the model's transfer carries them
(`_planned_transfer`); deployments that plan different blocks, slots or rows
are refused before the run starts. The write-request
address comes with the deployment's own ``kv_transfer_config``. Above DP
width one each DP rank is a member of its side's engine LP, and ATOM's DP dispatch on each side picks the
rank a request runs on. Optional ``prefill_data_parallel_size`` and
``decode_data_parallel_size``, ``data_parallel_size`` when absent, give each
side its own DP width, and only such a run may name them. Its optional
``prefill_pipeline_parallel_size``, 1 when absent, is the prefill side's
pipeline width, in place of ``pipeline_parallel_size``; above one, prefill
stage ``k`` is the engine LP ``engine-P-pp<k>``, and the decode side runs one
stage. Optional ``prefill_instances`` and ``decode_instances``,
1 when absent, run that many deployments of a side, each naming its index in
its ``kv_transfer_config`` as ``compass_instance``; above one, its LPs carry
the index (``engine-P1``), and the router is the ``router`` LP, which chooses
among them. An optional ``worker_order`` gives, by side (``P`` or ``D``), the
(instance, DP rank) pairs in the order that LP's policies list the side's
workers, each once (`worker_order`); a side it does not name is sorted by URL
and rank. `frontend(config)` refuses such a
deployment unless the authority is standalone: a prefill-decode run spans two
API servers that must share one.

A deployment that offloads KV to an lmcache server (``lmcache_mp``), the
prefill side of a prefill-decode run, needs an ``lmcache`` entry, and a run
file with one needs such a deployment. `frontend(config)` then starts LP
``lmcache`` (`lmcache`, `atom.compass.kv.lmcache_server.LMCacheLP`) in a
process of its own, given the deployment's KV block tokens, of which the
server's chunk must be a whole number. The entry holds ``server``, the
``lmcache server`` flags; the request loop's laws, ``lookup_s`` (``[fixed, per
prompt chunk]``) and ``request_s``; and optional ``copy_width``, 1 when absent.
It states no ``writers`` or ``state_bytes``, which the launch refuses: every
writer rank of a DP rank shares that rank's one transfer channel, and at its
first stamped request LP ``lmcache`` sizes a chunk and a rank's state image
from what each member of the engine it serves planned (`_lmcache_plan`), as a
prefill-decode connector prices its transfer. The copy engine
prices on the machine's ``interconnect.host_link``. Its channels' lookahead is
``lmcache_ipc_s``, the run's ``ipc_s`` when absent (`add_lmcache`); with
``OFFLOAD_MP_LOOKUP_THREAD`` set, in the authority's process as in each
engine's, the channels whose handlers offer a lookup are offering. `engine(config)`
has the connector factory build ``lmcache_mp`` as Compass's
(`atom.compass.kv.lmcache_mp`), whose scheduler half sends each transfer to LP
``lmcache`` from the step loop.

A deployment that offloads KV to a shared Mooncake Store (``mooncake_store``),
the prefill side of a prefill-decode run or the run's one deployment, needs a
``mooncake_store`` entry (`atom.compass.kv.mooncake_store_server`), and a run
file with one needs such a deployment. Each such frontend checks its
deployment against the entry, and the first prefill deployment's, or the one
deployment's, starts LP ``mooncake_store`` (`mooncake_store`) in a process of
its own. Its channels to each DP rank of each engine it serves take half the
entry's fixed lookup term, and each such engine's request channels are
offering, as its input thread offers an arrival's lookup
(`add_mooncake_store`); the run file alone sets both, so the authority's
process marks them as each engine's does. Each such engine has the
connector factory build ``mooncake_store`` as Compass's
(`atom.compass.kv.mooncake_store`), whose scheduler half sends the Store's
lookups and renewals to LP ``mooncake_store`` and prices its copies on each
rank's lanes, and a DP member of one parks before its forward, as one
offloading to ``lmcache_mp`` does.

A deployment that offloads KV through ATOM's in-process ``lmcache_offload``
needs an ``lmcache_offload`` entry, which `atom.compass.kv.lmcache_offload`
states, and a run file with one needs such a deployment, at DP and pipeline
width 1. `engine(config)` then has the connector factory build it as Compass's,
whose scheduler half keeps each TP rank's CPU tier and prices the tier's
lookups and copies from the entry. Each such engine's hand-in holds
``offload``, and the summary carries it over the client run's steps, with the
GPU and CPU hit rates: prefix-cache hits and tier loads over the prompt tokens
the prefix cache counted.

An optional ``ingress`` states an encoding ingress in front of the run's
first server, or its router (`atom.compass.ingress`), which a ``host`` entry
may state as a term of its own: the traffic LP prices its hop on the HTTP
channel, so no server LP reads it and the channel table is the same.

The API server's ``--compass-clock-endpoint`` sets `CLOCK_ENV`: the authority
is not co-hosted but served by `authority`, its own process started before the
engines (``python -m atom.compass.run``), and every LP connects to that
endpoint. Its frontend is ``frontend-P`` or ``frontend-D`` by ``kv_role``. The
standalone authority writes the step table and the run summary once every
server LP has handed in its refusals, which the out_dir both API servers share
carries.

A multi-node DP group runs on a standalone authority too, named on every node.
The coordinator's API server, global DP rank 0's, is the one frontend LP; a
worker node's runs no frontend LP and serves no authority, and each of its
engine processes joins the engine LP over TCP as the member of its global rank.
Hand-ins go to the out_dir, which every node must see at one path: the
authority writes `AUTHORITY_FILE` there before it serves, and a node that does
not find it naming its endpoint's port is refused.
"""

import ast
import atexit
import codecs
import contextlib
import functools
import gc
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

import atom
from atom.compass.artifacts import Reading, provenance
from atom.compass.clock import (
    TAR,
    ClockAbort,
    ClockAuthority,
    LpId,
    RefusalTally,
    RunSummary,
    prefill_decode_table,
    single_engine_table,
)
from atom.compass.clock.channels import add_lmcache, add_mooncake_store
from atom.compass.clock_transport import connect, connect_here, serve
from atom.compass.detect.determinism import StepTable
from atom.compass.ingress import KEY as INGRESS
from atom.compass.ingress import check as check_ingress
from atom.compass.ingress import stated as stated_ingress
from atom.compass.runner import COMPASS_RAPIDSERVE_QUALNAME, COMPASS_RUNNER_QUALNAME
from atom.utils import clock, envs

ENV = "ATOM_COMPASS_RUN"
CLOCK_ENV = "ATOM_COMPASS_CLOCK_ENDPOINT"
#: The file an engine LP's clock is mapped from, for its model runner workers;
#: by its hand-in name in the out_dir.
NOW_ENV, NOW_FILE = "ATOM_COMPASS_NOW", "now-{}.bin"
FRONTEND, ENGINE, TRAFFIC = LpId("frontend"), LpId("engine"), LpId("traffic")
ROUTER, LMCACHE = LpId("router"), LpId("lmcache")
MOONCAKE_STORE = LpId("mooncake_store")
#: The run-file key, and the connector name, of a shared Mooncake Store.
ENTRY_STORE = MOONCAKE_STORE.name
#: Run-file keys for how many deployments of each prefill-decode side run.
INSTANCES = {"P": "prefill_instances", "D": "decode_instances"}
#: Run-file keys for each prefill-decode side's DP width.
DP_WIDTHS = {"P": "prefill_data_parallel_size", "D": "decode_data_parallel_size"}
#: The run-file key of a prefill-decode run's prefill pipeline width.
PREFILL_PP = "prefill_pipeline_parallel_size"
#: The run-file key that a prefill-decode run transfers its KV chunked
#: (Mooncake's ``enable_chunked_transfer`` on the prefill side, which
#: `deploy.sh run` sets from PREFILL_KV_EXTRA): `prefill_decode_table`'s
#: `chunked`.
KV_CHUNKED = "kv_chunked"
#: The run-file key that each stage of a pipelined prefill serves its own KV
#: writes (opt-in, `kv_serves`: the KV link states a stage's serving,
#: `kv.transfer.ServeModel`; `deploy.sh run` sets it from the network file):
#: `prefill_decode_table`'s `serve`.
KV_SERVE = "kv_serve"
#: The kv_transfer_config key a deployment names its index among them by.
INSTANCE_KEY = "compass_instance"
#: The run-file key of each side's worker order in the router, a per-run outcome.
WORKER_ORDER = "worker_order"
#: The run-file key that gives grants their ceilings.
LOCAL_GRANTS = "local_grants"
ATOM_RUNNER = "atom.model_engine.model_runner.ModelRunner"
RAPIDSERVE_RUNNER = "atom.model_engine.model_runner.RapidServeModelRunner"
#: The run-file key that says the deployment runs RapidServe.
RAPIDSERVE = "rapidserve"
#: Keys that differ between two runs of one configuration, kept out of its name.
#: The store is a place: the memory readings a run reads from it are keyed by
#: the configuration and the machine spec, which the name holds. Local grants
#: change how many calls a run makes, not its schedule.
PER_RUN = (
    "clock_endpoint",
    "out_dir",
    "client_runs",
    "router",
    "store",
    WORKER_ORDER,
    LOCAL_GRANTS,
)
LP_FILE, STEP_TABLE_FILE, SUMMARY_FILE = (
    "lp-{}.json",  # by LP name, or <lp>.<member> for a member
    "step_table.txt",
    "summary.json",
)
#: Each client run's hand-ins and summary, by its index.
CLIENT_DIR = "client-{}"
#: A real run's draft width and accepted-draft histogram, by DP rank.
ACCEPTANCE_FILE = "dp{}.acceptance.json"
#: The run-file key naming the record that gave the acceptance rates.
ACCEPTANCE_RECORD = "acceptance_record"
#: The endpoint a standalone authority serves at, written to its out_dir.
AUTHORITY_FILE = "authority_endpoint.txt"
#: Wall seconds the summary's writer waits for the server LPs' refusals.
HAND_IN_WAIT_S = 300.0
#: Wall seconds a prefill-decode engine, LP lmcache or LP mooncake_store waits
#: for every engine's planned block.
PLANNED_BLOCK_WAIT_S = 300.0
#: The format of the run summary; raise it when a field is added, changes
#: meaning or goes.
SUMMARY_SCHEMA_VERSION = 15
#: The reading kind of ATOM's source as `atom_digest` reads it.
ATOM_PY = "atom_py"
#: The terms of a ``host`` entry the frontend charges, each 0 s where it is absent.
HOST_TERMS = ("http_intake", "tokenizer")
#: A prefill-decode run's prefill host entry's term pricing the relay of its
#: response through the router to decode (`relay`).
RELAY_TERM = "relay"
#: A streaming server's host entry's term pricing its API loop's work per SSE
#: event it sends (`http_stream`).
STREAM_TERM = "http_stream"
#: The kinds of call a ``tokenizer`` term prices, each 0 s where it is absent.
TOKENIZER_KINDS = ("encode", "encode_batch", "decode", "segment")
#: A ``host`` entry's ``hops``, each optional and the run-file key of its name:
#: the lookaheads of the traffic's request, an engine's ZMQ message in, its
#: output, the stream back, and a prefill-decode run's write request and router
#: forward. Such a run states its decode side's under `DECODE_HOPS`.
HOPS = ("admission_s", "ipc_s", "ipc_out_s", "stream_s", "kv_write_req_s", "router_s")
DECODE_HOPS = "decode_hops"
#: A hop's lookahead where the run file states none: not 0 s, which puts a
#: serving run on deadlock recovery.
HOP_FLOOR_S = 2.0**-14
#: Compass source the host-law key leaves out, by path under the package: all
#: of it (None), or the top-level names (def, class or assigned) only a simulated
#: run reaches, with the comments above each and the module docstring.
COMPASS_ONLY = {
    "compass": None,
    "utils/compass_loop.py": None,
    "utils/clock.py": {
        *("DIAG_S", "job", "current_job_time", "install", "Straggler", "_Wakeup"),
        *("LPRuntime", "UnsentRelayItem", "WrappedSocket", "WrappedPoller"),
        *("RelayQueue", "channel_of", "_of_kind", "socket_channels"),
        *("REPLY_WAIT_S", "_WaitingReplyQueue"),
        *("COLLECTIVES", "_on_host", "settle_before_collectives"),
    },
    "utils/zmq_shim.py": {"_Socket", "_ChannelSocket", "_Context"},
}
#: The module `pip install -e .` writes (setuptools_scm) and git ignores: no
#: commit holds it, so the host-law key leaves it out.
INSTALL_WRITTEN = Path("_version.py")
#: The run-file key of each backend a flag names (`atom.compass.drive.runfile`).
STEP_STUB, STEP_LINEAR, STEP_GRID = "step_stub", "step_linear", "step_grid"
OP_GRAPH, PRICE_LIST = "op_graph", "price_list"
#: The run-file key of ATOM's arguments an op graph run's server was started with.
ATOM_ARGS = "atom_args"
#: The run-file key of the frontend's seconds to prepare one image of a chat request.
IMAGE_PREPROCESS = "image_preprocess_s"
#: Run-file keys that were renamed, to their names now.
RENAMED_KEYS = {
    "coefficients": STEP_STUB,
    "law": STEP_LINEAR,
    "step_fit": STEP_GRID,
    "roofline_floor_s": "step_roofline",
}


def refuse_renamed_keys(run: dict, where) -> None:
    """Refuse a run file that holds a renamed key, naming the key's name now."""
    if old := [k for k in RENAMED_KEYS if k in run]:
        named = ", ".join(f"{k} (now {RENAMED_KEYS[k]})" for k in old)
        raise ValueError(f"{where} holds run-file keys that were renamed: {named}")


def entries(run: dict, key: str) -> list:
    """The run file's `key` entries: its list, or its one mapping."""
    value = run.get(key, [])
    return [value] if isinstance(value, dict) else value


def _differ(*mappings) -> list[str]:
    """The fields `mappings` differ in, a nested mapping's as `key.field`."""
    flat = [
        {
            f"{k}.{s}" if s else k: w
            for k, v in (m or {}).items()
            for s, w in (v.items() if isinstance(v, dict) else [("", v)])
        }
        for m in mappings
    ]
    return sorted(
        {n for f in flat for n in f if any(g.get(n) != f.get(n) for g in flat)}
    )


def _fitted_here(listed: list, here: dict, engine, what: str) -> dict:
    """The entry of `listed` whose engine config, read by `engine`, is `here`
    at its ladder (`at_ladder`); a lone entry is checked as a mapping is."""
    from atom.compass.backends.step_grid_plan import at_ladder

    at = [at_ladder(here, engine(e)) for e in listed]
    found = [e for e, h in zip(listed, at) if engine(e) == h]
    if len(listed) == 1 or len(found) == 1:
        return (found or listed)[0]
    if found:
        raise ValueError(
            f"{len(found)} {what} entries were fitted under this engine config, "
            f"and they differ in {_differ(*found)}"
        )
    named = "; ".join(
        f"entry {i} in {_differ(engine(e), h)}"
        for i, (e, h) in enumerate(zip(listed, at))
    )
    raise ValueError(f"no {what} entry was fitted under this engine config: {named}")


def pick_fitted(run: dict, name: str, config) -> None:
    """Replace the run file's `name` entries, a step linear law's or host
    laws', when a list, by the one fitted under `config`'s engine config
    (`_fitted_here`), host laws' read less the graph sets their key states."""
    if not isinstance(run.get(name), list):
        return
    from atom.compass.backends.launch_features import GRAPHS
    from atom.compass.run.step_grid import keyed

    engines = {
        STEP_LINEAR: lambda e: e.get("engine_config"),
        "host": lambda e: {
            k: v
            for k, v in (e.get("key", {}).get("engine") or {}).items()
            if k != GRAPHS
        },
    }
    run[name] = _fitted_here(run[name], keyed(config), engines[name], name)


def refuse_unfitted_law(run: dict, config) -> None:
    """Refuse the run file's step linear law where `config` is not what it was
    fitted under: its engine config, at its ladder where this run's is a prefix
    of it (`at_ladder`), which states its tensor-parallel and decode
    context-parallel widths (`keyed`), prefill context-parallel width 1, a
    data-parallel group on one node, and its platform, which the run file's
    machine spec must state."""
    from atom.compass.artifacts import Kind, keyed_platform, machine_platform
    from atom.compass.backends.step_grid_plan import at_ladder
    from atom.compass.run.step_grid import differ, keyed

    law = run[STEP_LINEAR]
    if "engine_config" not in law:
        raise ValueError(
            "the step linear law states no engine_config, the engine config "
            "it was fitted under, so no run can check its parallel layout"
        )
    fitted = law["engine_config"]
    if named := differ(fitted, at_ladder(keyed(config), fitted), "the law"):
        raise ValueError(
            "the step linear law was fitted under another engine config: "
            + "; ".join(named)
        )
    refuse_context_parallel(STEP_LINEAR, config, PREFILL_CONTEXT_PARALLEL)
    # A node running fewer ranks than the group holds is one of several, and
    # ATOM hands MoRI its ranks times the TP width as the GPUs per node
    # (`moe.py::FusedMoEParallelConfig.make`).
    dp = config.parallel_config.data_parallel_size
    local = getattr(config.parallel_config, "data_parallel_size_local", dp)
    if local < dp:
        raise ValueError(
            "the step linear law states no data_parallel_size_local, so it "
            "prices a data-parallel group on one node, and this run's group "
            f"spans nodes: data_parallel_size_local {local} of "
            f"data_parallel_size {dp}"
        )
    fitted_on = law.get("platform") or {}
    here = keyed_platform(Kind.STEP_LINEAR_LAW, machine_platform(_machine(run)))
    if named := [
        f"{f} {fitted_on.get(f)!r} in the law and {v!r} here"
        for f, v in here.items()
        if fitted_on.get(f) != v
    ]:
        raise ValueError(
            "the step linear law was fitted on another platform: "
            + "; ".join(named)
            + "; a law measured on one platform is no evidence about another"
        )


def refuse_other_buckets(config, buckets: tuple[int, ...], run=None) -> None:
    """Refuse the run file's step linear law on a runner that records the
    prefill token-graph buckets `buckets`, ascending, where the law states none
    or others: in its prefill vector, those its records' runner recorded, or
    else in its engine config, which `refuse_unfitted_law` held to `config`'s.
    An engine config states them only off ATOM's default, which can move
    between a fit and a run, so one stating none says nothing of them; one
    stating them states ATOM's list, which a runner records only up to
    `warmup_max_tokens`, a bound no engine config keys, so a prefill vector's
    are held to `buckets` either way. A law with no prefill vector is kept:
    the buckets pad prefill steps alone, and it prices none (`StepLinearBackend`
    refuses each where it meets one), so no bucket set misprices a step of
    it. A prefill-decode deployment's decode side fits one: its records run no
    prefill step a client run times, yet its runner records ATOM's buckets up
    to its batch budget (#3720 cell B, 1P1D: [256, 512, 1024] on D). Host laws
    refuse where their key's graph sets state others; `run` replaces `spec()`."""
    if not buckets or (run := spec() if run is None else run) is None:
        return
    from atom.compass.backends import launch_features
    from atom.compass.run.step_grid import BUCKETS

    pick_fitted(run, "host", config)
    host = run.get("host") or {}
    graphs = host.get("laws") and launch_features.stated(host.get("key"))[1]
    if graphs and (theirs := graphs["token"]["prefill"]) != list(buckets):
        raise ValueError(
            f"the host laws' key states the prefill token-graph buckets {theirs}, "
            f"and this run's runner records {list(buckets)}"
        )
    if STEP_LINEAR not in run:
        return
    pick_fitted(run, STEP_LINEAR, config)
    law = run[STEP_LINEAR]
    if not law.get("prefill"):
        return
    stated = sorted(law["prefill"].get("buckets") or ())
    env = law["engine_config"]["env"]
    if (stated or BUCKETS not in env) and stated != list(buckets):
        said = f"{BUCKETS}={env[BUCKETS]}" if BUCKETS in env else f"no {BUCKETS}"
        raise ValueError(
            f"the step linear law states {said} in its engine config and "
            f"{stated or 'none'} as its prefill vector's buckets, and this run's "
            f"runner records the prefill token-graph buckets {list(buckets)}"
        )


#: The context-parallel widths, each a config field.
CONTEXT_PARALLEL = ("prefill_context_parallel_size", "decode_context_parallel_size")
#: Those a step linear law's engine config cannot state: the decode width it
#: keys (`step_grid.keyed`), so `refuse_unfitted_law` holds it to this run's.
PREFILL_CONTEXT_PARALLEL = CONTEXT_PARALLEL[:1]


def refuse_context_parallel(name: str, config, widths=CONTEXT_PARALLEL) -> None:
    """Refuse a `config` above width 1 in any of the context-parallel `widths`
    for the run file's `name`, a law or a bundle, whose engine config states
    none of them: a bundle states neither, a law no prefill width."""
    cp = {
        width: getattr(config, width) for width in widths if getattr(config, width) != 1
    }
    if cp:
        raise ValueError(
            f"the run file's {name} is measured at context-parallel width 1 and "
            "states no context-parallel width, and this run has "
            + " and ".join(f"{width} {n}" for width, n in cp.items())
        )


def refuse_unfitted_host(host: dict, config, platform: dict) -> None:
    """Refuse the run file's host laws `host` when fitted under another host
    key than this run's, on `platform` (`machine_platform`), the host CPU and
    graph sets their key states taken as this run's, which records none, and
    its engine config read at their ladder where this run's is a prefix of it."""
    from atom.compass.artifacts import Kind, keyed_platform
    from atom.compass.backends.launch_features import GRAPHS, stated
    from atom.compass.backends.step_grid_plan import at_ladder

    ours = host.get("laws") and {
        **host_key(config),
        **keyed_platform(Kind.HOST_LAWS, platform),
    }
    theirs = host.get("key") or {}
    launch = stated(theirs)
    if ours and "host_cpu" in theirs:
        # The host they timed is the host the run predicts, from any CPU.
        ours["host_cpu"] = theirs["host_cpu"]
    if ours:
        ours["engine"] = at_ladder(ours["engine"], theirs.get("engine"))
    if ours and launch[1] is not None:
        ours["engine"] = {**ours["engine"], GRAPHS: launch[1]}
    if ours and theirs != ours:
        differ = [k for k in ours if theirs.get(k) != ours[k]]
        if isinstance(engine := theirs.get("engine"), dict):
            mine = ours["engine"]
            differ = [k for k in differ if k != "engine"] + [
                f"engine.{n}"
                for n in sorted(mine.keys() | engine.keys())
                if engine.get(n) != mine.get(n)
            ]
        raise ValueError(
            f"the run file's host laws were fitted under another {differ}; "
            f"this run's key is {ours}, and theirs is {theirs}"
        )


_authority: "_RecordingAuthority | None" = None
#: The co-hosted authority's early-end writer.
_early_end: threading.Thread | None = None
#: This engine process's member name in the engine LP, None at DP width one.
_member: str | None = None
#: The SIGTERM handler this engine process set aside as it said READY.
_sigterm_at_ready: list = []
#: The files a recorded real run's frontend writes beside its step record:
#: each served tokenizer call, and each HTTP request's intake stamps.
TOKENIZER_RECORD, INTAKE_RECORD = "tokenizer.jsonl", "intake.stamps"
#: Beside one of those, ``<name>.local`` names the file its lines are in until
#: the server's stop moves them there (`_LineFile`).
LOCAL = ".local"
#: This server's ``host`` entry among several, which `tokenizer` resolves.
_frontend_host: dict | None = None


def spec() -> dict | None:
    """The run file's mapping, or None on a real run."""
    path = os.environ.get(ENV)
    if not path:
        return None
    run = json.loads(Path(path).read_text())
    bound = run.get("bound_s")
    if not isinstance(bound, (int, float)) or not math.isfinite(bound):
        raise ValueError(
            f"{path}: bound_s is {bound!r}; a run needs a finite simulated-time "
            "bound, or a timer nobody declared daemon keeps it alive forever"
        )
    if run.get("on_refusal", "mark") not in ("mark", "abort"):
        raise ValueError(
            f"{path}: on_refusal is {run['on_refusal']!r}; a step no cost source "
            "answers is marked and the run continues (mark), or ends it (abort)"
        )
    count = run.get("client_runs", 1)
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError(
            f"{path}: client_runs is {count!r}; a server serves a positive number "
            "of client runs, or null for no limit"
        )
    for host in entries(run, "host"):
        _check_host(path, host)
    stated_ingress(run)
    run["clock_endpoint"] = os.environ.get(CLOCK_ENV, run.get("clock_endpoint"))
    return run


def _check_host(path, host: dict) -> None:
    """Refuse a ``host`` entry the runner would read as zero host time, or
    whose ``tokenizer`` or ``http_intake`` term the frontend would misread."""
    from atom.compass.backends.launch_features import PATHS
    from atom.compass.backends.step_labels import FIRST_LAUNCH_KEY, STALLS
    from atom.compass.runner.overrides import ADMISSION_IN_E, HOST_LAW_KINDS, SEGMENTS

    laws, columns = host.get("laws", {}), host.get("columns", {})
    paths, stalls = host.get(PATHS, {}), host.get(STALLS, {})
    if (
        set(host)
        - {
            "laws",
            "skip",
            "key",
            "admit",
            "hops",
            "columns",
            PATHS,
            STALLS,
            *HOST_TERMS,
            INGRESS,
            RELAY_TERM,
            STREAM_TERM,
        }
        or set(host.get("skip", ())) - set(SEGMENTS)
        or {*laws, *columns, *paths} - set(HOST_LAW_KINDS)
        or {s for v in (*laws.values(), *columns.values(), *paths.values()) for s in v}
        - set(SEGMENTS)
        or not all(
            s in laws.get(k, {})
            and isinstance(p, dict)
            and all(isinstance(c, (int, float)) and c >= 0 for c in p.values())
            for k, v in paths.items()
            for s, p in v.items()
        )
        or {len(c) for v in laws.values() for c in v.values()} - {3}
        or len(host.get("admit") or ADMISSION_IN_E) != 3
        or not all(
            isinstance(v, dict)
            and set(v) == {"first_s", "key"}
            and v["key"] == FIRST_LAUNCH_KEY
            and isinstance(v["first_s"], (int, float))
            and v["first_s"] >= 0
            for v in stalls.values()
        )
    ):
        raise ValueError(
            f"{path}: host is {host!r}; it takes skip, a list of {SEGMENTS}, "
            f"laws, mapping {HOST_LAW_KINDS} to segments to three coefficients, "
            "columns, mapping them to segments to launch features' coefficients, "
            f"{PATHS}, mapping them to segments their laws state to launch paths' "
            f"non-negative constants, {STALLS}, mapping launch paths to the "
            "non-negative first_s a first launch of a shape costs and the key "
            f"{FIRST_LAUNCH_KEY} it was counted on, "
            "admit, three coefficients on a forward's admissions or null, "
            "key, the host key the laws were fitted under, hops, the terms "
            f"{' and '.join(HOST_TERMS)}, {RELAY_TERM}, {STREAM_TERM}, and {INGRESS}, "
            "the encoding ingress in front of the deployment (atom.compass.ingress)"
        )
    if INGRESS in host:
        check_ingress(host[INGRESS], f"{path}: host's {INGRESS}")
    hops = host.get("hops", {})
    if not (_seconds(hops, set(), set(HOPS)) and all(v > 0 for v in hops.values())):
        raise ValueError(
            f"{path}: host's hops is {hops!r}; it takes {', '.join(HOPS)}, each "
            "optional and a positive number of seconds"
        )
    tok, intake = host.get("tokenizer"), host.get("http_intake")
    if tok is not None and not (
        isinstance(tok, dict)
        and {"backend", "fingerprint"}
        <= tok.keys()
        <= {"backend", "fingerprint", *TOKENIZER_KINDS}
        and tok["backend"] in ("fast", "slow")
        and (
            isinstance(tok["fingerprint"], str)
            or (tok["fingerprint"] is None and tok["backend"] == "slow")
        )
        and all(_rate(tok[k], k) for k in TOKENIZER_KINDS if k in tok)
    ):
        raise ValueError(
            f"{path}: host's tokenizer is {tok!r}; it takes backend, fast or slow, "
            "fingerprint, the tokenizer.json it was fitted on (null for a slow "
            "tokenizer, which loads none), and, each "
            "optional, encode, encode_batch, decode and segment, each fixed_s and a "
            "positive tokens_per_s, a segment optionally a positive chars_per_s "
            "too, a decode fixed_s and tokens, [lo, hi], in their place"
        )
    hull = intake.get("bytes", [0, 0]) if isinstance(intake, dict) else None
    if intake is not None and not (
        _seconds(
            {k: v for k, v in intake.items() if k != "bytes"},
            {"fixed_s", "per_byte_s"},
            {"idle_s"},
        )
        and isinstance(hull, list)
        and _seconds(dict(enumerate(hull)), {0, 1})
        and hull[0] <= hull[1]
    ):
        raise ValueError(
            f"{path}: host's http_intake is {intake!r}; it takes fixed_s and "
            "per_byte_s, and idle_s optional, each a non-negative number, and "
            "bytes optional, the body bytes it was fitted over, [lo, hi]"
        )
    term = host.get(RELAY_TERM)
    hull = term.get("bytes", [0, 0]) if isinstance(term, dict) else None
    if term is not None and not (
        _seconds(
            {k: v for k, v in term.items() if k != "bytes"},
            {"fixed_s", "per_byte_s"},
        )
        and isinstance(hull, list)
        and _seconds(dict(enumerate(hull)), {0, 1})
        and hull[0] <= hull[1]
    ):
        raise ValueError(
            f"{path}: host's relay is {term!r}; it takes fixed_s and per_byte_s, "
            "each a non-negative number, and bytes optional, the bytes of the "
            "requests it was fitted over, [lo, hi]"
        )
    term = host.get(STREAM_TERM)
    if term is not None and not _seconds(term, {"per_event_s"}):
        raise ValueError(
            f"{path}: host's {STREAM_TERM} is {term!r}; it takes per_event_s, a "
            "non-negative number"
        )


def _rate(term, kind: str) -> bool:
    """Whether `term` is a valid ``tokenizer`` term's `kind` (`_check_host`)."""
    if kind == "decode" and isinstance(term, dict) and "tokens" in term:
        hull = term["tokens"]
        return (
            _seconds({k: v for k, v in term.items() if k != "tokens"}, {"fixed_s"})
            and isinstance(hull, list)
            and _seconds(dict(enumerate(hull)), {0, 1})
            and hull[0] <= hull[1]
        )
    search = {"chars_per_s"} if kind == "segment" else set()
    return _seconds(term, {"fixed_s", "tokens_per_s"}, search) and all(
        v > 0 for k, v in term.items() if k != "fixed_s"
    )


def _seconds(term, required: set, optional: set = frozenset()) -> bool:
    """Whether `term` is a mapping of the `required` fields and any `optional`
    ones, each a finite non-negative number."""
    return (
        isinstance(term, dict)
        and required <= term.keys() <= required | optional
        and all(
            type(v) in (int, float) and math.isfinite(v) and v >= 0
            for v in term.values()
        )
    )


def _dp(run: dict, side: str | None = None) -> int:
    """The DP width of `side`'s deployments, ``P`` or ``D``, else the run's."""
    width = run.get("data_parallel_size", 1)
    return run.get(DP_WIDTHS[side], width) if side else width


def worker_order(run: dict) -> dict:
    """The run file's ``worker_order``, ``{}`` when absent. Refuses a side
    other than ``P`` and ``D``, and a list that is not each of the side's
    (instance, DP rank) pairs once."""
    order = run.get(WORKER_ORDER, {})
    for side, pairs in order.items():
        known = side in INSTANCES
        n, dp = (run.get(INSTANCES[side], 1), _dp(run, side)) if known else (0, 0)
        if not n or sorted(pairs) != [[k, r] for k in range(n) for r in range(dp)]:
            raise ValueError(
                f"{WORKER_ORDER}[{side!r}] is {pairs}: name side P or D, and list "
                "each of its deployments' (instance, DP rank) pairs once"
            )
    return order


def _widths(run: dict) -> int | tuple[int, int]:
    """The run's DP width, or a prefill-decode run's (prefill, decode) widths."""
    if "router_s" in run:
        return _dp(run, "P"), _dp(run, "D")
    if named := [k for k in DP_WIDTHS.values() if k in run]:
        raise ValueError(
            f"the run file names {', '.join(named)}, and only a prefill-decode "
            "run has a prefill and a decode side"
        )
    return _dp(run)


def _pp(run: dict, side: str | None = None) -> int:
    """The pipeline width of `side`'s deployments, ``P`` or ``D``, else the
    run's. A prefill-decode run states its prefill side's as `PREFILL_PP`, and
    its decode side runs one stage."""
    pd = "router_s" in run
    if (named := "pipeline_parallel_size" if pd else PREFILL_PP) in run:
        raise ValueError(
            f"the run file names {named}: a prefill-decode run states its prefill "
            f"side's pipeline width as {PREFILL_PP}, and any other run its own as "
            "pipeline_parallel_size"
        )
    if not pd:
        return run.get("pipeline_parallel_size", 1)
    return run.get(PREFILL_PP, 1) if side == "P" else 1


def channel_table(run: dict):
    from atom.compass.kv import mooncake_store_server as store

    pd = "router_s" in run
    if "lmcache" in run and _pp(run, "P") > 1:
        raise ValueError(
            "the run file's lmcache entry and a pipeline above one stage: ATOM's "
            "lmcache_mp does not run under PP"
        )
    table = _engines_table(run)
    if "lmcache" in run:
        add_lmcache(
            table,
            _lmcache_engine(run),
            run.get("lmcache_ipc_s", hop(run, "ipc_s")),
            _dp(run, "P" if pd else None),
            envs.OFFLOAD_MP_LOOKUP_THREAD,
        )
    if store.ENTRY in run:
        # Every engine LP of the prefill side, or the run's one engine; of a
        # pipeline, its head stage, the one that schedules.
        names = [lp.name for lp in table.registry.ids()]
        engines = [n for n in names if re.fullmatch(r"engine(-P\d*)?(-pp0)?", n)]
        store.check_entry(run[store.ENTRY])
        rpc_s = store.store_rpc_s(run[store.ENTRY])
        add_mooncake_store(table, engines, rpc_s, _dp(run, "P" if pd else None))
    return table


def _lmcache_engine(run: dict) -> str:
    """The engine LP that LP `lmcache` serves: a prefill-decode run's prefill
    side, else the run's one engine."""
    return "engine-P" if "router_s" in run else ENGINE.name


def _engines_table(run: dict):
    common = {k: hop(run, k) for k in HOPS[:3]}
    pp, links = _pp(run, "P"), ()
    if pp > 1:
        from atom.compass.runner.pp_send import carrier_models, lookaheads

        links = lookaheads(carrier_models(_machine(run)))
    if "router_s" in run:
        return prefill_decode_table(
            **common,
            stream_s=hop(run, f"{DECODE_HOPS}.stream_s"),
            decode_ipc=tuple(hop(run, f"{DECODE_HOPS}.{k}") for k in HOPS[1:3]),
            router_s=hop(run, "router_s"),
            kv_write_req_s=hop(run, "kv_write_req_s"),
            dp=_widths(run),
            router_in_s=run.get("router_in_s"),
            prefill=run.get(INSTANCES["P"], 1),
            decode=run.get(INSTANCES["D"], 1),
            prefill_pp=pp,
            pp_links=links,
            chunked=run.get(KV_CHUNKED, False),
            serve=run.get(KV_SERVE, False),
        )
    return single_engine_table(
        **common,
        stream_s=hop(run, "stream_s"),
        dp=_widths(run),
        pp=pp,
        pp_links=links,
        rapidserve=run.get(RAPIDSERVE, False),
    )


def hops(run: dict) -> dict:
    """Each hop `run`'s channel table reads, by name: a prefill-decode run's
    prefill side's, and its decode side's ``ipc_s``, ``ipc_out_s`` and
    ``stream_s`` as ``decode_hops.<name>``; the run file's value, None where it
    states none."""
    if "router_s" not in run:
        return {k: run.get(k) for k in HOPS[:4]}
    decode = run.get(DECODE_HOPS, {})
    return {k: run.get(k) for k in HOPS if k != "stream_s"} | {
        f"{DECODE_HOPS}.{k}": decode.get(k) for k in HOPS[1:4]
    }


def hop(run: dict, name: str) -> float:
    """The hop `name` of `hops`, `HOP_FLOOR_S` where the run file states none."""
    value = hops(run)[name]
    return HOP_FLOOR_S if value is None else value


def _machine(run: dict):
    from atom.compass.spec import MachineSpec

    return MachineSpec.from_mapping(run["machine"])


def kv_serves(run: dict) -> bool:
    """Whether a prefill-decode run's KV link states a stage's serving
    (`kv.transfer.ServeModel`), which `KV_SERVE` then turns on; refused on a
    prefill side of one stage, whose head serves every layer's writes."""
    from atom.compass.kv.transfer import Scope, ServeModel

    if "router_s" not in run or "kv_link" not in run:
        return False
    if ServeModel.from_spec(_machine(run), Scope(run["kv_link"])) is None:
        return False
    if _pp(run, "P") < 2:
        raise ValueError(
            "the KV link states a stage's serving and the prefill side runs one "
            "stage: a stage's serving prices a pipelined prefill's writes"
        )
    return True


def _members(table, dp: int | tuple[int, int]) -> dict | None:
    """Each engine LP's members at its DP width above one: rank ``dp<r>`` owns
    the channels whose ``#`` tag names it at that LP's end, ``#dp<r>`` or, on a
    channel between two engine LPs, ``#dp<sender rank>.dp<receiver rank>``. A
    tag's parts other than ``dp<r>`` name no member.
    `dp` is every engine LP's width, or ``(prefill, decode)`` by side."""

    def rank(c, lp):
        tags = [t for t in c.name.rpartition("#")[2].split(".") if t.startswith("dp")]
        return tags[0] if c.source == lp else tags[-1]

    def width(lp):
        if isinstance(dp, int):
            return dp
        return dp[1] if lp.name.startswith(f"{ENGINE.name}-D") else dp[0]

    members = {
        lp: {
            f"dp{r}": [
                c.name
                for c in table.channels_into(lp) + table.channels_from(lp)
                if rank(c, lp) == f"dp{r}"
            ]
            for r in range(width(lp))
        }
        for lp in table.registry.ids()
        if lp.name.startswith(ENGINE.name) and width(lp) > 1
    }
    return members or None


def _pd_kv(config) -> dict | None:
    """The simulated KV connector's ``kv_transfer_config``, alone or one
    connector of a ``multi``, else None."""
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    leaves = KVConnectorFactory.leaf_connectors(config) or ()
    return next((c.kv_transfer_config for n, c in leaves if n == "compass"), None)


def _pd_side(config) -> str | None:
    """``P`` or ``D`` for a deployment on the simulated KV connector, else None."""
    kv = _pd_kv(config)
    if kv is None:
        return None
    return "P" if kv.get("kv_role", "kv_producer") == "kv_producer" else "D"


def _pd_name(config, run: dict) -> str | None:
    """`_pd_side`, followed above one deployment of that side by the index
    ``compass_instance`` in its ``kv_transfer_config`` names."""
    side = _pd_side(config)
    if side is None:
        return None
    n, k = run.get(INSTANCES[side], 1), _pd_kv(config).get(INSTANCE_KEY)
    if n == 1 and k is None:
        return side
    if type(k) is not int or not 0 <= k < n:
        raise ValueError(
            f"kv_transfer_config's {INSTANCE_KEY} is {k!r} and the run file "
            f"declares {n} as {INSTANCES[side]}; each deployment names its own "
            "index, from 0"
        )
    return side if n == 1 else f"{side}{k}"


def _frontend_lp(config, run: dict) -> LpId:
    """The LP of this deployment's frontend: ``frontend-<name>`` for
    `_pd_name`'s name, else ``frontend``."""
    name = _pd_name(config, run)
    return FRONTEND if name is None else LpId(f"{FRONTEND.name}-{name}")


def _engine_name(config, run: dict) -> str | None:
    """The suffix of this engine's LP: RapidServe's ``P`` or ``D`` engine,
    else `_pd_name`."""
    if getattr(config, "enable_rapidserve", False):
        return "D" if config.disagg_is_decode else "P"
    return _pd_name(config, run)


def _bind_transfer(config, run: dict) -> None:
    """Have the simulated connector price its transfers from the machine spec
    and the paged KV blocks and state slots the run's engines planned, read as
    it is built (`_planned_transfer`): after this engine's runner has planned
    its pool."""
    from atom.compass.kv import TRANSFER_KEY

    _pd_kv(config)[TRANSFER_KEY] = _planned_transfer


def publish_transfer(
    path: Path,
    block_bytes: int,
    tp: int,
    slot_bytes: int = 0,
    staging_bytes: int = 0,
    slot_after_blocks: bool = False,
    image_bytes: int = 0,
) -> None:
    """Write one paged KV block's bytes, the state slot a transfer moves beside
    a request's blocks, or after them, the staging row it moves in a second
    write, and the state image an lmcache save carries, as a runner at TP `tp`
    planned them, where `_planned_transfer` and `_lmcache_plan` read them."""
    planned = {
        "block_bytes": int(block_bytes),
        "tp": tp,
        "slot_bytes": int(slot_bytes),
        "staging_bytes": int(staging_bytes),
        "slot_after_blocks": bool(slot_after_blocks),
        "image_bytes": int(image_bytes),
    }
    _write_whole(path, json.dumps(planned))


def _published_plans(run: dict, names: list[str]) -> dict:
    """By engine name, what each engine of `names` published
    (`publish_transfer`), once every one has; refused, naming each that has
    not, after `PLANNED_BLOCK_WAIT_S` wall seconds."""
    out = Path(run["out_dir"])
    files = {n: _block_file(out / NOW_FILE.format(n)) for n in names}
    deadline = time.monotonic() + PLANNED_BLOCK_WAIT_S
    while missing := [n for n, f in files.items() if not f.exists()]:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"{', '.join(missing)} published no planned KV block in "
                f"{PLANNED_BLOCK_WAIT_S} wall seconds"
            )
        time.sleep(0.1)
    return {n: json.loads(f.read_text()) for n, f in files.items()}


def _planned_transfer() -> tuple:
    """One transfer model per prefill stage, each pricing the block, the state
    slot and the staging row that stage's runner planned, once every engine of
    the prefill-decode run, each member and stage, has published them
    (`publish_transfer`).

    Refused unless every deployment plans one block, one slot and one row,
    each summed over its stages, and every prefill deployment splits them
    alike: both ends of a write price it, each from its own copy."""
    import dataclasses

    from atom.compass.kv import Scope, TransferModel

    run = spec()
    if not _block_file(Path(os.environ[NOW_ENV])).exists():
        raise RuntimeError("this engine's runner published no planned KV block")
    engines = [n for n in _hand_ins(run) if n.startswith(f"{ENGINE.name}-")]
    # Each deployment member's stages: engine-P1-pp0.dp2 is stage 0 of
    # engine-P1.dp2.
    planned: dict = {}
    for n, published in _published_plans(run, engines).items():
        lp, dot, member = n.partition(".")
        deployment, _, stage = lp.partition("-pp")
        stages = planned.setdefault(deployment + dot + member, {})
        stages[int(stage or 0)] = published
    prefill = [u for u in planned if u.startswith(f"{ENGINE.name}-P")]
    landing = 1.0
    for key, what in (
        ("block_bytes", "KV block"),
        ("slot_bytes", "state slot"),
        ("staging_bytes", "staging row"),
    ):
        sizes = {u: [s[k][key] for k in sorted(s)] for u, s in planned.items()}
        first = sizes[prefill[0]]
        for u, b in sizes.items():
            tp_u, tp_p = planned[u][0]["tp"], planned[prefill[0]][0]["tp"]
            if (
                key == "block_bytes"
                and u not in prefill
                and tp_u > tp_p
                and tp_u * sum(b) >= sum(first) > sum(b)
            ):
                # A decode at a wider TP than prefill's (the 1P2D recipe's TP1
                # PP4 prefill landing on TP4 decodes, ATOM_PD_MINIMAX_LANDING):
                # each decode rank takes its own block, its head slice of each
                # sharded region and each replicated region whole, so the wire
                # carries the decode's TP times its block per prefill block,
                # each prefill stage its share of it.
                scale = tp_u * sum(b) / sum(first)
                if landing not in (1.0, scale):
                    raise ValueError(
                        f"{u} lands a prefill block at {scale:.4f} its bytes and "
                        f"another decode at {landing:.4f}: one prefill prices one "
                        "write"
                    )
                landing = scale
                continue
            if b != first if u in prefill else sum(b) != sum(first):
                (a, x), (c, y) = (
                    (f"{v} at TP {planned[v][0]['tp']}", " + ".join(map(str, sizes[v])))
                    for v in (u, prefill[0])
                )
                raise ValueError(
                    f"{a} plans a {what} of {x} B and {c} one of {y} B: both ends "
                    f"of a transfer price one {what.split()[-1]}, so every "
                    "deployment must plan it"
                )
    machine, scope = _machine(run), Scope(run["kv_link"])
    stages = planned[prefill[0]]
    models = tuple(
        TransferModel.from_spec(
            machine,
            round(s["block_bytes"] * landing),
            scope,
            s["slot_bytes"],
            s["staging_bytes"],
            s["slot_after_blocks"],
        )
        for _, s in sorted(stages.items())
    )
    if (models[0].serve is not None) != bool(run.get(KV_SERVE, False)):
        raise ValueError(
            f"the KV link {'states' if models[0].serve else 'states no'} stage "
            f"serving and the run file's {KV_SERVE} is "
            f"{run.get(KV_SERVE, False)!r}: deploy.sh sets one from the other"
        )
    if models[0].serve is None:
        return models
    # A stage writes one write a decode rank (Mooncake's TP landing): the first
    # decode deployment's TP.
    decode = [u for u in planned if u not in prefill]
    writes = planned[decode[0]][0]["tp"] if decode else 1
    return tuple(
        dataclasses.replace(m, serve=dataclasses.replace(m.serve, writes=writes))
        for m in models
    )


def _lmcache_stations(run: dict) -> dict:
    """LP `lmcache`'s station laws, from the run file's ``lmcache`` entry, and
    the link its copy engine prices a chunk on, the machine spec's host link,
    as ``(latency, bandwidth)``."""
    entry = run["lmcache"]
    if missing := {"server", "lookup_s", "request_s"} - set(entry):
        raise ValueError(
            f"the run file's lmcache entry states no {', '.join(sorted(missing))}: "
            "the lmcache server flags, and its request loop's laws"
        )
    machine = _machine(run)
    return {
        "lookup_s": tuple(entry["lookup_s"]),
        "request_s": entry["request_s"],
        "link": (
            machine.value("interconnect.host_link.latency_s"),
            machine.value("interconnect.host_link.bandwidth_bytes_per_s"),
        ),
        "copy_width": entry.get("copy_width", 1),
    }


def _lmcache_plan(run: dict, blocks: int) -> tuple[int, int]:
    """A chunk's bytes, `blocks` KV blocks, and a rank's state image, as every
    member of the engine LP `lmcache` serves planned them (`publish_transfer`).
    Refused, naming both, where two members plan different blocks or images."""
    served = _lmcache_engine(run)
    members = [n for n in _hand_ins(run) if n.partition(".")[0] == served]
    (first, plan), *rest = _published_plans(run, members).items()
    for name, other in rest:
        for key, what in (("block_bytes", "KV block"), ("image_bytes", "state image")):
            if other[key] != plan[key]:
                raise ValueError(
                    f"{name} plans a {what} of {other[key]} B and {first} one of "
                    f"{plan[key]} B: LP lmcache sizes every object it holds at "
                    "one chunk and one image"
                )
    return plan["block_bytes"] * blocks, plan["image_bytes"]


def _launch_lmcache(config, run: dict, side: str | None) -> None:
    """Start LP `lmcache`, the stand-in lmcache server, in a process of its own
    when this deployment offloads KV to one, given its KV block tokens; refuse
    a deployment and a run file that disagree on whether one runs, a decode
    side that would start it, an entry stating ``writers`` or ``state_bytes``,
    which nothing reads, or a server chunk that is not a whole number of the
    deployment's KV blocks."""
    from atom.compass.kv.lmcache_server import parse_server_flags
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    leaves = KVConnectorFactory.leaf_connectors(config) or ()
    lm = next((c for name, c in leaves if name == "lmcache_mp"), None)
    offloads = lm is not None
    if offloads != ("lmcache" in run) and (offloads or side != "D"):
        raise ValueError(
            f"this deployment {'offloads' if offloads else 'offloads no'} KV to an "
            f"lmcache server and the run file {'has no' if offloads else 'has an'} "
            "lmcache entry; LP lmcache runs exactly when the prefill or only "
            "deployment offloads"
        )
    if not offloads:
        return
    if side == "D":
        raise ValueError("LP lmcache serves the prefill side; this is the decode side")
    if retired := sorted({"writers", "state_bytes"} & run["lmcache"].keys()):
        raise ValueError(
            f"the run file's lmcache entry states {' and '.join(retired)}, which "
            "nothing reads: every writer rank of a DP rank sends on that rank's "
            "one transfer channel, and LP lmcache sizes a rank's state image from "
            "its engine's plan"
        )
    _lmcache_stations(run)  # refused here, before the run starts
    chunk = parse_server_flags(run["lmcache"]["server"]).chunk_size
    block = config.kv_cache_block_size
    if chunk % block:
        raise ValueError(
            f"the lmcache server's --chunk-size is {chunk} tokens, not a whole "
            f"number of this deployment's {block}-token KV blocks "
            "(kv_cache_block_size), which ATOM's lmcache_mp workers refuse"
        )
    _start_lp(LMCACHE_MAIN, str(block))


def _outlive_the_group_stop() -> None:
    """A shared server's LP ignores SIGTERM. `compass run` stops its server's
    whole process group (`drive/serve.py::_stop`), the LPs its frontend started
    among it, while the frontend's stop is what finishes the run: an LP that
    took the SIGTERM died before the ``+inf`` grant that ends it, its hand-in
    unwritten, and the run summary waited `HAND_IN_WAIT_S` for it and failed.
    The grant ends it, and the group's SIGKILL after the stop's grace one that
    never gets the grant."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _start_lp(main: str, *args: str) -> None:
    """Start a shared server's LP in a process of its own: `main` run with
    `args`, importing this tree's ATOM."""
    paths = [str(Path(atom.__file__).parents[1]), os.environ.get("PYTHONPATH")]
    subprocess.Popen(
        [sys.executable, "-c", main, *args],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, paths))},
    )


#: LP `lmcache`'s process, given the deployment's KV block tokens.
LMCACHE_MAIN = "import sys; from atom.compass import run; run.lmcache(int(sys.argv[1]))"


def lmcache(block_tokens: int) -> None:
    """LP `lmcache`: the stand-in server at its flags' URL, on the run's clock
    until its ``+inf`` grant, then its hand-in. At its first stamped request it
    sizes a chunk, its ``--chunk-size`` over `block_tokens` KV blocks, and a
    rank's state image from its engine's plan (`_lmcache_plan`), and its copy
    engine's model with them."""
    import zmq

    from atom.compass.kv import TransferModel
    from atom.compass.kv.lmcache_server import LMCacheLP, LMCacheServer

    _outlive_the_group_stop()
    run = spec()
    rt = _runtime(LMCACHE, run)
    rt.client_run_ended = lambda k: _hand_in(run, LMCACHE.name, rt.refusals, k)
    stations = _lmcache_stations(run)
    link = stations.pop("link")
    # Unsized until `size_copy`: the startup frames before it need no bytes.
    server = LMCacheServer(run["lmcache"]["server"], chunk_bytes=0)

    def size_copy():
        """Size the server's chunk and state image from the plan, and return
        the copy engine's model at that chunk."""
        blocks = server.chunk_size // block_tokens
        server.chunk_bytes, server.state_bytes = _lmcache_plan(run, blocks)
        return TransferModel(*link, server.chunk_bytes)

    sock = zmq.Context.instance().socket(zmq.ROUTER)
    # No high-water mark: past one, ROUTER drops a reply to an engine that lags
    # behind, whose future then never resolves, as PUB dropped the traffic LP's
    # cancels (compass_harness/traffic_lp.py). Set before the bind.
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.bind(server.url)
    ended = []
    try:
        LMCacheLP(rt, sock, server, size_copy=size_copy, **stations).serve()
    except BaseException as e:
        ended = [f"ended:{LMCACHE.name} raised {e!r}"]
        raise
    finally:
        _hand_in(run, LMCACHE.name, rt.refusals + ended)
        sock.close(linger=0)


def _launch_mooncake_store(config, run: dict, side: str | None) -> None:
    """Check a deployment that offloads KV to a shared Mooncake Store against
    the run file's entry, and on the first prefill deployment, or the run's
    one, start LP `mooncake_store` in a process of its own, given its KV block
    tokens. Refuse a deployment and a run file that disagree on whether the
    Store runs, a decode side or a pipeline that offloads to it, pools that
    address several masters, and a deployment whose Store settings or worker
    widths differ from those the entry was fitted at."""
    from atom.compass.kv import mooncake_store_server as store
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
    from atom.kv_transfer.offload.kv_store.pool import _SLOT_ALIGN, _round_up
    from atom.kv_transfer.offload.mooncake_store.config import (
        parse_mooncake_store_config,
    )
    from atom.utils import envs

    leaves = KVConnectorFactory.leaf_connectors(config) or ()
    sub = next((c for name, c in leaves if name == store.ENTRY), None)
    offloads = sub is not None
    if offloads != (store.ENTRY in run) and (offloads or side != "D"):
        raise ValueError(
            f"this deployment {'offloads' if offloads else 'offloads no'} KV to a "
            f"Mooncake Store and the run file {'has no' if offloads else 'has a'} "
            f"{store.ENTRY} entry; LP mooncake_store runs exactly when the "
            "prefill or only deployments offload to one"
        )
    if not offloads:
        return
    if side == "D":
        raise ValueError(
            "this decode side offloads KV to a Mooncake Store; LP mooncake_store "
            "serves the prefill side"
        )
    entry = run[store.ENTRY]
    store.check_entry(entry)
    cfg = parse_mooncake_store_config(sub.kv_transfer_config)
    masters = cfg.store_masters()
    pooled = isinstance(entry["capacity_bytes"], list)
    if len(masters) > 1 and not pooled:
        raise ValueError(
            f"mooncake_store.pools addresses {len(masters)} masters "
            f"({', '.join(m.master for m in masters)}); LP mooncake_store plays one, "
            "or a pool a writer rank where the entry states capacity_bytes a rank"
        )
    # ATOM's writer ranks: a DP replica's TP ranks, of each pipeline stage
    # (`pp_aware_rank_and_world`); each rank's slots from its own chunk.
    tp = 1 if config.enable_dp_attention else config.tensor_parallel_size
    world = tp * config.pipeline_parallel_size
    here = {
        "protocol": cfg.protocol,
        "chunk_tokens": cfg.chunk_tokens,
        "world": world,
        "save_workers": envs.OFFLOAD_COPY_WORKERS,
        "load_workers": envs.OFFLOAD_LOAD_WORKERS,
    }
    differ = [
        f"{k} {v} (the entry's {entry[k]})" for k, v in here.items() if v != entry[k]
    ]
    for rank in range(world if world == entry["world"] else 0):
        stride = _round_up(store.of_rank(entry, "chunk_bytes", rank), _SLOT_ALIGN)
        for k, pool in (
            ("save_slots", "save_pool_bytes"),
            ("load_slots", "load_pool_bytes"),
        ):
            if (v := getattr(cfg, pool) // stride) != store.of_rank(entry, k, rank):
                differ.append(
                    f"rank {rank}'s {k} {v} (the entry's "
                    f"{store.of_rank(entry, k, rank)})"
                )
    if differ:
        raise ValueError(
            f"this deployment's {', '.join(differ)}: the {store.ENTRY} entry's "
            "laws were fitted at its fields"
        )
    if (_pd_kv(config) or {}).get(INSTANCE_KEY, 0) == 0:
        _start_lp(MOONCAKE_STORE_MAIN, str(config.kv_cache_block_size))


#: LP `mooncake_store`'s process, given the deployment's KV block tokens.
MOONCAKE_STORE_MAIN = (
    "import sys; from atom.compass import run; run.mooncake_store(int(sys.argv[1]))"
)


def mooncake_store(block_tokens: int) -> None:
    """LP `mooncake_store`: a shared Mooncake Store's master on the run's clock
    until its ``+inf`` grant, then its hand-in, which carries its counts
    (`mooncake_store_server.COUNTS`) as ``store``. It serves at a port on the
    clock endpoint's host and names that endpoint in the out_dir's
    `ENDPOINT_FILE`. At its first stamped request, once every member of the
    engines it serves has planned its pool, it checks their plans against the
    entry (`check_plans`), each chunk `block_tokens` KV blocks of tokens. At
    the end of each client run, after its hand-in, an entry stating
    ``clear_between_client_runs`` has it empty the Store and mount it at the
    next client run's size (`MooncakeStoreLP.clear`)."""
    import zmq

    from atom.compass.kv import mooncake_store_server as store

    _outlive_the_group_stop()
    run = spec()
    entry = run[store.ENTRY]
    rt = _runtime(MOONCAKE_STORE, run)
    served = {c.source.name for c in rt.table.channels_into(MOONCAKE_STORE)}
    # A pipeline's head stage is served; every stage's members plan a chunk.
    heads = {re.sub(r"-pp0$", "", n) for n in sorted(served)}
    members = [
        n
        for n in _hand_ins(run)
        if re.sub(r"-pp\d+$", "", n.partition(".")[0]) in heads
    ]

    def check_plans() -> None:
        store.check_plans(entry, _published_plans(run, members), block_tokens)

    sock = zmq.Context.instance().socket(zmq.ROUTER)
    host = run["clock_endpoint"].rpartition(":")[0]
    port = sock.bind_to_random_port(host)
    _write_whole(Path(run["out_dir"]) / store.ENDPOINT_FILE, f"{host}:{port}")
    lp = store.MooncakeStoreLP(
        rt,
        sock,
        store.master_of(entry),
        lookup_s=entry["lookup_s"],
        check_plans=check_plans,
    )
    name = MOONCAKE_STORE.name
    rt.client_run_ended = _store_client_run_ended(run, rt, lp)
    ended = []
    try:
        lp.serve()
    except BaseException as e:
        ended = [f"ended:{name} raised {e!r}"]
        raise
    finally:
        _hand_in(run, name, rt.refusals + ended, store=lp.counts)
        sock.close(linger=0)


def _store_client_run_ended(run: dict, rt, lp):
    """LP `mooncake_store`'s `client_run_ended`: at the end of client run ``k``
    its hand-in, then, where the entry states ``clear_between_client_runs``, an
    empty Store mounted at client run ``k + 1``'s size (`MooncakeStoreLP.clear`)."""
    from atom.compass.kv import mooncake_store_server as store

    entry = run[store.ENTRY]

    def ended(k: int) -> None:
        _hand_in(run, MOONCAKE_STORE.name, rt.refusals, k, store=lp.counts)
        if entry.get("clear_between_client_runs"):
            lp.clear(store.client_run_capacity(entry, k + 1))

    return ended


def configuration(run: dict) -> dict:
    """The run file without its per-run keys."""
    return {k: v for k, v in run.items() if k not in PER_RUN}


def router_binary(path: str, mesh: str | None = None) -> dict:
    """The router binary `path` names, resolved on PATH, as the run summary
    records it: where it is, and the commit and git status it was built at.

    Refuses, naming the binary, one built from an uncommitted tree or at no
    commit, and one whose router source differs from `mesh` (this tree's
    ``atom/mesh`` by default), the working tree included. A stamped snapshot
    has no history to diff, so there only its own commit matches."""
    exe = shutil.which(path) or path
    text = subprocess.run(
        [exe, "--version-verbose"], capture_output=True, text=True, check=True
    ).stdout
    built = dict(re.findall(r"^\s*Git (Commit|Status): *(\S*)", text, re.MULTILINE))
    commit, status = built.get("Commit", ""), built.get("Status", "")
    named = f"router binary {exe} (commit {commit or '?'}, status {status or '?'})"
    if status != "clean":
        raise ValueError(f"{named} was built from an uncommitted tree; rebuild it")
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise ValueError(f"{named} names no commit; build it in a git checkout")
    mesh = mesh or os.path.join(provenance.module_root("atom"), "mesh")
    tree = provenance.git_tree_root("atom", mesh)
    if tree.revision_kind == "commit":
        if not tree.revision.startswith(commit):
            raise ValueError(
                f"{named} is not this snapshot's commit {tree.revision}; "
                "a snapshot has no history to compare the router source across"
            )
    else:
        diff = subprocess.run(
            ["git", "-C", mesh, "diff", "--quiet", commit, "--", "."],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff.returncode:
            why = diff.stderr.strip() or "it differs from the tree's"
            raise ValueError(f"{named}: router source at {mesh}: {why}")
    return {"path": exe, "commit": commit, "status": status}


def _servers(run: dict) -> list[str]:
    """The run's server LPs by name: every LP but the traffic and router LPs."""
    return [
        lp.name
        for lp in channel_table(run).registry.ids()
        if lp not in (TRAFFIC, ROUTER)
    ]


def _hand_ins(run: dict) -> list[str]:
    """The name each server LP hands its refusals in under: ``<lp>.<member>`` for
    each member of an LP declared with members, else the LP's name."""
    members = _members(channel_table(run), _widths(run)) or {}
    return [
        f"{lp}.{m}" if m else lp
        for lp in _servers(run)
        for m in members.get(LpId(lp), [None])
    ]


class _RecordingAuthority(ClockAuthority):
    """The authority, recording the step table: a row per grant, each step an
    LP took on its own below a ceiling included, and a row per message each
    releases, from the receiver's clock before that grant to
    the message's arrival; and the wall time of each client run, from its
    client's first request in `starts` to its end in `ends`, which `wall`
    sums; the finish sets `finished` and `done`. Each client run that ends
    before the last has its summary written once every server LP has handed
    in. A frontend's TAR to ``+inf`` before the finish is a stop: it registers
    that frontend's send log, finishes the run there and sets `stopped`. Every
    LP but the client is granted ``+inf``, and the client too while a client
    run is open, from its bind to its end; outside one it holds no connection.
    Each such grant releases every message registered for its receiver, and to
    a member also those a park or a call of later rounds held for it. A stop
    with none open, on a server whose ``client_runs`` is ``null``, sets
    `stopped_whole`: the run ended as its run file says, as at the finish."""

    def __init__(self, run: dict) -> None:
        # The tree that runs, read before the run so an unidentifiable one is
        # refused before it costs anything.
        self.source_root = provenance.git_tree_root(
            "atom", provenance.module_root("atom")
        )
        self.steps = StepTable(json.dumps(configuration(run), sort_keys=True))
        self.run = run
        self.finished = None
        self.starts, self.ends = [], []
        self.stopped = self.stopped_whole = self.client_run_open = False
        self.done = threading.Event()
        self._clock_before = {}  # LP -> its clock before its latest grant
        # A rerun into the same out_dir starts from none of the last run's files.
        out = Path(run["out_dir"])
        for name in (
            LP_FILE,
            STEP_TABLE_FILE,
            SUMMARY_FILE,
            "now-{}.raised",
            "now-{}.kv_block",
        ):
            for f in out.glob(name.format("*")):
                f.unlink()
        for d in out.glob(CLIENT_DIR.format("*")):
            shutil.rmtree(d)
        table = channel_table(run)
        count = run.get("client_runs", 1)
        local = run.get(LOCAL_GRANTS, True)
        if not isinstance(local, bool):
            raise TypeError(
                f"the run file's {LOCAL_GRANTS} is {local!r}; it is true, which "
                "gives grants their ceilings, or false"
            )
        super().__init__(
            table,
            timeline=self,
            members=_members(table, _widths(run)),
            bound_s=run["bound_s"],
            client=TRAFFIC,
            client_runs=math.inf if count is None else count,
            ceilings=local,
        )

    def record(self, lp, time_from, time_to, kind, recovered, released=(), *_):
        self._clock_before[lp] = time_from
        self.steps.record(lp, time_from, time_to, kind, detail="recovered" * recovered)
        for ch, seq, arrival in released:
            # From the receiver's clock as the grant found it.
            self.steps.record(lp, time_from, arrival, "release", ch, seq)

    def on_bind(self, lp):
        self.client_run_open |= lp == TRAFFIC
        return super().on_bind(lp)

    def on_request(
        self,
        lp,
        kind,
        t,
        log,
        t_daemon=math.inf,
        member=None,
        steps=(),
        ahead=False,
        rounds=(),
    ) -> list:
        first = lp == TRAFFIC and len(self.starts) == len(self.ends)
        if first and self.final_clocks is None:  # none opens after the finish
            self.starts.append(time.monotonic())
        stop = kind == TAR and t == math.inf and lp.name.startswith(FRONTEND.name)
        if stop and self.final_clocks is None:
            self.requests_served[lp.name] += 1
            for ch, seq, arrival in self._catch_up(lp, steps, log, member):
                self._register(lp, ch, seq, arrival, member)
            self.stopped = True
            self.stopped_whole = (
                self._client_runs == math.inf and not self.client_run_open
            )
            replies = []
            for to, g, held in self._address(self._finish()):
                if to == TRAFFIC and not self.client_run_open:
                    continue
                # What the stop releases, which no recorded grant does, beside
                # what was held for a member, which its own grant recorded.
                added = self._release_all(held)
                i = to[0] if isinstance(to, tuple) else to  # a member's: (lp, member)
                for ch, msgs in added.items():
                    for seq, arrival in msgs:
                        clock = self._clock_before[i]
                        self.steps.record(i, clock, arrival, "release", ch, seq)
                replies.append((to, g, {ch: [*held[ch], *added[ch]] for ch in held}))
        else:
            replies = super().on_request(
                lp, kind, t, log, t_daemon, member, steps, ahead, rounds
            )
        while len(self.ends) < len(self.ended):
            self.ends.append(time.monotonic())
            self.client_run_open = False
            if self.final_clocks is None:  # an end before the last
                out = Path(self.run["out_dir"])
                _write_whole(out / STEP_TABLE_FILE, self.steps.text())
                threading.Thread(
                    target=_write_client_run,
                    args=(self.run, self, len(self.ends) - 1),
                    name="compass client run summary",
                    daemon=True,
                ).start()
        if self.final_clocks is not None and self.finished is None:
            self.finished = time.monotonic()
            self.done.set()
        return replies

    def wall(self, k: int | None = None) -> float:
        """The wall seconds client runs 0 to `k` served, all of them with `k`
        None, each from its client's first request to its end: for one still
        open, the finish a stop brought or, before the finish, now."""
        if k is None:
            ends = [*self.ends, self.finished or time.monotonic()]
        else:
            ends = self.ends[: k + 1]
        return sum(end - start for start, end in zip(self.starts, ends))

    def _release_all(self, into: dict) -> dict:
        """Every message registered on the channels of `into` and not yet
        released, now released: by channel, its ``(seq, arrival)`` pairs."""
        released = {ch: [*self._undelivered[ch].items()] for ch in into}
        for ch in into:
            self._undelivered[ch].clear()
        return released


class _PublishedClock(clock.LPRuntime):
    """An LP runtime whose clock is also written to the file `NOW_ENV` names."""

    def __init__(self, *args) -> None:
        self.slot = np.memmap(os.environ[NOW_ENV], np.float64, "w+", shape=(1,))
        super().__init__(*args)

    @property
    def now(self) -> float:
        return self._now

    @now.setter
    def now(self, t: float) -> None:
        self._now = self.slot[0] = t


class _StoppingClock(clock.LPRuntime):
    """An LP runtime that, when its run ends before its ``+inf`` grant, at a
    stop, first ends every LP's run through the authority with a TAR to
    ``+inf``, carrying its send log, so each engine leaves its loop and hands
    in before the server closes the engines. A run a refusal ended has nothing
    left to end."""

    def end_run(self) -> None:
        if self.now != math.inf:
            with contextlib.suppress(ClockAbort):
                self._ca_send(TAR, math.inf, math.inf)
                while self.conn.recv()[0] != math.inf:  # up to its own +inf grant
                    pass
        super().end_run()


def _runtime(
    lp: LpId, run: dict, member=None, cls=clock.LPRuntime, here=False
) -> clock.LPRuntime:
    """`lp`'s runtime on the run's clock; `here` when this process serves it."""
    reach = connect_here if here else connect
    rt = cls(lp, channel_table(run), reach(lp, run["clock_endpoint"], member))
    clock.install(rt)
    return rt


@contextlib.contextmanager
def _start_on_leaving(rt: clock.LPRuntime):
    yield
    rt.start_run()


def frontend(config):
    """Around the `CoreManager` construction; a no-op context on a real run.

    A real run with the parity record's variable set runs ATOM's runner
    through the recording one, and refuses any other runner, which would
    record nothing. With it set, real or simulated, each request's sequence
    carries its response id (`parity.carry_response_ids`), and each request
    routed across DP ranks is recorded (`_record_routes`).
    """
    global _authority, _early_end
    from atom.compass import parity

    if os.environ.get(parity.ENV):
        _record_routes(Path(os.environ[parity.ENV]))
        parity.carry_response_ids()
    run = spec()
    if run is None:
        if os.environ.get(parity.ENV):
            recording = {ATOM_RUNNER: parity.RUNNER, RAPIDSERVE_RUNNER: parity.RAPID}
            if config.runner_qualname not in recording:
                raise ValueError(
                    f"{parity.ENV} is set and runner {config.runner_qualname!r} "
                    f"is named; only {' and '.join(recording)} each has a "
                    "recording runner"
                )
            config.runner_qualname = recording[config.runner_qualname]
        return contextlib.nullcontext()
    _refuse_unserved()
    pc = config.parallel_config
    rapid = config.enable_rapidserve
    if rapid != run.get(RAPIDSERVE, False):
        raise ValueError(
            f"the deployment's enable_rapidserve is {rapid} and the run file's "
            f"{RAPIDSERVE} is {run.get(RAPIDSERVE, False)}; the channel table "
            "every LP reads would declare the wrong engines"
        )
    if rapid and config.disagg_constrained:
        raise ValueError(
            "--disagg-constrained has decode write its decode tokens to shared "
            "memory that prefill reads on wall time, off the clock; run RapidServe "
            "without it"
        )
    side, pp = _pd_side(config), config.pipeline_parallel_size
    if side == "D" and pp > 1:
        raise ValueError(
            f"a decode side of {pp} pipeline stages: ATOM's decode head reports "
            "finished_recving from its own stage only, so ATOM runs no pipelined "
            "decode side"
        )
    if pp != _pp(run, side):
        key = PREFILL_PP if side == "P" else "pipeline_parallel_size"
        raise ValueError(
            f"the deployment runs {pp} pipeline stages and the run file's {key} "
            f"is {_pp(run, side)}; the channel table every LP reads would declare "
            "the wrong stages"
        )
    atom_runner, ours = ATOM_RUNNER, COMPASS_RUNNER_QUALNAME
    if rapid:
        atom_runner, ours = RAPIDSERVE_RUNNER, COMPASS_RAPIDSERVE_QUALNAME
    if config.runner_qualname not in (atom_runner, ours):
        raise ValueError(
            f"runner {config.runner_qualname!r} is named, and a simulated run "
            f"prices its steps with {ours}"
        )
    standalone = CLOCK_ENV in os.environ
    if side is not None and not standalone:
        raise ValueError(
            "this API server would co-host the clock authority, and a "
            "prefill-decode run spans two API servers that must share one; "
            "start it on its own and name it with --compass-clock-endpoint"
        )
    # Read before `CoreManager` makes every TP rank a DP rank under DP-attention.
    dp = pc.data_parallel_size * (
        config.tensor_parallel_size if config.enable_dp_attention else 1
    )
    if dp != _dp(run, side):
        key = DP_WIDTHS[side] if DP_WIDTHS.get(side) in run else "data_parallel_size"
        raise ValueError(
            f"the deployment runs {dp} data-parallel ranks and the run file's "
            f"{key} is {_dp(run, side)}; the channel table every LP "
            "reads would declare the wrong ranks"
        )
    if pc.data_parallel_size > 1 and pc.is_multinode_dp:
        _refuse_unshared(run, standalone)
    if dp > 1 and config.fake_eplb:
        raise ValueError(
            "--fake-eplb may start fewer engines than the data-parallel ranks "
            "the channel table declares, and a rank that never joins holds the run"
        )
    config.runner_qualname = ours
    if pc.data_parallel_rank > 0:
        # A worker node of a multi-node group: engines only, each a member.
        return contextlib.nullcontext()
    _refuse_sync_utility_commands()
    _price_images(run.get(IMAGE_PREPROCESS))
    lp = _frontend_lp(config, run)
    authority = None
    if not standalone:
        _authority = authority = _RecordingAuthority(run)
        serve(_authority, run["clock_endpoint"])
        # Not a daemon: the API server's main thread may die of the early end
        # before this thread has written the summary.
        _early_end = threading.Thread(
            target=_write_on_early_end, args=(run, authority), name="compass early end"
        )
        _early_end.start()
    _launch_lmcache(config, run, side)
    _launch_mooncake_store(config, run, side)
    # A co-hosting frontend reaches the authority in-process, with no socket.
    rt = _runtime(lp, run, cls=_StoppingClock, here=authority is not None)
    rt.client_run_ended = lambda k: _frontend_hand_in(run, rt, k)
    from atom.compass.run import metrics

    rt.sampler = metrics.Sampler(rt, "frontend", metrics.ROLES[side], lp.name)
    metrics.sample_frontend(rt.sampler)
    return _start_on_leaving(rt)


def _refuse_unshared(run: dict, standalone: bool) -> None:
    """Refuse a node of a multi-node data-parallel group that would not reach
    the run's one authority, or whose hand-ins would miss the authority's
    out_dir: that directory holds no `AUTHORITY_FILE` naming the port this
    node's clock endpoint does."""
    if not standalone:
        raise ValueError(
            "this node runs part of a multi-node data-parallel group, whose nodes "
            "share one clock authority; start it on its own and name it with "
            "--compass-clock-endpoint on every node"
        )
    mark = Path(run["out_dir"]) / AUTHORITY_FILE
    port = run["clock_endpoint"].rpartition(":")[2]
    if not mark.is_file() or mark.read_text().rpartition(":")[2] != port:
        raise ValueError(
            f"{mark} names no authority at {run['clock_endpoint']}: every node of "
            "a multi-node data-parallel group hands in to the authority's out_dir, "
            "so each must see it at the same path; start the authority first"
        )


def _refuse_unserved() -> None:
    """Refuse the frontends no client of a simulated run reaches: the Atomesh
    standalone service, and an `LLMEngine` that the API server's ``main`` is
    not building, such as an offline one."""
    from atom.utils import envs

    if envs.USE_ATOMESH_ENTRYPOINTS:
        raise ValueError(
            "USE_ATOMESH_ENTRYPOINTS=1 serves through the Atomesh standalone "
            "service, whose worker thread sends each request, and only the "
            "frontend's clock owner may; serve a simulated run with the API server"
        )
    root = Path(atom.__file__).parent
    on_stack = {
        (Path(f.f_code.co_filename), f.f_code.co_name)
        for f, _ in traceback.walk_stack(sys._getframe())
    }
    building = (root / "model_engine" / "llm_engine.py", "__init__") in on_stack
    served = (root / "entrypoints" / "openai" / "api_server.py", "main") in on_stack
    if building and not served:
        raise ValueError(
            "an offline LLMEngine has no client to bind the traffic LP, so no "
            "grant passes its first request; serve a simulated run with the API "
            "server's --compass-run"
        )


def _refuse_sync_utility_commands() -> None:
    """Make `CoreManager.broadcast_utility_command_sync` refuse a call on the
    frontend's clock owner, which waits on wall time for replies that reach it
    only at that owner's next clock call."""
    from atom.model_engine.engine_core_mgr import CoreManager

    sync = CoreManager.broadcast_utility_command_sync

    @functools.wraps(sync)
    def refusing(self, cmd, *args, **kwargs):
        rt = clock.installed()
        if rt is not None and threading.current_thread() is rt.owner:
            why = f"sync utility command {cmd!r} on the clock owner"
            rt.refusals.append(f"clock:{why}")
            raise RuntimeError(
                f"{rt.me}: {why}, which would wait on wall time for replies "
                "released only at its next clock call"
            )
        return sync(self, cmd, *args, **kwargs)

    CoreManager.broadcast_utility_command_sync = refusing


def _price_images(price: float | None) -> None:
    """Make `MediaLoader.load_image` charge `price` seconds to the station job
    loading the image. Refuse an image named by an http(s) URL, which the real
    server fetches on wall time, and any image when the run file states no
    price. The server fetches a chat's URLs itself before its media loader
    sees them (`images.load_chat_images`), so its download refuses too."""
    from atom.entrypoints.openai import images
    from atom.multimodal.media import MediaLoader
    from atom.utils.compass_loop import Refused, charge

    download = images._download

    @functools.wraps(download)
    def refusing(url):
        rt = clock.installed()
        if rt is None:  # no LP clock in this process: a load outside any run
            return download(url)
        why = "an image named by a remote URL"
        rt.refusals.append(f"media:{why}")
        raise Refused(("media", why))

    images._download = refusing
    load = MediaLoader.load_image

    @functools.wraps(load)
    def priced(self, data):
        rt, why = clock.installed(), None
        if rt is None:  # no LP clock in this process: a load outside any run
            return load(self, data)
        if isinstance(data, str) and data.startswith(("http://", "https://")):
            why = "an image named by a remote URL"
        elif price is None:
            why = f"an image with no {IMAGE_PREPROCESS} in the run file"
        if why is not None:
            rt.refusals.append(f"media:{why}")
            raise Refused(("media", why))
        charge(price)
        return load(self, data)

    MediaLoader.load_image = priced


#: `atom.compass.drive.serve.KEEP_ENV`, stated here as that module's import is
#: not this one's to make.
KEEP_ENV = "COMPASS_KEEP_COMPILED"


def kept_cache_dir(config) -> tuple[str, dict]:
    """The compiled-graph cache directory a real server of `config` keeps under
    `KEEP_ENV`, ``~/.cache/atom/torch_compile_cache/kept-<k>``, and its key's
    inputs: ``k`` is ATOM's own `Config.compute_hash` with the DP master's
    address and port blank (they change at every start, which is why ATOM's own
    directory is never found again), so every other configuration ATOM keys its
    graphs by still keys this one."""
    pc = config.parallel_config
    held = pc.data_parallel_master_ip, pc.data_parallel_master_port
    pc.data_parallel_master_ip, pc.data_parallel_master_port = "", 0
    try:
        k = config.compute_hash()[:16]
    finally:
        pc.data_parallel_master_ip, pc.data_parallel_master_port = held
    path = os.path.join(
        os.path.expanduser("~/.cache/atom"), "torch_compile_cache", f"kept-{k}"
    )
    inputs = {
        "key": k,
        "dir": path,
        "model": str(getattr(config, "model", "")),
        "tensor_parallel_size": getattr(config, "tensor_parallel_size", None),
        "pipeline_parallel": [
            getattr(pc, "pipeline_parallel_rank", None),
            getattr(config, "pipeline_parallel_size", None),
        ],
        "blanked": {"data_parallel_master_ip": held[0], "data_parallel_master_port": held[1]},
    }
    return path, inputs


def _keep_compiled(config, record) -> None:
    """`KEEP_ENV`: point `config`'s compiled-graph cache at its kept directory
    (`kept_cache_dir`) before the engine builds its workers, and state the key's
    inputs beside the record (``kept_cache.<pid>.json``) when one is kept."""
    path, inputs = kept_cache_dir(config)
    config.compilation_config.cache_dir = path
    if record:
        _write_whole(Path(record) / f"kept_cache.{os.getpid()}.json", json.dumps(inputs))


def engine(config):
    """Around the engine's construction in its process; a no-op context on a real run.

    At DP width above one, the process joins the engine LP as the member of its
    DP rank, and its host collectives first join the group rounds it owes; the
    DP group the engine core builds runs them over shared memory when its
    ranks share a host (`shm_collectives`). A
    pipeline stage is the engine LP ``engine-pp<stage>``, or a prefill side's
    ``engine-P-pp<stage>``.

    A real run recording its steps (`parity.ENV`) outside a pipeline stamps the
    engine's input pulls and output puts into ``dp<rank>.pulls.jsonl`` beside
    the record, and its IO-thread queue handoffs into ``dp<rank>.handoff.jsonl``
    (`clock.stamp_pulls`).
    """
    global _member
    run = spec()
    if run is None:
        from atom.compass import parity

        if os.environ.get(KEEP_ENV) == "1":
            _keep_compiled(config, os.environ.get(parity.ENV))
        record = os.environ.get(parity.ENV)
        if record and getattr(config, "pipeline_parallel_size", 1) == 1:
            record = parity.record_dir(record, config)
            record.mkdir(parents=True, exist_ok=True)
            rank = config.parallel_config.data_parallel_rank
            # The worker's step timestamps read the same host clock.
            pulls = record / f"dp{rank}.pulls.jsonl"
            handoff = record / f"dp{rank}.handoff.jsonl"
            clock.stamp_pulls(pulls, handoff, time.monotonic_ns)
        return contextlib.nullcontext()
    from atom.compass.run import metrics
    from atom.compass.runner import shm_collectives
    from atom.compass.runner.doorbell import sleeping_workers
    from atom.utils.distributed import utils

    lp, side = ENGINE, _pd_side(config)
    pc = config.parallel_config
    if (name := _engine_name(config, run)) is not None:
        lp = LpId(f"{ENGINE.name}-{name}")
    if getattr(config, "pipeline_parallel_size", 1) > 1:
        lp = LpId(f"{lp.name}-pp{pc.pipeline_parallel_rank}")
    if side is not None:
        _bind_transfer(config, run)
    if side == "D" or (
        side == "P"
        and (
            (_pd_kv(config) or {}).get("enable_chunked_transfer")
            or run.get(KV_SERVE, False)
        )
    ):
        # Decode posts its write requests at a forward's device end; a chunked
        # producer publishes its prompts' blocks at each forward's, and a stage
        # that serves its writes (`KV_SERVE`) queues each forward on its device.
        _watch_forwards()
    if "lmcache" in run:
        from atom.compass.kv.lmcache_mp import substitute

        substitute()
    _offload_tier(config, run)
    _store_shell(config, run)
    utils.LP_OF_RANK = dict.fromkeys(range(pc.data_parallel_size), lp)
    _member = f"dp{pc.data_parallel_rank}" if pc.data_parallel_size > 1 else None
    name = lp.name if _member is None else f"{lp.name}.{_member}"
    os.environ[NOW_ENV] = str(Path(run["out_dir"]) / NOW_FILE.format(name))
    if _member is not None:
        # Under the clock's wrap, so a call settles before it waits on its peers.
        shm_collectives.install()
    clock.settle_before_collectives()
    if _member is not None and any(
        _offloads(config.kv_transfer_config, c) for c in ("lmcache_mp", ENTRY_STORE)
    ):
        _park_before_dispatch()
    rt = _runtime(lp, run, _member, _PublishedClock)
    rt.sampler = metrics.Sampler(
        rt, "engine", metrics.ROLES[side], _frontend_lp(config, run).name
    )
    inner = _ignoring_sigterm_from_ready(_named(_start_on_leaving(rt)))
    if _member is not None:
        inner = shm_collectives.attaching_engine_group(inner)
    return sleeping_workers(inner)


@contextlib.contextmanager
def _ignoring_sigterm_from_ready(inner):
    """`inner`, in which the engine ignores SIGTERM from just before it says
    READY and sets ATOM's handler aside, for `engine_loop` to put back.

    The construction goes on after READY: a pipeline stage's transport, a DP
    engine's prefill delayer, a prefill-decode engine's `_init_disagg`. A group
    stop from then on would run ATOM's handler, which stops the workers before
    the engine hands in; it reaches the engine as the ``+inf`` grant instead.
    A construction that raises has the handler back."""
    from atom.model_engine.engine_core import EngineCore

    ready = EngineCore._send_ready_signal

    @functools.wraps(ready)
    def ignoring(self):
        _sigterm_at_ready.append(signal.signal(signal.SIGTERM, signal.SIG_IGN))
        return ready(self)

    EngineCore._send_ready_signal = ignoring
    try:
        with inner:
            yield
    except BaseException:
        if _sigterm_at_ready and (before := _sigterm_at_ready.pop()) is not None:
            signal.signal(signal.SIGTERM, before)
        raise
    finally:
        EngineCore._send_ready_signal = ready


def _offloads(kv, connector: str) -> bool:
    """Whether a KV transfer config offloads KV through `connector`, on its own
    or as one connector of a ``multi``."""
    if not kv:
        return False
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    def name(c):
        return KVConnectorFactory.canonical_name(c.get("kv_connector", "moriio"))

    subs = kv.get("connectors", []) if name(kv) == "multi" else [kv]
    return any(name(c) == connector for c in subs)


def _offload_tier(config, run: dict) -> None:
    """Have the connector factory build ``lmcache_offload`` as Compass's
    (`atom.compass.kv.lmcache_offload`) when this deployment offloads through
    it, once `refuse_offload_tier` holds."""
    from atom.compass.kv import lmcache_offload

    refuse_offload_tier(config, run)
    if lmcache_offload.ENTRY in run:
        lmcache_offload.substitute()


def refuse_offload_tier(config, run: dict) -> None:
    """Refuse a deployment and a run file that disagree on whether it offloads
    KV through ``lmcache_offload``, an entry the connector cannot read, and a
    deployment above DP or pipeline width 1, whose ranks' tiers this does not
    join."""
    from atom.compass.kv import lmcache_offload

    offloads = _offloads(config.kv_transfer_config, "lmcache_offload")
    if offloads != (lmcache_offload.ENTRY in run):
        raise ValueError(
            f"this deployment {'offloads' if offloads else 'offloads no'} KV "
            f"through lmcache_offload and the run file "
            f"{'has no' if offloads else 'has an'} {lmcache_offload.ENTRY} entry"
        )
    if not offloads:
        return
    widths = {
        "data_parallel_size": config.parallel_config.data_parallel_size,
        "pipeline_parallel_size": getattr(config, "pipeline_parallel_size", 1),
    }
    if wide := [f"{k} {v}" for k, v in widths.items() if v > 1]:
        raise ValueError(
            f"lmcache_offload is simulated at DP and pipeline width 1; this "
            f"deployment runs {', '.join(wide)}"
        )
    lmcache_offload.check_entry(run[lmcache_offload.ENTRY])


def _store_shell(config, run: dict) -> None:
    """Have the connector factory build ``mooncake_store`` as Compass's
    (`atom.compass.kv.mooncake_store`) when this deployment offloads to a
    Mooncake Store; refuse one whose run file has no ``mooncake_store`` entry,
    or under ``ATOM_STORE_ASYNC_RENEW``. Its frontend checked the rest against
    the entry (`_launch_mooncake_store`)."""
    from atom.compass.kv import mooncake_store
    from atom.utils import envs

    if not _offloads(config.kv_transfer_config, ENTRY_STORE):
        return
    if ENTRY_STORE not in run:
        raise ValueError(
            "this deployment offloads KV to a Mooncake Store and the run file "
            f"has no {ENTRY_STORE} entry"
        )
    if envs.ATOM_STORE_ASYNC_RENEW:
        raise ValueError(
            "ATOM_STORE_ASYNC_RENEW renews a load's prebuilt keys and polls the "
            "renewal on the wall clock; the simulated Store renews a sequence's "
            "range on LP mooncake_store"
        )
    mooncake_store.substitute()


#: The engine's worker calls that run a forward, real or the DP dummy, and
#: reply with its price.
FORWARD_CALLS = ("forward", "dummy_execution")


def _park_before_dispatch() -> None:
    """A DP member parks as its last clock call before it dispatches a forward,
    real or the DP dummy, and asks no TAR from its DP state sync to that park.

    A member waiting on the lmcache MP server, or on a Mooncake Store
    renewal, asks for time alone while its peers, their forwards dispatched,
    wait on it in the predicted forward's gather and make no clock call; a
    park stands in for each of them."""
    from atom.model_engine.async_proc import AsyncIOProcManager
    from atom.model_engine.engine_core import DPEngineCoreProc

    call, sync = AsyncIOProcManager.call_func, DPEngineCoreProc._sync_dp_state

    @functools.wraps(call)
    def parking(self, func_name, *args, **kwargs):
        rt = clock.installed()
        if func_name in FORWARD_CALLS and rt and rt.in_run:
            cpu, at = time.thread_time(), rt.now
            rt.park()
            parks = rt.__dict__.setdefault("parks", [])
            parks.append((at, rt.now, time.thread_time() - cpu))
        return call(self, func_name, *args, **kwargs)

    @functools.wraps(sync)
    def synced(self, *args, **kwargs):
        out = sync(self, *args, **kwargs)
        rt = clock.installed()
        if rt is not None and rt.in_run:
            rt.parking = True
        return out

    AsyncIOProcManager.call_func = parking
    DPEngineCoreProc._sync_dp_state = synced


def _watch_forwards() -> None:
    """Have the engine's `call_func` tell the simulated KV connector's
    consumer of each forward, the DP group's dummy too, whose priced reply
    it returns (`atom.compass.kv.connector.forward_returned`), once in a
    process."""
    from atom.compass.kv import connector
    from atom.model_engine.async_proc import AsyncIOProcManager

    call = AsyncIOProcManager.call_func
    if getattr(call, "compass_watched", False):
        return

    @functools.wraps(call)
    def watched(self, func_name, *args, **kwargs):
        out = call(self, func_name, *args, **kwargs)
        if func_name in FORWARD_CALLS and getattr(out, "predicted_s", None) is not None:
            if func_name == "forward" and args:
                connector.forward_returned(out, args[0])  # the batch it ran
            else:
                connector.forward_returned(out)
        return out

    watched.compass_watched = True
    AsyncIOProcManager.call_func = watched


@contextlib.contextmanager
def _named(inner):
    """`inner`, with `NOW_ENV` set for the processes started in it."""
    try:
        with inner:
            yield
    finally:
        del os.environ[NOW_ENV]


def runner(model_runner, tree) -> None:
    """Install the cost backend and the device readings from the run file.

    `tree` is the model built on the meta device, which the weights are read
    off."""
    run = spec()
    if run is None:
        return
    from atom.compass.artifacts import machine_platform
    from atom.compass.backends.geometry import Parallelism
    from atom.compass.backends.launch_features import PATHS, stated
    from atom.compass.backends.step_labels import STALLS
    from atom.compass.backends.step_linear import StepLinearBackend, StepLinearLaw
    from atom.compass.backends.step_roofline import (
        StepFallbackBackend,
        StepRooflineBackend,
    )
    from atom.compass.backends.step_stub import Coefficients, StepStubBackend
    from atom.compass.memory import ModelTerms, device_readings
    from atom.compass.memory.check import record
    from atom.compass.runner.overrides import (
        ADMISSION_IN_E,
        HostDeviceRecurrence,
        install_cost_backend,
        install_device_readings,
    )

    config = model_runner.config
    tp = config.tensor_parallel_size
    refuse_renamed_keys(run, "the run file")
    for name in (STEP_LINEAR, "host"):
        pick_fitted(run, name, config)
    backends = (STEP_STUB, STEP_LINEAR, STEP_GRID, OP_GRAPH)
    keys = [k for k in backends if k in run]
    if len(keys) != 1:
        raise ValueError(
            f"a run file prices its steps from one of {', '.join(backends[:-1])} "
            f"and {backends[-1]}, and this one has {' and '.join(keys) or 'none'}"
        )
    replay = None
    if "replay" in run:
        if "host" in run:
            raise ValueError(
                "a run file's replay ends each step where the real one did, in "
                "place of its host laws, and this one has both"
            )
        from atom.compass.parity.replay import ReplayHost

        dp_rank = config.parallel_config.data_parallel_rank
        stage = config.parallel_config.pipeline_parallel_rank
        # This engine's record: prefill/, or prefill1/ for engine-P1.
        name = _engine_name(config, run) or ""
        side = {"P": "prefill", "D": "decode"}.get(name[:1], "") + name[1:]
        replay = ReplayHost(Path(run["replay"]) / side, dp_rank, stage)
    if STEP_LINEAR in run:
        # A law prices a rank at the tensor-parallel width its engine config states.
        refuse_unfitted_law(run, config)
        # Under DP-attention every rank runs at TP 1, and experts shard over the group.
        dpa = config.enable_dp_attention
        dp = config.parallel_config.data_parallel_size if dpa else 1
        # A pipeline stage prices its prefill by its own stage's vector, where
        # the law states one (`StepLinearLaw.at_stage`).
        stage = getattr(config.parallel_config, "pipeline_parallel_rank", 0)
        law = StepLinearLaw(run[STEP_LINEAR])
        backend = StepLinearBackend(law.at_stage(stage), dp)
        if law.stage_prefill:
            # A law fitted on a pipelined prefill's stage records (`compass fit
            # steps --stage-record`) prices each stage's own layers: the whole
            # of this stage's work, which the runner's stage check asks for.
            backend.stage_share = (1.0, 1.0)
    elif STEP_STUB in run:
        # A pipeline stage is priced for its own span of the stack, the span
        # ATOM's `make_layers` builds it from.
        hf, pp = config.hf_config, getattr(config, "pipeline_parallel_size", 1)
        depth, share = hf.num_hidden_layers, None
        start, end = 0, depth
        if pp > 1:
            from atom.compass.backends.geometry import paged_layers
            from atom.models.utils import get_pp_indices

            pp_rank = config.parallel_config.pipeline_parallel_rank
            start, end = get_pp_indices(depth, pp_rank, pp)
            text = getattr(hf, "text_config", hf)
            paged = paged_layers(text, 0, depth)
            if not paged:
                raise ValueError(
                    "a pipeline stage prices its history terms on its share of the "
                    "paged layers, and this stack has no layer that holds a cache "
                    "of every past token"
                )
            share = ((end - start) / depth, paged_layers(text, start, end) / paged)
        backend = StepStubBackend(
            Coefficients(**run[STEP_STUB]),
            Parallelism(tp_size=tp, pp_size=pp),
            stack_layers=end - start,
            stage_share=share,
        )
    machine = _machine(run)
    platform = machine_platform(machine)
    bundle = None
    if STEP_GRID in run:
        from atom.compass.run.step_grid import load

        refuse_context_parallel(STEP_GRID, config)
        backend, bundle = load(run[STEP_GRID], config, machine)
    if "store" not in run:
        raise ValueError(
            "the run file names no store, and a run reads its configuration's "
            "memory readings from one; compass measure memory takes them"
        )
    from atom.compass.run import memory

    carried = run.get(memory.CARRIED, ())
    measured = memory.load(run["store"], config, machine, model_runner.rank, carried)
    # `ModelRunner.warmup_model` sizes its batch from the scheduler bounds and
    # the DP and PCP widths, all in the readings' key, so the tokens the
    # reading warmed up with are the ones this deployment's warmup runs.
    warmup = measured.values["warmup_tokens"]
    model = ModelTerms.from_tree(tree, activations=measured.activations())
    fallback = run.get("step_roofline", False)
    if not isinstance(fallback, bool):
        raise TypeError(
            f"the run file's step_roofline is {fallback!r}; it is true, which "
            "prices the steps the backend refuses by the step roofline, or false"
        )
    padding = run.get("padding", "executed")
    if padding not in ("executed", "none"):
        raise ValueError(
            f"the run file's padding is {padding!r}; it is none, which prices a "
            "replayed graph's real rows, or executed or absent, which prices the "
            "rows it pads to"
        )
    if OP_GRAPH in run:
        if run[OP_GRAPH] is not True:
            raise TypeError(
                f"the run file's op_graph is {run[OP_GRAPH]!r}; it is true, which "
                "prices each step from the op_graph the store holds at its shape"
            )
        model_runner.compass_lazy = {"lazy_traces": 0, "lazy_trace_wall_seconds": 0}
        backend = op_graph(
            run,
            config,
            machine,
            ep=ep_width(tree),
            lazy=model_runner.compass_lazy,
        )
    elif PRICE_LIST in run:
        raise ValueError(
            "the run file names a price_list and no op_graph, the one backend "
            "that reads it"
        )
    if fallback or STEP_GRID in run:
        from atom.compass.run.modules import drafter, walk

        modules = walk(tree, drafter(config))
        roofline = StepRooflineBackend(machine, modules, padded=padding != "none")
        backend = StepFallbackBackend(backend, roofline)
    pp = getattr(config, "pipeline_parallel_size", 1)
    if pp > 1 and tp > 1 and not hasattr(backend, "all_gather_s"):
        from atom.utils import envs

        if envs.ATOM_PP_SEND_ALLGATHER:
            raise ValueError(
                "a pipeline stage at TP above 1 all-gathers each sharded tensor it "
                f"receives, and {type(backend).__name__} prices no all-gather; "
                "set ATOM_PP_SEND_ALLGATHER=0"
            )
    install_cost_backend(model_runner, backend)
    if NOW_ENV in os.environ:
        now = Path(os.environ[NOW_ENV])
        model_runner.compass_raised_file = _raised_file(now, model_runner.rank)
        # What it planned prices a prefill-decode transfer and sizes LP lmcache,
        # and LP mooncake_store checks it.
        kv = config.kv_transfer_config
        planned = _pd_side(config) is not None or any(
            _offloads(kv, c) for c in ("lmcache_mp", "mooncake_store")
        )
        if planned and model_runner.rank == 0:
            model_runner.compass_block_file = _block_file(now)
        slot = np.memmap(now, np.float64, "r", shape=(1,))
        host = run.get("host", {})
        refuse_unfitted_host(host, config, platform)
        launch = stated(host.get("key"))
        pp = getattr(config, "pipeline_parallel_size", 1)
        stage = (config.parallel_config.pipeline_parallel_rank, pp) if pp > 1 else None
        # A drafter's runner waits for the last output's MTP status when it
        # defers its output.
        mtp = bool(getattr(config, "speculative_config", None)) and (
            model_runner.tokenID_processor.is_deferred_out
        )
        # RapidServe's prefill_forward synchronizes its stream before it returns.
        rapid = getattr(config, "enable_rapidserve", False)
        processor = getattr(model_runner, "tokenID_processor", None)
        model_runner.compass_host = replay or HostDeviceRecurrence(
            host.get("laws", {}),
            host.get("skip", ()),
            lambda: float(slot[0]),
            stage,
            mtp=mtp,
            sync=rapid and not config.disagg_is_decode,
            admit=host.get("admit", ADMISSION_IN_E),
            columns=host.get("columns"),
            launch=launch,
            defer=getattr(processor, "is_deferred_out", True),
            paths=host.get(PATHS),
            stalls=host.get(STALLS),
        )
        if replay is not None:
            replay.now = lambda: float(slot[0])
    model_runner.compass_abort = run.get("on_refusal") == "abort"
    reserved = _graph_pool(model_runner, machine, model)
    # The readings' label: under DP-attention the layout, with the EP width
    # when the experts shard; one rank of it is a TP1 engine.
    dp = config.parallel_config.data_parallel_size
    layout = tp
    if dp > 1 and config.enable_dp_attention:
        ep = f"x{tp * dp}" if config.enable_expert_parallel else ""
        layout = f"{tp}x{dp}{ep}"
    readings = device_readings(
        machine,
        memory=measured,
        tp_width=layout,
        model=model,
        cudagraph_overhead=reserved,
    )
    install_device_readings(model_runner, readings)
    model_runner.compass_memory = record(readings, tokens=warmup, memory=measured)
    if bundle is not None:
        model_runner.compass_step_grid = {**bundle, "describe": backend.describe()}


def ep_width(tree) -> int:
    """The EP width as ATOM's MoE in `tree` resolves it, 1 when it does not shard."""
    moe = [m for m in tree.modules() if hasattr(m, "moe_parallel_config")]
    return moe[0].moe_parallel_config.ep_size if moe else 1


def op_graph(
    run: dict,
    config,
    machine,
    ep: int = 1,
    lazy: dict | None = None,
    env: dict | None = None,
):
    """The `OpGraphBackend` the run file `run` names for `config` on
    `machine` at EP width `ep`, with its price list when it names one; with
    `lazy`, a shape no graph holds is traced from the run file's `ATOM_ARGS`
    when it holds them (`_op_graphs`), counted in `lazy`, which also sums each
    `price` call's wall seconds, its lookups among them, as
    ``price_wall_seconds``; with `env`, its graphs are read under that
    performance environment (`forward.structure`)."""
    from atom.compass.backends.op_graph import OpGraphBackend

    drafter = getattr(config, "speculative_config", None)
    lazily = (run[ATOM_ARGS], lazy) if lazy is not None and ATOM_ARGS in run else ()
    backend = OpGraphBackend(
        machine,
        _op_graphs(
            run["store"], config, machine.value("device.arch"), *lazily, env=env
        ),
        tp=config.tensor_parallel_size,
        ep=ep,
        dp=config.parallel_config.data_parallel_size,
        kv=config.kv_cache_dtype,
        padded=run.get("padding", "executed") != "none",
        split_send_recv=config.enable_expert_parallel
        and getattr(config, "enable_low_latency", False),
        price_list=PRICE_LIST in run and _price_list(run, config, machine),
        drafts=drafter.num_speculative_tokens if drafter else 0,
        block=bool(drafter) and drafter.use_dspark(),
    )
    if lazy is None:
        return backend
    price = backend.price

    def timed(view):
        began = time.monotonic()
        try:
            return price(view)
        finally:
            spent = time.monotonic() - began
            lazy["price_wall_seconds"] = lazy.get("price_wall_seconds", 0) + spent

    backend.price = timed
    return backend


def _op_graphs(root: str, config, arch: str, atom_args=None, lazy=None, env=None):
    """The region of the eager `op_graph` of `config` on an `arch` device the
    store at `root` holds at a shape, of a step that produces output or one
    that does not, of a two-batch-overlap micro-batch (`micro_batch`), or of
    its drafter's part (`draft`, a block drafter's draft-KV pass where it is
    `forward.KV`), or when it holds none
    `forward.missed`'s text, None for no text; under DP, at the rows the
    step's MoE gathers; a prefill's drafter at the `width` its later passes
    pad to; keyed under the performance environment `env` when given, else
    this process's.

    With ATOM's arguments `atom_args`, a shape the store holds no graph at is
    first traced once, as `compass trace ops` traces a record's step from
    them, in the one process this lookup spawns at its first such shape
    (`_trace`), its count and wall seconds added to `lazy`'s ``lazy_traces``
    and ``lazy_trace_wall_seconds``; a trace that fails raises, naming the
    shape. A trace from a tree with uncommitted files answers this lookup alone
    and is not published.

    Under DP a prefill the store holds no graph at, with no drafter, reads
    the graph of its tokens, sequences and width that holds G
    (`forward.Tracer.symbolic`, keyed by `forward.symbolic_shape`) at the
    step's G, when its hints span G and its guards admit it (`peers.at`),
    counted in `lazy`'s ``held_lookups``; with `atom_args`, the first such
    step of a shape traces that graph, and when its two ends differ
    (`forward.PerG`) each G of the shape is traced as above instead, the
    shape and why named in `lazy`'s ``lazy_per_g``. With `lazy`, each
    lookup's wall seconds, its traces among them, are added to its
    ``lookup_wall_seconds``.

    Each graph read from the store is parsed once and then frozen out of the
    collector's reach with the rest of the live heap (`parsed`).

    A graph is looked up under the torch this process traces under, one whose
    files each read here as they did in its trace (`capture.lookup`), so one
    traced under another torch, or that ran a file since edited, is a miss
    like any other. One the store holds, at a shape or holding G, is
    loaded through `ArtifactStore.load` under this process's `graph_readings`,
    its files' reading here (`capture.source_of`),
    and its engine config as it records it: that cell holds the tracer's own
    ATOM arguments, which no reader restates, and the key's structure names
    the engine config in their place."""
    from atom.compass.artifacts import (
        ArtifactStore,
        Axis,
        Conditions,
        GateState,
        Row,
        Topology,
    )
    from atom.compass.capture import graph_key, lookup, peers, source_of
    from atom.compass.capture.forward import (
        PerG,
        Shape,
        admits,
        missed,
        structure,
        symbolic_shape,
    )
    from atom.utils import envs

    store, rank = ArtifactStore(root), Topology().ranks()[0]
    dp = config.parallel_config.data_parallel_size
    drafter = getattr(config, "speculative_config", None)
    drafts = drafter.num_speculative_tokens if drafter else 0
    traced, held_traced, unpublished, process = set(), set(), {}, []

    def lazily(shape, symbolic=False):
        """Trace `shape` (`_trace`), counted in `lazy`; a `symbolic` trace whose
        two ends differ is named in its ``lazy_per_g``."""
        began = time.monotonic()
        # `symbolic` only when set: a concrete trace's call is as before.
        kwargs = {"symbolic": True} if symbolic else {}
        try:
            if not process:
                process.append(_trace_process())
            done = process[0].submit(_trace, root, atom_args, arch, shape, **kwargs)
            unpublished.update(done.result())
        except PerG as differ:
            lazy.setdefault("lazy_per_g", {})[str(differ)] = 1
        # An argument parser's error exits.
        except (Exception, SystemExit) as failed:
            why = f"{type(failed).__name__}: {failed}"
            raise RuntimeError(f"a lazy trace of {shape} failed: {why}") from failed
        lazy["lazy_traces"] += 1
        lazy["lazy_trace_wall_seconds"] += time.monotonic() - began

    @functools.cache
    def here() -> dict:
        return graph_readings(config.model)

    def parsed(key) -> dict:
        """The graph the store holds at `key`, loaded under `here` and the
        engine config it records, parsed and frozen with the rest of the live
        heap, a collection first so no garbage is kept, as
        `gc_utils.freeze_gc_heap` freezes the startup heap and under the same
        `ATOM_GC_FREEZE`: this lookup's caches keep what it parses alive, and
        every full collection walked all of it."""
        cells = dict(store.read(key).fingerprints[Row.OP_GRAPH].cells)
        readings = here() | {"atom_src": source_of(store, key)}
        conditions = Conditions.of(**readings, engine_config=cells[Axis.ENGINE_CONFIG])
        entry = store.load(key, conditions=conditions, gates=GateState.of())
        graph = json.loads(entry.read_member("graph", rank, "json"))
        if envs.ATOM_GC_FREEZE:
            gc.collect()
            gc.freeze()
        return graph

    def read(at: str, shape: str) -> dict | None:
        """The graph the store, or this lookup's unpublished traces, hold at
        structure `at` and `shape`, as `capture.graph.to_json` writes it."""
        if (at, shape) in unpublished:
            return unpublished[at, shape]
        key = lookup(store, at, shape)
        return None if key is None else parsed(key)

    @functools.lru_cache(maxsize=64)
    def held(at: str, shape) -> dict | None:
        """The graph that holds G of the DP prefill `shape`, at any G, at
        structure `at`, or None."""
        text = symbolic_shape(shape)
        found = read(at, text)
        if found is None and atom_args is not None and shape not in held_traced:
            held_traced.add(shape)
            lazily(shape, symbolic=True)
            found = read(at, text)
        return found

    @functools.lru_cache(maxsize=64)
    def graph(
        phase,
        tokens,
        seqs,
        output=True,
        gathered=None,
        draft=False,
        width=None,
        micro_batch=False,
    ) -> dict | str | None:
        shape = Shape(phase, tokens, seqs, gathered, width, micro_batch).at(dp, drafts)
        at = structure(config, phase, True, arch, output, draft, env, micro_batch)
        key = lookup(store, at, str(shape))
        missing = key is None
        if missing and dp > 1 and phase == "prefill" and not (draft or drafts):
            g = shape.gathered or dp * tokens
            if found := held(at, shape._replace(gathered=None)):
                (lo, hi), guards = found["hints"], found["guards"]
                if lo <= g <= hi and admits(guards, g):
                    if lazy is not None:
                        lazy["held_lookups"] = lazy.get("held_lookups", 0) + 1
                    return peers.at(found["region"], found["slots"], g, dp)
        if missing and atom_args is not None and shape not in traced:
            traced.add(shape)
            lazily(shape)
            key = lookup(store, at, str(shape))
            missing = key is None
        if (at, str(shape)) in unpublished:
            return unpublished[at, str(shape)]["region"]
        if missing:
            return missed(store, graph_key(at, str(shape))) or None
        return parsed(key)["region"]

    if lazy is None:
        return graph

    def timed(*args, **kwargs):
        began = time.monotonic()
        try:
            return graph(*args, **kwargs)
        finally:
            spent = time.monotonic() - began
            lazy["lookup_wall_seconds"] = lazy.get("lookup_wall_seconds", 0) + spent

    return timed


def graph_readings(model: str) -> dict:
    """This process's readings, by axis, of an `op_graph` of the model `model`
    names, less its engine config, each as `compass trace ops` takes it: its
    torch and ATOM source (`capture.traced_under`), which a graph keyed by its
    files reads as they read here (`capture.source_of`), and the model's
    config.json; an axis no op_graph depends on is unread, of a kind no entry
    records."""
    from atom.compass.artifacts import Axis, Row, axes_of
    from atom.compass.capture import traced_under

    unread = Reading("unread", f"no {Row.OP_GRAPH} depends on it")
    return {a.field: unread for a in Axis if a not in axes_of(Row.OP_GRAPH)} | {
        **traced_under(),
        "model": _file_digest(model, "config.json"),
    }


def _trace_process():
    """A process of its own, spawned, for a lookup's lazy traces (`_trace`): a
    capture's process-wide patches stay in it, and the runner its first trace
    builds steps every later one. It ends when the process that spawned it
    does, however that ends: the runner takes SIGTERM for its own exit, not
    the process's."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from atom.utils import enable_orphan_reaping

    spawn = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(1, mp_context=spawn, initializer=enable_orphan_reaping)


@functools.cache
def _tracer(model: str, tp: int, atom_args: tuple, arch: str):
    """This process's `forward.Tracer` of `model` at TP `tp` under ATOM's engine
    arguments `atom_args` on a declared `arch`, made at its first lazy trace and
    kept for every later one, beside the directory its declared device lives
    in."""
    import tempfile

    from atom.compass.capture.forward import Tracer

    tmpdir = tempfile.TemporaryDirectory(prefix="compass-capture-")
    return tmpdir, Tracer(model, tp, tmpdir.name, atom_args, arch)


def _trace(root: str, atom_args, arch: str, shape, symbolic=False) -> dict:
    """Trace `shape` as `compass trace ops` traces a record's step, a prefill
    as one forward, or with `symbolic` a DP prefill's graphs that hold G
    (`forward.Tracer.symbolic`, which raises `forward.PerG` where they
    cannot), from ATOM's arguments `atom_args` less the server's, on a
    declared `arch`, in this process's tracer (`_tracer`), its KV pool the one
    a capture of `shape` alone allocates; and publish each graph in the store
    at `root`, one another engine of the run published first aside. From a
    tree with uncommitted files, which may run but not publish, return each
    graph by its structure and shape instead, as `capture.graph.to_json`
    writes it; else return an empty dict."""
    from atom.compass.artifacts import ArtifactRefusal, ArtifactStore, Rule, roots_for
    from atom.compass.capture import publish
    from atom.compass.capture.forward import symbolic_shape
    from atom.compass.capture.graph import to_json
    from atom.compass.drive.runfile import _engine_args, engine_only

    engine, kept = _engine_args(atom_args), engine_only(atom_args)
    store, tp = ArtifactStore(root), engine.tensor_parallel_size
    what = symbolic_shape(shape) if symbolic else shape
    command = f"compass run: a lazy trace of {what}"
    dirty = any(r.dirty for r in roots_for({"atom": "atom", "aiter": "aiter"}))
    unpublished = {}
    _, tracer = _tracer(engine.model, tp, tuple(kept), arch)
    graphs = tracer.symbolic(shape) if symbolic else tracer.trace([shape], [shape])
    for graph, model, declared in graphs:
        if dirty:
            at = graph.applicability
            unpublished[at.structure, at.shape] = json.loads(to_json(graph))
            continue
        try:
            publish(store, graph, model, declared, tp, kept, command)
        except ArtifactRefusal as refused:
            if refused.rule is not Rule.IMMUTABLE:
                raise
    return unpublished


def _price_list(run: dict, config, machine):
    """A `fit.ops.PriceList` of the `price_list` the run file names by its key's
    fields, loaded from its store under this run's readings, or a refusal naming
    the reading that differs."""
    from atom.compass.artifacts import ArtifactStore, GateState, Key, Kind
    from atom.compass.fit.ops import PriceList
    from atom.compass.run.step_grid import conditions

    key = Key.of(Kind.PRICE_LIST, **run[PRICE_LIST])
    entry = ArtifactStore(run["store"]).load(
        key, conditions=conditions(config, machine), gates=GateState.of()
    )
    rank = entry.topology.ranks()[0]
    return PriceList(json.loads(entry.read_member("params", rank, "json")))


def _graph_pool(model_runner, machine, model):
    """What ATOM's `_estimate_cudagraph_overhead` reserves for this deployment:
    nothing under ``enforce_eager``, else the branch the runner's own
    `_tr_hybrid_active` picks, over the same config fields it reads."""
    from atom.compass.memory import (
        PiecewiseCapture,
        capture_token_shapes,
        piecewise_per_token_bytes,
        reserves,
    )

    config = model_runner.config
    if config.enforce_eager:
        return reserves(enforce_eager=True)
    if not model_runner._tr_hybrid_active():
        return reserves(activations=model.activations)
    if getattr(config, "speculative_config", None):
        raise ValueError(
            "under a drafter ATOM's token-graph capture reserves for each draft "
            "q bucket, and its DP token cap, which this mirror does not take"
        )
    hf = config.hf_config
    sizes = config.compilation_config.cudagraph_capture_sizes or [config.max_num_seqs]
    capacity = machine.value("device.memory.capacity_bytes")
    capture = PiecewiseCapture(
        per_token_bytes=piecewise_per_token_bytes(
            hidden_size=int(hf.hidden_size),
            layers=int(hf.num_hidden_layers),
            dtype_bytes=config.torch_dtype.itemsize,
            dp_size=config.parallel_config.data_parallel_size,
        ),
        token_shapes=capture_token_shapes(
            sizes, max_num_batched_tokens=config.max_num_batched_tokens
        ),
        budget_bytes=int(config.gpu_memory_utilization * capacity),
    )
    return reserves(piecewise=capture)


def tokenizer(tok, config) -> None:
    """Wrap `tok`'s ``encode`` and ``decode``, and its Rust backend's
    ``encode_batch``, with the ``tokenizer`` term of this server's ``host``
    entry, its engine config's among several: a term fitted on another backend
    is refused, and one fitted on another ``tokenizer.json`` warns. A refusal
    ends the run before the server serves (`_end_before_serving`). With no
    term, `tok` is left as it is, charging 0 s; ``segment`` wraps the segment tokenizer,
    and its ``chars_per_s`` charges each comparison of its stored-prefix search.

    On a real run recording its steps (`parity.ENV`), it records instead
    (`_record_host`); with neither, `tok` is left as it is."""
    global _frontend_host
    run = spec()
    if run is None:
        from atom.compass import parity

        if record := os.environ.get(parity.ENV):
            _record_host(tok, Path(record))
        return
    from tokenizers import Tokenizer

    from atom.compass.spec.rules import FingerprintMismatch
    from atom.utils.compass_loop import wrap_decode, wrap_encode, wrap_encode_batch

    try:
        host, term = tokenizer_term(run, tok, config)
    except BaseException as e:
        _end_before_serving(run, e)
        raise
    if isinstance(run.get("host"), list):
        _frontend_host = host
    if term is None:
        return
    if (loaded := _fingerprint(tok)) not in (None, term["fingerprint"]):
        warnings.warn(
            f"the host laws' tokenizer was fitted on {term['fingerprint']}, but "
            f"the tokenizer.json now loaded hashes to {loaded}",
            FingerprintMismatch,
            stacklevel=2,
        )
    tok.encode = wrap_encode(tok.encode, term.get("encode"))
    tok.decode = wrap_decode(tok.decode, term.get("decode"))
    # ATOM's segment tokenizer, which only a Rust backend gets, encodes a long
    # prompt's uncached segments with its ``encode_batch``, not ``encode``.
    rust = getattr(tok, "_tokenizer", None)
    if isinstance(rust, Tokenizer):
        rust.encode_batch = wrap_encode_batch(
            rust.encode_batch, term.get("encode"), term.get("encode_batch")
        )
    if "segment" in term:
        from atom.model_engine.segment_tokenizer import SegmentTokenizer
        from atom.utils.compass_loop import charge, wrap_segment

        SegmentTokenizer._encode_cached = wrap_segment(
            SegmentTokenizer._encode_cached, term["segment"]
        )
        if "chars_per_s" in term["segment"]:
            _on_search(lambda n: charge(n / term["segment"]["chars_per_s"]))


def tokenizer_term(run: dict, tok, config) -> tuple[dict, dict | None]:
    """This server's ``host`` entry in the run file `run`, its engine config's
    among several as the runner picks its own (`pick_fitted`), and that entry's
    ``tokenizer`` term, None when it states none: a term fitted on another
    backend than the one `tok`, the tokenizer ATOM's server loaded, runs on is
    refused."""
    picked = {"host": run.get("host", {})}
    pick_fitted(picked, "host", config)
    host = picked["host"]
    term = host.get("tokenizer")
    if term is None:
        return host, None
    backend = "fast" if tok.is_fast else "slow"
    if term["backend"] != backend:
        raise ValueError(
            f"the host laws' tokenizer was fitted on the {term['backend']} "
            f"backend and the server loaded the {backend} one, which runs an "
            "order of magnitude apart on one tokenizer.json"
        )
    return host, term


def _record_host(tok, record: Path) -> None:
    """Record, beside the step record in `record`, each call of `tok`'s
    ``encode``, ``decode``, its Rust backend's ``encode_batch`` and ``segment``,
    with what its stored-prefix search compared, into `TOKENIZER_RECORD`, after
    a first line naming `tok`'s ``backend`` and the ``fingerprint`` of its
    ``tokenizer.json``; and stamp each HTTP request's intake into
    `INTAKE_RECORD` (`_stamp_intake`)."""
    from tokenizers import Tokenizer

    from atom.model_engine.segment_tokenizer import SegmentTokenizer

    record.mkdir(parents=True, exist_ok=True)
    calls = _LineFile(record / TOKENIZER_RECORD)
    backend = "fast" if tok.is_fast else "slow"
    calls.write(
        json.dumps({"backend": backend, "fingerprint": _fingerprint(tok)}) + "\n"
    )
    tok.encode = _recorded(tok.encode, "encode", calls, lambda _, ids: (len(ids), 1))
    tok.decode = _recorded(tok.decode, "decode", calls, lambda a, _: (len(a[0]), 1))
    rust = getattr(tok, "_tokenizer", None)
    if isinstance(rust, Tokenizer):
        rust.encode_batch = _recorded(
            rust.encode_batch,
            "encode_batch",
            calls,
            lambda _, got: (sum(len(e.ids) for e in got), len(got)),
        )
    _on_search(lambda n: setattr(_search, "chars", _search.chars + n))
    encode_cached = SegmentTokenizer._encode_cached

    def segment(*args, **kwargs):
        _search.chars = 0
        return encode_cached(*args, **kwargs)

    SegmentTokenizer._encode_cached = _recorded(
        segment,
        "segment",
        calls,
        lambda _, got: (
            len(getattr(got, "token_ids", got)),
            1,
            {"compared_chars": _search.chars},
        ),
    )
    _stamp_intake(_LineFile(record / INTAKE_RECORD))


def _record_routes(record: Path) -> None:
    """Write `parity.ROUTES` into `record` from the first request `CoreManager`
    routes across DP ranks, a line as each is routed; its key, as `parity.read`
    keys an uncached prompt, is digested before the router's lock. Installed
    beneath `parity.carry_response_ids`, which reads its caller's frame."""
    from atom.compass import parity
    from atom.model_engine.engine_core_mgr import CoreManager

    add, select = CoreManager.add_request, CoreManager._select_dp_rank_for_seq_locked
    lines = []

    def add_request(self, seqs):
        if self.pp_size > 1 or self._routable_engine_count == 1:
            return add(self, seqs)
        if not lines:
            lines.append(_LineFile(record / parity.ROUTES))
        keys = {}
        for seq in seqs:
            prompt = memoryview(seq.token_ids)[: seq.num_prompt_tokens]
            keys[seq.id] = hashlib.blake2b(prompt, digest_size=8).hexdigest()

        def routed(seq, explicit_rank):
            load = {"reqs": list(self._rank_reqs), "tokens": list(self._rank_tokens)}
            route = select(self, seq, explicit_rank)
            line = {"key": keys[seq.id], "rank": route.rank} | load
            lines[0].write(json.dumps(line) + "\n")
            return route

        self._select_dp_rank_for_seq_locked = routed
        try:
            return add(self, seqs)
        finally:
            del self._select_dp_rank_for_seq_locked

    CoreManager.add_request = add_request


class _LineFile:
    """A file of lines at `path`, each written whole and flushed as it is
    written, from any thread, into a file in the temp directory, which
    ``<path>.local`` names until `move`, at the server's stop, moves them to
    `path`. On NFS a line written right after another waits for the first's
    round trip, inside the request stage the line stamps, so a temp directory
    on a network file system refuses (`_local_dir`)."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        fd, self.local = tempfile.mkstemp(
            prefix=f"compass.{self.path.name}.", dir=_local_dir()
        )
        self.file = os.fdopen(fd, "w", buffering=1)
        self.lock = threading.RLock()  # a SIGTERM's move may interrupt a write
        self.path.unlink(missing_ok=True)
        self.path.with_name(self.path.name + LOCAL).write_text(f"{self.local}\n")
        _line_files.append(self)
        _move_at_stop()

    def write(self, line: str) -> None:
        with self.lock:
            if not self.file.closed:
                self.file.write(line)

    def move(self) -> None:
        """Move the lines to `path`; a line written after is dropped."""
        with self.lock:
            self.file.close()
        shutil.copyfile(self.local, self.path)
        self.path.with_name(self.path.name + LOCAL).unlink()
        os.remove(self.local)


#: The `_LineFile`s not yet moved to their record.
_line_files: list[_LineFile] = []
#: The SIGTERM handler `_move_at_stop` found, once it has set its own.
_sigterm_moving: list = []
#: The mount table `_local_dir` reads, and the file systems it refuses.
MOUNTINFO, NETWORK_FS = "/proc/self/mountinfo", ("nfs", "nfs4", "cifs", "smb3")


def _move_line_files() -> None:
    """Move each `_LineFile`'s lines to its record, as the server stops."""
    while _line_files:
        _line_files.pop().move()


def _move_at_stop() -> None:
    """Move the line files into their records at the process's exit, and at a
    SIGTERM, a stop's signal: uvicorn takes it while it serves and raises it
    again once it has stopped, after putting back the handler it found, which
    by default ends the process before any exit handler runs."""
    atexit.unregister(_move_line_files)
    atexit.register(_move_line_files)
    if _sigterm_moving or threading.current_thread() is not threading.main_thread():
        return
    previous = signal.getsignal(signal.SIGTERM)
    if previous in (signal.SIG_IGN, None):  # None: not set from Python
        return
    signal.signal(signal.SIGTERM, functools.partial(_stop_moving, previous))
    _sigterm_moving.append(previous)


def _stop_moving(previous, signum, frame) -> None:
    """A SIGTERM: move the line files, then `previous` takes it, the default
    delivered again so that the process ends of the signal as it would have."""
    try:
        _move_line_files()
    finally:
        signal.signal(signum, previous)
        if callable(previous):
            previous(signum, frame)
        else:
            os.kill(os.getpid(), signum)


def _local_dir() -> str:
    """The temp directory, refused when `MOUNTINFO` puts it, or the upper
    directory of the overlay holding it, on a network file system."""
    where = os.path.realpath(tempfile.gettempdir())
    point, fs, source, options = _mount(where)
    upper = dict(o.partition("=")[::2] for o in options.split(",")).get("upperdir")
    if fs not in NETWORK_FS and upper and _mount(upper)[0] != point:
        _, fs, source, _ = _mount(upper)
    if fs in NETWORK_FS:
        raise ValueError(
            f"the stamp lines' temp directory {where} is on {fs} ({source}), where "
            "a line written right after another waits for the first's round trip; "
            "set TMPDIR to a directory on this host's own disk or a tmpfs"
        )
    return where


def _mount(path: str) -> tuple[str, str, str, str]:
    """The mount point, file system, source and options of the mount
    `MOUNTINFO` lists as holding `path`."""
    held = ("", "", "", "")
    with open(MOUNTINFO) as table:
        for line in table:
            left, _, right = line.partition(" - ")
            point = codecs.decode(left.split()[4], "unicode_escape")
            fs, source, options = (right.split() + ["", "", ""])[:3]
            inside = path == point or path.startswith(point.rstrip("/") + "/")
            if inside and len(point) >= len(held[0]):
                held = (point, fs, source, options)
    return held


#: Whether this thread is inside a `_recorded` ``encode`` call.
_in_encode = threading.local()
#: The characters the stored-prefix search of this thread's recorded segment
#: tokenizer call compared.
_search = threading.local()


def _on_search(then) -> None:
    """Call `then` with the characters each comparison of ATOM's segment
    tokenizer's stored-prefix search compares: what the prompt shares with one
    stored prompt (``segment_tokenizer._shared_chars``)."""
    from atom.model_engine import segment_tokenizer

    shared_chars = segment_tokenizer._shared_chars

    def compared(a, b, limit):
        n = shared_chars(a, b, limit)
        then(n)
        return n

    segment_tokenizer._shared_chars = compared


def _recorded(call, kind: str, out: _LineFile, count):
    """`call`, writing to `out` per call a JSON line of `kind`, its ``tokens``
    and ``items`` as `count(args, result)` gives them, with the fields of any
    mapping it gives after them, its ``thread`` id and its
    start and end on the monotonic clock in ns; a call made inside a recorded
    ``encode`` call is that call's, as `compass_loop.wrap_encode_batch` charges
    it."""

    def timed(*args, **kwargs):
        if getattr(_in_encode, "on", False):
            return call(*args, **kwargs)
        _in_encode.on = kind == "encode"
        start = time.monotonic_ns()
        try:
            got = call(*args, **kwargs)
        finally:
            _in_encode.on = False
        end = time.monotonic_ns()
        tokens, items, *more = count(args, got)
        more = "".join(f', "{k}": {v}' for m in more for k, v in m.items())
        out.write(
            f'{{"call": "{kind}", "tokens": {tokens}, "items": {items}{more}, '
            f'"thread": {threading.get_ident()}, '
            f'"t_start_ns": {start}, "t_end_ns": {end}}}\n'
        )
        return got

    return timed


def _stamp_intake(out: _LineFile) -> None:
    """Stamp each HTTP request's intake into `out` as
    ``scripts/compass/http_intake.py serve`` does, one line
    ``<end ns> <thread id> <kind> <start ns> <size> <end cpu ns> <start cpu ns>``
    each, on the monotonic clock and the stamping thread's CPU clock: ``app``,
    the request handed to the ASGI app, size 0; ``body``, its body first read,
    size its bytes; ``tmpl``, a chat template rendered, size its characters;
    ``prep``, a tokenizer pool job, and ``send``, a request handed to the
    engine, size 0; on the loop from an await's start to the coroutine's
    resume, size 0, ``recv``, each receive of the body's first read, and
    ``wait``, the API server's tokenizer pool job, from its submit; ``out``,
    each row of an engine output its output thread took in, and
    ``head``, an SSE response's head sent, size 0. Patches the
    classes, and every loaded module's binding of ``apply_chat_template``, for
    the rest of the process."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from starlette.applications import Starlette
    from starlette.requests import Request

    from atom.compass import parity
    from atom.entrypoints.openai import chat_encoders
    from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
    from atom.model_engine.llm_engine import InputOutputProcessor

    def stamp(kind: str, start: int, size: int, cpu: int) -> None:
        out.write(
            f"{time.monotonic_ns()} {threading.get_ident()} {kind} {start} {size} "
            f"{time.thread_time_ns()} {cpu}\n"
        )

    def timed(call, kind: str, size=lambda got: 0):
        def timed(*args, **kwargs):
            start, cpu = time.monotonic_ns(), time.thread_time_ns()
            got = call(*args, **kwargs)
            stamp(kind, start, size(got), cpu)
            return got

        return timed

    read, handle = Request.body, Starlette.__call__

    def waited(receive):
        async def received():
            start, cpu = time.monotonic_ns(), time.thread_time_ns()
            try:
                return await receive()
            finally:
                stamp("recv", start, 0, cpu)

        return received

    async def body(self):
        cached, receive = hasattr(self, "_body"), self._receive
        start, cpu = time.monotonic_ns(), time.thread_time_ns()
        if not cached:
            self._receive = waited(receive)
        try:
            got = await read(self)
        finally:
            self._receive = receive
        if not cached:
            stamp("body", start, len(got), cpu)
        return got

    async def app(self, scope, receive, send):
        if scope["type"] == "http":
            stamp("app", time.monotonic_ns(), 0, time.thread_time_ns())
            send = heads(send)
        await handle(self, scope, receive, send)

    def heads(send):
        async def sent(message):
            start, cpu = time.monotonic_ns(), time.thread_time_ns()
            await send(message)
            if message["type"] == "http.response.start":
                kind = dict(message.get("headers", [])).get(b"content-type", b"")
                if kind.startswith(b"text/event-stream"):
                    stamp("head", start, 0, cpu)

        return sent

    Request.body, Starlette.__call__ = body, app
    # The API server imports the function by name, so each binding is replaced.
    render = chat_encoders.apply_chat_template
    rendered = timed(render, "tmpl", len)
    for module in list(sys.modules.values()):
        if getattr(module, "__dict__", {}).get("apply_chat_template") is render:
            module.apply_chat_template = rendered
    fanout = InputOutputProcessor.preprocess_fanout
    InputOutputProcessor.preprocess_fanout = timed(fanout, "prep")

    def sent(add):
        def add_request(*args, **kwargs):
            # `parity.carry_response_ids`, installed beneath this as the engine
            # was built, reads the API server's `request_id` off its caller's
            # frame, which is now this one. Found as it finds it, up to
            # `parity.CALLER_DEPTH` frames above: since the upstream merge the
            # caller is `_submit_engine_requests`, which holds none, and a
            # `request_id = None` here hid the handler's from it.
            request_id = parity._caller_request_id()  # noqa: F841
            start, cpu = time.monotonic_ns(), time.thread_time_ns()
            got = add(*args, **kwargs)
            stamp("send", start, 0, cpu)
            return got

        return add_request

    for cls in (CoreManager, DisaggCoreManager):
        cls.add_request = sent(cls.add_request)
    # Each called once per row of an output, as the queue handoff counts them
    # (`clock.PullStamps`): a streamed one's, and a finished sequence's.
    for name in ("_mark_seq_prefill_complete", "_deliver_terminal_output"):
        setattr(CoreManager, name, timed(getattr(CoreManager, name), "out"))

    class Resumed(asyncio.Future):
        def __await__(self):
            try:
                return (yield from super().__await__())
            finally:
                stamp("wait", *self.since)

    submit = ThreadPoolExecutor.submit

    def submitted(self, fn, /, *args, **kwargs):
        since = time.monotonic_ns(), 0, time.thread_time_ns()
        job = submit(self, fn, *args, **kwargs)
        if getattr(fn, "__name__", None) != "do_preprocess":
            return job
        # The loop hands the API server's `do_preprocess` to its executor, and
        # wraps a returned future of its own loop's as it is.
        resumed = Resumed(loop=asyncio.get_running_loop())
        resumed.since = since
        asyncio.futures._chain_future(job, resumed)
        return resumed

    ThreadPoolExecutor.submit = submitted


def _end_before_serving(run: dict, e: BaseException) -> None:
    """`e`, raised after the engines started and before the server serves, ends
    the run in progress: stop it, as a signal stops the frontend's event loop, so
    each engine leaves its loop at its ``+inf`` grant, hands in and its process
    ends, which the server's exit waits for; then hand in `e` as the frontend's
    ``ended:`` reason, which a standalone authority's summary waits for."""
    rt = clock.installed()
    if rt is None or not rt.in_run:
        return
    rt.end_run()
    _hand_in(run, rt.me.name, rt.refusals + [f"ended:{rt.me.name} raised {e!r}"])


def http_intake() -> tuple[float, float, float] | None:
    """``(fixed_s, per_byte_s, idle_s)``, the frontend loop's time to take in one
    HTTP request, from the ``http_intake`` term of this server's ``host`` entry,
    the one `tokenizer` resolved among several, ``fixed_s`` standing for an
    unstated ``idle_s``; None on a real run or with no term."""
    if (term := _intake_term()) is None:
        return None
    return term["fixed_s"], term["per_byte_s"], term.get("idle_s", term["fixed_s"])


def relay() -> tuple[float, float, float] | None:
    """``(fixed_s, per_byte_s, router_s)``: the relay's time to carry one of
    this prefill server's responses to decode's app call, from the ``relay``
    term of its ``host`` entry, the one `tokenizer` resolved among several,
    and the lookahead of the relay's path through the router, which the
    channels price already; None on a real run or with no term.
    `atom.utils.compass_loop.HttpChannel` writes ``fixed_s + n x per_byte_s -
    router_s``, where that is positive, on a relayed response of `n` bytes as
    its relay's service time, and decode's `HttpChannel` charges it on its
    loop at the request's turn, queued with the relays and intakes in flight
    there. A run that is not prefill-decode relays nothing: None."""
    run = spec()
    if run is None or "router_s" not in run:
        return None
    if (term := _host_term(RELAY_TERM)) is None:
        return None
    return term["fixed_s"], term["per_byte_s"], hop(run, "router_s")


def http_stream() -> float | None:
    """``per_event_s``: the frontend loop's time to send one SSE event of a
    streamed response, from the ``http_stream`` term of this server's ``host``
    entry, the one `tokenizer` resolved among several; None on a real run or
    with no term. `atom.utils.compass_loop.HttpChannel` charges it on the loop
    before it stamps each event, queued with the requests it takes in, as a
    decode server's loop at c=256 (#3720, r256) spends ~50 us an event, a
    fifth of its wall time."""
    term = _host_term(STREAM_TERM)
    return None if term is None else term["per_event_s"]


def http_intake_hull() -> tuple[float, float] | None:
    """The body bytes, ``(lo, hi)``, the ``http_intake`` term `http_intake`
    reads was fitted over, outside which `HttpChannel` refuses a request's
    intake; None on a real run, with no term, or a term stating none."""
    term = _intake_term() or {}
    return tuple(term["bytes"]) if "bytes" in term else None


def _intake_term() -> dict | None:
    return _host_term("http_intake")


def _host_term(name: str) -> dict | None:
    """The term `name` of this server's ``host`` entry, the one `tokenizer`
    resolved among several; None on a real run or with no term."""
    run = spec()
    if run is None:
        return None
    host = run.get("host", {})
    if isinstance(host, list):
        if _frontend_host is None:
            raise ValueError(
                "the run file holds a host entry per engine config, and none was "
                "resolved for this server's: compass_run.tokenizer resolves it"
            )
        host = _frontend_host
    return host.get(name)


def _fingerprint(tok) -> str | None:
    """``sha256:<hex>`` of the ``tokenizer.json`` a fast tokenizer was loaded
    from, resolved from its ``name_or_path`` as loading did; None without one."""
    name = getattr(tok, "name_or_path", None)
    if not (tok.is_fast and name):
        return None
    return _file_digest(name, "tokenizer.json")


def host_cpu() -> str:
    """This host's CPU model, as /proc/cpuinfo names it, else as
    `platform.processor` does."""
    cpu = re.search(
        r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.MULTILINE
    )
    return cpu.group(1) if cpu else platform.processor()


def host_key(config) -> dict:
    """What host laws are keyed by: this host's CPU, ATOM's source less
    `COMPASS_ONLY`, the served config.json and the engine config a step grid
    bundle is keyed by."""
    from atom.compass.run.step_grid import keyed

    return host_key_of(config.model, keyed(config))


def host_key_of(model: str, engine: dict) -> dict:
    """`host_key` for the model `model` names and the engine config `engine`
    that `keyed` gave, for a process that holds no resolved config."""
    return {
        "host_cpu": host_cpu(),
        "atom": atom_digest(),
        "model": _file_digest(model, "config.json"),
        "engine": engine,
    }


def atom_source(digest: str | None = None) -> Reading:
    """ATOM's source as every artifact reads it: `atom_digest`, here or as a
    record's run.json ``host_key.atom`` states it in `digest`, a reading of
    kind `ATOM_PY`."""
    return Reading(ATOM_PY, atom_digest() if digest is None else digest)


def source_root(digest: str | None = None) -> str:
    """The ``source_root`` a key holds for `atom_source`, ``<kind>-<value>``."""
    reading = atom_source(digest)
    return f"{reading.kind}-{reading.value}"


def atom_digest(root: Path | None = None) -> str:
    """The sha256 of ATOM's Python modules here, or in the package directory
    `root`, less `COMPASS_ONLY` and `INSTALL_WRITTEN`."""
    root = Path(atom.__file__).parent if root is None else root
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if (text := atom_text(path, relative)) is not None:
            digest.update(f"{relative}\0".encode() + text)
    return digest.hexdigest()


def atom_text(path: Path, relative: Path) -> bytes | None:
    """The bytes of ATOM's file at `path`, `relative` under its package, as
    `atom_digest` takes them: less the names `COMPASS_ONLY` leaves out; None
    for a file it leaves out whole and for `INSTALL_WRITTEN`."""
    if relative == INSTALL_WRITTEN:
        return None
    skip = next((v for k, v in COMPASS_ONLY.items() if relative.is_relative_to(k)), ())
    if skip is None:
        return None
    lines = path.read_bytes().splitlines(keepends=True)
    tree = ast.parse(b"".join(lines)) if skip else None
    doc = tree.body[0] if tree and ast.get_docstring(tree) is not None else None
    body = tree.body if tree else []
    for prev, node in reversed(list(zip([None, *body], body))):
        named = (node, *getattr(node, "targets", ()))
        names = {getattr(n, "name", getattr(n, "id", None)) for n in named}
        if node is doc or names & skip:
            del lines[prev.end_lineno if prev else 0 : node.end_lineno]
    return b"".join(lines)


def committed_atom_digest(repo: Path, commit: str) -> str:
    """`atom_digest` of the ``atom/`` that `commit` holds in the git repository
    at `repo`, as `git archive` ships it."""
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", commit, "atom"],
        capture_output=True,
        check=False,
    )
    if archive.returncode:
        raise ValueError(
            f"git archive {commit} atom in {repo}: {archive.stderr.decode().strip()}"
        )
    with tempfile.TemporaryDirectory(prefix="compass-head-") as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(tmp, filter="data")
        return atom_digest(Path(tmp) / "atom")


def _file_digest(name: str, filename: str) -> str | None:
    """``sha256:<hex>`` of a model's `filename`, resolved from `name` as loading
    does; None when it is not there."""
    from transformers.utils import cached_file

    try:
        path = cached_file(name, filename, local_files_only=True)
    except OSError:
        return None
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_whole(path: Path, text: str) -> None:
    """Write `path` whole or not at all, so a reader that polls for it never
    reads it part-written."""
    part = path.with_name(path.name + ".part")
    part.write_text(text)
    part.replace(path)


def _hand_in(
    run: dict, lp: str, reasons: list, client_run: int | None = None, **sections
) -> None:
    """Write `lp`'s refusals, and its other sections, for the summary's writer:
    at the end of client run `client_run`, or None for the finish. When its
    runtime printed stall diagnostics, ``diagnostics`` counts them. At the
    finish, first the Prometheus series its runtime sampled, if it did, with a
    final sample at the LP's clock, or at the clock it held before its
    ``+inf`` grant; a final sample that raises is skipped and named among
    `reasons`."""
    out = Path(run["out_dir"])
    if client_run is not None:
        out = out / CLIENT_DIR.format(client_run)
        out.mkdir(exist_ok=True)
    elif (sampler := getattr(clock.installed(), "sampler", None)) is not None:
        rt = sampler.rt
        t = rt.now if rt.now < math.inf else rt.before_inf
        try:
            sampler.final(t)
        except Exception as e:  # noqa: BLE001 - skipped, named among its reasons
            reasons = [*reasons, f"metrics:{lp} final sample at {t} raised {e!r}"]
        sampler.write(out, lp)
    if diagnostics := getattr(clock.installed(), "diagnostics", 0):
        sections = {"diagnostics": diagnostics, **sections}
    _write_whole(
        out / LP_FILE.format(lp), json.dumps({"refusals": reasons, **sections})
    )


def _workers_say(engine_core) -> dict:
    """The workers' command refusals, what their steps cost and their memory."""
    workers = engine_core.runner_mgr
    return {
        "refusals": list(workers.call_func("refused_commands", wait_out=True)),
        "priced": dict(workers.call_func("priced_steps", wait_out=True)),
        "memory": workers.call_func("predicted_memory", wait_out=True),
    }


def engine_done(engine_core, **sections) -> None:
    """After the engine's loop: leave the clock and hand in."""
    clock.close()
    run = spec()
    if run is not None:
        _engine_hand_in(run, engine_core, sections)


def _engine_hand_in(run: dict, engine_core, sections: dict, client_run=None) -> None:
    """Hand in what the worker's steps cost and the memory it predicts, with
    `sections`, its command refusals and the engine's refused clock calls; a
    DP member that parked also hands in ``parks``: their count, CPU seconds, and
    each park's clock and the clock it left at; a DP member hands in ``owed``,
    `LPRuntime.paid`. An engine whose offload
    scheduler keeps lookup counters hands in ``lookups``: its tier lookups,
    the prefetches among them, and the deferred prefetches it stopped
    waiting for. A speculative engine hands in ``speculative``, `_acceptance`.
    An engine whose ``lmcache_offload`` or ``mooncake_store`` is simulated
    hands in ``offload``: `OffloadShell.offload_statistics` or
    `StoreShell.offload_statistics`, and under ``cache`` its prefix cache's
    counters (`EngineStats.cache_statistics`)."""
    said = _workers_say(engine_core)
    rt = clock.installed()
    lp = rt.me.name if _member is None else f"{rt.me.name}.{_member}"
    reasons = said.pop("refusals") + rt.refusals
    if parks := getattr(rt, "parks", None):
        cpu_s, at = sum(p[2] for p in parks), [p[:2] for p in parks]
        sections = sections | {"parks": {"count": len(parks), "cpu_s": cpu_s, "at": at}}
    if rt.member is not None:
        sections = sections | {"owed": dict(rt.paid)}
    kv = getattr(getattr(engine_core, "scheduler", None), "kv_connector", None)
    if (counts := getattr(kv, "_perf_counters", None)) is not None:
        keys = ("lookup_n", "lookup_prefetched", "lookup_defer_expired")
        sections = sections | {"lookups": {k: counts[k] for k in keys}}
    scheduler = getattr(engine_core, "scheduler", None)
    if getattr(getattr(scheduler, "engine_stats", None), "spec_enabled", False):
        sections = sections | {"speculative": _acceptance(scheduler, run)}
    for half in getattr(kv, "_connectors", None) or [kv]:
        # A later pipeline stage's Store shell is inert and states no tier.
        tier = getattr(half, "offload_statistics", None)
        if callable(tier) and (stated := tier()):
            cache = scheduler.engine_stats.cache_statistics()
            sections = sections | {"offload": stated | {"cache": cache}}
    _hand_in(run, lp, reasons, client_run, **said, **sections)


def _acceptance(scheduler, run: dict) -> dict:
    """A speculative engine's drafter, its draft width, the acceptance input
    ATOM's flag stated with its method, the per-position rates it resolved to,
    and the scheduler's accepted-draft histogram (`EngineStats.distribution`).
    The input is ``measured`` from a run file's `ACCEPTANCE_RECORD`."""
    spec = scheduler.config.speculative_config
    # A stated rate also resolves a length, so it is read first.
    stated = (
        ("--spec-decode-acceptance-rate", spec.synthetic_acceptance_rate),
        ("--spec-decode-acceptance-length", spec.synthetic_acceptance_length),
        ("--spec-decode-acceptance-rates", spec.synthetic_acceptance_rates),
    )
    flag, value = next((f, v) for f, v in stated if v is not None)
    said = {"flag": flag, "value": value, "method": "declared"}
    if record := run.get(ACCEPTANCE_RECORD):
        said |= {"method": "measured", "record": record}
    return {
        "drafter": spec.method,
        "draft_tokens": spec.num_speculative_tokens,
        "input": said,
        "rates": spec.synthetic_acceptance_rates,
        "distribution": dict(scheduler.engine_stats.distribution),
    }


#: The methods ATOM's engine loops run a step through: a pipeline head's, and
#: every other engine's.
STEP_METHODS = ("_pp_head_step", "_process_engine_step")


def _hit_counters(scheduler) -> list:
    """``cached_tokens`` and ``full_tokens`` (`EngineStats.cache_statistics`)
    and the KV connector's ``loaded_tokens``, 0 with none, read as ATOM's
    ``/metrics`` reads them (`EngineUtilityHandler.collect_metrics`)."""
    cache = scheduler.engine_stats.cache_statistics()
    kv = getattr(scheduler, "kv_connector", None)
    tier = kv.get_statistics() if hasattr(kv, "get_statistics") else {}
    return [cache["cached_tokens"], cache["full_tokens"], tier.get("loaded_tokens", 0)]


def _sample_hit_counters(engine_core) -> list | None:
    """The series of `engine_core`'s hit counters on its LP clock, cumulative
    from the server's start: ``[t, *_hit_counters]`` at the run's start, and
    after each step that changed one, through its `STEP_METHODS`, wrapped on
    this engine alone. None for an engine with no scheduler statistics."""
    scheduler = getattr(engine_core, "scheduler", None)
    if getattr(scheduler, "engine_stats", None) is None:
        return None
    rt = clock.installed()
    series = [[rt.now, *_hit_counters(scheduler)]]

    def sampled(step):
        @functools.wraps(step)
        def stepped(*args, **kwargs):
            out = step(*args, **kwargs)
            if (now := _hit_counters(scheduler)) != series[-1][1:]:
                series.append([rt.now, *now])
            return out

        return stepped

    for name in STEP_METHODS:
        if hasattr(engine_core, name):
            setattr(engine_core, name, sampled(getattr(engine_core, name)))
    return series


def _in_force_from(series: list, t: float) -> list:
    """`series`' samples from the one in force at `t`, the last at or before
    it, on."""
    return [s for s in series if s[0] <= t][-1:] + [s for s in series if s[0] > t]


def _block_file(now: Path) -> Path:
    """Where the engine whose clock file is `now` publishes its planned KV block,
    the state slot and staging row a transfer moves and the state image an
    lmcache save carries (`publish_transfer`)."""
    return now.with_suffix(".kv_block")


def _raised_file(now: Path, rank) -> Path:
    """Where worker `rank` of the engine whose clock file is `now` writes what
    it raised; ``*`` globs every rank's."""
    return now.with_suffix(f".{rank}.raised")


#: A run's switch (``1``) for stamping each scheduled batch with the requests its
#: engine's scheduler had taken (`_stamp_arrivals`), which a host law's queue
#: column prices from (`step_labels.QueueDepth`, `runner.overrides`).
QUEUE_ENV = "COMPASS_HOST_QUEUE"


def _stamp_arrivals(engine_core) -> None:
    """Each batch `engine_core`'s scheduler schedules carries
    ``compass_arrived``, the requests the scheduler had taken (every `add`, and
    each of an `extend`'s, counted here: ATOM's own ``_arrived_requests``
    restarts at each schedule) as it began scheduling it; a stage with no
    scheduler stamps none."""
    scheduler = getattr(engine_core, "scheduler", None)
    if scheduler is None or not hasattr(scheduler, "schedule"):
        return
    taken = [0]
    schedule, add, extend = scheduler.schedule, scheduler.add, scheduler.extend

    def counted_add(seq, *args, **kwargs):
        taken[0] += 1
        return add(seq, *args, **kwargs)

    def counted_extend(seqs, *args, **kwargs):
        seqs = list(seqs)
        taken[0] += len(seqs)
        return extend(seqs, *args, **kwargs)

    def stamped(*args, **kwargs):
        arrived = taken[0]
        result = schedule(*args, **kwargs)
        if result is not None and result[0] is not None:
            result[0].compass_arrived = arrived
        return result

    scheduler.add, scheduler.extend, scheduler.schedule = counted_add, counted_extend, stamped


@contextlib.contextmanager
def engine_loop(engine_core):
    """Around the engine's loop: `engine_done` once it leaves, which raises if
    that was before the finish. If either raises in a run, hand in the engine's
    refused clock calls and the exception, and what the workers say in 10 wall
    seconds unless one died. A dead worker reaches the engine as a bare
    `SystemExit`, so its ``ended:`` reason carries what each worker rank wrote
    it raised. Under ``step_grid`` the workers are asked for the bundle
    before the loop, so a run that ends early still hands it in. In a run the
    engine hands in ``hit_counters`` too (`_sample_hit_counters`), and with a
    utility handler samples its Prometheus series before each metrics push
    (`metrics.sample_engine`).

    In a run the engine ignores SIGTERM until it has handed in: ATOM's handler
    would stop the workers inside a clock call, before they say what their
    steps cost. A stop reaches the engine as the ``+inf`` grant instead. The
    engine has ignored SIGTERM since it said READY (`engine`); the loop
    ignores it first too, before the run file's read, a pipeline stage's sends
    or the bundle's request, and however it leaves puts back the handler set
    aside at READY, or else the one it found.

    A pipeline stage's runner calls go through `pp_send.install`'s sends. A
    recorded real run's SIGTERM goes through `_stop_recording`, its input
    queue stamps each request frame's handoff (`clock.stamp_input`), and
    however it leaves the loop, the stamps its engine held are written
    (`clock.write_stamps`), as its workers then write their step lines."""
    ignoring = bool(os.environ.get(ENV))  # a run, as `spec` would say, unread
    before = signal.signal(signal.SIGTERM, signal.SIG_IGN) if ignoring else None
    if _sigterm_at_ready:
        before = _sigterm_at_ready.pop()
    try:
        from atom.compass import parity

        run, sections = spec(), {}
        if run is not None and getattr(engine_core, "pp_size", 1) > 1:
            from atom.compass.runner import pp_send

            pp_send.install(engine_core, _machine(run))
        if run is not None and os.environ.get(QUEUE_ENV) == "1":
            _stamp_arrivals(engine_core)
        if run is not None and STEP_GRID in run:
            workers = engine_core.runner_mgr
            bundle = workers.call_func("step_grid_bundle", wait_out=True)
            sections["step_grid"] = bundle
        if run is not None:
            if (series := _sample_hit_counters(engine_core)) is not None:
                sections["hit_counters"] = series
            sampler = getattr(clock.installed(), "sampler", None)
            if sampler is not None and hasattr(engine_core, "utility_handler"):
                from atom.compass.run import metrics

                metrics.sample_engine(sampler, engine_core)
            clock.installed().client_run_ended = lambda k: _engine_hand_in(
                run, engine_core, sections, k
            )
        elif os.environ.get(parity.ENV):
            clock.stamp_input(engine_core)
            stop = functools.partial(
                _stop_recording, engine_core, signal.getsignal(signal.SIGTERM)
            )
            before = signal.signal(signal.SIGTERM, stop)
    except BaseException:
        if before is not None:  # None: not set from Python
            signal.signal(signal.SIGTERM, before)
        raise
    try:
        yield
        engine_done(engine_core, **sections)
    except BaseException as e:
        if run is not None:
            box = []
            if not isinstance(e, SystemExit):
                ask = threading.Thread(
                    target=lambda: box.append(_workers_say(engine_core)), daemon=True
                )
                ask.start()
                ask.join(10.0)
            said = box[0] if box else {}
            rt = clock.installed()
            lp = rt.me.name if _member is None else f"{rt.me.name}.{_member}"
            reasons = said.pop("refusals", []) + rt.refusals
            ended = f"ended:{lp} raised {e!r}"
            if isinstance(e, SystemExit):
                now = Path(run["out_dir"]) / NOW_FILE.format(lp)
                for f in sorted(now.parent.glob(_raised_file(now, "*").name)):
                    ended += f"; {f.read_text()}"
            _hand_in(run, lp, reasons + [ended], **said, **sections)
        raise
    finally:
        try:
            clock.write_stamps()
        finally:
            if before is not None:  # None: not set from Python
                signal.signal(signal.SIGTERM, before)


def _stop_recording(engine_core, previous, signum, frame) -> None:
    """A recorded real run's SIGTERM, where its engine stops: it writes its
    held stamps, a speculative engine writes its draft width and
    accepted-draft histogram beside the record (`ACCEPTANCE_FILE`), then ATOM's
    handler, `previous`, takes it, whether or not the writes succeeded."""
    from atom.compass import parity

    scheduler = getattr(engine_core, "scheduler", None)
    config, stats = getattr(scheduler, "config", None), getattr(
        scheduler, "engine_stats", None
    )
    try:
        try:
            clock.write_stamps()
        finally:
            if getattr(stats, "spec_enabled", False):
                width = config.speculative_config.num_speculative_tokens
                held = {"draft_tokens": width, "distribution": dict(stats.distribution)}
                name = ACCEPTANCE_FILE.format(config.parallel_config.data_parallel_rank)
                _write_whole(Path(os.environ[parity.ENV]) / name, json.dumps(held))
    finally:
        previous(signum, frame)


def _until_end(run: dict, a: "_RecordingAuthority") -> bool:
    """Wait for the finish, or for an LP to hand in before it; True for the
    latter, an early end. Gives up, False, once this process's main thread ends."""
    out = Path(run["out_dir"])
    while not a.done.wait(0.1):
        # Every LP hands in after the finish sets `done`, so read it second.
        if any(out.glob(LP_FILE.format("*"))) and not a.done.is_set():
            return True
        if not threading.main_thread().is_alive():
            return False
    return False


def _write_on_early_end(run: dict, a: "_RecordingAuthority") -> None:
    if _until_end(run, a):
        _write_summary(run, a, ended=True)


def frontend_done(llm_engine) -> bool:
    """Whether the server left at a simulated run's finish, or at a stop: a
    signal that ended its event loop's run. At either, stop the engines and
    hand in the frontend's refusals, at a stop with the signal that ended it.
    With the authority co-hosted, then write the step table and the run summary;
    at a stop, after the early end's writer, which has written its own if an LP
    handed in before the stop, so the summary left counts every hand-in. A
    recorded real run's served lines move into its record first (`_LineFile`)."""
    _move_line_files()
    rt = clock.installed()
    stop = None if rt is None else getattr(rt.loop, "stopped", None)
    if rt is None or (rt.now != math.inf and stop is None):
        return False
    llm_engine.close()
    run = spec()
    _frontend_hand_in(run, rt, stopped=stop)
    if _authority is not None:
        if stop is not None:
            _early_end.join()
        early = stop is not None and not _authority.stopped_whole
        _write_summary(run, _authority, ended=early)
    return True


def _frontend_hand_in(run: dict, rt, client_run=None, stopped=None) -> None:
    """Hand in the frontend's executor's refused jobs and its refused clock
    calls, and the signal that stopped it, if one did."""
    reasons = rt.loop.executor.refusals + rt.refusals
    sections = {} if stopped is None else {"stopped": stopped}
    _hand_in(run, rt.me.name, reasons, client_run, **sections)


def _read_hand_ins(run: dict, where: Path, wait: bool) -> dict:
    """Every server LP's hand-in in `where`, by name; with `wait`, once all are."""
    files = {n: where / LP_FILE.format(n) for n in _hand_ins(run)}
    deadline = time.monotonic() + HAND_IN_WAIT_S
    while wait and (missing := [f.name for f in files.values() if not f.exists()]):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"no {', '.join(missing)} in {where} after {HAND_IN_WAIT_S} wall "
                "seconds, so the run summary would miss those LPs' refusals"
            )
        time.sleep(0.1)
    return {n: json.loads(f.read_text()) for n, f in files.items() if f.exists()}


def _write_summary(run: dict, a: _RecordingAuthority, ended: bool = False) -> None:
    """Write the step table, then the run summary once every server LP has
    handed in its refusals, or after an early end, from those handed in so far;
    after the finish, unless a stop brought it, the last client run's summary
    too."""
    out = Path(run["out_dir"])
    _write_whole(out / STEP_TABLE_FILE, a.steps.text())
    hand_ins = _read_hand_ins(run, out, not ended)
    summary = _summary(run, a, a, a.wall(), hand_ins, {})
    _write_whole(out / SUMMARY_FILE, json.dumps(summary, sort_keys=True, indent=1))
    if not ended and a.ended and not a.stopped:
        _write_client_run(run, a, len(a.ended) - 1, hand_ins)


def _write_client_run(run: dict, a: _RecordingAuthority, k: int, hand_ins=None):
    """Write client run `k`'s summary from the hand-ins at its end, those in its
    directory unless given, less those at client run ``k - 1``'s end, and
    under ``hit_counters`` each engine's from the sample in force at its start."""
    out = Path(run["out_dir"])
    here = out / CLIENT_DIR.format(k)
    if hand_ins is None:
        hand_ins = _read_hand_ins(run, here, True)
    since = _read_hand_ins(run, out / CLIENT_DIR.format(k - 1), False) if k else {}
    record = a.ended[k]
    since_wall = a.wall(k - 1) if k else 0.0
    summary = _summary(run, a, record, a.wall(k), hand_ins, since, since_wall)
    summary["client_run"] = {"index": k, "start": record.start, "end": record.end}
    hits = {
        n: _in_force_from(h["hit_counters"], record.start)
        for n, h in hand_ins.items()
        if "hit_counters" in h
    }
    if hits:
        summary["hit_counters"] = hits
    here.mkdir(exist_ok=True)
    _write_whole(here / SUMMARY_FILE, json.dumps(summary, sort_keys=True, indent=1))


def _summary(
    run: dict, a, record, wall: float, hand_ins: dict, since: dict, since_wall=0.0
) -> dict:
    """The summary of `record`, the authority or one client run, over the steps
    the hand-ins count beyond those in `since`; `since_wall` is the part of
    `wall` the client runs `since` counts served."""
    from atom.compass.backends.cost import GROUPS

    reasons = Counter(r for h in hand_ins.values() for r in h["refusals"])
    # A frontend's stop ends the run early, unless it left the run whole.
    if not a.stopped_whole:
        stops = [(n, h["stopped"]) for n, h in hand_ins.items() if "stopped" in h]
        reasons.update(f"ended:{n} stopped by {sig}" for n, sig in stops)
    reasons -= Counter(r for h in since.values() for r in h["refusals"])
    reasons = sorted(reasons.elements())
    priced = [h["priced"] for h in hand_ins.values() if "priced" in h]
    before = [h["priced"] for h in since.values() if "priced" in h]

    def total(key):
        return sum(p.get(key, 0) for p in priced) - sum(p.get(key, 0) for p in before)

    def counted(key):
        c = sum((Counter(p.get(key, {})) for p in priced), Counter())
        return c - sum((Counter(p.get(key, {})) for p in before), Counter())

    seconds = total("seconds")
    tally = RefusalTally.of(reasons, total("steps"))
    lazy = total("lazy_traces"), total("lazy_trace_wall_seconds")
    # The authority's stalls in `record`, and each LP's beyond those in `since`.
    diagnostics = record.diagnostics + sum(
        h.get("diagnostics", 0) for h in hand_ins.values()
    )
    diagnostics -= sum(h.get("diagnostics", 0) for h in since.values())
    summary = RunSummary.of(
        record, wall, *lazy, diagnostics=diagnostics, refusals=tally
    ).as_record()
    # Each DP prefill whose G were each traced, and why, by the engines that did.
    if per_g := counted("lazy_per_g"):
        summary["cost"]["lazy_per_g"] = dict(per_g)
    # The DP prefill lookups a graph holding G served, and every lookup's wall.
    for key in ("held_lookups", "lookup_wall_seconds"):
        if value := total(key):
            summary["cost"][key] = value
    # Each DP group's wall waited on lookups, on the rest of pricing, and on
    # neither, by the engine of its first rank, over `record`'s own wall.
    waited = {}
    for n, h in hand_ins.items():
        if w := h.get("priced", {}).get("group_waited_seconds"):
            gone = since.get(n, {}).get("priced", {}).get("group_waited_seconds")
            lookups, pricing = (a - b for a, b in zip(w, gone or (0, 0)))
            rest = wall - since_wall - lookups - pricing
            waited[n] = {"lookups": lookups, "pricing": pricing, "rest": rest}
    if waited:
        summary["cost"]["group_waited_seconds"] = waited
    schedule = summary["schedule"]
    schedule["predicted_seconds"] = seconds
    # How the seconds were obtained: reported, never gated.
    schedule["seconds_by_method"] = {
        g: {
            "steps": total(f"{g}_steps"),
            "fraction_of_predicted_seconds": (
                total(f"{g}_seconds") / seconds if seconds else 0.0
            ),
        }
        for g in GROUPS
    }
    # Steps answered past a source's support domain, and below a source that declined.
    for key in ("extrapolated_steps", "fallback_steps"):
        schedule[key] = total(key)
    # Each assumed treatment the steps were priced under, with its step count.
    schedule["treatments"] = dict(sorted(counted("treatments").items()))
    # Each reason a source declined a step a fallback then priced, with its steps.
    schedule["fallback_reasons"] = dict(counted("fallback_reasons").most_common())
    # The steps extrapolated on each DP group step's critical rank, apart from
    # every rank's, off the step records in out_dir, over a client run's own
    # span; or why there are none.
    from atom.compass.report import extrapolation

    span = None if record is a else (record.start, record.end)
    try:
        schedule[extrapolation.FIELD] = extrapolation.of_run(run["out_dir"], span)
        schedule[extrapolation.REFUSED] = None
    except (OSError, ValueError) as refused:
        schedule[extrapolation.FIELD] = None
        schedule[extrapolation.REFUSED] = str(refused)
    # An engine whose workers died, or that never handed in, priced steps no
    # one counted: they are absent, not 0.
    if any(
        "priced" not in hand_ins.get(n, {})
        for n in _hand_ins(run)
        if n.startswith("engine")
    ):
        schedule.update(predicted_seconds=None, seconds_by_method=None, treatments=None)
        schedule.update(extrapolated_steps=None, fallback_steps=None)
        schedule["fallback_reasons"] = None
        schedule["refusals"].update(steps=None, fraction_of_steps=None)
    summary["configuration"] = configuration(run)
    summary["uncharged_host_terms"] = sorted(
        {t for h in entries(run, "host") or [{}] for t in HOST_TERMS if t not in h}
        | {
            f"tokenizer.{k}"
            for h in entries(run, "host") or []
            for k in TOKENIZER_KINDS
            if k not in h.get("tokenizer", {k: None})
        }
        | {
            "tokenizer.segment.chars_per_s"
            for h in entries(run, "host") or []
            if "segment" in h.get("tokenizer", {})
            and "chars_per_s" not in h["tokenizer"]["segment"]
        }
        | (
            {RELAY_TERM}
            if "router_s" in run
            and not any(RELAY_TERM in h for h in entries(run, "host"))
            else set()
        )
    )
    summary["unmeasured_hops"] = sorted(k for k, v in hops(run).items() if v is None)
    summary["schema_version"] = SUMMARY_SCHEMA_VERSION
    summary["source_root"] = a.source_root.as_json()
    if "router" in run:
        summary["router"] = run["router"]
    # Any refusal, of any source, leaves part of the run unsimulated; a run
    # that ended early handed in why.
    summary["coverage_report"] = bool(reasons)
    if grids := {n: h["step_grid"] for n, h in hand_ins.items() if "step_grid" in h}:
        summary["step_grid"] = grids
    # Each speculative engine's acceptance input, and what it accepted over these steps.
    speculative = {}
    for n, h in hand_ins.items():
        if "speculative" in h:
            gone = since.get(n, {}).get("speculative", {}).get("distribution", {})
            seen = {
                a: c - gone.get(a, 0)
                for a, c in h["speculative"]["distribution"].items()
            }
            rows = sum(seen.values())
            mean = 1 + sum(int(a) * c for a, c in seen.items()) / rows if rows else None
            speculative[n] = h["speculative"] | {
                "distribution": seen,
                "mean_acceptance_length": mean,
            }
    if speculative:
        summary["speculative"] = speculative
    # Each offloading engine's tier and copies over these steps, and the hit
    # rates ATOM's counters give: prefix-cache hits and tier loads, each over
    # the prompt tokens its prefix cache counted.
    offload = {}
    for n, h in hand_ins.items():
        # A later pipeline stage's Store shell is inert and counts nothing: the
        # head's holds the stages' tier (`StoreShell.offload_statistics`).
        if h.get("offload"):
            seen = _counted_since(h["offload"], since.get(n, {}).get("offload", {}))
            full = seen["cache"]["full_tokens"]
            gpu, cpu = (
                seen["cache"]["cached_tokens"],
                seen["connector"]["loaded_tokens"],
            )
            offload[n] = seen | {
                "gpu_hit_rate": gpu / full if full else None,
                "cpu_hit_rate": cpu / full if full else None,
            }
    if offload:
        summary["offload"] = offload
    return summary


#: Counters of an ``offload`` hand-in that are levels, not counts: reported as
#: they stand, never less an earlier client run's.
OFFLOAD_LEVELS = frozenset(
    {"held_chunks", "loads_pending", "saves_pending", "blocks_waiting_for_store"}
)


def _counted_since(now: dict, then: dict) -> dict:
    """`now`'s counts less `then`'s, nested alike; a level as it stands."""
    return {
        k: (
            _counted_since(v, then.get(k, {}))
            if isinstance(v, dict)
            else v if k in OFFLOAD_LEVELS else v - then.get(k, 0)
        )
        for k, v in now.items()
    }


def authority(endpoint: str) -> None:
    """Name `endpoint` in the out_dir's `AUTHORITY_FILE`, then serve the run
    file's authority there until the finish or an early
    end, then write its step table, and the run summary once the server LPs
    have handed in."""
    run = spec()
    a = _RecordingAuthority(run)
    _write_whole(Path(run["out_dir"]) / AUTHORITY_FILE, endpoint)
    serve(a, endpoint)
    _write_summary(run, a, _until_end(run, a))
