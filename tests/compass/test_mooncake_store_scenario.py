# SPDX-License-Identifier: MIT
"""Two prefill instances at DP2 share one LP mooncake_store under `share_cache`,
one loading the prefix the other saved, through `pd_sim.sh`.

DeepSeek-V4.1-Flash as the 6P16D C1228 recipe's prefill instance (``-tp 2
--enable-dp-attention``, prefix caching on, ``--dp-load-balance share_cache``),
two of them, and one decode instance at DP2. `pd_sim.sh` starts each prefill
with `PREFILL_ENV` selecting ``ATOM_KV_OFFLOAD=mooncake_store``; the router
takes the prefill instances in turn. The run file's ``mooncake_store`` entry
declares its laws, matches the deployment, and holds `CAPACITY` chunks with
`TTL_MS` leases, so the master evicts.

The workload, one request every `GAP` seconds (`ORDER`): family 0 on P0, which
saves its chunks; family 0 on P1, which loads them; four other families, whose
saves fill the Store past its mark; family 0 last, whose lookup finds its
groups gone. Three runs: seeds 1 and 2, and seed 1 again, byte for byte.

Needs `ATOM_COMPASS_DSPARK_MODEL` and atomesh, built from this tree, on PATH;
no driver.
"""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest
from test_dp_slice import DP_MACHINE
from test_pd_slice import PD
from test_pd_xpyd import DP2_OFFSETS, DSV41_ARGS
from test_vertical_slice import (
    CONSTANT,
    TREE,
    WALL_S,
    Traffic,
    _run_file,
    publish_entry_on_arch,
)

from atom.compass import run as compass_run
from atom.compass.carriers import tracestate_with
from atom.compass.detect.scenario import check_run
from atom.compass.kv import mooncake_store_server as server
from atom.compass.parity.traffic import bodies
from atom.kv_transfer.offload.mooncake_store.pool import _SLOT_ALIGN, _round_up
from atom.kv_transfer.offload.mooncake_store.worker import (
    MooncakeStoreOffloadConnector,
)
from atom.utils import envs
from tests.compass.held_ports import held_ports

DSV41 = os.environ.get("ATOM_COMPASS_DSPARK_MODEL")
NEEDS_A_RUN = pytest.mark.skipif(
    not (DSV41 and shutil.which("atomesh")),
    reason="a run needs ATOM_COMPASS_DSPARK_MODEL and atomesh on PATH",
)
CHUNK = 256  # tokens a Store chunk holds: two of the recipe's 128-token blocks
#: What a rank of DeepSeek-V4.1-Flash plans a 128-token KV block at bf16 and
#: one rank to take, which LP mooncake_store checks the entry's chunk against.
BLOCK_BYTES = 369920
CHUNK_BYTES = BLOCK_BYTES * CHUNK // 128
POOL_MIB = 64  # each of a rank's save and load pools
CAPACITY = 12  # chunks the owners hold
TTL_MS = 50  # a lease, in Mooncake's milliseconds
STORE = {
    "mooncake_store.master": "127.0.0.1:26051",
    "mooncake_store.metadata": "http://127.0.0.1:26080/metadata",
    "mooncake_store.local_hostname": "127.0.0.1",
    "mooncake_store.protocol": "tcp",
    "mooncake_store.chunk_tokens": CHUNK,
    "mooncake_store.save_pool_mib": POOL_MIB,
    "mooncake_store.load_pool_mib": POOL_MIB,
}
#: Each prefill's environment, word for word as `pd_sim.sh` reads it.
PREFILL_ENV = {
    "ATOM_KV_OFFLOAD": "mooncake_store",
    "ATOM_KV_OFFLOAD_EXTRA_CONFIG": json.dumps(STORE, separators=(",", ":")),
    "ATOM_FRONTEND_PREFIX_HASH": "1",
    "OFFLOAD_MIN_LOAD_TOKENS": str(CHUNK),
    "OFFLOAD_MIN_SAVE_TOKENS": str(CHUNK),
}
PREFILL_ARGS = "--enable_prefix_caching --dp-load-balance share_cache"
HOP = 2.0**-12  # half the lookup's fixed term: one hop to or from the master
ENTRY = {
    "lookup_s": [2 * HOP, 2.0**-16],
    "store_s": [2.0**-7, 2.0**-9],
    "load_s": [2.0**-7, 2.0**-9],
    "protocol": "tcp",
    "chunk_tokens": CHUNK,
    "chunk_bytes": CHUNK_BYTES,
    "world": 1,
    "save_slots": (POOL_MIB << 20) // _round_up(CHUNK_BYTES, _SLOT_ALIGN),
    "load_slots": (POOL_MIB << 20) // _round_up(CHUNK_BYTES, _SLOT_ALIGN),
    "save_workers": envs.OFFLOAD_COPY_WORKERS,
    "load_workers": envs.OFFLOAD_LOAD_WORKERS,
    "master": [
        "--rpc_thread_num=4",
        f"--default_kv_lease_ttl={TTL_MS}",
        "--eviction_high_watermark_ratio=0.5",
        "--eviction_ratio=0.5",
    ],
    "capacity_bytes": CAPACITY * CHUNK_BYTES,
}
GAP, FAMILIES, PREFIX_WORDS = 0.5, 5, 600
#: The request of `bodies` sent each `GAP` s: family 0 twice, the others, and
#: family 0 again.
ORDER = (0, FAMILIES, 1, 2, 3, 4, 2 * FAMILIES)


class Workload(Traffic):
    """The traffic LP, through the router: request k of `ORDER` at k * `GAP`."""

    http, stream = "traffic->router:http", "router->traffic:stream"

    def __init__(self, requests, *args) -> None:
        super().__init__(*args)
        self.requests = requests

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        for k, body in enumerate(self.requests):
            while rt.now < k * GAP:
                rt.next_event(k * GAP)
                self._take()
            self.sent, seq = rt.stamp_send(self.http)
            stamp = tracestate_with(None, self.sent, seq)
            self._carry("POST", "/v1/completions", body, stamp, self.port)
        while rt.next_event(float("inf")) != float("inf"):
            self._take()
        rt.close()


def run_store(out: Path, seed: str) -> dict:
    """One 2P1D run at DP2 under `pd_sim.sh`, the prefills on the Store."""
    out.mkdir()
    flags = DSV41_ARGS
    env_words = " ".join(f"{k}={v}" for k, v in PREFILL_ENV.items())
    cell = {
        "MODEL": DSV41,
        "SERVER_ARGS": flags,
        "PREFILL_ARGS": PREFILL_ARGS,
        "PREFILL_ENV": env_words,
        "ROUTER_POLICY": "round_robin",
    }
    with held_ports(1, lambda q: range(q[0] + 1, q[0] + 72)) as [base]:
        ports = {name: base + k for name, k in DP2_OFFSETS.items()}
        run_file = _run_file(
            out,
            clock_endpoint=f"tcp://127.0.0.1:{ports['CLOCK_PORT']}",
            step_stub=CONSTANT,
            machine=DP_MACHINE,
            prefill_instances=2,
            decode_instances=1,
            data_parallel_size=2,
            bound_s=60.0,
            **PD,
            **{server.ENTRY: ENTRY},
        )
        env = dict(
            os.environ,
            PYTHONHASHSEED=seed,
            PYTHONPATH=str(TREE),
            AITER_LOG_LEVEL="WARNING",
            ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
            **cell,
            **{name: str(port) for name, port in ports.items()},
        )
        for side, role, more, words in (
            ("prefill", "kv_producer", PREFILL_ARGS, PREFILL_ENV),
            ("decode", "kv_consumer", "", {}),
        ):
            kv = json.dumps({"kv_role": role, "kv_connector": "compass"})
            argv = ["--model", DSV41, *flags.split(), *more.split()]
            argv += ["--kv-transfer-config", kv]
            publish_entry_on_arch(out / "store", DP_MACHINE, argv, env | words)
        requests = [
            bodies(DSV41, 2 * FAMILIES + 1, PREFIX_WORDS, families=FAMILIES)[k]
            for k in ORDER
        ]
        log = out / "pd_sim.log"
        with open(log, "w") as f:
            sim = subprocess.Popen(
                ["bash", str(TREE / "scripts/compass/pd_sim.sh"), str(run_file)],
                cwd=TREE,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        watchdog = threading.Timer(WALL_S * 2, os.killpg, (sim.pid, signal.SIGKILL))
        watchdog.start()
        try:
            deadline = time.monotonic() + WALL_S * 2
            while "pd_sim: ready" not in log.read_text():
                assert sim.poll() is None, f"pd_sim.sh exited:\n{log.read_text()}"
                assert time.monotonic() < deadline, "no router in time"
                time.sleep(0.5)
            run = json.loads(run_file.read_text())
            traffic = Workload(requests, run, ports["ROUTER_PORT"], sim)
            traffic.run()
            code = sim.wait(timeout=120)
        finally:
            watchdog.cancel()
            if sim.poll() is None:
                os.killpg(sim.pid, signal.SIGKILL)
    logs = {p.stem: p.read_text() for p in out.glob("*.log")}
    assert code == 0, f"pd_sim.sh exited {code}:\n{logs['pd_sim'][-4000:]}"
    table = out / compass_run.STEP_TABLE_FILE
    return {
        "out": out,
        "table": table.read_text(),
        "digest": hashlib.sha256(table.read_bytes()).hexdigest(),
        "summary": json.loads((out / compass_run.SUMMARY_FILE).read_text()),
        "hand_ins": {
            p.stem[3:]: json.loads(p.read_text()) for p in out.glob("lp-*.json")
        },
        "logs": logs,
        "sent": [r["max_tokens"] for r in requests],
        "taken": [line for _, line in traffic.events],
    }


@pytest.fixture(scope="module")
def store_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("mooncake_store")
    return [
        run_store(root / name, seed)
        for name, seed in (("seed1", "1"), ("seed2", "2"), ("seed1again", "1"))
    ]


#: A query's reply, as the shell logs it.
QUERY = re.compile(
    r"compass: mooncake_store query of (\S+) from (\S+): (\d+) of its (\d+) keys "
    r"present, (\d+) put by another engine, lease to \S+; first key \S+/([0-9a-f]+)"
)
ROUND = re.compile(
    r"eviction round at (\S+): .*evicted (\d+) objects of groups (\[.*?\])"
)


@NEEDS_A_RUN
def test_two_prefill_instances_share_the_store_one_loading_the_other_s_save(
    store_runs,
):
    """Both seeds meet every scenario check, and seed 1 run twice gives one
    step table; a prefix P0 saved loads on P1; an eviction round runs and family
    0's last lookup misses the groups it evicted; each prefill rank's KV budget
    holds the Store worker's reserve."""
    runs = store_runs
    parity = runs[0]["out"] / "parity"
    records = {
        "engine-P0": parity / "prefill0",
        "engine-P1": parity / "prefill1",
        "engine-D": parity / "decode",
    }
    failed = check_run(
        [r["out"] for r in runs[:2]], runs[0]["sent"], runs[0]["taken"], records
    )
    assert not failed, "\n".join(failed)
    digests = [r["digest"] for r in runs]
    print("\nstep table digests:", dict(zip(("seed1", "seed2", "seed1again"), digests)))
    assert digests[0] == digests[2]

    for run in runs[:2]:
        lp = run["hand_ins"]["mooncake_store"]
        shells = {
            n: h["offload"]["store"]
            for n, h in run["hand_ins"].items()
            if "offload" in h
        }
        print(f"\n{run['out'].name}: LP mooncake_store {lp}\nshells {shells}")
        assert lp["refusals"] == []
        assert lp["store"]["foreign_hit_keys"] > 0 and lp["store"]["evicted"] > 0
        assert sum(s["foreign_loads"] for s in shells.values()) > 0
        assert sum(s["past_lease_loads"] for s in shells.values()) == 0

        prefills = "\n".join(
            t for n, t in run["logs"].items() if n.startswith("prefill")
        )
        queries = sorted(
            (float(m[2]), m[6], int(m[3]), int(m[5])) for m in QUERY.finditer(prefills)
        )
        rounds = [(float(m[1]), m[3]) for m in ROUND.finditer(prefills)]
        first = queries[0][1]  # family 0's first chunk, which P0 asked first
        family0 = [q for q in queries if q[1] == first]
        print(
            "family 0's queries (start, present, foreign):",
            [(q[0], q[2], q[3]) for q in family0],
        )
        print("eviction rounds:", [(t, g.count("group/")) for t, g in rounds])
        evicted = min(t for t, groups in rounds if first in groups)
        assert family0[0][2] == 0  # nothing saved yet
        assert any(q[2] and q[3] for q in family0 if q[0] < evicted)  # P0's save, on P1
        assert family0[-1][0] > evicted and family0[-1][2] == 0

        reserve = MooncakeStoreOffloadConnector.kv_budget_reserve_bytes(
            type(
                "Config",
                (),
                {"kv_transfer_config": {"kv_connector_extra_config": STORE}},
            )
        )
        said = f"Reserving {reserve / (1 << 30):.2f}GB for KV connector buffers"
        for name, text in run["logs"].items():
            if name.startswith("prefill"):
                assert text.count(said) == 2, f"{name}: no `{said}` per rank"
