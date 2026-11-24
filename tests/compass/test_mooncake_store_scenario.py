# SPDX-License-Identifier: MIT
"""Two prefill instances at DP2 share one LP `mooncake_store` under
`share_cache`, one loading the other's saved prefix, and an eviction round
drops a prompt past its lease (#3806).

DeepSeek-V4.1-Flash runs as two prefill instances of ROCm/ATOM's 6P16D C1228
share-cache recipe's form (`-tp 2 --enable-dp-attention`, `--dp-load-balance
share_cache`) and one decode instance at DP2, through `pd_sim.sh`, under a
constant step stub. Each prefill side's environment (`PREFILL_ENV`) selects
``ATOM_KV_OFFLOAD=mooncake_store``; the run file's entry states declared laws,
a lease of `TTL_MS` and a Store of `CAPACITY_CHUNKS` chunks whose rounds run
above `MARK`, small enough to evict. The router LP takes the prefill instances in turn.

The workload (`TIMES`): a family-0 prompt, saved by the instance that serves
it; the same prompt on the other instance, which loads it from the Store;
`FILLERS` prompts of other families, which take the Store over its mark so an
eviction round runs, past family 0's lease; and family 0 once more, whose
lookup misses the evicted groups.

The runs need `ATOM_COMPASS_DSPARK_MODEL` (V4.1's config and tokenizer, as
`test_spec_scenario` takes it) and atomesh on PATH, built from this tree; they
read readings published for each side and need no driver, as
`test_lmcache_mp_scenario`'s V4.1 runs do.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import pytest
import test_dp_slice
from test_dp_slice import DP_MACHINE
from test_lmcache_mp_scenario import Workload
from test_pd_dp import RouterLpRequests
from test_pd_sim_router_args import RECIPE
from test_pd_slice import PD
from test_pd_xpyd import DP2_OFFSETS
from test_vertical_slice import (
    CONSTANT,
    TREE,
    WALL_S,
    _run_file,
    publish_entry_on_arch,
)

from atom.compass import run as compass_run
from atom.compass.carriers import tracestate_with
from atom.compass.detect.scenario import check_run
from atom.compass.kv import mooncake_store_server as store
from atom.compass.parity.traffic import bodies
from tests.compass.held_ports import held_ports

DSV41 = os.environ.get("ATOM_COMPASS_DSPARK_MODEL")
NEEDS_A_STORE_RUN = pytest.mark.skipif(
    not (DSV41 and shutil.which("atomesh")),
    reason="a run needs ATOM_COMPASS_DSPARK_MODEL and atomesh on PATH",
)
#: Every server: eager, ROCm/ATOM main's recipe's (5effb67d4) KV and index
#: dtypes, block size (V4.1 resolves 16 to 128) and MoE path.
SERVER_ARGS = "--trust-remote-code -tp 2 --enable-dp-attention --enforce-eager"
SERVER_ARGS += " --max-model-len 2048 --kv_cache_dtype fp8 --index-cache-dtype fp8"
SERVER_ARGS += " --block-size 16 --moe-backend standard"
SERVER_ARGS += " --all2all-backend high-throughput --enable_prefix_caching"
#: The recipe's prefill: EP over its two ranks, and its policy over the tier
#: every rank of both instances shares.
PREFILL_ARGS = "--enable-expert-parallel --dp-load-balance share_cache"
#: The recipe's environment beside it: MoRI on, fp8 MoE activations off.
RECIPE_ENV = {"ATOM_DSV41_MOE_FP8_ACTIVATIONS": "0", "NCCL_IB_DISABLE": "0"}
CHUNK = 256
#: DeepSeek-V4.1-Flash's planned 128-token KV block at fp8 KV and an fp8
#: index, as its runner publishes it, two to a chunk.
CHUNK_BYTES = 2 * 211200
POOL_MIB = 64
SLOTS = (POOL_MIB << 20) // (-(-CHUNK_BYTES // 4096) * 4096)
CAPACITY_CHUNKS = 24
#: Rounds run above half the Store, so each family's put, at most a few
#: chunks, fits in what a round leaves: a put the capacity cannot hold is
#: refused (Mooncake fails it, and ATOM's failed-save path is not modelled).
MARK = 0.5
TTL_MS = 100
STORE_EXTRA = {
    "mooncake_store.master": "127.0.0.1:26051",
    "mooncake_store.metadata": "http://127.0.0.1:26080/metadata",
    "mooncake_store.protocol": "tcp",
    "mooncake_store.local_hostname": "127.0.0.1",
    "mooncake_store.chunk_tokens": CHUNK,
    "mooncake_store.save_pool_mib": POOL_MIB,
    "mooncake_store.load_pool_mib": POOL_MIB,
}
#: Each prefill server's own variables (PREFILL_ENV): the Store, one copy and
#: one load worker, and a load from one chunk.
PREFILL_ENV = {
    "ATOM_KV_OFFLOAD": "mooncake_store",
    "ATOM_KV_OFFLOAD_EXTRA_CONFIG": json.dumps(STORE_EXTRA, separators=(",", ":")),
    "OFFLOAD_COPY_WORKERS": "1",
    "OFFLOAD_LOAD_WORKERS": "1",
    "OFFLOAD_MIN_LOAD_TOKENS": str(CHUNK),
    "OFFLOAD_MIN_SAVE_TOKENS": str(CHUNK),
}
#: The Store's run-file entry: declared laws, the recipe's master with a short
#: lease, and the owners' bytes `CAPACITY_CHUNKS` chunks.
ENTRY = {
    "lookup_s": [2.0**-11, 2.0**-17],
    "store_s": [2.0**-9, 2.0**-12],
    "load_s": [2.0**-9, 2.0**-12],
    "protocol": "tcp",
    "chunk_tokens": CHUNK,
    "chunk_bytes": CHUNK_BYTES,
    "world": 1,
    "save_slots": SLOTS,
    "load_slots": SLOTS,
    "save_workers": 1,
    "load_workers": 1,
    "master": [
        "--rpc_port=26051",
        "--rpc_thread_num=32",
        f"--default_kv_lease_ttl={TTL_MS}",
        f"--eviction_high_watermark_ratio={MARK}",
        "--eviction_ratio=0.05",
    ],
    "capacity_bytes": CAPACITY_CHUNKS * CHUNK_BYTES,
}
PREFIX_WORDS = 600  # two or three whole 256-token chunks
FAMILIES = 8
FILLERS = FAMILIES - 1
#: When each request is sent: family 0, family 0 again on the other instance,
#: one of each other family, then family 0 once more.
TIMES = [0.0, 1.0, *(2.0 + 0.5 * k for k in range(FILLERS)), 3.0 + 0.5 * FILLERS]
#: Each request's family.
FAMILY = [0, 0, *range(1, FAMILIES), 0]
#: The router LP takes the prefill instances in turn, and ATOM's share_cache
#: the rank within one: no --dp-aware, which would name the rank and leave
#: share_cache only accounting.
ROUTER = RECIPE.replace("--prefill-policy cache_aware", "--prefill-policy round_robin")
#: A query's reply, as the Store shell logs it.
QUERY = re.compile(
    r"compass: mooncake_store query of (\S+) at (\S+): chunk 0 (\w+), "
    r"(\d+) of (\d+) keys present, (\d+) another instance's"
)


class StoreWorkload(Workload):
    """The traffic LP through the router LP: request k at ``TIMES[k]``."""

    http, stream = RouterLpRequests.http, RouterLpRequests.stream

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        for k, body in enumerate(self.requests):
            while rt.now < TIMES[k]:
                rt.next_event(TIMES[k])
                self._take()
            self.sent, seq = rt.stamp_send(self.http)
            stamp = tracestate_with(None, self.sent, seq)
            self._carry("POST", "/v1/completions", body, stamp, self.port)
        while rt.next_event(math.inf) != math.inf:
            self._take()
        rt.close()


def _publish(store_dir: Path, machine: dict, env: dict) -> None:
    """Each side's readings, its config resolved as `pd_sim.sh` starts it, the
    prefill side's under its own variables: its key names the Store."""
    for side, role, own in (
        ("PREFILL", "kv_producer", PREFILL_ENV),
        ("DECODE", "kv_consumer", {}),
    ):
        kv = json.dumps({"kv_role": role, "kv_connector": "compass"})
        argv = ["--model", DSV41, *SERVER_ARGS.split()]
        argv += env.get(f"{side}_ARGS", "").split()
        publish_entry_on_arch(
            store_dir, machine, [*argv, "--kv-transfer-config", kv], env | own
        )


def run_store(out: Path, seed: str) -> dict:
    """One 2P1D run under `pd_sim.sh`, both prefill instances on the Store."""
    out.mkdir()
    with held_ports(1, lambda q: range(q[0] + 1, q[0] + 72)) as [base]:
        ports = {name: base + k for name, k in DP2_OFFSETS.items()}
        run_file = _run_file(
            out,
            clock_endpoint=f"tcp://127.0.0.1:{ports['CLOCK_PORT']}",
            step_stub=CONSTANT,
            prefill_instances=2,
            decode_instances=1,
            data_parallel_size=2,
            machine=DP_MACHINE,
            **PD,
            **{store.ENTRY: ENTRY},
        )
        env = dict(
            {k: v for k, v in os.environ.items() if k != "ATOM_DISABLE_MORI_EP"},
            **RECIPE_ENV,
            PYTHONHASHSEED=seed,
            PYTHONPATH=str(TREE),
            AITER_LOG_LEVEL="WARNING",
            ATOM_COMPASS_PARITY_RECORD=str(out / "parity"),
            MODEL=DSV41,
            SERVER_ARGS=SERVER_ARGS,
            PREFILL_ARGS=PREFILL_ARGS,
            PREFILL_ENV=" ".join(f"{k}={v}" for k, v in PREFILL_ENV.items()),
            ROUTER_ARGS=f"{ROUTER} --model-path {DSV41}",
            **{name: str(port) for name, port in ports.items()},
        )
        _publish(out / "store", DP_MACHINE, env)
        # One prompt per family, its number and then the words every family
        # shares (`bodies`' family form), all of one length; each request its
        # family's, its output a few tokens.
        [body] = bodies(DSV41, 1)
        words = bodies(DSV41, 1, PREFIX_WORDS)[0]["prompt"].split()[:PREFIX_WORDS]
        requests = [
            body | {"prompt": f"{f} " + " ".join(words), "max_tokens": 8 + k}
            for k, f in enumerate(FAMILY)
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
            traffic = StoreWorkload(requests, run, ports["ROUTER_PORT"], sim)
            traffic.run()
            code = sim.wait(timeout=120)
        finally:
            watchdog.cancel()
            if sim.poll() is None:
                os.killpg(sim.pid, signal.SIGKILL)
    assert code == 0, f"pd_sim.sh exited {code}:\n{log.read_text()[-4000:]}"
    return {
        "out": out,
        "run": run,
        "table": (out / compass_run.STEP_TABLE_FILE).read_text(),
        "sent": [r["max_tokens"] for r in requests],
        "taken": [line for _, line in traffic.events],
    }


@pytest.fixture(scope="module")
def store_runs(tmp_path_factory):
    if not DSV41:
        pytest.skip("a run needs ATOM_COMPASS_DSPARK_MODEL")
    root = tmp_path_factory.mktemp("mooncake_store")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(test_dp_slice, "MODEL", DSV41)
        return [run_store(root / f"seed{s}", s) for s in ("1", "2")]


def _hand_ins(out: Path) -> dict:
    return {p.name[3:-5]: json.loads(p.read_text()) for p in out.glob("lp-*.json")}


def _queries(out: Path) -> list[tuple]:
    """Each query reply a prefill engine took, in its clock's order:
    (clock, instance, first chunk's digest, present, asked, another's)."""
    found = []
    for k in (0, 1):
        for line in (out / f"prefill{k}.log").read_text().splitlines():
            if m := QUERY.search(line):
                _, at, digest, present, asked, foreign = m.groups()
                found.append(
                    (float(at), k, digest, int(present), int(asked), int(foreign))
                )
    return sorted(found)


def named_result(run: dict) -> dict:
    """What a run hands in: the LP's counts, each prefill engine's Store
    counts summed, and the step table's digest."""
    hand = _hand_ins(run["out"])
    shells = defaultdict(int)
    for said in hand.values():
        for key, value in said.get("offload", {}).get("store", {}).items():
            if isinstance(value, int):
                shells[key] += value
    lp = hand[store.ENTRY]["store"]
    return {
        "table": hashlib.sha256(run["table"].encode()).hexdigest()[:16],
        "foreign_hit_keys": lp["foreign_hit_keys"],
        "foreign_loads": shells["foreign_loads"],
        "evicted": lp["evicted"],
        "renewals": lp["renewals"],
        "past_lease_loads": shells["past_lease_loads"],
        "lp": lp,
        "shells": dict(shells),
    }


@NEEDS_A_STORE_RUN
def test_both_seeds_meet_every_scenario_check(store_runs):
    run = store_runs[0]
    records = {
        f"engine-{s}{k}": run["out"] / "parity" / f"{n}{k}"
        for s, n, ks in (("P", "prefill", (0, 1)), ("D", "decode", ("",)))
        for k in ks
    }
    failed = check_run(
        [r["out"] for r in store_runs], run["sent"], run["taken"], records
    )
    for r in store_runs:
        print(f"\n{r['out'].name}: {json.dumps(named_result(r))}")
    assert not failed, "\n".join(failed)
    assert store_runs[0]["table"] == store_runs[1]["table"]


@NEEDS_A_STORE_RUN
def test_a_prefix_one_instance_saved_loads_on_the_other(store_runs):
    for run in store_runs:
        said = named_result(run)
        assert said["foreign_hit_keys"] > 0 and said["foreign_loads"] > 0, said
        assert said["past_lease_loads"] == 0, said


@NEEDS_A_STORE_RUN
def test_an_eviction_round_runs_and_a_later_lookup_of_its_groups_misses(store_runs):
    for run in store_runs:
        said = named_result(run)
        assert said["evicted"] > 0, said
        queries = _queries(run["out"])
        print("", *queries, sep="\n  ")
        first = [q for q in queries if q[5] > 0]  # family 0's, found on the other
        assert first, "no lookup found another instance's objects"
        family0 = [q for q in queries if q[2] == first[0][2]]
        assert (
            family0[-1][3] < first[0][3]
        ), "family 0's last lookup found what its first did, past an eviction"


@NEEDS_A_STORE_RUN
def test_the_kv_budget_holds_the_store_s_reserve(store_runs):
    """Each prefill rank held back the Store's pools, their alignment and a
    staging bound per worker before it sized its KV pool, as the real worker
    class states them at this test's pools and workers."""
    from atom.kv_transfer.offload._block_gpu_connector import (
        _DEFAULT_GPU_STAGING_BYTES,
    )
    from atom.kv_transfer.offload.mooncake_store.pool import pool_allocation_bytes

    pools = pool_allocation_bytes(POOL_MIB << 20, POOL_MIB << 20)
    reserve = pools + 2 * _DEFAULT_GPU_STAGING_BYTES
    for run in store_runs:
        for k in (0, 1):
            text = (run["out"] / f"prefill{k}.log").read_text()
            said = re.findall(r"Reserving (\S+)GB for KV connector buffers", text)
            assert said == [f"{reserve / (1 << 30):.2f}"] * 2, said
