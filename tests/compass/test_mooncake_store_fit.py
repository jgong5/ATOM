# SPDX-License-Identifier: MIT
"""`fit_offload`'s ``mooncake_store`` entry, written by `offload_main`.

The lines are ATOM's own: each timed lookup line is what `_StoreLookup.lookup`
logs over a stand-in master, and every other line is formatted by the format
string of the logger call in ATOM's source that writes it, its values looked up
by the format's field names, so a changed line fails here.
"""

import ast
import collections
import json
import logging
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.compass.backends.step_linear_fit import offload_main
from atom.kv_transfer.offload.mooncake_store import scheduler as store_scheduler

# The Store worker's log calls: kv_store's where the tree split the Store into
# kv_store and mooncake_store (the 1P2D repro), else mooncake_store's own.
_KV_STORE_WORKER = Path(store_scheduler.__file__).parents[1] / "kv_store" / "worker.py"
WORKER = (
    _KV_STORE_WORKER
    if _KV_STORE_WORKER.exists()
    else Path(store_scheduler.__file__).with_name("worker.py")
)
CHUNKED = Path(store_scheduler.__file__).parents[1] / "chunked_scheduler.py"
FLAGS = (
    "--rpc_thread_num=32 --default_kv_lease_ttl=10000 "
    "--eviction_high_watermark_ratio=0.90 --eviction_ratio=0.05"
)
CAPACITY = 1690069630976
#: A worker's start-line values, by its format's field names.
START = {
    "rank": 0,
    "world": 2,
    "namespace": "ns",
    "nic": "rdma6",
    "master": "10.0.0.2:26051",
    "bytes_per_block": 39504,
    "chunk": 256,
    "chunk_bytes": 10113024,
    "pool": "cuda:0",
    "save_slots": 26,
    "load_slots": 106,
    "gpu_staging_buffer_bytes": 50565120,
    "save": "<save>",
    "load": "<load>",
    "save_workers": 4,
    "load_workers": 4,
}
MATCHED = {
    "protocol": "rdma",
    "chunk_tokens": 256,
    "chunk_bytes": 10113024,
    "world": 2,
    "save_slots": 26,
    "load_slots": 106,
    "save_workers": 4,
    "load_workers": 4,
}
#: The stand-in master's milliseconds per lookup call and per key it asks.
FIXED_MS, PER_KEY_MS = 0.30, 0.02


#: The Store worker's start line where kv_store logs it for every Store: its
#: label and endpoint as `%s` (`LABEL`, `_endpoint_log`), a draft window's
#: chunk bytes last (`%s`, empty without one), as Mooncake's worker writes them.
LABELLED = {
    "%s worker rank=": "Mooncake Store offload worker rank=",
    "namespace=%s %s ": "namespace=%s nic=%s master=%s ",
}


def logged(start: str, values, source: Path = WORKER) -> str:
    """The line `source`'s logger call whose format starts `start` writes for
    `values`."""
    def head(arg):
        # A format whose tail a call appends (`"..." + self._draft_prof(...)`,
        # empty without a draft window) is its constant head.
        while isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
            arg = arg.left
        return arg.value if isinstance(arg, ast.Constant) else None

    formats = [
        fmt
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) in ("info", "debug")
        and node.args
        and (fmt := head(node.args[0])) is not None
    ]
    if not any(str(f).startswith(start) for f in formats):
        labelled = []
        for f in formats:
            for a, b in LABELLED.items():
                f = str(f).replace(a, b)
            labelled.append(f.removesuffix("%s"))
        formats = labelled
    [fmt] = [f for f in formats if str(f).startswith(start)]
    names = re.findall(r"(\w+)=%", fmt)
    return "[atom 12:00:00] " + fmt % tuple(values[name] for name in names)


def copy(tag: str, chunks: int, status: str = "ok", **ms) -> str:
    """A copy's line under `OFFLOAD_PROFILE`, every field not given 1."""
    fields = {"chunks": chunks, "status": status, **ms}
    return logged(tag, collections.defaultdict(lambda: 1, fields))


def lookups(monkeypatch, caplog, chunk_counts) -> list[str]:
    """`_StoreLookup.lookup`'s lines at DEBUG for prompts of `chunk_counts`
    chunks, over a master that holds every key and takes `FIXED_MS` and
    `PER_KEY_MS` a key on the lookup's clock."""
    clock = SimpleNamespace(now=0.0)

    class Master:
        def exists(self, asked):
            clock.now += (FIXED_MS + PER_KEY_MS * (len(asked) - 1)) * 1e-3
            return [0] + [1] * (len(asked) - 1)  # the taint marker first

    fake_time = SimpleNamespace(
        perf_counter=lambda: clock.now, monotonic=time.monotonic
    )
    monkeypatch.setattr(store_scheduler, "time", fake_time)
    pool = SimpleNamespace(master=START["master"], metadata="http://m/metadata")
    lookup = store_scheduler._StoreLookup(
        SimpleNamespace(store_masters=lambda: [pool]),
        namespace="ns",
        world=START["world"],
        chunk_tokens=START["chunk"],
    )
    lookup._pools[0].client = Master()
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="atom"):
        for n in chunk_counts:
            tokens = list(range(256 * n))
            seq = SimpleNamespace(token_ids=tokens, num_prompt_tokens=len(tokens))
            assert lookup.lookup(seq, f"q{n}") == 256 * n
    return ["[atom 12:00:00] " + r.getMessage() for r in caplog.records]


def store_logs(tmp_path, monkeypatch, caplog, **changed) -> list[Path]:
    """Two Store prefill logs of two ranks each, the second's start lines with
    the values `changed`; their copies and lookups fit known laws."""
    lines = [[], []]
    for i, half in enumerate((range(1, 5), range(5, 9))):
        for rank in range(START["world"]):
            values = START | {"rank": rank} | (changed if i else {})
            lines[i].append(logged("Mooncake Store offload worker", values))
        for n in half:
            ms = {"total_ms": 1.0 + 2.0 * n, "pack_ms": 0.5 * n, "put_ms": 1.5 * n}
            lines[i].append(copy("[OFFLOAD-SAVE-PROF]", n, **ms))
            ms = {"total_ms": 0.5 + 1.5 * n, "get_ms": 1.0 * n, "unpack_ms": 0.25 * n}
            lines[i].append(copy("[OFFLOAD-LOAD-PROF]", n, **ms))
    lookup_lines = lookups(monkeypatch, caplog, range(1, 7))
    lines[0] += lookup_lines[:3]
    lines[1] += lookup_lines[3:]
    # A failed copy is priced nowhere, nor a lookup line that times no call;
    # the scheduler's counters, which fit lmcache_offload's lookups, time none.
    untimed = collections.defaultdict(lambda: 1)
    lines[1].append(logged("[OFFLOAD-LOOKUP] seq=%s num_prompt", untimed, CHUNKED))
    failed = {"total_ms": 500.0, "pack_ms": 100.0, "put_ms": 400.0}
    lines[0].append(copy("[OFFLOAD-SAVE-PROF]", 3, status="put_failed", **failed))
    failed = {"total_ms": 300.0, "get_ms": 300.0, "unpack_ms": 0.0}
    lines[0].append(copy("[OFFLOAD-LOAD-PROF]", 2, status="get_failed", **failed))
    for minute in (1, 2):
        counts = {"lookup_n": 9 * minute, "lookup_prompt_tokens": 1, "lookup_us": 1}
        lines[1].append(f"[atom 12:00:00] [OFFLOAD-PERF] {json.dumps(counts)} | {{}}")
    logs = [tmp_path / "p0.log", tmp_path / "p1.log"]
    for log, said in zip(logs, lines):
        log.write_text("\n".join(said) + "\n")
    return logs


def test_store_logs_give_the_entry_on_ok_copies_and_timed_lookups(
    tmp_path, monkeypatch, caplog
):
    logs = store_logs(tmp_path, monkeypatch, caplog, nic="rdma7")
    out = tmp_path / "entry.json"
    argv = [*map(str, logs), "--out", str(out), "--master-flags", FLAGS]
    offload_main([*argv, "--capacity-bytes", str(CAPACITY)])

    entry = json.loads(out.read_text())
    print(json.dumps(entry, indent=1))
    laws = {"lookup_s", "store_s", "load_s"}
    assert set(entry) == {*laws, *MATCHED, "master", "capacity_bytes", "fit"}
    assert {name: entry[name] for name in MATCHED} == MATCHED
    assert (entry["master"], entry["capacity_bytes"]) == (FLAGS.split(), CAPACITY)
    assert entry["store_s"] == pytest.approx([1e-3, 2e-3], rel=1e-6)
    assert entry["load_s"] == pytest.approx([0.5e-3, 1.5e-3], rel=1e-6)
    assert entry["lookup_s"] == pytest.approx([FIXED_MS * 1e-3, PER_KEY_MS * 1e-3])
    fit = entry["fit"]
    assert fit["logs"] == [str(log) for log in logs]
    samples = {law: fit[law]["samples"] for law in laws}
    assert samples == {"store_s": 8, "load_s": 8, "lookup_s": 6}
    parts = {
        "pack_ms": ("store_s", 2.25, 3.65),
        "put_ms": ("store_s", 6.75, 10.95),
        "get_ms": ("load_s", 4.5, 7.3),
        "unpack_ms": ("load_s", 1.125, 1.825),
    }
    for part, (law, p50, p90) in parts.items():
        assert fit[law][part] == pytest.approx({"p50": p50, "p90": p90}), part


@pytest.mark.parametrize(
    "changed",
    [
        {"world": 4},
        {"nic": "tcp"},
        {"chunk": 512},
        {"chunk_bytes": 1},
        {"save_slots": 1},
        {"load_slots": 1},
        {"save_workers": 1},
        {"load_workers": 1},
    ],
)
def test_store_logs_differing_in_a_matched_field_refuse(
    tmp_path, monkeypatch, caplog, changed
):
    logs = store_logs(tmp_path, monkeypatch, caplog, **changed)
    argv = [*map(str, logs), "--out", str(tmp_path / "entry.json")]
    flags = ["--master-flags", FLAGS, "--capacity-bytes", str(CAPACITY)]
    with pytest.raises(ValueError, match="state none; an entry is fitted on one"):
        offload_main(argv + flags)


def test_the_master_and_its_owners_go_with_store_logs_alone(
    tmp_path, monkeypatch, caplog
):
    logs = store_logs(tmp_path, monkeypatch, caplog)
    argv = [*map(str, logs), "--out", str(tmp_path / "entry.json")]
    with pytest.raises(ValueError, match="--master-flags and --capacity-bytes"):
        offload_main([*argv, "--master-flags", FLAGS])
    bare = tmp_path / "no_start.log"
    bare.write_text(logs[1].read_text().split("\n", 2)[2])
    capacity = ["--capacity-bytes", str(CAPACITY)]
    with pytest.raises(ValueError, match=r"and 1 state none"):
        offload_main(
            [str(logs[0]), str(bare), *argv[2:], "--master-flags", FLAGS, *capacity]
        )
    lmcache = tmp_path / "lmcache.log"
    lmcache.write_text(
        "[atom] LMCache M3 PAGE offload worker rank=0: page_regions=301 "
        "bytes_per_block=5056512 chunk=256 gpu_staging_chunk_bytes=10113024\n"
    )
    with pytest.raises(ValueError, match="lmcache_offload's logs state no Store"):
        offload_main([str(lmcache), *argv[2:], *capacity])
