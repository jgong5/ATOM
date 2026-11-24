# SPDX-License-Identifier: MIT
"""The RDMA link probe: its writes are cut as the Mooncake connector cuts
them, its link fit recovers a line, and its replay of a recorded P/D run's
`_rdma_write` calls tells a FIFO rail from a shared one and from per-pair or
per-prefill-rank queues."""

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import (
    MooncakeConnector,
)

TREE = Path(__file__).resolve().parents[2]
_script = importlib.util.spec_from_file_location(
    "rdma_link", TREE / "scripts/compass/rdma_link.py"
)
rdma = importlib.util.module_from_spec(_script)
_script.loader.exec_module(rdma)


class Engine:
    """A Mooncake engine that logs each batch it is handed."""

    def __init__(self):
        self.calls = []

    def batch_transfer_sync_write(self, target, src, dst, sizes):
        self.calls.append((target, src, dst, sizes))
        return 0


def test_a_write_posts_the_connectors_batches_of_entries():
    per_batch = MooncakeConnector._MAX_RDMA_ENTRIES_PER_BATCH
    n = (2 * per_batch + 3) * rdma.ENTRY - 5
    e = Engine()
    start, end, ok = rdma.write(e, "t:1", rdma.batches(1 << 20, 1 << 40, n, per_batch))
    assert ok and start <= end
    assert [len(c[3]) for c in e.calls] == [per_batch, per_batch, 3]
    assert sum(sum(c[3]) for c in e.calls) == n
    src = [a for c in e.calls for a in c[1]]
    dst = [a for c in e.calls for a in c[2]]
    assert src == [(1 << 20) + i * rdma.ENTRY for i in range(len(src))]
    assert [d - s for s, d in zip(src, dst)] == [(1 << 40) - (1 << 20)] * len(src)


def test_an_engine_registers_its_buffer_as_the_connector_would(monkeypatch):
    registered = []

    class TransferEngine:
        def initialize(self, *args):
            return 0

        def register_memory(self, addr, size):
            registered.append((addr, size))
            return 0

    engine = types.SimpleNamespace(TransferEngine=TransferEngine)
    monkeypatch.setitem(sys.modules, "mooncake.engine", engine)
    monkeypatch.setattr(rdma, "hca", lambda gpu: f"ionic_{gpu}")
    buf = types.SimpleNamespace(numel=lambda: rdma.BUFFER, data_ptr=lambda: 1 << 40)
    rdma.engine("ip", 0, buf)
    sizes = object.__new__(MooncakeConnector)._rdma_chunk_sizes(rdma.BUFFER, rdma.ENTRY)
    assert [s for _, s in registered] == sizes
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    assert [a - (1 << 40) for a, _ in registered] == offsets


def test_the_link_fit_recovers_latency_and_bandwidth():
    rows = [(n, 7e-6 + n / 9e9) for n in rdma.SIZES]
    latency, bandwidth = rdma.link(rows)
    assert latency == pytest.approx(7e-6) and bandwidth == pytest.approx(9e9)


def test_shared_bandwidth_splits_between_writes_in_flight():
    # 1 GB at 10 GB/s: alone 0.1 s; the second arrives with half the first left.
    jobs = [(0.0, 1e9), (0.05, 1e9)]
    assert rdma.shared(jobs, 0.0, 1e10) == pytest.approx([0.15, 0.2])
    assert rdma.fifo(jobs, 0.0, 1e10) == pytest.approx([0.1, 0.2])
    assert rdma.shared([(0.0, 0), (0.0, 1e9)], 1e-3, 1e10) == pytest.approx(
        [1e-3, 0.101]
    )


def record(tmp_path, calls):
    """A record directory holding the producer lines of `calls`: (worker, hca,
    start s, end s, bytes), one write each."""
    for i, (worker, hca, s, e, n) in enumerate(calls):
        line = {"request_id": f"r{i}", "write_nonce": 0, "hca": hca}
        line["rdma"] = [[round(s * 1e9), round(e * 1e9), n]]
        path = tmp_path / "prefill" / f"{worker}.writes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(line) + "\n")
    (tmp_path / "decode").mkdir()
    (tmp_path / "decode" / "dp0.tp0.writes.jsonl").write_text(
        json.dumps({"request_id": "r0", "t_write_done_ns": 1}) + "\n"
    )
    return tmp_path


#: Two prefill ranks write 1 GB each into rail ionic_1, one after the other at
#: 10 GB/s, then one more write each on rails of their own.
RECORD = [
    ("dp0.tp0", "ionic_1", 0.0, 0.1, 10**9),
    ("dp1.tp0", "ionic_1", 0.05, 0.2, 10**9),
    ("dp1.tp0", "ionic_2", 0.3, 0.35, 10**9),
    ("dp0.tp0", "ionic_3", 0.5, 0.6, 10**9),
]


def test_a_record_served_fifo_per_rail_is_fitted_by_the_rail_fifo_alone(tmp_path):
    found, _ = rdma.calls(record(tmp_path, RECORD[:3]))
    assert [c[2] for c in found] == [0.0, 0.05, 0.3]
    assert rdma.contended(found) == set(found[:2])
    errors = {
        (unit, d.__name__): list(rdma.replay(found, unit, d, 0.0, 1e10).values())
        for unit in rdma.UNITS
        for d in (rdma.fifo, rdma.shared)
    }
    assert errors["rail", "fifo"] == pytest.approx([0, 0, 1], abs=1e-9)
    assert errors["rail", "shared"] == pytest.approx([0.5, 0, 1], abs=1e-9)
    for unit in ("pair", "prefill rank"):
        assert sorted(errors[unit, "fifo"]) == pytest.approx([-1 / 3, 0, 1])


def test_fit_publishes_the_bench_latency_and_the_rate_writes_attained(tmp_path, capsys):
    rates = {r: 5e10 for r in range(rdma.GPUS)}
    targets = {r: {"target": f"d:{r}", "addr": 0, "rate": 5e10} for r in rates}
    lines = [{"env": {}, "rates": rates, "targets": targets}]
    for i in range(rdma.REPS):
        for g, r, n in ((g, r, n) for g in (0, 1) for r in (0, 1) for n in rdma.SIZES):
            t = (6e-6 if i % 2 else 4e-6) + n / (1e10 if g == r else 8e9)
            lines.append({"exp": "sweep", "gpu": g, "rail": r, "round": i,
                          "bytes": n, "t": [0, round(t * 1e9), True]})  # fmt: skip
    bench = tmp_path / "bench.jsonl"
    bench.write_text("".join(json.dumps(x) + "\n" for x in lines))
    run = record(tmp_path / "run", RECORD)
    rdma.main(["fit", str(bench), str(run)])
    out = capsys.readouterr().out
    files = sorted(run.rglob("*.writes.jsonl"))
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()
    want = f"record: 4 writes, 4 calls, 2 contended, writes files sha256:{digest}"
    assert want in out.splitlines()
    found = json.loads(out.split("uncontended writes: ")[1].splitlines()[0])
    assert found["link_bandwidth_bytes_per_s"] == [5e10, 5e10, 5e10]
    # Latency 4 or 6 us by round; the uncontended writes moved 20 and 10 GB/s.
    assert found["link_latency_s"] == pytest.approx([5e-6, 4e-6, 6e-6], rel=0.05)
    assert found["derate"][0] == pytest.approx(1.5e10 / 5e10, rel=1e-3)
