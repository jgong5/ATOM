# SPDX-License-Identifier: MIT
"""A launch's time in HIP's blocking calls, stamped and left out of `W_launch`.

The tool is built and loaded as a server loads it, through
``ROCP_TOOL_LIBRARIES``, into a process with no GPU: HIP still routes each call
through its dispatch table, so a call that fails for want of a device is timed.
A recording runner on the planted device reads planted sums at its stamps, and
a planted record states the time its launches blocked.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_backend_step_linear_fit import HOST, _steps, _timed_record
from test_parity_device_window import BUSY, _Processor, _recorded
from test_parity_record import _config, _drive

from atom.compass.backends.step_labels import host_kind, label
from atom.compass.backends.step_linear_fit import fit_host, label_records
from atom.compass.parity import ENV, blocked, compare, read

ROOT = Path(__file__).resolve().parents[2]

#: Run in a process of its own: blocking calls on this thread and another, read
#: by the tool's sums.
CHILD = """
import ctypes, json, threading
from atom.compass.parity import blocked

hip = ctypes.CDLL("libamdhip64.so")
hip.hipGetDeviceCount(ctypes.byref(ctypes.c_int()))
tool = blocked.loaded()
before = tool()
hip.hipDeviceSynchronize()
hip.hipStreamSynchronize(None)
hip.hipGetLastError()
mine = tool.between(before, tool())
other = {}

def block():
    start = tool()
    hip.hipStreamSynchronize(None)
    other.update(tool.between(start, tool()))

thread = threading.Thread(target=block)
thread.start()
thread.join()
print(json.dumps({"names": tool.names, "mine": mine, "other": other,
                  "since": tool.between(before, tool())}))
"""


def _child(code, **env):
    """`code` run by Python in a process of its own, `atom` this tree's."""
    inherited = {k: v for k, v in os.environ.items() if k != "ROCP_TOOL_LIBRARIES"}
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**inherited, "PYTHONPATH": str(ROOT), **env},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_the_tool_times_each_thread_s_blocking_hip_calls(tmp_path):
    """Loaded by rocprofiler-sdk, the tool times a device and a stream
    synchronize on the thread that made them, and the other thread's
    synchronize on that thread alone; an error query is not a blocking call.
    It times synchronous copies and frees, and no async call or launch."""
    library = blocked.build(tmp_path)
    run = _child(CHILD, ROCP_TOOL_LIBRARIES=str(library))
    assert run.returncode == 0, run.stderr
    got = json.loads(run.stdout.splitlines()[-1])

    assert set(got["mine"]) == {"hipDeviceSynchronize", "hipStreamSynchronize"}
    assert set(got["other"]) == {"hipStreamSynchronize"}
    assert min(got["mine"].values()) > 0 and got["other"]["hipStreamSynchronize"] > 0
    assert got["since"] == got["mine"]
    names = set(got["names"])
    assert {"hipMemcpy", "hipMemcpyWithStream", "hipMemcpyDtoH", "hipFree"} <= names
    assert {"hipEventSynchronize", "hipStreamSynchronize_spt"} <= names
    assert not {n for n in names if "Async" in n or "Launch" in n or "Graph" in n}


def test_a_process_without_the_tool_reads_none_and_one_hip_never_hooked_refuses(
    tmp_path,
):
    library = blocked.build(tmp_path)
    run = _child("from atom.compass.parity import blocked; print(blocked.loaded())")
    assert (run.returncode, run.stdout.strip()) == (0, "None")
    run = _child(
        f"import ctypes; ctypes.CDLL({str(library)!r})\n"
        "from atom.compass.parity import blocked; blocked.loaded()"
    )
    assert run.returncode == 1
    assert "HIP handed it no dispatch table to hook" in run.stderr


def _planted_tool(names):
    """`blocked.Blocked` over the planted sums of one thread, and `spend(call,
    ns)`, which adds to a call's."""
    sums = [0] * len(names)

    def read(out):
        out[:] = sums

    def spend(call, ns):
        sums[names.index(call)] += ns

    lib = SimpleNamespace(
        compass_blocked_calls=lambda: len(names),
        compass_blocked_name=lambda i: names[i].encode(),
        compass_blocked_read=read,
    )
    return blocked.Blocked(lib), spend


def test_a_line_states_what_its_launch_spent_blocked_up_to_its_token_copy_wait(
    tmp_path, monkeypatch
):
    """Each forward's staging gate blocks 1000 ns and its launch 300 ns; an
    output forward's postprocess blocks 50 ns more before its token copy wait,
    which blocks 7000 ns, each on the host clock. A line's `blocked_ns` holds
    what its launch's thread spent from the launch's entry to that wait's
    entry, or to the launch's exit on a middle chunk, and `W_launch` leaves it
    out. It times the step and takes no part in its decision."""
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    recorded, device = _recorded(batches[-1])
    base = recorded.__mro__[2]
    tool, spend = _planted_tool(("hipEventSynchronize", "hipMemcpyWithStream"))
    late = [0]

    class Blocking(recorded):
        def step_clock_ns(self):
            return recorded.step_clock_ns() + late[0]

    def wrap(cls, name, call, ns):
        method = getattr(cls, name)

        def blocking(self, *args):
            spend(call, ns)
            late[0] += ns
            return method(self, *args)

        monkeypatch.setattr(cls, name, blocking)

    wrap(base, "_gate_staging_reuse", "hipEventSynchronize", 1000)
    wrap(base, "run_model", "hipMemcpyWithStream", 300)
    wrap(_Processor, "prepare_sampled_ids", "hipMemcpyWithStream", 50)
    wrap(_Processor, "recv_async_output", "hipEventSynchronize", 7000)
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = Blocking(0, _config())
    runner._blocked = tool
    for batch in batches:
        runner.forward(batch)
    device.tick = BUSY // 3
    runner.exit()

    lines = read(tmp_path / "real")[0]
    outputs = [b.produces_output() for b in batches]
    assert any(outputs) and not all(outputs)
    want = [{"hipMemcpyWithStream": 350 if o else 300} for o in outputs]
    assert [line["blocked_ns"] for line in lines] == want
    for s, line, w in zip(label(tmp_path / "real" / "dp0.jsonl")[0], lines, want):
        l0, l1 = line["t_launch_ns"]
        wall = line.get("t_recv_ns", [l1])[0] - l0
        assert wall > sum(w.values())
        assert s.host["W_launch"] == pytest.approx((wall - sum(w.values())) * 1e-9)
        assert s.blocked == pytest.approx({k: v * 1e-9 for k, v in w.items()})
    first = compare(tmp_path / "real", tmp_path / "simulated")["first_divergence"]
    assert first == {0: None}


def test_a_launch_s_blocked_time_leaves_its_W_launch_label_held_or_not(tmp_path):
    """From step 40 on, each middle chunk's launch blocks in a device read for
    as long as its device work, and ends past a fifth of its device window,
    where the held cut took a launch to have waited on the device; decodes 1,
    11, 21 and 31 block for 30 us, the device not held. With each line's `blocked_ns`, every `W_launch` label is its law's,
    the host laws are recovered whole, and the report counts the launches that
    blocked per kind with the call they blocked in. The same record without
    the stamp labels those launches by their wall times, and the report names
    it."""
    steps = _steps(14, n=80)
    spent, scale = {}, {}
    for n in [*range(40, 80, 4), 1, 11, 21, 31]:
        rows = steps[n][0]
        x = (1, sum(r[0] for r in rows), len(rows))
        kind = "decode" if n % 2 else "middle"
        law = round(float(np.dot(HOST[kind]["W_launch"], x)) * 1e9)
        b = 30_000 if n % 2 else round(steps[n][2] * 1e9)
        scale[(n, "W_launch")] = (law + b) / law
        spent[n] = round(law * scale[(n, "W_launch")]) - law
    record = _timed_record(tmp_path / "dp0.jsonl", steps, scale=scale)[0]
    wall = record.read_text()
    lines = [json.loads(line) for line in wall.splitlines()]
    for line in lines:
        b = spent.get(line["step"])
        line["blocked_ns"] = {} if b is None else {"hipMemcpyWithStream": b}
    record.write_text("".join(json.dumps(line) + "\n" for line in lines))

    samples = label_records([record])[0]
    for s in samples:
        x = (1, sum(r.query_tokens for r in s.view.requests), len(s.view.requests))
        law = round(float(np.dot(HOST[host_kind(s.view)]["W_launch"], x)) * 1e9)
        assert s.host["W_launch"] == pytest.approx(law * 1e-9, rel=1e-9)
        b = spent.get(s.step)
        assert s.blocked == ({} if b is None else {"hipMemcpyWithStream": b * 1e-9})
    host, report = fit_host(samples)
    for kind, segments in HOST.items():
        for segment, c in segments.items():
            got = host["laws"][kind][segment]
            assert got == pytest.approx(c, rel=1e-4, abs=1e-12), f"{kind}.{segment}"
    (run,) = report["blocked"]
    got = report["blocked"][run]
    assert [got[k]["blocked"] for k in ("decode", "prefill", "middle")] == [4, 0, 10]
    assert got["decode"]["ms"] == pytest.approx([0.03, 0.03])
    assert got["middle"]["calls"].keys() == {"hipMemcpyWithStream"}
    assert report["unstamped"] == []

    record.write_text(wall)
    samples = label_records([record])[0]
    for s in samples:
        if s.step in spent:
            l0, l1 = lines[s.step]["t_launch_ns"]
            r0 = lines[s.step].get("t_recv_ns", [l1])[0]
            assert s.host["W_launch"] == pytest.approx((r0 - l0) * 1e-9, rel=1e-9)
    report = fit_host(samples)[1]
    (run,) = {s.provenance["run"] for s in samples}
    assert (report["blocked"], report["unstamped"]) == ({}, [run])
