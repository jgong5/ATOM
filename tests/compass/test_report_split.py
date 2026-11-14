# SPDX-License-Identifier: MIT
"""Each request's split, `python -m atom.compass.report.split OUT`, on run
directories laid out as `compass run`, `compass record`, `deploy.sh` and
`client.sh` write them; and the split `client.sh --stop` and `deploy.sh` write
as a run ends.

A simulated instance's record and cost lines are the simulated runner's own,
driven through ATOM's scheduler on a clock each forward's charge moves, as the
engine's moves; a real deployment's are written by hand, its decode side on a
clock `DECODE_S` ahead of its client's and its prefill side's.
"""

import gzip
import json
import signal
import subprocess
import sys
import time

import pytest
from test_client_sh import COMMAND, GOOD, _client, _setup
from test_deploy_sh import _until, argv, planted  # noqa: F401  the planted fixture
from test_parity_client_run import raw, sse
from test_parity_record import _config, _Runner
from test_runner_step_semantics import _drive

from atom.compass import parity
from atom.compass.carriers import EPOCH_NS
from atom.compass.parity.client_run import NAME
from atom.compass.report import split
from atom.compass.runner import overrides
from atom.compass.runner.overrides import HostDeviceRecurrence

DECODE_S = 1000.0


def _export(run, requests, simulated=False) -> None:
    """aiperf's raw records and per-request export of `requests`, each
    ``(request id, response id, send, TTFT, end-to-end)`` in seconds, its
    TTFT None for a request aiperf records as an error; and the client run's
    export, which `deploy.sh` waits for."""
    artifacts = run / "artifacts"
    artifacts.mkdir(parents=True)
    raws, rows = [], []
    for x, name, sent, ttft, e2e in requests:
        packets = [sse(("data", json.dumps({"id": name})))] if name else []
        ns = round(sent * 1e9) + (EPOCH_NS if simulated else 0)
        raws.append(raw(packets) | {"start_perf_ns": ns})
        raws[-1]["metadata"] = raws[-1]["metadata"] | {"x_request_id": x}
        meta = {"x_request_id": x, "benchmark_phase": "profiling"}
        meta |= {"request_start_ns": ns, "request_end_ns": ns + 1}
        if ttft is None:
            rows.append({"metadata": meta, "error": {"type": "TimeoutError"}})
            continue
        ms = {"time_to_first_token": ttft * 1e3, "request_latency": e2e * 1e3}
        metrics = {k: {"value": v, "unit": "ms"} for k, v in ms.items()}
        rows.append({"metadata": meta, "metrics": metrics})
    with gzip.open(artifacts / "profile_export_raw.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in raws)
    (artifacts / "profile_export.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (artifacts / NAME).write_text(json.dumps({"requests": {}}))


def _line(step, enter, out, batch=(), named=(), **more) -> dict:
    """A real step line: (request id, tokens, context) per row, and (request
    id, key, response id) per request it names; times in seconds."""
    line = {"step": step, "dp_rank": 0, "is_dummy_run": False}
    line |= {"batch": [list(b) for b in batch], "named": [list(n) for n in named]}
    return line | {
        "t_enter_ns": round(enter * 1e9),
        "t_exit_ns": round(out * 1e9),
        **more,
    }


def _record(side, lines, simulated, transfers=()) -> None:
    """`side`'s record of `lines`; `simulated`, with cost lines holding their
    times in place of the lines' own."""
    steps = side / "steps"
    steps.mkdir(parents=True)
    if simulated:
        cost = [{"version": overrides.COST_TABLE_VERSION}]
        for s in lines:
            t = s.pop("t_enter_ns") * 1e-9
            cost.append(
                {"step": s["step"], "t_s": t, "group_s": s.pop("t_exit_ns") * 1e-9 - t}
            )
        (steps / "dp0.cost.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in cost)
        )
    (steps / "dp0.jsonl").write_text("".join(json.dumps(s) + "\n" for s in lines))
    if transfers:
        text = [
            {"response_id": n, "remote_dp_rank": 0, "transfer_id": t}
            for t, n in transfers
        ]
        (steps / parity.TRANSFERS).write_text(
            "".join(json.dumps(t) + "\n" for t in text)
        )


def _deployment(run, kv_link="inter_node", d=DECODE_S, simulated=False) -> None:
    """A 1P1D run of two requests, A and B, its decode side's clock `d` ahead,
    and three aiperf did not split: one errored, one with no response id, one
    no step names."""
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"kv_link": kv_link}))
    load = [[1, "load", 0, 64], [1, "save", 64, 200]]
    _record(
        run / "prefill",
        [
            _line(0, 1.2, 1.5, [(1, 64, 64)], kv=load),
            _line(1, 1.5, 1.9, [(1, 136, 200)], [(1, "ka", "pa")]),
            _line(2, 1.9, 2.2, [(2, 50, 50)], [(2, "kb", "pb")]),
        ],
        simulated,
    )
    _record(
        run / "decode",
        [
            _line(0, d + 2.3, d + 2.32, [(5, 1, 201)], [(5, 5, "a")]),
            _line(1, d + 2.6, d + 2.62, [(5, 1, 202), (6, 1, 51)], [(6, 6, "b")]),
        ],
        simulated,
        transfers=[(1, "a"), (2, "b")],
    )
    _export(
        run,
        [
            ("xa", "a", 1.0, 1.33, 2.0),
            ("xb", "b", 1.1, 1.54, 1.8),
            ("xe", "e", 1.1, None, None),
            ("xn", None, 1.1, 0.5, 0.6),
            ("xu", "u", 1.1, 0.5, 0.6),
        ],
        simulated,
    )


def test_a_real_deployment_joins_its_decode_node_to_its_clients_clock(tmp_path):
    run = tmp_path / "run"
    _deployment(run)

    out = split.write(run)

    assert json.loads((run / split.FILE).read_text()) == out
    # Prefill return less decode entry: 1.9 - 1002.3, 2.2 - 1002.6; first
    # token less decode return: 2.33 - 1002.32, 2.64 - 1002.62.
    assert out["clock"] == "monotonic"
    assert out["decode_offset_s"] == pytest.approx([-1000.4, -999.99])
    a, b = out["requests"]
    assert (a["response_id"], a["request_id"], b["response_id"]) == ("a", "xa", "b")
    for r, expected in (
        (
            a,
            {"queue_s": 0.2, "prefill_s": 0.7, "kv_transfer_s": 0.41, "decode_s": 0.69},
        ),
        (
            b,
            {"queue_s": 0.8, "prefill_s": 0.3, "kv_transfer_s": 0.41, "decode_s": 0.29},
        ),
    ):
        assert {k: r[k] for k in expected} == pytest.approx(expected)
        assert sum(r[k] for k in expected) == pytest.approx(r["e2el_s"], abs=1e-9)
    assert (a["forward_ttft_s"], b["forward_ttft_s"]) == pytest.approx((1.33, 1.53))
    assert (a["lmcache_load_tokens"], a["lmcache_store_tokens"]) == (64, 136)
    assert (b["lmcache_load_tokens"], b["lmcache_store_tokens"]) == (0, 0)
    assert out["unsplit"] == {
        "xe": "aiperf records it error",
        "xn": "it got no response id",
        "xu": "no step names it on every side",
    }


def test_a_deployment_on_one_node_keeps_one_clock_and_refuses_events_out_of_order(
    tmp_path,
):
    """Intra-node, the decode side's forwards are read on the client's clock,
    `DECODE_S` before its first forward's return, so each request's first
    token at the client comes before the forward that produced it."""
    run = tmp_path / "run"
    _deployment(run, kv_link="intra_node")

    out = split.write(run)

    assert out == {
        "refused": "request a's first token is 999.990000000 s before its decode "
        "return, so its events are not on one clock"
    }


def _simulated_instance(run, monkeypatch, cost=None) -> list[dict]:
    """A `compass run` instance's record and cost lines, its forwards driven
    on a clock each moves by its charge and 2 ms more, `cost` applied to each
    cost line; each request's expected split."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(overrides, "_group_step_seconds", lambda *_: 0.0)
        batches = _drive().batches
    monkeypatch.setenv(parity.ENV, str(run / "steps"))
    r, clock, windows = _Runner(_config()), [10.0], []
    r.compass_host = HostDeviceRecurrence({}, (), lambda: clock[0])
    for b in batches:
        b.external_request_ids = [f"r{i}" for i in b.req_ids]
        enter = clock[0]
        windows.append((enter, enter + r.forward(b).predicted_s))
        clock[0] = windows[-1][1] + 0.002
    r._cost_table.close()
    if cost is not None:
        path = run / "steps" / "dp0.cost.jsonl"
        version, *lines = path.read_text().splitlines()
        lines = [json.dumps(cost(json.loads(line))) for line in lines]
        path.write_text("\n".join([version, *lines]) + "\n")
    (run / "run.json").write_text(json.dumps({"run_file": {}}))
    expected, requests = {}, []
    for b, (enter, out) in zip(batches, windows):
        for i, request in enumerate(b.req_ids):
            e = expected.setdefault(request, {"entry": enter})
            if b.is_final_chunk is not None and b.is_final_chunk[i]:
                e.setdefault("return", out)
            e["last"] = out
    for request, e in expected.items():
        sent, first, end = e["entry"] - 0.25, e["return"] + 0.01, e["last"] + 0.01
        requests.append((f"x{request}", f"r{request}", sent, first - sent, end - sent))
        e |= {"queue_s": 0.25, "prefill_s": e["return"] - e["entry"]}
        e |= {"decode_s": end - e["return"], "forward_ttft_s": e["return"] - sent}
    _export(run, requests, simulated=True)
    return [expected[r] for r in expected]


def test_a_simulated_instance_splits_on_its_engines_clock_from_its_cost_lines(
    tmp_path, monkeypatch
):
    run = tmp_path / "run"
    expected = _simulated_instance(run, monkeypatch)
    assert len(expected) > 1

    out = split.write(run)

    assert (out["clock"], out["unsplit"]) == ("virtual", {})
    assert len(out["requests"]) == len(expected)
    for r, e in zip(out["requests"], expected):
        segments = ("queue_s", "prefill_s", "decode_s")
        assert {k: r[k] for k in segments} == pytest.approx(
            {k: e[k] for k in segments}, abs=1e-8
        )
        assert "kv_transfer_s" not in r
        assert r["forward_ttft_s"] == pytest.approx(e["forward_ttft_s"], abs=1e-8)
        assert sum(r[k] for k in segments) == pytest.approx(r["e2el_s"], abs=1e-9)


def test_a_simulated_record_whose_cost_lines_hold_no_step_times_is_refused(
    tmp_path, monkeypatch
):
    run = tmp_path / "run"
    _simulated_instance(run, monkeypatch, lambda line: {**line, "t_s": None})

    out = split.write(run)

    cost = run / "steps" / "dp0.cost.jsonl"
    assert out == {"refused": f"{cost} holds no step times (t_s)"}


def test_a_run_with_no_step_record_is_refused_naming_where_it_looked(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text("{}")

    assert split.write(run) == {
        "refused": f"{run / 'steps'} holds no DP rank's step record"
    }


def test_client_sh_stop_writes_an_instances_split(tmp_path):
    """The stand-in command records no steps, so its split is refused."""
    out, env = _setup(tmp_path)
    (tmp_path / "compass_command.py").write_text(COMMAND)
    command = subprocess.Popen([sys.executable, tmp_path / "compass_command.py", out])
    while not (out / "server.json").exists():
        assert command.poll() is None
        time.sleep(0.1)

    done = _client(out, *GOOD, "--stop", env=env)

    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.endswith(
        f"split: {out / split.FILE}: refused: {out / 'steps'} holds no DP rank's "
        "step record\n"
    )
    written = json.loads((out / split.FILE).read_text())
    assert written == {"refused": f"{out / 'steps'} holds no DP rank's step record"}


def test_deploy_sh_writes_the_split_once_its_clients_export_is_complete(
    tmp_path, planted  # noqa: F811
):
    env, flags = planted
    d = tmp_path / "d"
    script = subprocess.Popen(
        argv(tmp_path, flags), env=env, cwd=tmp_path, stdout=subprocess.PIPE, text=True
    )
    _until(d / "server.json", script)
    real = tmp_path / "real"
    _deployment(real, d=0.0, simulated=True)
    for side in ("prefill", "decode"):
        (real / side / "steps").rename(d / side / "steps")
    (real / "artifacts").rename(d / "artifacts")

    script.send_signal(signal.SIGTERM)
    out, _ = script.communicate(timeout=60)

    assert script.returncode == 0, out
    written = json.loads((d / split.FILE).read_text())
    assert written["clock"] == "virtual" and "decode_offset_s" not in written
    a, b = written["requests"]
    for r, expected in (
        (a, {"queue_s": 0.2, "prefill_s": 0.7, "kv_transfer_s": 0.4, "decode_s": 0.7}),
        (b, {"queue_s": 0.8, "prefill_s": 0.3, "kv_transfer_s": 0.4, "decode_s": 0.3}),
    ):
        assert {k: r[k] for k in expected} == pytest.approx(expected)
    assert (a["forward_ttft_s"], b["forward_ttft_s"]) == pytest.approx((1.32, 1.52))
