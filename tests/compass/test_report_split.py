# SPDX-License-Identifier: MIT
"""Each request's split, `python -m atom.compass.report.split OUT`, on run
directories laid out as `compass run`, `compass record`, `deploy.sh` and
`client.sh` write them; and the split `client.sh --stop` and `deploy.sh` write
as a run ends.

A simulated instance's record and cost lines are the simulated runner's own,
driven through ATOM's scheduler on a clock each forward's charge moves, as the
engine's moves; a real deployment's are written by hand, its decode side on a
clock `DECODE_S` ahead of its client's and its prefill side's, or, across two
nodes, the second node's instances on a clock `NODE_S` ahead of the first's.
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
from atom.compass.drive import clock_pair
from atom.compass.parity.client_run import NAME
from atom.compass.report import split
from atom.compass.runner import overrides
from atom.compass.runner.overrides import HostDeviceRecurrence

DECODE_S = 1000.0
#: How far a second node's clock runs ahead of the client's node's.
NODE_S = 1.056


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
    line = {"step": step, "dp_rank": 0, "is_dummy_run": False, "produces_output": True}
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
    each decode forward handing the one before's tokens over, and three aiperf
    did not split: one errored, one with no response id, one no step names."""
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"kv_link": kv_link}))
    load = [[1, "load", 0, 64], [1, "save", 64, 200]]
    _record(
        run / "prefill",
        [
            _line(0, 1.2, 1.5, [(1, 64, 64)], kv=load, produces_output=False),
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
            _line(2, d + 2.7, d + 2.72, is_dummy_run=True),
        ],
        simulated,
        transfers=[(1, "a"), (2, "b")],
    )
    _export(
        run,
        [
            ("xa", "a", 1.0, 1.64, 2.0),
            ("xb", "b", 1.1, 1.65, 1.8),
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
    # token less hand-over: 2.64 - 1002.62, 2.75 - 1002.72.
    assert out["clock"] == "monotonic"
    assert out["decode_offset_s"] == pytest.approx([-1000.4, -999.98])
    a, b = out["requests"]
    assert (a["response_id"], a["request_id"], b["response_id"]) == ("a", "xa", "b")
    for r, expected in (
        (
            a,
            {"queue_s": 0.2, "prefill_s": 0.7, "kv_transfer_s": 0.42, "decode_s": 0.68},
        ),
        (
            b,
            {"queue_s": 0.8, "prefill_s": 0.3, "kv_transfer_s": 0.42, "decode_s": 0.28},
        ),
    ):
        assert {k: r[k] for k in expected} == pytest.approx(expected)
        assert sum(r[k] for k in expected) == pytest.approx(r["e2el_s"], abs=1e-9)
    assert (a["forward_ttft_s"], b["forward_ttft_s"]) == pytest.approx((1.64, 1.64))
    assert (a["lmcache_load_tokens"], a["lmcache_store_tokens"]) == (64, 136)
    assert (b["lmcache_load_tokens"], b["lmcache_store_tokens"]) == (0, 0)
    assert out["unsplit"] == {
        "xe": "aiperf records it error",
        "xn": "it got no response id",
        "xu": "no step names it on every side",
    }


def _two_prefills(run, ports=(17100, 17200)) -> None:
    """A real 2P1D run on one node: each prefill instance's request 1 at DP
    rank 0, request A on the first and B on the second, which the decode
    side's transfers tell apart by their Mooncake handshake ports alone."""
    run.mkdir()
    instances = [
        {"name": "prefill0", "side": "P", "index": 0, "handshake_port": 17100},
        {"name": "prefill1", "side": "P", "index": 1, "handshake_port": 17200},
        {"name": "decode", "side": "D", "index": 0},
    ]
    meta = {"kv_link": "intra_node", "across_nodes": False, "instances": instances}
    (run / "run.json").write_text(json.dumps(meta))
    _record(
        run / "prefill0",
        [_line(0, 1.2, 1.9, [(1, 136, 200)], [(1, "ka", "pa")])],
        False,
    )
    _record(
        run / "prefill1", [_line(0, 1.3, 2.2, [(1, 50, 50)], [(1, "kb", "pb")])], False
    )
    _record(
        run / "decode",
        [
            _line(0, 2.3, 2.32, [(5, 1, 201)], [(5, 5, "a")]),
            _line(1, 2.6, 2.62, [(5, 1, 202), (6, 1, 51)], [(6, 6, "b")]),
            _line(2, 2.7, 2.72, is_dummy_run=True),
        ],
        False,
    )
    sent = [
        {
            "response_id": n,
            "remote_dp_rank": 0,
            "transfer_id": 1,
            "remote_handshake_port": port,
        }
        for n, port in zip("ab", ports)
    ]
    (run / "decode" / "steps" / parity.TRANSFERS).write_text(
        "".join(json.dumps(t) + "\n" for t in sent)
    )
    _export(run, [("xa", "a", 1.0, 1.64, 2.0), ("xb", "b", 1.1, 1.65, 1.8)])


def test_a_deployment_of_two_prefill_instances_joins_each_transfer_to_its_sender(
    tmp_path,
):
    run = tmp_path / "run"
    _two_prefills(run)

    out = split.write(run)

    assert out["clock"] == "monotonic" and "decode_offset_s" not in out
    a, b = out["requests"]
    assert (a["response_id"], b["response_id"]) == ("a", "b")
    for r, expected in (
        (a, {"queue_s": 0.2, "prefill_s": 0.7, "kv_transfer_s": 0.4, "decode_s": 0.7}),
        (b, {"queue_s": 0.2, "prefill_s": 0.9, "kv_transfer_s": 0.4, "decode_s": 0.3}),
    ):
        assert {k: r[k] for k in expected} == pytest.approx(expected)
    assert out["unsplit"] == {}


def test_a_transfer_naming_no_prefill_instance_refuses_the_split(tmp_path):
    run = tmp_path / "run"
    _two_prefills(run, ports=(17100, 17300))

    out = split.write(run)

    assert "names 0 of the 2 prefill instances" in out["refused"]


def test_a_deployment_on_one_node_keeps_one_clock_and_refuses_events_out_of_order(
    tmp_path,
):
    """Intra-node, the decode side's forwards are read on the client's clock,
    `DECODE_S` ahead of it, so each request's first token at the client comes
    before the forward that handed it over."""
    run = tmp_path / "run"
    _deployment(run, kv_link="intra_node")

    out = split.write(run)

    assert out == {
        "refused": "request a's first token is 999.980000000 s before its "
        "hand-over, so its events are not on one clock"
    }


def test_a_one_node_deployment_whose_kv_crossed_the_network_keeps_one_clock(
    tmp_path,
):
    """`deploy.sh --kv-over-network`: the KV link is inter_node, but the decode
    ran on this node (`across_nodes` false), so its forwards are read on the
    client's clock, as an intra-node deployment's are, and not joined to it."""
    run = tmp_path / "run"
    _deployment(run)
    meta = json.loads((run / "run.json").read_text())
    (run / "run.json").write_text(json.dumps(meta | {"across_nodes": False}))

    out = split.write(run)

    assert out == {
        "refused": "request a's first token is 999.980000000 s before its "
        "hand-over, so its events are not on one clock"
    }


def _ahead(t: float) -> float:
    """How far the second node's clock runs ahead of the client's at `t` on
    the client's."""
    return NODE_S


def _here(t: float) -> float:
    """The client's node's clock, which runs ahead of none."""
    return 0.0


def _two_nodes(run, remote=("prefill1", "decode1"), ahead=_ahead) -> None:
    """A real 2P2D run across two nodes: the instances `remote` names on
    10.0.0.2, or each on the address it maps them to, whose clock runs
    `ahead` of the client's node's, `NODE_S` unless given, as g27's ran ahead
    of g08's (#3720), and the others on the client's node. Request A goes from
    prefill0 to decode0, B from prefill1 to decode0, C from prefill0 to decode1
    and D from prefill1 to decode1, each event of each a round time on the
    client's clock; C's first token reaches the client as the forward handing
    it over returns."""
    if not isinstance(remote, dict):
        remote = dict.fromkeys(remote, "10.0.0.2")
    run.mkdir()
    instances = [
        {"name": "prefill0", "side": "P", "index": 0, "handshake_port": 17100},
        {"name": "prefill1", "side": "P", "index": 1, "handshake_port": 17200},
        {"name": "decode0", "side": "D", "index": 0},
        {"name": "decode1", "side": "D", "index": 1},
    ]
    meta = {"kv_link": "inter_node", "across_nodes": True, "instances": instances}
    (run / "run.json").write_text(json.dumps(meta))
    # Each instance's forwards: entry, return, rows and the requests it names.
    forwards = {
        "prefill0": [
            (1.2, 1.9, [(1, 136, 200)], [(1, "ka", "pa")]),
            (1.9, 2.1, [(2, 80, 80)], [(2, "kc", "pc")]),
        ],
        "prefill1": [
            (1.3, 2.2, [(1, 50, 50)], [(1, "kb", "pb")]),
            (2.2, 2.3, [(2, 30, 30)], [(2, "kd", "pd")]),
        ],
        "decode0": [
            (2.3, 2.32, [(5, 1, 201)], [(5, 5, "a")]),
            (2.6, 2.62, [(5, 1, 202), (6, 1, 51)], [(6, 6, "b")]),
            (2.7, 2.72, [], []),
        ],
        "decode1": [
            (2.4, 2.42, [(7, 1, 81)], [(7, 7, "c")]),
            (2.5, 2.52, [(7, 1, 82), (8, 1, 31)], [(8, 8, "d")]),
            (2.65, 2.67, [], []),
        ],
    }
    for name, held in forwards.items():
        there, host = (ahead, remote[name]) if name in remote else (_here, "127.0.0.1")
        lines = [
            _line(
                i,
                enter + there(enter),
                out + there(out),
                batch,
                named,
                is_dummy_run=not batch,
            )
            for i, (enter, out, batch, named) in enumerate(held)
        ]
        _record(run / name, lines, False)
        ready = {"url": f"http://{host}:8000", "pid": 1}
        (run / name / "server.json").write_text(json.dumps(ready))
    for name, sent in (("decode0", "ab"), ("decode1", "cd")):
        transfers = [
            {
                "response_id": n,
                "remote_dp_rank": 0,
                "transfer_id": 1 if name == "decode0" else 2,
                "remote_handshake_port": port,
            }
            for n, port in zip(sent, (17100, 17200))
        ]
        (run / name / "steps" / parity.TRANSFERS).write_text(
            "".join(json.dumps(t) + "\n" for t in transfers)
        )
    _export(
        run,
        [
            ("xa", "a", 1.0, 1.64, 2.0),
            ("xb", "b", 1.1, 1.65, 1.8),
            ("xc", "c", 1.5, 1.02, 1.3),
            ("xd", "d", 2.0, 0.69, 1.1),
        ],
    )


#: Each request of `_two_nodes` split as it ran, its `SEGMENTS`.
AS_RUN = {
    "a": (0.2, 0.7, 0.4, 0.7, 1.62),
    "b": (0.2, 0.9, 0.4, 0.3, 1.62),
    "c": (0.4, 0.2, 0.3, 0.4, 1.02),
    "d": (0.2, 0.1, 0.2, 0.6, 0.67),
}
SEGMENTS = ("queue_s", "prefill_s", "kv_transfer_s", "decode_s", "forward_ttft_s")


def _drifting(t: float) -> float:
    """A second node's clock ahead of the client's as g27's moved against
    g08's, a timesyncd slew one way then the other (#3720), though a thousand
    times as fast: `NODE_S` at 0 s, 10 ms more each second to 2 s, then 5 ms
    less each second."""
    return NODE_S + (0.01 * t if t <= 2.0 else 0.02 - 0.005 * (t - 2.0))


def _pairs(run, offsets, node="10.0.0.2", error_ns=20_000) -> None:
    """OUT's clock pairs, as `deploy.sh` record writes them, of `node`: each
    of `offsets` the client's clock and how far that node's runs ahead of it
    then, in seconds; each stating `error_ns`."""
    rows = [
        {"node": node, "mono_ns": round(t * 1e9), "offset_ns": round(o * 1e9)}
        | {"error_ns": error_ns}
        for t, o in offsets
    ]
    (run / clock_pair.FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_two_node_deployment_joins_each_instance_by_its_nodes_offset(tmp_path):
    """prefill1 and decode1 ran on the second node: B's and D's sends bound
    its offset from below, B's decode entry and C's and D's first tokens from
    above, and its upper bound, C's first token's way out, is the offset
    itself, so each request splits as it ran, wherever its sides ran."""
    run = tmp_path / "run"
    _two_nodes(run)

    out = split.write(run)

    assert out["clock"] == "monotonic" and "decode_offset_s" not in out
    # At least B's and D's send less their prefill entry, -NODE_S - 0.2; at
    # most C's first token less its hand-over's return, -NODE_S.
    ((node, joined),) = out["node_offsets_s"].items()
    assert node == "10.0.0.2"
    assert joined["offset_s"] == pytest.approx(-NODE_S)
    assert joined["bounds_s"] == pytest.approx([-NODE_S - 0.2, -NODE_S])
    assert [r["response_id"] for r in out["requests"]] == ["a", "b", "c", "d"]
    for r in out["requests"]:
        assert [r[k] for k in SEGMENTS] == pytest.approx(AS_RUN[r["response_id"]])
        assert sum(r[k] for k in SEGMENTS[:4]) == pytest.approx(r["e2el_s"], abs=1e-9)
    assert out["unsplit"] == {}


def test_a_two_node_deployment_whose_instances_name_no_node_is_refused(tmp_path):
    """Without its instances' server.json, every prefill instance is read on
    the client's clock, as the split read them before it read nodes: prefill1's
    return of B lands `NODE_S` late, after decode0 took B in."""
    run = tmp_path / "run"
    _two_nodes(run)
    for name in ("prefill0", "prefill1", "decode0", "decode1"):
        (run / name / "server.json").unlink()

    out = split.write(run)

    # decode0's offset is A's first token less its hand-over, 0.02 s.
    assert out == {
        "refused": "request b's decode entry is 0.636000000 s before its prefill "
        "return, so its events are not on one clock"
    }


def test_a_node_of_prefill_instances_alone_takes_its_lower_bound(tmp_path):
    """prefill1 alone ran on the second node: B's and D's way in, each one's
    send to its prefill entry, bounds its offset from below, and the faster of
    the two is taken as none."""
    run = tmp_path / "run"
    _two_nodes(run, remote=("prefill1",))

    out = split.write(run)

    # At most B's decode entry less its prefill return there, -NODE_S + 0.4,
    # and D's, -NODE_S + 0.2.
    joined = out["node_offsets_s"]["10.0.0.2"]
    assert joined["offset_s"] == pytest.approx(-NODE_S - 0.2)
    assert joined["bounds_s"] == pytest.approx([-NODE_S - 0.2, -NODE_S + 0.2])
    b, d = (r for r in out["requests"] if r["response_id"] in ("b", "d"))
    assert (b["queue_s"], b["kv_transfer_s"]) == pytest.approx((0.0, 0.6))
    assert (d["queue_s"], d["kv_transfer_s"]) == pytest.approx((0.0, 0.4))


def test_a_one_node_deployment_s_split_is_the_same_whatever_its_instances_serve_at(
    tmp_path,
):
    """`across_nodes` false: one clock, read the same whatever node an
    instance's server.json names, byte for byte."""
    run = tmp_path / "run"
    _two_prefills(run)
    out = split.write(run)
    written = (run / split.FILE).read_bytes()
    for name, host in (
        ("prefill0", "127.0.0.1"),
        ("prefill1", "10.0.0.2"),
        ("decode", "10.0.0.3"),
    ):
        ready = {"url": f"http://{host}:8000", "pid": 1}
        (run / name / "server.json").write_text(json.dumps(ready))

    assert split.write(run) == out
    assert (run / split.FILE).read_bytes() == written
    assert "node_offsets_s" not in out and "decode_offset_s" not in out


def test_a_node_whose_clock_is_measured_joins_each_event_by_the_offset_at_it(
    tmp_path,
):
    """The second node's clock drifts between the three pairs OUT holds of it,
    10 ms a second then back 5: each of its events moves by the offset at its
    own time, so each request splits as it ran, as no one offset, the bounds',
    can split it; the bounds stay beside the pairs as they were."""
    run = tmp_path / "run"
    _two_nodes(run, ahead=_drifting)
    bounded = split.split(run)
    _pairs(run, [(t, _drifting(t)) for t in (0.0, 2.0, 4.0)])

    out = split.write(run)

    assert out["node_offsets_s"] == {
        "10.0.0.2": {
            "source": "measured",
            "samples": 3,
            "measured_s": pytest.approx([-NODE_S - 0.02, -NODE_S]),
            "error_s": pytest.approx(2e-5),
            "bounds_s": bounded["node_offsets_s"]["10.0.0.2"]["bounds_s"],
            "order_violations": 0,
        }
    }
    assert [r["response_id"] for r in out["requests"]] == ["a", "b", "c", "d"]
    for r in out["requests"]:
        expected = AS_RUN[r["response_id"]]
        assert [r[k] for k in SEGMENTS] == pytest.approx(expected, abs=2e-9)
    # The bounds' one offset, C's first token's way out taken as none, moves
    # B's prefill side 4.4 ms early and D's 1.6 ms late.
    queues = {r["response_id"]: r["queue_s"] for r in bounded["requests"]}
    assert queues["b"] - AS_RUN["b"][0] == pytest.approx(-0.0044, abs=1e-8)
    assert queues["d"] - AS_RUN["d"][0] == pytest.approx(0.0016, abs=1e-8)


def test_without_pairs_of_its_nodes_the_split_is_the_bounds_byte_for_byte(
    tmp_path,
):
    """OUT holding no clock pair, or pairs of no node its instances ran on,
    leaves the split the bounds give, byte for byte, naming no source."""
    run = tmp_path / "run"
    _two_nodes(run)
    out = split.write(run)
    written = (run / split.FILE).read_bytes()
    _pairs(run, [(0.0, 0.5), (4.0, 0.5)], node="10.0.0.9")

    assert split.write(run) == out
    assert (run / split.FILE).read_bytes() == written
    joined = out["node_offsets_s"]["10.0.0.2"]
    assert list(joined) == ["offset_s", "bounds_s"]
    assert joined["offset_s"] == pytest.approx(-NODE_S)


@pytest.mark.parametrize("error_ns", [20_000, 2_000])
def test_a_request_the_measured_offset_puts_out_of_order_counts_within_its_error(
    tmp_path, error_ns
):
    """The pairs read the second node's clock 5 us behind where it ran, so C's
    first token, which reached the client as its hand-over returned there,
    lands 5 us before it: within the pairs' stated error, 20 us, C is split as
    joined and counted; past it, 2 us, the split is refused."""
    run = tmp_path / "run"
    _two_nodes(run)
    _pairs(run, [(0.0, NODE_S - 5e-6), (4.0, NODE_S - 5e-6)], error_ns=error_ns)

    out = split.write(run)

    if error_ns < 5_000:
        assert out == {
            "refused": "request c's first token is 0.000005000 s before its "
            "hand-over, so its events are not on one clock"
        }
        return
    assert out["node_offsets_s"]["10.0.0.2"]["order_violations"] == 1
    # Each event on the second node lands 5 us late.
    late = {"b": (5e-6, 0, -5e-6, 0, 0), "c": (0, 0, 5e-6, -5e-6, 5e-6)}
    late["d"] = (5e-6, 0, 0, -5e-6, 5e-6)
    for r in out["requests"]:
        shift = late.get(r["response_id"], (0,) * 5)
        expected = [v + d for v, d in zip(AS_RUN[r["response_id"]], shift)]
        assert [r[k] for k in SEGMENTS] == pytest.approx(expected, abs=2e-9)


def test_beside_a_measured_node_a_node_the_pairs_miss_keeps_its_bounds(tmp_path):
    """prefill1 ran on 10.0.0.2, whose clock OUT holds pairs of, and decode1
    on 10.0.0.3, whose it does not: that one takes its upper bound, C's first
    token's way out taken as none, and says so; D, from one to the other, is
    checked across both."""
    run = tmp_path / "run"
    _two_nodes(run, remote={"prefill1": "10.0.0.2", "decode1": "10.0.0.3"})
    _pairs(run, [(0.0, NODE_S), (4.0, NODE_S)])

    out = split.write(run)

    joined = out["node_offsets_s"]
    assert list(joined) == ["10.0.0.2", "10.0.0.3"]
    assert joined["10.0.0.2"]["source"] == "measured"
    assert joined["10.0.0.2"]["bounds_s"] == pytest.approx(
        [-NODE_S - 0.2, -NODE_S + 0.4]
    )
    assert joined["10.0.0.3"] == {
        "source": "bounds",
        "offset_s": pytest.approx(-NODE_S),
        "bounds_s": pytest.approx([-NODE_S - 0.3, -NODE_S]),
    }
    for r in out["requests"]:
        expected = AS_RUN[r["response_id"]]
        assert [r[k] for k in SEGMENTS] == pytest.approx(expected, abs=2e-9)


def test_a_request_no_later_forward_hands_over_is_unsplit(tmp_path):
    """The decode side's record ends at the forward naming B."""
    run = tmp_path / "run"
    _deployment(run)
    record = run / "decode" / "steps" / "dp0.jsonl"
    record.write_text("".join(record.read_text().splitlines(keepends=True)[:-1]))

    out = split.write(run)

    assert [r["response_id"] for r in out["requests"]] == ["a"]
    assert out["unsplit"]["xb"] == "no later forward hands its first token over"


def test_a_request_aiperf_finished_with_no_first_token_is_unsplit(tmp_path):
    """aiperf records a success with no content response with no TTFT."""
    run = tmp_path / "run"
    _deployment(run)
    rows = run / "artifacts" / "profile_export.jsonl"
    lines = [json.loads(line) for line in rows.read_text().splitlines()]
    del lines[1]["metrics"]["time_to_first_token"]
    rows.write_text("".join(json.dumps(line) + "\n" for line in lines))

    out = split.write(run)

    assert [r["response_id"] for r in out["requests"]] == ["a"]
    assert out["unsplit"]["xb"] == "aiperf records no first token for it"


def test_a_pipelines_record_is_refused(tmp_path):
    run = tmp_path / "run"
    _deployment(run)
    steps = run / "prefill" / "steps"
    (steps / "pp1.dp0.jsonl").write_text("")

    assert split.write(run) == {
        "refused": f"{steps} is a pipeline's record, whose first tokens leave a "
        "stage the split does not read"
    }


def _simulated_instance(run, monkeypatch, cost=None) -> list[dict]:
    """A `compass run` instance's record and cost lines, its forwards driven
    on a clock each moves by its charge and 2 ms more, `cost` applied to each
    cost line; each request's expected split, its first token handed over by
    the forward whose reply ATOM's scheduler first streamed it from."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(overrides, "_group_step_seconds", lambda *_: 0.0)
        driven = _drive()
    batches = driven.batches
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
        handed = windows[driven.first_offer[request] - 1][1]
        sent, first, end = e["entry"] - 0.25, handed + 0.01, e["last"] + 0.01
        requests.append((f"x{request}", f"r{request}", sent, first - sent, end - sent))
        e |= {"queue_s": 0.25, "prefill_s": e["return"] - e["entry"]}
        e |= {"decode_s": end - e["return"], "forward_ttft_s": handed - sent}
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


@pytest.mark.parametrize(
    "cost, why",
    [
        (lambda line: {**line, "t_s": None}, "{cost} holds no step times (t_s)"),
        (
            lambda line: {**line, "step": line["step"] + 1},
            "{record}'s cost line of step 1 stands beside step 0",
        ),
    ],
    ids=["no t_s", "another step's"],
)
def test_a_simulated_record_whose_cost_lines_do_not_time_its_steps_is_refused(
    tmp_path, monkeypatch, cost, why
):
    run = tmp_path / "run"
    _simulated_instance(run, monkeypatch, cost)

    out = split.write(run)

    steps = run / "steps"
    why = why.format(cost=steps / "dp0.cost.jsonl", record=steps / "dp0.jsonl")
    assert out == {"refused": why}


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
    assert (a["forward_ttft_s"], b["forward_ttft_s"]) == pytest.approx((1.62, 1.62))
