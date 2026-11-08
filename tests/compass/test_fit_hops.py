# SPDX-License-Identifier: MIT
"""`compass fit host` fits the host entry's ``hops`` from a recorded server's
intake stamps, its engine's queue handoffs and step lines, and the aiperf raw
records beside each client run's export, and `runfile.build` hands them to a
run's channels from the published entry."""

import gzip
import json

import pytest
from test_fit_host import TOKENIZER, _stamp, host_laws_key, stamped
from test_fit_steps import (  # noqa: F401
    HOST,
    clients,
    fit,
    member,
    planted_roots,
    record,
)
from test_run_hops import lookaheads, model, parsed

from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactStore
from atom.compass.backends.step_linear_fit import fit_intake
from atom.compass.drive import runfile
from atom.compass.parity.client_run import RAW

#: Each hop's planted time, ns, on every message alone on it.
PLANTED = {
    "admission_s": 540_000,
    "ipc_s": 107_000,
    "ipc_out_s": 244_000,
    "stream_s": 159_000,
}
#: What two messages in flight at once on a hop take, ns: far under any hop.
OVERLAP = 2_000
#: The wall clock aiperf pairs with its perf counter, less that counter.
WALL = 1_700_000_000_000_000_000


def hopped(directory, alone=10, raw=RAW + ".gz") -> None:
    """`directory`'s record with what its server and client leave of each hop:
    `alone` messages taking the hop's `PLANTED` time, then two in flight at
    once taking `OVERLAP`; on ``ipc_s`` each received inside a forward, and two
    more alone: one taking `OVERLAP`, received once the engine left its last,
    and one received before its send call returned. The client's raw records
    are written as `raw` beside the export."""
    steps = directory / "steps"
    forwards = [json.loads(x) for x in (steps / "dp0.jsonl").read_text().splitlines()]
    windows = [(s["t_enter_ns"], s["t_exit_ns"]) for s in forwards]
    long = [w for w in windows if w[1] - w[0] > 2 * PLANTED["ipc_s"]][: alone + 1]
    assert len(long) == alone + 1
    ipc = [(e + 10_000, e + 10_000 + PLANTED["ipc_s"]) for e, _ in long[:alone]]
    e = long[alone][0]
    ipc += [(e + 10_000, e + 10_000 + OVERLAP), (e + 11_000, e + 11_000 + OVERLAP)]
    last = windows[-1][1]
    ipc += [(last + 1_000, last + 1_000 + OVERLAP)]
    called = [s - 10 for s, _ in ipc] + [last + 10_000]
    ipc += [(last + 10_000 + OVERLAP, last + 10_000 + OVERLAP // 2)]

    def messages(hop, t):
        """The hop's sends and arrivals from `t` on, 10 ms apart."""
        out = [(t + k * 10**7, t + k * 10**7 + PLANTED[hop]) for k in range(alone)]
        t += alone * 10**7
        return out + [(t, t + 1_500), (t + 1_000, t + 3_000)]

    t = last + 10**9
    admission = messages("admission_s", t)
    outputs = messages("ipc_out_s", t + 10**9)
    stream = messages("stream_s", t + 2 * 10**9)
    lines = [_stamp("app", t - 10**8, t - 10**8)]  # a health probe: no body
    lines += [_stamp("send", c, s) for c, (s, _) in zip(called, ipc)]
    for _, a in admission:
        lines += [_stamp("app", a, a), _stamp("body", a + 100, a + 200, 9000)]
    lines += [_stamp("out", g, g + 10) for _, g in outputs]
    lines += [_stamp("head", s - 10, s) for s, _ in stream]
    (steps / compass_run.INTAKE_RECORD).write_text("\n".join(lines) + "\n")
    (steps / compass_run.TOKENIZER_RECORD).write_text(json.dumps(TOKENIZER) + "\n")
    handoff = [{"t_recv_ns": g, "prompt_tokens": 9} for _, g in ipc]
    handoff += [
        {"t_out_put_ns": p, "t_out_sent_ns": p + 1, "rows": 1} for p, _ in outputs
    ]
    text = "".join(json.dumps(h) + "\n" for h in handoff)
    (steps / "dp0.handoff.jsonl").write_text(text)
    # aiperf's records, each a send and the ack of a stream: one per request.
    records = [
        {
            "metadata": {"request_start_ns": WALL + c, "request_ack_ns": WALL + g},
            "start_perf_ns": c,
        }
        for (c, _), (_, g) in zip(admission, stream)
    ]
    text = "".join(json.dumps(r) + "\n" for r in records).encode()
    path = directory / raw
    path.write_bytes(gzip.compress(text) if raw.endswith(".gz") else text)


@pytest.mark.parametrize("raw", [RAW + ".gz", RAW])
def test_each_hop_is_the_p10_of_its_messages_alone_on_it_and_a_run_takes_it(
    tmp_path, raw
):
    a = record(tmp_path / "a", 19)
    hopped(a, raw=raw)
    fit(tmp_path, "fit host", a, *clients(a))

    key = host_laws_key(HOST)
    host = member(tmp_path / "store", key, "params")
    report = member(tmp_path / "store", key, "report")
    assert host["hops"] == pytest.approx({h: ns * 1e-9 for h, ns in PLANTED.items()})
    assert {h: report["hops"][h]["messages"] for h in PLANTED} == dict.fromkeys(
        PLANTED, 10
    )
    assert not {k for k in report["unfitted"] if k.startswith("hops")}
    # A run built on the published entry charges them on its channels.
    token = ArtifactStore(tmp_path / "store").read(key).key.token()
    run = runfile.build(parsed(tmp_path, "--host-laws", token), model(tmp_path))
    assert lookaheads(run) == pytest.approx(
        {
            "traffic->frontend:http": host["hops"]["admission_s"],
            "frontend->traffic:stream": host["hops"]["stream_s"],
            "frontend->engine:request#dp0": host["hops"]["ipc_s"],
            "frontend->engine:control#dp0": host["hops"]["ipc_s"],
            "engine->frontend:output#dp0": host["hops"]["ipc_out_s"],
        }
    )


def test_a_hop_the_records_cannot_give_is_left_out_and_listed_unfitted(tmp_path):
    a = record(tmp_path / "a", 19)
    hopped(a, raw="elsewhere.jsonl")
    (a / "steps/dp0.handoff.jsonl").write_text("")
    fit(tmp_path, "fit host", a, *clients(a))

    key = host_laws_key(HOST)
    host = member(tmp_path / "store", key, "params")
    report = member(tmp_path / "store", key, "report")
    stamps = a / "steps" / compass_run.INTAKE_RECORD
    assert "hops" not in host
    assert {k: v for k, v in report["unfitted"].items() if k.startswith("hops")} == {
        "hops.admission_s": f"{stamps}: no {RAW} beside {a / 'client.json'}",
        "hops.stream_s": f"{stamps}: no {RAW} beside {a / 'client.json'}",
        "hops.ipc_s": f"{stamps}: 0 arrivals, not one after each of 14 sends",
        "hops.ipc_out_s": f"{stamps}: 12 rows taken in, not the 0 put",
    }


def test_a_deployment_side_fits_its_engine_hops_and_not_its_client_s(tmp_path):
    d = record(tmp_path / "decode", 19)
    (tmp_path / "prefill").mkdir()
    hopped(d)
    fit(tmp_path, "fit host", d, *clients(d))

    key = host_laws_key(HOST)
    host = member(tmp_path / "store", key, "params")
    report = member(tmp_path / "store", key, "report")
    assert set(host["hops"]) == {"ipc_s", "ipc_out_s"}
    behind = f"{d / 'steps' / compass_run.INTAKE_RECORD}: {d} is a side of a deployment, behind its router"
    assert {k: v for k, v in report["unfitted"].items() if k.startswith("hops")} == {
        "hops.admission_s": behind,
        "hops.stream_s": behind,
    }


def test_rows_taken_in_and_heads_sent_leave_the_intake_fit_as_it_was(tmp_path):
    a = record(tmp_path / "a", 19)
    stamped(a, burst=(40_000, 60_000, 50_000, 45_000))
    path = a / "steps" / compass_run.INTAKE_RECORD
    before = fit_intake([path])
    # The output thread takes rows in, and the loop sends other requests'
    # heads, while each request is taken in.
    lines = path.read_text().splitlines()
    apps = [int(x.split()[3]) for x in lines if x.split()[2] == "app"]
    lines += [
        _stamp(kind, t + 1, t + 2, thread=4) for t in apps for kind in ("out", "head")
    ]
    path.write_text("\n".join(lines) + "\n")

    assert fit_intake([path]) == before
