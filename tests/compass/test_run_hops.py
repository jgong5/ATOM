# SPDX-License-Identifier: MIT
"""The hops a run's channels take: a ``host`` entry's ``hops``, which
`drive.runfile` writes to the run-file keys of their names, each side of a
prefill-decode run from its own entry, and the floor where none is stated,
which the run summary names.

Each run file is built from the flags `compass run` and `deploy.sh run` take
and read back through `run.channel_table`, as the traffic LP reads it.
"""

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import platforms
import pytest
from test_memory_readings import DOCUMENT
from test_run_step_grid import deployment
from test_served_tokenizer import REPO, _run_file

from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients
from atom.compass.drive import runfile
from atom.compass.run.step_grid import keyed
from atom.utils import clock
from atom.utils.clock import LPRuntime

LAWS = REPO / "scripts/compass/laws"
#: The node-18 TP1 law file, which a TP1 1P1D cell runs.
TP1 = LAWS / "qwen3.8-27b-tp1-mi308x-ptl-disabled-661ef7661.json"
FLOOR = compass_run.HOP_FLOOR_S
#: Each channel's lookahead before the hops left the code and the spec: the
#: serving path's HTTP transport, the ZMQ request and output and the SSE stream
#: as node-18 medians, the write request as a node-18 p10, and the router's
#: forward, assumed.
ROUTER = 1.5e-3
SINGLE = {
    "traffic->frontend:http": 0.54e-3,
    "frontend->traffic:stream": 159e-6,
    "frontend->engine:request#dp0": 107e-6,
    "frontend->engine:control#dp0": 107e-6,
    "engine->frontend:output#dp0": 244e-6,
}
PREFILL_DECODE = {
    "traffic->frontend-P:http": 0.54e-3 + ROUTER,
    "frontend-D->traffic:stream": 159e-6 + ROUTER,
    "frontend-P->frontend-D:relay": ROUTER,
    "engine-D->engine-P:kv_write_req": 106e-6,
    **{
        f"frontend-{s}->engine-{s}:{kind}#dp0": 107e-6
        for s in "PD"
        for kind in ("request", "control")
    },
    **{f"engine-{s}->frontend-{s}:output#dp0": 244e-6 for s in "PD"},
}


def lookaheads(run: dict) -> dict:
    table = compass_run.channel_table(run)
    return {
        c.name: c.lookahead_s for lp in table.registry for c in table.channels_from(lp)
    }


def parsed(tmp_path, *flags):
    """`flags` after a step stub, the suite's machine, at the stack a law file's
    host laws state, and the store."""
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))
    parser = argparse.ArgumentParser()
    runfile.add_arguments(parser)
    common = ["--step-stub", str(stub)]
    laws = [json.loads(Path(f).read_text()) for f in flags if f.endswith(".json")]
    at = next(
        (law["host"]["key"] for law in laws if "torch" in law["host"]["key"]), None
    )
    common += platforms.flags(tmp_path / "store", DOCUMENT, tmp_path, at)
    return parser.parse_args([*common, "--store", str(tmp_path / "store"), *flags])


def model(tmp_path, *more):
    return ["--model", str(tmp_path), *more]


#: Law files whose ``host.hops`` `compass fit host` fitted on their own records
#: (#3036), which carry their cell's hops rather than node 18's.
FITTED_HOPS = {"deepseek-v4.1-flash-tp4-ep4-dspark5-mi355x-gpu40.json"}


@pytest.mark.parametrize(
    "law", sorted(p.name for p in LAWS.glob("*.json") if p.name not in FITTED_HOPS)
)
def test_each_law_file_builds_the_lookaheads_a_run_charged_before(tmp_path, law):
    args = parsed(tmp_path, "--host-laws", str(LAWS / law))

    run = runfile.build(args, model(tmp_path))

    assert lookaheads(run) == pytest.approx(SINGLE)
    assert None not in compass_run.hops(run).values()
    # The entry's router hop would make the run prefill-decode.
    assert "router_s" not in run


@pytest.mark.parametrize("law", sorted(FITTED_HOPS))
def test_a_law_file_with_fitted_hops_builds_its_own_lookaheads(tmp_path, law):
    hops = json.loads((LAWS / law).read_text())["host"]["hops"]
    args = parsed(tmp_path, "--host-laws", str(LAWS / law))

    run = runfile.build(args, model(tmp_path))

    assert lookaheads(run) == pytest.approx(
        {
            "traffic->frontend:http": hops["admission_s"],
            "frontend->traffic:stream": hops["stream_s"],
            "frontend->engine:request#dp0": hops["ipc_s"],
            "frontend->engine:control#dp0": hops["ipc_s"],
            "engine->frontend:output#dp0": hops["ipc_out_s"],
        }
    )
    assert None not in compass_run.hops(run).values()


def test_a_tp1_prefill_decode_run_builds_the_lookaheads_it_charged_before(tmp_path):
    args = parsed(tmp_path, "--host-laws", str(TP1))

    run = runfile.prefill_decode(args, model(tmp_path), model(tmp_path), "intra_node")

    assert lookaheads(run) == pytest.approx(PREFILL_DECODE)
    assert None not in compass_run.hops(run).values()
    # The write request is prefill's receipt, so prefill's entry states it.
    assert run["kv_write_req_s"] == 106e-6


def entry(tmp_path, name, r, hops: dict) -> str:
    """A law file whose host entry, fitted under `r`'s engine config, states `hops`."""
    host = {"key": {"engine": keyed(r.config)}, "laws": {}, "hops": hops}
    (tmp_path / name).write_text(json.dumps({"host": host}))
    return str(tmp_path / name)


def test_each_side_of_a_prefill_decode_run_takes_the_hops_fitted_under_its_config(
    monkeypatch, tmp_path
):
    p, d = (deployment(tmp_path, kv_cache_block_size=b) for b in (16, 32))
    hops_p = {
        "admission_s": 1e-3,
        "ipc_s": 2e-4,
        "ipc_out_s": 3e-4,
        "kv_write_req_s": 4e-4,
        "router_s": 2e-3,
    }
    hops_d = {
        "admission_s": 9e-3,
        "ipc_s": 5e-4,
        "ipc_out_s": 6e-4,
        "stream_s": 7e-4,
        "router_s": 8e-3,
    }
    laws = [entry(tmp_path, "p.json", p, hops_p), entry(tmp_path, "d.json", d, hops_d)]
    sides = {"P": model(tmp_path, "-dp", "1"), "D": model(tmp_path)}
    # Each side's config as resolved on the machine's arch, the only one stated.
    gfx = DOCUMENT["device"]["arch"]
    configs = {(tuple(sides["P"]), gfx): p.config, (tuple(sides["D"]), gfx): d.config}
    monkeypatch.setattr(
        "atom.compass.drive.run._config", lambda a, arch: configs[tuple(a), arch]
    )

    for order in (laws, laws[::-1]):
        flags = [f for law in order for f in ("--host-laws", law)]
        run = runfile.prefill_decode(
            parsed(tmp_path, *flags), sides["P"], sides["D"], "intra_node"
        )
        got = lookaheads(run)

        assert {k: run[k] for k in hops_p} == hops_p
        assert run[compass_run.DECODE_HOPS] == hops_d
        assert got["traffic->frontend-P:http"] == pytest.approx(1e-3 + 2e-3)
        assert got["frontend-D->traffic:stream"] == pytest.approx(7e-4 + 2e-3)
        assert got["frontend-P->frontend-D:relay"] == 2e-3
        assert got["engine-D->engine-P:kv_write_req"] == 4e-4
        assert got["frontend-P->engine-P:request#dp0"] == 2e-4
        assert got["engine-P->frontend-P:output#dp0"] == 3e-4
        assert got["frontend-D->engine-D:request#dp0"] == 5e-4
        assert got["engine-D->frontend-D:output#dp0"] == 6e-4
        # Prefill's entry states no stream, decode's no write request, and
        # decode's router hop is not read either.
        assert None not in compass_run.hops(run).values()


def test_an_entry_with_no_router_hop_charges_its_floor_and_lists_it(tmp_path):
    law = tmp_path / "law.json"
    law.write_text(
        json.dumps({"host": {"key": {}, "laws": {}, "hops": {"ipc_s": 1e-4}}})
    )
    args = parsed(tmp_path, "--host-laws", str(law))

    run = runfile.prefill_decode(args, model(tmp_path), model(tmp_path), "intra_node")

    assert run["router_s"] is None
    assert lookaheads(run)["frontend-P->frontend-D:relay"] == FLOOR
    assert compass_run.hops(run)["router_s"] is None
    assert compass_run.hops(run)["ipc_s"] == 1e-4


def test_a_router_lp_whose_router_hop_is_the_floor_refuses(tmp_path):
    run = runfile.prefill_decode(
        parsed(tmp_path), model(tmp_path), model(tmp_path), "intra_node"
    )
    run["router_in_s"] = compass_run.hop(run, "ipc_s")

    with pytest.raises(ValueError, match="0 < router_in_s < router_s"):
        compass_run.channel_table(run)


def test_without_host_laws_every_hop_takes_the_floor(tmp_path):
    single = runfile.build(parsed(tmp_path), model(tmp_path))
    pd = runfile.prefill_decode(
        parsed(tmp_path), model(tmp_path), model(tmp_path), "intra_node"
    )

    assert lookaheads(single) == {name: FLOOR for name in SINGLE}
    assert lookaheads(pd) == pytest.approx(
        {
            name: FLOOR * (1 + (":http" in name or ":stream" in name))
            for name in PREFILL_DECODE
        }
    )
    assert sorted(compass_run.hops(pd)) == [
        "admission_s",
        "decode_hops.ipc_out_s",
        "decode_hops.ipc_s",
        "decode_hops.stream_s",
        "ipc_out_s",
        "ipc_s",
        "kv_write_req_s",
        "router_s",
    ]
    assert set(compass_run.hops(pd).values()) == {None}


def test_a_run_file_with_an_admission_path_runs_as_one_without(monkeypatch, tmp_path):
    run = runfile.build(parsed(tmp_path, "--host-laws", str(TP1)), model(tmp_path))
    for path in ("serving", "offline_batch"):
        _run_file(monkeypatch, tmp_path, run | {"admission_path": path})
        assert lookaheads(compass_run.spec()) == lookaheads(run)


@pytest.mark.parametrize(
    "hops", [{"ipc_s": 0.0}, {"ipc_s": -1e-4}, {"relay_s": 1e-3}, {"ipc_s": "1e-4"}]
)
def test_a_host_entry_with_a_hop_a_channel_cannot_take_is_refused(
    monkeypatch, tmp_path, hops
):
    _run_file(monkeypatch, tmp_path, {"host": {"hops": hops}})
    with pytest.raises(ValueError, match="host's hops is"):
        compass_run.spec()


@pytest.mark.parametrize(
    "stated, named",
    [
        ({}, ["admission_s", "ipc_out_s", "ipc_s", "stream_s"]),
        ({"ipc_s": 1e-4, "stream_s": 2e-4, "admission_s": 3e-4}, ["ipc_out_s"]),
        ({"ipc_s": 1e-4, "stream_s": 2e-4, "admission_s": 3e-4, "ipc_out_s": 4e-4}, []),
    ],
    ids=["none", "no-output", "all"],
)
def test_the_summary_names_each_hop_charged_the_floor(
    monkeypatch, tmp_path, stated, named
):
    run = {"clock_endpoint": "inproc:unused", "out_dir": str(tmp_path)} | stated
    run = _run_file(monkeypatch, tmp_path, run)
    authority = compass_run._RecordingAuthority(run)
    authority.started = authority.finished = time.monotonic()
    monkeypatch.setattr(compass_run, "_authority", authority)
    compass_run._hand_in(run, compass_run.ENGINE.name, [])
    frontend = LPRuntime(compass_run.FRONTEND, compass_run.channel_table(run), None)
    frontend.now = math.inf
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]))
    monkeypatch.setattr(clock, "_installed", frontend)

    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))

    summary = json.loads((tmp_path / compass_run.SUMMARY_FILE).read_text())
    assert summary["unmeasured_hops"] == named
