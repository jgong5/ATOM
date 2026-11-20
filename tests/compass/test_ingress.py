# SPDX-License-Identifier: MIT
"""The encoding ingress's hop: the ``ingress`` term (`atom.compass.ingress`),
the queue `Ingress` prices it on, and the run file a host entry stating one
builds (`atom.compass.drive.runfile`), read back through `run.spec`.

The traffic LP's pricing on the clock, and the worker's size report, are
`test_harness_traffic_lp.py`'s and `test_harness_transport.py`'s.
"""

import argparse
import dataclasses
import json
import math

import platforms
import pytest
from test_memory_readings import DOCUMENT

from atom.compass import ingress as compass_ingress
from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients
from atom.compass.drive import runfile
from atom.compass.ingress import Ingress, check

#: An ingress of two threads, its terms dyadic so its sums are exact.
TERM = {"fixed_s": 0.5, "per_byte_s": 2.0**-10, "workers": 2}
ROUTED = {**TERM, "router_per_byte_s": 2.0**-13}
L = 2.0**-14  # the HTTP channel's lookahead


@pytest.mark.parametrize(
    "entry",
    [
        TERM,
        ROUTED,
        {**ROUTED, "bytes": [100, 3_000_000]},
        {**TERM, "fixed_s": 0, "per_byte_s": 0},
    ],
    ids=["bare", "router", "hull", "zero"],
)
def test_an_ingress_term_states_its_thread_law_threads_router_and_hull(entry):
    assert check(entry) is entry


@pytest.mark.parametrize(
    "entry",
    [
        {k: v for k, v in TERM.items() if k != "workers"},
        {**TERM, "workers": 0},
        {**TERM, "workers": 2.0},
        {**TERM, "per_byte_s": -1e-9},
        {**TERM, "fixed_s": math.inf},
        {**TERM, "router_per_byte_s": "fast"},
        {**TERM, "bytes": [5, 4]},
        {**TERM, "bytes": [5]},
        {**TERM, "queued": 1280},
        [TERM],
    ],
    ids=[
        "no-workers",
        "no-thread",
        "float-workers",
        "negative",
        "infinite",
        "no-number",
        "hull-backwards",
        "hull-short",
        "unknown-term",
        "no-mapping",
    ],
)
def test_an_ingress_term_refuses_what_it_does_not_take(entry):
    with pytest.raises(ValueError, match="takes fixed_s and per_byte_s"):
        check(entry, "here")


def test_a_lone_request_holds_a_thread_for_its_body_then_the_router_parses_it():
    """Sent at 1.0 with a 1 KiB body: at the ingress after the lookahead, on a
    thread 0.5 s plus 1 s, then the router's parse of 1/8 s."""
    assert Ingress(ROUTED, L).admit(1.0, 1024) == (1.0 + L, 1.0 + L + 1.5 + 0.125)


def test_requests_past_the_free_threads_wait_for_the_first_to_come_free():
    """Three requests at 0 on two threads: the third, the smallest, waits for
    the first thread to finish, first come first served whatever its size; a
    request sent once both are free waits for none."""
    ingress = Ingress(TERM, L)
    got = [ingress.admit(0.0, n) for n in (1024, 2048, 512)]
    assert got == [(L, L + 1.5), (L, L + 2.5), (L, L + 1.5 + 1.0)]
    assert ingress.admit(10.0, 0) == (10.0 + L, 10.0 + L + 0.5)


def test_a_request_with_no_body_takes_no_thread_and_reaches_the_router_at_once():
    ingress = Ingress({**TERM, "workers": 1}, L)
    assert ingress.admit(0.0, None) == (L, L)
    assert ingress.admit(0.0, 1024) == (L, L + 1.5)
    assert ingress.admit(0.0, None) == (L, L)


def test_the_same_requests_in_the_same_order_give_the_same_arrivals():
    sizes = [(k * 0.25, (k * 7919) % 40_000) for k in range(200)]
    runs = [
        [Ingress({**ROUTED, "workers": 3}, L).admit(t, n) for t, n in sizes]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_a_run_file_states_an_ingress_or_none(tmp_path, monkeypatch):
    assert compass_ingress.RUN_ENV == compass_run.ENV
    assert compass_ingress.stated(None) is None
    assert compass_ingress.stated({"bound_s": 1.0}) is None
    assert compass_ingress.stated({"ingress": TERM}) == TERM
    with pytest.raises(ValueError, match="the run file's ingress is"):
        compass_ingress.stated({"ingress": {**TERM, "workers": 0}})
    monkeypatch.delenv(compass_run.ENV, raising=False)
    assert compass_ingress.stated_here() is None
    path = tmp_path / "run_file.json"
    path.write_text(json.dumps({"bound_s": 1.0, "ingress": ROUTED}))
    monkeypatch.setenv(compass_run.ENV, str(path))
    assert compass_ingress.stated_here() == ROUTED


@pytest.mark.parametrize("where", ["top", "host"])
def test_run_spec_refuses_a_run_file_whose_ingress_is_no_term(
    where, tmp_path, monkeypatch
):
    bad = {**TERM, "per_byte_s": -1.0}
    run = {"bound_s": 1.0}
    if where == "top":
        run["ingress"] = bad
    else:
        run["host"] = {"ingress": bad}
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    with pytest.raises(ValueError, match="ingress is .* it takes fixed_s"):
        compass_run.spec()


def test_run_spec_reads_a_host_entry_s_ingress_term(tmp_path, monkeypatch):
    run = {"bound_s": 1.0, "host": {"hops": {"router_s": 1e-3}, "ingress": TERM}}
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    assert compass_run.spec()["host"]["ingress"] == TERM


def _pd_run(tmp_path, host: dict | None) -> dict:
    """A prefill-decode run file `runfile.prefill_decode` builds on the step
    stub, with the law file `host` as its host laws if given."""
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))
    parser = argparse.ArgumentParser()
    runfile.add_arguments(parser)
    flags = ["--step-stub", str(stub), *platforms.network(DOCUMENT, tmp_path)]
    flags += platforms.flags(tmp_path / "store", DOCUMENT, tmp_path)
    if host is not None:
        (tmp_path / "host.json").write_text(json.dumps({"host": host}))
        flags += ["--host-laws", str(tmp_path / "host.json")]
    args = parser.parse_args([*flags, "--store", str(tmp_path / "store")])
    model = ["--model", str(tmp_path)]
    return runfile.prefill_decode(args, model, model, "inter_node")


def test_a_host_entry_s_ingress_goes_to_the_run_file_beside_its_router_hop(
    tmp_path, monkeypatch
):
    """A law file's host, as a cell's stub states it, with an ingress: the run
    file names it, and it is the term the traffic LP reads; without one it
    names none, the channel table as it was."""
    hops = {"router_s": 0.0015, "kv_write_req_s": 0.000106}
    run = _pd_run(tmp_path, {"hops": hops, "ingress": ROUTED})
    assert run["ingress"] == ROUTED
    assert run["router_s"] == 0.0015
    plain = _pd_run(tmp_path, {"hops": hops})
    assert "ingress" not in plain
    assert {k: v for k, v in run.items() if k not in ("ingress", "host")} == {
        k: v for k, v in plain.items() if k != "host"
    }
    (tmp_path / "run.json").write_text(json.dumps({**run, "bound_s": 1.0}))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    assert compass_ingress.stated_here() == ROUTED
    table = compass_run.channel_table(compass_run.spec())
    assert table.lookahead("traffic->frontend-P:http") == compass_run.channel_table(
        plain
    ).lookahead("traffic->frontend-P:http")
