# SPDX-License-Identifier: MIT
"""The encoding ingress's hop: the ``ingress`` term (`atom.compass.ingress`),
the queue `Ingress` prices it on, `compass fit ingress` on an ingress's stamps
(`atom.compass.fit.ingress`), and the run file a host entry stating one builds
(`atom.compass.drive.runfile`), read back through `run.spec`.

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

from atom.compass import cli
from atom.compass import ingress as compass_ingress
from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients
from atom.compass.drive import runfile
from atom.compass.fit.ingress import fit_ingress, router_term, samples, stamps
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


# ---- compass fit ingress ----


def _row(recv: int, nbytes: int, held_ns: int, proc=0, **more) -> dict:
    """An encoded request's stamp line: received at `recv` ns, its body
    `nbytes`, holding its thread `held_ns`, a fifth of it outside the encode."""
    start = recv + 2_000
    encode = held_ns - held_ns // 5
    return {
        "recv": recv,
        "proc": proc,
        "body": recv + 1_000,
        "bytes": nbytes,
        "start": start,
        "parsed": start + held_ns // 10,
        "rendered": start + held_ns // 5,
        "encoded": start + held_ns // 5 + encode,
        "serialized": start + held_ns,
        "kind": "encoded",
        "tokens": nbytes // 4,
        "sent": start + held_ns + 3_000,
        "status": 200,
        **more,
    }


#: Encoded requests whose threads hold 1 ms + 250 ns a byte, and a cancel and
#: a health probe the ingress passed through.
LAW = (1e-3, 250e-9)
ROWS = [
    _row(k * 10**8, n, round(1e9 * (LAW[0] + LAW[1] * n)), proc=k % 2)
    for k, n in enumerate([1000, 40_000, 400_000, 9_000, 120_000, 2_000_000])
] + [
    {"recv": 7, "kind": "passthrough", "bytes": 23, "sent": 9, "status": 204},
    {"recv": 8, "kind": "passthrough", "bytes": 0, "path": "/health", "sent": 9},
]


def _write(tmp_path, rows, split=2) -> str:
    """`rows` as the ingress's processes write them, ``<prefix>.<pid>``; the prefix."""
    prefix = tmp_path / "stamps"
    for k in range(split):
        lines = [json.dumps(r) for r in rows[k::split]]
        (tmp_path / f"stamps.{4000 + k}").write_text("\n".join(lines) + "\n")
    return str(prefix)


def test_a_linear_thread_law_fits_back_exactly_with_its_hull_and_threads():
    term, report = fit_ingress(samples(ROWS), 64)
    assert term["fixed_s"] == pytest.approx(LAW[0], rel=1e-6)
    assert term["per_byte_s"] == pytest.approx(LAW[1], rel=1e-6)
    assert term["workers"] == 64 and term["bytes"] == [1000, 2_000_000]
    assert report["method"] == "least squares" and report["requests"] == 6
    assert report["thread_ms"]["law"] == pytest.approx(report["thread_ms"]["measured"])
    assert report["bytes_per_token"] == pytest.approx(4.0, rel=1e-3)
    assert report["most_threads_busy"] == {"0": 1, "1": 1}


def test_an_encode_growing_faster_than_its_body_fits_through_zero_at_its_mean():
    """Threads that hold 1e-15 s a byte squared: least squares would put
    fixed_s below 0, so it is 0 and per_byte_s keeps the samples' mean."""
    sizes = [10_000 * k for k in range(1, 30)]
    rows = [_row(k, n, round(1e9 * 1e-15 * n * n)) for k, n in enumerate(sizes)]
    term, report = fit_ingress(samples(rows), 4)
    held = [1e-15 * n * n for n in sizes]
    assert term["fixed_s"] == 0.0
    assert term["per_byte_s"] == pytest.approx(sum(held) / sum(sizes), rel=1e-6)
    assert report["least_squares"]["fixed_s"] < 0
    assert report["method"].startswith("least squares puts fixed_s at -")
    assert report["thread_ms"]["law"]["mean"] == pytest.approx(
        report["thread_ms"]["measured"]["mean"]
    )


def test_skip_leaves_out_the_first_requests_by_receipt():
    chosen = samples(list(reversed(ROWS)), skip=2)
    assert [r["bytes"] for r in chosen] == [400_000, 9_000, 120_000, 2_000_000]


def test_stamps_read_a_prefix_s_files_and_a_last_line_cut_short(tmp_path):
    prefix = _write(tmp_path, ROWS)
    with open(f"{prefix}.4001", "a") as f:
        f.write('{"recv": 99, "kind": "enc')  # the ingress was stopped mid-write
    assert sorted(map(json.dumps, stamps([prefix]))) == sorted(map(json.dumps, ROWS))
    assert len(stamps([f"{prefix}.4000"])) == 4
    with pytest.raises(ValueError, match="neither a stamps file nor the prefix"):
        stamps([str(tmp_path / "none")])
    (tmp_path / "bad.1").write_text('{"recv": 1\n{}\n')
    with pytest.raises(ValueError, match=r"bad.1:1: not a stamp line"):
        stamps([str(tmp_path / "bad")])


def test_a_sample_without_every_stage_refuses():
    row = {k: v for k, v in ROWS[0].items() if k != "rendered"}
    with pytest.raises(ValueError, match="1 encoded requests lack a stamp"):
        samples([row, *ROWS[1:]])


def _metrics(seconds: float, count: int) -> str:
    return (
        "# TYPE mesh_prepare_stage_seconds summary\n"
        f'mesh_prepare_stage_seconds{{stage="json_parse",quantile="0.5"}} 0\n'
        f'mesh_prepare_stage_seconds_sum{{stage="json_parse"}} {seconds}\n'
        f'mesh_prepare_stage_seconds_count{{stage="json_parse"}} {count}\n'
        'mesh_http_requests_total{method="POST",path="/v1/chat/completions"} 7\n'
    )


def test_the_router_term_is_atomesh_s_parse_seconds_over_the_client_bytes():
    """atomesh parsed the six requests the ingress sent it, not the cancel it
    answered itself; a fallback, sent on as it came, counts too."""
    nbytes = sum(r["bytes"] for r in ROWS[:6])
    per_byte, report = router_term(ROWS, _metrics(0.5, 6))
    assert per_byte == 0.5 / nbytes
    assert report["parsed"] == 6 and report["mean_parse_ms"] == pytest.approx(500 / 6)
    fallback = {**ROWS[0], "kind": "fallback", "bytes": 10}
    assert router_term([*ROWS, fallback], _metrics(1.0, 7))[0] == 1.0 / (nbytes + 10)
    with pytest.raises(ValueError, match="atomesh parsed 5 requests .* sent it 6"):
        router_term(ROWS, _metrics(0.5, 5))
    with pytest.raises(ValueError, match="not atomesh's /metrics"):
        router_term(ROWS, "mesh_http_requests_total 1\n")


def test_a_forwarded_body_s_size_gives_the_router_its_rate_a_forwarded_byte():
    rows = [{**r, "sent_bytes": 3 * r["bytes"]} for r in ROWS[:6]]
    _, report = fit_ingress(samples(rows), 64)
    assert report["forwarded_bytes_per_byte"] == 3.0
    per_byte, router = router_term(rows, _metrics(0.5, 6))
    assert router["per_forwarded_byte_s"] == pytest.approx(per_byte / 3)


def test_compass_fit_ingress_writes_the_term_a_host_entry_states(tmp_path, capsys):
    prefix = _write(tmp_path, ROWS)
    (tmp_path / "metrics.prom").write_text(_metrics(0.5, 6))
    out = tmp_path / "ingress.json"
    cli.main(
        [
            *("fit", "ingress", prefix, "--skip", "1", "--processes", "2"),
            *("--workers", "8", "--metrics", str(tmp_path / "metrics.prom")),
            *("--out", str(out)),
        ]
    )
    got = json.loads(out.read_text())
    term = check(got["ingress"])
    assert term["workers"] == 16 and term["bytes"] == [9_000, 2_000_000]
    assert term["fixed_s"] == pytest.approx(LAW[0], rel=1e-6)
    assert term["router_per_byte_s"] == 0.5 / sum(r["bytes"] for r in ROWS[:6])
    assert got["report"]["skip"] == 1 and got["report"]["requests"] == 5
    assert json.loads(capsys.readouterr().out)["ingress"] == term
    cli.main(["fit", "ingress", prefix, "--out", str(out)])
    got = json.loads(out.read_text())
    assert "router_per_byte_s" not in got["ingress"] and got["ingress"]["workers"] == 64
    assert got["report"]["unfitted"] == {"router_per_byte_s": "no --metrics"}
    from atom.compass.fit.host import stated_ingress

    assert stated_ingress(str(out)) == got["ingress"]
    (tmp_path / "other.json").write_text(json.dumps({"host": {}}))
    with pytest.raises(ValueError, match="holds no ingress term"):
        stated_ingress(str(tmp_path / "other.json"))


def test_a_diagnostic_client_ingress_stands_in_for_a_run_file_with_none(monkeypatch):
    # COMPASS_DIAG_CLIENT_INGRESS: a run file with no ``ingress`` takes it,
    # checked; one that states its own keeps that; a real run has none.
    monkeypatch.setenv(compass_ingress.DIAG_ENV, json.dumps(TERM))
    assert compass_ingress.stated({"bound_s": 1.0}) == TERM
    assert compass_ingress.stated({"ingress": ROUTED}) == ROUTED
    assert compass_ingress.stated(None) is None
    monkeypatch.setenv(compass_ingress.DIAG_ENV, json.dumps({**TERM, "workers": 0}))
    with pytest.raises(ValueError, match=compass_ingress.DIAG_ENV):
        compass_ingress.stated({"bound_s": 1.0})
    monkeypatch.setenv(compass_ingress.DIAG_ENV, "")
    assert compass_ingress.stated({"bound_s": 1.0}) is None
