# SPDX-License-Identifier: MIT
"""`compass fit host`, run through `cli.main` on `test_fit_steps.py`'s record
directories, and the host laws it publishes handed to `run.runner` in a run
file on `test_run_step_grid.py`'s deployment."""

import json
import re

import numpy as np
import pytest
from test_backend_step_labels import pulled
from test_fit_samples import edit
from test_fit_steps import (  # noqa: F401
    DIGEST,
    ENGINE,
    HOST,
    clients,
    fit,
    member,
    planted_roots,
    record,
    served,
)
from test_memory_readings import META_8B
from test_run_step_grid import B37, decode, deployment, publish, run_file

from atom.compass import run as compass_run
from atom.compass.artifacts import Key, Kind
from atom.compass.backends.step_labels import digest
from atom.compass.backends.step_linear_fit import (
    fit_host,
    fit_tokenizer,
    label_records,
)
from atom.compass.drive.aiter_jit import BUILDS
from atom.compass.runner.overrides import OUT_SEGMENT, forward_shape


def host_laws_key(key) -> Key:
    return Key.of(
        Kind.HOST_LAWS,
        host_cpu=key["host_cpu"],
        atom_src=key["atom"],
        model=key["model"],
        engine_config=json.dumps(key["engine"], sort_keys=True),
        machine_spec=DIGEST,
    )


def test_the_host_laws_published_are_fit_hosts_and_a_run_file_carrying_them_runs(
    tmp_path, monkeypatch
):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = compass_run.host_key(r.config)
    a = record(tmp_path / "a", 19, key=key, trace="trace-a")
    b = record(tmp_path / "b", 20, key=key, trace="trace-b")
    fit(tmp_path, "fit host", a, b, *clients(a, b))

    host = member(tmp_path / "store", host_laws_key(key), "params")
    # In run digest order, as the command reads them: a least-squares fit's
    # last bits depend on the order of its samples.
    paths = sorted((d / "steps/dp0.jsonl" for d in (a, b)), key=digest)
    laws, report = fit_host(label_records(paths)[0])
    assert host == {**laws, "key": {**laws["key"], "machine_spec": DIGEST}}
    # Records with no tokenizer calls or intake stamps give neither term.
    report["unfitted"].update(
        tokenizer=f"no record holds steps/{compass_run.TOKENIZER_RECORD}",
        http_intake=f"no record holds steps/{compass_run.INTAKE_RECORD}",
    )
    assert member(tmp_path / "store", host_laws_key(key), "report") == {
        "provenance": served(key, {a: "trace-a", b: "trace-b"}),
        **report,
    }
    run = run_file(tmp_path, monkeypatch, host=host)
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    assert r.compass_host.laws == host["laws"]


def test_a_record_without_its_host_key_refuses(tmp_path):
    a = record(tmp_path / "a", 19, keyed=False)

    named = f"{digest(a / 'steps/dp0.jsonl')} holds steps with no host labels under a host key"
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit host", a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_a_record_whose_server_built_an_aiter_module_refuses_naming_it(tmp_path):
    a = record(tmp_path / "a", 19)
    edit(a / "run.json", lambda r: r.update({BUILDS: ["mha_varlen_fwd"]}))

    named = (
        f"{a} is not fitted: its server ran a JIT compile of ['mha_varlen_fwd'] while"
    )
    with pytest.raises(SystemExit, match=re.escape(named)):
        fit(tmp_path, "fit host", a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_records_with_pull_stamps_give_every_kind_an_E_out_law_a_run_prices(
    tmp_path, monkeypatch
):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = compass_run.host_key(r.config)
    a = record(tmp_path / "a", 19, key=key)
    pulled(a / "steps/dp0.jsonl")
    fit(tmp_path, "fit host", a, *clients(a))

    host = member(tmp_path / "store", host_laws_key(key), "params")
    assert {k: OUT_SEGMENT in v for k, v in host["laws"].items()} == {
        k: True for k in ("decode", "prefill", "middle")
    }
    run = run_file(tmp_path, monkeypatch, host=host)
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    batch = decode(B37)
    kind, x = forward_shape(batch)
    out_s = np.dot(host["laws"][kind][OUT_SEGMENT], x)
    assert r.forward(batch).out_s == pytest.approx(out_s) and out_s > 0


def test_a_dp_pair_s_host_laws_start_E_at_the_group_s_latest_pull(tmp_path):
    key = {**HOST, "engine": {**ENGINE, "data_parallel_size": 2}}
    a = record(tmp_path / "a", 19, key=key)
    steps = a / "steps"
    for suffix in (".key.json", ".engine.json"):
        (steps / f"dp1{suffix}").write_text((steps / f"dp0{suffix}").read_text())
    rank0 = [json.loads(x) for x in (steps / "dp0.jsonl").open()]
    rank1 = "".join(json.dumps(x | {"dp_rank": 1}) + "\n" for x in rank0)
    (steps / "dp1.jsonl").write_text(rank1)
    # The same forwards on both ranks; rank 1 always pulls after rank 0.
    paths = [steps / "dp0.jsonl", steps / "dp1.jsonl"]
    for path, parts in zip(paths, ((3,), (2,))):
        pulled(path, parts)
    fit(tmp_path, "fit host", a, *clients(a))

    laws = member(tmp_path / "store", host_laws_key(key), "params")["laws"]
    alone = [fit_host(label_records([p])[0])[0]["laws"] for p in paths]
    assert set(laws) == {"decode", "prefill", "middle"}
    for kind, law in laws.items():
        assert law["E"] == pytest.approx(alone[1][kind]["E"], rel=1e-9)
        assert law["E"] != pytest.approx(alone[0][kind]["E"], rel=1e-3)


#: The planted tokenizer and intake terms the stamped records below are timed by.
ENCODE, DECODE = (5e-6, 9e5), (1e-6, 7.5e6)
IDLE, PER_BYTE, FIXED = 1.1e-3, 8.8e-9, 4.3e-3
TOKENIZER = {"backend": "fast", "fingerprint": "sha256:tok"}


def stamped(directory, lone=(30_000, 400_000), burst=(), rates=(ENCODE, DECODE)):
    """`directory`'s steps with the tokenizer calls and HTTP intake stamps a
    recorded server writes: calls timed at `rates`, the encode's and the
    decode's, a lone request per body size in `lone`, timed at `IDLE` and
    `PER_BYTE`, then one burst of `burst`'s body sizes, its templates
    `FIXED + PER_BYTE x bytes` apart."""
    steps, (enc, dec) = directory / "steps", rates
    calls = [json.dumps(TOKENIZER)]
    for i, (kind, tokens) in enumerate(
        [("encode", 9000), ("encode_batch", 120), ("encode", 88_768)]
        + [("decode", n) for n in (1, 5, 7, 40)]
    ):
        fixed, rate = dec if kind == "decode" else enc
        start = 10**12 + i * 10**9
        end = start + round((fixed + tokens / rate) * 1e9)
        calls.append(
            json.dumps(
                {"call": kind, "tokens": tokens, "t_start_ns": start, "t_end_ns": end}
            )
        )
    (steps / compass_run.TOKENIZER_RECORD).write_text("\n".join(calls) + "\n")
    lines, t = [f"{10**12 + 5} 1 app {10**12 + 5} 0"], 10**12  # a health probe
    sizes = np.linspace(*lone, 8).astype(int).tolist() if lone else []
    for size in sizes:
        loop = round((IDLE + PER_BYTE * size) * 1e9)
        t += 10**9
        lines += [f"{t} 1 app {t} 0", f"{t + 2000} 1 body {t + 1000} {size}"]
        lines += [f"{t + 4000} 1 tmpl {t + 3000} {size}"]
        lines += [f"{t + loop // 2 + 50_000} 2 prep {t + loop // 2} 0"]
        lines += [f"{t + loop + 50_000} 1 send {t + loop + 49_000} 0"]
    t += 10**9
    gaps = [round((FIXED + PER_BYTE * size) * 1e9) for size in burst]
    starts = t + 10**6 + np.cumsum([0] + gaps[1:])
    for k, (size, start) in enumerate(zip(burst, starts)):
        lines += [f"{t + 10 * k} 1 app {t + 10 * k} 0"]
        lines += [f"{start - 100} 1 body {t + 10 * k + 5} {size}"]
        lines += [f"{start + 10} 1 tmpl {start} {size}"]
        lines += [
            f"{start + 30} 2 prep {start + 20} 0",
            f"{start + 50} 1 send {start + 40} 0",
        ]
    (steps / compass_run.INTAKE_RECORD).write_text("\n".join(lines) + "\n")


def test_a_record_s_tokenizer_calls_and_intake_stamps_fit_the_entry_s_terms(
    tmp_path, monkeypatch
):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = compass_run.host_key(r.config)
    a = record(tmp_path / "a", 19, key=key)
    stamped(a, burst=(40_000, 60_000, 50_000, 45_000))
    fit(tmp_path, "fit host", a, *clients(a))

    host = member(tmp_path / "store", host_laws_key(key), "params")
    tok, intake = host["tokenizer"], host["http_intake"]
    assert {k: tok[k] for k in TOKENIZER} == TOKENIZER
    for kind, (fixed, rate) in (("encode", ENCODE), ("decode", DECODE)):
        assert tok[kind] == pytest.approx(
            {"fixed_s": fixed, "tokens_per_s": rate}, rel=1e-3
        )
    assert intake == pytest.approx(
        {
            "fixed_s": FIXED,
            "per_byte_s": PER_BYTE,
            "idle_s": IDLE,
            "bytes": [30_000, 400_000],
        },
        rel=1e-6,
    )
    report = member(tmp_path / "store", host_laws_key(key), "report")
    assert report["unfitted"] == {} and report["http_intake"]["burst"]["requests"] == 4
    run_file(tmp_path, monkeypatch, host=host)
    assert compass_run.http_intake() == (
        intake["fixed_s"],
        intake["per_byte_s"],
        intake["idle_s"],
    )
    assert compass_run.http_intake_hull() == (30_000, 400_000)


@pytest.mark.parametrize(
    "lone, burst, unfitted",
    [
        (
            None,
            (),
            {
                "tokenizer": "no record holds steps/tokenizer.jsonl",
                "http_intake": "no record holds steps/intake.stamps",
            },
        ),
        ((30_000, 400_000), (), {"http_intake": "no burst among 8 lone requests"}),
        (
            (),
            (40_000, 60_000, 50_000),
            {"http_intake": "0 forwards for 2 coefficients"},
        ),
    ],
    ids=["no-stamps", "no-burst", "no-lone"],
)
def test_a_term_the_records_cannot_give_is_left_out_and_listed_unfitted(
    tmp_path, lone, burst, unfitted
):
    a = record(tmp_path / "a", 19)
    if lone is not None:
        stamped(a, lone, burst)
    fit(tmp_path, "fit host", a, *clients(a))

    host = member(tmp_path / "store", host_laws_key(HOST), "params")
    report = member(tmp_path / "store", host_laws_key(HOST), "report")
    assert report["unfitted"] == unfitted and not set(unfitted) & set(host)


def test_the_tokenizer_fit_reports_each_record_s_residuals_apart(tmp_path):
    # Record b's encodes run at half a's rate: one fit over both overprices
    # a's calls and underprices b's, and the report shows each record's apart.
    a, b = (tmp_path / d / "steps" for d in "ab")
    for d, enc in ((a, ENCODE), (b, (ENCODE[0], ENCODE[1] / 2))):
        d.mkdir(parents=True)
        stamped(d.parent, lone=(), rates=(enc, DECODE))
    paths = [d / compass_run.TOKENIZER_RECORD for d in (a, b)]
    _, report = fit_tokenizer(paths)

    p50 = {p: report["encode"]["records"][str(p)][1] for p in paths}
    assert p50[paths[0]] > 0.1 and p50[paths[1]] < -0.1
    assert report["decode"]["records"][str(paths[0])] == pytest.approx(
        [0, 0, 0], abs=1e-3
    )
