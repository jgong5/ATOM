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
    fit_intake,
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
    # Records with no tokenizer calls or intake stamps give none of these terms.
    report["unfitted"].update(
        tokenizer=f"no record holds steps/{compass_run.INTAKE_RECORD}",
        http_intake=f"no record holds steps/{compass_run.INTAKE_RECORD}",
        hops=f"no record holds steps/{compass_run.INTAKE_RECORD}",
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


#: The planted tokenizer terms the stamped records below are timed by, each
#: (fixed, tokens/s): a one-item encode's, a several-item encode_batch's and a
#: decode's, here with no positive slope: a nanosecond less per token.
ENCODE, BATCH, DECODE = (5e-6, 9e5), (2e-5, 3e6), (2.8e-5, -1e9)
#: Each lone request's encode call (kind, items, tokens), the request's own.
ENCODES = [
    ("encode", 1, 9000),
    ("encode_batch", 3, 5000),
    ("encode", 1, 88_768),
    ("encode_batch", 7, 90_000),
    ("encode_batch", 1, 120),
    ("encode_batch", 2, 60_000),
    ("encode", 1, 40_000),
    ("encode_batch", 5, 20_000),
]
#: The planted segment tokenizer term: the call's own work over the ids its
#: encode_batch returned, half before that call and half after it.
SEGMENT = (6e-4, 4e6)
#: The two pool threads the lone requests' jobs alternate between.
POOL = (3, 5)
#: The planted intake terms, on the loop thread's CPU clock.
IDLE, PER_BYTE, FIXED = 1.1e-3, 8.8e-9, 4.3e-3
#: What the wall clock holds of an intake past its CPU time: GIL waits.
WAIT = 700_000
#: The loop's CPU time on other requests' work while a pool job runs, in ns.
OTHERS = 2_000_000
TOKENIZER = {"backend": "fast", "fingerprint": "sha256:tok"}


def _seconds(rates, kind, items, tokens) -> float:
    enc, batch, dec = rates
    fixed, rate = dec if kind == "decode" else enc if items == 1 else batch
    return fixed + tokens / rate


def _stamp(kind, start, end, size=0, cpu=(0, 0), thread=1) -> str:
    return f"{end} {thread} {kind} {start} {size} {cpu[1]} {cpu[0]}"


def stamped(
    directory,
    lone=(30_000, 400_000),
    burst=(),
    rates=(ENCODE, BATCH, DECODE),
    encodes=ENCODES,
):
    """`directory`'s steps with the tokenizer calls and HTTP intake stamps a
    recorded server writes. A lone request per body size in `lone`, its loop
    thread's CPU time `IDLE + PER_BYTE x bytes` and its wall that plus
    `WAIT`, besides `OTHERS` of others' work on the loop while its pool job runs,
    its pool job, on one `POOL` thread and then the other, holding its call of
    `encodes`, an ``encode_batch`` inside a segment tokenizer call (one of
    which another thread's decode overlaps), and a decode after its send, each
    timed at `rates`; then calls that do not run lone, ten times slower, and one
    burst of `burst`'s body sizes, its templates `FIXED + PER_BYTE x bytes` of
    the loop's CPU time apart."""
    calls, lines = [], [_stamp("app", 10**12 + 5, 10**12 + 5)]  # a health probe

    def call(kind, items, tokens, start, slow=1.0, thread=3, segment=True) -> int:
        if kind == "encode_batch" and segment:
            # The segment tokenizer's, inside its call on the same thread.
            half = round(slow * (SEGMENT[0] + tokens / SEGMENT[1]) * 5e8)
            end = call(kind, items, tokens, start + half, slow, thread, False) + half
            calls.append(
                {"call": "segment", "tokens": 3 * tokens, "items": 1, "thread": thread}
                | {"t_start_ns": start, "t_end_ns": end}
            )
            return end
        end = start + round(slow * _seconds(rates, kind, items, tokens) * 1e9)
        calls.append(
            {"call": kind, "tokens": tokens, "items": items, "thread": thread}
            | {"t_start_ns": start, "t_end_ns": end}
        )
        return end

    t, cpu = 10**12, 0
    sizes = np.linspace(*lone, 8).astype(int).tolist() if lone else []
    for k, (size, encode) in enumerate(zip(sizes, encodes)):
        t += 10**9
        loop = round((IDLE + PER_BYTE * size) * 1e9) + OTHERS
        # The pool runs the jobs on rotating threads.
        done = call(*encode, t + 20_000, thread=POOL[k % 2])
        if encode == ("encode_batch", 7, 90_000):
            # Another thread's call inside the segment call's own work: not lone.
            call("decode", 1, 2, t + 21_000, thread=4)
        lines += [
            _stamp("app", t, t, cpu=(cpu, cpu)),
            _stamp("recv", t + 1100, t + 1900, 0, (cpu + 1100, cpu + 1900)),
            _stamp("body", t + 1000, t + 2000, size, (cpu + 1000, cpu + 2000)),
            _stamp("tmpl", t + 3000, t + 4000, size, (cpu + 3000, cpu + 4000)),
            _stamp("prep", t + 10_000, done + 1000, thread=2),
            _stamp("wait", t + 9000, done + 1500, 0, (cpu + 5000, cpu + 5000 + OTHERS)),
            _stamp(
                "send", done + 2000, done + 3000 + WAIT, 0, (cpu + loop - 1, cpu + loop)
            ),
        ]
        call("decode", 1, k + 1, done + 10**6)
        cpu += loop + 10**6
    # Request x's encode, while request y arrives; a decode while y is in
    # flight; two decodes at once.
    t += 10**9
    x = call("encode", 1, 9000, t + 20_000, slow=10)
    lines += [_stamp("app", t, t), _stamp("body", t + 1000, t + 2000, 9000)]
    lines += [_stamp("tmpl", t + 3000, t + 4000, 9000)]
    lines += [_stamp("prep", t + 10_000, x + 1000, thread=2)]
    lines += [_stamp("app", t + 30_000, t + 30_000), _stamp("send", x + 2000, x + 3000)]
    lines += [_stamp("body", x + 4000, x + 5000, 9000)]
    lines += [_stamp("tmpl", x + 6000, x + 7000, 9000)]
    y = call("decode", 1, 3, x + 8000, slow=10)
    lines += [_stamp("prep", y + 1000, y + 2000, thread=2)]
    lines += [_stamp("send", y + 3000, y + 4000)]
    call("decode", 1, 4, y + 10**6, slow=10)
    call("decode", 1, 4, y + 10**6 + 500, slow=10)
    # A segment tokenizer call while another pool thread's encode runs.
    call("encode", 1, 9000, y + 5 * 10**6, slow=10, thread=4)
    call("encode_batch", 4, 30_000, y + 6 * 10**6, slow=10)
    t = y + 10**9
    gaps = [round((FIXED + PER_BYTE * size) * 1e9) for size in burst]
    cpus = cpu + np.cumsum([0] + gaps[1:])
    starts = t + 10**6 + np.cumsum([0] + [g + WAIT for g in gaps[1:]])
    for k, (size, start, c) in enumerate(zip(burst, starts, cpus)):
        lines += [_stamp("app", t + 10 * k, t + 10 * k)]
        lines += [_stamp("body", t + 10 * k + 5, start - 100, size)]
        lines += [_stamp("tmpl", start, start + 10, size, (c, c + 10))]
        lines += [_stamp("prep", start + 20, start + 30, thread=2)]
        lines += [_stamp("send", start + 40, start + 50)]
    steps = directory / "steps"
    calls = [TOKENIZER] + calls
    text = "".join(json.dumps(c) + "\n" for c in calls)
    (steps / compass_run.TOKENIZER_RECORD).write_text(text)
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
    assert tok["encode"] == pytest.approx(
        {"fixed_s": ENCODE[0], "tokens_per_s": ENCODE[1]}, rel=1e-3
    )
    assert tok["encode_batch"] == pytest.approx(
        {"fixed_s": BATCH[0], "tokens_per_s": BATCH[1]}, rel=1e-3
    )
    # The decodes take no time per token: a fixed charge over the tokens they span.
    assert tok["decode"]["fixed_s"] == pytest.approx(DECODE[0], rel=1e-3)
    assert tok["decode"]["tokens"] == [1, 8] and "tokens_per_s" not in tok["decode"]
    # On the loop's CPU clock, not the wall's, which holds the GIL waits. The
    # hull spans every body taken in, the two 9000 B requests that overlap too.
    assert intake == pytest.approx(
        {
            "fixed_s": FIXED,
            "per_byte_s": PER_BYTE,
            "idle_s": IDLE,
            "bytes": [9000, 400_000],
        },
        rel=1e-6,
    )
    report = member(tmp_path / "store", host_laws_key(key), "report")
    assert not _unfitted(report) and report["http_intake"]["burst"]["requests"] == 4
    # The segment tokenizer call's own work: its wall less its encode_batch's,
    # over the ids that returned.
    assert tok["segment"] == pytest.approx(
        {"fixed_s": SEGMENT[0], "tokens_per_s": SEGMENT[1]}, rel=1e-3
    )
    # Only the lone calls are fitted: each lone request's encode and decode, and
    # its segment tokenizer call, lone among the calls it did not make, those on
    # its own thread: not the one another thread's decode overlaps.
    fitted = {k: report["tokenizer"][k] for k in compass_run.TOKENIZER_KINDS}
    assert {k: v["samples"] + v["dropped"] for k, v in fitted.items()} == {
        "encode": 4,
        "encode_batch": 4,
        "decode": 8,
        "segment": 4,
    }
    run_file(tmp_path, monkeypatch, host=host)
    assert compass_run.http_intake() == (
        intake["fixed_s"],
        intake["per_byte_s"],
        intake["idle_s"],
    )
    assert compass_run.http_intake_hull() == (9000, 400_000)


def test_the_intake_hull_spans_bodies_past_the_lone_requests_both_ways(tmp_path):
    # Lone requests of 30 to 400 KB; a burst's 600 KB body above them, and
    # two overlapping requests' 9000 B below.
    (tmp_path / "steps").mkdir()
    stamped(tmp_path, burst=(40_000, 600_000, 50_000))
    term, report = fit_intake([tmp_path / "steps" / compass_run.INTAKE_RECORD])
    assert term["bytes"] == [9000, 600_000] and report["samples"] == 8


@pytest.mark.parametrize(
    "lone, burst, encodes, unfitted",
    [
        (
            None,
            (),
            ENCODES,
            {
                "tokenizer": "no record holds steps/intake.stamps",
                "http_intake": "no record holds steps/intake.stamps",
            },
        ),
        (
            (30_000, 400_000),
            (),
            ENCODES,
            {"http_intake": "no burst among 8 lone requests"},
        ),
        (
            (),
            (40_000, 60_000, 50_000),
            ENCODES,
            {"http_intake": "0 forwards for 2 coefficients"}
            | {
                f"tokenizer.{k}": f"the records hold no lone {k} call"
                for k in compass_run.TOKENIZER_KINDS
            },
        ),
        (
            (30_000, 400_000),
            (40_000, 60_000, 50_000),
            [(kind, 1, tokens) for kind, _, tokens in ENCODES],
            {"tokenizer.encode_batch": "the records hold no lone encode_batch call"},
        ),
    ],
    ids=["no-stamps", "no-burst", "no-lone", "no-batch"],
)
def test_a_term_the_records_cannot_give_is_left_out_and_listed_unfitted(
    tmp_path, lone, burst, encodes, unfitted
):
    a = record(tmp_path / "a", 19)
    if lone is not None:
        stamped(a, lone, burst, encodes=encodes)
    fit(tmp_path, "fit host", a, *clients(a))

    host = member(tmp_path / "store", host_laws_key(HOST), "params")
    report = member(tmp_path / "store", host_laws_key(HOST), "report")
    assert _unfitted(report) == unfitted
    left = {t.split(".")[-1] for t in unfitted}
    assert not left & (set(host) | set(host.get("tokenizer", {})))


def _unfitted(report) -> dict:
    """The report's unfitted terms but the hops, which these records, with no
    queue handoffs or client records, cannot give (`test_fit_hops.py`)."""
    return {k: v for k, v in report["unfitted"].items() if k.split(".")[0] != "hops"}


def test_the_tokenizer_fit_reports_each_record_s_residuals_apart(tmp_path):
    # Record b's one-item encodes run at half a's rate: one fit over both
    # overprices a's calls and underprices b's, and the report shows each
    # record's apart. Their decodes take time per token.
    a, b = (tmp_path / d / "steps" for d in "ab")
    sloped = (1e-6, 7.5e6)
    for d, enc in ((a, ENCODE), (b, (ENCODE[0], ENCODE[1] / 2))):
        d.mkdir(parents=True)
        stamped(d.parent, rates=(enc, BATCH, sloped))
    paths = [d / compass_run.TOKENIZER_RECORD for d in (a, b)]
    term, report = fit_tokenizer(paths, [d / compass_run.INTAKE_RECORD for d in (a, b)])

    p50 = {p: report["encode"]["records"][str(p)][1] for p in paths}
    assert p50[paths[0]] > 0.1 and p50[paths[1]] < -0.1
    assert term["decode"] == pytest.approx(
        {"fixed_s": sloped[0], "tokens_per_s": sloped[1]}, rel=1e-3
    )
    assert report["decode"]["records"][str(paths[0])] == pytest.approx(
        [0, 0, 0], abs=1e-3
    )
