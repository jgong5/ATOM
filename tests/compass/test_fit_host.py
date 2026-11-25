# SPDX-License-Identifier: MIT
"""`compass fit host`, run through `cli.main` on `test_fit_steps.py`'s record
directories, and the host laws it publishes handed to `run.runner` in a run
file on `test_run_step_grid.py`'s deployment."""

import json
import re

import numpy as np
import pytest
from test_backend_step_labels import QUARTER, RECV, pp2, pulled
from test_backend_step_linear_fit import HOST as LAWS
from test_backend_step_linear_fit import HOST_ADMIT
from test_fit_samples import edit
from test_fit_steps import (  # noqa: F401
    ENGINE,
    HOST,
    PLATFORM,
    PP2,
    as_record,
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
from atom.compass.artifacts import Key, Kind, keyed_platform
from atom.compass.backends.step_labels import digest
from atom.compass.backends.step_linear_fit import (
    _intakes,
    fit_host,
    fit_intake,
    fit_relay,
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
        fit_src=compass_run.fit_digest(),
        **keyed_platform(Kind.HOST_LAWS, PLATFORM),
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
    platform = keyed_platform(Kind.HOST_LAWS, PLATFORM)
    fitted = {**platform, "fit_src": compass_run.fit_digest()}
    assert host == {**laws, "key": {**laws["key"], **fitted}}
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


def test_a_stated_hop_is_written_into_the_entry_s_hops_and_named_in_its_report(
    tmp_path,
):
    """`--stated-hop`: a hop no record times, such as the router's forward a
    `--dp-aware` deployment's router LP needs above the IPC hop it borrows."""
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = compass_run.host_key(r.config)
    a = record(tmp_path / "a", 19, key=key, trace="trace-a")
    stated = ["--stated-hop", "router_s=0.0015", "--stated-hop", "kv_write_req_s=1e-4"]
    fit(tmp_path, "fit host", a, *clients(a), *stated)

    host = member(tmp_path / "store", host_laws_key(key), "params")
    assert host["hops"] == {"router_s": 0.0015, "kv_write_req_s": 1e-4}
    report = member(tmp_path / "store", host_laws_key(key), "report")
    assert report["stated_hops"] == host["hops"]


def test_a_stated_ingress_is_written_as_the_entry_s_term_and_named_in_its_report(
    tmp_path,
):
    """`--ingress FILE`: the encoding ingress in front of the deployment, as
    `compass fit ingress` wrote it from the ingress's stamps, which no record
    holds; an entry stated with one goes to a run file as its ingress."""
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = compass_run.host_key(r.config)
    a = record(tmp_path / "a", 19, key=key, trace="trace-a")
    term = {"fixed_s": 0.0, "per_byte_s": 4.4e-7, "workers": 64}
    term |= {"router_per_byte_s": 1.2e-8, "bytes": [875, 3620123]}
    (tmp_path / "ingress.json").write_text(json.dumps({"ingress": term}))
    fit(
        tmp_path,
        "fit host",
        a,
        *clients(a),
        "--ingress",
        str(tmp_path / "ingress.json"),
    )

    host = member(tmp_path / "store", host_laws_key(key), "params")
    assert host["ingress"] == term
    report = member(tmp_path / "store", host_laws_key(key), "report")
    assert report["stated_ingress"] == {
        "file": str(tmp_path / "ingress.json"),
        "ingress": term,
    }
    (tmp_path / "bad.json").write_text(json.dumps({"ingress": {**term, "workers": 0}}))
    with pytest.raises(SystemExit, match="bad.json: its ingress is"):
        fit(
            tmp_path,
            "fit host",
            a,
            *clients(a),
            "--ingress",
            str(tmp_path / "bad.json"),
        )


@pytest.mark.parametrize(
    "words, refusal",
    [
        (["ipc_s=0.0001"], "only router_s, kv_write_req_s"),
        (["router_s=0.001", "router_s=0.002"], "state router_s already"),
        (["router_s=-1"], "not a positive number of seconds"),
        (["router_s=fast"], "not a positive number of seconds"),
    ],
    ids=["fitted-hop", "twice", "negative", "no-number"],
)
def test_a_stated_hop_refuses_a_fitted_hop_a_repeat_or_no_positive_seconds(
    words, refusal
):
    from atom.compass.fit.host import stated_hops

    with pytest.raises(ValueError, match=refusal):
        stated_hops(words, {"ipc_s": 1e-4})


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


def test_a_record_whose_server_did_not_exit_refuses_naming_its_lines(tmp_path):
    a = record(tmp_path / "a", 19)
    stamped(a)
    stamps = a / "steps" / compass_run.INTAKE_RECORD
    local = tmp_path / "compass.intake.stamps.x"
    stamps.rename(local)
    stamps.with_name(stamps.name + compass_run.LOCAL).write_text(f"{local}\n")

    named = (
        f"{a}: its server did not exit, so steps/intake.stamps does not hold its "
        f"lines; steps/intake.stamps.local names the file that holds them, {local}"
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
    fixed=FIXED,
):
    """`directory`'s steps with the tokenizer calls and HTTP intake stamps a
    recorded server writes. A lone request per body size in `lone`, its loop
    thread's CPU time `IDLE + PER_BYTE x bytes` and its wall that plus
    `WAIT`, besides `OTHERS` of others' work on the loop while its pool job runs,
    its pool job, on one `POOL` thread and then the other, holding its call of
    `encodes`, an ``encode_batch`` inside a segment tokenizer call (one of
    which another thread's decode overlaps), and a decode after its send, each
    timed at `rates`; then calls that do not run lone, ten times slower, and one
    burst of `burst`'s body sizes, its templates `fixed + PER_BYTE x bytes` of
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
    gaps = [round((fixed + PER_BYTE * size) * 1e9) for size in burst]
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


def test_requests_lone_after_two_that_overlap_fit_with_their_pool_rendered_template(
    tmp_path,
):
    # Two requests whose app calls both come before either body (one app call
    # then names both), a metrics scrape's app call with no body, then lone
    # requests whose template a pool thread renders: each lone request's
    # seconds are its loop CPU time plus that render's wall.
    t, cpu, lines, expect = 10**12, 0, [], []
    lines += [_stamp("app", t, t), _stamp("app", t + 10, t + 10)]
    for k in range(2):
        s = t + 1000 * (k + 1)
        lines += [_stamp("body", s, s + 100, 5000)]
        lines += [_stamp("tmpl", s + 200, s + 300, 5000)]
        lines += [_stamp("prep", s + 400, s + 500, thread=2)]
        lines += [_stamp("send", s + 600, s + 700)]
    for k, size in enumerate(np.linspace(30_000, 400_000, 8).astype(int).tolist()):
        t += 10**9
        loop, render = round((IDLE + PER_BYTE * size) * 1e9), 300_000 + 2 * size
        w, p, e = t + 9000 + render, t + 10_000 + render, t + 60_000 + render
        lines += [_stamp("app", t - 5000, t - 5000)]  # a scrape
        lines += [
            _stamp("app", t, t, cpu=(cpu, cpu)),
            _stamp("body", t + 1000, t + 2000, size, (cpu + 1000, cpu + 2000)),
            _stamp("tmpl", t + 3000, t + 3000 + render, size, thread=POOL[0]),
            _stamp("wait", w, w + 41_000, 0, (cpu + 5000,) * 2),
            _stamp("prep", p, p + 30_000, thread=POOL[1]),
            _stamp("send", e, e + 10_000, 0, (cpu, cpu + loop)),
        ]
        expect.append((size, (loop + render) * 1e-9))
        cpu += loop + 10**6
    path = tmp_path / compass_run.INTAKE_RECORD
    path.write_text("\n".join(lines) + "\n")
    lone, bursts, sizes = _intakes(path)
    assert lone == pytest.approx(expect) and not bursts and len(sizes) == 10


def test_a_server_that_renders_no_template_fits_its_intake_on_its_pool_jobs(
    tmp_path,
):
    # A prefill-decode deployment's decode side renders no chat template: each
    # request reads its body, submits its pool job and is sent. The same requests
    # stamped so fit the same term, each request of a burst marked by its pool
    # job's submit, stamped where its template began. Each pool job encodes, as
    # its record's tokenizer record says, so no job's wall joins its intake.
    rendered, bare = tmp_path / "rendered", tmp_path / "bare"
    for d in (rendered, bare):
        (d / "steps").mkdir(parents=True)
        stamped(d, burst=(40_000, 60_000, 50_000, 45_000))
    stamps = bare / "steps" / compass_run.INTAKE_RECORD
    rows = [line.split() for line in stamps.read_text().splitlines()]
    kept = [row for row in rows if row[2] != "tmpl"]
    for k, row in enumerate(rows):
        later = [r[2] for r in rows[k + 1 :] if r[2] in ("app", "wait")]
        if row[2] == "tmpl" and later[:1] != ["wait"]:
            # `stamped` gives only the lone requests a pool job's wait.
            kept.append([row[3], row[1], "wait", row[3], "0", row[6], row[6]])
    stamps.write_text("".join(" ".join(row) + "\n" for row in kept))
    record = bare / "steps" / compass_run.TOKENIZER_RECORD
    jobs = [
        {"call": "encode", "tokens": 1, "items": 1, "thread": int(row[1])}
        | {"t_start_ns": int(row[3]), "t_end_ns": int(row[0])}
        for row in kept
        if row[2] == "prep"
    ]
    record.write_text(record.read_text() + "".join(json.dumps(j) + "\n" for j in jobs))

    got = [
        fit_intake(
            [d / "steps" / compass_run.INTAKE_RECORD],
            [d / "steps" / compass_run.TOKENIZER_RECORD],
        )
        for d in (rendered, bare)
    ]
    assert got[1] == got[0]
    term, report = got[1]
    assert term == pytest.approx(
        {
            "fixed_s": FIXED,
            "per_byte_s": PER_BYTE,
            "idle_s": IDLE,
            "bytes": [9000, 400_000],
        },
        rel=1e-6,
    )
    assert report["samples"] == 8 and report["burst"]["requests"] == 4


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
    ids=["no-stamps", "no-lone", "no-batch"],
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


@pytest.mark.parametrize(
    "burst, fixed, why",
    [
        ((), FIXED, "no burst among 8 lone requests"),
        (
            (900_000, 1_000_000, 1_100_000),
            -3e-3,
            (
                "the bursts' 2 later requests take 0.00624 s of the loop's "
                "CPU each, less than per_byte_s 8.8e-09 times their mean body of "
                "1050000 B, so fixed_s would be -0.003 s"
            ),
        ),
    ],
    ids=["no-burst", "burst-closer-than-per-byte-s"],
)
def test_with_no_burst_or_a_negative_one_fixed_s_is_idle_s_and_a_run_charges_it(
    tmp_path, monkeypatch, burst, fixed, why
):
    # No burst, as a c=1 client's records hold, or a burst of megabyte bodies
    # whose requests run closer together than the lone requests' per_byte_s
    # prices their bodies, so fixed_s from it would be negative, which a run
    # refuses at start: fixed_s is the lone requests' idle_s instead.
    a = record(tmp_path / "a", 19)
    stamped(a, burst=burst, fixed=fixed)
    fit(tmp_path, "fit host", a, *clients(a))

    host = member(tmp_path / "store", host_laws_key(HOST), "params")
    report = member(tmp_path / "store", host_laws_key(HOST), "report")
    assert host["http_intake"] == pytest.approx(
        {
            "fixed_s": IDLE,
            "per_byte_s": PER_BYTE,
            "idle_s": IDLE,
            "bytes": [9000, max((400_000, *burst))],
        },
        rel=1e-6,
    )
    assert not _unfitted(report) and "burst" not in report["http_intake"]
    assert report["http_intake"]["fixed_s_is_idle_s"] == why
    # A run file carrying the entry passes the run's check of its host entry,
    # and the frontend charges a queued request's intake as a lone one's.
    run_file(tmp_path, monkeypatch, host=host)
    assert compass_run.http_intake() == pytest.approx((IDLE, PER_BYTE, IDLE), rel=1e-6)


def test_fixed_s_is_the_mean_over_every_burst_of_every_record(tmp_path):
    # Record a's burst of 3 runs its later requests 2 ms past per_byte_s apart,
    # b's longer burst of 5 runs them 1 ms apart: fixed_s weighs each later
    # request once, not the longest burst alone.
    a, b = record(tmp_path / "a", 19), record(tmp_path / "b", 20)
    stamped(a, burst=(40_000, 60_000, 50_000), fixed=2e-3)
    stamped(b, burst=(45_000, 55_000, 65_000, 35_000, 50_000), fixed=1e-3)
    fit(tmp_path, "fit host", a, b, *clients(a, b))

    host = member(tmp_path / "store", host_laws_key(HOST), "params")
    report = member(tmp_path / "store", host_laws_key(HOST), "report")
    values = [2e-3] * 2 + [1e-3] * 4
    assert host["http_intake"]["fixed_s"] == pytest.approx(np.mean(values), rel=1e-6)
    assert report["http_intake"]["burst"] == pytest.approx(
        {
            "bursts": 2,
            "requests": 8,
            "p10_s": np.percentile(values, 10),
            "p90_s": np.percentile(values, 90),
        },
        rel=1e-6,
    )


def _unfitted(report) -> dict:
    """The report's unfitted terms but the hops, which these records, with no
    queue handoffs or client records, cannot give (`test_fit_hops.py`), and the
    segment term's search rate, which `stamped` records without
    ``compared_chars`` cannot give."""
    return {
        k: v
        for k, v in report["unfitted"].items()
        if k.split(".")[0] != "hops" and k != "tokenizer.segment.chars_per_s"
    }


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


def test_the_segment_term_takes_what_its_stored_prefix_search_compared(tmp_path):
    # Each lone segment tokenizer call's own work: per call, per id its
    # encode_batch returned, and per character its stored-prefix search compared.
    fixed, per_id, per_char = 6e-4, 1 / 4e6, 1 / 2.5e9
    work = [(0, 0), (500, 4_000_000), (9000, 1_000_000), (120, 2_000_000)]
    work += [(30_000, 0), (2000, 3_000_000), (64, 500_000)]
    lines, t = [], 10**12
    for ids, chars in work:
        own = round((fixed + ids * per_id + chars * per_char) * 1e9)
        batch = 1000 + 100 * ids if ids else 0
        lines.append(
            {"call": "segment", "tokens": 3 * ids + 10, "items": 1}
            | {"compared_chars": chars, "thread": 3}
            | {"t_start_ns": t, "t_end_ns": t + own + batch}
        )
        if ids:  # its encode_batch, inside it on its thread
            start = t + own // 2
            lines.append(
                {"call": "encode_batch", "tokens": ids, "items": 2, "thread": 3}
                | {"t_start_ns": start, "t_end_ns": start + batch}
            )
        t += 10**9
    path, intake = tmp_path / "tokenizer.jsonl", tmp_path / "intake.stamps"
    intake.write_text("")

    def segment(lines):
        path.write_text("".join(json.dumps(c) + "\n" for c in [TOKENIZER, *lines]))
        term, report = fit_tokenizer([path], [intake])
        return term["segment"], report["unfitted"].get("segment.chars_per_s")

    assert segment(lines) == (
        pytest.approx(
            {"fixed_s": fixed, "tokens_per_s": 1 / per_id, "chars_per_s": 1 / per_char},
            rel=1e-3,
        ),
        None,
    )

    # A call whose record lacks the characters keeps the term on ids alone,
    # and the search rate is listed unfitted, naming how many lack them.
    def without(lines, k):
        cut = [c for c in lines if c["call"] == "segment"][:k]
        return [
            {n: v for n, v in c.items() if n != "compared_chars" or c not in cut}
            for c in lines
        ]

    for k in (3, len(work)):
        term, why = segment(without(lines, k))
        assert set(term) == {"fixed_s", "tokens_per_s"}
        assert (
            why
            == f"{k} of {len(work)} lone segment calls' records hold no compared_chars"
        )
    # Searches that compared nothing give the rate no time either.
    zero = [c | {"compared_chars": 0} if c["call"] == "segment" else c for c in lines]
    term, why = segment(zero)
    assert set(term) == {"fixed_s", "tokens_per_s"}
    assert why == "the records give no time per compared character"


def test_a_request_given_its_token_ids_takes_its_pool_job_in_with_its_intake(tmp_path):
    # A prefill-decode decode side's requests: no template, and a pool job
    # that builds the sequence and calls no tokenizer, which no tokenizer law
    # prices, so its wall joins the loop's CPU time. One whose job encodes
    # (a text prompt) leaves its job to the tokenizer laws.
    t, cpu, lines, expect, calls = 10**12, 0, [], [], [TOKENIZER]
    for k, size in enumerate(np.linspace(30_000, 900_000, 6).astype(int).tolist()):
        t += 10**9
        loop, job = round((IDLE + PER_BYTE * size) * 1e9), 100_000 + 3 * size
        p, e = t + 10_000, t + 20_000 + job
        lines += [
            _stamp("app", t, t, cpu=(cpu, cpu)),
            _stamp("body", t + 1000, t + 2000, size, (cpu + 1000, cpu + 2000)),
            _stamp("wait", p - 1000, p + job + 1000, 0, (cpu + 5000,) * 2),
            _stamp("prep", p, p + job, thread=POOL[k % 2]),
            _stamp("send", e, e + 10_000, 0, (cpu, cpu + loop)),
        ]
        encodes = k == 5
        if encodes:
            calls.append(
                {"call": "encode", "tokens": 9, "items": 1, "thread": POOL[k % 2]}
                | {"t_start_ns": p + 10, "t_end_ns": p + 20}
            )
        expect.append((size, (loop + (0 if encodes else job)) * 1e-9))
        cpu += loop + 10**6
    path = tmp_path / compass_run.INTAKE_RECORD
    path.write_text("\n".join(lines) + "\n")
    record = tmp_path / compass_run.TOKENIZER_RECORD
    record.write_text("".join(json.dumps(c) + "\n" for c in calls))
    lone, bursts, sizes = _intakes(path, record)
    assert lone == pytest.approx(expect) and not bursts and len(sizes) == 6
    # With no tokenizer record, no job is known to have called one.
    assert _intakes(path)[0][-1] == pytest.approx(
        (sizes[-1], expect[-1][1] + 3e-9 * sizes[-1] + 1e-4)
    )


def _relayed(directory, sizes, fixed, per_byte, rows=3, others=()):
    """A prefill-decode deployment's two sides' intake stamps: each request's
    `rows` output rows on prefill, one per 100 us, then decode's app call for
    it `fixed + per_byte x bytes` after the last row's return, and its body;
    `others`, app calls on decode with no output on prefill before them."""
    pre, dec = [], [_stamp("app", t, t) for t in others]
    t = 10**12
    for k, size in enumerate(sizes):
        t += 10**9
        for r in range(rows):
            pre.append(_stamp("out", t + r * 100_000, t + r * 100_000 + 1000, thread=7))
        back = t + (rows - 1) * 100_000 + 1000
        app = back + round((fixed + per_byte * size) * 1e9)
        dec += [_stamp("app", app, app), _stamp("body", app + 100, app + 500, size)]
        dec += [_stamp("send", app + 900, app + 1000)]
    for side, lines in (("prefill", pre), ("decode", dec)):
        (directory / side / "steps").mkdir(parents=True)
        path = directory / side / "steps" / compass_run.INTAKE_RECORD
        path.write_text("".join(line + "\n" for line in lines))
    return [tuple(directory / s / "steps" / compass_run.INTAKE_RECORD for s in SIDES)]


SIDES = ("prefill", "decode")


def test_the_relay_term_is_fitted_from_prefill_s_last_output_row_to_decode_s_app_call(
    tmp_path,
):
    sizes = np.linspace(20_000, 3_000_000, 12).astype(int).tolist()
    pairs = _relayed(tmp_path, sizes, 1.5e-3, 11e-9, others=(10**11,))
    term, report = fit_relay(pairs)
    assert term["fixed_s"] == pytest.approx(1.5e-3, rel=1e-3)
    assert term["per_byte_s"] == pytest.approx(11e-9, rel=1e-3)
    assert term["bytes"] == [sizes[0], sizes[-1]] and report["samples"] == 12


def test_a_relay_with_no_request_alone_on_it_is_left_unfitted(tmp_path):
    with pytest.raises(ValueError, match="0 forwards for 2 coefficients"):
        fit_relay(_relayed(tmp_path, [], 1.5e-3, 11e-9))


def _pipeline(directory):
    """`directory`, a record of `pp2`'s two stages, its frontend's tokenizer
    calls and intake stamped (`stamped`)."""
    key = {**HOST, "engine": PP2}
    (directory / "steps").mkdir(parents=True)
    pp2(directory / "steps", 19, key)
    stamped(as_record(directory, key))
    return directory


def test_a_pipeline_s_entry_holds_each_stage_s_laws_and_its_frontend_s_terms_once(
    tmp_path,
):
    """Each stage's laws are its own, each `E` from its release; stage 1's
    hold `W_recv`, and the head's admissions' `admit`."""
    a = _pipeline(tmp_path / "a")
    fit(tmp_path, "fit host", a, *clients(a))

    key = host_laws_key({**HOST, "engine": PP2})
    host = member(tmp_path / "store", key, "params")
    report = member(tmp_path / "store", key, "report")
    assert {"tokenizer", "http_intake"} <= set(host) and "laws" not in host
    assert [set(part) for part in host["stages"]] == [{"laws", "admit"}] * 2
    assert len(report["stages"]) == 2 and "tokenizer" in report
    first, last = host["stages"]
    assert first["admit"] == pytest.approx(HOST_ADMIT, rel=1e-4, abs=1e-12)
    for kind in ("prefill", "middle"):
        assert "W_recv" not in first["laws"][kind]
        recv = last["laws"][kind].pop("W_recv")
        assert recv == pytest.approx(RECV, rel=1e-4, abs=1e-12)
        for part, scale in zip(host["stages"], (1, QUARTER)):
            assert set(part["laws"][kind]) == set(LAWS[kind])
            for segment, c in LAWS[kind].items():
                got = part["laws"][kind][segment]
                assert got == pytest.approx([scale * v for v in c], rel=1e-4, abs=1e-12)


def test_a_pipeline_whose_stages_hold_two_host_keys_refuses(tmp_path):
    a = _pipeline(tmp_path / "a")
    edit(a / "steps/pp1.dp0.key.json", lambda k: k.update(host_cpu="another cpu"))

    with pytest.raises(SystemExit, match="the pipeline's stages carry the host keys"):
        fit(tmp_path, "fit host", a, *clients(a))
