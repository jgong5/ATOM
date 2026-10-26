# SPDX-License-Identifier: MIT
"""One labelling of a real run's recorded steps, read by the fit, the cost check
and the per-step table alike.

Records are written two ways: by `StepRecord` on `test_parity_device_window.py`'s
planted device, through a real runner's stamps, and as
`test_backend_step_linear_fit.py` plants them, from a known law.
"""

import dataclasses
import itertools
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
from test_backend_step_linear_fit import (
    GAP_NS,
    HOST,
    SPEC,
    _record,
    _rowmod_steps,
    _steps,
    _timed_record,
    _view,
)
from test_memory_readings import DOCUMENT, META_8B
from test_parity_device_window import BUSY, GAP, _recorded
from test_parity_record import _config, _drive

from atom.compass.artifacts import machine_spec_digest
from atom.compass.backends import (
    BatchView,
    RequestShape,
    StepLinearBackend,
    StepLinearLaw,
)
from atom.compass.backends.step_labels import (
    TABLE_VERSION,
    digest,
    label,
    label_run,
    write_table,
)
from atom.compass.backends.step_linear_fit import (
    fit,
    fit_host,
    label_records,
    main,
    mark_held_out,
)
from atom.compass.parity import ENV, StepRecord
from atom.compass.runner import projection
from atom.compass.spec import MachineSpec

KEY = {"host_cpu": "planted cpu", "atom": "0" * 64, "model": None, "engine": {}}


def _stamped(tmp_path, monkeypatch):
    """A record `StepRecord` wrote on the planted device, under a host key."""
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0, 1]})[0][0]
    recorded, device = _recorded(batches[-1])
    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = recorded(0, _config())
    runner.host_key = KEY
    for batch in batches:
        runner.forward(batch)
    device.tick = BUSY // 3
    runner.exit()
    return tmp_path / "real" / "dp0.jsonl"


def test_a_stamped_record_is_labelled_by_step_kind(tmp_path, monkeypatch):
    record = _stamped(tmp_path, monkeypatch)
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    samples, table = label(record)

    assert [s.step for s in samples] == [x["step"] for x in lines]
    kinds = {}
    for n, (s, line, row) in enumerate(zip(samples, lines, table)):
        rows = [[r.query_tokens, r.context_tokens, r.decode] for r in s.view.requests]
        assert rows == line["rows"]
        assert s.view.capture_rung == line["rung"]
        assert s.view.produces_output == line["produces_output"]
        assert s.view.moe_rows is None and s.cut is None
        busy = BUSY + 2 * n + 1
        decode = bool(s.view.decode)
        kinds.setdefault((decode, line["produces_output"]), []).append(n)
        # A decode takes the device gap before it; no prefill chunk is ever cut.
        assert s.device_s == pytest.approx((busy + GAP * decode) * 1e-9, rel=1e-12)
        assert ("W_post" in s.host) == line["produces_output"]
        assert s.provenance == {
            "source": "record",
            "run": digest(record),
            "requests": [r[0] for r in line["batch"]],
            "config": KEY,
        }
        assert row["label_s"] == s.device_s and row["idle"] == s.idle
        assert {k: row[k] for k in s.host} == s.host
    # Middle chunks, output prefills and decodes all ran; the planted device
    # starts every prefill before its host launch ends, and none is cut.
    assert set(kinds) == {(False, False), (False, True), (True, True)}
    exposed = [row["step"] for row in table if row["launch_exposed"]]
    assert exposed and all(not samples[n].view.decode for n in exposed)
    # The second request's first chunk follows the first's last: no request in
    # common, so its `E` is cut as idle, as the record's first forward has none.
    assert [s.step for s in samples if s.idle] == [4]
    assert [s.step for s in samples if "E" not in s.host] == [0, 4]


def test_a_rank_s_moe_rows_ride_in_each_sample_s_view(tmp_path):
    from test_step_linear_dp import _groups, _records

    groups = _groups(1, n=24)
    for rank, record in enumerate(_records(tmp_path, groups)):
        samples = label(record)[0]
        assert [s.view.moe_rows for s in samples] == [
            views[rank].moe_rows for views, _ in groups
        ]
        assert {s.rank for s in samples} == {rank}


def test_the_command_writes_each_record_s_table_with_the_labels_it_fitted(tmp_path):
    # A decode after idle finds the device waiting on its launch: it is cut.
    records = [
        _timed_record(tmp_path / f"dp0-{s}.jsonl", _steps(s), {60}, gap=GAP_NS)[0]
        for s in (19, 20)
    ]
    main([*map(str, records), "--out", str(tmp_path / "law.json"), *SPEC])
    law = StepLinearBackend(
        StepLinearLaw(json.loads((tmp_path / "law.json").read_text()))
    )

    cuts = set()
    for record in records:
        header, *table = map(
            json.loads, (tmp_path / f"{record.stem}.cost.jsonl").open()
        )
        assert header == {"version": TABLE_VERSION}
        fitted = {s.step: s for s in label(record)[0]}
        assert [row["step"] for row in table] == list(fitted)
        for row in table:
            s = fitted[row["step"]]
            assert row["label_s"] == s.device_s and row["cut"] == s.cut
            if row["cut"] is None and s.view.decode:
                gap = pytest.approx(row["d_busy_s"] + row["d_gap_s"], rel=1e-12)
                assert row["label_s"] == gap
                # The law was fitted on these labels: it prices each one back.
                price = law.price(s.view).seconds
                assert price == pytest.approx(row["label_s"], rel=1e-5)
            elif row["cut"] is None:
                assert row["label_s"] == row["d_busy_s"]
        cuts |= {row["cut"] for row in table}
    assert cuts == {None, "exposed"}


def test_the_report_states_inputs_vectors_and_held_out_errors_per_run(tmp_path):
    records = [_record(tmp_path / f"dp0-{s}.jsonl", _steps(s)) for s in (6, 7)]
    samples = label_records(records)[0]
    law, report = fit(samples)

    # Labelling writes nothing; the command writes the tables.
    assert not list(tmp_path.glob("*.cost.jsonl"))
    runs = [digest(r) for r in records]
    assert report["provenance"] == [
        {"source": "record", "run": run, "config": None} for run in runs
    ]
    vectors = {"prefill": law["prefill"], **law["decode"]}
    assert report["vectors"] == {
        k: {**v["fit"], "hull": v["hull"]} for k, v in vectors.items()
    }
    assert report["held_out"] == {}

    mark_held_out(law, report, samples)
    assert list(report["held_out"]) == runs
    for key, vector in vectors.items():
        for band, b in vector["held_out"]["bands"].items():
            per_run = [report["held_out"][run].get(band, {}) for run in runs]
            assert sum(r.get("steps", 0) for r in per_run) == b["steps"], key


def test_a_seed_fixes_the_shuffles_and_the_same_inputs_give_the_same_bytes(tmp_path):
    records = [
        _record(tmp_path / f"dp0-{s}.jsonl", _rowmod_steps(s, 150)) for s in (20, 21)
    ]
    samples = label_records(records)[0]
    once, again = fit(samples), fit(label_records(records)[0])

    assert json.dumps(once) == json.dumps(again)
    reseeded = fit(samples, seed=1)[0]["decode"]["8"]["fit"]["rowmod"]
    evidence = once[0]["decode"]["8"]["fit"]["rowmod"]
    assert reseeded["shuffled_mad"] != evidence["shuffled_mad"]
    assert reseeded["mad_without"] == evidence["mad_without"]


def test_record_readers_take_rank_records_and_leave_their_tables(tmp_path):
    from atom.compass.parity import read
    from atom.compass.validate.runs import _steps as graded_steps

    (tmp_path / "steps").mkdir()
    record = _record(tmp_path / "steps" / "dp0.jsonl", _steps(6, n=40))
    write_table(record, label(record)[1])

    assert (tmp_path / "steps" / "dp0.cost.jsonl").exists()
    assert list(read(tmp_path / "steps")) == [0]
    assert len(graded_steps(tmp_path)) == 40


def test_the_command_s_laws_load_in_a_run_file(tmp_path, monkeypatch):
    from test_run_step_grid import deployment, run_file

    from atom.compass import run as compass_run

    r = deployment(tmp_path, enable_dp_attention=False, enable_expert_parallel=False)
    key = compass_run.host_key(r.config)
    records = [
        str(_timed_record(tmp_path / f"dp0-{s}.jsonl", _steps(s), key=key)[0])
        for s in (19, 20)
    ]
    main(
        [
            *records,
            "--out",
            str(tmp_path / "law.json"),
            "--machine-spec",
            machine_spec_digest(MachineSpec.from_mapping(DOCUMENT)),
        ]
    )
    law = json.loads((tmp_path / "law.json").read_text())

    run = run_file(tmp_path, monkeypatch, step_linear=law, host=law["host"])
    del run["step_grid"]
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    assert isinstance(r.compass_backend, StepLinearBackend)
    assert r.compass_host.laws == law["host"]["laws"]
    rows = [(1, 900, True), (1, 400, True)]
    assert r.compass_backend.price(_view(rows, 2)).seconds > 0


def test_host_laws_fitted_under_a_stated_engine_load_at_that_engine(
    tmp_path, monkeypatch
):
    from test_run_step_grid import deployment, run_file

    from atom.compass import run as compass_run

    r = deployment(tmp_path, enable_dp_attention=False, enable_expert_parallel=False)
    key = compass_run.host_key(r.config)
    later = ("data_parallel_size", "enable_dp_attention", "enable_expert_parallel")
    later += ("enable_tbo", "enable_tbo_decode", "env")
    # Records whose host key predates those engine fields, as a capture's do.
    old = key | {"engine": {k: v for k, v in key["engine"].items() if k not in later}}
    records = [
        str(_timed_record(tmp_path / f"dp0-{s}.jsonl", _steps(s), key=old)[0])
        for s in (19, 20)
    ]
    stated = tmp_path / "stated.engine.json"
    stated.write_text(json.dumps(key["engine"]))
    spec = machine_spec_digest(MachineSpec.from_mapping(DOCUMENT))
    out = ["--out", str(tmp_path / "law.json"), "--machine-spec", spec]
    main([*records, *out, "--engine", str(stated)])
    law = json.loads((tmp_path / "law.json").read_text())
    assert law["host"]["key"]["engine"] == law["engine_config"] == key["engine"]

    run = run_file(tmp_path, monkeypatch, step_linear=law, host=law["host"])
    del run["step_grid"]
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    compass_run._PublishedClock(
        compass_run.ENGINE, compass_run.channel_table(run), None
    )
    compass_run.runner(r, META_8B)
    assert r.compass_host.laws == law["host"]["laws"]


def pulled(record, parts=(3,)):
    """Write `record`'s pull stamps beside it, a put and then a pull at
    1 / `parts[n % len(parts)]` of the way from line n's exit to the next
    line's entry; returns the pulls."""
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    ends = [s["t_enter_ns"] for s in lines[1:]] + [lines[-1]["t_exit_ns"] + 9]
    pulls = [
        s["t_exit_ns"] + (e - s["t_exit_ns"]) // parts[n % len(parts)]
        for n, (s, e) in enumerate(zip(lines, ends))
    ]
    stamps = [x for p in pulls for x in ({"t_put_ns": p - 1}, {"t_pull_ns": p})]
    path = record.with_suffix(".pulls.jsonl")
    path.write_text("".join(json.dumps(x) + "\n" for x in stamps))
    return pulls


def test_a_record_with_pull_stamps_labels_E_out_and_E_from_the_pull(tmp_path):
    record = _timed_record(tmp_path / "dp0.jsonl", _steps(19), {30})[0]
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    unstamped = label(record)[0]
    assert not any("E_out" in s.host for s in unstamped)

    pulls = pulled(record)
    samples, table = label(record)
    for n, (s, u, row) in enumerate(zip(samples, unstamped, table)):
        assert (
            s.host["E_out"] == row["E_out"] == (pulls[n] - lines[n]["t_exit_ns"]) * 1e-9
        )
        assert ("E" in s.host) == ("E" in u.host)
        if "E" in s.host:
            assert s.host["E"] == (lines[n]["t_enter_ns"] - pulls[n - 1]) * 1e-9
            assert (
                u.host["E"]
                == (lines[n]["t_enter_ns"] - lines[n - 1]["t_exit_ns"]) * 1e-9
            )
        others = {k: v for k, v in s.host.items() if k not in ("E", "E_out")}
        assert others == {k: v for k, v in u.host.items() if k != "E"}


def _pair(steps, parts):
    """Two DP ranks' records of the same forwards in `steps`, rank r pulling as
    `pulled(record, parts[r])` does; returns the records and their pulls."""
    from test_drive_record import ENGINE

    steps.mkdir(exist_ok=True)
    records, pulls = [], []
    for rank in (0, 1):
        path = _timed_record(steps / f"dp{rank}.jsonl", _steps(19))[0]
        lines = [json.loads(x) | {"dp_rank": rank} for x in path.open()]
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        path.with_suffix(".engine.json").write_text(json.dumps(ENGINE))
        pulls.append(pulled(path, parts[rank]))
        records.append(path)
    return records, pulls


def test_a_dp_pair_s_cost_lines_start_E_at_the_group_s_latest_pull(tmp_path):
    from atom.compass.drive.record import write_engine_and_costs

    # Rank 1 pulls after rank 0 after even lines and before it after odd ones.
    records, pulls = _pair(tmp_path / "steps", ((3,), (2, 6)))
    (tmp_path / "run.json").write_text("{}")
    write_engine_and_costs(tmp_path)

    lines = [json.loads(x) for x in records[0].open()]
    latest = set()
    for rank, record in enumerate(records):
        _, *table = map(json.loads, record.with_suffix(".cost.jsonl").open())
        assert [row["step"] for row in table] == list(range(len(lines)))
        for n, row in enumerate(table):
            own = (pulls[rank][n] - lines[n]["t_exit_ns"]) * 1e-9
            assert row["E_out"] == pytest.approx(own, rel=1e-12)
            if n:
                start = max(p[n - 1] for p in pulls)
                latest.add(pulls[1][n - 1] == start)
                gap = (lines[n]["t_enter_ns"] - start) * 1e-9
                assert row["E"] == pytest.approx(gap, rel=1e-12)
    assert latest == {True, False}


def test_a_dummy_only_rank_s_later_pull_starts_its_peer_s_E(tmp_path):
    records, pulls = _pair(tmp_path, ((3,), (2,)))
    lines = [json.loads(x) | {"is_dummy_run": True} for x in records[1].open()]
    records[1].write_text("".join(json.dumps(x) + "\n" for x in lines))

    (samples, _), dummy = label_run(records)
    assert dummy == ([], [])
    timed = [s for s in samples if "E" in s.host]
    assert timed
    for s in timed:
        start = pulls[1][s.step - 1]
        assert start > pulls[0][s.step - 1]
        assert s.host["E"] == (lines[s.step]["t_enter_ns"] - start) * 1e-9


def test_a_dp_pair_fits_the_E_law_of_its_later_pulling_rank(tmp_path):
    records = _pair(tmp_path, ((3,), (2,)))[0]

    pair = fit_host(label_records(records, 2)[0])[0]["laws"]
    alone = [fit_host(label_records([r])[0])[0]["laws"] for r in records]
    assert set(pair) == {"decode", "prefill", "middle"}
    for kind, laws in pair.items():
        assert laws["E"] == pytest.approx(alone[1][kind]["E"], rel=1e-9)
        assert laws["E"] != pytest.approx(alone[0][kind]["E"], rel=1e-3)
        assert laws["E_out"] != pytest.approx(alone[1][kind]["E_out"], rel=1e-3)


@pytest.mark.parametrize("short", [1, 2])
def test_ranks_one_forward_apart_are_labelled_and_further_apart_refuse(tmp_path, short):
    record = _timed_record(tmp_path / "dp0.jsonl", _steps(19))[0]
    lines = record.read_text().splitlines(keepends=True)
    (tmp_path / "dp1.jsonl").write_text("".join(lines[:-short]))
    records = [record, tmp_path / "dp1.jsonl"]

    if short == 1:
        assert len(label_records(records, 2)[0]) == 2 * len(lines) - 1
        return
    counts = {"dp0.jsonl": len(lines), "dp1.jsonl": len(lines) - 2}
    refusal = f"hold {counts} lines, so their forwards do not pair up"
    with pytest.raises(ValueError, match=re.escape(refusal)):
        label_records(records, 2)


def _dp_view(rows, rank, **group):
    rows = tuple(RequestShape(q, c, d) for q, c, d in rows)
    return BatchView(rows, produces_output=True, dp_rank=rank, **group)


#: Each rank's views of a DP2 group's steps: a decode graph at rung 4 whose
#: gather pads the group's 4 real rows to 8, then a prefill step split in two.
GROUP = [
    [
        _dp_view(decode, rank, capture_rung=4, moe_rows=8, moe_tokens=4),
        _dp_view(
            prefill,
            rank,
            moe_rows=900,
            moe_tokens=900,
            tbo_tokens=((300, 200), (200, 200)),
        ),
    ]
    for rank, decode, prefill in (
        (0, [(1, 9, True)] * 3, [(300, 300, False), (200, 200, False)]),
        (1, [(1, 7, True)], [(400, 400, False)]),
    )
]


def _dp2_record(directory, monkeypatch):
    """`GROUP`, recorded on each rank by `StepRecord`."""
    monkeypatch.setattr(projection, "batch_view", lambda batch, mode, run: batch.view)
    clock = itertools.count()
    for rank, views in enumerate(GROUP):
        record = StepRecord(directory, rank)
        for view in views:
            t = [next(clock)] * 2
            stamps = {"forward_mode": "settled", "t_launch_ns": t, "t_recv_ns": t}
            runner = SimpleNamespace(
                step_clock_ns=lambda: next(clock),
                step_stamps=lambda stamps=stamps: (stamps, lambda _: {"d_busy_ns": 1}),
            )
            rows = view.requests
            batch = SimpleNamespace(
                view=view,
                is_dummy_run=False,
                req_ids=list(range(len(rows))),
                num_scheduled_tokens=np.array([r.query_tokens for r in rows]),
                context_lens=np.array([r.context_tokens for r in rows]),
                is_final_chunk=None,
                external_request_ids=[None] * len(rows),
                produces_output=lambda: True,
            )
            record.add(batch, runner, lambda batch: None)


def test_a_dp2_record_rebuilds_each_step_s_group_counts(tmp_path, monkeypatch):
    _dp2_record(tmp_path, monkeypatch)
    for rank, views in enumerate(GROUP):
        assert [s.view for s in label(tmp_path / f"dp{rank}.jsonl")[0]] == views

    # A new record keeps its counts on a step another rank's record lacks.
    path = tmp_path / "dp1.jsonl"
    full = path.read_text()
    path.write_text(full.splitlines(keepends=True)[0])
    assert [s.view for s in label(tmp_path / "dp0.jsonl")[0]] == GROUP[0]
    path.write_text(full)

    # An older record states neither count: `moe_tokens` is rebuilt from every
    # rank's line at the step, the split is lost.
    for path in tmp_path.glob("dp*.jsonl"):
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        for s in lines:
            del s["moe_tokens"]
            s.pop("tbo_tokens", None)
        path.write_text("".join(json.dumps(s) + "\n" for s in lines))
    for rank, views in enumerate(GROUP):
        got = [s.view for s in label(tmp_path / f"dp{rank}.jsonl")[0]]
        assert got == [dataclasses.replace(v, tbo_tokens=None) for v in views]

    # A step some rank's record lacks has no group count.
    path = tmp_path / "dp1.jsonl"
    path.write_text(path.read_text().splitlines(keepends=True)[0])
    assert [s.view.moe_tokens for s in label(tmp_path / "dp0.jsonl")[0]] == [4, None]


def test_a_prefill_launch_the_device_held_gives_no_W_launch_label(tmp_path):
    """From step 40 on, each middle chunk's launch runs twice its device price,
    so the device runs about half its label before the launch ends: the launch
    waited on the device and gives no `W_launch`. The middle chunks before it,
    the output prefills and the decodes keep theirs, and the middle chunk's
    `W_launch` law is fitted on the launches the host ran alone."""
    steps, held = _steps(14, n=80), list(range(40, 80, 4))
    scale = {}
    for n in held:
        rows = steps[n][0]
        x = (1, sum(r[0] for r in rows), len(rows))
        launch = float(np.dot(HOST["middle"]["W_launch"], x))
        scale[(n, "W_launch")] = 2 * steps[n][2] / launch
    record = _timed_record(tmp_path / "dp0.jsonl", steps, scale=scale)[0]
    samples, table = label(record)

    assert [s.step for s in samples if "W_launch" not in s.host] == held
    assert [row["step"] for row in table if row["W_launch"] is None] == held
    got = fit_host(samples)[0]["laws"]["middle"]["W_launch"]
    assert got == pytest.approx(HOST["middle"]["W_launch"], rel=1e-4, abs=1e-12)
