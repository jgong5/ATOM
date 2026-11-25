# SPDX-License-Identifier: MIT
"""Labelled samples from record directories and step measurements, under one configuration.

A record directory holds `test_parity_served.py`'s record and a run.json
written by hand with the identity entries `record` writes there; its server
served two client runs. Measurements are published through `ArtifactStore`.
"""

import copy
import dataclasses
import json
import re

import pytest
from test_artifact_store import CONDITIONS, stanza
from test_memory_readings import STACK
from test_parity_record import _key, _prompt
from test_parity_served import SAMPLING, client_run, recorded

from atom.compass.artifacts import (
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Topology,
    keyed_platform,
    member_name,
)
from atom.compass.backends.step_labels import digest, label
from atom.compass.fit.publish import platform
from atom.compass.fit.samples import (
    GRADED_ONLY_ENV,
    MEMBER,
    client_spans,
    graded_steps,
    load,
)
from atom.compass.parity import ARRIVALS
from atom.compass.parity.served import served
from atom.compass.run.step_grid import keyed_of

HOST = {"host_cpu": "planted cpu", "atom": "0" * 64, "model": "sha256:" + "1" * 64}
DEVICE = {"index": 7, "name": "MI355X", "arch": "gfx950"}
SOURCE = {
    "name": "atom",
    "root": "/workspace/ATOM/atom",
    "revision": "67c176f0",
    "revision_kind": "tree",
    "method": "git rev-parse HEAD^{tree}, the tree git archive ships",
    "dirty": 0,
}
RUNS = [client_run([0, 1]), client_run([2, 3])]
#: The batch a measurement holds, as a step-record line holds it.
LINE = {"rows": [[1, 300, True], [1, 210, True]], "rung": 2, "produces_output": True}


def record(tmp_path, monkeypatch, name="a", groups=((0, 1), (2, 3)), **run):
    """`tmp_path/name`, a record of `groups` and its run.json, `run` set over it."""
    directory = tmp_path / name
    steps = recorded(directory, monkeypatch, groups, HOST)
    engine = json.loads((steps / "dp0.engine.json").read_text())
    keyed = keyed_of(engine)
    identity = {
        "format_version": 2,
        "argv": ["compass", "record", "--out", str(directory)],
        "source": SOURCE,
        "host_key": {**HOST, "engine": keyed},
        "device": [DEVICE],
        "stack": STACK,
        "ptl": "disabled",
        "profiled": False,
    }
    (directory / "run.json").write_text(json.dumps({**identity, **run}))
    return directory


def idle_rank(steps):
    """`steps/dp1.jsonl`, written as a DP rank no request reached: at each of
    rank 0's steps, the dummy forward it runs to keep step with its group."""
    dummy = {"dp_rank": 1, "is_dummy_run": True, "batch": [], "named": []}
    rank0 = (steps / "dp0.jsonl").read_text().splitlines()
    path = steps / "dp1.jsonl"
    path.write_text(
        "".join(json.dumps({**json.loads(x), **dummy}) + "\n" for x in rank0)
    )
    return path


def edit(path, change):
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def measured(store, config, sampling=SAMPLING, origin="replay"):
    """A step measurement keyed by `config`, as `measure steps` publishes one."""
    key = Key.of(
        Kind.STEP_MEASUREMENT,
        **{f: config[f] for f in ("model", "source_root")},
        **keyed_platform(Kind.STEP_MEASUREMENT, platform(config)),
        engine_config=json.dumps(config["engine_config"], sort_keys=True),
        sampling=json.dumps(sampling, sort_keys=True),
        step_kind="decode",
        view="sha256:5e1f",
    )
    member = {**LINE, "seconds": 1.5e-4, "origin": origin, "traces": ["trace-9"]}
    member["sources"] = ["dp0.jsonl:0a1b2c3d4e5f"]
    ArtifactStore(store).publish(
        key,
        provenance=stanza("compass measure steps"),
        topology=Topology(),
        conditions=CONDITIONS,
        gates=GateState.of(),
        members={member_name(*MEMBER): json.dumps(member).encode()},
    )
    return key


def test_two_records_differing_only_in_kv_cache_dtype_refuse_naming_both_values(
    tmp_path, monkeypatch
):
    a, b = record(tmp_path, monkeypatch, "a"), record(tmp_path, monkeypatch, "b")
    edit(b / "steps/dp0.engine.json", lambda e: e.update(kv_cache_dtype="fp8"))

    with pytest.raises(ValueError) as refused:
        load({a: RUNS, b: RUNS})
    assert str(refused.value) == (
        "the inputs are not one configuration:\n"
        f'engine_config kv_cache_dtype: "bf16" in {a}; "fp8" in {b}'
    )


@pytest.mark.parametrize(
    "field, file, change",
    [
        ("model", "run.json", lambda r: r["host_key"].update(model="sha256:2")),
        ("device", "run.json", lambda r: r["device"][0].update(name="MI300X")),
        (
            "engine_config tensor_parallel_size",
            "steps/dp0.engine.json",
            lambda e: e.update(tensor_parallel_size=2),
        ),
        ("source_root", "run.json", lambda r: r["host_key"].update(atom="1" * 64)),
        ("host_key host_cpu", "run.json", lambda r: r["host_key"].update(host_cpu="x")),
        ("sampling temperature", None, None),
    ],
)
def test_each_field_read_from_the_input_refuses_when_it_differs(
    tmp_path, monkeypatch, field, file, change
):
    a, b = record(tmp_path, monkeypatch, "a"), record(tmp_path, monkeypatch, "b")
    runs = [client_run([0, 1, 2, 3], {**SAMPLING, "temperature": 0.7})]
    if file:
        edit(b / file, change)
        runs = RUNS

    named = rf"(?m)^{re.escape(field)}: .+ in {re.escape(str(a))}; .+ in {re.escape(str(b))}$"
    with pytest.raises(ValueError, match=named):
        load({a: RUNS, b: runs})


@pytest.mark.parametrize(
    "named, change",
    [
        (
            "was profiled",
            lambda d: edit(d / "run.json", lambda r: r.update(profiled=True)),
        ),
        (
            "is format 1",
            lambda d: edit(d / "run.json", lambda r: r.update(format_version=1)),
        ),
        (f"holds {ARRIVALS}", lambda d: (d / "steps" / ARRIVALS).write_text("{}")),
    ],
)
def test_a_record_not_read_as_an_unprofiled_run_s_lines_refuses(
    tmp_path, monkeypatch, named, change
):
    a = record(tmp_path, monkeypatch)
    change(a)

    with pytest.raises(ValueError, match=re.escape(named)):
        load({a: RUNS})


def test_a_record_serving_two_sets_of_request_parameters_refuses(tmp_path, monkeypatch):
    a = record(tmp_path, monkeypatch)
    runs = [RUNS[0], client_run([2, 3], {**SAMPLING, "temperature": 0.7})]

    with pytest.raises(ValueError, match="under 2 sets of request parameters"):
        load({a: runs})


def test_a_lane_s_pin_to_a_data_parallel_rank_is_no_set_of_request_parameters(
    tmp_path, monkeypatch
):
    # A calibration design pins a lane by `data_parallel_rank`, and leaves the
    # others to the router: the record serves one sampling, its samples those
    # of the same record unpinned.
    a = record(tmp_path, monkeypatch)
    pinned = [client_run([0, 1], {**SAMPLING, "data_parallel_rank": 0}), RUNS[1]]

    samples, config = load({a: pinned})

    assert config["sampling"] == SAMPLING
    assert samples == load({a: RUNS})[0]
    other = client_run([2, 3], {**SAMPLING, "temperature": 0.7})
    with pytest.raises(ValueError, match="under 2 sets of request parameters"):
        load({a: [pinned[0], other]})


def test_a_record_run_on_a_dirty_tree_refuses_naming_it(tmp_path, monkeypatch):
    a = record(tmp_path, monkeypatch, source={**SOURCE, "dirty": 1})

    with pytest.raises(ValueError) as refused:
        load({a: RUNS})
    assert str(refused.value) == (
        f"{a} ran with 1 uncommitted files in its source tree; "
        "commit them and run again"
    )


def test_each_sample_copies_its_input_s_provenance_and_the_traces_it_served(
    tmp_path, monkeypatch
):
    a = record(tmp_path, monkeypatch)
    samples, config = load({a: RUNS})
    path = a / "steps/dp0.jsonl"
    split = {s["step"]: s["traces"] for s in served(a / "steps", RUNS)[0]}

    labelled = label(path)[0]
    assert [s.step for s in samples] == [s.step for s in labelled]
    for s, plain in zip(samples, labelled):
        traced = {"traces": split[s.step], "profiled": False}
        assert s.provenance == {**plain.provenance, **traced}
        assert dataclasses.replace(s, provenance=plain.provenance) == plain
    assert {t for s in samples for t in s.provenance["traces"]} == {
        f"trace-{i}" for i in range(4)
    }
    assert samples[0].provenance["run"] == digest(path)
    assert config == {
        "model": HOST["model"],
        "device": "MI355X gfx950",
        "engine_config": json.loads(a.joinpath("run.json").read_text())["host_key"][
            "engine"
        ],
        "prefill_buckets": [],
        "source_root": "atom_py-" + "0" * 64,
        "stack": STACK,
        "ptl": "disabled",
        "sampling": SAMPLING,
        "host_key": {"host_cpu": "planted cpu", "atom": "0" * 64},
    }


def test_a_measurement_loads_beside_records_of_its_configuration_only(
    tmp_path, monkeypatch
):
    a, store = record(tmp_path, monkeypatch), tmp_path / "store"
    config = load({a: RUNS})[1]
    key = measured(store, config, origin="plan")

    samples, both = load({a: RUNS}, [key], store)
    [sample] = [s for s in samples if s.provenance["source"] == "plan"]
    assert both == config
    assert sample.device_s == 1.5e-4 and sample.view.capture_rung == 2
    assert sample.provenance == {
        "source": "plan",
        "run": ArtifactStore(store).read(key).digest,
        "requests": [],
        "config": None,
        "traces": ["trace-9"],
        "inputs": ["dp0.jsonl:0a1b2c3d4e5f"],
        "profiled": False,
    }


@pytest.mark.parametrize(
    "named, field, inner, value",
    [
        ("model", "model", None, "sha256:2"),
        ("device", "device", None, "AMD Instinct MI355X M3550 gfx950"),
        ("source_root", "source_root", None, "tree-0e724ed5"),
        ("engine_config kv_cache_dtype", "engine_config", "kv_cache_dtype", "fp8"),
        ("sampling top_p", "sampling", "top_p", 0.9),
    ],
)
def test_a_measurement_differing_from_the_records_in_one_key_field_refuses(
    tmp_path, monkeypatch, named, field, inner, value
):
    a, store = record(tmp_path, monkeypatch), tmp_path / "store"
    config = load({a: RUNS})[1]
    there = copy.deepcopy(config)
    held = there[field] if inner else there
    old, held[inner or field] = held[inner or field], value
    key = measured(store, there, there["sampling"])

    with pytest.raises(ValueError) as refused:
        load({a: RUNS}, [key], store)
    assert str(refused.value) == (
        "the inputs are not one configuration:\n"
        f"{named}: {json.dumps(old)} in {a}; {json.dumps(value)} in {key}"
    )


def test_samples_come_in_one_order_whatever_order_the_inputs_came_in(
    tmp_path, monkeypatch
):
    a = record(tmp_path, monkeypatch, "a")
    b = record(tmp_path, monkeypatch, "b", groups=((2, 3), (0, 1)))
    store = tmp_path / "store"
    key = measured(store, load({a: RUNS})[1])

    forward = load({a: RUNS, b: RUNS}, [key], store)
    backward = load({b: RUNS[::-1], a: RUNS}, [key], store)
    assert forward == backward
    runs = [s.provenance["run"] for s in forward[0]]
    assert runs == sorted(runs) and len(set(runs)) == 3


def test_a_dp_rank_no_request_reached_adds_no_sample(tmp_path, monkeypatch):
    a = record(tmp_path, monkeypatch)
    alone, config = load({a: RUNS})
    rank1 = idle_rank(a / "steps")

    samples, both = load({a: RUNS})
    run = f"{digest(a / 'steps/dp0.jsonl')}, {digest(rank1)}"
    assert samples == [
        dataclasses.replace(s, provenance={**s.provenance, "run": run}) for s in alone
    ]
    assert both == config


def test_an_empty_rank_file_refuses_as_a_rank_whose_lines_never_reached_disk(
    tmp_path, monkeypatch
):
    a = record(tmp_path, monkeypatch)
    (a / "steps/dp1.jsonl").write_text("")

    with pytest.raises(ValueError, match="dp1.jsonl has no timed steps"):
        load({a: RUNS})


def test_graded_only_drops_a_later_client_run_s_warmup_after_the_window_starts(
    tmp_path, monkeypatch
):
    # Three groups served one after another, the client runs sending the first
    # and the last: the middle group is the second client run's warmup, after
    # the first graded step, which a cut at that step kept.
    a = record(tmp_path, monkeypatch, groups=((0, 1), (2, 3), (4, 5)))
    runs = [client_run([0, 1]), client_run([4, 5])]
    every, config = load({a: runs})
    split = served(a / "steps", runs)
    warmup = {s["step"] for s in split[0] if s["client_run"] is None}
    assert warmup and min(warmup) > min(graded_steps(split))

    monkeypatch.setenv(GRADED_ONLY_ENV, "1")
    graded, graded_config = load({a: runs})
    assert [s.step for s in graded] == [
        s.step for s in every if s.step in graded_steps(split)
    ]
    assert not {s.step for s in graded} & warmup
    assert {s.step for s in every} & warmup
    assert graded_config == config


def test_graded_only_cuts_each_rank_s_dummy_forwards_to_the_graded_steps(
    tmp_path, monkeypatch
):
    from atom.compass.fit import samples as fit_samples

    a = record(tmp_path, monkeypatch, groups=((0, 1), (2, 3), (4, 5)))
    runs = [client_run([0, 1]), client_run([4, 5])]
    labelled = fit_samples.label_run
    steps = tuple((n, None, {}, {}) for n in range(64))

    def planted(*args):
        return [
            ([dataclasses.replace(s, dummy_host=steps) for s in got], table)
            for got, table in labelled(*args)
        ]

    monkeypatch.setattr(fit_samples, "label_run", planted)
    every = load({a: runs})[0]
    assert all(s.dummy_host == steps for s in every)
    monkeypatch.setenv(GRADED_ONLY_ENV, "1")
    graded = load({a: runs})[0]
    kept = graded_steps(served(a / "steps", runs))
    assert graded and kept < {n for n, *_ in steps}
    for s in graded:
        assert [n for n, *_ in s.dummy_host] == sorted(kept)


def test_graded_only_keeps_a_step_holding_warmup_and_sent_requests_together(
    tmp_path, monkeypatch
):
    # Requests 1 and 3 are no client run's, and run beside 0 and 2.
    a = record(tmp_path, monkeypatch)
    runs = [client_run([0]), client_run([2])]
    every, _ = load({a: runs})
    unsent = {_key(_prompt(i)) for i in (1, 3)}
    split = served(a / "steps", runs)
    mixed = {
        s["step"]
        for s in split[0]
        if s["client_run"] is not None and {r[0] for r in s["batch"]} & unsent
    }
    assert mixed & {s.step for s in every}

    monkeypatch.setenv(GRADED_ONLY_ENV, "1")
    graded, _ = load({a: runs})
    assert mixed & {s.step for s in every} <= {s.step for s in graded}
    assert [s.step for s in graded] == [
        s.step for s in every if s.step in graded_steps(split)
    ]


def test_graded_only_keeps_every_step_of_one_client_run_sending_every_request(
    tmp_path, monkeypatch
):
    a = record(tmp_path, monkeypatch)
    runs = [client_run([0, 1, 2, 3])]
    every, _ = load({a: runs})

    monkeypatch.setenv(GRADED_ONLY_ENV, "1")
    assert load({a: runs})[0] == every


def test_the_graded_steps_are_each_client_run_s_span_whatever_its_steps_hold():
    def line(step, client_run, *batch):
        return {"step": step, "client_run": client_run, "batch": [[r] for r in batch]}

    # Client run 0 sends a and c, client run 1 sends b, on rank 1 only; w and v
    # are warmup requests, and x one in flight that run 0's export does not name.
    record = {
        0: [
            line(0, None, "w"),
            line(1, 0, "a", "w"),
            line(2, None, "x"),
            line(3, 0, "c"),
            line(4, None, "v"),
            line(5, None),
            line(6, None),
            line(7, None, "x"),
        ],
        1: [line(n, None) for n in range(6)] + [line(6, 1, "b"), line(7, None)],
    }
    assert graded_steps(record) == {1, 2, 3, 6}


def test_a_ladder_client_run_s_samples_state_the_rungs_it_calibrates(
    tmp_path, monkeypatch
):
    # Three groups served one after another; the second client run, sending the
    # last group, is a calibration ladder of rungs 2 and 4.
    a = record(tmp_path, monkeypatch, groups=((0, 1), (2, 3), (4, 5)))
    runs = [client_run([0, 1]), client_run([4, 5])]
    every, config = load({a: runs})
    first, last = client_spans(served(a / "steps", runs))[1]

    laddered, laddered_config = load({a: runs}, ladders={a: {1: (4, 2)}})
    assert laddered_config == config
    assert [dataclasses.replace(s, ladder=None) for s in laddered] == every
    within = {s.step for s in every if first <= s.step <= last}
    assert within and within < {s.step for s in every}
    assert {s.step for s in laddered if s.ladder == (2, 4)} == within
    assert all(s.ladder is None for s in laddered if s.step not in within)
    assert all(s.ladder is None for s in every)


def test_each_client_run_s_span_runs_from_its_first_step_to_its_last():
    def line(step, client_run):
        return {"step": step, "client_run": client_run, "batch": []}

    record = {
        0: [line(0, None), line(1, 0), line(2, None), line(3, 0), line(4, 1)],
        1: [line(0, None), line(1, None), line(2, None), line(3, None), line(4, 1)],
    }
    assert client_spans(record) == {0: (1, 3), 1: (4, 4)}
