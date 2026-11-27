# SPDX-License-Identifier: MIT
"""#734 cell B (P at DP2): a DP rank's W_pre less its wait for a later peer at the forward's DP collective
(COMPASS_FIT_DP_PEER_CUT), and host segments priced by the engine's queue over every DP rank
(COMPASS_FIT_HOST_QUEUE, a run's COMPASS_HOST_QUEUE), which a record's labels and a run count the same way."""

import json
import types

import pytest
from test_backend_step_labels import _pair

from atom.compass.backends import step_labels
from atom.compass.backends.step_labels import (
    DP_PEER_CUT_ENV,
    QueueDepth,
    dp_peer_cut_on,
    dp_peer_cuts,
    label_run,
    queue_columns,
    queue_depths,
)
from atom.compass.backends.step_linear_fit import HOST_QUEUE_ENV, host_queue_spec


def _line(step, enter, gate, launch, mtp=None, dummy=False):
    s = {"step": step, "is_dummy_run": dummy, "t_enter_ns": enter, "t_gate_ns": list(gate), "t_launch_ns": [launch, launch + 5]}
    if mtp is not None:
        s["t_mtp_ns"] = list(mtp)
    return s


def test_a_rank_s_cut_is_the_later_peer_s_arrival_at_the_collective_up_to_its_after_gate_part():
    zero = [
        _line(0, 0, (10, 20), 100),  # its peer arrives at 50: 30 of its 80 after the gate
        _line(1, 200, (210, 220), 240, mtp=(230, 235)),  # its peer at 400: all 10 before its MTP wait
        _line(2, 500, (510, 520), 600),  # its peer earlier: none
        {"step": 3, "is_dummy_run": False, "t_enter_ns": 700, "t_launch_ns": [720, 730]},  # no gate: none
    ]
    one = [
        _line(0, 5, (30, 50), 120),
        _line(1, 300, (390, 400), 450, dummy=True),  # a dummy is a peer
        _line(2, 450, (460, 470), 590, dummy=True),  # and is cut as a forward is: 50
        _line(3, 650, (660, 690), 760),  # a peer with no gate arrives at its entry: 10
    ]
    assert dp_peer_cuts([zero, one]) == [{0: 30, 1: 10}, {2: 50, 3: 10}]
    assert dp_peer_cuts([one, zero]) == [{2: 50, 3: 10}, {0: 30, 1: 10}]
    assert dp_peer_cuts([zero]) == [{}]


def test_the_cut_switch(monkeypatch):
    monkeypatch.delenv(DP_PEER_CUT_ENV, raising=False)
    assert not dp_peer_cut_on()
    monkeypatch.setenv(DP_PEER_CUT_ENV, "1")
    assert dp_peer_cut_on()
    monkeypatch.setenv(DP_PEER_CUT_ENV, "yes")
    with pytest.raises(ValueError, match="not 0 or 1"):
        dp_peer_cut_on()


def test_label_run_cuts_w_pre_under_the_switch_and_nothing_else(tmp_path, monkeypatch):
    records = _pair(tmp_path, ((3,), (2,)))[0]
    lines = [json.loads(x) for x in records[1].open()]
    for s in lines:  # rank 1 reaches the collective 400 ns later, its own W_pre unchanged
        s["t_gate_ns"] = [g + 400 for g in s["t_gate_ns"]]
    records[1].write_text("".join(json.dumps(x) + "\n" for x in lines))
    monkeypatch.delenv(DP_PEER_CUT_ENV, raising=False)
    off = label_run(records)
    monkeypatch.setenv(DP_PEER_CUT_ENV, "1")
    on = label_run(records)
    zero = [json.loads(x) for x in records[0].open()]
    for (a, _), (b, _), rank in zip(off, on, (0, 1)):
        assert a and len(a) == len(b)
        for x, y in zip(a, b):
            assert {k: v for k, v in x.host.items() if k != "W_pre"} == {k: v for k, v in y.host.items() if k != "W_pre"}
            if rank == 1:
                assert x.host["W_pre"] == y.host["W_pre"]
                continue
            s = zero[x.step]
            after = s["t_launch_ns"][0] - s["t_gate_ns"][1]
            assert x.host["W_pre"] - y.host["W_pre"] == pytest.approx(min(400, after) * 1e-9, abs=1e-15)
    monkeypatch.setenv(DP_PEER_CUT_ENV, "0")
    assert [[s.host for s in x] for x, _ in label_run(records)] == [[s.host for s in x] for x, _ in off]


def _timeline():
    """Arrivals (send ends, ns; request i+1 sent at sends[i]) and each step's forwards on two DP ranks: (enter ns,
    request ids, final-chunk flags), None a dummy forward. A rank runs a dummy only with nothing of its own waiting."""
    sends = [10, 20, 30, 35, 70, 71, 72, 200]
    steps = [
        ((25, [1], [False]), (26, None, None)),  # 1's first chunk on rank 0; 1, 2 in
        ((40, [1], [True]), (41, [3], [False])),  # 1 done here; 1-4 in
        ((50, [2], [True]), (52, [3, 4], [True, False])),
        ((60, None, None), (61, [4], [False])),
        ((75, [5], [True]), (74, [4], [True])),  # 7 in by the earlier entry (74)
        ((90, [7], [False]), (91, [4, 6], [True, True])),  # a replayed final chunk: 4 is named once
        ((100, [7], [True]), (101, None, None)),
        ((150, None, None), (151, None, None)),  # both dummies: no depth
        ((210, None, None), (211, [8], [True])),
    ]
    return sends, steps


def _lines(steps):
    """Each rank's record lines as `parity` writes them: a request named once, at its first final chunk."""
    named, ranks = set(), [[], []]
    for step, pair in enumerate(steps):
        for rank, (enter, reqs, final) in enumerate(pair):
            if reqs is None:
                ranks[rank].append({"step": step, "is_dummy_run": True, "t_enter_ns": enter, "named": []})
                continue
            new = [[r, f"key{r}"] for r, f in zip(reqs, final) if f and r not in named]
            named.update(r for r, _ in new)
            ranks[rank].append({"step": step, "is_dummy_run": False, "t_enter_ns": enter, "named": new})
    return ranks


def test_a_record_s_engine_queue_and_a_run_s_summed_parts_agree():
    sends, steps = _timeline()
    labels = queue_depths(sends, _lines(steps))
    assert labels == {0: 2, 1: 4, 2: 3, 3: 1, 4: 4, 5: 2, 6: 1, 8: 1}
    # A run: each rank's scheduler took the requests the API server handed it before the step entered; a dummy's
    # part is as at its last batch; the ranks' parts summed are the engine's queue.
    owner = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1, 7: 0, 8: 1}
    depth, run = [QueueDepth(), QueueDepth()], {}
    for step, pair in enumerate(steps):
        enter = min(e for e, _, _ in pair)
        parts = []
        for rank, (_, reqs, final) in enumerate(pair):
            taken = sum(1 for i, t in enumerate(sends, 1) if t < enter and owner[i] == rank)
            parts.append(depth[rank].at(None, ()) if reqs is None else depth[rank].at(taken, reqs, final))
        if not all(r is None for _, r, _ in pair):
            run[step] = sum(parts)
    assert run == labels


def test_a_batch_marking_no_final_chunk_names_every_row():
    depth = QueueDepth()
    assert depth.at(3, [1, 2], None) == 3
    assert depth.at(None, ()) == 1


def test_the_queue_columns_are_named_with_their_saturation():
    assert queue_columns({"queue_sat4", "queue_sat16", "eager_body", "queue_satx"}) == {"queue_sat4": 4, "queue_sat16": 16}
    assert queue_columns(()) == {}


def test_the_switch_names_a_saturation_per_segment_group(monkeypatch):
    monkeypatch.delenv(HOST_QUEUE_ENV, raising=False)
    assert host_queue_spec() is None
    monkeypatch.setenv(HOST_QUEUE_ENV, "16:prefill.E,prefill.W_pre")
    assert host_queue_spec() == {"prefill.E": 16, "prefill.W_pre": 16}
    monkeypatch.setenv(HOST_QUEUE_ENV, "16:prefill.E;64:prefill.E_out")
    assert host_queue_spec() == {"prefill.E": 16, "prefill.E_out": 64}
    monkeypatch.setenv(HOST_QUEUE_ENV, "4:prefill.E;8:prefill.E")
    with pytest.raises(ValueError, match="twice"):
        host_queue_spec()
    for bad in ("prefill.E", "0:prefill.E", "4:", "4:prefill.E;"):
        monkeypatch.setenv(HOST_QUEUE_ENV, bad)
        with pytest.raises(ValueError, match="Q0:kind.segment"):
            host_queue_spec()


def test_a_run_stamps_each_batch_with_the_arrivals_before_it_was_scheduled():
    from atom.compass.run import _stamp_arrivals

    class Scheduler:
        def __init__(self):
            self.waiting = []

        def add(self, seq):
            self.waiting.append(seq)

        def extend(self, seqs):
            self.waiting.extend(seqs)

        def schedule(self):
            self.add("late")  # one arrives while it schedules: not this batch's
            return (types.SimpleNamespace(), [])

    engine = types.SimpleNamespace(scheduler=Scheduler())
    _stamp_arrivals(engine)
    engine.scheduler.add("a")
    engine.scheduler.extend(iter(["b", "c"]))
    assert engine.scheduler.schedule()[0].compass_arrived == 3
    assert engine.scheduler.schedule()[0].compass_arrived == 4
    assert engine.scheduler.waiting == ["a", "b", "c", "late", "late"]
    _stamp_arrivals(types.SimpleNamespace())  # an engine with no scheduler stamps nothing


def _host(columns, features=()):
    from atom.compass.runner.overrides import HostDeviceRecurrence

    laws = {"prefill": {"E": [0.001, 0.0, 0.0], "W_pre": [0.0, 0.0, 0.0005], "W_launch": [0.02, 0.0, 0.0], "W_post": [0.0, 0.0, 0.0]}}
    launch = (types.SimpleNamespace(features=features, predict=lambda view, graphs: {"eager_body": 1.0}), None)
    return HostDeviceRecurrence(laws, (), lambda: 0.0, columns=columns, launch=launch)


def test_a_law_prices_its_queue_column_and_predicts_launch_features_only_for_a_launch_column():
    host = _host({"prefill": {"E": {"queue_sat16": 0.002}, "W_pre": {"queue_sat16": 0.0001}}})
    assert host.queue_columns() == {"queue_sat16": 16}
    assert host.predicted(None) == {}  # no launch column: nothing to predict
    _, seg = host.prepare_end(0.0, "prefill", (1, 8192, 4), features={"queue_sat16": 16})
    assert seg["E"] == pytest.approx(0.001 + 0.032) and seg["W_pre"] == pytest.approx(0.002 + 0.0016)
    both = _host({"prefill": {"E": {"queue_sat16": 0.002, "eager_body": 0.003}}}, ("eager_body",))
    assert both.queue_columns() == {"queue_sat16": 16} and both.predicted(None) == {"eager_body": 1.0}
    plain = _host({"prefill": {"E": {"eager_body": 0.003}}}, ("eager_body",))
    assert plain.queue_columns() == {} and plain.predicted(None) == {"eager_body": 1.0}
    with pytest.raises(ValueError, match="does not declare"):
        _host({"prefill": {"E": {"eager_body": 0.003}}})


def test_one_rank_s_engine_queue_is_its_own_part_and_an_unstamped_batch_has_none():
    from atom.compass.runner.overrides import _engine_queue

    host = _host({"prefill": {"E": {"queue_sat16": 0.002}}})
    mode = types.SimpleNamespace(sync=None)
    runner = types.SimpleNamespace()

    def batch(arrived, reqs, final):
        b = types.SimpleNamespace(is_dummy_run=False, total_seqs_num=len(reqs), req_ids=reqs, is_final_chunk=final)
        if arrived is not None:
            b.compass_arrived = arrived
        return b

    assert _engine_queue(runner, batch(5, [1, 2], [True, False]), mode, host) == 5
    assert _engine_queue(runner, batch(6, [2], [True]), mode, host) == 5
    assert _engine_queue(runner, batch(None, [3], [True]), mode, host) is None
    dummy = types.SimpleNamespace(is_dummy_run=True, total_seqs_num=0)
    assert _engine_queue(runner, dummy, mode, host) == 6 - 3  # as at its last batch
    assert _engine_queue(types.SimpleNamespace(), batch(5, [1], [True]), mode, _host({})) is None
    assert step_labels.QUEUE_COLUMN == "queue_sat"


def test_fit_host_gives_each_listed_segment_its_queue_column_and_none_other(tmp_path, monkeypatch):
    from atom.compass.backends.step_linear_fit import fit_host, label_records

    records = _pair(tmp_path, ((3,), (2,)))[0]
    samples = label_records(records, 2)[0]
    run = samples[0].provenance["run"]
    queue = {run: {s.step: s.step % 7 for s in samples}}
    monkeypatch.delenv(HOST_QUEUE_ENV, raising=False)
    off = fit_host(samples)
    assert fit_host(samples, queue=queue) == off  # unset: the queue is not read
    monkeypatch.setenv(HOST_QUEUE_ENV, "4:prefill.W_pre")
    host, report = fit_host(samples, queue=queue)
    assert "queue_sat4" in host["columns"]["prefill"]["W_pre"]
    assert report["prefill.W_pre"]["queue"] == {"column": "queue_sat4", "Q0": 4}
    named = {(k, s) for k, v in host["columns"].items() for s, c in v.items() if "queue_sat4" in c}
    assert named == {("prefill", "W_pre")}
    with pytest.raises(ValueError, match="no queue depth"):
        fit_host(samples, queue={})
