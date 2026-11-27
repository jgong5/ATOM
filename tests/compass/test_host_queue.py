# SPDX-License-Identifier: MIT
"""COMPASS_FIT_HOST_QUEUE (#3720): a host segment priced by the prefill engine's queue, which a record's labels and a
run count the same way: requests handed to the engine less those whose prompt's final chunk ran before."""

import types

import pytest

from atom.compass.backends import step_labels
from atom.compass.backends.step_labels import QueueDepth, queue_columns, queue_depths
from atom.compass.backends.step_linear_fit import HOST_QUEUE_ENV, host_queue_spec


def _timeline():
    """Arrivals (send ends, ns) and forwards (enter ns, request ids, final-chunk flags; None: a dummy forward)."""
    sends = [10, 20, 30, 35, 70, 71, 72, 200]
    forwards = [
        (25, [1], [False]),  # 1's first chunk; 1, 2 in
        (40, [1], [True]),  # 1 done here; 1-4 in
        (50, None, None),  # a dummy forward
        (60, [2], [True]),
        (75, [3], [False]),  # 7 in, 1 and 2 done
        (80, [3], [True]),
        (90, [3], [True]),  # a replayed final chunk: 3 is named once
        (100, [4, 5], [True, False]),
        (210, [5, 6], [True, True]),
    ]
    return sends, forwards


def _lines(forwards):
    """The record's lines as `parity` writes them: a request named once, at its first final chunk."""
    named, lines = set(), []
    for step, (enter, reqs, final) in enumerate(forwards):
        if reqs is None:
            lines.append({"step": step, "is_dummy_run": True, "t_enter_ns": enter, "named": []})
            continue
        new = [[r, f"key{r}"] for r, f in zip(reqs, final) if f and r not in named]
        named.update(r for r, _ in new)
        lines.append({"step": step, "is_dummy_run": False, "t_enter_ns": enter, "named": new})
    return lines


def test_a_records_labels_and_a_run_count_the_same_queue():
    sends, forwards = _timeline()
    labels = queue_depths(sends, _lines(forwards))
    depth, run = QueueDepth(), {}
    for step, (enter, reqs, final) in enumerate(forwards):
        if reqs is None:
            continue
        # The engine's scheduler had taken every request sent before the forward entered.
        run[step] = depth.at(sum(s < enter for s in sends), reqs, final)
    assert run == labels
    assert labels == {0: 2, 1: 4, 3: 3, 4: 5, 5: 5, 6: 4, 7: 4, 8: 4}


def test_the_queue_columns_are_named_with_their_saturation():
    assert queue_columns({"queue_sat4", "queue_sat16", "graphs", "queue_satx"}) == {"queue_sat4": 4, "queue_sat16": 16}
    assert queue_columns(()) == {}


def test_the_switch_names_a_saturation_and_segments(monkeypatch):
    monkeypatch.delenv(HOST_QUEUE_ENV, raising=False)
    assert host_queue_spec() is None
    monkeypatch.setenv(HOST_QUEUE_ENV, "4:middle.E,prefill.E")
    assert host_queue_spec() == {"middle.E": 4, "prefill.E": 4}
    monkeypatch.setenv(HOST_QUEUE_ENV, "4:middle.E;128:middle.W_launch")
    assert host_queue_spec() == {"middle.E": 4, "middle.W_launch": 128}
    monkeypatch.setenv(HOST_QUEUE_ENV, "4:middle.E;8:middle.E")
    with pytest.raises(ValueError, match="twice"):
        host_queue_spec()
    for bad in ("middle.E", "0:middle.E", "4:", "4:middle.E;"):
        monkeypatch.setenv(HOST_QUEUE_ENV, bad)
        with pytest.raises(ValueError, match="Q0:kind.segment"):
            host_queue_spec()


def test_a_run_stamps_each_batch_with_the_arrivals_before_it_was_scheduled():
    from atom.compass.run import _stamp_arrivals

    class Scheduler:
        """ATOM's: `_arrived_requests` counts the arrivals since the last schedule, which restarts it."""

        def __init__(self):
            self._arrived_requests, self.waiting = 0, []

        def add(self, seq):
            self.waiting.append(seq)
            self._arrived_requests += 1

        def extend(self, seqs):
            self.waiting.extend(seqs)
            self._arrived_requests += len(seqs)

        def schedule(self):
            self._arrived_requests = 0
            self.add("late")  # one arrives while it schedules: not this batch's
            return (types.SimpleNamespace(), [])

    engine = types.SimpleNamespace(scheduler=Scheduler())
    _stamp_arrivals(engine)
    engine.scheduler.add("a")
    engine.scheduler.extend(iter(["b", "c"]))
    assert engine.scheduler.schedule()[0].compass_arrived == 3
    assert engine.scheduler.schedule()[0].compass_arrived == 4
    assert engine.scheduler.waiting == ["a", "b", "c", "late", "late"]
    # A stage with no scheduler stamps nothing.
    _stamp_arrivals(types.SimpleNamespace())
    assert step_labels.QUEUE_COLUMN == "queue_sat"


def test_the_head_prices_a_queue_column_and_a_later_stage_takes_none():
    from atom.compass.runner.overrides import HostDeviceRecurrence

    laws = {"middle": {"E": [0.0, 0.0, 0.0], "W_pre": [0.0, 0.0, 0.0], "W_launch": [0.02, 0.0, 0.0]}}
    columns = {"middle": {"E": {"queue_sat4": 0.002}, "W_launch": {"queue_sat4": 0.003}}}
    launch = (types.SimpleNamespace(features=()), None)  # a provider that declares no launch feature
    head = HostDeviceRecurrence(laws, (), lambda: 0.0, (0, 4), columns=columns, launch=launch)
    assert head.queue_columns() == {"queue_sat4": 4}
    assert head.predicted(None) == {}  # no launch column: nothing to predict
    _, seg = head.prepare_end(0.0, "middle", (1, 8192, 1), features={"queue_sat4": 4})
    assert seg["E"] == pytest.approx(0.008) and seg["W_launch"] == pytest.approx(0.032)
    later = HostDeviceRecurrence(laws, (), lambda: 0.0, (1, 4), columns=columns, launch=launch)
    assert later.columns == {} and later.queue_columns() == {}
    # A real launch column still refuses a pipeline stage, as before.
    with pytest.raises(ValueError, match="launch columns"):
        HostDeviceRecurrence(
            laws, (), lambda: 0.0, (0, 4),
            columns={"middle": {"E": {"graphs": 0.001}}},
            launch=(types.SimpleNamespace(features=("graphs",)), None),
        )


def test_a_segment_on_its_mean_prices_its_tail():
    from atom.compass.backends.step_linear_fit import _host_lsq

    ys = [0.020] * 9 + [0.200]  # a robust centre prices 20 ms; the mean is 38 ms
    coef, report = _host_lsq(ys, [])
    assert coef == pytest.approx([0.038, 0.0, 0.0]) and report["lsq"]["samples"] == 10
    # With a queue column: the conditional mean at each queue.
    coef, _ = _host_lsq([0.02, 0.02, 0.05, 0.05], [("queue_sat8", [1, 1, 8, 8])])
    assert coef[0] + coef[3] * 1 == pytest.approx(0.02) and coef[0] + coef[3] * 8 == pytest.approx(0.05)
