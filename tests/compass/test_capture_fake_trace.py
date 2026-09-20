# SPDX-License-Identifier: MIT
"""CPU-only cover for the P0.4 capture mechanism (`16` D98 gate 2).

Nothing here builds a model or touches a driver. What it pins is the part of
the mechanism that broke silently during P0.4: the `torch.cuda` stub set, and
the post-trace assertions that are the difference between a symbolic trace and
a specialised one -- and, the case P0.4 actually hit, between either of those
and a trace that created no symbol at all.
"""

from __future__ import annotations

import pytest
import torch

from atom.compass.capture.fake_trace import (
    CaptureRefusal,
    DeviceReadings,
    Recorder,
    capture,
    install_device_stubs,
    install_runner_stubs,
    shape_entry_census,
)

D18_STUBS = (
    "is_available",
    "device_count",
    "_lazy_init",
    "get_rng_state",
    "set_rng_state",
)

# The names beyond `04` D18's five, enumerated rather than counted. Pinning the
# boundary ("everything else is marked") lets a new stub be added marked and
# unnoticed, and the BEYOND-D18 list is exactly what D18 needs told.
BEYOND_D18_DEVICE_STUBS = (
    "get_device_properties",
    "current_device",
    "get_device_capability",
)

BEYOND_D18_RUNNER_STUBS = (
    "Stream",
    "Event",
    "current_stream",
    "default_stream",
    "set_device",
    "synchronize",
    "empty_cache",
    "reset_peak_memory_stats",
    "memory_stats",
    "mem_get_info",
    "max_memory_allocated",
    "memory_allocated",
    "memory_reserved",
    "stream",
)


@pytest.fixture
def restore_cuda():
    """Both installers mutate `torch.cuda` in place; put it back."""
    saved = {k: getattr(torch.cuda, k, None) for k in dir(torch.cuda)}
    yield
    for k, v in saved.items():
        if v is not None:
            setattr(torch.cuda, k, v)


def test_device_stubs_cover_d18_list_and_mark_the_rest(restore_cuda):
    names = install_device_stubs()
    plain = [n for n in names if "(BEYOND-D18)" not in n]
    assert tuple(plain) == D18_STUBS, (
        "`04` D18 names exactly five stubs; anything else must be marked "
        "BEYOND-D18 so the doc can be told what it is missing."
    )
    assert torch.cuda.is_available() is True
    assert torch.cuda.device_count() == 1


def test_the_beyond_d18_lists_are_pinned_by_name(restore_cuda):
    """Pin the LIST, not only the boundary.

    Marking a new stub BEYOND-D18 makes it invisible to a test that only
    checks that everything unmarked is on D18's five. The names are the
    amendment `04` D18 is owed, so they are the thing to pin.
    """
    device = install_device_stubs()
    runner = install_runner_stubs()
    beyond_device = tuple(
        n.replace("(BEYOND-D18)", "") for n in device if "(BEYOND-D18)" in n
    )
    beyond_runner = tuple(
        n.replace("(BEYOND-D18)", "") for n in runner if "(BEYOND-D18)" in n
    )
    assert beyond_device == BEYOND_D18_DEVICE_STUBS
    assert beyond_runner == BEYOND_D18_RUNNER_STUBS
    assert not [
        n for n in runner if "(BEYOND-D18)" not in n
    ], "every runner stub is beyond D18's list by construction"


def test_declared_cu_count_is_what_the_stub_reports(restore_cuda):
    install_device_stubs(DeviceReadings(cu_count=304))
    assert torch.cuda.get_device_properties(0).multi_processor_count == 304, (
        "principle 2: device capability is configured. A stub that ignores its "
        "argument reads the host instead."
    )


def test_mem_get_info_is_the_declared_reading_not_zero(restore_cuda):
    """One device fact, one declared answer.

    `model_runner.py:1590` sizes the KV budget as
    `gpu_memory_utilization * torch.cuda.mem_get_info()[1]`, so a `(0, 0)` here
    is a confident precise zero -- the README's archetypal failure -- and it
    contradicts `DeviceReadings.total_memory_bytes` one module away.
    """
    readings = DeviceReadings(total_memory_bytes=7 * (1 << 30))
    install_device_stubs(readings)
    install_runner_stubs(readings)
    free, total = torch.cuda.mem_get_info()
    assert total == readings.total_memory_bytes
    assert free == readings.total_memory_bytes
    assert torch.cuda.get_device_properties(0).total_memory == total, (
        "`_Props.total_memory` and `mem_get_info` are the same device fact and "
        "must not be able to disagree."
    )


def test_event_stub_is_a_type_not_a_function(restore_cuda):
    """`model_runner.py:216` evaluates `torch.cuda.Event | None` at import."""
    install_runner_stubs()
    assert isinstance(torch.cuda.Event, type)
    assert torch.cuda.Event | None is not None  # the TypeError this pins


def test_event_refuses_to_report_an_elapsed_time(restore_cuda):
    """A capture runs no kernel, so it has no duration to report.

    `model_runner.py:4144` appends `start.elapsed_time(end)` to a timings
    list; a 0.0 there is fiction with a decimal point on it (principle 6).
    """
    install_runner_stubs()
    ev = torch.cuda.Event()
    with pytest.raises(CaptureRefusal, match="elapsed_time"):
        ev.elapsed_time(ev)


def test_recorder_records_op_and_shapes():
    rec = Recorder()
    with rec:
        torch.ones(3, 4) + torch.ones(3, 4)
    assert rec.ops, "a dispatch mode that records nothing is not recording"
    ops = [o.op for o in rec.ops]
    assert any("add" in o for o in ops), ops
    added = next(o for o in rec.ops if "add" in o.op)
    assert added.in_shapes == [["3", "4"], ["3", "4"]]
    assert added.out_shapes == [["3", "4"]]


def test_capture_refuses_a_specialised_trace():
    """`04` D18 discipline 2, first half: a symbol that was created and lost."""
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    shape_env.replacements[object()] = object()  # what specialisation looks like
    with (
        pytest.raises(CaptureRefusal, match="specialised"),
        capture(shape_env, Recorder()),
    ):
        pass


def _concrete_recorder() -> Recorder:
    """A recorder holding exactly what the P0.4 runs recorded: int shapes."""
    rec = Recorder()
    with rec:
        torch.ones(2, 5120) + torch.ones(2, 5120)
    return rec


def test_capture_refuses_a_trace_that_created_no_symbol():
    """`04` D18 discipline 2, second half, and the detector P0.4 lacked.

    This is probe A of the review's A/B pair. Before this assertion existed,
    `shape_env.replacements == {}` on a trace with no free symbol read as
    "clean" and `capture()` returned. It means "nothing was checked": all four
    P0.4 records hold 0 non-numeric shape entries out of 12,425-12,520.
    """
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    rec = _concrete_recorder()
    entries, free = shape_entry_census(rec)
    assert entries > 0 and free == 0, "the fixture must be the concrete case"
    assert not shape_env.replacements, "and replacements must be empty, as in P0.4"
    with pytest.raises(CaptureRefusal, match="no free symbol"), capture(shape_env, rec):
        pass


def test_capture_records_a_concrete_trace_when_it_is_asked_for():
    """Refusing is not the same as making the T5 fallback unreachable.

    `concrete_ok=True` is a different question, asked on purpose and recorded
    on the run, rather than a fallback taken when the first one fails.
    """
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    with capture(shape_env, _concrete_recorder(), concrete_ok=True):
        pass
    assert shape_env.frozen


def test_capture_accepts_a_trace_that_kept_its_free_symbols():
    """Probe B's other half: a symbolic trace passes without `concrete_ok`."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        ShapeEnv,
        StatelessSymbolicContext,
    )

    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)
    # D18 trap 1 inverted: build the real tensor OUTSIDE the mode, then convert
    # it with an explicit symbolic context.
    real = torch.ones(4, 8)
    sym = fake_mode.from_tensor(
        real,
        static_shapes=False,
        symbolic_context=StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
        ),
    )
    rec = Recorder()
    with fake_mode, capture(shape_env, rec):
        sym + sym
    entries, free = shape_entry_census(rec)
    assert free > 0, f"{free} of {entries} entries carried a symbol"
    assert shape_env.frozen


def test_shape_entry_census_counts_both_halves():
    rec = Recorder()
    with rec:
        torch.ones(3, 4) + torch.ones(3, 4)
    entries, free = shape_entry_census(rec)
    assert entries > 0
    assert free == 0


def test_capture_freezes_the_shape_env_on_a_clean_trace():
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    with capture(shape_env, Recorder(), concrete_ok=True):
        pass
    assert shape_env.frozen, (
        "an unfrozen ShapeEnv keeps accepting guards after the trace, so a "
        "later specialisation would not be visible in it"
    )
