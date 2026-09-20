# SPDX-License-Identifier: MIT
"""CPU-only cover for the device-free capture mechanism.

Nothing here builds a model or touches a driver. What it pins is the part of
the mechanism that broke silently while it was being built: the `torch.cuda`
stub set, and the post-trace checks that are the difference between a symbolic
trace and a specialised one -- and, the case actually hit, between either of
those and a trace that created no symbol at all.
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

# The stub sets, enumerated rather than counted, so that adding or dropping one
# has to be done here too. Each installer tags its entries with when the stub is
# first needed; both the names and the tag are pinned.
IMPORT_STUBS = (
    "is_available",
    "device_count",
    "_lazy_init",
    "get_rng_state",
    "set_rng_state",
    "get_device_properties",
    "current_device",
    "get_device_capability",
)

MODEL_RUNNER_STUBS = (
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


def test_import_stubs_are_pinned_by_name(restore_cuda):
    stubs = install_device_stubs()
    assert tuple(s["name"] for s in stubs) == IMPORT_STUBS
    assert {s["needed_for"] for s in stubs} == {"import"}, (
        "these have to be in place before ATOM is imported: aiter's Triton "
        "attention kernels read get_device_properties at import time."
    )
    assert torch.cuda.is_available() is True
    assert torch.cuda.device_count() == 1


def test_model_runner_stubs_are_pinned_by_name(restore_cuda):
    """The names `ModelRunner.__init__` needs on top of the import set."""
    stubs = install_runner_stubs()
    assert tuple(s["name"] for s in stubs) == MODEL_RUNNER_STUBS
    assert {s["needed_for"] for s in stubs} == {"model_runner"}


def test_declared_cu_count_is_what_the_stub_reports(restore_cuda):
    install_device_stubs(DeviceReadings(cu_count=304))
    assert torch.cuda.get_device_properties(0).multi_processor_count == 304, (
        "device capability is configured, not queried. A stub that ignores "
        "its argument reads the host instead."
    )


def test_mem_get_info_is_the_declared_reading_not_zero(restore_cuda):
    """One device fact, one declared answer.

    `model_runner.py:1590` sizes the KV budget as
    `gpu_memory_utilization * torch.cuda.mem_get_info()[1]`, so a `(0, 0)` here
    is a confident precise zero, and it contradicts
    `DeviceReadings.total_memory_bytes` one module away.
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
    list; a 0.0 there is fiction with a decimal point on it.
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
    """A symbol that was created and then lost to a constant."""
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    shape_env.replacements[object()] = object()  # what specialisation looks like
    with (
        pytest.raises(CaptureRefusal, match="specialised"),
        capture(shape_env, Recorder()),
    ):
        pass


def _concrete_recorder() -> Recorder:
    """A recorder holding exactly what the committed runs recorded: int shapes."""
    rec = Recorder()
    with rec:
        torch.ones(2, 5120) + torch.ones(2, 5120)
    return rec


def test_capture_refuses_a_trace_that_created_no_symbol():
    """The case the `replacements` check cannot see.

    Before this assertion existed, `shape_env.replacements == {}` on a trace
    with no free symbol read as "clean" and `capture()` returned. It means
    "nothing was checked": all four committed records hold 0 non-numeric shape
    entries out of 12,425-12,520.
    """
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    rec = _concrete_recorder()
    entries, free = shape_entry_census(rec)
    assert entries > 0 and free == 0, "the fixture must be the concrete case"
    assert not shape_env.replacements, "and replacements must be empty, as in the runs"
    with pytest.raises(CaptureRefusal, match="no free symbol"), capture(shape_env, rec):
        pass


def test_capture_records_a_concrete_trace_when_it_is_asked_for():
    """Refusing is not the same as making a concrete trace unreachable.

    `concrete_ok=True` is a different question, asked on purpose and recorded
    on the run, rather than a fallback taken when the first one fails.
    """
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    with capture(shape_env, _concrete_recorder(), concrete_ok=True):
        pass
    assert shape_env.frozen


def test_capture_accepts_a_trace_that_kept_its_free_symbols():
    """The other half: a symbolic trace passes without `concrete_ok`."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        ShapeEnv,
        StatelessSymbolicContext,
    )

    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)
    # The inverse of what makes a trace concrete: build the real tensor OUTSIDE
    # the mode, then convert it with an explicit symbolic context.
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
