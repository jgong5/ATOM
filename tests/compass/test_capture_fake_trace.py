# SPDX-License-Identifier: MIT
"""CPU-only cover for the P0.4 capture mechanism (`16` D98 gate 2).

Nothing here builds a model or touches a driver. What it pins is the part of
the mechanism that broke silently during P0.4: the `torch.cuda` stub set, and
the post-trace assertion that is the difference between a symbolic trace and a
specialised one.
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
)

D18_STUBS = (
    "is_available",
    "device_count",
    "_lazy_init",
    "get_rng_state",
    "set_rng_state",
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


def test_declared_cu_count_is_what_the_stub_reports(restore_cuda):
    install_device_stubs(DeviceReadings(cu_count=304))
    assert torch.cuda.get_device_properties(0).multi_processor_count == 304, (
        "principle 2: device capability is configured. A stub that ignores its "
        "argument reads the host instead."
    )


def test_event_stub_is_a_type_not_a_function(restore_cuda):
    """`model_runner.py:216` evaluates `torch.cuda.Event | None` at import."""
    install_runner_stubs()
    assert isinstance(torch.cuda.Event, type)
    assert torch.cuda.Event | None is not None  # the TypeError this pins


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
    """`04` D18's post-trace assertion, which is the whole of discipline 2."""
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    shape_env.replacements[object()] = object()  # what specialisation looks like
    with (
        pytest.raises(CaptureRefusal, match="specialised"),
        capture(shape_env, Recorder()),
    ):
        pass


def test_capture_freezes_the_shape_env_on_a_clean_trace():
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    with capture(shape_env, Recorder()):
        pass
    assert shape_env.frozen, (
        "an unfrozen ShapeEnv keeps accepting guards after the trace, so a "
        "later specialisation would not be visible in it"
    )
