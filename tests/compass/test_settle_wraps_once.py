# SPDX-License-Identifier: MIT
"""A process wraps each `torch.distributed` collective once, however often it
builds a simulated engine.

Each engine construction calls `clock.settle_before_collectives`. Run in one
process, `tests/compass` builds engines in many tests, and a wrap per build
made one host collective settle the clock owner's rounds once per wrap and
pass through a wrapper frame per wrap on its way to gloo.
"""

import threading
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist
from test_dp_member_step_end import collectives  # noqa: F401
from test_dp_slice import _deployment
from test_vertical_slice import _run_file

from atom.compass import run as compass_run
from atom.utils import clock
from atom.utils.distributed import utils


def _owner(monkeypatch):
    """Install a runtime this thread owns; return the settles it counts."""
    settled = []
    stub = SimpleNamespace(
        owner=threading.current_thread(), now=0.0, settle=lambda: settled.append(1)
    )
    monkeypatch.setattr(clock, "_installed", stub)
    return settled


def test_a_second_install_leaves_each_collective_one_settle(
    monkeypatch, collectives  # noqa: F811
):
    clock.settle_before_collectives()
    clock.settle_before_collectives()
    settled = _owner(monkeypatch)

    for name in clock.COLLECTIVES:
        getattr(dist, name)()

    assert len(collectives) == len(clock.COLLECTIVES)
    assert len(settled) == len(clock.COLLECTIVES)


def test_engines_built_twice_in_one_process_wrap_the_gather_once(monkeypatch, tmp_path):
    for name in clock.COLLECTIVES:  # the engine's bootstrap wraps each
        monkeypatch.setattr(dist, name, getattr(dist, name))
    frames = []

    def gather(tensor_list, tensor, group=None, async_op=False):
        frames.append(
            sum(
                f.name == "call" and f.filename.endswith("atom/utils/clock.py")
                for f in traceback.extract_stack()
            )
        )
        tensor_list[0].copy_(tensor)

    monkeypatch.setattr(dist, "all_gather", gather)
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path)))
    monkeypatch.setattr(compass_run, "connect", lambda *_: None)
    monkeypatch.setattr(utils, "LP_OF_RANK", None, raising=False)
    for _ in range(2):  # two engines, as two tests build them in one process
        monkeypatch.setattr(clock, "_installed", None)
        monkeypatch.setattr(compass_run, "_member", None)
        with compass_run.engine(_deployment()):
            pass
    settled = _owner(monkeypatch)
    out = [torch.zeros(2)]

    dist.all_gather(out, torch.ones(2))

    assert frames == [1]
    assert settled == [1]
    assert out[0].tolist() == [1.0, 1.0]
