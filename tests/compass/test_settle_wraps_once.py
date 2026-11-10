# SPDX-License-Identifier: MIT
"""A process wraps each `torch.distributed` collective once, however often it
builds a simulated engine, at any DP width.

Each engine construction calls `clock.settle_before_collectives`, and at DP
width above one `shm_collectives.install` before it. Run in one process,
`tests/compass` builds engines in many tests, and a wrap per build made one
host collective settle the clock owner's rounds once per wrap and pass through
a wrapper frame per wrap on its way to gloo. The clock's wrap stays outermost,
so a call settles before the shared-memory route waits on its peers.
"""

import threading
import traceback
from types import SimpleNamespace
from unittest import mock

import pytest
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


def _wraps_on_the_stack():
    """The collective wrappers the current call passed through, outermost
    first: the clock's settling wrap and the shared-memory route."""
    return [
        "clock" if f.name == "call" else "shm"
        for f in traceback.extract_stack()
        if (f.name == "call" and f.filename.endswith("atom/utils/clock.py"))
        or (
            f.name in ("all_gather", "all_reduce")
            and f.filename.endswith("atom/compass/runner/shm_collectives.py")
        )
    ]


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


def test_a_mocked_collective_is_wrapped_like_any_other(
    monkeypatch, collectives  # noqa: F811
):
    monkeypatch.setattr(dist, "barrier", mock.Mock())
    clock.settle_before_collectives()
    settled = _owner(monkeypatch)

    dist.barrier()

    assert settled == [1]


@pytest.mark.parametrize(
    "widths",
    [(1, 1), (2, 2), (2, 2, 2), (1, 2), (2, 1)],
    ids=lambda w: "dp" + "-".join(map(str, w)),
)
def test_engines_built_again_in_one_process_wrap_each_collective_once(
    monkeypatch, tmp_path, widths
):
    for name in clock.COLLECTIVES:  # the engine's bootstrap wraps each
        monkeypatch.setattr(dist, name, getattr(dist, name))
    seen = []

    def gather(tensor_list, tensor, group=None, async_op=False):
        seen.append(_wraps_on_the_stack())
        tensor_list[0].copy_(tensor)

    def reduce(tensor, op=None, group=None, async_op=False):
        seen.append(_wraps_on_the_stack())

    monkeypatch.setattr(dist, "all_gather", gather)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    monkeypatch.delenv(compass_run.NOW_ENV, raising=False)
    monkeypatch.setattr(compass_run, "connect", lambda *_: None)
    monkeypatch.setattr(utils, "LP_OF_RANK", None, raising=False)
    for dp in widths:  # an engine per test that builds one, in one process
        path = _run_file(tmp_path, data_parallel_size=dp)
        monkeypatch.setenv(compass_run.ENV, str(path))
        monkeypatch.setattr(clock, "_installed", None)
        monkeypatch.setattr(compass_run, "_member", None)
        with compass_run.engine(_deployment(dp=dp)):
            pass
    settled = _owner(monkeypatch)
    out = [torch.zeros(2)]

    dist.all_gather(out, torch.ones(2))
    dist.all_reduce(torch.ones(2))

    routed = ["shm"] if 2 in widths else []
    assert seen == [["clock", *routed]] * 2
    assert settled == [1, 1]
    assert out[0].tolist() == [1.0, 1.0]
