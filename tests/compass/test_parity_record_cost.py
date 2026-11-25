# SPDX-License-Identifier: MIT
"""A simulated server's step record, written by its writer thread off the
step's path: the bytes a record written at each step holds, steps that return
before their lines are written, and a write that fails failing a later step."""

import errno
import itertools
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from test_drive_serve import ARGS, planted  # noqa: F401  the planted fixture
from test_parity_record import _drive

from atom.compass import parity
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.drive import serve
from atom.compass.runner import projection
from atom.model_engine.sequence import Sequence

#: The variable a simulated server is started with.
WRITER = "ATOM_COMPASS_PARITY_WRITER"

pytestmark = pytest.mark.usefixtures("unpriced")


def _recorded(directory, monkeypatch, writer: bool) -> dict:
    """Each file of the record that `_drive` writes for two DP ranks, request
    ids numbered from 0, with and without the writer."""
    monkeypatch.setattr(Sequence, "counter", itertools.count())
    if writer:
        monkeypatch.setenv(WRITER, "1")
    else:
        monkeypatch.delenv(WRITER, raising=False)
    for _, _, runner in _drive(directory, monkeypatch, {0: [0, 1], 1: [2, 3]}).values():
        record = runner._parity_record
        assert (getattr(record, "writer", None) is not None) == writer
        if writer:
            record.writer.close()
    return {p.name: p.read_bytes() for p in sorted(directory.glob("dp*.jsonl"))}


def test_a_record_written_off_the_step_holds_the_bytes_one_written_at_each_step_does(
    tmp_path, monkeypatch
):
    at_each_step = _recorded(tmp_path / "at", monkeypatch, writer=False)
    assert at_each_step["dp0.jsonl"] and at_each_step["dp1.jsonl"]
    assert _recorded(tmp_path / "off", monkeypatch, writer=True) == at_each_step


def test_simulated_steps_return_before_their_lines_are_written(tmp_path, monkeypatch):
    released, dumps = threading.Event(), json.dumps

    def held(*args, **kwargs):
        # Off the recording thread only: the engine file is the step's own.
        if threading.current_thread() is not threading.main_thread():
            released.wait()
        return dumps(*args, **kwargs)

    monkeypatch.setattr(parity.json, "dumps", held)
    monkeypatch.setenv(WRITER, "1")
    try:
        record = _drive(tmp_path, monkeypatch, {0: [0, 1]})[0][2]._parity_record
        # Every step ran while no line could be encoded.
        assert record.step > 1 and record.path.read_bytes() == b""
    finally:
        released.set()
    record.writer.close()
    assert len(record.path.read_text().splitlines()) == record.step


def test_a_line_the_writer_fails_to_write_fails_the_next_step(tmp_path, monkeypatch):
    dumps = json.dumps

    def full(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return dumps(*args, **kwargs)

    view = BatchView((RequestShape(1, 9, True),))
    monkeypatch.setattr(projection, "forward_mode", lambda batch, runner: None)
    monkeypatch.setattr(projection, "batch_view", lambda batch, mode, runner: view)
    monkeypatch.setattr(parity.json, "dumps", full)
    dummy = SimpleNamespace(
        is_dummy_run=True,
        num_scheduled_tokens=np.array([1]),
        context_lens=np.array([9]),
        produces_output=lambda: True,
    )
    record = parity.StepRecord(tmp_path, 0, writer=True)
    record.add(dummy, None)
    record.writer.thread.join(timeout=60)  # it ends at the failed write
    with pytest.raises(OSError, match="quota"):
        record.add(dummy, None)
    with pytest.raises(OSError, match="quota"):
        record.writer.close()


@pytest.mark.usefixtures("planted")  # the planted server, in `tmp_path`
def test_only_a_simulated_server_writes_its_record_off_the_step(tmp_path, monkeypatch):
    started, popen = [], serve.subprocess.Popen

    def start(argv, *args, **kwargs):
        if list(argv[:2]) == list(serve.SERVER):
            started.append(kwargs["env"].get(WRITER))
        return popen(argv, *args, **kwargs)

    monkeypatch.setattr(serve.subprocess, "Popen", start)
    monkeypatch.setenv("PLANTED_EXIT", "late:0")
    monkeypatch.delenv(WRITER, raising=False)
    serve.serve(ARGS, tmp_path / "simulated", run={"bound_s": 1.0})
    serve.serve(ARGS, tmp_path / "real", (6, 7))
    assert started == ["1", None]
