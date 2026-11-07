# SPDX-License-Identifier: MIT
"""`compass measure ops` traces a graph its store lacks under two-batch overlap
as one forward, a prefill ATOM would split too: a point times a leaf at its own
shape. The capture runs `trace ops` as the bench starts it, in a process of its
own, on the Qwen3.8-27B at TP1.
"""

import os
import re
import subprocess
import sys
from types import SimpleNamespace

from test_capture_ir import CONFIG, MAIN, TREE

from atom.compass import capture
from atom.compass.bench import ops
from atom.compass.capture.forward import prefill


def test_a_prefill_point_two_batch_overlap_would_split_is_traced_as_one_forward(
    tmp_path, monkeypatch
):
    # ATOM splits a prefill under --enable-tbo from this many tokens on.
    monkeypatch.setenv("ATOM_TBO_PREFILL_MIN_TOKENS", "256")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    done = []

    def child(fn, argv):
        assert fn is capture.main
        command = [sys.executable, "-c", MAIN, "trace", "ops", *argv]
        env = {**os.environ, "PYTHONPATH": str(TREE)}
        run = subprocess.run(
            command, capture_output=True, text=True, timeout=1800, cwd=TREE, env=env
        )
        done.append(run)

    monkeypatch.setattr(ops, "_child", child)
    config = SimpleNamespace(model=str(tmp_path / "model"))

    store = str(tmp_path / "store")
    ops._capture(config, [prefill(512, 2)], store, "gfx942", ["--enable-tbo"])

    (run,) = done
    captured = re.findall(r"^captured (.*) in ", run.stdout, re.MULTILINE)
    # A prefill shape is captured twice: as a middle chunk, then a final one.
    assert captured == ["prefill T=512 B=2"] * 2, run.stderr[-4000:]
