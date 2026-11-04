# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` on an MI455 card, with no device: a graph the
store lacks is traced on the run's first GPU as that card reads, so its
stepping and compute units pick the kernels, and `trace ops` holds an
`--arch` beside `--gpu` to the card's.
"""

from types import SimpleNamespace

import pytest
from test_capture_mi455_cells import A0, CELL, _trace

from atom.compass import capture
from atom.compass.bench import ops
from atom.compass.capture import driverless
from atom.compass.capture.forward import decode


@pytest.fixture
def started(monkeypatch):
    """Each `trace ops` argv the plan's capture starts, its child stood in."""
    argvs = []
    monkeypatch.setattr(ops, "_child", lambda fn, argv: argvs.append(argv))
    monkeypatch.setattr("atom.compass.capture.forward.device_arch", lambda: "gfx1250")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
    return argvs


def test_a_plan_traces_on_its_first_gpu_as_that_card_reads(started):
    config = SimpleNamespace(model="m", tensor_parallel_size=2)
    ops._capture(config, [decode(4)], "s")

    (argv,) = started
    assert argv[argv.index("--gpu") + 1] == "2"
    assert argv[argv.index("--arch") + 1] == "gfx1250"


def test_coverage_traces_the_arch_its_run_recorded_on_no_card(started):
    config = SimpleNamespace(model="m", tensor_parallel_size=2)
    ops._capture(config, [decode(4)], "s", arch="gfx950")

    (argv,) = started
    assert "--gpu" not in argv and argv[argv.index("--arch") + 1] == "gfx950"


def test_an_arch_beside_a_card_is_the_card_s(tmp_path, monkeypatch):
    monkeypatch.setattr(driverless, "card", {0: A0}.get)
    argv = ["--model", "m", "--arch", "gfx950", "--gpu", "0", "--width", "4"]
    with pytest.raises(ValueError, match="card 0 reads as gfx1250, not gfx950"):
        capture.main([*argv, "--store", str(tmp_path)])

    read = "from atom.compass.capture import driverless\n"
    read += f"driverless.card = {{0: {A0!r}}}.get\n"
    where = ["--arch", "gfx1250", "--gpu", "0"]
    traced = _trace(tmp_path / "t", "--width 4" + CELL, where, read)
    for graph, stated in traced.values():
        assert stated["card"] == A0 and graph["structure"].split()[1] == "gfx1250"
