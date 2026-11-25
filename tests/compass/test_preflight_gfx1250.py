# SPDX-License-Identifier: MIT
"""`preflight.sh`'s bookable list on gfx1250 (MI455) cards, which read 100% use
with no process on them, with `rocm-smi` and `rocminfo` planted first on PATH,
so these need no GPU. The census reads this container's load, so they assert
on the output, not the exit."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PREFLIGHT = Path(__file__).resolve().parents[2] / "scripts/compass/preflight.sh"

#: `rocm-smi --showhw` on the MI455 node: four A0 samples on KFD nodes 2 to 5.
SHOWHW = """\
GPU  NODE  DID     GUID   GFX VER  GFX RAS  SDMA RAS  UMC RAS  VBIOS              BUS           PARTITION ID
0    2     0x75c7  25919  gfx1250  ENABLED  ENABLED   ENABLED  113-M4500001-000B  0001:04:00.0  0
1    3     0x75c7  39643  gfx1250  ENABLED  ENABLED   ENABLED  113-M4500001-000B  0002:04:00.0  0
2    4     0x75c7  3974   gfx1250  ENABLED  ENABLED   ENABLED  113-M4500001-000B  0003:04:00.0  0
3    5     0x75c7  9488   gfx1250  ENABLED  ENABLED   ENABLED  113-M4500001-000B  0004:04:00.0  0
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _preflight(tmp_path, vram, queued):
    """Run preflight.sh on four idle-reading gfx1250 cards holding `vram`
    percent, a node agent with a KFD entry on none of them, and one process
    with a queue on each GPU of `queued`, as `rocm-smi --showpidgpus` lists
    them."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    pids = "PID 4763 is using 0 DRM device(s)\n" + "".join(
        f"PID {700 + g} is using 1 DRM device(s):\n{g} \n" for g in queued
    )
    files = {
        "showhw": SHOWHW,
        "use": "".join(f"GPU[{i}]\t\t: GPU use (%): 100\n" for i in range(4)),
        "mem": "".join(
            f"GPU[{i}]\t\t: GPU Memory Allocated (VRAM%): {p}\n"
            for i, p in enumerate(vram)
        ),
        "pids": pids,
    }
    for name, text in files.items():
        (bindir / name).write_text(text)
    (bindir / "rocm-smi").write_text(f"""#!/bin/bash
case "$*" in *--showhw*) cat {bindir}/showhw;; *--showuse*) cat {bindir}/use;;
*--showmemuse*) cat {bindir}/mem;; *--showpids*) cat {bindir}/pids;; esac
""")
    (bindir / "rocminfo").write_text("#!/bin/sh\n")
    for tool in ("rocm-smi", "rocminfo"):
        (bindir / tool).chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    return subprocess.run(
        ["bash", str(PREFLIGHT)], capture_output=True, text=True, check=False, env=env
    ).stdout


@pytest.mark.parametrize(
    "vram, queued, free, hip",
    [
        ((0, 0, 0, 0), (), "0 1 2 3", "0,1,2,3"),
        ((0, 0, 0, 0), (1,), "0 2 3", "0,2,3"),
        ((0, 0, 3, 0), (0,), "1 3", "1,3"),
    ],
)
def test_a_gfx1250_card_is_bookable_with_no_queue_and_no_vram_at_100_use(
    tmp_path, vram, queued, free, hip
):
    out = _preflight(tmp_path, vram, queued)
    assert f"idle and empty: {free}\nexport HIP_VISIBLE_DEVICES={hip}\n" in out


def test_preflight_names_the_gfx1250_cards_and_the_queues_on_them(tmp_path):
    out = _preflight(tmp_path, (0, 0, 0, 0), (1, 3))
    assert "gfx1250 (MI455) GPU(s) 0 1 2 3: " in out
    assert out.count("queues on: 1 3\n") == 1


def test_every_gfx1250_card_with_a_queue_leaves_none_bookable(tmp_path):
    out = _preflight(tmp_path, (0, 0, 0, 0), (0, 1, 2, 3))
    assert "none idle and empty -- every device is busy or holds VRAM.\n" in out
