# SPDX-License-Identifier: MIT
"""A capture answers aiter's OPUS MXFP4 MQA-logits registry for its declared
device: the instances aiter's table compiles for the arch, by aiter's own
names for them, each mode's slots one a CU; and refuses an arch it declares
none for, naming those it does.

DeepSeek-V4's attention builder asks for the config at its build since
upstream b7b5ec3f2 (#2809), and aiter reads the registry off the device. The
declarations are process-global, so each check runs in a process of its own.
"""

import os
import pathlib
import subprocess
import sys

TREE = pathlib.Path(__file__).resolve().parents[2]

READ = """
import sys, torch
from atom.compass.capture import driverless
driverless.install(sys.argv[1], sys.argv[2])
try:
    found = driverless.mqa_logits_instances(torch.device("cuda", 0))
except ValueError as error:
    print("refused:", error)
else:
    for i in found:
        print(i.id, i.name, *i.resident)
"""
#: gfx950's instances in the order aiter's registry lists them
#: (`pa_mqa_logits_mxfp4_gfx950.cu`), each by the alias the table names it.
GFX950 = [
    "h64_r1w1s2L",
    "h64_r1w2s4T",
    "h64_r1w4s4T",
    *(
        f"h32_r{rows}w1s{stages}Ln{x}"
        for x in ("", "x4")
        for rows, stages in ((1, 3), (2, 3), (3, 3), (4, 2), (6, 3), (8, 3))
    ),
    "h32_r4w1s3L",
    "h32g_r1w1s2L",
]


def _read(tmp_path, arch) -> list[str]:
    done = subprocess.run(
        [sys.executable, "-c", READ, str(tmp_path), arch],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return [line for line in done.stdout.splitlines() if not line.startswith("[")]


def test_a_gfx950_capture_holds_aiter_s_instances_by_their_names(tmp_path):
    lines = _read(tmp_path, "gfx950")

    ids, names = zip(*(line.split()[:2] for line in lines))
    assert list(names) == GFX950 and list(ids) == [str(i) for i in range(17)]
    # A slot a CU of the declared device's 256 in each mode; a slot-wave
    # instance's four waves are four slots.
    resident = {name: line.split()[2:] for name, line in zip(names, lines)}
    assert resident["h64_r1w1s2L"] == ["256"] * 3
    assert resident["h32_r1w1s3Lnx4"] == ["1024"] * 3


def test_an_arch_with_no_declared_instances_is_refused_naming_those_declared(
    tmp_path,
):
    [line] = _read(tmp_path, "gfx1250")

    assert line == (
        "refused: no declared MXFP4 MQA-logits instances for gfx1250; "
        "declared: ['gfx950']"
    )
