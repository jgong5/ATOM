# SPDX-License-Identifier: MIT
"""`gate_cpu.sh`'s failure paragraph has to survive being piped.

A caller who runs the gate through `2>&1 | tail -6` is shown five of the
block's nine rendered lines plus the `GATE_CPU_RC=` line. The block is
arranged so the flaky class's name and `scripts/compass/README.md` both land
inside those five, and the comment above the block says outright that nothing
fails if an edit breaks it. This file is what fails.

Measured on the block as it stands, one filler line inserted after rendered
line n: at n of 1-4 the class name survives, at n of 5 -- the line that
carries it -- through 9 it does not. Prepending is safe at every length tried,
up to 1000, because `tail` counts from the end and so moves the text and the
window by the same amount. The two strings survive independently -- appending
one line drops the class name and keeps the path, appending four drops both --
so each is asserted separately.

The block is rendered by bash rather than parsed, so what is asserted is the
text a reader is actually shown. Nothing here needs a driver or an
`import atom`.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "scripts" / "compass" / "gate_cpu.sh"
BASH = shutil.which("bash")

# `tail -6` keeps five lines of the block; the sixth is the GATE_CPU_RC line
# that finish() prints after it.
SURVIVING = 5
CLASS = "TestTheRegionIsNotCopiedPerChunk"
DOC = "scripts/compass/README.md"

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def _rendered():
    """The `RC != 0` block's printf lines, as the lines a reader is shown."""
    body, inside = [], False
    for line in GATE.read_text().splitlines():
        if line == 'if [ "$RC" -ne 0 ]; then':
            inside = True
        elif inside and line.strip() == 'finish "$RC"':
            break
        elif inside and line.strip().startswith("printf "):
            body.append(line)
    return subprocess.run(
        [BASH, "-c", "\n".join(body)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    ).stdout.splitlines()


def test_the_failure_block_is_still_there_to_be_guarded():
    # Without this, a block that was deleted or restructured out of reach
    # would leave the assertions below passing over an empty window.
    lines = _rendered()
    assert len(lines) >= SURVIVING, (
        f"gate_cpu.sh's RC != 0 block rendered {len(lines)} lines, fewer than "
        f"the {SURVIVING} a piping reader sees. It was removed or restructured, "
        "and the guard below is asserting nothing."
    )


def test_a_piping_reader_keeps_both_the_class_name_and_the_file():
    window = "\n".join(_rendered()[-SURVIVING:])
    assert CLASS in window, (
        f"{CLASS} is outside the last {SURVIVING} rendered lines of "
        "gate_cpu.sh's failure block, so `2>&1 | tail -6` sends the reader to "
        "a named file with the flaky test unnamed. Move it back inside the "
        "window: prepending to the block is safe, appending is not."
    )
    assert DOC in window, (
        f"{DOC} is outside the last {SURVIVING} rendered lines, so a piping "
        "reader is told to go and read a README the surviving text does not "
        "name."
    )
