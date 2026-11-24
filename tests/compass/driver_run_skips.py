# SPDX-License-Identifier: MIT
"""A compass test that skips for want of a driver-run input fails unless
`scripts/compass/gpu_gate_driver_run.txt` lists its file.

`gate_gpu.sh` sets those inputs, and by default runs of `tests/compass` only the
files that list names. A driver-run skip in a file it does not name would run in
no gate: the CPU gate skips it and the GPU gate never collects it. So the CPU
gate, which runs every compass file without the inputs, holds the list complete.

`tests/conftest.py` loads these hooks. They are not a `tests/compass/conftest.py`
because compass tests import `tests/conftest.py`'s helpers as `conftest`, a name
a second conftest would take over.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIST = "scripts/compass/gpu_gate_driver_run.txt"
#: A skip reason naming an input gate_gpu.sh sets, or the router it puts on PATH;
#: `_lib.sh`'s COMPASS_DRIVER_RUN_SKIP matches a SKIPPED line by the same pattern.
DRIVER_RUN = re.compile(
    r"ATOM_COMPASS_(SLICE|IMAGE|MTP|DSPARK|MOE|V4_FLASH|R1|CCTRACES)|COMPASS_(HARNESS|AIPERF)_PYTHON"
    r"|ATOMESH|needs atomesh|atomesh on PATH"
)


LISTED = {
    line.split("#")[0].strip() for line in (ROOT / LIST).read_text().splitlines()
} - {""}


def _unlisted(report, path):
    """Turns a driver-run skip in a compass file LIST does not name into a failure."""
    if not report.skipped or hasattr(report, "wasxfail"):
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else ""
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    if (
        rel.startswith("tests/compass/")
        and DRIVER_RUN.search(reason)
        and rel not in LISTED
    ):
        report.outcome = "failed"
        report.longrepr = (
            f"{rel} skipped for want of a driver-run input, but {LIST} does not "
            f"list it, so no gate would run it; add it there. {reason}"
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    _unlisted(outcome.get_result(), item.path)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    _unlisted(outcome.get_result(), collector.path)
