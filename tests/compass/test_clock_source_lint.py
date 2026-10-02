# SPDX-License-Identifier: MIT
"""The clock-source lint: clean over `atom/compass`, and fired by a seeded read."""

import os

import pytest

import atom.compass
import atom.utils.clock
from atom.compass.detect.clock_source import DEFAULT_ALLOW_LIST, ClockSourceLint

#: A cost backend that charges the time its own arithmetic took.
SIMULATED_PATH_WITH_A_REAL_READ = '''# SPDX-License-Identifier: MIT
"""A cost backend that charges a decode step, and its own arithmetic with it."""

import time

DECODE_STEP_SECONDS = 0.054


def step_seconds(rows):
    """What the model says a decode step of this many rows costs."""
    started = time.monotonic()
    total = DECODE_STEP_SECONDS * rows
    return total + (time.monotonic() - started)
'''

#: The tree the lint is run over: the package that was actually imported,
#: not a path relative to wherever pytest happened to be started.
SIMULATED_PATH = os.path.dirname(atom.compass.__file__)

_SOURCE_LINES = SIMULATED_PATH_WITH_A_REAL_READ.splitlines()
FIRST_READ_LINE = _SOURCE_LINES.index("    started = time.monotonic()") + 1
SECOND_READ_LINE = (
    _SOURCE_LINES.index("    return total + (time.monotonic() - started)") + 1
)

#: Spellings of a real-clock read that the pass has to find, each beside the
#: report line it must produce. The three `datetime` forms all resolve to one
#: name, which is why the list entry is written out in full rather than as the
#: tail a caller happens to type.
CLOCK_READS_THAT_FIRE = {
    "import": (
        "import time\n\n\ndef f():\n    return time.monotonic()\n",
        "engine.py:5  time.monotonic  in f",
    ),
    "from-import": (
        "from time import monotonic\n\n\ndef f():\n    return monotonic()\n",
        "engine.py:5  time.monotonic  in f",
    ),
    "import-renamed": (
        "from time import monotonic as clock\n\n\ndef f():\n    return clock()\n",
        "engine.py:5  time.monotonic  in f",
    ),
    "assigned-alias": (
        "import time\n\nnow = time.monotonic\n\n\ndef f():\n    return now()\n",
        "engine.py:7  time.monotonic  in f",
    ),
    "alias-of-alias": (
        "import time as t\n\nnow = t.monotonic\n\n\ndef f():\n    return now()\n",
        "engine.py:7  time.monotonic  in f",
    ),
    "nanoseconds": (
        "import time\n\n\ndef f():\n    return time.perf_counter_ns()\n",
        "engine.py:5  time.perf_counter_ns  in f",
    ),
    "clock-gettime": (
        "import time\n\n\ndef f():\n    return time.clock_gettime(0)\n",
        "engine.py:5  time.clock_gettime  in f",
    ),
    "clock-gettime-ns": (
        "import time\n\n\ndef f():\n    return time.clock_gettime_ns(0)\n",
        "engine.py:5  time.clock_gettime_ns  in f",
    ),
    "datetime-module": (
        "import datetime\n\n\ndef f():\n    return datetime.datetime.now()\n",
        "engine.py:5  datetime.datetime.now  in f",
    ),
    "datetime-class": (
        "from datetime import datetime\n\n\ndef f():\n    return datetime.now()\n",
        "engine.py:5  datetime.datetime.now  in f",
    ),
    "datetime-alias": (
        "import datetime as dt\n\n\ndef f():\n    return dt.datetime.now()\n",
        "engine.py:5  datetime.datetime.now  in f",
    ),
    "utcnow": (
        "from datetime import datetime\n\n\ndef f():\n    return datetime.utcnow()\n",
        "engine.py:5  datetime.datetime.utcnow  in f",
    ),
    "innermost-scope": (
        "import time\n\n\nclass A:\n    def g(self):\n        return time.time()\n",
        "engine.py:6  time.time  in g",
    ),
}

#: Calls whose name ends in a listed one and which are not it. Each is a
#: receiver the pass has no reason to think is the standard library.
CLOCK_READS_THAT_ARE_NOT = {
    "self-attribute": "class C:\n    def g(self):\n        return self.time.monotonic()\n",
    "row-attribute": "def f(row):\n    return row.datetime.now()\n",
    "imported-namespace": "import mock\n\n\ndef f():\n    return mock.time.time()\n",
    "another-package": "from mypkg import time\n\n\ndef f():\n    return time.monotonic()\n",
}


class TestTheClockSourceLint:
    """A real clock read on the simulated path fails CI."""

    def test_the_simulated_path_is_clean_today(self):
        """This assertion is the gate. It is what breaks on the day a read lands."""
        lint = ClockSourceLint()
        code, report = lint.check(SIMULATED_PATH)
        assert code == 0, report
        assert report == (
            f"clock-source lint: clean over {len(lint.modules(SIMULATED_PATH))} "
            f"module(s), {len(DEFAULT_ALLOW_LIST)} file(s) allow-listed"
        )

    def test_the_lp_runtime_is_clean_today(self):
        """The LP runtime sits outside `atom/compass` and is on the simulated path."""
        assert ClockSourceLint().check(atom.utils.clock.__file__) == (
            0,
            "clock-source lint: clean over 1 module(s), 0 file(s) allow-listed",
        )

    def test_the_lint_fires_on_a_seeded_read_and_names_every_one(self, tmp_path):
        """The seeded module sits one directory down, so the walk has to recurse."""
        (tmp_path / "backends").mkdir()
        module = tmp_path / "backends" / "step.py"
        module.write_text(SIMULATED_PATH_WITH_A_REAL_READ, encoding="utf-8")
        code, report = ClockSourceLint().check(str(tmp_path))
        assert code == 1
        assert report == "\n".join(
            [
                "clock-source lint: 2 real-clock read(s) on the simulated path:",
                f"  {module}:{FIRST_READ_LINE}  time.monotonic  in step_seconds",
                f"  {module}:{SECOND_READ_LINE}  time.monotonic  in step_seconds",
                (
                    "A run charges simulated seconds for the work it models. A "
                    "real read puts machine seconds into the same record, and the "
                    "mixture is reported as one modelled number. Take the time "
                    "from the run's clock, or add the file to the allow-list with "
                    "the reason it stays real."
                ),
            ]
        )

    def test_a_root_holding_no_module_is_refused(self, tmp_path):
        """A docs directory or a mistyped root would otherwise report clean."""
        (tmp_path / "README.md").write_text("# docs\n")
        root = str(tmp_path)
        assert ClockSourceLint().check(root) == (
            1,
            f"clock-source lint: no module under {root}, nothing checked",
        )

    def test_an_asyncio_timer_is_not_a_read(self):
        """A simulated run's event loop keeps virtual time, so its timers do too."""
        source = "import asyncio\n\n\nasync def f():\n    await asyncio.sleep(1)\n"
        assert ClockSourceLint().scan_source(source, "engine.py") == ()

    def test_the_clean_line_counts_what_this_scan_skipped(self, tmp_path):
        """A tree holding none of the listed files is a tree with no exemptions.

        The root is the module itself, which is a scan of one file.
        """
        module = tmp_path / "step.py"
        module.write_text("x = 1\n", encoding="utf-8")
        lint = ClockSourceLint(allow_list={"detect/sampler.py": "the sampler"})
        code, report = lint.check(str(module))
        assert code == 0
        assert report == (
            "clock-source lint: clean over 1 module(s), 0 file(s) allow-listed"
        )


class TestHowFarOneAllowListEntryReaches:
    """What a path entry stands for, and the guard that keeps it to one file."""

    def test_each_entry_names_exactly_one_module_it_excuses(self):
        """The guard. An entry matching two files excuses both of them.

        The matcher compares a tail of the path, so an entry naming a short
        enough tail excuses every file ending in it, and an entry naming a file
        that has been moved or deleted excuses nothing while still reading as a
        recorded decision.
        """
        modules = ClockSourceLint.modules(SIMULATED_PATH)
        for entry, reason in DEFAULT_ALLOW_LIST.items():
            alone = ClockSourceLint(allow_list={entry: reason})
            matched = [path for path in modules if alone.allowed(path) is not None]
            assert len(matched) == 1, f"{entry} matches {len(matched)} module(s)"

    def test_two_files_sharing_a_whole_tail_are_both_excused(self, tmp_path):
        """The gap the guard above looks for: multiplicity, not anchoring."""
        lint = ClockSourceLint(allow_list={"detect/sampler.py": "the sampler"})
        assert lint.allowed("/one/atom/compass/detect/sampler.py")
        assert lint.allowed("/another/vendored/detect/sampler.py")

    def test_a_directory_merely_ending_in_the_entry_is_not_excused(self):
        """Counting how many modules an entry stands for needs a segment boundary."""
        lint = ClockSourceLint(allow_list={"detect/sampler.py": "the sampler"})
        assert lint.allowed("detect/sampler.py")
        assert lint.allowed("/one/atom/compass/NOTdetect/sampler.py") is None


class TestWhatCountsAsAClockRead:
    """Which call is the machine's clock, and which one merely reads like it."""

    @pytest.mark.parametrize(
        "source, expected",
        CLOCK_READS_THAT_FIRE.values(),
        ids=CLOCK_READS_THAT_FIRE.keys(),
    )
    def test_every_spelling_of_a_real_read_is_found(self, source, expected):
        reads = ClockSourceLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == [expected]

    @pytest.mark.parametrize(
        "source", CLOCK_READS_THAT_ARE_NOT.values(), ids=CLOCK_READS_THAT_ARE_NOT.keys()
    )
    def test_a_name_merely_ending_in_a_listed_one_is_not_a_read(self, source):
        """A false positive here is a red gate on code that is doing nothing wrong."""
        assert ClockSourceLint().scan_source(source, "engine.py") == ()
