# SPDX-License-Identifier: MIT
"""The two static checks, each shown firing on a module written to break it.

A check that has never been seen failing is not a check, so the set-iteration
pass here is run against a module that reads sets in eight different ways and
the whole of its report is pinned -- the text a future reader meets is the
thing being delivered, not the exit code.

The same module is the record of what the pass cannot see. Four of its methods
iterate a set through a shape the parser has no way to recognise: a set handed
back by a call with no annotation, one arriving in an unannotated parameter,
one pulled out of a list, and a subclass of `set`. They are asserted **absent**
from the report, in the same place as the ones that are present, because a
limitation that lives only in a docstring is a limitation nobody re-checks.

The clock-source pass is CA-5's and is not rebuilt here. Two things about it
were left open and are settled here instead: whether a real `sleep` belongs
beside the reads, and what happens when two files share the suffix an
allow-list entry matches by.
"""

import os

import pytest

import atom.compass
from atom.compass.detect.clock_source import DEFAULT_ALLOW_LIST, ClockSourceLint
from atom.compass.detect.set_iteration import SetIterationLint

#: The tree the rule applies to.
SIMULATED_PATH = os.path.dirname(atom.compass.__file__)

#: A step loop that reads its participants out of a set eight ways, and four
#: more the parser cannot see. Everything here runs; nothing here raises.
SIMULATED_PATH_READING_SETS = '''# SPDX-License-Identifier: MIT
"""A step loop that decides what to charge by reading a set."""


def ready_for(names):
    return set(names)


class SubclassedSet(set):
    pass


class Step:
    def __init__(self, participants):
        self.ready = set(participants)
        self.done: set[str] = set()
        self.held = [set(participants)]

    def charge(self, now):
        table = []
        for lp_id in self.ready:
            table.append((lp_id, now))
        return table

    def names(self):
        return [lp_id.upper() for lp_id in self.done]

    def first(self):
        head, *rest = self.ready
        return head

    def joined(self):
        return ", ".join(self.done)

    def fresh(self, other):
        for lp_id in self.ready - other:
            yield lp_id

    def viewed(self, table):
        for lp_id in table.keys() & self.ready:
            yield lp_id

    def listed(self):
        return list(self.done)

    def spread(self):
        return [*self.ready]

    def sorted_out(self):
        return sorted(self.ready)

    def holds(self, lp_id):
        return lp_id in self.ready and len(self.ready) > 0

    def counted(self, table):
        return [lp_id for lp_id in table.keys()]

    def returned(self, names):
        return [lp_id for lp_id in ready_for(names)]

    def passed(self, arriving):
        return [lp_id for lp_id in arriving]

    def nested(self):
        return [lp_id for lp_id in self.held[0]]

    def subclassed(self, names):
        return [lp_id for lp_id in SubclassedSet(names)]
'''

#: What the pass says about that module, whole. The line numbers are read off
#: the text above so the two cannot drift apart silently.
_LINES = SIMULATED_PATH_READING_SETS.splitlines()


def line_of(fragment):
    """The line the fragment is on, one-based, refusing an ambiguous one."""
    hits = [index + 1 for index, line in enumerate(_LINES) if fragment in line]
    assert len(hits) == 1, f"{fragment!r} is on {len(hits)} lines, not one"
    return hits[0]


EXPECTED_READS = (
    ("for lp_id in self.ready:", "for-loop over self.ready", "charge"),
    ("lp_id.upper() for lp_id in self.done", "comprehension over self.done", "names"),
    ("head, *rest = self.ready", "unpacking of self.ready", "first"),
    ('", ".join(self.done)', "join() over self.done", "joined"),
    ("self.ready - other", "for-loop over self.ready - other", "fresh"),
    (
        "table.keys() & self.ready",
        "for-loop over table.keys() & self.ready",
        "viewed",
    ),
    ("return list(self.done)", "list() over self.done", "listed"),
    ("[*self.ready]", "unpacking of self.ready", "spread"),
)

#: The four shapes the parser has no way to recognise, by the method they are
#: in. Each is a real iteration of a real set and none of them is reported.
BEYOND_THE_PARSER = ("returned", "passed", "nested", "subclassed")


@pytest.fixture(scope="module")
def reading_sets(tmp_path_factory):
    """The module above, written out and scanned."""
    module = tmp_path_factory.mktemp("sets") / "step.py"
    module.write_text(SIMULATED_PATH_READING_SETS, encoding="utf-8")
    return str(module), SetIterationLint().check(str(module))


class TestTheSetIterationLint:
    """The static half: a set read in order on the simulated path fails CI."""

    def test_the_simulated_path_is_clean_today(self):
        """This assertion is the gate. It is what breaks on the day one lands."""
        lint = SetIterationLint()
        code, report = lint.check(SIMULATED_PATH)
        assert code == 0
        assert report == (
            f"set-iteration lint: clean over {len(lint.modules(SIMULATED_PATH))} "
            "module(s)"
        )

    def test_it_fires_on_an_injected_read_and_names_every_one(self, reading_sets):
        module, (code, report) = reading_sets
        assert code == 1
        assert report == "\n".join(
            [
                "set-iteration lint: 8 ordered read(s) of a set:",
                *(
                    f"  {module}:{line_of(fragment)}  {form}  in {scope}"
                    for fragment, form, scope in EXPECTED_READS
                ),
                (
                    "A set hands its members back in hash order, and a string is "
                    "hashed from a seed the interpreter picks per process, so two "
                    "runs of one configuration read the same set in two orders and "
                    "diverge with nothing to point at. Sort at the point of "
                    "iteration, or hold the members in a dict whose values are "
                    "None and iterate that."
                ),
            ]
        )

    @pytest.mark.parametrize("scope", BEYOND_THE_PARSER)
    def test_what_it_cannot_see_is_recorded_rather_than_claimed(
        self, reading_sets, scope
    ):
        """Each of these iterates a set, and the report does not mention it."""
        _module, (_code, report) = reading_sets
        assert f"in {scope}" not in report

    @pytest.mark.parametrize(
        "source",
        [
            "def f(s):\n    return sorted(s | set())\n",
            "def f(s, x):\n    return x in set(s)\n",
            "def f(s):\n    return len(set(s))\n",
            "def f(d):\n    return [k for k in d.keys()]\n",
            "def f(d):\n    return [k for k in d]\n",
            "def f(s):\n    t = set(s)\n    t.add(1)\n    return t\n",
        ],
        ids=["sorted", "membership", "len", "dict-view", "dict", "mutation"],
    )
    def test_reading_a_set_without_reading_its_order_is_not_reported(self, source):
        assert SetIterationLint().scan_source(source, "engine.py") == ()

    def test_an_annotated_parameter_is_enough_to_recognise_one(self):
        source = "def charge(ready: set[str]):\n    return [x for x in ready]\n"
        reads = SetIterationLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == [
            "engine.py:2  comprehension over ready  in charge"
        ]

    def test_a_set_bound_through_another_name_is_still_one(self):
        """The binding can come after the read; the pass runs to a fixed point."""
        source = (
            "def charge():\n    for x in later:\n        yield x\n\n\nlater = set()\n"
        )
        reads = SetIterationLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == [
            "engine.py:2  for-loop over later  in charge"
        ]

    def test_summing_a_set_is_reported_because_float_addition_reorders(self):
        source = "def total(costs: set[float]):\n    return sum(costs)\n"
        reads = SetIterationLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == [
            "engine.py:2  sum() over costs  in total"
        ]

    def test_turning_the_lint_off_makes_the_tree_claim_nothing(self):
        off = SetIterationLint(enabled=False)
        assert off.scan_source(SIMULATED_PATH_READING_SETS, "step.py") == ()
        assert off.report((), 1) == (
            "set-iteration lint: disabled, so this tree makes no claim about "
            "its set order"
        )


class TestARealSleepBesideTheReads:
    """Why `time.sleep` was added to the calls the clock-source pass refuses."""

    def test_a_sleep_is_refused_the_same_as_a_read(self):
        source = "import time\n\n\ndef step():\n    time.sleep(0.001)\n"
        reads = ClockSourceLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == ["engine.py:5  time.sleep  in step"]

    def test_the_renamed_import_is_found_too(self):
        source = "from time import sleep as pause\n\n\ndef step():\n    pause(0.001)\n"
        reads = ClockSourceLint().scan_source(source, "engine.py")
        assert [str(read) for read in reads] == ["engine.py:5  time.sleep  in step"]

    def test_adding_it_left_the_simulated_path_clean(self):
        assert ClockSourceLint().check(SIMULATED_PATH)[0] == 0


class TestTheAllowListMatchesBySuffix:
    """What a suffix match costs, and the guard that makes the cost visible."""

    def test_each_entry_names_exactly_one_module_it_excuses(self):
        """The guard. A suffix that matches two files excuses both of them.

        The matcher is a suffix comparison, so an entry naming a short enough
        tail of a path excuses every file ending in it, and an entry naming a
        file that has been moved or deleted excuses nothing while still
        reading as a recorded decision. Both are the same question -- how many
        modules does this entry stand for -- and one is the answer.
        """
        lint = ClockSourceLint()
        modules = lint.modules(SIMULATED_PATH)
        for entry in DEFAULT_ALLOW_LIST:
            matched = [
                path for path in modules if path.replace(os.sep, "/").endswith(entry)
            ]
            assert len(matched) == 1, f"{entry} matches {len(matched)} module(s)"

    def test_two_files_sharing_a_suffix_are_both_excused(self, tmp_path):
        """The gap itself, written down rather than assumed harmless.

        Nothing in the matcher distinguishes them, and nothing in the report
        says a second file was covered. It is reachable from an entry short
        enough to name a tail two files share, which is what the guard above
        is looking for.
        """
        lint = ClockSourceLint(allow_list={"detect/watchdog.py": "the sampler"})
        assert lint.allowed("/one/atom/compass/detect/watchdog.py")
        assert lint.allowed("/another/vendored/detect/watchdog.py")
